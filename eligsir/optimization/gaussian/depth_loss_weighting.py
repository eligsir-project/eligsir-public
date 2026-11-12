"""Depth-supervision weighting and zero-safe reductions."""

from __future__ import annotations

from typing import Optional

import torch

#: Selectable --streaming_depth_loss_quantity values. "d" (default) and "ed"
#: predate issue #1833. "ed_alpha_weighted" (#1833) also reads the alpha-
#: normalised ED but additionally weights each valid pixel's residual by its
#: rendered alpha (detached) instead of a hard alpha_gate cutoff -- see
#: arguments/__init__.py's streaming_depth_loss_quantity docstring.
DEPTH_LOSS_QUANTITIES = ("d", "ed", "ed_alpha_weighted")


def combine_depth_loss_weights(*weight_maps: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Multiply configured broadcastable weight maps; return ``None`` if absent."""
    result: Optional[torch.Tensor] = None
    for weights in weight_maps:
        if weights is not None:
            result = weights if result is None else result * weights
    return result


def masked_depth_error(
    rendered: torch.Tensor,
    sensor: torch.Tensor,
    valid: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    """Return dense per-pixel error with invalid rays detached from the graph.

    Building Huber/L1 only from boolean-indexed valid pixels requires a host
    cardinality gate to avoid an empty reduction. Replacing invalid inputs with
    finite zeros first keeps the dense path zero-safe and, importantly, keeps a
    NaN at an invalid rendered ray from leaking through Huber backward despite
    a later mask.
    """
    if rendered.shape != sensor.shape or valid.shape != rendered.shape:
        raise ValueError("rendered, sensor and valid depth tensors must share shape")
    safe_rendered = torch.where(valid, rendered, torch.zeros_like(rendered))
    safe_sensor = torch.where(valid, sensor, torch.zeros_like(sensor))
    if loss_type == "huber":
        return torch.nn.functional.huber_loss(
            safe_rendered,
            safe_sensor,
            reduction="none",
            delta=0.1,
        )
    if loss_type == "l1":
        return (safe_rendered - safe_sensor).abs()
    raise ValueError(f"unsupported depth loss type {loss_type!r}")


def weighted_depth_error_mean(
    depth_err: torch.Tensor,
    weights: Optional[torch.Tensor],
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the weighted mean; equal-shaped masks are zero-safe on device."""
    if valid is not None and valid.shape == depth_err.shape:
        masked_err = torch.where(valid, depth_err, torch.zeros_like(depth_err))
        if weights is None:
            return masked_err.sum() / valid.sum().to(depth_err.dtype).clamp_min(1.0)
        try:
            weights = torch.broadcast_to(weights, depth_err.shape)
        except RuntimeError as exc:
            raise ValueError("depth weights must broadcast to the depth-error shape") from exc
        masked_weights = torch.where(
            valid, weights.clamp_min(1e-4), torch.zeros_like(depth_err)
        )
        return (masked_err * masked_weights).sum() / masked_weights.sum().clamp_min(1e-4)

    if weights is None:
        return depth_err.mean()
    selected = weights[valid] if valid is not None else weights
    selected = selected.clamp_min(1e-4)
    return (depth_err * selected).sum() / selected.sum().clamp_min(1e-4)


__all__ = [
    "DEPTH_LOSS_QUANTITIES",
    "combine_depth_loss_weights",
    "masked_depth_error",
    "weighted_depth_error_mean",
]
