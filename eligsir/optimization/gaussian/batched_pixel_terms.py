"""Batched RGB/depth supervision for true-camera groups.

Depth remains normalised per view, so stacking cannot change each view's
weight in the objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from eligsir.optimization.gaussian.depth_loss_weighting import (
    DEPTH_LOSS_QUANTITIES,
    combine_depth_loss_weights,
    masked_depth_error,
)
from eligsir.utils.loss_utils import ssim_per_view


@dataclass(frozen=True)
class PixelTermsView:
    """One view's graph-attached RGB/depth terms."""

    Ll1: Optional[torch.Tensor]  # None means not measured, never a zero residual
    ssim: Optional[torch.Tensor]
    depth: Optional[torch.Tensor]  # scalar, graph attached, masked + per-view normalised
    depth_valid_count: Optional[torch.Tensor]  # 0-d device tensor (no .item() here)
    photometric_evwb_state: Optional[torch.Tensor] = None
    photometric_evwb_prior: Optional[torch.Tensor] = None


def assemble_batched_pixel_terms(
    *,
    renders: torch.Tensor,
    gts: torch.Tensor,
    rendered_depth: Optional[torch.Tensor] = None,
    sensor_depth: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    alpha_gate: float = 0.0,
    depth_quantity: str = "d",
    depth_loss_type: str = "l1",
    weight_maps: Optional[Sequence[Optional[torch.Tensor]]] = None,
    required_rgb: tuple[bool, bool] = (True, True),
) -> tuple[PixelTermsView, ...]:
    """Return graph-attached per-view terms from equal-shaped batch tensors."""

    if renders.dim() != 4 or renders.shape != gts.shape:
        raise ValueError("renders/gts must be equal [B,3,H,W] stacks")
    b = int(renders.shape[0])

    Ll1 = (renders - gts).abs().mean(dim=(1, 2, 3)) if required_rgb[0] else None
    ssim_values = ssim_per_view(renders, gts) if required_rgb[1] else None

    depth_per_view: Optional[torch.Tensor] = None
    depth_valid_count_per_view: Optional[torch.Tensor] = None
    if rendered_depth is not None and sensor_depth is not None:
        if depth_quantity not in DEPTH_LOSS_QUANTITIES:
            raise ValueError(
                f"Unrecognized depth_quantity={depth_quantity!r}; "
                f"expected one of {DEPTH_LOSS_QUANTITIES!r}."
            )
        if not math.isfinite(alpha_gate) or not (0.0 <= alpha_gate < 1.0):
            raise ValueError(
                f"Invalid alpha_gate={alpha_gate!r}; expected a finite value in [0.0, 1.0)."
            )
        if rendered_depth.shape != sensor_depth.shape:
            raise ValueError("rendered_depth/sensor_depth must share shape")

        alpha_weighted = depth_quantity == "ed_alpha_weighted"
        valid = (sensor_depth > 0) & (rendered_depth.detach() > 0)
        _alpha_detached: Optional[torch.Tensor] = None
        if alpha_gate > 0.0 or alpha_weighted:
            if alpha is None:
                raise RuntimeError(
                    (
                        "alpha_gate > 0.0"
                        if alpha_gate > 0.0
                        else "depth_quantity='ed_alpha_weighted'"
                    )
                    + " requires an alpha stack -- refusing to silently apply no "
                    "gate/weight at all, since that would silently corrupt an "
                    "experiment comparison."
                )
            if alpha.shape != rendered_depth.shape:
                raise ValueError("alpha must share shape with rendered_depth")
            _alpha_detached = alpha.detach()
            if alpha_gate > 0.0:
                valid = valid & (_alpha_detached > alpha_gate)

        depth_err = masked_depth_error(
            rendered_depth, sensor_depth, valid, loss_type=depth_loss_type
        )
        # Issue #1833: multiply the per-pixel alpha (detached, no gradient)
        # into the weight stack so weighted aggregation below weights each
        # valid pixel's residual by its rendered coverage, normalised by the
        # sum of weights -- see objective.py's non-batched path for the
        # matching single-view implementation and rationale.
        _alpha_weight = _alpha_detached if alpha_weighted else None
        combined_weights = combine_depth_loss_weights(*(weight_maps or ()), _alpha_weight)
        masked_err = torch.where(valid, depth_err, torch.zeros_like(depth_err))
        if combined_weights is None:
            depth_per_view = masked_err.sum(dim=(1, 2, 3)) / valid.sum(
                dim=(1, 2, 3)
            ).to(depth_err.dtype).clamp_min(1.0)
        else:
            weights = torch.broadcast_to(combined_weights, depth_err.shape)
            masked_weights = torch.where(
                valid, weights.clamp_min(1e-4), torch.zeros_like(depth_err)
            )
            depth_per_view = (masked_err * masked_weights).sum(dim=(1, 2, 3)) / (
                masked_weights.sum(dim=(1, 2, 3)).clamp_min(1e-4)
            )
        depth_valid_count_per_view = valid.sum(dim=(1, 2, 3))

    return tuple(
        PixelTermsView(
            Ll1=None if Ll1 is None else Ll1[i],
            ssim=None if ssim_values is None else ssim_values[i],
            depth=None if depth_per_view is None else depth_per_view[i],
            depth_valid_count=(
                None
                if depth_valid_count_per_view is None
                else depth_valid_count_per_view[i]
            ),
        )
        for i in range(b)
    )


__all__ = ["PixelTermsView", "assemble_batched_pixel_terms"]
