"""Soft surface-manifold losses for geometry-aware streaming."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geometry.priors import SurfaceState


@dataclass
class SurfaceLossResult:
    total: torch.Tensor
    normal_drift: torch.Tensor
    tangent_drift: torch.Tensor
    normal_align: torch.Tensor
    thinness: torch.Tensor

    def to_tensorboard_dict(self, prefix: str = "surface_loss") -> dict[str, float]:
        return {
            f"{prefix}/total": float(self.total.detach()),
            f"{prefix}/normal_drift": float(self.normal_drift.detach()),
            f"{prefix}/tangent_drift": float(self.tangent_drift.detach()),
            f"{prefix}/normal_align": float(self.normal_align.detach()),
            f"{prefix}/thinness": float(self.thinness.detach()),
        }


def compute_surface_anchor_losses(
    gaussians,
    *,
    min_confidence: float = 0.1,
    normal_weight: float = 0.0,
    tangent_weight: float = 0.0,
    normal_align_weight: float = 0.0,
    thinness_weight: float = 0.0,
) -> SurfaceLossResult:
    device = gaussians.get_xyz.device
    zero = torch.zeros((), device=device)
    if not hasattr(gaussians, "anchor_xyz") or not hasattr(gaussians, "anchor_normal"):
        return SurfaceLossResult(zero, zero, zero, zero, zero)
    if gaussians.anchor_xyz.shape[0] != gaussians.get_xyz.shape[0] or gaussians.anchor_normal.shape[0] != gaussians.get_xyz.shape[0]:
        return SurfaceLossResult(zero, zero, zero, zero, zero)

    confidence = getattr(gaussians, "anchor_confidence", torch.ones(gaussians.get_xyz.shape[0], device=device))
    mask = confidence >= float(min_confidence)
    if hasattr(gaussians, "surface_state"):
        # Anchor only non-rejected surfaces (REJECTED geometry must not pull
        # Gaussians). CANDIDATE/PROVISIONAL/TRUSTED all participate; the
        # confidence gate above already suppresses unknown/low-confidence mass.
        mask = mask & (gaussians.surface_state != int(SurfaceState.REJECTED))
    if not mask.any():
        return SurfaceLossResult(zero, zero, zero, zero, zero)

    xyz = gaussians.get_xyz[mask]
    anchor = gaussians.anchor_xyz[mask].detach()
    n = torch.nn.functional.normalize(gaussians.anchor_normal[mask].detach(), dim=-1)
    conf = confidence[mask].detach().clamp(0.0, 1.0)
    delta = xyz - anchor
    normal_component = (delta * n).sum(dim=-1)
    tangent = delta - normal_component.unsqueeze(-1) * n
    L_normal = (conf * normal_component.pow(2)).mean()
    L_tangent = (conf * tangent.pow(2).sum(dim=-1)).mean()

    L_align = zero
    if normal_align_weight > 0 and hasattr(gaussians, "get_rotation"):
        # Approximate Gaussian local normal as the rotated +Z axis.
        q = torch.nn.functional.normalize(gaussians.get_rotation[mask], dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        local_z = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
        local_z = torch.nn.functional.normalize(local_z, dim=-1)
        L_align = (conf * (1.0 - (local_z * n).sum(dim=-1).abs())).mean()

    L_thin = zero
    if thinness_weight > 0 and hasattr(gaussians, "get_scaling"):
        scales = gaussians.get_scaling[mask]
        min_s = scales.min(dim=-1).values
        max_s = scales.max(dim=-1).values.clamp_min(1e-6)
        L_thin = (conf * (min_s / max_s).clamp(0.0, 1.0)).mean()

    total = normal_weight * L_normal + tangent_weight * L_tangent + normal_align_weight * L_align + thinness_weight * L_thin
    return SurfaceLossResult(total, L_normal, L_tangent, L_align, L_thin)
