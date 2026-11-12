"""Surface-manifold MCMC proposal helpers."""

from __future__ import annotations

import torch


def sample_surface_manifold_noise(
    gaussians,
    *,
    noise_idx: torch.Tensor,
    xyz_lr: float,
    noise_lr: float,
    normal_scale: float = 0.15,
    tangent_scale: float = 1.0,
    provisional_normal_scale: float = 0.75,
) -> torch.Tensor:
    """Sample tangent-biased noise for surface-supported Gaussians."""

    device = gaussians.get_xyz.device
    noise = torch.zeros_like(gaussians.get_xyz)
    if noise_idx.numel() == 0:
        return noise
    base = torch.randn((noise_idx.numel(), 3), device=device) * float(noise_lr) * float(xyz_lr)
    if not hasattr(gaussians, "anchor_normal") or gaussians.anchor_normal.shape[0] != gaussians.get_xyz.shape[0]:
        noise[noise_idx] = base
        return noise
    # Finding I: guard against zero/degenerate anchor normals (freshly relocated
    # or uninitialized slots). Such Gaussians get plain isotropic base noise
    # rather than a NaN/ill-defined tangent decomposition.
    raw_normals = gaussians.anchor_normal[noise_idx].detach()
    normal_norm = raw_normals.norm(dim=-1, keepdim=True)
    has_normal = normal_norm > 1e-6
    normals = raw_normals / normal_norm.clamp_min(1e-8)
    normal_component = (base * normals).sum(dim=-1, keepdim=True) * normals
    tangent_component = base - normal_component
    n_scale = torch.full((noise_idx.numel(), 1), float(normal_scale), device=device)
    if hasattr(gaussians, "provisional") and gaussians.provisional.shape[0] == gaussians.get_xyz.shape[0]:
        n_scale = torch.where(
            gaussians.provisional[noise_idx].unsqueeze(-1),
            torch.full_like(n_scale, float(provisional_normal_scale)),
            n_scale,
        )
    confidence = getattr(gaussians, "anchor_confidence", torch.zeros(gaussians.get_xyz.shape[0], device=device))
    conf = confidence[noise_idx].detach().clamp(0.0, 1.0).unsqueeze(-1)
    surface_noise = tangent_component * float(tangent_scale) + normal_component * n_scale
    mixed = surface_noise * conf + base * (1.0 - conf)
    mixed = torch.where(has_normal, mixed, base)  # no usable normal -> isotropic base noise
    # Cap the per-step displacement relative to scene scale rather than a fixed
    # absolute (0.01 was tuned for a unit-radius scene and over-clamps large maps).
    scene_scale = float(getattr(gaussians, "spatial_lr_scale", 1.0)) or 1.0
    max_step = 0.01 * scene_scale
    step_norm = mixed.norm(dim=-1, keepdim=True).clamp_min(max_step)
    mixed = mixed * (max_step / step_norm).clamp(max=1.0)
    noise[noise_idx] = mixed
    return noise
