from __future__ import annotations

import types

import torch

from eligsir.utils.strategies.surface_mcmc_strategy import sample_surface_manifold_noise


def test_surface_manifold_noise_damps_trusted_normal_component():
    torch.manual_seed(1)
    g = types.SimpleNamespace()
    g.get_xyz = torch.zeros(8, 3)
    g.anchor_normal = torch.zeros(8, 3)
    g.anchor_normal[:, 2] = 1.0
    g.anchor_confidence = torch.ones(8)
    g.provisional = torch.zeros(8, dtype=torch.bool)
    noise = sample_surface_manifold_noise(g, noise_idx=torch.arange(8), xyz_lr=1.0, noise_lr=0.01, normal_scale=0.0)
    assert noise.shape == (8, 3)
    assert torch.allclose(noise[:, 2], torch.zeros(8), atol=1e-6)


def test_surface_manifold_noise_zero_normal_is_finite():
    """Finding I: a zero/degenerate anchor normal must not produce NaN; it
    falls back to isotropic base noise."""
    torch.manual_seed(2)
    g = types.SimpleNamespace()
    g.get_xyz = torch.zeros(6, 3)
    g.anchor_normal = torch.zeros(6, 3)  # all-zero normals (relocated/uninitialized slots)
    g.anchor_confidence = torch.ones(6)
    g.provisional = torch.zeros(6, dtype=torch.bool)
    g.spatial_lr_scale = 1.0
    noise = sample_surface_manifold_noise(g, noise_idx=torch.arange(6), xyz_lr=1.0, noise_lr=0.05)
    assert torch.isfinite(noise).all()
    # With no usable normal, the z-component is not forcibly damped to zero.
    assert noise.abs().sum() > 0.0


def test_surface_manifold_noise_provisional_explores_more():
    """Provisional Gaussians keep a larger normal-direction component than trusted ones."""
    torch.manual_seed(3)
    n = 256
    g = types.SimpleNamespace()
    g.get_xyz = torch.zeros(n, 3)
    g.anchor_normal = torch.zeros(n, 3)
    g.anchor_normal[:, 2] = 1.0
    g.anchor_confidence = torch.ones(n)
    g.provisional = torch.zeros(n, dtype=torch.bool)
    g.provisional[n // 2:] = True
    noise = sample_surface_manifold_noise(
        g, noise_idx=torch.arange(n), xyz_lr=1.0, noise_lr=1.0,
        normal_scale=0.1, provisional_normal_scale=0.9,
    )
    trusted_z = noise[: n // 2, 2].abs().mean()
    provisional_z = noise[n // 2:, 2].abs().mean()
    assert provisional_z > trusted_z
