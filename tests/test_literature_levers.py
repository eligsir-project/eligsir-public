"""Unit tests for the literature-motivated levers added on
feat/local-bounds-program:

  1. Decaying isotropy regularizer on immature Gaussians
     (train_streaming.young_isotropy_ramp).
  2. Depth-error-gated temperature multiplier on growth utility
     (utils.energy_mcmc.compute_gaussian_utility w_depth_residual).

CPU-only, no docker/GPU required (mirrors tests/test_spatial_bubble_mask.py).
"""
from __future__ import annotations

import types

import pytest
import torch

from eligsir.utils.compiled_kernels import configure_torch_compile, young_isotropy_ramp
from eligsir.utils.energy_mcmc import compute_gaussian_utility
from eligsir.utils.gaussian_energy_policy import GaussianEnergyConfig


@pytest.fixture(scope="session", autouse=True)
def _register_kernels():
    """Register compiled kernels in eager mode before any energy test
    (mirrors tests/test_energy_policy_parity.py)."""
    _args = types.SimpleNamespace(
        compile_mode="off",
        compile_dynamic=False,
        compile_after_iter=0,
        compile_sh=False,
        sh_impl="",
        compile_energy=False,
        compile_growth_margin=0,
    )
    configure_torch_compile(_args)


# ---------------------------------------------------------------------------
# Mechanism 1: isotropy ramp math
# ---------------------------------------------------------------------------

def test_ramp_scalar_endpoints_and_midpoint():
    ramp_frames = 15
    assert young_isotropy_ramp(0, ramp_frames) == 1.0
    assert young_isotropy_ramp(ramp_frames, ramp_frames) == 0.0
    assert young_isotropy_ramp(ramp_frames / 2, ramp_frames) == 0.5


def test_ramp_clamps_out_of_range_scalar():
    ramp_frames = 15
    assert young_isotropy_ramp(-5, ramp_frames) == 1.0
    assert young_isotropy_ramp(ramp_frames * 2, ramp_frames) == 0.0


def test_ramp_tensor_matches_scalar():
    ramp_frames = 15
    age = torch.tensor([0.0, 7.5, 15.0, -3.0, 30.0])
    out = young_isotropy_ramp(age, ramp_frames)
    expected = torch.tensor([1.0, 0.5, 0.0, 1.0, 0.0])
    assert torch.allclose(out, expected)


def test_ramp_degenerate_zero_window():
    assert young_isotropy_ramp(0, 0) == 0.0
    age = torch.tensor([0.0, 5.0])
    out = young_isotropy_ramp(age, 0)
    assert torch.allclose(out, torch.zeros_like(age))


# ---------------------------------------------------------------------------
# Mechanism 2: depth-residual gate on growth utility
# ---------------------------------------------------------------------------

def test_energy_config_has_depth_residual_field_default_zero():
    cfg = GaussianEnergyConfig()
    assert hasattr(cfg, "energy_w_depth_residual")
    assert cfg.energy_w_depth_residual == 0.0


class _FakeGaussians:
    """Minimal stand-in exposing the tensors compute_gaussian_utility reads."""

    def __init__(self, n=6):
        self._xyz = torch.zeros(n, 3, requires_grad=True)
        self._opacity = torch.full((n, 1), 0.5, requires_grad=True)
        self._scaling = torch.full((n, 3), 0.05, requires_grad=True)
        # Populate .grad directly (compute_gaussian_utility reads leaf .grad).
        self._xyz.grad = torch.linspace(0.0, 1.0, n).unsqueeze(-1).repeat(1, 3)
        self._opacity.grad = torch.zeros(n, 1)
        self._scaling.grad = torch.zeros(n, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return self._scaling.abs() + 1e-3


def _render_pkg(n=6):
    return {"visibility_filter": torch.ones(n, dtype=torch.bool)}


def test_depth_residual_zero_weight_is_noop_vs_explicit_zero_scalar():
    gaussians = _FakeGaussians()
    args = types.SimpleNamespace(energy_mode="legacy_utility")
    base = compute_gaussian_utility(
        gaussians=gaussians, render_pkg=_render_pkg(), iteration=100, args=args,
    )
    with_zero = compute_gaussian_utility(
        gaussians=gaussians, render_pkg=_render_pkg(), iteration=100, args=args,
        w_depth_residual=0.0, depth_residual_scalar=0.0,
    )
    assert torch.allclose(base, with_zero)


def test_depth_residual_nonzero_weight_changes_utility():
    gaussians = _FakeGaussians()
    args = types.SimpleNamespace(energy_mode="legacy_utility")
    zero_weight = compute_gaussian_utility(
        gaussians=gaussians, render_pkg=_render_pkg(), iteration=100, args=args,
        w_depth_residual=0.0, depth_residual_scalar=0.5,
    )
    nonzero_weight = compute_gaussian_utility(
        gaussians=gaussians, render_pkg=_render_pkg(), iteration=100, args=args,
        w_depth_residual=1.0, depth_residual_scalar=0.5,
    )
    assert not torch.allclose(zero_weight, nonzero_weight)
    # Gaussians with nonzero xyz gradient must actually shift (not a global
    # no-op multiplier) — confirms the signal reaches growth-utility ranking.
    assert not torch.allclose(zero_weight[1:], nonzero_weight[1:])
