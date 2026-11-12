"""Unit tests for birth-iteration-gated Gaussian pruning (Lever 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.utils.energy_mcmc import compute_dead_mask  # noqa: E402


class _FakeGaussians:
    def __init__(self, alpha: torch.Tensor, birth_iter: torch.Tensor):
        self._alpha = alpha
        self.birth_iter = birth_iter.to(dtype=torch.int32)
        n = alpha.shape[0]
        self.get_opacity = alpha.unsqueeze(-1)
        self.get_scaling = torch.ones((n, 3)) * 0.01  # tiny, no scale kills
        # visibility_ema: below threshold so conservative kill (opacity+support) fires
        self.visibility_ema = torch.ones((n, 1)) * 0.001

    @property
    def get_xyz(self):
        return torch.zeros((self._alpha.shape[0], 3))


def test_young_prot_blocks_opacity_death():
    # Two Gaussians: both have opacity below dead threshold.
    # Gaussian 0 was born at iter 0 (old), Gaussian 1 born at iter 500 (young).
    alpha = torch.tensor([0.001, 0.001])  # both "dead" by opacity
    birth_iter = torch.tensor([0, 500], dtype=torch.int32)
    g = _FakeGaussians(alpha, birth_iter)

    # Without protection: both die
    mask_no_prot = compute_dead_mask(g, opacity_threshold=0.005, use_utility_quantile=False)
    assert mask_no_prot.tolist() == [True, True]

    # With protection at current_iter=600, young_prot_iters=300:
    # Gaussian 0: age=600, not young → dies
    # Gaussian 1: age=100, young → protected
    mask_prot = compute_dead_mask(
        g, opacity_threshold=0.005, use_utility_quantile=False,
        current_iter=600, young_prot_iters=300,
    )
    assert mask_prot.tolist() == [True, False]


def test_young_prot_zero_is_noop():
    """young_prot_iters=0 must be bit-exact with no-protection path."""
    alpha = torch.tensor([0.001, 0.001])
    birth_iter = torch.tensor([0, 500], dtype=torch.int32)
    g = _FakeGaussians(alpha, birth_iter)

    mask_zero = compute_dead_mask(
        g, opacity_threshold=0.005, use_utility_quantile=False,
        current_iter=600, young_prot_iters=0,
    )
    mask_none = compute_dead_mask(g, opacity_threshold=0.005, use_utility_quantile=False)
    assert torch.equal(mask_zero, mask_none)


def test_young_prot_after_window_expires():
    """Once iter age >= young_prot_iters, the Gaussian can die normally."""
    alpha = torch.tensor([0.001])
    birth_iter = torch.tensor([500], dtype=torch.int32)
    g = _FakeGaussians(alpha, birth_iter)

    # At iter=800, age=300 — within protection window (500 iters)
    mask_early = compute_dead_mask(
        g, opacity_threshold=0.005, use_utility_quantile=False,
        current_iter=800, young_prot_iters=500,
    )
    assert mask_early.tolist() == [False]

    # At iter=1100, age=600 — past protection window
    mask_late = compute_dead_mask(
        g, opacity_threshold=0.005, use_utility_quantile=False,
        current_iter=1100, young_prot_iters=500,
    )
    assert mask_late.tolist() == [True]
