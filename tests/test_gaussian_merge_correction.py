"""Tests for the non-rigid per-pair Gaussian merge correction (task 6)."""
from __future__ import annotations

import torch

from loop_closure.submap_correction import apply_gaussian_merge_correction


class _FakeModel:
    """Minimal model exposing exactly what apply_gaussian_merge_correction needs."""

    def __init__(self, xyz, support_count, provisional, opacities):
        self.params = {"means": torch.nn.Parameter(xyz)}
        self.support_count = support_count
        self.provisional = provisional
        self.raw_opacities = torch.nn.Parameter(opacities)
        self.visibility_ema = torch.ones((xyz.shape[0], 1))

    @property
    def get_xyz(self):
        return self.params["means"]


def test_close_duplicate_pair_is_absorbed_not_pulled():
    """A near-exact duplicate (mature, well-supported vs. provisional,
    barely-supported, same position) should be absorbed: the provisional
    duplicate's support folds into the mature one, position of the mature
    Gaussian is untouched (it's the ground truth)."""
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],   # mature, index 0
        [0.001, 0.0, 0.0],  # provisional duplicate, index 1 -- essentially same spot
    ])
    support_count = torch.tensor([50, 1], dtype=torch.int16)
    provisional = torch.tensor([False, True])
    opacities = torch.zeros(2)

    model = _FakeModel(xyz, support_count, provisional, opacities)
    n_corrected = apply_gaussian_merge_correction(model, match_radius=0.1)

    assert n_corrected == 1
    # The mature Gaussian's own position must not move -- absorption pulls
    # the DUPLICATE's evidence in, it doesn't relocate the established point.
    assert torch.allclose(model.get_xyz[0], torch.tensor([0.0, 0.0, 0.0]))
    # The provisional duplicate's support_count is folded into the mature one.
    assert model.support_count[0] > 50


def test_moderately_offset_pair_gets_local_pull_not_absorption():
    """A pair far enough apart that they're plausibly the same surface with
    real disagreement (not just noise) gets a small local pull toward the
    mature Gaussian, NOT a full absorb -- and critically, this must be a
    PER-PAIR delta, not a single rigid transform shared by unrelated
    Gaussians elsewhere in the map."""
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],   # mature
        [0.05, 0.0, 0.0],  # provisional, 5cm off -- real disagreement, not noise
        [10.0, 10.0, 10.0],  # unrelated mature Gaussian far away, must be untouched
    ])
    support_count = torch.tensor([50, 1, 50], dtype=torch.int16)
    provisional = torch.tensor([False, True, False])
    opacities = torch.zeros(3)

    model = _FakeModel(xyz, support_count, provisional, opacities)
    original_far_point = model.get_xyz[2].clone()
    n_corrected = apply_gaussian_merge_correction(model, match_radius=0.1)

    assert n_corrected == 1
    # The provisional point moved TOWARD the mature one, but not all the way
    # (a "pull", not a snap) -- x coordinate strictly between 0 and 0.05.
    assert 0.0 < model.get_xyz[1, 0].item() < 0.05
    # The unrelated far-away Gaussian is untouched -- proves this is a
    # per-pair local correction, not a submap-wide rigid transform.
    assert torch.allclose(model.get_xyz[2], original_far_point)


def test_no_matches_returns_zero():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    support_count = torch.tensor([50, 1], dtype=torch.int16)
    provisional = torch.tensor([False, True])
    opacities = torch.zeros(2)
    model = _FakeModel(xyz, support_count, provisional, opacities)
    assert apply_gaussian_merge_correction(model, match_radius=0.1) == 0


def test_many_to_one_absorption_accumulates_support_not_last_write_wins():
    """Regression test for whole-branch-review finding I-1: when TWO
    provisional duplicates absorb into the SAME mature target (normal --
    match_gaussians is many-to-one), plain advanced-index `+=` is
    last-write-wins and silently drops one source's support. index_add_
    must accumulate both."""
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],     # mature, index 0
        [0.001, 0.0, 0.0],   # provisional duplicate #1, index 1
        [-0.001, 0.0, 0.0],  # provisional duplicate #2, index 2 -- same target
    ])
    support_count = torch.tensor([50, 3, 7], dtype=torch.int16)
    provisional = torch.tensor([False, True, True])
    opacities = torch.zeros(3)

    model = _FakeModel(xyz, support_count, provisional, opacities)
    n_corrected = apply_gaussian_merge_correction(model, match_radius=0.1)

    assert n_corrected == 2
    # Both sources' support must be folded in: 50 + 3 + 7, not just one of them.
    assert model.support_count[0] == 60


def test_absorbed_provisional_is_not_reabsorbed_on_a_second_call():
    """Regression test for whole-branch-review finding I-2: an absorbed
    provisional must not keep provisional=True, or it gets re-selected as a
    source on every subsequent loop-closure event and its (already-merged)
    support_count compounds into the mature target again, unbounded."""
    xyz = torch.tensor([
        [0.0, 0.0, 0.0],    # mature
        [0.001, 0.0, 0.0],  # provisional duplicate
    ])
    support_count = torch.tensor([50, 3], dtype=torch.int16)
    provisional = torch.tensor([False, True])
    opacities = torch.zeros(2)

    model = _FakeModel(xyz, support_count, provisional, opacities)
    n_corrected_first = apply_gaussian_merge_correction(model, match_radius=0.1)
    assert n_corrected_first == 1
    assert model.support_count[0] == 53
    assert bool(model.provisional[1]) is False

    n_corrected_second = apply_gaussian_merge_correction(model, match_radius=0.1)
    assert n_corrected_second == 0
    assert model.support_count[0] == 53
