"""Part D1: parallax / baseline-gated consensus voting — unit tests.

Tests the standalone, host-CPU-testable helper ``baseline_gate_mask``. No GPU
and no train_streaming import (which pulls torchvision) is required.
"""

import math

import torch

from eligsir.utils.streaming_depth_lock import baseline_gate_mask


def test_nan_rows_always_accept():
    # First voter (NaN row) is always accepted, regardless of camera position.
    centers = torch.full((3, 3), float("nan"))
    cam = torch.tensor([1.0, 2.0, 3.0])
    mask = baseline_gate_mask(centers, cam, min_baseline=0.03)
    assert mask.dtype == torch.bool
    assert bool(mask.all())


def test_near_rows_rejected_far_rows_accepted():
    cam = torch.tensor([0.0, 0.0, 0.0])
    centers = torch.tensor(
        [
            [0.001, 0.0, 0.0],   # 1mm  -> reject (< 3cm)
            [0.10, 0.0, 0.0],    # 10cm -> accept (>= 3cm)
            [0.029, 0.0, 0.0],   # just under threshold -> reject
            [0.030, 0.0, 0.0],   # exactly at threshold -> accept (>=)
        ]
    )
    mask = baseline_gate_mask(centers, cam, min_baseline=0.03)
    assert mask.tolist() == [False, True, False, True]


def test_mixed_nan_and_distance():
    cam = torch.tensor([0.0, 0.0, 0.0])
    centers = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],  # first voter -> accept
            [0.001, 0.0, 0.0],                           # too close   -> reject
            [0.0, 0.0, 5.0],                             # far         -> accept
        ]
    )
    mask = baseline_gate_mask(centers, cam, min_baseline=0.03)
    assert mask.tolist() == [True, False, True]


def test_partial_nan_row_treated_as_first_voter():
    # A row with any NaN component counts as "no voter yet".
    cam = torch.tensor([0.0, 0.0, 0.0])
    centers = torch.tensor([[0.001, float("nan"), 0.001]])
    mask = baseline_gate_mask(centers, cam, min_baseline=0.03)
    assert bool(mask.all())


def test_empty_input():
    centers = torch.empty((0, 3))
    cam = torch.tensor([0.0, 0.0, 0.0])
    mask = baseline_gate_mask(centers, cam, min_baseline=0.03)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_distance_is_euclidean():
    cam = torch.tensor([0.0, 0.0, 0.0])
    # 3-4-5: norm == 0.05 -> accept at 0.03, reject at 0.06
    centers = torch.tensor([[0.03, 0.04, 0.0]])
    assert bool(baseline_gate_mask(centers, cam, 0.03).item())
    assert not bool(baseline_gate_mask(centers, cam, 0.06).item())
    assert math.isclose((centers[0]).norm().item(), 0.05, rel_tol=1e-6)
