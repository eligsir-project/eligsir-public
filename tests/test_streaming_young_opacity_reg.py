"""Unit tests for the live-path young-opacity-reg relaxation (Tasks 1 + 2).

Pure-function tests for ``young_opacity_reg_weights`` — no CUDA / no training.
Pins the composable per-Gaussian opacity-reg weighting that was DEAD on the live
streaming path (train.py had the young-exempt mask; train_streaming.py did not):

* Task 1 — ``streaming_young_reg_exempt_iters``: Gaussians younger than the
  threshold (age = iter - birth_iter) are exempt (weight 0).
* Task 2 — ``streaming_maturity_reg_ramp``: weight ramps with support_count from
  ``ramp_min`` (newborn) to 1.0 (well-supervised).

Both default OFF; with both off the weights are all 1.0 (== full opacity reg,
behavior-neutral vs. the unmodified active_reg_core call).
"""
import torch

from eligsir.utils.compiled_kernels import young_opacity_reg_weights


def _w(age, support, exempt, ramp, target=10.0, ramp_min=0.2):
    return young_opacity_reg_weights(
        age=torch.tensor(age, dtype=torch.int64) if age is not None else None,
        support_count=torch.tensor(support, dtype=torch.float32) if support is not None else None,
        young_exempt_iters=exempt,
        maturity_ramp=ramp,
        target_support=target,
        ramp_min=ramp_min,
    )


def test_both_off_is_full_strength():
    # Off/off -> every weight 1.0 (full opacity reg = behavior-neutral).
    w = _w(age=[0, 5, 100], support=[0, 1, 50], exempt=0, ramp=False)
    assert torch.allclose(w, torch.ones(3))


def test_young_exempt_zeros_young_only():
    # Task 1: age < exempt -> weight 0; age >= exempt -> weight 1.
    w = _w(age=[0, 99, 100, 500], support=None, exempt=100, ramp=False)
    assert torch.equal(w, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_young_exempt_boundary_is_inclusive_at_threshold():
    # Exactly at the threshold counts as mature (>= threshold).
    w = _w(age=[100], support=None, exempt=100, ramp=False)
    assert w.item() == 1.0


def test_maturity_ramp_floor_and_ceiling():
    # Task 2: clamp(support/target, ramp_min, 1.0).
    # target=10, ramp_min=0.2: support 0 -> 0.2 (floor), 5 -> 0.5, 10 -> 1.0,
    # 50 -> 1.0 (ceiling).
    w = _w(age=[0, 0, 0, 0], support=[0, 5, 10, 50], exempt=0, ramp=True, target=10.0, ramp_min=0.2)
    assert torch.allclose(w, torch.tensor([0.2, 0.5, 1.0, 1.0]))


def test_ramp_min_is_respected():
    w = _w(age=[0], support=[0], exempt=0, ramp=True, target=10.0, ramp_min=0.4)
    assert abs(w.item() - 0.4) < 1e-6


def test_exempt_and_ramp_compose_multiplicatively():
    # A young, low-support Gaussian: ramp would give ~0.2 but young-exempt
    # forces it to 0. A mature, low-support one keeps the ramp weight.
    w = _w(age=[10, 200], support=[0, 0], exempt=100, ramp=True, target=10.0, ramp_min=0.2)
    # idx0: age 10 < 100 -> exempt -> 0 ; idx1: age 200 >= 100, support 0 -> 0.2
    assert torch.allclose(w, torch.tensor([0.0, 0.2]))


def test_ramp_off_with_exempt_keeps_mature_at_one():
    # Only exemption active: mature Gaussians get 1.0 regardless of support.
    w = _w(age=[200], support=[0], exempt=100, ramp=False)
    assert w.item() == 1.0


# --- Recency-of-supervision ramp (2026-07-18) -------------------------------

def _w_recency(age, recency_gap, horizon, ramp_min=0.2):
    return young_opacity_reg_weights(
        age=torch.tensor(age, dtype=torch.int64),
        support_count=None,
        young_exempt_iters=0,
        maturity_ramp=False,
        target_support=10.0,
        ramp_min=0.2,
        recency_gap=torch.tensor(recency_gap, dtype=torch.int64) if recency_gap is not None else None,
        recency_horizon=horizon,
        recency_ramp_min=ramp_min,
    )


def test_recency_horizon_zero_is_full_strength():
    # Disabled (horizon <= 0) -> no-op, weight 1.0 regardless of gap.
    w = _w_recency(age=[0, 0], recency_gap=[0, 1000], horizon=0.0)
    assert torch.allclose(w, torch.ones(2))


def test_recency_gap_zero_is_full_strength():
    # Just supervised this iteration (gap=0) -> full pressure.
    w = _w_recency(age=[0], recency_gap=[0], horizon=100.0)
    assert w.item() == 1.0


def test_recency_gap_at_horizon_hits_floor():
    # gap >= horizon -> clamped to ramp_min.
    w = _w_recency(age=[0, 0], recency_gap=[100, 500], horizon=100.0, ramp_min=0.3)
    assert torch.allclose(w, torch.tensor([0.3, 0.3]))


def test_recency_ramps_linearly_between_zero_and_horizon():
    # gap=50, horizon=100 -> 1 - 50/100 = 0.5 (no ramp_min clamp in effect).
    w = _w_recency(age=[0], recency_gap=[50], horizon=100.0, ramp_min=0.2)
    assert abs(w.item() - 0.5) < 1e-6


def test_recency_composes_multiplicatively_with_young_exempt():
    # A young Gaussian stays fully exempt (0) even with a high recency weight;
    # a mature one keeps the recency ramp.
    w = young_opacity_reg_weights(
        age=torch.tensor([10, 200], dtype=torch.int64),
        support_count=None,
        young_exempt_iters=100,
        maturity_ramp=False,
        target_support=10.0,
        ramp_min=0.2,
        recency_gap=torch.tensor([0, 100], dtype=torch.int64),
        recency_horizon=100.0,
        recency_ramp_min=0.3,
    )
    # idx0: age 10 < 100 -> exempt -> 0 (recency weight of 1.0 doesn't matter)
    # idx1: age 200 >= 100 -> not exempt; recency_gap==horizon -> floor 0.3
    assert torch.allclose(w, torch.tensor([0.0, 0.3]))
