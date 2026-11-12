"""CPU parity coverage for ``assemble_step_regularizers`` (issue #1670 Phase
A -- see ``.superpowers/sdd/2026-09-03-throughput-recovery-tracks-1-2/
step-regularizers-brief.md`` -- and its divisor-contract fix -- see
``.superpowers/sdd/2026-09-03-throughput-recovery-tracks-1-2/
step-regularizers-fix-brief.md``).

Every per-Gaussian regularizer term the per-view loop adds under
``objective.assemble_loss``'s ``m_loss_regs_gaussian`` segment (sparse
active-set branch) is a masked MEAN of a view-independent per-Gaussian
quantity ``f_i``, so ``(1/B) sum_v reg_v`` reduces to exactly ONE evaluation
with per-Gaussian weights ``weights[i] = sum_v mask_v[i] / n_v`` and an
explicit, caller-supplied ``batch_size=B`` divisor (see
``assemble_step_regularizers``'s docstring). These tests pin that identity
-- forward value AND gradients -- against the existing per-view functions
(``active_reg_core_masked`` + ``_streaming_young_opacity_reg``), for B in
(1, 2, 4, 8) and random masks, including a dedicated case with a Gaussian
visible in every view, plus a launch-count check mirroring
``eligsir-lab/tests/test_batched_supervision_loss.py``.

RETRACTED: earlier revisions of this module (and of
``assemble_step_regularizers`` itself) treated ``weights.sum()`` -- "count
of non-empty views" -- as if it were always equal to ``B``, and pinned a
genuinely-empty-view test against a reference computed by dividing by that
non-empty count instead of the true ``batch_size``. That was the bug: see
``test_one_empty_view_is_finite_and_matches_the_non_empty_views`` below,
now rewritten to assert the CORRECT ``/batch_size`` answer, and the two new
tests (``test_two_groups_of_two_sum_to_the_single_group_of_four``,
``test_skipped_view_count_scales_by_the_requested_batch_not_the_group_size``)
that exercise the multi-group/skipped-view cases where the two diverge.

A genuinely EMPTY view is still exercised through
``.mean()``-on-empty-subset separately
(``test_streaming_young_opacity_reg_nans_on_a_genuinely_empty_mask``): the
OLD per-view ``_streaming_young_opacity_reg`` is ``NaN`` for a genuinely
empty ``active_mask`` in every configuration -- a pre-existing hazard of
dynamic compaction, unrelated to this slice, that the dense fixed-shape
form does not reproduce (same rationale as issue #1670's
``active_reg_core_masked``, whose own docstring documents the
empty-mask-returns-zero behavior this function inherits).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.losses import (
    _streaming_young_opacity_reg,
    assemble_step_regularizers,
)
from eligsir.utils.compiled_kernels import active_reg_core_masked

N = 6
ITERATION = 500


def _gaussians(*, with_state: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    opacity_raw = torch.randn(N, 1, generator=g, dtype=torch.float32).requires_grad_(True)
    scaling_raw = (torch.rand(N, 3, generator=g, dtype=torch.float32) + 0.1).requires_grad_(True)
    gaussians = SimpleNamespace(
        get_opacity=torch.sigmoid(opacity_raw),
        get_scaling=scaling_raw,
        provisional=torch.zeros(N, dtype=torch.bool),
    )
    if with_state:
        gaussians.birth_iter = torch.randint(0, 2 * ITERATION, (N,), generator=g)
        gaussians.support_count = torch.randint(0, 20, (N,), generator=g).float()
        gaussians.last_supervised_iter = torch.randint(0, ITERATION, (N,), generator=g)
    return gaussians, opacity_raw, scaling_raw


def _args(*, young_exempt_iters=0, maturity_ramp=False, recency_horizon=0.0):
    return SimpleNamespace(
        opacity_reg=0.05,
        scale_reg=0.02,
        anisotropy_reg=0.1,
        streaming_young_reg_exempt_iters=young_exempt_iters,
        streaming_maturity_reg_ramp=maturity_ramp,
        streaming_recency_reg_horizon=recency_horizon,
        streaming_debt_replay_target_support=10,
        streaming_maturity_reg_ramp_min=0.2,
        streaming_recency_reg_ramp_min=0.2,
    )


def _dense_masks(b: int, *, seed: int) -> torch.Tensor:
    """Random masks with every view guaranteed non-empty (the regime the
    exact-parity derivation assumes -- ``n_v`` is a real, non-clamped
    divisor for every view)."""
    g = torch.Generator().manual_seed(seed)
    masks = torch.rand(b, N, generator=g) > 0.5
    # Force at least one True per row so no view is accidentally empty.
    masks[:, 0] = True
    return masks


def _dense_masks_with_one_always_visible(b: int, *, seed: int) -> torch.Tensor:
    """Like `_dense_masks`, plus one Gaussian visible in EVERY view (exercises
    the upper end of the per-Gaussian weight, ``weights[i] == B``)."""
    masks = _dense_masks(b, seed=seed)
    masks[:, 1] = True
    return masks


def _weights(masks: torch.Tensor) -> torch.Tensor:
    n_v = masks.sum(dim=1).clamp_min(1).to(torch.float32)
    return (masks.to(torch.float32) / n_v[:, None]).sum(dim=0)


def _reference_loop(gaussians, args, masks: torch.Tensor) -> torch.Tensor:
    """What the executor's per-view loop computes today, summed x 1/B."""
    opacity = gaussians.get_opacity
    scaling = gaussians.get_scaling
    b = masks.shape[0]
    total = opacity.new_zeros(())
    for v in range(b):
        mask_v = masks[v]
        term = active_reg_core_masked(
            opacity, scaling, mask_v,
            w_opacity=0.0, w_scale=args.scale_reg, w_aniso=float(args.anisotropy_reg),
        )
        term = term + _streaming_young_opacity_reg(
            gaussians, args, ITERATION, mask_v,
            int(args.streaming_young_reg_exempt_iters),
            bool(args.streaming_maturity_reg_ramp),
        )
        total = total + term
    return total / float(b)


_CASES = {
    "nothing_applicable": dict(young_exempt_iters=0, maturity_ramp=False, recency_horizon=0.0),
    "young_and_maturity": dict(young_exempt_iters=50, maturity_ramp=True, recency_horizon=0.0),
    "young_maturity_recency": dict(young_exempt_iters=50, maturity_ramp=True, recency_horizon=100.0),
}


@pytest.mark.parametrize("b", [1, 2, 4, 8])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_matches_per_view_loop_value_and_gradients(b, case):
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=1)
    args = _args(**_CASES[case])
    masks = _dense_masks(b, seed=100 + b)
    weights = _weights(masks)

    batched = assemble_step_regularizers(
        gaussians, args, ITERATION, weights, batch_size=b,
    )
    reference = _reference_loop(gaussians, args, masks)

    torch.testing.assert_close(batched, reference, atol=1e-6, rtol=1e-5)

    g_batched = torch.autograd.grad(batched, [opacity_raw, scaling_raw], retain_graph=True)
    g_reference = torch.autograd.grad(reference, [opacity_raw, scaling_raw], retain_graph=True)
    for gb, gr in zip(g_batched, g_reference):
        torch.testing.assert_close(gb, gr, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("b", [2, 4, 8])
def test_matches_per_view_loop_with_a_gaussian_visible_in_every_view(b):
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=6)
    args = _args(young_exempt_iters=50, maturity_ramp=True, recency_horizon=100.0)
    masks = _dense_masks_with_one_always_visible(b, seed=300 + b)
    weights = _weights(masks)

    batched = assemble_step_regularizers(gaussians, args, ITERATION, weights, batch_size=b)
    reference = _reference_loop(gaussians, args, masks)

    torch.testing.assert_close(batched, reference, atol=1e-6, rtol=1e-5)
    g_batched = torch.autograd.grad(batched, [opacity_raw, scaling_raw], retain_graph=True)
    g_reference = torch.autograd.grad(reference, [opacity_raw, scaling_raw], retain_graph=True)
    for gb, gr in zip(g_batched, g_reference):
        torch.testing.assert_close(gb, gr, atol=1e-6, rtol=1e-5)


def test_matches_per_view_loop_without_optional_lifecycle_state():
    """Same identity when birth_iter/support_count/last_supervised_iter are
    entirely absent (dense/no-state gaussians fake) -- both sides fall back
    to the "nothing applicable" full-strength opacity reg."""
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=False, seed=2)
    args = _args(young_exempt_iters=50, maturity_ramp=True, recency_horizon=100.0)
    masks = _dense_masks(4, seed=7)
    weights = _weights(masks)

    batched = assemble_step_regularizers(gaussians, args, ITERATION, weights, batch_size=4)
    reference = _reference_loop(gaussians, args, masks)

    torch.testing.assert_close(batched, reference, atol=1e-6, rtol=1e-5)
    g_batched = torch.autograd.grad(batched, [opacity_raw, scaling_raw], retain_graph=True)
    g_reference = torch.autograd.grad(reference, [opacity_raw, scaling_raw], retain_graph=True)
    for gb, gr in zip(g_batched, g_reference):
        torch.testing.assert_close(gb, gr, atol=1e-6, rtol=1e-5)


def test_streaming_young_opacity_reg_returns_zero_for_an_empty_mask():
    """The current per-view and dense paths both treat an empty view as zero."""
    gaussians, _opacity_raw, _scaling_raw = _gaussians(with_state=True, seed=3)
    args = _args(young_exempt_iters=50, maturity_ramp=True)
    empty_mask = torch.zeros(N, dtype=torch.bool)
    out = _streaming_young_opacity_reg(
        gaussians, args, ITERATION, empty_mask,
        int(args.streaming_young_reg_exempt_iters),
        bool(args.streaming_maturity_reg_ramp),
    )
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_one_empty_view_is_finite_and_matches_the_non_empty_views():
    """A genuinely empty view among otherwise-normal views: the dense form
    stays finite (the empty view contributes exactly 0 weight, matching
    ``_active_reg_core_masked``'s own empty-mask handling) and the result
    equals the sum over the 3 NON-EMPTY views divided by the TRUE batch
    size ``b=4`` -- NOT divided by 3. Dividing by the non-empty count (the
    old, retracted answer) is exactly the ``B / weights.sum()``
    over-weighting bug from step-regularizers-fix-brief.md: this pins the
    corrected contract instead."""
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=8)
    args = _args(young_exempt_iters=50, maturity_ramp=True)
    b = 4
    masks = _dense_masks(b, seed=42)
    masks[0] = False  # genuinely empty view

    weights = _weights(masks)
    assert bool(torch.isfinite(weights).all())

    batched = assemble_step_regularizers(gaussians, args, ITERATION, weights, batch_size=b)
    assert bool(torch.isfinite(batched))

    non_empty = masks[1:]
    non_empty_count = non_empty.shape[0]
    # `_reference_loop` divides by its own `masks.shape[0]` (3, the
    # non-empty count here) -- rescale back up to the TRUE divisor `b` (4).
    reference_over_non_empty_count = _reference_loop(gaussians, args, non_empty)
    reference_over_b = reference_over_non_empty_count * (float(non_empty_count) / float(b))
    torch.testing.assert_close(batched, reference_over_b, atol=1e-6, rtol=1e-5)

    g_batched = torch.autograd.grad(batched, [opacity_raw, scaling_raw])
    assert all(bool(torch.isfinite(g).all()) for g in g_batched)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_two_groups_of_two_sum_to_the_single_group_of_four(case):
    """Mirrors the executor's microbatch/multi-B4-group path: splitting one
    B=4 call's views into two groups of 2 -- each scaled exactly as the
    executor does (``batch_size`` is the call's TRUE global divisor,
    ``requested_b=4``, for BOTH groups, not each group's own size) -- must
    sum to the same value as evaluating the whole group of 4 at once, and
    to the per-view loop's ``sum_v reg_v / 4``. Before the fix, passing
    each group's own ``len(group)=2`` as ``batch_size`` (equivalently,
    normalising by ``weights.sum()``) reproduced the reviewer's 2.0000x
    over-weight: two 2-view groups, each internally normalised as if it
    were the WHOLE batch, summed to 2x the correct total."""
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=9)
    args = _args(**_CASES[case])
    requested_b = 4
    masks = _dense_masks(requested_b, seed=500)

    weights_single = _weights(masks)
    single = assemble_step_regularizers(
        gaussians, args, ITERATION, weights_single, batch_size=requested_b,
    )

    two_group_total = None
    for group_masks in (masks[0:2], masks[2:4]):
        group_weights = _weights(group_masks)
        term = assemble_step_regularizers(
            gaussians, args, ITERATION, group_weights, batch_size=requested_b,
        )
        two_group_total = term if two_group_total is None else two_group_total + term

    reference = _reference_loop(gaussians, args, masks)

    torch.testing.assert_close(two_group_total, single, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(two_group_total, reference, atol=1e-6, rtol=1e-5)

    g_two_group = torch.autograd.grad(two_group_total, [opacity_raw, scaling_raw], retain_graph=True)
    g_single = torch.autograd.grad(single, [opacity_raw, scaling_raw], retain_graph=True)
    for g2, g1 in zip(g_two_group, g_single):
        torch.testing.assert_close(g2, g1, atol=1e-6, rtol=1e-5)


def test_skipped_view_count_scales_by_the_requested_batch_not_the_group_size():
    """Mirrors the executor's ``skipped_view_count`` path
    (``run_true_camera_batched_step``): a skipped view never reaches this
    function at all (it contributes 0 weight, same as a genuinely-empty
    mask), but still counts toward the TRUE divisor ``requested_b`` -- here
    ``requested_b = len(requests) + skipped_view_count = 3 + 1 = 4``, while
    only 3 views' masks actually produced ``weights``."""
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=11)
    args = _args(young_exempt_iters=50, maturity_ramp=True)
    requested_b = 4
    masks = _dense_masks(3, seed=77)  # 3 views present; 1 more was skipped upstream

    weights = _weights(masks)
    batched = assemble_step_regularizers(gaussians, args, ITERATION, weights, batch_size=requested_b)

    reference_over_3 = _reference_loop(gaussians, args, masks)
    reference_over_4 = reference_over_3 * (3.0 / float(requested_b))
    torch.testing.assert_close(batched, reference_over_4, atol=1e-6, rtol=1e-5)

    g_batched = torch.autograd.grad(batched, [opacity_raw, scaling_raw])
    assert all(bool(torch.isfinite(g).all()) for g in g_batched)


def _count_ops(fn) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
    return sum(int(e.count) for e in prof.key_averages() if str(e.key).startswith("aten::"))


@pytest.mark.parametrize("b", [2, 4, 8])
def test_launches_fewer_aten_ops_than_the_per_view_loop(b):
    gaussians, opacity_raw, scaling_raw = _gaussians(with_state=True, seed=5)
    args = _args(young_exempt_iters=50, maturity_ramp=True)
    masks = _dense_masks(b, seed=200 + b)
    weights = _weights(masks)

    def run_batched():
        assemble_step_regularizers(gaussians, args, ITERATION, weights, batch_size=b).backward(
            retain_graph=True
        )
        opacity_raw.grad = None
        scaling_raw.grad = None

    def run_loop():
        _reference_loop(gaussians, args, masks).backward(retain_graph=True)
        opacity_raw.grad = None
        scaling_raw.grad = None

    ops_batched = _count_ops(run_batched)
    ops_loop = _count_ops(run_loop)
    assert ops_batched < ops_loop, (b, ops_batched, ops_loop)
