"""Wave 3, PR W3-09: direct coverage of every ``SafetyFuses.apply`` branch,
plus the reseed/growth reason-accumulation fix (a real bug found during
self-review: the first version counted each distinct reason string as `1`
regardless of how many requests shared it)."""

from __future__ import annotations

import torch

from eligsir.optimization.gaussian.strategies.contracts import (
    GaussianLifecycleState,
    SurfaceSeedRequest,
)
from eligsir.optimization.gaussian.strategies.live.mutation_plan import SafetyFuses
from eligsir.optimization.gaussian.strategies.live.state import state_to_code


def _fuses_kwargs(**overrides):
    n = 10
    base = dict(
        retire_indices=torch.arange(n, dtype=torch.int64),
        retire_scores=torch.linspace(1.0, 10.0, n),
        prune_indices=torch.arange(n, dtype=torch.int64),
        prune_scores=torch.linspace(1.0, 10.0, n),
        reseed_requests=tuple(SurfaceSeedRequest(cell_id=i, utility=float(i), reason="t") for i in range(n)),
        growth_requests=tuple(SurfaceSeedRequest(cell_id=i, utility=float(i), reason="t") for i in range(n)),
        geometry_confidence=torch.zeros(n),
        n_live=n,
    )
    base.update(overrides)
    return base


def test_fuse_caps_retire_fraction():
    fuses = SafetyFuses(max_retire_fraction=0.3, max_prune_fraction=1.0, max_reseed_requests=1000, max_growth_requests=1000)
    retire, prune, reseed, growth, triggers = fuses.apply(**_fuses_kwargs())
    assert retire.numel() == 3  # 0.3 * 10
    assert triggers["safety_fuse:max_retire_fraction"] == 7
    assert prune.numel() == 10


def test_fuse_caps_retire_fraction_tie_break_is_deterministic_index_ascending():
    """SafetyFuses.apply()'s own max_retire_fraction truncation re-derives
    priority order via its own argsort (separate code path from
    build_mutation_plan's priority_order sort) -- it needs the same
    deterministic tie-break. Ties by construction (identical scores for a
    block of indices); with a permissive cap that keeps only the tied
    block plus one extra, the kept set must be the lowest-index members of
    the tie, repeatably."""
    n = 10
    scores = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    fuses = SafetyFuses(max_retire_fraction=0.3, max_prune_fraction=1.0, max_reseed_requests=1000, max_growth_requests=1000)
    retire, _prune, _reseed, _growth, triggers = fuses.apply(
        **_fuses_kwargs(retire_indices=torch.arange(n, dtype=torch.int64), retire_scores=scores)
    )
    assert retire.numel() == 3  # 0.3 * 10
    assert triggers["safety_fuse:max_retire_fraction"] == 7
    # All score-10.0 ties: kept set must be the lowest-index 3 of the tie,
    # AND in ascending-index order (not just membership -- sorted() would
    # defeat exactly the ordering guarantee stable=True provides).
    assert retire.tolist() == [0, 1, 2]


def test_fuse_caps_prune_fraction():
    fuses = SafetyFuses(max_retire_fraction=1.0, max_prune_fraction=0.2, max_reseed_requests=1000, max_growth_requests=1000)
    retire, prune, reseed, growth, triggers = fuses.apply(**_fuses_kwargs())
    assert prune.numel() == 2  # 0.2 * 10
    assert triggers["safety_fuse:max_prune_fraction"] == 8


def test_fuse_caps_reseed_and_growth_requests():
    fuses = SafetyFuses(max_retire_fraction=1.0, max_prune_fraction=1.0, max_reseed_requests=3, max_growth_requests=4)
    retire, prune, reseed, growth, triggers = fuses.apply(**_fuses_kwargs())
    assert len(reseed) == 3
    assert len(growth) == 4
    assert triggers["safety_fuse:max_reseed_requests"] == 7
    assert triggers["safety_fuse:max_growth_requests"] == 6
    # Highest-utility requests must be the ones kept.
    assert {r.cell_id for r in reseed} == {7, 8, 9}


def test_fuse_never_touches_high_confidence_gaussians():
    n = 5
    kwargs = _fuses_kwargs(
        retire_indices=torch.arange(n, dtype=torch.int64),
        retire_scores=torch.ones(n),
        geometry_confidence=torch.tensor([0.99, 0.1, 0.1, 0.1, 0.1]),
        n_live=n,
    )
    fuses = SafetyFuses(max_retire_fraction=1.0, max_prune_fraction=1.0, max_reseed_requests=1000, max_growth_requests=1000)
    retire, _prune, _reseed, _growth, triggers = fuses.apply(**kwargs)
    assert 0 not in retire.tolist()
    assert triggers["safety_fuse:never_touch_high_confidence"] == 1


def test_fuse_no_triggers_when_within_all_caps():
    fuses = SafetyFuses()
    kwargs = _fuses_kwargs(
        retire_indices=torch.tensor([0], dtype=torch.int64),
        retire_scores=torch.tensor([1.0]),
        prune_indices=torch.tensor([], dtype=torch.int64),
        prune_scores=torch.tensor([]),
        reseed_requests=(),
        growth_requests=(),
        geometry_confidence=torch.zeros(1000),
        n_live=1000,
    )
    retire, prune, reseed, growth, triggers = fuses.apply(**kwargs)
    assert triggers == {}
    assert retire.numel() == 1


def test_build_mutation_plan_accumulates_reseed_reason_counts_not_constant_one():
    """Regression test for the counting bug found during self-review:
    reseed_reasons/growth_reasons must count *how many* admitted requests
    share a reason, not just whether at least one did."""
    from eligsir.optimization.gaussian.strategies.live.cell_budget import CellBudgetTracker
    from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
    from eligsir.optimization.gaussian.strategies.live.mutation_plan import build_mutation_plan
    from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState

    state = LivePopulationState()
    state.align(4, topology_version=0, iteration=0)

    reseed_candidates = tuple(
        SurfaceSeedRequest(cell_id=i, utility=1.0, reason="under_supported_mass") for i in range(5)
    ) + tuple(SurfaceSeedRequest(cell_id=100 + i, utility=1.0, reason="new_surface") for i in range(2))

    plan, _streaks = build_mutation_plan(
        state=state,
        thresholds=LifecycleThresholds(),
        fuses=SafetyFuses(),
        iteration=0,
        topology_version=0,
        reseed_candidates=reseed_candidates,
        cell_budget=CellBudgetTracker(budget_per_cell=1, max_total_requests=1000),
    )
    assert plan.reasons.reseed_reasons["under_supported_mass"] == 5
    assert plan.reasons.reseed_reasons["new_surface"] == 2


def test_build_mutation_plan_retire_indices_are_priority_ordered():
    """W3-55 regression test: PopulationMutationPlan.retire_indices must
    arrive in descending retire_score order (highest-priority-to-retire
    first), not physical-index order -- this is the only ordering signal
    available to AlignedBufferRegistry.apply()'s max_unreclaimed_retired
    cap truncation (live/backend.py), since the frozen plan contract
    carries no separate scores field. Without this, a partial retirement
    (once that cap is ever configured > 0) would keep the lowest-index
    candidates instead of the highest-priority ones -- an audit finding
    from the #57-#78 review, not urgent while the cap defaults to 0 but a
    real latent bug once task #47's Phase C candidate producer exists.
    """
    from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
    from eligsir.optimization.gaussian.strategies.live.mutation_plan import build_mutation_plan
    from eligsir.optimization.gaussian.strategies.live.scoring import retire_score
    from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState

    n = 6
    state = LivePopulationState()
    state.align(n, topology_version=0, iteration=0)
    # All rows SUSPECT (the only lifecycle state build_mutation_plan
    # considers as retire candidates) with deliberately non-monotonic
    # conflict/contribution/support so physical-index order and score
    # order disagree -- if the fix regressed to index order, this would
    # catch it.
    state.geometry_lifecycle[:] = state_to_code(GaussianLifecycleState.SUSPECT)
    state.free_space_conflict_ema = torch.tensor([0.9, 0.1, 0.5, 0.9, 0.2, 0.7])
    state.contribution_ema = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state.support_count = torch.tensor([0.0, 10.0, 2.0, 1.0, 5.0, 0.0])

    expected_scores = retire_score(
        conflict_ema=state.free_space_conflict_ema,
        contribution_ema=state.contribution_ema,
        support_count=state.support_count,
    )

    plan, _streaks = build_mutation_plan(
        state=state,
        thresholds=LifecycleThresholds(),
        fuses=SafetyFuses(
            max_retire_fraction=1.0, max_prune_fraction=1.0,
            max_reseed_requests=1000, max_growth_requests=1000,
        ),
        iteration=0,
        topology_version=0,
    )

    assert plan.retire_indices.numel() == n  # every row is a candidate here
    scores_in_plan_order = expected_scores[plan.retire_indices]
    assert torch.all(scores_in_plan_order[:-1] >= scores_in_plan_order[1:]), (
        f"retire_indices not in descending-score order: {plan.retire_indices.tolist()} "
        f"-> scores {scores_in_plan_order.tolist()}"
    )
    # Sanity: this genuinely differs from physical-index order for this input.
    assert plan.retire_indices.tolist() != sorted(plan.retire_indices.tolist())


def test_build_mutation_plan_retire_indices_tie_break_is_deterministic_index_ascending():
    """Follow-up to the priority-order test above (filed after the #80-#83
    review): when retire_score is TIED, ordering must be deterministic --
    secondary key index ascending -- not whatever torch.argsort's default
    (unstable) tie-breaking happens to produce on a given backend/kernel.
    Constructs a state where several rows share an identical score by
    construction (same conflict/contribution/support inputs) and asserts
    the tied block comes out in ascending physical-index order, repeatably.
    """
    from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
    from eligsir.optimization.gaussian.strategies.live.mutation_plan import build_mutation_plan
    from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState

    n = 6
    state = LivePopulationState()
    state.align(n, topology_version=0, iteration=0)
    state.geometry_lifecycle[:] = state_to_code(GaussianLifecycleState.SUSPECT)
    # Rows 1, 3, 5 share identical inputs (tied score); rows 0, 2, 4 differ
    # and are strictly higher, so the tied block is unambiguous in the
    # output and not just accidentally already-sorted input order.
    state.free_space_conflict_ema = torch.tensor([0.9, 0.5, 0.9, 0.5, 0.9, 0.5])
    state.contribution_ema = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state.support_count = torch.tensor([0.0, 3.0, 0.0, 3.0, 0.0, 3.0])

    plan, _streaks = build_mutation_plan(
        state=state,
        thresholds=LifecycleThresholds(),
        fuses=SafetyFuses(
            max_retire_fraction=1.0, max_prune_fraction=1.0,
            max_reseed_requests=1000, max_growth_requests=1000,
        ),
        iteration=0,
        topology_version=0,
    )

    assert plan.retire_indices.numel() == n
    tied_block = [i for i in plan.retire_indices.tolist() if i in (1, 3, 5)]
    assert tied_block == [1, 3, 5], (
        f"tied-score rows did not come out in deterministic ascending-index "
        f"order: {tied_block} (full order: {plan.retire_indices.tolist()})"
    )
