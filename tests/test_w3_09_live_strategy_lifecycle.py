"""Wave 3, PR W3-09: lifecycle transitions (plan section 14.6) + deliverable
5 ("stable Gaussians are not proposed for global teleportation under
ordinary conditions")."""

from __future__ import annotations

import torch

from eligsir.optimization.gaussian.strategies.contracts import GaussianLifecycleState
from eligsir.optimization.gaussian.strategies.live.mutation_plan import SafetyFuses
from eligsir.optimization.gaussian.strategies.live.state import code_to_state, state_to_code
from eligsir.optimization.gaussian.strategies.live.strategy import (
    LiveStrategyConfig,
    EliGSiRLiveGaussianStrategy,
)


def _run_steps(strat, n, iters, *, contribution, rgb_residual, depth_residual, free_space_conflict, geometry_confidence):
    vis = torch.ones(n, dtype=torch.bool)
    plan = report = None
    for it in iters:
        strat.observe(
            n_gaussians=n,
            iteration=it,
            topology_version=0,
            visible_mask=vis,
            contribution=torch.full((n,), contribution),
            rgb_residual=torch.full((n,), rgb_residual),
            depth_residual=torch.full((n,), depth_residual),
            free_space_conflict=torch.full((n,), free_space_conflict),
            geometry_confidence=torch.full((n,), geometry_confidence),
        )
        plan, report = strat.propose(iteration=it, topology_version=0)
    return plan, report


def test_provisional_to_active_requires_support():
    strat = EliGSiRLiveGaussianStrategy()
    n = 4
    # Only 1 step of observation: support_count=1 < default threshold (3).
    _run_steps(
        strat, n, range(1),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.9,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.PROVISIONAL)).all()

    _run_steps(
        strat, n, range(1, 5),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.9,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.ACTIVE)).all()


def test_active_to_stable_requires_sustained_geometry_convergence():
    strat = EliGSiRLiveGaussianStrategy()
    n = 4
    _run_steps(
        strat, n, range(0, 40),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.95,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.STABLE)).all()


def test_persistent_conflict_drives_suspect_then_retired():
    strat = EliGSiRLiveGaussianStrategy()
    n = 4
    # Reach ACTIVE first (small support), then hit it with sustained conflict.
    _run_steps(
        strat, n, range(0, 5),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.5,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.ACTIVE)).all()

    _run_steps(
        strat, n, range(5, 5 + 40),
        contribution=0.0, rgb_residual=0.5, depth_residual=0.5,
        free_space_conflict=0.9, geometry_confidence=0.1,
    )
    final_states = {code_to_state(int(c)) for c in strat.state.geometry_lifecycle.tolist()}
    assert final_states <= {GaussianLifecycleState.SUSPECT, GaussianLifecycleState.RETIRED}


def test_deliverable_5_stable_gaussians_never_proposed_for_retire_under_ordinary_conditions():
    """Construct STABLE-lifecycle Gaussians under ordinary (good) conditions
    and confirm the strategy never proposes them for retire (the only path
    to a subsequent global teleport/relocation -- see
    ``gsplat_mcmc_adapter.py``'s docstring), and that the lifecycle batch
    never demotes them out of STABLE under ordinary conditions either.
    """
    # max_retire_fraction/max_prune_fraction=1.0: at small N (10), the
    # default 0.05/0.02 fractions floor to int(0.05*10)=0 / int(0.02*10)=0,
    # which would make "retire_indices stays empty" trivially true via the
    # fuse clamp regardless of whether the SUSPECT-only candidacy guard
    # this test is actually about does anything at all (see
    # test_deliverable_5_stable_excluded_even_if_scoring_would_favor_it's
    # docstring for the full discriminator argument). Widening the fuse
    # here removes that confound.
    strat = EliGSiRLiveGaussianStrategy(
        LiveStrategyConfig(fuses=SafetyFuses(max_retire_fraction=1.0, max_prune_fraction=1.0, min_stable_geometry_confidence_for_never_touch=1.1))
    )
    n = 10
    # Phase 1: reach STABLE.
    _run_steps(
        strat, n, range(0, 40),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.95,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.STABLE)).all()

    # Phase 2: continue under ordinary (good, non-adversarial) conditions
    # for many more steps -- STABLE Gaussians must never appear in
    # retire_indices, and the lifecycle should stay STABLE.
    vis = torch.ones(n, dtype=torch.bool)
    for it in range(40, 200):
        strat.observe(
            n_gaussians=n, iteration=it, topology_version=0, visible_mask=vis,
            contribution=torch.full((n,), 0.85), rgb_residual=torch.full((n,), 0.02),
            depth_residual=torch.full((n,), 0.02), free_space_conflict=torch.full((n,), 0.05),
            geometry_confidence=torch.full((n,), 0.9),
        )
        plan, _report = strat.propose(iteration=it, topology_version=0)
        assert plan.retire_indices.numel() == 0, (
            f"iteration {it}: strategy proposed retire under ordinary conditions "
            f"(indices={plan.retire_indices.tolist()})"
        )
        assert plan.prune_indices.numel() == 0
        # Discriminator: with fuses effectively disabled above, an empty
        # retire/prune set here can only mean the candidacy guard itself
        # excluded these Gaussians -- not that a fuse clamped a non-empty
        # proposal down to zero (which would make this assertion pass
        # vacuously regardless of whether the guard does anything).
        assert "safety_fuse:max_retire_fraction" not in plan.reasons.safety_fuse_triggers
        assert "safety_fuse:max_prune_fraction" not in plan.reasons.safety_fuse_triggers

    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.STABLE)).all()


def test_deliverable_5_stable_excluded_even_if_scoring_would_favor_it():
    """Structural check (not scenario-dependent): the retire-candidate mask
    built in ``mutation_plan.build_mutation_plan`` unconditionally excludes
    STABLE, regardless of what ``scoring.retire_score`` would compute --
    verified here by monkeypatching retire_score to always return a huge
    positive value (which would normally guarantee inclusion) and
    confirming STABLE Gaussians are still absent from retire_indices.

    ``max_retire_fraction`` is set to ``1.0`` (fuses effectively disabled)
    so this is a genuine discriminator, not a vacuous one: at the default
    ``0.05`` fraction and this test's small N, ``int(0.05 * n)`` floors to
    ``0`` regardless of whether the SUSPECT-only candidacy guard does
    anything at all, which would make ``retire_indices.numel() == 0`` true
    either way -- a real gap caught in this PR's own self-review (see
    ``docs/architecture/2026-08-09-live-gaussian-strategy-observe-only.md``'s
    "Self-review note"). The ``safety_fuse_triggers`` assertion below
    additionally confirms the empty result did not come from a fuse clamp.
    """
    import eligsir.optimization.gaussian.strategies.live.mutation_plan as mp

    strat = EliGSiRLiveGaussianStrategy(
        LiveStrategyConfig(fuses=SafetyFuses(max_retire_fraction=1.0, max_prune_fraction=1.0, min_stable_geometry_confidence_for_never_touch=1.1))
    )
    n = 6
    _run_steps(
        strat, n, range(0, 40),
        contribution=0.9, rgb_residual=0.01, depth_residual=0.01,
        free_space_conflict=0.0, geometry_confidence=0.95,
    )
    assert (strat.state.geometry_lifecycle == state_to_code(GaussianLifecycleState.STABLE)).all()

    original = mp.retire_score
    try:
        mp.retire_score = lambda **kwargs: torch.full_like(kwargs["conflict_ema"], 1e6)
        plan, _report = strat.propose(iteration=41, topology_version=0)
        assert plan.retire_indices.numel() == 0
        assert "safety_fuse:max_retire_fraction" not in plan.reasons.safety_fuse_triggers
    finally:
        mp.retire_score = original
