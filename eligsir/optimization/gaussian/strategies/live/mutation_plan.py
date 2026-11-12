"""Mutation-plan assembly (Wave 3, PR W3-09, plan section 14.4/14.8
Subagent D).

Orchestrates, in order:

1. lifecycle transitions (``lifecycle.transition_lifecycle_track``, called
   twice: once for the geometry track, once for appearance);
2. raw candidate scoring (``scoring.py``);
3. cell-budget admission for reseed/growth candidates (``cell_budget.py``);
4. **safety fuses** (``SafetyFuses`` below) — applied last, as a clamp over
   whatever steps 1-3 produced.

Deliverable 6 ("safety fuses remain separate from normal strategy
scoring"): ``SafetyFuses`` is a separate dataclass with its own
``apply()`` method that only ever *removes* candidates that scoring
already proposed (or drops indices from a lifecycle-transition batch) and
records why under a distinct ``safety_fuse:*``-prefixed reason namespace
(see ``contracts.MutationReasonSummary.safety_fuse_triggers``). No fuse
threshold is read by ``scoring.py``, and no scoring weight is read by
``SafetyFuses`` — this is checkable by grep
(``tests/test_w3_09_live_strategy_scoring_and_budget.py::test_safety_fuses_are_a_separate_module_from_scoring``).

Deliverable 5 ("stable Gaussians are not proposed for global teleportation
under ordinary conditions") is enforced by restricting retire/prune
candidacy to the ``SUSPECT`` lifecycle state only (``candidate_mask =
is_suspect``, below) — a Gaussian never becomes a retire candidate while it
is ``STABLE``, regardless of what ``scoring.retire_score`` would compute
for it (see the monkeypatch test
``test_deliverable_5_stable_excluded_even_if_scoring_would_favor_it``,
which forces the score arbitrarily high and confirms STABLE Gaussians are
still absent — because they were never in the candidate set to begin
with). The *only* way a Gaussian can ever reach the candidate set after
being STABLE is a prior ``STABLE -> SUSPECT`` transition, which
``lifecycle.py`` gates behind strong, *sustained* conflict evidence
(``stable_suspect_persistence_steps``, stricter than the ordinary
suspect threshold) — i.e. not "ordinary conditions" by this module's own
definition of the term. This is a real, checkable exclusion; it is not an
exclusion of STABLE Gaussians *for all time*, only for as long as they
remain STABLE, which is the scope plan section 14.6 actually describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from eligsir.optimization.gaussian.strategies.contracts import (
    GaussianLifecycleState,
    LifecycleUpdateBatch,
    MutationReasonSummary,
    PopulationMutationPlan,
    SurfaceSeedRequest,
)
from eligsir.optimization.gaussian.strategies.live.cell_budget import CellBudgetTracker
from eligsir.optimization.gaussian.strategies.live.lifecycle import (
    LifecycleThresholds,
    transition_lifecycle_track,
)
from eligsir.optimization.gaussian.strategies.live.scoring import prune_score, retire_score
from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState, state_to_code


@dataclass(frozen=True)
class SafetyFuses:
    """Hard caps applied last, independent of scoring (deliverable 6).

    Every threshold here is a *fraction of current population* or an
    absolute count, never a quality/utility score — that's what keeps this
    module structurally distinct from ``scoring.py``.
    """

    max_retire_fraction: float = 0.05
    max_prune_fraction: float = 0.02
    max_reseed_requests: int = 500
    max_growth_requests: int = 500
    min_stable_geometry_confidence_for_never_touch: float = 0.95
    """Any Gaussian at/above this geometry_confidence is exempt from retire
    even if it slipped through scoring (belt-and-braces on top of the
    unconditional STABLE-lifecycle exclusion in ``build_mutation_plan``)."""

    def apply(
        self,
        *,
        retire_indices: torch.Tensor,
        retire_scores: torch.Tensor,
        prune_indices: torch.Tensor,
        prune_scores: torch.Tensor,
        reseed_requests: tuple[SurfaceSeedRequest, ...],
        growth_requests: tuple[SurfaceSeedRequest, ...],
        geometry_confidence: torch.Tensor,
        n_live: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[SurfaceSeedRequest, ...], tuple[SurfaceSeedRequest, ...], dict[str, int]]:
        triggers: dict[str, int] = {}

        if retire_indices.numel() > 0:
            never_touch = geometry_confidence[retire_indices] >= self.min_stable_geometry_confidence_for_never_touch
            if bool(never_touch.any()):
                triggers["safety_fuse:never_touch_high_confidence"] = int(never_touch.sum().item())
                keep = ~never_touch
                retire_indices = retire_indices[keep]
                retire_scores = retire_scores[keep]

        max_retire = int(self.max_retire_fraction * max(n_live, 1))
        if retire_indices.numel() > max_retire:
            # stable=True: deterministic tie-break -- equal-score entries
            # keep their INCOMING relative order (this is what stable=True
            # actually guarantees; it is not itself an index-ascending
            # sort). That incoming order is ascending physical/logical
            # index here because every upstream step that produced
            # retire_indices/retire_scores (the never_touch filter just
            # above, and ultimately build_mutation_plan's own
            # torch.nonzero()-based candidate selection) is order-
            # preserving -- if a future refactor introduces a step that
            # reorders retire_indices before this fuse runs, this
            # tie-break would silently stop being index-ascending. Without
            # stable=True at all, equal-score ties have backend/kernel-
            # dependent ordering, which would make the retained set
            # non-reproducible across ROCm/CUDA or even across runs.
            order = torch.argsort(retire_scores, descending=True, stable=True)
            dropped = int(retire_indices.numel() - max_retire)
            triggers["safety_fuse:max_retire_fraction"] = dropped
            keep_idx = order[:max_retire]
            retire_indices = retire_indices[keep_idx]

        max_prune = int(self.max_prune_fraction * max(n_live, 1))
        if prune_indices.numel() > max_prune:
            # stable=True: same deterministic-tie-break rationale as the
            # retire_indices truncation above.
            order = torch.argsort(prune_scores, descending=True, stable=True)
            dropped = int(prune_indices.numel() - max_prune)
            triggers["safety_fuse:max_prune_fraction"] = dropped
            keep_idx = order[:max_prune]
            prune_indices = prune_indices[keep_idx]

        if len(reseed_requests) > self.max_reseed_requests:
            triggers["safety_fuse:max_reseed_requests"] = len(reseed_requests) - self.max_reseed_requests
            reseed_requests = tuple(
                sorted(reseed_requests, key=lambda r: r.utility, reverse=True)[: self.max_reseed_requests]
            )
        if len(growth_requests) > self.max_growth_requests:
            triggers["safety_fuse:max_growth_requests"] = len(growth_requests) - self.max_growth_requests
            growth_requests = tuple(
                sorted(growth_requests, key=lambda r: r.utility, reverse=True)[: self.max_growth_requests]
            )

        return retire_indices, prune_indices, reseed_requests, growth_requests, triggers


def _merge_tracks(geo_result, app_result) -> tuple[LifecycleUpdateBatch, dict[str, int]]:
    batch = LifecycleUpdateBatch(
        geometry_indices=geo_result.indices,
        geometry_old_state=geo_result.old_state,
        geometry_new_state=geo_result.new_state,
        geometry_reasons=geo_result.reasons,
        appearance_indices=app_result.indices,
        appearance_old_state=app_result.old_state,
        appearance_new_state=app_result.new_state,
        appearance_reasons=app_result.reasons,
    )
    counts: dict[str, int] = {}
    for r in geo_result.reasons + app_result.reasons:
        counts[r] = counts.get(r, 0) + 1
    return batch, counts


def build_mutation_plan(
    *,
    state: LivePopulationState,
    thresholds: LifecycleThresholds,
    fuses: SafetyFuses,
    residual_ema_for_geometry: Optional[torch.Tensor] = None,
    iteration: int,
    topology_version: int,
    reseed_candidates: tuple[SurfaceSeedRequest, ...] = (),
    growth_candidates: tuple[SurfaceSeedRequest, ...] = (),
    cell_budget: Optional[CellBudgetTracker] = None,
) -> tuple[PopulationMutationPlan, dict]:
    """Assemble a :class:`PopulationMutationPlan` from the current
    ``state`` + candidate reseed/growth requests, plus a ``streaks`` dict
    (``geometry_suspect_streak``/``geometry_recover_streak``/
    ``geometry_stable_streak``) the caller may persist via
    ``LivePopulationState.persist_streaks``.

    **Never writes to ``state`` itself** — this function only reads
    ``state`` and returns proposals; ``live/strategy.py``'s ``propose()``
    is the only place that calls ``state.apply_lifecycle_batch``/
    ``persist_streaks``, and it does so explicitly and separately from
    this call, so a caller that only wants a dry-run preview (no state
    mutation of the strategy's own bookkeeping) can call this function
    directly without side effects.

    Appearance-track streak counters currently share the same underlying
    ``state._suspect_streak``/``_recover_streak``/``_stable_streak``
    tensors as the geometry track (a documented simplification -- proper
    per-track independent streaks would need three more per-Gaussian
    buffers on ``LivePopulationState`` beyond the "at minimum" list in
    plan section 14.5; not implemented in this PR).
    """
    state.require_valid(topology_version)
    cell_budget = cell_budget or CellBudgetTracker()
    residual_ema = residual_ema_for_geometry if residual_ema_for_geometry is not None else state.depth_residual_ema

    geo_result, geo_suspect, geo_recover, geo_stable = transition_lifecycle_track(
        current_state_codes=state.geometry_lifecycle.to(torch.int64),
        support_count=state.support_count,
        geometry_confidence=state.geometry_confidence,
        conflict_ema=state.free_space_conflict_ema,
        contribution_ema=state.contribution_ema,
        residual_ema=residual_ema,
        suspect_streak=state._suspect_streak,
        recover_streak=state._recover_streak,
        stable_streak=state._stable_streak,
        thresholds=thresholds,
    )
    app_result, _app_suspect, _app_recover, _app_stable = transition_lifecycle_track(
        current_state_codes=state.appearance_lifecycle.to(torch.int64),
        support_count=state.support_count,
        geometry_confidence=state.geometry_confidence,
        conflict_ema=state.free_space_conflict_ema,
        contribution_ema=state.contribution_ema,
        residual_ema=state.rgb_residual_ema,
        suspect_streak=state._suspect_streak,
        recover_streak=state._recover_streak,
        stable_streak=state._stable_streak,
        thresholds=thresholds,
    )
    streaks = {
        "geometry_suspect_streak": geo_suspect,
        "geometry_recover_streak": geo_recover,
        "geometry_stable_streak": geo_stable,
    }
    lifecycle_updates, lifecycle_reason_counts = _merge_tracks(geo_result, app_result)

    # --- retire/prune candidates, restricted to SUSPECT (never STABLE — deliverable 5) ---
    device = state.geometry_lifecycle.device
    suspect_code = state_to_code(GaussianLifecycleState.SUSPECT)
    retired_code = state_to_code(GaussianLifecycleState.RETIRED)

    is_suspect = state.geometry_lifecycle.to(torch.int64) == suspect_code
    # Deliverable 5: retire/prune candidacy is restricted to SUSPECT only.
    # `geometry_lifecycle` is a single mutually-exclusive int8 code, so this
    # already excludes STABLE (and every other state) by construction.
    candidate_mask = is_suspect
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()

    if candidate_indices.numel() > 0:
        r_scores = retire_score(
            conflict_ema=state.free_space_conflict_ema[candidate_indices],
            contribution_ema=state.contribution_ema[candidate_indices],
            support_count=state.support_count[candidate_indices],
        )
        keep_retire = r_scores > 0.0
        retire_indices = candidate_indices[keep_retire]
        retire_scores = r_scores[keep_retire]
        # W3-55: sort by descending retire_score (highest-priority first)
        # before handing off to the safety-fuse pass below.
        # PopulationMutationPlan is a frozen, spec-locked contract (plan
        # section 14.4, "reproduced verbatim field-for-field") that carries
        # retire_indices onward but not retire_scores -- so this is the one
        # place score information is still available to establish priority
        # order. The safety-fuse pass's own max_retire_fraction truncation
        # already re-derives this order via its own argsort when it fires;
        # this sort matters for the (common) case where that fuse does NOT
        # truncate, so retire_indices passes through the fuse pass unchanged
        # and reaches AlignedBufferRegistry's apply() (live/backend.py)
        # still in descending-priority order. That downstream truncation
        # (plan.retire_indices[:n_to_apply], gated by
        # max_unreclaimed_retired) has no score tensor to sort by itself --
        # relying on retire_indices already arriving in priority order is
        # what keeps a partial retirement (once max_unreclaimed_retired > 0
        # is ever configured) keeping the highest-priority candidates
        # instead of the lowest-physical-index ones.
        # stable=True: deterministic tie-break -- equal-score entries keep
        # their incoming (ascending physical/logical index) order instead of
        # a backend/kernel-dependent permutation. Primary key retire_score
        # descending, secondary key index ascending.
        priority_order = torch.argsort(retire_scores, descending=True, stable=True)
        retire_indices = retire_indices[priority_order]
        retire_scores = retire_scores[priority_order]
    else:
        retire_indices = torch.empty(0, dtype=torch.int64, device=device)
        retire_scores = torch.empty(0, dtype=torch.float32, device=device)

    is_retired = state.geometry_lifecycle.to(torch.int64) == retired_code
    retired_indices = torch.nonzero(is_retired, as_tuple=False).flatten()
    if retired_indices.numel() > 0:
        p_scores = prune_score(
            geometry_confidence=state.geometry_confidence[retired_indices],
            conflict_ema=state.free_space_conflict_ema[retired_indices],
        )
        keep_prune = p_scores > 0.5
        prune_indices = retired_indices[keep_prune]
        prune_scores = p_scores[keep_prune]
    else:
        prune_indices = torch.empty(0, dtype=torch.int64, device=device)
        prune_scores = torch.empty(0, dtype=torch.float32, device=device)

    admitted_reseed, reseed_rejections = cell_budget.admit(reseed_candidates)
    admitted_growth, growth_rejections = cell_budget.admit(growth_candidates)

    retire_indices, prune_indices, admitted_reseed, admitted_growth, fuse_triggers = fuses.apply(
        retire_indices=retire_indices,
        retire_scores=retire_scores,
        prune_indices=prune_indices,
        prune_scores=prune_scores,
        reseed_requests=admitted_reseed,
        growth_requests=admitted_growth,
        geometry_confidence=state.geometry_confidence,
        n_live=state.n,
    )

    def _count_reasons(requests: tuple[SurfaceSeedRequest, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in requests:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return counts

    reasons = MutationReasonSummary(
        retire_reasons={"free_space_conflict_or_low_contribution": int(retire_indices.numel())} if retire_indices.numel() else {},
        prune_reasons={"retired_high_conflict_low_confidence": int(prune_indices.numel())} if prune_indices.numel() else {},
        reseed_reasons=_count_reasons(admitted_reseed),
        growth_reasons=_count_reasons(admitted_growth),
        lifecycle_reasons=lifecycle_reason_counts,
        safety_fuse_triggers={**fuse_triggers, **reseed_rejections, **growth_rejections},
    )

    plan = PopulationMutationPlan(
        retire_indices=retire_indices,
        prune_indices=prune_indices,
        reseed_requests=admitted_reseed,
        growth_requests=admitted_growth,
        lifecycle_updates=lifecycle_updates,
        reasons=reasons,
        iteration=iteration,
        topology_version=topology_version,
    )
    return plan, streaks
