"""Typed contracts for the live Gaussian population strategy (Wave 3, PR
W3-09, plan section 14.4/14.5).

This module is types only, mirroring the convention already established by
``eligsir/optimization/gaussian/contracts.py`` (W3-02) and
``eligsir/mapping/evidence/contracts.py`` (W3-05): no strategy logic
lives here, only the dataclasses/enums the rest of the
``strategies/live/`` package builds and consumes.

**Observe-only by default, mutation is opt-in per-phase (W3-10).**
``PopulationMutationPlan.observe_only`` defaults to ``True``; every plan
``live/mutation_plan.py::build_mutation_plan`` produces is still
``observe_only=True`` (unchanged since W3-09 -- this module's proposal
logic never opts into mutation itself). ``observe_only=False`` is now
constructible (as of W3-10, plan section 15.2/15.4), but only from
``live/backend.py``'s ``AlignedBufferRegistry.apply()`` -- see that
module's docstring for the per-phase (A-E) gating that decides what, if
anything, actually gets mutated. This is the single, visible crossing
point W3-09's docstring predicted this PR would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

PLAN_CONTRACT_VERSION = "1.0"


class GaussianLifecycleState(Enum):
    """Per-Gaussian lifecycle state (plan section 14.6).

    Not an ``IntEnum`` — ordering is not meaningful, callers must not rely
    on numeric comparison (matches the existing convention in
    ``eligsir.mapping.geometry.ray_semantics.RayEvidenceKind``).
    """

    PROVISIONAL = auto()
    ACTIVE = auto()
    SUSPECT = auto()
    STABLE = auto()
    RETIRED = auto()
    RESEEDED = auto()


# Allowed transitions per plan section 14.6, used by lifecycle.py to reject
# any transition this contract does not explicitly list.
ALLOWED_LIFECYCLE_TRANSITIONS: dict[GaussianLifecycleState, frozenset[GaussianLifecycleState]] = {
    GaussianLifecycleState.PROVISIONAL: frozenset(
        {GaussianLifecycleState.ACTIVE, GaussianLifecycleState.SUSPECT}
    ),
    GaussianLifecycleState.ACTIVE: frozenset(
        {GaussianLifecycleState.STABLE, GaussianLifecycleState.SUSPECT}
    ),
    GaussianLifecycleState.STABLE: frozenset({GaussianLifecycleState.SUSPECT}),
    GaussianLifecycleState.SUSPECT: frozenset(
        {GaussianLifecycleState.ACTIVE, GaussianLifecycleState.RETIRED}
    ),
    GaussianLifecycleState.RETIRED: frozenset({GaussianLifecycleState.RESEEDED}),
    GaussianLifecycleState.RESEEDED: frozenset(),
}


@dataclass(frozen=True)
class SurfaceSeedRequest:
    """A proposed reseed/growth site (plan section 14.4).

    ``cell_id`` keys into the same surface-cell space as
    ``LivePopulationState.source_surface_cell`` (see ``live/cell_budget.py``
    for the budget that admits/rejects these). ``position``/``normal`` are
    optional ``[3]`` CPU-or-device tensors describing the proposed site in
    world space; ``utility`` is the strategy's own score (higher = more
    valuable), used purely for cell-budget ranking, never for a direct
    tensor mutation in this PR.
    """

    cell_id: int
    utility: float
    reason: str
    position: Optional["torch.Tensor"] = None
    normal: Optional["torch.Tensor"] = None
    source_gaussian_index: Optional[int] = None


@dataclass(frozen=True)
class LifecycleUpdateBatch:
    """Proposed per-Gaussian lifecycle transitions (plan section 14.4).

    Geometry and appearance maturity are tracked as two independent tracks
    (plan section 14.6, "Geometry and appearance maturity are separate"),
    so this carries two parallel index/old/new/reason groups rather than
    one merged array — merging them would make ``old_geometry_state``/
    ``new_geometry_state`` ambiguous for an index that only transitioned on
    the appearance track (or vice versa).

    Each ``*_indices`` indexes into the *current* topology (same convention
    as ``GaussianEvidenceHistory`` — only valid for the topology_version
    this batch was built under). This is a *proposal*: applying it (writing
    the new state into a live ``LivePopulationState``) is the caller's
    decision — see ``live/state.py``'s ``apply_lifecycle_batch`` — not
    something this dataclass does itself.
    """

    geometry_indices: "torch.Tensor"
    geometry_old_state: "torch.Tensor"
    geometry_new_state: "torch.Tensor"
    geometry_reasons: tuple[str, ...]
    appearance_indices: "torch.Tensor"
    appearance_old_state: "torch.Tensor"
    appearance_new_state: "torch.Tensor"
    appearance_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        n_geo = int(self.geometry_indices.shape[0])
        if len(self.geometry_reasons) != n_geo:
            raise ValueError(
                f"LifecycleUpdateBatch has {n_geo} geometry indices but "
                f"{len(self.geometry_reasons)} geometry reasons — every proposed "
                "lifecycle transition must have a reason (deliverable 4)."
            )
        n_app = int(self.appearance_indices.shape[0])
        if len(self.appearance_reasons) != n_app:
            raise ValueError(
                f"LifecycleUpdateBatch has {n_app} appearance indices but "
                f"{len(self.appearance_reasons)} appearance reasons — every proposed "
                "lifecycle transition must have a reason (deliverable 4)."
            )

    def total_transitions(self) -> int:
        return int(self.geometry_indices.shape[0]) + int(self.appearance_indices.shape[0])


@dataclass
class MutationReasonSummary:
    """Reason breakdown for every proposed action (plan section 14.7,
    deliverable 4: "every proposed action has a reason").

    Deliberately keeps ``safety_fuse_triggers`` in its own dict, distinct
    from the scoring-derived reason dicts above it — this is the structural
    half of deliverable 6 ("safety fuses remain separate from normal
    strategy scoring"): a fuse-driven clamp is always recorded under a
    ``safety_fuse:*``-prefixed reason in this dict, never folded into
    ``retire_reasons``/``prune_reasons``/etc., so a reader (or a test) can
    tell "the strategy scored this as worth retiring" apart from "a fuse
    vetoed/capped what scoring wanted to do" just by which dict a count
    landed in.
    """

    retire_reasons: dict[str, int] = field(default_factory=dict)
    prune_reasons: dict[str, int] = field(default_factory=dict)
    reseed_reasons: dict[str, int] = field(default_factory=dict)
    growth_reasons: dict[str, int] = field(default_factory=dict)
    lifecycle_reasons: dict[str, int] = field(default_factory=dict)
    safety_fuse_triggers: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "retire_reasons": dict(self.retire_reasons),
            "prune_reasons": dict(self.prune_reasons),
            "reseed_reasons": dict(self.reseed_reasons),
            "growth_reasons": dict(self.growth_reasons),
            "lifecycle_reasons": dict(self.lifecycle_reasons),
            "safety_fuse_triggers": dict(self.safety_fuse_triggers),
        }


@dataclass
class PopulationMutationPlan:
    """The strategy's proposed mutation (plan section 14.4, reproduced
    verbatim field-for-field). **Does not resize or replace model tensors**
    — see module docstring.
    """

    retire_indices: "torch.Tensor"
    prune_indices: "torch.Tensor"
    reseed_requests: tuple[SurfaceSeedRequest, ...]
    growth_requests: tuple[SurfaceSeedRequest, ...]
    lifecycle_updates: LifecycleUpdateBatch
    reasons: MutationReasonSummary
    version: str = PLAN_CONTRACT_VERSION
    observe_only: bool = True
    iteration: int = -1
    topology_version: int = -1

    def __post_init__(self) -> None:
        # W3-10: the single, visible place the W3-09 hard invariant
        # ("observe_only can never be False") is crossed, exactly as that
        # PR's docstring predicted. Construction with observe_only=False is
        # now permitted, but ONLY from
        # ``live/backend.py``'s ``AlignedBufferRegistry.apply()`` -- the one
        # real-mutation call site this PR adds -- never from
        # ``live/mutation_plan.py::build_mutation_plan`` (which still always
        # builds an observe_only=True plan; see that module's docstring,
        # unchanged by this PR). ``live/strategy.py``'s
        # ``EliGSiRLiveGaussianStrategy.propose()`` still only ever
        # returns an observe_only=True plan -- applying it for real is a
        # deliberate, separate, opt-in step the caller must take by handing
        # that plan to ``AlignedBufferRegistry.apply()`` with explicit
        # per-phase flags, not something this dataclass does implicitly.
        if self.version != PLAN_CONTRACT_VERSION:
            raise ValueError(
                f"Unknown PopulationMutationPlan version {self.version!r}; "
                f"this build only knows {PLAN_CONTRACT_VERSION!r}."
            )

    def total_proposed_actions(self) -> int:
        retire_n = int(self.retire_indices.shape[0]) if hasattr(self.retire_indices, "shape") else len(self.retire_indices)
        prune_n = int(self.prune_indices.shape[0]) if hasattr(self.prune_indices, "shape") else len(self.prune_indices)
        return retire_n + prune_n + len(self.reseed_requests) + len(self.growth_requests)


@dataclass
class DryRunReport:
    """Observe-only telemetry (plan section 14.7): proposed counts, overlap
    with the currently-active MCMC strategy's own (would-be) decisions,
    predicted affected cells, expected contribution change, and safety-fuse
    trigger counts. Never consumed by any mutation path -- purely a logging
    artifact, e.g. for ``tb_writer`` scalars or an offline JSON dump (see
    ``scripts/w3_09_live_strategy_dry_run.py``).
    """

    proposed_retire_count: int
    proposed_prune_count: int
    proposed_reseed_count: int
    proposed_growth_count: int
    reasons: MutationReasonSummary
    overlap_with_current_mcmc: dict
    predicted_affected_cells: int
    expected_contribution_change: float
    safety_fuse_trigger_counts: dict

    def as_dict(self) -> dict:
        return {
            "proposed_retire_count": self.proposed_retire_count,
            "proposed_prune_count": self.proposed_prune_count,
            "proposed_reseed_count": self.proposed_reseed_count,
            "proposed_growth_count": self.proposed_growth_count,
            "reasons": self.reasons.as_dict(),
            "overlap_with_current_mcmc": dict(self.overlap_with_current_mcmc),
            "predicted_affected_cells": self.predicted_affected_cells,
            "expected_contribution_change": self.expected_contribution_change,
            "safety_fuse_trigger_counts": dict(self.safety_fuse_trigger_counts),
        }
