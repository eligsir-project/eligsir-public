"""CPU/reference birth-target contract for issue #1365 migration scaffolding.

Canonical architecture:
``docs/architecture/2026-08-31-probabilistic-gaussian-belief-and-population-control.md``.
Pass-5/9 execution amendments:
``docs/architecture/2026-08-31-probabilistic-gaussian-control-pass5-pass9-audit.md``.

Source-depth insertion and MCMC growth still contain historically independent
Gaussian-creation semantics.  The final architecture does **not** introduce a
standalone ``GaussianBirthOwner`` beside the rest of population control.
Instead, depth/appearance/reconstruction mechanisms produce evidence and a
single ``GaussianPopulationPolicy`` decides ordinary
``NONE/CORRECT/EXPLORE/CREATE/RELOCATE/RETIRE`` actions before one
``GaussianMutationArbiter`` commit.

**This module is pure scaffolding and changes zero production behaviour.**
Nothing in production constructs or consumes these types yet.  They preserve a
small, serializable vocabulary for migration tests, reporting and compatibility
adapters while a real producer/consumer is built under #1365.

These objects are deliberately CPU-side and torch-free.  That is also why they
are **not the production hot-path contract**.  The live implementation target is
a bounded device-resident tensor batch (conceptually
``GaussianPopulationTargetBatch``) with continuous confidence/uncertainty,
valid masks and a version/snapshot token.  No production population decision
should D2H candidate tensors merely to construct the dataclasses below.

Likewise, the categorical ``GeometryConfidence`` and birth-only ``BirthAction``
below are compatibility/debug projections, not the canonical scientific belief
or final action space.  See the canonical architecture/audit for the reversible
probabilistic state and full population-policy semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple


class ProposalKind(str, Enum):
    """Serializable/debug reason that a target was proposed.

    Production policy should consume continuous evidence and may expose these
    labels only as a projection for telemetry/compatibility.
    """

    FRONTIER_SURFACE = "frontier_surface"
    STRUCTURAL_ANCHOR = "structural_anchor"
    DETAIL_DENSIFICATION = "detail_densification"
    GEOMETRY_CORRECTION = "geometry_correction"
    MODEL_SURFACE_EXPLORATION = "model_surface_exploration"


class GeometryConfidence(str, Enum):
    """Legacy/serialization projection of target geometry confidence.

    The final #1365 policy represents geometry with continuous confidence and
    uncertainty.  These categories remain useful for reports and migration
    tests but must not become a second hard-gating truth source.

    ``GEOMETRY_UNKNOWN`` must never by itself justify creating arbitrary 3-D
    topology; unresolved demand should request/await defensible geometry.
    """

    SURFACE_CONFIRMED = "surface_confirmed"
    SURFACE_PROBABLE = "surface_probable"
    MODEL_SURFACE = "model_surface"
    GEOMETRY_UNKNOWN = "geometry_unknown"


class BirthAction(str, Enum):
    """Compatibility subset of historical birth-owner decisions.

    The final population-policy action space also contains ``EXPLORE`` and
    ``RETIRE``.  Do not extend production control by making this CPU enum the
    hot-path authority; the device population policy owns the real decision.
    """

    NONE = "none"
    CORRECT = "correct"
    CREATE = "create"
    RELOCATE = "relocate"


@dataclass(frozen=True)
class GaussianBirthTarget:
    """Serializable candidate surface site for compatibility/migration use.

    Geometry fields describe where valid surface may exist; reconstruction
    fields describe why representational capacity might be useful there.
    Constructing this object never modifies topology.

    The production implementation should keep equivalent evidence in a bounded
    device tensor batch and materialize this object only for cold/debug/report
    consumers when useful.
    """

    target_xyz: Tuple[float, float, float]
    proposal_kind: ProposalKind
    geometry_confidence: GeometryConfidence

    target_normal: Optional[Tuple[float, float, float]] = None
    geometry_covariance: Optional[Tuple[float, ...]] = None
    anchor_id: Optional[str] = None

    rgb_residual: Optional[float] = None
    depth_residual: Optional[float] = None
    gradient_utility: Optional[float] = None

    coverage_debt: Optional[float] = None
    spatial_region_debt: Optional[float] = None
    local_gaussian_density: Optional[float] = None
    current_support: Optional[int] = None

    supporting_sources: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.target_xyz) != 3:
            raise ValueError(
                f"target_xyz must be a 3-tuple, got {self.target_xyz!r}"
            )
        if self.target_normal is not None and len(self.target_normal) != 3:
            raise ValueError(
                f"target_normal must be a 3-tuple, got {self.target_normal!r}"
            )


@dataclass(frozen=True)
class GaussianBirthProposal:
    """Serializable target + optional donor/parent association.

    ``parent_uid`` is intentionally separate from ``target`` because a donor
    may provide appearance/lineage priors without defining the new target XYZ.
    The final GPU policy represents this association in its bounded tensor plan.
    """

    target: GaussianBirthTarget
    parent_uid: Optional[int] = None
    score: Optional[float] = None
    supporting_sources: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GaussianBirthDecision:
    """Compatibility/reference verdict for migration tests and telemetry.

    No standalone ``GaussianBirthOwner`` is the final architecture.  Future
    production decisions belong to ``GaussianPopulationPolicy`` and should be
    emitted as a versioned device-resident population plan before arbiter
    commit.  This class is retained so existing scaffolding/tests remain useful.
    """

    action: BirthAction
    proposal: GaussianBirthProposal
    reason: Optional[str] = None
    donor_uid: Optional[int] = None


def birth_targets_from_kind(
    targets: Sequence[GaussianBirthTarget], kind: ProposalKind
) -> Tuple[GaussianBirthTarget, ...]:
    """Filter serialized/reference targets by one debug proposal kind."""

    return tuple(t for t in targets if t.proposal_kind is kind)
