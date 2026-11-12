"""Immutable CPU control-plane contracts for the canonical live runtime.

These records deliberately carry identities, scalar metadata, completed compact
telemetry and typed budgets only.  Pixel buffers and live device tensors cross
the separate transfer boundary and must never be added here.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import FrozenInstanceError, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

import numpy as np

from .runtime import SourceIdentity, TrackingObservation
from .types import Pose


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


Scalar = bool | int | float | str | None


class _ImmutableSnapshotPose(Pose):
    """Snapshot-owned pose whose matrix cannot be replaced or mutated."""

    def __init__(self, matrix: np.ndarray) -> None:
        owned = np.array(matrix, dtype=np.float32, copy=True)
        if owned.shape != (4, 4):
            raise ValueError(f"Pose matrix must be (4, 4), got {owned.shape}")
        owned.setflags(write=False)
        object.__setattr__(self, "matrix", owned)
        object.__setattr__(self, "_snapshot_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_snapshot_sealed", False):
            raise FrozenInstanceError(f"cannot assign to snapshot pose field {name!r}")
        object.__setattr__(self, name, value)


def _freeze_diagnostic(value: Any) -> Any:
    """Recursively own diagnostic evidence or reject unknown mutable types."""

    if isinstance(value, np.ndarray):
        owned = np.array(value, copy=True)
        owned.setflags(write=False)
        return owned
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_diagnostic(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_diagnostic(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_diagnostic(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str, bytes, type(None), Enum)):
        return value
    raise TypeError(
        f"snapshot tracking diagnostic {type(value).__name__} is not immutable"
    )


def _owned_tracking_evidence(tracking: TrackingObservation) -> TrackingObservation:
    pose = None if tracking.pose is None else _ImmutableSnapshotPose(tracking.pose.matrix)
    return TrackingObservation(
        source=tracking.source,
        pose=pose,
        status=tracking.status,
        provider=tracking.provider,
        confidence=tracking.confidence,
        covariance=tracking.covariance,
        diagnostics={
            key: _freeze_diagnostic(value)
            for key, value in tracking.diagnostics.items()
        },
    )


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    """One named, immutable CPU-visible evidence value."""

    name: str
    value: Scalar
    version: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evidence name must be non-empty")
        if not isinstance(self.value, (bool, int, float, str, type(None))):
            raise TypeError("evidence values must be scalar")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("floating evidence must be finite")
        object.__setattr__(self, "version", _integer(self.version, "version"))


@dataclass(frozen=True, slots=True)
class ControlProposal:
    """Evidence-only proposal from a legacy controller or policy component."""

    provider: str
    actuator: str
    value: Scalar
    reason: str

    def __post_init__(self) -> None:
        if not self.provider or not self.actuator or not self.reason:
            raise ValueError("proposal provider, actuator and reason must be non-empty")
        if not isinstance(self.value, (bool, int, float, str, type(None))):
            raise TypeError("proposal values must be scalar")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("floating proposal values must be finite")


class FreshnessMode(str, Enum):
    HEALTHY = "healthy"
    CATCH_UP = "catch_up"
    CRITICAL = "critical"
    RECOVERY = "recovery"


class ResolutionTier(str, Enum):
    """Canonical supervision scale selected only by WorkBudgetOwner."""

    NATIVE = "native"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"

    @property
    def downsample_factor(self) -> int:
        return {self.NATIVE: 1, self.HALF: 2, self.QUARTER: 4, self.EIGHTH: 8}[self]


@dataclass(frozen=True, slots=True)
class PreparedTierPolicy:
    """Immutable declared set of resolution tiers materialized before READY.

    Resolved once from the configured resolution-tier policy/controller at
    application initialization. Prepared sources materialize exactly these tiers;
    READY consumers reject any request for a non-declared tier with a clear error.

    No lazy tier construction after READY; all required tiers are materialized
    before a lease becomes READY.
    """

    declared_tiers: frozenset[ResolutionTier]

    def __post_init__(self) -> None:
        if not isinstance(self.declared_tiers, frozenset):
            raise TypeError("declared_tiers must be frozenset[ResolutionTier]")
        if not self.declared_tiers:
            raise ValueError("declared_tiers must not be empty")
        if not all(isinstance(tier, ResolutionTier) for tier in self.declared_tiers):
            raise TypeError("all declared_tiers must be ResolutionTier values")

    @classmethod
    def from_supported_tiers(
        cls, supported_tiers: tuple[ResolutionTier, ...] | frozenset[ResolutionTier]
    ) -> "PreparedTierPolicy":
        """Create a policy from the set of tiers supported by the resolution policy.

        This is the sole creation point; the resolved policy is immutable and
        shared across all prepared sources produced by the application.
        """
        if isinstance(supported_tiers, frozenset):
            declared = supported_tiers
        else:
            declared = frozenset(supported_tiers)
        return cls(declared_tiers=declared)

    def highest(self) -> ResolutionTier:
        """Return the highest-quality (lowest downsample factor) declared tier.

        Used to deterministically select a tier when the policy permits multiple
        tiers (e.g., bootstrap). Under adaptive policy (which includes native/half/quarter/eighth),
        this returns NATIVE (unchanged behavior). Under FIXED quarter, this returns QUARTER.
        """
        # Sort by downsample_factor ascending (lowest = highest quality = native)
        return min(self.declared_tiers, key=lambda tier: tier.downsample_factor)


class MappingTransactionClass(str, Enum):
    """Identity composition of one frozen optimizer transaction."""

    FRESH = "fresh"
    MIXED = "mixed"
    REPLAY_ONLY = "replay_only"
    POST_REFINEMENT = "post_refinement"


@dataclass(frozen=True, slots=True)
class ControlLimits:
    """Hard per-tick limits resolved before policy proposals are considered."""

    max_supervision_slots: int
    max_replay_slots: int
    max_optimizer_updates: int
    max_insertion_seeds: int
    max_mutations: int
    max_transfer_bytes: int
    max_replay_insertion_seeds: int = 0
    max_optimization_transaction_ns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_supervision_slots",
            "max_replay_slots",
            "max_optimizer_updates",
            "max_insertion_seeds",
            "max_mutations",
            "max_transfer_bytes",
            "max_replay_insertion_seeds",
        ):
            minimum = 1 if name == "max_supervision_slots" else 0
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        if self.max_optimization_transaction_ns is not None:
            object.__setattr__(
                self,
                "max_optimization_transaction_ns",
                _integer(
                    self.max_optimization_transaction_ns,
                    "max_optimization_transaction_ns",
                    minimum=1,
                ),
            )


@dataclass(frozen=True, slots=True)
class CompactGpuTelemetry:
    """Latest completed tiny GPU telemetry packet; never a live tensor view."""

    transaction_id: int
    map_content_version: int
    topology_version: int
    parameter_version: int
    facts: tuple[EvidenceFact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in (
            "transaction_id",
            "map_content_version",
            "topology_version",
            "parameter_version",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "facts", tuple(self.facts))
        if not all(isinstance(item, EvidenceFact) for item in self.facts):
            raise TypeError("telemetry facts must be EvidenceFact values")


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    """One coherent, CPU-only view sampled for a control tick."""

    snapshot_id: int
    control_version: int
    source: Optional[SourceIdentity]
    tracking: Optional[TrackingObservation]
    pending_sources: tuple[SourceIdentity, ...]
    newest_frontier: Optional[SourceIdentity]
    freshness_mode: FreshnessMode
    pending_oldest_age_ns: int
    ingress_queue_depth: int
    candidate_queue_depth: int
    transfer_queue_depth: int
    gpu_queue_depth: int
    map_content_version: int
    topology_version: int
    parameter_version: int
    gaussian_count: int
    optimizer_steps_completed: int
    limits: ControlLimits
    supervision_views_completed: int = 0
    control_tick_epoch: int = 0
    supervision_candidates: tuple[SourceIdentity, ...] = field(default_factory=tuple)
    replay_candidates: tuple[SourceIdentity, ...] = field(default_factory=tuple)
    replay_work_modalities: tuple["ModalityEligibility", ...] = field(default_factory=tuple)
    replay_candidate_weights: tuple[float, ...] = field(default_factory=tuple)
    reserved_global_replay_candidate: Optional[SourceIdentity] = None
    latest_gpu_telemetry: Optional[CompactGpuTelemetry] = None
    evidence: tuple[EvidenceFact, ...] = field(default_factory=tuple)
    source_release_rate_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "snapshot_id",
            "control_version",
            "pending_oldest_age_ns",
            "ingress_queue_depth",
            "candidate_queue_depth",
            "transfer_queue_depth",
            "gpu_queue_depth",
            "map_content_version",
            "topology_version",
            "parameter_version",
            "gaussian_count",
            "optimizer_steps_completed",
            "supervision_views_completed",
            "control_tick_epoch",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.limits, ControlLimits):
            raise TypeError("limits must be ControlLimits")
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity or None")
        if self.tracking is not None:
            if not isinstance(self.tracking, TrackingObservation):
                raise TypeError("tracking must be TrackingObservation or None")
            if self.source is not None and self.tracking.source != self.source:
                raise ValueError("tracking.source must match snapshot source")
            object.__setattr__(self, "tracking", _owned_tracking_evidence(self.tracking))

        pending = tuple(self.pending_sources)
        if not all(isinstance(item, SourceIdentity) for item in pending):
            raise TypeError("pending_sources must contain SourceIdentity values")
        if len(set(pending)) != len(pending):
            raise ValueError("pending_sources must be unique")
        if any(a.ordering_key() >= b.ordering_key() for a, b in zip(pending, pending[1:])):
            raise ValueError("pending_sources must be in canonical oldest-to-newest order")
        object.__setattr__(self, "pending_sources", pending)
        if pending:
            if self.newest_frontier != pending[-1]:
                raise ValueError("newest_frontier must equal the final pending source")
        elif self.newest_frontier is not None:
            raise ValueError("newest_frontier must be None when no sources are pending")

        for name in ("supervision_candidates", "replay_candidates"):
            candidates = tuple(getattr(self, name))
            if not all(isinstance(item, SourceIdentity) for item in candidates):
                raise TypeError(f"{name} must contain SourceIdentity values")
            if len(set(candidates)) != len(candidates):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, candidates)
        replay_modalities = tuple(self.replay_work_modalities)
        if len(replay_modalities) != len(self.replay_candidates):
            raise ValueError("replay_work_modalities must align with replay_candidates")
        if not all(isinstance(item, ModalityEligibility) for item in replay_modalities):
            raise TypeError("replay_work_modalities must contain ModalityEligibility values")
        if any(
            not (item.rgb_supervision or item.depth_supervision)
            for item in replay_modalities
        ):
            raise ValueError("replay work modalities require RGB or depth supervision")
        object.__setattr__(self, "replay_work_modalities", replay_modalities)
        replay_weights = tuple(self.replay_candidate_weights)
        if replay_weights and len(replay_weights) != len(self.replay_candidates):
            raise ValueError("replay_candidate_weights must align with replay_candidates")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
            for weight in replay_weights
        ):
            raise ValueError("replay candidate weights must be finite and non-negative")
        object.__setattr__(
            self, "replay_candidate_weights", tuple(float(weight) for weight in replay_weights)
        )
        reserved_global = self.reserved_global_replay_candidate
        if reserved_global is not None:
            if not isinstance(reserved_global, SourceIdentity):
                raise TypeError(
                    "reserved_global_replay_candidate must be SourceIdentity or None"
                )
            if reserved_global not in self.replay_candidates:
                raise ValueError(
                    "reserved_global_replay_candidate must be replay eligible"
                )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not all(isinstance(item, EvidenceFact) for item in self.evidence):
            raise TypeError("evidence must contain EvidenceFact values")
        evidence_names = tuple(item.name for item in self.evidence)
        if len(set(evidence_names)) != len(evidence_names):
            raise ValueError("control snapshot evidence names must be unique")
        telemetry = self.latest_gpu_telemetry
        if telemetry is not None:
            if not isinstance(telemetry, CompactGpuTelemetry):
                raise TypeError("latest_gpu_telemetry must be CompactGpuTelemetry or None")
            for name in ("map_content_version", "topology_version", "parameter_version"):
                if getattr(telemetry, name) > getattr(self, name):
                    raise ValueError("completed GPU telemetry cannot be newer than its snapshot")
        rate_ratio = self.source_release_rate_ratio
        if rate_ratio is not None:
            if isinstance(rate_ratio, bool) or not isinstance(rate_ratio, (int, float)):
                raise TypeError("source_release_rate_ratio must be a float or None")
            if not (math.isfinite(float(rate_ratio)) and float(rate_ratio) >= 0.0):
                raise ValueError("source_release_rate_ratio must be finite and non-negative")


class DispositionKind(str, Enum):
    ADMIT = "admit"
    METADATA_ONLY = "metadata_only"
    COALESCE = "coalesce"
    SKIP_REDUNDANT = "skip_redundant"
    SKIP_CAPACITY = "skip_capacity"
    SKIP_QUALITY = "skip_quality"


class SemanticAction(str, Enum):
    """Independent semantic lifetime decision for an observation.

    ``DispositionKind`` remains the detailed, backwards-compatible reason
    class used by existing telemetry.  This action states whether the source
    is usable immediately, retained for later consideration, or terminally
    rejected.
    """

    COMMIT = "commit"
    DEFER = "defer"
    REJECT = "reject"


class HistoricalEligibility(str, Enum):
    """Whether retained pixels may be selected by historical supervision."""

    NONE = "none"
    ELIGIBLE = "eligible"


class ResidencyIntent(str, Enum):
    NONE = "none"
    METADATA_ONLY = "metadata_only"
    HOT = "hot"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class ModalityEligibility:
    """Independent RGB, depth, insertion, and population-target eligibility."""

    rgb_supervision: bool = False
    rgb_quality: str = "unknown"
    depth_supervision: bool = False
    depth_quality: str = "unknown"
    depth_insertion: bool = False
    geometry_population_target: bool = False

    def __post_init__(self) -> None:
        for name in (
            "rgb_supervision",
            "depth_supervision",
            "depth_insertion",
            "geometry_population_target",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in ("rgb_quality", "depth_quality"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty quality class")


@dataclass(frozen=True, slots=True)
class ObservationDisposition:
    """Terminal semantic disposition for one full source identity."""

    source: SourceIdentity
    snapshot_id: int
    kind: DispositionKind
    observe: bool
    mapping_candidate: bool
    admit_supervision: bool
    allow_geometry_insertion: bool
    residency: ResidencyIntent
    reason: str
    representative: Optional[SourceIdentity] = None
    evidence_trace: tuple[EvidenceFact, ...] = field(default_factory=tuple)
    modality_eligibility: ModalityEligibility = field(default_factory=ModalityEligibility)
    semantic_action: Optional[SemanticAction] = None
    historical_eligibility: Optional[HistoricalEligibility] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        object.__setattr__(self, "snapshot_id", _integer(self.snapshot_id, "snapshot_id"))
        if not self.observe:
            raise ValueError("post-transport semantic dispositions must observe the source")
        if not self.reason:
            raise ValueError("disposition reason must be non-empty")
        action = self.semantic_action
        if action is None:
            action = (
                SemanticAction.COMMIT
                if self.kind is DispositionKind.ADMIT
                else SemanticAction.REJECT
            )
        if not isinstance(action, SemanticAction):
            raise TypeError("semantic_action must be SemanticAction")
        object.__setattr__(self, "semantic_action", action)
        historical = self.historical_eligibility
        if historical is None:
            historical = (
                HistoricalEligibility.ELIGIBLE
                if action is SemanticAction.COMMIT
                else HistoricalEligibility.NONE
            )
        if not isinstance(historical, HistoricalEligibility):
            raise TypeError("historical_eligibility must be HistoricalEligibility")
        object.__setattr__(self, "historical_eligibility", historical)
        eligibility = self.modality_eligibility
        if not isinstance(eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        admits_supervision = (
            eligibility.rgb_supervision or eligibility.depth_supervision
        )
        if (self.admit_supervision or self.allow_geometry_insertion) and not self.mapping_candidate:
            raise ValueError("eligible mapping work requires a mapping candidate")
        if action is SemanticAction.COMMIT:
            if self.kind is not DispositionKind.ADMIT:
                raise ValueError("COMMIT semantic action requires an ADMIT disposition")
            if self.admit_supervision != admits_supervision:
                raise ValueError("admit_supervision must match modality eligibility")
            if self.allow_geometry_insertion != eligibility.depth_insertion:
                raise ValueError(
                    "allow_geometry_insertion must match modality eligibility"
                )
            if not (self.admit_supervision or self.allow_geometry_insertion):
                raise ValueError("ADMIT dispositions must authorize mapping work")
            if self.residency not in {ResidencyIntent.HOT, ResidencyIntent.COLD}:
                raise ValueError("ADMIT dispositions require pixel residency intent")
            if historical is not HistoricalEligibility.ELIGIBLE:
                raise ValueError("COMMIT dispositions must remain historically eligible")
        elif action is SemanticAction.DEFER:
            if self.kind is not DispositionKind.SKIP_REDUNDANT:
                raise ValueError("DEFER semantic action requires SKIP_REDUNDANT")
            if self.mapping_candidate or self.admit_supervision or self.allow_geometry_insertion:
                raise ValueError("DEFER dispositions cannot authorize immediate mapping work")
            if self.residency is not ResidencyIntent.COLD:
                raise ValueError("DEFER dispositions require COLD residency")
            if historical is not HistoricalEligibility.ELIGIBLE:
                raise ValueError("DEFER dispositions must be historically eligible")
            if not admits_supervision:
                raise ValueError("DEFER dispositions require a historically usable modality")
        else:
            if self.kind is DispositionKind.ADMIT:
                raise ValueError("ADMIT dispositions require COMMIT semantic action")
            if self.admit_supervision or self.allow_geometry_insertion:
                raise ValueError(
                    "only ADMIT/COMMIT dispositions may authorize materialization"
                )
            if self.residency not in {ResidencyIntent.NONE, ResidencyIntent.METADATA_ONLY}:
                raise ValueError("REJECT dispositions cannot request pixel residency")
            if historical is not HistoricalEligibility.NONE:
                raise ValueError("REJECT dispositions cannot be historically eligible")
            if admits_supervision or eligibility.depth_insertion:
                raise ValueError("REJECT dispositions cannot retain modality eligibility")
        if self.kind is DispositionKind.COALESCE:
            if self.representative is None or self.representative == self.source:
                raise ValueError("COALESCE requires a distinct representative")
        elif self.representative is not None:
            raise ValueError("only COALESCE may name a representative")
        object.__setattr__(self, "evidence_trace", tuple(self.evidence_trace))
        if not all(isinstance(item, EvidenceFact) for item in self.evidence_trace):
            raise TypeError("evidence_trace must contain EvidenceFact values")

    @property
    def terminal(self) -> bool:
        return True

    @property
    def materialization_permitted(self) -> bool:
        return self.semantic_action is SemanticAction.COMMIT and self.admit_supervision

    @property
    def cold_retention_permitted(self) -> bool:
        """Whether CPU/reload-token retention may occur without mapping work."""

        return (
            self.semantic_action is SemanticAction.DEFER
            and self.residency is ResidencyIntent.COLD
            and self.historical_eligibility is HistoricalEligibility.ELIGIBLE
        )

    @property
    def historical_promotion_eligible(self) -> bool:
        """Whether a future control snapshot may mint a promotion capability."""

        return (
            self.semantic_action is SemanticAction.DEFER
            and self.historical_eligibility is HistoricalEligibility.ELIGIBLE
        )


@dataclass(frozen=True, slots=True)
class DepthInsertionPlan:
    """Identity-bearing insertion work for one control tick."""

    source: Optional[SourceIdentity]
    enabled: bool
    sample_stride: int
    candidate_budget: int
    seed_budget: int
    voxel_size_m: float
    minimum_confidence: float

    def __post_init__(self) -> None:
        if self.source is not None and not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity or None")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        for name in ("sample_stride", "candidate_budget", "seed_budget"):
            minimum = 1 if name == "sample_stride" else 0
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=minimum))
        for name in ("voxel_size_m", "minimum_confidence"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")
        if self.minimum_confidence < 0.0:
            raise ValueError("minimum_confidence must be non-negative")
        if self.seed_budget > self.candidate_budget:
            raise ValueError("seed_budget cannot exceed candidate_budget")
        if self.enabled and (self.source is None or self.seed_budget == 0):
            raise ValueError("enabled depth insertion requires source and seed budget")
        if not self.enabled and self.seed_budget != 0:
            raise ValueError("disabled depth insertion cannot carry a seed budget")

    @property
    def max_seeds(self) -> int:
        """Temporary scalar view for lower-level candidate truncation."""

        return self.seed_budget


@dataclass(frozen=True, slots=True)
class MutationBudget:
    max_total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_total", _integer(self.max_total, "max_total"))


@dataclass(frozen=True, slots=True)
class SupervisionWork:
    """One identity-bearing optimization view with authoritative modalities."""

    source: SourceIdentity
    resolution_tier: ResolutionTier
    modality_eligibility: ModalityEligibility

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if not isinstance(self.resolution_tier, (str, ResolutionTier)):
            raise TypeError("resolution_tier must be str or ResolutionTier")
        try:
            object.__setattr__(self, "resolution_tier", ResolutionTier(self.resolution_tier))
        except ValueError as exc:
            raise ValueError("resolution_tier must be native, half, quarter, or eighth") from exc
        if not isinstance(self.modality_eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        if not (
            self.modality_eligibility.rgb_supervision
            or self.modality_eligibility.depth_supervision
        ):
            raise ValueError("supervision work requires RGB or depth supervision eligibility")


@dataclass(frozen=True, slots=True)
class SupervisionConstellation:
    """Requested and realized optimizer/camera concurrency for one plan.

    ``requested_batch_size`` (B) is the loss/optimizer transaction budget;
    ``requested_camera_count`` (C) is the bounded render concurrency target.
    Underfill is legal and immediate.  It reduces the realized values without
    duplicating identities, while ``microbatch_sizes`` records how B>C is
    accumulated before the single optimizer transaction.
    """

    requested_batch_size: int
    requested_camera_count: int
    realized_batch_size: int
    realized_camera_count: int
    unique_eligible_count: int
    fill_ratio: float
    microbatch_count: int
    mode: str
    reason: str
    hysteresis_state: str
    cumulative_realized_supervised_views: int = 0
    cumulative_optimizer_updates: int = 0
    latest_completed_transaction_id: int | None = None
    latest_completed_optimization_transaction_ns: int | None = None
    p95_optimization_transaction_ns: int | None = None
    max_optimization_transaction_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("requested_batch_size", "requested_camera_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        for name in (
            "realized_batch_size",
            "realized_camera_count",
            "unique_eligible_count",
            "microbatch_count",
            "cumulative_realized_supervised_views",
            "cumulative_optimizer_updates",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        ratio = self.fill_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("fill_ratio must be numeric")
        ratio = float(ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("fill_ratio must be finite and in [0, 1]")
        object.__setattr__(self, "fill_ratio", ratio)
        for name in ("mode", "reason", "hysteresis_state"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "latest_completed_transaction_id",
            "latest_completed_optimization_transaction_ns",
            "p95_optimization_transaction_ns",
            "max_optimization_transaction_ns",
        ):
            value = getattr(self, name)
            if value is not None:
                minimum = 1 if name != "latest_completed_transaction_id" else 0
                object.__setattr__(self, name, _integer(value, name, minimum=minimum))
        if self.requested_camera_count > self.requested_batch_size:
            raise ValueError("requested camera count cannot exceed requested batch size")
        if self.realized_batch_size > min(
            self.requested_batch_size, self.unique_eligible_count
        ):
            raise ValueError("realized batch exceeds requested or unique eligible work")
        if self.realized_batch_size == 0:
            if self.realized_camera_count != 0 or self.microbatch_count != 0:
                raise ValueError("empty realization requires C=0 and zero microbatches")
        else:
            expected_c = min(self.requested_camera_count, self.unique_eligible_count)
            if self.realized_camera_count != expected_c:
                raise ValueError("realized C must equal min(requested C, unique eligible)")
            expected_microbatches = math.ceil(
                self.realized_batch_size / self.realized_camera_count
            )
            if self.microbatch_count != expected_microbatches:
                raise ValueError("microbatch_count does not match realized B/C")
        expected_ratio = self.realized_batch_size / self.requested_batch_size
        if not math.isclose(self.fill_ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("fill_ratio must equal realized B / requested B")

    @property
    def microbatch_sizes(self) -> tuple[int, ...]:
        if self.realized_batch_size == 0:
            return ()
        full, remainder = divmod(
            self.realized_batch_size, self.realized_camera_count
        )
        return (
            *((self.realized_camera_count,) * full),
            *((remainder,) if remainder else ()),
        )


@dataclass(frozen=True, slots=True)
class MappingWorkPlan:
    """The sole identity-bearing behavior-changing plan for one control tick."""

    plan_id: str
    snapshot_id: int
    control_version: int
    supervision_sources: tuple[SourceIdentity, ...]
    replay_sources: tuple[SourceIdentity, ...]
    resolution_tier: ResolutionTier
    optimizer_updates: int
    insertion: DepthInsertionPlan
    allow_growth: bool
    allow_relocation: bool
    mutation_budget: MutationBudget
    freshness_mode: FreshnessMode
    map_content_version: int
    topology_version: int
    parameter_version: int
    supervision_work: tuple[SupervisionWork, ...]
    replay_work: tuple[SupervisionWork, ...] = field(default_factory=tuple)
    replay_insertion: Optional[DepthInsertionPlan] = None
    reason_trace: tuple[str, ...] = field(default_factory=tuple)
    replay_population_size: int = 0
    replay_sampling_seed: int = 0
    replay_sampling_epoch: int = 0
    replay_selected_weights: tuple[float, ...] = field(default_factory=tuple)
    replay_candidate_sources: tuple[SourceIdentity, ...] = field(default_factory=tuple)
    replay_candidate_weights: tuple[float, ...] = field(default_factory=tuple)
    replay_trial_sources: tuple[SourceIdentity, ...] = field(default_factory=tuple)
    requested_supervision_slots: int = 0
    requested_replay_slots: int = 0
    supervision_constellation: Optional[SupervisionConstellation] = None
    materialization_sources: Optional[tuple[SourceIdentity, ...]] = None
    transaction_class: Optional[MappingTransactionClass] = None
    post_refinement: bool = False
    resolution_action: str = "hold"
    resolution_reason: str = "compatibility_default"
    resolution_budget_ms: Optional[float] = None
    resolution_probe: bool = False
    native_geometry_evidence_purpose: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str):
            raise TypeError("plan_id must be str")
        if not self.plan_id:
            raise ValueError("plan_id and resolution_tier must be non-empty")
        if not isinstance(self.resolution_tier, (str, ResolutionTier)):
            raise TypeError("resolution_tier must be str or ResolutionTier")
        tier = "native" if self.resolution_tier == "full" else self.resolution_tier
        try:
            object.__setattr__(self, "resolution_tier", ResolutionTier(tier))
        except ValueError as exc:
            raise ValueError("resolution_tier must be native, half, quarter, or eighth") from exc
        for name in (
            "snapshot_id",
            "control_version",
            "optimizer_updates",
            "map_content_version",
            "topology_version",
            "parameter_version",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.insertion, DepthInsertionPlan):
            raise TypeError("insertion must be DepthInsertionPlan")
        if not isinstance(self.mutation_budget, MutationBudget):
            raise TypeError("mutation_budget must be MutationBudget")
        if not isinstance(self.freshness_mode, FreshnessMode):
            raise TypeError("freshness_mode must be FreshnessMode")
        for name in ("allow_growth", "allow_relocation"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        supervision = tuple(self.supervision_sources)
        replay = tuple(self.replay_sources)
        if not all(isinstance(item, SourceIdentity) for item in supervision + replay):
            raise TypeError("work identities must be SourceIdentity values")
        if len(set(supervision)) != len(supervision) or len(set(replay)) != len(replay):
            raise ValueError("supervision and replay identities must each be unique")
        if set(supervision).intersection(replay):
            raise ValueError("an identity cannot be both supervision and replay work")
        object.__setattr__(self, "supervision_sources", supervision)
        object.__setattr__(self, "replay_sources", replay)
        materialization = (
            supervision
            if self.materialization_sources is None
            else tuple(self.materialization_sources)
        )
        if not all(isinstance(item, SourceIdentity) for item in materialization):
            raise TypeError("materialization_sources must contain SourceIdentity values")
        if len(set(materialization)) != len(materialization):
            raise ValueError("materialization identities must be unique")
        if len(materialization) > 1:
            raise ValueError("a work plan may materialize at most one submitted source")
        if self.insertion.enabled and self.insertion.source not in materialization:
            raise ValueError("live insertion source must be authorized materialization")
        object.__setattr__(self, "materialization_sources", materialization)
        work = tuple(self.supervision_work)
        if not all(isinstance(item, SupervisionWork) for item in work):
            raise TypeError("supervision_work must contain SupervisionWork values")
        if tuple(item.source for item in work) != supervision:
            raise ValueError("supervision_work identities must exactly match supervision_sources")
        if any(item.resolution_tier is not self.resolution_tier for item in work):
            raise ValueError("supervision_work tiers must match plan resolution_tier")
        object.__setattr__(self, "supervision_work", work)
        replay_work = tuple(self.replay_work)
        if not all(isinstance(item, SupervisionWork) for item in replay_work):
            raise TypeError("replay_work must contain SupervisionWork values")
        if tuple(item.source for item in replay_work) != replay:
            raise ValueError("replay_work identities must exactly match replay_sources")
        if any(item.resolution_tier is not self.resolution_tier for item in replay_work):
            raise ValueError("replay_work tiers must match plan resolution_tier")
        object.__setattr__(self, "replay_work", replay_work)
        if type(self.post_refinement) is not bool:
            raise TypeError("post_refinement must be bool")
        if type(self.resolution_probe) is not bool:
            raise TypeError("resolution_probe must be bool")
        if type(self.native_geometry_evidence_purpose) is not bool:
            raise TypeError("native_geometry_evidence_purpose must be bool")
        # Plan-level resolution homogeneity: after the sole owner selects the
        # final tier, every view in the transaction carries it.  A
        # heterogeneous plan must fail here, at construction, never survive to
        # materialization or the renderer.
        for label, items in (("supervision_work", work), ("replay_work", replay_work)):
            if any(item.resolution_tier is not self.resolution_tier for item in items):
                raise ValueError(f"{label} tiers must match plan resolution_tier")
        # Classification is by composition, with the post-refinement phase
        # flag taking precedence so a refinement transaction stays separately
        # attributable even when its replay candidate pool is exhausted.
        if self.post_refinement:
            transaction_class = MappingTransactionClass.POST_REFINEMENT
        elif supervision and replay:
            transaction_class = MappingTransactionClass.MIXED
        elif supervision:
            transaction_class = MappingTransactionClass.FRESH
        elif replay:
            transaction_class = MappingTransactionClass.REPLAY_ONLY
        else:
            # A plan carrying no work at all is a degenerate no-op tick: there
            # is no executed work to attribute, so it is deliberately left
            # unclassified rather than mislabelled as replay or fresh.
            transaction_class = None
        object.__setattr__(self, "transaction_class", transaction_class)
        # Only the positive composition invariants are enforced.  Replay-class
        # transactions must carry no fresh supervision work; an *empty* replay
        # pool stays legal, because post-refinement legitimately runs with an
        # exhausted candidate catalog and must degrade to a no-op rather than
        # raise from the contract layer.
        if transaction_class in {
            MappingTransactionClass.REPLAY_ONLY,
            MappingTransactionClass.POST_REFINEMENT,
        } and supervision:
            raise ValueError("replay transaction must not carry fresh supervision work")
        if self.resolution_action not in {"upscale", "hold", "downscale"}:
            raise ValueError("resolution_action must be upscale, hold, or downscale")
        if not isinstance(self.resolution_reason, str) or not self.resolution_reason:
            raise ValueError("resolution_reason must be a non-empty string")
        budget_ms = self.resolution_budget_ms
        if budget_ms is not None:
            if (
                isinstance(budget_ms, bool)
                or not isinstance(budget_ms, (int, float))
                or not math.isfinite(float(budget_ms))
                or float(budget_ms) < 0.0
            ):
                raise ValueError("resolution_budget_ms must be finite and non-negative")
            object.__setattr__(self, "resolution_budget_ms", float(budget_ms))
        constellation = self.supervision_constellation
        work_count = len(work) + len(replay_work)
        if constellation is None or (
            isinstance(constellation, SupervisionConstellation)
            and constellation.mode == "compatibility_default"
            and constellation.realized_batch_size != work_count
        ):
            # Compatibility construction for tests and non-production callers.
            # WorkBudgetOwner always supplies the explicit adaptive decision.
            from eligsir.mapping.batch_contract import (
                qualified_camera_count_for_batch,
            )

            requested_batch = max(1, work_count)
            try:
                requested_camera = qualified_camera_count_for_batch(requested_batch)
            except ValueError:
                requested_camera = 1
            realized_camera = min(requested_camera, work_count)
            constellation = SupervisionConstellation(
                requested_batch_size=requested_batch,
                requested_camera_count=requested_camera,
                realized_batch_size=work_count,
                realized_camera_count=realized_camera,
                unique_eligible_count=work_count,
                fill_ratio=work_count / requested_batch,
                microbatch_count=(
                    0 if work_count == 0 else math.ceil(work_count / realized_camera)
                ),
                mode="compatibility_default",
                reason="implicit_noncanonical_plan",
                hysteresis_state="unmanaged",
            )
            object.__setattr__(self, "supervision_constellation", constellation)
        elif not isinstance(constellation, SupervisionConstellation):
            raise TypeError("supervision_constellation must be SupervisionConstellation")
        if constellation.realized_batch_size != work_count:
            raise ValueError(
                "supervision_constellation realized B must match planned work identities"
            )
        replay_insertion = self.replay_insertion
        if replay_insertion is not None:
            if not isinstance(replay_insertion, DepthInsertionPlan):
                raise TypeError("replay_insertion must be DepthInsertionPlan or None")
            if not replay_insertion.enabled:
                raise ValueError("replay_insertion must be enabled when present")
            if replay_insertion.source not in replay:
                raise ValueError("replay insertion source must be planned replay work")
            replay_modality = next(
                work.modality_eligibility
                for work in replay_work
                if work.source == replay_insertion.source
            )
            if not replay_modality.depth_insertion:
                raise ValueError("replay insertion source requires depth eligibility")
        for name in (
            "replay_population_size",
            "replay_sampling_seed",
            "replay_sampling_epoch",
            "requested_supervision_slots",
            "requested_replay_slots",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        selected_weights = tuple(float(weight) for weight in self.replay_selected_weights)
        if selected_weights and len(selected_weights) != len(replay):
            raise ValueError("replay_selected_weights must align with replay_sources")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in selected_weights):
            raise ValueError("selected replay weights must be finite and non-negative")
        object.__setattr__(self, "replay_selected_weights", selected_weights)
        candidate_sources = tuple(self.replay_candidate_sources)
        candidate_weights = tuple(self.replay_candidate_weights)
        if not all(isinstance(source, SourceIdentity) for source in candidate_sources):
            raise TypeError("replay_candidate_sources must contain SourceIdentity values")
        if len(set(candidate_sources)) != len(candidate_sources):
            raise ValueError("replay_candidate_sources must be unique")
        if len(candidate_weights) != len(candidate_sources):
            raise ValueError("replay candidate weights must align with candidate sources")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
            for weight in candidate_weights
        ):
            raise ValueError("replay candidate weights must be finite and non-negative")
        if candidate_sources and len(candidate_sources) != self.replay_population_size:
            raise ValueError("replay candidate sources must match replay population size")
        if candidate_sources and not set(replay).issubset(candidate_sources):
            raise ValueError("replay sources must be selected from candidate sources")
        object.__setattr__(self, "replay_candidate_sources", candidate_sources)
        object.__setattr__(
            self,
            "replay_candidate_weights",
            tuple(float(weight) for weight in candidate_weights),
        )
        trials = tuple(self.replay_trial_sources)
        if not all(isinstance(source, SourceIdentity) for source in trials):
            raise TypeError("replay_trial_sources must contain SourceIdentity values")
        if len(set(trials)) != len(trials) or not set(trials).issubset(replay):
            raise ValueError("replay_trial_sources must be unique realized replay sources")
        object.__setattr__(self, "replay_trial_sources", trials)
        if not isinstance(self.reason_trace, tuple):
            raise TypeError("reason_trace must be tuple[str, ...]")
        if not all(isinstance(reason, str) and reason for reason in self.reason_trace):
            raise ValueError("reason_trace entries must be non-empty strings")


__all__ = [
    "CompactGpuTelemetry",
    "ControlLimits",
    "ControlProposal",
    "ControlSnapshot",
    "DepthInsertionPlan",
    "DispositionKind",
    "EvidenceFact",
    "FreshnessMode",
    "HistoricalEligibility",
    "MappingTransactionClass",
    "MappingWorkPlan",
    "ModalityEligibility",
    "MutationBudget",
    "ObservationDisposition",
    "PreparedTierPolicy",
    "ResidencyIntent",
    "ResolutionTier",
    "SemanticAction",
    "SupervisionWork",
    "SupervisionConstellation",
]
