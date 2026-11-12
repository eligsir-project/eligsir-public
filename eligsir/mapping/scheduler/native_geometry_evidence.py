"""Bounded purpose handoff for native B4 geometry evidence.

The population policy may ask for more geometry evidence, but it must not
choose a frame, batch tier, or force a GPU operation.  This lane therefore
retains one version-current demand and merely observes the scheduler's normal
plan.  A demand becomes in-flight only when that ordinary plan already offers
one homogeneous, qualified native B4 opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import Any

from eligsir.mapping.evidence.evidence_demand import EvidenceDemand
from eligsir.mapping.evidence.evidence_demand_scheduler import (
    EvidenceDemandCurrentVersions,
)
from eligsir.mapping.evidence.surface_target_attestation import (
    SurfaceTargetQualificationAttestation,
)


class NativeGeometryEvidenceReason(str, Enum):
    """Compact feedback for one native-evidence demand lifecycle."""

    ABSENT = "absent"
    QUEUE_FULL = "queue_full"
    EXPIRED = "expired"
    STALE = "stale"
    DEFERRED_UNREADY = "deferred_unready"
    DEFERRED_BUDGET = "deferred_budget"
    SCHEDULED = "scheduled"
    INCOMPLETE = "incomplete"
    SATISFIED = "satisfied"


@dataclass(frozen=True, slots=True)
class NativeGeometryEvidenceTelemetry:
    """CPU-only counters; no tensor or source payload is retained here."""

    published: int = 0
    queue_full: int = 0
    expired: int = 0
    stale: int = 0
    deferred_unready: int = 0
    deferred_budget: int = 0
    scheduled: int = 0
    incomplete: int = 0
    satisfied: int = 0


@dataclass(frozen=True, slots=True)
class NativeGeometryEvidenceOutcome:
    """Completed E0 provenance plus one post-fence target attestation."""

    map_version: int
    topology_version: int
    coordinate_version: int
    schema_version: int
    optimizer_version: int
    surface_version: int
    policy_version: int
    parameter_version: int
    realized_batch_size: int
    realized_camera_count: int
    resolution_tier: str
    native_geometry_leases: int
    target_attestation: SurfaceTargetQualificationAttestation | None

    def __post_init__(self) -> None:
        for name in (
            "map_version", "topology_version", "coordinate_version",
            "schema_version", "optimizer_version", "surface_version",
            "policy_version", "parameter_version", "realized_batch_size", "realized_camera_count",
            "native_geometry_leases",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if not isinstance(self.resolution_tier, str) or not self.resolution_tier:
            raise ValueError("resolution_tier must be non-empty")
        if self.target_attestation is not None and not isinstance(
            self.target_attestation, SurfaceTargetQualificationAttestation
        ):
            raise TypeError(
                "target_attestation must be SurfaceTargetQualificationAttestation or None"
            )

    def matches(self, demand: EvidenceDemand) -> bool:
        expected = demand.snapshot_versions
        observed = self.snapshot_versions
        return (
            all(
                observed[name] == expected[name]
                for name in (
                    "map", "topology", "coordinate", "schema", "surface", "policy"
                )
            )
            and observed["optimizer"] == expected["optimizer"] + 1
            and self.realized_batch_size == 4
            and self.realized_camera_count == 4
            and self.resolution_tier == "native"
            and self.native_geometry_leases == 4
            and self.target_attestation is not None
            and self.target_attestation.qualified
            and self._attestation_matches_post_adam()
        )

    def _attestation_matches_post_adam(self) -> bool:
        """Require the attestation's exact post-Adam map/parameter vector."""

        attestation = self.target_attestation
        if attestation is None:
            return False
        snapshot = attestation.snapshot
        return (
            snapshot.map_content == self.map_version + 1
            and snapshot.topology == self.topology_version
            and snapshot.coordinate_frame == self.coordinate_version
            and snapshot.schema == self.schema_version
            and snapshot.optimizer_update == self.optimizer_version
            and snapshot.surface_evidence == self.surface_version
            and snapshot.policy == self.policy_version
            and attestation.parameter_version == self.parameter_version + 1
        )

    @property
    def snapshot_versions(self) -> dict[str, int]:
        """Version vector used to reject cross-snapshot completion."""

        return {
            "map": self.map_version,
            "topology": self.topology_version,
            "coordinate": self.coordinate_version,
            "schema": self.schema_version,
            "optimizer": self.optimizer_version,
            "surface": self.surface_version,
            "policy": self.policy_version,
            "parameter": self.parameter_version,
        }


class NativeGeometryEvidenceLane:
    """Hold at most one asynchronous policy demand through one E0 result.

    ``next_purpose`` and ``observe_scheduler_outcome`` are intentionally CPU
    metadata operations.  The first never inspects device decision tensors;
    the second observes a plan that the normal candidate/tier/preparation
    machinery already chose.  It can neither stall that machinery nor make a
    native-tier override.
    """

    def __init__(self) -> None:
        self._pending: EvidenceDemand | None = None
        self._in_flight: EvidenceDemand | None = None
        self.telemetry = NativeGeometryEvidenceTelemetry()

    @property
    def in_flight(self) -> EvidenceDemand | None:
        return self._in_flight

    @property
    def pending(self) -> EvidenceDemand | None:
        return self._pending

    def _count(self, field: str) -> None:
        self.telemetry = replace(
            self.telemetry, **{field: getattr(self.telemetry, field) + 1}
        )

    def publish(self, demand: EvidenceDemand) -> bool:
        """Accept one demand without replacing a pending/in-flight epoch."""

        if not isinstance(demand, EvidenceDemand):
            raise TypeError("demand must be an EvidenceDemand")
        if self._pending is not None or self._in_flight is not None:
            self._count("queue_full")
            return False
        self._pending = demand
        self._count("published")
        return True

    def next_purpose(
        self, *, current_versions: EvidenceDemandCurrentVersions, epoch: Real
    ) -> EvidenceDemand | None:
        """Return a current demand as a purpose signal, never a view choice."""

        if not isinstance(current_versions, EvidenceDemandCurrentVersions):
            raise TypeError("current_versions must be EvidenceDemandCurrentVersions")
        if isinstance(epoch, bool) or not isinstance(epoch, Real):
            raise TypeError("epoch must be a real number")
        demand = self._pending
        if demand is None:
            return None
        if epoch > demand.expiry_epoch:
            self._pending = None
            self._count("expired")
            return None
        if not current_versions.matches(demand):
            self._pending = None
            self._count("stale")
            return None
        return demand

    def observe_scheduler_outcome(
        self, plan: Any, *, purpose: EvidenceDemand | None = None
    ) -> NativeGeometryEvidenceReason:
        """Observe a normal plan and retain unmet demand for a later epoch."""

        if purpose is None:
            return NativeGeometryEvidenceReason.ABSENT
        if purpose is not self._pending:
            raise ValueError("purpose must be the lane's current pending demand")
        constellation = getattr(plan, "supervision_constellation", None)
        batch = getattr(
            plan, "realized_batch_size", getattr(constellation, "realized_batch_size", None)
        )
        cameras = getattr(
            plan, "realized_camera_count", getattr(constellation, "realized_camera_count", None)
        )
        tier = getattr(plan, "resolution_tier", None)
        geometry = getattr(plan, "geometry_eligible_views", None)
        if geometry is None:
            geometry = sum(
                bool(getattr(work.modality_eligibility, "geometry_population_target", False))
                for work in (*getattr(plan, "supervision_work", ()), *getattr(plan, "replay_work", ()))
            )
        if any(type(value) is not int or value < 0 for value in (batch, cameras, geometry)):
            raise TypeError("scheduler outcome must expose non-negative exact B/C/geometry counts")
        tier = getattr(tier, "value", tier)
        if not isinstance(tier, str):
            raise TypeError("scheduler outcome must expose a string resolution_tier")
        if batch != 4 or cameras != 4 or geometry != 4:
            self._count("deferred_unready")
            return NativeGeometryEvidenceReason.DEFERRED_UNREADY
        if tier != "native":
            self._count("deferred_budget")
            return NativeGeometryEvidenceReason.DEFERRED_BUDGET
        self._in_flight = purpose
        self._pending = None
        self._count("scheduled")
        return NativeGeometryEvidenceReason.SCHEDULED

    def complete(self, outcome: NativeGeometryEvidenceOutcome) -> NativeGeometryEvidenceReason:
        """Retire one terminal E0 result; satisfy only an exact target."""

        if not isinstance(outcome, NativeGeometryEvidenceOutcome):
            raise TypeError("outcome must be NativeGeometryEvidenceOutcome")
        demand = self._in_flight
        if demand is None:
            self._count("incomplete")
            return NativeGeometryEvidenceReason.INCOMPLETE
        # Once Adam has advanced past the policy deadline, the in-flight
        # snapshot cannot be recovered. Retire it before accepting a fresh
        # demand instead of pinning this one-slot lane forever.
        post_optimizer_version = (
            outcome.target_attestation.snapshot.optimizer_update
            if outcome.target_attestation is not None
            else outcome.optimizer_version
        )
        if post_optimizer_version > demand.expiry_epoch:
            self._in_flight = None
            self._count("expired")
            return NativeGeometryEvidenceReason.EXPIRED
        expected = demand.snapshot_versions
        observed = outcome.snapshot_versions
        if (
            any(
                observed[name] != expected[name]
                for name in (
                    "map", "topology", "coordinate", "schema", "surface", "policy"
                )
            )
            or observed["optimizer"] != expected["optimizer"] + 1
        ):
            self._in_flight = None
            self._count("stale")
            return NativeGeometryEvidenceReason.STALE
        if (
            outcome.target_attestation is not None
            and not outcome._attestation_matches_post_adam()
        ):
            self._in_flight = None
            self._count("stale")
            return NativeGeometryEvidenceReason.STALE
        if not outcome.matches(demand):
            self._in_flight = None
            self._count("incomplete")
            return NativeGeometryEvidenceReason.INCOMPLETE
        self._in_flight = None
        self._count("satisfied")
        return NativeGeometryEvidenceReason.SATISFIED

    def discard_stale_completion(self) -> NativeGeometryEvidenceReason:
        """Retire an in-flight slot whose runtime plan identity was superseded."""

        if self._in_flight is None:
            self._count("incomplete")
            return NativeGeometryEvidenceReason.INCOMPLETE
        self._in_flight = None
        self._count("stale")
        return NativeGeometryEvidenceReason.STALE


__all__ = [
    "NativeGeometryEvidenceLane",
    "NativeGeometryEvidenceOutcome",
    "NativeGeometryEvidenceReason",
    "NativeGeometryEvidenceTelemetry",
]
