"""Deterministic scheduler contract for native geometry evidence demand."""

from __future__ import annotations

from dataclasses import dataclass

from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandKind,
    EvidenceDemandModality,
)
from eligsir.mapping.evidence.evidence_demand_scheduler import (
    EvidenceDemandCurrentVersions,
)
from eligsir.mapping.evidence.surface_target_attestation import (
    SurfaceTargetQualificationAttestation,
)
from eligsir.mapping.scheduler.native_geometry_evidence import (
    NativeGeometryEvidenceLane,
    NativeGeometryEvidenceOutcome,
    NativeGeometryEvidenceReason,
)
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
)


def _versions(*, offset: int = 0) -> EvidenceDemandCurrentVersions:
    return EvidenceDemandCurrentVersions(
        map=4 + offset, topology=5 + offset, coordinate=6 + offset,
        schema=7 + offset, optimizer=8 + offset, surface=9 + offset,
        policy=10 + offset,
    )


def _demand(*, expiry_epoch: int = 10, offset: int = 0) -> EvidenceDemand:
    return EvidenceDemand(
        kind=EvidenceDemandKind.DEPTH_SURFACE,
        requested_modality=EvidenceDemandModality.DEPTH,
        target_ids=(), region_id="uncertain_geometry", cell_id=None,
        map_version=4 + offset, topology_version=5 + offset,
        coordinate_version=6 + offset, schema_version=7 + offset,
        optimizer_version=8 + offset, surface_version=9 + offset,
        policy_version=10 + offset,
        priority=1.0, expiry_epoch=expiry_epoch,
    )


def _attestation(*, qualified: bool, optimizer_version: int = 9, parameter_version: int = 12):
    return SurfaceTargetQualificationAttestation(
        snapshot=EvidenceSnapshotToken(
            map_content=5,
            topology=5,
            coordinate_frame=6,
            schema=7,
            optimizer_update=optimizer_version,
            surface_evidence=9,
            policy=10,
            parameter=parameter_version,
        ),
        parameter_version=parameter_version,
        qualified=qualified,
    )


@dataclass(frozen=True)
class _Plan:
    realized_batch_size: int
    realized_camera_count: int
    resolution_tier: str
    geometry_eligible_views: int


def test_absent_demand_leaves_normal_scheduler_outcome_unchanged():
    lane = NativeGeometryEvidenceLane()
    normal = _Plan(1, 1, "quarter", 0)

    assert lane.next_purpose(current_versions=_versions(), epoch=8) is None
    assert lane.observe_scheduler_outcome(normal) is NativeGeometryEvidenceReason.ABSENT
    assert lane.in_flight is None


def test_affordable_normal_native_b4_becomes_one_in_flight_epoch():
    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand()) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)

    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED
    assert lane.in_flight is purpose


def test_unavailable_or_budget_rejected_demand_defers_while_quarter_progresses():
    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand()) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None

    assert lane.observe_scheduler_outcome(
        _Plan(2, 2, "quarter", 2), purpose=purpose
    ) is NativeGeometryEvidenceReason.DEFERRED_UNREADY
    assert lane.in_flight is None
    assert lane.next_purpose(current_versions=_versions(), epoch=8) == purpose

    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "quarter", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.DEFERRED_BUDGET
    assert lane.in_flight is None


def test_incomplete_native_e0_retires_then_a_fresh_demand_can_schedule():
    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand()) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    wrong = NativeGeometryEvidenceOutcome(
        map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
        optimizer_version=9, surface_version=9, policy_version=10, parameter_version=11,
        realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
        native_geometry_leases=4, target_attestation=_attestation(qualified=False),
    )
    assert lane.complete(wrong) is NativeGeometryEvidenceReason.INCOMPLETE
    assert lane.in_flight is None
    assert lane.telemetry.incomplete == 1

    fresh = _demand(offset=1)
    assert lane.publish(fresh) is True
    fresh_purpose = lane.next_purpose(current_versions=_versions(offset=1), epoch=9)
    assert fresh_purpose is fresh
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=fresh_purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    cross_snapshot = NativeGeometryEvidenceOutcome(
        map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
        optimizer_version=8, surface_version=9, policy_version=10, parameter_version=11,
        realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
        native_geometry_leases=4, target_attestation=_attestation(qualified=True),
    )
    assert lane.complete(cross_snapshot) is NativeGeometryEvidenceReason.STALE
    assert lane.in_flight is None
    assert lane.telemetry.stale == 1


def test_only_complete_native_e0_b4_target_satisfies_the_in_flight_demand():
    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand()) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    complete = NativeGeometryEvidenceOutcome(
        map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
        optimizer_version=9, surface_version=9, policy_version=10, parameter_version=11,
        realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
        native_geometry_leases=4, target_attestation=_attestation(qualified=True),
    )
    assert lane.complete(complete) is NativeGeometryEvidenceReason.SATISFIED
    assert lane.in_flight is None


def test_completion_accepts_only_the_e0_optimizer_step_after_the_demand():
    """Demand t is rendered as E0 t+1 and attested at the same optimizer step."""

    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand()) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    def outcome(optimizer_version: int) -> NativeGeometryEvidenceOutcome:
        return NativeGeometryEvidenceOutcome(
            map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
            optimizer_version=optimizer_version, surface_version=9, policy_version=10,
            parameter_version=11,
            realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
            native_geometry_leases=4,
            target_attestation=_attestation(
                qualified=True, optimizer_version=optimizer_version
            ),
        )

    assert lane.complete(outcome(9)) is NativeGeometryEvidenceReason.SATISFIED


def test_completion_rejects_nonmatching_e0_optimizer_version():
    """E0 provenance must be the demand's exact completed optimizer version."""

    for optimizer_version in (8, 10):
        lane = NativeGeometryEvidenceLane()
        assert lane.publish(_demand(expiry_epoch=20)) is True
        purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
        assert purpose is not None
        assert lane.observe_scheduler_outcome(
            _Plan(4, 4, "native", 4), purpose=purpose
        ) is NativeGeometryEvidenceReason.SCHEDULED

        assert lane.complete(
            NativeGeometryEvidenceOutcome(
                map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
                optimizer_version=optimizer_version, surface_version=9, policy_version=10,
                parameter_version=11,
                realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
                native_geometry_leases=4,
                target_attestation=_attestation(
                    qualified=True, optimizer_version=optimizer_version
                ),
            )
        ) is NativeGeometryEvidenceReason.STALE


def test_expired_native_e0_outcome_retires_the_in_flight_demand():
    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand(expiry_epoch=8)) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    expired = NativeGeometryEvidenceOutcome(
        map_version=4, topology_version=5, coordinate_version=6, schema_version=7,
        optimizer_version=9, surface_version=9, policy_version=10, parameter_version=11,
        realized_batch_size=4, realized_camera_count=4, resolution_tier="native",
        native_geometry_leases=4, target_attestation=_attestation(qualified=True),
    )
    assert lane.complete(expired) is NativeGeometryEvidenceReason.EXPIRED
    assert lane.in_flight is None
    assert lane.telemetry.expired == 1


def test_missing_attestation_at_the_valid_e0_expiry_is_incomplete_not_expired():
    """E0 is already at t+1; missing fence output must not invent a t+2 step."""

    lane = NativeGeometryEvidenceLane()
    assert lane.publish(_demand(expiry_epoch=9)) is True
    purpose = lane.next_purpose(current_versions=_versions(), epoch=8)
    assert purpose is not None
    assert lane.observe_scheduler_outcome(
        _Plan(4, 4, "native", 4), purpose=purpose
    ) is NativeGeometryEvidenceReason.SCHEDULED

    outcome = NativeGeometryEvidenceOutcome(
        map_version=4,
        topology_version=5,
        coordinate_version=6,
        schema_version=7,
        optimizer_version=9,
        surface_version=9,
        policy_version=10,
        parameter_version=11,
        realized_batch_size=4,
        realized_camera_count=4,
        resolution_tier="native",
        native_geometry_leases=4,
        target_attestation=None,
    )

    assert lane.complete(outcome) is NativeGeometryEvidenceReason.INCOMPLETE
    assert lane.in_flight is None
    assert lane.telemetry.incomplete == 1
    assert lane.telemetry.expired == 0
