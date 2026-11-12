"""Runtime binding guards for native geometry evidence purposes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from eligsir.contracts import ModalityEligibility, SourceIdentity
from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandKind,
    EvidenceDemandModality,
)
from eligsir.mapping.evidence.surface_target_attestation import (
    SurfaceTargetQualificationAttestation,
)
from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease
from eligsir.mapping.scheduler.native_geometry_evidence import (
    NativeGeometryEvidenceLane,
)
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.runtime.causal_replay import decision_trace_row
from eligsir.runtime.production_application import CanonicalMappingRuntime


def _demand() -> EvidenceDemand:
    return EvidenceDemand(
        kind=EvidenceDemandKind.DEPTH_SURFACE,
        requested_modality=EvidenceDemandModality.DEPTH,
        target_ids=(),
        region_id="uncertain_geometry",
        cell_id=None,
        map_version=4,
        topology_version=5,
        coordinate_version=6,
        schema_version=7,
        optimizer_version=8,
        surface_version=9,
        policy_version=10,
        priority=1.0,
        expiry_epoch=12,
    )


def _plan(plan_id: str) -> SimpleNamespace:
    eligibility = SimpleNamespace(geometry_population_target=True)
    return SimpleNamespace(
        plan_id=plan_id,
        native_geometry_evidence_purpose=True,
        supervision_constellation=SimpleNamespace(
            realized_batch_size=4, realized_camera_count=4
        ),
        resolution_tier="native",
        insertion=None,
        supervision_sources=(),
        replay_sources=(),
        reason_trace=(),
        mutation_budget=SimpleNamespace(max_total=0),
        supervision_work=tuple(
            SimpleNamespace(modality_eligibility=eligibility) for _ in range(4)
        ),
        replay_work=(),
    )


def _runtime(*, active_plan: SimpleNamespace | None) -> CanonicalMappingRuntime:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._native_geometry_evidence_lane = NativeGeometryEvidenceLane()
    runtime._native_geometry_evidence_outcome_trace = []
    runtime._population_shadow_active_plan = active_plan
    return runtime


def _native_b4_result(
    *,
    map_version: int = 4,
    optimizer_version: int = 9,
    parameter_version: int = 11,
) -> SimpleNamespace:
    token = EvidenceSnapshotToken(
        map_content=map_version,
        topology=5,
        coordinate_frame=6,
        schema=7,
        optimizer_update=optimizer_version,
        surface_evidence=9,
        policy=10,
        parameter=parameter_version,
    )
    eligibility = ModalityEligibility(geometry_population_target=True)
    sources = tuple(
        SourceIdentity("replay", 1, ordinal, 1_700_000_000 + ordinal)
        for ordinal in range(4)
    )
    leases = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=eligibility,
            alpha=torch.ones(2, 2, 1),
            rendered_depth=torch.ones(2, 2, 1),
        )
        for source in sources
    )
    return SimpleNamespace(
        evidence=SupervisionEvidenceBatch(
            snapshot_token=token,
            transaction_id="native-b4",
            source_identities=sources,
            modalities=tuple(
                SupervisionModality(geometry_evidence_enabled=True)
                for _ in sources
            ),
            modality_eligibility=(eligibility,) * 4,
            geometry_render_leases=leases,
        ),
        realized_batch_size=4,
        realized_camera_count=4,
    )


def _attestation(
    *,
    qualified: bool,
    map_version: int = 5,
    optimizer_version: int = 9,
    parameter_version: int = 12,
):
    return SurfaceTargetQualificationAttestation(
        snapshot=EvidenceSnapshotToken(
            map_content=map_version,
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


def test_target_attestation_exposes_the_post_adam_parameter_version():
    attestation = _attestation(qualified=True)

    assert attestation.parameter_version == 12
    assert attestation.trace()["snapshot"]["parameter"] == 12


def test_superseded_b4_native_plan_cannot_schedule_the_pending_demand():
    actual = _plan("actual")
    runtime = _runtime(active_plan=actual)
    assert runtime._native_geometry_evidence_lane.publish(_demand())

    runtime._bind_native_geometry_evidence_after_success(_plan("superseded"))

    assert runtime._native_geometry_evidence_lane.in_flight is None
    assert runtime._native_geometry_evidence_lane.pending is not None


def test_missing_active_b4_native_plan_cannot_schedule_the_pending_demand():
    runtime = _runtime(active_plan=None)
    assert runtime._native_geometry_evidence_lane.publish(_demand())

    runtime._bind_native_geometry_evidence_after_success(_plan("unbound"))

    assert runtime._native_geometry_evidence_lane.in_flight is None
    assert runtime._native_geometry_evidence_lane.pending is not None
    assert runtime.native_geometry_evidence_telemetry()["outcome_trace"] == []


def test_actual_b4_native_plan_schedules_and_serializes_its_lane_trace():
    actual = _plan("actual")
    runtime = _runtime(active_plan=actual)
    assert runtime._native_geometry_evidence_lane.publish(_demand())

    runtime._bind_native_geometry_evidence_after_success(actual)

    lane = runtime.native_geometry_evidence_telemetry()
    row = decision_trace_row(
        SimpleNamespace(
            work_plan=actual,
            native_geometry_evidence_telemetry=lane,
        )
    )
    assert lane["state"] == "in_flight"
    assert lane["outcome_trace"] == [
        {"plan_id": "actual", "phase": "schedule", "reason": "scheduled"}
    ]
    assert row["native_geometry_evidence"] == {"purpose": True, "lane": lane}
    json.dumps(row)


def test_post_fence_qualified_target_attestation_satisfies_scheduled_native_b4():
    """Breaking the post-fence attestation must retire the lane as incomplete."""

    actual = _plan("actual")
    runtime = _runtime(active_plan=actual)
    assert runtime._native_geometry_evidence_lane.publish(_demand())
    runtime._bind_native_geometry_evidence_after_success(actual)

    runtime._complete_native_geometry_evidence(
        _native_b4_result(), _attestation(qualified=True), actual
    )

    lane = runtime.native_geometry_evidence_telemetry()
    assert lane["state"] == "idle"
    assert lane["counters"]["scheduled"] == 1
    assert lane["counters"]["satisfied"] == 1
    assert lane["outcome_trace"][-1] == {
        "plan_id": "actual",
        "phase": "completion",
        "reason": "satisfied",
        "attestation": {
            "qualified": True,
            "snapshot": {
                "map": 5,
                "topology": 5,
                "coordinate": 6,
                "schema": 7,
                "optimizer": 9,
                "surface": 9,
                "policy": 10,
                "parameter": 12,
            },
        },
    }


def test_non_null_target_candidate_without_a_qualified_attestation_remains_incomplete():
    """A returned target-shaped B4 result cannot satisfy without fence output."""

    for attestation in (
        SimpleNamespace(count=1),
        None,
        _attestation(qualified=False),
    ):
        actual = _plan("actual")
        runtime = _runtime(active_plan=actual)
        assert runtime._native_geometry_evidence_lane.publish(_demand())
        runtime._bind_native_geometry_evidence_after_success(actual)

        runtime._complete_native_geometry_evidence(
            _native_b4_result(), attestation, actual
        )

        lane = runtime.native_geometry_evidence_telemetry()
        assert lane["state"] == "idle"
        assert lane["counters"]["incomplete"] == 1
        assert lane["counters"]["satisfied"] == 0


def test_stale_or_superseded_target_attestation_cannot_satisfy_the_native_lane():
    """The attestation and completed E0 must name the same post-Adam vector."""

    for result, attestation in (
        (_native_b4_result(optimizer_version=8), _attestation(qualified=True)),
        (_native_b4_result(), _attestation(qualified=True, optimizer_version=8)),
        (_native_b4_result(), _attestation(qualified=True, map_version=4)),
    ):
        actual = _plan("actual")
        runtime = _runtime(active_plan=actual)
        assert runtime._native_geometry_evidence_lane.publish(_demand())
        runtime._bind_native_geometry_evidence_after_success(actual)

        runtime._complete_native_geometry_evidence(result, attestation, actual)

        lane = runtime.native_geometry_evidence_telemetry()
        assert lane["state"] == "idle"
        assert lane["counters"]["satisfied"] == 0
        assert lane["counters"]["stale"] == 1


def test_superseded_plan_cannot_complete_a_qualified_native_target_attestation():
    """A valid fence result belongs only to the plan that scheduled the lane."""

    actual = _plan("actual")
    runtime = _runtime(active_plan=actual)
    assert runtime._native_geometry_evidence_lane.publish(_demand())
    runtime._bind_native_geometry_evidence_after_success(actual)

    runtime._complete_native_geometry_evidence(
        _native_b4_result(), _attestation(qualified=True), _plan("superseded")
    )

    lane = runtime.native_geometry_evidence_telemetry()
    assert lane["state"] == "idle"
    assert lane["counters"]["satisfied"] == 0
    assert lane["counters"]["stale"] == 1


def test_parameter_only_post_adam_attestation_mismatch_cannot_satisfy_native_b4():
    """An attestation must name the E0 parameter version after exactly one Adam step."""

    actual = _plan("actual")
    runtime = _runtime(active_plan=actual)
    assert runtime._native_geometry_evidence_lane.publish(_demand())
    runtime._bind_native_geometry_evidence_after_success(actual)

    runtime._complete_native_geometry_evidence(
        _native_b4_result(),
        _attestation(qualified=True, parameter_version=11),
        actual,
    )

    lane = runtime.native_geometry_evidence_telemetry()
    assert lane["state"] == "idle"
    assert lane["counters"]["satisfied"] == 0
    assert lane["counters"]["stale"] == 1
