"""Integration contracts for #1190 at the canonical session boundary."""

from __future__ import annotations

import numpy as np

from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    EvidenceFact,
    FreshnessMode,
)
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.mapping.admission.observation_owner import ObservationOwner
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureDecision,
)
from eligsir.runtime.session import MappingSession
from eligsir.runtime.config import IngressTelemetrySnapshot
from eligsir.runtime.application import (
    MappingApplicationOptions,
    create_mapping_application,
)
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    observation_from_pose_matrix,
)


def _packet(ordinal: int = 0) -> SensorPacket:
    source = SourceIdentity("camera", 0, ordinal, 1_000_000_000 + ordinal)
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="external")
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((4, 4), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(
            fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4
        ),
        tracking=tracking,
    )


def _snapshot(packet, observation, *, snapshot_id: int = 7) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=snapshot_id,
        control_version=1,
        source=packet.source,
        tracking=observation,
        pending_sources=(packet.source,),
        newest_frontier=packet.source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(1, 0, 1, 1, 1, 1 << 20),
        supervision_candidates=(packet.source,),
    )


def _owner(action: BackpressureAction) -> ObservationOwner:
    return ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=action,
            reason=(
                "soft_all_capacity_available"
                if action is BackpressureAction.ADMIT
                else "skipped_capacity_exhausted"
            ),
            admit_supervision=action is BackpressureAction.ADMIT,
            allow_geometry_insertion=action is BackpressureAction.ADMIT,
        )
    )


def test_declined_disposition_stops_before_materialization_and_mapper(monkeypatch):
    calls: list[str] = []

    class Tracking(ExternalTrackingProvider):
        def observe(self, packet, *, predicted_pose=None):
            calls.append("tracking")
            return super().observe(packet, predicted_pose=predicted_pose)

    def snapshot_provider(packet, observation, _ingress):
        calls.append("snapshot")
        return _snapshot(packet, observation)

    def poison_materializer(*_args, **_kwargs):
        raise AssertionError("declined source reached materialization")

    monkeypatch.setattr(
        "eligsir.runtime.session.materialize_mapping_frame", poison_materializer
    )
    session = MappingSession(
        Tracking(),
        mapper=lambda *_args: calls.append("mapper"),
        observation_owner=_owner(BackpressureAction.SKIP_CAPACITY),
        control_snapshot_provider=snapshot_provider,
    )

    outcome = session.submit(_packet())

    assert calls == ["tracking", "snapshot"]
    assert outcome.accepted is True
    assert outcome.disposition.kind is DispositionKind.SKIP_CAPACITY
    assert outcome.frame is None
    assert session.accepted_count == 1
    assert session.mapped_count == 0


def test_admitted_disposition_materializes_and_maps_once():
    calls: list[str] = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: calls.append("mapper"),
        observation_owner=_owner(BackpressureAction.ADMIT),
        control_snapshot_provider=lambda packet, observation, _ingress: _snapshot(packet, observation),
    )

    outcome = session.submit(_packet())

    assert calls == ["mapper"]
    assert outcome.disposition.kind is DispositionKind.ADMIT
    assert outcome.frame is not None
    assert session.mapped_count == 1


def test_owner_turns_rejected_modality_evidence_into_a_quality_skip_before_materialization():
    calls: list[str] = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: calls.append("mapper"),
        observation_owner=_owner(BackpressureAction.ADMIT),
        control_snapshot_provider=lambda packet, observation, _ingress: _snapshot(
            packet,
            observation,
        ).__class__(
            **{
                field: getattr(_snapshot(packet, observation), field)
                for field in _snapshot(packet, observation).__dataclass_fields__
            }
            | {
                "evidence": (
                    EvidenceFact("rgb_supervision_eligible", False),
                    EvidenceFact("rgb_quality", "blurred"),
                    EvidenceFact("depth_supervision_eligible", False),
                    EvidenceFact("depth_quality", "missing"),
                    EvidenceFact("depth_insertion_eligible", False),
                )
            }
        ),
    )

    outcome = session.submit(_packet())

    assert calls == []
    assert outcome.frame is None
    assert outcome.disposition.kind is DispositionKind.SKIP_QUALITY


def test_control_owner_and_snapshot_provider_must_be_injected_together():
    try:
        MappingSession(
            ExternalTrackingProvider(),
            mapper=lambda *_args: None,
            observation_owner=_owner(BackpressureAction.ADMIT),
        )
    except ValueError as exc:
        assert "injected together" in str(exc)
    else:
        raise AssertionError("partial control-owner injection was accepted")


def test_application_composition_preserves_the_single_control_owner():
    owner = _owner(BackpressureAction.ADMIT)
    provider = lambda packet, observation, _ingress: _snapshot(packet, observation)
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda *_args: None,
            optimizer_tick=lambda: None,
            observation_owner=owner,
            control_snapshot_provider=provider,
        )
    )

    outcome = application.session.submit(_packet())

    assert outcome.disposition.kind is DispositionKind.ADMIT
    assert application.session.mapped_count == 1


def test_session_publishes_terminal_disposition_before_mapping():
    calls: list[tuple[str, object]] = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, *_args: calls.append(("mapper", frame.source)),
        observation_owner=_owner(BackpressureAction.ADMIT),
        control_snapshot_provider=lambda packet, observation, _ingress: _snapshot(packet, observation),
        disposition_sink=lambda disposition: calls.append(
            ("disposition", disposition.source)
        ),
    )

    session.submit(_packet())

    assert [name for name, _value in calls] == ["disposition", "mapper"]


def test_immutable_ingress_telemetry_is_merged_into_the_decided_control_snapshot():
    seen = []
    ingress_seen = []
    owner = ObservationOwner(
        lambda snapshot: (
            seen.append(snapshot)
            or BackpressureDecision(
                frame_id=snapshot.source.source_ordinal,
                action=BackpressureAction.ADMIT,
                reason="soft_all_capacity_available",
                admit_supervision=True,
                allow_geometry_insertion=True,
            )
        )
    )
    ingress = IngressTelemetrySnapshot(
        (EvidenceFact("transport.drop_count", 3), EvidenceFact("transport.age_ns", 5))
    )

    def snapshot_provider(packet, observation, ingress_telemetry):
        ingress_seen.append(ingress_telemetry)
        base = _snapshot(packet, observation)
        return ControlSnapshot(
            **{
                field: getattr(base, field)
                for field in base.__dataclass_fields__
            }
            | {"evidence": ingress_telemetry.evidence}
        )

    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: None,
        observation_owner=owner,
        control_snapshot_provider=snapshot_provider,
        ingress_telemetry_provider=lambda _packet, _observation: ingress,
    )

    session.submit(_packet())

    assert ingress_seen == [ingress]
    assert {fact.name for fact in seen[0].evidence} == {
        "transport.drop_count",
        "transport.age_ns",
    }


def test_snapshot_provider_must_include_every_ingress_fact_during_construction():
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: None,
        observation_owner=_owner(BackpressureAction.ADMIT),
        control_snapshot_provider=lambda packet, observation, _ingress: _snapshot(
            packet, observation
        ),
        ingress_telemetry_provider=lambda _packet, _observation: IngressTelemetrySnapshot(
            (EvidenceFact("transport.drop_count", 3),)
        ),
    )

    try:
        session.submit(_packet())
    except ValueError as exc:
        assert "must include immutable ingress telemetry" in str(exc)
    else:
        raise AssertionError("snapshot provider omitted ingress evidence without failing")
