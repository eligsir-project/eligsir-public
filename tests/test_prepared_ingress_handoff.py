from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    IngressAction,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.types import CameraIntrinsics
from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    FreshnessMode,
)
from eligsir.mapping.admission.observation_owner import ObservationOwner
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureDecision,
)
from eligsir.runtime.prepared_ingress import (
    PreparedIngressDeferred,
    PreparedIngressItem,
    PreparedIngressOfferResult,
    PreparedIngressReservation,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import ExternalTrackingProvider, observation_from_pose_matrix


INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48)


def identity(ordinal: int) -> SourceIdentity:
    return SourceIdentity("cam", 0, ordinal, 1_000_000_000 + ordinal)


def packet(ordinal: int) -> SensorPacket:
    source = identity(ordinal)
    tracking = observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose")
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((48, 64, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((48, 64), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=INTRINSICS,
        tracking=tracking,
    )


def control_snapshot(packet: SensorPacket, observation) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=1,
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


def test_prepared_mode_enqueues_accepted_source_without_calling_mapper():
    handoff = PreparedSourceIngressHandoff(capacity=2)
    mapped = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, _observation: mapped.append(frame),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
    )

    outcome = session.submit(packet(0))

    assert outcome.accepted is True
    assert outcome.frame is None
    assert mapped == []
    assert session.mapped_count == 0
    item = handoff.try_poll()
    assert item is not None
    assert item.sequence == outcome.sequence
    assert item.packet.source == identity(0)
    assert item.observation.source == identity(0)


def test_retention_capacity_defers_before_semantic_commit_and_retries_same_packet():
    """Capacity pressure keeps the original packet pending, not terminally rejected."""
    allowed = {"value": False}
    observed = []
    committed = []
    consumed = []
    handoff = PreparedSourceIngressHandoff(
        capacity=1,
        admission_preflight=lambda _packet: allowed["value"],
        commit_sink=lambda item: committed.append(item.packet.source),
    )
    class StatefulTracking:
        provider_name = "stateful-test"

        def observe(self, source_packet, *, predicted_pose=None):
            observed.append(source_packet.source)
            return observation_from_pose_matrix(
                source_packet.source,
                np.eye(4, dtype=np.float32),
                provider=self.provider_name,
            )

        def close(self):
            pass

    session = MappingSession(
        StatefulTracking(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
        source_consumed_observer=lambda packet: consumed.append(packet.source),
    )

    source_packet = packet(0)
    with pytest.raises(PreparedIngressDeferred):
        session.submit(source_packet)

    assert session.submitted_count == 0
    assert session.accepted_count == 0
    assert session.events() == ()
    assert handoff.depth == 0
    assert committed == []
    assert consumed == []
    assert observed == []

    allowed["value"] = True
    outcome = session.submit(source_packet)
    assert outcome.accepted is True
    assert outcome.sequence == 0
    assert committed == [source_packet.source]
    assert consumed == [source_packet.source]
    assert observed == [source_packet.source]
    queued = handoff.try_poll()
    assert queued.packet.source == source_packet.source
    np.testing.assert_array_equal(
        queued.observation.pose.matrix, source_packet.tracking.pose.matrix
    )


def test_prepared_handoff_preserves_session_order_and_control_is_single_pass():
    observed = []
    control_calls = []

    class Tracking:
        provider_name = "recording"

        def observe(self, source_packet, *, predicted_pose=None):
            observed.append(source_packet.source.source_ordinal)
            return source_packet.tracking

        def close(self):
            pass

    class Control:
        def __init__(self):
            self.owner = ObservationOwner(
                lambda snapshot: BackpressureDecision(
                    frame_id=snapshot.source.source_ordinal,
                    action=BackpressureAction.ADMIT,
                    reason="soft_all_capacity_available",
                    admit_supervision=True,
                    allow_geometry_insertion=True,
                )
            )

        def decide(self, snapshot):
            control_calls.append(("decide", snapshot.source.source_ordinal))
            return self.owner.decide(snapshot)

        def permit(self, disposition):
            control_calls.append(("permit", disposition.source.source_ordinal))
            return self.owner.permit(disposition)

    handoff = PreparedSourceIngressHandoff(capacity=3)
    session = MappingSession(
        Tracking(),
        mapper=lambda *_: pytest.fail("prepared mode called mapper synchronously"),
        observation_owner=Control(),
        control_snapshot_provider=lambda source_packet, observation, _ingress: control_snapshot(
            source_packet, observation
        ),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
    )

    for ordinal in (0, 1, 2):
        session.submit(packet(ordinal))

    queued = [handoff.try_poll() for _ in range(3)]
    assert observed == [0, 1, 2]
    assert control_calls == [
        (kind, ordinal)
        for ordinal in (0, 1, 2)
        for kind in ("decide", "permit")
    ]
    assert [item.packet.source.source_ordinal for item in queued if item] == [0, 1, 2]


def test_prepared_handoff_keeps_each_accepted_control_token_with_its_source():
    handoff = PreparedSourceIngressHandoff(capacity=2)
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        )
    )
    session = MappingSession(
        ExternalTrackingProvider(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
        observation_owner=owner,
        control_snapshot_provider=lambda source_packet, observation, _ingress: control_snapshot(
            source_packet, observation
        ),
        disposition_sink=lambda _disposition: None,
    )

    assert session.submit(packet(0)).accepted
    assert session.submit(packet(1)).accepted
    first = handoff.try_poll()
    second = handoff.try_poll()

    assert first is not None and second is not None
    assert first.control_token is not None and second.control_token is not None
    assert (first.control_token.sequence, first.control_token.source) == (0, identity(0))
    assert (second.control_token.sequence, second.control_token.source) == (1, identity(1))
    assert first.control_token.disposition is first.disposition
    assert second.control_token.disposition is second.disposition


def test_reentrant_disposition_sink_cannot_steal_a_reserved_prepared_handoff_slot():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        )
    )
    reentrant = []
    session = None

    def sink(disposition):
        if disposition.source == identity(0):
            assert session is not None
            reentrant.append(session.submit(packet(1)))

    session = MappingSession(
        ExternalTrackingProvider(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
        observation_owner=owner,
        control_snapshot_provider=lambda source_packet, observation, _ingress: control_snapshot(
            source_packet, observation
        ),
        disposition_sink=sink,
    )

    outer = session.submit(packet(0))

    assert outer.accepted is True
    assert len(reentrant) == 1
    assert reentrant[0].accepted is False
    assert "queue_full" in reentrant[0].event.reason
    queued = handoff.try_poll()
    assert queued is not None
    assert queued.packet.source == identity(0)


def test_prepared_reservation_releases_when_disposition_sink_raises():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        )
    )
    session = MappingSession(
        ExternalTrackingProvider(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
        observation_owner=owner,
        control_snapshot_provider=lambda source_packet, observation, _ingress: control_snapshot(
            source_packet, observation
        ),
        disposition_sink=lambda _disposition: (_ for _ in ()).throw(RuntimeError("sink failed")),
    )

    with pytest.raises(RuntimeError, match="sink failed"):
        session.submit(packet(0))

    assert handoff.depth == 0
    next_packet = packet(1)
    assert handoff.try_offer(
        PreparedIngressItem(1, next_packet, next_packet.tracking)
    ) is PreparedIngressOfferResult.ENQUEUED


def test_prepared_reservation_commits_after_handoff_closes_during_control():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        )
    )
    session = MappingSession(
        ExternalTrackingProvider(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
        observation_owner=owner,
        control_snapshot_provider=lambda source_packet, observation, _ingress: control_snapshot(
            source_packet, observation
        ),
        disposition_sink=lambda _disposition: handoff.close(),
    )

    assert session.submit(packet(0)).accepted
    assert handoff.try_poll().packet.source == identity(0)
    next_packet = packet(1)
    assert handoff.try_offer(
        PreparedIngressItem(1, next_packet, next_packet.tracking)
    ) is PreparedIngressOfferResult.DROPPED_QUEUE_CLOSED


def test_prepared_handoff_reservation_releases_capacity_without_committing():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    first_packet = packet(0)
    reservation = handoff.reserve(
        PreparedIngressItem(0, first_packet, first_packet.tracking)
    )

    assert isinstance(reservation, PreparedIngressReservation)
    assert handoff.depth == 1
    reservation.cancel()
    assert handoff.depth == 0
    next_packet = packet(1)
    assert handoff.try_offer(
        PreparedIngressItem(1, next_packet, next_packet.tracking)
    ) is PreparedIngressOfferResult.ENQUEUED


def test_prepared_mode_can_omit_the_legacy_mapper():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    session = MappingSession(
        ExternalTrackingProvider(),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
    )

    outcome = session.submit(packet(0))

    assert outcome.accepted is True
    assert session.has_mapper is False
    assert handoff.try_poll() is not None


def test_legacy_mode_still_materializes_and_calls_mapper():
    mapped = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, _observation: mapped.append(frame.source.source_ordinal),
    )

    outcome = session.submit(packet(0))

    assert outcome.accepted is True
    assert outcome.frame is not None
    assert mapped == [0]
    assert session.mapped_count == 1


def test_prepared_handoff_saturation_fails_closed_and_attributes_drop():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    mapped = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, _observation: mapped.append(frame),
        prepared_mode=True,
        prepared_ingress_handoff=handoff,
    )

    first = session.submit(packet(0))
    second = session.submit(packet(1))

    assert first.accepted is True
    assert second.accepted is False
    assert second.frame is None
    assert second.event.action is IngressAction.REJECT
    assert "queue_full" in second.event.reason
    assert mapped == []
    assert handoff.dropped_by_reason["queue_full"] == 1
    metrics = handoff.metrics()
    assert (metrics.capacity, metrics.depth) == (1, 1)
    assert (metrics.enqueued_count, metrics.dequeued_count, metrics.dropped_count) == (
        1,
        0,
        1,
    )


def test_closed_handoff_fails_closed_without_discarding_accepted_items():
    handoff = PreparedSourceIngressHandoff(capacity=2)
    first_packet = packet(0)
    second_packet = packet(1)

    assert (
        handoff.try_offer(PreparedIngressItem(0, first_packet, first_packet.tracking))
        is PreparedIngressOfferResult.ENQUEUED
    )
    handoff.close()

    assert (
        handoff.try_offer(PreparedIngressItem(1, second_packet, second_packet.tracking))
        is PreparedIngressOfferResult.DROPPED_QUEUE_CLOSED
    )
    assert handoff.try_poll().packet.source == identity(0)
    assert handoff.try_poll() is None
    assert handoff.dropped_by_reason["queue_closed"] == 1


def test_handoff_rejects_duplicate_and_out_of_order_sources():
    handoff = PreparedSourceIngressHandoff(capacity=3)
    first_packet = packet(1)
    first_observation = first_packet.tracking
    first = PreparedIngressItem(4, first_packet, first_observation)
    duplicate = PreparedIngressItem(5, first_packet, first_observation)
    older_packet = packet(0)
    older = PreparedIngressItem(6, older_packet, older_packet.tracking)

    assert handoff.try_offer(first) is PreparedIngressOfferResult.ENQUEUED
    assert handoff.try_offer(duplicate) is PreparedIngressOfferResult.DROPPED_DUPLICATE_SOURCE
    assert handoff.try_offer(older) is PreparedIngressOfferResult.DROPPED_OUT_OF_ORDER
    assert handoff.dropped_by_reason["duplicate_source"] == 1
    assert handoff.dropped_by_reason["out_of_order"] == 1


def test_handoff_claims_exclusive_producer_and_consumer_roles():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    producer = handoff.claim_producer(owner="mapping-session")
    consumer = handoff.claim_consumer(owner="gpu-owner")

    with pytest.raises(RuntimeError, match="producer ownership"):
        handoff.claim_producer(owner="another-session")
    with pytest.raises(RuntimeError, match="consumer ownership"):
        handoff.claim_consumer(owner="another-gpu-owner")

    source_packet = packet(0)
    assert producer.offer(
        PreparedIngressItem(0, source_packet, source_packet.tracking)
    ) is PreparedIngressOfferResult.ENQUEUED
    assert consumer.poll().packet.source == identity(0)


def test_prepared_mode_requires_explicit_bounded_handoff():
    with pytest.raises(ValueError, match="prepared_ingress_handoff"):
        MappingSession(
            ExternalTrackingProvider(),
            mapper=lambda *_: None,
            prepared_mode=True,
        )


def test_prepared_item_rejects_mismatched_identity_before_handoff():
    source_packet = packet(0)
    foreign_observation = observation_from_pose_matrix(identity(1), np.eye(4), provider="other")

    with pytest.raises(ValueError, match="identities differ"):
        PreparedIngressItem(0, source_packet, foreign_observation)
