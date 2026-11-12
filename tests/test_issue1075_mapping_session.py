"""Contract tests for ``MappingSession`` (eligsir#1075/#1162).

Production construction is fail-closed: a session must have a mapping sink.
Mapperless operation exists only as explicit ``ingress_only=True``
qualification, and never increments ``mapped_count``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    IngressAction,
    SensorPacket,
    SourceIdentity,
    TrackingStatus,
)
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.session import (
    MappingSession,
    MappingSessionClosed,
    SubmitOutcome,
)
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    observation_from_pose_matrix,
)

_INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48)


def _identity(ordinal: int) -> SourceIdentity:
    return SourceIdentity(
        source_id="cam",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal,
    )


def _packet(ordinal: int, *, tracked: bool = True) -> SensorPacket:
    source = _identity(ordinal)
    tracking = (
        observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose")
        if tracked
        else None
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((48, 64, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((48, 64), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=_INTRINSICS,
        tracking=tracking,
    )


def _session(**kw) -> MappingSession:
    kw.setdefault("mapper", lambda _frame, _observation: None)
    return MappingSession(ExternalTrackingProvider(), **kw)


# -- construction / declared surface -------------------------------------


def test_production_session_requires_a_mapping_sink():
    with pytest.raises(ValueError, match="requires a mapping sink"):
        MappingSession(ExternalTrackingProvider())


def test_ingress_only_and_mapper_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        MappingSession(
            ExternalTrackingProvider(),
            mapper=lambda _frame, _observation: None,
            ingress_only=True,
        )


def test_session_exposes_submit_flush_close_surface():
    session = _session()
    for name in ("submit", "flush", "close"):
        assert callable(getattr(session, name))


def test_submit_returns_an_explicit_outcome_not_none():
    session = _session()
    outcome = session.submit(_packet(0))

    assert isinstance(outcome, SubmitOutcome)
    assert outcome.accepted is True
    assert outcome.sequence == 0
    assert outcome.event.action is IngressAction.DELIVER
    assert outcome.frame is not None


def test_mapper_receives_materialized_frame_in_arrival_order():
    seen = []
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, _observation: seen.append(frame.source.source_ordinal),
    )

    for i in (0, 1, 2):
        session.submit(_packet(i))

    assert seen == [0, 1, 2]
    assert session.accepted_count == 3
    assert session.mapped_count == 3


def test_packet_capture_precedes_consumed_observer_and_preserves_borrowed_rgb():
    """A transport may reuse its buffer as soon as the consumed callback returns."""

    packet = _packet(0)
    borrowed = packet.rgb.data
    borrowed[...] = 17
    seen = []
    consumed_packets = []

    def capture(value):
        owned = np.array(value.rgb.data, copy=True)
        owned.setflags(write=False)
        return replace(value, rgb=BufferRef.from_numpy(owned, encoding="rgb8"))

    def consumed(value):
        consumed_packets.append(value)
        borrowed[...] = 99

    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda frame, _observation: seen.append(frame.rgb.data.copy()),
        packet_capture=capture,
        source_consumed_observer=consumed,
    )

    assert session.submit(packet).accepted is True
    assert consumed_packets == [packet]
    assert int(seen[0][0, 0, 0]) == 17


def test_explicit_ingress_only_harness_never_claims_mapping():
    session = MappingSession(ExternalTrackingProvider(), ingress_only=True)

    outcome = session.submit(_packet(0))

    assert outcome.accepted is True
    assert outcome.frame is None
    assert session.accepted_count == 1
    assert session.mapped_count == 0
    assert "ingress-only" in outcome.event.reason
    assert "not mapped" in outcome.event.reason


# -- owner safety ---------------------------------------------------------


def test_untracked_packet_is_rejected_before_the_mapper():
    mapped = []
    session = MappingSession(
        ExternalTrackingProvider(), mapper=lambda frame, _observation: mapped.append(frame)
    )

    outcome = session.submit(_packet(0, tracked=False))

    assert outcome.accepted is False
    assert outcome.observation.status is TrackingStatus.INVALID
    assert outcome.event.action is IngressAction.REJECT
    assert mapped == []
    assert session.accepted_count == 0
    assert session.rejected_count == 1


def test_invalid_tracking_terminal_keeps_captured_packet_after_source_release():
    """Terminal pre-disposition paths retain owned bytes without inventing a pose."""

    packet = _packet(0, tracked=False)
    borrowed = packet.rgb.data
    borrowed[...] = 17
    terminal = []

    def capture(value):
        owned = np.array(value.rgb.data, copy=True)
        owned.setflags(write=False)
        return replace(value, rgb=BufferRef.from_numpy(owned, encoding="rgb8"))

    def consumed(_value):
        borrowed[...] = 99

    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: None,
        packet_capture=capture,
        terminal_packet_sink=lambda value, outcome: terminal.append((value, outcome)),
        source_consumed_observer=consumed,
    )

    outcome = session.submit(packet)

    assert outcome.accepted is False
    assert outcome.observation.status is TrackingStatus.INVALID
    assert len(terminal) == 1
    captured, terminal_outcome = terminal[0]
    assert terminal_outcome is outcome
    assert captured.tracking is None
    assert int(captured.rgb.data[0, 0, 0]) == 17


def test_require_tracking_can_be_disabled_for_an_explicit_ingress_test():
    session = MappingSession(
        ExternalTrackingProvider(), ingress_only=True, require_tracking=False
    )

    assert session.submit(_packet(0, tracked=False)).accepted is True
    assert session.accepted_count == 1
    assert session.mapped_count == 0


def test_events_are_a_read_only_snapshot():
    session = _session()
    session.submit(_packet(0))
    events = session.events()

    assert isinstance(events, tuple)
    with pytest.raises((AttributeError, TypeError)):
        events[0].reason = "tampered"
    assert len(session.events()) == 1


def test_event_ring_is_bounded():
    session = _session(max_events=4)
    for i in range(20):
        session.submit(_packet(i))
    assert len(session.events()) == 4


def test_provider_returning_a_foreign_identity_fails_the_submit():
    class _Liar:
        provider_name = "liar"

        def observe(self, packet, *, predicted_pose=None):
            return observation_from_pose_matrix(_identity(999), np.eye(4), provider="liar")

        def close(self):
            pass

    outcome = MappingSession(
        _Liar(), mapper=lambda _frame, _observation: None
    ).submit(_packet(0))

    assert outcome.accepted is False
    assert outcome.event.action is IngressAction.FAIL


# -- lifecycle ------------------------------------------------------------


def test_close_is_idempotent_and_closes_the_provider_once():
    closes = []

    class _Counting(ExternalTrackingProvider):
        def close(self):
            closes.append(1)

    session = MappingSession(_Counting(), mapper=lambda _frame, _observation: None)
    session.close()
    session.close()

    assert closes == [1]
    assert session.closed is True


def test_submit_after_close_fails_closed():
    session = _session()
    session.close()

    with pytest.raises(MappingSessionClosed):
        session.submit(_packet(0))


def test_flush_does_not_close_the_session():
    session = _session()
    session.submit(_packet(0))

    assert session.flush() == 1
    assert session.closed is False
    assert session.submit(_packet(1)).accepted is True


def test_ingress_only_flush_reports_zero_mapped():
    session = MappingSession(ExternalTrackingProvider(), ingress_only=True)
    session.submit(_packet(0))

    assert session.flush() == 0
    assert session.accepted_count == 1


def test_context_manager_closes():
    with _session() as session:
        session.submit(_packet(0))
    assert session.closed is True


# -- determinism ----------------------------------------------------------


def test_same_trace_produces_the_same_event_sequence():
    def run():
        session = _session()
        for i in range(6):
            session.submit(_packet(i, tracked=(i % 2 == 0)))
        return [
            (e.source.fingerprint(), e.action, e.details["sequence"])
            for e in session.events()
        ]

    assert run() == run()


def test_sequence_increments_for_rejected_packets_too():
    session = _session()

    first = session.submit(_packet(0, tracked=False))
    second = session.submit(_packet(1))

    assert first.sequence == 0
    assert second.sequence == 1
    assert session.submitted_count == 2
    assert session.accepted_count == 1
    assert session.mapped_count == 1
