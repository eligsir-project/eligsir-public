from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    IngressAction,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.runtime.session import MappingSession


class _CountingTrackingProvider:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.closed = False

    def observe(self, packet: SensorPacket) -> TrackingObservation:
        self.observe_calls += 1
        assert packet.tracking is not None
        return packet.tracking

    def close(self) -> None:
        self.closed = True


def _packet(ordinal: int) -> SensorPacket:
    source = SourceIdentity(
        source_id="camera0",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )
    tracking = TrackingObservation(
        source=source,
        pose=Pose(np.eye(4, dtype=np.float32)),
        status=TrackingStatus.VALID,
        provider="test",
        confidence=1.0,
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), encoding="rgb8"),
        depth=None,
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4),
        tracking=tracking,
    )


def test_exact_duplicate_reuses_its_single_persisted_terminal_fate():
    provider = _CountingTrackingProvider()
    mapped = []
    session = MappingSession(provider, mapper=lambda frame, observation: mapped.append(frame))
    packet = _packet(10)

    first = session.submit(packet)
    duplicate = session.submit(packet)

    assert first.accepted is True
    assert duplicate is first
    assert duplicate.event.action is IngressAction.DELIVER
    assert provider.observe_calls == 1
    assert len(mapped) == 1
    assert session.accepted_count == 1
    assert session.mapped_count == 1
    assert session.rejected_count == 0
    assert len(session.events()) == 1


def test_older_identity_is_out_of_order_failure_not_duplicate():
    provider = _CountingTrackingProvider()
    session = MappingSession(provider, mapper=lambda _frame, _observation: None)

    assert session.submit(_packet(10)).accepted is True
    older = session.submit(_packet(9))

    assert older.accepted is False
    assert older.event.action is IngressAction.FAIL
    assert "out of canonical order" in older.event.reason
    assert provider.observe_calls == 1
    assert session.accepted_count == 1
    assert session.mapped_count == 1


def test_duplicate_identity_with_different_packet_metadata_fails_closed():
    provider = _CountingTrackingProvider()
    session = MappingSession(provider, mapper=lambda _frame, _observation: None)
    packet = _packet(10)
    session.submit(packet)
    conflicting = SensorPacket(
        source=packet.source,
        rgb=BufferRef.from_numpy(
            np.zeros((8, 8, 3), dtype=np.uint8), encoding="rgb8"
        ),
        depth=None,
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=4.0, cy=4.0, width=8, height=8),
        tracking=packet.tracking,
    )

    with pytest.raises(ValueError, match="metadata conflicts"):
        session.submit(conflicting)

    assert provider.observe_calls == 1
