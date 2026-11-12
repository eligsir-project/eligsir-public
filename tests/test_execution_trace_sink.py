"""Contracts for the optional MappingSession execution-lineage trace."""

from __future__ import annotations

import numpy as np

from eligsir.contracts.control import EvidenceFact
from eligsir.contracts.runtime import (
    BufferRef,
    IngressAction,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.config import IngressTelemetrySnapshot
from eligsir.runtime.profiling.execution_trace import ExecutionTraceSink
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    observation_from_pose_matrix,
)


_INTRINSICS = CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4)


class _Clock:
    def __init__(self, timestamps: list[int]) -> None:
        self._timestamps = iter(timestamps)

    def __call__(self) -> int:
        return next(self._timestamps)


def _packet(ordinal: int, *, tracked: bool = True) -> SensorPacket:
    source = SourceIdentity("trace-camera", 0, ordinal, 1_000_000 + ordinal)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8), encoding="rgb8"),
        depth=None,
        intrinsics=_INTRINSICS,
        tracking=(
            observation_from_pose_matrix(source, np.eye(4), provider="test")
            if tracked
            else None
        ),
    )


def test_session_trace_correlates_terminal_fates_and_queue_stalls_by_fingerprint():
    sink = ExecutionTraceSink(monotonic_ns=_Clock([10, 15, 20, 28, 30, 39, 50, 65, 80, 82]))
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: None,
        execution_trace_sink=sink,
        ingress_telemetry_provider=lambda packet, _observation: IngressTelemetrySnapshot(
            (
                EvidenceFact("transport.queue_depth", packet.source.source_ordinal + 2),
                EvidenceFact("transport.age_ns", packet.source.source_ordinal * 100),
            )
        ),
    )

    delivered = session.submit(_packet(0))
    rejected = session.submit(_packet(1, tracked=False))
    session.submit(_packet(3))
    failed = session.submit(_packet(2))
    duplicate = session.submit(_packet(3))

    assert delivered.event.action is IngressAction.DELIVER
    assert rejected.event.action is IngressAction.REJECT
    assert failed.event.action is IngressAction.FAIL
    assert duplicate.event is not None

    records = sink.records()
    assert [record.action for record in records] == [
        "deliver",
        "reject",
        "deliver",
        "fail",
        "deliver",
    ]
    assert [record.duplicate for record in records] == [False, False, False, False, True]
    assert records[0].source_fingerprint == delivered.event.source.fingerprint()
    assert records[0].reason == "materialized and mapped"
    assert records[1].source_fingerprint == rejected.event.source.fingerprint()
    assert "not usable" in records[1].reason
    assert records[3].source_fingerprint == failed.event.source.fingerprint()
    assert "out of canonical order" in records[3].reason
    assert sink.records_for(_packet(3).source.fingerprint()) == (records[2], records[4])
    assert records[0].queue_depth == 2
    assert records[0].source_age_ns == 0
    assert records[3].queue_depth is None
    assert records[3].source_age_ns is None
    assert records[0].cpu_span_ns == 5
    assert records[1].terminal_gap_ns == 13
    assert records[2].terminal_gap_ns == 11
    assert records[2].queue_depth == 5
    assert records[2].source_age_ns == 300

    exported = sink.export()
    assert exported == tuple(record.as_dict() for record in records)
    assert all("payload" not in row for row in exported)


def test_session_without_trace_sink_does_not_allocate_trace_events(monkeypatch):
    import eligsir.runtime.profiling.execution_trace as trace_module

    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("trace event allocated while tracing is disabled")

    monkeypatch.setattr(trace_module, "ExecutionTraceRecord", fail_if_constructed)
    session = MappingSession(ExternalTrackingProvider(), mapper=lambda *_args: None)

    outcome = session.submit(_packet(0))

    assert outcome.accepted is True


def test_pending_oldest_age_is_queue_age_not_the_submitted_source_age():
    sink = ExecutionTraceSink(monotonic_ns=_Clock([10, 20]))
    session = MappingSession(
        ExternalTrackingProvider(),
        mapper=lambda *_args: None,
        execution_trace_sink=sink,
        ingress_telemetry_provider=lambda _packet, _observation: IngressTelemetrySnapshot(
            (
                EvidenceFact("transport.queue_depth", 2),
                EvidenceFact("pending_oldest_age_ns", 9_000),
            )
        ),
    )

    session.submit(_packet(7))

    record = sink.records()[0]
    assert record.queue_depth == 2
    assert record.queue_oldest_age_ns == 9_000
    assert record.source_age_ns is None
    assert "queue_oldest_age_ns" in record.as_dict()
