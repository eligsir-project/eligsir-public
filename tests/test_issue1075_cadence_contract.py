"""MappingSession/application cadence contract (#1075/#1162).

The session owns ordered frame delivery. The caller/application owns one
optimizer tick per iteration, including iterations that drain zero frames.
"""

from __future__ import annotations

from typing import List, Optional
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.driver import (
    DEFAULT_DRAIN_CAP,
    CadenceViolation,
    IterationTick,
    check_cadence_invariant,
    drain_tick,
    total_mapped,
)
from eligsir.runtime.session import MappingSession
from eligsir.runtime.prepared_ingress import PreparedIngressDeferred
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    observation_from_pose_matrix,
)

_INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48)


def _packet(ordinal: int, *, tracked: bool = True) -> SensorPacket:
    source = SourceIdentity(
        source_id="cam",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal,
    )
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


class _FakeSource:
    def __init__(self, n: int, *, ready: Optional[List[int]] = None, skip_ordinals=()):
        self._pending = list(range(n))
        self._ready_schedule = ready
        self._skip = set(skip_ordinals)
        self.tick = 0
        self._consumed_this_tick = 0

    def ready_count(self) -> int:
        if self._ready_schedule is None:
            return len(self._pending)
        allowed = self._ready_schedule[self.tick] if self.tick < len(self._ready_schedule) else 0
        return min(max(0, allowed - self._consumed_this_tick), len(self._pending))

    def begin_tick(self, tick: int) -> None:
        self.tick = tick
        self._consumed_this_tick = 0

    def has_ready_frame(self) -> bool:
        return self.ready_count() > 0

    def next_packet(self) -> Optional[SensorPacket]:
        ordinal = self._pending.pop(0)
        self._consumed_this_tick += 1
        if ordinal in self._skip:
            return None
        return _packet(ordinal)


def _session(**kw) -> MappingSession:
    kw.setdefault("mapper", lambda _frame, _observation: None)
    return MappingSession(ExternalTrackingProvider(), **kw)


def _run(source: _FakeSource, *, iterations: int, drain_cap=DEFAULT_DRAIN_CAP, session=None):
    session = session or _session()
    ticks: List[IterationTick] = []
    optimizer_steps = 0
    for iteration in range(1, iterations + 1):
        source.begin_tick(iteration)
        ticks.append(
            drain_tick(
                session,
                iteration=iteration,
                has_ready_frame=source.has_ready_frame,
                next_packet=source.next_packet,
                drain_cap=drain_cap,
            )
        )
        optimizer_steps += 1
    return session, ticks, optimizer_steps


def test_zero_frame_tick_is_normal_not_an_error():
    source = _FakeSource(2)
    session, ticks, optimizer_steps = _run(source, iterations=6)

    assert any(tick.submitted == 0 for tick in ticks)
    assert optimizer_steps == 6
    assert [tick.iteration for tick in ticks] == [1, 2, 3, 4, 5, 6]
    check_cadence_invariant(ticks, expect_zero_frame_ticks=True)
    assert session.mapped_count == 2


def test_optimizer_cadence_is_one_per_tick_regardless_of_frames_drained():
    source = _FakeSource(9)
    _, ticks, optimizer_steps = _run(source, iterations=10, drain_cap=2)

    assert optimizer_steps == 10
    assert total_mapped(ticks) == 9
    assert sorted({tick.submitted for tick in ticks}) == [0, 1, 2]
    assert len(ticks) == optimizer_steps


def test_never_more_than_drain_cap_submits_per_tick():
    for cap in (1, 2, 4):
        source = _FakeSource(20)
        _, ticks, _ = _run(source, iterations=30, drain_cap=cap)
        assert max(tick.submitted for tick in ticks) <= cap
        check_cadence_invariant(ticks, drain_cap=cap)


def test_drain_cap_below_one_is_clamped_like_the_real_loop():
    source = _FakeSource(3)
    _, ticks, _ = _run(source, iterations=3, drain_cap=0)
    assert total_mapped(ticks) == 3
    assert max(tick.submitted for tick in ticks) == 1


def test_surplus_frames_are_deferred_not_dropped():
    source = _FakeSource(7)
    session, ticks, _ = _run(source, iterations=10, drain_cap=2)

    assert any(tick.deferred for tick in ticks)
    assert total_mapped(ticks) == 7
    assert session.mapped_count == 7


def test_sequences_are_arrival_ordered_and_gapless_across_ticks():
    source = _FakeSource(6)
    _, ticks, _ = _run(source, iterations=6, drain_cap=2)

    flat = [seq for tick in ticks for seq in tick.sequences]
    assert flat == list(range(6))
    check_cadence_invariant(ticks)


def test_reordered_submission_is_rejected():
    ticks = [
        IterationTick(iteration=1, submitted=2, mapped=2, sequences=(0, 1)),
        IterationTick(iteration=2, submitted=2, mapped=2, sequences=(3, 2)),
    ]
    with pytest.raises(CadenceViolation, match="arrival order"):
        check_cadence_invariant(ticks)


def test_exceeding_the_cap_is_rejected():
    ticks = [IterationTick(iteration=1, submitted=5, mapped=5, sequences=(0, 1, 2, 3, 4))]
    with pytest.raises(CadenceViolation, match="drain_cap"):
        check_cadence_invariant(ticks, drain_cap=2)


def test_stalled_iteration_clock_is_rejected():
    ticks = [
        IterationTick(iteration=4, submitted=0, mapped=0),
        IterationTick(iteration=4, submitted=0, mapped=0),
    ]
    with pytest.raises(CadenceViolation, match="iteration clock"):
        check_cadence_invariant(ticks)


def test_a_suite_that_never_hits_a_zero_frame_tick_is_flagged():
    ticks = [IterationTick(iteration=1, submitted=1, mapped=1, sequences=(0,))]
    with pytest.raises(CadenceViolation, match="zero-frame tick"):
        check_cadence_invariant(ticks, drain_cap=2, expect_zero_frame_ticks=True)


def test_a_policy_skip_consumes_a_drain_unit():
    source = _FakeSource(4, skip_ordinals={0})
    session = _session()
    source.begin_tick(1)
    tick = drain_tick(
        session,
        iteration=1,
        has_ready_frame=source.has_ready_frame,
        next_packet=source.next_packet,
        drain_cap=2,
    )

    assert tick.skipped_by_policy == 1
    assert tick.submitted == 1
    assert tick.consumed == 2
    check_cadence_invariant([tick], drain_cap=2)


def test_prepared_retention_defer_retries_the_same_packet_before_later_sources():
    class Session:
        def __init__(self):
            self.packets = []

        def submit(self, packet):
            self.packets.append(packet)
            if len(self.packets) == 1:
                raise PreparedIngressDeferred("spill pending")
            return SimpleNamespace(accepted=True, sequence=len(self.packets) - 2)

    packets = [_packet(0), _packet(1)]
    pending: list[SensorPacket] = []
    session = Session()
    first = drain_tick(
        session,
        iteration=1,
        has_ready_frame=lambda: bool(packets),
        next_packet=lambda: packets.pop(0),
        pending_packet=pending,
    )
    assert first.submitted == 0 and first.deferred == 1
    assert pending == [session.packets[0]]
    second = drain_tick(
        session,
        iteration=2,
        has_ready_frame=lambda: bool(packets),
        next_packet=lambda: packets.pop(0),
        pending_packet=pending,
    )
    assert [packet.source.source_ordinal for packet in session.packets] == [0, 0, 1]
    assert session.packets[0] is session.packets[1]
    assert second.submitted == 2 and pending == []


def test_consumption_over_the_cap_is_rejected_even_when_submits_are_not():
    ticks = [
        IterationTick(
            iteration=1, submitted=2, mapped=2, skipped_by_policy=3, sequences=(0, 1)
        )
    ]
    with pytest.raises(CadenceViolation, match="consumed"):
        check_cadence_invariant(ticks, drain_cap=2)


def test_policy_skipped_frames_consume_no_sequence():
    source = _FakeSource(6, skip_ordinals={1, 3})
    session, ticks, _ = _run(source, iterations=8, drain_cap=2)

    assert sum(tick.skipped_by_policy for tick in ticks) == 2
    assert session.submitted_count == 4
    flat = [seq for tick in ticks for seq in tick.sequences]
    assert flat == list(range(4))
    check_cadence_invariant(ticks, drain_cap=2)


def test_untracked_frame_is_rejected_but_consumes_a_sequence():
    session = _session()
    outcome = session.submit(_packet(0, tracked=False))

    assert outcome.accepted is False
    assert outcome.sequence == 0
    assert session.submitted_count == 1
    assert session.mapped_count == 0


def test_close_is_explicit_and_the_driver_owns_when():
    source = _FakeSource(2)
    session, _, _ = _run(source, iterations=3)
    assert session.flush() == 2
    assert session.closed is False
    session.close()
    session.close()
    assert session.closed is True


def test_a_release_gate_ends_tick_without_consuming_frame():
    source = _FakeSource(4)
    session = _session()
    source.begin_tick(1)
    tick = drain_tick(
        session,
        iteration=1,
        has_ready_frame=source.has_ready_frame,
        next_packet=source.next_packet,
        drain_cap=2,
        should_release=lambda _iteration, _dt: False,
    )
    assert tick.submitted == 0
    assert session.submitted_count == 0
