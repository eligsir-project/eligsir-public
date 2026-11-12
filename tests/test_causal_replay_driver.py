from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

from eligsir.contracts.control import FreshnessMode, SupervisionConstellation
from eligsir.contracts.runtime import BufferOwnership, BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import MappingApplicationOptions, create_mapping_application
from eligsir.runtime.causal_replay import (
    CausalReplayDriver,
    CausalIngressTelemetryState,
    CausalReplayTickPolicy,
    DatasetPacketAdapter,
    JsonlDecisionTraceObserver,
    PacedCausalReplayDriver,
    StreamingFramePacketAdapter,
    decision_trace_row,
)
from eligsir.runtime.causal_replay import _MappingDrainTracker
from eligsir.runtime.causal_replay import _prepared_ready_progress_fields
from eligsir.io.sources.decode_cache import StreamingRGBDFrame
from eligsir.mapping.population_shadow_runtime import (
    PopulationShadowDropReason,
    PopulationShadowSummary,
)
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.prepared_ingress import PreparedIngressDeferred
from eligsir.tracking.providers import ExternalTrackingProvider, observation_from_pose_matrix
from eligsir.utils.coverage_ledger import CoverageLedger


@pytest.mark.parametrize("paced", [False, True])
def test_evaluation_source_complete_waits_for_prepared_bootstrap(paced):
    """The production completion gate must not publish a pending endpoint."""

    from eligsir.runtime.causal_entry import _SourceCompleteEvaluationGate

    class _Application:
        source_complete = False

        def mark_source_complete(self):
            if self.source_complete:
                return SimpleNamespace(state=SimpleNamespace(value="bootstrapped"))
            return SimpleNamespace(
                state=SimpleNamespace(value="bootstrap_pending")
            )

    class _Evaluation:
        def __init__(self):
            self.completed = []

        def source_complete(self, tick):
            self.completed.append(tick)

    application = _Application()
    evaluation = _Evaluation()
    gate = _SourceCompleteEvaluationGate(application, evaluation)
    pending = SimpleNamespace(state=SimpleNamespace(value="bootstrap_pending"))
    complete = SimpleNamespace(state=SimpleNamespace(value="bootstrapped"))

    gate.source_complete(pending)
    assert evaluation.completed == []
    application.source_complete = True
    gate.observe_tick(complete)
    gate.observe_tick(complete)
    assert evaluation.completed == [complete]


@pytest.mark.parametrize("paced", [False, True])
def test_prepared_pending_tick_notifies_eof_before_drain_barrier(paced):
    """Prepared EOF polling must run when bootstrap is pending, not after it."""

    source = _Adapter().packet_from_record(_Record(0)).source

    class _Session:
        def submit(self, packet):
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def __init__(self):
            self.eof = False

        def mark_source_complete(self):
            self.eof = True

        def tick(self):
            if not self.eof:
                return SimpleNamespace(
                    state=SimpleNamespace(value="bootstrap_pending"),
                    executed=False,
                    optimizer_steps=0,
                    supervision_views=0,
                    completed_supervision_sources=(),
                )
            return SimpleNamespace(
                state=SimpleNamespace(value="executed"),
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                completed_supervision_sources=(source,),
            )

    application = _Application()
    observed = []

    def observer(result):
        observed.append(result)
        application.mark_source_complete()

    if paced:
        driver = PacedCausalReplayDriver(
            application,
            _Adapter(),
            source_hz=1000.0,
            source_complete_observer=observer,
        )
    else:
        driver = CausalReplayDriver(
            application,
            _Adapter(),
            source_complete_observer=observer,
        )

    summary = driver.run([_Record(0)])

    assert len(observed) == 1
    assert getattr(observed[0].state, "value", None) == "bootstrap_pending"
    assert summary.optimizer_steps == 1
    assert summary.supervision_views == 1


@dataclass(frozen=True)
class _Record:
    ordinal: int


class _Adapter(DatasetPacketAdapter[_Record]):
    def packet_from_record(self, record: _Record) -> SensorPacket:
        source = SourceIdentity("dataset", 0, record.ordinal, 1_000 + record.ordinal)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
            depth=None,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
        )


@pytest.mark.parametrize("paced", [False, True])
def test_cache_pressure_retries_the_same_packet_before_later_sources(paced):
    """A packet-only retention defer does not consume the source or re-track it."""
    source = _Adapter().packet_from_record(_Record(0)).source

    class _Session:
        prepared_mode = True

        def __init__(self):
            self.packets = []
            self.accepted = []

        def submit(self, packet):
            self.packets.append(packet)
            if len(self.packets) == 1:
                raise PreparedIngressDeferred("spill pending")
            self.accepted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()
            self.ticks = 0

        def tick(self):
            self.ticks += 1
            completed = () if not self.session.accepted else (self.session.accepted.pop(0),)
            return SimpleNamespace(
                executed=self.ticks > 1,
                retryable=self.ticks == 1,
                optimizer_steps=1 if self.ticks > 1 else 0,
                supervision_views=1 if self.ticks > 1 else 0,
                completed_supervision_sources=completed,
                work_plan=SimpleNamespace(
                    supervision_sources=completed, replay_sources=()
                ),
            )

    application = _Application()
    driver = (
        PacedCausalReplayDriver(application, _Adapter(), source_hz=1_000.0)
        if paced
        else CausalReplayDriver(application, _Adapter())
    )
    summary = driver.run((_Record(0), _Record(1)))

    assert [packet.source.source_ordinal for packet in application.session.packets] == [0, 0, 1]
    assert application.session.packets[0] is application.session.packets[1]
    assert summary.submitted == 2
    assert application.ticks >= 3


@pytest.mark.parametrize("paced", [False, True])
def test_causal_bypass_capture_retries_before_consuming_a_holdout(paced):
    """A holdout is cached before its source clock advances under pressure."""

    captures = []
    consumed = []

    class _Session:
        prepared_mode = True

        def __init__(self):
            self.submitted = []

        def submit(self, packet):
            self.submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()
            self.ticks = 0

        def tick(self):
            self.ticks += 1
            completed = tuple(self.session.submitted[:1])
            del self.session.submitted[:1]
            return SimpleNamespace(
                executed=True,
                retryable=True,
                optimizer_steps=1,
                supervision_views=1,
                completed_supervision_sources=completed,
                work_plan=SimpleNamespace(supervision_sources=completed, replay_sources=()),
            )

    def cache_bypass(packet, *, role):
        captures.append((packet, role))
        if len(captures) == 1:
            raise PreparedIngressDeferred("spill pending")
        return replace(
            packet,
            rgb=BufferRef.from_numpy(
                packet.rgb.data.copy(), encoding="rgb8", ownership=BufferOwnership.OWNED
            ),
        )

    application = _Application()
    driver = (
        PacedCausalReplayDriver(
            application,
            _Adapter(),
            source_hz=1_000.0,
            packet_role=lambda packet: "holdout" if packet.source.source_ordinal == 0 else "train",
            bypass_packet_capture=cache_bypass,
            source_consumed_observer=lambda packet: consumed.append(packet),
        )
        if paced
        else CausalReplayDriver(
            application,
            _Adapter(),
            packet_role=lambda packet: "holdout" if packet.source.source_ordinal == 0 else "train",
            bypass_packet_capture=cache_bypass,
            source_consumed_observer=lambda packet: consumed.append(packet),
        )
    )

    summary = driver.run((_Record(0), _Record(1)))

    assert [role for _packet, role in captures] == ["holdout", "holdout"]
    assert captures[0][0] is captures[1][0]
    assert consumed == [captures[1][0]]
    assert application.session.submitted == []
    assert summary.holdout_reserved == 1
    assert summary.submitted == 1


@pytest.mark.parametrize("paced", [False, True])
def test_direct_causal_bypasses_use_cache_owned_pixels_before_source_consumption(
    paced, tmp_path
):
    """Holdout/guard bypasses share prepared cache pressure and never train."""

    arrays = {
        ordinal: (
            np.full((2, 2, 3), 40 + ordinal, np.uint8),
            np.full((2, 2), 1040 + ordinal, np.uint16),
        )
        for ordinal in range(3)
    }

    class _BorrowingAdapter(DatasetPacketAdapter[_Record]):
        def packet_from_record(self, record):
            source = SourceIdentity("dataset", 0, record.ordinal, 1_000 + record.ordinal)
            rgb, depth = arrays[record.ordinal]
            return SensorPacket(
                source=source,
                rgb=BufferRef.from_numpy(
                    rgb, encoding="rgb8", ownership=BufferOwnership.BORROWED,
                    owner_token=object(),
                ),
                depth=BufferRef.from_numpy(
                    depth, encoding="16UC1", units="mm",
                    ownership=BufferOwnership.BORROWED, owner_token=object(),
                ),
                intrinsics=CameraIntrinsics(
                    fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2
                ),
                tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
            )

    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        cold_payload_spill_root=tmp_path / "prepared_cold_payloads",
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db

    class _Session:
        prepared_mode = True

        def __init__(self):
            self.submitted = []

        def submit(self, packet):
            self.submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()

        def tick(self):
            db.commit_cold_payload_spill_receipts(
                db.write_pending_cold_payload_spills(max_items=2)
            )
            completed = tuple(self.session.submitted[:1])
            del self.session.submitted[:1]
            return SimpleNamespace(
                executed=True,
                retryable=True,
                optimizer_steps=1,
                supervision_views=1,
                completed_supervision_sources=completed,
                work_plan=SimpleNamespace(supervision_sources=completed, replay_sources=()),
            )

    application = _Application()
    consumed = []

    def source_consumed(packet):
        consumed.append(packet)
        packet.rgb.data.fill(0)
        packet.depth.data.fill(0)

    driver = (
        PacedCausalReplayDriver(
            application,
            _BorrowingAdapter(),
            source_hz=1_000.0,
            packet_role=lambda packet: ("holdout", "guard", "train")[packet.source.source_ordinal],
            bypass_packet_capture=runtime.cache_causal_bypass_packet,
            source_consumed_observer=source_consumed,
        )
        if paced
        else CausalReplayDriver(
            application,
            _BorrowingAdapter(),
            packet_role=lambda packet: ("holdout", "guard", "train")[packet.source.source_ordinal],
            bypass_packet_capture=runtime.cache_causal_bypass_packet,
            source_consumed_observer=source_consumed,
        )
    )

    summary = driver.run(tuple(_Record(ordinal) for ordinal in range(3)))
    for rgb, depth in arrays.values():
        rgb.fill(0)
        depth.fill(0)

    first = _Adapter().packet_from_record(_Record(0)).source
    first_id = db.frame_id_for_source(first)
    assert first_id is not None
    assert db.historical_candidate_for_source(first) is None
    handle = db.get_cold_pixels(first_id)
    assert handle is not None and handle.reload_token is not None
    rgb, depth = db.load_cold_payload_from_token(handle.reload_token)
    assert np.array_equal(rgb, np.full((2, 2, 3), 40, np.uint8))
    assert np.array_equal(depth, np.full((2, 2), 1040, np.uint16))
    assert consumed[0].rgb.data is arrays[0][0]
    assert consumed[0].depth.data is arrays[0][1]
    assert summary.holdout_reserved == 1
    assert summary.guard_withheld == 1
    assert summary.submitted == 1


class _FakePacedClock:
    def __init__(self):
        self.now_s = 0.0

    def __call__(self):
        return self.now_s

    def advance(self, seconds):
        self.now_s += seconds


class _ImmediatePacedProducer:
    def __init__(self, records, adapter, **_kwargs):
        self._packets = [adapter.packet_from_record(record) for record in records]
        self._index = 0

    def start(self):
        pass

    def wait_for_initial_packet_or_terminal(self):
        pass

    def take(self, *, timeout_s):
        del timeout_s
        if self._index == len(self._packets):
            return None
        packet = self._packets[self._index]
        self._index += 1
        return packet

    @property
    def complete(self):
        return self._index == len(self._packets)

    @property
    def offered(self):
        return len(self._packets)

    queue_depth = capacity = dropped = dropped_at_arrival = dropped_at_enqueue = 0
    high_watermark = 0
    offered_fps = None

    def source_ingress_report_block(self):
        return {}

    def close(self):
        pass


def test_paced_driver_closes_producer_when_initial_wait_fails(monkeypatch):
    import eligsir.runtime.causal_replay as causal_replay

    closed = []

    class _Producer:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def wait_for_initial_packet_or_terminal(self):
            raise RuntimeError("decode failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _Producer)

    with pytest.raises(RuntimeError, match="decode failed"):
        PacedCausalReplayDriver(object(), _Adapter(), source_hz=30.0).run((_Record(7),))
    assert closed == [True]


def test_decision_trace_coalesces_identical_zero_progress_rows(tmp_path):
    """A stalled opt-in trace keeps evidence without becoming an I/O storm."""

    trace = JsonlDecisionTraceObserver(tmp_path / "decisions.jsonl")

    def tick(iteration, *, executed=False):
        return SimpleNamespace(
            iteration=iteration,
            executed=executed,
            reason="prepared_authorized_sources_not_ready",
            state=SimpleNamespace(value="starved"),
            optimizer_steps=int(executed),
            supervision_views=4 if executed else 0,
        )

    trace(tick(1))
    trace(tick(2))
    trace(tick(3, executed=True))
    trace(tick(4))
    # Issue #1664: rows are batched in memory; flush explicitly before
    # reading the file back (this is exactly the behavior finalize()/close()
    # trigger in production).
    trace.flush()

    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 3, 4]


def test_drain_tracker_counts_completed_bootstrap_sources_without_optimizer_step():
    """Bootstrap completes semantic source work before it is an optimizer update."""

    source = SourceIdentity("dataset", 0, 0, 1_000)
    tracker = _MappingDrainTracker()
    tracker.record_submit(
        SimpleNamespace(
            event=SimpleNamespace(source=source),
            disposition=SimpleNamespace(admit_supervision=True),
        )
    )

    assert tracker.record_tick(
        SimpleNamespace(
            executed=False,
            optimizer_steps=0,
            completed_supervision_sources=(source,),
        )
    )
    assert tracker.pending_new_sources == 0


def test_drain_tracker_keeps_oldest_admitted_debt_after_newest_completion():
    oldest = SourceIdentity("dataset", 0, 0, 1_000)
    newest = SourceIdentity("dataset", 0, 1, 1_001)
    tracker = _MappingDrainTracker()
    for source in (oldest, newest):
        tracker.record_submit(
            SimpleNamespace(
                event=SimpleNamespace(source=source),
                disposition=SimpleNamespace(admit_supervision=True),
            )
        )

    assert tracker.record_tick(
        SimpleNamespace(
            executed=True,
            completed_supervision_sources=(newest,),
        )
    )
    assert tracker.pending_sources == (oldest,)
    assert not tracker.reached_barrier(
        SimpleNamespace(executed=True, completed_supervision_sources=(newest,)),
        submitted=2,
    )
    assert tracker.record_tick(
        SimpleNamespace(
            executed=True,
            completed_supervision_sources=(oldest,),
        )
    )
    assert tracker.reached_barrier(
        SimpleNamespace(executed=True, completed_supervision_sources=(oldest,)),
        submitted=2,
    )


def test_paced_prepared_driver_retains_all_fifo_identity_debt(monkeypatch):
    """Completed FIFO identities reset the drain watchdog beyond the old bound."""

    import eligsir.runtime.causal_replay as causal_replay

    clock = _FakePacedClock()
    monkeypatch.setattr(causal_replay.time, "perf_counter", clock)
    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _ImmediatePacedProducer)

    class _Session:
        prepared_mode = True

        def __init__(self):
            self.submitted = []

        def submit(self, packet):
            self.submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()
            self._ticks = self._completed = 0

        def tick(self):
            clock.advance(10.0)
            self._ticks += 1
            completed = ()
            if (
                len(self.session.submitted) == 20
                and self._ticks > len(self.session.submitted)
            ):
                completed = (self.session.submitted[self._completed],)
                self._completed += 1
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                iteration=self._ticks,
                completed_supervision_sources=completed,
                work_plan=SimpleNamespace(requested_supervision_slots=1),
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=10_000.0,
        source_buffer_frames=32,
        max_new_sources_per_tick=15,
        semantic_drain_timeout_s=30.0,
    ).run(tuple(_Record(ordinal) for ordinal in range(20)))

    assert summary.submitted == 20
    assert summary.ticks == 40
    assert clock.now_s > 30.0

    class _EvictingTracker(_MappingDrainTracker):
        def record_submit(self, outcome):
            super().record_submit(outcome)
            while self.pending_new_sources > 15:
                self._pending_new_sources.popitem(last=False)

    monkeypatch.setattr(causal_replay, "_MappingDrainTracker", _EvictingTracker)
    with pytest.raises(RuntimeError, match="made no semantic progress for 30s"):
        PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=10_000.0,
            source_buffer_frames=32,
            max_new_sources_per_tick=15,
            semantic_drain_timeout_s=30.0,
        ).run(tuple(_Record(ordinal) for ordinal in range(20)))


def test_paced_prepared_replay_only_ticks_do_not_reset_semantic_watchdog(monkeypatch):
    import eligsir.runtime.causal_replay as causal_replay

    clock = _FakePacedClock()
    monkeypatch.setattr(causal_replay.time, "perf_counter", clock)
    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _ImmediatePacedProducer)

    class _Session:
        prepared_mode = True

        def __init__(self):
            self.replay_source = SourceIdentity("dataset", 0, 99, 1_099)

        def submit(self, packet):
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()
            self._ticks = 0

        def tick(self):
            clock.advance(10.0)
            self._ticks += 1
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                iteration=self._ticks,
                completed_supervision_sources=(self.session.replay_source,),
                work_plan=SimpleNamespace(
                    requested_supervision_slots=1,
                    supervision_sources=(),
                    replay_sources=(self.session.replay_source,),
                ),
            )

    with pytest.raises(RuntimeError, match="made no semantic progress for 30s"):
        PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=10_000.0,
            source_buffer_frames=1,
            semantic_drain_timeout_s=30.0,
        ).run((_Record(0),))


def test_paced_prepared_replays_resident_b4_during_source_gap(monkeypatch):
    """A prepared mapper keeps its B4 optimizer clock running between inputs.

    This catches a regression where a no-packet poll either bypasses the
    prepared tick or re-enters packet preparation instead of consuming the
    already-resident replay plan.
    """

    import eligsir.runtime.causal_replay as causal_replay

    class _FakeClock:
        def __init__(self):
            self._now = 0.0

        def __call__(self):
            self._now += 0.001
            return self._now

    class _CountingAdapter(_Adapter):
        def __init__(self):
            self.prepared_ordinals = []

        def packet_from_record(self, record):
            self.prepared_ordinals.append(record.ordinal)
            return super().packet_from_record(record)

    class Producer:
        last = None

        def __init__(self, records, adapter, **_kwargs):
            records = tuple(records)
            assert records == (_Record(0),)
            self._scheduled = [adapter.packet_from_record(records[0]), None, None]
            self.timeouts = []
            self._taken = 0
            type(self).last = self

        def start(self):
            pass

        def wait_for_initial_packet_or_terminal(self):
            pass

        def take(self, *, timeout_s):
            self.timeouts.append(timeout_s)
            packet = self._scheduled[self._taken]
            self._taken += 1
            return packet

        @property
        def complete(self):
            return self._taken == len(self._scheduled)

        queue_depth = 0
        capacity = 1
        dropped = 0
        dropped_at_arrival = 0
        dropped_at_enqueue = 0
        high_watermark = 1
        offered_fps = 1.0
        offered = 1

        def source_ingress_report_block(self):
            return {}

        def close(self):
            pass

    source = _Adapter().packet_from_record(_Record(0)).source
    resident = tuple(
        SourceIdentity("resident", 0, ordinal, 2_000 + ordinal)
        for ordinal in range(4)
    )
    plan = SimpleNamespace(
        requested_supervision_slots=1,
        supervision_sources=(),
        replay_sources=resident,
        supervision_constellation=SimpleNamespace(
            requested_batch_size=4,
            requested_camera_count=4,
            realized_batch_size=4,
            realized_camera_count=4,
            mode="true_camera_batch",
            reason="resident_ready_replay",
        ),
    )

    class Session:
        prepared_mode = True

        def __init__(self):
            self.submitted = []

        def submit(self, packet):
            self.submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class Application:
        def __init__(self):
            self.session = Session()
            self.progress_waits = []
            self.executed_replay_batches = []
            self._ticks = 0

        def wait_for_prepared_progress(self, *, timeout_s):
            self.progress_waits.append(timeout_s)

        def tick(self):
            self._ticks += 1
            self.executed_replay_batches.append(plan.replay_sources)
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=4,
                completed_supervision_sources=(source,) if self._ticks == 1 else (),
                work_plan=plan,
            )

    adapter = _CountingAdapter()
    application = Application()
    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", Producer)
    monkeypatch.setattr(causal_replay.time, "perf_counter", _FakeClock())

    summary = PacedCausalReplayDriver(
        application, adapter, source_hz=1_000.0
    ).run((_Record(0),))

    assert summary.submitted == 1
    assert summary.zero_frame_ticks == 2
    assert summary.optimizer_steps == 3
    assert summary.supervision_views == 12
    assert summary.realized_supervision_constellations == {"B4_C4": 3}
    assert adapter.prepared_ordinals == [0]
    assert application.session.submitted == [source]
    assert application.progress_waits == [pytest.approx(0.001), pytest.approx(0.001)]
    assert application.executed_replay_batches == [resident, resident, resident]
    assert Producer.last is not None
    assert Producer.last.timeouts == [pytest.approx(0.001), 0.0, 0.0]


def test_causal_replay_driver_only_adapts_submits_and_ticks_in_source_order():
    mapped = []
    ticks = []
    observed = []
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda frame, _: mapped.append(frame.source),
            optimizer_tick=lambda: ticks.append(len(ticks)),
            tracking_provider=ExternalTrackingProvider(),
        )
    )

    summary = CausalReplayDriver(
        application, _Adapter(), tick_observer=observed.append
    ).run(
        (_Record(0), _Record(1), _Record(2))
    )

    assert summary.submitted == 3
    assert summary.ticks == 4
    assert summary.source_identities == tuple(mapped)
    assert [source.source_ordinal for source in mapped] == [0, 1, 2]
    assert ticks == [0, 1, 2, 3]
    assert observed == [None, None, None, None]


def test_causal_replay_driver_persists_terminal_coverage_by_source_identity():
    ledger = CoverageLedger.fresh(
        resolved_source_frames=3,
        raw_registry_source_frames=None,
        source_denominator_provenance="causal_source_catalog",
        bootstrap_frames_consumed=0,
        source_identity="/data/fr3",
        effective_stride=1,
    )
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda _frame, _owner: None,
            optimizer_tick=lambda: None,
            tracking_provider=ExternalTrackingProvider(),
        )
    )

    summary = CausalReplayDriver(
        application,
        _Adapter(),
        coverage_ledger=ledger,
        bootstrap_frames=1,
    ).run((_Record(0), _Record(1), _Record(2)))

    assert summary.coverage_ledger is not None
    assert summary.coverage_ledger["source_frames_ingested_total"] == 3
    assert summary.coverage_ledger["bootstrap_frames_consumed"] == 1
    assert summary.coverage_ledger["causal_source_accounting"] == {
        "mode": "terminal_disposition",
        "unique_source_count": 3,
    }


def test_unpaced_source_eof_waits_for_a_replay_only_mapping_drain_barrier():
    events = []
    admitted = []

    class _Session:
        def submit(self, packet):
            admitted.append(packet.source)
            events.append(("submit", packet.source.source_ordinal))
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def tick(self):
            if admitted:
                source = admitted.pop(0)
                lane = "new"
                supervision_sources = (source,)
                replay_sources = ()
            else:
                source = _Adapter().packet_from_record(_Record(0)).source
                lane = "replay"
                supervision_sources = ()
                replay_sources = (source,)
            events.append(("tick", lane))
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                iteration=len(events),
                work_plan=SimpleNamespace(
                    supervision_sources=supervision_sources,
                    replay_sources=replay_sources,
                ),
            )

    summary = CausalReplayDriver(
        _Application(),
        _Adapter(),
        source_complete_observer=lambda result: events.append(
            ("source_complete", tuple(result.work_plan.replay_sources))
        ),
    ).run((_Record(0),))

    assert summary.submitted == 1
    assert summary.ticks == 2
    assert events[-2] == ("tick", "replay")
    assert events[-1][0] == "source_complete"
    assert events[-1][1]


def test_paced_source_eof_drains_failed_new_lane_then_crosses_replay_barrier():
    events = []
    admitted = []
    new_attempts = 0
    replay_attempts = 0

    class _Session:
        def submit(self, packet):
            admitted.append(packet.source)
            events.append(("submit", packet.source.source_ordinal))
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def tick(self):
            nonlocal new_attempts, replay_attempts
            if admitted:
                source = admitted[0]
                new_attempts += 1
                executed = new_attempts >= 2
                lane = "new_completed" if executed else "new_failed"
                if executed:
                    admitted.pop(0)
                supervision_sources = (source,)
                replay_sources = ()
            else:
                source = _Adapter().packet_from_record(_Record(0)).source
                replay_attempts += 1
                executed = replay_attempts >= 2
                lane = "replay_completed" if executed else "replay_failed"
                supervision_sources = ()
                replay_sources = (source,)
            events.append(("tick", lane))
            return SimpleNamespace(
                executed=executed,
                optimizer_steps=int(executed),
                supervision_views=int(executed),
                iteration=len(events),
                work_plan=SimpleNamespace(
                    requested_supervision_slots=1,
                    supervision_sources=supervision_sources,
                    replay_sources=replay_sources,
                ),
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=10_000.0,
        source_buffer_frames=4,
        source_complete_observer=lambda result: events.append(
            ("source_complete", tuple(result.work_plan.replay_sources))
        ),
    ).run((_Record(0),))

    assert summary.submitted == 1
    assert ("tick", "new_failed") in events
    assert ("tick", "new_completed") in events
    assert ("tick", "replay_failed") in events
    assert events[-2] == ("tick", "replay_completed")
    assert events[-1][0] == "source_complete"
    assert events[-1][1]


def test_paced_eof_rejects_replay_barrier_with_uncompleted_new_identity():
    events = []
    submitted = False
    replay_source = _Adapter().packet_from_record(_Record(99)).source

    class _Session:
        def submit(self, packet):
            nonlocal submitted
            submitted = True
            events.append(("submit", packet.source))
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def tick(self):
            events.append(("tick", submitted))
            if len(events) > 12:
                raise RuntimeError("endpoint did not accept executed replay barrier")
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                iteration=len(events),
                work_plan=SimpleNamespace(
                    requested_supervision_slots=1,
                    supervision_sources=(),
                    replay_sources=(replay_source,),
                ),
            )

    with pytest.raises(RuntimeError, match="endpoint did not accept"):
        PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=10_000.0,
            source_buffer_frames=4,
            source_complete_observer=lambda result: events.append(
                ("source_complete", result.executed)
            ),
        ).run((_Record(0),))

    submit_index = next(
        index for index, event in enumerate(events) if event[0] == "submit"
    )
    assert events[submit_index] == (
        "submit", _Adapter().packet_from_record(_Record(0)).source
    )
    assert events[-1] == ("tick", True)


def test_decision_trace_row_exposes_stale_population_axes_under_policy():
    row = decision_trace_row(
        SimpleNamespace(
            iteration=1,
            executed=True,
            reason="",
            population_policy_disposition="stale_current_snapshot",
            population_stale_current_mismatches=(("optimizer_update", 8, 9),),
        )
    )

    assert row["policy"]["stale_current_mismatches"] == [["optimizer_update", 8, 9]]


def test_causal_mapping_tick_policy_rejects_a_fixed_post_source_tail():
    with pytest.raises(TypeError):
        CausalReplayTickPolicy(ticks_per_submission=1, final_ticks=3)


def test_jsonl_decision_trace_is_opt_in_and_records_basis_and_mutation(tmp_path):
    from types import SimpleNamespace

    source = SourceIdentity("dataset", 0, 4, 4_000)
    trace = JsonlDecisionTraceObserver(tmp_path / "decisions.jsonl")
    trace(
        SimpleNamespace(
            snapshot_id=9,
            iteration=12,
            executed=True,
            retained_catalog_population_post_dispositions=1,
            reason="",
            optimizer_steps=2,
            supervision_views=2,
            map_content_version=13,
            topology_version=3,
            parameter_version=12,
            control_snapshot=SimpleNamespace(
                ingress_queue_depth=8,
                freshness_mode=FreshnessMode.CRITICAL,
                evidence=(SimpleNamespace(name="pressure_state", value="critical"),),
            ),
            work_plan=SimpleNamespace(
                plan_id="work-9",
                supervision_sources=(source, source, source),
                replay_sources=(source,),
                reason_trace=("critical_bounded_replay_preserved",),
                insertion=SimpleNamespace(
                    source=None,
                    sample_stride=8,
                    candidate_budget=0,
                    seed_budget=0,
                ),
                mutation_budget=SimpleNamespace(max_total=0),
            ),
            selection=SimpleNamespace(
                leases=(object(), object(), object()),
                fresh_count=1,
                replay_count=2,
                starved=True,
            ),
            insertion_candidates=16,
            inserted=4,
            pruned=1,
            relocated=0,
            corrected=0,
            structural_mutation=True,
            transfers=SimpleNamespace(
                h2d_ops=2,
                h2d_bytes=128,
                d2h_ops=1,
                d2h_bytes=16,
                pool_waits=0,
                pool_drops=0,
            ),
            telemetry=SimpleNamespace(
                gpu_transaction=SimpleNamespace(
                    transaction_id=44,
                    transaction_type=SimpleNamespace(value="optimization"),
                    duration_ms=3.25,
                    map_content_version=13,
                    topology_version=3,
                    parameter_version=12,
                    batch_size=4,
                    pixels=4096,
                    rasterizer_calls=1,
                    optimizer_updates=1,
                    candidates=0,
                    mutation_type="none",
                    inserted=0,
                    pruned=0,
                    relocated=0,
                    corrected=0,
                ),
                map_state=SimpleNamespace(
                    gaussian_count=123,
                    map_content_version=13,
                    topology_version=3,
                    parameter_version=12,
                ),
            ),
            mutation_events=(
                SimpleNamespace(
                    kind="replay_depth",
                    source=source,
                    candidates=16,
                    inserted=4,
                    pruned=0,
                    relocated=0,
                    corrected=0,
                ),
                SimpleNamespace(
                    kind="mcmc",
                    source=None,
                    candidates=0,
                    inserted=0,
                    pruned=1,
                    relocated=0,
                    corrected=0,
                ),
            ),
            primary_training_psnr_db=23.5,
            primary_training_source_id=source.fingerprint(),
            dispositions=(SimpleNamespace(
                source=source,
                kind=SimpleNamespace(value="admit"),
                reason="critical_frontier",
                mapping_candidate=True,
                admit_supervision=True,
                allow_geometry_insertion=False,
                residency=SimpleNamespace(value="hot"),
                representative=None,
                evidence_trace=(
                    SimpleNamespace(name="pressure_state", value="critical"),
                ),
                modality_eligibility=SimpleNamespace(
                    rgb_supervision=True,
                    rgb_quality="valid",
                    depth_supervision=False,
                    depth_quality="coarse_only",
                    depth_insertion=False,
                ),
            ),),
        )
    )

    trace.flush()
    row = __import__("json").loads((tmp_path / "decisions.jsonl").read_text())
    assert row["snapshot"]["evidence"]["pressure_state"] == "critical"
    assert row["retained_catalog"] == {
        "population": 1,
        "position": "post_dispositions",
    }
    assert row["reason_trace"] == ["critical_bounded_replay_preserved"]
    assert row["allocation_request"] == {
        "supervision_slots": 0,
        "replay_slots": 0,
    }
    assert row["ready_batch"] == {
        "requested": 4,
        "selected": 3,
        "fresh": 1,
        "replay": 2,
        "full": False,
        "waited_for_ready_source": False,
        "starved": True,
    }
    assert row["resolution_tier"] == "native"
    assert row["replay_source_records"] == [
        {
            "fingerprint": source.fingerprint(),
            "source_ordinal": source.source_ordinal,
            "timestamp_ns": source.timestamp_ns,
        }
    ]
    assert row["primary_training_psnr_db"] == 23.5
    assert row["transfers"] == {
        "h2d_ops": 2,
        "h2d_bytes": 128,
        "d2h_ops": 1,
        "d2h_bytes": 16,
        "pool_waits": 0,
        "pool_drops": 0,
    }
    assert row["telemetry"]["gpu_transaction"] == {
        "transaction_id": 44,
        "transaction_type": "optimization",
        "duration_ms": 3.25,
        "map_content_version": 13,
        "topology_version": 3,
        "parameter_version": 12,
        "batch_size": 4,
        "pixels": 4096,
        "rasterizer_calls": 1,
        "optimizer_updates": 1,
        "candidates": 0,
        "mutation_type": "none",
        "inserted": 0,
        "pruned": 0,
        "relocated": 0,
        "corrected": 0,
    }
    assert row["telemetry"]["map_state"]["gaussian_count"] == 123


    assert row["dispositions"][0]["reason"] == "critical_frontier"
    assert row["dispositions"][0]["evidence"]["pressure_state"] == "critical"
    assert row["mutation"] == {
        "candidates": 16,
        "inserted": 4,
        "pruned": 1,
        "relocated": 0,
        "corrected": 0,
        "structural_mutation": True,
        "events": [
            {
                "kind": "replay_depth",
                "source": source.fingerprint(),
                "candidates": 16,
                "inserted": 4,
                "pruned": 0,
                "relocated": 0,
                "corrected": 0,
            },
            {
                "kind": "mcmc",
                "source": None,
                "candidates": 0,
                "inserted": 0,
                "pruned": 1,
                "relocated": 0,
                "corrected": 0,
            },
        ],
    }


def test_causal_entry_resolves_opt_in_trace_under_model_path(tmp_path):
    from types import SimpleNamespace

    from eligsir.runtime.causal_entry import decision_trace_observer_from_args

    disabled = decision_trace_observer_from_args(
        dataset=SimpleNamespace(model_path=str(tmp_path)),
        args=SimpleNamespace(causal_decision_trace=""),
    )
    enabled = decision_trace_observer_from_args(
        dataset=SimpleNamespace(model_path=str(tmp_path)),
        args=SimpleNamespace(causal_decision_trace="decision_mutation_trace.jsonl"),
    )

    assert disabled is None
    assert isinstance(enabled, JsonlDecisionTraceObserver)
    enabled(SimpleNamespace())
    # Issue #1664: the row is buffered, not written synchronously -- flush
    # (as finalize()/close() do in production) before checking the file.
    enabled.flush()
    assert (tmp_path / "decision_mutation_trace.jsonl").is_file()


def test_decision_trace_finalizer_writes_sampling_ledger_and_heatmap(tmp_path):
    from types import SimpleNamespace

    trace = JsonlDecisionTraceObserver(tmp_path / "decisions.jsonl")
    sources = tuple(
        SourceIdentity("dataset", 0, index, 1_000 + index) for index in range(8)
    )
    for iteration in range(1, 17):
        admitted = sources[min(iteration - 1, len(sources) - 1)]
        replay = (sources[(iteration * 3) % len(sources)],)
        trace(
            SimpleNamespace(
                snapshot_id=iteration,
                iteration=iteration,
                executed=True,
                optimizer_steps=1,
                supervision_views=2,
                map_content_version=iteration,
                topology_version=0,
                parameter_version=iteration,
                control_snapshot=SimpleNamespace(
                    ingress_queue_depth=0,
                    freshness_mode=FreshnessMode.HEALTHY,
                    replay_candidates=sources[: min(iteration, len(sources))],
                    replay_candidate_weights=(1.0,) * min(iteration, len(sources)),
                    evidence=(),
                ),
                work_plan=SimpleNamespace(
                    plan_id=f"work-{iteration}",
                    supervision_sources=(admitted,),
                    replay_sources=replay,
                    replay_population_size=min(iteration, len(sources)),
                    replay_sampling_seed=iteration,
                    replay_sampling_epoch=iteration,
                    replay_selected_weights=(1.0,),
                    resolution_tier=SimpleNamespace(value="half"),
                    reason_trace=("weighted_random_full_cache",),
                    insertion=SimpleNamespace(
                        source=None,
                        sample_stride=12,
                        candidate_budget=0,
                        seed_budget=0,
                    ),
                    mutation_budget=SimpleNamespace(max_total=0),
                ),
                insertion_candidates=0,
                inserted=0,
                pruned=0,
                relocated=0,
                corrected=0,
                structural_mutation=False,
                dispositions=(SimpleNamespace(
                    source=admitted,
                    kind=SimpleNamespace(value="admit"),
                    reason="qualified",
                    mapping_candidate=True,
                    admit_supervision=True,
                    allow_geometry_insertion=False,
                    residency=SimpleNamespace(value="hot"),
                    representative=None,
                    evidence_trace=(),
                    modality_eligibility=SimpleNamespace(
                        rgb_supervision=True,
                        rgb_quality="valid",
                        depth_supervision=False,
                        depth_quality="coarse_only",
                        depth_insertion=False,
                    ),
                ),),
            )
        )

    summary = trace.finalize()

    assert summary["total_replay_selections"] == 16
    assert sum(summary["resolution_tiers"].values()) == 16
    assert sum(summary["planned_resolution_tiers"].values()) == 16
    assert summary["executed_ticks"] == 16
    assert summary["optimizer_steps_total"] == 16
    assert summary["supervision_views_total"] == 32
    assert summary["control_phases"] == {"online_mapping": 16}
    assert summary["requested_allocation_constellations"] == {
        "new=0;replay=0": 16
    }
    assert summary["ingress_queue_depth_max"] == 0
    assert summary["mcmc_requested_ticks"] == 0
    assert summary["unique_admitted_frames"] == 8
    assert summary["unique_replayed_frames"] == 8
    assert summary["replay_coverage_fraction"] == 1.0
    assert "eligibility_normalized_sampling_gini" in summary
    assert (tmp_path / "replay_sampling_summary.json").is_file()
    assert (tmp_path / "replay_sampling_counts.jsonl").is_file()
    assert (tmp_path / "replay_sampling_heatmap.svg").is_file()
    rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "replay_sampling_counts.jsonl").read_text().splitlines()
    ]
    assert all("eligible_tick_count" in row for row in rows)
    assert all("uniform_expected_replay_count" in row for row in rows)


def test_sampling_telemetry_v3_separates_catalog_shortlist_and_deferred_coverage(
    tmp_path,
):
    """A bounded shortlist must not masquerade as the retained replay catalog."""

    from eligsir.contracts.control import EvidenceFact

    trace = JsonlDecisionTraceObserver(tmp_path / "mixed-decisions.jsonl")
    admitted = SourceIdentity("dataset", 0, 0, 1_000)
    deferred_a = SourceIdentity("dataset", 0, 1, 1_001)
    deferred_b = SourceIdentity("dataset", 0, 2, 1_002)

    def disposition(source, *, action, kind, residency, admits):
        return SimpleNamespace(
            source=source,
            kind=SimpleNamespace(value=kind),
            semantic_action=SimpleNamespace(value=action),
            historical_eligibility=SimpleNamespace(value="eligible"),
            reason="qualified" if admits else "no_motion",
            mapping_candidate=admits,
            admit_supervision=admits,
            allow_geometry_insertion=False,
            residency=SimpleNamespace(value=residency),
            representative=None,
            evidence_trace=(),
            modality_eligibility=SimpleNamespace(
                rgb_supervision=True,
                rgb_quality="valid",
                depth_supervision=False,
                depth_quality="coarse_only",
                depth_insertion=False,
            ),
        )

    def snapshot(*, catalog_population, candidates):
        return SimpleNamespace(
            control_tick_epoch=catalog_population,
            ingress_queue_depth=0,
            freshness_mode=FreshnessMode.HEALTHY,
            replay_candidates=tuple(candidates),
            replay_candidate_weights=(1.0,) * len(candidates),
            evidence=(
                EvidenceFact("replay_catalog_population", catalog_population),
                EvidenceFact("replay_shortlist_configured_limit", 2),
            ),
        )

    terminal_rows = (
        (
            snapshot(catalog_population=0, candidates=()),
            disposition(
                admitted,
                action="commit",
                kind="admit",
                residency="hot",
                admits=True,
            ),
        ),
        (
            snapshot(catalog_population=1, candidates=(admitted,)),
            disposition(
                deferred_a,
                action="defer",
                kind="skip_redundant",
                residency="cold",
                admits=False,
            ),
        ),
        (
            snapshot(
                catalog_population=2,
                candidates=(admitted, deferred_a),
            ),
            disposition(
                deferred_b,
                action="defer",
                kind="skip_redundant",
                residency="cold",
                admits=False,
            ),
        ),
    )
    for index, (control_snapshot, terminal) in enumerate(terminal_rows, start=1):
        trace(
            SimpleNamespace(
                snapshot_id=index,
                iteration=0,
                executed=False,
                retained_catalog_population_post_dispositions=index,
                optimizer_steps=0,
                supervision_views=0,
                map_content_version=0,
                topology_version=0,
                parameter_version=0,
                control_snapshot=control_snapshot,
                work_plan=None,
                dispositions=(terminal,),
            )
        )

    trace(
        SimpleNamespace(
            snapshot_id=4,
            iteration=1,
            executed=True,
            retained_catalog_population_post_dispositions=3,
            optimizer_steps=1,
            supervision_views=2,
            map_content_version=1,
            topology_version=0,
            parameter_version=1,
            control_snapshot=snapshot(
                catalog_population=3,
                candidates=(admitted, deferred_a),
            ),
            work_plan=SimpleNamespace(
                plan_id="mixed-replay",
                supervision_sources=(),
                replay_sources=(admitted, deferred_a),
                replay_population_size=2,
                replay_sampling_seed=1,
                replay_sampling_epoch=1,
                replay_selected_weights=(1.0, 1.0),
                resolution_tier=SimpleNamespace(value="half"),
                reason_trace=("weighted_random_full_cache",),
                insertion=SimpleNamespace(
                    source=None,
                    sample_stride=12,
                    candidate_budget=0,
                    seed_budget=0,
                ),
                mutation_budget=SimpleNamespace(max_total=0),
            ),
            dispositions=(),
            insertion_candidates=0,
            inserted=0,
            pruned=0,
            relocated=0,
            corrected=0,
            structural_mutation=False,
        )
    )

    summary = trace.finalize()

    assert summary["schema"] == "eligsir.replay-sampling-telemetry/v3"
    assert summary["retained_catalog_population"] == 3
    assert summary["retained_catalog_admitted_population"] == 1
    assert summary["retained_catalog_deferred_population"] == 2
    assert summary["replay_shortlist_configured_limit"] == 2
    assert summary["replay_shortlist_realized_population_max"] == 2
    assert summary["full_replay_coverage_fraction"] == pytest.approx(2 / 3)
    assert summary["replay_coverage_fraction"] == pytest.approx(2 / 3)
    assert summary["admitted_replay_coverage_fraction"] == 1.0
    assert summary["deferred_replay_coverage_fraction"] == 0.5
    assert summary["eligibility_metrics_status"] == "available_exact_membership"
    assert summary["eligibility_normalized_sampling_gini"] is not None
    assert "replay_cache_capacity" not in summary
    assert "sampling_gini" not in summary

    trace_rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "mixed-decisions.jsonl").read_text().splitlines()
    ]
    assert trace_rows[-1]["snapshot"]["replay_shortlist_realized_population"] == 2
    assert {
        record["fingerprint"]
        for record in trace_rows[-1]["snapshot"]["replay_candidate_records"]
    } == {admitted.fingerprint(), deferred_a.fingerprint()}
    assert trace_rows[-1]["snapshot"]["replay_candidate_membership_source"] == "control_snapshot"
    assert trace_rows[1]["dispositions"][0]["semantic_action"] == "defer"
    assert trace_rows[1]["dispositions"][0]["historical_eligibility"] == "eligible"


def test_sampling_telemetry_uses_idle_plan_ready_filtered_membership(tmp_path):
    """An idle plan's sampled population must not inherit a stale mixed shortlist."""

    from eligsir.contracts.control import EvidenceFact

    trace = JsonlDecisionTraceObserver(tmp_path / "idle-decisions.jsonl")
    candidates = tuple(
        SourceIdentity("dataset", 0, ordinal, 1_000 + ordinal)
        for ordinal in range(4)
    )
    stale_snapshot = SimpleNamespace(
        control_tick_epoch=3,
        ingress_queue_depth=0,
        freshness_mode=FreshnessMode.HEALTHY,
        replay_candidates=candidates[:3],
        replay_candidate_weights=(1.0, 1.0, 1.0),
        evidence=(
            EvidenceFact("replay_catalog_population", 4),
            EvidenceFact("replay_shortlist_configured_limit", 4),
        ),
    )

    def eligible_disposition(source):
        return SimpleNamespace(
            source=source,
            kind=SimpleNamespace(value="admit"),
            semantic_action=SimpleNamespace(value="commit"),
            historical_eligibility=SimpleNamespace(value="eligible"),
            reason="qualified",
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=False,
            residency=SimpleNamespace(value="hot"),
            representative=None,
            evidence_trace=(),
            modality_eligibility=SimpleNamespace(
                rgb_supervision=True,
                rgb_quality="valid",
                depth_supervision=False,
                depth_quality="coarse_only",
                depth_insertion=False,
            ),
        )

    trace(
        SimpleNamespace(
            snapshot_id=4,
            iteration=1,
            executed=True,
            retained_catalog_population_post_dispositions=4,
            optimizer_steps=1,
            supervision_views=1,
            map_content_version=1,
            topology_version=0,
            parameter_version=1,
            control_snapshot=stale_snapshot,
            work_plan=SimpleNamespace(
                plan_id="idle-replay",
                supervision_sources=(),
                replay_sources=(candidates[3],),
                replay_population_size=4,
                replay_sampling_seed=1,
                replay_sampling_epoch=1,
                replay_selected_weights=(2.0,),
                replay_candidate_sources=candidates,
                replay_candidate_weights=(1.0, 1.0, 1.0, 2.0),
                resolution_tier=SimpleNamespace(value="quarter"),
                reason_trace=("source_idle_replay_only",),
                insertion=SimpleNamespace(
                    source=None,
                    sample_stride=12,
                    candidate_budget=0,
                    seed_budget=0,
                ),
                mutation_budget=SimpleNamespace(max_total=0),
            ),
            dispositions=tuple(eligible_disposition(source) for source in candidates),
            insertion_candidates=0,
            inserted=0,
            pruned=0,
            relocated=0,
            corrected=0,
            structural_mutation=False,
        )
    )

    summary = trace.finalize()
    row = json.loads((tmp_path / "idle-decisions.jsonl").read_text())

    assert summary["eligibility_metrics_status"] == "available_exact_membership"
    assert summary["eligibility_population_mismatch_ticks"] == 0
    assert {
        record["fingerprint"] for record in row["snapshot"]["replay_candidate_records"]
    } == {source.fingerprint() for source in candidates}
    assert row["snapshot"]["replay_weight_sum"] == pytest.approx(5.0)
    assert row["snapshot"]["replay_candidate_membership_source"] == "work_plan"


def test_sampling_telemetry_v3_withholds_eligibility_metrics_without_membership(
    tmp_path,
):
    """Population counts alone cannot reconstruct per-frame eligibility."""

    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    row = {
        "iteration": 1,
        "executed": True,
        "optimizer_steps": 1,
        "supervision_views": 1,
        "snapshot": {
            "ingress_queue_depth": 0,
            "replay_candidate_population": 1,
            "evidence": {"replay_catalog_population": 4},
        },
        "replay_sampling": {"population": 1},
        "replay_source_records": [],
    }
    trace_path = tmp_path / "legacy-decisions.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["full_replay_coverage_fraction"] is None
    assert summary["replay_coverage_fraction"] is None
    assert summary["eligibility_metrics_status"] == "unavailable"
    assert summary["eligibility_normalized_sampling_gini"] is None
    assert summary["realized_to_uniform_expected_ratio_median"] is None
    assert summary["starvation_gap_eligible_ticks_max"] is None


def test_sampling_telemetry_v3_catalog_sequence_contradiction_fails_closed(
    tmp_path,
):
    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    admitted = SourceIdentity("dataset", 0, 0, 1_000)
    rejected = SourceIdentity("dataset", 0, 1, 1_001)
    rows = [
        {
            "iteration": 0,
            "executed": False,
            "optimizer_steps": 0,
            "retained_catalog": {
                "population": 1,
                "position": "post_dispositions",
            },
            "snapshot": {
                "retained_catalog_population": 0,
                "replay_candidate_records": [],
            },
            "dispositions": [
                {
                    "source": admitted.fingerprint(),
                    "source_ordinal": admitted.source_ordinal,
                    "timestamp_ns": admitted.timestamp_ns,
                    "kind": "admit",
                    "semantic_action": "commit",
                    "historical_eligibility": "eligible",
                }
            ],
        },
        {
            "iteration": 1,
            "executed": False,
            "optimizer_steps": 0,
            "retained_catalog": {
                "population": 0,
                "position": "post_dispositions",
            },
            "snapshot": {
                "retained_catalog_population": 0,
                "replay_candidate_records": [
                    {
                        "fingerprint": admitted.fingerprint(),
                        "source_ordinal": admitted.source_ordinal,
                        "timestamp_ns": admitted.timestamp_ns,
                    }
                ],
            },
            "dispositions": [
                {
                    "source": rejected.fingerprint(),
                    "source_ordinal": rejected.source_ordinal,
                    "timestamp_ns": rejected.timestamp_ns,
                    "kind": "skip_quality",
                    "semantic_action": "reject",
                    "historical_eligibility": "none",
                }
            ],
        },
        {
            "iteration": 2,
            "executed": True,
            "optimizer_steps": 1,
            "supervision_views": 1,
            "retained_catalog": {
                "population": 1,
                "position": "post_dispositions",
            },
            "snapshot": {
                "retained_catalog_population": 1,
                "replay_shortlist_realized_population": 1,
                "replay_candidate_records": [
                    {
                        "fingerprint": admitted.fingerprint(),
                        "source_ordinal": admitted.source_ordinal,
                        "timestamp_ns": admitted.timestamp_ns,
                    }
                ],
            },
            "replay_sampling": {"population": 1},
            "replay_source_records": [],
        },
    ]
    trace_path = tmp_path / "contradictory-catalog.jsonl"
    trace_path.write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows)
    )

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["catalog_sequence_mismatch_rows"] == 1
    assert summary["retained_catalog_identity_membership_status"] == "unavailable"
    assert summary["full_replay_coverage_fraction"] is None
    assert summary["eligibility_metrics_status"] == "unavailable"
    counts = [
        __import__("json").loads(line)
        for line in (tmp_path / "replay_sampling_counts.jsonl").read_text().splitlines()
    ]
    assert [row["fingerprint"] for row in counts] == [admitted.fingerprint()]


def test_sampling_telemetry_v3_validates_multi_disposition_post_catalog_population(
    tmp_path,
):
    """A post-disposition population is compared with the row's terminal union."""

    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    sources = tuple(
        SourceIdentity("dataset", 0, ordinal, 1_000 + ordinal)
        for ordinal in range(3)
    )
    dispositions = [
        {
            "source": source.fingerprint(),
            "source_ordinal": source.source_ordinal,
            "timestamp_ns": source.timestamp_ns,
            "kind": "admit" if index == 0 else "skip_redundant",
            "semantic_action": "commit" if index == 0 else "defer",
            "historical_eligibility": "eligible",
        }
        for index, source in enumerate(sources)
    ]
    row = {
        "iteration": 0,
        "executed": False,
        "optimizer_steps": 0,
        "retained_catalog": {
            "population": 3,
            "position": "post_dispositions",
        },
        "snapshot": {
            # This latest-submit snapshot was captured after two earlier
            # dispositions but before the final disposition in this tick.
            "retained_catalog_population": 2,
            "replay_candidate_records": [],
        },
        "dispositions": dispositions,
    }
    trace_path = tmp_path / "multi-submit-valid.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["catalog_sequence_mismatch_rows"] == 0
    assert summary["catalog_population_position_ambiguous_rows"] == 0
    assert summary["retained_catalog_population"] == 3
    assert summary["retained_catalog_identity_membership_status"] == (
        "available_exact_terminal_union"
    )


def test_sampling_telemetry_v3_contradictory_post_catalog_population_fails_closed(
    tmp_path,
):
    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    source = SourceIdentity("dataset", 0, 0, 1_000)
    row = {
        "iteration": 0,
        "executed": False,
        "optimizer_steps": 0,
        "retained_catalog": {
            "population": 0,
            "position": "post_dispositions",
        },
        "snapshot": {"replay_candidate_records": []},
        "dispositions": [
            {
                "source": source.fingerprint(),
                "source_ordinal": source.source_ordinal,
                "timestamp_ns": source.timestamp_ns,
                "kind": "admit",
                "semantic_action": "commit",
                "historical_eligibility": "eligible",
            }
        ],
    }
    trace_path = tmp_path / "post-catalog-contradiction.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["catalog_sequence_mismatch_rows"] == 1
    assert summary["retained_catalog_identity_membership_status"] == "unavailable"
    assert summary["full_replay_coverage_fraction"] is None


def test_sampling_telemetry_v3_ambiguous_legacy_catalog_position_fails_closed(
    tmp_path,
):
    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    source = SourceIdentity("dataset", 0, 0, 1_000)
    row = {
        "iteration": 0,
        "executed": False,
        "optimizer_steps": 0,
        "snapshot": {
            "retained_catalog_population": 0,
            "replay_candidate_records": [],
        },
        "dispositions": [
            {
                "source": source.fingerprint(),
                "source_ordinal": source.source_ordinal,
                "timestamp_ns": source.timestamp_ns,
                "kind": "admit",
                "semantic_action": "commit",
                "historical_eligibility": "eligible",
            }
        ],
    }
    trace_path = tmp_path / "legacy-ambiguous-catalog.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["catalog_population_position_ambiguous_rows"] == 1
    assert summary["retained_catalog_identity_membership_status"] == "unavailable"
    assert summary["full_replay_coverage_fraction"] is None


def test_sampling_telemetry_v3_zero_executed_ticks_withholds_eligibility_metrics(
    tmp_path,
):
    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    retained = SourceIdentity("dataset", 0, 0, 1_000)
    row = {
        "iteration": 0,
        "executed": False,
        "optimizer_steps": 0,
        "retained_catalog": {
            "population": 1,
            "position": "post_dispositions",
        },
        "snapshot": {
            "retained_catalog_population": 0,
            "replay_candidate_records": [],
        },
        "dispositions": [
            {
                "source": retained.fingerprint(),
                "source_ordinal": retained.source_ordinal,
                "timestamp_ns": retained.timestamp_ns,
                "kind": "admit",
                "semantic_action": "commit",
                "historical_eligibility": "eligible",
            }
        ],
    }
    trace_path = tmp_path / "zero-tick-catalog.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["retained_catalog_identity_membership_status"] == (
        "available_exact_terminal_union"
    )
    assert summary["executed_ticks"] == 0
    assert summary["eligibility_metrics_status"] == "unavailable"
    assert summary["eligibility_normalized_sampling_gini"] is None
    assert summary["realized_to_uniform_expected_ratio_median"] is None
    assert summary["starvation_gap_eligible_ticks_max"] is None


def test_sampling_telemetry_uses_exact_mutation_lane_events(tmp_path):
    from eligsir.runtime.replay_sampling_telemetry import (
        write_replay_sampling_telemetry,
    )

    row = {
        "iteration": 10,
        "executed": True,
        "optimizer_steps": 1,
        "supervision_views": 4,
        "snapshot": {"ingress_queue_depth": 0},
        "mutation_request": 64,
        "mutation": {
            "candidates": 8,
            "inserted": 5,
            "pruned": 2,
            "structural_mutation": True,
            "events": [
                {
                    "kind": "replay_depth",
                    "candidates": 8,
                    "inserted": 5,
                    "pruned": 0,
                    "relocated": 0,
                    "corrected": 0,
                },
                {
                    "kind": "mcmc",
                    "candidates": 0,
                    "inserted": 0,
                    "pruned": 2,
                    "relocated": 0,
                    "corrected": 0,
                },
            ],
        },
    }
    trace_path = tmp_path / "decisions.jsonl"
    trace_path.write_text(__import__("json").dumps(row) + "\n")

    summary = write_replay_sampling_telemetry(trace_path)

    assert summary["mcmc_structural_epochs"] == 1
    assert summary["live_depth_structural_epochs"] == 0
    assert summary["replay_depth_structural_epochs"] == 1
    assert summary["mutation_by_kind"]["replay_depth"]["inserted"] == 5
    assert summary["mutation_by_kind"]["mcmc"]["pruned"] == 2


def test_causal_replay_driver_uses_per_submission_ticks_then_one_drain_barrier():
    ticks = []
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda _frame, _tracking: None,
            optimizer_tick=lambda: ticks.append(len(ticks)),
            tracking_provider=ExternalTrackingProvider(),
        )
    )

    summary = CausalReplayDriver(
        application,
        _Adapter(),
        tick_policy=CausalReplayTickPolicy(ticks_per_submission=2),
    ).run((_Record(0), _Record(1)))

    assert summary.submitted == 2
    assert summary.ticks == 5
    assert ticks == list(range(5))


def test_paced_driver_keeps_mapping_until_the_eof_drain_barrier():
    from types import SimpleNamespace

    mapped = []
    optimizer = []
    ingress = CausalIngressTelemetryState(batch_size=1)
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda frame, _: mapped.append(frame.source),
            optimizer_tick=lambda: (
                optimizer.append(len(optimizer))
                or SimpleNamespace(optimizer_steps=1, executed=True)
            ),
            tracking_provider=ExternalTrackingProvider(),
            ingress_telemetry_provider=ingress,
        )
    )

    summary = PacedCausalReplayDriver(
        application,
        _Adapter(),
        source_hz=1000.0,
        source_buffer_frames=8,
        ingress_telemetry=ingress,
    ).run((_Record(0), _Record(1), _Record(2)))

    assert summary.submitted == 3
    assert summary.optimizer_steps >= summary.submitted + 1
    assert summary.ticks == len(optimizer)
    assert summary.zero_frame_ticks > 0
    assert summary.transport_dropped == 0
    assert summary.elapsed_s > 0.0
    assert summary.submission_mean_ms >= 0.0
    assert summary.mapping_active_s > 0.0
    assert summary.optimizer_steps_per_active_s > 0.0
    assert mapped == list(summary.source_identities)


def test_paced_driver_marks_input_exhausted_only_after_its_buffer_drains(monkeypatch):
    """EOF status must not hide a source lag while a producer buffer remains."""
    import eligsir.runtime.causal_replay as causal_replay

    class _DoneButBufferedProducer(_ImmediatePacedProducer):
        @property
        def complete(self):
            return self._index == len(self._packets)

        @property
        def queue_depth(self):
            return len(self._packets) - self._index

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _DoneButBufferedProducer)
    ingress = CausalIngressTelemetryState(batch_size=1)
    exhaustion_states = []
    submitted = []

    class _Session:
        def submit(self, packet):
            submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def tick(self):
            exhaustion_states.append(ingress.input_exhausted)
            return SimpleNamespace(
                optimizer_steps=1,
                executed=True,
                completed_supervision_sources=(
                    tuple(submitted) if len(exhaustion_states) >= 3 else ()
                ),
            )

    source_complete = []

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=1000.0,
        ingress_telemetry=ingress,
        source_complete_observer=lambda _: source_complete.append(
            len(exhaustion_states)
        ),
    ).run((_Record(0), _Record(1)))

    assert exhaustion_states[0] is False
    assert exhaustion_states[1:] == [True, True]
    assert source_complete == [3]
    assert [source.source_ordinal for source in summary.source_identities] == [0, 1]


def test_paced_driver_joins_exact_mapping_active_clock_to_each_tick():
    """Catch cross-batch plots accidentally using evaluation-paused wall time."""

    clocks = []

    class _Session:
        def submit(self, packet):
            return SimpleNamespace(event=SimpleNamespace(source=packet.source))

    class _Application:
        session = _Session()

        def tick(self):
            time.sleep(0.002)
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=4,
                iteration=len(clocks) + 1,
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=1000.0,
        source_buffer_frames=8,
        tick_clock_observer=lambda _tick, clock: clocks.append(clock),
    ).run((_Record(0), _Record(1)))

    assert len(clocks) == summary.ticks
    assert [clock.optimizer_steps for clock in clocks] == list(
        range(1, summary.optimizer_steps + 1)
    )
    assert [clock.supervision_views for clock in clocks] == [
        4 * index for index in range(1, summary.supervision_views // 4 + 1)
    ]
    assert all(clock.mapping_active_s > 0.0 for clock in clocks)
    assert all(
        later.mapping_active_s > earlier.mapping_active_s
        for earlier, later in zip(clocks, clocks[1:])
    )
    assert clocks[-1].mapping_active_s == pytest.approx(summary.mapping_active_s)


def test_paced_driver_clock_tracks_drained_source_ordinals_through_reserved_roles(
    monkeypatch,
):
    import eligsir.runtime.causal_replay as causal_replay

    monkeypatch.setattr(
        causal_replay, "_PacedPacketProducer", _ImmediatePacedProducer
    )
    clocks = []

    class _Session:
        def submit(self, packet):
            return SimpleNamespace(event=SimpleNamespace(source=packet.source))

    class _Application:
        session = _Session()

        def tick(self):
            return SimpleNamespace(optimizer_steps=1, supervision_views=1)

    PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=1000.0,
        packet_role=lambda packet: (
            "holdout"
            if packet.source.source_ordinal == 400
            else "guard"
            if packet.source.source_ordinal == 900
            else "train"
        ),
        tick_clock_observer=lambda _tick, clock: clocks.append(clock),
    ).run((_Record(100), _Record(400), _Record(900)))

    assert [clock.source_frame_index for clock in clocks[:3]] == [100, 400, 900]


def test_paced_summary_histograms_requested_and_realized_constellations():
    class _Session:
        def submit(self, packet):
            return SimpleNamespace(event=SimpleNamespace(source=packet.source))

    class _Application:
        session = _Session()

        def tick(self):
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=4,
                iteration=1,
                work_plan=SimpleNamespace(
                    requested_supervision_slots=1,
                    supervision_constellation=SimpleNamespace(
                        requested_batch_size=4,
                        requested_camera_count=4,
                        realized_batch_size=4,
                        realized_camera_count=4,
                        mode="true_camera_batch",
                        reason="fixed_constellation_ablation",
                    ),
                ),
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=1000.0,
        source_buffer_frames=8,
    ).run((_Record(0), _Record(1)))

    assert summary.requested_supervision_constellations == {
        "B4_C4": summary.optimizer_steps
    }
    assert summary.realized_supervision_constellations == {
        "B4_C4": summary.optimizer_steps
    }
    assert summary.supervision_constellation_modes == {
        "true_camera_batch": summary.optimizer_steps
    }
    assert summary.supervision_constellation_reasons == {
        "fixed_constellation_ablation": summary.optimizer_steps
    }


def test_paced_driver_drains_three_only_after_typed_constellation_request():
    submitted_per_tick = []
    submitted_since_tick = []

    class _Session:
        def submit(self, packet):
            submitted_since_tick.append(packet.source.source_ordinal)
            return SimpleNamespace(event=SimpleNamespace(source=packet.source))

    class _Application:
        session = _Session()

        def tick(self):
            submitted_per_tick.append(tuple(submitted_since_tick))
            submitted_since_tick.clear()
            time.sleep(0.002)
            requested = 3 if len(submitted_per_tick) >= 1 else 1
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=4,
                iteration=len(submitted_per_tick),
                work_plan=SimpleNamespace(requested_supervision_slots=requested),
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=10_000.0,
        source_buffer_frames=64,
        max_new_sources_per_tick=3,
    ).run(tuple(_Record(index) for index in range(12)))

    assert summary.submitted == 12
    assert submitted_per_tick[0] == (0,)
    assert any(len(batch) == 3 for batch in submitted_per_tick[1:])
    assert tuple(
        ordinal for batch in submitted_per_tick for ordinal in batch
    ) == tuple(range(12))


def test_paced_driver_waits_for_initial_packet_before_first_mapper_tick():
    """Startup may wait for decode, but may not invent an empty first tick.

    The first source is scheduled at t=0.  If its decode worker has not run
    before the mapper observes the producer queue, an ordinary 1 ms dequeue
    timeout used to let the mapper tick with no source at all.  Later source
    gaps may legitimately produce drain ticks; the startup transaction has no
    preceding request, so it must instead wait until packet 0 is ready (or
    the finite source is terminal).
    """

    decode_started = threading.Event()
    release_decode = threading.Event()
    first_tick = threading.Event()
    submitted_since_tick: list[int] = []
    submitted_per_tick: list[tuple[int, ...]] = []

    class _DelayedInitialAdapter(_Adapter):
        def packet_from_record(self, record: _Record) -> SensorPacket:
            if record.ordinal == 0:
                decode_started.set()
                assert release_decode.wait(timeout=1.0), "test failed to release decode"
            return super().packet_from_record(record)

    class _Session:
        def submit(self, packet):
            submitted_since_tick.append(packet.source.source_ordinal)
            return SimpleNamespace(event=SimpleNamespace(source=packet.source))

    class _Application:
        session = _Session()

        def tick(self):
            batch = tuple(submitted_since_tick)
            submitted_per_tick.append(batch)
            submitted_since_tick.clear()
            first_tick.set()
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=1,
                iteration=len(submitted_per_tick),
                work_plan=SimpleNamespace(
                    requested_supervision_slots=1,
                    supervision_sources=tuple(
                        _Adapter().packet_from_record(_Record(ordinal)).source
                        for ordinal in batch
                    ),
                    replay_sources=(),
                ),
            )

    result: list[object] = []
    driver_thread = threading.Thread(
        target=lambda: result.append(
            PacedCausalReplayDriver(
                _Application(),
                _DelayedInitialAdapter(),
                source_hz=10_000.0,
                source_buffer_frames=4,
            ).run((_Record(0),))
        ),
    )
    driver_thread.start()
    try:
        assert decode_started.wait(timeout=1.0)
        assert not first_tick.wait(timeout=0.05)
    finally:
        release_decode.set()
        driver_thread.join(timeout=2.0)
    assert not driver_thread.is_alive()
    assert len(result) == 1
    assert submitted_per_tick[0] == (0,)


def test_new_biased_drain_counts_admitted_keyframes_not_raw_packets(monkeypatch):
    import eligsir.runtime.causal_replay as causal_replay

    submitted_per_tick = []
    submitted_since_tick = []

    class _FullyPublishedProducer(_ImmediatePacedProducer):
        @property
        def queue_depth(self):
            return len(self._packets) - self._index

        @property
        def capacity(self):
            return len(self._packets)

    class _Session:
        def submit(self, packet):
            ordinal = packet.source.source_ordinal
            admitted = ordinal % 5 == 0
            submitted_since_tick.append(ordinal)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                accepted=True,
                disposition=SimpleNamespace(admit_supervision=admitted),
            )

    class _Application:
        session = _Session()

        def tick(self):
            batch = tuple(submitted_since_tick)
            submitted_per_tick.append(batch)
            submitted_since_tick.clear()
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=4,
                iteration=len(submitted_per_tick),
                work_plan=SimpleNamespace(
                    requested_supervision_slots=3,
                    supervision_sources=tuple(
                        _Adapter().packet_from_record(_Record(ordinal)).source
                        for ordinal in batch
                        if ordinal % 5 == 0
                    ),
                    replay_sources=(),
                ),
            )

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _FullyPublishedProducer)

    PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=30.0,
        source_buffer_frames=128,
        max_new_sources_per_tick=3,
        max_source_packets_per_tick=48,
    ).run(tuple(_Record(index) for index in range(40)))

    nonempty_batches = [batch for batch in submitted_per_tick if batch]
    # The initial one-source budget is followed by three semantic-admission
    # drains; the raw cap is intentionally higher than each 15-packet drain.
    assert nonempty_batches == [
        (0,),
        tuple(range(1, 16)),
        tuple(range(16, 31)),
        tuple(range(31, 40)),
    ]


def test_paced_driver_never_submits_reserved_holdout_or_guard_identities():
    mapped = []
    source_complete = []
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda frame, _: mapped.append(frame.source.source_ordinal),
            optimizer_tick=lambda: SimpleNamespace(
                optimizer_steps=1, supervision_views=1, iteration=len(mapped)
            ),
            tracking_provider=ExternalTrackingProvider(),
        )
    )

    summary = PacedCausalReplayDriver(
        application,
        _Adapter(),
        source_hz=1000.0,
        source_buffer_frames=16,
        packet_role=lambda packet: (
            "holdout" if packet.source.source_ordinal == 2 else
            "guard" if packet.source.source_ordinal == 3 else
            "train"
        ),
        source_complete_observer=lambda result: source_complete.append(result),
    ).run(tuple(_Record(i) for i in range(6)))

    assert mapped == [0, 1, 4, 5]
    assert summary.submitted == 4
    assert summary.holdout_reserved == 1
    assert summary.guard_withheld == 1
    assert summary.source_offered == 6
    assert summary.last_ingest_iteration is not None
    assert len(source_complete) == 1


def test_paced_driver_persists_terminal_role_coverage_by_identity(monkeypatch):
    import eligsir.runtime.causal_replay as causal_replay

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", _ImmediatePacedProducer)
    ledger = CoverageLedger.fresh(
        resolved_source_frames=6,
        raw_registry_source_frames=None,
        source_denominator_provenance="causal_source_catalog",
        bootstrap_frames_consumed=0,
        source_identity="/data/fr3",
        effective_stride=1,
    )

    class _Session:
        def __init__(self):
            self.submitted = []

        def submit(self, packet):
            self.submitted.append(packet.source)
            admitted = packet.source.source_ordinal in {0, 4}
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=admitted),
            )

    class _Application:
        def __init__(self):
            self.session = _Session()
            self.completed = set()
            self.iteration = 0

        def tick(self):
            self.iteration += 1
            pending = [
                source
                for source in self.session.submitted
                if source.source_ordinal not in self.completed
            ]
            completed = pending[:1]
            self.completed.update(source.source_ordinal for source in completed)
            return SimpleNamespace(
                executed=True,
                optimizer_steps=1,
                supervision_views=1,
                iteration=self.iteration,
                completed_supervision_sources=tuple(completed),
            )

    summary = PacedCausalReplayDriver(
        _Application(),
        _Adapter(),
        source_hz=1000.0,
        coverage_ledger=ledger,
        bootstrap_frames=1,
        packet_role=lambda packet: (
            "holdout" if packet.source.source_ordinal == 2 else
            "guard" if packet.source.source_ordinal == 3 else
            "train"
        ),
    ).run(tuple(_Record(i) for i in range(6)))

    assert summary.coverage_ledger is not None
    assert summary.coverage_ledger["source_frames_ingested_total"] == 6
    assert summary.coverage_ledger["bootstrap_frames_consumed"] == 1
    assert summary.coverage_ledger["reserved_holdout_frames"] == 1
    assert summary.coverage_ledger["reserved_guard_frames"] == 1
    assert summary.coverage_ledger["eligible_mapping_frames"] == 3
    assert summary.coverage_ledger["frames_admitted"] == 1
    assert summary.coverage_ledger["mapping_skipped_frames"] == 2
    assert summary.coverage_ledger["unprocessed_frames"] == 0
    assert summary.coverage_ledger["causal_source_accounting"]["unique_source_count"] == 6


def test_stalled_paced_startup_preserves_roles_before_semantic_completion():
    submitted = []
    completed = []
    source_complete = []
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()

    class _StalledAdapter(_Adapter):
        def packet_from_record(self, record):
            if record.ordinal == 0:
                first_decode_entered.set()
                release_first_decode.wait()
            return super().packet_from_record(record)

    class _Session:
        def submit(self, packet):
            submitted.append(packet.source)
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class _Application:
        session = _Session()

        def tick(self):
            pending = tuple(source for source in submitted if source not in completed)
            if pending and len(submitted) == 2:
                completed.append(pending[0])
            return SimpleNamespace(
                executed=bool(pending and len(submitted) == 2),
                optimizer_steps=int(bool(pending and len(submitted) == 2)),
                supervision_views=int(bool(pending and len(submitted) == 2)),
                iteration=len(completed),
                completed_supervision_sources=tuple(completed[-1:]),
                work_plan=SimpleNamespace(requested_supervision_slots=1),
            )
    result = []

    def run():
        result.append(
            PacedCausalReplayDriver(
                _Application(),
                _StalledAdapter(),
                source_hz=1000.0,
                source_buffer_frames=2,
                packet_role=lambda packet: (
                    "holdout" if packet.source.source_ordinal == 1 else
                    "guard" if packet.source.source_ordinal == 2 else "train"
                ),
                source_complete_observer=lambda _tick: source_complete.append(
                    tuple(source.source_ordinal for source in completed)
                ),
            ).run(tuple(_Record(i) for i in range(4)))
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert first_decode_entered.wait(2.0)
    release_first_decode.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert [source.source_ordinal for source in submitted] == [0, 3]
    assert result[0].holdout_reserved == result[0].guard_withheld == 1
    assert source_complete == [(0, 3)]


def test_paced_source_clock_is_independent_from_slow_dataset_decode():
    # The property under test is that the paced source clock tracks the
    # configured source rate rather than dataset decode latency: decode runs
    # on a bounded thread pool alongside (not serialized with) the pacing
    # loop, so a decoder ten times slower than the pace interval must not
    # proportionally slow the offer rate. An absolute wall-clock fps floor
    # is flaky on a shared/loaded runner because host scheduling jitter
    # alone can push offered fps below any fixed threshold even with zero
    # decode-coupling. Instead we run a fast-decode baseline immediately
    # before the slow-decode case, in the same process, and compare the two
    # offered rates: only decode-latency coupling (not ambient host load,
    # which affects both runs alike) can make the slow run's rate collapse
    # relative to the baseline.
    class SlowAdapter:
        def packet_from_record(self, record):
            time.sleep(0.02)
            return _Adapter().packet_from_record(record)

    def run_once(adapter):
        application = create_mapping_application(
            MappingApplicationOptions(
                mapper_sink=lambda _frame, _tracking: None,
                optimizer_tick=lambda: None,
                tracking_provider=ExternalTrackingProvider(),
            )
        )
        return PacedCausalReplayDriver(
            application,
            adapter,
            source_hz=100.0,
            source_buffer_frames=64,
        ).run(tuple(_Record(index) for index in range(20)))

    baseline_summary = run_once(_Adapter())
    slow_summary = run_once(SlowAdapter())

    assert baseline_summary.submitted == 20
    assert baseline_summary.transport_dropped == 0
    assert baseline_summary.source_offered_fps is not None

    assert slow_summary.submitted == 20
    assert slow_summary.transport_dropped == 0
    assert slow_summary.source_offered_fps is not None

    # A decode-latency-coupled pacing bug would cut the offered rate roughly
    # in half or worse (0.02s decode vs the 0.01s pace interval, serialized
    # against a 2-worker decode pool). Ambient host load slows both runs
    # together, so a generous 70% floor on the ratio still catches real
    # coupling regressions without keying on an absolute fps number.
    assert slow_summary.source_offered_fps >= 0.7 * baseline_summary.source_offered_fps


def test_paced_driver_emits_low_rate_scalar_progress_without_decision_trace():
    progress = []
    application = create_mapping_application(
        MappingApplicationOptions(
            mapper_sink=lambda _frame, _tracking: None,
            optimizer_tick=lambda: SimpleNamespace(
                optimizer_steps=1, supervision_views=4, executed=True
            ),
            tracking_provider=ExternalTrackingProvider(),
        )
    )

    PacedCausalReplayDriver(
        application,
        _Adapter(),
        source_hz=1000.0,
        source_buffer_frames=8,
        progress_interval_s=1e-9,
        progress_sink=progress.append,
    ).run((_Record(0), _Record(1), _Record(2)))

    assert progress
    assert "optimizer_steps=" in progress[-1]
    assert "supervision_views_per_s=" in progress[-1]
    assert "transport_dropped=" in progress[-1]


def test_prepared_progress_reports_exact_b4_fill_and_ready_counts():
    counters = SimpleNamespace(
        fresh_pending=2,
        replay_pending=11,
        in_flight=4,
        full_not_resident_count=3,
        fresh_ready_count=1,
        replay_ready_count=2,
        ready_bytes_current=536870912,  # 512 MiB
        ready_bytes_peak=1073741824,    # 1024 MiB
    )
    tick = SimpleNamespace(
        state=SimpleNamespace(value="starved"),
        reason="prepared_authorized_sources_not_ready",
        ready_counters=counters,
        ready_count=3,
        selection=SimpleNamespace(leases=()),
        work_plan=SimpleNamespace(
            supervision_constellation=SimpleNamespace(realized_batch_size=4)
        ),
    )

    assert _prepared_ready_progress_fields(tick) == (
        " prepared_b4=0/4 prepared_exact_ready=3/4 "
        "ready_post_select=f1,r2 replay_ready_batches=0 "
        "lane_occupancy=f2,r11,i4,nonresident=3 "
        "ready_bytes=512.0/1024.0MiB "
        "prepared_state=starved "
        "prepared_reason=prepared_authorized_sources_not_ready"
    )


def test_prepared_progress_distinguishes_ready_counts_from_occupancy():
    """Verify that progress line reports READY counts, not lane occupancy.

    When 3 replay leases are pending but only 1 is READY (has no pending event),
    the progress line should report ready_count=1, not occupancy=3.
    """

    from eligsir.runtime.ready_source import ReadySourceOwner, ReadySourceLane
    from eligsir.runtime.prepared_source import PreparedSourceLease
    from eligsir.contracts import ModalityEligibility, BufferRef, BufferSpec, MemoryDomain, BufferOwnership

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    producer = owner.claim_producer()

    # Create 3 replay leases: 1 ready (event=None), 2 not ready (un-fired events)
    def make_lease(ordinal, ready_event):
        payload_token = f"token_{ordinal}"
        payload = BufferRef(
            data=torch.zeros(2),
            spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
            ownership=BufferOwnership.BORROWED,
            owner_token=payload_token,
        )
        return PreparedSourceLease(
            source_identity=SourceIdentity("test", 0, ordinal, 1_000_000_000),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
            source_metadata={"width": 8, "height": 6},
            tier_preparation_available=("native",),
            borrowed_device_payloads={"rgb": payload},
            device="cpu",
            generation=0,
            lifetime_token=payload_token,
            ready_event=ready_event,
        )

    # Create simple gate objects for ready_event
    class ReadyGate:
        def __init__(self, is_ready: bool = False):
            self.is_ready = is_ready

        def query(self) -> bool:
            return self.is_ready

    lease_ready = make_lease(0, None)  # None means immediately READY
    lease_not_ready_1 = make_lease(1, ReadyGate(is_ready=False))  # Pending event
    lease_not_ready_2 = make_lease(2, ReadyGate(is_ready=False))  # Pending event

    # Offer all 3 leases to replay lane
    producer.offer(lease_ready, lane=ReadySourceLane.REPLAY)
    producer.offer(lease_not_ready_1, lane=ReadySourceLane.REPLAY)
    producer.offer(lease_not_ready_2, lane=ReadySourceLane.REPLAY)

    # Verify occupancy and ready counts
    counters = owner.counters
    assert counters.replay_pending == 3, f"Expected 3 pending, got {counters.replay_pending}"
    assert counters.replay_ready_count == 1, f"Expected 1 ready, got {counters.replay_ready_count}"

    # Verify the progress line reports the READY count, not occupancy
    tick = SimpleNamespace(
        state=SimpleNamespace(value="starved"),
        reason="test",
        ready_counters=counters,
        ready_count=None,
        selection=SimpleNamespace(leases=()),
        work_plan=SimpleNamespace(
            supervision_constellation=SimpleNamespace(realized_batch_size=4)
        ),
    )

    progress = _prepared_ready_progress_fields(tick)
    # Should report ready_post_select=f0,r1 (ready count), not the occupancy value of 3
    assert "ready_post_select=f0,r1" in progress, f"Expected ready_post_select=f0,r1 in {progress}"
    assert "lane_occupancy=f0,r3" in progress, f"Expected lane_occupancy=f0,r3 in {progress}"


def test_causal_ingress_telemetry_exposes_raw_queue_pressure_and_transport_loss():
    state = CausalIngressTelemetryState(batch_size=4)
    packet = _Adapter().packet_from_record(_Record(0))

    state.publish(queue_depth=7, capacity=256, dropped=2)
    high = {fact.name: fact.value for fact in state(packet, packet.tracking).evidence}
    state.publish(queue_depth=8, capacity=256, dropped=2)
    critical = {
        fact.name: fact.value for fact in state(packet, packet.tracking).evidence
    }

    assert high == {
        "transport.queue_depth": 7,
        "transport.queue_capacity": 256,
        "transport.dropped": 2,
        "pressure_state": "high",
    }
    assert critical["pressure_state"] == "critical"


def test_causal_entry_sizes_ingress_pressure_threshold_from_true_camera_batch():
    from eligsir.runtime.causal_entry import ingress_telemetry_from_config

    state = ingress_telemetry_from_config(
        SimpleNamespace(hard_limits={"optimization_batch_size": 4}), paced=True
    )
    packet = _Adapter().packet_from_record(_Record(0))

    state.publish(queue_depth=2, capacity=256, dropped=0)
    evidence = {fact.name: fact.value for fact in state(packet, None).evidence}

    assert evidence["pressure_state"] == "low"


def test_streaming_frame_packet_adapter_owns_dataset_decode_and_metric_depth():
    record = StreamingRGBDFrame(
        index=7,
        timestamp=1.5,
        rgb_path="",
        c2w=np.eye(4),
        fx=10.0,
        fy=11.0,
        cx=2.0,
        cy=3.0,
        width=4,
        height=3,
        depth_scale=5000.0,
        _rgb_np=np.zeros((3, 4, 3), dtype=np.uint8),
        _depth_np=np.full((3, 4), 5000, dtype=np.uint16),
        source_ordinal=42,
    )

    packet = StreamingFramePacketAdapter("tum", stream_epoch=2).packet_from_record(record)

    assert packet.source == SourceIdentity("tum", 2, 42, 1_500_000_000)
    assert packet.depth is not None
    assert packet.depth.spec.units == "m"
    assert np.allclose(packet.depth.data, 1.0)
    assert packet.tracking is not None
    assert packet.tracking.provider == "dataset_pose"


def test_decision_trace_separates_legacy_adaptive_growth_from_live_policy_disposition():
    row = decision_trace_row(
        SimpleNamespace(
            iteration=1,
            executed=True,
            control_snapshot=None,
            work_plan=None,
            mutation_events=(),
            adaptive_growth_enabled=True,
            population_policy_disposition="missing_belief",
        )
    )

    assert row["policy"]["legacy_adaptive_growth_enabled"] is True
    assert row["policy"]["population_policy_disposition"] == "missing_belief"


@pytest.mark.parametrize(("batch_size", "camera_count"), ((1, 1), (4, 4), (16, 4)))
def test_jsonl_projects_replay_challenger_trace_with_constellation_and_completeness(
    tmp_path, batch_size, camera_count
):
    sources = tuple(
        SourceIdentity("train", 0, ordinal, ordinal)
        for ordinal in range(batch_size)
    )
    constellation = SupervisionConstellation(
        requested_batch_size=batch_size,
        requested_camera_count=camera_count,
        realized_batch_size=batch_size,
        realized_camera_count=camera_count,
        unique_eligible_count=batch_size,
        fill_ratio=1.0,
        microbatch_count=(batch_size + camera_count - 1) // camera_count,
        mode="fixed",
        reason="test",
        hysteresis_state="hold",
    )
    plan = SimpleNamespace(
        plan_id=f"trace-B{batch_size}-C{camera_count}",
        supervision_sources=sources,
        replay_sources=(),
        replay_trial_sources=(),
        supervision_constellation=constellation,
        reason_trace=(),
        insertion=SimpleNamespace(
            source=None, sample_stride=1, candidate_budget=0, seed_budget=0
        ),
        mutation_budget=SimpleNamespace(max_total=0),
        resolution_tier="native",
        native_geometry_evidence_purpose=False,
        requested_supervision_slots=batch_size,
        requested_replay_slots=0,
        replay_population_size=0,
        replay_sampling_seed=0,
        replay_sampling_epoch=0,
        replay_selected_weights=(),
    )
    tick = SimpleNamespace(
        work_plan=plan,
        selection=SimpleNamespace(
            leases=tuple(object() for _ in sources),
            fresh_count=batch_size,
            replay_count=0,
            starved=False,
        ),
        microbatch_sizes=constellation.microbatch_sizes,
        replay_challenger_trace=(
            {
                "stage": "arena",
                "source": sources[-1].fingerprint(),
                "outcome": "promote",
                "snapshot": {"parameter": 7},
            },
        ),
        replay_challenger_trace_truncated=False,
    )

    observer = JsonlDecisionTraceObserver(
        tmp_path / f"B{batch_size}-C{camera_count}.jsonl"
    )
    observer(tick)
    observer.flush()
    saved = json.loads(
        (tmp_path / f"B{batch_size}-C{camera_count}.jsonl").read_text()
    )

    assert saved["ready_batch"] == {
        "requested": batch_size,
        "selected": batch_size,
        "fresh": batch_size,
        "replay": 0,
        "full": True,
        "waited_for_ready_source": False,
        "starved": False,
    }
    assert saved["supervision_constellation"] == {
        "requested_batch_size": batch_size,
        "requested_camera_count": camera_count,
        "realized_batch_size": batch_size,
        "realized_camera_count": camera_count,
        "microbatch_sizes": list(constellation.microbatch_sizes),
    }
    assert (
        saved["replay_challenger_trace"][0]["source"]
        == sources[-1].fingerprint()
    )
    assert saved["replay_challenger_trace_truncated"] is False
    assert saved["replay_challenger_trace_complete"] is True


def test_jsonl_marks_replay_challenger_trace_incomplete_when_capped():
    tick = SimpleNamespace(
        work_plan=None,
        replay_challenger_trace=({"stage": "turnover", "outcome": "retain"},),
        replay_challenger_trace_truncated=True,
    )
    row = decision_trace_row(tick)

    assert row["replay_challenger_trace"] == [
        {"stage": "turnover", "outcome": "retain"}
    ]
    assert row["replay_challenger_trace_truncated"] is True
    assert row["replay_challenger_trace_complete"] is False


def test_decision_trace_preserves_measured_zero_insertion_filter_receipt():
    source = SourceIdentity("dataset", 0, 7, 7_000)
    row = decision_trace_row(
        SimpleNamespace(
            iteration=3,
            executed=True,
            control_snapshot=None,
            work_plan=None,
            mutation_events=(),
            insertion_filter={
                "source": source.fingerprint(),
                "stats": {
                    "raw_candidates": 0,
                    "after_alpha_depth_mask": 0,
                    "after_grazing_filter": 0,
                    "device_native_candidates": 0,
                },
            },
        )
    )

    assert row["insertion_filter"] == {
        "source": source.fingerprint(),
        "stats": {
            "raw_candidates": 0,
            "after_alpha_depth_mask": 0,
            "after_grazing_filter": 0,
            "device_native_candidates": 0,
        },
    }


def test_decision_trace_serializes_bounded_population_selection_telemetry():
    summary = PopulationShadowSummary(
        transaction_id="b4",
        epoch=8,
        units=1,
        versions=None,
        demand_count=0,
        action_counts_cpu=torch.tensor([1, 0, 0, 0, 1, 0, 0]),
        drop_reason=None,
        native_target_count=1,
        native_target_status_cpu=torch.tensor([2], dtype=torch.int8),
        selected_action_histogram_cpu=torch.tensor([1, 0, 0, 0, 1, 0, 0]),
        adapter_plan_drop_reason="capacity_conflict",
        decision_token=17,
    )
    tick = SimpleNamespace(
        iteration=2,
        executed=True,
        control_snapshot=None,
        work_plan=None,
        mutation_events=(),
        population_policy_selection_telemetry=(
            CanonicalMappingRuntime._summary_cpu_values(summary),
        ),
    )
    row = decision_trace_row(tick)

    assert row["policy"]["selection_telemetry"] == [
        {
            "transaction_id": "b4",
            "epoch": 8,
            "units": 1,
            "decision_token": 17,
            "drop_reason": None,
            "native_target_count": 1,
            "native_target_status": [2],
            "selected_action_histogram": [1, 0, 0, 0, 1, 0, 0],
            "adapter_plan_drop_reason": "capacity_conflict",
            "adapter_rejected": None,
        }
    ]

    # The normal opt-in JSONL sink accepts the same bounded record without
    # exposing any tensor object to json.dumps.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        observer = JsonlDecisionTraceObserver(path)
        observer(tick)
        observer.flush()
        assert (
            json.loads(path.read_text())["policy"]["selection_telemetry"]
            == row["policy"]["selection_telemetry"]
        )


def test_decision_trace_preserves_identity_for_multiple_selection_summaries():
    first = PopulationShadowSummary(
        transaction_id="b4-1", epoch=8, units=1, versions=None,
        demand_count=0, action_counts_cpu=None, drop_reason=None,
        decision_token=3,
    )
    second = replace(
        first, transaction_id="b4-2", epoch=9, units=2, decision_token=4,
        drop_reason=PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT,
    )
    tick = SimpleNamespace(
        population_policy_selection_telemetry=tuple(
            CanonicalMappingRuntime._summary_cpu_values(summary)
            for summary in (first, second)
        )
    )

    records = decision_trace_row(tick)["policy"]["selection_telemetry"]

    assert [(item["transaction_id"], item["decision_token"]) for item in records] == [
        ("b4-1", 3), ("b4-2", 4)
    ]
    assert records[1]["drop_reason"] == "stale_current_snapshot"


def test_decision_trace_drops_unready_async_compact_fields_without_waiting():
    class _Unready:
        def try_take_cpu_tensor(self):
            return None

    summary = PopulationShadowSummary(
        transaction_id="b4-pending", epoch=8, units=1, versions=None,
        demand_count=1, action_counts_cpu=torch.tensor([1]), drop_reason=None,
        native_target_count=1, native_target_status_cpu=torch.tensor([2]),
        compact_packed_readback=_Unready(), decision_token=9,
    )
    record = CanonicalMappingRuntime._summary_cpu_values(summary)

    assert record["decision_token"] == 9
    assert record["selected_action_histogram"] is None
    assert record["native_target_status"] is None


def test_runtime_retries_unready_packed_summary_and_emits_once_after_poll():
    class _Readback:
        def __init__(self):
            self.ready = False
            self.takes = 0
            self.event = self

        def query(self):
            return self.ready

        def try_take_cpu_tensor(self):
            self.takes += 1
            if not self.ready:
                return None
            return torch.tensor([1, 0, 0, 0, 0, 0, 0, 2, 0], dtype=torch.uint8)

        def try_release_if_ready(self):
            return self.ready

    readback = _Readback()
    summary = PopulationShadowSummary(
        transaction_id="retry", epoch=8, units=1, versions=None,
        demand_count=1, action_counts_cpu=None, drop_reason=None,
        native_target_count=1, compact_packed_readback=readback, decision_token=4,
    )
    class _Bridge:
        def __init__(self):
            self.sent = False
            self.calls = 0

        def drain_async_summaries(self):
            self.calls += 1
            if self.sent:
                return ()
            self.sent = True
            return (summary,)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._population_selection_trace_enabled = True
    runtime._optimization_runtime = SimpleNamespace(
        population_shadow_runtime=_Bridge()
    )
    assert runtime._drain_population_selection_telemetry() == ()
    for _ in range(100):
        assert runtime._drain_population_selection_telemetry() == ()
    assert runtime._optimization_runtime.population_shadow_runtime.calls == 1
    readback.ready = True
    records = runtime._drain_population_selection_telemetry()
    assert len(records) == 1 and records[0]["decision_token"] == 4
    assert readback.takes == 1
    assert runtime._drain_population_selection_telemetry() == ()


def test_paced_decode_workers_passed_to_producer():
    """PacedCausalReplayDriver passes decode_workers kwarg to _PacedPacketProducer."""
    captured_kwargs = {}

    def mock_producer_init(self, *args, **kwargs):
        captured_kwargs.update(kwargs)

    class _Adapter:
        def packet_from_record(self, record):
            return SimpleNamespace(source=SimpleNamespace())

    class _Application:
        def tick(self):
            return SimpleNamespace(
                executed=True,
                optimizer_steps=0,
                supervision_views=0,
                iteration=0,
                work_plan=SimpleNamespace(
                    requested_supervision_slots=0,
                    supervision_sources=(),
                    replay_sources=(),
                ),
            )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "eligsir.runtime.causal_replay._PacedPacketProducer.__init__",
            mock_producer_init,
        )
        driver = PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=1.0,
            decode_workers=7,
        )
        try:
            driver.run(())
        except (KeyError, AttributeError, TypeError):
            # The mock doesn't implement full producer logic, but we only
            # care that it received the kwargs.
            pass

    assert captured_kwargs.get("decode_workers") == 7


def test_paced_decode_workers_validation_rejects_zero():
    """PacedCausalReplayDriver rejects decode_workers < 1."""
    class _Adapter:
        pass

    class _Application:
        pass

    with pytest.raises(ValueError, match="decode_workers must be a positive integer"):
        PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=1.0,
            decode_workers=0,
        )


def test_paced_decode_workers_validation_rejects_bool():
    """PacedCausalReplayDriver rejects decode_workers if bool is passed."""
    class _Adapter:
        pass

    class _Application:
        pass

    with pytest.raises(ValueError, match="decode_workers must be a positive integer"):
        PacedCausalReplayDriver(
            _Application(),
            _Adapter(),
            source_hz=1.0,
            decode_workers=True,
        )


def test_causal_replay_summary_includes_ready_bytes_fields():
    """Verify that CausalReplaySummary has ready_bytes fields for observability."""

    from eligsir.runtime.causal_replay import CausalReplaySummary

    # Create a summary with default ready_bytes values
    summary = CausalReplaySummary(
        submitted=10,
        ticks=5,
        source_identities=(),
    )

    # Verify the fields exist and have default values
    assert hasattr(summary, "ready_bytes_peak")
    assert hasattr(summary, "ready_bytes_current_at_end")
    assert hasattr(summary, "ready_bytes_by_tier_kind")

    assert summary.ready_bytes_peak == 0
    assert summary.ready_bytes_current_at_end == 0
    assert summary.ready_bytes_by_tier_kind == {}

    # Create a summary with explicit ready_bytes values
    summary2 = CausalReplaySummary(
        submitted=10,
        ticks=5,
        source_identities=(),
        ready_bytes_peak=1073741824,  # 1 GiB
        ready_bytes_current_at_end=536870912,  # 512 MiB
        ready_bytes_by_tier_kind={"native/rgb": 268435456, "native/depth": 268435456},
    )

    assert summary2.ready_bytes_peak == 1073741824
    assert summary2.ready_bytes_current_at_end == 536870912
    assert summary2.ready_bytes_by_tier_kind == {"native/rgb": 268435456, "native/depth": 268435456}


def test_prepared_insertion_counters_distinguish_execution_zero_seed_and_skip():
    from eligsir.runtime.causal_replay import _extract_prepared_insertion_counters

    source = SourceIdentity("dataset", 0, 0, 1_000)
    executed = SimpleNamespace(
        state=SimpleNamespace(value="executed"),
        prepared_insertion_skip=None,
        mutation_events=(SimpleNamespace(kind="live_depth", source=source, inserted=15),),
    )
    assert _extract_prepared_insertion_counters(executed) == (1, 15, 0, None)

    zero_seed = SimpleNamespace(
        state=SimpleNamespace(value="executed"),
        prepared_insertion_skip=None,
        mutation_events=(SimpleNamespace(kind="live_depth", source=source, inserted=0),),
    )
    assert _extract_prepared_insertion_counters(zero_seed) == (1, 0, 1, None)

    skipped = SimpleNamespace(
        state=SimpleNamespace(value="executed"),
        prepared_insertion_skip="render_reuse_unavailable:missing_render_lease",
        mutation_events=(),
    )
    assert _extract_prepared_insertion_counters(skipped) == (
        0,
        0,
        0,
        "render_reuse_unavailable:missing_render_lease",
    )


def test_causal_replay_summary_exposes_prepared_insertion_counters():
    from eligsir.runtime.causal_replay import CausalReplaySummary

    summary = CausalReplaySummary(
        submitted=1,
        ticks=1,
        source_identities=(SourceIdentity("dataset", 0, 0, 0),),
    )
    assert summary.prepared_insertions_executed == 0
    assert summary.prepared_insertion_seeds_accepted == 0
    assert summary.prepared_insertions_zero_seed == 0
    assert summary.prepared_insertion_skips == {}
    assert summary.prepared_insertion_skip_total == 0

    skipped = CausalReplaySummary(
        submitted=1,
        ticks=1,
        source_identities=(SourceIdentity("dataset", 0, 0, 0),),
        prepared_insertion_skips={"missing_render_lease": 2, "stale_snapshot": 1},
    )
    assert skipped.prepared_insertion_skip_total == 3
