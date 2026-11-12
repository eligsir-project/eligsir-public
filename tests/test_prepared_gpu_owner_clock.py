from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import ComposedMappingApplication
from eligsir.runtime.causal_replay import DatasetPacketAdapter, PacedCausalReplayDriver
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult
from eligsir.runtime.prepared_source_prep_worker import READY_WAIT_FALLBACK_S
from eligsir.runtime.ready_b4_streaming_adapter import (
    ReadyB4StreamingStepResult,
    ReadyB4StreamingStepState,
)
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection
from eligsir.tracking.providers import observation_from_pose_matrix


@dataclass(frozen=True)
class _Record:
    ordinal: int


class _Adapter(DatasetPacketAdapter[_Record]):
    def packet_from_record(self, record: _Record) -> SensorPacket:
        source = SourceIdentity("gpu-owner-clock", 0, record.ordinal, 1_000 + record.ordinal)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
            depth=None,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
        )


@pytest.mark.parametrize(
    ("first_state", "next_timeout", "expected_waits"),
    [
        (ReadyB4StreamingStepState.DEFERRED, 0.001, 0),
        (ReadyB4StreamingStepState.STARVED, 0.0, 1),
        (ReadyB4StreamingStepState.EXECUTED, 0.0, 0),
    ],
)
def test_prepared_gpu_owner_waits_only_for_the_matching_progress_source(
    monkeypatch, first_state, next_timeout, expected_waits,
):
    import eligsir.runtime.causal_replay as causal_replay

    source = _Adapter().packet_from_record(_Record(0)).source

    class Producer:
        last = None

        def __init__(self, records, adapter, **_kwargs):
            self._packets = [adapter.packet_from_record(record) for record in records]
            self.timeouts = []
            type(self).last = self

        def start(self):
            pass

        def wait_for_initial_packet_or_terminal(self):
            pass

        def take(self, *, timeout_s):
            self.timeouts.append(timeout_s)
            return self._packets.pop(0) if self._packets else None

        @property
        def complete(self):
            return not self._packets

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

    class Session:
        prepared_mode = True

        def __init__(self):
            self.submitted = 0

        def submit(self, packet):
            self.submitted += 1
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    first = ReadyB4StreamingStepResult(
        first_state, PreparedRuntimeBridgeResult(0, 0, 0, 0),
        ReadyOnlyTickSelection((), 0, 0, False, True),
        reason="prepared_work_plan_unavailable" if first_state is ReadyB4StreamingStepState.DEFERRED else None,
    )
    results = iter((first, SimpleNamespace(
        executed=True, retryable=False, optimizer_steps=1, supervision_views=4,
        completed_supervision_sources=(source,),
    )))
    waits = []
    application = object.__new__(ComposedMappingApplication)
    application._session = Session()
    application._closed = application._flush_required = application._prepared_retry_pending = False
    application._tick_count = 0
    application._options = SimpleNamespace(
        prepared_mode=True, prepared_tick=lambda: next(results),
        prepared_progress_waiter=lambda *, timeout_s: waits.append(timeout_s) or True,
    )

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", Producer)
    summary = PacedCausalReplayDriver(
        application,
        _Adapter(),
        source_hz=1000.0,
    ).run([_Record(0)])

    assert application.session.submitted == 1
    assert application.tick_count == 2
    assert summary.optimizer_steps == 1
    assert summary.supervision_views == 4
    assert Producer.last is not None
    # Source take stays a 1 ms poll (not notified by READY callbacks); only the
    # prepared-progress wait uses the bounded READY_WAIT_FALLBACK_S.
    assert Producer.last.timeouts == [pytest.approx(0.001), next_timeout]
    assert waits == [pytest.approx(READY_WAIT_FALLBACK_S)] * expected_waits


def test_prepared_gpu_owner_yields_for_ready_preparation_without_source_wait(
    monkeypatch,
):
    import eligsir.runtime.causal_replay as causal_replay

    source = _Adapter().packet_from_record(_Record(0)).source
    prepared = False
    waits: list[float] = []

    class Producer:
        last = None

        def __init__(self, records, adapter, **_kwargs):
            self._packets = [adapter.packet_from_record(record) for record in records]
            self.timeouts = []
            type(self).last = self

        def start(self):
            pass

        def wait_for_initial_packet_or_terminal(self):
            pass

        def take(self, *, timeout_s):
            self.timeouts.append(timeout_s)
            return self._packets.pop(0) if self._packets else None

        @property
        def complete(self):
            return not self._packets

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

    class Session:
        prepared_mode = True

        def submit(self, packet):
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class Application:
        def __init__(self):
            self.session = Session()
            self.ticks = 0

        def tick(self):
            self.ticks += 1
            if self.ticks == 1:
                return SimpleNamespace(
                    state=SimpleNamespace(value="starved"),
                    retryable=True,
                    executed=False,
                    optimizer_steps=0,
                    supervision_views=0,
                    completed_supervision_sources=(),
                )
            assert prepared, "retry ran before the READY-preparation yield"
            return SimpleNamespace(
                state=SimpleNamespace(value="executed"),
                executed=True,
                optimizer_steps=1,
                supervision_views=4,
                completed_supervision_sources=(source,),
            )

        def wait_for_prepared_progress(self, *, timeout_s):
            nonlocal prepared
            waits.append(timeout_s)
            prepared = True
            return True

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", Producer)
    application = Application()
    summary = PacedCausalReplayDriver(
        application,
        _Adapter(),
        source_hz=1000.0,
    ).run([_Record(0)])

    assert application.ticks == 2
    assert summary.optimizer_steps == 1
    assert Producer.last is not None
    assert Producer.last.timeouts == [pytest.approx(0.001), 0.0]
    assert waits == [pytest.approx(READY_WAIT_FALLBACK_S)]


def test_prepared_gpu_owner_fails_after_thirty_seconds_without_semantic_progress(
    monkeypatch,
):
    """EOF starvation must fail by elapsed time, not an arbitrary tick count."""

    import eligsir.runtime.causal_replay as causal_replay

    class Producer:
        def __init__(self, records, adapter, **_kwargs):
            self._packets = [adapter.packet_from_record(record) for record in records]

        def start(self):
            pass

        def wait_for_initial_packet_or_terminal(self):
            pass

        def take(self, *, timeout_s):
            return self._packets.pop(0) if self._packets else None

        @property
        def complete(self):
            return not self._packets

        queue_depth = 0
        capacity = 1
        dropped = dropped_at_arrival = dropped_at_enqueue = 0
        high_watermark = offered = 1
        offered_fps = 1.0

        def source_ingress_report_block(self):
            return {}

        def close(self):
            pass

    class Session:
        prepared_mode = True

        def submit(self, packet):
            return SimpleNamespace(
                event=SimpleNamespace(source=packet.source),
                disposition=SimpleNamespace(admit_supervision=True),
            )

    class Application:
        session = Session()

        def tick(self):
            return SimpleNamespace(
                state=SimpleNamespace(value="starved"),
                reason="prepared_authorized_sources_not_ready",
                executed=False,
                optimizer_steps=0,
                supervision_views=0,
                completed_supervision_sources=(),
            )

        def wait_for_prepared_progress(self, *, timeout_s):
            return False

    now = 0.0

    def clock():
        nonlocal now
        now += 1.0
        return now

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", Producer)
    monkeypatch.setattr(causal_replay.time, "perf_counter", clock)

    with pytest.raises(RuntimeError, match="no semantic progress for 30s"):
        PacedCausalReplayDriver(Application(), _Adapter(), source_hz=1_000.0).run(
            [_Record(0)]
        )
