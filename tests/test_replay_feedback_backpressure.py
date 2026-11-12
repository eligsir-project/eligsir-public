"""Shared transfer pressure must defer work without losing source ownership."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts.control import SemanticAction
from eligsir.runtime.causal_prepared_ready import CausalPreparedReadyB4Path
from eligsir.runtime.prepared_cold_promotion import PreparedColdPromotionOwner
from eligsir.runtime.prepared_ingress import PreparedSourceIngressHandoff
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge, PreparedRuntimeBridgeResult
from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer, PreparedSourceProduceResult
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner
from test_prepared_cold_promotion import _candidate, _resolution
from test_prepared_runtime_bridge import controlled_item


class Gate:
    ready = False

    def query(self):
        return self.ready

    def wait(self, *args):
        raise AssertionError("host wait is forbidden")

    synchronize = wait

    def wait_on_current_stream(self, device):
        assert str(device) == "cpu"  # CPU copy already happened; only ownership is delayed.


def bridge(transfer):
    ready = ReadySourceOwner(max_fresh=4, max_replay=4)
    return PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=4),
        PreparedSourceProducer(ready, transfer_owner=transfer),
        ready_owner=ready,
    ), ready


def saturated_transfer(monkeypatch):
    """Real CPU copies with independently delayed H2D/D2H completion events."""
    transfer = DeviceTransferOwner("cpu", max_inflight=4)
    for _ in range(4):
        transfer.upload_packed((np.zeros(64, np.float32),), purpose="warmup")
    stream = transfer.transfer_stream
    d2h, h2d = stream.d2h_into, stream.h2d
    feedback_gate, upload_gate = Gate(), Gate()

    def readback(value, host):
        d2h(value, host)
        return feedback_gate

    def upload(host, **kwargs):
        device, _ = h2d(host, **kwargs)
        return device, upload_gate

    monkeypatch.setattr(stream, "d2h_into", readback)
    feedback = transfer.readback_supervision_evidence(torch.ones((4, 4)))
    monkeypatch.setattr(stream, "h2d", upload)
    for _ in range(3):
        transfer.upload_packed((np.zeros(64, np.float32),), purpose="burst")
    return transfer, feedback, feedback_gate


def test_full_burst_defers_fresh_source_and_retries_same_identity(monkeypatch):
    transfer, feedback, gate = saturated_transfer(monkeypatch)
    staged, ready = bridge(transfer)
    item = controlled_item(17, SemanticAction.COMMIT)
    staged.handoff.claim_producer("session").try_offer(item)
    assert staged.drain() == PreparedRuntimeBridgeResult(1, 0, 1, 0)
    assert staged._deferred_item is item
    assert not ready.claim_scheduler("test").has_source(item.packet.source)
    assert feedback.try_take_cpu_tensor() is None
    gate.ready = True
    torch.testing.assert_close(feedback.try_take_cpu_tensor(), torch.ones(16))
    assert staged.drain() == PreparedRuntimeBridgeResult(1, 1, 0, 0)
    assert staged._deferred_item is None
    lease = ready.claim_consumer("test").poll(ReadySourceLane.FRESH)
    assert lease.source_identity == item.packet.source
    assert lease.control_token is item.control_token
    assert lease.generation == 0
    assert staged.drain() == PreparedRuntimeBridgeResult(0, 0, 0, 0)
    assert transfer.stats.explicit_sync_waits == 0


def test_full_burst_retains_cold_request_and_generation(monkeypatch):
    transfer, feedback, gate = saturated_transfer(monkeypatch)
    staged, ready = bridge(transfer)
    resolution = _resolution(18, action=SemanticAction.DEFER)
    promotion = PreparedColdPromotionOwner(
        staged, ready_scheduler=ready.claim_scheduler("promotion"),
        resolve=lambda candidate: resolution,
    )
    assert promotion.schedule_shortlist((_candidate(18),), promotion_fence=(4, 5, 6)) == 1
    request = promotion._scheduled[18]
    assert promotion.pump_once() is False
    assert promotion._scheduled[18] is request
    gate.ready = True
    assert feedback.try_take_cpu_tensor() is not None
    assert promotion.pump_once() is True
    assert not promotion._scheduled
    lease = ready.claim_consumer("test").poll(ReadySourceLane.REPLAY)
    assert lease.source_identity == resolution.item.packet.source
    assert lease.generation == request.generation == 1
    assert transfer.stats.explicit_sync_waits == 0


def test_starved_consumer_drains_feedback_before_planning(monkeypatch):
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    transfer, feedback, gate = saturated_transfer(monkeypatch)
    staged, _ = bridge(transfer)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.initialized = True
    items = [controlled_item(index, SemanticAction.COMMIT) for index in range(4)]
    rows = tuple((item.packet.source, item.disposition.modality_eligibility, (None, None)) for item in items)
    runtime._pending_replay_evidence = (feedback, rows, frozenset((items[0].packet.source,)))
    runtime._optimization_policy = SimpleNamespace()
    runtime._hot_cold_challenger_arena = None
    runtime._historical_frame_ids_by_source = {}
    planned = []

    def plan():
        planned.append(runtime._pending_replay_evidence is None)
        return None

    path = CausalPreparedReadyB4Path(
        runtime=runtime, bridge=staged,
        adapter_factory=lambda: SimpleNamespace(tick=lambda _: SimpleNamespace(state="starved")),
        step_inputs_factory=plan,
    )
    assert path.tick().state == "starved"
    assert runtime._pending_replay_evidence is not None
    gate.ready = True
    assert path.tick().state == "starved"
    assert runtime._pending_replay_evidence is None
    assert transfer._packed_pool.pending_count == 3
    assert planned == [False, True]
    assert transfer.stats.explicit_sync_waits == 0


def test_deferred_fresh_work_uses_condition_wait_not_progress_spin(monkeypatch):
    transfer, _, _ = saturated_transfer(monkeypatch)
    staged, _ = bridge(transfer)
    item = controlled_item(22, SemanticAction.COMMIT)
    staged.handoff.claim_producer("session").try_offer(item)
    waiting = threading.Event()
    calls = []
    original_drain = staged.drain

    def drain():
        calls.append(1)
        assert len(calls) == 1, "deferred work spun instead of waiting"
        return original_drain()

    monkeypatch.setattr(staged, "drain", drain)
    worker = PreparedSourcePrepWorker(staged, poll_interval_s=10.0)
    original_wait = worker._progress.wait

    def wait(timeout):
        assert timeout == 10.0
        waiting.set()
        return original_wait(timeout)

    worker._progress.wait = wait
    worker.start()
    try:
        assert waiting.wait(1), "deferred source was incorrectly reported as progress"
        with worker._progress:  # Acquired only after the real condition wait releases it.
            assert calls == [1]
            assert worker._progress_generation == 0
        assert worker.take_result() == PreparedRuntimeBridgeResult(1, 0, 1, 0)
        assert staged._deferred_item is item
    finally:
        worker.stop()


@pytest.mark.parametrize("phase", ["admission", "post_upload"])
def test_real_errors_abort_reservation_but_are_not_deferred(monkeypatch, phase):
    transfer = DeviceTransferOwner("cpu")
    staged, ready = bridge(transfer)
    item = controlled_item(19, SemanticAction.COMMIT)

    def fail(*args, **kwargs):
        raise RuntimeError("device preparation failed")

    with monkeypatch.context() as patch:
        patch.setattr(transfer if phase == "admission" else transfer.transfer_stream,
                      "upload_packed" if phase == "admission" else "prepare", fail)
        with pytest.raises(RuntimeError, match="device preparation failed"):
            staged.submit_retained_commit(item)
    assert transfer.stats.h2d_ops == int(phase == "post_upload")
    assert staged.submit_retained_commit(item).value == "enqueued"
    assert ready.claim_consumer("test").poll(ReadySourceLane.FRESH).source_identity == item.packet.source


def test_permanent_pinned_byte_limit_raises_without_h2d_or_residency():
    transfer = DeviceTransferOwner("cpu", max_pinned_bytes=1)
    staged, ready = bridge(transfer)
    item = controlled_item(20, SemanticAction.COMMIT)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="pinned byte limit exceeded"):
            staged.submit_retained_commit(item)
        assert transfer.stats.h2d_ops == 0
    assert not ready.claim_scheduler("test").has_source(item.packet.source)


def test_busy_pool_admission_defers_without_h2d_then_retries(monkeypatch):
    transfer = DeviceTransferOwner("cpu", max_inflight=1)
    staged, ready = bridge(transfer)
    item = controlled_item(21, SemanticAction.COMMIT)
    acquire = transfer._packed_pool.acquire
    with monkeypatch.context() as patch:
        patch.setattr(transfer._packed_pool, "acquire", lambda size: acquire(size, blocking=False))
        with transfer._packed_pool._lock:
            assert staged.submit_retained_commit(item) is PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED
    assert transfer.stats.h2d_ops == 0
    assert staged.submit_retained_commit(item).value == "enqueued"
    assert ready.claim_consumer("test").poll(ReadySourceLane.FRESH).source_identity == item.packet.source


@pytest.mark.parametrize(("submitted", "deferred", "progressed"), [
    (0, 0, False), (1, 1, False), (1, 0, True), (2, 1, True),
])
def test_worker_counts_completed_work_not_only_attempts(submitted, deferred, progressed):
    result = PreparedRuntimeBridgeResult(submitted, submitted - deferred, deferred, 0)
    worker = PreparedSourcePrepWorker(SimpleNamespace(drain=lambda: result))
    assert worker.pump_once() is progressed
    assert worker._progress_generation == int(progressed)
    assert worker.take_result() == result
