"""CPU characterization of the canonical bootstrap-to-host-wait wiring.

No source decoding or GPU bootstrap is simulated. Empty ingress produces real
bootstrap-pending outcomes; only the final optimizer result and OS wait are seams.
Removing the canonical waiter binding or retryability must fail the assertions.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from eligsir.runtime import causal_replay
from eligsir.runtime.application import ComposedMappingApplication
from eligsir.runtime.causal_entry import causal_mapping_application_options
from eligsir.runtime.causal_prepared_ready import (
    CausalPreparedReadyB4Path,
    PreparedBootstrapOutcome,
)
from eligsir.runtime.prepared_ingress import PreparedSourceIngressHandoff
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.ready_source import ReadySourceOwner


@pytest.mark.parametrize("fault", [None, "missing_waiter", "unarmed_retry"])
def test_canonical_bootstrap_wait_reaches_real_worker_condition(monkeypatch, fault):
    now = [0.0]
    condition_timeouts = []
    worker_waits = []
    source_timeouts = []

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    producer = PreparedSourceProducer(
        owner, transfer_owner=SimpleNamespace(prepare_source=lambda **_: pytest.fail("GPU work"))
    )
    bridge = PreparedRuntimeBridge(
        PreparedSourceIngressHandoff(capacity=4), producer, ready_owner=owner
    )
    worker = PreparedSourcePrepWorker(bridge)
    runtime = SimpleNamespace(initialized=False, record_prepared_disposition=None)
    path = CausalPreparedReadyB4Path(
        runtime=runtime, bridge=bridge, prep_worker=worker,
        adapter_factory=lambda: pytest.fail("optimizer adapter"),
        step_inputs_factory=lambda: pytest.fail("optimizer inputs"),
    )
    binding = SimpleNamespace(
        mapper_sink=None, optimizer_tick=lambda: pytest.fail("legacy optimizer"),
        materialization_policy=None, materialization_backend=None,
        flush_hook=None, close_hook=None, observation_owner=None,
        control_snapshot_provider=None, disposition_sink=None,
        source_consumed_observer=None,
    )
    options = causal_mapping_application_options(
        SimpleNamespace(binding=binding, prepared_ready_path=path, runtime=runtime),
        args=SimpleNamespace(), ingress_telemetry=None,
    )
    if fault == "missing_waiter":
        options = replace(options, prepared_progress_waiter=None)
    if fault == "unarmed_retry":
        monkeypatch.setattr(PreparedBootstrapOutcome, "retryable", property(lambda _: False))

    ticks = []
    real_tick = options.prepared_tick

    def tick():
        if len(ticks) == 3:
            return SimpleNamespace(executed=True, retryable=False, optimizer_steps=1,
                                   supervision_views=4, completed_supervision_sources=())
        result = real_tick()
        ticks.append(result)
        return result

    application = ComposedMappingApplication(replace(options, prepared_tick=tick))

    # Keep Condition.wait_for's real predicate/timeout loop; replace only the OS
    # blocking primitive and its monotonic clock, avoiding CI wall-time bounds.
    def condition_wait(timeout=None):
        assert worker._progress._is_owned()
        condition_timeouts.append(timeout)
        now[0] += timeout
        return False

    real_wait = worker.wait_for_progress

    def measured_wait(*, timeout_s):
        started = now[0]
        progressed = real_wait(timeout_s=timeout_s)
        worker_waits.append((progressed, now[0] - started))
        return progressed

    monkeypatch.setattr(worker._progress, "wait", condition_wait)
    monkeypatch.setattr(worker, "wait_for_progress", measured_wait)
    monkeypatch.setattr("threading._time", lambda: now[0])
    monkeypatch.setattr(causal_replay.time, "perf_counter", lambda: now[0])

    class EmptySource:
        complete = True
        queue_depth = dropped = dropped_at_arrival = dropped_at_enqueue = offered = 0
        capacity = high_watermark = 1
        offered_fps = None

        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def wait_for_initial_packet_or_terminal(self): pass
        def close(self): pass
        def source_ingress_report_block(self): return {}

        def take(self, *, timeout_s):
            source_timeouts.append(timeout_s)
            now[0] += timeout_s
            return None

    monkeypatch.setattr(causal_replay, "_PacedPacketProducer", EmptySource)
    summary = causal_replay.PacedCausalReplayDriver(application, None, source_hz=30).run([])

    def assert_wiring():
        assert len(worker_waits) == 3, "bootstrap did not reach the worker waiter"
        # TransferEvent completion is poll-only.  Until it has a real completion
        # notification, retryable prepared work must recheck after 1 ms.
        assert condition_timeouts == pytest.approx([0.001] * 3)
        assert all(not progressed for progressed, _ in worker_waits)
        assert [elapsed for _, elapsed in worker_waits] == pytest.approx([0.001] * 3)
        assert summary.outer_unaccounted_s == pytest.approx(0.001 * 3)

    assert all(isinstance(result, PreparedBootstrapOutcome) for result in ticks)
    assert summary.ticks == 4 and summary.optimizer_steps == 1
    assert source_timeouts == pytest.approx(
        [0.001] + [0.001 if fault == "unarmed_retry" else 0] * 3  # source take stays a 1 ms poll
    )
    if fault is None:
        assert_wiring()
        application.wait_for_prepared_progress(timeout_s=0.001)
        assert len(worker_waits) == 3, "successful execution must disarm the waiter"
    else:
        with pytest.raises(AssertionError, match="did not reach the worker waiter"):
            assert_wiring()
