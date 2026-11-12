"""Tests for the asynchronous prepared-source device worker."""

from __future__ import annotations

import threading

from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult


def test_pump_once_prepares_without_gpu_tick_and_publishes_result():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    class Bridge:
        def __init__(self) -> None:
            self.calls = 0

        def drain(self) -> PreparedRuntimeBridgeResult:
            self.calls += 1
            return PreparedRuntimeBridgeResult(1, 1, 0, 0)

    bridge = Bridge()
    worker = PreparedSourcePrepWorker(bridge)
    assert worker.pump_once() is True
    assert bridge.calls == 1
    assert worker.take_result() == PreparedRuntimeBridgeResult(1, 1, 0, 0)
    assert worker.take_result() == PreparedRuntimeBridgeResult(0, 0, 0, 0)


def test_worker_accumulates_results_until_consumer_takes_them():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(1, 1, 0, 0)

    worker = PreparedSourcePrepWorker(Bridge())
    assert worker.pump_once() is True
    assert worker.pump_once() is True
    assert worker.take_result() == PreparedRuntimeBridgeResult(2, 2, 0, 0)
    assert worker.take_result() == PreparedRuntimeBridgeResult(0, 0, 0, 0)


def test_started_worker_polls_ingress_and_stops_without_tick_participation():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    first_drain = threading.Event()
    calls = 0

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_drain.set()
                return PreparedRuntimeBridgeResult(1, 1, 0, 0)
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    worker = PreparedSourcePrepWorker(Bridge(), poll_interval_s=0.001).start()
    assert first_drain.wait(1.0)
    worker.stop()
    assert worker.running is False
    assert worker.take_result() == PreparedRuntimeBridgeResult(1, 1, 0, 0)


def test_ready_tick_uses_worker_result_without_draining_bridge(monkeypatch):
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )

    class WorkerBridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    worker = PreparedSourcePrepWorker(WorkerBridge())
    assert worker.pump_once() is False

    class FakeBridge:
        ready_owner = object()

        def drain(self):
            raise AssertionError("GPU tick must not prepare accepted sources")

    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", FakeBridge)
    adapter = ReadyB4StreamingOptimizationAdapter(
        FakeBridge(),
        runtime=type("Runtime", (), {"tick": lambda self, **_kwargs: None})(),
        transfer_owner=object(),
        ingress_result_provider=worker.take_result,
        warmup_complete=True,
    )
    outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot=None, work_plan=None))
    assert outcome.ingress == PreparedRuntimeBridgeResult(0, 0, 0, 0)


def test_prep_worker_uses_promotion_progress_without_result_side_channel():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    class Promotions:
        calls = 0

        def pump_once(self) -> bool:
            self.calls += 1
            return True

    promotions = Promotions()
    worker = PreparedSourcePrepWorker(Bridge(), promotion_owner=promotions)
    assert worker.pump_once() is True
    assert promotions.calls == 1
    assert not hasattr(worker, "take_promotion_result")


def test_started_worker_does_not_sleep_while_promotions_make_progress():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    second = threading.Event()

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    class Promotions:
        calls = 0

        def pump_once(self):
            self.calls += 1
            if self.calls == 2:
                second.set()
            return True

    worker = PreparedSourcePrepWorker(
        Bridge(), poll_interval_s=1.0, promotion_owner=Promotions()
    ).start()
    try:
        assert second.wait(0.2)
    finally:
        worker.stop()


def test_worker_yields_when_promotion_is_transiently_blocked():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    class Promotions:
        def pump_once(self):
            return False

    # Track the wait_for calls on the condition
    wait_for_calls = []

    def mock_wait_for(original_wait_for):
        def wrapper(predicate, timeout=None):
            wait_for_calls.append(timeout)
            # Return immediately with a stop signal to exit the loop
            raise StopIteration()
        return wrapper

    worker = PreparedSourcePrepWorker(
        Bridge(), poll_interval_s=0.5, promotion_owner=Promotions()
    )

    # Wrap the condition's wait_for method to track calls
    original_condition_wait_for = worker._progress.wait_for
    worker._progress.wait_for = mock_wait_for(original_condition_wait_for)

    try:
        worker._run()
    except StopIteration:
        pass  # Expected from our mock

    # The worker should call wait_for with the poll_interval_s timeout
    assert wait_for_calls == [0.5], f"Expected [0.5] but got {wait_for_calls}"


def test_lost_wakeup_race_commit_injected_in_pump_wait_window():
    """Test that a commit injected between pump_once() returning False and wait_for() entry is detected.

    This is the critical race condition: the commit lands after pump returns no progress
    but before the worker enters wait_for(). Without a generation token, the notification
    would be lost. With the token pattern, the predicate detects the generation change
    even if the notification happens before wait_for() is called.
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
    import time

    commit_during_pump = threading.Event()
    pump_count = [0]

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            pump_count[0] += 1
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    class Promotions:
        def pump_once(self):
            return False

    worker = PreparedSourcePrepWorker(
        Bridge(), poll_interval_s=0.5, promotion_owner=Promotions()
    )

    # Monkeypatch pump_once to inject a commit right before returning False.
    # This simulates the race: commit lands after pump returns no progress but
    # before the worker acquires the lock for wait_for().
    original_pump_once = worker.pump_once

    def pump_once_with_inject():
        # Call original pump_once
        result = original_pump_once()
        # If it returned False (no progress), inject a commit signal
        if not result:
            # This simulates a commit happening after pump_once returns.
            # We do this by directly calling the signal (which would normally
            # be called by handoff._commit()).
            worker._signal_progress()
        return result

    worker.pump_once = pump_once_with_inject

    # Run a single iteration and measure the behavior.
    # With proper generation token handling, the wait_for predicate will see
    # that generation changed and return immediately without waiting.

    # Track whether wait_for is called with the full timeout or returns early
    wait_for_returned_early = threading.Event()

    original_wait_for = worker._progress.wait_for

    def instrumented_wait_for(predicate, timeout=None):
        start = time.time()
        result = original_wait_for(predicate, timeout=timeout)
        elapsed = time.time() - start
        # If we returned in less than 10% of the timeout, we returned early
        # (not due to timeout). This is the expected behavior.
        if elapsed < (timeout * 0.1 if timeout else 0.01):
            wait_for_returned_early.set()
        return result

    worker._progress.wait_for = instrumented_wait_for

    # Start the worker
    worker.start()
    try:
        # Let it do a few iterations
        time.sleep(0.1)
    finally:
        worker.stop()

    # With the generation token pattern, the injected signal should be detected
    # immediately and cause early returns from wait_for, not waiting for the
    # 500ms timeout.
    assert wait_for_returned_early.is_set(), (
        "Lost-wakeup race: commit signal was not detected by wait_for predicate"
    )


def test_blocked_promotion_work_no_busy_spin():
    """Regression: blocked COLD promotion work should not busy-spin the worker.

    When a promotion request is queued but cannot progress (blocked), the worker
    should fall back to the bounded timeout (50ms), not busy-spin checking the
    level-triggered pending work state. This tests the EDGE-triggered promotion
    wakeup: the signal is sent at enqueue time, and then blocked work falls back
    to the bounded timeout.
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
    import time

    pump_call_times = []

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            pump_call_times.append(time.time())
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    blocked_iteration = threading.Event()

    class PromotionOwner:
        def __init__(self):
            self._queue_full = True
            self._progress_callback = None

        def set_progress_callback(self, callback):
            """Public API for registering progress callback."""
            self._progress_callback = callback

        def pump_once(self):
            # First call: return False (blocked/no progress)
            if self._queue_full:
                # Signal that we're blocked
                blocked_iteration.set()
                return False
            # Later (when unblocked): return True (simulating progress)
            return True

    promotion_owner = PromotionOwner()
    worker = PreparedSourcePrepWorker(
        Bridge(), poll_interval_s=0.05, promotion_owner=promotion_owner
    ).start()

    # When the worker is started with a promotion owner, it sets the callback
    # which is now available on the promotion owner
    assert promotion_owner._progress_callback is not None, "Progress callback not set"

    try:
        # Wait for the worker to hit the blocked state
        assert blocked_iteration.wait(1.0), "Worker did not reach blocked state"

        # Clear the timestamp list to measure from here
        pump_call_times.clear()

        # Run for 300ms with a 50ms timeout
        # Expected: ceil(300 / 50) + 1 = 7 pump calls max
        test_duration = 0.3
        start_time = time.time()
        time.sleep(test_duration)
        end_time = time.time()

        # Filter to calls within the window
        actual_duration = end_time - start_time
        window_calls = len(pump_call_times)
        max_expected_calls = int(actual_duration / 0.05) + 2  # ~8 max

        # The worker should NOT busy-spin (which would be hundreds of calls)
        # It should only pump at the timeout boundary
        assert window_calls <= max_expected_calls, (
            f"Blocked promotion caused busy-spin: {window_calls} pump calls in {test_duration}s; "
            f"max expected {max_expected_calls} (one per 50ms timeout)"
        )

        # Now unblock the promotion and verify it eventually processes
        promotion_owner._queue_full = False
        # Signal the worker that work is unblocked (normally done by schedule_shortlist)
        if promotion_owner._progress_callback is not None:
            promotion_owner._progress_callback()

        time.sleep(0.1)

        # Worker should have pumped at least once more (to detect unblocked work)
        # and returned True from the promotion pump_once
        assert len(pump_call_times) > window_calls, (
            "Worker did not process unblocked promotion"
        )
    finally:
        worker.stop()


def _wait_for_failure(worker, expected):
    for _ in range(1000):
        try:
            worker.check_health()
        except BaseException as raised:  # noqa: BLE001 - identity assertion below
            assert raised is expected
            return
        threading.Event().wait(0.001)
    raise AssertionError("worker failure was not surfaced")


def test_worker_failure_is_raised_by_consumer_instead_of_looking_starved():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    failure = ValueError("invalid prepared depth scale")

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            raise failure

    worker = PreparedSourcePrepWorker(Bridge(), poll_interval_s=0.001).start()
    try:
        _wait_for_failure(worker, failure)
        try:
            worker.take_result()
        except ValueError as raised:
            assert raised is failure
        else:
            raise AssertionError("failed preparation was hidden as empty ingress")
    finally:
        worker.stop()


def test_normal_worker_shutdown_does_not_report_failure():
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    worker = PreparedSourcePrepWorker(Bridge(), poll_interval_s=0.001).start()
    worker.stop()
    worker.check_health()
    assert worker.take_result() == PreparedRuntimeBridgeResult(0, 0, 0, 0)


def test_prepared_runtime_consumer_surfaces_worker_failure():
    from eligsir.runtime.causal_prepared_ready import CausalPreparedReadyB4Path
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    failure = RuntimeError("GPU preparation failed")

    class Bridge:
        def drain(self) -> PreparedRuntimeBridgeResult:
            raise failure

    worker = PreparedSourcePrepWorker(Bridge(), poll_interval_s=0.001).start()
    try:
        _wait_for_failure(worker, failure)
        path = object.__new__(CausalPreparedReadyB4Path)
        path._prep_worker = worker
        try:
            path._take_ingress_result()
        except RuntimeError as raised:
            assert raised is failure
        else:
            raise AssertionError("runtime converted preparation failure to starvation")
    finally:
        worker.stop()
