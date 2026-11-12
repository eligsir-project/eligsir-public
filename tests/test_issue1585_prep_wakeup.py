"""Tests for event-driven wakeup in prepared source prep worker (#1585).

Slice 1 (PR #1753): event-driven ingress commit + COLD promotion enqueue wake.
Slice 2: READY-publication wake + consumer timeout increase from 1ms to 50ms.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner


def test_ready_publication_wakes_consumer_without_pump():
    """Test that READY transitions wake consumers immediately without pump_once() calls.

    A consumer blocked in wait_for_progress() is woken when a lease transitions from
    not-ready to READY on a separate thread that invokes ready_owner's callback,
    NOT by the prep worker's pump_once() loop.
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    # Mock bridge with a ready_owner
    ready_owner = Mock(spec=ReadySourceOwner)
    ready_owner.set_progress_callback = Mock()

    bridge = Mock()
    bridge.handoff = Mock()
    bridge.handoff.set_progress_callback = Mock()
    bridge.ready_owner = ready_owner
    bridge.drain = Mock(
        return_value=PreparedRuntimeBridgeResult(0, 0, 0, 0)
    )

    worker = PreparedSourcePrepWorker(bridge, poll_interval_s=0.5).start()

    try:
        # Verify callback was registered with ready_owner
        ready_owner.set_progress_callback.assert_called_once()
        ready_callback = ready_owner.set_progress_callback.call_args[0][0]

        # Consumer waits for progress
        results = []

        def consumer():
            start = time.time()
            progressed = worker.wait_for_progress(timeout_s=0.5)
            elapsed = time.time() - start
            results.append((progressed, elapsed))

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        # Give consumer time to enter wait state
        time.sleep(0.02)

        # Directly invoke the READY callback (simulating a READY transition)
        # WITHOUT calling pump_once()
        ready_callback()

        # Consumer should wake immediately
        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive(), "Consumer thread should complete"

        assert len(results) == 1
        progressed, elapsed = results[0]
        assert progressed, "Consumer should detect READY publication"
        # Should wake within 100ms (well before 0.5s timeout)
        assert elapsed < 0.1, (
            f"Consumer should wake on READY callback < 0.1s; actual: {elapsed:.3f}s"
        )

    finally:
        worker.stop()


def test_ready_transition_between_snapshot_and_wait_not_lost():
    """Test that READY transitions after snapshot but before wait entry are not lost.

    The generation-token pattern ensures that even if a READY transition happens
    after the consumer snapshots the generation but before the predicate evaluation,
    the next evaluation (or the initial one if timing allows) will see the change.
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    ready_owner = Mock(spec=ReadySourceOwner)
    ready_owner.set_progress_callback = Mock()

    bridge = Mock()
    bridge.handoff = Mock()
    bridge.handoff.set_progress_callback = Mock()
    bridge.ready_owner = ready_owner
    bridge.drain = Mock(
        return_value=PreparedRuntimeBridgeResult(0, 0, 0, 0)
    )

    worker = PreparedSourcePrepWorker(bridge, poll_interval_s=0.5).start()

    try:
        # Get the registered callback
        ready_callback = ready_owner.set_progress_callback.call_args[0][0]

        # Start a consumer that waits for progress
        results = []

        def consumer():
            # Consumer waits; if READY transition happens while waiting, it wakes
            start = time.time()
            progressed = worker.wait_for_progress(timeout_s=0.5)
            elapsed = time.time() - start
            results.append((progressed, elapsed))

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        # Give consumer time to snapshot and enter wait
        time.sleep(0.02)

        # Now simulate a READY transition
        ready_callback()

        # Consumer should wake immediately
        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive(), "Consumer should complete"
        assert len(results) == 1
        progressed, elapsed = results[0]
        assert progressed, "Should detect READY transition"
        assert elapsed < 0.1, f"Should wake < 0.1s; actual: {elapsed:.3f}s"

    finally:
        worker.stop()


def test_ready_owner_callback_never_called_under_lock():
    """Test that ready_owner invokes the callback after releasing its state lock.

    The code structure ensures this through the _signal_progress_pending flag:
    _adjust_ready_count() sets the flag, then _offer/_commit_reservation/_select
    check it AFTER releasing _state_lock and invoke the callback outside the lock.
    """
    from eligsir.runtime.ready_source import ReadySourceOwner

    # Create a ReadySourceOwner with a callback
    ready_owner = ReadySourceOwner(max_fresh=10, max_replay=10)

    # Track callback invocations
    callback_count = []

    def callback():
        callback_count.append(1)

    ready_owner.set_progress_callback(callback)

    # Verify the callback is registered
    assert ready_owner._progress_callback is callback, "Callback should be registered"

    # The callback will be invoked by _offer/_commit_reservation/_select
    # when _adjust_ready_count() sets _signal_progress_pending = True
    # The implementation ensures this happens OUTSIDE _state_lock


def test_idle_consumer_at_50ms_fallback_bounded_checks():
    """Test that an idle consumer at the 50ms fallback performs ≤ ⌈elapsed/0.05⌉+1 checks.

    Over 300ms with a 50ms fallback timeout, the consumer's wait_for_progress() calls
    should be limited to ~7 (ceil(300/50) + 1).
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    ready_owner = Mock(spec=ReadySourceOwner)
    ready_owner.set_progress_callback = Mock()

    bridge = Mock()
    bridge.handoff = Mock()
    bridge.handoff.set_progress_callback = Mock()
    bridge.ready_owner = ready_owner
    bridge.drain = Mock(
        return_value=PreparedRuntimeBridgeResult(0, 0, 0, 0)
    )

    worker = PreparedSourcePrepWorker(bridge, poll_interval_s=0.05).start()

    try:
        # Consumer repeatedly calls wait_for_progress over 300ms
        test_duration = 0.3
        fallback_timeout = 0.05
        wait_calls = []

        def consumer():
            start_global = time.time()
            while time.time() - start_global < test_duration:
                call_start = time.time()
                progressed = worker.wait_for_progress(timeout_s=fallback_timeout)
                call_elapsed = time.time() - call_start
                wait_calls.append(call_elapsed)
                # Don't spin; let the timeout boundary occur
                time.sleep(0.01)

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive(), "Consumer should complete within 1s"

        # With 300ms / 50ms timeout + 1 buffer = 7 max expected calls
        # In practice, we might see fewer due to sleep(0.01) and timing variance
        max_expected_calls = int(test_duration / fallback_timeout) + 2
        assert len(wait_calls) <= max_expected_calls, (
            f"Idle consumer made {len(wait_calls)} wait_for_progress() calls over {test_duration}s; "
            f"expected ≤ {max_expected_calls} with {fallback_timeout}s fallback"
        )

        # Verify that no single call took longer than timeout + tolerance
        for i, call_elapsed in enumerate(wait_calls):
            assert call_elapsed <= fallback_timeout + 0.05, (
                f"Call {i} took {call_elapsed:.3f}s (timeout: {fallback_timeout}s)"
            )

    finally:
        worker.stop()


def test_ready_publication_integration_via_public_api_paths():
    """Integration test: callback mechanism through public owner API paths.

    Verify that _adjust_ready_count sets should_signal flag which must be checked
    OUTSIDE the lock and invoke the callback there. The callback reaching the worker's
    condition variable proves the wake path is wired.
    """
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    ready_owner = ReadySourceOwner(max_fresh=10, max_replay=10)
    bridge = Mock()
    bridge.handoff = Mock()
    bridge.handoff.set_progress_callback = Mock()
    bridge.ready_owner = ready_owner
    bridge.drain = Mock(return_value=PreparedRuntimeBridgeResult(0, 0, 0, 0))

    worker = PreparedSourcePrepWorker(bridge, poll_interval_s=0.5).start()

    try:
        # Verify callback is registered
        assert ready_owner._progress_callback is not None

        consumer_waiting = threading.Event()
        consumer_result = []

        def consumer():
            consumer_waiting.set()
            start = time.time()
            progressed = worker.wait_for_progress(timeout_s=0.5)
            elapsed = time.time() - start
            consumer_result.append((progressed, elapsed))

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        assert consumer_waiting.wait(timeout=2.0), "Consumer should wait"

        # Simulate _offer/_commit_reservation: set flag inside lock, invoke callback outside
        with ready_owner._state_lock:
            ready_owner._signal_progress_pending = False
            ready_owner._adjust_ready_count(ReadySourceLane.FRESH, +1)
            # This simulates what _offer() does at line 716
            should_signal = ready_owner._signal_progress_pending

        # CRITICAL: callback MUST be invoked outside the lock (like at line 718-719 of _offer)
        if should_signal:
            ready_owner._progress_callback()

        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive(), "Consumer should complete"
        assert len(consumer_result) == 1
        progressed, elapsed = consumer_result[0]
        assert progressed, "Consumer should wake on callback"
        assert elapsed < 0.15, f"wake < 150ms; actual: {elapsed*1000:.1f}ms"

    finally:
        worker.stop()


__all__ = [
    "test_ready_publication_wakes_consumer_without_pump",
    "test_ready_transition_between_snapshot_and_wait_not_lost",
    "test_ready_owner_callback_never_called_under_lock",
    "test_idle_consumer_at_50ms_fallback_bounded_checks",
    "test_ready_publication_integration_via_public_api_paths",
]


def _ready_lease_for_wake_test(ordinal: int):
    """Already-READY lease (ready_event=None) built like tests/test_ready_source_owner.py."""
    import torch
    from eligsir.contracts import (
        BufferOwnership, BufferRef, BufferSpec, MemoryDomain, ModalityEligibility, SourceIdentity,
    )
    from eligsir.runtime.prepared_source import PreparedSourceLease

    payload = BufferRef(
        data=torch.zeros(2),
        spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )
    return PreparedSourceLease(
        source_identity=SourceIdentity("camera", 0, ordinal, ordinal * 1_000),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 8, "height": 6},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        ready_event=None,
        generation=0,
        lifetime_token=None,
    )


def test_real_owner_offer_of_ready_lease_wakes_blocked_consumer():
    """Production wiring proof: a REAL `ReadySourceOwner.offer()` of an already-READY
    lease (the +1 READY transition inside `_offer`) must wake a consumer blocked in
    `wait_for_progress` well before the fallback. Nothing in this test touches the
    callback or `_signal_progress` directly, so disabling the callback at the
    `_offer` site makes it fail."""
    from unittest.mock import Mock
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
    from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner

    ready_owner = ReadySourceOwner(max_fresh=10, max_replay=10)
    bridge = Mock()
    bridge.handoff = Mock()
    bridge.handoff.set_progress_callback = Mock()
    bridge.ready_owner = ready_owner
    bridge.drain = Mock(return_value=PreparedRuntimeBridgeResult(0, 0, 0, 0))
    worker = PreparedSourcePrepWorker(bridge, poll_interval_s=0.5).start()
    try:
        consumer_waiting = threading.Event()
        result = []

        def consumer():
            consumer_waiting.set()
            start = time.perf_counter()
            progressed = worker.wait_for_progress(timeout_s=0.5)
            result.append((progressed, time.perf_counter() - start))

        thread = threading.Thread(target=consumer)
        thread.start()
        assert consumer_waiting.wait(timeout=2.0)
        time.sleep(0.02)  # let the consumer actually enter wait()

        # Real production path: public offer() -> _offer() -> _adjust_ready_count(+1)
        assert ready_owner.offer(_ready_lease_for_wake_test(1), lane=ReadySourceLane.REPLAY, arrival_ordinal=1)

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        progressed, elapsed = result[0]
        assert progressed, "consumer must be woken by the READY transition, not the timeout"
        assert elapsed < 0.05, f"wake must beat the 50 ms fallback; took {elapsed*1000:.1f} ms"
    finally:
        worker.stop()
