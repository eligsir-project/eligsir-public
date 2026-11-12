"""``ExternalSLAMWorker`` -- presents an already-external SLAM backend
(e.g. ``ros_slam``, ORB-SLAM3's dedicated ROS2 process) as a
:class:`~.worker.SLAMWorker` (plan section 9.4).

Unlike :class:`~.process_worker.ProcessSLAMWorker`, this does NOT spawn a
new OS process -- the real SLAM computation for these backends already
runs entirely outside the calling Python process (a separate ROS2 node,
in ORB-SLAM3's case a distinct container: ``eligsir-orbslam3``). What
this class wraps is only the local read of the most recently received
pose (:meth:`SLAMBackend.track`, for e.g.
:class:`eligsir.slam.adapters.ros_slam.ROSSLAMBackend`, is already a
cheap, non-blocking cache lookup updated by a background ROS subscriber
callback thread -- see that class's docstring).

This class still runs that read on its own background thread rather than
inline in :meth:`submit`, for two reasons:

1. Interface uniformity with :class:`~.process_worker.ProcessSLAMWorker`
   -- every :class:`~.worker.SLAMWorker` caller gets the identical
   non-blocking submit/poll shape regardless of which kind of backend is
   underneath, so a caller never needs backend-specific knowledge to stay
   correct (deliverable 1: "No production SLAM backend executes
   synchronously in the optimization thread" -- true here by construction,
   not by the wrapped backend happening to be fast today).
2. Defense in depth: if a future change to an "already cheap" backend's
   ``track()`` accidentally introduces blocking work (e.g. a retry loop,
   a lock wait), that regression stays contained to this worker's own
   thread instead of silently reintroducing a stall on the optimization
   thread.

Built on the same thread + :class:`~eligsir.runtime.queues
.bounded_queue.BoundedQueue` shape as ``eligsir.runtime.workers
.frame_preparation_worker.FramePreparationWorker`` (W3-03), not a new
pattern.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from eligsir.mapping.candidates.contracts import RawFramePacket, TrackedFramePacket
from eligsir.runtime.queues.bounded_queue import BoundedQueue, QueuePollState
from .health import WorkerHealth, WorkerState
from .process_worker import _raw_packet_to_rgbdframe, _tracking_result_to_tracked_packet
from .worker import SubmitOutcome, SubmitResult


class ExternalSLAMWorker:
    """Wraps an already-constructed, already-external
    :class:`~eligsir.contracts.slam_backend.SLAMBackend` instance
    (e.g. a live ``ROSSLAMBackend``) as a :class:`~.worker.SLAMWorker`.

    Args:
        backend: An already-constructed backend whose ``track()`` is
            expected to be cheap/non-blocking (the real computation runs
            externally). Passed as an instance, not reconstructed inside
            a subprocess, since it typically owns live ROS
            subscriptions/handles that must live in this process.
        capacity: Bounded input-queue capacity (frames submitted but not
            yet handed to the wrapped backend's ``track()``).
        poll_interval_s: Sleep between empty-queue polls on the worker
            thread.
    """

    def __init__(
        self,
        backend: Any,
        *,
        capacity: int = 8,
        poll_interval_s: float = 0.002,
        close_timeout_s: float = 2.0,
    ) -> None:
        self._backend = backend
        self._backend_name = getattr(backend, "name", backend.__class__.__name__)
        self._input_queue: BoundedQueue[tuple[int, RawFramePacket]] = BoundedQueue(capacity=capacity)
        self._output_queue: BoundedQueue[TrackedFramePacket] = BoundedQueue(capacity=capacity)
        self._poll_interval_s = poll_interval_s
        self._close_timeout_s = close_timeout_s

        self._next_seq = 0
        self._completed_count = 0
        self._consecutive_failures = 0
        self._last_error: Optional[str] = None
        self._last_result_at: Optional[float] = None
        self._state = WorkerState.STARTING
        self._initialized = False

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"external-slam-worker-{self._backend_name}", daemon=True
        )
        self._thread.start()
        self._state = WorkerState.RUNNING

    # -- SLAMWorker protocol -------------------------------------------------

    def submit(self, frame: RawFramePacket) -> SubmitResult:
        """Submit a raw frame for SLAM tracking.

        Args:
            frame: The raw frame data to track.

        Returns:
            A SubmitResult indicating whether the frame was accepted or rejected.
        """
        now_ns = time.monotonic_ns()
        if self._state == WorkerState.CLOSED:
            return SubmitResult(SubmitOutcome.REJECTED_WORKER_CLOSED, now_ns)
        if self._state == WorkerState.FAILED:
            return SubmitResult(SubmitOutcome.REJECTED_WORKER_FAILED, now_ns)
        seq = self._next_seq
        self._next_seq += 1
        result = self._input_queue.try_offer((seq, frame))
        from eligsir.runtime.queues.bounded_queue import QueueOfferResult

        if result is QueueOfferResult.ENQUEUED:
            return SubmitResult(SubmitOutcome.ACCEPTED, now_ns)
        if result is QueueOfferResult.DROPPED_QUEUE_FULL:
            return SubmitResult(SubmitOutcome.REJECTED_QUEUE_FULL, now_ns)
        return SubmitResult(SubmitOutcome.REJECTED_WORKER_CLOSED, now_ns)

    def try_get(self) -> Optional[TrackedFramePacket]:
        """Retrieve a tracked result if one is ready, without blocking.

        Returns:
            A TrackedFramePacket if tracking is complete, or None if not ready.
        """
        polled = self._output_queue.try_poll()
        if polled.state is QueuePollState.ITEM:
            return polled.item
        return None

    def health(self) -> WorkerHealth:
        """Get current worker health and diagnostics.

        Returns:
            A WorkerHealth object containing state, pending work, and error info.
        """
        last_age_ms = None
        if self._last_result_at is not None:
            last_age_ms = (time.monotonic() - self._last_result_at) * 1000.0
        pending = self._input_queue.metrics().depth + self._output_queue.metrics().depth
        return WorkerHealth(
            state=self._state,
            backend_name=self._backend_name,
            pending_submitted=pending,
            completed_count=self._completed_count,
            consecutive_failures=self._consecutive_failures,
            last_error=self._last_error,
            last_result_age_ms=last_age_ms,
        )

    def close(self) -> None:
        """Shut down the worker and its background thread."""
        if self._state == WorkerState.CLOSED:
            return
        self._input_queue.close()
        self._stop.set()
        self._thread.join(timeout=self._close_timeout_s)
        self._state = WorkerState.CLOSED

    # -- internal -------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            polled = self._input_queue.try_poll()
            if polled.state is QueuePollState.EMPTY:
                self._stop.wait(self._poll_interval_s)
                continue
            if polled.state in (QueuePollState.CLOSED, QueuePollState.FAILED):
                return
            seq, raw = polled.item
            try:
                frame = _raw_packet_to_rgbdframe(raw)
                if not self._initialized:
                    result = self._backend.initialize(frame)
                    self._initialized = True
                else:
                    result = self._backend.track(frame)
                tracked = _tracking_result_to_tracked_packet(raw, result)
                self._output_queue.try_offer(tracked)
                self._completed_count += 1
                self._consecutive_failures = 0
                self._last_result_at = time.monotonic()
                if self._state == WorkerState.DEGRADED:
                    self._state = WorkerState.RUNNING
            except Exception as exc:  # a bad frame must not kill the worker
                self._consecutive_failures += 1
                self._last_error = repr(exc)
                if self._state != WorkerState.FAILED:
                    self._state = WorkerState.DEGRADED
                continue
