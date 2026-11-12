"""``ProcessSLAMWorker`` -- wraps an in-process :class:`~eligsir
.contracts.slam_backend.SLAMBackend` (e.g. ``open3d_odometry_live``) in a
real OS subprocess so its ``track()`` latency can never block the
optimization thread (plan section 9.4, "Wrap in-process backends in
worker processes").

Why a real process and not just a background thread: this project has
already found (see CLAUDE.md's "High-throughput, low-complexity data
processing belongs in a fast language" note, and the RVizSplat publish-
thread GIL-contention finding it documents) that CPU-bound Python work on
a background thread still measurably stalls the main thread because both
contend for the same GIL. ``open3d_odometry_live``'s Open3D odometry call
is exactly this class of CPU-bound work. A separate OS process has its
own GIL and cannot stall the caller's interpreter no matter how long
``track()`` takes or how badly it hangs.

The backend is re-constructed **inside** the child process (by name, via
the existing ``get_backend()`` registry) rather than pickled across the
process boundary -- most SLAM backend instances hold non-picklable
internal state (Open3D objects, ROS handles) once running, so passing a
``(backend_name, backend_kwargs)`` pair and reconstructing is the only
robust option. Uses the ``"spawn"`` multiprocessing start method
explicitly (never ``"fork"``): forking a process that may already have
initialized a ROCm/HIP or CUDA context is a well-known source of hangs
and crashes.

Known limitation (explicitly not fixed by this PR): each submitted frame
crosses the process boundary via ``multiprocessing.Pipe``, which pickles
the frame's numpy RGB/depth buffers. For large frames at high frame rate
this is real, measurable serialization overhead -- exactly the kind of
"process a lot of data per second with comparatively little logic" work
CLAUDE.md's standing project-wide guidance says belongs in a fast
language rather than Python IPC. Tracked as follow-up, not required for
this PR's non-negotiable deliverables (none of which are about
throughput).
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
)
from eligsir.runtime.queues.bounded_queue import (
    BoundedQueue,
    QueueOfferResult as BoundedQueueOfferResult,
    QueuePollState,
)
from .health import WorkerHealth, WorkerState
from .worker import SubmitOutcome, SubmitResult

# Sentinel sent down the pipe to ask the child process to shut down.
_SHUTDOWN = ("__shutdown__", None)


def _raw_packet_to_rgbdframe(raw: RawFramePacket):
    """Convert a decoded :class:`RawFramePacket` into the canonical
    ``contracts.types.RGBDFrame`` a :class:`SLAMBackend` expects.

    Requires ``rgb_ref.array``/``depth_ref.array`` to already hold
    decoded numpy buffers (the "already decoded in memory" case
    documented on :class:`~eligsir.mapping.candidates.contracts
    .UndecodedBufferRef` -- true for every live/streaming source today; a
    path-only or bytes-only ``RawFramePacket`` is not yet supported by
    this worker and raises ``ValueError`` rather than silently tracking
    garbage).
    """
    from eligsir.contracts.types import CameraIntrinsics as CanonicalIntrinsics
    from eligsir.contracts.types import RGBDFrame as CanonicalRGBDFrame

    if raw.rgb_ref.array is None or raw.depth_ref.array is None:
        raise ValueError(
            "ProcessSLAMWorker requires RawFramePacket.rgb_ref/depth_ref "
            "to carry already-decoded numpy arrays (UndecodedBufferRef.array); "
            f"got rgb={raw.rgb_ref}, depth={raw.depth_ref}"
        )
    rgb = raw.rgb_ref.array
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    depth = raw.depth_ref.array.astype(np.float32)

    src_intr = raw.intrinsics
    intr = CanonicalIntrinsics(
        fx=src_intr.fx, fy=src_intr.fy, cx=src_intr.cx, cy=src_intr.cy,
        width=src_intr.width, height=src_intr.height,
    )
    return CanonicalRGBDFrame(
        index=raw.source_index,
        timestamp=raw.timestamp_ns / 1e9,
        rgb=rgb.astype(np.float32),
        depth=depth,
        intrinsics=intr,
        pose=None,
    )


def _tracking_result_to_tracked_packet(
    raw: RawFramePacket, result: Any
) -> TrackedFramePacket:
    """Build a :class:`TrackedFramePacket` from a backend's
    ``TrackingResult`` (or ``None`` on failure)."""
    if result is not None and getattr(result, "valid", False):
        world_T_camera = np.asarray(result.pose.matrix, dtype=np.float32)
        state = TrackingState.OK
        confidence = float(getattr(result, "confidence", 1.0))
    else:
        world_T_camera = np.eye(4, dtype=np.float32)
        state = TrackingState.LOST
        confidence = 0.0
    return TrackedFramePacket(
        frame=raw,
        world_T_camera=world_T_camera,
        tracking_state=state,
        tracking_confidence=confidence,
        slam_map_id=0,
    )


def _child_main(
    backend_name: str,
    backend_kwargs: Dict[str, Any],
    conn,
    artificial_delay_s: float,
) -> None:
    """Entry point run inside the child process. Never imports torch/CUDA
    at module scope -- only once actually invoked as a subprocess, so the
    parent process's own GPU context is unaffected."""
    from eligsir.tracking.factory import build_tracking_frontend

    # #1075 removed the ``SLAMBackend`` registry. The four-module import block
    # that used to be needed here is gone with it: registration was an import
    # side effect, so a spawned child started with an empty registry and every
    # call site had to repeat the same hand-maintained import list. An explicit
    # factory has no import-order coupling, so name resolution in this child is
    # now identical to the parent's by construction rather than by convention.
    try:
        backend = build_tracking_frontend(backend_name, **backend_kwargs)
    except Exception as exc:  # construction failure -- report and exit
        try:
            conn.send(("init_error", None, repr(exc)))
        finally:
            conn.close()
        return

    initialized = False
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is _SHUTDOWN or (isinstance(msg, tuple) and msg[0] == "__shutdown__"):
            break
        seq, raw = msg
        try:
            if artificial_delay_s > 0:
                time.sleep(artificial_delay_s)
            frame = _raw_packet_to_rgbdframe(raw)
            if not initialized:
                result = backend.initialize(frame)
                initialized = True
            else:
                result = backend.track(frame)
            tracked = _tracking_result_to_tracked_packet(raw, result)
            conn.send(("ok", seq, tracked))
        except Exception as exc:  # a bad frame must not kill the worker
            conn.send(("error", seq, repr(exc)))
    conn.close()


class ProcessSLAMWorker:
    """Runs one in-process :class:`SLAMBackend` inside a dedicated OS
    subprocess and presents it as a :class:`~.worker.SLAMWorker`.

    Args:
        backend_name: Registry key (see ``contracts.slam_backend
            .get_backend``), e.g. ``"open3d_odometry_live"``.
        backend_kwargs: Keyword args forwarded to the backend constructor,
            evaluated inside the child process.
        capacity: Max number of frames in flight (submitted but not yet
            retrieved via :meth:`try_get`) before :meth:`submit` rejects
            with ``REJECTED_QUEUE_FULL``.
        artificial_delay_s: Extra ``time.sleep`` injected before every
            ``track()``/``initialize()`` call inside the child, purely for
            :mod:`scripts.benchmark_slam_isolation` (deliverable 2) --
            never set outside a benchmark/test.
    """

    def __init__(
        self,
        backend_name: str,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        *,
        capacity: int = 8,
        artificial_delay_s: float = 0.0,
        close_timeout_s: float = 2.0,
    ) -> None:
        """Initialize the subprocess SLAM worker.

        Args:
            backend_name: Registry key for the SLAM backend (e.g. "open3d_odometry_live").
            backend_kwargs: Keyword arguments for backend construction.
            capacity: Maximum number of frames in flight before submit() rejects.
            artificial_delay_s: Optional extra sleep before each track() (testing only).
            close_timeout_s: Maximum time to wait for subprocess shutdown.
        """
        self._backend_name = backend_name
        self._backend_kwargs = dict(backend_kwargs or {})
        self._capacity = capacity
        self._artificial_delay_s = artificial_delay_s
        self._close_timeout_s = close_timeout_s

        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._process = self._ctx.Process(
            target=_child_main,
            args=(self._backend_name, self._backend_kwargs, child_conn, self._artificial_delay_s),
            daemon=True,
            name=f"slam-worker-{backend_name}",
        )
        self._process.start()
        # The child holds its own dup of child_conn; close our reference so
        # EOF is detected correctly if the child dies.
        child_conn.close()

        self._next_seq = 0
        self._pending: "dict[int, RawFramePacket]" = {}
        self._completed_count = 0
        self._consecutive_failures = 0
        self._last_error: Optional[str] = None
        self._last_result_at: Optional[float] = None
        self._state = WorkerState.STARTING
        self._closed = False

        # Real frames (decoded RGB+depth) can be several MB. Pipe.send()
        # blocks once the OS pipe buffer is full -- fine on a dedicated
        # feeder thread, NOT fine on the caller's (optimization) thread,
        # which is exactly the stall this whole class exists to prevent.
        # submit() only ever does a non-blocking BoundedQueue.try_offer;
        # this feeder thread owns the one place that may block on
        # send().
        self._send_queue: BoundedQueue[tuple[int, RawFramePacket]] = BoundedQueue(capacity=capacity)
        self._feeder_stop = threading.Event()
        self._feeder_thread = threading.Thread(
            target=self._feeder_run, name=f"slam-worker-feeder-{backend_name}", daemon=True
        )
        self._feeder_thread.start()

    # -- SLAMWorker protocol -------------------------------------------------

    def submit(self, frame: RawFramePacket) -> SubmitResult:
        """Submit a raw frame for SLAM tracking in the subprocess.

        Args:
            frame: The raw frame data to track.

        Returns:
            A SubmitResult indicating whether the frame was accepted or rejected.
        """
        now_ns = time.monotonic_ns()
        if self._closed or self._state == WorkerState.CLOSED:
            return SubmitResult(SubmitOutcome.REJECTED_WORKER_CLOSED, now_ns)
        self._poll_process_liveness()
        if self._state == WorkerState.FAILED:
            return SubmitResult(SubmitOutcome.REJECTED_WORKER_FAILED, now_ns)
        if len(self._pending) >= self._capacity:
            # Capacity bounds total in-flight frames -- submitted but not
            # yet retrieved via try_get() -- not merely the small
            # feeder-queue staging depth (checked separately below, but
            # this check dominates since pending only grows in step with
            # successful sends below).
            return SubmitResult(SubmitOutcome.REJECTED_QUEUE_FULL, now_ns)

        seq = self._next_seq
        offer = self._send_queue.try_offer((seq, frame))
        if offer is not BoundedQueueOfferResult.ENQUEUED:
            return SubmitResult(SubmitOutcome.REJECTED_QUEUE_FULL, now_ns)
        self._next_seq += 1
        self._pending[seq] = frame
        if self._state == WorkerState.STARTING:
            self._state = WorkerState.RUNNING
        return SubmitResult(SubmitOutcome.ACCEPTED, now_ns)

    def _feeder_run(self) -> None:
        """Owns the only blocking ``Connection.send()`` call -- runs on
        its own thread so a full OS pipe buffer (slow/wedged child) stalls
        this thread, never the caller's."""
        while not self._feeder_stop.is_set():
            polled = self._send_queue.try_poll()
            if polled.state is QueuePollState.EMPTY:
                self._feeder_stop.wait(0.002)
                continue
            if polled.state in (QueuePollState.CLOSED, QueuePollState.FAILED):
                return
            seq, frame = polled.item
            try:
                self._parent_conn.send((seq, frame))
            except (BrokenPipeError, OSError) as exc:
                self._state = WorkerState.FAILED
                self._last_error = repr(exc)
                return

    def try_get(self) -> Optional[TrackedFramePacket]:
        """Retrieve a tracked result if one is ready, without blocking.

        Returns:
            A TrackedFramePacket if a tracking result is available, or None if not ready.
        """
        self._poll_process_liveness()
        try:
            if not self._parent_conn.poll(0):
                return None
            msg = self._parent_conn.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            self._state = WorkerState.FAILED
            self._last_error = repr(exc)
            return None

        kind = msg[0]
        if kind == "init_error":
            self._state = WorkerState.FAILED
            self._last_error = msg[2]
            return None
        if kind == "error":
            _seq, error_repr = msg[1], msg[2]
            self._pending.pop(msg[1], None)
            self._consecutive_failures += 1
            self._last_error = error_repr
            if self._state != WorkerState.FAILED:
                self._state = WorkerState.DEGRADED
            return None

        # kind == "ok"
        seq, tracked = msg[1], msg[2]
        self._pending.pop(seq, None)
        self._completed_count += 1
        self._consecutive_failures = 0
        self._last_result_at = time.monotonic()
        if self._state == WorkerState.DEGRADED:
            self._state = WorkerState.RUNNING
        return tracked

    def health(self) -> WorkerHealth:
        """Get current worker health and diagnostics.

        Returns:
            A WorkerHealth object containing state, pending work, and error info.
        """
        self._poll_process_liveness()
        last_age_ms = None
        if self._last_result_at is not None:
            last_age_ms = (time.monotonic() - self._last_result_at) * 1000.0
        return WorkerHealth(
            state=self._state,
            backend_name=self._backend_name,
            pending_submitted=len(self._pending),
            completed_count=self._completed_count,
            consecutive_failures=self._consecutive_failures,
            last_error=self._last_error,
            last_result_age_ms=last_age_ms,
        )

    def close(self) -> None:
        """Shut down the worker process and cleanup resources."""
        if self._closed:
            return
        self._closed = True
        self._send_queue.close()
        self._feeder_stop.set()
        # Bounded: if the feeder is itself stuck mid-send() to a wedged
        # child, this join can time out without blocking close() overall
        # -- the process terminate/kill below breaks the pipe out from
        # under it, so the (daemon) feeder thread unwinds on its own even
        # if we stop waiting for it here.
        self._feeder_thread.join(timeout=self._close_timeout_s)
        # Known edge case (not hit by this PR's tests, which use small
        # payloads): if the feeder thread is still stuck mid-send() past
        # close_timeout_s (a large in-flight backlog against a truly
        # wedged child), this send() and the feeder's could race on the
        # same Connection. Acceptable for now -- terminate()/kill() below
        # still bounds close() either way; a lock around all sends would
        # close this gap if it's ever observed in practice.
        try:
            self._parent_conn.send(_SHUTDOWN)
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=self._close_timeout_s)
        if self._process.is_alive():
            # Deliverable 3: a wedged/failed backend must never hang
            # shutdown -- terminate rather than block indefinitely.
            self._process.terminate()
            self._process.join(timeout=self._close_timeout_s)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self._close_timeout_s)
        try:
            self._parent_conn.close()
        except OSError:
            pass
        self._state = WorkerState.CLOSED

    # -- internal -------------------------------------------------------

    def _poll_process_liveness(self) -> None:
        if self._closed:
            return
        if not self._process.is_alive() and self._state not in (
            WorkerState.FAILED,
            WorkerState.CLOSED,
        ):
            self._state = WorkerState.FAILED
            if self._last_error is None:
                exitcode = self._process.exitcode
                self._last_error = f"worker process died (exitcode={exitcode})"
