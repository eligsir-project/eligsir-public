"""Real (not mocked) tests for ProcessSLAMWorker (PR W3-04, plan section
9). Uses multiprocessing for real, so these exercise the actual isolation
boundary the deliverables are about -- not a stand-in.

Deliverable 3 ("SLAM failure cannot deadlock the mapper") is covered here
as an actual timed test: a worker configured with a long artificial delay
must still let the caller's submit()/try_get()/close() calls return
promptly, and a killed/crashed child process must be observable via
health() rather than hanging the caller.
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np
import pytest

from eligsir.mapping.candidates.contracts import RawFramePacket, UndecodedBufferRef
from eligsir.slam.runtime.health import WorkerState
from eligsir.slam.runtime.process_worker import ProcessSLAMWorker
from eligsir.slam.runtime.worker import SubmitOutcome
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _frame(index: int) -> RawFramePacket:
    rng = np.random.default_rng(index)
    return RawFramePacket(
        source_index=index,
        timestamp_ns=index * 33_000_000,
        rgb_ref=UndecodedBufferRef(array=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)),
        depth_ref=UndecodedBufferRef(array=np.ones((8, 8), dtype=np.float32)),
        intrinsics=CameraIntrinsics(width=8, height=8, fx=10.0, fy=10.0, cx=4.0, cy=4.0),
    )


def _wait_until(predicate, timeout_s: float = 10.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_submit_accepted():
    w = ProcessSLAMWorker("dataset_pose", {"fallback_identity": True}, capacity=4)
    try:
        result = w.submit(_frame(0))
        assert result.outcome is SubmitOutcome.ACCEPTED
    finally:
        w.close()


def test_submit_then_poll_returns_tracked_packet():
    w = ProcessSLAMWorker("dataset_pose", {"fallback_identity": True}, capacity=4)
    try:
        w.submit(_frame(0))
        tracked = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and tracked is None:
            tracked = w.try_get()
            if tracked is None:
                time.sleep(0.02)
        assert tracked is not None
        assert tracked.frame.source_index == 0
    finally:
        w.close()


def test_500ms_artificial_delay_does_not_block_submit_or_tryget():
    """Deliverable 2's mechanism, isolated: submit()/try_get() must return
    in well under the artificial delay even while the child is inside a
    500ms-delayed track() call. (scripts/benchmark_slam_isolation.py is
    the full end-to-end optimization-loop benchmark; this is the unit-level
    proof of the same property.)"""
    w = ProcessSLAMWorker(
        "dataset_pose", {"fallback_identity": True}, capacity=4, artificial_delay_s=0.5
    )
    try:
        w.submit(_frame(0))
        time.sleep(0.05)  # let the child start the 500ms-delayed track()

        t0 = time.perf_counter()
        w.submit(_frame(1))
        submit_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        w.try_get()
        tryget_elapsed = time.perf_counter() - t0

        assert submit_elapsed < 0.1, f"submit() blocked for {submit_elapsed:.3f}s"
        assert tryget_elapsed < 0.1, f"try_get() blocked for {tryget_elapsed:.3f}s"
    finally:
        w.close()


def _large_frame(index: int) -> RawFramePacket:
    """A realistic 720p RGB-D frame -- large enough (~2.7MB) that a naive
    ``submit()`` implementation calling ``Connection.send()`` directly on
    the caller's thread would block once the OS pipe buffer (typically
    64KB) fills, defeating deliverables 1/2/3. See ProcessSLAMWorker's
    feeder-thread design (module docstring)."""
    return RawFramePacket(
        source_index=index,
        timestamp_ns=index * 33_000_000,
        rgb_ref=UndecodedBufferRef(array=np.zeros((720, 1280, 3), dtype=np.uint8)),
        depth_ref=UndecodedBufferRef(array=np.zeros((720, 1280), dtype=np.float32)),
        intrinsics=CameraIntrinsics(width=1280, height=720, fx=600.0, fy=600.0, cx=640.0, cy=360.0),
    )


def test_submit_never_blocks_caller_thread_even_with_large_frames_and_wedged_child():
    """Deliverable 1/2/3, real test at realistic frame size: submit() must
    stay non-blocking on the caller's thread even once the OS pipe buffer
    would fill, which is exactly the scenario a direct-send()-on-submit()
    implementation gets wrong (large frame payloads + a slow/wedged
    child)."""
    w = ProcessSLAMWorker(
        "dataset_pose",
        {"fallback_identity": True},
        capacity=8,
        artificial_delay_s=3600.0,  # wedge the child so nothing drains
    )
    try:
        max_submit_s = 0.0
        accepted = 0
        for i in range(8):
            t0 = time.perf_counter()
            result = w.submit(_large_frame(i))
            elapsed = time.perf_counter() - t0
            max_submit_s = max(max_submit_s, elapsed)
            if result.outcome is SubmitOutcome.ACCEPTED:
                accepted += 1
        assert accepted >= 1
        assert max_submit_s < 0.2, (
            f"submit() took up to {max_submit_s:.3f}s with large frames against a "
            "wedged child -- caller thread was blocked"
        )
    finally:
        t0 = time.perf_counter()
        w.close()
        assert time.perf_counter() - t0 < 5.0


def test_close_does_not_hang_on_wedged_backend():
    """Deliverable 3, real test: a worker whose child is stuck in an
    (effectively) infinite artificial delay must still have close() return
    within its bounded timeout, via terminate/kill -- never hang the
    caller indefinitely."""
    w = ProcessSLAMWorker(
        "dataset_pose",
        {"fallback_identity": True},
        capacity=4,
        artificial_delay_s=3600.0,  # effectively "hung"
        close_timeout_s=1.0,
    )
    w.submit(_frame(0))
    time.sleep(0.1)  # ensure the child has entered the long sleep

    t0 = time.perf_counter()
    w.close()
    elapsed = time.perf_counter() - t0

    # Bounded by close_timeout_s (terminate/join cycles), not by the
    # 3600s artificial delay.
    assert elapsed < 5.0, f"close() took {elapsed:.1f}s -- did not bound the wedged child"
    assert w.health().state is WorkerState.CLOSED


def test_killed_child_process_reported_as_failed_not_hung():
    """Deliverable 3: a SLAM backend crashing outright must not silently
    stall the mapper -- health() must observe FAILED, and submit() must
    reject rather than hang, once the child process is actually dead."""
    w = ProcessSLAMWorker("dataset_pose", {"fallback_identity": True}, capacity=4)
    try:
        w.submit(_frame(0))
        time.sleep(0.2)  # let the child fully start
        pid = w._process.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)

        detected = _wait_until(lambda: w.health().state is WorkerState.FAILED, timeout_s=5.0)
        assert detected, f"worker health never reflected process death: {w.health()}"

        t0 = time.perf_counter()
        result = w.submit(_frame(1))
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5
        assert result.outcome is SubmitOutcome.REJECTED_WORKER_FAILED
    finally:
        w.close()


def test_submit_rejected_when_queue_full():
    w = ProcessSLAMWorker(
        "dataset_pose", {"fallback_identity": True}, capacity=1, artificial_delay_s=2.0
    )
    try:
        first = w.submit(_frame(0))
        assert first.outcome is SubmitOutcome.ACCEPTED
        time.sleep(0.05)
        second = w.submit(_frame(1))
        assert second.outcome is SubmitOutcome.REJECTED_QUEUE_FULL
    finally:
        w.close()


def test_backend_track_error_is_counted_not_raised():
    # fallback_identity=False -> FramePosePassthroughTracker.track() raises inside
    # the child (frame has no embedded pose); this must surface as a
    # counted failure, never an exception out of try_get().
    w = ProcessSLAMWorker("dataset_pose", {"fallback_identity": False}, capacity=4)
    try:
        w.submit(_frame(0))
        _wait_until(lambda: w.health().consecutive_failures > 0 or w.try_get() is not None, timeout_s=5.0)
        health = w.health()
        assert health.consecutive_failures >= 1
        assert health.last_error is not None
    finally:
        w.close()
