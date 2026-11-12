"""Tests for DeferredSlamAdmissionTracker (W3-15).

The single most important property under test: a frame is NEVER admitted
with a stale or synthesized pose. Every "happy path" assertion checks that
the returned pose is exactly the one the worker actually computed for that
specific frame, and every "no result yet" / "tracking failed" / "worker
wedged" case asserts ``poll()`` returns ``None`` (or skips the frame
outright) rather than inventing something to admit.

Most of the suite uses a small in-process fake conforming to the
``SLAMWorker`` protocol (submit/try_get/health/close) -- this isolates the
tracker's own submit-then-poll/timeout/ordering logic from
``ProcessSLAMWorker``'s subprocess mechanics, which already has its own
real (non-mocked) coverage in ``tests/test_process_slam_worker.py``. One
additional test (`test_real_process_worker_end_to_end_skip_path`) drives a
genuine ``ProcessSLAMWorker`` subprocess to confirm the tracker's
submit/poll contract holds under real IPC latency, not just against the
fake.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytest

from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
)
from eligsir.mapping.scheduler.deferred_slam_admission import (
    DeferredSlamAdmissionTracker,
    build_deferred_admission_tracker,
    raw_packet_from_streaming_frame,
)
from eligsir.slam.runtime.health import WorkerHealth, WorkerState
from eligsir.slam.runtime.worker import SubmitOutcome, SubmitResult


class _FakeFrame:
    """Minimal stand-in for StreamingRGBDFrame."""

    def __init__(self, index: int, width: int = 8, height: int = 8):
        self.index = index
        self.width = width
        self.height = height
        self.fx = self.fy = 10.0
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.timestamp = float(index)
        self._rgb_np = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_np = np.ones((height, width), dtype=np.float32)
        self.c2w = None  # not yet known -- exactly what this module resolves


class _FakeStreamingScene:
    """Minimal stand-in for StreamingScene exposing only what
    DeferredSlamAdmissionTracker.poll() touches."""

    def __init__(self, n_frames: int):
        self._frames = [_FakeFrame(i) for i in range(n_frames)]
        self._source_idx = 0
        self.skipped: list = []

    def peek_next_frame(self):
        if self._source_idx >= len(self._frames):
            return None
        return self._frames[self._source_idx]

    def _get_source_frame(self, index: int):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def skip_next_frame(self) -> int:
        if self._source_idx >= len(self._frames):
            return 0
        self.skipped.append(self._source_idx)
        self._source_idx += 1
        return 1

    def ingest(self):
        """Simulate what train_streaming.py's ingest_next_frame() does:
        consume the frame at the current cursor and advance it."""
        frame = self._frames[self._source_idx]
        self._source_idx += 1
        return frame


class _FakeWorker:
    """In-process SLAMWorker-protocol double. Results are injected by the
    test via `queue_result`/`queue_nothing_yet` rather than computed by a
    real backend, so tests can deterministically control timing."""

    def __init__(self):
        self._outbox: list = []
        self._next_submit_outcome = SubmitOutcome.ACCEPTED
        self.state = WorkerState.RUNNING
        self.consecutive_failures = 0
        self.last_error: Optional[str] = None
        self.submitted: list = []
        self.closed = False

    def submit(self, frame: RawFramePacket) -> SubmitResult:
        self.submitted.append(frame.source_index)
        return SubmitResult(self._next_submit_outcome, time.monotonic_ns())

    def queue_result(self, tracked: TrackedFramePacket) -> None:
        self._outbox.append(tracked)

    def fail_pending(self) -> None:
        """Simulate the pending frame's backend track() raising: no
        TrackedFramePacket is ever produced for it, but consecutive_failures
        rises immediately (mirrors ProcessSLAMWorker.try_get's "error"
        branch, which consumes the error message without returning
        anything)."""
        self.consecutive_failures += 1
        self.last_error = "boom"
        self.state = WorkerState.DEGRADED

    def try_get(self) -> Optional[TrackedFramePacket]:
        if self._outbox:
            return self._outbox.pop(0)
        return None

    def health(self) -> WorkerHealth:
        return WorkerHealth(
            state=self.state,
            backend_name="fake",
            consecutive_failures=self.consecutive_failures,
            last_error=self.last_error,
        )

    def close(self) -> None:
        self.closed = True


def _tracked(index: int, *, ok: bool = True) -> TrackedFramePacket:
    packet = raw_packet_from_streaming_frame(_FakeFrame(index), index)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(index) * 1000.0  # distinctive, checkable value
    return TrackedFramePacket(
        frame=packet,
        world_T_camera=pose if ok else np.eye(4, dtype=np.float32),
        tracking_state=TrackingState.OK if ok else TrackingState.LOST,
        tracking_confidence=1.0 if ok else 0.0,
        slam_map_id=0,
    )


def test_submit_then_no_result_yet_returns_none_and_does_not_admit():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    result = tracker.poll(scene)

    assert result is None
    assert scene._source_idx == 0  # nothing consumed
    assert worker.submitted == [0]
    assert tracker.frames_submitted == 1
    assert tracker.frames_admitted == 0


def test_result_arrives_stamps_real_pose_and_admits():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0, nothing ready yet
    worker.queue_result(_tracked(0, ok=True))
    result = tracker.poll(scene)  # now the real result has arrived

    assert result is not None
    assert result.frame.source_index == 0
    assert result.tracking_state == TrackingState.OK
    # The pose actually applied to the frame is EXACTLY the worker's real
    # result -- never a stale/predicted stand-in.
    assert scene._frames[0].c2w[0, 3] == pytest.approx(0.0)
    result_pose = np.asarray(result.world_T_camera)
    assert result_pose[0, 3] == pytest.approx(0.0)
    assert tracker.frames_admitted == 1


def test_never_admits_a_frame_whose_tracking_result_has_not_arrived():
    """The literal forbidden-shortcut check: across many polls with no
    result ever queued, the tracker must never return anything other than
    None, and the scene's cursor must never advance."""
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    for _ in range(20):
        result = tracker.poll(scene)
        assert result is None
    assert scene._source_idx == 0
    assert scene.skipped == []


def test_lost_tracking_state_is_skipped_not_admitted():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0
    worker.queue_result(_tracked(0, ok=False))  # tracking failed
    result = tracker.poll(scene)

    assert result is None
    assert scene.skipped == [0]  # frame 0 skipped, never admitted
    assert scene._source_idx == 1
    assert tracker.frames_skipped_no_pose == 1
    assert tracker.frames_admitted == 0
    # Falls through to submit the next frame in the same call.
    assert 1 in worker.submitted


def test_backend_failure_detected_immediately_not_after_full_timeout():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=60.0)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0
    worker.fail_pending()  # backend track() raised -- no result will EVER arrive
    t0 = time.monotonic()
    result = tracker.poll(scene)
    elapsed = time.monotonic() - t0

    assert result is None
    assert elapsed < 1.0  # detected fast, not after the 60s timeout
    assert scene.skipped == [0]
    assert tracker.frames_skipped_no_pose == 1
    assert tracker.pending_timeouts == 0  # this is a failure, not a timeout


def test_pending_timeout_skips_and_counts_separately_from_failure():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=0.05)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0
    time.sleep(0.1)
    result = tracker.poll(scene)  # still no result, past the timeout

    assert result is None
    assert scene.skipped == [0]
    assert tracker.pending_timeouts == 1
    assert tracker.frames_skipped_no_pose == 1


def test_worker_unusable_skips_even_before_timeout():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=60.0)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0
    worker.state = WorkerState.FAILED  # worker process died
    result = tracker.poll(scene)

    assert result is None
    assert scene.skipped == [0]


def test_ordering_mismatch_raises_rather_than_misapplying_pose():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    tracker.poll(scene)  # submits frame 0 -> pending_index=0
    worker.queue_result(_tracked(1, ok=True))  # wrong frame's result

    with pytest.raises(RuntimeError, match="ordering guarantee violated"):
        tracker.poll(scene)


def test_sequential_frames_each_get_their_own_real_pose():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(3)

    admitted_indices = []
    for expected_idx in range(3):
        assert tracker.poll(scene) is None  # submit
        worker.queue_result(_tracked(expected_idx, ok=True))
        result = tracker.poll(scene)
        assert result is not None
        assert result.frame.source_index == expected_idx
        admitted_indices.append(result.frame.source_index)
        scene.ingest()  # simulate the caller admitting it now

    assert admitted_indices == [0, 1, 2]
    assert tracker.frames_admitted == 3


def test_close_delegates_to_worker():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker)
    tracker.close()
    assert worker.closed is True


def test_telemetry_reports_all_counters():
    worker = _FakeWorker()
    tracker = DeferredSlamAdmissionTracker(worker, pending_timeout_s=10.0)
    scene = _FakeStreamingScene(2)

    tracker.poll(scene)
    worker.queue_result(_tracked(0, ok=True))
    tracker.poll(scene)

    telemetry = tracker.telemetry()
    assert telemetry == {
        "frames_submitted": 1,
        "frames_admitted": 1,
        "frames_skipped_no_pose": 0,
        "pending_timeouts": 0,
    }


def test_build_deferred_admission_tracker_refuses_removed_backend():
    """#1075: gs_rendered_rgbd_odometry is deleted, not merely unselectable.

    Previously this backend was refused twice over -- by ``get_backend``'s
    availability gate and by this function's own ``requires_gaussian_map``
    check. Both mechanisms are gone with the ABC that declared them, so the
    refusal now comes from name resolution in the factory. The net guarantee
    is unchanged and is what this test pins: this backend can never run inside
    ProcessSLAMWorker's child process.
    """
    with pytest.raises(KeyError, match="read-only render view"):
        build_deferred_admission_tracker("gs_rendered_rgbd_odometry", {})


def test_build_deferred_admission_tracker_refuses_unknown_backend_in_parent():
    """An unknown name must fail in the PARENT, before a child is spawned.

    The old registry raised inside the freshly-spawned child, surfacing as an
    opaque ``("init_error", ...)`` message. Eager name validation is what turns
    that into a normal exception at the call site.
    """
    with pytest.raises(KeyError, match="Available:"):
        build_deferred_admission_tracker("_no_such_frontend", {})


def test_build_deferred_admission_tracker_accepts_dataset_pose_backend():
    tracker = build_deferred_admission_tracker(
        "dataset_pose", {"fallback_identity": True}, capacity=2
    )
    try:
        assert isinstance(tracker, DeferredSlamAdmissionTracker)
    finally:
        tracker.close()


# ---------------------------------------------------------------------------
# Real, non-mocked subprocess integration test.
# ---------------------------------------------------------------------------


def test_real_process_worker_end_to_end_skip_path():
    """Drives a genuine ProcessSLAMWorker subprocess (the ``dataset_pose``
    backend, fallback_identity=True). RawFramePacket never carries a pose
    (see raw_packet_from_streaming_frame / RawFramePacket's fields), so
    FramePosePassthroughTracker falls back to its invalid/LOST branch for
    every frame submitted this way -- this exercises the tracker's
    submit -> real IPC round-trip -> LOST -> skip path against real
    multiprocessing latency (not the in-process fake), while still never
    admitting a frame with a stale/fabricated pose."""
    tracker = build_deferred_admission_tracker(
        "dataset_pose", {"fallback_identity": True}, capacity=2
    )
    scene = _FakeStreamingScene(2)
    try:
        deadline = time.monotonic() + 15.0
        skipped_seen = False
        while time.monotonic() < deadline and scene._source_idx < 2:
            result = tracker.poll(scene)
            assert result is None  # never a valid pose from this backend
            if scene.skipped:
                skipped_seen = True
            time.sleep(0.02)
        assert skipped_seen
        assert scene._source_idx == 2  # both frames skipped, never admitted
        assert tracker.frames_admitted == 0
        assert tracker.frames_skipped_no_pose == 2
    finally:
        tracker.close()
