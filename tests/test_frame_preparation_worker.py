"""Tests for eligsir.runtime.workers.frame_preparation_worker (PR W3-03).

Includes the plan's section 8.8 "no deadlock under producer failure or
consumer shutdown" acceptance gate as a real test, not an assertion.
"""

from __future__ import annotations

import time

import numpy as np

from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.runtime.queues.bounded_queue import BoundedQueue
from eligsir.runtime.workers.frame_preparation_worker import (
    FramePreparationWorker,
    FramePreparationWorkerConfig,
)
from eligsir.mapping.candidates.prefilter import PrefilterConfig
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _intrinsics():
    return CameraIntrinsics(width=8, height=8, fx=10.0, fy=10.0, cx=4.0, cy=4.0, depth_scale=1000.0)


def _c2w(tx: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return m


def _tracked(index: int, tx: float, state: TrackingState = TrackingState.OK) -> TrackedFramePacket:
    rng = np.random.default_rng(index)
    raw = RawFramePacket(
        source_index=index, timestamp_ns=index * 1_000_000,
        rgb_ref=UndecodedBufferRef(array=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)),
        depth_ref=UndecodedBufferRef(), intrinsics=_intrinsics(),
    )
    return TrackedFramePacket(
        frame=raw, world_T_camera=_c2w(tx), tracking_state=state,
        tracking_confidence=1.0, slam_map_id=0,
    )


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_worker_prepares_novel_frames_onto_output_queue():
    in_q: BoundedQueue = BoundedQueue(capacity=4)
    out_q: BoundedQueue = BoundedQueue(capacity=4)
    worker = FramePreparationWorker(in_q, out_q).start()
    try:
        in_q.try_offer(_tracked(0, 0.0))
        in_q.try_offer(_tracked(1, 2.0))
        assert _wait_until(lambda: worker.n_prepared >= 2)
        polled = out_q.try_poll()
        assert polled.item is not None
    finally:
        worker.stop()


def test_worker_skips_decode_for_low_novelty_frames_without_crashing():
    in_q: BoundedQueue = BoundedQueue(capacity=4)
    out_q: BoundedQueue = BoundedQueue(capacity=4)
    config = FramePreparationWorkerConfig(
        prefilter=PrefilterConfig(min_pose_novelty=0.5, novelty_translation_norm_m=1.0)
    )
    worker = FramePreparationWorker(in_q, out_q, config=config).start()
    try:
        in_q.try_offer(_tracked(0, 0.0))
        in_q.try_offer(_tracked(1, 0.0001))  # barely moved -> should be skipped
        assert _wait_until(lambda: worker.n_prepared >= 1 or worker.n_skipped_decode >= 1)
        time.sleep(0.05)  # let the second (skipped) item settle
        assert worker.n_skipped_decode >= 1
    finally:
        worker.stop()


def test_worker_shuts_down_promptly_without_deadlock():
    """Acceptance gate (plan 8.8): no deadlock under consumer shutdown."""
    in_q: BoundedQueue = BoundedQueue(capacity=4)
    out_q: BoundedQueue = BoundedQueue(capacity=4)
    worker = FramePreparationWorker(in_q, out_q).start()
    for i in range(4):
        in_q.try_offer(_tracked(i, float(i)))

    t0 = time.monotonic()
    worker.stop(join_timeout_s=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.5, "worker.stop() did not return promptly -- possible deadlock"


def test_worker_propagates_producer_failure_to_output_queue_without_deadlock():
    """Acceptance gate (plan 8.8): no deadlock under producer failure."""
    in_q: BoundedQueue = BoundedQueue(capacity=4)
    out_q: BoundedQueue = BoundedQueue(capacity=4)
    worker = FramePreparationWorker(in_q, out_q).start()
    try:
        in_q.fail(RuntimeError("SLAM frontend crashed"))
        assert _wait_until(lambda: out_q.try_poll().state.value == "failed")
    finally:
        worker.stop(join_timeout_s=2.0)


def test_worker_closed_input_queue_closes_output_queue_without_deadlock():
    in_q: BoundedQueue = BoundedQueue(capacity=4)
    out_q: BoundedQueue = BoundedQueue(capacity=4)
    worker = FramePreparationWorker(in_q, out_q).start()
    try:
        in_q.close()
        assert _wait_until(lambda: out_q.try_poll().state.value == "closed")
    finally:
        worker.stop(join_timeout_s=2.0)
