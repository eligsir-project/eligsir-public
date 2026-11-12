"""Tests for MappingCoordinator's bounded-queue prepared-candidate pipeline
integration (plan section 8.6, Subagent D; PR W3-03).

Imports eligsir.mapping.coordinator, which transitively pulls in torch
via rgbd_integrator_impl -- run inside the training container
(``docker compose run --rm train python -m pytest tests/test_coordinator_bounded_pipeline.py``),
not on host Python.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.mapping.coordinator import MappingCoordinator
from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.mapping.commit_queue import CommitRequest
from eligsir.runtime.queues.bounded_queue import QueueOfferResult
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _intrinsics():
    return CameraIntrinsics(width=8, height=8, fx=10.0, fy=10.0, cx=4.0, cy=4.0, depth_scale=1000.0)


def _tracked(index: int, tx: float) -> TrackedFramePacket:
    rng = np.random.default_rng(index)
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    raw = RawFramePacket(
        source_index=index, timestamp_ns=index,
        rgb_ref=UndecodedBufferRef(array=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)),
        depth_ref=UndecodedBufferRef(), intrinsics=_intrinsics(),
    )
    return TrackedFramePacket(
        frame=raw, world_T_camera=m, tracking_state=TrackingState.OK,
        tracking_confidence=1.0, slam_map_id=0,
    )


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_coordinator_disabled_by_default():
    coordinator = MappingCoordinator()
    assert coordinator.bounded_pipeline is None
    coordinator.sync_source_backlog_from_pipeline()  # must be a safe no-op
    coordinator.shutdown_bounded_queue_pipeline()  # must be a safe no-op


def test_enable_bounded_queue_pipeline_is_idempotent():
    coordinator = MappingCoordinator()
    try:
        p1 = coordinator.enable_bounded_queue_pipeline(input_capacity=2, output_capacity=2, commit_capacity=2)
        p2 = coordinator.enable_bounded_queue_pipeline(input_capacity=99)
        assert p1 is p2
        assert coordinator.bounded_pipeline.input_queue._capacity == 2
    finally:
        coordinator.shutdown_bounded_queue_pipeline()


def test_bounded_pipeline_end_to_end_and_backlog_sync():
    coordinator = MappingCoordinator()
    try:
        coordinator.enable_bounded_queue_pipeline(input_capacity=4, output_capacity=4, commit_capacity=4)
        for i in range(3):
            result = coordinator.bounded_pipeline.submit_raw_frame(_tracked(i, float(i)))
            assert result is QueueOfferResult.ENQUEUED

        coordinator.sync_source_backlog_from_pipeline()
        summary = coordinator.four_clocks.summary()
        assert summary.source_queue_depth is not None

        assert _wait_until(lambda: coordinator.bounded_pipeline.worker.n_prepared >= 1)
        polled = coordinator.bounded_pipeline.poll_prepared_candidate()
        assert polled.item is not None

        request = CommitRequest(candidate=polled.item, slam_map_id=0)
        assert coordinator.bounded_pipeline.submit_commit(request) is QueueOfferResult.ENQUEUED
        drained = coordinator.bounded_pipeline.drain_commits()
        assert len(drained) == 1
    finally:
        coordinator.shutdown_bounded_queue_pipeline()


def test_shutdown_stops_worker_thread_without_deadlock():
    coordinator = MappingCoordinator()
    coordinator.enable_bounded_queue_pipeline(input_capacity=2, output_capacity=2, commit_capacity=2)
    worker_thread = coordinator.bounded_pipeline.worker._thread
    t0 = time.monotonic()
    coordinator.shutdown_bounded_queue_pipeline()
    assert time.monotonic() - t0 < 2.5
    assert coordinator.bounded_pipeline is None
    assert worker_thread is not None and not worker_thread.is_alive()
