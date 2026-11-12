"""Tests for train_streaming._mirror_commit_to_pending_queue (PR W3-03,
"missing piece 1" -- shadow-mirroring the real accepted-frame commit into
PendingCommitQueue, see that function's docstring for why it is
observation-only and does not become the real commit path in this PR).

Imports train_streaming, which transitively pulls in torch -- run inside
the training container
(``docker compose run --rm train python -m pytest tests/test_mirror_commit_to_pending_queue.py``),
not on host Python.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_streaming import _mirror_commit_to_pending_queue
from eligsir.mapping.coordinator import MappingCoordinator
from eligsir.runtime.queues.bounded_queue import QueueOfferResult


def _frame(index: int = 3, timestamp: float = 1.5):
    return SimpleNamespace(
        index=index,
        timestamp=timestamp,
        c2w=np.eye(4, dtype=np.float32),
    )


def test_mirror_submits_and_drains_without_touching_real_state():
    coordinator = MappingCoordinator()
    try:
        coordinator.enable_bounded_queue_pipeline(
            input_capacity=2, output_capacity=2, commit_capacity=2,
        )
        _mirror_commit_to_pending_queue(coordinator, _frame())
        metrics = coordinator.bounded_pipeline.commit_queue.metrics()
        # The function drains immediately after submitting (barrier drain
        # every call, see its docstring) -- steady-state depth is 0, but
        # the enqueue/dequeue counters prove the queue was really
        # exercised, not skipped.
        assert metrics.depth == 0
        assert metrics.enqueued_count == 1
        assert metrics.dequeued_count == 1
        assert metrics.dropped_count == 0
    finally:
        coordinator.shutdown_bounded_queue_pipeline()


def test_mirror_raises_when_called_without_enabled_pipeline():
    """Guard against a future caller invoking this without checking
    ``bounded_pipeline is not None`` first -- the real call site in
    streaming_training() always guards it, this documents why (fail-fast,
    not a silent no-op, when that guard is skipped)."""
    coordinator = MappingCoordinator()
    assert coordinator.bounded_pipeline is None
    with pytest.raises(AttributeError):
        _mirror_commit_to_pending_queue(coordinator, _frame())


def test_mirror_no_pose_is_a_safe_noop():
    coordinator = MappingCoordinator()
    try:
        coordinator.enable_bounded_queue_pipeline(
            input_capacity=2, output_capacity=2, commit_capacity=2,
        )
        frame = SimpleNamespace(index=0, timestamp=0.0, c2w=None)
        _mirror_commit_to_pending_queue(coordinator, frame)
        metrics = coordinator.bounded_pipeline.commit_queue.metrics()
        assert metrics.enqueued_count == 0
    finally:
        coordinator.shutdown_bounded_queue_pipeline()


def test_mirror_backpressure_when_commit_queue_full_and_worker_not_draining():
    """submit_commit can legitimately return DROPPED_QUEUE_FULL under real
    backpressure (see PendingCommitQueue's module docstring) -- verify the
    mirror does not raise or silently corrupt state when that happens; it
    should just leave the drop counted in metrics()."""
    coordinator = MappingCoordinator()
    try:
        coordinator.enable_bounded_queue_pipeline(
            input_capacity=2, output_capacity=2, commit_capacity=1,
        )
        # Pre-fill the commit queue to capacity directly so the mirror's
        # own submit is guaranteed to observe DROPPED_QUEUE_FULL (the
        # mirror always drains immediately after submitting, so we can't
        # force this purely through repeated calls).
        from eligsir.mapping.candidates.contracts import (
            InputEvidence,
            PreparedDepthRef,
            PreparedFrameCandidate,
            PreparedImageRef,
        )
        from eligsir.mapping.commit_queue import CommitRequest
        from eligsir.contracts.types import Pose

        filler = CommitRequest(
            candidate=PreparedFrameCandidate(
                source_index=0, timestamp_ns=0,
                pose=Pose(np.eye(4, dtype=np.float32)),
                rgb_ref=PreparedImageRef(array=None, width=0, height=0),
                depth_ref=PreparedDepthRef(array=None, width=0, height=0, valid_fraction=0.0),
                input_evidence=InputEvidence(
                    blur_laplacian=0.0, exposure_luma=0.0, white_clip_ratio=0.0,
                    black_clip_ratio=0.0, valid_depth_fraction=0.0, quality_weight=1.0,
                ),
                coarse_novelty=0.0,
            ),
            slam_map_id=0,
        )
        assert coordinator.bounded_pipeline.submit_commit(filler) is QueueOfferResult.ENQUEUED

        _mirror_commit_to_pending_queue(coordinator, _frame())
        metrics = coordinator.bounded_pipeline.commit_queue.metrics()
        # The mirrored request itself was dropped (queue was already at
        # capacity=1 from the filler above); the function's own barrier
        # drain then still runs unconditionally and drains the retained
        # filler, so steady-state depth is 0 again.
        assert metrics.dropped_count == 1
        assert metrics.depth == 0
        assert metrics.dequeued_count == 1
    finally:
        coordinator.shutdown_bounded_queue_pipeline()
