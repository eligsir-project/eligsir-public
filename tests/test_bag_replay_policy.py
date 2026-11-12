"""Tests for BagReplayPolicy (PR W3-04): lossless bag-replay source
policy over BoundedQueue. Covers deliverable 4, "Bag frame loss forces
mapping_result_valid=false", as a real behavioral test."""

from __future__ import annotations

import threading
import time

from eligsir.io.policies.bag_replay import BagOfferOutcome, BagReplayPolicy
from eligsir.runtime.queues.bounded_queue import BoundedQueue


def test_direct_enqueue_when_queue_has_room():
    q = BoundedQueue(capacity=4)
    policy = BagReplayPolicy(q)
    outcome = policy.offer("frame-0")
    assert outcome is BagOfferOutcome.ENQUEUED_DIRECT
    assert policy.report.mapping_result_valid
    assert policy.report.frames_enqueued_direct == 1


def test_full_queue_spills_instead_of_dropping():
    q = BoundedQueue(capacity=1)
    policy = BagReplayPolicy(q, backpressure_max_wait_s=0.0)  # skip backpressure
    assert policy.offer("frame-0") is BagOfferOutcome.ENQUEUED_DIRECT
    outcome = policy.offer("frame-1")  # queue now full
    assert outcome is BagOfferOutcome.SPILLED
    assert policy.spill_depth == 1
    # No frame was lost -- still valid.
    assert policy.report.mapping_result_valid
    assert policy.report.frames_lost == 0


def test_spill_drains_in_order_once_queue_has_room():
    q = BoundedQueue(capacity=1)
    policy = BagReplayPolicy(q, backpressure_max_wait_s=0.0)
    policy.offer("frame-0")
    policy.offer("frame-1")  # spilled
    policy.offer("frame-2")  # also spilled (frame-1 still queued behind frame-0)
    assert policy.spill_depth == 2

    assert q.try_poll().item == "frame-0"
    moved = policy.flush_spill()
    assert moved == 1
    assert q.try_poll().item == "frame-1"
    assert policy.spill_depth == 1


def test_backpressure_succeeds_once_consumer_drains():
    q = BoundedQueue(capacity=1)
    policy = BagReplayPolicy(q, backpressure_max_wait_s=1.0, backpressure_poll_interval_s=0.01)
    policy.offer("frame-0")  # fills the queue

    def _drain_soon():
        time.sleep(0.05)
        q.try_poll()

    t = threading.Thread(target=_drain_soon)
    t.start()
    outcome = policy.offer("frame-1")
    t.join()

    assert outcome is BagOfferOutcome.ENQUEUED_AFTER_BACKPRESSURE
    assert policy.report.mapping_result_valid


def test_spill_exhaustion_forces_mapping_result_invalid():
    q = BoundedQueue(capacity=1)
    policy = BagReplayPolicy(q, backpressure_max_wait_s=0.0, max_spill=1)
    policy.offer("frame-0")  # fills queue
    policy.offer("frame-1")  # spills (spill depth 1, at cap)
    outcome = policy.offer("frame-2")  # spill cap exceeded -> real loss

    assert outcome is BagOfferOutcome.LOST_SPILL_EXHAUSTED
    assert policy.report.frames_lost == 1
    assert not policy.report.mapping_result_valid


def test_mapping_result_valid_is_a_one_way_latch():
    q = BoundedQueue(capacity=1)
    policy = BagReplayPolicy(q, backpressure_max_wait_s=0.0, max_spill=0)
    policy.offer("frame-0")  # fills queue
    policy.offer("frame-1")  # immediate loss (spill cap 0)
    assert not policy.report.mapping_result_valid

    # Draining the queue and offering more frames successfully afterward
    # must not "recover" validity -- a lost frame invalidates the run.
    q.try_poll()
    policy.offer("frame-2")
    assert not policy.report.mapping_result_valid


def test_geometry_fields_shape():
    q = BoundedQueue(capacity=4)
    policy = BagReplayPolicy(q)
    policy.offer("frame-0")
    fields = policy.report.as_geometry_fields()
    assert fields["mapping_result_valid"] is True
    assert fields["bag_frames_offered"] == 1
    assert "bag_loss_events" in fields
