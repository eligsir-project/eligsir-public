"""Tests for eligsir.runtime.profiling.queue_metrics.QueueMetrics."""

import time
from unittest.mock import patch

import pytest

from eligsir.runtime.profiling import events as events_module
from eligsir.runtime.profiling.queue_metrics import QueueMetrics


@pytest.fixture(autouse=True)
def _profiling_enabled():
    prior = events_module.PROFILING_ENABLED
    events_module.set_profiling_enabled(True)
    yield
    events_module.set_profiling_enabled(prior)


def test_depth_tracks_enqueue_dequeue_sequence():
    q = QueueMetrics(max_depth=4)
    assert q.depth == 0
    q.on_enqueue()
    q.on_enqueue()
    assert q.depth == 2
    q.on_dequeue()
    assert q.depth == 1
    q.on_dequeue()
    assert q.depth == 0
    # Dequeue below zero must not go negative.
    q.on_dequeue()
    assert q.depth == 0


def test_max_depth_is_configured_capacity():
    q = QueueMetrics(max_depth=17)
    assert q.max_depth == 17


def test_drops_accumulate_with_last_reason():
    q = QueueMetrics(max_depth=1)
    assert q.cumulative_drops == 0
    assert q.last_drop_reason is None

    q.on_drop("queue_full")
    assert q.cumulative_drops == 1
    assert q.last_drop_reason == "queue_full"

    q.on_drop("stale_source")
    assert q.cumulative_drops == 2
    assert q.last_drop_reason == "stale_source"


def test_oldest_item_age_grows_and_resets_on_dequeue():
    q = QueueMetrics(max_depth=4)
    q.on_enqueue()
    time.sleep(0.03)
    age_before = q.oldest_item_age_ms()
    assert age_before > 0.0

    q.on_enqueue()
    q.on_dequeue()  # removes the oldest item
    age_after = q.oldest_item_age_ms()
    # Oldest is now the second (more recent) item, so age should be smaller.
    assert age_after < age_before


def test_oldest_item_age_zero_on_empty_queue():
    q = QueueMetrics(max_depth=4)
    assert q.oldest_item_age_ms() == 0.0


def test_snapshot_shape():
    q = QueueMetrics(max_depth=8)
    q.on_enqueue()
    q.on_drop("full")
    snap = q.snapshot()
    assert snap == {
        "depth": 1,
        "capacity": 8,
        "max_depth": 8,
        "peak_depth": 1,
        "oldest_item_age_ms": pytest.approx(snap["oldest_item_age_ms"]),
        "cumulative_drops": 1,
        "last_drop_reason": "full",
    }


def test_profiling_disabled_skips_timestamp_bookkeeping():
    """When profiling is off, oldest_item_age_ms must be a fixed 0.0 without ever reading
    a timestamp for enqueued items -- verified by inspecting internal state, not just the
    reported value."""
    events_module.set_profiling_enabled(False)
    q = QueueMetrics(max_depth=4)
    q.on_enqueue()
    q.on_enqueue()

    # Depth bookkeeping (cheap counters) still works even with profiling off.
    assert q.depth == 2
    # But no enqueue timestamps were ever recorded.
    assert len(q._enqueue_times) == 0
    assert q.oldest_item_age_ms() == 0.0


def test_profiling_disabled_never_calls_perf_counter_for_enqueue():
    events_module.set_profiling_enabled(False)
    q = QueueMetrics(max_depth=4)
    with patch("time.perf_counter") as mock_perf_counter:
        q.on_enqueue()
        q.on_enqueue()
        mock_perf_counter.assert_not_called()
