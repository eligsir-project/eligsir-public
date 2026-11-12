"""Unit tests for issue #333 phase 3 -- access-aware LRU / touch_resolution_cache.

Tests verify that:
1. bind_resolution_tier with db parameter calls touch_resolution_cache on resident hits
2. touch_resolution_cache refreshes recency, moving a frame to the end of the LRU order
3. LRU eviction is access-aware (MRU frames are kept, LRU frames evicted)
4. Existing eviction accounting and telemetry counters keep their semantics
5. touch_resolution_cache returns correct status (True for resident, False otherwise)
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import BufferStage, FrameObservationDB, FrameRecord
from eligsir.mapping.resolution_cache import CacheHandle
from eligsir.mapping.tier_bind import (
    bind_resolution_tier,
    OUTCOME_RESIDENT_HIT,
    OUTCOME_ALREADY_AT_TIER,
)


class MockStreamingFrame:
    """Mock streaming frame for testing."""
    def __init__(self, frame_id, width=640, height=480):
        self.frame_id = frame_id
        self.width = width
        self.height = height


class MockIntrinsics:
    """Mock camera intrinsics."""
    def __init__(self, fx=500, fy=500, cx=320, cy=240, width=640, height=480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height


class MockCacheHandle:
    """Mock cache handle for testing."""
    def __init__(self, tier_name="high"):
        self.rgb = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)
        self.depth = None  # Simplified: depth-less for test
        self.intrinsics = MockIntrinsics()
        self.tier_name = tier_name


class MockCamera:
    """Mock camera object for testing."""
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.original_image = None
        self.image_height = 480
        self.image_width = 640
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self._sensor_depth_cache = None
        self._bound_resolution_tier = None
        self._resolution_tier_mismatch = None
        self._native_resolution_state = None
        self._native_matches_high_tier = None
        self._streaming_frame = MockStreamingFrame(0)


def make_frame_record(frame_id: int, visible_uids=None) -> FrameRecord:
    """Create a properly initialized FrameRecord for testing."""
    if visible_uids is None:
        visible_uids = torch.tensor([1, 2], dtype=torch.int64)
    elif not isinstance(visible_uids, torch.Tensor):
        visible_uids = torch.tensor(visible_uids, dtype=torch.int64)

    pose_c2w = np.eye(4, dtype=np.float32)
    timestamp = float(frame_id)
    submap_id = 0
    buffer_stage = BufferStage.MAPPING_KEYFRAME

    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=pose_c2w,
        timestamp=timestamp,
        submap_id=submap_id,
        buffer_stage=buffer_stage,
        visible_gaussian_uids=visible_uids,
    )


@pytest.fixture
def observation_db():
    """Create a minimal observation DB for testing."""
    return FrameObservationDB(max_resolution_cache_frames=3, device="cpu")


@pytest.fixture
def cache_handle():
    """Create a mock cache handle."""
    handle = MockCacheHandle()
    return handle


def test_touch_resolution_cache_returns_false_for_nonexistent_frame(observation_db):
    """touch_resolution_cache returns False for a frame that was never added."""
    result = observation_db.touch_resolution_cache(999)
    assert result is False


def test_touch_resolution_cache_returns_false_for_evicted_frame(observation_db):
    """touch_resolution_cache returns False for a frame whose cache was evicted."""
    # Add a frame with a cache
    record = make_frame_record(0, visible_uids=[1, 2, 3])
    record.resolution_cache = {"high": MockCacheHandle()}
    observation_db.add_frame(record)

    # Manually mark it as evicted (simulate LRU eviction)
    record.resolution_cache = {}
    record.resolution_cache_evicted = True

    # touch_resolution_cache should return False
    result = observation_db.touch_resolution_cache(0)
    assert result is False


def test_touch_resolution_cache_refreshes_resident_frame(observation_db):
    """touch_resolution_cache returns True and moves a resident frame to the end."""
    # Add two frames with caches
    record0 = make_frame_record(0, visible_uids=[1, 2])
    record0.resolution_cache = {"high": MockCacheHandle()}
    observation_db.add_frame(record0)

    record1 = make_frame_record(1, visible_uids=[3, 4])
    record1.resolution_cache = {"high": MockCacheHandle()}
    observation_db.add_frame(record1)

    # Initially, order should be [0, 1]
    initial_order = list(observation_db._resolution_cache_order.keys())
    assert initial_order == [0, 1]

    # Touch frame 0 to refresh its recency
    result = observation_db.touch_resolution_cache(0)
    assert result is True

    # After touch, frame 0 should be at the end: [1, 0]
    new_order = list(observation_db._resolution_cache_order.keys())
    assert new_order == [1, 0]


def test_lru_eviction_is_access_aware(observation_db):
    """LRU eviction prioritizes access recency, not just insertion order.

    With max_frames=3:
    - Add frames 0, 1, 2 (all resident)
    - Touch frame 0 (refresh its recency) -> order: [1, 2, 0]
    - Add frame 3 (triggers eviction) -> frame 1 (least recently used) should be evicted
    - Without access-aware LRU, frame 0 (oldest insertion) would be evicted instead
    """
    # Add three frames to fill the cache
    for fid in [0, 1, 2]:
        record = make_frame_record(fid, visible_uids=[1, 2])
        record.resolution_cache = {"high": MockCacheHandle()}
        observation_db.add_frame(record)

    # At this point, order is [0, 1, 2]
    order_before_touch = list(observation_db._resolution_cache_order.keys())
    assert order_before_touch == [0, 1, 2]

    # Touch frame 0 to refresh its recency
    observation_db.touch_resolution_cache(0)
    order_after_touch = list(observation_db._resolution_cache_order.keys())
    assert order_after_touch == [1, 2, 0]  # 0 moves to the end

    # Add frame 3 (the oldest insertion frame 0 now has highest recency due to touch)
    record3 = make_frame_record(3, visible_uids=[1, 2])
    record3.resolution_cache = {"high": MockCacheHandle()}
    observation_db.add_frame(record3)

    # Frame 1 (the LRU after touch) should be evicted, not frame 0
    order_after_eviction = list(observation_db._resolution_cache_order.keys())
    assert order_after_eviction == [2, 0, 3]

    # Verify frame 1's cache was evicted
    frame1 = observation_db.get_frame(1)
    assert frame1.resolution_cache == {}
    assert frame1.resolution_cache_evicted is True

    # Verify frames 2, 0, 3 still have caches
    for fid in [2, 0, 3]:
        frame = observation_db.get_frame(fid)
        assert frame.resolution_cache != {}
        assert frame.resolution_cache_evicted is False


def test_bind_resolution_tier_touches_cache_on_resident_hit(observation_db):
    """bind_resolution_tier calls touch_resolution_cache on a resident cache hit."""
    # Create a camera and a frame record with a resident cache
    cam = MockCamera()
    record = make_frame_record(0, visible_uids=[1, 2])

    # Create a minimal cache handle (simplified for CPU testing)
    handle = MockCacheHandle("high")
    record.resolution_cache = {"high": handle}
    observation_db.add_frame(record)

    # Add a second frame so we can see the LRU order change
    record2 = make_frame_record(1, visible_uids=[1, 2])
    record2.resolution_cache = {"high": handle}
    observation_db.add_frame(record2)

    # Order should be [0, 1]
    order_before = list(observation_db._resolution_cache_order.keys())
    assert order_before == [0, 1]

    # Bind the tier (with db parameter) -- this should call touch_resolution_cache
    result = bind_resolution_tier(cam, record, "high", db=observation_db)

    # Binding should succeed
    assert result.ok is True
    assert result.outcome == OUTCOME_RESIDENT_HIT

    # Frame 0 should now be at the end due to the touch
    order_after = list(observation_db._resolution_cache_order.keys())
    assert order_after == [1, 0]


def test_bind_resolution_tier_without_db_parameter(observation_db):
    """bind_resolution_tier works without db parameter (backward compatibility)."""
    cam = MockCamera()
    record = make_frame_record(0, visible_uids=[1, 2])

    handle = MockCacheHandle("high")
    record.resolution_cache = {"high": handle}
    observation_db.add_frame(record)

    # Bind without db parameter
    result = bind_resolution_tier(cam, record, "high")  # No db parameter

    # Should still succeed
    assert result.ok is True
    assert result.outcome == OUTCOME_RESIDENT_HIT


def test_eviction_accounting_unchanged(observation_db):
    """Existing eviction accounting telemetry counters keep their semantics."""
    # Add frames and trigger eviction
    for fid in range(5):
        record = make_frame_record(fid, visible_uids=[1])
        record.resolution_cache = {"high": MockCacheHandle()}
        observation_db.add_frame(record)

    # With max_frames=3 and 5 frames added, 2 evictions should have occurred
    assert observation_db.resolution_cache_evictions == 2


def test_touch_nonexistent_cache_returns_false(observation_db):
    """touch_resolution_cache returns False for a frame in the DB but without a cache."""
    # Add a frame record WITHOUT a cache
    record = make_frame_record(0, visible_uids=[1, 2])
    record.resolution_cache = {}  # Empty cache
    observation_db.add_frame(record)

    # Should not be in _resolution_cache_order since it has no cache
    assert 0 not in observation_db._resolution_cache_order

    # Touch should return False
    result = observation_db.touch_resolution_cache(0)
    assert result is False


def test_repeated_touches_maintain_lru_order(observation_db):
    """Repeated touches keep a frame at the end (most recently used)."""
    # Add three frames
    for fid in [0, 1, 2]:
        record = make_frame_record(fid, visible_uids=[1])
        record.resolution_cache = {"high": MockCacheHandle()}
        observation_db.add_frame(record)

    # Touch frame 0 multiple times
    observation_db.touch_resolution_cache(0)  # [1, 2, 0]
    observation_db.touch_resolution_cache(0)  # [1, 2, 0] (no change, already at end)
    observation_db.touch_resolution_cache(0)  # [1, 2, 0] (no change, already at end)

    # Frame 0 should still be at the end
    order = list(observation_db._resolution_cache_order.keys())
    assert order == [1, 2, 0]

    # Now touch frame 1
    observation_db.touch_resolution_cache(1)  # [2, 0, 1]
    order = list(observation_db._resolution_cache_order.keys())
    assert order == [2, 0, 1]


def test_touch_after_eviction_returns_false(observation_db):
    """After a frame's cache is evicted, touch returns False."""
    # Add three frames to fill the cache
    for fid in [0, 1, 2]:
        record = make_frame_record(fid, visible_uids=[1])
        record.resolution_cache = {"high": MockCacheHandle()}
        observation_db.add_frame(record)

    # Add a fourth frame, causing frame 0 to be evicted
    record3 = make_frame_record(3, visible_uids=[1])
    record3.resolution_cache = {"high": MockCacheHandle()}
    observation_db.add_frame(record3)

    # Now frame 0 is evicted
    frame0 = observation_db.get_frame(0)
    assert frame0.resolution_cache_evicted is True

    # Trying to touch frame 0 should return False
    result = observation_db.touch_resolution_cache(0)
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
