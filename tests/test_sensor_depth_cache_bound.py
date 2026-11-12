"""Tests for bounded sensor depth cache (issue #399).

CPU-only tests validating LRU eviction of camera depth caches under
a configured frame bound. No torch.cuda or GPU operations.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from eligsir.mapping.integration import depth_helpers
from eligsir.mapping.integration.depth_helpers import (
    _enforce_sensor_depth_cache_bound,
    _record_sensor_depth_cache_load,
    _SENSOR_DEPTH_CACHE_LRU,
    _SENSOR_DEPTH_CACHE_REFS,
    set_sensor_depth_cache_bound,
)


class MockCamera:
    """Mock camera object for testing depth cache eviction."""

    def __init__(self, uid: int):
        self.uid = uid
        self._sensor_depth_cache = None


class TestSensorDepthCacheBound(unittest.TestCase):
    """Tests for sensor depth cache LRU eviction."""

    def setUp(self):
        """Clear the global LRU state before each test."""
        # Clear in correct order: LRU deque and REFS dict
        while _SENSOR_DEPTH_CACHE_LRU:
            _SENSOR_DEPTH_CACHE_LRU.popleft()
        _SENSOR_DEPTH_CACHE_REFS.clear()
        # Issue #399 follow-up: reset the module-level auto-enforcement bound
        # too. Without this, a test elsewhere in this file (or a future one)
        # that calls `set_sensor_depth_cache_bound(N)` and forgets to reset it
        # would silently leak a nonzero bound into every subsequent test's
        # calls to `_record_sensor_depth_cache_load` -- exactly the kind of
        # module-global cross-test pollution that caused the LRU/REFS
        # rebinding bug fixed alongside this test.
        set_sensor_depth_cache_bound(0)

    def tearDown(self):
        set_sensor_depth_cache_bound(0)

    def test_unbounded_mode_no_eviction(self):
        """Test that max_cached_frames=0 (unbounded) does not evict."""
        cam1 = MockCamera(uid=1)
        cam1._sensor_depth_cache = "tensor_1"
        _record_sensor_depth_cache_load(cam1)

        cam2 = MockCamera(uid=2)
        cam2._sensor_depth_cache = "tensor_2"
        _record_sensor_depth_cache_load(cam2)

        # Enforce with bound=0 (unbounded); should not evict.
        _enforce_sensor_depth_cache_bound(0)
        self.assertEqual(cam1._sensor_depth_cache, "tensor_1")
        self.assertEqual(cam2._sensor_depth_cache, "tensor_2")

    def test_bound_zero_preserves_existing_logic(self):
        """Test that bound=0 preserves today's unbounded behavior exactly."""
        cameras = [MockCamera(uid=i) for i in range(100)]
        for cam in cameras:
            cam._sensor_depth_cache = f"depth_{cam.uid}"
            _record_sensor_depth_cache_load(cam)

        _enforce_sensor_depth_cache_bound(0)
        # All caches should remain; nothing was evicted.
        for cam in cameras:
            self.assertEqual(cam._sensor_depth_cache, f"depth_{cam.uid}")

    def test_lru_eviction_under_bound(self):
        """Test that eviction does not occur when under the bound."""
        cam1 = MockCamera(uid=1)
        cam1._sensor_depth_cache = "tensor_1"
        _record_sensor_depth_cache_load(cam1)

        cam2 = MockCamera(uid=2)
        cam2._sensor_depth_cache = "tensor_2"
        _record_sensor_depth_cache_load(cam2)

        # Enforce with bound=10; both cameras are under the bound.
        _enforce_sensor_depth_cache_bound(10)
        self.assertEqual(cam1._sensor_depth_cache, "tensor_1")
        self.assertEqual(cam2._sensor_depth_cache, "tensor_2")

    def test_lru_eviction_direct(self):
        """Test LRU eviction by directly manipulating the tracking structures."""
        # This test directly manipulates the LRU to avoid pytest module import issues.
        # Create cameras and add them to the LRU manually.
        cameras = [MockCamera(uid=i) for i in range(5)]
        for cam in cameras:
            cam._sensor_depth_cache = f"depth_{cam.uid}"
            _SENSOR_DEPTH_CACHE_REFS[cam.uid] = cam
            _SENSOR_DEPTH_CACHE_LRU.append((cam.uid,))

        # Enforce with bound=3; should evict the oldest 2 caches.
        _enforce_sensor_depth_cache_bound(3)

        # The oldest two (uid 0 and 1) should be evicted (set to None).
        self.assertIsNone(cameras[0]._sensor_depth_cache)
        self.assertIsNone(cameras[1]._sensor_depth_cache)

        # The newest three (uid 2, 3, 4) should remain.
        self.assertEqual(cameras[2]._sensor_depth_cache, "depth_2")
        self.assertEqual(cameras[3]._sensor_depth_cache, "depth_3")
        self.assertEqual(cameras[4]._sensor_depth_cache, "depth_4")

    def test_false_sentinel_not_evicted(self):
        """Test that the False sentinel is never evicted."""
        cam1 = MockCamera(uid=1)
        cam1._sensor_depth_cache = "tensor_1"
        _record_sensor_depth_cache_load(cam1)

        cam2 = MockCamera(uid=2)
        cam2._sensor_depth_cache = False  # Sentinel: permanent "no depth"
        # Do not call _record_sensor_depth_cache_load for the sentinel.

        cam3 = MockCamera(uid=3)
        cam3._sensor_depth_cache = "tensor_3"
        _record_sensor_depth_cache_load(cam3)

        # Enforce with bound=1; should evict cam1 but not cam2's sentinel.
        _enforce_sensor_depth_cache_bound(1)

        self.assertIsNone(cam1._sensor_depth_cache)
        self.assertIs(cam2._sensor_depth_cache, False)  # Sentinel untouched.
        self.assertEqual(cam3._sensor_depth_cache, "tensor_3")

    def test_none_cache_not_tracked(self):
        """Test that None caches are not counted toward the bound."""
        cam1 = MockCamera(uid=1)
        cam1._sensor_depth_cache = None  # Not tracked.

        cam2 = MockCamera(uid=2)
        cam2._sensor_depth_cache = "tensor_2"
        _record_sensor_depth_cache_load(cam2)

        # Enforce with bound=1; should not evict cam2 (only one cache tracked).
        _enforce_sensor_depth_cache_bound(1)
        self.assertIsNone(cam1._sensor_depth_cache)
        self.assertEqual(cam2._sensor_depth_cache, "tensor_2")

    def test_multiple_calls_preserve_order(self):
        """Test that multiple enforcement calls respect LRU order."""
        # Create 5 cameras and manually add to tracking.
        cameras = [MockCamera(uid=i) for i in range(5)]
        for cam in cameras:
            cam._sensor_depth_cache = f"depth_{cam.uid}"
            _SENSOR_DEPTH_CACHE_REFS[cam.uid] = cam
            _SENSOR_DEPTH_CACHE_LRU.append((cam.uid,))

        # First enforcement: evict 2 oldest.
        _enforce_sensor_depth_cache_bound(3)
        self.assertIsNone(cameras[0]._sensor_depth_cache)
        self.assertIsNone(cameras[1]._sensor_depth_cache)
        self.assertEqual(cameras[2]._sensor_depth_cache, "depth_2")
        self.assertEqual(cameras[3]._sensor_depth_cache, "depth_3")
        self.assertEqual(cameras[4]._sensor_depth_cache, "depth_4")

        # Add a new camera to the tracking (most recent).
        cam_new = MockCamera(uid=10)
        cam_new._sensor_depth_cache = "depth_10"
        _SENSOR_DEPTH_CACHE_REFS[cam_new.uid] = cam_new
        _SENSOR_DEPTH_CACHE_LRU.append((cam_new.uid,))

        # Second enforcement: bound=2 should evict cam2 and cam3.
        _enforce_sensor_depth_cache_bound(2)
        self.assertIsNone(cameras[2]._sensor_depth_cache)
        self.assertIsNone(cameras[3]._sensor_depth_cache)
        self.assertEqual(cameras[4]._sensor_depth_cache, "depth_4")
        self.assertEqual(cam_new._sensor_depth_cache, "depth_10")

    def test_garbage_collected_camera_handled(self):
        """Test that garbage-collected cameras are handled gracefully."""
        cam1 = MockCamera(uid=1)
        cam1._sensor_depth_cache = "tensor_1"
        _record_sensor_depth_cache_load(cam1)

        cam2 = MockCamera(uid=2)
        cam2._sensor_depth_cache = "tensor_2"
        _record_sensor_depth_cache_load(cam2)

        # Simulate cam1 being garbage-collected by removing its reference.
        del cam1

        # Enforce with bound=0; should gracefully handle the missing camera.
        _enforce_sensor_depth_cache_bound(0)

        # cam2 should still be tracked and accessible.
        self.assertEqual(cam2._sensor_depth_cache, "tensor_2")

    def test_duplicate_tracking_prevented(self):
        """Test that the same camera is not added twice to the LRU deque."""
        cam = MockCamera(uid=1)
        cam._sensor_depth_cache = "tensor_1"

        # Record the same camera multiple times.
        _record_sensor_depth_cache_load(cam)
        _record_sensor_depth_cache_load(cam)
        _record_sensor_depth_cache_load(cam)

        # The LRU deque should have only one entry for this camera.
        # (Counting how many entries in the deque correspond to uid 1.)
        count = sum(1 for (uid,) in _SENSOR_DEPTH_CACHE_LRU if uid == 1)
        self.assertEqual(count, 1)

    def test_no_uid_camera_not_tracked(self):
        """Test that cameras without a UID are not tracked."""
        cam = MagicMock()
        cam.uid = None
        cam._sensor_depth_cache = "tensor"

        # Should not crash and should not add to the tracker.
        initial_size = len(_SENSOR_DEPTH_CACHE_LRU)
        _record_sensor_depth_cache_load(cam)
        self.assertEqual(len(_SENSOR_DEPTH_CACHE_LRU), initial_size)


class TestSensorDepthCacheAutoEnforcement(unittest.TestCase):
    """Regression coverage for the production wiring gap fixed alongside this
    test (issue #399 follow-up): the original PR defined
    `_enforce_sensor_depth_cache_bound` and the CLI flag
    (`--streaming_sensor_depth_cache_max_frames`), and 10 CPU unit tests
    verified the eviction ARITHMETIC in isolation -- but NO production code
    path anywhere in the repo ever called `_enforce_sensor_depth_cache_bound`.
    The flag was parsed into `args` and had zero runtime effect: a real
    streaming run passing a nonzero bound behaved identically to one leaving
    it at the default 0 (unbounded).

    These tests exercise the actual wiring a real run uses --
    `set_sensor_depth_cache_bound()` (called once at `train_streaming.py`
    startup) + `_record_sensor_depth_cache_load()` (called from every real
    `_get_sensor_depth`/`bind_resolution_tier` cache-populate site) -- with
    real (small) CPU tensors, and assert the property #399 exists to
    guarantee: total resident depth-cache bytes across all live cameras stays
    bounded as the number of distinct ingested frames grows, instead of
    scaling linearly with it. This is a synthetic proxy for the real
    ScanNet++ 7040-frame / 2.76-MB-per-frame GPU leak (real-GPU confirmation
    of the plateau vs. linear-growth distinction is tracked separately as
    UNMANAGED dev-verification evidence, not in this CPU-only suite -- see
    the PR/issue #399 comment thread), not a claim that it reproduces real
    GPU allocator behavior bit-for-bit.
    """

    def setUp(self):
        while _SENSOR_DEPTH_CACHE_LRU:
            _SENSOR_DEPTH_CACHE_LRU.popleft()
        _SENSOR_DEPTH_CACHE_REFS.clear()
        set_sensor_depth_cache_bound(0)

    def tearDown(self):
        set_sensor_depth_cache_bound(0)

    @staticmethod
    def _resident_bytes(cameras) -> int:
        total = 0
        for cam in cameras:
            cache = getattr(cam, "_sensor_depth_cache", None)
            if isinstance(cache, torch.Tensor):
                total += cache.numel() * cache.element_size()
        return total

    def test_bound_unset_defaults_to_zero_no_automatic_eviction(self):
        """Without ever calling `set_sensor_depth_cache_bound`, loading many
        cameras must NOT auto-evict -- this is the exact legacy/default
        behavior #399 is required to preserve when the flag is left at 0."""
        depth = torch.zeros((1, 8, 8), dtype=torch.float32)  # 256 bytes
        cameras = []
        for i in range(50):
            cam = MockCamera(uid=i)
            cam._sensor_depth_cache = depth.clone()
            _record_sensor_depth_cache_load(cam)
            cameras.append(cam)

        self.assertEqual(self._resident_bytes(cameras), 50 * depth.numel() * depth.element_size())

    def test_auto_enforcement_bounds_resident_memory_as_frames_grow(self):
        """This is the load-bearing test the PR's own 'What Remains' section
        owed: set a real bound via the production entry point
        (`set_sensor_depth_cache_bound`, exactly what `train_streaming.py`
        now calls at startup -- see the wiring added alongside this test),
        then simulate many distinct frames each populating and recording a
        real tensor via `_record_sensor_depth_cache_load` (exactly what
        `_get_sensor_depth`/`bind_resolution_tier` do on every real cache
        miss), with NO explicit call to `_enforce_sensor_depth_cache_bound`
        anywhere in this test. If the auto-enforcement wiring regresses
        (e.g. someone reverts the `if _SENSOR_DEPTH_CACHE_MAX_FRAMES:` call
        inside `_record_sensor_depth_cache_load`), resident bytes will grow
        unboundedly with frame count and this test fails."""
        bound = 20
        set_sensor_depth_cache_bound(bound)
        depth = torch.zeros((1, 16, 16), dtype=torch.float32)  # 1024 bytes
        per_frame_bytes = depth.numel() * depth.element_size()

        cameras = []
        n_frames = 200  # 10x the bound -- unbounded growth would be obvious.
        for i in range(n_frames):
            cam = MockCamera(uid=i)
            cam._sensor_depth_cache = depth.clone()
            _record_sensor_depth_cache_load(cam)
            cameras.append(cam)

            # Plateau check partway through: by the time we've loaded 2x the
            # bound, resident bytes must already have stopped growing
            # linearly with i -- this is the "plateau, not linear growth"
            # distinction the GPU verification looks for at real scale.
            if i == 2 * bound:
                resident_at_2x_bound = self._resident_bytes(cameras)
                self.assertLessEqual(resident_at_2x_bound, bound * per_frame_bytes)

        final_resident = self._resident_bytes(cameras)
        self.assertLessEqual(
            final_resident,
            bound * per_frame_bytes,
            msg=(
                f"resident sensor-depth-cache bytes ({final_resident}) exceeded "
                f"the configured bound ({bound} frames x {per_frame_bytes} bytes = "
                f"{bound * per_frame_bytes} bytes) after loading {n_frames} distinct "
                "frames -- auto-enforcement is not bounding resident memory."
            ),
        )
        # And it must be UNDER the naive unbounded total, i.e. real eviction
        # happened, not just "coincidentally small".
        self.assertLess(final_resident, n_frames * per_frame_bytes)

        # Exactly the most-recently-loaded `bound` cameras should be the ones
        # still resident (LRU eviction order), oldest-evicted.
        resident_uids = {
            cam.uid for cam in cameras if isinstance(cam._sensor_depth_cache, torch.Tensor)
        }
        expected_resident_uids = set(range(n_frames - bound, n_frames))
        self.assertEqual(resident_uids, expected_resident_uids)

    def test_disabling_bound_mid_run_stops_future_eviction(self):
        """`set_sensor_depth_cache_bound(0)` must disable enforcement for all
        SUBSEQUENT loads (already-evicted cameras stay evicted -- there is no
        un-eviction), matching a run that never set the flag in the first
        place from that point on."""
        depth = torch.zeros((1, 4, 4), dtype=torch.float32)
        set_sensor_depth_cache_bound(3)
        cameras = []
        for i in range(5):
            cam = MockCamera(uid=i)
            cam._sensor_depth_cache = depth.clone()
            _record_sensor_depth_cache_load(cam)
            cameras.append(cam)
        # Bound of 3 -> only the last 3 (uid 2,3,4) resident.
        self.assertEqual(self._resident_bytes(cameras), 3 * depth.numel() * depth.element_size())

        set_sensor_depth_cache_bound(0)
        for i in range(5, 15):
            cam = MockCamera(uid=i)
            cam._sensor_depth_cache = depth.clone()
            _record_sensor_depth_cache_load(cam)
            cameras.append(cam)
        # The 10 newly-loaded cameras all stay resident (bound disabled);
        # the 2 evicted-before-disable cameras (uid 0, 1) stay evicted.
        self.assertEqual(
            self._resident_bytes(cameras),
            (3 + 10) * depth.numel() * depth.element_size(),
        )


if __name__ == "__main__":
    unittest.main()
