"""Regression tests for StreamingScene's partition-focus training-sampling
bias (--streaming_partition_focus_weight): once a set of source frame
indices is flagged via set_priority_frame_indices, sampling should draw
from cameras in that set with the configured probability, and fall back to
normal sampling otherwise (including when the flagged set is empty, or
intersects none of the current train cameras).
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch

from scene.streaming_scene import StreamingScene


class _FakeCam:
    def __init__(self, index: int, frame=None) -> None:
        self.image_name = f"frame_{index}"
        self._is_bootstrap = False
        self._streaming_frame = frame
        # Real tensor (not a GPU one -- this test suite is CPU-only) so
        # scene.cameras.offload_camera_image's real logic (device-type
        # check + .to("cpu")) actually exercises something meaningful
        # instead of a no-op on a missing attribute.
        self.original_image = torch.zeros(3, 2, 2)


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._rgb_np = None
        self._depth_np = None
        self._rgb_bytes = None
        self._depth_bytes = None


class _FakeLiveFrame(_FakeFrame):
    """Stand-in for a ros2_live-sourced StreamingRGBDFrame: carries decoded
    pixels (no seekable file to re-decode from), so caching must use the
    smaller --streaming_live_*_cache_size bounds instead of the file-backed
    (metadata-only) ones."""

    def __init__(self, index: int) -> None:
        super().__init__(index)
        self._rgb_np = object()
        self._depth_np = object()


def _index_of(cam) -> int:
    return int(cam.image_name.split("_")[1])


class _FakeObservationDB:
    """Minimal FrameObservationDB stand-in: just the stage-tracking surface
    StreamingScene's tiered cache actually calls."""

    def __init__(self, known_ids):
        self.stages: dict = {i: "recorded" for i in known_ids}

    def set_buffer_stage(self, frame_id, stage) -> bool:
        if frame_id not in self.stages:
            return False
        self.stages[frame_id] = stage
        return True


class _FakeSamplingSelf:
    """Minimal StreamingScene stand-in exercising the real
    sample_training_camera / set_priority_frame_indices implementations."""

    def __init__(self, train_cameras, partition_focus_weight: float):
        self.args = SimpleNamespace(
            streaming_new_frame_warmup_steps=0,
            streaming_consolidation_sampler="off",
            streaming_partition_focus_weight=partition_focus_weight,
            streaming_residual_replay_weight=0.0,
            streaming_sampling_mode="legacy",
            streaming_keyframe_sampling_weight=2.0,
            streaming_global_replay_ratio=0.0,
        )
        self.train_cameras = train_cameras
        self._priority_frame_indices: set = set()
        self._priority_frame_weights: dict = {}
        self._steps_since_new_frame = 0
        self.current_camera = train_cameras[0] if train_cameras else None
        self._replay_buffer = []
        self._global_reservoir = []
        self.args.streaming_keyframe_window = len(train_cameras)
        self.args.streaming_keyframe_coverage = 0.0
        self.args.streaming_inactive_cache_size = 2000
        self.args.streaming_cold_window = -1  # off by default in tests (no windowing)
        self.args.streaming_icecold_cache_size = 20000
        self.args.streaming_live_inactive_cache_size = 150
        self.args.streaming_live_icecold_cache_size = 1500
        self._bootstrap_count = 0
        self._inactive_frame_cache: "OrderedDict" = OrderedDict()
        self._icecold_frame_cache: "OrderedDict" = OrderedDict()
        self._last_admitted_source_idx = None
        self._active_cam_by_idx = {
            _index_of(c): c for c in train_cameras if _index_of(c) is not None
        }
        self._active_source_indices = set(self._active_cam_by_idx.keys())
        self.observation_db = None

    def _frame_to_camera(self, frame, load_image=True):
        return _FakeCam(frame.index, frame=frame)

    set_priority_frame_indices = StreamingScene.set_priority_frame_indices
    set_priority_frame_weights = StreamingScene.set_priority_frame_weights
    set_observation_db = StreamingScene.set_observation_db
    _mark_buffer_stage = StreamingScene._mark_buffer_stage
    promote_from_cache = StreamingScene.promote_from_cache
    # issue #481(b): promote_from_cache rebuilds a promoted frame's multires
    # resolution_cache through this helper (no-op without a builder / DB).
    _attach_promoted_resolution_cache = StreamingScene._attach_promoted_resolution_cache
    _predecode_frames_parallel = StreamingScene._predecode_frames_parallel
    _enforce_hot_window = StreamingScene._enforce_hot_window
    # Issue #396: the shared raw-keyframe-window boundary helper that
    # _enforce_hot_window / _offload_evicted_camera / hot_camera_ids all route
    # through, so none of them can disagree on what "hot" means.
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    cached_indices_in_partitions = StreamingScene.cached_indices_in_partitions
    demote_partitions = StreamingScene.demote_partitions
    _cache_inactive_frame = StreamingScene._cache_inactive_frame
    _archive_to_icecold = StreamingScene._archive_to_icecold
    _frame_carries_pixel_data = staticmethod(StreamingScene._frame_carries_pixel_data)
    sample_training_camera = StreamingScene.sample_training_camera
    _sample_legacy = StreamingScene._sample_legacy
    _sample_local = StreamingScene._sample_local
    _sample_global_with_debt = StreamingScene._sample_global_with_debt
    _weighted_choice = staticmethod(StreamingScene._weighted_choice)
    _weighted_choice_weighted = staticmethod(StreamingScene._weighted_choice_weighted)
    get_local_cameras = StreamingScene.get_local_cameras
    _effective_keyframe_window = StreamingScene._effective_keyframe_window


def test_zero_weight_never_uses_priority_set():
    cams = [_FakeCam(i) for i in range(10)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.set_priority_frame_indices({0, 1})
    for _ in range(20):
        cam = fake.sample_training_camera()
        assert cam is not None  # feature off -> normal sampler runs, never errors


def test_weight_one_always_samples_from_priority_set_when_nonempty():
    cams = [_FakeCam(i) for i in range(10)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    fake.set_priority_frame_indices({3, 7})
    for _ in range(30):
        cam = fake.sample_training_camera()
        idx = int(cam.image_name.split("_")[1])
        assert idx in {3, 7}


def test_empty_priority_set_falls_back_to_normal_sampling():
    cams = [_FakeCam(i) for i in range(5)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    # No set_priority_frame_indices call -- starts empty.
    cam = fake.sample_training_camera()
    assert cam is not None  # must not crash / must not hang picking from nothing


def test_promote_from_cache_splices_cached_frame_into_train_cameras():
    cams = [_FakeCam(i) for i in range(3)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake._cache_inactive_frame(_FakeFrame(99))
    assert 99 in fake._inactive_frame_cache
    promoted = fake.promote_from_cache({99})
    assert promoted == 1
    assert 99 not in fake._inactive_frame_cache
    assert 99 in fake._active_source_indices
    assert any(_index_of(c) == 99 for c in fake.train_cameras)


def test_promote_from_cache_skips_already_active_and_missing():
    cams = [_FakeCam(0)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    # index 0 is already active (in train_cameras) -- must not be re-promoted
    # or duplicated even though it's not in the cache.
    promoted = fake.promote_from_cache({0, 42})  # 42 not cached at all
    assert promoted == 0
    assert len(fake.train_cameras) == 1


def test_cache_is_bounded_lru():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_inactive_cache_size = 2
    for i in range(5):
        fake._cache_inactive_frame(_FakeFrame(i))
    assert len(fake._inactive_frame_cache) == 2
    # Only the two most recently cached survive.
    assert set(fake._inactive_frame_cache.keys()) == {3, 4}


def test_pixel_carrying_frame_uses_live_cache_size_not_file_backed_size():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_cold_window = -1
    fake.args.streaming_inactive_cache_size = 2000  # would NOT bound this if used
    fake.args.streaming_live_inactive_cache_size = 2
    for i in range(5):
        fake._cache_inactive_frame(_FakeLiveFrame(i))
    assert len(fake._inactive_frame_cache) == 2
    assert set(fake._inactive_frame_cache.keys()) == {3, 4}


def test_pixel_carrying_frame_uses_live_icecold_size_not_file_backed_size():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_icecold_cache_size = 20000  # would NOT bound this if used
    fake.args.streaming_live_icecold_cache_size = 2
    for i in range(5):
        fake._archive_to_icecold(i, _FakeLiveFrame(i))
    assert len(fake._icecold_frame_cache) == 2
    assert set(fake._icecold_frame_cache.keys()) == {3, 4}


def test_metadata_only_frame_still_uses_file_backed_cache_size():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_cold_window = -1
    fake.args.streaming_inactive_cache_size = 2
    fake.args.streaming_live_inactive_cache_size = 2000  # would NOT bound this if used
    for i in range(5):
        fake._cache_inactive_frame(_FakeFrame(i))
    assert len(fake._inactive_frame_cache) == 2
    assert set(fake._inactive_frame_cache.keys()) == {3, 4}


def test_set_priority_frame_indices_promotes_from_cache_automatically():
    cams = [_FakeCam(i) for i in range(3)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    fake._cache_inactive_frame(_FakeFrame(50))
    fake.set_priority_frame_indices({50})
    assert any(_index_of(c) == 50 for c in fake.train_cameras)
    for _ in range(10):
        cam = fake.sample_training_camera()
        assert _index_of(cam) == 50


def _cams_with_frames(n):
    frames = [_FakeFrame(i) for i in range(n)]
    return [_FakeCam(i, frame=frames[i]) for i in range(n)]


def test_demote_partitions_moves_active_camera_to_icecold_tier():
    cams = _cams_with_frames(5)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = None  # don't accidentally protect index 3 via current_camera
    demoted = fake.demote_partitions({3})
    assert demoted == 1
    assert 3 not in fake._active_source_indices
    assert 3 in fake._icecold_frame_cache
    assert not any(_index_of(c) == 3 for c in fake.train_cameras)


def test_demote_partitions_never_demotes_bootstrap_or_current():
    cams = _cams_with_frames(3)
    cams[0]._is_bootstrap = True
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = cams[1]
    demoted = fake.demote_partitions({0, 1, 2})
    assert demoted == 1  # only index 2 is eligible
    assert 2 in fake._icecold_frame_cache
    assert 0 in fake._active_source_indices
    assert 1 in fake._active_source_indices


def test_demote_partitions_skips_camera_with_no_source_frame():
    cams = [_FakeCam(0)]  # no _streaming_frame -- nothing to archive to
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = None
    demoted = fake.demote_partitions({0})
    assert demoted == 0
    assert 0 in fake._active_source_indices  # left active rather than lost


def test_demoted_frame_is_re_promotable_from_icecold_tier():
    cams = _cams_with_frames(3)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = None
    fake.demote_partitions({1})
    assert 1 not in fake._active_source_indices
    promoted = fake.promote_from_cache({1})
    assert promoted == 1
    assert 1 in fake._active_source_indices
    assert any(_index_of(c) == 1 for c in fake.train_cameras)


def test_icecold_cache_is_bounded_lru():
    cams = _cams_with_frames(5)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = None
    fake.args.streaming_icecold_cache_size = 2
    fake.demote_partitions({0, 1, 2})
    assert len(fake._icecold_frame_cache) == 2
    assert set(fake._icecold_frame_cache.keys()) == {1, 2}


def test_cold_tier_overflow_spills_into_icecold_not_discarded():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_inactive_cache_size = 2
    for i in range(5):
        fake._cache_inactive_frame(_FakeFrame(i))
    assert set(fake._inactive_frame_cache.keys()) == {3, 4}
    # The 3 evicted-from-cold entries (0, 1, 2) must still be reachable
    # via the icecold tier, not lost.
    assert set(fake._icecold_frame_cache.keys()) == {0, 1, 2}
    promoted = fake.promote_from_cache({0})
    assert promoted == 1


def test_no_observation_db_is_a_safe_noop():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    assert fake.observation_db is None
    fake._cache_inactive_frame(_FakeFrame(1))  # must not raise with no DB attached
    fake._archive_to_icecold(2, _FakeFrame(2))


def test_observation_db_tracks_cold_and_icecold_transitions():
    from eligsir.mapping.observation_db import BufferStage

    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    db = _FakeObservationDB(known_ids=range(10))
    fake.set_observation_db(db)

    fake._cache_inactive_frame(_FakeFrame(5))
    assert db.stages[5] == BufferStage.COLD_CACHE

    fake._archive_to_icecold(5, _FakeFrame(5))
    assert db.stages[5] == BufferStage.ICECOLD_CACHE

    fake._inactive_frame_cache[5] = _FakeFrame(5)  # simulate it being in cold tier
    fake.promote_from_cache({5})
    assert db.stages[5] == BufferStage.MAPPING_KEYFRAME


def test_observation_db_tracks_demotion():
    from eligsir.mapping.observation_db import BufferStage

    cams = _cams_with_frames(3)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.current_camera = None
    db = _FakeObservationDB(known_ids=range(3))
    fake.set_observation_db(db)

    fake.demote_partitions({1})
    assert db.stages[1] == BufferStage.ICECOLD_CACHE


def test_observation_db_unknown_frame_id_is_safe_noop():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    db = _FakeObservationDB(known_ids=[])  # frame 7 never recorded (e.g. pregate-rejected)
    fake.set_observation_db(db)
    fake._cache_inactive_frame(_FakeFrame(7))  # must not raise
    assert 7 not in db.stages


def test_cached_indices_in_partitions_finds_cold_and_icecold_candidates():
    from eligsir.utils.partition_eval import partition_id

    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake._inactive_frame_cache[10] = _FakeFrame(10)
    fake._icecold_frame_cache[500] = _FakeFrame(500)
    fake._icecold_frame_cache[900] = _FakeFrame(900)  # different partition -- must be excluded
    max_src, k = 999, 6
    target_pid = partition_id(10, max_src, k=k)
    other_pid = partition_id(900, max_src, k=k)
    assert other_pid != target_pid  # sanity: fixture actually separates partitions
    found = fake.cached_indices_in_partitions({target_pid}, max_src, k)
    assert 10 in found
    assert 900 not in found


def test_cached_indices_in_partitions_empty_when_no_match():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake._inactive_frame_cache[10] = _FakeFrame(10)
    found = fake.cached_indices_in_partitions({999}, 999, 6)  # partition that can't exist
    assert found == set()


def test_bad_partition_pulls_new_frames_from_cache_end_to_end():
    """Regression for the real bug found 2026-07-27: a priority-weight dict
    built only from already-active train_cameras is a silent no-op for
    promote_from_cache (which skips anything already active), so a thin
    partition never actually gains new frames. This models the fixed
    train_streaming.py call-site pattern: union active-partition cameras
    with cached_indices_in_partitions() before calling
    set_priority_frame_weights."""
    cams = [_FakeCam(i) for i in range(3)]  # sparse active set for this partition
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    fake._cache_inactive_frame(_FakeFrame(1))  # would-be neighbor, currently rejected/cached
    assert 1 in fake._inactive_frame_cache

    bad_pids = {0}  # everything lands in partition 0 with the tiny max_src below
    max_src, k = 5, 1
    cached_idx = fake.cached_indices_in_partitions(bad_pids, max_src, k)
    assert 1 in cached_idx

    weights = {0: 2.0, 2: 2.0}
    for idx in cached_idx:
        weights.setdefault(idx, 2.0)
    fake.set_priority_frame_weights(weights)
    assert 1 in fake._active_source_indices  # promoted, not left stranded in cache
    assert any(_index_of(c) == 1 for c in fake.train_cameras)


def test_enforce_hot_window_offloads_everything_but_the_tail():
    from unittest.mock import patch

    cams = [_FakeCam(i) for i in range(10)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.args.streaming_keyframe_window = 3
    with patch("scene.cameras.offload_camera_image") as mock_offload:
        fake._enforce_hot_window()
    offloaded = [c.args[0] for c in mock_offload.call_args_list]
    assert offloaded == cams[:-3]  # everything except the last 3 (hot window)


def test_enforce_hot_window_noop_when_disabled():
    from unittest.mock import patch

    cams = [_FakeCam(i) for i in range(10)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.args.streaming_keyframe_window = 3
    fake.args.streaming_offload_camera_images = False
    with patch("scene.cameras.offload_camera_image") as mock_offload:
        fake._enforce_hot_window()
    mock_offload.assert_not_called()


def test_promote_from_cache_triggers_hot_window_sweep():
    """Regression for the 2026-07-27 live OOM: promote_from_cache appending
    full-res GPU-resident cameras with no eviction let images accumulate
    unbounded (baseline safely reached N=300k+; this mechanism OOM'd at a
    LOWER N=261743 on the same config, from GPU image memory alone)."""
    from unittest.mock import patch

    cams = _cams_with_frames(5)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.args.streaming_keyframe_window = 2
    fake._cache_inactive_frame(_FakeFrame(50))
    with patch("scene.cameras.offload_camera_image") as mock_offload:
        promoted = fake.promote_from_cache({50})
    assert promoted == 1
    mock_offload.assert_called()  # the sweep ran, not just a silent append


def test_demote_partitions_triggers_hot_window_sweep():
    """Regression: list.remove() in demote_partitions shifts later indices
    down by one, desyncing the admission path's position-based eviction
    counter -- must trigger a full resync sweep."""
    from unittest.mock import patch

    cams = _cams_with_frames(6)
    fake = _FakeSamplingSelf(cams, partition_focus_weight=0.0)
    fake.args.streaming_keyframe_window = 2
    fake.current_camera = None
    with patch("scene.cameras.offload_camera_image") as mock_offload:
        demoted = fake.demote_partitions({3})
    assert demoted == 1
    mock_offload.assert_called()


def test_priority_weights_bias_sampling_toward_higher_weight():
    cams = [_FakeCam(i) for i in range(2)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    fake.set_priority_frame_weights({0: 1.0, 1: 99.0})
    counts = {0: 0, 1: 0}
    for _ in range(500):
        cam = fake.sample_training_camera()
        counts[_index_of(cam)] += 1
    assert counts[1] > counts[0] * 5  # heavily weighted index dominates


def test_cold_window_scopes_cache_to_neighbors_of_last_admitted_keyframe():
    fake = _FakeSamplingSelf([], partition_focus_weight=0.0)
    fake.args.streaming_cold_window = 2
    fake._last_admitted_source_idx = 100
    fake._cache_inactive_frame(_FakeFrame(101))  # within window -> cached
    fake._cache_inactive_frame(_FakeFrame(150))  # far away -> not cached
    assert 101 in fake._inactive_frame_cache
    assert 150 not in fake._inactive_frame_cache


def test_set_priority_frame_indices_replaces_not_accumulates():
    cams = [_FakeCam(i) for i in range(5)]
    fake = _FakeSamplingSelf(cams, partition_focus_weight=1.0)
    fake.set_priority_frame_indices({0, 1})
    fake.set_priority_frame_indices({4})
    for _ in range(20):
        cam = fake.sample_training_camera()
        idx = int(cam.image_name.split("_")[1])
        assert idx == 4  # old flagged indices {0, 1} must be gone, not merged
