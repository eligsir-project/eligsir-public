"""Regression tests for StreamingScene's dense (full, unfiltered) trajectory
recording -- Part A of the render-report plan (docs/superpowers/plans render
report). Every frame that reaches a real Camera (passed the pose gate) must be
recorded in ``_dense_trajectory``, regardless of keyframe-admission outcome,
so a post-hoc renderer can play back the full trajectory, not just the
admitted keyframes kept in ``train_cameras``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scene.streaming_scene import StreamingScene


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True
        self.timestamp = float(index)


class _FakeCam:
    """Minimal cam with everything camera_to_JSON needs."""

    def __init__(self, index: int) -> None:
        self.R = np.eye(3, dtype=np.float32)
        self.T = np.zeros(3, dtype=np.float32)
        self.image_name = f"frame_{index}"
        self.width = 64
        self.height = 64
        self.fx = self.fy = 50.0
        self.cx = self.cy = 32.0
        self.c2w = np.eye(4, dtype=np.float32)

    def materialize_pending_image(self):
        return None


class _FakeIngestSelf:
    """Minimal StreamingScene stand-in, mirroring the harness in
    tests/test_observation_db_ingestion_wiring.py, extended with the
    dense-trajectory buffer and the real bound helper method."""

    def __init__(self, frames):
        self.args = SimpleNamespace(
            streaming_eval_hold=0,
            streaming_offload_camera_images=False,
            streaming_replay_buffer=512,
            streaming_frame_admission="all",
            streaming_global_reservoir_stride=0,
            streaming_keyframe_window=8,
            streaming_track_scene_complexity=False,
            streaming_inactive_cache_size=0,
        )
        self._active_source_indices = set()
        self._active_cam_by_idx = {}
        self._last_admitted_source_idx = None
        self._frames = frames
        self._source_idx = 0
        self._keyframes_rejected = 0
        self._keyframes_admitted = 0
        self._holdout_indices = None
        self._guard_indices = set()
        self._test_cameras = []
        self.train_cameras = []
        self._replay_buffer = []
        self._n_train_frames_ingested = 0
        self._global_reservoir = []
        self._steps_since_new_frame = 0
        self._pending_extent_update = False
        self.current_camera = None
        self._dense_trajectory = []
        self._dense_trajectory_flushed = 0

    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def accumulate_observed_surface(self, frame):
        return None

    def _frame_to_camera(self, frame, load_image=True):
        return _FakeCam(frame.index)

    def update_cameras_extent(self):
        return None

    def _release_source_frames_upto_cursor(self):
        # WS-D1's live-frame-buffer eviction hook; irrelevant to this
        # test's dense-trajectory bookkeeping, no-op like the other stubs.
        return None

    # Bind the real implementations (unbound methods) so the test exercises
    # the actual production logic, not a re-implementation.
    _append_dense_trajectory = StreamingScene._append_dense_trajectory
    dense_trajectory_snapshot = StreamingScene.dense_trajectory_snapshot
    _cache_inactive_frame = StreamingScene._cache_inactive_frame
    _frame_carries_pixel_data = staticmethod(StreamingScene._frame_carries_pixel_data)
    # Issue #396: the hot-window eviction cursor, extracted out of
    # ingest_next_frame's inline body into its own method.
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: bound from the REAL StreamingScene -- #1084 extracted
    # `ingest_next_frame`'s inline holdout/guard computation into this
    # predicate verbatim. Tests here set `_holdout_indices` to {0}/{2} to
    # assert the dense-trajectory `is_holdout` flag, so it must be the
    # production rule, not a stub.
    holdout_status = StreamingScene.holdout_status


def _run_ingest(fake_self, admission_fn=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(
            "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
        )
    return StreamingScene.ingest_next_frame(fake_self, admission_fn=admission_fn)


def test_admitted_frame_recorded_with_admitted_true(monkeypatch):
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    _run_ingest(fake_self, admission_fn=lambda cam, frame: (True, {}, None), monkeypatch=monkeypatch)

    assert len(fake_self._dense_trajectory) == 1
    entry = fake_self._dense_trajectory[0]
    assert entry["admitted"] is True
    assert entry["is_holdout"] is False


def test_admission_rejected_frame_still_recorded_with_admitted_false(monkeypatch):
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    _run_ingest(fake_self, admission_fn=lambda cam, frame: (False, {}, None), monkeypatch=monkeypatch)

    assert len(fake_self._dense_trajectory) == 1
    entry = fake_self._dense_trajectory[0]
    assert entry["admitted"] is False
    assert entry["is_holdout"] is False
    # The camera itself was never kept as a training view.
    assert fake_self.train_cameras == []


def test_holdout_frame_recorded_with_is_holdout_true_and_admitted_false(monkeypatch):
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    fake_self._holdout_indices = {0}
    _run_ingest(fake_self, admission_fn=lambda cam, frame: (True, {}, None), monkeypatch=monkeypatch)

    assert len(fake_self._dense_trajectory) == 1
    entry = fake_self._dense_trajectory[0]
    assert entry["admitted"] is False
    assert entry["is_holdout"] is True


def test_one_entry_per_ingested_frame_across_a_run(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    frames = [_FakeFrame(i) for i in range(6)]
    fake_self = _FakeIngestSelf(frames)
    fake_self._holdout_indices = {2}

    # Alternate admit/reject for non-holdout frames.
    def admission_fn(cam, frame):
        return (frame.index % 2 == 0), {}, None

    for _ in range(len(frames)):
        StreamingScene.ingest_next_frame(fake_self, admission_fn=admission_fn)

    assert len(fake_self._dense_trajectory) == len(frames)
    assert [e["id"] for e in fake_self._dense_trajectory] == list(range(len(frames)))
    holdout_flags = [e["is_holdout"] for e in fake_self._dense_trajectory]
    assert holdout_flags == [False, False, True, False, False, False]


def test_dense_trajectory_snapshot_returns_only_new_entries_since_offset(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    frames = [_FakeFrame(i) for i in range(3)]
    fake_self = _FakeIngestSelf(frames)
    for _ in range(3):
        StreamingScene.ingest_next_frame(fake_self, admission_fn=lambda cam, frame: (True, {}, None))

    first_batch = fake_self.dense_trajectory_snapshot(since=0)
    assert len(first_batch) == 3
    fake_self._dense_trajectory_flushed = len(fake_self._dense_trajectory)
    second_batch = fake_self.dense_trajectory_snapshot(since=fake_self._dense_trajectory_flushed)
    assert second_batch == []
