"""Transport-neutral holdout selection by stable source ordinal (#745)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scene.streaming_scene import StreamingScene


class _FakeFrame:
    def __init__(self, index: int, source_ordinal=None) -> None:
        self.index = index
        self._odom_valid = True
        self.timestamp = float(index)
        self.source_ordinal = source_ordinal


class _FakeCam:
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
    """Minimal stand-in exercising the real holdout predicate and ingest path."""

    def __init__(self, frames, eval_hold: int):
        self.args = SimpleNamespace(
            streaming_eval_hold=eval_hold,
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
        return None

    _append_dense_trajectory = StreamingScene._append_dense_trajectory
    dense_trajectory_snapshot = StreamingScene.dense_trajectory_snapshot
    _cache_inactive_frame = StreamingScene._cache_inactive_frame
    _frame_carries_pixel_data = staticmethod(StreamingScene._frame_carries_pixel_data)
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    holdout_status = StreamingScene.holdout_status


def _run_all(fake_self, monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )
    for _ in fake_self._frames:
        StreamingScene.ingest_next_frame(
            fake_self, admission_fn=lambda cam, frame: (True, {}, None)
        )


def test_is_holdout_prefers_source_ordinal_over_index(monkeypatch):
    frame = _FakeFrame(index=5, source_ordinal=8)
    fake_self = _FakeIngestSelf([frame], eval_hold=4)
    _run_all(fake_self, monkeypatch)

    assert len(fake_self._test_cameras) == 1
    assert fake_self._dense_trajectory[0]["is_holdout"] is True


def test_is_holdout_falls_back_to_index_when_source_ordinal_absent(monkeypatch):
    frames = [_FakeFrame(index=i, source_ordinal=None) for i in range(9)]
    fake_self = _FakeIngestSelf(frames, eval_hold=4)
    _run_all(fake_self, monkeypatch)

    holdout_indices = sorted(
        frame.index
        for frame in frames
        if any(
            camera.image_name == f"frame_{frame.index}"
            for camera in fake_self._test_cameras
        )
    )
    assert holdout_indices == [0, 4, 8]


def test_drop_shifted_index_does_not_reshuffle_holdout_membership(monkeypatch):
    shifted_holdout = _FakeFrame(index=5, source_ordinal=8)
    shifted_non_holdout = _FakeFrame(index=4, source_ordinal=6)
    fake_self = _FakeIngestSelf(
        [shifted_holdout, shifted_non_holdout], eval_hold=4
    )
    _run_all(fake_self, monkeypatch)

    assert [entry["is_holdout"] for entry in fake_self._dense_trajectory] == [
        True,
        False,
    ]
