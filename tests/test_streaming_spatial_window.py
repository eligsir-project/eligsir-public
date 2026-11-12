"""Unit tests for the spatial pose-window selection math (Part B).

Pure-function tests for ``spatial_window_weights`` — no CUDA / no training.
Verifies that near + aligned cameras are weighted higher than far / opposite
cameras, and that the radius gate and direction gate zero out the right ones.
"""
import numpy as np
import torch

from scene.streaming_scene import spatial_window_weights


def _w(centers, dirs, cur_c, cur_d, radius=1.5, sigma=0.75):
    return spatial_window_weights(
        torch.tensor(centers, dtype=torch.float64),
        torch.tensor(dirs, dtype=torch.float64),
        torch.tensor(cur_c, dtype=torch.float64),
        torch.tensor(cur_d, dtype=torch.float64),
        radius,
        sigma,
    )


def test_near_aligned_beats_far_aligned():
    # cam0 near + same dir; cam1 far (still within radius) + same dir.
    w = _w(
        centers=[[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
        dirs=[[0, 0, 1], [0, 0, 1]],
        cur_c=[0.0, 0.0, 0.0],
        cur_d=[0, 0, 1],
    )
    assert w[0] > w[1] > 0.0


def test_near_aligned_beats_near_opposite():
    # both near; cam0 looks same way, cam1 looks opposite -> cam1 zeroed by dir gate.
    w = _w(
        centers=[[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dirs=[[0, 0, 1], [0, 0, -1]],
        cur_c=[0.0, 0.0, 0.0],
        cur_d=[0, 0, 1],
    )
    assert w[0] > 0.0
    assert w[1] == 0.0


def test_outside_radius_is_zero():
    w = _w(
        centers=[[0.2, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dirs=[[0, 0, 1], [0, 0, 1]],
        cur_c=[0.0, 0.0, 0.0],
        cur_d=[0, 0, 1],
        radius=1.5,
    )
    assert w[0] > 0.0
    assert w[1] == 0.0


def test_perpendicular_is_zero():
    # orthogonal view dir -> cos clamped to 0.
    w = _w(
        centers=[[0.2, 0.0, 0.0]],
        dirs=[[1, 0, 0]],
        cur_c=[0.0, 0.0, 0.0],
        cur_d=[0, 0, 1],
    )
    assert w[0] == 0.0


def test_empty_input():
    w = spatial_window_weights(
        torch.zeros((0, 3), dtype=torch.float64),
        torch.zeros((0, 3), dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        1.5,
        0.75,
    )
    assert w.shape == (0,)


def test_non_unit_dirs_normalized():
    # scaling a dir vector must not change its weight (normalization).
    w_unit = _w([[0.2, 0, 0]], [[0, 0, 1]], [0, 0, 0], [0, 0, 1])
    w_scaled = _w([[0.2, 0, 0]], [[0, 0, 7.0]], [0, 0, 0], [0, 0, 1])
    assert torch.allclose(w_unit, w_scaled)


def test_spatial_local_cameras_does_not_crash_when_enabled():
    """Regression: enabling the spatial window must not AttributeError.

    `_spatial_local_cameras` reads `_camera_viewdirs_np`, which was dropped on
    merge (read but never assigned). It must now lazily build the host-side
    caches and return a valid (cameras, weights) selection.
    """
    from types import SimpleNamespace

    from scene.streaming_scene import StreamingScene

    class _Cam:
        def __init__(self, x):
            self.camera_center = torch.tensor([float(x), 0.0, 0.0])
            # camera-to-world R; +Z column is the optical axis (look down +Z).
            self.R = np.eye(3)

    cams = [_Cam(0.0), _Cam(0.3), _Cam(5.0)]
    s = StreamingScene.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    s._keyframe_store = KeyframeStore()
    s.train_cameras = cams
    s.current_camera = cams[0]
    s._camera_centers_np = []
    s._camera_viewdirs_np = []
    s._bootstrap_count = 0
    s.args = SimpleNamespace(
        streaming_spatial_window_enabled=True,
        streaming_spatial_window_radius=1.5,
        streaming_spatial_window_sigma=0.75,
        streaming_keyframe_window=10,
        streaming_keyframe_coverage=0.0,
        streaming_bootstrap_in_window=True,
    )

    cams_out, weights = s._spatial_local_cameras()  # must not raise
    # caches got populated index-aligned with train_cameras
    assert len(s._camera_centers_np) == 3
    assert len(s._camera_viewdirs_np) == 3
    # near cams (x=0.0, 0.3) selected; the far one (x=5.0) is outside radius.
    assert weights is not None
    assert cams[2] not in cams_out


def test_spatial_local_cameras_no_current_camera_falls_back_temporal():
    from types import SimpleNamespace

    from scene.streaming_scene import StreamingScene

    s = StreamingScene.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    s._keyframe_store = KeyframeStore()
    s.train_cameras = []
    s.current_camera = None
    s._camera_centers_np = []
    s._camera_viewdirs_np = []
    s._bootstrap_count = 0
    s.args = SimpleNamespace(streaming_keyframe_window=10, streaming_keyframe_coverage=0.0, streaming_bootstrap_in_window=True)
    cams_out, weights = s._spatial_local_cameras()
    assert weights is None  # temporal fallback


def test_recent_camera_baseline_parallax_d2():
    """D2: recent_camera_baseline returns max pairwise translation over last n+1 cams."""
    import torch
    from scene.streaming_scene import StreamingScene

    class _Cam:
        def __init__(self, x):
            self.camera_center = torch.tensor([float(x), 0.0, 0.0])

    s = StreamingScene.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    s._keyframe_store = KeyframeStore()
    s.train_cameras = [_Cam(0), _Cam(1), _Cam(2), _Cam(5)]
    assert s.recent_camera_baseline(5) == 5.0          # full span
    assert s.recent_camera_baseline(1) == 3.0          # last 2 cams: x=2,5
    s.train_cameras = [_Cam(0)]
    assert s.recent_camera_baseline(5) == 0.0          # stalled / single cam
    s.train_cameras = []
    assert s.recent_camera_baseline(5) == 0.0          # empty
