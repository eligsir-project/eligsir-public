"""Tests for the external SfM-seed bootstrap path in StreamingScene.

RealityScan Mobile scenes have no depth at all -- only RGB + poses + a sparse
SfM point cloud (`RealityScanFrameSource.seed_points`/`.seed_colors`, Task 1).
Today `_build_pcd_from_frames` only knows how to backproject depth or fall
back to a scale/structure-free random scatter; this adds a third path that
consumes that real seed cloud when the frame source provides one, taking
precedence over the random fallback.
"""

from __future__ import annotations

import inspect

import numpy as np

from scene.streaming_scene import StreamingScene, _external_seed_to_pcd


def test_external_seed_to_pcd_voxel_downsample_reduces_count():
    rng = np.random.default_rng(0)
    # Dense cluster of points inside a single 0.5m voxel cell -> collapses to 1.
    xyz = rng.uniform(0.0, 0.05, size=(200, 3)).astype(np.float32)
    rgb = rng.uniform(0.0, 1.0, size=(200, 3)).astype(np.float32)

    pcd = _external_seed_to_pcd(xyz, rgb, voxel_size=0.5, max_points=0)

    assert pcd.points.shape[0] < 200
    assert pcd.points.shape[0] >= 1


def test_external_seed_to_pcd_cap_respected():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-10.0, 10.0, size=(500, 3)).astype(np.float32)
    rgb = rng.uniform(0.0, 1.0, size=(500, 3)).astype(np.float32)

    pcd = _external_seed_to_pcd(xyz, rgb, voxel_size=0.0, max_points=100)

    assert pcd.points.shape[0] == 100


def test_external_seed_to_pcd_gray_fill_when_no_colors():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    pcd = _external_seed_to_pcd(xyz, None, voxel_size=0.0, max_points=0)

    assert pcd.colors.shape == (2, 3)
    assert np.allclose(pcd.colors, 0.5)


def test_external_seed_to_pcd_dtype_is_float32():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    rgb = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)

    pcd = _external_seed_to_pcd(xyz, rgb, voxel_size=0.0, max_points=0)

    assert pcd.points.dtype == np.float32
    assert pcd.colors.dtype == np.float32
    assert pcd.normals.dtype == np.float32


def test_external_seed_to_pcd_no_voxel_no_cap_passthrough():
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    rgb = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], dtype=np.float32)

    pcd = _external_seed_to_pcd(xyz, rgb, voxel_size=0.0, max_points=0)

    assert pcd.points.shape[0] == 3


def test_set_external_seed_precedes_insert_from_depth_early_out():
    """Source-ordering check: the external-seed branch must be inspected
    (in _build_pcd_from_frames' source text) before the
    streaming_insert_from_depth early-out, so external-seed sources take
    precedence over the random fallback that flag would otherwise trigger.

    Brittle by construction (string-offset check on source text rather than
    a real control-flow assertion) -- accepted trade-off since there is no
    cheap way to assert "branch order" without either executing both paths
    (which requires a full StreamingScene + args fixture) or parsing the AST.
    """
    src = inspect.getsource(StreamingScene._build_pcd_from_frames)
    external_seed_marker = src.find("_external_seed")
    insert_from_depth_marker = src.find("streaming_insert_from_depth")

    assert external_seed_marker != -1, "external-seed branch not found in _build_pcd_from_frames"
    assert insert_from_depth_marker != -1, "streaming_insert_from_depth early-out not found"
    assert external_seed_marker < insert_from_depth_marker
