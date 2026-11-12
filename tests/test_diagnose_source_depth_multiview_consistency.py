import numpy as np
import pytest

pytest.importorskip("PIL")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_source_depth_multiview_consistency import (
    _backproject_grid,
    _project_and_check,
)


def _identity_c2w():
    return np.eye(4, dtype=np.float32)


def test_backproject_grid_roundtrips_center_pixel():
    h, w = 64, 64
    depth = np.full((h, w), 2.0, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    pts, px = _backproject_grid(depth, fx, fy, cx, cy, _identity_c2w(), stride=32,
                                 min_depth=0.1, max_depth=8.0, edge_thresh=0.0)
    assert pts.shape[0] > 0
    # center pixel (32,32) backprojects to (0,0,2) in camera==world frame here
    center_idx = np.argmin(np.abs(px[:, 0] - 32) + np.abs(px[:, 1] - 32))
    np.testing.assert_allclose(pts[center_idx], [0.0, 0.0, 2.0], atol=1e-4)


def test_backproject_grid_excludes_depth_edges():
    h, w = 32, 32
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[:, 16:] = 5.0  # sharp depth discontinuity at column 16
    fx = fy = 50.0
    cx, cy = 16.0, 16.0
    pts_no_edge, px_no_edge = _backproject_grid(
        depth, fx, fy, cx, cy, _identity_c2w(), stride=4,
        min_depth=0.1, max_depth=8.0, edge_thresh=0.5,
    )
    pts_with_edge, px_with_edge = _backproject_grid(
        depth, fx, fy, cx, cy, _identity_c2w(), stride=4,
        min_depth=0.1, max_depth=8.0, edge_thresh=0.0,
    )
    # Edge filtering must strictly reduce the accepted point count near the
    # discontinuity (column 16 boundary pixels get excluded).
    assert pts_no_edge.shape[0] < pts_with_edge.shape[0]


def test_project_and_check_supported_when_matching_depth():
    # Point at (0, 0, 2) in world == camera frame (identity pose).
    pts_world = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    width = height = 64
    sensor_depth = np.full((height, width), 2.0, dtype=np.float32)

    support, conflict, residual = _project_and_check(
        pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
        consistency_thresh=0.05, conflict_thresh=0.05,
    )
    assert support[0]
    assert not conflict[0]
    assert residual[0] == pytest.approx(0.0, abs=1e-4)


def test_project_and_check_conflict_when_point_closer_than_measured_surface():
    # Point claims z=2.0 (distance from camera), but the sensor at that
    # pixel measures the real surface FARTHER away (z=5.0) -- the sensor's
    # ray passed straight through where the point claims to exist (2.0)
    # without detecting anything until 5.0, i.e. valid depth says the
    # intervening space is empty. This must be CONFLICT, matching
    # frame_preparation.py's exact production formula:
    # z_p < z_lookup - conflict_thresh  =>  2.0 < 5.0 - 0.05  =>  True.
    pts_world = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    width = height = 64
    sensor_depth = np.full((height, width), 5.0, dtype=np.float32)

    support, conflict, residual = _project_and_check(
        pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
        consistency_thresh=0.05, conflict_thresh=0.05,
    )
    assert not support[0]
    assert conflict[0]


def test_project_and_check_no_conflict_when_point_farther_than_measured_surface():
    # Point claims z=2.0, sensor measures the surface at that pixel CLOSER
    # (z=1.0) -- the point is occluded/behind the real surface, which is
    # plausible (not proof of wrong geometry the way "space in front reads
    # empty" is). Must NOT be flagged as conflict -- the production
    # formula's asymmetric design only flags z_p < z_lookup - thresh, never
    # z_p > z_lookup + thresh.
    pts_world = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    width = height = 64
    sensor_depth = np.full((height, width), 1.0, dtype=np.float32)

    support, conflict, residual = _project_and_check(
        pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
        consistency_thresh=0.05, conflict_thresh=0.05,
    )
    assert not support[0]
    assert not conflict[0]


def test_project_and_check_out_of_frame_gives_no_signal():
    pts_world = np.array([[100.0, 100.0, 2.0]], dtype=np.float32)  # projects way outside
    w2c = np.eye(4, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    width = height = 64
    sensor_depth = np.full((height, width), 2.0, dtype=np.float32)

    support, conflict, residual = _project_and_check(
        pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
        consistency_thresh=0.05, conflict_thresh=0.05,
    )
    assert not support[0]
    assert not conflict[0]
    assert np.isnan(residual[0])


def test_project_and_check_invalid_sensor_reading_gives_no_signal():
    pts_world = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    fx = fy = 50.0
    cx, cy = 32.0, 32.0
    width = height = 64
    sensor_depth = np.zeros((height, width), dtype=np.float32)  # all invalid

    support, conflict, residual = _project_and_check(
        pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
        consistency_thresh=0.05, conflict_thresh=0.05,
    )
    assert not support[0]
    assert not conflict[0]
    assert np.isnan(residual[0])
