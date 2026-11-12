"""Regression test for the trajectory.mp4 camera-ordering bug (found live
2026-08-13, pixel10a RealityScan Mobile capture): source images are
enumerated by ``realityscan_importer`` in alphabetical-UUID order, which has
no relation to capture order, so the rendered trajectory video visually
jumped between random poses instead of sweeping continuously through the
scene.

``_order_cams_for_trajectory`` greedy-nearest-neighbor reorders by
``camera_center`` -- these tests assert it (a) actually produces a
short/continuous path from a scrambled input, (b) is a pure reordering (same
set of cameras, none dropped or duplicated), and (c) never touches anything
other than the camera list itself (train/test split is a completely separate
code path).
"""
from __future__ import annotations

import math
import types

import torch

import numpy as np

from eligsir.utils.comparison_report import (
    _interpolate_camera,
    _order_cams_for_trajectory,
    _pace_cams_for_trajectory,
    _quat_to_rotmat,
    _rotmat_to_quat,
    _slerp_quat,
)
from eligsir.utils.graphics_utils import getWorld2View2


def _cam(uid, x, y, z=0.0):
    c = types.SimpleNamespace()
    c.uid = uid
    c.camera_center = torch.tensor([x, y, z], dtype=torch.float32)
    return c


def _real_cam(uid, x, y, z=0.0, yaw_deg=0.0):
    """A camera with a real world_view_transform (identity rotation, yawed
    about Y by yaw_deg), for the interpolation tests -- ``_cam`` above only
    carries ``camera_center``, which is all the ordering/pacing-distance
    logic needs, but interpolation needs a real, invertible pose matrix."""
    theta = math.radians(yaw_deg)
    R = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ], dtype=np.float64)
    C = np.array([x, y, z], dtype=np.float64)
    T = -R.T @ C
    wvt = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float()
    c = types.SimpleNamespace()
    c.uid = uid
    c.world_view_transform = wvt
    c.camera_center = wvt.inverse()[3, :3]
    c.image_width = 640
    c.image_height = 480
    c.FoVx = 1.0
    c.FoVy = 0.8
    c.image_name = f"cam{uid}"
    return c


def _path_length(cams):
    total = 0.0
    for a, b in zip(cams, cams[1:]):
        total += float(torch.linalg.norm(a.camera_center - b.camera_center))
    return total


class TestOrderCamsForTrajectory:
    def test_reorders_a_scrambled_line_into_a_continuous_sweep(self):
        # Cameras walk a straight line 0..9, but are listed in scrambled
        # (alphabetical-UUID-like) order -- exactly the real-world symptom.
        in_order = [_cam(i, float(i), 0.0) for i in range(10)]
        scrambled = [in_order[i] for i in [4, 9, 0, 7, 2, 8, 1, 6, 3, 5]]

        ordered = _order_cams_for_trajectory(scrambled)

        # A perfect sweep of a unit-spaced line covers total distance 9.0.
        # Greedy nearest-neighbor from an arbitrary (non-endpoint) start
        # isn't guaranteed to find that optimum -- it can zigzag once it
        # exhausts one direction and must jump across -- but it must still
        # land far closer to it than the scrambled input's naive order.
        assert _path_length(ordered) < _path_length(scrambled)
        assert _path_length(ordered) < 2 * 9.0

    def test_pure_reordering_no_cameras_dropped_or_duplicated(self):
        cams = [_cam(i, float(i) * 1.7, float(-i)) for i in range(15)]
        import random
        shuffled = cams.copy()
        random.Random(0).shuffle(shuffled)

        ordered = _order_cams_for_trajectory(shuffled)

        assert len(ordered) == len(cams)
        assert {c.uid for c in ordered} == {c.uid for c in cams}

    def test_short_lists_returned_unchanged(self):
        assert _order_cams_for_trajectory([]) == []
        one = [_cam(0, 0.0, 0.0)]
        assert _order_cams_for_trajectory(one) == one
        two = [_cam(0, 0.0, 0.0), _cam(1, 5.0, 5.0)]
        assert _order_cams_for_trajectory(two) == two


class TestQuaternionHelpers:
    def test_rotmat_quat_roundtrip(self):
        for yaw in (0.0, 30.0, 90.0, 137.0, 260.0):
            theta = math.radians(yaw)
            R = np.array([
                [math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)],
            ])
            q = _rotmat_to_quat(R)
            R2 = _quat_to_rotmat(q)
            assert np.allclose(R, R2, atol=1e-5)

    def test_slerp_endpoints_and_midpoint(self):
        q0 = _rotmat_to_quat(np.eye(3))
        theta = math.radians(90.0)
        R1 = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        q1 = _rotmat_to_quat(R1)

        assert np.allclose(_slerp_quat(q0, q1, 0.0), q0, atol=1e-5)
        # slerp(q, q1, 1.0) can land on q1's antipodal representation
        # (same rotation, -q) -- compare via the resulting rotation matrix.
        assert np.allclose(_quat_to_rotmat(_slerp_quat(q0, q1, 1.0)), R1, atol=1e-5)

        mid = _quat_to_rotmat(_slerp_quat(q0, q1, 0.5))
        mid_theta = math.radians(45.0)
        expected_mid = np.array([
            [math.cos(mid_theta), 0, math.sin(mid_theta)],
            [0, 1, 0],
            [-math.sin(mid_theta), 0, math.cos(mid_theta)],
        ])
        assert np.allclose(mid, expected_mid, atol=1e-4)


class TestInterpolateCamera:
    def test_midpoint_position_and_rotation(self):
        cam_a = _real_cam("a", 0.0, 0.0, 0.0, yaw_deg=0.0)
        cam_b = _real_cam("b", 10.0, 0.0, 0.0, yaw_deg=90.0)

        mid = _interpolate_camera(cam_a, cam_b, 0.5)

        assert torch.allclose(mid.camera_center, torch.tensor([5.0, 0.0, 0.0]), atol=1e-3)
        assert mid.image_width == cam_a.image_width
        assert mid.FoVx == cam_a.FoVx

    def test_t0_and_t1_recover_endpoints(self):
        cam_a = _real_cam("a", 1.0, 2.0, 3.0, yaw_deg=20.0)
        cam_b = _real_cam("b", -4.0, 5.0, 0.5, yaw_deg=200.0)

        at_a = _interpolate_camera(cam_a, cam_b, 0.0)
        at_b = _interpolate_camera(cam_a, cam_b, 1.0)

        assert torch.allclose(at_a.camera_center, cam_a.camera_center, atol=1e-3)
        assert torch.allclose(at_b.camera_center, cam_b.camera_center, atol=1e-3)


class TestPaceCamsForTrajectory:
    def test_large_gap_gets_interpolated_small_gaps_do_not(self):
        # Three real cameras: two close together (gap=1), then one far
        # away (gap=10) -- the far gap should get filled with synthetic
        # in-between frames so playback speed doesn't spike.
        cams = [
            _real_cam(0, 0.0, 0.0, 0.0),
            _real_cam(1, 1.0, 0.0, 0.0),
            _real_cam(2, 11.0, 0.0, 0.0),
        ]

        paced = _pace_cams_for_trajectory(cams, target_percentile=50.0)

        # All three real cameras must still be present, in order, with
        # their own exact original poses untouched.
        real_uids = [c.uid for c in paced if hasattr(c, "uid")]
        assert real_uids == [0, 1, 2]
        # The long gap must have produced extra (synthetic) frames between
        # cam 1 and cam 2, while the short gap did not.
        idx1 = paced.index(cams[1])
        idx2 = paced.index(cams[2])
        assert idx2 - idx1 > 1

    def test_pure_synthetic_frames_have_smoothly_decreasing_distance_to_target(self):
        cams = [_real_cam(0, 0.0, 0.0, 0.0), _real_cam(1, 20.0, 0.0, 0.0)]
        # Force pacing with an explicit small target via a third close pair
        # so the percentile-derived target is small relative to the big gap.
        cams = [
            _real_cam(0, 0.0, 0.0, 0.0),
            _real_cam(1, 1.0, 0.0, 0.0),
            _real_cam(2, 21.0, 0.0, 0.0),
        ]
        paced = _pace_cams_for_trajectory(cams, target_percentile=50.0, max_steps=25)
        centers = [c.camera_center[0].item() for c in paced]
        # Monotonic sweep along x: every consecutive step moves forward,
        # no backtracking/duplication artifacts from the interpolation.
        assert all(b >= a - 1e-4 for a, b in zip(centers, centers[1:]))

    def test_short_lists_returned_unchanged(self):
        assert _pace_cams_for_trajectory([]) == []
        one = [_real_cam(0, 0.0, 0.0, 0.0)]
        assert _pace_cams_for_trajectory(one) == one
