"""Tests for ``_expand_cams_with_interpolation`` (issue #397 follow-up,
"trajectory.mp4 looks chonky/stuttery at 30fps").

``distribute_frame_repeats`` fixes total video DURATION by repeating each
real render for as many output frames as its real time slice covers --
correct duration, but each slice visibly holds a static frame. This module
instead spreads a slice's existing repeat budget across the real camera plus
a few SLERP/lerp-interpolated poses swept toward the NEXT real camera, so
the same output-frame budget shows genuine motion.

These tests are CPU-only (torch, no GPU, no real render) -- they cover the
frame/duration ACCOUNTING invariant (mandatory: total encoded frame count
and implied duration must be identical whether interpolation is on or off)
and the POSE math (reusing the already-slerp-tested ``_interpolate_camera``
via the same synthetic-camera helper pattern as
``test_trajectory_video_camera_ordering.py``).
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from eligsir.utils.comparison_report import _expand_cams_with_interpolation
from eligsir.utils.graphics_utils import getWorld2View2


def _real_cam(uid, x, y, z=0.0, yaw_deg=0.0):
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


def _line_of_cams(n, spacing=1.0):
    return [_real_cam(i, float(i) * spacing, 0.0, 0.0) for i in range(n)]


# --- Frame/duration accounting invariant (mandatory) -------------------------

def test_total_encoded_frame_count_unchanged_by_interpolation():
    cams = _line_of_cams(4)
    frame_repeats = [7, 3, 12, 1]  # arbitrary, from distribute_frame_repeats

    new_cams, new_repeats = _expand_cams_with_interpolation(
        cams, frame_repeats, max_extra_renders=None, max_per_slice=None)

    assert sum(new_repeats) == sum(frame_repeats)
    assert len(new_cams) == len(new_repeats)


def test_implied_video_duration_unchanged():
    """Duration = sum(repeats) / fps must be identical with interpolation on
    or off -- the whole point is changing WHICH pose backs a frame, never
    how many frames there are."""
    cams = _line_of_cams(5)
    frame_repeats = [10, 4, 20, 6, 3]
    fps = 30.0

    duration_before = sum(frame_repeats) / fps
    _, new_repeats = _expand_cams_with_interpolation(cams, frame_repeats)
    duration_after = sum(new_repeats) / fps

    assert duration_after == pytest.approx(duration_before)


def test_none_frame_repeats_is_a_no_op():
    cams = _line_of_cams(3)
    new_cams, new_repeats = _expand_cams_with_interpolation(cams, None)
    assert new_cams == cams
    assert new_repeats is None


def test_empty_cams_is_a_no_op():
    new_cams, new_repeats = _expand_cams_with_interpolation([], [1, 2, 3])
    assert new_cams == []


def test_last_slice_never_interpolated_holds_verbatim():
    """No 'next' camera for the final real camera -- it must still appear
    exactly once, repeated frame_repeats[-1] times by the CALLER's repeat
    mechanism (this function only ever emits the real cam once for that
    slice, with the full repeat count attached to it)."""
    cams = _line_of_cams(3)
    frame_repeats = [1, 1, 9]

    new_cams, new_repeats = _expand_cams_with_interpolation(cams, frame_repeats)

    assert new_cams[-1] is cams[-1]
    assert new_repeats[-1] == 9


def test_interpolated_cameras_are_inserted_between_real_ones_in_order():
    cams = _line_of_cams(3)
    frame_repeats = [4, 4, 1]  # slice 0 and 1 each get 3 interpolated extras

    new_cams, new_repeats = _expand_cams_with_interpolation(
        cams, frame_repeats, max_extra_renders=None, max_per_slice=None)

    # Real cameras must all still be present, in original relative order.
    real_positions = [i for i, c in enumerate(new_cams) if getattr(c, "uid", None) in (0, 1, 2)]
    real_uids_in_order = [new_cams[i].uid for i in real_positions]
    assert real_uids_in_order == [0, 1, 2]

    # Between real cam 0 and real cam 1, the interpolated poses' camera
    # centers must move smoothly (monotonically) from cam0 toward cam1.
    i0, i1 = real_positions[0], real_positions[1]
    xs = [float(new_cams[i].camera_center[0]) for i in range(i0, i1 + 1)]
    assert xs == sorted(xs)  # monotonically increasing x, cam0 -> cam1
    assert xs[0] == 0.0
    assert xs[-1] == 1.0  # spacing=1.0 between cam0 and cam1


def test_interpolated_poses_are_synthetic_not_the_real_camera_objects():
    cams = _line_of_cams(2)
    frame_repeats = [5, 1]

    new_cams, _ = _expand_cams_with_interpolation(cams, frame_repeats, max_per_slice=None)

    # 5 output frames for slice 0 -> 5 unique poses: real cam0 + 4 interpolated.
    assert len(new_cams) == 6  # cam0 + 4 interpolated + cam1
    assert new_cams[0] is cams[0]
    for interp_cam in new_cams[1:5]:
        assert interp_cam is not cams[0]
        assert interp_cam is not cams[1]
        assert not hasattr(interp_cam, "uid")  # synthetic, not a real Camera
    assert new_cams[5] is cams[1]


def test_max_per_slice_caps_unique_renders_added():
    cams = _line_of_cams(2)
    frame_repeats = [100, 1]

    new_cams, new_repeats = _expand_cams_with_interpolation(
        cams, frame_repeats, max_extra_renders=None, max_per_slice=3)

    # At most 3 interpolated extras + the real camera = 4 unique poses for
    # slice 0, regardless of its large repeat count -- but total frames
    # (100) must still be preserved via new_repeats.
    n_unique_in_slice0 = len(new_cams) - 1  # minus the trailing real cam1
    assert n_unique_in_slice0 <= 4
    assert sum(new_repeats) == 101


def test_max_extra_renders_bounds_total_added_renders_across_whole_video():
    cams = _line_of_cams(5)
    frame_repeats = [50, 50, 50, 50, 1]
    n_before = len(cams)

    new_cams, new_repeats = _expand_cams_with_interpolation(
        cams, frame_repeats, max_extra_renders=10, max_per_slice=None)

    added = len(new_cams) - n_before
    assert added <= 10
    assert sum(new_repeats) == sum(frame_repeats)  # accounting still exact


def test_interpolation_disabled_via_zero_caps_falls_back_to_real_cam_only():
    """max_per_slice=0 means no interpolated poses are added anywhere --
    every slice keeps exactly its one real camera, repeated verbatim (the
    pre-existing behavior when this feature is effectively off)."""
    cams = _line_of_cams(3)
    frame_repeats = [5, 5, 5]

    new_cams, new_repeats = _expand_cams_with_interpolation(
        cams, frame_repeats, max_extra_renders=None, max_per_slice=0)

    assert len(new_cams) == 3
    assert new_cams == cams
    assert new_repeats == frame_repeats


def test_length_mismatch_is_a_no_op_not_an_indexerror():
    cams = _line_of_cams(3)
    mismatched_repeats = [1, 2]  # deliberately wrong length

    new_cams, new_repeats = _expand_cams_with_interpolation(cams, mismatched_repeats)

    assert new_cams == cams
    assert new_repeats == mismatched_repeats


# --- _infer_realtime_fps regression guard (issue #397 original bug) --------
#
# The original #397 bug: an interpolated camera carries no
# _streaming_timestamp, so if one ever reaches _infer_realtime_fps it forces
# the "not a real capture" default-fps fallback. This feature must never
# cause that -- _expand_cams_with_interpolation's output is only ever handed
# to the renderer, strictly after _infer_realtime_fps (or, on the
# timestamped branch, the equivalent duration/fps derivation) has already
# run on the un-expanded camera list.

def test_infer_realtime_fps_falls_back_when_an_interpolated_cam_is_present():
    """Locks in the ORIGINAL #397 trap this feature must never reintroduce:
    _infer_realtime_fps has no special-case for interpolated cameras --
    it is the CALLER's job (comparison_report.py's trajectory-video block)
    to run it only on the real, un-expanded camera/timestamp list, before
    any interpolation happens."""
    from eligsir.utils.comparison_report import _infer_realtime_fps, _interpolate_camera

    real_a = _real_cam(0, 0.0, 0.0, 0.0)
    real_a._streaming_timestamp = 0.0
    real_b = _real_cam(1, 1.0, 0.0, 0.0)
    real_b._streaming_timestamp = 1.0
    interp = _interpolate_camera(real_a, real_b, 0.5)
    assert not hasattr(interp, "_streaming_timestamp")

    default_fps = 30
    assert _infer_realtime_fps([real_a, interp, real_b], default_fps) == default_fps


# --- SSAA compatibility (advisor review, 2026-08-31) ------------------------
#
# trajectory.mp4 is rendered through utils/presentation_ssaa.py's
# make_supersampled_camera when Full SSAA is on (default). An interpolated
# camera is a bare types.SimpleNamespace (see _interpolate_camera's
# docstring) -- confirm it survives that path unchanged in shape, i.e. it
# carries every attribute the renderer (eligsir/mapping/renderer/
# gsplat_backend.py) actually reads: world_view_transform, image_width,
# image_height, FoVx, FoVy, camera_center. This is CPU-only (no GPU render
# needed) since make_supersampled_camera is pure attribute copying/scaling.

def test_interpolated_camera_survives_ssaa_supersampling():
    from eligsir.utils.presentation_ssaa import make_supersampled_camera

    cam_a = _real_cam(0, 0.0, 0.0, 0.0)
    cam_b = _real_cam(1, 1.0, 0.0, 0.0)
    from eligsir.utils.comparison_report import _interpolate_camera
    interp = _interpolate_camera(cam_a, cam_b, 0.5)

    scaled = make_supersampled_camera(interp, 8)

    # Every attribute the renderer reads must survive, and width/height must
    # actually scale (the whole point of SSAA).
    for attr in ("world_view_transform", "FoVx", "FoVy", "camera_center"):
        assert hasattr(scaled, attr)
    assert scaled.image_width == interp.image_width * 8
    assert scaled.image_height == interp.image_height * 8


def test_interpolated_camera_ssaa_factor_1_is_unchanged():
    from eligsir.utils.presentation_ssaa import make_supersampled_camera

    cam_a = _real_cam(0, 0.0, 0.0, 0.0)
    cam_b = _real_cam(1, 1.0, 0.0, 0.0)
    from eligsir.utils.comparison_report import _interpolate_camera
    interp = _interpolate_camera(cam_a, cam_b, 0.5)

    assert make_supersampled_camera(interp, 1) is interp
