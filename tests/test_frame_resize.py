"""Tests for the shared RGB-D resize function (multires-cache plan, task 1).

Depth must NEVER be interpolated across a discontinuity -- every output
depth pixel must equal one of its source neighbors, never a blend.
"""
from __future__ import annotations

import numpy as np
import pytest

from eligsir.utils.streaming_frames import resize_rgbd_frame
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _make_intrinsics(width, height):
    return CameraIntrinsics(
        width=width, height=height, fx=500.0, fy=500.0, cx=width / 2.0, cy=height / 2.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )


def test_divisor_one_is_a_noop():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.random.randint(100, 5000, (64, 64), dtype=np.uint16)
    intr = _make_intrinsics(64, 64)
    out_rgb, out_depth, out_intr = resize_rgbd_frame(rgb, depth, intr, divisor=1)
    assert np.array_equal(out_rgb, rgb)
    assert np.array_equal(out_depth, depth)
    assert out_intr.width == 64 and out_intr.height == 64


def test_depth_never_interpolated_across_a_step_edge():
    # Sharp step: left half depth=1000mm, right half depth=4000mm. A
    # bilinear/area resize would produce intermediate values (e.g. 2500mm)
    # near the boundary; nearest-neighbor must not.
    depth = np.full((64, 64), 1000, dtype=np.uint16)
    depth[:, 32:] = 4000
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    intr = _make_intrinsics(64, 64)
    _, out_depth, _ = resize_rgbd_frame(rgb, depth, intr, divisor=4)
    valid_values = {1000, 4000}
    assert set(np.unique(out_depth).tolist()).issubset(valid_values)


def test_intrinsics_scaled_with_pixel_center_correction():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.uint16)
    intr = _make_intrinsics(64, 64)
    _, _, out_intr = resize_rgbd_frame(rgb, depth, intr, divisor=2)
    assert out_intr.width == 32 and out_intr.height == 32
    assert out_intr.fx == pytest.approx(250.0)
    # cx' = (cx+0.5)*s - 0.5 with s=0.5, cx=32.0 -> (32.5)*0.5 - 0.5 = 15.75
    assert out_intr.cx == pytest.approx(15.75)


def test_rgb_uses_area_averaging_not_nearest():
    # A fine checkerboard averaged down should NOT reproduce only the two
    # original colors everywhere (nearest would); area-averaging blends.
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[::2, ::2] = 255
    depth = np.zeros((64, 64), dtype=np.uint16)
    intr = _make_intrinsics(64, 64)
    out_rgb, _, _ = resize_rgbd_frame(rgb, depth, intr, divisor=4)
    # At least one output pixel must be a blended (non-0, non-255) value.
    assert np.any((out_rgb > 0) & (out_rgb < 255))
