"""Exact crop-camera geometry (#310 item 1-2).

The load-bearing test is `test_projection_parity_*`: a 3D point projected
through the parent camera and then mapped through the crop transform must land
on the same pixel as projecting it directly through the crop camera. #310 says
"approximate camera geometry is not acceptable", and this is what makes that
checkable rather than asserted.

Torch-free by construction -- the intrinsics/pixel math in
`eligsir.mapping.supervision.crop_camera` imports torch lazily, so this file
runs in the lean CPU CI image.
"""
from __future__ import annotations

import math

import pytest

from eligsir.mapping.supervision.crop_camera import (
    CropIntrinsics,
    CropTransform,
    derive_crop_intrinsics,
    project_point,
    project_to_crop_pixel,
    parent_intrinsics,
)

PARENT = CropIntrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480)


class _FakeCamera:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------
# CropTransform validation
# --------------------------------------------------------------------------

def test_crop_must_fit_inside_parent():
    with pytest.raises(ValueError):
        CropTransform(x0=600, y0=0, width=100, height=100,
                      out_width=100, out_height=100,
                      parent_width=640, parent_height=480)


def test_crop_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        CropTransform(x0=0, y0=0, width=0, height=10, out_width=10, out_height=10,
                      parent_width=640, parent_height=480)


def test_identity_transform_round_trips():
    t = CropTransform.full_frame(640, 480)
    assert t.is_identity
    assert t.scale_x == 1.0 and t.scale_y == 1.0
    assert t.supervision_pixels == 640 * 480
    assert t.as_dict()["parent_width"] == 640


# --------------------------------------------------------------------------
# Intrinsics derivation
# --------------------------------------------------------------------------

def test_identity_crop_preserves_intrinsics_exactly():
    got = derive_crop_intrinsics(PARENT, CropTransform.full_frame(640, 480))
    assert got == PARENT


def test_pure_crop_shifts_principal_point_only():
    t = CropTransform(x0=100, y0=60, width=200, height=150,
                      out_width=200, out_height=150,
                      parent_width=640, parent_height=480)
    got = derive_crop_intrinsics(PARENT, t)
    assert got.fx == PARENT.fx and got.fy == PARENT.fy      # focal untouched
    assert got.cx == pytest.approx(PARENT.cx - 100)
    assert got.cy == pytest.approx(PARENT.cy - 60)
    assert (got.width, got.height) == (200, 150)


def test_resize_uses_the_resolution_materialization_half_pixel_convention():
    """Must match `resolution_materialization.py`'s formula verbatim.

    A different convention here would make crop-then-resize and
    resize-then-crop disagree by half a pixel, and the parity test below would
    then encode the bug instead of catching it.
    """
    t = CropTransform(x0=0, y0=0, width=640, height=480,
                      out_width=320, out_height=240,
                      parent_width=640, parent_height=480)
    got = derive_crop_intrinsics(PARENT, t)
    assert got.fx == pytest.approx(PARENT.fx * 0.5)
    assert got.cx == pytest.approx((PARENT.cx + 0.5) * 0.5 - 0.5)
    assert got.cy == pytest.approx((PARENT.cy + 0.5) * 0.5 - 0.5)


def test_transform_parent_size_must_match_intrinsics():
    t = CropTransform(x0=0, y0=0, width=10, height=10, out_width=10, out_height=10,
                      parent_width=320, parent_height=240)
    with pytest.raises(ValueError):
        derive_crop_intrinsics(PARENT, t)


# --------------------------------------------------------------------------
# Projection parity -- the #310 acceptance test
# --------------------------------------------------------------------------

_POINTS = [
    (0.0, 0.0, 2.0),
    (0.31, -0.22, 1.7),
    (-0.9, 0.4, 3.3),
    (0.05, 0.05, 1.05),
]


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize(
    "transform",
    [
        CropTransform(x0=0, y0=0, width=640, height=480, out_width=640, out_height=480,
                      parent_width=640, parent_height=480),
        CropTransform(x0=137, y0=91, width=200, height=160, out_width=200, out_height=160,
                      parent_width=640, parent_height=480),
        # off-centre AND rescaled -- the case a centred-crop shortcut gets wrong
        CropTransform(x0=137, y0=91, width=200, height=160, out_width=100, out_height=80,
                      parent_width=640, parent_height=480),
        # upscaled crop: native detail promotion, the #309 motivating case
        CropTransform(x0=400, y0=300, width=120, height=90, out_width=240, out_height=180,
                      parent_width=640, parent_height=480),
    ],
)
def test_projection_parity_parent_then_transform_equals_direct(point, transform):
    crop = derive_crop_intrinsics(PARENT, transform)
    via_parent = project_to_crop_pixel(project_point(PARENT, point), transform)
    direct = project_point(crop, point)
    assert via_parent[0] == pytest.approx(direct[0], abs=1e-9)
    assert via_parent[1] == pytest.approx(direct[1], abs=1e-9)


def test_crop_corner_maps_to_crop_origin():
    """The crop rectangle's own corner must land at the crop's pixel origin."""
    t = CropTransform(x0=137, y0=91, width=200, height=160, out_width=200, out_height=160,
                      parent_width=640, parent_height=480)
    assert project_to_crop_pixel((137.0, 91.0), t) == pytest.approx((0.0, 0.0))
    assert project_to_crop_pixel((337.0, 251.0), t) == pytest.approx((200.0, 160.0))


# --------------------------------------------------------------------------
# parent_intrinsics fallback ladder -- must mirror the renderer's
# --------------------------------------------------------------------------

def test_parent_intrinsics_prefers_explicit_pinhole():
    cam = _FakeCamera(image_width=640, image_height=480, fx=525.0, fy=520.0, cx=310.0, cy=250.0)
    assert parent_intrinsics(cam) == CropIntrinsics(525.0, 520.0, 310.0, 250.0, 640, 480)


def test_parent_intrinsics_falls_back_to_fov_and_centred_pp():
    """Same ladder as gsplat_backend._camera_tensors, so a FoV-only parent's
    crop still lands where the parent's own render put it."""
    fov_x = 2.0 * math.atan(640 / (2.0 * 525.0))
    fov_y = 2.0 * math.atan(480 / (2.0 * 525.0))
    cam = _FakeCamera(image_width=640, image_height=480, FoVx=fov_x, FoVy=fov_y)
    got = parent_intrinsics(cam)
    assert got.fx == pytest.approx(525.0)
    assert got.fy == pytest.approx(525.0)
    assert (got.cx, got.cy) == (320.0, 240.0)


def test_parent_intrinsics_refuses_a_camera_with_no_geometry():
    with pytest.raises(ValueError):
        parent_intrinsics(_FakeCamera(image_width=640, image_height=480))


def test_fov_helpers_are_consistent_with_focal():
    got = derive_crop_intrinsics(
        PARENT,
        CropTransform(x0=100, y0=100, width=320, height=240, out_width=320, out_height=240,
                      parent_width=640, parent_height=480),
    )
    assert got.fov_x == pytest.approx(2.0 * math.atan(320 / (2.0 * 525.0)))
    assert got.fov_y == pytest.approx(2.0 * math.atan(240 / (2.0 * 525.0)))
