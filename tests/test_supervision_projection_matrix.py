"""`getProjectionMatrixFromIntrinsics` -- asymmetric frustum for a crop camera.

`Camera.projection_matrix`/`full_proj_transform` are built from FoV alone, so
their frustum is symmetric and cannot represent an off-centre principal point.
The rasterizer does not care (it takes a real 3x3 K), but every other consumer
of those fields would silently disagree with the render for a crop camera. This
file pins the replacement: exact agreement with the existing function in the
centred case, and correct NDC mapping in the off-centre case.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="projection matrices are torch tensors")

from eligsir.utils.graphics_utils import (  # noqa: E402
    focal2fov,
    getProjectionMatrix,
    getProjectionMatrixFromIntrinsics,
)

ZNEAR, ZFAR = 0.01, 100.0


@pytest.mark.parametrize(
    "fx,fy,width,height",
    [(525.0, 525.0, 640, 480), (300.0, 410.0, 320, 240), (1200.0, 1200.0, 1920, 1080)],
)
def test_centred_case_reproduces_the_fov_form_exactly(fx, fy, width, height):
    ours = getProjectionMatrixFromIntrinsics(
        ZNEAR, ZFAR, fx, fy, width / 2.0, height / 2.0, width, height
    )
    theirs = getProjectionMatrix(
        ZNEAR, ZFAR, focal2fov(fx, width), focal2fov(fy, height)
    )
    assert torch.allclose(ours, theirs, atol=1e-6), f"\n{ours}\n!=\n{theirs}"


def test_centred_case_has_no_frustum_shear():
    P = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, 525.0, 525.0, 320.0, 240.0, 640, 480)
    assert P[0, 2].item() == pytest.approx(0.0, abs=1e-7)
    assert P[1, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_offcentre_principal_point_shears_the_frustum():
    P = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, 525.0, 525.0, 100.0, 240.0, 640, 480)
    # Principal point left of centre => the frustum leans, P[0,2] != 0.
    assert abs(P[0, 2].item()) > 1e-3
    assert P[1, 2].item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("cx,cy", [(320.0, 240.0), (100.0, 240.0), (500.0, 90.0), (0.0, 0.0)])
def test_ndc_agrees_with_the_pinhole_k_the_rasterizer_uses(cx, cy):
    """The decisive property: the projection matrix and the K matrix handed to
    gsplat must place the same camera-space point on the same pixel."""
    fx, fy, width, height = 525.0, 525.0, 640, 480
    P = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, fx, fy, cx, cy, width, height)
    for point in [(0.0, 0.0, 2.0), (0.4, -0.3, 1.6), (-0.7, 0.55, 3.1)]:
        x, y, z = point
        clip = P @ torch.tensor([x, y, z, 1.0])
        ndc_x = (clip[0] / clip[3]).item()
        ndc_y = (clip[1] / clip[3]).item()
        # NDC in [-1, 1] -> pixel
        px = (ndc_x + 1.0) * 0.5 * width
        py = (ndc_y + 1.0) * 0.5 * height
        assert px == pytest.approx(fx * x / z + cx, abs=1e-3)
        assert py == pytest.approx(fy * y / z + cy, abs=1e-3)


def test_depth_terms_are_untouched_by_the_principal_point():
    a = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, 525.0, 525.0, 320.0, 240.0, 640, 480)
    b = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, 525.0, 525.0, 90.0, 400.0, 640, 480)
    assert a[2, 2].item() == pytest.approx(b[2, 2].item())
    assert a[2, 3].item() == pytest.approx(b[2, 3].item())
    assert a[3, 2].item() == pytest.approx(b[3, 2].item()) == pytest.approx(1.0)


def test_matches_the_analytic_focal_scale():
    P = getProjectionMatrixFromIntrinsics(ZNEAR, ZFAR, 525.0, 480.0, 137.0, 311.0, 640, 480)
    assert P[0, 0].item() == pytest.approx(2.0 * 525.0 / 640)
    assert P[1, 1].item() == pytest.approx(2.0 * 480.0 / 480)


def test_fov_helpers_round_trip():
    assert focal2fov(525.0, 640) == pytest.approx(2.0 * math.atan(640 / (2 * 525.0)))


# --------------------------------------------------------------------------
# CropCamera / crop_image -- execution coverage for the camera object itself
#
# NOTE: this is analytic geometry only. #310's "parent/crop render
# correspondence on synthetic geometry" needs a real rasterization and is
# deliberately NOT claimed here -- see the plan doc, section 5.2.
# --------------------------------------------------------------------------

from eligsir.mapping.supervision.crop_camera import (  # noqa: E402
    CropTransform,
    crop_camera_from,
    crop_image,
    derive_crop_intrinsics,
    parent_intrinsics,
)


class _ParentCam:
    def __init__(self):
        self.uid = 7
        self.colmap_id = 7
        self.image_name = "frame_0007"
        self.fx, self.fy, self.cx, self.cy = 525.0, 525.0, 319.5, 239.5
        self.image_width, self.image_height = 640, 480
        self.znear, self.zfar = 0.01, 100.0
        self.world_view_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.original_image = torch.rand(3, 480, 640)


_T = CropTransform(x0=137, y0=91, width=200, height=160,
                   out_width=200, out_height=160,
                   parent_width=640, parent_height=480)


def test_crop_camera_carries_exact_intrinsics_and_size():
    cam = crop_camera_from(_ParentCam(), _T)
    expected = derive_crop_intrinsics(parent_intrinsics(_ParentCam()), _T)
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (
        expected.fx, expected.fy, expected.cx, expected.cy
    )
    assert (cam.image_width, cam.image_height) == (200, 160)
    assert cam.width == cam.image_width and cam.height == cam.image_height


def test_crop_camera_projection_matrix_is_asymmetric_and_matches_its_own_k():
    cam = crop_camera_from(_ParentCam(), _T)
    # projection_matrix is stored transposed (matching scene/cameras.py), so
    # undo that before applying it as a left-multiplied matrix.
    P = cam.projection_matrix.transpose(0, 1)
    assert abs(P[0, 2].item()) > 1e-3, "an off-centre crop must shear the frustum"
    x, y, z = 0.21, -0.14, 1.9
    clip = P @ torch.tensor([x, y, z, 1.0])
    px = ((clip[0] / clip[3]).item() + 1.0) * 0.5 * cam.image_width
    py = ((clip[1] / clip[3]).item() + 1.0) * 0.5 * cam.image_height
    assert px == pytest.approx(cam.fx * x / z + cam.cx, abs=1e-3)
    assert py == pytest.approx(cam.fy * y / z + cam.cy, abs=1e-3)


def test_crop_camera_shares_the_parent_pose_and_drops_stale_gsplat_cache():
    parent = _ParentCam()
    parent._gsplat_K = torch.eye(3)          # a full-frame K that must NOT be inherited
    cam = crop_camera_from(parent, _T)
    assert cam.world_view_transform is parent.world_view_transform
    assert cam.camera_center is parent.camera_center
    assert not hasattr(cam, "_gsplat_K")
    assert not hasattr(cam, "gsplat")
    assert cam.parent_camera is parent
    assert "crop137_91" in cam.image_name


def test_crop_camera_can_carry_the_cropped_ground_truth():
    parent = _ParentCam()
    cam = crop_camera_from(parent, _T, crop_parent_image=True)
    assert tuple(cam.original_image.shape) == (3, 160, 200)
    assert torch.equal(cam.original_image, parent.original_image[:, 91:251, 137:337])


def test_crop_image_resizes_when_the_transform_rescales():
    t = CropTransform(x0=0, y0=0, width=64, height=64, out_width=32, out_height=32,
                      parent_width=64, parent_height=64)
    out = crop_image(torch.rand(3, 64, 64), t)
    assert tuple(out.shape) == (3, 32, 32)


def test_crop_image_of_a_constant_region_is_that_constant():
    img = torch.zeros(3, 64, 64)
    img[:, 0:32, 0:32] = 0.75
    t = CropTransform(x0=0, y0=0, width=32, height=32, out_width=16, out_height=16,
                      parent_width=64, parent_height=64)
    assert torch.allclose(crop_image(img, t), torch.full((3, 16, 16), 0.75))
