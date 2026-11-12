"""Issue #1323: presentation-only SSAA + sharpening.

Pure-logic / CPU-only tests -- no CUDA render involved. The critical safety property
this module exists to preserve (scored metrics are byte-identical whether the feature is
on or off) is verified separately via a real GPU render regression check against the fr3
champion checkpoint (see docs/CHAMPION_CHECKPOINTS.md's fast regression-check recipe) --
not practical to reproduce here without a GPU, but these tests pin the building blocks
that check depends on: no-op defaults, correct resolution scaling, and that the original
camera object passed to make_supersampled_camera is never mutated.
"""
import argparse

import numpy as np
import torch

from scene.cameras import MiniCam
from eligsir.utils.presentation_ssaa import (
    PresentationSSAAConfig,
    config_from_args,
    downsample_area,
    downsample_tensor_chw,
    make_supersampled_camera,
    presentation_postprocess,
    presentation_ssaa_kwargs_from_args,
    unsharp_mask,
)


def _minicam(w=100, h=60):
    wvt = torch.eye(4)
    full = torch.eye(4)
    return MiniCam(w, h, 1.0, 1.2, 0.01, 100.0, wvt, full)


def test_make_supersampled_camera_scales_resolution_and_does_not_mutate_original():
    cam = _minicam(100, 60)
    cam2 = make_supersampled_camera(cam, 2)
    assert (cam2.image_width, cam2.image_height) == (200, 120)
    assert (cam.image_width, cam.image_height) == (100, 60)
    # World-view/projection are resolution-independent -- same objects, no recompute needed.
    assert cam2.world_view_transform is cam.world_view_transform
    assert cam2.full_proj_transform is cam.full_proj_transform


def test_make_supersampled_camera_scales_explicit_intrinsics():
    """Issue #1330: cameras with explicit pixel-space fx/fy/cx/cy (e.g. TUM RGB-D) must
    have those intrinsics scaled by the same factor as image_width/image_height, or the K
    matrix desyncs from the enlarged canvas (zoomed-out/shifted framing)."""
    cam = _minicam(100, 60)
    cam.fx, cam.fy, cam.cx, cam.cy = 500.0, 510.0, 50.0, 30.0
    cam2 = make_supersampled_camera(cam, 4)
    assert (cam2.image_width, cam2.image_height) == (400, 240)
    assert cam2.fx == cam.fx * 4
    assert cam2.fy == cam.fy * 4
    assert cam2.cx == cam.cx * 4
    assert cam2.cy == cam.cy * 4
    # Original camera's intrinsics must be untouched.
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (500.0, 510.0, 50.0, 30.0)


def test_make_supersampled_camera_leaves_fovonly_camera_intrinsics_absent():
    """A camera with no explicit fx/fy/cx/cy (FoV-only construction) must not gain them --
    scene/cameras.py::_camera_intrinsics derives fresh values from FoV + the already-scaled
    resolution in that case, so make_supersampled_camera must leave them None."""
    cam = _minicam(100, 60)
    cam2 = make_supersampled_camera(cam, 2)
    assert getattr(cam2, "fx", None) is None
    assert getattr(cam2, "fy", None) is None
    assert getattr(cam2, "cx", None) is None
    assert getattr(cam2, "cy", None) is None


def test_make_supersampled_camera_factor_1_is_true_noop():
    cam = _minicam()
    assert make_supersampled_camera(cam, 1) is cam
    assert make_supersampled_camera(cam, 0) is cam
    assert make_supersampled_camera(cam, None) is cam


def test_downsample_area_noop_for_factor_1():
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    assert np.array_equal(downsample_area(img, 1), img)


def test_downsample_area_scales_down():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = downsample_area(img, 2)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_unsharp_mask_noop_for_zero_strength():
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    assert np.array_equal(unsharp_mask(img, 0.0), img)


def test_unsharp_mask_preserves_shape_and_dtype():
    img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    out = unsharp_mask(img, 0.2)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_presentation_postprocess_composes_downsample_then_sharpen():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = presentation_postprocess(img, supersample_factor=2, sharpen_strength=0.15)
    assert out.shape == (32, 32, 3)


def test_downsample_tensor_chw_noop_for_factor_1_no_target():
    img = torch.rand(3, 60, 100)
    out = downsample_tensor_chw(img, 1)
    assert out is img


def test_downsample_tensor_chw_scales_to_factor():
    img = torch.rand(3, 120, 200)
    out = downsample_tensor_chw(img, 2)
    assert out.shape == (3, 60, 100)


def test_downsample_tensor_chw_scales_to_explicit_target():
    img = torch.rand(3, 120, 200)
    out = downsample_tensor_chw(img, 1, target_hw=(60, 100))
    assert out.shape == (3, 60, 100)


def test_config_defaults_match_documented_flag_defaults():
    cfg = PresentationSSAAConfig()
    assert cfg.enabled is True
    assert cfg.supersample_factor == 8
    assert cfg.sharpen_strength == 0.15


def test_config_disabled_forces_effective_noop():
    cfg = PresentationSSAAConfig(enabled=False, supersample_factor=2, sharpen_strength=0.15)
    assert cfg.effective_factor == 1
    assert cfg.effective_sharpen == 0.0


def test_config_from_args_falls_back_on_missing_attrs():
    cfg = config_from_args(argparse.Namespace())
    assert cfg.enabled is True
    assert cfg.supersample_factor == 8
    assert cfg.sharpen_strength == 0.15


def test_presentation_ssaa_kwargs_from_args_shape():
    args = argparse.Namespace(
        presentation_supersample_enabled=True,
        presentation_supersample_factor=3,
        presentation_sharpen_strength=0.2,
    )
    kwargs = presentation_ssaa_kwargs_from_args(args)
    assert kwargs == {
        "presentation_supersample_factor": 3,
        "presentation_sharpen_strength": 0.2,
    }


def test_presentation_ssaa_kwargs_from_args_disabled_is_noop_kwargs():
    args = argparse.Namespace(
        presentation_supersample_enabled=False,
        presentation_supersample_factor=3,
        presentation_sharpen_strength=0.2,
    )
    kwargs = presentation_ssaa_kwargs_from_args(args)
    assert kwargs == {
        "presentation_supersample_factor": 1,
        "presentation_sharpen_strength": 0.0,
    }
