"""Regression tests for issue #162 -- HSA_STATUS_ERROR_EXCEPTION on ScanNet.

Root cause: sources whose DEPTH sensor resolution differs from their COLOR
camera's (ScanNet: 640x480 depth vs 1296x968 color -- see
StreamingRGBDFrame.depth_width/depth_height) fed ``build_resolution_cache``
a depth array that ``resize_rgbd_frame`` passes straight through at
divisor=1 (the "high" tier). ``bind_resolution_tier`` then wrote that
480x640 depth into ``cam._sensor_depth_cache`` while setting
``cam.image_height/width`` from the 968x1296 rgb of the same tier.

Every consumer that indexes sensor depth with pixel coordinates bounds-
checked against ``cam.image_height`` / ``cam.image_width`` -- the
load-bearing one being
``eligsir.mapping.integration.frame_preparation._update_provisional_support``'s
``z_lookup = sensor_d[0, vi[valid_px], ui[valid_px]]`` -- then read out of
bounds. On CPU torch that is a deterministic ``IndexError``; on ROCm the
same advanced-index op is ``index_elementwise_kernel``, which faults the
GPU queue with ``HSA_STATUS_ERROR_EXCEPTION`` (code 0x1016) instead.

These tests run entirely on CPU: the bug is in index ARITHMETIC / cached
tensor shape, not in the GPU kernel itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.mapping.integration.depth_helpers import _get_sensor_depth
from eligsir.mapping.observation_db import BufferStage, FrameRecord
from eligsir.mapping.resolution_cache import build_resolution_cache
from train_streaming import bind_resolution_tier
from eligsir.utils.rgbd_frames import CameraIntrinsics

# The exact ScanNet scene0231_00 geometry from the issue.
COLOR_H, COLOR_W = 968, 1296
DEPTH_H, DEPTH_W = 480, 640


class _Cam:
    """Minimal stand-in for the attributes bind_resolution_tier touches."""

    def __init__(self) -> None:
        self.original_image = None
        self.image_height = 0
        self.image_width = 0
        self.fx = self.fy = self.cx = self.cy = 0.0
        self._streaming_depth_path = None
        self._streaming_depth_scale = 1000.0
        self._sensor_depth_cache = None


def _scannet_like_record() -> FrameRecord:
    rgb = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
    depth = np.full((DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    intr = CameraIntrinsics(
        width=COLOR_W, height=COLOR_H,
        fx=1170.0, fy=1170.0, cx=COLOR_W / 2.0, cy=COLOR_H / 2.0,
        depth_scale=1.0,
    )
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    return FrameRecord(
        frame_id=5,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=0.0,
        submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache=cache,
    )


@pytest.mark.parametrize("tier", ["high", "low"])
def test_cached_tier_depth_sits_on_that_tiers_rgb_grid(tier: str) -> None:
    """Both tiers' depth must have the tier's own rgb pixel dimensions.

    Pre-fix, "high" was 480x640 against a 968x1296 rgb.
    """
    cache = _scannet_like_record().resolution_cache
    handle = cache[tier]
    assert handle.depth is not None
    assert tuple(handle.depth.shape[-2:]) == tuple(handle.rgb.shape[:2]), (
        f"tier={tier} depth {tuple(handle.depth.shape)} does not match its own "
        f"rgb grid {tuple(handle.rgb.shape[:2])}"
    )


@pytest.mark.parametrize("tier", ["high", "low"])
def test_bound_sensor_depth_is_indexable_over_the_full_camera_grid(tier: str) -> None:
    """The op that faults on GPU, run on CPU where it raises IndexError.

    This mirrors ``_update_provisional_support`` verbatim: pixel coordinates
    are validated against ``cam.image_width`` / ``cam.image_height`` and then
    used to index ``sensor_d[0, vi, ui]``. Pre-fix this raises
    ``IndexError: index 967 is out of bounds for dimension 1 with size 480``
    on the "high" tier (and is an out-of-bounds GPU read on ROCm).
    """
    record = _scannet_like_record()
    cam = _Cam()
    assert bind_resolution_tier(cam, record, tier) is True

    sensor_d = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    assert sensor_d is not None

    W, H = int(cam.image_width), int(cam.image_height)
    # Corner-inclusive coordinates: exactly the extremes valid_px admits.
    vi = torch.tensor([0, H // 2, H - 1], dtype=torch.long)
    ui = torch.tensor([0, W // 2, W - 1], dtype=torch.long)
    valid_px = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    assert bool(valid_px.all())

    z_lookup = sensor_d[0, vi[valid_px], ui[valid_px]]  # <-- index_elementwise_kernel
    assert z_lookup.shape == (3,)
    assert torch.allclose(z_lookup, torch.full((3,), 2.0))


def test_get_sensor_depth_guards_a_mismatched_externally_set_cache() -> None:
    """Chokepoint guard: any producer may set _sensor_depth_cache directly.

    ``bind_resolution_tier`` is not the only one (``render_report.py`` also
    assigns it without resizing), so ``_get_sensor_depth`` must reconcile a
    cached tensor against the requested target instead of trusting it.
    """
    cam = _Cam()
    cam.image_height, cam.image_width = COLOR_H, COLOR_W
    cam._sensor_depth_cache = torch.full((1, DEPTH_H, DEPTH_W), 2.0)

    out = _get_sensor_depth(cam, COLOR_H, COLOR_W)
    assert out is not None
    assert tuple(out.shape) == (1, COLOR_H, COLOR_W)
    # Resize-and-return must NOT write back: a later low-tier bind would
    # otherwise resample an already-stretched tensor (lossy double-resample).
    assert tuple(cam._sensor_depth_cache.shape) == (1, DEPTH_H, DEPTH_W)


def test_get_sensor_depth_normalizes_canonical_hw_cache_without_rewriting_it() -> None:
    cam = _Cam()
    cam.image_height, cam.image_width = DEPTH_H, DEPTH_W
    cam._sensor_depth_cache = torch.full((DEPTH_H, DEPTH_W), 2.0)

    out = _get_sensor_depth(cam, DEPTH_H, DEPTH_W)

    assert tuple(out.shape) == (1, DEPTH_H, DEPTH_W)
    assert tuple(cam._sensor_depth_cache.shape) == (DEPTH_H, DEPTH_W)


def test_comparison_report_get_sensor_depth_also_guards_mismatched_cache() -> None:
    """utils/comparison_report.py carries its OWN duplicate _get_sensor_depth
    (not shared with depth_helpers.py) that PR #185 did not touch. It has the
    identical unchecked-cache-return bug pattern; fixed 2026-08-13 alongside
    this issue's GPU-verification follow-up work. See that function's own
    docstring for why the duplication exists and what call sites use it.
    """
    from eligsir.utils.comparison_report import _get_sensor_depth as report_get_sensor_depth

    cam = _Cam()
    cam.image_height, cam.image_width = COLOR_H, COLOR_W
    cam._sensor_depth_cache = torch.full((1, DEPTH_H, DEPTH_W), 2.0)

    out = report_get_sensor_depth(cam, COLOR_H, COLOR_W)
    assert out is not None
    assert tuple(out.shape) == (1, COLOR_H, COLOR_W)
    assert tuple(cam._sensor_depth_cache.shape) == (1, DEPTH_H, DEPTH_W)


class _FakeGaussians:
    """Minimal stand-in for the attributes _update_provisional_support touches."""

    def __init__(self, n: int, device: str) -> None:
        self.provisional = torch.ones(n, dtype=torch.bool, device=device)
        self._xyz = torch.zeros((n, 3), device=device)
        self.support_count = torch.zeros(n, dtype=torch.int32, device=device)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz


class _FakeArgs:
    streaming_depth_consistency_thresh = 0.05
    streaming_depth_multiview_conflict_enabled = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="_update_provisional_support is CUDA-only")
def test_update_provisional_support_guard_fires_on_mismatched_sensor_depth(monkeypatch) -> None:
    """Defensive guard (2026-08-13): even with PR #185's two-layer fix in
    place, if some future producer of ``cam._sensor_depth_cache`` (or a
    ``_get_sensor_depth`` call site) reintroduces a resolution mismatch,
    ``_update_provisional_support`` must fail with a clear Python
    ``RuntimeError`` at the real origin -- not silently hand a
    wrong-shaped tensor to the ``sensor_d[0, vi, ui]`` gather that faulted
    the ROCm queue with HSA_STATUS_ERROR_EXCEPTION in the original report.
    Simulated here by monkeypatching ``_get_sensor_depth`` (imported by
    name into ``frame_preparation``) to return a deliberately wrong shape,
    bypassing #185's own reconciliation so the new guard is what catches it.
    """
    import eligsir.mapping.integration.frame_preparation as fp

    device = "cuda"
    monkeypatch.setattr(
        fp,
        "_get_sensor_depth",
        lambda cam, h, w: torch.full((1, DEPTH_H, DEPTH_W), 2.0, device="cpu"),
    )

    cam = _Cam()
    cam.image_height, cam.image_width = COLOR_H, COLOR_W
    cam.fx = cam.fy = 1170.0
    cam.cx, cam.cy = COLOR_W / 2.0, COLOR_H / 2.0
    cam.world_view_transform = torch.eye(4, device=device)

    gaussians = _FakeGaussians(n=4, device=device)
    render_pkg = {"visibility_filter": torch.ones(4, dtype=torch.bool, device=device)}

    with pytest.raises(RuntimeError, match=r"issue #162 guard"):
        fp._update_provisional_support(gaussians, cam, render_pkg, _FakeArgs())
