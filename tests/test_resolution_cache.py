"""Tests for the dual-tier resolution cache builder (task 2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import CacheHandle
from eligsir.mapping.resolution_cache import build_resolution_cache
from eligsir.utils.rgbd_frames import CameraIntrinsics


def _make_intrinsics():
    return CameraIntrinsics(
        width=64, height=64, fx=500.0, fy=500.0, cx=32.0, cy=32.0,
        depth_scale=1000.0, camera_frame="camera_color_optical_frame",
    )


def test_build_resolution_cache_produces_both_tiers_on_cpu():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.random.randint(100, 5000, (64, 64), dtype=np.uint16)
    intr = _make_intrinsics()
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    assert set(cache.keys()) == {"low", "high"}
    assert isinstance(cache["low"], CacheHandle)
    assert isinstance(cache["high"], CacheHandle)


def test_high_tier_matches_native_resolution():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.uint16)
    intr = _make_intrinsics()
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    assert cache["high"].rgb.shape[:2] == (64, 64)
    assert cache["high"].intrinsics.width == 64


def test_low_tier_is_actually_smaller():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.uint16)
    intr = _make_intrinsics()
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    assert cache["low"].rgb.shape[0] < cache["high"].rgb.shape[0]
    assert cache["low"].intrinsics.width == 16


def test_tensors_land_on_requested_device():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.uint16)
    intr = _make_intrinsics()
    cache = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    assert cache["low"].rgb.device.type == "cpu"
    assert cache["high"].depth.device.type == "cpu"


# ---------------------------------------------------------------------------
# 2026-08-12 fix: depth-less sources (RealityScan Mobile -- RGB + poses +
# sparse SfM cloud, no per-frame depth) previously left the entire
# resolution_cache permanently empty (bind_resolution_tier failed 100% of
# calls for a whole run). The low/high tier DECISION (ResolutionController)
# is driven purely by ingestion throughput, not depth content, so a
# depth-less source can still meaningfully use rgb-only tiers.
# ---------------------------------------------------------------------------


def test_build_resolution_cache_with_no_depth_still_produces_both_rgb_tiers():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    intr = _make_intrinsics()
    cache = build_resolution_cache(rgb, None, intr, low_res_divisor=4, device="cpu")
    assert set(cache.keys()) == {"low", "high"}
    assert isinstance(cache["low"], CacheHandle)
    assert isinstance(cache["high"], CacheHandle)
    assert cache["high"].rgb.shape[:2] == (64, 64)
    assert cache["low"].rgb.shape[:2] == (16, 16)
    # No depth to cache -- CacheHandle.depth must stay None, never a
    # fabricated placeholder tensor.
    assert cache["low"].depth is None
    assert cache["high"].depth is None


def test_build_resolution_cache_no_depth_scales_intrinsics_same_as_depth_path():
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    depth = np.random.randint(100, 5000, (64, 64), dtype=np.uint16)
    intr = _make_intrinsics()
    cache_with_depth = build_resolution_cache(rgb, depth, intr, low_res_divisor=4, device="cpu")
    cache_no_depth = build_resolution_cache(rgb, None, intr, low_res_divisor=4, device="cpu")
    for tier in ("low", "high"):
        with_d = cache_with_depth[tier].intrinsics
        no_d = cache_no_depth[tier].intrinsics
        assert with_d.width == no_d.width
        assert with_d.height == no_d.height
        assert with_d.fx == pytest.approx(no_d.fx)
        assert with_d.cx == pytest.approx(no_d.cx)
