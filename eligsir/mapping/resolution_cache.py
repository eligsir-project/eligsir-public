"""Dual-tier frame cache with one heterogeneous pooled H2D per source frame."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from eligsir.io.sources.decode_cache import resize_rgbd_frame
from eligsir.mapping.resolution_materialization import (
    DEPTH_DOWNSAMPLE_MODES,
    _resize_depth,
)
from eligsir.mapping.observation_db import CacheHandle
from eligsir.runtime.transfers import DeviceTransferOwner, get_device_transfer_owner


def resolution_cache_transfer_stats(device: str) -> dict[str, int]:
    return get_device_transfer_owner(device).stats.snapshot()


def _resize_rgb_only(rgb: np.ndarray, intrinsics, divisor: int):
    _R = max(1, int(divisor))
    if _R <= 1:
        return rgb, intrinsics
    import cv2 as _cv2
    width, height = int(intrinsics.width), int(intrinsics.height)
    new_w = max(1, width // _R)
    new_h = max(1, height // _R)
    out_rgb = _cv2.resize(rgb, (new_w, new_h), interpolation=_cv2.INTER_AREA)
    sx, sy = new_w / float(width), new_h / float(height)
    new_cx = (intrinsics.cx + 0.5) * sx - 0.5
    new_cy = (intrinsics.cy + 0.5) * sy - 0.5
    from dataclasses import replace
    out_intrinsics = replace(
        intrinsics,
        width=new_w,
        height=new_h,
        fx=intrinsics.fx * sx,
        fy=intrinsics.fy * sy,
        cx=new_cx,
        cy=new_cy,
    )
    return out_rgb, out_intrinsics


def _match_depth_to_rgb(rgb: np.ndarray, depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    target_h, target_w = int(rgb.shape[0]), int(rgb.shape[1])
    if depth.shape[0] == target_h and depth.shape[1] == target_w:
        return depth
    import cv2 as _cv2
    return _cv2.resize(depth, (target_w, target_h), interpolation=_cv2.INTER_NEAREST)


def _cpu_tensors(arrays):
    out = []
    for arr in arrays:
        if arr is None:
            out.append(None)
            continue
        if not (arr.flags["C_CONTIGUOUS"] and arr.flags["WRITEABLE"]):
            arr = np.array(arr, copy=True, order="C")
        out.append(torch.from_numpy(arr))
    return tuple(out)


def build_resolution_cache(
    rgb: np.ndarray,
    depth: Optional[np.ndarray],
    intrinsics,
    low_res_divisor: int,
    device: str = "cuda",
    transfer_owner: Optional[DeviceTransferOwner] = None,
    depth_downsample: str = "nearest",
) -> Dict[str, CacheHandle]:
    """Build native/high + low caches with one physical H2D on accelerators.

    RGB stays area-resampled; depth stays nearest by default, while
    ``valid_area`` uses the canonical validity-aware depth reduction. The four
    heterogeneous arrays are byte-packed with alignment into one reusable
    pinned slab, copied once, then reinterpreted as typed device views. No
    per-frame ``pin_memory()`` and no swallowed pinning failure remain.
    """
    if depth_downsample not in DEPTH_DOWNSAMPLE_MODES:
        raise ValueError(
            f"depth_downsample must be one of {DEPTH_DOWNSAMPLE_MODES!r}, got {depth_downsample!r}"
        )
    if depth is None:
        high_rgb, high_intr = _resize_rgb_only(rgb, intrinsics, divisor=1)
        low_rgb, low_intr = _resize_rgb_only(rgb, intrinsics, divisor=low_res_divisor)
        high_depth = None
        low_depth = None
    else:
        high_rgb, high_depth, high_intr = resize_rgbd_frame(
            rgb, depth, intrinsics, divisor=1
        )
        high_depth = _match_depth_to_rgb(high_rgb, high_depth)
        if depth_downsample == "valid_area":
            low_rgb, low_intr = _resize_rgb_only(
                rgb, intrinsics, divisor=low_res_divisor
            )
            low_depth = _resize_depth(
                torch.from_numpy(high_depth),
                height=int(low_rgb.shape[0]),
                width=int(low_rgb.shape[1]),
                depth_downsample=depth_downsample,
            ).numpy()
        else:
            low_rgb, low_depth, low_intr = resize_rgbd_frame(
                rgb, depth, intrinsics, divisor=low_res_divisor
            )
        low_depth = _match_depth_to_rgb(low_rgb, low_depth)

    arrays = (high_rgb, high_depth, low_rgb, low_depth)
    target = torch.device(device)
    if target.type == "cpu":
        high_rgb_t, high_depth_t, low_rgb_t, low_depth_t = _cpu_tensors(arrays)
    else:
        owner = transfer_owner or get_device_transfer_owner(target)
        high_rgb_t, high_depth_t, low_rgb_t, low_depth_t = owner.upload_packed(
            arrays, purpose="resolution_cache"
        )[0]

    high = CacheHandle(rgb=high_rgb_t, depth=high_depth_t, intrinsics=high_intr)
    low = CacheHandle(rgb=low_rgb_t, depth=low_depth_t, intrinsics=low_intr)
    # Training-format representations are memoized lazily by tier binding;
    # source-format caches remain compact until a tier is actually consumed.
    high.training_rgb = None
    high.training_depth = None
    low.training_rgb = None
    low.training_depth = None
    return {"high": high, "low": low}
