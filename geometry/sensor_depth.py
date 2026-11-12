"""Sensor-depth SurfacePrior provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from geometry.normals import NormalEstimationConfig, estimate_depth_normals
from geometry.priors import GeometrySource, SurfacePrior


@dataclass(frozen=True)
class SensorDepthConfig:
    min_depth: float = 0.1
    max_depth: float = 8.0
    filter_enabled: bool = True
    filter_window: int = 5
    filter_min_valid_ratio: float = 0.75
    filter_max_range: float = 0.08
    filter_median_thresh: float = 0.04
    normal_mode: str = "finite_difference"
    normal_window: int = 5
    normal_max_depth_jump: float = 0.05
    normal_min_valid_ratio: float = 0.6


def resize_nearest_np(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if arr.shape[:2] == (target_h, target_w):
        return arr
    src_h, src_w = arr.shape[:2]
    ys = np.clip(np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int32), 0, src_h - 1)
    xs = np.clip(np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int32), 0, src_w - 1)
    return arr[ys[:, None], xs[None, :]]


def stable_depth_mask(
    depth_m: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    window: int,
    min_valid_ratio: float,
    max_range: float,
    median_thresh: float,
) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    radius = max(0, int(window) // 2)
    if radius <= 0:
        return valid
    local_depths = []
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]
    for dy in range(-radius, radius + 1):
        yy = np.clip(ys + dy, 0, h - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(xs + dx, 0, w - 1)
            local_depths.append(z[yy, xx])
    local = np.stack(local_depths, axis=0)
    local_valid = np.isfinite(local) & (local > min_depth) & (local < max_depth)
    stable = valid & (local_valid.mean(axis=0) >= min_valid_ratio)
    if max_range > 0:
        local_min = np.where(local_valid, local, np.inf).min(axis=0)
        local_max = np.where(local_valid, local, -np.inf).max(axis=0)
        stable &= np.isfinite(local_min) & np.isfinite(local_max) & ((local_max - local_min) <= max_range)
    if median_thresh > 0:
        import warnings
        local_nan = np.where(local_valid, local, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(local_nan, axis=0)
        stable &= np.isfinite(median) & (np.abs(z - median) <= median_thresh)
    return stable


def load_frame_depth_meters(frame) -> Optional[np.ndarray]:
    try:
        from eligsir.utils.rgbd_frames import depth_to_meters
        from eligsir.utils.streaming_frames import load_frame_depth_np
        depth_raw = load_frame_depth_np(frame)
        if depth_raw is None:
            return None
        return depth_to_meters(np.asarray(depth_raw), frame.depth_scale).astype(np.float32)
    except Exception:
        return None


class SensorDepthProvider:
    def __init__(self, config: Optional[SensorDepthConfig] = None) -> None:
        self.config = config or SensorDepthConfig()

    @classmethod
    def from_args(cls, args) -> "SensorDepthProvider":
        return cls(
            SensorDepthConfig(
                min_depth=float(getattr(args, "streaming_min_depth", 0.1)),
                max_depth=float(getattr(args, "streaming_max_depth", 8.0)),
                filter_enabled=bool(args.streaming_depth_filter_enabled),
                filter_window=int(getattr(args, "streaming_depth_filter_window", 5)),
                filter_min_valid_ratio=float(getattr(args, "streaming_depth_filter_min_valid_ratio", 0.75)),
                filter_max_range=float(getattr(args, "streaming_depth_filter_max_range", 0.08)),
                filter_median_thresh=float(getattr(args, "streaming_depth_filter_median_thresh", 0.04)),
                normal_mode=str(getattr(args, "surface_normal_mode", "finite_difference")),
                normal_window=int(getattr(args, "surface_normal_window", 5)),
                normal_max_depth_jump=float(getattr(args, "surface_normal_max_depth_jump", 0.05)),
                normal_min_valid_ratio=float(getattr(args, "surface_normal_min_valid_ratio", 0.6)),
            )
        )

    def available(self) -> bool:
        return True

    def predict(self, frame, cam=None) -> SurfacePrior:
        depth = load_frame_depth_meters(frame)
        if depth is None:
            raise ValueError("Frame has no loadable sensor depth")
        if cam is not None:
            target_hw = (int(cam.image_height), int(cam.image_width))
        else:
            target_hw = (int(frame.height), int(frame.width))
        depth = resize_nearest_np(depth, target_hw)
        cfg = self.config
        valid = np.isfinite(depth) & (depth > cfg.min_depth) & (depth < cfg.max_depth)
        confidence = valid.astype(np.float32)
        if cfg.filter_enabled:
            stable = stable_depth_mask(
                depth,
                min_depth=cfg.min_depth,
                max_depth=cfg.max_depth,
                window=cfg.filter_window,
                min_valid_ratio=cfg.filter_min_valid_ratio,
                max_range=cfg.filter_max_range,
                median_thresh=cfg.filter_median_thresh,
            )
            valid &= stable
            confidence *= stable.astype(np.float32)

        fx = float(getattr(frame, "depth_fx", None) or getattr(frame, "fx"))
        fy = float(getattr(frame, "depth_fy", None) or getattr(frame, "fy"))
        cx = float(getattr(frame, "depth_cx", None) or getattr(frame, "cx"))
        cy = float(getattr(frame, "depth_cy", None) or getattr(frame, "cy"))
        normals, normal_conf = estimate_depth_normals(
            depth,
            fx,
            fy,
            cx,
            cy,
            NormalEstimationConfig(
                mode=cfg.normal_mode,
                window=cfg.normal_window,
                max_depth_jump=cfg.normal_max_depth_jump,
                min_valid_ratio=cfg.normal_min_valid_ratio,
                min_depth=cfg.min_depth,
                max_depth=cfg.max_depth,
            ),
        )
        confidence *= normal_conf
        source = np.full(depth.shape, int(GeometrySource.SENSOR), dtype=np.int16)
        return SurfacePrior(
            depth=torch.from_numpy(depth.astype(np.float32))[None],
            confidence=torch.from_numpy(confidence.astype(np.float32))[None],
            valid=torch.from_numpy(valid.astype(bool))[None],
            normals=torch.from_numpy(normals.astype(np.float32)),
            uncertainty=torch.from_numpy((1.0 - confidence).astype(np.float32))[None],
            source=torch.from_numpy(source)[None],
            frame_uid=int(getattr(frame, "index", 0)),
            normal_frame="camera",
        )
