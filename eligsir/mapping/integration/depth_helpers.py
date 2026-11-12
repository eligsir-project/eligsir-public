"""Canonical home for sensor-depth and surface-prior helpers."""

from __future__ import annotations

import os
from collections import deque
from contextlib import nullcontext
from typing import Optional
from weakref import WeakValueDictionary

import numpy as np
import torch

from geometry.sensor_depth import SensorDepthProvider


_DEPTH_UNAVAILABLE_STATS: dict = {}
_SENSOR_DEPTH_CACHE_LRU: deque = deque()
_SENSOR_DEPTH_CACHE_REFS: WeakValueDictionary = WeakValueDictionary()
_SENSOR_DEPTH_CACHE_MAX_FRAMES: int = 0


def set_sensor_depth_cache_bound(max_cached_frames: int) -> None:
    """Configure the process-local sensor-depth cache bound; 0 is unbounded."""

    global _SENSOR_DEPTH_CACHE_MAX_FRAMES
    _SENSOR_DEPTH_CACHE_MAX_FRAMES = int(max_cached_frames)


def reset_sensor_depth_cache_state() -> None:
    """Drop process-global depth-cache references at a runtime boundary."""

    for cam in list(_SENSOR_DEPTH_CACHE_REFS.values()):
        try:
            cam._sensor_depth_cache = None
        except Exception:
            pass
    _SENSOR_DEPTH_CACHE_LRU.clear()
    _SENSOR_DEPTH_CACHE_REFS.clear()
    _DEPTH_UNAVAILABLE_STATS.clear()
    set_sensor_depth_cache_bound(0)


def _enforce_sensor_depth_cache_bound(max_cached_frames: int) -> None:
    """Evict oldest resident depth tensors until the configured bound holds."""

    if max_cached_frames == 0:
        return
    resident_count = 0
    live_entries = []
    while _SENSOR_DEPTH_CACHE_LRU:
        uid, = _SENSOR_DEPTH_CACHE_LRU.popleft()
        cam = _SENSOR_DEPTH_CACHE_REFS.get(uid)
        if cam is None:
            continue
        cache = getattr(cam, "_sensor_depth_cache", None)
        if cache is not False and cache is not None:
            resident_count += 1
            live_entries.append((uid,))
    _SENSOR_DEPTH_CACHE_LRU.extend(live_entries)
    if resident_count <= max_cached_frames:
        return

    to_evict = resident_count - max_cached_frames
    remaining_entries = []
    while _SENSOR_DEPTH_CACHE_LRU:
        uid, = _SENSOR_DEPTH_CACHE_LRU.popleft()
        if to_evict > 0:
            cam = _SENSOR_DEPTH_CACHE_REFS.get(uid)
            if cam is not None and getattr(cam, "_sensor_depth_cache", None) is not False:
                cam._sensor_depth_cache = None
                to_evict -= 1
        else:
            remaining_entries.append((uid,))
    _SENSOR_DEPTH_CACHE_LRU.extend(remaining_entries)


def _record_sensor_depth_cache_load(cam) -> None:
    """Record one real cache load and enforce the configured LRU bound."""

    cam_uid = getattr(cam, "uid", None)
    if cam_uid is None:
        return
    _SENSOR_DEPTH_CACHE_REFS[cam_uid] = cam
    if not any(uid == cam_uid for (uid,) in _SENSOR_DEPTH_CACHE_LRU):
        _SENSOR_DEPTH_CACHE_LRU.append((cam_uid,))
    if _SENSOR_DEPTH_CACHE_MAX_FRAMES:
        _enforce_sensor_depth_cache_bound(_SENSOR_DEPTH_CACHE_MAX_FRAMES)


def _note_depth_unavailable(cam, reason: str, *, latched: bool) -> None:
    """Count no-depth outcomes and report on an escalating cadence."""

    n = _DEPTH_UNAVAILABLE_STATS.get(reason, 0) + 1
    _DEPTH_UNAVAILABLE_STATS[reason] = n
    if n in (1, 10, 100, 1000, 10000, 100000, 1000000):
        name = getattr(cam, "image_name", None) or getattr(cam, "uid", "?")
        state = "LATCHED (this camera will not retry)" if latched else "not latched (will retry)"
        print(
            f"[sensor-depth] no depth for camera {name} ({reason}) -- {state} [occurrence #{n}]",
            flush=True,
        )


def _get_resident_sensor_depth(cam) -> Optional[torch.Tensor]:
    """Inspect depth without loading, resizing, cache mutation or absence logging."""
    cache = getattr(cam, "_sensor_depth_cache", None)
    if cache is False:
        return None
    if cache is None:
        frame = getattr(cam, "_streaming_frame", None)
        buffered = hasattr(cam, "_streaming_depth_path") and (
            getattr(frame, "_depth_np", None) is not None
            or getattr(frame, "_depth_bytes", None) is not None
        )
        if (
            callable(getattr(cam, "_sensor_depth_loader", None))
            or getattr(cam, "_streaming_depth_path", None) is not None
            or buffered
        ):
            raise ValueError("prepared camera auxiliary 'sensor_depth' is not resident")
        return None
    if cache.ndim != 2 and (cache.ndim != 3 or cache.shape[0] != 1):
        raise ValueError("cached sensor depth must be [H,W] or [1,H,W]")
    return cache


def _get_sensor_depth(cam, target_h: int, target_w: int) -> Optional[torch.Tensor]:
    """Return [1,H,W] metric sensor depth on its cache-owned device.

    Resident depth is authoritative regardless of how it was produced. A cache
    miss may use the legacy streaming/lazy producer paths, which cache on CPU.
    """

    cache = getattr(cam, "_sensor_depth_cache", None)
    if cache is False:
        return None
    if cache is not None:
        cache = _get_resident_sensor_depth(cam)
        cached = cache.unsqueeze(0) if cache.ndim == 2 else cache
        if cached.shape[-2:] != (target_h, target_w):
            return torch.nn.functional.interpolate(
                cached[None], (target_h, target_w), mode="nearest"
            )[0]
        return cached

    depth_loader = getattr(cam, "_sensor_depth_loader", None)
    if not hasattr(cam, "_streaming_depth_path") and not callable(depth_loader):
        return None

    try:
        frame = getattr(cam, "_streaming_frame", None)
        if callable(depth_loader):
            depth_raw = depth_loader()
        elif frame is not None and (
            getattr(frame, "_depth_np", None) is not None
            or getattr(frame, "_depth_bytes", None) is not None
        ):
            from eligsir.io.sources.decode_cache import load_frame_depth_np

            depth_raw = load_frame_depth_np(frame)
        else:
            depth_path = cam._streaming_depth_path
            if depth_path is None:
                _note_depth_unavailable(cam, "no_depth_payload", latched=False)
                return None
            if not os.path.exists(depth_path):
                cam._sensor_depth_cache = False
                _note_depth_unavailable(cam, "depth_path_missing", latched=True)
                return None
            from PIL import Image as _Image

            depth_raw = np.array(_Image.open(depth_path))
        if depth_raw is None:
            cam._sensor_depth_cache = False
            _note_depth_unavailable(cam, "depth_decode_returned_none", latched=True)
            return None
        from eligsir.utils.rgbd_frames import depth_to_meters

        depth_m = depth_to_meters(depth_raw, cam._streaming_depth_scale).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - preserve fail-open depth semantics
        cam._sensor_depth_cache = False
        _note_depth_unavailable(cam, f"exception:{type(exc).__name__}", latched=True)
        return None

    depth = torch.from_numpy(depth_m)[None, None]
    if depth.shape[2:] != (target_h, target_w):
        depth = torch.nn.functional.interpolate(depth, (target_h, target_w), mode="nearest")
    cam._sensor_depth_cache = depth[0]
    _record_sensor_depth_cache_load(cam)
    return cam._sensor_depth_cache


def _get_surface_prior(cam, args, *, timed=None, resident_only=False):
    """Return the matching prior; residency inspection never predicts or caches."""

    frame = getattr(cam, "_streaming_frame", None)
    if frame is None:
        return None
    cache_key = (
        int(cam.image_height),
        int(cam.image_width),
        str(getattr(args, "surface_normal_mode", "finite_difference")),
        bool(getattr(args, "streaming_depth_filter_enabled", True)),
    )
    cached = getattr(cam, "_surface_prior_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if resident_only:
        raise ValueError("prepared camera auxiliary 'surface_confidence' is not resident")
    try:
        with (timed("surface_prior_compute") if timed is not None else nullcontext()):
            prior = SensorDepthProvider.from_args(args).predict(frame, cam)
    except Exception:
        return None
    cam._surface_prior_cache = (cache_key, prior)
    return prior
