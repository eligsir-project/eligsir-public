"""
Ordered RGB-D frame sources for streaming replay simulation.

Each source reads dataset metadata upfront (file paths + poses + intrinsics,
no image pixels) and yields StreamingRGBDFrame records in temporal order.
Images are loaded lazily by StreamingScene when a frame "arrives".

Use make_frame_source() to auto-detect the dataset type and return the
appropriate source for a given dataset path.
"""
from __future__ import annotations

import io
import os
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np

from eligsir.utils.rgbd_frames import CameraIntrinsics, reproject_depth_to_color
from eligsir.io.sources.rosbag import _TfGraph, _transform_to_matrix
from eligsir.io.sources.rosbag_sync import (
    StampedMsg,
    SyncStats,
    SyncedRGBD,
    attach_pose,
    check_p95_within_threshold,
    estimate_stream_offset_ns,
    format_stats_oneline,
    merge_stats,
    sync_color_depth_unique,
    write_sync_report,
)
from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
    load_frame_rgb,
    frame_has_pose,
    load_frame_depth_np,
    set_decode_cache,
    get_decode_cache,
    FrameDecodeCache,
    streaming_frame_from_synced_rgbd,
    resize_rgbd_frame,
)
from eligsir.io.sources.debug_viz import (
    _rot_to_quat,
    _quat_to_rot,
    _slerp,
    _depth_to_vis_rgb,
    _depth_edge_mask,
    _resize_mask_nearest,
    _overlay_depth_edges,
    _add_label_bar,
    _write_depth_color_alignment_debug,
)
from eligsir.io.sources.datasets.rgbd_sequence import RGBDSequenceFrameSource
from eligsir.io.sources.datasets.tum import TUMFrameSource
from eligsir.io.sources.datasets.scannet import ScanNetFrameSource, ScanNetSensFrameSource
from eligsir.io.sources.datasets.replica import ReplicaFrameSource
from eligsir.io.sources.datasets.hypersim import HyperSimFrameSource
from eligsir.io.sources.datasets.orbbec_export import (
    OrbbecExportFrameSource,
    _parse_icl_poses,
    _icl_frame_index,
)
from eligsir.io.sources.datasets.orbbec_rosbag import OrbbecRosBagFrameSource, _ROS1_BAG_PROFILES


def _ros1_bag_topics(path: str) -> set[str]:
    try:
        from rosbags.highlevel import AnyReader
        with AnyReader([Path(path)]) as reader:
            return {c.topic for c in reader.connections}
    except Exception:
        return set()


def _select_ros1_bag_profile(path: str, profile: str) -> str:
    profile = str(profile or "auto").lower()
    if profile in _ROS1_BAG_PROFILES:
        return profile
    topics = _ros1_bag_topics(path)
    tum_topics = {
        _ROS1_BAG_PROFILES["tum"]["color_topic"],
        _ROS1_BAG_PROFILES["tum"]["depth_topic"],
        _ROS1_BAG_PROFILES["tum"]["color_info_topic"],
    }
    if tum_topics.issubset(topics):
        return "tum"
    return "realsense"


def _parse_ms_offset(value, default: float = 0.0) -> float:
    text = str(value).strip().lower()
    if not text or text == "auto":
        return default
    return float(text)


from eligsir.io.sources.datasets.realsense_rosbag import RealsenseRosBagFrameSource
from eligsir.io.sources.datasets.ros2_live import (
    ROS2LiveStreamingFrameSource,
    SourceFailedError,
)


def resize_rgbd_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    divisor: int,
) -> tuple:
    """Resize an RGB-D frame + intrinsics by an integer divisor.

    RGB uses area-averaging (best for downscale quality). Depth uses
    nearest-neighbor so depth values are NEVER blended across
    discontinuities -- interpolating depth produces false intermediate
    values at object boundaries. divisor=1 is a no-op passthrough.
    """
    _R = max(1, int(divisor))
    if _R <= 1:
        return rgb, depth, intrinsics
    import cv2 as _cv2
    width, height = int(intrinsics.width), int(intrinsics.height)
    new_w = max(1, width // _R)
    new_h = max(1, height // _R)
    out_rgb = _cv2.resize(rgb, (new_w, new_h), interpolation=_cv2.INTER_AREA)
    out_depth = _cv2.resize(depth, (new_w, new_h), interpolation=_cv2.INTER_NEAREST)
    sx, sy = new_w / float(width), new_h / float(height)
    new_cx = (intrinsics.cx + 0.5) * sx - 0.5
    new_cy = (intrinsics.cy + 0.5) * sy - 0.5
    from dataclasses import replace
    out_intrinsics = replace(
        intrinsics,
        width=new_w, height=new_h,
        fx=intrinsics.fx * sx, fy=intrinsics.fy * sy,
        cx=new_cx, cy=new_cy,
    )
    return out_rgb, out_depth, out_intrinsics


def make_frame_source(source_path: str, args) -> "RGBDSequenceFrameSource | TUMFrameSource | ScanNetFrameSource | ScanNetSensFrameSource | ReplicaFrameSource | HyperSimFrameSource | OrbbecExportFrameSource | OrbbecRosBagFrameSource":
    """Auto-detect the RGB-D dataset layout and return an ordered frame source.

    Compatibility shim: the canonical source-detection/factory logic now
    lives in ``eligsir.io.sources.registry.make_frame_source``. This
    module still owns the concrete frame-source classes (imported by the
    canonical registry), so this just forwards. Kept for old callers still
    importing from ``utils.streaming_frames`` directly.
    """

    from eligsir.io.sources.registry import make_frame_source as _make_frame_source

    return _make_frame_source(source_path, args)
