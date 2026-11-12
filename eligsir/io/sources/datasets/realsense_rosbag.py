"""RealsenseRosBagFrameSource — RealSense .bag frame source via pyrealsense2.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). Subclasses
OrbbecRosBagFrameSource, so it moves right after that extraction per the
scope doc's ordering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)
from eligsir.io.sources.debug_viz import (
    _write_depth_color_alignment_debug,
)
from eligsir.io.sources.datasets.orbbec_rosbag import OrbbecRosBagFrameSource


class RealsenseRosBagFrameSource(OrbbecRosBagFrameSource):
    """Frames from a RealSense .bag file via pyrealsense2 SDK with Open3D live odometry.

    Uses librealsense2's native playback + ``rs.align()`` for hardware-accelerated
    depth-to-color alignment.
    Only ``open3d_odometry_live`` is supported as pose source (no camera_pose
    topic in RealSense Viewer or TUM public bags).

    Parameters marked "rosbags-based" in the constructor are ignored — the SDK
    auto-discovers streams and handles synchronization natively.
    """

    def __init__(
        self,
        path: str,
        color_topic: str = "/device_0/sensor_1/Color_0/image/data",
        depth_topic: str = "/device_0/sensor_0/Depth_0/image/data",
        color_info_topic: str = "/device_0/sensor_1/Color_0/info/camera_info",
        depth_info_topic: str = "/device_0/sensor_0/Depth_0/info/camera_info",
        sync_threshold_ms: float = 33.0,
        max_frames: int = 0,
        frame_stride: int = 1,
        open3d_odom_max_failure_ratio: float = 0.25,
        open3d_odom_cache: bool = True,
        open3d_odom_cache_dir: str = "",
        open3d_odom_stride: int = 1,
        open3d_odom_downscale: int = 1,
        open3d_odom_async: bool = False,
        open3d_odom_async_queue_size: int = 32,
        open3d_odom_motion_prior: bool = False,
        open3d_odom_motion_gate: bool = False,
        open3d_odom_max_trans_per_edge: float = 0.15,
        open3d_odom_max_rot_deg_per_edge: float = 8.0,
        open3d_odom_depth_min: float = 0.1,
        open3d_odom_depth_max: float = 8.0,
        open3d_odom_depth_diff_max: float = 0.07,
        open3d_odom_method: str = "hybrid",
        open3d_icp_max_distance: float = 0.07,
        open3d_icp_robust_kernel: str = "huber",
        open3d_icp_sigma: float = 0.05,
        vbg_voxel_size: float = 0.0058,
        vbg_block_count: int = 40000,
        vbg_depth_max: float = 3.0,
        streaming_resolution: int = 1,
        depth_scale: float = 1000.0,
        sync_estimate_offset: bool = False,
        sync_offset_ms: str | float = "0",
        sync_report_json: str = "",
        rgb_encoding_override: str = "",
        alignment_debug_dir: str = "",
        alignment_debug_frames: int = 0,
        source_label: str = "RealsenseBag",
        profile: str = "realsense",
        realsense_depth_filters: bool = True,
    ):
        # Intentionally bypass OrbbecRosBagFrameSource.__init__ — we use
        # pyrealsense2 native playback instead of rosbags-based reading.
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is required for RealsenseRosBagFrameSource. "
                "Install it with: uv pip install pyrealsense2"
            )
        import numpy as np

        # ── Start pyrealsense2 playback ────────────────────────────────────────
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(str(path), repeat_playback=False)
        cfg.enable_all_streams()
        profile = pipeline.start(cfg)

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        # ── Stream profiles ────────────────────────────────────────────────────
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        _ci = color_stream.get_intrinsics()
        _di = depth_stream.get_intrinsics()
        # pyrealsense2 may provide RGB8 or BGR8; check at init time
        _need_bgr_to_rgb = (color_stream.format() == rs.format.bgr8)

        fx, fy, cx, cy = _ci.fx, _ci.fy, _ci.ppx, _ci.ppy
        width, height = _ci.width, _ci.height
        depth_fx, depth_fy, depth_cx, depth_cy = _di.fx, _di.fy, _di.ppx, _di.ppy
        depth_width, depth_height = _di.width, _di.height

        # ── Extrinsics ────────────────────────────────────────────────────────
        # We align color → depth, so depth is the reference frame and the
        # warped color image lives in depth coordinates. T_color_from_depth is
        # identity after alignment; downstream consumers treat depth intrinsics
        # as the camera intrinsics for both streams.
        T_color_from_depth = np.eye(4, dtype=np.float32)
        self._T_color_from_depth = T_color_from_depth
        self._extrinsic_source = "pyrealsense2_color_to_depth"
        self._depth_alignment_enabled = True

        align = rs.align(rs.stream.depth)
        print(
            f"[streaming] RealSense: aligning color → depth ({depth_width}×{depth_height}); "
            f"native color was {width}×{height}",
            flush=True,
        )

        # ── Optional depth post-processing filters (D400 stereo cameras) ──────
        _use_filters = bool(realsense_depth_filters)
        if _use_filters:
            # Recommended filter chain: disparity → spatial → temporal → depth → hole-fill
            _d2d = rs.disparity_transform(True)
            _spatial = rs.spatial_filter()
            _spatial.set_option(rs.option.filter_smooth_alpha, 0.55)
            _spatial.set_option(rs.option.filter_smooth_delta, 20)
            _temporal = rs.temporal_filter()
            _temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            _temporal.set_option(rs.option.filter_smooth_delta, 20)
            _temporal.set_option(rs.option.holes_fill, 3)
            _d2d_inv = rs.disparity_transform(False)
            _hole_fill = rs.hole_filling_filter()
            _hole_fill.set_option(rs.option.holes_fill, 1)

        # Raw numpy storage — avoids PNG encode/decode roundtrip entirely.
        # load_frame_rgb / load_frame_depth_np check _rgb_np / _depth_np first.

        # ── Read all frames ────────────────────────────────────────────────────
        _frame_stride = max(1, int(frame_stride))
        _max_frames = max(0, int(max_frames))
        need_debug = bool(alignment_debug_dir) and int(alignment_debug_frames) > 0
        debug_frames = max(0, int(alignment_debug_frames))

        self._frames: List[StreamingRGBDFrame] = []
        debug_records: list[dict[str, Any]] = []
        total_count = 0

        try:
            while True:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                total_count += 1

                # Apply frame_stride — only process every Nth frameset
                if (total_count - 1) % _frame_stride != 0:
                    continue

                # Capture unaligned depth for debug before alignment
                orig_depth_frame = frames.get_depth_frame()
                if not orig_depth_frame:
                    continue
                orig_depth_m = np.asanyarray(
                    orig_depth_frame.get_data()
                ).astype(np.float32) * 0.001  # mm → m

                # Align depth to color
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not aligned_depth_frame:
                    continue

                # Numpy arrays (color may be RGB8 or BGR8; check _need_bgr_to_rgb)
                color_img = np.asanyarray(color_frame.get_data())

                # Apply depth post-processing filters if enabled
                if _use_filters:
                    filtered = _d2d.process(aligned_depth_frame)
                    filtered = _spatial.process(filtered)
                    filtered = _temporal.process(filtered)
                    filtered = _d2d_inv.process(filtered)
                    filtered = _hole_fill.process(filtered)
                    depth_for_np = filtered
                else:
                    depth_for_np = aligned_depth_frame
                depth_u16 = np.asanyarray(depth_for_np.get_data())  # uint16

                # Convert BGR→RGB if needed, otherwise data is already RGB
                rgb = color_img[..., ::-1] if _need_bgr_to_rgb else color_img

                # Timestamps (ms → ns)
                color_ts_ns = int(color_frame.get_timestamp() * 1e6)
                depth_ts_ns = int(aligned_depth_frame.get_timestamp() * 1e6)

                # Store copies of the numpy arrays — no PNG encoding needed.
                rgb_np = rgb.copy()
                depth_np = depth_u16.copy()

                # Aligned depth in meters for debug
                aligned_depth_m = depth_u16.astype(np.float32) * 0.001

                # Color aligned to depth → both streams use depth intrinsics
                depth_kwargs = {
                    "depth_fx": depth_fx,
                    "depth_fy": depth_fy,
                    "depth_cx": depth_cx,
                    "depth_cy": depth_cy,
                    "depth_width": int(depth_width),
                    "depth_height": int(depth_height),
                }

                # Debug record
                if need_debug and len(debug_records) < debug_frames:
                    debug_records.append(_write_depth_color_alignment_debug(
                        alignment_debug_dir,
                        len(self._frames),
                        rgb,
                        orig_depth_m,
                        aligned_depth_m,
                        {
                            "bag": str(path),
                            "profile": str(profile),
                            "depth_alignment_enabled": True,
                            "depth_topic_aligned": False,
                            "extrinsic_source": "pyrealsense2",
                            "depth_tf_topic": "",
                            "color_tf_topic": "",
                            "T_color_from_depth": T_color_from_depth.astype(float).tolist(),
                            "color_intrinsics": {
                                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                "width": width, "height": height,
                            },
                            "depth_intrinsics": {
                                "fx": depth_fx, "fy": depth_fy, "cx": depth_cx, "cy": depth_cy,
                                "width": depth_width, "height": depth_height,
                            },
                            "color_ts_ns": color_ts_ns,
                            "depth_ts_ns": depth_ts_ns,
                            "rgb_depth_dt_ns": int(depth_ts_ns - color_ts_ns),
                        },
                    ))

                self._frames.append(StreamingRGBDFrame(
                    index=len(self._frames),
                    timestamp=float(color_ts_ns) * 1e-9,
                    rgb_path="",
                    depth_path=None,
                    c2w=None,
                    fx=depth_fx, fy=depth_fy, cx=depth_cx, cy=depth_cy,
                    width=int(depth_width), height=int(depth_height),
                    depth_scale=1000.0,
                    _rgb_bytes=None,
                    _depth_bytes=None,
                    _rgb_np=rgb_np,
                    _depth_np=depth_np,
                    color_ts_ns=color_ts_ns,
                    depth_ts_ns=depth_ts_ns,
                    rgb_depth_dt_ns=int(depth_ts_ns - color_ts_ns),
                    color_bag_ns=color_ts_ns,
                    depth_bag_ns=depth_ts_ns,
                    **depth_kwargs,
                ))

                if _max_frames > 0 and len(self._frames) >= _max_frames:
                    break

        except RuntimeError:
            pass  # end of playback

        pipeline.stop()

        if need_debug and debug_records:
            import json
            manifest = {
                "bag": str(path),
                "profile": str(profile),
                "depth_alignment_enabled": True,
                "depth_topic_aligned": False,
                "extrinsic_source": "pyrealsense2",
                "T_color_from_depth": T_color_from_depth.astype(float).tolist(),
                "frames": debug_records,
            }
            manifest_path = Path(alignment_debug_dir) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Color is aligned to depth — use depth intrinsics as the camera model.
        self._fx = depth_fx
        self._fy = depth_fy
        self._cx = depth_cx
        self._cy = depth_cy
        self._width = int(depth_width)
        self._height = int(depth_height)
        self._depth_scale = 1000.0

        # ── Resolution downscaling ──────────────────────────────────────────────
        _R = int(streaming_resolution)
        if _R > 1:
            self._width  = self._width  // _R
            self._height = self._height // _R
            self._fx /= _R
            self._fy /= _R
            self._cx /= _R
            self._cy /= _R
            for _fr in self._frames:
                _fr.width  = _fr.width  // _R
                _fr.height = _fr.height // _R
                _fr.fx = _fr.fx / _R
                _fr.fy = _fr.fy / _R
                _fr.cx = _fr.cx / _R
                _fr.cy = _fr.cy / _R

        # ── Live odometry state (mirrors OrbbecRosBagFrameSource) ──────────────
        self._pose_source = "open3d_odometry_live"
        self._live_odom_enabled = True
        self._live_odom_stride = max(1, int(open3d_odom_stride))
        self._live_odom_downscale = max(1, int(open3d_odom_downscale))
        self._live_odom_max_failure_ratio = float(open3d_odom_max_failure_ratio)
        self._live_odom_motion_prior = bool(open3d_odom_motion_prior)
        self._live_odom_motion_gate = bool(open3d_odom_motion_gate)
        self._live_odom_max_trans = float(open3d_odom_max_trans_per_edge) if open3d_odom_max_trans_per_edge > 0 else float("inf")
        self._live_odom_max_rot = float(open3d_odom_max_rot_deg_per_edge) if open3d_odom_max_rot_deg_per_edge > 0 else float("inf")
        self._odom_depth_min = float(open3d_odom_depth_min)
        self._odom_depth_max = float(open3d_odom_depth_max)
        self._odom_depth_diff_max = float(open3d_odom_depth_diff_max)
        self._live_odom_method = open3d_odom_method
        self._live_odom_icp_max_distance = open3d_icp_max_distance
        self._live_odom_icp_robust_kernel = open3d_icp_robust_kernel
        self._live_odom_icp_sigma = open3d_icp_sigma
        self._live_odom_vbg_voxel_size = float(vbg_voxel_size)
        self._live_odom_vbg_block_count = int(vbg_block_count)
        self._live_odom_vbg_depth_max = float(vbg_depth_max)
        self._vbg_model = None  # lazily created on first VBG frame
        self._vbg_frame_obj = None
        self._vbg_model_frame = None
        self._live_odom_successes = 0
        self._live_odom_failures = 0
        self._live_odom_pairs = 0
        self._live_odom_computed_idx = -1
        self._live_odom_prev_key_idx = 0
        self._live_odom_prev_key_rgbd = None
        self._live_odom_last_trans: np.ndarray | None = None
        self._live_odom_last_n_edges: int = 1
        self._live_odom_cache_path = None
        self._live_odom_cache_key = None
        self._live_odom_cache_enabled = bool(open3d_odom_cache)
        self._live_odom_cached_full = False
        self._live_odom_async_enabled = bool(open3d_odom_async)
        self._live_odom_async_done = False
        self._live_odom_async_error = None
        self._live_odom_async_thread = None
        self._live_odom_cond = None
        # Phase 5: producer-consumer backpressure
        self._live_odom_consumer_idx = 0
        self._live_odom_queue_size = max(1, int(open3d_odom_async_queue_size))
        self._live_odom_async_stop = False

        if self._frames:
            self._frames[0].c2w = np.eye(4, dtype=np.float32)

        live_cache_key = self._open3d_odom_cache_key(
            path=path,
            color_topic="pyrealsense2",
            depth_topic="pyrealsense2",
            camera_info_topic="pyrealsense2",
            sync_threshold_ms=0.0,
            frame_stride=frame_stride,
            max_frames=max_frames,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            synced=[(fr.color_ts_ns,) for fr in self._frames],
            odom_stride=open3d_odom_stride,
            odom_downscale=open3d_odom_downscale,
            odom_method=open3d_odom_method,
            icp_max_distance=open3d_icp_max_distance,
            icp_robust_kernel=open3d_icp_robust_kernel,
            icp_sigma=open3d_icp_sigma,
            motion_prior=open3d_odom_motion_prior,
            motion_gate=open3d_odom_motion_gate,
            vbg_voxel_size=vbg_voxel_size,
            vbg_block_count=vbg_block_count,
            vbg_depth_max=vbg_depth_max,
            mode="live",
            depth_scale=1000.0,
            depth_info_topic="pyrealsense2",
            sync_offset_ns=0,
            profile=profile,
            depth_alignment={
                "enabled": True,
                "topic_aligned": False,
                "extrinsic_source": "pyrealsense2",
                "T_color_from_depth": [
                    round(float(v), 8) for v in T_color_from_depth.reshape(-1)
                ],
            },
        )
        live_cache_path = self._open3d_odom_cache_path(
            bag_path=str(Path(path).parent),
            cache_dir=open3d_odom_cache_dir,
            cache_key=live_cache_key,
        )
        self._live_odom_cache_key = live_cache_key
        self._live_odom_cache_path = live_cache_path
        if open3d_odom_cache:
            cached_poses, cached_stats = self._load_open3d_odom_cache(
                live_cache_path, live_cache_key, expected_frames=len(self._frames)
            )
            if cached_poses is not None:
                for idx, pose in enumerate(cached_poses):
                    self._frames[idx].c2w = pose
                self._live_odom_cached_full = True
                self._live_odom_computed_idx = len(self._frames) - 1
                self._live_odom_successes = int(cached_stats.get("successes", 0))
                self._live_odom_failures = int(cached_stats.get("failures", 0))
                self._live_odom_pairs = self._live_odom_successes + self._live_odom_failures
        if self._live_odom_enabled and not self._live_odom_cached_full and self._frames and self._live_odom_async_enabled:
            self._start_live_odom_worker()

        _t = T_color_from_depth[:3, 3]
        debug_msg = f", alignment_debug={alignment_debug_dir}" if debug_records else ""
        filters_msg = f", depth_filters=enabled" if _use_filters else ", depth_filters=off"
        print(
            f"[streaming] {source_label}: {len(self._frames)} frames from {path} "
            f"(pyrealsense2 playback, pose_source=open3d_odometry_live, "
            f"motion_prior={self._live_odom_motion_prior}, "
            f"motion_gate={self._live_odom_motion_gate}, "
            f"depth_align=enabled "
            f"extrinsic_source=pyrealsense2 t_cd=({_t[0]:.5f},{_t[1]:.5f},{_t[2]:.5f}) "
            f"color={width}x{height}{filters_msg}{debug_msg})"
        )
