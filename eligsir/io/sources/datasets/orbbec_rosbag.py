"""OrbbecRosBagFrameSource — Orbbec Femto Bolt slam MCAP/rosbag2 frame source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). By far the largest
FrameSource (~2050 lines), split first in isolation per that plan since
it is the highest-risk single move.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np

from eligsir.utils.rgbd_frames import CameraIntrinsics, reproject_depth_to_color
from eligsir.io.sources.rosbag import _TfGraph, _transform_to_matrix
from eligsir.io.sources.rosbag_sync import (
    StampedMsg,
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
    streaming_frame_from_synced_rgbd,
)
from eligsir.io.sources.debug_viz import (
    _rot_to_quat,
    _quat_to_rot,
    _slerp,
)


def _resize_png_payload(
    payload: bytes, *, width: int, height: int, resample: int
) -> bytes:
    """Resize one encoded RGB or depth image without changing its pixel type."""
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        resized = image.resize((int(width), int(height)), resample=resample)
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return buffer.getvalue()


def _scale_intrinsics_for_resize(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, float, float, float]:
    scale_x = int(target_width) / int(source_width)
    scale_y = int(target_height) / int(source_height)
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


class OrbbecRosBagFrameSource:
    """Frames from an Orbbec Femto Bolt slam MCAP rosbag (rosbag2 format).

    Requires a slam bag that publishes ``/camera_pose``
    (``geometry_msgs/msg/PoseStamped``). Pure navigation bags (no
    ``/camera_pose``) are not supported — raise a clear error instead.

    Topic conventions (Orbbec ROS2 nodes, default namespace)::
        /camera_pose                         — PoseStamped (map frame, c2w)
        /camera/color/image_raw/compressed   — CompressedImage (JPEG)
        /camera/depth/image_raw/compressed   — CompressedImage (16UC1 PNG, mm)
        /camera/color/camera_info            — CameraInfo

    .. note::
        Always pass ``--streaming_camera_profile orbbec_femto_bolt`` when
        training on bags from this source.  Without it, Kinect v1 depth
        defaults are used, which clip valid Femto Bolt depth range and
        introduce floaters.
    """

    RECOMMENDED_CAMERA_PROFILE = "orbbec_femto_bolt"

    def __init__(
        self,
        path: str,
        color_topic: str = "/camera/color/image_raw/compressed",
        depth_topic: str = "/camera/depth/image_raw/compressed",
        pose_topic: str = "/camera_pose",
        pose_source: str = "camera_pose",
        camera_info_topic: str = "/camera/color/camera_info",
        sync_threshold_ms: float = 5.0,
        pose_sync_threshold_ms: float = 10.0,
        sync_estimate_offset: bool = False,
        sync_offset_ms: str | float = "auto",
        sync_strict: bool = True,
        sync_report_json: str = "",
        model_path: str = "",
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
    ):
        try:
            from rosbags.rosbag2 import Reader
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            raise ImportError(
                "rosbags is required for OrbbecRosBagFrameSource. "
                "Install it with: uv pip install rosbags"
            )
        import numpy as np

        is_single_mcap = os.path.isfile(path) and str(path).lower().endswith(".mcap")

        def _mcap_topics(mcap_path: str) -> set[str]:
            try:
                from mcap.records import Channel, Message
                from mcap.stream_reader import StreamReader
            except ImportError as exc:
                raise ImportError(
                    "Single-file MCAP input requires optional MCAP packages. "
                    "Install them with: pip install mcap mcap-ros2-support"
                ) from exc

            topics_found: set[str] = set()
            try:
                with open(mcap_path, "rb") as stream:
                    for record in StreamReader(stream, skip_magic=False).records:
                        if isinstance(record, Channel):
                            topics_found.add(record.topic)
                        elif isinstance(record, Message):
                            break
            except Exception as exc:
                # Some recovered MCAPs are missing the footer. The stream reader
                # still yields records up to EOF; treat that as a usable scan.
                if type(exc).__name__ != "EndOfFile":
                    raise
            return topics_found

        def _iter_selected_messages(want_topics: list[str]):
            want_set = set(want_topics)
            if is_single_mcap:
                try:
                    from mcap.records import Channel, Message, Schema
                    from mcap.stream_reader import StreamReader
                    from mcap_ros2.decoder import DecoderFactory
                except ImportError as exc:
                    raise ImportError(
                        "Single-file MCAP input requires optional MCAP packages. "
                        "Install them with: pip install mcap mcap-ros2-support"
                    ) from exc

                schemas = {}
                channels = {}
                decoders = {}
                decoder_factory = DecoderFactory()
                try:
                    with open(path, "rb") as stream:
                        for record in StreamReader(stream, skip_magic=False).records:
                            if isinstance(record, Schema):
                                schemas[record.id] = record
                            elif isinstance(record, Channel):
                                channels[record.id] = record
                            elif isinstance(record, Message):
                                channel = channels.get(record.channel_id)
                                if channel is None or channel.topic not in want_set:
                                    continue
                                schema = schemas[channel.schema_id]
                                key = (channel.message_encoding, schema.id)
                                decoder = decoders.get(key)
                                if decoder is None:
                                    decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
                                    decoders[key] = decoder
                                yield channel.topic, int(record.log_time), decoder(record.data)
                except Exception as exc:
                    if type(exc).__name__ != "EndOfFile":
                        raise
                return

            import struct as _struct
            with Reader(path) as reader:
                conns = [c for c in reader.connections if c.topic in want_set]
                n_trunc = 0
                try:
                    for conn, ts, data in reader.messages(connections=conns):
                        try:
                            yield conn.topic, int(ts), typestore.deserialize_cdr(data, conn.msgtype)
                        except (ValueError, _struct.error):
                            n_trunc += 1
                except (TypeError, OSError, EOFError):
                    # Truncated bag: reader hit EOF mid-record (interrupted recording)
                    n_trunc += 1
                if n_trunc:
                    print(
                        f"[streaming] OrbbecRosBag: skipped {n_trunc} truncated/corrupt messages "
                        f"in {os.path.basename(str(path))!r}",
                        flush=True,
                    )

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        sync_ns = int(sync_threshold_ms * 1e6)
        pose_sync_ns = int(pose_sync_threshold_ms * 1e6)
        pose_source = str(pose_source or "camera_pose")
        open3d_odom_stride = max(1, int(open3d_odom_stride))
        open3d_odom_downscale = max(1, int(open3d_odom_downscale))
        self._odom_depth_min = float(open3d_odom_depth_min)
        self._odom_depth_max = float(open3d_odom_depth_max)
        self._odom_depth_diff_max = float(open3d_odom_depth_diff_max)
        if pose_source not in {"camera_pose", "auto", "open3d_odometry", "open3d_odometry_live"}:
            raise ValueError(
                "orbbec_pose_source must be one of: camera_pose, auto, open3d_odometry, "
                "open3d_odometry_live "
                f"(got '{pose_source}')"
            )

        # First pass: inspect topics and choose the pose source.
        if is_single_mcap:
            topics = _mcap_topics(path)
        else:
            with Reader(path) as reader:
                topics = {c.topic for c in reader.connections}
        has_pose_topic = pose_topic in topics
        if pose_source == "auto":
            if not has_pose_topic:
                from eligsir.slam.pose_provenance import flag_orbbec_auto_pose_fallback

                flag_orbbec_auto_pose_fallback(pose_topic)
            pose_source = "camera_pose" if has_pose_topic else "open3d_odometry_live"
        if pose_source == "camera_pose" and not has_pose_topic:
            raise ValueError(
                f"[streaming] OrbbecRosBagFrameSource: no '{pose_topic}' topic found in {path}.\n"
                f"  Available topics: {sorted(topics)}\n"
                "  Pass --orbbec_pose_source open3d_odometry_live to estimate poses from RGB-D, "
                "or --orbbec_pose_source auto to use /camera_pose when present and Open3D otherwise."
            )

        # Read all messages in one pass.
        pose_msgs: list[StampedMsg] = []
        color_msgs: list[StampedMsg] = []
        depth_msgs: list[StampedMsg] = []
        camera_infos: list[dict] = []
        depth_camera_infos: list[dict] = []
        tf_graph = _TfGraph()
        intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy, W, H)
        depth_intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy, W, H)

        _msg_ns_zero_warned: set = set()

        def _raw_image_to_png(msg) -> bytes:
            """Encode a raw sensor_msgs/Image to PNG bytes for PIL compatibility."""
            from PIL import Image as _PILImage
            h, w = int(msg.height), int(msg.width)
            enc = str(getattr(msg, "encoding", "")).lower()
            raw = bytes(msg.data)
            if enc in ("16uc1", "16sc1", "mono16"):
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
                pil = _PILImage.fromarray(arr, mode="I;16")
            elif enc in ("bgr8", "rgb8", "8uc3"):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                if enc == "bgr8":
                    arr = arr[:, :, ::-1]  # BGR→RGB
                pil = _PILImage.fromarray(arr, mode="RGB")
            elif enc in ("mono8", "8uc1"):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
                pil = _PILImage.fromarray(arr, mode="L")
            else:
                # Unknown encoding — attempt to return raw and let PIL try
                return raw
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return buf.getvalue()

        def _msg_is_raw_image(msg) -> bool:
            return bool(getattr(msg, "encoding", None))

        def _msg_ns(msg, fallback_ns, topic=""):
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None)
            if stamp is None:
                return fallback_ns
            sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
            nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0)))
            if sec == 0 and nsec == 0:
                if topic not in _msg_ns_zero_warned:
                    _msg_ns_zero_warned.add(topic)
                    print(
                        f"[streaming] Warning: {topic!r} has unset header.stamp (sec=0, nsec=0); "
                        "falling back to bag receive timestamp for this topic.",
                        flush=True,
                    )
                return fallback_ns
            return sec * 1_000_000_000 + nsec

        def _frame_id(msg) -> str:
            header = getattr(msg, "header", None)
            return str(getattr(header, "frame_id", "") or "")

        depth_info_topic = "/camera/depth/camera_info"
        _distortion_coeffs: list = []
        want = [color_topic, depth_topic, camera_info_topic, depth_info_topic, "/tf_static"]
        if pose_source == "camera_pose":
            want.append(pose_topic)
        for topic, ts, msg in _iter_selected_messages(want):
            if topic == pose_topic:
                msg_ts = _msg_ns(msg, ts, topic)
                p = msg.pose.position
                o = msg.pose.orientation
                qx, qy, qz, qw = o.x, o.y, o.z, o.w
                R = np.array([
                    [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),  1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx*qx+qy*qy)],
                ], dtype=np.float32)
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R
                c2w[:3, 3] = [p.x, p.y, p.z]
                pose_msgs.append(StampedMsg(
                    topic=topic,
                    header_ns=msg_ts,
                    bag_ns=int(ts),
                    payload=c2w,
                    frame_id=_frame_id(msg),
                ))
            elif topic == color_topic:
                msg_ts = _msg_ns(msg, ts, topic)
                encoding = str(getattr(msg, "encoding", "") or "")
                fmt = str(getattr(msg, "format", "") or "")
                if _msg_is_raw_image(msg):
                    payload = _raw_image_to_png(msg)
                else:
                    payload = bytes(msg.data)
                color_msgs.append(StampedMsg(
                    topic=topic,
                    header_ns=msg_ts,
                    bag_ns=int(ts),
                    payload=payload,
                    frame_id=_frame_id(msg),
                    encoding=encoding,
                    format=fmt,
                ))
            elif topic == depth_topic:
                msg_ts = _msg_ns(msg, ts, topic)
                encoding = str(getattr(msg, "encoding", "") or "")
                fmt = str(getattr(msg, "format", "") or "")
                if _msg_is_raw_image(msg):
                    payload = _raw_image_to_png(msg)
                else:
                    if fmt and "png" not in fmt.lower() and "16uc1" not in fmt.lower():
                        raise RuntimeError(
                            f"Depth topic '{depth_topic}' has format '{fmt}' — "
                            "expected lossless PNG/16UC1, not JPEG. "
                            "JPEG-compressed depth corrupts metric values."
                        )
                    payload = bytes(msg.data)
                depth_msgs.append(StampedMsg(
                    topic=topic,
                    header_ns=msg_ts,
                    bag_ns=int(ts),
                    payload=payload,
                    frame_id=_frame_id(msg),
                    encoding=encoding,
                    format=fmt,
                ))
            elif topic == camera_info_topic:
                K = msg.k  # row-major 3×3
                _d_raw = getattr(msg, "d", None)
                D = list(_d_raw) if _d_raw is not None and len(_d_raw) > 0 else []
                info = {
                    "topic": topic,
                    "frame_id": _frame_id(msg),
                    "header_ns": _msg_ns(msg, ts, topic),
                    "bag_ns": int(ts),
                    "K": [float(v) for v in K],
                    "D": [float(v) for v in D],
                    "width": int(msg.width),
                    "height": int(msg.height),
                }
                camera_infos.append(info)
                if intrinsics is None:
                    intrinsics = (
                        float(K[0]), float(K[4]),   # fx, fy
                        float(K[2]), float(K[5]),   # cx, cy
                        int(msg.width), int(msg.height),
                    )
                    _distortion_coeffs = [float(v) for v in D]
            elif topic == depth_info_topic:
                K = msg.k  # row-major 3×3
                info = {
                    "topic": topic,
                    "frame_id": _frame_id(msg),
                    "header_ns": _msg_ns(msg, ts, topic),
                    "bag_ns": int(ts),
                    "K": [float(v) for v in K],
                    "width": int(msg.width),
                    "height": int(msg.height),
                }
                depth_camera_infos.append(info)
                if depth_intrinsics is None:
                    depth_intrinsics = (
                        float(K[0]), float(K[4]),   # fx, fy
                        float(K[2]), float(K[5]),   # cx, cy
                        int(msg.width), int(msg.height),
                    )
            elif topic == "/tf_static":
                for t in getattr(msg, "transforms", []):
                    tf_graph.add(
                        _msg_ns(msg, ts, topic) * 1e-9,
                        str(t.header.frame_id),
                        str(t.child_frame_id),
                        _transform_to_matrix(t.transform),
                        is_static=True,
                    )

        if intrinsics is None:
            raise RuntimeError(f"No '{camera_info_topic}' messages found in {path}")
        fx, fy, cx, cy, width, height = intrinsics
        camera_info_width, camera_info_height = int(width), int(height)
        if color_msgs:
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(io.BytesIO(color_msgs[0].payload)) as _img:
                    image_width, image_height = _img.size
                if (image_width, image_height) != (camera_info_width, camera_info_height):
                    raise RuntimeError(
                        f"[streaming] Selected color intrinsics from '{camera_info_topic}' "
                        f"are {camera_info_width}x{camera_info_height}, but the first color frame is "
                        f"{image_width}x{image_height}. Use matching color CameraInfo for '{color_topic}'."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"[streaming] Could not verify color frame size against '{camera_info_topic}': {exc}"
                ) from exc

        # Undistortion must run in the native camera domain.  Applying the
        # resolution divisor before building the remap grids makes those grids
        # sample only a small corner of the native payload (and can turn valid
        # registered depth entirely into zeros).  Downscale both payloads and
        # intrinsics together after the native-resolution remap below.
        _R = max(1, int(streaming_resolution))
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._width = width
        self._height = height
        self._depth_scale = 1000.0
        color_intrinsics_obj = CameraIntrinsics(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            depth_scale=1000.0,
            camera_frame=color_frame_ids[0] if "color_frame_ids" in locals() and color_frame_ids else "camera_color_optical_frame",
        )

        color_msgs.sort(key=lambda m: m.header_ns)
        depth_msgs.sort(key=lambda m: m.header_ns)
        pose_msgs.sort(key=lambda m: m.header_ns)

        color_frame_ids = sorted({m.frame_id for m in color_msgs if m.frame_id})
        depth_frame_ids = sorted({m.frame_id for m in depth_msgs if m.frame_id})
        frame_ids_match = bool(color_frame_ids and depth_frame_ids and set(color_frame_ids) == set(depth_frame_ids))
        depth_topic_aligned = "aligned" in str(depth_topic).lower()
        depth_aligned = frame_ids_match or depth_topic_aligned
        depth_to_color = None
        raw_depth_reprojected = False
        if not depth_aligned:
            if not depth_intrinsics:
                raise RuntimeError(
                    "[streaming] OrbbecRosBagFrameSource requires depth already aligned to the color camera, "
                    "or /camera/depth/camera_info for raw-depth reprojection. "
                    f"Color frame_id(s)={color_frame_ids or ['<unset>']} depth frame_id(s)={depth_frame_ids or ['<unset>']} "
                    f"and depth topic '{depth_topic}' is not an aligned-depth topic."
                )
            if not color_frame_ids or not depth_frame_ids:
                raise RuntimeError(
                    "[streaming] OrbbecRosBagFrameSource cannot reproject raw depth because color/depth frame_id is unset. "
                    f"Color frame_id(s)={color_frame_ids or ['<unset>']} depth frame_id(s)={depth_frame_ids or ['<unset>']}."
                )
            depth_to_color = tf_graph.lookup(
                color_frame_ids[0],
                depth_frame_ids[0],
                timestamp=0.0,
                tolerance=None,
            )
            if depth_to_color is None:
                raise RuntimeError(
                    "[streaming] OrbbecRosBagFrameSource cannot reproject raw depth because /tf_static lacks "
                    f"a transform from {depth_frame_ids[0]!r} to {color_frame_ids[0]!r}."
                )

            print(
                "[streaming] OrbbecRosBag: reprojecting raw depth "
                f"{depth_frame_ids[0]} -> {color_frame_ids[0]} using /tf_static",
                flush=True,
            )
            color_intrinsics_obj = CameraIntrinsics(
                width=int(width),
                height=int(height),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                depth_scale=1000.0,
                camera_frame=color_frame_ids[0],
            )
            dfx, dfy, dcx, dcy, dwidth, dheight = depth_intrinsics
            depth_intrinsics_obj = CameraIntrinsics(
                width=int(dwidth),
                height=int(dheight),
                fx=float(dfx),
                fy=float(dfy),
                cx=float(dcx),
                cy=float(dcy),
                depth_scale=1000.0,
                camera_frame=depth_frame_ids[0],
            )
            from PIL import Image as _PILImage

            for msg in depth_msgs:
                with _PILImage.open(io.BytesIO(msg.payload)) as _img:
                    raw_depth = np.asarray(_img)
                aligned_depth = reproject_depth_to_color(
                    raw_depth,
                    depth_intrinsics_obj,
                    color_intrinsics_obj,
                    depth_to_color,
                    min_depth=float(open3d_odom_depth_min),
                    max_depth=float(open3d_odom_depth_max),
                )
                depth_u16 = np.clip(np.rint(aligned_depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
                buf = io.BytesIO()
                _PILImage.fromarray(depth_u16, mode="I;16").save(buf, format="PNG")
                msg.payload = buf.getvalue()
                msg.frame_id = color_frame_ids[0]
                msg.encoding = "16UC1"
                msg.format = "png"
            depth_aligned = True
            raw_depth_reprojected = True

        # --- Automatic undistortion ---
        # Read distortion coefficients captured from the first camera_info message.
        # When D is non-trivial (any |d_i| > 1e-6), compute undistort maps once and
        # remap every color and depth frame in-place.  For hardware-registered depth
        # (depth_aligned=True) both images share the same lens model, so the same maps
        # apply to both.  Intrinsics are updated to the rectified camera matrix.
        _dist = _distortion_coeffs
        _has_distortion = bool(_dist) and any(abs(float(d)) > 1e-6 for d in _dist)
        _undistort_map1 = _undistort_map2 = None
        _undistorted_K = None
        if _has_distortion:
            import cv2 as _cv2
            _K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            _D_arr = np.array([float(d) for d in _dist], dtype=np.float64)
            _sz = (int(width), int(height))
            _new_K, _ = _cv2.getOptimalNewCameraMatrix(_K_mat, _D_arr, _sz, alpha=0)
            _undistort_map1, _undistort_map2 = _cv2.initUndistortRectifyMap(
                _K_mat, _D_arr, None, _new_K, _sz, _cv2.CV_32FC1
            )
            _undistorted_K = _new_K
            # Update intrinsics to undistorted camera matrix
            fx, fy, cx, cy = float(_new_K[0, 0]), float(_new_K[1, 1]), float(_new_K[0, 2]), float(_new_K[1, 2])
            self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
            print(
                f"[streaming] OrbbecRosBag: distortion detected D={[round(d, 5) for d in _dist[:5]]} — "
                f"applying cv2.undistort to {len(color_msgs)} color + {len(depth_msgs)} depth frames; "
                f"new_K fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}",
                flush=True,
            )
            from PIL import Image as _PILImage
            from concurrent.futures import ThreadPoolExecutor as _TPE
            import os as _os

            def _remap_png(payload: bytes, interp: int) -> bytes:
                with _PILImage.open(io.BytesIO(payload)) as _img:
                    arr = np.asarray(_img)
                remapped = _cv2.remap(arr, _undistort_map1, _undistort_map2, interp)
                buf = io.BytesIO()
                _PILImage.fromarray(remapped).save(buf, format="PNG")
                return buf.getvalue()

            _n_workers = min(len(color_msgs), (_os.cpu_count() or 4) * 2, 32)
            with _TPE(max_workers=_n_workers) as _pool:
                _color_futures = [_pool.submit(_remap_png, m.payload, _cv2.INTER_LINEAR) for m in color_msgs]
                _depth_futures = (
                    [_pool.submit(_remap_png, m.payload, _cv2.INTER_NEAREST) for m in depth_msgs]
                    if depth_aligned else []
                )
                for msg, fut in zip(color_msgs, _color_futures):
                    msg.payload = fut.result()
                for msg, fut in zip(depth_msgs, _depth_futures):
                    msg.payload = fut.result()
        else:
            _has_distortion = False
            print(
                f"[streaming] OrbbecRosBag: distortion D={[round(d, 5) for d in (_dist or [])[:5]] or [0.0]} — "
                "no undistortion needed",
                flush=True,
            )

        if _R > 1:
            from PIL import Image as _PILImage

            reduced_width = max(1, int(width) // _R)
            reduced_height = max(1, int(height) // _R)

            for msg in color_msgs:
                msg.payload = _resize_png_payload(
                    msg.payload,
                    width=reduced_width,
                    height=reduced_height,
                    resample=_PILImage.Resampling.BILINEAR,
                )
            for msg in depth_msgs:
                msg.payload = _resize_png_payload(
                    msg.payload,
                    width=reduced_width,
                    height=reduced_height,
                    resample=_PILImage.Resampling.NEAREST,
                )

            source_width = int(width)
            source_height = int(height)
            fx, fy, cx, cy = _scale_intrinsics_for_resize(
                fx,
                fy,
                cx,
                cy,
                source_width=source_width,
                source_height=source_height,
                target_width=reduced_width,
                target_height=reduced_height,
            )
            scale_x = reduced_width / source_width
            scale_y = reduced_height / source_height
            width = reduced_width
            height = reduced_height
            self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
            self._width, self._height = width, height
            if _undistorted_K is not None:
                _undistorted_K = _undistorted_K.copy()
                _undistorted_K[0, :] *= scale_x
                _undistorted_K[1, :] *= scale_y

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"[streaming] Invalid selected color intrinsics from '{camera_info_topic}': width={width}, height={height}"
            )

        estimated_offset_ns = 0
        if sync_estimate_offset:
            estimated_offset_ns = estimate_stream_offset_ns(
                [m.header_ns for m in color_msgs],
                [m.header_ns for m in depth_msgs],
            )
        offset_config = str(sync_offset_ms)
        if offset_config.strip().lower() == "auto":
            offset_ns = estimated_offset_ns if sync_estimate_offset else 0
        else:
            try:
                offset_ns = int(round(float(offset_config) * 1e6))
            except ValueError as exc:
                raise ValueError(
                    "--orbbec_sync_offset_ms must be 'auto' or a numeric millisecond value "
                    f"(got {sync_offset_ms!r})"
                ) from exc

        pairs, pair_stats = sync_color_depth_unique(color_msgs, depth_msgs, sync_ns, offset_ns=offset_ns)
        synced_all, pose_stats = attach_pose(
            pairs,
            color_msgs,
            depth_msgs,
            pose_msgs if pose_source == "camera_pose" else [],
            max_pose_dt_ns=pose_sync_ns,
            pose_required=(pose_source == "camera_pose"),
        )
        sync_stats = merge_stats(
            pair_stats,
            pose_stats,
            color_msgs,
            depth_msgs,
            pose_msgs,
            estimated_offset_ns=estimated_offset_ns,
            offset_applied_ns=offset_ns,
        )
        sync_stats.p95_violation = not check_p95_within_threshold(sync_stats, sync_ns)

        report_path = sync_report_json or (
            os.path.join(model_path, "sync_report.json") if model_path else os.path.join(path, "sync_report.json")
        )
        selected_camera_info = camera_infos[0] if camera_infos else None
        write_sync_report(report_path, sync_stats, extras={
            "source_path": path,
            "topics": {
                "color": color_topic,
                "depth": depth_topic,
                "pose": pose_topic if pose_source == "camera_pose" else "",
                "camera_info": camera_info_topic,
            },
            "thresholds": {
                "rgb_depth_ms": float(sync_threshold_ms),
                "rgb_pose_ms": float(pose_sync_threshold_ms),
            },
            "offset": {
                "estimate_enabled": bool(sync_estimate_offset),
                "config_ms": sync_offset_ms,
                "estimated_offset_ns": int(estimated_offset_ns),
                "applied_offset_ns": int(offset_ns),
            },
            "frame_counts": {
                "color": len(color_msgs),
                "depth": len(depth_msgs),
                "pose": len(pose_msgs),
                "accepted_rgbd": len(synced_all),
            },
            "selected_camera_info": selected_camera_info,
            "selected_frame_size": {"width": int(width), "height": int(height)},
            "camera_infos": camera_infos,
            "depth_camera_infos": depth_camera_infos,
            "alignment": {
                "depth_aligned": bool(depth_aligned),
                "reason": (
                    "frame_id_match"
                    if frame_ids_match
                    else ("depth_topic_contains_aligned" if depth_topic_aligned else "reprojected_with_tf_static")
                ),
                "color_frame_ids": color_frame_ids,
                "depth_frame_ids": depth_frame_ids,
                "raw_depth_reprojected": bool(raw_depth_reprojected),
            },
            "distortion": {
                "D": [float(d) for d in (_dist or [])],
                "has_distortion": bool(_has_distortion),
                "undistortion_applied": bool(_has_distortion),
                "undistorted_K": _undistorted_K.tolist() if _undistorted_K is not None else None,
            },
            "strict": bool(sync_strict),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            "pose_source": pose_source,
        })
        print(f"[streaming] OrbbecRosBag sync: {format_stats_oneline(sync_stats)} report={report_path}", flush=True)

        if not synced_all:
            raise RuntimeError(
                "[streaming] OrbbecRosBagFrameSource accepted no RGB-D frames after strict timestamp sync. "
                f"threshold={sync_threshold_ms}ms color={len(color_msgs)} depth={len(depth_msgs)} pose_source={pose_source}"
            )
        if sync_strict and sync_stats.p95_violation:
            p95_ms = float(sync_stats.rgb_depth_dt_ns.get("abs_p95_ns", 0.0)) / 1e6
            raise RuntimeError(
                "[streaming] OrbbecRosBagFrameSource strict sync rejected this bag: "
                f"RGB-depth abs p95={p95_ms:.3f}ms exceeds threshold={float(sync_threshold_ms):.3f}ms. "
                "Pass --no-orbbec_sync_strict or increase --orbbec_sync_threshold_ms only if this drift is acceptable."
            )

        stride = max(1, int(frame_stride))
        synced_records = synced_all[::stride]
        if max_frames > 0:
            synced_records = synced_records[:int(max_frames)]
        synced: list = [
            (
                rec.color.header_ns,
                rec.color.payload,
                rec.depth.payload,
                rec.pose.payload if rec.pose is not None else None,
            )
            for rec in synced_records
        ]

        if pose_source == "open3d_odometry":
            cache_key = self._open3d_odom_cache_key(
                path=path,
                color_topic=color_topic,
                depth_topic=depth_topic,
                camera_info_topic=camera_info_topic,
                sync_threshold_ms=sync_threshold_ms,
                frame_stride=frame_stride,
                max_frames=max_frames,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                synced=synced,
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
                mode="precompute",
            )
            cache_path = self._open3d_odom_cache_path(
                bag_path=path,
                cache_dir=open3d_odom_cache_dir,
                cache_key=cache_key,
            )
            odom_poses = None
            odom_stats = None
            if open3d_odom_cache:
                odom_poses, odom_stats = self._load_open3d_odom_cache(
                    cache_path,
                    cache_key,
                    expected_frames=len(synced),
                )
            if odom_poses is None:
                odom_poses, odom_stats = self._estimate_open3d_odometry(
                    synced,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    depth_scale=1000.0,
                    max_failure_ratio=open3d_odom_max_failure_ratio,
                    odom_stride=open3d_odom_stride,
                    odom_downscale=open3d_odom_downscale,
                    odom_depth_min=self._odom_depth_min,
                    odom_depth_max=self._odom_depth_max,
                    odom_depth_diff_max=self._odom_depth_diff_max,
                    odom_method=open3d_odom_method,
                    icp_max_distance=open3d_icp_max_distance,
                    icp_robust_kernel=open3d_icp_robust_kernel,
                    icp_sigma=open3d_icp_sigma,
                )
                if open3d_odom_cache:
                    self._save_open3d_odom_cache(cache_path, cache_key, odom_poses, odom_stats)
            synced = [
                (color_ts, rgb_bytes, depth_bytes, odom_poses[idx])
                for idx, (color_ts, rgb_bytes, depth_bytes, _) in enumerate(synced)
            ]
            path_len = odom_stats["path_length"]
            bbox_min = odom_stats["bbox_min"]
            bbox_max = odom_stats["bbox_max"]
            print(
                "[streaming] OrbbecRosBag: pose_source=open3d_odometry "
                f"success={odom_stats['successes']} failures={odom_stats['failures']} "
                f"path_length={path_len:.3f}m camera_bbox={bbox_min}->{bbox_max} "
                f"stride={open3d_odom_stride} downscale={open3d_odom_downscale}"
            )

        # Invariant: the odometry detour above rebuilds `synced` but never
        # reorders or filters it, so synced[i] still corresponds to
        # synced_records[i]. Guard against a future filter desyncing them.
        assert len(synced) == len(synced_records), (
            f"streaming sync invariant violated: len(synced)={len(synced)} "
            f"!= len(synced_records)={len(synced_records)}"
        )
        self._frames: List[StreamingRGBDFrame] = []
        for sync_idx, (color_ts, rgb_bytes, depth_bytes, c2w) in enumerate(synced):
            rec = synced_records[sync_idx]
            self._frames.append(streaming_frame_from_synced_rgbd(
                rec,
                index=len(self._frames),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                c2w_override=c2w,
                depth_scale=1000.0,
            ))

        self._pose_source = pose_source
        self._live_odom_enabled = pose_source == "open3d_odometry_live"
        self._live_odom_stride = open3d_odom_stride
        self._live_odom_downscale = open3d_odom_downscale
        self._live_odom_max_failure_ratio = float(open3d_odom_max_failure_ratio)
        self._live_odom_motion_prior = bool(open3d_odom_motion_prior)
        self._live_odom_motion_gate = bool(open3d_odom_motion_gate)
        self._live_odom_max_trans = float(open3d_odom_max_trans_per_edge) if open3d_odom_max_trans_per_edge > 0 else float("inf")
        self._live_odom_max_rot = float(open3d_odom_max_rot_deg_per_edge) if open3d_odom_max_rot_deg_per_edge > 0 else float("inf")
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
        # Optional constant-velocity prior state. Reset to None on failure so a
        # bad estimate does not compound when the prior is enabled.
        self._live_odom_last_trans: np.ndarray | None = None
        self._live_odom_last_n_edges: int = 1
        self._live_odom_cache_path = None
        self._live_odom_cache_key = None
        self._live_odom_cache_enabled = bool(open3d_odom_cache)
        self._live_odom_cached_full = False
        self._live_odom_async_enabled = bool(open3d_odom_async) and self._live_odom_enabled
        self._live_odom_async_done = False
        self._live_odom_async_error = None
        self._live_odom_async_thread = None
        self._live_odom_cond = None
        # Phase 5: producer-consumer backpressure
        self._live_odom_consumer_idx = 0
        self._live_odom_queue_size = max(1, int(open3d_odom_async_queue_size))
        self._live_odom_async_stop = False
        if self._live_odom_enabled:
            live_cache_key = self._open3d_odom_cache_key(
                path=path,
                color_topic=color_topic,
                depth_topic=depth_topic,
                camera_info_topic=camera_info_topic,
                sync_threshold_ms=sync_threshold_ms,
                frame_stride=frame_stride,
                max_frames=max_frames,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                synced=synced,
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
            )
            live_cache_path = self._open3d_odom_cache_path(
                bag_path=path,
                cache_dir=open3d_odom_cache_dir,
                cache_key=live_cache_key,
            )
            self._live_odom_cache_key = live_cache_key
            self._live_odom_cache_path = live_cache_path
            if open3d_odom_cache:
                cached_poses, cached_stats = self._load_open3d_odom_cache(
                    live_cache_path,
                    live_cache_key,
                    expected_frames=len(self._frames),
                )
                if cached_poses is not None:
                    for idx, pose in enumerate(cached_poses):
                        self._frames[idx].c2w = pose
                    # Restore per-frame registration diagnostics so the input-quality
                    # tracking metric works on cache hits (matches a fresh-odometry run).
                    per_frame = cached_stats.get("per_frame")
                    if isinstance(per_frame, list) and len(per_frame) == len(self._frames):
                        for idx, st in enumerate(per_frame):
                            if isinstance(st, dict):
                                self._frames[idx]._odom_stats = st
                                self._frames[idx]._odom_valid = bool(st.get("valid", True))
                    self._live_odom_cached_full = True
                    self._live_odom_computed_idx = len(self._frames) - 1
                    self._live_odom_successes = int(cached_stats.get("successes", 0))
                    self._live_odom_failures = int(cached_stats.get("failures", 0))
                    self._live_odom_pairs = self._live_odom_successes + self._live_odom_failures
            if not self._live_odom_cached_full and self._frames:
                self._frames[0].c2w = np.eye(4, dtype=np.float32)
                if self._live_odom_async_enabled:
                    self._start_live_odom_worker()

        print(
            f"[streaming] OrbbecRosBag: {len(self._frames)} synced frames from {path} "
            f"(color={len(color_msgs)}, depth={len(depth_msgs)}, pose={len(pose_msgs)}, "
            f"pose_source={pose_source}"
            + (
                f", motion_prior={self._live_odom_motion_prior}, "
                f"motion_gate={self._live_odom_motion_gate}"
                if self._live_odom_enabled else ""
            )
            + ")"
        )

    @staticmethod
    def _open3d_odom_cache_key(
        *,
        path: str,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        sync_threshold_ms: float,
        frame_stride: int,
        max_frames: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        synced,
        odom_stride: int,
        odom_downscale: int,
        odom_method: str = "hybrid",
        icp_max_distance: float = 0.07,
        icp_robust_kernel: str = "huber",
        icp_sigma: float = 0.05,
        motion_prior: bool = False,
        motion_gate: bool = False,
        vbg_voxel_size: float = 0.0058,
        vbg_block_count: int = 40000,
        vbg_depth_max: float = 3.0,
        mode: str = "precompute",
        depth_scale: float = 1000.0,
        depth_info_topic: str = "",
        sync_offset_ns: int = 0,
        profile: str = "",
        depth_alignment: dict[str, Any] | None = None,
    ) -> str:
        import hashlib
        import json

        metadata_path = os.path.join(path, "metadata.yaml")
        try:
            stat = os.stat(metadata_path)
            metadata_sig = [int(stat.st_mtime_ns), int(stat.st_size)]
        except OSError:
            metadata_sig = [0, 0]
        timestamps = [int(item[0]) for item in synced]
        timestamps_digest = hashlib.sha256(
            ",".join(str(ts) for ts in timestamps).encode("ascii")
        ).hexdigest()[:16]
        payload = {
            "version": 2,
            "mode": str(mode),
            "path": os.path.abspath(path),
            "metadata": metadata_sig,
            "topics": [color_topic, depth_topic, camera_info_topic, depth_info_topic],
            "sync_threshold_ms": float(sync_threshold_ms),
            "sync_offset_ns": int(sync_offset_ns),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            "profile": str(profile),
            "depth_scale": round(float(depth_scale), 6),
            "intrinsics": [
                round(float(fx), 6),
                round(float(fy), 6),
                round(float(cx), 6),
                round(float(cy), 6),
                int(width),
                int(height),
            ],
            "odom_stride": int(odom_stride),
            "odom_downscale": int(odom_downscale),
            "odom_method": str(odom_method),
            "icp_params": [
                round(float(icp_max_distance), 6),
                str(icp_robust_kernel),
                round(float(icp_sigma), 6),
            ],
            "motion_prior": bool(motion_prior),
            "motion_gate": bool(motion_gate),
            "vbg_voxel_size": round(float(vbg_voxel_size), 6) if odom_method == "vbg" else None,
            "vbg_block_count": int(vbg_block_count) if odom_method == "vbg" else None,
            "vbg_depth_max": round(float(vbg_depth_max), 6) if odom_method == "vbg" else None,
            "n_frames": len(timestamps),
            "first_ts": timestamps[0] if timestamps else None,
            "last_ts": timestamps[-1] if timestamps else None,
            "timestamps_digest": timestamps_digest,
        }
        if depth_alignment is not None:
            payload["depth_alignment"] = depth_alignment
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:24]

    @staticmethod
    def _open3d_odom_cache_path(bag_path: str, cache_dir: str, cache_key: str) -> Path:
        if cache_dir:
            root = Path(cache_dir).expanduser()
        else:
            # bag_path may be a file or a directory; store cache beside the bag
            p = Path(bag_path)
            root = (p if p.is_dir() else p.parent) / ".open3d_odometry_cache"
        return root / f"{cache_key}.npz"

    @staticmethod
    def _load_open3d_odom_cache(cache_path: Path, cache_key: str, expected_frames: int):
        if not cache_path.exists():
            return None, None
        try:
            import json
            import numpy as np

            data = np.load(cache_path, allow_pickle=False)
            if str(data["cache_key"]) != cache_key:
                return None, None
            # Format version: bump invalidates pre-existing caches so they are recomputed
            # (v2 added per-frame odometry stats for the input-quality tracking metric).
            if "cache_format" not in data.files or int(data["cache_format"]) != 2:
                return None, None
            poses = data["poses"].astype(np.float32)
            if poses.shape != (expected_frames, 4, 4):
                return None, None
            stats = json.loads(str(data["stats_json"]))
            stats["cache_hit"] = True
            print(f"[streaming] OrbbecRosBag: loaded Open3D odometry cache {cache_path}", flush=True)
            return [poses[i] for i in range(poses.shape[0])], stats
        except Exception as exc:
            print(f"[streaming] OrbbecRosBag: ignoring bad Open3D odometry cache {cache_path}: {exc}", flush=True)
            return None, None

    @staticmethod
    def _save_open3d_odom_cache(cache_path: Path, cache_key: str, poses, stats) -> None:
        try:
            import json
            import numpy as np

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp_path,
                cache_key=np.asarray(cache_key),
                cache_format=np.asarray(2),  # v2: stats_json carries per-frame odom diagnostics
                poses=np.stack(poses, axis=0).astype(np.float32),
                stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
            )
            os.replace(tmp_path, cache_path)
            print(f"[streaming] OrbbecRosBag: saved Open3D odometry cache {cache_path}", flush=True)
        except Exception as exc:
            print(f"[streaming] OrbbecRosBag: failed to save Open3D odometry cache: {exc}", flush=True)

    @staticmethod
    def _estimate_icp_odometry(
        prev_rgbd,
        curr_rgbd,
        intrinsic,
        max_distance: float,
        robust_kernel: str = "huber",
        sigma: float = 0.05,
        init_guess: np.ndarray | None = None,
    ):
        import open3d as o3d
        import numpy as np

        # Convert RGBDImage to PointCloud
        pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, intrinsic)
        pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, intrinsic)

        # Estimate normals for point-to-plane (on target/prev)
        pcd_prev.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Setup robust kernel
        loss = None
        if robust_kernel == "huber":
            loss = o3d.pipelines.registration.HuberLoss(k=sigma)
        elif robust_kernel == "tukey":
            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)

        trans_init = init_guess if init_guess is not None else np.eye(4)

        # Point-to-plane ICP
        reg = o3d.pipelines.registration.registration_icp(
            pcd_curr,
            pcd_prev,
            max_distance,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        )

        # Success check: fitness is ratio of overlapping points
        success = reg.fitness > 0.01
        info = {
            "fitness": float(reg.fitness),
            "inlier_rmse": float(reg.inlier_rmse),
        }
        return success, reg.transformation, info

    @staticmethod
    def _estimate_open3d_odometry(
        synced_frames,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        depth_scale: float,
        max_failure_ratio: float,
        odom_stride: int = 1,
        odom_downscale: int = 1,
        odom_depth_min: float = 0.1,
        odom_depth_max: float = 8.0,
        odom_depth_diff_max: float = 0.07,
        odom_method: str = "hybrid",
        icp_max_distance: float = 0.07,
        icp_robust_kernel: str = "huber",
        icp_sigma: float = 0.05,
    ):
        try:
            import sys as _sys
            if not hasattr(_sys.stdout, "isatty"):
                _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
            if not hasattr(_sys.stderr, "isatty"):
                _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "--orbbec_pose_source open3d_odometry requires Open3D in the project container."
            ) from exc
        import numpy as np

        if not synced_frames:
            raise RuntimeError("Cannot estimate Open3D odometry: no synchronized RGB-D frames.")

        odom_stride = max(1, int(odom_stride))
        odom_downscale = max(1, int(odom_downscale))
        odom_width = max(1, int(width) // odom_downscale)
        odom_height = max(1, int(height) // odom_downscale)
        odom_fx = float(fx) / odom_downscale
        odom_fy = float(fy) / odom_downscale
        odom_cx = float(cx) / odom_downscale
        odom_cy = float(cy) / odom_downscale

        key_indices = list(range(0, len(synced_frames), odom_stride))
        if key_indices[-1] != len(synced_frames) - 1:
            key_indices.append(len(synced_frames) - 1)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(odom_width),
            int(odom_height),
            odom_fx,
            odom_fy,
            odom_cx,
            odom_cy,
        )
        option = o3d.pipelines.odometry.OdometryOption(
            depth_min=odom_depth_min,
            depth_max=odom_depth_max,
            depth_diff_max=odom_depth_diff_max,
        )
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

        # Prefer Tensor API (multi-threaded CPU, 2-4× faster than legacy single-threaded API)
        _use_tensor = hasattr(o3d, "t") and hasattr(o3d.t.pipelines, "odometry") and \
            hasattr(o3d.t.pipelines.odometry, "rgbd_odometry_multi_scale") and odom_method != "icp"
        _t_intrinsic = None
        _t_criteria = None
        if _use_tensor:
            _t_intrinsic = o3d.core.Tensor(np.array([
                [odom_fx, 0, odom_cx], [0, odom_fy, odom_cy], [0, 0, 1],
            ], dtype=np.float64))
            _t_criteria = [
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20, 1e-6, 1e-6),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10, 1e-5, 1e-5),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5, 1e-4, 1e-4),
            ]

        def _decode_raw_np(rgb_bytes, depth_bytes):
            from PIL import Image
            color_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            depth_img = Image.open(io.BytesIO(depth_bytes))
            if color_img.size != (odom_width, odom_height):
                color_img = color_img.resize((odom_width, odom_height), Image.Resampling.BILINEAR)
            if depth_img.size != (odom_width, odom_height):
                depth_img = depth_img.resize((odom_width, odom_height), Image.Resampling.NEAREST)
            return np.ascontiguousarray(np.asarray(color_img)), np.ascontiguousarray(np.asarray(depth_img))

        def to_rgbd(rgb_bytes, depth_bytes):
            color_np, depth_np = _decode_raw_np(rgb_bytes, depth_bytes)
            if _use_tensor:
                c = o3d.t.geometry.Image(color_np)
                d = o3d.t.geometry.Image(depth_np.astype(np.uint16))
                return o3d.t.geometry.RGBDImage(c, d)
            color = o3d.geometry.Image(color_np)
            depth = o3d.geometry.Image(depth_np)
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=float(depth_scale),
                depth_trunc=odom_depth_max, convert_rgb_to_intensity=True,
            )

        def _run_hybrid_odom(prev_rgbd, curr_rgbd, init_guess):
            if _use_tensor:
                result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    prev_rgbd, curr_rgbd, _t_intrinsic,
                    o3d.core.Tensor(init_guess.astype(np.float64)),
                    depth_scale=float(depth_scale), depth_max=odom_depth_max,
                    criteria_list=_t_criteria,
                    method=o3d.t.pipelines.odometry.Method.Hybrid,
                )
                return (bool(result.fitness > 0),
                        result.transformation.numpy().astype(np.float64),
                        {"fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse)})
            return o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, curr_rgbd, intrinsic, init_guess, jacobian, option,
            )

        def rot_to_quat(R):
            return _rot_to_quat(R)

        def quat_to_rot(q):
            return _quat_to_rot(q)

        def slerp(q0, q1, t):
            return _slerp(q0, q1, t)

        key_poses = [np.eye(4, dtype=np.float32)]
        prev_rgbd = to_rgbd(synced_frames[key_indices[0]][1], synced_frames[key_indices[0]][2])
        failures = 0
        successes = 0
        total_pairs = max(1, len(key_indices) - 1)
        for pair_idx, frame_idx in enumerate(key_indices[1:], start=1):
            _, rgb_bytes, depth_bytes, _ = synced_frames[frame_idx]
            curr_rgbd = to_rgbd(rgb_bytes, depth_bytes)

            if odom_method == "icp":
                success, trans_prev_to_curr, odom_info = OrbbecRosBagFrameSource._estimate_icp_odometry(
                    prev_rgbd,
                    curr_rgbd,
                    intrinsic,
                    max_distance=icp_max_distance,
                    robust_kernel=icp_robust_kernel,
                    sigma=icp_sigma,
                )
            else:
                success, trans_prev_to_curr, odom_info = _run_hybrid_odom(
                    prev_rgbd, curr_rgbd, np.eye(4, dtype=np.float64),
                )
            delta = np.linalg.inv(trans_prev_to_curr)
            edge_t = float(np.linalg.norm(delta[:3, 3]))
            cos_r = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
            edge_r = float(np.degrees(np.arccos(cos_r)))

            def _odom_info_summary(info) -> str:
                if isinstance(info, dict):
                    parts = []
                    if info.get("fitness") is not None:
                        parts.append(f"fitness={float(info['fitness']):.4f}")
                    if info.get("inlier_rmse") is not None:
                        parts.append(f"rmse={float(info['inlier_rmse']):.4f}")
                    return " ".join(parts)
                try:
                    arr = np.asarray(info)
                    if arr.size:
                        return f"info_trace={float(np.trace(arr)):.3e}"
                except Exception:
                    pass
                return ""

            if success:
                key_poses.append((key_poses[-1] @ np.linalg.inv(trans_prev_to_curr)).astype(np.float32))
                prev_rgbd = curr_rgbd
                successes += 1
            else:
                key_poses.append(key_poses[-1].copy())
                failures += 1
            if pair_idx == 1 or pair_idx == total_pairs or pair_idx % 50 == 0:
                print(
                    "[streaming] Open3D odometry "
                    f"{pair_idx}/{total_pairs} keyframe pairs "
                    f"(source_frame={frame_idx}, successes={successes}, failures={failures}, "
                    f"method={odom_method}, stride={odom_stride}, dt={edge_t:.4f}m, "
                    f"dr={edge_r:.2f}deg"
                    + (f", {_odom_info_summary(odom_info)}" if _odom_info_summary(odom_info) else "")
                    + ")",
                    flush=True,
                )

        failure_ratio = failures / total_pairs
        if failure_ratio > float(max_failure_ratio):
            raise RuntimeError(
                "Open3D odometry failed too often: "
                f"{failures}/{total_pairs} keyframe pairs "
                f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio={max_failure_ratio}"
            )

        poses = [None] * len(synced_frames)
        for seg_idx in range(len(key_indices) - 1):
            a = key_indices[seg_idx]
            b = key_indices[seg_idx + 1]
            pose_a = key_poses[seg_idx].astype(np.float64)
            pose_b = key_poses[seg_idx + 1].astype(np.float64)
            qa = rot_to_quat(pose_a[:3, :3])
            qb = rot_to_quat(pose_b[:3, :3])
            for frame_idx in range(a, b):
                alpha = 0.0 if b == a else (frame_idx - a) / float(b - a)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = quat_to_rot(slerp(qa, qb, alpha))
                pose[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
                poses[frame_idx] = pose.astype(np.float32)
        poses[key_indices[-1]] = key_poses[-1].astype(np.float32)
        for i in range(len(poses)):
            if poses[i] is None:
                poses[i] = poses[i - 1].copy() if i > 0 else np.eye(4, dtype=np.float32)

        centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
        deltas = centers[1:] - centers[:-1]
        path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(centers) > 1 else 0.0
        stats = {
            "successes": successes,
            "failures": failures,
            "path_length": path_length,
            "bbox_min": centers.min(axis=0).round(4).tolist(),
            "bbox_max": centers.max(axis=0).round(4).tolist(),
            "odom_stride": odom_stride,
            "odom_downscale": odom_downscale,
            "keyframes": len(key_indices),
            "cache_hit": False,
        }
        return poses, stats

    def _live_odom_to_rgbd(self, frame: StreamingRGBDFrame, tensor_api: bool = False):
        try:
            import sys as _sys
            if not hasattr(_sys.stdout, "isatty"):
                _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
            if not hasattr(_sys.stderr, "isatty"):
                _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "--orbbec_pose_source open3d_odometry_live requires Open3D in the project container."
            ) from exc
        import numpy as np
        from PIL import Image

        odom_width = max(1, int(self._width) // self._live_odom_downscale)
        odom_height = max(1, int(self._height) // self._live_odom_downscale)
        if frame._rgb_np is not None:
            color_img = Image.fromarray(frame._rgb_np)
        else:
            color_img = Image.open(io.BytesIO(frame._rgb_bytes)).convert("RGB")
        if frame._depth_np is not None:
            depth_img = Image.fromarray(frame._depth_np)
        else:
            depth_img = Image.open(io.BytesIO(frame._depth_bytes))
        if color_img.size != (odom_width, odom_height):
            color_img = color_img.resize((odom_width, odom_height), Image.Resampling.BILINEAR)
        if depth_img.size != (odom_width, odom_height):
            depth_img = depth_img.resize((odom_width, odom_height), Image.Resampling.NEAREST)
        color_np = np.ascontiguousarray(np.asarray(color_img))
        depth_np = np.ascontiguousarray(np.asarray(depth_img))
        if tensor_api:
            c = o3d.t.geometry.Image(color_np)
            d = o3d.t.geometry.Image(depth_np.astype(np.uint16))
            return o3d.t.geometry.RGBDImage(c, d)
        color = o3d.geometry.Image(color_np)
        depth = o3d.geometry.Image(depth_np)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=float(self._depth_scale),
            depth_trunc=self._odom_depth_max, convert_rgb_to_intensity=True,
        )

    def _save_live_odom_cache(self) -> None:
        if not self._live_odom_cache_enabled or self._live_odom_cache_path is None:
            return
        if self._live_odom_computed_idx < len(self._frames) - 1:
            return
        import numpy as np

        poses = [f.c2w if f.c2w is not None else np.eye(4, dtype=np.float32) for f in self._frames]
        centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
        deltas = centers[1:] - centers[:-1]
        stats = {
            "successes": self._live_odom_successes,
            "failures": self._live_odom_failures,
            "path_length": float(np.linalg.norm(deltas, axis=1).sum()) if len(centers) > 1 else 0.0,
            "bbox_min": centers.min(axis=0).round(4).tolist(),
            "bbox_max": centers.max(axis=0).round(4).tolist(),
            "odom_stride": self._live_odom_stride,
            "odom_downscale": self._live_odom_downscale,
            "keyframes": self._live_odom_pairs + 1,
            "cache_hit": False,
            # Persist per-frame registration diagnostics (fitness/inlier_rmse/valid) so the
            # input-quality tracking metric survives cache hits (otherwise a cached run loses
            # _odom_stats and reports tracking_available=False on an estimated-pose run).
            "per_frame": [getattr(f, "_odom_stats", None) for f in self._frames],
        }
        self._save_open3d_odom_cache(
            self._live_odom_cache_path,
            self._live_odom_cache_key,
            poses,
            stats,
        )

    def _notify_live_odom_progress(self) -> None:
        cond = getattr(self, "_live_odom_cond", None)
        if cond is not None:
            with cond:
                cond.notify_all()

    def _live_odom_initial_guess(self, idx: int):
        import numpy as np

        if not self._live_odom_motion_prior or self._live_odom_last_trans is None:
            return np.eye(4, dtype=np.float64)
        n_prev = max(1, self._live_odom_last_n_edges)
        n_curr = max(1, idx - self._live_odom_prev_key_idx)
        if n_curr == n_prev:
            return self._live_odom_last_trans.copy()
        trans = self._live_odom_last_trans.copy()
        trans[:3, 3] *= float(n_curr) / float(n_prev)
        return trans

    def _live_odom_motion_rejected(self, edge_t: float, edge_r: float, n_edges: int) -> bool:
        if not self._live_odom_motion_gate:
            return False
        n_edges = max(1, int(n_edges))
        return (
            edge_t > self._live_odom_max_trans * n_edges
            or edge_r > self._live_odom_max_rot * n_edges
        )

    def _start_live_odom_worker(self) -> None:
        """Phase 5: queue-bounded producer for live Open3D odometry.

        Producer advances `_live_odom_computed_idx` one frame at a time via
        `_ensure_live_odom_until(idx)`.  When the producer would get more than
        `_live_odom_queue_size` frames ahead of `_live_odom_consumer_idx`, it
        blocks on the condition variable instead of racing through the whole
        dataset.  Covers both RGBD-odom and ICP backends because the method
        dispatch is inside `_ensure_live_odom_until`.
        """
        if self._live_odom_async_thread is not None:
            return
        import threading

        self._live_odom_cond = threading.Condition()
        n_total = len(self._frames)

        def _run():
            try:
                for idx in range(1, n_total):
                    with self._live_odom_cond:
                        while (
                            (idx - self._live_odom_consumer_idx) > self._live_odom_queue_size
                            and not self._live_odom_async_stop
                        ):
                            self._live_odom_cond.wait(timeout=0.5)
                        if self._live_odom_async_stop:
                            return
                    self._ensure_live_odom_until(idx)
            except BaseException as exc:
                self._live_odom_async_error = exc
            finally:
                self._live_odom_async_done = True
                self._notify_live_odom_progress()

        self._live_odom_async_thread = threading.Thread(
            target=_run,
            name="open3d-live-odom",
            daemon=True,
        )
        self._live_odom_async_thread.start()
        print(
            f"[streaming] live Open3D odometry async worker started "
            f"(queue_size={self._live_odom_queue_size}, queue-bounded, "
            f"motion_prior={self._live_odom_motion_prior}, "
            f"motion_gate={self._live_odom_motion_gate})",
            flush=True,
        )

    def _signal_consumer_progress(self, idx: int) -> None:
        """Consumer-side: announce that frame `idx` has been consumed.

        Wakes the producer so it can advance past the backpressure gate.
        """
        cond = getattr(self, "_live_odom_cond", None)
        if cond is None:
            return
        with cond:
            if idx > self._live_odom_consumer_idx:
                self._live_odom_consumer_idx = idx
                cond.notify_all()

    def drain_pending(self) -> None:
        """Release backpressure and block until the producer finishes.

        Used by reports/trajectory eval that need every frame's pose realized.
        No-op when live odom is disabled or the cache covered everything.
        """
        if not self._live_odom_enabled or self._live_odom_cached_full:
            return
        last = len(self._frames) - 1
        if last < 0:
            return
        if self._live_odom_async_enabled:
            self._signal_consumer_progress(last)
            self._wait_live_odom_until(last)
        else:
            self._ensure_live_odom_until(last)

    def stop_pending(self) -> None:
        """Stop the async live-odometry producer without advancing unseen frames."""
        if not self._live_odom_enabled or self._live_odom_cached_full:
            return
        thread = getattr(self, "_live_odom_async_thread", None)
        if thread is None:
            return
        cond = getattr(self, "_live_odom_cond", None)
        self._live_odom_async_stop = True
        if cond is not None:
            with cond:
                cond.notify_all()
        thread.join(timeout=10.0)
        if thread.is_alive():
            print(
                "[streaming] Warning: live Open3D odometry worker did not stop within 10s; "
                "leaving daemon thread for process shutdown.",
                flush=True,
            )

    def _wait_live_odom_until(self, index: int) -> None:
        if index <= self._live_odom_computed_idx:
            return
        cond = getattr(self, "_live_odom_cond", None)
        if cond is None:
            self._ensure_live_odom_until(index)
            return
        with cond:
            while index > self._live_odom_computed_idx and self._live_odom_async_error is None:
                cond.wait(timeout=0.25)
                if self._live_odom_async_done and index > self._live_odom_computed_idx:
                    break
        if self._live_odom_async_error is not None:
            raise RuntimeError("Open3D live odometry async worker failed") from self._live_odom_async_error
        if index > self._live_odom_computed_idx:
            self._ensure_live_odom_until(index)

    def _ensure_live_odom_until(self, index: int) -> None:
        if not self._live_odom_enabled or self._live_odom_cached_full:
            return
        if index <= self._live_odom_computed_idx:
            return
        import sys as _sys
        if not hasattr(_sys.stdout, "isatty"):
            _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
        if not hasattr(_sys.stderr, "isatty"):
            _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
        import numpy as np
        import open3d as o3d

        if not self._frames:
            return

        _use_tensor = (
            hasattr(o3d, "t") and hasattr(o3d.t.pipelines, "odometry")
            and hasattr(o3d.t.pipelines.odometry, "rgbd_odometry_multi_scale")
            and self._live_odom_method not in ("icp", "vbg")
        ) or self._live_odom_method == "vbg"  # VBG requires tensor API

        if self._live_odom_computed_idx < 0:
            self._frames[0].c2w = np.eye(4, dtype=np.float32)
            self._live_odom_prev_key_idx = 0
            self._live_odom_prev_key_rgbd = self._live_odom_to_rgbd(self._frames[0], tensor_api=_use_tensor)
            self._live_odom_computed_idx = 0

        odom_width = max(1, int(self._width) // self._live_odom_downscale)
        odom_height = max(1, int(self._height) // self._live_odom_downscale)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(odom_width),
            int(odom_height),
            float(self._fx) / self._live_odom_downscale,
            float(self._fy) / self._live_odom_downscale,
            float(self._cx) / self._live_odom_downscale,
            float(self._cy) / self._live_odom_downscale,
        )
        option = o3d.pipelines.odometry.OdometryOption(
            depth_min=self._odom_depth_min,
            depth_max=self._odom_depth_max,
            depth_diff_max=self._odom_depth_diff_max,
        )
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        _t_intrinsic = None
        _t_criteria = None
        if _use_tensor:
            _t_intrinsic = o3d.core.Tensor(np.array([
                [float(self._fx) / self._live_odom_downscale, 0, float(self._cx) / self._live_odom_downscale],
                [0, float(self._fy) / self._live_odom_downscale, float(self._cy) / self._live_odom_downscale],
                [0, 0, 1],
            ], dtype=np.float64))
            _t_criteria = [
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20, 1e-6, 1e-6),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10, 1e-5, 1e-5),
                o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5, 1e-4, 1e-4),
            ]

        def _live_run_hybrid(prev_rgbd, curr_rgbd, init_guess):
            if _use_tensor:
                result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    prev_rgbd, curr_rgbd, _t_intrinsic,
                    o3d.core.Tensor(init_guess.astype(np.float64)),
                    depth_scale=float(self._depth_scale), depth_max=self._odom_depth_max,
                    criteria_list=_t_criteria,
                    method=o3d.t.pipelines.odometry.Method.Hybrid,
                )
                return (bool(result.fitness > 0),
                        result.transformation.numpy().astype(np.float64),
                        {"fitness": float(result.fitness), "inlier_rmse": float(result.inlier_rmse)})
            return o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, curr_rgbd, intrinsic, init_guess, jacobian, option,
            )

        def _live_run_vbg(curr_rgbd):
            """Track current frame via rgbd_odometry against VBG-synthesized model raycast.

            Returns (success, T_curr_from_prev, info).  The VBG Model is kept as an
            accumulated, drift-resistant map, while tracking delegates to the stable
            rgbd_odometry_multi_scale path (``track_frame_to_model`` is not used
            because of a property-lookup bug in Open3D 0.19.0).
            """
            # ---- lazy init on first call -------------------------------------------
            if self._vbg_model is None:
                _vbg_intrinsic = o3d.core.Tensor(np.array([
                    [float(self._fx) / self._live_odom_downscale, 0, float(self._cx) / self._live_odom_downscale],
                    [0, float(self._fy) / self._live_odom_downscale, float(self._cy) / self._live_odom_downscale],
                    [0, 0, 1],
                ], dtype=np.float64))
                for _device_name in ("CUDA:0", "CPU:0"):
                    try:
                        device = o3d.core.Device(_device_name)
                        _vbg_frame_obj = o3d.t.pipelines.slam.Frame(
                            odom_height, odom_width, _vbg_intrinsic, device,
                        )
                        _vbg_model_frame = o3d.t.pipelines.slam.Frame(
                            odom_height, odom_width, _vbg_intrinsic, device,
                        )
                        _vbg_model = o3d.t.pipelines.slam.Model(
                            self._live_odom_vbg_voxel_size,
                            16,
                            self._live_odom_vbg_block_count,
                            o3d.core.Tensor(np.eye(4, dtype=np.float64)),
                            device,
                        )
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError(
                        "VBG SLAM: could not initialise Model on any device (tried CUDA:0, CPU:0)."
                    )
                self._vbg_frame_obj = _vbg_frame_obj
                self._vbg_model = _vbg_model
                self._vbg_model_frame = _vbg_model_frame
                self._live_odom_vbg_frame_count = 0  # counter used as frame_id
                _vbg_init_pose_t = o3d.core.Tensor(np.eye(4, dtype=np.float64))
                self._vbg_model.update_frame_pose(0, _vbg_init_pose_t)
                self._live_odom_vbg_prev_T_model_from_world = _vbg_init_pose_t.cpu().numpy().astype(np.float64)

            # ---- set data on input frame -------------------------------------------
            try:
                self._vbg_frame_obj.set_data_from_image("color", curr_rgbd.color)
                self._vbg_frame_obj.set_data_from_image("depth", curr_rgbd.depth)
            except Exception:
                try:
                    self._vbg_frame_obj.set_data("color", curr_rgbd.color.as_tensor())
                    self._vbg_frame_obj.set_data("depth", curr_rgbd.depth.as_tensor())
                except Exception as exc:
                    return (False, np.eye(4, dtype=np.float64), {"fitness": 0.0, "error": str(exc)})

            _is_first = (self._live_odom_vbg_frame_count == 0)

            # ---- first frame -> integrate + synthesize, return identity ------------
            if _is_first:
                self._live_odom_vbg_frame_count += 1
                self._vbg_model.integrate(
                    self._vbg_frame_obj, float(self._depth_scale),
                    self._live_odom_vbg_depth_max, 8.0,
                )
                self._vbg_model.synthesize_model_frame(
                    self._vbg_model_frame, float(self._depth_scale),
                    0.1, self._live_odom_vbg_depth_max, 8.0, False,
                )
                return (True, np.eye(4, dtype=np.float64),
                        {"fitness": 1.0, "method": "vbg-init"})

            # ---- subsequent frames: track via rgbd_odometry against raycast --------
            try:
                # Build RGBDImage from the model raycast (synthesized previously)
                _raycast_color = self._vbg_model_frame.get_data_as_image("color")
                _raycast_depth = self._vbg_model_frame.get_data_as_image("depth")
                _raycast_rgbd = o3d.t.geometry.RGBDImage(_raycast_color, _raycast_depth)

                # Track current frame against raycast
                _t_criteria = [
                    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20, 1e-6, 1e-6),
                    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10, 1e-6, 1e-6),
                    o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5, 1e-6, 1e-6),
                ]
                result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    curr_rgbd, _raycast_rgbd, _t_intrinsic,
                    o3d.core.Tensor(np.eye(4, dtype=np.float64)),
                    depth_scale=float(self._depth_scale),
                    depth_max=self._live_odom_vbg_depth_max,
                    criteria_list=_t_criteria,
                    method=o3d.t.pipelines.odometry.Method.Hybrid,
                )

                if result.fitness <= 0:
                    return (False, np.eye(4, dtype=np.float64),
                            {"fitness": float(result.fitness)})

                # result.transformation = T_raycast_from_curr  (target_from_source)
                _T_raycast_from_curr = result.transformation.cpu().numpy().astype(np.float64)
                # The raycast viewpoint is known from the model's prior integration pose.
                _T_model_from_world = self._live_odom_vbg_prev_T_model_from_world
                # Compose: T_model_from_curr = T_raycast_from_curr  (both in model coords)
                #     p_model = T_raycast_from_curr @ p_curr
                # Then curr_c2w = inv(T_model_from_world) @ T_raycast_from_curr⁻¹
                #              = inv(T_raycast_from_curr) @ inv(T_model_from_world)
                # Actually: p_world = inv(T_model_from_world) @ p_model
                #          = inv(T_model_from_world) @ T_raycast_from_curr @ p_curr
                # so T_world_from_curr = inv(T_model_from_world) @ T_raycast_from_curr
                _T_world_from_curr = np.linalg.inv(_T_model_from_world) @ _T_raycast_from_curr
                curr_c2w = np.linalg.inv(_T_world_from_curr)

                if self._live_odom_prev_key_idx >= 0:
                    key_pose = self._frames[self._live_odom_prev_key_idx].c2w
                else:
                    key_pose = curr_c2w.copy()
                trans_prev_to_curr = np.linalg.inv(curr_c2w) @ key_pose

                # Motion gate
                if self._live_odom_motion_gate:
                    _dt = np.linalg.norm(trans_prev_to_curr[:3, 3])
                    _dr = np.arccos(max(-1, min(1, (np.trace(trans_prev_to_curr[:3, :3]) - 1) / 2)))
                    _dr_deg = np.rad2deg(_dr)
                    if _dt > self._live_odom_max_trans or _dr_deg > self._live_odom_max_rot:
                        return (False, np.eye(4, dtype=np.float64),
                                {"fitness": float(result.fitness), "rejected": True,
                                 "dt": _dt, "dr_deg": _dr_deg})

            except Exception as exc:
                return (False, np.eye(4, dtype=np.float64),
                        {"fitness": 0.0, "error": str(exc)})

            # ---- integrate current frame + synthesize model raycast for next frame --
            self._live_odom_vbg_frame_count += 1
            _T_model_from_curr = _T_raycast_from_curr  # same transform, in model coords
            try:
                self._vbg_model.update_frame_pose(
                    self._live_odom_vbg_frame_count,
                    o3d.core.Tensor(_T_model_from_curr.astype(np.float64)),
                )
                self._vbg_model.integrate(
                    self._vbg_frame_obj, float(self._depth_scale),
                    self._live_odom_vbg_depth_max, 8.0,
                )
                self._vbg_model.synthesize_model_frame(
                    self._vbg_model_frame, float(self._depth_scale),
                    0.1, self._live_odom_vbg_depth_max, 8.0, False,
                )
            except Exception as exc:
                print(f"[live-o3d] VBG integrate/synthesize failed: {exc}", flush=True)

            self._live_odom_vbg_prev_T_model_from_world = _T_model_from_curr
            return (True, trans_prev_to_curr.astype(np.float64),
                    {"fitness": float(result.fitness),
                     "inlier_rmse": float(getattr(result, "inlier_rmse", 0))})

        target = index
        if target > self._live_odom_stride and target % self._live_odom_stride != 0:
            target = ((target + self._live_odom_stride - 1) // self._live_odom_stride) * self._live_odom_stride
        target = min(target, len(self._frames) - 1)

        for idx in range(self._live_odom_computed_idx + 1, target + 1):
            should_estimate = (
                idx <= self._live_odom_stride
                or (idx % self._live_odom_stride) == 0
                or idx == len(self._frames) - 1
            )
            if not should_estimate:
                continue

            curr_rgbd = self._live_odom_to_rgbd(self._frames[idx], tensor_api=_use_tensor)
            init_guess = self._live_odom_initial_guess(idx)

            if self._live_odom_method == "icp":
                success, trans_prev_to_curr, odom_info = self._estimate_icp_odometry(
                    self._live_odom_prev_key_rgbd,
                    curr_rgbd,
                    intrinsic,
                    max_distance=self._live_odom_icp_max_distance,
                    robust_kernel=self._live_odom_icp_robust_kernel,
                    sigma=self._live_odom_icp_sigma,
                    init_guess=init_guess,
                )
            elif self._live_odom_method == "vbg":
                success, trans_prev_to_curr, odom_info = _live_run_vbg(curr_rgbd)
            else:
                success, trans_prev_to_curr, odom_info = _live_run_hybrid(
                    self._live_odom_prev_key_rgbd, curr_rgbd, init_guess,
                )
            self._live_odom_pairs += 1
            key_pose = self._frames[self._live_odom_prev_key_idx].c2w

            # Optional sanity gate on estimated per-edge motion.
            edge_t = None
            edge_r = None
            n_edges = max(1, idx - self._live_odom_prev_key_idx)
            if success:
                delta = np.linalg.inv(trans_prev_to_curr)
                edge_t = float(np.linalg.norm(delta[:3, 3]))
                cos_r = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
                edge_r = float(np.degrees(np.arccos(cos_r)))
                if self._live_odom_motion_rejected(edge_t, edge_r, n_edges):
                    print(
                        f"[live-o3d] idx={idx} sanity-gate reject: "
                        f"dt={edge_t:.3f}m dr={edge_r:.1f}° over {n_edges} edges "
                        f"(limits {self._live_odom_max_trans * n_edges:.3f}m "
                        f"/ {self._live_odom_max_rot * n_edges:.1f}°)",
                        flush=True,
                    )
                    success = False

            def _odom_info_summary(info) -> str:
                if isinstance(info, dict):
                    parts = []
                    if info.get("fitness") is not None:
                        parts.append(f"fitness={float(info['fitness']):.4f}")
                    if info.get("inlier_rmse") is not None:
                        parts.append(f"rmse={float(info['inlier_rmse']):.4f}")
                    return " ".join(parts)
                try:
                    arr = np.asarray(info)
                    if arr.size:
                        return f"info_trace={float(np.trace(arr)):.3e}"
                except Exception:
                    pass
                return ""

            if success:
                curr_pose = (key_pose @ np.linalg.inv(trans_prev_to_curr)).astype(np.float32)
                self._frames[idx]._odom_valid = True
                self._live_odom_successes += 1
                # Update constant-velocity prior with the successful transform
                self._live_odom_last_trans = trans_prev_to_curr.copy()
                self._live_odom_last_n_edges = max(1, idx - self._live_odom_prev_key_idx)
                # Attach diagnostics for TB logging in training loop
                _info = odom_info if isinstance(odom_info, dict) else {}
                self._frames[idx]._odom_stats = {
                    "valid": True,
                    "translation_m": edge_t or 0.0,
                    "rotation_deg": edge_r or 0.0,
                    "fitness": _info.get("fitness"),
                    "inlier_rmse": _info.get("inlier_rmse"),
                    "method": self._live_odom_method,
                }
            else:
                # Phase 2.2: on failure, keep the previous key pose but do NOT advance
                # the odometry reference — the next estimate will still compare against
                # the last good keyframe, preventing one bad edge from corrupting
                # the rest of the trajectory.
                curr_pose = key_pose.copy()
                self._frames[idx].c2w = curr_pose
                self._frames[idx]._odom_valid = False
                # Attach failure diagnostics for TB logging in training loop
                self._frames[idx]._odom_stats = {
                    "valid": False,
                    "translation_m": edge_t or 0.0,
                    "rotation_deg": edge_r or 0.0,
                    "fitness": None,
                    "inlier_rmse": None,
                    "method": self._live_odom_method,
                }
                for i in range(self._live_odom_prev_key_idx + 1, idx):
                    self._frames[i].c2w = curr_pose.copy()
                    self._frames[i]._odom_valid = False
                self._live_odom_failures += 1
                # Reset the velocity prior so a bad estimate doesn't compound
                self._live_odom_last_trans = None
                _motion = ""
                if edge_t is not None and edge_r is not None:
                    _motion = f" dt={edge_t:.4f}m dr={edge_r:.2f}deg"
                _quality = _odom_info_summary(odom_info)
                if _quality:
                    _quality = " " + _quality
                print(
                    "[streaming] live Open3D odometry failed "
                    f"source_frame={idx} successes={self._live_odom_successes} "
                    f"failures={self._live_odom_failures} method={self._live_odom_method} "
                    f"stride={self._live_odom_stride} "
                    f"motion_prior={self._live_odom_motion_prior} "
                    f"motion_gate={self._live_odom_motion_gate}{_motion}{_quality}",
                    flush=True,
                )
                failure_ratio = self._live_odom_failures / max(1, self._live_odom_pairs)
                if failure_ratio > self._live_odom_max_failure_ratio:
                    raise RuntimeError(
                        "Open3D live odometry failed too often: "
                        f"{self._live_odom_failures}/{self._live_odom_pairs} keyframes "
                        f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio="
                        f"{self._live_odom_max_failure_ratio}"
                    )
                self._live_odom_computed_idx = idx
                self._notify_live_odom_progress()
                continue  # skip reference advance

            self._frames[idx].c2w = curr_pose

            if idx > self._live_odom_prev_key_idx + 1:
                qa = _rot_to_quat(key_pose[:3, :3])
                qb = _rot_to_quat(curr_pose[:3, :3])
                for i in range(self._live_odom_prev_key_idx + 1, idx):
                    alpha = (i - self._live_odom_prev_key_idx) / float(idx - self._live_odom_prev_key_idx)
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, :3] = _quat_to_rot(_slerp(qa, qb, alpha))
                    pose[:3, 3] = (1.0 - alpha) * key_pose[:3, 3] + alpha * curr_pose[:3, 3]
                    self._frames[i].c2w = pose.astype(np.float32)
                    self._frames[i]._odom_valid = True

            self._live_odom_prev_key_idx = idx
            self._live_odom_prev_key_rgbd = curr_rgbd

            failure_ratio = self._live_odom_failures / max(1, self._live_odom_pairs)
            if failure_ratio > self._live_odom_max_failure_ratio:
                raise RuntimeError(
                    "Open3D live odometry failed too often: "
                    f"{self._live_odom_failures}/{self._live_odom_pairs} keyframes "
                    f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio="
                    f"{self._live_odom_max_failure_ratio}"
                )
            if self._live_odom_pairs == 1 or self._live_odom_pairs % 50 == 0:
                _motion = ""
                if edge_t is not None and edge_r is not None:
                    _motion = f" dt={edge_t:.4f}m dr={edge_r:.2f}deg"
                _quality = _odom_info_summary(odom_info)
                if _quality:
                    _quality = " " + _quality
                print(
                    "[streaming] live Open3D odometry "
                    f"source_frame={idx} successes={self._live_odom_successes} "
                    f"failures={self._live_odom_failures} method={self._live_odom_method} "
                    f"stride={self._live_odom_stride} "
                    f"motion_prior={self._live_odom_motion_prior} "
                    f"motion_gate={self._live_odom_motion_gate}{_motion}{_quality}",
                    flush=True,
                )
            self._live_odom_computed_idx = idx
            self._notify_live_odom_progress()

        self._save_live_odom_cache()

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        for idx in range(len(self._frames)):
            yield self.get_frame(idx)

    def get_all(self) -> List[StreamingRGBDFrame]:
        if self._live_odom_async_enabled and not self._live_odom_cached_full:
            self._signal_consumer_progress(len(self._frames) - 1)
            self._wait_live_odom_until(len(self._frames) - 1)
        elif self._live_odom_enabled:
            self._ensure_live_odom_until(len(self._frames) - 1)
        return self._frames

    @property
    def lazy_frames(self) -> bool:
        return self._live_odom_enabled and not self._live_odom_cached_full

    def get_frame(self, index: int) -> StreamingRGBDFrame:
        if index < 0 or index >= len(self._frames):
            raise IndexError(index)
        if self._live_odom_async_enabled and not self._live_odom_cached_full:
            self._wait_live_odom_until(index)
            # Phase 5: release backpressure so producer can advance past this idx
            self._signal_consumer_progress(index)
        elif self._live_odom_enabled:
            self._ensure_live_odom_until(index)
        return self._frames[index]


_ROS1_BAG_PROFILES = {
    "realsense": {
        "label": "RealsenseBag",
        "color_topic": "/device_0/sensor_1/Color_0/image/data",
        "depth_topic": "/device_0/sensor_0/Depth_0/image/data",
        "color_info_topic": "/device_0/sensor_1/Color_0/info/camera_info",
        "depth_info_topic": "/device_0/sensor_0/Depth_0/info/camera_info",
        "depth_scale": 1000.0,
    },
    "tum": {
        "label": "TUMRosbag",
        "color_topic": "/camera/rgb/image_color",
        "depth_topic": "/camera/depth/image",
        "color_info_topic": "/camera/rgb/camera_info",
        "depth_info_topic": "/camera/depth/camera_info",
        "depth_scale": 5000.0,
    },
}
