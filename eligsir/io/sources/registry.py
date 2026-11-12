"""Frame-source registry and lightweight source-kind detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict

from .base import FrameSource


@dataclass(frozen=True)
class SourceInfo:
    name: str
    status: str
    imports: str
    detector: Callable[[str], bool]


def _has(path: str, *parts: str) -> bool:
    return os.path.exists(os.path.join(path, *parts))


def _is_rgbd_sequence(path: str) -> bool:
    return _has(path, "frames.jsonl") and _has(path, "intrinsics.json")


def _is_tum_rgbd(path: str) -> bool:
    return _has(path, "rgb.txt") and _has(path, "depth.txt") and _has(path, "groundtruth.txt")


def _is_scannet_sens(path: str) -> bool:
    return os.path.isfile(path) and str(path).lower().endswith(".sens")


def _is_scannet_folder(path: str) -> bool:
    return _has(path, "color") and _has(path, "pose") and _has(path, "intrinsic")


def _is_hypersim(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "_detail"))


def _is_replica(path: str) -> bool:
    return _has(path, "mesh.ply")


def _is_orbbec_export(path: str) -> bool:
    return _has(path, "icl.yaml") and _has(path, "poses.gt.sim")


def _is_ros1_bag(path: str) -> bool:
    if os.path.isfile(path) and str(path).lower().endswith(".bag"):
        return True
    if not os.path.isdir(path):
        return False
    return len([f for f in os.listdir(path) if f.endswith(".bag")]) == 1


def _is_ros2_bag(path: str) -> bool:
    return _has(path, "metadata.yaml") or (os.path.isfile(path) and str(path).lower().endswith(".mcap"))


def _is_realityscan(path: str) -> bool:
    # Detection twin of scene/readers/realityscan.py:12-15 (is_realityscan_mobile);
    # duplicated here (and again in eligsir/io/sources/datasets/realityscan.py)
    # rather than imported -- eligsir/io/ must not import from scene/.
    has_sfm = _has(path, "sfm.txt")
    has_xmp = os.path.isdir(path) and any(f.endswith(".xmp") for f in os.listdir(path))
    return has_sfm and has_xmp


SOURCES: Dict[str, SourceInfo] = {
    "rgbd_sequence": SourceInfo("rgbd_sequence", "stable", "io-image", _is_rgbd_sequence),
    "tum_rgbd": SourceInfo("tum_rgbd", "stable", "io-image", _is_tum_rgbd),
    "scannet_sens": SourceInfo("scannet_sens", "offline-only", "io-image", _is_scannet_sens),
    "scannet_folder": SourceInfo("scannet_folder", "offline-only", "io-image", _is_scannet_folder),
    "hypersim": SourceInfo("hypersim", "offline-only", "hdf5/io-image", _is_hypersim),
    "replica": SourceInfo("replica", "offline-only", "preprocess/open3d", _is_replica),
    "orbbec_export": SourceInfo("orbbec_export", "legacy/offline", "io-image", _is_orbbec_export),
    "ros1_bag": SourceInfo("ros1_bag", "offline integration", "bag", _is_ros1_bag),
    "ros2_bag": SourceInfo("ros2_bag", "offline integration", "bag", _is_ros2_bag),
    "realityscan": SourceInfo("realityscan", "offline-only", "realityscan_importer", _is_realityscan),
}


def list_sources() -> list[str]:
    """Return registered source names."""
    return sorted(SOURCES)


def detect_source_kind(source_path: str) -> str | None:
    """Return the first registered source kind matching *source_path*."""
    for name, info in SOURCES.items():
        if info.detector(source_path):
            return name
    return None


def make_frame_source(source_path: str, args) -> FrameSource:
    """Auto-detect the RGB-D dataset layout and return an ordered frame source.

    ROS live transport is intentionally absent. Live ROS observations enter core
    only as canonical ``SensorPacket`` values through ``MappingApplication``.
    Offline ROS bag files remain ordinary deterministic dataset sources.
    """
    import eligsir.utils.streaming_frames as _sf
    from eligsir.io.sources.datasets.realsense_rosbag import RealsenseRosBagFrameSource

    path = source_path
    if (
        os.path.exists(os.path.join(path, "frames.jsonl"))
        and os.path.exists(os.path.join(path, "intrinsics.json"))
    ):
        _full_res_raw = str(getattr(args, "streaming_full_res_frame_indices", "") or "")
        _full_res_indices = {
            int(_s) for _s in _full_res_raw.split(",") if _s.strip().isdigit()
        }
        return _sf.RGBDSequenceFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
            streaming_resolution=getattr(args, "streaming_resolution", 1),
            full_res_indices=_full_res_indices,
        )
    if (
        os.path.exists(os.path.join(path, "rgb.txt"))
        and os.path.exists(os.path.join(path, "depth.txt"))
        and os.path.exists(os.path.join(path, "groundtruth.txt"))
    ):
        return _sf.TUMFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "tum_frame_stride", 1),
            association_max_dt=getattr(args, "tum_association_max_dt", 0.03),
            sequence=getattr(args, "tum_sequence", ""),
            capture_time_pose_interpolation=bool(
                getattr(args, "tum_capture_time_pose_interpolation", False)
            ),
            motion_depth_confidence_mode=str(
                getattr(args, "streaming_motion_depth_confidence_mode", "off")
            ),
        )
    from scene.readers.scannet import _find_scannet_sens
    if _find_scannet_sens(path) is not None:
        return _sf.ScanNetSensFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "scannet_frame_stride", 10),
            depth_scale=getattr(args, "scannet_depth_scale", 1000.0),
        )
    if (
        os.path.exists(os.path.join(path, "color"))
        and os.path.exists(os.path.join(path, "pose"))
        and os.path.exists(os.path.join(path, "intrinsic"))
    ):
        return _sf.ScanNetFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "scannet_frame_stride", 10),
            depth_scale=getattr(args, "scannet_depth_scale", 1000.0),
        )
    _detail_dir = os.path.join(path, "_detail")
    if os.path.isdir(_detail_dir):
        cam_id = getattr(args, "hypersim_cam_id", "cam_00")
        if os.path.isdir(os.path.join(_detail_dir, cam_id)):
            return _sf.HyperSimFrameSource(
                path,
                cam_id=cam_id,
                max_frames=getattr(args, "streaming_max_frames", 0),
                frame_stride=getattr(args, "hypersim_frame_stride", 1),
            )
    if os.path.exists(os.path.join(path, "mesh.ply")):
        return _sf.ReplicaFrameSource(
            path,
            num_views=getattr(args, "replica_num_views", 120),
            width=getattr(args, "replica_width", 640),
            height=getattr(args, "replica_height", 480),
            fov_degrees=getattr(args, "replica_fov", 70.0),
            render_points=getattr(args, "replica_render_points", 300000),
            splat_radius=getattr(args, "replica_splat_radius", 1),
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
        )
    if (
        os.path.exists(os.path.join(path, "icl.yaml"))
        and os.path.exists(os.path.join(path, "poses.gt.sim"))
    ):
        return _sf.OrbbecExportFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
        )

    _bag_file = None
    if os.path.isfile(path) and path.endswith(".bag"):
        _bag_file = path
    elif os.path.isdir(path):
        _bags = [f for f in os.listdir(path) if f.endswith(".bag")]
        if len(_bags) == 1:
            _bag_file = os.path.join(path, _bags[0])
    if _bag_file is not None:
        _profile = _sf._select_ros1_bag_profile(_bag_file, getattr(args, "rosbag_profile", "auto"))
        _defaults = _sf._ROS1_BAG_PROFILES[_profile]
        _color_topic = getattr(args, "rosbag_color_topic", "") or (
            getattr(args, "realsense_color_topic", "") if _profile == "realsense" else ""
        ) or _defaults["color_topic"]
        _depth_topic = getattr(args, "rosbag_depth_topic", "") or (
            getattr(args, "realsense_depth_topic", "") if _profile == "realsense" else ""
        ) or _defaults["depth_topic"]
        _color_info_topic = getattr(args, "rosbag_color_info_topic", "") or (
            getattr(args, "realsense_color_info_topic", "") if _profile == "realsense" else ""
        ) or _defaults["color_info_topic"]
        _depth_info_topic = getattr(args, "rosbag_depth_info_topic", "") or (
            getattr(args, "realsense_depth_info_topic", "") if _profile == "realsense" else ""
        ) or _defaults["depth_info_topic"]
        _sync_report_json = getattr(args, "rosbag_sync_report_json", "")
        if not _sync_report_json and getattr(args, "model_path", ""):
            _sync_report_json = os.path.join(args.model_path, "rosbag_sync_report.json")
        _depth_scale = float(getattr(args, "rosbag_depth_scale", 0.0) or _defaults["depth_scale"])
        _alignment_debug_frames = max(0, int(getattr(args, "rosbag_alignment_debug_frames", 3)))
        _alignment_debug_dir = getattr(args, "rosbag_alignment_debug_dir", "")
        if (
            not _alignment_debug_dir
            and _profile == "realsense"
            and _alignment_debug_frames > 0
            and getattr(args, "model_path", "")
        ):
            _alignment_debug_dir = os.path.join(args.model_path, "depth_alignment_debug")
        return RealsenseRosBagFrameSource(
            _bag_file,
            color_topic=_color_topic,
            depth_topic=_depth_topic,
            color_info_topic=_color_info_topic,
            depth_info_topic=_depth_info_topic,
            sync_threshold_ms=getattr(args, "rosbag_sync_threshold_ms", getattr(args, "realsense_sync_threshold_ms", 33.0)),
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
            open3d_odom_max_failure_ratio=getattr(args, "orbbec_open3d_odom_max_failure_ratio", 0.25),
            open3d_odom_cache=getattr(args, "orbbec_open3d_odom_cache", True),
            open3d_odom_cache_dir=getattr(args, "orbbec_open3d_odom_cache_dir", ""),
            open3d_odom_stride=getattr(args, "orbbec_open3d_odom_stride", 1),
            open3d_odom_downscale=getattr(args, "orbbec_open3d_odom_downscale", 1),
            open3d_odom_async=getattr(args, "orbbec_open3d_odom_async", True),
            open3d_odom_async_queue_size=getattr(args, "orbbec_open3d_odom_async_queue_size", 32),
            open3d_odom_motion_prior=getattr(args, "orbbec_open3d_odom_motion_prior", False),
            open3d_odom_motion_gate=getattr(args, "orbbec_open3d_odom_motion_gate", False),
            open3d_odom_max_trans_per_edge=getattr(args, "orbbec_open3d_odom_max_trans_per_edge", 0.15),
            open3d_odom_max_rot_deg_per_edge=getattr(args, "orbbec_open3d_odom_max_rot_deg_per_edge", 8.0),
            open3d_odom_depth_min=getattr(args, "streaming_min_depth", 0.1),
            open3d_odom_depth_max=getattr(args, "streaming_max_depth", 8.0),
            open3d_odom_depth_diff_max=getattr(args, "orbbec_open3d_odom_depth_diff_max", 0.07),
            open3d_odom_method=getattr(args, "orbbec_open3d_odom_method", "hybrid"),
            open3d_icp_max_distance=getattr(args, "orbbec_open3d_icp_max_distance", 0.07),
            open3d_icp_robust_kernel=getattr(args, "orbbec_open3d_icp_robust_kernel", "huber"),
            open3d_icp_sigma=getattr(args, "orbbec_open3d_icp_sigma", 0.05),
            vbg_voxel_size=getattr(args, "orbbec_vbg_voxel_size", 0.0058),
            vbg_block_count=getattr(args, "orbbec_vbg_block_count", 40000),
            vbg_depth_max=getattr(args, "orbbec_vbg_depth_max", 3.0),
            streaming_resolution=getattr(args, "streaming_resolution", 1),
            depth_scale=_depth_scale,
            sync_estimate_offset=getattr(args, "rosbag_sync_estimate_offset", False),
            sync_offset_ms=getattr(args, "rosbag_sync_offset_ms", "0"),
            sync_report_json=_sync_report_json,
            rgb_encoding_override=getattr(args, "rosbag_rgb_encoding_override", ""),
            alignment_debug_dir=_alignment_debug_dir,
            alignment_debug_frames=_alignment_debug_frames,
            source_label=str(_defaults["label"]),
            profile=_profile,
            realsense_depth_filters=bool(getattr(args, "rosbag_realsense_depth_filters", True)),
        )

    if os.path.exists(os.path.join(path, "metadata.yaml")) or (
        os.path.isfile(path) and str(path).lower().endswith(".mcap")
    ):
        return _sf.OrbbecRosBagFrameSource(
            path,
            color_topic=getattr(args, "orbbec_color_topic", "/camera/color/image_raw/compressed"),
            depth_topic=getattr(args, "orbbec_depth_topic", "/camera/depth/image_raw/compressed"),
            pose_topic=getattr(args, "orbbec_pose_topic", "/camera_pose"),
            pose_source=getattr(args, "orbbec_pose_source", "camera_pose"),
            camera_info_topic=getattr(args, "orbbec_camera_info_topic", "/camera/color/camera_info"),
            sync_threshold_ms=getattr(args, "orbbec_sync_threshold_ms", 5.0),
            pose_sync_threshold_ms=getattr(args, "orbbec_pose_sync_threshold_ms", 10.0),
            sync_estimate_offset=getattr(args, "orbbec_sync_estimate_offset", False),
            sync_offset_ms=getattr(args, "orbbec_sync_offset_ms", "auto"),
            sync_strict=getattr(args, "orbbec_sync_strict", True),
            sync_report_json=getattr(args, "orbbec_sync_report_json", ""),
            model_path=getattr(args, "model_path", ""),
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
            open3d_odom_max_failure_ratio=getattr(args, "orbbec_open3d_odom_max_failure_ratio", 0.25),
            open3d_odom_cache=getattr(args, "orbbec_open3d_odom_cache", True),
            open3d_odom_cache_dir=getattr(args, "orbbec_open3d_odom_cache_dir", ""),
            open3d_odom_stride=getattr(args, "orbbec_open3d_odom_stride", 1),
            open3d_odom_downscale=getattr(args, "orbbec_open3d_odom_downscale", 1),
            open3d_odom_async=getattr(args, "orbbec_open3d_odom_async", True),
            open3d_odom_async_queue_size=getattr(args, "orbbec_open3d_odom_async_queue_size", 32),
            open3d_odom_motion_prior=getattr(args, "orbbec_open3d_odom_motion_prior", False),
            open3d_odom_motion_gate=getattr(args, "orbbec_open3d_odom_motion_gate", False),
            open3d_odom_max_trans_per_edge=getattr(args, "orbbec_open3d_odom_max_trans_per_edge", 0.15),
            open3d_odom_max_rot_deg_per_edge=getattr(args, "orbbec_open3d_odom_max_rot_deg_per_edge", 8.0),
            open3d_odom_depth_min=getattr(args, "streaming_min_depth", 0.1),
            open3d_odom_depth_max=getattr(args, "streaming_max_depth", 8.0),
            open3d_odom_depth_diff_max=getattr(args, "orbbec_open3d_odom_depth_diff_max", 0.07),
            open3d_odom_method=getattr(args, "orbbec_open3d_odom_method", "hybrid"),
            open3d_icp_max_distance=getattr(args, "orbbec_open3d_icp_max_distance", 0.07),
            open3d_icp_robust_kernel=getattr(args, "orbbec_open3d_icp_robust_kernel", "huber"),
            open3d_icp_sigma=getattr(args, "orbbec_open3d_icp_sigma", 0.05),
            vbg_voxel_size=getattr(args, "orbbec_vbg_voxel_size", 0.0058),
            vbg_block_count=getattr(args, "orbbec_vbg_block_count", 40000),
            vbg_depth_max=getattr(args, "orbbec_vbg_depth_max", 3.0),
            streaming_resolution=getattr(args, "streaming_resolution", 1),
        )

    if _is_realityscan(path):
        from eligsir.io.sources.datasets.realityscan import RealityScanFrameSource
        return RealityScanFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
            streaming_resolution=getattr(args, "streaming_resolution", 1),
            auto_downscale_max_pixels=int(getattr(args, "streaming_auto_downscale_max_pixels", 0) or 0),
        )
    raise ValueError(
        f"[streaming] No supported RGB-D dataset layout found at: {path}\n"
        "Supported: generic RGBDSequence (frames.jsonl + intrinsics.json), "
        "TUM (rgb.txt + depth.txt + groundtruth.txt), "
        "ScanNet (color/ + pose/ + intrinsic/ or .sens file), "
        "HyperSim (_detail/cam_00/ with HDF5 frames), "
        "Replica (mesh.ply), "
        "OrbbecExport (icl.yaml + poses.gt.sim), "
        "OrbbecRosBag (metadata.yaml), "
        "RealsenseBag (.bag file or directory with one .bag file), "
        "RealityScan Mobile (sfm.txt + *.xmp)."
    )
