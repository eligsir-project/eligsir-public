"""TUMFrameSource -- TUM RGB-D benchmark frame source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


class TUMFrameSource:
    """
    Ordered frames from a TUM RGB-D dataset (rgb.txt + depth.txt + groundtruth.txt).
    Reuses private helpers from scene.readers.tum.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
        association_max_dt: float = 0.03,
        sequence: str = "",
        capture_time_pose_interpolation: bool = False,
        motion_depth_confidence_mode: str = "off",
    ):
        import numpy as np
        from scene.readers.tum import (
            _read_tum_list,
            _read_tum_poses,
            _associate_tum,
            _interpolate_tum_capture_pose,
            _resolve_tum_camera_config,
            _tum_pose_to_c2w,
        )

        rgb_entries = _read_tum_list(os.path.join(path, "rgb.txt"))
        depth_entries = _read_tum_list(os.path.join(path, "depth.txt"))
        pose_entries = _read_tum_poses(os.path.join(path, "groundtruth.txt"))
        assocs = _associate_tum(rgb_entries, depth_entries, pose_entries, max_dt=association_max_dt)
        if motion_depth_confidence_mode not in {"off", "identity", "motion"}:
            raise ValueError("motion_depth_confidence_mode must be off, identity, or motion")
        pose_times = (
            np.asarray([entry[0] for entry in pose_entries], dtype=np.float64)
            if capture_time_pose_interpolation or motion_depth_confidence_mode == "motion"
            else None
        )
        if frame_stride > 1:
            assocs = assocs[::frame_stride]
        if max_frames > 0:
            assocs = assocs[:max_frames]

        camera_config = _resolve_tum_camera_config(path, sequence=sequence)
        fx, fy, cx, cy = (
            camera_config.fx,
            camera_config.fy,
            camera_config.cx,
            camera_config.cy,
        )
        width = int(camera_config.width or 640)
        height = int(camera_config.height or 480)

        self._frames: List[StreamingRGBDFrame] = []
        for i, assoc in enumerate(assocs):
            rgb_path = os.path.join(path, assoc["rgb_rel"])
            if not os.path.exists(rgb_path):
                continue
            c2w = _tum_pose_to_c2w(assoc["tvec"], assoc["qxyzw"])
            if not np.isfinite(c2w).all():
                continue
            depth_path = os.path.join(path, assoc["depth_rel"])
            color_ts_ns = int(round(float(assoc["rgb_t"]) * 1_000_000_000))
            depth_ts_ns = int(round(float(assoc["depth_t"]) * 1_000_000_000))
            pose_ts_ns = int(round(float(assoc["pose_t"]) * 1_000_000_000))
            frame = StreamingRGBDFrame(
                index=i,
                timestamp=float(assoc["rgb_t"]),
                rgb_path=rgb_path,
                depth_path=depth_path if os.path.exists(depth_path) else None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width,
                height=height,
                depth_scale=float(camera_config.depth_scale),
                color_ts_ns=color_ts_ns,
                depth_ts_ns=depth_ts_ns,
                pose_ts_ns=pose_ts_ns,
                rgb_depth_dt_ns=color_ts_ns - depth_ts_ns,
                rgb_pose_dt_ns=color_ts_ns - pose_ts_ns,
            )
            if capture_time_pose_interpolation:
                mapping_c2w, reason = _interpolate_tum_capture_pose(
                    assoc["rgb_t"], pose_entries, pose_times=pose_times
                )
                frame._mapping_c2w = c2w if mapping_c2w is None else mapping_c2w
                frame._pose_time_diagnostics = {
                    "tum_pose_time_method": "nearest" if mapping_c2w is None else "slerp",
                    "tum_pose_time_reason": reason,
                }
            if motion_depth_confidence_mode == "motion":
                rgb_c2w, rgb_reason = _interpolate_tum_capture_pose(
                    assoc["rgb_t"], pose_entries, pose_times=pose_times
                )
                depth_c2w, depth_reason = _interpolate_tum_capture_pose(
                    assoc["depth_t"], pose_entries, pose_times=pose_times
                )
                reason = rgb_reason if rgb_c2w is None else depth_reason
                transform = None
                if rgb_c2w is not None and depth_c2w is not None:
                    transform = np.asarray(
                        np.linalg.inv(rgb_c2w) @ depth_c2w, dtype=np.float32
                    )
                    if not np.isfinite(transform).all():
                        transform = None
                        reason = "nonfinite_capture_transform"
                    else:
                        transform.setflags(write=False)
                frame._motion_depth_confidence_diagnostics = {
                    "motion_depth_confidence_source": "offline_tum",
                    "motion_depth_confidence_reason": reason,
                    "motion_depth_confidence_transform": transform,
                }
            self._frames.append(frame)
        print(f"[streaming] TUM: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
