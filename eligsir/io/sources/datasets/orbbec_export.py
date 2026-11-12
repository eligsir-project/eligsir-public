"""OrbbecExportFrameSource -- Orbbec Femto Bolt ICL-format export frame source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). Bundled with its
_parse_icl_poses/_icl_frame_index helpers (used only by this source).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


def _parse_icl_poses(poses_path: str):
    """Parse poses.gt.sim: blank-separated 3×4 c2w blocks → list of (4,4) float32 arrays."""
    import numpy as np
    with open(poses_path) as f:
        content = f.read()
    poses = []
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        rows = [list(map(float, r.split())) for r in block.split("\n") if r.strip()]
        if len(rows) == 3 and all(len(r) == 4 for r in rows):
            mat34 = np.array(rows, dtype=np.float32)
            mat44 = np.eye(4, dtype=np.float32)
            mat44[:3, :] = mat34
            poses.append(mat44)
    return poses


def _icl_frame_index(filename: str) -> int:
    """Extract sequential index N from ICL filenames like frame_TS_N.jpg or TS_N.png."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return int(stem.rsplit("_", 1)[-1])


class OrbbecExportFrameSource:
    """Frames from an Orbbec Femto Bolt ICL export (rgbd_export2 tool).

    Directory layout::
        icl.yaml           — camera intrinsics + depth scale
        poses.gt.sim       — blank-separated 3×4 c2w matrices (one per frame)
        rgb/               — frame_TS_N.jpg  (JPEG colour)
        depth/             — TS_N.png        (uint16 PNG, mm)
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import yaml

        icl_yaml = os.path.join(path, "icl.yaml")
        poses_file = os.path.join(path, "poses.gt.sim")
        rgb_dir = os.path.join(path, "rgb")
        depth_dir = os.path.join(path, "depth")

        with open(icl_yaml) as f:
            cfg = yaml.safe_load(f)
        cam = cfg["camera_params"]
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = float(cam["cx"])
        cy = float(cam["cy"])
        width = int(cam["image_width"])
        height = int(cam["image_height"])
        depth_scale = float(cam.get("png_depth_scale", 1000.0))

        poses = _parse_icl_poses(poses_file)

        rgb_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg"))],
            key=_icl_frame_index,
        )
        depth_files = sorted(
            [f for f in os.listdir(depth_dir) if f.lower().endswith(".png")],
            key=_icl_frame_index,
        )

        # Match by sequential index
        rgb_by_idx = {_icl_frame_index(f): f for f in rgb_files}
        depth_by_idx = {_icl_frame_index(f): f for f in depth_files}
        common = sorted(set(rgb_by_idx) & set(depth_by_idx) & set(range(len(poses))))

        self._frames: List[StreamingRGBDFrame] = []
        for pose_idx in common[::frame_stride]:
            self._frames.append(StreamingRGBDFrame(
                index=len(self._frames),
                timestamp=float(pose_idx),
                rgb_path=os.path.join(rgb_dir, rgb_by_idx[pose_idx]),
                depth_path=os.path.join(depth_dir, depth_by_idx[pose_idx]),
                c2w=poses[pose_idx],
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                depth_scale=depth_scale,
            ))
            if max_frames > 0 and len(self._frames) >= max_frames:
                break

        print(f"[streaming] OrbbecExport: {len(self._frames)} frames from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
