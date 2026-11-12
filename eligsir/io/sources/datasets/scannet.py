"""ScanNetFrameSource / ScanNetSensFrameSource -- ScanNet frame sources.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md). Moved together per
that plan (small, closely related).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


class ScanNetFrameSource:
    """
    Ordered frames from a ScanNet scene (color/ + pose/ + intrinsic/).
    Reuses private helpers from scene.readers.scannet.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
        depth_scale: float = 1000.0,
    ):
        from scene.readers.scannet import (
            _list_scannet_color_frames,
            _scannet_frame_id_from_color,
            _find_scannet_intrinsics,
            _read_scannet_pose,
        )
        from PIL import Image

        intrinsics = _find_scannet_intrinsics(path)
        fx, fy, cx, cy = intrinsics["color"]
        dfx, dfy, dcx, dcy = intrinsics["depth"]

        color_files = _list_scannet_color_frames(path)
        if frame_stride > 1:
            color_files = color_files[::frame_stride]
        if max_frames > 0:
            color_files = color_files[:max_frames]

        pose_dir = os.path.join(path, "pose")
        depth_dir = os.path.join(path, "depth")

        # Detect color image dimensions from first existing frame
        width, height = 1296, 968
        for cf in color_files:
            try:
                img = Image.open(cf)
                width, height = img.size
                break
            except Exception:
                pass

        # Detect depth image dimensions from first existing depth frame
        depth_width, depth_height = 640, 480
        for cf in color_files:
            fid = _scannet_frame_id_from_color(cf)
            df = os.path.join(depth_dir, f"{fid}.png")
            if os.path.exists(df):
                try:
                    img = Image.open(df)
                    depth_width, depth_height = img.size
                except Exception:
                    pass
                break

        # Warn if color and depth intrinsics differ (common in ScanNet)
        if abs(fx - dfx) > 1.0 or abs(fy - dfy) > 1.0:
            print(
                f"[streaming] ScanNet: color intrinsics (fx={fx:.1f}) differ from "
                f"depth intrinsics (fx={dfx:.1f}). Using depth intrinsics for "
                f"backprojection, color intrinsics for rendering."
            )

        self._frames: List[StreamingRGBDFrame] = []
        for color_file in color_files:
            fid = _scannet_frame_id_from_color(color_file)
            pose_file = os.path.join(pose_dir, f"{fid}.txt")
            c2w = _read_scannet_pose(pose_file)
            if c2w is None:
                continue
            depth_file = os.path.join(depth_dir, f"{fid}.png")
            self._frames.append(StreamingRGBDFrame(
                index=len(self._frames),
                timestamp=float(fid) if fid.isdigit() else float(len(self._frames)),
                rgb_path=color_file,
                depth_path=depth_file if os.path.exists(depth_file) else None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                depth_scale=depth_scale,
                depth_fx=dfx, depth_fy=dfy, depth_cx=dcx, depth_cy=dcy,
                depth_width=depth_width, depth_height=depth_height,
            ))
        print(f"[streaming] ScanNet: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class ScanNetSensFrameSource:
    """
    Ordered frames streamed directly from a ScanNet .sens binary file.
    No pre-extraction needed — images are stored in-memory as compressed bytes.
    Memory: ~200KB/frame (JPEG) + ~75KB/frame (depth) ≈ 275KB/frame.
    At stride=5 and 2000 frames → ~110MB.  At stride=1 and 2000 frames → ~550MB.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 10,
        depth_scale: float = 1000.0,
    ):
        import numpy as np
        from scene.readers.scannet import (
            _find_scannet_sens,
            _read_sens_header,
            _read_sens_frame_record,
        )

        sens_path = _find_scannet_sens(path)
        if sens_path is None:
            raise FileNotFoundError(f"No .sens file found in {path}")

        self._frames: List[StreamingRGBDFrame] = []

        with open(sens_path, "rb") as f:
            header = _read_sens_header(f)
            num_source = header["num_frames"]
            fx = float(header["intrinsic_color"][0, 0])
            fy = float(header["intrinsic_color"][1, 1])
            cx = float(header["intrinsic_color"][0, 2])
            cy = float(header["intrinsic_color"][1, 2])
            dfx = float(header["intrinsic_depth"][0, 0])
            dfy = float(header["intrinsic_depth"][1, 1])
            dcx = float(header["intrinsic_depth"][0, 2])
            dcy = float(header["intrinsic_depth"][1, 2])
            color_w = header["color_width"]
            color_h = header["color_height"]
            depth_w = header["depth_width"]
            depth_h = header["depth_height"]
            ds = depth_scale if depth_scale > 0 else float(header.get("depth_shift", 1000.0))

            if abs(fx - dfx) > 1.0 or abs(fy - dfy) > 1.0:
                print(
                    f"[streaming] ScanNet .sens: color intrinsics (fx={fx:.1f}) differ from "
                    f"depth intrinsics (fx={dfx:.1f}). Using depth intrinsics for backprojection."
                )

            emitted = 0
            for src_idx in range(num_source):
                c2w, ts_color, ts_depth, color_data, depth_data = _read_sens_frame_record(f)
                if frame_stride > 1 and src_idx % frame_stride != 0:
                    continue
                if max_frames > 0 and emitted >= max_frames:
                    break
                if not np.isfinite(c2w).all() or abs(np.linalg.det(c2w[:3, :3])) < 1e-6:
                    continue
                self._frames.append(StreamingRGBDFrame(
                    index=emitted,
                    timestamp=float(ts_color),
                    rgb_path="",
                    depth_path=None,
                    c2w=c2w.astype(np.float32),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    width=color_w, height=color_h,
                    depth_scale=ds,
                    depth_fx=dfx, depth_fy=dfy, depth_cx=dcx, depth_cy=dcy,
                    depth_width=depth_w, depth_height=depth_h,
                    _rgb_bytes=color_data if color_data else None,
                    _depth_bytes=depth_data if depth_data else None,
                    _sens_header=header,
                ))
                emitted += 1

        print(
            f"[streaming] ScanNet .sens: {len(self._frames)} frames loaded "
            f"(stride={frame_stride}, source={num_source} total) from {sens_path}"
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
