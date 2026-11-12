"""HyperSimFrameSource -- HyperSim dataset frame source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


class HyperSimFrameSource:
    """
    Ordered frames from an ML-HyperSim scene (ai_NNN_NNN/_detail/cam_00/).

    Reads camera poses from HDF5 files and lazily loads color + depth at
    __init__ time (all frames stored in-memory as JPEG/PNG bytes).
    Requires h5py (pip install h5py).

    depth_scale=1000.0 — depth stored as uint16 mm in _depth_bytes.
    """

    def __init__(
        self,
        path: str,
        cam_id: str = "cam_00",
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import numpy as np
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HyperSim streaming. "
                "Install it with: pip install h5py"
            )

        cam_dir = os.path.join(path, "_detail", cam_id)
        final_hdf5_dir = os.path.join(path, "images", f"scene_{cam_id}_final_hdf5")
        geom_hdf5_dir = os.path.join(path, "images", f"scene_{cam_id}_geometry_hdf5")

        for d in (cam_dir, final_hdf5_dir, geom_hdf5_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"HyperSim directory not found: {d}")

        # Read scene scale: positions are in asset units; scale to meters
        scene_meta = os.path.join(path, "_detail", "metadata_scene.csv")
        meters_per_unit = 0.0254  # default: 1 inch
        if os.path.exists(scene_meta):
            import csv
            with open(scene_meta) as f:
                for row in csv.DictReader(f):
                    if row.get("parameter_name") == "meters_per_asset_unit":
                        meters_per_unit = float(row["parameter_value"])

        # Camera poses
        with h5py.File(os.path.join(cam_dir, "camera_keyframe_positions.hdf5"), "r") as f:
            positions = f["dataset"][:].astype(np.float64)  # [N, 3] in asset units
        with h5py.File(os.path.join(cam_dir, "camera_keyframe_orientations.hdf5"), "r") as f:
            orientations = f["dataset"][:].astype(np.float64)  # [N, 3, 3]

        n_keyframes = positions.shape[0]

        # Map keyframe index → actual HDF5 file frame number
        fi_hdf5 = os.path.join(cam_dir, "camera_keyframe_frame_indices.hdf5")
        if os.path.exists(fi_hdf5):
            with h5py.File(fi_hdf5, "r") as f:
                file_frame_indices = f["dataset"][:].astype(int)  # [N]
        else:
            file_frame_indices = np.arange(n_keyframes, dtype=int)

        # Intrinsics: HyperSim default is 1024×768, fov=60° horizontal
        # (V-Ray renderer with standard settings; we use a fixed approximation)
        import math
        width, height = 1024, 768
        fov_h_deg = 60.0
        fx = width / (2.0 * math.tan(math.radians(fov_h_deg) * 0.5))
        fy = fx
        cx_f = width * 0.5
        cy_f = height * 0.5

        # Build keyframe index list (into positions/orientations arrays)
        kf_indices = list(range(n_keyframes))
        if frame_stride > 1:
            kf_indices = kf_indices[::frame_stride]
        if max_frames > 0:
            kf_indices = kf_indices[:max_frames]

        self._frames: List[StreamingRGBDFrame] = []
        import io as _io
        from PIL import Image

        for out_idx, kfi in enumerate(kf_indices):
            # kfi = index into positions/orientations; fi = HDF5 file frame number
            fi = int(file_frame_indices[kfi])
            # Build c2w from position + orientation
            # HyperSim orientation: columns = [right, up, backward] in world space
            # We use RDF convention: x=right, y=down, z=forward
            ori = orientations[kfi]  # [3, 3]
            pos = positions[kfi] * meters_per_unit  # convert to meters
            right = ori[:, 0]
            up = ori[:, 1]
            backward = ori[:, 2]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right.astype(np.float32)
            c2w[:3, 1] = (-up).astype(np.float32)       # down = -up
            c2w[:3, 2] = (-backward).astype(np.float32) # forward = -backward
            c2w[:3, 3] = pos.astype(np.float32)

            if not np.isfinite(c2w).all():
                continue

            # Color: float32 linear HDR → uint8 gamma-corrected
            color_path = os.path.join(final_hdf5_dir, f"frame.{fi:04d}.color.hdf5")
            depth_path = os.path.join(geom_hdf5_dir, f"frame.{fi:04d}.depth_meters.hdf5")

            if not os.path.exists(color_path) or not os.path.exists(depth_path):
                continue

            try:
                with h5py.File(color_path, "r") as f:
                    color_data = f["dataset"][:].astype(np.float32)  # [H, W, 3] linear
                # Gamma correction + clip
                color_uint8 = np.clip(np.power(np.maximum(color_data, 0.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
                rgb_img = Image.fromarray(color_uint8, mode="RGB")
                rgb_buf = _io.BytesIO()
                rgb_img.save(rgb_buf, format="JPEG", quality=90)
                rgb_bytes = rgb_buf.getvalue()

                with h5py.File(depth_path, "r") as f:
                    depth_m = f["dataset"][:].astype(np.float32)  # [H, W] in meters
                # Convert to uint16 mm; cap at 65.535m
                depth_mm = np.round(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
                depth_img = Image.fromarray(depth_mm, mode="I;16")
                depth_buf = _io.BytesIO()
                depth_img.save(depth_buf, format="PNG")
                depth_bytes = depth_buf.getvalue()

            except Exception as e:
                print(f"[streaming] HyperSim: skipping frame {fi}: {e}", flush=True)
                continue

            self._frames.append(StreamingRGBDFrame(
                index=out_idx,
                timestamp=float(fi),
                rgb_path="",
                depth_path=None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx_f, cy=cy_f,
                width=width, height=height,
                depth_scale=1000.0,
                _rgb_bytes=rgb_bytes,
                _depth_bytes=depth_bytes,
            ))

        print(f"[streaming] HyperSim: {len(self._frames)} frames loaded from {path} (cam={cam_id})")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
