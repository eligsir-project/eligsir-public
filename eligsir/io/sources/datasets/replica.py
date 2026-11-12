"""ReplicaFrameSource -- Replica dataset frame source.

Extracted from eligsir.utils/streaming_frames.py (Wave 2 split, see
docs/architecture/streaming_frames_wave2_scope.md).
"""
from __future__ import annotations

import os
from typing import Iterator, List


from eligsir.io.sources.decode_cache import (
    StreamingRGBDFrame,
)


class ReplicaFrameSource:
    """
    Ordered frames synthesised from a Replica scene mesh (mesh.ply).

    Generates a virtual camera trajectory around the mesh using the same
    `_replica_camera_path` as the offline Replica reader, then renders RGB
    and depth images using Open3D raycasting over a triangulated Replica mesh.

    Image domain (issue #641): frames carry raw arrays, NOT encoded bytes --
    ``_rgb_np`` is (H, W, 3) uint8 in the canonical Replica RGB domain
    (``scene.readers.replica.replica_rgb_to_uint8``) and ``_depth_np`` is
    (H, W) uint16 millimetres, i.e. depth_scale=1000.0. No codec (previously
    JPEG q92 for RGB, PNG for depth) sits between the raycaster and training /
    evaluation.
    """

    def __init__(
        self,
        path: str,
        num_views: int = 120,
        width: int = 640,
        height: int = 480,
        fov_degrees: float = 70.0,
        render_points: int = 300000,   # kept for API compat; unused
        splat_radius: int = 1,         # kept for API compat; unused
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import math

        import numpy as np

        try:
            import sys as _sys
            if not hasattr(_sys.stdout, "isatty"):
                _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
            if not hasattr(_sys.stderr, "isatty"):
                _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
            import open3d as o3d
            import open3d.t.geometry as o3tg
        except ImportError:
            raise ImportError(
                "open3d is required for ReplicaFrameSource. "
                "Install it with: pip install open3d"
            )
        from scene.readers.replica import (
            _load_replica_raycast_mesh,
            _raycast_replica_frame,
            _replica_raycast_camera_path,
        )

        mesh_path = os.path.join(path, "mesh.ply")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Replica mesh not found: {mesh_path}")

        vertices_np, mesh_t = _load_replica_raycast_mesh(mesh_path)
        scene = o3tg.RaycastingScene()
        scene.add_triangles(mesh_t)

        c2ws = _replica_raycast_camera_path(scene, vertices_np, int(num_views), fov_degrees=fov_degrees)
        if frame_stride > 1:
            c2ws = c2ws[::frame_stride]
        if max_frames > 0:
            c2ws = c2ws[:max_frames]

        fx = width / (2.0 * math.tan(math.radians(fov_degrees) * 0.5))
        fy = fx
        cx = width * 0.5
        cy = height * 0.5

        self._frames: List[StreamingRGBDFrame] = []
        for idx, c2w in enumerate(c2ws):
            rgb_np, depth_np = _raycast_replica_frame(
                scene, mesh_t, c2w, width, height, fx, fy, cx, cy
            )
            self._frames.append(StreamingRGBDFrame(
                index=idx,
                timestamp=float(idx) / 30.0,
                rgb_path="",
                depth_path=None,
                c2w=c2w.astype(np.float32),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=int(width), height=int(height),
                depth_scale=1000.0,
                _rgb_np=rgb_np,
                _depth_np=depth_np,
            ))

        print(
            f"[streaming] Replica: {len(self._frames)} raycasted frames from triangulated {mesh_path} "
            f"(vertices={len(vertices_np)})"
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames
