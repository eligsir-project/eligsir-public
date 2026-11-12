"""Frame-to-frame Open3D RGB-D odometry frontend.

Wraps the Open3D Hybrid/ICP odometry originally in
``utils.streaming_frames._estimate_open3d_odometry``.

A registry-free tracking *algorithm*, like its sibling
:mod:`eligsir.tracking.open3d_icp`: it subclasses nothing, registers with
nothing, and imports no registry. #1075 retired the ``SLAMBackend`` ABC and its
``@register_backend`` decorator, which is why this module moved here from
``eligsir/slam/adapters/``. Selection by the ``open3d_odometry_live`` name
now happens in :mod:`eligsir.tracking.factory`; binding to the canonical
contract happens in :class:`eligsir.tracking.providers.FrontendTrackingProvider`.

The estimator numerics below are unchanged by that move -- only the base class,
the registration decorator and the unused ``gaussian_ctx`` parameter were
removed.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame, TrackingResult


def _import_open3d():
    """Import Open3D or raise a clear error message.

    Returns:
        The ``open3d`` module.
    """
    import sys as _sys

    if not hasattr(_sys.stdout, "isatty"):
        _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
    if not hasattr(_sys.stderr, "isatty"):
        _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d_odometry_live requires Open3D. "
            "Install it in the container: pip install open3d"
        ) from exc
    return o3d


def _np_to_o3d_rgbd(o3d, rgb: np.ndarray, depth: np.ndarray):
    """Convert numpy rgb/depth to an Open3D RGBDImage.

    Args:
        o3d: The open3d module.
        rgb: HxWx3 float32 [0, 1] or uint8.
        depth: HxW float32 metres.

    Returns:
        ``o3d.geometry.RGBDImage``.
    """
    if rgb.dtype != np.uint8:
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        rgb_u8 = rgb
    color_o3d = o3d.geometry.Image(rgb_u8)
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=8.0,
        convert_rgb_to_intensity=False,
    )


class Open3DOdometryTracker:
    """Frame-to-frame Open3D RGB-D odometry.

    Implements RGBD Hybrid or ICP odometry, mapping Open3D
    fitness/inlier_rmse/info_trace to :class:`contracts.types.TrackingResult`
    confidence using a sigmoid on fitness.

    Args:
        method: ``"hybrid"`` (default), ``"intensity"``, or ``"depth"`` for
            RGBD odometry; ``"icp"`` for point-to-plane ICP.
        depth_min: Minimum depth (metres). Default 0.1.
        depth_max: Maximum depth (metres). Default 8.0.
        depth_diff_max: Max depth difference for ICP. Default 0.07.
        icp_max_distance: ICP correspondence distance. Default 0.07.
        min_fitness: Fitness below this → confidence 0; above 1 → confidence 1.
            Default 0.3.
    """

    name = "open3d_odometry_live"

    def __init__(
        self,
        method: str = "hybrid",
        depth_min: float = 0.1,
        depth_max: float = 8.0,
        depth_diff_max: float = 0.07,
        icp_max_distance: float = 0.07,
        min_fitness: float = 0.3,
    ) -> None:
        self._method = method
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._depth_diff_max = depth_diff_max
        self._icp_max_distance = icp_max_distance
        self._min_fitness = min_fitness

        self._prev_frame: Optional[RGBDFrame] = None
        self._prev_c2w: Optional[np.ndarray] = None
        self._last_pose: Optional[Pose] = None

    def initialize(self, first_frame: RGBDFrame) -> TrackingResult:
        """Accept first frame as the world-frame anchor.

        Args:
            first_frame: First RGB-D frame; pose (if set) becomes the anchor.

        Returns:
            :class:`TrackingResult` with confidence 1.0.
        """
        if first_frame.pose is not None:
            pose = first_frame.pose
        else:
            pose = Pose.identity()
        self._prev_frame = first_frame
        self._prev_c2w = pose.matrix.copy()
        self._last_pose = pose
        return TrackingResult(
            pose=pose,
            confidence=1.0,
            valid=True,
            method=self.name,
        )

    def track(
        self,
        frame: RGBDFrame,
        T_pred: Optional[Pose] = None,
    ) -> TrackingResult:
        """Estimate pose for *frame* relative to the previous accepted frame.

        Args:
            frame: Incoming RGB-D frame.
            T_pred: Optional motion prediction (used as odometry initialisation).

        Returns:
            :class:`TrackingResult` with pose and Open3D fitness/rmse stats.
        """
        t0 = time.perf_counter()
        o3d = _import_open3d()

        if self._prev_frame is None:
            return self.initialize(frame)

        intr = frame.intrinsics
        result = self._run_odometry(o3d, self._prev_frame, frame, intr, T_pred)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if result is None:
            confidence = 0.0
            valid = False
            c2w = self._prev_c2w.copy() if self._prev_c2w is not None else np.eye(4, dtype=np.float32)
            fitness = None
            inlier_rmse = None
        else:
            T_rel, fitness, inlier_rmse, info_trace = result
            c2w = (self._prev_c2w @ T_rel).astype(np.float32)
            confidence = float(np.clip(fitness, 0.0, 1.0))
            valid = confidence >= self._min_fitness

        if valid:
            self._prev_frame = frame
            self._prev_c2w = c2w

        pose = Pose(c2w)
        self._last_pose = pose

        delta_m = None
        if self._last_pose is not None:
            dt = c2w[:3, 3] - (self._prev_c2w[:3, 3] if self._prev_c2w is not None else np.zeros(3))
            delta_m = float(np.linalg.norm(dt))

        return TrackingResult(
            pose=pose,
            confidence=confidence,
            valid=valid,
            method=self.name,
            inlier_ratio=fitness,
            depth_rmse=inlier_rmse,
            pose_delta_m=delta_m,
            tracking_time_ms=elapsed_ms,
            raw_stats={
                "fitness": fitness,
                "inlier_rmse": inlier_rmse,
            },
        )

    def _run_odometry(self, o3d, prev: RGBDFrame, curr: RGBDFrame, intr: CameraIntrinsics, T_pred):
        """Run Open3D RGBD odometry between prev and curr.

        Args:
            o3d: Open3D module.
            prev: Previous frame.
            curr: Current frame.
            intr: Camera intrinsics.
            T_pred: Optional SE3 prediction (4×4 numpy) as init.

        Returns:
            Tuple (T_rel, fitness, inlier_rmse, info_trace) or None on failure.
        """
        try:
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(intr.width),
                int(intr.height),
                intr.fx,
                intr.fy,
                intr.cx,
                intr.cy,
            )
            option = o3d.pipelines.odometry.OdometryOption(
                depth_min=self._depth_min,
                depth_max=self._depth_max,
                depth_diff_max=self._depth_diff_max,
            )

            prev_rgbd = _np_to_o3d_rgbd(o3d, prev.rgb, prev.depth)
            curr_rgbd = _np_to_o3d_rgbd(o3d, curr.rgb, curr.depth)

            init = np.eye(4) if T_pred is None else T_pred.w2c @ (self._prev_c2w or np.eye(4))

            if self._method == "icp":
                success, T_rel, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    prev_rgbd,
                    curr_rgbd,
                    intrinsic,
                    init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
                    option,
                )
            else:
                success, T_rel, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    prev_rgbd,
                    curr_rgbd,
                    intrinsic,
                    init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option,
                )

            if not success:
                return None

            fitness = float(np.sum(info) / max(1, info.shape[0] * info.shape[1]))
            inlier_rmse = float(np.sqrt(np.mean(info**2))) if info.size > 0 else None
            return T_rel.astype(np.float32), fitness, inlier_rmse, None

        except Exception:
            return None
