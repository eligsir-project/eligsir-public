"""Rerun live viewer — primary introspection backend.

On construction, spawns the Rerun viewer (or connects to an existing one via
gRPC) and **pushes the correct blueprint** so the layout opens immediately
without any manual arrangement.

Blueprint layout:
- 3D panel: Gaussian map (point cloud), camera frustum, estimated trajectory,
  GT trajectory (if available).
- 2D panel row: live RGB, live depth (colormapped), rendered RGB, residual.
- Time-series panel: PSNR, depth-RMSE, FPS, Gaussian count, dead count,
  energy mean, tracking confidence.

Rerun runs headless in-container and streams to a host viewer via gRPC
(port 9876 by default, mapped in docker-compose.yml).

Connect on host: ``rerun --connect 127.0.0.1:9876``
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from eligsir.contracts.types import Pose, RGBDFrame, TrackingResult
from eligsir.viewers.base import Viewer


def _import_rerun():
    """Import rerun-sdk or raise a descriptive error.

    Returns:
        The ``rerun`` module.
    """
    try:
        import rerun as rr

        return rr
    except ImportError as exc:
        raise ImportError(
            "Rerun viewer requires 'rerun-sdk'. "
            "Install it: pip install 'rerun-sdk>=0.21' "
            "or use the viewer Docker service."
        ) from exc


def _build_blueprint(rr):
    """Build the canonical EliGSiR Rerun blueprint.

    Args:
        rr: The rerun module (already imported).

    Returns:
        A ``rerun.blueprint`` object to pass to ``rr.send_blueprint``.
    """
    rrb = rr.blueprint

    gaussian_map_view = rrb.Spatial3DView(
        name="Gaussian Map",
        origin="/world",
        contents=[
            "/world/gaussians",
            "/world/trajectory/estimated",
            "/world/trajectory/gt",
            "/world/camera",
        ],
    )

    rgb_view = rrb.Spatial2DView(name="RGB", origin="/sensor/rgb")
    depth_view = rrb.Spatial2DView(name="Depth", origin="/sensor/depth")
    rendered_view = rrb.Spatial2DView(name="Rendered", origin="/render/rgb")
    residual_view = rrb.Spatial2DView(name="Residual", origin="/render/residual")

    metrics_view = rrb.TimeSeriesView(
        name="Metrics",
        origin="/metrics",
    )

    tracking_view = rrb.TimeSeriesView(
        name="Tracking",
        origin="/tracking",
    )

    sensor_row = rrb.Horizontal(rgb_view, depth_view, rendered_view, residual_view)
    bottom_row = rrb.Horizontal(metrics_view, tracking_view)

    return rrb.Blueprint(
        rrb.Vertical(
            gaussian_map_view,
            sensor_row,
            bottom_row,
            row_shares=[3, 2, 1],
        ),
        collapse_panels=True,
    )


class RerunViewer(Viewer):
    """Live Rerun introspection viewer for EliGSiR.

    Args:
        app_id: Rerun application identifier. Default ``"eligsir"``.
        serve_port: gRPC port to serve on (for headless container mode).
            Set to 0 to disable serving (spawn local viewer instead).
            Default 9876.
        spawn: If True and serve_port==0, spawn a local Rerun viewer process.
            Default False.
        max_gaussians_logged: Cap Gaussian count logged per step (random sample
            if model exceeds this). Default 50000.
    """

    def __init__(
        self,
        app_id: str = "eligsir",
        serve_port: int = 9876,
        spawn: bool = False,
        max_gaussians_logged: int = 50_000,
    ) -> None:
        self._rr = _import_rerun()
        rr = self._rr
        self._max_gaussians = max_gaussians_logged

        rr.init(app_id, spawn=False)

        if serve_port > 0:
            rr.serve_grpc(port=serve_port, open_browser=False)
        elif spawn:
            rr.spawn()

        blueprint = _build_blueprint(rr)
        rr.send_blueprint(blueprint)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    def close(self) -> None:
        """Disconnect from Rerun."""
        pass  # rerun-sdk manages connection lifecycle

    def log_rgb(self, frame: RGBDFrame, step: int) -> None:
        """Log live RGB image.

        Args:
            frame: Source RGB-D frame.
            step: Frame / training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)
        rgb_u8 = (np.clip(frame.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        rr.log("sensor/rgb", rr.Image(rgb_u8))

    def log_depth(self, frame: RGBDFrame, step: int) -> None:
        """Log live depth image (colormapped).

        Args:
            frame: Source RGB-D frame.
            step: Frame / training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)
        rr.log("sensor/depth", rr.DepthImage(frame.depth, meter=1.0))

    def log_rendered(self, rgb: np.ndarray, depth: Optional[np.ndarray], step: int) -> None:
        """Log rendered intermediate view.

        Args:
            rgb: Rendered RGB, HxWx3 float32 [0, 1].
            depth: Optional rendered depth, HxW float32 metres.
            step: Training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        rr.log("render/rgb", rr.Image(rgb_u8))
        if depth is not None:
            rr.log("render/depth", rr.DepthImage(depth, meter=1.0))

    def log_gaussian_map(
        self,
        positions: np.ndarray,
        colors: Optional[np.ndarray],
        scales: Optional[np.ndarray],
        step: int,
    ) -> None:
        """Log Gaussian centres as a live 3D point cloud.

        Args:
            positions: (N, 3) float32 centres.
            colors: Optional (N, 3) float32 RGB [0, 1].
            scales: Unused (Rerun point-cloud doesn't render ellipsoids yet).
            step: Training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)

        n = len(positions)
        if n > self._max_gaussians:
            idx = np.random.choice(n, self._max_gaussians, replace=False)
            positions = positions[idx]
            if colors is not None:
                colors = colors[idx]

        kwargs: Dict[str, Any] = {"positions": positions}
        if colors is not None:
            colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
            kwargs["colors"] = colors_u8
        kwargs["radii"] = 0.005

        rr.log("world/gaussians", rr.Points3D(**kwargs))

    def log_trajectory(
        self,
        poses_est: List[Pose],
        poses_gt: Optional[List[Pose]],
        step: int,
    ) -> None:
        """Log estimated and GT camera trajectory.

        Args:
            poses_est: Estimated poses.
            poses_gt: Optional GT poses.
            step: Training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)

        est_pts = np.array([p.translation for p in poses_est], dtype=np.float32)
        rr.log("world/trajectory/estimated", rr.LineStrips3D([est_pts], colors=[[0, 200, 255]]))

        if poses_gt and len(poses_gt) > 1:
            gt_pts = np.array([p.translation for p in poses_gt], dtype=np.float32)
            rr.log("world/trajectory/gt", rr.LineStrips3D([gt_pts], colors=[[255, 150, 0]]))

        # Log current camera frustum
        if poses_est:
            last_pose = poses_est[-1]
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=last_pose.translation,
                    mat3x3=last_pose.rotation,
                ),
            )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics as Rerun time-series.

        Args:
            metrics: Dict of metric name → float value.
            step: Training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)
        for key, value in metrics.items():
            safe_key = key.replace("/", "_")
            rr.log(f"metrics/{safe_key}", rr.Scalar(float(value)))

    def log_tracking(self, result: TrackingResult, step: int) -> None:
        """Log tracking result metrics.

        Args:
            result: Tracking result from the SLAM adapter.
            step: Training step.
        """
        rr = self._rr
        rr.set_time_sequence("step", step)
        for key, value in result.to_tensorboard_dict().items():
            safe_key = key.replace("/", "_")
            rr.log(f"tracking/{safe_key}", rr.Scalar(value))

    def set_params_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Register live parameter update callback (boilerplate for future controls).

        Args:
            cb: Callable accepting a dict of param→value updates.
        """
        self._params_callback = cb
