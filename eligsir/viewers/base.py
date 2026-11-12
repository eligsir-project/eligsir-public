"""Backend-agnostic viewer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame, TrackingResult

if TYPE_CHECKING:
    from eligsir.map.views import GaussianMapReadView


class Viewer(ABC):
    """Abstract live-introspection viewer."""

    @abstractmethod
    def close(self) -> None:
        """Release any viewer resources."""

    def log_rgb(self, frame: RGBDFrame, step: int) -> None:
        """Log a live RGB image."""

    def log_depth(self, frame: RGBDFrame, step: int) -> None:
        """Log a live depth image."""

    def log_rendered(self, rgb: np.ndarray, depth: Optional[np.ndarray], step: int) -> None:
        """Log a rendered intermediate view from the Gaussian map."""

    def log_gaussian_map(
        self,
        positions: np.ndarray,
        colors: Optional[np.ndarray],
        scales: Optional[np.ndarray],
        step: int,
    ) -> None:
        """Log Gaussian positions as a point-cloud or splat proxy."""

    def log_gaussian_model(
        self, view: "GaussianMapReadView", step: int, map_version: Optional[int] = None
    ) -> None:
        """Log a full Gaussian map when the backend supports it.

        Takes a read-only ``GaussianMapReadView``, never the live mutable
        model -- see docs/architecture/gaussian_ownership_audit.md and the
        Wave 1 consolidation plan §5/§6 for why the viewer boundary must not
        see the model object itself. Callers without a long-lived
        ``GaussianMapBackend`` yet can build one via
        ``eligsir.map.views.read_view_from_model(gaussians)``.
        """

    def log_trajectory(
        self,
        poses_est: List[Pose],
        poses_gt: Optional[List[Pose]],
        step: int,
    ) -> None:
        """Log estimated and optional ground-truth trajectory."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar process metrics."""

    def log_tracking(self, result: TrackingResult, step: int) -> None:
        """Log a SLAM adapter tracking result."""

    def log_camera(self, frame: RGBDFrame, pose: Pose, step: int) -> None:
        """Log the current camera image, intrinsics, and pose."""

    def log_keyframe(self, pose: Pose, intrinsics: CameraIntrinsics, step: int) -> None:
        """Log a newly admitted keyframe."""

    def log_controller_stats(self, stats: Any, step: int) -> None:
        """Log online controller or scheduler diagnostics."""

    def set_params_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for live mapping-parameter updates."""


class NullViewer(Viewer):
    """No-op viewer used when ``--viewer none``."""

    def close(self) -> None:
        """No-op."""
