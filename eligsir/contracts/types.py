"""Internal data types crossing the SLAM↔Gaussian mapping boundary.

These types are the mapper-facing contract, not a public transport wire
contract. ROS2 message conversion and transport ownership live in the
separate ``eligsir-ros2`` repository; this core package remains transport
neutral.

Convention table:

- ``Pose``: 4×4 camera-to-world, ``np.float32``, metric (meters),
  OpenCV/COLMAP convention (x-right, y-down, z-forward).
- ``RGBDFrame.rgb``: HxWx3, ``float32``, range [0, 1].
- ``RGBDFrame.depth``: HxW, ``float32``, meters (depth scale applied).
- ``TrackingResult.confidence``: ``float`` in [0, 1].
- Timestamps: seconds, ``float64``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraIntrinsics:
    """THE canonical pinhole intrinsics/calibration record (issue #1074).

    This is the single definition in the codebase. It replaces the former
    duplicate ``utils.rgbd_frames.CameraIntrinsics``, whose disjoint field set
    (``depth_scale``, frame names, distortion) is merged in here rather than
    kept alongside. Equivalent information is carried by
    ``sensor_msgs/CameraInfo`` at ROS boundaries; this record itself contains
    no ROS type, import or convention.

    ``kw_only=True`` is deliberate and load-bearing: the two merged definitions
    had DIFFERENT positional orders (``(width, height, fx, ...)`` vs
    ``(fx, fy, cx, cy, width, height)``), so a positional construction that
    survived the merge would silently bind ``width`` to ``fx``. That failure is
    silent and costly (see this repo's ``tum-wrong-intrinsics-impact`` note:
    ~5.6-8.2 dB, no crash), so every call site must name its arguments.

    ``frozen=True``: calibration is a fact about the sensor for the lifetime of
    a stream, never a scratch value to patch in place. Build a new record (or
    use ``dataclasses.replace``) instead.

    Args:
        fx/fy/cx/cy: Colour-sensor focal length and principal point (pixels).
        width/height: Colour image size (pixels).
        depth_fx/depth_fy/depth_cx/depth_cy: Depth-sensor intrinsics; each
            defaults to its colour counterpart when omitted.
        depth_width/depth_height: Depth image size; default to colour size.
        depth_scale: Divisor mapping raw depth counts to metres (e.g. 1000.0
            for 16UC1 millimetres). Units metadata, not a pixel transform.
        camera_frame/world_frame: Frame names for downstream transport
            annotation. Plain strings -- no ROS/tf2 type is referenced here.
        distortion_model/D: Distortion model name and coefficients. ``D`` is
            normalized to a tuple so a JSON round-trip cannot change the
            record's identity or its fingerprint.
        intrinsics_source: Provenance of these numbers (e.g. ``"K"``,
            ``"P_rectified"``, a dataset-specific tag).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    depth_scale: float = 1000.0
    camera_frame: str = ""
    world_frame: str = ""
    distortion_model: str = ""
    D: Optional[tuple] = None
    intrinsics_source: str = "K"

    def __post_init__(self) -> None:
        for attr, fallback in (
            ("depth_fx", self.fx),
            ("depth_fy", self.fy),
            ("depth_cx", self.cx),
            ("depth_cy", self.cy),
            ("depth_width", self.width),
            ("depth_height", self.height),
        ):
            if getattr(self, attr) is None:
                object.__setattr__(self, attr, fallback)
        if self.D is not None and not isinstance(self.D, tuple):
            object.__setattr__(self, "D", tuple(self.D))

    @property
    def K(self) -> np.ndarray:
        """Return 3×3 intrinsic matrix K (colour sensor)."""
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )

    @property
    def K_depth(self) -> np.ndarray:
        """Return 3×3 intrinsic matrix K for the depth sensor."""
        return np.array(
            [
                [self.depth_fx, 0, self.depth_cx],
                [0, self.depth_fy, self.depth_cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )


@dataclass
class Pose:
    """4×4 camera-to-world rigid transform.

    Uses OpenCV/COLMAP convention: x-right, y-down, z-forward.
    Matrix is ``np.float32`` and metric (translation in meters).

    Args:
        matrix: 4×4 c2w transform, shape (4, 4), dtype float32.
    """

    matrix: np.ndarray  # (4, 4) float32 c2w

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float32)
        if m.shape != (4, 4):
            raise ValueError(f"Pose matrix must be (4, 4), got {m.shape}")
        self.matrix = m

    @classmethod
    def identity(cls) -> "Pose":
        """Return identity pose (camera at world origin)."""
        return cls(np.eye(4, dtype=np.float32))

    @classmethod
    def from_Rt(cls, R: np.ndarray, t: np.ndarray) -> "Pose":
        """Construct from 3×3 rotation and 3-vector translation.

        Args:
            R: 3×3 rotation matrix.
            t: 3-vector translation (meters).

        Returns:
            Pose with c2w = [[R, t], [0, 1]].
        """
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R
        m[:3, 3] = t
        return cls(m)

    @property
    def c2w(self) -> np.ndarray:
        """Camera-to-world matrix (4×4 float32)."""
        return self.matrix

    @property
    def w2c(self) -> np.ndarray:
        """World-to-camera (inverse) matrix (4×4 float32)."""
        return np.linalg.inv(self.matrix).astype(np.float32)

    @property
    def translation(self) -> np.ndarray:
        """Camera position in world frame (3-vector float32)."""
        return self.matrix[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        """3×3 rotation submatrix (float32)."""
        return self.matrix[:3, :3].copy()

    def compose(self, other: "Pose") -> "Pose":
        """Return ``self @ other`` as a new Pose.

        Args:
            other: Right-hand pose.

        Returns:
            Composed pose.
        """
        return Pose(self.matrix @ other.matrix)

    def relative_to(self, reference: "Pose") -> "Pose":
        """Return relative pose T_ref_self = inv(ref) @ self.

        Args:
            reference: Reference (anchor) pose.

        Returns:
            Transform from reference to self.
        """
        return Pose(reference.w2c @ self.matrix)


@dataclass
class RGBDFrame:
    """Canonical RGB-D frame crossing subsystem boundaries.

    Consolidates ``StreamingRGBDFrame`` (streaming) and
    ``utils.rgbd_frames.RGBDFrame`` (preprocessing) into one type.

    Args:
        index: Frame index in the source sequence.
        timestamp: Capture time (seconds, float64).
        rgb: Colour image, HxWx3 float32 [0, 1].
        depth: Depth image, HxW float32 metres.
        intrinsics: Camera intrinsics.
        pose: Camera-to-world pose, or None if not yet estimated.
    """

    index: int
    timestamp: float
    rgb: np.ndarray        # (H, W, 3) float32 [0, 1]
    depth: np.ndarray      # (H, W) float32 metres
    intrinsics: CameraIntrinsics
    pose: Optional[Pose] = None

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.rgb.shape[1]


@dataclass
class TrackingResult:
    """Output of a :class:`contracts.slam_backend.SLAMBackend` track call.

    One ``to_tensorboard_dict()`` so all adapters log identically.

    Args:
        pose: Estimated camera-to-world pose.
        confidence: Tracking confidence in [0, 1].
        valid: Whether the estimate is usable.
        method: Name of the backend that produced this result.
        depth_rmse: Optional depth residual (metres).
        rgb_residual: Optional mean photometric residual.
        inlier_ratio: Optional fraction of inlier correspondences.
        pose_delta_m: Optional translation magnitude from last pose (metres).
        pose_delta_deg: Optional rotation magnitude from last pose (degrees).
        render_alpha_coverage: Optional fraction of pixels with alpha > threshold
            (used by :class:`eligsir.slam.adapters.gs_rendered_rgbd`).
        tracking_time_ms: Wall-clock time for this track call (ms).
        raw_stats: Untyped dict for adapter-specific diagnostics.
    """

    pose: Pose
    confidence: float
    valid: bool
    method: str
    depth_rmse: Optional[float] = None
    rgb_residual: Optional[float] = None
    inlier_ratio: Optional[float] = None
    pose_delta_m: Optional[float] = None
    pose_delta_deg: Optional[float] = None
    render_alpha_coverage: Optional[float] = None
    tracking_time_ms: Optional[float] = None
    raw_stats: Dict[str, Any] = field(default_factory=dict)

    def to_tensorboard_dict(self) -> Dict[str, float]:
        """Return scalar metrics for TensorBoard logging.

        Returns:
            Dict mapping metric names to float values, skipping None fields.
        """
        d: Dict[str, float] = {
            "tracking/confidence": self.confidence,
            "tracking/valid": float(self.valid),
        }
        for attr, key in [
            ("depth_rmse", "tracking/depth_rmse"),
            ("rgb_residual", "tracking/rgb_residual"),
            ("inlier_ratio", "tracking/inlier_ratio"),
            ("pose_delta_m", "tracking/pose_delta_m"),
            ("pose_delta_deg", "tracking/pose_delta_deg"),
            ("render_alpha_coverage", "tracking/render_alpha_coverage"),
            ("tracking_time_ms", "tracking/time_ms"),
        ]:
            v = getattr(self, attr)
            if v is not None:
                d[key] = float(v)
        return d
