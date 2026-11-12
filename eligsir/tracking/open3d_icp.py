"""True point-cloud ICP tracking frontend implemented with Open3D registration.

This is a **registry-free** in-process tracking frontend.  It deliberately does
not subclass or register with the retired ``SLAMBackend`` registry (#1073/#1075):
the class is a plain object owning one ICP semantic contract, so the canonical
``TrackingProvider`` binding can be added without touching this algorithm.  See
``docs/architecture/2026-08-23-open3d-icp-trackingprovider-port.md``.

This frontend is intentionally separate from the RGB-D odometry frontend: RGB-D
odometry and point-cloud ICP are different frontend algorithms and must remain
separate rows in the EliGSiR frontend-independence benchmark.

``engine="auto"`` prefers Open3D's Tensor registration pipeline when a GPU
accelerator is available.  With the pinned upstream HIP port (Open3D #7509),
that Tensor device is physically ROCm/HIP even though Open3D intentionally
preserves the logical device spelling ``CUDA:0`` for compatibility.  The two
engines are an implementation-acceleration detail of ONE frontend, never two
benchmark methods — ``artifacts/open3d_icp_engine_equivalence_gfx1151.json``
records the real-hardware evidence that they agree.

Open3D ``registration_icp(source=current, target=previous, ...)`` returns a
transform from the current camera coordinate frame into the previous camera
coordinate frame.  The canonical EliGSiR pose is ``T_world_camera``, so
composition is ``T_world_current = T_world_previous @ T_previous_current``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame, TrackingResult
from eligsir.tracking.open3d_runtime import (
    import_open3d,
    is_accelerator_device,
    open3d_build_identity,
    physical_accelerator,
    tensor_device,
)


def _compose_current_c2w(previous_c2w: np.ndarray, current_to_previous: np.ndarray) -> np.ndarray:
    """Compose Open3D's current->previous ICP transform into canonical c2w."""
    return (previous_c2w @ current_to_previous).astype(np.float32)


def _prediction_to_icp_init(previous_c2w: np.ndarray, predicted_c2w: np.ndarray) -> np.ndarray:
    """Convert an absolute predicted c2w pose into current->previous ICP init."""
    return (np.linalg.inv(previous_c2w) @ predicted_c2w).astype(np.float64)


def _depth_for_pointcloud(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Return metric depth with invalid/out-of-range pixels zeroed for Open3D."""
    clean = np.asarray(depth, dtype=np.float32).copy()
    invalid = ~np.isfinite(clean) | (clean < depth_min) | (clean > depth_max)
    clean[invalid] = 0.0
    return clean


def _depth_to_xyz(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min: float,
    depth_max: float,
) -> np.ndarray:
    """Back-project valid metric depth pixels to OpenCV camera-frame XYZ."""
    clean = _depth_for_pointcloud(depth, depth_min, depth_max)
    rows, cols = np.nonzero(clean > 0.0)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = clean[rows, cols]
    x = (cols.astype(np.float32) - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (rows.astype(np.float32) - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


class Open3DICPTracker:
    """Frame-to-frame point-cloud ICP over metric RGB-D depth.

    A registry-free tracking frontend: no base class, no ``register_backend``
    decorator, no import of ``eligsir.contracts.slam_backend``.  It exposes
    ``initialize(frame)`` / ``track(frame, T_pred=None)`` returning the canonical
    ``TrackingResult``, which is the semantic contract #1075 retains.

    Args:
        estimator: ``"point_to_plane"`` (default) or ``"point_to_point"``.
        engine: ``"auto"`` (prefer Tensor GPU), ``"tensor"`` (force the
            Open3D Tensor pipeline), or ``"legacy"`` (force legacy CPU ICP).
        device: Open3D Tensor device string. ``"auto"`` selects the available
            accelerator, otherwise CPU.  Under upstream Open3D #7509 a ROCm
            accelerator is logically named ``CUDA:0`` by Open3D.
        depth_min: Minimum accepted metric depth.
        depth_max: Maximum accepted metric depth.
        voxel_size: Optional voxel downsampling size in metres; ``0`` disables it.
        normal_radius: Radius in metres used to estimate normals for point-to-plane ICP.
        normal_max_nn: Maximum neighbours used for normal estimation.
        max_correspondence_distance: Maximum ICP correspondence distance in metres.
        max_iterations: Maximum ICP optimizer iterations.
        min_fitness: Minimum Open3D fitness required to accept a pose.
        max_inlier_rmse: Maximum Open3D inlier RMSE required to accept a pose.
        min_points: Minimum source and target point count after filtering/downsampling.
    """

    #: Stable frontend identity, recorded as ``TrackingResult.method`` and as
    #: ``TrackingObservation.provider``.  Preserved verbatim from the original
    #: implementation so existing hardware evidence stays addressable.
    name = "open3d_icp_live"

    def __init__(
        self,
        estimator: str = "point_to_plane",
        engine: str = "auto",
        device: str = "auto",
        depth_min: float = 0.1,
        depth_max: float = 8.0,
        voxel_size: float = 0.02,
        normal_radius: float = 0.05,
        normal_max_nn: int = 30,
        max_correspondence_distance: float = 0.07,
        max_iterations: int = 30,
        min_fitness: float = 0.3,
        max_inlier_rmse: float = 0.07,
        min_points: int = 100,
    ) -> None:
        if estimator not in {"point_to_plane", "point_to_point"}:
            raise ValueError("estimator must be 'point_to_plane' or 'point_to_point'")
        if engine not in {"auto", "tensor", "legacy"}:
            raise ValueError("engine must be 'auto', 'tensor', or 'legacy'")
        if not device:
            raise ValueError("device must be 'auto' or an Open3D device string")
        if engine == "legacy" and device != "auto" and not device.upper().startswith("CPU:"):
            raise ValueError("legacy Open3D ICP is CPU-only; use engine='tensor' for GPU")
        if depth_min < 0 or depth_max <= depth_min:
            raise ValueError("depth range must satisfy 0 <= depth_min < depth_max")
        if voxel_size < 0 or normal_radius <= 0 or normal_max_nn < 1:
            raise ValueError("invalid Open3D ICP geometry parameters")
        if max_correspondence_distance <= 0 or max_iterations < 1:
            raise ValueError("invalid Open3D ICP optimization parameters")
        if not 0.0 <= min_fitness <= 1.0 or max_inlier_rmse <= 0 or min_points < 3:
            raise ValueError("invalid Open3D ICP acceptance parameters")

        self._estimator = estimator
        self._engine_requested = engine
        self._device_requested = device
        self._depth_min = float(depth_min)
        self._depth_max = float(depth_max)
        self._voxel_size = float(voxel_size)
        self._normal_radius = float(normal_radius)
        self._normal_max_nn = int(normal_max_nn)
        self._max_correspondence_distance = float(max_correspondence_distance)
        self._max_iterations = int(max_iterations)
        self._min_fitness = float(min_fitness)
        self._max_inlier_rmse = float(max_inlier_rmse)
        self._min_points = int(min_points)

        self._prev_frame: Optional[RGBDFrame] = None
        self._prev_c2w: Optional[np.ndarray] = None
        self._runtime_engine = "unresolved"
        self._logical_device = "unresolved"
        self._physical_accelerator = "unresolved"

    def initialize(self, first_frame: RGBDFrame) -> TrackingResult:
        """Anchor the first frame in its supplied pose or identity."""
        pose = first_frame.pose if first_frame.pose is not None else Pose.identity()
        self._prev_frame = first_frame
        self._prev_c2w = pose.matrix.copy()
        return TrackingResult(
            pose=pose,
            confidence=1.0,
            valid=True,
            method=self.name,
            raw_stats=self._provenance_stats(),
        )

    def track(self, frame: RGBDFrame, T_pred: Optional[Pose] = None) -> TrackingResult:
        """Estimate the current camera pose by registering depth point clouds.

        Point-cloud ICP never reads the Gaussian map, so this frontend takes no
        render context — the old ``gaussian_ctx`` parameter was registry
        vocabulary the algorithm always discarded.
        """
        t0 = time.perf_counter()
        if self._prev_frame is None or self._prev_c2w is None:
            return self.initialize(frame)

        previous_c2w = self._prev_c2w.copy()
        try:
            o3d = import_open3d()
            registration = self._run_icp(
                o3d,
                self._prev_frame,
                frame,
                frame.intrinsics,
                previous_c2w,
                T_pred,
            )
        except Exception as exc:
            registration = None
            error = repr(exc)
        else:
            error = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if registration is None:
            pose = Pose(previous_c2w)
            return TrackingResult(
                pose=pose,
                confidence=0.0,
                valid=False,
                method=self.name,
                tracking_time_ms=elapsed_ms,
                raw_stats={**self._provenance_stats(), "error": error or "registration_failed"},
            )

        current_to_previous, fitness, inlier_rmse, source_points, target_points = registration
        c2w = _compose_current_c2w(previous_c2w, current_to_previous)
        valid = fitness >= self._min_fitness and inlier_rmse <= self._max_inlier_rmse
        confidence = float(np.clip(fitness, 0.0, 1.0)) if valid else 0.0
        delta_m = float(np.linalg.norm(c2w[:3, 3] - previous_c2w[:3, 3]))
        pose = Pose(c2w)

        if valid:
            self._prev_frame = frame
            self._prev_c2w = c2w

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
                **self._provenance_stats(),
                "fitness": fitness,
                "inlier_rmse": inlier_rmse,
                "source_points": source_points,
                "target_points": target_points,
            },
        )

    def _resolve_runtime(self, o3d):
        device = tensor_device(o3d, self._device_requested)
        if self._engine_requested == "legacy":
            engine = "legacy"
            logical_device = "CPU:0"
            physical = "cpu"
        elif self._engine_requested == "tensor":
            engine = "tensor"
            logical_device = str(device)
            physical = physical_accelerator(device)
        elif is_accelerator_device(device):
            engine = "tensor"
            logical_device = str(device)
            physical = physical_accelerator(device)
        else:
            engine = "legacy"
            logical_device = "CPU:0"
            physical = "cpu"

        self._runtime_engine = engine
        self._logical_device = logical_device
        self._physical_accelerator = physical
        return engine, device

    def _run_icp(
        self,
        o3d,
        previous: RGBDFrame,
        current: RGBDFrame,
        intrinsics: CameraIntrinsics,
        previous_c2w: np.ndarray,
        prediction: Optional[Pose],
    ):
        """Run the selected Open3D ICP engine current->previous."""
        engine, device = self._resolve_runtime(o3d)
        if engine == "tensor":
            return self._run_tensor_icp(
                o3d,
                previous,
                current,
                intrinsics,
                previous_c2w,
                prediction,
                device,
            )
        return self._run_legacy_icp(
            o3d,
            previous,
            current,
            intrinsics,
            previous_c2w,
            prediction,
        )

    def _run_tensor_icp(
        self,
        o3d,
        previous: RGBDFrame,
        current: RGBDFrame,
        intrinsics: CameraIntrinsics,
        previous_c2w: np.ndarray,
        prediction: Optional[Pose],
        device,
    ):
        """Run Open3D Tensor ICP, including the ROCm/HIP path from upstream #7509."""
        source = self._tensor_pointcloud(o3d, current.depth, intrinsics, device)
        target = self._tensor_pointcloud(o3d, previous.depth, intrinsics, device)
        source_points = int(source.point.positions.shape[0])
        target_points = int(target.point.positions.shape[0])
        if source_points < self._min_points or target_points < self._min_points:
            return None

        if self._estimator == "point_to_plane":
            source.estimate_normals(max_nn=self._normal_max_nn, radius=self._normal_radius)
            target.estimate_normals(max_nn=self._normal_max_nn, radius=self._normal_radius)
            estimator = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimator = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        init = np.eye(4, dtype=np.float64)
        if prediction is not None:
            init = _prediction_to_icp_init(previous_c2w, prediction.matrix)
        init_tensor = o3d.core.Tensor(
            init,
            dtype=o3d.core.float64,
            device=device,
        )
        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self._max_iterations
        )
        result = o3d.t.pipelines.registration.icp(
            source,
            target,
            self._max_correspondence_distance,
            init_tensor,
            estimator,
            criteria,
        )
        transformation = np.asarray(result.transformation.cpu().numpy(), dtype=np.float32)
        if transformation.shape != (4, 4) or not np.all(np.isfinite(transformation)):
            return None
        fitness = float(result.fitness)
        inlier_rmse = float(result.inlier_rmse)
        if not np.isfinite(fitness) or not np.isfinite(inlier_rmse):
            return None
        return transformation, fitness, inlier_rmse, source_points, target_points

    def _run_legacy_icp(
        self,
        o3d,
        previous: RGBDFrame,
        current: RGBDFrame,
        intrinsics: CameraIntrinsics,
        previous_c2w: np.ndarray,
        prediction: Optional[Pose],
    ):
        """Run the legacy CPU point-cloud registration implementation."""
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(intrinsics.width),
            int(intrinsics.height),
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        source = self._legacy_pointcloud(o3d, current.depth, intrinsic)
        target = self._legacy_pointcloud(o3d, previous.depth, intrinsic)
        source_points = len(source.points)
        target_points = len(target.points)
        if source_points < self._min_points or target_points < self._min_points:
            return None

        if self._estimator == "point_to_plane":
            search = o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._normal_radius,
                max_nn=self._normal_max_nn,
            )
            source.estimate_normals(search_param=search)
            target.estimate_normals(search_param=search)
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        init = np.eye(4, dtype=np.float64)
        if prediction is not None:
            init = _prediction_to_icp_init(previous_c2w, prediction.matrix)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self._max_iterations
        )
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self._max_correspondence_distance,
            init,
            estimator,
            criteria,
        )
        transformation = np.asarray(result.transformation, dtype=np.float32)
        if transformation.shape != (4, 4) or not np.all(np.isfinite(transformation)):
            return None
        fitness = float(result.fitness)
        inlier_rmse = float(result.inlier_rmse)
        if not np.isfinite(fitness) or not np.isfinite(inlier_rmse):
            return None
        return transformation, fitness, inlier_rmse, source_points, target_points

    def _legacy_pointcloud(self, o3d, depth: np.ndarray, intrinsic):
        """Back-project metric depth into a legacy CPU Open3D point cloud."""
        clean = _depth_for_pointcloud(depth, self._depth_min, self._depth_max)
        depth_image = o3d.geometry.Image(clean)
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsic,
            depth_scale=1.0,
            depth_trunc=self._depth_max,
            stride=1,
        )
        if self._voxel_size > 0:
            cloud = cloud.voxel_down_sample(self._voxel_size)
        return cloud

    def _tensor_pointcloud(self, o3d, depth: np.ndarray, intrinsics: CameraIntrinsics, device):
        """Build a Tensor point cloud on *device* from aligned metric depth."""
        points = _depth_to_xyz(depth, intrinsics, self._depth_min, self._depth_max)
        cloud = o3d.t.geometry.PointCloud(device)
        cloud.point.positions = o3d.core.Tensor(
            points,
            dtype=o3d.core.float32,
            device=device,
        )
        if self._voxel_size > 0 and points.shape[0] > 0:
            cloud = cloud.voxel_down_sample(self._voxel_size)
        return cloud

    def _provenance_stats(self) -> dict:
        """Return explicit frontend parameters and accelerator provenance."""
        return {
            "estimator": self._estimator,
            "engine_requested": self._engine_requested,
            "engine": self._runtime_engine,
            "device_requested": self._device_requested,
            "logical_device": self._logical_device,
            "physical_accelerator": self._physical_accelerator,
            "depth_min": self._depth_min,
            "depth_max": self._depth_max,
            "voxel_size": self._voxel_size,
            "normal_radius": self._normal_radius,
            "normal_max_nn": self._normal_max_nn,
            "max_correspondence_distance": self._max_correspondence_distance,
            "max_iterations": self._max_iterations,
            "min_fitness": self._min_fitness,
            "max_inlier_rmse": self._max_inlier_rmse,
            "min_points": self._min_points,
            **open3d_build_identity(),
        }
