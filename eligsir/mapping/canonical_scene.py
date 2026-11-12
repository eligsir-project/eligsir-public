"""Minimal scene owner for the canonical MappingFrame runtime."""

from __future__ import annotations

import os
from collections import deque

import numpy as np

from eligsir.mapping.canonical_bootstrap import (
    canonical_bootstrap_point_cloud,
    canonical_camera_from_observation,
    prepared_device_seed_from_bindings,
)
from eligsir.runtime.config import (
    CanonicalBootstrapPolicy,
    CanonicalRuntimePolicy,
)


class CanonicalStreamingScene:
    """Residency-bounded canonical cameras with no source or decoder surface."""

    def __init__(
        self,
        policy: CanonicalRuntimePolicy,
        bootstrap_policy: CanonicalBootstrapPolicy,
        gaussians,
        initial_frames,
    ) -> None:
        if not isinstance(policy, CanonicalRuntimePolicy):
            raise TypeError("canonical scene requires CanonicalRuntimePolicy")
        if not isinstance(bootstrap_policy, CanonicalBootstrapPolicy):
            raise TypeError("canonical scene requires CanonicalBootstrapPolicy")
        frames = tuple(initial_frames)
        self.residency_policy = policy
        self.bootstrap_policy = bootstrap_policy
        self.gaussians = gaussians
        self.model_path = policy.model_path
        self.train_cameras: list[object] = []
        self._admitted_cameras_by_source: dict[object, object] = {}
        self._test_cameras: list[object] = []
        self._causal_eval_metadata: dict | None = None
        self._replay_buffer = deque(maxlen=policy.keyframe_window)
        self._global_reservoir: list[object] = []
        self.current_camera = None
        self._payload_residency_dirty = True
        self.cameras_extent = 1.0
        self._camera_centers: list[np.ndarray] = []
        self._bootstrap_point_radius = 0.0
        self._initial_frames = frames
        # Tier-D's independent sensor reference. Keep a bounded, CPU-owned
        # point cloud because canonical scenes do not use StreamingScene's
        # input-quality voxel store.
        self._accumulated_sensor_points: list[np.ndarray] = []
        self._sensor_point_cap = 120000
        self._device_transfer_owner = None
        self._observation_db = None
        self.occupancy_voxel_size = bootstrap_policy.occupancy_voxel_size_m
        self._occupancy_keys_gpu = None

    def bind_causal_evaluation(self, test_cameras, eval_metadata: dict) -> None:
        """Attach evaluation-only cameras without admitting them to the map."""

        cameras = list(test_cameras)
        if any(camera in self.train_cameras for camera in cameras):
            raise ValueError("causal test cameras must be disjoint from training")
        self._test_cameras = cameras
        self._causal_eval_metadata = dict(eval_metadata)

    def eval_metadata(self) -> dict:
        """Return a copy of the causal evaluation metadata, or an empty dict when unset."""
        if self._causal_eval_metadata is None:
            return {}
        return dict(self._causal_eval_metadata)

    def bind_device_transfer_owner(self, owner) -> None:
        """Bind the single device-transfer owner used to materialise cameras; rebinding is an error."""
        if owner is None or not callable(getattr(owner, "prefetch_cameras", None)):
            raise TypeError("device transfer owner must provide prefetch_cameras")
        if self._device_transfer_owner is not None and self._device_transfer_owner is not owner:
            raise RuntimeError("canonical scene already has a transfer owner")
        self._device_transfer_owner = owner

    def set_observation_db(self, observation_db) -> None:
        """Attach the observation database consulted by admission and evidence lookups."""
        self._observation_db = observation_db

    def _camera(self, observation):
        owner = self._device_transfer_owner
        if owner is None:
            raise RuntimeError("canonical scene transfer owner must bind before cameras")
        return canonical_camera_from_observation(
            observation, data_device=owner.device
        )

    def _prepare(self, camera) -> None:
        owner = self._device_transfer_owner
        if owner is None:
            raise RuntimeError("canonical scene has no transfer owner")
        owner.prefetch_cameras((camera,))

    def _register_admitted_camera(self, observation, camera) -> None:
        from eligsir.runtime.payload_residency import mark_payload_residency_dirty

        source = observation.mapping_frame.source
        existing = self._admitted_cameras_by_source.get(source)
        if existing is not None and existing is not camera:
            raise RuntimeError("source identity already owns an admitted camera")
        self._admitted_cameras_by_source[source] = camera
        mark_payload_residency_dirty(self)

    def admitted_camera_for_source(self, source):
        """Return the permanent camera for one admitted source in O(1)."""

        return self._admitted_cameras_by_source.get(source)

    @staticmethod
    def _camera_center(observation) -> np.ndarray:
        mapping = getattr(observation, "mapping_frame", None)
        if mapping is None or mapping.pose is None:
            raise ValueError("canonical scene extent requires a MappingFrame pose")
        c2w = np.asarray(mapping.pose.c2w, dtype=np.float32)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError("canonical scene extent requires a finite 4x4 pose")
        return np.array(c2w[:3, 3], dtype=np.float32, copy=True)

    def _refresh_camera_extent(self) -> None:
        if not self._camera_centers:
            self.cameras_extent = max(1.0, self._bootstrap_point_radius)
            return
        centers = np.stack(self._camera_centers, axis=0)
        centroid = centers.mean(axis=0)
        camera_radius = float(
            np.linalg.norm(centers - centroid, axis=1).max(initial=0.0)
        )
        self.cameras_extent = max(
            1.0,
            camera_radius * 1.1,
            self._bootstrap_point_radius,
        )

    def initialize_from_canonical_frames(self, record_fn=None) -> None:
        """Seed the map from the bootstrap frames and register them as admitted cameras."""
        frames = self._initial_frames
        pcd = canonical_bootstrap_point_cloud(frames, self.bootstrap_policy)
        for observation in frames:
            self._accumulate_sensor_points(observation)
            camera = self._camera(observation)
            camera._is_bootstrap = True
            self._register_admitted_camera(observation, camera)
            self.train_cameras.append(camera)
            self._replay_buffer.append(camera)
            self._global_reservoir.append(camera)
            self._prepare(camera)
            if record_fn is not None:
                record_fn(camera, observation, is_train=True, is_holdout=False)
            self._camera_centers.append(self._camera_center(observation))
        self.current_camera = self.train_cameras[-1]
        points = np.asarray(pcd.points, dtype=np.float32)
        if points.shape[0]:
            self._bootstrap_point_radius = float(
                np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.5
            )
        self._refresh_camera_extent()
        policy = self.bootstrap_policy
        self.gaussians.create_from_pcd(
            pcd,
            self.cameras_extent,
            init_scale_mode=policy.init_scale_mode,
            init_scale=policy.init_scale,
            voxel_size=policy.model_voxel_size_m,
        )
        self.maintain_occupancy_hash(policy.occupancy_voxel_size_m)
        self._export_seed(pcd)

    def initialize_from_ready_bindings(self, bindings, record_fn=None) -> None:
        """Initialize once from attested, native device cameras.

        Prepared cameras do not carry a MappingFrame or a cold reload route.
        This path is deliberately explicit and rejects any binding that would
        fall back to the legacy CPU bootstrap semantics.
        """

        import torch

        from eligsir.runtime.ready_camera_binder import ReadyCameraBinding
        from eligsir.runtime.payload_residency import mark_payload_residency_dirty

        bindings = tuple(bindings)
        if self.train_cameras or self._admitted_cameras_by_source:
            raise RuntimeError("canonical scene is already initialized")
        if not bindings or not all(isinstance(binding, ReadyCameraBinding) for binding in bindings):
            raise TypeError("prepared bootstrap requires ReadyCameraBinding values")
        if any(binding.prepared_tier != "native" or binding.metadata.tier != "native" for binding in bindings):
            raise ValueError("prepared bootstrap requires the native ready tier")
        if any(binding.source_identity is not binding.metadata.source_identity for binding in bindings):
            raise ValueError("prepared bootstrap source metadata must retain authoritative identity")
        initializer = getattr(self.gaussians, "initialize_from_prepared_device_seed", None)
        if not callable(initializer):
            raise RuntimeError("Gaussian model does not support prepared device bootstrap")
        seed = prepared_device_seed_from_bindings(bindings, self.bootstrap_policy)
        centers = []
        for binding in bindings:
            pose = np.asarray(binding.metadata.pose_c2w, dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("prepared bootstrap requires a finite authoritative pose")
            centers.append(np.array(pose[:3, 3], dtype=np.float32, copy=True))
        points = seed.points
        radius = float(torch.linalg.vector_norm(points.max(dim=0).values - points.min(dim=0).values).item()) * 0.5
        centroid = np.stack(centers, axis=0).mean(axis=0)
        camera_radius = float(np.linalg.norm(np.stack(centers, axis=0) - centroid, axis=1).max(initial=0.0))
        extent = max(1.0, camera_radius * 1.1, radius)
        initializer(
            seed,
            extent,
            init_scale_mode=self.bootstrap_policy.init_scale_mode,
            init_scale=self.bootstrap_policy.init_scale,
            voxel_size=self.bootstrap_policy.model_voxel_size_m,
        )
        self._camera_centers.extend(centers)
        self._bootstrap_point_radius = radius
        self.cameras_extent = extent
        for binding in bindings:
            camera = binding.camera
            source = binding.source_identity
            camera._is_bootstrap = True
            camera._source_identity = source
            camera._prepared_control_token = binding.control_token
            self._admitted_cameras_by_source[source] = camera
            self.train_cameras.append(camera)
            self._replay_buffer.append(camera)
            self._global_reservoir.append(camera)
            if record_fn is not None:
                record_fn(camera, binding, is_train=True, is_holdout=False)
        self.current_camera = self.train_cameras[-1]
        mark_payload_residency_dirty(self)
        self.maintain_occupancy_hash(self.bootstrap_policy.occupancy_voxel_size_m)

    def _export_seed(self, pcd) -> None:
        from scene.readers.common import storePly

        seed_dir = os.path.join(self.model_path, "point_cloud", "initial_seed")
        os.makedirs(seed_dir, exist_ok=True)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        storePly(os.path.join(seed_dir, "point_cloud.ply"), points, colors, normals=normals)
        storePly(os.path.join(self.model_path, "input_seed.ply"), points, colors, normals=normals)

    def ingest_canonical_frame(self, observation, record_fn=None):
        """Admit one canonical observation: accumulate sensor points and register its camera for replay."""
        self._accumulate_sensor_points(observation)
        camera = self._camera(observation)
        self._register_admitted_camera(observation, camera)
        self.train_cameras.append(camera)
        self._replay_buffer.append(camera)
        self._global_reservoir.append(camera)
        self.current_camera = camera
        self._prepare(camera)
        self._camera_centers.append(self._camera_center(observation))
        self._refresh_camera_extent()
        if record_fn is not None:
            record_fn(camera, observation, is_train=True, is_holdout=False)
        return camera, observation, True

    def _accumulate_sensor_points(self, observation) -> None:
        """Accumulate independent world-space metric-depth points for Tier D."""

        from eligsir.contracts.streaming_bridge import mapping_frame_source_arrays

        depth, _ = mapping_frame_source_arrays(observation)
        intrinsics = observation.mapping_frame.intrinsics
        stride = max(1, int(self.bootstrap_policy.depth_stride))
        sampled = np.asarray(depth[::stride, ::stride], dtype=np.float32)
        rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
        cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)
        ys, xs = np.meshgrid(rows, cols, indexing="ij")
        valid = (
            np.isfinite(sampled)
            & (sampled > float(self.bootstrap_policy.min_depth_m))
            & (sampled < float(self.bootstrap_policy.max_depth_m))
        )
        if not valid.any():
            return
        z = sampled[valid]
        camera_points = np.stack(
            ((xs[valid] - float(intrinsics.cx)) * z / float(intrinsics.fx),
             (ys[valid] - float(intrinsics.cy)) * z / float(intrinsics.fy), z),
            axis=1,
        )
        c2w = np.asarray(observation.mapping_frame.pose.c2w, dtype=np.float32)
        world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        self._accumulated_sensor_points.append(world_points.astype(np.float32, copy=False))
        total = sum(points.shape[0] for points in self._accumulated_sensor_points)
        if total > self._sensor_point_cap:
            merged = np.concatenate(self._accumulated_sensor_points, axis=0)
            indices = np.linspace(0, merged.shape[0] - 1, self._sensor_point_cap, dtype=np.int64)
            self._accumulated_sensor_points = [merged[indices]]

    def get_accumulated_sensor_points(self) -> np.ndarray:
        """Return the bounded, Gaussian-independent sensor point cloud for Tier D."""

        if not self._accumulated_sensor_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(self._accumulated_sensor_points, axis=0)

    def promote_historical_frame(self, observation):
        """Materialize one selected COLD observation without live insertion.

        Historical selection is already fenced by the runtime work plan and
        the observation DB's durable DEFER eligibility. Promotion only creates
        the camera needed for supervision; it does not become the newest live
        camera and cannot schedule a depth-insertion job.
        """

        source = observation.mapping_frame.source
        if self.admitted_camera_for_source(source) is not None:
            raise RuntimeError("historical source already has a canonical camera")
        camera = self._camera(observation)
        camera._source_identity = source
        # Historical cameras are transaction-local. Durable identity/pose and
        # bounded HOT tensors live in FrameObservationDB; appending here would
        # grow scene camera objects and extent scans with total history.
        self._prepare(camera)
        from eligsir.runtime.payload_residency import mark_payload_residency_dirty

        mark_payload_residency_dirty(self)
        return camera

    def promote_resident_historical_frame(self, resident):
        """Create transaction-local camera geometry around DB-owned HOT pixels.

        The 1x1 host placeholder satisfies ``Camera``'s construction contract
        without decoding or copying the resident frame.  The runtime replaces
        it immediately with the exact validated tier tensors; this method does
        not call the transfer owner because those tensors are already resident.
        """

        import torch
        from scene.cameras import Camera
        from eligsir.utils.graphics_utils import focal2fov
        from eligsir.utils.rgbd_frames import c2w_to_camera_rt

        owner = self._device_transfer_owner
        if owner is None:
            raise RuntimeError("canonical scene has no transfer owner")
        width, height, fx, fy, cx, cy = tuple(resident.intrinsics)
        c2w = np.asarray(resident.pose_c2w, dtype=np.float32)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError("resident historical pose must be finite 4x4 c2w")
        rotation, translation = c2w_to_camera_rt(c2w)
        camera = Camera(
            colmap_id=int(resident.frame_id),
            R=rotation,
            T=translation,
            FoVx=focal2fov(float(fx), int(width)),
            FoVy=focal2fov(float(fy), int(height)),
            image=torch.zeros((3, 1, 1), dtype=torch.float32),
            gt_alpha_mask=None,
            image_name=f"{int(resident.frame_id):06d}",
            uid=int(resident.frame_id),
            data_device=owner.device,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )
        camera._mapping_frame = None
        camera._streaming_frame = None
        camera._streaming_depth_path = None
        camera._streaming_depth_scale = 1.0
        camera._native_image_loader = None
        return camera

    def maintain_occupancy_hash(self, voxel_size: float) -> None:
        """Rebuild the GPU voxel-occupancy hash of the current Gaussian positions at ``voxel_size``."""
        import torch
        from eligsir.utils.torch_geometry import pack_voxel_keys_torch

        self.occupancy_voxel_size = float(voxel_size)
        xyz = self.gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            self._occupancy_keys_gpu = torch.empty(
                0, dtype=torch.int64, device=xyz.device
            )
            return
        self._occupancy_keys_gpu = torch.unique(
            pack_voxel_keys_torch(xyz, self.occupancy_voxel_size), sorted=True
        )

    def add_to_occupancy_hash(self, xyz) -> None:
        """Insert the voxels covering ``xyz`` into the occupancy hash."""
        import torch
        from eligsir.utils.torch_geometry import pack_voxel_keys_torch

        if xyz.shape[0] == 0:
            return
        new_keys = torch.unique(
            pack_voxel_keys_torch(xyz.detach(), self.occupancy_voxel_size),
            sorted=True,
        )
        existing = self._occupancy_keys_gpu
        self._occupancy_keys_gpu = (
            new_keys
            if existing is None or existing.numel() == 0
            else torch.unique(torch.cat((existing.to(xyz.device), new_keys)), sorted=True)
        )

    def check_occupancy_torch(self, xyz, voxel_size: float, check_neighbors=True):
        """Return a bool mask of which ``xyz`` rows fall in occupied voxels (optionally including neighbours)."""
        import torch
        from eligsir.utils.torch_geometry import pack_voxel_keys_torch

        query_voxel_size = float(voxel_size)
        if abs(query_voxel_size - self.occupancy_voxel_size) < 1e-9:
            keys = self._occupancy_keys_gpu
        else:
            # Replay-tail refinement intentionally uses a finer candidate and
            # occupancy voxel than coarse live admission.  The maintained
            # index remains the canonical live fast path; a mismatched tail
            # query builds an exact device-local view of current topology.
            # This runs only at the bounded replay-depth cadence and performs
            # no host transfer or synchronization.
            existing_xyz = self.gaussians.get_xyz.detach()
            keys = torch.unique(
                pack_voxel_keys_torch(existing_xyz, query_voxel_size),
                sorted=True,
            )
        if keys is None or keys.numel() == 0 or xyz.shape[0] == 0:
            return torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
        sorted_keys = keys.to(xyz.device)

        def contains(query):
            """Return whether each packed voxel key is present in the sorted occupancy keys."""
            positions = torch.searchsorted(sorted_keys, query)
            clamped = positions.clamp(max=sorted_keys.shape[0] - 1)
            return sorted_keys[clamped] == query

        occupied = contains(pack_voxel_keys_torch(xyz, query_voxel_size))
        if not check_neighbors:
            return occupied
        offset = 1000.0
        coordinates = torch.floor(
            (xyz + offset) / max(query_voxel_size, 1e-6)
        ).to(torch.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    neighbour = (
                        ((coordinates[:, 0] + dx) << 42)
                        | ((coordinates[:, 1] + dy) << 21)
                        | (coordinates[:, 2] + dz)
                    )
                    occupied |= contains(neighbour)
        return occupied

    def hot_camera_ids(self) -> set[int]:
        """Return the object ids of cameras that must stay resident (replay buffer plus current)."""
        hot = {id(camera) for camera in self._replay_buffer}
        if self.current_camera is not None:
            hot.add(id(self.current_camera))
        return hot

    def retire_live_admitted_payloads(self) -> int:
        """Retire admitted payloads that are no longer hot; returns the number retired."""
        from eligsir.runtime.payload_residency import retire_live_admitted_payloads

        return retire_live_admitted_payloads(self)

    def close_decode_cache(self) -> None:
        """No-op: the canonical scene keeps no decode cache to close."""
        return None


__all__ = ["CanonicalStreamingScene"]
