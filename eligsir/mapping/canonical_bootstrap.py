"""Fail-closed camera and point-cloud construction for canonical mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eligsir.contracts.streaming_bridge import mapping_frame_source_arrays
from eligsir.runtime.config import CanonicalBootstrapPolicy


@dataclass(frozen=True, slots=True)
class PreparedDeviceSeed:
    """A device-owned bootstrap population with no CPU point-cloud boundary."""

    points: object
    colors: object


def prepared_device_seed_from_bindings(bindings, policy: CanonicalBootstrapPolicy):
    """Backproject native ready cameras entirely on their existing device.

    This is intentionally a separate opt-in path: the canonical MappingFrame
    bootstrap remains CPU-owned and unchanged.  Temporal seed support has no
    device contract yet, so a policy that requires it is rejected rather than
    silently weakening the established bootstrap semantics.
    """

    import torch

    if not isinstance(policy, CanonicalBootstrapPolicy):
        raise TypeError("prepared device bootstrap requires CanonicalBootstrapPolicy")
    if policy.require_temporal_for_seed:
        raise ValueError("prepared device bootstrap does not support required temporal seed validation")
    bindings = tuple(bindings)
    if not bindings:
        raise RuntimeError("prepared device bootstrap requires at least one ready camera")
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = None
    stride = int(policy.depth_stride)
    for binding in bindings:
        camera = getattr(binding, "camera", None)
        rgb = getattr(camera, "original_image", None)
        depth = getattr(camera, "_sensor_depth_cache", None)
        c2w = getattr(camera, "c2w", None)
        gsplat = getattr(camera, "gsplat", None)
        K = getattr(gsplat, "K", None)
        tensors = (rgb, depth, c2w, K)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("prepared bootstrap requires device RGB-D, pose, and intrinsics tensors")
        if rgb.ndim != 3 or rgb.shape[0] != 3 or depth.ndim != 2:
            raise ValueError("prepared bootstrap requires CHW RGB and HW depth tensors")
        if tuple(rgb.shape[1:]) != tuple(depth.shape) or c2w.shape != (4, 4) or K.shape != (3, 3):
            raise ValueError("prepared bootstrap camera tensor geometry is invalid")
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("prepared bootstrap tensors must be float32")
        if any(value.device != rgb.device for value in tensors):
            raise ValueError("prepared bootstrap tensors must share one device")
        if device is None:
            device = rgb.device
        elif rgb.device != device:
            raise ValueError("prepared bootstrap bindings must share one device")
        sampled_depth = depth[::stride, ::stride]
        valid = torch.isfinite(sampled_depth) & (sampled_depth > policy.min_depth_m) & (sampled_depth < policy.max_depth_m)
        if not bool(valid.any()):
            continue
        rows = torch.arange(0, depth.shape[0], stride, device=device, dtype=torch.float32)
        cols = torch.arange(0, depth.shape[1], stride, device=device, dtype=torch.float32)
        ys, xs = torch.meshgrid(rows, cols, indexing="ij")
        z = sampled_depth[valid]
        x = (xs[valid] - K[0, 2]) * z / K[0, 0]
        y = (ys[valid] - K[1, 2]) * z / K[1, 1]
        camera_points = torch.stack((x, y, z), dim=1)
        world_points = camera_points @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]
        colors = rgb.permute(1, 2, 0)[::stride, ::stride][valid]
        batches.append((world_points, colors))
    if not batches:
        raise RuntimeError("prepared device bootstrap has no valid metric depth points")
    points = torch.cat(tuple(batch[0] for batch in batches), dim=0)
    colors = torch.cat(tuple(batch[1] for batch in batches), dim=0)
    if policy.seed_voxel_size_m > 0.0:
        cells = torch.floor(points / policy.seed_voxel_size_m).to(torch.int64)
        _, inverse, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
        reduced_points = torch.zeros((counts.shape[0], 3), dtype=torch.float32, device=device)
        reduced_colors = torch.zeros_like(reduced_points)
        reduced_points.index_add_(0, inverse, points)
        reduced_colors.index_add_(0, inverse, colors)
        points = reduced_points / counts.to(torch.float32).unsqueeze(1)
        colors = reduced_colors / counts.to(torch.float32).unsqueeze(1)
    if points.shape[0] > policy.max_points:
        indices = torch.linspace(0, points.shape[0] - 1, policy.max_points, device=device).to(torch.int64)
        points, colors = points[indices], colors[indices]
    return PreparedDeviceSeed(points=points, colors=colors)


def _voxel_centroids(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors
    cells = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    clustered_points = np.stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)], axis=1
    ) / counts[:, None]
    clustered_colors = np.stack(
        [np.bincount(inverse, weights=colors[:, axis]) for axis in range(3)], axis=1
    ) / counts[:, None]
    return clustered_points.astype(np.float32), clustered_colors.astype(np.float32)


def _temporal_depth_support(
    *,
    sampled_depth: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mapping,
    peer_mapping,
    peer_depth: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    """Reproject sampled metric depth into one adjacent bootstrap frame."""

    intrinsics = mapping.intrinsics
    peer_intrinsics = peer_mapping.intrinsics
    z = np.asarray(sampled_depth, dtype=np.float32)
    x_camera = (
        (np.asarray(xs, dtype=np.float32) - float(intrinsics.depth_cx))
        * z
        / float(intrinsics.depth_fx)
    )
    y_camera = (
        (np.asarray(ys, dtype=np.float32) - float(intrinsics.depth_cy))
        * z
        / float(intrinsics.depth_fy)
    )
    camera_points = np.stack((x_camera, y_camera, z), axis=1)
    c2w = np.asarray(mapping.pose.c2w, dtype=np.float32)
    peer_c2w = np.asarray(peer_mapping.pose.c2w, dtype=np.float32)
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    try:
        peer_w2c = np.linalg.inv(peer_c2w)
    except np.linalg.LinAlgError:
        return np.zeros(z.shape, dtype=bool)
    peer_points = world_points @ peer_w2c[:3, :3].T + peer_w2c[:3, 3]
    peer_z = peer_points[:, 2]

    peer_h, peer_w = peer_depth.shape
    declared_w = int(peer_intrinsics.depth_width)
    declared_h = int(peer_intrinsics.depth_height)
    scale_x = float(peer_w) / float(declared_w)
    scale_y = float(peer_h) / float(declared_h)
    peer_fx = float(peer_intrinsics.depth_fx) * scale_x
    peer_fy = float(peer_intrinsics.depth_fy) * scale_y
    peer_cx = float(peer_intrinsics.depth_cx) * scale_x
    peer_cy = float(peer_intrinsics.depth_cy) * scale_y
    safe_z = np.maximum(peer_z, 1e-6)
    u = np.rint(peer_points[:, 0] / safe_z * peer_fx + peer_cx).astype(np.int64)
    v = np.rint(peer_points[:, 1] / safe_z * peer_fy + peer_cy).astype(np.int64)
    in_bounds = (
        (peer_z > 0.0)
        & (u >= 0)
        & (u < peer_w)
        & (v >= 0)
        & (v < peer_h)
    )
    supported = np.zeros(z.shape, dtype=bool)
    if in_bounds.any():
        measured = peer_depth[v[in_bounds], u[in_bounds]]
        supported[in_bounds] = (
            np.isfinite(measured)
            & (measured > 0.0)
            & (np.abs(peer_z[in_bounds] - measured) <= float(tolerance_m))
        )
    return supported


def canonical_bootstrap_point_cloud(frames, policy: CanonicalBootstrapPolicy):
    """Backproject already validated metric MappingFrames, or fail closed."""

    from eligsir.utils.graphics_utils import BasicPointCloud

    if not isinstance(policy, CanonicalBootstrapPolicy):
        raise TypeError("canonical bootstrap requires CanonicalBootstrapPolicy")
    observations = tuple(frames)
    if not observations:
        raise RuntimeError("canonical bootstrap requires at least one MappingFrame")

    decoded = [mapping_frame_source_arrays(observation) for observation in observations]
    point_batches: list[np.ndarray] = []
    color_batches: list[np.ndarray] = []
    stride = policy.depth_stride
    for observation_index, observation in enumerate(observations):
        depth, rgb = decoded[observation_index]
        mapping = observation.mapping_frame
        intrinsics = mapping.intrinsics
        sampled = depth[::stride, ::stride]
        rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
        cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)
        ys, xs = np.meshgrid(rows, cols, indexing="ij")
        valid = (
            np.isfinite(sampled)
            & (sampled > policy.min_depth_m)
            & (sampled < policy.max_depth_m)
        )
        if policy.require_temporal_for_seed:
            peer_index = (
                observation_index - 1
                if observation_index > 0
                else observation_index + 1
            )
            if peer_index >= len(observations):
                valid &= False
            elif valid.any():
                peer_depth, _ = decoded[peer_index]
                support = _temporal_depth_support(
                    sampled_depth=sampled[valid],
                    xs=xs[valid],
                    ys=ys[valid],
                    mapping=mapping,
                    peer_mapping=observations[peer_index].mapping_frame,
                    peer_depth=peer_depth,
                    tolerance_m=policy.temporal_tolerance_m,
                )
                temporal_valid = np.zeros_like(valid, dtype=bool)
                temporal_valid[valid] = support
                valid &= temporal_valid
        if not valid.any():
            continue
        z = sampled[valid]
        camera_points = np.stack(
            (
                (xs[valid] - float(intrinsics.depth_cx))
                * z
                / float(intrinsics.depth_fx),
                (ys[valid] - float(intrinsics.depth_cy))
                * z
                / float(intrinsics.depth_fy),
                z,
                np.ones_like(z),
            ),
            axis=1,
        )
        c2w = np.asarray(mapping.pose.c2w, dtype=np.float32)
        world_points = camera_points @ c2w.T
        point_batches.append(world_points[:, :3].astype(np.float32, copy=False))
        color_batches.append(rgb[::stride, ::stride][valid].astype(np.float32, copy=False))

    if not point_batches:
        if policy.require_temporal_for_seed:
            raise RuntimeError(
                "canonical bootstrap has no temporally supported metric depth points"
            )
        raise RuntimeError("canonical bootstrap has no valid metric depth points")
    points = np.concatenate(point_batches, axis=0)
    colors = np.concatenate(color_batches, axis=0)
    points, colors = _voxel_centroids(
        points, colors, policy.seed_voxel_size_m
    )
    if points.shape[0] > policy.max_points:
        # Deterministic population cap only; this never manufactures geometry.
        indices = np.linspace(
            0, points.shape[0] - 1, policy.max_points, dtype=np.int64
        )
        points, colors = points[indices], colors[indices]
    return BasicPointCloud(
        points=points,
        colors=colors,
        normals=np.zeros_like(points, dtype=np.float32),
    )


def canonical_camera_from_observation(observation, *, data_device):
    """Create one camera directly from canonical pixels, pose and calibration."""

    import torch
    from scene.cameras import Camera
    from eligsir.utils.graphics_utils import focal2fov
    from eligsir.utils.rgbd_frames import c2w_to_camera_rt

    depth, rgb = mapping_frame_source_arrays(observation)
    mapping = observation.mapping_frame
    intrinsics = mapping.intrinsics
    c2w = np.asarray(mapping.pose.c2w, dtype=np.float32)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("canonical bootstrap pose must be a finite 4x4 matrix")
    rotation, translation = c2w_to_camera_rt(c2w)
    image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    camera = Camera(
        colmap_id=int(mapping.source.source_ordinal),
        R=rotation,
        T=translation,
        FoVx=focal2fov(float(intrinsics.fx), int(intrinsics.width)),
        FoVy=focal2fov(float(intrinsics.fy), int(intrinsics.height)),
        image=image,
        gt_alpha_mask=None,
        image_name=f"{int(mapping.source.source_ordinal):06d}",
        uid=int(mapping.source.source_ordinal),
        data_device=data_device,
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.cx),
        cy=float(intrinsics.cy),
        width=int(intrinsics.width),
        height=int(intrinsics.height),
    )
    camera._mapping_frame = mapping
    camera._streaming_frame = observation
    camera._streaming_depth_path = None
    camera._streaming_depth_scale = 1.0
    camera._sensor_depth_cache = torch.from_numpy(np.ascontiguousarray(depth))
    camera._streaming_timestamp = float(mapping.source.timestamp_seconds)

    def native_image_loader():
        _, native_rgb = mapping_frame_source_arrays(observation)
        native = torch.from_numpy(
            np.ascontiguousarray(native_rgb.transpose(2, 0, 1))
        )
        return native, None

    camera._native_image_loader = native_image_loader
    return camera


def canonical_camera_extent(frames, pcd) -> float:
    """Return a finite metric radius from canonical poses and points."""

    positions = []
    for observation in tuple(frames):
        mapping_frame_source_arrays(observation)
        positions.append(
            np.asarray(observation.mapping_frame.pose.c2w, dtype=np.float32)[:3, 3]
        )
    centers = np.stack(positions, axis=0)
    if not np.isfinite(centers).all():
        raise ValueError("canonical bootstrap camera centers must be finite")
    centroid = centers.mean(axis=0)
    camera_radius = float(np.linalg.norm(centers - centroid, axis=1).max(initial=0.0))
    points = np.asarray(pcd.points, dtype=np.float32)
    point_radius = 0.0
    if points.shape[0]:
        point_radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.5)
    return max(1.0, camera_radius * 1.1, point_radius)


__all__ = [
    "CanonicalBootstrapPolicy",
    "PreparedDeviceSeed",
    "canonical_bootstrap_point_cloud",
    "canonical_camera_extent",
    "canonical_camera_from_observation",
    "prepared_device_seed_from_bindings",
]
