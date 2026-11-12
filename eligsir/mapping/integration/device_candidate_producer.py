"""Device-native finalization of RGB-D insertion candidates.

This module owns the post-backprojection half of the GPU-resident insertion
producer.  It deliberately returns :class:`geometry.insertion.SurfaceCandidatesT`
without converting indices, points, normals, colors, or scales through NumPy.
The legacy NumPy planner remains the compatibility fallback for feature
combinations that do not yet have an owner-safe tensor implementation.
"""

from __future__ import annotations

from typing import Optional

import torch

from geometry.device_fusion import fuse_points_with_knn_torch
from geometry.insertion import SurfaceCandidatesT
from geometry.normals import quaternion_from_normal_torch
from geometry.priors import GeometrySource
from eligsir.mapping.integration.geometry_helpers import _gaussian_min_axis_normals
from eligsir.utils.rocm_knn_fallback import distCUDA2, knn_indices
from eligsir.utils.torch_geometry import voxel_downsample_indices_torch


def device_candidate_path_supported(args) -> bool:
    """Return whether every enabled post-backprojection feature has a tensor path.

    The two excluded options require state that is still represented only by
    legacy host helpers.  Falling back is explicit rather than silently changing
    their semantics.
    """

    return not bool(getattr(args, "streaming_surface_cell_dedup", False)) and not bool(
        getattr(args, "streaming_insert_confidence_opacity", False)
    )


def _select_many(index: torch.Tensor, *values):
    return tuple(None if value is None else value[index] for value in values)


def finalize_device_surface_candidates(
    *,
    pts: torch.Tensor,
    colors: torch.Tensor,
    normals_cam: torch.Tensor,
    z: torch.Tensor,
    score: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    rgb_hwc: torch.Tensor,
    c2w: torch.Tensor,
    frame,
    insertion_policy,
    streaming_scene,
    gaussians,
    depth_native_hw: tuple[int, int],
    target_hw: tuple[int, int],
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
    depth_stride: int,
    voxel_size: float,
    cover_voxel: float,
    max_new: int,
    init_opacity: float,
    current_frame_idx: int,
) -> Optional[SurfaceCandidatesT]:
    """Finish one already-device-resident candidate set entirely in torch.

    Composition has already rejected optional features without tensor-native
    parity; this executor neither reads parser arguments nor selects a fallback.
    """
    if pts.device.type == "cpu":
        return None
    device = pts.device
    tensors = (colors, normals_cam, z, score, xs, ys, rgb_hwc, c2w)
    if any(value.device != device for value in tensors):
        raise ValueError("device-native candidate fields must share one device")

    pts = pts.to(torch.float32)
    colors = colors.to(torch.float32)
    normals_cam = normals_cam.to(torch.float32)
    z = z.to(torch.float32)
    score = score.to(torch.float32)

    # Tensor indexing above can produce a statically empty candidate batch.  A
    # shape check reads tensor metadata only, so preserve the empty fast path
    # without the device synchronization caused by ``valid.any().item()``.
    if pts.shape[0] == 0:
        return _empty(device, current_frame_idx)

    # Separate depth/color intrinsics: reproduce the legacy color-FOV reproject
    # and filter, but keep both the mask and gather on-device.
    native_h, native_w = depth_native_hw
    target_h, target_w = target_hw
    if getattr(frame, "depth_fx", None) is not None and (
        native_w != target_w or native_h != target_h
    ):
        x_over_z = (xs.to(torch.float32) - float(d_cx)) / float(d_fx)
        y_over_z = (ys.to(torch.float32) - float(d_cy)) / float(d_fy)
        u = torch.round(x_over_z * float(frame.fx) + float(frame.cx)).to(torch.long)
        v = torch.round(y_over_z * float(frame.fy) + float(frame.cy)).to(torch.long)
        in_fov = (u >= 0) & (u < int(frame.width)) & (v >= 0) & (v < int(frame.height))
        pts, normals_cam, z, score, xs, ys, u, v = _select_many(
            in_fov, pts, normals_cam, z, score, xs, ys, u, v
        )
        if pts.shape[0] == 0:
            return _empty(device, current_frame_idx)
        colors = rgb_hwc[v.clamp(0, rgb_hwc.shape[0] - 1), u.clamp(0, rgb_hwc.shape[1] - 1), :3]

    normals_world = normals_cam @ c2w[:3, :3].transpose(0, 1)
    normals_world = torch.nn.functional.normalize(normals_world, dim=-1, eps=1e-8)

    occupied = streaming_scene.check_occupancy_torch(
        pts, float(cover_voxel), check_neighbors=True
    )
    keep = ~occupied
    pts, colors, normals_world, z, score = _select_many(
        keep, pts, colors, normals_world, z, score
    )
    if pts.shape[0] == 0:
        return _empty(device, current_frame_idx)

    fusion_on = bool(insertion_policy.surface_fusion_enabled)
    pts_pre = pts if fusion_on else None
    pts, first = voxel_downsample_indices_torch(pts, float(voxel_size))
    colors, normals_world, z, score = _select_many(first, colors, normals_world, z, score)

    if fusion_on and pts_pre is not None and pts.shape[0] > 0 and pts_pre.shape[0] >= 2:
        k = max(1, min(int(insertion_policy.surface_fusion_neighbors), int(pts_pre.shape[0])))
        _, nn = knn_indices(pts, pts_pre, k=k, exclude_self=False)
        radius_mult = float(insertion_policy.surface_fusion_radius_multiplier)
        max_radius = radius_mult * float(voxel_size) if radius_mult > 0 else None
        pts, normals_world = fuse_points_with_knn_torch(
            pts, pts_pre, nn.clamp_min(0), normals_world, max_radius=max_radius
        )

    if bool(insertion_policy.knn_dedup_enabled) and pts.shape[0] > 0:
        existing = gaussians.get_xyz.detach()
        if existing.shape[0] > 0:
            normal_aware = bool(insertion_policy.knn_normal_aware)
            exist_q = gaussians.get_rotation.detach() if normal_aware else None
            exist_s = gaussians.get_scaling.detach() if normal_aware else None
            max_existing = int(insertion_policy.knn_max_existing)
            if existing.shape[0] > max_existing:
                stride = max(1, int(existing.shape[0]) // max_existing)
                existing = existing[::stride]
                if normal_aware:
                    exist_q = exist_q[::stride]
                    exist_s = exist_s[::stride]
            dist2, nn = knn_indices(pts, existing, k=1, exclude_self=False)
            min_dist = dist2[:, 0].clamp_min(0).sqrt()
            radius = z * float(insertion_policy.knn_radius_factor)
            too_close = min_dist < radius
            if normal_aware:
                nearest = nn[:, 0].clamp_min(0)
                exist_n = _gaussian_min_axis_normals(exist_q[nearest], exist_s[nearest])
                aligned = (normals_world * exist_n).sum(dim=-1).abs() > float(
                    insertion_policy.knn_normal_cosine
                )
                too_close = too_close & aligned
            keep = ~too_close
            pts, colors, normals_world, z, score = _select_many(
                keep, pts, colors, normals_world, z, score
            )
            if pts.shape[0] == 0:
                return _empty(device, current_frame_idx)

    if int(max_new) > 0 and pts.shape[0] > int(max_new):
        if str(insertion_policy.score_mode) == "weighted_topk":
            idx = torch.topk(score, k=int(max_new), largest=True, sorted=False).indices
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
            idx = torch.randperm(int(pts.shape[0]), generator=generator, device=device)[: int(max_new)]
        pts, colors, normals_world, z, score = _select_many(
            idx, pts, colors, normals_world, z, score
        )

    scale_mult = float(insertion_policy.scale_multiplier)
    scale_max = float(insertion_policy.scale_max_m)
    normal_ratio = float(insertion_policy.normal_scale_ratio)
    tx = torch.clamp(scale_mult * z / float(d_fx) * int(depth_stride), 1e-4, scale_max)
    ty = torch.clamp(scale_mult * z / float(d_fy) * int(depth_stride), 1e-4, scale_max)

    scale_mode = str(insertion_policy.scale_mode)
    if scale_mode in ("knn", "hybrid") and pts.shape[0] > 8:
        knn_iso = torch.clamp(
            distCUDA2(pts).clamp_min(1e-12).sqrt() * scale_mult,
            1e-4,
            scale_max,
        )
        if scale_mode == "knn":
            tx = knn_iso
            ty = knn_iso
        else:
            tx = torch.clamp(torch.sqrt(tx * knn_iso), 1e-4, scale_max)
            ty = torch.clamp(torch.sqrt(ty * knn_iso), 1e-4, scale_max)

    if bool(insertion_policy.distance_scale_enabled):
        coeff = float(insertion_policy.distance_scale_coefficient)
        tx = torch.maximum(tx, torch.clamp(coeff * z / float(d_fx), 1e-4, scale_max))
        ty = torch.maximum(ty, torch.clamp(coeff * z / float(d_fy), 1e-4, scale_max))

    if bool(insertion_policy.isotropic_scale):
        iso = torch.sqrt(tx * ty)
        scales = torch.stack((iso, iso, iso), dim=1)
    else:
        tz = torch.clamp(normal_ratio * torch.minimum(tx, ty), 1e-5, scale_max * normal_ratio)
        scales = torch.stack((tx, ty, tz), dim=1)
    log_scales = torch.log(scales).to(torch.float32)

    confidence = score.clamp(0.0, 1.0).to(torch.float32)
    has_positive_confidence = (confidence > 0).any()
    confidence = torch.where(
        has_positive_confidence, confidence, torch.ones_like(confidence)
    )
    source = torch.full(
        (pts.shape[0],), int(GeometrySource.SENSOR), dtype=torch.int16, device=device
    )
    quats = quaternion_from_normal_torch(normals_world).to(torch.float32)

    return SurfaceCandidatesT(
        pts=pts.contiguous().to(torch.float32),
        colors=colors.contiguous().to(torch.float32),
        normals=normals_world.contiguous().to(torch.float32),
        log_scales=log_scales.contiguous(),
        quats=quats.contiguous(),
        confidence=confidence.contiguous(),
        source=source.contiguous(),
        score=score.contiguous().to(torch.float32),
        birth_frame=int(current_frame_idx),
    )


def _empty(device: torch.device, birth_frame: int) -> SurfaceCandidatesT:
    return SurfaceCandidatesT(
        pts=torch.empty((0, 3), device=device, dtype=torch.float32),
        colors=torch.empty((0, 3), device=device, dtype=torch.float32),
        normals=torch.empty((0, 3), device=device, dtype=torch.float32),
        log_scales=torch.empty((0, 3), device=device, dtype=torch.float32),
        quats=torch.empty((0, 4), device=device, dtype=torch.float32),
        confidence=torch.empty((0,), device=device, dtype=torch.float32),
        source=torch.empty((0,), device=device, dtype=torch.int16),
        score=torch.empty((0,), device=device, dtype=torch.float32),
        birth_frame=int(birth_frame),
    )


def finalize_legacy_device_surface_candidates(*, args, **kwargs):
    """Historical host-planner adapter; never imported by canonical runtime."""

    if not device_candidate_path_supported(args):
        return None
    return finalize_device_surface_candidates(**kwargs)


__all__ = [
    "device_candidate_path_supported",
    "finalize_device_surface_candidates",
    "finalize_legacy_device_surface_candidates",
]
