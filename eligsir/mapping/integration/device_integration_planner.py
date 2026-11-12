"""Canonical device-resident RGB-D insertion planner.

Selected by ``rgbd_integrator_impl`` when both ``gpu_frame_prep`` and
``gpu_frame_resident`` are enabled. The source RGB, metric depth and 4x4 pose
are heterogeneous-packed into one pinned slab / one physical H2D. From there
all candidate masks, normals, backprojection, occupancy, fusion/KNN/scale
construction and sparse reliable source seeding remain on the target device.

The legacy planner remains an explicit compatibility fallback for optional
features that have no tensor-native parity implementation. Falling back is
decided *before* the packed transfer, never halfway through a device candidate
set after a D2H round-trip.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from eligsir.contracts.streaming_bridge import mapping_frame_source_arrays
from eligsir.contracts.control import DepthInsertionPlan, ModalityEligibility
from eligsir.mapping.integration.device_candidate_producer import (
    device_candidate_path_supported,
    finalize_device_surface_candidates,
)
from eligsir.mapping.integration.geometry_helpers import (
    _gpu_render_occupancy_masks,
    _kornia_normal_map,
    _sample_stable_depth_mask_gpu,
    _sample_temporal_depth_mask_gpu,
)
from eligsir.mapping.integration.sparse_seed_policy import (
    SparseSeedConfig,
    select_sparse_reliable_seeds,
)
from eligsir.mapping.integration.depth_render_reuse import (
    DepthInsertionRenderReuse,
)
from eligsir.runtime.transfers import DeviceTransferOwner, get_device_transfer_owner
from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch


def device_insertion_transfer_stats(device) -> dict[str, int]:
    return get_device_transfer_owner(device).stats.snapshot()


def device_resident_path_supported(args) -> tuple[bool, str]:
    if not bool(getattr(args, "gpu_frame_prep", False)):
        return False, "gpu_frame_prep_disabled"
    if not bool(getattr(args, "gpu_frame_resident", False)):
        return False, "gpu_frame_resident_disabled"
    if not device_candidate_path_supported(args):
        return False, "legacy_only_post_backprojection_feature_enabled"
    return True, "supported"


def _target_device(gaussians, render_alpha, render_depth) -> torch.device:
    if render_alpha is not None:
        return render_alpha.device
    if render_depth is not None:
        return render_depth.device
    return gaussians.get_xyz.device


def _load_source(frame):
    canonical = mapping_frame_source_arrays(frame)
    z, rgb = canonical
    mapping = frame.mapping_frame
    height, width = z.shape
    return (
        z,
        rgb,
        np.asarray(mapping.pose.c2w, dtype=np.float32),
        (height, width),
        (height, width),
    )


def _intrinsics(frame, native_hw, target_hw):
    if native_hw != target_hw:
        raise ValueError("canonical insertion does not rescale source calibration")
    intrinsics = frame.mapping_frame.intrinsics
    return (
        float(intrinsics.depth_fx),
        float(intrinsics.depth_fy),
        float(intrinsics.depth_cx),
        float(intrinsics.depth_cy),
    )


def _packed_source_to_device(z, rgb, c2w, device, transfer_owner=None):
    owner = transfer_owner or get_device_transfer_owner(device)
    fields, _event = owner.upload_packed(
        (
            np.ascontiguousarray(z, dtype=np.float32),
            np.ascontiguousarray(rgb, dtype=np.float32),
            np.ascontiguousarray(c2w, dtype=np.float32),
        ),
        purpose="frame_insertion",
    )
    z_t, rgb_t, c2w_t = fields
    if z_t is None or rgb_t is None or c2w_t is None:
        raise RuntimeError("packed RGB-D source transfer returned an empty field")
    return z_t, rgb_t, c2w_t


def _grid(height: int, width: int, stride: int, device):
    ys_1d = torch.arange(0, height, stride, dtype=torch.long, device=device)
    xs_1d = torch.arange(0, width, stride, dtype=torch.long, device=device)
    ys, xs = torch.meshgrid(ys_1d, xs_1d, indexing="ij")
    return ys, xs


def _validate_insertion_authority(
    frame,
    insertion_plan,
    insertion_policy,
    modality_eligibility,
) -> None:
    """Fail before GPU work unless the resolver authorized this exact source."""

    from eligsir.runtime.config import DepthInsertionPolicy

    if not isinstance(insertion_plan, DepthInsertionPlan):
        raise TypeError("device insertion requires a DepthInsertionPlan")
    if not isinstance(insertion_policy, DepthInsertionPolicy):
        raise TypeError("device insertion requires a DepthInsertionPolicy")
    if not isinstance(modality_eligibility, ModalityEligibility):
        raise TypeError("device insertion requires ModalityEligibility")
    if not insertion_plan.enabled:
        raise ValueError("device insertion requires an enabled plan")
    if not modality_eligibility.depth_insertion:
        raise ValueError("depth insertion is not modality-eligible")
    mapping = getattr(frame, "mapping_frame", None)
    if mapping is None:
        mapping = getattr(frame, "_mapping_frame", None)
    if mapping is None:
        raise RuntimeError("canonical insertion requires a canonical MappingFrame")
    if insertion_plan.source != mapping.source:
        raise ValueError("depth insertion plan source must match MappingFrame source")


def insert_gaussians_from_frame_device_native(
    gaussians,
    frame,
    streaming_scene,
    *,
    prev_frame=None,
    prev_depth_meters: Optional[np.ndarray] = None,
    render_alpha=None,
    render_depth=None,
    current_frame_idx: int = 0,
    transfer_owner: Optional[DeviceTransferOwner] = None,
    insertion_plan=None,
    insertion_policy=None,
    modality_eligibility=None,
    render_reuse: DepthInsertionRenderReuse | None = None,
):
    """Produce typed RGB-D candidates without choosing work or committing topology."""

    _validate_insertion_authority(
        frame, insertion_plan, insertion_policy, modality_eligibility
    )
    stats: dict = {
        "producer": "device_resident_v1",
        "candidate_d2h_ops": 0,
        # These are producer-local accounting facts.  The composition root
        # records the separately-attributed legacy occupancy render itself.
        "insertion_render_reused": False,
        "source_lease_hit": False,
        "source_h2d_ops": 0,
        # Source preparation is upload-only.  Keep its D2H attribution
        # explicit so a reuse regression cannot hide a full-frame readback
        # behind the producer's unrelated bounded candidate accounting.
        "source_d2h_ops": 0,
        "extra_rasterizer_calls": 0,
    }
    debug = bool(insertion_policy.debug_statistics)
    target = _target_device(gaussians, render_alpha, render_depth)
    if target.type == "cpu":
        raise RuntimeError("device-resident insertion requires an accelerator target")
    if render_reuse is not None:
        if not isinstance(render_reuse, DepthInsertionRenderReuse):
            raise TypeError("render_reuse must be DepthInsertionRenderReuse or None")
        if render_reuse.depth_meters.device != target:
            raise ValueError("render reuse device must match insertion target")
        z_t = render_reuse.depth_meters
        rgb_t = render_reuse.rgb
        c2w_t = render_reuse.c2w
        render_alpha = render_reuse.alpha
        render_depth = render_reuse.rendered_depth
        native_hw = target_hw = tuple(int(value) for value in z_t.shape)
        stats["insertion_render_reused"] = True
        stats["source_lease_hit"] = True
    else:
        z, rgb, c2w, native_hw, target_hw = _load_source(frame)
        z_t, rgb_t, c2w_t = _packed_source_to_device(
            z, rgb, c2w, target, transfer_owner
        )
        stats["source_h2d_ops"] = 1

    h, w = int(z_t.shape[0]), int(z_t.shape[1])
    if insertion_plan is None or insertion_policy is None:
        raise ValueError("device insertion requires typed policy and plan")
    # Sparse-to-dense depth-insertion density schedule (issue #1279's
    # device-native follow-up to
    # docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md,
    # mirroring the legacy-path wiring in
    # eligsir/mapping/integration/integration_planner.py::insert_gaussians_from_frame).
    # Resolves to the plan's own nominal ``sample_stride``, unchanged, when
    # --streaming_depth_sparsity_schedule is "fixed" (the default) -- this is
    # a LOCAL variable, never a mutation of ``insertion_plan``/
    # ``insertion_policy`` (both are shared, reused-elsewhere objects), so
    # every other reader of ``insertion_plan.sample_stride`` is unaffected.
    # Every downstream use of ``stride`` below (the sample grid, edge/normal
    # finite-difference span, and -- via ``finalize_device_surface_candidates``
    # -- the footprint-mode seeded scale) automatically picks up the
    # schedule's per-call value, keeping the seeded scale internally
    # consistent with the density actually sampled at THIS frame (design
    # doc Sec 4.1's scale confound) exactly as on the legacy path.
    from eligsir.utils.depth_sparsity_schedule import (
        config_from_policy as _depth_sparsity_config_from_policy,
        resolve_effective_depth_stride as _resolve_effective_depth_stride,
    )
    _depth_sparsity_schedule = _depth_sparsity_config_from_policy(
        insertion_policy, insertion_plan.sample_stride
    )
    stride = _resolve_effective_depth_stride(
        _depth_sparsity_schedule,
        current_frame_idx,
        insertion_plan.sample_stride,
    )
    if debug and _depth_sparsity_schedule is not None:
        stats["depth_sparsity_stride_eff"] = stride
        stats["depth_sparsity_frames_ingested"] = int(current_frame_idx)
    min_depth = float(insertion_policy.min_depth_m)
    max_depth = float(insertion_policy.max_depth_m)
    voxel_size = float(insertion_plan.voxel_size_m)
    max_new = int(insertion_plan.candidate_budget)
    init_opacity = float(insertion_policy.initial_opacity)
    edge_threshold = float(insertion_policy.edge_threshold)
    max_view_angle = float(insertion_policy.max_view_angle_deg)
    d_fx, d_fy, d_cx, d_cy = _intrinsics(frame, native_hw, target_hw)

    ys_i, xs_i = _grid(h, w, stride, target)
    ys = ys_i.to(torch.float32)
    xs = xs_i.to(torch.float32)
    z_v = z_t[ys_i, xs_i]
    valid = torch.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
    score = torch.ones_like(z_v, dtype=torch.float32)
    if debug:
        stats["raw_candidates"] = int(valid.sum().item())

    if bool(insertion_policy.stability_filter_enabled):
        before_stable = int(valid.sum().item()) if debug else 0
        stable = _sample_stable_depth_mask_gpu(
            z_t,
            xs,
            ys,
            min_depth,
            max_depth,
            erode_px=int(insertion_policy.stability_erode_px),
            window=int(insertion_policy.stability_window_px),
            min_valid_ratio=float(insertion_policy.stability_min_valid_ratio),
            max_range=float(insertion_policy.stability_max_range_m),
            median_thresh=float(insertion_policy.stability_median_threshold_m),
        )
        valid &= stable
        bottom_margin = float(insertion_policy.stability_bottom_margin)
        if bottom_margin > 0:
            valid &= ys < int(round(h * (1.0 - bottom_margin)))
        if debug:
            after_stable = int(valid.sum().item())
            stats["after_stable_depth_filter"] = after_stable
            stats["rejected_unstable_depth"] = before_stable - after_stable

    temporal = None
    if bool(insertion_policy.temporal_check_enabled):
        temporal = _sample_temporal_depth_mask_gpu(
            z_v,
            xs,
            ys,
            frame,
            prev_frame,
            prev_depth_meters,
            d_fx,
            d_fy,
            d_cx,
            d_cy,
            thresh=float(insertion_policy.temporal_tolerance_m),
        )
        if temporal is None and (
            prev_frame is not None
            and prev_depth_meters is not None
            and bool(insertion_policy.require_temporal_stability)
        ):
            temporal = torch.zeros_like(valid)

    if render_alpha is not None or render_depth is not None:
        low_alpha, no_depth, sensor_closer, score = _gpu_render_occupancy_masks(
            render_alpha, render_depth, xs, ys, z_v, h, w, insertion_policy
        )
        if (
            render_depth is not None
            and temporal is not None
            and bool(insertion_policy.require_temporal_for_sensor_closer)
        ):
            sensor_closer &= temporal
        mode = str(insertion_policy.score_mode)
        if mode == "alpha_only":
            insert_mask = low_alpha
        elif mode == "depth_gap_only":
            insert_mask = no_depth | sensor_closer
        elif mode == "sensor_closer_only":
            insert_mask = sensor_closer
        elif mode == "weighted_topk":
            insert_mask = score > 0
        else:
            insert_mask = low_alpha | no_depth | sensor_closer
        valid &= insert_mask
        if debug:
            stats["after_alpha_depth_mask"] = int(valid.sum().item())
            stats["insert_path_low_alpha"] = int((valid & low_alpha).sum().item())
            stats["insert_path_no_depth"] = int(
                (valid & no_depth & ~low_alpha).sum().item()
            )
            stats["insert_path_sensor_closer"] = int(
                (valid & sensor_closer & ~low_alpha & ~no_depth).sum().item()
            )

    if temporal is not None and bool(insertion_policy.require_temporal_stability):
        before_temporal = int(valid.sum().item()) if debug else 0
        valid &= temporal
        if debug:
            after_temporal = int(valid.sum().item())
            stats["after_temporal_depth_filter"] = after_temporal
            stats["rejected_temporal_depth"] = before_temporal - after_temporal
    elif debug and bool(insertion_policy.temporal_check_enabled):
        stats["after_temporal_depth_filter"] = int(valid.sum().item())
        stats["rejected_temporal_depth"] = 0

    if edge_threshold > 0:
        xp = torch.clamp(xs_i + stride, 0, w - 1)
        xm = torch.clamp(xs_i - stride, 0, w - 1)
        yp = torch.clamp(ys_i + stride, 0, h - 1)
        ym = torch.clamp(ys_i - stride, 0, h - 1)
        dzdx = torch.abs(z_t[ys_i, xp] - z_t[ys_i, xm])
        dzdy = torch.abs(z_t[yp, xs_i] - z_t[ym, xs_i])
        valid &= ((dzdx + dzdy) / z_v.clamp_min(1e-6)) < edge_threshold
        if debug:
            stats["after_edge_filter"] = int(valid.sum().item())

    # The compatibility path stops before the grazing stage when all prior
    # gates rejected the frame.  Compact one coordinate vector so the empty
    # decision comes from batch-shape metadata rather than ``any().item()``.
    if xs_i[valid].shape[0] == 0:
        return None, stats

    backend = str(insertion_policy.normal_estimation)
    if backend == "kornia":
        normal_map = _kornia_normal_map(z_t, d_fx, d_fy, d_cx, d_cy)
        nx = normal_map[0][ys_i, xs_i]
        ny = normal_map[1][ys_i, xs_i]
        nz = normal_map[2][ys_i, xs_i]
        flip = nz < 0
        nx = torch.where(flip, -nx, nx)
        ny = torch.where(flip, -ny, ny)
        nz = torch.where(flip, -nz, nz)
    else:
        xp = torch.clamp(xs_i + stride, 0, w - 1)
        xm = torch.clamp(xs_i - stride, 0, w - 1)
        yp = torch.clamp(ys_i + stride, 0, h - 1)
        ym = torch.clamp(ys_i - stride, 0, h - 1)
        dzdx = z_t[ys_i, xp] - z_t[ys_i, xm]
        dzdy = z_t[yp, xs_i] - z_t[ym, xs_i]
        nx = -dzdx / float(d_fx * 2.0)
        ny = -dzdy / float(d_fy * 2.0)
        nz = torch.ones_like(nx)
        norm = torch.sqrt(nx * nx + ny * ny + nz * nz).clamp_min(1e-8)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
    valid &= nz > float(np.cos(np.deg2rad(max_view_angle)))
    if debug:
        stats["after_grazing_filter"] = int(valid.sum().item())

    xs_v = xs_i[valid].to(torch.float32)
    ys_v = ys_i[valid].to(torch.float32)
    z_sel = z_v[valid].to(torch.float32)
    score_sel = score[valid].to(torch.float32)
    normals_cam = torch.stack((nx[valid], ny[valid], nz[valid]), dim=1)
    pts, colors = backproject_depth_pixels_torch(
        z_sel, xs_v, ys_v, d_fx, d_fy, d_cx, d_cy, c2w_t, rgb_t, (h, w)
    )

    candidates = finalize_device_surface_candidates(
        pts=pts,
        colors=colors,
        normals_cam=normals_cam,
        z=z_sel,
        score=score_sel,
        xs=xs_v,
        ys=ys_v,
        rgb_hwc=rgb_t,
        c2w=c2w_t,
        frame=frame,
        insertion_policy=insertion_policy,
        streaming_scene=streaming_scene,
        gaussians=gaussians,
        depth_native_hw=native_hw,
        target_hw=target_hw,
        d_fx=d_fx,
        d_fy=d_fy,
        d_cx=d_cx,
        d_cy=d_cy,
        depth_stride=stride,
        voxel_size=voxel_size,
        cover_voxel=voxel_size,
        max_new=max_new,
        init_opacity=init_opacity,
        current_frame_idx=current_frame_idx,
    )
    if candidates is None:
        raise RuntimeError("device candidate producer declined after path support was accepted")
    if len(candidates) == 0:
        # Preserve the legacy empty-result contract after the mask has been
        # compacted.  ``len`` reads batch-shape metadata and does not force a
        # device-to-host synchronization.
        return None, stats

    # #109/#192: source-born geometry is deliberately sparse/reliable. The
    # existing insertion score is the confidence evidence; no extra network or
    # host round-trip is introduced. A threshold of zero preserves all already-
    # accepted evidence while the deterministic voxel/cap prevents dense source
    # flooding. Experiments may raise the threshold by setting the namespaced
    # attribute through a ResearchFlow run profile.
    sparse_cfg = SparseSeedConfig(
        min_confidence=float(
            insertion_plan.minimum_confidence
            if insertion_plan is not None
            else insertion_policy.minimum_confidence
        ),
        voxel_size_m=float(
            insertion_plan.voxel_size_m
            if insertion_plan is not None
            else insertion_policy.voxel_size_m
        ),
        max_seeds_per_frame=int(
            insertion_plan.seed_budget
            if insertion_plan is not None
            else max_new
        ),
    )
    candidates, sparse_stats = select_sparse_reliable_seeds(candidates, sparse_cfg)
    stats.update({
        "sparse_seed_input": sparse_stats.input_count,
        "sparse_seed_reliable": sparse_stats.reliable_count,
        "sparse_seed_voxel": sparse_stats.voxel_count,
        "sparse_seed_output": sparse_stats.output_count,
        "device_native_candidates": len(candidates),
    })
    if len(candidates) == 0:
        return None, stats
    return candidates, stats


__all__ = [
    "device_insertion_transfer_stats",
    "device_resident_path_supported",
    "insert_gaussians_from_frame_device_native",
]
