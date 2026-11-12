"""``insert_gaussians_from_frame`` -- the main ~900-line RGB-D-frame-to-Gaussians
hot path, split out of ``rgbd_integrator_impl.py`` (WS-E5) -- see that
module's docstring for the history of why this cluster ended up as one file.

Verbatim relocation, no math/logic change. ``rgbd_integrator_impl.py``
re-exports this function so every existing call site keeps working unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image as _Image

from scene.streaming_scene import StreamingScene
from geometry.priors import GeometrySource
from eligsir.mapping.integration.geometry_helpers import (
    _gaussian_min_axis_normals,
    _gpu_render_occupancy_masks,
    _kornia_normal_map,
    _quaternion_from_normal,
    _sample_stable_depth_mask,
    _sample_stable_depth_mask_gpu,
    _sample_temporal_depth_mask,
    _sample_temporal_depth_mask_gpu,
    finite_diff_normal_denominators,
)
from eligsir.mapping.integration.frame_preparation import _voxel_downsample_indices
from eligsir.utils.rgbd_frames import depth_to_meters
from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb
from eligsir.contracts.streaming_bridge import mapping_frame_source_arrays


def _load_source_arrays(frame):
    """Prefer canonical MappingFrame pixels, preserving the legacy fallback."""

    canonical = mapping_frame_source_arrays(
        frame, required=False, require_pose=False
    )
    if canonical is not None:
        depth_m, rgb = canonical
        return depth_m, rgb, tuple(depth_m.shape[:2]), True
    depth = np.asarray(load_frame_depth_np(frame))
    native_hw = tuple(depth.shape[:2])
    target_h, target_w = int(frame.height), int(frame.width)
    if depth.shape[:2] != (target_h, target_w):
        depth = np.array(
            _Image.fromarray(depth).resize((target_w, target_h), _Image.NEAREST)
        )
    rgb_pil = load_frame_rgb(frame)
    if rgb_pil.size != (target_w, target_h):
        rgb_pil = rgb_pil.resize((target_w, target_h), _Image.BILINEAR)
    rgb = np.array(rgb_pil).astype(np.float32) / 255.0
    return depth, rgb, native_hw, False


def insert_gaussians_from_frame(
    gaussians, frame, args,
    streaming_scene: "StreamingScene",
    prev_frame=None,
    prev_depth_meters: Optional[np.ndarray] = None,
    render_alpha: Optional[torch.Tensor] = None,
    render_depth: Optional[torch.Tensor] = None,
    current_frame_idx: int = 0,
    iteration: int = 0,
    _collect_only: bool = False,
    map_backend=None,
) -> tuple:
    """
    Phase 2: backproject an RGB-D frame, voxel-filter, remove already-covered
    regions, and append new Gaussians as surface-aligned surfels.

    When _collect_only=True, returns (candidates, stats_dict) without
    committing or updating the occupancy hash; the caller commits the batch.
    ``candidates`` is a :class:`geometry.insertion.SurfaceCandidatesT` on the
    device-native producer path, and otherwise the legacy dict of pre-commit
    NumPy arrays (pts, cols, log_scales, q_world, birth_frame) -- an empty
    dict when every point was filtered out. Accumulating callers must use
    ``device_candidates.collected_candidate_points`` rather than indexing
    ``["pts"]``, which would silently drop the device-native shape.
    When False (default), returns (n_inserted, stats_dict) as before.

    ``iteration``: the real training iteration this insertion commits at,
    stamped into each new Gaussian's ``birth_iter`` on BOTH commit branches
    below. Distinct from ``current_frame_idx`` (a source-frame index, stored
    separately as ``birth_frame``) -- ``birth_iter`` is what age/young-
    protection logic compares against the optimizer step counter, so mixing
    the two units silently breaks it. Defaults to 0 so pre-existing callers
    that don't track an iteration keep working unchanged; the real streaming
    loop always passes it.

    ``map_backend``: optional :class:`eligsir.map.contracts.GaussianMapBackend`
    for ``gaussians`` (Wave 2 backend-ownership pass, 2026-08-06). The
    direct (non-``_collect_only``) insertion below commits through it so
    ``map_version``/``topology_version`` advance. Omitting it no longer
    selects a different write mechanism (PR #1142 removed the direct
    ``add_points_as_gaussians`` fallback): a transaction-only ephemeral
    backend is constructed instead, so the write path is the same either
    way and only the version counters' lifetime differs.
    """
    from eligsir.utils.rgbd_frames import depth_to_meters
    from PIL import Image as _Image

    _debug = getattr(args, "streaming_insertion_debug", False)
    _stats: dict = {}

    if frame._depth_np is None and frame._depth_bytes is None and (frame.depth_path is None or not os.path.exists(frame.depth_path)):
        return ({} if _collect_only else 0), _stats

    # Sparse-to-dense depth-insertion density schedule (issue #1279 follow-up
    # to docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md).
    # Resolves to the plain nominal streaming_depth_stride, unchanged, when
    # --streaming_depth_sparsity_schedule is "fixed" (the default) -- this is
    # a LOCAL variable, never a mutation of args.streaming_depth_stride, so
    # scene/streaming_scene.py's two other bare readers of that attribute
    # (bootstrap point-cloud construction, scene-complexity telemetry) are
    # unaffected. Every downstream use of `depth_stride` in this function
    # (mgrid subsampling, edge/normal finite-difference span, footprint-mode
    # seeded scale) automatically picks up the schedule's per-call value,
    # which is also what keeps the seeded scale internally consistent with
    # the density actually sampled at THIS frame rather than a stale nominal
    # constant (design doc §4.1's scale confound).
    from eligsir.utils.depth_sparsity_schedule import (
        config_from_args as _depth_sparsity_config_from_args,
        resolve_effective_depth_stride as _resolve_effective_depth_stride,
    )
    _depth_sparsity_schedule = _depth_sparsity_config_from_args(args)
    depth_stride = _resolve_effective_depth_stride(
        _depth_sparsity_schedule,
        current_frame_idx,
        getattr(args, "streaming_depth_stride", 8),
    )
    if _debug and _depth_sparsity_schedule is not None:
        _stats["depth_sparsity_stride_eff"] = depth_stride
        _stats["depth_sparsity_frames_ingested"] = int(current_frame_idx)
    min_depth = getattr(args, "streaming_min_depth", 0.1)
    max_depth = getattr(args, "streaming_max_depth", 8.0)
    voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
    # Occupancy check always uses the same voxel_size as the hash (cover_voxel_size /
    # cover_voxel_multiplier are deprecated; the hash enforces a single voxel size).
    cover_voxel = voxel_size
    # No getattr fallback: streaming_max_new_gaussians_per_frame is an
    # unconditional StreamingParams default, always present on the real
    # parsed args this hot path runs with (issue #363 residue -- the old
    # getattr(..., 2000) fallback literal had drifted from the real default
    # of 1000; reachability-audited dead code, removed rather than restated).
    max_new = args.streaming_max_new_gaussians_per_frame
    init_opacity = getattr(args, "streaming_insert_opacity", 0.05)
    edge_threshold = getattr(args, "streaming_depth_edge_threshold", 0.02)
    max_view_angle = getattr(args, "streaming_max_view_angle", 70.0)
    use_knn_scale = getattr(args, "streaming_insert_knn_scale", True)

    try:
        depth, rgb, (_depth_native_h, _depth_native_w), _depth_is_metric = (
            _load_source_arrays(frame)
        )
        _tgt_h, _tgt_w = int(frame.height), int(frame.width)
    except Exception:
        return ({} if _collect_only else 0), _stats

    # Use sensor-specific depth intrinsics when available (e.g. ScanNet/RealSense
    # have separate color and depth cameras with different resolutions).
    d_fx = frame.depth_fx if frame.depth_fx is not None else frame.fx
    d_fy = frame.depth_fy if frame.depth_fy is not None else frame.fy
    d_cx = frame.depth_cx if frame.depth_cx is not None else frame.cx
    d_cy = frame.depth_cy if frame.depth_cy is not None else frame.cy
    # Scale depth intrinsics when the depth image was resized to color dimensions.
    # Without this, backprojection uses native-resolution intrinsics on a resized
    # image, placing 3D points at wrong lateral positions.
    if frame.depth_fx is not None and (_depth_native_w != _tgt_w or _depth_native_h != _tgt_h):
        d_fx = d_fx * _tgt_w / _depth_native_w
        d_fy = d_fy * _tgt_h / _depth_native_h
        d_cx = d_cx * _tgt_w / _depth_native_w
        d_cy = d_cy * _tgt_h / _depth_native_h

    # GPU fast path (Phase 1 perf): keep render-occupancy masking, backprojection,
    # finite-diff normals and grazing/edge masks on the GPU. The numpy path below
    # remains the byte-for-byte default fallback when the flag is off.
    use_gpu = bool(getattr(args, "gpu_frame_prep", False))
    # WS2 M1: additionally keep the stable/temporal/occupancy/edge MASK
    # computation itself on-device (no interleaved .cpu().numpy() downloads,
    # no numpy round-trip before backprojection). Requires gpu_frame_prep;
    # default off => byte-for-byte identical to gpu_frame_prep alone.
    gpu_resident = use_gpu and bool(getattr(args, "gpu_frame_resident", False))
    # Choose the device from the render tensors when available; else fall back to
    # the gaussian model's device (cuda during training).
    _gpu_device = None
    if use_gpu:
        if render_alpha is not None:
            _gpu_device = render_alpha.device
        elif render_depth is not None:
            _gpu_device = render_depth.device
        else:
            try:
                _gpu_device = gaussians.get_xyz.device
            except Exception:
                _gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = depth if _depth_is_metric else depth_to_meters(depth, frame.depth_scale)
    h, w = z.shape
    ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
    z_v = z[ys, xs]
    valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
    score_grid = np.ones_like(valid, dtype=np.float32)

    # GPU-resident copies of the strided sensor grid (one small transfer of the
    # depth map; the full render tensors are gathered on-device and never crossed).
    if use_gpu:
        z_t = torch.from_numpy(np.ascontiguousarray(z)).to(device=_gpu_device, dtype=torch.float32)
        ys_t = torch.from_numpy(np.ascontiguousarray(ys)).to(device=_gpu_device, dtype=torch.float32)
        xs_t = torch.from_numpy(np.ascontiguousarray(xs)).to(device=_gpu_device, dtype=torch.float32)
        z_v_t = z_t[ys_t.to(torch.int64), xs_t.to(torch.int64)]
        if gpu_resident:
            # M1: valid_t/score_grid_t are the master copies from here on;
            # the numpy valid/score_grid above are NOT kept in sync in this
            # branch and must not be read again until the single conversion
            # at the "Sampling for normals" boundary below.
            valid_t = torch.isfinite(z_v_t) & (z_v_t > min_depth) & (z_v_t < max_depth)
            score_grid_t = torch.ones_like(valid_t, dtype=torch.float32)
    if _debug:
        _stats["raw_candidates"] = int(valid.sum())

    if args.streaming_depth_filter_enabled:
        _filter_kwargs = dict(
            erode_px=int(getattr(args, "streaming_depth_filter_erode_px", 2)),
            window=int(getattr(args, "streaming_depth_filter_window", 5)),
            min_valid_ratio=float(getattr(args, "streaming_depth_filter_min_valid_ratio", 0.75)),
            max_range=float(getattr(args, "streaming_depth_filter_max_range", 0.08)),
            median_thresh=float(getattr(args, "streaming_depth_filter_median_thresh", 0.04)),
        )
        if gpu_resident:
            # M1: stays a tensor -- no .cpu().numpy() download.
            stable_depth_t = _sample_stable_depth_mask_gpu(
                z_t, xs_t, ys_t, min_depth, max_depth, **_filter_kwargs
            )
            before_stable = int(valid_t.sum().item())
            valid_t = valid_t & stable_depth_t
            bottom_margin = float(getattr(args, "streaming_depth_filter_bottom_margin", 0.0))
            if bottom_margin > 0:
                valid_t = valid_t & (ys_t < int(round(h * (1.0 - bottom_margin))))
            if _debug:
                _stats["after_stable_depth_filter"] = int(valid_t.sum().item())
                _stats["rejected_unstable_depth"] = before_stable - int(valid_t.sum().item())
        elif use_gpu:
            # GPU path: the 25-plane gather-stack + reductions run on-device over
            # z_t/xs_t/ys_t (already resident); only the small bool result crosses
            # back. This removes the dominant per-frame numpy cost on dense sensors.
            stable_depth = (
                _sample_stable_depth_mask_gpu(
                    z_t, xs_t, ys_t, min_depth, max_depth, **_filter_kwargs
                )
                .cpu()
                .numpy()
            )
        else:
            stable_depth = _sample_stable_depth_mask(
                z, xs, ys, min_depth, max_depth, **_filter_kwargs
            )
        if not gpu_resident:
            before_stable = int(valid.sum())
            valid = valid & stable_depth
            bottom_margin = float(getattr(args, "streaming_depth_filter_bottom_margin", 0.0))
            if bottom_margin > 0:
                valid = valid & (ys < int(round(h * (1.0 - bottom_margin))))
            if _debug:
                _stats["after_stable_depth_filter"] = int(valid.sum())
                _stats["rejected_unstable_depth"] = before_stable - int(valid.sum())
    else:
        stable_depth = np.ones_like(valid, dtype=bool)
        if gpu_resident:
            pass  # valid_t already reflects the un-filtered initial validity

    temporal_ok = None
    temporal_check_enabled = bool(args.streaming_depth_temporal_check)
    temporal_thresh = float(getattr(args, "streaming_depth_temporal_thresh", 0.07))
    if gpu_resident:
        temporal_ok_t = None
        if temporal_check_enabled and bool(valid_t.any().item()):
            temporal_ok_t = _sample_temporal_depth_mask_gpu(
                z_v_t, xs_t, ys_t, frame, prev_frame, prev_depth_meters,
                d_fx, d_fy, d_cx, d_cy, thresh=temporal_thresh,
            )
            if temporal_ok_t is None and (
                prev_frame is not None and prev_depth_meters is not None and
                args.streaming_depth_temporal_require_for_insert
            ):
                temporal_ok_t = torch.zeros_like(valid_t)
    elif temporal_check_enabled and valid.any():
        temporal_flat = _sample_temporal_depth_mask(
            z_v[valid].astype(np.float32),
            xs[valid].astype(np.int32),
            ys[valid].astype(np.int32),
            frame,
            prev_frame,
            prev_depth_meters,
            d_fx,
            d_fy,
            d_cx,
            d_cy,
            thresh=temporal_thresh,
        )
        if temporal_flat is not None:
            temporal_ok = np.zeros_like(valid, dtype=bool)
            temporal_ok[valid] = temporal_flat
        elif (prev_frame is not None and prev_depth_meters is not None and
              args.streaming_depth_temporal_require_for_insert):
            temporal_ok = np.zeros_like(valid, dtype=bool)

    # ---- Alpha + Depth occupancy masking (Step 4) ----------------------------
    # Render may be at a different resolution than the raw sensor (--resolution
    # rescaling). Map sensor pixel coords into render pixel coords before indexing.
    #
    # Insert a depth candidate if ANY of these hold:
    #   (a) alpha < 0.3  → pixel is unoccupied in the current map
    #   (b) rendered depth = 0 → nothing rendered here at all
    #   (c) rendered depth > sensor depth + thresh → sensor found a closer surface
    #       the map hasn't captured yet (e.g. behind a floater)
    #
    # Previously only (a) was tested, which caused N to plateau once every
    # visible pixel reached alpha ≥ 0.3 — regardless of new geometry in-frame.
    if gpu_resident and (render_alpha is not None or render_depth is not None):
        # M1: same gather/scoring math as _gpu_render_occupancy_masks below,
        # but everything stays a tensor -- no .cpu().numpy() of any of the
        # four masks.
        la_t, nrd_t, sc_t, sg_t = _gpu_render_occupancy_masks(
            render_alpha, render_depth, xs_t, ys_t, z_v_t, h, w, args
        )
        if (render_depth is not None and temporal_ok_t is not None and
                args.streaming_depth_temporal_require_for_sensor_closer):
            sc_t = sc_t & temporal_ok_t

        score_mode = str(getattr(args, "streaming_insert_score_mode", "boolean_or"))
        if score_mode == "alpha_only":
            insert_mask_t = la_t
        elif score_mode == "depth_gap_only":
            insert_mask_t = nrd_t | sc_t
        elif score_mode == "sensor_closer_only":
            insert_mask_t = sc_t
        elif score_mode == "weighted_topk":
            insert_mask_t = sg_t > 0
        else:
            insert_mask_t = la_t | nrd_t | sc_t
        valid_t = valid_t & insert_mask_t
        score_grid_t = sg_t
        if _debug:
            _stats["after_alpha_depth_mask"] = int(valid_t.sum().item())
            _stats["insert_path_low_alpha"] = int((valid_t & la_t).sum().item())
            _stats["insert_path_no_depth"] = int((valid_t & nrd_t & ~la_t).sum().item())
            _stats["insert_path_sensor_closer"] = int((valid_t & sc_t & ~la_t & ~nrd_t).sum().item())
    elif render_alpha is not None or render_depth is not None:
        sh2d = xs.shape  # (Nh, Nw) — all working arrays must stay 2D
        low_alpha = np.zeros(sh2d, dtype=bool)
        no_rend_depth = np.ones(sh2d, dtype=bool)   # default: treat as "no depth"
        sensor_closer = np.zeros(sh2d, dtype=bool)
        score_grid = np.zeros(sh2d, dtype=np.float32)

        if use_gpu:
            # Gather render_alpha/render_depth ON the GPU at the strided grid coords
            # (same rounding/clip math as the numpy branch). Only the final small
            # strided masks cross to CPU (one small transfer of the strided grid).
            la_t, nrd_t, sc_t, sg_t = _gpu_render_occupancy_masks(
                render_alpha, render_depth, xs_t, ys_t, z_v_t, h, w, args
            )
            low_alpha = la_t.cpu().numpy()
            no_rend_depth = nrd_t.cpu().numpy()
            sensor_closer = sc_t.cpu().numpy()
            score_grid = sg_t.cpu().numpy().astype(np.float32)
            if (render_depth is not None and temporal_ok is not None and
                    args.streaming_depth_temporal_require_for_sensor_closer):
                sensor_closer = sensor_closer & temporal_ok
        elif render_alpha is not None:
            alpha_cpu = render_alpha.detach().cpu().numpy().squeeze()  # [Hr, Wr]
            Hr, Wr = alpha_cpu.shape
            xs_r = np.clip(np.round(xs / max(w - 1, 1) * (Wr - 1)).astype(np.int32), 0, Wr - 1)
            ys_r = np.clip(np.round(ys / max(h - 1, 1) * (Hr - 1)).astype(np.int32), 0, Hr - 1)
            alpha_v = alpha_cpu[ys_r, xs_r]
            alpha_thresh = float(getattr(args, "streaming_insert_alpha_threshold", 0.3))
            low_alpha = alpha_v < alpha_thresh
            score_grid += np.maximum(alpha_thresh - alpha_v, 0.0).astype(np.float32) * float(
                getattr(args, "streaming_insert_score_alpha_weight", 1.0)
            )

        if (not use_gpu) and render_depth is not None:
            rend_d_cpu = render_depth.detach().cpu().numpy().squeeze()
            Hr_d, Wr_d = rend_d_cpu.shape
            xs_rd = np.clip(np.round(xs / max(w - 1, 1) * (Wr_d - 1)).astype(np.int32), 0, Wr_d - 1)
            ys_rd = np.clip(np.round(ys / max(h - 1, 1) * (Hr_d - 1)).astype(np.int32), 0, Hr_d - 1)
            rend_d_v = rend_d_cpu[ys_rd, xs_rd]
            consistency_thresh = getattr(args, "streaming_depth_consistency_thresh", 0.05)
            no_rend_depth = rend_d_v <= 0
            sensor_closer = rend_d_v - z_v > consistency_thresh
            score_grid += no_rend_depth.astype(np.float32) * float(
                getattr(args, "streaming_insert_score_no_depth_weight", 1.0)
            )
            score_grid += np.maximum(rend_d_v - z_v, 0.0).astype(np.float32) * float(
                getattr(args, "streaming_insert_score_sensor_closer_weight", 2.0)
            )
            if (temporal_ok is not None and
                    args.streaming_depth_temporal_require_for_sensor_closer):
                sensor_closer = sensor_closer & temporal_ok

        score_mode = str(getattr(args, "streaming_insert_score_mode", "boolean_or"))
        if score_mode == "alpha_only":
            insert_mask = low_alpha
        elif score_mode == "depth_gap_only":
            insert_mask = no_rend_depth | sensor_closer
        elif score_mode == "sensor_closer_only":
            insert_mask = sensor_closer
        elif score_mode == "weighted_topk":
            insert_mask = score_grid > 0
        else:
            insert_mask = low_alpha | no_rend_depth | sensor_closer
        valid = valid & insert_mask
        if _debug:
            _stats["after_alpha_depth_mask"] = int(valid.sum())
            _stats["insert_path_low_alpha"] = int((valid & low_alpha).sum())
            _stats["insert_path_no_depth"] = int((valid & no_rend_depth & ~low_alpha).sum())
            _stats["insert_path_sensor_closer"] = int((valid & sensor_closer & ~low_alpha & ~no_rend_depth).sum())

    if gpu_resident:
        if temporal_ok_t is not None and args.streaming_depth_temporal_require_for_insert:
            before_temporal = int(valid_t.sum().item())
            valid_t = valid_t & temporal_ok_t
            if _debug:
                _stats["after_temporal_depth_filter"] = int(valid_t.sum().item())
                _stats["rejected_temporal_depth"] = before_temporal - int(valid_t.sum().item())
        elif _debug and temporal_check_enabled:
            _stats["after_temporal_depth_filter"] = int(valid_t.sum().item())
            _stats["rejected_temporal_depth"] = 0
    elif temporal_ok is not None and args.streaming_depth_temporal_require_for_insert:
        before_temporal = int(valid.sum())
        valid = valid & temporal_ok
        if _debug:
            _stats["after_temporal_depth_filter"] = int(valid.sum())
            _stats["rejected_temporal_depth"] = before_temporal - int(valid.sum())
    elif _debug and temporal_check_enabled:
        _stats["after_temporal_depth_filter"] = int(valid.sum())
        _stats["rejected_temporal_depth"] = 0

    # ---- Relative depth discontinuity masking (Step 5) ----------------------
    if edge_threshold > 0:
        s = depth_stride
        if gpu_resident:
            # M1: same central-difference edge metric as the use_gpu branch
            # below, but stays a tensor -- no .cpu().numpy() download.
            xs_i = xs_t.to(torch.int64)
            ys_i = ys_t.to(torch.int64)
            xp = torch.clamp(xs_i + s, 0, w - 1)
            xm = torch.clamp(xs_i - s, 0, w - 1)
            yp = torch.clamp(ys_i + s, 0, h - 1)
            ym = torch.clamp(ys_i - s, 0, h - 1)
            dzdx_t = torch.abs(z_t[ys_i, xp] - z_t[ys_i, xm])
            dzdy_t = torch.abs(z_t[yp, xs_i] - z_t[ym, xs_i])
            edge_rel_t = (dzdx_t + dzdy_t) / torch.clamp(z_v_t, min=1e-6)
            valid_t = valid_t & (edge_rel_t < edge_threshold)
        elif use_gpu:
            # Same central-difference edge metric, evaluated on GPU then crossed
            # to CPU as the small strided mask.
            xs_i = xs_t.to(torch.int64)
            ys_i = ys_t.to(torch.int64)
            xp = torch.clamp(xs_i + s, 0, w - 1)
            xm = torch.clamp(xs_i - s, 0, w - 1)
            yp = torch.clamp(ys_i + s, 0, h - 1)
            ym = torch.clamp(ys_i - s, 0, h - 1)
            dzdx_t = torch.abs(z_t[ys_i, xp] - z_t[ys_i, xm])
            dzdy_t = torch.abs(z_t[yp, xs_i] - z_t[ym, xs_i])
            edge_rel_t = (dzdx_t + dzdy_t) / torch.clamp(z_v_t, min=1e-6)
            edge_ok = (edge_rel_t < edge_threshold).cpu().numpy()
            valid = valid & edge_ok
        else:
            dzdx = np.abs(z[ys, np.clip(xs + s, 0, w - 1)] - z[ys, np.clip(xs - s, 0, w - 1)])
            dzdy = np.abs(z[np.clip(ys + s, 0, h - 1), xs] - z[np.clip(ys - s, 0, h - 1), xs])
            edge_rel = (dzdx + dzdy) / np.maximum(z_v, 1e-6)
            valid = valid & (edge_rel < edge_threshold)
        if _debug:
            if gpu_resident:
                _stats["after_edge_filter"] = int(valid_t.sum().item())
            else:
                _stats["after_edge_filter"] = int(valid.sum())

    if gpu_resident:
        if not bool(valid_t.any().item()):
            return ({} if _collect_only else 0), _stats
    else:
        if not valid.any():
            return ({} if _collect_only else 0), _stats

    # Sampling for normals
    s = depth_stride

    if gpu_resident:
        # M2: normals + grazing computed DENSELY on the full (Gh,Gw) grid and
        # folded directly into valid_t (one mask, no separate compact-then-
        # filter-then-compact-again) -- per the design doc's "folded into
        # keep_mask, not compacted" M2 requirement. Safe because the grazing
        # test is purely per-pixel (finite differences against neighboring
        # z_t values), independent of any other pixel's validity: computing
        # it everywhere then ANDing into valid_t is exactly equivalent to
        # computing it only on the already-valid subset (AND is commutative/
        # associative), and NaN/garbage values at already-invalid positions
        # can't matter since valid_t is already False there (`NaN > x` is
        # always False in IEEE 754, so no exception risk either). Only ONE
        # compaction happens below, pulling every field (position, score,
        # normal) from the same final valid_t.
        from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch
        xs_i_all = xs_t.to(torch.int64)
        ys_i_all = ys_t.to(torch.int64)
        cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
        _normals_backend = str(getattr(args, "gpu_frame_prep_normals", "finite_diff"))
        if _normals_backend == "kornia":
            _nmap = _kornia_normal_map(z_t, d_fx, d_fy, d_cx, d_cy)  # [3, H, W]
            nx_grid_t = _nmap[0][ys_i_all, xs_i_all]
            ny_grid_t = _nmap[1][ys_i_all, xs_i_all]
            nz_grid_t = _nmap[2][ys_i_all, xs_i_all]
            _flip = nz_grid_t < 0
            nx_grid_t = torch.where(_flip, -nx_grid_t, nx_grid_t)
            ny_grid_t = torch.where(_flip, -ny_grid_t, ny_grid_t)
            nz_grid_t = torch.where(_flip, -nz_grid_t, nz_grid_t)
        else:
            xp = torch.clamp(xs_i_all + s, 0, w - 1)
            xm = torch.clamp(xs_i_all - s, 0, w - 1)
            yp = torch.clamp(ys_i_all + s, 0, h - 1)
            ym = torch.clamp(ys_i_all - s, 0, h - 1)
            dzdx_g_t = z_t[ys_i_all, xp] - z_t[ys_i_all, xm]
            dzdy_g_t = z_t[yp, xs_i_all] - z_t[ym, xs_i_all]

            denom_x, denom_y = finite_diff_normal_denominators(d_fx, d_fy, s)
            nx_grid_t = -dzdx_g_t / denom_x
            ny_grid_t = -dzdy_g_t / denom_y
            nz_grid_t = torch.ones_like(nx_grid_t)
            norm_g_t = torch.sqrt(nx_grid_t * nx_grid_t + ny_grid_t * ny_grid_t + nz_grid_t * nz_grid_t) + 1e-8
            nx_grid_t = nx_grid_t / norm_g_t; ny_grid_t = ny_grid_t / norm_g_t; nz_grid_t = nz_grid_t / norm_g_t

        angle_ok_grid_t = nz_grid_t > cos_thresh
        valid_t = valid_t & angle_ok_grid_t

        if _debug:
            _stats["after_grazing_filter"] = int(valid_t.sum().item())
        if not bool(valid_t.any().item()):
            return ({} if _collect_only else 0), _stats

        # Single compaction: every field pulled from the same final valid_t.
        xs_v_t = xs_i_all[valid_t].to(torch.float32)
        ys_v_t = ys_i_all[valid_t].to(torch.float32)
        z_v_sel_t = z_v_t[valid_t].to(torch.float32)
        score_v_t = score_grid_t[valid_t]
        nx_t = nx_grid_t[valid_t]
        ny_t = ny_grid_t[valid_t]
        nz_t = nz_grid_t[valid_t]

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            device=_gpu_device, dtype=torch.float32
        )
        c2w_t = torch.from_numpy(np.ascontiguousarray(frame.c2w)).to(
            device=_gpu_device, dtype=torch.float32
        )
        pts_t, cols_t = backproject_depth_pixels_torch(
            z_v_sel_t, xs_v_t, ys_v_t, d_fx, d_fy, d_cx, d_cy, c2w_t, rgb_t, (h, w)
        )

        # Device-native finalizer: occupancy, voxel selection, optional KNN
        # fusion/dedup and scale construction stay in torch and emit the
        # production SurfaceCandidatesT, with no bulk point/index D2H. It
        # returns None when an enabled option has no tensor-native parity
        # implementation, in which case the NumPy compatibility tail below
        # runs exactly as before.
        from eligsir.mapping.integration.device_candidate_producer import (
            finalize_legacy_device_surface_candidates,
        )

        _device_candidates = finalize_legacy_device_surface_candidates(
            pts=pts_t,
            colors=cols_t,
            normals_cam=torch.stack((nx_t, ny_t, nz_t), dim=1),
            z=z_v_sel_t,
            score=score_v_t,
            xs=xs_v_t,
            ys=ys_v_t,
            rgb_hwc=rgb_t,
            c2w=c2w_t,
            frame=frame,
            args=args,
            streaming_scene=streaming_scene,
            gaussians=gaussians,
            depth_native_hw=(_depth_native_h, _depth_native_w),
            target_hw=(_tgt_h, _tgt_w),
            d_fx=d_fx,
            d_fy=d_fy,
            d_cx=d_cx,
            d_cy=d_cy,
            depth_stride=depth_stride,
            voxel_size=voxel_size,
            cover_voxel=cover_voxel,
            max_new=max_new,
            init_opacity=init_opacity,
            current_frame_idx=current_frame_idx,
        )
        if _device_candidates is not None:
            if _debug:
                _stats["device_native_candidates"] = int(_device_candidates.pts.shape[0])
            if _collect_only:
                return _device_candidates, _stats
            from eligsir.mapping.integration.insertion_committer import (
                flush_insertion_batch,
            )

            added = flush_insertion_batch(
                [_device_candidates],
                gaussians,
                streaming_scene,
                iteration,
                args,
                map_backend=map_backend,
            )
            if _debug:
                _stats["inserted"] = int(added)
            return int(added), _stats

        xs_v = xs_v_t.cpu().numpy().astype(np.float32)
        ys_v = ys_v_t.cpu().numpy().astype(np.float32)
        z_v = z_v_sel_t.cpu().numpy().astype(np.float32)
        nx = nx_t.cpu().numpy(); ny = ny_t.cpu().numpy(); nz = nz_t.cpu().numpy()
        score_v = score_v_t.cpu().numpy().astype(np.float32)
        pts = pts_t.cpu().numpy().astype(np.float32)
        cols = cols_t.cpu().numpy().astype(np.float32)
    elif use_gpu:
        # GPU port of the finite-diff normals + grazing rejection + backproject
        # (gpu_frame_prep alone, gpu_resident off). Algebra is identical to the
        # numpy branch (central diffs, normalize, nz > cos(max_view_angle));
        # finite-diff is used (NOT kornia) so the rejected set matches the
        # numpy path within float32 tolerance. Unlike the gpu_resident branch
        # above, this compacts to the valid subset FIRST (re-deriving valid_t/
        # score_grid_t from numpy, since M1 only fuses the mask when
        # gpu_resident is also on) and THEN computes grazing -- two separate
        # compactions, kept exactly as before M2 for this non-resident path.
        from eligsir.utils.rgbd_frames import backproject_depth_pixels_torch
        valid_t = torch.from_numpy(np.ascontiguousarray(valid)).to(device=_gpu_device)
        score_grid_t = torch.from_numpy(np.ascontiguousarray(score_grid)).to(
            device=_gpu_device, dtype=torch.float32
        )
        xs_i_all = xs_t.to(torch.int64)
        ys_i_all = ys_t.to(torch.int64)
        xs_vi_t = xs_i_all[valid_t]
        ys_vi_t = ys_i_all[valid_t]
        z_vi_t = z_v_t[valid_t]
        score_v_t = score_grid_t[valid_t]

        cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
        _normals_backend = str(getattr(args, "gpu_frame_prep_normals", "finite_diff"))
        if _normals_backend == "kornia":
            # kornia.depth_to_normals on the full depth map, sampled at the valid
            # strided grid. Full-res Sobel gradient (not strided central diff), so
            # the rejected set differs from finite_diff by design (the A/B target).
            _nmap = _kornia_normal_map(z_t, d_fx, d_fy, d_cx, d_cy)  # [3, H, W]
            nx_t = _nmap[0][ys_vi_t, xs_vi_t]
            ny_t = _nmap[1][ys_vi_t, xs_vi_t]
            nz_t = _nmap[2][ys_vi_t, xs_vi_t]
            # kornia's normal sign is gradient-order dependent; flip to the same
            # +z hemisphere the finite_diff path uses so downstream surfel
            # quaternions and the grazing test agree.
            _flip = nz_t < 0
            nx_t = torch.where(_flip, -nx_t, nx_t)
            ny_t = torch.where(_flip, -ny_t, ny_t)
            nz_t = torch.where(_flip, -nz_t, nz_t)
            angle_ok_t = nz_t > cos_thresh
        else:
            xp = torch.clamp(xs_vi_t + s, 0, w - 1)
            xm = torch.clamp(xs_vi_t - s, 0, w - 1)
            yp = torch.clamp(ys_vi_t + s, 0, h - 1)
            ym = torch.clamp(ys_vi_t - s, 0, h - 1)
            dzdx_v_t = z_t[ys_vi_t, xp] - z_t[ys_vi_t, xm]
            dzdy_v_t = z_t[yp, xs_vi_t] - z_t[ym, xs_vi_t]

            denom_x, denom_y = finite_diff_normal_denominators(d_fx, d_fy, s)
            nx_t = -dzdx_v_t / denom_x
            ny_t = -dzdy_v_t / denom_y
            nz_t = torch.ones_like(nx_t)
            norm_t = torch.sqrt(nx_t * nx_t + ny_t * ny_t + nz_t * nz_t) + 1e-8
            nx_t = nx_t / norm_t; ny_t = ny_t / norm_t; nz_t = nz_t / norm_t

            angle_ok_t = nz_t > cos_thresh

        # Apply the grazing mask first (index_select stays on GPU); only then do
        # the single early-exit sync the numpy path also performs. Debug stats are
        # gated behind --streaming_insertion_debug so they add no sync when off.
        xs_v_t = xs_vi_t[angle_ok_t].to(torch.float32)
        ys_v_t = ys_vi_t[angle_ok_t].to(torch.float32)
        z_v_sel_t = z_vi_t[angle_ok_t].to(torch.float32)
        nx_t = nx_t[angle_ok_t]; ny_t = ny_t[angle_ok_t]; nz_t = nz_t[angle_ok_t]
        score_v_t = score_v_t[angle_ok_t]

        if _debug:
            _stats["after_grazing_filter"] = int(xs_v_t.shape[0])
        if xs_v_t.shape[0] == 0:
            return ({} if _collect_only else 0), _stats

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            device=_gpu_device, dtype=torch.float32
        )
        c2w_t = torch.from_numpy(np.ascontiguousarray(frame.c2w)).to(
            device=_gpu_device, dtype=torch.float32
        )
        pts_t, cols_t = backproject_depth_pixels_torch(
            z_v_sel_t, xs_v_t, ys_v_t, d_fx, d_fy, d_cx, d_cy, c2w_t, rgb_t, (h, w)
        )

        # Cross once to CPU/numpy; downstream (FOV align, voxel occupancy, KNN
        # dedup) is byte-identical to the numpy path.
        xs_v = xs_v_t.cpu().numpy().astype(np.float32)
        ys_v = ys_v_t.cpu().numpy().astype(np.float32)
        z_v = z_v_sel_t.cpu().numpy().astype(np.float32)
        nx = nx_t.cpu().numpy(); ny = ny_t.cpu().numpy(); nz = nz_t.cpu().numpy()
        score_v = score_v_t.cpu().numpy().astype(np.float32)
        pts = pts_t.cpu().numpy().astype(np.float32)
        cols = cols_t.cpu().numpy().astype(np.float32)
    else:
        xs_vi = xs[valid].astype(np.int32)
        ys_vi = ys[valid].astype(np.int32)
        z_vi = z_v[valid]
        score_v = score_grid[valid].astype(np.float32)

        dzdx_v = (z[ys_vi, np.clip(xs_vi + s, 0, w - 1)] - z[ys_vi, np.clip(xs_vi - s, 0, w - 1)])
        dzdy_v = (z[np.clip(ys_vi + s, 0, h - 1), xs_vi] - z[np.clip(ys_vi - s, 0, h - 1), xs_vi])

        # ---- Grazing-angle rejection -----------------------------------------
        _denom_x, _denom_y = finite_diff_normal_denominators(d_fx, d_fy, s)
        nx = -dzdx_v / _denom_x
        ny = -dzdy_v / _denom_y
        nz = np.ones_like(nx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= norm; ny /= norm; nz /= norm

        cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
        angle_ok = nz > cos_thresh

        if _debug:
            _stats["after_grazing_filter"] = int(angle_ok.sum())
        if not angle_ok.any():
            return ({} if _collect_only else 0), _stats

        xs_v = xs_vi[angle_ok].astype(np.float32)
        ys_v = ys_vi[angle_ok].astype(np.float32)
        z_v = z_vi[angle_ok].astype(np.float32)
        nx = nx[angle_ok]; ny = ny[angle_ok]; nz = nz[angle_ok]
        score_v = score_v[angle_ok]

        from eligsir.utils.rgbd_frames import backproject_depth_pixels
        pts, cols = backproject_depth_pixels(
            z_v, xs_v, ys_v, d_fx, d_fy, d_cx, d_cy, frame.c2w, rgb, (h, w))

    # ---- Color FOV alignment (when depth and color cameras differ in FOV) ----
    # After scaling depth intrinsics to resized resolution, 3D positions are correct
    # but the depth image may cover a wider FOV than the color image. Re-project each
    # 3D point onto the color camera to get correct RGB and filter out-of-FOV points.
    if frame.depth_fx is not None and (_depth_native_w != _tgt_w or _depth_native_h != _tgt_h):
        x_over_z = (xs_v - d_cx) / d_fx
        y_over_z = (ys_v - d_cy) / d_fy
        u_color = (x_over_z * frame.fx + frame.cx).round().astype(np.int32)
        v_color = (y_over_z * frame.fy + frame.cy).round().astype(np.int32)
        in_fov = (u_color >= 0) & (u_color < frame.width) & (v_color >= 0) & (v_color < frame.height)
        if not in_fov.all():
            pts = pts[in_fov]; z_v = z_v[in_fov]; score_v = score_v[in_fov]
            nx = nx[in_fov]; ny = ny[in_fov]; nz = nz[in_fov]
            u_color = u_color[in_fov]; v_color = v_color[in_fov]
        rgb_h, rgb_w = rgb.shape[:2]
        uc = np.clip(u_color, 0, rgb_w - 1)
        vc = np.clip(v_color, 0, rgb_h - 1)
        cols = rgb[vc, uc].astype(np.float32)

    normals_cam_pre = np.stack([nx, ny, nz], axis=1)
    normals_world_pre = (frame.c2w[:3, :3] @ normals_cam_pre.T).T

    # ---- Persistent Voxel coverage filter (Step 7) --------------------------
    occupied = streaming_scene.check_occupancy(pts, cover_voxel, check_neighbors=True)
    if bool(getattr(args, "streaming_surface_cell_dedup", False)):
        occupied = occupied | streaming_scene.check_surface_cells(pts, normals_world_pre)
    pts = pts[~occupied]
    cols = cols[~occupied]
    nx = nx[~occupied]; ny = ny[~occupied]; nz = nz[~occupied]
    normals_world_pre = normals_world_pre[~occupied]
    z_v = z_v[~occupied]
    score_v = score_v[~occupied]
    if _debug:
        _stats["after_voxel_occupancy"] = pts.shape[0]

    if pts.shape[0] == 0:
        return ({} if _collect_only else 0), _stats

    # Downsample remaining
    _fusion_on = bool(getattr(args, "streaming_insert_surface_fusion", False))
    _pts_pre = pts if _fusion_on else None
    pts, indices = _voxel_downsample_indices(pts, voxel_size)
    cols = cols[indices]
    nx = nx[indices]; ny = ny[indices]; nz = nz[indices]
    normals_world_pre = normals_world_pre[indices]
    z_v = z_v[indices]
    score_v = score_v[indices]

    # ---- LEVER 2: KNN surface-fusion of the voxel-dedup representatives -------
    # Replace each kept representative's position with the mean of its K nearest
    # pre-downsample neighbours (denoise depth jitter) and its normal with the
    # PCA-plane normal (oriented to the finite-diff normal). Fail-open: any error
    # leaves the plain voxel-dedup result untouched. Default OFF => byte-identical.
    if _fusion_on and _pts_pre is not None and pts.shape[0] > 0 and _pts_pre.shape[0] >= 2:
        try:
            from eligsir.utils.rocm_knn_fallback import knn_indices as _knn_indices
            from geometry.insertion import fuse_points_with_knn
            _fk = int(getattr(args, "streaming_insert_surface_fusion_k", 8))
            _fk = max(1, min(_fk, _pts_pre.shape[0]))
            # Edge-aware radius: only fuse neighbours on the SAME surface. Neighbours
            # farther than this (across a depth edge) are dropped so averaging doesn't
            # smear edge geometry off-surface (the cause of the initial fusion PSNR
            # regression). Default = 2x the insert voxel size.
            _vox = float(getattr(args, "streaming_insert_voxel_size", 0.02))
            _fmr = float(getattr(args, "streaming_insert_surface_fusion_max_radius_mult", 2.0))
            _max_radius = _fmr * _vox if _fmr > 0 else None
            _dev = gaussians.get_xyz.device
            _q = torch.from_numpy(np.ascontiguousarray(pts)).to(_dev, torch.float32)
            _r = torch.from_numpy(np.ascontiguousarray(_pts_pre)).to(_dev, torch.float32)
            _, _nn = _knn_indices(_q, _r, k=_fk, exclude_self=False)
            _nn = _nn.clamp_min(0).cpu().numpy()
            # degenerate_mask_out (issue #1149): a rank-deficient neighbourhood
            # (too few distinct points / near-collinear) makes the PCA normal
            # numerically ill-defined, so fuse_points_with_knn falls back to
            # normal_hints for that query instead of a spurious PCA estimate.
            # Never silently dropped — counted below so it's visible in _debug
            # stats, matching this function's rejection-counter convention.
            _degen_mask = np.zeros(pts.shape[0], dtype=bool)
            _fused_pts, _fused_normals = fuse_points_with_knn(
                pts, _pts_pre, _nn, normals_world_pre, max_radius=_max_radius,
                degenerate_mask_out=_degen_mask,
            )
            pts = _fused_pts.astype(pts.dtype, copy=False)
            normals_world_pre = _fused_normals.astype(normals_world_pre.dtype, copy=False)
            if _debug:
                _stats["surface_fusion_degenerate_normals"] = int(_degen_mask.sum())
                _stats["surface_fusion_candidates"] = int(_degen_mask.shape[0])
        except Exception:
            pass  # fail-open: keep the plain voxel-dedup representatives

    # ---- KNN insertion dedup (Component E) ----------------------------------
    # Reject candidates whose nearest existing Gaussian is within depth*radius_factor.
    # Uses torch.cdist; subsamples existing set if too large for memory.
    use_knn_dedup = getattr(args, "streaming_insert_knn_dedup", False)
    if use_knn_dedup and pts.shape[0] > 0:
        n_existing = gaussians.get_xyz.shape[0]
        if n_existing > 0:
            knn_radius_factor = float(getattr(args, "streaming_insert_knn_radius_factor", 0.005))
            max_existing = int(getattr(args, "streaming_insert_knn_max_existing", 200_000))
            normal_aware = bool(getattr(args, "streaming_insert_knn_dedup_normal_aware", False))
            _dev = gaussians.get_xyz.device
            with torch.no_grad():
                cand_t = torch.from_numpy(pts).to(device=_dev, dtype=torch.float32)
                exist_t = gaussians.get_xyz.detach()
                exist_q = exist_s = None
                if normal_aware:
                    exist_q = gaussians.get_rotation.detach()
                    exist_s = gaussians.get_scaling.detach()
                if n_existing > max_existing:
                    # Subsample via uniform stride to keep memory manageable
                    stride = max(1, n_existing // max_existing)
                    exist_t = exist_t[::stride]
                    if normal_aware:
                        exist_q = exist_q[::stride]; exist_s = exist_s[::stride]
                # Exact cross-set 1-NN via the simple_knn knnIndices kernel (no Nq x Nr
                # matrix; falls back to chunked cdist when the extension is absent).
                from eligsir.utils.rocm_knn_fallback import knn_indices
                dist2, nn_idx = knn_indices(cand_t, exist_t, k=1, exclude_self=False)
                min_dists = dist2[:, 0].clamp_min(0).sqrt()
                per_pt_radius = torch.from_numpy(z_v * knn_radius_factor).to(device=_dev, dtype=torch.float32)
                too_close = min_dists < per_pt_radius
                if normal_aware and too_close.any():
                    # Only treat a near candidate as a duplicate if the nearest existing
                    # splat faces the SAME way — keeps layered/opposite-facing surfaces
                    # (thin walls, front/back) that pure-distance dedup would false-merge.
                    cos_thresh = float(getattr(args, "streaming_insert_knn_dedup_normal_cos", 0.85))
                    nn0 = nn_idx[:, 0].clamp_min(0)
                    exist_n = _gaussian_min_axis_normals(exist_q[nn0], exist_s[nn0])  # [Ncand,3]
                    cand_n = torch.from_numpy(np.ascontiguousarray(normals_world_pre)).to(
                        device=_dev, dtype=torch.float32)
                    cand_n = cand_n / cand_n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    aligned = (cand_n * exist_n).sum(dim=-1).abs() > cos_thresh
                    too_close = too_close & aligned
                keep = ~too_close
            if keep.any():
                keep_np = keep.cpu().numpy()
                pts = pts[keep_np]; cols = cols[keep_np]
                nx = nx[keep_np]; ny = ny[keep_np]; nz = nz[keep_np]
                normals_world_pre = normals_world_pre[keep_np]
                z_v = z_v[keep_np]
                score_v = score_v[keep_np]
                if _debug:
                    _stats["after_knn_dedup"] = pts.shape[0]
            else:
                return ({} if _collect_only else 0), _stats

    if max_new > 0 and pts.shape[0] > max_new:
        if str(getattr(args, "streaming_insert_score_mode", "boolean_or")) == "weighted_topk":
            idx = np.argpartition(-score_v, max_new - 1)[:max_new]
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(pts.shape[0], size=max_new, replace=False)
        pts, cols, nx, ny, nz, z_v = pts[idx], cols[idx], nx[idx], ny[idx], nz[idx], z_v[idx]
        normals_world_pre = normals_world_pre[idx]
        score_v = score_v[idx]

    # ---- Surface-aligned initialisation (Step 6) ----------------------------
    # Align Gaussians to surface normals
    normals_world = normals_world_pre
    q_world = _quaternion_from_normal(normals_world)
    geom_confidence = np.clip(score_v.astype(np.float32), 0.0, 1.0)
    if not np.any(geom_confidence > 0):
        geom_confidence = np.ones((pts.shape[0],), dtype=np.float32)
    geom_source = np.full((pts.shape[0],), int(GeometrySource.SENSOR), dtype=np.int16)

    # Scales: depth-derived per-axis pixel footprint in world space, with
    # an optional multiplier and hard upper clamp to avoid over-large splats.
    scale_mult  = getattr(args, "streaming_insert_scale_mult", 0.5)
    scale_max   = getattr(args, "streaming_insert_scale_max", 0.05)
    normal_ratio = getattr(args, "streaming_insert_normal_scale_ratio", 0.15)
    tx = np.clip(scale_mult * z_v / d_fx * depth_stride, 1e-4, scale_max)
    ty = np.clip(scale_mult * z_v / d_fy * depth_stride, 1e-4, scale_max)
    # Phase B: optional KNN-based per-frame scale (self-KNN over this frame's
    # candidate batch via the fast simple_knn distCUDA2) instead of the analytic
    # depth-footprint above. footprint=default; knn replaces the in-plane footprint
    # with the local-density spacing; hybrid takes their geometric mean. The result
    # flows through the same isotropic/anisotropic/parallax logic below. Per admitted
    # frame (not per-iteration), N<=max_new, so the KNN cost is a few ms.
    _scale_mode = str(args.streaming_insert_scale_mode)
    if _scale_mode in ("knn", "hybrid") and pts.shape[0] > 8:
        try:
            from eligsir.utils.rocm_knn_fallback import distCUDA2
            _pc = torch.from_numpy(np.ascontiguousarray(pts)).to(gaussians.get_xyz.device, torch.float32)
            _knn_iso = np.clip(
                distCUDA2(_pc).clamp_min(1e-12).sqrt().cpu().numpy() * float(scale_mult),
                1e-4, scale_max,
            ).astype(np.float32)
            if _scale_mode == "knn":
                tx = _knn_iso; ty = _knn_iso
            else:  # hybrid: geometric mean of footprint and KNN spacing
                tx = np.clip(np.sqrt(tx * _knn_iso), 1e-4, scale_max)
                ty = np.clip(np.sqrt(ty * _knn_iso), 1e-4, scale_max)
        except Exception:
            pass  # fall back to the footprint tx/ty already computed
    # ---- LEVER 1: explicit distance-aware in-plane scale (default OFF) --------
    # Enlarge far Gaussians (z/f footprint) so they tile the surface without
    # holes, keep near ones sharp. Composed via element-wise MAX with the
    # active footprint/knn/hybrid scale: never shrinks below local density, and
    # restores far-tiling when scale_mode=knn replaced the depth footprint.
    if getattr(args, "streaming_insert_distance_scale", False):
        from geometry.insertion import distance_aware_inplane_scale
        _dcoeff = float(getattr(args, "streaming_insert_distance_scale_coeff", 1.0))
        _dx, _dy = distance_aware_inplane_scale(
            z_v, d_fx, d_fy, coeff=_dcoeff, scale_max=scale_max, scale_min=1e-4,
        )
        tx = np.maximum(tx, _dx)
        ty = np.maximum(ty, _dy)
    # ---- D2: parallax-based insertion uncertainty (locational cloud) ---------
    # A depth point from a single view is uncertain ALONG the ray; first-order
    # stereo depth error grows as δz ∝ z²/(f·b) (b = trajectory baseline). When
    # the camera has little parallax (stalled), seed an elongated along-normal
    # cloud so multi-view consensus (Part C) can compress it onto the surface;
    # with ample baseline the cloud stays thin. Gated, off by default.
    tz_parallax = None
    if getattr(args, "streaming_insert_parallax_uncertainty", False):
        _bn = int(getattr(args, "streaming_insert_parallax_baseline_frames", 5))
        _baseline = float(streaming_scene.recent_camera_baseline(_bn))
        _coeff = float(getattr(args, "streaming_insert_parallax_coeff", 1.0))
        # δz ∝ z²/(f·b); use d_fx as focal (px). Guard tiny baselines so a stall
        # yields a large-but-bounded cloud rather than a divide-by-zero spike.
        _sigma_z = _coeff * (z_v.astype(np.float32) ** 2) / (d_fx * max(_baseline, 1e-3))
        tz_parallax = np.clip(_sigma_z, 1e-5, scale_max)

    if getattr(args, "streaming_insert_isotropic_scale", False):
        # Isotropic sphere (H3): geometric mean of in-plane footprint.
        # Avoids edge-on streaking from flat surfels seen off-axis.
        t_iso = np.sqrt(tx * ty)
        log_scales = np.log(np.stack([t_iso, t_iso, t_iso], axis=1))
    else:
        tz = np.clip(normal_ratio * np.minimum(tx, ty), 1e-5, scale_max * normal_ratio)
        log_scales = np.log(np.stack([tx, ty, tz], axis=1))

    if _debug:
        _stats["after_per_frame_cap"] = pts.shape[0]

    # ---- #21: depth-quality-aware seed opacity --------------------------------
    # Penalize points that are far away (Orbbec/Kinect depth noise grows ~z²) or
    # seen at a grazing angle. Seed their opacity proportional to that confidence
    # so uncertain far-range depth must EARN its place photometrically instead of
    # starting as a solid floater. The confidence is also folded into the geometry
    # confidence for source-aware pruning. Per-point arrays available here are
    # z_v (range) and nz (surface-normal-z, grazing proxy); both are reindexed by
    # the FOV/occupancy/KNN filters above. cos_thresh and max_depth are bound on
    # every path reaching this point. Gated, off by default.
    opacities_raw_np = None
    if getattr(args, "streaming_insert_confidence_opacity", False) and pts.shape[0] > 0:
        from eligsir.mapping.scheduler.frame_quality import depth_confidence_opacity
        _c_range = np.clip(1.0 - z_v / max(max_depth, 1e-6), 0.3, 1.0)
        _c_angle = np.clip((nz - cos_thresh) / max(1.0 - cos_thresh, 1e-6), 0.3, 1.0)
        depth_conf = (_c_range * _c_angle).astype(np.float32)
        geom_confidence = np.clip(geom_confidence * depth_conf, 0.0, 1.0)
        _op_floor = float(getattr(args, "streaming_insert_confidence_opacity_floor", 0.3))
        opacities_raw_np = depth_confidence_opacity(depth_conf, init_opacity, _op_floor)

    if _collect_only:
        # Return candidates without committing — caller will batch-flush.
        candidates = {
            "pts": pts,
            "cols": cols,
            "log_scales": log_scales,
            "q_world": q_world,
            "normals": normals_world.astype(np.float32),
            "confidence": geom_confidence.astype(np.float32),
            "source": geom_source,
            "birth_frame": current_frame_idx,
            "init_opacity": init_opacity,
            "opacities_raw": opacities_raw_np,
        }
        if _debug:
            _stats["collected"] = pts.shape[0]
        return candidates, _stats

    # PR #1142 hard cut: there is no direct population writer left here. The
    # planner never calls ``add_points_as_gaussians``; every insertion goes
    # through a backend mutation transaction. When no backend was supplied
    # (small isolated model tests that do not build a runtime), the same
    # transaction-only ephemeral backend the insertion committer uses is
    # constructed here, so the write path is identical either way.
    from eligsir.map.mutation import GaussianInsertionBatch
    from eligsir.mapping.integration.insertion_committer import (
        _ephemeral_backend_for_tests,
    )

    _backend = map_backend if map_backend is not None else _ephemeral_backend_for_tests(gaussians)
    _txn = _backend.begin_mutation()
    added = _txn.insert(GaussianInsertionBatch(
        points=torch.from_numpy(pts),
        colors=torch.from_numpy(cols),
        init_opacity=init_opacity,
        scales=torch.from_numpy(log_scales),
        rotations=torch.from_numpy(q_world),
        normals=torch.from_numpy(normals_world.astype(np.float32)),
        confidence=torch.from_numpy(geom_confidence.astype(np.float32)),
        source=torch.from_numpy(geom_source),
        opacities_raw=(torch.from_numpy(opacities_raw_np) if opacities_raw_np is not None else None),
        is_provisional=True,
        birth_frame=current_frame_idx,
        birth_iter=iteration,
    ))
    _backend.commit_mutation(_txn)
    if added > 0:
        streaming_scene.add_to_occupancy_hash(gaussians.get_xyz[-added:])
        if bool(getattr(args, "streaming_surface_cell_dedup", False)):
            streaming_scene.add_to_surface_cells(gaussians.get_xyz[-added:], gaussians.anchor_normal[-added:])
    if _debug:
        _stats["inserted"] = added
    return added, _stats
