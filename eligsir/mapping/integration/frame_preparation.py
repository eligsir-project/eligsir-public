"""Per-frame preparation, bootstrap diagnostics, and streaming snapshot helpers."""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image

from eligsir.io.snapshots import (
    snapshot_gaussians_for_ply as _snapshot_for_ply,
    subset_ply_snapshot as _subset_snapshot,
    write_ply_from_snapshot as _write_ply,
)
from eligsir.mapping.integration.depth_helpers import _get_sensor_depth
from eligsir.runtime.save_worker import AsyncSaveWorker
from eligsir.utils.score_contract import (
    apply_declared_score_clamp_pair,
    scientific_psnr,
    score_tensor_from_camera,
)
from eligsir.utils.streaming_depth_lock import (
    depth_conflict_enabled,
    depth_lock_candidate_mask,
    depth_lock_conflict_threshold,
    ensure_depth_lock_buffers,
    record_depth_conflicts,
    record_depth_support_views,
)


def _voxel_downsample_indices(*args, **kwargs):
    """Lazy delegate to the canonical training helper."""
    from train_streaming import _voxel_downsample_indices as _f

    return _f(*args, **kwargs)


def _timed(name):
    """Delegate to the training process's shared phase-timing registry."""
    from train_streaming import _timed as _cls

    return _cls(name)


def _bootstrap_mask(gaussians) -> Optional[torch.Tensor]:
    birth_frame = getattr(gaussians, "birth_frame", None)
    if birth_frame is None:
        return None
    if birth_frame.shape[0] != gaussians.get_xyz.shape[0]:
        return None
    mask = birth_frame == 0
    return mask if mask.any() else None


def _capture_bootstrap_positions(gaussians) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    mask = _bootstrap_mask(gaussians)
    if mask is None:
        return None
    return mask.detach().clone(), gaussians.get_xyz[mask].detach().clone()


def _bootstrap_motion_stats(
    gaussians,
    *,
    before: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    threshold: float = 0.05,
) -> Optional[dict]:
    with torch.no_grad():
        if before is not None:
            mask, ref_xyz = before
            if mask.shape[0] != gaussians.get_xyz.shape[0] or not mask.any():
                return {
                    "n": 0,
                    "mean": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                    "count_gt_2cm": 0,
                    "count_gt_5cm": 0,
                    "count_gt_10cm": 0,
                    "count_gt_threshold": 0,
                    "shape_mismatch": 1,
                }
            cur_xyz = gaussians.get_xyz[mask]
            if cur_xyz.shape != ref_xyz.shape:
                return None
            delta = torch.linalg.norm(cur_xyz - ref_xyz, dim=1)
        else:
            mask = _bootstrap_mask(gaussians)
            anchor_xyz = getattr(gaussians, "anchor_xyz", None)
            if mask is None or anchor_xyz is None or anchor_xyz.shape[0] != gaussians.get_xyz.shape[0]:
                return None
            delta = torch.linalg.norm(gaussians.get_xyz[mask] - anchor_xyz[mask], dim=1)
        if delta.numel() == 0:
            return None
        return {
            "n": int(delta.numel()),
            "mean": float(delta.mean().item()),
            "p95": float(torch.quantile(delta.float(), 0.95).item()),
            "max": float(delta.max().item()),
            "count_gt_2cm": int((delta > 0.02).sum().item()),
            "count_gt_5cm": int((delta > 0.05).sum().item()),
            "count_gt_10cm": int((delta > 0.10).sum().item()),
            "count_gt_threshold": int((delta > float(threshold)).sum().item()),
            "shape_mismatch": 0,
        }


def _log_bootstrap_motion(
    tb_writer,
    args,
    iteration: int,
    stage: str,
    stats: Optional[dict],
) -> None:
    if not stats:
        return
    prefix = f"streaming/bootstrap_motion/{stage}"
    if tb_writer is not None:
        for key, value in stats.items():
            tb_writer.add_scalar(f"{prefix}/{key}", float(value), iteration)
    threshold = float(getattr(args, "streaming_debug_bootstrap_motion_threshold", 0.05))
    should_print = (
        bool(getattr(args, "streaming_debug_bootstrap_motion", False))
        and (stats.get("count_gt_threshold", 0) > 0 or stats.get("max", 0.0) > threshold)
    )
    if should_print:
        print(
            f"[bootstrap-motion] step={iteration} stage={stage} "
            f"n={stats.get('n', 0)} mean={stats.get('mean', 0.0):.5f}m "
            f"p95={stats.get('p95', 0.0):.5f}m max={stats.get('max', 0.0):.5f}m "
            f">2cm={stats.get('count_gt_2cm', 0)} >5cm={stats.get('count_gt_5cm', 0)} "
            f">10cm={stats.get('count_gt_10cm', 0)}",
            flush=True,
        )


def _write_bootstrap_motion_ply(gaussians, args, iteration: int, label: str) -> None:
    if not getattr(args, "streaming_debug_bootstrap_motion", False):
        return
    mask = _bootstrap_mask(gaussians)
    if mask is None:
        return
    try:
        snap = _snapshot_for_ply(gaussians)
        boot_snap = _subset_snapshot(snap, mask)
        out_path = os.path.join(
            args.model_path,
            "bootstrap_motion",
            f"iter_{int(iteration):06d}_{label}_bootstrap.ply",
        )
        _write_ply(boot_snap, out_path)
    except Exception as exc:
        print(f"[bootstrap-motion] failed to write PLY step={iteration} label={label}: {exc}", flush=True)


def _render_streaming_snapshot(
    gaussians, streaming_scene, render_fn, pipe, background, args, n_frames, tb_writer,
    save_worker: Optional[AsyncSaveWorker] = None,
):
    """Render train/test views at streaming frame-save milestones."""
    train_cams = streaming_scene.get_local_cameras()
    test_cams = streaming_scene.getTestCameras()
    if not train_cams and not test_cams:
        return

    output_base = os.path.join(args.model_path, "streaming_renders", f"frame_{n_frames}")

    for split, cameras in [("train", train_cams), ("test", test_cams)]:
        if not cameras:
            continue
        render_dir = os.path.join(output_base, split, "renders")
        gt_dir = os.path.join(output_base, split, "gt")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        psnrs = []
        render_data = []
        clamp_records = []
        for cam in cameras:
            with torch.no_grad():
                raw_img = render_fn(cam, gaussians, pipe, background)["render"]
                raw_gt = score_tensor_from_camera(cam, device=gaussians.get_xyz.device)
                img, gt, clamp_record = apply_declared_score_clamp_pair(
                    raw_img,
                    raw_gt,
                    context=f"streaming_snapshot:{split}",
                )
                clamp_records.append(clamp_record)
            psnrs.append(
                scientific_psnr(
                    img,
                    gt,
                    context=f"streaming_snapshot:{split}",
                ).mean().item()
            )
            render_data.append((
                img.cpu(), gt.cpu(),
                os.path.join(render_dir, f"{cam.image_name}.png"),
                os.path.join(gt_dir, f"{cam.image_name}.png"),
            ))
        if psnrs:
            mean_psnr = float(np.mean(psnrs))
            changed = sum(r["render"]["samples_changed"] for r in clamp_records)
            total = sum(r["render"]["samples_total"] for r in clamp_records)
            clamp_summary = {
                "nonfinite_policy": "rejected before clamp",
                "samples_changed": changed,
                "samples_total": total,
                "fraction_changed": changed / max(1, total),
                "value_min": min(r["render"]["value_min"] for r in clamp_records),
                "value_max": max(r["render"]["value_max"] for r in clamp_records),
            }
            with open(os.path.join(output_base, split, "score_domain_clamp.json"), "w") as f:
                json.dump(clamp_summary, f, indent=2)
            print(
                f"[streaming-render] frame={n_frames} {split} PSNR={mean_psnr:.2f}dB ({len(psnrs)} views)",
                flush=True,
            )
            if tb_writer:
                tb_writer.add_scalar(f"streaming_eval/{split}_psnr", mean_psnr, n_frames)
                tb_writer.add_scalar(
                    f"streaming_eval/{split}_score_clamp_fraction",
                    clamp_summary["fraction_changed"],
                    n_frames,
                )
                tb_writer.add_scalar(
                    f"streaming_eval/{split}_score_raw_max",
                    clamp_summary["value_max"],
                    n_frames,
                )

        def _save_batch(data=render_data):
            for img_cpu, gt_cpu, r_path, g_path in data:
                save_image(img_cpu, r_path)
                save_image(gt_cpu, g_path)

        if save_worker is not None:
            with _timed("save_enqueue"):
                save_worker.enqueue(_save_batch)
        else:
            _save_batch()


def _update_provisional_support(gaussians, cam, render_pkg, args):
    """Project visible depth-evidence candidates without accelerator scalar reads."""
    visible = render_pkg["visibility_filter"]
    has_provisional = (
        hasattr(gaussians, "provisional")
        and gaussians.provisional.shape[0] == visible.shape[0]
    )
    track_conflicts = depth_conflict_enabled(args)
    if not has_provisional and not track_conflicts:
        return

    to_check = torch.zeros_like(visible, dtype=torch.bool)
    if has_provisional:
        to_check |= gaussians.provisional & visible

    candidate_mask = None
    if track_conflicts:
        ensure_depth_lock_buffers(gaussians)
        candidate_mask = depth_lock_candidate_mask(gaussians, args)
        if candidate_mask.shape[0] == visible.shape[0]:
            to_check |= candidate_mask & visible

    sensor_depth = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    if sensor_depth is None:
        return

    device = gaussians.get_xyz.device
    sensor_d = (
        sensor_depth
        if sensor_depth.device == device
        else sensor_depth.to(device=device, non_blocking=True)
    )
    if sensor_d.shape[-2:] != (cam.image_height, cam.image_width):
        raise RuntimeError(
            f"sensor depth shape {tuple(sensor_d.shape)} does not match camera image grid "
            f"({cam.image_height}, {cam.image_width})"
        )

    xyz_v = gaussians.get_xyz[to_check]
    pts_h = torch.cat(
        [xyz_v, torch.ones((xyz_v.shape[0], 1), device=device)], dim=1
    )
    w2c = cam.world_view_transform.transpose(0, 1)
    pts_cam = pts_h @ w2c.T
    z_p = pts_cam[:, 2]

    width, height = cam.image_width, cam.image_height
    fx = getattr(cam, "fx", width / (2.0 * math.tan(cam.FoVx / 2.0)))
    fy = getattr(cam, "fy", height / (2.0 * math.tan(cam.FoVy / 2.0)))
    cx = getattr(cam, "cx", width / 2.0)
    cy = getattr(cam, "cy", height / 2.0)
    u = (pts_cam[:, 0] / torch.clamp(z_p, min=1e-6)) * fx + cx
    v = (pts_cam[:, 1] / torch.clamp(z_p, min=1e-6)) * fy + cy
    ui, vi = u.round().long(), v.round().long()

    valid_px = (
        (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
        & (z_p > 0)
    )
    z_lookup = sensor_d[0, vi[valid_px], ui[valid_px]]
    depth_error = torch.abs(z_p[valid_px] - z_lookup)
    sensor_valid = z_lookup > 0
    consistent = sensor_valid & (
        depth_error < float(args.streaming_depth_consistency_thresh)
    )
    indices_all = torch.where(to_check)[0][valid_px]

    if has_provisional:
        support_indices = indices_all[
            consistent & gaussians.provisional[indices_all]
        ]
        gaussians.support_count[support_indices] += 1

    if (
        track_conflicts
        and candidate_mask is not None
        and candidate_mask.shape[0] == visible.shape[0]
    ):
        frame = getattr(cam, "_streaming_frame", None)
        view_uid = int(getattr(frame, "index", getattr(cam, "uid", 0)))
        candidate_checked = candidate_mask[indices_all]
        record_depth_support_views(
            gaussians,
            indices_all[consistent & candidate_checked],
            view_uid,
        )
        conflicts = (
            sensor_valid
            & candidate_checked
            & (
                z_p[valid_px]
                < (z_lookup - depth_lock_conflict_threshold(args))
            )
        )
        record_depth_conflicts(gaussians, indices_all[conflicts], view_uid)
