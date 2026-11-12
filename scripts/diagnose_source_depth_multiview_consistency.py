"""Task #100 diagnostic: multi-view consistency of source-depth XYZ at birth.

Answers a direct H1 test (docs/architecture/2026-08-10-debt-controlled-sparse-assimilation-plan.md
section 4.5's revised experiment priority, step 2): is the RAW SENSOR DEPTH DATA itself
geometrically self-consistent across views, independent of anything the trained Gaussian
model does with it? This bypasses the trained model entirely and tests the dataset's own
depth+pose data.

Reuses the EXACT production tri-state consistency/conflict formula from
eligsir/mapping/integration/frame_preparation.py (the same code path used at training
time to populate depth_conflict_count/support_count) rather than reimplementing a
different check -- so this diagnostic's verdict is directly comparable to what the
production pipeline itself would have measured, had --streaming_depth_multiview_conflict_enabled
been on during a real run:

  - SUPPORTED (this frame): sensor_valid AND |z_projected - z_sensor| < consistency_thresh
  - CONFLICT (this frame): sensor_valid AND z_projected < (z_sensor - conflict_thresh)
    ("conservative proof of wrong geometry: the point is in front of the measured surface,
    i.e. valid depth says the intervening space is empty" -- frame_preparation.py's own
    comment on this exact formula).
  - Otherwise: no information from this frame (out of view, or sensor invalid there).

For each SOURCE frame, backprojects a stride-sampled grid of valid, non-edge depth pixels to
world XYZ, then checks each point against a WINDOW of SUBSEQUENT frames only (matching the
"project into subsequent nearby frames" framing in the plan doc -- at insertion time only
future observations are available). A point is classified CERTIFIED_CONTRADICTION if at
least `--min_conflict_views` subsequent frames register a conflict, SUPPORTED if it has any
supporting frame and isn't a certified contradiction, else UNKNOWN.

No GPU/trained model needed -- pure geometry on the dataset's own depth+pose data. Run inside
the container (numpy/PIL only, no torch dependency beyond what's already installed):
  docker compose run --rm train python scripts/diagnose_source_depth_multiview_consistency.py \
    --source_path /data/tum_fr3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.readers.tum import _tum_default_intrinsics, readTUMCameras  # noqa: E402


def _load_depth(path, depth_rel, depth_scale):
    depth = np.array(Image.open(f"{path}/{depth_rel}")).astype(np.float32) / float(depth_scale)
    return depth


def _backproject_grid(depth, fx, fy, cx, cy, c2w, stride, min_depth, max_depth, edge_thresh):
    """Return world-space XYZ [N,3] + source pixel (u,v) [N,2] for valid,
    non-edge depth pixels on a stride-sampled grid."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)

    if edge_thresh > 0:
        # Simple depth-edge exclusion: compare each sampled pixel's depth
        # against its stride-step neighbors; reject if any neighbor differs
        # by more than edge_thresh (matches this codebase's
        # streaming_depth_edge_threshold convention -- edge pixels are a
        # known confound for depth consistency checks).
        pad_z = np.pad(z, 1, mode="edge")
        dz_right = np.abs(pad_z[1:-1, 2:] - z)
        dz_left = np.abs(pad_z[1:-1, :-2] - z)
        dz_down = np.abs(pad_z[2:, 1:-1] - z)
        dz_up = np.abs(pad_z[:-2, 1:-1] - z)
        max_dz = np.maximum(np.maximum(dz_right, dz_left), np.maximum(dz_down, dz_up))
        valid &= max_dz < edge_thresh

    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    z_v = z[valid].astype(np.float32)

    x_cam = (xs_v - cx) / fx * z_v
    y_cam = (ys_v - cy) / fy * z_v
    pts_cam = np.stack([x_cam, y_cam, z_v, np.ones_like(z_v)], axis=1)  # [N, 4]
    pts_world = (c2w @ pts_cam.T).T[:, :3]

    return pts_world.astype(np.float32), np.stack([xs_v, ys_v], axis=1)


def _project_and_check(pts_world, w2c, fx, fy, cx, cy, width, height, sensor_depth,
                        consistency_thresh, conflict_thresh):
    """Vectorized version of frame_preparation.py's exact consistency/conflict
    formula, for a batch of world points against one target frame."""
    n = pts_world.shape[0]
    pts_h = np.concatenate([pts_world, np.ones((n, 1), dtype=np.float32)], axis=1)
    pts_cam = (w2c @ pts_h.T).T  # [N, 4]
    z_p = pts_cam[:, 2]

    u = (pts_cam[:, 0] / np.clip(z_p, 1e-6, None)) * fx + cx
    v = (pts_cam[:, 1] / np.clip(z_p, 1e-6, None)) * fy + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    in_frame = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & (z_p > 0)

    support = np.zeros(n, dtype=bool)
    conflict = np.zeros(n, dtype=bool)
    residual = np.full(n, np.nan, dtype=np.float32)

    if in_frame.any():
        idx = np.where(in_frame)[0]
        z_lookup = sensor_depth[vi[idx], ui[idx]]
        sensor_valid = z_lookup > 0
        abs_err = np.abs(z_p[idx] - z_lookup)
        consistent = sensor_valid & (abs_err < consistency_thresh)
        conflicted = sensor_valid & (z_p[idx] < (z_lookup - conflict_thresh))

        support[idx[consistent]] = True
        conflict[idx[conflicted]] = True
        valid_idx = idx[sensor_valid]
        residual[valid_idx] = abs_err[sensor_valid]

    return support, conflict, residual


def run(source_path, n_source_frames, subsequent_window, pixel_stride,
        min_depth, max_depth, consistency_thresh, conflict_thresh, edge_thresh,
        min_conflict_views):
    train_cams, _, assocs = readTUMCameras(source_path, frame_stride=1, max_frames=0, eval=False)
    fx, fy, cx, cy = _tum_default_intrinsics(source_path)
    n_total = len(assocs)
    print(f"[diagnose] {n_total} associated frames loaded from {source_path}")

    depth_scale = 5000.0  # matches this project's tum_depth_scale default

    # Precompute c2w/w2c for every frame up front (cheap, pure numpy).
    from scene.readers.tum import _tum_pose_to_c2w
    c2ws = []
    for a in assocs:
        c2w = _tum_pose_to_c2w(a["tvec"], a["qxyzw"])
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    w2cs = np.linalg.inv(c2ws)

    source_indices = np.linspace(0, n_total - subsequent_window - 1, n_source_frames).astype(int)
    source_indices = sorted(set(int(i) for i in source_indices if i >= 0))

    per_frame_results = []
    all_residuals_supported = []
    all_residuals_conflicted = []
    total_points = 0
    total_supported = 0
    total_contradicted = 0
    total_unknown = 0

    depth_cache = {}

    def get_depth(i):
        if i not in depth_cache:
            depth_cache[i] = _load_depth(source_path, assocs[i]["depth_rel"], depth_scale)
        return depth_cache[i]

    for src_i in source_indices:
        src_depth = get_depth(src_i)
        pts_world, src_px = _backproject_grid(
            src_depth, fx, fy, cx, cy, c2ws[src_i], pixel_stride, min_depth, max_depth, edge_thresh
        )
        if pts_world.shape[0] == 0:
            continue

        n_pts = pts_world.shape[0]
        support_count = np.zeros(n_pts, dtype=np.int32)
        conflict_count = np.zeros(n_pts, dtype=np.int32)
        residual_sum = np.zeros(n_pts, dtype=np.float64)
        residual_n = np.zeros(n_pts, dtype=np.int32)
        h, w = src_depth.shape

        target_indices = range(src_i + 1, min(src_i + 1 + subsequent_window, n_total))
        for tgt_i in target_indices:
            tgt_depth = get_depth(tgt_i)
            support, conflict, residual = _project_and_check(
                pts_world, w2cs[tgt_i], fx, fy, cx, cy, w, h, tgt_depth,
                consistency_thresh, conflict_thresh,
            )
            support_count += support.astype(np.int32)
            conflict_count += conflict.astype(np.int32)
            finite = np.isfinite(residual)
            residual_sum[finite] += residual[finite]
            residual_n[finite] += 1

        certified_contradiction = conflict_count >= min_conflict_views
        supported = (support_count >= 1) & (~certified_contradiction)
        unknown = (~certified_contradiction) & (~supported)

        mean_residual = np.divide(
            residual_sum, residual_n, out=np.full(n_pts, np.nan), where=residual_n > 0
        )
        all_residuals_supported.extend(mean_residual[supported][np.isfinite(mean_residual[supported])].tolist())
        all_residuals_conflicted.extend(mean_residual[certified_contradiction][np.isfinite(mean_residual[certified_contradiction])].tolist())

        frame_result = {
            "source_frame_idx": src_i,
            "trajectory_frac": src_i / max(1, n_total - 1),
            "n_points": int(n_pts),
            "n_supported": int(supported.sum()),
            "n_certified_contradiction": int(certified_contradiction.sum()),
            "n_unknown": int(unknown.sum()),
            "frac_supported": float(supported.mean()),
            "frac_contradiction": float(certified_contradiction.mean()),
            "frac_unknown": float(unknown.mean()),
        }
        per_frame_results.append(frame_result)
        total_points += n_pts
        total_supported += int(supported.sum())
        total_contradicted += int(certified_contradiction.sum())
        total_unknown += int(unknown.sum())
        print(f"[diagnose] source_frame={src_i} ({src_i/max(1,n_total-1)*100:.0f}% through trajectory): "
              f"n={n_pts} supported={frame_result['frac_supported']:.3f} "
              f"contradiction={frame_result['frac_contradiction']:.3f} "
              f"unknown={frame_result['frac_unknown']:.3f}")

    summary = {
        "n_source_frames": len(per_frame_results),
        "total_points": total_points,
        "frac_supported": total_supported / max(1, total_points),
        "frac_contradiction": total_contradicted / max(1, total_points),
        "frac_unknown": total_unknown / max(1, total_points),
        "mean_residual_supported_m": float(np.mean(all_residuals_supported)) if all_residuals_supported else None,
        "mean_residual_conflicted_m": float(np.mean(all_residuals_conflicted)) if all_residuals_conflicted else None,
        "per_frame": per_frame_results,
        "params": {
            "n_source_frames": n_source_frames, "subsequent_window": subsequent_window,
            "pixel_stride": pixel_stride, "min_depth": min_depth, "max_depth": max_depth,
            "consistency_thresh": consistency_thresh, "conflict_thresh": conflict_thresh,
            "edge_thresh": edge_thresh, "min_conflict_views": min_conflict_views,
        },
    }
    print()
    print(f"[diagnose] TOTAL: n={total_points} supported={summary['frac_supported']:.4f} "
          f"contradiction={summary['frac_contradiction']:.4f} unknown={summary['frac_unknown']:.4f}")
    print(f"[diagnose] mean residual (supported): {summary['mean_residual_supported_m']}")
    print(f"[diagnose] mean residual (conflicted, excess): {summary['mean_residual_conflicted_m']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--n_source_frames", type=int, default=20)
    parser.add_argument("--subsequent_window", type=int, default=30,
                         help="How many frames AFTER the source frame to check against.")
    parser.add_argument("--pixel_stride", type=int, default=8,
                         help="Matches this project's streaming_depth_stride default.")
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--consistency_thresh", type=float, default=0.05,
                         help="Matches streaming_depth_consistency_thresh default.")
    parser.add_argument("--conflict_thresh", type=float, default=0.05,
                         help="Matches depth_lock_conflict_threshold's fallback default.")
    parser.add_argument("--edge_thresh", type=float, default=0.02,
                         help="Matches streaming_depth_edge_threshold default.")
    parser.add_argument("--min_conflict_views", type=int, default=2,
                         help="Matches streaming_ray_semantics_min_views-style certification convention.")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    result = run(
        args.source_path, args.n_source_frames, args.subsequent_window, args.pixel_stride,
        args.min_depth, args.max_depth, args.consistency_thresh, args.conflict_thresh,
        args.edge_thresh, args.min_conflict_views,
    )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[diagnose] wrote {args.output_json}")
