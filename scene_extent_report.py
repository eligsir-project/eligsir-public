"""Standalone scene-vs-trajectory extent/plume diagnostic CLI.

Loads a checkpoint's Gaussians + the run's camera trajectory, runs the
scene-vs-trajectory extent metrics (utils/scene_geometry_metrics), writes
scene_extent_report.json, and renders a 3-panel figure (top-down + 2 side
views) showing the point cloud, trajectory overlay, and both bounding boxes.

This is the tool for the "how much of the reconstruction is outside where
the camera actually went" investigation this session did ad-hoc by hand for
ScanNet++ (~360x251x85m vs a real ~6x6m room) and floor3 (~82x58x77m).

Example:
  python scene_extent_report.py -m output/run1
  python scene_extent_report.py -m output/run1 --iteration 52000 --margin_m 3.0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eligsir.plot_style import svg_output_path
from render_report import _latest_checkpoint, _list_checkpoints, _load_cfg_args, load_gaussians
from eligsir.utils.scene_geometry_metrics import (
    compute_voxel_occupancy, compute_trajectory_extent, compare_scene_vs_trajectory,
)


def load_trajectory_positions(model_path: str) -> np.ndarray:
    """Prefer trajectory_full.jsonl (dense, every ingested frame); fall back
    to cameras.json (admitted train keyframes)."""
    dense_path = os.path.join(model_path, "trajectory_full.jsonl")
    if os.path.exists(dense_path):
        positions = []
        with open(dense_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    positions.append(json.loads(line)["position"])
        if positions:
            return np.asarray(positions, dtype=np.float64)

    cameras_path = os.path.join(model_path, "cameras.json")
    if os.path.exists(cameras_path):
        with open(cameras_path) as f:
            entries = json.load(f)
        return np.asarray([e["position"] for e in entries], dtype=np.float64)

    raise FileNotFoundError(
        f"Neither trajectory_full.jsonl nor cameras.json found under {model_path}")


def _render_figure(out_path: str, points: np.ndarray, traj: np.ndarray,
                    scene_bbox, traj_bbox_margin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    views = [("Top-down (X-Y)", 0, 1), ("Side (X-Z)", 0, 2), ("Side (Y-Z)", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, i, j) in zip(axes, views):
        ax.scatter(points[:, i], points[:, j], s=0.5, c="gray", alpha=0.3, label="Gaussians")
        ax.scatter(traj[:, i], traj[:, j], s=4, c="red", label="trajectory")

        smin, smax = scene_bbox
        ax.add_patch(Rectangle((smin[i], smin[j]), smax[i] - smin[i], smax[j] - smin[j],
                                fill=False, edgecolor="blue", linewidth=1.5, label="scene bbox"))
        tmin, tmax = traj_bbox_margin
        ax.add_patch(Rectangle((tmin[i], tmin[j]), tmax[i] - tmin[i], tmax[j] - tmin[j],
                                fill=False, edgecolor="green", linewidth=1.5, label="trajectory bbox+margin"))

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(svg_output_path(out_path), dpi=150)
    plt.close(fig)


def run(model_path: str, iteration: int, ply_path: str, margin_m: float, voxel_size: float,
        out_dir: str, opacity_threshold: float = 0.05):
    cfg_args = _load_cfg_args(model_path)
    sh_degree = getattr(cfg_args, "sh_degree", 3)
    model_layout = getattr(cfg_args, "model_layout", "gsplat")
    gaussians = load_gaussians(model_path, sh_degree, model_layout, ply_path)

    xyz_all = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    opacity_all = gaussians.get_opacity.detach().cpu().numpy().reshape(-1).astype(np.float64)
    # Live mass only, matching scene_geometry_metrics' comparability contract
    # (Tier A restricts to opacity > opacity_threshold) -- a handful of
    # near-dead outlier-scale Gaussians can otherwise dominate the bbox.
    live = opacity_all > opacity_threshold
    xyz = xyz_all[live]
    opacity = opacity_all[live]
    # trajectory_full.jsonl (dense, every ingested frame) vs cameras.json
    # (admitted train keyframes only) -- prefer the former when present, so
    # this CLI's bbox_volume_ratio is NOT directly comparable to the one
    # utils/comparison_report.py's write_post_training_report wires into
    # geometry_metrics.json, which uses train_cams (admitted keyframes) only.
    traj = load_trajectory_positions(model_path)

    voxel_occupancy = compute_voxel_occupancy(xyz, voxel_size)
    trajectory_extent = compute_trajectory_extent(traj)
    comparison = compare_scene_vs_trajectory(xyz, opacity, traj, margin_m=margin_m,
                                             opacity_threshold=0.0)  # already live-filtered above

    report = {
        "model_path": model_path,
        "iteration": iteration,
        "n_gaussians_total": int(xyz_all.shape[0]),
        "n_gaussians_live": int(xyz.shape[0]),
        "opacity_threshold": float(opacity_threshold),
        "voxel_occupancy": voxel_occupancy,
        "trajectory_extent": trajectory_extent,
        "scene_vs_trajectory": comparison,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "scene_extent_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[scene_extent_report] wrote {json_path}", flush=True)
    print(f"[scene_extent_report] n_gaussians={xyz.shape[0]} "
          f"scene_bbox_size={voxel_occupancy.get('bbox_size')} "
          f"trajectory_bbox_size={trajectory_extent.get('bbox_size')} "
          f"bbox_volume_ratio={comparison.get('bbox_volume_ratio')} "
          f"frac_mass_outside={comparison.get('frac_mass_outside_trajectory_bbox')}", flush=True)

    scene_bbox = (np.asarray(voxel_occupancy["bbox_min"]), np.asarray(voxel_occupancy["bbox_max"]))
    traj_bbox_margin = (np.asarray(trajectory_extent["bbox_min"]) - margin_m,
                        np.asarray(trajectory_extent["bbox_max"]) + margin_m)
    svg_path = os.path.join(out_dir, "scene_extent.svg")
    _render_figure(svg_path, xyz, traj, scene_bbox, traj_bbox_margin)
    print(f"[scene_extent_report] wrote {svg_path}", flush=True)

    return report


def main():
    parser = argparse.ArgumentParser(description="Scene-vs-trajectory extent/plume diagnostic")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest checkpoint")
    parser.add_argument("--margin_m", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--opacity_threshold", type=float, default=0.05,
                         help="live-mass cutoff, matches scene_geometry_metrics' Tier A default")
    args = parser.parse_args()

    model_path = args.model_path
    if args.iteration is not None:
        ckpts = dict(_list_checkpoints(model_path))
        if args.iteration not in ckpts:
            raise FileNotFoundError(f"No checkpoint at iteration {args.iteration} under {model_path}")
        iteration, ply_path = args.iteration, ckpts[args.iteration]
    else:
        iteration, ply_path = _latest_checkpoint(model_path)

    out_dir = os.path.join(model_path, "comparison", f"iter_{iteration}")
    run(model_path, iteration, ply_path, args.margin_m, args.voxel_size, out_dir, args.opacity_threshold)


if __name__ == "__main__":
    main()
