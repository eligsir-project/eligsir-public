"""Export a trained checkpoint to a conventional RGB point-cloud PLY.

The repo's native point_cloud.ply is the raw Gaussian schema (SH + opacity +
scale + rotation) -- CloudCompare/MeshLab don't render that as colored points.
This writes a standard x,y,z,red,green,blue PLY, directly loadable there.

Examples:
  python export_pointcloud.py -m output/run1
  python export_pointcloud.py -m output/run1 --iteration 52000 --min_opacity 0.1
  python export_pointcloud.py -m output/run1 --history
"""
from __future__ import annotations

import argparse
import os

from render_report import _latest_checkpoint, _list_checkpoints, _load_cfg_args, load_gaussians
from eligsir.utils.gs_export import gaussians_to_dense_rgb_pointcloud, gaussians_to_rgb_pointcloud


def _export_one(model_path: str, cfg_args, iteration: int, ply_path: str, min_opacity: float, out: str,
                 dense: bool = False, samples_per_gaussian: int = 8, max_points: int = 20_000_000):
    import open3d as o3d

    sh_degree = getattr(cfg_args, "sh_degree", 3)
    model_layout = getattr(cfg_args, "model_layout", "gsplat")
    gaussians = load_gaussians(model_path, sh_degree, model_layout, ply_path)
    if dense:
        pcd = gaussians_to_dense_rgb_pointcloud(
            gaussians, min_opacity=min_opacity, samples_per_gaussian=samples_per_gaussian, max_points=max_points
        )
    else:
        pcd = gaussians_to_rgb_pointcloud(gaussians, min_opacity=min_opacity)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_point_cloud(out, pcd)
    print(f"[export_pointcloud] iter {iteration}: {len(pcd.points)} points -> {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export a checkpoint to a CloudCompare-loadable RGB point-cloud PLY")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest checkpoint")
    parser.add_argument("--history", action="store_true",
                         help="export every saved checkpoint for a time-series inspection")
    parser.add_argument("--min_opacity", type=float, default=0.1,
                         help="drop Gaussians below this activated opacity (default: 0.1)")
    parser.add_argument("--dense", action="store_true",
                         help="sample multiple points per Gaussian's 3D covariance instead of just its center")
    parser.add_argument("--samples_per_gaussian", type=int, default=8,
                         help="dense mode: samples per Gaussian (opacity-weighted; default: 8)")
    parser.add_argument("--max_points", type=int, default=20_000_000,
                         help="dense mode: hard cap on total emitted points (default: 20M)")
    parser.add_argument("-o", "--out", default=None,
                         help="output PLY path (default: <model_path>/comparison/iter_N/pointcloud.ply)")
    args = parser.parse_args()

    model_path = args.model_path
    cfg_args = _load_cfg_args(model_path)

    if args.history:
        ckpts = _list_checkpoints(model_path)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found under {model_path}")
        print(f"[export_pointcloud] history sweep over {len(ckpts)} checkpoints")
        for iteration, ply_path in ckpts:
            out = args.out or os.path.join(model_path, "comparison", "history", f"iter_{iteration}", "pointcloud.ply")
            _export_one(model_path, cfg_args, iteration, ply_path, args.min_opacity, out)
    else:
        if args.iteration is not None:
            ckpts = dict(_list_checkpoints(model_path))
            if args.iteration not in ckpts:
                raise FileNotFoundError(f"No checkpoint at iteration {args.iteration} under {model_path}")
            iteration, ply_path = args.iteration, ckpts[args.iteration]
        else:
            iteration, ply_path = _latest_checkpoint(model_path)
        out = args.out or os.path.join(model_path, "comparison", f"iter_{iteration}", "pointcloud.ply")
        _export_one(model_path, cfg_args, iteration, ply_path, args.min_opacity, out)


if __name__ == "__main__":
    main()
