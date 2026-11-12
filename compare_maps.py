"""Compare a Gaussian-splat checkpoint against an independent, non-neural
point-cloud-only reference map (e.g. an Open3D TSDF mesh from embed_slam).

See CLAUDE.md's "Gaussian vs. point-cloud-only map comparison" section for
the full cross-repo workflow that makes this apples-to-apples (rosbag-colmap
-pipeline's pseudo-GT trajectory feeding both the reference map and this
checkpoint's training).

Example:
  python compare_maps.py -m output/run1 --reference_map ref_map.ply
  python compare_maps.py -m output/run1 --reference_map ref_map.ply --no-align
"""
from __future__ import annotations

import argparse
import json
import os

from render_report import _latest_checkpoint, _list_checkpoints, _load_cfg_args, load_gaussians
from eligsir.utils.map_comparison import compare_gaussian_vs_reference_map, load_reference_pointcloud, render_map_overlay


def main():
    parser = argparse.ArgumentParser(description="Compare a Gaussian map vs. an independent point-cloud-only map")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest checkpoint")
    parser.add_argument("--reference_map", required=True, help="path to the reference .ply (mesh or point cloud)")
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--tau_m", type=float, default=0.05, help="off-surface distance threshold [m]")
    parser.add_argument("--align", dest="align", action="store_true", default=True,
                         help="ICP-align the reference map onto the Gaussian map (default)")
    parser.add_argument("--no-align", dest="align", action="store_false",
                         help="skip alignment -- only valid when both maps share the same pose source")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--viz", action="store_true", help="write a two-cloud overlay figure alongside the report")
    args = parser.parse_args()

    model_path = args.model_path
    cfg_args = _load_cfg_args(model_path)

    if args.iteration is not None:
        ckpts = dict(_list_checkpoints(model_path))
        if args.iteration not in ckpts:
            raise FileNotFoundError(f"No checkpoint at iteration {args.iteration} under {model_path}")
        iteration, ply_path = args.iteration, ckpts[args.iteration]
    else:
        iteration, ply_path = _latest_checkpoint(model_path)

    sh_degree = getattr(cfg_args, "sh_degree", 3)
    model_layout = getattr(cfg_args, "model_layout", "gsplat")
    gaussians = load_gaussians(model_path, sh_degree, model_layout, ply_path)

    print(f"[compare_maps] loading reference map: {args.reference_map}", flush=True)
    reference_pcd = load_reference_pointcloud(args.reference_map)

    print(f"[compare_maps] comparing iter {iteration} vs {len(reference_pcd.points)} reference points "
          f"(align={args.align})", flush=True)
    report = compare_gaussian_vs_reference_map(
        gaussians, reference_pcd,
        min_opacity=args.min_opacity, tau_m=args.tau_m,
        align=args.align, voxel_size=args.voxel_size,
    )
    report["model_path"] = model_path
    report["iteration"] = iteration
    report["reference_map_path"] = args.reference_map

    out = args.out or os.path.join(model_path, "comparison", f"iter_{iteration}", "map_comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[compare_maps] wrote {out}", flush=True)

    if report.get("valid"):
        sym = report.get("symmetric_chamfer_m")
        g2r = report["gaussian_to_reference"]["offsurface_mass_frac"]
        r2g = report["reference_to_gaussian"]["offsurface_mass_frac"]
        print(f"[compare_maps] symmetric_chamfer_m={sym:.4f} "
              f"gaussian->reference_offsurface_mass={g2r:.4f} "
              f"reference->gaussian_offsurface_mass={r2g:.4f}", flush=True)
        if args.align:
            align_info = report["alignment"]
            print(f"[compare_maps] alignment fitness={align_info['fitness']:.4f} "
                  f"rmse={align_info['inlier_rmse']:.4f}", flush=True)

    if args.viz:
        import numpy as np
        gaussian_xyz = gaussians.get_xyz.detach().float().cpu().numpy()
        reference_xyz = np.asarray(reference_pcd.points)
        viz_out = os.path.join(os.path.dirname(out), "map_comparison.svg")
        render_map_overlay(viz_out, gaussian_xyz, reference_xyz)
        print(f"[compare_maps] wrote {viz_out}", flush=True)


if __name__ == "__main__":
    main()
