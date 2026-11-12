#!/usr/bin/env python3
"""Run sequential Open3D RGB-D odometry over a TUM sequence and compare to GT.

Loads TUM frames (rgb.txt + depth.txt + groundtruth.txt), drops the GT poses,
estimates camera-to-world poses frame-by-frame using
``o3d.pipelines.odometry.compute_rgbd_odometry`` (Hybrid term, matching the
settings of the live odometry path used for rosbag inputs), writes both
estimated and GT trajectories in TUM format, and computes ATE/RPE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.utils.streaming_frames import TUMFrameSource
from eligsir.utils.trajectory_eval import (
    associate_by_timestamp,
    discover_gt_trajectory,
    load_tum_trajectory,
    plot_trajectory,
    save_tum_trajectory,
    trajectory_metrics,
)


def _has_nonempty_file(path: str | None) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    return True


def _to_o3d_rgbd(rgb_path: str, depth_path: str, depth_scale: float, depth_max: float):
    color = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path))
    color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=True,
    )


def _estimate_icp(prev_rgbd, cur_rgbd, intrinsic, max_distance: float, robust_kernel: str, sigma: float):
    pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, intrinsic)
    pcd_cur = o3d.geometry.PointCloud.create_from_rgbd_image(cur_rgbd, intrinsic)
    if len(pcd_prev.points) < 20 or len(pcd_cur.points) < 20:
        return False, np.eye(4, dtype=np.float64), {"fitness": 0.0, "inlier_rmse": None}
    pcd_prev.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_cur.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    loss_name = str(robust_kernel).lower()
    if loss_name == "huber":
        loss = o3d.pipelines.registration.HuberLoss(k=sigma)
    elif loss_name == "tukey":
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    else:
        loss = o3d.pipelines.registration.L2Loss()
    reg = o3d.pipelines.registration.registration_icp(
        pcd_prev,
        pcd_cur,
        float(max_distance),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    return (
        bool(reg.fitness > 0.01),
        np.asarray(reg.transformation, dtype=np.float64),
        {"fitness": float(reg.fitness), "inlier_rmse": float(reg.inlier_rmse)},
    )


def _info_summary(info) -> str:
    if isinstance(info, dict):
        parts = []
        if info.get("fitness") is not None:
            parts.append(f"fitness={float(info['fitness']):.4f}")
        if info.get("inlier_rmse") is not None:
            parts.append(f"rmse={float(info['inlier_rmse']):.4f}")
        return " ".join(parts)
    try:
        arr = np.asarray(info)
        if arr.size:
            return f"info_trace={float(np.trace(arr)):.3e}"
    except Exception:
        pass
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-s", "--source_path", required=True, help="TUM sequence root")
    p.add_argument("-m", "--model_path", required=True, help="Output dir")
    p.add_argument("--tum_sequence", default="", help="freiburg1/2/3 for default intrinsics")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--depth_min", type=float, default=0.1)
    p.add_argument("--depth_max", type=float, default=3.0)
    p.add_argument("--depth_diff_max", type=float, default=0.07)
    p.add_argument("--method", choices=["hybrid", "icp"], default="hybrid")
    p.add_argument("--icp_max_distance", type=float, default=0.07)
    p.add_argument("--icp_robust_kernel", choices=["none", "huber", "tukey"], default="huber")
    p.add_argument("--icp_sigma", type=float, default=0.05)
    p.add_argument("--label", default="o3d_rgbd_hybrid")
    args = p.parse_args()

    out_dir = Path(args.model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = TUMFrameSource(
        args.source_path,
        max_frames=0,
        frame_stride=args.frame_stride,
        sequence=args.tum_sequence,
    )
    all_frames = src.get_all()
    valid_frames = [
        f for f in all_frames
        if _has_nonempty_file(f.rgb_path)
        and _has_nonempty_file(f.depth_path)
    ]
    dropped = len(all_frames) - len(valid_frames)
    frames = valid_frames
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    if dropped:
        print(f"[o3d-eval] dropped {dropped}/{len(all_frames)} frames with missing/unreadable RGB-D", flush=True)
    if len(frames) < len(valid_frames):
        print(f"[o3d-eval] capped to {len(frames)}/{len(valid_frames)} readable RGB-D frames", flush=True)
    if len(frames) < 2:
        print(f"[o3d-eval] not enough frames with depth: {len(frames)}", flush=True)
        return 1
    f0 = frames[0]
    print(f"[o3d-eval] {len(frames)} frames, intrinsics fx={f0.fx:.2f} fy={f0.fy:.2f} "
          f"cx={f0.cx:.2f} cy={f0.cy:.2f} size={f0.width}x{f0.height} depth_scale={f0.depth_scale}",
          flush=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(f0.width), int(f0.height),
        float(f0.fx), float(f0.fy), float(f0.cx), float(f0.cy),
    )
    option = o3d.pipelines.odometry.OdometryOption(
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_diff_max=args.depth_diff_max,
    )
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    # Save GT trajectory (from TUMFrameSource c2w, which == GT)
    gt_ts = np.asarray([f.timestamp for f in frames], dtype=np.float64)
    gt_c2w = np.stack([np.asarray(f.c2w, dtype=np.float64) for f in frames])
    save_tum_trajectory(str(out_dir / "groundtruth_full.txt"), gt_ts, gt_c2w)

    # Run sequential odometry
    est_c2w = np.zeros((len(frames), 4, 4), dtype=np.float64)
    est_c2w[0] = np.eye(4)
    n_fail = 0
    t0 = time.perf_counter()
    prev_rgbd = _to_o3d_rgbd(frames[0].rgb_path, frames[0].depth_path, frames[0].depth_scale, args.depth_max)
    for i in range(1, len(frames)):
        cur = frames[i]
        try:
            cur_rgbd = _to_o3d_rgbd(cur.rgb_path, cur.depth_path, cur.depth_scale, args.depth_max)
        except Exception as e:
            n_fail += 1
            est_c2w[i] = est_c2w[i - 1]
            print(f"[o3d-eval] frame {i} depth unreadable ({e}); carrying pose forward", flush=True)
            continue
        if args.method == "icp":
            ok, T_prev_to_cur, _info = _estimate_icp(
                prev_rgbd,
                cur_rgbd,
                intrinsic,
                args.icp_max_distance,
                args.icp_robust_kernel,
                args.icp_sigma,
            )
        else:
            ok, T_prev_to_cur, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, cur_rgbd, intrinsic, np.eye(4), jacobian, option,
            )
        if not ok:
            n_fail += 1
            T_prev_to_cur = np.eye(4)
        # Match the live Open3D odometry path: compute prev->cur, then compose
        # world_from_cur = world_from_prev @ cur_from_prev.
        est_c2w[i] = est_c2w[i - 1] @ np.linalg.inv(T_prev_to_cur)
        prev_rgbd = cur_rgbd
        if (i % 50) == 0:
            elapsed_s = max(time.perf_counter() - t0, 1e-9)
            info_text = _info_summary(_info)
            print(f"[o3d-eval] {i}/{len(frames)}  failures={n_fail}  "
                  f"elapsed={elapsed_s:.1f}s  odometry_fps={(i + 1) / elapsed_s:.2f}"
                  + (f"  {info_text}" if info_text else ""),
                  flush=True)

    est_ts = gt_ts.copy()
    save_tum_trajectory(str(out_dir / "trajectory_tum.txt"), est_ts, est_c2w)
    elapsed_s = max(time.perf_counter() - t0, 1e-9)
    odometry_fps = float(len(frames) / elapsed_s)
    odometry_pairs_per_s = float(max(0, len(frames) - 1) / elapsed_s)
    print(
        f"[o3d-eval] done in {elapsed_s:.1f}s, failures={n_fail}/{len(frames)-1}, "
        f"odometry_fps={odometry_fps:.2f}, pairs_per_s={odometry_pairs_per_s:.2f}",
        flush=True,
    )

    # Compare against discovered GT (round-trip through TUM I/O to mimic real eval)
    gt_path = discover_gt_trajectory(args.source_path) or ""
    metrics: dict = {
        "method": args.label,
        "odometry_method": args.method,
        "source_path": args.source_path,
        "n_frames": int(len(est_c2w)),
        "n_failed": int(n_fail),
        "elapsed_s": float(elapsed_s),
        "odometry_fps": odometry_fps,
        "odometry_pairs_per_s": odometry_pairs_per_s,
    }
    plot_gt = None
    plot_est = est_c2w
    if gt_path:
        metrics["gt_tum_path"] = gt_path
        gt_ts_disk, gt_all = load_tum_trajectory(gt_path)
        m_ts, m_est, m_gt, assoc = associate_by_timestamp(est_ts, est_c2w, gt_ts_disk, gt_all, max_dt=0.03)
        metrics["association"] = assoc
        if len(m_ts) > 0:
            save_tum_trajectory(str(out_dir / "trajectory_tum_associated.txt"), m_ts, m_est)
            save_tum_trajectory(str(out_dir / "groundtruth_tum_associated.txt"), m_ts, m_gt)
            metrics.update(trajectory_metrics(m_est, m_gt))
            plot_est = m_est
            plot_gt = m_gt
    else:
        print("[o3d-eval] no GT auto-discovered", flush=True)

    plot_trajectory(str(out_dir / "trajectory.svg"), plot_est, title=args.label, gt=plot_gt, metrics=metrics)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
