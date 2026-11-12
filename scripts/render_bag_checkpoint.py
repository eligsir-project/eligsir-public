"""Render a bag-sourced streaming checkpoint's held-out test views, using
logged test-camera poses (utils/trajectory_eval.py) and RGB read directly from
the archived bag (scripts/bag_image_reader.py) -- no training, no live ROS2,
no SLAM re-run.
"""
import argparse
import json
import os
import sys

import numpy as np

from scripts.bag_image_reader import read_camera_info, read_rgb_frames_near_timestamps


def _quat_to_rot(q):
    # q = (qx, qy, qz, qw), matches utils/trajectory_eval.py's TUM convention.
    qx, qy, qz, qw = q
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ])


def load_test_poses(model_path):
    traj_dir = os.path.join(model_path, "trajectory_eval")
    tum_path = os.path.join(traj_dir, "trajectory_test_tum.txt")
    index_path = os.path.join(traj_dir, "trajectory_test_index.json")

    if not os.path.exists(tum_path) or not os.path.exists(index_path):
        raise FileNotFoundError(
            f"{model_path} has no trajectory_test_tum.txt/trajectory_test_index.json -- "
            "this checkpoint predates held-out test-camera pose logging "
            "(utils/trajectory_eval.py's test_cams parameter, commit 92e4866 on "
            "feat/local-bounds-program). Re-run training with current code to generate them; "
            "this tool cannot reconstruct poses for a checkpoint that never logged them."
        )

    rows = []
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in line.split())
            c2w = np.eye(4)
            c2w[:3, :3] = _quat_to_rot((qx, qy, qz, qw))
            c2w[:3, 3] = [tx, ty, tz]
            rows.append({"timestamp": t, "c2w": c2w})

    with open(index_path) as f:
        index = json.load(f)

    if len(index) != len(rows):
        raise ValueError(
            f"trajectory_test_index.json has {len(index)} entries but "
            f"trajectory_test_tum.txt has {len(rows)} rows -- files out of sync"
        )

    return [
        {"name": entry["name"], "source_index": entry["source_index"],
         "timestamp": row["timestamp"], "c2w": row["c2w"]}
        for entry, row in zip(index, rows)
    ]


def sanity_check(rendered, report_test_per_view, psnr_tol=0.1, max_mismatches=2):
    rendered_by_key = {(v["name"], v["source_index"]): v for v in rendered}
    mismatches = []
    structural_mismatch = False

    for expected in report_test_per_view:
        key = (expected["name"], expected["source_index"])
        got = rendered_by_key.pop(key, None)
        if got is None:
            mismatches.append({"name": expected["name"], "reason": "missing from rendered set"})
            structural_mismatch = True
            continue
        delta = abs(got["psnr"] - expected["psnr"])
        if delta > psnr_tol:
            mismatches.append({
                "name": expected["name"],
                "reason": f"psnr delta {delta:.3f}dB exceeds tolerance {psnr_tol}dB "
                          f"(rendered={got['psnr']:.3f}, report={expected['psnr']:.3f})",
            })

    for name, source_index in rendered_by_key:
        mismatches.append({"name": name, "reason": "present in rendered set but not in report"})
        structural_mismatch = True

    ok = (not structural_mismatch) and len(mismatches) <= max_mismatches
    return {"ok": ok, "mismatches": mismatches}


def build_camera(pose_entry, rgb, intrinsics, args):
    from eligsir.utils.rgbd_frames import c2w_to_camera_rt
    from eligsir.utils.graphics_utils import focal2fov
    from scene.readers.common import CameraInfo
    from eligsir.utils.camera_utils import loadCam

    R, T = c2w_to_camera_rt(pose_entry["c2w"])
    width, height = intrinsics["width"], intrinsics["height"]
    cam_info = CameraInfo(
        uid=pose_entry["source_index"],
        R=R, T=T,
        FovY=focal2fov(intrinsics["fy"], height),
        FovX=focal2fov(intrinsics["fx"], width),
        image=None,
        image_path="",
        image_name=pose_entry["name"],
        width=width, height=height,
        fx=intrinsics["fx"], fy=intrinsics["fy"],
        cx=intrinsics["cx"], cy=intrinsics["cy"],
    )
    return loadCam(args, pose_entry["source_index"], cam_info, resolution_scale=1.0, rgb_uint8=rgb)


def render_and_score(gaussians, cams, pipeline, background):
    from gaussian_renderer import render
    from eligsir.utils.image_utils import psnr
    import torch

    results = []
    for cam in cams:
        with torch.no_grad():
            out = render(cam, gaussians, pipeline, background)
        rendered_rgb = out["render"].clamp(0.0, 1.0)
        gt_rgb = cam.original_image[0:3, :, :].to(rendered_rgb.device).clamp(0.0, 1.0)
        view_psnr = float(psnr(rendered_rgb, gt_rgb).mean().item())
        results.append({
            "name": cam.image_name, "source_index": int(cam.image_name),
            "psnr": view_psnr, "render_rgb": rendered_rgb, "gt_rgb": gt_rgb,
        })
    return results


def main(argv=None):
    from arguments import ModelParams, PipelineParams
    from scene import GsplatGaussianModel
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--psnr-tol", type=float, default=0.1)
    parser.add_argument("--max-mismatches", type=int, default=2)
    args = parser.parse_args(argv)

    out_dir = args.out or os.path.join(args.model_path, "bag_replay_render", f"iter_{args.iteration}")
    os.makedirs(out_dir, exist_ok=True)

    poses = load_test_poses(args.model_path)
    intrinsics = read_camera_info(args.bag_path)
    frames = read_rgb_frames_near_timestamps(
        args.bag_path, [p["timestamp"] for p in poses], max_dt=0.02)

    class _Args:
        resolution = -1
        data_device = "cuda"

    cams = [
        build_camera(p, frames[p["timestamp"]], intrinsics, _Args())
        for p in poses
    ]

    gaussians = GsplatGaussianModel(3)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}",
                             "point_cloud.ply")
    gaussians.load_ply(ply_path)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParams(argparse.ArgumentParser())
    rendered = render_and_score(gaussians, cams, pipeline, bg_color)

    # Issue #685: this used to hardcode "iter_2147483647" (the --online_mode
    # sentinel dir) unconditionally, even though args.iteration above is a
    # real, parameterized checkpoint iteration -- silently wrong for any
    # non-online_mode/non-sentinel checkpoint this tool is pointed at, and
    # fragile even for sentinel checkpoints if a _bootstrap_views debug
    # variant sorted differently. Delegates to the corrected, numeric-sort
    # report resolver (utils.coverage_access.find_latest_report_path, a thin
    # re-export of utils.baseline_comparison._find_latest_report_json) so
    # this always finds the same report.json the rest of the reporting
    # pipeline would consider authoritative for this model_path.
    from eligsir.utils.coverage_access import find_latest_report_path

    report_path = find_latest_report_path(args.model_path)
    if not report_path:
        raise FileNotFoundError(
            f"{args.model_path} has no comparison/iter_N/report.json -- cannot "
            "locate the report this checkpoint's test_per_view sanity-check "
            "needs (see utils.coverage_access.find_latest_report_path)."
        )
    with open(report_path) as f:
        report = json.load(f)

    # report["test_per_view"] is only a 128-view representative sample of the full
    # 218-entry logged test-pose set (superset relationship, by design -- see
    # docs/... task brief correction). Restrict sanity_check's input to the views
    # the report actually scored so the extra 90 aren't flagged as structural
    # mismatches; they are still rendered and written to disk above.
    scored_keys = {(v["name"], v["source_index"]) for v in report["test_per_view"]}
    rendered_scored = [v for v in rendered if (v["name"], v["source_index"]) in scored_keys]
    result = sanity_check(rendered_scored, report["test_per_view"],
                           psnr_tol=args.psnr_tol, max_mismatches=args.max_mismatches)

    for view in rendered:
        import torchvision
        torchvision.utils.save_image(view["render_rgb"], os.path.join(out_dir, f"{view['name']}_render.png"))
        torchvision.utils.save_image(view["gt_rgb"], os.path.join(out_dir, f"{view['name']}_gt.png"))

    if not result["ok"]:
        for m in result["mismatches"]:
            print(f"MISMATCH {m['name']}: {m['reason']}", file=sys.stderr)
        return 1

    print(f"sanity check OK: {len(rendered)} views rendered, {len(result['mismatches'])} tolerated mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
