"""Dense pointcloud + mesh export via the external Lewis-Stuart-11/3DGS-to-PC tool.

Native gs_export.py (utils/gs_export.py) already does a lightweight
per-Gaussian-covariance dense sample. This script instead drives the real
3DGS-to-PC tool (github.com/Lewis-Stuart-11/3DGS-to-PC, checked out as a peer
repo at ~/git/3DGS-to-PC, same convention as eligsir-orbslam3 etc in
CLAUDE.md) for its rendered-colour dense sampling and Poisson mesh
reconstruction, which the native exporter does not attempt.

3DGS-to-PC's colour-rendering and meshing paths need real camera poses. Our
cameras.json/test_cameras.json store camera-to-world (position + rotation,
OpenCV/COLMAP axis convention -- see utils/camera_utils.py:camera_to_JSON,
confusingly named "W2C" locally but actually C2W: it's the inverse of the
world-to-camera Rt built from camera.R/camera.T). 3DGS-to-PC's own
transforms.json loader expects an OpenGL-convention transform_matrix with no
conversion applied -- feeding our COLMAP-convention C2W there directly would
silently mis-render colours (exactly the pose-convention bug class this repo
has repeatedly root-caused elsewhere). Its COLMAP-images.txt loader, by
contrast, has an explicit, well-tested W2C -> C2W + axis-flip step
(convert_sfm_pose_to_nerf), so this script converts our cameras to COLMAP's
images.txt/cameras.txt format (world-to-camera qvec/tvec) instead of writing
a transforms.json, deliberately routing through the safer, already-verified
conversion path.

Usage (inside the container):
  python scripts/export_dense_pointcloud.py -m output/run1 \
      --tool_path /workspace/3DGS-to-PC --generate_mesh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_report import _latest_checkpoint, _list_checkpoints


def _rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP (w, x, y, z) quaternion. Standard Shepperd's method."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _load_cameras(model_path: str):
    for fname in ("test_cameras.json", "cameras.json"):
        path = os.path.join(model_path, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), fname
    raise FileNotFoundError(f"No cameras.json/test_cameras.json found under {model_path}")


def _write_colmap_transforms(cameras, out_dir: str):
    """Write minimal COLMAP images.txt/cameras.txt (poses + PINHOLE intrinsics only)."""
    images_path = os.path.join(out_dir, "images.txt")
    cameras_path = os.path.join(out_dir, "cameras.txt")

    with open(cameras_path, "w") as cf, open(images_path, "w") as imf:
        cf.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx,fy,cx,cy]\n")
        imf.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")

        for cam in cameras:
            cam_id = cam["id"]
            width, height = cam["width"], cam["height"]
            fx, fy = cam["fx"], cam["fy"]
            cx = cam.get("cx", width / 2.0)
            cy = cam.get("cy", height / 2.0)
            cf.write(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

            c2w = np.eye(4)
            c2w[:3, :3] = np.array(cam["rotation"])
            c2w[:3, 3] = np.array(cam["position"])
            w2c = np.linalg.inv(c2w)

            qvec = _rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            name = cam.get("img_name", f"{cam_id:06d}") + ".png"

            imf.write(
                f"{cam_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} {cam_id} {name}\n"
            )
            imf.write("\n")  # empty POINTS2D line, dense export doesn't need it

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Dense pointcloud + mesh export via 3DGS-to-PC")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest checkpoint")
    parser.add_argument("--tool_path", default="/workspace/3DGS-to-PC",
                         help="path to the checked-out Lewis-Stuart-11/3DGS-to-PC repo inside the container")
    parser.add_argument("--num_points", type=int, default=5_000_000)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--colour_quality", default="high")
    parser.add_argument("--camera_skip_rate", type=int, default=0,
                         help="render every (N+1)-th camera only -- trades colour accuracy for speed "
                              "on the pure-Python renderer, which has no CUDA/HIP fused kernel")
    parser.add_argument("--generate_mesh", action="store_true")
    parser.add_argument("--poisson_depth", type=int, default=10)
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--mesh_out", default=None)
    args = parser.parse_args()

    if args.iteration is not None:
        ckpts = dict(_list_checkpoints(args.model_path))
        if args.iteration not in ckpts:
            raise FileNotFoundError(f"No checkpoint at iteration {args.iteration} under {args.model_path}")
        iteration, ply_path = args.iteration, ckpts[args.iteration]
    else:
        iteration, ply_path = _latest_checkpoint(args.model_path)

    cameras, source = _load_cameras(args.model_path)
    print(f"[export_dense_pointcloud] loaded {len(cameras)} cameras from {source}", flush=True)

    out = args.out or os.path.join(args.model_path, "comparison", f"iter_{iteration}", "dense_pointcloud_3dgstopc.ply")
    mesh_out = args.mesh_out or os.path.join(args.model_path, "comparison", f"iter_{iteration}", "dense_mesh_3dgstopc.ply")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _write_colmap_transforms(cameras, tmp_dir)

        cmd = [
            sys.executable, os.path.join(args.tool_path, "gauss_to_pc.py"),
            "--input_path", os.path.abspath(ply_path),
            "--output_path", os.path.abspath(out),
            "--transform_path", tmp_dir,
            "--renderer_type", "python",
            "--num_points", str(args.num_points),
            "--min_opacity", str(args.min_opacity),
            "--colour_quality", args.colour_quality,
            "--camera_skip_rate", str(args.camera_skip_rate),
        ]
        if args.generate_mesh:
            cmd += ["--generate_mesh", "--poisson_depth", str(args.poisson_depth),
                    "--mesh_output_path", os.path.abspath(mesh_out)]

        print(f"[export_dense_pointcloud] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=args.tool_path, check=True)

    print(f"[export_dense_pointcloud] iter {iteration}: dense pointcloud -> {out}", flush=True)
    if args.generate_mesh:
        print(f"[export_dense_pointcloud] iter {iteration}: mesh -> {mesh_out}", flush=True)


if __name__ == "__main__":
    main()
