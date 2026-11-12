import argparse
import json
import os

import numpy as np
import torch
from plyfile import PlyData

from eligsir.utils.sh_utils import eval_sh


def eval_sh_playcanvas_style(shs: torch.Tensor, dirs: torch.Tensor, deg: int) -> torch.Tensor:
    """Re-evaluate SH coefficients using PlayCanvas-style lower-only clamping."""
    raw = eval_sh(deg, shs.transpose(1, 2), dirs)
    return torch.clamp_min(raw + 0.5, 0.0)


def _load_ply_raw(ply_path: str, sh_degree: int):
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    f_rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1)
    f_rest = f_rest.reshape(-1, 3, (sh_degree + 1) ** 2 - 1).transpose(0, 2, 1)
    shs = np.concatenate([f_dc[:, None, :], f_rest], axis=1)
    return torch.from_numpy(xyz).float(), torch.from_numpy(shs).float()


def compute_phase3_metrics(model_path, model_layout, output_dir):
    # Native rendering is needed only for the end-to-end diagnostic, not for
    # eval_sh_playcanvas_style/_load_ply_raw. Keep those helpers importable on
    # the device-free CPU-Torch CI surface.
    from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint
    from scripts.sh_clamp_diagnostic.phase1_renderer_ablation import render_variant
    from eligsir.mapping.renderer.gsplat_backend import render, _camera_tensors
    from eligsir.utils.image_utils import psnr

    os.makedirs(output_dir, exist_ok=True)
    gaussians, scene, pipe = load_checkpoint(model_path, model_layout)
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    background = torch.zeros(3, device=gaussians.get_xyz.device)

    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "point_cloud.ply")
    xyz, shs = _load_ply_raw(ply_path, gaussians.active_sh_degree)
    xyz = xyz.to(gaussians.get_xyz.device)
    shs = shs.to(gaussians.get_xyz.device)

    current_renders = render_variant(gaussians, scene, pipe, "clamped", cameras)
    lower_renders = render_variant(gaussians, scene, pipe, "lower_only", cameras)

    per_camera = []
    frac_over_list = []
    for cam, r_current, r_lower in zip(cameras, current_renders, lower_renders):
        _, _, cam_center = _camera_tensors(cam, xyz.device)
        dirs = torch.nn.functional.normalize(xyz - cam_center, dim=1, eps=1e-8)
        export_colors = eval_sh_playcanvas_style(shs, dirs, gaussians.active_sh_degree)
        frac_over_list.append((export_colors > 1.0).float().mean().item())

        with torch.no_grad():
            export_render = render(
                cam,
                gaussians,
                pipe,
                background,
                override_color=export_colors.clamp(0, 1),
            )["render"]

        r_current_c = r_current.clamp(0, 1)
        r_lower_c = r_lower.clamp(0, 1)
        export_render_c = export_render.clamp(0, 1)

        per_camera.append({
            "psnr_export_vs_current": psnr(export_render_c, r_current_c).mean().item(),
            "psnr_export_vs_lower": psnr(export_render_c, r_lower_c).mean().item(),
        })

    n = len(per_camera)
    return {
        "psnr_export_vs_current": sum(c["psnr_export_vs_current"] for c in per_camera) / n,
        "psnr_export_vs_lower": sum(c["psnr_export_vs_lower"] for c in per_camera) / n,
        "mean_frac_pixels_gt_1_0": sum(frac_over_list) / n,
        "n_cameras": n,
        "model_path": model_path,
        "note": "export render approximated via override_color on the existing rasterizer "
                "(same geometry/opacity as the live model, PLY-derived colors) -- not a "
                "literal SuperSplat/PlayCanvas render.",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: export-parity approximation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    result = compute_phase3_metrics(args.model_path, args.model_layout, args.output_dir)
    with open(os.path.join(args.output_dir, "phase3_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
