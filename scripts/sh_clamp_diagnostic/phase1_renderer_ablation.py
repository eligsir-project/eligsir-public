import argparse
import json
import os

import torch


def render_variant(gaussians, scene, pipe, clamp_mode, cameras, sh_impl="python"):
    """Render every camera with pipe.sh_color_clamp_mode set to clamp_mode.

    Native renderer import is local so pure summary helpers remain usable on the
    CPU-Torch CI surface without a native gsplat installation.
    """
    from eligsir.mapping.renderer.gsplat_backend import render

    pipe.sh_impl = sh_impl
    pipe.sh_color_clamp_mode = clamp_mode
    background = torch.zeros(3, device=gaussians.get_xyz.device)
    outs = []
    try:
        with torch.no_grad():
            for cam in cameras:
                outs.append(render(cam, gaussians, pipe, background)["render"])
    finally:
        pipe.sh_impl = "python"
    return outs


def summarize_metrics(per_camera: list) -> dict:
    """Backend-independent reduction over already-computed camera metrics."""
    n = len(per_camera)
    keys = [k for k in per_camera[0].keys() if k != "frac_lower_pixels_gt_1_0"]
    summary = {f"mean_{k}": sum(c[k] for c in per_camera) / n for k in keys}
    summary["mean_delta_psnr"] = summary["mean_psnr_current_vs_gt"] - summary["mean_psnr_lower_vs_gt"]
    summary["mean_frac_lower_pixels_gt_1_0"] = sum(c["frac_lower_pixels_gt_1_0"] for c in per_camera) / n
    summary["n_cameras"] = n
    return summary


def compute_phase1_metrics(model_path, model_layout, output_dir, max_cameras=None):
    # Keep heavy/native dependencies off the import path used by summarize_metrics.
    from lpipsPyTorch import lpips
    from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint
    from eligsir.utils.comparison_report import _abs_diff, _save_png
    from eligsir.utils.image_utils import psnr
    from eligsir.utils.loss_utils import ssim

    os.makedirs(output_dir, exist_ok=True)
    gaussians, scene, pipe = load_checkpoint(model_path, model_layout)
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    if max_cameras is not None:
        cameras = cameras[:max_cameras]

    current_renders = render_variant(gaussians, scene, pipe, "clamped", cameras)
    lower_renders_raw = render_variant(gaussians, scene, pipe, "lower_only", cameras)
    native_renders_raw = render_variant(gaussians, scene, pipe, "clamped", cameras, sh_impl="gsplat")

    per_camera = []
    for idx, (cam, r_current, r_lower_raw, r_native_raw) in enumerate(
        zip(cameras, current_renders, lower_renders_raw, native_renders_raw)
    ):
        gt = cam.original_image[0:3, :, :].to(r_current.device)
        frac_gt_1 = (r_lower_raw > 1.0).float().mean().item()
        r_current_c = r_current.clamp(0, 1)
        r_lower = r_lower_raw.clamp(0, 1)
        r_native = r_native_raw.clamp(0, 1)

        entry = {
            "psnr_current_vs_gt": psnr(r_current_c, gt).mean().item(),
            "psnr_lower_vs_gt": psnr(r_lower, gt).mean().item(),
            "ssim_current_vs_gt": ssim(r_current_c, gt).item(),
            "ssim_lower_vs_gt": ssim(r_lower, gt).item(),
            "lpips_current_vs_gt": lpips(r_current_c, gt, net_type="vgg").item(),
            "lpips_lower_vs_gt": lpips(r_lower, gt, net_type="vgg").item(),
            "psnr_current_vs_lower": psnr(r_current_c, r_lower).mean().item(),
            "frac_lower_pixels_gt_1_0": frac_gt_1,
            "psnr_native_vs_gt": psnr(r_native, gt).mean().item(),
            "psnr_native_vs_current": psnr(r_native, r_current_c).mean().item(),
            "psnr_native_vs_lower": psnr(r_native, r_lower).mean().item(),
            "ssim_native_vs_gt": ssim(r_native, gt).item(),
            "lpips_native_vs_gt": lpips(r_native, gt, net_type="vgg").item(),
        }
        per_camera.append(entry)

        if idx == 0:
            diff_img = _abs_diff(r_current_c, r_lower)
            _save_png(os.path.join(output_dir, "diff_heatmap_cam0.png"), diff_img)

    summary = summarize_metrics(per_camera)
    summary["per_camera"] = per_camera
    summary["model_path"] = model_path
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 1: frozen-checkpoint clamp[0,1] vs clamp_min ablation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cameras", type=int, default=None,
                         help="Cap the number of cameras rendered/scored (default: all). "
                              "Reduces wall-clock for a quick real-number pass; omit for the full run.")
    args = parser.parse_args()

    summary = compute_phase1_metrics(args.model_path, args.model_layout, args.output_dir, args.max_cameras)
    with open(os.path.join(args.output_dir, "phase1_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_camera"}, indent=2))


if __name__ == "__main__":
    main()
