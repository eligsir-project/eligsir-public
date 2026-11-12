import argparse
import json
import os

import torch

VARIANTS = {
    "variant_A": {"sh_color_clamp_mode": "clamped", "eps2d": 0.3},
    "variant_B": {"sh_color_clamp_mode": "lower_only", "eps2d": 0.3},
    "variant_C": {"sh_color_clamp_mode": "lower_only", "eps2d": 0.0},
}


def isolate_effects(variant_summaries: dict) -> dict:
    """Backend-independent A/B/C reduction used by the CPU-Torch tests."""
    return {
        "A_to_B_delta_psnr": variant_summaries["variant_A"]["mean_psnr_vs_gt"]
        - variant_summaries["variant_B"]["mean_psnr_vs_gt"],
        "B_to_C_delta_psnr": variant_summaries["variant_B"]["mean_psnr_vs_gt"]
        - variant_summaries["variant_C"]["mean_psnr_vs_gt"],
    }


def _render_and_score(gaussians, pipe, cameras, background, settings):
    from lpipsPyTorch import lpips
    from eligsir.mapping.renderer.gsplat_backend import render
    from eligsir.utils.image_utils import psnr
    from eligsir.utils.loss_utils import ssim

    for key, value in settings.items():
        setattr(pipe, key, value)
    per_camera = []
    with torch.no_grad():
        for cam in cameras:
            gt = cam.original_image[0:3, :, :].to(gaussians.get_xyz.device)
            out = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
            per_camera.append({
                "psnr_vs_gt": psnr(out, gt).mean().item(),
                "ssim_vs_gt": ssim(out, gt).item(),
                "lpips_vs_gt": lpips(out, gt, net_type="vgg").item(),
            })
    n = len(per_camera)
    return {
        "mean_psnr_vs_gt": sum(c["psnr_vs_gt"] for c in per_camera) / n,
        "mean_ssim_vs_gt": sum(c["ssim_vs_gt"] for c in per_camera) / n,
        "mean_lpips_vs_gt": sum(c["lpips_vs_gt"] for c in per_camera) / n,
        "n_cameras": n,
    }


def compute_phase2_metrics(model_path, model_layout, output_dir, max_cameras=None):
    from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint

    os.makedirs(output_dir, exist_ok=True)
    gaussians, scene, pipe = load_checkpoint(model_path, model_layout)
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    if max_cameras is not None:
        cameras = cameras[:max_cameras]
    background = torch.zeros(3, device=gaussians.get_xyz.device)

    variant_summaries = {
        name: _render_and_score(gaussians, pipe, cameras, background, settings)
        for name, settings in VARIANTS.items()
    }
    deltas = isolate_effects(variant_summaries)
    result = {**variant_summaries, **deltas, "model_path": model_path}
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2: eps2d isolation (variants A/B/C)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cameras", type=int, default=None,
                         help="Cap the number of cameras rendered/scored (default: all). "
                              "Reduces wall-clock for a quick real-number pass; omit for the full run.")
    args = parser.parse_args()

    result = compute_phase2_metrics(args.model_path, args.model_layout, args.output_dir, args.max_cameras)
    with open(os.path.join(args.output_dir, "phase2_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
