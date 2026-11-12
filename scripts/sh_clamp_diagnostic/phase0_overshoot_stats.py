import argparse
import json

import torch

from eligsir.utils.sh_utils import eval_sh

# NOTE on weighting (judgment call, see task-2-report.md for full rationale):
# render()'s return dict exposes "alpha" as render_alphas[0], which is
# gsplat's PER-PIXEL accumulated alpha ([H, W, 1]), not a per-Gaussian
# contribution weight (transmittance x opacity summed over the pixels a
# given Gaussian actually covers). No such per-Gaussian weight is computed
# anywhere in the existing rasterization call without deeper instrumentation
# of the CUDA/HIP kernel, which is out of scope for this diagnostic. We
# therefore restrict to the Gaussians gsplat actually rendered for the view
# (result["visibility_filter"], a genuine per-Gaussian boolean already
# computed from radii > 0) and fall back to a uniform weight of 1.0 across
# those visible Gaussians for the "weighted" statistic. This is the
# fallback explicitly permitted by the brief; see
# stats["weighted_frac_approximation_note"] below.
WEIGHTED_FRAC_APPROXIMATION_NOTE = (
    "gsplat's render() output does not expose a genuine per-Gaussian "
    "contribution weight (transmittance x opacity accumulated over covered "
    "pixels); render_alphas is per-pixel, not per-Gaussian. The "
    "'weighted_frac_*' statistics therefore use a uniform weight of 1.0 "
    "across Gaussians that were actually rendered for the view "
    "(visibility_filter, i.e. radii > 0), which is equivalent to the "
    "unweighted 'raw_frac_*' statistics restricted to visible Gaussians. "
    "This is an approximation, not a true visibility/contribution-weighted "
    "statistic."
)


def compute_overshoot_stats_from_raw(raw_max_per_sample: torch.Tensor, weights: torch.Tensor) -> dict:
    """Pure-Torch reduction used by CPU CI as well as the native diagnostic."""
    thresholds = [1.0, 1.1, 1.5, 2.0]
    stats = {}
    total = raw_max_per_sample.numel()
    total_w = weights.sum().item()
    for t in thresholds:
        mask = raw_max_per_sample > t
        key = str(t).replace(".", "_")
        stats[f"raw_frac_gt_{key}"] = mask.float().sum().item() / total
        stats[f"weighted_frac_gt_{key}"] = (weights[mask].sum() / total_w).item()
    return stats


def _raw_overshoot_and_weights_for_camera(cam, gaussians, pipe):
    """Per-Gaussian raw SH overshoot for one camera.

    Native renderer imports are intentionally local: the reduction helpers in
    this module are CPU-safe and should remain collectible on the CPU-Torch CI
    surface even when the native gsplat package is absent.
    """
    from eligsir.mapping.renderer.gsplat_backend import render, _camera_tensors

    means = gaussians.get_xyz.contiguous()
    _, _, cam_center = _camera_tensors(cam, means.device)
    dir_pp_normalized = torch.nn.functional.normalize(means - cam_center, dim=1, eps=1e-8)

    features_dc = gaussians._features_dc if hasattr(gaussians, "_features_dc") else gaussians.get_features[:, :1, :]
    features_rest = gaussians._features_rest if hasattr(gaussians, "_features_rest") else gaussians.get_features[:, 1:, :]
    shs = torch.cat([features_dc, features_rest], dim=1)
    raw_colors = eval_sh(gaussians.active_sh_degree, shs.transpose(1, 2), dir_pp_normalized)
    raw_max_per_gaussian = (raw_colors + 0.5).max(dim=-1).values

    with torch.no_grad():
        out = render(cam, gaussians, pipe, torch.zeros(3, device=means.device))

    visibility_filter = out.get("visibility_filter")
    if visibility_filter is not None:
        raw_max_per_gaussian = raw_max_per_gaussian[visibility_filter]

    weights = torch.ones_like(raw_max_per_gaussian)
    return raw_max_per_gaussian.detach().cpu(), weights.detach().cpu()


def compute_overshoot_stats(gaussians, scene, pipe, cameras) -> dict:
    all_raw = []
    all_w = []
    for cam in cameras:
        raw, w = _raw_overshoot_and_weights_for_camera(cam, gaussians, pipe)
        all_raw.append(raw)
        all_w.append(w)
    raw_max_per_sample = torch.cat(all_raw)
    weights = torch.cat(all_w)
    stats = compute_overshoot_stats_from_raw(raw_max_per_sample, weights)
    stats["weighted_frac_approximation_note"] = WEIGHTED_FRAC_APPROXIMATION_NOTE
    return stats


def main():
    from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint

    parser = argparse.ArgumentParser(description="Phase 0: SH-clamp overshoot statistics")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    gaussians, scene, pipe = load_checkpoint(args.model_path, args.model_layout)
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    stats = compute_overshoot_stats(gaussians, scene, pipe, cameras)
    stats["model_path"] = args.model_path
    stats["n_cameras"] = len(cameras)

    with open(args.output_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
