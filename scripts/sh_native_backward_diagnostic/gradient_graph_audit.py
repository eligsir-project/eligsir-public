import argparse
import json

import torch

from eligsir.mapping.renderer.gsplat_backend import render
from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint


def _one_backward_pass(gaussians, scene, pipe, cam, sh_impl):
    """Zero any existing .grad, render one camera with the given sh_impl,
    compute a simple L1 photometric loss against ground truth, backward.
    Returns (means_grad, sh0_grad_norm, shN_grad_norm).

    Isolation note: .grad is explicitly reset to None both before and
    after each pass, so no gradient can leak or accumulate between the
    "python" and "gsplat" backward calls on the same Gaussian model --
    each pass starts from a clean .grad=None state and its result tensors
    are detached+cloned before the second pass mutates .grad again.
    """
    for param in (gaussians.get_xyz, gaussians._features_dc, gaussians._features_rest):
        if param.grad is not None:
            param.grad = None

    pipe.sh_impl = sh_impl
    background = torch.zeros(3, device=gaussians.get_xyz.device)
    out = render(cam, gaussians, pipe, background)["render"]
    gt = cam.original_image[0:3, :, :].to(out.device)
    loss = (out - gt).abs().mean()
    loss.backward()

    means_grad = gaussians.get_xyz.grad.detach().clone()
    sh0_grad_norm = gaussians._features_dc.grad.detach().norm().item()
    shN_grad_norm = gaussians._features_rest.grad.detach().norm().item()

    gaussians.get_xyz.grad = None
    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None

    return means_grad, sh0_grad_norm, shN_grad_norm


def compute_gradient_audit(
    model_path: str,
    model_layout: str,
    max_cameras: int = 1,
    sh_color_clamp_mode: str = "clamped",
) -> dict:
    """
    sh_color_clamp_mode: forwarded onto `pipe.sh_color_clamp_mode` before
    either backward pass runs. Default "clamped" preserves this script's
    original (Task 2) behavior. That default is a KNOWN CONFOUND on this
    checkpoint when compared against sh_impl="gsplat": native gsplat's SH
    op applies its own internal (lower-only) clamp regardless of this
    setting, so "clamped" makes the sh_impl="python" and sh_impl="gsplat"
    passes render measurably different forward images (max abs diff
    ~0.045 on the fr3 champion checkpoint) before backward is even run,
    contaminating the gradient comparison with an unrelated forward
    difference. Passing sh_color_clamp_mode="lower_only" (matching native
    gsplat's real clamp behavior on both sides) removes this confound --
    that is the deconfounded, decisive measurement for Phase B's actual
    question (whether the backward passes themselves agree).
    """
    gaussians, scene, pipe = load_checkpoint(model_path, model_layout)
    pipe.sh_color_clamp_mode = sh_color_clamp_mode
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    cam = cameras[0]

    for p in (gaussians.get_xyz, gaussians._features_dc, gaussians._features_rest):
        p.requires_grad_(True)

    means_grad_python, sh0_norm_python, shN_norm_python = _one_backward_pass(
        gaussians, scene, pipe, cam, "python"
    )
    means_grad_gsplat, sh0_norm_gsplat, shN_norm_gsplat = _one_backward_pass(
        gaussians, scene, pipe, cam, "gsplat"
    )

    means_grad_max_abs_diff = (means_grad_python - means_grad_gsplat).abs().max().item()

    return {
        "means_grad_norm_python": means_grad_python.norm().item(),
        "means_grad_norm_gsplat": means_grad_gsplat.norm().item(),
        "sh0_grad_norm_python": sh0_norm_python,
        "sh0_grad_norm_gsplat": sh0_norm_gsplat,
        "shN_grad_norm_python": shN_norm_python,
        "shN_grad_norm_gsplat": shN_norm_gsplat,
        "means_grad_max_abs_diff": means_grad_max_abs_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gradient-graph audit: custom eval_sh vs native gsplat SH backward, real checkpoint"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument(
        "--sh_color_clamp_mode",
        default="clamped",
        help="Forwarded to pipe.sh_color_clamp_mode before both backward passes. "
        "Default 'clamped' preserves this script's original behavior. "
        "'lower_only' matches native gsplat's real clamp behavior and removes "
        "the forward-image confound documented in compute_gradient_audit's docstring.",
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    result = compute_gradient_audit(
        args.model_path, args.model_layout, sh_color_clamp_mode=args.sh_color_clamp_mode
    )
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
