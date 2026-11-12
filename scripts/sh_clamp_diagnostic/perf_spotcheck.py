import argparse
import time

import torch

from scripts.sh_clamp_diagnostic.load_checkpoint import load_checkpoint
from scripts.sh_clamp_diagnostic.phase1_renderer_ablation import render_variant


def timed_render(gaussians, scene, pipe, cameras, sh_impl, warmup=2, reps=10):
    render_variant(gaussians, scene, pipe, "clamped", cameras[:warmup], sh_impl=sh_impl)
    torch.cuda.synchronize()
    start = time.perf_counter()
    render_variant(gaussians, scene, pipe, "clamped", cameras[:reps], sh_impl=sh_impl)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / reps


def main():
    parser = argparse.ArgumentParser(description="Informal forward-render timing spot-check: python vs gsplat SH backend")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    gaussians, scene, pipe = load_checkpoint(args.model_path, args.model_layout)
    cameras = scene.getTestCameras() or scene.getTrainCameras()
    print(f"n_gaussians={gaussians.get_xyz.shape[0]}")

    t_python = timed_render(gaussians, scene, pipe, cameras, "python", reps=args.reps)
    t_gsplat = timed_render(gaussians, scene, pipe, cameras, "gsplat", reps=args.reps)

    print(f"python  sh_impl: {t_python * 1000:.3f} ms/frame")
    print(f"gsplat  sh_impl: {t_gsplat * 1000:.3f} ms/frame")
    print(f"speedup (python/gsplat): {t_python / t_gsplat:.3f}x")


if __name__ == "__main__":
    main()
