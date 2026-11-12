"""Frozen-checkpoint diagnostic (Phase 1.5, "conflict backlog" experiment,
task #99): tests whether removing recently-inserted source-depth Gaussians
from an existing mid-trough checkpoint recovers PSNR on the same fixed real
test-holdout views, with ZERO optimizer steps.

Motivation: the fr3 PSNR trough (11-13dB through most of a 52000-iteration
run) recovers sharply (+7dB) in the final ~2000 iterations, right after
source-depth insertion stops. Two prior hypotheses (insertion-anchor
gradient suppression, D-vs-ED depth-loss coverage bias) both came back
null/near-null on real GPU runs. The current best hypothesis is a "conflict
backlog" model: continuous insertion introduces some fraction of
geometrically bad/conflicting Gaussians faster than the optimizer's cleanup
rate (move/suppress-opacity/kill) can drain them, so a backlog of bad
geometry accumulates during the mapping phase and drains once insertion
stops.

This script tests the model's PUREST prediction, with no training at all:
render the SAME fixed test cameras against a mid-trough checkpoint, at
successive removal cutoffs on "recently inserted" Gaussians. If removing
recent insertions recovers PSNR by itself, that's strong evidence for the
conflict-backlog model. If it does nothing (or removal only helps once ALL
inserted Gaussians -- not just recent ones -- are gone), a different model
is needed.

IMPORTANT CAVEAT discovered while writing this script: birth_frame/anchor_iter
(the per-Gaussian metadata this diagnostic depends on) are NOT persisted in
the point_cloud/iteration_N/point_cloud.ply format -- scene/gsplat_model.py's
load_ply() -> _reset_running_state() unconditionally zeros both buffers on
load. They ARE persisted in the optimizer checkpoint (chkpntN.pth), via
GsplatGaussianModel.capture()/restore()'s streaming_state dict. This script
therefore loads from chkpntN.pth (torch.load + gaussians.restore(model_args,
opt), the same pattern train.py's own checkpoint-resume path uses at
train.py:438-439), NOT from render_report.py's load_gaussians()/load_ply()
path that Task 1's sibling diagnostic used -- Task 1 didn't need birth_frame/
anchor_iter so this gap was invisible there.

Camera reconstruction (_camera_json_to_minicam + the cx/cy fix + frame.index
lookup convention) is otherwise reused verbatim from
scripts/diagnose_d_vs_ed_depth_bias.py -- read that script first.

Run inside the container:
  docker compose run --rm train python scripts/diagnose_cohort_removal_psnr.py \
    --model_path output/w3_champion_fr3_baseline_20260810 \
    --iteration 44000
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_report import _load_cfg_args, _camera_json_to_minicam  # noqa: E402
from eligsir.io.sources.registry import make_frame_source  # noqa: E402


def _build_camera(entry: dict, frame) -> "object":
    """MiniCam from the saved test_cameras.json entry -- same construction
    as diagnose_d_vs_ed_depth_bias.py's _build_camera, minus the depth-path
    wiring (this script only needs RGB, not sensor depth)."""
    cam = _camera_json_to_minicam(entry)
    cam.fx = float(entry["fx"])
    cam.fy = float(entry["fy"])
    cam.cx = float(entry["cx"])
    cam.cy = float(entry["cy"])
    return cam


def _load_gaussians_from_checkpoint(model_path: str, iteration: int, cfg_args):
    """Load a GsplatGaussianModel with real birth_frame/anchor_iter buffers
    restored from chkpntN.pth (NOT the PLY path -- see module docstring)."""
    from scene import GsplatGaussianModel

    chk_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(
            f"No optimizer checkpoint at {chk_path} -- this diagnostic needs "
            f"chkpntN.pth (birth_frame/anchor_iter are not persisted in the "
            f"point_cloud/iteration_N/point_cloud.ply format, see module docstring)"
        )
    # Streaming checkpoints write a 3-tuple (state, iteration, meta) -- confirmed
    # against both eligsir/mapping/integration/checkpoint_io.py's
    # _save_streaming_checkpoint (the regular periodic writer) and
    # train_streaming.py:1480's SIGTERM emergency-checkpoint handler, both of
    # which do torch.save((state, iteration, meta), tmp). This differs from
    # train.py:438's legacy 2-tuple (model_args, first_iter) format used by
    # train.py's own checkpoint-resume path.
    loaded = torch.load(chk_path, weights_only=False)
    if len(loaded) == 3:
        model_args, saved_first_iter, _meta = loaded
    else:
        model_args, saved_first_iter = loaded
    assert int(saved_first_iter) == int(iteration), (
        f"Checkpoint's own saved iteration ({saved_first_iter}) does not match "
        f"--iteration ({iteration}) -- wrong checkpoint file or a stale copy; "
        f"age computation below would be silently wrong against the wrong baseline."
    )
    sh_degree = int(getattr(cfg_args, "sh_degree", 3))
    gaussians = GsplatGaussianModel(sh_degree)
    gaussians.restore(model_args, cfg_args)
    print(f"[diagnose] loaded {chk_path} (checkpoint's own saved first_iter={saved_first_iter}, "
          f"requested --iteration={iteration})")
    return gaussians


def _masked_copy(gaussians, mask: torch.Tensor):
    """Return a shallow copy of `gaussians` whose .params dict is masked by
    `mask` (a boolean tensor over the N dimension). render() only ever reads
    pc.get_xyz / get_scaling / _rotation / get_rotation / get_opacity /
    active_sh_degree / _features_dc / _features_rest -- all properties that
    index straight into pc.params -- so swapping .params on a shallow copy is
    sufficient; no need to touch scene/gsplat_model.py itself (verified by
    reading eligsir/mapping/renderer/gsplat_backend.py's render()).
    """
    sub = copy.copy(gaussians)
    sub.params = {name: tensor[mask] for name, tensor in gaussians.params.items()}
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--max_cameras", type=int, default=0,
                         help="0 = use all real test-holdout cameras")
    args = parser.parse_args()

    model_path = args.model_path
    cfg_args = _load_cfg_args(model_path)

    gaussians = _load_gaussians_from_checkpoint(model_path, args.iteration, cfg_args)
    n_total = int(gaussians.get_xyz.shape[0])
    birth_frame = gaussians.birth_frame.detach()
    anchor_iter = gaussians.anchor_iter.detach()
    n_inserted = int((birth_frame > 0).sum().item())
    print(f"[diagnose] checkpoint has {n_total} total Gaussians, "
          f"{n_inserted} with birth_frame>0 (source-depth inserted), "
          f"{n_total - n_inserted} bootstrap/MCMC-grown (birth_frame==0)")
    if n_inserted == 0:
        print("[diagnose] WARNING: zero inserted Gaussians in this checkpoint -- "
              "either depth insertion never ran, or birth_frame wasn't populated. "
              "The removal-level experiment below will be a no-op at every cutoff.")

    age = (int(args.iteration) - anchor_iter.long()).clamp(min=0)

    inserted_mask = birth_frame > 0
    removal_levels = []

    def add_level(name, remove_mask):
        keep_mask = ~remove_mask
        removal_levels.append((name, remove_mask, keep_mask))

    add_level("full_map_baseline", torch.zeros(n_total, dtype=torch.bool, device=birth_frame.device))
    for cutoff in (200, 500, 1000, 2000):
        remove = inserted_mask & (age < cutoff)
        add_level(f"remove_age_lt_{cutoff}", remove)
    add_level("remove_all_inserted", inserted_mask)
    # NOTE: an earlier version of this script also added a "mcmc_only" level
    # here with the identical mask (`inserted_mask`) -- that label was wrong:
    # it did NOT isolate MCMC-grown Gaussians from the bootstrap population,
    # since everything with birth_frame==0 (bootstrap AND MCMC-grown alike,
    # per the print statement above) stays in the kept set either way. It was
    # just a second name for "remove_all_inserted"'s exact same result
    # (2026-08-11 review finding). Removed rather than renamed -- there is no
    # value in keeping two identically-masked levels, and this checkpoint
    # format has no provenance metadata to actually separate bootstrap from
    # MCMC-grown Gaussians (both share birth_frame==0; see the module
    # docstring's IMPORTANT CAVEAT above for why birth_frame/anchor_iter are
    # even available here at all -- they're checkpoint-only, not PLY-persisted).

    for name, remove_mask, keep_mask in removal_levels:
        print(f"[diagnose] level={name}: remove {int(remove_mask.sum().item())}, "
              f"keep {int(keep_mask.sum().item())}")

    test_cameras_path = os.path.join(model_path, "test_cameras.json")
    if not os.path.exists(test_cameras_path):
        raise FileNotFoundError(f"{test_cameras_path} not found -- this run has no real test holdout")
    with open(test_cameras_path) as f:
        test_entries = json.load(f)
    if args.max_cameras > 0:
        test_entries = test_entries[: args.max_cameras]
    print(f"[diagnose] {len(test_entries)} real test-holdout cameras loaded from {test_cameras_path}")

    print(f"[diagnose] reconstructing source frame order from {cfg_args.source_path}")
    frame_source = make_frame_source(cfg_args.source_path, cfg_args)
    frames = frame_source.get_all() if hasattr(frame_source, "get_all") else list(frame_source)
    frame_by_index = {int(getattr(f, "index")): f for f in frames if f is not None}
    print(f"[diagnose] {len(frames)} source frames reconstructed")

    # Same index-mapping pose sanity check as Task 1's diagnostic.
    n_pose_checked = 0
    for entry in test_entries:
        source_idx = int(entry["img_name"])
        frame = frame_by_index.get(source_idx)
        if frame is None or frame.c2w is None:
            continue
        pos_json = np.asarray(entry["position"], dtype=np.float64)
        pos_frame = np.asarray(frame.c2w[:3, 3], dtype=np.float64)
        if not np.allclose(pos_json, pos_frame, atol=1e-3):
            raise AssertionError(
                f"Index-mapping check FAILED at source_idx={source_idx}: "
                f"cameras.json position={pos_json} vs frame_by_index[{source_idx}].c2w translation={pos_frame}."
            )
        n_pose_checked += 1
    print(f"[diagnose] index-mapping pose check PASSED for {n_pose_checked}/{len(test_entries)} test cameras")

    from PIL import Image as _PILImage
    from gaussian_renderer import render as render_fn

    bg_color = [1.0, 1.0, 1.0] if getattr(cfg_args, "white_background", False) else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_pipe_args = argparse.Namespace(**vars(cfg_args))
    render_pipe_args.gsplat_sparse_grad = False

    # Resolve (camera, gt_rgb) pairs once -- reused across all removal levels
    # so every level renders the exact same view set against the exact same
    # ground truth.
    view_pairs = []
    n_skipped_no_rgb = 0
    for entry in test_entries:
        source_idx = int(entry["img_name"])
        frame = frame_by_index.get(source_idx)
        if frame is None or frame.rgb_path is None or not os.path.exists(frame.rgb_path):
            n_skipped_no_rgb += 1
            continue
        cam = _build_camera(entry, frame)
        gt = np.array(_PILImage.open(frame.rgb_path).convert("RGB")).astype(np.float32) / 255.0
        view_pairs.append((source_idx, cam, gt))
    print(f"[diagnose] {len(view_pairs)} usable test views resolved "
          f"(skipped {n_skipped_no_rgb} for missing rgb_path)")
    if not view_pairs:
        raise RuntimeError("No usable test views -- cannot report PSNR")

    results = []
    baseline_mean_psnr = None
    with torch.no_grad():
        for name, remove_mask, keep_mask in removal_levels:
            n_removed = int(remove_mask.sum().item())
            n_remaining = int(keep_mask.sum().item())
            sub_gaussians = _masked_copy(gaussians, keep_mask)

            per_view_psnr = []
            for source_idx, cam, gt in view_pairs:
                pkg = render_fn(cam, sub_gaussians, render_pipe_args, background)
                rendered = pkg["render"].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
                mse = float(np.mean((rendered - gt) ** 2))
                view_psnr = float(-10.0 * np.log10(mse)) if mse > 0 else float("inf")
                per_view_psnr.append(view_psnr)

            finite = [p for p in per_view_psnr if np.isfinite(p)]
            mean_psnr = float(np.mean(finite)) if finite else float("nan")
            if name == "full_map_baseline":
                baseline_mean_psnr = mean_psnr
            delta = (mean_psnr - baseline_mean_psnr) if baseline_mean_psnr is not None else None

            print(f"[diagnose] level={name:24s} removed={n_removed:7d} remaining={n_remaining:7d} "
                  f"mean_PSNR={mean_psnr:8.4f}dB delta_vs_baseline="
                  f"{('%+.4f' % delta) if delta is not None else '  n/a'}dB")

            results.append({
                "removal_level": name,
                "n_gaussians_removed": n_removed,
                "n_gaussians_remaining": n_remaining,
                "mean_psnr": mean_psnr,
                "n_views_finite": len(finite),
                "n_views_total": len(per_view_psnr),
                "delta_vs_baseline_db": delta,
            })

    report_dir = ".superpowers/sdd/2026-08-11-d-vs-ed-depth-loss-fix"
    os.makedirs(report_dir, exist_ok=True)
    out_json = os.path.join(
        report_dir,
        f"task99-cohort-removal-psnr-iter{args.iteration}.json",
    )
    with open(out_json, "w") as f:
        json.dump({
            "model_path": model_path,
            "iteration": args.iteration,
            "n_total_gaussians": n_total,
            "n_inserted_gaussians": n_inserted,
            "n_test_views_used": len(view_pairs),
            "n_test_views_skipped_no_rgb": n_skipped_no_rgb,
            "removal_levels": results,
        }, f, indent=2)
    print(f"\n[diagnose] wrote {out_json}")


if __name__ == "__main__":
    main()
