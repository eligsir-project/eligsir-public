"""Frozen-checkpoint diagnostic for Phase 1.5 experiment #1 (D-vs-ED depth-loss
fix): renders D, ED, and alpha for a fixed set of real fr3 test cameras on an
existing trained checkpoint, bins pixels by alpha, and reports signed depth
bias per bin for both D and ED against sensor depth. Read-only measurement --
does not modify any production code, no gradients computed.

Reuses:
  - render_report.py's checkpoint-loading pattern (_load_cfg_args,
    load_gaussians, _camera_json_to_minicam) -- the standalone post-hoc
    render/eval module this codebase already has for exactly this kind of
    "reconstruct a GaussianModel + camera list from a saved model_path"
    task.
  - eligsir/mapping/integration/depth_helpers.py's _get_sensor_depth()
    for ground-truth depth loading (lazy-loads + caches from disk, same
    function the training loop itself uses for depth supervision).
  - eligsir.io.sources.registry.make_frame_source() to reconstruct the
    ORDERED source-frame list the run was trained against (same factory
    train_streaming.py uses), so a test camera's stored source index maps
    back to a real on-disk depth file. ``test_cameras.json``'s ``img_name``
    field is set by ``scene/streaming_scene.py``'s ``_frame_to_camera`` as
    ``f"{frame.index:06d}"`` -- i.e. it genuinely IS the frame's own
    ``.index`` attribute (assigned pre-filter, via ``enumerate(assocs)`` in
    ``TUMFrameSource``), NOT necessarily its position in the post-filter
    ``frames`` list this script reconstructs (those two can diverge on any
    dataset where a frame gets dropped for a missing rgb file or a
    non-finite pose). This script therefore looks a frame up by matching
    ``frame.index == img_name`` (a dict keyed by ``.index``), not by
    ``frames[int(img_name)]`` list position -- correct in general, and
    additionally verified per-camera below via an independent pose check.

NOTE (2026-08-11): the task brief this script implements was written against
a pre-Task-2 renderer, where the ED-equivalent quantity was only available
via the render_normals=True / rendered_normals_depth diagnostic path. Task 2
has since landed (see gaussian_renderer/gsplat_backend.py's render_depth
branch): a plain render(..., render_depth=True) call now directly returns
both result["rendered_depth"] (D) and result["rendered_depth_ed"] (ED) from
ONE undetached rasterization pass -- this script uses that path instead, since
it is simpler and (being un-detached) closer to what training actually sees.
Also note D is now DEFINED as ED * alpha (not independently rasterized), so
its bias numbers below are arithmetic, not new empirical evidence -- see the
in-script comment at the reporting step.

Run inside the container:
  docker compose run --rm train python scripts/diagnose_d_vs_ed_depth_bias.py \
    --model_path output/w3_champion_fr3_baseline_20260810 \
    --iteration 52000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_report import _load_cfg_args, load_gaussians, _camera_json_to_minicam  # noqa: E402
from eligsir.io.sources.registry import make_frame_source  # noqa: E402
from eligsir.mapping.integration.depth_helpers import _get_sensor_depth  # noqa: E402


def _build_camera(entry: dict, frame) -> "object":
    """MiniCam from the saved test_cameras.json entry, wired up so
    _get_sensor_depth() can lazy-load this camera's real on-disk depth file.
    """
    cam = _camera_json_to_minicam(entry)
    # _camera_json_to_minicam only carries fx/fy through into FoVx/FoVy and
    # drops cx/cy entirely -- make_gsplat_camera_tensors() then falls back to
    # cx=width/2, cy=height/2 for any camera missing these attrs (see
    # scene/cameras.py:make_gsplat_camera_tensors). For fr3 the real
    # principal point (320.1, 247.6) differs from the width/2,height/2
    # fallback (320.0, 240.0) by ~7.6px vertically -- explicitly restore the
    # real intrinsics here so the render is pixel-aligned with the sensor
    # depth map it gets differenced against.
    cam.fx = float(entry["fx"])
    cam.fy = float(entry["fy"])
    cam.cx = float(entry["cx"])
    cam.cy = float(entry["cy"])
    cam._streaming_depth_path = frame.depth_path if frame is not None else None
    cam._streaming_depth_scale = frame.depth_scale if frame is not None else 1000.0
    cam._sensor_depth_cache = None
    cam._streaming_frame = None
    return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--alpha_bins",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        help="Bin edges for alpha, e.g. [0, 0.01, 0.05, ...]",
    )
    parser.add_argument("--max_cameras", type=int, default=0,
                         help="0 = use all real test-holdout cameras")
    args = parser.parse_args()

    model_path = args.model_path
    cfg_args = _load_cfg_args(model_path)

    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"No checkpoint at {ply_path}")

    gaussians = load_gaussians(model_path, cfg_args.sh_degree, getattr(cfg_args, "model_layout", "gsplat"), ply_path)

    test_cameras_path = os.path.join(model_path, "test_cameras.json")
    if not os.path.exists(test_cameras_path):
        raise FileNotFoundError(f"{test_cameras_path} not found -- this run has no real test holdout to diagnose against")
    with open(test_cameras_path) as f:
        test_entries = json.load(f)
    if args.max_cameras > 0:
        test_entries = test_entries[: args.max_cameras]
    print(f"[diagnose] {len(test_entries)} real test-holdout cameras loaded from {test_cameras_path}")

    print(f"[diagnose] reconstructing source frame order from {cfg_args.source_path} "
          f"(same factory used at training time)")
    frame_source = make_frame_source(cfg_args.source_path, cfg_args)
    frames = frame_source.get_all() if hasattr(frame_source, "get_all") else list(frame_source)
    print(f"[diagnose] {len(frames)} source frames reconstructed")

    # img_name (test_cameras.json) is frame.index (pre-filter), NOT list
    # position -- see the module docstring. Build the correct lookup: a dict
    # keyed by each frame's own .index attribute. Also report, for
    # informational purposes, whether .index happened to equal list position
    # on this run (it does iff no frame was ever dropped by the rgb-exists/
    # finite-pose filters) -- NOT the reason the lookup below is correct.
    frame_by_index = {int(getattr(f, "index")): f for f in frames if f is not None}
    index_matches_position = all(getattr(f, "index", i) == i for i, f in enumerate(frames))
    print(f"[diagnose] frame.index == list position for every frame: {index_matches_position} "
          f"(informational only -- lookup below is by frame.index, not list position)")

    # Independent verification that this index->frame lookup is correct:
    # both cameras.json (id-1) and frame.c2w (id-2, reparsed straight from
    # the dataset) carry a real C2W translation derived from the same TUM
    # groundtruth.txt / _tum_pose_to_c2w path -- if the lookup were wrong,
    # camera position (from cameras.json) and frame.c2w[:3, 3] would
    # disagree by more than float noise.
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
                f"cameras.json position={pos_json} vs frame_by_index[{source_idx}].c2w translation={pos_frame}. "
                "img_name does not map to the expected frame in the reconstructed frame source -- "
                "do not trust any bias numbers until this is root-caused."
            )
        n_pose_checked += 1
    print(f"[diagnose] index-mapping pose check PASSED for {n_pose_checked}/{len(test_entries)} test cameras "
          f"(cameras.json position matches frame_by_index[source_idx].c2w translation to 1e-3)")

    if test_entries:
        e0 = test_entries[0]
        f0 = frame_by_index.get(int(e0["img_name"]))
        if f0 is not None:
            print(f"[diagnose] intrinsics cross-check: json fx/fy/cx/cy="
                  f"{e0['fx']}/{e0['fy']}/{e0['cx']}/{e0['cy']} vs "
                  f"frame fx/fy/cx/cy={f0.fx}/{f0.fy}/{f0.cx}/{f0.cy}")

    from gaussian_renderer import render as render_fn

    bg_color = [1.0, 1.0, 1.0] if getattr(cfg_args, "white_background", False) else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Pure measurement, no gradients needed -- force the dense (non-sparse-grad)
    # rotation path (pc.get_rotation, normalized) rather than cfg_args'
    # training-time gsplat_sparse_grad=True (which would feed raw,
    # un-normalized pc._rotation into the rasterizer). Everything else about
    # cfg_args (sh_impl, tile_size, near/far plane, eps2d, render_mode, ...)
    # is passed through unchanged so the render matches training as closely
    # as possible.
    render_pipe_args = argparse.Namespace(**vars(cfg_args))
    render_pipe_args.gsplat_sparse_grad = False

    # ------------------------------------------------------------------
    # End-to-end pipeline sanity check: render RGB for a handful of test
    # cameras and compare against ground-truth RGB loaded straight from the
    # reconstructed frame source's rgb_path, and (best-effort) against this
    # run's own comparison/iter_N/report.json per-view PSNR. This is the
    # single strongest piece of evidence that camera reconstruction (pose,
    # intrinsics, index mapping) and the render call are faithful to
    # training-time rendering -- if this doesn't match, the depth-bias
    # numbers below cannot be trusted regardless of how clean they look.
    # ------------------------------------------------------------------
    report_json_path = os.path.join(model_path, "comparison", f"iter_{args.iteration}", "report.json")
    expected_psnr_by_name = {}
    if os.path.exists(report_json_path):
        try:
            with open(report_json_path) as f:
                _report = json.load(f)
            for _row in _report.get("test_per_view", []):
                expected_psnr_by_name[str(_row.get("name"))] = _row.get("psnr")
        except Exception as e:
            print(f"[diagnose] WARNING: could not read {report_json_path} for PSNR cross-check: {e}")
    else:
        print(f"[diagnose] WARNING: {report_json_path} not found -- PSNR sanity check will only "
              f"report absolute values, no cross-check against a prior report.json")

    from PIL import Image as _PILImage

    n_psnr_checked = min(3, len(test_entries))
    psnr_checks = []
    with torch.no_grad():
        for entry in test_entries[:n_psnr_checked]:
            source_idx = int(entry["img_name"])
            frame = frame_by_index.get(source_idx)
            if frame is None or frame.rgb_path is None or not os.path.exists(frame.rgb_path):
                print(f"[diagnose] PSNR sanity check: skipping cam {source_idx} (no rgb_path)")
                continue
            cam = _build_camera(entry, frame)
            pkg = render_fn(cam, gaussians, render_pipe_args, background)
            rendered = pkg["render"].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
            gt = np.array(_PILImage.open(frame.rgb_path).convert("RGB")).astype(np.float32) / 255.0
            mse = float(np.mean((rendered - gt) ** 2))
            psnr = float(-10.0 * np.log10(mse)) if mse > 0 else float("inf")
            expected = expected_psnr_by_name.get(entry["img_name"])
            match_str = ""
            if expected is not None:
                match_str = f" (report.json expected {expected:.4f}dB, delta {psnr - expected:+.4f}dB)"
            print(f"[diagnose] PSNR sanity check cam {source_idx}: computed {psnr:.4f}dB{match_str}")
            psnr_checks.append({"source_idx": source_idx, "computed_psnr": psnr, "expected_psnr": expected})

    d_biases = []
    ed_biases = []
    alphas = []

    n_used = 0
    n_skipped_no_depth_path = 0
    n_skipped_no_sensor_depth = 0

    with torch.no_grad():
        for entry in test_entries:
            source_idx = int(entry["img_name"])
            frame = frame_by_index.get(source_idx)
            if frame is None or frame.depth_path is None:
                n_skipped_no_depth_path += 1
                continue

            cam = _build_camera(entry, frame)
            pkg = render_fn(cam, gaussians, render_pipe_args, background, render_depth=True)

            sensor_d = _get_sensor_depth(cam, cam.image_height, cam.image_width)
            if sensor_d is None:
                n_skipped_no_sensor_depth += 1
                continue
            sensor_d = sensor_d.to(device=pkg["rendered_depth"].device)

            valid = sensor_d > 0
            if not bool(valid.any()):
                continue

            d = pkg["rendered_depth"]
            ed = pkg["rendered_depth_ed"]
            alpha_map = pkg["alpha"].permute(2, 0, 1)  # [H,W,1] -> [1,H,W]

            d_biases.append((d - sensor_d)[valid].detach().cpu())
            ed_biases.append((ed - sensor_d)[valid].detach().cpu())
            alphas.append(alpha_map[valid].detach().cpu())
            n_used += 1

    print(f"[diagnose] rendered {n_used} cameras "
          f"(skipped {n_skipped_no_depth_path} for no depth_path, "
          f"{n_skipped_no_sensor_depth} for no loadable sensor depth)")

    if n_used == 0:
        raise RuntimeError("No cameras produced valid pixels -- cannot report bias")

    d_bias = torch.cat(d_biases).numpy()
    ed_bias = torch.cat(ed_biases).numpy()
    alpha = torch.cat(alphas).numpy()

    edges = np.asarray(sorted(args.alpha_bins), dtype=np.float64)
    bin_idx = np.digitize(alpha, edges[1:-1], right=False)  # len(edges)-1 bins

    # rendered_depth (D) is now DEFINITIONALLY ed * alpha (gsplat_backend.py's
    # Task-2 change), not an independently-measured quantity -- so
    # bias_D = ED*alpha - sensor_d necessarily trends toward -sensor_d as
    # alpha -> 0. The D column below is arithmetic, not new empirical
    # evidence; the real empirical content of this diagnostic is whether ED
    # STAYS metrically accurate (mean bias near 0, low std) at low alpha, and
    # where its std starts blowing up -- that's what the alpha-gate
    # threshold should be chosen from, not from the D column.
    print()
    print("NOTE: D bias is definitional (D = ED*alpha since the Task-2 renderer change), not "
          "independent evidence -- read the ED columns for the real finding.")
    print()
    header = (f"{'alpha_bin':>18} | {'n_px':>10} | {'D bias (m)':>12} | {'ED bias (m)':>12} | "
              f"{'ED bias std':>12} | {'ED median':>12} | {'ED p10':>10} | {'ED p90':>10} | {'n_nonfinite_ed':>15}")
    print(header)
    print("-" * len(header))
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n_px = int(mask.sum())
        lo, hi = edges[b], edges[b + 1]
        if n_px == 0:
            print(f"{lo:>8.3f}-{hi:<8.3f} | {0:>10} | {'--':>12} | {'--':>12} | {'--':>12} | "
                  f"{'--':>12} | {'--':>10} | {'--':>10} | {0:>15}")
            rows.append({"alpha_lo": float(lo), "alpha_hi": float(hi), "n_px": 0,
                         "d_bias_mean": None, "ed_bias_mean": None, "ed_bias_std": None,
                         "ed_bias_median": None, "ed_bias_p10": None, "ed_bias_p90": None,
                         "n_nonfinite_ed": 0})
            continue
        d_slice = d_bias[mask]
        ed_slice_all = ed_bias[mask]
        finite_mask = np.isfinite(ed_slice_all)
        n_nonfinite = int((~finite_mask).sum())
        ed_slice = ed_slice_all[finite_mask]

        d_mean = float(d_slice.mean()) if d_slice.size else float("nan")
        if ed_slice.size:
            ed_mean = float(ed_slice.mean())
            ed_std = float(ed_slice.std())
            ed_median = float(np.median(ed_slice))
            ed_p10 = float(np.percentile(ed_slice, 10))
            ed_p90 = float(np.percentile(ed_slice, 90))
        else:
            ed_mean = ed_std = ed_median = ed_p10 = ed_p90 = float("nan")

        print(f"{lo:>8.3f}-{hi:<8.3f} | {n_px:>10} | {d_mean:>12.4f} | {ed_mean:>12.4f} | "
              f"{ed_std:>12.4f} | {ed_median:>12.4f} | {ed_p10:>10.4f} | {ed_p90:>10.4f} | {n_nonfinite:>15}")
        rows.append({"alpha_lo": float(lo), "alpha_hi": float(hi), "n_px": n_px,
                      "d_bias_mean": d_mean, "ed_bias_mean": ed_mean, "ed_bias_std": ed_std,
                      "ed_bias_median": ed_median, "ed_bias_p10": ed_p10, "ed_bias_p90": ed_p90,
                      "n_nonfinite_ed": n_nonfinite})

    report_dir = ".superpowers/sdd/2026-08-11-d-vs-ed-depth-loss-fix"
    os.makedirs(report_dir, exist_ok=True)
    out_json = os.path.join(
        report_dir,
        f"task1-d-vs-ed-depth-bias-iter{args.iteration}.json",
    )
    with open(out_json, "w") as f:
        json.dump({
            "model_path": model_path,
            "iteration": args.iteration,
            "n_cameras_used": n_used,
            "n_cameras_skipped_no_depth_path": n_skipped_no_depth_path,
            "n_cameras_skipped_no_sensor_depth": n_skipped_no_sensor_depth,
            "rgb_psnr_sanity_check": psnr_checks,
            "alpha_bins": rows,
        }, f, indent=2)
    print(f"\n[diagnose] wrote {out_json}")


if __name__ == "__main__":
    main()
