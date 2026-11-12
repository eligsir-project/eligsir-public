"""Standalone post-training render/eval module.

Renders and evaluates a saved checkpoint on demand, independent of the
training process that produced it. Reuses the same rendering primitives as
the in-training comparison report (`utils/comparison_report.py`) so results
are identical whether generated inline or after the fact.

Trajectory sources (--trajectory):
  recorded  cameras.json (admitted train keyframes) from the model_path root.
  dense     trajectory_full.jsonl -- every ingested frame's pose, live/bag
            runs only, regardless of keyframe-admission outcome.
  test      test_cameras.json + holdout_gt/ -- the real, deterministic test
            holdout with GT, enabling --eval (PSNR/LPIPS/report.json).
  custom    a user-supplied JSON file in the same camera_to_JSON schema
            (--custom_trajectory PATH).
  orbit     a synthetic fly-around path fit to the loaded Gaussian cloud --
            no persisted trajectory needed, works for any checkpoint.

--full-report (issue #401) regenerates the COMPLETE `comparison/iter_N/`
artifact bundle from a frozen checkpoint in one call -- the same
`write_post_training_report` (utils/comparison_report.py) that
train.py/train_streaming.py call inline, just run standalone and after the
fact. Test cameras come from test_cameras.json + holdout_gt/ (always
GT-backed). Train cameras come from --trajectory {recorded,dense}
(cameras.json / trajectory_full.jsonl) and are, by default, re-resolved
against the original --source_path dataset on disk to reattach real GT
RGB/depth (see attach_source_gt) -- this works retroactively on any
finished run whose dataset is still reachable, with no schema change to
the persisted trajectory files. Writes `regeneration_manifest.json`
alongside the other artifacts (checkpoint identity, render config, camera-
set provenance, and both the training run's and this regeneration's own
git revision). Idempotent: a prior manifest in the target directory is
treated as "already regenerated" and skipped unless --force is passed.

NON-DESTRUCTIVE by construction. `write_post_training_report` builds its
report.json purely from the arguments it is handed, so a standalone
regeneration -- which has no training process to pass it `seed_fade=`,
`pose_provenance=`, `ingestion=`, `git_provenance=` or the streaming
scene's `eval_metadata()` -- cannot reproduce those fields. Since
--full-report writes into the run's OWN comparison/iter_N/, that used to
permanently DESTROY several Run-Contract gates (holdout_hash gates every
cross-run comparison; seed_fade gates whether a run's PSNR may be quoted
at all). Three layers now prevent it:
  1. the prior report.json is copied to report.pre_regeneration.json
     BEFORE any rendering starts, so even a crash mid-regeneration cannot
     lose it;
  2. run-property fields are merged forward into the regenerated report
     (see merge_prior_report / CARRY_OVER_FIELDS / NESTED_MERGE_FIELDS);
  3. anything old-only that is deliberately NOT carried (render-derived,
     and therefore potentially stale) is printed as a loud warning and
     listed in the manifest's `report_merge` block -- nothing is ever lost
     silently.
--out_dir DIR sidesteps the question entirely: the bundle is written there
and the source run is only ever read.

Examples:
  python render_report.py -m output/run1 --trajectory test --eval
  python render_report.py -m output/run1 --trajectory dense --mp4 dense.mp4
  python render_report.py -m output/run1 --history --trajectory orbit --mp4 orbit.mp4 --contact-sheet
  python render_report.py -m output/run1 --full-report
  python render_report.py -m output/run1 --full-report --trajectory dense --no-reload_gt
  python render_report.py -m output/run1 --full-report --out_dir /tmp/regen1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import torch

from arguments import PipelineParams


# ---------------------------------------------------------------------------
# Run-context loading (cfg_args, model class, checkpoints)
# ---------------------------------------------------------------------------

def _load_cfg_args(model_path: str) -> argparse.Namespace:
    path = os.path.join(model_path, "cfg_args")
    with open(path) as f:
        cfgfile_string = f.read()
    return eval(cfgfile_string, {"Namespace": argparse.Namespace})


def resolve_checkpoint_cfg(model_path: str, args) -> argparse.Namespace:
    """Load saved renderer settings or require an explicit legacy fallback."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        return _load_cfg_args(model_path)
    if not args.source_holdout_path:
        raise FileNotFoundError(f"cfg_args not found under {model_path}")
    values = {
        "sh_degree": args.checkpoint_sh_degree,
        "model_layout": args.checkpoint_model_layout,
        "rasterize_mode": args.checkpoint_rasterize_mode,
        "white_background": args.checkpoint_white_background,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "legacy source-derived rescore requires explicit frozen-checkpoint "
            "renderer settings; missing: " + ", ".join(missing)
        )
    return argparse.Namespace(
        **values,
        source_path=os.path.abspath(args.source_holdout_path),
        git_commit="unknown_legacy_checkpoint",
        git_branch="unknown_legacy_checkpoint",
        git_dirty="unknown",
    )


def _model_class(model_layout: str):
    from scene import GaussianModel, GsplatGaussianModel
    return GsplatGaussianModel if (model_layout or "gsplat").lower() == "gsplat" else GaussianModel


def _list_checkpoints(model_path: str) -> list:
    """Return (iteration, ply_path) pairs for every point_cloud/iteration_N/,
    sorted by iteration. Frame-milestone dirs (point_cloud/frame_N) are not
    included -- iteration_N is the reliable, evenly-paced cadence."""
    pattern = os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply")
    out = []
    for p in glob.glob(pattern):
        m = re.search(r"iteration_(\d+)", p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _latest_checkpoint(model_path: str):
    ckpts = _list_checkpoints(model_path)
    if not ckpts:
        raise FileNotFoundError(f"No point_cloud/iteration_*/point_cloud.ply found under {model_path}")
    return ckpts[-1]


def load_gaussians(model_path: str, sh_degree: int, model_layout: str, ply_path: str):
    gaussians = _model_class(model_layout)(sh_degree)
    gaussians.load_ply(ply_path)
    return gaussians


# ---------------------------------------------------------------------------
# Trajectory sources -> list of cameras (MiniCam or real Camera JSON entries)
# ---------------------------------------------------------------------------

def _camera_json_to_minicam(entry: dict, znear: float = 0.01, zfar: float = 1000.0,
                             device: str = "cuda"):
    """Build a MiniCam from one camera_to_JSON-schema entry.

    ``device`` defaults to "cuda" (real rendering always needs it) but is
    overridable to "cpu" so pose-reconstruction logic (this function, and the
    C2W/W2C convention it encodes) is unit-testable without a GPU -- the
    downstream renderer still requires CUDA and is out of scope for those
    tests.
    """
    from scene.cameras import MiniCam
    from eligsir.utils.graphics_utils import getProjectionMatrix, focal2fov

    width, height = int(entry["width"]), int(entry["height"])
    fovx = focal2fov(float(entry["fx"]), width)
    fovy = focal2fov(float(entry["fy"]), height)
    rot = np.asarray(entry["rotation"], dtype=np.float32)
    pos = np.asarray(entry["position"], dtype=np.float32)
    # camera_to_JSON (utils/camera_utils.py) builds the real W2C matrix
    # internally, then inverts it before extracting 'rotation'/'position' --
    # so despite the misleading local variable name in that function (it
    # calls the inverted result "W2C"), the JSON fields are actually the
    # CAMERA-TO-WORLD rotation and the camera center (C2W translation).
    # Build C2W here and invert it back to W2C -- treating these fields as
    # W2C directly (as an earlier version of this function did) silently
    # renders with a wrong-but-plausible-looking camera orientation at every
    # pose (position is coincidentally still correct, since C2W translation
    # IS the camera center either way -- only the rotation was wrong).
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos
    w2c = np.linalg.inv(c2w).astype(np.float32)
    wvt = torch.from_numpy(w2c).transpose(0, 1).contiguous().to(device)
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    full = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam = MiniCam(width, height, fovy, fovx, znear, zfar, wvt, full)
    cam.image_name = entry.get("img_name", str(entry.get("id", "")))
    # fx/fy/cx/cy: several downstream consumers (render_report's own overview
    # camera builder, attach_source_gt's frame-index matching) and
    # comparison_report._get_sensor_depth (via image_height/image_width, set
    # by MiniCam itself) expect these -- carry them through when present so a
    # reconstructed MiniCam behaves like a real Camera for those call sites.
    cam.fx = float(entry.get("fx", 0.0)) or None
    cam.fy = float(entry.get("fy", 0.0)) or None
    cam.cx = entry.get("cx")
    cam.cy = entry.get("cy")
    return cam


def load_recorded_trajectory(model_path: str, ply_dir: str = None, device: str = "cuda") -> list:
    """Load the admitted-keyframe trajectory. Prefers the root-level
    cameras.json (kept in sync at checkpoint-interval cadence going forward),
    but falls back to the per-checkpoint cameras.json written alongside every
    point_cloud/iteration_N/point_cloud.ply by _export_ply_snapshot -- that
    one predates the root-sync addition and exists for every historical run.
    """
    path = os.path.join(model_path, "cameras.json")
    if not os.path.exists(path) and ply_dir is not None:
        fallback = os.path.join(ply_dir, "cameras.json")
        if os.path.exists(fallback):
            path = fallback
    if not os.path.exists(path):
        raise FileNotFoundError(f"cameras.json not found under {model_path} (--trajectory recorded needs it)")
    with open(path) as f:
        entries = json.load(f)
    return [_camera_json_to_minicam(e, device=device) for e in entries]


def load_dense_trajectory(model_path: str, device: str = "cuda") -> list:
    path = os.path.join(model_path, "trajectory_full.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"trajectory_full.jsonl not found under {model_path} -- this run predates "
            "dense-trajectory persistence, or is not a live/bag streaming run."
        )
    cams = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cams.append(_camera_json_to_minicam(json.loads(line), device=device))
    return cams


def load_custom_trajectory(path: str, device: str = "cuda") -> list:
    with open(path) as f:
        entries = json.load(f)
    return [_camera_json_to_minicam(e, device=device) for e in entries]


def load_test_cameras(model_path: str, require_exact_gt: bool = False,
                      device: str = "cuda", provenance_out: dict = None):
    """Returns (cams, has_gt). GT is attached from holdout_gt/ when present so
    the returned cams behave like real Camera objects for PSNR/LPIPS.

    ``holdout_gt/`` is a SCORE-BEARING store (issue #639): what is attached
    here becomes ``cam.original_image``, i.e. the very samples PSNR/SSIM/LPIPS
    are computed against. It is loaded through
    ``eligsir.io.holdout_gt_exact``, which verifies the decoded samples
    against a persisted canonical sha256 and refuses anything outside its
    exact-format allowlist -- no visualization artifact can be selected as a
    metric input.

    Pre-#639 runs carry only JPEG q92 GT. Those still load (refusing would
    make every finished run unrenderable) but are marked lossy, and
    ``require_exact_gt=True`` refuses them outright. Pass ``provenance_out``
    (a dict) to receive the aggregated provenance block -- callers must
    surface it, because a post-hoc score against JPEG GT is measurably
    INFLATED (+0.175 dB on output/wave1_verify_fr3, 8/8 views; issue #630).
    """
    path = os.path.join(model_path, "test_cameras.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"test_cameras.json not found under {model_path} (--trajectory test needs it)")
    with open(path) as f:
        entries = json.load(f)
    gt_dir = os.path.join(model_path, "holdout_gt")
    cams = []
    has_gt = os.path.isdir(gt_dir)
    provs = []
    for e in entries:
        cam = _camera_json_to_minicam(e, device=device)
        if has_gt:
            provs.append(_attach_holdout_gt(cam, gt_dir, e.get("id"),
                                            require_exact=require_exact_gt, device=device))
        cams.append(cam)
    if has_gt:
        from eligsir.io.holdout_gt_exact import summarize_provenance
        summary = summarize_provenance(provs)
        if provenance_out is not None:
            provenance_out.clear()
            provenance_out.update(summary)
        if summary.get("lossy_gt_present"):
            print(f"[render_report] WARNING: holdout GT for {model_path} is LOSSY "
                  f"({summary['formats']}). Post-hoc scores against pre-#639 JPEG q92 GT are "
                  f"INFLATED (+0.175 dB measured, issue #630) and are NOT comparable to this "
                  f"run's own inline metrics.", flush=True)
        if summary.get("n_refused"):
            raise RuntimeError(
                f"--require_exact_holdout_gt: {summary['n_refused']}/{summary['n_cameras']} "
                f"holdout GT files are not bit-exact ({summary['formats']}). Re-run the training "
                f"job on post-#639 code, or drop the flag to score against lossy GT knowingly.")
    return cams, has_gt


def load_source_holdout_cameras(
    source_path: str,
    *,
    eval_hold: int,
    initial_frames: int,
    eval_guard: int,
    min_baseline_m: float,
    min_angle_deg: float,
    expected_n_test: int,
    expected_realized_hash: str,
    device: str = "cuda",
):
    """Recover the canonical causal holdout directly from a finite source.

    This is the fail-closed fallback for a frozen streaming checkpoint whose
    PLY and source survive but whose exported ``test_cameras.json`` /
    ``holdout_gt`` bundle does not.  It deliberately reuses the production
    source registry and causal split/camera builders; it is not a second
    evaluator or an approximation based on a legacy report's sampled rows.
    """
    source_path = os.path.abspath(source_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"source-derived holdout path does not exist: {source_path}")
    if int(expected_n_test) <= 0:
        raise ValueError("source-derived holdout requires expected_n_test > 0")
    expected_realized_hash = str(expected_realized_hash or "").strip()
    if not expected_realized_hash:
        raise ValueError(
            "source-derived holdout requires expected_realized_hash"
        )

    from eligsir.io.sources.registry import make_frame_source
    from eligsir.runtime.causal_evaluation import (
        build_causal_holdout_cameras,
        causal_holdout_plan,
    )
    from eligsir.utils.baseline_comparison import _camera_set_hash

    source_args = argparse.Namespace(
        source_path=source_path,
        model_path="",
        streaming_max_frames=0,
        streaming_frame_stride=1,
        streaming_resolution=1,
        streaming_full_res_frame_indices="",
        tum_frame_stride=1,
        tum_association_max_dt=0.03,
        tum_sequence="",
        streaming_eval_hold=int(eval_hold),
        streaming_initial_frames=int(initial_frames),
        streaming_eval_guard=int(eval_guard),
        streaming_eval_min_baseline_m=float(min_baseline_m),
        streaming_eval_min_angle_deg=float(min_angle_deg),
    )
    source = make_frame_source(source_path, source_args)
    get_all = getattr(source, "get_all", None)
    frames = tuple(get_all() if callable(get_all) else source)
    plan = causal_holdout_plan(frames, source_args)
    cameras = build_causal_holdout_cameras(plan, data_device=device)
    if len(cameras) != int(expected_n_test):
        raise ValueError(
            "source-derived holdout count mismatch: "
            f"expected {int(expected_n_test)}, realized {len(cameras)}"
        )
    realized_hash = _camera_set_hash(cameras)
    if realized_hash != expected_realized_hash:
        raise ValueError(
            "source-derived realized holdout hash mismatch: "
            f"expected {expected_realized_hash}, realized {realized_hash}"
        )
    metadata = dict(plan.eval_metadata)
    metadata["realized_holdout_hash"] = realized_hash
    metadata["scored_holdout_hash"] = realized_hash
    metadata["source_recovery"] = "canonical_finite_source_reconstruction"
    return cameras, metadata


def validate_source_rescore_destination(model_path: str, out_dir: str) -> str:
    """Require a fresh output directory outside the immutable source run."""
    model_path = os.path.abspath(model_path)
    out_dir = os.path.abspath(out_dir)
    if os.path.commonpath((model_path, out_dir)) == model_path:
        raise ValueError(
            "source-derived checkpoint rescore output must be outside the source run"
        )
    if os.path.exists(out_dir):
        raise FileExistsError(
            f"source-derived checkpoint rescore output already exists: {out_dir}"
        )
    return out_dir


def _attach_holdout_gt(cam, gt_dir: str, cam_id, require_exact: bool = False,
                       device: str = "cuda") -> dict:
    """Attach exact holdout GT RGB (+ sensor depth) to a pose-only MiniCam.

    Returns the per-camera provenance dict. The device move is deliberately
    the LAST step and parameterized, so the bit-exactness of the decode is
    testable on a CPU-only surface.
    """
    from eligsir.io.holdout_gt_exact import load_holdout_rgb, load_holdout_depth

    rgb, prov = load_holdout_rgb(gt_dir, cam_id, require_exact=require_exact)
    if rgb is not None:
        cam.original_image = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)

    depth, dprov = load_holdout_depth(gt_dir, cam_id)
    if depth is not None:
        from eligsir.io.holdout_gt_depth import HOLDOUT_GT_DEPTH_SCALE
        cam._sensor_depth_cache = torch.from_numpy(np.ascontiguousarray(depth))[None].to(device)
        # Kept for readers that consult it; the depth attached above is already
        # metres (exact .npy, or the legacy uint16-mm PNG decoded by that
        # module's own helper -- never the dataset's own depth_scale, #358).
        cam._streaming_depth_scale = HOLDOUT_GT_DEPTH_SCALE
    prov["depth"] = dprov
    return prov


# ---------------------------------------------------------------------------
# GT/depth reattachment for pose-only train cameras (issue #401)
#
# cameras.json / trajectory_full.jsonl (camera_to_JSON schema) never carried
# image_path/depth_path -- adding those fields was considered and rejected:
# several Run Contract fingerprints (holdout_hash's siblings) are computed
# over camera_to_JSON dicts, and every existing run's persisted trajectory
# already lacks the fields regardless, so a schema change would only help
# runs launched after this lands. Instead, cameras' `image_name` (always
# f"{frame.index:06d}", see StreamingScene._append_dense_trajectory /
# _make_camera) is re-resolved against the ORIGINAL dataset on disk via the
# same frame-source factory training used (`make_frame_source`) -- this
# works retroactively on any already-finished run whose --source_path is
# still reachable, with no schema change and no risk to hash comparability.
# ---------------------------------------------------------------------------

def attach_source_gt(cams: list, cfg_args, source_path: str = None, max_scan: int = 200000) -> dict:
    """Best-effort: attach real GT RGB + a lazy sensor-depth reference to
    pose-only MiniCams, by re-resolving each camera's admitted source frame
    from the original dataset.

    Never raises. Returns a stats dict ``{"n_total", "n_matched", "reason"}``
    -- ``reason`` is None on a clean full match, otherwise a short
    human-readable note about why some/all cameras were left pose-only. This
    lets a caller (or a report reader) know whether train_contact_sheet /
    train_depth_timeseries.mp4 reflect real GT or were skipped/partial.

    Cameras are matched purely by ``cam.image_name == f"{frame.index:06d}"``
    (parsed as an int) -- the convention every dataset/streaming frame
    source in this repo uses. A scene whose source uses a different naming
    convention (e.g. raw COLMAP filenames) simply yields zero matches.
    """
    stats = {"n_total": len(cams), "n_matched": 0, "reason": None}
    if not cams:
        return stats
    src = source_path or getattr(cfg_args, "source_path", None)
    if not src or not os.path.isdir(src):
        stats["reason"] = f"source_path not reachable on this host: {src!r}"
        return stats
    try:
        from eligsir.io.sources.registry import make_frame_source
        source = make_frame_source(src, cfg_args)
    except Exception as e:
        stats["reason"] = f"make_frame_source({src!r}) failed: {e!r}"
        return stats

    wanted: dict = {}
    for cam in cams:
        name = getattr(cam, "image_name", None)
        try:
            idx = int(name)
        except (TypeError, ValueError):
            continue
        wanted.setdefault(idx, []).append(cam)
    if not wanted:
        stats["reason"] = "no camera had a parseable integer image_name (frame-index convention)"
        return stats

    remaining = set(wanted.keys())
    scanned = 0
    try:
        for frame in source:
            scanned += 1
            idx = int(getattr(frame, "index", -1))
            if idx in remaining:
                n = _attach_frame_gt(wanted[idx], frame)
                stats["n_matched"] += n
                remaining.discard(idx)
            if not remaining or scanned >= max_scan:
                break
    except Exception as e:
        stats["reason"] = f"frame source iteration failed after {scanned} frames: {e!r}"
        return stats

    if remaining:
        note = f"{len(remaining)}/{len(wanted)} frame indices not found in source (scanned {scanned})"
        stats["reason"] = note if stats["reason"] is None else f"{stats['reason']}; {note}"
    return stats


def _attach_frame_gt(cams_for_frame: list, frame) -> int:
    """Attach GT RGB (eager, small) + sensor-depth REFERENCES (lazy -- reuses
    utils.comparison_report._get_sensor_depth's existing loader, never a
    second depth-decode implementation) to every cam mapped to this frame.
    Returns how many cams were successfully attached."""
    from eligsir.io.sources.decode_cache import load_frame_rgb

    n_ok = 0
    img = None
    for cam in cams_for_frame:
        try:
            if img is None:
                pil = load_frame_rgb(frame)
                w = int(getattr(cam, "image_width", pil.width))
                h = int(getattr(cam, "image_height", pil.height))
                if (pil.width, pil.height) != (w, h):
                    pil = pil.resize((w, h))
                arr = np.asarray(pil, dtype=np.float32) / 255.0  # HWC, [0, 1]
                img = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
            cam.original_image = img
            # Mirrors exactly what StreamingScene stamps on a real Camera
            # (scene/streaming_scene.py) -- comparison_report._get_sensor_depth
            # already knows how to lazily load+scale depth from these two
            # attributes, so this file never re-implements that decode.
            cam._streaming_frame = frame
            cam._streaming_depth_path = frame.depth_path
            cam._streaming_depth_scale = float(getattr(frame, "depth_scale", 1000.0))
            n_ok += 1
        except Exception as e:
            print(f"[render_report] GT attach failed for frame {getattr(frame, 'index', '?')}: {e!r}")
    return n_ok


def _synthetic_orbit_seed_cams(n: int, width: int = 1280, height: int = 720,
                                fov_deg: float = 60.0, elevation_deg: float = 20.0) -> list:
    """n seed MiniCams orbiting azimuthally at a fixed elevation, arbitrary
    translation. Only their ROTATION is used downstream (by
    utils.comparison_report._make_overview_cameras, which recomputes
    translation to frame the loaded Gaussian cloud) -- this just needs to
    supply n distinct viewing directions."""
    from scene.cameras import MiniCam
    from eligsir.utils.graphics_utils import getProjectionMatrix

    fov = math.radians(fov_deg)
    elev = math.radians(elevation_deg)
    cams = []
    for i in range(n):
        az = 2.0 * math.pi * i / max(n, 1)
        cy, sy = math.cos(az), math.sin(az)
        ce, se = math.cos(elev), math.sin(elev)
        fwd = np.array([cy * ce, se, sy * ce], dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(fwd, world_up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        # Columns = camera axes (X, Y, Z=forward) expressed in world coords (c2w rotation).
        Rcw = np.stack([right, true_up, fwd], axis=1).astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = Rcw.T
        wvt = torch.from_numpy(w2c).transpose(0, 1).contiguous().cuda()
        proj = getProjectionMatrix(znear=0.01, zfar=1000.0, fovX=fov, fovY=fov).transpose(0, 1).cuda()
        full = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cams.append(MiniCam(width, height, fov, fov, 0.01, 1000.0, wvt, full))
    return cams


def build_orbit_trajectory(gaussians, n_frames: int = 90) -> list:
    from eligsir.utils.comparison_report import _make_overview_cameras
    seed_cams = _synthetic_orbit_seed_cams(n_frames)
    return _make_overview_cameras(seed_cams, gaussians, n=n_frames, fov_scale=1.0, margin=1.3)


# ---------------------------------------------------------------------------
# Render modes
# ---------------------------------------------------------------------------

def run_mp4(cams: list, gaussians, render_fn, pipe, background, out_path: str, fps: int) -> None:
    from gaussian_renderer.gsplat_backend import render_batch
    from eligsir.utils.comparison_report import render_camera_path
    render_camera_path(cams, gaussians, render_fn, pipe, background,
                        out_path=out_path, fps=fps, batch_render_fn=render_batch)


def run_contact_sheet(cams: list, gaussians, render_fn, pipe, background, out_path: str,
                       cols: int = 3, cell_width: int = 1600) -> None:
    """Render-only grid (no GT row -- these trajectories have no ground truth).
    Reuses the generic resize/concat/save helpers from comparison_report.py."""
    from eligsir.utils.comparison_report import _render_image, _to_uint8_hwc, _resize_to, _hconcat_with_gap, _save_png
    if not cams:
        print("[render_report] no cameras — skipping contact sheet")
        return
    cells = []
    for cam in cams:
        try:
            img = _render_image(cam, gaussians, render_fn, pipe, background)
            img_u8 = _to_uint8_hwc(img)
            # Never upscale past native render resolution -- only downscale
            # when the sheet's target cell width is smaller than the camera's
            # actual width, so cells stay crisp instead of blurry on upscale.
            if cell_width < img_u8.shape[1]:
                img_u8 = _resize_to(img_u8, cell_width)
            cells.append(img_u8)
        except Exception as e:
            print(f"[render_report] contact-sheet render failed for {getattr(cam, 'image_name', '?')}: {e}")
    if not cells:
        print("[render_report] all contact-sheet renders failed — skipping")
        return
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    row_imgs = [_hconcat_with_gap(row) for row in rows]
    max_w = max(r.shape[1] for r in row_imgs)
    padded = []
    for r in row_imgs:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    sheet = np.concatenate(padded, axis=0)
    _save_png(out_path, sheet)
    print(f"[render_report] {out_path} written ({len(cells)} views)")


def run_eval(model_path: str, iteration: int, gaussians, test_cams: list, render_fn, pipe, background,
             subdir: str, *, eval_metadata: dict = None,
             test_metrics_max: int = None, scene_source: str = None) -> dict:
    from gaussian_renderer.gsplat_backend import render_batch
    from eligsir.utils.comparison_report import write_post_training_report
    return write_post_training_report(
        model_path=model_path,
        iteration=iteration,
        gaussians=gaussians,
        train_cams=[],
        test_cams=test_cams,
        render_fn=render_fn,
        pipe=pipe,
        background=background,
        log_prefix="render_report",
        subdir=subdir,
        batch_render_fn=render_batch,
        eval_metadata=eval_metadata,
        test_metrics_max=test_metrics_max,
        scene_source=scene_source,
    )


def run_full_report(model_path: str, iteration: int, gaussians, train_cams: list, test_cams: list,
                     render_fn, pipe, background, subdir: str) -> dict:
    """Regenerate the COMPLETE `comparison/iter_N/` artifact bundle (per
    CLAUDE.md's "Standard benchmark artifacts" contract) from a frozen
    checkpoint -- the single, unforked call to write_post_training_report
    that produces every sheet/video train.py/train_streaming.py produce
    inline, just after the fact instead of during training.
    """
    from gaussian_renderer.gsplat_backend import render_batch
    from eligsir.utils.comparison_report import write_post_training_report
    return write_post_training_report(
        model_path=model_path,
        iteration=iteration,
        gaussians=gaussians,
        train_cams=train_cams,
        test_cams=test_cams,
        render_fn=render_fn,
        pipe=pipe,
        background=background,
        log_prefix="render_report_full",
        subdir=subdir,
        batch_render_fn=render_batch,
    )


# ---------------------------------------------------------------------------
# Regeneration manifest (issue #401)
# ---------------------------------------------------------------------------

def _find_dot_git(start_dir: str) -> str:
    """Walk up from ``start_dir`` and return the resolved .git DIRECTORY, or
    "" when none is found. Handles the `.git`-is-a-file worktree/submodule
    form (`gitdir: <path>`) -- this repo's agents run inside
    `.claude/worktrees/agent-*`, which is exactly that form."""
    d = os.path.abspath(start_dir)
    while True:
        cand = os.path.join(d, ".git")
        if os.path.isdir(cand):
            return cand
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    line = f.read().strip()
            except OSError:
                return ""
            if line.startswith("gitdir:"):
                p = line.split(":", 1)[1].strip()
                return p if os.path.isabs(p) else os.path.normpath(os.path.join(d, p))
            return ""
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent


def _git_metadata_from_dotgit(start_dir: str) -> dict:
    """Read commit/branch straight out of `.git/` with no `git` binary.

    The training image has no `git` installed, so the subprocess path in
    `_current_git_metadata` always failed there and the manifest recorded a
    bare "unknown" for the PRIMARY (containerized) regeneration path -- i.e.
    the field recorded nothing exactly where it mattered most. This parses
    `HEAD` -> `refs/heads/<branch>` (falling back to `packed-refs`) directly.

    `git_dirty` is genuinely NOT determinable this way (it needs a real index
    vs. worktree diff), so it stays "unknown" -- but with an explicit
    `git_dirty_reason`, never a bare, unexplained "unknown"."""
    meta = {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown",
            "source": "unavailable", "reason": None}
    git_dir = _find_dot_git(start_dir)
    if not git_dir:
        meta["reason"] = f"no .git found at or above {start_dir!r}"
        return meta
    try:
        with open(os.path.join(git_dir, "HEAD")) as f:
            head = f.read().strip()
    except OSError as e:
        meta["reason"] = f"could not read {git_dir}/HEAD: {e!r}"
        return meta

    ref = None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        meta["git_branch"] = ref.split("refs/heads/", 1)[-1]
        commit = ""
        ref_file = os.path.join(git_dir, ref)
        if os.path.exists(ref_file):
            try:
                with open(ref_file) as f:
                    commit = f.read().strip()
            except OSError:
                commit = ""
        if not commit:  # packed-refs fallback
            try:
                with open(os.path.join(git_dir, "packed-refs")) as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            commit = parts[0]
                            break
            except OSError:
                pass
        if commit:
            meta["git_commit"] = commit
    else:
        meta["git_commit"] = head
        meta["git_branch"] = "HEAD (detached)"

    if meta["git_commit"] == "unknown":
        meta["reason"] = f"resolved HEAD ref {ref!r} but found no commit for it under {git_dir}"
        return meta
    meta["source"] = "dotgit"
    meta["reason"] = ("read directly from .git (no `git` binary available -- the training "
                      "image does not ship one)")
    meta["git_dirty_reason"] = ("not determinable without a `git` binary: it requires a real "
                                "index-vs-worktree diff, which .git alone cannot answer")
    return meta


def _current_git_metadata() -> dict:
    """Git commit/branch/dirty of THIS process's checkout, right now -- i.e.
    the code that is about to do the regenerating. Deliberately separate
    from the training run's own recorded git_provenance (cfg_args'
    git_commit/git_branch/git_dirty, set once by train.py at launch, see
    utils.comparison_report.git_provenance_from_args): the two can differ
    whenever code has changed since the checkpoint was produced, and that
    difference is exactly what a regeneration manifest needs to make
    visible (issue #259).

    Prefers the `git` CLI (the only way to learn `git_dirty`); falls back to
    parsing `.git/` by hand when no binary is present (the container case);
    and, failing even that, records an explicit `reason` rather than a bare
    "unknown"."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    meta = {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown",
            "source": "unavailable", "reason": None}
    try:
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=here,
        ).strip()
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=here,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL, cwd=here,
        )
        meta["git_dirty"] = bool(status.strip())
        meta["source"] = "git_cli"
        return meta
    except Exception as e:
        cli_reason = f"`git` CLI unavailable or failed: {e!r}"

    fallback = _git_metadata_from_dotgit(here)
    if fallback.get("source") == "dotgit":
        fallback["reason"] = f"{cli_reason}; {fallback['reason']}"
        return fallback
    fallback["reason"] = f"{cli_reason}; {fallback.get('reason')}"
    return fallback


# ---------------------------------------------------------------------------
# Non-destructive report merge (issue #401 follow-up)
#
# `write_post_training_report` builds its summary from the arguments it is
# GIVEN -- so a standalone regeneration, which has no training process to
# hand it `seed_fade=`, `pose_provenance=`, `ingestion=`, `git_provenance=`
# or the streaming scene's `eval_metadata()`, emits a report.json missing all
# of them. Since --full-report writes into the run's OWN
# comparison/iter_N/, regenerating a finished run used to permanently DESTROY
# those fields. Several are Run-Contract gates (CLAUDE.md): holdout_hash
# gates every cross-run comparison, seed_fade gates whether a run's PSNR may
# be quoted at all.
#
# Fix, in three layers:
#   1. the prior report.json is snapshotted to report.pre_regeneration.json
#      BEFORE any rendering starts (so a crash mid-render cannot lose it),
#   2. run-property fields are merged forward from it (below),
#   3. anything old-only that is NOT merged is reported loudly and recorded
#      in the manifest -- nothing is ever lost silently.
# ---------------------------------------------------------------------------

#: Top-level report.json fields that describe the TRAINING RUN, not this
#: render -- regeneration structurally cannot recompute them, so they are
#: carried forward from the prior report whenever regeneration did not
#: produce a non-None value of its own.
CARRY_OVER_FIELDS = (
    "seed_fade",
    "pose_provenance",
    "git_provenance",
    "ingestion",
    "input_quality",
    "expressiveness",
    "mapping_clocks",
    "depth_insertion",
    "lineage_preserving_setup",
    "multires_cache",
    "bag_replay_report",
    "baseline_comparison",
    "trajectory_coverage",
    "frame_admission",
    "resumed_from_iteration",
    "true_final_iteration",
    "last_ingest_iter",
    "mapping_phase_source",
    "scene_source",
    "scene_fingerprint",
    "partition_denominator_source",
    "partition_max_source_index",
)

#: Fields merged KEY-BY-KEY rather than wholesale. `eval_metadata` is the
#: motivating case: regeneration re-derives realized/scored_holdout_hash from
#: test_cameras.json but has no way to know the selection-rule `holdout_hash`,
#: so a whole-dict overlay would drop it. `geometry` likewise carries
#: training-only sub-blocks (trajectory_coverage, seed_cohort, map_comparison)
#: alongside sub-blocks regeneration genuinely recomputes from the PLY.
NESTED_MERGE_FIELDS = ("eval_metadata", "geometry")

#: eval_metadata hashes regeneration recomputes for itself. If a recomputed
#: value DISAGREES with the training run's, the regenerated report advertises
#: a different comparability fingerprint than the run it claims to describe --
#: the inverse of the bug above, and just as quietly damaging.
_RECOMPUTED_HASH_KEYS = ("realized_holdout_hash", "scored_holdout_hash")


def merge_prior_report(old: dict, new: dict, regeneration_marker: dict = None):
    """Pure, side-effect-free, torch-free: ``(old, new) -> (merged, provenance)``.

    ``new`` (the freshly regenerated report) always wins for anything it
    actually computed. ``old`` contributes only:
      * the CARRY_OVER_FIELDS whose regenerated value is absent or None,
      * per-key fills inside NESTED_MERGE_FIELDS,
      * the two seed_fade top-level mirrors, which are DERIVED from whichever
        seed_fade block ends up in the merged report (never carried
        independently -- that could produce an internally inconsistent
        report).

    Every other old-only key is deliberately NOT carried (it may be a stale
    render-derived value, e.g. a `mean_train_psnr` from a run whose GT this
    regeneration could not reattach) and is instead listed in
    ``provenance["dropped_fields"]`` for the caller to report loudly. The
    verbatim prior report is separately preserved on disk, so "dropped" here
    never means "lost".

    ``regeneration_marker``, when given, is stamped in as
    ``merged["regenerated"]``. A merged report is otherwise
    INDISTINGUISHABLE from a real training-run report -- it carries the
    original holdout_hash, seed_fade verdict and pose_provenance beside a
    mean_test_psnr computed later, by a different code revision, from a
    PLY-loaded model. regeneration_manifest.json records that, but it is a
    separate file, and every consumer in this repo (compare_against_baseline,
    scripts/compare_geometry.py, the ablator ledger, a human quoting a
    number) reads report.json alone -- which is exactly why every comparable
    caveat in this project (n_test_metrics_failed, seed_fade_indeterminate,
    map_reinit_invalidated) lives INSIDE report.json.
    """
    merged = dict(new)
    if regeneration_marker is not None:
        merged["regenerated"] = dict(regeneration_marker)
    prov = {
        "carried_over": [],
        "dropped_fields": [],
        "derived": [],
        "recomputed_hash_mismatch": {},
    }
    if not old:
        prov["freshly_computed"] = sorted(merged.keys())
        return merged, prov

    def _absent(container: dict, key: str) -> bool:
        return key not in container or container.get(key) is None

    for key in CARRY_OVER_FIELDS:
        if key in old and old[key] is not None and _absent(merged, key):
            merged[key] = old[key]
            prov["carried_over"].append(key)

    for key in NESTED_MERGE_FIELDS:
        old_sub, new_sub = old.get(key), merged.get(key)
        if not isinstance(old_sub, dict):
            continue
        if not isinstance(new_sub, dict):
            merged[key] = dict(old_sub)
            prov["carried_over"].append(key)
            continue
        sub = dict(new_sub)
        for sk, sv in old_sub.items():
            if sv is None:
                continue
            if sk not in sub or sub[sk] is None:
                sub[sk] = sv
                prov["carried_over"].append(f"{key}.{sk}")
            elif key == "eval_metadata" and sk in _RECOMPUTED_HASH_KEYS and sub[sk] != sv:
                prov["recomputed_hash_mismatch"][sk] = {"prior": sv, "regenerated": sub[sk]}
        merged[key] = sub

    # seed_fade mirrors: derived, never independently carried.
    sf = merged.get("seed_fade")
    if isinstance(sf, dict):
        for mirror, src in (("seed_fade_confounded", "confounded"),
                            ("seed_fade_indeterminate", "indeterminate")):
            val = bool(sf.get(src, False))
            if merged.get(mirror) != val:
                merged[mirror] = val
                prov["derived"].append(mirror)

    prov["dropped_fields"] = sorted(k for k in old if k not in merged)
    prov["carried_over"].sort()
    prov["derived"].sort()
    prov["freshly_computed"] = sorted(k for k in new if new.get(k) is not None)
    return merged, prov


def find_prior_report(model_path: str, out_dir: str, iteration: int):
    """Locate the report.json this regeneration is about to replace/shadow.

    Search order, most-specific first:
      1. ``out_dir/report.json`` -- the file literally about to be overwritten,
      2. ``model_path/comparison/iter_{N}/report.json`` -- the canonical
         location, and the one that matters when --out_dir points elsewhere,
      3. ``model_path/report.json`` -- runs launched with `--benchmark_dir
         model` write there instead (see CLAUDE.md's "never conclude 'no
         report.json' from a run-root glob").

    Returns ``(path_or_None, dict_or_None, reason_or_None)``. A miss is
    reported with an explicit reason rather than left to look like "there was
    nothing to carry over" -- that ambiguity is the same silent-loss class
    this whole merge exists to close.
    """
    candidates = [
        os.path.join(out_dir, "report.json"),
        os.path.join(model_path, "comparison", f"iter_{iteration}", "report.json"),
        os.path.join(model_path, "report.json"),
    ]
    seen = []
    for path in candidates:
        if path in seen:
            continue
        seen.append(path)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            return None, None, f"prior report {path} exists but could not be parsed: {e!r}"
        if isinstance(data, dict):
            return path, data, None
        return None, None, f"prior report {path} is not a JSON object"
    return None, None, ("no prior report.json found at any of: " + ", ".join(seen) +
                        " -- nothing could be carried forward")


def snapshot_prior_report(prior_path: str, out_dir: str) -> str:
    """Copy the prior report.json to ``out_dir/report.pre_regeneration.json``
    BEFORE any rendering happens, so a crash (or an OOM kill) part-way
    through a long GPU regeneration cannot leave the run with neither the
    original nor a complete replacement. Never clobbers an existing backup --
    repeated --force regenerations must not overwrite the true original with
    an already-regenerated one. Returns the backup path ("" on failure)."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    backup = os.path.join(out_dir, "report.pre_regeneration.json")
    if os.path.exists(backup):
        return backup
    if os.path.abspath(prior_path) == os.path.abspath(backup):
        return backup
    try:
        shutil.copy2(prior_path, backup)
    except OSError as e:
        print(f"[render_report] WARNING: could not back up {prior_path} -> {backup}: {e!r}")
        return ""
    return backup


def build_regeneration_manifest(*, model_path: str, iteration: int, ply_path: str, cfg_args,
                                 trajectory_source: str, gt_stats: dict, n_train: int, n_test: int,
                                 test_has_gt: bool, summary: dict, render_width=None, render_height=None,
                                 report_merge: dict = None, out_dir: str = None) -> dict:
    """Pure, side-effect-free manifest builder (unit-testable without
    torch/CUDA) -- see write_regeneration_manifest for the disk-writing
    wrapper. Records everything the issue asks for: checkpoint identity,
    render config, camera set provenance, and code revision (both the
    training run's own recorded revision, when available, and this
    regeneration process's current one -- they can legitimately differ)."""
    from eligsir.utils.comparison_report import git_provenance_from_args

    ply_stat = None
    try:
        st = os.stat(ply_path)
        ply_stat = {"size_bytes": int(st.st_size), "mtime": st.st_mtime}
    except OSError:
        pass

    return {
        "schema": "eligsir.render_report.regeneration_manifest.v1",
        "generated_at_utc": _utc_now_iso(),
        "model_path": os.path.abspath(model_path),
        **({"out_dir": os.path.abspath(out_dir)} if out_dir else {}),
        # Which report.json fields came from the prior report vs. this
        # regeneration -- see merge_prior_report. Always present (even when
        # nothing was found to merge), because "no prior report" and "a prior
        # report with nothing worth carrying" are different facts and a
        # reader must be able to tell them apart.
        "report_merge": dict(report_merge or {"prior_report_path": None,
                                              "reason": "merge not attempted"}),
        "checkpoint": {
            "iteration": iteration,
            "ply_path": ply_path,
            **({"ply_stat": ply_stat} if ply_stat is not None else {}),
        },
        "render_config": {
            "width": render_width,
            "height": render_height,
            "trajectory_fps": summary.get("trajectory_fps"),
        },
        "camera_set": {
            "train_trajectory_source": trajectory_source,
            "n_train": n_train,
            "n_train_with_gt": gt_stats.get("n_matched", 0),
            "train_gt_reason": gt_stats.get("reason"),
            "n_test": n_test,
            "test_has_gt": bool(test_has_gt),
        },
        "code_revision": {
            "training_run": git_provenance_from_args(cfg_args),
            "regeneration": _current_git_metadata(),
        },
        "artifacts": list(summary.get("artifacts", [])),
    }


def merge_report_into_out_dir(out_dir: str, prior_path, prior_report, prior_reason,
                               backup_path: str, summary: dict, *, iteration: int = None,
                               ply_path: str = None) -> dict:
    """Rewrite ``out_dir/report.json`` as prior-merged-with-regenerated, and
    return the manifest-ready provenance block.

    Deliberately re-reads the report.json that write_post_training_report
    actually WROTE rather than merging into the summary dict it returned: the
    file is the artifact of record, and this stays correct even if the writer
    adds fields between building its summary and dumping it.
    """
    report_path = os.path.join(out_dir, "report.json")
    info = {
        "prior_report_path": prior_path,
        "prior_report_backup": backup_path or None,
        "reason": prior_reason,
        "carried_over": [],
        "dropped_fields": [],
        "derived": [],
        "recomputed_hash_mismatch": {},
        "merged": False,
    }
    try:
        with open(report_path) as f:
            regenerated = json.load(f)
    except Exception as e:
        # Fall back to the in-memory summary so a carry-over still happens.
        regenerated = dict(summary or {})
        info["reason"] = f"{info['reason'] or ''} (report.json unreadable at {report_path}: {e!r})".strip()

    marker = {
        "tool": "render_report.py --full-report",
        "generated_at_utc": _utc_now_iso(),
        "iteration": iteration,
        "from_checkpoint": ply_path,
        "prior_report_path": prior_path,
        "prior_report_backup": backup_path or None,
        "code_revision": _current_git_metadata(),
        "note": ("this report.json was regenerated from a frozen checkpoint after the fact; "
                 "its metrics were NOT produced by the training run itself, while the fields "
                 "listed in carried_over were carried forward verbatim from the training "
                 "run's own report"),
    }
    merged, prov = merge_prior_report(prior_report or {}, regenerated, regeneration_marker=marker)
    merged["regenerated"]["carried_over"] = list(prov["carried_over"])
    info.update({k: prov[k] for k in ("carried_over", "dropped_fields", "derived",
                                      "recomputed_hash_mismatch")})
    info["freshly_computed"] = prov.get("freshly_computed", [])
    try:
        with open(report_path, "w") as f:
            json.dump(merged, f, indent=2)
        info["merged"] = True
    except OSError as e:
        print(f"[render_report] WARNING: could not write merged report {report_path}: {e!r}")
        return info

    if info["carried_over"]:
        print(f"[render_report] report.json: carried {len(info['carried_over'])} field(s) "
              f"forward from the prior report: {', '.join(info['carried_over'])}")
    if info["dropped_fields"]:
        print(f"[render_report] WARNING: {len(info['dropped_fields'])} field(s) present in the "
              f"prior report are NOT in the regenerated one and were NOT carried forward "
              f"(they are render-derived and would be stale): {', '.join(info['dropped_fields'])}. "
              f"The verbatim original is preserved at {backup_path or 'NO BACKUP — see above'}.")
    for key, delta in (info["recomputed_hash_mismatch"] or {}).items():
        print(f"[render_report] WARNING: recomputed eval_metadata.{key} DISAGREES with the "
              f"training run's ({delta['regenerated']} vs. prior {delta['prior']}) — this "
              f"regenerated report is NOT holdout-comparable with the original run.")
    failed = merged.get("n_test_metrics_failed")
    if failed:
        print(f"[render_report] WARNING: n_test_metrics_failed={failed} — the regenerated "
              f"mean_test_psnr is an average over a partial, failure-selected subset "
              f"(CLAUDE.md issue #437) and is not comparable to the original run's.")
    return info


def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_regeneration_manifest(target_dir: str, **kwargs) -> str:
    """``target_dir`` (not ``out_dir``): ``build_regeneration_manifest`` takes
    its own ``out_dir=`` kwarg (recorded IN the manifest), so a same-named
    positional here would collide with it."""
    manifest = build_regeneration_manifest(**kwargs)
    path = os.path.join(target_dir, "regeneration_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _get_trajectory(args, model_path: str, gaussians, ply_dir: str = None):
    if args.trajectory == "recorded":
        return load_recorded_trajectory(model_path, ply_dir=ply_dir)
    if args.trajectory == "dense":
        return load_dense_trajectory(model_path)
    if args.trajectory == "custom":
        if not args.custom_trajectory:
            raise ValueError("--trajectory custom requires --custom_trajectory PATH")
        return load_custom_trajectory(args.custom_trajectory)
    if args.trajectory == "orbit":
        return build_orbit_trajectory(gaussians, n_frames=args.orbit_frames)
    if args.trajectory == "test":
        cams, _ = load_test_cameras(model_path)
        return cams
    raise ValueError(f"unknown --trajectory {args.trajectory!r}")


def _inject_gt_provenance(summary: dict, out_dir: str, prov: dict) -> None:
    """Stamp holdout-GT provenance into the report a post-hoc eval just wrote.

    Top level (``report.json["holdout_gt_provenance"]``), alongside
    ``eval_valid`` / ``seed_fade_confounded``, because that is where graders
    and humans look -- a sidecar file would be ignored exactly the way issue
    #630 documents ``regeneration_manifest.json`` being ignored.

    An ABSENT key means the report PREDATES issue #639, NOT that its GT was
    exact. Injection failure is LOUD and never silently leaves an unmarked
    report claiming exactness it cannot support.
    """
    summary["holdout_gt_provenance"] = prov
    path = os.path.join(out_dir, "report.json")
    try:
        with open(path) as f:
            doc = json.load(f)
        doc["holdout_gt_provenance"] = prov
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
    except Exception as e:
        print(f"[render_report] ERROR: could not stamp holdout_gt_provenance into {path}: {e!r}. "
              f"That report is UNMARKED -- do not read the absence of the key as "
              f"'GT was exact' (issue #639).", flush=True)


def _process_full_report(args, model_path: str, cfg_args, iteration: int, ply_path: str,
                          pipe, background, out_dir: str, device: str = "cuda") -> None:
    """--full-report: regenerate the complete standard artifact bundle
    (CLAUDE.md's "Standard benchmark artifacts" contract) from a frozen
    checkpoint. Idempotent -- a prior regeneration_manifest.json in out_dir
    is treated as "already done" and skipped unless --force is passed, so a
    restarted/rerun invocation is cheap and safe by default.

    ``device`` (default "cuda") threads through camera reconstruction only
    -- ``load_gaussians``/the renderer are always CUDA (this project has no
    CPU rasterizer); overriding it to "cpu" is for unit-testing the plumbing
    around them, never for real regeneration.
    """
    manifest_path = os.path.join(out_dir, "regeneration_manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        print(f"[render_report] {manifest_path} already exists — skipping "
              f"(pass --force to regenerate anyway)")
        return

    from gaussian_renderer import render as render_fn
    gaussians = load_gaussians(model_path, cfg_args.sh_degree, getattr(cfg_args, "model_layout", "gsplat"), ply_path)
    os.makedirs(out_dir, exist_ok=True)

    # Locate + BACK UP the prior report before anything expensive (and
    # destructive) runs. A crash during the render phase must never be able to
    # leave the run with neither its original report.json nor a complete
    # replacement.
    prior_path, prior_report, prior_reason = find_prior_report(model_path, out_dir, iteration)
    backup_path = ""
    if prior_path:
        backup_path = snapshot_prior_report(prior_path, out_dir)
        print(f"[render_report] prior report: {prior_path} (backed up to {backup_path or 'NOWHERE'})")
    else:
        print(f"[render_report] WARNING: {prior_reason}")

    test_cams, test_has_gt = load_test_cameras(model_path, device=device)
    if not test_has_gt:
        print(f"[render_report] WARNING: no holdout_gt/ under {model_path} — test metrics will be empty.")

    trajectory_source = "dense" if args.trajectory == "dense" else "recorded"
    if trajectory_source == "dense":
        train_cams = load_dense_trajectory(model_path, device=device)
    else:
        train_cams = load_recorded_trajectory(model_path, ply_dir=os.path.dirname(ply_path), device=device)

    gt_stats = {"n_total": len(train_cams), "n_matched": 0, "reason": "GT reload disabled (--no-reload_gt)"}
    if args.reload_gt and train_cams:
        gt_stats = attach_source_gt(train_cams, cfg_args, source_path=args.source_path_override)
        print(f"[render_report] train GT reattach: {gt_stats['n_matched']}/{gt_stats['n_total']} "
              f"cameras matched to source frames" + (f" ({gt_stats['reason']})" if gt_stats["reason"] else ""))

    print(f"[render_report] iter={iteration}: {len(train_cams)} train cams ({trajectory_source}, "
          f"{gt_stats['n_matched']} with GT) + {len(test_cams)} test cams (has_gt={test_has_gt})")

    summary = run_full_report(
        # abspath: the subdir below is a relpath against model_path/comparison
        # and must resolve against the SAME absolute base the writer joins it
        # onto, or an out-of-tree --out_dir lands somewhere else entirely.
        os.path.abspath(model_path), iteration, gaussians, train_cams, test_cams,
        render_fn, pipe, background,
        # write_post_training_report resolves its output as
        # model_path/comparison/<subdir>; a relative path with `..` segments
        # is how an out-of-tree --out_dir is expressed. abspath both sides so
        # a relative -m and an absolute --out_dir cannot be mixed.
        # normpath so an out-of-tree --out_dir does not leave
        # `<run>/comparison/../../../tmp/x` strings in summary["output_dir"]
        # and every `artifacts` entry.
        subdir=os.path.normpath(os.path.relpath(
            os.path.abspath(out_dir), os.path.join(os.path.abspath(model_path), "comparison"))),
    )

    merge_info = merge_report_into_out_dir(out_dir, prior_path, prior_report, prior_reason,
                                            backup_path, summary, iteration=iteration,
                                            ply_path=ply_path)

    render_w = render_h = None
    if train_cams:
        render_w = getattr(train_cams[0], "image_width", None)
        render_h = getattr(train_cams[0], "image_height", None)
    elif test_cams:
        render_w = getattr(test_cams[0], "image_width", None)
        render_h = getattr(test_cams[0], "image_height", None)

    written = write_regeneration_manifest(
        out_dir,
        model_path=model_path, iteration=iteration, ply_path=ply_path, cfg_args=cfg_args,
        trajectory_source=trajectory_source, gt_stats=gt_stats,
        n_train=len(train_cams), n_test=len(test_cams), test_has_gt=test_has_gt,
        summary=summary, render_width=render_w, render_height=render_h,
        report_merge=merge_info, out_dir=out_dir,
    )
    print(f"[render_report] iter={iteration} full report done: eval_valid={summary.get('eval_valid')} "
          f"n_test={summary.get('n_test')} mean_test_psnr={summary.get('mean_test_psnr')} "
          f"manifest={written}")


def _process_source_holdout_rescore(
    args,
    model_path: str,
    cfg_args,
    iteration: int,
    ply_path: str,
    pipe,
    background,
    out_dir: str,
    device: str = "cuda",
) -> None:
    """Score a frozen PLY against a reconstructed canonical finite holdout."""
    out_dir = validate_source_rescore_destination(model_path, out_dir)
    test_cams, eval_metadata = load_source_holdout_cameras(
        args.source_holdout_path,
        eval_hold=args.streaming_eval_hold,
        initial_frames=args.streaming_initial_frames,
        eval_guard=args.streaming_eval_guard,
        min_baseline_m=args.streaming_eval_min_baseline_m,
        min_angle_deg=args.streaming_eval_min_angle_deg,
        expected_n_test=args.expected_n_test,
        expected_realized_hash=args.expected_realized_holdout_hash,
        device=device,
    )
    from gaussian_renderer import render as render_fn

    gaussians = load_gaussians(
        model_path,
        cfg_args.sh_degree,
        getattr(cfg_args, "model_layout", "gsplat"),
        ply_path,
    )
    summary = run_eval(
        os.path.abspath(model_path),
        iteration,
        gaussians,
        test_cams,
        render_fn,
        pipe,
        background,
        subdir=os.path.normpath(
            os.path.relpath(
                out_dir,
                os.path.join(os.path.abspath(model_path), "comparison"),
            )
        ),
        eval_metadata=eval_metadata,
        test_metrics_max=int(args.expected_n_test),
        scene_source=os.path.abspath(args.source_holdout_path),
    )
    if not summary.get("eval_valid"):
        raise RuntimeError("source-derived checkpoint rescore produced invalid evaluation")
    if int(summary.get("n_test") or 0) != int(args.expected_n_test):
        raise RuntimeError(
            "source-derived checkpoint rescore scored an incomplete holdout: "
            f"expected {int(args.expected_n_test)}, scored {summary.get('n_test')}"
        )
    scored_metadata = summary.get("eval_metadata") or {}
    expected_hash = str(args.expected_realized_holdout_hash)
    if scored_metadata.get("scored_holdout_hash") != expected_hash:
        raise RuntimeError(
            "source-derived checkpoint rescore scored holdout hash mismatch: "
            f"expected {expected_hash}, got "
            f"{scored_metadata.get('scored_holdout_hash')}"
        )

    render_width = getattr(test_cams[0], "image_width", None) if test_cams else None
    render_height = getattr(test_cams[0], "image_height", None) if test_cams else None
    written = write_regeneration_manifest(
        out_dir,
        model_path=model_path,
        iteration=iteration,
        ply_path=ply_path,
        cfg_args=cfg_args,
        trajectory_source="source_derived_causal_holdout",
        gt_stats={
            "n_total": 0,
            "n_matched": 0,
            "reason": "evaluation-only causal holdout rescore",
        },
        n_train=0,
        n_test=len(test_cams),
        test_has_gt=True,
        summary=summary,
        render_width=render_width,
        render_height=render_height,
        report_merge={
            "prior_report_path": None,
            "reason": "external source-derived evaluation; no carry-over attempted",
            "merged": False,
        },
        out_dir=out_dir,
    )
    print(
        f"[render_report] iter={iteration} source-derived rescore done: "
        f"n_test={summary.get('n_test')} "
        f"mean_test_psnr={summary.get('mean_test_psnr')} manifest={written}"
    )


def _process_checkpoint(args, model_path: str, cfg_args, iteration: int, ply_path: str,
                         pipe, background, out_dir: str) -> None:
    if args.full_report:
        _process_full_report(args, model_path, cfg_args, iteration, ply_path, pipe, background, out_dir)
        return

    from gaussian_renderer import render as render_fn

    gaussians = load_gaussians(model_path, cfg_args.sh_degree, getattr(cfg_args, "model_layout", "gsplat"), ply_path)
    os.makedirs(out_dir, exist_ok=True)

    if args.eval:
        if args.trajectory != "test":
            raise ValueError("--eval is only valid with --trajectory test")
        gt_prov: dict = {}
        cams, has_gt = load_test_cameras(
            model_path, require_exact_gt=getattr(args, "require_exact_holdout_gt", False),
            provenance_out=gt_prov)
        if not has_gt:
            print(f"[render_report] WARNING: no holdout_gt/ under {model_path} — "
                  f"report.json will have empty metrics.")
        summary = run_eval(model_path, iteration, gaussians, cams, render_fn, pipe, background,
                            subdir=os.path.relpath(out_dir, os.path.join(model_path, "comparison")))
        if gt_prov:
            _inject_gt_provenance(summary, out_dir, gt_prov)
        print(f"[render_report] iter={iteration} eval_valid={summary.get('eval_valid')} "
              f"n_test={summary.get('n_test')} mean_test_psnr={summary.get('mean_test_psnr')}")
        return

    cams = _get_trajectory(args, model_path, gaussians, ply_dir=os.path.dirname(ply_path))
    print(f"[render_report] iter={iteration}: {len(cams)} cameras ({args.trajectory})")
    if args.mp4:
        run_mp4(cams, gaussians, render_fn, pipe, background, os.path.join(out_dir, args.mp4), args.fps)
    if args.contact_sheet:
        run_contact_sheet(cams, gaussians, render_fn, pipe, background,
                           os.path.join(out_dir, "contact_sheet.png"),
                           cols=args.sheet_cols, cell_width=args.sheet_cell_width)


def main():
    parser = argparse.ArgumentParser(description="Standalone post-training render/eval module")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=None, help="default: latest checkpoint")
    parser.add_argument("--history", action="store_true",
                         help="apply the selected mode(s) to every saved checkpoint")
    parser.add_argument("--trajectory", choices=["recorded", "dense", "test", "custom", "orbit"],
                         default="recorded")
    parser.add_argument("--custom_trajectory", default=None, help="JSON file, camera_to_JSON schema")
    parser.add_argument("--orbit_frames", type=int, default=90)
    parser.add_argument("--mp4", default=None, help="output filename (relative to the run dir)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--contact-sheet", dest="contact_sheet", action="store_true")
    parser.add_argument("--sheet-cell-width", dest="sheet_cell_width", type=int, default=960,
                         help="per-view render width in the contact sheet (native camera res if omitted/larger)")
    parser.add_argument("--sheet-cols", dest="sheet_cols", type=int, default=4)
    parser.add_argument("--eval", action="store_true", help="only valid with --trajectory test")
    parser.add_argument("--require_exact_holdout_gt", action="store_true",
                        help="refuse to score against pre-#639 lossy (JPEG q92) holdout GT "
                             "instead of scoring against it with a warning")
    parser.add_argument("--full-report", dest="full_report", action="store_true",
                         help="regenerate the COMPLETE comparison/iter_N/ artifact bundle "
                              "(contact/normals/centers/rings sheets, trajectory.mp4, "
                              "train_depth_timeseries.mp4, report.json, geometry_metrics.json, "
                              "regeneration_manifest.json) via the same write_post_training_report "
                              "call train.py/train_streaming.py make inline. --trajectory "
                              "{recorded,dense} selects the train-camera set (default recorded).")
    parser.add_argument("--no-reload_gt", dest="reload_gt", action="store_false", default=True,
                         help="--full-report only: skip re-resolving train cameras against the "
                              "original --source_path dataset for GT RGB/depth (pose-only train "
                              "cams still yield trajectory.mp4, just no train_contact_sheet/"
                              "train_depth_timeseries/train PSNR)")
    parser.add_argument("--source_path_override", default=None,
                         help="--full-report only: use this dataset path instead of cfg_args."
                              "source_path when re-resolving train-camera GT (e.g. the dataset was "
                              "remounted at a different path than the one recorded at training time)")
    parser.add_argument("--out_dir", default=None,
                         help="write the regenerated artifact bundle HERE instead of into the "
                              "run's own comparison/iter_N/. Non-destructive: the source run "
                              "is only ever read. Use this whenever you want to inspect a "
                              "regeneration without modifying a finished run.")
    parser.add_argument("--force", action="store_true",
                         help="--full-report only: regenerate even if a regeneration_manifest.json "
                              "already exists for this checkpoint's output dir")
    parser.add_argument("--source-holdout-path", default=None,
                        help="recover the canonical finite-source causal holdout from this "
                             "dataset when the frozen run lacks test_cameras.json/holdout_gt; "
                             "requires --trajectory test --eval and an external --out_dir")
    parser.add_argument("--expected-n-test", type=int, default=None,
                        help="required fail-closed holdout count for --source-holdout-path")
    parser.add_argument("--expected-realized-holdout-hash", default=None,
                        help="required fail-closed camera-set hash for --source-holdout-path")
    parser.add_argument("--streaming-eval-hold", type=int, default=8)
    parser.add_argument("--streaming-initial-frames", type=int, default=5)
    parser.add_argument("--streaming-eval-guard", type=int, default=0)
    parser.add_argument("--streaming-eval-min-baseline-m", type=float, default=0.03)
    parser.add_argument("--streaming-eval-min-angle-deg", type=float, default=2.0)
    parser.add_argument("--checkpoint-sh-degree", type=int, default=None,
                        help="explicit frozen PLY SH degree when cfg_args is absent")
    parser.add_argument("--checkpoint-model-layout", choices=["gsplat", "legacy"], default=None,
                        help="explicit frozen PLY parameter layout when cfg_args is absent")
    parser.add_argument("--checkpoint-rasterize-mode", choices=["classic", "antialiased"],
                        default=None,
                        help="explicit frozen-checkpoint renderer when cfg_args is absent")
    checkpoint_bg = parser.add_mutually_exclusive_group()
    checkpoint_bg.add_argument("--checkpoint-white-background",
                               dest="checkpoint_white_background", action="store_true")
    checkpoint_bg.add_argument("--checkpoint-black-background",
                               dest="checkpoint_white_background", action="store_false")
    parser.set_defaults(checkpoint_white_background=None)
    pipeline_group = PipelineParams(argparse.ArgumentParser(add_help=False))
    args = parser.parse_args()

    model_path = args.model_path
    if args.source_holdout_path:
        if args.history or args.full_report or not args.eval or args.trajectory != "test":
            parser.error(
                "--source-holdout-path requires --trajectory test --eval and cannot "
                "be combined with --history/--full-report"
            )
        if not args.out_dir:
            parser.error("--source-holdout-path requires a fresh external --out_dir")
        if args.expected_n_test is None or not args.expected_realized_holdout_hash:
            parser.error(
                "--source-holdout-path requires --expected-n-test and "
                "--expected-realized-holdout-hash"
            )
    cfg_args = resolve_checkpoint_cfg(model_path, args)
    pipe = pipeline_group
    pipe.rasterize_mode = getattr(cfg_args, "rasterize_mode", "classic")
    bg_color = [1, 1, 1] if getattr(cfg_args, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.history:
        ckpts = _list_checkpoints(model_path)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found under {model_path}")
        print(f"[render_report] history sweep over {len(ckpts)} checkpoints")
        for iteration, ply_path in ckpts:
            out_dir = (os.path.join(args.out_dir, f"iter_{iteration}") if args.out_dir
                       else os.path.join(model_path, "comparison", "history", f"iter_{iteration}"))
            _process_checkpoint(args, model_path, cfg_args, iteration, ply_path, pipe, background, out_dir)
    else:
        if args.iteration is not None:
            ckpts = dict(_list_checkpoints(model_path))
            if args.iteration not in ckpts:
                raise FileNotFoundError(f"No checkpoint at iteration {args.iteration} under {model_path}")
            iteration, ply_path = args.iteration, ckpts[args.iteration]
        else:
            iteration, ply_path = _latest_checkpoint(model_path)
        out_dir = args.out_dir or os.path.join(model_path, "comparison", f"iter_{iteration}")
        if args.source_holdout_path:
            _process_source_holdout_rescore(
                args, model_path, cfg_args, iteration, ply_path, pipe, background, out_dir
            )
        else:
            _process_checkpoint(args, model_path, cfg_args, iteration, ply_path, pipe, background, out_dir)


if __name__ == "__main__":
    main()
