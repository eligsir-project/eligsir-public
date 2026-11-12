#!/usr/bin/env python3
"""Seed-cohort geometric frustum coverage, per source frame (issue #287 / #681).

Why this exists
----------------
Issue #287 asks whether `optimization/seed_psnr` (the fixed bootstrap camera's
render quality) falls because of occlusion/visibility interference from later
geometry, or because of intrinsic degradation of the seed Gaussians
themselves. PR #733 (issue #681 follow-up) narrowed this with a crude
single-vector heading-dot-product proxy and explicitly named the analysis
that would upgrade the verdict from "suggestive" to closer to "decisive":
compute, per source frame, what fraction of the SEED COHORT's own Gaussians
fall inside the ROAMING camera's view frustum, and compare that time series
against the independently-measured seed-PSNR fade onset (source frames
1113/1125/1160/1220 across the four `issue289cadv2_*` cadence arms).

This is NOT a visibility-into-the-seed-VIEW test (i.e. it does not ask "is
the seed viewpoint occluded by later geometry"); it is the complementary and
arguably more direct question the #287 framing also cares about: as the
ROAMING camera moves through the scene, does it keep re-observing the seed
cohort's own geometry (giving it continued photometric gradient support), or
does the seed cohort drop out of view for a stretch? A seed cohort that is
never revisited by the training camera is starved of supervision -- distinct
from, but consistent with, an "occlusion" story; a seed cohort that stays
visible throughout is evidence AGAINST simple visibility/occlusion driving
the fade.

Relationship to other issue #287 work (read this before reusing the script)
-----------------------------------------------------------------------------
* Issue #694 (GPU, dense S0/S1 substrate, a REAL renderer, lineage-aware
  seed-only/full-render decomposition) is the eventual, definitive answer to
  #287. This script does NOT do that -- it has no renderer anywhere in it,
  computes no pixel-level alpha/depth contribution, and runs on a DIFFERENT
  substrate (`issue289cadv2_*`, the #681 cadence quartet). Do not conflate
  the two or treat this script's output as a substitute for #694's.
* `scripts/seed_cohort_drift.py` / `utils/seed_cohort_drift.py` (#287's first
  pass, PR #407) measured intrinsic parameter drift (opacity, scale, colour,
  position) of the seed cohort over iterations, with no camera/visibility
  component at all. This script is the geometric-visibility complement to
  that work, not a replacement.
* PR #733's heading-dot-product proxy asks "does the roaming camera's own
  optical axis point back toward the seed viewpoint's *position*". This
  script asks a different and stronger question: "does the seed cohort's own
  Gaussian geometry actually fall inside the roaming camera's *frustum*",
  using each camera's real intrinsics (fx/fy/cx/cy/width/height) and a
  pinhole projection test, not a single heading vector.

Method
------
1. Cohort resolution: EXACT ONLY, via the canonical resolver
   `utils.gaussian_lineage.seed_cohort_mask` (through the
   `CheckpointLineageView` duck-type shim used by
   `scripts/seed_cohort_offline_replay.py` and `utils/seed_cohort_drift.py`
   for exactly this purpose -- static `chkpnt*.pth` state dicts). Raises
   `SeedCohortUnavailable` rather than approximating; this script does the
   same and refuses to run rather than substitute a hand-rolled mask.
2. Camera geometry: reuses this repo's own convention, documented at
   `render_report.py`'s `_camera_json_to_minicam` -- `trajectory_full.jsonl`
   rows store `rotation`/`position` as the CAMERA-TO-WORLD rotation and
   camera center (despite `utils/camera_utils.py:camera_to_JSON`'s misleading
   local variable name calling the inverted result "W2C"). This script
   reimplements just the CPU/numpy half of that same conversion (build C2W,
   invert to W2C) -- no CUDA, no MiniCam, no renderer needed for a pure
   geometric frustum test -- and uses the same `fx/fy/cx/cy/width/height`
   pinhole intrinsics `trajectory_full.jsonl` already carries per frame.
3. Frustum test: standard pinhole in-frustum test in the SAME camera
   convention `getProjectionMatrix`/`MiniCam`/`make_gsplat_camera_tensors`
   use elsewhere in this repo (x right, y down, z forward into the scene) --
   a point is "in frustum" iff it is in front of the camera (z > znear) and
   its projected pixel falls inside [0, width) x [0, height). No far-plane /
   max-depth clip is applied (this scene's extent is a few metres, well
   inside any sensor's max range, so a depth cutoff would not change the
   result in practice -- this is a geometric FOV-cone test, not a
   sensor-range test).

CPU-only, read-only. Does not import train.py/train_streaming.py or any
training code path -- only `utils.gaussian_lineage` (pure torch, no CUDA)
and stdlib/numpy/torch for I/O and array math.

Usage
-----
    python scripts/seed_frustum_coverage.py <run_dir> [--checkpoint ITER]
        [--stride N] [--out out.json]

`--checkpoint` selects which `chkpntN.pth` supplies the seed cohort's own
Gaussian positions/opacities (default: nearest fully-alive checkpoint at or
after 10000, since #287's own audit found the seed cohort's position is
already ~static -- median drift ~0.03m -- by iteration 4000-8000; earlier
checkpoints are noisier). `--stability` additionally reports cohort
alive-count/position-drift across ALL checkpoints in the run, to check (not
assume) that this static-position premise holds on THIS substrate.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.utils.gaussian_lineage import (  # noqa: E402
    CheckpointLineageView,
    SeedCohortUnavailable,
    seed_cohort_mask,
    seed_uid_range,
)

LIVE_OPACITY_THRESHOLD = 0.05  # repo convention, see CLAUDE.md "live mass"
ZNEAR = 0.01  # matches utils/graphics_utils.py getProjectionMatrix's own default


def numeric_checkpoints(run_dir: str):
    found = []
    for path in glob.glob(os.path.join(run_dir, "chkpnt*.pth")):
        match = re.search(r"chkpnt(\d+)\.pth$", os.path.basename(path))
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_shim(path: str) -> CheckpointLineageView:
    blob = torch.load(path, map_location="cpu", weights_only=False)[0]
    return CheckpointLineageView(blob)


def seed_positions_and_opacity(shim: CheckpointLineageView):
    """Returns (positions[N,3] float64 numpy, opacity[N] float64 numpy, uid[N] numpy)
    for the EXACT surviving seed cohort. Raises SeedCohortUnavailable via the
    canonical resolver if this checkpoint cannot testify to the cohort."""
    mask = seed_cohort_mask(shim)
    pos = shim.get_xyz[mask].detach().cpu().numpy().astype(np.float64)
    op = shim.get_opacity[mask].squeeze(-1).detach().cpu().numpy().astype(np.float64)
    uid = shim.gaussian_uid[mask].detach().cpu().numpy()
    return pos, op, uid


def build_w2c(entry: dict) -> np.ndarray:
    """entry['rotation']/entry['position'] are C2W rotation / camera center
    (see module docstring). Returns the 4x4 world-to-camera matrix."""
    rot = np.asarray(entry["rotation"], dtype=np.float64)
    pos = np.asarray(entry["position"], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos
    return np.linalg.inv(c2w)


def in_frustum_mask(points_world: np.ndarray, entry: dict, znear: float = ZNEAR) -> np.ndarray:
    w2c = build_w2c(entry)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p_cam = points_world @ R.T + t
    z = p_cam[:, 2]
    fx, fy, cx, cy = entry["fx"], entry["fy"], entry["cx"], entry["cy"]
    width, height = entry["width"], entry["height"]
    in_front = z > znear
    # avoid div-by-zero warnings for points behind/at the camera; their u/v
    # are meaningless anyway and masked out by in_front below
    safe_z = np.where(in_front, z, 1.0)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return in_front & in_frame


def load_trajectory(run_dir: str) -> dict:
    path = os.path.join(run_dir, "trajectory_full.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory_full.jsonl not found under {run_dir}")
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows[int(entry["id"])] = entry
    return rows


def pick_default_checkpoint(checkpoints, min_iter=10000):
    for it, path in checkpoints:
        if it >= min_iter:
            return it, path
    return checkpoints[-1] if checkpoints else (None, None)


def compute_coverage(pos, op, traj, stride=1, id_lo=None, id_hi=None):
    live = op > LIVE_OPACITY_THRESHOLD
    n_seed = pos.shape[0]
    n_live = int(live.sum())
    ids = sorted(traj.keys())
    if id_lo is not None:
        ids = [i for i in ids if i >= id_lo]
    if id_hi is not None:
        ids = [i for i in ids if i <= id_hi]
    ids = ids[::stride]
    rows = []
    for fid in ids:
        entry = traj[fid]
        mask = in_frustum_mask(pos, entry)
        raw_frac = float(mask.mean()) if n_seed else float("nan")
        live_frac = float(mask[live].mean()) if n_live else float("nan")
        op_sum = op.sum()
        weighted_frac = float((op * mask).sum() / op_sum) if op_sum > 0 else float("nan")
        rows.append({
            "id": fid,
            "admitted": bool(entry.get("admitted", False)),
            "is_holdout": bool(entry.get("is_holdout", False)),
            "raw_in_frustum_frac": raw_frac,
            "live_in_frustum_frac": live_frac,
            "opacity_weighted_in_frustum_frac": weighted_frac,
            "n_in_frustum": int(mask.sum()),
        })
    return rows, n_seed, n_live


def stability_report(run_dir: str, checkpoints):
    """Cross-checkpoint cohort alive-count / position-drift check -- does NOT
    assume the seed cohort's own position is static, measures it on THIS
    substrate."""
    out = []
    ref_pos_by_uid = None
    ref_iter = None
    for it, path in checkpoints:
        shim = load_shim(path)
        try:
            lo, hi = seed_uid_range(shim)
            pos, op, uid = seed_positions_and_opacity(shim)
        except SeedCohortUnavailable as exc:
            out.append({"iteration": it, "error": str(exc)})
            continue
        n_born = hi - lo
        row = {
            "iteration": it,
            "n_seed_born": n_born,
            "n_alive": int(pos.shape[0]),
            "survival_fraction": (pos.shape[0] / n_born) if n_born else None,
            "n_live_mass": int((op > LIVE_OPACITY_THRESHOLD).sum()),
            "mean_opacity": float(op.mean()) if op.size else None,
        }
        if ref_pos_by_uid is None:
            ref_pos_by_uid = dict(zip(uid.tolist(), pos.tolist()))
            ref_iter = it
            row["position_drift_vs_ref_median_m"] = 0.0
            row["ref_iteration"] = ref_iter
        else:
            drifts = []
            for u_, p_ in zip(uid.tolist(), pos.tolist()):
                ref = ref_pos_by_uid.get(u_)
                if ref is not None:
                    d = float(np.linalg.norm(np.asarray(p_) - np.asarray(ref)))
                    drifts.append(d)
            row["position_drift_vs_ref_median_m"] = float(np.median(drifts)) if drifts else None
            row["position_drift_vs_ref_n_matched"] = len(drifts)
            row["ref_iteration"] = ref_iter
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="checkpoint iteration to source seed positions/opacity from")
    parser.add_argument("--stride", type=int, default=1,
                        help="sample every Nth source frame from trajectory_full.jsonl")
    parser.add_argument("--id-lo", type=int, default=None)
    parser.add_argument("--id-hi", type=int, default=None)
    parser.add_argument("--stability", action="store_true",
                        help="also report cross-checkpoint cohort alive-count/position-drift")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoints = numeric_checkpoints(args.run_dir)
    if not checkpoints:
        raise SystemExit(f"no chkpnt*.pth found under {args.run_dir}")

    if args.checkpoint is not None:
        matches = [p for it, p in checkpoints if it == args.checkpoint]
        if not matches:
            raise SystemExit(f"no chkpnt{args.checkpoint}.pth under {args.run_dir}")
        chosen_iter, chosen_path = args.checkpoint, matches[0]
    else:
        chosen_iter, chosen_path = pick_default_checkpoint(checkpoints)

    shim = load_shim(chosen_path)
    lo, hi = seed_uid_range(shim)  # raises SeedCohortUnavailable if not resolvable
    pos, op, uid = seed_positions_and_opacity(shim)
    print(f"[seed_frustum_coverage] {args.run_dir}: seed cohort resolved from "
          f"chkpnt{chosen_iter}.pth -- uid range [{lo},{hi}), alive {pos.shape[0]}, "
          f"live-mass {(int((op > LIVE_OPACITY_THRESHOLD).sum()))}", flush=True)

    traj = load_trajectory(args.run_dir)
    rows, n_seed, n_live = compute_coverage(
        pos, op, traj, stride=args.stride, id_lo=args.id_lo, id_hi=args.id_hi)

    result = {
        "run_dir": args.run_dir,
        "checkpoint_iteration": chosen_iter,
        "seed_uid_lo": lo,
        "seed_uid_hi": hi,
        "n_seed_born": hi - lo,
        "n_seed_alive_at_checkpoint": n_seed,
        "n_seed_live_mass_at_checkpoint": n_live,
        "live_opacity_threshold": LIVE_OPACITY_THRESHOLD,
        "znear": ZNEAR,
        "coverage": rows,
    }
    if args.stability:
        result["stability"] = stability_report(args.run_dir, checkpoints)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
