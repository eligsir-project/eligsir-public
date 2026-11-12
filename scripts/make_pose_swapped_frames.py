#!/usr/bin/env python3
"""Build a pose-swapped sibling RGB-D dataset.

DIAGNOSTIC EXPERIMENT 2 — pose-swap.

Produces a dataset identical to ``--src`` EXCEPT that every frame's ``c2w`` in
``frames.jsonl`` is replaced by the pose from a TUM-format trajectory file
(``--poses``), matched by timestamp (nearest within ``--max-dt``, default 0.05s).
This lets two streaming reconstructions differ in NOTHING but the camera
trajectory (orbslam3 vs tracker), isolating the trajectory as the cause of the
roll/drift.

Everything else (images/, depth/, intrinsics.json and any other sibling files)
is symlinked unchanged into the output dataset; only ``frames.jsonl`` is
rewritten. Frames with no pose match within --max-dt are dropped.

The c2w is built from the TUM ``tx ty tz qx qy qz qw`` line using the EXACT
same convention as the production reader by importing
``_read_tum_poses`` / ``_tum_pose_to_c2w`` from ``scene/readers/tum.py``.
Reimplementing that math here would risk introducing the very kind of
transpose/sign bug we are hunting.

Usage:
    python scripts/make_pose_swapped_frames.py \
        --src  <exported dataset dir with frames.jsonl> \
        --poses <TUM-format trajectory file (t tx ty tz qx qy qz qw)> \
        --out  <new dataset dir>

This script is CPU-only. It launches no GPU / training work.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable so we can reuse the production conversions.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene.readers.tum import _read_tum_poses, _tum_pose_to_c2w  # noqa: E402
from eligsir.utils.rgbd_frames import read_frame_records  # noqa: E402


def _nearest_pose(ts, pose_times, poses, max_dt):
    """Return (tvec, qxyzw, dt) for the pose nearest to ``ts`` or None."""
    idx = int(np.searchsorted(pose_times, ts))
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(pose_times):
            dt = abs(pose_times[cand] - ts)
            if best is None or dt < best[0]:
                best = (dt, cand)
    if best is None or best[0] > max_dt:
        return None
    _, t, q = poses[best[1]]
    return t, q, best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Source dataset dir containing frames.jsonl")
    ap.add_argument("--poses", required=True,
                    help="TUM-format trajectory file (t tx ty tz qx qy qz qw)")
    ap.add_argument("--out", required=True,
                    help="Output dataset dir to create")
    ap.add_argument("--max-dt", type=float, default=0.05,
                    help="Max timestamp delta (s) for a pose match (default 0.05)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy sibling files instead of symlinking")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    poses_path = Path(args.poses).resolve()

    if not (src / "frames.jsonl").is_file():
        ap.error(f"no frames.jsonl in --src: {src}")
    if not poses_path.is_file():
        ap.error(f"--poses not found: {poses_path}")
    if out.exists():
        ap.error(f"--out already exists, refusing to overwrite: {out}")

    # Load poses (list of (t, tvec, qxyzw)) and build a sorted time index.
    poses = _read_tum_poses(str(poses_path))
    if not poses:
        ap.error(f"no poses parsed from {poses_path}")
    poses.sort(key=lambda p: p[0])
    pose_times = np.array([p[0] for p in poses], dtype=np.float64)

    records = read_frame_records(str(src / "frames.jsonl"))

    out.mkdir(parents=True, exist_ok=False)

    # Symlink (or copy) everything in src except frames.jsonl.
    import shutil
    for entry in sorted(os.listdir(src)):
        if entry == "frames.jsonl":
            continue
        s = src / entry
        d = out / entry
        if args.copy:
            if s.is_dir():
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        else:
            os.symlink(s, d)

    kept = []
    dropped = 0
    max_dt_seen = 0.0
    first_change = None

    for rec in records:
        ts = float(rec["timestamp"])
        match = _nearest_pose(ts, pose_times, poses, args.max_dt)
        if match is None:
            dropped += 1
            continue
        tvec, qxyzw, dt = match
        max_dt_seen = max(max_dt_seen, dt)
        c2w = _tum_pose_to_c2w(tvec, qxyzw)

        old_c2w = np.asarray(rec["c2w"], dtype=np.float32)
        new_rec = dict(rec)
        new_rec["c2w"] = np.asarray(c2w, dtype=np.float32).tolist()
        kept.append(new_rec)

        if first_change is None:
            first_change = (rec["id"], old_c2w[:3, 3].copy(),
                            np.asarray(c2w, dtype=np.float32)[:3, 3].copy(), dt)

    if not kept:
        # Clean up the half-built output so a failed run leaves no debris.
        shutil.rmtree(out)
        ap.error("no frames matched a pose within --max-dt; nothing written")

    with open(out / "frames.jsonl", "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, separators=(",", ":")))
            f.write("\n")

    print("=" * 64)
    print("pose-swap complete")
    print("=" * 64)
    print(f"  src           : {src}")
    print(f"  poses         : {poses_path}  ({len(poses)} poses)")
    print(f"  out           : {out}")
    print(f"  input frames  : {len(records)}")
    print(f"  kept (matched): {len(kept)}")
    print(f"  dropped       : {dropped}")
    print(f"  max |dt|      : {max_dt_seen * 1000.0:.2f} ms (limit {args.max_dt * 1000:.0f} ms)")
    if first_change is not None:
        fid, old_t, new_t, dt = first_change
        print(f"  sanity (frame {fid}): translation old {old_t} -> new {new_t} "
              f"(|dt|={dt * 1000:.2f} ms)")
    print("=" * 64)


if __name__ == "__main__":
    main()
