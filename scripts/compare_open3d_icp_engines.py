#!/usr/bin/env python3
"""Prove the Tensor and legacy ``open3d_icp_live`` engines are the same frontend.

Blocker 4 of the frontend-independence benchmark PR: the ROCm Tensor path is an
*acceleration* of one ICP frontend, not a second benchmark method.  This driver
feeds one deterministic RGB-D sequence through :class:`Open3DICPTracker` twice --
``engine="legacy"`` (CPU) and ``engine="tensor"`` (accelerator) -- under
identical ICP parameters, then compares the two runs on the axes that would
reveal a different algorithm or configuration rather than mere numerical noise:

* **pose convention** -- the composed ``T_world_camera`` trajectories must agree
  in direction and magnitude, not merely in shape.  An inverted or transposed
  convention on one engine shows up here as a large divergence even when both
  engines individually converge.
* **frame validity / acceptance** -- the accept/reject decision per frame must
  match exactly; both engines apply the same ``min_fitness`` /
  ``max_inlier_rmse`` gates to their own registration result.
* **registration telemetry** -- both engines must populate the same
  ``raw_stats`` keys, and their fitness / inlier RMSE must be close.

Numerical results are permitted to differ slightly (different reduction order,
different correspondence search implementation); the thresholds below are sized
to catch a *semantic* divergence, not to demand bit-equality.

Deterministic by construction: frames are a fixed slice of a fixed dataset in
capture order, ICP is seeded from identity (or the supplied prediction), and no
RNG is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eligsir.contracts.types import CameraIntrinsics, RGBDFrame
from eligsir.tracking.open3d_icp import Open3DICPTracker

# TUM RGB-D freiburg1 intrinsics and the dataset's fixed 5000-unit depth scale.
_TUM_FR1 = CameraIntrinsics(fx=517.3, fy=516.5, cx=318.6, cy=255.3, width=640, height=480)
_TUM_DEPTH_SCALE = 5000.0


def _read_index(path: Path) -> List[tuple]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), parts[1]))
    return rows


def load_tum_frames(root: Path, count: int, start: int, stride: int) -> List[RGBDFrame]:
    """Load a deterministic slice of a TUM RGB-D sequence in capture order."""
    from PIL import Image

    rgb_rows = _read_index(root / "rgb.txt")
    depth_rows = _read_index(root / "depth.txt")
    if not rgb_rows or not depth_rows:
        raise SystemExit(f"no rgb.txt/depth.txt entries under {root}")

    depth_ts = np.array([t for t, _ in depth_rows], dtype=np.float64)
    frames: List[RGBDFrame] = []
    for i in range(count):
        idx = start + i * stride
        if idx >= len(rgb_rows):
            break
        ts, rgb_rel = rgb_rows[idx]
        # Nearest-timestamp depth association (TUM's standard pairing rule).
        d_idx = int(np.argmin(np.abs(depth_ts - ts)))
        depth_rel = depth_rows[d_idx][1]

        rgb = np.asarray(Image.open(root / rgb_rel).convert("RGB"), dtype=np.float32) / 255.0
        raw = np.asarray(Image.open(root / depth_rel), dtype=np.float32)
        depth = raw / _TUM_DEPTH_SCALE
        depth[raw == 0] = 0.0  # TUM encodes "no return" as 0
        frames.append(
            RGBDFrame(
                index=idx,
                timestamp=ts,
                rgb=rgb,
                depth=depth.astype(np.float32),
                intrinsics=_TUM_FR1,
                pose=None,
            )
        )
    if len(frames) < 2:
        raise SystemExit("need at least 2 frames to register")
    return frames


def run_engine(engine: str, frames: List[RGBDFrame], icp_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Track the whole sequence with one engine and record per-frame telemetry."""
    backend = Open3DICPTracker(engine=engine, **icp_kwargs)
    per_frame: List[Dict[str, Any]] = []

    first = backend.initialize(frames[0])
    per_frame.append(
        {
            "index": frames[0].index,
            "valid": bool(first.valid),
            "c2w": np.asarray(first.pose.matrix, dtype=np.float64).tolist(),
        }
    )

    for frame in frames[1:]:
        result = backend.track(frame)
        stats = dict(result.raw_stats or {})
        per_frame.append(
            {
                "index": frame.index,
                "valid": bool(result.valid),
                "confidence": float(result.confidence),
                "fitness": _as_float(stats.get("fitness")),
                "inlier_rmse": _as_float(stats.get("inlier_rmse")),
                "source_points": stats.get("source_points"),
                "target_points": stats.get("target_points"),
                "pose_delta_m": _as_float(result.pose_delta_m),
                "c2w": np.asarray(result.pose.matrix, dtype=np.float64).tolist(),
                "stat_keys": sorted(stats.keys()),
                "error": stats.get("error"),
            }
        )

    return {
        "engine_requested": engine,
        "runtime_engine": backend._runtime_engine,
        "logical_device": backend._logical_device,
        "physical_accelerator": backend._physical_accelerator,
        "frames": per_frame,
    }


def _as_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _translations(run: Dict[str, Any]) -> np.ndarray:
    return np.array([np.asarray(f["c2w"], dtype=np.float64)[:3, 3] for f in run["frames"]])


def compare(legacy: Dict[str, Any], tensor: Dict[str, Any], tol: Dict[str, float]) -> Dict[str, Any]:
    """Compare the two runs on convention, acceptance, and telemetry."""
    findings: List[str] = []

    # --- frame validity / acceptance ------------------------------------
    lv = [f["valid"] for f in legacy["frames"]]
    tv = [f["valid"] for f in tensor["frames"]]
    validity_match = lv == tv
    if not validity_match:
        mismatched = [
            legacy["frames"][i]["index"] for i in range(min(len(lv), len(tv))) if lv[i] != tv[i]
        ]
        findings.append(f"frame validity differs at indices {mismatched}")

    # --- pose convention -------------------------------------------------
    lt, tt = _translations(legacy), _translations(tensor)
    n = min(len(lt), len(tt))
    lt, tt = lt[:n], tt[:n]
    per_frame_pos_err = np.linalg.norm(lt - tt, axis=1)
    max_pos_err = float(per_frame_pos_err.max()) if n else 0.0

    # A flipped/inverted convention on one engine yields a trajectory that is
    # anti-correlated with the other. Compare net displacement direction.
    l_disp, t_disp = lt[-1] - lt[0], tt[-1] - tt[0]
    l_norm, t_norm = float(np.linalg.norm(l_disp)), float(np.linalg.norm(t_disp))
    if l_norm > 1e-6 and t_norm > 1e-6:
        cos = float(np.dot(l_disp, t_disp) / (l_norm * t_norm))
    else:
        cos = 1.0  # degenerate/near-static sequence: direction undefined, not divergent
    convention_match = cos >= tol["min_direction_cos"] and max_pos_err <= tol["max_position_err_m"]
    if not convention_match:
        findings.append(
            f"pose convention divergence: direction cos={cos:.4f}, "
            f"max position delta={max_pos_err:.4f} m"
        )

    # --- registration telemetry -----------------------------------------
    l_keys = {tuple(f.get("stat_keys") or []) for f in legacy["frames"][1:]}
    t_keys = {tuple(f.get("stat_keys") or []) for f in tensor["frames"][1:]}
    telemetry_keys_match = l_keys == t_keys
    if not telemetry_keys_match:
        findings.append(f"raw_stats keys differ: legacy={l_keys} tensor={t_keys}")

    fit_deltas, rmse_deltas = [], []
    for lf, tf in zip(legacy["frames"][1:], tensor["frames"][1:]):
        if lf["fitness"] is not None and tf["fitness"] is not None:
            fit_deltas.append(abs(lf["fitness"] - tf["fitness"]))
        if lf["inlier_rmse"] is not None and tf["inlier_rmse"] is not None:
            rmse_deltas.append(abs(lf["inlier_rmse"] - tf["inlier_rmse"]))
    max_fit_delta = float(max(fit_deltas)) if fit_deltas else 0.0
    max_rmse_delta = float(max(rmse_deltas)) if rmse_deltas else 0.0
    telemetry_close = (
        max_fit_delta <= tol["max_fitness_delta"] and max_rmse_delta <= tol["max_rmse_delta_m"]
    )
    if not telemetry_close:
        findings.append(
            f"registration telemetry diverges: max|dfitness|={max_fit_delta:.4f}, "
            f"max|drmse|={max_rmse_delta:.4f} m"
        )

    # The engines must genuinely have run on different substrates, otherwise
    # this comparison proves nothing about the Tensor path.
    engines_distinct = (
        legacy["runtime_engine"] == "legacy" and tensor["runtime_engine"] == "tensor"
    )
    if not engines_distinct:
        findings.append(
            f"engines did not resolve as requested: legacy={legacy['runtime_engine']}, "
            f"tensor={tensor['runtime_engine']}"
        )

    equivalent = (
        validity_match
        and convention_match
        and telemetry_keys_match
        and telemetry_close
        and engines_distinct
    )
    return {
        "semantically_equivalent": equivalent,
        "checks": {
            "engines_distinct": engines_distinct,
            "frame_validity_match": validity_match,
            "pose_convention_match": convention_match,
            "telemetry_keys_match": telemetry_keys_match,
            "telemetry_values_close": telemetry_close,
        },
        "measurements": {
            "frames_compared": int(n),
            "valid_frames_legacy": int(sum(lv)),
            "valid_frames_tensor": int(sum(tv)),
            "trajectory_direction_cos": cos,
            "max_position_delta_m": max_pos_err,
            "mean_position_delta_m": float(per_frame_pos_err.mean()) if n else 0.0,
            "net_displacement_legacy_m": l_norm,
            "net_displacement_tensor_m": t_norm,
            "max_abs_fitness_delta": max_fit_delta,
            "max_abs_inlier_rmse_delta_m": max_rmse_delta,
        },
        "tolerances": tol,
        "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path, help="TUM RGB-D sequence root")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--estimator", default="point_to_plane")
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--max-correspondence-distance", type=float, default=0.07)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    # Semantic-divergence thresholds (see module docstring).
    ap.add_argument("--max-position-err-m", type=float, default=0.05)
    ap.add_argument("--min-direction-cos", type=float, default=0.9)
    ap.add_argument("--max-fitness-delta", type=float, default=0.15)
    ap.add_argument("--max-rmse-delta-m", type=float, default=0.02)
    args = ap.parse_args()

    frames = load_tum_frames(args.dataset, args.frames, args.start, args.stride)
    icp_kwargs = dict(
        estimator=args.estimator,
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_correspondence_distance,
        max_iterations=args.max_iterations,
    )

    legacy = run_engine("legacy", frames, icp_kwargs)
    tensor = run_engine("tensor", frames, icp_kwargs)
    verdict = compare(
        legacy,
        tensor,
        {
            "max_position_err_m": args.max_position_err_m,
            "min_direction_cos": args.min_direction_cos,
            "max_fitness_delta": args.max_fitness_delta,
            "max_rmse_delta_m": args.max_rmse_delta_m,
        },
    )

    record = {
        "dataset": str(args.dataset),
        "sequence": {
            "frames": len(frames),
            "start": args.start,
            "stride": args.stride,
            "source_indices": [f.index for f in frames],
        },
        "icp_parameters": icp_kwargs,
        "legacy_run": legacy,
        "tensor_run": tensor,
        "verdict": verdict,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)

    if not verdict["semantically_equivalent"]:
        print("ENGINE EQUIVALENCE FAILED", file=sys.stderr)
        raise SystemExit(2)
    print("ENGINE EQUIVALENCE PASSED", file=sys.stderr)


if __name__ == "__main__":
    main()
