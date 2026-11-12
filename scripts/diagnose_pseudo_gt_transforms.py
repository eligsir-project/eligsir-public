#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene.readers.tum import _read_tum_poses, _tum_pose_to_c2w
from eligsir.utils.graphics_utils import getWorld2View2
from eligsir.utils.rgbd_frames import c2w_to_camera_rt, read_frame_records


_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_intrinsics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "width": int(data["width"]),
        "height": int(data["height"]),
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "depth_scale": float(data.get("depth_scale", data.get("depth_factor", 1000.0))),
    }


def _depth_meters(path: Path, depth_scale: float) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(depth_scale)
    return arr.astype(np.float32)


def _rotation_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    rel = np.asarray(ra, dtype=np.float64).T @ np.asarray(rb, dtype=np.float64)
    cos_theta = (float(np.trace(rel)) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def _pose_variant(c2w: np.ndarray, variant: str) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if variant == "direct":
        return c2w
    if variant == "inverse":
        return np.linalg.inv(c2w)
    if variant == "direct_cv_to_gl":
        return c2w @ _CV_TO_GL
    if variant == "inverse_cv_to_gl":
        return np.linalg.inv(c2w) @ _CV_TO_GL
    raise ValueError(f"unknown pose variant: {variant}")


def _reprojection_residual(
    dataset: Path,
    records: list[dict[str, Any]],
    intr: dict[str, float],
    pair: tuple[int, int],
    variant: str,
    sample_points: int,
    min_depth: float,
    max_depth: float,
) -> dict[str, Any]:
    i, j = pair
    rec_a = records[i]
    rec_b = records[j]
    depth_a = _depth_meters(dataset / rec_a["depth"], intr["depth_scale"])
    depth_b = _depth_meters(dataset / rec_b["depth"], intr["depth_scale"])

    valid_y, valid_x = np.where((depth_a >= min_depth) & (depth_a <= max_depth))
    if valid_x.size == 0:
        return {"pair": [i, j], "variant": variant, "valid": 0, "reason": "no_source_depth"}

    step = max(1, valid_x.size // max(1, sample_points))
    xs = valid_x[::step][:sample_points]
    ys = valid_y[::step][:sample_points]
    z = depth_a[ys, xs].astype(np.float64)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    pts_a = np.stack(
        [
            (xs.astype(np.float64) - cx) * z / fx,
            (ys.astype(np.float64) - cy) * z / fy,
            z,
            np.ones_like(z),
        ],
        axis=0,
    )

    c2w_a = _pose_variant(np.asarray(rec_a["c2w"], dtype=np.float64), variant)
    c2w_b = _pose_variant(np.asarray(rec_b["c2w"], dtype=np.float64), variant)
    pts_world = c2w_a @ pts_a
    pts_b = np.linalg.inv(c2w_b) @ pts_world
    z_b = pts_b[2]

    u = np.rint(fx * pts_b[0] / z_b + cx).astype(np.int64)
    v = np.rint(fy * pts_b[1] / z_b + cy).astype(np.int64)
    height, width = depth_b.shape
    in_bounds = (
        (z_b >= min_depth)
        & (z_b <= max_depth)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    if int(in_bounds.sum()) == 0:
        return {"pair": [i, j], "variant": variant, "valid": 0, "reason": "no_reprojected_depth"}

    observed = depth_b[v[in_bounds], u[in_bounds]].astype(np.float64)
    predicted = z_b[in_bounds]
    valid_depth = (observed >= min_depth) & (observed <= max_depth)
    if int(valid_depth.sum()) == 0:
        return {"pair": [i, j], "variant": variant, "valid": 0, "reason": "no_target_depth"}

    errors = np.abs(observed[valid_depth] - predicted[valid_depth])
    return {
        "pair": [i, j],
        "variant": variant,
        "valid": int(errors.size),
        "median_abs_depth_error_m": float(np.median(errors)),
        "mean_abs_depth_error_m": float(np.mean(errors)),
        "p95_abs_depth_error_m": float(np.percentile(errors, 95)),
    }


def _default_pairs(count: int) -> list[tuple[int, int]]:
    candidates = [
        (0, 1),
        (0, 5),
        (10, 20),
        (100, 110),
        (count // 4, min(count - 1, count // 4 + 20)),
        (count // 2, min(count - 1, count // 2 + 20)),
        ((3 * count) // 4, min(count - 1, (3 * count) // 4 + 20)),
        (max(0, count - 200), max(0, count - 180)),
    ]
    out: list[tuple[int, int]] = []
    for i, j in candidates:
        if 0 <= i < j < count and (i, j) not in out:
            out.append((i, j))
    return out


def _roundtrip_stats(records: list[dict[str, Any]], max_records: int) -> dict[str, Any]:
    diffs = []
    center_diffs = []
    for rec in records[:max_records]:
        c2w = np.asarray(rec["c2w"], dtype=np.float64)
        rot, trans = c2w_to_camera_rt(c2w)
        roundtrip = np.linalg.inv(getWorld2View2(rot, trans))
        diffs.append(float(np.max(np.abs(c2w - roundtrip))))
        center_diffs.append(float(np.linalg.norm(c2w[:3, 3] - roundtrip[:3, 3])))
    return {
        "checked": int(len(diffs)),
        "max_matrix_abs_diff": float(max(diffs, default=0.0)),
        "max_center_diff_m": float(max(center_diffs, default=0.0)),
        "mean_center_diff_m": float(np.mean(center_diffs)) if center_diffs else 0.0,
    }


def _trajectory_match_stats(dataset: Path, records: list[dict[str, Any]], max_dt: float) -> dict[str, Any]:
    pose_path = dataset / "groundtruth.txt"
    if not pose_path.exists():
        return {"status": "missing_groundtruth"}
    pose_entries = _read_tum_poses(pose_path)
    times = [float(t) for t, _, _ in pose_entries]
    if not times:
        return {"status": "empty_groundtruth"}

    dt_values = []
    trans_errors = []
    rot_errors = []
    for rec in records:
        ts = float(rec.get("timestamp", 0.0))
        idx = bisect.bisect_left(times, ts)
        choices = [i for i in (idx - 1, idx) if 0 <= i < len(times)]
        if not choices:
            continue
        best = min(choices, key=lambda i: abs(times[i] - ts))
        dt = abs(times[best] - ts)
        if dt > max_dt:
            continue
        _, tvec, qxyzw = pose_entries[best]
        tum_c2w = _tum_pose_to_c2w(tvec, qxyzw).astype(np.float64)
        rec_c2w = np.asarray(rec["c2w"], dtype=np.float64)
        dt_values.append(float(dt))
        trans_errors.append(float(np.linalg.norm(rec_c2w[:3, 3] - tum_c2w[:3, 3])))
        rot_errors.append(_rotation_angle_deg(rec_c2w[:3, :3], tum_c2w[:3, :3]))

    return {
        "status": "ok" if dt_values else "no_matches",
        "matches": int(len(dt_values)),
        "max_dt_sec": float(max(dt_values, default=0.0)),
        "max_translation_error_m": float(max(trans_errors, default=0.0)),
        "max_rotation_error_deg": float(max(rot_errors, default=0.0)),
    }


def _summarize_reprojection(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[float]] = {}
    for row in results:
        median = row.get("median_abs_depth_error_m")
        if median is None:
            continue
        by_variant.setdefault(str(row["variant"]), []).append(float(median))
    summary: dict[str, Any] = {}
    for variant, values in sorted(by_variant.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[variant] = {
            "pairs": int(arr.size),
            "median_of_pair_medians_m": float(np.median(arr)),
            "mean_of_pair_medians_m": float(np.mean(arr)),
        }
    if summary:
        summary["best_variant"] = min(
            summary,
            key=lambda key: summary[key]["median_of_pair_medians_m"]
            if isinstance(summary[key], dict)
            else math.inf,
        )
    return summary


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    repro = report["reprojection_summary"]
    lines = [
        f"# Transform Diagnostic: {report['scene']}",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Frames: `{report['records']}`",
        f"- Best reprojection variant: `{repro.get('best_variant', 'none')}`",
        f"- Roundtrip max matrix diff: `{report['camera_roundtrip']['max_matrix_abs_diff']:.3e}`",
        f"- Frames-vs-TUM max translation error: `{report['trajectory_match']['max_translation_error_m']:.3e} m`",
        f"- Frames-vs-TUM max rotation error: `{report['trajectory_match']['max_rotation_error_deg']:.3e} deg`",
        "",
        "## Reprojection Summary",
        "",
        "| Variant | Pairs | Median of pair medians (m) | Mean of pair medians (m) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant, values in sorted(repro.items()):
        if not isinstance(values, dict):
            continue
        lines.append(
            f"| `{variant}` | {values['pairs']} | "
            f"{values['median_of_pair_medians_m']:.6f} | "
            f"{values['mean_of_pair_medians_m']:.6f} |"
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    records = read_frame_records(dataset / "frames.jsonl")
    intr = _load_intrinsics(dataset / "intrinsics.json")
    if args.max_frames > 0:
        records = records[: args.max_frames]
    if len(records) < 2:
        raise RuntimeError(f"Need at least two frames in {dataset}")

    pairs = _default_pairs(len(records))
    variants = ["direct", "inverse", "direct_cv_to_gl", "inverse_cv_to_gl"]
    reprojection_rows = []
    for pair in pairs:
        for variant in variants:
            reprojection_rows.append(
                _reprojection_residual(
                    dataset=dataset,
                    records=records,
                    intr=intr,
                    pair=pair,
                    variant=variant,
                    sample_points=args.sample_points,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                )
            )

    repro_summary = _summarize_reprojection(reprojection_rows)
    best = repro_summary.get("best_variant")
    if best == "direct":
        conclusion = (
            "Stored frames.jsonl c2w is the best-supported convention. "
            "Do not invert pseudo-GT poses; inspect visualization/report code if rotation appears wrong."
        )
    else:
        conclusion = (
            f"Stored frames.jsonl c2w is not the best-supported convention; `{best}` had lower RGB-D reprojection error. "
            "Pose conversion needs review before trusting this pseudo-GT run."
        )

    return {
        "scene": args.scene,
        "dataset": str(dataset),
        "records": int(len(records)),
        "intrinsics": intr,
        "pairs": [list(pair) for pair in pairs],
        "camera_roundtrip": _roundtrip_stats(records, args.roundtrip_frames),
        "trajectory_match": _trajectory_match_stats(dataset, records, args.association_max_dt),
        "reprojection": reprojection_rows,
        "reprojection_summary": repro_summary,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose pseudo-GT RGB-D transform conventions.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-points", type=int, default=6000)
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--roundtrip-frames", type=int, default=512)
    parser.add_argument("--association-max-dt", type=float, default=0.03)
    args = parser.parse_args()

    report = diagnose(args)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{args.scene}_transform_diagnostic.json"
    md_path = args.out / f"{args.scene}_transform_diagnostic.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, report)
    print(f"[diagnostic] wrote {json_path}")
    print(f"[diagnostic] wrote {md_path}")
    print(f"[diagnostic] {args.scene}: {report['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
