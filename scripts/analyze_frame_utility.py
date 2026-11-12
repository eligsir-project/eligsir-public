#!/usr/bin/env python3
"""Analysis/calibration report for shadow frame-utility scoring (plan
section 11.5/11.6, Subagent D; deliverables 3 and 6).

Reads the two JSONL logs a run's :class:`ShadowUtilityPolicy` writes under
``<model_path>/shadow_utility/`` (``scores.jsonl``, every scored candidate;
``realized_gain.jsonl``, completed before/after loss records for admitted
frames) and reports:

- correlation between predicted ``utility`` and realized gain
  (``loss_before - loss_after``) for frames that have both -- this is the
  actual calibration signal (deliverable 3): does the shadow score predict
  anything real?
- (issue #118) a PER-COMPONENT correlation breakdown against realized gain --
  joins each ``realized_gain.jsonl`` record back to the ``scores.jsonl``
  record it came from (matched on ``(frame_id, admitted_iteration)`` vs.
  ``(frame_id, meta.map_version)``, which train_streaming.py always sets to
  the same value at admission time) so every individual
  ``FrameUtilityComponents`` field -- not just the aggregate ``utility`` --
  gets its own Pearson r against ``realized_gain``. This is the "still
  needed" per-component breakdown the issue's diagnosis comment asked for
  (the aggregate-only correlation can't distinguish "the whole formula is
  bad" from "one unbounded term is swamping seven reasonable ones").
- a breakdown of which component (new_surface / residual / debt / gradient
  / parallax / redundancy / specular / cost) each of the TOP-scored and
  BOTTOM-scored frames were dominated by -- the qualitative sanity check
  (deliverable 6): if the top scorers are dominated by
  ``persistent_depth_supported_residual``/``specular_risk`` rather than a
  mix including ``new_surface``/``under_supported_mass``, that's a red flag
  the scorer degenerated into "raw RGB residual with extra steps" despite
  the formula's design.
- policy overhead, read back from any co-located ``shadow_utility_cost.json``
  (written by ``ShadowUtilityPolicy.cost_overhead_ms_summary()`` at run end,
  if the caller chose to persist it) -- deliverable 5.

Usage:

    python scripts/analyze_frame_utility.py <model_path> [<model_path> ...]

Each ``<model_path>`` is expected to contain a ``shadow_utility/`` dir (as
written by a run with ``--streaming_shadow_utility_enabled``). Multiple
paths are reported separately AND pooled together at the end (for the
cross-scene comparison required by deliverable 4) -- pooling only combines
correlation statistics, never mixes weights/formula across runs (both are
fixed/deterministic per plan section 11.3, so this is safe).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _dominant_component(components: dict, weights: dict) -> str:
    """Which single weighted term contributed the largest MAGNITUDE to this
    score -- used for the deliverable-6 qualitative check.
    """
    contributions = {
        "new_surface": weights.get("w_new_surface", 1.0) * components.get("new_surface", 0.0),
        "persistent_depth_supported_residual": weights.get("w_residual", 1.0)
        * components.get("persistent_depth_supported_residual", 0.0),
        "under_supported_mass": weights.get("w_debt", 1.0) * components.get("under_supported_mass", 0.0),
        "expected_gradient_value": weights.get("w_gradient", 1.0) * components.get("expected_gradient_value", 0.0),
        "useful_parallax": weights.get("w_parallax", 1.0) * components.get("useful_parallax", 0.0),
        "redundancy (penalty)": -weights.get("w_redundancy", 1.0) * components.get("redundancy", 0.0),
        "specular_risk (penalty)": -weights.get("w_specular", 1.0) * components.get("specular_risk", 0.0),
        "predicted_gpu_cost (penalty)": -weights.get("w_cost", 1.0) * components.get("predicted_gpu_cost", 0.0),
    }
    return max(contributions, key=lambda k: abs(contributions[k]))


_COMPONENT_FIELDS = (
    "pose_quality",
    "image_quality",
    "new_surface",
    "persistent_depth_supported_residual",
    "under_supported_mass",
    "expected_gradient_value",
    "useful_parallax",
    "redundancy",
    "specular_risk",
    "predicted_gpu_cost",
)


def _join_scores_to_gains(scores: List[dict], gains: List[dict]) -> List[dict]:
    """Pair each realized_gain.jsonl record (that has a realized_gain) with
    the scores.jsonl record it was scored from, matched on
    ``(frame_id, score_map_version)``. Usually the score and admission happen
    in the same iteration. A bounded cohort can defer a scored candidate and
    commit it later, so gain rows retain both the original score version and
    the later ``admitted_iteration``. Historical rows fall back to the old
    same-iteration key.
    """
    index = {}
    for s in scores:
        key = (s.get("frame_id"), (s.get("meta") or {}).get("map_version"))
        index[key] = s
    paired = []
    for g in gains:
        if g.get("realized_gain") is None:
            continue
        key = (
            g.get("frame_id"),
            g.get("score_map_version", g.get("admitted_iteration")),
        )
        s = index.get(key)
        if s is None:
            continue
        paired.append({"gain": g, "score": s})
    return paired


def _per_component_correlations(paired: List[dict]) -> dict:
    out = {}
    realized = [p["gain"]["realized_gain"] for p in paired]
    for field in _COMPONENT_FIELDS:
        xs = [p["score"]["components"].get(field, 0.0) for p in paired]
        out[field] = _pearson(xs, realized)
    return out


def analyze_run(model_path: str) -> dict:
    shadow_dir = os.path.join(model_path, "shadow_utility")
    scores = _read_jsonl(os.path.join(shadow_dir, "scores.jsonl"))
    gains = _read_jsonl(os.path.join(shadow_dir, "realized_gain.jsonl"))

    report: Dict[str, object] = {
        "model_path": model_path,
        "n_scores": len(scores),
        "n_realized_gain_records": len(gains),
    }

    if not scores:
        report["status"] = "no shadow_utility/scores.jsonl found or empty -- run did not have shadow scoring enabled"
        return report

    utilities = [s["utility"] for s in scores]
    report["utility_mean"] = statistics.fmean(utilities)
    report["utility_stdev"] = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0

    # Deliverable 3: correlation between predicted utility (at admission
    # time) and realized gain (measured later, for admitted frames only).
    predicted = [g["predicted_utility"] for g in gains if g.get("predicted_utility") is not None and g.get("realized_gain") is not None]
    realized = [g["realized_gain"] for g in gains if g.get("predicted_utility") is not None and g.get("realized_gain") is not None]
    report["n_paired_prediction_vs_realized"] = len(predicted)
    report["prediction_vs_realized_gain_pearson_r"] = _pearson(predicted, realized)

    # Issue #118: per-component correlation breakdown (see module docstring).
    paired = _join_scores_to_gains(scores, gains)
    report["n_paired_component_breakdown"] = len(paired)
    if len(paired) != report["n_paired_prediction_vs_realized"]:
        report["component_breakdown_join_warning"] = (
            f"joined {len(paired)} of {report['n_paired_prediction_vs_realized']} "
            "predicted-vs-realized rows to a scores.jsonl record -- unmatched rows excluded from "
            "the per-component breakdown below (see _join_scores_to_gains docstring)"
        )
    report["per_component_pearson_r_vs_realized_gain"] = _per_component_correlations(paired)

    # Deliverable 6: qualitative sanity check on top/bottom scorers.
    sorted_scores = sorted(scores, key=lambda s: s["utility"], reverse=True)
    top_n = sorted_scores[: min(10, len(sorted_scores))]
    bottom_n = sorted_scores[-min(10, len(sorted_scores)) :]
    top_dominant = [_dominant_component(s["components"], s["weights"]) for s in top_n]
    bottom_dominant = [_dominant_component(s["components"], s["weights"]) for s in bottom_n]
    report["top10_dominant_components"] = top_dominant
    report["bottom10_dominant_components"] = bottom_dominant

    residual_or_specular_dominated_top = sum(
        1 for c in top_dominant if c in ("persistent_depth_supported_residual", "specular_risk (penalty)")
    )
    report["top10_residual_or_specular_dominated_fraction"] = residual_or_specular_dominated_top / len(top_dominant)
    report["deliverable_6_flag"] = (
        "FAIL: top-scored frames are dominated by raw residual/specular terms -- scorer likely "
        "degenerated into 'RGB residual with extra steps', re-check weights/formula"
        if report["top10_residual_or_specular_dominated_fraction"] > 0.6
        else "OK: top-scored frames are not dominated by residual/specular alone"
    )

    # Deliverable 5: policy cost, if the run persisted it.
    cost_path = os.path.join(shadow_dir, "shadow_utility_cost.json")
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            report["policy_cost_overhead"] = json.load(f)

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_paths", nargs="+", help="One or more <model_path> dirs containing shadow_utility/")
    parser.add_argument("--out", default=None, help="Optional path to write the combined JSON report")
    args = parser.parse_args(argv)

    per_run_reports = [analyze_run(p) for p in args.model_paths]

    pooled_predicted: List[float] = []
    pooled_realized: List[float] = []
    pooled_paired: List[dict] = []
    for r, p in zip(per_run_reports, args.model_paths):
        scores = _read_jsonl(os.path.join(p, "shadow_utility", "scores.jsonl"))
        gains = _read_jsonl(os.path.join(p, "shadow_utility", "realized_gain.jsonl"))
        for g in gains:
            if g.get("predicted_utility") is not None and g.get("realized_gain") is not None:
                pooled_predicted.append(g["predicted_utility"])
                pooled_realized.append(g["realized_gain"])
        pooled_paired.extend(_join_scores_to_gains(scores, gains))

    combined = {
        "per_run": per_run_reports,
        "pooled_n_paired": len(pooled_predicted),
        "pooled_prediction_vs_realized_gain_pearson_r": _pearson(pooled_predicted, pooled_realized),
        "pooled_n_paired_component_breakdown": len(pooled_paired),
        "pooled_per_component_pearson_r_vs_realized_gain": _per_component_correlations(pooled_paired),
    }

    print(json.dumps(combined, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nWrote combined report to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
