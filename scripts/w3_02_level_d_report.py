#!/usr/bin/env python3
"""Build the real W3-02 engine-OFF vs. engine-ON Level-D comparison report.

Reads only artifacts written by two real training runs (baseline = engine
OFF, treatment = engine ON) that shared: source dataset, poses, admitted
keyframes (--streaming_frame_admission all, unchanged), the same
--start_checkpoint, the same seed, and a frozen/verified mutation trace
(eligsir/optimization/gaussian/mutation_trace.py). It does not run any
training itself and does not assert pass/fail thresholds it wasn't handed;
every number here is read directly from the two run directories.

Usage:
    python scripts/w3_02_level_d_report.py \
        --baseline-dir output/w3_02_baseline_fr3 \
        --treatment-dir output/w3_02_treatment_fr3 \
        --manifest output/w3_02_evidence/manifest.json \
        --checkpoint-iterations 500 1000 1500 2000 \
        --out output/w3_02_evidence/level_d_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_report(run_dir: Path) -> dict[str, Any]:
    # Exclude debug variants like "iter_2000_bootstrap_views" (degenerate
    # train-view-only split) -- see the identical fix in
    # eligsir/experiments/w3_02_runtime.py, found the same way.
    reports = sorted(
        (
            p for p in run_dir.glob("comparison/iter_*/report.json")
            if p.parent.name.removeprefix("iter_").isdigit()
        ),
        key=lambda p: int(p.parent.name.removeprefix("iter_")),
    )
    if not reports:
        raise SystemExit(f"no comparison/iter_*/report.json under {run_dir}")
    return json.loads(reports[-1].read_text())


def _load_runtime_evidence(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "w3_02_runtime_evidence.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _tensor_diff(b: torch.Tensor, t: torch.Tensor) -> dict[str, Any]:
    if b.shape != t.shape:
        return {"shape_mismatch": [list(b.shape), list(t.shape)]}
    diff = (b.float() - t.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "shape": list(b.shape),
    }


def _checkpoint_diff(baseline_ckpt: Path, treatment_ckpt: Path) -> dict[str, Any]:
    """Compare a matched pair of chkpntN.pth files: Gaussian params + Adam state.

    Checkpoint layout (see scene/gsplat_model.py's capture()/GaussianModel
    checkpoint save, and train_streaming.py's _save_streaming_checkpoint):
    a (blob, iteration[, streaming_meta]) tuple where blob['params'] is
    {means, opacities, quats, scales, sh0, shN} -> Tensor, and
    blob['optimizers'] is {same param names} -> torch Adam
    state_dict()-shaped dict ({'state': {0: {'step','exp_avg','exp_avg_sq'}},
    'param_groups': [...]}).
    """
    if not baseline_ckpt.is_file() or not treatment_ckpt.is_file():
        return {"available": False, "reason": "checkpoint missing on one or both arms"}
    baseline_blob = torch.load(baseline_ckpt, map_location="cpu", weights_only=False)
    treatment_blob = torch.load(treatment_ckpt, map_location="cpu", weights_only=False)
    base, base_iter = baseline_blob[0], baseline_blob[1]
    treat, treat_iter = treatment_blob[0], treatment_blob[1]
    result: dict[str, Any] = {
        "available": True,
        "baseline_iteration": int(base_iter),
        "treatment_iteration": int(treat_iter),
        "iteration_match": int(base_iter) == int(treat_iter),
    }
    if not isinstance(base, dict) or not isinstance(treat, dict) or "params" not in base or "params" not in treat:
        result["parameter_diffs"] = {
            "note": "unrecognized checkpoint blob shape -- see report generation log",
            "baseline_type": type(base).__name__,
            "treatment_type": type(treat).__name__,
        }
        return result

    param_diffs: dict[str, Any] = {}
    for key in sorted(set(base["params"]) & set(treat["params"])):
        b, t = base["params"][key], treat["params"][key]
        if isinstance(b, torch.Tensor) and isinstance(t, torch.Tensor):
            param_diffs[key] = _tensor_diff(b, t)
    result["parameter_diffs"] = param_diffs

    optimizer_diffs: dict[str, Any] = {}
    base_opt, treat_opt = base.get("optimizers", {}), treat.get("optimizers", {})
    for key in sorted(set(base_opt) & set(treat_opt)):
        b_state = (base_opt[key] or {}).get("state", {})
        t_state = (treat_opt[key] or {}).get("state", {})
        slot_diffs: dict[str, Any] = {}
        for slot in sorted(set(b_state) & set(t_state), key=str):
            b_entry, t_entry = b_state[slot], t_state[slot]
            per_field = {}
            for field in ("exp_avg", "exp_avg_sq"):
                bv, tv = b_entry.get(field), t_entry.get(field)
                if isinstance(bv, torch.Tensor) and isinstance(tv, torch.Tensor):
                    per_field[field] = _tensor_diff(bv, tv)
            slot_diffs[str(slot)] = per_field
        optimizer_diffs[key] = {
            "n_baseline_slots": len(b_state),
            "n_treatment_slots": len(t_state),
            "slots_matched": len(set(b_state) & set(t_state)),
            "sample_slot_diffs": dict(list(slot_diffs.items())[:1]),
            "max_abs_diff_over_matched_slots": max(
                (
                    v
                    for slot in slot_diffs.values()
                    for field in slot.values()
                    if isinstance(field.get("max_abs_diff"), (int, float))
                    for v in [field["max_abs_diff"]]
                ),
                default=None,
            ),
        }
    result["optimizer_state_diffs"] = optimizer_diffs
    result["n_gaussians_baseline"] = int(base["params"]["means"].shape[0]) if "means" in base["params"] else None
    result["n_gaussians_treatment"] = int(treat["params"]["means"].shape[0]) if "means" in treat["params"] else None
    return result


def _throughput_summary(
    baseline_evidence: dict[str, Any] | None,
    treatment_evidence: dict[str, Any] | None,
    *,
    regression_budget_percent: float = 2.0,
    regression_budget_ms: float | None = None,
) -> dict[str, Any]:
    """Gate methodology per W3-17 (2026-08-10), fixing the two flaws
    docs/architecture/w3_02_completion.md's 2026-08-09 "round 3" section
    found and explicitly recommended re-specifying (see
    eligsir/experiments/w3_02_runtime.py's SPAN_DEFINITION docstring for
    the full incident history -- two prior rounds reported false 12.08%/
    17.84% "regressions" that were actually a begin()/end() span mismatch
    between arms, not a real code cost):

    1. **Pinned span, enforced, not just documented**: refuses to compute a
       comparison at all if the two evidence files' `span_definition_version`
       differ (or either is missing, i.e. pre-dates this field / pre-dates
       commit `4b2e22e`'s span fix) -- comparing across span versions is
       exactly the bug class that produced the false regressions, so this is
       a hard fail-safe, not a warning.
    2. **Absolute floor alongside the relative budget**: `regression_budget_percent`
       alone cannot reliably distinguish a real regression from run-to-run
       noise when the *relative* denominator (baseline speed) itself varies
       significantly session-to-session (w3_02_completion.md documented an
       ~1.8x baseline swing, 3.39-6.08ms/step, attributable to session/
       thermal/GPU-clock noise, not code changes). `regression_budget_ms`
       defaults to `None` (not enforced -- preserves prior behavior exactly)
       because this repo does not yet have a real intra-session noise-floor
       calibration to set it responsibly; a caller with real calibration
       data (e.g. two runs of the identical commit, same session, to measure
       intra-session ms/step variance) should pass a real value via
       --regression-budget-ms. When set, a regression only fails
       `within_budget` if it clears BOTH floors -- the relative percentage
       AND the absolute ms delta -- so a tiny absolute delta computed against
       a fast baseline can no longer trip the gate purely off a large
       relative percentage of a small number.
    """
    if baseline_evidence is None or treatment_evidence is None:
        return {
            "available": False,
            "reason": "w3_02_runtime_evidence.json missing on one or both arms (no --w3_02_evidence_manifest run, or CUDA/HIP unavailable)",
        }
    base_span = (baseline_evidence.get("runtime") or {}).get("span_definition_version")
    treat_span = (treatment_evidence.get("runtime") or {}).get("span_definition_version")
    if base_span != treat_span:
        return {
            "available": False,
            "reason": (
                f"span_definition_version mismatch between arms (baseline={base_span!r}, "
                f"treatment={treat_span!r}) -- refusing to compare optimization_gpu_ms across "
                "differently-bracketed spans; this is the exact bug class that produced the "
                "false 12.08%/17.84% W3-02 regressions (see w3_02_completion.md's round-3 "
                "section). Re-run both arms against the same code so their evidence carries "
                "matching span_definition_version, or if one predates that field entirely, "
                "treat it as untrustworthy for a throughput comparison."
            ),
        }
    base_rows = {r["iteration"]: r["gpu_ms"] for r in baseline_evidence.get("optimization_gpu_events", [])}
    treat_rows = {r["iteration"]: r["gpu_ms"] for r in treatment_evidence.get("optimization_gpu_events", [])}
    common = sorted(set(base_rows) & set(treat_rows))
    if len(common) < 3:
        return {"available": False, "reason": f"only {len(common)} matching GPU-event iterations between arms"}
    steady_start = (2 * len(common)) // 3
    steady = common[steady_start:]
    base_mean = sum(base_rows[i] for i in steady) / len(steady)
    treat_mean = sum(treat_rows[i] for i in steady) / len(steady)
    base_hz = 1000.0 / base_mean
    treat_hz = 1000.0 / treat_mean
    regression_percent = ((base_hz - treat_hz) / base_hz) * 100.0
    regression_ms = treat_mean - base_mean
    within_percent_budget = regression_percent <= regression_budget_percent
    within_ms_budget = regression_budget_ms is None or regression_ms <= regression_budget_ms
    return {
        "available": True,
        "span_definition_version": base_span,
        "steady_state_iterations": steady,
        "baseline_mean_gpu_ms": base_mean,
        "treatment_mean_gpu_ms": treat_mean,
        "baseline_steps_hz": base_hz,
        "treatment_steps_hz": treat_hz,
        "regression_percent": regression_percent,
        "regression_ms": regression_ms,
        "regression_budget_percent": regression_budget_percent,
        "regression_budget_ms": regression_budget_ms,
        "within_percent_budget": within_percent_budget,
        "within_ms_budget": within_ms_budget,
        "within_budget": within_percent_budget and within_ms_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--treatment-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-iterations", nargs="+", type=int, default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--regression-budget-percent", type=float, default=2.0,
        help="Relative throughput-regression budget (default: 2.0, matches the historical W3-02 gate).",
    )
    parser.add_argument(
        "--regression-budget-ms", type=float, default=None,
        help=(
            "Absolute ms/step regression floor (default: not enforced -- see "
            "_throughput_summary's docstring for why this repo does not yet have a "
            "responsibly-calibrated default; pass a real value derived from measured "
            "intra-session noise before relying on this for a pass/fail decision)."
        ),
    )
    args = parser.parse_args()

    baseline_report = _load_report(args.baseline_dir)
    treatment_report = _load_report(args.treatment_dir)
    baseline_evidence = _load_runtime_evidence(args.baseline_dir)
    treatment_evidence = _load_runtime_evidence(args.treatment_dir)

    checkpoint_diffs = {}
    for it in args.checkpoint_iterations:
        checkpoint_diffs[str(it)] = _checkpoint_diff(
            args.baseline_dir / f"chkpnt{it}.pth",
            args.treatment_dir / f"chkpnt{it}.pth",
        )

    def _n_live(report: dict[str, Any]) -> Any:
        return report.get("n_live") or (report.get("geometry") or {}).get("n_live")

    def _quality(report: dict[str, Any]) -> dict[str, Any]:
        return {
            "eval_valid": report.get("eval_valid"),
            "n_test": report.get("n_test"),
            "mean_test_psnr": report.get("mean_test_psnr"),
            "mean_test_ssim": report.get("mean_test_ssim"),
            "mean_test_lpips": report.get("mean_test_lpips"),
            "holdout_hash": (report.get("eval_metadata") or {}).get("holdout_hash"),
            "n_live": _n_live(report),
            "seed_fade": report.get("seed_fade"),
        }

    baseline_quality = _quality(baseline_report)
    treatment_quality = _quality(treatment_report)
    holdout_matches = (
        baseline_quality["holdout_hash"] is not None
        and baseline_quality["holdout_hash"] == treatment_quality["holdout_hash"]
    )

    report = {
        "manifest_path": str(args.manifest),
        "manifest": json.loads(args.manifest.read_text()) if args.manifest.is_file() else None,
        "run_directories": {"baseline": str(args.baseline_dir), "treatment": str(args.treatment_dir)},
        "holdout_hash_matches": holdout_matches,
        "quality": {"baseline": baseline_quality, "treatment": treatment_quality},
        "mutation_trace": {
            "baseline_runtime_mutation_trace": (baseline_evidence or {}).get("runtime", {}).get("mutation_trace_applied"),
            "treatment_runtime_mutation_trace": (treatment_evidence or {}).get("runtime", {}).get("mutation_trace_applied"),
            "treatment_mutation_trace_summary": (treatment_evidence or {}).get("runtime", {}).get("mutation_trace_summary"),
        },
        "checkpoint_parity": checkpoint_diffs,
        "throughput": _throughput_summary(
            baseline_evidence, treatment_evidence,
            regression_budget_percent=args.regression_budget_percent,
            regression_budget_ms=args.regression_budget_ms,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
