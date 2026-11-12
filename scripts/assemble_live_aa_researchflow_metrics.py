#!/usr/bin/env python3
"""Assemble fail-closed ResearchFlow evidence for the live AA A/B."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_MAPPING_SAMPLE_FIELDS = (
    "mean_edge_psnr",
    "temporal_flicker_rms",
    "temporal_flicker_psnr_db",
    "temporal_flicker_valid_pairs",
    "temporal_flicker_status",
    "gaussian_count",
    "rasterize_mode",
    "training_resolution_tiers",
    "evaluation_resolutions",
    "rasterizer_gpu_ms",
    "rasterizer_timing_status",
    "mapping_active_s",
    "optimizer_updates",
    "supervision_views",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is invalid: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"PSNR history row {line_number} is not an object")
        rows.append(row)
    return rows


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def assemble_metrics(
    model_path: str | Path,
    *,
    expected_rasterize_mode: str,
    expected_git_sha: str,
    expected_batch_size: int,
    expected_camera_count: int,
) -> dict[str, Any]:
    root = Path(model_path).resolve()
    report = _read_json(root / "comparison/mapping_endpoint/report.json")
    summary = _read_json(root / "causal_replay_summary.json")
    if summary.get("throughput_evidence_valid") is False:
        status = str(
            summary.get("throughput_evidence_status", "invalid_unspecified")
        )
        raise ValueError(
            "causal summary throughput evidence is invalid and cannot be "
            f"assembled for ResearchFlow: {status}"
        )
    phase = _read_json(root / "streaming_phase_metadata.json")
    rows = _read_jsonl(root / "psnr_history.jsonl")
    if len(rows) < 3:
        raise ValueError("live AA evidence requires at least 3 PSNR samples")

    endpoint = phase.get("mapping_endpoint")
    if not isinstance(endpoint, dict) or "iteration" not in endpoint:
        raise ValueError("mapping endpoint iteration is missing")
    endpoint_iteration = int(endpoint["iteration"])
    mapping_rows = [row for row in rows if int(row.get("iteration", -1)) <= endpoint_iteration]
    if not mapping_rows:
        raise ValueError("PSNR history has no mapping-endpoint sample")
    mapping_sample = max(mapping_rows, key=lambda row: int(row["iteration"]))
    mapping_sample_iteration = int(mapping_sample["iteration"])
    mapping_sample_gap_updates = endpoint_iteration - mapping_sample_iteration
    if mapping_sample_gap_updates > 200:
        raise ValueError("mapping PSNR sample is more than 200 updates before the endpoint")

    for index, row in enumerate(rows):
        actual_mode = row.get("rasterize_mode")
        if actual_mode != expected_rasterize_mode:
            raise ValueError(
                "rasterize mode mismatch in PSNR history row "
                f"{index}: expected {expected_rasterize_mode!r}, got {actual_mode!r}"
            )
    missing = [name for name in _MAPPING_SAMPLE_FIELDS if name not in mapping_sample]
    if missing:
        raise ValueError("mapping PSNR sample is missing: " + ", ".join(missing))

    git_provenance = report.get("git_provenance")
    if not isinstance(git_provenance, dict):
        raise ValueError("mapping report git provenance is missing")
    actual_git_sha = str(git_provenance.get("git_commit", ""))
    if actual_git_sha != expected_git_sha:
        raise ValueError(
            f"git commit mismatch: expected {expected_git_sha}, got {actual_git_sha}"
        )
    if git_provenance.get("git_dirty") is not False:
        raise ValueError("mapping report must prove git_dirty=false")

    eval_metadata = report.get("eval_metadata")
    if not isinstance(eval_metadata, dict) or not eval_metadata.get("holdout_hash"):
        raise ValueError("mapping report holdout hash is missing")
    if report.get("eval_valid") is not True or int(report.get("n_test", 0)) <= 0:
        raise ValueError("mapping endpoint evaluation is not valid")

    report_psnr = _finite_number(report.get("mean_test_psnr"), name="mean_test_psnr")
    _finite_number(mapping_sample.get("mean_test_psnr"), name="mapping history PSNR")

    experiment_origin = report.get("experiment_origin")
    if not isinstance(experiment_origin, dict):
        raise ValueError("mapping report experiment origin is missing")
    for field in ("researchflow_run_id", "researchflow_plan_sha256"):
        if not experiment_origin.get(field):
            raise ValueError(f"mapping report experiment origin lacks {field}")

    throughput = {
        "optimizer_updates_per_s": _finite_number(
            summary.get("optimizer_steps_per_active_s"),
            name="optimizer updates per active second",
        ),
        "supervised_views_per_s": _finite_number(
            summary.get("supervision_views_per_active_s"),
            name="supervision views per active second",
        ),
        "mapping_active_s": _finite_number(
            summary.get("mapping_active_s"), name="mapping active seconds"
        ),
        "optimizer_updates": int(summary.get("optimizer_steps", 0)),
        "supervision_views": int(summary.get("supervision_views", 0)),
    }
    if expected_batch_size < 1 or expected_camera_count < 1:
        raise ValueError("expected B/C must be positive")
    if expected_camera_count > expected_batch_size:
        raise ValueError("expected C cannot exceed expected B")
    expected_constellation = f"B{expected_batch_size}_C{expected_camera_count}"
    requested_constellations = summary.get(
        "requested_supervision_constellations"
    )
    realized_constellations = summary.get("realized_supervision_constellations")
    constellation_modes = summary.get("supervision_constellation_modes")
    constellation_reasons = summary.get("supervision_constellation_reasons")
    expected_histogram = {expected_constellation: throughput["optimizer_updates"]}
    if requested_constellations != expected_histogram:
        raise ValueError(
            "requested supervision constellation does not match the registered "
            f"{expected_constellation} contract: {requested_constellations!r}"
        )
    if realized_constellations != expected_histogram:
        raise ValueError(
            "realized supervision constellation does not match the registered "
            f"{expected_constellation} contract: {realized_constellations!r}"
        )
    expected_mode = (
        "single_camera_immediate"
        if expected_batch_size == expected_camera_count == 1
        else "sequential_accumulation"
        if expected_camera_count == 1
        else "true_camera_microbatch_accumulation"
        if expected_camera_count < expected_batch_size
        else "true_camera_batch"
    )
    if constellation_modes != {expected_mode: throughput["optimizer_updates"]}:
        raise ValueError(
            "realized supervision mode does not match registered B/C: "
            f"{constellation_modes!r}"
        )
    if constellation_reasons != {
        "fixed_constellation_ablation": throughput["optimizer_updates"]
    }:
        raise ValueError(
            "supervision constellation was not controlled by the fixed "
            f"ablation policy: {constellation_reasons!r}"
        )
    mapping_updates = int(phase.get("mapping_optimizer_steps", -1))
    refinement_updates = int(phase.get("refinement_optimizer_steps", -1))
    requested_refinement_updates = int(
        phase.get("refinement_optimizer_steps_requested", -1)
    )
    if mapping_updates != throughput["optimizer_updates"]:
        raise ValueError("mapping optimizer clocks disagree across artifacts")
    if refinement_updates != requested_refinement_updates:
        raise ValueError("post-refinement did not realize its requested optimizer updates")
    total_updates = mapping_updates + refinement_updates
    if total_updates < 1000:
        raise ValueError(
            "live AA evidence requires at least 1000 total optimizer updates"
        )

    psnr_history = {
        "mapping_endpoint_iteration": endpoint_iteration,
        "mapping_sample_iteration": mapping_sample_iteration,
        "mapping_sample_gap_updates": mapping_sample_gap_updates,
        "sample_count": len(rows),
        "samples": rows,
    }
    psnr_history.update({name: mapping_sample[name] for name in _MAPPING_SAMPLE_FIELDS})

    return {
        "schema_version": 1,
        "eval_valid": True,
        "n_test": int(report["n_test"]),
        "mean_test_psnr": report_psnr,
        "psnr_history": psnr_history,
        "throughput": throughput,
        "supervision_constellations": {
            "expected": expected_constellation,
            "requested": requested_constellations,
            "realized": realized_constellations,
            "modes": constellation_modes,
            "reasons": constellation_reasons,
        },
        "phase": {**phase, "total_optimizer_updates": total_updates},
        "eval_metadata": eval_metadata,
        "git_provenance": git_provenance,
        "experiment_origin": experiment_origin,
        "source_artifacts": {
            "mapping_report": str(root / "comparison/mapping_endpoint/report.json"),
            "psnr_history": str(root / "psnr_history.jsonl"),
            "causal_summary": str(root / "causal_replay_summary.json"),
            "phase_metadata": str(root / "streaming_phase_metadata.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-rasterize-mode",
        choices=("classic", "antialiased"),
        required=True,
    )
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-batch-size", required=True, type=int)
    parser.add_argument("--expected-camera-count", required=True, type=int)
    args = parser.parse_args()
    metrics = assemble_metrics(
        args.model_path,
        expected_rasterize_mode=args.expected_rasterize_mode,
        expected_git_sha=args.expected_git_sha,
        expected_batch_size=args.expected_batch_size,
        expected_camera_count=args.expected_camera_count,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
