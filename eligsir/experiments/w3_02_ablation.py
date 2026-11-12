"""Frozen A/B evidence reader for the Wave 3 W3-02 engine extraction.

This module intentionally consumes artifacts written by real training runs:
``psnr_history.jsonl`` and the existing ``--benchmark_dir``
``timings.jsonl``.  It never runs a benchmark and never substitutes missing
measurements.  The timing source is *wall-clock total iteration time*, not a
GPU-event-only span; the engine does not yet emit a per-step GPU timing field,
so callers must not represent this report as GPU-exclusive evidence.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping

from eligsir.experiments.manifest import ExperimentManifest
from eligsir.experiments.hashing import content_hash
from eligsir.experiments.validation import ManifestValidationError, validate_manifest

W3_02_ENGINE_CHANGED_FIELD = "implementation.gaussian_optimization_engine"
_THROUGHPUT_REGRESSION_BUDGET_PERCENT = 2.0


class EvidenceValidationError(ValueError):
    """Raised when recorded artifacts cannot support a frozen W3-02 A/B claim."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_w3_02_manifest(manifest: ExperimentManifest) -> None:
    try:
        validate_manifest(manifest, strict=True)
    except ManifestValidationError as error:
        raise EvidenceValidationError(str(error)) from error

    if manifest.frozen_level != "D":
        raise EvidenceValidationError(
            "W3-02 engine parity evidence requires frozen_level='D' so the mutation trace is frozen"
        )
    if not manifest.validity.mapping_result_valid:
        raise EvidenceValidationError("mapping_result_valid must be true for quotable W3-02 evidence")
    if manifest.validity.source_frames_dropped:
        raise EvidenceValidationError("frozen W3-02 evidence requires source_frames_dropped=0")
    if manifest.validity.map_reinitializations:
        raise EvidenceValidationError("frozen W3-02 evidence requires map_reinitializations=0")

    arms = {arm.name: arm.changed_fields for arm in manifest.arms}
    if not {"baseline", "treatment"}.issubset(arms) or set(arms) - {"baseline", "treatment", "negative_control"}:
        raise EvidenceValidationError("W3-02 evidence requires baseline/treatment arms (and optional negative_control)")
    if arms["treatment"] != [W3_02_ENGINE_CHANGED_FIELD]:
        raise EvidenceValidationError(
            "W3-02 treatment must declare only "
            f"{W3_02_ENGINE_CHANGED_FIELD!r} as its independent variable"
        )

    commits = manifest.arm_repository_commits
    if commits is None or not {"baseline", "treatment"}.issubset(commits):
        raise EvidenceValidationError(
            "W3-02 evidence requires arm_repository_commits for baseline and treatment"
        )
    if any(len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit) for commit in commits.values()):
        raise EvidenceValidationError("W3-02 arm_repository_commits must be 40-character lowercase Git SHAs")
    if commits["baseline"] == commits["treatment"]:
        raise EvidenceValidationError("W3-02 baseline and treatment must use distinct repository commits")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceValidationError(f"required recorded artifact is missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise EvidenceValidationError(f"invalid JSON in {path}:{line_number}: {error.msg}") from error
        if not isinstance(data, dict):
            raise EvidenceValidationError(f"JSON record in {path}:{line_number} must be an object")
        records.append(data)
    if not records:
        raise EvidenceValidationError(f"recorded artifact has no JSON records: {path}")
    return records


def _finite_number(value: Any, *, field: str, path: Path, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceValidationError(f"{path}:{line_number} has no finite numeric {field}")
    return float(value)


def _indexed_values(
    records: Iterable[dict[str, Any]],
    *,
    path: Path,
    value_key: str,
    nested_timing: bool = False,
) -> dict[int, float]:
    values: dict[int, float] = {}
    previous_iteration: int | None = None
    for line_number, row in enumerate(records, start=1):
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise EvidenceValidationError(f"{path}:{line_number} has no non-negative integer iteration")
        if previous_iteration is not None and iteration <= previous_iteration:
            raise EvidenceValidationError(
                f"{path}:{line_number} iterations must be strictly increasing; use a fresh run directory"
            )
        previous_iteration = iteration
        if nested_timing:
            timing = row.get("timing_ms")
            if not isinstance(timing, dict):
                raise EvidenceValidationError(f"{path}:{line_number} timing_ms must be an object")
            raw_value = timing.get(value_key)
        else:
            raw_value = row.get(value_key)
        values[iteration] = _finite_number(raw_value, field=value_key, path=path, line_number=line_number)
    return values


def _same_schedule(kind: str, baseline: Mapping[int, float], treatment: Mapping[int, float]) -> list[int]:
    baseline_iterations = list(baseline)
    treatment_iterations = list(treatment)
    if baseline_iterations != treatment_iterations:
        raise EvidenceValidationError(
            f"{kind} iterations differ between baseline and treatment; a frozen A/B needs the same schedule"
        )
    return baseline_iterations


def _throughput_summary(baseline: Mapping[int, float], treatment: Mapping[int, float]) -> dict[str, Any]:
    iterations = _same_schedule("GPU-event benchmark", baseline, treatment)
    if len(iterations) < 3:
        raise EvidenceValidationError("W3-02 throughput evidence requires at least three benchmark samples")
    steady_start = (2 * len(iterations)) // 3
    steady_iterations = iterations[steady_start:]
    baseline_mean_ms = sum(baseline[i] for i in steady_iterations) / len(steady_iterations)
    treatment_mean_ms = sum(treatment[i] for i in steady_iterations) / len(steady_iterations)
    baseline_hz = 1000.0 / baseline_mean_ms
    treatment_hz = 1000.0 / treatment_mean_ms
    regression_percent = ((baseline_hz - treatment_hz) / baseline_hz) * 100.0
    return {
        "metric": "optimization_gpu_events.gpu_ms",
        "measurement_scope": "CUDA/HIP GPU event for the complete fixed-topology optimization step",
        "steady_state_selection": "final third of matching GPU-event samples",
        "steady_state_iterations": steady_iterations,
        "baseline_mean_ms": baseline_mean_ms,
        "treatment_mean_ms": treatment_mean_ms,
        "baseline_steps_hz": baseline_hz,
        "treatment_steps_hz": treatment_hz,
        "regression_percent": regression_percent,
        "regression_budget_percent": _THROUGHPUT_REGRESSION_BUDGET_PERCENT,
    }


def _gpu_event_values(run_dir: Path, manifest: ExperimentManifest, arm: str) -> tuple[dict[int, float], bool]:
    path = run_dir / "w3_02_runtime_evidence.json"
    if not path.is_file():
        raise EvidenceValidationError(
            f"required GPU-event runtime evidence is missing: {path}; wall-only timings cannot support W3-02 performance acceptance"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise EvidenceValidationError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("optimization_gpu_events"), list):
        raise EvidenceValidationError(f"{path} has no optimization_gpu_events list")
    if payload.get("manifest_hash") != content_hash(manifest.to_dict()):
        raise EvidenceValidationError(f"{path} is not bound to the supplied Level-D manifest")
    if payload.get("arm") != arm:
        raise EvidenceValidationError(f"{path} arm does not match {arm!r}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise EvidenceValidationError(f"{path} has no runtime provenance")
    if runtime.get("git_commit") != manifest.arm_repository_commits[arm]:
        raise EvidenceValidationError(f"{path} Git commit does not match the {arm} manifest commit")
    if runtime.get("clean_checkout") is not True:
        raise EvidenceValidationError(f"{path} was produced from a dirty checkout")
    seed_match = re.search(r"\bfixed\s+seed\s*=\s*(\d+)\b", manifest.seed_policy)
    if seed_match is None or runtime.get("seed") != int(seed_match.group(1)):
        raise EvidenceValidationError(f"{path} seed does not match a 'fixed seed=N' manifest policy")
    if runtime.get("initial_checkpoint_hash") != manifest.initial_checkpoint_hash:
        raise EvidenceValidationError(f"{path} initial checkpoint hash does not match the manifest")
    if runtime.get("mutation_trace_hash") != manifest.mutation_trace_hash:
        raise EvidenceValidationError(f"{path} mutation trace hash does not match the manifest")
    frozen = runtime.get("frozen_artifacts")
    if not isinstance(frozen, dict):
        raise EvidenceValidationError(f"{path} has no frozen runtime artifact manifest")
    for field in ("config_hash", "source_artifact_hash", "pose_hash", "keyframe_hash", "holdout_hash"):
        entry = frozen.get(field)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise EvidenceValidationError(f"{path} is missing frozen artifact {field}")
        try:
            actual = _sha256_file(entry["path"])
        except OSError as error:
            raise EvidenceValidationError(f"{path} frozen artifact {field} is unavailable: {error}") from error
        if actual != entry.get("sha256") or actual != getattr(manifest, field):
            raise EvidenceValidationError(f"{path} frozen artifact {field} hash does not match manifest/runtime")
    output_hashes = runtime.get("artifact_hashes")
    output_paths = runtime.get("artifact_paths")
    if not isinstance(output_hashes, dict) or not isinstance(output_paths, dict):
        raise EvidenceValidationError(f"{path} has no output artifact hash/path manifest")
    for name in ("cfg_args", "reproduce.sh", "psnr_history.jsonl", "final_report"):
        try:
            actual = _sha256_file(output_paths[name])
        except (KeyError, OSError) as error:
            raise EvidenceValidationError(f"{path} required output artifact {name} is unavailable") from error
        if actual != output_hashes.get(name):
            raise EvidenceValidationError(f"{path} output artifact {name} was modified after runtime evidence")
    report = json.loads(Path(output_paths["final_report"]).read_text())
    if not report.get("eval_valid") or int(report.get("n_test", 0)) <= 0:
        raise EvidenceValidationError(f"{path} final report is not evaluator-valid")
    report_holdout = (report.get("eval_metadata") or {}).get("holdout_hash")
    if not isinstance(report_holdout, str) or not re.fullmatch(r"[0-9a-f]{64}", report_holdout):
        raise EvidenceValidationError(f"{path} final report holdout hash is not a full SHA-256 digest")
    if report_holdout != manifest.holdout_hash:
        raise EvidenceValidationError(f"{path} final report holdout hash does not match manifest")
    coverage = report.get("trajectory_coverage") or (report.get("geometry") or {}).get("trajectory_coverage")
    if not isinstance(coverage, dict):
        raise EvidenceValidationError(f"{path} final report has no trajectory coverage validity record")
    if int(coverage.get("frames_dropped_by_inner_queue", 0) or 0) != 0:
        raise EvidenceValidationError(f"{path} final report records dropped source frames")
    if coverage.get("map_reinit_invalidated") is True or coverage.get("tracking_reinit_events"):
        raise EvidenceValidationError(f"{path} final report records tracking/map reinitialization")
    if str(coverage.get("verdict", "")).upper() in {"SUSPECTED_TRUNCATED", "TRUNCATED", "SUSPECTED"}:
        raise EvidenceValidationError(f"{path} final report has a truncated or suspect coverage verdict")
    values: dict[int, float] = {}
    previous_iteration: int | None = None
    for index, row in enumerate(payload["optimization_gpu_events"], start=1):
        if not isinstance(row, dict):
            raise EvidenceValidationError(f"{path}: GPU-event row {index} must be an object")
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise EvidenceValidationError(f"{path}: GPU-event row {index} has no non-negative integer iteration")
        if previous_iteration is not None and iteration <= previous_iteration:
            raise EvidenceValidationError(f"{path}: GPU-event iterations must be strictly increasing")
        previous_iteration = iteration
        values[iteration] = _finite_number(row.get("gpu_ms"), field="gpu_ms", path=path, line_number=index)
    if not values:
        raise EvidenceValidationError(f"{path} has no GPU-event samples; performance acceptance is unavailable")
    return values, runtime.get("mutation_trace_applied") is True and runtime.get("trustworthy_attestation") is True


def build_w3_02_evidence_report(
    manifest: ExperimentManifest,
    baseline_run_dir: str | Path,
    treatment_run_dir: str | Path,
    *,
    claim_improvement: bool = False,
    negative_control_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read real run artifacts and return an auditable W3-02 A/B summary.

    Each run directory must contain the standard streaming PSNR history plus
    ``timings.jsonl`` produced by ``--benchmark_dir model``.  Quality is
    reported, not thresholded: the plan leaves the stochastic PSNR envelope
    to measured controls.  The throughput acceptance gate is exactly the
    plan's no-more-than-two-percent regression rule.
    """
    _require_w3_02_manifest(manifest)
    if claim_improvement:
        if "negative_control" not in {arm.name for arm in manifest.arms}:
            raise EvidenceValidationError("a claimed improvement requires a declared negative_control arm")
        if negative_control_run_dir is None or not Path(negative_control_run_dir).is_dir():
            raise EvidenceValidationError("a claimed improvement requires a completed negative_control run directory")
        if "negative_control" not in (manifest.arm_repository_commits or {}):
            raise EvidenceValidationError("a claimed improvement requires a negative_control commit")
        control_events, control_trace = _gpu_event_values(Path(negative_control_run_dir), manifest, "negative_control")
    baseline_dir = Path(baseline_run_dir)
    treatment_dir = Path(treatment_run_dir)
    baseline_history_path = baseline_dir / "psnr_history.jsonl"
    treatment_history_path = treatment_dir / "psnr_history.jsonl"
    baseline_records = _read_jsonl(baseline_history_path)
    treatment_records = _read_jsonl(treatment_history_path)
    control_history_path = None
    control_records = None
    if claim_improvement:
        control_history_path = Path(negative_control_run_dir) / "psnr_history.jsonl"
        control_records = _read_jsonl(control_history_path)
    from eligsir.utils.psnr_history_io import psnr_reduction

    try:
        reductions = {
            "baseline": psnr_reduction(baseline_records, source=str(baseline_history_path)),
            "treatment": psnr_reduction(treatment_records, source=str(treatment_history_path)),
        }
        if control_records is not None:
            reductions["negative_control"] = psnr_reduction(
                control_records,
                source=str(control_history_path),
            )
    except ValueError as error:
        raise EvidenceValidationError(str(error)) from error
    if len(set(reductions.values())) != 1:
        raise EvidenceValidationError(
            "PSNR reduction mismatch: " + " ".join(
                f"{arm}={reduction!r}" for arm, reduction in reductions.items()
            )
        )
    baseline_reduction = reductions["baseline"]
    baseline_psnr = _indexed_values(
        baseline_records,
        path=baseline_dir / "psnr_history.jsonl",
        value_key="mean_test_psnr",
    )
    treatment_psnr = _indexed_values(
        treatment_records,
        path=treatment_dir / "psnr_history.jsonl",
        value_key="mean_test_psnr",
    )
    psnr_iterations = _same_schedule("PSNR evaluation", baseline_psnr, treatment_psnr)
    negative_control = None
    if claim_improvement:
        control_psnr = _indexed_values(control_records, path=control_history_path, value_key="mean_test_psnr")
        _same_schedule("negative-control PSNR evaluation", baseline_psnr, control_psnr)
        negative_control = {"evaluation_iterations": psnr_iterations, "final_psnr_delta_db": control_psnr[psnr_iterations[-1]] - baseline_psnr[psnr_iterations[-1]], "mutation_trace_and_attestation_valid": control_trace, "gpu_event_samples": len(control_events)}
    if not psnr_iterations:
        raise EvidenceValidationError("W3-02 evidence requires at least one PSNR evaluation")
    baseline_gpu, baseline_trace_applied = _gpu_event_values(baseline_dir, manifest, "baseline")
    treatment_gpu, treatment_trace_applied = _gpu_event_values(treatment_dir, manifest, "treatment")
    throughput = _throughput_summary(baseline_gpu, treatment_gpu)
    final_iteration = psnr_iterations[-1]
    final_delta = treatment_psnr[final_iteration] - baseline_psnr[final_iteration]
    mean_delta = sum(treatment_psnr[i] - baseline_psnr[i] for i in psnr_iterations) / len(psnr_iterations)
    throughput_within_budget = throughput["regression_percent"] <= _THROUGHPUT_REGRESSION_BUDGET_PERCENT
    mutation_trace_applied = baseline_trace_applied and treatment_trace_applied
    return {
        "manifest": manifest.to_dict(),
        "run_directories": {"baseline": str(baseline_dir), "treatment": str(treatment_dir)},
        "quality": {
            "metric": "mean_test_psnr_db",
            "psnr_reduction": baseline_reduction,
            "evaluation_iterations": psnr_iterations,
            "baseline_curve_db": [baseline_psnr[i] for i in psnr_iterations],
            "treatment_curve_db": [treatment_psnr[i] for i in psnr_iterations],
            "baseline_final_psnr_db": baseline_psnr[final_iteration],
            "treatment_final_psnr_db": treatment_psnr[final_iteration],
            "final_psnr_delta_db": final_delta,
            "mean_psnr_delta_db": mean_delta,
            "acceptance_note": "Reported for comparison; no PSNR tolerance is asserted without a measured control envelope.",
        },
        "throughput": throughput,
        "negative_control": negative_control,
        "acceptance": {
            "mapping_result_valid": True,
            "mutation_trace_applied": mutation_trace_applied,
            "negative_control_verified": not claim_improvement or negative_control_run_dir is not None,
            "throughput_within_two_percent": throughput_within_budget,
            "passed": throughput_within_budget and mutation_trace_applied,
        },
    }


def write_w3_02_evidence_report(
    manifest: ExperimentManifest,
    baseline_run_dir: str | Path,
    treatment_run_dir: str | Path,
    output_path: str | Path,
    *,
    claim_improvement: bool = False,
    negative_control_run_dir: str | Path | None = None,
) -> Path:
    """Build and atomically write a W3-02 report after validating all inputs."""
    report = build_w3_02_evidence_report(
        manifest, baseline_run_dir, treatment_run_dir,
        claim_improvement=claim_improvement, negative_control_run_dir=negative_control_run_dir,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return target


def build_ablator_spec(
    manifest: ExperimentManifest,
    base: Mapping[str, Any],
    *,
    arm_name: str,
    runtime_manifest_path: str,
    mutation_trace_path: str,
    frozen_artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build one standard ablator spec for a frozen W3-02 arm.

    ``base`` is copied from an experiment owner-approved existing ablator
    spec.  The helper adds only the required PSNR and timing artifact flags;
    the arm has no additional train arguments because its sole change is the
    checkout identified by ``manifest.arm_repository_commits``.  Generating
    one arm at a time is intentional: the ablator spec format available in
    this checkout has no code-checkout selector, so emitting both arms from
    one working tree would accidentally benchmark the same implementation.
    """
    _require_w3_02_manifest(manifest)
    if arm_name not in {"baseline", "treatment", "negative_control"} or arm_name not in {arm.name for arm in manifest.arms}:
        raise EvidenceValidationError("arm_name must be a manifest-declared baseline, treatment, or negative_control")
    if not runtime_manifest_path or runtime_manifest_path.startswith("/"):
        raise EvidenceValidationError(
            "runtime_manifest_path must be a non-empty repository-relative path visible in the train container"
        )
    if not mutation_trace_path or mutation_trace_path.startswith("/"):
        raise EvidenceValidationError("mutation_trace_path must be a repository-relative path visible in the train container")
    required_artifacts = {
        "config_hash": "w3_02_config_artifact",
        "source_artifact_hash": "w3_02_source_artifact",
        "pose_hash": "w3_02_pose_artifact",
        "keyframe_hash": "w3_02_keyframe_artifact",
        "holdout_hash": "w3_02_holdout_artifact",
    }
    if set(frozen_artifact_paths) != set(required_artifacts):
        raise EvidenceValidationError(
            "frozen_artifact_paths must provide exactly: " + ", ".join(sorted(required_artifacts))
        )
    if any(not isinstance(path, str) or not path for path in frozen_artifact_paths.values()):
        raise EvidenceValidationError("each frozen artifact path must be a non-empty string")
    required = {"type", "scene", "iterations", "machine", "base_args"}
    missing = sorted(required - set(base))
    if missing:
        raise EvidenceValidationError(f"ablator base is missing required fields: {missing}")
    if not isinstance(base["base_args"], str):
        raise EvidenceValidationError("ablator base_args must be a string")
    existing_args = shlex.split(base["base_args"])
    conflicting = {
        flag
        for flag in (
            "--benchmark_dir",
            "--streaming_psnr_chart_interval",
            "--w3_02_evidence_manifest",
            "--w3_02_evidence_arm",
        )
        if flag in existing_args
    }
    if conflicting:
        raise EvidenceValidationError(
            "ablator base_args must not override W3-02 evidence logging flags: " + ", ".join(sorted(conflicting))
        )
    spec_base = dict(base)
    spec_base["base_args"] = (
        f"{base['base_args'].strip()} --benchmark_dir model "
        f"--streaming_psnr_chart_interval 500 "
        f"--w3_02_evidence_manifest {shlex.quote(runtime_manifest_path)} "
        f"--w3_02_evidence_arm {arm_name} "
        f"--w3_02_mutation_trace {shlex.quote(mutation_trace_path)} "
        f"--w3_02_git_commit {manifest.arm_repository_commits[arm_name]} --w3_02_clean_checkout "
        + " ".join(
            f"--{flag} {shlex.quote(frozen_artifact_paths[field])}"
            for field, flag in required_artifacts.items()
        )
    ).strip()
    return {
        "name": f"{manifest.experiment_id}-{arm_name}".replace("_", "-")[:48],
        "parallel": False,
        "lane": 1,
        "base": spec_base,
        "arms": [{"id": arm_name, "extra_args": ""}],
        "notes": (
            f"Generated frozen W3-02 {arm_name} engine spec for repository commit "
            f"{manifest.arm_repository_commits[arm_name]}. Run this spec only from that verified "
            "checkout; do not add train flags per arm. Use ablator sequentially and validate the "
            "two resulting model directories with scripts/validate_w3_02_ablation.py."
        ),
    }
