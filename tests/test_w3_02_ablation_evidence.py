"""Tests for the frozen W3-02 engine A/B evidence contract.

These tests use only tiny JSON artifacts.  They deliberately do not exercise a
GPU or fabricate a benchmark result: the production helper must consume the
files a real ablator run wrote.
"""

from __future__ import annotations

import json
import hashlib
import shlex

import pytest

from eligsir.experiments.manifest import (
    ArmSpec,
    ExperimentManifest,
    HardwareInfo,
    ValidityInfo,
)
from eligsir.experiments.hashing import content_hash
from eligsir.experiments.w3_02_ablation import (
    W3_02_ENGINE_CHANGED_FIELD,
    EvidenceValidationError,
    build_ablator_spec,
    build_w3_02_evidence_report,
)
from eligsir.cli.parser import build_training_parser


_HASH = "a" * 64


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path=None, **overrides) -> ExperimentManifest:
    data = dict(
        experiment_id="w3-02-engine-parity",
        hypothesis="Extracting the fixed-topology engine preserves quality and throughput.",
        independent_variable="implementation location of the Gaussian optimization step",
        frozen_level="D",
        repository_commit="1" * 40,
        arm_repository_commits={"baseline": "2" * 40, "treatment": "3" * 40},
        source_artifact_hash=_HASH,
        pose_hash=_HASH,
        holdout_hash=_HASH,
        keyframe_hash=_HASH,
        initial_checkpoint_hash=_HASH,
        mutation_trace_hash=_HASH,
        config_hash=_HASH,
        seed_policy="fixed seed=7 and recorded generator states",
        hardware=HardwareInfo(gpu="R9700", driver="7.2", torch="2.7", backend="rocm"),
        validity=ValidityInfo(
            mapping_result_valid=True,
            source_frames_dropped=0,
            map_reinitializations=0,
        ),
        arms=[
            ArmSpec(name="baseline", changed_fields=[]),
            ArmSpec(name="treatment", changed_fields=[W3_02_ENGINE_CHANGED_FIELD]),
        ],
    )
    if tmp_path is not None:
        frozen = tmp_path / "frozen"
        frozen.mkdir(exist_ok=True)
        fields = {
            "config_hash": "config.json",
            "source_artifact_hash": "source.jsonl",
            "pose_hash": "poses.jsonl",
            "keyframe_hash": "keyframes.jsonl",
            "holdout_hash": "holdout.json",
        }
        for field, filename in fields.items():
            artifact = frozen / filename
            artifact.write_text(field + "\n")
            data[field] = _sha256(artifact)
    data.update(overrides)
    return ExperimentManifest(**data)


def _write_run(run_dir, manifest, *, psnr, wall_ms, psnr_reduction=None) -> None:
    run_dir.mkdir()
    (run_dir / "psnr_history.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "iteration": iteration,
                    "mean_test_psnr": value,
                    **({"psnr_reduction": psnr_reduction} if psnr_reduction else {}),
                }
            ) + "\n"
            for iteration, value in psnr
        )
    )
    (run_dir / "timings.jsonl").write_text(
        "".join(
            json.dumps({"iteration": iteration, "timing_ms": {"total_wall": value}}) + "\n"
            for iteration, value in wall_ms
        )
    )
    (run_dir / "cfg_args").write_text("Namespace(seed=7)\n")
    (run_dir / "reproduce.sh").write_text("#!/bin/sh\n")
    report_dir = run_dir / "comparison" / "iter_600"
    report_dir.mkdir(parents=True)
    report = report_dir / "report.json"
    report.write_text(json.dumps({
        "eval_valid": True,
        "n_test": 2,
        "eval_metadata": {"holdout_hash": manifest.holdout_hash},
        "trajectory_coverage": {"frames_dropped_by_inner_queue": 0, "map_reinit_invalidated": False, "tracking_reinit_events": [], "verdict": "COMPLETE"},
    }))
    frozen = run_dir.parent / "frozen"
    frozen_entries = {
        field: {"path": str(frozen / filename), "sha256": _sha256(frozen / filename)}
        for field, filename in {
            "config_hash": "config.json",
            "source_artifact_hash": "source.jsonl",
            "pose_hash": "poses.jsonl",
            "keyframe_hash": "keyframes.jsonl",
            "holdout_hash": "holdout.json",
        }.items()
    }
    (run_dir / "w3_02_runtime_evidence.json").write_text(
        json.dumps(
            {
                "manifest_hash": content_hash(manifest.to_dict()),
                "arm": run_dir.name,
                "optimization_gpu_events": [
                    {"iteration": iteration, "gpu_ms": value}
                    for iteration, value in wall_ms
                ],
                "runtime": {
                    "git_commit": manifest.arm_repository_commits[run_dir.name],
                    "clean_checkout": True,
                    "seed": 7,
                    "initial_checkpoint_hash": _HASH,
                    "mutation_trace_hash": _HASH,
                    "mutation_trace_applied": True,
                    "frozen_artifacts": frozen_entries,
                    "artifact_paths": {
                        "cfg_args": str(run_dir / "cfg_args"),
                        "reproduce.sh": str(run_dir / "reproduce.sh"),
                        "psnr_history.jsonl": str(run_dir / "psnr_history.jsonl"),
                        "final_report": str(report),
                    },
                    "artifact_hashes": {
                        "cfg_args": _sha256(run_dir / "cfg_args"),
                        "reproduce.sh": _sha256(run_dir / "reproduce.sh"),
                        "psnr_history.jsonl": _sha256(run_dir / "psnr_history.jsonl"),
                        "final_report": _sha256(report),
                    },
                },
            }
        )
    )


def test_build_ablator_spec_builds_one_logged_arm_for_its_verified_checkout():
    spec = build_ablator_spec(
        _manifest(),
        base={
            "type": "replay",
            "scene": "/data/frozen-scene",
            "iterations": 6000,
            "machine": "r9700",
            "base_args": "--streaming_replay --cap_max 200000",
        },
        arm_name="baseline",
        runtime_manifest_path="specs/w3-02-engine-manifest.json",
        mutation_trace_path="specs/w3-02-mutation-trace.jsonl",
        frozen_artifact_paths={
            "config_hash": "specs/config.json",
            "source_artifact_hash": "/data/source.jsonl",
            "pose_hash": "/data/poses.jsonl",
            "keyframe_hash": "/data/keyframes.jsonl",
            "holdout_hash": "/data/holdout.json",
        },
    )

    assert spec["parallel"] is False
    assert "--benchmark_dir model" in spec["base"]["base_args"]
    assert "--streaming_psnr_chart_interval 500" in spec["base"]["base_args"]
    assert "--w3_02_evidence_manifest specs/w3-02-engine-manifest.json" in spec["base"]["base_args"]
    assert "--w3_02_evidence_arm baseline" in spec["base"]["base_args"]
    assert "--w3_02_mutation_trace specs/w3-02-mutation-trace.jsonl" in spec["base"]["base_args"]
    assert "--w3_02_config_artifact specs/config.json" in spec["base"]["base_args"]
    assert spec["arms"] == [{"id": "baseline", "extra_args": ""}]
    assert "2222222222222222222222222222222222222222" in spec["notes"]


def test_generated_ablator_arguments_parse_into_the_runtime_artifact_bridge():
    spec = build_ablator_spec(
        _manifest(),
        base={"type": "replay", "scene": "/data/frozen", "iterations": 10, "machine": "r9700", "base_args": "--cap_max 2000"},
        arm_name="baseline",
        runtime_manifest_path="specs/manifest.json",
        mutation_trace_path="specs/trace.jsonl",
        frozen_artifact_paths={
            "config_hash": "specs/config.json",
            "source_artifact_hash": "/data/source.jsonl",
            "pose_hash": "/data/poses.jsonl",
            "keyframe_hash": "/data/keyframes.jsonl",
            "holdout_hash": "/data/holdout.json",
        },
    )

    args = build_training_parser().parser.parse_args(shlex.split(spec["base"]["base_args"]))

    assert args.w3_02_mutation_trace == "specs/trace.jsonl"
    assert args.w3_02_config_artifact == "specs/config.json"
    assert args.w3_02_source_artifact == "/data/source.jsonl"
    assert args.w3_02_holdout_artifact == "/data/holdout.json"


def test_negative_control_spec_and_parser_use_the_declared_control_checkout():
    manifest = _manifest(
        arms=[
            ArmSpec(name="baseline", changed_fields=[]),
            ArmSpec(name="treatment", changed_fields=[W3_02_ENGINE_CHANGED_FIELD]),
            ArmSpec(name="negative_control", changed_fields=["implementation.deliberate_noop_control"]),
        ],
        arm_repository_commits={"baseline": "2" * 40, "treatment": "3" * 40, "negative_control": "4" * 40},
    )
    spec = build_ablator_spec(
        manifest,
        base={"type": "replay", "scene": "/data/frozen", "iterations": 10, "machine": "r9700", "base_args": "--cap_max 2000"},
        arm_name="negative_control",
        runtime_manifest_path="specs/manifest.json",
        mutation_trace_path="specs/trace.jsonl",
        frozen_artifact_paths={
            "config_hash": "specs/config.json", "source_artifact_hash": "/data/source.jsonl",
            "pose_hash": "/data/poses.jsonl", "keyframe_hash": "/data/keyframes.jsonl",
            "holdout_hash": "/data/holdout.json",
        },
    )

    args = build_training_parser().parser.parse_args(shlex.split(spec["base"]["base_args"]))

    assert args.w3_02_evidence_arm == "negative_control"
    assert "4444444444444444444444444444444444444444" in spec["notes"]


def test_claim_report_parses_and_reports_negative_control_artifacts(tmp_path):
    manifest = _manifest(
        tmp_path,
        arms=[ArmSpec(name="baseline"), ArmSpec(name="treatment", changed_fields=[W3_02_ENGINE_CHANGED_FIELD]), ArmSpec(name="negative_control", changed_fields=["implementation.deliberate_noop_control"])],
        arm_repository_commits={"baseline": "2" * 40, "treatment": "3" * 40, "negative_control": "4" * 40},
    )
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    baseline, treatment, control = tmp_path / "baseline", tmp_path / "treatment", tmp_path / "negative_control"
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(control, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])

    report = build_w3_02_evidence_report(manifest, baseline, treatment, claim_improvement=True, negative_control_run_dir=control)

    assert report["negative_control"]["evaluation_iterations"] == [100, 200, 300]


def test_claim_report_refuses_negative_control_psnr_reduction_mismatch(tmp_path):
    manifest = _manifest(
        tmp_path,
        arms=[
            ArmSpec(name="baseline"),
            ArmSpec(name="treatment", changed_fields=[W3_02_ENGINE_CHANGED_FIELD]),
            ArmSpec(name="negative_control", changed_fields=["implementation.deliberate_noop_control"]),
        ],
        arm_repository_commits={"baseline": "2" * 40, "treatment": "3" * 40, "negative_control": "4" * 40},
    )
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    baseline, treatment, control = tmp_path / "baseline", tmp_path / "treatment", tmp_path / "negative_control"
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(
        control,
        manifest,
        psnr=points,
        wall_ms=[(i, 10.0) for i, _ in points],
        psnr_reduction="rgb_per_view_v2",
    )

    with pytest.raises(EvidenceValidationError, match="PSNR reduction mismatch"):
        build_w3_02_evidence_report(
            manifest,
            baseline,
            treatment,
            claim_improvement=True,
            negative_control_run_dir=control,
        )


def test_evidence_report_compares_same_psnr_curve_and_enforces_two_percent_budget(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    iterations = [100, 200, 300, 400, 500, 600]
    _write_run(
        baseline, manifest,
        psnr=list(zip(iterations, [20.0, 21.0, 22.0, 23.0, 24.0, 25.0])),
        wall_ms=list(zip(iterations, [10.0] * len(iterations))),
    )
    _write_run(
        treatment, manifest,
        psnr=list(zip(iterations, [20.0, 21.1, 22.0, 23.1, 24.0, 25.1])),
        wall_ms=list(zip(iterations, [10.15] * len(iterations))),
    )

    report = build_w3_02_evidence_report(manifest, baseline, treatment)

    assert report["quality"]["evaluation_iterations"] == iterations
    assert report["quality"]["psnr_reduction"] == "legacy_chw_channel_mean_v1"
    assert report["quality"]["final_psnr_delta_db"] == pytest.approx(0.1)
    assert report["throughput"]["steady_state_iterations"] == [500, 600]
    assert report["throughput"]["regression_percent"] == pytest.approx(1.4778325)
    assert report["acceptance"]["throughput_within_two_percent"] is True
    assert report["acceptance"]["passed"] is False


def test_evidence_report_refuses_cross_run_psnr_reduction_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(
        treatment,
        manifest,
        psnr=points,
        wall_ms=[(i, 10.0) for i, _ in points],
        psnr_reduction="rgb_per_view_v2",
    )

    with pytest.raises(EvidenceValidationError, match="PSNR reduction mismatch"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_refuses_a_mixed_psnr_history(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    with (treatment / "psnr_history.jsonl").open("a") as history:
        history.write(
            json.dumps(
                {
                    "iteration": 400,
                    "mean_test_psnr": 23.0,
                    "psnr_reduction": "rgb_per_view_v2",
                }
            )
            + "\n"
        )

    with pytest.raises(EvidenceValidationError, match="mixed PSNR reductions"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_records_a_throughput_gate_failure_without_hiding_measurement(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    manifest = _manifest(tmp_path)
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.5) for i, _ in points])

    report = build_w3_02_evidence_report(manifest, baseline, treatment)

    assert report["throughput"]["regression_percent"] == pytest.approx(4.7619047)
    assert report["acceptance"]["throughput_within_two_percent"] is False
    assert report["acceptance"]["passed"] is False


def test_evidence_report_rejects_mismatched_frozen_evaluation_schedule(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    _write_run(
        baseline, manifest,
        psnr=[(100, 20.0), (200, 21.0)],
        wall_ms=[(100, 10.0), (200, 10.0)],
    )
    _write_run(
        treatment, manifest,
        psnr=[(100, 20.0), (300, 21.0)],
        wall_ms=[(100, 10.0), (200, 10.0)],
    )

    with pytest.raises(EvidenceValidationError, match="PSNR evaluation iterations"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_w3_02_evidence_requires_a_level_d_valid_manifest(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    points = [(100, 20.0), (200, 21.0)]
    manifest = _manifest(tmp_path)
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])

    with pytest.raises(EvidenceValidationError, match="frozen_level='D'"):
        build_w3_02_evidence_report(_manifest(tmp_path, frozen_level="C"), baseline, treatment)


def test_w3_02_evidence_requires_distinct_commits_for_the_two_implementations(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    points = [(100, 20.0), (200, 21.0)]
    manifest = _manifest(tmp_path)
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])

    with pytest.raises(EvidenceValidationError, match="distinct repository commits"):
        build_w3_02_evidence_report(
            _manifest(tmp_path, arm_repository_commits={"baseline": "2" * 40, "treatment": "2" * 40}),
            baseline,
            treatment,
        )


def test_evidence_report_rejects_malformed_gpu_event_records_loudly(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    manifest = _manifest(tmp_path)
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    runtime_path = treatment / "w3_02_runtime_evidence.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["optimization_gpu_events"] = [{"iteration": 100, "gpu_ms": "bad"}]
    runtime_path.write_text(json.dumps(runtime))

    with pytest.raises(EvidenceValidationError, match="finite numeric gpu_ms"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_forbids_performance_acceptance_from_wall_only_rows(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    manifest = _manifest(tmp_path)
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    (baseline / "w3_02_runtime_evidence.json").unlink()
    (treatment / "w3_02_runtime_evidence.json").unlink()

    with pytest.raises(EvidenceValidationError, match="GPU-event"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_rejects_tampered_frozen_source_artifact(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])

    (tmp_path / "frozen" / "source.jsonl").write_text("tampered\n")

    with pytest.raises(EvidenceValidationError, match="source_artifact_hash hash does not match"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_rejects_tampered_final_report(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    report_path = treatment / "comparison" / "iter_600" / "report.json"
    report_path.write_text(json.dumps({"eval_valid": False, "n_test": 0}))

    with pytest.raises(EvidenceValidationError, match="final_report was modified"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_report_requires_a_valid_final_evaluator_report(tmp_path):
    baseline = tmp_path / "baseline"
    treatment = tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    report_path = treatment / "comparison" / "iter_600" / "report.json"
    report_path.write_text(json.dumps({"eval_valid": False, "n_test": 0}))
    runtime_path = treatment / "w3_02_runtime_evidence.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["runtime"]["artifact_hashes"]["final_report"] = _sha256(report_path)
    runtime_path.write_text(json.dumps(runtime))

    with pytest.raises(EvidenceValidationError, match="not evaluator-valid"):
        build_w3_02_evidence_report(manifest, baseline, treatment)


def test_evidence_rejects_legacy_short_holdout_hash_even_with_self_attested_trace(tmp_path):
    baseline, treatment = tmp_path / "baseline", tmp_path / "treatment"
    manifest = _manifest(tmp_path)
    points = [(100, 20.0), (200, 21.0), (300, 22.0)]
    _write_run(baseline, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    _write_run(treatment, manifest, psnr=points, wall_ms=[(i, 10.0) for i, _ in points])
    report_path = baseline / "comparison" / "iter_600" / "report.json"
    report = json.loads(report_path.read_text())
    report["eval_metadata"]["holdout_hash"] = manifest.holdout_hash[:16]
    report_path.write_text(json.dumps(report))
    runtime_path = baseline / "w3_02_runtime_evidence.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["runtime"]["artifact_hashes"]["final_report"] = _sha256(report_path)
    runtime["runtime"]["mutation_trace_applied"] = True
    runtime["runtime"]["trustworthy_attestation"] = True
    runtime_path.write_text(json.dumps(runtime))

    with pytest.raises(EvidenceValidationError, match="full SHA-256"):
        build_w3_02_evidence_report(manifest, baseline, treatment)
