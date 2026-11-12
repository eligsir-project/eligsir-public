"""Contract tests for the pre-registered #1365 population comparison launcher."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_MANIFEST = REPO_ROOT / "experiments" / "researchflow" / "issue1365_population_gpu_qualification.yaml"
REPRODUCTION_MANIFEST = REPO_ROOT / "experiments" / "researchflow" / "issue1365_population_reproduction.yaml"
PROFILE_PATH = REPO_ROOT / "experiments" / "researchflow" / "profiles.issue1365_population_control.toml"


def _launcher_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "launch_issue1365_population_qualification.py"
    )
    spec = importlib.util.spec_from_file_location("issue1365_population_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "assemble_issue1365_population_researchflow_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("issue1365_population_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_script_imports_package_without_pythonpath():
    """The container invokes the launcher by script path, not ``-m``."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/launch_issue1365_population_qualification.py", "--help"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_population_profile_uses_direct_rendered_compose_environment_values():
    """Every profile Compose environment argument is a concrete assignment."""
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"][
        "eligsir-population-control"
    ]
    argv = tuple(profile["argv"])
    assignments = [
        argv[index + 1]
        for index, value in enumerate(argv[:-1])
        if value == "-e"
    ]
    assert assignments
    assert all("=" in token and "$" not in token for token in assignments)
    assert all("{" in token for token in assignments)

    spec = importlib.util.spec_from_file_location(
        "issue1365_compose_wrapper",
        REPO_ROOT / "scripts" / "run_issue1365_population_compose.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.inject_runtime_environment(
        ["docker", "compose", "run", "--rm", "train"],
        {"RESEARCHFLOW_RUN_ID": "run-1", "RESEARCHFLOW_PLAN_SHA256": "plan-1"},
    )
    assert rendered[3:7] == [
        "-e", "RESEARCHFLOW_RUN_ID=run-1",
        "-e", "RESEARCHFLOW_PLAN_SHA256=plan-1",
    ]
    assert all("$" not in token for token in rendered)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _population_metric_run(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create only the immutable completed artifacts a producer may consume."""
    model = tmp_path / "model"
    evaluation_rows = []
    for index in range(6):
        evaluation_rows.append(
            {
                "iteration": (index + 1) * 100,
                "mean_test_psnr_db": 21.0 + index / 10.0,
                "accumulated_iteration_wall_s": 1.5 + index,
                "supervision_views": (index + 1) * 4,
                "scored_holdout_hash": "scored-hash-v1",
                "realized_holdout_hash": "realized-hash-v1",
                "metric_provenance": {"renderer": "gsplat-v1"},
                "throughput_evidence_valid": True,
                "mapping_execution_paused": True,
                "evaluation_phase": "mapping",
                "evaluation_observer_execution": "synchronous_mapping_thread_pause",
                "evaluation_cadence_clock": "cumulative_optimizer_updates",
            }
        )
    _write_jsonl(model / "evaluation_ticks.jsonl", evaluation_rows)
    _write_jsonl(
        model / "psnr_history.jsonl",
        [
            {
                "iteration": row["iteration"],
                "mean_test_psnr": row["mean_test_psnr_db"],
                "mapping_active_s": row["accumulated_iteration_wall_s"],
                "supervision_views": row["supervision_views"],
                "gaussian_count": 1000 + index,
                "throughput_evidence_valid": True,
            }
            for index, row in enumerate(evaluation_rows)
        ],
    )
    _write_json(
        model / "causal_replay_summary.json",
        {"elapsed_s": 12.5, "mapping_active_s": 8.5},
    )
    _write_json(
        model / "experiment_origin.json",
        {
            "evidence_class": "MANAGED",
            "researchflow_run_id": "population-run-1",
            "researchflow_plan_sha256": "p" * 64,
        },
    )
    _write_jsonl(
        model / "decision_trace.jsonl",
        [
            {
                "iteration": 100,
                "transfers": {
                    "h2d_ops": 2,
                    "h2d_bytes": 100,
                    "d2h_ops": 0,
                    "d2h_bytes": 0,
                    "explicit_sync_wait_ms": 0.25,
                },
                "telemetry": {
                    "gpu_transaction": {
                        "transaction_id": 7,
                        "transaction_type": "optimization",
                        "duration_ms": 4.25,
                    }
                },
                "mutation": {
                    "events": [
                        {
                            "kind": "population_policy",
                            "inserted": 1,
                            "pruned": 0,
                            "relocated": 0,
                            "corrected": 0,
                        }
                    ]
                },
                "gaussian_count": 1001,
            },
            {
                "iteration": 200,
                "transfers": {
                    "h2d_ops": 3,
                    "h2d_bytes": 120,
                    "d2h_ops": 1,
                    "d2h_bytes": 8,
                    "explicit_sync_wait_ms": 0.75,
                },
                "telemetry": {
                    "gpu_transaction": {
                        "transaction_id": 8,
                        "transaction_type": "optimization",
                        "duration_ms": 4.75,
                    }
                },
                "mutation": {
                    "events": [
                        {
                            "kind": "population_policy",
                            "inserted": 0,
                            "pruned": 1,
                            "relocated": 2,
                            "corrected": 3,
                        }
                    ]
                },
                "gaussian_count": 1002,
            },
        ],
    )
    environment = {
        "RESEARCHFLOW_SOURCE_MANIFEST_SHA256": "a" * 64,
        "RESEARCHFLOW_EVALUATION_MANIFEST_SHA256": "b" * 64,
        "RESEARCHFLOW_CONTAINER_DIGEST": "sha256:" + "c" * 64,
        "RESEARCHFLOW_HOST_IDENTITY_SHA256": "d" * 64,
        "RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256": "e" * 64,
        "RESEARCHFLOW_PLAN_SHA256": "p" * 64,
        "RESEARCHFLOW_RUN_ID": "population-run-1",
    }
    return model, environment


def test_population_qualification_arms_map_to_the_only_allowed_policy_flags():
    launcher = _launcher_module()

    assert launcher.training_args_for_arm("legacy_compatibility", ["-s", "/data/fr3"]) == [
        "-s",
        "/data/fr3",
        "--no-streaming_population_live_enabled",
        "--no-streaming_population_shadow_enabled",
    ]
    assert launcher.training_args_for_arm("live_probabilistic", ["-s", "/data/fr3"]) == [
        "-s",
        "/data/fr3",
        "--streaming_population_live_enabled",
        "--streaming_population_shadow_enabled",
        "--no-streaming_lifecycle_enabled",
        "--no-streaming_post_mapping_mature_mobility",
        "--no-streaming_depth_lock_centers",
        "--no-streaming_insert_from_depth",
    ]


def test_population_qualification_launcher_rejects_a_conflicting_policy_override():
    launcher = _launcher_module()

    with pytest.raises(ValueError, match="policy flags are owned by the launcher"):
        launcher.training_args_for_arm(
            "legacy_compatibility",
            ["--streaming_population_live_enabled"],
        )


def test_live_launcher_owns_the_m7_conflicting_legacy_controls():
    launcher = _launcher_module()

    with pytest.raises(ValueError, match="policy flags are owned by the launcher"):
        launcher.training_args_for_arm(
            "live_probabilistic", ["--streaming_lifecycle_enabled"]
        )


@pytest.mark.parametrize(
    "legacy_control",
    [
        "--streaming_lifecycle_enabled",
        "--streaming_post_mapping_mature_mobility",
        "--streaming_depth_lock_centers",
        "--streaming_depth_lock_bootstrap",
        "--streaming_depth_lock_insertions",
        "--streaming_depth_lock_prune_conflicts",
        "--streaming_insert_from_depth",
        "--densification_strategy=gsplat_mcmc",
        "--mcmc_proposal_mode=legacy_clamped",
        "--streaming_mcmc_local_only",
    ],
)
def test_live_launcher_rejects_every_m9_legacy_population_control(legacy_control):
    launcher = _launcher_module()

    with pytest.raises(ValueError, match="legacy population controls are incompatible"):
        launcher.training_args_for_arm("live_probabilistic", [legacy_control])


def test_legacy_reproduction_launcher_preserves_m9_controls_verbatim():
    launcher = _launcher_module()
    legacy_controls = [
        "--streaming_lifecycle_enabled",
        "--streaming_post_mapping_mature_mobility",
        "--streaming_depth_lock_centers",
        "--streaming_depth_lock_bootstrap",
        "--streaming_depth_lock_insertions",
        "--streaming_depth_lock_prune_conflicts",
        "--streaming_insert_from_depth",
        "--densification_strategy=gsplat_mcmc",
        "--mcmc_proposal_mode=legacy_clamped",
        "--streaming_mcmc_local_only",
    ]

    assert launcher.training_args_for_arm(
        "legacy_compatibility", legacy_controls
    ) == [
        *legacy_controls,
        "--no-streaming_population_live_enabled",
        "--no-streaming_population_shadow_enabled",
    ]


def test_population_qualification_launcher_rejects_unresolved_provenance_placeholders():
    launcher = _launcher_module()

    with pytest.raises(ValueError, match="unresolved immutable provenance"):
        launcher.require_immutable_provenance(
            {
                "RESEARCHFLOW_RUN_ID": "population-run-1",
                "RESEARCHFLOW_PLAN_SHA256": "p" * 64,
                "RESEARCHFLOW_SOURCE_MANIFEST_SHA256": "UNRESOLVED_REPLACE_AT_REGISTRATION",
                "RESEARCHFLOW_EVALUATION_MANIFEST_SHA256": "f" * 64,
                "RESEARCHFLOW_CONTAINER_DIGEST": "sha256:" + "a" * 64,
                "RESEARCHFLOW_HOST_IDENTITY_SHA256": "e" * 64,
                "RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256": "d" * 64,
                "RESEARCHFLOW_METRICS_CONTRACT_SHA256": "c" * 64,
                "RESEARCHFLOW_METRICS_PATH": "/tmp/metrics.json",
                "ELIGSIR_GIT_COMMIT": "1" * 40,
            }
        )


def test_population_metric_producer_emits_all_contract_metrics_with_plan_bound_provenance(
    tmp_path,
):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    contract = {
        "schema": "eligsir.researchflow.population-metrics-contract/v1",
        "producer_status": "READY",
        "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
        "required_metrics": list(metrics_module.REQUIRED_METRICS),
        "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
    }

    metrics = metrics_module.assemble_metrics(
        model,
        contract=contract,
        environment=environment,
    )

    assert set(metrics["metrics"]) == set(metrics_module.REQUIRED_METRICS)
    assert metrics["metrics"]["quality.scored_holdout_psnr"] == pytest.approx(21.5)
    assert metrics["metrics"]["runtime.transfer.h2d_bytes"] == 120
    assert metrics["metrics"]["runtime.transfer.d2h_bytes"] == 8
    assert metrics["metrics"]["runtime.sync.wait_ms"] == pytest.approx(0.75)
    assert metrics["metrics"]["runtime.gpu_transaction_latency_ms"]["count"] == 2
    assert metrics["metrics"]["population.mutation_events"]["inserted"] == 1
    assert metrics["metrics"]["population.mutation_events"]["pruned"] == 1
    assert metrics["metrics"]["population.mutation_events"]["relocated"] == 2
    assert metrics["provenance"] == {
        "source_manifest_sha256": "a" * 64,
        "evaluation_manifest_sha256": "b" * 64,
        "container_digest": "sha256:" + "c" * 64,
        "host_identity_sha256": "d" * 64,
        "activation_manifest_sha256": "e" * 64,
        "researchflow_plan_sha256": "p" * 64,
    }


def test_population_metric_producer_exposes_plan_evidence_paths_at_top_level(tmp_path):
    """ResearchFlow plan paths resolve without a producer-specific wrapper."""
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    contract = {
        "schema": "eligsir.researchflow.population-metrics-contract/v1",
        "producer_status": "READY",
        "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
        "required_metrics": list(metrics_module.REQUIRED_METRICS),
        "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
    }

    metrics = metrics_module.assemble_metrics(
        model,
        contract=contract,
        environment=environment,
    )

    for path in metrics_module.REQUIRED_METRICS:
        assert path in metrics
        assert metrics[path] == metrics["metrics"][path]


def test_population_metric_producer_rejects_missing_explicit_sync_wait_evidence(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "decision_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["transfers"].pop("explicit_sync_wait_ms")
    _write_jsonl(model / "decision_trace.jsonl", rows)

    with pytest.raises(ValueError, match="explicit synchronization wait"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_metric_producer_rejects_changed_scored_holdout_identity(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["scored_holdout_hash"] = "different-scored-hash"
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)

    with pytest.raises(ValueError, match="scored holdout identity changed"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_metric_producer_rejects_changed_renderer_identity(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["metric_provenance"] = {"renderer": "different-renderer"}
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)

    with pytest.raises(ValueError, match="renderer/evaluation identity changed"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_metric_producer_rejects_invalid_evaluation_phase_semantics(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["evaluation_observer_execution"] = "post_source_completion"
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)

    with pytest.raises(ValueError, match="evaluation phase semantics"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_metric_producer_rejects_evaluation_manifest_mismatch(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["view_source_indices"] = [3, 8]
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {
                "ids": [3, 8],
                "hash": "wrong-scored-hash",
            },
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phase": "mapping",
                "observer": "synchronous_mapping_thread_pause",
                "mapping_execution_paused": True,
                "cadence_clock": "cumulative_optimizer_updates",
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="evaluation manifest scored holdout hash mismatch"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
            evaluation_manifest=evaluation_manifest,
        )


def test_population_metric_producer_emits_verified_evaluation_attestation(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["view_source_indices"] = [3, 8]
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phase": "mapping",
                "observer": "synchronous_mapping_thread_pause",
                "mapping_execution_paused": True,
                "cadence_clock": "cumulative_optimizer_updates",
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()
    metrics = metrics_module.assemble_metrics(
        model,
        contract={
            "schema": "eligsir.researchflow.population-metrics-contract/v1",
            "producer_status": "READY",
            "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
            "required_metrics": list(metrics_module.REQUIRED_METRICS),
            "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
        },
        environment=environment,
        evaluation_manifest=evaluation_manifest,
    )
    assert metrics["evaluation_attestation"]["scored_holdout_ids"] == [3, 8]
    assert metrics["evaluation_attestation"]["scored_holdout_hash"] == "scored-hash-v1"


def test_population_metric_producer_accepts_mapping_endpoint_manifest_semantics(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row.update(
            {
                "view_source_indices": [3, 8],
                "evaluation_phase": "mapping_endpoint",
                "phase": "mapping_endpoint",
                "evaluation_observer_execution": "post_source_completion",
                "mapping_execution_paused": False,
            }
        )
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phase": "mapping_endpoint",
                "observer": "post_source_completion",
                "mapping_execution_paused": False,
                "cadence_clock": "cumulative_optimizer_updates",
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()
    metrics = metrics_module.assemble_metrics(
        model,
        contract={
            "schema": "eligsir.researchflow.population-metrics-contract/v1",
            "producer_status": "READY",
            "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
            "required_metrics": list(metrics_module.REQUIRED_METRICS),
            "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
        },
        environment=environment,
        evaluation_manifest=evaluation_manifest,
    )
    assert metrics["evaluation_attestation"]["evaluation_semantics"]["phase"] == (
        "mapping_endpoint"
    )


def test_population_metric_producer_filters_unbound_post_mapping_rows(tmp_path):
    """Only rows matching the manifest's declared observation are attested/scored."""
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row.update(
            {
                "view_source_indices": [3, 8],
                "evaluation_phase": "mapping_endpoint",
                "phase": "mapping_endpoint",
                "evaluation_observer_execution": "post_source_completion",
                "mapping_execution_paused": False,
            }
        )
    post_mapping = dict(rows[-1])
    post_mapping.update(
        {
            "iteration": 999,
            "evaluation_phase": "post_mapping_report",
            "phase": "post_mapping_report",
            "evaluation_observer_execution": "post_source_completion",
            "mapping_execution_paused": False,
        }
    )
    rows.append(post_mapping)
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phase": "mapping_endpoint",
                "observer": "post_source_completion",
                "mapping_execution_paused": False,
                "cadence_clock": "cumulative_optimizer_updates",
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()
    contract = {
        "schema": "eligsir.researchflow.population-metrics-contract/v1",
        "producer_status": "READY",
        "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
        "required_metrics": list(metrics_module.REQUIRED_METRICS),
        "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
    }
    metrics = metrics_module.assemble_metrics(
        model,
        contract=contract,
        environment=environment,
        evaluation_manifest=evaluation_manifest,
    )
    assert len(metrics["metrics"]["quality.psnr_trajectory"]) == 6
    assert all(
        point["evaluation_phase"] == "mapping_endpoint"
        for point in metrics["metrics"]["quality.psnr_trajectory"]
    )


def test_population_metric_producer_accepts_declared_mapping_and_refinement_phases(tmp_path):
    """A manifest may explicitly bind the endpoint and refinement tail together."""
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    phases = ["mapping_endpoint", *(["refinement"] * 5)]
    # Six rows are the minimum, while the real shape is one endpoint plus an
    # eight-tick refinement tail.
    phases.extend(["refinement"] * 3)
    expanded = []
    for index, phase in enumerate(phases):
        row = dict(rows[index % len(rows)])
        row.update(
            {
                "iteration": (index + 1) * 100,
                "view_source_indices": [3, 8],
                "evaluation_phase": phase,
                "phase": phase,
                "evaluation_observer_execution": "post_source_completion",
                "mapping_execution_paused": False,
            }
        )
        expanded.append(row)
    _write_jsonl(model / "evaluation_ticks.jsonl", expanded)
    _write_jsonl(
        model / "psnr_history.jsonl",
        [
            {
                "iteration": row["iteration"],
                "mean_test_psnr": 21.0 + index / 10.0,
                "mapping_active_s": 1.5 + index,
                "supervision_views": (index + 1) * 4,
                "gaussian_count": 1000 + index,
                "throughput_evidence_valid": True,
            }
            for index, row in enumerate(expanded)
        ],
    )
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phases": [
                    {
                        "phase": "mapping_endpoint",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                    {
                        "phase": "refinement",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                ]
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()
    contract = {
        "schema": "eligsir.researchflow.population-metrics-contract/v1",
        "producer_status": "READY",
        "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
        "required_metrics": list(metrics_module.REQUIRED_METRICS),
        "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
    }
    metrics = metrics_module.assemble_metrics(
        model,
        contract=contract,
        environment=environment,
        evaluation_manifest=evaluation_manifest,
    )
    trajectory = metrics["metrics"]["quality.psnr_trajectory"]
    assert len(trajectory) == 9
    assert [point["evaluation_phase"] for point in trajectory] == phases
    assert [phase["phase"] for phase in metrics["evaluation_attestation"]["evaluation_semantics"]["phases"]] == [
        "mapping_endpoint",
        "refinement",
    ]


def _multiphase_metric_fixture(tmp_path):
    """Return a nine-tick endpoint/refinement fixture and its manifest."""
    model, environment = _population_metric_run(tmp_path)
    source_rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    phases = ["mapping_endpoint", *(["refinement"] * 8)]
    rows = []
    for index, phase in enumerate(phases):
        row = dict(source_rows[index % len(source_rows)])
        row.update(
            {
                "iteration": (index + 1) * 100,
                "view_source_indices": [3, 8],
                "evaluation_phase": phase,
                "evaluation_observer_execution": "post_source_completion",
                "mapping_execution_paused": False,
            }
        )
        rows.append(row)
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    _write_jsonl(
        model / "psnr_history.jsonl",
        [
            {
                "iteration": row["iteration"],
                "mean_test_psnr": 21.0 + index / 10.0,
                "mapping_active_s": 1.5 + index,
                "supervision_views": (index + 1) * 4,
                "gaussian_count": 1000 + index,
                "throughput_evidence_valid": True,
            }
            for index, row in enumerate(rows)
        ],
    )
    manifest = tmp_path / "evaluation.json"
    _write_json(
        manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phases": [
                    {
                        "phase": "mapping_endpoint",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                    {
                        "phase": "refinement",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                ]
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    contract = {
        "schema": "eligsir.researchflow.population-metrics-contract/v1",
        "producer_status": "READY",
        "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
        "required_metrics": list(_metrics_module().REQUIRED_METRICS),
        "required_provenance": list(_metrics_module().REQUIRED_PROVENANCE),
    }
    return model, environment, manifest, contract


def test_population_metric_producer_rejects_missing_declared_phase(tmp_path):
    """Every phase in a multi-phase manifest must contribute an admitted tick."""
    metrics_module = _metrics_module()
    model, environment, manifest, contract = _multiphase_metric_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["evaluation_phase"] = "refinement"
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    with pytest.raises(ValueError, match="declared evaluation phase has no matching rows"):
        metrics_module.assemble_metrics(
            model,
            contract=contract,
            environment=environment,
            evaluation_manifest=manifest,
        )


def test_population_metric_producer_rejects_nonmonotonic_declared_tick_order(tmp_path):
    """Admitted ticks must retain source order within the declared phase sequence."""
    metrics_module = _metrics_module()
    model, environment, manifest, contract = _multiphase_metric_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[-1]["iteration"] = 50
    _write_jsonl(model / "evaluation_ticks.jsonl", rows)
    with pytest.raises(ValueError, match="evaluation manifest tick order mismatch"):
        metrics_module.assemble_metrics(
            model,
            contract=contract,
            environment=environment,
            evaluation_manifest=manifest,
        )


def test_population_metric_producer_rejects_identity_mismatch_in_declared_refinement_phase(
    tmp_path,
):
    """A declared phase does not waive immutable identity checks."""
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    rows = [
        json.loads(line)
        for line in (model / "evaluation_ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expanded = []
    for index in range(9):
        row = dict(rows[index % len(rows)])
        row.update(
            {
                "iteration": (index + 1) * 100,
                "view_source_indices": [3, 8],
                "evaluation_phase": "mapping_endpoint" if index == 0 else "refinement",
                "evaluation_observer_execution": "post_source_completion",
                "mapping_execution_paused": False,
            }
        )
        expanded.append(row)
    expanded[-1]["scored_holdout_hash"] = "different-scored-hash"
    _write_jsonl(model / "evaluation_ticks.jsonl", expanded)
    evaluation_manifest = tmp_path / "evaluation.json"
    _write_json(
        evaluation_manifest,
        {
            "schema": "eligsir.evaluation-manifest/v1",
            "scored_holdout": {"ids": [3, 8], "hash": "scored-hash-v1"},
            "renderer_semantics": {"renderer": "gsplat-v1"},
            "evaluation_semantics": {
                "phases": [
                    {
                        "phase": "mapping_endpoint",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                    {
                        "phase": "refinement",
                        "observer": "post_source_completion",
                        "mapping_execution_paused": False,
                        "cadence_clock": "cumulative_optimizer_updates",
                    },
                ]
            },
        },
    )
    environment["RESEARCHFLOW_EVALUATION_MANIFEST_SHA256"] = hashlib.sha256(
        evaluation_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="evaluation manifest scored holdout hash mismatch"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
            evaluation_manifest=evaluation_manifest,
        )


def test_population_metric_producer_fails_closed_on_actual_decision_trace_row_without_sync(
    tmp_path,
):
    from eligsir.runtime.causal_replay import decision_trace_row

    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    trace_row = decision_trace_row(
        SimpleNamespace(
            iteration=1,
            transfers=SimpleNamespace(
                h2d_ops=1,
                h2d_bytes=64,
                d2h_ops=0,
                d2h_bytes=0,
                pool_waits=0,
                pool_drops=0,
            ),
            telemetry=SimpleNamespace(
                gpu_transaction=SimpleNamespace(
                    transaction_id=1,
                    transaction_type=SimpleNamespace(value="optimization"),
                    duration_ms=2.0,
                    map_content_version=1,
                    topology_version=0,
                    parameter_version=1,
                    batch_size=4,
                    pixels=1024,
                    rasterizer_calls=1,
                    optimizer_updates=1,
                    candidates=0,
                    mutation_type="none",
                    inserted=0,
                    pruned=0,
                    relocated=0,
                    corrected=0,
                ),
                map_state=SimpleNamespace(
                    gaussian_count=10,
                    map_content_version=1,
                    topology_version=0,
                    parameter_version=1,
                ),
            ),
            mutation_events=(),
        )
    )
    assert trace_row["transfers"]["h2d_bytes"] == 64
    assert trace_row["telemetry"]["gpu_transaction"]["duration_ms"] == 2.0
    _write_jsonl(model / "decision_trace.jsonl", [trace_row])

    with pytest.raises(ValueError, match="explicit synchronization wait"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_metric_producer_accepts_owner_measured_sync_wait(tmp_path):
    from eligsir.runtime.causal_replay import decision_trace_row

    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    trace_row = decision_trace_row(
        SimpleNamespace(
            iteration=1,
            transfers=SimpleNamespace(
                h2d_ops=1,
                h2d_bytes=64,
                d2h_ops=0,
                d2h_bytes=0,
                pool_waits=0,
                pool_drops=0,
                explicit_sync_waits=1,
                explicit_sync_wait_ms=0.75,
            ),
            telemetry=SimpleNamespace(
                gpu_transaction=SimpleNamespace(
                    transaction_id=1,
                    transaction_type=SimpleNamespace(value="optimization"),
                    duration_ms=2.0,
                    map_content_version=1,
                    topology_version=0,
                    parameter_version=1,
                    batch_size=4,
                    pixels=1024,
                    rasterizer_calls=1,
                    optimizer_updates=1,
                    candidates=0,
                    mutation_type="none",
                    inserted=0,
                    pruned=0,
                    relocated=0,
                    corrected=0,
                ),
                map_state=SimpleNamespace(
                    gaussian_count=10,
                    map_content_version=1,
                    topology_version=0,
                    parameter_version=1,
                ),
            ),
            mutation_events=(),
        )
    )
    assert trace_row["transfers"]["explicit_sync_wait_ms"] == pytest.approx(0.75)
    _write_jsonl(model / "decision_trace.jsonl", [trace_row])

    metrics = metrics_module.assemble_metrics(
        model,
        contract={
            "schema": "eligsir.researchflow.population-metrics-contract/v1",
            "producer_status": "READY",
            "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
            "required_metrics": list(metrics_module.REQUIRED_METRICS),
            "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
        },
        environment=environment,
    )

    assert metrics["metrics"]["runtime.sync.wait_ms"] == pytest.approx(0.75)


def test_population_metric_producer_rejects_terminal_trace_without_sync_wait(tmp_path):
    metrics_module = _metrics_module()
    model, environment = _population_metric_run(tmp_path)
    _write_jsonl(
        model / "decision_trace.jsonl",
        [
            {"transfers": {"h2d_bytes": 64, "d2h_bytes": 0, "explicit_sync_wait_ms": 0.75}},
            {"transfers": {"h2d_bytes": 128, "d2h_bytes": 0}},
        ],
    )

    with pytest.raises(ValueError, match="explicit synchronization wait"):
        metrics_module.assemble_metrics(
            model,
            contract={
                "schema": "eligsir.researchflow.population-metrics-contract/v1",
                "producer_status": "READY",
                "metrics_path": "$RESEARCHFLOW_METRICS_PATH",
                "required_metrics": list(metrics_module.REQUIRED_METRICS),
                "required_provenance": list(metrics_module.REQUIRED_PROVENANCE),
            },
            environment=environment,
        )


def test_population_launcher_refuses_blocked_metric_contract_before_training(tmp_path):
    launcher = _launcher_module()
    source = tmp_path / "source.json"
    evaluation = tmp_path / "evaluation.json"
    host = tmp_path / "host.json"
    activation = tmp_path / "activation.json"
    contract = tmp_path / "contract.json"
    container = tmp_path / "container.json"
    for path in (source, evaluation, host, activation):
        path.write_text("{}", encoding="utf-8")
    container.write_text(json.dumps({"container_digest": "sha256:" + "c" * 64}), encoding="utf-8")
    contract.write_text(
        json.dumps({"producer_status": "BLOCKED_UNTIL_RUNTIME_TELEMETRY"}),
        encoding="utf-8",
    )
    environment = {
        "RESEARCHFLOW_RUN_ID": "population-run-1",
        "RESEARCHFLOW_PLAN_SHA256": "p" * 64,
        "RESEARCHFLOW_SOURCE_MANIFEST_SHA256": launcher._sha256(str(source)),
        "RESEARCHFLOW_EVALUATION_MANIFEST_SHA256": launcher._sha256(str(evaluation)),
        "RESEARCHFLOW_CONTAINER_DIGEST": "sha256:" + "c" * 64,
        "RESEARCHFLOW_HOST_IDENTITY_SHA256": launcher._sha256(str(host)),
        "RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256": launcher._sha256(str(activation)),
        "RESEARCHFLOW_METRICS_CONTRACT_SHA256": launcher._sha256(str(contract)),
        "RESEARCHFLOW_METRICS_PATH": "/output/researchflow_metrics.json",
        "ELIGSIR_GIT_COMMIT": "1" * 40,
    }
    args = SimpleNamespace(
        source_manifest=str(source),
        evaluation_manifest=str(evaluation),
        container_manifest=str(container),
        host_manifest=str(host),
        activation_manifest=str(activation),
        metrics_contract=str(contract),
    )

    with pytest.raises(ValueError, match="metrics producer is not ready"):
        launcher.require_pre_dispatch_contract(args, environment)


def test_population_launcher_assembles_metrics_only_after_train_succeeds(monkeypatch):
    launcher = _launcher_module()
    calls = []

    def fake_call(command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(launcher.subprocess, "call", fake_call)
    result = launcher.assemble_metrics_after_success(
        model_path="/output/model",
        args=type("Args", (), {"metrics_contract": "/contract.json"})(),
        environment={"RESEARCHFLOW_METRICS_PATH": "/output/model/researchflow_metrics.json"},
    )

    assert result == 0
    assert calls == [
        (
            [
                launcher.sys.executable,
                "scripts/assemble_issue1365_population_researchflow_metrics.py",
                "--model-path",
                "/output/model",
                "--output",
                "/output/model/researchflow_metrics.json",
                "--contract",
                "/contract.json",
            ],
            {
                "env": {
                    "RESEARCHFLOW_METRICS_PATH": "/output/model/researchflow_metrics.json"
                }
            },
        )
    ]


def test_population_launcher_requires_exactly_one_model_path_for_metric_assembly():
    launcher = _launcher_module()

    assert launcher.model_path_from_training_args(["-m", "/output/model"]) == "/output/model"
    with pytest.raises(ValueError, match="exactly one model path"):
        launcher.model_path_from_training_args(["-m", "/one", "--model_path", "/two"])


def test_population_manifests_preserve_the_same_semantic_metric_contract():
    qualification = yaml.safe_load(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    reproduction = yaml.safe_load(REPRODUCTION_MANIFEST.read_text(encoding="utf-8"))

    assert qualification["type"] == "validation"
    assert qualification["matrix"]["population_arm"] == [
        "legacy_compatibility",
        "live_probabilistic",
    ]
    assert reproduction["matrix"]["population_arm"] == qualification["matrix"]["population_arm"]
    assert reproduction["matrix"]["seed"] == [17, 29, 41]
    assert reproduction["matrix"]["scene"] == ["tum_fr3_long_office_household"]

    required_metrics = {
        "quality.psnr_trajectory",
        "runtime.wall_clock_s",
        "runtime.mapping_active_s",
        "runtime.transfer.h2d_bytes",
        "runtime.transfer.d2h_bytes",
        "runtime.sync.wait_ms",
        "runtime.gpu_transaction_latency_ms",
        "population.mutation_events",
        "population.gaussian_count_trajectory",
    }
    for manifest in (qualification, reproduction):
        names = {item["name"] for item in manifest["evidence"]["metrics"]}
        assert required_metrics <= names
        required_capabilities = {item["capability"] for item in manifest["requires"] if "capability" in item}
        assert "eligsir.activation.rocm.preflight-attested" in required_capabilities
        assert "eligsir.population.policy.semantic-identity.v1" in required_capabilities
        assert "eligsir.container.rocm.digest-qualified" in required_capabilities


def test_population_manifests_declare_the_m9_legacy_reproduction_boundary():
    qualification = yaml.safe_load(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    reproduction = yaml.safe_load(REPRODUCTION_MANIFEST.read_text(encoding="utf-8"))

    for manifest in (qualification, reproduction):
        assert (
            manifest["metadata"]["legacy_population_controls_status"]
            == "reproduction_only_pending_researchflow_horizon"
        )
        assert manifest["metadata"]["live_legacy_control_policy"] == (
            "reject_conflicts"
        )
    assert reproduction["metadata"]["legacy_reproduction_args"] == "verbatim"


def test_population_profile_is_containerized_and_declares_every_required_metric():
    profile = tomllib.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"][
        "eligsir-population-control"
    ]

    assert profile["argv"][:2] == [
        "python",
        "scripts/run_issue1365_population_compose.py",
    ]
    assert profile["argv"][2:6] == ["docker", "compose", "run", "--rm"]
    assert "--scientific_mode" in profile["argv"]
    assert profile["argv"][profile["argv"].index("--streaming_frame_admission") + 1] == "preselected"
    assert "--gpu_frame_prep" in profile["argv"]
    assert "--gpu_frame_resident" in profile["argv"]
    assert profile["argv"][profile["argv"].index("--streaming_optimization_batch_size") + 1] == "4"
    declared = set(profile["evidence"])
    assert "metric:quality.psnr_trajectory" in declared
    assert "metric:runtime.transfer.h2d_bytes" in declared
    assert "metric:population.mutation_events" in declared
    assert "RESEARCHFLOW_METRICS_PATH" in profile["env"]
    assert "--causal_decision_trace" in profile["argv"]
    assert "--activation-manifest" in profile["argv"]


def test_population_metric_contract_is_ready_with_truthful_runtime_telemetry():
    contract = json.loads(
        (
            REPO_ROOT
            / "experiments"
            / "researchflow"
            / "issue1365_population_metrics_contract.json"
        ).read_text(encoding="utf-8")
    )

    assert contract["producer_status"] == "READY"
    assert contract["metrics_path"] == "$RESEARCHFLOW_METRICS_PATH"
    assert contract["producer"] == (
        "scripts/assemble_issue1365_population_researchflow_metrics.py"
    )
