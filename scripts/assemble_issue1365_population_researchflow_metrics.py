#!/usr/bin/env python3
"""Assemble fail-closed #1365 ResearchFlow runtime evidence.

This is deliberately a post-run assembler, not a second evaluator.  It only
joins artifacts produced by the registered causal-mapping transaction:
``evaluation_ticks.jsonl``, ``psnr_history.jsonl``,
``causal_replay_summary.json``, ``decision_trace.jsonl``, and
``experiment_origin.json``.  A missing field is never represented as zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "quality.psnr_trajectory",
    "quality.scored_holdout_psnr",
    "runtime.wall_clock_s",
    "runtime.mapping_active_s",
    "runtime.transfer.h2d_bytes",
    "runtime.transfer.d2h_bytes",
    "runtime.sync.wait_ms",
    "runtime.gpu_transaction_latency_ms",
    "population.mutation_events",
    "population.gaussian_count_trajectory",
)
REQUIRED_PROVENANCE = (
    "source_manifest_sha256",
    "evaluation_manifest_sha256",
    "container_digest",
    "host_identity_sha256",
    "activation_manifest_sha256",
    "researchflow_plan_sha256",
)

_PROVENANCE_ENV = {
    "source_manifest_sha256": "RESEARCHFLOW_SOURCE_MANIFEST_SHA256",
    "evaluation_manifest_sha256": "RESEARCHFLOW_EVALUATION_MANIFEST_SHA256",
    "container_digest": "RESEARCHFLOW_CONTAINER_DIGEST",
    "host_identity_sha256": "RESEARCHFLOW_HOST_IDENTITY_SHA256",
    "activation_manifest_sha256": "RESEARCHFLOW_ACTIVATION_MANIFEST_SHA256",
    "researchflow_plan_sha256": "RESEARCHFLOW_PLAN_SHA256",
}


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
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"required JSONL artifact is empty: {path}")
    return rows


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite numeric value")
    return result


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("no completed GPU optimization transaction timing is available")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "unit": "ms",
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _require_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != "eligsir.researchflow.population-metrics-contract/v1":
        raise ValueError("metrics contract schema does not match producer schema")
    if contract.get("producer_status") != "READY":
        raise ValueError("metrics producer contract is not READY")
    if contract.get("metrics_path") != "$RESEARCHFLOW_METRICS_PATH":
        raise ValueError("metrics contract path is not plan-bound")
    required_metrics = contract.get("required_metrics")
    required_provenance = contract.get("required_provenance")
    if required_metrics != list(REQUIRED_METRICS):
        raise ValueError("metrics contract required_metrics does not match producer schema")
    if required_provenance != list(REQUIRED_PROVENANCE):
        raise ValueError("metrics contract required_provenance does not match producer schema")


def _provenance(environment: Mapping[str, str], origin: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in REQUIRED_PROVENANCE:
        env_name = _PROVENANCE_ENV[name]
        value = str(environment.get(env_name, "")).strip()
        if not value or value.startswith("UNRESOLVED_"):
            raise ValueError(f"missing plan-bound provenance: {env_name}")
        result[name] = value
    if origin.get("evidence_class") != "MANAGED":
        raise ValueError("experiment origin does not prove a managed run")
    if origin.get("researchflow_plan_sha256") != result["researchflow_plan_sha256"]:
        raise ValueError("experiment origin plan SHA does not match plan-bound provenance")
    run_id = str(environment.get("RESEARCHFLOW_RUN_ID", "")).strip()
    if not run_id:
        raise ValueError("missing ResearchFlow run ID")
    if origin.get("researchflow_run_id") != run_id:
        raise ValueError("experiment origin run ID does not match the registered run")
    return result


def _evaluation_manifest_identity(path: Path) -> tuple[dict[str, Any], str]:
    """Read the frozen evaluation identity used by the ResearchFlow plan."""
    manifest = _read_json(path)
    if manifest.get("schema") != "eligsir.evaluation-manifest/v1":
        raise ValueError("evaluation manifest schema does not match producer schema")
    scored = manifest.get("scored_holdout")
    if not isinstance(scored, Mapping):
        raise ValueError("evaluation manifest lacks scored_holdout identity")
    ids = scored.get("ids")
    scored_hash = str(scored.get("hash", "")).strip()
    renderer = manifest.get("renderer_semantics")
    evaluation = manifest.get("evaluation_semantics")
    if (
        not isinstance(ids, list)
        or not ids
        or any(isinstance(item, (bool, dict, list)) for item in ids)
        or not scored_hash
        or not isinstance(renderer, Mapping)
        or not renderer
        or not (
            (isinstance(evaluation, Mapping) and evaluation)
            or (
                isinstance(evaluation, list)
                and evaluation
                and all(isinstance(item, Mapping) and item for item in evaluation)
            )
        )
    ):
        raise ValueError("evaluation manifest has incomplete scored/renderer/evaluation identity")
    identity = {
        "scored_holdout_ids": list(ids),
        "scored_holdout_hash": scored_hash,
        "renderer_semantics": dict(renderer),
        "evaluation_semantics": dict(evaluation),
    }
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return identity, digest


def _attest_evaluation_manifest(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_digest: str,
) -> dict[str, Any]:
    identity, digest = _evaluation_manifest_identity(path)
    if digest != expected_digest:
        raise ValueError("evaluation manifest digest does not match plan-bound provenance")
    expected_phase_semantics = {
        "mapping": {
            "observer": "synchronous_mapping_thread_pause",
            "mapping_execution_paused": True,
        },
        # The persisted mapping-endpoint checkpoint is evaluated after source
        # completion.  It is still the mapping result, but is not a paused
        # in-loop mapping observation.
        "mapping_endpoint": {
            "observer": "post_source_completion",
            "mapping_execution_paused": False,
        },
        "refinement": {
            "observer": "post_source_completion",
            "mapping_execution_paused": False,
        },
    }
    declared_semantics = identity["evaluation_semantics"]
    if isinstance(declared_semantics, Mapping):
        phases_value = declared_semantics.get("phases")
        declared_phases = (
            [dict(item) for item in phases_value]
            if isinstance(phases_value, list)
            else [dict(declared_semantics)]
        )
    elif isinstance(declared_semantics, list):
        declared_phases = [dict(item) for item in declared_semantics]
    else:  # guarded by _evaluation_manifest_identity, retained for type safety
        raise ValueError("evaluation manifest evaluation semantics are unsupported")
    normalized_phases = []
    for phase_semantics in declared_phases:
        declared_phase = str(phase_semantics.get("phase", "")).strip()
        expected_phase = expected_phase_semantics.get(declared_phase)
        expected = (
            {
                "phase": declared_phase,
                **expected_phase,
                "cadence_clock": "cumulative_optimizer_updates",
            }
            if expected_phase is not None
            else None
        )
        if expected is None or phase_semantics != expected:
            raise ValueError("evaluation manifest evaluation semantics are unsupported")
        normalized_phases.append(expected)
    if len(normalized_phases) > 1 and [item["phase"] for item in normalized_phases] != [
        "mapping_endpoint",
        "refinement",
    ]:
        raise ValueError("evaluation manifest evaluation semantics are unsupported")
    declared_by_phase = {item["phase"]: (index, item) for index, item in enumerate(normalized_phases)}
    matching_rows = []
    matching_phase_indices: list[int] = []
    matching_iterations: list[int] = []
    for row in rows:
        actual_semantics = {
            "phase": row.get("evaluation_phase"),
            "observer": row.get("evaluation_observer_execution"),
            "mapping_execution_paused": row.get("mapping_execution_paused"),
            "cadence_clock": row.get("evaluation_cadence_clock"),
        }
        # A run may contain other legitimate evaluation observations (for
        # example a post-mapping report).  The manifest binds one declared
        # observation semantics, so only rows with that exact semantics are
        # part of this attestation.  Rows selected here still undergo every
        # immutable identity check below.
        declared_row = declared_by_phase.get(actual_semantics["phase"])
        if declared_row is None or actual_semantics != declared_row[1]:
            continue
        if matching_phase_indices and declared_row[0] < matching_phase_indices[-1]:
            raise ValueError("evaluation manifest phase order mismatch")
        iteration = _nonnegative_integer(row.get("iteration"), field="evaluation iteration")
        if matching_iterations and iteration < matching_iterations[-1]:
            raise ValueError("evaluation manifest tick order mismatch")
        matching_rows.append(row)
        matching_phase_indices.append(declared_row[0])
        matching_iterations.append(iteration)
        if str(row.get("scored_holdout_hash", "")).strip() != identity["scored_holdout_hash"]:
            raise ValueError("evaluation manifest scored holdout hash mismatch")
        actual_ids = row.get("view_source_indices")
        if actual_ids != identity["scored_holdout_ids"]:
            raise ValueError("evaluation manifest scored holdout IDs mismatch")
        if row.get("metric_provenance") != identity["renderer_semantics"]:
            raise ValueError("evaluation manifest renderer semantics mismatch")
    if not matching_rows:
        raise ValueError("evaluation manifest has no rows matching declared evaluation semantics")
    missing_phases = [
        phase for phase in declared_by_phase if phase not in {row.get("evaluation_phase") for row in matching_rows}
    ]
    if missing_phases:
        raise ValueError("declared evaluation phase has no matching rows")
    if len(matching_rows) < 6:
        raise ValueError("evaluation manifest admits fewer than 6 evaluation ticks")
    return {"manifest_sha256": digest, **identity}


def _rows_matching_evaluation_manifest_semantics(
    rows: Sequence[Mapping[str, Any]],
    evaluation_semantics: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Select only observations covered by the manifest's declared semantics."""
    if isinstance(evaluation_semantics, Mapping):
        phases_value = evaluation_semantics.get("phases")
        declared = (
            [dict(item) for item in phases_value]
            if isinstance(phases_value, list)
            else [dict(evaluation_semantics)]
        )
    elif isinstance(evaluation_semantics, list):
        declared = [dict(item) for item in evaluation_semantics]
    else:
        return []
    declared_by_phase = {item["phase"]: (index, item) for index, item in enumerate(declared)}
    selected = [
        row
        for row in rows
        if row.get("evaluation_phase") in declared_by_phase
        and {
            "phase": row.get("evaluation_phase"),
            "observer": row.get("evaluation_observer_execution"),
            "mapping_execution_paused": row.get("mapping_execution_paused"),
            "cadence_clock": row.get("evaluation_cadence_clock"),
        }
        == declared_by_phase[row.get("evaluation_phase")][1]
    ]
    return selected


def _quality_trajectory(
    evaluation_rows: Sequence[Mapping[str, Any]],
    psnr_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    psnr_by_iteration: dict[int, Mapping[str, Any]] = {}
    for row in psnr_rows:
        iteration = _nonnegative_integer(row.get("iteration"), field="psnr iteration")
        if iteration in psnr_by_iteration:
            raise ValueError(f"duplicate PSNR history iteration: {iteration}")
        psnr_by_iteration[iteration] = row

    trajectory: list[dict[str, Any]] = []
    population: list[dict[str, int]] = []
    immutable_identity: tuple[str, str, str] | None = None
    for row in evaluation_rows:
        if row.get("throughput_evidence_valid") is not True:
            raise ValueError("scored holdout trajectory has invalid throughput evidence")
        iteration = _nonnegative_integer(row.get("iteration"), field="evaluation iteration")
        psnr = psnr_by_iteration.get(iteration)
        if psnr is None:
            raise ValueError(f"evaluation tick lacks matching PSNR history: {iteration}")
        point = {
            "iteration": iteration,
            "psnr_db": _finite_number(row.get("mean_test_psnr_db"), field="holdout PSNR"),
            "mapping_active_s": _finite_number(
                psnr.get("mapping_active_s"), field="PSNR mapping_active_s"
            ),
            "supervision_views": _nonnegative_integer(
                psnr.get("supervision_views"), field="PSNR supervision_views"
            ),
            "scored_holdout_hash": str(row.get("scored_holdout_hash", "")).strip(),
            "realized_holdout_hash": str(row.get("realized_holdout_hash", "")).strip(),
            "metric_provenance": row.get("metric_provenance"),
            "gaussian_count": _nonnegative_integer(
                psnr.get("gaussian_count"), field="PSNR gaussian_count"
            ),
            "evaluation_phase": str(row.get("evaluation_phase", "")).strip(),
        }
        if not point["scored_holdout_hash"] or not point["realized_holdout_hash"]:
            raise ValueError("scored holdout trajectory lacks immutable view identity")
        if not isinstance(point["metric_provenance"], dict) or not point["metric_provenance"]:
            raise ValueError("scored holdout trajectory lacks renderer/evaluation identity")
        expected_phase_semantics = {
            "mapping": ("synchronous_mapping_thread_pause", True),
            "mapping_endpoint": ("post_source_completion", False),
            "refinement": ("post_source_completion", False),
        }
        expected = expected_phase_semantics.get(point["evaluation_phase"])
        actual = (
            str(row.get("evaluation_observer_execution", "")).strip(),
            row.get("mapping_execution_paused"),
        )
        if expected is None or actual != expected:
            raise ValueError("scored holdout evaluation phase semantics are invalid")
        identity = (
            point["scored_holdout_hash"],
            point["realized_holdout_hash"],
            json.dumps(point["metric_provenance"], sort_keys=True, separators=(",", ":")),
        )
        if immutable_identity is None:
            immutable_identity = identity
        elif identity[:2] != immutable_identity[:2]:
            raise ValueError("scored holdout identity changed within one run")
        elif identity[2] != immutable_identity[2]:
            raise ValueError("renderer/evaluation identity changed within one run")
        trajectory.append(point)
        population.append({"iteration": iteration, "gaussian_count": point["gaussian_count"]})
    if len(trajectory) < 6:
        raise ValueError("population qualification requires at least 6 scored holdout ticks")
    return trajectory, population


def _runtime_and_population_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int, float, dict[str, Any], list[dict[str, int]]]:
    terminal_transfers: Mapping[str, Any] | None = None
    terminal_sync_wait_ms: float | None = None
    completed_gpu: dict[int, float] = {}
    mutation_events: list[dict[str, Any]] = []
    mutation_counts = {"inserted": 0, "pruned": 0, "relocated": 0, "corrected": 0}
    mutation_population: list[dict[str, int]] = []

    for index, row in enumerate(rows):
        transfers = row.get("transfers")
        if isinstance(transfers, dict):
            terminal_transfers = transfers
            terminal_sync_wait_ms = None
            # This is an owner-attributed, cumulative measurement made at the
            # actual host-side event wait.  Do not manufacture a zero for old
            # traces: a missing field means this run cannot prove the metric.
            if "explicit_sync_wait_ms" in transfers:
                terminal_sync_wait_ms = _finite_number(
                    transfers["explicit_sync_wait_ms"],
                    field="explicit synchronization wait",
                )
        telemetry = row.get("telemetry")
        if isinstance(telemetry, dict):
            gpu = telemetry.get("gpu_transaction")
            if isinstance(gpu, dict) and gpu.get("transaction_type") == "optimization":
                transaction_id = _nonnegative_integer(
                    gpu.get("transaction_id"), field="GPU transaction ID"
                )
                duration_ms = _finite_number(
                    gpu.get("duration_ms"), field="GPU transaction duration"
                )
                prior = completed_gpu.get(transaction_id)
                if prior is not None and prior != duration_ms:
                    raise ValueError("completed GPU transaction was serialized inconsistently")
                completed_gpu[transaction_id] = duration_ms
            map_state = telemetry.get("map_state")
            if isinstance(map_state, dict) and "gaussian_count" in map_state:
                mutation_population.append(
                    {
                        "iteration": _nonnegative_integer(row.get("iteration"), field="trace iteration"),
                        "gaussian_count": _nonnegative_integer(
                            map_state["gaussian_count"], field="trace gaussian_count"
                        ),
                    }
                )
        mutation = row.get("mutation")
        if not isinstance(mutation, dict):
            continue
        events = mutation.get("events")
        if not isinstance(events, list):
            raise ValueError("decision trace mutation events are missing")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("decision trace mutation event is invalid")
            counts = {
                name: _nonnegative_integer(event.get(name, 0), field=f"mutation {name}")
                for name in mutation_counts
            }
            mutation_events.append(
                {
                    "iteration": _nonnegative_integer(row.get("iteration"), field="trace iteration"),
                    "kind": str(event.get("kind", "unknown")),
                    **counts,
                }
            )
            for name, value in counts.items():
                mutation_counts[name] += value

    if terminal_transfers is None:
        raise ValueError("decision trace lacks cumulative transfer accounting")
    h2d_bytes = _nonnegative_integer(terminal_transfers.get("h2d_bytes"), field="H2D bytes")
    d2h_bytes = _nonnegative_integer(terminal_transfers.get("d2h_bytes"), field="D2H bytes")
    if terminal_sync_wait_ms is None:
        raise ValueError("decision trace lacks explicit synchronization wait evidence")
    population_metric = {"events": mutation_events, **mutation_counts}
    return (
        h2d_bytes,
        d2h_bytes,
        terminal_sync_wait_ms,
        {"latency": _summary(list(completed_gpu.values())), "mutation": population_metric},
        mutation_population,
    )


def assemble_metrics(
    model_path: str | Path,
    *,
    contract: Mapping[str, Any],
    environment: Mapping[str, str],
    evaluation_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Return one complete, ingestion-ready metric bundle or raise."""
    _require_contract(contract)
    root = Path(model_path).resolve()
    origin = _read_json(root / "experiment_origin.json")
    provenance = _provenance(environment, origin)
    evaluation_rows = _read_jsonl(root / "evaluation_ticks.jsonl")
    evaluation_attestation = None
    if evaluation_manifest is not None:
        evaluation_attestation = _attest_evaluation_manifest(
            Path(evaluation_manifest).resolve(),
            evaluation_rows,
            expected_digest=provenance["evaluation_manifest_sha256"],
        )
        evaluation_rows = _rows_matching_evaluation_manifest_semantics(
            evaluation_rows,
            evaluation_attestation["evaluation_semantics"],
        )
    summary = _read_json(root / "causal_replay_summary.json")
    trajectory, quality_population = _quality_trajectory(
        evaluation_rows,
        _read_jsonl(root / "psnr_history.jsonl"),
    )
    h2d, d2h, sync_wait, runtime, mutation_population = _runtime_and_population_metrics(
        _read_jsonl(root / "decision_trace.jsonl")
    )
    gaussian_trajectory = [*quality_population, *mutation_population]
    metrics = {
        "quality.psnr_trajectory": trajectory,
        "quality.scored_holdout_psnr": trajectory[-1]["psnr_db"],
        "runtime.wall_clock_s": _finite_number(summary.get("elapsed_s"), field="wall clock"),
        "runtime.mapping_active_s": _finite_number(
            summary.get("mapping_active_s"), field="mapping active time"
        ),
        "runtime.transfer.h2d_bytes": h2d,
        "runtime.transfer.d2h_bytes": d2h,
        "runtime.sync.wait_ms": sync_wait,
        "runtime.gpu_transaction_latency_ms": runtime["latency"],
        "population.mutation_events": runtime["mutation"],
        "population.gaussian_count_trajectory": gaussian_trajectory,
    }
    result = {
        "schema": "eligsir.researchflow.population-metrics/v1",
        # ResearchFlow compiles manifest evidence names as dotted paths from
        # the metrics artifact root (for example, ``quality.psnr_trajectory``).
        # Keep the producer-specific bundle for existing consumers, while
        # exposing each declared path where the managed plan looks for it.
        **metrics,
        "metrics": metrics,
        "provenance": provenance,
        "source_artifacts": {
            "evaluation_ticks": str(root / "evaluation_ticks.jsonl"),
            "psnr_history": str(root / "psnr_history.jsonl"),
            "causal_summary": str(root / "causal_replay_summary.json"),
            "decision_trace": str(root / "decision_trace.jsonl"),
            "experiment_origin": str(root / "experiment_origin.json"),
        },
    }
    if evaluation_attestation is not None:
        result["evaluation_attestation"] = evaluation_attestation
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--evaluation-manifest")
    parsed = parser.parse_args(argv)
    metrics = assemble_metrics(
        parsed.model_path,
        contract=_read_json(Path(parsed.contract)),
        environment=os.environ,
        evaluation_manifest=parsed.evaluation_manifest,
    )
    output = Path(parsed.output).resolve()
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
