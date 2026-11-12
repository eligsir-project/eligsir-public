"""Evaluation-only ownership for the canonical causal mapping runner.

The split is decided from source metadata before any packet is submitted.
Holdout and guard identities therefore never cross the MappingSession boundary
and cannot enter supervision, replay, depth insertion, or mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from eligsir.contracts.runtime import SourceIdentity
from eligsir.plot_style import svg_output_path
from eligsir.runtime.causal_replay import CausalReplayClock
from eligsir.runtime.causal_throughput import (
    plot_causal_throughput,
    record_causal_throughput,
)
from eligsir.mapping.scheduler.historical_candidate_index import (
    MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
)
from eligsir.mapping.scheduler.resolution_tier_usage import (
    flush_resolution_tier_usage,
    plot_resolution_tier_over_time,
)

if TYPE_CHECKING:
    # `_evaluate_sync`'s `gaussians` argument is either the live model or,
    # on the async mapping-holdout path (issue #1314), one of these frozen
    # snapshots -- see eligsir/runtime/async_holdout_eval.py.
    from eligsir.runtime.async_holdout_eval import GaussianRenderSnapshot


def _scene_source_for_report(dataset: Any) -> str:
    return os.environ.get("SCENE_SOURCE", "") or str(dataset.source_path)


def _bounded_replay_identity_focus_updates(weights):
    """Select a deterministic bounded quality-focus update before publication.

    The quality controller may score the complete retained history at its
    low-cadence evaluation boundary.  The runtime control barrier receives
    only the highest-weight identities, keeping publication work independent
    of historical population.
    """

    focused = (
        (source, float(weight))
        for source, weight in weights
        if float(weight) > 1.0
    )
    selected = heapq.nsmallest(
        MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
        focused,
        key=lambda item: (-item[1], item[0].ordering_key()),
    )
    return dict(selected)


def _compose_replay_identity_focus_updates(*components):
    """Combine absolute controller boosts without double-counting debt."""

    combined = {}
    for component in components:
        for source, weight in component.items():
            combined[source] = max(combined.get(source, 1.0), weight)
    return _bounded_replay_identity_focus_updates(combined.items())


def _replay_candidate_shortlist_identity_hash(
    candidates: Iterable[SourceIdentity],
) -> str:
    """Hash the exact bounded scheduler shortlist in canonical source order."""

    identities = tuple(candidates)
    if not all(isinstance(source, SourceIdentity) for source in identities):
        raise TypeError("replay control candidates must be SourceIdentity values")
    ordered = sorted(identities, key=SourceIdentity.ordering_key)
    payload = "\n".join(source.fingerprint() for source in ordered).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def mapping_holdout_curve_metadata(args: Any) -> dict[str, Any]:
    """Resolve the opt-in mapping observer and its timing-validity contract.

    Three execution modes for a mapping-phase holdout sample:

    - disabled: no mapping-phase samples at all (post-source-completion only).
    - synchronous (``streaming_mapping_holdout_curve=True``, async off, or
      partition-control focus enabled): the historical behaviour -- render on
      the mapping thread, pausing it, unconditionally throughput-invalid.
    - async (``streaming_mapping_holdout_curve_async=True`` too, issue
      #1314): render off a snapshot stream + background thread instead.
      ``throughput_evidence_valid``/``_status`` here are PROVISIONAL --
      ``CausalEvaluationObserver`` overwrites them once real measured
      snapshot-copy-stall data exists (see
      ``_resolve_async_throughput_evidence``). Partition-control focus is
      out of scope for async mode (issue #1314 is specifically about the
      holdout-curve render, not the partition-control probe render) and
      keeps the synchronous path when both are requested together.
    """

    enabled = bool(getattr(args, "streaming_mapping_holdout_curve", False))
    async_requested = bool(
        getattr(args, "streaming_mapping_holdout_curve_async", False)
    )
    if async_requested and not enabled:
        raise ValueError(
            "streaming_mapping_holdout_curve_async requires "
            "streaming_mapping_holdout_curve"
        )
    partition_control_enabled = (
        float(getattr(args, "streaming_partition_focus_weight", 0.0)) > 0.0
    )
    interval = int(
        getattr(args, "streaming_psnr_chart_interval", 1000) or 0
    )
    max_views = int(
        getattr(args, "streaming_report_test_metrics_max", 128)
    )
    if enabled and interval <= 0:
        raise ValueError(
            "streaming_mapping_holdout_curve requires "
            "streaming_psnr_chart_interval > 0"
        )
    if enabled and max_views <= 0:
        raise ValueError(
            "streaming_mapping_holdout_curve requires "
            "streaming_report_test_metrics_max > 0"
        )
    async_mode = enabled and async_requested and not partition_control_enabled
    synchronous_pause = (enabled or partition_control_enabled) and not async_mode
    if async_mode:
        observer_execution = "async_snapshot_stream"
        mapping_execution_paused = False
        throughput_evidence_valid = True
        throughput_evidence_status = "pending_measured_async_snapshot_stall"
    elif synchronous_pause:
        observer_execution = "synchronous_mapping_thread_pause"
        mapping_execution_paused = True
        throughput_evidence_valid = False
        if enabled and partition_control_enabled:
            throughput_evidence_status = (
                "invalid_mapping_holdout_and_partition_control_observer_pause"
            )
        elif enabled:
            throughput_evidence_status = "invalid_mapping_holdout_observer_pause"
        else:
            throughput_evidence_status = "invalid_partition_control_observer_pause"
    else:
        observer_execution = "post_source_completion_only"
        mapping_execution_paused = False
        throughput_evidence_valid = True
        throughput_evidence_status = "valid_no_mapping_holdout_observer_pause"
    return {
        "enabled": enabled,
        "async_mode": async_mode,
        "async_requested": async_requested,
        "cadence_clock": "cumulative_optimizer_updates",
        "cadence_interval": interval,
        "max_scored_holdout_views": max_views,
        "observer_execution": observer_execution,
        "mapping_execution_paused": mapping_execution_paused,
        "throughput_evidence_valid": throughput_evidence_valid,
        "throughput_evidence_status": throughput_evidence_status,
        "async_snapshot_copy_stall_threshold_ms": float(
            getattr(
                args,
                "streaming_mapping_holdout_curve_async_stall_threshold_ms",
                5.0,
            )
        ),
    }


def mapping_clocks_for_report(
    mapping_clocks: Mapping[str, Any], curve_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Mark wall-clock throughput evidence invalid when mapping evaluation paused it."""

    clocks = dict(mapping_clocks)
    if bool(curve_metadata.get("throughput_evidence_valid", True)):
        return clocks
    clocks["throughput_evidence_valid"] = False
    clocks["throughput_evidence_status"] = str(
        curve_metadata.get(
            "throughput_evidence_status",
            "invalid_mapping_holdout_observer_pause",
        )
    )
    return clocks


def _holdout_rasterizer_context(registry: Any) -> dict[str, Any]:
    """Flush holdout timing once after its complete evaluation render pass."""
    registry.resolve_pending()
    stats = registry.snapshot().get("causal_holdout.rasterizer", {})
    if int(stats.get("count", 0)) <= 0:
        return {
            "rasterizer_gpu_ms": None,
            "rasterizer_timing_status": "unavailable_stage_gpu_events",
        }
    return {
        "rasterizer_gpu_ms": float(stats["total_ms"]),
        "rasterizer_timing_status": "available_async_gpu_events_total_tick",
    }


def _tick_constellation_telemetry(tick: Any) -> dict[str, Any] | None:
    plan = getattr(tick, "work_plan", None)
    constellation = getattr(plan, "supervision_constellation", None)
    if constellation is None:
        return None
    microbatch_sizes = tuple(
        int(size) for size in getattr(tick, "microbatch_sizes", ())
    )
    return {
        "requested_batch_size": int(constellation.requested_batch_size),
        "realized_batch_size": int(constellation.realized_batch_size),
        "requested_camera_count": int(constellation.requested_camera_count),
        "realized_camera_count": int(constellation.realized_camera_count),
        "unique_eligible_count": int(constellation.unique_eligible_count),
        "fill_ratio": float(constellation.fill_ratio),
        "microbatch_count": (
            len(microbatch_sizes)
            if microbatch_sizes
            else int(constellation.microbatch_count)
        ),
        "microbatch_sizes": list(microbatch_sizes),
        "optimizer_updates": int(getattr(tick, "optimizer_steps", 0)),
        "supervised_views": int(getattr(tick, "supervision_views", 0)),
        "unique_supervision_views": getattr(
            tick, "unique_supervision_views", None
        ),
        "supervised_pixels": int(getattr(tick, "supervised_pixels", 0)),
        "cumulative_realized_supervised_views": int(
            getattr(
                tick,
                "cumulative_supervision_views",
                constellation.cumulative_realized_supervised_views
                + int(getattr(tick, "supervision_views", 0)),
            )
        ),
        "cumulative_optimizer_updates": int(
            getattr(
                tick,
                "cumulative_optimizer_updates",
                constellation.cumulative_optimizer_updates
                + int(getattr(tick, "optimizer_steps", 0)),
            )
        ),
        "mode": str(constellation.mode),
        "reason": str(constellation.reason),
        "hysteresis_state": str(constellation.hysteresis_state),
    }


def _tick_resolution_tiers(tick: Any) -> list[str]:
    plan = getattr(tick, "work_plan", None)
    if plan is None:
        return []
    work_items = (
        *tuple(getattr(plan, "supervision_work", ())),
        *tuple(getattr(plan, "replay_work", ())),
    )
    return sorted(
        {
            str(getattr(getattr(work, "resolution_tier", None), "value", "unknown"))
            for work in work_items
        }
    )


def _terminal_evaluation_tick_clock_snapshot(observer: Any) -> dict[str, Any]:
    """Return the exact post-transaction state joined to the final metric pass."""

    tick = getattr(observer, "_latest_tick", None)
    if tick is None:
        raise RuntimeError("final report lacks an exact terminal mapping tick")
    required_tick_fields = (
        "cumulative_optimizer_updates",
        "cumulative_supervision_views",
        "microbatch_sizes",
        "optimizer_steps",
        "supervision_views",
        "unique_supervision_views",
        "supervised_pixels",
        "snapshot_id",
        "control_snapshot",
    )
    missing = [name for name in required_tick_fields if not hasattr(tick, name)]
    if missing:
        raise RuntimeError(
            "exact terminal mapping tick is missing fields: " + ", ".join(missing)
        )
    if tick.snapshot_id is None:
        raise RuntimeError("exact terminal mapping tick has no snapshot_id")
    control_epoch = getattr(tick.control_snapshot, "control_tick_epoch", None)
    if control_epoch is None:
        raise RuntimeError("exact terminal mapping tick has no control_tick_epoch")
    constellation = _tick_constellation_telemetry(tick)
    if constellation is None:
        raise RuntimeError("exact terminal mapping tick has no supervision constellation")

    optimizer_updates = int(tick.cumulative_optimizer_updates)
    supervision_views = int(tick.cumulative_supervision_views)
    if optimizer_updates != int(observer._optimizer_updates):
        raise RuntimeError("terminal optimizer update clock drifted from latest tick")
    if supervision_views != int(observer._supervision_views):
        raise RuntimeError("terminal supervision view clock drifted from latest tick")
    wall_clock_s = getattr(observer, "_elapsed_s", None)
    if wall_clock_s is None:
        wall_clock_s = getattr(observer, "_latest_tick_wall_clock_s", None)
    if wall_clock_s is None:
        raise RuntimeError("exact terminal mapping tick has no observed wall clock")
    mapping_active_s = getattr(observer, "_mapping_active_s", None)
    if mapping_active_s is None:
        raise RuntimeError("exact terminal mapping tick has no mapping-active clock")

    return {
        "source_frame_index": getattr(observer, "_source_frame_index", None),
        "frames_ingested": int(observer.runtime.frames_seen),
        "frames_trained": _completed_training_source_count(observer.runtime),
        "keyframes_admitted": int(observer.runtime._admitted_observations_total),
        "optimizer_step": optimizer_updates,
        "supervision_views": supervision_views,
        "mapping_snapshot_id": int(tick.snapshot_id),
        "control_tick_epoch": int(control_epoch),
        "supervision_constellation": constellation,
        "wall_clock_s": float(wall_clock_s),
        "accumulated_iteration_wall_s": float(mapping_active_s),
        "last_ingest_iter": observer.last_ingest_iteration,
    }


def _completed_training_source_count(runtime: Any) -> int:
    """Count unique completed sources, retaining legacy camera telemetry only if absent."""

    provider = getattr(runtime, "successful_supervision_uses_by_identity", None)
    if callable(provider):
        return sum(int(uses) > 0 for uses in provider().values())
    return len(runtime.scene.train_cameras)


def bounded_evaluation_cameras(
    cameras: Iterable[Any], *, max_items: int
) -> list[Any]:
    """Return a deterministic trajectory-spanning periodic-evaluation pool.

    The frozen holdout identity remains the complete camera pool.  Only the
    repeated render workload is bounded, using the same evenly-spaced policy
    as the final comparison report, so evaluation cannot dominate a long
    refinement tail while still covering the beginning and end of the path.
    """

    cameras = list(cameras)
    max_items = int(max_items)
    if not cameras or max_items <= 0:
        return []
    if len(cameras) <= max_items:
        return cameras
    indices = np.linspace(
        0, len(cameras) - 1, num=max_items, dtype=np.int64
    ).tolist()
    return [cameras[index] for index in dict.fromkeys(indices)]


def prepared_report_source_identity_records(
    records: Iterable[Any],
    *,
    successful_supervision_uses: Mapping[SourceIdentity, int],
    max_items: int,
    packet_source_id: str | None = None,
    packet_stream_epoch: int = 0,
) -> list[Any]:
    """Choose a bounded, trajectory-spanning report-only mapped-view sample.

    Matches by the full canonical ``SourceIdentity`` rather than
    ``source_ordinal`` alone, so a reconnected stream epoch or another source
    namespace reusing the same ordinal (issue #1688) cannot alias a record
    that was never actually supervised into the report-camera selection.
    """
    from eligsir.contracts.streaming_bridge import frame_source_identity
    from eligsir.runtime.replay import streaming_frame_source_identity

    selected = []
    for record in records:
        identity = (
            frame_source_identity(record)
            if packet_source_id is None
            else streaming_frame_source_identity(
                record, source_id=packet_source_id, stream_epoch=packet_stream_epoch
            )
        )
        if int(successful_supervision_uses.get(identity, 0)) > 0:
            selected.append(record)
    return bounded_evaluation_cameras(selected, max_items=max_items)


def append_primary_training_psnr(
    model_path: str | Path,
    *,
    iteration: int,
    psnr_db: float,
    source_id: str,
    mapping_active_s: float | None = None,
    supervision_views: int | None = None,
    resolution_tiers: list[str] | None = None,
    rasterize_mode: str | None = None,
    evaluation_phase: str | None = None,
) -> None:
    """Append one already-computed primary-supervision PSNR sample."""

    row = {
        "iteration": int(iteration),
        "primary_training_psnr_db": float(psnr_db),
        "source_id": str(source_id),
        "score_scope": "diagnostic_realized_supervision_grid_pre_update",
    }
    if evaluation_phase is not None:
        row["evaluation_phase"] = str(evaluation_phase)
    if rasterize_mode is not None:
        row["rasterize_mode"] = str(rasterize_mode)
    if resolution_tiers is not None:
        row["training_resolution_tiers"] = list(resolution_tiers)
    if mapping_active_s is not None:
        row["mapping_active_s"] = float(mapping_active_s)
    if supervision_views is not None:
        row["supervision_views"] = int(supervision_views)
    path = Path(model_path) / "primary_training_psnr.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_paired_psnr_axes(
    model_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load comparable PSNR coordinates without treating transactions as work."""

    root = Path(model_path)
    series: dict[str, list[dict[str, float]]] = {
        "holdout": [],
        "training": [],
    }
    holdout_rows = _jsonl_rows(root / "evaluation_ticks.jsonl")
    from eligsir.utils.psnr_history_io import psnr_reduction

    psnr_reduction(
        holdout_rows,
        source=str(root / "evaluation_ticks.jsonl"),
        value_key="mean_test_psnr_db",
    )
    for row in holdout_rows:
        # The final report uses a distinct metric path at the same map/work
        # state; connecting it would invent a quality change without training.
        if row.get("evaluation_phase") == "post_mapping_report":
            continue
        values = (
            row.get("iteration"),
            row.get("accumulated_iteration_wall_s"),
            row.get("supervision_views"),
            row.get("mean_test_psnr_db"),
        )
        if all(value is not None for value in values):
            point = {
                "iteration": float(values[0]),
                "mapping_active_s": float(values[1]),
                "supervision_views": float(values[2]),
                "psnr_db": float(values[3]),
            }
            if "throughput_evidence_valid" in row:
                point["throughput_evidence_valid"] = row[
                    "throughput_evidence_valid"
                ]
            series["holdout"].append(point)
    for row in _jsonl_rows(root / "primary_training_psnr.jsonl"):
        values = (
            row.get("iteration"),
            row.get("mapping_active_s"),
            row.get("supervision_views"),
            row.get("primary_training_psnr_db"),
        )
        if all(value is not None for value in values):
            point = {
                "iteration": float(values[0]),
                "mapping_active_s": float(values[1]),
                "supervision_views": float(values[2]),
                "psnr_db": float(values[3]),
            }
            if "throughput_evidence_valid" in row:
                point["throughput_evidence_valid"] = row[
                    "throughput_evidence_valid"
                ]
            series["training"].append(point)
    return series


def plot_paired_psnr(
    model_path: str | Path, *, last_ingest_iteration: int | None
) -> Path:
    """Plot optimizer-primary and frozen-holdout PSNR without conflating them."""

    root = Path(model_path)

    training = _jsonl_rows(root / "primary_training_psnr.jsonl")
    holdout = _jsonl_rows(root / "psnr_history.jsonl")
    from eligsir.utils.psnr_history_io import psnr_reduction

    psnr_reduction(holdout, source=str(root / "psnr_history.jsonl"))
    if not training and not holdout:
        raise RuntimeError("paired PSNR plot requires at least one curve sample")
    missing_mapping_holdout = not any(
        row.get("evaluation_phase") == "mapping"
        or (last_ingest_iteration is not None and int(row["iteration"]) < last_ingest_iteration)
        for row in holdout
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _mark_execution_transitions(axis, rows, x_key):
        previous_tiers = None
        previous_mode = None
        for row in rows:
            tiers = tuple(row.get("training_resolution_tiers") or ())
            mode = row.get("rasterize_mode")
            x_value = row.get(x_key)
            if tiers and x_value is not None and tiers != previous_tiers:
                axis.axvline(float(x_value), color="0.5", linestyle=":", alpha=0.45)
                axis.annotate(
                    "/".join(tiers),
                    (float(x_value), 1.0),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                )
            if tiers:
                previous_tiers = tiers
            if mode is not None and x_value is not None and mode != previous_mode:
                axis.axvline(float(x_value), color="C2", linestyle="--", alpha=0.35)
                axis.annotate(
                    str(mode),
                    (float(x_value), 0.0),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    va="bottom",
                    ha="left",
                    fontsize=7,
                    color="C2",
                )
                previous_mode = mode

    fig, axis = plt.subplots(figsize=(10, 5))
    if holdout:
        axis.plot(
            [int(row["iteration"]) for row in holdout],
            [float(row["mean_test_psnr"]) for row in holdout],
            marker="o",
            label="frozen original-frame holdout",
        )
    if training:
        axis.plot(
            [int(row["iteration"]) for row in training],
            [float(row["primary_training_psnr_db"]) for row in training],
            marker="x",
            label="primary supervision (pre-update; realized grid diagnostic)",
        )
    _mark_execution_transitions(axis, training or holdout, "iteration")
    if last_ingest_iteration is not None:
        axis.axvline(
            int(last_ingest_iteration),
            color="grey",
            linestyle="--",
            linewidth=1,
            label="source complete",
        )
    axis.set_title("Frozen holdout and primary training PSNR")
    if missing_mapping_holdout:
        axis.text(0.02, 0.02, "No mid-mapping holdout samples recorded",
                  transform=axis.transAxes, fontsize=9, color="0.35")
    axis.set_xlabel("optimizer iteration")
    axis.set_ylabel("PSNR (dB)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    output = root / "psnr_paired_over_time.svg"
    fig.savefig(svg_output_path(output), dpi=150)
    plt.close(fig)

    work_series = _load_paired_psnr_axes(root)
    if work_series["holdout"] or work_series["training"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        wall_time_valid = all(
            point.get("throughput_evidence_valid") is not False
            for points in work_series.values()
            for point in points
        )
        for name, label, marker in (
            ("holdout", "frozen original-frame holdout", "o"),
            ("training", "primary supervision (pre-update)", "x"),
        ):
            points = work_series[name]
            if not points:
                continue
            if wall_time_valid:
                axes[0].plot(
                    [point["mapping_active_s"] for point in points],
                    [point["psnr_db"] for point in points],
                    marker=marker,
                    label=label,
                )
            axes[1].plot(
                [point["supervision_views"] for point in points],
                [point["psnr_db"] for point in points],
                marker=marker,
                label=label,
            )
        if wall_time_valid:
            _mark_execution_transitions(
                axes[0], training or holdout, "mapping_active_s"
            )
        else:
            axes[0].text(
                0.5,
                0.5,
                "UNAVAILABLE\nthroughput-invalid mapping observer pause",
                transform=axes[0].transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        _mark_execution_transitions(axes[1], training or holdout, "supervision_views")
        axes[0].set_xlabel(
            "mapping-active wall time (s)"
            if wall_time_valid
            else "mapping-active wall time unavailable"
        )
        axes[1].set_xlabel("cumulative supervised views")
        for axis in axes:
            axis.set_ylabel("PSNR (dB)")
            axis.grid(True, alpha=0.3)
        if wall_time_valid:
            axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "PSNR efficiency (cross-batch comparable axes)"
            if wall_time_valid
            else "PSNR by supervised views; wall-time evidence unavailable"
        )
        if missing_mapping_holdout:
            axes[1].text(0.02, 0.02, "No mid-mapping holdout samples recorded",
                         transform=axes[1].transAxes, fontsize=9, color="0.35")
        fig.tight_layout()
        fig.savefig(svg_output_path(root / "psnr_paired_over_work.svg"), dpi=150)
        plt.close(fig)
    return output


# Row/column caps for the per-cell heatmap panel below. A real run's
# "cells" population grows unboundedly as the camera visits new world
# space, so both axes are bounded for readability and render cost.
_SPATIAL_FOCUS_HEATMAP_MAX_ROWS = 120
_SPATIAL_FOCUS_HEATMAP_MAX_COLS = 60


def plot_spatial_replay_focus(model_path: str | Path) -> Path | None:
    """Plot bounded world-cell PSNR summaries and a per-cell PSNR heatmap.

    Panel A is the pooled median + p10-p90 envelope across every cell
    scored at each check-in -- the informative fast-ramp-then-plateau
    training-quality story, kept as-is.

    Panel B used to overlay a handful of individually-selected "weakest
    final cell" lines, labeled by their long numeric ``cell_key`` (an
    opaque bijective world-space packing, not a hash, but unreadable at a
    glance either way) -- dense, crossing, and uninformative once more
    than a couple of cells are involved. It is replaced here by a
    cell x iteration-bin heatmap covering every cell ever scored (subject
    to the row/column caps above), colored by pooled RGB PSNR (the only
    per-cell time series ``spatial_replay_focus.jsonl`` carries -- no
    per-cell visit/replay-frequency field exists in the schema). Rows are
    ordered by each cell's first-observed iteration, a proxy for
    trajectory admission order that keeps spatially-nearby cells visually
    nearby without needing a separate arc-length computation. Any
    remaining per-row label uses only the cell's (x, y, z) center in
    meters -- never the numeric ``cell_key``.
    """

    root = Path(model_path)
    trace = root / "spatial_replay_focus.jsonl"
    if not trace.is_file():
        return None
    rows = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row.get("cells")]
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: int(row["iteration"]))
    iterations = np.asarray([int(row["iteration"]) for row in rows])
    values = [
        np.asarray([float(cell["psnr_db"]) for cell in row["cells"]])
        for row in rows
    ]
    p10 = [float(np.quantile(item, 0.1)) for item in values]
    median_values = [float(np.median(item)) for item in values]
    p90 = [float(np.quantile(item, 0.9)) for item in values]

    # ---- Panel B data: per-cell PSNR over time ----
    cell_centers: dict[int, tuple[float, float, float]] = {}
    first_seen_iteration: dict[int, int] = {}
    per_cell_series: dict[int, dict[int, float]] = {}
    for row in rows:
        iteration = int(row["iteration"])
        for cell in row["cells"]:
            key = int(cell["cell_key"])
            per_cell_series.setdefault(key, {})[iteration] = float(cell["psnr_db"])
            if key not in first_seen_iteration:
                first_seen_iteration[key] = iteration
                center = cell.get("center_xyz_m", (0.0, 0.0, 0.0))
                cell_centers[key] = (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                )

    # Admission-order-then-key for a stable, spatially-meaningful sort.
    ordered_keys = sorted(
        per_cell_series, key=lambda key: (first_seen_iteration[key], key)
    )

    # Cap displayed rows by evenly subsampling the ordered list -- keeps
    # real, unaveraged per-cell values rather than synthesizing blended
    # ones, while still spanning the full trajectory when there are more
    # cells than fit.
    if len(ordered_keys) > _SPATIAL_FOCUS_HEATMAP_MAX_ROWS:
        stride = len(ordered_keys) / _SPATIAL_FOCUS_HEATMAP_MAX_ROWS
        picks = sorted(
            {
                min(int(i * stride), len(ordered_keys) - 1)
                for i in range(_SPATIAL_FOCUS_HEATMAP_MAX_ROWS)
            }
        )
        ordered_keys = [ordered_keys[i] for i in picks]

    n_cols = max(1, min(_SPATIAL_FOCUS_HEATMAP_MAX_COLS, len(iterations)))
    bin_edges = np.linspace(float(iterations[0]), float(iterations[-1]), n_cols + 1)
    bin_edges[-1] += 1.0  # right-inclusive last bin

    heatmap = np.full((len(ordered_keys), n_cols), np.nan)
    for row_idx, key in enumerate(ordered_keys):
        series = per_cell_series[key]
        sums = np.zeros(n_cols)
        counts = np.zeros(n_cols)
        for iteration, psnr_db in series.items():
            col = int(np.searchsorted(bin_edges, float(iteration), side="right") - 1)
            col = min(max(col, 0), n_cols - 1)
            sums[col] += psnr_db
            counts[col] += 1
        with np.errstate(invalid="ignore"):
            heatmap[row_idx] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    fig, (axis, heat_axis) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 2]}
    )
    axis.fill_between(iterations, p10, p90, alpha=0.18, label="cell p10-p90")
    axis.plot(iterations, median_values, color="black", linewidth=2, label="cell median")
    axis.set_title("Training-only world-microcell PSNR history (all scored cells)")
    axis.set_xlabel("optimizer iteration")
    axis.set_ylabel("pooled RGB PSNR (dB)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")

    masked = np.ma.masked_invalid(heatmap)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")
    mesh = heat_axis.pcolormesh(
        bin_edges,
        np.arange(len(ordered_keys) + 1),
        masked,
        cmap=cmap,
        shading="flat",
    )
    heat_axis.invert_yaxis()
    heat_axis.set_xlabel("optimizer iteration (binned)")
    heat_axis.set_ylabel(
        f"cell (n={len(ordered_keys)}, top-to-bottom = trajectory admission order)"
    )
    heat_axis.set_title(
        "Per-cell PSNR heatmap (grey = not scored in that iteration bin)"
    )
    if len(ordered_keys) <= 20:
        heat_axis.set_yticks(np.arange(len(ordered_keys)) + 0.5)
        heat_axis.set_yticklabels(
            [
                "({:.2f},{:.2f},{:.2f})m".format(*cell_centers[key])
                for key in ordered_keys
            ],
            fontsize="x-small",
        )
    else:
        heat_axis.set_yticks([])
    fig.colorbar(mesh, ax=heat_axis, label="pooled RGB PSNR (dB)")

    fig.tight_layout()
    output = root / "spatial_replay_focus_over_time.svg"
    fig.savefig(svg_output_path(output), dpi=150)
    plt.close(fig)
    return output


def _ordinal(frame: Any) -> int:
    value = getattr(frame, "source_ordinal", None)
    return int(frame.index if value is None else value)


def _finite_pose(frame: Any) -> np.ndarray | None:
    value = getattr(frame, "c2w", None)
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float32)
    return pose if pose.shape == (4, 4) and np.isfinite(pose).all() else None


def _pose_gap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation, math.degrees(math.acos(cosine))


@dataclass(frozen=True, slots=True)
class CausalHoldoutPlan:
    """Immutable source role assignment and its selection-rule evidence."""

    holdout_ordinals: frozenset[int]
    guard_ordinals: frozenset[int]
    holdout_records: tuple[Any, ...]
    eval_metadata: Mapping[str, Any]
    control_probe_records: tuple[Any, ...] = ()

    def role_for_ordinal(self, ordinal: int) -> str:
        ordinal = int(ordinal)
        if ordinal in self.holdout_ordinals:
            return "holdout"
        if ordinal in self.guard_ordinals:
            return "guard"
        return "train"

    def role_for_packet(self, packet: Any) -> str:
        return self.role_for_ordinal(packet.source.source_ordinal)


def validate_train_evwb_reference_source(
    frames: Iterable[Any],
    plan: CausalHoldoutPlan,
    args: Any,
    *,
    source_id: str,
    stream_epoch: int = 0,
) -> None:
    """Require the opt-in EV/WB gauge source to be a canonical train frame."""
    if not bool(getattr(args, "streaming_train_evwb_enabled", False)):
        return
    reference = str(
        getattr(args, "streaming_train_evwb_reference_source", "") or ""
    )
    if not reference:
        raise ValueError(
            "streaming_train_evwb_enabled requires an exact reference source identity"
        )
    from eligsir.runtime.replay import streaming_frame_source_identity

    matches = [
        frame
        for frame in frames
        if streaming_frame_source_identity(
            frame, source_id=source_id, stream_epoch=stream_epoch
        ).fingerprint()
        == reference
    ]
    if len(matches) != 1:
        raise ValueError(
            "streaming train EV/WB reference does not resolve to the source catalogue"
        )
    role = plan.role_for_ordinal(_ordinal(matches[0]))
    if role != "train":
        raise ValueError(
            "streaming train EV/WB reference must resolve to a canonical train source "
            f"(got {role})"
        )


def enable_training_capture_pose_candidates(
    frames: Iterable[Any], plan: CausalHoldoutPlan
) -> None:
    """Permit precomputed capture-time poses only for the mapping partition."""
    for frame in frames:
        if hasattr(frame, "_mapping_c2w"):
            frame._mapping_pose_for_training = (
                plan.role_for_ordinal(_ordinal(frame)) == "train"
            )


@dataclass(frozen=True, slots=True)
class _PartitionControlMeasurement:
    """Read-only probe result, published only after the map branch is accepted."""

    roles: Any
    decision: Any
    published: Mapping[SourceIdentity, float]
    telemetry: Mapping[str, Any]


def causal_holdout_plan(frames: Iterable[Any], args: Any) -> CausalHoldoutPlan:
    """Precompute the legacy every-Nth + pose-gap split for a finite source."""

    records = tuple(frames)
    eval_hold = int(args.streaming_eval_hold or 0)
    bootstrap = min(
        max(0, int(getattr(args, "streaming_initial_frames", 5))), len(records)
    )
    guard = max(0, int(getattr(args, "streaming_eval_guard", 0)))
    min_translation = float(
        getattr(args, "streaming_eval_min_baseline_m", 0.03)
    )
    min_angle = float(getattr(args, "streaming_eval_min_angle_deg", 2.0))
    source_id = str(getattr(args, "source_path", "") or "")

    candidates = (
        []
        if eval_hold <= 0
        else [
            position
            for position, frame in enumerate(records)
            if _ordinal(frame) % eval_hold == 0 and _ordinal(frame) >= bootstrap
        ]
    )
    candidate_positions = set(candidates)
    holdouts: set[int] = set()
    guards: set[int] = set()
    for position in candidates:
        pose = _finite_pose(records[position])
        left_guards = range(
            position - 1, max(position - guard - 1, -1), -1
        )
        right_guards = range(position + 1, min(position + guard + 1, len(records)))
        gap_translation = gap_angle = float("inf")
        if pose is not None:
            for neighbour in range(position - guard - 1, -1, -1):
                if neighbour in candidate_positions:
                    continue
                neighbour_pose = _finite_pose(records[neighbour])
                if neighbour_pose is not None:
                    gap_translation, gap_angle = _pose_gap(pose, neighbour_pose)
                    break
            for neighbour in range(position + guard + 1, len(records)):
                if neighbour in candidate_positions:
                    continue
                neighbour_pose = _finite_pose(records[neighbour])
                if neighbour_pose is not None:
                    other_translation, other_angle = _pose_gap(pose, neighbour_pose)
                    gap_translation = min(gap_translation, other_translation)
                    gap_angle = min(gap_angle, other_angle)
                    break
        if (
            pose is None
            or gap_translation >= min_translation
            or gap_angle >= min_angle
        ):
            holdouts.add(_ordinal(records[position]))
            guards.update(_ordinal(records[p]) for p in (*left_guards, *right_guards))

    if candidates and not holdouts:
        holdouts = {_ordinal(records[position]) for position in candidates}
        guards.clear()
    guards.difference_update(holdouts)
    # issue #1337: a too-permissive --streaming_eval_guard (default 0) on a
    # smooth/high-fps trajectory can make nearly every candidate fail the
    # pose-gap check against its immediately-adjacent neighbor, collapsing
    # the realized holdout set to a tiny fraction of the reserved candidate
    # count -- confirmed to go as low as 1 of ~243 on Replica's 30Hz path --
    # while report.json still shows eval_valid: True. The existing
    # candidates-but-no-holdouts fallback above only catches the n=0 case, so
    # a non-empty-but-collapsed set slips through silently. Flag it loudly
    # (both here and in the returned metadata) rather than let a run quietly
    # score against ~1 view while claiming a valid measurement.
    holdout_severely_collapsed = (
        len(candidates) >= 5 and 0 < len(holdouts) < 0.2 * len(candidates)
    )
    if holdout_severely_collapsed:
        print(
            f"[causal-holdout] WARNING: realized holdout set collapsed to "
            f"{len(holdouts)} of {len(candidates)} reserved candidates "
            f"(--streaming_eval_guard={guard}). mean_test_psnr is likely "
            f"scored against too few views to be meaningful. Raise "
            f"--streaming_eval_guard for this trajectory's frame rate.",
            flush=True,
        )
    control_probe_records: tuple[Any, ...] = ()
    if (
        float(getattr(args, "streaming_partition_focus_weight", 0.0)) > 0.0
        and str(
            getattr(
                args,
                "streaming_partition_focus_signal_source",
                "training_derived",
            )
        )
        == "control_probe"
    ):
        control_probe_records = tuple(bounded_evaluation_cameras(
            tuple(
                frame
                for frame in records
                if _ordinal(frame) >= bootstrap
                and _ordinal(frame) not in holdouts
                and _ordinal(frame) not in guards
                and _finite_pose(frame) is not None
            ),
            max_items=int(
                getattr(args, "streaming_partition_focus_control_probe_n", 32)
            ),
        ))
    holdout_records = tuple(
        frame for frame in records if _ordinal(frame) in holdouts
    )
    rule_payload = "|".join(
        (
            "causal_precomputed_v1",
            f"source={source_id}",
            f"eval_hold={eval_hold}",
            f"bootstrap={bootstrap}",
            f"guard={guard}",
            f"min_t={min_translation:.6f}",
            f"min_a={min_angle:.6f}",
            "holdouts=" + ",".join(str(value) for value in sorted(holdouts)),
        )
    )
    holdout_hash = (
        hashlib.sha1(rule_payload.encode("utf-8")).hexdigest()[:16]
        if holdouts
        else None
    )
    metadata = MappingProxyType(
        {
            "eval_hold": eval_hold,
            "eval_min_baseline_m": min_translation,
            "eval_min_angle_deg": min_angle,
            "eval_guard": guard,
            "n_holdout": len(holdouts),
            "n_holdout_candidates": len(candidates),
            "holdout_severely_collapsed": holdout_severely_collapsed,
            "n_guard": len(guards),
            "holdout_hash": holdout_hash,
            "holdout_selection": "precomputed",
            "selection_rule_version": "causal_precomputed_v1",
        }
    )
    return CausalHoldoutPlan(
        holdout_ordinals=frozenset(holdouts),
        guard_ordinals=frozenset(guards),
        holdout_records=holdout_records,
        eval_metadata=metadata,
        control_probe_records=control_probe_records,
    )


def build_causal_holdout_cameras(
    plan: CausalHoldoutPlan | Iterable[Any],
    *,
    data_device: Any,
    report_source_id: str | None = None,
    report_stream_epoch: int = 0,
) -> list[Any]:
    """Create render cameras with lazy CPU GT and stable source identities."""

    import torch

    from scene.cameras import Camera
    from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb
    from eligsir.runtime.replay import streaming_frame_source_identity
    from eligsir.utils.graphics_utils import focal2fov
    from eligsir.utils.rgbd_frames import c2w_to_camera_rt

    records = (
        plan.holdout_records
        if isinstance(plan, CausalHoldoutPlan)
        else tuple(plan)
    )
    cameras = []
    for frame in records:
        pose = _finite_pose(frame)
        if pose is None:
            continue
        ordinal = _ordinal(frame)
        width, height = int(frame.width), int(frame.height)
        rotation, translation = c2w_to_camera_rt(pose)

        def load_rgb(record=frame):
            array = np.asarray(load_frame_rgb(record), dtype=np.float32) / 255.0
            image = torch.from_numpy(
                np.ascontiguousarray(array[..., :3].transpose(2, 0, 1))
            )
            return image, None

        def load_depth(record=frame):
            return load_frame_depth_np(record)

        camera = Camera(
            colmap_id=ordinal,
            R=rotation,
            T=translation,
            FoVx=focal2fov(float(frame.fx), width),
            FoVy=focal2fov(float(frame.fy), height),
            image=None,
            gt_alpha_mask=None,
            image_name=f"{ordinal:06d}",
            uid=ordinal,
            data_device=data_device,
            fx=float(frame.fx),
            fy=float(frame.fy),
            cx=float(frame.cx),
            cy=float(frame.cy),
            width=width,
            height=height,
            pending_image_loader=load_rgb,
        )
        # Deliberately do not attach the training-path ``_streaming_frame``
        # marker: Camera.materialize_pending_image treats that marker as a
        # request to construct a depth-bearing MappingFrame.  Evaluation GT
        # needs RGB only.  ``image_name`` is the report contract's stable
        # identity fallback, while uid/colmap_id retain the partition index.
        camera._causal_holdout_record = frame
        if report_source_id is not None:
            # Report-only identity uses the same packet namespace/epoch
            # resolver as prepared_report_source_identity_records().
            camera._report_source_identity = streaming_frame_source_identity(
                frame,
                source_id=report_source_id,
                stream_epoch=report_stream_epoch,
            )
        # Evaluation cameras remain semantically separate from training
        # cameras: do not attach ``_streaming_frame``.  The reloadable loader
        # gives observational temporal metrics access to the original sensor
        # depth for path-, bytes-, and numpy-backed sources without granting
        # the holdout any mapping/materialization authority.
        camera._sensor_depth_loader = load_depth
        camera._streaming_depth_scale = float(
            getattr(frame, "depth_scale", 1000.0)
        )
        camera._sensor_depth_cache = None
        camera._streaming_timestamp = float(getattr(frame, "timestamp", 0.0) or 0.0)
        cameras.append(camera)
    return cameras


def _render_partition_control_tick(runtime: Any, cameras: Iterable[Any]) -> Any:
    """Render only the role-resolved non-oracle control cameras."""

    from gaussian_renderer import render
    import torch
    from eligsir.utils.comparison_report import _render_image
    from eligsir.utils.psnr_history import (
        PsnrTickResult,
        _aggregate_clamp_records,
    )
    from eligsir.utils.psnr_history_io import PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1
    from eligsir.utils.score_contract import (
        apply_declared_camera_grid_score_clamp_pair,
        score_tensor_from_camera,
        scientific_psnr,
    )

    rendered_cameras = []
    values = []
    clamp_records = []
    tiers = []
    resolutions = []
    with torch.no_grad():
        for camera in cameras:
            if getattr(camera, "has_pending_image", False):
                camera.materialize_pending_image()
            render_image = _render_image(
                camera,
                runtime.gaussians,
                render,
                runtime.pipe,
                runtime._background,
            )
            gt = score_tensor_from_camera(camera, device=render_image.device)
            render_score, gt_score, clamp_record = (
                apply_declared_camera_grid_score_clamp_pair(
                    render_image,
                    gt,
                    context="causal_partition_control:control_probe",
                )
            )
            values.append(
                scientific_psnr(
                    render_score,
                    gt_score,
                    context="causal_partition_control:control_probe",
                ).mean()
            )
            rendered_cameras.append(camera)
            clamp_records.append(clamp_record)
            tier = getattr(camera, "_canonical_resolution_tier", None)
            tiers.append(getattr(tier, "value", "native"))
            resolutions.append(
                (int(camera.image_width), int(camera.image_height))
            )
    scores = None if not values else torch.stack(values)
    return PsnrTickResult(
        cams=rendered_cameras,
        psnr_tensor=scores,
        mean_psnr=(
            None if scores is None else float(scores.mean().item())
        ),
        score_domain_clamp=(
            None
            if not clamp_records
            else {
                "aggregate": _aggregate_clamp_records(clamp_records),
                "per_view": clamp_records,
            }
        ),
        training_resolution_tiers=tuple(tiers),
        evaluation_resolutions=tuple(resolutions),
        psnr_reduction=PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1,
    )


def _partition_focus_publication_priorities(decision) -> dict[int, float]:
    """Severities handed to ``_partition_focus_publication_weights``.

    The debt allocator (issue #1723) already ranked and selected its batch;
    every selected frame must receive equal focus mass. Publishing raw debt as
    severity let a single debt outlier take the whole mass and floored the
    rest to 1.0, which the bounded publisher then drops -- 19 of 20 selected
    frames in the real runs. ``decision.priority_weights`` keeps the raw debt
    for telemetry; only the publication severities are flattened. Legacy
    paths keep their ranked severities unchanged.
    """

    priorities = dict(decision.priority_weights)
    if getattr(decision, "repair_mode", None) == "debt_allocator":
        return {index: 2.0 for index in priorities}
    return priorities


def _partition_focus_publication_weights(
    priorities: Mapping[int, float],
    selected_indices: Iterable[int],
    *,
    candidate_population: int,
    focus_probability: float,
) -> tuple[dict[int, float], float, float]:
    """Translate the legacy focus-mixture probability into sampler weights.

    A probability ``p`` means ``p`` targeted sampling plus ``1-p`` broad
    sampling. The weighted provider has one distribution, so preserve the
    equivalent aggregate focused mass while leaving every global identity at
    neutral weight. The 0.95 ceiling keeps a nonzero broad component even for
    the historical ``p=1`` request; the reserved-global cadence remains the
    stronger realized-selection starvation guard.
    """

    selected = tuple(dict.fromkeys(int(index) for index in selected_indices))
    population = max(len(selected), int(candidate_population))
    requested = min(1.0, max(0.0, float(focus_probability)))
    resolved = min(0.95, requested)
    if not selected:
        return {}, requested, resolved
    neutral_count = max(0, population - len(selected))
    if neutral_count == 0:
        return {index: 1.0 for index in selected}, requested, resolved

    base_share = len(selected) / population
    target_share = base_share + resolved * (1.0 - base_share)
    target_mass = neutral_count * target_share / max(1e-9, 1.0 - target_share)
    severities = {
        index: max(1e-6, float(priorities.get(index, 1.0)) - 1.0)
        for index in selected
    }
    severity_sum = sum(severities.values())
    return (
        {
            index: max(1.0, target_mass * severity / severity_sum)
            for index, severity in severities.items()
        },
        requested,
        resolved,
    )


def _debt_allocator_partition_focus_decision(
    *,
    source: Any,
    cell_psnr: Mapping[int, float],
    cell_view_counts: Mapping[int, int],
    active_region_ids: Mapping[int, tuple[int, ...]],
    cached_region_ids: Mapping[int, tuple[int, ...]],
    debt: Mapping[int, float],
    batch_size: int,
) -> tuple["PartitionFocusDecision", dict[int, float]]:
    """Debt-allocator branch of ``_measure_partition_control``'s spatial-substrate path.

    Extracted as a standalone, dependency-light function (issue #1723 follow-up)
    so it is directly unit-testable without constructing a full
    ``CausalEvaluationObserver`` or render pipeline -- everything it needs is
    passed in explicitly. Wraps
    ``utils.partition_control.decide_partition_focus_debt_allocator`` and adapts
    its ``DebtAllocatorDecision`` (debt/selected_indices/demoted_indices) into
    the same ``PartitionFocusDecision`` shape the rest of
    ``_measure_partition_control`` already consumes downstream (publication
    weighting, recorder, etc.).

    Builds ``target = {cell: 1.0 for cell in cell_psnr}`` fresh every call so
    every cell in ``cell_psnr`` is always present in the allocator's ``target``
    argument -- satisfying its documented caller obligation that cells missing
    from ``target`` silently lose accumulated debt.

    Returns ``(decision, updated_debt)``; the caller owns persisting
    ``updated_debt`` across ticks (same caller-owns-debt contract as
    ``decide_partition_focus_debt_allocator`` itself).
    """

    from eligsir.utils.partition_control import (
        PartitionFocusDecision,
        decide_partition_focus_debt_allocator,
    )

    if not cell_psnr:
        # cell_psnr is legitimately empty until SpatialMicrocellReplayFocus
        # Controller.current_cell_quality() clears its _min_history sample
        # floor, and can empty out again mid-run via stale-source pruning.
        # Calling the allocator with an empty target would wipe every cell's
        # accumulated debt to {} (target={cell: 1.0 for cell in cell_psnr}
        # is empty too, so decide_partition_focus_debt_allocator's "cells
        # missing from target lose debt" contract nukes it all) while
        # active_region_ids/cached_region_ids stay non-empty, degenerating
        # into a priority-0.0-for-everyone actuation instead of a real
        # no-op. Make this tick a genuine no-op: leave debt untouched and
        # promote/demote nothing.
        return (
            PartitionFocusDecision(
                source=source,
                partition_means=cell_psnr,
                partition_counts=cell_view_counts,
                bad_partitions=(),
                good_partitions=(),
                priority_weights={},
                promote_indices=(),
                demote_indices=(),
                repair_mode="debt_allocator_no_signal",
                repair_extra_optimizer_steps=0,
            ),
            dict(debt),
        )

    target = {cell: 1.0 for cell in cell_psnr}
    allocator_decision = decide_partition_focus_debt_allocator(
        cell_psnr=cell_psnr,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        debt=debt,
        target=target,
        batch_size=batch_size,
    )

    def _worst_covered_debt(index: int) -> float:
        cells = active_region_ids.get(index) or cached_region_ids.get(index) or ()
        return max((allocator_decision.debt.get(c, 0.0) for c in cells), default=0.0)

    promoted = tuple(
        idx for idx in allocator_decision.selected_indices
        if idx in cached_region_ids
    )
    decision = PartitionFocusDecision(
        source=source,
        partition_means=cell_psnr,
        partition_counts=cell_view_counts,
        # Intentionally empty: the debt allocator does not use the ordinal
        # ranked-outlier-detection concept these two fields describe. This
        # is safe because partition_pressure (derived from bad_partitions)
        # is only consumed by orphaned dead code, and prior_bad_partitions
        # hysteresis is only read on the ordinal decide_partition_focus
        # `else` branch, which a debt-allocator run never reaches.
        bad_partitions=(),
        good_partitions=(),
        priority_weights={
            idx: _worst_covered_debt(idx)
            for idx in allocator_decision.selected_indices
        },
        promote_indices=promoted,
        # NOTE: not comparable to the legacy paths' demote count -- see
        # decide_partition_focus_debt_allocator's docstring in
        # utils/partition_control.py for why.
        demote_indices=allocator_decision.demoted_indices,
        repair_mode="debt_allocator",
        repair_extra_optimizer_steps=0,
    )
    return decision, allocator_decision.debt


class CausalEvaluationObserver:
    """Evaluate a frozen causal holdout without perturbing live mapping by default.

    Rendering while the source clock is live would pause submission and turn
    evaluation overhead into artificial transport pressure.  By default, the
    first tick is therefore the exact source-completion boundary.  An explicit
    development/evaluation flag can request mapping-period samples; those rows
    and the final report are then marked throughput-invalid.  Every periodic
    sample uses cumulative optimizer updates, never raw loop ticks.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        plan: CausalHoldoutPlan,
        dataset: Any,
        args: Any,
        source_total: int | None,
        source_records: Iterable[Any] = (),
        packet_source_id: str | None = None,
        packet_stream_epoch: int = 0,
    ) -> None:
        self.runtime = runtime
        self.plan = plan
        self.dataset = dataset
        self.args = args
        self.source_total = source_total
        self._source_records = tuple(source_records)
        self._packet_source_id = packet_source_id
        self._packet_stream_epoch = int(packet_stream_epoch)
        self.interval = max(
            1, int(getattr(args, "streaming_psnr_chart_interval", 1000) or 1000)
        )
        self._mapping_holdout_curve_metadata = mapping_holdout_curve_metadata(args)
        self._mapping_holdout_curve_enabled = bool(
            self._mapping_holdout_curve_metadata["enabled"]
        )
        self._async_holdout_evaluator = None
        if bool(self._mapping_holdout_curve_metadata["async_mode"]):
            from eligsir.runtime.async_holdout_eval import AsyncHoldoutEvaluator

            self._async_holdout_evaluator = AsyncHoldoutEvaluator()
        self.test_cameras: list[Any] = []
        self.eval_cameras: list[Any] = []
        self.control_probe_cameras: list[Any] = []
        self.last_ingest_iteration: int | None = None
        self._next_iteration: int | None = None
        self._next_evaluation_optimizer_update: int | None = (
            self.interval if self._mapping_holdout_curve_enabled else None
        )
        self._last_evaluated_iteration: int | None = None
        self._last_evaluated_phase_clock: tuple[str, int] | None = None
        self._source_complete = False
        self._optimizer_updates = 0
        self._supervision_views = 0
        self._started_s = None
        self._mapping_active_s: float | None = None
        self._elapsed_s: float | None = None
        self._source_frame_index: int | None = None
        self._latest_tick = None
        self._latest_tick_wall_clock_s = None
        # State for the periodic causal_throughput.jsonl sampler (see
        # _maybe_record_causal_throughput / eligsir.runtime.causal_throughput):
        # the previously-recorded row's cumulative counters (for the
        # instantaneous/delta rate) and the next elapsed_s at which a new
        # sample is due, gated at the same cadence as the printed
        # [causal-progress] log line.
        self._throughput_prev_elapsed_s = 0.0
        self._throughput_prev_optimizer_steps = 0
        self._throughput_prev_supervision_views = 0
        self._throughput_prev_mapping_active_s = 0.0
        self._next_throughput_trace_s = 0.0
        self._partition_replay_focus = None
        self._partition_replay_focus_accumulator = None
        self._spatial_replay_focus = None
        self._partition_replay_focus_updates: dict[SourceIdentity, float] = {}
        self._spatial_replay_focus_updates: dict[SourceIdentity, float] = {}
        self._partition_control_runtime = None
        self._partition_control_roles = None
        self._partition_control_last_decision = None
        self._next_partition_coverage_optimizer_update: int | None = None
        self._partition_control_last_optimizer_update: int | None = None
        self._gpu_memory_tracker = None
        if bool(getattr(args, "streaming_memory_telemetry", True)):
            from eligsir.utils.gpu_memory_telemetry import GPUMemoryResidencyTracker

            self._gpu_memory_tracker = GPUMemoryResidencyTracker()
        self._partition_control_focus_weight = float(
            getattr(args, "streaming_partition_focus_weight", 0.0)
        )
        self._partition_control_max_promote = max(
            0, int(args.streaming_partition_focus_max_promote)
        )
        self._partition_control_outlier_k = float(
            getattr(args, "streaming_partition_focus_outlier_k", 1.5)
        )
        self._next_partition_control_optimizer_update: int | None = None
        temporal_focus = bool(getattr(args, "streaming_partition_replay_focus", False))
        spatial_focus = bool(getattr(args, "streaming_spatial_replay_focus", False))
        partition_control = self._partition_control_focus_weight > 0.0
        # #1723: the invasive actuation engine can be driven by cell-substrate
        # evidence instead of ordinal-partition evidence -- in that mode it
        # DEPENDS on the spatial weight-only controller's live telemetry, so
        # the two are no longer mutually exclusive with each other (they
        # remain mutually exclusive with the ordinal temporal_focus
        # weight-only controller, which is a different, incompatible signal
        # space -- see decide_spatial_partition_focus's docstring).
        spatial_substrate = bool(
            getattr(args, "streaming_partition_focus_spatial_substrate", False)
        )
        self._spatial_partition_control_substrate = spatial_substrate
        debt_allocator = bool(
            getattr(args, "streaming_partition_focus_debt_allocator", False)
        )
        self._partition_focus_debt_allocator = debt_allocator
        self._partition_focus_debt: dict[int, float] = {}
        if debt_allocator and not spatial_substrate:
            raise ValueError(
                "--streaming_partition_focus_debt_allocator requires "
                "--streaming_partition_focus_spatial_substrate"
            )
        if partition_control and temporal_focus:
            raise ValueError(
                "partition control cannot share replay-weight ownership"
            )
        if partition_control and spatial_focus and not spatial_substrate:
            raise ValueError(
                "partition control cannot share replay-weight ownership"
            )
        if spatial_substrate and not partition_control:
            raise ValueError(
                "--streaming_partition_focus_spatial_substrate requires "
                "--streaming_partition_focus_weight > 0"
            )
        if spatial_substrate and not spatial_focus:
            raise ValueError(
                "--streaming_partition_focus_spatial_substrate requires "
                "--streaming_spatial_replay_focus"
            )
        if partition_control:
            from eligsir.utils.partition_control import (
                ControlSignalSource,
                PartitionControlRuntime,
            )

            signal_source = ControlSignalSource(
                getattr(
                    args,
                    "streaming_partition_focus_signal_source",
                    "training_derived",
                )
            )
            if signal_source is not ControlSignalSource.CONTROL_PROBE:
                raise ValueError(
                    "canonical causal partition control requires signal_source="
                    "control_probe; training_derived and scored_holdout_oracle "
                    "remain legacy/research-only"
                )
            control_probe_n = int(
                getattr(
                    args,
                    "streaming_partition_focus_control_probe_n",
                    32,
                )
            )
            if not 1 <= control_probe_n <= 128:
                raise ValueError("control_probe_n must be in [1, 128]")
            self._partition_control_runtime = PartitionControlRuntime(
                source=signal_source,
                control_probe_n=control_probe_n,
                model_path=dataset.model_path,
                controller_enabled=True,
            )
            self._next_partition_control_optimizer_update = self.interval
        if temporal_focus:
            # issue #1552 bug 2: this controller is only ever fed through
            # _observe_tick's `if primary_psnr is not None`, and primary_psnr
            # is unconditionally None whenever streaming_track_primary_psnr
            # is off (production_application.py::_sample_primary_training_psnr
            # returns None as its very first check). Without this guard the
            # controller silently constructs and never receives a single
            # observation -- no telemetry file, no replay weight ever
            # published, `active` never leaves its initial False -- with no
            # error or warning that the flag combination is a no-op.
            if not bool(getattr(args, "streaming_track_primary_psnr", False)):
                raise ValueError(
                    "--streaming_partition_replay_focus requires "
                    "--streaming_track_primary_psnr: the controller is only "
                    "ever fed primary training PSNR observations, so without "
                    "that flag it would construct and silently never receive "
                    "a single observation."
                )
            from eligsir.mapping.scheduler.partition_replay_focus import (
                PartitionReplayFocusController,
                TrainingPartitionPsnrAccumulator,
            )
            from eligsir.utils.partition_eval import PARTITIONS, registry_n_source_frames

            n_source_frames = int(
                source_total
                or registry_n_source_frames(str(dataset.source_path))
                or 0
            )
            if n_source_frames < 2:
                raise ValueError(
                    "partition replay focus requires a fixed finite source denominator"
                )

            self._partition_replay_focus = PartitionReplayFocusController(
                min_history=int(args.streaming_partition_replay_focus_min_history),
                min_views_per_partition=int(
                    args.streaming_partition_replay_focus_min_views
                ),
                low_psnr_gain=float(
                    args.streaming_partition_replay_focus_low_gain
                ),
                regression_gain=float(
                    args.streaming_partition_replay_focus_regression_gain
                ),
                regression_tolerance_db=float(
                    args.streaming_partition_replay_focus_regression_tolerance_db
                ),
                max_weight=float(
                    args.streaming_partition_replay_focus_max_weight
                ),
            )
            self._partition_replay_focus_accumulator = (
                TrainingPartitionPsnrAccumulator(
                    n_source_frames=n_source_frames,
                    window_per_partition=16,
                    # Issue #1552: bucket into the SAME fixed 6-way scheme
                    # report.json's partition_psnr ("p0".."p5") uses, not the
                    # trajectory-length-scaled utils.partition_eval.partition_
                    # count() granularity (16-way on a 2000-frame Replica
                    # scene) -- otherwise the controller's weight elevations
                    # can never be validated against, or even correspond to,
                    # the partitions the report/holdout accounting judges.
                    partition_count=PARTITIONS,
                )
            )
            # Successful-source identities are already available on every
            # completed B4. Refresh the CPU-only coverage debt often enough
            # to prevent late trajectory regions from staying invisible, but
            # not on every transaction of the GPU submitter's hot path.
            self._next_partition_coverage_optimizer_update = 1
        if spatial_focus:
            from eligsir.mapping.scheduler.partition_replay_focus import (
                SpatialMicrocellReplayFocusController,
            )

            self._spatial_replay_focus = SpatialMicrocellReplayFocusController(
                min_history=int(args.streaming_partition_replay_focus_min_history),
                minimum_cell_samples=64,
                low_psnr_gain=float(args.streaming_partition_replay_focus_low_gain),
                regression_gain=float(
                    args.streaming_partition_replay_focus_regression_gain
                ),
                regression_tolerance_db=float(
                    args.streaming_partition_replay_focus_regression_tolerance_db
                ),
                max_weight=float(args.streaming_partition_replay_focus_max_weight),
                cell_debt_actuation=bool(
                    getattr(
                        args,
                        "streaming_spatial_replay_focus_cell_debt_actuation",
                        False,
                    )
                ),
            )

    def _ensure_cameras(self) -> None:
        if not self.runtime.initialized or self.runtime.scene is None:
            raise RuntimeError("causal holdout evaluation requires an initialized map")
        device = self.runtime.gaussians.get_xyz.device
        if self.plan.holdout_ordinals and not self.test_cameras:
            self.test_cameras = build_causal_holdout_cameras(
                self.plan, data_device=device
            )
            if not self.test_cameras:
                raise RuntimeError("causal holdout rule realized no pose-valid test cameras")
            self.eval_cameras = bounded_evaluation_cameras(
                self.test_cameras,
                max_items=int(
                    getattr(self.args, "streaming_report_test_metrics_max", 128)
                ),
            )
            if not self.eval_cameras:
                raise RuntimeError(
                    "causal holdout evaluation requires "
                    "streaming_report_test_metrics_max > 0"
                )
            self.runtime.scene.bind_causal_evaluation(
                self.test_cameras, dict(self.plan.eval_metadata)
            )
        if (
            self._partition_control_runtime is not None
            and self.plan.control_probe_records
            and not self.control_probe_cameras
        ):
            self.control_probe_cameras = build_causal_holdout_cameras(
                self.plan.control_probe_records, data_device=device
            )
            if not self.control_probe_cameras:
                raise RuntimeError("partition control realized no pose-valid probe cameras")

    def _report_training_cameras(self) -> list[Any]:
        """Rebuild a bounded report-only sample of sources that actually mapped."""

        max_items = int(getattr(self.args, "streaming_report_trajectory_max_frames", 120))
        if max_items <= 0:
            return []
        provider = getattr(self.runtime, "successful_supervision_uses_by_identity", None)
        uses = {} if not callable(provider) else provider()
        records = prepared_report_source_identity_records(
            self._source_records,
            successful_supervision_uses=uses,
            max_items=max_items,
            packet_source_id=self._packet_source_id,
            packet_stream_epoch=self._packet_stream_epoch,
        )
        if not records:
            if self._source_records and callable(provider):
                if any(int(count) > 0 for count in uses.values()):
                    raise RuntimeError(
                        "report source identity does not match supervised source ledger"
                    )
                return []
            return list(self.runtime.scene.train_cameras)
        return build_causal_holdout_cameras(
            records,
            data_device=self.runtime.gaussians.get_xyz.device,
            report_source_id=self._packet_source_id,
            report_stream_epoch=self._packet_stream_epoch,
        )

    def __call__(self, tick: Any) -> None:
        import time

        if self._started_s is None:
            self._started_s = time.perf_counter()
        cumulative_optimizer_updates = getattr(
            tick, "cumulative_optimizer_updates", None
        )
        self._optimizer_updates = (
            self._optimizer_updates + int(getattr(tick, "optimizer_steps", 0))
            if cumulative_optimizer_updates is None
            else int(cumulative_optimizer_updates)
        )
        cumulative_supervision_views = getattr(
            tick, "cumulative_supervision_views", None
        )
        self._supervision_views = (
            self._supervision_views + int(getattr(tick, "supervision_views", 0))
            if cumulative_supervision_views is None
            else int(cumulative_supervision_views)
        )
        self._latest_tick = tick
        self._latest_tick_wall_clock_s = time.perf_counter() - self._started_s
        self._elapsed_s = self._latest_tick_wall_clock_s
        self._observe_tick(tick)

    def observe_tick(self, tick: Any, clock: CausalReplayClock) -> None:
        """Observe a tick with the driver's evaluation-excluding work clocks."""

        self._optimizer_updates = int(clock.optimizer_steps)
        self._supervision_views = int(clock.supervision_views)
        self._mapping_active_s = float(clock.mapping_active_s)
        self._elapsed_s = float(clock.elapsed_s)
        self._source_frame_index = clock.source_frame_index
        self._latest_tick = tick
        self._latest_tick_wall_clock_s = self._elapsed_s
        self._maybe_record_causal_throughput(
            elapsed_s=self._elapsed_s,
            optimizer_steps=self._optimizer_updates,
            supervision_views=self._supervision_views,
            mapping_active_s=self._mapping_active_s,
        )
        self._observe_tick(tick)

    def _maybe_record_causal_throughput(
        self,
        *,
        elapsed_s: float,
        optimizer_steps: int,
        supervision_views: int,
        mapping_active_s: float,
    ) -> None:
        """Append one row to causal_throughput.jsonl, gated to a fixed cadence.

        Every real tick would be far too dense to persist over a multi-hour
        run, so this samples at the same wall-clock cadence as the printed
        ``[causal-progress]`` log line (``--causal_progress_interval_s``,
        default 10s) -- reusing the same knob keeps the printed line and the
        plotted trace describing comparable granularity. Uses ``getattr``
        defaults throughout because several unit tests construct this
        observer via ``object.__new__`` and set only the attributes their
        specific assertion needs (see tests/test_causal_evaluation.py) --
        this must not require ``__init__`` to have run.
        """

        interval_s = float(
            getattr(getattr(self, "args", None), "causal_progress_interval_s", 10.0)
            or 10.0
        )
        next_due = getattr(self, "_next_throughput_trace_s", 0.0)
        if elapsed_s < next_due:
            return
        record_causal_throughput(
            self.dataset.model_path,
            elapsed_s=elapsed_s,
            optimizer_steps=optimizer_steps,
            supervision_views=supervision_views,
            mapping_active_s=mapping_active_s,
            prev_elapsed_s=getattr(self, "_throughput_prev_elapsed_s", 0.0),
            prev_optimizer_steps=getattr(self, "_throughput_prev_optimizer_steps", 0),
            prev_supervision_views=getattr(
                self, "_throughput_prev_supervision_views", 0
            ),
            prev_mapping_active_s=getattr(
                self, "_throughput_prev_mapping_active_s", 0.0
            ),
        )
        self._throughput_prev_elapsed_s = elapsed_s
        self._throughput_prev_optimizer_steps = optimizer_steps
        self._throughput_prev_supervision_views = supervision_views
        self._throughput_prev_mapping_active_s = mapping_active_s
        self._next_throughput_trace_s = elapsed_s + interval_s

    def _observe_tick(self, tick: Any) -> None:
        primary_psnr = getattr(tick, "primary_training_psnr_db", None)
        if primary_psnr is not None:
            append_primary_training_psnr(
                self.dataset.model_path,
                iteration=int(getattr(tick, "iteration", 0)),
                psnr_db=float(primary_psnr),
                source_id=str(
                    getattr(tick, "primary_training_source_id", "") or "unknown"
                ),
                mapping_active_s=self._mapping_active_s,
                supervision_views=self._supervision_views,
                evaluation_phase="refinement" if self._source_complete else "mapping",
                resolution_tiers=_tick_resolution_tiers(tick),
                rasterize_mode=str(
                    getattr(
                        getattr(getattr(self, "runtime", None), "pipe", None),
                        "rasterize_mode",
                        "classic",
                    )
                ),
            )
        # Both replay-focus controllers consume only already-produced training
        # supervision.  Temporal partition focus is valid as soon as its
        # finite source denominator has been established at construction; it
        # must accumulate evidence during mapping rather than spend most of a
        # bounded refinement tail merely warming up.
        replay_focus_updated = False
        coverage_due = getattr(
            self, "_next_partition_coverage_optimizer_update", None
        )
        completed_sources = tuple(
            getattr(
                tick,
                "completed_supervision_sources",
                getattr(tick, "successful_supervision_sources", ()),
            )
            or ()
        )
        if (
            completed_sources
            and coverage_due is not None
            and self._optimizer_updates >= coverage_due
        ):
            observed = self._observe_completed_partition_coverage(completed_sources)
            if observed is not None:
                updates, sources = observed
                self._replace_replay_focus_component(
                    "_partition_replay_focus_updates", updates, sources
                )
                replay_focus_updated = True
            self._next_partition_coverage_optimizer_update = (
                int(self._optimizer_updates) + 16
            )
        if (
            primary_psnr is not None
            and getattr(self, "_partition_replay_focus_accumulator", None)
            is not None
        ):
            observed = self._observe_training_partition_replay_focus(tick)
            if observed is not None:
                updates, sources = observed
                self._replace_replay_focus_component(
                    "_partition_replay_focus_updates", updates, sources
                )
                replay_focus_updated = True
        if getattr(tick, "primary_training_spatial", None) is not None:
            observed = self._observe_training_spatial_replay_focus(tick)
            if observed is not None:
                updates, sources = observed
                self._replace_replay_focus_component(
                    "_spatial_replay_focus_updates", updates, sources
                )
                replay_focus_updated = True
        if replay_focus_updated:
            self._publish_composed_replay_focus_weights()
        partition_due = getattr(
            self, "_next_partition_control_optimizer_update", None
        )
        if (
            self._source_complete
            and partition_due is not None
            and self._optimizer_updates >= partition_due
        ):
            if self._observe_partition_control(tick):
                # One bounded opportunity per barrier. Never catch up missed
                # controller events in a burst after a large work-clock jump.
                self._next_partition_control_optimizer_update = (
                    int(self._optimizer_updates) + self.interval
                )
        next_update = getattr(
            self, "_next_evaluation_optimizer_update", None
        )
        if not self._source_complete:
            if (
                bool(getattr(self, "_mapping_holdout_curve_enabled", False))
                and next_update is not None
                and self._optimizer_updates >= next_update
            ):
                self._ensure_cameras()
                self._evaluate(tick, evaluation_phase="mapping")
            return
        if next_update is not None and self._optimizer_updates >= next_update:
            self._evaluate(tick, evaluation_phase="refinement")

    def source_complete(self, tick: Any) -> None:
        # Any still-in-flight async mapping-phase evaluate (issue #1314) must
        # finish writing psnr_history/eval-tick artifacts before the
        # synchronous mapping_endpoint evaluate below writes to the same
        # files on this (the mapping) thread.
        self.drain_async_holdout_evaluator()
        self._source_complete = True
        self.last_ingest_iteration = int(getattr(tick, "iteration", 0))
        partition_control = getattr(self, "_partition_control_runtime", None)
        if (
            partition_control is not None
            and getattr(self, "_partition_control_last_optimizer_update", None)
            != int(self._optimizer_updates)
        ):
            self._observe_partition_control(tick)
        plan = getattr(self, "plan", None)
        if plan is None or plan.holdout_ordinals:
            self._ensure_cameras()
            self._evaluate(tick, evaluation_phase="mapping_endpoint")

    def _measure_partition_control(
        self, tick: Any
    ) -> _PartitionControlMeasurement | None:
        """Measure one bounded non-oracle weak-region control decision.

        Measurement deliberately has no replay, recorder, or observer-state
        side effects.  A speculative refinement branch publishes it only when
        its candidate map is accepted.
        """

        control = self._partition_control_runtime
        if control is None or not bool(getattr(self.runtime, "initialized", False)):
            return None
        if (
            (self.plan.holdout_ordinals and not self.test_cameras)
            or (self.plan.control_probe_records and not self.control_probe_cameras)
        ):
            self._ensure_cameras()
        training_cameras = tuple(self.control_probe_cameras)
        if not training_cameras:
            raise RuntimeError(
                "canonical partition control requires a source-reserved probe cohort"
            )
        scored_holdout_cameras = tuple(self.test_cameras)
        def render_control_tick(cameras):
            return _render_partition_control_tick(self.runtime, cameras)

        roles, control_cameras = control.prepare(
            training_cameras,
            scored_holdout_cameras,
            publish=False,
        )
        spatial_substrate = bool(
            getattr(self, "_spatial_partition_control_substrate", False)
        )
        if spatial_substrate:
            # #1723: cell-substrate actuation reads its PSNR signal from the
            # spatial weight-only controller's own live telemetry, not from a
            # rendered control-probe tick -- skip that render entirely.
            # control_tick stays None: it is read again below (purely for
            # optional training_resolution_tiers/evaluation_resolutions
            # telemetry) regardless of which branch ran.
            control_tick = None
            observations = {"training_derived": {}, "control_probe": {}, "scored_holdout": {}}
        else:
            control_tick = (
                None
                if roles.oracle_control_enabled
                else render_control_tick(control_cameras)
            )
            observations = {
                "training_derived": {},
                "control_probe": control.observations_from_tick(control_tick),
                "scored_holdout": {},
            }
            if roles.source.value == "training_derived":
                observations["training_derived"] = observations["control_probe"]
                observations["control_probe"] = {}

        candidate_provider = getattr(
            self.runtime, "replay_control_candidate_identities", None
        )
        if not callable(candidate_provider):
            raise RuntimeError(
                "canonical partition control requires a bounded replay candidate seam"
            )
        candidates = tuple(candidate_provider())
        if len(candidates) > 128:
            raise RuntimeError(
                "canonical partition control candidate seam exceeded 128 identities"
            )
        if not all(isinstance(source, SourceIdentity) for source in candidates):
            raise TypeError(
                "canonical partition control candidates must be SourceIdentity values"
            )
        candidate_hash = _replay_candidate_shortlist_identity_hash(candidates)
        sources_by_ordinal = {
            int(source.source_ordinal): source for source in candidates
        }
        for camera in control_cameras:
            source = getattr(camera, "_source_identity", None)
            if isinstance(source, SourceIdentity):
                sources_by_ordinal.setdefault(int(source.source_ordinal), source)

        from eligsir.utils.partition_control import decide_partition_focus
        from eligsir.utils.partition_eval import PARTITIONS, partition_id

        population_provider = getattr(
            self.runtime, "replay_catalog_population", None
        )
        replay_population = (
            int(population_provider())
            if callable(population_provider)
            else len(candidates)
        )
        known_ordinals = tuple(sources_by_ordinal)
        max_source_index = (
            int(self.source_total) - 1
            if self.source_total is not None and int(self.source_total) > 1
            else max(known_ordinals, default=0)
        )
        # The control signal and public quality contract must have identical
        # temporal regions.  Otherwise replay can improve a private finer
        # bucket while the reported worst partition remains neglected.
        trajectory_partition_count = PARTITIONS
        probe_ordinals = tuple(int(index) for index in roles.control_probe_indices)
        probe_partition_ids = tuple(
            sorted(
                {
                    partition_id(
                        index,
                        max_source_index,
                        k=trajectory_partition_count,
                    )
                    for index in probe_ordinals
                }
            )
        )
        source_watermark = max(roles.training_indices, default=-1)
        active_indices = set(roles.training_indices)
        bounded_active_indices = tuple(
            sorted(index for index in sources_by_ordinal if index in active_indices)
        )
        cached_indices = tuple(
            sorted(
                int(source.source_ordinal)
                for source in candidates
                if int(source.source_ordinal) not in active_indices
            )
        )
        # Issue #1308's second wire: without this, TARGETED_HISTORY repair
        # ranking's prior_usefulness stays permanently 0.0 (#953's T0
        # preflight finding) even once a real per-view loss producer is
        # wired -- this is the sole production call site of
        # decide_partition_focus, so it is the only place that gap can be
        # closed. Returns {} (byte-identical to the prior implicit None)
        # unless --streaming_historical_usefulness_tracking is enabled.
        historical_usefulness_export = getattr(
            self.runtime, "historical_usefulness_by_source_ordinal", None
        )
        historical_usefulness = (
            historical_usefulness_export() if historical_usefulness_export else None
        )
        prior_decision = self._partition_control_last_decision
        if spatial_substrate:
            # #1723: same actuation shape (PartitionFocusDecision), driven by
            # world-microcell coverage instead of ordinal-partition means.
            spatial_controller = self._spatial_replay_focus
            if spatial_controller is None:
                raise RuntimeError(
                    "spatial partition-control substrate requires an active "
                    "spatial replay focus controller"
                )
            from eligsir.utils.partition_control import decide_spatial_partition_focus

            cell_psnr, cell_view_counts = spatial_controller.current_cell_quality()
            # #1723 follow-up (2026-09-03): active_source_cell_keys() only
            # covers sources the spatial controller has actually sampled as
            # its primary-training view (one per streaming_spatial_replay_
            # focus_interval), which barely overlaps the bounded ≤128
            # actuation candidate pool in practice -- a real GPU run showed
            # priority_weights carrying ~1 active entry against ~8 promoted
            # cached entries per tick, and demote structurally could never
            # fire as a result (regardless of how many cells were flagged
            # bad). Cell MEMBERSHIP never needed a live sample in the first
            # place (geometry-only, see covered_cell_keys_for_sources's
            # docstring) -- resolve active and cached region-ids through the
            # SAME geometry-only path instead of relying on the sparse
            # sampled-history one, so coverage is complete for the whole
            # actuation candidate pool every tick, not just whichever
            # sources happened to be recently primary-sampled.
            all_candidate_sources = tuple(
                sources_by_ordinal[index]
                for index in (*bounded_active_indices, *cached_indices)
            )
            covered_cells_provider = getattr(
                self.runtime, "covered_cell_keys_for_sources", None
            )
            candidate_cell_keys = (
                covered_cells_provider(all_candidate_sources)
                if callable(covered_cells_provider)
                else {}
            )
            active_region_ids = {
                index: candidate_cell_keys[sources_by_ordinal[index]]
                for index in bounded_active_indices
                if sources_by_ordinal[index] in candidate_cell_keys
            }
            cached_region_ids = {
                index: candidate_cell_keys[sources_by_ordinal[index]]
                for index in cached_indices
                if sources_by_ordinal[index] in candidate_cell_keys
            }
            if getattr(self, "_partition_focus_debt_allocator", False):
                decision, self._partition_focus_debt = (
                    _debt_allocator_partition_focus_decision(
                        source=roles.source,
                        cell_psnr=cell_psnr,
                        cell_view_counts=cell_view_counts,
                        active_region_ids=active_region_ids,
                        cached_region_ids=cached_region_ids,
                        debt=self._partition_focus_debt,
                        batch_size=min(
                            self._partition_control_max_promote,
                            MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
                        ),
                    )
                )
            else:
                decision = decide_spatial_partition_focus(
                    roles=roles,
                    cell_psnr=cell_psnr,
                    cell_view_counts=cell_view_counts,
                    outlier_k=self._partition_control_outlier_k,
                    active_region_ids=active_region_ids,
                    cached_region_ids=cached_region_ids,
                    max_promote=min(
                        self._partition_control_max_promote,
                        MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
                    ),
                    historical_usefulness=historical_usefulness or None,
                )
        else:
            decision = decide_partition_focus(
                roles=roles,
                training_observations=observations["training_derived"],
                control_probe_observations=observations["control_probe"],
                scored_holdout_observations={},
                max_source_index=max_source_index,
                partition_count=trajectory_partition_count,
                outlier_k=self._partition_control_outlier_k,
                # Canonical mapping does not mutate an active camera list here.
                # Probes and bounded cold targets become scheduler evidence only.
                active_training_indices=bounded_active_indices,
                cached_indices=cached_indices,
                max_promote=min(
                    self._partition_control_max_promote,
                    MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
                ),
                historical_usefulness=historical_usefulness or None,
                # Hysteresis retention against bad/good flip-flop -- see
                # decide_partition_focus's own docstring comment. Feeds the
                # decision this same observer already stores every tick
                # (self._partition_control_last_decision, set below) back in
                # as the next tick's prior state; a fresh observer has none.
                prior_bad_partitions=(
                    frozenset(prior_decision.bad_partitions)
                    if prior_decision is not None
                    else None
                ),
            )

        raw_priority = _partition_focus_publication_priorities(decision)
        promote_first = tuple(decision.promote_indices)
        promote_set = set(promote_first)
        remaining = tuple(
            index
            for index, _weight in sorted(
                raw_priority.items(), key=lambda item: (-item[1], item[0])
            )
            if index not in promote_set
        )
        selected_indices = tuple(
            index
            for index in (*promote_first, *remaining)
            if index in sources_by_ordinal
        )[:MAX_REPLAY_IDENTITY_FOCUS_UPDATES]
        (
            published_by_ordinal,
            focus_probability_requested,
            focus_probability_resolved,
        ) = _partition_focus_publication_weights(
            raw_priority,
            selected_indices,
            # Issue #1723 wiring fix: the sampler draws over the whole replay
            # catalog, so the focus probability must be realized against it.
            # Scaling against the bounded 128-identity control shortlist made a
            # requested p=0.5 realize as ~6% over a 2210-frame catalog.
            candidate_population=replay_population,
            focus_probability=self._partition_control_focus_weight,
        )
        published = {
            sources_by_ordinal[index]: weight
            for index, weight in published_by_ordinal.items()
        }

        from dataclasses import replace

        realized_decision = replace(
            decision,
            priority_weights=published_by_ordinal,
            promote_indices=tuple(
                index for index in promote_first if index in published_by_ordinal
            ),
        )
        control_tiers = tuple(
            str(value)
            for value in getattr(
                control_tick, "training_resolution_tiers", ()
            )
        )
        control_resolutions = tuple(
            tuple(int(value) for value in resolution)
            for resolution in getattr(
                control_tick, "evaluation_resolutions", ()
            )
        )
        return _PartitionControlMeasurement(
            roles=roles,
            decision=realized_decision,
            published=MappingProxyType(published),
            telemetry=MappingProxyType(
                {
                "replay_catalog_population": replay_population,
                "control_source_watermark": source_watermark,
                "replay_control_candidate_population": len(candidates),
                "replay_control_candidate_identity_hash": candidate_hash,
                "focus_publication_n": len(published),
                "focus_probability_requested": focus_probability_requested,
                "focus_probability_resolved": focus_probability_resolved,
                "focus_weight_min": min(published.values(), default=1.0),
                "focus_weight_max": max(published.values(), default=1.0),
                "score_scope": "control_probe_same_camera_grid_rgb_psnr",
                "score_signal_source": "control_probe",
                "control_probe_source_ordinal_min": min(
                    probe_ordinals, default=None
                ),
                "control_probe_source_ordinal_max": max(
                    probe_ordinals, default=None
                ),
                "control_probe_temporal_partition_ids": list(
                    probe_partition_ids
                ),
                "control_probe_temporal_partition_count": len(
                    probe_partition_ids
                ),
                "control_probe_trajectory_partition_count": (
                    trajectory_partition_count
                ),
                "control_resolution_tiers": list(control_tiers),
                "control_evaluation_resolutions": [
                    list(resolution) for resolution in control_resolutions
                ],
                "promotion_semantics": (
                    "scheduler_focus_membership_not_immediate_residency"
                ),
                }
            ),
        )

    def _publish_partition_control(
        self, tick: Any, measurement: _PartitionControlMeasurement
    ) -> None:
        """Publish a previously measured control decision after map acceptance."""

        control = self._partition_control_runtime
        if control is None:
            raise RuntimeError("partition control publication requires a control runtime")
        # Issue #1723 wiring fix: publish through the same composition the
        # per-tick weight-only controllers use. A direct
        # publish_replay_focus_weights() call here was reset to neutral by the
        # spatial controller's next composed republish (~7 iterations later),
        # leaving this controller's actuation a ~0.7% duty cycle.
        self._partition_control_focus_updates = dict(measurement.published)
        self._publish_composed_replay_focus_weights()
        control.recorder.record(
            iteration=int(getattr(tick, "iteration", 0)),
            roles=measurement.roles,
            decision=measurement.decision,
            telemetry=measurement.telemetry,
        )
        self._partition_control_roles = measurement.roles
        self._partition_control_last_decision = measurement.decision
        self._partition_control_last_optimizer_update = int(self._optimizer_updates)

    def _observe_partition_control(self, tick: Any) -> bool:
        """Measure then publish one accepted weak-region replay-focus update."""

        measurement = self._measure_partition_control(tick)
        if measurement is None:
            return False
        self._publish_partition_control(tick, measurement)
        return True

    def _evaluate(self, tick: Any, *, evaluation_phase: str | None = None) -> None:
        iteration = int(getattr(tick, "iteration", 0))
        source_complete = bool(getattr(self, "_source_complete", False))
        if evaluation_phase is None:
            evaluation_phase = "refinement" if source_complete else "mapping"
        phase_clock = (str(evaluation_phase), int(self._optimizer_updates))
        if phase_clock == getattr(self, "_last_evaluated_phase_clock", None):
            return
        curve_metadata = getattr(
            self, "_mapping_holdout_curve_metadata", None
        )
        if curve_metadata is None:
            curve_metadata = mapping_holdout_curve_metadata(
                getattr(self, "args", None)
            )
        # Scheduling/bookkeeping advances immediately regardless of whether
        # the actual render+score happens inline (sync) or on a background
        # thread (async, issue #1314) -- the async worker's completion time
        # must never gate when the NEXT eval tick is allowed to fire.
        self._last_evaluated_iteration = iteration
        self._last_evaluated_phase_clock = phase_clock
        self._next_iteration = iteration + self.interval
        self._next_evaluation_optimizer_update = (
            int(self._optimizer_updates) + self.interval
        )
        # Capture every mutable scalar this evaluate needs BY VALUE, on the
        # calling (mapping) thread, right now -- never read `self.*`/
        # `self.runtime.*` later from `_evaluate_sync` itself, because on the
        # async path that method runs on the background evaluator thread,
        # arbitrarily far in wall-clock/iteration-count terms from this
        # instant (issue #1314 review). Reading them lazily there would
        # silently mislabel the resulting PSNR row with whatever
        # optimizer_updates/mapping_active_s/etc. the mapping thread had
        # reached by the time the render finished, not what was true when
        # the snapshot was actually taken.
        import time as _time

        _elapsed_s = getattr(self, "_elapsed_s", None)
        context = {
            "optimizer_updates": int(self._optimizer_updates),
            "supervision_views": int(self._supervision_views),
            "mapping_active_s": (
                None
                if getattr(self, "_mapping_active_s", None) is None
                else float(self._mapping_active_s)
            ),
            "elapsed_s": (
                _elapsed_s
                if _elapsed_s is not None
                else (
                    None
                    if self._started_s is None
                    else _time.perf_counter() - self._started_s
                )
            ),
            "source_frame_index": getattr(self, "_source_frame_index", None),
            "frames_seen": int(self.runtime.frames_seen),
            "frames_trained": _completed_training_source_count(self.runtime),
            "keyframes_admitted": int(self.runtime._admitted_observations_total),
            "last_ingest_iteration": self.last_ingest_iteration,
        }
        evaluator = getattr(self, "_async_holdout_evaluator", None)
        if (
            evaluation_phase == "mapping"
            and bool(curve_metadata.get("async_mode", False))
            and evaluator is not None
        ):
            live_gaussians = self.runtime.gaussians

            def _do_eval(snapshot: Any) -> None:
                self._evaluate_sync(
                    tick=tick,
                    iteration=iteration,
                    evaluation_phase=evaluation_phase,
                    curve_metadata=curve_metadata,
                    gaussians=snapshot,
                    context=context,
                )

            evaluator.submit(gaussians=live_gaussians, evaluate_fn=_do_eval)
            return
        self._evaluate_sync(
            tick=tick,
            iteration=iteration,
            evaluation_phase=evaluation_phase,
            curve_metadata=curve_metadata,
            gaussians=self.runtime.gaussians,
            context=context,
        )

    def _evaluate_sync(
        self,
        *,
        tick: Any,
        iteration: int,
        evaluation_phase: str,
        curve_metadata: Mapping[str, Any],
        gaussians: "Any | GaussianRenderSnapshot",
        context: Mapping[str, Any],
    ) -> None:
        """Render + score the holdout against ``gaussians`` and record it.

        ``gaussians`` is either the live model (default/synchronous path, or
        any non-mapping phase) or a frozen
        :class:`~eligsir.runtime.async_holdout_eval.GaussianRenderSnapshot`
        (async mapping-phase path, issue #1314). Runs inline on whichever
        thread calls it -- the mapping thread for the synchronous path, the
        background evaluator thread for the async path. ``context`` carries
        every mutable-state scalar this needs, captured BY VALUE on the
        mapping thread at the instant the eval tick actually fired -- never
        re-read from ``self``/``self.runtime`` here, which could be
        arbitrarily stale (or simply wrong) by the time the async path's
        background thread executes this.
        """
        from gaussian_renderer import render
        from eligsir.runtime.profiling.events import GpuSpanRegistry, gpu_timed_span
        from eligsir.utils.eval_tick import record_eval_tick
        from eligsir.utils.psnr_history import record_and_plot

        resolution_tiers = _tick_resolution_tiers(tick)
        xyz = getattr(gaussians, "get_xyz", None)
        gaussian_count = (
            int(xyz.shape[0])
            if xyz is not None
            else int(
                getattr(getattr(tick, "control_snapshot", None), "gaussian_count", 0)
            )
        )
        metric_context = {
            "training_resolution_tiers": resolution_tiers,
            "optimizer_updates": int(context["optimizer_updates"]),
            "supervision_views": int(context["supervision_views"]),
            "mapping_active_s": context["mapping_active_s"],
            "gaussian_count": gaussian_count,
            # The canonical runtime currently measures the complete GPU
            # optimization envelope. Do not mislabel that as raster-only time.
            "rasterizer_gpu_ms": None,
            "rasterizer_timing_status": "unavailable_stage_gpu_events",
            "evaluation_phase": str(evaluation_phase),
            "evaluation_observer_execution": (
                str(curve_metadata["observer_execution"])
                if evaluation_phase == "mapping"
                else "post_source_completion"
            ),
            "mapping_execution_paused": (
                bool(curve_metadata["mapping_execution_paused"])
                if evaluation_phase == "mapping"
                else False
            ),
            "throughput_evidence_valid": bool(
                curve_metadata["throughput_evidence_valid"]
            ),
            "throughput_evidence_status": str(
                curve_metadata["throughput_evidence_status"]
            ),
            "evaluation_cadence_clock": "cumulative_optimizer_updates",
        }
        rasterizer_timing = GpuSpanRegistry()

        def _render_holdout(*args: Any, **kwargs: Any) -> Any:
            # This wraps only the renderer already required for the score.
            # Events are polled after the complete tick, never synchronized per render.
            with gpu_timed_span("causal_holdout.rasterizer", rasterizer_timing):
                return render(*args, **kwargs)

        def _resolve_metric_context(_context: Any) -> dict[str, Any]:
            return {**metric_context, **_holdout_rasterizer_context(rasterizer_timing)}

        psnr_tick = record_and_plot(
            str(self.dataset.model_path),
            iteration,
            gaussians,
            self.eval_cameras,
            _render_holdout,
            self.runtime.pipe,
            self.runtime._background,
            metric_context=metric_context,
            metric_context_resolver=_resolve_metric_context,
        )
        evaluation_record = record_eval_tick(
            model_path=str(self.dataset.model_path),
            iteration=iteration,
            psnr_tick=psnr_tick,
            test_cams=self.eval_cameras,
            scene_source=_scene_source_for_report(self.dataset),
            source_frame_index=context["source_frame_index"],
            frames_ingested=int(context["frames_seen"]),
            frames_trained=int(context["frames_trained"]),
            keyframes_admitted=int(context["keyframes_admitted"]),
            optimizer_step=context["optimizer_updates"],
            supervision_views=context["supervision_views"],
            mapping_snapshot_id=getattr(tick, "snapshot_id", None),
            control_tick_epoch=getattr(
                getattr(tick, "control_snapshot", None),
                "control_tick_epoch",
                None,
            ),
            supervision_constellation=_tick_constellation_telemetry(tick),
            wall_clock_s=context["elapsed_s"],
            accumulated_iteration_wall_s=context["mapping_active_s"],
            last_ingest_iter=context["last_ingest_iteration"],
            partition_k=None,
            evaluation_phase=str(evaluation_phase),
            evaluation_observer_execution=str(
                metric_context["evaluation_observer_execution"]
            ),
            mapping_execution_paused=bool(
                metric_context["mapping_execution_paused"]
            ),
            throughput_evidence_valid=bool(
                metric_context["throughput_evidence_valid"]
            ),
            throughput_evidence_status=str(
                metric_context["throughput_evidence_status"]
            ),
            evaluation_cadence_clock=str(
                metric_context["evaluation_cadence_clock"]
            ),
            plot=True,
        )
        plot_paired_psnr(
            self.dataset.model_path,
            last_ingest_iteration=self.last_ingest_iteration,
        )

    def wait_for_pending_snapshot_copy(self) -> None:
        """Choke point every in-place Gaussian mutation must pass through.

        No-op unless async mapping-holdout evaluation (issue #1314) is
        active. Wired into the causal-mapping execution path by wrapping
        ``binding.optimizer_tick`` in
        :func:`eligsir.runtime.causal_entry.run_causal_mapping`.
        """

        evaluator = getattr(self, "_async_holdout_evaluator", None)
        if evaluator is not None:
            evaluator.wait_for_pending_copy()

    def drain_async_holdout_evaluator(self) -> None:
        """Block until every already-submitted async evaluate has completed.

        Must run before any synchronous evaluation/report step that touches
        the same on-disk artifacts (psnr_history, eval ticks, plots) an
        in-flight async task might still be writing to -- called from
        :meth:`source_complete` and :meth:`finalize`.
        """

        evaluator = getattr(self, "_async_holdout_evaluator", None)
        if evaluator is not None:
            evaluator.drain()

    def _resolve_async_throughput_evidence(self) -> None:
        """Overwrite the provisional async throughput verdict with measured data.

        ``mapping_holdout_curve_metadata`` cannot know the real
        snapshot-copy stall before any tick has run, so it marks async mode
        ``pending_measured_async_snapshot_stall``. Once the evaluator has
        drained, replace that placeholder with the actual measured verdict
        -- never a blanket True or False (issue #1314 acceptance criterion).

        Note this verdict is written into ``self._mapping_holdout_curve_metadata``
        (and from there into the FINAL report's ``mapping_holdout_curve``
        block) only -- it is resolved after every mapping-phase tick has
        already run and recorded its own per-row
        ``throughput_evidence_valid``/``_status`` (via ``eval_tick`` /
        ``psnr_history``, at ``pending_measured_async_snapshot_stall`` /
        ``True``, since the real measurement doesn't exist yet at tick
        time). If the measured stall later comes out significant, this
        report-level verdict is the AUTHORITATIVE one; per-row values from
        during mapping stay at their tick-time placeholder and must not be
        read as a second, independent confirmation of validity.
        """

        evaluator = getattr(self, "_async_holdout_evaluator", None)
        metadata = getattr(self, "_mapping_holdout_curve_metadata", None)
        if (
            evaluator is None
            or metadata is None
            or not bool(metadata.get("async_mode", False))
        ):
            return
        stats = evaluator.stats()
        metadata["async_snapshot_evaluator_stats"] = dict(stats)
        threshold_ms = float(
            metadata.get("async_snapshot_copy_stall_threshold_ms", 5.0)
        )
        samples = int(stats.get("mapping_thread_stall_samples", 0))
        if samples == 0:
            metadata["throughput_evidence_valid"] = True
            metadata["throughput_evidence_status"] = (
                "valid_async_snapshot_stream_no_mapping_phase_evaluations"
            )
            return
        stall_max_ms = float(stats.get("mapping_thread_stall_ms_max", 0.0))
        negligible = stall_max_ms <= threshold_ms
        metadata["throughput_evidence_valid"] = bool(negligible)
        metadata["throughput_evidence_status"] = (
            "valid_async_snapshot_stream_measured_negligible_stall"
            if negligible
            else "invalid_async_snapshot_stream_measured_significant_stall"
        )

    def _publish_composed_replay_focus_weights(self) -> None:
        """Publish one conservative focus replacement for both controllers."""

        self.runtime.publish_replay_focus_weights(
            _compose_replay_identity_focus_updates(
                getattr(self, "_partition_replay_focus_updates", {}),
                getattr(self, "_spatial_replay_focus_updates", {}),
                getattr(self, "_partition_control_focus_updates", {}),
            )
        )

    def _replace_replay_focus_component(
        self,
        attribute: str,
        updates: dict[SourceIdentity, float],
        sources: tuple[SourceIdentity, ...],
    ) -> None:
        """Refresh one controller result and discard identities retired since it ran."""

        eligible = frozenset(sources)
        for cached_attribute in (
            "_partition_replay_focus_updates",
            "_spatial_replay_focus_updates",
            "_partition_control_focus_updates",
        ):
            cached = getattr(self, cached_attribute, {})
            setattr(
                self,
                cached_attribute,
                {source: weight for source, weight in cached.items() if source in eligible},
            )
        setattr(self, attribute, updates)

    def _observe_training_partition_replay_focus(
        self, tick: Any
    ) -> tuple[dict[SourceIdentity, float], tuple[SourceIdentity, ...]] | None:
        controller = self._partition_replay_focus
        accumulator = self._partition_replay_focus_accumulator
        if controller is None or accumulator is None:
            return None
        source_fingerprint = str(
            getattr(tick, "primary_training_source_id", "") or ""
        )
        source = self.runtime.source_identity_for_fingerprint(source_fingerprint)
        psnr_db = getattr(tick, "primary_training_psnr_db", None)
        if source is None or psnr_db is None:
            return None
        training_record = accumulator.observe(source, float(psnr_db))
        decision, sources = self._partition_replay_focus_decision(training_record)
        updates = _bounded_replay_identity_focus_updates(decision.weights)
        weights = tuple(weight for _, weight in decision.weights)
        partition_telemetry = {}
        if {
            "partition_k",
            "partition_denominator",
            "partition_n",
            "partition_psnr_db",
        }.issubset(training_record):
            from eligsir.utils.partition_eval import partition_id

            partition_k = int(training_record["partition_k"])
            partition_denominator = int(training_record["partition_denominator"])
            partition_counts = list(training_record["partition_n"])
            partition_telemetry = {
                "source_ordinal": int(source.source_ordinal),
                "source_partition_id": partition_id(
                    source.source_ordinal, partition_denominator, k=partition_k
                ),
                "partition_n": partition_counts,
                "partition_supervision_n": list(
                    training_record.get("partition_supervision_n", ())
                ),
                "partition_psnr_db": list(training_record["partition_psnr_db"]),
                "accumulator_samples_total": sum(partition_counts),
            }
        row = {
            "iteration": int(getattr(tick, "iteration", 0)),
            "active": bool(decision.active),
            "reason": decision.reason,
            "actuation_evidence": "training_primary_psnr",
            "score_scope": training_record["score_scope"],
            "source_id": source.fingerprint(),
            "history_samples": int(decision.history_samples),
            "controller_history_samples": int(decision.history_samples),
            "partition_weights": list(decision.partition_weights),
            "replay_population": len(weights),
            "weight_min": min(weights, default=1.0),
            "weight_max": max(weights, default=1.0),
            "weight_mean": (
                sum(weights) / len(weights) if weights else 1.0
            ),
        }
        row.update(partition_telemetry)
        path = Path(self.dataset.model_path) / "partition_replay_focus.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return updates, tuple(sources)

    def _observe_completed_partition_coverage(
        self, sources: tuple[SourceIdentity, ...]
    ) -> tuple[dict[SourceIdentity, float], tuple[SourceIdentity, ...]] | None:
        """Refresh bounded temporal coverage debt from already-completed B4 views."""

        accumulator = getattr(self, "_partition_replay_focus_accumulator", None)
        if accumulator is None:
            return None
        record = accumulator.record_successful_supervision(sources)
        decision, candidates = self._partition_replay_focus_decision(record)
        return _bounded_replay_identity_focus_updates(decision.weights), candidates

    def _partition_replay_focus_decision(
        self, record: dict[str, Any]
    ) -> tuple[Any, tuple[SourceIdentity, ...]]:
        controller = self._partition_replay_focus
        if controller is None:
            raise RuntimeError("partition replay focus controller is unavailable")
        sources = tuple(self.runtime.replay_candidate_identities())
        return controller.update(record, sources), sources

    def _observe_training_spatial_replay_focus(
        self, tick: Any
    ) -> tuple[dict[SourceIdentity, float], tuple[SourceIdentity, ...]] | None:
        controller = self._spatial_replay_focus
        sample = getattr(tick, "primary_training_spatial", None)
        if controller is None or sample is None:
            return None
        sources = self.runtime.replay_candidate_identities()
        decision = controller.update(sample, sources)
        updates = _bounded_replay_identity_focus_updates(decision.weights)
        weights = tuple(weight for _, weight in decision.weights)
        drops = dict(decision.cell_drop_from_peak_db)
        cell_scores = tuple(float(value) for _, value in decision.cell_psnr_db)
        aggregate_psnr_db = (
            None if not cell_scores else float(np.median(cell_scores))
        )
        worst_drop_db = max(drops.values(), default=0.0)
        growth_publisher = getattr(
            self.runtime, "publish_growth_quality_evidence", None
        )
        if aggregate_psnr_db is not None and callable(growth_publisher):
            gaussian_count = getattr(
                tick, "primary_training_gaussian_count", None
            )
            if gaussian_count is None:
                raise RuntimeError(
                    "adaptive growth evidence lacks its pre-mutation Gaussian count"
                )
            growth_publisher(
                iteration=int(getattr(tick, "iteration", 0)),
                aggregate_psnr_db=aggregate_psnr_db,
                gaussian_count=int(gaussian_count),
                worst_drop_from_peak_db=float(worst_drop_db),
            )

        from eligsir.utils.local_quality import cell_center

        cumulative_samples = dict(decision.cell_cumulative_samples)
        gain_since_first = dict(decision.cell_gain_since_first_observed_db)
        row = {
            "iteration": int(getattr(tick, "iteration", 0)),
            "active": bool(decision.active),
            "reason": decision.reason,
            "actuation_evidence": "training_primary_spatial_psnr",
            "actuation_mode": decision.actuation_mode,
            "score_scope": decision.score_scope,
            "history_samples": int(decision.history_samples),
            "growth_aggregate_psnr_db": aggregate_psnr_db,
            "growth_gaussian_count_pre_mutation": getattr(
                tick, "primary_training_gaussian_count", None
            ),
            "growth_worst_drop_from_peak_db": float(worst_drop_db),
            "cell_size_m": float(sample.cell_size_m),
            "source_id": sample.source.fingerprint(),
            "cells": [
                {
                    "cell_key": int(key),
                    "center_xyz_m": [
                        float(value)
                        for value in cell_center(key, sample.cell_size_m).tolist()
                    ],
                    "psnr_db": float(psnr_db),
                    "drop_from_peak_db": float(drops.get(key, 0.0)),
                    # gain_per_exposure classification (issue #1259):
                    # cumulative_samples = total supervised samples this
                    # cell has ever received; gain_since_first_observed_db =
                    # current - first-observed PSNR. High exposure + no
                    # gain => mobility/representation problem (defer to the
                    # MCMC-mobility thread). Low exposure + low quality =>
                    # scheduler/discovery problem (this mechanism's target).
                    "cumulative_samples": cumulative_samples.get(key),
                    "gain_since_first_observed_db": gain_since_first.get(key),
                }
                for key, psnr_db in decision.cell_psnr_db
            ],
            "replay_population": len(weights),
            "weight_min": min(weights, default=1.0),
            "weight_max": max(weights, default=1.0),
            "weight_mean": sum(weights) / len(weights) if weights else 1.0,
        }
        path = Path(self.dataset.model_path) / "spatial_replay_focus.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return updates, tuple(sources)

    def _report_eval_metadata(self) -> dict[str, Any]:
        metadata = dict(self.plan.eval_metadata)
        metadata["mapping_holdout_curve"] = dict(
            self._mapping_holdout_curve_metadata
        )
        return metadata

    def _gpu_memory_report_block(self) -> dict[str, Any]:
        """Sample causal runtime memory at a report boundary for its artifact."""
        tracker = self._gpu_memory_tracker
        if tracker is None:
            from eligsir.utils.gpu_memory_telemetry import disabled_report_block

            return disabled_report_block(
                "telemetry disabled (--no-streaming_memory_telemetry)"
            )
        # The opt-in report phase sampler takes the single entry snapshot.  Do
        # not duplicate that metadata walk here; the normal report remains
        # exactly backward compatible when the phase feature is disabled.
        if os.environ.get("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "0") != "1":
            try:
                tracker.snapshot(
                    iteration=int(self.runtime.iteration),
                    frames_ingested=int(self.runtime.frames_seen),
                    gaussians=self.runtime.gaussians,
                    streaming_scene=self.runtime.scene,
                    observation_db=None,
                )
            except Exception:
                from eligsir.utils.gpu_memory_telemetry import disabled_report_block

                return disabled_report_block(
                    "telemetry snapshot failed at causal report boundary"
                )
        return tracker.report_block()

    def _experiment_validity_source_indices(self) -> tuple[int, ...]:
        """Return the exact terminal COMMIT+DEFER trajectory population."""

        ordinals = tuple(
            int(source.source_ordinal)
            for source in self.runtime.replay_candidate_identities()
        )
        if len(ordinals) != len(set(ordinals)):
            raise RuntimeError(
                "retained causal catalog contains duplicate source ordinals"
            )
        return tuple(sorted(ordinals))

    def finalize(
        self,
        summary: Any,
        *,
        subdir: str | None = None,
        experiment_validity_is_final: bool = True,
        mapping_phase_source: str = "causal_source_completion_boundary",
    ) -> dict[str, Any] | None:
        """Write the heavy final report from the already-fenced camera pools."""

        # Any still-in-flight async mapping-phase evaluate (issue #1314) must
        # complete -- and its measured stall stats must be resolved into a
        # real (non-provisional) throughput verdict -- before this report is
        # written.
        self.drain_async_holdout_evaluator()
        self._resolve_async_throughput_evidence()
        # The replay driver owns evaluation-excluding work clocks.  Preserve
        # its terminal values for both paced and unpaced execution before the
        # final metric pass adds any reporting overhead.
        self._mapping_active_s = float(summary.mapping_active_s)
        self._elapsed_s = float(summary.elapsed_s)
        # Issue #1664: resolution-tier usage rows are buffered in memory and
        # only flushed at a cadence (or at this finalization point) rather
        # than on every completed transaction -- flush explicitly before
        # reading the trace back for the plot below, since this report can
        # run before ``CanonicalMappingRuntime.close()``'s own flush.
        flush_resolution_tier_usage(self.dataset.model_path)
        plot_spatial_replay_focus(self.dataset.model_path)
        plot_causal_throughput(self.dataset.model_path)
        plot_resolution_tier_over_time(self.dataset.model_path)
        if not self.plan.holdout_ordinals:
            return None
        self._ensure_cameras()
        from dataclasses import asdict, is_dataclass

        from gaussian_renderer import render
        from eligsir.utils.comparison_report import (
            determinism_from_args,
            git_provenance_from_args,
            write_post_training_report,
        )
        from eligsir.utils.execution_semantics import build_execution_semantics_block
        from eligsir.utils.experiment_origin import experiment_origin_from_args
        from eligsir.runtime.transfers import transfer_pool_report_block

        clocks = self.runtime._services.four_clocks.summary()
        mapping_clocks = asdict(clocks) if is_dataclass(clocks) else dict(clocks)
        mapping_clocks = mapping_clocks_for_report(
            mapping_clocks, self._mapping_holdout_curve_metadata
        )
        evaluation_tick_clock_snapshot = _terminal_evaluation_tick_clock_snapshot(self)
        evaluation_tick_clock_snapshot.update({
            "evaluation_phase": "post_mapping_report",
            "evaluation_observer_execution": "post_source_completion",
            "mapping_execution_paused": False,
            "throughput_evidence_valid": bool(
                self._mapping_holdout_curve_metadata["throughput_evidence_valid"]
            ),
            "throughput_evidence_status": str(
                self._mapping_holdout_curve_metadata["throughput_evidence_status"]
            ),
            "evaluation_cadence_clock": str(
                self._mapping_holdout_curve_metadata["cadence_clock"]
            ),
        })
        from eligsir.utils.coverage_ledger import CoverageLedger

        coverage_counters = None
        frames_admitted_override = None
        ledger = CoverageLedger.restore(
            getattr(summary, "coverage_ledger", None),
            expected_source={
                "source_identity": str(self.dataset.source_path),
                "raw_registry_source_frames": None,
                "effective_stride": 1,
                "resolved_source_frames": int(
                    getattr(self, "source_total", None) or summary.source_offered
                ),
                "source_denominator_provenance": "causal_source_catalog",
            },
        )
        if ledger is not None:
            snapshot = ledger.snapshot()
            accounting = snapshot.get("causal_source_accounting")
            if (
                isinstance(accounting, dict)
                and accounting.get("mode") == "terminal_disposition"
                and accounting.get("unique_source_count") == snapshot["resolved_source_frames"]
                and snapshot["source_frames_ingested_total"] == snapshot["resolved_source_frames"]
                and snapshot["unprocessed_frames"] == 0
            ):
                coverage_counters = snapshot
                frames_admitted_override = int(snapshot["frames_admitted"])
        return write_post_training_report(
            model_path=str(self.dataset.model_path),
            iteration=int(self.runtime.iteration),
            gaussians=self.runtime.gaussians,
            train_cams=self._report_training_cameras(),
            test_cams=self.eval_cameras,
            render_fn=render,
            pipe=self.runtime.pipe,
            background=self.runtime._background,
            log_prefix=(
                "causal_mapping_endpoint"
                if subdir == "mapping_endpoint"
                else "causal_final"
            ),
            subdir=subdir,
            report_videos=bool(getattr(self.args, "streaming_report_videos", True)),
            train_metrics_max=int(
                getattr(self.args, "streaming_report_train_metrics_max", 128)
            ),
            test_metrics_max=int(
                getattr(self.args, "streaming_report_test_metrics_max", 128)
            ),
            trajectory_max_frames=int(
                getattr(self.args, "streaming_report_trajectory_max_frames", 120)
            ),
            streaming_scene=self.runtime.scene,
            eval_metadata=self._report_eval_metadata(),
            stream_end_reason="source_exhausted_then_optimizer_horizon",
            frame_admission="preselected",
            frames_admitted_override=frames_admitted_override,
            coverage_counters=coverage_counters,
            mapping_clocks=mapping_clocks,
            baseline_run=getattr(self.args, "baseline_run", None),
            last_ingest_iter=self.last_ingest_iteration,
            true_final_iteration=int(self.runtime.iteration),
            mapping_phase_source=mapping_phase_source,
            git_provenance=git_provenance_from_args(self.args),
            determinism=determinism_from_args(self.args),
            experiment_origin=experiment_origin_from_args(self.args),
            execution_semantics=build_execution_semantics_block(self.args),
            run_intent=self.args.run_intent,
            experiment_context=self.args.experiment_context,
            experiment_validity_class=getattr(
                self.args, "experiment_validity_class", None
            ),
            experiment_validity_is_final=experiment_validity_is_final,
            experiment_validity_source_indices=(
                self._experiment_validity_source_indices()
            ),
            experiment_validity_source_index_provenance=(
                "terminal_admit_defer_retained_catalog"
            ),
            partition_control=(
                None
                if self._partition_control_runtime is None
                or self._partition_control_roles is None
                else self._partition_control_runtime.recorder.report_block(
                    self._partition_control_roles
                )
            ),
            evaluation_tick_clock_snapshot=evaluation_tick_clock_snapshot,
            scene_source=_scene_source_for_report(self.dataset),
            transfer_pool=transfer_pool_report_block(
                getattr(self.runtime, "_transfer_owner", None)
            ),
            gpu_memory=self._gpu_memory_report_block(),
            allocator_phase_tracker=self._gpu_memory_tracker,
            # Issue #1273: the causal source buffer's own per-frame drop
            # attribution. Purely additive -- a REAL_TIME_DROP boundary, so it
            # participates in no coverage verdict. NOTE this reporter is also
            # invoked mid-run as the mapping_endpoint reporter, where the
            # counts are legitimately partial; only the final report's block
            # is the run total.
            source_ingress=(dict(summary.source_ingress) or None)
            if getattr(summary, "source_ingress", None)
            else None,
            supervision_work=(
                self.runtime._supervision_work_clocks.report_block(
                    wall_time_s=float(self._elapsed_s)
                )
                if getattr(self.runtime, "_supervision_work_clocks", None) is not None
                else None
            ),
        )


__all__ = [
    "CausalHoldoutPlan",
    "CausalEvaluationObserver",
    "append_primary_training_psnr",
    "bounded_evaluation_cameras",
    "build_causal_holdout_cameras",
    "causal_holdout_plan",
    "plot_paired_psnr",
    "mapping_holdout_curve_metadata",
    "mapping_clocks_for_report",
]
