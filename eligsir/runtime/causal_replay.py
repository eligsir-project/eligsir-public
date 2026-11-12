"""Dataset-neutral causal replay through the canonical application boundary."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Sequence
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar

import numpy as np

from eligsir.contracts.control import EvidenceFact
from eligsir.contracts.runtime import SensorPacket, SourceIdentity
from eligsir.runtime.config import IngressTelemetrySnapshot
from eligsir.runtime.source_ingress import (
    CausalIngressRecorder,
    IngestionValueConfig,
)

from eligsir.utils.ingest_phase_profiler import _timed as _phase_timed
from eligsir.utils.ingest_phase_profiler import record_timed as _phase_record_timed
from eligsir.utils.tick_torch_profile import TickTorchProfileWindow

from .application import MappingApplication
from .prepared_ingress import PreparedIngressDeferred
from .prepared_source_prep_worker import READY_WAIT_FALLBACK_S
from .replay import packet_from_streaming_frame
from eligsir.utils.coverage_ledger import CoverageLedger


RecordT = TypeVar("RecordT")
TickObserver = Callable[[Any], None]
TickClockObserver = Callable[[Any, "CausalReplayClock"], None]


def _record_causal_coverage(
    ledger: CoverageLedger | None,
    *,
    source: SourceIdentity,
    role: str,
    admitted: bool,
    bootstrap_frames: int,
) -> None:
    if ledger is not None:
        ledger.record_causal_source(
            source=source,
            role=role,
            admitted=admitted,
            bootstrap=(
                role == "train" and source.source_ordinal < bootstrap_frames
            ),
        )


def _plan_replay_candidate_evidence(
    plan: Any,
) -> tuple[tuple[SourceIdentity, ...], tuple[float, ...]] | None:
    """Return complete plan-carried membership, never a reconstructed catalog."""

    candidates = tuple(getattr(plan, "replay_candidate_sources", ()) or ())
    weights = tuple(getattr(plan, "replay_candidate_weights", ()) or ())
    if not candidates or len(candidates) != len(weights):
        return None
    if int(getattr(plan, "replay_population_size", -1)) != len(candidates):
        return None
    if (
        not all(isinstance(source, SourceIdentity) for source in candidates)
        or len(set(candidates)) != len(candidates)
        or not set(getattr(plan, "replay_sources", ()) or ()).issubset(candidates)
    ):
        return None
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or float(weight) < 0.0
        for weight in weights
    ):
        return None
    return candidates, tuple(float(weight) for weight in weights)


def _extract_prepared_insertion_counters(tick: Any) -> tuple[int, int, int, str | None]:
    """Return executed, accepted-seed, zero-seed, and skip deltas for READY."""
    if getattr(tick, "state", None) is None:
        return 0, 0, 0, None
    if (skip := getattr(tick, "prepared_insertion_skip", None)) is not None:
        return 0, 0, 0, str(skip)
    executed = seeds = zero_seed = 0
    for event in tuple(getattr(tick, "mutation_events", ()) or ()):
        if str(getattr(event, "kind", "")) == "live_depth":
            inserted = int(getattr(event, "inserted", 0))
            executed += 1
            seeds += max(0, inserted)
            zero_seed += int(inserted == 0)
    return executed, seeds, zero_seed, None


def _prepared_ready_progress_fields(tick: Any) -> str:
    """Format the already-captured READY snapshot for low-rate progress logs."""

    counters = getattr(tick, "ready_counters", None)
    selection = getattr(tick, "selection", None)
    if counters is None or selection is None:
        return ""
    constellation = getattr(
        getattr(tick, "work_plan", None), "supervision_constellation", None
    )
    requested = int(getattr(constellation, "realized_batch_size", 0))
    if requested <= 0:
        return ""
    selected = len(tuple(getattr(selection, "leases", ()) or ()))
    ready_count = getattr(tick, "ready_count", None)
    exact_ready = "?" if ready_count is None else str(int(ready_count))
    state = getattr(tick, "state", "unknown")
    state = getattr(state, "value", state)
    reason = str(getattr(tick, "reason", "") or "none").split(":", 1)[0]
    replay_ready = int(counters.replay_ready_count)
    replay_batches = replay_ready // requested if requested > 0 else 0
    batches_field = f"replay_ready_batches={replay_batches}" if requested > 0 else ""
    ready_bytes_cur = getattr(counters, "ready_bytes_current", 0)
    ready_bytes_peak = getattr(counters, "ready_bytes_peak", 0)
    ready_bytes_cur_mb = float(ready_bytes_cur) / (1024 * 1024)
    ready_bytes_peak_mb = float(ready_bytes_peak) / (1024 * 1024)
    absent_replans = getattr(tick, "absent_replans", None)
    absent = "" if absent_replans is None else f" absent_replans={int(absent_replans)}"
    return (
        f" prepared_b4={selected}/{requested} "
        f"prepared_exact_ready={exact_ready}/{requested} "
        f"ready_post_select=f{int(counters.fresh_ready_count)},r{replay_ready} "
        f"{batches_field} "
        f"lane_occupancy=f{int(counters.fresh_pending)},r{int(counters.replay_pending)},"
        f"i{int(counters.in_flight)},nonresident={int(counters.full_not_resident_count)} "
        f"ready_bytes={ready_bytes_cur_mb:.1f}/{ready_bytes_peak_mb:.1f}MiB "
        f"prepared_state={state} prepared_reason={reason}{absent}"
    )


def decision_trace_row(tick: Any) -> dict[str, Any]:
    """Serialize completed CPU control and mutation evidence only."""

    snapshot = getattr(tick, "control_snapshot", None)
    plan = getattr(tick, "work_plan", None)
    plan_candidate_evidence = _plan_replay_candidate_evidence(plan)
    if plan_candidate_evidence is None:
        replay_candidates = tuple(getattr(snapshot, "replay_candidates", ()) or ())
        replay_candidate_weights = tuple(
            getattr(snapshot, "replay_candidate_weights", ()) or ()
        )
        candidate_membership_source = "control_snapshot"
    else:
        replay_candidates, replay_candidate_weights = plan_candidate_evidence
        candidate_membership_source = "work_plan"
    snapshot_evidence = (
        {}
        if snapshot is None
        else {fact.name: fact.value for fact in snapshot.evidence}
    )
    retained_catalog_population = getattr(
        tick, "retained_catalog_population_post_dispositions", None
    )
    prepared_state = getattr(tick, "state", None)
    if hasattr(prepared_state, "value"):
        prepared_state = prepared_state.value
    selection = getattr(tick, "selection", None)
    ready_batch = None
    if selection is not None:
        requested = len(
            tuple(getattr(plan, "supervision_sources", ()))
            + tuple(getattr(plan, "replay_sources", ()))
        )
        selected = len(getattr(selection, "leases", ()))
        ready_batch = {
            "requested": requested or None,
            "selected": selected,
            "fresh": int(getattr(selection, "fresh_count", 0)),
            "replay": int(getattr(selection, "replay_count", 0)),
            "full": None if requested == 0 else selected == requested,
            "waited_for_ready_source": False,
            "starved": bool(getattr(selection, "starved", False)),
        }

    transfers = getattr(tick, "transfers", None)
    transfer_record = None
    if transfers is not None:
        transfer_record = {
            name: int(getattr(transfers, name))
            for name in (
                "h2d_ops",
                "h2d_bytes",
                "d2h_ops",
                "d2h_bytes",
                "pool_waits",
                "pool_drops",
                "explicit_sync_waits",
                "explicit_sync_wait_ms",
            )
            if hasattr(transfers, name)
        }
        if hasattr(transfers, "explicit_sync_wait_ms"):
            transfer_record["explicit_sync_wait_ms"] = float(
                getattr(transfers, "explicit_sync_wait_ms")
            )

    runtime_telemetry = getattr(tick, "telemetry", None)
    telemetry_record = None
    if runtime_telemetry is not None:
        gpu = getattr(runtime_telemetry, "gpu_transaction", None)
        gpu_record = None
        if gpu is not None:
            transaction_type = getattr(gpu, "transaction_type", None)
            gpu_record = {
                "transaction_id": int(getattr(gpu, "transaction_id")),
                "transaction_type": str(
                    getattr(transaction_type, "value", transaction_type)
                ),
                "duration_ms": float(getattr(gpu, "duration_ms")),
                "map_content_version": int(
                    getattr(gpu, "map_content_version")
                ),
                "topology_version": int(getattr(gpu, "topology_version")),
                "parameter_version": int(
                    getattr(gpu, "parameter_version")
                ),
                "batch_size": int(getattr(gpu, "batch_size")),
                "pixels": int(getattr(gpu, "pixels")),
                "rasterizer_calls": int(getattr(gpu, "rasterizer_calls")),
                "optimizer_updates": int(getattr(gpu, "optimizer_updates")),
                "candidates": int(getattr(gpu, "candidates")),
                "mutation_type": str(getattr(gpu, "mutation_type")),
                "inserted": int(getattr(gpu, "inserted")),
                "pruned": int(getattr(gpu, "pruned")),
                "relocated": int(getattr(gpu, "relocated")),
                "corrected": int(getattr(gpu, "corrected")),
            }
        map_state = getattr(runtime_telemetry, "map_state", None)
        map_state_record = None
        if map_state is not None:
            map_state_record = {
                name: int(getattr(map_state, name))
                for name in (
                    "gaussian_count",
                    "map_content_version",
                    "topology_version",
                    "parameter_version",
                )
            }
        telemetry_record = {
            "gpu_transaction": gpu_record,
            "map_state": map_state_record,
        }

    def source_ids(sources: Any) -> list[str]:
        return [source.fingerprint() for source in tuple(sources)]

    def source_records(sources: Any) -> list[dict[str, Any]]:
        return [
            {
                "fingerprint": source.fingerprint(),
                "source_ordinal": int(source.source_ordinal),
                "timestamp_ns": int(source.timestamp_ns),
            }
            for source in tuple(sources)
        ]

    insertion = None if plan is None else plan.insertion
    mutation_events = []
    for event in tuple(getattr(tick, "mutation_events", ()) or ()):
        source = getattr(event, "source", None)
        mutation_events.append({
            "kind": str(getattr(event, "kind", "unknown")),
            "source": None if source is None else source.fingerprint(),
            "candidates": int(getattr(event, "candidates", 0)),
            "inserted": int(getattr(event, "inserted", 0)),
            "pruned": int(getattr(event, "pruned", 0)),
            "relocated": int(getattr(event, "relocated", 0)),
            "corrected": int(getattr(event, "corrected", 0)),
        })
    policy_disposition = getattr(tick, "population_policy_disposition", None)
    if hasattr(policy_disposition, "value"):
        policy_disposition = policy_disposition.value
    if policy_disposition is None:
        policy_disposition = "not_observed"
    stale_current_mismatches = tuple(
        getattr(tick, "population_stale_current_mismatches", ()) or ()
    )
    selection_telemetry = tuple(
        getattr(tick, "population_policy_selection_telemetry", ()) or ()
    )
    dispositions = []
    for disposition in tuple(getattr(tick, "dispositions", ()) or ()):
        eligibility = disposition.modality_eligibility
        representative = getattr(disposition, "representative", None)
        dispositions.append({
            "source": disposition.source.fingerprint(),
            "source_ordinal": int(disposition.source.source_ordinal),
            "timestamp_ns": int(disposition.source.timestamp_ns),
            "kind": disposition.kind.value,
            "semantic_action": getattr(
                getattr(disposition, "semantic_action", None), "value", None
            ),
            "historical_eligibility": getattr(
                getattr(disposition, "historical_eligibility", None), "value", None
            ),
            "reason": disposition.reason,
            "mapping_candidate": bool(disposition.mapping_candidate),
            "admit_supervision": bool(disposition.admit_supervision),
            "allow_geometry_insertion": bool(disposition.allow_geometry_insertion),
            "residency": disposition.residency.value,
            "representative": (
                None if representative is None else representative.fingerprint()
            ),
            "evidence": {
                fact.name: fact.value for fact in disposition.evidence_trace
            },
            "modalities": {
                "rgb_supervision": bool(eligibility.rgb_supervision),
                "rgb_quality": eligibility.rgb_quality,
                "depth_supervision": bool(eligibility.depth_supervision),
                "depth_quality": eligibility.depth_quality,
                "depth_insertion": bool(eligibility.depth_insertion),
            },
        })
    row = {
        "snapshot_id": getattr(tick, "snapshot_id", None),
        "iteration": int(getattr(tick, "iteration", 0)),
        "executed": bool(getattr(tick, "executed", False)),
        "reason": str(getattr(tick, "reason", "")),
        "prepared_state": prepared_state,
        "prepared_insertion_skip": getattr(tick, "prepared_insertion_skip", None),
        "ready_batch": ready_batch,
        "policy": {
            "legacy_adaptive_growth_enabled": bool(
                getattr(tick, "adaptive_growth_enabled", False)
            ),
            "population_policy_disposition": str(policy_disposition),
            "stale_current_mismatches": [list(item) for item in stale_current_mismatches],
            "selection_telemetry": list(selection_telemetry),
        },
        "optimizer_steps": int(getattr(tick, "optimizer_steps", 0)),
        "supervision_views": int(getattr(tick, "supervision_views", 0)),
        "primary_training_psnr_db": getattr(
            tick, "primary_training_psnr_db", None
        ),
        "primary_training_source_id": getattr(
            tick, "primary_training_source_id", None
        ),
        "dispositions": dispositions,
        "retained_catalog": (
            None
            if retained_catalog_population is None
            else {
                "population": int(retained_catalog_population),
                "position": "post_dispositions",
            }
        ),
        "versions": {
            "map_content": int(getattr(tick, "map_content_version", 0)),
            "topology": int(getattr(tick, "topology_version", 0)),
            "parameter": int(getattr(tick, "parameter_version", 0)),
        },
        "transfers": transfer_record,
        "telemetry": telemetry_record,
        "snapshot": None
        if snapshot is None
        else {
            "control_tick_epoch": int(getattr(snapshot, "control_tick_epoch", 0)),
            "ingress_queue_depth": int(snapshot.ingress_queue_depth),
            "freshness": str(snapshot.freshness_mode),
            "retained_catalog_population": snapshot_evidence.get(
                "replay_catalog_population"
            ),
            "retained_catalog_population_position": "pre_snapshot_disposition",
            "replay_shortlist_configured_limit": snapshot_evidence.get(
                "replay_shortlist_configured_limit"
            ),
            "replay_shortlist_realized_population": len(replay_candidates),
            "replay_candidate_population": len(replay_candidates),
            "replay_candidate_records": source_records(replay_candidates),
            "replay_candidate_membership_source": candidate_membership_source,
            "replay_weight_min": min(replay_candidate_weights, default=None),
            "replay_weight_max": max(replay_candidate_weights, default=None),
            "replay_weight_sum": sum(replay_candidate_weights),
            "evidence": snapshot_evidence,
        },
        "plan_id": None if plan is None else plan.plan_id,
        "native_geometry_evidence": {
            "purpose": bool(
                getattr(plan, "native_geometry_evidence_purpose", False)
            ),
            "lane": getattr(tick, "native_geometry_evidence_telemetry", None),
        },
        "supervision_sources": []
        if plan is None
        else source_ids(plan.supervision_sources),
        "supervision_source_records": []
        if plan is None
        else source_records(plan.supervision_sources),
        "replay_sources": [] if plan is None else source_ids(plan.replay_sources),
        "replay_source_records": []
        if plan is None
        else source_records(plan.replay_sources),
        "replay_trial_sources": []
        if plan is None
        else source_ids(getattr(plan, "replay_trial_sources", ())),
        "replay_trial_source_records": []
        if plan is None
        else source_records(getattr(plan, "replay_trial_sources", ())),
        "allocation_request": None
        if plan is None
        else {
            "supervision_slots": int(
                getattr(plan, "requested_supervision_slots", 0)
            ),
            "replay_slots": int(getattr(plan, "requested_replay_slots", 0)),
        },
        "replay_sampling": None
        if plan is None
        else {
            "population": int(getattr(plan, "replay_population_size", 0)),
            "seed": int(getattr(plan, "replay_sampling_seed", 0)),
            "epoch": int(getattr(plan, "replay_sampling_epoch", 0)),
            "selected_weights": list(getattr(plan, "replay_selected_weights", ())),
        },
        "resolution_tier": None
        if plan is None
        else str(getattr(getattr(plan, "resolution_tier", "native"), "value", "native")),
        "reason_trace": [] if plan is None else list(plan.reason_trace),
        "insertion": None
        if insertion is None
        else {
            "source": None
            if insertion.source is None
            else insertion.source.fingerprint(),
            "sample_stride": int(insertion.sample_stride),
            "candidate_budget": int(insertion.candidate_budget),
            "seed_budget": int(insertion.seed_budget),
        },
        "mutation_request": 0
        if plan is None
        else int(plan.mutation_budget.max_total),
        "mutation": {
            "candidates": int(getattr(tick, "insertion_candidates", 0)),
            "inserted": int(getattr(tick, "inserted", 0)),
            "pruned": int(getattr(tick, "pruned", 0)),
            "relocated": int(getattr(tick, "relocated", 0)),
            "corrected": int(getattr(tick, "corrected", 0)),
            "structural_mutation": bool(
                getattr(tick, "structural_mutation", False)
            ),
            "events": mutation_events,
        },
        "insertion_filter": getattr(tick, "insertion_filter", None),
    }
    constellation = getattr(plan, "supervision_constellation", None)
    if constellation is not None:
        row["supervision_constellation"] = {
            "requested_batch_size": int(constellation.requested_batch_size),
            "requested_camera_count": int(constellation.requested_camera_count),
            "realized_batch_size": int(constellation.realized_batch_size),
            "realized_camera_count": int(constellation.realized_camera_count),
            "microbatch_sizes": list(
                getattr(tick, "microbatch_sizes", constellation.microbatch_sizes)
            ),
        }
    if hasattr(tick, "replay_challenger_trace"):
        truncated = bool(getattr(tick, "replay_challenger_trace_truncated", False))
        row["replay_challenger_trace"] = list(
            getattr(tick, "replay_challenger_trace", ()) or ()
        )
        row["replay_challenger_trace_truncated"] = truncated
        row["replay_challenger_trace_complete"] = not truncated
    return row


class JsonlDecisionTraceObserver:
    """Opt-in append-only debug trace; normal replay constructs no observer.

    Rows are serialized and appended to a bounded in-memory buffer as they
    are observed, never opened/written/closed to disk one at a time (issue
    #1664): this observer is called from the same host thread that submits
    GPU transactions, and per-tick file I/O there is exactly the structural
    problem this issue targets. The buffer is flushed in batches -- every
    ``flush_every`` accepted rows, and explicitly by ``flush()``/
    ``finalize()`` or a caller's own exception-safe cleanup -- so at most a
    small, bounded number of rows can ever be lost to an unclean process
    death between flushes.

    This trace is used for correctness diagnosis, so it must never *quietly*
    drop rows: a failed disk write is recorded in ``write_error`` (surfaced
    by ``finalize()``) but does not raise, since the underlying failure may
    be transient and mapping must not crash on a diagnostic write; the
    buffered rows are retained and retried on the next flush. Only if that
    retry pressure ever pushes the buffer past ``max_buffered_rows`` -- i.e.
    the failure is not transient -- does this raise, loudly and immediately,
    rather than silently discarding causal rows.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        flush_every: int = 200,
        max_buffered_rows: int = 20_000,
    ) -> None:
        self._path = Path(path)
        self._zero_progress_signature: str | None = None
        self._flush_every = max(1, int(flush_every))
        self._max_buffered_rows = max(self._flush_every, int(max_buffered_rows))
        self._buffer: list[str] = []
        self._rows_appended = 0
        self._rows_flushed = 0
        self._write_error: str | None = None

    def __call__(self, tick: Any) -> None:
        row = decision_trace_row(tick)
        zero_progress = not bool(getattr(tick, "executed", False)) and (
            int(getattr(tick, "optimizer_steps", 0)) == 0
            and int(getattr(tick, "supervision_views", 0)) == 0
        )
        signature_row = dict(row)
        signature_row.pop("iteration", None)
        signature_row.pop("snapshot_id", None)
        signature = json.dumps(signature_row, sort_keys=True)
        if zero_progress and signature == self._zero_progress_signature:
            return
        self._zero_progress_signature = signature if zero_progress else None
        # Serialize now (row order must match observation order exactly) but
        # defer the file write to the next batched flush.
        self._buffer.append(json.dumps(row, sort_keys=True) + "\n")
        self._rows_appended += 1
        if len(self._buffer) >= self._flush_every:
            self.flush()
        if len(self._buffer) > self._max_buffered_rows:
            raise RuntimeError(
                "JsonlDecisionTraceObserver buffer overflow: "
                f"{len(self._buffer)} unflushed rows exceed "
                f"max_buffered_rows={self._max_buffered_rows} for "
                f"{self._path} (last write error: {self._write_error!r}); "
                "refusing to silently discard causal decision-trace rows."
            )

    def flush(self) -> None:
        """Batch-write buffered rows to disk; a no-op when the buffer is empty.

        On an ``OSError`` the buffer is left intact (nothing is dropped) and
        ``write_error`` records the failure so ``finalize()`` can surface it
        -- the next ``flush()`` call retries the same rows.
        """

        if not self._buffer:
            return
        try:
            with self._path.open("a") as handle:
                handle.writelines(self._buffer)
        except OSError as exc:
            self._write_error = repr(exc)
            return
        self._rows_flushed += len(self._buffer)
        self._buffer.clear()
        self._write_error = None

    def finalize(self) -> dict[str, Any]:
        """Flush remaining rows, then write bounded, post-run replay diagnostics."""

        self.flush()
        from .replay_sampling_telemetry import write_replay_sampling_telemetry

        summary = write_replay_sampling_telemetry(self._path)
        summary["decision_trace_rows_appended"] = self._rows_appended
        summary["decision_trace_rows_flushed"] = self._rows_flushed
        summary["decision_trace_write_error"] = self._write_error
        return summary


class DatasetPacketAdapter(Protocol[RecordT]):
    """Source-specific conversion; datasets never enter application construction."""

    def packet_from_record(self, record: RecordT) -> SensorPacket: ...


@dataclass(frozen=True, slots=True)
class StreamingFramePacketAdapter:
    """Decode one dataset frame into the canonical metric RGB-D packet.

    This remains outside :class:`MappingApplication`: it is the only place a
    dataset frame's path/bytes/cache representation is interpreted.  The core
    receives a fully owned ``SensorPacket`` with RGB uint8 and metric float32
    depth, matching the ROS packet boundary.
    """

    source_id: str
    stream_epoch: int = 0
    tracking_provider_name: str = "dataset_pose"

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if self.stream_epoch < 0:
            raise ValueError("stream_epoch must be non-negative")
        if not self.tracking_provider_name:
            raise ValueError("tracking_provider_name must be non-empty")

    def packet_from_record(self, record: Any) -> SensorPacket:
        from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb
        from eligsir.utils.rgbd_frames import depth_to_meters

        rgb = np.asarray(load_frame_rgb(record), dtype=np.uint8)
        raw_depth = load_frame_depth_np(record)
        depth = None
        if raw_depth is not None:
            depth = depth_to_meters(
                raw_depth, float(getattr(record, "depth_scale"))
            ).astype(np.float32, copy=False)
        return packet_from_streaming_frame(
            record,
            source_id=self.source_id,
            stream_epoch=self.stream_epoch,
            provider=self.tracking_provider_name,
            rgb=rgb,
            depth=depth,
        )


@dataclass(frozen=True, slots=True)
class CausalReplayClock:
    """Exact completed-tick work clocks plus its drained source frontier.

    ``source_frame_index`` is the newest canonical source ordinal drained by
    the mapping owner, including reserved holdout and guard packets.
    """

    optimizer_steps: int
    supervision_views: int
    mapping_active_s: float
    elapsed_s: float
    source_frame_index: int | None = None


@dataclass(frozen=True, slots=True)
class CausalReplaySummary:
    submitted: int
    ticks: int
    source_identities: tuple[SourceIdentity, ...]
    optimizer_steps: int = 0
    supervision_views: int = 0
    zero_frame_ticks: int = 0
    transport_dropped: int = 0
    transport_dropped_at_arrival: int = 0
    transport_dropped_at_enqueue: int = 0
    source_ingress: dict = field(default_factory=dict)
    coverage_ledger: dict[str, Any] | None = None
    source_queue_high_watermark: int = 0
    source_offered_fps: float | None = None
    elapsed_s: float = 0.0
    submission_mean_ms: float = 0.0
    mapping_active_s: float = 0.0
    outer_loop_wall_s: float = 0.0
    outer_take_wait_s: float = 0.0
    outer_submit_s: float = 0.0
    outer_tick_total_s: float = 0.0
    outer_bookkeeping_s: float = 0.0
    outer_unaccounted_s: float = 0.0
    outer_decode_s: float = 0.0
    outer_real_ticks: int = 0
    outer_drain_ticks: int = 0
    outer_real_tick_s: float = 0.0
    outer_drain_tick_s: float = 0.0
    optimizer_steps_per_active_s: float = 0.0
    supervision_views_per_active_s: float = 0.0
    supervision_views_per_elapsed_s: float = 0.0
    source_offered: int = 0
    holdout_reserved: int = 0
    guard_withheld: int = 0
    last_ingest_iteration: int | None = None
    requested_supervision_constellations: dict[str, int] = field(
        default_factory=dict
    )
    realized_supervision_constellations: dict[str, int] = field(
        default_factory=dict
    )
    supervision_constellation_modes: dict[str, int] = field(default_factory=dict)
    supervision_constellation_reasons: dict[str, int] = field(default_factory=dict)
    admitted_new_total: int = 0
    camera_materialize_cache_hits: int = 0
    camera_materialize_cache_misses: int = 0
    ready_bytes_peak: int = 0
    ready_bytes_current_at_end: int = 0
    ready_bytes_by_tier_kind: dict[str, int] = field(default_factory=dict)
    prepared_insertions_executed: int = 0
    prepared_insertion_seeds_accepted: int = 0
    prepared_insertions_zero_seed: int = 0
    prepared_insertion_skips: dict[str, int] = field(default_factory=dict)

    @property
    def prepared_insertion_skip_total(self) -> int:
        """Derive the scalar skip count from its reason histogram."""

        return sum(int(count) for count in self.prepared_insertion_skips.values())


class _MappingDrainTracker:
    """Track outstanding admitted identities using completed tick evidence."""

    def __init__(self) -> None:
        self._terminal_submissions = 0
        self._pending_new_sources: OrderedDict[SourceIdentity, None] = OrderedDict()

    @property
    def pending_new_sources(self) -> int:
        return len(self._pending_new_sources)

    @property
    def pending_sources(self) -> tuple[SourceIdentity, ...]:
        return tuple(self._pending_new_sources)

    @property
    def terminal_submissions(self) -> int:
        return self._terminal_submissions

    def record_submit(self, outcome: Any) -> None:
        event = getattr(outcome, "event", None)
        source = getattr(event, "source", None)
        if not isinstance(source, SourceIdentity):
            raise RuntimeError("causal submit returned no terminal source identity")
        self._terminal_submissions += 1
        disposition = getattr(outcome, "disposition", None)
        if disposition is None or not bool(
            getattr(disposition, "admit_supervision", False)
        ):
            return
        self._pending_new_sources[source] = None
        self._pending_new_sources.move_to_end(source)

    def record_tick(self, result: Any) -> bool:
        pending_before = len(self._pending_new_sources)
        completed_sources = getattr(result, "completed_supervision_sources", None)
        if completed_sources is not None:
            for source in tuple(completed_sources):
                self._pending_new_sources.pop(source, None)
            return len(self._pending_new_sources) < pending_before
        completed = int(getattr(result, "optimizer_steps", 0))
        executed = bool(getattr(result, "executed", completed > 0))
        if not executed:
            return False
        plan = getattr(result, "work_plan", None)
        if plan is None:
            return False
        for source in (
            *tuple(getattr(plan, "supervision_sources", ()) or ()),
            *tuple(getattr(plan, "replay_sources", ()) or ()),
        ):
            self._pending_new_sources.pop(source, None)
        return len(self._pending_new_sources) < pending_before

    def reached_barrier(self, result: Any, *, submitted: int) -> bool:
        if (
            self._terminal_submissions != int(submitted)
            or self._pending_new_sources
        ):
            return False
        completed_sources = getattr(result, "completed_supervision_sources", None)
        if completed_sources is not None:
            return bool(getattr(result, "executed", False))
        plan = getattr(result, "work_plan", None)
        if tuple(getattr(plan, "supervision_sources", ()) or ()):
            return False
        replay_sources = tuple(getattr(plan, "replay_sources", ()) or ())
        if replay_sources:
            completed = int(getattr(result, "optimizer_steps", 0))
            return bool(getattr(result, "executed", completed > 0))
        return True


def _prepared_bootstrap_pending(result: Any) -> bool:
    """Recognize the prepared EOF state without importing its state enum."""

    state = getattr(result, "state", None)
    return getattr(state, "value", state) == "bootstrap_pending"


class CausalIngressTelemetryState:
    """Transport-owned scalar evidence sampled by the canonical controller."""

    def __init__(self, *, batch_size: int = 1) -> None:
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self._batch_size = int(batch_size)
        self._lock = threading.Lock()
        self._depth = 0
        self._capacity = 0
        self._dropped = 0
        self._input_exhausted = False

    def publish(self, *, queue_depth: int, capacity: int, dropped: int) -> None:
        values = (int(queue_depth), int(capacity), int(dropped))
        if min(values) < 0:
            raise ValueError("ingress telemetry counters must be non-negative")
        with self._lock:
            self._depth, self._capacity, self._dropped = values

    def set_input_exhausted(self, exhausted: bool) -> None:
        """Publish whether the finite producer has no buffered input left."""
        with self._lock:
            self._input_exhausted = bool(exhausted)

    @property
    def input_exhausted(self) -> bool:
        with self._lock:
            return self._input_exhausted

    def __call__(self, _packet: SensorPacket, _observation: Any) -> IngressTelemetrySnapshot:
        with self._lock:
            depth, capacity, dropped = self._depth, self._capacity, self._dropped
        pressure = (
            "critical"
            if depth >= 2 * self._batch_size
            else "high"
            if depth >= self._batch_size
            else "low"
        )
        return IngressTelemetrySnapshot(
            evidence=(
                EvidenceFact("transport.queue_depth", depth),
                EvidenceFact("transport.queue_capacity", capacity),
                EvidenceFact("transport.dropped", dropped),
                EvidenceFact("pressure_state", pressure),
            )
        )


class _PacedPacketProducer(Generic[RecordT]):
    """Release a finite catalogue on its source clock with bounded decode."""

    def __init__(
        self,
        records: Sequence[RecordT],
        adapter: DatasetPacketAdapter[RecordT],
        *,
        source_hz: float,
        capacity: int,
        decode_workers: int = 2,
        ingestion_config: IngestionValueConfig | None = None,
    ) -> None:
        if float(source_hz) <= 0.0:
            raise ValueError("source_hz must be positive")
        if isinstance(capacity, bool) or int(capacity) < 1:
            raise ValueError("capacity must be positive")
        if isinstance(decode_workers, bool) or int(decode_workers) < 1:
            raise ValueError("decode_workers must be positive")
        if not isinstance(records, Sequence):
            raise TypeError("records must be a finite stable source catalogue")
        if ingestion_config is not None and ingestion_config.enabled:
            raise ValueError(
                "adaptive causal ingestion is incompatible with finite preservation"
            )
        self._records = records
        self._adapter = adapter
        self._decode_workers = int(decode_workers)
        self.source_hz = float(source_hz)
        self.capacity = int(capacity)
        self._queue: queue.Queue[SensorPacket] = queue.Queue(maxsize=self.capacity)
        self._done = threading.Event()
        self._initial_packet_or_terminal = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._offered = 0
        self._ingress = CausalIngressRecorder()
        self._high_watermark = 0
        self._first_offer_s: float | None = None
        self._last_offer_s: float | None = None
        self._futures: set[Future[SensorPacket]] = set()
        self._max_decode_backlog = 0
        self._max_publication_delay_s = 0.0
        self._drain_s = 0.0
        self._shutdown_incomplete = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("paced producer already started")
        self._thread = threading.Thread(
            target=self._run, name="canonical-causal-source", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        started_s = time.perf_counter()
        pending: deque[tuple[Future[SensorPacket], RecordT, float]] = deque()
        released = next_decode = 0
        decoder = ThreadPoolExecutor(
            max_workers=self._decode_workers,
            thread_name_prefix="canonical-causal-decode",
        )

        try:
            while not self._stop.is_set():
                now_s = time.perf_counter()
                while (
                    released < len(self._records)
                    and started_s + released / self.source_hz <= now_s
                ):
                    with self._lock:
                        self._offered += 1
                        self._first_offer_s = self._first_offer_s or now_s
                        self._last_offer_s = now_s
                    released += 1
                while pending and pending[0][0].done():
                    future, record, deadline_s = pending.popleft()
                    with _phase_timed("m_ingest_decode_wait"):
                        packet = future.result()
                    published_s = time.perf_counter()
                    with self._lock:
                        if self._stop.is_set():
                            return
                        self._queue.put_nowait(packet)
                        self._futures.discard(future)
                        self._ingress.record_admitted(record)
                        self._high_watermark = max(
                            self._high_watermark, self._queue.qsize()
                        )
                        self._max_publication_delay_s = max(
                            self._max_publication_delay_s, published_s - deadline_s
                        )
                    self._initial_packet_or_terminal.set()
                while (
                    next_decode < released
                    and len(pending) + self._queue.qsize() < self.capacity
                ):
                    with self._lock:
                        if self._stop.is_set():
                            return
                        future = decoder.submit(
                            self._adapter.packet_from_record,
                            self._records[next_decode],
                        )
                        self._futures.add(future)
                    pending.append((future, self._records[next_decode], started_s + next_decode / self.source_hz))
                    next_decode += 1
                with self._lock:
                    self._max_decode_backlog = max(
                        self._max_decode_backlog, released - next_decode
                    )
                if next_decode == len(self._records) and not pending:
                    self._drain_s = max(0.0, time.perf_counter() - started_s)
                    return
                next_deadline_s = (
                    started_s + released / self.source_hz
                    if released < len(self._records)
                    else None
                )
                wait_s = 0.001 if next_deadline_s is None else min(
                    0.001, max(0.0, next_deadline_s - time.perf_counter())
                )
                with _phase_timed("m_ingest_pace_wait"):
                    self._stop.wait(wait_s)
        except BaseException as exc:
            self._error = exc
            self._stop.set()
        finally:
            with self._lock:
                for future in self._futures:
                    future.cancel()
                self._shutdown_incomplete |= any(not future.done() for future in self._futures)
            decoder.shutdown(wait=False, cancel_futures=True)
            self._done.set()
            self._initial_packet_or_terminal.set()

    def take(self, *, timeout_s: float = 0.001) -> SensorPacket | None:
        if self._error is not None:
            raise RuntimeError("paced source producer failed") from self._error
        try:
            return self._queue.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError("paced source producer failed") from self._error
            return None

    def wait_for_initial_packet_or_terminal(self) -> None:
        self._initial_packet_or_terminal.wait()
        if self._error is not None:
            raise RuntimeError("paced source producer failed") from self._error

    @property
    def complete(self) -> bool:
        return self._done.is_set() and self._queue.empty() and self._error is None and not self._stop.is_set()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return 0

    @property
    def dropped_at_arrival(self) -> int:
        return 0

    @property
    def dropped_at_enqueue(self) -> int:
        return 0

    def source_ingress_report_block(self) -> dict:
        with self._lock:
            block = self._ingress.as_report_block()
            block["ingress_frames_offered"] = self._offered
            block["ingress_queue_capacity"] = self.capacity
            block["ingress_queue_high_watermark"] = self._high_watermark
            block["ingress_max_observed_buffer_lag_s"] = (
                float(self._high_watermark) / self.source_hz
            )
            block["ingestion_policy"] = "finite_preservation"
            block["ingress_decode_backlog_high_watermark"] = self._max_decode_backlog
            block["ingress_source_publication_delay_s"] = self._max_publication_delay_s
            block["ingress_drain_s"] = self._drain_s
            block["ingress_shutdown_incomplete"] = self._shutdown_incomplete
        return block

    @property
    def offered(self) -> int:
        with self._lock:
            return self._offered

    @property
    def high_watermark(self) -> int:
        with self._lock:
            return self._high_watermark

    @property
    def offered_fps(self) -> float | None:
        with self._lock:
            if self._offered < 2 or self._first_offer_s is None or self._last_offer_s is None:
                return None
            return (self._offered - 1) / max(
                self._last_offer_s - self._first_offer_s, 1e-9
            )

    def close(self) -> None:
        with self._lock:
            was_done = self._done.is_set()
            self._stop.set()
            for future in self._futures:
                future.cancel()
            self._shutdown_incomplete |= any(not future.done() for future in self._futures)
        if self._thread is not None:
            self._thread.join(timeout=0.1)
            self._shutdown_incomplete |= self._thread.is_alive()
        if not was_done and self._error is None:
            self._error = RuntimeError("paced source producer cancelled")


class PacedCausalReplayDriver(Generic[RecordT]):
    """Run the mapper clock continuously beside a bounded paced source."""

    def __init__(
        self,
        application: MappingApplication,
        adapter: DatasetPacketAdapter[RecordT],
        *,
        source_hz: float,
        source_buffer_frames: int = 256,
        decode_workers: int = 2,
        ingestion_config: IngestionValueConfig | None = None,
        ingress_telemetry: CausalIngressTelemetryState | None = None,
        tick_observer: TickObserver | None = None,
        tick_clock_observer: TickClockObserver | None = None,
        progress_interval_s: float = 0.0,
        progress_sink: Callable[[str], None] | None = None,
        packet_role: Callable[[SensorPacket], str] | None = None,
        bypass_packet_capture: Callable[[SensorPacket], object] | None = None,
        source_consumed_observer: Callable[[SensorPacket], None] | None = None,
        source_complete_observer: TickObserver | None = None,
        coverage_ledger: CoverageLedger | None = None,
        bootstrap_frames: int = 0,
        max_new_sources_per_tick: int = 1,
        max_source_packets_per_tick: int | None = None,
        semantic_drain_timeout_s: float = 30.0,
    ) -> None:
        self._application = application
        self._adapter = adapter
        self._source_hz = float(source_hz)
        self._source_buffer_frames = int(source_buffer_frames)
        if isinstance(decode_workers, bool) or int(decode_workers) < 1:
            raise ValueError("decode_workers must be a positive integer")
        self._decode_workers = int(decode_workers)
        self._ingestion_config = ingestion_config or IngestionValueConfig()
        self._ingress_telemetry = ingress_telemetry
        self._tick_observer = tick_observer
        if tick_clock_observer is not None and not callable(tick_clock_observer):
            raise TypeError("tick_clock_observer must be callable or None")
        self._tick_clock_observer = tick_clock_observer
        self._progress_interval_s = float(progress_interval_s)
        if not np.isfinite(self._progress_interval_s) or self._progress_interval_s < 0.0:
            raise ValueError("progress_interval_s must be finite and non-negative")
        if progress_sink is not None and not callable(progress_sink):
            raise TypeError("progress_sink must be callable or None")
        self._progress_sink = progress_sink
        if packet_role is not None and not callable(packet_role):
            raise TypeError("packet_role must be callable or None")
        if bypass_packet_capture is not None and not callable(bypass_packet_capture):
            raise TypeError("bypass_packet_capture must be callable or None")
        if source_consumed_observer is not None and not callable(source_consumed_observer):
            raise TypeError("source_consumed_observer must be callable or None")
        if source_complete_observer is not None and not callable(source_complete_observer):
            raise TypeError("source_complete_observer must be callable or None")
        self._packet_role = packet_role
        self._bypass_packet_capture = bypass_packet_capture
        self._source_consumed_observer = source_consumed_observer
        self._source_complete_observer = source_complete_observer
        self._coverage_ledger = coverage_ledger
        self._bootstrap_frames = max(0, int(bootstrap_frames))
        if (
            isinstance(max_new_sources_per_tick, bool)
            or int(max_new_sources_per_tick) < 1
        ):
            raise ValueError("max_new_sources_per_tick must be a positive integer")
        self._max_new_sources_per_tick = int(max_new_sources_per_tick)
        if max_source_packets_per_tick is None:
            max_source_packets_per_tick = self._max_new_sources_per_tick
        if (
            isinstance(max_source_packets_per_tick, bool)
            or int(max_source_packets_per_tick) < self._max_new_sources_per_tick
        ):
            raise ValueError(
                "max_source_packets_per_tick must be at least "
                "max_new_sources_per_tick"
            )
        self._max_source_packets_per_tick = int(max_source_packets_per_tick)
        self._semantic_drain_timeout_s = float(semantic_drain_timeout_s)
        if not np.isfinite(self._semantic_drain_timeout_s) or self._semantic_drain_timeout_s <= 0.0:
            raise ValueError("semantic_drain_timeout_s must be finite and positive")

    def run(self, records: Iterable[RecordT]) -> CausalReplaySummary:
        started_s = time.perf_counter()
        try:
            source_total = len(records)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            source_total = None
        producer = _PacedPacketProducer(
            records,
            self._adapter,
            source_hz=self._source_hz,
            capacity=self._source_buffer_frames,
            decode_workers=self._decode_workers,
            ingestion_config=self._ingestion_config,
        )
        producer.start()
        try:
            producer.wait_for_initial_packet_or_terminal()
        except BaseException:
            producer.close()
            raise
        sources: list[SourceIdentity] = []
        submitted = ticks = optimizer_steps = supervision_views = zero_frame_ticks = 0
        admitted_new_total = 0
        camera_materialize_cache_hits = camera_materialize_cache_misses = 0
        prepared_insertions_executed = prepared_insertion_seeds_accepted = 0
        prepared_insertions_zero_seed = 0
        prepared_insertion_skips: Counter[str] = Counter()
        torch_profile_window = TickTorchProfileWindow()
        holdout_reserved = guard_withheld = 0
        last_ingest_iteration = None
        drain_without_progress = 0
        no_drain_progress_since_s: float | None = None
        source_complete_notified = False
        submission_s = 0.0
        mapping_active_s = 0.0
        outer_loop_wall_s = 0.0
        outer_take_wait_s = 0.0
        outer_tick_total_s = 0.0
        outer_bookkeeping_s = 0.0
        requested_constellations: Counter[str] = Counter()
        realized_constellations: Counter[str] = Counter()
        constellation_modes: Counter[str] = Counter()
        constellation_reasons: Counter[str] = Counter()
        next_progress_s = started_s + self._progress_interval_s
        requested_new_slots = 1
        prepared_mode = bool(getattr(self._application.session, "prepared_mode", False))
        previous_tick_can_progress = False
        source_frontier: SourceIdentity | None = None
        pending_packet: SensorPacket | None = None
        drain = _MappingDrainTracker()
        try:
            while True:
                loop_iter_started_s = time.perf_counter()
                pre_drain_depth = producer.queue_depth
                drain_target = 1
                if (
                    requested_new_slots > 1
                    and pre_drain_depth >= requested_new_slots
                ):
                    drain_target = min(
                        self._max_new_sources_per_tick,
                        requested_new_slots,
                    )
                scan_cap = (
                    1
                    if drain_target == 1
                    else self._max_source_packets_per_tick
                )
                drained_packets = []
                admitted_new = 0
                for scan_index in range(scan_cap):
                    retrying = pending_packet is not None
                    if retrying:
                        packet = pending_packet
                    else:
                        take_started_s = time.perf_counter()
                        with _phase_timed("m_outer_take_wait"):
                            packet = producer.take(
                                timeout_s=(
                                    0.0
                                    if prepared_mode and previous_tick_can_progress
                                    else 0.001
                                    if scan_index == 0
                                    else 0.0
                                )
                            )
                        outer_take_wait_s += time.perf_counter() - take_started_s
                    if packet is None:
                        break
                    if not retrying and len(drained_packets) == 0 and self._ingress_telemetry is not None:
                        bookkeeping_started_s = time.perf_counter()
                        with _phase_timed("m_outer_ingress_publish"):
                            published_depth = max(
                                pre_drain_depth,
                                producer.queue_depth + 1,
                            )
                            self._ingress_telemetry.publish(
                                queue_depth=published_depth,
                                capacity=producer.capacity,
                                dropped=producer.dropped,
                            )
                        outer_bookkeeping_s += time.perf_counter() - bookkeeping_started_s
                    role = "train" if self._packet_role is None else self._packet_role(packet)
                    if role == "train":
                        submit_started_s = time.perf_counter()
                        try:
                            with _phase_timed("m_outer_submit"):
                                outcome = self._application.session.submit(packet)
                        except PreparedIngressDeferred:
                            pending_packet = packet
                            break
                        submission_s += time.perf_counter() - submit_started_s
                        pending_packet = None
                        sources.append(outcome.event.source)
                        submitted += 1
                        drain.record_submit(outcome)
                        disposition = getattr(outcome, "disposition", None)
                        if disposition is not None:
                            admitted = bool(
                                getattr(disposition, "admit_supervision", False)
                            )
                        elif getattr(outcome, "frame", None) is not None:
                            admitted = True
                        else:
                            admitted = bool(getattr(outcome, "accepted", True))
                        admitted_new += int(admitted)
                        admitted_new_total += int(admitted)
                        _record_causal_coverage(
                            self._coverage_ledger,
                            source=outcome.event.source,
                            role="train",
                            admitted=admitted,
                            bootstrap_frames=self._bootstrap_frames,
                        )
                    elif role in {"holdout", "guard"}:
                        source_packet = packet
                        if self._bypass_packet_capture is not None:
                            try:
                                captured = self._bypass_packet_capture(packet, role=role)
                                if not isinstance(captured, SensorPacket):
                                    raise TypeError("bypass_packet_capture must return SensorPacket")
                                packet = captured
                            except PreparedIngressDeferred:
                                pending_packet = packet
                                break
                        pending_packet = None
                        if self._source_consumed_observer is not None:
                            self._source_consumed_observer(source_packet)
                        if role == "holdout":
                            holdout_reserved += 1
                        else:
                            guard_withheld += 1
                        _record_causal_coverage(
                            self._coverage_ledger,
                            source=packet.source,
                            role=role,
                            admitted=False,
                            bootstrap_frames=self._bootstrap_frames,
                        )
                    else:
                        raise ValueError(f"unknown causal packet role {role!r}")
                    drained_packets.append(packet)
                    if source_frontier is None or (
                        packet.source.ordering_key() > source_frontier.ordering_key()
                    ):
                        source_frontier = packet.source
                    if admitted_new >= drain_target:
                        break
                if not drained_packets:
                    zero_frame_ticks += 1
                    if self._ingress_telemetry is not None:
                        bookkeeping_started_s = time.perf_counter()
                        with _phase_timed("m_outer_ingress_publish"):
                            self._ingress_telemetry.publish(
                                queue_depth=producer.queue_depth,
                                capacity=producer.capacity,
                                dropped=producer.dropped,
                            )
                        outer_bookkeeping_s += time.perf_counter() - bookkeeping_started_s
                if prepared_mode and previous_tick_can_progress and not drained_packets:
                    wait_for_progress = getattr(
                        self._application, "wait_for_prepared_progress", None
                    )
                    if callable(wait_for_progress):
                        wait_for_progress(timeout_s=READY_WAIT_FALLBACK_S)
                if self._ingress_telemetry is not None:
                    self._ingress_telemetry.set_input_exhausted(producer.complete)
                tick_started_s = time.perf_counter()
                with _phase_timed("m_outer_tick_total"), torch_profile_window.tick():
                    result = self._application.tick()
                tick_elapsed_s = time.perf_counter() - tick_started_s
                outer_tick_total_s += tick_elapsed_s
                ticks += 1
                completed = int(getattr(result, "optimizer_steps", 0))
                previous_tick_can_progress = bool(
                    getattr(result, "executed", completed > 0) or getattr(result, "retryable", False)
                )
                optimizer_steps += completed
                supervision_views += int(getattr(result, "supervision_views", 0))
                camera_materialize_cache_hits += int(
                    getattr(result, "camera_materialize_cache_hits", 0)
                )
                camera_materialize_cache_misses += int(
                    getattr(result, "camera_materialize_cache_misses", 0)
                )
                executed, seeds, zero_seed, skip = _extract_prepared_insertion_counters(result)
                prepared_insertions_executed += executed
                prepared_insertion_seeds_accepted += seeds
                prepared_insertions_zero_seed += zero_seed
                if skip is not None:
                    prepared_insertion_skips[skip] += 1
                constellation = getattr(
                    getattr(result, "work_plan", None),
                    "supervision_constellation",
                    None,
                )
                if completed > 0 and constellation is not None:
                    requested_constellations[
                        f"B{int(constellation.requested_batch_size)}_"
                        f"C{int(constellation.requested_camera_count)}"
                    ] += completed
                    realized_constellations[
                        f"B{int(constellation.realized_batch_size)}_"
                        f"C{int(constellation.realized_camera_count)}"
                    ] += completed
                    constellation_modes[str(constellation.mode)] += completed
                    constellation_reasons[str(constellation.reason)] += completed
                bookkeeping_started_s = time.perf_counter()
                with _phase_timed("m_outer_drain_bookkeeping"):
                    pending_before_tick = drain.pending_new_sources
                    drained_pending_source = drain.record_tick(result)
                    work_plan = getattr(result, "work_plan", None)
                    requested_new_slots = max(
                        1,
                        min(
                            self._max_new_sources_per_tick,
                            int(getattr(work_plan, "requested_supervision_slots", 1)),
                        ),
                    )
                outer_bookkeeping_s += time.perf_counter() - bookkeeping_started_s
                if completed > 0:
                    mapping_active_s += tick_elapsed_s
                if self._tick_clock_observer is not None:
                    self._tick_clock_observer(
                        result,
                        CausalReplayClock(
                            optimizer_steps=optimizer_steps,
                            supervision_views=supervision_views,
                            mapping_active_s=mapping_active_s,
                            elapsed_s=time.perf_counter() - started_s,
                            source_frame_index=(
                                None
                                if source_frontier is None
                                else source_frontier.source_ordinal
                            ),
                        ),
                    )
                if self._tick_observer is not None:
                    self._tick_observer(result)
                if drained_packets:
                    last_ingest_iteration = int(
                        getattr(result, "iteration", optimizer_steps)
                    )
                bookkeeping_started_s = time.perf_counter()
                with _phase_timed("m_outer_drain_bookkeeping"):
                    producer_complete = producer.complete
                    mapping_drained = bool(
                        producer_complete
                        and not drained_packets
                        and drain.reached_barrier(result, submitted=submitted)
                    )
                    if mapping_drained:
                        if (
                            self._source_complete_observer is not None
                            and not source_complete_notified
                        ):
                            self._source_complete_observer(result)
                            source_complete_notified = True
                    elif (
                        producer_complete
                        and not drained_packets
                        and not source_complete_notified
                        and self._source_complete_observer is not None
                        and _prepared_bootstrap_pending(result)
                    ):
                        self._source_complete_observer(result)
                        source_complete_notified = True
                outer_bookkeeping_s += time.perf_counter() - bookkeeping_started_s
                now_s = time.perf_counter()
                if (
                    self._progress_sink is not None
                    and self._progress_interval_s > 0.0
                    and now_s >= next_progress_s
                ):
                    bookkeeping_started_s = now_s
                    with _phase_timed("m_outer_progress_sink"):
                        active_s = max(mapping_active_s, 1e-9)
                        source_progress = (
                            str(producer.offered)
                            if source_total is None
                            else f"{producer.offered}/{source_total}"
                        )
                        self._progress_sink(
                            "[causal-progress] "
                            f"elapsed_s={now_s - started_s:.1f} "
                            f"source_offered={source_progress} "
                            f"source_submitted={submitted} "
                            f"queue_depth={producer.queue_depth} "
                            f"optimizer_steps={optimizer_steps} "
                            f"supervision_views={supervision_views} "
                            f"optimizer_steps_per_s={optimizer_steps / active_s:.2f} "
                            f"supervision_views_per_s={supervision_views / active_s:.2f} "
                            f"transport_dropped={producer.dropped}"
                            f"{_prepared_ready_progress_fields(result)}"
                        )
                        next_progress_s = now_s + self._progress_interval_s
                    outer_bookkeeping_s += time.perf_counter() - bookkeeping_started_s
                loop_iter_elapsed_s = time.perf_counter() - loop_iter_started_s
                outer_loop_wall_s += loop_iter_elapsed_s
                if mapping_drained:
                    break
                if producer_complete and not drained_packets:
                    if no_drain_progress_since_s is None or drained_pending_source:
                        no_drain_progress_since_s = now_s
                    elif (
                        prepared_mode
                        and now_s - no_drain_progress_since_s
                        >= self._semantic_drain_timeout_s
                    ):
                        state = getattr(result, "state", None)
                        raise RuntimeError(
                            "paced causal replay made no semantic progress for "
                            f"{self._semantic_drain_timeout_s:g}s: "
                            f"state={getattr(state, 'value', state)!r} "
                            f"reason={getattr(result, 'reason', None)!r} "
                            f"executed={bool(getattr(result, 'executed', False))} "
                            f"optimizer_steps={int(getattr(result, 'optimizer_steps', 0))} "
                            f"pending_new_sources={drain.pending_new_sources} "
                            f"pending_sources={[source.fingerprint() for source in drain.pending_sources]} "
                            f"terminal_submissions={drain.terminal_submissions} "
                            f"submitted={submitted} offered={producer.offered} dropped={producer.dropped}"
                        )
                    if drain.pending_new_sources >= pending_before_tick:
                        drain_without_progress += 1
                    else:
                        drain_without_progress = 0
                    if drain_without_progress > 100_000:
                        raise RuntimeError(
                            "paced causal replay cannot reach its semantic mapping "
                            "drain barrier"
                        )
                else:
                    drain_without_progress = 0
                    no_drain_progress_since_s = None
        finally:
            producer.close()
        elapsed_s = time.perf_counter() - started_s
        return CausalReplaySummary(
            submitted=submitted,
            ticks=ticks,
            source_identities=tuple(sources),
            coverage_ledger=(
                None
                if self._coverage_ledger is None
                else self._coverage_ledger.snapshot()
            ),
            optimizer_steps=optimizer_steps,
            supervision_views=supervision_views,
            admitted_new_total=admitted_new_total,
            zero_frame_ticks=zero_frame_ticks,
            transport_dropped=producer.dropped,
            transport_dropped_at_arrival=producer.dropped_at_arrival,
            transport_dropped_at_enqueue=producer.dropped_at_enqueue,
            source_ingress=producer.source_ingress_report_block(),
            source_queue_high_watermark=producer.high_watermark,
            source_offered_fps=producer.offered_fps,
            elapsed_s=elapsed_s,
            submission_mean_ms=(
                0.0 if submitted == 0 else 1000.0 * submission_s / submitted
            ),
            mapping_active_s=mapping_active_s,
            optimizer_steps_per_active_s=(
                0.0 if mapping_active_s <= 0.0 else optimizer_steps / mapping_active_s
            ),
            supervision_views_per_active_s=(
                0.0
                if mapping_active_s <= 0.0
                else supervision_views / mapping_active_s
            ),
            supervision_views_per_elapsed_s=(
                0.0 if elapsed_s <= 0.0 else supervision_views / elapsed_s
            ),
            source_offered=producer.offered,
            holdout_reserved=holdout_reserved,
            guard_withheld=guard_withheld,
            last_ingest_iteration=last_ingest_iteration,
            requested_supervision_constellations=dict(
                sorted(requested_constellations.items())
            ),
            realized_supervision_constellations=dict(
                sorted(realized_constellations.items())
            ),
            supervision_constellation_modes=dict(
                sorted(constellation_modes.items())
            ),
            supervision_constellation_reasons=dict(
                sorted(constellation_reasons.items())
            ),
            outer_loop_wall_s=outer_loop_wall_s,
            outer_take_wait_s=outer_take_wait_s,
            outer_submit_s=submission_s,
            outer_tick_total_s=outer_tick_total_s,
            outer_bookkeeping_s=outer_bookkeeping_s,
            outer_unaccounted_s=max(
                0.0,
                outer_loop_wall_s
                - outer_take_wait_s
                - submission_s
                - outer_tick_total_s
                - outer_bookkeeping_s,
            ),
            camera_materialize_cache_hits=camera_materialize_cache_hits,
            camera_materialize_cache_misses=camera_materialize_cache_misses,
            ready_bytes_peak=0,
            ready_bytes_current_at_end=0,
            ready_bytes_by_tier_kind={},
            prepared_insertions_executed=prepared_insertions_executed,
            prepared_insertion_seeds_accepted=prepared_insertion_seeds_accepted,
            prepared_insertions_zero_seed=prepared_insertions_zero_seed,
            prepared_insertion_skips=dict(sorted(prepared_insertion_skips.items())),
        )


@dataclass(frozen=True, slots=True)
class CausalReplayTickPolicy:
    """Source-pacing-only tick schedule for deterministic replay harnesses.

    The policy can add per-submission ticks, but it never carries
    admission, replay, resolution, insertion, or mutation choices. Those
    remain inside the canonical application and its control owners.
    """

    ticks_per_submission: int = 1
    def __post_init__(self) -> None:
        for name in ("ticks_per_submission",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class CausalReplayDriver(Generic[RecordT]):
    """Own source pacing only; all mapping decisions stay inside the application."""

    def __init__(
        self,
        application: MappingApplication,
        adapter: DatasetPacketAdapter[RecordT],
        *,
        tick_policy: CausalReplayTickPolicy = CausalReplayTickPolicy(),
        tick_observer: TickObserver | None = None,
        packet_role: Callable[[SensorPacket], str] | None = None,
        bypass_packet_capture: Callable[[SensorPacket], object] | None = None,
        source_consumed_observer: Callable[[SensorPacket], None] | None = None,
        source_complete_observer: TickObserver | None = None,
        coverage_ledger: CoverageLedger | None = None,
        bootstrap_frames: int = 0,
    ) -> None:
        self._application = application
        self._adapter = adapter
        if not isinstance(tick_policy, CausalReplayTickPolicy):
            raise TypeError("tick_policy must be CausalReplayTickPolicy")
        self._tick_policy = tick_policy
        if tick_observer is not None and not callable(tick_observer):
            raise TypeError("tick_observer must be callable or None")
        if packet_role is not None and not callable(packet_role):
            raise TypeError("packet_role must be callable or None")
        if bypass_packet_capture is not None and not callable(bypass_packet_capture):
            raise TypeError("bypass_packet_capture must be callable or None")
        if source_consumed_observer is not None and not callable(source_consumed_observer):
            raise TypeError("source_consumed_observer must be callable or None")
        if source_complete_observer is not None and not callable(source_complete_observer):
            raise TypeError("source_complete_observer must be callable or None")
        self._tick_observer = tick_observer
        self._packet_role = packet_role
        self._bypass_packet_capture = bypass_packet_capture
        self._source_consumed_observer = source_consumed_observer
        self._source_complete_observer = source_complete_observer
        self._coverage_ledger = coverage_ledger
        self._bootstrap_frames = max(0, int(bootstrap_frames))

    def _tick(self) -> None:
        result = self._application.tick()
        if self._tick_observer is not None:
            self._tick_observer(result)

    def run(self, records: Iterable[RecordT]) -> CausalReplaySummary:
        started_s = time.perf_counter()
        sources: list[SourceIdentity] = []
        submitted = 0
        admitted_new_total = 0
        ticks = 0
        optimizer_steps = supervision_views = 0
        mapping_active_s = 0.0
        holdout_reserved = guard_withheld = 0
        last_result = None
        last_ingest_iteration = None
        drain = _MappingDrainTracker()
        outer_loop_wall_s = 0.0
        outer_decode_s = 0.0
        outer_submit_s = 0.0
        outer_tick_total_s = 0.0
        outer_real_ticks = 0
        outer_drain_ticks = 0
        outer_real_tick_s = 0.0
        outer_drain_tick_s = 0.0
        camera_materialize_cache_hits = camera_materialize_cache_misses = 0
        prepared_insertions_executed = prepared_insertion_seeds_accepted = 0
        prepared_insertions_zero_seed = 0
        prepared_insertion_skips: Counter[str] = Counter()
        torch_profile_window = TickTorchProfileWindow()
        pending_packet: SensorPacket | None = None

        def _run_tick() -> Any:
            nonlocal ticks, optimizer_steps, supervision_views, mapping_active_s
            nonlocal outer_tick_total_s, outer_real_ticks, outer_drain_ticks
            nonlocal outer_real_tick_s, outer_drain_tick_s
            nonlocal camera_materialize_cache_hits, camera_materialize_cache_misses
            nonlocal prepared_insertions_executed, prepared_insertion_seeds_accepted
            nonlocal prepared_insertions_zero_seed, prepared_insertion_skips
            nonlocal torch_profile_window
            tick_started_s = time.perf_counter()
            with _phase_timed("m_outer_tick_total"), torch_profile_window.tick():
                result = self._application.tick()
            tick_elapsed_s = time.perf_counter() - tick_started_s
            outer_tick_total_s += tick_elapsed_s
            if self._tick_observer is not None:
                self._tick_observer(result)
            drain.record_tick(result)
            ticks += 1
            completed = int(getattr(result, "optimizer_steps", 0))
            optimizer_steps += completed
            supervision_views += int(getattr(result, "supervision_views", 0))
            camera_materialize_cache_hits += int(
                getattr(result, "camera_materialize_cache_hits", 0)
            )
            camera_materialize_cache_misses += int(
                getattr(result, "camera_materialize_cache_misses", 0)
            )
            executed, seeds, zero_seed, skip = _extract_prepared_insertion_counters(result)
            prepared_insertions_executed += executed
            prepared_insertion_seeds_accepted += seeds
            prepared_insertions_zero_seed += zero_seed
            if skip is not None:
                prepared_insertion_skips[skip] += 1
            if completed > 0:
                mapping_active_s += tick_elapsed_s
                outer_real_ticks += 1
                outer_real_tick_s += tick_elapsed_s
                _phase_record_timed("m_outer_tick_real", tick_elapsed_s)
            else:
                outer_drain_ticks += 1
                outer_drain_tick_s += tick_elapsed_s
                _phase_record_timed("m_outer_tick_drain", tick_elapsed_s)
            return result

        records_iter = iter(records)
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="causal-replay-decode"
        ) as decode_executor:
            try:
                next_record = next(records_iter)
            except StopIteration:
                next_record = None
                next_future = None
            else:
                next_future = decode_executor.submit(
                    self._adapter.packet_from_record, next_record
                )

            while next_future is not None or pending_packet is not None:
                loop_iter_started_s = time.perf_counter()
                if pending_packet is None:
                    future = next_future
                    assert future is not None
                    try:
                        next_record = next(records_iter)
                    except StopIteration:
                        next_future = None
                    else:
                        next_future = decode_executor.submit(
                            self._adapter.packet_from_record, next_record
                        )

                    decode_started_s = time.perf_counter()
                    with _phase_timed("m_ingest_decode"):
                        packet = future.result()
                    outer_decode_s += time.perf_counter() - decode_started_s
                else:
                    packet = pending_packet
                role = "train" if self._packet_role is None else self._packet_role(packet)
                if role == "train":
                    submit_started_s = time.perf_counter()
                    try:
                        with _phase_timed("m_outer_submit"):
                            outcome = self._application.session.submit(packet)
                    except PreparedIngressDeferred:
                        pending_packet = packet
                        last_result = _run_tick()
                        outer_loop_wall_s += time.perf_counter() - loop_iter_started_s
                        continue
                    outer_submit_s += time.perf_counter() - submit_started_s
                    pending_packet = None
                    sources.append(outcome.event.source)
                    submitted += 1
                    drain.record_submit(outcome)
                    disposition = getattr(outcome, "disposition", None)
                    if disposition is not None:
                        admitted = bool(
                            getattr(disposition, "admit_supervision", False)
                        )
                    elif getattr(outcome, "frame", None) is not None:
                        admitted = True
                    else:
                        admitted = bool(getattr(outcome, "accepted", True))
                    admitted_new_total += int(admitted)
                    _record_causal_coverage(
                        self._coverage_ledger,
                        source=outcome.event.source,
                        role="train",
                        admitted=admitted,
                        bootstrap_frames=self._bootstrap_frames,
                    )
                elif role in {"holdout", "guard"}:
                    source_packet = packet
                    if self._bypass_packet_capture is not None:
                        try:
                            captured = self._bypass_packet_capture(packet, role=role)
                            if not isinstance(captured, SensorPacket):
                                raise TypeError("bypass_packet_capture must return SensorPacket")
                            packet = captured
                        except PreparedIngressDeferred:
                            pending_packet = packet
                            last_result = _run_tick()
                            outer_loop_wall_s += time.perf_counter() - loop_iter_started_s
                            continue
                    pending_packet = None
                    if self._source_consumed_observer is not None:
                        self._source_consumed_observer(source_packet)
                    if role == "holdout":
                        holdout_reserved += 1
                    else:
                        guard_withheld += 1
                    _record_causal_coverage(
                        self._coverage_ledger,
                        source=packet.source,
                        role=role,
                        admitted=False,
                        bootstrap_frames=self._bootstrap_frames,
                    )
                else:
                    raise ValueError(f"unknown causal packet role {role!r}")
                for _ in range(self._tick_policy.ticks_per_submission):
                    last_result = _run_tick()
                if last_result is not None:
                    last_ingest_iteration = int(
                        getattr(last_result, "iteration", ticks)
                    )
                outer_loop_wall_s += time.perf_counter() - loop_iter_started_s
        drain_without_progress = 0
        source_complete_notified = False
        while True:
            loop_iter_started_s = time.perf_counter()
            pending_before_tick = drain.pending_new_sources
            last_result = _run_tick()
            reached_barrier = drain.reached_barrier(last_result, submitted=submitted)
            outer_loop_wall_s += time.perf_counter() - loop_iter_started_s
            if (
                not reached_barrier
                and not source_complete_notified
                and self._source_complete_observer is not None
                and _prepared_bootstrap_pending(last_result)
            ):
                self._source_complete_observer(last_result)
                source_complete_notified = True
            if reached_barrier:
                break
            if drain.pending_new_sources >= pending_before_tick:
                drain_without_progress += 1
            else:
                drain_without_progress = 0
            if drain_without_progress > 100_000:
                raise RuntimeError(
                    "causal replay cannot reach its semantic mapping drain barrier"
                )
        if self._source_complete_observer is not None and not source_complete_notified:
            self._source_complete_observer(last_result)
        elapsed_s = time.perf_counter() - started_s
        return CausalReplaySummary(
            submitted=submitted,
            ticks=ticks,
            source_identities=tuple(sources),
            coverage_ledger=(
                None
                if self._coverage_ledger is None
                else self._coverage_ledger.snapshot()
            ),
            optimizer_steps=optimizer_steps,
            supervision_views=supervision_views,
            admitted_new_total=admitted_new_total,
            elapsed_s=elapsed_s,
            mapping_active_s=mapping_active_s,
            optimizer_steps_per_active_s=(
                0.0 if mapping_active_s <= 0.0 else optimizer_steps / mapping_active_s
            ),
            supervision_views_per_active_s=(
                0.0
                if mapping_active_s <= 0.0
                else supervision_views / mapping_active_s
            ),
            supervision_views_per_elapsed_s=(
                0.0 if elapsed_s <= 0.0 else supervision_views / elapsed_s
            ),
            source_offered=submitted + holdout_reserved + guard_withheld,
            holdout_reserved=holdout_reserved,
            guard_withheld=guard_withheld,
            last_ingest_iteration=last_ingest_iteration,
            outer_loop_wall_s=outer_loop_wall_s,
            outer_submit_s=outer_submit_s,
            outer_tick_total_s=outer_tick_total_s,
            outer_decode_s=outer_decode_s,
            outer_real_ticks=outer_real_ticks,
            outer_drain_ticks=outer_drain_ticks,
            outer_real_tick_s=outer_real_tick_s,
            outer_drain_tick_s=outer_drain_tick_s,
            outer_unaccounted_s=max(
                0.0,
                outer_loop_wall_s
                - outer_decode_s
                - outer_submit_s
                - outer_tick_total_s,
            ),
            camera_materialize_cache_hits=camera_materialize_cache_hits,
            camera_materialize_cache_misses=camera_materialize_cache_misses,
            ready_bytes_peak=0,
            ready_bytes_current_at_end=0,
            ready_bytes_by_tier_kind={},
            prepared_insertions_executed=prepared_insertions_executed,
            prepared_insertion_seeds_accepted=prepared_insertion_seeds_accepted,
            prepared_insertions_zero_seed=prepared_insertions_zero_seed,
            prepared_insertion_skips=dict(sorted(prepared_insertion_skips.items())),
        )


__all__ = [
    "CausalIngressTelemetryState",
    "CausalReplayClock",
    "CausalReplayDriver",
    "CausalReplaySummary",
    "CausalReplayTickPolicy",
    "DatasetPacketAdapter",
    "JsonlDecisionTraceObserver",
    "PacedCausalReplayDriver",
    "StreamingFramePacketAdapter",
    "decision_trace_row",
]
