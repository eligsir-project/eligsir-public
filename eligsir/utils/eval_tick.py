"""Issue #698: ONE canonical per-evaluation-tick record.

What this is
------------
The mandatory holdout PSNR tick (``utils.psnr_history.record_and_plot``, run
from ``train_streaming.py``) already renders the whole test-holdout set once
and returns the **GPU-resident per-view PSNR tensor** in
:class:`~utils.psnr_history.PsnrTickResult`. Everything persisted from that
tick, until now, was::

    {"iteration": 17000, "mean_test_psnr": 20.7}

This module persists the full record instead — global PSNR, the longitudinal
**temporal-partition** PSNR vector derived from *that same render*, and the
causal clocks (source progress / wall clock / supervision work) joined at the
exact tick — into ``evaluation_ticks.jsonl``.

Hard constraints honored here
-----------------------------
* **No second holdout render.** This module never renders anything: it
  consumes a ``PsnrTickResult`` the caller already has. The only renderer in
  sight is the one ``record_and_plot`` already ran.
* **No per-view ``.item()`` loop.** The partition reduction is a
  ``scatter_add_`` on-device, and every value this module needs is moved to
  the host in **exactly one** ``.cpu()`` call per tick (sums, counts and the
  per-view vector are concatenated first — see :func:`reduce_partition_psnr`).
* **No fabricated fields.** Anything not resolvable is written as ``null``
  with a machine-readable reason in the record's ``field_provenance`` block,
  mirroring ``utils/experiment_validity.py``'s convention. In particular
  ``source_progress`` is emitted **only** against a declared registry
  denominator (``eval/scenes.json``'s ``n_source_frames``) — never
  renormalized against the realized/truncated prefix, per the issue's
  correctness rules.
* **No training behavior change.** Nothing here mutates the model, the
  cameras, the sampling weights, or which cameras are scored.

What was reused vs. added
-------------------------
Reused, not re-implemented:

* ``utils.partition_eval`` — ``resolve_max_source_index`` (the declared-vs-
  realized denominator resolution *and* its ``denominator_source`` label),
  the canonical ``PARTITIONS``/``partition_id`` scheme,
  ``partition_index_tensor``, ``camera_source_index``,
  ``registry_n_source_frames``.
* ``utils.baseline_comparison._realized_holdout_hash`` — the same holdout
  identity fingerprint ``report.json``'s ``eval_metadata`` uses (imported
  from the torch-free module, not via ``utils.comparison_report``).
* ``utils.psnr_history`` — the tick, its render, and its mean PSNR. The
  legacy ``psnr_history.jsonl`` keeps being written by that module,
  unchanged; :func:`psnr_history_from_eval_ticks` reproduces it from the new
  record (acceptance criterion 7).

Added here: the record schema, the on-device partition reduction, the JSONL
writer/readers and the longitudinal partition plot.

Clock-field naming is deliberate: ``source_progress`` / ``wall_clock_s`` /
``supervision_views`` / ``iteration`` are named exactly as
``utils.causal_quality.TickObservation``'s axis coordinates, so #428 can join
this timeline without a translation table. This module does **not** compute
CVQ — that needs ``local_quality``'s per-cell accumulators, which are opt-in
and owned elsewhere.

Torch is imported lazily and only by :func:`reduce_partition_psnr`; every
other function here (record assembly, JSONL IO, plotting, the CPU reference
reduction) is torch-free so it stays collectable and testable on CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eligsir.plot_style import svg_output_path
from eligsir.utils.partition_eval import (
    PARTITIONS,
    camera_source_index,
    partition_id,
    registry_n_source_frames,
    resolve_max_source_index,
)
from eligsir.utils.psnr_history_io import PSNR_REDUCTION_RGB_PER_VIEW_V2, psnr_reduction

__all__ = [
    "CANONICAL_PARTITION_SCHEMA",
    "DIAGNOSTIC_PARTITION_SCHEMA",
    "EVAL_TICK_SCHEMA_VERSION",
    "EVAL_TICK_FILENAME",
    "RECORD_HOT_PATH_ELAPSED_SCOPE",
    "PartitionReduction",
    "reduce_partition_psnr",
    "reduce_partition_psnr_cpu",
    "partition_summary",
    "resolve_source_progress",
    "derive_phase",
    "build_eval_tick_record",
    "append_eval_tick",
    "read_eval_ticks",
    "psnr_history_from_eval_ticks",
    "plot_partition_psnr_over_time",
    "record_eval_tick",
    "record_final_report_eval_tick",
    "finalize_eval_tick_history",
]

#: Bump the MINOR on any additive field; the MAJOR on any removal/retype.
EVAL_TICK_SCHEMA_VERSION = "1.5.0"

# Evaluation-tick history is compared with the post-training report, whose
# canonical temporal aggregation is the fixed six-way ``PARTITIONS`` scheme.
# A caller may still request another ``partition_k`` for a diagnostic row, but
# that choice is stamped so plots/readers cannot silently compare unlike bins.
CANONICAL_PARTITION_SCHEMA = "canonical_report_partitions_v1"
DIAGNOSTIC_PARTITION_SCHEMA = "explicit_diagnostic_partitions_v1"

EVAL_TICK_FILENAME = "evaluation_ticks.jsonl"

# This is intentionally not named "tick latency": it excludes JSON encoding,
# the file append, and plotting.  It begins immediately before the compact
# same-render partition reduction, so it includes that reduction's one D2H
# sync, then ends after record assembly.  Measuring through the append while
# also persisting the elapsed value in that same append would require a
# self-referential rewrite and would make the append-only log dishonest.
RECORD_HOT_PATH_ELAPSED_SCOPE = (
    "same_render_partition_reduce_and_record_build_before_serialization"
)

#: Metric provenance stamped into every row, so a reader never has to guess
#: which quantity was reduced or where it came from.
METRIC_PROVENANCE = {
    "metric": "psnr_db",
    "source": "utils.psnr_history.PsnrTickResult.psnr_tensor",
    "reduction": "per-view psnr(...).mean() -> scatter_add_ segment mean per partition",
    "render_pass": "shared_with_psnr_history_tick",
    "extra_renders": 0,
}


# ---------------------------------------------------------------------------
# Partition reduction
# ---------------------------------------------------------------------------
@dataclass
class PartitionReduction:
    """One tick's partition breakdown, fully host-side (already synced)."""

    k: int
    max_source_index: int
    denominator_source: str
    #: views per partition, length ``k``
    counts: List[int]
    #: mean PSNR per partition, ``None`` for partitions with zero views
    mean_db: List[Optional[float]]
    #: per-view PSNR, index-aligned with ``view_source_indices``
    view_psnr_db: List[float]
    #: per-view source-frame index (``None`` when unresolvable for that cam)
    view_source_indices: List[Optional[int]]
    #: views whose source index could not be resolved (bucketed into p0 by
    #: ``partition_index_tensor``; recorded so the row is honest about it)
    n_views_unresolved_source_index: int = 0


def _round(x: Optional[float], nd: int = 4) -> Optional[float]:
    if x is None:
        return None
    xf = float(x)
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, nd)


def reduce_partition_psnr(psnr_tensor, cams: Sequence, max_source_index: int,
                          k: int, denominator_source: str = "unknown") -> PartitionReduction:
    """On-device segment reduction of an existing tick's per-view PSNR tensor.

    ``psnr_tensor[i]`` must correspond to ``cams[i]`` (exactly the
    ``PsnrTickResult`` contract). Partition ids come from
    ``partition_eval.partition_index_tensor`` — the SAME assignment the
    report and the partition-focus health check use.

    **Exactly one host sync**: sums, counts and the per-view vector are
    concatenated into one tensor before the single ``.cpu()`` call. There is
    no per-view ``.item()``.
    """
    import torch  # local: keeps this module importable without torch

    n = int(psnr_tensor.shape[0])
    if n != len(cams):
        raise ValueError(
            f"psnr_tensor length {n} != len(cams) {len(cams)}; the tick's "
            "per-view tensor and camera list must stay index-aligned"
        )
    from eligsir.utils.partition_eval import partition_index_tensor

    device = psnr_tensor.device
    pidx = partition_index_tensor(cams, max_source_index, k=k, device=device)
    vals = psnr_tensor.detach().to(torch.float32).reshape(-1)
    counts = torch.zeros(k, dtype=torch.float32, device=device)
    counts.scatter_add_(0, pidx, torch.ones_like(vals))
    sums = torch.zeros(k, dtype=torch.float32, device=device)
    sums.scatter_add_(0, pidx, vals)
    fused = torch.cat([sums, counts, vals]).cpu().tolist()  # <- the one sync

    sums_l = fused[0:k]
    counts_l = fused[k: 2 * k]
    vals_l = fused[2 * k:]
    counts_i = [int(round(c)) for c in counts_l]
    mean_db: List[Optional[float]] = [
        (_round(sums_l[i] / counts_l[i]) if counts_l[i] > 0 else None) for i in range(k)
    ]
    src_idx = [camera_source_index(c) for c in cams]
    return PartitionReduction(
        k=int(k),
        max_source_index=int(max_source_index),
        denominator_source=str(denominator_source),
        counts=counts_i,
        mean_db=mean_db,
        view_psnr_db=[_round(v) for v in vals_l],
        view_source_indices=[(int(s) if s is not None else None) for s in src_idx],
        n_views_unresolved_source_index=sum(1 for s in src_idx if s is None),
    )


def reduce_partition_psnr_cpu(view_psnr_db: Sequence[float],
                              view_source_indices: Sequence[Optional[int]],
                              max_source_index: int, k: int,
                              denominator_source: str = "unknown") -> PartitionReduction:
    """Plain-Python reference reduction (no torch), for parity testing and
    for re-deriving a partition vector offline from a persisted row.

    Mirrors ``partition_index_tensor``'s convention exactly, including
    bucketing an unresolvable source index into partition 0.
    """
    if len(view_psnr_db) != len(view_source_indices):
        raise ValueError("view_psnr_db and view_source_indices must be parallel")
    sums = [0.0] * k
    counts = [0] * k
    for v, s in zip(view_psnr_db, view_source_indices):
        pid = partition_id(s, max_source_index, k=k) if s is not None else 0
        sums[pid] += float(v)
        counts[pid] += 1
    mean_db = [(_round(sums[i] / counts[i]) if counts[i] > 0 else None) for i in range(k)]
    return PartitionReduction(
        k=int(k),
        max_source_index=int(max_source_index),
        denominator_source=str(denominator_source),
        counts=counts,
        mean_db=mean_db,
        view_psnr_db=[_round(v) for v in view_psnr_db],
        view_source_indices=[(int(s) if s is not None else None) for s in view_source_indices],
        n_views_unresolved_source_index=sum(1 for s in view_source_indices if s is None),
    )


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile over an already-sorted list.

    Hand-rolled (rather than ``numpy.percentile``) only so this module stays
    dependency-light on the reader side; the definition is numpy's default
    ``linear`` method, which the tests pin against numpy.
    """
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def partition_summary(mean_db: Sequence[Optional[float]],
                      counts: Sequence[int]) -> Dict[str, Any]:
    """Across-partition summary of one tick: median / p10 / p90 / worst.

    **Empty partitions are excluded, never counted as 0dB** — a partition with
    no holdout view is unmeasured at this tick, and folding a 0.0 into the
    envelope would manufacture a catastrophic-looking quality floor.
    """
    observed = [(i, v) for i, v in enumerate(mean_db) if v is not None]
    vals = sorted(v for _i, v in observed)
    if not vals:
        return {
            "n_partitions_observed": 0,
            "n_partitions_empty": len(mean_db),
            "partition_median_db": None,
            "partition_p10_db": None,
            "partition_p90_db": None,
            "partition_min_db": None,
            "partition_max_db": None,
            "worst_partition_id": None,
            "worst_partition_psnr_db": None,
            "worst_partition_n_views": None,
        }
    worst_id, worst_val = min(observed, key=lambda t: t[1])
    return {
        "n_partitions_observed": len(vals),
        "n_partitions_empty": len(mean_db) - len(vals),
        "partition_median_db": _round(_percentile(vals, 0.5)),
        "partition_p10_db": _round(_percentile(vals, 0.10)),
        "partition_p90_db": _round(_percentile(vals, 0.90)),
        "partition_min_db": _round(vals[0]),
        "partition_max_db": _round(vals[-1]),
        "worst_partition_id": int(worst_id),
        "worst_partition_psnr_db": _round(worst_val),
        "worst_partition_n_views": int(counts[worst_id]) if worst_id < len(counts) else None,
    }


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------
def resolve_source_progress(source_frame_index: Optional[int],
                            declared_n_source_frames: Optional[int]) -> Tuple[Optional[float], Optional[str]]:
    """``(source_progress, unavailable_reason)`` on the DECLARED denominator.

    ``source_progress`` is ``source_frame_index / declared_n_source_frames``.

    Returns ``(None, reason)`` — never a renormalized fallback — when the
    scene has no registry entry. Renormalizing against the realized/truncated
    prefix is explicitly forbidden by issue #698's correctness rules: it makes
    a run that covered 5% of a trajectory report progress 1.0.

    An out-of-range numerator is **refused, not clamped**
    (``progress_denominator_mismatch``, the failure mode
    ``causal_quality.STATUS_PROGRESS_DENOMINATOR_MISMATCH`` names): a
    numerator above the declared length means the two are not in the same
    index space, and clamping to 1.0 would report "trajectory finished" for
    what is actually a ruler mismatch.

    **Ruler caveat, stated rather than hidden.** The numerator is a position
    in the frame source the run actually loaded, which is the same space
    ``partition_eval.camera_source_index`` reads (dataset sources assign
    ``StreamingRGBDFrame.index`` by enumeration *after* any stride/max-frames
    truncation), so progress and the partition vector within one row are
    always measured against the same ruler. Under
    ``--streaming_frame_stride > 1`` that space is compressed relative to the
    registry's full-trajectory ``n_source_frames``, so progress under-reports
    — the same known denominator gap already documented for
    ``_coverage_n_frames``. Not fixed here: it is a repo-wide denominator
    question, not a #698 one.
    """
    if not declared_n_source_frames or declared_n_source_frames <= 0:
        return None, "no_declared_denominator: scene missing from eval/scenes.json registry"
    if source_frame_index is None:
        return None, "source_frame_index_unavailable"
    if source_frame_index < 0 or source_frame_index > declared_n_source_frames:
        return None, (
            f"progress_denominator_mismatch: source_frame_index "
            f"{int(source_frame_index)} outside [0, {int(declared_n_source_frames)}]"
        )
    return float(source_frame_index) / float(declared_n_source_frames), None


def derive_phase(iteration: int, last_ingest_iter: Optional[int],
                 mapping_idle_threshold_iters: Optional[int]) -> Tuple[Optional[str], str, Optional[str]]:
    """``(phase, phase_source, unavailable_reason)`` for one tick.

    LIVE phase attribution can only ever be provisional: the true mapping-end
    boundary is knowable only once ingestion has actually stopped for good
    (which is what ``utils/mapping_phase_backfill.py`` reconstructs post-hoc).
    So this reports the *observable* fact — how long it has been since the
    last ingest, relative to the run's own idle threshold — and labels it as
    such in ``phase_source``, rather than claiming a boundary it cannot know.
    """
    if last_ingest_iter is None:
        return None, "unavailable", "no_last_ingest_iter"
    if not mapping_idle_threshold_iters or mapping_idle_threshold_iters <= 0:
        return None, "unavailable", "no_idle_threshold"
    gap = int(iteration) - int(last_ingest_iter)
    phase = "mapping" if gap < int(mapping_idle_threshold_iters) else "post_ingest_idle"
    return phase, f"last_ingest_idle_gap(threshold={int(mapping_idle_threshold_iters)})", None


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def build_eval_tick_record(
    *,
    iteration: int,
    mean_test_psnr_db: Optional[float],
    reduction: PartitionReduction,
    scene_source: Optional[str] = None,
    declared_n_source_frames: Optional[int] = None,
    source_frame_index: Optional[int] = None,
    frames_ingested: Optional[int] = None,
    frames_trained: Optional[int] = None,
    keyframes_admitted: Optional[int] = None,
    optimizer_step: Optional[int] = None,
    supervision_views: Optional[int] = None,
    mapping_snapshot_id: Optional[int] = None,
    control_tick_epoch: Optional[int] = None,
    supervision_constellation: Optional[Mapping[str, Any]] = None,
    wall_clock_s: Optional[float] = None,
    accumulated_iteration_wall_s: Optional[float] = None,
    last_ingest_iter: Optional[int] = None,
    mapping_idle_threshold_iters: Optional[int] = None,
    realized_holdout_hash: Optional[str] = None,
    scored_holdout_hash: Optional[str] = None,
    n_test_views_attempted: Optional[int] = None,
    partition_schema: Optional[str] = None,
    evaluation_phase: Optional[str] = None,
    evaluation_observer_execution: Optional[str] = None,
    mapping_execution_paused: Optional[bool] = None,
    throughput_evidence_valid: Optional[bool] = None,
    throughput_evidence_status: Optional[str] = None,
    evaluation_cadence_clock: Optional[str] = None,
    psnr_reduction: str = PSNR_REDUCTION_RGB_PER_VIEW_V2,
) -> Dict[str, Any]:
    """Assemble one canonical tick row (a plain dict, JSON-ready).

    Pure function of already-synced values: no torch, no IO, no clock reads
    of its own — every clock is passed in by the caller so it is joined at
    the *exact* tick, never re-derived later from final run state.
    """
    provenance: Dict[str, str] = {}
    n_test = len(reduction.view_psnr_db)
    progress, progress_reason = resolve_source_progress(
        source_frame_index, declared_n_source_frames
    )
    if progress_reason:
        provenance["source_progress"] = progress_reason
    phase, phase_source, phase_reason = derive_phase(
        iteration, last_ingest_iter, mapping_idle_threshold_iters
    )
    if evaluation_phase is not None:
        phase = str(evaluation_phase)
        phase_source = "causal_evaluation_observer"
        phase_reason = None
    if phase_reason:
        provenance["phase"] = phase_reason
    if last_ingest_iter is None:
        provenance["last_ingest_iter"] = "no_ingest_recorded_yet"
    for name, value in (
        ("source_frame_index", source_frame_index),
        ("frames_ingested", frames_ingested),
        ("frames_trained", frames_trained),
        ("supervision_views", supervision_views),
        ("mapping_snapshot_id", mapping_snapshot_id),
        ("control_tick_epoch", control_tick_epoch),
        ("supervision_constellation", supervision_constellation),
        ("optimizer_step", optimizer_step),
        ("wall_clock_s", wall_clock_s),
        ("accumulated_iteration_wall_s", accumulated_iteration_wall_s),
        ("keyframes_admitted", keyframes_admitted),
        ("mapping_idle_threshold_iters", mapping_idle_threshold_iters),
        ("evaluation_phase", evaluation_phase),
        ("evaluation_observer_execution", evaluation_observer_execution),
        ("mapping_execution_paused", mapping_execution_paused),
        ("throughput_evidence_valid", throughput_evidence_valid),
        ("throughput_evidence_status", throughput_evidence_status),
        ("evaluation_cadence_clock", evaluation_cadence_clock),
    ):
        if value is None:
            provenance[name] = "not_supplied_by_caller"
    # A null hash has TWO distinct causes and they must not be conflated:
    # either the caller passed no camera list at all, or `_camera_set_hash`
    # WITHHELD the hash because some camera carried neither a
    # `_streaming_frame.index` nor an `image_name` (it deliberately refuses to
    # fall back to `uid`). The second is a property of the cameras, not of the
    # caller, and reporting it as "not supplied" would misattribute it.
    if realized_holdout_hash is None:
        provenance["realized_holdout_hash"] = (
            "no_camera_pool_supplied" if n_test_views_attempted is None
            else "hash_withheld_camera_lacks_stable_identifier"
        )
    if scored_holdout_hash is None:
        provenance["scored_holdout_hash"] = (
            "hash_withheld_camera_lacks_stable_identifier" if n_test > 0
            else "no_scored_cameras"
        )

    if partition_schema is None:
        partition_schema = (
            CANONICAL_PARTITION_SCHEMA
            if int(reduction.k) == PARTITIONS
            else DIAGNOSTIC_PARTITION_SCHEMA
        )
    summary = partition_summary(reduction.mean_db, reduction.counts)
    record: Dict[str, Any] = {
        "schema_version": EVAL_TICK_SCHEMA_VERSION,
        # --- tick identity ------------------------------------------------
        "iteration": int(iteration),
        "optimizer_step": (int(optimizer_step) if optimizer_step is not None else None),
        "mapping_snapshot_id": (
            int(mapping_snapshot_id) if mapping_snapshot_id is not None else None
        ),
        "control_tick_epoch": (
            int(control_tick_epoch) if control_tick_epoch is not None else None
        ),
        "supervision_constellation": (
            dict(supervision_constellation)
            if supervision_constellation is not None
            else None
        ),
        # --- causal clocks (#428 axis names) ------------------------------
        "source_progress": _round(progress, 6),
        "source_frame_index": (int(source_frame_index) if source_frame_index is not None else None),
        "declared_n_source_frames": (int(declared_n_source_frames)
                                     if declared_n_source_frames else None),
        "wall_clock_s": _round(wall_clock_s, 3),
        "accumulated_iteration_wall_s": _round(accumulated_iteration_wall_s, 3),
        "supervision_views": (int(supervision_views) if supervision_views is not None else None),
        "frames_ingested": (int(frames_ingested) if frames_ingested is not None else None),
        "frames_trained": (int(frames_trained) if frames_trained is not None else None),
        "keyframes_admitted": (int(keyframes_admitted) if keyframes_admitted is not None else None),
        "last_ingest_iter": (int(last_ingest_iter) if last_ingest_iter is not None else None),
        "phase": phase,
        "phase_source": phase_source,
        "evaluation_phase": (
            str(evaluation_phase) if evaluation_phase is not None else None
        ),
        "evaluation_observer_execution": (
            str(evaluation_observer_execution)
            if evaluation_observer_execution is not None
            else None
        ),
        "mapping_execution_paused": (
            bool(mapping_execution_paused)
            if mapping_execution_paused is not None
            else None
        ),
        "throughput_evidence_valid": (
            bool(throughput_evidence_valid)
            if throughput_evidence_valid is not None
            else None
        ),
        "throughput_evidence_status": (
            str(throughput_evidence_status)
            if throughput_evidence_status is not None
            else None
        ),
        "evaluation_cadence_clock": (
            str(evaluation_cadence_clock)
            if evaluation_cadence_clock is not None
            else None
        ),
        # --- global quality -----------------------------------------------
        "mean_test_psnr_db": _round(mean_test_psnr_db),
        "psnr_reduction": str(psnr_reduction),
        "n_test_views": int(n_test),
        "n_test_views_attempted": (int(n_test_views_attempted)
                                   if n_test_views_attempted is not None else None),
        # --- partition quality, SAME render ---------------------------------
        "partition_k": int(reduction.k),
        "partition_schema": str(partition_schema),
        "partition_denominator": int(reduction.max_source_index),
        "partition_denominator_source": reduction.denominator_source,
        "partition_n": [int(c) for c in reduction.counts],
        "partition_psnr_db": list(reduction.mean_db),
        "partition_summary": summary,
        # --- per-view identity (reproducible reduction) ---------------------
        "view_source_indices": list(reduction.view_source_indices),
        "view_psnr_db": list(reduction.view_psnr_db),
        "n_views_unresolved_source_index": int(reduction.n_views_unresolved_source_index),
        # --- holdout / metric provenance ------------------------------------
        "scene_source": (str(scene_source) if scene_source else None),
        # `realized_holdout_hash` = the holdout POOL this tick attempted;
        # `scored_holdout_hash` = the subset that actually rendered and was
        # averaged into `mean_test_psnr_db` (same distinction, and the same
        # hash function, as report.json's eval_metadata ladder -- issue #420).
        # The selection-RULE hash (`holdout_hash`) is not derivable here and
        # is deliberately absent rather than approximated.
        "realized_holdout_hash": realized_holdout_hash,
        "scored_holdout_hash": scored_holdout_hash,
        "metric_provenance": dict(METRIC_PROVENANCE),
        "field_provenance": provenance,
    }
    return record


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _encoded_eval_tick_line(record: Dict[str, Any]) -> bytes:
    """Return the exact UTF-8 line to append, fixing its byte count first."""
    cost = record.get("record_hot_path")
    if isinstance(cost, dict):
        # The byte count includes its own decimal representation.  Iterating
        # reaches a fixed point in at most one digit-width transition.
        cost["serialized_bytes"] = int(cost.get("serialized_bytes", 0) or 0)
        for _ in range(10):
            encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            size = len(encoded)
            if cost["serialized_bytes"] == size:
                return encoded
            cost["serialized_bytes"] = size
        raise RuntimeError("evaluation-tick serialized byte count did not converge")
    return (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")


def append_eval_tick(model_path: str, record: Dict[str, Any]) -> str:
    path = os.path.join(model_path, EVAL_TICK_FILENAME)
    encoded = _encoded_eval_tick_line(record)
    with open(path, "ab") as f:
        f.write(encoded)
    return path


def read_eval_ticks(model_path: str) -> List[Dict[str, Any]]:
    """All rows of ``evaluation_ticks.jsonl``, iteration-ordered.

    Malformed lines are skipped (a run can be killed mid-write); this is a
    reader over an append-only telemetry log, not a validator.
    """
    path = os.path.join(model_path, EVAL_TICK_FILENAME)
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and "iteration" in rec:
                rows.append(rec)
    rows.sort(key=lambda r: int(r.get("iteration", 0)))
    psnr_reduction(
        rows, source=path, value_key="mean_test_psnr_db"
    )
    return rows


def psnr_history_from_eval_ticks(model_path: str) -> Tuple[List[int], List[float]]:
    """Legacy ``psnr_history.jsonl`` compatibility view (acceptance crit. 7).

    Reproduces exactly what ``utils.psnr_history_io._read_psnr_history``
    returns — ``(iterations, mean_test_psnr)`` — from the canonical record,
    so the global ``psnr_over_time`` curve remains reproducible even if the
    legacy file is absent. Rows whose PSNR is null are skipped.
    """
    iters: List[int] = []
    vals: List[float] = []
    for rec in read_eval_ticks(model_path):
        v = rec.get("mean_test_psnr_db")
        if v is None:
            continue
        iters.append(int(rec["iteration"]))
        vals.append(float(v))
    return iters, vals


# ---------------------------------------------------------------------------
# Longitudinal plot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PartitionPlotWorkAxis:
    values: List[int]
    label: str
    last_ingest_value: Optional[int]
    last_ingest_label: Optional[str]


def _partition_plot_work_axis(rows: Sequence[Mapping[str, Any]]) -> _PartitionPlotWorkAxis:
    """Resolve one honest work clock for a longitudinal partition plot.

    Batched optimizer updates are not comparable across B1/B2/B4.  Prefer the
    cumulative supervised-view clock recorded at each evaluation tick.  Old
    histories without that clock fall back to optimizer updates, and only
    legacy rows without either typed clock fall back to ``iteration``.
    """

    def _complete_monotonic_clock(name: str) -> Optional[List[int]]:
        values: List[int] = []
        for row in rows:
            value = row.get(name)
            if value is None:
                return None
            value = int(value)
            if values and value < values[-1]:
                return None
            values.append(value)
        return values

    supervised = _complete_monotonic_clock("supervision_views")
    if supervised is not None:
        values = supervised
        label = "cumulative supervised views"
        ingest_suffix = "views"
        ingest_prefix = ""
    else:
        optimizer_updates = _complete_monotonic_clock("optimizer_step")
        if optimizer_updates is not None:
            values = optimizer_updates
        else:
            values = [int(row["iteration"]) for row in rows]
        label = "optimizer updates"
        ingest_suffix = ""
        ingest_prefix = "update "

    last_ingest_iters = [
        int(row["last_ingest_iter"])
        for row in rows
        if row.get("last_ingest_iter") is not None
    ]
    last_ingest_value: Optional[int] = None
    last_ingest_label: Optional[str] = None
    if last_ingest_iters:
        target_iteration = last_ingest_iters[-1]
        for row, value in zip(rows, values):
            if int(row["iteration"]) == target_iteration:
                last_ingest_value = value
                formatted = f"{value:,}"
                last_ingest_label = (
                    f"last ingest ({ingest_prefix}{formatted}"
                    f"{(' ' + ingest_suffix) if ingest_suffix else ''})"
                )
                break

    return _PartitionPlotWorkAxis(
        values=values,
        label=label,
        last_ingest_value=last_ingest_value,
        last_ingest_label=last_ingest_label,
    )


def _partition_plot_schema(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, str]:
    """Describe the aggregation represented by a plot's complete history.

    A plot is a presentation surface, but silently joining six-way report
    bins with trajectory-scaled diagnostic bins is still a misleading claim.
    Untagged legacy rows remain readable and are labelled conservatively; a
    mixed history is explicitly marked non-comparable rather than inferred to
    be one homogeneous series.
    """
    keys = set()
    for row in rows:
        schema = row.get("partition_schema")
        try:
            k = int(row.get("partition_k"))
        except (TypeError, ValueError):
            k = None
        if schema == CANONICAL_PARTITION_SCHEMA and k == PARTITIONS:
            keys.add(("canonical", PARTITIONS))
        elif schema == DIAGNOSTIC_PARTITION_SCHEMA and k not in (None, PARTITIONS):
            keys.add(("diagnostic", k))
        elif schema in (CANONICAL_PARTITION_SCHEMA, DIAGNOSTIC_PARTITION_SCHEMA):
            keys.add(("invalid", str(schema), k))
        elif schema:
            keys.add(("unknown", str(schema), k))
        else:
            keys.add(("legacy_unlabelled", k))
    if len(keys) != 1:
        return (
            "MIXED PARTITION SCHEMAS — NOT COMPARABLE",
            "worst recorded bin (mixed; not comparable)",
        )
    key = next(iter(keys))
    if key[0] == "canonical":
        return (
            f"canonical {PARTITIONS}-way report bins; worst=bin mean",
            "worst report partition mean",
        )
    if key[0] == "diagnostic":
        k = key[1]
        return (
            f"DIAGNOSTIC {k}-way bins; single-view bins possible",
            "worst diagnostic bin mean",
        )
    if key[0] == "invalid":
        return (
            "INVALID PARTITION SCHEMA — NOT COMPARABLE",
            "worst recorded bin (invalid schema)",
        )
    return (
        f"legacy unlabelled {key[-1]}-way bins; schema unavailable",
        "worst unlabelled bin mean",
    )


def _partition_plot_xlim(work: Sequence[int],
                          include: Optional[int] = None) -> Optional[Tuple[float, float]]:
    """Tight x-axis bounds spanning the ticks that actually exist.

    Matplotlib's default ~5% autoscale margin pads the axis beyond the real
    data range on both sides. For a canonical-mapping run whose evaluation
    observer only executes ``post_source_completion`` (the default -- see
    ``mapping_holdout_curve_metadata``; mid-mapping evaluation is opt-in via
    ``--streaming_mapping_holdout_curve`` because it synchronously pauses
    the mapping thread), EVERY recorded tick sits at or after
    ``last_ingest_iter``. Left to the default margin, the plot then shows a
    visually empty region before the first real point that looks exactly
    like dropped mapping-phase data but is not: no evaluation ever runs
    there by design, so there is nothing to plot, and the axis should not
    imply otherwise. Confirmed against a real run's own
    ``evaluation_ticks.jsonl`` (every row's ``evaluation_observer_execution``
    read ``"post_source_completion"``, none earlier).

    ``include`` widens the bound (never narrows it) to guarantee a given
    x-value stays on-axis AND has a small margin around it. Used for the
    ``last ingest`` marker: its work-clock value is the supervision count AT
    mapping end, which either sits strictly before the first post-mapping
    evaluation tick (clipped outside the tight bound above -- drawn, but
    invisible, with only its legend entry showing) or coincides exactly with
    it (the common case, since the mapping-endpoint tick is itself usually
    the first post-mapping evaluation tick) -- and in BOTH cases it lands
    exactly on the axis spine, where the dotted marker line renders flush
    against, and visually indistinguishable from, the solid border. Found
    2026-08-30 on a real run's ``partition_psnr_over_time.svg``: a
    higher-contrast marker style alone (see the caller) fixed the clipped
    case but not the coincident one -- a real margin is unavoidable on that
    side. A small, deliberate margin here (a few percent of the span) is not
    the failure this function's tight-bound docstring above guards against:
    that was matplotlib's ~5% autoscale margin creating a large, UNLABELED
    empty region that looked like silently dropped mapping-phase data. A
    small margin drawn specifically to make a real, labeled marker visible
    is a different thing entirely.
    """
    if not work and include is None:
        return None
    work_lo = min(work) if work else include
    work_hi = max(work) if work else include
    lo, hi = work_lo, work_hi
    if include is not None:
        lo = min(lo, include)
        hi = max(hi, include)
        span = hi - lo
        pad = max(span * 0.03, 1.0) if span > 0 else 1.0
        if include <= work_lo:
            lo = include - pad
        if include >= work_hi:
            hi = include + pad
    return (float(lo), float(hi))


def plot_partition_psnr_over_time(model_path: str,
                                  out_name: str = "partition_psnr_over_time.svg") -> Optional[str]:
    """Deterministic overview plot, regenerated purely from the JSONL.

    Global mean + across-partition median + p10-p90 envelope + worst
    partition, with a marker at the last recorded ``last_ingest_iter``. The
    complete per-partition series stays machine-readable in the JSONL; this
    plot is the overview, not the data.

    Deterministic: regenerating from identical input yields byte-identical
    PNGs, verified across separate interpreter processes by the tests. PNG
    metadata is additionally pinned as belt-and-braces against matplotlib
    stamping per-invocation info (measured: byte equality holds either way on
    the current matplotlib, so the pin is defensive, not load-bearing).
    """
    rows = read_eval_ticks(model_path)
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "eligsir"
        import matplotlib.pyplot as plt

        work_axis = _partition_plot_work_axis(rows)
        work = work_axis.values
        gmean = [r.get("mean_test_psnr_db") for r in rows]
        med = [(r.get("partition_summary") or {}).get("partition_median_db") for r in rows]
        p10 = [(r.get("partition_summary") or {}).get("partition_p10_db") for r in rows]
        p90 = [(r.get("partition_summary") or {}).get("partition_p90_db") for r in rows]
        worst = [(r.get("partition_summary") or {}).get("worst_partition_psnr_db") for r in rows]
        partition_label, worst_label = _partition_plot_schema(rows)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        env = [(i, a, b) for i, a, b in zip(work, p10, p90) if a is not None and b is not None]
        if env:
            ax.fill_between([e[0] for e in env], [e[1] for e in env], [e[2] for e in env],
                            alpha=0.2, color="C0", label="partition p10-p90")

        def _line(xs, ys, **kw):
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)

        _line(work, med, color="C0", label="partition median")
        _line(work, gmean, color="C1", label="global mean (all holdout views)")
        _line(work, worst, color="C3", linestyle="--", label=worst_label)

        if work_axis.last_ingest_value is not None:
            # The mapping-endpoint evaluation tick often lands EXACTLY at the
            # tight left axis bound (the first post-mapping tick's own
            # work-clock value) -- a plain thin gray dotted line there
            # renders flush against, and visually indistinguishable from,
            # the solid axis spine (found 2026-08-30: legend entry present,
            # line invisible). Higher zorder/linewidth/contrast color keeps
            # it visible in exactly that boundary case, independent of the
            # xlim padding above (which only fires when the marker falls
            # strictly outside the real data range).
            ax.axvline(
                work_axis.last_ingest_value,
                color="tab:purple",
                linestyle=":",
                linewidth=1.8,
                zorder=5,
                label=work_axis.last_ingest_label,
            )
        ax.set_xlabel(work_axis.label)
        ax.set_ylabel("test PSNR (dB)")
        ax.set_title(
            "Temporal-partition holdout PSNR\n"
            f"{partition_label}; {len(rows)} evaluation ticks"
        )
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)
        xlim = _partition_plot_xlim(work, include=work_axis.last_ingest_value)
        if xlim is not None and xlim[0] != xlim[1]:
            ax.set_xlim(*xlim)
        out_path = str(svg_output_path(os.path.join(model_path, out_name)))
        fig.tight_layout()
        fig.savefig(out_path, dpi=100, metadata={"Date": None})
        plt.close(fig)
        return out_path
    except Exception as e:  # plotting must never take down a run
        print(f"[eval_tick] plot failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# The single wiring entry point
# ---------------------------------------------------------------------------
def record_eval_tick(
    *,
    model_path: str,
    iteration: int,
    psnr_tick,
    test_cams: Optional[Sequence] = None,
    scene_source: Optional[str] = None,
    source_frame_index: Optional[int] = None,
    frames_ingested: Optional[int] = None,
    frames_trained: Optional[int] = None,
    keyframes_admitted: Optional[int] = None,
    optimizer_step: Optional[int] = None,
    supervision_views: Optional[int] = None,
    mapping_snapshot_id: Optional[int] = None,
    control_tick_epoch: Optional[int] = None,
    supervision_constellation: Optional[Mapping[str, Any]] = None,
    wall_clock_s: Optional[float] = None,
    accumulated_iteration_wall_s: Optional[float] = None,
    last_ingest_iter: Optional[int] = None,
    mapping_idle_threshold_iters: Optional[int] = None,
    partition_k: Optional[int] = None,
    evaluation_phase: Optional[str] = None,
    evaluation_observer_execution: Optional[str] = None,
    mapping_execution_paused: Optional[bool] = None,
    throughput_evidence_valid: Optional[bool] = None,
    throughput_evidence_status: Optional[str] = None,
    evaluation_cadence_clock: Optional[str] = None,
    plot: bool = False,
) -> Optional[Dict[str, Any]]:
    """Reduce -> build -> append, from an ALREADY-RENDERED tick.

    ``psnr_tick`` is a ``utils.psnr_history.PsnrTickResult``. This function
    renders nothing and mutates nothing; returns the written record, or
    ``None`` when the tick carried no usable per-view tensor.

    ``partition_k`` defaults to the canonical report ``PARTITIONS`` (six-way)
    aggregation. A noncanonical ``partition_k`` is retained as an explicitly
    diagnostic aggregation and stamped as such in the row schema. ``k`` is
    written into every row, so a reader never has to guess the vector length.
    """
    if psnr_tick is None or getattr(psnr_tick, "psnr_tensor", None) is None:
        return None
    cams = list(getattr(psnr_tick, "cams", []) or [])
    if not cams:
        return None
    declared = registry_n_source_frames(scene_source)
    max_src, denom_source = resolve_max_source_index(scene_source, cams)
    k = int(partition_k) if partition_k is not None else PARTITIONS
    partition_schema = (
        DIAGNOSTIC_PARTITION_SCHEMA
        if partition_k is not None and k != PARTITIONS
        else CANONICAL_PARTITION_SCHEMA
    )
    started = time.perf_counter()
    reduction = reduce_partition_psnr(
        psnr_tick.psnr_tensor, cams, max_src, k, denominator_source=denom_source
    )
    realized_hash = scored_hash = None
    try:
        from eligsir.utils.baseline_comparison import _camera_set_hash
        # cams == the views that actually rendered and were averaged.
        scored_hash = _camera_set_hash(cams)
        realized_hash = _camera_set_hash(test_cams) if test_cams else None
    except Exception:
        pass
    record = build_eval_tick_record(
        iteration=iteration,
        mean_test_psnr_db=getattr(psnr_tick, "mean_psnr", None),
        reduction=reduction,
        scene_source=scene_source,
        declared_n_source_frames=declared,
        source_frame_index=source_frame_index,
        frames_ingested=frames_ingested,
        frames_trained=frames_trained,
        keyframes_admitted=keyframes_admitted,
        optimizer_step=optimizer_step,
        supervision_views=supervision_views,
        mapping_snapshot_id=mapping_snapshot_id,
        control_tick_epoch=control_tick_epoch,
        supervision_constellation=supervision_constellation,
        wall_clock_s=wall_clock_s,
        accumulated_iteration_wall_s=accumulated_iteration_wall_s,
        last_ingest_iter=last_ingest_iter,
        mapping_idle_threshold_iters=mapping_idle_threshold_iters,
        realized_holdout_hash=realized_hash,
        scored_holdout_hash=scored_hash,
        n_test_views_attempted=(len(test_cams) if test_cams is not None else None),
        partition_schema=partition_schema,
        evaluation_phase=evaluation_phase,
        evaluation_observer_execution=evaluation_observer_execution,
        mapping_execution_paused=mapping_execution_paused,
        throughput_evidence_valid=throughput_evidence_valid,
        throughput_evidence_status=throughput_evidence_status,
        evaluation_cadence_clock=evaluation_cadence_clock,
        psnr_reduction=getattr(
            psnr_tick, "psnr_reduction", PSNR_REDUCTION_RGB_PER_VIEW_V2
        ),
    )
    edge_value = getattr(psnr_tick, "mean_edge_psnr", None)
    temporal_status = getattr(psnr_tick, "temporal_flicker_status", "not_computed")
    def _finite_optional(value):
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    if edge_value is not None or temporal_status != "not_computed":
        record["antialiasing_metrics"] = {
            "rasterize_mode": str(
                getattr(psnr_tick, "rasterize_mode", "classic")
            ),
            "gaussian_count": getattr(psnr_tick, "gaussian_count", None),
            "training_resolution_tiers": list(
                getattr(psnr_tick, "training_resolution_tiers", ()) or ()
            ),
            "evaluation_resolutions": [
                [int(width), int(height)]
                for width, height in (
                    getattr(psnr_tick, "evaluation_resolutions", ()) or ()
                )
            ],
            "mean_edge_psnr_db": (
                _finite_optional(edge_value)
            ),
            "edge_score_scope": "sobel_gradient_same_evaluation_resolution",
            "temporal_flicker_rms": _finite_optional(
                getattr(psnr_tick, "temporal_flicker_rms", None)
            ),
            "temporal_flicker_psnr_db": _finite_optional(
                getattr(psnr_tick, "temporal_flicker_psnr_db", None)
            ),
            "temporal_flicker_valid_pairs": int(
                getattr(psnr_tick, "temporal_flicker_valid_pairs", 0)
            ),
            "temporal_flicker_status": str(temporal_status),
        }
    record["record_hot_path"] = {
        "elapsed_scope": RECORD_HOT_PATH_ELAPSED_SCOPE,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
        "serialized_bytes": 0,
    }
    append_eval_tick(model_path, record)
    if plot:
        plot_partition_psnr_over_time(model_path)
    return record


def _record_sha256(record: Dict[str, Any]) -> str:
    """Content address for an exact, canonical evaluation-tick row."""
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_terminal_int(
    value: Any, *, field: str, scope: str, minimum: int = 0
) -> int:
    if type(value) is not int:
        raise ValueError(f"exact terminal {scope} field {field} must be an integer")
    if value < minimum:
        raise ValueError(
            f"exact terminal {scope} field {field} must be >= {minimum}"
        )
    return value


def _validate_exact_terminal_snapshot(
    clocks: Mapping[str, Any], constellation: Mapping[str, Any]
) -> None:
    clock_integer_fields = {
        "source_frame_index",
        "frames_ingested",
        "frames_trained",
        "keyframes_admitted",
        "optimizer_step",
        "supervision_views",
        "mapping_snapshot_id",
        "control_tick_epoch",
        "last_ingest_iter",
        "mapping_idle_threshold_iters",
    }
    for name in sorted(clock_integer_fields.intersection(clocks)):
        if clocks[name] is not None:
            _exact_terminal_int(clocks[name], field=name, scope="clock")

    def exact_int(name: str, minimum: int = 0) -> int:
        return _exact_terminal_int(
            constellation[name],
            field=name,
            scope="supervision constellation",
            minimum=minimum,
        )

    requested_b = exact_int("requested_batch_size", 1)
    requested_c = exact_int("requested_camera_count", 1)
    realized_b = exact_int("realized_batch_size")
    realized_c = exact_int("realized_camera_count")
    eligible = exact_int("unique_eligible_count")
    microbatch_count = exact_int("microbatch_count")
    optimizer_updates = exact_int("optimizer_updates")
    supervised_views = exact_int("supervised_views")
    supervised_pixels = exact_int("supervised_pixels")
    cumulative_views = exact_int("cumulative_realized_supervised_views")
    cumulative_updates = exact_int("cumulative_optimizer_updates")
    unique_value = constellation["unique_supervision_views"]
    unique_views = (
        None
        if unique_value is None
        else _exact_terminal_int(
            unique_value,
            field="unique_supervision_views",
            scope="supervision constellation",
        )
    )
    sizes_value = constellation["microbatch_sizes"]
    if not isinstance(sizes_value, (list, tuple)):
        raise ValueError(
            "exact terminal supervision constellation field microbatch_sizes "
            "must be a sequence"
        )
    microbatch_sizes = tuple(
        _exact_terminal_int(
            size,
            field="microbatch_sizes",
            scope="supervision constellation",
            minimum=1,
        )
        for size in sizes_value
    )

    def reject(reason: str) -> None:
        raise ValueError("exact terminal supervision constellation " + reason)

    from eligsir.contracts.control import SupervisionConstellation

    planned_microbatch_count = (
        0
        if realized_b == 0
        else math.ceil(realized_b / max(1, realized_c))
    )
    try:
        SupervisionConstellation(
            requested_batch_size=requested_b,
            requested_camera_count=requested_c,
            realized_batch_size=realized_b,
            realized_camera_count=realized_c,
            unique_eligible_count=eligible,
            fill_ratio=constellation["fill_ratio"],
            microbatch_count=planned_microbatch_count,
            mode=constellation["mode"],
            reason=constellation["reason"],
            hysteresis_state=constellation["hysteresis_state"],
            cumulative_realized_supervised_views=cumulative_views,
            cumulative_optimizer_updates=cumulative_updates,
        )
    except (TypeError, ValueError) as exc:
        reject(str(exc))
    if microbatch_count != len(microbatch_sizes):
        reject("microbatch count does not match actual microbatch sizes")
    if any(size > realized_c for size in microbatch_sizes):
        reject("microbatch size exceeds realized C")
    if sum(microbatch_sizes) != supervised_views:
        reject("microbatch sizes do not sum to actual supervised views")
    if realized_b == 0:
        if optimizer_updates != 0 or supervised_views != 0:
            reject("empty realization reports optimization work")
    elif requested_b == 1:
        if (
            optimizer_updates != microbatch_count
            or supervised_views != optimizer_updates
        ):
            reject("legacy B1 work does not match its optimizer updates")
    else:
        expected_sizes = tuple(
            min(realized_c, realized_b - offset)
            for offset in range(0, realized_b, realized_c)
        )
        if optimizer_updates != 1 or supervised_views != realized_b:
            reject("adaptive work does not describe one optimizer transaction")
        if microbatch_sizes != expected_sizes:
            reject("microbatch sizes do not partition realized B by realized C")
    expected_unique_views = realized_b if supervised_views > 0 else 0
    if unique_views is not None and unique_views != expected_unique_views:
        reject("known unique supervised views do not equal realized B")
    if supervised_views == 0 and supervised_pixels != 0:
        reject("reports pixels without supervised work")
    if supervised_views > 0 and supervised_pixels == 0:
        reject("reports supervised work without participating pixels")
    if cumulative_updates != clocks["optimizer_step"]:
        reject("optimizer clock mismatch")
    if cumulative_views != clocks["supervision_views"]:
        reject("view clock mismatch")


def record_final_report_eval_tick(
    *,
    model_path: str,
    iteration: int,
    report_iteration: int,
    view_psnr_db: Sequence[float],
    scored_cams: Sequence,
    test_cams: Optional[Sequence] = None,
    scene_source: Optional[str] = None,
    reported_mean_test_psnr_db: Optional[float] = None,
    clock_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the final report's already-measured test metrics as one tick.

    The final report intentionally does not go through the loop-time
    :class:`PsnrTickResult` path.  Its own metrics pass nevertheless produces
    the exact per-view PSNR values and camera list.  Reusing those values keeps
    the final report on #698's canonical timeline without another render.

    The reported aggregate is checked before anything is appended: a report
    that disagrees with the per-view evidence must not gain a deceptively
    authoritative tick row.
    """
    values = [float(v) for v in view_psnr_db]
    cams = list(scored_cams)
    if len(values) != len(cams):
        raise ValueError(
            "final report per-view PSNR and scored camera lists must be index-aligned"
        )
    if not values:
        raise ValueError("final report has no scored test views")
    if reported_mean_test_psnr_db is None:
        raise ValueError("final report has no mean_test_psnr to bind")
    computed_mean = sum(values) / len(values)
    if not math.isfinite(computed_mean) or not math.isfinite(float(reported_mean_test_psnr_db)):
        raise ValueError("final report mean_test_psnr is not finite")
    if not math.isclose(
        computed_mean, float(reported_mean_test_psnr_db), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            "final report mean_test_psnr mismatch: "
            f"report={float(reported_mean_test_psnr_db)!r} per_view={computed_mean!r}"
        )

    clocks = dict(clock_snapshot or {})
    allowed_clock_fields = {
        "source_frame_index",
        "frames_ingested",
        "frames_trained",
        "keyframes_admitted",
        "optimizer_step",
        "supervision_views",
        "mapping_snapshot_id",
        "control_tick_epoch",
        "supervision_constellation",
        "wall_clock_s",
        "accumulated_iteration_wall_s",
        "last_ingest_iter",
        "mapping_idle_threshold_iters",
        "evaluation_phase",
        "evaluation_observer_execution",
        "mapping_execution_paused",
        "throughput_evidence_valid",
        "throughput_evidence_status",
        "evaluation_cadence_clock",
    }
    unknown = sorted(set(clocks) - allowed_clock_fields)
    if unknown:
        raise ValueError(f"unknown final evaluation-tick clock fields: {unknown}")
    if clock_snapshot is not None:
        required_clock_fields = {
            "optimizer_step",
            "supervision_views",
            "mapping_snapshot_id",
            "control_tick_epoch",
            "supervision_constellation",
        }
        missing = sorted(
            name
            for name in required_clock_fields
            if name not in clocks or clocks[name] is None
        )
        if missing:
            raise ValueError(
                "exact terminal clock snapshot missing fields: " + ", ".join(missing)
            )
        constellation = clocks["supervision_constellation"]
        if not isinstance(constellation, Mapping):
            raise ValueError("exact terminal supervision constellation must be a mapping")
        required_constellation_fields = {
            "requested_batch_size",
            "realized_batch_size",
            "requested_camera_count",
            "realized_camera_count",
            "unique_eligible_count",
            "fill_ratio",
            "microbatch_count",
            "microbatch_sizes",
            "optimizer_updates",
            "supervised_views",
            "unique_supervision_views",
            "supervised_pixels",
            "cumulative_realized_supervised_views",
            "cumulative_optimizer_updates",
            "mode",
            "reason",
            "hysteresis_state",
        }
        missing_constellation = sorted(
            required_constellation_fields - set(constellation)
        )
        if missing_constellation:
            raise ValueError(
                "exact terminal supervision constellation missing fields: "
                + ", ".join(missing_constellation)
            )
        _validate_exact_terminal_snapshot(clocks, constellation)

    declared = registry_n_source_frames(scene_source)
    max_src, denominator_source = resolve_max_source_index(scene_source, cams)
    started = time.perf_counter()
    reduction = reduce_partition_psnr_cpu(
        values,
        [camera_source_index(cam) for cam in cams],
        max_src,
        PARTITIONS,
        denominator_source=denominator_source,
    )
    realized_hash = scored_hash = None
    try:
        from eligsir.utils.baseline_comparison import _camera_set_hash

        scored_hash = _camera_set_hash(cams)
        realized_hash = _camera_set_hash(test_cams) if test_cams is not None else None
    except Exception:
        pass
    record = build_eval_tick_record(
        iteration=int(iteration),
        mean_test_psnr_db=float(reported_mean_test_psnr_db),
        reduction=reduction,
        scene_source=scene_source,
        declared_n_source_frames=declared,
        source_frame_index=clocks.get("source_frame_index"),
        frames_ingested=clocks.get("frames_ingested"),
        frames_trained=clocks.get("frames_trained"),
        keyframes_admitted=clocks.get("keyframes_admitted"),
        optimizer_step=clocks.get("optimizer_step"),
        supervision_views=clocks.get("supervision_views"),
        mapping_snapshot_id=clocks.get("mapping_snapshot_id"),
        control_tick_epoch=clocks.get("control_tick_epoch"),
        supervision_constellation=clocks.get("supervision_constellation"),
        wall_clock_s=clocks.get("wall_clock_s"),
        accumulated_iteration_wall_s=clocks.get("accumulated_iteration_wall_s"),
        last_ingest_iter=clocks.get("last_ingest_iter"),
        mapping_idle_threshold_iters=clocks.get("mapping_idle_threshold_iters"),
        realized_holdout_hash=realized_hash,
        scored_holdout_hash=scored_hash,
        n_test_views_attempted=(len(test_cams) if test_cams is not None else None),
        evaluation_phase=clocks.get("evaluation_phase"),
        evaluation_observer_execution=clocks.get(
            "evaluation_observer_execution"
        ),
        mapping_execution_paused=clocks.get("mapping_execution_paused"),
        throughput_evidence_valid=clocks.get("throughput_evidence_valid"),
        throughput_evidence_status=clocks.get("throughput_evidence_status"),
        evaluation_cadence_clock=clocks.get("evaluation_cadence_clock"),
    )
    record["origin"] = "final_report"
    record["final_report_iteration"] = int(report_iteration)
    record["clock_snapshot_status"] = (
        "caller_supplied_exact_terminal_snapshot"
        if clock_snapshot is not None
        else "not_supplied_nonstreaming_or_legacy_caller"
    )
    record["metric_provenance"] = {
        "metric": "psnr_db",
        "source": "utils.comparison_report.write_post_training_report._compute_metrics",
        "reduction": "per-view report PSNR mean, partitioned without another render",
        "render_pass": "final_report_metrics",
        "extra_renders": 0,
    }
    record["record_hot_path"] = {
        "elapsed_scope": RECORD_HOT_PATH_ELAPSED_SCOPE,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 6),
        "serialized_bytes": 0,
    }
    _encoded_eval_tick_line(record)
    digest = _record_sha256(record)
    append_eval_tick(model_path, record)
    return {
        "status": "bound",
        "evaluation_tick_iteration": int(iteration),
        "report_iteration": int(report_iteration),
        "sha256": digest,
    }


def finalize_eval_tick_history(model_path: str) -> Dict[str, Any]:
    """Generate the final plot and summarize the explicitly scoped hot path."""
    rows = read_eval_ticks(model_path)
    costs = [
        row["record_hot_path"]
        for row in rows
        if isinstance(row.get("record_hot_path"), dict)
        and row["record_hot_path"].get("elapsed_scope")
        == RECORD_HOT_PATH_ELAPSED_SCOPE
    ]
    elapsed = sorted(float(cost["elapsed_ms"]) for cost in costs)
    byte_counts = [int(cost["serialized_bytes"]) for cost in costs]
    plot_path = plot_partition_psnr_over_time(model_path)
    return {
        "status": (
            "available" if costs and len(costs) == len(rows)
            else "partial_legacy_rows" if costs
            else "no_timed_ticks"
        ),
        "count": len(rows),
        "timed_count": len(costs),
        "elapsed_scope": RECORD_HOT_PATH_ELAPSED_SCOPE,
        "elapsed_ms_p50": _round(_percentile(elapsed, 0.50), 6),
        "elapsed_ms_p95": _round(_percentile(elapsed, 0.95), 6),
        "serialized_bytes_total": sum(byte_counts),
        "serialized_bytes_count": len(byte_counts),
        "plot_status": "written" if plot_path else "unavailable",
        "plot_path": os.path.basename(plot_path) if plot_path else None,
    }
