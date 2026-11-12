"""Fail-closed partition-control roles plus bounded historical repair."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from eligsir.utils.partition_eval import camera_source_index, partition_id
from eligsir.utils.partition_health import detect_outlier_partitions


class ControlSignalSource(str, Enum):
    TRAINING_DERIVED = "training_derived"
    CONTROL_PROBE = "control_probe"
    SCORED_HOLDOUT_ORACLE = "scored_holdout_oracle"


class EvaluationLeakageError(ValueError):
    """Raised before actuation when normal control overlaps scored holdout."""


def _indices(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _index_hash(values: Iterable[int]) -> Optional[str]:
    ordered = _indices(values)
    if not ordered:
        return None
    payload = ",".join(f"idx:{value}" for value in ordered)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _deterministic_probe_indices(training_indices: tuple[int, ...], requested_n: int) -> tuple[int, ...]:
    if requested_n <= 0 or requested_n >= len(training_indices):
        return training_indices
    if requested_n == 1:
        return training_indices[-1:]
    last = len(training_indices) - 1
    positions = [(i * last) // (requested_n - 1) for i in range(requested_n)]
    return tuple(training_indices[position] for position in positions)


@dataclass(frozen=True)
class PartitionControlRoles:
    source: ControlSignalSource
    training_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    control_probe_indices: tuple[int, ...]
    scored_holdout_indices: tuple[int, ...]
    controller_enabled: bool

    @property
    def control_probe_hash(self) -> Optional[str]:
        return _index_hash(self.control_probe_indices)

    @property
    def control_probe_n(self) -> int:
        return len(self.control_probe_indices)

    @property
    def scored_holdout_hash(self) -> Optional[str]:
        return _index_hash(self.scored_holdout_indices)

    @property
    def scored_holdout_n(self) -> int:
        return len(self.scored_holdout_indices)

    @property
    def control_signal_hash(self) -> Optional[str]:
        return _index_hash(self.control_indices)

    @property
    def control_signal_n(self) -> int:
        return len(self.control_indices)

    @property
    def control_scored_overlap_n(self) -> int:
        return len(set(self.control_indices).intersection(self.scored_holdout_indices))

    @property
    def oracle_control_enabled(self) -> bool:
        return bool(self.controller_enabled and self.source is ControlSignalSource.SCORED_HOLDOUT_ORACLE)

    @property
    def scientific_eval_status(self) -> str:
        if not self.controller_enabled:
            return "CONTROL_DISABLED"
        if self.oracle_control_enabled:
            return "ORACLE_INTERVENTION"
        return "VALID_NON_ORACLE_CONTROL"

    def to_report_dict(self) -> dict:
        return {
            "control_signal_source": self.source.value,
            "control_signal_hash": self.control_signal_hash,
            "control_signal_n": self.control_signal_n,
            "control_probe_hash": self.control_probe_hash,
            "control_probe_n": self.control_probe_n,
            "scored_holdout_hash": self.scored_holdout_hash,
            "scored_holdout_n": self.scored_holdout_n,
            "control_scored_overlap_n": self.control_scored_overlap_n,
            "oracle_control_enabled": self.oracle_control_enabled,
            "control_mode_label": (
                "CONTROL_DISABLED" if not self.controller_enabled
                else "ORACLE_CONTROL" if self.oracle_control_enabled
                else "NON_ORACLE_CONTROL"
            ),
            "scientific_eval_status": self.scientific_eval_status,
            "controller_actuation_used": self.controller_enabled,
        }


def assign_partition_control_roles(
    *,
    source: "ControlSignalSource | str",
    training_indices: Iterable[int],
    scored_holdout_indices: Iterable[int],
    control_probe_n: int = 32,
    controller_enabled: bool,
) -> PartitionControlRoles:
    resolved_source = ControlSignalSource(source)
    training = _indices(training_indices)
    scored = _indices(scored_holdout_indices)
    probes: tuple[int, ...] = ()
    if resolved_source is ControlSignalSource.TRAINING_DERIVED:
        control = training
    elif resolved_source is ControlSignalSource.CONTROL_PROBE:
        probes = _deterministic_probe_indices(training, int(control_probe_n))
        control = probes
    else:
        control = scored
    roles = PartitionControlRoles(
        source=resolved_source,
        training_indices=training,
        control_indices=control,
        control_probe_indices=probes,
        scored_holdout_indices=scored,
        controller_enabled=bool(controller_enabled),
    )
    if roles.controller_enabled and not roles.oracle_control_enabled and roles.control_scored_overlap_n > 0:
        raise EvaluationLeakageError(
            "INVALID_EVALUATION_LEAKAGE: non-oracle partition control overlaps "
            f"{roles.control_scored_overlap_n} scored-holdout camera(s)"
        )
    return roles


_LATEST_HISTORICAL_REPAIR = None
_PENDING_HISTORICAL_REPAIR_EXTRA_STEPS = 0
_REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS = 0


def reset_historical_repair_state() -> None:
    """Reset per-run historical-repair decision and work accounting."""
    global _LATEST_HISTORICAL_REPAIR
    global _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS
    global _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS
    _LATEST_HISTORICAL_REPAIR = None
    _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS = 0
    _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS = 0


def latest_historical_repair_extra_steps() -> int:
    return max(0, int(_PENDING_HISTORICAL_REPAIR_EXTRA_STEPS))


def consume_historical_repair_extra_steps() -> int:
    """Consume the latest repair event's optimizer budget exactly once."""
    global _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS
    global _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS
    extra = max(0, int(_PENDING_HISTORICAL_REPAIR_EXTRA_STEPS))
    _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS = 0
    _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS += extra
    return extra


def latest_historical_repair_report() -> dict:
    decision = _LATEST_HISTORICAL_REPAIR
    if decision is None:
        return {
            "mode": None,
            "selected_indices": [],
            "extra_optimizer_steps": 0,
            "pending_extra_optimizer_steps": 0,
            "realized_cumulative_extra_optimizer_steps": int(
                _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS
            ),
            "reason": "no_decision_yet",
        }
    return {
        "mode": decision.mode.value,
        "target_regions": list(decision.target_regions),
        "selected_indices": list(decision.selected_indices),
        "requested_views": decision.requested_views,
        "realized_views": decision.realized_views,
        "extra_optimizer_steps": decision.extra_optimizer_steps,
        "pending_extra_optimizer_steps": int(_PENDING_HISTORICAL_REPAIR_EXTRA_STEPS),
        "realized_cumulative_extra_optimizer_steps": int(
            _REALIZED_HISTORICAL_REPAIR_EXTRA_STEPS
        ),
        "protected_frontier_fraction": decision.protected_frontier_fraction,
        "reason": decision.reason,
    }


@dataclass(frozen=True)
class PartitionFocusDecision:
    source: ControlSignalSource
    partition_means: dict[int, float]
    partition_counts: dict[int, int]
    bad_partitions: tuple[int, ...]
    good_partitions: tuple[int, ...]
    priority_weights: dict[int, float]
    promote_indices: tuple[int, ...]
    demote_indices: tuple[int, ...]
    repair_mode: str = "targeted_history"
    repair_extra_optimizer_steps: int = 0

    @property
    def partition_pressure(self) -> float:
        if not self.partition_means:
            return 0.0
        return min(1.0, len(self.bad_partitions) / len(self.partition_means))

    def action_trace(self) -> dict:
        return {
            "bad_partitions": list(self.bad_partitions),
            "good_partitions": list(self.good_partitions),
            "partition_means": {str(key): self.partition_means[key] for key in sorted(self.partition_means)},
            "priority_weights": {str(key): self.priority_weights[key] for key in sorted(self.priority_weights)},
            "promote_indices": list(self.promote_indices),
            "demote_indices": list(self.demote_indices),
            "historical_repair_mode": self.repair_mode,
            "historical_repair_extra_optimizer_steps": self.repair_extra_optimizer_steps,
            "historical_repair": latest_historical_repair_report(),
        }


def _selected_observations(
    roles: PartitionControlRoles,
    *,
    training_observations: Mapping[int, float],
    control_probe_observations: Mapping[int, float],
    scored_holdout_observations: Mapping[int, float],
) -> dict[int, float]:
    if roles.source is ControlSignalSource.TRAINING_DERIVED:
        candidate = training_observations
    elif roles.source is ControlSignalSource.CONTROL_PROBE:
        candidate = control_probe_observations
    else:
        candidate = scored_holdout_observations
    selected = {index: float(candidate[index]) for index in roles.control_indices if index in candidate}
    missing = sorted(set(roles.control_indices).difference(selected))
    if missing:
        raise ValueError(
            "partition control signal incomplete: missing observation(s) for "
            f"source indices {missing[:8]}"
        )
    return selected


def _historical_repair_selection_by_region(
    cached: tuple[int, ...],
    *,
    cached_region_ids: Mapping[int, tuple[int, ...]],
    target_regions: tuple[int, ...],
    max_promote: int,
    prior_usefulness: "Mapping[int, float] | None" = None,
    region_weights: "Mapping[int, float] | None" = None,
):
    """Shared selection engine behind both ``decide_partition_focus`` and
    ``decide_spatial_partition_focus`` (#1723).  Region identity is supplied
    by the caller rather than derived here, so the same ranking/eligibility
    policy in ``historical_repair.py`` serves an ordinal-partition caller
    (one region per frame) and a spatial-cell caller (many regions per
    frame) without duplication.
    """
    global _LATEST_HISTORICAL_REPAIR
    global _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS
    from eligsir.mapping.scheduler.historical_repair import (
        HistoricalFrameEvidence,
        RepairMode,
        select_historical_repair,
    )
    raw_mode = os.environ.get("ELIGSIR_HISTORICAL_REPAIR_MODE", RepairMode.TARGETED_HISTORY.value)
    try:
        mode = RepairMode(raw_mode)
    except ValueError as exc:
        raise ValueError(f"unknown ELIGSIR_HISTORICAL_REPAIR_MODE={raw_mode!r}") from exc
    usefulness = prior_usefulness or {}
    evidence = [
        HistoricalFrameEvidence(
            source_index=index,
            tier="icecold",
            prior_usefulness=float(usefulness.get(index, 0.0)),
            region_ids=tuple(cached_region_ids.get(index, ())),
        )
        for index in cached
    ]
    _LATEST_HISTORICAL_REPAIR = select_historical_repair(
        evidence,
        target_regions=target_regions,
        max_views=max(0, int(max_promote)),
        mode=mode,
        region_weights=region_weights,
        event_ordinal=int(os.environ.get("ELIGSIR_HISTORICAL_REPAIR_EVENT_ORDINAL", "0")),
        extra_optimizer_step_cap=max(
            0, int(os.environ.get("ELIGSIR_HISTORICAL_REPAIR_EXTRA_STEP_CAP", "0"))
        ),
        protected_frontier_fraction=float(
            os.environ.get("ELIGSIR_HISTORICAL_REPAIR_FRONTIER_FRACTION", "0.5")
        ),
    )
    _PENDING_HISTORICAL_REPAIR_EXTRA_STEPS = max(
        0, int(_LATEST_HISTORICAL_REPAIR.extra_optimizer_steps)
    )
    return _LATEST_HISTORICAL_REPAIR


def _historical_repair_selection(
    cached: tuple[int, ...],
    *,
    bad_partitions: tuple[int, ...],
    max_source_index: int,
    partition_count: int,
    max_promote: int,
    prior_usefulness: "Mapping[int, float] | None" = None,
    partition_weights: "Mapping[int, float] | None" = None,
):
    """Run #116/#953 policy over the existing cache identities.

    Existing partition-focus behavior is the default: TARGETED_HISTORY with the
    same max-promote budget. ResearchFlow arms can select random_history,
    active_only or targeted_plus_budget through environment variables without
    introducing a second cache or trainer path.

    ``prior_usefulness`` (issue #953/#1191, default ``None``/empty -- exactly
    reproduces prior behavior) is an optional ``{source_index: weight}`` map of
    a real, train-time-only usefulness signal (e.g. from
    ``eligsir.mapping.scheduler.historical_usefulness.EmaUsefulnessTracker``)
    threaded into each candidate's ``HistoricalFrameEvidence.prior_usefulness``.
    TARGETED_HISTORY already ranks by this field
    (``historical_repair.py::_target_score``); without a real producer it was
    always ``0.0`` for every candidate, so ranking within a target region
    degenerated to an arbitrary ascending-``source_index`` tie-break.
    """
    cached_region_ids = {
        index: (partition_id(index, max_source_index, k=partition_count),)
        for index in cached
    }
    return _historical_repair_selection_by_region(
        cached,
        cached_region_ids=cached_region_ids,
        target_regions=bad_partitions,
        max_promote=max_promote,
        prior_usefulness=prior_usefulness,
        region_weights=partition_weights,
    )


def decide_partition_focus(
    *,
    roles: PartitionControlRoles,
    training_observations: Mapping[int, float],
    control_probe_observations: Mapping[int, float],
    scored_holdout_observations: Mapping[int, float],
    max_source_index: int,
    partition_count: int,
    outlier_k: float,
    active_training_indices: Iterable[int],
    cached_indices: Iterable[int],
    max_promote: int,
    historical_usefulness: "Mapping[int, float] | None" = None,
    prior_bad_partitions: "frozenset[int] | None" = None,
    hysteresis_release_factor: float = 0.5,
) -> PartitionFocusDecision:
    observations = _selected_observations(
        roles,
        training_observations=training_observations,
        control_probe_observations=control_probe_observations,
        scored_holdout_observations=scored_holdout_observations,
    )
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for index, score in observations.items():
        pid = partition_id(index, max_source_index, k=partition_count)
        sums[pid] = sums.get(pid, 0.0) + score
        counts[pid] = counts.get(pid, 0) + 1
    means = {pid: sums[pid] / counts[pid] for pid in sorted(counts)}
    bad = tuple(sorted(detect_outlier_partitions(
        means, k=float(outlier_k), partition_n=counts,
        min_views_per_partition=2, direction="low",
    )))
    if not bad:
        target = max(means.values(), default=float("inf"))
        bad = tuple(
            partition for partition, value in sorted(means.items())
            if target - value >= float(outlier_k)
        )
    # Hysteresis retention (issue: partition focus produced an unstable
    # per-partition PSNR curve). `bad`/`good` above are recomputed from a
    # bare per-tick MAD-outlier snapshot with no memory of the previous
    # decision -- a partition promoted last tick for being bad can improve
    # just enough to clear the SAME entry threshold this tick, get dropped
    # from `bad`, lose its promoted weight, drift back down from lack of
    # attention, and re-enter `bad` next tick: a bang-bang control loop with
    # no dead-band. `prior_bad_partitions` (the caller's own previous
    # decision.bad_partitions, threaded back in) lets a still-recovering
    # partition remain flagged bad until it clears a STRICTER exit
    # threshold (closer to the median than the entry threshold), rather than
    # merely ceasing to be a fresh outlier. Additive and backward
    # compatible: with the default empty `prior_bad_partitions`, this block
    # is a no-op and `bad` is unchanged from the historical computation
    # above. Reuses `detect_outlier_partitions` at a scaled-down `k` for the
    # exit threshold rather than re-deriving median/MAD by hand -- keeps the
    # same `min_views_per_partition` filter the entry computation above
    # already applies (otherwise a low-view partition could be retained via
    # hysteresis without ever having qualified as an outlier in the first
    # place). The >= 3 guard mirrors detect_outlier_partitions' own
    # min_partitions=3 floor explicitly -- below that it always returns
    # empty anyway, so gating here makes the "no-op with too few
    # partitions" behavior visible at the call site instead of silently
    # falling out of the helper.
    if prior_bad_partitions and len(means) >= 3:
        still_below_exit_threshold = detect_outlier_partitions(
            means, k=float(outlier_k) * float(hysteresis_release_factor),
            partition_n=counts, min_views_per_partition=2, direction="low",
        )
        retained = prior_bad_partitions & still_below_exit_threshold
        if retained:
            bad = tuple(sorted(set(bad) | retained))
    good = tuple(sorted(detect_outlier_partitions(
        means, k=float(outlier_k), partition_n=counts,
        min_views_per_partition=2, direction="high",
    )))

    partition_weights: dict[int, float] = {}
    if bad:
        vals = list(means.values())
        median = float(np.median(vals))
        mad = max(float(np.median(np.abs(np.asarray(vals) - median))) * 1.4826, 1e-6)
        partition_weights = {
            pid: 1.0 + max(0.0, median - means.get(pid, median)) / mad
            for pid in bad
        }

    active = _indices(active_training_indices)
    cached = _indices(cached_indices)
    priorities = {
        index: partition_weights[partition_id(index, max_source_index, k=partition_count)]
        for index in active
        if partition_id(index, max_source_index, k=partition_count) in partition_weights
    }

    repair = _historical_repair_selection(
        cached,
        bad_partitions=bad,
        max_source_index=max_source_index,
        partition_count=partition_count,
        max_promote=max_promote,
        prior_usefulness=historical_usefulness,
        partition_weights=partition_weights,
    )
    promote = tuple(int(i) for i in repair.selected_indices)
    for index in promote:
        pid = partition_id(index, max_source_index, k=partition_count)
        priorities[index] = partition_weights.get(pid, 1.0)

    probe_ids = set(roles.control_probe_indices)
    demotable = tuple(
        index for index in active
        if index not in probe_ids
        and partition_id(index, max_source_index, k=partition_count) in good
    )
    demote = demotable[: len(promote)]
    return PartitionFocusDecision(
        source=roles.source,
        partition_means=means,
        partition_counts=counts,
        bad_partitions=bad,
        good_partitions=good,
        priority_weights=priorities,
        promote_indices=promote,
        demote_indices=demote,
        repair_mode=repair.mode.value,
        repair_extra_optimizer_steps=int(repair.extra_optimizer_steps),
    )


def decide_spatial_partition_focus(
    *,
    roles: PartitionControlRoles,
    cell_psnr: Mapping[int, float],
    cell_view_counts: Mapping[int, int],
    outlier_k: float,
    active_region_ids: Mapping[int, tuple[int, ...]],
    cached_region_ids: Mapping[int, tuple[int, ...]],
    max_promote: int,
    historical_usefulness: "Mapping[int, float] | None" = None,
    max_bad_fraction: float = 0.2,
) -> PartitionFocusDecision:
    """Cell-substrate promote/demote actuation (design: issue #1723).

    Same output shape and actuation semantics as ``decide_partition_focus``
    -- "partition" reads as "world microcell" throughout.  The critical
    difference is region membership: an ordinal partition assigns exactly
    one region to each frame (``partition_id``), whereas a frame here can
    cover many microcells at once (``active_region_ids``/``cached_region_ids``
    are many-to-many, resolved by the *caller* -- typically via
    ``tracker.cell_map_for(camera, sensor_depth)``, which is geometry-only
    and needs no render, so it is available even for a cached/evicted frame
    with no recent training pass). Not feeding spatial debt through a single
    ordinal bin per frame avoids the dilution ``cell_debt_actuation`` mode
    was built to prevent (issue #1259 finding 3): a frame's weight here is
    driven by its single worst matched region (see
    ``historical_repair.py::_target_score``'s ``severity = max(...)``), not
    an average across its cells.

    ``cell_psnr``/``cell_view_counts`` are expected to come from
    ``SpatialMicrocellReplayFocusController``'s already-computed per-cell
    telemetry (``cell_psnr_db``, sample counts) -- this function does not
    itself track cell quality history, only turns a snapshot into an
    actuation decision, mirroring ``decide_partition_focus``'s relationship
    to the ordinal partition-mean computation it performs inline.

    Demote eligibility is intentionally conservative: an active frame is
    only demotable if *none* of its covered cells are flagged bad (not the
    stricter "all covered cells are flagged good", which would make demote
    near-empty since most cells are statistically neutral, and not the
    laxer "at least one covered cell is good", which could demote a frame
    still needed for a weak cell it also happens to cover).

    ``max_bad_fraction`` bounds how many cells ``bad`` can ever contain.
    ``detect_outlier_partitions``'s median/MAD threshold was tuned against
    ordinal_partition_count=6 (where 1-2 of 6 partitions typically clear it).
    Reused unmodified at real cell counts (~89 on FR3), the same threshold
    flagged 64-84 of ~89 cells bad simultaneously in a real GPU run (2026-09-03,
    #1723 A/B) -- noisier per-cell PSNR estimates (far fewer samples per cell
    than per ordinal bin) trip the same k more often. With ~80% of cells
    flagged, "target the worst regions" collapses: promote spreads across
    nearly the whole scene instead of concentrating on a genuine weak
    subset, AND demote structurally can never fire (virtually every active
    frame touches at least one of the many "bad" cells). Capping bad to its
    worst ``max_bad_fraction`` keeps both promote and demote meaningfully
    selective regardless of how permissive the underlying outlier test is
    at this cell count.
    """
    means = dict(cell_psnr)
    counts = dict(cell_view_counts)
    bad = tuple(sorted(detect_outlier_partitions(
        means, k=float(outlier_k), partition_n=counts,
        min_views_per_partition=2, direction="low",
    )))
    if not bad:
        target = max(means.values(), default=float("inf"))
        bad = tuple(
            cell for cell, value in sorted(means.items())
            if target - value >= float(outlier_k)
        )
    max_bad_cells = max(1, math.ceil(float(max_bad_fraction) * len(means))) if means else 0
    if len(bad) > max_bad_cells:
        bad = tuple(sorted(sorted(bad, key=lambda cell: means[cell])[:max_bad_cells]))
    good = tuple(sorted(detect_outlier_partitions(
        means, k=float(outlier_k), partition_n=counts,
        min_views_per_partition=2, direction="high",
    )))
    max_good_cells = max(1, math.ceil(float(max_bad_fraction) * len(means))) if means else 0
    if len(good) > max_good_cells:
        good = tuple(sorted(
            sorted(good, key=lambda cell: means[cell], reverse=True)[:max_good_cells]
        ))

    cell_weights: dict[int, float] = {}
    if bad:
        vals = list(means.values())
        median = float(np.median(vals))
        mad = max(float(np.median(np.abs(np.asarray(vals) - median))) * 1.4826, 1e-6)
        cell_weights = {
            cell: 1.0 + max(0.0, median - means.get(cell, median)) / mad
            for cell in bad
        }

    bad_set = set(bad)
    active = _indices(active_region_ids)
    cached = _indices(cached_region_ids)

    priorities: dict[int, float] = {}
    for index in active:
        matched = bad_set.intersection(active_region_ids.get(index, ()))
        if matched:
            priorities[index] = max(cell_weights.get(cell, 1.0) for cell in matched)

    repair = _historical_repair_selection_by_region(
        cached,
        cached_region_ids=cached_region_ids,
        target_regions=bad,
        max_promote=max_promote,
        prior_usefulness=historical_usefulness,
        region_weights=cell_weights,
    )
    promote = tuple(int(i) for i in repair.selected_indices)
    for index in promote:
        matched = bad_set.intersection(cached_region_ids.get(index, ()))
        if matched:
            priorities[index] = max(cell_weights.get(cell, 1.0) for cell in matched)

    probe_ids = set(roles.control_probe_indices)
    demotable = tuple(
        index for index in active
        if index not in probe_ids
        and active_region_ids.get(index)
        and bad_set.isdisjoint(active_region_ids[index])
    )
    demote = demotable[: len(promote)]
    return PartitionFocusDecision(
        source=roles.source,
        partition_means=means,
        partition_counts=counts,
        bad_partitions=bad,
        good_partitions=good,
        priority_weights=priorities,
        promote_indices=promote,
        demote_indices=demote,
        repair_mode=repair.mode.value,
        repair_extra_optimizer_steps=int(repair.extra_optimizer_steps),
    )


@dataclass(frozen=True)
class DebtAllocatorDecision:
    """One tick's unified promote+demote batch selection (issue #1723 v2).

    Replaces decide_partition_focus/decide_spatial_partition_focus's
    separate promote-via-historical-repair and demote-via-exclusion logic
    with a single deficit-round-robin ranking: every known frame (active
    or cached) is ranked by the debt of its worst covered cell (max debt
    across cells it covers), debt ties are broken by lowest PSNR of that
    worst cell (prefer serving frames covering lower-quality cells), and
    the remaining ties are broken by index (stable ordering). The top
    `batch_size` are selected, and any PREVIOUSLY-active index that did not
    make this tick's selection is reported as demoted. Demote therefore
    falls out of the same ranking that drives promote, rather than being a
    separate mechanism that has to independently agree with it.
    """

    debt: dict[int, float]
    selected_indices: tuple[int, ...]
    demoted_indices: tuple[int, ...]


def decide_partition_focus_debt_allocator(
    *,
    cell_psnr: Mapping[int, float],
    active_region_ids: Mapping[int, tuple[int, ...]],
    cached_region_ids: Mapping[int, tuple[int, ...]],
    debt: Mapping[int, float],
    target: Mapping[int, float],
    batch_size: int,
) -> DebtAllocatorDecision:
    """Rank all known frames by worst-covered-cell debt; select top batch_size.

    ``debt``/``target`` are per-cell, keyed the same as ``cell_psnr``. Debt
    accrues by ``target[cell]`` every call for every cell present in
    ``target`` (whether or not it was served last tick) and is paid down by
    1.0 per cell for every selected frame that covers it. Caller owns
    persisting ``debt`` across ticks (pure function, no hidden state --
    same contract as the rest of this module).

    **Ranking and tiebreaking:** Frames are ranked by the max debt among
    their covered cells (worst-covered-cell debt). Ties are broken by the
    lowest PSNR of that worst cell (prefer serving frames covering lower-
    quality cells), then by index (stable ordering).

    **CRITICAL caller obligation — debt-carry-forward domain:** Any cell
    present in the incoming ``debt`` mapping but ABSENT from ``target`` this
    tick will lose its accumulated debt entirely (not frozen -- dropped). The
    caller MUST pass the full/stable cell domain in ``target`` every tick,
    or debt for omitted cells is silently lost. Failure to do so makes the
    allocator's deficit-round-robin reasoning incoherent (a cell can drop off
    the priority ranking if the caller forgets to include it in target).
    """
    new_debt = {cell: float(debt.get(cell, 0.0)) + float(rate) for cell, rate in target.items()}

    all_region_ids: dict[int, tuple[int, ...]] = {**cached_region_ids, **active_region_ids}
    if not all_region_ids:
        return DebtAllocatorDecision(debt=new_debt, selected_indices=(), demoted_indices=())

    def priority(index: int) -> float:
        cells = all_region_ids[index]
        return max((new_debt.get(cell, 0.0) for cell in cells), default=0.0)

    def worst_psnr(index: int) -> float:
        cells = all_region_ids[index]
        return min((cell_psnr.get(cell, float('inf')) for cell in cells), default=float('inf'))

    ranked = sorted(all_region_ids, key=lambda idx: (-priority(idx), worst_psnr(idx), idx))
    selected = tuple(ranked[: max(0, int(batch_size))])

    for index in selected:
        for cell in all_region_ids[index]:
            if cell in new_debt:
                new_debt[cell] -= 1.0

    # NOTE: not comparable to decide_partition_focus/decide_spatial_partition_
    # focus's demote counts -- those legacy paths cap demotes at
    # demotable[:len(promote)] (roughly symmetric with promotes), while this
    # reports every previously-active index that didn't make this tick's
    # selection (structurally larger, not a bug).
    selected_set = set(selected)
    demoted = tuple(sorted(idx for idx in active_region_ids if idx not in selected_set))

    return DebtAllocatorDecision(debt=new_debt, selected_indices=selected, demoted_indices=demoted)


class PartitionControlRecorder:
    """Append-only decision trace plus compact report summary."""

    def __init__(self, model_path) -> None:
        self.path = Path(model_path) / "partition_control_trace.jsonl"
        self._hasher = hashlib.sha256()
        self._n_decisions = 0
        self._latest: Optional[dict] = None
        if self.path.is_file():
            for raw_line in self.path.read_bytes().splitlines(keepends=True):
                self._hasher.update(raw_line)
                self._n_decisions += 1
                try:
                    self._latest = json.loads(raw_line)
                except Exception:
                    self._latest = None

    def record(
        self,
        *,
        iteration: int,
        roles: PartitionControlRoles,
        decision: PartitionFocusDecision,
        telemetry: Optional[Mapping[str, object]] = None,
    ) -> None:
        record = {
            "iteration": int(iteration),
            **roles.to_report_dict(),
            **decision.action_trace(),
            "partition_pressure": decision.partition_pressure,
        }
        if telemetry:
            overlap = set(record).intersection(telemetry)
            if overlap:
                raise ValueError(
                    "partition control telemetry cannot overwrite canonical fields: "
                    + ", ".join(sorted(overlap))
                )
            record.update(dict(telemetry))
        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(line)
        self._hasher.update(line)
        self._n_decisions += 1
        self._latest = record

    def report_block(self, roles: PartitionControlRoles) -> dict:
        report = roles.to_report_dict()
        report.update({
            "n_control_decisions": self._n_decisions,
            "action_trace_hash": self._hasher.hexdigest()[:16] if self._n_decisions else None,
            "action_trace_artifact": "partition_control_trace.jsonl",
            "latest_decision": self._latest,
            "historical_repair": latest_historical_repair_report(),
        })
        return report


class PartitionControlRuntime:
    """Refresh bounded runtime control roles and expose the current cohort."""

    def __init__(
        self,
        *,
        source: "ControlSignalSource | str",
        control_probe_n: int,
        model_path,
        controller_enabled: bool = True,
    ) -> None:
        self.source = ControlSignalSource(source)
        self.control_probe_n = int(control_probe_n)
        self.controller_enabled = bool(controller_enabled)
        self.recorder = PartitionControlRecorder(model_path)
        self._probe_cameras: dict[int, object] = {}

    @property
    def control_cameras(self) -> tuple[object, ...]:
        """Return the current bounded probe cohort in stable source order."""

        return tuple(
            self._probe_cameras[index] for index in sorted(self._probe_cameras)
        )

    @staticmethod
    def _camera_map(cameras: Iterable[object], *, strict: bool = True) -> dict[int, object]:
        result: dict[int, object] = {}
        for camera in cameras:
            index = camera_source_index(camera)
            if index is None:
                if strict:
                    raise ValueError("partition control requires stable source-frame identity for every camera")
                continue
            index = int(index)
            if index in result:
                if strict:
                    raise ValueError(f"partition control camera identity is not unique: source index {index}")
                continue
            result[index] = camera
        return result

    def prepare(
        self,
        training_cameras: Iterable[object],
        scored_holdout_cameras: Iterable[object],
        *,
        publish: bool = True,
    ) -> tuple[PartitionControlRoles, list[object]]:
        training = self._camera_map(training_cameras, strict=self.controller_enabled)
        scored = self._camera_map(scored_holdout_cameras, strict=self.controller_enabled)
        if self.source is ControlSignalSource.CONTROL_PROBE:
            roles = assign_partition_control_roles(
                source=self.source,
                training_indices=_indices(training),
                scored_holdout_indices=_indices(scored),
                control_probe_n=self.control_probe_n,
                controller_enabled=self.controller_enabled,
            )
            probe_cameras = {
                index: training[index] for index in roles.control_indices
            }
            if publish:
                self._probe_cameras = probe_cameras
            selected = [probe_cameras[index] for index in roles.control_indices]
            return roles, selected

        roles = assign_partition_control_roles(
            source=self.source,
            training_indices=training,
            scored_holdout_indices=scored,
            control_probe_n=self.control_probe_n,
            controller_enabled=self.controller_enabled,
        )
        source_cameras = scored if self.source is ControlSignalSource.SCORED_HOLDOUT_ORACLE else training
        return roles, [source_cameras[index] for index in roles.control_indices]

    @staticmethod
    def observations_from_tick(tick) -> dict[int, float]:
        if tick is None or tick.psnr_tensor is None:
            return {}
        values = tick.psnr_tensor.detach().cpu().tolist()
        observations: dict[int, float] = {}
        for camera, value in zip(tick.cams, values):
            index = camera_source_index(camera)
            if index is None:
                raise ValueError("partition control tick contains a camera without stable source identity")
            observations[int(index)] = float(value)
        return observations

    def prepare_tick(
        self,
        *,
        training_cameras: Iterable[object],
        scored_holdout_cameras: Iterable[object],
        scored_holdout_tick,
        render_control_tick,
    ) -> tuple[PartitionControlRoles, dict[str, dict[int, float]]]:
        roles, control_cameras = self.prepare(training_cameras, scored_holdout_cameras)
        scored_observations = self.observations_from_tick(scored_holdout_tick)
        control_tick = scored_holdout_tick if roles.oracle_control_enabled else render_control_tick(control_cameras)
        control_observations = self.observations_from_tick(control_tick)
        observations = {
            "training_derived": {},
            "control_probe": {},
            "scored_holdout": scored_observations,
        }
        if roles.source is ControlSignalSource.TRAINING_DERIVED:
            observations["training_derived"] = control_observations
        elif roles.source is ControlSignalSource.CONTROL_PROBE:
            observations["control_probe"] = control_observations
        return roles, observations


__all__ = [
    "ControlSignalSource",
    "EvaluationLeakageError",
    "PartitionControlRecorder",
    "PartitionControlRoles",
    "PartitionControlRuntime",
    "PartitionFocusDecision",
    "assign_partition_control_roles",
    "decide_partition_focus",
    "decide_spatial_partition_focus",
    "latest_historical_repair_extra_steps",
    "latest_historical_repair_report",
    "consume_historical_repair_extra_steps",
    "reset_historical_repair_state",
]
