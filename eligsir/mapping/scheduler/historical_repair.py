"""Bounded historical-repair selection over the existing frame archive.

No second frame database is introduced.  This module only ranks source-frame
identities; StreamingScene/FrameArchive remain the transition executors.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence


class RepairMode(str, Enum):
    ACTIVE_ONLY = "active_only"
    RANDOM_HISTORY = "random_history"
    TARGETED_HISTORY = "targeted_history"
    TARGETED_PLUS_BUDGET = "targeted_plus_budget"


@dataclass(frozen=True, slots=True)
class HistoricalFrameEvidence:
    source_index: int
    tier: str
    region_ids: tuple[int, ...]
    supervision_age_steps: int = 0
    prior_usefulness: float = 0.0
    baseline_diversity: float = 0.0


@dataclass(frozen=True, slots=True)
class HistoricalRepairDecision:
    mode: RepairMode
    target_regions: tuple[int, ...]
    selected_indices: tuple[int, ...]
    requested_views: int
    realized_views: int
    extra_optimizer_steps: int
    protected_frontier_fraction: float
    reason: str


def _target_score(
    frame: HistoricalFrameEvidence,
    target_regions: set[int],
    region_weights: Mapping[int, float],
) -> tuple:
    matched = target_regions.intersection(frame.region_ids)
    overlap = len(matched)
    severity = max((float(region_weights.get(region, 1.0)) for region in matched), default=1.0)
    return (
        -severity,
        -overlap,
        -float(frame.prior_usefulness),
        -float(frame.baseline_diversity),
        -int(frame.supervision_age_steps),
        int(frame.source_index),
    )


def select_historical_repair(
    frames: Sequence[HistoricalFrameEvidence],
    *,
    target_regions: Iterable[int],
    max_views: int,
    mode: RepairMode | str,
    region_weights: Mapping[int, float] | None = None,
    event_ordinal: int = 0,
    extra_optimizer_step_cap: int = 0,
    protected_frontier_fraction: float = 0.5,
) -> HistoricalRepairDecision:
    """Choose a bounded same-cost history set; deterministic for every event.

    RANDOM_HISTORY is a deterministic rotated sample rather than process RNG,
    which makes matched-work control arms reproducible. TARGETED_HISTORY sorts
    by target-region severity, region coverage, then prior usefulness/diversity.
    ``region_weights`` is optional so callers without measured target severity
    retain the historical ordering. Neither mode can
    exceed ``max_views``.
    """
    resolved = RepairMode(mode)
    budget = max(0, int(max_views))
    target = tuple(sorted({int(x) for x in target_regions}))
    target_set = set(target)
    eligible = [f for f in frames if f.tier.lower() in {"cold", "icecold", "golden"}]

    if resolved is RepairMode.ACTIVE_ONLY or budget == 0 or not eligible:
        selected: list[HistoricalFrameEvidence] = []
    elif resolved is RepairMode.RANDOM_HISTORY:
        ordered = sorted(eligible, key=lambda f: int(f.source_index))
        shift = int(event_ordinal) % len(ordered)
        rotated = ordered[shift:] + ordered[:shift]
        selected = rotated[:budget]
    else:
        weights = region_weights or {}
        ranked = sorted(eligible, key=lambda f: _target_score(f, target_set, weights))
        selected = [f for f in ranked if target_set.intersection(f.region_ids)][:budget]

    extra = 0
    if resolved is RepairMode.TARGETED_PLUS_BUDGET and selected:
        extra = max(0, int(extra_optimizer_step_cap))
    return HistoricalRepairDecision(
        mode=resolved,
        target_regions=target,
        selected_indices=tuple(int(f.source_index) for f in selected),
        requested_views=budget,
        realized_views=len(selected),
        extra_optimizer_steps=extra,
        protected_frontier_fraction=max(0.0, min(1.0, float(protected_frontier_fraction))),
        reason=("selected" if selected else "no_eligible_historical_support"),
    )
