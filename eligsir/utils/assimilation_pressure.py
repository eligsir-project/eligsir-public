"""Canonical assimilation-pressure telemetry and bounded control state."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class AssimilationPressureTier(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class AssimilationSupportSummary:
    """Compact tensor-free support-debt snapshot, latched after production.

    Provenance fields (optimizer_updates, topology_version, parameter_version,
    population_size) track the state at the time the summary was latched.
    When assimilation control is enabled, these fields allow the controller to
    detect stale snapshots (e.g., when topology has changed since the summary
    was produced) and avoid blocking on same-transaction reductions.
    """
    debt: float
    support_eq_0: int
    support_eq_1: int
    support_ge_target: int
    support_pool_n: int
    min_support_views: int
    generation: int = 0
    optimizer_updates: int = 0
    topology_version: int = 0
    parameter_version: int = 0
    population_size: int = 0


_TIER_THRESHOLDS = (0.25, 0.5, 0.75)
_LATEST_SUPPORT_DEBT: float | None = None
_LATEST_SUPPORT_HISTOGRAM: dict[str, int] = {}
_LATEST_ASSIMILATION_SUPPORT_SUMMARY: AssimilationSupportSummary | None = None
_LATEST_SIGNALS: dict[str, float] | None = None
_ACTIVE_CONTROLLER = None
_SUMMARY_GENERATION = 0

# Canonical run configuration.  Defaults are behavior-neutral and are replaced
# once per streaming run from persisted cfg_args by PartitionControlRuntime.
_CONTROL_ENABLED = False
_CONTROL_DWELL_TICKS = 3
_CONTROL_MAX_EXTRA_STEPS = 2
_CONTROL_MAX_REPLAY_BIAS = 2.0


def configure_assimilation_control(
    *,
    enabled: bool = False,
    dwell_ticks: int = 3,
    max_extra_steps: int = 2,
    max_replay_bias: float = 2.0,
) -> None:
    """Install the run's persisted controller configuration.

    Reconfiguration resets the hysteretic owner so two sequential runs in one
    Python process cannot inherit controller state from each other.
    """
    global _CONTROL_ENABLED, _CONTROL_DWELL_TICKS, _CONTROL_MAX_EXTRA_STEPS
    global _CONTROL_MAX_REPLAY_BIAS, _ACTIVE_CONTROLLER
    _CONTROL_ENABLED = bool(enabled)
    _CONTROL_DWELL_TICKS = max(1, int(dwell_ticks))
    _CONTROL_MAX_EXTRA_STEPS = max(0, int(max_extra_steps))
    _CONTROL_MAX_REPLAY_BIAS = max(1.0, float(max_replay_bias))
    _ACTIVE_CONTROLLER = None


def assimilation_control_config() -> dict:
    return {
        "enabled": bool(_CONTROL_ENABLED),
        "dwell_ticks": int(_CONTROL_DWELL_TICKS),
        "max_extra_steps": int(_CONTROL_MAX_EXTRA_STEPS),
        "max_replay_bias": float(_CONTROL_MAX_REPLAY_BIAS),
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def support_debt_from_counts(
    support_counts: Iterable[float],
    *,
    min_support_views: int,
    smoothing: float = 1.0,
) -> float:
    """Continuous deficit over the full support population, not only provisional rows."""
    target = int(min_support_views)
    if target <= 0:
        raise ValueError("min_support_views must be > 0")
    counts = [max(0.0, float(x)) for x in support_counts]
    if not counts:
        return 0.0
    denom = float(target) + max(0.0, float(smoothing))
    debts = [max(0.0, float(target) - min(c, float(target))) / denom for c in counts]
    return _clamp01(sum(debts) / len(debts))


def support_histogram(support_counts: Iterable[float], *, min_support_views: int) -> dict[str, int]:
    counts = [max(0, int(x)) for x in support_counts]
    target = max(1, int(min_support_views))
    return {
        "support_eq_0": sum(x == 0 for x in counts),
        "support_eq_1": sum(x == 1 for x in counts),
        "support_ge_target": sum(x >= target for x in counts),
        "support_pool_n": len(counts),
    }


def publish_support_counts(
    support_counts,
    *,
    min_support_views: int,
    smoothing: float = 1.0,
    optimizer_updates: int = 0,
    topology_version: int = 0,
    parameter_version: int = 0,
    population_size: int = 0,
) -> float:
    """Produce support debt with one compact tensor-to-host reduction; latch immutable summary.

    Returns the debt value. The complete summary (including histogram and provenance) is latched
    in _LATEST_ASSIMILATION_SUPPORT_SUMMARY and accessible via
    latest_assimilation_support_summary().

    Provenance parameters (optimizer_updates, topology_version, parameter_version, population_size)
    are attached to the summary at latch time to enable the controller to detect stale snapshots
    when topology or parameters have changed.
    """
    global _LATEST_SUPPORT_DEBT, _LATEST_SUPPORT_HISTOGRAM, _LATEST_ASSIMILATION_SUPPORT_SUMMARY, _SUMMARY_GENERATION
    target = max(1, int(min_support_views))
    try:
        import torch
        if torch.is_tensor(support_counts):
            counts = support_counts.detach().to(dtype=torch.float32).clamp_min(0)
            if counts.numel() == 0:
                debt = 0.0
                support_eq_0 = 0
                support_eq_1 = 0
                support_ge_target = 0
                support_pool_n = 0
            else:
                denom = float(target) + max(0.0, float(smoothing))
                deficits = (float(target) - counts.clamp_max(float(target))).clamp_min(0) / denom
                compact = torch.stack((
                    deficits.mean(),
                    (counts == 0).sum().to(torch.float32),
                    (counts == 1).sum().to(torch.float32),
                    (counts >= target).sum().to(torch.float32),
                    torch.tensor(float(counts.numel()), device=counts.device),
                )).detach().cpu().tolist()
                debt = float(compact[0])
                support_eq_0 = int(compact[1])
                support_eq_1 = int(compact[2])
                support_ge_target = int(compact[3])
                support_pool_n = int(compact[4])
        else:
            values = list(support_counts)
            debt = support_debt_from_counts(values, min_support_views=target, smoothing=smoothing)
            hist = support_histogram(values, min_support_views=target)
            support_eq_0 = hist["support_eq_0"]
            support_eq_1 = hist["support_eq_1"]
            support_ge_target = hist["support_ge_target"]
            support_pool_n = hist["support_pool_n"]
    except Exception:
        values = list(support_counts)
        debt = support_debt_from_counts(values, min_support_views=target, smoothing=smoothing)
        hist = support_histogram(values, min_support_views=target)
        support_eq_0 = hist["support_eq_0"]
        support_eq_1 = hist["support_eq_1"]
        support_ge_target = hist["support_ge_target"]
        support_pool_n = hist["support_pool_n"]
    # Latch the immutable tensor-free summary FIRST
    debt_clamped = _clamp01(debt)
    _SUMMARY_GENERATION += 1
    _LATEST_ASSIMILATION_SUPPORT_SUMMARY = AssimilationSupportSummary(
        debt=debt_clamped,
        support_eq_0=support_eq_0,
        support_eq_1=support_eq_1,
        support_ge_target=support_ge_target,
        support_pool_n=support_pool_n,
        min_support_views=target,
        generation=_SUMMARY_GENERATION,
        optimizer_updates=max(0, int(optimizer_updates)),
        topology_version=max(0, int(topology_version)),
        parameter_version=max(0, int(parameter_version)),
        population_size=max(0, int(population_size)),
    )
    # Derive legacy globals from the latched summary
    _LATEST_SUPPORT_DEBT = debt_clamped
    _LATEST_SUPPORT_HISTOGRAM = {
        "support_eq_0": support_eq_0,
        "support_eq_1": support_eq_1,
        "support_ge_target": support_ge_target,
        "support_pool_n": support_pool_n,
    }
    return _LATEST_SUPPORT_DEBT


def latest_support_debt(default: float = 0.0) -> float:
    return float(default if _LATEST_SUPPORT_DEBT is None else _LATEST_SUPPORT_DEBT)


def latest_support_histogram() -> dict[str, int]:
    return dict(_LATEST_SUPPORT_HISTOGRAM)


def latest_assimilation_support_summary() -> AssimilationSupportSummary | None:
    """Return the latest latched immutable support summary, or None if not yet produced."""
    return _LATEST_ASSIMILATION_SUPPORT_SUMMARY


def flush_assimilation_support() -> AssimilationSupportSummary | None:
    """Emit the latest latched support summary (source-complete explicit flush).

    Reads and returns the most recently latched summary without performing
    any new device reduction or host readback. Used by final reporting to
    ensure an exact final snapshot of assimilation support state.
    """
    return _LATEST_ASSIMILATION_SUPPORT_SUMMARY


def pressure_score(
    source_insert_rate: float,
    support_debt: float,
    residual_pressure: float,
    partition_pressure: float,
    newborn_pressure: float,
    optimization_rate: float,
) -> float:
    signals = [
        _clamp01(source_insert_rate),
        _clamp01(support_debt),
        _clamp01(residual_pressure),
        _clamp01(partition_pressure),
        _clamp01(newborn_pressure),
        1.0 - _clamp01(optimization_rate),
    ]
    return sum(signals) / len(signals)


def classify_assimilation_pressure(
    source_insert_rate: float,
    support_debt: float,
    residual_pressure: float,
    partition_pressure: float,
    newborn_pressure: float,
    optimization_rate: float,
) -> AssimilationPressureTier:
    """Classify and publish pressure; canonical support state wins when available."""
    global _LATEST_SIGNALS
    canonical_support = latest_support_debt(support_debt)
    _LATEST_SIGNALS = {
        "source_insert_rate": float(source_insert_rate),
        "support_debt": canonical_support,
        "residual_pressure": float(residual_pressure),
        "partition_pressure": float(partition_pressure),
        "newborn_pressure": float(newborn_pressure),
        "optimization_rate": float(optimization_rate),
    }
    score = pressure_score(**_LATEST_SIGNALS)
    lo, mid, hi = _TIER_THRESHOLDS
    if score < lo:
        return AssimilationPressureTier.LOW
    if score < mid:
        return AssimilationPressureTier.MEDIUM
    if score < hi:
        return AssimilationPressureTier.HIGH
    return AssimilationPressureTier.CRITICAL


_CONSOLIDATION_STEP_MULTIPLIER = 1.5


@dataclass(frozen=True)
class ShadowAssimilationDecision:
    would_suppress_insertion: bool
    would_reduce_steps: bool
    would_be_steps: int
    would_bias_replay_weak: bool
    would_suppress_growth: bool


def shadow_decision(tier: AssimilationPressureTier, base_steps: int) -> ShadowAssimilationDecision:
    would_bias_replay_weak = tier >= AssimilationPressureTier.MEDIUM
    would_suppress_insertion = tier >= AssimilationPressureTier.HIGH
    would_reduce_steps = tier >= AssimilationPressureTier.HIGH
    would_be_steps = int(round(base_steps * _CONSOLIDATION_STEP_MULTIPLIER)) if would_reduce_steps else int(base_steps)
    would_suppress_growth = tier >= AssimilationPressureTier.CRITICAL
    return ShadowAssimilationDecision(
        would_suppress_insertion=would_suppress_insertion,
        would_reduce_steps=would_reduce_steps,
        would_be_steps=would_be_steps,
        would_bias_replay_weak=would_bias_replay_weak,
        would_suppress_growth=would_suppress_growth,
    )


@dataclass(frozen=True)
class AssimilationControlDecision:
    tier: AssimilationPressureTier
    score: float
    allow_insertion: bool
    allow_growth: bool
    optimizer_steps: int
    replay_weak_bias: float
    reason: str


class HystereticAssimilationController:
    """Bounded controller with dwell-time hysteresis and hard action caps."""

    def __init__(self, *, dwell_ticks: int = 3, max_extra_steps: int = 2, max_replay_weak_bias: float = 2.0):
        self.dwell_ticks = max(1, int(dwell_ticks))
        self.max_extra_steps = max(0, int(max_extra_steps))
        self.max_replay_weak_bias = max(1.0, float(max_replay_weak_bias))
        self._tier = AssimilationPressureTier.LOW
        self._candidate = self._tier
        self._candidate_ticks = 0

    @property
    def tier(self) -> AssimilationPressureTier:
        return self._tier

    def decide(
        self,
        *,
        source_insert_rate: float,
        support_debt: float,
        residual_pressure: float,
        partition_pressure: float,
        newborn_pressure: float,
        optimization_rate: float,
        base_steps: int,
    ) -> AssimilationControlDecision:
        score = pressure_score(
            source_insert_rate, support_debt, residual_pressure,
            partition_pressure, newborn_pressure, optimization_rate,
        )
        observed = classify_assimilation_pressure(
            source_insert_rate, support_debt, residual_pressure,
            partition_pressure, newborn_pressure, optimization_rate,
        )
        if observed == self._candidate:
            self._candidate_ticks += 1
        else:
            self._candidate = observed
            self._candidate_ticks = 1
        if self._candidate != self._tier and self._candidate_ticks >= self.dwell_ticks:
            self._tier = self._candidate
            self._candidate_ticks = 0

        tier = self._tier
        extra = 0
        if tier >= AssimilationPressureTier.HIGH:
            extra = min(self.max_extra_steps, max(1, int(round(max(1, base_steps) * 0.5))))
        replay_bias = 1.0
        if tier >= AssimilationPressureTier.MEDIUM:
            replay_bias = min(self.max_replay_weak_bias, 1.0 + 0.5 * int(tier))
        return AssimilationControlDecision(
            tier=tier,
            score=float(score),
            allow_insertion=tier < AssimilationPressureTier.HIGH,
            allow_growth=tier < AssimilationPressureTier.CRITICAL,
            optimizer_steps=max(0, int(base_steps)) + extra,
            replay_weak_bias=float(replay_bias),
            reason=f"assimilation_pressure_{tier.name.lower()}",
        )


def assimilation_control_enabled() -> bool:
    return bool(_CONTROL_ENABLED)


def active_control_decision(
    base_steps: int,
    *,
    current_topology_version: int = 0,
) -> AssimilationControlDecision | None:
    """Return the bounded active decision, or None for behavior-neutral default.

    When control is enabled, the decision is based on the latest completed latched
    snapshot, not on the current transaction's reduction. If the latest snapshot's
    topology_version differs from current_topology_version, the snapshot is treated
    as stale and a no-op decision is returned (previous decision is preserved).

    Args:
        base_steps: base optimizer step count
        current_topology_version: the current topology version to check staleness
    """
    global _ACTIVE_CONTROLLER
    if not _CONTROL_ENABLED or _LATEST_SIGNALS is None:
        return None
    if _ACTIVE_CONTROLLER is None:
        _ACTIVE_CONTROLLER = HystereticAssimilationController(
            dwell_ticks=_CONTROL_DWELL_TICKS,
            max_extra_steps=_CONTROL_MAX_EXTRA_STEPS,
            max_replay_weak_bias=_CONTROL_MAX_REPLAY_BIAS,
        )
    # Check if the latest snapshot is stale (topology changed since latch)
    latest_summary = latest_assimilation_support_summary()
    if latest_summary is not None:
        summary_topology = latest_summary.topology_version
        current_topology = max(0, int(current_topology_version))
        if current_topology > 0 and summary_topology > 0 and summary_topology != current_topology:
            # Snapshot is stale due to topology change. Return current tier without updating.
            # This prevents blocking on a same-transaction D2H reduction while topology is
            # in flux. The next decided transaction (after topology stabilizes) will use
            # the current snapshot produced from the new population size.
            return AssimilationControlDecision(
                tier=_ACTIVE_CONTROLLER.tier,
                score=float(_LATEST_SIGNALS.get("support_debt", 0.0)),
                allow_insertion=_ACTIVE_CONTROLLER.tier < AssimilationPressureTier.HIGH,
                allow_growth=_ACTIVE_CONTROLLER.tier < AssimilationPressureTier.CRITICAL,
                optimizer_steps=int(base_steps),
                replay_weak_bias=1.0,
                reason=f"assimilation_pressure_stale_topology_snapshot",
            )
    return _ACTIVE_CONTROLLER.decide(base_steps=int(base_steps), **_LATEST_SIGNALS)


__all__ = [
    "AssimilationControlDecision",
    "AssimilationPressureTier",
    "AssimilationSupportSummary",
    "HystereticAssimilationController",
    "active_control_decision",
    "assimilation_control_config",
    "assimilation_control_enabled",
    "classify_assimilation_pressure",
    "configure_assimilation_control",
    "flush_assimilation_support",
    "latest_assimilation_support_summary",
    "latest_support_debt",
    "latest_support_histogram",
    "pressure_score",
    "publish_support_counts",
    "shadow_decision",
    "support_debt_from_counts",
    "support_histogram",
]
