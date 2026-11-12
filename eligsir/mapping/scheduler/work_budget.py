"""Deterministic mapping-work allocation from one frozen control snapshot."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from typing import TYPE_CHECKING, Mapping

from eligsir.contracts.control import (
    ControlProposal,
    ControlSnapshot,
    FreshnessMode,
    DepthInsertionPlan,
    MappingTransactionClass,
    MappingWorkPlan,
    MutationBudget,
    ResolutionTier,
    SupervisionWork,
    SupervisionConstellation,
    ObservationDisposition,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.batch_contract import (
    BOUNDED_SUPERVISION_BATCH_SIZES,
    QUALIFIED_SUPERVISION_CAMERA_COUNTS,
    qualified_camera_count_for_batch,
    validate_supervision_batch_size,
    validate_supervision_camera_count,
)
from eligsir.mapping.scheduler.tier_cost_model import (
    FORWARD_BACKWARD,
    LIVE_TIERS,
    UNKNOWN_DEVICE,
    TierCostContext,
    TierCostEstimator,
    TierCostSample,
    TierSelection,
    quality_rank,
    residency_class,
)
from eligsir.mapping.scheduler.tier_diagnostics import (
    RasterTileDiagnostics,
    TierTransactionDiagnostic,
    TierTransactionDiagnostics,
)

if TYPE_CHECKING:
    from eligsir.runtime.config import DepthInsertionPolicy


@dataclass(frozen=True, slots=True)
class _ResolutionDecision:
    """The sole final resolution decision stamped onto one work plan."""

    tier: ResolutionTier
    action: str
    reason: str
    budget_ms: float | None
    probe: bool
    ceiling: ResolutionTier
    selection: TierSelection
    native_geometry_evidence_allowance_spent_ms: float = 0.0
    native_geometry_evidence_allowance_available_ms: float | None = None


class WorkBudgetActuator(str, Enum):
    """Proposal names understood by :class:`WorkBudgetOwner`."""

    RESOLUTION_TIER = "resolution_tier"
    REPLAY_RESOLUTION_TIER = "replay_resolution_tier"
    SUPERVISION_SLOTS = "supervision_slots"
    REPLAY_SLOTS = "replay_slots"
    OPTIMIZER_UPDATES = "optimizer_updates"
    INSERTION_ENABLED = "insertion_enabled"
    INSERTION_SEEDS = "insertion_seeds"
    REPLAY_INSERTION_SEEDS = "replay_insertion_seeds"
    ALLOW_GROWTH = "allow_growth"
    ALLOW_RELOCATION = "allow_relocation"
    MUTATION_BUDGET = "mutation_budget"
    LIVE_POPULATION_AUTHORIZATION = "live_population_authorization"
    REGIONAL_REPAIR_SUPERVISION_SLOTS = "regional_repair_supervision_slots"
    REGIONAL_REPAIR_REPLAY_SLOTS = "regional_repair_replay_slots"
    REGIONAL_REPAIR_OPTIMIZER_UPDATES = "regional_repair_optimizer_updates"


def _integer_request(
    proposals: tuple[ControlProposal, ...], actuator: WorkBudgetActuator, default: int
) -> tuple[int, bool]:
    values = tuple(
        proposal.value
        for proposal in proposals
        if proposal.actuator == actuator.value
        and isinstance(proposal.value, int)
        and not isinstance(proposal.value, bool)
        and proposal.value >= 0
    )
    return (min(values), True) if values else (default, False)


def _boolean_request(proposals: tuple[ControlProposal, ...], actuator: WorkBudgetActuator) -> bool:
    values = tuple(
        proposal.value
        for proposal in proposals
        if proposal.actuator == actuator.value and isinstance(proposal.value, bool)
    )
    return bool(values) and all(values)


def _resolution_request(
    proposals: tuple[ControlProposal, ...],
    actuator: WorkBudgetActuator = WorkBudgetActuator.RESOLUTION_TIER,
    *,
    default: ResolutionTier = ResolutionTier.NATIVE,
) -> ResolutionTier:
    values = tuple(
        proposal.value
        for proposal in proposals
        if proposal.actuator == actuator.value
        and isinstance(proposal.value, str)
        and proposal.value
    )
    value = values[-1] if values else default.value
    return ResolutionTier.NATIVE if value == "full" else ResolutionTier(value)


def _resolution_request_reason(
    proposals: tuple[ControlProposal, ...],
    actuator: WorkBudgetActuator,
    *,
    default: str,
) -> str:
    matching = tuple(
        proposal
        for proposal in proposals
        if proposal.actuator == actuator.value
        and isinstance(proposal.value, str)
        and proposal.value
    )
    if not matching:
        return default
    proposal = matching[-1]
    return f"{proposal.provider}: {proposal.reason}"


def _ordered(candidates: tuple[SourceIdentity, ...]) -> tuple[SourceIdentity, ...]:
    return tuple(sorted(candidates, key=SourceIdentity.ordering_key))


@dataclass(frozen=True, slots=True)
class ReplayCandidateSelection:
    """Reproducible weighted sample from the complete eligible replay cache."""

    sources: tuple[SourceIdentity, ...]
    weights: tuple[float, ...]
    population_size: int
    seed: int
    epoch: int


@dataclass(frozen=True, slots=True)
class _SupervisionTarget:
    """Barrier-frozen B/C target owned and actuated by WorkBudgetOwner."""

    requested_batch_size: int
    requested_camera_count: int
    cumulative_realized_supervised_views: int
    cumulative_optimizer_updates: int
    reason: str
    hysteresis_state: str
    latest_completed_transaction_id: int | None = None
    latest_completed_optimization_transaction_ns: int | None = None
    p95_optimization_transaction_ns: int | None = None
    max_optimization_transaction_ns: int | None = None


def _evidence_values(snapshot: ControlSnapshot) -> dict[str, object]:
    values = {fact.name: fact.value for fact in snapshot.evidence}
    if snapshot.latest_gpu_telemetry is not None:
        values.update(
            {fact.name: fact.value for fact in snapshot.latest_gpu_telemetry.facts}
        )
    return values


class _AdaptiveConstellationPolicy:
    """Shrink B/C on risk and grow bounded rungs after stable replay dwell.

    State transitions are keyed by ``control_tick_epoch``. Repeated snapshot
    resolutions within one canonical optimizer/control barrier therefore
    receive the exact same target and cannot change constellation mid-step.
    """

    def __init__(
        self,
        *,
        requested_batch_size: int,
        maximum_camera_count: int | None = None,
        grow_dwell: int = 3,
        latency_window_size: int = 16,
    ) -> None:
        self._requested_batch_size = validate_supervision_batch_size(
            requested_batch_size
        )
        self._adaptive_batch_sizes = tuple(
            size
            for size in BOUNDED_SUPERVISION_BATCH_SIZES
            if size != 3 or self._requested_batch_size == 3
        )
        if isinstance(grow_dwell, bool) or int(grow_dwell) < 1:
            raise ValueError("grow_dwell must be a positive integer")
        self._grow_dwell = int(grow_dwell)
        if isinstance(latency_window_size, bool) or int(latency_window_size) < 2:
            raise ValueError("latency_window_size must be an integer >= 2")
        self._optimization_latency_ns: deque[int] = deque(
            maxlen=int(latency_window_size)
        )
        self._last_optimization_transaction_id: int | None = None
        self._latest_optimization_transaction_ns: int | None = None
        qualified_for_batch = qualified_camera_count_for_batch(
            self._requested_batch_size
        )
        if maximum_camera_count is None:
            self._maximum_camera_count = qualified_for_batch
        else:
            if isinstance(maximum_camera_count, bool) or not isinstance(
                maximum_camera_count, int
            ):
                raise TypeError("maximum_camera_count must be an exact integer")
            configured_camera_count = validate_supervision_camera_count(
                maximum_camera_count
            )
            self._maximum_camera_count = min(
                configured_camera_count, qualified_for_batch
            )
        self._batch_size = self._requested_batch_size
        self._camera_count = self._maximum_camera_count
        self._stable_streak = 0
        self._last_tick_epoch: int | None = None
        self._last_decision: _SupervisionTarget | None = None

    @property
    def requested_batch_size(self) -> int:
        return self._requested_batch_size

    def _observe_completed_optimization_latency(
        self, snapshot: ControlSnapshot
    ) -> bool:
        telemetry = snapshot.latest_gpu_telemetry
        if telemetry is None:
            return False
        transaction_id = int(telemetry.transaction_id)
        if (
            self._last_optimization_transaction_id is not None
            and transaction_id <= self._last_optimization_transaction_id
        ):
            return False
        facts = {fact.name: fact.value for fact in telemetry.facts}
        duration_ns = facts.get("optimization_transaction_duration_ns")
        if duration_ns is None:
            return False
        if (
            isinstance(duration_ns, bool)
            or not isinstance(duration_ns, int)
            or duration_ns < 1
        ):
            raise TypeError(
                "completed optimization transaction duration must be a positive integer"
            )
        self._last_optimization_transaction_id = transaction_id
        self._latest_optimization_transaction_ns = duration_ns
        self._optimization_latency_ns.append(duration_ns)
        return True

    def _p95_optimization_transaction_ns(self) -> int | None:
        if not self._optimization_latency_ns:
            return None
        ordered = sorted(self._optimization_latency_ns)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]

    def _latency_fields(self, snapshot: ControlSnapshot) -> dict[str, int | None]:
        return {
            "latest_completed_transaction_id": self._last_optimization_transaction_id,
            "latest_completed_optimization_transaction_ns": (
                self._latest_optimization_transaction_ns
            ),
            "p95_optimization_transaction_ns": (
                self._p95_optimization_transaction_ns()
            ),
            "max_optimization_transaction_ns": (
                snapshot.limits.max_optimization_transaction_ns
            ),
        }

    def _shrink_one_latency_rung(self) -> None:
        batch_index = self._adaptive_batch_sizes.index(self._batch_size)
        self._batch_size = self._adaptive_batch_sizes[max(0, batch_index - 1)]
        self._batch_size = min(self._batch_size, self._requested_batch_size)
        camera_index = QUALIFIED_SUPERVISION_CAMERA_COUNTS.index(self._camera_count)
        self._camera_count = QUALIFIED_SUPERVISION_CAMERA_COUNTS[
            max(0, camera_index - 1)
        ]
        self._camera_count = min(
            self._camera_count, self._maximum_camera_count, self._batch_size
        )

    def _risk_reason(
        self, snapshot: ControlSnapshot, *, stable_replay: bool
    ) -> str | None:
        if snapshot.freshness_mode is FreshnessMode.CRITICAL:
            return "critical_backlog_pressure"
        if snapshot.freshness_mode is FreshnessMode.CATCH_UP:
            return "sustained_input_backlog_pressure"
        evidence = _evidence_values(snapshot)
        for name in ("partition_fade", "quality_risk"):
            if evidence.get(name) is True:
                return f"{name}_evidence"
        if stable_replay:
            return None
        if evidence.get("new_region") is True:
            return "new_region_evidence"
        coverage_gain = evidence.get("coverage_gain")
        if isinstance(coverage_gain, (int, float)) and not isinstance(
            coverage_gain, bool
        ) and float(coverage_gain) > 0.0:
            return "new_region_coverage_gain"
        novelty = evidence.get("novelty")
        if isinstance(novelty, (int, float)) and not isinstance(novelty, bool) and float(
            novelty
        ) >= 0.5:
            return "new_region_novelty"
        return None

    def decide(
        self,
        snapshot: ControlSnapshot,
        *,
        stable_replay: bool,
    ) -> _SupervisionTarget:
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        if type(stable_replay) is not bool:
            raise TypeError("stable_replay must be bool")
        tick_epoch = int(snapshot.control_tick_epoch)
        if self._last_tick_epoch is not None and tick_epoch < self._last_tick_epoch:
            raise ValueError("control_tick_epoch must be monotonic")

        new_latency = self._observe_completed_optimization_latency(snapshot)
        latency_budget = snapshot.limits.max_optimization_transaction_ns
        latency_p95 = self._p95_optimization_transaction_ns()
        latency_over = bool(
            latency_budget is not None
            and new_latency
            and (
                int(self._latest_optimization_transaction_ns or 0) > latency_budget
                or int(latency_p95 or 0) > latency_budget
            )
        )
        latency_window_over = bool(
            latency_budget is not None
            and latency_p95 is not None
            and latency_p95 > latency_budget
        )
        latency_headroom = bool(
            latency_budget is not None
            and latency_p95 is not None
            and self._latest_optimization_transaction_ns is not None
            and latency_p95 * 5 <= latency_budget * 4
            and self._latest_optimization_transaction_ns * 5 <= latency_budget * 4
        )
        risk_reason = self._risk_reason(snapshot, stable_replay=stable_replay)
        if (
            risk_reason is None
            and latency_budget is not None
            and self._last_decision is None
            and latency_p95 is None
        ):
            self._batch_size = 1
            self._camera_count = 1
            self._stable_streak = 0
            risk_reason = "latency_unknown_conservative_start"
        elif risk_reason is None and latency_over:
            self._shrink_one_latency_rung()
            self._stable_streak = 0
            risk_reason = "optimization_latency_over_budget"
        if self._last_tick_epoch == tick_epoch:
            assert self._last_decision is not None
            if risk_reason is None:
                return self._last_decision
            risk_batch_size = (
                min(2, self._requested_batch_size)
                if risk_reason in {
                    "critical_backlog_pressure",
                    "sustained_input_backlog_pressure",
                }
                else 1
            )
            if (
                risk_batch_size >= self._batch_size
                and self._camera_count <= 1
                and self._last_decision.hysteresis_state == "risk_shrunk"
            ):
                return self._last_decision
            self._batch_size = min(self._batch_size, risk_batch_size)
            self._camera_count = 1
            self._stable_streak = 0
            decision = _SupervisionTarget(
                requested_batch_size=self._batch_size,
                requested_camera_count=self._camera_count,
                cumulative_realized_supervised_views=(
                    snapshot.supervision_views_completed
                ),
                cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
                reason=risk_reason,
                hysteresis_state="risk_shrunk",
            )
            self._last_decision = decision
            return decision
        if risk_reason in {
            "latency_unknown_conservative_start",
            "optimization_latency_over_budget",
        }:
            reason = risk_reason
            hysteresis_state = "risk_shrunk"
        elif risk_reason is not None:
            self._batch_size = (
                min(2, self._requested_batch_size)
                if risk_reason in {
                    "critical_backlog_pressure",
                    "sustained_input_backlog_pressure",
                }
                else 1
            )
            self._camera_count = 1
            self._stable_streak = 0
            reason = risk_reason
            hysteresis_state = "risk_shrunk"
        elif latency_window_over:
            self._stable_streak = 0
            reason = "optimization_latency_window_over_budget"
            hysteresis_state = "latency_hold"
        elif latency_budget is not None and not latency_headroom:
            self._stable_streak = 0
            reason = "optimization_latency_headroom_hold"
            hysteresis_state = "latency_hold"
        elif (stable_replay or (latency_budget is not None and latency_headroom)) and (
            self._batch_size < self._requested_batch_size
            or self._camera_count < self._maximum_camera_count
        ):
            self._stable_streak += 1
            if self._stable_streak >= self._grow_dwell:
                batch_index = self._adaptive_batch_sizes.index(
                    self._batch_size
                )
                self._batch_size = min(
                    self._requested_batch_size,
                    self._adaptive_batch_sizes[batch_index + 1],
                )
                if self._camera_count < min(
                    self._maximum_camera_count, self._batch_size
                ):
                    camera_index = QUALIFIED_SUPERVISION_CAMERA_COUNTS.index(
                        self._camera_count
                    )
                    self._camera_count = min(
                        self._maximum_camera_count,
                        self._batch_size,
                        QUALIFIED_SUPERVISION_CAMERA_COUNTS[camera_index + 1],
                    )
                self._stable_streak = 0
                reason = (
                    "stable_replay_slow_grow"
                    if stable_replay
                    else "latency_headroom_slow_grow"
                )
                hysteresis_state = (
                    f"grown_to_B{self._batch_size}_C{self._camera_count}"
                )
            else:
                reason = (
                    "stable_replay_grow_dwell"
                    if stable_replay
                    else "latency_headroom_grow_dwell"
                )
                hysteresis_state = (
                    f"stable_grow_{self._stable_streak}/{self._grow_dwell}"
                )
        else:
            self._stable_streak = 0
            reason = "configured_qualified_target"
            hysteresis_state = "stable_hold"

        decision = _SupervisionTarget(
            requested_batch_size=self._batch_size,
            requested_camera_count=self._camera_count,
            cumulative_realized_supervised_views=(
                snapshot.supervision_views_completed
            ),
            cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
            reason=reason,
            hysteresis_state=hysteresis_state,
            **self._latency_fields(snapshot),
        )
        self._last_tick_epoch = tick_epoch
        self._last_decision = decision
        return decision

    def post_refinement_target(self, snapshot: ControlSnapshot) -> _SupervisionTarget:
        """Resolve the finite-source replay phase outside live hysteresis."""

        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        return _SupervisionTarget(
            requested_batch_size=self._requested_batch_size,
            requested_camera_count=self._maximum_camera_count,
            cumulative_realized_supervised_views=snapshot.supervision_views_completed,
            cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
            reason="post_refinement_phase_resolved",
            hysteresis_state="phase_resolved",
        )


class _FixedConstellationPolicy(_AdaptiveConstellationPolicy):
    """Freeze the requested B/C for controlled execution ablations.

    Candidate scarcity may still underfill a transaction; latency, freshness,
    and quality evidence may not change the requested constellation.  This is
    intentionally not the live production default.
    """

    def decide(
        self,
        snapshot: ControlSnapshot,
        *,
        stable_replay: bool,
    ) -> _SupervisionTarget:
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        if type(stable_replay) is not bool:
            raise TypeError("stable_replay must be bool")
        tick_epoch = int(snapshot.control_tick_epoch)
        if self._last_tick_epoch is not None and tick_epoch < self._last_tick_epoch:
            raise ValueError("control_tick_epoch must be monotonic")
        self._observe_completed_optimization_latency(snapshot)
        if self._last_tick_epoch == tick_epoch:
            assert self._last_decision is not None
            return self._last_decision
        decision = _SupervisionTarget(
            requested_batch_size=self._requested_batch_size,
            requested_camera_count=self._maximum_camera_count,
            cumulative_realized_supervised_views=(
                snapshot.supervision_views_completed
            ),
            cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
            reason="fixed_constellation_ablation",
            hysteresis_state="fixed",
            **self._latency_fields(snapshot),
        )
        self._last_tick_epoch = tick_epoch
        self._last_decision = decision
        return decision


def _supervision_constellation(
    *,
    decision: _SupervisionTarget,
    realized_batch_size: int,
    unique_eligible_count: int,
) -> SupervisionConstellation:
    realized_c = (
        0
        if realized_batch_size == 0
        else min(decision.requested_camera_count, unique_eligible_count)
    )
    microbatch_count = (
        0 if realized_batch_size == 0 else math.ceil(realized_batch_size / realized_c)
    )
    if realized_batch_size == 0:
        mode = "empty"
    elif realized_c == realized_batch_size == 1:
        mode = "single_camera_immediate"
    elif realized_c == 1:
        mode = "sequential_accumulation"
    elif microbatch_count > 1:
        mode = "true_camera_microbatch_accumulation"
    else:
        mode = "true_camera_batch"
    reason = decision.reason
    if (
        realized_batch_size < decision.requested_batch_size
        and reason == "configured_qualified_target"
    ):
        reason = "eligible_underfill"
    return SupervisionConstellation(
        requested_batch_size=decision.requested_batch_size,
        requested_camera_count=decision.requested_camera_count,
        realized_batch_size=realized_batch_size,
        realized_camera_count=realized_c,
        unique_eligible_count=unique_eligible_count,
        fill_ratio=realized_batch_size / decision.requested_batch_size,
        microbatch_count=microbatch_count,
        mode=mode,
        reason=reason,
        hysteresis_state=decision.hysteresis_state,
        cumulative_realized_supervised_views=(
            decision.cumulative_realized_supervised_views
        ),
        cumulative_optimizer_updates=decision.cumulative_optimizer_updates,
        latest_completed_transaction_id=decision.latest_completed_transaction_id,
        latest_completed_optimization_transaction_ns=(
            decision.latest_completed_optimization_transaction_ns
        ),
        p95_optimization_transaction_ns=decision.p95_optimization_transaction_ns,
        max_optimization_transaction_ns=decision.max_optimization_transaction_ns,
    )


class WeightedReplayCandidateProvider:
    """Sample without replacement while preserving a non-zero global floor.

    The provider ranks every eligible identity with a deterministic random key.
    A controller may raise individual weights to focus replay on struggling
    areas; ``global_weight_floor`` ensures the rest of the trajectory remains
    reachable and cannot be terminally starved.
    """

    def __init__(
        self, *, random_seed: int = 1, global_weight_floor: float = 0.05
    ) -> None:
        if isinstance(random_seed, bool) or int(random_seed) < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if (
            isinstance(global_weight_floor, bool)
            or not math.isfinite(float(global_weight_floor))
            or float(global_weight_floor) <= 0.0
        ):
            raise ValueError("global_weight_floor must be finite and positive")
        self._random_seed = int(random_seed)
        self._global_weight_floor = float(global_weight_floor)

    def sample(
        self,
        candidates: tuple[SourceIdentity, ...],
        weights: tuple[float, ...],
        *,
        slots: int,
        epoch: int,
        snapshot_id: int,
        reserved_candidate: SourceIdentity | None = None,
        reserved_candidates: tuple[SourceIdentity, ...] = (),
    ) -> ReplayCandidateSelection:
        """Return weighted-sampled replay candidates without replacement."""
        candidates = tuple(candidates)
        weights = tuple(float(weight) for weight in weights)
        if len(weights) != len(candidates):
            raise ValueError("replay weights must align with replay candidates")
        if isinstance(slots, bool) or int(slots) < 0:
            raise ValueError("slots must be a non-negative integer")
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        if isinstance(snapshot_id, bool) or int(snapshot_id) < 0:
            raise ValueError("snapshot_id must be a non-negative integer")
        if len(set(candidates)) != len(candidates):
            raise ValueError("replay candidates must be unique")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("replay weights must be finite and non-negative")
        if reserved_candidate is not None:
            if not isinstance(reserved_candidate, SourceIdentity):
                raise TypeError("reserved_candidate must be SourceIdentity or None")
            if reserved_candidate not in candidates:
                raise ValueError("reserved_candidate must be replay eligible")
        if not isinstance(reserved_candidates, tuple) or not all(
            isinstance(source, SourceIdentity) for source in reserved_candidates
        ):
            raise TypeError("reserved_candidates must be a tuple of SourceIdentity")
        if any(source not in candidates for source in reserved_candidates):
            raise ValueError("reserved_candidates must be replay eligible")

        ordered = sorted(
            zip(candidates, weights, strict=True),
            key=lambda item: item[0].ordering_key(),
        )
        population_size = len(ordered)
        requested = min(int(slots), population_size)
        seed_payload = f"{self._random_seed}:{int(snapshot_id)}:{int(epoch)}".encode()
        seed = int.from_bytes(hashlib.blake2b(seed_payload, digest_size=8).digest(), "big")
        if requested == 0:
            return ReplayCandidateSelection((), (), population_size, seed, int(epoch))

        ranked = []
        reserved_selection: list[SourceIdentity] = []
        if reserved_candidate is not None:
            reserved_selection.append(reserved_candidate)
        for source in reserved_candidates:
            if source not in reserved_selection and len(reserved_selection) < requested:
                reserved_selection.append(source)
        reserved_weights = {
            source: max(self._global_weight_floor, weight)
            for source, weight in ordered
            if source in reserved_selection
        }
        for source, requested_weight in ordered:
            effective_weight = max(self._global_weight_floor, requested_weight)
            if source in reserved_selection:
                continue
            digest = hashlib.blake2b(
                seed.to_bytes(8, "big") + source.fingerprint().encode(),
                digest_size=8,
            ).digest()
            # Efraimidis-Spirakis weighted sampling without replacement.  The
            # open unit interval avoids log(0), while per-identity hashing
            # makes results independent of dictionary/list traversal order.
            unit = (int.from_bytes(digest, "big") + 1) / ((1 << 64) + 1)
            priority = -math.log(unit) / effective_weight
            ranked.append((priority, source.ordering_key(), source, effective_weight))
        ranked.sort(key=lambda item: (item[0], item[1]))
        selected = ranked[: requested - len(reserved_selection)]
        return ReplayCandidateSelection(
            sources=(*reserved_selection, *(item[2] for item in selected)),
            weights=(
                *(reserved_weights[source] for source in reserved_selection),
                *(item[3] for item in selected),
            ),
            population_size=population_size,
            seed=seed,
            epoch=int(epoch),
        )


class WorkBudgetOwner:
    """Resolve every mapping-work actuator from one immutable snapshot.

    Existing controllers remain evidence/proposal providers. This owner is
    the sole actuator and owns the small barrier-keyed B/C hysteresis state.
    """

    def __init__(
        self,
        *,
        depth_insertion: "DepthInsertionPolicy | None" = None,
        sample_stride: int = 1,
        replay_candidate_provider: WeightedReplayCandidateProvider | None = None,
        replay_insertion_voxel_size_m: float | None = None,
        requested_batch_size: int | None = None,
        maximum_camera_count: int | None = None,
        constellation_policy: str = "adaptive",
        global_history_supervision_cadence: int = 8,
        target_supervised_views_per_s: float = 100.0,
        supported_resolution_tiers: tuple[ResolutionTier, ...] = LIVE_TIERS,
        tier_cost_estimator: TierCostEstimator | None = None,
        device_id: str = UNKNOWN_DEVICE,
        native_geometry_evidence_allowance_ms: float = 480.0,
        native_geometry_evidence_allowance_refill_ms: float = 40.0,
        refinement_resolution_policy: str = "native",
        refinement_progressive_switch_frac: float = 0.5,
        budget_relax_max: float = 4.0,
        budget_relax_step: float = 1.25,
        relax_source_rate_floor: float = 0.9,
    ) -> None:
        if depth_insertion is None:
            from eligsir.runtime.config import DepthInsertionPolicy

            depth_insertion = DepthInsertionPolicy()
        self._depth_insertion = depth_insertion
        if isinstance(sample_stride, bool) or int(sample_stride) < 1:
            raise ValueError("sample_stride must be a positive integer")
        self._sample_stride = int(sample_stride)
        if replay_insertion_voxel_size_m is None:
            replay_insertion_voxel_size_m = depth_insertion.voxel_size_m
        if (
            isinstance(replay_insertion_voxel_size_m, bool)
            or not math.isfinite(float(replay_insertion_voxel_size_m))
            or float(replay_insertion_voxel_size_m) <= 0.0
        ):
            raise ValueError(
                "replay_insertion_voxel_size_m must be finite and positive"
            )
        self._replay_insertion_voxel_size_m = float(
            replay_insertion_voxel_size_m
        )
        self._replay_candidate_provider = (
            replay_candidate_provider or WeightedReplayCandidateProvider()
        )
        if (
            isinstance(global_history_supervision_cadence, bool)
            or int(global_history_supervision_cadence) < 1
        ):
            raise ValueError(
                "global_history_supervision_cadence must be a positive integer"
            )
        self._global_history_supervision_cadence = int(
            global_history_supervision_cadence
        )
        self._last_reserved_global_bucket = -1
        self._cached_reservation_snapshot_id: int | None = None
        self._cached_reservation_source: SourceIdentity | None = None
        self._adaptive_supervision_enabled = requested_batch_size is not None
        self._requested_batch_size = validate_supervision_batch_size(
            1 if requested_batch_size is None else requested_batch_size
        )
        if not isinstance(constellation_policy, str):
            raise TypeError("constellation_policy must be a string")
        if constellation_policy not in {"adaptive", "fixed"}:
            raise ValueError("constellation_policy must be adaptive or fixed")
        policy_type = (
            _AdaptiveConstellationPolicy
            if constellation_policy == "adaptive"
            else _FixedConstellationPolicy
        )
        self._supervision_policy = policy_type(
            requested_batch_size=self._requested_batch_size,
            maximum_camera_count=maximum_camera_count,
        )
        if (
            isinstance(target_supervised_views_per_s, bool)
            or not math.isfinite(float(target_supervised_views_per_s))
            or float(target_supervised_views_per_s) <= 0.0
        ):
            raise ValueError(
                "target_supervised_views_per_s must be finite and positive"
            )
        self._target_supervised_views_per_s = float(
            target_supervised_views_per_s
        )
        # The sole final resolution authority.  Policy providers contribute
        # evidence (a proposed quality ceiling); only this estimator's
        # selection is stamped into executable work.
        if not isinstance(device_id, str) or not device_id:
            raise ValueError("device_id must be a non-empty string")
        # Cost calibration is hardware-local: a measurement taken on one
        # accelerator must never reach another one's scheduling decision.
        self._device_id = device_id
        self._tier_costs = tier_cost_estimator or TierCostEstimator(
            legal_tiers=supported_resolution_tiers
        )
        self._last_transaction_resolution_tier: ResolutionTier | None = None
        # Closed-loop budget relaxation (2026-09-04, tier-budget floor-lock).
        # The per-transaction budget is open-loop: 1000*C/target_views_per_s.
        # When even the cheapest *measured* tier costs more than that, no
        # tier is ever feasible, the probe path can never fire (it needs
        # headroom under the same budget), and the selector holds the
        # cheapest tier forever regardless of whether the pipeline is in
        # fact keeping up (r9700 fr3: 10ms budget vs ~19ms quarter -> 93% of
        # transactions floor-locked at quarter). Freshness is the real
        # closed-loop signal: while it reads HEALTHY the source is being kept
        # up with, so the strict budget is demonstrably pessimistic. In that
        # exact regime -- budget infeasible AND freshness HEALTHY -- the
        # effective budget is relaxed geometrically (x`step` per decision,
        # capped at x`max`); any non-HEALTHY reading resets it to 1.0, so a
        # relaxed tier that starts costing freshness is dropped at once.
        # `budget_relax_max <= 1.0` disables the mechanism entirely.
        for name, value in (
            ("budget_relax_max", budget_relax_max),
            ("budget_relax_step", budget_relax_step),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 1.0
            ):
                raise ValueError(f"{name} must be finite and >= 1.0")
        if (
            isinstance(relax_source_rate_floor, bool)
            or not math.isfinite(float(relax_source_rate_floor))
            or float(relax_source_rate_floor) < 0.0
            or float(relax_source_rate_floor) > 1.0
        ):
            raise ValueError(
                "relax_source_rate_floor must be finite and in [0.0, 1.0]"
            )
        self._budget_relax_max = float(budget_relax_max)
        self._budget_relax_step = float(budget_relax_step)
        self._relax_source_rate_floor = float(relax_source_rate_floor)
        self._budget_relax_factor = 1.0
        if (
            isinstance(native_geometry_evidence_allowance_ms, bool)
            or not math.isfinite(float(native_geometry_evidence_allowance_ms))
            or float(native_geometry_evidence_allowance_ms) < 0.0
        ):
            raise ValueError(
                "native_geometry_evidence_allowance_ms must be finite and non-negative"
            )
        if (
            isinstance(native_geometry_evidence_allowance_refill_ms, bool)
            or not math.isfinite(float(native_geometry_evidence_allowance_refill_ms))
            or float(native_geometry_evidence_allowance_refill_ms) < 0.0
        ):
            raise ValueError(
                "native_geometry_evidence_allowance_refill_ms must be finite and non-negative"
            )
        # A bounded, scheduler-owned token bucket reserves a small amount of
        # additional measured latency for an E0 geometry target. It begins
        # full so an eligible demand can make progress immediately; C4's
        # normal 40ms control budget refills it once per control epoch.
        self._native_geometry_evidence_allowance_capacity_ms = float(
            native_geometry_evidence_allowance_ms
        )
        self._native_geometry_evidence_allowance_refill_ms = float(
            native_geometry_evidence_allowance_refill_ms
        )
        self._native_geometry_evidence_allowance_available_ms = float(
            native_geometry_evidence_allowance_ms
        )
        self._native_geometry_evidence_allowance_last_epoch: int | None = None
        # Explanatory only -- see scheduler/tier_diagnostics.py. Never read by
        # any selection path in this class.
        self._tier_diagnostics = TierTransactionDiagnostics()
        if not isinstance(refinement_resolution_policy, str):
            raise TypeError("refinement_resolution_policy must be a string")
        if refinement_resolution_policy not in {"native", "bounded", "progressive"}:
            raise ValueError(
                "refinement_resolution_policy must be native, bounded, or progressive"
            )
        # issue #1346: "native" preserves the historical, deliberate
        # "post-refinement has no realtime constraint" behavior below.
        # "bounded" keeps the ordinary freshness/throughput ladder engaged
        # through refinement instead, so a short refinement tail keeps
        # sustaining target_supervised_views_per_s rather than falling off
        # the measured ~2.9x native-tier cliff. See
        # docs/architecture/2026-08-31-issue1346-live-path-refinement-cliff.md.
        # "progressive" (follow-up to #1346): spends the early portion of
        # refinement (progress fraction below
        # ``refinement_progressive_switch_frac``) behaving exactly like
        # "bounded" -- cheap-tier, high-throughput steps while the
        # population still has the most to gain per optimizer step -- then
        # switches to "native"'s forced-native/unbounded behavior for the
        # remainder, spending the final stretch of a short refinement tail
        # on full-quality supervision once cheap gains plateau. The switch
        # is driven by ``refinement_progress_fraction`` threaded in at each
        # ``_resolve_transaction_resolution``/``_transaction_budget_ms`` call
        # (see ``run_causal_post_mapping_refinement``'s
        # ``completed / requested`` accounting) -- never by wall-clock time
        # directly, so it is deterministic given a fixed step budget.
        self._refinement_resolution_policy = refinement_resolution_policy
        if (
            isinstance(refinement_progressive_switch_frac, bool)
            or not math.isfinite(float(refinement_progressive_switch_frac))
            or not (0.0 <= float(refinement_progressive_switch_frac) <= 1.0)
        ):
            raise ValueError(
                "refinement_progressive_switch_frac must be a finite number in [0, 1]"
            )
        self._refinement_progressive_switch_frac = float(
            refinement_progressive_switch_frac
        )

    @property
    def refinement_resolution_policy(self) -> str:
        """Return the refinement resolution policy."""
        return self._refinement_resolution_policy

    @property
    def refinement_progressive_switch_frac(self) -> float:
        """Return the progressive refinement switch fraction."""
        return self._refinement_progressive_switch_frac

    def forces_native_post_refinement(
        self, refinement_progress_fraction: float | None = None
    ) -> bool:
        """Public wrapper of :meth:`_post_refinement_forces_native`.

        Lets a caller (e.g. ``production_application.py``'s idle-replay-plan
        seam) decide whether to bother consulting
        ``ResolutionPolicyProvider`` at all for a post-refinement transaction
        *before* minting a plan -- its proposal would be discarded anyway
        once this returns True, mirroring the "native" skip-the-provider
        optimization that already existed for that case.
        """

        return self._post_refinement_forces_native(refinement_progress_fraction)

    def _post_refinement_forces_native(
        self, refinement_progress_fraction: float | None
    ) -> bool:
        """True iff post-refinement transactions must be forced to NATIVE.

        ``"native"`` always forces it. ``"progressive"`` forces it only once
        ``refinement_progress_fraction`` reaches the configured switch
        point; an unknown (``None``) fraction is treated as "not yet
        switched" -- the safer, bounded-like default, since the caller not
        supplying progress evidence is a reason to fall through to the
        measured-cost path, not to silently assume the run is nearly done.
        ``"bounded"`` never forces it.
        """

        if self._refinement_resolution_policy == "native":
            return True
        if self._refinement_resolution_policy == "progressive":
            return (
                refinement_progress_fraction is not None
                and float(refinement_progress_fraction)
                >= self._refinement_progressive_switch_frac
            )
        return False

    @property
    def requested_batch_size(self) -> int:
        """Return the requested batch size."""
        return self._requested_batch_size

    @property
    def tier_costs(self) -> TierCostEstimator:
        """Bounded measured per-tier transaction cost evidence."""

        return self._tier_costs

    @property
    def tier_diagnostics(self) -> TierTransactionDiagnostics:
        """Bounded EXPLANATORY per-transaction records.

        Read by reporting only.  Nothing in this class's selection path
        consults it, and nothing should ever start to -- see the module
        docstring of ``scheduler/tier_diagnostics.py``.
        """

        return self._tier_diagnostics

    @property
    def native_geometry_evidence_allowance_available_ms(self) -> float:
        """Currently unspent bounded native-evidence latency allowance."""

        return self._native_geometry_evidence_allowance_available_ms

    def _refill_native_geometry_evidence_allowance(self, *, epoch: int) -> None:
        """Refill at most once per monotonic scheduler epoch."""

        epoch = int(epoch)
        previous_epoch = self._native_geometry_evidence_allowance_last_epoch
        if previous_epoch is None:
            self._native_geometry_evidence_allowance_last_epoch = epoch
            return
        if epoch <= previous_epoch:
            return
        self._native_geometry_evidence_allowance_available_ms = min(
            self._native_geometry_evidence_allowance_capacity_ms,
            self._native_geometry_evidence_allowance_available_ms
            + (epoch - previous_epoch)
            * self._native_geometry_evidence_allowance_refill_ms,
        )
        self._native_geometry_evidence_allowance_last_epoch = epoch

    def _cost_context(
        self,
        *,
        gaussian_count: int,
        camera_count: int,
        pass_mode: str = FORWARD_BACKWARD,
    ) -> TierCostContext:
        """Identity of the calibration bucket one transaction belongs to.

        Cost evidence is never shared across devices, Gaussian-count regimes,
        render-concurrency regimes, render modes or pass modes: the fixed
        component of transaction cost belongs to those, not to the tier.
        The render mode is *derived* here rather than supplied, so a recorded
        measurement and a later selection can never land in different buckets
        by disagreeing about how the batch was executed.
        """

        camera_count = max(1, int(camera_count))
        return TierCostContext(
            device_id=self._device_id,
            gaussian_count=max(0, int(gaussian_count)),
            camera_count=camera_count,
            render_mode=(
                "true_camera_batch"
                if camera_count > 1
                else "sequential_accumulation"
            ),
            pass_mode=pass_mode,
        )

    def record_completed_transaction(
        self,
        *,
        transaction_class: MappingTransactionClass | None,
        tier: ResolutionTier,
        gaussian_count: int,
        realized_camera_count: int,
        prepare_ms: float,
        optimization_ms: float,
        transaction_wall_ms: float,
        kernel_ms: float | None = None,
        cache_hits: int = 0,
        cache_misses: int = 0,
        pass_mode: str = FORWARD_BACKWARD,
        raster_diagnostics: RasterTileDiagnostics | None = None,
    ) -> TierCostSample:
        """Fold one executed transaction's measured cost back into evidence.

        This is the only cost-feedback seam.  Completed wall-clock cost never
        reaches a plan-minting call as an ad-hoc argument: it updates bounded
        estimator state here, and the selection path reads only that state.
        """

        # kernel_ms is telemetry only -- selection reads transaction_wall_ms --
        # and the ROCm GPU-event path is already known to produce occasional
        # non-finite durations.  Drop a bad reading rather than refusing the
        # whole (otherwise valid) measurement from inside the training tick.
        if kernel_ms is not None:
            kernel_ms = float(kernel_ms)
            if not math.isfinite(kernel_ms) or kernel_ms < 0.0:
                kernel_ms = None
        sample = TierCostSample(
            context=self._cost_context(
                gaussian_count=gaussian_count,
                camera_count=realized_camera_count,
                pass_mode=pass_mode,
            ),
            tier=tier,
            transaction_class=transaction_class,
            prepare_ms=float(prepare_ms),
            optimization_ms=float(optimization_ms),
            transaction_wall_ms=float(transaction_wall_ms),
            kernel_ms=None if kernel_ms is None else float(kernel_ms),
            cache_hits=int(cache_hits),
            cache_misses=int(cache_misses),
        )
        self._tier_costs.record(sample)
        self._tier_diagnostics.record(
            TierTransactionDiagnostic(
                resolution_tier=str(tier.value),
                gaussian_count=max(0, int(gaussian_count)),
                camera_count=max(1, int(realized_camera_count)),
                transaction_ms=float(sample.transaction_wall_ms),
                prepare_ms=float(sample.prepare_ms),
                residency=residency_class(sample),
                **(
                    {}
                    if raster_diagnostics is None
                    else {
                        "tile_count": raster_diagnostics.tile_count,
                        "visible_nnz": raster_diagnostics.visible_nnz,
                        "intersection_count": raster_diagnostics.intersection_count,
                        "isects_per_tile": raster_diagnostics.isects_per_tile,
                        "tile_list_p50": raster_diagnostics.tile_list_p50,
                        "tile_list_p95": raster_diagnostics.tile_list_p95,
                        "tile_list_max": raster_diagnostics.tile_list_max,
                        "tile_imbalance_p95_p50": (
                            raster_diagnostics.tile_imbalance_p95_p50
                        ),
                    }
                ),
            )
        )
        return sample

    def tier_prepare_saving_ms(
        self,
        tier_value: str,
        *,
        gaussian_count: int,
        camera_count: int,
        pass_mode: str = FORWARD_BACKWARD,
    ) -> float | None:
        """Measured preparation time a HOT hit on ``tier_value`` avoids.

        This is the ``T_prepare_saved`` term ``HotTierCache`` prices retention
        with.  It reads the SAME calibration bucket the allocator keys on, so a
        CUDA measurement can no more reach a ROCm retention decision than it can
        reach a ROCm selection decision.  Returns ``None`` when this bucket has
        no evidence yet.
        """

        try:
            tier = ResolutionTier(str(tier_value))
        except ValueError:
            return None
        return self._tier_costs.prepare_saving_ms(
            tier,
            context=self._cost_context(
                gaussian_count=gaussian_count,
                camera_count=camera_count,
                pass_mode=pass_mode,
            ),
        )

    def _transaction_budget_ms(
        self,
        *,
        transaction_class: MappingTransactionClass | None,
        realized_camera_count: int,
        refinement_progress_fraction: float | None = None,
    ) -> float | None:
        """Latency budget for one transaction, or None when unconstrained."""

        if (
            transaction_class is MappingTransactionClass.POST_REFINEMENT
            and self._post_refinement_forces_native(refinement_progress_fraction)
        ):
            return None  # post-refinement carries no realtime constraint
        if int(realized_camera_count) <= 0:
            return None
        return (
            1000.0
            * int(realized_camera_count)
            / self._target_supervised_views_per_s
        )

    def _resolve_transaction_resolution(
        self,
        *,
        snapshot: ControlSnapshot,
        transaction_class: MappingTransactionClass | None,
        proposals: tuple[ControlProposal, ...],
        realized_camera_count: int,
        default_tier: ResolutionTier,
        refinement_progress_fraction: float | None = None,
        native_geometry_evidence_purpose: bool = False,
        source_depth_requires_native: bool = False,
    ) -> _ResolutionDecision:
        """Select the one final tier for a whole planned transaction.

        Evidence in, one decision out.  Freshness/pressure policy supplies a
        quality *ceiling*; measured per-tier transaction cost decides which
        tier at or below that ceiling is actually usable.  Ordinal tier
        position is never used as a proxy for cost anywhere in this path.
        """

        post_refinement = transaction_class is MappingTransactionClass.POST_REFINEMENT
        replay_only = transaction_class in {
            MappingTransactionClass.REPLAY_ONLY,
            MappingTransactionClass.POST_REFINEMENT,
        }
        actuator = (
            WorkBudgetActuator.REPLAY_RESOLUTION_TIER
            if replay_only
            else WorkBudgetActuator.RESOLUTION_TIER
        )
        ceiling = _resolution_request(proposals, actuator, default=default_tier)
        provider_reason = _resolution_request_reason(
            proposals,
            actuator,
            default="no resolution proposal; active tier held",
        )
        budget_ms = self._transaction_budget_ms(
            transaction_class=transaction_class,
            realized_camera_count=realized_camera_count,
            refinement_progress_fraction=refinement_progress_fraction,
        )
        # Required source geometry is not discretionary B4 population evidence.
        # Keep the throughput target visible, but never spend the E0 allowance
        # or execute low-tier RGB that would consume this source's obligation.
        native_geometry_evidence_purpose = (
            native_geometry_evidence_purpose and not source_depth_requires_native
        )
        allowance_spent_ms = 0.0
        allowance_available_ms = (
            self._native_geometry_evidence_allowance_available_ms
            if source_depth_requires_native else None
        )
        native_geometry_calibration_probe = False
        valid_native = False
        context = self._cost_context(
            gaussian_count=int(snapshot.gaussian_count),
            camera_count=max(1, int(realized_camera_count)),
        )
        selection_ceiling = ceiling
        selection_budget_ms = budget_ms
        if native_geometry_evidence_purpose and budget_ms is not None:
            # Policy provides only a purpose signal. The scheduler may expand
            # this plan's ceiling only after recent native cost evidence fits
            # within its bounded rolling allowance.
            self._refill_native_geometry_evidence_allowance(
                epoch=snapshot.control_tick_epoch
            )
            allowance_available_ms = (
                self._native_geometry_evidence_allowance_available_ms
            )
            valid_native = ResolutionTier.NATIVE in self._tier_costs.select(
                ceiling=ResolutionTier.NATIVE,
                budget_ms=budget_ms,
                context=context,
            ).measured_tiers
            native_state = (
                self._tier_costs.state(ResolutionTier.NATIVE, context=context)
                if valid_native
                else None
            )
            if native_state is not None:
                required_ms = max(0.0, native_state.estimate_ms - budget_ms)
                if required_ms <= allowance_available_ms:
                    selection_ceiling = ResolutionTier.NATIVE
                    selection_budget_ms = budget_ms + required_ms
                    provider_reason = (
                        f"{provider_reason}; native geometry evidence purpose "
                        f"with {required_ms:.3f}ms scheduler allowance"
                    )
        if source_depth_requires_native:
            if ResolutionTier.NATIVE not in self._tier_costs.legal_tiers:
                raise ValueError("READY source-depth insertion requires a legal native tier")
            selection_ceiling = ResolutionTier.NATIVE
            selection = TierSelection(
                tier=ResolutionTier.NATIVE,
                reason="required native source-depth evidence; throughput target is soft",
                budget_ms=budget_ms,
            )
        elif post_refinement and self._post_refinement_forces_native(
            refinement_progress_fraction
        ):
            # Post-refinement has no realtime constraint and is reported as a
            # separate transaction class; supervision quality is maximal.
            # issue #1346: this is a deliberate, historical default -- see
            # ``refinement_resolution_policy`` -- not a correctness invariant.
            # "bounded" (opt-in) falls through to the measured-cost selection
            # below instead, so a short refinement tail keeps sustaining
            # target_supervised_views_per_s rather than forcing native.
            # "progressive" (opt-in) also falls through until
            # ``refinement_progress_fraction`` crosses
            # ``refinement_progressive_switch_frac``, then forces native for
            # the remainder -- see ``_post_refinement_forces_native``.
            reason = (
                "post-refinement phase forces native supervision"
                if self._refinement_resolution_policy == "native"
                else (
                    "progressive post-refinement: progress "
                    f"{refinement_progress_fraction} >= switch frac "
                    f"{self._refinement_progressive_switch_frac}, forcing "
                    "native supervision"
                )
            )
            selection = TierSelection(
                tier=ResolutionTier.NATIVE,
                reason=reason,
                budget_ms=None,
            )
        else:
            allow_probe = transaction_class in (
                MappingTransactionClass.REPLAY_ONLY,
                MappingTransactionClass.POST_REFINEMENT,
            ) or (
                transaction_class
                in (MappingTransactionClass.FRESH, MappingTransactionClass.MIXED)
                and snapshot.freshness_mode is FreshnessMode.HEALTHY
                and not native_geometry_evidence_purpose
            )
            # Closed-loop budget relaxation against the tier-budget
            # floor-lock (see __init__). Only the strict, non-calibration
            # budget is relaxed: the native-geometry allowance path above
            # already widened `selection_budget_ms` deliberately and is left
            # untouched.
            if (
                selection_budget_ms is not None
                and selection_budget_ms == budget_ms
                and self._budget_relax_max > 1.0
            ):
                measured = self._tier_costs.measured_estimates(context=context)
                healthy = snapshot.freshness_mode is FreshnessMode.HEALTHY
                source_rate_shortfall = (
                    snapshot.source_release_rate_ratio is not None
                    and snapshot.source_release_rate_ratio < self._relax_source_rate_floor
                )
                if not healthy or source_rate_shortfall:
                    self._budget_relax_factor = 1.0
                    if source_rate_shortfall:
                        provider_reason = (
                            f"{provider_reason}; budget relaxation reset due to "
                            f"source_rate_ratio="
                            f"{snapshot.source_release_rate_ratio:.2f}"
                        )
                elif any(
                    cost_ms > selection_budget_ms * self._budget_relax_factor
                    for tier, cost_ms in measured.items()
                    if quality_rank(tier) <= quality_rank(selection_ceiling)
                ):
                    # Some tier at or below the quality ceiling is still
                    # unaffordable at the current relaxation while the
                    # pipeline demonstrably keeps up: ramp one bounded step.
                    # (Ramping stops by itself once every allowed measured
                    # tier fits, or at the cap.)
                    self._budget_relax_factor = min(
                        self._budget_relax_max,
                        self._budget_relax_factor * self._budget_relax_step,
                    )
                if self._budget_relax_factor > 1.0:
                    selection_budget_ms = budget_ms * self._budget_relax_factor
                    provider_reason = (
                        f"{provider_reason}; budget relaxed x"
                        f"{self._budget_relax_factor:.2f} (strict "
                        f"{budget_ms:.3f}ms infeasible for every measured "
                        "tier while freshness is healthy)"
                    )
            selection = self._tier_costs.select(
                ceiling=selection_ceiling,
                budget_ms=selection_budget_ms,
                context=context,
                # Probing avoids floor-lock (#1259/#1346): replay/refinement
                # can probe freely; fresh/mixed work needs HEALTHY freshness.
                # The estimator bounds probe frequency, not unknown latency,
                # and may exceed the provider's quality ceiling within legal
                # tiers. This offline-qualified behavior needs live-camera
                # qualification. E0 purposes use the separate allowance probe;
                # required source-depth work bypasses discretionary selection.
                allow_probe=allow_probe,
            )
        if (
            native_geometry_evidence_purpose
            and budget_ms is not None
            and allowance_available_ms is not None
            and ResolutionTier.NATIVE in self._tier_costs.legal_tiers
            and not valid_native
            and self._native_geometry_evidence_allowance_capacity_ms > 0.0
            and allowance_available_ms > 0.0
            and allowance_available_ms
            >= self._native_geometry_evidence_allowance_capacity_ms
        ):
            # A current purpose may spend one *fully reserved* allowance
            # window to calibrate native B4 cost when this exact bucket has no
            # native evidence. Ordinary selection remains at its normal
            # realtime budget; only this calibration-only selector receives
            # the exceptional owner-owned allowance envelope.
            calibration_budget_ms = (
                budget_ms + self._native_geometry_evidence_allowance_capacity_ms
            )
            calibration_probe = self._tier_costs.select(
                ceiling=ResolutionTier.NATIVE,
                budget_ms=calibration_budget_ms,
                context=context,
                allow_probe=True,
            )
            if (
                calibration_probe.probe
                and calibration_probe.tier is ResolutionTier.NATIVE
            ):
                native_geometry_calibration_probe = True
                selection_ceiling = ResolutionTier.NATIVE
                selection_budget_ms = calibration_budget_ms
                selection = calibration_probe
        if (
            native_geometry_evidence_purpose
            and selection.tier is ResolutionTier.NATIVE
            and budget_ms is not None
            and allowance_available_ms is not None
        ):
            native_state = self._tier_costs.state(ResolutionTier.NATIVE, context=context)
            if native_geometry_calibration_probe:
                allowance_spent_ms = self._native_geometry_evidence_allowance_capacity_ms
                self._native_geometry_evidence_allowance_available_ms = max(
                    0.0,
                    self._native_geometry_evidence_allowance_available_ms
                    - allowance_spent_ms,
                )
                allowance_available_ms = (
                    self._native_geometry_evidence_allowance_available_ms
                )
            elif native_state is not None:
                allowance_spent_ms = max(
                    0.0, native_state.estimate_ms - budget_ms
                )
                self._native_geometry_evidence_allowance_available_ms = max(
                    0.0,
                    self._native_geometry_evidence_allowance_available_ms
                    - allowance_spent_ms,
                )
                allowance_available_ms = (
                    self._native_geometry_evidence_allowance_available_ms
                )
        prior = self._last_transaction_resolution_tier
        if prior is None or prior is selection.tier:
            action = "hold"
        elif selection.tier.downsample_factor < prior.downsample_factor:
            action = "upscale"
        else:
            action = "downscale"
        self._last_transaction_resolution_tier = selection.tier
        return _ResolutionDecision(
            tier=selection.tier,
            action=action,
            reason=f"{selection.reason} [evidence: {provider_reason}]",
            budget_ms=selection.budget_ms,
            probe=selection.probe,
            ceiling=selection_ceiling,
            selection=selection,
            native_geometry_evidence_allowance_spent_ms=allowance_spent_ms,
            native_geometry_evidence_allowance_available_ms=allowance_available_ms,
        )

    @property
    def global_history_supervision_cadence(self) -> int:
        """Maximum optimizer-update spacing between reserved global slots."""

        return self._global_history_supervision_cadence

    def global_history_reservation_epoch(self, optimizer_steps_completed: int) -> int:
        """Return the O(1) history-rotation epoch for this cadence bucket."""

        if isinstance(optimizer_steps_completed, bool) or int(optimizer_steps_completed) < 0:
            raise ValueError("optimizer_steps_completed must be a non-negative integer")
        return int(optimizer_steps_completed) // self._global_history_supervision_cadence

    def _reserved_global_candidate(
        self,
        snapshot: ControlSnapshot,
        candidates: tuple[SourceIdentity, ...],
        *,
        force: bool = False,
    ) -> SourceIdentity | None:
        if type(force) is not bool:
            raise TypeError("force must be bool")
        candidate = snapshot.reserved_global_replay_candidate
        if candidate is None or candidate not in candidates:
            return None
        if force:
            return candidate
        if (
            self._cached_reservation_snapshot_id == snapshot.snapshot_id
            and self._cached_reservation_source == candidate
        ):
            return candidate
        bucket = self.global_history_reservation_epoch(
            snapshot.optimizer_steps_completed
        )
        if bucket <= self._last_reserved_global_bucket:
            return None
        self._cached_reservation_snapshot_id = snapshot.snapshot_id
        self._cached_reservation_source = candidate
        return candidate

    def _record_reserved_global_selection(
        self,
        snapshot: ControlSnapshot,
        replay_sources: tuple[SourceIdentity, ...],
    ) -> None:
        candidate = snapshot.reserved_global_replay_candidate
        if candidate is None or candidate not in replay_sources:
            return
        self._last_reserved_global_bucket = max(
            self._last_reserved_global_bucket,
            self.global_history_reservation_epoch(
                snapshot.optimizer_steps_completed
            ),
        )

    def resolve(
        self,
        snapshot: ControlSnapshot,
        *,
        disposition: ObservationDisposition,
        proposals: tuple[ControlProposal, ...],
        freshness_mode: FreshnessMode | None = None,
        native_geometry_evidence_purpose: bool = False,
        source_depth_requires_native: bool = False,
    ) -> MappingWorkPlan:
        """Allocate work to a mapping transaction from the control snapshot and proposals."""
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        if not isinstance(disposition, ObservationDisposition):
            raise TypeError("disposition must be ObservationDisposition")
        if disposition.snapshot_id != snapshot.snapshot_id:
            raise ValueError("disposition must trace to the resolved snapshot")
        if snapshot.source is not None and disposition.source != snapshot.source:
            raise ValueError("disposition source must match the resolved snapshot")
        proposals = tuple(proposals)
        if not all(isinstance(item, ControlProposal) for item in proposals):
            raise TypeError("proposals must contain ControlProposal values")
        if type(native_geometry_evidence_purpose) is not bool:
            raise TypeError("native_geometry_evidence_purpose must be bool")
        if type(source_depth_requires_native) is not bool:
            raise TypeError("source_depth_requires_native must be bool")
        resolved_freshness = freshness_mode or snapshot.freshness_mode
        if not isinstance(resolved_freshness, FreshnessMode):
            raise TypeError("freshness_mode must be FreshnessMode")
        supervision_slots, _ = _integer_request(
            proposals,
            WorkBudgetActuator.SUPERVISION_SLOTS,
            len(snapshot.supervision_candidates),
        )
        replay_slots, _ = _integer_request(proposals, WorkBudgetActuator.REPLAY_SLOTS, 0)
        optimizer_updates, _ = _integer_request(
            proposals, WorkBudgetActuator.OPTIMIZER_UPDATES, 0
        )
        insertion_seeds, _ = _integer_request(proposals, WorkBudgetActuator.INSERTION_SEEDS, 0)
        mutation_budget, _ = _integer_request(proposals, WorkBudgetActuator.MUTATION_BUDGET, 0)
        live_population_authorization = _boolean_request(
            proposals, WorkBudgetActuator.LIVE_POPULATION_AUTHORIZATION
        )
        repair_supervision, _ = _integer_request(
            proposals, WorkBudgetActuator.REGIONAL_REPAIR_SUPERVISION_SLOTS, 0
        )
        repair_replay, _ = _integer_request(
            proposals, WorkBudgetActuator.REGIONAL_REPAIR_REPLAY_SLOTS, 0
        )
        repair_optimizer, _ = _integer_request(
            proposals, WorkBudgetActuator.REGIONAL_REPAIR_OPTIMIZER_UPDATES, 0
        )
        trace = [
            f"proposal:{proposal.provider}:{proposal.actuator}={proposal.value}:"
            f"{proposal.reason}"
            for proposal in proposals
        ]
        insertion_enabled = _boolean_request(
            proposals, WorkBudgetActuator.INSERTION_ENABLED
        )

        supervision_candidates = _ordered(snapshot.supervision_candidates)
        replay_candidates = _ordered(snapshot.replay_candidates)
        replay_modalities = dict(
            zip(snapshot.replay_candidates, snapshot.replay_work_modalities, strict=True)
        )
        supervision_modalities = dict(replay_modalities)
        supervision_modalities[disposition.source] = disposition.modality_eligibility
        replay_weights = (
            dict(zip(snapshot.replay_candidates, snapshot.replay_candidate_weights, strict=True))
            if snapshot.replay_candidate_weights
            else {source: 1.0 for source in snapshot.replay_candidates}
        )
        replay_selection = ReplayCandidateSelection(
            (), (), len(replay_candidates), 0, snapshot.optimizer_steps_completed
        )
        if not (
            disposition.modality_eligibility.rgb_supervision
            or disposition.modality_eligibility.depth_supervision
        ):
            supervision_candidates = ()
            replay_candidates = ()
        else:
            # The current disposition owns newest-frame modality evidence.
            # Older pending candidates are eligible only when their own
            # identity-bound admitted modality record is in the replay store.
            supervision_candidates = tuple(
                source for source in supervision_candidates if source in supervision_modalities
            )
            replay_candidates = tuple(
                source for source in replay_candidates if source in replay_modalities
            )

        constellation_decision = (
            self._supervision_policy.decide(
                snapshot,
                stable_replay=False,
            )
            if self._adaptive_supervision_enabled
            else None
        )
        unique_eligible = tuple(
            dict.fromkeys((*supervision_candidates, *replay_candidates))
        )

        limits = snapshot.limits
        if not disposition.materialization_permitted:
            supervision_sources = ()
            replay_sources = ()
            optimizer_updates = 0
            insertion_seeds = 0
            mutation_budget = 0
            allow_growth = False
            allow_relocation = False
            trace.append("declined_observation_no_work")
        elif resolved_freshness is FreshnessMode.CRITICAL:
            if limits.max_optimizer_updates < 1:
                raise ValueError(
                    "critical snapshots require limits.max_optimizer_updates >= 1"
                )
            frontier = snapshot.newest_frontier
            if frontier is None or frontier not in supervision_candidates:
                raise ValueError("critical snapshot requires newest_frontier supervision candidate")
            requested_supervision = min(
                limits.max_supervision_slots,
                max(1, supervision_slots),
            )
            supervision_sources = supervision_candidates[-requested_supervision:]
            if frontier not in supervision_sources:
                raise ValueError("critical plan must retain newest_frontier")
            optimizer_updates = min(
                max(optimizer_updates, 1),
                limits.max_optimizer_updates,
            )
            available_replay = tuple(
                source
                for source in replay_candidates
                if source not in supervision_sources
            )
            requested_replay = min(
                limits.max_replay_slots,
                replay_slots
                + max(0, supervision_slots - len(supervision_sources)),
            )
            replay_selection = self._replay_candidate_provider.sample(
                available_replay,
                tuple(replay_weights[source] for source in available_replay),
                slots=requested_replay,
                epoch=snapshot.optimizer_steps_completed,
                snapshot_id=snapshot.snapshot_id,
                reserved_candidate=self._reserved_global_candidate(
                    snapshot, available_replay
                ),
            )
            replay_sources = replay_selection.sources
            insertion_seeds = 0
            mutation_budget = 0
            allow_growth = False
            allow_relocation = False
            trace.append("critical_newest_frontier")
            if replay_sources:
                trace.append("critical_bounded_replay_preserved")
            if repair_supervision or repair_replay or repair_optimizer:
                trace.append("critical_regional_repair_suppressed")
        else:
            requested_supervision = min(
                limits.max_supervision_slots, supervision_slots + repair_supervision
            )
            supervision_sources = (
                supervision_candidates[-requested_supervision:] if requested_supervision else ()
            )
            available_replay = tuple(
                source for source in replay_candidates if source not in supervision_sources
            )
            requested_replay = min(
                limits.max_replay_slots,
                replay_slots
                + repair_replay
                + max(0, supervision_slots - len(supervision_sources)),
            )
            replay_selection = self._replay_candidate_provider.sample(
                available_replay,
                tuple(replay_weights[source] for source in available_replay),
                slots=requested_replay,
                epoch=snapshot.optimizer_steps_completed,
                snapshot_id=snapshot.snapshot_id,
                reserved_candidate=self._reserved_global_candidate(
                    snapshot, available_replay
                ),
            )
            replay_sources = replay_selection.sources
            requested_optimizer = optimizer_updates + repair_optimizer
            optimizer_updates = min(limits.max_optimizer_updates, requested_optimizer)
            insertion_seeds = min(limits.max_insertion_seeds, insertion_seeds)
            mutation_budget = min(limits.max_mutations, mutation_budget)
            allow_growth = _boolean_request(proposals, WorkBudgetActuator.ALLOW_GROWTH)
            allow_relocation = _boolean_request(proposals, WorkBudgetActuator.ALLOW_RELOCATION)
            if not disposition.modality_eligibility.depth_insertion:
                insertion_seeds = 0
            if not insertion_enabled:
                insertion_seeds = 0
            if (
                repair_supervision > 0
                and len(supervision_sources) < supervision_slots + repair_supervision
            ) or (
                repair_replay > 0 and len(replay_sources) < replay_slots + repair_replay
            ) or (repair_optimizer > 0 and optimizer_updates < requested_optimizer):
                trace.append("regional_repair_exhausted")

        # B1 has no spare replay lane. Reserve one complete optimizer
        # transaction per cadence bucket for rotating global history while
        # retaining the newly admitted observation as materialized historical
        # evidence for a later transaction. This is update-clock owned: source
        # arrival timing neither creates nor advances the reservation.
        active_b1_reserved = False
        if (
            disposition.materialization_permitted
            and self._requested_batch_size == 1
            and limits.max_optimizer_updates >= 1
        ):
            b1_history = tuple(
                source
                for source in replay_candidates
                if source != disposition.source
            )
            reserved_candidate = self._reserved_global_candidate(
                snapshot, b1_history
            )
            if reserved_candidate is not None:
                replay_selection = self._replay_candidate_provider.sample(
                    b1_history,
                    tuple(replay_weights[source] for source in b1_history),
                    slots=1,
                    epoch=snapshot.optimizer_steps_completed,
                    snapshot_id=snapshot.snapshot_id,
                    reserved_candidate=reserved_candidate,
                )
                if replay_selection.sources == (reserved_candidate,):
                    supervision_sources = ()
                    replay_sources = replay_selection.sources
                    optimizer_updates = 1
                    insertion_seeds = 0
                    mutation_budget = 0
                    allow_growth = False
                    allow_relocation = False
                    active_b1_reserved = True
                    trace.append("admitted_source_retained_for_later_replay")

        # B is a transaction budget, never a fill prerequisite. Reconcile all
        # provider proposals at this sole actuator, choose each identity at
        # most once, and immediately realize whatever unique work exists.
        target_batch_size = (
            constellation_decision.requested_batch_size
            if constellation_decision is not None
            else self._requested_batch_size
        )
        desired_batch_size = min(
            target_batch_size,
            len(unique_eligible),
        )
        if not disposition.materialization_permitted:
            desired_batch_size = 0
        chosen_supervision = list(supervision_sources)
        chosen_replay = list(replay_sources)
        if not self._adaptive_supervision_enabled:
            desired_batch_size = len(chosen_supervision) + len(chosen_replay)
        committed_source_missing = (
            disposition.materialization_permitted
            and not active_b1_reserved
            and disposition.source not in chosen_supervision
        )
        if disposition.materialization_permitted:
            # ObservationOwner has already made the terminal semantic COMMIT.
            # MappingSession therefore materializes this exact source after the
            # disposition sink returns.  A proposal may reduce optional
            # optimization work, but it cannot revoke the identity-bearing
            # materialization lane owned by that commit.
            desired_batch_size = max(1, desired_batch_size)
        if len(chosen_supervision) + len(chosen_replay) > desired_batch_size:
            if (
                resolved_freshness is FreshnessMode.CRITICAL
                and chosen_replay
                and desired_batch_size > 1
            ):
                chosen_supervision = chosen_supervision[-(desired_batch_size - 1) :]
                chosen_replay = chosen_replay[:1]
            else:
                combined = [
                    *((source, True) for source in chosen_supervision),
                    *((source, False) for source in chosen_replay),
                ][:desired_batch_size]
                chosen_supervision = [source for source, is_new in combined if is_new]
                chosen_replay = [source for source, is_new in combined if not is_new]
        chosen = set((*chosen_supervision, *chosen_replay))
        if len(chosen) < desired_batch_size:
            for source in replay_candidates:
                if source not in chosen:
                    chosen_replay.append(source)
                    chosen.add(source)
                    if len(chosen) == desired_batch_size:
                        break
        if len(chosen) < desired_batch_size:
            for source in reversed(supervision_candidates):
                if source not in chosen:
                    chosen_supervision.append(source)
                    chosen.add(source)
                    if len(chosen) == desired_batch_size:
                        break
        if (
            disposition.materialization_permitted
            and not active_b1_reserved
            and disposition.source not in chosen_supervision
        ):
            if disposition.source in chosen_replay:
                chosen_replay.remove(disposition.source)
                chosen_supervision.append(disposition.source)
            elif chosen_supervision:
                displaced = chosen_supervision.pop(0)
                chosen.remove(displaced)
                chosen_supervision.append(disposition.source)
                chosen.add(disposition.source)
            elif chosen_replay:
                displaced = chosen_replay.pop(0)
                chosen.remove(displaced)
                chosen_supervision.append(disposition.source)
                chosen.add(disposition.source)
            else:
                raise RuntimeError(
                    "committed source has no realizable materialization lane"
                )
        if committed_source_missing:
            trace.append("committed_source_materialization_floor")
        supervision_sources = _ordered(tuple(chosen_supervision))
        replay_sources = tuple(chosen_replay)
        if len(chosen) != desired_batch_size:
            raise RuntimeError("WorkBudgetOwner failed to realize unique eligible work")
        if (
            self._adaptive_supervision_enabled
            and resolved_freshness is FreshnessMode.CRITICAL
            and replay_candidates
            and desired_batch_size > 1
            and not replay_sources
        ):
            raise RuntimeError("critical backlog must preserve an eligible replay lane")
        self._record_reserved_global_selection(snapshot, replay_sources)

        if constellation_decision is None:
            constellation_decision = _SupervisionTarget(
                requested_batch_size=max(1, desired_batch_size),
                requested_camera_count=max(1, desired_batch_size),
                cumulative_realized_supervised_views=(
                    snapshot.supervision_views_completed
                ),
                cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
                reason="adaptive_disabled",
                hysteresis_state="disabled",
            )
        constellation = _supervision_constellation(
            decision=constellation_decision,
            realized_batch_size=desired_batch_size,
            unique_eligible_count=len(unique_eligible),
        )
        native_geometry_evidence_ready = bool(
            native_geometry_evidence_purpose
            and constellation.realized_batch_size == 4
            and constellation.realized_camera_count == 4
            and all(
                supervision_modalities[source].geometry_population_target
                for source in (*supervision_sources, *replay_sources)
            )
        )
        if native_geometry_evidence_purpose and not native_geometry_evidence_ready:
            trace.append("native_geometry_evidence:deferred_unready")
        if live_population_authorization:
            resolved_b4_c4_opportunity = bool(
                supervision_sources
                and optimizer_updates == 1
                and constellation.realized_batch_size == 4
                and constellation.realized_camera_count == 4
            )
            mutation_budget = min(
                mutation_budget, int(resolved_b4_c4_opportunity)
            )
        trace.append(
            "adaptive_supervision:"
            f"B={constellation.requested_batch_size}->{constellation.realized_batch_size}:"
            f"C={constellation.requested_camera_count}->{constellation.realized_camera_count}:"
            f"microbatches={constellation.microbatch_count}:"
            f"reason={constellation.reason}:hysteresis={constellation.hysteresis_state}"
        )

        if replay_sources:
            trace.append("weighted_random_full_cache")
            if snapshot.reserved_global_replay_candidate in replay_sources:
                trace.append("reserved_global_history_replay")
            trace.append(
                "replay_sample:"
                f"population={replay_selection.population_size}:"
                f"slots={len(replay_selection.sources)}:"
                f"epoch={replay_selection.epoch}:seed={replay_selection.seed}"
            )

        if supervision_sources and replay_sources:
            transaction_class = MappingTransactionClass.MIXED
        elif supervision_sources:
            transaction_class = MappingTransactionClass.FRESH
        elif replay_sources:
            transaction_class = MappingTransactionClass.REPLAY_ONLY
        else:
            transaction_class = None
        decision = self._resolve_transaction_resolution(
            snapshot=snapshot,
            transaction_class=transaction_class,
            proposals=proposals,
            realized_camera_count=constellation.realized_camera_count,
            default_tier=ResolutionTier.NATIVE,
            native_geometry_evidence_purpose=native_geometry_evidence_ready,
            source_depth_requires_native=bool(
                source_depth_requires_native
                and not live_population_authorization
                and insertion_seeds > 0
                and optimizer_updates > 0
                and supervision_sources
                and disposition.source == supervision_sources[-1]
            ),
        )
        transaction_tier = decision.tier
        trace.append(
            "resolution_decision:"
            f"class={None if transaction_class is None else transaction_class.value}:"
            f"tier={transaction_tier.value}:action={decision.action}:"
            f"budget_ms={decision.budget_ms}:probe={decision.probe}:"
            f"ceiling={decision.ceiling.value}:reason={decision.reason}"
        )
        if native_geometry_evidence_ready:
            trace.append(
                "native_geometry_evidence:considered_native:"
                f"allowance_spent_ms={decision.native_geometry_evidence_allowance_spent_ms:.3f}:"
                f"allowance_available_ms={decision.native_geometry_evidence_allowance_available_ms:.3f}"
                if transaction_tier is ResolutionTier.NATIVE
                else "native_geometry_evidence:deferred_budget:"
                f"allowance_available_ms={decision.native_geometry_evidence_allowance_available_ms:.3f}"
            )

        return MappingWorkPlan(
            plan_id=(
                f"work-{snapshot.snapshot_id}-v{snapshot.control_version}"
                f"-m{snapshot.map_content_version}-t{snapshot.topology_version}"
                f"-p{snapshot.parameter_version}"
            ),
            snapshot_id=snapshot.snapshot_id,
            control_version=snapshot.control_version,
            supervision_sources=supervision_sources,
            replay_sources=replay_sources,
            resolution_tier=transaction_tier,
            optimizer_updates=optimizer_updates,
            insertion=DepthInsertionPlan(
                source=(
                    disposition.source if insertion_seeds > 0 else None
                ),
                enabled=insertion_seeds > 0,
                sample_stride=self._sample_stride,
                candidate_budget=insertion_seeds,
                seed_budget=insertion_seeds,
                voxel_size_m=self._depth_insertion.voxel_size_m,
                minimum_confidence=self._depth_insertion.minimum_confidence,
            ),
            allow_growth=allow_growth,
            allow_relocation=allow_relocation,
            mutation_budget=MutationBudget(max_total=mutation_budget),
            freshness_mode=resolved_freshness,
            map_content_version=snapshot.map_content_version,
            topology_version=snapshot.topology_version,
            parameter_version=snapshot.parameter_version,
            supervision_work=tuple(
                SupervisionWork(
                    source=source,
                    resolution_tier=transaction_tier,
                    modality_eligibility=supervision_modalities[source],
                )
                for source in supervision_sources
            ),
            replay_work=tuple(
                SupervisionWork(
                    source=source,
                    resolution_tier=transaction_tier,
                    modality_eligibility=replay_modalities[source],
                )
                for source in replay_sources
            ),
            reason_trace=tuple(trace),
            replay_population_size=replay_selection.population_size,
            replay_sampling_seed=replay_selection.seed,
            replay_sampling_epoch=replay_selection.epoch,
            replay_selected_weights=tuple(
                replay_weights[source] for source in replay_sources
            ),
            requested_supervision_slots=supervision_slots,
            requested_replay_slots=replay_slots,
            supervision_constellation=constellation,
            materialization_sources=(
                (disposition.source,)
                if disposition.materialization_permitted
                else ()
            ),
            resolution_action=decision.action,
            resolution_reason=decision.reason,
            resolution_budget_ms=decision.budget_ms,
            resolution_probe=decision.probe,
            native_geometry_evidence_purpose=native_geometry_evidence_purpose,
        )

    def resolve_idle_replay(
        self,
        snapshot: ControlSnapshot,
        active_plan: MappingWorkPlan,
        *,
        replay_modalities: Mapping[SourceIdentity, "ModalityEligibility"],
        replay_weights: Mapping[SourceIdentity, float],
        batch_size: int,
        optimizer_steps_completed: int,
        versions: tuple[int, int, int],
        proposals: tuple[ControlProposal, ...] = (),
        post_refinement: bool = False,
        refinement_progress_fraction: float | None = None,
        force_reserved_global: bool = False,
        cold_challenger_sources: tuple[SourceIdentity, ...] = (),
        challenger_fraction: float = 0.0,
    ) -> MappingWorkPlan:
        """Mint zero-frame replay work through the same identity actuator.

        Completed transaction cost is deliberately *not* an argument here.
        Cost feedback enters through :meth:`record_completed_transaction`,
        which updates bounded per-tier estimator state; this method only reads
        that state through the single resolution owner.
        """

        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(post_refinement) is not bool:
            raise TypeError("post_refinement must be bool")
        if type(force_reserved_global) is not bool:
            raise TypeError("force_reserved_global must be bool")
        if (
            isinstance(challenger_fraction, bool)
            or not math.isfinite(float(challenger_fraction))
            or not 0.0 <= float(challenger_fraction) <= 1.0
        ):
            raise ValueError("challenger_fraction must be finite and within [0, 1]")
        if refinement_progress_fraction is not None and (
            isinstance(refinement_progress_fraction, bool)
            or not math.isfinite(float(refinement_progress_fraction))
        ):
            raise TypeError(
                "refinement_progress_fraction must be None or a finite number"
            )
        proposals = tuple(proposals)
        if not all(isinstance(item, ControlProposal) for item in proposals):
            raise TypeError("idle proposals must contain ControlProposal values")
        candidates = _ordered(tuple(replay_modalities))
        candidate_weights = tuple(
            float(replay_weights.get(source, 1.0)) for source in candidates
        )
        reserved_global = self._reserved_global_candidate(
            snapshot, candidates, force=force_reserved_global
        )
        if post_refinement:
            constellation_decision = self._supervision_policy.post_refinement_target(
                snapshot
            )
        elif (
            self._adaptive_supervision_enabled
            and int(batch_size) == self._requested_batch_size
        ):
            constellation_decision = self._supervision_policy.decide(
                snapshot,
                stable_replay=True,
            )
        else:
            # Compatibility for direct legacy callers that supplied B only at
            # this method. Canonical production binds B in the owner itself.
            constellation_decision = _SupervisionTarget(
                requested_batch_size=int(batch_size),
                requested_camera_count=qualified_camera_count_for_batch(
                    int(batch_size)
                ),
                cumulative_realized_supervised_views=(
                    snapshot.supervision_views_completed
                ),
                cumulative_optimizer_updates=snapshot.optimizer_steps_completed,
                reason=(
                    "legacy_idle_batch_size"
                    if self._adaptive_supervision_enabled
                    else "adaptive_disabled"
                ),
                hysteresis_state=(
                    "unmanaged" if self._adaptive_supervision_enabled else "disabled"
                ),
            )
        challenger_reservations: tuple[SourceIdentity, ...] = ()
        if challenger_fraction > 0.0 and not post_refinement:
            if not isinstance(cold_challenger_sources, tuple) or not all(
                isinstance(source, SourceIdentity) for source in cold_challenger_sources
            ):
                raise TypeError("cold_challenger_sources must be a tuple of SourceIdentity")
            if len(set(cold_challenger_sources)) != len(cold_challenger_sources):
                raise ValueError("cold_challenger_sources must be unique")
            if any(source not in candidates for source in cold_challenger_sources):
                raise ValueError("cold_challenger_sources must be replay eligible")
            if cold_challenger_sources:
                available_slots = min(
                    constellation_decision.requested_batch_size, len(candidates)
                )
                quota = min(
                    len(cold_challenger_sources),
                    math.floor(available_slots * float(challenger_fraction)),
                    max(0, available_slots - 1 - int(reserved_global is not None)),
                )
                offset = int(optimizer_steps_completed) % len(cold_challenger_sources)
                rotated = cold_challenger_sources[offset:] + cold_challenger_sources[:offset]
                challenger_reservations = rotated[:quota]
        selection = self._replay_candidate_provider.sample(
            candidates,
            candidate_weights,
            slots=min(constellation_decision.requested_batch_size, len(candidates)),
            epoch=int(optimizer_steps_completed),
            snapshot_id=snapshot.snapshot_id,
            reserved_candidate=reserved_global,
            reserved_candidates=challenger_reservations,
        )
        self._record_reserved_global_selection(snapshot, selection.sources)
        # The constellation only needs the realized view count, which the
        # candidate selection already fixes.  Resolving it here lets the sole
        # resolution owner see the real transaction composition (class and C)
        # before it selects the one tier the whole plan is stamped with.
        constellation = _supervision_constellation(
            decision=constellation_decision,
            realized_batch_size=len(selection.sources),
            unique_eligible_count=len(candidates),
        )
        transaction_class = (
            MappingTransactionClass.POST_REFINEMENT
            if post_refinement
            else MappingTransactionClass.REPLAY_ONLY
        )
        decision = self._resolve_transaction_resolution(
            snapshot=snapshot,
            transaction_class=transaction_class,
            proposals=proposals,
            realized_camera_count=constellation.realized_camera_count,
            default_tier=(
                active_plan.replay_work[0].resolution_tier
                if active_plan.replay_work
                else active_plan.resolution_tier
            ),
            refinement_progress_fraction=refinement_progress_fraction,
        )
        replay_tier = decision.tier
        allow_growth = _boolean_request(
            proposals, WorkBudgetActuator.ALLOW_GROWTH
        )
        allow_relocation = _boolean_request(
            proposals, WorkBudgetActuator.ALLOW_RELOCATION
        )
        mutation_budget, _ = _integer_request(
            proposals, WorkBudgetActuator.MUTATION_BUDGET, 0
        )
        replay_insertion_seeds, _ = _integer_request(
            proposals, WorkBudgetActuator.REPLAY_INSERTION_SEEDS, 0
        )
        mutation_budget = min(snapshot.limits.max_mutations, mutation_budget)
        replay_work = tuple(
            SupervisionWork(
                source=source,
                resolution_tier=replay_tier,
                modality_eligibility=replay_modalities[source],
            )
            for source in selection.sources
        )
        replay_insertion_seeds = min(
            snapshot.limits.max_replay_insertion_seeds,
            replay_insertion_seeds,
        )
        replay_insertion_source = next(
            (
                work.source
                for work in replay_work
                if work.modality_eligibility.depth_insertion
            ),
            None,
        )
        replay_insertion = None
        if replay_insertion_source is not None and replay_insertion_seeds > 0:
            replay_insertion = DepthInsertionPlan(
                source=replay_insertion_source,
                enabled=True,
                sample_stride=self._sample_stride,
                candidate_budget=replay_insertion_seeds,
                seed_budget=replay_insertion_seeds,
                voxel_size_m=self._replay_insertion_voxel_size_m,
                minimum_confidence=self._depth_insertion.minimum_confidence,
            )
            allow_growth = False
            allow_relocation = False
            mutation_budget = 0
        content, topology, parameters = versions
        return replace(
            active_plan,
            plan_id=(
                f"work-{snapshot.snapshot_id}-v{snapshot.control_version}"
                f"-m{content}-t{topology}-p{parameters}-idle"
            ),
            snapshot_id=snapshot.snapshot_id,
            control_version=snapshot.control_version,
            resolution_tier=replay_tier,
            supervision_sources=(),
            supervision_work=(),
            materialization_sources=(),
            replay_sources=selection.sources,
            replay_work=replay_work,
            optimizer_updates=max(1, int(active_plan.optimizer_updates)),
            insertion=replace(
                active_plan.insertion,
                source=None,
                enabled=False,
                candidate_budget=0,
                seed_budget=0,
            ),
            replay_insertion=replay_insertion,
            allow_growth=allow_growth,
            allow_relocation=allow_relocation,
            mutation_budget=replace(
                active_plan.mutation_budget, max_total=mutation_budget
            ),
            freshness_mode=(
                FreshnessMode.HEALTHY
                if post_refinement
                else snapshot.freshness_mode
            ),
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            reason_trace=(
                *( () if post_refinement else active_plan.reason_trace),
                *(
                    f"proposal:{proposal.provider}:{proposal.actuator}="
                    f"{proposal.value}:{proposal.reason}"
                    for proposal in proposals
                ),
                "source_idle_replay_only",
                *(
                    (
                        "post_refinement",
                        "post_refinement_live_freshness_bypassed",
                        "post_refinement_constellation_phase_resolved",
                    )
                    if post_refinement
                    else ()
                ),
                *(
                    ("replay_depth_refinement", "replay_depth_suppresses_mcmc")
                    if replay_insertion is not None
                    else ()
                ),
                "weighted_random_full_cache",
                *(
                    ("reserved_global_history_replay",)
                    if snapshot.reserved_global_replay_candidate in selection.sources
                    else ()
                ),
                "replay_sample:"
                f"population={selection.population_size}:slots={len(selection.sources)}:"
                f"epoch={selection.epoch}:seed={selection.seed}",
                "adaptive_supervision:"
                f"B={constellation.requested_batch_size}->{constellation.realized_batch_size}:"
                f"C={constellation.requested_camera_count}->{constellation.realized_camera_count}:"
                f"microbatches={constellation.microbatch_count}:"
                f"reason={constellation.reason}:hysteresis={constellation.hysteresis_state}",
                "resolution_decision:"
                f"class={transaction_class.value}:tier={replay_tier.value}:"
                f"action={decision.action}:budget_ms={decision.budget_ms}:"
                f"probe={decision.probe}:ceiling={decision.ceiling.value}:"
                f"reason={decision.reason}",
            ),
            post_refinement=post_refinement,
            resolution_action=decision.action,
            resolution_reason=decision.reason,
            resolution_budget_ms=decision.budget_ms,
            resolution_probe=decision.probe,
            replay_population_size=selection.population_size,
            replay_sampling_seed=selection.seed,
            replay_sampling_epoch=selection.epoch,
            replay_selected_weights=selection.weights,
            replay_candidate_sources=candidates,
            replay_candidate_weights=candidate_weights,
            replay_trial_sources=tuple(
                source for source in selection.sources if source in challenger_reservations
            ),
            requested_supervision_slots=0,
            requested_replay_slots=int(batch_size),
            supervision_constellation=constellation,
        )


def collect_control_proposals(
    snapshot: ControlSnapshot, providers: tuple[object, ...]
) -> tuple[ControlProposal, ...]:
    """Collect evidence-only policy outputs from the identical frozen snapshot."""

    if not isinstance(snapshot, ControlSnapshot):
        raise TypeError("snapshot must be ControlSnapshot")
    collected: list[ControlProposal] = []
    for provider in tuple(providers):
        propose = getattr(provider, "propose", None)
        if not callable(propose):
            raise TypeError("control proposal providers must define propose(snapshot)")
        output = propose(snapshot)
        items = (output,) if isinstance(output, ControlProposal) else tuple(output)
        if not items or not all(isinstance(item, ControlProposal) for item in items):
            raise TypeError("proposal providers must return ControlProposal values")
        collected.extend(items)
    return tuple(collected)


class CurrentSupervisionProposalProvider:
    """Propose one supervision slot for a current observation."""

    def propose(self, snapshot: ControlSnapshot) -> tuple[ControlProposal, ...]:
        """Return control proposals for current observation supervision."""
        slots = int(snapshot.source is not None)
        return (
            ControlProposal(
                self.__class__.__name__,
                "supervision_slots",
                slots,
                "one current-observation supervision slot",
            ),
        )


class BatchConstellationProposalProvider:
    """Propose one hysteretic new/replay mix at the frozen snapshot barrier."""

    def __init__(self, *, batch_size: int, pressure_dwell: int = 3) -> None:
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(pressure_dwell, bool) or int(pressure_dwell) < 1:
            raise ValueError("pressure_dwell must be a positive integer")
        self._batch_size = int(batch_size)
        self._pressure_dwell = int(pressure_dwell)
        self._pressure_streak = 0
        self._new_biased = False
        self._last_tick_epoch: int | None = None
        self._last_proposals: tuple[ControlProposal, ...] | None = None

    def propose(self, snapshot: ControlSnapshot) -> tuple[ControlProposal, ...]:
        """Return proposals for supervision/replay batch allocation."""
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        tick_epoch = int(snapshot.control_tick_epoch)
        if self._last_tick_epoch == tick_epoch:
            assert self._last_proposals is not None
            return self._last_proposals
        if self._last_tick_epoch is not None and tick_epoch < self._last_tick_epoch:
            raise ValueError("control_tick_epoch must be monotonic")
        pressured = snapshot.freshness_mode in {
            FreshnessMode.CATCH_UP,
            FreshnessMode.CRITICAL,
        }
        if pressured:
            self._pressure_streak += 1
            if self._pressure_streak >= self._pressure_dwell:
                self._new_biased = True
        else:
            # Protect old-scene detail immediately once backlog pressure drops.
            self._pressure_streak = 0
            self._new_biased = False

        if self._batch_size == 1:
            supervision_slots, replay_slots = 1, 0
        elif self._new_biased:
            supervision_slots = min(
                self._batch_size - 1,
                snapshot.limits.max_supervision_slots,
            )
            replay_slots = min(
                self._batch_size - supervision_slots,
                snapshot.limits.max_replay_slots,
            )
        else:
            supervision_slots = 1
            replay_slots = min(
                self._batch_size - 1,
                snapshot.limits.max_replay_slots,
            )
        phase = "new_biased" if self._new_biased else "replay_biased"
        basis = (
            f"{phase};freshness={snapshot.freshness_mode.value};"
            f"pressure_streak={self._pressure_streak}/{self._pressure_dwell}"
        )
        proposals = (
            ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.SUPERVISION_SLOTS.value,
                supervision_slots,
                basis,
            ),
            ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.REPLAY_SLOTS.value,
                replay_slots,
                basis,
            ),
        )
        self._last_tick_epoch = tick_epoch
        self._last_proposals = proposals
        return proposals


class OptimizerProposalProvider:
    """Propose optimizer updates for the snapshot."""

    def propose(self, snapshot: ControlSnapshot) -> ControlProposal:
        """Return optimizer update count proposal."""
        true_batch = snapshot.limits.max_optimizer_updates == 1
        updates = 1 if true_batch else 1 + int(bool(snapshot.replay_candidates))
        reason = "one true-camera batch update" if true_batch else "sequential B=1 updates"
        return ControlProposal(self.__class__.__name__, "optimizer_updates", updates, reason)


class InsertionProposalProvider:
    """Propose depth insertion seeding and enablement."""

    def __init__(self, requested_seeds: int, *, enabled: bool = True) -> None:
        if isinstance(requested_seeds, bool) or int(requested_seeds) < 0:
            raise ValueError("requested_seeds must be a non-negative integer")
        self._requested_seeds = int(requested_seeds)
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self._enabled = enabled

    def propose(self, snapshot: ControlSnapshot) -> tuple[ControlProposal, ...]:
        """Return insertion enablement and seed count proposals."""
        requested = 0 if snapshot.freshness_mode is FreshnessMode.CRITICAL else self._requested_seeds
        return (
            ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.INSERTION_ENABLED.value,
                self._enabled,
                "static depth insertion configuration",
            ),
            ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.INSERTION_SEEDS.value,
                requested,
                "bounded depth insertion",
            ),
        )


class AdaptiveGrowthProposalProvider:
    """Steer MCMC growth from training-only quality and population evidence.

    This provider owns hysteresis and evidence history, but it does not mutate
    the map.  It emits one ``allow_growth`` proposal which
    :class:`WorkBudgetOwner` combines fail-closed with the independent mutation
    and VRAM safety proposals at an optimizer barrier.  Local spatial fading
    remains replay-allocation evidence; it can corroborate negative aggregate
    growth utility but cannot brake topology growth by itself.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        marginal_utility_brake: float,
        marginal_utility_release: float,
        hysteresis: int,
        minimum_population_delta: int,
        utility_ema: float,
        regression_tolerance_db: float,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        for name, value in (
            ("marginal_utility_brake", marginal_utility_brake),
            ("marginal_utility_release", marginal_utility_release),
            ("utility_ema", utility_ema),
            ("regression_tolerance_db", regression_tolerance_db),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(marginal_utility_release) < float(marginal_utility_brake):
            raise ValueError(
                "marginal_utility_release must be greater than or equal to "
                "marginal_utility_brake"
            )
        if isinstance(hysteresis, bool) or int(hysteresis) < 1:
            raise ValueError("hysteresis must be a positive integer")
        if (
            isinstance(minimum_population_delta, bool)
            or int(minimum_population_delta) < 1
        ):
            raise ValueError("minimum_population_delta must be a positive integer")
        if not 0.0 < float(utility_ema) <= 1.0:
            raise ValueError("utility_ema must be in (0, 1]")
        if float(regression_tolerance_db) < 0.0:
            raise ValueError("regression_tolerance_db must be non-negative")

        self._enabled = enabled
        self._marginal_utility_brake = float(marginal_utility_brake)
        self._marginal_utility_release = float(marginal_utility_release)
        self._hysteresis = int(hysteresis)
        self._minimum_population_delta = int(minimum_population_delta)
        self._utility_ema_alpha = float(utility_ema)
        self._regression_tolerance_db = float(regression_tolerance_db)

        self._last_iteration: int | None = None
        self._anchor_psnr_db: float | None = None
        self._anchor_population: int | None = None
        self._peak_psnr_db: float | None = None
        self._marginal_utility_ema: float | None = None
        self._braked = False
        self._brake_streak = 0
        self._release_streak = 0
        self._last_transition = "warmup"
        self._latest_local_fading = False

    def observe_training_spatial_quality(
        self,
        *,
        iteration: int,
        aggregate_psnr_db: float,
        gaussian_count: int,
        worst_drop_from_peak_db: float,
    ) -> None:
        """Consume one pre-mutation, training-camera quality observation."""

        if isinstance(iteration, bool) or int(iteration) < 0:
            raise ValueError("iteration must be a non-negative integer")
        if isinstance(gaussian_count, bool) or int(gaussian_count) < 0:
            raise ValueError("gaussian_count must be a non-negative integer")
        for name, value in (
            ("aggregate_psnr_db", aggregate_psnr_db),
            ("worst_drop_from_peak_db", worst_drop_from_peak_db),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(worst_drop_from_peak_db) < 0.0:
            raise ValueError("worst_drop_from_peak_db must be non-negative")

        resolved_iteration = int(iteration)
        resolved_population = int(gaussian_count)
        resolved_psnr = float(aggregate_psnr_db)
        resolved_drop = float(worst_drop_from_peak_db)
        if (
            self._last_iteration is not None
            and resolved_iteration <= self._last_iteration
        ):
            raise ValueError("iteration must increase monotonically")
        self._last_iteration = resolved_iteration

        if self._peak_psnr_db is None:
            self._peak_psnr_db = resolved_psnr
            self._anchor_psnr_db = resolved_psnr
            self._anchor_population = resolved_population
            return
        self._peak_psnr_db = max(self._peak_psnr_db, resolved_psnr)
        self._latest_local_fading = (
            resolved_drop > self._regression_tolerance_db
        )

        assert self._anchor_psnr_db is not None
        assert self._anchor_population is not None
        population_delta = resolved_population - self._anchor_population
        utility_sampled = False
        if population_delta < 0:
            # Relocation/retirement may lower N.  That is not evidence about
            # growth utility, so establish a new causal anchor.
            self._anchor_psnr_db = resolved_psnr
            self._anchor_population = resolved_population
        elif population_delta >= self._minimum_population_delta:
            utility = (resolved_psnr - self._anchor_psnr_db) / population_delta
            if self._marginal_utility_ema is None:
                self._marginal_utility_ema = utility
            else:
                alpha = self._utility_ema_alpha
                self._marginal_utility_ema = (
                    alpha * utility + (1.0 - alpha) * self._marginal_utility_ema
                )
            self._anchor_psnr_db = resolved_psnr
            self._anchor_population = resolved_population
            utility_sampled = True

        if not self._braked:
            poor_utility = (
                utility_sampled
                and self._marginal_utility_ema is not None
                and self._marginal_utility_ema < self._marginal_utility_brake
            )
            # A local partition regression is replay-allocation evidence, not
            # sufficient authority to stop global topology growth.  Fine
            # spatial cells legitimately dip as new scene content arrives.
            # Brake only when that local signal corroborates measured negative
            # aggregate PSNR utility from increasing the population.
            corroborated_regression = poor_utility and self._latest_local_fading
            if corroborated_regression:
                self._brake_streak += 1
            else:
                self._brake_streak = 0
            if self._brake_streak >= self._hysteresis:
                self._braked = True
                self._release_streak = 0
                self._last_transition = "braked"
            return

        # Once growth is braked there may be no population delta from which to
        # estimate fresh marginal utility.  Replay recovery of aggregate PSNR
        # must therefore be sufficient to reopen a bounded growth opportunity;
        # persistent local debt remains owned by replay weighting and cannot
        # deadlock topology growth indefinitely.
        recovered_to_peak = (
            resolved_psnr
            >= self._peak_psnr_db - self._regression_tolerance_db
        )
        positive_utility = (
            utility_sampled
            and self._marginal_utility_ema is not None
            and self._marginal_utility_ema >= self._marginal_utility_release
        )
        if recovered_to_peak or positive_utility:
            self._release_streak += 1
        else:
            self._release_streak = 0
        if self._release_streak >= self._hysteresis:
            self._braked = False
            self._brake_streak = 0
            self._last_transition = "released"

    def propose_idle(self, snapshot: ControlSnapshot) -> ControlProposal:
        """Return growth allowance proposal based on quality and utility evidence."""
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        if not self._enabled:
            return ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.ALLOW_GROWTH.value,
                True,
                "adaptive growth disabled",
            )
        if self._last_iteration is None:
            reason = "warmup: no training spatial quality evidence"
        elif self._braked:
            utility = (
                "unknown"
                if self._marginal_utility_ema is None
                else f"{self._marginal_utility_ema:.6g} dB/gaussian"
            )
            reason = (
                "braked: local fade corroborated negative growth utility;"
                f"utility={utility}"
            )
        elif self._last_transition == "released":
            reason = "released: replay restored training spatial quality"
        else:
            reason = (
                "growth active: positive aggregate utility; local replay repair active"
                if self._latest_local_fading
                else "growth active: training spatial quality remains stable"
            )
        return ControlProposal(
            self.__class__.__name__,
            WorkBudgetActuator.ALLOW_GROWTH.value,
            not self._braked,
            reason,
        )


class MutationProposalProvider:
    """Propose mutation control policies and budgets."""

    def __init__(self, *, live_population_authorization: bool = False) -> None:
        if type(live_population_authorization) is not bool:
            raise TypeError("live_population_authorization must be bool")
        self._live_population_authorization = live_population_authorization

    def propose(self, snapshot: ControlSnapshot) -> tuple[ControlProposal, ...]:
        """Return fail-closed mutation proposals for supervised transactions."""
        del snapshot
        return (
            ControlProposal(self.__class__.__name__, "allow_growth", False, "fail closed"),
            ControlProposal(self.__class__.__name__, "allow_relocation", False, "fail closed"),
            ControlProposal(
                self.__class__.__name__,
                WorkBudgetActuator.LIVE_POPULATION_AUTHORIZATION.value,
                self._live_population_authorization,
                "live policy authorization mode",
            ),
            ControlProposal(
                self.__class__.__name__,
                "mutation_budget",
                int(self._live_population_authorization),
                (
                    "live population policy may evaluate completed supervision"
                    if self._live_population_authorization
                    else "fail closed"
                ),
            ),
        )

    def propose_idle(self, snapshot: ControlSnapshot) -> tuple[ControlProposal, ...]:
        """Return mutation proposals for idle replay-only transactions."""
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        return (
            ControlProposal(
                self.__class__.__name__, "allow_growth", True,
                "replay-only optimizer barrier permits scheduled MCMC growth",
            ),
            ControlProposal(
                self.__class__.__name__, "allow_relocation", True,
                "replay-only optimizer barrier permits scheduled MCMC relocation",
            ),
            ControlProposal(
                self.__class__.__name__, "mutation_budget", 1,
                "at most one delegated MCMC epoch after fixed-topology work",
            ),
        )


__all__ = [
    "AdaptiveGrowthProposalProvider",
    "BatchConstellationProposalProvider",
    "CurrentSupervisionProposalProvider",
    "InsertionProposalProvider",
    "MutationProposalProvider",
    "OptimizerProposalProvider",
    "WorkBudgetActuator",
    "WorkBudgetOwner",
    "collect_control_proposals",
]
