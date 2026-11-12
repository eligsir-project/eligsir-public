"""Evidence-only realtime resolution proposal for canonical work allocation."""

from __future__ import annotations

import math

from eligsir.contracts.control import (
    ControlProposal,
    ControlSnapshot,
    FreshnessMode,
    ResolutionTier,
)


class ResolutionPolicyProvider:
    """Choose a conservative tier from immutable freshness/queue evidence.

    WorkBudgetOwner remains the sole actuator: this provider only proposes a
    typed tier and never mutates frame state or runtime arguments.
    """

    def __init__(
        self,
        *,
        supported_tiers: tuple[ResolutionTier, ...] = tuple(ResolutionTier),
        target_views_per_s: float = 100.0,
        idle_upscale_dwell: int = 8,
        idle_upscale_margin: float = 0.7,
        batch_size: int = 1,
        fresh_upscale_dwell: int | None = None,
    ):
        if not supported_tiers:
            raise ValueError("supported_tiers must not be empty")
        if not all(isinstance(tier, ResolutionTier) for tier in supported_tiers):
            raise TypeError("supported_tiers must contain ResolutionTier values")
        if not math.isfinite(float(target_views_per_s)) or float(target_views_per_s) <= 0:
            raise ValueError("target_views_per_s must be finite and positive")
        if isinstance(idle_upscale_dwell, bool) or int(idle_upscale_dwell) < 1:
            raise ValueError("idle_upscale_dwell must be a positive integer")
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        if fresh_upscale_dwell is not None and (
            isinstance(fresh_upscale_dwell, bool) or int(fresh_upscale_dwell) < 1
        ):
            raise ValueError("fresh_upscale_dwell must be positive when supplied")
        if (
            not math.isfinite(float(idle_upscale_margin))
            or not 0.0 < float(idle_upscale_margin) < 1.0
        ):
            raise ValueError("idle_upscale_margin must be between zero and one")
        supported = frozenset(supported_tiers)
        self._ladder = tuple(
            tier
            for tier in (
                ResolutionTier.EIGHTH,
                ResolutionTier.QUARTER,
                ResolutionTier.HALF,
                ResolutionTier.NATIVE,
            )
            if tier in supported
        )
        self._tier = (
            ResolutionTier.HALF
            if ResolutionTier.HALF in supported
            else max(supported, key=lambda item: item.downsample_factor)
        )
        self._pressure_samples = 0
        self._last_admitted_total = 0
        self._tier_changed_at_admitted = 0
        self._healthy_started_at_admitted = 0
        self._has_changed_tier = False
        self._target_views_per_s = float(target_views_per_s)
        self._idle_upscale_dwell = int(idle_upscale_dwell)
        self._idle_upscale_margin = float(idle_upscale_margin)
        self._idle_upscale_samples = 0
        self._last_idle_optimizer_steps = -1
        self._batch_size = int(batch_size)
        # Deliberately NOT tied to idle_upscale_dwell's default of 8 (issue
        # #1346): a real measured gpu_cost_ms sample is a single point of
        # direct evidence about the *current* tier's own cost, not a proxy
        # signal that benefits from confirmation across many ticks the way
        # frontier-age/queue-occupancy pressure does. Requiring eight
        # consecutive such samples before recovering was found to leave the
        # mapping ceiling stuck at its most recent downscale for the rest of
        # a real run, because ordinary tick-to-tick jitter rarely sustains a
        # margin for that long even while the device is genuinely idle.
        self._fresh_upscale_dwell = int(
            1 if fresh_upscale_dwell is None else fresh_upscale_dwell
        )
        self._fresh_margin_samples = 0
        self._last_fresh_margin_tick = -1
        # Mapping and replay are independent actuators in MappingWorkPlan.
        # Keep their hysteresis state independent too: an expensive replay-only
        # tick must not undo a healthy source-resolution recovery decision.
        self._idle_tier: ResolutionTier | None = None

    def _move(self, direction: int, admitted_total: int) -> bool:
        index = self._ladder.index(self._tier)
        target = max(0, min(len(self._ladder) - 1, index + direction))
        if target == index:
            return False
        self._tier = self._ladder[target]
        self._tier_changed_at_admitted = admitted_total
        self._healthy_started_at_admitted = admitted_total
        self._has_changed_tier = True
        return True

    def _move_to_top(self, admitted_total: int) -> bool:
        """Jump the ceiling directly to the highest supported tier.

        Used for the measured-margin recovery path (issue #1346): a single
        real transaction's measured cost is real evidence the *current* tier
        has headroom, so there is no reason to spend one dwell window per
        ladder rung on the way back up.  This only ever widens the *ceiling*
        `TierCostEstimator.select()` is allowed to consider -- the actual
        per-transaction tier is still clamped by that estimator's own
        measured, budget-fitting selection (`work_budget.py::
        _resolve_transaction_resolution`), so an over-eager ceiling cannot by
        itself cause a transaction to exceed its real measured budget.
        """

        top = self._ladder[-1]
        if self._tier is top:
            return False
        self._tier = top
        self._tier_changed_at_admitted = admitted_total
        self._healthy_started_at_admitted = admitted_total
        self._has_changed_tier = True
        return True

    def _idle_resolution_tier(self) -> ResolutionTier:
        if self._idle_tier is None:
            self._idle_tier = self._tier
        return self._idle_tier

    def _move_idle(self, direction: int) -> bool:
        tier = self._idle_resolution_tier()
        index = self._ladder.index(tier)
        target = max(0, min(len(self._ladder) - 1, index + direction))
        if target == index:
            return False
        self._idle_tier = self._ladder[target]
        return True

    def propose(self, snapshot: ControlSnapshot) -> ControlProposal:
        evidence = {item.name: item.value for item in snapshot.evidence}
        # Note: "observation_value" evidence is intentionally no longer read
        # here (issue #1346) -- see the fallback-path comment below for why.
        admitted_total = int(evidence.get("admitted_observations_total", 0))
        if admitted_total < self._last_admitted_total:
            raise ValueError("admitted_observations_total must be monotonic")
        self._last_admitted_total = admitted_total
        if snapshot.freshness_mode is FreshnessMode.CRITICAL:
            changed = self._tier is not self._ladder[0]
            self._tier = self._ladder[0]
            self._pressure_samples = 0
            self._idle_upscale_samples = 0
            self._healthy_started_at_admitted = admitted_total
            if changed:
                self._tier_changed_at_admitted = admitted_total
                self._has_changed_tier = True
            reason = "critical freshness; immediate downscale"
        # FreshnessMode is the canonical, hysteretic pressure decision.  Raw
        # queue occupancy is not pressure by itself (one slot is the normal
        # producer/consumer hand-off), while frontier age follows intentional
        # keyframe cadence and commit rate does not count optimizer updates.
        # Reinterpreting those facts here created a second pressure owner and
        # forced healthy 30 Hz streams permanently to the eighth tier.
        elif snapshot.freshness_mode is FreshnessMode.CATCH_UP:
            self._healthy_started_at_admitted = admitted_total
            self._idle_upscale_samples = 0
            dwell_complete = (
                not self._has_changed_tier
                or admitted_total - self._tier_changed_at_admitted >= 4
            )
            self._pressure_samples = self._pressure_samples + 1 if dwell_complete else 0
            if self._pressure_samples >= 2:
                changed = self._move(-1, admitted_total)
                self._pressure_samples = 0
                reason = (
                    "two consecutive pressure samples; downscale"
                    if changed
                    else "pressure at minimum supported tier"
                )
            else:
                reason = (
                    "pressure sample pending confirmation"
                    if dwell_complete
                    else "pressure held by four-admitted-observation dwell"
                )
        else:
            self._pressure_samples = 0
            gpu_cost_ms = evidence.get("gpu_cost_ms")
            fresh_margin_observed = (
                gpu_cost_ms is not None
                and snapshot.control_tick_epoch > self._last_fresh_margin_tick
            )
            if fresh_margin_observed:
                if not math.isfinite(float(gpu_cost_ms)) or float(gpu_cost_ms) < 0.0:
                    raise ValueError("gpu_cost_ms must be finite and non-negative")
                self._last_fresh_margin_tick = snapshot.control_tick_epoch
                target_ms = 1000.0 * self._batch_size / self._target_views_per_s
                if float(gpu_cost_ms) <= target_ms * self._idle_upscale_margin:
                    self._fresh_margin_samples += 1
                    if self._fresh_margin_samples >= self._fresh_upscale_dwell:
                        # Immediate, opportunistic recovery to the highest
                        # supported ceiling (issue #1346): the measured-cost
                        # tier selector downstream (TierCostEstimator.select)
                        # is the real budget-fitting decision, so widening
                        # this ceiling in one shot costs nothing when there
                        # is no margin for it and unblocks native the moment
                        # there genuinely is.
                        changed = self._move_to_top(admitted_total)
                        self._fresh_margin_samples = 0
                        reason = (
                            "measured fresh mapping margin; immediate ceiling recovery"
                            if changed
                            else "measured fresh mapping margin at maximum tier"
                        )
                        return ControlProposal(
                            "ResolutionPolicyProvider", "resolution_tier", self._tier.value, reason
                        )
                else:
                    self._fresh_margin_samples = 0
            # Fallback path only: reached when no gpu_cost_ms evidence has
            # been observed this tick at all (fresh_margin_observed False),
            # so there is no direct cost measurement to recover from.
            # Previously this ALSO required observation_value >= 0.75 to
            # reach native specifically, on top of the eight-admission dwell
            # every other rung already needed. That extra gate (issue #1346)
            # is removed: it is not needed as a safety net -- the same
            # measured, budget-fitting `TierCostEstimator.select()` downstream
            # still clamps the actually-used tier -- and in practice it was
            # the second half of what left a mapping ceiling stuck below
            # native indefinitely once observation_value happened to be low
            # at the moment eight healthy admissions accrued.
            healthy_admissions = admitted_total - self._healthy_started_at_admitted
            if healthy_admissions >= 8:
                changed = self._move(1, admitted_total)
                reason = (
                    "eight healthy admitted observations; upscale"
                    if changed
                    else "healthy at maximum supported tier"
                )
            else:
                reason = "healthy observation dwell"
        if len(self._ladder) == 1 and self._tier is not ResolutionTier.HALF:
            reason += "; constrained by materialization capability"
        return ControlProposal(
            "ResolutionPolicyProvider", "resolution_tier", self._tier.value, reason
        )

    def propose_idle(
        self,
        snapshot: ControlSnapshot,
        *,
        optimizer_steps_completed: int,
        mapping_cost_ms: float | None,
        batch_size: int,
        realtime_constraints: bool = True,
    ) -> ControlProposal:
        """Propose replay resolution at the zero-frame optimizer barrier.

        A discrete tier ladder with asymmetric hysteresis is intentional here:
        overload falls back immediately, while higher resolution requires a
        sustained measured wall-cost margin. A continuous PID would integrate
        across tier discontinuities and delayed feedback, inviting oscillation.
        """

        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        if isinstance(optimizer_steps_completed, bool) or int(optimizer_steps_completed) < 0:
            raise ValueError("optimizer_steps_completed must be non-negative")
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(realtime_constraints) is not bool:
            raise TypeError("realtime_constraints must be bool")
        optimizer_steps_completed = int(optimizer_steps_completed)
        if optimizer_steps_completed == self._last_idle_optimizer_steps:
            return ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                self._idle_resolution_tier().value,
                "duplicate idle optimizer epoch; tier held",
            )
        if optimizer_steps_completed < self._last_idle_optimizer_steps:
            raise ValueError("idle optimizer steps must be monotonic")
        self._last_idle_optimizer_steps = optimizer_steps_completed

        if not realtime_constraints:
            self._idle_tier = self._ladder[-1]
            self._idle_upscale_samples = 0
            return ControlProposal(
                "ResolutionPolicyProvider",
                "replay_resolution_tier",
                self._idle_tier.value,
                "post-refinement has no realtime constraint; native replay",
            )

        self._idle_resolution_tier()
        target_ms = 1000.0 * int(batch_size) / self._target_views_per_s
        if mapping_cost_ms is None:
            self._idle_upscale_samples = 0
            reason = "idle mapping cost unavailable; tier held"
        elif not math.isfinite(float(mapping_cost_ms)) or float(mapping_cost_ms) < 0.0:
            raise ValueError("mapping_cost_ms must be finite and non-negative")
        elif float(mapping_cost_ms) > target_ms:
            changed = self._move_idle(-1)
            self._idle_upscale_samples = 0
            reason = (
                f"idle mapping cost {float(mapping_cost_ms):.3f}ms exceeds "
                f"{target_ms:.3f}ms budget; immediate downscale"
                if changed
                else "idle mapping cost exceeds budget at minimum tier"
            )
        elif float(mapping_cost_ms) <= target_ms * self._idle_upscale_margin:
            self._idle_upscale_samples += 1
            if self._idle_upscale_samples >= self._idle_upscale_dwell:
                changed = self._move_idle(1)
                self._idle_upscale_samples = 0
                reason = (
                    "sustained idle mapping margin; upscale"
                    if changed
                    else "sustained idle mapping margin at maximum tier"
                )
            else:
                reason = (
                    "idle mapping margin pending hysteresis "
                    f"{self._idle_upscale_samples}/{self._idle_upscale_dwell}"
                )
        else:
            self._idle_upscale_samples = 0
            reason = "idle mapping cost inside hold band; tier held"
        return ControlProposal(
            "ResolutionPolicyProvider",
            "replay_resolution_tier",
            self._idle_resolution_tier().value,
            reason,
        )


__all__ = ["ResolutionPolicyProvider"]
