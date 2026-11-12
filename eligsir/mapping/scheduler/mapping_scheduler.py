"""MappingScheduler — sole owner of per-frame mapping budget decisions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from eligsir.contracts.types import TrackingResult
from eligsir.mapping.scheduler.spf_resolution import (
    RESOLVED_BY_COVERAGE,
    RESOLVED_BY_MIN_SPF_FLOOR,
    RESOLVED_BY_REQUEST,
)


@dataclass
class SchedulerStats:
    """Resolved scheduling parameters reported at startup and in telemetry."""

    effective_spf: int
    keyframe_window: int
    replay_ratio: float
    full_coverage: bool
    reservoir_stride: int = 1


@dataclass
class MappingDecision:
    """Per-frame mapping work and mutation permissions."""

    do_mapping: bool
    steps: int
    allow_insertion: bool
    allow_growth: bool
    freeze_persistent: bool
    keyframe_indices: list
    replay_indices: list
    confidence: float = 1.0
    assimilation_tier: int | None = None
    assimilation_reason: str | None = None
    historical_repair_extra_steps: int = 0


class MappingScheduler:
    """Decide mapping parameters for each incoming frame.

    Tracking confidence is the highest-priority safety gate. On normal mapping,
    optional assimilation control may only suppress insertion/growth and add a
    bounded number of optimizer steps. The historical-repair policy may add its
    own separately bounded consolidation budget. Neither may bypass tracking
    safety, source ordering or keyframe budgets.
    """

    def __init__(
        self,
        steps_per_frame: int,
        n_frames: int,
        iterations: int,
        enforce_full_coverage: bool = True,
        keyframe_window: int = 120,
        keyframe_coverage: float = 0.6,
        global_replay_ratio: float = 0.3,
        reservoir_stride: int = 1,
        tracking_min_conf_persistent: float = 0.7,
        tracking_min_conf_provisional: float = 0.4,
        ingest_front_load_ratio: float = 1.0,
        min_steps_per_frame: int = 0,
    ) -> None:
        self._front_load_ratio = max(1e-3, min(1.0, float(ingest_front_load_ratio)))
        self._n_frames_capped: int | None = None
        self._requested_spf = int(steps_per_frame)
        self._min_steps_per_frame = int(min_steps_per_frame)
        self._coverage_required_spf: int | None = None
        self._spf_resolved_by = RESOLVED_BY_REQUEST
        if enforce_full_coverage and n_frames > 0 and iterations > 0:
            ingest_budget = math.floor(iterations * self._front_load_ratio)
            max_spf = math.floor(ingest_budget / n_frames)
            effective_spf_raw = min(steps_per_frame, max(1, max_spf))
            self._coverage_required_spf = int(effective_spf_raw)
            if min_steps_per_frame > 0 and effective_spf_raw < min_steps_per_frame:
                n_frames_eff = max(1, math.floor(ingest_budget / min_steps_per_frame))
                print(
                    f"[MappingScheduler] min_spf floor: enforcement would give spf={effective_spf_raw} "
                    f"< min={min_steps_per_frame}; capping admitted frames {n_frames}->{n_frames_eff} "
                    f"(remaining {ingest_budget - n_frames_eff * min_steps_per_frame} optimization steps = replay refinement)"
                )
                self._effective_spf = min_steps_per_frame
                self._n_frames_capped = n_frames_eff
                self._spf_resolved_by = RESOLVED_BY_MIN_SPF_FLOOR
            else:
                self._effective_spf = effective_spf_raw
                self._spf_resolved_by = (
                    RESOLVED_BY_COVERAGE if effective_spf_raw < steps_per_frame
                    else RESOLVED_BY_REQUEST
                )
            self._full_coverage = True
        else:
            self._effective_spf = steps_per_frame
            self._full_coverage = False

        self._keyframe_window = keyframe_window
        self._keyframe_coverage = keyframe_coverage
        self._global_replay_ratio = global_replay_ratio
        self._reservoir_stride = reservoir_stride
        self._tracking_min_conf_persistent = tracking_min_conf_persistent
        self._tracking_min_conf_provisional = tracking_min_conf_provisional
        self._last_assimilation_decision = None

    @property
    def n_frames_capped(self) -> "int | None":
        """Return the admission cap imposed by the minimum-SPF floor."""

        return self._n_frames_capped

    @property
    def effective_spf(self) -> int:
        """Return the resolved optimizer steps assigned to each mapped frame."""

        return self._effective_spf

    @property
    def requested_spf(self) -> int:
        """Return the caller-requested steps per frame before resolution."""

        return self._requested_spf

    @property
    def min_steps_per_frame(self) -> int:
        """Return the configured minimum steps-per-frame floor."""

        return self._min_steps_per_frame

    @property
    def coverage_required_spf(self) -> "int | None":
        """Return the SPF compatible with the full-coverage ingest budget."""

        return self._coverage_required_spf

    @property
    def spf_resolved_by(self) -> str:
        """Return the reason code that selected the effective SPF."""

        return self._spf_resolved_by

    @property
    def last_assimilation_decision(self):
        """Return the most recent optional assimilation-control decision."""

        return self._last_assimilation_decision

    def log_startup(self) -> None:
        """Print the resolved scheduling contract once at startup."""

        s = self.stats
        if s.full_coverage:
            print(
                f"[MappingScheduler] Full-coverage enforcement active: "
                f"effective_spf={s.effective_spf} keyframe_window={s.keyframe_window} "
                f"replay_ratio={s.replay_ratio:.3f} reservoir_stride={s.reservoir_stride}",
                flush=True,
            )
        else:
            print(
                f"[MappingScheduler] steps_per_frame={s.effective_spf} (no coverage enforcement) "
                f"keyframe_window={s.keyframe_window} replay_ratio={s.replay_ratio:.3f}",
                flush=True,
            )

    @property
    def stats(self) -> SchedulerStats:
        """Return an immutable snapshot of resolved scheduler parameters."""

        return SchedulerStats(
            effective_spf=self._effective_spf,
            keyframe_window=self._keyframe_window,
            replay_ratio=self._global_replay_ratio,
            full_coverage=self._full_coverage,
            reservoir_stride=self._reservoir_stride,
        )

    def _apply_quality_control(
        self,
        decision: MappingDecision,
        *,
        topology_version: int = 0,
    ) -> MappingDecision:
        from eligsir.utils.assimilation_pressure import active_control_decision
        from eligsir.utils.partition_control import consume_historical_repair_extra_steps

        control = active_control_decision(
            decision.steps,
            current_topology_version=topology_version,
        )
        self._last_assimilation_decision = control
        steps = int(decision.steps)
        allow_insertion = bool(decision.allow_insertion)
        allow_growth = bool(decision.allow_growth)
        tier = None
        reason = None
        if control is not None:
            steps = max(0, int(control.optimizer_steps))
            allow_insertion = bool(allow_insertion and control.allow_insertion)
            allow_growth = bool(allow_growth and control.allow_growth)
            tier = int(control.tier)
            reason = control.reason

        repair_extra = max(0, int(consume_historical_repair_extra_steps()))
        steps += repair_extra
        return MappingDecision(
            do_mapping=decision.do_mapping,
            steps=steps,
            allow_insertion=allow_insertion,
            allow_growth=allow_growth,
            freeze_persistent=decision.freeze_persistent,
            keyframe_indices=decision.keyframe_indices,
            replay_indices=decision.replay_indices,
            confidence=decision.confidence,
            assimilation_tier=tier,
            assimilation_reason=reason,
            historical_repair_extra_steps=repair_extra,
        )

    def decide(
        self,
        tracking_result: Optional[TrackingResult],
        keyframe_indices: list,
        replay_indices: list,
        *,
        topology_version: int = 0,
    ) -> MappingDecision:
        """Resolve mapping work from tracking safety and bounded controls.

        Args:
            tracking_result: camera pose tracking result
            keyframe_indices: candidate keyframe indices
            replay_indices: candidate replay indices
            topology_version: current map topology version (for assimilation control staleness detection)
        """

        conf = tracking_result.confidence if tracking_result is not None else 1.0
        valid = tracking_result.valid if tracking_result is not None else True

        if not valid or conf < self._tracking_min_conf_provisional:
            self._last_assimilation_decision = None
            return MappingDecision(
                do_mapping=False,
                steps=0,
                allow_insertion=False,
                allow_growth=False,
                freeze_persistent=True,
                keyframe_indices=[],
                replay_indices=[],
                confidence=conf,
            )

        if conf < self._tracking_min_conf_persistent:
            self._last_assimilation_decision = None
            reduced = max(1, self._effective_spf // 2)
            return MappingDecision(
                do_mapping=True,
                steps=reduced,
                allow_insertion=False,
                allow_growth=False,
                freeze_persistent=True,
                keyframe_indices=keyframe_indices,
                replay_indices=[],
                confidence=conf,
            )

        return self._apply_quality_control(
            MappingDecision(
                do_mapping=True,
                steps=self._effective_spf,
                allow_insertion=True,
                allow_growth=True,
                freeze_persistent=False,
                keyframe_indices=keyframe_indices,
                replay_indices=replay_indices,
                confidence=conf,
            ),
            topology_version=topology_version,
        )
