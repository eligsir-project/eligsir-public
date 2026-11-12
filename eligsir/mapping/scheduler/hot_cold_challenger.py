"""HOT-incumbent vs COLD-challenger decisions for issue #1322.

This torch-free engine keeps per-modality EMA evidence, bounded incumbency
leases, and hysteretic promotion decisions.  Runtime feedback records measured
train-only modality evidence and uses successful decisions to update the
challenger's existing durable replay priority before bounded READY turnover.
The engine does not own residency or source materialization.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Dict, Optional

from eligsir.mapping.scheduler.historical_usefulness import (
    NEUTRAL_WEIGHT,
    EmaUsefulnessTracker,
)

#: Closed vocabulary of independently-tracked usefulness modalities (design
#: doc section 6/9: "do not collapse into one raw-loss scalar").
MODALITY_RGB = "rgb"
MODALITY_DEPTH = "depth"
MODALITY_INSERTION = "insertion"
_VALID_MODALITIES = (MODALITY_RGB, MODALITY_DEPTH, MODALITY_INSERTION)

#: Default HOT incumbency lease length, in the caller's own iteration/tick
#: counter. Named rather than inlined so it is a single tunable; matches
#: this project's convention of small, documented, unit-tested defaults.
DEFAULT_LEASE_TICKS: int = 200

#: Default promotion margin: a challenger must exceed the incumbent's
#: trusted smoothed weight by at least this much (same units as
#: ``EmaUsefulnessTracker`` weights, i.e. ``[NEUTRAL_WEIGHT, MAX_WEIGHT]``)
#: before it is allowed to displace it. Deliberately nonzero -- a margin of
#: 0 would let single-tick numerical noise between two near-tied trusted
#: estimates flip status every cycle.
DEFAULT_PROMOTION_MARGIN: float = 0.25

#: Decision reasons, kept as a small closed vocabulary (mirrors
#: ``hot_tier_cache.py``'s ``EVICT_*`` convention) so telemetry can
#: distinguish *why* a cycle produced no promotion.
REASON_INCUMBENT_PROTECTED = "incumbent_protected"
REASON_UNTRUSTED = "untrusted_evidence"
REASON_MARGIN_NOT_MET = "margin_not_met"
REASON_PROMOTED = "promoted"
REASON_SAME_FRAME = "same_frame"


@dataclass(frozen=True, slots=True)
class ChallengeDecision:
    """Result of one incumbent-vs-challenger evaluation."""

    incumbent_id: int
    challenger_id: int
    modality: str
    promote: bool
    reason: str
    incumbent_weight: float
    challenger_weight: float


@dataclass(frozen=True, slots=True)
class HotIncumbentLease:
    """One frame's protected-residency window, for inspection/telemetry."""

    frame_id: int
    granted_at: int
    protected_until: int


def _validate_modality(modality: str) -> str:
    modality = str(modality)
    if modality not in _VALID_MODALITIES:
        raise ValueError(
            f"modality must be one of {_VALID_MODALITIES}, got {modality!r}"
        )
    return modality


class HotColdChallengerArena:
    """Bounded, hysteretic HOT-incumbent vs COLD-challenger decision engine.

    Fully CPU-only and unit-testable without a rendering stack -- deliberately
    holds no reference to any camera, tensor, cache handle or GPU state, only
    ``frame_id -> lease`` and ``(frame_id, modality) -> EmaUsefulnessTracker``
    bookkeeping. Bounded memory via each tracker's own
    ``max_tracked_frames`` LRU eviction plus a bounded lease table (evicts the
    oldest-granted lease once over ``max_tracked_leases``), matching this
    project's standing convention of bounded-not-unbounded per-frame state.
    """

    def __init__(
        self,
        *,
        lease_ticks: int = DEFAULT_LEASE_TICKS,
        promotion_margin: float = DEFAULT_PROMOTION_MARGIN,
        max_tracked_leases: int = 4096,
    ) -> None:
        if isinstance(lease_ticks, bool) or int(lease_ticks) < 0:
            raise ValueError("lease_ticks must be a non-negative integer")
        if (
            isinstance(promotion_margin, bool)
            or not math.isfinite(float(promotion_margin))
            or float(promotion_margin) < 0.0
        ):
            raise ValueError("promotion_margin must be a finite, non-negative number")
        if isinstance(max_tracked_leases, bool) or int(max_tracked_leases) < 1:
            raise ValueError("max_tracked_leases must be a positive integer")
        self._lease_ticks = int(lease_ticks)
        self._promotion_margin = float(promotion_margin)
        self._max_tracked_leases = int(max_tracked_leases)
        self._leases: "OrderedDict[int, HotIncumbentLease]" = OrderedDict()
        self._trackers: Dict[str, EmaUsefulnessTracker] = {
            modality: EmaUsefulnessTracker() for modality in _VALID_MODALITIES
        }
        # Telemetry (design doc section 12: "promotion/demotion counts").
        self.evaluations = 0
        self.promotions = 0
        self.demotions = 0
        self.decisions_by_reason: Dict[str, int] = {}

    # --------------------------------------------------------------- leases

    @property
    def lease_ticks(self) -> int:
        return self._lease_ticks

    def grant_lease(self, frame_id: int, iteration: int) -> HotIncumbentLease:
        """Grant (or refresh) a protected HOT incumbency lease for ``frame_id``.

        Real, unconditional production caller: every newly-admitted historical
        record (``MappingApplication._register_historical_record``) grants a
        lease the moment it becomes a HOT keyframe candidate -- this is what
        stops a mature historical pool from immediately out-competing genuinely
        new frontier evidence (design doc section 8.1).
        """

        frame_id = int(frame_id)
        iteration = int(iteration)
        lease = HotIncumbentLease(
            frame_id=frame_id,
            granted_at=iteration,
            protected_until=iteration + self._lease_ticks,
        )
        self._leases.pop(frame_id, None)
        self._leases[frame_id] = lease
        self._leases.move_to_end(frame_id)
        while len(self._leases) > self._max_tracked_leases:
            self._leases.popitem(last=False)
        return lease

    def is_protected(self, frame_id: int, iteration: int) -> bool:
        """Whether ``frame_id`` is still inside its incumbency lease window.

        An unknown frame (never granted a lease) is never protected -- the
        lease is an explicit grant, not an ambient default.
        """

        lease = self._leases.get(int(frame_id))
        if lease is None:
            return False
        return int(iteration) < lease.protected_until

    def lease_for(self, frame_id: int) -> Optional[HotIncumbentLease]:
        return self._leases.get(int(frame_id))

    def revoke_lease(self, frame_id: int) -> bool:
        """Drop lease state (e.g. on retirement from the historical index)."""

        return self._leases.pop(int(frame_id), None) is not None

    # ---------------------------------------------------- modality tracking

    def record_modality_observation(
        self, frame_id: int, modality: str, raw_weight: float
    ):
        """Fold one modality-specific usefulness observation into its tracker.

        Kept strictly separate per modality (design doc section 2/6): a frame
        that is excellent RGB supervision and poor depth supervision must be
        representable, not collapsed into one number.
        """

        modality = _validate_modality(modality)
        return self._trackers[modality].observe(frame_id, raw_weight)

    def modality_weight(
        self, frame_id: int, modality: str, *, default: float = NEUTRAL_WEIGHT
    ) -> float:
        modality = _validate_modality(modality)
        return self._trackers[modality].weight(frame_id, default=default)

    def modality_trusted(self, frame_id: int, modality: str) -> bool:
        """Whether ``frame_id`` has a trusted smoothed weight for ``modality``.

        ``EmaUsefulnessTracker.weight()`` has no direct "is this trusted"
        query keyed only by ``frame_id`` -- it either returns the trusted
        smoothed value or a caller-supplied ``default``. A NaN sentinel
        default is used here (rather than comparing against
        ``NEUTRAL_WEIGHT``) because a genuinely-neutral *trusted* observation
        is indistinguishable from "no evidence yet" if the default and the
        real neutral value were the same number.
        """

        modality = _validate_modality(modality)
        value = self._trackers[modality].weight(frame_id, default=math.nan)
        return not math.isnan(value)

    def forget_modality(self, frame_id: int, modality: Optional[str] = None) -> None:
        """Drop tracked modality state for a frame (e.g. on retirement)."""

        modalities = _VALID_MODALITIES if modality is None else (
            _validate_modality(modality),
        )
        for name in modalities:
            self._trackers[name].forget(frame_id)

    # ------------------------------------------------------------- decision

    def evaluate_challenge(
        self,
        incumbent_id: int,
        challenger_id: int,
        modality: str,
        iteration: int,
    ) -> ChallengeDecision:
        """Decide whether ``challenger_id`` should displace ``incumbent_id``.

        Hysteretic by construction (design doc section 8.4): promotion
        requires BOTH the incumbent's lease to have expired AND both frames'
        smoothed modality weight to be trusted (``EmaUsefulnessTracker``'s own
        ``min_observations`` gate -- a single noisy render cannot promote or
        demote) AND the challenger to exceed the incumbent by at least
        ``promotion_margin``. Missing/untrusted evidence never promotes: the
        safe default is "keep the incumbent."
        """

        incumbent_id = int(incumbent_id)
        challenger_id = int(challenger_id)
        modality = _validate_modality(modality)
        iteration = int(iteration)
        self.evaluations += 1

        def _record(reason: str, promote: bool, iw: float, cw: float) -> ChallengeDecision:
            self.decisions_by_reason[reason] = self.decisions_by_reason.get(reason, 0) + 1
            if promote:
                self.promotions += 1
                self.demotions += 1
            return ChallengeDecision(
                incumbent_id=incumbent_id,
                challenger_id=challenger_id,
                modality=modality,
                promote=promote,
                reason=reason,
                incumbent_weight=iw,
                challenger_weight=cw,
            )

        if incumbent_id == challenger_id:
            return _record(REASON_SAME_FRAME, False, NEUTRAL_WEIGHT, NEUTRAL_WEIGHT)

        if self.is_protected(incumbent_id, iteration):
            return _record(REASON_INCUMBENT_PROTECTED, False, NEUTRAL_WEIGHT, NEUTRAL_WEIGHT)

        tracker = self._trackers[modality]
        default_sentinel = math.nan
        incumbent_raw = tracker.weight(incumbent_id, default=default_sentinel)
        challenger_raw = tracker.weight(challenger_id, default=default_sentinel)
        incumbent_trusted = not math.isnan(incumbent_raw)
        challenger_trusted = not math.isnan(challenger_raw)
        incumbent_weight = incumbent_raw if incumbent_trusted else NEUTRAL_WEIGHT
        challenger_weight = challenger_raw if challenger_trusted else NEUTRAL_WEIGHT

        if not (incumbent_trusted and challenger_trusted):
            return _record(REASON_UNTRUSTED, False, incumbent_weight, challenger_weight)

        if challenger_weight <= incumbent_weight + self._promotion_margin:
            return _record(REASON_MARGIN_NOT_MET, False, incumbent_weight, challenger_weight)

        return _record(REASON_PROMOTED, True, incumbent_weight, challenger_weight)

    # ------------------------------------------------------------ telemetry

    def snapshot(self) -> dict:
        """Bounded telemetry view (design doc section 12)."""

        return {
            "active_leases": len(self._leases),
            "evaluations": int(self.evaluations),
            "promotions": int(self.promotions),
            "demotions": int(self.demotions),
            "decisions_by_reason": dict(self.decisions_by_reason),
            "tracked_frames_by_modality": {
                modality: len(tracker) for modality, tracker in self._trackers.items()
            },
        }


__all__ = [
    "DEFAULT_LEASE_TICKS",
    "DEFAULT_PROMOTION_MARGIN",
    "MODALITY_DEPTH",
    "MODALITY_INSERTION",
    "MODALITY_RGB",
    "REASON_INCUMBENT_PROTECTED",
    "REASON_MARGIN_NOT_MET",
    "REASON_PROMOTED",
    "REASON_SAME_FRAME",
    "REASON_UNTRUSTED",
    "ChallengeDecision",
    "HotColdChallengerArena",
    "HotIncumbentLease",
]
