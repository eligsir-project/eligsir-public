"""Shadow-only, nonblocking resolution of policy ``EvidenceDemand`` values.

This module deliberately turns a version-current demand into a *bounded
shortlist intent*, never a chosen replay frame.  The existing supervision
scheduler retains ownership of replay/view/modality/tier selection and of any
source preparation.  Frame/Gaussian UID incidence is only a historical
shortlisting hint: it neither asserts current visibility nor proves that a
frame can resolve a demand.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import chain
import math
from numbers import Real
from typing import Iterable

from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandModality,
)
from eligsir.mapping.observation_db import BufferStage, ModalityEligibility


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceDemandCurrentVersions:
    """Versions the scheduler observed immediately before resolving demand."""

    map: int
    topology: int
    coordinate: int
    schema: int
    optimizer: int
    surface: int
    policy: int

    def __post_init__(self) -> None:
        for name in (
            "map",
            "topology",
            "coordinate",
            "schema",
            "optimizer",
            "surface",
            "policy",
        ):
            _nonnegative_int(name, getattr(self, name))

    def matches(self, demand: EvidenceDemand) -> bool:
        return demand.snapshot_versions == {
            "map": self.map,
            "topology": self.topology,
            "coordinate": self.coordinate,
            "schema": self.schema,
            "optimizer": self.optimizer,
            "surface": self.surface,
            "policy": self.policy,
        }


@dataclass(frozen=True, slots=True)
class EvidenceDemandCandidate:
    """Pixel-free candidate metadata supplied by an existing scheduler index."""

    frame_id: int
    buffer_stage: BufferStage
    modality_eligibility: ModalityEligibility
    observed_target_ids: tuple[str, ...] = ()
    resident: bool = False
    materializable: bool = True

    def __post_init__(self) -> None:
        _nonnegative_int("frame_id", self.frame_id)
        if not isinstance(self.buffer_stage, BufferStage):
            raise TypeError("buffer_stage must be a BufferStage")
        if not isinstance(self.modality_eligibility, ModalityEligibility):
            raise TypeError("modality_eligibility must be ModalityEligibility")
        if not isinstance(self.observed_target_ids, tuple):
            raise TypeError("observed_target_ids must be a tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.observed_target_ids):
            raise ValueError("observed_target_ids must contain non-empty strings")
        if not isinstance(self.resident, bool) or not isinstance(self.materializable, bool):
            raise TypeError("resident and materializable must be bool")
        if self.resident and not self.materializable:
            raise ValueError("resident candidate must be materializable")


@dataclass(frozen=True, slots=True)
class EvidenceDemandShortlistIntent:
    """A scheduler-owned shortlist request, intentionally not a selection."""

    demand: EvidenceDemand
    candidate_frame_ids: tuple[int, ...]
    incidence_is_shortlist_evidence: bool = True
    resident_fallback_used: bool = False
    selected_frame_id: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.demand, EvidenceDemand):
            raise TypeError("demand must be an EvidenceDemand")
        if not self.candidate_frame_ids:
            raise ValueError("shortlist intent requires at least one candidate")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.candidate_frame_ids):
            raise ValueError("candidate_frame_ids must contain nonnegative integers")
        if len(self.candidate_frame_ids) != len(set(self.candidate_frame_ids)):
            raise ValueError("candidate_frame_ids must be distinct")
        if self.selected_frame_id is not None:
            raise ValueError("demand resolution cannot select a frame")


@dataclass(slots=True)
class EvidenceDemandResolutionTelemetry:
    """Small CPU counters for shadow-mode observability."""

    queue_overflow_drops: int = 0
    expired_drops: int = 0
    stale_version_drops: int = 0
    deferred_no_candidate: int = 0
    unready_cold_skips: int = 0
    resolved_intents: int = 0


class EvidenceDemandResolver:
    """A bounded SPSC-style handoff from policy demand to scheduler intent.

    ``enqueue`` never blocks.  ``resolve_available`` inspects only the bounded
    candidate views supplied by the caller and performs no source paging,
    synchronization, render, or device operation.  It is deliberately a
    shadow contract: existing scheduler decisions are unchanged until a later
    owner explicitly consumes its intents.
    """

    def __init__(
        self,
        *,
        max_pending_demands: int,
        max_candidates_per_intent: int,
        max_candidates_examined: int = 128,
    ) -> None:
        self.max_pending_demands = _nonnegative_int(
            "max_pending_demands", max_pending_demands
        )
        self.max_candidates_per_intent = _nonnegative_int(
            "max_candidates_per_intent", max_candidates_per_intent
        )
        self.max_candidates_examined = _nonnegative_int(
            "max_candidates_examined", max_candidates_examined
        )
        if self.max_pending_demands == 0 or self.max_candidates_per_intent == 0:
            raise ValueError("resolver bounds must be positive")
        if self.max_candidates_per_intent > self.max_candidates_examined:
            raise ValueError("max_candidates_per_intent exceeds examined bound")
        self._pending: deque[EvidenceDemand] = deque()
        self.telemetry = EvidenceDemandResolutionTelemetry()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def enqueue(self, demand: EvidenceDemand) -> bool:
        """Try to enqueue one asynchronous demand without waiting or paging."""
        if not isinstance(demand, EvidenceDemand):
            raise TypeError("demand must be an EvidenceDemand")
        if len(self._pending) >= self.max_pending_demands:
            self.telemetry.queue_overflow_drops += 1
            return False
        self._pending.append(demand)
        return True

    @staticmethod
    def _matches_modality(
        demand: EvidenceDemand, candidate: EvidenceDemandCandidate
    ) -> bool:
        allowed = candidate.modality_eligibility
        requested = demand.requested_modality
        if requested is EvidenceDemandModality.RGB:
            return allowed.rgb_supervision
        if requested is EvidenceDemandModality.DEPTH:
            return allowed.depth_supervision
        if requested is EvidenceDemandModality.RGB_DEPTH:
            return allowed.rgb_supervision and allowed.depth_supervision
        return allowed.rgb_supervision or allowed.depth_supervision

    @classmethod
    def _eligible(
        cls,
        demand: EvidenceDemand,
        candidate: EvidenceDemandCandidate,
        *,
        require_incidence: bool = True,
    ) -> bool:
        if candidate.buffer_stage is BufferStage.HOLDOUT_TEST:
            return False
        if not candidate.materializable or not cls._matches_modality(demand, candidate):
            return False
        if require_incidence and demand.target_ids and not set(demand.target_ids).intersection(
            candidate.observed_target_ids
        ):
            return False
        return True

    def _bounded_unique_ids(
        self, candidates: Iterable[EvidenceDemandCandidate]
    ) -> list[int]:
        """Preserve scheduler order while preventing duplicate intent entries."""
        frame_ids: list[int] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate.frame_id in seen:
                continue
            seen.add(candidate.frame_id)
            frame_ids.append(candidate.frame_id)
            if len(frame_ids) == self.max_candidates_per_intent:
                break
        return frame_ids

    def resolve_available(
        self,
        *,
        current_versions: EvidenceDemandCurrentVersions,
        current_epoch: Real,
        candidates: Iterable[EvidenceDemandCandidate],
        resident_fallbacks: Iterable[EvidenceDemandCandidate] = (),
    ) -> tuple[EvidenceDemandShortlistIntent, ...]:
        """Consume queued demands into bounded, nonblocking shortlist intents."""
        if not isinstance(current_versions, EvidenceDemandCurrentVersions):
            raise TypeError("current_versions must be EvidenceDemandCurrentVersions")
        if (
            isinstance(current_epoch, bool)
            or not isinstance(current_epoch, Real)
            or not math.isfinite(current_epoch)
        ):
            raise ValueError("current_epoch must be a finite real number")
        candidates = tuple(candidates)
        fallbacks = tuple(resident_fallbacks)
        if len(candidates) > self.max_candidates_examined:
            raise ValueError("candidate input exceeds max_candidates_examined")
        if len(fallbacks) > self.max_candidates_examined:
            raise ValueError("resident fallback input exceeds max_candidates_examined")
        if any(not isinstance(item, EvidenceDemandCandidate) for item in candidates + fallbacks):
            raise TypeError("candidate inputs must be EvidenceDemandCandidate instances")

        intents: list[EvidenceDemandShortlistIntent] = []
        # A current demand without a usable candidate remains pending for a
        # later scheduler pass.  Snapshot the initial queue length so such a
        # deferral cannot be reprocessed indefinitely in this call.
        for _ in range(len(self._pending)):
            demand = self._pending.popleft()
            if current_epoch > demand.expiry_epoch:
                self.telemetry.expired_drops += 1
                continue
            if not current_versions.matches(demand):
                self.telemetry.stale_version_drops += 1
                continue
            # A ready resident fallback is intentionally preferred to a COLD
            # candidate: demand must never force a GPU wait for source paging.
            # Its lack of historical UID incidence is allowed because that
            # incidence is a shortlist hint, not a truth claim.
            shortlist = self._bounded_unique_ids(
                candidate
                for candidate in fallbacks
                if candidate.resident
                and self._eligible(demand, candidate, require_incidence=False)
            )
            fallback_used = bool(shortlist)
            if fallback_used:
                self.telemetry.unready_cold_skips += sum(
                    candidate.buffer_stage is not BufferStage.HOLDOUT_TEST
                    and not candidate.resident
                    and self._matches_modality(demand, candidate)
                    and (
                        not demand.target_ids
                        or bool(
                            set(demand.target_ids).intersection(
                                candidate.observed_target_ids
                            )
                        )
                    )
                    for candidate in candidates
                )
            else:
                ordinary_eligible = tuple(
                    candidate
                    for candidate in candidates
                    if self._eligible(demand, candidate)
                )
                shortlist = self._bounded_unique_ids(
                    chain(
                        (
                            candidate
                            for candidate in ordinary_eligible
                            if candidate.resident
                        ),
                        (
                            candidate
                            for candidate in ordinary_eligible
                            if not candidate.resident
                        ),
                    )
                )
            if not shortlist:
                self._pending.append(demand)
                self.telemetry.deferred_no_candidate += 1
                continue
            intents.append(
                EvidenceDemandShortlistIntent(
                    demand=demand,
                    candidate_frame_ids=tuple(shortlist),
                    resident_fallback_used=fallback_used,
                )
            )
            self.telemetry.resolved_intents += 1
        return tuple(intents)
