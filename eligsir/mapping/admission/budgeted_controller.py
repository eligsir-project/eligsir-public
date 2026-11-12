"""Single-owner budgeted observation scheduler for #802.

This composes eligibility evidence, optional read-only Gaussian probing and one
bounded cohort.  It returns COMMIT/DEFER/DROP decisions but does not itself
mutate the map or optimization state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Generic, TypeVar

from eligsir.mapping.admission.cohort import (
    CohortAction,
    CohortDecision,
    PendingCandidate,
    PendingKeyframeCohort,
)
from eligsir.mapping.candidates.prefilter import (
    CandidatePrefilterResult,
    PrefilterDisposition,
)
from eligsir.mapping.evidence.canonical import FrameEvidence

T = TypeVar("T")

RANK_POLICIES = frozenset({"combined", "motion", "coverage", "residual", "random"})


@dataclass(frozen=True, slots=True)
class AdmissionCandidate(Generic[T]):
    """Payload paired with canonical prefilter and optional probe evidence."""

    payload: T
    prefilter: CandidatePrefilterResult
    probe: FrameEvidence | None = None


class BudgetedAdmissionController(Generic[T]):
    """One final pre-commit decision owner under a fixed K budget."""

    def __init__(
        self,
        *,
        max_candidates: int,
        max_source_age: int,
        commit_budget: int,
        rank_key: Callable[[AdmissionCandidate[T]], tuple] | None = None,
        rank_policy: str = "combined",
        rank_seed: int = 0,
    ) -> None:
        self.cohort = PendingKeyframeCohort[AdmissionCandidate[T]](
            max_candidates=max_candidates,
            max_source_age=max_source_age,
            commit_budget=commit_budget,
        )
        self.rank_seed = int(rank_seed)
        if rank_key is not None:
            self.rank_policy = "custom"
            self.rank_key = rank_key
        else:
            self.rank_policy = str(rank_policy)
            if self.rank_policy not in RANK_POLICIES:
                raise ValueError(
                    f"unsupported admission rank policy {self.rank_policy!r}; "
                    f"expected one of {sorted(RANK_POLICIES)}"
                )
            self.rank_key = self._rank_key_for_policy(self.rank_policy)

    @staticmethod
    def _scalar(frame: FrameEvidence | None, name: str, default: float) -> float:
        if frame is None:
            return default
        value = getattr(frame.probe, name).value
        return default if value is None else float(value)

    @staticmethod
    def _source_scalar(frame: FrameEvidence | None, name: str, default: float) -> float:
        if frame is None:
            return default
        value = getattr(frame.source, name).value
        return default if value is None else float(value)

    @staticmethod
    def _escalated(item: AdmissionCandidate[T]) -> int:
        return int(item.prefilter.disposition is PrefilterDisposition.ESCALATE)

    @staticmethod
    def _source_ordinal(item: AdmissionCandidate[T]) -> int:
        return int(item.prefilter.evidence.identity.source_ordinal)

    @classmethod
    def _default_rank_key(cls, item: AdmissionCandidate[T]) -> tuple:
        # Interpretable, deterministic initial ranking: escalation first, then
        # unexplained map fraction and residual, finally source order.  This is
        # a baseline policy, not a learned universal utility score.
        unexplained = cls._scalar(item.probe, "unexplained_pixel_fraction", -1.0)
        residual = cls._scalar(item.probe, "rgb_l1_mean", -1.0)
        return (-cls._escalated(item), -unexplained, -residual, cls._source_ordinal(item))

    @classmethod
    def _motion_rank_key(cls, item: AdmissionCandidate[T]) -> tuple:
        translation = cls._source_scalar(item.probe, "translation_delta_m", -1.0)
        rotation = cls._source_scalar(item.probe, "rotation_delta_deg", -1.0)
        return (-cls._escalated(item), -translation, -rotation, cls._source_ordinal(item))

    @classmethod
    def _coverage_rank_key(cls, item: AdmissionCandidate[T]) -> tuple:
        unexplained = cls._scalar(item.probe, "unexplained_pixel_fraction", -1.0)
        return (-cls._escalated(item), -unexplained, cls._source_ordinal(item))

    @classmethod
    def _residual_rank_key(cls, item: AdmissionCandidate[T]) -> tuple:
        residual = cls._scalar(item.probe, "rgb_l1_mean", -1.0)
        return (-cls._escalated(item), -residual, cls._source_ordinal(item))

    def _random_rank_key(self, item: AdmissionCandidate[T]) -> tuple:
        identity = item.prefilter.evidence.identity
        stable_identity = (
            f"{self.rank_seed}:{identity.source_id}:{identity.source_ordinal}:"
            f"{identity.timestamp_ns}"
        ).encode("utf-8")
        digest = hashlib.sha256(stable_identity).digest()
        return (-self._escalated(item), digest, self._source_ordinal(item))

    def _rank_key_for_policy(self, policy: str) -> Callable[[AdmissionCandidate[T]], tuple]:
        return {
            "combined": self._default_rank_key,
            "motion": self._motion_rank_key,
            "coverage": self._coverage_rank_key,
            "residual": self._residual_rank_key,
            "random": self._random_rank_key,
        }[policy]

    def submit(self, candidate: AdmissionCandidate[T]) -> tuple[CohortDecision[AdmissionCandidate[T]], ...]:
        """Submit one source-ordered candidate and return immediate drops."""

        ident = candidate.prefilter.evidence.identity
        eligible = candidate.prefilter.disposition is not PrefilterDisposition.INELIGIBLE
        return self.cohort.add(PendingCandidate(
            source_ordinal=int(ident.source_ordinal),
            timestamp_ns=int(ident.timestamp_ns),
            payload=candidate,
            eligible=eligible,
            escalated=candidate.prefilter.disposition is PrefilterDisposition.ESCALATE,
        ))

    def should_flush(self) -> bool:
        """Return whether capacity or escalation requires a cohort decision."""

        return self.cohort.should_flush()

    def flush(self) -> tuple[CohortDecision[AdmissionCandidate[T]], ...]:
        """Rank the pending cohort and enforce the configured commit budget."""

        def ranker(pending):
            """Return cohort indices in deterministic policy-rank order."""

            return sorted(range(len(pending)), key=lambda idx: self.rank_key(pending[idx].payload))
        return self.cohort.flush(ranker=ranker)

    @staticmethod
    def committed(decisions) -> tuple[AdmissionCandidate[T], ...]:
        """Extract candidates whose cohort decision authorizes commitment."""

        return tuple(d.candidate.payload for d in decisions if d.action is CohortAction.COMMIT)
