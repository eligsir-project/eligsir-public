"""Bounded pre-commit candidate cohort with one hard-budget decision owner."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, Sequence, TypeVar

T = TypeVar("T")


class CohortAction(str, Enum):
    """Terminal action emitted by the bounded cohort owner."""

    COMMIT = "commit"
    DEFER = "defer"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class PendingCandidate(Generic[T]):
    """Source-ordered candidate held before the final budget decision."""

    source_ordinal: int
    timestamp_ns: int
    payload: T
    eligible: bool = True
    escalated: bool = False

    @property
    def key(self) -> tuple[int, int]:
        """Return the deterministic source/timestamp ordering key."""

        return (int(self.source_ordinal), int(self.timestamp_ns))


@dataclass(frozen=True, slots=True)
class CohortDecision(Generic[T]):
    """Final ranked action and reason for one pending candidate."""

    candidate: PendingCandidate[T]
    action: CohortAction
    rank: int | None
    reason: str


class PendingKeyframeCohort(Generic[T]):
    """Only holding structure between eligibility and durable commitment.

    Ranking can reorder candidates but cannot expand ``commit_budget``.
    Capacity and source-age bounds are deterministic and fail closed.
    """

    def __init__(self, *, max_candidates: int, max_source_age: int, commit_budget: int):
        if max_candidates <= 0 or commit_budget < 0 or max_source_age < 0:
            raise ValueError("invalid cohort bounds")
        self.max_candidates = int(max_candidates)
        self.max_source_age = int(max_source_age)
        self.commit_budget = min(int(commit_budget), self.max_candidates)
        self._pending: list[PendingCandidate[T]] = []

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, candidate: PendingCandidate[T]) -> tuple[CohortDecision[T], ...]:
        """Add one candidate while enforcing identity, age, and size bounds."""

        if self._pending and candidate.key <= self._pending[-1].key:
            if any(item.key == candidate.key for item in self._pending):
                return (CohortDecision(candidate, CohortAction.DROP, None, "duplicate_source_identity"),)
            raise ValueError("candidate source order must be strictly increasing")
        if not candidate.eligible:
            return (CohortDecision(candidate, CohortAction.DROP, None, "prefilter_ineligible"),)
        self._pending.append(candidate)
        decisions: list[CohortDecision[T]] = []
        while len(self._pending) > self.max_candidates:
            dropped = self._pending.pop(0)
            decisions.append(CohortDecision(dropped, CohortAction.DROP, None, "cohort_capacity"))
        if self.max_source_age > 0:
            newest = candidate.source_ordinal
            while self._pending and newest - self._pending[0].source_ordinal > self.max_source_age:
                expired = self._pending.pop(0)
                decisions.append(CohortDecision(expired, CohortAction.DROP, None, "cohort_source_age"))
        return tuple(decisions)

    def should_flush(self) -> bool:
        """Return whether the cohort is full or contains an escalation."""

        return len(self._pending) >= self.max_candidates or any(item.escalated for item in self._pending)

    def flush(
        self,
        *,
        ranker: Callable[[Sequence[PendingCandidate[T]]], Iterable[int]] | None = None,
    ) -> tuple[CohortDecision[T], ...]:
        """Consume the cohort and commit at most the hard budget."""

        pending = tuple(self._pending)
        self._pending.clear()
        if not pending:
            return ()
        order = list(range(len(pending))) if ranker is None else [int(i) for i in ranker(pending)]
        if sorted(order) != list(range(len(pending))):
            raise ValueError("ranker must return each cohort index exactly once")
        committed = set(order[: self.commit_budget])
        rank_of = {idx: rank for rank, idx in enumerate(order)}
        out: list[CohortDecision[T]] = []
        for idx, item in enumerate(pending):
            if idx in committed:
                out.append(CohortDecision(item, CohortAction.COMMIT, rank_of[idx], "hard_budget_rank"))
            else:
                out.append(CohortDecision(item, CohortAction.DROP, rank_of[idx], "hard_budget_exhausted"))
        return tuple(out)

    def snapshot(self) -> tuple[PendingCandidate[T], ...]:
        """Return an immutable view of the currently pending candidates."""

        return tuple(self._pending)
