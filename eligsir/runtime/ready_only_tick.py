"""Ready-only source selection at the GPU-owner transaction boundary.

This module is intentionally a small seam between the existing work planner
and the future prepared-source binder.  It selects only leases that the
``ReadySourceOwner`` reports as complete, never waits for a preferred source,
and binds the current map versions only after selection.  It does not create
cameras, perform transfers, or rewrite a ``MappingWorkPlan``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any, Iterable

from eligsir.contracts import SourceIdentity
from eligsir.runtime.prepared_pose_correction import PreparedPoseCorrection
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner


def _exact_non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative exact integer")
    return value


@dataclass(frozen=True, slots=True)
class ReadySourceVersions:
    """Map/version fence sampled immediately before JIT binding."""

    map_content_version: int
    topology_version: int
    parameter_version: int

    def __post_init__(self) -> None:
        for name in (
            "map_content_version",
            "topology_version",
            "parameter_version",
        ):
            object.__setattr__(
                self,
                name,
                _exact_non_negative_int(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class ReadySourceBinding:
    """One map-version-independent lease with a current JIT fence."""

    lease: PreparedSourceLease
    versions: ReadySourceVersions
    pose_correction: PreparedPoseCorrection | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lease, PreparedSourceLease):
            raise TypeError("lease must be PreparedSourceLease")
        if not isinstance(self.versions, ReadySourceVersions):
            raise TypeError("versions must be ReadySourceVersions")
        if self.pose_correction is not None and not isinstance(
            self.pose_correction, PreparedPoseCorrection
        ):
            raise TypeError("pose_correction must be PreparedPoseCorrection or None")

    def handoff_to_compute(self, transfer_owner: Any) -> None:
        """Handoff a selected lease immediately before consumer use."""
        self.lease.handoff_to_compute(transfer_owner)
        if self.pose_correction is not None:
            self.pose_correction.handoff_to_compute(transfer_owner)


@dataclass(frozen=True, slots=True)
class ReadyOnlyTickSelection:
    """Bounded ready work selected for one optimizer transaction."""

    leases: tuple[PreparedSourceLease, ...]
    fresh_count: int
    replay_count: int
    fallback_to_replay: bool
    starved: bool

    def __post_init__(self) -> None:
        leases = tuple(self.leases)
        if not all(isinstance(lease, PreparedSourceLease) for lease in leases):
            raise TypeError("leases must contain PreparedSourceLease values")
        object.__setattr__(self, "leases", leases)
        for name in ("fresh_count", "replay_count"):
            value = _exact_non_negative_int(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.fresh_count + self.replay_count != len(leases):
            raise ValueError("ready selection counts must match lease count")
        if type(self.fallback_to_replay) is not bool or type(self.starved) is not bool:
            raise TypeError("ready selection flags must be bool")


@dataclass(frozen=True, slots=True)
class ReadyOnlyTickDiagnostics:
    """Bounded selection counters suitable for a runtime tick trace."""

    selection_count: int
    fallback_count: int
    starvation_count: int


class ReadyOnlyTickSelector:
    """Select ready fresh/replay leases without introducing a wait path."""

    def __init__(
        self,
        owner: ReadySourceOwner,
        *,
        batch_size: int,
        warmup_complete: bool = False,
    ) -> None:
        if not isinstance(owner, ReadySourceOwner):
            raise TypeError("owner must be ReadySourceOwner")
        if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
            raise TypeError("batch_size must be a positive integer")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(warmup_complete) is not bool:
            raise TypeError("warmup_complete must be bool")
        self._owner = owner
        self._batch_size = batch_size
        self._warmup_complete = warmup_complete
        self._selection_count = 0
        self._fallback_count = 0
        self._starvation_count = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    def mark_warmup_complete(self) -> None:
        """Enable starvation diagnostics once bootstrap has established B views."""

        self._warmup_complete = True

    @property
    def diagnostics(self) -> ReadyOnlyTickDiagnostics:
        return ReadyOnlyTickDiagnostics(
            selection_count=self._selection_count,
            fallback_count=self._fallback_count,
            starvation_count=self._starvation_count,
        )

    def select(
        self,
        *,
        fresh_sources: Iterable[SourceIdentity] | None = None,
        replay_sources: Iterable[SourceIdentity] | None = None,
        demanded_replay_sources: Iterable[SourceIdentity] = (),
    ) -> ReadyOnlyTickSelection:
        """Select up to B already-ready leases, with replay as immediate fallback.

        ``fresh_sources`` identifies the new-source identities named by the
        current planner snapshot.  The owner itself enforces the fresh-first
        lane ordering and polls completion events without waiting.
        """

        fresh = None if fresh_sources is None else frozenset(fresh_sources)
        requested_replay = None if replay_sources is None else frozenset(replay_sources)
        demanded = frozenset(demanded_replay_sources)
        replay = (
            requested_replay
            if requested_replay is None
            else requested_replay | demanded
        )
        leases = tuple(
            self._owner.select(
                self._batch_size,
                fresh_sources=fresh,
                replay_sources=replay,
            )
        )
        fresh_count = sum(
            lease.source_identity in fresh
            if fresh is not None
            else lease.source_identity not in replay
            if replay is not None
            else False
            for lease in leases
        )
        replay_count = len(leases) - fresh_count
        fallback = replay_count > 0 and fresh_count < self._batch_size
        starved = self._warmup_complete and len(leases) < self._batch_size
        self._selection_count += 1
        if fallback:
            self._fallback_count += 1
        if starved:
            self._starvation_count += 1
        return ReadyOnlyTickSelection(
            leases=leases,
            fresh_count=fresh_count,
            replay_count=replay_count,
            fallback_to_replay=fallback,
            starved=starved,
        )

    @staticmethod
    def bind_current_versions(
        selection: ReadyOnlyTickSelection,
        versions: ReadySourceVersions,
    ) -> tuple[ReadySourceBinding, ...]:
        """Create version-fenced bindings after the previous transaction fence."""

        if not isinstance(selection, ReadyOnlyTickSelection):
            raise TypeError("selection must be ReadyOnlyTickSelection")
        if not isinstance(versions, ReadySourceVersions):
            raise TypeError("versions must be ReadySourceVersions")
        return tuple(
            ReadySourceBinding(lease=lease, versions=versions)
            for lease in selection.leases
        )

    def release(self, selection: ReadyOnlyTickSelection) -> None:
        """Return selected leases after the GPU owner completes their use."""

        if not isinstance(selection, ReadyOnlyTickSelection):
            raise TypeError("selection must be ReadyOnlyTickSelection")
        for lease in selection.leases:
            self._owner.release(lease)

    def requeue(
        self,
        selection: ReadyOnlyTickSelection,
        *,
        ready_event: Any,
        lane: ReadySourceLane | str | None = None,
    ) -> None:
        """Return selected leases after a compute-side continuation event."""

        if not isinstance(selection, ReadyOnlyTickSelection):
            raise TypeError("selection must be ReadyOnlyTickSelection")
        if not callable(getattr(ready_event, "query", None)):
            raise TypeError("ready_event must provide a callable query() method")
        self._owner.requeue_batch(selection.leases, ready_event=ready_event, lane=lane)

    def requeue_batch(
        self,
        leases: tuple[PreparedSourceLease, ...],
        *,
        ready_event: Any,
        lane: ReadySourceLane | str | None = None,
    ) -> tuple[PreparedSourceLease, ...]:
        """Atomically return a combined selection batch to one target lane."""

        return self._owner.requeue_batch(leases, ready_event=ready_event, lane=lane)


__all__ = [
    "ReadyOnlyTickDiagnostics",
    "ReadyOnlyTickSelection",
    "ReadyOnlyTickSelector",
    "ReadySourceBinding",
    "ReadySourceVersions",
]
