"""Compact, in-memory terminal lineage records for a mapping session.

This is deliberately a CPU-only inspection primitive: it records no payloads,
does no file I/O, and makes no backend or Perfetto claim.  Callers opt in by
injecting one sink into the authoritative terminal owner.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from eligsir.contracts.runtime import SourceIdentity


_QUEUE_DEPTH_FACTS = frozenset({"transport.queue_depth", "ingress_queue_depth"})
_SOURCE_AGE_FACTS = frozenset({"transport.age_ns", "ingress_age_ns"})
_QUEUE_OLDEST_AGE_FACTS = frozenset({"pending_oldest_age_ns"})


def _integer_fact(evidence: Iterable[object], names: frozenset[str]) -> int | None:
    for fact in evidence:
        if getattr(fact, "name", None) not in names:
            continue
        value = getattr(fact, "value", None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
    return None


@dataclass(frozen=True, slots=True)
class ExecutionTraceRecord:
    """One terminal mapping-session submission with no packet payload fields."""

    source_fingerprint: str
    action: str
    reason: str
    queue_depth: int | None
    source_age_ns: int | None  # Current submitted source only.
    queue_oldest_age_ns: int | None  # May belong to a different pending source.
    started_monotonic_ns: int
    completed_monotonic_ns: int
    cpu_span_ns: int
    terminal_gap_ns: int | None
    duplicate: bool

    def as_dict(self) -> dict[str, str | int | bool | None]:
        """Return a deterministic, JSON-ready record without writing it anywhere."""

        return asdict(self)


class ExecutionTraceSink:
    """Append-only terminal records keyed by ``SourceIdentity.fingerprint()``."""

    def __init__(self, *, monotonic_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._monotonic_ns = monotonic_ns
        self._records: list[ExecutionTraceRecord] = []
        self._last_terminal_ns: int | None = None

    def begin(self) -> int:
        """Capture the start of one CPU submit span."""

        return int(self._monotonic_ns())

    def record_terminal(
        self,
        *,
        source: SourceIdentity,
        action: str,
        reason: str,
        evidence: Iterable[object],
        started_monotonic_ns: int,
        duplicate: bool,
    ) -> ExecutionTraceRecord:
        """Append one terminal outcome; no packet or transport object is retained."""

        completed = int(self._monotonic_ns())
        started = int(started_monotonic_ns)
        if completed < started:
            raise ValueError("monotonic trace clock moved backwards during a CPU span")
        if self._last_terminal_ns is not None and completed < self._last_terminal_ns:
            raise ValueError("monotonic trace clock moved backwards between terminal records")
        record = ExecutionTraceRecord(
            source_fingerprint=source.fingerprint(),
            action=str(action),
            reason=str(reason),
            queue_depth=_integer_fact(evidence, _QUEUE_DEPTH_FACTS),
            source_age_ns=_integer_fact(evidence, _SOURCE_AGE_FACTS),
            queue_oldest_age_ns=_integer_fact(evidence, _QUEUE_OLDEST_AGE_FACTS),
            started_monotonic_ns=started,
            completed_monotonic_ns=completed,
            cpu_span_ns=completed - started,
            terminal_gap_ns=(
                None if self._last_terminal_ns is None else completed - self._last_terminal_ns
            ),
            duplicate=bool(duplicate),
        )
        self._records.append(record)
        self._last_terminal_ns = completed
        return record

    def records(self) -> tuple[ExecutionTraceRecord, ...]:
        """Return records in the deterministic terminal-observation order."""

        return tuple(self._records)

    def records_for(self, source_fingerprint: str) -> tuple[ExecutionTraceRecord, ...]:
        """Return every terminal observation for one canonical source identity."""

        return tuple(
            record for record in self._records if record.source_fingerprint == source_fingerprint
        )

    def export(self) -> tuple[dict[str, str | int | bool | None], ...]:
        """Return a deterministic in-memory export; persistence stays caller-owned."""

        return tuple(record.as_dict() for record in self._records)


__all__ = ["ExecutionTraceRecord", "ExecutionTraceSink"]
