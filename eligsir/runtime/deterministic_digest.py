"""Stable control-plane digest for deterministic mapping replay.

The digest deliberately never hashes tensor payload bytes.  Doing so would add
large synchronization/D2H costs and would conflate floating-point numerical
parity with deterministic scheduling/ownership.  Instead callers record the
facts whose replay order must be identical: source identities and ingress
actions, supervision batch identities/work counts, mutation proposal order and
population/version deltas.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
from threading import RLock
from typing import Any, Mapping


_REQUIRED_EVENT_KINDS = frozenset({"ingress", "optimizer_batch", "mutation_commit"})


def _canonical(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, set):
        return sorted((_canonical(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True))
    # Enums and path-like values have stable string/value forms without
    # importing their defining modules here.
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    return str(value)


class DeterministicReplayDigest:
    """Incremental SHA-256 over length-framed canonical JSON events."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        """Reset the digest, event ordinal, and per-kind counters."""

        with self._lock:
            self._hasher = hashlib.sha256()
            self._events = 0
            self._counts: Counter[str] = Counter()

    def record(self, kind: str, payload: Mapping[str, Any] | None = None) -> str:
        """Append one canonical length-framed event and return the digest."""

        event = {
            "ordinal": self._events,
            "kind": str(kind),
            "payload": _canonical(dict(payload or {})),
        }
        raw = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        frame = len(raw).to_bytes(8, "big") + raw
        with self._lock:
            self._hasher.update(frame)
            self._events += 1
            self._counts[str(kind)] += 1
            return self._hasher.hexdigest()

    @property
    def hexdigest(self) -> str:
        """Return the current SHA-256 hexadecimal digest."""

        with self._lock:
            return self._hasher.hexdigest()

    @property
    def event_count(self) -> int:
        """Return the number of framed events incorporated so far."""

        with self._lock:
            return int(self._events)

    def snapshot(self) -> dict:
        """Return report-ready digest metadata and event counts."""

        with self._lock:
            missing = sorted(_REQUIRED_EVENT_KINDS.difference(self._counts))
            return {
                "algorithm": "sha256-canonical-json-v1",
                "digest": self._hasher.hexdigest(),
                "event_count": int(self._events),
                "event_counts": dict(sorted(self._counts.items())),
                "tensor_payload_bytes_hashed": 0,
                "coverage_complete": not missing,
                "missing_event_kinds": missing,
            }


GLOBAL_REPLAY_DIGEST = DeterministicReplayDigest()


def record_global(kind: str, **payload) -> str:
    """Record one event in the process-global replay digest."""

    return GLOBAL_REPLAY_DIGEST.record(kind, payload)


def global_snapshot() -> dict:
    """Return a report-ready snapshot of the global digest."""

    return GLOBAL_REPLAY_DIGEST.snapshot()


def reset_global() -> None:
    """Reset the process-global digest before a new canonical run."""

    GLOBAL_REPLAY_DIGEST.reset()


def report_with_global_digest(determinism: Mapping[str, Any] | None) -> dict:
    """Append the current replay digest to a determinism report block."""

    block = {} if determinism is None else dict(determinism)
    block["replay_digest"] = global_snapshot()
    return block


__all__ = [
    "DeterministicReplayDigest",
    "GLOBAL_REPLAY_DIGEST",
    "global_snapshot",
    "record_global",
    "report_with_global_digest",
    "reset_global",
]
