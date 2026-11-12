"""Single owner for coverage-v2 finite-source accounting."""

from __future__ import annotations

from copy import deepcopy
from math import ceil
from typing import Any
from urllib.parse import urlparse

from eligsir.contracts.runtime import SourceIdentity


COVERAGE_LEDGER_VERSION = 2
LAZY_SOURCE_LENGTH_SENTINEL = 999_999
MANAGED_STAGED_SOURCE_FRAMES_ENV = "ELIGSIR_MANAGED_STAGED_SOURCE_FRAMES"
COUNTERS = (
    "source_frames_ingested_total",
    "bootstrap_frames_consumed",
    "reserved_holdout_frames",
    "reserved_guard_frames",
    "rejected_invalid_frames",
    "eligible_mapping_frames",
    "frames_admitted",
    "mapping_skipped_frames",
    "unprocessed_frames",
)


def resolve_finite_source_denominator(
    *, raw_registry_source_frames: int | None,
    frame_source_length: int | None,
    effective_stride: int = 1,
) -> tuple[int | None, str]:
    """Resolve the selected replay cohort size without trusting lazy sentinels."""
    raw = int(raw_registry_source_frames or 0)
    source_len = int(frame_source_length or 0)
    stride = max(1, int(effective_stride or 1))
    if source_len > 0 and source_len < LAZY_SOURCE_LENGTH_SENTINEL:
        return source_len, "post_stride_frame_source"
    if raw > 0:
        return ceil(raw / stride), "registry_after_stride"
    return None, "unknown_unbounded"


def resolve_managed_staged_source_denominator(
    *,
    scientific_mode: bool,
    researchflow_run_id: str | None,
    researchflow_plan_sha256: str | None,
    staged_source_frames: str | None,
    source_identity: str | None,
) -> tuple[int | None, str]:
    """Accept a finite staged count only for managed filesystem-backed sources."""
    if not (scientific_mode and researchflow_run_id and researchflow_plan_sha256):
        return None, "not_managed_staged_source"
    if not source_identity or urlparse(source_identity).scheme:
        return None, "not_managed_staged_source"
    try:
        count = int(staged_source_frames or 0)
    except (TypeError, ValueError):
        return None, "not_managed_staged_source"
    if count <= 0:
        return None, "not_managed_staged_source"
    return count, "managed_staged_source_contract"


class CoverageLedger:
    """Mutable producer ledger with immutable report/checkpoint snapshots."""

    def __init__(self, state: dict[str, Any]):
        self._state = state
        self._causal_source_fingerprints: set[str] = set()

    @classmethod
    def fresh(
        cls,
        *,
        resolved_source_frames: int,
        raw_registry_source_frames: int | None,
        source_denominator_provenance: str,
        bootstrap_frames_consumed: int,
        source_identity: str,
        effective_stride: int,
    ) -> "CoverageLedger":
        resolved = int(resolved_source_frames)
        bootstrap = int(bootstrap_frames_consumed)
        if resolved < 0 or bootstrap < 0 or bootstrap > resolved:
            raise ValueError("invalid coverage ledger source/bootstrap counts")
        return cls({
            "coverage_ledger_version": COVERAGE_LEDGER_VERSION,
            "resolved_source_frames": resolved,
            "raw_registry_source_frames": (
                int(raw_registry_source_frames)
                if raw_registry_source_frames is not None else None
            ),
            "source_identity": str(source_identity),
            "effective_stride": int(effective_stride),
            "source_denominator_provenance": str(source_denominator_provenance),
            "source_frames_ingested_total": bootstrap,
            "bootstrap_frames_consumed": bootstrap,
            "reserved_holdout_frames": 0,
            "reserved_guard_frames": 0,
            "rejected_invalid_frames": 0,
            "eligible_mapping_frames": 0,
            "frames_admitted": 0,
            "mapping_skipped_frames": 0,
            "unprocessed_frames": resolved - bootstrap,
        })

    @classmethod
    def restore(
        cls, snapshot: object, *, expected_source: dict[str, Any] | None = None
    ) -> "CoverageLedger | None":
        if not isinstance(snapshot, dict):
            return None
        if snapshot.get("coverage_ledger_version") != COVERAGE_LEDGER_VERSION:
            return None
        if any(key not in snapshot for key in COUNTERS):
            return None
        if not isinstance(snapshot.get("resolved_source_frames"), int):
            return None
        if expected_source is not None:
            required = (
                "source_identity",
                "raw_registry_source_frames",
                "effective_stride",
                "resolved_source_frames",
                "source_denominator_provenance",
            )
            if any(snapshot.get(key) != expected_source.get(key) for key in required):
                return None
        return cls(deepcopy(snapshot))

    def snapshot(self) -> dict[str, Any]:
        snapshot = deepcopy(self._state)
        if self._causal_source_fingerprints:
            snapshot["causal_source_accounting"] = {
                "mode": "terminal_disposition",
                "unique_source_count": len(self._causal_source_fingerprints),
            }
        elif "causal_source_accounting" in self._state:
            snapshot["causal_source_accounting"] = deepcopy(
                self._state["causal_source_accounting"]
            )
        return snapshot

    def _record(self, *, is_holdout: bool, is_guard: bool, odom_valid: bool | None,
                admitted: bool | None, is_bootstrap: bool = False) -> None:
        self._state["source_frames_ingested_total"] += 1
        self._state["unprocessed_frames"] = max(
            0,
            self._state["resolved_source_frames"]
            - self._state["source_frames_ingested_total"],
        )
        if is_bootstrap:
            self._state["bootstrap_frames_consumed"] += 1
        elif odom_valid is False:
            self._state["rejected_invalid_frames"] += 1
        elif is_holdout:
            self._state["reserved_holdout_frames"] += 1
        elif is_guard:
            self._state["reserved_guard_frames"] += 1
        else:
            self._state["eligible_mapping_frames"] += 1
            if admitted:
                self._state["frames_admitted"] += 1
            else:
                self._state["mapping_skipped_frames"] += 1

    def record_preingest_skip(self, *, is_holdout: bool, is_guard: bool,
                              odom_valid: bool | None) -> None:
        self._record(
            is_holdout=is_holdout,
            is_guard=is_guard,
            odom_valid=odom_valid,
            admitted=False,
        )

    def record_materialized(self, *, is_holdout: bool, is_guard: bool,
                            odom_valid: bool | None, admitted: bool) -> None:
        self._record(
            is_holdout=is_holdout,
            is_guard=is_guard,
            odom_valid=odom_valid,
            admitted=admitted,
        )

    def record_causal_source(
        self,
        *,
        source: SourceIdentity,
        role: str,
        admitted: bool,
        bootstrap: bool = False,
    ) -> None:
        """Record one causal terminal disposition by its immutable identity.

        Causal replay has already classified transport and semantic outcomes at
        the admission owner.  This method only transfers that classification
        into the existing finite-source ledger; it never infers admission from
        materialization counters or runtime frame totals.
        """
        if not isinstance(source, SourceIdentity):
            raise TypeError("causal coverage requires SourceIdentity")
        if role not in {"train", "holdout", "guard"}:
            raise ValueError(f"unknown causal coverage role {role!r}")
        fingerprint = source.fingerprint()
        if fingerprint in self._causal_source_fingerprints:
            raise ValueError("causal coverage source identity was recorded twice")
        if self._state["source_frames_ingested_total"] >= self._state["resolved_source_frames"]:
            raise ValueError("causal coverage source catalogue was over-consumed")
        self._causal_source_fingerprints.add(fingerprint)
        self._record(
            is_holdout=role == "holdout",
            is_guard=role == "guard",
            odom_valid=True,
            admitted=bool(admitted),
            is_bootstrap=bool(bootstrap),
        )
