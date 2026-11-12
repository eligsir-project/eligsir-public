"""Birth-cohort / mutation-batch telemetry for the evidence-driven-mutation-cadence work.

This module is pure telemetry infrastructure: three JSONL-serializable
dataclasses (``ProposalRecord``, ``CohortRecord``, ``MutationBatchRecord``)
plus a recorder (``CohortTraceRecorder``) that accumulates instances of each
and flushes them to a single JSONL file, one record per line, tagged with a
``type`` discriminator so a reader can tell the three kinds apart without
guessing from field names alone.

Nothing in here is wired into the training loop yet -- ``train_streaming.py``
does not construct a ``CohortTraceRecorder`` or call any of its ``record_*``
methods. That wiring (including populating ``ProposalRecord.source_frame_index``
from the depth-insertion candidate dict, see the provenance note in the task
this module was built for) is a separate, later integration task.

Mirrors the pattern established by ``mutation_trace.py``'s ``MutationEvent`` /
``MutationTraceRecorder`` in this same package: dataclasses with explicit
``to_dict()``/``from_dict()`` methods (no relying on ``dataclasses.asdict()``
so the on-disk schema is an explicit, reviewable contract), and a recorder
whose disabled mode (here: ``enabled=False``) is a true no-op -- zero file
I/O, matching every caller that isn't part of an active cohort-telemetry
capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ProposalRecord:
    """One candidate Gaussian proposed for insertion, before any batch commit."""

    proposal_id: str
    source_frame_index: int
    world_xyz: tuple[float, float, float]
    birth_iteration: int
    birth_topology_version: int
    voxel_key: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_frame_index": self.source_frame_index,
            "world_xyz": list(self.world_xyz),
            "birth_iteration": self.birth_iteration,
            "birth_topology_version": self.birth_topology_version,
            "voxel_key": list(self.voxel_key),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProposalRecord":
        world_xyz = payload["world_xyz"]
        voxel_key = payload["voxel_key"]
        return cls(
            proposal_id=str(payload["proposal_id"]),
            source_frame_index=int(payload["source_frame_index"]),
            world_xyz=(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
            birth_iteration=int(payload["birth_iteration"]),
            birth_topology_version=int(payload["birth_topology_version"]),
            voxel_key=(int(voxel_key[0]), int(voxel_key[1]), int(voxel_key[2])),
        )


@dataclass
class CohortRecord:
    """A group of proposals committed to the map together in one mutation batch."""

    cohort_id: str
    birth_batch_id: str
    birth_iteration: int
    member_proposal_ids: list[str]
    maturity_score_history: list[tuple[int, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "birth_batch_id": self.birth_batch_id,
            "birth_iteration": self.birth_iteration,
            "member_proposal_ids": list(self.member_proposal_ids),
            "maturity_score_history": [
                [iteration, score] for iteration, score in self.maturity_score_history
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CohortRecord":
        return cls(
            cohort_id=str(payload["cohort_id"]),
            birth_batch_id=str(payload["birth_batch_id"]),
            birth_iteration=int(payload["birth_iteration"]),
            member_proposal_ids=list(payload["member_proposal_ids"]),
            maturity_score_history=[
                (int(iteration), float(score))
                for iteration, score in payload["maturity_score_history"]
            ],
        )


@dataclass
class MutationBatchRecord:
    """One mutation-batch commit event (a group of proposals applied to the map)."""

    batch_id: str
    commit_iteration: int
    topology_version_before: int
    topology_version_after: int
    n_proposals_committed: int
    n_proposals_deduped: int
    trigger_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "commit_iteration": self.commit_iteration,
            "topology_version_before": self.topology_version_before,
            "topology_version_after": self.topology_version_after,
            "n_proposals_committed": self.n_proposals_committed,
            "n_proposals_deduped": self.n_proposals_deduped,
            "trigger_reason": self.trigger_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MutationBatchRecord":
        return cls(
            batch_id=str(payload["batch_id"]),
            commit_iteration=int(payload["commit_iteration"]),
            topology_version_before=int(payload["topology_version_before"]),
            topology_version_after=int(payload["topology_version_after"]),
            n_proposals_committed=int(payload["n_proposals_committed"]),
            n_proposals_deduped=int(payload["n_proposals_deduped"]),
            trigger_reason=str(payload["trigger_reason"]),
        )


@dataclass
class CohortTraceRecorder:
    """Accumulates proposal/cohort/mutation-batch records for one streaming run.

    ``enabled=False`` is a true no-op (default): zero overhead, zero file I/O,
    matching every run that isn't part of an active cohort-telemetry capture --
    mirrors ``MutationTraceRecorder``'s ``mode="off"`` contract in
    ``mutation_trace.py``.
    """

    enabled: bool = False
    path: str | Path = ""
    _events: list[tuple[str, Any]] = field(default_factory=list, repr=False)

    def record_proposal(
        self,
        *,
        proposal_id: str,
        source_frame_index: int,
        world_xyz: tuple[float, float, float],
        birth_iteration: int,
        birth_topology_version: int,
        voxel_key: tuple[int, int, int],
    ) -> ProposalRecord:
        """Record one proposal event. No-op (but still returns the record) when disabled."""
        record = ProposalRecord(
            proposal_id=proposal_id,
            source_frame_index=source_frame_index,
            world_xyz=world_xyz,
            birth_iteration=birth_iteration,
            birth_topology_version=birth_topology_version,
            voxel_key=voxel_key,
        )
        if self.enabled:
            self._events.append(("proposal", record))
        return record

    def record_cohort_birth(
        self,
        *,
        cohort_id: str,
        birth_batch_id: str,
        birth_iteration: int,
        member_proposal_ids: list[str],
        maturity_score_history: list[tuple[int, float]],
    ) -> CohortRecord:
        """Record one cohort-birth event. No-op (but still returns the record) when disabled."""
        record = CohortRecord(
            cohort_id=cohort_id,
            birth_batch_id=birth_batch_id,
            birth_iteration=birth_iteration,
            member_proposal_ids=member_proposal_ids,
            maturity_score_history=maturity_score_history,
        )
        if self.enabled:
            self._events.append(("cohort", record))
        return record

    def record_mutation_batch(
        self,
        *,
        batch_id: str,
        commit_iteration: int,
        topology_version_before: int,
        topology_version_after: int,
        n_proposals_committed: int,
        n_proposals_deduped: int,
        trigger_reason: str,
    ) -> MutationBatchRecord:
        """Record one mutation-batch-commit event. No-op (but still returns the record) when disabled."""
        record = MutationBatchRecord(
            batch_id=batch_id,
            commit_iteration=commit_iteration,
            topology_version_before=topology_version_before,
            topology_version_after=topology_version_after,
            n_proposals_committed=n_proposals_committed,
            n_proposals_deduped=n_proposals_deduped,
            trigger_reason=trigger_reason,
        )
        if self.enabled:
            self._events.append(("mutation_batch", record))
        return record

    def finalize(self) -> dict[str, Any]:
        """Flush all recorded events to ``self.path`` as JSONL, in the exact order recorded.

        Each line is the record's ``to_dict()`` output plus a ``"type"`` key
        (``"proposal"`` / ``"cohort"`` / ``"mutation_batch"``) so a reader can
        dispatch on it. No-op (no file written) when disabled.

        Returns a small provenance dict; callers may persist it into their own
        evidence payload, mirroring ``MutationTraceRecorder.finalize()``.
        """
        if not self.enabled:
            return {"cohort_trace_enabled": False}

        counts = {"proposal": 0, "cohort": 0, "mutation_batch": 0}
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as handle:
            for type_name, record in self._events:
                row = record.to_dict()
                row["type"] = type_name
                counts[type_name] += 1
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        return {
            "cohort_trace_enabled": True,
            "cohort_trace_path": str(target),
            "cohort_trace_proposal_count": counts["proposal"],
            "cohort_trace_cohort_count": counts["cohort"],
            "cohort_trace_batch_count": counts["mutation_batch"],
        }


def read_cohort_trace_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Read a cohort-trace JSONL file back as raw dicts (each tagged with 'type')."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"cohort trace file not found: {target}")
    for line in target.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
