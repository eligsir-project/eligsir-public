"""Tests for eligsir.optimization.gaussian.cohort_trace."""

from __future__ import annotations

import json

from eligsir.optimization.gaussian.cohort_trace import (
    CohortRecord,
    CohortTraceRecorder,
    MutationBatchRecord,
    ProposalRecord,
)


def test_proposal_record_roundtrip():
    record = ProposalRecord(
        proposal_id="p-0001",
        source_frame_index=42,
        world_xyz=(1.5, -2.25, 0.0),
        birth_iteration=1200,
        birth_topology_version=7,
        voxel_key=(3, -1, 8),
    )
    assert ProposalRecord.from_dict(record.to_dict()) == record


def test_cohort_record_roundtrip():
    record = CohortRecord(
        cohort_id="c-0001",
        birth_batch_id="b-0001",
        birth_iteration=1200,
        member_proposal_ids=["p-0001", "p-0002"],
        maturity_score_history=[(1200, 0.0), (1250, 0.3)],
    )
    assert CohortRecord.from_dict(record.to_dict()) == record


def test_mutation_batch_record_roundtrip():
    record = MutationBatchRecord(
        batch_id="b-0001",
        commit_iteration=1200,
        topology_version_before=6,
        topology_version_after=7,
        n_proposals_committed=18,
        n_proposals_deduped=3,
        trigger_reason="frame_count",
    )
    assert MutationBatchRecord.from_dict(record.to_dict()) == record


def test_recorder_enabled_writes_jsonl(tmp_path):
    out_path = tmp_path / "cohort_trace.jsonl"
    recorder = CohortTraceRecorder(enabled=True, path=out_path)

    proposal = recorder.record_proposal(
        proposal_id="p-0001",
        source_frame_index=42,
        world_xyz=(1.5, -2.25, 0.0),
        birth_iteration=1200,
        birth_topology_version=7,
        voxel_key=(3, -1, 8),
    )
    cohort = recorder.record_cohort_birth(
        cohort_id="c-0001",
        birth_batch_id="b-0001",
        birth_iteration=1200,
        member_proposal_ids=["p-0001"],
        maturity_score_history=[(1200, 0.0)],
    )
    batch = recorder.record_mutation_batch(
        batch_id="b-0001",
        commit_iteration=1200,
        topology_version_before=6,
        topology_version_after=7,
        n_proposals_committed=1,
        n_proposals_deduped=0,
        trigger_reason="step_count",
    )

    assert proposal.to_dict()["proposal_id"] == "p-0001"
    assert cohort.to_dict()["cohort_id"] == "c-0001"
    assert batch.to_dict()["batch_id"] == "b-0001"

    recorder.finalize()

    lines = out_path.read_text().splitlines()
    assert len(lines) == 3

    rows = [json.loads(line) for line in lines]
    types = [row["type"] for row in rows]
    assert types == ["proposal", "cohort", "mutation_batch"]

    proposal_row = rows[0]
    assert proposal_row["proposal_id"] == "p-0001"
    assert proposal_row["source_frame_index"] == 42
    assert tuple(proposal_row["world_xyz"]) == (1.5, -2.25, 0.0)
    assert proposal_row["birth_iteration"] == 1200
    assert proposal_row["birth_topology_version"] == 7
    assert tuple(proposal_row["voxel_key"]) == (3, -1, 8)

    cohort_row = rows[1]
    assert cohort_row["cohort_id"] == "c-0001"
    assert cohort_row["birth_batch_id"] == "b-0001"
    assert cohort_row["birth_iteration"] == 1200
    assert cohort_row["member_proposal_ids"] == ["p-0001"]
    assert [tuple(x) for x in cohort_row["maturity_score_history"]] == [(1200, 0.0)]

    batch_row = rows[2]
    assert batch_row["batch_id"] == "b-0001"
    assert batch_row["commit_iteration"] == 1200
    assert batch_row["topology_version_before"] == 6
    assert batch_row["topology_version_after"] == 7
    assert batch_row["n_proposals_committed"] == 1
    assert batch_row["n_proposals_deduped"] == 0
    assert batch_row["trigger_reason"] == "step_count"


def test_recorder_disabled_is_noop(tmp_path):
    out_path = tmp_path / "cohort_trace_disabled.jsonl"
    recorder = CohortTraceRecorder(enabled=False, path=out_path)

    recorder.record_proposal(
        proposal_id="p-0001",
        source_frame_index=42,
        world_xyz=(1.5, -2.25, 0.0),
        birth_iteration=1200,
        birth_topology_version=7,
        voxel_key=(3, -1, 8),
    )
    recorder.record_cohort_birth(
        cohort_id="c-0001",
        birth_batch_id="b-0001",
        birth_iteration=1200,
        member_proposal_ids=["p-0001"],
        maturity_score_history=[(1200, 0.0)],
    )
    recorder.record_mutation_batch(
        batch_id="b-0001",
        commit_iteration=1200,
        topology_version_before=6,
        topology_version_after=7,
        n_proposals_committed=1,
        n_proposals_deduped=0,
        trigger_reason="point_cap",
    )
    recorder.finalize()

    assert not out_path.exists()
