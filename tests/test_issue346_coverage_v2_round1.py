"""Round-1 producer and denominator regressions for #346 coverage-v2."""

from __future__ import annotations

from eligsir.utils.coverage_ledger import (
    CoverageLedger,
    resolve_finite_source_denominator,
    resolve_managed_staged_source_denominator,
)
from eligsir.utils.partition_eval import compute_trajectory_coverage


_SCENE = "/data/datasets/floor3"


def test_burst_preingest_reservations_are_never_reclassified_as_mapping_skips():
    """The BurstBudget producer must give holdout/guard frames their own bins."""
    ledger = CoverageLedger.fresh(
        resolved_source_frames=6,
        raw_registry_source_frames=12,
        source_denominator_provenance="registry_after_stride",
        bootstrap_frames_consumed=1,
        source_identity="scene",
        effective_stride=2,
    )

    ledger.record_preingest_skip(is_holdout=True, is_guard=False, odom_valid=True)
    ledger.record_preingest_skip(is_holdout=False, is_guard=True, odom_valid=True)
    ledger.record_preingest_skip(is_holdout=False, is_guard=False, odom_valid=True)

    snap = ledger.snapshot()
    assert snap["reserved_holdout_frames"] == 1
    assert snap["reserved_guard_frames"] == 1
    assert snap["eligible_mapping_frames"] == 1
    assert snap["mapping_skipped_frames"] == 1


def test_resolved_denominator_uses_selected_stride_not_raw_registry_or_lazy_placeholder():
    """A selected stride changes coverage denominator; lazy sentinel never does."""
    assert resolve_finite_source_denominator(
        raw_registry_source_frames=101,
        frame_source_length=51,
        effective_stride=2,
    ) == (51, "post_stride_frame_source")
    assert resolve_finite_source_denominator(
        raw_registry_source_frames=101,
        frame_source_length=999_999,
        effective_stride=2,
    ) == (51, "registry_after_stride")


def test_managed_staged_source_contract_accepts_unregistered_filesystem_source():
    """A managed staged comparison may use an unregistered finite path."""
    assert resolve_managed_staged_source_denominator(
        scientific_mode=True,
        researchflow_run_id="run-123",
        researchflow_plan_sha256="a" * 64,
        staged_source_frames="1800",
        source_identity="/data/staged/custom-comparison",
    ) == (1800, "managed_staged_source_contract")


def test_managed_staged_source_contract_rejects_live_uri_source():
    """A live URI cannot acquire a finite denominator from managed env strings."""
    assert resolve_managed_staged_source_denominator(
        scientific_mode=True,
        researchflow_run_id="run-123",
        researchflow_plan_sha256="a" * 64,
        staged_source_frames="1800",
        source_identity="ros2://live",
    ) == (None, "not_managed_staged_source")


def test_staged_source_contract_cannot_change_unmanaged_or_invalid_runs():
    for scientific_mode, run_id, plan_sha, staged_count in (
        (False, "run-123", "a" * 64, "1800"),
        (True, "", "a" * 64, "1800"),
        (True, "run-123", "", "1800"),
        (True, "run-123", "a" * 64, "0"),
        (True, "run-123", "a" * 64, "not-a-count"),
    ):
        assert resolve_managed_staged_source_denominator(
            scientific_mode=scientific_mode,
            researchflow_run_id=run_id,
            researchflow_plan_sha256=plan_sha,
            staged_source_frames=staged_count,
            source_identity="/data/staged/custom-comparison",
        ) == (None, "not_managed_staged_source")


def _strict_all_counters() -> dict:
    return {
        "coverage_ledger_version": 2,
        "source_frames_ingested_total": 10,
        "bootstrap_frames_consumed": 0,
        "reserved_holdout_frames": 0,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 10,
        "frames_admitted": 9,
        "mapping_skipped_frames": 1,
        "unprocessed_frames": 0,
        "resolved_source_frames": 10,
        "raw_registry_source_frames": 10,
        "source_denominator_provenance": "post_stride_frame_source",
    }


def test_soft_all_is_budgeted_but_force_all_never_excuses_a_mapping_skip():
    """#600/#602: resolved production soft-all differs from force-all stress."""
    expected = {
        ("all", "soft_all"): ("PARTIAL_BY_BUDGET", True),
        ("force_all", "force_all_stress"): ("PARTIAL", False),
    }
    for (raw_mode, target), (verdict, valid) in expected.items():
        report = compute_trajectory_coverage(
            scene_source=_SCENE,
            n_reg=10,
            frames_admitted=9,
            partition_psnr={"p0": 20.0},
            frame_admission=raw_mode,
            admission_target_resolved=target,
            eval_valid=True,
            coverage_counters=_strict_all_counters(),
        )
        assert report["verdict"] == verdict
        assert report["mapping_result_valid"] is valid


def test_immutable_snapshot_cannot_gain_future_source_dispositions():
    """The report worker must receive its submission-time ledger snapshot."""
    ledger = CoverageLedger.fresh(
        resolved_source_frames=3,
        raw_registry_source_frames=3,
        source_denominator_provenance="post_stride_frame_source",
        bootstrap_frames_consumed=1,
        source_identity="scene",
        effective_stride=1,
    )
    submitted = ledger.snapshot()
    ledger.record_materialized(is_holdout=False, is_guard=False, odom_valid=True, admitted=True)

    assert submitted["source_frames_ingested_total"] == 1
    assert submitted["frames_admitted"] == 0
    assert ledger.snapshot()["source_frames_ingested_total"] == 2
    assert ledger.snapshot()["frames_admitted"] == 1


def test_public_coverage_denominator_is_the_selected_cohort_not_raw_registry():
    counters = _strict_all_counters()
    counters.update({
        "source_frames_ingested_total": 51,
        "eligible_mapping_frames": 51,
        "frames_admitted": 51,
        "mapping_skipped_frames": 0,
        "resolved_source_frames": 51,
        "raw_registry_source_frames": 101,
    })
    report = compute_trajectory_coverage(
        scene_source=_SCENE,
        n_reg=101,
        frames_admitted=51,
        partition_psnr={"p0": 20.0},
        frame_admission="all",
        eval_valid=True,
        coverage_counters=counters,
    )
    assert report["n_source_frames"] == 51
    assert report["raw_registry_n_source_frames"] == 101


def test_checkpoint_ledger_rejects_same_sized_different_source():
    ledger = CoverageLedger.fresh(
        resolved_source_frames=10,
        raw_registry_source_frames=10,
        source_denominator_provenance="post_stride_frame_source",
        bootstrap_frames_consumed=1,
        source_identity="scene-a",
        effective_stride=1,
    )
    expected = {
        "source_identity": "scene-a",
        "raw_registry_source_frames": 10,
        "effective_stride": 1,
        "resolved_source_frames": 10,
        "source_denominator_provenance": "post_stride_frame_source",
    }
    assert CoverageLedger.restore(ledger.snapshot(), expected_source=expected) is not None
    assert CoverageLedger.restore(
        ledger.snapshot(),
        expected_source={
            "source_identity": "scene-b",
            "raw_registry_source_frames": 10,
            "effective_stride": 1,
            "resolved_source_frames": 10,
            "source_denominator_provenance": "post_stride_frame_source",
        },
    ) is None
