"""Issue #346: trajectory coverage must describe source traversal, not test bins.

The old contract treated correctly reserved bootstrap/holdout frames as missing
mapping frames and let test-partition placement decide whether a source was
fully covered.  These fixtures are hand-derived from the #494/#499 shape:
the whole finite source is traversed, while bootstrap and holdout frames are
deliberately never admitted to mapping.
"""

from __future__ import annotations

import pytest

from eligsir.contracts.runtime import SourceIdentity
from eligsir.utils.coverage_ledger import CoverageLedger
from eligsir.utils.partition_eval import compute_trajectory_coverage


_SCENE = "/data/datasets/floor3"


def _complete_all_mode_counters() -> dict:
    # 596 source frames: 5 bootstrap + 6 reserved holdouts + 585 mapping.
    return {
        "coverage_ledger_version": 2,
        "resolved_source_frames": 596,
        "raw_registry_source_frames": 596,
        "source_denominator_provenance": "post_stride_frame_source",
        "source_frames_ingested_total": 596,
        "bootstrap_frames_consumed": 5,
        "reserved_holdout_frames": 6,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 585,
        "frames_admitted": 585,
        "mapping_skipped_frames": 0,
        "unprocessed_frames": 0,
    }


def test_complete_source_with_reserved_holdouts_is_full_without_all_test_bins():
    """Removing a correct holdout from mapping must not look like frame loss.

    A regression to the old ``frames_admitted / n_reg`` or partition-bin gate
    makes this fail: 585/596 is below one, and only one test bin is populated.
    """
    report = compute_trajectory_coverage(
        scene_source=_SCENE,
        n_reg=596,
        frames_admitted=585,
        partition_psnr={"p0": 18.4, "p1": None, "p2": None, "p3": None, "p4": None, "p5": None},
        frame_admission="all",
        eval_valid=True,
        coverage_counters=_complete_all_mode_counters(),
    )

    assert report["coverage_contract_version"] == 2
    assert report["source_coverage_fraction"] == 1.0
    assert report["eligible_mapping_accounted"] is True
    assert report["mapping_admission_complete"] is True
    assert report["full_coverage"] is True
    assert report["verdict"] == "FULL"
    assert report["mapping_result_valid"] is True


def test_registered_historical_report_without_v2_counters_fails_closed():
    """Old report artifacts have insufficient facts to prove source coverage."""
    report = compute_trajectory_coverage(
        scene_source=_SCENE,
        n_reg=596,
        frames_admitted=596,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        eval_valid=True,
    )

    assert report["coverage_accounting_status"] == "MISSING"
    assert report["full_coverage"] is None
    assert report["verdict"] == "UNKNOWN_COVERAGE_COUNTERS"
    assert report["mapping_result_valid"] is False


def test_gated_mapping_is_valid_when_every_eligible_frame_is_accounted():
    """A keyframe gate may intentionally skip mapping without losing source data."""
    counters = {
        "coverage_ledger_version": 2,
        "resolved_source_frames": 100,
        "raw_registry_source_frames": 100,
        "source_denominator_provenance": "post_stride_frame_source",
        "source_frames_ingested_total": 100,
        "bootstrap_frames_consumed": 5,
        "reserved_holdout_frames": 5,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 90,
        "frames_admitted": 12,
        "mapping_skipped_frames": 78,
        "unprocessed_frames": 0,
    }
    report = compute_trajectory_coverage(
        scene_source=_SCENE,
        n_reg=100,
        frames_admitted=12,
        partition_psnr={"p0": 19.0},
        frame_admission="hybrid_keyframe",
        eval_valid=True,
        coverage_counters=counters,
    )

    assert report["source_coverage_fraction"] == 1.0
    assert report["eligible_mapping_accounted"] is True
    assert report["mapping_admission_complete"] is False
    assert report["verdict"] == "PARTIAL_BY_DESIGN"
    assert report["mapping_result_valid"] is True


def test_counter_invariant_failure_cannot_be_marked_valid():
    """An unclassified source frame is a missing fact, never a benign gap."""
    counters = _complete_all_mode_counters()
    counters["mapping_skipped_frames"] = 1
    report = compute_trajectory_coverage(
        scene_source=_SCENE,
        n_reg=596,
        frames_admitted=585,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="all",
        eval_valid=True,
        coverage_counters=counters,
    )

    assert report["coverage_accounting_status"] == "INVALID"
    assert report["mapping_result_valid"] is False


def test_causal_ledger_records_terminal_roles_by_unique_source_identity():
    ledger = CoverageLedger.fresh(
        resolved_source_frames=6,
        raw_registry_source_frames=None,
        source_denominator_provenance="causal_source_catalog",
        bootstrap_frames_consumed=0,
        source_identity="/data/fr3",
        effective_stride=1,
    )
    sources = [
        SourceIdentity("fr3", 0, ordinal, ordinal * 1_000_000)
        for ordinal in range(6)
    ]
    ledger.record_causal_source(source=sources[0], role="train", admitted=False, bootstrap=True)
    ledger.record_causal_source(source=sources[1], role="train", admitted=True)
    ledger.record_causal_source(source=sources[2], role="train", admitted=False)
    ledger.record_causal_source(source=sources[3], role="holdout", admitted=False)
    ledger.record_causal_source(source=sources[4], role="guard", admitted=False)
    ledger.record_causal_source(source=sources[5], role="train", admitted=False)

    snapshot = ledger.snapshot()
    assert snapshot["bootstrap_frames_consumed"] == 1
    assert snapshot["reserved_holdout_frames"] == 1
    assert snapshot["reserved_guard_frames"] == 1
    assert snapshot["eligible_mapping_frames"] == 3
    assert snapshot["frames_admitted"] == 1
    assert snapshot["mapping_skipped_frames"] == 2
    assert snapshot["unprocessed_frames"] == 0
    assert snapshot["causal_source_accounting"] == {
        "mode": "terminal_disposition",
        "unique_source_count": 6,
    }
    with pytest.raises(ValueError, match="recorded twice"):
        ledger.record_causal_source(source=sources[1], role="train", admitted=True)


def test_causal_ledger_requires_exact_source_identity():
    ledger = CoverageLedger.fresh(
        resolved_source_frames=1,
        raw_registry_source_frames=None,
        source_denominator_provenance="causal_source_catalog",
        bootstrap_frames_consumed=0,
        source_identity="/data/fr3",
        effective_stride=1,
    )
    with pytest.raises(TypeError, match="SourceIdentity"):
        ledger.record_causal_source(source=object(), role="train", admitted=True)
