"""Tests for AdmissionFingerprint (issue #795).

Verifies: owner attribution, gap distribution quantiles, mismatch flag,
and serialization correctness.
"""

import pytest
import numpy as np
from eligsir.utils.admission_fingerprint import AdmissionFingerprint


class TestAdmissionFingerprintOwnerAttribution:
    """Owner attribution: each mechanism gets counted correctly."""

    def test_admit_owners_tracked_separately(self) -> None:
        """Admits by different owners are counted separately."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.record_admit("pose_delta", gap=5, forced_hard=False)
        fp.record_admit("new_content", gap=3, forced_hard=False)

        assert fp.admit_owners == {"max_gap": 1, "pose_delta": 1, "new_content": 1}
        assert fp.admitted_total == 3

    def test_reject_owners_tracked_separately(self) -> None:
        """Rejects by different owners are counted separately."""
        fp = AdmissionFingerprint()
        fp.record_reject("no_motion")
        fp.record_reject("no_motion")
        fp.record_reject("low_overlap")

        assert fp.reject_owners == {"no_motion": 2, "low_overlap": 1}
        assert fp.rejected_total == 3

    def test_force_hard_vs_soft_counted(self) -> None:
        """Hard and soft forces are counted separately."""
        fp = AdmissionFingerprint()
        fp.record_admit("max_gap", gap=10, forced_hard=True, forced_soft=False)
        fp.record_admit("low_overlap_force_admit", gap=5, forced_hard=False, forced_soft=True)
        fp.record_admit("pose_delta", gap=3, forced_hard=False, forced_soft=False)

        assert fp.force_hard_count == 1
        assert fp.force_soft_count == 1

    def test_gate_bypasses_recorded(self) -> None:
        """Gate bypass counts are tracked."""
        fp = AdmissionFingerprint()
        fp.record_gate_bypass("exposure")
        fp.record_gate_bypass("exposure")
        fp.record_gate_bypass("blur")

        assert fp.gate_bypasses == {"exposure": 2, "blur": 1}


class TestAdmissionFingerprintNoveltyGate:
    """Issue #928: tri-state novelty-gate outcome tallying, read off
    KeyframeDecision.trace dicts (as _keyframe_admission_closure passes
    them)."""

    def test_evaluated_outcomes_counted(self) -> None:
        fp = AdmissionFingerprint()
        fp.record_novelty_gate({"novelty_gate_outcome": "evaluated_novel"})
        fp.record_novelty_gate({"novelty_gate_outcome": "evaluated_novel"})
        fp.record_novelty_gate({"novelty_gate_outcome": "evaluated_redundant"})

        assert fp.novelty_gate_outcome_counts == {
            "evaluated_novel": 2,
            "evaluated_redundant": 1,
        }
        assert fp.novelty_gate_unavailable_reason_counts == {}

    def test_not_evaluated_reason_breakdown(self) -> None:
        fp = AdmissionFingerprint()
        fp.record_novelty_gate({
            "novelty_gate_outcome": "not_evaluated",
            "novelty_gate_unavailable_reason": "gate_disabled",
        })
        fp.record_novelty_gate({
            "novelty_gate_outcome": "not_evaluated",
            "novelty_gate_unavailable_reason": "hard_ceiling",
        })
        fp.record_novelty_gate({
            "novelty_gate_outcome": "not_evaluated",
            "novelty_gate_unavailable_reason": "gate_disabled",
        })

        assert fp.novelty_gate_outcome_counts == {"not_evaluated": 3}
        assert fp.novelty_gate_unavailable_reason_counts == {
            "gate_disabled": 2,
            "hard_ceiling": 1,
        }

    def test_not_applicable_and_missing_trace_are_not_counted(self) -> None:
        """The vast majority of ordinary (non-forced) candidates must not
        dilute the tri-state denominator -- see record_novelty_gate's
        docstring for why."""
        fp = AdmissionFingerprint()
        fp.record_novelty_gate({"novelty_gate_outcome": "not_applicable"})
        fp.record_novelty_gate(None)
        fp.record_novelty_gate({})

        assert fp.novelty_gate_outcome_counts == {}
        assert fp.novelty_gate_unavailable_reason_counts == {}

    def test_serialized_into_to_dict(self) -> None:
        fp = AdmissionFingerprint()
        fp.record_novelty_gate({"novelty_gate_outcome": "evaluated_novel"})
        fp.record_novelty_gate({
            "novelty_gate_outcome": "not_evaluated",
            "novelty_gate_unavailable_reason": "no_occupancy_fn",
        })
        d = fp.to_dict()
        assert d["novelty_gate"] == {
            "outcome_counts": {"evaluated_novel": 1, "not_evaluated": 1},
            "unavailable_reason_counts": {"no_occupancy_fn": 1},
            "none_cause_counts": {},
        }


class TestAdmissionFingerprintGapDistribution:
    """Gap distribution: quantiles computed correctly on synthetic sequences."""

    def test_gap_histogram_simple_periodic(self) -> None:
        """Periodic frame admission (every Nth frame) produces exact histogram."""
        fp = AdmissionFingerprint()
        fp.max_gap = 10
        # Periodic every 10 frames
        for _ in range(5):
            fp.record_admit("max_gap", gap=10, forced_hard=True)

        assert fp.gap_histogram == {10: 5}
        assert fp.gaps_at_max_gap == 5

    def test_gap_percentiles_simple_sequence(self) -> None:
        """Percentiles on known sequence: [3, 3, 5, 10, 20]."""
        fp = AdmissionFingerprint()
        fp.max_gap = 10
        gaps_to_record = [3, 3, 5, 10, 20]
        for gap in gaps_to_record:
            fp.record_admit(f"reason_{gap}", gap=gap, forced_hard=(gap == 10))

        fp.finalize(total_source_frames=100)

        # Verify histogram
        assert fp.gap_histogram == {3: 2, 5: 1, 10: 1, 20: 1}

        # Verify percentiles
        assert fp.gap_percentiles["p50"] == 5.0  # median of [3, 3, 5, 10, 20]
        assert fp.gap_percentiles["mean"] == 8.2  # (3+3+5+10+20)/5
        assert fp.gap_percentiles["max"] == 20.0
        assert fp.gap_percentiles["p10"] < fp.gap_percentiles["p50"]
        assert fp.gap_percentiles["p50"] < fp.gap_percentiles["p99"]

    def test_gap_at_max_gap_boundary(self) -> None:
        """Frames at gap == max_gap or gap == max_gap+1 are counted."""
        fp = AdmissionFingerprint()
        fp.max_gap = 10
        fp.record_admit("max_gap", gap=10, forced_hard=True)  # counts
        fp.record_admit("max_gap", gap=11, forced_hard=True)  # counts (within 1)
        fp.record_admit("max_gap", gap=12, forced_hard=True)  # does not count
        fp.record_admit("pose_delta", gap=9, forced_hard=False)  # does not count

        assert fp.gaps_at_max_gap == 2  # only gap=10 and gap=11


class TestAdmissionFingerprintMismatchFlag:
    """Nominal-vs-realized mismatch flag: fires correctly."""

    def test_mismatch_flag_off_for_nonvoted_admission(self) -> None:
        """admission=all runs get UNKNOWN, not flagged as mismatch."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "all"
        fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.finalize()

        assert not fp.admission_realized_mismatch
        assert "NON_GATED_ADMISSION" in fp.mismatch_reason

    def test_mismatch_flag_off_for_insufficient_admits(self) -> None:
        """Fewer than 20 admits: mismatch judgment is UNKNOWN."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        for i in range(5):
            fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.finalize()

        assert not fp.admission_realized_mismatch
        assert "INSUFFICIENT_ADMITS" in fp.mismatch_reason

    def test_mismatch_flag_fires_on_force_dominated(self) -> None:
        """78% force-owned admits (>70% threshold) fires mismatch."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        fp.max_gap = 10

        # 78 max_gap (force), 22 pose_delta -> 78% owned by force
        for _ in range(78):
            fp.record_admit("max_gap", gap=10, forced_hard=True)
        for _ in range(22):
            fp.record_admit("pose_delta", gap=5, forced_hard=False)

        fp.finalize()

        assert fp.admission_realized_mismatch
        assert "FORCE_DOMINATED" in fp.mismatch_reason
        assert "78.0%" in fp.mismatch_reason or "0.78" in fp.mismatch_reason.lower()

    def test_mismatch_flag_off_when_force_below_threshold(self) -> None:
        """Force-owned below 70% threshold: no mismatch."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        fp.max_gap = 10

        # 50 max_gap (force), 50 pose_delta -> 50% owned by force
        for _ in range(50):
            fp.record_admit("max_gap", gap=10, forced_hard=True)
        for _ in range(50):
            fp.record_admit("pose_delta", gap=5, forced_hard=False)

        fp.finalize()

        assert not fp.admission_realized_mismatch
        assert "NOMINAL_POLICY_REALIZED" in fp.mismatch_reason


class TestAdmissionFingerprintSerialization:
    """Serialization to JSON: all fields present and correct."""

    def test_to_dict_schema_version(self) -> None:
        """to_dict() includes schema_version=2.

        Bumped 1 -> 2 by issue #823, which appended the ``admission_gate``
        block. Every key this test file already asserts on keeps its
        meaning; the bump exists so a consumer can tell whether the gate
        block is expected to be present at all.
        """
        fp = AdmissionFingerprint()
        d = fp.to_dict()

        assert d["schema_version"] == 2

    def test_to_dict_preserves_legacy_realized_rate_basis(self) -> None:
        fp = AdmissionFingerprint()
        assert fp.to_dict()["realized_admit_fps_legacy_basis"] == (
            "nominal_input_fps_over_total_source_frames"
        )

    def test_to_dict_all_fields_present(self) -> None:
        """to_dict() includes all required fields."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.record_reject("no_motion")
        fp.max_gap = 10
        fp.target_admit_fps = 2.0
        fp.input_fps = 30.0
        fp.finalize()

        d = fp.to_dict()

        required_fields = [
            "schema_version",
            "nominal_policy",
            "candidates_evaluated",
            "admitted_total",
            "rejected_total",
            "admit_owners",
            "reject_owners",
            "force_hard_count",
            "force_soft_count",
            "force_share",
            "gate_bypasses",
            "gap_histogram",
            "gaps_at_max_gap",
            "gaps_at_max_gap_share",
            "max_gap",
            "target_admit_fps",
            "input_fps",
            "admission_realized_mismatch",
            "mismatch_reason",
        ]
        for field in required_fields:
            assert field in d, f"Missing field: {field}"

    def test_to_dict_shares_sum_to_one(self) -> None:
        """force_share and gaps_at_max_gap_share are valid fractions."""
        fp = AdmissionFingerprint()
        fp.max_gap = 10
        for _ in range(50):
            fp.record_admit("max_gap", gap=10, forced_hard=True)
        for _ in range(50):
            fp.record_admit("pose_delta", gap=5, forced_hard=False)
        fp.finalize()

        d = fp.to_dict()

        assert 0.0 <= d["force_share"] <= 1.0
        assert 0.0 <= d["gaps_at_max_gap_share"] <= 1.0

    def test_to_dict_nominal_policy_preserved(self) -> None:
        """Nominal policy string is preserved in serialization."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        d = fp.to_dict()

        assert d["nominal_policy"] == "hybrid_keyframe"


class TestAdmissionFingerprintEdgeCases:
    """Edge cases: no admits, no rejects, empty fingerprint."""

    def test_empty_fingerprint(self) -> None:
        """Empty fingerprint serializes cleanly."""
        fp = AdmissionFingerprint()
        d = fp.to_dict()

        assert d["admitted_total"] == 0
        assert d["rejected_total"] == 0
        assert d["admit_owners"] == {}
        assert d["reject_owners"] == {}

    def test_no_admits_no_crash(self) -> None:
        """Fingerprint with only rejects doesn't crash on finalize()."""
        fp = AdmissionFingerprint()
        fp.nominal_policy = "hybrid_keyframe"
        for _ in range(100):
            fp.record_reject("no_motion")
        fp.finalize()

        assert fp.admitted_total == 0
        assert fp.rejected_total == 100
        # Should not crash; realized metrics left as None/empty

    def test_realized_source_frames_per_admit_exact(self) -> None:
        """Realized source frames/admit is mean of gap histogram."""
        fp = AdmissionFingerprint()
        fp.max_gap = 10
        # 3 admits at gaps [10, 20, 30] -> mean = 20
        fp.record_admit("max_gap", gap=10, forced_hard=True)
        fp.record_admit("max_gap", gap=20, forced_hard=True)
        fp.record_admit("max_gap", gap=30, forced_hard=True)

        fp.finalize(total_source_frames=60)

        assert fp.realized_source_frames_per_admit == 20.0


class TestAdmissionFingerprintRecordCandidate:
    """Candidate evaluation counter."""

    def test_candidates_evaluated_incremented(self) -> None:
        """record_candidate increments counter."""
        fp = AdmissionFingerprint()
        for _ in range(100):
            fp.record_candidate()

        assert fp.candidates_evaluated == 100

    def test_admit_and_reject_sum_to_evaluated(self) -> None:
        """admitted + rejected <= candidates (unless pregate cuts before record_candidate)."""
        fp = AdmissionFingerprint()
        for _ in range(50):
            fp.record_candidate()
            fp.record_admit("max_gap", gap=10, forced_hard=True)
        for _ in range(30):
            fp.record_candidate()
            fp.record_reject("no_motion")

        assert fp.candidates_evaluated == 80
        assert fp.admitted_total + fp.rejected_total == 80
