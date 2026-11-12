"""Issue #823 admission-gate aggregation (the #947 manipulation check).

These tests pin the gate's *classification*, not its thresholds' wisdom: the
question #947 asked, and #823 carries forward, is whether a classical
admission arm is materially owned by real motion/coverage evidence or by a
periodic starvation backstop wearing the policy's name. Before this block
existed that question could only be answered by hand-parsing
`keyframe_admission_log.jsonl`, so a finished run could not state its own
verdict -- which is why #792's 78%-force-dominated finding took a bespoke
investigation to surface.
"""

from __future__ import annotations

import json

import pytest

from eligsir.utils.admission_fingerprint import AdmissionFingerprint

# The thresholds are deliberately NOT imported by name. The gate block echoes
# both of them into its own payload (`threshold_backstop_owner_share_max`,
# `min_admits_for_verdict`) precisely so a consumer never has to reach into
# the module for them, and asserting against the echoed values tests the
# contract a report reader actually sees. Importing them here would also make
# them "public symbols imported by tests with no other production reference"
# under `scripts/check_production_reachability.py`'s unwired-primitive rule.
_BACKSTOP_OWNER_SHARE_MAX = 0.50
_MIN_ADMITS_FOR_GATE = 20


class _Decision:
    """Minimal stand-in for the #946 fields of a KeyframeDecision."""

    def __init__(self, trigger=None, motion_trigger=None, age_s=None, gap=None,
                 soft=False, hard=False, basis=None, reason="pose_delta"):
        if trigger is not None:
            self.trace = {"candidate_trigger": trigger}
        if motion_trigger is not None:
            self.motion_trigger = motion_trigger
        if age_s is not None:
            self.starvation_age_source_s = age_s
        if gap is not None:
            self.starvation_age_source_frames = gap
        self.soft_backstop_triggered = soft
        self.hard_backstop_triggered = hard
        if basis is not None:
            self.starvation_age_basis = basis
        self.reason = reason


def _run(fp, decisions):
    for d, admitted in decisions:
        fp.record_candidate()
        if admitted:
            fp.record_admit(reason=d.reason, gap=int(getattr(d, "starvation_age_source_frames", 0) or 0))
        else:
            fp.record_reject(d.reason)
        fp.record_backstop_state(d, admitted)
    fp.finalize()
    return fp.to_dict()["admission_gate"]


def test_backstop_dominated_arm_fails_the_gate():
    """#792's measured failure mode: the policy name says motion, the
    admissions say max_gap."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = [
        (_Decision(trigger="soft_max_gap", motion_trigger="none", age_s=0.33,
                   gap=10, soft=True, basis="source_timestamp", reason="max_gap"), True)
        for _ in range(78)
    ] + [
        (_Decision(trigger="normal", motion_trigger="translation", age_s=0.20,
                   gap=6, basis="source_timestamp"), True)
        for _ in range(22)
    ]
    gate = _run(fp, decisions)
    assert gate["verdict"] == "BACKSTOP_DOMINATED"
    assert gate["backstop_owner_share"] == pytest.approx(0.78)
    assert gate["evidence_owner_share"] == pytest.approx(0.22)
    # The gate must echo its own threshold, and it must be the #947 bar.
    assert gate["threshold_backstop_owner_share_max"] == _BACKSTOP_OWNER_SHARE_MAX
    assert gate["min_admits_for_verdict"] == _MIN_ADMITS_FOR_GATE


def test_evidence_owned_arm_passes_the_gate():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = [
        (_Decision(trigger="normal", motion_trigger="translation", age_s=0.2, gap=6), True)
        for _ in range(80)
    ] + [
        (_Decision(trigger="soft_max_gap", motion_trigger="none", age_s=0.33,
                   gap=10, soft=True, reason="max_gap"), True)
        for _ in range(20)
    ]
    gate = _run(fp, decisions)
    assert gate["verdict"] == "CLASSICAL_LIVE"
    assert gate["backstop_owner_share"] == pytest.approx(0.20)


def test_motion_ownership_splits_translation_from_rotation():
    """`reason == "pose_delta"` cannot distinguish these; #823 requires it."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = (
        [(_Decision(trigger="normal", motion_trigger="translation", gap=5), True)] * 50
        + [(_Decision(trigger="normal", motion_trigger="rotation", gap=5), True)] * 30
        + [(_Decision(trigger="normal", motion_trigger="both", gap=5), True)] * 20
    )
    gate = _run(fp, decisions)
    m = gate["motion_owner_decomposition"]
    assert m["n_motion_admits"] == 100
    assert m["translation_only"] == pytest.approx(0.50)
    assert m["rotation_only"] == pytest.approx(0.30)
    assert m["both"] == pytest.approx(0.20)


def test_hard_ceiling_counts_toward_the_backstop_share():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = (
        [(_Decision(trigger="hard_starvation_ceiling", motion_trigger="none",
                    gap=30, hard=True, reason="max_gap"), True)] * 60
        + [(_Decision(trigger="normal", motion_trigger="translation", gap=5), True)] * 40
    )
    gate = _run(fp, decisions)
    assert gate["backstop_owner_share"] == pytest.approx(0.60)
    assert gate["verdict"] == "BACKSTOP_DOMINATED"
    assert gate["hard_backstop_raw_crossings"] == 60


def test_novelty_force_counts_as_evidence_not_backstop():
    """#934's novelty force fires because evidence is PRESENT. Counting it
    as a starvation relief would make a successful selector fail its own
    manipulation check."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = [(_Decision(trigger="novelty_force", motion_trigger="none",
                            gap=4, reason="novelty_admit"), True)] * 100
    gate = _run(fp, decisions)
    assert gate["evidence_owner_share"] == pytest.approx(1.0)
    assert gate["verdict"] == "CLASSICAL_LIVE"


def test_fixed_stride_control_purity():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "fixed_stride"
    decisions = [(_Decision(motion_trigger=None, reason="fixed_stride"), True)] * 50
    gate = _run(fp, decisions)
    assert gate["verdict"] == "CONTROL_PURE"
    assert gate["control_purity"] == pytest.approx(1.0)


def test_fixed_stride_control_leak_is_detected():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "fixed_stride"
    decisions = (
        [(_Decision(reason="fixed_stride"), True)] * 45
        + [(_Decision(reason="max_gap"), True)] * 5
    )
    gate = _run(fp, decisions)
    assert gate["verdict"] == "CONTROL_LEAKED"


def test_insufficient_admits_never_reads_as_a_pass():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = [(_Decision(trigger="normal", motion_trigger="translation", gap=5), True)] * (
        _MIN_ADMITS_FOR_GATE - 1
    )
    gate = _run(fp, decisions)
    assert gate["verdict"] == "INSUFFICIENT_ADMITS"


def test_pre_946_run_grades_not_measured_rather_than_passing():
    """A run whose decisions carry no precedence trigger at all must not be
    silently classified as evidence-owned."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"

    class _Bare:
        reason = "pose_delta"

    for _ in range(50):
        fp.record_candidate()
        fp.record_admit(reason="pose_delta", gap=5)
        # No trace, no motion_trigger -- but `reason` still provides a
        # trigger bucket, so this exercises the unclassified path.
        fp.record_backstop_state(_Bare(), True)
    fp.finalize()
    gate = fp.to_dict()["admission_gate"]
    # "pose_delta" is not a ForcePath value, so it must land in
    # unclassified rather than being counted as evidence.
    assert gate["unclassified_owner_share"] == pytest.approx(1.0)
    assert gate["evidence_owner_share"] == pytest.approx(0.0)
    assert gate["backstop_owner_share"] == pytest.approx(0.0)
    # The verdict is the part that matters: a 0.0 backstop share here means
    # "unmeasured", and grading it CLASSICAL_LIVE would be a false pass.
    assert gate["verdict"] == "NOT_MEASURED"


def test_partially_measured_is_not_a_pass():
    """A sub-threshold backstop share cannot be certified while a material
    share of ownership is unknown -- the unknown remainder could itself be
    backstop-owned."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = (
        [(_Decision(trigger="normal", motion_trigger="translation", gap=5), True)] * 60
        + [(_Decision(trigger="soft_max_gap", motion_trigger="none", gap=10,
                      soft=True, reason="max_gap"), True)] * 20
        # 20% with an unrecognized trigger string.
        + [(_Decision(trigger="some_future_trigger", gap=7), True)] * 20
    )
    gate = _run(fp, decisions)
    assert gate["backstop_owner_share"] == pytest.approx(0.20)
    assert gate["unclassified_owner_share"] == pytest.approx(0.20)
    assert gate["verdict"] == "PARTIALLY_MEASURED"


def test_dominated_verdict_survives_unclassified_ownership():
    """A backstop share already at/over the bar is definitive: unclassified
    ownership can only push the share up, so it must not be able to
    downgrade a real failure to PARTIALLY_MEASURED."""
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = (
        [(_Decision(trigger="soft_max_gap", motion_trigger="none", gap=10,
                    soft=True, reason="max_gap"), True)] * 55
        + [(_Decision(trigger="some_future_trigger", gap=7), True)] * 45
    )
    gate = _run(fp, decisions)
    assert gate["verdict"] == "BACKSTOP_DOMINATED"


def test_trigger_ages_are_recorded_in_both_domains():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    decisions = [
        (_Decision(trigger="normal", motion_trigger="translation",
                   age_s=0.1 * (i + 1), gap=i + 1), True)
        for i in range(30)
    ]
    gate = _run(fp, decisions)
    assert gate["trigger_age_source_s"]["n"] == 30
    assert gate["trigger_age_source_frames"]["n"] == 30
    assert gate["trigger_age_source_s"]["max"] == pytest.approx(3.0)


def test_gate_block_is_json_serializable_and_versioned():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "hybrid_keyframe"
    _run(fp, [(_Decision(trigger="normal", motion_trigger="both", gap=3), True)] * 25)
    payload = fp.to_dict()
    assert payload["schema_version"] == 2
    json.dumps(payload)  # must not raise


def test_ungated_admission_is_not_applicable():
    fp = AdmissionFingerprint()
    fp.nominal_policy = "all"
    gate = _run(fp, [(_Decision(reason="all"), True)] * 40)
    assert gate["verdict"] == "NOT_APPLICABLE_UNGATED"
