"""Focused regression coverage for the issue #282/#346 sticky-recovered defect.

These tests are intentionally torch-free: both the severity grader and champion
promotion gate are offline correctness tooling and must stay runnable on the CPU
smoke surface.
"""

from eligsir.utils.champion_promotion import validate_report_contract
from eligsir.utils.seed_fade_severity import ended_healthy, grade_seed_fade


def _report(seed_fade):
    return {
        "eval_valid": True,
        "n_test": 8,
        "n_test_metrics": 8,
        "split_kind": "holdout",
        "eval_metadata": {
            "holdout_hash": "rule-hash",
            "realized_holdout_hash": "pool-hash",
            "scored_holdout_hash": "scored-hash",
        },
        "seed_fade": seed_fade,
    }


def test_ended_healthy_prefers_explicit_recovered_final():
    assert ended_healthy({"detected": True, "recovered": True, "recovered_final": False}) is False
    assert ended_healthy({"detected": True, "recovered": False, "recovered_final": True}) is True


def test_ended_healthy_accepts_fade_active_at_end_inverse():
    assert ended_healthy({"detected": True, "fade_active_at_end": True}) is False
    assert ended_healthy({"detected": True, "fade_active_at_end": False}) is True


def test_legacy_sticky_true_is_not_interpreted_as_final_health():
    assert ended_healthy({"detected": True, "recovered": True}) is None


def test_legacy_recovered_false_is_still_conclusively_unhealthy():
    assert ended_healthy({"detected": True, "recovered": False}) is False


def test_grader_escalates_when_run_ends_inside_fade_even_if_sticky_latch_is_true():
    graded = grade_seed_fade({
        "detected": True,
        "recovered": True,
        "recovered_final": False,
        "max_dip_db": 2.93,
    })
    assert graded["severity"] == "moderate"
    assert graded["confounded"] is True
    assert graded["end_state_verifiable"] is True
    assert any("fade active at end" in reason for reason in graded["severity_reasons"])


def test_grader_does_not_escalate_when_explicit_final_state_is_healthy():
    graded = grade_seed_fade({
        "detected": True,
        "recovered": True,
        "recovered_final": True,
        "max_dip_db": 2.93,
    })
    assert graded["severity"] == "minor"
    assert graded["confounded"] is False
    assert graded["end_state_verifiable"] is True


def test_grader_marks_legacy_sticky_true_end_state_unverifiable():
    graded = grade_seed_fade({
        "detected": True,
        "recovered": True,
        "max_dip_db": 2.93,
    })
    assert graded["severity"] == "minor"
    assert graded["end_state_verifiable"] is False
    assert any("unverifiable" in reason for reason in graded["severity_reasons"])


def test_promotion_blocks_explicit_unhealthy_final_state_despite_sticky_true():
    result = validate_report_contract(_report({
        "detected": True,
        "recovered": True,
        "recovered_final": False,
        "max_dip_db": 2.93,
    }))
    assert any("ENDED INSIDE AN ACTIVE FADE" in blocker for blocker in result.blockers)


def test_promotion_legacy_sticky_true_warns_instead_of_claiming_recovery():
    result = validate_report_contract(_report({
        "detected": True,
        "recovered": True,
        "max_dip_db": 2.93,
    }))
    assert not any("ACTIVE FADE" in blocker for blocker in result.blockers)
    assert any("UNVERIFIABLE" in warning for warning in result.warnings)


def test_resume_indeterminate_blocker_survives_final_health_fix():
    result = validate_report_contract({
        **_report({
            "detected": False,
            "indeterminate": True,
            "indeterminate_reasons": ["state not restored"],
        }),
        "seed_fade_indeterminate": True,
    })
    assert any("INDETERMINATE" in blocker for blocker in result.blockers)


def test_scored_holdout_gate_survives_final_health_fix():
    """Integration guard with #446: both promotion correctness gates must coexist."""
    report = _report({"detected": False})
    report["eval_metadata"].pop("scored_holdout_hash")
    result = validate_report_contract(report)
    assert any("scored_holdout_hash" in blocker for blocker in result.blockers)


def test_scored_holdout_override_api_survives_final_health_fix():
    """Also pin #446's extended validation API on the synthetic merge result."""
    report = _report({"detected": False})
    report["eval_metadata"].pop("scored_holdout_hash")
    result = validate_report_contract(report, allow_unverifiable_scored_subset=True)
    assert result.ok, result.blockers
