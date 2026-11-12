"""Unit tests for issue #792 admission preflight detection.

Tests the pure function check_admission_preflight() which detects when
hybrid_keyframe + velocity_adaptive is configured such that max_gap
preempts the adaptive motion criterion.
"""
import pytest

from eligsir.mapping.scheduler.admission_preflight import (
    check_admission_preflight,
    AdmissionPreflightResult,
)
from scripts.audit_admission_mismatch import (
    _resolve_arm,
    _classify_tokens,
    _cli_present,
    classify_admission_config,
    compose_arm_tokens,
    BAG_SCRIPT_ADMISSION_HARDCODE,
)
import shlex


def test_default_config_triggers_mismatch():
    """Default config: max_gap=10, input_fps=30, target_admit_fps=2 -> mismatch."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=False,
    )
    assert result.mismatch is True
    assert result.verdict == "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION"
    assert result.frames_between_admits == pytest.approx(15.0)
    assert result.warning_text is not None
    assert "degenerate" in result.warning_text.lower()


def test_admission_mode_all_not_applicable():
    """admission_mode='all' -> NOT_APPLICABLE_MODE, no mismatch."""
    result = check_admission_preflight(
        admission_mode="all",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=False,
    )
    assert result.mismatch is False
    assert result.verdict == "NOT_APPLICABLE_MODE"
    assert result.warning_text is None


def test_velocity_adaptive_off_not_applicable():
    """velocity_adaptive=False -> NOT_APPLICABLE_ADAPTIVE_OFF, no mismatch."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=False,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=False,
    )
    assert result.mismatch is False
    assert result.verdict == "NOT_APPLICABLE_ADAPTIVE_OFF"
    assert result.warning_text is None


def test_max_gap_equals_frames_between_no_mismatch():
    """max_gap = frames_between_admits -> no mismatch (boundary)."""
    # input_fps=30, target_admit_fps=3 -> frames_between = 10
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=3.0,
        max_gap=10,
        max_gap_require_novelty=False,
    )
    assert result.frames_between_admits == pytest.approx(10.0)
    assert result.mismatch is False
    assert result.verdict == "OK"
    assert result.warning_text is None


def test_max_gap_greater_than_frames_between_no_mismatch():
    """max_gap > frames_between_admits -> no mismatch."""
    # input_fps=30, target_admit_fps=2 -> frames_between = 15
    # max_gap=20 > 15 -> no mismatch
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=20,
        max_gap_require_novelty=False,
    )
    assert result.frames_between_admits == pytest.approx(15.0)
    assert result.mismatch is False
    assert result.verdict == "OK"


def test_target_admit_fps_divide_by_zero_clamping():
    """target_admit_fps very close to zero -> clamped to 0.01, no crash."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=0.001,  # very small
        max_gap=10,
        max_gap_require_novelty=False,
    )
    # Should clamp to 0.01: frames_between = 30 / 0.01 = 3000
    assert result.frames_between_admits == pytest.approx(3000.0)
    assert result.verdict == "DIVIDE_BY_ZERO"
    assert result.warning_text is not None


def test_novelty_gating_softens_warning():
    """max_gap_require_novelty=True softens the warning."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=True,
    )
    assert result.mismatch is True
    assert result.verdict == "NOT_APPLICABLE_NOVELTY_GATED"
    assert result.warning_text is not None
    assert "mitigated" in result.warning_text.lower()
    assert "degenerate" not in result.warning_text.lower()


def test_computed_frames_between_matches_keyframe_formula():
    """Verify frames_between_admits computation matches keyframes.py:591."""
    # Test a few cases to ensure the formula is correct.
    cases = [
        (30.0, 2.0, 15.0),    # default config
        (30.0, 3.0, 10.0),    # tighter admission
        (60.0, 2.0, 30.0),    # higher input fps
        (30.0, 0.5, 60.0),    # slower admission target
    ]
    for input_fps, target_admit_fps, expected_frames_between in cases:
        result = check_admission_preflight(
            admission_mode="hybrid_keyframe",
            velocity_adaptive=True,
            input_fps=input_fps,
            target_admit_fps=target_admit_fps,
            max_gap=10,
        )
        assert result.frames_between_admits == pytest.approx(expected_frames_between)


def test_iphone_lidar_profile_config():
    """Test the iPhone LiDAR profile config which sets input_fps=60 and max_gap=20."""
    # From _CAMERA_PROFILES["iphone_lidar"]:
    # input_fps=60, max_gap=20, max_gap_require_novelty=True, target_admit_fps=2 (default)
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=60.0,
        target_admit_fps=2.0,
        max_gap=20,
        max_gap_require_novelty=True,
    )
    # frames_between = 60 / 2 = 30; max_gap=20 < 30 -> mismatch
    assert result.frames_between_admits == pytest.approx(30.0)
    assert result.mismatch is True
    # But novelty_gating=True -> NOT_APPLICABLE_NOVELTY_GATED
    assert result.verdict == "NOT_APPLICABLE_NOVELTY_GATED"
    assert "mitigated" in result.warning_text.lower()


# --------------------------------------------------------------------------
# Issue #928: honest, conditional wording once ``coverage_gain_gate`` is
# supplied. Omitting it (every test above) must reproduce the exact pre-#928
# wording byte-for-byte -- pinned by the two tests just above, unchanged.
# --------------------------------------------------------------------------

def test_novelty_gate_definitely_inert_without_coverage_gain_gate():
    """max_gap_require_novelty=True but coverage_gain_gate=False is
    STATICALLY knowable to be inert at preflight time (the gate needs both
    flags -- see keyframes.py's max_gap_require_novelty docstring). This must
    NOT be reported as mitigated."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=True,
        coverage_gain_gate=False,
    )
    assert result.mismatch is True
    assert result.coverage_gain_gate is False
    # Honest verdict: exactly as unmitigated as the plain preempts-adaptive-
    # motion case, since the novelty gate genuinely cannot run.
    assert result.verdict == "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION"
    assert "mitigated" not in result.warning_text.lower()
    assert "CANNOT evaluate" in result.warning_text
    assert "gate_disabled" in result.warning_text


def test_novelty_gate_requested_not_confirmed_with_coverage_gain_gate_on():
    """max_gap_require_novelty=True AND coverage_gain_gate=True is a REQUEST
    the preflight cannot confirm (occupancy_fn/load_depth_m wiring happens
    later, at KeyframeAdmitter construction, which this earlier check cannot
    see) -- the old 'mitigated' claim must be gone."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=True,
        coverage_gain_gate=True,
    )
    assert result.mismatch is True
    assert result.coverage_gain_gate is True
    assert result.verdict == "NOT_APPLICABLE_NOVELTY_GATED"
    assert "mitigated" not in result.warning_text.lower()
    assert "REQUEST" in result.warning_text
    assert "novelty_gate_outcome" in result.warning_text


def test_coverage_gain_gate_omitted_is_byte_identical_to_pre_928():
    """Explicitly re-pin: omitting the new parameter (the default, ``None``)
    reproduces test_novelty_gating_softens_warning/test_iphone_lidar_profile_config's
    exact old wording -- no behavior change for any caller that doesn't know
    about issue #928's new parameter."""
    result_omitted = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=True,
    )
    result_explicit_none = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=True,
        coverage_gain_gate=None,
    )
    assert result_omitted.warning_text == result_explicit_none.warning_text
    assert result_omitted.verdict == "NOT_APPLICABLE_NOVELTY_GATED"
    assert "mitigated" in result_omitted.warning_text.lower()
    assert result_omitted.coverage_gain_gate is None


def test_result_is_dataclass():
    """Verify the result is a proper dataclass with expected fields."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
    )
    assert isinstance(result, AdmissionPreflightResult)
    assert hasattr(result, "frames_between_admits")
    assert hasattr(result, "max_gap")
    assert hasattr(result, "mismatch")
    assert hasattr(result, "verdict")
    assert hasattr(result, "max_gap_require_novelty")
    assert hasattr(result, "warning_text")


# --------------------------------------------------------------------------
# Corpus-audit parsing tests (issue #792 false-negative fix). These test the
# REAL parsing/resolution functions in scripts/audit_admission_mismatch.py
# against realistic spec fragments, not AST/string matching.
# --------------------------------------------------------------------------


def test_flag_only_in_base_args_is_detected():
    """A spec whose hybrid_keyframe flag lives only in base.base_args (never
    repeated in any arm's extra_args) must still classify as hybrid_keyframe
    with the default mismatch (max_gap=10 < frames_between=15). Since issue
    #792/#794's 2026-08-15 fix (streaming_keyframe_max_gap_require_novelty
    default flipped False->True), an unspecified arm now inherits the
    novelty-gated variant of the mismatch, not the fully vulnerable one."""
    spec = {
        "base": {
            "type": "replay",
            "base_args": "--online_mode --streaming_frame_admission hybrid_keyframe --cap_max 50000",
        },
        "arms": [{"id": "ctrl", "extra_args": ""}],
    }
    job_type, combined, _scene = _resolve_arm(spec, spec["arms"][0])
    assert job_type == "replay"
    tokens = shlex.split(combined)
    assert "hybrid_keyframe" in combined
    parse_ok, classification, detail = _classify_tokens(tokens)
    assert parse_ok
    assert classification == "mismatch_with_novelty_gating"
    assert "max_gap=10" in detail


def test_arm_extra_args_overrides_base_args_on_conflict():
    """Per ablator's spec.py expand_spec(): arm.extra_args is appended AFTER
    base_args, so an arm that explicitly widens max_gap past
    frames_between_admits must resolve to no_mismatch, even though the
    spec's own base_args alone would mismatch."""
    spec = {
        "base": {
            "type": "replay",
            "base_args": "--streaming_frame_admission hybrid_keyframe",
        },
        "arms": [
            {"id": "mismatched", "extra_args": ""},
            {"id": "fixed", "extra_args": "--streaming_keyframe_max_gap 20"},
        ],
    }
    # Arm 0: no override -> inherits the default max_gap=10 mismatch, now
    # novelty-gated by default (issue #792/#794, 2026-08-15).
    job_type0, combined0, _ = _resolve_arm(spec, spec["arms"][0])
    _ok0, classification0, _detail0 = _classify_tokens(shlex.split(combined0))
    assert classification0 == "mismatch_with_novelty_gating"

    # Arm 1: extra_args raises max_gap to 20 >= frames_between=15 -> fixed.
    job_type1, combined1, _ = _resolve_arm(spec, spec["arms"][1])
    tokens1 = shlex.split(combined1)
    assert tokens1.index("hybrid_keyframe") < tokens1.index("--streaming_keyframe_max_gap")
    _ok1, classification1, _detail1 = _classify_tokens(tokens1)
    assert classification1 == "no_mismatch"


def test_arm_level_base_args_overrides_spec_base_args():
    """An arm's own `base_args` (not just `extra_args`) also overrides the
    spec-level `base.base_args` per ablator's expand_spec() -- e.g. an arm
    that switches admission mode back to `all` entirely."""
    spec = {
        "base": {
            "type": "replay",
            "base_args": "--streaming_frame_admission hybrid_keyframe",
        },
        "arms": [
            {"id": "override", "base_args": "--streaming_frame_admission all", "extra_args": ""},
        ],
    }
    _job_type, combined, _ = _resolve_arm(spec, spec["arms"][0])
    assert combined == "--streaming_frame_admission all"
    _ok, classification, _detail = _classify_tokens(shlex.split(combined))
    assert classification == "not_hybrid"


def test_bag_type_arm_silently_realizes_hybrid_keyframe():
    """A bag-type spec whose own args say NOTHING about admission still
    realizes hybrid_keyframe at launch time, because
    human_launch_scripts/05_run_offline_rosbag.sh (and 06_run_k8s_bag.sh)
    hardcode --streaming_frame_admission hybrid_keyframe into ADMIT_ARGS,
    placed on the command line before the job's own extra_args. This is
    invisible to a naive audit that only reads the spec's JSON text."""
    spec = {
        "base": {
            "type": "bag",
            "base_args": "--online_mode --cap_max 50000",
        },
        "arms": [{"id": "smoke", "extra_args": ""}],
    }
    job_type, combined, _scene = _resolve_arm(spec, spec["arms"][0])
    assert job_type == "bag"
    # The spec's own text mentions nothing about admission.
    assert "hybrid_keyframe" not in combined
    assert not _cli_present(shlex.split(combined), "streaming_frame_admission")

    # Use the SAME token-composition function audit_specs() itself calls --
    # not a paraphrase of it -- so a precedence regression there fails here.
    job_type2, tokens = compose_arm_tokens(spec, spec["arms"][0])
    assert job_type2 == "bag"
    assert tokens[: len(shlex.split(BAG_SCRIPT_ADMISSION_HARDCODE))] == shlex.split(BAG_SCRIPT_ADMISSION_HARDCODE)
    parse_ok, classification, _detail = _classify_tokens(tokens)
    assert parse_ok
    # Novelty-gated by default since issue #792/#794's 2026-08-15 fix.
    assert classification == "mismatch_with_novelty_gating"


def test_bag_type_arm_explicit_override_beats_hardcode():
    """A bag-type arm that explicitly sets --streaming_frame_admission all
    in its own extra_args must win over the script's hardcode, because the
    hardcode is placed BEFORE the job's own extra_args on the real command
    line (argparse last-token-wins)."""
    spec = {
        "base": {"type": "bag", "base_args": ""},
        "arms": [{"id": "explicit_all", "extra_args": "--streaming_frame_admission all"}],
    }
    job_type, tokens = compose_arm_tokens(spec, spec["arms"][0])
    assert job_type == "bag"
    parse_ok, classification, _detail = _classify_tokens(tokens)
    assert parse_ok
    assert classification == "not_hybrid"


def test_classify_admission_config_delegates_to_preflight_checker():
    """classify_admission_config must never duplicate the mismatch
    condition -- it should agree exactly with check_admission_preflight
    for the same inputs."""
    result = check_admission_preflight(
        admission_mode="hybrid_keyframe",
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_fps=2.0,
        max_gap=10,
        max_gap_require_novelty=False,
    )
    classification, _detail = classify_admission_config(
        "hybrid_keyframe", True, 30.0, 2.0, 10, False,
    )
    assert result.verdict == "MAX_GAP_PREEMPTS_ADAPTIVE_MOTION"
    assert classification == "mismatch_vulnerable"
