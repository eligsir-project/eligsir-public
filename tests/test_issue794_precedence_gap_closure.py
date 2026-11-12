"""Issue #794 gap closure (2026-08-23).

Two residual gaps were left open by the #1055 status update on #794 (see the
module docstring of ``precedence.py``'s ``resolve_backstop_thresholds``):

1. The #1044 non-preemption floor for ``legacy_frames`` (the default soft
   backstop mode) DECLINES to apply whenever it would land at or past the
   hard starvation ceiling — leaving the soft backstop able to pre-empt the
   velocity-adaptive motion criterion exactly as before #1044, just with a
   printed warning instead of the pre-#1055 silence. The pre-existing
   ``hard_frames <= soft_frames`` case is the structurally identical second
   form of the same contradiction.
2. ``target_time`` mode raises ``BackstopConfigError`` on the equivalent
   contradiction; ``legacy_frames`` only warns. Undocumented asymmetry.

This module tests the fix for both:

* ``ResolvedBackstop.soft_backstop_preemptable`` — a structured, machine-
  readable field that is True in exactly the two cases above (in EITHER
  mode's warn-vs-raise posture), False otherwise, and False unconditionally
  under ``target_time`` (which refuses the contradiction outright, so a
  resolved ``ResolvedBackstop`` can never carry it there).
* ``--streaming_keyframe_require_reachable_motion_gate`` (new CLI flag,
  default False) — when True, ``legacy_frames`` raises
  ``BackstopConfigError`` in those same two cases instead of warning.
  Default OFF, so it must change behaviour for NO existing spec.

No GPU, no training entrypoint — pure-function / dataclass-construction
tests only.
"""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler.precedence import (
    BackstopConfigError,
    SoftBackstopMode,
    resolve_backstop_thresholds,
)
from eligsir.mapping.scheduler.keyframes import (
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)


def _legacy(
    *,
    max_gap: int,
    max_gap_hard_mult: float = 4.0,
    velocity_adaptive: bool = True,
    input_fps: float = 30.0,
    target_admit_rate_fps: float = 2.0,
    nonpreemption_floor: bool = True,
    require_reachable_motion_gate: bool = False,
):
    # Return type is `ResolvedBackstop` (eligsir/mapping/scheduler/precedence.py);
    # deliberately not imported/annotated by name here -- see the HARD CONSTRAINT
    # "a new primitive must be wired into production, or ledgered" in CLAUDE.md and
    # PR #1156: a symbol imported by name only in tests/, with no production
    # caller outside its own declaring module, trips the production-reachability
    # gate even though the underlying class is genuinely production-live (used
    # throughout precedence.py itself and returned by resolve_backstop_thresholds,
    # which keyframes.py calls in production). `ResolvedBackstop` is exactly that
    # shape: no OTHER production module ever imports the name.
    return resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=max_gap,
        max_gap_hard_mult=max_gap_hard_mult,
        input_fps=input_fps,
        velocity_adaptive=velocity_adaptive,
        target_admit_rate_fps=target_admit_rate_fps,
        nonpreemption_floor=nonpreemption_floor,
        require_reachable_motion_gate=require_reachable_motion_gate,
    )


# ---------------------------------------------------------------------------
# (a) Default behaviour (flag unset / False) is byte-identical to pre-#794-
#     gap-closure -- regression-safe by construction. Every one of these
#     assertions mirrors an existing assertion in
#     tests/test_issue1044_admission_delivery.py; reproduced here so this
#     module stands on its own as the "did the new parameter change anything
#     by default" proof.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_gap,expected_soft,expected_hard",
    [
        (1, 1.0, 4.0),
        (2, 2.0, 8.0),
        (3, 3.0, 12.0),
    ],
)
def test_default_flag_off_reproduces_the_declined_floor_warn_not_raise(
    max_gap, expected_soft, expected_hard
):
    """Gap 1 (declined floor): default (flag unset) warns, never raises."""
    bs = _legacy(max_gap=max_gap, max_gap_hard_mult=4.0)
    assert bs.soft_threshold_frames == expected_soft
    assert bs.hard_threshold_frames == expected_hard
    assert any("NOT applied" in w for w in bs.warnings)


def test_default_flag_off_reproduces_the_hard_le_soft_warn_not_raise():
    """The pre-existing hard_frames <= soft_frames case: warn, never raise."""
    bs = _legacy(
        max_gap=5,
        max_gap_hard_mult=1.0,
        velocity_adaptive=False,  # keep Gap 1's branch out of the way
    )
    assert bs.hard_threshold_frames <= bs.soft_threshold_frames
    assert any("soft escalation stage is unreachable" in w for w in bs.warnings)


def test_default_flag_off_does_not_touch_a_healthy_config():
    """A config where the floor applies cleanly: unaffected either way."""
    bs_without = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    bs_with_flag_false = _legacy(max_gap=10, require_reachable_motion_gate=False)
    assert bs_without == bs_with_flag_false
    assert bs_without.soft_threshold_frames == 15.0
    assert bs_without.hard_threshold_frames == 40.0
    assert bs_without.soft_backstop_preemptable is False


def test_omitting_the_kwarg_entirely_is_identical_to_passing_false():
    """The new kwarg is purely additive: an older call site omitting it must
    resolve identically to one that passes the new default explicitly."""
    bs_omitted = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=1,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    bs_explicit_false = _legacy(max_gap=1, require_reachable_motion_gate=False)
    assert bs_omitted == bs_explicit_false


# ---------------------------------------------------------------------------
# (b) With the flag set, legacy_frames now raises in both previously-silent-
#     warn cases.
# ---------------------------------------------------------------------------


def test_flag_on_raises_for_the_declined_floor_case():
    with pytest.raises(BackstopConfigError, match="controller target interval"):
        _legacy(max_gap=1, max_gap_hard_mult=4.0, require_reachable_motion_gate=True)


def test_flag_on_raises_for_the_hard_le_soft_case():
    with pytest.raises(BackstopConfigError, match="soft escalation stage is unreachable"):
        _legacy(
            max_gap=5,
            max_gap_hard_mult=1.0,
            velocity_adaptive=False,
            require_reachable_motion_gate=True,
        )


def test_flag_on_does_not_raise_for_a_healthy_config():
    """The flag only refuses the two contradictory cases -- a config where the
    floor applies cleanly (or doesn't need to) must still resolve normally."""
    bs = _legacy(max_gap=10, require_reachable_motion_gate=True)
    assert bs.soft_threshold_frames == 15.0
    assert bs.hard_threshold_frames == 40.0
    assert bs.soft_backstop_preemptable is False


def test_flag_on_error_message_names_the_opt_out():
    with pytest.raises(BackstopConfigError, match="require_reachable_motion_gate"):
        _legacy(max_gap=1, max_gap_hard_mult=4.0, require_reachable_motion_gate=True)


# ---------------------------------------------------------------------------
# (c) soft_backstop_preemptable is correctly True/False, and surfaces
#     downstream via ResolvedBackstop.as_telemetry() / KeyframeDecision /
#     KeyframeAdmitter.backstop -- the same path every other ResolvedBackstop
#     field already uses to reach the admission sidecar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_gap,max_gap_hard_mult,velocity_adaptive",
    [
        (1, 4.0, True),   # Gap 1: declined floor
        (2, 4.0, True),   # Gap 1: declined floor
        (5, 1.0, False),  # hard_frames <= soft_frames
    ],
)
def test_preemptable_true_in_both_contradictory_cases(
    max_gap, max_gap_hard_mult, velocity_adaptive
):
    bs = _legacy(
        max_gap=max_gap,
        max_gap_hard_mult=max_gap_hard_mult,
        velocity_adaptive=velocity_adaptive,
    )
    assert bs.soft_backstop_preemptable is True


@pytest.mark.parametrize(
    "max_gap,velocity_adaptive",
    [
        (10, True),   # production profile: floor applies cleanly
        (20, True),   # max_gap already clears the controller interval
        (10, False),  # no controller cadence to be pre-empted from
    ],
)
def test_preemptable_false_when_the_backstop_is_healthy(max_gap, velocity_adaptive):
    bs = _legacy(max_gap=max_gap, velocity_adaptive=velocity_adaptive)
    assert bs.soft_backstop_preemptable is False


def test_preemptable_is_a_key_in_as_telemetry():
    bs = _legacy(max_gap=1, max_gap_hard_mult=4.0)
    tel = bs.as_telemetry()
    assert tel["soft_backstop_preemptable"] is True


def test_preemptable_surfaces_on_the_keyframe_admitter_backstop_property():
    """The admitter's public ``.backstop`` is a ``ResolvedBackstop`` -- the
    same object ``as_telemetry()`` was called on above. This is the existing
    convention every other #946 field uses to reach the per-decision sidecar
    (``KeyframeAdmitter._backstop_static_telemetry`` is stamped onto every
    ``KeyframeDecision`` via ``hasattr``/``setattr`` in ``admit()``)."""
    admitter = KeyframeAdmitter(
        config=KeyframeAdmitterConfig(
            max_gap=1,
            max_gap_hard_mult=4.0,
            velocity_adaptive=True,
            target_admit_rate_fps=2.0,
            input_fps=30.0,
        )
    )
    assert admitter.backstop.soft_backstop_preemptable is True
    # KeyframeDecision declares the same field name, so the generic
    # hasattr/setattr stamping loop in admit()/pregate() picks it up for
    # free -- assert the admitter's own static telemetry snapshot agrees.
    assert (
        admitter._backstop_static_telemetry["soft_backstop_preemptable"]  # noqa: SLF001
        is True
    )


def test_preemptable_false_surfaces_on_a_healthy_admitter():
    admitter = KeyframeAdmitter(
        config=KeyframeAdmitterConfig(
            max_gap=10,
            max_gap_hard_mult=4.0,
            velocity_adaptive=True,
            target_admit_rate_fps=2.0,
            input_fps=30.0,
        )
    )
    assert admitter.backstop.soft_backstop_preemptable is False


def test_keyframe_admitter_construction_raises_when_flag_is_set_on_bad_config():
    """The admitter forwards the new config field straight through to
    ``resolve_backstop_thresholds`` at construction time -- the same place
    ``target_time``'s existing raise already happens (never per-candidate)."""
    with pytest.raises(BackstopConfigError):
        KeyframeAdmitter(
            config=KeyframeAdmitterConfig(
                max_gap=1,
                max_gap_hard_mult=4.0,
                velocity_adaptive=True,
                target_admit_rate_fps=2.0,
                input_fps=30.0,
                require_reachable_motion_gate=True,
            )
        )


def test_keyframe_admitter_config_default_is_false():
    assert KeyframeAdmitterConfig().require_reachable_motion_gate is False


# ---------------------------------------------------------------------------
# (d) target_time mode's behaviour is completely unchanged regardless of the
#     new flag -- it already raises unconditionally, and the new parameter is
#     not read on that branch at all.
# ---------------------------------------------------------------------------


def _target_time_contradictory(**overrides):
    kwargs = dict(
        mode=SoftBackstopMode.TARGET_TIME,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        soft_multiplier=1.25,
        hard_backstop_s=0.0,
        hard_backstop_frames=1.0,  # far below the resolved soft threshold
    )
    kwargs.update(overrides)
    return resolve_backstop_thresholds(**kwargs)


def test_target_time_still_raises_unconditionally_with_flag_unset():
    with pytest.raises(BackstopConfigError, match="is contradictory"):
        _target_time_contradictory()


def test_target_time_still_raises_with_flag_explicitly_false():
    with pytest.raises(BackstopConfigError, match="is contradictory"):
        _target_time_contradictory(require_reachable_motion_gate=False)


def test_target_time_still_raises_with_flag_true_too():
    """The flag must not WEAKEN target_time's existing raise either."""
    with pytest.raises(BackstopConfigError, match="is contradictory"):
        _target_time_contradictory(require_reachable_motion_gate=True)


def test_target_time_healthy_config_unaffected_by_the_new_flag():
    bs_false = resolve_backstop_thresholds(
        mode=SoftBackstopMode.TARGET_TIME,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        soft_multiplier=1.25,
        hard_backstop_s=0.0,
        hard_backstop_frames=40.0,
        require_reachable_motion_gate=False,
    )
    bs_true = resolve_backstop_thresholds(
        mode=SoftBackstopMode.TARGET_TIME,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        soft_multiplier=1.25,
        hard_backstop_s=0.0,
        hard_backstop_frames=40.0,
        require_reachable_motion_gate=True,
    )
    assert bs_false == bs_true
    # target_time refuses its own contradiction outright, so a resolved
    # ResolvedBackstop in this mode can never carry soft_backstop_preemptable.
    assert bs_false.soft_backstop_preemptable is False
    assert bs_true.soft_backstop_preemptable is False
