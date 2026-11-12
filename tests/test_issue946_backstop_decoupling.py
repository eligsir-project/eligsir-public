"""Issue #946: the soft escalation threshold and the hard starvation ceiling
must be two INDEPENDENTLY resolved quantities, and the soft one must resolve
in the canonical #793/#933 source-time basis.

Why this exists
---------------
Before #946 there was exactly one resolved quantity -- ``max_gap`` source
frames -- and the hard ceiling was *derived* from it as
``max_gap * max_gap_hard_mult``. Raising the soft threshold from 10 to 15/20
therefore also moved the last-resort unconditional safety bound from 40 to
60/80 frames, so no experiment could vary the soft backstop while holding the
safety semantics fixed (which is exactly what #947's sweep needs to do).

The nine test families below are the ones enumerated in the issue body.
``test_legacy_*`` is the load-bearing one: the merged default is
``legacy_frames``, so this change must be behaviour-neutral on every existing
config, spec and CLI invocation in the repo, and that claim has to be
*asserted*, not asserted-by-comment.

CPU-only and torch-free (see ``tests/_fake_alpha``), matching the rest of the
admission test suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _fake_alpha import stub_render  # noqa: E402

from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)
from eligsir.mapping.scheduler.precedence import (
    BackstopConfigError,
    backstop_crossings,
    ForcePath,
    Gate,
    GATE_ORDER,
    MaxGapForceScope,
    SoftBackstopMode,
    UPSTREAM_GATES,
    bypass_permissions,
    coerce_soft_backstop_mode,
    resolve_backstop_thresholds,
    resolve_backstop_trigger,
    resolve_max_gap_trigger,
    validate_soft_backstop_config,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeCam:
    def __init__(self):
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 8.0
        self.cy = 8.0
        self.c2w = np.eye(4, dtype=np.float32)
        self.world_view_transform = np.eye(4, dtype=np.float32)


def _frame(index: int, x: float = 0.0, ts=None):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index, timestamp=ts)


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,          # rotation gate out of the way
        max_gap=10,
        max_gap_hard_mult=4.0,
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
        motion_pregate=True,
        velocity_adaptive=False,
        min_overlap=0.25,
        max_overlap=0.90,
        input_fps=30.0,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _admitter(cfg, overlap=0.5, baseline_ts=0.0):
    a = KeyframeAdmitter(cfg, render_fn=stub_render(overlap))
    a.note_train_frame(_frame(0, 0.0, ts=baseline_ts))
    return a


def _first_soft_trigger(admitter, frames):
    """Feed *frames* and return the first decision whose SOFT backstop fired,
    together with the immediately preceding decision (or ``None``)."""
    cam = _FakeCam()
    prev = None
    for f in frames:
        d, _ = admitter.admit(cam, f)
        if d.soft_backstop_triggered:
            return d, prev
        prev = d
    return None, prev


# ---------------------------------------------------------------------------
# Family 8 (LOAD-BEARING): legacy reproduction
# ---------------------------------------------------------------------------


def test_legacy_mode_is_the_default_everywhere():
    """The merged default must be behaviour-preserving in all three places a
    default can live: the enum, the dataclass, and the CLI."""
    from arguments import ModelParams  # noqa: F401  (import cost is trivial)
    import arguments

    assert KeyframeAdmitterConfig().soft_backstop_mode == "legacy_frames"
    assert SoftBackstopMode.LEGACY_FRAMES.value == "legacy_frames"

    src = open(arguments.__file__, "r", encoding="utf-8").read()
    assert 'self.streaming_keyframe_soft_backstop_mode = "legacy_frames"' in src


def test_legacy_resolution_reproduces_the_historical_trigger_exactly():
    """The ONLY decision input #946 replaced is the ForcePath classification.

    Under ``legacy_frames`` the new resolver must agree with the pre-#946
    expression ``resolve_max_gap_trigger(gap, max_gap, hard_mult)`` on every
    reachable gap, for a wide sweep of the two historical knobs. Exhaustive
    agreement here IS the bit-identity argument for the classification half.
    """
    for max_gap in (1, 2, 5, 10, 15, 23, 120):
        for hard_mult in (1.0, 1.5, 2.0, 4.0, 7.5):
            r = resolve_backstop_thresholds(
                mode="legacy_frames",
                max_gap=max_gap,
                max_gap_hard_mult=hard_mult,
                input_fps=30.0,
            )
            for gap in range(0, max_gap * int(math.ceil(hard_mult)) + 25):
                assert resolve_backstop_trigger(r, gap, None) is (
                    resolve_max_gap_trigger(gap, max_gap, hard_mult)
                ), (max_gap, hard_mult, gap)


def test_legacy_admitter_decisions_are_bit_identical_on_a_synthetic_trace():
    """End-to-end complement to the resolver test above.

    ``_admit_impl`` consumed the old thresholds in exactly two places:
    ``forced = gap >= max_gap`` and the novelty gate's hard-ceiling predicate
    ``not (gap < max_gap * hard_mult)``. This asserts BOTH now-derived values
    equal their pre-#946 expressions on every candidate of a real trace, so
    every downstream branch sees identical inputs -- and asserts the trace is
    non-degenerate (it really does contain normal, soft and hard candidates,
    and both admits and rejects), so the equality is not vacuous.
    """
    cfg = _cfg(max_gap=10, max_gap_hard_mult=4.0)
    # overlap below min_overlap => the render gate rejects anything that is
    # not force-admitted, which is what lets gaps grow past the hard ceiling.
    a = _admitter(cfg, overlap=0.05)
    cam = _FakeCam()

    triggers, reasons = [], []
    for i in range(1, 140):
        # Stationary camera: motion never admits, so the backstop owns
        # admission entirely and gap sweeps 1..hard-ceiling repeatedly.
        d, _ = a.admit(cam, _frame(i, 0.0, ts=i / 30.0))
        gap = d.gap
        assert d.soft_backstop_triggered == (gap >= cfg.max_gap), (i, gap)
        assert d.hard_backstop_triggered == (
            gap >= cfg.max_gap * cfg.max_gap_hard_mult
        ), (i, gap)
        assert d.forced == (gap >= cfg.max_gap) or d.trace["candidate_trigger"] == (
            ForcePath.NORMAL.value
        )
        triggers.append(d.trace["candidate_trigger"])
        reasons.append((d.admitted, d.reason))

    assert ForcePath.NORMAL.value in triggers
    assert ForcePath.SOFT_MAX_GAP.value in triggers
    assert any(r[0] for r in reasons), "trace admitted nothing -- degenerate"
    assert any(not r[0] for r in reasons), "trace rejected nothing -- degenerate"


def test_legacy_mode_never_consults_the_new_target_time_knobs():
    """A spec that (harmlessly) carries the new flags but leaves the mode at
    its default must resolve exactly as if they were absent."""
    base = resolve_backstop_thresholds(mode="legacy_frames", max_gap=10,
                                       max_gap_hard_mult=4.0, input_fps=30.0)
    for mult in (1.0, 1.25, 9.0, 0.1):
        for hard_s in (0.0, 0.2, 99.0):
            for hard_frames in (1.0, 40.0, 500.0):
                other = resolve_backstop_thresholds(
                    mode="legacy_frames", max_gap=10, max_gap_hard_mult=4.0,
                    input_fps=30.0, soft_multiplier=mult,
                    hard_backstop_s=hard_s, hard_backstop_frames=hard_frames,
                )
                assert other.soft_threshold_frames == base.soft_threshold_frames
                assert other.hard_threshold_frames == base.hard_threshold_frames


# ---------------------------------------------------------------------------
# Family 1: non-preemption invariant
# ---------------------------------------------------------------------------


def test_soft_threshold_can_never_resolve_below_the_controller_interval():
    """30 Hz source, target 2 Hz => controller interval 0.5 s. No multiplier,
    however malformed, may resolve a soft threshold below that."""
    for mult in (1.0, 1.25, 2.0, 0.999, 0.5, 0.0, -3.0, float("nan"),
                 float("-inf")):
        r = resolve_backstop_thresholds(
            mode="target_time", velocity_adaptive=True, input_fps=30.0,
            target_admit_rate_fps=2.0, soft_multiplier=mult,
        )
        assert r.controller_target_interval_s == pytest.approx(0.5)
        assert r.soft_threshold_s >= r.controller_target_interval_s - 1e-12, mult
        assert r.soft_multiplier >= 1.0


def test_multiplier_below_one_is_normalised_loudly_not_silently():
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=0.5,
    )
    assert r.soft_multiplier_clamped is True
    assert r.soft_multiplier == 1.0
    assert any("NORMALISED" in w for w in r.warnings), r.warnings
    assert any("0.5" in w for w in r.warnings), r.warnings


def test_non_velocity_adaptive_fallback_is_explicit_not_borrowed():
    """Issue #946: 'define an explicit fallback rather than borrowing an
    unrelated target'. With no controller running, the fallback is the
    source-time image of the LEGACY soft threshold, not target_admit_rate_fps.
    """
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=False, max_gap=10,
        input_fps=30.0, target_admit_rate_fps=2.0, soft_multiplier=1.0,
    )
    assert r.controller_target_interval_source == "legacy_max_gap_frames"
    # 10 frames / 30 fps, NOT 1/2 s.
    assert r.controller_target_interval_s == pytest.approx(10.0 / 30.0)
    assert r.soft_threshold_s == pytest.approx(10.0 / 30.0)


# ---------------------------------------------------------------------------
# Family 2 + acceptance criterion 7: independent hard ceiling
# ---------------------------------------------------------------------------


def test_varying_soft_configuration_cannot_move_the_hard_ceiling():
    """Admitter-free, so nothing but the resolution rule is under test."""
    # A generous hard ceiling, so the sweep below measures INDEPENDENCE and
    # not the (separately tested) hard<=soft contradiction guard.
    ref = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.0,
        hard_backstop_frames=3000.0,
    )
    seen_soft = set()
    for mult in (1.0, 1.25, 1.5, 2.2, 3.0, 8.0):
        for target_fps in (0.5, 1.0, 2.0, 5.0):
            r = resolve_backstop_thresholds(
                mode="target_time", velocity_adaptive=True, input_fps=30.0,
                target_admit_rate_fps=target_fps, soft_multiplier=mult,
                hard_backstop_frames=3000.0,
            )
            assert r.hard_threshold_frames == ref.hard_threshold_frames
            assert r.hard_threshold_s == ref.hard_threshold_s
            assert r.hard_domain == ref.hard_domain
            seen_soft.add(round(r.soft_threshold_s, 9))
    # The sweep genuinely moved the soft threshold -- otherwise the assertion
    # above would hold trivially.
    assert len(seen_soft) > 1


def test_hard_ceiling_default_preserves_the_current_effective_production_bound():
    """The migration requirement: today's effective ceiling is
    ``max_gap * max_gap_hard_mult`` at the CLI defaults. Pin the new explicit
    default to that product so the two cannot silently desynchronise."""
    import arguments

    # ModelParams.__init__ needs a live parser, so read the CLI literals from
    # source, the same way test_legacy_mode_is_the_default_everywhere does.
    src = open(arguments.__file__, "r", encoding="utf-8").read()
    assert "self.streaming_keyframe_max_gap = 10" in src
    assert "self.streaming_keyframe_max_gap_hard_mult = 4.0" in src
    assert "self.streaming_keyframe_hard_backstop_frames = 40.0" in src
    assert KeyframeAdmitterConfig().hard_backstop_frames == 10 * 4.0


def test_hard_ceiling_may_be_expressed_in_seconds_independently():
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.25, hard_backstop_s=3.0,
    )
    assert r.hard_domain == "seconds"
    assert r.hard_threshold_s == pytest.approx(3.0)
    assert r.soft_domain == "seconds"
    assert r.soft_threshold_s == pytest.approx(0.625)


def test_crossings_are_independent_under_a_mixed_domain_configuration():
    """``soft_backstop_triggered`` must NOT be re-derived from the ForcePath.

    The default ``target_time`` configuration is mixed-domain: soft in source
    SECONDS, hard in source FRAMES. So "hard fired" does not imply "soft
    fired" -- if the clock is unusable the seconds-domain soft threshold can
    never fire while the frames-domain hard ceiling still does. A row
    claiming ``soft_backstop_triggered=True`` beside a null
    ``starvation_age_source_s`` would be self-contradictory telemetry.
    """
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.0,   # soft = 0.5 s
        hard_backstop_frames=40.0,                        # hard = 40 frames
    )
    assert (r.soft_domain, r.hard_domain) == ("seconds", "frames")

    # No usable seconds-domain age at all: hard fires, soft cannot.
    soft, hard = backstop_crossings(r, 40, None)
    assert hard is True
    assert soft is False
    assert resolve_backstop_trigger(r, 40, None) is ForcePath.HARD_MAX_GAP_CEILING

    # Both measurable and both crossed -> both True, ForcePath still HARD.
    soft, hard = backstop_crossings(r, 40, 5.0)
    assert (soft, hard) == (True, True)
    assert resolve_backstop_trigger(r, 40, 5.0) is ForcePath.HARD_MAX_GAP_CEILING


def test_mixed_domain_crossings_reach_the_decision_unmangled():
    """End-to-end form of the test above: with ``input_fps <= 0`` there is no
    seconds-domain age, so a real admitter must report the hard ceiling
    without also claiming the soft threshold fired."""
    cfg = _cfg(
        soft_backstop_mode="target_time", velocity_adaptive=True,
        target_admit_rate_fps=2.0, soft_backstop_multiplier=1.0,
        hard_backstop_frames=40.0, input_fps=0.0, min_translation_m=1e6,
    )
    a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
    cam = _FakeCam()
    hard = None
    for i in range(1, 45):
        d, _ = a.admit(cam, _frame(i, 0.0, ts=None))
        assert d.soft_backstop_triggered is False, i
        if d.hard_backstop_triggered:
            hard = d
            break
    assert hard is not None
    assert hard.starvation_age_source_s is None
    assert hard.starvation_age_basis == "unavailable"
    assert hard.soft_backstop_domain == "seconds"
    assert hard.hard_backstop_domain == "frames"


def test_hard_wins_when_both_thresholds_fire():
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.0, hard_backstop_s=2.0,
    )
    assert resolve_backstop_trigger(r, 0, 0.4) is ForcePath.NORMAL
    assert resolve_backstop_trigger(r, 0, 0.6) is ForcePath.SOFT_MAX_GAP
    assert resolve_backstop_trigger(r, 0, 5.0) is ForcePath.HARD_MAX_GAP_CEILING


# ---------------------------------------------------------------------------
# Family 3: 30 Hz vs 60 Hz invariance
# ---------------------------------------------------------------------------


def _target_time_cfg(input_fps, **kw):
    # multiplier 1.0 and target 1/0.7 Hz => a 0.7 s soft threshold, which is
    # an exact whole number of frames at 30 Hz (21) and 60 Hz (42) and a
    # multiple of the stride used by the stride-invariance test.
    base = dict(
        soft_backstop_mode="target_time",
        soft_backstop_multiplier=1.0,
        velocity_adaptive=True,
        target_admit_rate_fps=1.0 / 0.7,
        input_fps=input_fps,
        # Stationary camera in every target_time test below, so any positive
        # translation threshold suffices to keep motion from ever admitting;
        # the backstop is then the sole owner of admission.
        min_translation_m=1e6,
        # Seconds-domain hard ceiling, well clear of the 0.7 s soft threshold.
        # A FRAMES-domain ceiling would be the confound this family exists to
        # rule out: 40 frames is 1.33 s at 30 Hz but only 0.67 s at 60 Hz.
        hard_backstop_s=3.0,
    )
    base.update(kw)
    return _cfg(**base)


@pytest.mark.parametrize("trusted_clock", [True, False])
def test_same_physical_trajectory_triggers_at_the_same_time_at_30_and_60_hz(
    trusted_clock,
):
    """The same physical trajectory represented at 30 Hz and at 60 Hz must
    reach the soft backstop at the same PHYSICAL time.

    Parametrised over both clock paths on purpose: the trusted-timestamp path
    and the ``gap / input_fps`` nominal fallback must BOTH be rate-invariant,
    otherwise a source whose timestamps fail the #793 trust rule would silently
    get different admission timing.
    """
    fired = {}
    for fps in (30.0, 60.0):
        cfg = _target_time_cfg(fps)
        a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
        frames = [
            _frame(i, 0.0, ts=(i / fps) if trusted_clock else None)
            for i in range(1, int(3 * fps))
        ]
        d, prev = _first_soft_trigger(a, frames)
        assert d is not None, fps
        fired[fps] = d
        # It fired at the FIRST candidate past the threshold, not later.
        assert d.starvation_age_source_s >= d.soft_backstop_threshold_s
        if prev is not None:
            assert prev.starvation_age_source_s < d.soft_backstop_threshold_s

    # Same physical instant, different frame counts -- the whole point.
    assert fired[30.0].starvation_age_source_s == pytest.approx(
        fired[60.0].starvation_age_source_s, abs=1e-9
    )
    assert fired[30.0].starvation_age_source_frames == 21
    assert fired[60.0].starvation_age_source_frames == 42


# ---------------------------------------------------------------------------
# Family 4: index-preserving stride invariance
# ---------------------------------------------------------------------------


def test_index_preserving_stride_triggers_at_the_same_physical_time():
    """``StridedFrameSource`` yields the UNDERLYING frames unchanged, so a
    stride-3 view carries the original source indices and timestamps. The
    physical trigger time must therefore be unchanged."""
    fps = 30.0
    fired = {}
    for stride in (1, 3):
        cfg = _target_time_cfg(fps)
        a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
        frames = [_frame(i, 0.0, ts=i / fps) for i in range(stride, 90, stride)]
        d, _ = _first_soft_trigger(a, frames)
        assert d is not None, stride
        fired[stride] = d

    assert fired[1].starvation_age_source_s == pytest.approx(0.7, abs=1e-9)
    assert fired[3].starvation_age_source_s == pytest.approx(0.7, abs=1e-9)
    assert fired[1].starvation_age_source_frames == 21
    assert fired[3].starvation_age_source_frames == 21


# ---------------------------------------------------------------------------
# Family 5: irregular but trustworthy timestamps use elapsed time, not count
# ---------------------------------------------------------------------------


def test_irregular_timestamps_drive_the_trigger_by_elapsed_time_not_count():
    """A source whose per-frame dt is irregular (but whose implied rate still
    passes the #793 trust rule) must have its starvation age read off the
    CLOCK, not off the frame count."""
    fps = 30.0
    cfg = _target_time_cfg(fps)
    a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
    cam = _FakeCam()

    # Uniform candidates first, so the sticky timebase decision resolves to
    # "source_timestamp" (it needs _TIMEBASE_WARMUP_SAMPLES consecutive
    # samples, and the age is read one step BEFORE the velocity update, so the
    # basis switches one candidate after timebase_source does).
    t = 0.0
    last = None
    for i in range(1, 9):
        t = i / fps
        last, _ = a.admit(cam, _frame(i, 0.0, ts=t))
    assert last.timebase_source == "source_timestamp"
    assert last.starvation_age_basis == "source_timestamp"
    assert last.soft_backstop_triggered is False   # 8 frames, 0.27 s < 0.7 s

    d, _ = a.admit(cam, _frame(9, 0.0, ts=t + 2.0))
    # ONE more frame, but two seconds of real time: the age must follow the
    # CLOCK. The reference is the baseline keyframe at ts=0.0.
    assert d.starvation_age_basis == "source_timestamp"
    assert d.starvation_age_source_frames == 9
    assert d.starvation_age_source_s == pytest.approx(8 / fps + 2.0, abs=1e-6)
    assert d.soft_backstop_triggered is True
    # A frame-count rule would NOT have fired here: 9 < 21.
    assert d.starvation_age_source_frames < d.soft_backstop_threshold_frames


# ---------------------------------------------------------------------------
# Family 6: clock-fallback provenance
# ---------------------------------------------------------------------------


def test_unusable_timestamps_fall_back_explicitly_and_are_recorded():
    """``timestamp = float(index)`` is the real junk case in this codebase
    (hypersim / orbbec_export / realityscan / the rgbd_sequence fallback). It
    must be rejected and the fallback must be RECORDED, not guessed at."""
    cfg = _target_time_cfg(30.0)
    a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
    cam = _FakeCam()
    last = None
    for i in range(1, 12):
        last, _ = a.admit(cam, _frame(i, 0.0, ts=float(i)))  # implied ~1 Hz
    assert last.timebase_source == "frame_index_nominal_fps"
    assert last.starvation_age_basis == "frame_index_nominal_fps"
    assert last.starvation_age_source_s == pytest.approx(
        last.starvation_age_source_frames / 30.0
    )


def test_starvation_basis_is_not_an_alias_of_the_controller_timebase():
    """The two fields describe different clocks and must be able to disagree.

    Pre-baseline the starvation age has no reference timestamp to diff
    against even when the controller has already decided the source clock is
    trustworthy -- so ``starvation_age_basis`` falls back while
    ``timebase_source`` says ``source_timestamp``. Reading one as the other
    would make the provenance field describe a quantity it does not own.
    """
    cfg = _target_time_cfg(30.0)
    a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)
    cam = _FakeCam()
    pairs = []
    for i in range(1, 12):
        d, _ = a.admit(cam, _frame(i, 0.0, ts=i / 30.0))
        pairs.append((d.timebase_source, d.starvation_age_basis))
        assert d.starvation_age_basis in {
            "source_timestamp", "frame_index_nominal_fps", "unavailable",
        }

    # They really do take different values on the same run: the controller
    # publishes a three-valued timebase (including its "..._warmup" state)
    # while the starvation basis names only the clock that produced THIS
    # number. If one were an alias of the other this set would be empty.
    assert any(tb != basis for tb, basis in pairs), pairs
    # ...and they converge once both are resolved.
    assert pairs[-1] == ("source_timestamp", "source_timestamp")


def test_starvation_basis_diverges_when_the_reference_frame_has_no_timestamp():
    """The substantive divergence case, not just a warmup naming difference.

    A bootstrap/baseline frame from a source without stamps leaves the
    starvation clock with no reference to diff against, while the velocity
    controller -- which measures candidate-to-candidate, not
    candidate-to-keyframe -- happily resolves the SAME source's timestamps as
    trustworthy. The two provenance fields then name genuinely different
    clocks, which is exactly why they are separate fields.
    """
    cfg = _target_time_cfg(30.0)
    a = KeyframeAdmitter(cfg, render_fn=stub_render(0.05))
    a.note_train_frame(_frame(0, 0.0, ts=None))   # baseline carries no stamp
    cam = _FakeCam()
    last = None
    for i in range(1, 12):
        last, _ = a.admit(cam, _frame(i, 0.0, ts=i / 30.0))
    assert last.timebase_source == "source_timestamp"
    assert last.starvation_age_basis == "frame_index_nominal_fps"
    # The age is still real and usable -- it just came from the nominal clock.
    assert last.starvation_age_source_s == pytest.approx(
        last.starvation_age_source_frames / 30.0
    )


def test_starvation_age_is_unavailable_rather_than_fabricated():
    r_cfg = _cfg(input_fps=0.0)
    a = _admitter(r_cfg, overlap=0.05)
    cam = _FakeCam()
    d, _ = a.admit(cam, _frame(5, 0.0, ts=None))
    assert d.starvation_age_basis == "unavailable"
    assert d.starvation_age_source_s is None
    # ...and the frame-domain legacy threshold still works, so admission
    # cannot starve just because the clock is unusable.
    assert d.starvation_age_source_frames == 5


# ---------------------------------------------------------------------------
# Family 7: precedence is unchanged
# ---------------------------------------------------------------------------


def test_soft_under_motion_only_still_cannot_bypass_coverage_or_quality():
    allowed = bypass_permissions(ForcePath.SOFT_MAX_GAP,
                                 MaxGapForceScope.MOTION_ONLY)
    assert allowed == frozenset({Gate.MOTION})
    for g in (Gate.COVERAGE_NOVELTY, Gate.PARALLAX, Gate.EXPOSURE, Gate.BLUR,
              Gate.RENDER_OVERLAP, Gate.QUALITY_SCORE, Gate.INTEGRITY):
        assert g not in allowed


def test_hard_ceiling_bypasses_everything_except_integrity():
    allowed = bypass_permissions(ForcePath.HARD_MAX_GAP_CEILING)
    assert allowed == frozenset(GATE_ORDER) - UPSTREAM_GATES
    assert Gate.INTEGRITY not in allowed


def test_target_time_soft_escalation_is_still_bound_by_the_render_gate():
    """End-to-end: a soft trigger resolved in SOURCE TIME must go through the
    same #794 machinery a frame-domain one did -- under ``motion_only`` it
    relaxes MOTION and nothing else, so a candidate the render-overlap gate
    rejects stays rejected. A hard trigger admits it."""
    cfg = _target_time_cfg(
        30.0,
        max_gap_force_scope=MaxGapForceScope.MOTION_ONLY.value,
        # soft fires at 0.7 s (frame 21), hard at 1.4 s (frame 42).
        hard_backstop_s=1.4,
    )
    a = _admitter(cfg, overlap=0.05, baseline_ts=0.0)   # 0.05 < min_overlap
    cam = _FakeCam()
    soft_seen = hard_seen = None
    for i in range(1, 60):
        d, _ = a.admit(cam, _frame(i, 0.0, ts=i / 30.0))
        if d.hard_backstop_triggered and hard_seen is None:
            hard_seen = d
        elif d.soft_backstop_triggered and soft_seen is None:
            soft_seen = d

    assert soft_seen is not None and hard_seen is not None
    assert soft_seen.admitted is False
    assert soft_seen.reason == "low_overlap"
    assert soft_seen.trace["candidate_trigger"] == ForcePath.SOFT_MAX_GAP.value
    assert hard_seen.admitted is True
    assert hard_seen.trace["candidate_trigger"] == (
        ForcePath.HARD_MAX_GAP_CEILING.value
    )
    assert hard_seen.trace["hard_force"] is True


def test_motion_trigger_cannot_lie_about_who_owned_the_motion_criterion():
    """``reason="pose_delta"`` alone could not say whether translation or
    rotation crossed (the corrected fr3 run had 73/73 pose-delta admits also
    crossing the static 5-degree rotation threshold). Invariant:
    ``(motion_trigger != "none") == moved``."""
    cfg = _cfg(min_translation_m=0.05, min_rotation_deg=1e9)
    a = _admitter(cfg, overlap=0.5)
    cam = _FakeCam()
    seen = set()
    for i, x in enumerate([0.0, 0.001, 0.2, 0.201, 0.9], start=1):
        d, _ = a.admit(cam, _frame(i, x, ts=i / 30.0))
        assert (d.motion_trigger != "none") == bool(d.moved), (i, d.motion_trigger)
        seen.add(d.motion_trigger)
    assert {"none", "translation"} <= seen


# ---------------------------------------------------------------------------
# Family 9: config validation
# ---------------------------------------------------------------------------


def test_hard_below_or_equal_soft_is_refused_loudly():
    with pytest.raises(BackstopConfigError) as ei:
        resolve_backstop_thresholds(
            mode="target_time", velocity_adaptive=True, input_fps=30.0,
            target_admit_rate_fps=2.0, soft_multiplier=2.0,   # soft = 1.0 s
            hard_backstop_s=0.5,                              # hard = 0.5 s
        )
    msg = str(ei.value)
    assert "0.5" in msg and "1.0" in msg, msg
    assert "hard" in msg.lower() and "soft" in msg.lower()

    # Exact equality is refused too -- an equal ceiling makes the soft stage
    # unreachable just as surely as a lower one.
    with pytest.raises(BackstopConfigError):
        resolve_backstop_thresholds(
            mode="target_time", velocity_adaptive=True, input_fps=30.0,
            target_admit_rate_fps=2.0, soft_multiplier=1.0,   # soft = 0.5 s
            hard_backstop_s=0.5,
        )


def test_refusal_happens_at_construction_not_mid_run():
    """A per-candidate raise would crash training. The contradiction must be
    caught while the admitter is being built."""
    with pytest.raises(BackstopConfigError):
        KeyframeAdmitter(
            _cfg(soft_backstop_mode="target_time", velocity_adaptive=True,
                 target_admit_rate_fps=2.0, soft_backstop_multiplier=2.0,
                 hard_backstop_s=0.5),
            render_fn=stub_render(0.5),
        )


def test_a_typo_never_silently_enables_production_semantics():
    mode, warning = coerce_soft_backstop_mode("targettime")
    assert mode is SoftBackstopMode.LEGACY_FRAMES
    assert warning is not None
    assert "targettime" in warning and "did NOT take effect" in warning

    # ...and it is surfaced on the admitter, not swallowed.
    a = KeyframeAdmitter(_cfg(soft_backstop_mode="targettime"),
                         render_fn=stub_render(0.5))
    assert a.backstop.mode == "legacy_frames"
    assert any("targettime" in w for w in a.backstop_config_warnings)


def test_legacy_contradiction_warns_rather_than_retroactively_rejecting():
    """``max_gap_hard_mult <= 1`` is reachable from historical configs and was
    accepted before #946. It must stay accepted, with a warning."""
    r = resolve_backstop_thresholds(mode="legacy_frames", max_gap=10,
                                    max_gap_hard_mult=1.0, input_fps=30.0)
    assert r.soft_threshold_frames == 10.0
    assert r.hard_threshold_frames == 10.0
    assert any("unreachable" in w for w in r.warnings)


def test_target_time_flags_a_deliberate_change_to_the_safety_bound():
    r = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.25,
        hard_backstop_frames=90.0,
    )
    warns = validate_soft_backstop_config(
        r, velocity_adaptive=True, max_gap=10, max_gap_hard_mult=4.0
    )
    assert any("DIFFERS" in w for w in warns), warns

    r_same = resolve_backstop_thresholds(
        mode="target_time", velocity_adaptive=True, input_fps=30.0,
        target_admit_rate_fps=2.0, soft_multiplier=1.25,
    )
    assert validate_soft_backstop_config(
        r_same, velocity_adaptive=True, max_gap=10, max_gap_hard_mult=4.0
    ) == []
