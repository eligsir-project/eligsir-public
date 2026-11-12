"""Issue #794 -- the admission precedence contract, pinned.

Three groups:

1. **Table tests** over
   :mod:`eligsir.mapping.scheduler.precedence` itself: the declarative
   force-path x gate bypass matrix, and the invariant that ``INTEGRITY``
   (the #806/#817 ORB-SLAM3 map-reinit veto, enforced upstream in
   ``StreamingScene.ingest_next_frame``) appears in NO force path's bypass
   set under ANY scope.

2. **Behavioural matrix tests** against the real ``KeyframeAdmitter``: for
   each force path, for each gate, does a frame that would fail that gate
   actually get admitted (bypassed) or rejected (enforced)? These are the
   tests that FAIL on unmodified ``origin/main`` for a contract *violation*
   rather than for a missing attribute -- specifically the
   ``motion_only``-scope rows, where main admits a blurry / badly-exposed /
   low-overlap frame purely because ``gap >= max_gap``.

3. **Trace tests**: every decision carries a reason chain, and a bypass of a
   DISABLED gate is bookkept separately from a bypass of a live one (so the
   telemetry answers "was a real quality check skipped?" and is not inflated
   by gates that were off for the whole run).

CPU-only. Mirrors ``tests/test_keyframe_max_gap_novelty.py``'s fake camera /
stub occupancy harness.
"""
from __future__ import annotations

import numpy as np
import pytest

from _fake_alpha import stub_render  # noqa: E402
from eligsir.mapping.scheduler.precedence import (
    GATE_ORDER,
    ForcePath,
    Gate,
    MaxGapForceScope,
    PrecedenceTrace,
    bypass_permissions,
    coerce_scope,
    may_bypass,
    resolve_max_gap_trigger,
    validate_backstop_config,
)
from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)


# ---------------------------------------------------------------------------
# Group 1 -- the declarative table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", list(MaxGapForceScope))
@pytest.mark.parametrize("trigger", list(ForcePath))
def test_integrity_is_never_bypassable(trigger, scope):
    """The #806/#817 safety ordering: the integrity veto sits at the top of
    the precedence contract and no force path may ever preempt it."""
    assert Gate.INTEGRITY not in bypass_permissions(trigger, scope)
    assert not may_bypass(trigger, Gate.INTEGRITY, scope)


def test_normal_candidate_bypasses_nothing():
    for scope in MaxGapForceScope:
        assert bypass_permissions(ForcePath.NORMAL, scope) == frozenset()


def test_hard_ceiling_bypasses_everything_except_integrity():
    expected = frozenset(GATE_ORDER) - {Gate.INTEGRITY}
    for scope in MaxGapForceScope:
        assert bypass_permissions(ForcePath.HARD_MAX_GAP_CEILING, scope) == expected


def test_soft_max_gap_scope_matrix():
    legacy = bypass_permissions(ForcePath.SOFT_MAX_GAP, MaxGapForceScope.LEGACY_ALL_GATES)
    motion_only = bypass_permissions(ForcePath.SOFT_MAX_GAP, MaxGapForceScope.MOTION_ONLY)
    # Legacy == the pre-#794 behaviour being documented (and preserved).
    assert legacy == frozenset(GATE_ORDER) - {Gate.INTEGRITY}
    # The #794 contract: motion only.
    assert motion_only == frozenset({Gate.MOTION})
    for gate in (Gate.EXPOSURE, Gate.BLUR, Gate.PARALLAX, Gate.RENDER_OVERLAP,
                 Gate.QUALITY_SCORE, Gate.COVERAGE_NOVELTY):
        assert gate not in motion_only


def test_starvation_reliefs_bypass_exactly_their_own_gate():
    assert bypass_permissions(ForcePath.EXPOSURE_STARVATION) == frozenset({Gate.EXPOSURE})
    assert bypass_permissions(ForcePath.BLUR_STARVATION) == frozenset({Gate.BLUR})
    assert bypass_permissions(ForcePath.LOW_OVERLAP_STARVATION) == frozenset(
        {Gate.RENDER_OVERLAP}
    )


def test_resolve_max_gap_trigger_boundaries():
    assert resolve_max_gap_trigger(9, 10, 4.0) is ForcePath.NORMAL
    assert resolve_max_gap_trigger(10, 10, 4.0) is ForcePath.SOFT_MAX_GAP
    assert resolve_max_gap_trigger(39, 10, 4.0) is ForcePath.SOFT_MAX_GAP
    assert resolve_max_gap_trigger(40, 10, 4.0) is ForcePath.HARD_MAX_GAP_CEILING


def test_unknown_scope_falls_back_to_behaviour_preserving_default():
    assert coerce_scope("typo_not_a_scope") is MaxGapForceScope.LEGACY_ALL_GATES
    assert coerce_scope("motion_only") is MaxGapForceScope.MOTION_ONLY
    assert coerce_scope(MaxGapForceScope.MOTION_ONLY) is MaxGapForceScope.MOTION_ONLY


def test_config_validation_warns_but_never_raises():
    warns = validate_backstop_config(
        max_gap=10,
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_rate_fps=2.0,   # nominal interval = 15 frames > max_gap 10
        scope=MaxGapForceScope.LEGACY_ALL_GATES,
        max_gap_require_novelty=True,
        coverage_gain_gate=True,
    )
    assert any("SHORTER" in w for w in warns)
    # Consistent config -> no warning.
    assert validate_backstop_config(
        max_gap=30,
        velocity_adaptive=True,
        input_fps=30.0,
        target_admit_rate_fps=2.0,
        scope=MaxGapForceScope.LEGACY_ALL_GATES,
        max_gap_require_novelty=True,
        coverage_gain_gate=True,
    ) == []
    # The rate check is scoped to the controller that actually reads those fields.
    assert validate_backstop_config(
        max_gap=10,
        velocity_adaptive=False,
        input_fps=30.0,
        target_admit_rate_fps=2.0,
        scope=MaxGapForceScope.LEGACY_ALL_GATES,
        max_gap_require_novelty=True,
        coverage_gain_gate=True,
    ) == []


# ---------------------------------------------------------------------------
# Harness for groups 2 and 3
# ---------------------------------------------------------------------------


class _FakeCam:
    def __init__(self):
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 8.0
        self.cy = 8.0
        self.c2w = np.eye(4, dtype=np.float32)
        self.world_view_transform = np.eye(4, dtype=np.float32)


def _frame(index: int, x: float = 0.0):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index)


def _stub_render(overlap_value: float):
    """Alpha whose (alpha > coverage_alpha).mean() == overlap_value.

    Torch-free: this file must stay collectable (and PASSING) in the CPU smoke
    image, which has no torch. See tests/_fake_alpha.py for why the previous
    inline `import torch` silently broke three tests here.
    """
    return stub_render(overlap_value)


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,
        max_gap=10,
        max_gap_hard_mult=4.0,
        # Keep the coverage/novelty machinery out of the way unless a test
        # asks for it: PR #826's novelty requirement is a separate, already
        # merged mechanism and these tests isolate the OTHER gates.
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
        motion_pregate=True,
        velocity_adaptive=False,
        min_overlap=0.25,
        max_overlap=0.90,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _admitter(cfg, overlap=0.5, **kw):
    a = KeyframeAdmitter(cfg, render_fn=_stub_render(overlap), **kw)
    # Establish a pose baseline so gap/pose deltas are real.
    a.note_train_frame(_frame(0, 0.0))
    return a


def _admit(a, index, x=0.0):
    return a.admit(_FakeCam(), _frame(index, x))[0]


# ---------------------------------------------------------------------------
# Group 2 -- behavioural precedence matrix
#
# Each test is one (force path, gate) cell of issue #794's requested matrix.
# ---------------------------------------------------------------------------


def test_soft_max_gap_x_render_overlap_legacy_bypasses():
    """DEFAULT scope: a frame whose alpha-coverage over the existing map is far
    below ``min_overlap`` (the "low_overlap" rejection) is admitted anyway,
    purely on gap. This is main's behaviour and is deliberately preserved."""
    a = _admitter(_cfg(), overlap=0.05)
    d = _admit(a, index=10, x=0.0)
    assert d.admitted and d.reason == "max_gap"
    assert d.trace["candidate_trigger"] == ForcePath.SOFT_MAX_GAP.value
    assert Gate.RENDER_OVERLAP.value in d.trace["bypassed_gates"]


def test_soft_max_gap_x_render_overlap_motion_only_enforces():
    """#794 contract: the same frame is judged by the render-overlap gate."""
    a = _admitter(
        _cfg(max_gap_force_scope=MaxGapForceScope.MOTION_ONLY.value), overlap=0.05
    )
    d = _admit(a, index=10, x=0.0)
    assert not d.admitted
    assert d.reason == "low_overlap"
    assert d.trace["final_owner"] == "low_overlap"
    assert Gate.RENDER_OVERLAP.value in d.trace["evaluated_gates"]


def test_soft_max_gap_motion_only_still_prevents_starvation():
    """Bypassing the motion gate means motion is SATISFIED, not skipped: a
    stationary frame over healthy, non-redundant map coverage must still be
    admitted, or the backstop would be silently disabled."""
    # overlap 0.95: comfortably above min_overlap, and ABOVE max_overlap so the
    # frame is not "novel" either -- without the motion-satisfied semantics it
    # would fall through to `overlap_ok and (moved or novel)` and be rejected
    # "redundant", silently disabling the backstop.
    a = _admitter(
        _cfg(max_gap_force_scope=MaxGapForceScope.MOTION_ONLY.value), overlap=0.95
    )
    d = _admit(a, index=10, x=0.0)
    assert d.admitted, "soft escalation must not reinstate starvation"
    assert d.moved is False
    assert d.forced is True
    # NOT "max_gap": that string is how #795's AdmissionFingerprint recognises
    # a hard force, and a gate-passing escalation is not one.
    assert d.reason == "soft_max_gap_admit"


def test_soft_max_gap_x_blur_motion_only_enforces():
    a = _admitter(
        _cfg(
            max_gap_force_scope=MaxGapForceScope.MOTION_ONLY.value,
            blur_gate_enabled=True,
            blur_gate_min_laplacian=1e9,      # everything counts as blurred
            blur_gate_force_admit_iters=10**6,  # starvation relief never fires
        ),
        overlap=0.5,
        load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32),
    )
    d = _admit(a, index=10, x=0.0)
    assert not d.admitted and d.reason == "blur_reject"
    assert d.trace["final_owner"] == "blur_reject"


def test_soft_max_gap_x_blur_legacy_bypasses():
    a = _admitter(
        _cfg(
            blur_gate_enabled=True,
            blur_gate_min_laplacian=1e9,
            blur_gate_force_admit_iters=10**6,
        ),
        overlap=0.5,
        load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32),
    )
    d = _admit(a, index=10, x=0.0)
    assert d.admitted and d.reason == "max_gap"
    assert Gate.BLUR.value in d.trace["bypassed_gates"]


def test_soft_max_gap_x_exposure_motion_only_enforces():
    a = _admitter(
        _cfg(
            max_gap_force_scope=MaxGapForceScope.MOTION_ONLY.value,
            exposure_gate_enabled=True,
            exposure_max_delta=0.01,
            exposure_force_admit_iters=10**6,
        ),
        overlap=0.5,
        load_rgb=lambda f: np.full((8, 8, 3), 0.9, dtype=np.float32),
    )
    # Seed the luminance reference with a DARK frame, then present a bright one.
    a._last_admitted_luma = 0.05
    d = _admit(a, index=10, x=0.0)
    assert not d.admitted and d.reason == "exposure_reject"


def test_soft_max_gap_x_exposure_legacy_bypasses():
    a = _admitter(
        _cfg(
            exposure_gate_enabled=True,
            exposure_max_delta=0.01,
            exposure_force_admit_iters=10**6,
        ),
        overlap=0.5,
        load_rgb=lambda f: np.full((8, 8, 3), 0.9, dtype=np.float32),
    )
    a._last_admitted_luma = 0.05
    d = _admit(a, index=10, x=0.0)
    assert d.admitted and d.reason == "max_gap"
    assert Gate.EXPOSURE.value in d.trace["bypassed_gates"]


def test_hard_ceiling_bypasses_every_gate_under_both_scopes():
    """The one explicitly-named unconditional force. Even under the #794
    contract scope it still forces -- that is what makes it the backstop."""
    for scope in MaxGapForceScope:
        a = _admitter(
            _cfg(
                max_gap_force_scope=scope.value,
                blur_gate_enabled=True,
                blur_gate_min_laplacian=1e9,
                blur_gate_force_admit_iters=10**6,
            ),
            overlap=1.0,
            load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32),
        )
        d = _admit(a, index=40, x=0.0)   # gap 40 == max_gap * hard_mult
        assert d.admitted and d.reason == "max_gap", scope
        assert d.trace["hard_force"] is True
        assert d.trace["candidate_trigger"] == ForcePath.HARD_MAX_GAP_CEILING.value
        assert Gate.BLUR.value in d.trace["bypassed_gates"]
        assert Gate.INTEGRITY.value not in d.trace["bypassed_gates"]
        assert Gate.INTEGRITY.value not in d.trace["bypassed_gates_inactive"]


def test_normal_candidate_never_records_a_bypass():
    a = _admitter(_cfg(), overlap=0.5)
    d = _admit(a, index=3, x=1.0)    # moved, gap 3 < max_gap
    assert d.admitted
    assert d.trace["candidate_trigger"] == ForcePath.NORMAL.value
    assert d.trace["bypassed_gates"] == []
    assert d.trace["hard_force"] is False


def test_low_overlap_starvation_bypasses_only_the_overlap_gate():
    a = _admitter(
        _cfg(low_overlap_force_admit_iters=1, min_overlap=0.9), overlap=0.1
    )
    d = _admit(a, index=3, x=1.0)   # moved, ordinary candidate, overlap too low
    assert d.admitted and d.reason == "low_overlap_force_admit"
    assert d.trace["bypassed_gates"] == [Gate.RENDER_OVERLAP.value]


# ---------------------------------------------------------------------------
# Group 3 -- trace / telemetry
# ---------------------------------------------------------------------------


def test_every_decision_carries_a_reason_chain():
    a = _admitter(_cfg(), overlap=0.5)
    for idx, x in ((3, 1.0), (10, 0.0), (1, 0.0)):
        d = a.admit(_FakeCam(), _frame(idx, x))[0]
        assert d.trace is not None
        for key in ("candidate_trigger", "bypassed_gates", "evaluated_gates",
                    "final_owner", "hard_force"):
            assert key in d.trace
        assert d.trace["final_owner"] == d.reason


def test_pregate_rejection_also_carries_a_reason_chain():
    a = _admitter(_cfg(), overlap=0.5)
    d = a.pregate(_frame(1, 0.0))
    assert d is not None and d.reason == "no_motion"
    assert d.trace["final_owner"] == "no_motion"
    assert Gate.MOTION.value in d.trace["evaluated_gates"]


def test_disabled_gate_bypass_is_bookkept_separately():
    """A soft force under the legacy scope is *permitted* to bypass the blur
    gate, but if that gate is off for the run, counting it as a real bypass
    would inflate the telemetry #794 asks for."""
    a = _admitter(_cfg(), overlap=1.0)   # blur/exposure/parallax/quality all off
    d = _admit(a, index=10, x=0.0)
    assert Gate.BLUR.value in d.trace["bypassed_gates_inactive"]
    assert Gate.BLUR.value not in d.trace["bypassed_gates"]
    # RENDER_OVERLAP is live (a render_fn is wired), so it counts for real.
    assert Gate.RENDER_OVERLAP.value in d.trace["bypassed_gates"]
    assert a.precedence_bypass_counts.get(Gate.RENDER_OVERLAP.value) == 1
    assert a.precedence_bypass_inactive_counts.get(Gate.BLUR.value) == 1
    assert a.precedence_trigger_counts[ForcePath.SOFT_MAX_GAP.value] == 1


def test_default_scope_is_behaviour_preserving():
    assert KeyframeAdmitterConfig().max_gap_force_scope == (
        MaxGapForceScope.LEGACY_ALL_GATES.value
    )
    assert KeyframeAdmitter(_cfg()).force_scope is MaxGapForceScope.LEGACY_ALL_GATES


def test_trace_to_dict_is_json_serialisable():
    import json

    t = PrecedenceTrace()
    t.record_bypass(Gate.BLUR, True)
    t.record_evaluated(Gate.MOTION)
    json.dumps(t.to_dict())
