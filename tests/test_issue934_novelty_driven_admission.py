"""Issue #934 -- novelty-DRIVEN admission (``ForcePath.NOVELTY_FORCE``).

The finding this suite pins
---------------------------
Two fr3 GPU runs (``i934stage1a_smart_admit``, ``i934stage1a2_smart_admit_repin``)
measured that ~96% of admissions were owned by the motion gate (``pose_delta``),
with only 4-5 ever owned by the coverage/novelty gate -- at BOTH ends of a
threshold sweep. The cause was structural, not a threshold choice: the coverage
signal only ever REJECTED (it withdrew a max_gap force, or rejected below
``min_new_voxel_frac``), and the motion pregate short-circuited to ``no_motion``
before the coverage gate was even consulted for a non-moving candidate.

``test_veto_only_trap_reproduced_when_flag_off`` is the regression anchor: it
reproduces that exact failure mode on a candidate with MAXIMAL novelty and zero
motion, so the fix is provably not the same trap. Every other test here is
about the fix not creating a NEW trap (the trigger firing but a downstream
AND-term eating the admission) or weakening a safety mechanism.

CPU-only. Harness mirrors ``tests/test_keyframe_max_gap_novelty.py`` (stub
occupancy + tiny fake pinhole camera) and ``tests/_fake_alpha.py`` (torch-free
alpha stand-in), so this file is collectable in the CPU smoke image.
"""
from __future__ import annotations

import numpy as np
import pytest

from _fake_alpha import stub_render
from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)
from eligsir.mapping.scheduler.precedence import (
    GATE_ORDER,
    ForcePath,
    Gate,
    MaxGapForceScope,
    bypass_permissions,
    may_bypass,
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


def _frame(index: int, x: float = 0.0, ts: float = None):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    f = FrameInfo(c2w=c2w, index=index)
    if ts is not None:
        f.timestamp = ts
    return f


def _depth_const(value: float = 2.0):
    arr = np.full((16, 16), value, dtype=np.float32)
    return lambda frame: arr


_ALL_OCCUPIED = lambda pts, vox: np.ones(len(pts), dtype=bool)      # noqa: E731
_ALL_UNOCCUPIED = lambda pts, vox: np.zeros(len(pts), dtype=bool)   # noqa: E731


def _partly_occupied(occupied_frac: float):
    """Occupancy mask giving new_voxel_frac == 1 - occupied_frac."""
    def _fn(pts, vox):
        n = len(pts)
        k = int(round(occupied_frac * n))
        m = np.zeros(n, dtype=bool)
        m[:k] = True
        return m
    return _fn


def _cfg(**kw):
    base = dict(
        # A LARGE motion threshold: no candidate in these tests "moves", so
        # the motion gate rejects everything unless something else admits it.
        min_translation_m=10.0,
        min_rotation_deg=1e9,
        # A large max_gap so the starvation backstop never fires and cannot be
        # confused for the mechanism under test.
        max_gap=10_000,
        max_gap_hard_mult=4.0,
        coverage_gain_gate=True,
        min_new_voxel_frac=0.1,
        coverage_gain_depth_stride=4,
        depth_max=8.0,
        max_gap_require_novelty=False,
        motion_pregate=True,
        # velocity_adaptive ON: the default stale-pose guard needs a real
        # per-candidate velocity sample, and production runs it on.
        velocity_adaptive=True,
        velocity_adaptive_min_trans_floor=10.0,
        velocity_adaptive_min_trans_ceil=10.0,
        input_fps=30.0,
        min_overlap=0.25,
        max_overlap=0.90,
        novelty_force_admit=True,
        novelty_force_frac=0.6,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _admitter(cfg, occupancy_fn=_ALL_UNOCCUPIED, overlap=0.99):
    """Admitter with a pose baseline already established.

    ``overlap=0.99`` is the measured fr3 operating point (every one of the 168
    candidates that reached the render gate on ``i934stage1a2_smart_admit_repin``
    scored 0.973-1.000 against ``min_overlap=0.25``), so the default harness
    reproduces the real substrate rather than a convenient one.
    """
    adm = KeyframeAdmitter(
        config=cfg,
        render_fn=stub_render(overlap),
        occupancy_fn=occupancy_fn,
        load_depth_m=_depth_const(2.0),
    )
    adm.note_train_frame(_frame(0, 0.0))
    return adm


def _advance(adm, index: int, x: float):
    """Feed one candidate. The stale-pose guard requires a real velocity
    sample, which only exists from the SECOND candidate onward, so every test
    that expects the trigger to fire must push at least two."""
    return adm.admit(cam=_FakeCam(), frame=_frame(index, x))


# ---------------------------------------------------------------------------
# Group 1 -- the declarative contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", list(MaxGapForceScope))
def test_novelty_force_never_bypasses_integrity(scope):
    """#806/#817: the map-reinit integrity veto sits above every force path."""
    assert Gate.INTEGRITY not in bypass_permissions(ForcePath.NOVELTY_FORCE, scope)
    assert not may_bypass(ForcePath.NOVELTY_FORCE, Gate.INTEGRITY, scope)


@pytest.mark.parametrize("scope", list(MaxGapForceScope))
def test_novelty_force_bypasses_motion_and_nothing_else(scope):
    """The whole safety argument: relax the cheap MOTION prior, keep every
    information/quality gate binding. Scope-independent by construction."""
    permitted = bypass_permissions(ForcePath.NOVELTY_FORCE, scope)
    assert permitted == frozenset({Gate.MOTION})
    for gate in GATE_ORDER:
        if gate is not Gate.MOTION:
            assert not may_bypass(ForcePath.NOVELTY_FORCE, gate, scope)


# ---------------------------------------------------------------------------
# Group 2 -- the regression anchor: the measured veto-only trap
# ---------------------------------------------------------------------------


def test_veto_only_trap_reproduced_when_flag_off():
    """THE anchor test. A candidate with MAXIMAL coverage novelty (every
    backprojected point lands in an unoccupied voxel) and zero motion is
    rejected ``no_motion`` with the flag off -- the coverage gate cannot cause
    an admission, which is exactly what the two fr3 GPU runs measured.

    If this ever starts passing as an admission with ``novelty_force_admit``
    off, the default path has silently changed.
    """
    adm = _admitter(_cfg(novelty_force_admit=False), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.0)
    decision, _ = _advance(adm, 2, 0.0)
    assert not decision.admitted
    assert decision.reason == "no_motion"
    # ...and the coverage machinery was never even consulted for it.
    assert Gate.COVERAGE_NOVELTY.value not in (decision.trace or {}).get(
        "evaluated_gates", []
    )
    assert adm.novelty_force_counts["triggered"] == 0
    assert adm.novelty_force_counts["decisive"] == 0


def test_novelty_alone_admits_when_flag_on():
    """The fix: same candidate, same zero motion, flag on -> admitted, owned by
    the novelty trigger."""
    adm = _admitter(_cfg(), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.admitted
    assert decision.reason == "novelty_admit"
    assert decision.novelty_forced is True
    assert decision.moved is False
    trace = decision.trace
    assert trace["candidate_trigger"] == ForcePath.NOVELTY_FORCE.value
    assert trace["final_owner"] == "novelty_admit"
    assert trace["novelty_force_outcome"] == "triggered"
    assert trace["novelty_force_decisive"] is True
    assert Gate.MOTION.value in trace["bypassed_gates"]
    assert adm.novelty_force_counts["triggered"] == 1
    assert adm.novelty_force_counts["decisive"] == 1


def test_forced_flag_stays_false_so_force_share_is_not_inflated():
    """``force_share`` (a #934 Stage 1a pass/fail gate) is derived from
    ``decision.forced``. Novelty force is the selector working, not starvation
    relief -- folding it in would make a SUCCESSFUL run fail its own
    operating-point check. #932 was the mirror image; do not "fix" symmetry."""
    adm = _admitter(_cfg(), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.admitted
    assert decision.forced is False
    assert decision.trace["hard_force"] is False


def test_below_threshold_novelty_does_not_admit():
    """Between ``min_new_voxel_frac`` and ``novelty_force_frac`` the trigger
    must NOT fire -- otherwise the threshold is decorative."""
    # new_voxel_frac == 0.3, above min_new_voxel_frac 0.1, below force 0.6.
    adm = _admitter(_cfg(), _partly_occupied(0.7))
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.trace["novelty_force_outcome"] == "below_threshold"
    assert decision.trace["novelty_force_fraction"] == pytest.approx(0.3, abs=0.05)
    assert adm.novelty_force_counts["triggered"] == 0


# ---------------------------------------------------------------------------
# Group 3 -- the NEW trap: does a downstream AND-term eat the admission?
# ---------------------------------------------------------------------------


def test_low_overlap_high_novelty_is_rejected_and_the_gap_is_visible():
    """The design decision, pinned.

    ``admitted = overlap_ok and (motion_ok or novel)``. The trigger only
    satisfies the OR-term; ``overlap_ok`` is an AND-term it deliberately does
    NOT bypass (RENDER_OVERLAP is not in its bypass set). High coverage-novelty
    correlates with LOW render overlap by construction, so on a scene where the
    map lags the frontier the trigger can fire and admit nothing -- the same
    trap one gate lower.

    That is accepted behaviour (the render-overlap gate is a real protection,
    and ``low_overlap_force_admit_iters`` is the existing orthogonal relief),
    but it MUST be visible rather than silent: ``triggered`` counts the firing,
    ``decisive`` stays 0, and the gap between them is the diagnostic.
    """
    adm = _admitter(_cfg(), _ALL_UNOCCUPIED, overlap=0.05)  # below min_overlap 0.25
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.reason == "low_overlap"
    # It DID fire -- the mechanism is reachable; the render gate refused it.
    assert decision.novelty_forced is True
    assert decision.trace["novelty_force_outcome"] == "triggered"
    assert decision.trace["novelty_force_decisive"] is False
    assert adm.novelty_force_counts["triggered"] == 1
    assert adm.novelty_force_counts["decisive"] == 0


def test_render_overlap_novel_term_keeps_ownership():
    """A candidate the render-overlap ``novel`` term would have admitted anyway
    is attributed to ``new_content``, NOT to this trigger. Over-claiming
    ownership is exactly what would make the Stage 1 ownership measurement
    unreadable."""
    # overlap 0.5: >= min_overlap 0.25 (admits) and <= max_overlap 0.9 (novel).
    adm = _admitter(_cfg(), _ALL_UNOCCUPIED, overlap=0.5)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.admitted
    assert decision.reason == "new_content"
    assert decision.novelty_forced is True          # it fired...
    assert decision.trace["novelty_force_decisive"] is False   # ...but did not own it
    assert adm.novelty_force_counts["triggered"] == 1
    assert adm.novelty_force_counts["decisive"] == 0


# ---------------------------------------------------------------------------
# Group 4 -- safety invariants
# ---------------------------------------------------------------------------


def test_stale_pose_guard_refuses_to_fire():
    """The motion pregate is a safety net against stale-pose frames under a
    sparse GT-pose stream. A stale pose backprojects depth at the WRONG pose,
    lands in unobserved voxels, and reads as HIGH novelty -- so an unguarded
    trigger would preferentially re-admit exactly those frames.

    Here the pose never advances between candidates (velocity sample == 0),
    with maximal novelty. The trigger must refuse.
    """
    adm = _admitter(_cfg(), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.0)
    decision, _ = _advance(adm, 2, 0.0)   # identical pose -> zero velocity
    assert not decision.admitted
    assert decision.trace["novelty_force_outcome"] == "not_evaluated"
    assert decision.trace["novelty_force_unavailable_reason"] == "stale_pose"
    assert adm.novelty_force_counts["triggered"] == 0


def test_guard_fails_closed_without_a_velocity_sample():
    """``velocity_adaptive=False`` means no per-candidate velocity sample
    exists at all. Fail closed -- never fall through unguarded."""
    adm = _admitter(_cfg(velocity_adaptive=False), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.trace["novelty_force_unavailable_reason"] == "stale_pose"


def test_guard_can_be_disabled_explicitly():
    """...and the operator can opt out, which is what makes the default
    meaningful rather than an unremovable constraint."""
    adm = _admitter(
        _cfg(velocity_adaptive=False, novelty_force_require_fresh_pose=False),
        _ALL_UNOCCUPIED,
    )
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.admitted
    assert decision.reason == "novelty_admit"


def test_novelty_force_keeps_none_cause_when_coverage_is_unavailable():
    """A novelty-force abstention caused by failed depth evaluation is not a
    generic ``frac_unavailable`` event."""
    adm = KeyframeAdmitter(
        config=_cfg(velocity_adaptive=False, novelty_force_require_fresh_pose=False),
        render_fn=stub_render(0.99),
        occupancy_fn=_ALL_UNOCCUPIED,
        load_depth_m=lambda frame: None,
    )
    adm.note_train_frame(_frame(0, 0.0))
    decision, _ = _advance(adm, 1, 0.001)

    assert decision.trace["novelty_force_outcome"] == "not_evaluated"
    assert decision.trace["novelty_force_unavailable_reason"] == "frac_unavailable"
    assert decision.trace["novelty_force_none_cause"] == "no_depth"
    assert decision.trace["coverage_consumers"]["novelty_force"]["none_cause"] == "no_depth"


def test_max_gap_candidate_keeps_its_own_trigger():
    """No double-counting (the #932 class): ``candidate_trigger`` stays
    single-valued, so the precedence buckets stay mutually exclusive."""
    adm = _admitter(_cfg(max_gap=5), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 20, 0.002)   # gap 20 >= max_gap 5
    assert decision.admitted
    assert decision.trace["candidate_trigger"] in (
        ForcePath.SOFT_MAX_GAP.value,
        ForcePath.HARD_MAX_GAP_CEILING.value,
    )
    assert decision.trace["novelty_force_outcome"] == "not_evaluated"
    assert decision.trace["novelty_force_unavailable_reason"] == "not_normal_trigger"
    assert decision.novelty_forced is False
    assert adm.novelty_force_counts["triggered"] == 0


def test_unwired_coverage_machinery_cannot_fire():
    adm = KeyframeAdmitter(
        config=_cfg(),
        render_fn=stub_render(0.99),
        occupancy_fn=None,
        load_depth_m=None,
    )
    adm.note_train_frame(_frame(0, 0.0))
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.trace["novelty_force_unavailable_reason"] == "machinery_unwired"


def test_pre_baseline_candidate_cannot_fire():
    """Before the first keyframe the pose deltas have no reference and the #933
    starvation clock owns admission; the trigger must stay out of it."""
    adm = KeyframeAdmitter(
        config=_cfg(),
        render_fn=stub_render(0.99),
        occupancy_fn=_ALL_UNOCCUPIED,
        load_depth_m=_depth_const(2.0),
    )
    # NOTE: no note_train_frame() -> _last_train_c2w is None.
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.trace["novelty_force_unavailable_reason"] == "no_baseline"


def test_moving_candidate_skips_the_trigger_entirely():
    """When motion would admit anyway the trigger is moot -- and skipping it is
    also what keeps the hoisted coverage backprojection off the already-
    admitting candidates."""
    # NB the velocity-adaptive clamps, not min_translation_m, are what the
    # effective threshold resolves to on this harness -- lower both floor and
    # ceiling or the candidate stays "not moved" however far it travels.
    adm = _admitter(
        _cfg(
            min_translation_m=0.001,
            velocity_adaptive_min_trans_floor=1e-6,
            velocity_adaptive_min_trans_ceil=1e-5,
        ),
        _ALL_UNOCCUPIED,
    )
    _advance(adm, 1, 0.5)
    decision, _ = _advance(adm, 2, 1.0)
    assert decision.admitted
    assert decision.reason == "pose_delta"
    assert decision.trace["novelty_force_unavailable_reason"] == "motion_already_ok"


def test_inverted_threshold_config_is_clamped_and_warned():
    """``novelty_force_frac <= min_new_voxel_frac`` would let a novelty-forced
    candidate be admitted past the motion gate and then rejected by the
    coverage gate -- a trigger that fires and admits nothing. Clamp + warn;
    never raise (a hard error would retroactively reject existing specs)."""
    cfg = _cfg(min_new_voxel_frac=0.5, novelty_force_frac=0.2)
    assert cfg.novelty_force_frac > cfg.min_new_voxel_frac
    adm = _admitter(cfg, _ALL_UNOCCUPIED)
    assert any("novelty_force_frac" in w for w in adm.precedence_config_warnings)


def test_no_clamp_when_the_feature_is_off():
    """A spec that leaves the feature off must not be rewritten by it."""
    cfg = _cfg(novelty_force_admit=False, min_new_voxel_frac=0.5, novelty_force_frac=0.2)
    assert cfg.novelty_force_frac == 0.2
    assert cfg.config_warnings == []


def test_information_gates_still_bind_on_a_novelty_forced_candidate():
    """The trigger relaxes MOTION only. A blur-gate rejection must still stand
    -- otherwise this is a second unconditional force, not a trigger."""
    adm = _admitter(
        _cfg(blur_gate_enabled=True, blur_gate_min_laplacian=1e9,
             blur_gate_force_admit_iters=10_000),
        _ALL_UNOCCUPIED,
    )
    # A flat grey RGB has ~zero Laplacian variance -> blur-rejected.
    adm._load_rgb = lambda frame: np.full((16, 16, 3), 128, dtype=np.uint8)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.reason == "blur_reject"
    assert decision.trace["novelty_force_outcome"] == "triggered"
    assert adm.novelty_force_counts["decisive"] == 0
    # PR #957 review: the flag must SURVIVE a gate rejection. Without this the
    # sidecar's `novelty_forced` column reads 0 on a row whose own
    # `precedence.novelty_force_outcome` says "triggered" -- the two columns
    # flatly disagree and the triggered-but-eaten population (the whole point
    # of the triggered-vs-decisive gap) is under-reported.
    assert decision.novelty_forced is True


def test_parallax_gate_rejection_keeps_the_novelty_forced_flag():
    """The parallax gate is the third exit a TRIGGERED candidate can reach.

    It can only bind for a triggered candidate when
    ``novelty_force_frac < 1 - parallax_covis_max`` (the gate needs
    ``covis_frac = 1 - new_voxel_frac >= parallax_covis_max``). That is legal
    config: ``__post_init__`` only enforces
    ``novelty_force_frac > min_new_voxel_frac``. At ``novelty_force_frac=0.3``
    and a measured fraction of 0.375 both conditions hold at once.
    """
    adm = _admitter(
        _cfg(
            novelty_force_frac=0.3,
            min_new_voxel_frac=0.1,
            parallax_gate=True,
            parallax_covis_max=0.6,
            parallax_min_deg=3.0,
        ),
        _partly_occupied(0.65),
    )
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert not decision.admitted
    assert decision.reason == "rej_parallax"
    assert decision.trace["novelty_force_outcome"] == "triggered"
    assert decision.novelty_forced is True
    assert adm.novelty_force_counts["decisive"] == 0


def test_withdrawn_max_gap_force_reaches_the_trigger_and_does_not_fire():
    """Production default path: ``max_gap_require_novelty=True``.

    The test harness pins it False, but the real CLI default is True, so this
    block runs on every real training run since 2026-08-15. A would-be
    max_gap-forced candidate found REDUNDANT has its force withdrawn
    (``forced=False``, ``trigger`` restored to NORMAL) and therefore falls into
    the novelty block -- the one reachable path where a max_gap candidate does
    reach the trigger.

    It is deterministically inert there: the same measured fraction that was
    ``< min_new_voxel_frac`` is necessarily ``< novelty_force_frac`` too (the
    ``__post_init__`` invariant), so the outcome is ``below_threshold``, never
    ``triggered``. Pinned here so a future reordering cannot turn the
    withdrawal into a re-admission.
    """
    adm = _admitter(
        _cfg(
            max_gap=2,
            max_gap_hard_mult=4.0,
            max_gap_require_novelty=True,
            coverage_gain_gate=True,
            min_new_voxel_frac=0.1,
            novelty_force_frac=0.6,
        ),
        _ALL_OCCUPIED,
    )
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    # The force really was withdrawn...
    assert decision.trace["novelty_gate_outcome"] == "evaluated_redundant"
    assert decision.trace["candidate_trigger"] == ForcePath.NORMAL.value
    assert decision.forced is False
    # ...and the trigger really did evaluate, and really did not fire.
    assert decision.trace["novelty_force_outcome"] == "below_threshold"
    assert decision.novelty_forced is False
    assert not decision.admitted
    assert adm.novelty_force_counts["triggered"] == 0


def test_trigger_works_without_the_coverage_gain_gate():
    """The trigger requires ``occupancy_fn`` + ``load_depth_m``, NOT
    ``coverage_gain_gate`` -- so ``coverage_gain_gate=False`` with the trigger
    on is a real supported config, in which step 1c's reject branch does not
    even exist."""
    adm = _admitter(_cfg(coverage_gain_gate=False), _ALL_UNOCCUPIED)
    _advance(adm, 1, 0.001)
    decision, _ = _advance(adm, 2, 0.002)
    assert decision.admitted
    assert decision.reason == "novelty_admit"
    assert decision.novelty_forced is True
    assert decision.forced is False
    assert decision.trace["novelty_force_outcome"] == "triggered"
    assert adm.novelty_force_counts["decisive"] == 1


def test_default_off_leaves_the_trigger_inert():
    """Every existing spec: the flag is off, so the trigger never evaluates and
    never pays the coverage cost."""
    adm = _admitter(_cfg(novelty_force_admit=False), _ALL_UNOCCUPIED)
    for i in range(1, 6):
        decision, _ = _advance(adm, i, 0.0)
        assert decision.trace["novelty_force_outcome"] == "not_evaluated"
        assert decision.trace["novelty_force_unavailable_reason"] == "disabled"
    assert adm.novelty_force_counts["triggered"] == 0
