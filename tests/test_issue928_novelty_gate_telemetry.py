"""Issue #928 -- explicit tri-state telemetry for the max_gap_require_novelty
gate (PR #826, ``keyframes.py``'s Component 2).

Before this fix, ``KeyframeDecision(forced=True, reason="max_gap")`` was
byte-identical whether the novelty gate had actually run and found the frame
novel, or had never run / abstained at all (gate disabled, occupancy_fn/
load_depth_m unwired, gap already past the hard ceiling, or
``_coverage_new_voxel_frac()`` returning ``None``). This is a REPORTING-ONLY
change: it must never alter which candidates are admitted -- every assertion
below about ``decision.admitted``/``decision.reason``/``decision.forced``
already existed in ``test_keyframe_max_gap_novelty.py`` and is repeated here
only to pin that this telemetry addition changed none of it.

CPU-only, reusing the exact fake camera / stub occupancy harness from
``test_keyframe_max_gap_novelty.py``.
"""
from __future__ import annotations

import numpy as np

from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
)


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


def _depth_const(value: float):
    arr = np.full((16, 16), value, dtype=np.float32)
    return lambda frame: arr


def _depth_none(frame):
    return None


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,        # disable rotation gate
        max_gap=10,
        coverage_gain_gate=True,
        min_new_voxel_frac=0.5,
        coverage_gain_depth_stride=4,
        depth_max=8.0,
        max_gap_require_novelty=True,
        max_gap_hard_mult=4.0,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


_UNSET = object()


def _make_admitter(cfg, occupancy_fn=None, load_depth_m=_UNSET):
    # A sentinel default (not `None`) so a test can explicitly request
    # `load_depth_m=None` (the "no_depth_loader" case) without it being
    # silently replaced by the usual stub depth loader.
    if load_depth_m is _UNSET:
        load_depth_m = _depth_const(2.0)
    adm = KeyframeAdmitter(
        config=cfg,
        occupancy_fn=occupancy_fn,
        load_depth_m=load_depth_m,
    )
    adm.note_train_frame(_frame(0, 0.0))   # baseline at index 0
    return adm


_ALL_OCCUPIED = lambda pts, vox: np.ones(len(pts), dtype=bool)
_ALL_UNOCCUPIED = lambda pts, vox: np.zeros(len(pts), dtype=bool)


# ---------------------------------------------------------------------------
# not_applicable: gate does not apply to an ordinary (non-forced) candidate.
# ---------------------------------------------------------------------------

def test_normal_candidate_is_not_applicable():
    """gap < max_gap -- this candidate was never a would-be force, so the
    novelty-gate tri-state stays at its 'not_applicable' default."""
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 1.0))
    assert decision.trace["novelty_gate_outcome"] == "not_applicable"
    assert decision.trace["novelty_gate_unavailable_reason"] is None
    assert decision.trace["novelty_gate_fraction"] is None
    assert adm.novelty_gate_outcome_counts == {}


# ---------------------------------------------------------------------------
# evaluated_novel / evaluated_redundant: the gate actually ran.
# ---------------------------------------------------------------------------

def test_novel_forced_frame_is_evaluated_novel():
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_UNOCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "evaluated_novel"
    assert decision.trace["novelty_gate_unavailable_reason"] is None
    assert decision.trace["novelty_gate_fraction"] == 1.0
    assert adm.novelty_gate_outcome_counts == {"evaluated_novel": 1}
    assert adm.novelty_gate_unavailable_reason_counts == {}


def test_redundant_forced_frame_is_evaluated_redundant():
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert not decision.admitted
    assert decision.reason != "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "evaluated_redundant"
    assert decision.trace["novelty_gate_unavailable_reason"] is None
    assert decision.trace["novelty_gate_fraction"] == 0.0
    assert adm.novelty_gate_outcome_counts == {"evaluated_redundant": 1}
    assert adm.novelty_gate_unavailable_reason_counts == {}


# ---------------------------------------------------------------------------
# not_evaluated, every reason.
# ---------------------------------------------------------------------------

def test_gate_disabled_by_max_gap_require_novelty_flag():
    """max_gap_require_novelty=False -> the gate is off; forced admits via
    the plain max_gap fast-path (occupancy never consulted), matching the
    pre-#928 'byte-identical' behavior test in
    test_keyframe_max_gap_novelty.py."""
    called = {"n": 0}

    def _spy(pts, vox):
        called["n"] += 1
        return np.ones(len(pts), dtype=bool)

    adm = _make_admitter(_cfg(max_gap_require_novelty=False), occupancy_fn=_spy)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert called["n"] == 0
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "gate_disabled"
    assert decision.trace["novelty_gate_fraction"] is None
    assert adm.novelty_gate_unavailable_reason_counts == {"gate_disabled": 1}


def test_gate_disabled_by_coverage_gain_gate_flag():
    """max_gap_require_novelty=True but coverage_gain_gate=False -- also
    'gate_disabled' (issue #928's investigation into the preflight message:
    this is the SAME reason, both are static-config knobs the gate needs)."""
    adm = _make_admitter(
        _cfg(coverage_gain_gate=False), occupancy_fn=_ALL_OCCUPIED
    )
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "gate_disabled"


def test_no_occupancy_fn():
    adm = _make_admitter(_cfg(), occupancy_fn=None)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "no_occupancy_fn"


def test_no_depth_loader():
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED, load_depth_m=None)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "no_depth_loader"


def test_hard_ceiling_is_not_evaluated():
    """gap >= max_gap*max_gap_hard_mult -- the unconditional starvation
    backstop always forces, and the novelty gate is never even consulted
    (matches test_keyframe_max_gap_novelty.py's
    test_hard_ceiling_forces_even_if_redundant, plus the new telemetry)."""
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(40, 0.0))
    assert decision.admitted and decision.reason == "max_gap" and decision.forced
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "hard_ceiling"


def test_frac_unavailable():
    """The gate runs (all wiring present) but _coverage_new_voxel_frac()
    itself returns None (here: load_depth_m returns None, one of several
    ways that can happen) -- forcing is unaffected (matches the pre-#928
    'nothing measured, forced stays True' behavior), but this is now
    distinguishable from 'evaluated_novel'."""
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED, load_depth_m=_depth_none)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_outcome"] == "not_evaluated"
    assert decision.trace["novelty_gate_unavailable_reason"] == "frac_unavailable"
    assert decision.trace["novelty_gate_none_cause"] == "no_depth"


def test_coverage_exception_has_a_distinct_none_cause():
    """A swallowed coverage error must remain visible in the decision trace."""
    def _raising_depth(frame):
        raise RuntimeError("synthetic depth failure")

    adm = _make_admitter(
        _cfg(), occupancy_fn=_ALL_OCCUPIED, load_depth_m=_raising_depth
    )
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))

    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_unavailable_reason"] == "frac_unavailable"
    assert decision.trace["novelty_gate_none_cause"] == "exception"


def test_coverage_with_too_few_valid_points_has_a_distinct_none_cause():
    """An under-populated depth sample is neither a missing depth nor an error."""
    sparse_depth = lambda frame: np.full((8, 8), 2.0, dtype=np.float32)
    adm = _make_admitter(
        _cfg(), occupancy_fn=_ALL_OCCUPIED, load_depth_m=sparse_depth
    )
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))

    assert decision.admitted and decision.reason == "max_gap"
    assert decision.trace["novelty_gate_unavailable_reason"] == "frac_unavailable"
    assert decision.trace["novelty_gate_none_cause"] == "too_few_valid_points"


def test_ordinary_coverage_gate_keeps_none_cause_when_depth_is_unavailable():
    """The ordinary coverage consumer must not collapse a failed measurement
    into a silent pass after the max-gap novelty veto learned the split."""
    adm = _make_admitter(
        _cfg(max_gap_require_novelty=False),
        occupancy_fn=_ALL_OCCUPIED,
        load_depth_m=_depth_none,
    )
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 1.0))

    coverage = decision.trace["coverage_consumers"]["ordinary_coverage_gate"]
    assert coverage["status"] == "unavailable"
    assert coverage["none_cause"] == "no_depth"


# ---------------------------------------------------------------------------
# Per-run counters accumulate across multiple decisions.
# ---------------------------------------------------------------------------

def test_per_run_counters_accumulate():
    adm = _make_admitter(_cfg(), occupancy_fn=_ALL_OCCUPIED)
    # frame 1: gap=1, not_applicable (not counted)
    adm.admit(cam=_FakeCam(), frame=_frame(1, 1.0))
    # frame 12: gap=11 >= max_gap(10), fully occupied -> evaluated_redundant
    adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert adm.novelty_gate_outcome_counts == {"evaluated_redundant": 1}
    assert adm.novelty_gate_unavailable_reason_counts == {}

    adm.reset()
    assert adm.novelty_gate_outcome_counts == {}
    assert adm.novelty_gate_unavailable_reason_counts == {}
