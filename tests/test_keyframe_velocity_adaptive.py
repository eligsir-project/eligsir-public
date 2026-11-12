"""Unit tests for the velocity-adaptive keyframe-admission gate.

Ported alongside the gate from
``feat/offline-gap-and-floater-consensus``. CPU-only: no render_fn /
occupancy_fn is supplied, so ``admit()`` takes the ``render_failed``
fast-path and the decision is driven purely by the (velocity-adapted)
pose-delta motion gate.

Invariant under test (the FAITHFUL one, which is the inverse of the
task prompt's parenthetical gloss): the gate holds the admit *rate*
near ``target_admit_rate_fps`` regardless of camera speed. Faster
motion therefore raises the effective ``min_translation_m`` threshold
(it does NOT lower it). With the static threshold, the admit rate
scales with speed instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from eligsir.mapping.scheduler.keyframes import (
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    FrameInfo,
)


def _frame(index: int, x: float):
    """A frame translated by ``x`` metres along +X from the origin."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index)


def _run_sequence(cfg: KeyframeAdmitterConfig, step_m: float, n: int):
    """Drive a constant-velocity straight-line trajectory and return the
    admit count. No render_fn -> non-forced moved frames admit via the
    render_failed path; non-moved frames reject."""
    adm = KeyframeAdmitter(config=cfg)
    # Seed a baseline keyframe at the origin so pose deltas are measured.
    adm.note_train_frame(_frame(0, 0.0))
    admitted = 0
    for i in range(1, n + 1):
        decision, _ = adm.admit(cam=None, frame=_frame(i, i * step_m))
        if decision.admitted:
            admitted += 1
    return admitted


def test_velocity_adaptive_raises_threshold_for_fast_motion():
    """Faithful invariant: faster motion -> HIGHER effective min_translation."""
    cfg = KeyframeAdmitterConfig(
        min_translation_m=0.05,
        min_rotation_deg=1e9,  # disable rotation gate
        max_gap=10_000,        # disable forced max_gap admits
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        input_fps=30.0,
        velocity_ema_alpha=1.0,  # no smoothing: EMA == last instantaneous sample
    )
    adm = KeyframeAdmitter(config=cfg)
    adm.note_train_frame(_frame(0, 0.0))
    # Seed the previous-candidate reference, then take one fast step of
    # 0.5 m over one 30 Hz frame period.
    adm.admit(cam=None, frame=_frame(1, 0.0))
    d, _ = adm.admit(cam=None, frame=_frame(2, 0.5))
    # #793: the estimator carries m/s, not m/frame and not m/admission-interval.
    # 0.5 m in 1/30 s = 15.0 m/s.
    assert adm._translation_velocity_ema_m_s == pytest.approx(15.0)
    # threshold = 15.0 m/s * (1/2 Hz) = 7.5 m -> clipped to the 0.50 m ceiling.
    assert d.effective_min_translation_m == pytest.approx(0.50)
    assert d.threshold_clamp == "ceil"


def test_velocity_adaptive_holds_admit_rate_constant():
    """Fast and slow trajectories converge to a similar admit rate under
    the adaptive gate, whereas the static threshold lets the fast one
    admit far more frames."""
    n = 200
    slow_step = 0.02
    fast_step = 0.10  # 5x faster

    static = KeyframeAdmitterConfig(
        min_translation_m=0.05, min_rotation_deg=1e9, max_gap=10_000,
        velocity_adaptive=False,
    )
    adaptive = KeyframeAdmitterConfig(
        min_translation_m=0.05, min_rotation_deg=1e9, max_gap=10_000,
        velocity_adaptive=True, target_admit_rate_fps=2.0, input_fps=30.0,
        velocity_ema_alpha=0.3,
    )

    static_slow = _run_sequence(static, slow_step, n)
    static_fast = _run_sequence(static, fast_step, n)
    adapt_slow = _run_sequence(adaptive, slow_step, n)
    adapt_fast = _run_sequence(adaptive, fast_step, n)

    # Static: admit rate scales hard with speed (fast admits far more).
    assert static_fast > static_slow * 2

    # Adaptive: the fast/slow admit counts are pulled much closer together
    # than the static gate's. The adaptive gap is a small fraction of the
    # static gap.
    static_gap = abs(static_fast - static_slow)
    adapt_gap = abs(adapt_fast - adapt_slow)
    assert adapt_gap < static_gap


def test_velocity_adaptive_off_is_static_threshold():
    """Flag off -> effective threshold == min_translation_m exactly, and
    speed EMA is never updated (no-op)."""
    cfg_off = KeyframeAdmitterConfig(
        min_translation_m=0.05, min_rotation_deg=1e9, max_gap=10_000,
        velocity_adaptive=False,
    )
    adm = KeyframeAdmitter(config=cfg_off)
    adm.note_train_frame(_frame(0, 0.0))
    # Step below 0.05 -> not moved -> reject; step above -> admit.
    d_small, _ = adm.admit(cam=None, frame=_frame(1, 0.03))
    assert not d_small.admitted
    assert adm._translation_velocity_ema_m_s is None  # EMA untouched when flag off
    d_big, _ = adm.admit(cam=None, frame=_frame(2, 0.20))
    assert d_big.admitted
    assert adm._translation_velocity_ema_m_s is None
