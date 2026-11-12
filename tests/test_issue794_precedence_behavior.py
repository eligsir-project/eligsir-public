"""Issue #794 -- BEHAVIOURAL contract violations, runnable against ``main``.

Companion to ``tests/test_issue794_admission_precedence.py``. That file imports
:mod:`eligsir.mapping.scheduler.precedence`, so on unmodified ``main`` it
fails at *collection* -- which proves the module is new, not that the contract
is violated. This file deliberately imports **nothing new**: it configures the
opt-in scope by ``setattr`` on the config object (a no-op field on ``main``,
where it is simply an unused attribute), so every failure here is a genuine
behavioural contract violation:

    on main -> a frame that is blurry / badly exposed / has almost no overlap
               with the existing map is ADMITTED anyway, purely because
               ``gap >= max_gap``.

Each test states the ``main`` behaviour it fails against in its docstring.
"""
from __future__ import annotations

import numpy as np

from _fake_alpha import stub_render  # noqa: E402
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


def _stub_render(overlap_value: float):
    # Torch-free: this file must stay collectable (and PASSING) in the CPU
    # smoke image, which has no torch. See tests/_fake_alpha.py for why the
    # previous inline `import torch` silently broke four tests here.
    return stub_render(overlap_value)


def _cfg_motion_only(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,
        max_gap=10,
        max_gap_hard_mult=4.0,
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
        motion_pregate=True,
        velocity_adaptive=False,
        min_overlap=0.25,
        max_overlap=0.90,
    )
    base.update(kw)
    cfg = KeyframeAdmitterConfig(**base)
    # #794 opt-in. On main this attribute does not exist on the dataclass and
    # setting it changes nothing -- which is exactly the point.
    setattr(cfg, "max_gap_force_scope", "motion_only")
    return cfg


def _admitter(cfg, overlap, **kw):
    a = KeyframeAdmitter(cfg, render_fn=_stub_render(overlap), **kw)
    a.note_train_frame(_frame(0, 0.0))
    return a


def test_soft_max_gap_must_not_bypass_the_blur_gate():
    """main: ADMITS with reason 'max_gap' -- the blur gate is never reached,
    because the ``if forced:`` fast path returns before it."""
    a = _admitter(
        _cfg_motion_only(
            blur_gate_enabled=True,
            blur_gate_min_laplacian=1e9,
            blur_gate_force_admit_iters=10**6,
        ),
        overlap=0.5,
        load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32),
    )
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert not d.admitted, f"blurry forced frame was admitted ({d.reason})"
    assert d.reason == "blur_reject"


def test_soft_max_gap_must_not_bypass_the_exposure_gate():
    """main: ADMITS with reason 'max_gap' -- exposure gate never reached."""
    a = _admitter(
        _cfg_motion_only(
            exposure_gate_enabled=True,
            exposure_max_delta=0.01,
            exposure_force_admit_iters=10**6,
        ),
        overlap=0.5,
        load_rgb=lambda f: np.full((8, 8, 3), 0.9, dtype=np.float32),
    )
    a._last_admitted_luma = 0.05
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert not d.admitted, f"exposure-jumped forced frame was admitted ({d.reason})"
    assert d.reason == "exposure_reject"


def test_soft_max_gap_must_not_bypass_the_render_overlap_gate():
    """main: ADMITS with reason 'max_gap' -- render-overlap gate never reached.
    This is the one bypass that is LIVE on stock CLI defaults (parallax /
    exposure / blur / quality all default off; the render gate does not)."""
    a = _admitter(_cfg_motion_only(), overlap=0.05)
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert not d.admitted, f"low-overlap forced frame was admitted ({d.reason})"
    assert d.reason == "low_overlap"


def test_soft_max_gap_must_not_bypass_the_parallax_gate():
    """main: ADMITS with reason 'max_gap'. The parallax gate even carries an
    explicit ``not forced`` guard -- dead code today, since the forced fast
    path returns long before it."""
    occupied = lambda pts, vs: np.ones(len(pts), dtype=bool)
    a = _admitter(
        _cfg_motion_only(
            coverage_gain_gate=False,
            parallax_gate=True,
            parallax_min_deg=1e9,     # every parallax counts as too small
            parallax_covis_max=0.0,   # every covisibility counts as too high
            coverage_gain_depth_stride=4,
            depth_max=8.0,
        ),
        overlap=0.5,
        occupancy_fn=occupied,
        load_depth_m=lambda f: np.full((16, 16), 2.0, dtype=np.float32),
    )
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert not d.admitted, f"zero-parallax forced frame was admitted ({d.reason})"
    assert d.reason == "rej_parallax"


def test_hard_ceiling_still_forces_unconditionally():
    """Passes on BOTH main and this branch -- pins that the #794 change does
    not weaken the absolute starvation backstop."""
    a = _admitter(
        _cfg_motion_only(
            blur_gate_enabled=True,
            blur_gate_min_laplacian=1e9,
            blur_gate_force_admit_iters=10**6,
        ),
        overlap=0.05,
        load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32),
    )
    d, _ = a.admit(_FakeCam(), _frame(40, 0.0))
    assert d.admitted and d.reason == "max_gap"


def test_soft_escalation_does_not_reinstate_starvation():
    """Passes on BOTH: a stationary frame over healthy map coverage is still
    admitted. On main via the unconditional force, on this branch via the
    motion-satisfied semantics -- either way the backstop works."""
    a = _admitter(_cfg_motion_only(), overlap=0.95)
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert d.admitted


def test_one_admit_call_carries_both_the_794_trace_and_the_793_velocity_fields():
    """REGRESSION GUARD for the #926/#927 merge (the two-layer nesting).

    PRs #926 (issue #793) and #927 (issue #794) each independently split
    ``KeyframeAdmitter.admit()`` into a public wrapper plus an inner
    implementation, for the same reason -- stamping recording-only data onto
    the returned decision -- with different names (``_admit_inner`` vs
    ``_admit_impl``). The two recording layers must be COMBINED, not chosen
    between. ``admit()`` therefore owns both, in this order:

        admit()
          -> _admit_impl()  : the real decision body
          -> stamp the #793/#795 velocity telemetry onto the decision
          -> _finalize_trace(): attach the #794 PrecedenceTrace

    The trace must be finalized LAST, so ``_finalize_trace`` observes a
    decision whose every field -- velocity telemetry included -- is already
    populated.

    Taking either side of that conflict wholesale silently deletes the other
    PR's entire recording layer while leaving every one of that PR's own tests
    green (each suite only ever exercises its own layer).

    This test is the discriminator: it asserts BOTH layers are populated on the
    return value of a SINGLE ``admit()`` call. Asserting them across two
    separate calls would NOT catch the failure mode.
    """
    cfg = KeyframeAdmitterConfig(
        min_translation_m=0.05,
        min_rotation_deg=1e9,       # static rotation criterion off
        max_gap=10_000,             # no max_gap forcing
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
        velocity_adaptive=True,     # required for the #793 fields to populate
        target_admit_rate_fps=2.0,
        input_fps=30.0,
        velocity_ema_alpha=1.0,     # EMA == last instantaneous sample
    )
    # Timestamped, constant-velocity candidates so a real velocity sample (and
    # therefore a real EMA) exists by the time the asserted call happens.
    frames = [
        FrameInfo(c2w=_frame(i, x=i * 0.01).c2w, index=i, timestamp=i / 30.0)
        for i in range(4)
    ]
    adm = KeyframeAdmitter(config=cfg)
    adm.note_train_frame(frames[0])
    for f in frames[1:-1]:
        adm.admit(cam=None, frame=f)

    # --- THE single call under test ---
    decision, _ = adm.admit(cam=None, frame=frames[-1])

    # Layer 1 (#794): the precedence trace.
    assert decision.trace is not None, (
        "the #794 precedence-trace layer is missing from admit() -- the merge "
        "resolution dropped the outer wrapper"
    )
    assert isinstance(decision.trace, dict)
    for key in ("candidate_trigger", "bypassed_gates", "evaluated_gates",
                "final_owner", "hard_force"):
        assert key in decision.trace, f"trace missing {key!r}: {decision.trace}"
    assert decision.trace["candidate_trigger"], "trace candidate_trigger unset"
    assert decision.trace["final_owner"], "trace final_owner unset (not finalized)"

    # Layer 2 (#793): the velocity-controller telemetry, on the SAME object.
    assert decision.translation_velocity_m_s is not None, (
        "the #793 velocity-telemetry layer is missing from admit() -- the merge "
        "resolution dropped the inner wrapper"
    )
    assert decision.translation_velocity_ema_m_s is not None
    assert decision.source_dt_s is not None
    assert decision.timebase_source != "unavailable"
    assert decision.effective_min_translation_m > 0.0
    assert decision.controller_target_interval_s is not None
    assert decision.threshold_clamp != "n/a", (
        "the adaptive law did not run -- this call cannot discriminate the "
        "velocity layer"
    )

    # And the ordering constraint the nesting exists to guarantee: the trace is
    # finalized AFTER the velocity fields are stamped, so _finalize_trace sees a
    # fully-populated decision. A trace built over a half-populated decision is
    # exactly what the wrong nesting order would produce.
    assert decision.trace["final_owner"] is not None


def test_default_scope_still_bypasses_every_gate():
    """PAIRED with the four tests above: the SAME admitter config, with the
    scope left at its default (i.e. exactly what every existing run gets),
    must still force-admit past the blur gate on gap alone.

    Passes on main and on this branch -- that is the point. Together with the
    four opt-in tests it shows the behaviour difference is entirely attributable
    to the flag, not to the refactor.
    """
    cfg = KeyframeAdmitterConfig(
        min_translation_m=0.05,
        min_rotation_deg=1e9,
        max_gap=10,
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
        blur_gate_enabled=True,
        blur_gate_min_laplacian=1e9,
        blur_gate_force_admit_iters=10**6,
    )
    # NOTE: deliberately no setattr of max_gap_force_scope here -- the whole
    # point is the untouched default.
    assert getattr(cfg, "max_gap_force_scope", "legacy_all_gates") == "legacy_all_gates"
    a = _admitter(cfg, overlap=0.05, load_rgb=lambda f: np.zeros((16, 16, 3), dtype=np.float32))
    d, _ = a.admit(_FakeCam(), _frame(10, 0.0))
    assert d.admitted and d.reason == "max_gap"
