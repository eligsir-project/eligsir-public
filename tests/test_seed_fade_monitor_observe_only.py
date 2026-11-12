"""Issue #282: observation-only mode must observe without intervening.

The seed-fade detector both measures a fade AND throttles
lambda_eff_count / lambda_opacity_entropy in response to it. That coupling
means every historical run's fade data is confounded by the mitigation the
detector itself triggered, which makes the #281/#346 causal program
impossible to run cleanly. These tests pin the decoupling:

1. observe-only must be behaviorally inert on the loss weight;
2. it must NOT change detection (the two modes must see the identical fade);
3. the default must remain the pre-#282 behavior.
"""

import pytest

from eligsir.utils.seed_fade_monitor import (
    SeedFadeMonitor,
    STATUS_CLEAN,
    STATUS_CONFOUNDED,
    STATUS_FADE_OBSERVED,
    STATUS_NO_SAMPLES,
)


def _sustained_fade_series():
    # rise, then a long sustained collapse, then partial recovery
    return (
        [(i * 100, 20.0 + i * 0.1) for i in range(10)]
        + [(1000 + i * 100, 6.0) for i in range(30)]
        + [(4000 + i * 100, 20.5) for i in range(10)]
    )


def _drive(mon, series, calls_per_sample=5):
    """Mimic the real wiring: update() at the sampling cadence, and the
    suppression scale pulled once per training iteration in between."""
    for it, val in series:
        mon.update(it, val)
        for _ in range(calls_per_sample):
            mon.effective_suppression_scale()


def test_observe_only_never_changes_the_loss_weight():
    mon = SeedFadeMonitor(warmup_iters=0, observe_only=True)
    scales = []
    for it, val in _sustained_fade_series():
        mon.update(it, val)
        for _ in range(5):
            scales.append(mon.effective_suppression_scale())

    assert mon.report_block()["detected"] is True, "must still DETECT the fade"
    assert set(scales) == {1.0}, "observe-only must never scale the loss weight"
    assert mon.suppression_applied is False
    assert mon.min_applied_scale == 1.0
    assert mon._scale == 1.0, "the ramp must never even be advanced"
    assert mon.n_suppression_calls == 0


def test_controller_mode_does_scale_the_loss_weight():
    mon = SeedFadeMonitor(warmup_iters=0, observe_only=False)
    _drive(mon, _sustained_fade_series())
    assert mon.suppression_applied is True
    assert mon.min_applied_scale == 0.0  # ramps all the way down during the fade
    assert mon.n_suppression_calls > 0


def test_observe_only_detection_is_identical_to_controller_detection():
    """Decoupling means the MEASUREMENT is the same in both modes; only the
    intervention differs. (In a real run the two diverge because the
    intervention changes the scene -- but the detector state machine itself
    must be mode-independent given the same input.)"""
    series = _sustained_fade_series()
    a = SeedFadeMonitor(warmup_iters=0, observe_only=False)
    b = SeedFadeMonitor(warmup_iters=0, observe_only=True)
    _drive(a, series)
    _drive(b, series)
    ka = a.report_block()
    kb = b.report_block()
    detection_keys = [
        "detected", "max_dip_db", "fade_start_iter", "recovered",
        "first_fade_start_iter", "last_fade_start_iter", "episode_count",
        "fade_active_at_end", "recovered_final", "deepest_dip_db",
        "trough_iter", "trough_psnr", "n_samples", "n_samples_in_fade",
    ]
    for k in detection_keys:
        assert ka[k] == kb[k], f"{k} differs between modes: {ka[k]} vs {kb[k]}"


def test_default_is_controller_enabled_behavior_neutral():
    mon = SeedFadeMonitor()
    assert mon.observe_only is False
    assert mon.report_block()["observe_only"] is False


def test_status_field():
    assert SeedFadeMonitor().report_block()["status"] == STATUS_NO_SAMPLES

    clean = SeedFadeMonitor(warmup_iters=0)
    _drive(clean, [(i * 100, 20.0 + i * 0.1) for i in range(10)])
    assert clean.report_block()["status"] == STATUS_CLEAN
    assert clean.report_block()["suppression_confounded"] is False

    confounded = SeedFadeMonitor(warmup_iters=0)
    _drive(confounded, _sustained_fade_series())
    blk = confounded.report_block()
    assert blk["status"] == STATUS_CONFOUNDED
    assert blk["suppression_confounded"] is True
    assert blk["controller_engaged"] is True

    observed = SeedFadeMonitor(warmup_iters=0, observe_only=True)
    _drive(observed, _sustained_fade_series())
    blk = observed.report_block()
    assert blk["status"] == STATUS_FADE_OBSERVED
    assert blk["suppression_confounded"] is False, (
        "an observe-only run's fade is real but NOT confounded by the "
        "controller -- that is the entire point of the mode"
    )
    assert blk["controller_engaged"] is False, (
        "pre-#282 this key was a literal alias of `detected` and reported "
        "True even when no suppression was ever applied"
    )


def test_deepest_dip_can_exceed_max_dip_db():
    """`max_dip_db` is only sampled while the sustained-fade latch is OPEN, so
    a deep dip that recovers before `sustained_checks` samples is invisible to
    it. `deepest_dip_db`/`trough_iter` track the drawdown unconditionally.

    (On both audited real runs the two happen to coincide, so this synthetic
    case is what actually exercises the divergence.)"""
    mon = SeedFadeMonitor(tolerance_db=2.0, sustained_checks=3, warmup_iters=0)
    mon.update(0, 30.0)          # running max 30
    mon.update(100, 5.0)         # 25 dB drawdown, streak 1
    mon.update(200, 29.5)        # recovers -> streak resets, never latched
    for i in range(6):           # a shallow but SUSTAINED 4 dB fade
        mon.update(300 + i * 100, 26.0)
    blk = mon.report_block()
    assert blk["detected"] is True
    assert blk["max_dip_db"] == pytest.approx(4.0, abs=0.01)
    assert blk["deepest_dip_db"] == pytest.approx(25.0, abs=0.01)
    assert blk["trough_iter"] == 100
    assert blk["trough_psnr"] == pytest.approx(5.0, abs=0.01)


def test_controller_engaged_false_when_suppression_never_pulled():
    """A run whose loss path never asks for the scale (e.g. energy losses
    disabled) has, factually, no controller engagement -- even though a fade
    was detected. Pre-#282 this reported controller_engaged: True."""
    mon = SeedFadeMonitor(warmup_iters=0)
    for it, val in _sustained_fade_series():
        mon.update(it, val)
    blk = mon.report_block()
    assert blk["detected"] is True
    assert blk["controller_engaged"] is False
    assert blk["status"] == STATUS_FADE_OBSERVED
