from eligsir.utils.seed_fade_monitor import SeedFadeMonitor


def test_no_fade_ever_keeps_scale_at_one_and_detected_false():
    mon = SeedFadeMonitor()
    for i, psnr in enumerate([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]):
        mon.update(i * 100, psnr)
        assert mon.suppression_scale() == 1.0
    block = mon.report_block()
    assert block["detected"] is False
    assert block["recovered"] is False
    assert block["controller_engaged"] is False
    assert block["max_dip_db"] == 0.0
    assert block["fade_start_iter"] is None


def test_single_noisy_dip_does_not_trigger_fade():
    mon = SeedFadeMonitor(tolerance_db=2.0, sustained_checks=3)
    # Rises to 20, one noisy single-sample dip to 10 (gap=10 > tolerance),
    # but recovers immediately -- must NOT count as a sustained fade.
    mon.update(0, 15.0)
    mon.update(100, 20.0)
    mon.update(200, 10.0)  # single dip
    mon.update(300, 20.0)  # immediate recovery
    assert mon.fade_active is False
    assert mon.report_block()["detected"] is False


def test_sustained_dip_triggers_fade_and_ramps_scale_down():
    mon = SeedFadeMonitor(tolerance_db=2.0, sustained_checks=3, ramp_down_rate=0.15)
    mon.update(0, 20.0)  # running max = 20.0
    scale_before = mon.suppression_scale()
    assert scale_before == 1.0
    assert mon.fade_active is False

    # Three consecutive updates below tolerance (20.0 - 5.0 = 15.0 gap > 2.0 tolerance)
    mon.update(100, 5.0)
    mon.update(200, 5.0)
    mon.update(300, 5.0)
    assert mon.fade_active is True

    scale_1 = mon.suppression_scale()
    scale_2 = mon.suppression_scale()
    assert scale_1 < scale_before
    assert scale_2 < scale_1  # ramps down further while still active

    block = mon.report_block()
    assert block["detected"] is True
    assert block["fade_start_iter"] == 300  # iteration where the 3rd consecutive check triggered
    assert block["max_dip_db"] >= 14.9  # ~15.0 gap


def test_recovery_ramps_scale_back_up():
    mon = SeedFadeMonitor(tolerance_db=2.0, sustained_checks=3, ramp_down_rate=0.15, ramp_up_rate=0.05)
    mon.update(0, 20.0)
    mon.update(100, 5.0)
    mon.update(200, 5.0)
    mon.update(300, 5.0)
    assert mon.fade_active is True
    scale_during_fade = mon.suppression_scale()
    assert scale_during_fade < 1.0

    # Recovery: back within tolerance of running max
    mon.update(400, 19.0)
    assert mon.fade_active is False

    scale_after = mon.suppression_scale()
    assert scale_after > scale_during_fade  # ramping back up

    block = mon.report_block()
    assert block["detected"] is True  # fade_ever_detected stays True even after recovery
    assert block["recovered"] is True
    assert block["controller_engaged"] is True


def test_warmup_ignores_early_anomalous_peak():
    # An early, anomalously easy render (e.g. a near-empty bootstrap-frame
    # seed view) sets an unrealistic running_max that the scene's real
    # steady-state quality never again reaches -- without a warmup this
    # reads as a permanent fade. Updates before warmup_iters must be
    # ignored entirely (not used to seed running_max, no streak credit).
    mon = SeedFadeMonitor(tolerance_db=2.0, sustained_checks=3, warmup_iters=1000)
    mon.update(500, 18.59)  # anomalous early peak, before warmup ends
    mon.update(600, 10.0)
    mon.update(700, 10.0)
    mon.update(800, 10.0)
    assert mon.fade_active is False
    assert mon.report_block()["detected"] is False

    # Post-warmup, a real sustained dip still triggers normally.
    mon.update(1000, 10.0)  # first post-warmup sample seeds running_max = 10.0
    mon.update(1100, 2.0)
    mon.update(1200, 2.0)
    mon.update(1300, 2.0)
    assert mon.fade_active is True
