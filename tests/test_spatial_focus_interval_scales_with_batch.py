from eligsir.runtime.production_application import spatial_replay_focus_interval_for_batch


def test_spatial_replay_focus_keeps_its_step_cadence_at_b16_and_b32():
    baseline = 25 / 3
    for batch_size in (4, 16, 32):
        interval = spatial_replay_focus_interval_for_batch(25, batch_size)
        assert abs(interval / max(1, batch_size - 1) - baseline) < 1.0
