from eligsir.mapping.scheduler import (
    FrameScheduler,
    resolve_ingestion_mode,
)


def test_auto_ingestion_mode_keeps_offline_replay_iteration_based():
    assert resolve_ingestion_mode("/data/tum/rgbd", "auto", online_mode=False) == "iter_based"


def test_auto_ingestion_mode_uses_dataset_fps_for_ros2_live():
    # ros2:// sources default to dataset_fps, not wallclock_strict: every
    # current launch path replays a bag via `ros2 bag play` (no --rate),
    # already real-time paced, so wallclock_strict is redundant and would
    # additionally drop frames on any transient training slowdown.
    assert resolve_ingestion_mode("ros2://live", "auto", online_mode=False) == "dataset_fps"


def test_explicit_wallclock_strict_still_available_for_ros2_live():
    assert resolve_ingestion_mode("ros2://live", "wallclock_strict", online_mode=False) == "wallclock_strict"


def test_auto_ingestion_mode_uses_dataset_fps_for_online_dataset_replay():
    assert resolve_ingestion_mode("/data/tum/rgbd", "auto", online_mode=True) == "dataset_fps"


def test_explicit_ingestion_mode_is_preserved():
    assert resolve_ingestion_mode("ros2://live", "iter_based", online_mode=True) == "iter_based"


def test_frame_scheduler_exposes_resolved_mode():
    scheduler = FrameScheduler(ingestion_mode="dataset_fps")
    assert scheduler.mode == "dataset_fps"


def test_utils_stream_scheduler_reexports_package_scheduler():
    from eligsir.utils.stream_scheduler import FrameScheduler as CompatFrameScheduler

    assert CompatFrameScheduler is FrameScheduler


# --- set_next_interval (issue #109 E3 adaptive-exposure arm) --------------

def test_default_iter_based_cadence_unaffected_by_new_method_existing():
    # Never calling set_next_interval() must reproduce the exact legacy
    # formula: one release every steps_per_frame iterations, first release
    # at iteration 1.
    scheduler = FrameScheduler(steps_per_frame=20, ingestion_mode="iter_based")
    released_at = []
    for it in range(1, 61):
        if scheduler.should_release(it):
            scheduler.mark_released()
            released_at.append(it)
    assert released_at == [1, 20, 40, 60]


def test_set_next_interval_overrides_next_release_point():
    scheduler = FrameScheduler(steps_per_frame=20, ingestion_mode="iter_based")
    assert scheduler.should_release(1)
    scheduler.mark_released()
    scheduler.set_next_interval(1, steps=5)  # next release forced to iter 6
    for it in range(2, 6):
        assert not scheduler.should_release(it)
    assert scheduler.should_release(6)


def test_set_next_interval_per_frame_makes_cadence_responsive_not_retroactive():
    # Simulates the E3 mechanism: after each release, the caller sizes the
    # NEXT interval only (based on that frame's insertion count) -- earlier
    # intervals must not be affected by a later call.
    scheduler = FrameScheduler(steps_per_frame=20, ingestion_mode="iter_based")
    released_at = []
    next_steps = iter([20, 5, 100, 1])  # first interval uses the base formula
    it = 0
    while len(released_at) < 4:
        it += 1
        if scheduler.should_release(it):
            scheduler.mark_released()
            released_at.append(it)
            scheduler.set_next_interval(it, next(next_steps))
    # release 1 always at iter 1 (legacy formula, no override set yet).
    # release 2 = 1 + 20 (first set_next_interval call, steps=20).
    # release 3 = 21 + 5. release 4 = 26 + 100.
    assert released_at == [1, 21, 26, 126]


def test_set_next_interval_clamps_steps_below_one_to_one():
    scheduler = FrameScheduler(steps_per_frame=20, ingestion_mode="iter_based")
    scheduler.set_next_interval(10, steps=0)
    assert not scheduler.should_release(10)
    assert scheduler.should_release(11)


def test_set_next_interval_has_no_effect_outside_iter_based_mode():
    scheduler = FrameScheduler(ingestion_mode="dataset_fps", fps_cap=30.0)
    scheduler.set_next_interval(1, steps=5)
    # dataset_fps mode ignores _next_release_iter entirely; should_release
    # still depends purely on simulated time vs due_time.
    assert scheduler.should_release(1, dt=0.0)  # due_time=0 at frame idx 0
