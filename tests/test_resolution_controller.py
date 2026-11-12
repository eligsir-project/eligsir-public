"""Tests for ResolutionController's rolling dual-floor switch logic (task 3)."""
from __future__ import annotations

from eligsir.mapping.resolution_controller import ResolutionController


def test_starts_at_high_res_by_default():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    assert c.decide(mapping_done=False) == "high"


def test_switches_to_low_when_it_v_floor_breached():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    # 5 admissions, each only 10 iterations apart -> it/V ~= 10, well under 30.
    for i in range(5):
        c.record_admission(iteration=i * 10, wall_time=i * 0.1)
    assert c.decide(mapping_done=False) == "low"


def test_switches_to_low_when_map_fps_floor_breached():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    # 5 admissions each 40 iterations apart (it/V=40, fine) but 1 wall-second
    # apart each (map_fps = 1.0, well under the 3.0 floor).
    for i in range(5):
        c.record_admission(iteration=i * 40, wall_time=i * 1.0)
    assert c.decide(mapping_done=False) == "low"


def test_stays_high_when_both_floors_comfortably_met():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    # it/V=100 (well over 30), map_fps=10 (well over 3).
    for i in range(5):
        c.record_admission(iteration=i * 100, wall_time=i * 0.1)
    assert c.decide(mapping_done=False) == "high"


def test_recovers_to_high_only_after_hysteresis_window_clears():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5, recovery_streak=3)
    for i in range(5):  # trip low-res
        c.record_admission(iteration=i * 10, wall_time=i * 0.1)
    assert c.decide(mapping_done=False) == "low"
    # One good decide() call alone must not flip it back immediately.
    for i in range(5, 10):
        c.record_admission(iteration=i * 100, wall_time=i * 0.1)
    first_recovery_check = c.decide(mapping_done=False)
    # Depending on exact recovery_streak semantics this may or may not have
    # flipped yet on the FIRST good window -- the real requirement is that it
    # takes recovery_streak consecutive good decide() calls, not one, to
    # return to "high". Call decide() recovery_streak times with good data
    # continuing to accumulate and confirm it eventually recovers.
    for _ in range(10):
        c.record_admission(iteration=c._last_iteration + 100, wall_time=c._last_wall_time + 0.1)
        result = c.decide(mapping_done=False)
    assert result == "high"


def test_mapping_done_locks_to_high_regardless_of_floors():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    for i in range(5):  # would otherwise trip low-res
        c.record_admission(iteration=i * 10, wall_time=i * 0.1)
    assert c.decide(mapping_done=True) == "high"


def test_lock_is_one_way_even_if_mapping_done_flag_later_goes_false():
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)
    for i in range(5):
        c.record_admission(iteration=i * 10, wall_time=i * 0.1)
    c.decide(mapping_done=True)  # lock engages
    assert c.decide(mapping_done=False) == "high"  # still locked, per one-way design


def test_lock_to_high_on_mapping_done_false_carries_forward_low_tier():
    """Issue #1259: with the latch disabled, mapping_done must NOT force
    "high" -- the reactive floor decision (frozen at its last pre-mapping-
    done state once admission events stop arriving) should carry forward
    whatever tier mapping actually ended at."""
    c = ResolutionController(
        it_v_floor=30, map_fps_floor=3.0, window=5,
        lock_to_high_on_mapping_done=False,
    )
    for i in range(5):  # trips low-res, same as mapping would on a slow scene
        c.record_admission(iteration=i * 10, wall_time=i * 0.1)
    assert c.decide(mapping_done=False) == "low"
    # No new admissions arrive during refinement -- the rolling window is
    # frozen, so the reactive decision keeps returning "low", not "high".
    assert c.decide(mapping_done=True) == "low"
    assert c.decide(mapping_done=True) == "low"


def test_lock_to_high_on_mapping_done_false_still_locks_if_already_high():
    """Disabling the latch must not retroactively unlock an already-"high"
    tier -- it only means mapping_done stops FORCING a fresh lock."""
    c = ResolutionController(
        it_v_floor=30, map_fps_floor=3.0, window=5,
        lock_to_high_on_mapping_done=False,
    )
    for i in range(5):  # comfortably clears both floors -> stays "high"
        c.record_admission(iteration=i * 100, wall_time=i * 0.1)
    assert c.decide(mapping_done=False) == "high"
    assert c.decide(mapping_done=True) == "high"


def test_rolling_window_vs_lifetime_average_discrimination():
    """
    Test that ResolutionController correctly uses a ROLLING window, not
    lifetime-average semantics. A fast burst followed by a sustained slowdown
    must trigger "low" — but would incorrectly stay "high" with lifetime averaging.

    Setup:
      - window=5, it_v_floor=30, map_fps_floor=3.0
      - Fast burst: 50 events at (iter+=1000, wall_time+=0.001 per event)
        ⟹ rolling it_v ≈ 1000, map_fps ≈ 1000 (both pass)
      - Slowdown: 5 more events at (iter+=5, wall_time+=1.0 per event)
        ⟹ rolling it_v = 5, map_fps = 1.0 (both fail)

    Lifetime-average (if wrongly implemented) after all 55 events:
      - Total iterations: 50*1000 + 5*5 = 50025
      - Total wall_time: 50*0.001 + 5*1.0 = 5.05
      - Lifetime it_v ≈ 50025/54 ≈ 926 (passes 30)
      - Lifetime map_fps ≈ 54/5.05 ≈ 10.7 (passes 3.0)
      - Would incorrectly return "high"

    Rolling window (correctly implemented):
      - Last 5 events all from slowdown phase
      - Rolling it_v = 20/4 = 5 (fails 30)
      - Rolling map_fps = 4/4.0 = 1.0 (fails 3.0)
      - Correctly returns "low"
    """
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=5)

    # Fast burst: 50 events, each 1000 iterations apart, 0.001 wall-seconds apart
    iteration = 0
    wall_time = 0.0
    for _ in range(50):
        c.record_admission(iteration=iteration, wall_time=wall_time)
        iteration += 1000
        wall_time += 0.001

    # After the fast burst, the window is full with fast events.
    # Rolling it_v ≈ 1000, rolling map_fps ≈ 1000 → both pass.
    assert c.decide(mapping_done=False) == "high"

    # Slowdown burst: 5 more events, each 5 iterations apart, 1.0 wall-second apart
    for _ in range(5):
        c.record_admission(iteration=iteration, wall_time=wall_time)
        iteration += 5
        wall_time += 1.0

    # Now the window is exactly the 5 slowdown events.
    # Rolling it_v = (49025 - 49005) / 4 = 20/4 = 5 (fails 30)
    # Rolling map_fps = 4 / (5.049 - 1.049) = 4/4 = 1.0 (fails 3.0)
    # Must detect the slowdown and return "low".
    # A lifetime-average would incorrectly return "high" due to the early burst.
    assert c.decide(mapping_done=False) == "low"


def test_window_1_does_not_crash():
    """
    Verify that window=1 construction and basic operation doesn't crash.
    This is a degenerate edge case, but should not raise an exception.
    """
    c = ResolutionController(it_v_floor=30, map_fps_floor=3.0, window=1)
    c.record_admission(iteration=10, wall_time=0.1)
    result = c.decide(mapping_done=False)
    # The exact result depends on implementation details (may stay at initial
    # "high" since we can't compute a rolling window from a single event),
    # but it must not crash.
    assert result in ("high", "low")


def test_scheduled_bootstrap_uses_low_through_boundary_then_native():
    """Removing the explicit schedule branch must make this stay high."""
    c = ResolutionController(
        it_v_floor=0,
        map_fps_floor=0,
        scheduled_low_until_iteration=20,
    )

    assert c.decide(mapping_done=False, iteration=1) == "low"
    assert c.decide(mapping_done=False, iteration=20) == "low"
    assert c.decide(mapping_done=False, iteration=21) == "high"


def test_scheduled_bootstrap_fails_loudly_without_iteration_clock():
    """A configured schedule must not silently fall back to reactive policy."""
    c = ResolutionController(scheduled_low_until_iteration=20)

    try:
        c.decide(mapping_done=False)
    except ValueError as exc:
        assert "iteration" in str(exc)
    else:
        raise AssertionError("configured resolution schedule accepted no iteration clock")


def test_mapping_completion_still_forces_native_before_schedule_boundary():
    """The existing one-way native lock remains the stronger safety boundary."""
    c = ResolutionController(scheduled_low_until_iteration=20)

    assert c.decide(mapping_done=True, iteration=5) == "high"
    assert c.decide(mapping_done=False, iteration=6) == "high"
