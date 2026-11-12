"""Tests for OnlineController (Milestone 2)."""

from __future__ import annotations

import time

import pytest

from eligsir.contracts.types import Pose, TrackingResult
from eligsir.mapping.scheduler import (
    ControllerMode,
    ControllerStats,
    MappingScheduler,
    OnlineConfig,
    OnlineController,
)


def _tracking(confidence: float = 0.9, valid: bool = True) -> TrackingResult:
    return TrackingResult(
        pose=Pose.identity(),
        confidence=confidence,
        valid=valid,
        method="test",
    )


@pytest.fixture
def scheduler() -> MappingScheduler:
    return MappingScheduler(
        steps_per_frame=10,
        n_frames=100,
        iterations=10000,
        enforce_full_coverage=False,
    )


@pytest.fixture
def controller(scheduler) -> OnlineController:
    return OnlineController(
        scheduler=scheduler,
        config=OnlineConfig(
            map_age_target_s=0.5,
            map_age_catch_up_enter_s=0.8,
            map_age_catch_up_exit_s=0.3,
            map_age_track_only_enter_s=1.5,
            map_age_track_only_exit_s=0.5,
            min_steps=1,
            max_steps=20,
            iter_dt_ema_beta=0.5,
            adaptive_steps=True,
        ),
        target_input_fps=30.0,
        budget_per_iter_s=0.05,
        insertion_cap=1500,
        sampling_mode="legacy",
    )


class TestModeTransitions:
    """Map-age hysteresis state machine."""

    def test_initial_mode_balanced(self, controller: OnlineController) -> None:
        assert controller.mode == ControllerMode.BALANCED

    def test_utils_online_controller_reexports_package_controller(self) -> None:
        from eligsir.utils.online_controller import OnlineController as CompatOnlineController

        assert CompatOnlineController is OnlineController

    def test_catch_up_when_map_age_above_threshold(
        self, controller: OnlineController
    ) -> None:
        # Bootstrap two frames: first establishes sensor_clock, second moves
        # map_clock forward so map_age is large.
        controller.record_ingest(timestamp=0.0)
        controller.record_mapping(timestamp=0.0)
        controller.record_ingest(timestamp=2.0)
        # map_clock still 0.0 → map_age = 2.0 → CATCH_UP (0.8 s threshold)
        controller._update_mode(backlog=0)
        assert controller.mode in (ControllerMode.CATCH_UP, ControllerMode.TRACK_ONLY)


    def test_track_only_hard_ceiling(self, controller: OnlineController) -> None:
        controller.record_ingest(timestamp=0.0)
        controller.record_mapping(timestamp=0.0)
        controller.record_ingest(timestamp=2.0)  # map_age = 2.0 → TRACK_ONLY
        controller._update_mode(backlog=0)
        assert controller.mode == ControllerMode.TRACK_ONLY

    def test_hysteresis_exits_catch_up(self, controller: OnlineController) -> None:
        # Enter CATCH_UP at age 1.0, then bring map_clock up.
        controller.record_ingest(timestamp=0.0)
        controller.record_mapping(timestamp=0.0)
        controller.record_ingest(timestamp=1.0)  # map_age = 1.0
        controller._update_mode(backlog=0)
        assert controller.mode == ControllerMode.CATCH_UP
        # Map the just-arrived frame — map_age collapses to 0.
        controller.record_mapping(timestamp=1.0)
        controller._update_mode(backlog=0)
        assert controller.mode == ControllerMode.BALANCED

    def test_quality_idle_requires_stable_input(
        self, controller: OnlineController
    ) -> None:
        # Stable input + map_age below catch_up_exit → QUALITY_IDLE.
        # Pre-seed input window at 30 Hz spanning the full stability window.
        now = time.perf_counter()
        # Spread 30 evenly-spaced arrivals across the last 1.0 s.
        for i in range(30):
            controller._input_window.append(now - 1.0 + i * (1.0 / 29.0))
        # Force the controller to see this stable state without adding a new
        # arrival at "now" (which would skew the rolling window).
        controller._update_mode(backlog=0)
        assert controller.mode == ControllerMode.QUALITY_IDLE


class TestEMA:
    """Iteration-cost EMA."""

    def test_ema_initialised_to_budget(
        self, controller: OnlineController
    ) -> None:
        assert controller.iter_dt_ema_s == pytest.approx(0.05, rel=1e-3)

    def test_ema_smooths_iter_dt(self, controller: OnlineController) -> None:
        # First sample: 0.10s (over budget) → EMA should bias upward.
        controller.record_iter_dt(0.10)
        assert 0.05 < controller.iter_dt_ema_s < 0.10
        # 50 samples of budget: EMA should converge to 0.05.
        for _ in range(50):
            controller.record_iter_dt(0.05)
        assert controller.iter_dt_ema_s == pytest.approx(0.05, rel=0.1)

    def test_ema_ignores_non_positive(self, controller: OnlineController) -> None:
        before = controller.iter_dt_ema_s
        controller.record_iter_dt(0.0)
        controller.record_iter_dt(-1.0)
        controller.record_iter_dt(None)
        assert controller.iter_dt_ema_s == before


class TestTick:
    """End-to-end tick behaviour."""

    def test_tick_does_not_feed_iter_dt_ema(
        self, controller: OnlineController
    ) -> None:
        """Regression test: ``tick()`` must never feed the iteration-cost
        EMA. Callers should call :meth:`record_iter_dt` exactly once per
        training iteration; the previous implementation accidentally
        double-counted by accepting ``iter_dt`` inside ``tick()``.
        """
        controller.record_iter_dt(0.10)  # caller's explicit call
        ema_after_one_call = controller.iter_dt_ema_s
        controller.tick(
            tracking_result=_tracking(0.9),
            timestamp=0.0,
            backlog=0,
        )
        assert controller.iter_dt_ema_s == pytest.approx(ema_after_one_call, rel=1e-9)

    def test_balanced_tick(self, controller: OnlineController) -> None:
        decision, _ = controller.tick(
            tracking_result=_tracking(0.9),
            timestamp=0.0,
            backlog=0,
        )
        assert decision.mode == ControllerMode.BALANCED
        assert decision.do_mapping is True
        assert decision.train_steps > 0

    def test_track_only_tick_no_steps(self, controller: OnlineController) -> None:
        controller.record_ingest(timestamp=0.0)
        controller.record_mapping(timestamp=0.0)
        # map_age 2.0 > 1.5 → TRACK_ONLY.
        controller.record_ingest(timestamp=2.0)
        decision, _ = controller.tick(
            tracking_result=_tracking(0.9),
            timestamp=2.0,
            backlog=0,
        )
        assert decision.mode == ControllerMode.TRACK_ONLY
        assert decision.do_mapping is False
        assert decision.train_steps == 0

    def test_low_confidence_blocks_mapping_even_in_balanced(
        self, controller: OnlineController
    ) -> None:
        decision, _ = controller.tick(
            tracking_result=_tracking(0.2),
            timestamp=0.0,
            backlog=0,
        )
        assert decision.do_mapping is False

    def test_adaptive_steps_reduce_on_overbudget(
        self, scheduler: MappingScheduler
    ) -> None:
        # EMA is way over budget → adaptive_steps scales base down.
        controller = OnlineController(
            scheduler=scheduler,
            config=OnlineConfig(
                min_steps=1,
                max_steps=20,
                iter_dt_ema_beta=0.0,  # no smoothing
                adaptive_steps=True,
            ),
            target_input_fps=30.0,
            budget_per_iter_s=0.01,  # 10 ms budget
        )
        # Force EMA to 100 ms (10× over budget).
        controller._iter_dt_ema = 0.1
        decision, _ = controller.tick(
            tracking_result=_tracking(0.9),
            timestamp=0.0,
            backlog=0,
        )
        # ratio = 0.01 / 0.1 = 0.1 → steps = round(base * 0.1) ≤ 2
        assert decision.train_steps <= 2

    def test_none_tracking_precomputed_path(
        self, controller: OnlineController
    ) -> None:
        decision, _ = controller.tick(
            tracking_result=None,
            timestamp=0.0,
            backlog=0,
        )
        assert decision.mode == ControllerMode.BALANCED
        assert decision.do_mapping is True


class TestDropCounters:
    def test_drop_counter_keyframe(self, controller: OnlineController) -> None:
        controller.record_drop(timestamp=0.0, is_keyframe=True)
        controller.record_drop(timestamp=0.0, is_keyframe=False)
        controller.record_drop(timestamp=0.0, is_keyframe=False)
        s = controller.stats
        assert s.dropped_keyframes == 1
        assert s.dropped_nonkeyframes == 2


class TestStats:
    def test_stats_keys(self, controller: OnlineController) -> None:
        s = controller.stats
        assert isinstance(s, ControllerStats)
        d = s.to_tensorboard_dict()
        for required in [
            "controller/map_age_s",
            "controller/iter_dt_ema_s",
            "controller/train_steps",
            "controller/pose_fps",
            "controller/measured_input_fps",
            "controller/confidence",
            "controller/pause",
            "controller/burst",
        ]:
            assert required in d, required

    def test_record_pose_arrival(self, controller: OnlineController) -> None:
        controller.record_pose_arrival()
        assert len(controller._pose_window) == 1
        controller.record_pose_arrival()
        assert len(controller._pose_window) == 2


    def test_windows_pruned_to_max_age(self, controller: OnlineController) -> None:
        """Long-running sessions must not leak memory via growing windows."""
        # Inject timestamps 60 s in the past — should be pruned on next
        # record_ingest/record_pose_arrival.
        old = time.perf_counter() - 60.0
        controller._input_window.append(old)
        controller._pose_window.append(old)
        # A fresh record must trigger a prune.
        controller.record_frame_arrival()
        controller.record_pose_arrival()
        # The 60 s old entries are gone.
        assert all(t >= time.perf_counter() - 6.0 for t in controller._input_window)
        assert all(t >= time.perf_counter() - 6.0 for t in controller._pose_window)


class TestConfigTunables:
    def test_disabled_controller_is_a_passthrough(
        self, scheduler: MappingScheduler
    ) -> None:
        """When adaptive_steps=False and config defaults, behaviour is
        near-identical to MappingScheduler.decide() — no hysteresis,
        no shaping."""
        ctrl = OnlineController(
            scheduler=scheduler,
            config=OnlineConfig(adaptive_steps=False),
            target_input_fps=30.0,
            budget_per_iter_s=0.05,
        )
        decision, base = ctrl.tick(
            tracking_result=_tracking(0.9),
            timestamp=0.0,
            backlog=0,
        )
        assert decision.mode == ControllerMode.BALANCED
        assert decision.do_mapping == base.do_mapping
        assert decision.allow_growth == base.allow_growth
        assert decision.allow_insertion == base.allow_insertion
        assert decision.train_steps == base.steps


class TestPauseIdleThreshold:
    """``OnlineConfig.pause_idle_threshold_s`` (Milestone 3 CLI flag) must
    actually drive the ``pause`` flag instead of being a no-op."""

    def test_threshold_respected_when_positive(
        self, scheduler: MappingScheduler
    ) -> None:
        ctrl = OnlineController(
            scheduler=scheduler,
            config=OnlineConfig(pause_idle_threshold_s=1.0),
            target_input_fps=30.0,
        )
        # Seed an input arrival 2 s in the past so the gap is 2 s.
        ctrl._input_window.append(time.perf_counter() - 2.0)
        ctrl._update_mode(backlog=0)
        # 2 s gap > 1.0 s threshold → pause=True.
        assert ctrl.stats.pause is True

    def test_threshold_legacy_fallback_when_zero(
        self, scheduler: MappingScheduler
    ) -> None:
        """``pause_idle_threshold_s=0`` keeps the twice-target-period heuristic."""
        ctrl = OnlineController(
            scheduler=scheduler,
            config=OnlineConfig(pause_idle_threshold_s=0.0),
            target_input_fps=30.0,
        )
        # 2 s gap > 1/15 s (twice target period) → pause=True (legacy).
        ctrl._input_window.append(time.perf_counter() - 2.0)
        ctrl._update_mode(backlog=0)
        assert ctrl.stats.pause is True


class TestClampedBacklog:
    def test_measured_input_fps_clamped_to_2x_target(
        self, controller: OnlineController
    ) -> None:
        """Bursts should not blow up the backlog signal.

        The published ``backlog`` and ``measured_input_fps`` must both
        be clamped to 2× target. The raw value is exposed as
        ``ControllerStats.extra['measured_input_fps_raw']``.
        """
        # 200 events in the last 0.1 s (~2000 Hz raw). 30 Hz target →
        # 2× target = 60 Hz cap. map_age=1.0 s so the backlog is positive.
        now = time.perf_counter()
        controller._input_window.extend(now - 0.1 + i * 0.0005 for i in range(200))
        controller._sensor_clock = 1.0
        controller._map_clock = 0.0
        controller._update_map_age()
        s = controller.stats
        assert s.measured_input_fps <= controller._target_input_fps * 2.0 + 1e-6
        assert s.extra["measured_input_fps_raw"] > controller._target_input_fps * 2.0
        assert s.backlog == int(max(0.0, s.measured_input_fps * s.map_age_s))
