"""Tests for BurstBudget (Milestone 3)."""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import (
    BurstBudget,
    BurstConfig,
    ControllerMode,
    MappingScheduler,
    OnlineController,
    OnlineConfig,
)


class _FakeFrame:
    def __init__(self, index: int, timestamp: float = 0.0, c2w=None) -> None:
        import numpy as np
        if c2w is None:
            c2w = np.eye(4, dtype=np.float32)
        self.index = index
        self.timestamp = timestamp
        self.c2w = c2w


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
        config=OnlineConfig(),
        target_input_fps=30.0,
        budget_per_iter_s=0.05,
    )


def _force_catch_up(controller: OnlineController) -> None:
    """Push the controller into CATCH_UP mode."""
    controller.record_ingest(timestamp=0.0)
    controller.record_mapping(timestamp=0.0)
    controller.record_ingest(timestamp=1.0)  # map_age = 1.0 > 0.8
    controller._update_mode(backlog=0)
    assert controller.mode == ControllerMode.CATCH_UP


class TestPerIterCap:
    def test_burst_budget_is_exported_by_scheduler(self) -> None:
        from eligsir.mapping.scheduler import BurstBudget as ExportedBurstBudget

        assert ExportedBurstBudget is BurstBudget

    def test_caps_at_max_per_iter(self, controller: OnlineController) -> None:
        budget = BurstBudget(controller, BurstConfig(max_per_iter=2))
        budget.begin_iter(iteration=1)
        f1 = _FakeFrame(0); f2 = _FakeFrame(1); f3 = _FakeFrame(2)
        assert budget.acquire(f1, is_keyframe=True)[0] is True
        assert budget.acquire(f2, is_keyframe=True)[0] is True
        ingest, _ = budget.acquire(f3, is_keyframe=True)
        assert ingest is False
        assert controller._dropped_keyframes == 1

    def test_counter_resets_per_iter(self, controller: OnlineController) -> None:
        budget = BurstBudget(controller, BurstConfig(max_per_iter=1))
        budget.begin_iter(iteration=1)
        budget.acquire(_FakeFrame(0), is_keyframe=True)
        budget.acquire(_FakeFrame(1), is_keyframe=True)  # dropped (cap)
        budget.begin_iter(iteration=2)
        # Next iter: cap resets.
        assert budget.acquire(_FakeFrame(2), is_keyframe=True)[0] is True


class TestDropPolicy:
    def test_drop_nonkf_in_catch_up(self, controller: OnlineController) -> None:
        _force_catch_up(controller)
        budget = BurstBudget(
            controller,
            BurstConfig(max_per_iter=10, drop_nonkf_in_catch_up=True, drop_nonkf_backlog=100),
        )
        budget.begin_iter(iteration=1)
        ingest, _ = budget.acquire(_FakeFrame(0), is_keyframe=False)
        assert ingest is False
        assert controller._dropped_nonkeyframes == 1
        # Keyframes are protected.
        ingest, _ = budget.acquire(_FakeFrame(1), is_keyframe=True)
        assert ingest is True

    def test_no_drop_in_catch_up_when_disabled(
        self, controller: OnlineController
    ) -> None:
        _force_catch_up(controller)
        budget = BurstBudget(
            controller,
            BurstConfig(max_per_iter=10, drop_nonkf_in_catch_up=False, drop_nonkf_backlog=100),
        )
        budget.begin_iter(iteration=1)
        ingest, _ = budget.acquire(_FakeFrame(0), is_keyframe=False)
        assert ingest is True

    def test_drop_when_backlog_high(self, controller: OnlineController) -> None:
        budget = BurstBudget(
            controller,
            BurstConfig(max_per_iter=10, drop_nonkf_in_catch_up=False, drop_nonkf_backlog=2),
        )
        # Establish a non-zero map_age so the backlog calculation is non-zero.
        controller.record_ingest(timestamp=0.0)
        controller.record_mapping(timestamp=0.0)
        controller.record_ingest(timestamp=1.0)  # map_age = 1.0
        # Simulate burst: 60 events spread over the last 0.5 s.
        import time
        now = time.perf_counter()
        for i in range(60):
            controller._input_window.append(now - 0.5 + i * (0.5 / 59.0))
        # After this _update_mode runs, stats.backlog = fps * map_age (capped).
        # 60 fps * 1.0 s = 60 → drop triggered.
        budget.begin_iter(iteration=1)
        ingest, _ = budget.acquire(_FakeFrame(0), is_keyframe=False)
        assert ingest is False
        # Keyframes still go through.
        ingest, _ = budget.acquire(_FakeFrame(1), is_keyframe=True)
        assert ingest is True


class TestPassthrough:
    def test_admit_keyframes_unconditionally(
        self, controller: OnlineController
    ) -> None:
        budget = BurstBudget(controller, BurstConfig(max_per_iter=5))
        budget.begin_iter(iteration=1)
        for i in range(5):
            ingest, _ = budget.acquire(_FakeFrame(i), is_keyframe=True)
            assert ingest is True
        # 6th is dropped.
        ingest, _ = budget.acquire(_FakeFrame(6), is_keyframe=True)
        assert ingest is False
