"""Tests for MappingScheduler (Phase 7/8)."""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import MappingDecision, MappingScheduler, SchedulerStats
from eligsir.contracts.types import Pose, TrackingResult


def _make_tracking_result(confidence: float, valid: bool = True) -> TrackingResult:
    return TrackingResult(
        pose=Pose.identity(),
        confidence=confidence,
        valid=valid,
        method="test",
    )


class TestMappingScheduler:
    """Phase 7: MappingScheduler tests."""

    def test_historical_repair_extra_steps_are_consumed_once(self, monkeypatch):
        from eligsir.utils import partition_control

        monkeypatch.setenv("ELIGSIR_HISTORICAL_REPAIR_MODE", "targeted_plus_budget")
        monkeypatch.setenv("ELIGSIR_HISTORICAL_REPAIR_EXTRA_STEP_CAP", "2")
        partition_control.reset_historical_repair_state()
        partition_control._historical_repair_selection(
            (1,),
            bad_partitions=(0,),
            max_source_index=1,
            partition_count=1,
            max_promote=1,
        )
        sched = MappingScheduler(
            steps_per_frame=10,
            n_frames=1,
            iterations=10,
            enforce_full_coverage=False,
        )

        first = sched.decide(None, [], [])
        second = sched.decide(None, [], [])

        assert first.steps == 12
        assert first.historical_repair_extra_steps == 2
        assert second.steps == 10
        assert second.historical_repair_extra_steps == 0
        assert partition_control.latest_historical_repair_report()[
            "realized_cumulative_extra_optimizer_steps"
        ] == 2

    def test_full_coverage_enforcement(self):
        """Full-coverage reduces steps_per_frame when configured_spf > budget."""
        sched = MappingScheduler(
            steps_per_frame=500,
            n_frames=100,
            iterations=10000,
            enforce_full_coverage=True,
        )
        assert sched.stats.effective_spf <= 100, "spf must be ≤ iterations / n_frames"
        assert sched.stats.full_coverage is True

    def test_no_full_coverage(self):
        """Without full-coverage, spf is unchanged."""
        sched = MappingScheduler(
            steps_per_frame=200,
            n_frames=100,
            iterations=10000,
            enforce_full_coverage=False,
        )
        assert sched.stats.effective_spf == 200
        assert sched.stats.full_coverage is False

    def test_normal_mapping_high_confidence(self):
        """High confidence → normal mapping with all flags enabled."""
        sched = MappingScheduler(steps_per_frame=50, n_frames=100, iterations=5000)
        result = _make_tracking_result(confidence=0.95)
        decision = sched.decide(result, keyframe_indices=[0, 1, 2], replay_indices=[5])
        assert decision.do_mapping is True
        assert decision.allow_growth is True
        assert decision.allow_insertion is True
        assert decision.freeze_persistent is False
        assert decision.steps == 50

    def test_provisional_mapping_medium_confidence(self):
        """Medium confidence → provisional mapping, no growth, frozen persistent."""
        sched = MappingScheduler(
            steps_per_frame=100,
            n_frames=100,
            iterations=10000,
            tracking_min_conf_persistent=0.7,
            tracking_min_conf_provisional=0.4,
        )
        result = _make_tracking_result(confidence=0.55)
        decision = sched.decide(result, keyframe_indices=[0], replay_indices=[])
        assert decision.do_mapping is True
        assert decision.allow_growth is False
        assert decision.freeze_persistent is True
        assert decision.steps < 100

    def test_tracking_only_low_confidence(self):
        """Low confidence → no mapping at all."""
        sched = MappingScheduler(steps_per_frame=50, n_frames=100, iterations=5000)
        result = _make_tracking_result(confidence=0.2)
        decision = sched.decide(result, keyframe_indices=[0], replay_indices=[])
        assert decision.do_mapping is False
        assert decision.steps == 0

    def test_invalid_tracking_no_mapping(self):
        """Invalid tracking result → no mapping."""
        sched = MappingScheduler(steps_per_frame=50, n_frames=100, iterations=5000)
        result = _make_tracking_result(confidence=0.9, valid=False)
        decision = sched.decide(result, keyframe_indices=[0], replay_indices=[])
        assert decision.do_mapping is False

    def test_none_tracking_result_normal_mapping(self):
        """None tracking result (precomputed) → treat as confidence 1.0."""
        sched = MappingScheduler(steps_per_frame=50, n_frames=100, iterations=5000)
        decision = sched.decide(None, keyframe_indices=[0], replay_indices=[])
        assert decision.do_mapping is True
        assert decision.allow_growth is True

    def test_utils_mapping_scheduler_reexports_package_scheduler(self):
        from eligsir.utils.mapping_scheduler import MappingScheduler as CompatMappingScheduler

        assert CompatMappingScheduler is MappingScheduler
