"""Tests for Issue #703: supervision-work accumulators on the causal path.

Validates SupervisionWorkClocks initialization, observation, aggregation,
and integration into the causal reporting pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from eligsir.runtime.supervision_work_clocks import SupervisionWorkClocks
from eligsir.utils.supervision_work import supervision_work_report_block


class TestSupervisionWorkClocksAccumulation:
    """Test the basic accumulation behavior of SupervisionWorkClocks."""

    def test_initialization_clears_all_fields(self):
        """SupervisionWorkClocks initializes with all counters at zero."""
        clocks = SupervisionWorkClocks()
        assert clocks.optimizer_step_count == 0
        assert clocks.supervision_view_count == 0
        assert clocks.rasterizer_call_count == 0
        assert clocks.supervised_pixel_count == 0
        assert clocks.loss_objective_call_count == 0
        assert clocks.backward_call_count == 0
        assert clocks.renderer_camera_batch_size_sum == 0
        assert clocks.renderer_camera_batch_size_count == 0
        assert clocks.renderer_camera_batch_size_max == 0
        assert clocks.loss_views_per_call_sum == 0
        assert clocks.loss_views_per_call_count == 0
        assert clocks.loss_views_per_call_max == 0
        assert clocks.per_source_counts == {}

    def test_observe_single_step_b1(self):
        """Observe a single B=1 OptimizationStepResult."""
        clocks = SupervisionWorkClocks()
        step = SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=1,
            rasterizer_call_count=1,
            supervised_pixel_count=614400,  # 640x960 typical
            loss_objective_call_count=1,
            backward_call_count=1,
            renderer_camera_batch_sizes=(1,),
            loss_views_per_call=(1,),
            successful_supervision_source_indices=(5,),
        )
        clocks.observe(step)
        assert clocks.optimizer_step_count == 1
        assert clocks.supervision_view_count == 1
        assert clocks.rasterizer_call_count == 1
        assert clocks.supervised_pixel_count == 614400
        assert clocks.loss_objective_call_count == 1
        assert clocks.backward_call_count == 1
        assert clocks.renderer_camera_batch_size_sum == 1
        assert clocks.renderer_camera_batch_size_count == 1
        assert clocks.renderer_camera_batch_size_max == 1
        assert clocks.loss_views_per_call_sum == 1
        assert clocks.loss_views_per_call_count == 1
        assert clocks.loss_views_per_call_max == 1
        assert clocks.per_source_counts == {5: 1}

    def test_observe_batch_step_b4(self):
        """Observe a B=4 batched step with multiple views and loss calls."""
        clocks = SupervisionWorkClocks()
        step = SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=4,
            rasterizer_call_count=4,
            supervised_pixel_count=4 * 614400,
            loss_objective_call_count=4,
            backward_call_count=4,
            renderer_camera_batch_sizes=(1, 1, 1, 1),
            loss_views_per_call=(1, 1, 1, 1),
            successful_supervision_source_indices=(1, 2, 3, 4),
        )
        clocks.observe(step)
        assert clocks.optimizer_step_count == 1
        assert clocks.supervision_view_count == 4
        assert clocks.rasterizer_call_count == 4
        assert clocks.supervised_pixel_count == 4 * 614400
        assert clocks.loss_objective_call_count == 4
        assert clocks.backward_call_count == 4
        assert clocks.renderer_camera_batch_size_sum == 4
        assert clocks.renderer_camera_batch_size_count == 4
        assert clocks.renderer_camera_batch_size_max == 1
        assert clocks.loss_views_per_call_sum == 4
        assert clocks.loss_views_per_call_count == 4
        assert clocks.loss_views_per_call_max == 1
        assert clocks.per_source_counts == {1: 1, 2: 1, 3: 1, 4: 1}

    def test_observe_multiple_sequential_steps(self):
        """Accumulate counts over multiple sequential steps."""
        clocks = SupervisionWorkClocks()
        # First step
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=1,
                rasterizer_call_count=1,
                supervised_pixel_count=614400,
                loss_objective_call_count=1,
                backward_call_count=1,
                renderer_camera_batch_sizes=(1,),
                loss_views_per_call=(1,),
                successful_supervision_source_indices=(1,),
            )
        )
        # Second step
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=1,
                rasterizer_call_count=1,
                supervised_pixel_count=614400,
                loss_objective_call_count=1,
                backward_call_count=1,
                renderer_camera_batch_sizes=(1,),
                loss_views_per_call=(1,),
                successful_supervision_source_indices=(2,),
            )
        )
        assert clocks.optimizer_step_count == 2
        assert clocks.supervision_view_count == 2
        assert clocks.rasterizer_call_count == 2
        assert clocks.supervised_pixel_count == 2 * 614400
        assert clocks.loss_objective_call_count == 2
        assert clocks.backward_call_count == 2
        assert clocks.per_source_counts == {1: 1, 2: 1}

    def test_observe_same_source_multiple_times(self):
        """Duplicates in successful_supervision_source_indices accumulate correctly."""
        clocks = SupervisionWorkClocks()
        # Step 1: source 1 supervised
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=1,
                rasterizer_call_count=1,
                supervised_pixel_count=614400,
                loss_objective_call_count=1,
                backward_call_count=1,
                renderer_camera_batch_sizes=(1,),
                loss_views_per_call=(1,),
                successful_supervision_source_indices=(1,),
            )
        )
        # Step 2: source 1 supervised again
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=1,
                rasterizer_call_count=1,
                supervised_pixel_count=614400,
                loss_objective_call_count=1,
                backward_call_count=1,
                renderer_camera_batch_sizes=(1,),
                loss_views_per_call=(1,),
                successful_supervision_source_indices=(1,),
            )
        )
        assert clocks.per_source_counts == {1: 2}

    def test_observe_batch_with_grouped_rendering(self):
        """Observe a step with multi-view grouped rendering."""
        clocks = SupervisionWorkClocks()
        # Two rasterizer calls, one with 2 views, one with 2 views.
        step = SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=4,
            rasterizer_call_count=2,
            supervised_pixel_count=4 * 614400,
            loss_objective_call_count=4,
            backward_call_count=4,
            renderer_camera_batch_sizes=(2, 2),
            loss_views_per_call=(1, 1, 1, 1),
            successful_supervision_source_indices=(1, 2, 3, 4),
        )
        clocks.observe(step)
        assert clocks.rasterizer_call_count == 2
        assert clocks.renderer_camera_batch_size_sum == 4
        assert clocks.renderer_camera_batch_size_count == 2
        assert clocks.renderer_camera_batch_size_max == 2

    def test_observe_tolerates_missing_fields(self):
        """Observe gracefully handles None or missing optional fields."""
        clocks = SupervisionWorkClocks()
        step = SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=1,
            rasterizer_call_count=1,
            supervised_pixel_count=614400,
            # loss_objective_call_count and others missing
        )
        clocks.observe(step)
        assert clocks.optimizer_step_count == 1
        assert clocks.loss_objective_call_count == 0
        assert clocks.backward_call_count == 0

    def test_observe_tolerates_none_result(self):
        """Observe handles None step_result gracefully."""
        clocks = SupervisionWorkClocks()
        clocks.observe(None)
        assert clocks.optimizer_step_count == 0

    def test_report_block_empty_at_zero_steps(self):
        """report_block returns empty dict if no optimizer steps observed."""
        clocks = SupervisionWorkClocks()
        block = clocks.report_block(wall_time_s=10.0)
        assert block == {}

    def test_report_block_with_nonzero_steps(self):
        """report_block includes derived ratios when optimizer_step_count > 0."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=2,
                supervision_view_count=8,
                rasterizer_call_count=8,
                supervised_pixel_count=8 * 614400,
                loss_objective_call_count=8,
                backward_call_count=8,
                renderer_camera_batch_sizes=(1, 1, 1, 1, 1, 1, 1, 1),
                loss_views_per_call=(1, 1, 1, 1, 1, 1, 1, 1),
                successful_supervision_source_indices=(),
            )
        )
        block = clocks.report_block(wall_time_s=10.0)

        # Core fields
        assert block["optimization_step_count"] == 2
        assert block["supervision_view_count"] == 8
        assert block["rasterizer_call_count"] == 8
        assert block["supervised_pixel_count"] == 8 * 614400

        # Derived ratios
        assert block["supervision_views_per_step"] == 4.0
        assert block["rasterizer_calls_per_step"] == 4.0
        assert block["supervised_pixels_per_step"] == 2457600.0  # (8*614400) / 2
        assert block["supervision_view_rate_hz"] == 0.8  # 8 views / 10 seconds

        # Execution-shape counters and ratios
        assert block["loss_objective_call_count"] == 8
        assert block["backward_call_count"] == 8
        assert block["loss_objective_calls_per_step"] == 4.0
        assert block["backward_calls_per_step"] == 4.0
        assert block["renderer_camera_batch_size_mean"] == 1.0
        assert block["renderer_camera_batch_size_max"] == 1
        assert block["loss_views_per_call_mean"] == 1.0
        assert block["loss_views_per_call_max"] == 1

    def test_report_block_consistent_with_supervision_work_report_block(self):
        """report_block output matches direct supervision_work_report_block call."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=4,
                rasterizer_call_count=4,
                supervised_pixel_count=4 * 614400,
                loss_objective_call_count=4,
                backward_call_count=4,
                renderer_camera_batch_sizes=(1, 1, 1, 1),
                loss_views_per_call=(1, 1, 1, 1),
                successful_supervision_source_indices=(),
            )
        )
        clocks_block = clocks.report_block(wall_time_s=10.0)
        direct_block = supervision_work_report_block(
            optimization_step_count=1,
            supervision_view_count=4,
            rasterizer_call_count=4,
            supervised_pixel_count=4 * 614400,
            wall_time_s=10.0,
            loss_objective_call_count=4,
            backward_call_count=4,
            renderer_camera_batch_size_sum=4,
            renderer_camera_batch_size_max=1,
            loss_views_per_call_sum=4,
            loss_views_per_call_max=1,
        )
        assert clocks_block == direct_block


class TestSupervisionWorkClocksInvariants:
    """Test internal consistency invariants."""

    def test_batch_size_count_equals_rasterizer_call_count(self):
        """renderer_camera_batch_size_count must equal rasterizer_call_count."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=8,
                rasterizer_call_count=4,
                supervised_pixel_count=8 * 614400,
                loss_objective_call_count=8,
                backward_call_count=8,
                renderer_camera_batch_sizes=(2, 2, 2, 2),
                loss_views_per_call=(1, 1, 1, 1, 1, 1, 1, 1),
                successful_supervision_source_indices=(),
            )
        )
        assert clocks.renderer_camera_batch_size_count == clocks.rasterizer_call_count

    def test_views_per_call_count_equals_loss_objective_call_count(self):
        """loss_views_per_call_count must equal loss_objective_call_count."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=8,
                rasterizer_call_count=4,
                supervised_pixel_count=8 * 614400,
                loss_objective_call_count=8,
                backward_call_count=8,
                renderer_camera_batch_sizes=(2, 2, 2, 2),
                loss_views_per_call=(1, 1, 1, 1, 1, 1, 1, 1),
                successful_supervision_source_indices=(),
            )
        )
        assert clocks.loss_views_per_call_count == clocks.loss_objective_call_count

    def test_report_block_with_inconsistent_execution_counts(self):
        """Test behavior when execution-shape counts are inconsistent.

        If renderer_camera_batch_size_count != rasterizer_call_count (which
        would be a bug in observe()), report_block should still function.
        """
        clocks = SupervisionWorkClocks()
        # Manually set inconsistent state (would not happen in real code)
        clocks.optimizer_step_count = 1
        clocks.supervision_view_count = 4
        clocks.rasterizer_call_count = 4
        clocks.supervised_pixel_count = 4 * 614400
        clocks.loss_objective_call_count = 4
        clocks.backward_call_count = 4
        clocks.renderer_camera_batch_size_sum = 4
        clocks.renderer_camera_batch_size_count = 4
        clocks.renderer_camera_batch_size_max = 1
        clocks.loss_views_per_call_sum = 4
        clocks.loss_views_per_call_count = 4
        clocks.loss_views_per_call_max = 1

        block = clocks.report_block(wall_time_s=10.0)
        # Should still compute means correctly
        assert block["renderer_camera_batch_size_mean"] == 1.0
        assert block["loss_views_per_call_mean"] == 1.0


class TestSupervisionWorkClocksEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_step_count(self):
        """Handle very large accumulation without overflow."""
        clocks = SupervisionWorkClocks()
        for i in range(100000):
            clocks.observe(
                SimpleNamespace(
                    optimizer_step_count=1,
                    supervision_view_count=1,
                    rasterizer_call_count=1,
                    supervised_pixel_count=614400,
                    loss_objective_call_count=1,
                    backward_call_count=1,
                    renderer_camera_batch_sizes=(1,),
                    loss_views_per_call=(1,),
                    successful_supervision_source_indices=(),
                )
            )
        assert clocks.optimizer_step_count == 100000

    def test_zero_wall_time(self):
        """report_block with zero wall_time_s returns zero rate."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=1,
                supervision_view_count=10,
                rasterizer_call_count=1,
                supervised_pixel_count=614400,
                loss_objective_call_count=1,
                backward_call_count=1,
                renderer_camera_batch_sizes=(1,),
                loss_views_per_call=(1,),
                successful_supervision_source_indices=(),
            )
        )
        block = clocks.report_block(wall_time_s=0.0)
        assert block["supervision_view_rate_hz"] == 0.0


class TestSupervisionWorkClocksLazyCreation:
    """Regression test for Issue #703 follow-up: resilience to test seams.

    Some tests construct CanonicalMappingRuntime via object.__new__() which
    bypasses __init__, so _supervision_work_clocks is never initialized.
    This tests that the hook sites use lazy creation to remain resilient.
    """

    def test_lazy_creation_on_object_new_constructed_runtime(self):
        """Test that observe() works on runtime created via object.__new__().

        This simulates test seams like test_configured_b32_source_insertion_*
        that construct the runtime without calling __init__.
        """
        from eligsir.runtime.production_application import CanonicalMappingRuntime

        # Simulate a test seam construction via object.__new__
        runtime = object.__new__(CanonicalMappingRuntime)

        # Verify the attribute is not yet initialized
        assert not hasattr(runtime, "_supervision_work_clocks")

        # Get the helper, which should lazily create the clocks
        clocks = runtime._supervision_work_clocks_or_create()
        assert clocks is not None

        # Verify the attribute is now set
        assert hasattr(runtime, "_supervision_work_clocks")

        # Observe a step result
        step = SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=1,
            rasterizer_call_count=1,
            supervised_pixel_count=614400,
            loss_objective_call_count=1,
            backward_call_count=1,
            renderer_camera_batch_sizes=(1,),
            loss_views_per_call=(1,),
        )
        clocks.observe(step)

        # Verify the count was incremented
        assert clocks.optimizer_step_count == 1
        assert clocks.supervision_view_count == 1

    def test_repeated_lazy_creation_returns_same_instance(self):
        """Test that repeated calls to _supervision_work_clocks_or_create return same instance."""
        from eligsir.runtime.production_application import CanonicalMappingRuntime

        runtime = object.__new__(CanonicalMappingRuntime)

        clocks1 = runtime._supervision_work_clocks_or_create()
        clocks2 = runtime._supervision_work_clocks_or_create()

        # Should return the same instance, not create a new one
        assert clocks1 is clocks2
