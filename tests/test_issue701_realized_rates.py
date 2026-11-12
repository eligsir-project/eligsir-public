"""Tests for Issue #701: realized-rate dashboard and execution summary.

Validates that the metric names are registered, rates are computed correctly,
and the execution-summary block has the expected format.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from eligsir.utils.metric_registry import CANONICAL_METRICS, get_canonical_metric
from eligsir.utils.supervision_work import (
    execution_summary_lines,
    optimization_step_rate_hz,
    rasterizer_calls_per_optimization_step,
    supervision_view_rate_hz,
    supervision_views_per_optimization_step,
)
from eligsir.runtime.supervision_work_clocks import SupervisionWorkClocks


class TestIssue701MetricNames:
    """Test (a): scalar names are registered in the metric registry."""

    def test_optimization_step_rate_hz_is_registered(self):
        """optimization_step_rate_hz must be a canonical metric."""
        metric = get_canonical_metric("optimization_step_rate_hz")
        assert metric is not None
        assert metric.name == "optimization_step_rate_hz"
        assert metric.unit == "Hz"
        assert metric.producer != "PENDING"  # Must have a producer
        assert metric.producer_kind.value == "computes"

    def test_supervision_view_rate_hz_is_registered(self):
        """supervision_view_rate_hz must be a canonical metric."""
        metric = get_canonical_metric("supervision_view_rate_hz")
        assert metric is not None
        assert metric.name == "supervision_view_rate_hz"
        assert metric.unit == "Hz"

    def test_realized_supervision_views_per_step_is_registered(self):
        """realized_supervision_views_per_step must be a canonical metric."""
        metric = get_canonical_metric("realized_supervision_views_per_step")
        assert metric is not None
        assert metric.name == "realized_supervision_views_per_step"
        assert metric.unit == "dimensionless"
        assert "supervision_views_per_step" in metric.legacy_aliases

    def test_rasterizer_calls_per_step_is_registered(self):
        """rasterizer_calls_per_step must be a canonical metric."""
        metric = get_canonical_metric("rasterizer_calls_per_step")
        assert metric is not None
        assert metric.name == "rasterizer_calls_per_step"
        assert metric.unit == "dimensionless"


class TestRateComputations:
    """Test (b): rates and per-step ratios are computed correctly."""

    def test_optimization_step_rate_hz_basic(self):
        """optimization_step_rate_hz = steps / wall_time_s."""
        # 10 steps over 5 seconds = 2 Hz
        assert optimization_step_rate_hz(10, 5.0) == 2.0

    def test_optimization_step_rate_hz_zero_time(self):
        """Zero wall time returns 0."""
        assert optimization_step_rate_hz(10, 0.0) == 0.0

    def test_supervision_view_rate_hz_basic(self):
        """supervision_view_rate_hz = views / wall_time_s."""
        # 40 views over 10 seconds = 4 Hz
        assert supervision_view_rate_hz(40, 10.0) == 4.0

    def test_rasterizer_calls_per_optimization_step_basic(self):
        """rasterizer_calls_per_step = calls / steps."""
        # 16 calls over 4 steps = 4.0 calls/step
        assert rasterizer_calls_per_optimization_step(4, 16) == 4.0

    def test_rasterizer_calls_per_optimization_step_zero_steps(self):
        """Zero steps returns 0.0."""
        assert rasterizer_calls_per_optimization_step(0, 16) == 0.0

    def test_realized_supervision_views_per_step_basic(self):
        """realized_supervision_views_per_step = views / steps."""
        # 8 views over 2 steps = 4.0 views/step
        from eligsir.utils.supervision_work import supervision_views_per_optimization_step
        assert supervision_views_per_optimization_step(2, 8) == 4.0


class TestSupervisionWorkReportBlockIncludesRates:
    """Test that report_block includes the required rate metrics."""

    def test_report_block_includes_optimization_step_rate_hz(self):
        """report_block with wall_time_s includes optimization_step_rate_hz."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=10,
                supervision_view_count=40,
                rasterizer_call_count=10,
                supervised_pixel_count=10 * 614400,
                loss_objective_call_count=10,
                backward_call_count=10,
                renderer_camera_batch_sizes=(4,) * 10,
                loss_views_per_call=(4,) * 10,
            )
        )
        block = clocks.report_block(wall_time_s=5.0)
        assert "optimization_step_rate_hz" in block
        assert block["optimization_step_rate_hz"] == 2.0  # 10 steps / 5 seconds

    def test_report_block_includes_supervision_view_rate_hz(self):
        """report_block with wall_time_s includes supervision_view_rate_hz."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=10,
                supervision_view_count=40,
                rasterizer_call_count=10,
                supervised_pixel_count=10 * 614400,
                loss_objective_call_count=10,
                backward_call_count=10,
                renderer_camera_batch_sizes=(4,) * 10,
                loss_views_per_call=(4,) * 10,
            )
        )
        block = clocks.report_block(wall_time_s=5.0)
        assert "supervision_view_rate_hz" in block
        assert block["supervision_view_rate_hz"] == 8.0  # 40 views / 5 seconds

    def test_report_block_includes_realized_supervision_views_per_step(self):
        """report_block includes realized_supervision_views_per_step."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=10,
                supervision_view_count=40,
                rasterizer_call_count=10,
                supervised_pixel_count=10 * 614400,
                loss_objective_call_count=10,
                backward_call_count=10,
                renderer_camera_batch_sizes=(4,) * 10,
                loss_views_per_call=(4,) * 10,
            )
        )
        block = clocks.report_block(wall_time_s=5.0)
        assert "realized_supervision_views_per_step" in block
        assert block["realized_supervision_views_per_step"] == 4.0  # 40 views / 10 steps

    def test_report_block_includes_rasterizer_calls_per_step(self):
        """report_block includes rasterizer_calls_per_step."""
        clocks = SupervisionWorkClocks()
        clocks.observe(
            SimpleNamespace(
                optimizer_step_count=4,
                supervision_view_count=16,
                rasterizer_call_count=16,
                supervised_pixel_count=16 * 614400,
                loss_objective_call_count=16,
                backward_call_count=16,
                renderer_camera_batch_sizes=(1,) * 16,
                loss_views_per_call=(1,) * 16,
            )
        )
        block = clocks.report_block(wall_time_s=10.0)
        assert "rasterizer_calls_per_step" in block
        assert block["rasterizer_calls_per_step"] == 4.0  # 16 calls / 4 steps


class TestExecutionSummaryLines:
    """Test (c): execution-summary block is formatted correctly."""

    def test_execution_summary_empty_clocks(self):
        """Empty clocks returns placeholder."""
        clocks = SupervisionWorkClocks()
        lines = execution_summary_lines(clocks)
        assert len(lines) == 1
        assert "(no optimization steps observed)" in lines[0]

    def test_execution_summary_basic_structure(self):
        """Summary block has expected sections."""
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
            )
        )
        lines = execution_summary_lines(
            clocks,
            requested_batch_size=4,
            wall_time_s=10.0,
        )
        text = "\n".join(lines)
        assert "optimization_steps:" in text
        assert "supervision_views:" in text
        assert "rasterizer_calls:" in text
        assert "supervised_pixels:" in text
        assert "Realized execution shape:" in text
        assert "requested_batch_size: 4" in text
        assert "realized_views_per_step:" in text
        assert "realized_rasterizer_calls_per_step:" in text
        assert "Realized rates:" in text
        assert "optimization_step_rate_hz:" in text
        assert "supervision_view_rate_hz:" in text

    def test_execution_summary_requested_batch_unknown(self):
        """Handles missing requested_batch_size gracefully."""
        clocks = SupervisionWorkClocks()
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
            )
        )
        lines = execution_summary_lines(
            clocks,
            requested_batch_size=None,
            wall_time_s=1.0,
        )
        text = "\n".join(lines)
        assert "requested_batch_size: unknown" in text

    def test_execution_summary_values_are_correct(self):
        """Summary block values match expected computations."""
        clocks = SupervisionWorkClocks()
        # 4 steps, 16 views (B=4), 8 rasterizer calls (C=2 batching)
        for i in range(4):
            clocks.observe(
                SimpleNamespace(
                    optimizer_step_count=1,
                    supervision_view_count=4,
                    rasterizer_call_count=2,
                    supervised_pixel_count=4 * 614400,
                    loss_objective_call_count=4,
                    backward_call_count=4,
                    renderer_camera_batch_sizes=(2, 2),
                    loss_views_per_call=(2, 2),
                )
            )
        lines = execution_summary_lines(
            clocks,
            requested_batch_size=4,
            wall_time_s=8.0,
        )
        text = "\n".join(lines)
        # 4 steps / 8 seconds = 0.5 Hz
        assert "optimization_step_rate_hz: 0.500" in text
        # 16 views / 8 seconds = 2.0 Hz
        assert "supervision_view_rate_hz: 2.000" in text
        # 16 views / 4 steps = 4.0
        assert "realized_views_per_step: 4.00" in text
        # 8 rasterizer calls / 4 steps = 2.0
        assert "realized_rasterizer_calls_per_step: 2.00" in text
        # mean batch: 4 / 2 = 2.0
        assert "renderer_camera_batch_size_mean: 2.00" in text

    def test_execution_summary_no_wall_time(self):
        """Summary block skips rate section without wall_time_s."""
        clocks = SupervisionWorkClocks()
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
            )
        )
        lines = execution_summary_lines(
            clocks,
            requested_batch_size=1,
            wall_time_s=None,
        )
        text = "\n".join(lines)
        assert "Realized execution shape:" in text
        assert "Realized rates:" not in text
