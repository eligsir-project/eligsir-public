"""Issue #703: Supervision-work accounting unit tests.

CPU-only tests of derived ratio and exposure-distribution math.
"""

from __future__ import annotations

import pytest

from eligsir.utils.supervision_work import (
    per_view_exposure_distribution,
    rasterizer_calls_per_optimization_step,
    supervised_pixels_per_optimization_step,
    supervision_view_rate_hz,
    supervision_views_per_optimization_step,
)


class TestDerivedRatios:
    """Test supervision-work derived ratios."""

    def test_supervision_views_per_step_b1(self):
        """B=1 unbatched: 1 view per step."""
        assert supervision_views_per_optimization_step(10, 10) == 1.0

    def test_supervision_views_per_step_b4_sequential(self):
        """B=4 sequential: 4 views per step."""
        assert supervision_views_per_optimization_step(10, 40) == 4.0

    def test_supervision_views_per_step_fractional(self):
        """Fractional supervision ratio (e.g., some dropped views)."""
        # 30 views over 20 steps = 1.5 views/step
        assert supervision_views_per_optimization_step(20, 30) == 1.5

    def test_supervision_views_per_step_zero_steps(self):
        """No steps -> ratio is 0."""
        assert supervision_views_per_optimization_step(0, 100) == 0.0

    def test_supervision_view_rate_counts_true_b1_views(self):
        assert supervision_view_rate_hz(10, 2.0) == 5.0

    def test_supervision_view_rate_counts_true_b4_camera_views(self):
        """B4 contributes four camera views per transaction, not one."""
        assert supervision_view_rate_hz(40, 2.0) == 20.0

    def test_supervision_view_rate_zero_before_wall_clock(self):
        assert supervision_view_rate_hz(40, 0.0) == 0.0

    def test_rasterizer_calls_per_step_b1(self):
        """B=1: 1 rasterizer call per step."""
        assert rasterizer_calls_per_optimization_step(10, 10) == 1.0

    def test_rasterizer_calls_per_step_b4_sequential(self):
        """B=4 sequential: 4 calls per step (one per view)."""
        assert rasterizer_calls_per_optimization_step(10, 40) == 4.0


    def test_rasterizer_calls_per_step_zero_steps(self):
        """No steps -> ratio is 0."""
        assert rasterizer_calls_per_optimization_step(0, 100) == 0.0

    def test_supervised_pixels_per_step_1080p(self):
        """1080p full render: ~2M pixels per view."""
        # 1920x1080 * 1 view * 10 steps = 20,736,000 pixels
        pixels_1080p = 1920 * 1080
        # 10 steps, 1 view per step
        assert supervised_pixels_per_optimization_step(10, pixels_1080p * 10) == pixels_1080p

    def test_supervised_pixels_per_step_720p(self):
        """720p full render: ~1M pixels per view."""
        pixels_720p = 1280 * 720
        # 10 steps, 1 view per step
        assert supervised_pixels_per_optimization_step(10, pixels_720p * 10) == pixels_720p

    def test_supervised_pixels_per_step_b4_sequential_1080p(self):
        """B=4 sequential 1080p: 4 renders per step."""
        pixels_1080p = 1920 * 1080
        # 10 steps, 4 views per step
        total_pixels = pixels_1080p * 4 * 10
        assert supervised_pixels_per_optimization_step(10, total_pixels) == (pixels_1080p * 4)

    def test_supervised_pixels_per_step_zero_steps(self):
        """No steps -> ratio is 0."""
        assert supervised_pixels_per_optimization_step(0, 1_000_000) == 0.0


class TestPerViewExposureDistribution:
    """Test per-camera supervision exposure distribution aggregates."""

    def test_uniform_exposure(self):
        """All cameras trained equally."""
        cameras = {"a": 100, "b": 100, "c": 100}
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_mean"] == 100
        assert result["supervision_exposures_per_view_min"] == 100
        assert result["supervision_exposures_per_view_p10"] == 100
        assert result["supervision_exposures_per_view_p50"] == 100
        assert result["supervision_exposures_per_view_p90"] == 100
        assert result["supervision_exposures_per_view_max"] == 100
        assert result["never_supervised_view_count"] == 0

    def test_wide_exposure_range(self):
        """Some cameras trained much more than others."""
        # 1 camera with high exposure, 9 with lower exposure
        cameras = {"a": 1000}
        for i in range(9):
            cameras[f"b{i}"] = 100
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_min"] == 100
        assert result["supervision_exposures_per_view_max"] == 1000
        assert result["supervision_exposures_per_view_mean"] == 190  # (1000 + 9*100) / 10
        assert result["never_supervised_view_count"] == 0

    def test_intentionally_starved_camera(self):
        """One camera never supervised (forgotten)."""
        cameras = {"a": 100, "b": 100, "c": 0}
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_min"] == 0
        assert result["supervision_exposures_per_view_max"] == 100
        assert result["never_supervised_view_count"] == 1
        assert result["supervision_exposures_per_view_p50"] == 100  # median of [0, 100, 100]

    def test_multiple_starved_cameras(self):
        """Multiple cameras never supervised."""
        cameras = {"a": 100, "b": 0, "c": 0, "d": 0}
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_min"] == 0
        assert result["supervision_exposures_per_view_max"] == 100
        assert result["never_supervised_view_count"] == 3
        # mean = (100 + 0 + 0 + 0) / 4 = 25
        assert result["supervision_exposures_per_view_mean"] == 25

    def test_percentile_computation_small_dataset(self):
        """Percentiles on a small dataset."""
        # 5 cameras: [0, 25, 50, 75, 100]
        cameras = {
            "a": 0,
            "b": 25,
            "c": 50,
            "d": 75,
            "e": 100,
        }
        result = per_view_exposure_distribution(cameras)
        # p10 should be between 0 and 25, closer to 0 (10% into the range)
        # p50 (median) should be 50
        # p90 should be between 75 and 100, closer to 100 (90% into the range)
        assert result["supervision_exposures_per_view_p50"] == 50
        assert 0 <= result["supervision_exposures_per_view_p10"] <= 25
        assert 75 <= result["supervision_exposures_per_view_p90"] <= 100

    def test_empty_input(self):
        """No cameras."""
        result = per_view_exposure_distribution({})
        assert result["supervision_exposures_per_view_mean"] == 0
        assert result["supervision_exposures_per_view_min"] == 0
        assert result["supervision_exposures_per_view_max"] == 0
        assert result["never_supervised_view_count"] == 0

    def test_single_camera(self):
        """Only one camera in the dataset."""
        cameras = {"only": 42}
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_mean"] == 42
        assert result["supervision_exposures_per_view_min"] == 42
        assert result["supervision_exposures_per_view_p50"] == 42
        assert result["supervision_exposures_per_view_max"] == 42
        assert result["never_supervised_view_count"] == 0

    def test_single_camera_never_supervised(self):
        """Only one camera, never supervised."""
        cameras = {"only": 0}
        result = per_view_exposure_distribution(cameras)
        assert result["supervision_exposures_per_view_mean"] == 0
        assert result["supervision_exposures_per_view_min"] == 0
        assert result["supervision_exposures_per_view_max"] == 0
        assert result["never_supervised_view_count"] == 1
