"""Issue #701: progress postfix denominator fix and label disambiguation.

Tests for the canonical progress-bar postfix computation, specifically:
- The it/V denominator bug fix (was keyframes_admitted, now n_train_cameras)
- Label consistency and disambiguation
- Torch-free implementation suitable for CPU CI
"""
from __future__ import annotations

import pytest

from eligsir.utils.execution_semantics import compute_progress_postfix


class TestProgressPostfixDenominator:
    """it/V denominator correctness (issue #701)."""

    def test_it_v_divides_by_n_train_cameras_not_keyframes_admitted(self):
        """it/V should be iteration // n_train_cameras, not // keyframes_admitted."""
        # Setup: 100 iterations, 25 training views (cameras), 50 keyframes admitted
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=50,
            n_train_cameras=25,  # The correct denominator
            keyframes_admitted=50,  # The old (wrong) denominator
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        # With 25 views: 100 // 25 = 4
        # With 50 keyframes (old, wrong): 100 // 50 = 2 (different result!)
        assert postfix["it/V"] == 4, (
            "it/V must divide by n_train_cameras (25), not keyframes_admitted (50). "
            "Got it/V={}, expected 4.".format(postfix["it/V"])
        )

    def test_it_v_label_matches_denominator(self):
        """Label 'V' must match the denominator in it/V."""
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=500,
            n_frames_ingested=100,
            n_train_cameras=20,
            keyframes_admitted=10,
            elapsed_s=5.0,
            ema_loss=0.3,
        )
        # Label says "V" is n_train_cameras
        assert postfix["V"] == 20
        # Denominator should use same value
        assert postfix["it/V"] == 100 // 20  # 5

    def test_it_v_with_zero_views_fallback(self):
        """Zero views should be clamped to 1 to avoid division by zero."""
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=100,
            n_frames_ingested=10,
            n_train_cameras=0,  # Edge case: no training cameras yet
            keyframes_admitted=0,
            elapsed_s=1.0,
            ema_loss=0.5,
        )
        # Should clamp to 1 and compute 100 // 1 = 100
        assert postfix["it/V"] == 100


class TestProgressPostfixLabels:
    """Label clarity and disambiguation (issue #701)."""

    def test_postfix_keys_are_standard(self):
        """Postfix must contain exactly the keys legacy parsers expect."""
        postfix = compute_progress_postfix(
            iteration=50,
            n_gaussians=2000,
            n_frames_ingested=20,
            n_train_cameras=10,
            keyframes_admitted=8,
            elapsed_s=2.5,
            ema_loss=0.4,
        )
        expected_keys = {"Loss", "N", "F", "V", "it/V", "mapFPS", "adm%"}
        assert set(postfix.keys()) == expected_keys, (
            "Postfix keys changed; legacy parsers will break. "
            f"Expected {expected_keys}, got {set(postfix.keys())}"
        )

    def test_loss_field_format(self):
        """Loss field should be formatted consistently."""
        postfix = compute_progress_postfix(
            iteration=10,
            n_gaussians=500,
            n_frames_ingested=5,
            n_train_cameras=5,
            keyframes_admitted=3,
            elapsed_s=1.0,
            ema_loss=0.12345678,
        )
        # Should be formatted to 7 decimal places
        assert postfix["Loss"] == "0.1234568"

    def test_mapfps_label_clarity(self):
        """mapFPS is a lifetime average of keyframes admitted per second."""
        # 60 keyframes admitted in 10 seconds = 6 FPS
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=50,
            keyframes_admitted=60,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        assert postfix["mapFPS"] == "6.0"

    def test_adm_percent_label_clarity(self):
        """adm% is the admission percentage: keyframes_admitted / n_frames_ingested."""
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,  # 100 frames ingested total
            n_train_cameras=50,
            keyframes_admitted=25,  # 25 of them admitted
            elapsed_s=5.0,
            ema_loss=0.5,
        )
        # 25 / 100 = 0.25 = 25%
        assert postfix["adm%"] == "25"

    def test_f_is_frames_ingested(self):
        """F should be the total frames ingested from source."""
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=200,  # The actual source frame count
            n_train_cameras=50,
            keyframes_admitted=40,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        assert postfix["F"] == 200

    def test_v_is_training_cameras(self):
        """V should be the number of training cameras."""
        postfix = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=42,  # The training camera count
            keyframes_admitted=30,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        assert postfix["V"] == 42


class TestProgressPostfixEdgeCases:
    """Edge cases and zero-division safeguards."""

    def test_zero_elapsed_time(self):
        """Zero elapsed time should not cause division by zero."""
        postfix = compute_progress_postfix(
            iteration=10,
            n_gaussians=100,
            n_frames_ingested=10,
            n_train_cameras=5,
            keyframes_admitted=5,
            elapsed_s=0.0,  # Edge case: just started
            ema_loss=0.5,
        )
        # Should clamp to 1e-6 to avoid /0
        # mapFPS = 5 / 1e-6 = 5_000_000
        assert float(postfix["mapFPS"]) == 5e6

    def test_zero_frames_ingested_division(self):
        """Zero frames ingested should not cause division by zero in adm%."""
        postfix = compute_progress_postfix(
            iteration=10,
            n_gaussians=100,
            n_frames_ingested=0,  # Edge case: no frames yet
            n_train_cameras=5,
            keyframes_admitted=0,
            elapsed_s=1.0,
            ema_loss=0.5,
        )
        # With safeguards: _adm = max(1, 0) = 1
        # adm% = 100 * 1 / max(1, 0) = 100 / 1 = 100
        # (Edge case: both zero gets clamped to 1/1, showing 100%)
        assert postfix["adm%"] == "100"

    def test_all_zero_initialization(self):
        """All-zero state at startup."""
        postfix = compute_progress_postfix(
            iteration=0,
            n_gaussians=0,
            n_frames_ingested=0,
            n_train_cameras=0,
            keyframes_admitted=0,
            elapsed_s=0.0,
            ema_loss=0.0,
        )
        assert postfix["Loss"] == "0.0000000"
        assert postfix["N"] == 0
        assert postfix["F"] == 0
        assert postfix["V"] == 1  # Clamped to 1 internally: max(1, 0) = 1
        assert postfix["it/V"] == 0  # 0 // 1 = 0
        assert postfix["adm%"] == "100"  # 100 * 1 / max(1, 0) = 100 / 1 = 100

    def test_high_iteration_count(self):
        """Large iteration counts should compute correctly."""
        postfix = compute_progress_postfix(
            iteration=1_000_000,
            n_gaussians=500_000,
            n_frames_ingested=10_000,
            n_train_cameras=100,
            keyframes_admitted=5_000,
            elapsed_s=1000.0,
            ema_loss=0.1,
        )
        # 1_000_000 // 100 = 10_000
        assert postfix["it/V"] == 10_000
        # 5_000 / 1000 = 5.0
        assert postfix["mapFPS"] == "5.0"


class TestProgressPostfixConsistency:
    """Consistency properties across different scenarios."""

    def test_doubling_views_halves_it_v(self):
        """Doubling views should halve it/V (same iterations)."""
        postfix1 = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=10,
            keyframes_admitted=10,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        postfix2 = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=20,  # Doubled
            keyframes_admitted=20,  # Also doubled for fair comparison
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        # it/V should be exactly half when views double
        assert postfix1["it/V"] == postfix2["it/V"] * 2, (
            f"Expected it/V to halve when views double. "
            f"Got {postfix1['it/V']} vs {postfix2['it/V']}"
        )

    def test_mapfps_independent_of_views(self):
        """mapFPS should be independent of n_train_cameras (depends on keyframes)."""
        postfix1 = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=10,  # Few views
            keyframes_admitted=50,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        postfix2 = compute_progress_postfix(
            iteration=100,
            n_gaussians=1000,
            n_frames_ingested=100,
            n_train_cameras=1000,  # Many views (shouldn't affect mapFPS)
            keyframes_admitted=50,
            elapsed_s=10.0,
            ema_loss=0.5,
        )
        # mapFPS depends on keyframes_admitted and elapsed_s, not n_train_cameras
        assert postfix1["mapFPS"] == postfix2["mapFPS"], (
            "mapFPS should be independent of n_train_cameras"
        )
