from __future__ import annotations

import pytest

from scripts.validate_causal_phase_throughput import validate_phase_throughput


def test_validate_phase_throughput_accepts_both_phase_rates_at_target():
    report = validate_phase_throughput(
        {
            "supervision_views_per_active_s": 120.0,
            "supervision_views_per_elapsed_s": 110.0,
        },
        {
            "refinement_supervision_views_per_active_s": 130.0,
            "refinement_supervision_views_per_wall_s": 105.0,
        },
        minimum_views_per_s=100.0,
    )

    assert report["mapping"]["active_views_per_s"] == 120.0
    assert report["refinement"]["wall_views_per_s"] == 105.0


def test_validate_phase_throughput_fails_closed_for_a_slow_phase():
    with pytest.raises(ValueError, match="refinement wall"):
        validate_phase_throughput(
            {
                "supervision_views_per_active_s": 120.0,
                "supervision_views_per_elapsed_s": 110.0,
            },
            {
                "refinement_supervision_views_per_active_s": 130.0,
                "refinement_supervision_views_per_wall_s": 99.9,
            },
            minimum_views_per_s=100.0,
        )
