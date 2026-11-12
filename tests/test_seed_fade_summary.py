"""scripts/seed_fade_summary.py must read the TB tag train_streaming.py actually emits
(``optimization/seed_psnr``), not the stale ``train/seed_psnr`` tag (issue #346 blocker).

Regression coverage for the fix: the tag resolver must prefer the current tag, fall back
to the legacy tag for pre-rename runs (with a loud warning, not a silent null), and warn
loudly -- never silently return an empty/None column -- when neither tag has data.
"""

from __future__ import annotations

import pytest

from scripts.seed_fade_summary import (
    SEED_PSNR_TAG,
    SEED_PSNR_TAG_LEGACY,
    curve_stats,
    seed_psnr_series,
)


def test_seed_psnr_tag_matches_emitted_tag():
    # The tag train_streaming.py actually calls tb_writer.add_scalar(...) with.
    assert SEED_PSNR_TAG == "optimization/seed_psnr"
    assert SEED_PSNR_TAG_LEGACY == "train/seed_psnr"


def test_seed_psnr_series_prefers_current_tag():
    data = {
        SEED_PSNR_TAG: [(0, 20.0), (100, 25.0)],
        SEED_PSNR_TAG_LEGACY: [(0, 1.0)],  # must not be used when current tag has data
    }
    series = seed_psnr_series(data, "run_a")
    assert series == [(0, 20.0), (100, 25.0)]


def test_seed_psnr_series_falls_back_to_legacy_with_warning(capsys):
    data = {SEED_PSNR_TAG_LEGACY: [(0, 18.0), (100, 22.0)]}
    series = seed_psnr_series(data, "run_b")
    assert series == [(0, 18.0), (100, 22.0)]
    err = capsys.readouterr().err
    assert "run_b" in err
    assert "falling back to legacy tag" in err


def test_seed_psnr_series_warns_loudly_when_absent(capsys):
    data = {}
    series = seed_psnr_series(data, "run_c")
    assert series == []
    err = capsys.readouterr().err
    assert "run_c" in err
    assert "NO seed_psnr data" in err


def test_curve_stats_on_resolved_series():
    data = {SEED_PSNR_TAG: [(0, 30.0), (500, 15.0), (1000, 28.0)]}
    series = seed_psnr_series(data, "run_d")
    stats = curve_stats(series)
    assert stats is not None
    assert stats["min"] == 15.0
    assert stats["last"] == 28.0
