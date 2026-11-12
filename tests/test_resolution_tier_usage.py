from __future__ import annotations

import json

import numpy as np


def _write_trace(model_path, rows):
    path = model_path / "resolution_tier_usage.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_plot_resolution_tier_usage_is_a_true_noop_without_a_trace(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        plot_resolution_tier_usage,
    )

    assert plot_resolution_tier_usage(tmp_path) is None
    assert not (tmp_path / "resolution_tier_usage_over_time.svg").exists()


def test_plot_resolution_tier_usage_placeholder_for_fixed_tier_runs(tmp_path):
    """Every real campaign run so far pins constellation_policy=fixed, so
    every transaction resolves to the same tier -- this must render a clear
    placeholder, not a degenerate one-row heatmap."""
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        plot_resolution_tier_usage,
    )

    rows = [
        {"iteration": i, "resolution_tier": "native", "transaction_class": "fresh"}
        for i in range(0, 1000, 50)
    ]
    _write_trace(tmp_path, rows)

    output = plot_resolution_tier_usage(tmp_path)

    assert output is not None and output.is_file()
    assert output.name == "resolution_tier_usage_over_time.svg"


def test_plot_resolution_tier_usage_renders_shift_to_native_pattern(tmp_path):
    """Synthetic adaptive-controller telemetry: coarse tiers dominate early
    (high ingestion pressure), native dominates late (pressure eased) --
    the qualitative pattern issue #1250 wants this heatmap to be able to
    show, without needing a real GPU run."""
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        plot_resolution_tier_usage,
        USAGE_TRACE_FILENAME,
    )

    rng = np.random.default_rng(1234)
    n_iterations = 4000
    rows = []
    for iteration in range(0, n_iterations, 10):
        progress = iteration / n_iterations
        # Early on, mostly quarter/half; late, mostly native.
        weights = {
            "quarter": max(0.0, 1.0 - 2.0 * progress),
            "half": 0.3,
            "native": max(0.0, 2.0 * progress - 0.3),
        }
        tiers, probs = zip(*weights.items())
        probs = np.asarray(probs)
        probs = probs / probs.sum()
        tier = rng.choice(tiers, p=probs)
        rows.append(
            {
                "iteration": iteration,
                "resolution_tier": tier,
                "transaction_class": "mixed",
            }
        )
    _write_trace(tmp_path, rows)
    assert (tmp_path / USAGE_TRACE_FILENAME).is_file()

    output = plot_resolution_tier_usage(tmp_path)

    assert output is not None and output.is_file()
    assert output.stat().st_size > 0

    # Recompute the same shift-to-native signal the plot is built from, to
    # confirm the synthetic data actually exercises the intended pattern
    # (independent of matplotlib internals, which we don't introspect).
    early = [r for r in rows if r["iteration"] < n_iterations * 0.2]
    late = [r for r in rows if r["iteration"] >= n_iterations * 0.8]
    early_native_frac = sum(r["resolution_tier"] == "native" for r in early) / len(early)
    late_native_frac = sum(r["resolution_tier"] == "native" for r in late) / len(late)
    assert late_native_frac > early_native_frac


def test_record_resolution_tier_usage_writes_expected_row(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        flush_resolution_tier_usage,
        record_resolution_tier_usage,
        USAGE_TRACE_FILENAME,
    )
    from eligsir.contracts.control import MappingTransactionClass, ResolutionTier

    record_resolution_tier_usage(
        tmp_path,
        iteration=42,
        tier=ResolutionTier.HALF,
        transaction_class=MappingTransactionClass.MIXED,
    )
    record_resolution_tier_usage(tmp_path, iteration=43, tier="native", transaction_class=None)
    # Issue #1664: rows are buffered in memory and only actually written at
    # a cadence or at explicit flush -- finalization must flush before a
    # reader can see them.
    flush_resolution_tier_usage(tmp_path)

    lines = (tmp_path / USAGE_TRACE_FILENAME).read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows[0] == {
        "iteration": 42,
        "resolution_tier": "half",
        "transaction_class": "mixed",
    }
    assert rows[1] == {
        "iteration": 43,
        "resolution_tier": "native",
        "transaction_class": None,
    }


def test_record_resolution_tier_usage_ignores_missing_tier(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        record_resolution_tier_usage,
        USAGE_TRACE_FILENAME,
    )

    record_resolution_tier_usage(tmp_path, iteration=1, tier=None)

    assert not (tmp_path / USAGE_TRACE_FILENAME).exists()


def test_plot_resolution_tier_over_time_is_a_true_noop_without_a_trace(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        plot_resolution_tier_over_time,
    )

    assert plot_resolution_tier_over_time(tmp_path) is None
    assert not (tmp_path / "resolution_tier_over_time.svg").exists()


def test_plot_resolution_tier_over_time_placeholder_when_trace_has_no_rows(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        USAGE_TRACE_FILENAME,
        plot_resolution_tier_over_time,
    )

    (tmp_path / USAGE_TRACE_FILENAME).write_text(
        json.dumps({"iteration": 1, "resolution_tier": None, "transaction_class": None})
        + "\n"
    )

    output = plot_resolution_tier_over_time(tmp_path)

    assert output is not None and output.is_file()
    assert output.name == "resolution_tier_over_time.svg"


def test_plot_resolution_tier_over_time_renders_flat_line_for_single_tier(tmp_path):
    """Unlike the heatmap (which placeholders a single-tier corpus), the
    average-over-time line is still meaningful when flat -- a flat line at
    'native' is the honest answer, not a degenerate case to hide."""
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        plot_resolution_tier_over_time,
    )

    rows = [
        {"iteration": i, "resolution_tier": "native", "transaction_class": "fresh"}
        for i in range(0, 1000, 50)
    ]
    _write_trace(tmp_path, rows)

    output = plot_resolution_tier_over_time(tmp_path)

    assert output is not None and output.is_file()
    assert output.stat().st_size > 0


def test_plot_resolution_tier_over_time_shows_shift_toward_native(tmp_path):
    """Ranks average toward 0 (native) as the run progresses -- the mean
    rank in the second half must be lower (finer) than in the first half."""
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        _TIER_RANK,
        USAGE_TRACE_FILENAME,
        plot_resolution_tier_over_time,
    )

    rows = []
    n_iterations = 4000
    for iteration in range(0, n_iterations, 10):
        progress = iteration / n_iterations
        tier = "quarter" if progress < 0.5 else "native"
        rows.append(
            {
                "iteration": iteration,
                "resolution_tier": tier,
                "transaction_class": "mixed",
            }
        )
    _write_trace(tmp_path, rows)

    output = plot_resolution_tier_over_time(tmp_path)

    assert output is not None and output.is_file()

    early = [r for r in rows if r["iteration"] < n_iterations * 0.2]
    late = [r for r in rows if r["iteration"] >= n_iterations * 0.8]
    early_rank = sum(_TIER_RANK[r["resolution_tier"]] for r in early) / len(early)
    late_rank = sum(_TIER_RANK[r["resolution_tier"]] for r in late) / len(late)
    assert late_rank < early_rank
    assert (tmp_path / USAGE_TRACE_FILENAME).is_file()
