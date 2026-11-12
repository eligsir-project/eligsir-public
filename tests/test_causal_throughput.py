from __future__ import annotations

import json


def test_plot_causal_throughput_is_a_true_noop_without_a_trace(tmp_path):
    from eligsir.runtime.causal_throughput import plot_causal_throughput

    assert plot_causal_throughput(tmp_path) is None
    assert not (tmp_path / "causal_throughput_over_time.svg").exists()


def test_plot_causal_throughput_is_a_true_noop_for_an_empty_trace(tmp_path):
    from eligsir.runtime.causal_throughput import (
        plot_causal_throughput,
    )

    (tmp_path / "causal_throughput.jsonl").write_text("")

    assert plot_causal_throughput(tmp_path) is None


def test_record_causal_throughput_writes_expected_row(tmp_path):
    from eligsir.runtime.causal_throughput import (
        record_causal_throughput,
    )

    record_causal_throughput(
        tmp_path,
        elapsed_s=20.0,
        optimizer_steps=200,
        supervision_views=800,
        mapping_active_s=10.0,
        prev_elapsed_s=10.0,
        prev_optimizer_steps=100,
        prev_supervision_views=400,
        prev_mapping_active_s=5.0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "causal_throughput.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["elapsed_s"] == 20.0
    assert row["mapping_active_s"] == 10.0
    assert row["optimizer_steps"] == 200
    assert row["supervision_views"] == 800
    # Instantaneous (delta) rate: (200-100) steps / (10.0-5.0)s = 20/s.
    assert row["optimizer_steps_per_s"] == 20.0
    # (800-400) views / 5.0s = 80/s.
    assert row["supervision_views_per_s"] == 80.0
    # Cumulative-average rate: total/total, matching the printed log line.
    assert row["optimizer_steps_per_s_cumulative"] == 200 / 10.0
    assert row["supervision_views_per_s_cumulative"] == 800 / 10.0
    assert row["queue_depth"] is None
    assert row["source_offered"] is None


def test_record_causal_throughput_null_rate_on_degenerate_interval(tmp_path):
    """A zero (or ~zero) active-time delta must not divide-by-zero into a
    spurious spike -- it should record null, not inf/nan."""
    from eligsir.runtime.causal_throughput import (
        record_causal_throughput,
    )

    record_causal_throughput(
        tmp_path,
        elapsed_s=1.0,
        optimizer_steps=5,
        supervision_views=20,
        mapping_active_s=0.0,
        prev_elapsed_s=0.0,
        prev_optimizer_steps=0,
        prev_supervision_views=0,
        prev_mapping_active_s=0.0,
    )

    row = json.loads((tmp_path / "causal_throughput.jsonl").read_text())
    assert row["optimizer_steps_per_s"] is None
    assert row["supervision_views_per_s"] is None
    # Cumulative rate still has a defined (if huge) value via the 1e-9 floor,
    # never an actual ZeroDivisionError.
    assert row["optimizer_steps_per_s_cumulative"] > 0


def test_record_causal_throughput_is_silent_on_write_failure(tmp_path):
    """Telemetry must never be able to break a training run (matches
    record_resolution_tier_usage's own contract)."""
    from eligsir.runtime.causal_throughput import record_causal_throughput

    # Not a directory -> writing "<path>/causal_throughput.jsonl" raises
    # NotADirectoryError (an OSError subclass), which must be swallowed.
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x")

    record_causal_throughput(
        not_a_dir,
        elapsed_s=1.0,
        optimizer_steps=1,
        supervision_views=1,
        mapping_active_s=1.0,
        prev_elapsed_s=0.0,
        prev_optimizer_steps=0,
        prev_supervision_views=0,
        prev_mapping_active_s=0.0,
    )  # must not raise


def test_plot_causal_throughput_renders_instantaneous_and_cumulative_series(tmp_path):
    from eligsir.runtime.causal_throughput import (
        plot_causal_throughput,
    )

    rows = []
    prev = dict(elapsed_s=0.0, optimizer_steps=0, supervision_views=0, mapping_active_s=0.0)
    for tick in range(1, 11):
        elapsed_s = 10.0 * tick
        # A deliberate mid-run slowdown (steps/s halves for ticks 5-7) that a
        # cumulative average would smear out but an instantaneous rate must
        # show -- this is the whole point of preferring delta rates.
        steps_per_tick = 50 if tick in (5, 6, 7) else 100
        optimizer_steps = prev["optimizer_steps"] + steps_per_tick
        supervision_views = prev["supervision_views"] + 4 * steps_per_tick
        mapping_active_s = elapsed_s
        rows.append(
            {
                "elapsed_s": elapsed_s,
                "mapping_active_s": mapping_active_s,
                "optimizer_steps": optimizer_steps,
                "supervision_views": supervision_views,
                "optimizer_steps_per_s": steps_per_tick / 10.0,
                "supervision_views_per_s": 4 * steps_per_tick / 10.0,
                "optimizer_steps_per_s_cumulative": optimizer_steps / mapping_active_s,
                "supervision_views_per_s_cumulative": supervision_views / mapping_active_s,
                "queue_depth": None,
                "source_offered": None,
                "source_submitted": None,
                "transport_dropped": None,
            }
        )
        prev = dict(
            elapsed_s=elapsed_s,
            optimizer_steps=optimizer_steps,
            supervision_views=supervision_views,
            mapping_active_s=mapping_active_s,
        )
    (tmp_path / "causal_throughput.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = plot_causal_throughput(tmp_path)

    assert output == tmp_path / "causal_throughput_over_time.svg"
    assert output.is_file()
    assert output.stat().st_size > 0

    # Confirm the synthetic trace actually exercises the slowdown-then-recovery
    # shape the plot is meant to make visible (independent of matplotlib
    # internals, which we don't introspect).
    slow_rows = [r for r in rows if r["optimizer_steps_per_s"] < 10.0]
    fast_rows = [r for r in rows if r["optimizer_steps_per_s"] >= 10.0]
    assert slow_rows and fast_rows


def test_plot_causal_throughput_handles_null_instantaneous_rates(tmp_path):
    """A trace containing a degenerate-interval row (null rate) must still
    render, treating the gap as a missing point rather than crashing."""
    from eligsir.runtime.causal_throughput import (
        plot_causal_throughput,
    )

    rows = [
        {
            "elapsed_s": 0.0,
            "mapping_active_s": 0.0,
            "optimizer_steps": 0,
            "supervision_views": 0,
            "optimizer_steps_per_s": None,
            "supervision_views_per_s": None,
            "optimizer_steps_per_s_cumulative": 0.0,
            "supervision_views_per_s_cumulative": 0.0,
        },
        {
            "elapsed_s": 10.0,
            "mapping_active_s": 10.0,
            "optimizer_steps": 100,
            "supervision_views": 400,
            "optimizer_steps_per_s": 10.0,
            "supervision_views_per_s": 40.0,
            "optimizer_steps_per_s_cumulative": 10.0,
            "supervision_views_per_s_cumulative": 40.0,
        },
    ]
    (tmp_path / "causal_throughput.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    output = plot_causal_throughput(tmp_path)

    assert output is not None and output.is_file()
