"""Per-tick optimizer/data throughput telemetry and its report plot.

Companion to ``spatial_replay_focus.jsonl``/``plot_spatial_replay_focus``
(``eligsir/runtime/causal_evaluation.py``) and
``resolution_tier_usage.jsonl``/``plot_resolution_tier_usage``
(``eligsir/mapping/scheduler/resolution_tier_usage.py``): one JSON line
per periodically-sampled causal-replay tick, carrying the same
optimizer-steps/s and supervision-views/s numbers the ``[causal-progress]``
log line (``eligsir/runtime/causal_replay.py``'s
``PacedCausalReplayDriver.run``) prints -- but persisted to disk so a real
time-series plot can be produced as part of the standard report, instead of
only ever existing as train.log scrollback.

Unlike the printed log line -- a CUMULATIVE average, ``total optimizer_steps
/ total mapping_active_s`` since the run started -- each row here is
primarily a rolling DELTA-based (instantaneous) rate over the interval since
the previously-recorded row. A cumulative average flattens out over a long
run and can hide a real mid-run throughput regression (e.g. the growth-event
slowdown documented in the mcmc-growth-turbulence-itspeed finding) inside an
ever-thickening denominator; a per-interval delta rate is what actually
shows the shape of a throughput change over time. Each row also carries the
raw cumulative counters plus the cumulative-average rate (matching the log
line exactly), so a consumer that wants the coarser cumulative view doesn't
have to re-derive it.

The caller (``CausalEvaluationObserver.observe_tick``) already tracks the
previous tick's cumulative counters for its own reasons, so
``record_causal_throughput`` stays a pure function with no held state of its
own -- both the current and previous cumulative values are passed in by the
caller, matching the rest of this telemetry family's style (e.g.
``record_resolution_tier_usage``).
"""

from __future__ import annotations

from eligsir.plot_style import svg_output_path

import json
from pathlib import Path
from typing import Any

THROUGHPUT_TRACE_FILENAME = "causal_throughput.jsonl"
THROUGHPUT_PLOT_FILENAME = "causal_throughput_over_time.svg"


def record_causal_throughput(
    model_path: str | Path,
    *,
    elapsed_s: float,
    optimizer_steps: int,
    supervision_views: int,
    mapping_active_s: float,
    prev_elapsed_s: float,
    prev_optimizer_steps: int,
    prev_supervision_views: int,
    prev_mapping_active_s: float,
    queue_depth: int | None = None,
    source_offered: int | None = None,
    source_submitted: int | None = None,
    transport_dropped: int | None = None,
) -> None:
    """Append one throughput sample to the run's trace.

    Best-effort and silent on failure (matching
    ``record_resolution_tier_usage``'s own pattern): telemetry must never be
    able to break a training run.

    The optional admission-side counters (``queue_depth``, ``source_offered``,
    ``source_submitted``, ``transport_dropped``) mirror fields already printed
    on the ``[causal-progress]`` log line. They are optional because they are
    only available from the paced replay driver's own loop scope
    (``causal_replay.py``'s ``PacedCausalReplayDriver.run``), not from the
    ``CausalReplayClock`` the tick-clock observer receives -- plumbing them
    through would require widening that dataclass, which is left as a
    follow-up rather than done speculatively here. When omitted the row
    simply carries ``null`` for those keys rather than inventing a value.
    """

    delta_active_s = max(0.0, float(mapping_active_s) - float(prev_mapping_active_s))
    delta_steps = max(0, int(optimizer_steps) - int(prev_optimizer_steps))
    delta_views = max(0, int(supervision_views) - int(prev_supervision_views))
    # Guard against a degenerate/zero interval (the very first sample, or a
    # paused-mapping period with no active work) rather than dividing by ~0
    # and writing a spurious spike -- null is the honest value there.
    if delta_active_s > 1e-6:
        steps_per_s: float | None = delta_steps / delta_active_s
        views_per_s: float | None = delta_views / delta_active_s
    else:
        steps_per_s = None
        views_per_s = None
    cumulative_active_s = max(1e-9, float(mapping_active_s))
    row: dict[str, Any] = {
        "elapsed_s": float(elapsed_s),
        "mapping_active_s": float(mapping_active_s),
        "optimizer_steps": int(optimizer_steps),
        "supervision_views": int(supervision_views),
        "optimizer_steps_per_s": steps_per_s,
        "supervision_views_per_s": views_per_s,
        "optimizer_steps_per_s_cumulative": optimizer_steps / cumulative_active_s,
        "supervision_views_per_s_cumulative": supervision_views / cumulative_active_s,
        "queue_depth": None if queue_depth is None else int(queue_depth),
        "source_offered": None if source_offered is None else int(source_offered),
        "source_submitted": None if source_submitted is None else int(source_submitted),
        "transport_dropped": None if transport_dropped is None else int(transport_dropped),
    }
    try:
        path = Path(model_path) / THROUGHPUT_TRACE_FILENAME
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return


def plot_causal_throughput(model_path: str | Path) -> Path | None:
    """Plot optimizer-steps/s and supervision-views/s over elapsed wall time.

    Two stacked panels (optimizer throughput, data/supervision throughput),
    each showing both the per-interval instantaneous rate and the
    cumulative-average rate for reference against the printed log line.

    Returns ``None`` when the trace file is absent or carries no usable rows
    -- a true no-op, matching the sibling plots' convention (e.g. any run
    predating this telemetry, or a non-paced/no-holdout run where the
    tick-clock observer is never wired).
    """

    root = Path(model_path)
    trace = root / THROUGHPUT_TRACE_FILENAME
    if not trace.is_file():
        return None
    rows = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    if not rows:
        return None

    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: float(row["elapsed_s"]))
    elapsed_s = np.asarray([float(row["elapsed_s"]) for row in rows])

    def _series(key: str) -> np.ndarray:
        return np.asarray(
            [np.nan if row.get(key) is None else float(row[key]) for row in rows]
        )

    steps_per_s = _series("optimizer_steps_per_s")
    views_per_s = _series("supervision_views_per_s")
    steps_cumulative = _series("optimizer_steps_per_s_cumulative")
    views_cumulative = _series("supervision_views_per_s_cumulative")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), dpi=100, sharex=True)

    top.plot(
        elapsed_s, steps_per_s, color="tab:blue", linewidth=1.5,
        label="instantaneous (per-interval delta)",
    )
    top.plot(
        elapsed_s, steps_cumulative, color="tab:blue", alpha=0.4, linestyle="--",
        linewidth=1.2, label="cumulative average",
    )
    top.set_ylabel("optimizer steps/s")
    top.set_title("Optimizer throughput over time")
    top.grid(True, alpha=0.3)
    top.legend(fontsize=8, loc="best")

    bottom.plot(
        elapsed_s, views_per_s, color="tab:orange", linewidth=1.5,
        label="instantaneous (per-interval delta)",
    )
    bottom.plot(
        elapsed_s, views_cumulative, color="tab:orange", alpha=0.4, linestyle="--",
        linewidth=1.2, label="cumulative average",
    )
    bottom.set_xlabel("elapsed wall time (s)")
    bottom.set_ylabel("supervision views/s")
    bottom.set_title("Supervision (data) throughput over time")
    bottom.grid(True, alpha=0.3)
    bottom.legend(fontsize=8, loc="best")

    fig.tight_layout()
    output = root / THROUGHPUT_PLOT_FILENAME
    fig.savefig(svg_output_path(output), dpi=100)
    plt.close(fig)
    return output
