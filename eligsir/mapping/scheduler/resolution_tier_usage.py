"""Per-transaction resolution-tier usage telemetry and its report heatmap.

Companion to ``eligsir/runtime/causal_evaluation.py``'s
``spatial_replay_focus.jsonl``/``plot_spatial_replay_focus`` pair, but for the
P0 single-owner resolution scheduler (``eligsir/mapping/scheduler/
tier_cost_model.py``, ``WorkBudgetOwner``): one JSON line per completed
mapping transaction records the ``resolution_tier`` /
``transaction_class`` (``MappingWorkPlan``, ``eligsir/contracts/
control.py``) that ``WorkBudgetOwner.resolve``/``resolve_idle_replay`` chose
for it. See GitHub issue #1250's second comment ("Requested addition:
resolution-tier usage heatmap").

As of 2026-08-29 no real GPU run has exercised the adaptive controller (every
campaign run to date pins ``constellation_policy=fixed``, so every recorded
transaction carries the same tier) -- see issue #1250's status check-in. The
heatmap below is therefore designed to activate automatically whenever real
multi-tier telemetry exists, and to render a clearly-labeled placeholder
(never raise, never guess) otherwise, so it is safe to leave wired in ahead
of that validation run.

``plot_resolution_tier_over_time`` (added for the ``causal_mapping``
standard-report throughput/tier plots) reads the same trace and renders a
single mean-ordinal-rank line instead of the fractional heatmap -- see its
own docstring for why a rank average rather than a pixel-count average, and
for the step-plot convention it follows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eligsir.plot_style import svg_output_path

USAGE_TRACE_FILENAME = "resolution_tier_usage.jsonl"
USAGE_HEATMAP_FILENAME = "resolution_tier_usage_over_time.svg"
TIER_OVER_TIME_FILENAME = "resolution_tier_over_time.svg"

# Column cap mirrors utils.causal_evaluation's spatial-focus heatmap: bound
# render cost/readability for arbitrarily long runs.
_MAX_HEATMAP_COLS = 60

# The canonical tier ordering (fine -> coarse) for the heatmap's y-axis, so
# "shift toward native" reads as "mass moving toward the top" regardless of
# which tiers a given run actually used.
_TIER_ORDER = ("native", "half", "quarter", "eighth")

# Ordinal rank (0=native ... 3=eighth) used by plot_resolution_tier_over_time
# below to average discrete tier labels into a single numeric line. Deliberately
# an evenly-spaced ordinal rank rather than ResolutionTier.downsample_factor's
# (1/2/4/8) pixel-count scale (eligsir/contracts/control.py): the
# downsample factor is exponential in tier coarseness, so an arithmetic mean
# of it would visually over-weight a few eighth-tier transactions relative to
# what "average tier used" actually means to a reader. Mirrors _TIER_ORDER so
# the two plots agree on direction (lower/finer = better).
_TIER_RANK = {tier: rank for rank, tier in enumerate(_TIER_ORDER)}


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


# Rows are batched in memory and flushed together rather than opening the
# trace file on every completed transaction (issue #1664): this sits inside
# the continuously-fed GPU submission loop, so a per-call
# open/write/close was structurally wrong there even though its per-call
# cost is small. A cadence flush keeps memory bounded for long runs without
# adding a worker thread -- the file is still only ever appended to from the
# single submitter that calls ``record_resolution_tier_usage``.
_FLUSH_EVERY_ROWS = 200


class _TierUsageBuffer:
    """Bounded in-memory row batch for one run's usage trace file.

    Best-effort and silent on write failure, matching
    ``record_resolution_tier_usage``'s pre-existing contract: this telemetry
    must never be able to break a training run. Unlike the decision trace
    (``JsonlDecisionTraceObserver``), a lost batch here is not a correctness
    problem -- it just thins one report's heatmap.
    """

    def __init__(self, path: Path, *, flush_every: int = _FLUSH_EVERY_ROWS) -> None:
        self._path = path
        self._flush_every = max(1, int(flush_every))
        self._rows: list[str] = []

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(json.dumps(row, sort_keys=True) + "\n")
        if len(self._rows) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        try:
            with self._path.open("a") as handle:
                handle.writelines(self._rows)
        except OSError:
            return
        self._rows.clear()


# Keyed by resolved model_path so concurrent/sequential runs against
# different output directories (e.g. in tests) never share a buffer.
_BUFFERS: dict[str, _TierUsageBuffer] = {}


def record_resolution_tier_usage(
    model_path: str | Path,
    *,
    iteration: int,
    tier: Any,
    transaction_class: Any = None,
) -> None:
    """Buffer one completed transaction's chosen tier for the run's trace.

    Best-effort and silent on failure (matching
    ``causal_evaluation._observe_training_spatial_replay_focus``'s own
    ``spatial_replay_focus.jsonl`` writer): telemetry must never be able to
    break a training run. The row is appended to a bounded in-memory buffer
    keyed by ``model_path`` and only actually written every
    ``_FLUSH_EVERY_ROWS`` rows, or when ``flush_resolution_tier_usage`` is
    called at run finalization -- never synchronously on this call.
    """

    tier_str = _coerce_str(tier)
    if tier_str is None:
        return
    row = {
        "iteration": int(iteration),
        "resolution_tier": tier_str,
        "transaction_class": _coerce_str(transaction_class),
    }
    key = str(Path(model_path))
    buffer = _BUFFERS.get(key)
    if buffer is None:
        buffer = _TierUsageBuffer(Path(model_path) / USAGE_TRACE_FILENAME)
        _BUFFERS[key] = buffer
    buffer.append(row)


def flush_resolution_tier_usage(model_path: str | Path) -> None:
    """Flush and drop this model path's buffered resolution-tier rows.

    Call at run finalization/close (``CanonicalMappingRuntime.close()``) so
    a normal shutdown never leaves buffered-but-unwritten rows behind. Same
    best-effort, silent-on-failure contract as ``record_resolution_tier_usage``
    -- a write failure here still must not raise into a shutdown path.
    """

    buffer = _BUFFERS.pop(str(Path(model_path)), None)
    if buffer is not None:
        buffer.flush()


def plot_resolution_tier_usage(model_path: str | Path) -> Path | None:
    """Render the tier-usage-over-time heatmap, or a clear no-data placeholder.

    Returns ``None`` (a true no-op) only when the trace file itself is
    absent -- e.g. every run predating this telemetry. When the trace
    exists but every transaction resolved to the same tier (today's real
    corpus: every campaign run pins ``constellation_policy=fixed``), a
    placeholder figure is written instead of a degenerate one-row heatmap,
    so the artifact itself documents why there is nothing to show.
    """

    root = Path(model_path)
    trace = root / USAGE_TRACE_FILENAME
    if not trace.is_file():
        return None
    rows = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row.get("resolution_tier")]
    output = root / USAGE_HEATMAP_FILENAME

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return _write_placeholder(
            output, "resolution_tier_usage.jsonl is present but has no rows"
        )

    rows = sorted(rows, key=lambda row: int(row["iteration"]))
    observed_tiers = sorted(
        {str(row["resolution_tier"]) for row in rows},
        key=lambda tier: _TIER_ORDER.index(tier) if tier in _TIER_ORDER else len(_TIER_ORDER),
    )

    if len(observed_tiers) < 2:
        only = observed_tiers[0] if observed_tiers else "unknown"
        return _write_placeholder(
            output,
            "No adaptive-tier data in this run\n"
            f"(every transaction resolved to a single tier: '{only}' -- "
            "likely constellation_policy=fixed)",
        )

    import numpy as np

    iterations = [int(row["iteration"]) for row in rows]
    n_cols = max(1, min(_MAX_HEATMAP_COLS, len(set(iterations))))
    lo, hi = float(min(iterations)), float(max(iterations))
    if n_cols == 1:
        bin_edges = np.asarray([lo, hi + 1.0])
    else:
        bin_edges = np.linspace(lo, hi, n_cols + 1)
        bin_edges[-1] += 1.0

    tier_index = {tier: i for i, tier in enumerate(observed_tiers)}
    counts = np.zeros((len(observed_tiers), n_cols))
    for row in rows:
        iteration = int(row["iteration"])
        col = int(np.searchsorted(bin_edges, float(iteration), side="right") - 1)
        col = min(max(col, 0), n_cols - 1)
        counts[tier_index[str(row["resolution_tier"])], col] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        col_totals = counts.sum(axis=0)
        fractions = np.where(col_totals > 0, counts / np.maximum(col_totals, 1), np.nan)

    fig, axis = plt.subplots(figsize=(11, 4.5))
    masked = np.ma.masked_invalid(fractions)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    mesh = axis.pcolormesh(
        bin_edges,
        np.arange(len(observed_tiers) + 1),
        masked,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    axis.invert_yaxis()
    axis.set_yticks(np.arange(len(observed_tiers)) + 0.5)
    axis.set_yticklabels(observed_tiers)
    axis.set_xlabel("optimizer iteration (binned)")
    axis.set_ylabel("resolution tier")
    axis.set_title(
        "Resolution-tier usage over time "
        "(fraction of transactions per iteration bin; top row = native)"
    )
    fig.colorbar(mesh, ax=axis, label="selection fraction")
    fig.tight_layout()
    fig.savefig(svg_output_path(output), dpi=150)
    plt.close(fig)
    return output


def plot_resolution_tier_over_time(model_path: str | Path) -> Path | None:
    """Plot the mean selected resolution tier per iteration bin, as a line.

    Companion to ``plot_resolution_tier_usage``'s per-bin fractional heatmap
    above: a single step line (the ordinal-rank mean per bin -- see
    ``_TIER_RANK``'s docstring for why rank rather than pixel count) for a
    quick "is the scheduler drifting toward coarser/finer supervision over
    the run" read, at a glance rather than from a heatmap. Reads the same
    ``resolution_tier_usage.jsonl`` trace; no new persistence.

    A step plot (``drawstyle="steps-mid"``, matching the existing
    ``utils/trajectory_quality.py`` per-segment-mean precedent) is used
    rather than a linearly-interpolated line: the underlying quantity is a
    discrete per-bin average, and interpolating between bin centers would
    imply a false in-between value the data never actually took.

    Returns ``None`` only when the trace file itself is absent (a true
    no-op, matching the sibling plots' convention). Unlike
    ``plot_resolution_tier_usage``, a single-tier corpus still renders here
    (a flat line at that tier's rank is itself the honest answer for
    "average tier used"), rather than a placeholder.
    """

    root = Path(model_path)
    trace = root / USAGE_TRACE_FILENAME
    if not trace.is_file():
        return None
    rows = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if str(row.get("resolution_tier")) in _TIER_RANK]
    output = root / TIER_OVER_TIME_FILENAME

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return _write_placeholder(
            output, "resolution_tier_usage.jsonl is present but has no rows"
        )

    import numpy as np

    rows = sorted(rows, key=lambda row: int(row["iteration"]))
    iterations = np.asarray([int(row["iteration"]) for row in rows], dtype=float)
    ranks = np.asarray(
        [_TIER_RANK[str(row["resolution_tier"])] for row in rows], dtype=float
    )

    n_bins = max(1, min(_MAX_HEATMAP_COLS, len(set(iterations.tolist()))))
    lo, hi = float(iterations.min()), float(iterations.max())
    if n_bins == 1:
        bin_edges = np.asarray([lo, hi + 1.0])
    else:
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        bin_edges[-1] += 1.0

    bin_centers: list[float] = []
    bin_means: list[float] = []
    for col in range(n_bins):
        lo_edge, hi_edge = bin_edges[col], bin_edges[col + 1]
        mask = (iterations >= lo_edge) & (iterations < hi_edge)
        if not mask.any():
            continue
        bin_centers.append(0.5 * (lo_edge + hi_edge))
        bin_means.append(float(ranks[mask].mean()))

    fig, axis = plt.subplots(figsize=(9, 4.5), dpi=100)
    axis.step(
        bin_centers, bin_means, where="mid", linewidth=2.0, color="tab:green",
        marker="o", markersize=3,
    )
    axis.set_xlabel("optimizer iteration (binned)")
    axis.set_ylabel("mean resolution tier")
    axis.set_yticks(range(len(_TIER_ORDER)))
    axis.set_yticklabels(_TIER_ORDER)
    # Top = native/finest, bottom = eighth/coarsest -- matches the heatmap's
    # inverted-y "shift toward native = mass moving up" convention above.
    axis.set_ylim(len(_TIER_ORDER) - 0.7, -0.3)
    axis.set_title(
        "Average selected resolution tier over time (top = native/finest)"
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(svg_output_path(output), dpi=100)
    plt.close(fig)
    return output


def _write_placeholder(output: Path, message: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 3))
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
        transform=axis.transAxes,
    )
    fig.tight_layout()
    fig.savefig(svg_output_path(output), dpi=150)
    plt.close(fig)
    return output
