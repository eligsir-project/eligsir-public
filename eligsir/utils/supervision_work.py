"""Issue #703: Supervision-work accounting and derived metrics.

Compute work-efficiency ratios and per-view exposure distributions from
rasterizer/supervision counters. These are torch-free utilities over plain
dicts/lists, suitable for report assembly and export.

No unbounded per-camera time series; all aggregates are bounded and
consumable by JSON/report serialization.
"""

from __future__ import annotations

from typing import Any


def supervision_views_per_optimization_step(
    optimization_step_count: int,
    supervision_view_count: int,
) -> float:
    """Average number of supervision views per optimizer step.

    Typical values:
    - B=1 unbatched: 1.0
    - B=4 sequential accumulation: 4.0
    - Future true camera batch: 4.0
    """
    if optimization_step_count == 0:
        return 0.0
    return float(supervision_view_count) / float(optimization_step_count)


def optimization_step_rate_hz(
    optimization_step_count: int,
    wall_time_s: float,
) -> float:
    """Return realized optimizer.step() calls per wall-clock second.

    Different from tqdm 'it/s' (outer-loop iteration rate). This is the true
    optimization throughput independent of loop structure. The denominator is
    the elapsed wall time of the enclosing mapping run/report snapshot.
    A zero denominator is intentionally reported as zero because no rate was
    measurable yet.
    """
    if wall_time_s <= 0.0:
        return 0.0
    return float(optimization_step_count) / float(wall_time_s)


def supervision_view_rate_hz(
    supervision_view_count: int,
    wall_time_s: float,
) -> float:
    """Return realized supervised camera views per wall-clock second.

    The numerator is the canonical completed B1/B4 supervision-view clock;
    true camera batches therefore contribute their realized camera count, not
    one count per optimizer transaction.  The denominator is explicitly the
    elapsed wall time of the enclosing mapping run/report snapshot.  A zero
    denominator is intentionally reported as zero because no rate was
    measurable yet.
    """
    if wall_time_s <= 0.0:
        return 0.0
    return float(supervision_view_count) / float(wall_time_s)


def rasterizer_calls_per_optimization_step(
    optimization_step_count: int,
    rasterizer_call_count: int,
) -> float:
    """Average rasterizer calls per optimizer step.

    Typical values:
    - B=1 unbatched: 1.0
    - B=4 sequential accumulation: 4.0 (one render per view)
    - Future true camera batch: 1.0 (one batched render)
    """
    if optimization_step_count == 0:
        return 0.0
    return float(rasterizer_call_count) / float(optimization_step_count)


def supervised_pixels_per_optimization_step(
    optimization_step_count: int,
    supervised_pixel_count: int,
) -> float:
    """Average supervised pixels per optimizer step.

    Use this to compare multi-resolution experiments: two runs at different
    resolutions but the same iteration budget will have different pixel counts,
    making pixel-work a more fair comparison than views/step alone.
    """
    if optimization_step_count == 0:
        return 0.0
    return float(supervised_pixel_count) / float(optimization_step_count)


def supervision_work_report_block(
    optimization_step_count: int,
    supervision_view_count: int,
    rasterizer_call_count: int,
    supervised_pixel_count: int,
    per_camera_supervision_counts: dict[Any, int] | None = None,
    wall_time_s: float | None = None,
    loss_objective_call_count: int | None = None,
    backward_call_count: int | None = None,
    renderer_camera_batch_size_sum: int | None = None,
    renderer_camera_batch_size_max: int | None = None,
    loss_views_per_call_sum: int | None = None,
    loss_views_per_call_max: int | None = None,
) -> dict[str, Any]:
    """Assemble ``report.json["supervision_work"]`` from work clocks and exposures.

    Returns an EMPTY dict when no optimizer step was ever taken: the three
    derived ratios are undefined at a zero denominator, and an empty (falsy)
    block makes ``write_post_training_report`` omit the key entirely rather
    than persist a block of meaningless zeros. When admitted-view counts are
    supplied, their bounded exposure distribution is retained even at zero
    optimizer steps so admitted-but-never-supervised views remain visible.

    Lives here rather than inline in ``train_streaming.py``'s
    ``_streaming_report_kwargs()`` deliberately: ``tests/
    test_issue646_report_kwargs_signature.py`` statically walks EVERY dict
    literal inside that builder and requires each key to be a real
    ``write_post_training_report`` parameter, so a nested literal there would
    (correctly) be read as seven unknown top-level report kwargs. Building the
    nested block behind a function call keeps the seam check meaningful --
    ``supervision_work`` is the one kwarg crossing it.

    Note the deliberate asymmetry between the two view-ish clocks:
    ``supervision_view_count`` counts ATTEMPTED/collected views, while
    ``supervised_pixel_count`` only accumulates for views whose loss was finite
    and actually backwarded. So ``supervised_pixels_per_step /
    supervision_views_per_step`` is NOT H*W whenever views were dropped -- that
    gap is signal, not a bug.
    """
    if optimization_step_count <= 0:
        return (
            per_view_exposure_distribution(per_camera_supervision_counts)
            if per_camera_supervision_counts is not None
            else {}
        )
    block = {
        "optimization_step_count": int(optimization_step_count),
        "supervision_view_count": int(supervision_view_count),
        "rasterizer_call_count": int(rasterizer_call_count),
        "supervised_pixel_count": int(supervised_pixel_count),
        "rasterizer_calls_per_step": rasterizer_calls_per_optimization_step(
            optimization_step_count, rasterizer_call_count),
        "realized_supervision_views_per_step": supervision_views_per_optimization_step(
            optimization_step_count, supervision_view_count),
        "supervised_pixels_per_step": supervised_pixels_per_optimization_step(
            optimization_step_count, supervised_pixel_count),
        "supervision_views_per_step": supervision_views_per_optimization_step(
            optimization_step_count, supervision_view_count),
    }
    if wall_time_s is not None:
        block["optimization_step_rate_hz"] = optimization_step_rate_hz(
            optimization_step_count, float(wall_time_s)
        )
        block["supervision_view_rate_hz"] = supervision_view_rate_hz(
            supervision_view_count, float(wall_time_s)
        )
    # Issue #703: execution-shape counters and derived ratios
    if loss_objective_call_count is not None and optimization_step_count > 0:
        block["loss_objective_call_count"] = int(loss_objective_call_count)
        block["loss_objective_calls_per_step"] = float(loss_objective_call_count) / float(optimization_step_count)
    if backward_call_count is not None and optimization_step_count > 0:
        block["backward_call_count"] = int(backward_call_count)
        block["backward_calls_per_step"] = float(backward_call_count) / float(optimization_step_count)
    if renderer_camera_batch_size_max is not None and rasterizer_call_count > 0:
        block["renderer_camera_batch_size_max"] = int(renderer_camera_batch_size_max)
        if renderer_camera_batch_size_sum is not None:
            block["renderer_camera_batch_size_mean"] = float(renderer_camera_batch_size_sum) / float(rasterizer_call_count)
    if loss_views_per_call_max is not None and loss_objective_call_count and loss_objective_call_count > 0:
        block["loss_views_per_call_max"] = int(loss_views_per_call_max)
        if loss_views_per_call_sum is not None:
            block["loss_views_per_call_mean"] = float(loss_views_per_call_sum) / float(loss_objective_call_count)
    if per_camera_supervision_counts is not None:
        block.update(per_view_exposure_distribution(per_camera_supervision_counts))
    return block


def execution_summary_lines(
    clocks,
    requested_batch_size: int | None = None,
    wall_time_s: float | None = None,
) -> list[str]:
    """Generate [execution-summary] block lines from SupervisionWorkClocks.

    Produces a compact summary of realized execution shape: requested vs realized
    batch sizes, renderer camera batching, rasterizer calls/step, etc.

    Args:
        clocks: A SupervisionWorkClocks instance (torch-free).
        requested_batch_size: Requested B from CLI/config. None if unavailable.
        wall_time_s: Wall-clock elapsed time in seconds.

    Returns:
        A list of formatted strings, one line per field, suitable for printing
        as the body of an [execution-summary] block.
    """
    lines = []

    # Core counts
    if clocks.optimizer_step_count > 0:
        lines.append(f"  optimization_steps: {clocks.optimizer_step_count}")
        lines.append(f"  supervision_views: {clocks.supervision_view_count}")
        lines.append(f"  rasterizer_calls: {clocks.rasterizer_call_count}")
        lines.append(f"  supervised_pixels: {clocks.supervised_pixel_count}")

        # Batch shape
        realized_views_per_step = (
            float(clocks.supervision_view_count) / float(clocks.optimizer_step_count)
        )
        realized_raster_per_step = (
            float(clocks.rasterizer_call_count) / float(clocks.optimizer_step_count)
        )
        lines.append("")
        lines.append("  Realized execution shape:")
        lines.append(f"    requested_batch_size: {requested_batch_size if requested_batch_size is not None else 'unknown'}")
        lines.append(f"    realized_views_per_step: {realized_views_per_step:.2f}")
        lines.append(f"    realized_rasterizer_calls_per_step: {realized_raster_per_step:.2f}")
        if clocks.rasterizer_call_count > 0:
            mean_batch_size = (
                float(clocks.renderer_camera_batch_size_sum) / float(clocks.rasterizer_call_count)
            )
            max_batch_size = clocks.renderer_camera_batch_size_max
            lines.append(f"    renderer_camera_batch_size_mean: {mean_batch_size:.2f}")
            lines.append(f"    renderer_camera_batch_size_max: {max_batch_size}")
        if clocks.loss_objective_call_count > 0:
            loss_calls_per_step = (
                float(clocks.loss_objective_call_count) / float(clocks.optimizer_step_count)
            )
            backward_calls_per_step = (
                float(clocks.backward_call_count) / float(clocks.optimizer_step_count)
            )
            lines.append(f"    loss_objective_calls_per_step: {loss_calls_per_step:.2f}")
            lines.append(f"    backward_calls_per_step: {backward_calls_per_step:.2f}")

        # Rates
        if wall_time_s is not None and wall_time_s > 0:
            lines.append("")
            lines.append("  Realized rates:")
            opt_step_hz = float(clocks.optimizer_step_count) / float(wall_time_s)
            sup_view_hz = float(clocks.supervision_view_count) / float(wall_time_s)
            lines.append(f"    optimization_step_rate_hz: {opt_step_hz:.3f}")
            lines.append(f"    supervision_view_rate_hz: {sup_view_hz:.3f}")
    else:
        lines.append("  (no optimization steps observed)")

    return lines


def per_view_exposure_distribution(
    per_camera_supervision_counts: dict[Any, int],
) -> dict[str, Any]:
    """Bounded aggregate exposure statistics across all training cameras.

    Args:
        per_camera_supervision_counts: dict mapping camera/view identity to
            the number of times that view was successfully supervised during
            training.

    Returns:
        A dict with these keys (all int, scaled in "exposures"):
        - supervision_exposures_per_view_mean
        - supervision_exposures_per_view_min
        - supervision_exposures_per_view_p10
        - supervision_exposures_per_view_p50 (median)
        - supervision_exposures_per_view_p90
        - supervision_exposures_per_view_max
        - never_supervised_view_count (views with 0 supervisions)

    Empty input (no cameras ever supervised) returns all zeros except count.
    """
    if not per_camera_supervision_counts:
        return {
            "supervision_exposures_per_view_mean": 0,
            "supervision_exposures_per_view_min": 0,
            "supervision_exposures_per_view_p10": 0,
            "supervision_exposures_per_view_p50": 0,
            "supervision_exposures_per_view_p90": 0,
            "supervision_exposures_per_view_max": 0,
            "never_supervised_view_count": 0,
        }

    counts = sorted(per_camera_supervision_counts.values())
    n = len(counts)
    mean = sum(counts) / n if n > 0 else 0
    never_supervised = sum(1 for c in counts if c == 0)

    # Percentile computation: (n-1) * percentile / 100, linear interpolation.
    def percentile(data: list, p: float) -> int:
        """Compute percentile using numpy-style linear interpolation."""
        if not data:
            return 0
        idx = (len(data) - 1) * p / 100.0
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, len(data) - 1)
        if lower_idx == upper_idx:
            return int(data[lower_idx])
        frac = idx - lower_idx
        return int(
            data[lower_idx] * (1 - frac) + data[upper_idx] * frac
        )

    return {
        "supervision_exposures_per_view_mean": int(round(mean)),
        "supervision_exposures_per_view_min": counts[0],
        "supervision_exposures_per_view_p10": percentile(counts, 10),
        "supervision_exposures_per_view_p50": percentile(counts, 50),
        "supervision_exposures_per_view_p90": percentile(counts, 90),
        "supervision_exposures_per_view_max": counts[-1],
        "never_supervised_view_count": never_supervised,
    }
