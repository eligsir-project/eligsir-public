"""Per-alpha-bin D/ED residual diagnostics for the depth-loss training-time
instrumentation (2026-08-11, see
docs/architecture/2026-08-10-debt-controlled-sparse-assimilation-plan.md
section 4.5's revised experiment priority, step 0).

The frozen-checkpoint diagnostic (`scripts/diagnose_d_vs_ed_depth_bias.py`)
found almost no low-alpha population on final-checkpoint test-holdout
renders, but the D-vs-ED SDD plan's Task 5 smoke run found the raw
depth-loss magnitude swings materially by quantity/gate choice during live
training -- these are not necessarily in tension, since training-time camera
sampling during active map expansion may see a very different
alpha-coverage distribution than final converged test-holdout renders do.
This module answers that directly: bin ACTUAL training-camera pixels by
rendered alpha and report the D/ED residual against sensor depth per bin, on
the cameras the depth loss is actually being computed against, not a frozen
checkpoint's test views.

Logging-only: every function here operates on already-rendered, detached
tensors and has zero effect on training. Callers are responsible for gating
invocation behind a TB-logging cadence (this module does not do that
itself, matching the existing streaming/depth_loss_raw_magnitude pattern).
"""

from __future__ import annotations

import torch

DEFAULT_ALPHA_BIN_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-6)


def compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha, bin_edges=DEFAULT_ALPHA_BIN_EDGES):
    """Bin valid-sensor pixels by rendered alpha coverage and report, per
    bin, the pixel-count fraction plus mean-absolute D and ED residuals
    against sensor depth.

    All inputs are broadcastable tensors of the same pixel count (any shape
    that flattens consistently, e.g. [1, H, W] or [H, W]); everything is
    detached internally, but callers must still avoid calling this on the
    hot path with tensors that are about to be used for backward -- treat it
    as diagnostic-only, invoked at a coarse logging cadence.

    Returns a dict: {"bins": [{"range": (lo, hi), "frac": float, "n": int,
    "d_mae": float|None, "ed_mae": float|None}, ...], "n_valid_sensor": int}.
    `d_mae`/`ed_mae` are None for empty bins (no valid pixels fell in that
    alpha range this frame) -- callers must not coerce None to 0.0 when
    aggregating, since that would silently misrepresent "no data" as
    "zero residual".
    """
    rend_d = rend_d.detach().reshape(-1)
    rend_ed = rend_ed.detach().reshape(-1)
    sensor_d = sensor_d.detach().reshape(-1)
    alpha = alpha.detach().reshape(-1)

    valid = sensor_d > 0
    n_valid = int(valid.sum().item())

    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = valid & (alpha >= lo) & (alpha < hi)
        n = int(in_bin.sum().item())
        frac = (n / n_valid) if n_valid > 0 else 0.0
        if n > 0:
            d_mae = (rend_d[in_bin] - sensor_d[in_bin]).abs().mean().item()
            ed_mae = (rend_ed[in_bin] - sensor_d[in_bin]).abs().mean().item()
        else:
            d_mae = None
            ed_mae = None
        bins.append({
            "range": (float(lo), float(hi)),
            "frac": frac,
            "n": n,
            "d_mae": d_mae,
            "ed_mae": ed_mae,
        })

    return {"bins": bins, "n_valid_sensor": n_valid}


def log_alpha_bin_residuals_to_tb(tb_writer, stats, iteration, tag_prefix="streaming/depth_alpha_bin"):
    """Write `compute_alpha_bin_residuals`'s output to TensorBoard as one
    scalar triple (frac/d_mae/ed_mae) per bin, keyed by the bin's lower edge
    (e.g. "streaming/depth_alpha_bin/frac_0.10"). Skips d_mae/ed_mae for
    empty bins (None) rather than logging a misleading 0.0 -- an empty bin
    should read as a gap in the TB curve, not a data point.
    """
    tb_writer.add_scalar(f"{tag_prefix}/n_valid_sensor", stats["n_valid_sensor"], iteration)
    for b in stats["bins"]:
        lo = b["range"][0]
        tb_writer.add_scalar(f"{tag_prefix}/frac_{lo:.2f}", b["frac"], iteration)
        if b["d_mae"] is not None:
            tb_writer.add_scalar(f"{tag_prefix}/d_mae_{lo:.2f}", b["d_mae"], iteration)
        if b["ed_mae"] is not None:
            tb_writer.add_scalar(f"{tag_prefix}/ed_mae_{lo:.2f}", b["ed_mae"], iteration)
