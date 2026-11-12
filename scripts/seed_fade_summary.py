#!/usr/bin/env python3
"""Summarize the Orbbec seed-fade ablation: TensorBoard scalars -> CSV + curve stats.

For each run directory (containing events.out.tfevents.*), extract the seed-fade
timeseries (optimization/seed_psnr, streaming/seed/mean_opacity, streaming/seed/alive_count,
streaming/gaussians, streaming/frames_ingested) and report, per run:

  - the seed_psnr / seed_opacity MINIMUM during ingestion (the "implosion" depth),
  - the value at end (recovery),
  - the dip magnitude = max_before - min_during  (smaller = more stable = better),
  - whether the curve is "flat" (dip < tol) -> the success criterion.

Usage:
  python scripts/seed_fade_summary.py output/abl_A0 output/abl_A1 ... [--csv out.csv]
"""
import argparse
import csv
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Current tag, as actually emitted by train_streaming.py (tb_writer.add_scalar(...)).
SEED_PSNR_TAG = "optimization/seed_psnr"
# Legacy tag some pre-2026-08 runs may still carry (renamed away from this; see git log
# for train_streaming.py). Kept only as a fallback for reading old event files.
SEED_PSNR_TAG_LEGACY = "train/seed_psnr"

TAGS = [
    # Wave 1 (a5390a3, 2026-08-05) renamed train/seed_psnr ->
    # optimization/seed_psnr. This script kept the old name, so every
    # seed_psnr_* column read None on every run since -- fixed in #362
    # (see seed_psnr_series below) and independently found by the #282
    # audit. Both names are collected; the resolver picks whichever the
    # run actually has.
    SEED_PSNR_TAG,
    SEED_PSNR_TAG_LEGACY,
    # KNOWN-CONTAMINATED (issue #282): streaming/seed/* is masked by
    # `birth_frame == 0`, which append_running_state() also assigns to every
    # MCMC-grown Gaussian -- measured at 97-99.7% of the whole population on
    # real fr3 runs. The `stable` verdict below is derived from it and is
    # therefore a whole-population opacity statistic, not a seed-cohort one.
    # Runs from 2026-08-13 on also log streaming/seed_cohort/* (masked by
    # birth_iter == 0), which is the correct cohort.
    "streaming/seed/mean_opacity",
    "streaming/seed/alive_count",
    "streaming/seed_cohort/mean_opacity",
    "streaming/seed_cohort/alive_count",
    "streaming/gaussians",
    "streaming/frames_ingested",
    # DEAD TAG (issue #282 audit): nothing in train_streaming.py ever writes
    # a test-PSNR TB scalar, so test_psnr_last has always printed None. The
    # real per-checkpoint test-PSNR trajectory lives in the run's
    # psnr_history.jsonl. Wiring that in is deferred to #346 Phase 0
    # (mandatory forgetting statistics from the full quality trajectory),
    # which needs a lot more than a last-value column.
    "streaming_eval/test_psnr",
]

#: Seed-cohort tags in preference order. The `seed_cohort/*` pair (masked by
#: birth_iter == 0, added by #282) is the CORRECT cohort; the `seed/*` pair
#: (birth_frame == 0) is the contaminated legacy one kept only so pre-#282
#: event files still produce a column. Resolved by `_first_present`.
SEED_OPACITY_TAGS = ("streaming/seed_cohort/mean_opacity", "streaming/seed/mean_opacity")
SEED_COUNT_TAGS = ("streaming/seed_cohort/alive_count", "streaming/seed/alive_count")


def _first_present(data, tags):
    for t in tags:
        if data.get(t):
            return data[t]
    return []


def load_run(run_dir):
    fs = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not fs:
        return None
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    out = {}
    for t in TAGS:
        if t in avail:
            out[t] = [(s.step, s.value) for s in ea.Scalars(t)]
    return out


def seed_psnr_series(data, run_name):
    """Resolve the seed-PSNR series, preferring the current tag over the legacy one.

    Loud on total absence -- this column silently reading as all-None (instead of
    erroring or warning) was the actual defect that made this script useless for
    seed-fade timing evidence; see issue #346.
    """
    series = data.get(SEED_PSNR_TAG)
    if series:
        return series
    legacy = data.get(SEED_PSNR_TAG_LEGACY)
    if legacy:
        print(
            f"[warn] {run_name}: no '{SEED_PSNR_TAG}' scalar found; falling back to "
            f"legacy tag '{SEED_PSNR_TAG_LEGACY}' (pre-rename run). Verify this run "
            "predates the tag rename before trusting the numbers.",
            file=sys.stderr,
        )
        return legacy
    print(
        f"[warn] {run_name}: NO seed_psnr data under either '{SEED_PSNR_TAG}' or "
        f"'{SEED_PSNR_TAG_LEGACY}' -- seed_psnr_* columns for this run will be None. "
        "This usually means seed-fade tracking wasn't active for this run, not that "
        "the tag is merely misspelled -- double check.",
        file=sys.stderr,
    )
    return []


def curve_stats(series):
    """Given [(step,val)], return (first, min, argmin_step, last, dip=first-min)."""
    if not series:
        return None
    vals = [v for _, v in series]
    steps = [s for s, _ in series]
    first, last = vals[0], vals[-1]
    mn = min(vals)
    argmin = steps[vals.index(mn)]
    # peak BEFORE the dip (the seed is best early, before travel)
    peak = max(vals[: max(1, len(vals) // 3)])  # best in first third
    return dict(first=first, peak=peak, min=mn, argmin_step=argmin, last=last,
                dip=peak - mn, recovered=last - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default="")
    ap.add_argument("--opacity-flat-tol", type=float, default=0.05,
                    help="seed mean_opacity dip below this = 'flat' (stable)")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        name = os.path.basename(run.rstrip("/"))
        data = load_run(run)
        if data is None:
            print(f"[skip] {name}: no events file", file=sys.stderr)
            continue
        # seed_psnr keeps #362's dedicated resolver (it warns loudly on a
        # legacy-tag fallback and on total absence -- silence there was the
        # original defect); opacity/count use #282's cohort-preferring one.
        ps = curve_stats(seed_psnr_series(data, name))
        op = curve_stats(_first_present(data, SEED_OPACITY_TAGS))
        cn = curve_stats(_first_present(data, SEED_COUNT_TAGS))
        g = data.get("streaming/gaussians", [])
        test = data.get("streaming_eval/test_psnr", [])
        row = dict(
            run=name,
            seed_psnr_peak=round(ps["peak"], 2) if ps else None,
            seed_psnr_min=round(ps["min"], 2) if ps else None,
            seed_psnr_dip=round(ps["dip"], 2) if ps else None,
            seed_psnr_last=round(ps["last"], 2) if ps else None,
            seed_op_peak=round(op["peak"], 3) if op else None,
            seed_op_min=round(op["min"], 3) if op else None,
            seed_op_dip=round(op["dip"], 3) if op else None,
            seed_op_last=round(op["last"], 3) if op else None,
            seed_alive_min=int(cn["min"]) if cn else None,
            seed_alive_last=int(cn["last"]) if cn else None,
            n_gauss_last=int(g[-1][1]) if g else None,
            test_psnr_last=round(test[-1][1], 2) if test else None,
            stable=("YES" if (op and op["dip"] < args.opacity_flat_tol) else "no"),
        )
        rows.append(row)

    if not rows:
        print("no runs with data", file=sys.stderr)
        return 1

    cols = list(rows[0].keys())
    # pretty print
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
