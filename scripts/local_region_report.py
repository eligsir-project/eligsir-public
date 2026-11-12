"""Level B region report: region x time forgetting heatmap + top-down health map.

Issue #358 Level B. Reads a run's Level A artifacts
(``local_quality_history.jsonl``, and ``local_quality_covisibility.json`` when
present), groups the immutable microcells into adaptive regions with
``utils.local_regions`` (geometry/topology only -- never PSNR), and writes:

    <out>/local_region_history.jsonl     per-tick derived region metrics
    <out>/local_region_lineage.json      split/merge lineage + policy + flags
    <out>/region_forgetting_heatmap.svg  region x time drop-from-peak, with
                                         split/merge markers
    <out>/region_health_map.svg          top-down cell map: region id | health

Usage::

    python scripts/local_region_report.py <run_dir> [-o <out_dir>] [--up-axis 1]

Read-only with respect to ``run_dir`` unless ``-o`` points at it.

Why a script and not a training-time hook: region curves are *derived* from
the per-cell history (see ``utils.local_regions``' module docstring), so the
whole layer is regenerable at any time and needs no call site in the render
path. That is also why Level B adds no training flag -- there is nothing to
turn on.

Loud-failure policy (matching Level A): a run whose region layer could not
measure anything renders an explicit "NOT MEASURED" panel and exits non-zero,
rather than an empty white PNG that looks like a healthy scene.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eligsir.plot_style import svg_output_path  # noqa: E402
from eligsir.utils.local_quality import cell_center, read_local_quality_history, \
    read_covisibility_snapshot  # noqa: E402
from eligsir.utils.local_regions import RegionPolicy, derive_region_history, \
    write_region_history  # noqa: E402
from eligsir.utils.local_health_state import (  # noqa: E402
    ADVERSE_STATES, HealthPolicy, cell_series, classify_all, health_records,
    region_series, severe_forgetting_marks, write_health_history,
    write_partition_events, write_partition_snapshot,
)
from eligsir.utils.local_health_plots import (  # noqa: E402
    plot_global_envelope, plot_heatmaps, plot_worst_k,
)

_NOT_ACTIVE = np.nan


def _infer_up_axis(centers: np.ndarray) -> int:
    """Smallest-extent axis of the observed cell centres.

    Only ever used for the top-down PROJECTION of the health map, never for
    region grouping (which requires an explicit ``--up-axis`` to use height at
    all). The inferred value is printed, never silently assumed.
    """
    extent = centers.max(axis=0) - centers.min(axis=0)
    return int(np.argmin(extent))


def _not_measured_plot(path: str, reason: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axis_off()
    ax.text(0.5, 0.6, "NOT MEASURED", ha="center", va="center",
            fontsize=28, color="crimson", weight="bold")
    ax.text(0.5, 0.35, reason, ha="center", va="center", fontsize=9, wrap=True)
    fig.tight_layout()
    fig.savefig(svg_output_path(path), dpi=100)
    plt.close(fig)


def plot_heatmap(derived: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = derived["ticks"]
    iters = [t["iteration"] for t in ticks]
    rids, order = set(), []
    for t in ticks:
        for r in t["regions"]:
            if r not in rids:
                rids.add(r)
                order.append(int(r))
    order.sort()
    mat = np.full((len(order), len(ticks)), _NOT_ACTIVE, dtype=np.float64)
    labels = {}
    for ci, t in enumerate(ticks):
        for rs, row in t["regions"].items():
            ri = order.index(int(rs))
            labels[int(rs)] = row["label"]
            if "drop_from_peak_db" in row:
                mat[ri, ci] = row["drop_from_peak_db"]

    fig, ax = plt.subplots(figsize=(max(8, len(ticks) * 0.35), max(3, len(order) * 0.5)))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="0.85")   # not active / not yet established
    vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0
    x_edges = np.arange(len(ticks) + 1) - 0.5
    y_edges = np.arange(len(order) + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(mat), cmap=cmap,
                       vmin=0.0, vmax=max(vmax, 0.5), shading="flat")
    ax.invert_yaxis()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels.get(r, f"R{r}") for r in order])
    step = max(1, len(ticks) // 12)
    ax.set_xticks(range(0, len(ticks), step))
    ax.set_xticklabels([str(iters[i]) for i in range(0, len(ticks), step)], rotation=45,
                       ha="right")
    ax.set_xlabel("iteration")
    ax.set_ylabel("region (lineage label)")
    ax.set_title("Region x time forgetting: drop from own established peak (dB)\n"
                 "grey = region not active / not yet established (NOT MEASURED, "
                 "not 'no forgetting')", fontsize=9)
    fig.colorbar(im, ax=ax, label="dB below own peak")

    # Lineage overlay: a topology change must never be mistakable for a
    # quality event, so mark every split/merge tick explicitly.
    for ci, t in enumerate(ticks):
        for ev in t["events"]:
            if ev["event"] not in ("split", "merge"):
                continue
            ax.axvline(ci - 0.5, color="tab:cyan", ls="--", lw=1.2)
            txt = (f"{ev['parent_label']}->{'+'.join(ev['child_labels'])}"
                   if ev["event"] == "split"
                   else f"{'+'.join(ev['parent_labels'])}->{ev['label']}")
            # Inside the axes (top edge is -0.5) so the label cannot collide
            # with the title.
            ax.text(ci - 0.4, -0.45, txt, rotation=90, fontsize=7, color="tab:cyan",
                    va="top", ha="left")
        if t.get("measurement_incomplete"):
            ax.text(ci, len(order) - 0.4, "!", color="crimson", fontsize=10,
                    ha="center", va="top")
    fig.tight_layout()
    fig.savefig(svg_output_path(path), dpi=110)
    plt.close(fig)


def plot_health_map(derived: dict, records, path: str, up_axis=None) -> int:
    """Top-down cell map: region membership | worst drop-from-peak per cell.

    Returns the projection axis actually used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_size = derived["cell_size_m"]
    final = derived["ticks"][-1]
    # cell -> region, from the FINAL partition; cell -> worst drop over the run,
    # straight from the Level A per-cell records (established cells only).
    lineage = {r["region_id"]: r for r in derived["lineage"]}
    cell_region, cell_label = {}, {}
    for rs in final["regions"]:
        rid = int(rs)
        for k in _members_of(derived, rid):
            cell_region[k] = rid
            cell_label[k] = lineage[rid]["label"]
    worst = {}
    for rec in records:
        for ks, cr in (rec.get("cells") or {}).items():
            if cr.get("established"):
                worst[int(ks)] = max(worst.get(int(ks), 0.0),
                                     float(cr.get("max_drop_from_peak_db", 0.0)))

    keys = sorted(cell_region)
    if not keys:
        _not_measured_plot(path, "no cells assigned to any region at the final tick")
        return -1
    centers = np.array([cell_center(k, cell_size) for k in keys], dtype=np.float64)
    ax_up = int(up_axis) if up_axis is not None else _infer_up_axis(centers)
    plane = [a for a in (0, 1, 2) if a != ax_up]

    # Several cells at different heights project onto the same column. Plot one
    # marker per column, taking the WORST drop in it -- overplotting would make
    # which cell you see depend on draw order, and a healthy cell drawn last
    # would hide a degrading one underneath it.
    col_worst, col_region = {}, {}
    for k in keys:
        c = cell_center(k, cell_size)
        xy = (round(float(c[plane[0]]), 6), round(float(c[plane[1]]), 6))
        w = worst.get(k, np.nan)
        prev = col_worst.get(xy, np.nan)
        if np.isfinite(w) and (not np.isfinite(prev) or w > prev):
            col_worst[xy] = w
            col_region[xy] = cell_region[k]
        elif xy not in col_worst:
            col_worst[xy] = prev if np.isfinite(prev) else np.nan
            col_region.setdefault(xy, cell_region[k])
    cols = sorted(col_worst)
    xs = np.array([c[0] for c in cols])
    ys = np.array([c[1] for c in cols])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rids = sorted({cell_region[k] for k in keys})
    colors = plt.get_cmap("tab20")
    for i, rid in enumerate(rids):
        m = np.array([col_region[c] == rid for c in cols])
        if not m.any():
            continue
        lab = next(cell_label[k] for k in keys if cell_region[k] == rid)
        axes[0].scatter(xs[m], ys[m], s=90, marker="s", color=colors(i % 20), label=lab)
    axes[0].set_title(f"Final region membership ({len(rids)} regions, "
                      f"{len(cols)} top-down columns)")
    axes[0].legend(fontsize=7, markerscale=0.5)

    vals = np.array([col_worst[c] for c in cols])
    sc = axes[1].scatter(xs, ys, s=90, marker="s", c=vals, cmap="magma_r",
                         vmin=0.0, vmax=max(1.0, float(np.nanmax(vals)) if
                                            np.isfinite(vals).any() else 1.0))
    nan = ~np.isfinite(vals)
    axes[1].scatter(xs[nan], ys[nan], s=90, marker="s", color="0.85")
    axes[1].set_title("Cell health: WORST max-drop-from-own-peak in each\n"
                      "top-down column (dB); grey = never established "
                      "(not measured)", fontsize=9)
    fig.colorbar(sc, ax=axes[1], label="dB")
    for a in axes:
        a.set_xlabel(f"world axis {plane[0]} (m)")
        a.set_ylabel(f"world axis {plane[1]} (m)")
        a.set_aspect("equal")
        a.grid(alpha=0.3)
    fig.suptitle(f"Top-down health map (projection drops axis {ax_up}"
                 f"{' [EXPLICIT --up-axis]' if up_axis is not None else ' [INFERRED: smallest extent]'})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(svg_output_path(path), dpi=110)
    plt.close(fig)
    return ax_up


def _members_of(derived: dict, rid: int):
    """Cell keys of a region at the final tick.

    Taken from the single partitioner run inside ``derive_region_history`` --
    deliberately NOT a second partitioner pass, which could diverge from the
    partition these ticks were derived against.
    """
    return derived.get("membership", {}).get(rid, [])


def build(run_dir: str, policy: RegionPolicy):
    records = read_local_quality_history(run_dir)
    cov, cov_cell_size = read_covisibility_snapshot(run_dir)
    if cov is not None and records:
        hist_cs = float(records[0].get("cell_size_m", 0.0))
        if cov_cell_size is not None and abs(float(cov_cell_size) - hist_cs) > 1e-9:
            raise ValueError(
                f"co-visibility snapshot was built at cell_size={cov_cell_size}m but "
                f"the history is at {hist_cs}m -- these are different partitions and "
                f"their cell keys are not interchangeable. Regenerate one of them."
            )
    derived = derive_region_history(records, policy=policy, covisibility=cov)
    return records, derived, cov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="directory holding local_quality_history.jsonl")
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--up-axis", type=int, default=None, choices=(0, 1, 2),
                    help="explicit world up axis; enables floor-based splitting AND "
                         "pins the top-down projection. Never inferred for grouping.")
    ap.add_argument("--split-min-cells", type=int, default=None)
    ap.add_argument("--split-max-bbox-diag-m", type=float, default=None)
    ap.add_argument("--split-hysteresis-ticks", type=int, default=None)
    ap.add_argument("--no-covisibility", action="store_true",
                    help="use 3D adjacency only (ablates the co-visibility edge)")
    # -- Level C (health state machine + issue #358 visualization items 1-3) --
    ap.add_argument("--rows", choices=("regions", "cells", "both"), default="both",
                    help="row granularity for the PRIMARY heatmaps. On a "
                         "single-room scene the region layer collapses to ~1 "
                         "measured row and carries almost no information, so "
                         "the cell granularity is where the signal is. Default "
                         "produces both.")
    ap.add_argument("--row-order", choices=("adjacency", "id", "lineage"),
                    default="adjacency",
                    help="row ordering. 'adjacency' = Morton code (cells) / "
                         "principal-axis projection of centroids (regions). "
                         "Never quality-derived.")
    ap.add_argument("--worst-k", type=int, default=6)
    ap.add_argument("--max-heatmap-rows", type=int, default=260)
    ap.add_argument("--stable-band-db", type=float, default=None)
    ap.add_argument("--degrading-db", type=float, default=None)
    ap.add_argument("--forgotten-db", type=float, default=None)
    ap.add_argument("--global-run-dir", default=None,
                    help="directory holding the ORIGINAL run's psnr_history.jsonl, "
                         "for the global curve on the envelope plot. Defaults to "
                         "run_dir; set it when the Level A history was written to "
                         "a separate output directory by "
                         "scripts/local_quality_from_run.py.")
    ap.add_argument("--no-health", action="store_true",
                    help="skip the Level C state machine and its plots")
    a = ap.parse_args()
    out_dir = a.out_dir or a.run_dir
    os.makedirs(out_dir, exist_ok=True)

    policy = RegionPolicy(up_axis=a.up_axis, use_covisibility=not a.no_covisibility)
    for src, dst in (("split_min_cells", "split_min_cells"),
                     ("split_max_bbox_diag_m", "split_max_bbox_diag_m"),
                     ("split_hysteresis_ticks", "split_hysteresis_ticks")):
        v = getattr(a, src)
        if v is not None:
            setattr(policy, dst, v)

    records, derived, cov = build(a.run_dir, policy)
    if not records:
        msg = (f"no local_quality_history.jsonl in {a.run_dir} -- Level A was not "
               f"enabled for this run (--streaming_local_quality_cells)")
        print(f"[local_regions] ERROR: {msg}", flush=True)
        _not_measured_plot(os.path.join(out_dir, "region_forgetting_heatmap.svg"), msg)
        return 2

    write_region_history(out_dir, derived)
    if not derived["regions_valid"]:
        msg = derived["ticks"][-1].get("invalid_reason") or "no regions could be formed"
        print(f"[local_regions] ERROR: {msg}", flush=True)
        _not_measured_plot(os.path.join(out_dir, "region_forgetting_heatmap.svg"), msg)
        _not_measured_plot(os.path.join(out_dir, "region_health_map.svg"), msg)
        return 3

    plot_heatmap(derived, os.path.join(out_dir, "region_forgetting_heatmap.svg"))
    used_axis = plot_health_map(derived, records,
                                os.path.join(out_dir, "region_health_map.svg"),
                                up_axis=a.up_axis)

    final = derived["ticks"][-1]
    n_split = sum(1 for t in derived["ticks"] for e in t["events"] if e["event"] == "split")
    n_merge = sum(1 for t in derived["ticks"] for e in t["events"] if e["event"] == "merge")
    print(f"[local_regions] ticks={len(derived['ticks'])} "
          f"cell_size={derived['cell_size_m']}m "
          f"covisibility_snapshot={'present' if derived['covisibility_available'] else 'ABSENT'}")
    print(f"[local_regions] evidence used: {final['evidence_sources_used']}")
    print(f"[local_regions] regions_active={final['n_regions_active']} "
          f"splits={n_split} merges={n_merge} "
          f"partition_version={final['partition_version']} "
          f"degenerate={final['degenerate_partition']}")
    print(f"[local_regions] top-down projection dropped axis {used_axis}")
    print(f"{'region':>10} {'cells':>6} {'est':>4} {'psnr':>7} {'peak':>7} "
          f"{'drop':>6} {'maxdrop':>8} {'diag_m':>7} {'split_state':>18}")
    for rs, row in sorted(final["regions"].items(), key=lambda kv: int(kv[0])):
        print(f"{row['label']:>10} {row['n_cells']:>6} {row['n_cells_established']:>4} "
              f"{(row['psnr_db'] or float('nan')):>7.2f} "
              f"{row.get('psnr_peak_db', float('nan')):>7.2f} "
              f"{row.get('drop_from_peak_db', float('nan')):>6.2f} "
              f"{row.get('max_drop_from_peak_db', float('nan')):>8.2f} "
              f"{row['bbox_diag_m']:>7.2f} {row['split_state']:>18}")
    # Loud-failure counters over the WHOLE history, not just the final tick --
    # a mid-run incomplete/unmeasured stretch is exactly the thing that must
    # not be invisible because the last tick happened to be clean.
    n_incomplete = sum(1 for t in derived["ticks"] if t["measurement_incomplete"])
    n_unmeasured = sum(1 for t in derived["ticks"] if not t["regions_valid"])
    n_degenerate = sum(1 for t in derived["ticks"] if t["degenerate_partition"])
    n_growth_conf = sum(1 for t in derived["ticks"]
                        for r in t["regions"].values() if r.get("growth_confounded"))
    if n_incomplete or n_unmeasured or n_degenerate:
        print(f"[local_regions] WARNING: {n_incomplete}/{len(derived['ticks'])} ticks "
              f"flagged measurement_incomplete by Level A, {n_unmeasured} ticks had NO "
              f"region evidence at all (NOT MEASURED, not 'no forgetting'), "
              f"{n_degenerate} ticks had a degenerate (mostly-singleton) partition")
    if derived["n_saturated_cell_records"]:
        print(f"[local_regions] NOTE: {derived['n_saturated_cell_records']} cell records "
              f"were at the 120dB PSNR clamp; their SSE could not be inverted exactly "
              f"and contributes the clamp value (region PSNR is a lower bound there)")
    if n_growth_conf:
        print(f"[local_regions] NOTE: {n_growth_conf} region-ticks are "
              f"growth_confounded (the established-cell pool grew since that "
              f"region's peak, so the drop mixes composition change with real change)")
    print(f"[local_regions] wrote {out_dir}/local_region_history.jsonl, "
          f"local_region_lineage.json, region_forgetting_heatmap.svg, "
          f"region_health_map.svg")

    if not a.no_health:
        rc = run_health_layer(a, records, derived, out_dir)
        if rc:
            return rc
    return 0


def _health_policy(a) -> HealthPolicy:
    pol = HealthPolicy()
    for src, dst in (("stable_band_db", "stable_band_db"),
                     ("degrading_db", "degrading_db"),
                     ("forgotten_db", "forgotten_db")):
        v = getattr(a, src)
        if v is not None:
            setattr(pol, dst, v)
    if not (pol.stable_band_db < pol.degrading_db < pol.forgotten_db):
        raise SystemExit(
            f"[local_health] band thresholds must satisfy stable < degrading < "
            f"forgotten, got {pol.stable_band_db}/{pol.degrading_db}/{pol.forgotten_db}"
        )
    return pol


def run_health_layer(a, records, derived, out_dir: str) -> int:
    """Level C: state machine over the derived series + issue #358 plots 1-3.

    Deliberately a post-processing pass, exactly like Level B: both region and
    cell curves are re-derived from the immutable per-cell records, so nothing
    here needs a call site in the render path.
    """
    pol = _health_policy(a)
    print(f"[local_health] bands: stable<={pol.stable_band_db}dB "
          f"degrading>={pol.degrading_db}dB forgotten>={pol.forgotten_db}dB "
          f"(issue #358's own >1/>3/>5 dB list)")

    wanted = ("regions", "cells") if a.rows == "both" else (a.rows,)
    all_records = []
    region_results = None
    for kind in wanted:
        if kind == "regions":
            series = region_series(derived)
            prefix, run_note = "region", None
        else:
            series = cell_series(records)
            prefix, run_note = "cell", None
        if not series:
            print(f"[local_health] no {kind} series to classify")
            continue
        results = classify_all(series, pol)
        if kind == "regions":
            region_results = results
        all_records.extend(health_records(results, series, pol))

        n_measured = sum(1 for r in results
                         if r.summary["n_ticks_measured_established"] > 0)
        adverse = [r for r in results if r.summary["state_final"] in ADVERSE_STATES]
        marks = severe_forgetting_marks(results, series, pol)
        n_conf = sum(1 for r in results if r.summary["n_ticks_confounded"] > 0)
        print(f"[local_health] {kind}: {len(results)} rows, {n_measured} ever "
              f"measured+established, {len(adverse)} adverse at the final tick, "
              f"{len(marks)} tick(s) with severe local forgetting, "
              f"{n_conf} growth-confounded rows (excluded from markers/worst-K)")
        # State census at the final tick -- a row that was never measured must
        # be visible as such, never absorbed into a healthy count.
        census = {}
        for r in results:
            census[r.summary["state_final"]] = census.get(r.summary["state_final"], 0) + 1
        print(f"[local_health] {kind}: final-tick states "
              + ", ".join(f"{k}={v}" for k, v in sorted(census.items())))

        note = (f"{os.path.basename(os.path.abspath(a.run_dir))[:52]}  |  "
                f"rows={kind}, {n_measured} measured, {len(marks)} severe tick(s)")
        paths = plot_heatmaps(results, series, out_dir, prefix, derived=derived,
                              row_order=a.row_order, max_rows=a.max_heatmap_rows,
                              policy=pol)
        gdir = a.global_run_dir or a.run_dir
        if not os.path.exists(os.path.join(gdir, "psnr_history.jsonl")):
            print(f"[local_health] NOTE: no psnr_history.jsonl under {gdir} -- the "
                  f"envelope plot will have NO global curve to compare against. "
                  f"Pass --global-run-dir <original run dir>.")
        p_env = plot_global_envelope(results, series, out_dir, prefix,
                                     run_dir=gdir, policy=pol, title_note=note)
        p_wk = plot_worst_k(results, series, out_dir, prefix, k=a.worst_k, policy=pol)
        for p in list(paths.values()) + [p_env, p_wk]:
            print(f"[local_health] wrote {p}")

    if all_records:
        p = write_health_history(out_dir, all_records)
        print(f"[local_health] wrote {p}")
    pe = write_partition_events(out_dir, derived)
    ps = write_partition_snapshot(out_dir, derived, region_results)
    print(f"[local_health] wrote {pe}")
    print(f"[local_health] wrote {ps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
