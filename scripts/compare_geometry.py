#!/usr/bin/env python3
"""Tabulate scene-geometry metrics across runs for side-by-side comparison.

Reads each run's geometry metrics (written by utils/comparison_report into
``geometry_metrics.json`` and ``report.json["geometry"]``) and prints a
transposed table (one column per run). Comparability holds only when the runs
were evaluated on the same scene / test cameras / resolution.

Usage:
    python scripts/compare_geometry.py RUN_OR_JSON [RUN_OR_JSON ...]
    # RUN_OR_JSON may be:
    #   - a run dir            (output/spike_fix/s2_control_30k)
    #   - a comparison iter dir (.../comparison/iter_30000)
    #   - a geometry_metrics.json / report.json path
"""
import glob
import json
import os
import sys

# Headline rows (label -> dotted path into the geometry dict).
ROWS = [
    ("n_total",                 "shape.n_total"),
    ("n_live",                  "shape.n_live"),
    ("SV (surfel demand)",      "expressiveness.sv"),
    ("gauss/surfel",            "expressiveness.gaussians_per_surfel"),
    ("n_eff_frac",              "shape.n_eff_frac"),
    ("complexity dim",          "expressiveness.complexity_dim"),
    ("psnr/kG*",                "expressiveness.psnr_per_kgaussian"),
    ("sensor_noise_m",          "input_quality.sensor_noise_m"),
    ("tracking_rmse_m",         "input_quality.tracking_inlier_rmse_m"),
    ("tracking_fail_frac",      "input_quality.tracking_fail_frac"),
    ("surface_thickness_m",     "input_quality.surface_thickness_m"),
    ("dim_excess",              "input_quality.dim_excess"),
    ("quality_verdict",         "input_quality.quality_verdict"),
    ("needle frac s2/s1<0.3",   "shape.needle_frac_lt_0.3"),
    ("needle frac s2/s1<0.2",   "shape.needle_frac_lt_0.2"),
    ("needle massfrac<0.3",     "shape.needle_massfrac_lt_0.3"),
    ("needle s2/s1 median",     "shape.needle_s2_s1_median"),
    ("disk s3/s2 median",       "shape.disk_s3_s2_median"),
    ("flat frac s3/s2<0.2",     "shape.flat_frac_lt_0.2"),
    ("aniso s1/s3 p99",         "shape.aniso_s1_s3_p99"),
    ("large splat frac",        "shape.large_splat_frac"),
    ("radius p99 (m)",          "shape.radius_p99"),
    ("opacity mean",            "shape.opacity_mean"),
    ("normal agree mean(deg)",  "normal_agreement.normal_angle_mean_deg"),
    ("normal within 20deg",     "normal_agreement.normal_within_20deg"),
    ("depth MAE (m)",           "sensor_accuracy.depth_mae_m"),
    ("depth RMSE (m)",          "sensor_accuracy.depth_rmse_m"),
    ("depth delta<1.25",        "sensor_accuracy.delta_1.25"),
    ("sensor normal mean(deg)", "sensor_accuracy.sensor_normal_normal_angle_mean_deg"),
]


def _dig(d, dotted):
    # Geometry paths are exactly section.key; metric keys themselves contain dots
    # (e.g. "needle_frac_lt_0.3"), so split only on the FIRST dot.
    section, _, key = dotted.partition(".")
    sub = d.get(section)
    if not isinstance(sub, dict):
        return None
    return sub.get(key)


def _load_geometry(arg):
    """Resolve an arg to a geometry dict + a short label."""
    path = arg
    if os.path.isdir(arg):
        cand = os.path.join(arg, "geometry_metrics.json")
        if os.path.isfile(cand):
            path = cand
        else:
            # Search comparison/iter_* for the highest-iteration report.
            reps = sorted(glob.glob(os.path.join(arg, "**", "geometry_metrics.json"),
                                    recursive=True))
            reps += sorted(glob.glob(os.path.join(arg, "**", "report.json"),
                                     recursive=True))
            if not reps:
                return None, arg
            path = reps[-1]
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  SKIP {arg}: {e}", file=sys.stderr)
        return None, arg
    geom = data.get("geometry", data)  # report.json nests under "geometry"
    label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    return geom, label


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories or geometry_metrics.json / report.json paths")
    ap.add_argument("--canonical-only", action="store_true", help="Reject unmanaged runs from comparison")
    ap.add_argument("--allow-unmanaged", action="store_true", default=True, help="Allow unmanaged runs (default True unless --canonical-only)")
    args = ap.parse_args(argv[1:] if len(argv) > 1 else [])

    if not args.runs:
        ap.print_help()
        return 1

    cols = []
    for arg in args.runs:
        if args.canonical_only:
            from eligsir.utils.run_summary import load_run_summary
            run_sum = load_run_summary(arg) if os.path.isdir(arg) else None
            ev = run_sum.get("evidence_class") if run_sum else None
            if ev == "UNMANAGED":
                raise ValueError(f"Run '{arg}' is UNMANAGED. Canonical comparisons reject unmanaged runs.")
        geom, label = _load_geometry(arg)
        if geom is not None:
            cols.append((label[-18:], geom))
    if not cols:
        print("No geometry metrics found.", file=sys.stderr)
        return 1
    w = 20
    hdr = f"{'metric':28s}" + "".join(f"{lab:>{w}s}" for lab, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for label, dotted in ROWS:
        cells = []
        for _, geom in cols:
            v = _dig(geom, dotted)
            cells.append(f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}")
        print(f"{label:28s}" + "".join(cells))
    print("-" * len(hdr))
    print("gauss/surfel = n_live / SV (size/growth-invariant efficiency; ~1 ideal, >>1 redundant).")
    print("* psnr/kG valid only within an IDENTICAL scene (size-confounded); not cross-scene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
