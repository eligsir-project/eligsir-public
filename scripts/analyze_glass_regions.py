"""Split an existing report.json's test-view PSNR into glass-partition vs.
non-glass populations, using source_index ranges already known from visual
inspection. No rendering, no training -- pure post-hoc analysis of data
utils/comparison_report.py already persisted.
"""
import argparse
import json
import os
import sys


def _in_any_range(idx, glass_ranges):
    return any(lo <= idx <= hi for lo, hi in glass_ranges)


def classify_glass_regions(test_per_view, glass_ranges, assumed_perfect_psnr=40.0):
    glass_psnrs = []
    nonglass_psnrs = []
    for view in test_per_view:
        idx = view.get("source_index")
        if idx is None:
            continue
        if _in_any_range(idx, glass_ranges):
            glass_psnrs.append(view["psnr"])
        else:
            nonglass_psnrs.append(view["psnr"])

    all_psnrs = glass_psnrs + nonglass_psnrs
    glass_mean = (sum(glass_psnrs) / len(glass_psnrs)) if glass_psnrs else None
    nonglass_mean = (sum(nonglass_psnrs) / len(nonglass_psnrs)) if nonglass_psnrs else None
    overall_mean = (sum(all_psnrs) / len(all_psnrs)) if all_psnrs else None

    if nonglass_psnrs:
        upper_bound = (
            (assumed_perfect_psnr * len(glass_psnrs) + sum(nonglass_psnrs))
            / (len(glass_psnrs) + len(nonglass_psnrs))
        )
    else:
        upper_bound = None

    return {
        "glass_n": len(glass_psnrs),
        "glass_psnr_mean": glass_mean,
        "nonglass_n": len(nonglass_psnrs),
        "nonglass_psnr_mean": nonglass_mean,
        "overall_n": len(all_psnrs),
        "overall_psnr_mean": overall_mean,
        "assumed_perfect_psnr": assumed_perfect_psnr,
        "upper_bound_psnr_if_glass_fixed": upper_bound,
    }


def _parse_range(spec):
    lo_str, hi_str = spec.split(":")
    return (int(lo_str), int(hi_str))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_path", help="Path to a report.json")
    parser.add_argument("--glass-range", action="append", required=True,
                         metavar="LO:HI",
                         help="Inclusive source_index range classified as glass, e.g. 1481:4533. "
                              "May be passed multiple times.")
    parser.add_argument("--assumed-perfect-psnr", type=float, default=40.0)
    parser.add_argument("--out", default=None,
                         help="Output path (default: glass_vs_nonglass_psnr.json next to report_path)")
    args = parser.parse_args(argv)

    with open(args.report_path) as f:
        report = json.load(f)

    if not report.get("eval_valid") or not report.get("n_test"):
        print(f"error: {args.report_path} is not a valid eval "
              f"(eval_valid={report.get('eval_valid')!r}, n_test={report.get('n_test')!r}) "
              "-- refusing to analyze, per the real-holdout Run Contract", file=sys.stderr)
        return 2

    glass_ranges = [_parse_range(r) for r in args.glass_range]
    result = classify_glass_regions(
        report.get("test_per_view", []), glass_ranges,
        assumed_perfect_psnr=args.assumed_perfect_psnr,
    )

    out_data = dict(result)
    out_data.update({
        "holdout_hash": report.get("eval_metadata", {}).get("holdout_hash"),
        "scene_source": report.get("scene_source"),
        "partition_max_source_index": report.get("partition_max_source_index"),
    })

    out_path = args.out or os.path.join(os.path.dirname(args.report_path) or ".",
                                        "glass_vs_nonglass_psnr.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    def _fmt(v):
        return f"{v:.2f}dB" if v is not None else "--"

    print(
        f"glass: n={result['glass_n']} mean={_fmt(result['glass_psnr_mean'])} | "
        f"non-glass: n={result['nonglass_n']} mean={_fmt(result['nonglass_psnr_mean'])} | "
        f"overall: n={result['overall_n']} mean={_fmt(result['overall_psnr_mean'])} | "
        f"upper_bound_if_glass_fixed={_fmt(result['upper_bound_psnr_if_glass_fixed'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
