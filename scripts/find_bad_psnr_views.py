#!/usr/bin/env python3
"""Identify held-out test views with poor PSNR from a saved report.json.

Reads report.json["test_per_view"] (already computed by
utils/comparison_report.py's write_post_training_report) and prints the
worst-scoring views by PSNR, optionally filtered to those below a threshold.
Intended as the first step of a two-pass "re-train bad views at native
resolution" workflow: the source_index values printed here feed
--streaming_full_res_frame_indices on a follow-up run.

Usage:
  python scripts/find_bad_psnr_views.py <report.json> [--threshold 20.0] [--top 20]
"""
import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="Path to report.json")
    ap.add_argument("--threshold", type=float, default=None,
                     help="Only list views with psnr below this value")
    ap.add_argument("--top", type=int, default=20,
                     help="Max number of worst views to print (default 20)")
    ap.add_argument("--indices-only", action="store_true",
                     help="Print only a comma-separated source_index list "
                          "(for feeding directly into --streaming_full_res_frame_indices)")
    args = ap.parse_args()

    with open(args.report) as f:
        d = json.load(f)

    per_view = d.get("test_per_view")
    if not per_view:
        print("No test_per_view data in this report.json — was it generated "
              "with an up-to-date write_post_training_report?", file=sys.stderr)
        sys.exit(1)

    rows = sorted(per_view, key=lambda r: r["psnr"])
    if args.threshold is not None:
        rows = [r for r in rows if r["psnr"] < args.threshold]
    rows = rows[: args.top]

    if args.indices_only:
        idxs = [str(r["source_index"]) for r in rows if r.get("source_index") is not None]
        print(",".join(idxs))
        return

    print(f"{'source_index':>12}  {'partition':>9}  {'psnr':>7}  name")
    for r in rows:
        print(f"{str(r.get('source_index')):>12}  {str(r.get('partition')):>9}  "
              f"{r['psnr']:7.2f}  {r.get('name', '')}")

    all_psnr = [r["psnr"] for r in per_view]
    print(f"\n{len(per_view)} total test views | "
          f"mean={sum(all_psnr)/len(all_psnr):.2f} min={min(all_psnr):.2f} max={max(all_psnr):.2f} | "
          f"{len(rows)} shown")


if __name__ == "__main__":
    main()
