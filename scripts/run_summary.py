#!/usr/bin/env python3
"""CLI over ``utils.run_summary``: one or many run directories -> one
consolidated ``RUN_SUMMARY.md`` per run directory (or stdout).

**Partial delivery of issue #699** (the "consolidated run summary" slice
only -- see ``docs/architecture/2026-08-15-issue699-consolidated-run-summary.md``
for the full #699 scope, including what this PR does NOT deliver: the
orchestrator, the local-quality/causal-metric generation, and automatic
wiring into the post-training pipeline).

Usage:
    python scripts/run_summary.py output/some_run
    python scripts/run_summary.py output/run_a output/run_b --stdout
    python scripts/run_summary.py output/some_run --json

Read-only over every path except its own output: with ``--stdout`` (or
``--json``) nothing is written at all; otherwise this writes exactly one
file, ``<run_dir>/RUN_SUMMARY.md``, and touches nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _run_one(run_dir: str, *, write: bool, as_json: bool) -> int:
    from eligsir.utils.run_summary import build_run_summary, render_markdown

    summary = build_run_summary(run_dir)

    if as_json:
        print(json.dumps(summary.to_dict(), indent=2, default=str))
        return 0

    md = render_markdown(summary)
    if write:
        out_path = os.path.join(run_dir, "RUN_SUMMARY.md")
        with open(out_path, "w") as f:
            f.write(md)
        print(f"[run_summary] wrote {out_path} (overall verdict: {summary.overall['verdict']})")
    else:
        print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run_dirs", nargs="+", help="One or more run directories.")
    ap.add_argument(
        "--stdout", action="store_true",
        help="Print the Markdown summary to stdout instead of writing RUN_SUMMARY.md.",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Print the structured summary as JSON instead of Markdown (implies --stdout).",
    )
    args = ap.parse_args(argv)

    write = not (args.stdout or args.json)
    status = 0
    for run_dir in args.run_dirs:
        if len(args.run_dirs) > 1 and not (args.stdout or args.json):
            print(f"=== {run_dir} ===")
        status |= _run_one(run_dir, write=write, as_json=args.json)
    return status


if __name__ == "__main__":
    sys.exit(main())
