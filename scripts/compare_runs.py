#!/usr/bin/env python3
"""Agent-facing run comparison: one flat markdown + montage PNGs, no UI.

Combines the quantitative back-half-degradation guard, a cross-checkpoint quality
table, an optional param-diff vs a golden baseline (e.g. output/tum_ceiling), and
optional fixed-view render strips that make "detail lost over time" visible.

Usage:
    python scripts/compare_runs.py RUN [RUN ...] \
        [--golden output/tum_ceiling] \
        [--out compare.md] [--strip] [--cam-index 0] \
        [--keys mcmc_noise_stop_iter mcmc_stop_growth_iter iterations cap_max]

The render strips (--strip) need torch/CUDA (run inside the container); the
markdown table + degradation verdicts are pure-data and run anywhere.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eligsir.eval.compare import (  # noqa: E402
    _execution_semantic_differences,
    emit_markdown,
    fixed_view_strip,
)

# Params most relevant to the construction / back-half-degradation story.
DEFAULT_KEYS = [
    "mcmc_noise_stop_iter", "mcmc_stop_growth_iter", "iterations", "cap_max",
    "init_type", "streaming_replay", "online_mode", "noise_lr",
    "streaming_lifecycle_enabled", "streaming_mature_anchor_xyz_weight",
    "streaming_depth_loss_weight", "streaming_keyframe_coverage",
    "streaming_global_replay_ratio", "streaming_ingest_front_load_ratio",
    "streaming_spatial_window_enabled", "streaming_floater_consensus_enabled",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--golden", default=None, help="golden baseline run dir (param-diff)")
    ap.add_argument("--out", default="compare.md", help="output markdown path")
    ap.add_argument("--strip", action="store_true", help="render fixed-view checkpoint strips (GPU)")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS, help="cfg keys to diff vs golden")
    ap.add_argument("--allow-unmanaged", action="store_true", help="Allow unmanaged runs as golden baseline without error")
    ap.add_argument(
        "--allow-semantic-mismatch",
        action="store_true",
        help="Explicitly compare runs whose resolved execution axes differ",
    )
    args = ap.parse_args(argv)

    if args.golden:
        from eligsir.utils.run_summary import load_run_summary
        golden_summary = load_run_summary(args.golden)
        golden_evidence = golden_summary.get("evidence_class") if golden_summary else None
        if golden_evidence == "UNMANAGED" and not args.allow_unmanaged:
            raise ValueError(
                f"Golden baseline '{args.golden}' is an UNMANAGED run. "
                "Pass --allow-unmanaged to compare against a non-canonical baseline."
            )

    semantic_differences, semantic_warnings = _execution_semantic_differences(
        args.runs, reference_dir=args.golden
    )
    for warning in semantic_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    mismatch_detail = "; ".join(
        f"{item['axis']}: {item['reference']!r} != {item['candidate']!r}"
        for item in semantic_differences
    )
    unverifiable_detail = "; ".join(semantic_warnings)
    semantic_issue_detail = "; ".join(
        part for part in (mismatch_detail, unverifiable_detail) if part
    )
    if semantic_issue_detail:
        if not args.allow_semantic_mismatch:
            raise ValueError(
                "execution semantics mismatch or unverifiable evidence; "
                f"comparison refused: {semantic_issue_detail}. Pass "
                "--allow-semantic-mismatch only when these axes are the "
                "declared variable under test or legacy evidence is explicitly accepted."
            )
        print(
            "WARNING: --allow-semantic-mismatch set; comparing despite: "
            f"{semantic_issue_detail}",
            file=sys.stderr,
        )

    montages = []
    if args.strip:
        strip_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".",
                                 "compare_strips")
        for rd in args.runs:
            png = os.path.join(strip_dir, os.path.basename(rd.rstrip("/")) + "_strip.png")
            try:
                res = fixed_view_strip(rd, png, cam_index=args.cam_index)
                if res:
                    montages.append(os.path.relpath(res, os.path.dirname(os.path.abspath(args.out)) or "."))
            except Exception as e:
                print(f"  strip failed for {rd}: {e}", file=sys.stderr)

    md = emit_markdown(args.runs, golden_dir=args.golden, cfg_keys=args.keys,
                       montage_paths=montages or None)
    with open(args.out, "w") as f:
        f.write(md)
    print(f"wrote {args.out}")
    # Echo the greppable verdict lines so an agent reading stdout gets the headline.
    for line in md.splitlines():
        if "DEGRADED:" in line:
            print(line.lstrip("- "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
