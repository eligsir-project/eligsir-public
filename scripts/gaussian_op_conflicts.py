#!/usr/bin/env python3
"""Issue #326 -- derive the conflict metrics + ranked conflict table from a trace.

    python3 scripts/gaussian_op_conflicts.py <run_dir_or_trace.jsonl> [-o out.json] [--md]

Reads the JSONL a traced run wrote (``<model_path>/gaussian_op_trace.jsonl``)
and emits #326's derived metrics. Stdlib + this repo's torch-free diagnostics
modules only, so it runs anywhere -- including the lean CPU CI image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eligsir.diagnostics.gaussian_op_trace import load_events  # noqa: E402
from eligsir.diagnostics.gaussian_op_conflicts import derive_all  # noqa: E402


def resolve_trace_path(target: str) -> str:
    if os.path.isdir(target):
        candidate = os.path.join(target, "gaussian_op_trace.jsonl")
        if not os.path.exists(candidate):
            raise SystemExit(
                f"no gaussian_op_trace.jsonl in {target}. A run only writes one when "
                "ELIGSIR_GAUSSIAN_OP_TRACE=1 was set (issue #326; default OFF)."
            )
        return candidate
    return target


def render_markdown(m: dict) -> str:
    lines = [
        "# Issue #326 -- runtime Gaussian operation conflict report",
        "",
        f"- events: **{m['n_events']}** ({m['n_uid_scoped_events']} uid-scoped, "
        f"{m['n_aggregate_events']} aggregate)",
        f"- iterations covered: `{m['iterations_covered']}`",
        f"- mechanisms observed: `{', '.join(m['mechanisms_seen'])}`",
        f"- operations observed: `{', '.join(m['operations_seen'])}`",
        "",
        "## Ranked conflict table",
        "",
        "| # | mechanism A | mechanism B | state | kind | n | magnitude | confirmed fighting | note |",
        "|--:|---|---|---|---|--:|--:|:-:|---|",
    ]
    for i, r in enumerate(m["ranked_conflict_table"], 1):
        lines.append(
            f"| {i} | `{r['mechanism_a']}` | `{r['mechanism_b']}` | {r['state']} | "
            f"{r['kind']} | {r['n_events']} | {r['magnitude']:.6g} | "
            f"{'YES' if r['confirmed_fighting'] else 'no'} | {r['note']} |"
        )
    if not m["ranked_conflict_table"]:
        lines.append("| - | - | - | - | - | - | - | - | *no conflicts observed in this trace* |")
    lines += [
        "",
        "## Headline metrics",
        "",
        f"- movement cancellation ratio: **{m['movement_cancellation']['ratio']:.4f}** "
        f"({m['movement_cancellation']['total_cancelled_magnitude']:.6g} of "
        f"{m['movement_cancellation']['total_applied_magnitude']:.6g} applied)",
        f"- same-UID opposing action rate: **{m['same_uid_opposing_actions']['rate']:.4f}** "
        f"({m['same_uid_opposing_actions']['n_opposing_cells']}/"
        f"{m['same_uid_opposing_actions']['n_multi_mechanism_cells']} multi-mechanism cells)",
        f"- grow-then-prune rate: **{m['grow_then_prune']['rate']:.4f}** "
        f"({m['grow_then_prune']['n_short_lived']} short-lived, "
        f"{m['grow_then_prune']['n_cross_mechanism']} cross-mechanism)",
        f"- relocate-then-restore rate: **{m['relocate_then_restore']['rate']:.4f}**",
        f"- topology mutation while scheduler gate off: "
        f"**{m['topology_mutation_while_gate_off']['n_while_gate_off']}** of "
        f"{m['topology_mutation_while_gate_off']['n_population_changing_events']}",
        f"- population change with NO topology_version advance (#543 finding 1): "
        f"**{m['population_change_without_topology_advance']['n_silent_population_changes']}** of "
        f"{m['population_change_without_topology_advance']['n_population_events_with_topology_evidence']}",
        f"- mutations on support-starved UIDs: "
        f"**{m['mutations_after_support_starvation']['n_on_starved']}**",
        "",
        "> " + m["evidence_class_caveat"],
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="run directory or gaussian_op_trace.jsonl path")
    ap.add_argument("-o", "--out", default=None, help="write JSON here (default: alongside the trace)")
    ap.add_argument("--md", action="store_true", help="also print a markdown report")
    ap.add_argument("--window-iters", type=int, default=500,
                    help="grow-then-prune window (default 500)")
    args = ap.parse_args(argv)

    path = resolve_trace_path(args.target)
    events = load_events(path)
    metrics = derive_all(events, window_iters=args.window_iters)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)),
                                   "gaussian_op_conflicts.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"[gaussian-op-conflicts] {len(events)} events -> {out}")
    if args.md:
        print()
        print(render_markdown(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
