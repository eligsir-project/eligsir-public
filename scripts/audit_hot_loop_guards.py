#!/usr/bin/env python3
"""Guard-resolution scan of the streaming main loop (issue #279, Phase 1).

Why this exists
---------------
Phase 0 (``scripts/audit_dataflow_sync_sites.py``, PR #546) published an honest
*range* rather than a number: "between 32 and 421 of the 1114 sites are
per-iteration". It said exactly why -- its rules 1-3 are lexical and see one
function at a time, and its pass-2 reachability walk resolves callee names
without resolving imports or method receivers, so ``HOT_REACHABLE`` (389 sites)
over-marks by construction.

This scanner attacks the *lower* half of that range from a different angle. It
does not try to resolve the call graph at all. Instead it takes the single place
that unambiguously executes once per training iteration -- the
``for iteration in range(first_iter, opt.iterations + 1):`` loop in
``streaming_training()`` -- and, for every transfer/readback site *lexically*
inside it, reconstructs the **full chain of enclosing conditional guards** and
reports whether the site is reached on every iteration or only under a gate.

That is a strictly stronger statement than Phase 0's rule 2, which classified a
site ``AMORTIZED`` if *some* enclosing ``if`` matched a cadence marker. Here the
whole ancestor chain is recorded verbatim, so a reader can re-derive every
verdict, and sites gated by something that is *not* a cadence marker (a feature
flag, a data-dependent test) are separated out rather than collapsed into
``UNCLASSIFIED``.

Scope, stated honestly
----------------------
* **Lexical only, and deliberately so.** A site inside a helper function that the
  loop calls every iteration is NOT counted here. This scanner therefore measures
  "unconditional readbacks written directly in the loop body", a *floor* on the
  true per-iteration count, not the count itself. Same direction of conservatism
  as Phase 0's ``HOT``, but with full guard context attached.
* **A guard chain is evidence about reachability, not about cost.** A site under
  ``if iteration % scalar_log_interval == 0:`` still synchronizes -- once per
  interval. ``CADENCE_GATED`` means amortized, not free.
* **``for``/``while``/``with``/``try`` ancestors are recorded but only nested
  loops and ``except`` handlers count as gates** (a nested loop body needs a
  non-empty iterable; a handler body needs a raise). This keeps ``UNCONDITIONAL``
  a true floor.

Verdicts
--------
``UNCONDITIONAL``   no conditional ancestor between the site and the loop body:
                    executes on every iteration that reaches it.
``CADENCE_GATED``   some ancestor guard matches a cadence/logging/viewer marker
                    (``% <interval> == 0``, ``*_interval``, ``should_log``,
                    ``tb_writer``, ``_viewer``, ``testing_iterations``,
                    ``saving_iterations``, ``checkpoint_iterations``, ``debug``,
                    ``verbose``, ``profil``).
``FEATURE_GATED``   some ancestor guard reads a config flag (``args.``/``opt.``/
                    ``pipe.``/``getattr(args``) and no cadence marker matched.
``DATA_GATED``      conditional, but neither of the above -- a data- or
                    state-dependent test. Published as the residual bucket for
                    the same reason Phase 0 published ``UNCLASSIFIED``: it bounds
                    how much of the split is assertion rather than derivation.

Usage
-----
    python scripts/audit_hot_loop_guards.py
    python scripts/audit_hot_loop_guards.py --json out.json
    python scripts/audit_hot_loop_guards.py --verdict UNCONDITIONAL
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from typing import Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGET = os.path.join(REPO_ROOT, "train_streaming.py")
DEFAULT_LOOP_FUNCTION = "streaming_training"

D2H_METHODS = {"cpu", "numpy", "item", "tolist"}
H2D_METHODS = {"cuda", "pin_memory"}
SCALAR_CASTS = {"float", "int", "bool"}

CADENCE_MARKERS = (
    "should_log",
    "tb_writer",
    "_interval",
    "testing_iterations",
    "saving_iterations",
    "checkpoint_iterations",
    "_viewer",
    "viewer_name",
    "verbose",
    "debug",
    "profil",
)

FEATURE_MARKERS = ("args.", "opt.", "pipe.", "getattr(args", "getattr(opt")


def _guard_is_cadence(text: str) -> bool:
    low = text.lower()
    if "%" in low and ("== 0" in low or "==0" in low):
        return True
    return any(m in low for m in CADENCE_MARKERS)


def _guard_is_feature(text: str) -> bool:
    return any(m in text for m in FEATURE_MARKERS)


def _seg(source: str, src_lines: list[str], node: ast.AST) -> str:
    """One-line rendering of a node's source, for verbatim guard recording."""
    text = None
    try:
        text = ast.get_source_segment(source, node)
    except Exception:  # pragma: no cover - defensive
        text = None
    if not text:
        lineno = getattr(node, "lineno", 1)
        text = src_lines[lineno - 1] if 0 < lineno <= len(src_lines) else ""
    return " ".join(text.split())


def _classify_site(node: ast.AST) -> Optional[tuple[str, str]]:
    """Return ``(direction, label)`` for a transfer/readback site, else ``None``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        if name in D2H_METHODS:
            return ("D2H", f".{name}()")
        if name in H2D_METHODS:
            return ("H2D", f".{name}()")
        if name == "synchronize":
            return ("SYNC", "synchronize()")
        if name == "to":
            for kw in node.keywords:
                if kw.arg == "device":
                    return ("H2D", ".to(device=...)")
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    if "cuda" in a.value or "hip" in a.value:
                        return ("H2D", f'.to("{a.value}")')
                if isinstance(a, ast.Name) and "device" in a.id.lower():
                    return ("H2D", ".to(<device>)")
            return None
    for kw in node.keywords:
        if kw.arg == "non_blocking":
            return ("H2D", "non_blocking=")
    return None


class _LoopFinder(ast.NodeVisitor):
    """Locate the per-iteration ``for iteration in ...`` loop."""

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.loop: Optional[ast.For] = None
        self._in_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.func_name:
            self._in_target = True
            self.generic_visit(node)
            self._in_target = False
        else:
            self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if (
            self._in_target
            and self.loop is None
            and isinstance(node.target, ast.Name)
            and node.target.id == "iteration"
        ):
            self.loop = node
        self.generic_visit(node)


def scan(path: str, func_name: str = DEFAULT_LOOP_FUNCTION) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    src_lines = source.splitlines()
    tree = ast.parse(source)

    finder = _LoopFinder(func_name)
    finder.visit(tree)
    loop = finder.loop
    if loop is None:
        raise SystemExit(
            f"could not locate `for iteration in ...` inside {func_name}() in {path}"
        )

    sites: list[dict[str, Any]] = []
    nested_defs: list[dict[str, Any]] = []

    def seg(node: ast.AST) -> str:
        return _seg(source, src_lines, node)

    def walk(node: ast.AST, guards: list[dict[str, Any]]) -> None:
        # A nested function defined in the loop body does not execute at its
        # definition site; record it and do not attribute its body to the
        # iteration.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            nested_defs.append(
                {"line": node.lineno, "name": getattr(node, "name", "<lambda>")}
            )
            return

        hit = _classify_site(node)
        if hit is not None:
            direction, label = hit
            if not guards:
                verdict = "UNCONDITIONAL"
            elif any(_guard_is_cadence(g["test"]) for g in guards):
                verdict = "CADENCE_GATED"
            elif any(_guard_is_feature(g["test"]) for g in guards):
                verdict = "FEATURE_GATED"
            else:
                verdict = "DATA_GATED"
            sites.append(
                {
                    "line": node.lineno,
                    "direction": direction,
                    "label": label,
                    "verdict": verdict,
                    "source": seg(node)[:200],
                    "guards": list(guards),
                }
            )

        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.If):
                test_text = seg(node.test)
                guard = {"line": node.test.lineno, "test": test_text[:200]}
                if child is node.test:
                    walk(child, guards)
                elif child in node.body:
                    walk(child, guards + [guard])
                else:
                    walk(
                        child,
                        guards + [{"line": node.test.lineno,
                                   "test": f"not ({test_text})"[:200]}],
                    )
                continue
            if isinstance(node, ast.Try) and child in node.handlers:
                walk(child, guards + [{"line": child.lineno, "test": "<except handler>"}])
                continue
            if isinstance(node, (ast.For, ast.While)) and child in node.body:
                walk(
                    child,
                    guards + [{"line": node.lineno,
                               "test": f"<nested loop> {seg(node.iter) if isinstance(node, ast.For) else seg(node.test)}"[:160]}],
                )
                continue
            walk(child, guards)

    for stmt in loop.body:
        walk(stmt, [])

    by_verdict: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    for s in sites:
        by_verdict[s["verdict"]] = by_verdict.get(s["verdict"], 0) + 1
        by_direction[s["direction"]] = by_direction.get(s["direction"], 0) + 1

    return {
        "target": os.path.relpath(path, REPO_ROOT),
        "function": func_name,
        "loop_line": loop.lineno,
        "loop_end_line": getattr(loop, "end_lineno", None),
        "n_sites": len(sites),
        "by_verdict": by_verdict,
        "by_direction": by_direction,
        "nested_def_count": len(nested_defs),
        "sites": sites,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="issue #279 Phase 1 guard-resolution scan")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--function", default=DEFAULT_LOOP_FUNCTION)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--verdict", default=None, help="print only sites with this verdict")
    args = ap.parse_args(argv)

    result = scan(args.target, args.function)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")

    print(
        f"{result['target']}::{result['function']} "
        f"loop at line {result['loop_line']}-{result['loop_end_line']}"
    )
    print(f"  transfer/readback sites in loop body (lexical): {result['n_sites']}")
    for k in sorted(result["by_verdict"], key=lambda x: -result["by_verdict"][x]):
        print(f"    {k:<16} {result['by_verdict'][k]}")
    print(
        "  by direction: "
        + ", ".join(f"{k}={v}" for k, v in sorted(result["by_direction"].items()))
    )
    print(f"  nested defs skipped: {result['nested_def_count']}")

    if args.verdict:
        print()
        for s in result["sites"]:
            if s["verdict"] != args.verdict:
                continue
            print(f"  {result['target']}:{s['line']}  [{s['direction']}] {s['label']}")
            print(f"      {s['source']}")
            for g in s["guards"]:
                print(f"      guard@{g['line']}: {g['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
