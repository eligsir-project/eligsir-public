#!/usr/bin/env python3
"""Enumerate ``getattr(args, "<flag>", <literal>)`` call sites repo-wide and
diff each fallback literal against the real argparse default.

Issue #368 (repo-wide follow-up to #363/#367): a stale ``getattr`` fallback
silently applies a different value than the parser's real default on any
path where the named attribute is genuinely absent from ``args`` -- a
config-constructed args object, a partially-populated namespace, a hand-built
test args, a restored (pre-flag) checkpoint's ``cfg_args``. This script is
the single, shared definition of "a call site" and "the real default" that
both a human triaging the corpus and
``tests/test_getattr_fallback_freshness.py``'s repo-wide guard use -- so
they can't quietly diverge from each other.

Ground truth: ``eligsir.cli.parser.build_training_parser()`` (the real,
full trainer parser -- covers every ``ParamGroup`` class AND every flag
declared directly in ``parser.py`` via ``parser.add_argument(...)``), not
``configs/profiles/defaults.reference.yml``. The generated YAML snapshot only
scans the four ``ParamGroup`` classes (see ``scripts/dump_default_config.py``)
and has a real, confirmed gap for parser.py-only flags (``--quiet``,
``--config``, ``--checkpoint_interval``, ``--streaming_early_stop_*``, etc.)
-- using it here would misreport ~40 genuine call sites as "flag not found"
instead of comparing them.

Usage:
    python3 scripts/audit_getattr_fallbacks.py            # human-readable report
    python3 scripts/audit_getattr_fallbacks.py --json      # machine-readable

Scans every ``.py`` file in the repo except ``tests/`` (test fixtures
legitimately hand-build partial ``args`` objects -- that's the point of a
unit test, not a drift bug) and this repo's git-ignored dirs.
"""
from __future__ import annotations

import argparse as _argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class GetattrSite:
    path: str  # repo-relative, POSIX separators
    lineno: int
    flag: str
    literal_repr: str  # repr() of the fallback literal (or "<non-literal>")
    is_literal: bool


def _literal_eval_or_none(node: ast.AST):
    try:
        return ast.literal_eval(node), True
    except Exception:
        return None, False


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.sites: list[GetattrSite] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            return
        if len(node.args) < 3:
            return
        obj = node.args[0]
        # Matches getattr(args, ...) and getattr(self.args, ...) / etc --
        # anything whose *terminal* name is "args".
        if isinstance(obj, ast.Name):
            argname = obj.id
        elif isinstance(obj, ast.Attribute):
            argname = obj.attr
        else:
            return
        if argname != "args":
            return
        flag_node = node.args[1]
        if not (isinstance(flag_node, ast.Constant) and isinstance(flag_node.value, str)):
            return
        flag = flag_node.value
        if flag.startswith("_"):
            # Underscore-prefixed names are runtime-assigned state stashed
            # directly on the args object (e.g. args._bubble_d_obs,
            # args._run_log_started) -- never an argparse flag, and the
            # getattr default IS the "not yet set" mechanism by design, not
            # a copy of a real default that can drift. Out of scope by rule.
            return
        lit_node = node.args[2]
        value, is_literal = _literal_eval_or_none(lit_node)
        literal_repr = repr(value) if is_literal else "<non-literal>"
        self.sites.append(GetattrSite(self.rel_path, node.lineno, flag, literal_repr, is_literal))


def find_all_sites(include_tests: bool = False) -> list[GetattrSite]:
    sites: list[GetattrSite] = []
    for f in sorted(_REPO_ROOT.rglob("*.py")):
        rel = f.relative_to(_REPO_ROOT).as_posix()
        if (
            "/.git/" in rel
            or rel.startswith(".git/")
            or rel.startswith(".claude")
            or rel.startswith(".worktrees/")
            or "/.worktrees/" in rel
            or rel.startswith(".venv")
            or rel.startswith(".test_venv")
        ):
            continue
        if not include_tests and (rel.startswith("tests/") or "/tests/" in rel):
            continue
        try:
            tree = ast.parse(f.read_text(), filename=rel)
        except SyntaxError:
            continue
        v = _Visitor(rel)
        v.visit(tree)
        sites.extend(v.sites)
    return sites


def real_defaults() -> dict:
    """{flag_name: real_default_value} from the live, full trainer parser --
    covers every ParamGroup class plus every parser.py-only flag."""
    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    return vars(bundle.parser.parse_args([]))


def _values_match(literal, real) -> bool:
    if isinstance(literal, bool) or isinstance(real, bool):
        return bool(literal) == bool(real)
    if isinstance(literal, (int, float)) and isinstance(real, (int, float)):
        return float(literal) == float(real)
    return literal == real


def diff_against_parser(sites: list[GetattrSite], defaults: dict):
    """Split sites into (mismatches, not_a_flag, nonliteral)."""
    mismatches = []
    not_a_flag = []
    nonliteral = []
    for site in sites:
        if not site.is_literal:
            nonliteral.append(site)
            continue
        if site.flag not in defaults:
            not_a_flag.append(site)
            continue
        real = defaults[site.flag]
        literal = ast.literal_eval(site.literal_repr)
        if not _values_match(literal, real):
            mismatches.append((site, real))
    return mismatches, not_a_flag, nonliteral


def main() -> int:
    p = _argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    ns = p.parse_args()

    sites = find_all_sites()
    defaults = real_defaults()
    mismatches, not_a_flag, nonliteral = diff_against_parser(sites, defaults)

    if ns.json:
        print(json.dumps({
            "total_sites": len(sites),
            "mismatches": [
                {"path": s.path, "line": s.lineno, "flag": s.flag,
                 "literal": s.literal_repr, "real": repr(real)}
                for s, real in sorted(mismatches, key=lambda t: (t[0].path, t[0].lineno))
            ],
            "not_a_flag": [
                {"path": s.path, "line": s.lineno, "flag": s.flag, "literal": s.literal_repr}
                for s in sorted(not_a_flag, key=lambda s: (s.path, s.lineno))
            ],
            "nonliteral_count": len(nonliteral),
        }, indent=2))
    else:
        print(f"Scanned {len(sites)} getattr(args, ...) call sites "
              f"(excluding tests/) against the live parser's {len(defaults)} real defaults.\n")
        print(f"=== {len(mismatches)} literal/default MISMATCHES ===")
        for s, real in sorted(mismatches, key=lambda t: (t[0].path, t[0].lineno)):
            print(f"  {s.path}:{s.lineno}: getattr(args, {s.flag!r}, {s.literal_repr})  real={real!r}")
        print(f"\n=== {len(not_a_flag)} sites whose flag is not a real argparse default ===")
        for s in sorted(not_a_flag, key=lambda s: (s.path, s.lineno)):
            print(f"  {s.path}:{s.lineno}: getattr(args, {s.flag!r}, {s.literal_repr})  -- no such flag")
        print(f"\n({len(nonliteral)} sites skipped: non-literal fallback expression)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
