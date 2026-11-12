"""Regression guard against silently-dropped streaming levers.

PR #20 ("package restructure") dropped ~20 ``streaming_*`` argument
declarations from ``arguments/__init__.py`` while leaving their
``getattr(args, "streaming_...", default)`` readers in place. The result
was "dark features": levers stuck at their default forever, with no CLI
or config switch to enable them.

This test asserts that every ``streaming_*`` name *read* anywhere in the
codebase has a corresponding *declaration* in one of the argument
ParamGroups, so a future restructure cannot silently drop a switch again.
"""

import re
from pathlib import Path

from eligsir.cli.parser import build_training_parser

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches reads of streaming_* args:
#   getattr(args, "streaming_foo", ...)   getattr(self.args, 'streaming_foo', ...)
#   args.streaming_foo                    self.args.streaming_foo
_READ_PATTERN = re.compile(
    r"""getattr\(\s*[A-Za-z_][\w.]*\s*,\s*["'](streaming_[A-Za-z0-9_]+)["']"""
    r"""|\bargs\.(streaming_[A-Za-z0-9_]+)"""
    r"""|self\.args\.(streaming_[A-Za-z0-9_]+)"""
)

def _declared_streaming_args() -> set[str]:
    """Read the canonical parser, including manually declared CLI switches."""
    return {action.dest for action in build_training_parser().parser._actions
            if action.dest.startswith("streaming_")}


def _referenced_streaming_args() -> dict[str, list[str]]:
    """Map streaming_* name -> list of files that read it (excluding tests/args)."""
    refs: dict[str, list[str]] = {}
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if (
            rel.startswith("tests/")
            or rel.startswith(".git/")
            or rel.startswith(".worktrees/")
            or "/worktrees/" in rel
            or rel == "arguments/__init__.py"
        ):
            continue
        text = path.read_text(errors="ignore")
        for match in _READ_PATTERN.finditer(text):
            name = next(g for g in match.groups() if g)
            refs.setdefault(name, [])
            if rel not in refs[name]:
                refs[name].append(rel)
    return refs


def test_every_read_streaming_arg_is_declared():
    declared = _declared_streaming_args()
    referenced = _referenced_streaming_args()

    missing = {
        name: files
        for name, files in referenced.items()
        if name not in declared
    }

    assert not missing, (
        "These streaming_* args are READ in code but have NO declaration in any "
        "canonical parser (dark features — unreachable via CLI/config):\n"
        + "\n".join(f"  {n}: read in {', '.join(f)}" for n, f in sorted(missing.items()))
    )
