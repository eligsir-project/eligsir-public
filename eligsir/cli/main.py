"""Unified public command line for mapping, rendering, and evaluation."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eligsir",
        description="EliGSiR RGB-D Gaussian mapping and post-hoc evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("map", "Map an RGB-D source into a Gaussian checkpoint."),
        ("render", "Render train/test views from a completed checkpoint."),
        ("evaluate", "Compute exact metrics for rendered score artifacts."),
    ):
        subparser = subparsers.add_parser(command, help=description, description=description)
        subparser.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="Arguments for the retained package stage (use --help on that stage).",
        )
    return parser


def _run_stage(module: str, argv: Sequence[str]) -> int:
    """Run a retained stage module with the stage arguments."""
    old_argv = sys.argv
    sys.argv = [f"eligsir {module}", *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the unified command and dispatch one real package stage."""
    parsed, remainder = _build_parser().parse_known_args(argv)
    stage_args = [*parsed.args, *remainder]
    modules = {
        "map": "train",
        "render": "render",
        "evaluate": "metrics",
    }
    if "--help" in stage_args:
        # The subparser owns this help, so this branch is reached only when
        # argparse's REMAINDER captures it after an explicit separator.
        _build_parser().parse_args([parsed.command, "--help"])
    return _run_stage(modules[parsed.command], stage_args)


if __name__ == "__main__":
    raise SystemExit(main())
