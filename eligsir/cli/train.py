"""Package entry point for the root training script."""

from __future__ import annotations

import runpy


def main() -> int:
    """Execute the train module as a script entry point."""
    runpy.run_module("train", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
