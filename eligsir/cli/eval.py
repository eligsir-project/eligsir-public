"""Package entry point for the root metrics script."""

from __future__ import annotations

import runpy


def main() -> int:
    """Package entry point for the metrics script.

    Returns:
        Exit code (0 on success).
    """
    runpy.run_module("metrics", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
