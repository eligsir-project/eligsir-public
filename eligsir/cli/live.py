"""Package entry point for live/online training profiles.

This is currently a thin alias to the trainer entry point. It exists so
profiles and launch docs can target a stable command while the shared
MappingEngine work proceeds.
"""

from __future__ import annotations

from .train import main


if __name__ == "__main__":
    raise SystemExit(main())
