"""Public configuration seam for optional tracker-discontinuity folding."""

from __future__ import annotations

from typing import Any


def normalize_tracker_fold_mode(raw: Any) -> str:
    """Normalize legacy bool and CLI values to the fold mode vocabulary."""
    if isinstance(raw, bool):
        return "events" if raw else "off"
    mode = str(raw).strip().lower()
    if mode == "on":
        mode = "events"
    if mode not in {"off", "events", "drift"}:
        raise ValueError(
            "--streaming_tracker_discontinuity_fold must be one of "
            "('off', 'events', 'drift', 'on')"
        )
    return mode
