"""Public configuration seam for optional anchored map deformation."""

from __future__ import annotations

from typing import Any


def normalize_map_deformation_mode(raw: Any) -> str:
    """Normalize the map-deformation mode without importing GPU code."""
    if isinstance(raw, bool):
        return "on" if raw else "off"
    mode = str(raw).strip().lower()
    if mode not in {"off", "on"}:
        raise ValueError(
            "--streaming_map_deformation must be one of ('off', 'on')"
        )
    return mode
