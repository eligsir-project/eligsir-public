"""Tri-state ray semantics and geometry-reliability package (Wave 3, PR W3-08).

Implements plan section 13 ("PR W3-08: tri-state ray semantics and geometry
reliability"): replaces the unsafe assumption that invalid depth implies free
space with an explicit three-way classification per ray/pixel
(:class:`~eligsir.mapping.geometry.ray_semantics.RayEvidenceKind`):

- ``SURFACE_HIT`` — a real measured depth return.
- ``CERTIFIED_FREE_SPACE`` — genuinely corroborated empty space (requires
  multi-view consensus, not a single frame's local invalid-depth mask).
- ``UNKNOWN`` — everything else (ordinary invalid depth, uncertain
  transmissive/specular regions). No geometry insertion, no free-space
  carving, no depth-geometry gradients for ``UNKNOWN`` rays.

See ``docs/architecture/2026-08-08-tri-state-ray-semantics.md`` for the
design rationale and the CLAUDE.md
"``streaming_carve_invalid_depth`` vs. long-corridor scenes" section for the
production incident this package fixes.
"""

from __future__ import annotations

from eligsir.mapping.geometry.ray_semantics import (
    ClassificationCounts,
    RayEvidenceKind,
    RaySemanticsPolicy,
)

__all__ = [
    "ClassificationCounts",
    "RayEvidenceKind",
    "RaySemanticsPolicy",
]
