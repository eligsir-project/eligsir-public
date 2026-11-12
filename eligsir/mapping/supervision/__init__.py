"""Spatially selective supervision primitives (issues #309/#310, #610/#611).

P0 tooling only: exact crop-camera geometry and a fixed supervision-tile
lattice with a cheap on-device evidence reduction. Nothing here selects,
ranks, promotes, schedules or weights anything — see
``docs/architecture/2026-08-29-tile-crop-supervision-implementation-plan.md``
for the phase boundaries and for why tile/crop selection is deliberately kept
*subordinate* to ``WorkBudgetOwner``'s resolution decision rather than becoming
a second actuator.

**Naming.** "tile" alone is already taken in this repo by gsplat's internal
16x16 rasterization tiles (``pipe.tile_size``,
``scheduler/tier_diagnostics.py``'s ``tile_count``/``isects_per_tile``). Every
symbol and telemetry key here says *supervision* tile or *crop*, never a bare
``tile``.
"""

from eligsir.mapping.supervision.crop_camera import (
    CropCamera,
    CropTransform,
    CropIntrinsics,
    crop_camera_from,
    crop_image,
    derive_crop_intrinsics,
    parent_intrinsics,
    project_point,
    project_to_crop_pixel,
)
from eligsir.mapping.supervision.tile_lattice import (
    MAX_LATTICE_DIM,
    SupervisionTileLattice,
    SupervisionTileEvidence,
    collect_tile_evidence,
    reduce_to_supervision_tiles,
)
from eligsir.mapping.supervision.frame_belief import (
    FrameModalityState,
    FrameSupervisionBelief,
    FrameSupervisionFact,
)

__all__ = [
    "CropCamera",
    "CropTransform",
    "CropIntrinsics",
    "crop_camera_from",
    "crop_image",
    "derive_crop_intrinsics",
    "parent_intrinsics",
    "project_point",
    "project_to_crop_pixel",
    "MAX_LATTICE_DIM",
    "SupervisionTileLattice",
    "SupervisionTileEvidence",
    "collect_tile_evidence",
    "reduce_to_supervision_tiles",
    "FrameModalityState",
    "FrameSupervisionBelief",
    "FrameSupervisionFact",
]
