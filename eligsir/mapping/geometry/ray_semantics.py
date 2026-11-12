"""Contracts and documented sensor semantics for tri-state ray evidence.

Plan section 13.3 ("Ray evidence model"). This module owns *shape and
policy*, not the actual per-pixel classification (see
:mod:`eligsir.mapping.geometry.ray_classifier` for that) nor the
cross-frame consensus bookkeeping (see
:mod:`eligsir.mapping.geometry.free_space_consensus`).

Sensor semantics (RGB-D, e.g. Orbbec Femto Mega/Bolt ToF)
-----------------------------------------------------------
A depth sensor reports, per pixel, either a finite positive range or an
"invalid" sentinel (``0``/``NaN``/``inf`` depending on driver). Historically
this repo's ``--streaming_carve_invalid_depth`` treated every invalid pixel
as "the sensor looked all the way to ``streaming_max_depth`` and found
nothing" — i.e. certified free space out to the max range. That assumption
is only sometimes true. Invalid returns are also produced by:

- glass / transmissive surfaces (the IR pulse passes through or scatters);
- specular surfaces (the IR pulse reflects away from the receiver);
- low-reflectivity / dark surfaces beyond the sensor's usable range;
- occlusion boundaries and multi-path interference;
- genuine "nothing there within range" (the one case the legacy carve
  code correctly modeled).

None of these are distinguishable from a *single* frame's depth mask alone.
The documented, safe sensor semantics this module encodes are:

1. **Valid ray traversal before a measured hit is real evidence.** If a
   pixel has a valid finite depth ``d``, every point strictly closer than
   ``d`` along that ray is a real, single-frame-sufficient free-space
   observation (the pulse traveled there and was not absorbed/reflected
   early) — this is the ``SURFACE_HIT`` rule in plan section 13.3, and
   matches what the existing (pre-W3-08) free-space/floater loss in
   ``eligsir/optimization/gaussian/objective.py`` already computes
   for pixels with valid sensor depth.
2. **An invalid return alone documents nothing about space beyond the
   sensor's usable range.** A single frame's invalid-depth pixel is
   ``UNKNOWN`` by construction — see
   :func:`eligsir.mapping.geometry.ray_classifier.classify_rays`.
   It can only be promoted to ``CERTIFIED_FREE_SPACE`` when a *different*
   frame's real, valid ray traversal (rule 1) already measured that same
   3-D region as free space, and did so from an adequately different
   viewpoint (baseline/pose-confidence requirement below) — this is
   genuine multi-view corroboration, not a relabelling of the single-frame
   mask under a new name.
3. **Baseline / pose-confidence requirement.** Two camera poses only a few
   millimetres apart trace almost the same ray and cannot independently
   corroborate anything (mirrors the existing
   ``streaming_floater_consensus_baseline_gate``/``consensus_last_voter_center``
   parallax gate in ``utils/streaming_depth_lock.py``, reused for the same
   reason here). A vote only counts toward ``CERTIFIED_FREE_SPACE`` when it
   comes from a pose whose confidence is adequate and whose camera center is
   at least ``min_baseline_m`` from the last accepted voter for that region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RayEvidenceKind(Enum):
    """Tri-state classification of what a ray/pixel's evidence supports.

    Plan section 13.3. Ordering is not meaningful (not an ``IntEnum``) —
    callers should not rely on numeric ordering/comparison.
    """

    SURFACE_HIT = auto()
    CERTIFIED_FREE_SPACE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class RaySemanticsPolicy:
    """Tunable thresholds for tri-state ray classification.

    All defaults match the existing, already-validated
    ``streaming_floater_consensus_*`` defaults in ``arguments/__init__.py``
    (this is deliberately the same policy, reused rather than reinvented —
    see the module docstring's "not a relabelling" note).
    """

    min_views: int = 3
    """Minimum number of distinct, baseline-gated corroborating views
    before an invalid-depth region may be certified free space."""

    min_baseline_m: float = 0.03
    """Minimum camera-center displacement (metres) between corroborating
    views for a vote to count (parallax/baseline gate)."""

    min_pose_confidence: float = 0.0
    """Minimum pose confidence (``InputEvidence.pose_confidence``, [0,1])
    required for a frame's votes to be accepted at all. ``0.0`` = no gate
    (matches current behaviour where pose confidence is not universally
    populated yet)."""

    voxel_size_m: float = 0.05
    """Voxel edge length (metres) used to key the free-space consensus
    hash. Coarser than the depth-insertion voxel size on purpose: free-space
    corroboration only needs to be spatially coarse enough to catch "this
    general region was seen through", not point-accurate."""

    max_depth_m: float = 5.0
    """Sensor documented max usable range (metres); mirrors
    ``--streaming_max_depth`` / the active camera profile. Rays are only
    ever classified up to this range."""

    surface_margin_m: float = 0.05
    """Margin (metres) kept before a measured hit when sampling
    "traversed-before-hit" free-space points, to avoid voting the surface
    voxel itself as free space."""


@dataclass
class ClassificationCounts:
    """Inspectable, per-classification-run counters (deliverable 2: the
    classification reason must be inspectable and counted, not just applied
    silently).

    ``unknown_reason_*`` breaks ``UNKNOWN`` down by *why* — this is what lets
    an evaluator/tensorboard scalar answer "how much of this frame is
    ordinary sensor drop-out vs. how much is still-uncorroborated
    invalid-depth that free-space carving would previously have erased".
    """

    surface_hit: int = 0
    certified_free_space: int = 0
    unknown: int = 0
    unknown_reason_invalid_depth_uncorroborated: int = 0
    unknown_reason_below_min_views: int = 0
    unknown_reason_baseline_gate_failed: int = 0
    unknown_reason_low_pose_confidence: int = 0

    @property
    def total(self) -> int:
        return self.surface_hit + self.certified_free_space + self.unknown

    def as_dict(self) -> dict:
        return {
            "surface_hit": self.surface_hit,
            "certified_free_space": self.certified_free_space,
            "unknown": self.unknown,
            "unknown_reason_invalid_depth_uncorroborated": self.unknown_reason_invalid_depth_uncorroborated,
            "unknown_reason_below_min_views": self.unknown_reason_below_min_views,
            "unknown_reason_baseline_gate_failed": self.unknown_reason_baseline_gate_failed,
            "unknown_reason_low_pose_confidence": self.unknown_reason_low_pose_confidence,
            "total": self.total,
        }

    def __add__(self, other: "ClassificationCounts") -> "ClassificationCounts":
        if not isinstance(other, ClassificationCounts):
            return NotImplemented
        return ClassificationCounts(
            surface_hit=self.surface_hit + other.surface_hit,
            certified_free_space=self.certified_free_space + other.certified_free_space,
            unknown=self.unknown + other.unknown,
            unknown_reason_invalid_depth_uncorroborated=(
                self.unknown_reason_invalid_depth_uncorroborated
                + other.unknown_reason_invalid_depth_uncorroborated
            ),
            unknown_reason_below_min_views=(
                self.unknown_reason_below_min_views + other.unknown_reason_below_min_views
            ),
            unknown_reason_baseline_gate_failed=(
                self.unknown_reason_baseline_gate_failed + other.unknown_reason_baseline_gate_failed
            ),
            unknown_reason_low_pose_confidence=(
                self.unknown_reason_low_pose_confidence + other.unknown_reason_low_pose_confidence
            ),
        )
