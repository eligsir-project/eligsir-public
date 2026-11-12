"""Cross-frame free-space voting/consensus voxel index.

This is the "multi-view consensus before penalizing rendered mass in an
invalid-depth region" building block CLAUDE.md's
"``streaming_carve_invalid_depth`` vs. long-corridor scenes" section
describes as missing (as of 2026-08-06): the repo already had a *different*
multi-view consensus mechanism (``streaming_floater_consensus_*`` in
``utils/streaming_depth_lock.py``), but that one operates over *existing
Gaussians* (per-Gaussian seen/support/conflict counters) — it has nothing to
say about a region of empty 3-D space where no Gaussian has been placed yet,
which is exactly the case a corridor-carving decision needs.

``FreeSpaceConsensusIndex`` fills that gap with a small voxel-hash store:

- ``vote_free`` records real, single-frame-sufficient free-space evidence
  (points strictly before a *measured* sensor hit — see
  :class:`eligsir.mapping.geometry.ray_semantics.RayEvidenceKind`'s
  ``SURFACE_HIT`` docstring) into a voxel grid, gated by a baseline/parallax
  requirement so near-duplicate viewpoints don't inflate the vote count.
- ``certified_mask`` answers, for an arbitrary batch of query points
  (e.g. the far/invalid-depth continuation of a *different* frame's ray),
  whether that region has already accumulated ``>= min_views`` distinct,
  adequately-separated corroborating votes — the only condition under which
  those points are allowed to become ``CERTIFIED_FREE_SPACE`` rather than
  ``UNKNOWN``.

Implementation notes
---------------------
Voxel keys are packed into a single ``int64`` (21 bits per axis, matching
the Morton/grid-hash convention already used by
``eligsir/mapping/stores/occupancy_index.py`` and
``scene/streaming_scene.py``'s occupancy hash) so the whole store is a plain
CPU dict of ``int -> _VoxelVote``; this keeps it dependency-light and easy to
unit test without a GPU. It is not built for millions of high-frequency
updates per second (unlike the render-space free-space loss it gates); it is
invoked once per admitted keyframe, at low resolution
(``voxel_size_m`` defaults to 5cm), which is the right cost/precision
trade-off for a *region-level* corroboration signal, not a per-Gaussian one.

**Scope note (honest limitation, not glossed over):** this index accumulates
votes by sampling several points along the traversed-before-hit segment of
each SURFACE_HIT ray (see ``ray_classifier.classify_rays``'s
``ray_samples_per_pixel``/``ray_sample_stride_px`` parameters), not a dense
per-voxel ray march, and not (an earlier version of this code) a single
point right before the hit -- a single point per ray was measured to almost
never corroborate anything on realistic (non-contrived) corridor geometry,
since it requires two different rays' single sample points to coincidentally
land in the same coarse voxel (see
``tests/test_ray_semantics_classification.py::test_realistic_corridor_requires_segment_sampling``).
This index does not perform true 3-D ray/ray intersection bookkeeping (e.g.
tracking exactly which ray each vote came from, or reconciling
partially-occluded traversals across dynamic scene changes) — that would
need a proper occupancy-mapping data structure (voxel hashing with
per-voxel hit/miss log-odds, à la OctoMap) which is out of this PR's scope.
What is implemented is real: a vote only exists because some frame's
genuine measured hit places that voxel strictly before the surface, and it
only counts toward certification when corroborated by another
adequately-separated view. That is sufficient to fix the documented floor2
bug (a single frame's *lack* of a hit can never certify anything) but is a
conservative under-approximation of "empty" for legitimately novel free
space nothing has yet looked all the way through — such regions correctly
stay ``UNKNOWN`` until a second view corroborates them, which is the
intended, safe behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_BITS = 21
_MASK = (1 << _BITS) - 1
_OFFSET = 1 << (_BITS - 1)  # allow negative voxel coords


def _voxel_keys(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Quantize ``[N,3]`` world points into packed int64 voxel keys."""
    if points.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=points.device)
    q = torch.floor(points.detach().double() / float(voxel_size)).to(torch.int64)
    q = q + _OFFSET
    q = q.clamp(0, _MASK)
    keys = (q[:, 0] << (2 * _BITS)) | (q[:, 1] << _BITS) | q[:, 2]
    return keys


@dataclass
class _VoxelVote:
    count: int = 0
    last_view_uid: int = -1
    voter_centers: Optional[list] = None  # list[tuple[float, float, float]]


class FreeSpaceConsensusIndex:
    """CPU voxel-hash store of cross-frame free-space corroboration votes.

    One instance persists for the lifetime of a streaming run (owned by the
    caller — e.g. attached to ``StreamingScene`` — not recreated per frame,
    or the whole point of "cross-frame" corroboration is lost).
    """

    def __init__(
        self,
        voxel_size_m: float = 0.05,
        min_baseline_m: float = 0.03,
    ) -> None:
        self.voxel_size_m = float(voxel_size_m)
        self.min_baseline_m = float(min_baseline_m)
        self._votes: dict[int, _VoxelVote] = {}
        # A view's traversed-before-hit ray points are a deterministic
        # function of that view's own (unchanging) sensor depth, so once a
        # view_uid has contributed its votes, every later call with the same
        # view_uid is a no-op by construction -- this set lets vote_free
        # skip the whole backproject-and-vote cost on repeat calls (the
        # streaming loop resamples the same keyframe across many training
        # iterations, so without this a view would otherwise redo full
        # per-pixel ray-segment voting every single iteration it's drawn).
        self._fully_voted_view_uids: set[int] = set()

    def __len__(self) -> int:
        return len(self._votes)

    def vote_free(
        self,
        points: torch.Tensor,
        *,
        view_uid: int,
        camera_center: torch.Tensor,
    ) -> int:
        """Record ``points`` (``[N,3]`` world-space, real measured-before-hit
        free space from a single frame) as one vote each, deduplicated by
        voxel and gated by the baseline/parallax requirement.

        Returns the number of voxels that received a *new* vote this call.
        """
        if points.numel() == 0:
            return 0
        if int(view_uid) in self._fully_voted_view_uids:
            return 0
        keys = _voxel_keys(points, self.voxel_size_m)
        unique_keys = torch.unique(keys).tolist()
        cam = tuple(float(x) for x in camera_center.detach().reshape(-1)[:3].tolist())
        n_new = 0
        for key in unique_keys:
            entry = self._votes.get(key)
            if entry is None:
                entry = _VoxelVote(count=0, last_view_uid=-1, voter_centers=[])
                self._votes[key] = entry
            if entry.last_view_uid == int(view_uid):
                continue  # same frame already voted this voxel
            if entry.voter_centers:
                min_d = min(
                    ((cam[0] - c[0]) ** 2 + (cam[1] - c[1]) ** 2 + (cam[2] - c[2]) ** 2) ** 0.5
                    for c in entry.voter_centers
                )
                if min_d < self.min_baseline_m:
                    continue  # baseline gate: too close to an existing voter
            entry.count += 1
            entry.last_view_uid = int(view_uid)
            entry.voter_centers.append(cam)
            n_new += 1
        self._fully_voted_view_uids.add(int(view_uid))
        return n_new

    def vote_counts(self, points: torch.Tensor) -> torch.Tensor:
        """Return the current corroboration-vote count per query point
        (``[N]`` int64, ``0`` for voxels never voted free)."""
        if points.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=points.device)
        keys = _voxel_keys(points, self.voxel_size_m)
        counts = torch.tensor(
            [self._votes.get(int(k), _VoxelVote()).count for k in keys.tolist()],
            dtype=torch.int64,
        )
        return counts.to(points.device)

    def certified_mask(self, points: torch.Tensor, *, min_views: int) -> torch.Tensor:
        """Boolean ``[N]`` mask: True where ``points`` fall in a voxel with
        ``>= min_views`` distinct, baseline-gated corroborating votes."""
        counts = self.vote_counts(points)
        return counts >= int(min_views)
