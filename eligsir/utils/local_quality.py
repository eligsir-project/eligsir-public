"""Level A local quality-under-construction: immutable world-space microcells.

Issue #358 (P0 VALIDITY). Design doc: ``docs/architecture/2026-08-13-local-
quality-partitioning.md`` (PR #356), sections 3 (Level A), 5.1 (pixel ->
world -> microcell), 6 and 7 (established-state gating).

Why this exists
---------------
Global test PSNR -- even as a curve over time -- can hide a serious local
failure: one established room losing several dB while a newly mapped region
improves leaves the scene-wide mean roughly unchanged. ``utils/partition_eval``
bins cameras by *trajectory index*, which cannot answer "which physical part of
the map is being forgotten" (a loop puts one room in several temporal bins).
This module adds a *spatial* measurement atom instead, and is strictly
additive: ``partition_eval`` / ``partition_health`` remain the compatibility
and reference paths and are not modified by this feature.

Scope of THIS increment (deliberately narrow -- see the PR/issue for the
deferred list): Level A microcells only. No Level B adaptive regions, no
split/merge triggers, no lineage/hysteresis, no full DEGRADING/FORGOTTEN/
RECOVERING/STABLE state machine, no live visualization.

Correctness property that the whole metric rests on
---------------------------------------------------
**Partition membership must not depend on the reconstruction being measured.**
A holdout pixel's microcell is derived from *fixed sensor/GT depth* and the
*fixed holdout pose* only -- never from the current map's rendered depth. If
that were violated, a degrading map would silently re-assign its own
evaluation pixels and the metric would be measuring itself. The design doc
permits rendered depth only as an explicitly-labeled lower-confidence RGB-only
fallback; that fallback is NOT implemented here (a cell map either comes from
real sensor depth or the camera is skipped and counted).

Two-level identity: durable key vs. dense slot
----------------------------------------------
``cell_key(ijk)`` is a packed, invertible int64 that depends ONLY on the world
point and ``cell_size`` -- it is the durable identity written to disk and is
immutable for the lifetime of the universe, not merely the run. It is NOT a
hash and never collides (a collision would silently fuse two physical places).

``MicrocellGrid.slots_for_keys`` maps keys to *contiguous* indices, which is
what ``scatter_add_`` needs. The slot table is **append-only**: an existing
key's slot never changes when new cells appear. Persisting the key (not the
slot) is what makes a history comparable across a run in which the mapped
volume kept growing.

Pooled vs. mean-of-per-view PSNR (documented, not a bug)
--------------------------------------------------------
Per-cell PSNR here is pooled: ``-10 * log10(sum_sq_err / n_channel_samples)``
over every contributing pixel from every contributing view. It will NOT
reconcile with ``psnr_history``'s ``mean_test_psnr``, for three independent
reasons -- all documented rather than accidental:

1. Across views, ``mean_test_psnr`` averages per-view dB (mean-of-logs);
   pooling is a log-of-means.
2. **Across channels, so does ``utils.image_utils.psnr`` itself** -- it takes
   ``.view(3, -1).mean(1)``, i.e. a PER-CHANNEL MSE, converts each to dB, and
   means those. Pooled per-cell PSNR reduces over all three channels first.
   The two agree exactly when the channel MSEs are equal and diverge (Jensen:
   the mean-of-logs sits higher) when they are not. Verified both ways in
   ``tests/test_local_quality_microcells.py``.
3. Pixels with invalid sensor depth have no world point and are excluded from
   every cell entirely, while they do count toward whole-image per-view PSNR.

All three metrics are correct; they answer different questions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Packed cell-key encoding
# ---------------------------------------------------------------------------
# Each axis index is stored in a fixed 21-bit field with a fixed bias, giving
# a signed range of [-2**20, 2**20 - 1] per axis. At the 0.5m default cell
# size that is +/- 524288 m per axis -- vastly beyond any indoor or building
# -scale scene this project maps, while keeping the packed key inside int64
# (3 * 21 = 63 bits, sign bit free). Out-of-range points are refused loudly
# rather than silently wrapping into another cell's identity.
_AXIS_BITS = 21
_AXIS_BIAS = 1 << (_AXIS_BITS - 1)        # 1048576
_AXIS_MIN = -_AXIS_BIAS                   # -1048576
_AXIS_MAX = _AXIS_BIAS - 1                # 1048575
_AXIS_MASK = (1 << _AXIS_BITS) - 1

# Default microcell edge length in metres. The issue/design doc names
# 0.5-1.0m as the indoor candidate range and explicitly says the exact scale
# must be tested rather than fixed prematurely; 0.5 is the conservative end
# (finer cells -> more, smaller cells -> the establishment gate below does the
# work of discarding the ones with too little evidence).
DEFAULT_CELL_SIZE_M = 0.5
_CANONICAL_TICK_NOT_SUPPLIED = object()


def cell_ijk(xyz: np.ndarray, cell_size: float) -> np.ndarray:
    """World XYZ -> integer microcell index triple, deterministically.

    ``floor(xyz / cell_size)`` against the world-frame origin. The origin is
    the *fixed* world frame, never a map-derived origin/extent/centroid --
    anything derived from the current map would move as the map grows and
    would silently renumber cells mid-run, which is exactly the discontinuity
    this design exists to avoid.

    ``xyz`` is ``[N, 3]`` (or ``[3]``); returns int64 ``[N, 3]``.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"cell_ijk expects [N,3] or [3], got {np.shape(xyz)}")
    if not (cell_size > 0):
        raise ValueError(f"cell_size must be > 0, got {cell_size}")
    ijk = np.floor(arr / float(cell_size)).astype(np.int64)
    return ijk[0] if single else ijk


def cell_key(ijk: np.ndarray) -> np.ndarray:
    """Integer cell index triple -> packed invertible int64 key.

    Not a hash: the packing is bijective over the representable range, so two
    distinct physical cells can never share a key. Out-of-range indices raise
    rather than wrapping.
    """
    arr = np.asarray(ijk, dtype=np.int64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"cell_key expects [N,3] or [3], got {np.shape(ijk)}")
    if arr.size and (arr.min() < _AXIS_MIN or arr.max() > _AXIS_MAX):
        raise ValueError(
            f"cell index out of representable range [{_AXIS_MIN}, {_AXIS_MAX}]: "
            f"min={int(arr.min())} max={int(arr.max())}. A point this far from "
            f"the world origin is almost certainly a pose/scale defect, not a "
            f"real observation -- refusing to alias it onto another cell."
        )
    b = arr + _AXIS_BIAS
    key = (b[:, 0] << (2 * _AXIS_BITS)) | (b[:, 1] << _AXIS_BITS) | b[:, 2]
    return key[0] if single else key


def key_to_ijk(key) -> np.ndarray:
    """Inverse of :func:`cell_key` (used by tests and by any later Level B
    grouping that needs to rebuild adjacency from persisted keys)."""
    arr = np.asarray(key, dtype=np.int64)
    single = arr.ndim == 0
    arr = arr.reshape(-1)
    i = ((arr >> (2 * _AXIS_BITS)) & _AXIS_MASK) - _AXIS_BIAS
    j = ((arr >> _AXIS_BITS) & _AXIS_MASK) - _AXIS_BIAS
    k = (arr & _AXIS_MASK) - _AXIS_BIAS
    out = np.stack([i, j, k], axis=1)
    return out[0] if single else out


def world_to_cell_key(xyz: np.ndarray, cell_size: float) -> np.ndarray:
    """Composition of :func:`cell_ijk` and :func:`cell_key`."""
    return cell_key(cell_ijk(xyz, cell_size))


def cell_center(key, cell_size: float) -> np.ndarray:
    """World-space centre of a cell, for later visualization/adjacency work."""
    ijk = key_to_ijk(key).astype(np.float64)
    return ((ijk + 0.5) * float(cell_size)).astype(np.float32)


# ---------------------------------------------------------------------------
# Append-only key -> dense slot table
# ---------------------------------------------------------------------------
class MicrocellGrid:
    """Append-only mapping from durable cell key to dense scatter slot.

    ``scatter_add_`` needs contiguous 0..n-1 indices; the durable identity is
    the key. Keeping them separate is the whole point: slots may be assigned
    in first-seen order (run-dependent), keys never change. A slot already
    assigned is never reassigned or renumbered when new cells appear, so a
    per-tick accumulator sized to ``len(grid)`` stays index-stable within a
    tick and history rows stay joinable across ticks via the key.
    """

    def __init__(self, cell_size: float = DEFAULT_CELL_SIZE_M):
        if not (cell_size > 0):
            raise ValueError(f"cell_size must be > 0, got {cell_size}")
        self.cell_size = float(cell_size)
        self._key_to_slot: Dict[int, int] = {}
        self._slot_to_key: List[int] = []

    def __len__(self) -> int:
        return len(self._slot_to_key)

    @property
    def keys(self) -> List[int]:
        """Keys in slot order (``keys[slot] == key``)."""
        return list(self._slot_to_key)

    def slot_of(self, key: int) -> Optional[int]:
        return self._key_to_slot.get(int(key))

    def key_of(self, slot: int) -> int:
        return self._slot_to_key[int(slot)]

    def slots_for_keys(self, keys) -> np.ndarray:
        """Map keys -> slots, allocating new slots for unseen keys.

        Append-only: previously-issued slots are never changed.
        """
        arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        out = np.empty(arr.shape[0], dtype=np.int64)
        k2s = self._key_to_slot
        s2k = self._slot_to_key
        for n, k in enumerate(arr.tolist()):
            s = k2s.get(k)
            if s is None:
                s = len(s2k)
                k2s[k] = s
                s2k.append(k)
            out[n] = s
        return out

    def slots_for_world_points(self, xyz: np.ndarray) -> np.ndarray:
        return self.slots_for_keys(world_to_cell_key(xyz, self.cell_size))


# ---------------------------------------------------------------------------
# Per-camera frozen pixel -> cell map
# ---------------------------------------------------------------------------
@dataclass
class CameraCellMap:
    """Frozen pixel -> microcell assignment for one holdout camera.

    Computed ONCE from the camera's fixed sensor/GT depth and its fixed
    holdout pose, then reused at every evaluation tick. Nothing about the
    current reconstruction enters this.

    ``slot_index`` is a flattened ``[H*W]`` int64 tensor of dense slots, with
    ``-1`` at every pixel whose depth was invalid (zero / non-finite / outside
    ``[min_depth_m, max_depth_m]``). ``valid_mask`` is the ``[H*W]`` bool
    counterpart. ``height``/``width`` pin the resolution the map was built at:
    a cell-ID map must never be interpolated to a different resolution (that
    would fabricate cell identities), so a mismatch means recompute or skip.
    """

    cam_key: str
    height: int
    width: int
    slot_index: "object"           # torch.LongTensor [H*W]
    valid_mask: "object"           # torch.BoolTensor [H*W]
    n_valid: int
    n_total: int
    source: str = "sensor_depth"   # never "rendered_depth" -- see module docstring
    # ---- precomputed per-tick constants (issue #358 GPU-cost measurement) ----
    # Everything below is a pure function of the FROZEN slot_index/valid_mask,
    # so it is identical at every tick for this camera. Computing it once here
    # rather than per view per tick removes two of add_view's three fp64
    # scatter_add_ calls and both of its boolean-mask gathers (whose output
    # shape is data-dependent, and therefore forces a device->host sync that
    # the "sync-free" claim in CellTickAccumulator did not account for).
    valid_px: "object" = None      # torch.LongTensor [n_valid] flat pixel ids
    valid_slots: "object" = None   # torch.LongTensor [n_valid] slot per valid px
    uniq_slots: "object" = None    # torch.LongTensor [n_touched] distinct slots
    uniq_counts: "object" = None   # float64 [n_touched] == 3 * pixels in slot
    uniq_ones: "object" = None     # float64 [n_touched] all-ones (view presence)

    @property
    def invalid_fraction(self) -> float:
        return 0.0 if self.n_total <= 0 else 1.0 - (self.n_valid / float(self.n_total))

    def index_on(self, device) -> Tuple["object", "object"]:
        """``(slot_index, valid_mask)`` resident on ``device``.

        The map is built from ``utils.comparison_report._get_sensor_depth``,
        which returns a **CPU** tensor for a holdout camera (holdout cams never
        go through ``train_streaming.bind_resolution_tier``, the only path that
        writes a CUDA ``cam._sensor_depth_cache``), while the render being
        measured lives on the training device. Without this transfer every
        ``scatter_add_`` in :meth:`CellTickAccumulator.add_view` raises
        ``RuntimeError: Expected all tensors to be on the same device`` --
        which is NOT the ``ValueError`` :meth:`LocalQualityTracker.observe`
        guards, so on a real GPU run every view was dropped silently and the
        whole history read ``n_cells_observed=0`` (issue #358, confirmed on
        ``issue358_floor3_local_quality_cells_cells_on``).

        **Deliberately NOT cached on the device.** Caching the migrated copy on
        the frozen map would be one transfer per camera per run instead of one
        per view per tick -- but the maps are dense ``[H*W]`` (int64 slots +
        bool mask) and every holdout camera keeps one for the whole run, so at
        this project's real holdout scale that residency is enormous: measured
        **8.44 MiB/camera at 720x1280, i.e. 5.40 GiB for floor3's 655 holdout
        cameras**, permanently, on a box whose single shared GPU is also
        holding the model. The transfer is ~8 MiB and happens once per view per
        tick (ticks are 1000 iterations apart), so trading it for O(1) instead
        of O(n_cameras) device memory is the right side of that deal. When the
        map already lives on ``device`` (every CPU-only path, including the
        offline drivers and this module's tests) this is a no-op.

        Cell identity is unchanged -- only the tensors' residency is.
        """
        import torch
        dev = torch.device(device)
        if self.slot_index.device == dev:
            return self.slot_index, self.valid_mask
        return (
            self.slot_index.to(device=dev, non_blocking=True),
            self.valid_mask.to(device=dev, non_blocking=True),
        )

    def hoisted_on(self, device) -> Tuple["object", "object", "object", "object", "object"]:
        """``(valid_px, valid_slots, uniq_slots, uniq_counts, uniq_ones)`` on
        ``device`` -- the precomputed per-tick constants (issue #358 hoist),
        transferred per call rather than cached, for the same reason
        :meth:`index_on` does not cache: these are dense per-camera tensors
        held for the whole run across every holdout camera, so caching a
        device-resident copy would multiply the frozen-map memory footprint
        by the number of cameras instead of paying one small transfer per
        view per tick.

        Callers must check ``valid_slots is not None`` first (``None`` means
        this map was frozen before the hoist landed, e.g. an old checkpoint's
        cached map) and fall back to :meth:`index_on` + on-the-fly histogram
        computation in that case.
        """
        import torch
        dev = torch.device(device)
        if self.valid_px.device == dev:
            return (
                self.valid_px, self.valid_slots,
                self.uniq_slots, self.uniq_counts, self.uniq_ones,
            )
        return (
            self.valid_px.to(device=dev, non_blocking=True),
            self.valid_slots.to(device=dev, non_blocking=True),
            self.uniq_slots.to(device=dev, non_blocking=True),
            self.uniq_counts.to(device=dev, non_blocking=True),
            self.uniq_ones.to(device=dev, non_blocking=True),
        )


def camera_c2w(cam) -> "object":
    """Camera-to-world in the convention
    :func:`utils.rgbd_frames.backproject_depth_pixels_torch` expects.

    Matches ``scene.cameras.Camera.camera_center``'s own derivation and
    ``eligsir/optimization/gaussian/strategies/live/reseed_producer.py``'s
    documented rule -- deliberately NOT re-derived here, because this repo has
    a documented history of w2c/c2w convention burns (see CLAUDE.md's
    ORB-SLAM3 pose-convention notes).
    """
    return cam.world_view_transform.inverse().transpose(0, 1)


def camera_intrinsics(cam) -> Tuple[float, float, float, float]:
    """(fx, fy, cx, cy) at the camera's own image resolution, falling back to
    the FoV-derived values exactly as ``make_gsplat_camera_tensors`` does."""
    w = int(cam.image_width)
    h = int(cam.image_height)
    fx = getattr(cam, "fx", None)
    fy = getattr(cam, "fy", None)
    cx = getattr(cam, "cx", None)
    cy = getattr(cam, "cy", None)
    fx = float(fx) if fx is not None else w / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = float(fy) if fy is not None else h / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    cx = float(cx) if cx is not None else w / 2.0
    cy = float(cy) if cy is not None else h / 2.0
    return fx, fy, cx, cy


def camera_identity(cam) -> str:
    """Stable per-camera cache key."""
    from eligsir.utils.partition_eval import camera_source_index
    src = camera_source_index(cam)
    if src is not None:
        return f"src{int(src)}"
    return str(getattr(cam, "image_name", None) or id(cam))


def camera_pixel_cell_keys(
    cam,
    grid: MicrocellGrid,
    sensor_depth,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
    device=None,
):
    """Packed world-cell key and validity tensors for every depth pixel."""
    import torch

    if sensor_depth is None:
        return None
    depth = sensor_depth
    if depth.dim() > 2:
        depth = depth.reshape(depth.shape[-2], depth.shape[-1])
    depth = depth.to(dtype=torch.float32, device=device or depth.device)
    height, width = map(int, depth.shape)
    valid = (
        torch.isfinite(depth)
        & (depth > float(min_depth_m))
        & (depth < float(max_depth_m))
    )
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = torch.where(valid, depth, 0.0)
    fx, fy, cx, cy = camera_intrinsics(cam)
    sx = width / float(max(int(cam.image_width), 1))
    sy = height / float(max(int(cam.image_height), 1))
    points = torch.stack(
        ((xs - cx * sx) / (fx * sx) * z, (ys - cy * sy) / (fy * sy) * z, z),
        dim=-1,
    ).reshape(-1, 3)
    c2w = camera_c2w(cam)
    if c2w.device != depth.device and device is None:
        raise ValueError("camera pose and sensor depth must share one device")
    c2w = c2w.to(device=depth.device, dtype=torch.float32)
    points = points @ c2w[:3, :3].T + c2w[:3, 3]
    ijk = torch.floor(points / grid.cell_size).to(torch.int64)
    in_range = ((ijk >= _AXIS_MIN) & (ijk <= _AXIS_MAX)).all(dim=1)
    biased = ijk + _AXIS_BIAS
    keys = (
        (biased[:, 0] << (2 * _AXIS_BITS))
        | (biased[:, 1] << _AXIS_BITS)
        | biased[:, 2]
    )
    valid = valid.reshape(-1)
    return torch.where(valid & in_range, keys, torch.where(valid, -2, -1)), valid


def build_camera_cell_map(
    cam,
    grid: MicrocellGrid,
    sensor_depth,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
    device=None,
) -> Optional[CameraCellMap]:
    """Freeze one holdout camera's pixel -> microcell map.

    ``sensor_depth`` is a ``[1,H,W]`` or ``[H,W]`` metric-depth tensor from
    the SENSOR/GT (``utils.comparison_report._get_sensor_depth``), never a
    rendered depth. Returns ``None`` when no depth is available or no pixel
    is valid -- the caller then skips this camera and counts the skip rather
    than falling back to anything model-dependent.
    """
    import torch

    mapped = camera_pixel_cell_keys(
        cam, grid, sensor_depth, min_depth_m, max_depth_m, device
    )
    if mapped is None:
        return None
    keys, valid_flat = mapped
    valid_px = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    if valid_px.numel() == 0:
        return None
    valid_keys = keys.index_select(0, valid_px)
    host_keys = valid_keys.detach().cpu().numpy()
    if (host_keys < 0).any():
        raise ValueError("cell index is outside the packed int64 range")
    slots = grid.slots_for_keys(host_keys)
    slot_index = torch.full_like(keys, -1)
    slot_index[valid_px] = torch.as_tensor(slots, device=keys.device)
    valid_slots = slot_index.index_select(0, valid_px)
    uniq_slots, counts = torch.unique(valid_slots, return_counts=True)
    height, width = map(int, sensor_depth.shape[-2:])
    return CameraCellMap(
        cam_key=camera_identity(cam),
        height=height,
        width=width,
        slot_index=slot_index,
        valid_mask=valid_flat,
        n_valid=int(valid_px.numel()),
        n_total=height * width,
        valid_px=valid_px,
        valid_slots=valid_slots,
        uniq_slots=uniq_slots,
        # 3 channel-samples per contributing pixel -- identical to what the
        # old per-pixel ``scatter_add_(full_like(idx_v, 3.0))`` summed.
        uniq_counts=(counts * 3).to(torch.float64),
        uniq_ones=torch.ones_like(uniq_slots, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# GPU-resident masked reduction
# ---------------------------------------------------------------------------
@dataclass
class CellTickAccumulator:
    """Per-tick per-cell accumulators, GPU-resident until one final sync.

    Three ``[n_slots]`` vectors: summed squared error, contributing channel
    -sample count, and distinct contributing views. ``add_view`` does ONE
    ``scatter_add_`` over the valid pixels of the view (the only
    residual-dependent reduction) plus two ``index_add_`` calls over the
    camera's precomputed per-cell histogram -- no ``.item()``/``.cpu()``
    anywhere. Only :meth:`finalize` syncs, transferring three compact
    ``[n_slots]`` vectors once per tick (not once per view, and never a full
    residual map).

    Accumulation is ``float64`` because a large cell can sum millions of
    squared residuals and float32 accumulation would lose the small ones.

    GPU cost of that fp64 choice: MEASURED ON BOTH TARGET ARCHS, no longer assumed
    --------------------------------------------------------------------------
    The original note here said fp64 ``scatter_add_`` may be emulated on
    this project's gfx1100/gfx1151 targets and that the CPU-torch figure
    (22.9 ms/tick, 2.29 ms/view at 480x640 / 4000 cells) may not transfer.
    ``scripts/bench_local_quality_scatter.py`` measured it on two real AMD
    archs: gfx1201 (RDNA4, r9700, ROCm 7.2) and gfx1151 (RDNA3.5, Radeon
    8060S iGPU, ROCm 7.2). **fp64 is the safe default on both** -- but the
    two archs disagree sharply on WHY, and a reader relying on only the
    gfx1201 numbers below would draw the wrong conclusion about fp32.

    * The fp64 penalty is a function of ATOMIC CONTENTION (pixels per cell),
      not of total element count -- consumer RDNA has no native fp64 atomic
      add, so HIP uses a CAS retry loop whose cost grows with the number of
      threads hitting one address. This holds on both archs.
    * At the DEFAULT ``cell_size=0.5`` the real fr3 holdout tick produces
      **691 cells / ~373 valid px per cell per view** (measured by
      ``scripts/local_quality_cell_count.py`` from fr3's own sensor depth +
      groundtruth poses, over a 10-view holdout spread across the whole
      trajectory). There, per view:

      - **gfx1201**: fp64 0.36 ms, fp32 0.41 ms -- fp64/fp32 = **0.88x**
        (fp64 is CHEAPER than fp32 on this arch).
      - **gfx1151**: fp64 0.45-0.46 ms, fp32 0.08-0.10 ms -- fp64/fp32 =
        **~5.4x** (fp32 is 2-40x CHEAPER than gfx1201's fp32 across the
        whole contention sweep and stays essentially flat, the shape of a
        real native fp32 atomic -- NOT the emulation-catastrophe shape the
        PR set out to rule out, which would scale hard with contention the
        way gfx1201's fp32 numbers actually do).

      Absolute cost at the default 10-view tick is what matters for
      whether fp64 is "safe" -- **3.6 ms on gfx1201, 4.5-4.6 ms on
      gfx1151**, both well under the 22.9 ms CPU-torch figure the original
      emulation concern was about. fp64 itself is not emulated or
      pathological on either arch; the two just land on opposite sides of
      1x relative to fp32.
    * The PR's 4000-cell benchmark configuration overstates the real cell
      count ~6x, but lands in the SAME flat, benign part of the contention
      curve on gfx1201 (0.84x there vs 0.82x at 691 pre-hoist); on gfx1151
      it is 1.8-1.95x, still a small multiple. Either way the conclusion
      about safety at the shipping config was not affected -- only the
      stated operating point.
    * Cost rises as ``cell_size`` COARSENS (fewer, larger cells == more
      threads per address), and this is where the archs diverge most in
      absolute terms:

      - gfx1201: fp64/fp32 is 1.19x at 1.0m (212 cells, 8.6 ms/tick) and
        3.55x at 2.0m (70 cells, 44.6 ms/tick) -- across the 0.5-1.0m
        "indoor candidate range" `local_quality.py` documents, the penalty
        is negligible.
      - gfx1151: fp64/fp32 is ~11.5x at 1.0m (10.5 ms/tick) and ~34x at
        2.0m (37.5 ms/tick) -- the penalty across that SAME 0.5-1.0m range
        is real (5.4x-11.5x), not negligible, though 10.5 ms is still
        small next to the render pass the tick rides on. At 2.0m the tick
        (37.5 ms) exceeds the CPU-torch reference figure, which the
        0.5-1.0m "negligible" framing above does not.

      Ticks fire every ``--streaming_psnr_chart_interval`` (default 1000)
      iterations, so even the gfx1151 2.0m case is a mild cost knob rather
      than a correctness risk -- but treat the "negligible across
      0.5-1.0m" framing as gfx1201-specific, not universal.
    * The two count accumulators used to be re-scattered per pixel per view
      even though they are frozen per camera; hoisting them into
      :class:`CameraCellMap` (see its field comments) cut fp64 add_view by
      **~2.0-2.3x at every operating point measured on both archs** and is
      what brings the default tick to 3.6 ms (gfx1201) / 4.5 ms (gfx1151).
      It also removed both boolean-mask gathers, whose data-dependent
      output shape forced a device->host sync per view that the original
      "sync-free" claim did not account for.
    * fp32 accumulation was measured as a fallback and is NOT used, but the
      reason differs by arch. On gfx1201 it buys nothing (0.88x-0.99x)
      while costing precision. **On gfx1151 fp32 IS meaningfully cheaper
      (~5.4x at default) -- adopting it there would genuinely save
      ~3.6 ms/tick, so the "fp32 buys nothing" framing is gfx1201-specific
      and would be false stated as a general claim.** The reason fp64 stays
      the default anyway: its absolute cost is sub-5ms/tick on every
      measured arch at the shipping config (well under the 22.9 ms
      CPU-torch reference that motivated this investigation), and the
      measured precision cost of dropping it is real, not zero -- max
      |dPSNR| = 1.5e-05 dB with 2.9e-06 dB run-to-run nondeterminism
      (atomic ``scatter_add_`` has no defined summation order). Given the
      absolute cost is already small everywhere, reproducibility and exact
      cross-run comparability are preferred over a further, arch-dependent
      speedup that isn't needed to hit the CPU-torch bar.
    """

    n_slots: int
    device: "object" = None
    sq_err: "object" = None
    n_samples: "object" = None
    n_views: "object" = None

    def _ensure(self, device) -> None:
        """Allocate on first use, on whatever device the render lives on.

        Deferred so the caller never has to guess the render device before the
        first view arrives. ``n_slots`` is fixed at construction time (after
        :meth:`LocalQualityTracker.prepare` has frozen every cell map for this
        tick), so no view can index past the end.
        """
        import torch
        if self.sq_err is not None:
            return
        self.device = device
        z = lambda: torch.zeros(self.n_slots, dtype=torch.float64, device=device)
        self.sq_err, self.n_samples, self.n_views = z(), z(), z()

    def add_view(self, render_img, gt_img, cell_map: CameraCellMap) -> None:
        """Accumulate one already-rendered view. No sync.

        ``render_img``/``gt_img`` are ``[3,H,W]`` in [0,1]. Squared error is
        summed over all three channels of each valid pixel, and ``n_samples``
        counts channel-samples, so ``sq_err / n_samples`` is exactly the MSE
        ``utils.image_utils.psnr`` would compute over that pixel set.
        """
        import torch
        self._ensure(render_img.device)
        h, w = cell_map.height, cell_map.width
        if int(render_img.shape[-2]) != h or int(render_img.shape[-1]) != w:
            raise ValueError(
                f"render {tuple(render_img.shape[-2:])} does not match frozen cell "
                f"map {(h, w)} -- a cell-ID map must never be interpolated across "
                f"resolutions; rebuild it for this resolution instead."
            )
        se = ((render_img.to(torch.float32) - gt_img.to(torch.float32)) ** 2)
        se = se.reshape(3, h * w).sum(dim=0)                     # [H*W] fp32
        # Frozen cell maps are built on whatever device the SENSOR depth lived
        # on (CPU for every holdout camera); the render lives on the training
        # device. Transfer per view rather than caching a device copy on the
        # map -- see CameraCellMap.index_on/hoisted_on for the measured reason.
        if cell_map.valid_slots is None:      # map frozen before the hoist
            idx, m = cell_map.index_on(self.device)
            idx_v = idx[m]
            se_v = se[m].to(torch.float64)
            uniq, cnt = torch.unique(idx_v, return_counts=True)
            uniq_counts = (cnt * 3).to(torch.float64)
            uniq_ones = torch.ones_like(uniq, dtype=torch.float64)
        else:
            # index_select over frozen indices: statically-known output shape,
            # so no data-dependent size and no implicit host sync.
            valid_px, valid_slots, uniq, uniq_counts, uniq_ones = (
                cell_map.hoisted_on(self.device)
            )
            idx_v = valid_slots
            se_v = se.index_select(0, valid_px).to(torch.float64)
        # The ONLY residual-dependent reduction, and the only one whose cost
        # scales with pixels-per-cell (fp64 atomic contention -- see
        # scripts/bench_local_quality_scatter.py).
        self.sq_err.scatter_add_(0, idx_v, se_v)
        # Channel-sample counts and distinct-view presence are pure functions
        # of the frozen cell map, so they are accumulated from the precomputed
        # per-cell histogram (one index_add_ over the cell's DISTINCT slots)
        # instead of re-scattering one element per pixel every tick. Exactly
        # equal to the old per-pixel form by construction.
        self.n_samples.index_add_(0, uniq, uniq_counts)
        self.n_views.index_add_(0, uniq, uniq_ones)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ONLY device->host sync: three ``[n_slots]`` vectors."""
        if self.sq_err is None:
            z = np.zeros(self.n_slots, dtype=np.float64)
            return z, z.copy(), z.copy()
        return (
            self.sq_err.detach().cpu().numpy(),
            self.n_samples.detach().cpu().numpy(),
            self.n_views.detach().cpu().numpy(),
        )


def pooled_psnr_db(sum_sq_err: np.ndarray, n_samples: np.ndarray) -> np.ndarray:
    """Pooled PSNR per cell, ``-10*log10(sse/n)`` for peak-1.0 images.

    Equivalent to ``utils.image_utils.psnr``'s ``20*log10(1/sqrt(mse))`` for
    the same pixel set, but pooled over the cell's pixels rather than averaged
    over per-view scalars (see the module docstring). Cells with no samples
    get NaN.
    """
    sse = np.asarray(sum_sq_err, dtype=np.float64)
    n = np.asarray(n_samples, dtype=np.float64)
    out = np.full(n.shape, np.nan, dtype=np.float64)
    ok = n > 0
    mse = np.zeros_like(out)
    mse[ok] = sse[ok] / n[ok]
    # A perfect cell (mse == 0) would be +inf dB; clamp to a finite ceiling so
    # downstream peak/drop arithmetic never produces inf-inf = NaN.
    mse[ok] = np.maximum(mse[ok], 1e-12)
    out[ok] = -10.0 * np.log10(mse[ok])
    return out


# ---------------------------------------------------------------------------
# Establishment gate + quality-under-construction state
# ---------------------------------------------------------------------------
@dataclass
class EstablishmentPolicy:
    """Evidence a cell must accumulate before a later decline counts as
    forgetting. A newly discovered region being poor is NOT forgetting.

    Three requirements, all of which are computable from what a PSNR tick
    already produces -- deliberately no requirement that would need new
    plumbing this increment does not build:

    ``min_eval_samples``
        Minimum channel-samples in the cell at the tick, i.e. enough holdout
        pixels for a pooled PSNR to mean anything at all. 300 channel-samples
        == 100 pixels.
    ``min_views``
        Minimum number of DISTINCT holdout cameras contributing pixels to the
        cell at the tick -- the multi-view-support requirement. A cell seen
        from a single viewpoint can look fine while being geometrically wrong,
        and its "quality" is not a novel-view claim.
    ``min_ticks``
        Minimum number of evaluation ticks at which the cell has already met
        the two conditions above. This is the OPTIMIZATION-EXPOSURE PROXY and
        is labeled as such in the emitted record: a true per-cell count of
        optimizer exposures would require per-Gaussian-to-cell attribution
        plumbing through the training loop, which this measurement-only
        increment deliberately does not add. Ticks-observed is a monotone
        proxy for "the mapper has had repeated opportunity to optimize here".
    """

    min_eval_samples: int = 300
    min_views: int = 2
    min_ticks: int = 3


@dataclass
class CellState:
    """Longitudinal per-cell state. ``peak_psnr_db`` is tracked ONLY from the
    establishment tick onward -- a peak taken from first sighting would
    include the cell's initial under-reconstructed measurements and would
    understate a genuine later drop."""

    key: int
    first_seen_iter: int
    qualifying_ticks: int = 0
    established: bool = False
    established_iter: Optional[int] = None
    peak_psnr_db: Optional[float] = None
    peak_iter: Optional[int] = None
    last_psnr_db: Optional[float] = None
    last_iter: Optional[int] = None
    max_drop_from_peak_db: float = 0.0


class LocalQualityTracker:
    """Owns the grid, the frozen per-camera cell maps, and per-cell history.

    Usage per evaluation tick (all inside the caller's existing try/except):

        tracker.prepare(test_cams)          # freeze/reuse cell maps (no render)
        acc = tracker.new_accumulator()     # sized to the now-final slot count
        # ... the SAME render pass psnr_history already does, per view:
        tracker.observe(cam, render_img, gt_img, acc)
        rec = tracker.record_tick(iteration, acc)

    Everything between ``prepare`` and ``record_tick`` is sync-free.
    ``prepare`` does one host transfer per camera the FIRST time that camera
    is seen (to allocate its dense slots) and is a pure dict lookup on every
    subsequent tick.
    """

    def __init__(
        self,
        cell_size: float = DEFAULT_CELL_SIZE_M,
        policy: Optional[EstablishmentPolicy] = None,
        min_depth_m: float = 0.1,
        max_depth_m: float = 10.0,
    ):
        self.grid = MicrocellGrid(cell_size)
        self.policy = policy or EstablishmentPolicy()
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.states: Dict[int, CellState] = {}
        self._cell_maps: Dict[Tuple[str, int, int], CameraCellMap] = {}
        self._prepared: Dict[str, CameraCellMap] = {}
        self.cams_without_depth: int = 0
        self.views_skipped_unprepared: int = 0
        self.views_skipped_resolution_mismatch: int = 0
        self.cumulative_skipped_resolution_mismatch: int = 0
        self._warned_resolution_mismatch: bool = False
        self.views_skipped_error: int = 0
        self.cumulative_skipped_error: int = 0
        self._warned_view_error: bool = False

    # -- frozen cell maps ---------------------------------------------------
    def cell_map_for(self, cam, sensor_depth) -> Optional[CameraCellMap]:
        """Frozen, cached pixel -> cell map for one camera.

        Cached on ``(camera identity, H, W)``: ``_get_sensor_depth`` nearest
        -resizes to whatever resolution is requested, and this project has
        live multi-resolution flags, so the render H/W can change mid-run. A
        different resolution gets its OWN frozen map rather than an
        interpolated cell-ID map (which would fabricate cell identities and
        break the determinism claim). Persistent maps are always CPU-owned;
        callers migrate their tensors only for the duration of one reduction.
        """
        if sensor_depth is None:
            self.cams_without_depth += 1
            return None
        d = sensor_depth
        if hasattr(d, "dim") and d.dim() > 2:
            h, w = int(d.shape[-2]), int(d.shape[-1])
        else:
            h, w = int(d.shape[0]), int(d.shape[1])
        ck = (camera_identity(cam), h, w)
        hit = self._cell_maps.get(ck)
        if hit is not None:
            return hit
        cm = build_camera_cell_map(
            cam, self.grid, sensor_depth,
            min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m,
            device="cpu",
        )
        if cm is None:
            self.cams_without_depth += 1
            return None
        self._cell_maps[ck] = cm
        return cm

    def prepare(self, cams, sensor_depth_fn=None) -> int:
        """Freeze (or reuse) the cell map of every camera in ``cams``.

        Must run BEFORE :meth:`new_accumulator`, because building a cell map
        can allocate new dense slots and the accumulator is sized once. On
        every tick after the first this is a dict lookup per camera.

        ``sensor_depth_fn(cam, h, w)`` defaults to
        ``utils.comparison_report._get_sensor_depth`` -- the single existing
        sensor/GT depth accessor in this repo. Returns the number of cameras
        with a usable frozen map.
        """
        if sensor_depth_fn is None:
            from eligsir.utils.comparison_report import _get_sensor_depth as sensor_depth_fn
        self._prepared = {}
        # Per-tick counters (prepare() runs once per tick and rebuilds
        # _prepared, so these describe THIS tick, matching the per-tick fields
        # they are emitted alongside). Cumulative variants below.
        self.cams_without_depth = 0
        self.views_skipped_unprepared = 0
        self.views_skipped_resolution_mismatch = 0
        self.views_skipped_error = 0
        for cam in (cams or []):
            try:
                d = sensor_depth_fn(cam, int(cam.image_height), int(cam.image_width))
            except Exception:
                d = None
            cm = self.cell_map_for(cam, d)
            if cm is not None:
                self._prepared[cm.cam_key] = cm
        return len(self._prepared)

    def observe(self, cam, render_img, gt_img, acc: CellTickAccumulator) -> bool:
        """Accumulate one already-rendered view against its frozen cell map.

        Skips (and counts) any camera :meth:`prepare` did not produce a map
        for -- a camera that first appeared after the accumulator was sized
        would otherwise index past the end of the accumulators. Sync-free.

        **Resolution mismatch is caught here, loudly.** ``_get_sensor_depth``
        only nearest-resizes on its COLD path -- an already-populated
        ``cam._sensor_depth_cache`` (which
        ``eligsir/mapping/resolution_cache.py`` writes directly) is
        returned verbatim regardless of the requested target size. With this
        project's live multi-resolution flags, ``prepare`` can therefore
        freeze a cell map at one resolution while the render arrives at
        another. ``add_view`` correctly refuses to interpolate a cell-ID map
        and raises -- but ``compute_per_view_test_psnr_gpu``'s ``except
        Exception: continue`` would swallow that silently, and EVERY view
        dropping out yields a record reading ``n_cells_observed=0`` /
        ``worst_drop=0.00dB``, which is indistinguishable from "the map is
        healthy". For a metric whose entire purpose is catching an otherwise
        invisible failure, a silent all-zero tick is the worst possible
        failure mode, so it is counted and warned about instead.
        """
        cm = self._prepared.get(camera_identity(cam))
        if cm is None:
            self.views_skipped_unprepared += 1
            return False
        try:
            acc.add_view(render_img, gt_img, cm)
        except ValueError as e:
            self.views_skipped_resolution_mismatch += 1
            self.cumulative_skipped_resolution_mismatch += 1
            if not self._warned_resolution_mismatch:
                self._warned_resolution_mismatch = True
                print(
                    f"[local_quality] WARNING: frozen cell map does not match the "
                    f"render resolution for camera "
                    f"{getattr(cam, 'image_name', '?')} -- this view (and likely "
                    f"every view) is EXCLUDED, so a zero-drop tick here means "
                    f"'not measured', NOT 'no forgetting'. {e}",
                    flush=True,
                )
            return False
        except Exception as e:
            # Anything that is NOT the anticipated resolution ValueError --
            # e.g. a device-mismatch RuntimeError from scatter_add_ when the
            # frozen cell map's index tensors live on a different device than
            # the render (issue #358: a CPU-resident cell map, built from
            # `_get_sensor_depth`'s CPU tensor, against a CUDA render). The
            # original code let this propagate, and the sole caller
            # (utils.psnr_history.compute_per_view_test_psnr_gpu) swallows it
            # in a bare `except Exception: continue` AFTER this view's PSNR
            # was already appended -- so the view vanished from this metric
            # with no counter, no log, and measurement_incomplete=False,
            # while mean_test_psnr was completely unaffected. That really
            # happened for 53/53 ticks of a real corridor run. Catch-all +
            # count + warn-once, so no future exception class can reproduce
            # this class of silent all-zero-looks-healthy tick.
            self.views_skipped_error += 1
            self.cumulative_skipped_error += 1
            if not self._warned_view_error:
                self._warned_view_error = True
                print(
                    f"[local_quality] WARNING: accumulating camera "
                    f"{getattr(cam, 'image_name', '?')} raised "
                    f"{type(e).__name__}: {e} -- this view (and likely every "
                    f"view) is EXCLUDED, so a zero-drop tick here means 'not "
                    f"measured', NOT 'no forgetting'.",
                    flush=True,
                )
            return False
        return True

    def new_accumulator(self, device=None) -> CellTickAccumulator:
        """Accumulator sized to the slot count as of the last :meth:`prepare`.

        Slots are append-only, so a map frozen on an earlier tick keeps its
        slots and history rows stay joinable across ticks by cell key.
        """
        return CellTickAccumulator(n_slots=max(len(self.grid), 1), device=device)

    # -- history ------------------------------------------------------------
    def record_tick(
        self,
        iteration: int,
        acc: CellTickAccumulator,
        *,
        canonical_evaluation_tick: object = _CANONICAL_TICK_NOT_SUPPLIED,
    ) -> dict:
        """Sync once, update per-cell state, return one machine-readable record.

        Emits ``psnr_peak``/``drop_from_peak_db`` ONLY for established cells --
        this is the guarantee that no "forgetting" signal can fire on a region
        that was never established.

        When supplied by the streaming composition root, ``canonical_evaluation_tick``
        is the exact #698 record reduced from this same holdout render.  The
        local row stores a content-addressed reference rather than copying its
        clocks, preventing a competing local-quality timeline from drifting.
        """
        sse, n_samples, n_views = acc.finalize()
        psnr = pooled_psnr_db(sse, n_samples)
        keys = self.grid.keys
        pol = self.policy

        cells: Dict[str, dict] = {}
        n_established = 0
        worst_drop = 0.0
        worst_drop_key = None
        for slot in range(min(len(keys), len(psnr))):
            n_s = float(n_samples[slot])
            if n_s <= 0:
                continue
            key = int(keys[slot])
            st = self.states.get(key)
            if st is None:
                st = CellState(key=key, first_seen_iter=int(iteration))
                self.states[key] = st
            val = float(psnr[slot])
            nv = int(round(float(n_views[slot])))
            st.last_psnr_db = val
            st.last_iter = int(iteration)

            qualifies = (n_s >= pol.min_eval_samples) and (nv >= pol.min_views)
            if qualifies:
                st.qualifying_ticks += 1
                if not st.established and st.qualifying_ticks >= pol.min_ticks:
                    st.established = True
                    st.established_iter = int(iteration)

            rec = {
                "cell_key": key,
                "n_eval_samples": int(n_s),
                "n_views": nv,
                "psnr_db": val,
                "established": bool(st.established),
                "qualifying_ticks": int(st.qualifying_ticks),
                "first_seen_iter": int(st.first_seen_iter),
            }
            if st.established:
                n_established += 1
                # Peak is tracked from the establishment tick onward only.
                if st.peak_psnr_db is None or val > st.peak_psnr_db:
                    st.peak_psnr_db = val
                    st.peak_iter = int(iteration)
                drop = max(0.0, float(st.peak_psnr_db) - val)
                st.max_drop_from_peak_db = max(st.max_drop_from_peak_db, drop)
                rec.update({
                    "established_iter": st.established_iter,
                    "psnr_peak_db": float(st.peak_psnr_db),
                    "peak_iter": st.peak_iter,
                    "drop_from_peak_db": drop,
                    "max_drop_from_peak_db": float(st.max_drop_from_peak_db),
                })
                if drop > worst_drop:
                    worst_drop = drop
                    worst_drop_key = key
            cells[str(key)] = rec

        tick_ref = {"status": "not_bound"}
        tick_invalid = False
        if canonical_evaluation_tick is None:
            # The real composition root supplied this argument but the #698
            # writer failed or had no usable tick.  Preserve the local row for
            # diagnostics, but make its unjoinable clock identity explicit.
            tick_ref = {"status": "unavailable"}
            tick_invalid = True
        elif canonical_evaluation_tick is not _CANONICAL_TICK_NOT_SUPPLIED:
            try:
                tick_iteration = int(canonical_evaluation_tick["iteration"])
                tick_schema = str(canonical_evaluation_tick["schema_version"])
                if tick_iteration != int(iteration):
                    tick_ref = {
                        "status": "iteration_mismatch",
                        "schema_version": tick_schema,
                        "iteration": tick_iteration,
                    }
                    tick_invalid = True
                else:
                    canonical_json = json.dumps(
                        canonical_evaluation_tick,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    tick_ref = {
                        "status": "bound",
                        "schema_version": tick_schema,
                        "iteration": tick_iteration,
                        "sha256": hashlib.sha256(
                            canonical_json.encode("utf-8")
                        ).hexdigest(),
                    }
            except (KeyError, TypeError, ValueError):
                tick_ref = {"status": "invalid"}
                tick_invalid = True

        return {
            "iteration": int(iteration),
            "canonical_evaluation_tick": tick_ref,
            "cell_size_m": self.grid.cell_size,
            "n_cells_observed": len(cells),
            "n_cells_established": n_established,
            "n_cells_total_seen": len(self.grid),
            "cams_without_sensor_depth": int(self.cams_without_depth),
            "views_skipped_unprepared": int(self.views_skipped_unprepared),
            # Nonzero here means this tick's numbers are INCOMPLETE. A tick
            # with zero observed cells AND a nonzero skip count is "not
            # measured", never "nothing wrong" -- see observe()'s docstring.
            "views_skipped_resolution_mismatch": int(self.views_skipped_resolution_mismatch),
            # Views dropped by any OTHER exception out of add_view (issue
            # #358: a device-mismatch RuntimeError -- e.g. a CPU cell map
            # against a CUDA render -- was previously invisible here, and
            # `observe()` did not even catch it). Same standing as the
            # mismatch count above: nonzero means this tick is INCOMPLETE,
            # never "nothing wrong".
            "views_skipped_error": int(self.views_skipped_error),
            "measurement_incomplete": bool(
                self.views_skipped_resolution_mismatch > 0
                or self.views_skipped_unprepared > 0
                or self.views_skipped_error > 0
                or self.cams_without_depth > 0
                # A supplied canonical tick that cannot join this record means
                # its local PSNR has no trustworthy causal clock identity.
                or tick_invalid
                # HARD INVARIANT (issue #358): zero observed cells must never
                # be reported as a complete/healthy measurement, REGARDLESS
                # of which (if any) of the skip counters above explain it.
                # This is what the original bug violated: a real corridor
                # run's device-mismatch exception escaped every single skip
                # counter that existed at the time (none of them were
                # `views_skipped_error`, because that counter did not exist),
                # so the four conditions above were all individually False
                # while n_cells_observed was 0 at all 53 of its telemetry
                # ticks -- a self-contradictory "measured nothing, measurement
                # complete" record that a reader (or a future exception class
                # not yet anticipated by any named counter) could reproduce
                # again. Folding `len(cells) == 0` directly into this
                # computation makes the pairing structurally impossible to
                # emit again, independent of whether the actual failure mode
                # is one this module's authors thought to name.
                or len(cells) == 0
            ),
            # Establishment evidence actually used, recorded so a reader never
            # has to guess which subset of the design doc's candidate list ran.
            "establishment_policy": {
                "min_eval_samples": pol.min_eval_samples,
                "min_views": pol.min_views,
                "min_ticks": pol.min_ticks,
                "optimization_exposure": "proxy:qualifying_ticks",
            },
            "worst_drop_from_peak_db": worst_drop,
            "worst_drop_cell_key": worst_drop_key,
            "cells": cells,
        }


    # -- co-visibility (Level B evidence, geometry only) --------------------
    def covisibility_snapshot(self) -> dict:
        """``{cam_key: [cell_key, ...]}`` over every frozen cell map so far.

        Issue #358 Level B needs one evidence source that pure adjacency
        cannot supply: whether a *single holdout camera actually saw through*
        the gap between two nearby-but-not-touching observed cells. That is
        what distinguishes physical connectivity from Euclidean proximity (two
        surfaces either side of a thin wall are close in XYZ but no camera's
        depth lands in both).

        Derived exclusively from the FROZEN, sensor-depth-derived cell maps --
        no PSNR, no residual, nothing from the reconstruction. Keyed on
        ``cam_key`` ONLY, deliberately not on ``(cam_key, H, W)``: a
        mid-run resolution change makes Level A cache a second
        ``CameraCellMap`` for the same physical camera, and treating those as
        two observers would inflate co-visibility weight with a camera that
        does not exist.

        Cheap and idempotent (a set-union over already-computed maps), so the
        caller can just rewrite the snapshot whenever it likes.
        """
        out: Dict[str, set] = {}
        for cm in self._cell_maps.values():
            keys = out.setdefault(cm.cam_key, set())
            idx = cm.slot_index[cm.valid_mask]
            # One sync per camera over a small unique set, not per pixel.
            for slot in set(int(s) for s in idx.unique().detach().cpu().tolist()):
                keys.add(int(self.grid.key_of(slot)))
        return {k: sorted(v) for k, v in out.items()}


def write_covisibility_snapshot(model_path: str, snapshot: dict, cell_size: float,
                                filename: str = "local_quality_covisibility.json") -> str:
    """Atomically (re)write the co-visibility snapshot next to the history.

    Rewritten rather than appended: the snapshot is cumulative by construction,
    so the newest one supersedes every earlier one.
    """
    path = os.path.join(model_path, filename)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"cell_size_m": float(cell_size), "cams": snapshot}, f)
    os.replace(tmp, path)
    return path


def read_covisibility_snapshot(model_path: str,
                               filename: str = "local_quality_covisibility.json"):
    """Returns ``(cams_dict, cell_size)`` or ``(None, None)`` when absent."""
    path = os.path.join(model_path, filename)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    cams = {str(k): [int(v) for v in vs] for k, vs in (d.get("cams") or {}).items()}
    return cams, d.get("cell_size_m")


def append_local_quality_history(model_path: str, record: dict,
                                 filename: str = "local_quality_history.jsonl") -> str:
    path = os.path.join(model_path, filename)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def read_local_quality_history(model_path: str,
                               filename: str = "local_quality_history.jsonl") -> List[dict]:
    path = os.path.join(model_path, filename)
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    warn_measurement_contradictions(out, source=path)
    return out


def find_measurement_contradictions(records: List[dict]) -> List[int]:
    """Iterations where ``n_cells_observed == 0`` but ``measurement_incomplete``
    is falsy -- the exact self-contradictory pairing issue #358 root-caused
    (a real corridor run reported this at all 53/53 of its telemetry ticks,
    because a device-mismatch exception escaped every skip counter that
    existed at the time). ``record_tick`` now makes this pairing structurally
    impossible to emit (see its own docstring/comment), so this exists to
    audit history files written by an older build -- or by any future code
    path this module's authors did not anticipate -- not as the primary
    defense.

    Read-only, no side effects. Returns the offending ``iteration`` values,
    in file order, tolerating malformed/partial records (missing keys count
    as neither observed nor incomplete, i.e. still a violation if cells==0).
    """
    bad = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if int(rec.get("n_cells_observed", 0)) == 0 and not rec.get("measurement_incomplete", False):
            bad.append(rec.get("iteration"))
    return bad


def warn_measurement_contradictions(records: List[dict], source: str = "<unknown>") -> List[int]:
    """Print a loud, one-shot warning if ``records`` contains the issue #358
    self-contradiction (see :func:`find_measurement_contradictions`).

    Cheap (one pass, no torch, no I/O beyond what the caller already did) and
    deliberately non-fatal -- a report script should still be able to work
    with the rest of a run's evidence, but a reader must never be able to
    miss this the way it was missed on ``issue358_floor3_local_quality_cells_
    cells_on`` (measured zero cells at every tick while reporting the tick as
    complete, and nothing printed or counted it).
    """
    bad = find_measurement_contradictions(records)
    if bad:
        preview = ", ".join(str(i) for i in bad[:8])
        more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        print(
            f"[local_quality] WARNING: {len(bad)}/{len(records)} record(s) in "
            f"{source} claim n_cells_observed=0 with measurement_incomplete="
            f"False -- this is 'NOT MEASURED', not 'no forgetting' (issue "
            f"#358). Iterations: {preview}{more}. Do not read a zero-drop "
            f"tick here as a healthy result.",
            flush=True,
        )
    return bad
