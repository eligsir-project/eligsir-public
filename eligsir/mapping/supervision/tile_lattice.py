"""Fixed supervision-tile lattice and cheap per-tile evidence (issues #310, #611).

This is the single artefact that **#310 items 4-5 and #611 Phase 0 both
specify** — per-region loss maps and tiled evidence summaries on one side,
"a small fixed grid per frame, record RGB/depth loss contribution and valid
pixels per tile" on the other. Building it once, here, is why the MVP for both
tracks is one deliverable rather than two (see
``docs/architecture/2026-08-29-tile-crop-supervision-implementation-plan.md``
section 1.1).

**Naming.** A *supervision tile* is a rectangle of the supervision image. It is
unrelated to gsplat's internal 16x16 *rasterization* tile (``pipe.tile_size``,
and ``scheduler/tier_diagnostics.py``'s ``tile_count``/``isects_per_tile``).
Nothing here is ever called a bare ``tile``.

**Cost discipline (normative, section 6 of the plan doc).**

* No extra rasterization, ever. The reduction consumes a residual the training
  step already computed and was about to discard.
* No per-iteration device->host transfer. :func:`reduce_to_supervision_tiles`
  returns an ``R x C`` tensor that stays on device; the caller transfers it only
  on iterations that already synchronise for scalar logging.
* Bounded: ``MAX_LATTICE_DIM`` caps the grid, so the summary is O(1) in image
  size.

**Scope.** Measurement only. Nothing here ranks, selects, promotes or weights a
tile, and no scheduler or allocator may read its output — the same rule
``scheduler/tier_diagnostics.py`` states for itself, and the reason this
evidence lives in its own module rather than being appended there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from eligsir.mapping.supervision.crop_camera import CropTransform

#: Hard cap on either lattice dimension. A supervision lattice is a coarse
#: evidence grid (#611 suggests 4x4 or 8x8), not a per-pixel map; the cap is
#: what keeps the summary O(1) against image size and keeps a misconfiguration
#: from silently turning the probe into a full-resolution D2H.
MAX_LATTICE_DIM = 16


@dataclass(frozen=True)
class SupervisionTileLattice:
    """A fixed ``rows x cols`` partition of a ``width x height`` image.

    The partition is exact and exhaustive: every pixel belongs to exactly one
    supervision tile, and remainder pixels from a non-divisible split are
    distributed to the leading tiles rather than dropped. That property is
    asserted by the unit tests — a lattice that silently loses an edge strip
    would bias every per-tile statistic computed from it.
    """

    rows: int
    cols: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not (1 <= int(self.rows) <= MAX_LATTICE_DIM) or not (1 <= int(self.cols) <= MAX_LATTICE_DIM):
            raise ValueError(
                f"lattice dims must be in [1, {MAX_LATTICE_DIM}], got {self.rows}x{self.cols}"
            )
        if int(self.width) < int(self.cols) or int(self.height) < int(self.rows):
            raise ValueError(
                f"image {self.width}x{self.height} is too small for a {self.rows}x{self.cols} lattice"
            )

    @classmethod
    def square(cls, grid: int, *, width: int, height: int) -> "SupervisionTileLattice":
        return cls(rows=int(grid), cols=int(grid), width=int(width), height=int(height))

    @property
    def tile_count(self) -> int:
        return int(self.rows) * int(self.cols)

    @staticmethod
    def _bounds(total: int, parts: int, index: int) -> Tuple[int, int]:
        """Start/size of split ``index`` of ``total`` into ``parts``.

        Remainder pixels go to the leading splits, one each, so the splits
        cover ``total`` exactly.
        """
        base, rem = divmod(int(total), int(parts))
        start = index * base + min(index, rem)
        size = base + (1 if index < rem else 0)
        return start, size

    def tile_rect(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """``(x0, y0, width, height)`` of one supervision tile."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"tile ({row}, {col}) outside {self.rows}x{self.cols} lattice")
        y0, h = self._bounds(self.height, self.rows, row)
        x0, w = self._bounds(self.width, self.cols, col)
        return x0, y0, w, h

    def crop_transform(self, row: int, col: int) -> CropTransform:
        """One supervision tile expressed as an exact crop of the parent.

        This is the bridge between the two tracks: a lattice cell and a
        promoted crop are the same geometric object, so a tile identified as
        valuable can be handed straight to ``crop_camera_from`` without any
        second geometry implementation (#610's "reuse #310 crop-camera math").
        """
        x0, y0, w, h = self.tile_rect(row, col)
        return CropTransform(
            x0=x0, y0=y0, width=w, height=h,
            out_width=w, out_height=h,
            parent_width=int(self.width), parent_height=int(self.height),
        )


@dataclass(frozen=True)
class SupervisionTileEvidence:
    """Level-A per-supervision-tile evidence for one supervision view.

    "Level A" is #611's own term for loss contribution alone — explicitly *not*
    sufficient to call a region useful. Named that way here so a future reader
    cannot mistake it for the influence signal #612 is gated on.

    Values are row-major ``rows * cols`` tuples of Python floats, materialised
    only after the caller has decided to pay a host transfer.
    """

    rows: int
    cols: int
    rgb_l1: Tuple[float, ...]
    valid_fraction: Tuple[float, ...]
    depth_l1: Optional[Tuple[float, ...]] = None

    def as_scalar_dict(self, prefix: str = "supervision_tile") -> dict:
        """Flat TB-ready scalars. Keys carry the lattice shape, not just an index."""
        out: dict = {}
        for i in range(self.rows * self.cols):
            r, c = divmod(i, self.cols)
            cell = f"{self.rows}x{self.cols}/r{r}c{c}"
            out[f"{prefix}/rgb_l1/{cell}"] = float(self.rgb_l1[i])
            out[f"{prefix}/valid_fraction/{cell}"] = float(self.valid_fraction[i])
            if self.depth_l1 is not None:
                out[f"{prefix}/depth_l1/{cell}"] = float(self.depth_l1[i])
        # Dispersion across tiles is the one aggregate worth logging every time:
        # a flat map means the frame is spatially uniform and there is nothing
        # for any future spatial policy to act on.
        if self.rgb_l1:
            mean = sum(self.rgb_l1) / len(self.rgb_l1)
            out[f"{prefix}/rgb_l1_max"] = float(max(self.rgb_l1))
            out[f"{prefix}/rgb_l1_mean"] = float(mean)
            out[f"{prefix}/rgb_l1_max_over_mean"] = float(max(self.rgb_l1) / mean) if mean > 0 else 0.0
        return out


def reduce_to_supervision_tiles(residual, lattice: SupervisionTileLattice):
    """Mean of ``residual`` over each supervision tile, **on device**.

    ``residual`` is ``[H, W]`` or ``[C, H, W]``; a channel dimension is averaged
    first. Returns an ``[rows, cols]`` tensor. No host transfer, no extra
    render, no gradient — the input is detached.

    The reduction is exact for a divisible split and, for a non-divisible one,
    matches :meth:`SupervisionTileLattice.tile_rect`'s remainder-to-leading-tiles
    partition, so a tile's reported statistic is the mean over exactly the
    pixels ``tile_rect`` says belong to it. ``adaptive_avg_pool2d`` is
    deliberately NOT used: its bin boundaries are a different (overlapping)
    partition, which would make the evidence disagree with the crop geometry the
    same lattice hands to ``crop_camera_from``.
    """
    import torch

    if residual.dim() == 3:
        residual = residual.mean(dim=0)
    elif residual.dim() != 2:
        raise ValueError(f"residual must be [H, W] or [C, H, W], got shape {tuple(residual.shape)}")
    height, width = int(residual.shape[0]), int(residual.shape[1])
    if height != int(lattice.height) or width != int(lattice.width):
        raise ValueError(
            f"residual {width}x{height} does not match lattice {lattice.width}x{lattice.height}"
        )
    values = residual.detach().float()
    out = torch.empty((lattice.rows, lattice.cols), dtype=torch.float32, device=values.device)
    for r in range(lattice.rows):
        y0, h = lattice._bounds(height, lattice.rows, r)
        for c in range(lattice.cols):
            x0, w = lattice._bounds(width, lattice.cols, c)
            out[r, c] = values[y0 : y0 + h, x0 : x0 + w].mean()
    return out


def collect_tile_evidence(
    *,
    image,
    gt_image,
    lattice: SupervisionTileLattice,
    depth_render=None,
    depth_gt=None,
    depth_valid=None,
) -> SupervisionTileEvidence:
    """Level-A evidence for one supervision view. Performs ONE host transfer.

    Call only on an iteration that already synchronises (e.g. a scalar-logging
    step); every tensor read here is one the training step already produced.
    """
    import torch

    rgb_residual = (image.detach() - gt_image.detach()).abs()
    rgb = reduce_to_supervision_tiles(rgb_residual, lattice)

    if depth_valid is not None:
        valid = reduce_to_supervision_tiles(depth_valid.detach().float(), lattice)
    else:
        valid = torch.ones_like(rgb)

    depth = None
    if depth_render is not None and depth_gt is not None:
        d_res = (depth_render.detach() - depth_gt.detach()).abs()
        if depth_valid is not None:
            d_res = d_res * depth_valid.detach().float()
        depth = reduce_to_supervision_tiles(d_res, lattice)

    stacked = [rgb.reshape(-1), valid.reshape(-1)] + ([depth.reshape(-1)] if depth is not None else [])
    # ONE transfer for every channel, rather than one per channel.
    host = torch.stack(stacked, dim=0).cpu().tolist()
    return SupervisionTileEvidence(
        rows=lattice.rows,
        cols=lattice.cols,
        rgb_l1=tuple(float(v) for v in host[0]),
        valid_fraction=tuple(float(v) for v in host[1]),
        depth_l1=tuple(float(v) for v in host[2]) if depth is not None else None,
    )
