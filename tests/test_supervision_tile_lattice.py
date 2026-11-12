"""Supervision-tile lattice and Level-A evidence (#310 items 4-5 / #611 Phase 0).

Two properties matter and are both asserted here:

1. the lattice partition is **exact and exhaustive** -- a lattice that silently
   drops an edge strip on a non-divisible split would bias every per-tile
   statistic computed from it;
2. the reduction's tile means are taken over **exactly** the pixels
   `tile_rect` claims, so the evidence and the crop geometry derived from the
   same lattice cannot disagree.

The lattice-geometry tests are torch-free and run in the lean CPU CI image; the
reduction tests need torch and are skipped without it.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.supervision.tile_lattice import (
    MAX_LATTICE_DIM,
    SupervisionTileEvidence,
    SupervisionTileLattice,
)

torch = pytest.importorskip("torch", reason="tensor reduction needs torch")

from eligsir.mapping.supervision.tile_lattice import (  # noqa: E402
    collect_tile_evidence,
    reduce_to_supervision_tiles,
)


# --------------------------------------------------------------------------
# Lattice geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("width,height", [(640, 480), (641, 481), (100, 7), (17, 33)])
@pytest.mark.parametrize("grid", [1, 2, 4, 8])
def test_partition_is_exact_and_exhaustive(width, height, grid):
    if width < grid or height < grid:
        pytest.skip("image too small for this lattice")
    lat = SupervisionTileLattice.square(grid, width=width, height=height)
    covered = set()
    total = 0
    for r in range(lat.rows):
        for c in range(lat.cols):
            x0, y0, w, h = lat.tile_rect(r, c)
            total += w * h
            for y in range(y0, y0 + h):
                for x in range(x0, x0 + w):
                    assert (x, y) not in covered, "supervision tiles overlap"
                    covered.add((x, y))
    assert total == width * height
    assert len(covered) == width * height


def test_remainder_goes_to_leading_tiles():
    lat = SupervisionTileLattice(rows=1, cols=3, width=10, height=4)
    assert [lat.tile_rect(0, c)[2] for c in range(3)] == [4, 3, 3]
    assert [lat.tile_rect(0, c)[0] for c in range(3)] == [0, 4, 7]


def test_lattice_rejects_out_of_range_dims():
    with pytest.raises(ValueError):
        SupervisionTileLattice.square(0, width=64, height=64)
    with pytest.raises(ValueError):
        SupervisionTileLattice.square(MAX_LATTICE_DIM + 1, width=640, height=480)


def test_lattice_rejects_an_image_smaller_than_the_grid():
    with pytest.raises(ValueError):
        SupervisionTileLattice.square(8, width=4, height=4)


def test_tile_rect_bounds_checked():
    lat = SupervisionTileLattice.square(2, width=64, height=64)
    with pytest.raises(IndexError):
        lat.tile_rect(2, 0)


def test_crop_transform_matches_tile_rect():
    """The bridge between tracks: a lattice cell IS a crop, with no second
    geometry implementation (#610's "reuse #310 crop-camera math")."""
    lat = SupervisionTileLattice(rows=3, cols=4, width=641, height=481)
    for r in range(3):
        for c in range(4):
            x0, y0, w, h = lat.tile_rect(r, c)
            t = lat.crop_transform(r, c)
            assert (t.x0, t.y0, t.width, t.height) == (x0, y0, w, h)
            assert (t.out_width, t.out_height) == (w, h)   # a lattice cell never rescales
            assert (t.parent_width, t.parent_height) == (641, 481)


def test_tile_count():
    assert SupervisionTileLattice(rows=3, cols=5, width=64, height=64).tile_count == 15


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------

def test_reduction_matches_a_manual_per_tile_mean():
    lat = SupervisionTileLattice(rows=3, cols=4, width=61, height=47)
    g = torch.Generator().manual_seed(0)
    residual = torch.rand(47, 61, generator=g)
    got = reduce_to_supervision_tiles(residual, lat)
    assert tuple(got.shape) == (3, 4)
    for r in range(3):
        for c in range(4):
            x0, y0, w, h = lat.tile_rect(r, c)
            expected = residual[y0:y0 + h, x0:x0 + w].mean()
            assert got[r, c].item() == pytest.approx(expected.item(), rel=1e-6)


def test_reduction_of_a_constant_map_is_that_constant_everywhere():
    """Normalisation is per-tile-mean, so it is independent of tile size --
    an unequal split must not make edge tiles read differently."""
    lat = SupervisionTileLattice(rows=3, cols=3, width=10, height=10)
    got = reduce_to_supervision_tiles(torch.full((10, 10), 0.25), lat)
    assert torch.allclose(got, torch.full((3, 3), 0.25))


def test_reduction_averages_channels():
    lat = SupervisionTileLattice.square(2, width=8, height=8)
    chw = torch.stack([torch.zeros(8, 8), torch.ones(8, 8), torch.full((8, 8), 2.0)])
    assert torch.allclose(reduce_to_supervision_tiles(chw, lat), torch.ones(2, 2))


def test_reduction_rejects_a_shape_mismatch():
    lat = SupervisionTileLattice.square(2, width=8, height=8)
    with pytest.raises(ValueError):
        reduce_to_supervision_tiles(torch.zeros(16, 16), lat)
    with pytest.raises(ValueError):
        reduce_to_supervision_tiles(torch.zeros(1, 3, 8, 8), lat)


def test_reduction_does_not_propagate_gradient():
    lat = SupervisionTileLattice.square(2, width=8, height=8)
    residual = torch.rand(8, 8, requires_grad=True)
    assert not reduce_to_supervision_tiles(residual, lat).requires_grad


# --------------------------------------------------------------------------
# Level-A evidence
# --------------------------------------------------------------------------

def test_collect_tile_evidence_localises_a_planted_error():
    lat = SupervisionTileLattice.square(2, width=8, height=8)
    gt = torch.zeros(3, 8, 8)
    img = torch.zeros(3, 8, 8)
    img[:, 0:4, 4:8] = 1.0            # top-right supervision tile only
    ev = collect_tile_evidence(image=img, gt_image=gt, lattice=lat)
    assert ev.rgb_l1 == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert ev.depth_l1 is None
    assert ev.valid_fraction == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_collect_tile_evidence_with_depth_and_validity():
    lat = SupervisionTileLattice.square(2, width=4, height=4)
    gt = torch.zeros(3, 4, 4)
    img = torch.zeros(3, 4, 4)
    d_gt = torch.zeros(4, 4)
    d_rend = torch.ones(4, 4)
    valid = torch.zeros(4, 4)
    valid[0:2, 0:2] = 1.0             # only the top-left tile has valid depth
    ev = collect_tile_evidence(
        image=img, gt_image=gt, lattice=lat,
        depth_render=d_rend, depth_gt=d_gt, depth_valid=valid,
    )
    assert ev.valid_fraction == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert ev.depth_l1 == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_scalar_dict_keys_are_shape_qualified_and_never_a_bare_tile():
    ev = SupervisionTileEvidence(rows=1, cols=2, rgb_l1=(0.0, 1.0), valid_fraction=(1.0, 1.0))
    d = ev.as_scalar_dict()
    assert "supervision_tile/rgb_l1/1x2/r0c1" in d
    assert d["supervision_tile/rgb_l1_max"] == pytest.approx(1.0)
    assert d["supervision_tile/rgb_l1_mean"] == pytest.approx(0.5)
    assert d["supervision_tile/rgb_l1_max_over_mean"] == pytest.approx(2.0)
    # Every key must be namespaced so it cannot alias gsplat's raster-tile
    # telemetry (`tile_count`, `isects_per_tile`, ...).
    assert all(k.startswith("supervision_tile/") for k in d)


def test_scalar_dict_handles_a_zero_residual_without_dividing_by_zero():
    ev = SupervisionTileEvidence(rows=1, cols=2, rgb_l1=(0.0, 0.0), valid_fraction=(1.0, 1.0))
    assert ev.as_scalar_dict()["supervision_tile/rgb_l1_max_over_mean"] == 0.0
