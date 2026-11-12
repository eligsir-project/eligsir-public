"""Level B adaptive regions (issue #358). CPU-only, no GPU, no torch needed.

Covers, in the order the PR argues them:

* grouping is a pure function of geometry (adjacency from persisted keys,
  co-visibility from the frozen sensor-depth maps) -- and is PROVABLY
  PSNR-independent (a test scrambles every PSNR and asserts the partition,
  lineage and events are bit-identical);
* region curves are DERIVED, not accumulated: a split's children reproduce
  exactly what they would have had if the split had existed from tick 0, and
  their SSE sums back to the parent's;
* split/merge fire on real topology, wait out hysteresis, and never renumber;
* every failure mode is loud (raise / explicit flag), never a silent zero.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eligsir.utils.local_quality import cell_key, pooled_psnr_db  # noqa: E402
from eligsir.utils.local_regions import (  # noqa: E402
    CellGraph, RegionPartitioner, RegionPolicy, derive_region_history,
    erosion_cut, floor_cut, sse_from_record, write_region_history,
)

CELL = 0.5


def key(i, j, k):
    return int(cell_key([i, j, k]))


def line(n, j=0, k=0, i0=0):
    return [key(i0 + i, j, k) for i in range(n)]


def blob(i0, ni, nj=3, nk=1):
    return [key(i0 + i, j, k) for i in range(ni) for j in range(nj) for k in range(nk)]


def dumbbell():
    """Two 4x3 lobes joined by a single-cell neck -- the canonical
    'two dense lobes through a narrow persistent bottleneck'."""
    left = blob(0, 4)
    right = blob(7, 4)
    neck = [key(4, 1, 0), key(5, 1, 0), key(6, 1, 0)]
    return left, right, neck


def cell_rec(k, psnr, n=900, views=3, established=True):
    return {"cell_key": k, "n_eval_samples": n, "n_views": views,
            "psnr_db": psnr, "established": established, "qualifying_ticks": 5,
            "first_seen_iter": 0}


def tick(iteration, keys, psnr=25.0, established=True, **extra):
    cells = {str(k): cell_rec(k, psnr if not callable(psnr) else psnr(k),
                              established=established) for k in keys}
    rec = {"iteration": iteration, "cell_size_m": CELL, "n_cells_observed": len(cells),
           "n_cells_established": len(cells) if established else 0,
           "measurement_incomplete": False, "cells": cells}
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Graph: adjacency + co-visibility
# ---------------------------------------------------------------------------
class TestGraph:
    def test_adjacency_is_face_only_and_symmetric(self):
        g = CellGraph(CELL)
        g.add_cells([key(0, 0, 0), key(1, 0, 0), key(1, 1, 0)])
        assert set(g.neighbors(key(0, 0, 0))) == {key(1, 0, 0)}
        # diagonal (corner-touching) is NOT an edge
        assert key(1, 1, 0) not in g.neighbors(key(0, 0, 0))
        assert key(0, 0, 0) in g.neighbors(key(1, 0, 0))

    def test_gap_splits_components_without_covisibility(self):
        g = CellGraph(CELL)
        cells = line(3) + line(3, i0=6)
        g.add_cells(cells)
        assert len(g.components(set(cells))) == 2

    def test_covisibility_bridges_a_two_cell_gap(self):
        """A camera whose fixed sensor depth landed in both sides bridges a
        hole in the observed-cell set that pure adjacency leaves open."""
        a, b = key(0, 0, 0), key(2, 0, 0)
        g = CellGraph(CELL, RegionPolicy(covis_max_gap_cells=2),
                      covisibility={"cam0": [a, b]})
        g.add_cells([a, b])
        assert b in g.neighbors(a)
        assert len(g.components({a, b})) == 1

    def test_covisibility_needs_a_shared_camera(self):
        a, b = key(0, 0, 0), key(2, 0, 0)
        g = CellGraph(CELL, RegionPolicy(covis_max_gap_cells=2),
                      covisibility={"cam0": [a], "cam1": [b]})
        g.add_cells([a, b])
        assert b not in g.neighbors(a)      # close in XYZ, never co-observed
        assert len(g.components({a, b})) == 2

    def test_covisibility_can_be_ablated(self):
        a, b = key(0, 0, 0), key(2, 0, 0)
        g = CellGraph(CELL, RegionPolicy(use_covisibility=False, covis_max_gap_cells=2),
                      covisibility={"cam0": [a, b]})
        g.add_cells([a, b])
        assert len(g.components({a, b})) == 2


# ---------------------------------------------------------------------------
# Cut detection
# ---------------------------------------------------------------------------
class TestCuts:
    def test_erosion_finds_the_dumbbell_neck(self):
        left, right, neck = dumbbell()
        cells = set(left + right + neck)
        g = CellGraph(CELL)
        g.add_cells(cells)
        cut = erosion_cut(g, cells, RegionPolicy(split_min_lobe_cells=5))
        assert cut is not None
        lobes, sep = cut
        assert len(lobes) == 2
        # Each 4x3 lobe keeps its 8 degree>=3 cells; its 4 corners (degree 2)
        # erode along with the 3 neck cells and become the separator, which the
        # split step reattaches to the lobe it touches most.
        assert {len(l) for l in lobes} == {8, 8}
        assert set(neck) <= set(sep) and len(sep) == 11

    def test_no_cut_in_a_solid_blob(self):
        cells = set(blob(0, 10, nj=4))
        g = CellGraph(CELL)
        g.add_cells(cells)
        assert erosion_cut(g, cells, RegionPolicy()) is None

    def test_floor_cut_requires_explicit_up_axis(self):
        lower = [key(i, j, 0) for i in range(4) for j in range(3)]
        upper = [key(i, j, 5) for i in range(4) for j in range(3)]
        cells = set(lower + upper)
        g = CellGraph(CELL)
        g.add_cells(cells)
        assert floor_cut(g, cells, RegionPolicy(up_axis=None)) is None, \
            "must never guess an up axis"
        cut = floor_cut(g, cells, RegionPolicy(up_axis=2, split_min_lobe_cells=5))
        assert cut is not None
        lobes, sep = cut
        assert sorted(len(l) for l in lobes) == [12, 12] and not sep


# ---------------------------------------------------------------------------
# Partitioner: growth, split, merge, lineage
# ---------------------------------------------------------------------------
class TestPartitioner:
    def test_disconnected_cells_open_separate_regions(self):
        p = RegionPartitioner(CELL)
        p.update(0, 0, line(3) + line(3, i0=6))
        assert len(p.active_regions) == 2
        assert {r.created_by for r in p.active_regions} == {"seed", "frontier"}

    def test_new_cells_join_without_renumbering_existing_regions(self):
        p = RegionPartitioner(CELL)
        p.update(0, 0, line(3))
        rid = p.active_regions[0].region_id
        p.update(1, 100, line(6))
        assert [r.region_id for r in p.active_regions] == [rid]
        assert len(p.active_regions[0].cells) == 6

    def test_distant_new_cells_create_a_region_and_leave_ids_alone(self):
        p = RegionPartitioner(CELL)
        p.update(0, 0, line(3))
        before = {r.region_id: set(r.cells) for r in p.active_regions}
        p.update(1, 100, line(3) + line(3, i0=20))
        after = {r.region_id: set(r.cells) for r in p.active_regions}
        for rid, cells in before.items():
            assert after[rid] == cells, "an existing region must not be disturbed"
        assert len(after) == 2

    def test_split_waits_out_hysteresis_then_records_lineage(self):
        left, right, neck = dumbbell()
        cells = left + right + neck
        pol = RegionPolicy(split_min_cells=20, split_min_lobe_cells=5,
                           split_hysteresis_ticks=3)
        p = RegionPartitioner(CELL, pol)
        outs = [p.update(t, t * 1000, cells) for t in range(3)]
        assert [list(o["split_states"].values()) for o in outs] == [
            ["candidate_pending"], ["candidate_pending"], ["executed"]]
        ev = [e for e in outs[2]["events"] if e["event"] == "split"]
        assert len(ev) == 1 and ev[0]["separator_cells"] > 0

        retired = [r for r in p.regions.values() if not r.active]
        assert len(retired) == 1, "exactly one region was retired by the split"
        par = retired[0]
        assert par.retired_by == "split" and len(par.child_ids) == 2
        kids = [p.regions[c] for c in par.child_ids]
        assert {k.label for k in kids} == {f"{par.label}.0", f"{par.label}.1"}
        assert all(k.parent_ids == [par.region_id] for k in kids)
        # children partition the parent exactly (separator assigned, not dropped)
        assert set().union(*[k.cells for k in kids]) == set(cells)
        assert not (kids[0].cells & kids[1].cells)

    def test_a_changing_cut_resets_the_hysteresis_counter(self):
        """A cut that regroups the cells differently is not the same cut, so
        the dwell must restart -- this is what stops split thrash."""
        left, right, neck = dumbbell()
        pol = RegionPolicy(split_min_cells=20, split_min_lobe_cells=5,
                           split_hysteresis_ticks=3, split_max_bbox_diag_m=1e9)
        p = RegionPartitioner(CELL, pol)
        base = left + right + neck
        assert list(p.update(0, 0, base)["split_states"].values()) == ["candidate_pending"]
        assert list(p.update(1, 1000, base)["split_states"].values()) == ["candidate_pending"]
        # The left lobe grows backwards, so the cut's left-lobe minimum key
        # moves: a different cut, hence a restarted dwell instead of the
        # execution the third consecutive identical cut would have earned.
        grown = base + blob(-3, 3)
        assert list(p.update(2, 2000, grown)["split_states"].values()) == ["candidate_pending"]
        assert list(p.update(3, 3000, grown)["split_states"].values()) == ["candidate_pending"]
        assert list(p.update(4, 4000, grown)["split_states"].values()) == ["executed"]

    def test_split_does_not_fire_below_the_size_trigger(self):
        left, right, neck = dumbbell()
        p = RegionPartitioner(CELL, RegionPolicy(split_min_cells=10_000,
                                                 split_max_bbox_diag_m=1e9,
                                                 split_hysteresis_ticks=1))
        outs = [p.update(t, t, left + right + neck) for t in range(5)]
        assert len(p.active_regions) == 1
        assert list(outs[-1]["split_states"].values()) == ["not_evaluated"]

    def test_split_state_distinguishes_no_cut_from_pending(self):
        p = RegionPartitioner(CELL, RegionPolicy(split_min_cells=5,
                                                 split_hysteresis_ticks=99))
        out = p.update(0, 0, blob(0, 10, nj=4))
        assert list(out["split_states"].values()) == ["no_cut_found"]
        left, right, neck = dumbbell()
        p2 = RegionPartitioner(CELL, RegionPolicy(split_min_cells=5,
                                                  split_min_lobe_cells=5,
                                                  split_hysteresis_ticks=99))
        out2 = p2.update(0, 0, left + right + neck)
        assert list(out2["split_states"].values()) == ["candidate_pending"]

    def test_bridging_two_small_regions_is_an_explicit_merge_event(self):
        pol = RegionPolicy(merge_max_cells=10, merge_hysteresis_ticks=2,
                           split_min_cells=10_000, split_max_bbox_diag_m=1e9)
        p = RegionPartitioner(CELL, pol)
        a, b = line(2), line(2, i0=5)
        p.update(0, 0, a + b)
        ids_before = sorted(r.region_id for r in p.active_regions)
        assert len(ids_before) == 2
        bridged = a + b + [key(2, 0, 0), key(3, 0, 0), key(4, 0, 0)]
        p.update(1, 100, bridged)          # bridge observed once -> dwell
        assert len(p.active_regions) == 2
        out = p.update(2, 200, bridged)    # dwell satisfied -> merge
        assert len(p.active_regions) == 1
        ev = [e for e in out["events"] if e["event"] == "merge"]
        assert len(ev) == 1
        assert sorted(ev[0]["parent_ids"]) == ids_before
        merged = p.active_regions[0]
        assert merged.region_id not in ids_before, "merge must not reuse an id"
        assert sorted(merged.parent_ids) == ids_before
        assert all(not p.regions[i].active for i in ids_before)

    def test_two_large_connected_regions_are_not_merged(self):
        """Conservative by design: no positive evidence two big regions are one
        room, and merging them would immediately re-trigger a size split."""
        pol = RegionPolicy(merge_max_cells=4, merge_hysteresis_ticks=1,
                           split_min_cells=10_000, split_max_bbox_diag_m=1e9)
        p = RegionPartitioner(CELL, pol)
        a, b = blob(0, 4), blob(10, 4)
        p.update(0, 0, a + b)
        assert len(p.active_regions) == 2
        joined = a + b + blob(4, 6)
        for t in range(1, 6):
            p.update(t, t * 10, joined)
        assert len(p.active_regions) == 2

    def test_partition_version_only_moves_on_topology_change(self):
        p = RegionPartitioner(CELL, RegionPolicy(split_min_cells=10_000,
                                                 split_max_bbox_diag_m=1e9))
        v0 = p.update(0, 0, line(3))["partition_version"]
        v1 = p.update(1, 10, line(3))["partition_version"]
        assert v1 == v0, "an unchanged tick must not bump the partition version"
        v2 = p.update(2, 20, line(3) + line(2, i0=20))["partition_version"]
        assert v2 == v0 + 1


# ---------------------------------------------------------------------------
# Derived histories
# ---------------------------------------------------------------------------
class TestDerivedHistory:
    def test_sse_inversion_is_exact(self):
        for psnr in (5.0, 18.3, 25.0, 41.7):
            n = 3 * 1234
            sse, nn = sse_from_record({"n_eval_samples": n, "psnr_db": psnr})
            assert nn == n
            back = float(pooled_psnr_db([sse], [n])[0])
            assert abs(back - psnr) < 1e-9

    def test_region_psnr_pools_its_cells(self):
        keys = line(3)
        recs = [tick(0, keys, psnr=lambda k: 20.0 + (k % 7)),
                tick(1000, keys, psnr=lambda k: 20.0 + (k % 7))]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000,
                                                     split_max_bbox_diag_m=1e9))
        row = list(d["ticks"][-1]["regions"].values())[0]
        sse = sum(sse_from_record(c)[0] for c in recs[-1]["cells"].values())
        n = sum(sse_from_record(c)[1] for c in recs[-1]["cells"].values())
        assert abs(row["psnr_db"] - float(pooled_psnr_db([sse], [n])[0])) < 1e-9

    def test_split_children_have_no_manufactured_discontinuity(self):
        """The decisive property. A child's curve after a split is IDENTICAL to
        the curve it would have had if that child had existed from tick 0 --
        no inherited parent peak (fake drop) and no fresh start
        (discontinuity)."""
        left, right, neck = dumbbell()
        cells = left + right + neck
        # left lobe degrades late; right lobe is flat.
        def psnr_at(t):
            return lambda k: (30.0 - 4.0 * (t >= 3)) if k in set(left) else 26.0
        recs = [tick(t * 1000, cells, psnr=psnr_at(t)) for t in range(6)]
        pol = RegionPolicy(split_min_cells=20, split_min_lobe_cells=5,
                           split_hysteresis_ticks=3, split_max_bbox_diag_m=1e9)
        d = derive_region_history(recs, pol)

        split_ticks = [t for t in d["ticks"] for e in t["events"] if e["event"] == "split"]
        assert len(split_ticks) == 1
        final = d["ticks"][-1]["regions"]
        assert len(final) == 2

        # Reference: run the two child memberships as independent scenes from
        # tick 0 and compare the derived curves.
        for row in final.values():
            members = d["membership"][row["region_id"]]
            ref_recs = [{"iteration": r["iteration"], "cell_size_m": CELL,
                         "measurement_incomplete": False,
                         "cells": {str(k): r["cells"][str(k)] for k in members}}
                        for r in recs]
            ref = derive_region_history(ref_recs, RegionPolicy(
                split_min_cells=10_000, split_max_bbox_diag_m=1e9))
            ref_row = list(ref["ticks"][-1]["regions"].values())[0]
            assert abs(ref_row["psnr_db"] - row["psnr_db"]) < 1e-9
            assert abs(ref_row["psnr_peak_db"] - row["psnr_peak_db"]) < 1e-9
            assert abs(ref_row["drop_from_peak_db"] - row["drop_from_peak_db"]) < 1e-9

        # And the degrading lobe is the one the metric flags.
        by_drop = sorted(final.values(), key=lambda r: -r["drop_from_peak_db"])
        worst, other = by_drop[0], by_drop[1]
        assert set(d["membership"][worst["region_id"]]) >= set(left)
        # The child holding the degrading lobe also absorbed the (flat) neck
        # cells as separator, so its pooled drop is smaller than the 4dB the
        # lobe alone lost -- pooling is over the child's real membership.
        assert 2.0 < worst["drop_from_peak_db"] < 4.0
        assert other["drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9)

    def test_children_sse_sums_to_parent_sse(self):
        left, right, neck = dumbbell()
        cells = left + right + neck
        recs = [tick(t * 1000, cells, psnr=lambda k: 20.0 + (k % 5)) for t in range(4)]
        pol = RegionPolicy(split_min_cells=20, split_min_lobe_cells=5,
                           split_hysteresis_ticks=3, split_max_bbox_diag_m=1e9)
        d = derive_region_history(recs, pol)
        parent_sse = sum(sse_from_record(c)[0] for c in recs[-1]["cells"].values())
        child_sse = 0.0
        for row in d["ticks"][-1]["regions"].values():
            for k in d["membership"][row["region_id"]]:
                child_sse += sse_from_record(recs[-1]["cells"][str(k)])[0]
        assert abs(child_sse - parent_sse) < 1e-9

    def test_frontier_region_cannot_report_forgetting(self):
        keys = line(3)
        recs = [tick(0, keys, psnr=30.0, established=False),
                tick(1000, keys, psnr=10.0, established=False)]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["established"] is False
        assert "drop_from_peak_db" not in row
        assert d["ticks"][-1]["worst_region_drop_from_peak_db"] == 0.0


# ---------------------------------------------------------------------------
# The non-circularity guarantee
# ---------------------------------------------------------------------------
class TestGroupingIsNotPsnrDriven:
    def test_scrambling_every_psnr_leaves_the_partition_bit_identical(self):
        import random
        left, right, neck = dumbbell()
        cells = left + right + neck
        pol = RegionPolicy(split_min_cells=20, split_min_lobe_cells=5,
                           split_hysteresis_ticks=3, split_max_bbox_diag_m=1e9)

        def topology(recs):
            d = derive_region_history(recs, pol)
            return json.dumps({
                "lineage": d["lineage"],
                "membership": {str(k): v for k, v in sorted(d["membership"].items())},
                "events": [t["events"] for t in d["ticks"]],
                "versions": [t["partition_version"] for t in d["ticks"]],
            }, sort_keys=True)

        base = [tick(t * 1000, cells, psnr=lambda k: 25.0) for t in range(6)]
        rng = random.Random(7)
        scrambled = json.loads(json.dumps(base))
        for r in scrambled:
            for c in r["cells"].values():
                c["psnr_db"] = rng.uniform(3.0, 45.0)
        assert topology(base) == topology(scrambled)

    def test_eval_sample_counts_are_counts_not_qualities(self):
        """The probe-count split trigger reads n_eval_samples, which is a pixel
        COUNT. Changing PSNR at fixed counts cannot change whether it fires."""
        cells = blob(0, 6)
        pol = RegionPolicy(split_min_cells=10_000, split_max_bbox_diag_m=1e9,
                           split_max_eval_samples=1)
        a = [tick(0, cells, psnr=5.0), tick(10, cells, psnr=5.0)]
        b = [tick(0, cells, psnr=44.0), tick(10, cells, psnr=44.0)]
        da, db = derive_region_history(a, pol), derive_region_history(b, pol)
        sa = [r["split_state"] for r in da["ticks"][-1]["regions"].values()]
        sb = [r["split_state"] for r in db["ticks"][-1]["regions"].values()]
        assert sa == sb != ["not_evaluated"]


# ---------------------------------------------------------------------------
# Loud failure
# ---------------------------------------------------------------------------
class TestLoudFailure:
    def test_cell_size_change_raises(self):
        a = tick(0, line(3))
        b = tick(1000, line(3))
        b["cell_size_m"] = 1.0
        with pytest.raises(ValueError, match="DIFFERENT partition"):
            derive_region_history([a, b])

    def test_empty_history_is_invalid_not_healthy(self):
        d = derive_region_history([])
        assert d["regions_valid"] is False and d["invalid_reason"]

    def test_tick_with_no_cells_is_not_measured(self):
        recs = [tick(0, line(3)), tick(1000, [])]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        last = d["ticks"][-1]
        assert all(r["measured"] is False for r in last["regions"].values())
        assert last["regions_valid"] is False
        assert "NOT MEASURED" in last["invalid_reason"]
        assert last["worst_region_drop_from_peak_db"] == 0.0

    def test_measurement_incomplete_propagates(self):
        recs = [tick(0, line(3)), tick(1000, line(3), measurement_incomplete=True)]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        assert d["ticks"][-1]["measurement_incomplete"] is True

    def test_degenerate_partition_is_flagged(self):
        keys = [key(0, 0, 0), key(10, 0, 0), key(20, 0, 0), key(30, 0, 0)]
        d = derive_region_history([tick(0, keys), tick(1, keys)],
                                  RegionPolicy(split_min_cells=10_000))
        assert d["ticks"][-1]["n_regions_active"] == 4
        assert d["ticks"][-1]["degenerate_partition"] is True

    def test_evidence_provenance_is_recorded(self):
        d = derive_region_history([tick(0, line(3))], RegionPolicy(split_min_cells=10_000))
        t = d["ticks"][-1]
        assert t["evidence_sources_used"] == ["3d_adjacency_6conn"]
        assert "mapped_surface_gaussian_continuity" in t["evidence_sources_skipped"]
        assert d["covisibility_available"] is False


class TestIO:
    def test_write_region_history_roundtrips(self, tmp_path):
        recs = [tick(0, line(3)), tick(1000, line(3))]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        p = write_region_history(str(tmp_path), d)
        rows = [json.loads(l) for l in open(p)]
        assert len(rows) == 2 and rows[-1]["iteration"] == 1000
        meta = json.load(open(os.path.join(str(tmp_path), "local_region_lineage.json")))
        assert meta["policy"]["split_min_cells"] == 10_000
        assert "ticks" not in meta and meta["lineage"]


# ---------------------------------------------------------------------------
# The one live-path addition Level B needed: the co-visibility snapshot
# ---------------------------------------------------------------------------
class TestCovisibilitySnapshot:
    """``LocalQualityTracker.covisibility_snapshot`` is the ONLY thing Level B
    added to the training path. It is derived purely from the frozen,
    sensor-depth cell maps -- no render, no residual, no PSNR."""

    def _tracker_with(self, maps, grid=None):
        import torch
        from eligsir.utils.local_quality import CameraCellMap, LocalQualityTracker
        t = LocalQualityTracker(cell_size=CELL)
        if grid is not None:
            t.grid = grid
        for (cam_key, h, w), slots in maps.items():
            si = torch.tensor(slots, dtype=torch.long)
            t._cell_maps[(cam_key, h, w)] = CameraCellMap(
                cam_key=cam_key, height=h, width=w, slot_index=si,
                valid_mask=si >= 0, n_valid=int((si >= 0).sum()), n_total=si.numel())
        return t

    def test_snapshot_maps_cameras_to_the_cells_they_saw(self):
        import numpy as np
        from eligsir.utils.local_quality import world_to_cell_key
        seed = self._tracker_with({})
        pts = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float64)
        slots = seed.grid.slots_for_world_points(pts)
        t = self._tracker_with({("src3", 1, 3): [int(slots[0]), -1, int(slots[1])]},
                               grid=seed.grid)
        snap = t.covisibility_snapshot()
        assert set(snap) == {"src3"}
        assert snap["src3"] == sorted(int(k) for k in world_to_cell_key(pts, CELL))

    def test_a_resolution_change_does_not_duplicate_a_camera(self):
        """Level A caches a second CameraCellMap per camera when the render
        resolution changes; counting that as a second observer would inflate
        co-visibility weight with a camera that does not exist."""
        seed = self._tracker_with({})
        seed.grid.slots_for_keys([key(0, 0, 0), key(1, 0, 0), key(2, 0, 0)])
        t = self._tracker_with({("src3", 2, 2): [0, 1, -1, -1],
                                ("src3", 4, 4): [0, 2] + [-1] * 14}, grid=seed.grid)
        snap = t.covisibility_snapshot()
        assert list(snap) == ["src3"], "one physical camera, one entry"
        assert snap["src3"] == sorted([key(0, 0, 0), key(1, 0, 0), key(2, 0, 0)])

    def test_snapshot_roundtrips_through_disk(self, tmp_path):
        from eligsir.utils.local_quality import (read_covisibility_snapshot,
                                         write_covisibility_snapshot)
        write_covisibility_snapshot(str(tmp_path), {"src0": [1, 2]}, CELL)
        cams, cs = read_covisibility_snapshot(str(tmp_path))
        assert cams == {"src0": [1, 2]} and cs == CELL
        assert read_covisibility_snapshot(str(tmp_path / "nope")) == (None, None)


    def test_snapshot_is_read_only_and_idempotent(self):
        """The training loop calls this; it must not mutate Level A state.

        (The call site itself is inside the existing
        ``--streaming_local_quality_cells`` gate AND its own try/except, and
        writes only to disk -- see the PR's grep of the added lines for the
        rest of the behaviour-neutrality argument.)"""
        seed = self._tracker_with({})
        seed.grid.slots_for_keys([key(0, 0, 0), key(1, 0, 0)])
        t = self._tracker_with({("src1", 1, 2): [0, 1]}, grid=seed.grid)
        before_grid, before_maps = len(t.grid), dict(t._cell_maps)
        first = t.covisibility_snapshot()
        second = t.covisibility_snapshot()
        assert first == second
        assert len(t.grid) == before_grid
        assert t._cell_maps == before_maps
        assert t.states == {} and t.cams_without_depth == 0


class TestRunningPeakSemantics:
    def test_growth_is_not_reported_as_forgetting(self):
        """A region measured low early and high later GREW. Comparing past
        samples against the FINAL peak (rather than the running peak) would
        report initial under-reconstruction as a multi-dB drop."""
        keys = line(3)
        recs = [tick(0, keys, psnr=18.0), tick(1000, keys, psnr=20.0),
                tick(2000, keys, psnr=22.0)]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["psnr_peak_db"] == pytest.approx(22.0, abs=1e-9)
        assert row["drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9)
        assert row["max_drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9)

    def test_a_recovered_dip_is_still_recorded(self):
        keys = line(3)
        recs = [tick(0, keys, psnr=22.0), tick(1000, keys, psnr=19.0),
                tick(2000, keys, psnr=23.0)]
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9)
        assert row["max_drop_from_peak_db"] == pytest.approx(3.0, abs=1e-9)


class TestRegionGrowthIsNotForgetting:
    """The issue's own rule, one level up from Level A's per-cell gate:
    'A newly discovered region being poor is NOT forgetting.'

    Level A gates each CELL, but a region pools cells -- so if a small
    established region at 30dB suddenly absorbs 90 fresh cells at 15dB, a naive
    pool over all members reports a ~14dB 'drop' that is pure growth. This is
    the normal case in a live streaming run (holdout cameras are admitted over
    time), and neither the fr3 replay nor the dumbbell fixture exercises it,
    because both instantiate their whole cell set up front.
    """

    def _grow(self):
        old = line(10)
        new = line(90, i0=10)
        recs = []
        for t in range(6):
            cells = {}
            for k in old:
                cells[str(k)] = cell_rec(k, 30.0, established=True)
            if t >= 3:
                for k in new:
                    cells[str(k)] = cell_rec(k, 15.0, established=False)
            recs.append({"iteration": t * 1000, "cell_size_m": CELL,
                         "n_cells_observed": len(cells),
                         "measurement_incomplete": False, "cells": cells})
        return recs

    def test_absorbing_fresh_poor_cells_is_not_reported_as_a_drop(self):
        d = derive_region_history(self._grow(),
                                  RegionPolicy(split_min_cells=10_000,
                                               split_max_bbox_diag_m=1e9))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["n_cells"] == 100, "the fresh cells did join the region"
        assert row["established"] is True
        assert row["drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9), \
            "growth must never read as forgetting"
        assert row["max_drop_from_peak_db"] == pytest.approx(0.0, abs=1e-9)
        assert row["psnr_db"] == pytest.approx(30.0, abs=1e-9)

    def test_a_real_drop_in_the_established_core_still_fires(self):
        recs = self._grow()
        for c in recs[-1]["cells"].values():
            if c["established"]:
                c["psnr_db"] = 26.0
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000,
                                                     split_max_bbox_diag_m=1e9))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["drop_from_peak_db"] == pytest.approx(4.0, abs=1e-9)

    def test_growth_confounding_is_flagged_when_the_pool_composition_moved(self):
        recs = self._grow()
        for r in recs[4:]:                      # the new cells establish late
            for c in r["cells"].values():
                c["established"] = True
        d = derive_region_history(recs, RegionPolicy(split_min_cells=10_000,
                                                     split_max_bbox_diag_m=1e9))
        row = list(d["ticks"][-1]["regions"].values())[0]
        assert row["growth_confounded"] is True
        assert row["n_cells_established_at_peak"] < row["n_cells_established"]


class TestReportScriptExitPaths:
    """The report script's two failure exits. An error path nobody has ever
    executed is exactly the shape of the swallowed-exception defect found in
    Level A review, so they are covered here rather than trusted."""

    def _main(self, argv, monkeypatch):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "local_region_report",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "local_region_report.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(sys, "argv", ["local_region_report.py"] + argv)
        return mod.main()

    @staticmethod
    def _assert_vector_svg(path):
        assert path.exists() and path.stat().st_size > 0
        content = path.read_text()
        assert "<svg" in content
        assert "<image" not in content
        assert not path.with_suffix(".png").exists()

    def test_missing_level_a_history_exits_2_with_a_not_measured_panel(
            self, tmp_path, monkeypatch):
        assert self._main([str(tmp_path)], monkeypatch) == 2
        self._assert_vector_svg(tmp_path / "region_forgetting_heatmap.svg")

    def test_a_tick_with_no_evidence_exits_3_and_says_so_on_disk(
            self, tmp_path, monkeypatch):
        (tmp_path / "local_quality_history.jsonl").write_text(json.dumps({
            "iteration": 1000, "cell_size_m": CELL, "n_cells_observed": 0,
            "measurement_incomplete": True, "cells": {}}) + "\n")
        assert self._main([str(tmp_path)], monkeypatch) == 3
        for name in ("region_forgetting_heatmap.svg", "region_health_map.svg"):
            self._assert_vector_svg(tmp_path / name)
        # The JSONL is still written (deliberately: it is the machine-readable
        # record of the failure) and must itself say the tick was not measured,
        # so a consumer reading only the file cannot mistake it for a healthy run.
        rows = [json.loads(l) for l in
                open(tmp_path / "local_region_history.jsonl") if l.strip()]
        assert rows and rows[-1]["regions_valid"] is False
        assert "NOT MEASURED" in rows[-1]["invalid_reason"]
        meta = json.load(open(tmp_path / "local_region_lineage.json"))
        assert meta["regions_valid"] is False
