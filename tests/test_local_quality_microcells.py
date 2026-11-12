"""Tests for issue #358 Level A local quality-under-construction.

Four things must hold for the metric to mean anything:

1. **Microcell determinism/immutability** -- the same world point maps to the
   same cell key across calls, and an existing cell's dense scatter slot never
   renumbers when the population changes (new cells appear).
2. **pixel -> world -> cell against known geometry** -- verified against a
   hand-constructed camera whose backprojection is analytically known.
3. **Masked PSNR reduction against a hand-computed expected value.**
4. **The establishment gate refuses to flag an unestablished cell** -- no
   ``drop_from_peak_db`` is ever emitted for a cell that never established,
   which is what stops "a newly discovered region is poor" being reported as
   forgetting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eligsir.utils.local_quality import (
    CellTickAccumulator,
    EstablishmentPolicy,
    LocalQualityTracker,
    MicrocellGrid,
    build_camera_cell_map,
    camera_identity,
    cell_center,
    cell_ijk,
    cell_key,
    key_to_ijk,
    pooled_psnr_db,
    world_to_cell_key,
)
from eligsir.utils.score_contract import declare_canonical_score_tensor


def _score_tensor(tensor):
    return declare_canonical_score_tensor(tensor, context="local_quality_test")[0]


# --------------------------------------------------------------------------
# 1. Determinism / immutability
# --------------------------------------------------------------------------
class TestCellIdentityDeterminism:
    def test_same_point_same_key_across_calls(self):
        p = np.array([1.234, -5.678, 9.012], dtype=np.float32)
        keys = {int(world_to_cell_key(p, 0.5)) for _ in range(50)}
        assert len(keys) == 1

    def test_key_is_bijective_not_a_hash(self):
        rng = np.random.default_rng(0)
        ijk = rng.integers(-5000, 5000, size=(20000, 3))
        keys = cell_key(ijk)
        # No collisions anywhere in a large random sample...
        assert len(np.unique(keys)) == len(keys)
        # ...and the packing round-trips exactly.
        np.testing.assert_array_equal(key_to_ijk(keys), ijk)

    def test_points_in_same_cell_share_a_key_neighbours_do_not(self):
        cs = 0.5
        a = world_to_cell_key(np.array([0.10, 0.10, 0.10]), cs)
        b = world_to_cell_key(np.array([0.49, 0.49, 0.49]), cs)
        c = world_to_cell_key(np.array([0.51, 0.10, 0.10]), cs)
        assert int(a) == int(b)
        assert int(a) != int(c)

    def test_negative_coordinates_floor_not_truncate(self):
        """-0.25 must land in cell -1, not cell 0 -- a truncating
        implementation silently fuses the two cells straddling each axis."""
        np.testing.assert_array_equal(
            cell_ijk(np.array([-0.25, -0.75, 0.25]), 0.5), np.array([-1, -2, 0])
        )

    def test_cell_center_round_trips_into_its_own_cell(self):
        cs = 0.7
        k = world_to_cell_key(np.array([3.3, -2.2, 0.4]), cs)
        assert int(world_to_cell_key(cell_center(k, cs), cs)) == int(k)

    def test_out_of_range_refused_not_wrapped(self):
        with pytest.raises(ValueError):
            cell_key(np.array([[2 ** 40, 0, 0]]))

    def test_slots_are_append_only_across_a_population_change(self):
        """Simulated population change: new cells appear on a later tick.
        Every slot issued earlier must be unchanged, and the durable key ->
        slot association must survive."""
        grid = MicrocellGrid(0.5)
        first = np.array([[0.1, 0.1, 0.1], [5.1, 0.1, 0.1], [-3.4, 2.2, 0.0]])
        slots1 = grid.slots_for_world_points(first)
        keys1 = world_to_cell_key(first, 0.5)
        snapshot = dict(zip([int(k) for k in keys1], [int(s) for s in slots1]))

        # Population grows by 500 previously-unseen cells.
        rng = np.random.default_rng(7)
        grown = rng.uniform(-40, 40, size=(500, 3))
        grid.slots_for_world_points(grown)

        # ...and shrinks conceptually (we simply stop observing some cells).
        slots1_again = grid.slots_for_world_points(first)
        np.testing.assert_array_equal(slots1, slots1_again)
        for k, s in snapshot.items():
            assert grid.slot_of(k) == s
            assert grid.key_of(s) == k

    def test_grid_is_independent_of_insertion_order_at_the_key_level(self):
        pts = np.array([[0.1, 0.1, 0.1], [9.9, 9.9, 9.9], [-4.0, 1.0, 1.0]])
        g1, g2 = MicrocellGrid(0.5), MicrocellGrid(0.5)
        g1.slots_for_world_points(pts)
        g2.slots_for_world_points(pts[::-1])
        # Slots differ (first-seen order), keys are identical -- which is
        # exactly why the history persists keys, not slots.
        assert set(g1.keys) == set(g2.keys)


# --------------------------------------------------------------------------
# 2. pixel -> world -> cell against known geometry
# --------------------------------------------------------------------------
class _FakeCam:
    """Minimal stand-in for scene.cameras.Camera with a known pose.

    ``world_view_transform`` follows this repo's convention: the ROW-vector
    world->camera matrix (i.e. the transpose of the usual column-vector w2c),
    so that ``world_view_transform.inverse().transpose(0,1)`` is the ordinary
    column-vector c2w -- the convention
    ``backproject_depth_pixels_torch``/``reseed_producer`` document.
    """

    def __init__(self, c2w, w, h, fx, fy, cx, cy):
        w2c_col = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32))
        self.world_view_transform = w2c_col.transpose(0, 1).contiguous()
        self.image_width, self.image_height = w, h
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.FoVx = self.FoVy = 1.0
        self.uid = 0
        self.image_name = "000000"


def _identity_cam(w=4, h=3):
    return _FakeCam(torch.eye(4), w, h, fx=2.0, fy=2.0, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0)


class TestPixelToWorldToCell:
    def test_identity_pose_principal_point_lands_at_known_world_point(self):
        """With c2w = I, fx=fy=2, principal point at pixel (cx,cy), a depth of
        3.0m at the principal pixel must backproject to world (0,0,3)."""
        cam = _identity_cam()
        depth = torch.zeros(3, 4)
        # cx=1.5, cy=1.0 -> nearest integer pixel to the principal point is
        # (x=1 or 2, y=1); pick x=2,y=1 and compute the expected point exactly.
        depth[1, 2] = 3.0
        grid = MicrocellGrid(0.5)
        cm = build_camera_cell_map(cam, grid, depth, min_depth_m=0.1, max_depth_m=10.0)
        assert cm is not None and cm.n_valid == 1
        x_exp = (2 - 1.5) / 2.0 * 3.0   # 0.75
        y_exp = (1 - 1.0) / 2.0 * 3.0   # 0.0
        expected_key = int(world_to_cell_key(np.array([x_exp, y_exp, 3.0]), 0.5))
        slot = int(cm.slot_index[1 * 4 + 2].item())
        assert grid.key_of(slot) == expected_key

    def test_translated_pose_shifts_cells_by_exactly_the_translation(self):
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([10.0, -6.0, 2.0])
        cam_t = _FakeCam(c2w, 4, 3, 2.0, 2.0, 1.5, 1.0)
        depth = torch.zeros(3, 4)
        depth[1, 2] = 3.0
        g0, g1 = MicrocellGrid(0.5), MicrocellGrid(0.5)
        cm0 = build_camera_cell_map(_identity_cam(), g0, depth)
        cm1 = build_camera_cell_map(cam_t, g1, depth)
        p0 = key_to_ijk(g0.key_of(int(cm0.slot_index[6].item())))
        p1 = key_to_ijk(g1.key_of(int(cm1.slot_index[6].item())))
        # +10,-6,+2 m at 0.5m cells == +20,-12,+4 cells.
        np.testing.assert_array_equal(p1 - p0, np.array([20, -12, 4]))

    def test_invalid_depth_pixels_are_excluded_not_backprojected(self):
        """Zero / non-finite / out-of-range depth has no world point. Putting
        it in a cell would place out-of-range returns onto real surfaces."""
        cam = _identity_cam()
        depth = torch.tensor([
            [0.0, float("nan"), 3.0, 50.0],
            [3.0, float("inf"), 0.0, 3.0],
            [0.05, 3.0, 3.0, -1.0],
        ])
        cm = build_camera_cell_map(cam, MicrocellGrid(0.5), depth,
                                   min_depth_m=0.1, max_depth_m=10.0)
        assert cm.n_valid == 5            # exactly the five 3.0m pixels
        assert cm.n_total == 12
        assert int(cm.valid_mask.sum()) == 5
        assert bool((cm.slot_index[~cm.valid_mask] == -1).all())
        assert cm.invalid_fraction == pytest.approx(7.0 / 12.0)

    def test_cell_map_never_uses_rendered_depth(self):
        """The correctness property in one assertion: the frozen map is a pure
        function of (sensor depth, pose, intrinsics, cell size). Rebuilding it
        with a completely different 'reconstruction' present changes nothing,
        because no reconstruction is an input at all."""
        cam = _identity_cam()
        depth = torch.full((3, 4), 3.0)
        a = build_camera_cell_map(cam, MicrocellGrid(0.5), depth)
        b = build_camera_cell_map(cam, MicrocellGrid(0.5), depth.clone())
        assert torch.equal(a.slot_index, b.slot_index)
        assert a.source == "sensor_depth"


# --------------------------------------------------------------------------
# 3. Masked reduction against a hand-computed value
# --------------------------------------------------------------------------
class TestMaskedReduction:
    def test_two_cell_reduction_matches_hand_computation(self):
        """4x1 image; pixels 0,1 in cell A, pixel 2 in cell B, pixel 3 invalid.
        Constant per-pixel error per channel, so the pooled MSE is exact."""
        h, w = 1, 4
        cam = _FakeCam(torch.eye(4), w, h, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        # Two pixels at the same depth+column-pair land in one 10m cell; the
        # third is pushed far away into its own cell. Build the map by hand to
        # keep this test about the REDUCTION, not the projection.
        from eligsir.utils.local_quality import CameraCellMap
        cm = CameraCellMap(
            cam_key="c", height=h, width=w,
            slot_index=torch.tensor([0, 0, 1, -1], dtype=torch.long),
            valid_mask=torch.tensor([True, True, True, False]),
            n_valid=3, n_total=4,
        )
        render = torch.zeros(3, h, w)
        gt = torch.zeros(3, h, w)
        gt[:, 0, 0] = 0.1      # err 0.1 on 3 channels
        gt[:, 0, 1] = 0.3      # err 0.3 on 3 channels
        gt[:, 0, 2] = 0.2      # err 0.2 on 3 channels
        gt[:, 0, 3] = 0.9      # INVALID pixel -- must not enter any cell

        acc = CellTickAccumulator(n_slots=2)
        acc.add_view(render, gt, cm)
        sse, n, nv = acc.finalize()

        # Cell A: 2 pixels x 3 channels = 6 samples, sse = 3*(0.01) + 3*(0.09)
        assert n[0] == 6 and n[1] == 3
        assert sse[0] == pytest.approx(3 * 0.01 + 3 * 0.09)
        assert sse[1] == pytest.approx(3 * 0.04)
        assert nv[0] == 1 and nv[1] == 1

        psnr = pooled_psnr_db(sse, n)
        mse_a = (3 * 0.01 + 3 * 0.09) / 6.0
        assert psnr[0] == pytest.approx(-10.0 * math.log10(mse_a))
        assert psnr[1] == pytest.approx(-10.0 * math.log10(0.04))

    def _whole_image_cell(self, h, w):
        from eligsir.utils.local_quality import CameraCellMap
        return CameraCellMap(
            cam_key="c", height=h, width=w,
            slot_index=torch.zeros(h * w, dtype=torch.long),
            valid_mask=torch.ones(h * w, dtype=torch.bool),
            n_valid=h * w, n_total=h * w,
        )

    def _whole_image_pooled(self, render, gt):
        acc = CellTickAccumulator(n_slots=1)
        acc.add_view(render, gt, self._whole_image_cell(*render.shape[-2:]))
        sse, n, _ = acc.finalize()
        return float(pooled_psnr_db(sse, n)[0])

    def test_pooled_psnr_equals_image_utils_when_channel_mses_are_equal(self):
        """One cell covering EVERY pixel, with the SAME error in each channel,
        reproduces utils.image_utils.psnr exactly -- i.e. the pooled formula is
        the same metric, not a different one."""
        from eligsir.utils.image_utils import psnr as ref_psnr
        torch.manual_seed(0)
        render = torch.rand(3, 8, 8) * 0.9
        gt = render + 0.07   # identical per-channel error, still canonical [0, 1]
        got = self._whole_image_pooled(render, gt)
        want = float(ref_psnr(_score_tensor(render), _score_tensor(gt)).mean())
        assert got == pytest.approx(want, abs=1e-4)

    def test_pooled_psnr_diverges_from_image_utils_across_unequal_channels(self):
        """Documented, deliberate divergence (see utils/local_quality.py's
        module docstring): utils.image_utils.psnr takes the MEAN OF PER-CHANNEL
        dB values, which is a mean-of-logs. Pooled per-cell PSNR is a
        log-of-means over all channels. They differ whenever the channels have
        unequal MSE, and Jensen puts the mean-of-logs ABOVE the pooled value.
        This is why a per-cell PSNR must never be expected to reconcile with
        mean_test_psnr -- it is not a bug in either."""
        from eligsir.utils.image_utils import psnr as ref_psnr
        render = torch.zeros(3, 8, 8)
        gt = torch.zeros(3, 8, 8)
        gt[0] = 0.01     # very different per-channel errors
        gt[1] = 0.30
        gt[2] = 0.30
        got = self._whole_image_pooled(render, gt)
        want = float(ref_psnr(_score_tensor(render), _score_tensor(gt)).mean())
        assert got != pytest.approx(want, abs=1e-3)
        assert want > got
        # And the pooled value is exactly the hand-computed log-of-means.
        mse_pooled = (0.01 ** 2 + 0.30 ** 2 + 0.30 ** 2) / 3.0
        assert got == pytest.approx(-10.0 * math.log10(mse_pooled))

    def test_distinct_view_counter_counts_views_not_pixels(self):
        from eligsir.utils.local_quality import CameraCellMap
        cm = CameraCellMap(
            cam_key="c", height=1, width=4,
            slot_index=torch.tensor([0, 0, 0, 0], dtype=torch.long),
            valid_mask=torch.ones(4, dtype=torch.bool),
            n_valid=4, n_total=4,
        )
        acc = CellTickAccumulator(n_slots=1)
        for _ in range(3):
            acc.add_view(torch.zeros(3, 1, 4), torch.zeros(3, 1, 4), cm)
        _sse, n, nv = acc.finalize()
        assert nv[0] == 3          # 3 views...
        assert n[0] == 3 * 4 * 3   # ...but 36 channel-samples

    def test_resolution_mismatch_is_refused_not_interpolated(self):
        from eligsir.utils.local_quality import CameraCellMap
        cm = CameraCellMap(
            cam_key="c", height=4, width=4,
            slot_index=torch.zeros(16, dtype=torch.long),
            valid_mask=torch.ones(16, dtype=torch.bool),
            n_valid=16, n_total=16,
        )
        with pytest.raises(ValueError, match="never be interpolated"):
            CellTickAccumulator(n_slots=1).add_view(
                torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), cm
            )


# --------------------------------------------------------------------------
# 4. Establishment gate
# --------------------------------------------------------------------------
def _tick(tracker, iteration, per_cell):
    """Drive one tick directly against the accumulators.

    ``per_cell`` maps slot -> (mse, n_pixels, n_views), letting a test state
    exactly what evidence a cell has without constructing renders.
    """
    acc = tracker.new_accumulator()
    acc._ensure(torch.device("cpu"))
    for slot, (mse, n_pix, n_views) in per_cell.items():
        n_samples = float(n_pix) * 3.0
        acc.sq_err[slot] += mse * n_samples
        acc.n_samples[slot] += n_samples
        acc.n_views[slot] += float(n_views)
    return tracker.record_tick(iteration, acc)


class TestEstablishmentGate:
    def _tracker(self, **kw):
        t = LocalQualityTracker(cell_size=0.5, policy=EstablishmentPolicy(**kw))
        # Register two real cells so slots 0/1 exist.
        t.grid.slots_for_world_points(np.array([[0.1, 0.1, 0.1], [9.1, 0.1, 0.1]]))
        return t

    def test_unestablished_cell_never_reports_a_drop(self):
        """The headline guarantee: a cell that never establishes emits NO
        drop_from_peak_db, however far its PSNR falls. 'A newly discovered
        region is poor' must never be reported as forgetting."""
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=3)
        # Single-view evidence only -> never qualifies, never establishes.
        for i, mse in enumerate([0.001, 0.01, 0.1, 0.5], start=1):
            rec = _tick(t, i * 1000, {0: (mse, 1000, 1)})
        cell = rec["cells"][str(t.grid.key_of(0))]
        assert cell["established"] is False
        assert cell["qualifying_ticks"] == 0
        assert "drop_from_peak_db" not in cell
        assert "psnr_peak_db" not in cell
        assert rec["n_cells_established"] == 0
        assert rec["worst_drop_from_peak_db"] == 0.0

    def test_too_few_pixels_blocks_establishment_even_with_many_views(self):
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=1)
        rec = _tick(t, 1000, {0: (0.001, 50, 8)})   # 150 samples < 300
        assert rec["cells"][str(t.grid.key_of(0))]["established"] is False

    def test_too_few_ticks_blocks_establishment(self):
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=3)
        r1 = _tick(t, 1000, {0: (0.001, 1000, 4)})
        r2 = _tick(t, 2000, {0: (0.001, 1000, 4)})
        assert r1["cells"][str(t.grid.key_of(0))]["established"] is False
        assert r2["cells"][str(t.grid.key_of(0))]["established"] is False
        r3 = _tick(t, 3000, {0: (0.001, 1000, 4)})
        assert r3["cells"][str(t.grid.key_of(0))]["established"] is True
        assert r3["cells"][str(t.grid.key_of(0))]["established_iter"] == 3000

    def test_established_cell_reports_peak_and_drop(self):
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=1)
        # mse 0.0001 -> 40dB, then 0.01 -> 20dB: a real 20dB drop.
        r1 = _tick(t, 1000, {0: (1e-4, 1000, 4)})
        c1 = r1["cells"][str(t.grid.key_of(0))]
        assert c1["established"] is True
        assert c1["psnr_db"] == pytest.approx(40.0)
        assert c1["psnr_peak_db"] == pytest.approx(40.0)
        assert c1["drop_from_peak_db"] == pytest.approx(0.0)

        r2 = _tick(t, 2000, {0: (1e-2, 1000, 4)})
        c2 = r2["cells"][str(t.grid.key_of(0))]
        assert c2["psnr_db"] == pytest.approx(20.0)
        assert c2["psnr_peak_db"] == pytest.approx(40.0)   # peak is sticky
        assert c2["peak_iter"] == 1000
        assert c2["drop_from_peak_db"] == pytest.approx(20.0)
        assert r2["worst_drop_from_peak_db"] == pytest.approx(20.0)
        assert r2["worst_drop_cell_key"] == t.grid.key_of(0)

        # Recovery: drop returns to 0 but max_drop_from_peak_db is retained.
        r3 = _tick(t, 3000, {0: (1e-4, 1000, 4)})
        c3 = r3["cells"][str(t.grid.key_of(0))]
        assert c3["drop_from_peak_db"] == pytest.approx(0.0)
        assert c3["max_drop_from_peak_db"] == pytest.approx(20.0)

    def test_peak_is_measured_from_establishment_not_first_sighting(self):
        """A pre-establishment measurement must not become the peak: it would
        be taken while the cell was still under-reconstructed and would make a
        later genuine drop look smaller (or manufacture a fake one)."""
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=2)
        _tick(t, 1000, {0: (1e-6, 1000, 4)})   # 60dB, but tick 1 of 2
        r2 = _tick(t, 2000, {0: (1e-3, 1000, 4)})  # 30dB, establishes here
        c2 = r2["cells"][str(t.grid.key_of(0))]
        assert c2["established"] is True
        assert c2["psnr_peak_db"] == pytest.approx(30.0)   # NOT 60
        assert c2["drop_from_peak_db"] == pytest.approx(0.0)

    def test_global_mean_can_hide_a_local_drop(self):
        """The motivating scenario, end to end: cell A loses 20dB while cell B
        gains 20dB. Any scene-wide mean is unchanged; the local signal is not."""
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=1)
        r1 = _tick(t, 1000, {0: (1e-4, 1000, 4), 1: (1e-2, 1000, 4)})   # 40 / 20
        r2 = _tick(t, 2000, {0: (1e-2, 1000, 4), 1: (1e-4, 1000, 4)})   # 20 / 40
        mean1 = np.mean([c["psnr_db"] for c in r1["cells"].values()])
        mean2 = np.mean([c["psnr_db"] for c in r2["cells"].values()])
        assert mean1 == pytest.approx(mean2)                 # global: invisible
        assert r2["worst_drop_from_peak_db"] == pytest.approx(20.0)   # local: loud

    def test_resolution_mismatch_reports_incomplete_not_clean(self):
        """The nastiest failure mode this metric can have: a frozen cell map
        built at one resolution while the render arrives at another. add_view
        refuses to interpolate and raises; compute_per_view_test_psnr_gpu's
        broad `except Exception: continue` would swallow that, and EVERY view
        dropping out yields a tick reading 0 cells / 0.00dB worst drop --
        indistinguishable from 'the map is healthy'. observe() must catch it,
        count it, and mark the record incomplete.

        This is reachable in practice, not hypothetical: _get_sensor_depth
        only nearest-resizes on its cold path, and resolution_cache.py writes
        cam._sensor_depth_cache directly, so a live multires run can produce
        exactly this mismatch.
        """
        cam = _identity_cam(8, 8)
        tracker = LocalQualityTracker(
            cell_size=0.5, policy=EstablishmentPolicy(min_eval_samples=3, min_views=1, min_ticks=1)
        )
        assert tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((8, 8), 3.0)) == 1
        acc = tracker.new_accumulator()
        # Render arrives at 16x16 -- a different resolution.
        ok = tracker.observe(cam, torch.zeros(3, 16, 16), torch.zeros(3, 16, 16), acc)
        rec = tracker.record_tick(1000, acc)
        assert ok is False
        assert rec["views_skipped_resolution_mismatch"] == 1
        assert rec["measurement_incomplete"] is True
        # The dangerous part: it DOES look clean on the headline numbers.
        assert rec["n_cells_observed"] == 0
        assert rec["worst_drop_from_peak_db"] == 0.0

    def test_healthy_tick_is_not_marked_incomplete(self):
        cam = _identity_cam(8, 8)
        tracker = LocalQualityTracker(
            cell_size=0.5, policy=EstablishmentPolicy(min_eval_samples=3, min_views=1, min_ticks=1)
        )
        tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((8, 8), 3.0))
        acc = tracker.new_accumulator()
        assert tracker.observe(cam, torch.zeros(3, 8, 8), torch.full((3, 8, 8), 0.1), acc)
        rec = tracker.record_tick(1000, acc)
        assert rec["measurement_incomplete"] is False
        assert rec["n_cells_observed"] > 0

    def test_per_tick_counters_do_not_accumulate_across_ticks(self):
        """prepare() rebuilds _prepared each tick, so the skip counters must
        describe THIS tick -- they are emitted alongside per-tick fields."""
        cam = _identity_cam(8, 8)
        tracker = LocalQualityTracker(cell_size=0.5)
        for _ in range(3):
            tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: None)
            rec = tracker.record_tick(1000, tracker.new_accumulator())
        assert rec["cams_without_sensor_depth"] == 1     # not 3

    def test_record_is_json_serializable(self):
        import json
        t = self._tracker(min_eval_samples=300, min_views=2, min_ticks=1)
        rec = _tick(t, 1000, {0: (1e-3, 1000, 4)})
        assert json.loads(json.dumps(rec))["iteration"] == 1000
        assert rec["establishment_policy"]["optimization_exposure"] == "proxy:qualifying_ticks"


class TestCanonicalEvaluationTickBinding:
    """#358 rows must be cryptographically joinable to #698's one tick."""

    def _measured_tick(self):
        cam = _identity_cam(8, 8)
        tracker = LocalQualityTracker(
            cell_size=0.5,
            policy=EstablishmentPolicy(min_eval_samples=3, min_views=1, min_ticks=1),
        )
        tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((8, 8), 3.0))
        acc = tracker.new_accumulator()
        assert tracker.observe(cam, torch.zeros(3, 8, 8), torch.full((3, 8, 8), 0.1), acc)
        return tracker, acc

    def test_measured_row_binds_to_exact_canonical_evaluation_tick(self):
        import hashlib
        import json

        tracker, acc = self._measured_tick()
        canonical_tick = {
            "schema_version": "1.0.0",
            "iteration": 1000,
            "optimizer_step": 1000,
            "source_progress": 0.25,
            "supervision_views": 37,
        }
        rec = tracker.record_tick(
            1000, acc, canonical_evaluation_tick=canonical_tick,
        )
        expected_hash = hashlib.sha256(
            json.dumps(canonical_tick, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert rec["canonical_evaluation_tick"] == {
            "status": "bound",
            "schema_version": "1.0.0",
            "iteration": 1000,
            "sha256": expected_hash,
        }
        assert rec["measurement_incomplete"] is False

    def test_mismatched_canonical_tick_is_not_a_valid_local_measurement(self):
        tracker, acc = self._measured_tick()
        rec = tracker.record_tick(
            1000,
            acc,
            canonical_evaluation_tick={"schema_version": "1.0.0", "iteration": 2000},
        )
        assert rec["canonical_evaluation_tick"]["status"] == "iteration_mismatch"
        assert rec["measurement_incomplete"] is True

    def test_explicitly_missing_canonical_tick_is_not_a_valid_local_measurement(self):
        tracker, acc = self._measured_tick()
        rec = tracker.record_tick(1000, acc, canonical_evaluation_tick=None)
        assert rec["canonical_evaluation_tick"]["status"] == "unavailable"
        assert rec["measurement_incomplete"] is True


# --------------------------------------------------------------------------
# Behavior-neutrality of the psnr_history hook
# --------------------------------------------------------------------------
class TestPsnrHistoryHookNeutrality:
    def test_hook_default_none_leaves_psnr_identical(self, monkeypatch):
        import eligsir.utils.psnr_history as ph
        cams = [object(), object()]
        imgs = {id(cams[0]): (torch.zeros(3, 2, 2), torch.full((3, 2, 2), 0.1)),
                id(cams[1]): (torch.zeros(3, 2, 2), torch.full((3, 2, 2), 0.2))}
        import eligsir.utils.comparison_report as cr
        monkeypatch.setattr(cr, "_render_image", lambda c, *a, **k: _score_tensor(imgs[id(c)][0]))
        monkeypatch.setattr(cr, "_gt_image", lambda c: _score_tensor(imgs[id(c)][1]))
        _c0, t0 = ph.compute_per_view_test_psnr_gpu(None, cams, None, None, None)
        seen = []
        _c1, t1 = ph.compute_per_view_test_psnr_gpu(
            None, cams, None, None, None, per_view_hook=lambda c, r, g: seen.append(c),
        )
        assert torch.equal(t0, t1)
        assert len(seen) == 2

    def test_throwing_hook_cannot_change_psnr(self, monkeypatch):
        import eligsir.utils.psnr_history as ph
        import eligsir.utils.comparison_report as cr
        cams = [object(), object()]
        monkeypatch.setattr(cr, "_render_image", lambda c, *a, **k: _score_tensor(torch.zeros(3, 2, 2)))
        monkeypatch.setattr(cr, "_gt_image", lambda c: _score_tensor(torch.full((3, 2, 2), 0.1)))
        _c0, t0 = ph.compute_per_view_test_psnr_gpu(None, cams, None, None, None)

        def _boom(*a, **k):
            raise RuntimeError("telemetry exploded")

        _c1, t1 = ph.compute_per_view_test_psnr_gpu(
            None, cams, None, None, None, per_view_hook=_boom,
        )
        assert torch.equal(t0, t1)

    def test_full_prepare_hook_record_composition(self, monkeypatch, tmp_path):
        """End-to-end shape of the train_streaming.py wiring, with fakes: the
        tracker's cell maps are frozen from sensor depth BEFORE the render
        loop, the hook accumulates during it, and one record lands on disk."""
        import eligsir.utils.psnr_history as ph
        import eligsir.utils.comparison_report as cr
        from eligsir.utils.local_quality import append_local_quality_history, read_local_quality_history

        cams = [_identity_cam(8, 8), _identity_cam(8, 8)]
        cams[1].uid = 1
        cams[1].image_name = "000001"
        depth = torch.full((8, 8), 3.0)
        renders = {0: torch.zeros(3, 8, 8), 1: torch.zeros(3, 8, 8)}
        monkeypatch.setattr(cr, "_render_image", lambda c, *a, **k: _score_tensor(renders[c.uid]))
        monkeypatch.setattr(cr, "_gt_image", lambda c: _score_tensor(torch.full((3, 8, 8), 0.1)))

        tracker = LocalQualityTracker(
            cell_size=0.5, policy=EstablishmentPolicy(min_eval_samples=3, min_views=2, min_ticks=1)
        )
        n_prepared = tracker.prepare(cams, sensor_depth_fn=lambda c, h, w: depth)
        assert n_prepared == 2
        acc = tracker.new_accumulator()
        ph.compute_per_view_test_psnr_gpu(
            None, cams, None, None, None,
            per_view_hook=lambda c, r, g: tracker.observe(c, r, g, acc),
        )
        rec = tracker.record_tick(1000, acc)
        # Both cameras are at the identical pose, so every cell sees 2 views.
        assert rec["n_cells_observed"] > 0
        assert rec["n_cells_established"] == rec["n_cells_observed"]
        assert all(c["n_views"] == 2 for c in rec["cells"].values())
        assert rec["views_skipped_unprepared"] == 0
        append_local_quality_history(str(tmp_path), rec)
        assert read_local_quality_history(str(tmp_path))[0]["iteration"] == 1000


class TestCrossDeviceAndSilentDrop:
    """Issue #358: the live GPU path measured nothing, and said so silently.

    A holdout camera's sensor depth comes from
    ``utils.comparison_report._get_sensor_depth``, which returns a CPU tensor
    (holdout cams never pass through ``train_streaming.bind_resolution_tier``,
    the only writer of a CUDA ``cam._sensor_depth_cache``), so the frozen cell
    map was CPU-resident while the render was CUDA. Every ``scatter_add_``
    raised ``RuntimeError`` -- not the ``ValueError`` ``observe()`` guarded --
    which escaped into ``compute_per_view_test_psnr_gpu``'s bare
    ``except Exception: continue`` and vanished. 53/53 ticks of a real corridor
    run read ``n_cells_observed=0`` with ``measurement_incomplete=False``.

    Every pre-existing test in this file runs entirely on CPU, where both
    tensors trivially share a device -- which is structurally why 30 passing
    tests could not catch it.
    """

    def test_non_valueerror_is_counted_not_silent(self):
        """The class of bug, independent of devices: anything other than the
        anticipated ValueError must still mark the tick incomplete."""
        cam = _identity_cam(4, 3)
        tracker = LocalQualityTracker(cell_size=0.5)
        assert tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((3, 4), 2.0)) == 1
        acc = tracker.new_accumulator()

        class _Boom:
            def add_view(self, *a, **k):
                raise RuntimeError("simulated non-ValueError failure "
                                    "(e.g. a device mismatch)")

        assert tracker.observe(cam, torch.zeros(3, 3, 4), torch.zeros(3, 3, 4), _Boom()) is False
        rec = tracker.record_tick(1000, acc)
        assert rec["n_cells_observed"] == 0
        assert rec["views_skipped_error"] == 1
        # The whole point: a zero-cell tick must never look healthy.
        assert rec["measurement_incomplete"] is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
    def test_cpu_cell_map_against_cuda_render_measures_cells(self):
        """The concrete instance: CPU cell map + CUDA render must now measure."""
        cam = _identity_cam(4, 3)
        tracker = LocalQualityTracker(cell_size=0.5)
        depth_cpu = torch.full((3, 4), 2.0)          # CPU, as _get_sensor_depth returns
        assert tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: depth_cpu) == 1
        acc = tracker.new_accumulator()
        render = torch.rand(3, 3, 4, device="cuda")  # CUDA, as the renderer produces
        gt = torch.rand(3, 3, 4, device="cuda")
        assert tracker.observe(cam, render, gt, acc) is True
        rec = tracker.record_tick(1000, acc)
        assert rec["n_cells_observed"] > 0
        assert rec["views_skipped_error"] == 0
        assert rec["measurement_incomplete"] is False
        # The Level B co-visibility artifact must survive the CUDA path too:
        # it is the input to the co-visibility ablation, and its only call site
        # (train_streaming.py) swallows failures with a print.
        snap = tracker.covisibility_snapshot()
        assert snap and all(len(v) > 0 for v in snap.values())
        # The frozen map itself stays CPU-resident: the device copy is made per
        # view and released, so device memory is O(1), not O(n_cameras).
        assert tracker._prepared[camera_identity(cam)].slot_index.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
    def test_cuda_depth_never_makes_the_persistent_cell_map_cache_cuda_resident(self):
        """A training camera may already be fully resident when control code
        asks for incidence. Its long-lived map must still remain CPU-owned."""
        cam = _identity_cam(4, 3)
        cam.world_view_transform = cam.world_view_transform.cuda()
        tracker = LocalQualityTracker(cell_size=0.5)

        cell_map = tracker.cell_map_for(
            cam, torch.full((3, 4), 2.0, device="cuda")
        )

        assert cell_map is not None
        cached_tensors = (
            cell_map.slot_index,
            cell_map.valid_mask,
            cell_map.valid_px,
            cell_map.valid_slots,
            cell_map.uniq_slots,
            cell_map.uniq_counts,
            cell_map.uniq_ones,
        )
        assert all(tensor.device.type == "cpu" for tensor in cached_tensors)


# --------------------------------------------------------------------------
# Issue #358 follow-up: the silent all-zero-looks-healthy failure class must
# not be able to recur, regardless of which exception causes it.
#
# PR #483 root-caused and fixed the SPECIFIC instance (a CPU-resident cell
# map, built from `_get_sensor_depth`'s CPU tensor, scattered against a CUDA
# render) via `CameraCellMap.index_on(device)` -- see
# TestCrossDeviceAndSilentDrop above, which already covers that concrete
# instance (including the CUDA-gated regression test). This class instead
# guards the GENERAL failure class two ways that hold with or without #483
# merged:
#   1. observe() must never let ANY exception other than the anticipated
#      resolution ValueError escape uncounted (already covered by
#      test_non_valueerror_is_counted_not_silent above; the test below
#      additionally reproduces it through the REAL call chain).
#   2. record_tick() must never be able to emit n_cells_observed=0 alongside
#      measurement_incomplete=False, no matter which (if any) named skip
#      counter explains the zero -- because the real bug's exception did not
#      match ANY skip counter that existed at the time.
# --------------------------------------------------------------------------
class TestSilentMeasurementFailureCannotRecur:
    def test_reproduces_original_silent_lie_through_the_real_call_chain(self, monkeypatch):
        """End-to-end reproduction of the EXACT failure chain issue #358
        found on a real run, using the real caller
        (compute_per_view_test_psnr_gpu) rather than calling observe()
        directly -- this is what makes it a reproduction and not just a
        unit test of observe() in isolation. Only the deepest layer
        (the accumulator's add_view) is faked, standing in for what a real
        device-mismatch scatter_add_ raises; everything above it (observe(),
        the per_view_hook plumbing, psnr_history's render loop) is the real
        production code path.
        """
        import eligsir.utils.psnr_history as ph
        import eligsir.utils.comparison_report as cr

        cam = _identity_cam(4, 3)
        tracker = LocalQualityTracker(cell_size=0.5)
        assert tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((3, 4), 2.0)) == 1
        acc = tracker.new_accumulator()

        def _boom_like_a_device_mismatch(*a, **k):
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got "
                "index is on cpu, different from other tensors on cuda:0"
            )

        monkeypatch.setattr(acc, "add_view", _boom_like_a_device_mismatch)
        monkeypatch.setattr(cr, "_render_image", lambda c, *a, **k: _score_tensor(torch.zeros(3, 3, 4)))
        monkeypatch.setattr(cr, "_gt_image", lambda c: _score_tensor(torch.zeros(3, 3, 4)))

        cams, psnr_tensor = ph.compute_per_view_test_psnr_gpu(
            None, [cam], None, None, None,
            per_view_hook=lambda c, r, g: tracker.observe(c, r, g, acc),
        )
        # The other half of the original bug, documented on purpose (this is
        # NOT what this fix changes, and must stay true): a raising hook
        # never touches mean_test_psnr -- this camera's PSNR is still
        # computed and returned exactly as if local-quality tracking were
        # off. psnr_history.py's own docstring guarantees this deliberately.
        assert psnr_tensor is not None and len(cams) == 1

        rec = tracker.record_tick(1000, acc)
        assert rec["n_cells_observed"] == 0
        assert rec["views_skipped_error"] == 1
        # Before this fix: measurement_incomplete was False here, because
        # the RuntimeError from a device mismatch matched none of the
        # ValueError-only skip path's counters. That is the exact
        # self-contradiction issue #358 found in a real report.json (53/53
        # ticks, all reading n_cells_observed=0, measurement_incomplete=False).
        assert rec["measurement_incomplete"] is True

    def test_invariant_holds_even_with_no_named_skip_reason(self):
        """The hardening this task adds beyond observe()'s catch-all: even a
        record_tick with EVERY skip counter at zero (no unprepared camera, no
        missing depth, no resolution mismatch, no caught error -- e.g. a tick
        where the holdout camera list was simply empty, or prepare() was
        never called before record_tick) must still report incomplete if it
        observed zero cells. This is the literal pairing issue #358's own
        artifact exhibited: all skip counters zero, n_cells_observed zero,
        measurement_incomplete False."""
        tracker = LocalQualityTracker(cell_size=0.5)
        acc = tracker.new_accumulator()  # never prepared, never observed
        rec = tracker.record_tick(1000, acc)
        assert rec["cams_without_sensor_depth"] == 0
        assert rec["views_skipped_unprepared"] == 0
        assert rec["views_skipped_resolution_mismatch"] == 0
        assert rec["views_skipped_error"] == 0
        assert rec["n_cells_observed"] == 0
        assert rec["measurement_incomplete"] is True

    def test_healthy_tick_is_still_not_marked_incomplete(self):
        """Guard against over-correcting: a genuinely healthy tick (cells
        really were observed) must still read incomplete=False. Duplicates
        part of test_healthy_tick_is_not_marked_incomplete above
        deliberately, as a belt-and-suspenders check right next to the new
        invariant so a future edit to the invariant clause is forced to keep
        both true at once."""
        cam = _identity_cam(8, 8)
        tracker = LocalQualityTracker(
            cell_size=0.5, policy=EstablishmentPolicy(min_eval_samples=3, min_views=1, min_ticks=1)
        )
        tracker.prepare([cam], sensor_depth_fn=lambda c, h, w: torch.full((8, 8), 3.0))
        acc = tracker.new_accumulator()
        assert tracker.observe(cam, torch.zeros(3, 8, 8), torch.full((3, 8, 8), 0.1), acc)
        rec = tracker.record_tick(1000, acc)
        assert rec["n_cells_observed"] > 0
        assert rec["measurement_incomplete"] is False


class TestMeasurementContradictionAudit:
    """utils.local_quality.find_measurement_contradictions /
    warn_measurement_contradictions -- the report-time (read-path) half of
    the guard, for auditing history files that predate the record_tick fix
    (or were written by some future code path nobody thought to name a skip
    counter for)."""

    def test_finds_the_exact_contradiction_pattern(self):
        from eligsir.utils.local_quality import find_measurement_contradictions

        records = [
            {"iteration": 1000, "n_cells_observed": 12, "measurement_incomplete": False},
            {"iteration": 2000, "n_cells_observed": 0, "measurement_incomplete": False},  # BAD
            {"iteration": 3000, "n_cells_observed": 0, "measurement_incomplete": True},   # OK: honest
            {"iteration": 4000, "n_cells_observed": 0, "measurement_incomplete": False},  # BAD
        ]
        assert find_measurement_contradictions(records) == [2000, 4000]

    def test_clean_history_has_no_contradictions(self):
        from eligsir.utils.local_quality import find_measurement_contradictions

        records = [
            {"iteration": i, "n_cells_observed": 5, "measurement_incomplete": False}
            for i in range(0, 5000, 1000)
        ]
        assert find_measurement_contradictions(records) == []

    def test_warn_prints_once_and_returns_the_bad_iterations(self, capsys):
        from eligsir.utils.local_quality import warn_measurement_contradictions

        records = [{"iteration": 53000, "n_cells_observed": 0, "measurement_incomplete": False}]
        bad = warn_measurement_contradictions(records, source="fake_run/local_quality_history.jsonl")
        assert bad == [53000]
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "53000" in out
        assert "issue358_floor3_local_quality_cells_cells_on" not in out  # generic, not one run's name

    def test_warn_is_silent_on_a_clean_history(self, capsys):
        from eligsir.utils.local_quality import warn_measurement_contradictions

        records = [{"iteration": 1000, "n_cells_observed": 3, "measurement_incomplete": False}]
        assert warn_measurement_contradictions(records) == []
        assert capsys.readouterr().out == ""

    def test_read_local_quality_history_surfaces_the_warning(self, tmp_path, capsys):
        """End-to-end through the real report-time reader: a jsonl file
        written by an unfixed build (or corrupted by hand) must be flagged
        the moment a report script loads it, not only when someone thinks to
        run the audit function directly."""
        import json

        from eligsir.utils.local_quality import read_local_quality_history

        path = tmp_path / "local_quality_history.jsonl"
        path.write_text(json.dumps(
            {"iteration": 7000, "n_cells_observed": 0, "measurement_incomplete": False}
        ) + "\n")
        records = read_local_quality_history(str(tmp_path))
        assert len(records) == 1
        assert "WARNING" in capsys.readouterr().out


class TestHoistedPerCameraConstants:
    """Issue #358 GPU-cost work: ``n_samples``/``n_views`` are pure functions
    of the FROZEN cell map, so they are precomputed once per camera in
    ``build_camera_cell_map`` and applied with ``index_add_`` over the
    camera's distinct slots instead of one fp64 ``scatter_add_`` element per
    pixel per view. That is a pure speed change, so the only thing worth
    testing is that it is EXACTLY equivalent to the old per-pixel form -- on
    the accumulators AND on the resulting per-cell PSNR.
    """

    def _built_map(self, cell_size=0.5):
        from eligsir.utils.local_quality import MicrocellGrid, build_camera_cell_map
        h, w = 6, 8
        cam = _FakeCam(torch.eye(4), w, h, fx=4.0, fy=4.0,
                       cx=(w - 1) / 2.0, cy=(h - 1) / 2.0)
        torch.manual_seed(7)
        depth = torch.rand(h, w) * 2.0 + 0.5
        depth[0, 0] = 0.0        # an invalid pixel must stay excluded
        depth[1, 3] = float("nan")
        grid = MicrocellGrid(cell_size)
        cm = build_camera_cell_map(cam, grid, depth)
        return cm, h, w

    def test_build_populates_the_precomputed_constants_consistently(self):
        cm, h, w = self._built_map()
        assert cm.valid_slots is not None and cm.uniq_slots is not None
        # The hoisted per-pixel view agrees with the legacy boolean-mask form.
        assert torch.equal(cm.valid_slots, cm.slot_index[cm.valid_mask])
        assert int(cm.valid_px.numel()) == cm.n_valid
        # And the per-cell histogram is exactly 3 channel-samples per pixel.
        assert float(cm.uniq_counts.sum()) == pytest.approx(3.0 * cm.n_valid)
        assert torch.equal(cm.uniq_ones, torch.ones_like(cm.uniq_ones))

    def test_hoisted_and_legacy_paths_are_bit_identical(self):
        """A cell map WITHOUT the precomputed fields (as any map pickled or
        constructed before this change would be) takes add_view's fallback
        branch. Both branches must produce identical accumulators."""
        from eligsir.utils.local_quality import CameraCellMap, pooled_psnr_db
        cm, h, w = self._built_map()
        legacy = CameraCellMap(
            cam_key=cm.cam_key, height=h, width=w,
            slot_index=cm.slot_index, valid_mask=cm.valid_mask,
            n_valid=cm.n_valid, n_total=cm.n_total,
        )
        assert legacy.valid_slots is None      # really the fallback branch
        n_slots = int(cm.slot_index.max()) + 1
        torch.manual_seed(11)
        views = [(torch.rand(3, h, w), torch.rand(3, h, w)) for _ in range(3)]

        outs = []
        for m in (cm, legacy):
            acc = CellTickAccumulator(n_slots=n_slots)
            for r, g in views:
                acc.add_view(r, g, m)
            outs.append(acc.finalize())
        (sse_a, n_a, v_a), (sse_b, n_b, v_b) = outs
        np.testing.assert_array_equal(n_a, n_b)
        np.testing.assert_array_equal(v_a, v_b)
        np.testing.assert_allclose(sse_a, sse_b, rtol=0, atol=0)
        np.testing.assert_allclose(pooled_psnr_db(sse_a, n_a),
                                   pooled_psnr_db(sse_b, n_b), rtol=0, atol=0)

    def test_counts_still_count_channel_samples_and_views_count_views(self):
        """Guards the meaning of the two hoisted accumulators: n_samples is
        3 per contributing PIXEL summed over views, n_views is 1 per view
        that touches the cell (not per pixel)."""
        cm, h, w = self._built_map()
        n_slots = int(cm.slot_index.max()) + 1
        acc = CellTickAccumulator(n_slots=n_slots)
        for _ in range(4):
            acc.add_view(torch.zeros(3, h, w), torch.zeros(3, h, w), cm)
        _sse, n, v = acc.finalize()
        assert n.sum() == pytest.approx(4 * 3 * cm.n_valid)
        touched = int(cm.uniq_slots.numel())
        assert int((v > 0).sum()) == touched
        assert v[v > 0].max() == pytest.approx(4.0)
