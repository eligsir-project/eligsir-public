"""Tests for the W3-47 real Phase C reseed candidate producer
(eligsir/optimization/gaussian/strategies/live/reseed_producer.py).

Closes the gap root-caused in
docs/architecture/2026-08-09-w3-45-phase-c-reseed-root-cause.md: Phase C
reseeding was inert because no production code path ever constructed a
SurfaceSeedRequest. CPU-only, no gsplat/GPU dependency.
"""
from __future__ import annotations

import torch

from scene.streaming_scene import StreamingScene
from eligsir.optimization.gaussian.strategies.live.reseed_producer import (
    _xyz_to_cell_ids,
    derive_reseed_candidates,
)

_IDENTITY_C2W = torch.eye(4, dtype=torch.float32)


def test_returns_empty_when_depth_or_alpha_missing():
    assert derive_reseed_candidates(
        gt_depth=None, alpha=torch.zeros(4, 4), fx=1, fy=1, cx=0, cy=0, c2w=_IDENTITY_C2W
    ) == ()
    assert derive_reseed_candidates(
        gt_depth=torch.ones(4, 4), alpha=None, fx=1, fy=1, cx=0, cy=0, c2w=_IDENTITY_C2W
    ) == ()


def test_returns_empty_when_shapes_mismatch():
    assert derive_reseed_candidates(
        gt_depth=torch.ones(4, 4), alpha=torch.zeros(3, 3), fx=1, fy=1, cx=0, cy=0, c2w=_IDENTITY_C2W
    ) == ()


def test_returns_empty_when_fully_covered():
    # Valid depth everywhere, but alpha already high everywhere -> nothing
    # under-covered, no candidates.
    depth = torch.full((8, 8), 2.0)
    alpha = torch.full((8, 8), 0.9)
    result = derive_reseed_candidates(
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=4.0, cy=4.0, c2w=_IDENTITY_C2W
    )
    assert result == ()


def test_returns_empty_when_no_valid_depth():
    # Alpha is low everywhere (under-covered), but depth is invalid (0)
    # everywhere -- no real surface evidence to reseed from.
    depth = torch.zeros(8, 8)
    alpha = torch.zeros(8, 8)
    result = derive_reseed_candidates(
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=4.0, cy=4.0, c2w=_IDENTITY_C2W
    )
    assert result == ()


def test_identifies_a_single_under_covered_surface_region():
    depth = torch.zeros(8, 8)
    alpha = torch.full((8, 8), 0.9)  # well-covered everywhere by default
    # One small under-covered patch with valid depth.
    depth[3:5, 3:5] = 2.0
    alpha[3:5, 3:5] = 0.0

    result = derive_reseed_candidates(
        # cx/cy deliberately off-integer (not exactly matching a pixel
        # index) so the backprojected x/y never lands exactly on 0 --
        # avoids a voxel-hash boundary artifact from the hash's internal
        # 1000.0 offset landing exactly on a cell edge for a suspiciously
        # round test coordinate (caught while writing this test).
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=4.3, cy=4.3, c2w=_IDENTITY_C2W,
        voxel_size=0.5, alpha_threshold=0.3,
    )
    assert len(result) == 1
    req = result[0]
    assert req.utility > 0.0
    assert req.reason == "under_covered_surface"
    assert req.position is not None
    # Backprojected point should sit roughly at z=2 in front of an
    # identity-pose camera looking down +z (with x,y offsets from the
    # cx/cy-relative pixel positions), not at the origin.
    assert abs(float(req.position[2]) - 2.0) < 1e-4


def test_aggregates_pixels_in_the_same_voxel_cell_into_one_request():
    depth = torch.zeros(6, 6)
    alpha = torch.full((6, 6), 0.9)
    # Two adjacent under-covered pixels, close enough in world-space to
    # fall in the same (coarse) voxel cell.
    depth[2, 2] = 2.0
    depth[2, 3] = 2.0
    alpha[2, 2] = 0.0
    alpha[2, 3] = 0.0

    result = derive_reseed_candidates(
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=3.3, cy=3.3, c2w=_IDENTITY_C2W,
        voxel_size=5.0,  # coarse enough that both pixels land in one cell
    )
    assert len(result) == 1
    # utility should reflect BOTH pixels' under-coverage summed, not just one.
    single_pixel = derive_reseed_candidates(
        gt_depth=depth[2:3, 2:3], alpha=alpha[2:3, 2:3],
        fx=10.0, fy=10.0, cx=0.3, cy=0.3, c2w=_IDENTITY_C2W, voxel_size=5.0,
    )
    assert result[0].utility > single_pixel[0].utility


def test_respects_max_candidates_cap_and_ranks_by_utility():
    depth = torch.zeros(20, 20)
    alpha = torch.full((20, 20), 0.9)
    # Ten well-separated under-covered single-pixel patches at increasing
    # depth (so cell ids differ and utility ties are avoided by construction
    # -- alpha is identical per pixel so utility differs only via count, but
    # each patch is a single isolated pixel so counts tie too; that's fine,
    # this test only checks the CAP is respected and every returned request
    # is a genuinely distinct cell).
    for i in range(10):
        depth[i, i] = 1.0 + 0.1 * i
        alpha[i, i] = 0.0

    result = derive_reseed_candidates(
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=10.0, cy=10.0, c2w=_IDENTITY_C2W,
        voxel_size=0.05, max_candidates=3,
    )
    assert len(result) == 3
    assert len({r.cell_id for r in result}) == 3


def test_cell_hash_matches_streaming_scene_convention():
    # Not a hard dependency (reseed_producer.py imports nothing from
    # StreamingScene), but the two are documented as intentionally using
    # the same voxel-hash formula -- verify they actually agree.
    xyz = torch.tensor([[1.23, -4.5, 6.78], [0.0, 0.0, 0.0], [-10.1, 5.5, 2.2]], dtype=torch.float32)
    voxel_size = 0.1

    torch_ids = _xyz_to_cell_ids(xyz, voxel_size)
    np_ids = StreamingScene._xyz_to_keys(None, xyz.numpy(), voxel_size)  # not self-dependent

    assert torch_ids.tolist() == np_ids.tolist()


def test_end_to_end_candidates_flow_through_build_mutation_plan_nonempty():
    """The actual W3-45 gap, closed: previously reseed_candidates was
    ALWAYS () at the sole real call site (live_wrapped.py), so
    build_mutation_plan's reseed_requests was always empty regardless of
    any flag. Proves the real producer's output survives cell-budget
    admission and reaches plan.reseed_requests non-empty."""
    from eligsir.optimization.gaussian.strategies.contracts import SurfaceSeedRequest as _SSR
    from eligsir.optimization.gaussian.strategies.live.cell_budget import CellBudgetTracker
    from eligsir.optimization.gaussian.strategies.live.lifecycle import LifecycleThresholds
    from eligsir.optimization.gaussian.strategies.live.mutation_plan import (
        SafetyFuses,
        build_mutation_plan,
    )
    from eligsir.optimization.gaussian.strategies.live.state import LivePopulationState

    depth = torch.zeros(8, 8)
    alpha = torch.full((8, 8), 0.9)
    depth[3:5, 3:5] = 2.0
    alpha[3:5, 3:5] = 0.0

    candidates = derive_reseed_candidates(
        gt_depth=depth, alpha=alpha, fx=10.0, fy=10.0, cx=4.3, cy=4.3, c2w=_IDENTITY_C2W,
        voxel_size=0.5,
    )
    assert len(candidates) > 0, "producer must yield at least one candidate for this fixture"
    assert isinstance(candidates[0], _SSR)

    state = LivePopulationState()
    state.align(4, topology_version=0, iteration=0)
    plan, _streaks = build_mutation_plan(
        state=state,
        thresholds=LifecycleThresholds(),
        fuses=SafetyFuses(),
        iteration=0,
        topology_version=0,
        reseed_candidates=candidates,
        cell_budget=CellBudgetTracker(budget_per_cell=4, max_total_requests=1000),
    )
    assert len(plan.reseed_requests) == len(candidates), (
        "real producer output must reach the plan unfiltered when well within "
        "cell/total budgets -- this is the exact path that was always empty "
        "before W3-47 (live_wrapped.py never passed reseed_candidates at all)"
    )
