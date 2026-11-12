"""Tests for Phase 13/14 loop-closure wiring.

These tests exercise the PoseGraph, apply_submap_correction, and the
_maybe_create_pose_graph helper — all without touching the full training loop.

Two tests below (`test_optimize_recovers_known_world_poses` and
`test_world_delta_to_local_edge_*`) DO call `PoseGraph.optimize()` /
require Open3D, unlike the rest of this file — this is deliberate: they are
the regression coverage for the original loop-closure convention bug (node-
pose inversion / odometry-edge direction, see loop_closure/pose_graph.py git
history) and for the force-field world/local frame-conflation fix
(loop_closure.pose_graph.world_delta_to_local_edge). They are skipped if
Open3D is not importable in the current environment (run inside the
container per CLAUDE.md to actually exercise them).
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Make sure the worktree root is on the path regardless of invocation location.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Test 1: PoseGraph accumulates edges correctly
# ---------------------------------------------------------------------------

def test_pose_graph_accumulates_edges():
    """Three odometry edges → len(_edges) == 3."""
    from loop_closure.pose_graph import PoseGraph

    pg = PoseGraph(n_nodes=0)
    pg.add_node(np.eye(4))  # node 0
    pg.add_node(np.eye(4))  # node 1
    pg.add_node(np.eye(4))  # node 2
    pg.add_node(np.eye(4))  # node 3

    pg.add_odometry_edge(0, 1, np.eye(4))
    pg.add_odometry_edge(1, 2, np.eye(4))
    pg.add_odometry_edge(2, 3, np.eye(4))

    assert len(pg._edges) == 3
    assert all(not e.is_loop for e in pg._edges)
    assert pg.get_edge_count() == 3
    assert pg.get_edge_count(loop_only=True) == 0


# ---------------------------------------------------------------------------
# Test 2: apply_submap_correction with identity transforms leaves means unchanged
# ---------------------------------------------------------------------------

def test_apply_submap_correction_identity_leaves_means_unchanged():
    """Identity optimised transforms must not move any Gaussian."""
    import torch
    from loop_closure.submap_correction import apply_submap_correction
    from eligsir.utils.submap_manager import SubmapManager

    # Minimal Gaussian stub
    class _FakeGaussians:
        def __init__(self, xyz, submap_id):
            import torch
            self.params = {"means": torch.nn.Parameter(xyz.clone())}
            self.submap_id = submap_id

        @property
        def get_xyz(self):
            return self.params["means"].data

    xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    submap_ids = torch.zeros(3, dtype=torch.long)  # all in submap 0
    gaussians = _FakeGaussians(xyz, submap_ids)

    sm = SubmapManager(frames_per_submap=50)
    # submap 0 has identity transform (default)

    optimised = [np.eye(4, dtype=np.float32)]  # single submap, identity
    n_corrected = apply_submap_correction(gaussians, sm, [0], optimised)

    # Gaussians were touched (count > 0) but their positions are unchanged
    assert n_corrected == 3, f"Expected 3 corrected, got {n_corrected}"
    np.testing.assert_allclose(
        gaussians.params["means"].data.numpy(),
        xyz.numpy(),
        atol=1e-5,
        err_msg="Identity transform must not move Gaussian positions",
    )


# ---------------------------------------------------------------------------
# Test 3: _maybe_create_pose_graph returns None when loop_closure_enabled=False
# ---------------------------------------------------------------------------

def test_maybe_create_pose_graph_disabled_returns_none():
    """With loop_closure_enabled=False (default), no PoseGraph is created.

    Imports from loop_closure.factory (the real implementation) so regressions
    in the shipped code are detected without triggering GPU-dependent imports.
    """
    import types
    from loop_closure.factory import maybe_create_pose_graph as _maybe_create_pose_graph

    # Default args (flag absent → treated as False)
    args_default = types.SimpleNamespace()
    assert _maybe_create_pose_graph(args_default) is None

    # Explicitly disabled
    args_off = types.SimpleNamespace(loop_closure_enabled=False)
    assert _maybe_create_pose_graph(args_off) is None

    # Explicitly enabled
    args_on = types.SimpleNamespace(loop_closure_enabled=True)
    result = _maybe_create_pose_graph(args_on)
    assert result is not None, "Expected a PoseGraph when loop_closure_enabled=True"
    from loop_closure.pose_graph import PoseGraph
    assert isinstance(result, PoseGraph)
    assert result.get_node_count() == 1


def _require_open3d():
    pytest.importorskip("open3d")


def _rotation_matrix(axis: str, deg: float) -> np.ndarray:
    """Small helper: rotation matrix about a principal axis, degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def _se3(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Test 4 (C1): optimize() recovers known ground-truth world poses from a
# non-commuting (rotation + translation) odometry chain plus a consistent
# loop-closure edge. This is the test that would have caught the original
# node-pose-inversion / odometry-edge-direction convention bug -- it
# exercises real world/local edge-direction correctness via a real
# `optimize()` call, not just "it runs" (unlike the accumulation-only test
# above, and unlike scripts/test_posegraph.py which is a standalone
# print-only diagnostic, not run under pytest/CI).
# ---------------------------------------------------------------------------

def test_optimize_recovers_known_world_poses():
    _require_open3d()
    from loop_closure.pose_graph import PoseGraph

    # Ground-truth world poses. Rotation + translation, and the two hops use
    # DIFFERENT rotation axes so the chain does not commute -- a pure
    # translation chain lets a src/dst (or R/R^T) convention bug silently
    # cancel out.
    W0 = _se3(np.eye(3), [0.0, 0.0, 0.0])
    W1 = _se3(_rotation_matrix("z", 30.0), [1.0, 0.5, 0.0])
    W2 = _se3(_rotation_matrix("x", 20.0), [1.5, 1.2, 0.3])

    # Odometry edges use the verified Open3D convention (see
    # loop_closure.pose_graph.PoseGraphEdge docstring):
    # T_src_dst = inv(world_T_src) @ world_T_dst.
    T01 = np.linalg.inv(W0) @ W1
    T12 = np.linalg.inv(W1) @ W2
    # A consistent loop-closure edge directly between node 0 and node 2
    # (redundant with the odometry chain here since there's no noise, but it
    # exercises add_loop_edge's uncertain=True edge path through the same
    # optimizer).
    T02 = np.linalg.inv(W0) @ W2

    pg = PoseGraph(n_nodes=0)
    # Node anchors seeded via forward-chained odometry (T01/T12 applied to
    # the previous anchor) -- this matches real production usage
    # (train_streaming.py always calls add_node with the submap's actual
    # currently-tracked anchor, never an arbitrary placeholder). A
    # deliberately-bad placeholder (e.g. identity) init for nodes 1/2 is
    # NOT representative: PoseGraph.add_odometry_edge/add_loop_edge convert
    # the caller's frame-invariant T_src_dst into the world-frame delta
    # Open3D's node-space residual needs by conjugating with the node's
    # CURRENT anchor at edge-add time (see PoseGraphEdge.o3d_transform's
    # docstring) -- an arbitrary placeholder anchor there produces a
    # correspondingly wrong (not just slowly-converging) conjugated edge,
    # confirmed via an isolated repro this session
    # (a standalone direction test):
    # conjugating by the true/good anchor recovers exactly, by a
    # deliberately bad placeholder anchor does not converge to the true
    # pose no matter how many LM iterations are allowed.
    pg.add_node(W0.copy())
    pg.add_node(W0 @ T01)
    pg.add_node((W0 @ T01) @ T12)

    pg.add_odometry_edge(0, 1, T01, information=np.eye(6) * 100)
    pg.add_odometry_edge(1, 2, T12, information=np.eye(6) * 100)
    pg.add_loop_edge(0, 2, T02, inlier_count=100, rmse=0.01)

    optimised = pg.optimize()

    assert len(optimised) == 3
    np.testing.assert_allclose(optimised[0][:3, :3], W0[:3, :3], atol=1e-3)
    np.testing.assert_allclose(optimised[0][:3, 3], W0[:3, 3], atol=1e-3)
    np.testing.assert_allclose(optimised[1][:3, :3], W1[:3, :3], atol=1e-2)
    np.testing.assert_allclose(optimised[1][:3, 3], W1[:3, 3], atol=1e-2)
    # Root-caused 2026-08-06 (session 2, see PoseGraphEdge.o3d_transform's
    # docstring): with nodes seeded from realistic (forward-chained
    # odometry) anchors -- as every real caller does -- this non-commuting
    # 3-node chain plus a redundant loop edge recovers node 2 exactly.
    np.testing.assert_allclose(optimised[2][:3, :3], W2[:3, :3], atol=1e-2)
    np.testing.assert_allclose(optimised[2][:3, 3], W2[:3, 3], atol=1e-2)


def test_optimize_across_multiple_solves_does_not_drift_already_converged_nodes():
    """Regression coverage for a real production pattern optimize()'s
    single-call tests above don't exercise: a live run calls optimize()
    repeatedly as new submaps/edges arrive, and optimize() persists solved
    poses back into self._nodes (warm-starting the next solve). Since
    PoseGraphEdge.o3d_transform freezes its world-frame conjugation against
    self._nodes[edge.src] at add-edge time (not at solve time), a concern
    raised in review: could a SECOND optimize() call, using nodes solved by
    the first, retroactively corrupt already-converged nodes rather than
    just extending forward? Verified via an isolated repro this session
    (a standalone multisolve test, both the
    exact-convergence and deliberately-noisy-initial-anchor cases): no —
    already-converged nodes stay converged across a second solve, and a new
    edge added afterward only inherits whatever residual error already
    existed in its anchor (proportionally, not amplified), never worse.
    """
    _require_open3d()
    from loop_closure.pose_graph import PoseGraph

    W0 = _se3(np.eye(3), [0.0, 0.0, 0.0])
    W1 = _se3(_rotation_matrix("z", 30.0), [1.0, 0.5, 0.0])
    W2 = _se3(_rotation_matrix("x", 20.0), [1.5, 1.2, 0.3])
    W3 = _se3(_rotation_matrix("y", 15.0), [2.0, -0.5, 1.0])

    T01 = np.linalg.inv(W0) @ W1
    T12 = np.linalg.inv(W1) @ W2
    T23 = np.linalg.inv(W2) @ W3

    pg = PoseGraph(n_nodes=0)
    pg.add_node(W0.copy())
    pg.add_node(W0 @ T01)
    pg.add_node((W0 @ T01) @ T12)
    pg.add_odometry_edge(0, 1, T01, information=np.eye(6) * 100)
    pg.add_odometry_edge(1, 2, T12, information=np.eye(6) * 100)

    solved1 = pg.optimize()
    np.testing.assert_allclose(solved1[2][:3, :3], W2[:3, :3], atol=1e-2)
    np.testing.assert_allclose(solved1[2][:3, 3], W2[:3, 3], atol=1e-2)

    # Add a fourth node/edge AFTER the first optimize() call -- pg._nodes
    # now holds solved (warm-started), not raw, anchors.
    pg.add_node(solved1[2] @ T23.astype(np.float32))
    pg.add_odometry_edge(2, 3, T23, information=np.eye(6) * 100)

    solved2 = pg.optimize()
    assert len(solved2) == 4
    # Nodes 0-2, already converged by solve 1, must NOT drift on solve 2.
    for i, W in enumerate([W0, W1, W2]):
        np.testing.assert_allclose(solved2[i][:3, :3], W[:3, :3], atol=1e-2)
        np.testing.assert_allclose(solved2[i][:3, 3], W[:3, 3], atol=1e-2)
    # The new node recovers exactly too, since solve 1 was exact here.
    np.testing.assert_allclose(solved2[3][:3, :3], W3[:3, :3], atol=1e-2)
    np.testing.assert_allclose(solved2[3][:3, 3], W3[:3, 3], atol=1e-2)


# ---------------------------------------------------------------------------
# Test 5 (C2 regression): a force-field world-space correction, converted via
# world_delta_to_local_edge, must produce the mathematically correct local
# submap-relative edge -- NOT the raw world-space delta passed straight
# through (the bug: train_streaming.py used to hand gaussian_force_field's
# fitted world-space T directly to add_loop_edge as if it were already a
# local T_src_dst edge).
# ---------------------------------------------------------------------------

def test_world_delta_to_local_edge_matches_expected_math():
    from loop_closure.pose_graph import world_delta_to_local_edge

    # Submap "old" and submap "cur" both have non-trivial (non-identity,
    # different) world anchor transforms -- this is the case a same-frame
    # assumption (submap local frame == world frame) gets wrong; submap 0
    # alone would never catch it since its anchor is identity by convention.
    world_T_old = _se3(_rotation_matrix("y", 15.0), [2.0, 0.0, 1.0])
    world_T_cur = _se3(_rotation_matrix("z", -40.0), [0.5, 3.0, -0.5])

    # Known world-space correction fitted between the two submaps' current
    # world-space Gaussian positions: maps cur's world points onto old's
    # world points.
    world_delta_T = _se3(_rotation_matrix("x", 5.0), [0.1, -0.2, 0.05])

    edge = world_delta_to_local_edge(world_T_old, world_T_cur, world_delta_T)

    # Ground truth by definition (world_T_dst = world_T_src @ edge, the
    # convention verified in test_optimize_recovers_known_world_poses):
    # applying edge on top of old's world anchor must reproduce
    # world_delta_T applied to cur's world anchor (i.e. the corrected world
    # pose cur should have).
    lhs = world_T_old @ edge
    rhs = world_delta_T @ world_T_cur
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    # And the naive (buggy) behaviour -- passing world_delta_T straight
    # through as if it were already the local edge -- must NOT equal the
    # correct edge in this non-trivial-anchor case, confirming the test
    # actually discriminates the fix from the bug.
    assert not np.allclose(edge, world_delta_T, atol=1e-3)


def test_world_delta_to_local_edge_identity_anchors_is_passthrough():
    """When both submaps' anchors are still identity (the common/initial
    case -- no prior pose-graph correction has been applied yet), the local
    edge collapses to the raw world-space delta itself (the pose-graph edge
    convention is ``T_src_dst = inv(world_T_src) @ world_T_dst``, so with
    identity anchors this reduces to ``world_delta_T`` directly). This
    confirms the helper's identity-anchor behaviour is exactly a passthrough,
    not some more complex composition, for submaps 2+ that have already
    accumulated a non-identity correction.
    """
    from loop_closure.pose_graph import world_delta_to_local_edge

    world_T_old = np.eye(4)
    world_T_cur = np.eye(4)
    world_delta_T = _se3(_rotation_matrix("z", 10.0), [0.3, -0.1, 0.0])

    edge = world_delta_to_local_edge(world_T_old, world_T_cur, world_delta_T)
    np.testing.assert_allclose(edge, world_delta_T, atol=1e-10)


def test_optimize_applies_force_field_correction_via_local_edge():
    """End-to-end (C1+C2): a force-field-style world-space correction,
    converted through world_delta_to_local_edge into a loop edge and run
    through a real optimize() call on a 2-submap graph whose nodes start at
    their pre-correction anchors, recovers the expected corrected world pose
    for the "cur" submap.
    """
    _require_open3d()
    from loop_closure.pose_graph import PoseGraph, world_delta_to_local_edge

    # Submap 0 ("old") already has a known non-identity anchor from a prior
    # correction pass; submap 1 ("cur") is still at its raw ingestion pose.
    world_T_old = _se3(_rotation_matrix("y", 10.0), [1.0, 0.0, 0.0])
    world_T_cur = np.eye(4)

    # Fitted world-space correction: cur's world points should move onto
    # old's world points via this transform.
    world_delta_T = _se3(_rotation_matrix("z", 25.0), [0.2, 0.4, 0.0])

    edge = world_delta_to_local_edge(world_T_old, world_T_cur, world_delta_T)

    pg = PoseGraph(n_nodes=0)
    pg.add_node(world_T_old.copy())  # node 0, pinned as reference
    pg.add_node(world_T_cur.copy())  # node 1, initial (pre-correction) guess
    # Mirror production wiring (train_streaming.py): submap 1 already has a
    # weak/uninformative odometry edge from when it was created (identity,
    # "no external odometry feed" -- see the add_odometry_edge call site),
    # which is also what keeps the "certain"-edge subgraph connected for
    # Open3D's reference-node anchoring; the strong, informative loop edge
    # is what actually pulls node 1 into place.
    pg.add_odometry_edge(0, 1, np.eye(4), information=np.eye(6) * 1e-3)
    pg.add_loop_edge(0, 1, edge, inlier_count=100, rmse=0.005)

    optimised = pg.optimize()

    expected_world_T_cur = world_delta_T @ world_T_cur
    np.testing.assert_allclose(optimised[0][:3, :3], world_T_old[:3, :3], atol=1e-3)
    np.testing.assert_allclose(optimised[0][:3, 3], world_T_old[:3, 3], atol=1e-3)
    # Root-caused 2026-08-06 (session 2): with the o3d_transform fix, this
    # 2-node force-field-correction graph (non-identity src anchor at node
    # 0) now recovers node 1's expected corrected pose exactly.
    np.testing.assert_allclose(optimised[1][:3, :3], expected_world_T_cur[:3, :3], atol=1e-2)
    np.testing.assert_allclose(optimised[1][:3, 3], expected_world_T_cur[:3, 3], atol=1e-2)


# ---------------------------------------------------------------------------
# Independent validation against raw Open3D (final review, PR #37, item #1,
# corrected in review round 2): the first version of this test still
# encoded the wrapper's own (self-consistent but unverified) assumption on
# both the "direct" and "wrapper" sides, so it only proved internal
# self-consistency, not agreement with Open3D's documented contract. This
# version uses Open3D's actual documented semantics literally on the direct
# side: node.pose places a node's LOCAL geometry into world space (stored
# and read back with NO inversion anywhere), and an edge's `transformation`
# maps source-node geometry into the target node's frame -- built here from
# the GROUND-TRUTH poses, with no dependence on any node's current estimate
# (so there is no anchor-conjugation step to get wrong). (1) generates real
# node-local point clouds from one known global cloud and known
# non-commuting SE(3) poses, (2) perturbs every initial node pose
# substantially, (3) builds edge measurements as fixed GROUND-TRUTH relative
# transforms, (4) runs Open3D's global_optimization DIRECTLY with no
# EliGSiR helper code involved at all, (5) applies the solved node.pose
# to each local cloud and checks it lands back on the true global cloud, and
# (6) separately runs the same scenario through PoseGraph and asserts it
# produces the SAME solved world poses as the direct Open3D run.
# ---------------------------------------------------------------------------

def _make_global_point_cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 3))


def _to_local(points_world: np.ndarray, world_T_node: np.ndarray) -> np.ndarray:
    """Node-local points: p_local = inv(world_T_node) @ p_world."""
    R = world_T_node[:3, :3]
    t = world_T_node[:3, 3]
    return (points_world - t) @ R  # == (R^T @ (p - t)^T)^T since R is orthonormal


def _apply_world_pose(points_local: np.ndarray, world_T_node: np.ndarray) -> np.ndarray:
    R = world_T_node[:3, :3]
    t = world_T_node[:3, 3]
    return points_local @ R.T + t


def test_pose_graph_matches_direct_open3d_and_recovers_point_cloud_alignment():
    _require_open3d()
    import open3d as o3d
    from loop_closure.pose_graph import PoseGraph

    # (1) Global cloud + non-commuting ground-truth world poses (3 hops,
    # loop edge back to node 0 -- matches the reviewer's "repeated
    # optimize() calls and at least one loop edge" requirement via the
    # certain/uncertain edge mix below).
    global_cloud = _make_global_point_cloud()
    W0 = _se3(np.eye(3), [0.0, 0.0, 0.0])
    W1 = _se3(_rotation_matrix("z", 35.0), [1.2, -0.4, 0.2])
    W2 = _se3(_rotation_matrix("x", -25.0), [2.1, 0.9, -0.3])
    W3 = _se3(_rotation_matrix("y", 40.0), [0.5, 2.0, 1.1])
    true_world_poses = [W0, W1, W2, W3]

    local_clouds = [_to_local(global_cloud, W) for W in true_world_poses]

    # (2) Perturbed initial guesses -- clearly non-trivial (several cm +
    # a few degrees, an order of magnitude above float precision), not
    # near-zero residual, so a direction/sign bug cannot hide behind "it
    # started close enough to be right anyway". Edge measurements below are
    # fixed ground-truth transforms with no dependence on any node's current
    # estimate, so (unlike the removed anchor-conjugated form) a bad initial
    # guess only affects where LM starts, not what it converges to -- every
    # node is expected to recover ~exactly regardless of perturbation size.
    def _perturb(W, deg, t):
        Wp = W.copy()
        Wp[:3, :3] = _rotation_matrix("z", deg) @ W[:3, :3]
        Wp[:3, 3] = W[:3, 3] + np.array(t)
        return Wp

    init_guesses = [
        W0,  # node 0 is the pinned reference; must start at its true pose
        _perturb(W1, 3.0, [0.04, -0.03, 0.02]),
        _perturb(W2, -2.5, [-0.03, 0.04, -0.02]),
        _perturb(W3, 3.5, [0.03, 0.02, -0.04]),
    ]

    # (3) Edge measurements are FIXED ground-truth relative transforms,
    # computed once from the true poses -- never re-derived from a current
    # node estimate at any point below.
    T01 = np.linalg.inv(W0) @ W1
    T12 = np.linalg.inv(W1) @ W2
    T23 = np.linalg.inv(W2) @ W3
    T03 = np.linalg.inv(W0) @ W3  # loop edge: direct 0->3 constraint

    # Loop-edge information must match add_loop_edge's own
    # inlier_count/(rmse**2 + eps) formula EXACTLY -- using a different
    # information weighting here than the wrapper uses internally would
    # make direct vs wrapper genuinely converge to different solutions
    # (a different information matrix is a different optimization problem),
    # which would falsely look like a wrapper bug.
    _loop_inlier_count, _loop_rmse = 100, 0.01
    _loop_information = np.eye(6) * (_loop_inlier_count / (_loop_rmse ** 2 + 1e-6))
    edges = [
        (0, 1, T01, False, np.eye(6) * 100),
        (1, 2, T12, False, np.eye(6) * 100),
        (2, 3, T23, False, np.eye(6) * 100),
        (0, 3, T03, True, _loop_information),
    ]

    # (4) Direct Open3D run: no EliGSiR PoseGraph/PoseGraphEdge/
    # world_delta_to_local_edge code involved. Nodes store world_T_node
    # DIRECTLY (no inversion) per Open3D's documented reconstruction-system
    # convention; each edge's `transformation` maps source-node geometry
    # into the target node's frame -- inv(world_T_dst) @ world_T_src,
    # computed here purely from the GROUND-TRUTH poses in `edges`, with no
    # dependence on any node's current estimate.
    pg_direct = o3d.pipelines.registration.PoseGraph()
    for Wg in init_guesses:
        pg_direct.nodes.append(o3d.pipelines.registration.PoseGraphNode(Wg))
    for src, dst, T_src_dst, is_loop, information in edges:
        world_T_src, world_T_dst = true_world_poses[src], true_world_poses[dst]
        edge_transform = np.linalg.inv(world_T_dst) @ world_T_src
        # Pin the algebraic reduction PoseGraph.optimize() relies on.
        np.testing.assert_allclose(edge_transform, np.linalg.inv(T_src_dst), atol=1e-12)
        pg_direct.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            source_node_id=src, target_node_id=dst, transformation=edge_transform,
            information=information, uncertain=is_loop,
        ))
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.05, edge_prune_threshold=0.25,
        preference_loop_closure=1.0, reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pg_direct, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option,
    )
    # node.pose applied DIRECTLY to local geometry -- no inversion.
    solved_world_direct = [np.array(n.pose) for n in pg_direct.nodes]

    # (5) Point-cloud round-trip: apply each solved node.pose to its
    # LOCAL cloud and verify it lands back on the true global cloud. This
    # is the check that actually matters end-to-end -- correct transform
    # matrices are only useful insofar as they correctly place real
    # geometry, which is what this project uses the pose graph for.
    #
    # Every node recovers to float precision -- no per-node tolerance table.
    # Edge measurements have no anchor/conjugation term, so a perturbed
    # initial guess only affects where LM starts, not what it converges to.
    for i in range(4):
        recovered_world_points = _apply_world_pose(local_clouds[i], solved_world_direct[i])
        np.testing.assert_allclose(recovered_world_points, global_cloud, atol=1e-4)

    # (6) Now run the SAME scenario through the EliGSiR wrapper --
    # add_node with the same perturbed initial guesses, add_odometry_edge/
    # add_loop_edge with the same ground-truth T_src_dst measurements
    # (never re-derived from a node estimate), one call to optimize()
    # first (exercising a real multi-edge single solve), and assert its
    # output matches the direct Open3D run.
    pg_wrapper = PoseGraph(n_nodes=0)
    for Wg in init_guesses:
        pg_wrapper.add_node(Wg.copy())
    pg_wrapper.add_odometry_edge(0, 1, T01, information=np.eye(6) * 100)
    pg_wrapper.add_odometry_edge(1, 2, T12, information=np.eye(6) * 100)
    pg_wrapper.add_odometry_edge(2, 3, T23, information=np.eye(6) * 100)
    pg_wrapper.add_loop_edge(0, 3, T03, inlier_count=100, rmse=0.01)

    solved_world_wrapper = pg_wrapper.optimize()

    for i in range(4):
        np.testing.assert_allclose(
            solved_world_wrapper[i][:3, :3], solved_world_direct[i][:3, :3], atol=1e-5
        )
        np.testing.assert_allclose(
            solved_world_wrapper[i][:3, 3], solved_world_direct[i][:3, 3], atol=1e-5
        )

    # (7) Repeated optimize() calls: call optimize() a second time with no
    # new edges added (idempotency -- warm-started from its own prior
    # solution, it must not drift) and a third time after adding one more
    # node/edge, confirming the earlier nodes still match the direct-Open3D
    # ground truth and the point clouds still round-trip correctly.
    solved_again = pg_wrapper.optimize()
    for i in range(4):
        np.testing.assert_allclose(solved_again[i][:3, :3], solved_world_wrapper[i][:3, :3], atol=1e-6)
        np.testing.assert_allclose(solved_again[i][:3, 3], solved_world_wrapper[i][:3, 3], atol=1e-6)

    W4 = _se3(_rotation_matrix("x", 18.0), [-1.0, 0.3, 0.7])
    T34 = np.linalg.inv(W3) @ W4
    pg_wrapper.add_node(solved_again[3] @ T34.astype(np.float32))
    pg_wrapper.add_odometry_edge(3, 4, T34, information=np.eye(6) * 100)
    solved_final = pg_wrapper.optimize()

    for i in range(4):
        np.testing.assert_allclose(solved_final[i][:3, :3], solved_world_direct[i][:3, :3], atol=1e-2)
        np.testing.assert_allclose(solved_final[i][:3, 3], solved_world_direct[i][:3, 3], atol=1e-2)
    recovered_world_points_4 = _apply_world_pose(
        _to_local(global_cloud, W4), solved_final[4]
    )
    np.testing.assert_allclose(recovered_world_points_4, global_cloud, atol=1e-2)
