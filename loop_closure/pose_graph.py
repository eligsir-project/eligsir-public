"""Pose-graph loop closure (Phase 13).

Uses force-field constraints from :mod:`loop_closure.gaussian_force_field`
to build loop-closure edges and optimise submap transforms via an Open3D
pose-graph backend.

Phase 12 submap metadata (per-Gaussian ``submap_id`` + ``anchor_keyframe_id``)
is required before Phase 13 corrections can be applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PoseGraphEdge:
    """A directed edge in the pose graph.

    Attributes:
        src: Source submap/node index.
        dst: Destination submap/node index.
        T_src_dst: 4×4 relative transform between src and dst, in the
            frame-invariant, body-relative convention every caller in this
            codebase already uses to compute it: ``world_T_dst = world_T_src
            @ T_src_dst``, i.e. ``T_src_dst = inv(world_T_src) @
            world_T_dst``. :meth:`PoseGraph.optimize` hands Open3D
            ``inv(T_src_dst)`` -- see its docstring for the derivation.
        information: 6×6 information matrix (inverse covariance).
        is_loop: True for loop-closure edges, False for odometry.
    """

    src: int
    dst: int
    T_src_dst: np.ndarray
    information: np.ndarray
    is_loop: bool = False


def world_delta_to_local_edge(
    world_T_src: np.ndarray,
    world_T_dst: np.ndarray,
    world_delta_T: np.ndarray,
) -> np.ndarray:
    """Convert a world-space correction delta into a local pose-graph edge.

    Some correction sources (e.g. ``loop_closure.gaussian_force_field``'s
    ``fit_weighted_se3``) fit a transform directly between WORLD-SPACE point
    sets belonging to two different submaps -- that fitted transform is a
    world-space delta, NOT the submap-relative ``T_src_dst`` edge that
    :meth:`PoseGraph.add_odometry_edge`/:meth:`PoseGraph.add_loop_edge`
    expect. Conflating the two silently corrupts the pose graph whenever a
    submap's own local/anchor frame differs from world frame (i.e. any
    submap other than the very first, un-corrected one).

    ``world_delta_T`` is defined as mapping ``dst``'s current world-space
    points onto ``src``'s current world-space points (the direction
    ``fit_weighted_se3(src=current/dst, tgt=old/src)`` fits). Applying it to
    ``dst``'s current world anchor gives the world pose ``dst`` *should*
    have to align with ``src``'s (trusted) frame; expressing that relative
    to ``src``'s own current anchor yields the local edge the pose graph
    expects.

    Convention (see :class:`PoseGraphEdge`'s docstring for the re-verified
    direction): an edge stored as ``PoseGraphEdge(src, dst, T)`` constrains
    ``world_T_dst = world_T_src @ T``, i.e. ``T = inv(world_T_src) @
    world_T_dst``. Do not flip this formula without re-running the isolated
    single-edge repro referenced there.

    Args:
        world_T_src: Current 4x4 world anchor transform of the src submap.
        world_T_dst: Current 4x4 world anchor transform of the dst submap.
        world_delta_T: Fitted 4x4 world-space correction mapping dst's
            current world points onto src's current world points.

    Returns:
        4x4 local ``T_src_dst`` edge transform, expressed in each submap's
        own anchor frame.
    """
    target_world_T_dst = world_delta_T @ world_T_dst
    src_T_dst = np.linalg.inv(world_T_src) @ target_world_T_dst
    return src_T_dst


class PoseGraph:
    """Simple pose graph for submap-level loop-closure optimisation.

    Wraps Open3D's ``PoseGraph`` and ``global_optimization``.

    Args:
        n_nodes: Number of submap nodes.
    """

    def __init__(self, n_nodes: int = 0) -> None:
        self._nodes: List[np.ndarray] = []
        self._edges: List[PoseGraphEdge] = []
        for _ in range(n_nodes):
            self._nodes.append(np.eye(4, dtype=np.float64))

    def add_node(self, T_world_submap: np.ndarray) -> int:
        """Add a new submap node.

        Args:
            T_world_submap: Initial 4×4 pose of the submap in world frame.

        Returns:
            Index of the new node.
        """
        idx = len(self._nodes)
        self._nodes.append(T_world_submap.astype(np.float64))
        return idx

    def add_odometry_edge(
        self,
        src: int,
        dst: int,
        T_src_dst: np.ndarray,
        information: Optional[np.ndarray] = None,
    ) -> None:
        """Add an odometry edge between consecutive submaps.

        Args:
            src: Source node index.
            dst: Destination node index.
            T_src_dst: Relative transform from src to dst.
            information: 6×6 information matrix. Identity if None.
        """
        if information is None:
            information = np.eye(6, dtype=np.float64)
        self._edges.append(
            PoseGraphEdge(
                src=src,
                dst=dst,
                T_src_dst=T_src_dst.astype(np.float64),
                information=information,
                is_loop=False,
            )
        )

    def add_loop_edge(
        self,
        src: int,
        dst: int,
        T_src_dst: np.ndarray,
        inlier_count: int,
        rmse: float,
        alpha_coverage: float = 1.0,
        support_confidence: float = 1.0,
    ) -> None:
        """Add a loop-closure edge weighted by quality metrics.

        Information is proportional to ``inlier_count / (rmse² + ε)
        × alpha_coverage × support_confidence``.

        Args:
            src: Source node index.
            dst: Destination node index.
            T_src_dst: Fitted relative transform from force field.
            inlier_count: Number of Gaussian inlier pairs.
            rmse: Force-field alignment RMSE.
            alpha_coverage: Fraction of rendered pixels with alpha > threshold.
            support_confidence: Mean support confidence of matched Gaussians.
        """
        info_scale = (
            inlier_count / (rmse ** 2 + 1e-6) * alpha_coverage * support_confidence
        )
        information = np.eye(6, dtype=np.float64) * info_scale
        self._edges.append(
            PoseGraphEdge(
                src=src,
                dst=dst,
                T_src_dst=T_src_dst.astype(np.float64),
                information=information,
                is_loop=True,
            )
        )

    def optimize(self) -> List[np.ndarray]:
        """Run pose-graph optimisation using Open3D.

        Returns:
            List of optimised 4×4 submap transforms (world_T_submap).

        Raises:
            ImportError: If Open3D is not available.
        """
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "loop_closure.pose_graph requires Open3D. "
                "Install it: pip install open3d"
            ) from exc

        # Open3D convention (settled empirically 2026-08-06, session 2 --
        # see the isolated pg_switch_gate.py repro this commit is based on):
        # PoseGraphNode.pose is the transform that places a node's LOCAL
        # geometry into world/global space, i.e. exactly this class's own
        # T_world_submap. No inversion at either boundary. Each edge's
        # `transformation` maps source-node geometry into the target node's
        # frame: inv(world_T_dst) @ world_T_src, which (since
        # world_T_dst = world_T_src @ T_src_dst) reduces ALGEBRAICALLY to
        # inv(T_src_dst) -- a pure function of the edge's own
        # frame-invariant measurement, with no dependence on any node's
        # current pose estimate. That anchor-independence is why this form
        # is used: the previous world-delta form
        # (anchor_src @ T_src_dst @ inv(anchor_src), removed in this commit)
        # ALSO recovered ground truth exactly when its conjugation anchor
        # was good, but its accuracy degraded in proportion to how far that
        # anchor sat from the truth -- so a node's initial-guess quality
        # silently leaked into its outgoing edges' measurements. Both forms
        # were verified against known ground truth with non-commuting poses,
        # non-identity reference anchors, and deliberately-bad initial
        # guesses; this one needs no anchor at all.
        pg = o3d.pipelines.registration.PoseGraph()
        for T in self._nodes:
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))

        for edge in self._edges:
            e = o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=edge.src,
                target_node_id=edge.dst,
                transformation=np.linalg.inv(edge.T_src_dst),
                information=edge.information,
                uncertain=edge.is_loop,
            )
            pg.edges.append(e)

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            preference_loop_closure=1.0,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pg,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        # node.pose IS T_world_submap -- no conversion needed (see the
        # convention comment above the node loop).
        solved = [np.array(n.pose, dtype=np.float64) for n in pg.nodes]
        # Root-cause fix (Phase 2b, 2026-08-06): persist the solved poses
        # back into self._nodes. Previously this method only READ
        # self._nodes to build the initial guess and never wrote the result
        # back -- every subsequent call re-derived the WHOLE graph from
        # scratch using each node's raw, never-corrected creation-time
        # anchor, discarding all prior convergence. Confirmed via
        # --loop_closure_debug: node drift grew monotonically with chain
        # length (up to ~6.4m by 38 nodes) even with ZERO loop edges present
        # in the graph at every one of those calls -- i.e. purely from
        # repeatedly re-solving an already energy-consistent odometry chain
        # from the same stale initial guess. Warm-starting from the last
        # converged state (standard incremental pose-graph practice) means
        # each call only has to account for genuinely NEW edges since the
        # last solve, not re-derive everything.
        self._nodes = [T.astype(np.float64) for T in solved]
        return [np.array(T, dtype=np.float32) for T in solved]

    def get_node_count(self) -> int:
        """Return the number of nodes in the graph.

        Returns:
            Node count.
        """
        return len(self._nodes)

    def get_edge_count(self, loop_only: bool = False) -> int:
        """Return the number of edges.

        Args:
            loop_only: If True, count only loop-closure edges.

        Returns:
            Edge count.
        """
        if loop_only:
            return sum(1 for e in self._edges if e.is_loop)
        return len(self._edges)

    def debug_snapshot(self) -> dict:
        """Diagnostic dump of the graph's current state (Phase 2b, 2026-08-06
        -- investigating why a corrected submap's pose can end up far outside
        the scene's real extent). Not used by any control-flow path; callers
        log this to a JSONL file, never print it inline (this is a per-node/
        per-edge dump, not a summary).

        Returns:
            ``{"node_translations": [[x,y,z], ...], "edges": [{"src", "dst",
            "is_loop", "information_trace"}, ...]}`` -- translation only (not
            full 4x4) to keep log lines small; ``information_trace`` is
            ``np.trace(information)``, a single number comparable across
            edges regardless of their exact information-matrix shape.
        """
        return {
            "node_translations": [n[:3, 3].tolist() for n in self._nodes],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "is_loop": e.is_loop,
                    "information_trace": float(np.trace(e.information)),
                }
                for e in self._edges
            ],
        }
