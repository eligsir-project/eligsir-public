"""Lightweight factory helpers for loop-closure objects.

Kept separate from train_streaming so tests can import without GPU deps.
"""

from __future__ import annotations

import numpy as np


def maybe_create_pose_graph(args):
    """Return a new PoseGraph seeded with node 0, or None if disabled.

    Args:
        args: Argument namespace; reads ``loop_closure_enabled`` (default False).

    Returns:
        A :class:`loop_closure.pose_graph.PoseGraph` initialised with one node
        at the world origin, or ``None`` when the feature is disabled.
    """
    if not getattr(args, "loop_closure_enabled", False):
        return None
    try:
        from loop_closure.pose_graph import PoseGraph
        pg = PoseGraph(n_nodes=0)
        pg.add_node(np.eye(4))  # submap 0 at world origin
        return pg
    except Exception:
        return None
