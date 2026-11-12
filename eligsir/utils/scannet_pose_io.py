"""Torch-free ScanNet pose-file reader (issue #449, same split pattern as #435).

``scene/readers/scannet.py`` is a torch carrier: it imports
``scene.readers.common`` / ``utils.graphics_utils``, and living under the
``scene`` package it also triggers ``scene/__init__.py`` (which imports the
Gaussian models). Reading a ScanNet ``pose/<id>.txt`` needs none of that --
it is twelve lines of numpy.

``utils/trajectory_eval.py`` used to reach for it as
``from scene.readers.scannet import _read_scannet_pose`` inside a bare
``except Exception: return None``, so in a torch-free environment the
ScanNet ground-truth auto-discovery silently reported "no GT" instead of
failing -- the wrong-answer failure mode #434/#449 are about. It now imports
from here; ``scene/readers/scannet.py`` re-exports the name, so every other
call site is unchanged.
"""
from __future__ import annotations

import numpy as np


def _read_scannet_pose(path):
    """Read a ScanNet 4x4 camera-to-world pose file. ``None`` if unusable."""
    try:
        pose = np.loadtxt(path).astype(np.float32)
    except (FileNotFoundError, OSError):
        return None
    if pose.shape != (4, 4):
        return None
    if not np.isfinite(pose).all():
        return None
    if abs(np.linalg.det(pose[:3, :3])) < 1e-6:
        return None
    return pose
