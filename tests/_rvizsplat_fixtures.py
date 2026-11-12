"""Deterministic snapshot builder + golden-fixture catalogue for the
RVizSplat CDR serializer.

Shared by:
  * ``tests/test_rvizsplat_export.py``    — byte-identical regression test
  * ``tests/test_rvizsplat_ros_roundtrip.py`` — rclpy deserialize roundtrip
  * ``scripts``/manual regeneration       — dump golden ``.bin`` files

The snapshots are fully deterministic (seeded RNG) so the serialized bytes
are reproducible across machines and Python/numpy versions.  Fixtures are
committed under ``tests/fixtures/`` and assert the serializer output never
changes silently (the strided-fill rewrite must reproduce them exactly).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from eligsir.viewers.rvizsplat_export import RVizSplatSnapshot

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FRAME_ID = "map"
STAMP_SEC = 1234
STAMP_NANOSEC = 567890123

# (sh_degree, N) cases.  deg 0/2 → K odd (p=4 every record);
# deg 1/3 → K even (p=4 record 0, p=0 records 1..N-1).  N=1/2/5 exercises
# the single-record, record-0-vs-1, and multi-record fill paths.
CASES: List[Tuple[int, int]] = [
    (deg, n) for deg in (0, 1, 2, 3) for n in (1, 2, 5)
]


def make_snapshot(sh_degree: int, n: int) -> RVizSplatSnapshot:
    """Build a deterministic snapshot for ``(sh_degree, n)``.

    Quaternions are random-but-normalized; positions/scales/opacities/SH
    are seeded so the wire bytes are fully reproducible.
    """
    rng = np.random.default_rng(1000 * sh_degree + n)
    k = (sh_degree + 1) ** 2 * 3

    positions = rng.standard_normal((n, 3)).astype(np.float32)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    scales = (0.01 + 0.2 * rng.random((n, 3))).astype(np.float32)
    opacities = rng.random(n).astype(np.float32)
    sh = rng.standard_normal((n, k)).astype(np.float32)

    return RVizSplatSnapshot(
        positions=positions,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        spherical_harmonics=sh,
        sh_degree=sh_degree,
    )


def fixture_path(sh_degree: int, n: int) -> str:
    return os.path.join(FIXTURE_DIR, f"splat_cdr_deg{sh_degree}_n{n}.bin")
