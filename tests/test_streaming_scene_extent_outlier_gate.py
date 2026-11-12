"""Tests for update_cameras_extent()'s outlier-rejection gate (W3-48,
2026-08-10) -- root-caused in
docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md as
the real floor2 scale-explosion amplifier: a single poorly-converged pose
used to permanently inflate cameras_extent via an unguarded, never-shrinks
max() over all camera positions ever seen.

Exercises the real StreamingScene.update_cameras_extent method directly
against a minimal fake object carrying only the state that method touches --
avoids constructing a full StreamingScene (heavy: gaussians model, dataset
I/O, etc.), matching how tests elsewhere in this repo stub this class.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import torch

from scene.streaming_scene import StreamingScene


class _FakeExtentScene:
    """Carries exactly the attributes update_cameras_extent() reads/writes."""

    def __init__(self, ceiling: float = 1.5, corrob: int = 2):
        self.train_cameras: List[SimpleNamespace] = []
        self._camera_centers_gpu: list = []
        self._extent_update_counter = 0
        self.cameras_extent = 1.0
        self._extent_pending_growth: Optional[float] = None
        self._extent_pending_corroborations = 0
        self._extent_pending_driver_idx: Optional[int] = None
        self._extent_outlier_rejections_total = 0
        self.args = SimpleNamespace(
            streaming_extent_growth_ceiling_multiplier=ceiling,
            streaming_extent_outlier_corroboration=corrob,
        )

    def add_window(self, positions) -> None:
        """Append new camera positions, then force exactly one recompute
        using the fully-synced cache for this window.

        Real callers invoke update_cameras_extent() once per admitted frame,
        so the every-8th-call recompute throttle would otherwise race a
        partial cache while a window's positions are still being added one
        at a time (an artifact of testing this incrementally from an empty,
        cameras_extent==1.0 starting state -- in production, bootstrap
        already establishes a realistic non-degenerate starting extent via
        _estimate_cameras_extent() before this per-frame path ever runs, so
        that race is not a real production concern). Aligning the counter
        so the window's positions are fully appended before the one call
        that actually recomputes avoids that test-harness artifact.
        """
        for pos in positions:
            self.train_cameras.append(
                SimpleNamespace(camera_center=torch.tensor(pos, dtype=torch.float32))
            )
        self._extent_update_counter = ((self._extent_update_counter // 8) + 1) * 8 - 1
        StreamingScene.update_cameras_extent(self)


def _spread_along_x(n: int, lo: float, hi: float):
    step = (hi - lo) / (n - 1)
    return [(lo + i * step, 0.0, 0.0) for i in range(n)]


def test_first_window_bypasses_gate_and_accepts_unconditionally():
    scene = _FakeExtentScene()
    scene.add_window(_spread_along_x(8, -5.0, 5.0))
    assert scene.cameras_extent == 5.0
    assert scene._extent_pending_growth is None
    assert scene._extent_pending_corroborations == 0


def test_modest_growth_within_ceiling_accepts_immediately():
    scene = _FakeExtentScene(ceiling=1.5, corrob=2)
    scene.add_window(_spread_along_x(8, -5.0, 5.0))
    baseline = scene.cameras_extent
    assert baseline == 5.0

    # A couple of new cameras extending the line only slightly (well within
    # 1.5x of baseline) must be accepted on the very next recompute, no
    # staging.
    scene.add_window([(5.3, 0.0, 0.0), (5.4, 0.0, 0.0)])
    assert scene.cameras_extent > baseline
    assert scene.cameras_extent < baseline * 1.5
    assert scene._extent_pending_growth is None
    assert scene._extent_pending_corroborations == 0


def test_single_outlier_jump_is_withheld_and_never_self_corroborates():
    scene = _FakeExtentScene(ceiling=1.5, corrob=2)
    scene.add_window(_spread_along_x(8, -5.0, 5.0))
    baseline = scene.cameras_extent
    assert baseline == 5.0

    # A single window of new cameras all far away (a repeated bad pose, or
    # one SLAM glitch admitted several times in a row) -- beyond the ceiling.
    scene.add_window([(100.0, 0.0, 0.0)] * 2)
    assert scene.cameras_extent == baseline, "outlier jump must not be accepted immediately"
    assert scene._extent_pending_corroborations == 1
    first_driver = scene._extent_pending_driver_idx
    assert first_driver is not None

    # Further recompute windows with NO new cameras: the cache still
    # contains the stale outlier points, so a naive "is the extent still
    # large" check would keep re-triggering on the SAME cached point forever
    # and eventually accept it. The driver-index check must catch this: the
    # argmax driver never changes (nothing new arrived), so corroboration
    # must NOT accumulate.
    for _ in range(4):
        scene.add_window([])  # no new cameras, just forces another recompute
    assert scene.cameras_extent == baseline, (
        "a single stale outlier must never get accepted no matter how many "
        "times its own cached position is re-examined"
    )
    assert scene._extent_pending_driver_idx == first_driver
    assert scene._extent_pending_corroborations == 1, (
        "corroboration count must not climb from the SAME point recurring"
    )


def test_sustained_growth_from_distinct_new_cameras_is_accepted():
    scene = _FakeExtentScene(ceiling=1.5, corrob=2)
    scene.add_window(_spread_along_x(8, -5.0, 5.0))
    baseline = scene.cameras_extent
    assert baseline == 5.0

    # Window 2: a real jump (e.g. a corridor continuing much further) --
    # staged, not yet accepted.
    scene.add_window([(100.0, 0.0, 0.0)] * 2)
    assert scene.cameras_extent == baseline
    assert scene._extent_pending_corroborations == 1

    # Window 3: genuinely NEW, more distant cameras arrive (the trajectory
    # keeps going) -- a DIFFERENT camera is now the extent-driver, which is
    # real corroborating evidence, not the same stale point.
    scene.add_window([(200.0, 0.0, 0.0)] * 2)
    assert scene.cameras_extent > baseline, "corroborated growth must be accepted"
    assert scene.cameras_extent > 50.0
    assert scene._extent_pending_growth is None
    assert scene._extent_pending_corroborations == 0


def test_gate_disabled_reproduces_pre_w3_48_unconditional_growth():
    scene = _FakeExtentScene(ceiling=0.0, corrob=2)  # ceiling<=0 disables the gate
    scene.add_window(_spread_along_x(8, -5.0, 5.0))
    baseline = scene.cameras_extent
    assert baseline == 5.0

    # With the gate off, even a single-window outlier jump is accepted
    # immediately -- matches the original, pre-fix behavior exactly.
    scene.add_window([(100.0, 0.0, 0.0)] * 2)
    assert scene.cameras_extent > baseline
    assert scene._extent_pending_growth is None
