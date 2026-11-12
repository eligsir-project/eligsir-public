"""Unit tests for novelty-gated max_gap admission (Component 2).

When ``max_gap_require_novelty`` is on, a frame that would be force-admitted by
``max_gap`` must ALSO reveal new geometry (coverage new-voxel-frac >=
min_new_voxel_frac). A redundant forced frame is deferred (and then rejected by
the coverage gate) instead of inflating the training-view count. A hard ceiling
(gap >= max_gap*max_gap_hard_mult) still force-admits unconditionally.

CPU-only, mirroring test_keyframe_coverage_gain_gate.py: occupancy is a stub and
the camera is a tiny fake exposing pinhole intrinsics + identity extrinsics.
"""
from __future__ import annotations

import numpy as np

from eligsir.mapping.scheduler.keyframes import (
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    FrameInfo,
)


class _FakeCam:
    def __init__(self):
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 8.0
        self.cy = 8.0
        self.c2w = np.eye(4, dtype=np.float32)
        self.world_view_transform = np.eye(4, dtype=np.float32)


def _frame(index: int, x: float):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index)


def _depth_const(value: float):
    arr = np.full((16, 16), value, dtype=np.float32)
    return lambda frame: arr


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,        # disable rotation gate
        max_gap=10,
        coverage_gain_gate=True,
        min_new_voxel_frac=0.5,
        coverage_gain_depth_stride=4,
        depth_max=8.0,
        max_gap_require_novelty=True,
        max_gap_hard_mult=4.0,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _make_admitter(cfg, occupancy_fn):
    adm = KeyframeAdmitter(
        config=cfg,
        occupancy_fn=occupancy_fn,
        load_depth_m=_depth_const(2.0),
    )
    adm.note_train_frame(_frame(0, 0.0))   # baseline at index 0
    return adm


_ALL_OCCUPIED = lambda pts, vox: np.ones(len(pts), dtype=bool)
_ALL_UNOCCUPIED = lambda pts, vox: np.zeros(len(pts), dtype=bool)


def test_redundant_forced_frame_is_deferred():
    """gap=12 (>= max_gap 10) would force-admit, but the frame is fully redundant
    (0 new voxels) and NOT moved -> novelty gating downgrades forcing -> the
    coverage gate rejects it instead of a blind max_gap admit."""
    adm = _make_admitter(_cfg(), _ALL_OCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert not decision.admitted
    assert decision.reason != "max_gap"


def test_novel_forced_frame_still_admits():
    """gap=12 forced AND the frame reveals all-new geometry -> novelty passes ->
    admitted (reason stays max_gap)."""
    adm = _make_admitter(_cfg(), _ALL_UNOCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted
    assert decision.reason == "max_gap"


def test_hard_ceiling_forces_even_if_redundant():
    """gap=40 (>= max_gap*hard_mult = 10*4) force-admits unconditionally even
    for a fully-redundant frame — anti-starvation safety floor."""
    adm = _make_admitter(_cfg(), _ALL_OCCUPIED)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(40, 0.0))
    assert decision.admitted
    assert decision.reason == "max_gap"
    assert decision.forced


def test_flag_off_is_byte_identical():
    """With max_gap_require_novelty=False a redundant forced frame admits via the
    plain max_gap fast-path (occupancy never consulted)."""
    called = {"n": 0}

    def _spy(pts, vox):
        called["n"] += 1
        return np.ones(len(pts), dtype=bool)

    adm = _make_admitter(_cfg(max_gap_require_novelty=False), _spy)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(12, 0.0))
    assert decision.admitted
    assert decision.reason == "max_gap"
    assert called["n"] == 0
