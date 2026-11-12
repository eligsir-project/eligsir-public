"""Unit tests for the depth-coverage-gain keyframe-admission gate (Track A).

Ported alongside the gate from
``feat/offline-gap-and-floater-consensus``. CPU-only: the occupancy
hash is a stub callable and the camera is a tiny fake exposing the
intrinsics + ``c2w`` / ``world_view_transform`` the gate reads. No CUDA
render is involved (the gate runs before the render gate; we keep the
frame "moved" so the cheap motion gate passes, then let the coverage
gate decide).
"""
from __future__ import annotations

import numpy as np

from eligsir.mapping.scheduler.keyframes import (
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    FrameInfo,
)


class _FakeCam:
    """Minimal camera exposing pinhole intrinsics + identity extrinsics."""

    def __init__(self):
        self.fx = 100.0
        self.fy = 100.0
        self.cx = 8.0
        self.cy = 8.0
        self.c2w = np.eye(4, dtype=np.float32)
        # Presence-only: the gate gates on hasattr(cam, "world_view_transform").
        self.world_view_transform = np.eye(4, dtype=np.float32)


def _frame(index: int, x: float):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    return FrameInfo(c2w=c2w, index=index)


def _depth_const(value: float):
    """A 16x16 constant-depth map loader."""
    arr = np.full((16, 16), value, dtype=np.float32)
    return lambda frame: arr


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,   # disable rotation gate
        max_gap=10_000,         # disable forced max_gap admits
        coverage_gain_gate=True,
        min_new_voxel_frac=0.5,
        coverage_gain_depth_stride=4,
        depth_max=8.0,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _make_admitter(cfg, occupancy_fn):
    adm = KeyframeAdmitter(
        config=cfg,
        occupancy_fn=occupancy_fn,
        load_depth_m=_depth_const(2.0),
    )
    # Seed a baseline so pose deltas (and thus `moved`) are measured.
    adm.note_train_frame(_frame(0, 0.0))
    return adm


def test_novel_frame_admits():
    """All backprojected points fall in UNOCCUPIED voxels (new_voxel_frac=1.0
    >= 0.5) -> coverage gate passes -> render_failed fast-path admits the
    moved frame."""
    all_unoccupied = lambda pts, vox: np.zeros(len(pts), dtype=bool)
    adm = _make_admitter(_cfg(), all_unoccupied)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.20))
    assert decision.admitted
    assert decision.reason != "rej_coverage"


def test_redundant_frame_rejects():
    """All points fall in ALREADY-OCCUPIED voxels (new_voxel_frac=0.0 < 0.5)
    -> coverage gate rejects with reason 'rej_coverage'."""
    all_occupied = lambda pts, vox: np.ones(len(pts), dtype=bool)
    adm = _make_admitter(_cfg(), all_occupied)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.20))
    assert not decision.admitted
    assert decision.reason == "rej_coverage"
    assert adm._keyframes_rejected_coverage == 1


def test_gate_off_is_noop():
    """With coverage_gain_gate=False the occupancy_fn is never consulted; a
    fully-redundant frame is NOT rejected by coverage (it reaches the render
    gate and, with no render_fn, admits via the render_failed/moved path)."""
    called = {"n": 0}

    def _spy(pts, vox):
        called["n"] += 1
        return np.ones(len(pts), dtype=bool)

    adm = _make_admitter(_cfg(coverage_gain_gate=False), _spy)
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.20))
    assert called["n"] == 0           # occupancy never queried
    assert decision.reason != "rej_coverage"
    assert decision.admitted          # falls through to render_failed -> moved admit
