"""Unit tests for the parallax-aware covisibility admission gate (5A).

Golden-rule admission: a pure forward walk produces large pose deltas
(passes the pose gate) and steady coverage, but re-sees the same surface
points from nearly the same rays — near-zero parallax. The gate rejects
candidates with high covisibility AND low median parallax; lateral motion
(high parallax) or genuinely novel frames (low covisibility) still admit.

CPU-only, reusing the fake camera/occupancy stubs of the coverage-gate
tests. The camera looks down +z at a constant-depth plane 2 m away.
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


def _frame(index: int, x: float = 0.0, z: float = 0.0):
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = x
    c2w[2, 3] = z
    return FrameInfo(c2w=c2w, index=index)


def _cfg(**kw):
    base = dict(
        min_translation_m=0.05,
        min_rotation_deg=1e9,
        max_gap=10_000,
        coverage_gain_gate=False,
        parallax_gate=True,
        parallax_min_deg=3.0,
        parallax_covis_max=0.6,
        coverage_gain_depth_stride=4,
        depth_max=8.0,
    )
    base.update(kw)
    return KeyframeAdmitterConfig(**base)


def _admitter(cfg, occupancy_fn):
    arr = np.full((16, 16), 2.0, dtype=np.float32)
    adm = KeyframeAdmitter(
        config=cfg,
        occupancy_fn=occupancy_fn,
        load_depth_m=lambda frame: arr,
    )
    adm.note_train_frame(_frame(0, 0.0))
    return adm


_ALL_OCCUPIED = lambda pts, vox: np.ones(len(pts), dtype=bool)
_ALL_NEW = lambda pts, vox: np.zeros(len(pts), dtype=bool)


def test_forward_walk_low_parallax_rejects():
    """Forward step onto fully-known geometry: covis=1.0 >= 0.6 and median
    parallax ~0 deg (rays from (0,0,0.2) vs (0,0,0) to points ~2 m ahead
    differ by well under 3 deg on-axis) -> rej_parallax.

    NOTE: the fake cam's c2w is identity (the gate backprojects with the CAM
    transform), so the candidate's camera center is the origin while the
    baseline keyframe sits at the FRAME pose. We move the baseline, not the
    candidate, to model the forward step: frame(1) at z=-0.2 means the last
    keyframe was 0.2 m behind the candidate — a pure forward walk."""
    adm = _admitter(_cfg(), _ALL_OCCUPIED)
    adm.note_train_frame(_frame(0, 0.0, z=-0.2))
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.0, z=0.0))
    # pose delta = 0.2 m -> moved; covis 1.0; parallax tiny -> reject
    assert not decision.admitted
    assert decision.reason == "rej_parallax"
    assert adm._keyframes_rejected_parallax == 1


def test_lateral_motion_high_parallax_admits():
    """Same covisibility but the last keyframe is 1.5 m to the SIDE of the
    candidate: median parallax over points 2 m ahead is ~tens of degrees ->
    gate passes despite covis=1.0."""
    adm = _admitter(_cfg(), _ALL_OCCUPIED)
    adm.note_train_frame(_frame(0, x=1.5, z=0.0))
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.0, 0.0))
    assert decision.reason != "rej_parallax"
    assert decision.admitted


def test_novel_geometry_admits_regardless_of_parallax():
    """covis below threshold (all-new voxels) -> the parallax term is never
    consulted; the frame admits."""
    adm = _admitter(_cfg(), _ALL_NEW)
    adm.note_train_frame(_frame(0, 0.0, z=-0.2))
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.0, 0.0))
    assert decision.reason != "rej_parallax"
    assert decision.admitted


def test_gate_off_is_noop():
    adm = _admitter(_cfg(parallax_gate=False), _ALL_OCCUPIED)
    adm.note_train_frame(_frame(0, 0.0, z=-0.2))
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(1, 0.0, 0.0))
    assert decision.reason != "rej_parallax"


def test_forced_frame_never_parallax_rejected():
    """max_gap forcing bypasses the parallax gate (starvation backstop)."""
    adm = _admitter(_cfg(max_gap=1), _ALL_OCCUPIED)
    adm.note_train_frame(_frame(0, 0.0, z=-0.2))
    decision, _ = adm.admit(cam=_FakeCam(), frame=_frame(5, 0.0, 0.0))
    assert decision.reason != "rej_parallax"
