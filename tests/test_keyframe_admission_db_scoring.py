"""Tests for database-driven keyframe admission (task 5)."""
from __future__ import annotations

import numpy as np
import torch

from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.mapping.scheduler.keyframes import KeyframeAdmitter, KeyframeAdmitterConfig


class _FakeFrame:
    def __init__(self, index: int, c2w: np.ndarray) -> None:
        self.index = index
        self.c2w = c2w.astype(np.float32)


class _FakeCam:
    def __init__(self, c2w: np.ndarray) -> None:
        self.c2w = c2w.astype(np.float32)
        R = c2w[:3, :3]
        self.R = R
        self.T = (-R.T @ c2w[:3, 3]).astype(np.float32)
        # `_coverage_new_voxel_frac` gates on `hasattr(cam, "world_view_transform")`
        # before even checking for `.c2w` (which it does use, since present) --
        # a dummy tensor here is enough to pass that guard. Not in the brief's
        # literal fixture; without it _coverage_new_voxel_frac always returns
        # None and the coverage/DB gate under test is never reached.
        self.world_view_transform = torch.eye(4)
        # `_coverage_new_voxel_frac` also needs fx/fy (or FoVx/FoVy) to
        # backproject depth -- FakeCam has neither in the brief's literal
        # fixture, which silently AttributeErrors into the function's
        # blanket except-return-None. Arbitrary plausible pinhole intrinsics.
        self.fx = 2.0
        self.fy = 2.0
        self.cx = 2.0
        self.cy = 2.0


class _FakeGaussianModel:
    def __init__(self, xyz: torch.Tensor, uids: torch.Tensor) -> None:
        self._xyz = xyz
        self.gaussian_uid = uids

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz


def _stub_render(cam, render_depth=False):
    return {"alpha": torch.ones(1, 4, 4)}  # full overlap if the render gate is reached


def test_diverse_revisit_admitted_despite_low_spatial_novelty():
    """A frame that mostly re-sees already-mapped Gaussians (low new-voxel
    fraction -- would be rejected by the OLD coverage gate) is admitted
    anyway because the DB finds it's a high-overlap revisit from a
    different viewing angle."""
    db = FrameObservationDB(device="cpu")
    # max_gap raised well above the candidate's frame-index gap (50) so the
    # forced-admit fast path (gap >= max_gap -> unconditional "max_gap"
    # admission, checked before the 1c coverage/DB gate) does not mask the
    # DB-diversity logic under test. Not in the brief's literal test code,
    # but required for the test to exercise gate 1c at all with the brief's
    # chosen frame indices (0 and 50) against the KeyframeAdmitterConfig
    # default max_gap=10.
    cfg = KeyframeAdmitterConfig(coverage_gain_gate=True, min_new_voxel_frac=0.5, max_gap=1000, coverage_gain_depth_stride=1)
    model = _FakeGaussianModel(
        torch.zeros((0, 3)), torch.zeros(0, dtype=torch.int64)
    )
    admitter = KeyframeAdmitter(
        config=cfg, render_fn=_stub_render, gaussians=model,
        # All points report already-occupied (True) so the spatial coverage
        # gate sees near-zero new-voxel fraction -- i.e. low spatial novelty,
        # per this test's docstring. The brief's literal lambda used
        # np.zeros (all-novel), which would never reach the coverage
        # rejection path at all; corrected to np.ones here.
        occupancy_fn=lambda pts, vs, check_neighbors=True: np.ones(len(pts), dtype=bool),
        load_depth_m=lambda frame: np.full((4, 4), 1.0, dtype=np.float32),
        observation_db=db,
    )
    # Seed the DB with a frame that saw the same content from a very
    # different angle (180 degrees rotated about Y).
    old_c2w = np.eye(4, dtype=np.float32)
    from eligsir.mapping.observation_db import BufferStage, FrameRecord
    db.add_frame(FrameRecord(
        frame_id=0, pose_c2w=old_c2w, timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        resolution_cache={},
    ))
    admitter.note_train_frame(_FakeFrame(0, old_c2w))

    rot180_y = np.array([
        [-1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, -1, 0.2],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    candidate_frame = _FakeFrame(50, rot180_y)
    candidate_cam = _FakeCam(rot180_y)
    candidate_cam._observed_gaussian_uids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    decision, _ = admitter.admit(candidate_cam, candidate_frame)
    assert decision.admitted
    assert decision.reason == "db_diverse_revisit"


def test_redundant_same_angle_revisit_still_rejected():
    """High overlap with a past frame from basically the SAME angle carries
    no new information -- must still be rejected, not admitted just because
    overlap is high."""
    db = FrameObservationDB(device="cpu")
    # See the sibling test above for why max_gap is raised above the frame-index
    # gap (50) used by the brief's test data.
    cfg = KeyframeAdmitterConfig(coverage_gain_gate=True, min_new_voxel_frac=0.5, max_gap=1000,
                                  coverage_gain_depth_stride=1, motion_pregate=False)
    model = _FakeGaussianModel(torch.zeros((0, 3)), torch.zeros(0, dtype=torch.int64))
    admitter = KeyframeAdmitter(
        config=cfg, render_fn=_stub_render, gaussians=model,
        # All points report already-occupied (True) so the spatial coverage
        # gate sees near-zero new-voxel fraction -- i.e. low spatial novelty,
        # per this test's docstring. The brief's literal lambda used
        # np.zeros (all-novel), which would never reach the coverage
        # rejection path at all; corrected to np.ones here.
        occupancy_fn=lambda pts, vs, check_neighbors=True: np.ones(len(pts), dtype=bool),
        load_depth_m=lambda frame: np.full((4, 4), 1.0, dtype=np.float32),
        observation_db=db,
    )
    old_c2w = np.eye(4, dtype=np.float32)
    from eligsir.mapping.observation_db import BufferStage, FrameRecord
    db.add_frame(FrameRecord(
        frame_id=0, pose_c2w=old_c2w, timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        resolution_cache={},
    ))
    admitter.note_train_frame(_FakeFrame(0, old_c2w))

    nearly_same_c2w = old_c2w.copy()
    nearly_same_c2w[0, 3] = 0.01  # tiny translation, same orientation
    candidate_frame = _FakeFrame(50, nearly_same_c2w)
    candidate_cam = _FakeCam(nearly_same_c2w)
    candidate_cam._observed_gaussian_uids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    decision, _ = admitter.admit(candidate_cam, candidate_frame)
    assert not decision.admitted
    assert decision.reason == "rej_redundant_angle"


def test_db_diverse_revisit_uses_min_overlap_not_min_new_voxel_frac():
    """Regression test for #937: the db_diverse_revisit branch's inner
    admission test must compare the overlap fraction against
    ``cfg.min_overlap`` (the field that means "overlap fraction" everywhere
    else in this file), not ``cfg.min_new_voxel_frac`` (a *novelty*
    threshold already consumed by the outer coverage-gain-gate test).

    Chosen so the two thresholds straddle the measured overlap_frac
    (0.1): min_new_voxel_frac=0.05 < overlap_frac=0.1 < min_overlap=0.25.
    Under the pre-fix code (comparing against min_new_voxel_frac) this
    candidate is wrongly ADMITTED as "db_diverse_revisit"; under the fix
    (comparing against min_overlap) it is correctly REJECTED as
    "rej_redundant_angle" -- the bug and the fix produce different
    admit/reject decisions on the identical input, so this test only
    passes for the correct comparison.
    """
    db = FrameObservationDB(device="cpu")
    cfg = KeyframeAdmitterConfig(
        coverage_gain_gate=True,
        min_new_voxel_frac=0.05,   # default; novelty threshold for the OUTER test
        min_overlap=0.25,          # default; correct threshold for the INNER test
        max_gap=1000,
        coverage_gain_depth_stride=1,
        motion_pregate=False,
    )
    model = _FakeGaussianModel(torch.zeros((0, 3)), torch.zeros(0, dtype=torch.int64))
    admitter = KeyframeAdmitter(
        config=cfg, render_fn=_stub_render, gaussians=model,
        # All points already-occupied -> near-zero new-voxel fraction, so
        # the outer `_new_voxel_frac < min_new_voxel_frac` test fires and
        # the db_diverse_revisit branch under test is reached.
        occupancy_fn=lambda pts, vs, check_neighbors=True: np.ones(len(pts), dtype=bool),
        load_depth_m=lambda frame: np.full((4, 4), 1.0, dtype=np.float32),
        observation_db=db,
    )
    old_c2w = np.eye(4, dtype=np.float32)
    from eligsir.mapping.observation_db import BufferStage, FrameRecord
    # Only uid 0 was seen by the past frame, so the CSR is indexed to a
    # single column (uid 0) -- see overlap_query()'s `in_range` filter.
    db.add_frame(FrameRecord(
        frame_id=0, pose_c2w=old_c2w, timestamp=0.0, submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.tensor([0], dtype=torch.int64),
        resolution_cache={},
    ))
    admitter.note_train_frame(_FakeFrame(0, old_c2w))

    # 180-degree rotation about Y -> angle_deg well above the 20-degree
    # diversity threshold, so the angle half of the inner test is never
    # the discriminator here.
    rot180_y = np.array([
        [-1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, -1, 0.2],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    candidate_frame = _FakeFrame(50, rot180_y)
    candidate_cam = _FakeCam(rot180_y)
    # 10 candidate uids, only uid 0 overlaps the past frame's [0] ->
    # overlap_frac = 1 / 10 = 0.1.
    candidate_cam._observed_gaussian_uids = torch.arange(10, dtype=torch.int64)

    decision, _ = admitter.admit(candidate_cam, candidate_frame)
    assert decision.score == 0.1
    assert not decision.admitted
    assert decision.reason == "rej_redundant_angle"
