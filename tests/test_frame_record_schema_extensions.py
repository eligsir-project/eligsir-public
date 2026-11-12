"""Confirms FrameRecord's buffer_stage/resolution_cache extensions (design
sections 6-7) are usable as designed, with no eviction/caching policy
attached (explicitly out of scope per this plan's non-goals)."""
from __future__ import annotations

import numpy as np
import torch

from eligsir.mapping.observation_db import BufferStage, FrameObservationDB, FrameRecord


def test_all_four_buffer_stages_are_distinct_and_assignable():
    stages = {
        BufferStage.MAPPING_KEYFRAME,
        BufferStage.HOLDOUT_TEST,
        BufferStage.EVICTED_REDUNDANT,
        BufferStage.GOLDEN,
    }
    assert len(stages) == 4
    rec = FrameRecord(
        frame_id=0, pose_c2w=np.eye(4, dtype=np.float32), timestamp=0.0,
        submap_id=0, buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
    )
    for stage in stages:
        rec.buffer_stage = stage
        assert rec.buffer_stage == stage


def test_golden_stage_is_independent_of_resolution_cache_contents():
    """A GOLDEN frame's metadata retention is independent of which
    resolution tiers happen to be cached for it -- confirms the two
    extensions (buffer_stage, resolution_cache) don't couple to each other,
    matching the design's explicit statement that pixel-cache eviction and
    FrameRecord metadata eviction are independent axes."""
    rec = FrameRecord(
        frame_id=0, pose_c2w=np.eye(4, dtype=np.float32), timestamp=0.0,
        submap_id=0, buffer_stage=BufferStage.GOLDEN,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
    )
    assert rec.resolution_cache == {}  # no tiers cached yet
    rec.resolution_cache["fast"] = "low_res_handle"
    assert rec.buffer_stage == BufferStage.GOLDEN  # unaffected by cache state
    del rec.resolution_cache["fast"]
    assert rec.buffer_stage == BufferStage.GOLDEN  # still unaffected


def test_db_never_auto_transitions_stage_or_evicts_frames():
    """No policy exists yet -- adding many frames never changes an
    existing frame's buffer_stage or removes it from the DB."""
    db = FrameObservationDB(device="cpu")
    for i in range(50):
        db.add_frame(FrameRecord(
            frame_id=i, pose_c2w=np.eye(4, dtype=np.float32), timestamp=float(i),
            submap_id=0, buffer_stage=BufferStage.MAPPING_KEYFRAME,
            visible_gaussian_uids=torch.tensor([i], dtype=torch.int64),
        ))
    assert len(db) == 50
    assert db.get_frame(0).buffer_stage == BufferStage.MAPPING_KEYFRAME
