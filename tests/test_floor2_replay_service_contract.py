"""CPU contracts separating replay identity retention from payload callability."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import (
    BufferStage,
    ColdPixelHandle,
    FrameObservationDB,
    FrameRecord,
    ModalityEligibility,
)
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateConfig,
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)


def _record(frame_id: int) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=frame_id / 30.0,
        submap_id=0,
        buffer_stage=BufferStage.COLD_CACHE,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        source_identity=f"live:{frame_id}",
        source_ordinal=frame_id,
        historical_eligible=True,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
        ),
    )


def test_bounded_shortlist_does_not_bound_identity_catalog() -> None:
    index = HistoricalCandidateIndex(
        config=HistoricalCandidateConfig(
            shortlist_size=128,
            recent_lane_size=32,
            focus_lane_size=32,
            protected_lane_size=16,
            global_reservoir_size=64,
            global_floor=16,
        ),
        seed=17,
    )
    for frame_id in range(700):
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=frame_id % 7)
        )

    shortlist = index.shortlist(snapshot_id="floor2", epoch=24)
    shortlist_ids = {candidate.frame_id for candidate in shortlist}
    omitted = set(range(700)) - shortlist_ids

    assert index.history_size == 700
    assert len(shortlist) >= 16
    assert len(shortlist) <= 128
    assert omitted
    assert index.metadata_for(min(omitted)).frame_id == min(omitted)


def test_same_byte_budget_distinguishes_payload_expiry_from_retained_identity() -> None:
    capped = FrameObservationDB(
        device="cpu", max_cold_payload_frames=128, max_cold_payload_bytes=1_000_000
    )
    retain_all = FrameObservationDB(
        device="cpu", max_cold_payload_frames=None, max_cold_payload_bytes=1_000_000
    )
    for frame_id in range(129):
        for db in (capped, retain_all):
            db.add_frame(_record(frame_id))
            db.store_cold_pixels(
                frame_id,
                ColdPixelHandle(
                    source_identity=f"live:{frame_id}",
                    rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                    depth=np.ones((1, 1), dtype=np.float32),
                    intrinsics="cpu-test",
                ),
            )

    assert [item.frame_id for item in capped.historical_candidates()] == list(
        range(1, 129)
    )
    assert [item.frame_id for item in retain_all.historical_candidates()] == list(
        range(129)
    )
    assert capped.cold_payload_occupancy == 128
    assert capped.cold_payload_bytes <= 1_000_000
    assert retain_all.cold_payload_occupancy == 129
    assert [
        item.frame_id
        for item in capped.historical_candidates(materializable_only=False)
    ] == list(range(129))
    assert capped.get_metadata(0).source_identity == "live:0"
    assert capped.get_cold_pixels(0).payload_evicted is True
    assert capped.get_cold_pixels(0).can_materialize is False
    assert retain_all.get_cold_pixels(0).can_materialize is True


@pytest.mark.parametrize("byte_budget", [13])
def test_count_disabled_still_enforces_byte_bound(byte_budget: int) -> None:
    db = FrameObservationDB(
        device="cpu", max_cold_payload_frames=None, max_cold_payload_bytes=byte_budget
    )
    for frame_id in range(2):
        db.add_frame(_record(frame_id))
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=f"live:{frame_id}",
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                depth=np.ones((1, 1), dtype=np.float32),
                intrinsics="cpu-test",
            ),
        )

    assert db.cold_payload_occupancy == 1
    assert db.cold_payload_bytes <= byte_budget
    assert [item.frame_id for item in db.historical_candidates()] == [1]
    assert [
        item.frame_id for item in db.historical_candidates(materializable_only=False)
    ] == [0, 1]
    assert db.get_metadata(0).source_identity == "live:0"
    assert db.get_cold_pixels(0).payload_evicted is True
