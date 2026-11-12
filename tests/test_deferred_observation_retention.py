"""Deferred observation retention contracts for issue #1209."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import (
    BufferStage,
    ColdPixelHandle,
    FrameObservationDB,
    FrameRecord,
    ModalityEligibility,
)


def _deferred_record(frame_id: int, *, source_identity: str) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=float(frame_id) / 30.0,
        submap_id=0,
        buffer_stage=BufferStage.COLD_CACHE,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        source_identity=source_identity,
        source_ordinal=frame_id,
        historical_eligible=True,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=False,
        ),
    )


def test_reloadable_deferred_candidate_keeps_only_a_cheap_reload_handle() -> None:
    """A reload token must prevent dataset pixels consuming the live archive."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        max_cold_payload_bytes=1,
    )
    db.add_frame(_deferred_record(7, source_identity="tum:fr3:7"))
    db.store_cold_pixels(
        7,
        ColdPixelHandle(
            source_identity="tum:fr3:7",
            rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            depth=np.zeros((4, 6), dtype=np.float32),
            intrinsics="tum-fr3-calibration",
            reload_token=("rgb/7.png", "depth/7.png"),
        ),
    )

    handle = db.get_cold_pixels(7)
    assert handle is not None
    assert handle.rgb is None
    assert handle.depth is None
    assert handle.reload_token == ("rgb/7.png", "depth/7.png")
    assert db.cold_payload_occupancy == 0
    assert db.cold_payload_bytes == 0

    candidates = db.historical_candidates()
    assert len(candidates) == 1
    assert candidates[0].source_identity == "tum:fr3:7"
    assert candidates[0].reloadable is True
    assert candidates[0].cold_payload_resident is False
    assert candidates[0].can_materialize is True
    assert not hasattr(candidates[0], "rgb")


def test_live_payload_eviction_preserves_metadata_and_marks_pixels_unavailable() -> None:
    """Evicting a non-reloadable frame must not delete historical provenance."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=1,
        max_cold_payload_bytes=10_000,
    )
    for frame_id in (0, 1):
        db.add_frame(_deferred_record(frame_id, source_identity=f"live:{frame_id}"))
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=f"live:{frame_id}",
                rgb=np.zeros((4, 6, 3), dtype=np.uint8),
                depth=np.zeros((4, 6), dtype=np.float32),
                intrinsics="live-calibration",
            ),
        )

    evicted = db.get_cold_pixels(0)
    assert evicted is not None
    assert evicted.can_materialize is False
    assert evicted.payload_evicted is True
    assert db.get_metadata(0).source_identity == "live:0"
    assert db.get_frame(0).buffer_stage is BufferStage.ICECOLD_CACHE
    assert db.residency_snapshot(0).cold_pixels_available is False
    assert db.residency_snapshot(0).cold_payload_resident is False

    # The full historical/provenance domain remains independently readable,
    # while work selection defaults to frames that can actually be paged.
    assert [item.frame_id for item in db.historical_candidates()] == [1]
    assert [
        item.frame_id
        for item in db.historical_candidates(materializable_only=False)
    ] == [0, 1]


def test_capacity_preflight_preserves_existing_live_history_without_orphaning_new_source() -> None:
    """A new tokenless packet may not evict an already eligible live record."""
    db = FrameObservationDB(
        device="cpu", max_cold_payload_frames=1, max_cold_payload_bytes=10_000
    )
    db.add_frame(_deferred_record(0, source_identity="live:0"))
    existing = ColdPixelHandle(
        source_identity="live:0",
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        depth=np.zeros((4, 6), dtype=np.float32),
        intrinsics="live-calibration",
    )
    db.store_cold_pixels(0, existing)
    incoming = ColdPixelHandle(
        source_identity="live:1",
        rgb=np.ones((4, 6, 3), dtype=np.uint8),
        depth=np.ones((4, 6), dtype=np.float32),
        intrinsics="live-calibration",
    )

    assert db.can_retain_cold_handle(incoming) is False
    assert [record.frame_id for record in db.historical_candidates()] == [0]
    assert db.frame_id_for_source("live:1") is None
    assert db.get_cold_pixels(0).can_materialize is True
    assert db.cold_payload_occupancy == 1


def test_cold_payload_owner_enforces_byte_bound_independently_of_count() -> None:
    """A generous count bound must not bypass the configured byte budget."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=10,
        max_cold_payload_bytes=12,
    )
    for frame_id in (0, 1):
        db.add_frame(_deferred_record(frame_id, source_identity=f"live:{frame_id}"))
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=f"live:{frame_id}",
                rgb=np.zeros(8, dtype=np.uint8),
                depth=None,
                intrinsics="live-calibration",
            ),
        )

    assert db.cold_payload_occupancy == 1
    assert db.cold_payload_bytes == 8
    assert db.get_cold_pixels(0).can_materialize is False
    assert db.get_cold_pixels(1).can_materialize is True


def test_reloadable_history_remains_selectable_past_resident_cache_bounds() -> None:
    """A 512-style resident bound must never become an identity bound."""
    db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=4,
        max_cold_payload_bytes=64,
    )
    for frame_id in range(600):
        source_identity = f"bag:fr3:{frame_id}"
        db.add_frame(_deferred_record(frame_id, source_identity=source_identity))
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=source_identity,
                rgb=None,
                depth=None,
                intrinsics="tum-fr3-calibration",
                reload_token=f"bag-offset:{frame_id}",
            ),
        )

    candidates = db.historical_candidates(require_rgb=True)
    assert len(candidates) == 600
    assert candidates[0].source_identity == "bag:fr3:0"
    assert candidates[-1].source_identity == "bag:fr3:599"
    assert db.cold_payload_occupancy == 0
    assert db.cold_payload_bytes == 0


def test_historical_candidate_filters_use_modality_eligibility_not_pixels() -> None:
    """Replay selection must filter by declared semantics, not payload shape."""
    db = FrameObservationDB(device="cpu")
    rgb_only = _deferred_record(0, source_identity="bag:rgb-only")
    rgb_only.modality_eligibility = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=False,
        depth_insertion=False,
    )
    depth_refinement = _deferred_record(1, source_identity="bag:depth-refinement")
    depth_refinement.modality_eligibility = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=True,
    )
    for record in (rgb_only, depth_refinement):
        db.add_frame(record)
        db.store_cold_pixels(
            record.frame_id,
            ColdPixelHandle(
                source_identity=record.source_identity,
                rgb=None,
                depth=None,
                intrinsics="calibration",
                reload_token=f"offset:{record.frame_id}",
            ),
        )

    assert [
        item.frame_id for item in db.historical_candidates(require_depth=True)
    ] == [1]
    assert [
        item.frame_id
        for item in db.historical_candidates(require_depth_insertion=True)
    ] == [1]


def test_record_candidate_publishes_deferred_semantics_atomically() -> None:
    """The canonical record boundary must publish retention semantics with metadata."""
    db = FrameObservationDB(device="cpu")
    modalities = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=False,
    )

    record = db.record_candidate(
        SimpleNamespace(c2w=np.eye(4, dtype=np.float32)),
        SimpleNamespace(
            index=3,
            timestamp=0.1,
            mapping_frame=None,
            rgb_path="rgb/3.png",
            depth_path="depth/3.png",
        ),
        source_identity="bag:fr3:3",
        is_train=False,
        is_holdout=False,
        historical_eligible=True,
        modality_eligibility=modalities,
    )

    assert record.historical_eligible is True
    assert record.modality_eligibility == modalities
    assert [item.source_identity for item in db.historical_candidates()] == [
        "bag:fr3:3"
    ]


def test_rejected_replacement_keeps_the_prior_cold_owner() -> None:
    """A mismatched source handle must fail before changing retained pixels."""
    db = FrameObservationDB(device="cpu", max_cold_payload_frames=1)
    db.add_frame(_deferred_record(0, source_identity="live:0"))
    original = ColdPixelHandle(
        source_identity="live:0",
        rgb=np.zeros(8, dtype=np.uint8),
        depth=None,
        intrinsics="live-calibration",
    )
    db.store_cold_pixels(0, original)

    with pytest.raises(ValueError, match="source identity mismatch"):
        db.store_cold_pixels(
            0,
            ColdPixelHandle(
                source_identity="other-stream:0",
                rgb=np.ones(8, dtype=np.uint8),
                depth=None,
                intrinsics="wrong-calibration",
            ),
        )

    assert db.get_cold_pixels(0) is original
    assert db.cold_payload_occupancy == 1


def test_record_boundary_accepts_a_structural_modality_contract() -> None:
    """DB retention must not depend on the control contract's concrete class."""
    db = FrameObservationDB(device="cpu")
    record = db.record_candidate(
        SimpleNamespace(c2w=np.eye(4, dtype=np.float32)),
        SimpleNamespace(
            index=9,
            timestamp=0.3,
            mapping_frame=None,
            rgb_path="rgb/9.png",
            depth_path="depth/9.png",
        ),
        source_identity="bag:fr3:9",
        is_train=False,
        is_holdout=False,
        historical_eligible=True,
        modality_eligibility=SimpleNamespace(
            rgb_supervision=True,
            depth_supervision=False,
            depth_insertion=False,
        ),
    )

    assert record.modality_eligibility == ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=False,
        depth_insertion=False,
    )


def test_rejected_source_cannot_be_published_as_historical_candidate() -> None:
    db = FrameObservationDB(device="cpu")
    source = SourceIdentity("camera", 0, 17, 17_000_000)
    handle = ColdPixelHandle(
        source_identity=source.fingerprint(),
        rgb=None,
        depth=None,
        intrinsics="calibration",
        reload_token="offset:17",
    )

    with pytest.raises(ValueError, match="DEFER semantic provenance"):
        db.record_historical_candidate(
            source_identity=source,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=0.17,
            intrinsics="calibration",
            cold_pixels=handle,
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
            disposition_snapshot_id=7,
            semantic_action="reject",
        )

    assert len(db) == 0
    assert db.frame_id_for_source(source) is None


def test_deferred_history_requires_terminal_disposition_provenance() -> None:
    db = FrameObservationDB(device="cpu")
    source = SourceIdentity("camera", 0, 18, 18_000_000)
    handle = ColdPixelHandle(
        source_identity=source.fingerprint(),
        rgb=None,
        depth=None,
        intrinsics="calibration",
        reload_token="offset:18",
    )

    with pytest.raises(ValueError, match="disposition_snapshot_id"):
        db.record_historical_candidate(
            source_identity=source,
            pose_c2w=np.eye(4, dtype=np.float32),
            timestamp=0.18,
            intrinsics="calibration",
            cold_pixels=handle,
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
        )

    assert len(db) == 0
    assert db.frame_id_for_source(source) is None
