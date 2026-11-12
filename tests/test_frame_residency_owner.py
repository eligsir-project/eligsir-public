"""Issue #1199 frame-history residency ownership contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import (
    BufferStage,
    CacheHandle,
    ColdPixelHandle,
    FrameObservationDB,
    FrameRecord,
)


def _record(frame_id: int, cache=None) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=float(frame_id) / 30.0,
        submap_id=7,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.tensor([frame_id, frame_id + 1]),
        resolution_cache={} if cache is None else cache,
    )


def _cache(frame_id: int) -> dict[str, CacheHandle]:
    return {
        "high": CacheHandle(
            rgb=torch.full((4, 6, 3), frame_id, dtype=torch.uint8),
            depth=torch.full((4, 6), float(frame_id), dtype=torch.float32),
            intrinsics=(6, 4, 4.0, 4.0, 2.5, 1.5),
        )
    }


def test_cold_pixel_handle_rejects_device_payloads() -> None:
    """A missing CPU-domain check must not let history retain device pixels."""

    class _DevicePayload:
        device = torch.device("cuda")

    with pytest.raises(ValueError, match="CPU/cold"):
        ColdPixelHandle(
            source_identity="camera:42",
            rgb=_DevicePayload(),
            depth=None,
            intrinsics="calibration-v1",
        )


def test_metadata_snapshot_is_cpu_immutable_after_hot_eviction() -> None:
    """Dropping HOT pixels must preserve immutable CPU metadata queries."""
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=1)
    db.add_frame(_record(0, _cache(0)))
    db.store_cold_pixels(
        0,
        ColdPixelHandle(
            source_identity="camera:0",
            rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            depth=np.zeros((4, 6), dtype=np.float32),
            intrinsics="calibration-v1",
        ),
    )
    db.add_frame(_record(1, _cache(1)))

    metadata = db.get_metadata(0)
    assert metadata is not None
    assert metadata.frame_id == 0
    assert metadata.timestamp == 0.0
    assert metadata.submap_id == 7
    assert metadata.pose_c2w == tuple(tuple(float(v) for v in row) for row in np.eye(4))
    assert db.get_frame(0).resolution_cache == {}
    assert db.get_cold_pixels(0).source_identity == "camera:0"
    with pytest.raises(FrozenInstanceError):
        metadata.timestamp = 9.0


def test_active_supervision_residency_plateaus_with_long_history() -> None:
    """Removing the HOT bound must fail even when metadata history stays complete."""
    hot_bound = 3
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=hot_bound)

    for frame_id in range(200):
        db.add_frame(_record(frame_id))
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=f"camera:{frame_id}",
                rgb=None,
                depth=None,
                intrinsics="calibration-v1",
                reload_token=f"bag-offset:{frame_id}",
            ),
        )
        assert db.activate_supervision(frame_id, _cache(frame_id)) is True
        assert db.active_supervision_occupancy <= hot_bound

    assert len(db) == 200
    assert db.cold_pixel_handle_count == 200
    assert db.active_supervision_frame_ids == (197, 198, 199)
    assert db.get_metadata(0).frame_id == 0
    frame_ids, counts = db.overlap_query(torch.tensor([0], dtype=torch.int64))
    assert frame_ids.tolist() == [0]
    assert counts.tolist() == [1.0]


def test_explicit_demotion_keeps_cold_pixels_and_reports_residency() -> None:
    """HOT-to-cold transition must clear only device working-set ownership."""
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=2)
    db.add_frame(_record(5))
    cold = ColdPixelHandle(
        source_identity="camera:5",
        rgb=torch.zeros((4, 6, 3), dtype=torch.uint8),
        depth=torch.zeros((4, 6), dtype=torch.float32),
        intrinsics="calibration-v1",
    )
    db.store_cold_pixels(5, cold)
    assert db.activate_supervision(5, _cache(5)) is True

    hot = db.residency_snapshot(5)
    assert hot.cold_pixels_available is True
    assert hot.active_tiers == ("high",)
    assert hot.visibility_device == "cpu"
    assert db.demote_supervision(5) is True

    cold_only = db.residency_snapshot(5)
    assert cold_only.cold_pixels_available is True
    assert cold_only.active_tiers == ()
    assert db.get_frame(5).resolution_cache_evicted is True


def test_active_cache_preserves_tier_intrinsics_without_reconstruction() -> None:
    """A wrong tier association must not silently substitute native geometry."""
    low_intrinsics = (3, 2, 2.0, 2.0, 1.0, 0.5)
    cache = {
        "low": CacheHandle(
            rgb=torch.zeros((2, 3, 3), dtype=torch.uint8),
            depth=torch.zeros((2, 3), dtype=torch.float32),
            intrinsics=low_intrinsics,
        )
    }
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=1)
    db.add_frame(_record(9))
    assert db.activate_supervision(9, cache) is True

    assert db.get_frame(9).resolution_cache["low"].intrinsics == low_intrinsics
    assert db.residency_snapshot(9).active_tiers == ("low",)


def test_activate_supervision_rejects_cache_from_wrong_memory_domain() -> None:
    """Publishing CPU tensors as a CUDA HOT set must fail before residency lies."""
    db = FrameObservationDB(device="cuda", max_active_supervision_frames=1)
    record = _record(3)
    record.visible_gaussian_uids = torch.zeros(0, dtype=torch.int64)
    db.add_frame(record)

    with pytest.raises(ValueError, match="cache tensor device"):
        db.activate_supervision(3, _cache(3))


def test_record_candidate_stores_metadata_and_cold_mapping_frame_without_hot_cache() -> None:
    """Canonical record_fn must not upload/build tiers while indexing a frame."""
    source = SimpleNamespace(
        source_ordinal=12,
        timestamp_seconds=4.25,
        fingerprint=lambda: "source-fingerprint-12",
    )
    mapping_frame = SimpleNamespace(
        rgb=SimpleNamespace(data=np.zeros((4, 6, 3), dtype=np.uint8)),
        depth=SimpleNamespace(data=np.zeros((4, 6), dtype=np.float32)),
        intrinsics="calibration-v2",
        device="cpu",
    )
    frame = SimpleNamespace(
        index=999,
        timestamp=99.0,
        mapping_frame=mapping_frame,
        rgb_path="",
        depth_path=None,
    )
    cam = SimpleNamespace(c2w=np.eye(4, dtype=np.float32))
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=2)

    record = db.record_candidate(
        cam,
        frame,
        source_identity=source,
        is_train=True,
        is_holdout=False,
        visible_uids=torch.tensor([4, 8], dtype=torch.int64),
    )

    assert record.frame_id == 12
    assert record.timestamp == 4.25
    assert record.buffer_stage is BufferStage.MAPPING_KEYFRAME
    assert record.resolution_cache == {}
    assert db.active_supervision_occupancy == 0
    assert db.get_cold_pixels(12).source_identity == "source-fingerprint-12"
    assert db.get_cold_pixels(12).rgb is mapping_frame.rgb.data
    assert db.get_metadata(12).pose_c2w[0] == (1.0, 0.0, 0.0, 0.0)


def test_record_candidate_retains_reload_handle_for_rejected_frame() -> None:
    """A cold record without resident arrays must remain deterministically pageable."""
    frame = SimpleNamespace(
        index=7,
        timestamp=1.5,
        mapping_frame=None,
        rgb_path="rgb/000007.png",
        depth_path="depth/000007.png",
    )
    cam = SimpleNamespace(c2w=np.eye(4, dtype=np.float32))
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=2)

    record = db.record_candidate(
        cam,
        frame,
        source_identity="bag:kitchen:7",
        is_train=False,
        is_holdout=False,
    )

    assert record.buffer_stage is BufferStage.EVICTED_REDUNDANT
    assert record.visible_gaussian_uids.numel() == 0
    cold = db.get_cold_pixels(7)
    assert cold.source_identity == "bag:kitchen:7"
    assert cold.rgb is None
    assert cold.reload_token == ("rgb/000007.png", "depth/000007.png")


def test_record_candidate_rejects_device_mapping_frame_atomically() -> None:
    """A bad live payload must not leave metadata without its promised cold owner."""
    frame = SimpleNamespace(
        index=4,
        timestamp=1.0,
        mapping_frame=SimpleNamespace(device="cuda"),
        rgb_path="",
        depth_path=None,
    )
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=2)

    with pytest.raises(ValueError, match="CPU-owned"):
        db.record_candidate(
            SimpleNamespace(c2w=np.eye(4, dtype=np.float32)),
            frame,
            source_identity="camera:4",
            is_train=True,
            is_holdout=False,
        )

    assert len(db) == 0


def test_same_source_ordinal_across_epochs_keeps_distinct_metadata_and_visibility() -> None:
    """Reset ordinals must not overwrite the earlier epoch's DB row."""
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=2)
    frame = SimpleNamespace(
        index=7,
        timestamp=0.0,
        mapping_frame=None,
        rgb_path="rgb/000007.png",
        depth_path="depth/000007.png",
    )
    cam = SimpleNamespace(c2w=np.eye(4, dtype=np.float32))
    first_source = SourceIdentity(
        source_id="orbbec/front",
        stream_epoch=0,
        source_ordinal=7,
        timestamp_ns=100,
    )
    restarted_source = SourceIdentity(
        source_id="realsense/replay",
        stream_epoch=1,
        source_ordinal=7,
        timestamp_ns=200,
    )

    first = db.record_candidate(
        cam,
        frame,
        source_identity=first_source,
        is_train=True,
        is_holdout=False,
        visible_uids=torch.tensor([41], dtype=torch.int64),
    )
    restarted = db.record_candidate(
        cam,
        frame,
        source_identity=restarted_source,
        is_train=True,
        is_holdout=False,
        visible_uids=torch.tensor([82], dtype=torch.int64),
    )

    assert first.frame_id == 7
    assert restarted.frame_id == 8
    assert len(db) == 2
    assert db.frame_id_for_source(first_source) == 7
    assert db.frame_id_for_source(restarted_source) == 8
    assert db.get_metadata(7).source_ordinal == 7
    assert db.get_metadata(8).source_ordinal == 7
    assert db.get_metadata(7).source_identity == first_source.fingerprint()
    assert db.get_metadata(8).source_identity == restarted_source.fingerprint()
    first_rows, _ = db.overlap_query(torch.tensor([41], dtype=torch.int64))
    restarted_rows, _ = db.overlap_query(torch.tensor([82], dtype=torch.int64))
    assert first_rows.tolist() == [7]
    assert restarted_rows.tolist() == [8]
