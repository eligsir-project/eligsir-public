from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts import (
    BufferRef,
    CameraIntrinsics,
    ModalityEligibility,
    Pose,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.runtime.prepared_source_producer import (
    PreparedSourceProduceResult,
    PreparedSourceProducer,
)
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner


def _packet(rgb: np.ndarray) -> SensorPacket:
    source = SourceIdentity("prepared-single-h2d", 0, 1, 1_000)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.full(rgb.shape[:2], 1000, dtype=np.uint16),
            encoding="16UC1",
            units="mm",
        ),
        intrinsics=CameraIntrinsics(
            fx=8.0,
            fy=9.0,
            cx=1.5,
            cy=1.0,
            width=rgb.shape[1],
            height=rgb.shape[0],
            depth_scale=1000.0,
        ),
        tracking=TrackingObservation(
            source=source,
            pose=Pose(np.eye(4, dtype=np.float32)),
            status=TrackingStatus.VALID,
            provider="test",
        ),
    )


def test_production_owner_copacks_pixels_pose_and_intrinsics_once():
    owner = DeviceTransferOwner("cpu", max_pinned_bytes=1 << 20)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
    )

    result = producer.submit(
        _packet(np.full((2, 3, 3), 128, dtype=np.uint8)),
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
        ),
    )

    assert result is PreparedSourceProduceResult.ENQUEUED
    assert owner.accounting().h2d_ops == 1
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert set(lease.payloads) >= {"rgb", "depth_meters", "c2w", "camera_intrinsics"}


def test_strict_prepared_rgb_rejects_noncanonical_source_dtype():
    owner = DeviceTransferOwner("cpu", max_pinned_bytes=1 << 20)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
    )

    with pytest.raises(ValueError, match="canonical uint8"):
        producer.submit(
            _packet(np.ones((2, 3, 3), dtype=np.float32)),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
        )
