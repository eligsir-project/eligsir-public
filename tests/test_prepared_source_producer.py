from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from eligsir.contracts import (
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    MemoryDomain,
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
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import PinnedFramePool


class Gate:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class FakeTransfer:
    device = torch.device("cpu")

    def __init__(self, event=None) -> None:
        self.event = event or Gate(True)
        self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.d2h_calls = 0

    def upload_packed(self, arrays, *, purpose: str):
        assert purpose == "prepared_source"
        rgb, depth = tuple(arrays)
        self.calls.append((np.asarray(rgb), None if depth is None else np.asarray(depth)))
        return (
            (torch.from_numpy(np.asarray(rgb)), None if depth is None else torch.from_numpy(np.asarray(depth))),
            self.event,
        )


@dataclass
class FakeSlot:
    generation: int = 0

    def stage(self, rgb, depth):
        self.generation += 1
        return torch.from_numpy(np.ascontiguousarray(rgb)), (
            None if depth is None else torch.from_numpy(np.ascontiguousarray(depth))
        )


class FakePool:
    def __init__(self, *, exhausted: bool = False) -> None:
        self.exhausted = exhausted
        self.slot = FakeSlot()
        self.release_events = []
        self.release_now_events = []
        self.pending_event = None
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        if self.exhausted:
            raise RuntimeError("pool exhausted")
        if self.pending_event is not None:
            if not self.pending_event.query():
                raise RuntimeError("pool slot pending")
            self.pending_event = None
        return self.slot

    def release_after(self, slot, event):
        self.release_events.append((slot, event))
        self.pending_event = event

    def release_now(self, slot):
        self.release_now_events.append(slot)
        self.pending_event = None


def packet(ordinal: int = 0) -> SensorPacket:
    source = SourceIdentity("camera", 0, ordinal, 1_000_000_000 + ordinal)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"
        ),
        depth=BufferRef.from_numpy(
            np.full((2, 3), 1000, dtype=np.uint16), encoding="16UC1", units="mm"
        ),
        intrinsics=CameraIntrinsics(
            fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
        ),
        tracking=TrackingObservation(
            source=source,
            pose=Pose.identity(),
            status=TrackingStatus.VALID,
            provider="producer-test",
        ),
    )


def test_producer_submits_one_h2d_converts_on_device_and_admits_ready_lease():
    ready_owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    transfer = FakeTransfer()
    pool = FakePool()
    producer = PreparedSourceProducer(
        ready_owner,
        transfer_owner=transfer,
        frame_pool=pool,
        producer_owner="producer",
    )

    result = producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        lane=ReadySourceLane.FRESH,
        generation=3,
    )

    assert result is PreparedSourceProduceResult.ENQUEUED
    assert len(transfer.calls) == 1
    assert transfer.d2h_calls == 0
    assert len(pool.release_events) == 1
    consumer = ready_owner.claim_consumer("gpu")
    lease = consumer.poll()
    assert lease is not None
    assert lease.source_identity == packet().source
    assert lease.generation == 3
    assert lease.lifetime_token.generation == 3
    assert lease.lifetime_token.slot_generation == 1
    assert lease.lifetime_token.slot is pool.slot
    assert lease.borrowed_device_payloads["rgb"].data.dtype is torch.float32
    assert torch.allclose(lease.borrowed_device_payloads["rgb"].data, torch.full((2, 3, 3), 128 / 255.0))
    assert torch.allclose(lease.borrowed_device_payloads["depth_meters"].data, torch.ones((2, 3)))


def test_deferred_ready_event_is_not_admitted_until_completion():
    gate = Gate(False)
    ready_owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready_owner,
        transfer_owner=FakeTransfer(gate),
        frame_pool=FakePool(),
        producer_owner="producer",
    )

    assert producer.submit(packet(), modality_eligibility=ModalityEligibility(rgb_supervision=True)) is PreparedSourceProduceResult.ENQUEUED
    consumer = ready_owner.claim_consumer("gpu")
    assert consumer.poll() is None
    gate.ready = True
    assert consumer.poll() is not None


def test_pool_exhaustion_is_a_bounded_nonblocking_defer():
    producer = PreparedSourceProducer(
        ReadySourceOwner(max_fresh=1, max_replay=1),
        transfer_owner=FakeTransfer(),
        frame_pool=FakePool(exhausted=True),
        producer_owner="producer",
    )

    assert producer.submit(
        packet(), modality_eligibility=ModalityEligibility(rgb_supervision=True)
    ) is PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED


def test_ready_owner_drop_is_attributed_without_consumer_wait():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=FakeTransfer(),
        frame_pool=FakePool(),
        producer_owner="producer",
    )
    modality = ModalityEligibility(rgb_supervision=True)

    assert producer.submit(packet(0), modality_eligibility=modality) is PreparedSourceProduceResult.ENQUEUED
    assert producer.submit(packet(1), modality_eligibility=modality) is PreparedSourceProduceResult.DROPPED_READY_OWNER
    assert owner.full_drop_count == 1


def test_ready_full_durable_source_stays_cold_without_h2d_or_pinned_staging():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    transfer = FakeTransfer()
    pool = FakePool()
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=transfer,
        frame_pool=pool,
        producer_owner="producer",
    )
    modality = ModalityEligibility(rgb_supervision=True)

    assert producer.submit(packet(0), modality_eligibility=modality) is PreparedSourceProduceResult.ENQUEUED
    assert producer.submit(
        packet(1), modality_eligibility=modality, durable_retained=True
    ) is PreparedSourceProduceResult.RETAINED_NOT_RESIDENT

    assert len(transfer.calls) == 1
    assert pool.acquire_count == 1
    assert owner.counters.full_not_resident_count == 1


def test_preparation_failure_aborts_reservation_for_a_later_source():
    gate = Gate(False)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    pool = FakePool()
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=FakeTransfer(gate),
        frame_pool=pool,
        producer_owner="producer",
    )
    invalid_depth = SensorPacket(
        source=packet(0).source,
        rgb=packet(0).rgb,
        depth=packet(0).depth,
        intrinsics=CameraIntrinsics(
            fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=0.0
        ),
        tracking=packet(0).tracking,
    )
    with pytest.raises(ValueError, match="depth_scale"):
        producer.submit(invalid_depth, modality_eligibility=ModalityEligibility(rgb_supervision=True))

    gate.ready = True
    assert producer.submit(
        packet(1), modality_eligibility=ModalityEligibility(rgb_supervision=True)
    ) is PreparedSourceProduceResult.ENQUEUED
    assert owner.counters.fresh_pending == 1


def test_invalid_source_payload_fails_before_transfer():
    source = packet()
    invalid = SensorPacket(
        source=source.source,
        rgb=BufferRef(
            data="not-an-array",
            spec=BufferSpec((1,), "uint8", MemoryDomain.HOST, "rgb8"),
            ownership=source.rgb.ownership,
        ),
        depth=source.depth,
        intrinsics=source.intrinsics,
    )
    transfer = FakeTransfer()
    producer = PreparedSourceProducer(
        ReadySourceOwner(max_fresh=1, max_replay=1),
        transfer_owner=transfer,
        frame_pool=FakePool(),
        producer_owner="producer",
    )

    with pytest.raises(TypeError, match="RGB"):
        producer.submit(invalid, modality_eligibility=ModalityEligibility(rgb_supervision=True))
    assert transfer.calls == []


def test_pinned_frame_pool_waits_for_h2d_completion_before_reuse():
    pool = PinnedFramePool(slots=1)
    slot = pool.acquire()
    gate = Gate(False)
    pool.release_after(slot, gate)
    assert pool.pending_count == 1

    with pytest.raises(RuntimeError, match="exhausted"):
        pool.acquire()

    gate.ready = True
    assert pool.pending_count == 0
    assert pool.acquire() is slot


def test_post_h2d_failure_keeps_frame_slot_pending_until_transfer_completes():
    gate = Gate(False)
    pool = FakePool()
    transfer = FakeTransfer(gate)
    producer = PreparedSourceProducer(
        ReadySourceOwner(max_fresh=1, max_replay=1),
        transfer_owner=transfer,
        frame_pool=pool,
        producer_owner="producer",
    )
    invalid_depth = SensorPacket(
        source=packet().source,
        rgb=packet().rgb,
        depth=packet().depth,
        intrinsics=CameraIntrinsics(
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=1.0,
            width=3,
            height=2,
            depth_scale=0.0,
        ),
        tracking=packet().tracking,
    )

    with pytest.raises(ValueError, match="depth_scale"):
        producer.submit(
            invalid_depth,
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            ),
        )

    assert len(transfer.calls) == 1
    assert pool.release_now_events == []
    assert len(pool.release_events) == 1
    assert pool.pending_event is gate
    with pytest.raises(RuntimeError, match="pending"):
        pool.acquire()
    gate.ready = True
    assert pool.acquire() is pool.slot
