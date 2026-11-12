from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
from eligsir.contracts.control import PreparedTierPolicy, ResolutionTier
from eligsir.mapping.resolution_materialization import materialize_resolution_tensors
from eligsir.runtime.prepared_source_producer import (
    PreparedSourceProduceResult,
    PreparedSourceProducer,
    _same_device,
)
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.ready_camera_binder import bind_ready_camera
from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import TransferEvent


class Gate:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


def test_same_device_accepts_default_cuda_and_current_cuda_index():
    assert _same_device(torch.device("cuda"), torch.device("cuda:0"))
    assert not _same_device(torch.device("cuda:1"), torch.device("cuda:0"))


class FakePrepStream:
    """CPU fake that records stream ordering without requiring CUDA."""

    device = torch.device("cpu")

    def __init__(self, *, transfer_gate: Gate | None = None, prep_gate: Gate | None = None):
        self.transfer_gate = transfer_gate or Gate(True)
        self.prep_gate = prep_gate or Gate(True)
        self.operations: list[str] = []
        self.handoffs = 0
        self.declared_tiers: frozenset[ResolutionTier] | None = None

    def prepare_source(self, arrays, *, depth_units, depth_scale):
        self.operations.append("h2d")
        rgb, depth = arrays
        rgb_raw = torch.from_numpy(np.asarray(rgb))
        depth_raw = None if depth is None else torch.from_numpy(np.asarray(depth))
        self.operations.append("rgb_convert")
        rgb_device = rgb_raw.to(dtype=torch.float32).div(255.0)
        depth_device = None
        if depth_raw is not None:
            self.operations.append("depth_convert")
            depth_device = depth_raw.to(dtype=torch.float32).div(float(depth_scale))
        self.operations.append("prep_event")
        return (rgb_device, depth_device), TransferEvent(self.transfer_gate), TransferEvent(self.prep_gate)

    def handoff_to_compute(self, *events):
        self.handoffs += len(events)
        self.operations.append("compute_handoff")

    def prepare_ready_calibration(self, *, pose, camera_intrinsics):
        self.operations.append("calibration")
        return (
            (
                torch.tensor(pose, dtype=torch.float32),
                torch.tensor(camera_intrinsics, dtype=torch.float32),
            ),
            TransferEvent(self.prep_gate),
        )

    def prepare_source_tiers(
        self, rgb, depth, camera_intrinsics, *, resampling_mode, declared_tiers,
        depth_downsample="nearest",
    ):
        self.operations.append("tiers")
        self.declared_tiers = frozenset(declared_tiers)
        return (
            {
                tier.value: materialize_resolution_tensors(
                    rgb,
                    depth,
                    camera_intrinsics,
                    tier,
                    resampling_mode=resampling_mode,
                    depth_downsample=depth_downsample,
                )
                for tier in declared_tiers
            },
            TransferEvent(self.prep_gate),
        )


class FakeTransferOwner:
    device = torch.device("cpu")

    def __init__(self, stream: FakePrepStream):
        self.stream = stream
        self.calls = 0
        self.declared_tiers: frozenset[ResolutionTier] | None = None

    def prepare_source(self, arrays, *, purpose, depth_units, depth_scale):
        assert purpose == "prepared_source"
        self.calls += 1
        return self.stream.prepare_source(
            arrays, depth_units=depth_units, depth_scale=depth_scale
        )

    def handoff_to_compute(self, *events):
        self.stream.handoff_to_compute(*events)

    def prepare_ready_calibration(self, *, pose, camera_intrinsics):
        return self.stream.prepare_ready_calibration(
            pose=pose,
            camera_intrinsics=camera_intrinsics,
        )

    def prepare_source_tiers(
        self, rgb, depth, camera_intrinsics, *, resampling_mode, declared_tiers,
        depth_downsample="nearest",
    ):
        self.declared_tiers = frozenset(declared_tiers)
        return self.stream.prepare_source_tiers(
            rgb,
            depth,
            camera_intrinsics,
            resampling_mode=resampling_mode,
            declared_tiers=declared_tiers,
            depth_downsample=depth_downsample,
        )


@dataclass
class ExplodingFramePool:
    acquired: int = 0

    def acquire(self):
        self.acquired += 1
        raise AssertionError("new prepared path must not acquire a frame-pool slot")


def packet() -> SensorPacket:
    source = SourceIdentity("camera", 0, 1, 1_000_000_001)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"
        ),
        depth=BufferRef.from_numpy(
            np.full((2, 3), 1000, dtype=np.uint16), encoding="16UC1", units="mm"
        ),
        intrinsics=CameraIntrinsics(
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=1.0,
            width=3,
            height=2,
            depth_scale=1000.0,
        ),
        tracking=TrackingObservation(
            source=source,
            pose=Pose.identity(),
            status=TrackingStatus.VALID,
            provider="producer-stream-test",
        ),
    )


def _packet_for_source(ordinal: int) -> SensorPacket:
    base = packet()
    source = SourceIdentity("camera", 0, ordinal, 1_000_000_000 + ordinal)
    return replace(
        base,
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((2, 3, 3), 32 * ordinal, dtype=np.uint8), encoding="rgb8"
        ),
        tracking=replace(base.tracking, source=source),
    )


def _producer(owner, frame_pool=None):
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    return PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        frame_pool=frame_pool,
        producer_owner="producer",
    ), ready


def test_owner_preparation_uses_one_stream_and_never_frame_pool_staging():
    stream = FakePrepStream()
    owner = FakeTransferOwner(stream)
    producer, ready = _producer(owner, ExplodingFramePool())

    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        lane=ReadySourceLane.FRESH,
    )

    assert stream.operations == [
        "h2d",
        "rgb_convert",
        "depth_convert",
        "prep_event",
        "calibration",
        "tiers",
    ]
    assert owner.declared_tiers == frozenset((
        ResolutionTier.NATIVE,
        ResolutionTier.HALF,
        ResolutionTier.QUARTER,
    ))
    assert stream.declared_tiers == owner.declared_tiers
    assert stream.handoffs == 0
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.is_ready()


def test_real_owner_preparation_accounts_one_h2d_and_zero_d2h():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)

    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    accounting = owner.accounting()
    assert accounting.h2d_ops == 1
    assert accounting.d2h_ops == 0
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.lifetime_token.slot is None
    assert torch.allclose(
        lease.payloads["rgb"].data,
        torch.full((2, 3, 3), 128 / 255.0),
    )
    assert torch.allclose(lease.payloads["depth_meters"].data, torch.ones((2, 3)))
    assert set(lease.available_tiers) == {"native", "half", "quarter"}
    assert set(lease.tier_payloads) == {"native", "half", "quarter"}
    assert lease.tier_payload("native").rgb.data.data_ptr() == lease.payloads["rgb"].data.data_ptr()
    assert (lease.tier_payload("half").width, lease.tier_payload("half").height) == (1, 1)
    assert (lease.tier_payload("quarter").width, lease.tier_payload("quarter").height) == (1, 1)
    assert owner.accounting().h2d_ops == 1
    assert owner.accounting().d2h_ops == 0


def test_real_owner_co_packs_prepared_sources_with_normalized_pixels_and_isolated_retained_storage():
    from eligsir.runtime.transfers import DeviceTransferOwner

    transfer = DeviceTransferOwner("cpu", max_inflight=4, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")
    packets = (_packet_for_source(1), _packet_for_source(2))

    results = producer.submit_many(
        packets,
        modality_eligibilities=(
            ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        ) * 2,
        lanes=(ReadySourceLane.REPLAY,) * 2,
        arrival_ordinals=tuple(packet.source.source_ordinal for packet in packets),
        priorities=(None, None),
        generations=(3, 4),
        control_tokens=(None, None),
        durable_retained=(True, True),
    )

    assert results == ("enqueued", "enqueued")
    assert transfer.accounting().h2d_ops == 1
    leases = ready.claim_consumer("consumer").select(2)
    assert [lease.source_identity for lease in leases] == [packet.source for packet in packets]
    assert [lease.generation for lease in leases] == [3, 4]
    assert torch.allclose(leases[0].payloads["rgb"].data, torch.full((2, 3, 3), 32 / 255.0))
    assert torch.allclose(leases[1].payloads["rgb"].data, torch.full((2, 3, 3), 64 / 255.0))
    assert leases[0].payloads["c2w"].data.untyped_storage().data_ptr() != leases[1].payloads["c2w"].data.untyped_storage().data_ptr()


def test_co_packed_submission_reserves_before_h2d_and_skips_full_sibling():
    """A nonresident durable sibling contributes neither bytes nor a pool slot."""
    from eligsir.runtime.transfers import DeviceTransferOwner

    first, rejected = _packet_for_source(5), _packet_for_source(6)
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    baseline_transfer = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    baseline_ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    baseline = PreparedSourceProducer(
        baseline_ready, transfer_owner=baseline_transfer, producer_owner="baseline"
    )
    assert baseline.submit(
        first,
        modality_eligibility=eligibility,
        lane=ReadySourceLane.REPLAY,
        durable_retained=True,
    ) is PreparedSourceProduceResult.ENQUEUED
    expected_bytes = baseline_transfer.accounting().h2d_bytes

    transfer = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")

    assert producer.submit_many(
        (first, rejected),
        modality_eligibilities=(eligibility, eligibility),
        lanes=(ReadySourceLane.REPLAY, ReadySourceLane.REPLAY),
        arrival_ordinals=(first.source.source_ordinal, rejected.source.source_ordinal),
        priorities=(None, None),
        generations=(1, 1),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == (
        PreparedSourceProduceResult.ENQUEUED,
        PreparedSourceProduceResult.RETAINED_NOT_RESIDENT,
    )
    assert transfer.accounting().h2d_ops == 1
    assert transfer.accounting().h2d_bytes == expected_bytes
    lease = ready.claim_consumer("consumer").poll(ReadySourceLane.REPLAY)
    assert lease is not None and lease.source_identity == first.source


def test_co_packed_submission_validates_each_admitted_source_once(monkeypatch):
    from eligsir.runtime.transfers import DeviceTransferOwner

    transfer = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")
    packets = (_packet_for_source(7), _packet_for_source(8))
    validated = []
    validate = producer._validate_camera_contract

    def record_validation(packet, rgb, depth, *, allow_neutral_depth=False):
        validated.append(packet.source)
        return validate(packet, rgb, depth, allow_neutral_depth=allow_neutral_depth)

    monkeypatch.setattr(producer, "_validate_camera_contract", record_validation)
    assert producer.submit_many(
        packets,
        modality_eligibilities=(ModalityEligibility(rgb_supervision=True, depth_supervision=True),) * 2,
        lanes=(ReadySourceLane.REPLAY,) * 2,
        arrival_ordinals=tuple(packet.source.source_ordinal for packet in packets),
        priorities=(None, None),
        generations=(1, 1),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == (PreparedSourceProduceResult.ENQUEUED,) * 2
    assert validated == [packet.source for packet in packets]


def test_co_packed_submission_uploads_byte_prefix_and_defers_retryable_tail():
    from eligsir.runtime.transfers import DeviceTransferOwner

    first, tail = _packet_for_source(9), _packet_for_source(10)
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    transfer = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=256)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")

    assert producer.submit_many(
        (first, tail),
        modality_eligibilities=(eligibility, eligibility),
        lanes=(ReadySourceLane.REPLAY, ReadySourceLane.REPLAY),
        arrival_ordinals=(first.source.source_ordinal, tail.source.source_ordinal),
        priorities=(None, None),
        generations=(1, 1),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == (
        PreparedSourceProduceResult.ENQUEUED,
        PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED,
    )
    assert transfer.accounting().h2d_ops == 1
    assert ready.claim_consumer("consumer").poll(ReadySourceLane.REPLAY).source_identity == first.source
    assert producer.submit(
        tail,
        modality_eligibility=eligibility,
        lane=ReadySourceLane.REPLAY,
        generation=1,
        durable_retained=True,
    ) is PreparedSourceProduceResult.ENQUEUED


def test_oversized_batch_head_does_not_block_a_later_fitting_source():
    from eligsir.runtime.transfers import DeviceTransferOwner

    oversized = _packet_for_source(11)
    oversized = replace(
        oversized,
        rgb=BufferRef.from_numpy(np.full((16, 16, 3), 11, dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(np.full((16, 16), 1_000, dtype=np.uint16), encoding="16UC1", units="mm"),
        intrinsics=replace(
            oversized.intrinsics,
            width=16,
            height=16,
            depth_width=16,
            depth_height=16,
            cx=8.0,
            cy=8.0,
        ),
    )
    fitting = _packet_for_source(12)
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    transfer = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=256)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")

    assert producer.submit_many(
        (oversized, fitting),
        modality_eligibilities=(eligibility, eligibility),
        lanes=(ReadySourceLane.REPLAY, ReadySourceLane.REPLAY),
        arrival_ordinals=(oversized.source.source_ordinal, fitting.source.source_ordinal),
        priorities=(None, None),
        generations=(1, 1),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == (
        PreparedSourceProduceResult.DEFERRED_POOL_EXHAUSTED,
        PreparedSourceProduceResult.ENQUEUED,
    )
    assert transfer.accounting().h2d_ops == 1
    assert ready.claim_consumer("consumer").poll(ReadySourceLane.REPLAY).source_identity == fitting.source


def test_co_packed_ready_leases_release_exact_resident_bytes_on_partial_eviction():
    from eligsir.runtime.transfers import DeviceTransferOwner

    transfer = DeviceTransferOwner("cpu", max_inflight=4, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")
    packets = (_packet_for_source(3), _packet_for_source(4))
    assert producer.submit_many(
        packets,
        modality_eligibilities=(ModalityEligibility(rgb_supervision=True, depth_supervision=True),) * 2,
        lanes=(ReadySourceLane.REPLAY,) * 2,
        arrival_ordinals=tuple(packet.source.source_ordinal for packet in packets),
        priorities=(None, None),
        generations=(5, 6),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == ("enqueued", "enqueued")
    leases = tuple(entry.lease for entry in ready._replay)

    def storage_ptrs(lease):
        refs = list(lease.payloads.values())
        for tier in lease.tier_payloads.values():
            refs.extend(value for value in (
                tier.rgb, tier.depth, tier.camera_intrinsics, tier.depth_confidence,
                tier.projection_matrix, tier.full_proj_transform,
            ) if value is not None)
        return {ref.data.untyped_storage().data_ptr() for ref in refs}

    def physical_bytes(lease):
        refs = list(lease.payloads.values())
        for tier in lease.tier_payloads.values():
            refs.extend(value for value in (
                tier.rgb, tier.depth, tier.camera_intrinsics, tier.depth_confidence,
                tier.projection_matrix, tier.full_proj_transform,
            ) if value is not None)
        return sum({
            ref.data.untyped_storage().data_ptr(): ref.data.untyped_storage().nbytes()
            for ref in refs
        }.values())

    assert storage_ptrs(leases[0]).isdisjoint(storage_ptrs(leases[1]))
    before = ready.counters.ready_bytes_current
    expected = sum(nbytes for _key, nbytes in ready._replay[0].resident_bytes)
    assert expected == physical_bytes(leases[0])
    evicted = ready.claim_scheduler("storage-test").evict_replay(
        count=1, protected_sources=frozenset()
    )
    assert evicted[0].source_identity == packets[0].source
    assert ready.counters.ready_bytes_current == before - expected
    assert ready.counters.replay_evicted_bytes >= expected


def test_float_metric_depth_cohort_does_not_retain_a_shared_packed_allocation():
    """A surviving metric-depth lease must not keep an evicted sibling's slab alive."""
    from eligsir.runtime.transfers import DeviceTransferOwner

    transfer = DeviceTransferOwner("cpu", max_inflight=4, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=2)
    producer = PreparedSourceProducer(ready, transfer_owner=transfer, producer_owner="producer")
    packets = tuple(
        replace(
            _packet_for_source(ordinal),
            depth=BufferRef.from_numpy(
                np.full((2, 3), 1.0, dtype=np.float32), encoding="32FC1", units="m"
            ),
        )
        for ordinal in (31, 32)
    )
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    assert producer.submit_many(
        packets,
        modality_eligibilities=(eligibility,) * 2,
        lanes=(ReadySourceLane.REPLAY,) * 2,
        arrival_ordinals=tuple(packet.source.source_ordinal for packet in packets),
        priorities=(None, None),
        generations=(1, 1),
        control_tokens=(None, None),
        durable_retained=(True, True),
    ) == (PreparedSourceProduceResult.ENQUEUED,) * 2

    first, second = tuple(entry.lease for entry in ready._replay)
    assert (
        first.tier_payloads["native"].depth.data.untyped_storage().data_ptr()
        != second.tier_payloads["native"].depth.data.untyped_storage().data_ptr()
    )


def test_real_owner_prepares_camera_geometry_before_ready_binding(monkeypatch):
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)
    source = packet().source
    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        control_token=PreparedIngressControlToken(1, source, None, None),
    )
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert {"world_view_transform", "camera_center", "gsplat_viewmat", "gsplat_camera_center"} <= set(lease.payloads)
    monkeypatch.setattr(
        "scene.cameras.torch.linalg.inv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("binding rebuilt camera geometry")),
    )

    bound = bind_ready_camera(
        ReadySourceBinding(lease, ReadySourceVersions(7, 11, 13)), tier="quarter"
    )

    assert bound.prepared_tier == "quarter"


def test_real_owner_precomputes_arbitrary_pose_inverse_before_upload(monkeypatch):
    """Prepared geometry must not launch a checked inverse after calibration H2D."""
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)
    source_packet = packet()
    affine = np.array(
        [[1.2, 0.1, 0.0, 0.3], [0.0, 0.8, 0.2, -0.4], [0.0, 0.0, 1.5, 2.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    source_packet = replace(
        source_packet,
        tracking=replace(source_packet.tracking, pose=Pose(affine)),
    )
    monkeypatch.setattr(
        "eligsir.runtime.transfers.torch.linalg.inv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared source used a device inverse")
        ),
    )

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    expected = torch.from_numpy(np.linalg.inv(affine)).transpose(0, 1)
    assert torch.allclose(lease.payloads["world_view_transform"].data, expected)
    assert owner.accounting().h2d_ops == 1


def test_singular_prepared_pose_aborts_before_upload_and_ready_offer():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)
    singular = np.eye(4, dtype=np.float32)
    singular[2, 2] = 0.0
    source_packet = replace(
        packet(),
        tracking=replace(packet().tracking, pose=Pose(singular)),
    )

    with pytest.raises(ValueError, match="pose must be invertible"):
        producer.submit(
            source_packet,
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            ),
        )

    assert owner.accounting().h2d_ops == 0
    assert ready._reservations == {}
    assert ready.claim_consumer("consumer").poll() is None


def test_nonfinite_prepared_pose_rejects_before_ready_reservation():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)
    nonfinite = np.eye(4, dtype=np.float32)
    nonfinite[0, 0] = np.nan
    source_packet = packet()
    source_packet = replace(
        source_packet,
        tracking=replace(source_packet.tracking, pose=Pose(nonfinite)),
    )

    with pytest.raises(ValueError, match="tracked pose must be a finite"):
        producer.submit(
            source_packet,
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            ),
        )

    assert owner.accounting().h2d_ops == 0
    assert ready._reservations == {}
    assert ready.claim_consumer("consumer").poll() is None


def test_masked_prepared_tier_carries_canonical_depth_confidence_to_binding():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
        resampling_mode="masked",
    )
    source = packet().source
    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        control_token=PreparedIngressControlToken(1, source, None, None),
    )
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    confidence = lease.tier_payload("half").depth_confidence
    assert confidence is not None

    bound = bind_ready_camera(
        ReadySourceBinding(lease, ReadySourceVersions(7, 11, 13)), tier="half"
    )

    assert bound.camera._resolution_depth_confidence_weight is confidence.data


def test_motion_source_facts_publish_spatial_depth_confidence_without_extra_h2d():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
        motion_depth_confidence_mode="motion",
    )
    source_packet = packet()
    source_packet = replace(
        source_packet,
        depth=BufferRef.from_numpy(
            np.array(((1000, 2000, 4000), (1000, 2000, 4000)), dtype=np.uint16),
            encoding="16UC1",
            units="mm",
        ),
        tracking=replace(
            source_packet.tracking,
            diagnostics={
                "motion_depth_confidence_source": "offline_tum",
                "motion_depth_confidence_reason": "interpolated",
                "motion_depth_confidence_transform": np.array(
                    ((1, 0, 0, 0.5), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
                    dtype=np.float32,
                ),
            },
        ),
    )

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    confidence = lease.tier_payload("native").depth_confidence
    assert confidence is not None
    torch.testing.assert_close(confidence.data[0], torch.tensor((0.5, 0.8, 16 / 17)))
    assert not torch.equal(confidence.data, torch.ones_like(confidence.data))
    assert owner.accounting().h2d_ops == 1
    assert owner.accounting().explicit_sync_waits == 0


@pytest.mark.parametrize("resampling_mode", ("legacy", "masked", "ray_preserving"))
@pytest.mark.parametrize("translation", ({"x": 0.5}, {"y": 0.5}))
def test_motion_confidence_uses_exact_selected_grid_tolerance_for_odd_native_shapes(
    resampling_mode, translation
):
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
        resampling_mode=resampling_mode,
        motion_depth_confidence_mode="motion",
    )
    transform = np.eye(4, dtype=np.float32)
    transform[0 if "x" in translation else 1, 3] = 0.5
    source_packet = replace(
        packet(),
        rgb=BufferRef.from_numpy(np.full((3, 5, 3), 128, dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.full((3, 5), 1000, dtype=np.uint16), encoding="16UC1", units="mm"
        ),
        intrinsics=CameraIntrinsics(
            fx=2.0, fy=2.0, cx=2.0, cy=1.0, width=5, height=3, depth_scale=1000.0
        ),
        tracking=replace(
            packet().tracking,
            diagnostics={
                "motion_depth_confidence_source": "offline_tum",
                "motion_depth_confidence_reason": "interpolated",
                "motion_depth_confidence_transform": transform,
            },
        ),
    )

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    for tier in ("native", "half", "quarter"):
        confidence = lease.tier_payload(tier).depth_confidence
        assert confidence is not None
        torch.testing.assert_close(confidence.data, torch.full_like(confidence.data, 0.5))


@pytest.mark.parametrize("tolerance", (1e-40, 1e40, float(torch.finfo(torch.float32).tiny * 2)))
def test_motion_tolerance_rejects_before_source_upload(tolerance):
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer = PreparedSourceProducer(
        ReadySourceOwner(max_fresh=1, max_replay=1),
        transfer_owner=owner,
        producer_owner="producer",
        motion_depth_confidence_mode="identity",
        motion_depth_confidence_native_tolerance_px=tolerance,
    )

    with pytest.raises(ValueError, match="tolerance"):
        producer.submit(
            packet(),
            modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        )

    assert owner.accounting().h2d_ops == 0


@pytest.mark.parametrize("mode", ("identity", "motion"))
def test_rgb_only_source_skips_motion_transform_and_confidence_work(mode, monkeypatch):
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    uploaded_arities = []
    upload = owner.upload_packed

    def record_upload(arrays, **kwargs):
        uploaded_arities.append(len(arrays))
        return upload(arrays, **kwargs)

    monkeypatch.setattr(owner, "upload_packed", record_upload)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready, transfer_owner=owner, producer_owner="producer", motion_depth_confidence_mode=mode
    )
    monkeypatch.setattr(
        producer,
        "_motion_transform_host",
        lambda _packet: (_ for _ in ()).throw(AssertionError("RGB-only must not extract motion")),
    )
    source_packet = replace(packet(), depth=None)

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata["motion_depth_confidence_preparation"] == "no_depth"
    assert uploaded_arities == [5]
    assert lease.tier_payload("native").depth_confidence is None


def test_compatibility_owner_uses_recorded_neutral_motion_fallback_without_upload():
    stream = FakePrepStream()
    owner = FakeTransferOwner(stream)
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        producer_owner="producer",
        motion_depth_confidence_mode="motion",
    )
    source_packet = packet()
    source_packet = replace(
        source_packet,
        tracking=replace(
            source_packet.tracking,
            diagnostics={
                "motion_depth_confidence_source": "offline_tum",
                "motion_depth_confidence_reason": "interpolated",
                "motion_depth_confidence_transform": np.eye(4, dtype=np.float32),
            },
        ),
    )

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert owner.calls == 1
    assert lease.source_metadata["motion_depth_confidence_preparation"] == "neutral_no_packed_calibration"
    torch.testing.assert_close(
        lease.tier_payload("native").depth_confidence.data,
        torch.ones((2, 3)),
    )


@pytest.mark.parametrize("mode", ("identity", "motion"))
def test_identity_and_unavailable_motion_preserve_masked_confidence_at_all_tiers(mode):
    from eligsir.runtime.transfers import DeviceTransferOwner

    source_packet = replace(
        packet(),
        depth=BufferRef.from_numpy(
            np.array(((1000, 1000, 4000), (1000, 1000, 4000)), dtype=np.uint16),
            encoding="16UC1", units="mm",
        ),
    )
    baseline_ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    baseline = PreparedSourceProducer(
        baseline_ready, transfer_owner=DeviceTransferOwner("cpu", max_pinned_bytes=4096),
        producer_owner="baseline", resampling_mode="masked",
        motion_depth_confidence_mode="identity",
    )
    treatment_ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    treatment = PreparedSourceProducer(
        treatment_ready, transfer_owner=DeviceTransferOwner("cpu", max_pinned_bytes=4096),
        producer_owner="treatment", resampling_mode="masked",
        motion_depth_confidence_mode=mode,
    )
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    baseline.submit(source_packet, modality_eligibility=eligibility)
    treatment.submit(source_packet, modality_eligibility=eligibility)
    baseline_lease = baseline_ready.claim_consumer("baseline").poll()
    treatment_lease = treatment_ready.claim_consumer("treatment").poll()

    assert baseline_lease is not None and treatment_lease is not None
    assert any(
        not torch.allclose(
            baseline_lease.tier_payload(tier).depth_confidence.data,
            torch.ones_like(baseline_lease.tier_payload(tier).depth_confidence.data),
        )
        for tier in ("half", "quarter")
    )
    if mode == "motion":
        assert treatment_lease.source_metadata["motion_depth_confidence_preparation"] == "neutral_source_fallback"
    for tier in ("native", "half", "quarter"):
        torch.testing.assert_close(
            treatment_lease.tier_payload(tier).depth_confidence.data,
            baseline_lease.tier_payload(tier).depth_confidence.data,
        )


def test_bound_prepared_c2_depth_terms_consume_published_motion_weights():
    from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
    from eligsir.optimization.gaussian.true_batch_executor import _resolve_batched_pixel_terms
    from eligsir.runtime.transfers import DeviceTransferOwner

    def bound_camera(diagnostics):
        owner = DeviceTransferOwner("cpu", max_pinned_bytes=4096)
        ready = ReadySourceOwner(max_fresh=1, max_replay=1)
        producer = PreparedSourceProducer(
            ready, transfer_owner=owner, producer_owner=str(id(owner)),
            motion_depth_confidence_mode="motion",
        )
        source_packet = replace(
            packet(),
            depth=BufferRef.from_numpy(
                np.array(((1000, 2000, 4000), (1000, 2000, 4000)), dtype=np.uint16),
                encoding="16UC1", units="mm",
            ),
            tracking=replace(packet().tracking, diagnostics=diagnostics),
        )
        producer.submit(
            source_packet,
            modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
            control_token=PreparedIngressControlToken(1, source_packet.source, None, None),
        )
        lease = ready.claim_consumer(str(id(ready))).poll()
        assert lease is not None
        return bind_ready_camera(ReadySourceBinding(lease, ReadySourceVersions(1, 2, 3))).camera

    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 0.5
    motion = bound_camera({
        "motion_depth_confidence_source": "offline_tum",
        "motion_depth_confidence_reason": "interpolated",
        "motion_depth_confidence_transform": transform,
    })
    fallback = bound_camera({})
    neutral = bound_camera({})
    assert motion._resolution_depth_confidence_weight is not None
    assert fallback._resolution_depth_confidence_weight is not None
    assert not torch.equal(motion._resolution_depth_confidence_weight, fallback._resolution_depth_confidence_weight)
    depth = torch.tensor([[[[101.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], [[[3.0, 101.0, 5.0], [6.0, 7.0, 8.0]]]], requires_grad=True)
    terms = _resolve_batched_pixel_terms(
        args=SimpleNamespace(streaming_objective_batched_pixel_terms=True),
        group=tuple(SimpleNamespace(view=SimpleNamespace(camera=camera)) for camera in (motion, fallback)),
        group_modalities=(SimpleNamespace(rgb_loss_enabled=True, depth_loss_enabled=True),) * 2,
        batch=TrueCameraBatchResult(torch.zeros((2, 3, 2, 3)), torch.ones((2, 1, 2, 3)), depth, None, {}, 2, False),
        iteration=1, tb_writer=None, scalar_log_interval=1, depth_loss_type="l1",
        opt=SimpleNamespace(lambda_dssim=1.0), rgb_measurements=(False, False),
    )
    assert terms is not None and terms[0].depth is not None
    gradient = torch.autograd.grad(terms[0].depth, depth)[0]
    neutral_depth = depth.detach().clone().requires_grad_()
    neutral_terms = _resolve_batched_pixel_terms(
        args=SimpleNamespace(streaming_objective_batched_pixel_terms=True),
        group=tuple(SimpleNamespace(view=SimpleNamespace(camera=camera)) for camera in (neutral, fallback)),
        group_modalities=(SimpleNamespace(rgb_loss_enabled=True, depth_loss_enabled=True),) * 2,
        batch=TrueCameraBatchResult(torch.zeros((2, 3, 2, 3)), torch.ones((2, 1, 2, 3)), neutral_depth, None, {}, 2, False),
        iteration=1, tb_writer=None, scalar_log_interval=1, depth_loss_type="l1",
        opt=SimpleNamespace(lambda_dssim=1.0), rgb_measurements=(False, False),
    )
    assert neutral_terms is not None and neutral_terms[0].depth is not None
    neutral_gradient = torch.autograd.grad(neutral_terms[0].depth, neutral_depth)[0]
    assert terms[0].depth != neutral_terms[0].depth
    assert torch.count_nonzero(gradient[0] - gradient[1]) > 0
    assert not torch.equal(gradient, neutral_gradient)


def test_real_owner_calibration_cannot_be_mutated_by_generic_prepare_hook():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    original_prepare = owner.transfer_stream.prepare
    calibration_hook_calls = 0

    def hostile_prepare(operation):
        nonlocal calibration_hook_calls
        payloads, event = original_prepare(operation)
        if (
            isinstance(payloads, tuple)
            and len(payloads) == 2
            and isinstance(payloads[0], torch.Tensor)
            and tuple(payloads[0].shape) == (4, 4)
        ):
            calibration_hook_calls += 1
            altered = payloads[0].clone()
            altered[0, 3] = 0.75
            return (altered, payloads[1]), event
        return payloads, event

    owner.transfer_stream.prepare = hostile_prepare
    producer, ready = _producer(owner)
    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert calibration_hook_calls == 0
    assert lease is not None
    assert torch.equal(lease.payloads["c2w"].data, torch.eye(4))
    assert owner.accounting().d2h_ops == 0


def test_preparation_is_poll_gated_and_compute_handoff_only_at_consumer():
    stream = FakePrepStream(transfer_gate=Gate(False), prep_gate=Gate(False))
    producer, ready = _producer(FakeTransferOwner(stream))
    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )
    consumer = ready.claim_consumer("consumer")
    assert consumer.poll() is None
    assert stream.handoffs == 0

    stream.transfer_gate.ready = True
    stream.prep_gate.ready = True
    lease = consumer.poll()
    assert lease is not None
    assert stream.handoffs == 0
    lease.handoff_to_compute(FakeTransferOwner(stream))
    assert stream.handoffs == 1
    assert stream.operations[-1] == "compute_handoff"


def test_final_preparation_event_is_the_single_ready_fence():
    stream = FakePrepStream(transfer_gate=Gate(False), prep_gate=Gate(True))
    producer, ready = _producer(FakeTransferOwner(stream))
    producer.submit(packet(), modality_eligibility=ModalityEligibility(rgb_supervision=True))

    # Production's final prep event is recorded after H2D on one stream, so it
    # transitively fences the copy; selection must not pay a second poll gate.
    assert ready.claim_consumer("consumer").poll() is not None


def test_preparation_failure_does_not_leave_owner_slot_claimed():
    """The owner-pool cleanup contract is exercised by its real packed path."""
    from eligsir.runtime.packed_staging import PinnedByteSlabPool
    from eligsir.runtime.transfers import DeviceTransferOwner

    class FailingTransfer:
        device = torch.device("cpu")

        def __init__(self):
            self.stats = type(
                "Stats",
                (),
                {"h2d_ops": 0, "h2d_bytes": 0, "pool_waits": 0, "pool_drops": 0},
            )()

        def h2d(self, host, *, dtype=None):
            raise RuntimeError("copy failed")

    owner = DeviceTransferOwner(
        "cpu", transfer_stream=FailingTransfer(), max_inflight=1, max_pinned_bytes=4096
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        owner.upload_packed((np.zeros((8,), np.uint8),), purpose="prepared_source")
    assert owner._packed_pool.pending_count == 0
    assert owner._packed_pool.acquire(8).claimed is True


def test_base_payload_with_fixed_half_tier_only():
    """Base payload should use full-resolution device tensors when native tier not declared (issue #1846)."""
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.HALF,))
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        prepared_tier_policy=policy,
    )

    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.is_ready()

    # Base payloads should be full-resolution device tensors (from the no-tier fallback)
    assert lease.payloads["rgb"].data.shape == (2, 3, 3)  # Full source resolution
    assert lease.payloads["depth_meters"].data.shape == (2, 3)  # Full source resolution
    assert torch.allclose(lease.payloads["rgb"].data, torch.full((2, 3, 3), 128 / 255.0))
    assert torch.allclose(lease.payloads["depth_meters"].data, torch.ones((2, 3)))
    assert torch.equal(
        lease.payloads["camera_intrinsics"].data,
        torch.tensor(((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))),
    )

    # Lease should expose only the declared "half" tier
    assert set(lease.available_tiers) == {"half"}
    assert set(lease.tier_payloads) == {"half"}
    assert lease.tier_payload("half").width == 1
    assert lease.tier_payload("half").height == 1

    # Base payloads should NOT be pointers to the half tier (they are full-resolution)
    assert lease.payloads["rgb"].data.data_ptr() != lease.tier_payload("half").rgb.data.data_ptr()
    assert (
        lease.payloads["camera_intrinsics"].data.data_ptr()
        != lease.tier_payload("half").camera_intrinsics.data.data_ptr()
    )


def test_base_payload_with_native_and_half_uses_native():
    """Base payload should use native tier when native is declared (existing behavior)."""
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    policy = PreparedTierPolicy.from_supported_tiers((ResolutionTier.NATIVE, ResolutionTier.HALF))
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        ready,
        transfer_owner=owner,
        prepared_tier_policy=policy,
    )

    producer.submit(
        packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.is_ready()

    # Base payloads should be the native tier payloads (existing behavior)
    assert set(lease.available_tiers) == {"native", "half"}
    assert set(lease.tier_payloads) == {"native", "half"}
    assert lease.payloads["rgb"].data.data_ptr() == lease.tier_payload("native").rgb.data.data_ptr()
    assert lease.payloads["depth_meters"].data.data_ptr() == lease.tier_payload("native").depth.data.data_ptr()
    assert (
        lease.payloads["camera_intrinsics"].data.data_ptr()
        == lease.tier_payload("native").camera_intrinsics.data.data_ptr()
    )
