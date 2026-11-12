from __future__ import annotations

from dataclasses import replace

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
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_source_producer import (
    PreparedSourceProduceResult,
    PreparedSourceProducer,
)
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.runtime.ready_camera_binder import bind_ready_camera
from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions
from eligsir.runtime.transfers import DeviceTransferOwner, DeviceTransferStream, TransferEvent


class _Gate:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class _PrepStream:
    """CPU stand-in for the owner's transfer/preparation stream."""

    device = torch.device("cpu")

    def __init__(self, *, metadata_gate: _Gate | None = None) -> None:
        self.h2d_calls = 0
        self.metadata_prepare_calls = 0
        self.metadata_gate = metadata_gate or _Gate()

    def prepare_source(self, arrays, *, depth_units, depth_scale):
        del depth_units, depth_scale
        self.h2d_calls += 1
        rgb, depth = arrays
        return (
            (
                torch.from_numpy(np.asarray(rgb)).to(dtype=torch.float32).div(255.0),
                (
                    None
                    if depth is None
                    else torch.from_numpy(np.asarray(depth)).to(dtype=torch.float32).div(1000.0)
                ),
            ),
            TransferEvent(_Gate()),
            TransferEvent(_Gate()),
        )

    def prepare_ready_calibration(self, *, pose, camera_intrinsics):
        self.metadata_prepare_calls += 1
        return (
            (
                torch.tensor(pose, dtype=torch.float32),
                torch.tensor(camera_intrinsics, dtype=torch.float32),
            ),
            TransferEvent(self.metadata_gate),
        )


class _TransferOwner:
    device = torch.device("cpu")

    def __init__(self, *, metadata_gate: _Gate | None = None) -> None:
        self.transfer_stream = _PrepStream(metadata_gate=metadata_gate)

    def prepare_source(self, arrays, *, purpose, depth_units, depth_scale):
        assert purpose == "prepared_source"
        return self.transfer_stream.prepare_source(
            arrays, depth_units=depth_units, depth_scale=depth_scale
        )

    def handoff_to_compute(self, *events):
        del events

    def prepare_ready_calibration(self, *, pose, camera_intrinsics):
        return self.transfer_stream.prepare_ready_calibration(
            pose=pose,
            camera_intrinsics=camera_intrinsics,
        )


def _packet(
    *,
    tracking: bool = True,
    depth_shape=(2, 3),
    width: int = 3,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    depth_units: str = "mm",
    depth_scale: float = 1000.0,
    source_ordinal: int = 9,
) -> SensorPacket:
    source = SourceIdentity(
        "metadata-camera", 4, source_ordinal, 1_700_000_000 + source_ordinal
    )
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 0.25
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((2, 3, 3), 128, dtype=np.uint8) if rgb is None else rgb,
            encoding="rgb8",
        ),
        depth=(
            BufferRef.from_numpy(
                np.full(depth_shape, 1000, dtype=np.uint16) if depth is None else depth,
                encoding="16UC1",
                units=depth_units,
            )
            if depth_shape is not None
            else None
        ),
        intrinsics=CameraIntrinsics(
            fx=8.0,
            fy=9.0,
            cx=1.5,
            cy=1.0,
            width=width,
            height=2 if rgb is None else int(rgb.shape[0]),
            depth_scale=depth_scale,
        ),
        tracking=(
            TrackingObservation(
                source=source,
                pose=Pose(pose),
                status=TrackingStatus.VALID,
                provider="metadata-test",
            )
            if tracking
            else None
        ),
    )


def _producer(owner: _TransferOwner, *, measure_replay_quality: bool = False):
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    quality_kwargs = (
        {"measure_replay_quality": True} if measure_replay_quality else {}
    )
    return (
        PreparedSourceProducer(
            ready,
            transfer_owner=owner,
            producer_owner="producer",
            **quality_kwargs,
        ),
        ready,
    )


def test_producer_emits_authoritative_metadata_and_device_calibration_payloads():
    owner = _TransferOwner()
    producer, ready = _producer(owner)
    source_packet = _packet()

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata == {
        "source_id": "metadata-camera",
        "stream_epoch": 4,
        "source_ordinal": 9,
        "timestamp_ns": 1_700_000_009,
        "rgb_shape": (2, 3, 3),
        "depth_shape": (2, 3),
        "depth_units": "m",
        "depth_aligned_to_rgb": True,
        "camera_intrinsics": (
            8.0, 9.0, 1.5, 1.0, 3, 2,
            8.0, 9.0, 1.5, 1.0, 3, 2,
            1000.0, "", "", "", None, "K",
        ),
        "pose_c2w": (
            (1.0, 0.0, 0.0, 0.25),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        "calibration_attestation": (
            "prepared-calibration-v1",
            (
                (1.0, 0.0, 0.0, 0.25),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            ((8.0, 0.0, 1.5), (0.0, 9.0, 1.0), (0.0, 0.0, 1.0)),
        ),
    }
    assert set(lease.payloads) == {"rgb", "depth_meters", "c2w", "camera_intrinsics"}
    assert torch.equal(
        lease.payloads["c2w"].data,
        torch.tensor(
            ((1.0, 0.0, 0.0, 0.25), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        lease.payloads["camera_intrinsics"].data,
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 9.0, 1.0), (0.0, 0.0, 1.0))),
    )
    assert owner.transfer_stream.h2d_calls == 1
    assert owner.transfer_stream.metadata_prepare_calls == 1


def test_measured_replay_quality_is_modality_specific_metric_depth_metadata():
    owner = _TransferOwner()
    producer, ready = _producer(owner, measure_replay_quality=True)
    rgb = np.indices((8, 8)).sum(axis=0).astype(np.uint8) % 2 * 128 + 64
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    depth_mm = np.full((8, 8), 1000, dtype=np.uint16)
    source_packet = _packet(
        rgb=rgb,
        depth=depth_mm,
        depth_shape=(8, 8),
        width=8,
    )

    assert producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata["replay_quality"] == (1.0, 1.0)
    assert owner.transfer_stream.h2d_calls == 1


def test_measured_replay_quality_keeps_good_rgb_when_depth_is_unusable():
    owner = _TransferOwner()
    producer, ready = _producer(owner, measure_replay_quality=True)
    rgb = np.indices((8, 8)).sum(axis=0).astype(np.uint8) % 2 * 128 + 64
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    depth_mm = np.zeros((8, 8), dtype=np.uint16)

    assert producer.submit(
        _packet(rgb=rgb, depth=depth_mm, depth_shape=(8, 8), width=8),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata["replay_quality"] == (1.0, 0.0)


def test_measured_replay_quality_uses_the_same_metric_depth_units_as_preparation():
    rgb = np.indices((8, 8)).sum(axis=0).astype(np.uint8) % 2 * 128 + 64
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    depth_mm = np.full((8, 8), 1000, dtype=np.uint16)
    depth_m = np.full((8, 8), 1.0, dtype=np.float32)

    producer_mm, ready_mm = _producer(_TransferOwner(), measure_replay_quality=True)
    producer_m, ready_m = _producer(_TransferOwner(), measure_replay_quality=True)
    modality = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    assert producer_mm.submit(
        _packet(rgb=rgb, depth=depth_mm, depth_shape=(8, 8), width=8),
        modality_eligibility=modality,
    ) is PreparedSourceProduceResult.ENQUEUED
    assert producer_m.submit(
        _packet(
            rgb=rgb,
            depth=depth_m,
            depth_shape=(8, 8),
            width=8,
            depth_units="m",
            depth_scale=1.0,
        ),
        modality_eligibility=modality,
    ) is PreparedSourceProduceResult.ENQUEUED

    mm_lease = ready_mm.claim_consumer("consumer-mm").poll()
    m_lease = ready_m.claim_consumer("consumer-m").poll()
    assert mm_lease is not None and m_lease is not None
    assert mm_lease.source_metadata["replay_quality"] == m_lease.source_metadata["replay_quality"]


def test_measured_replay_quality_marks_neutral_depth_unmeasured():
    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner, measure_replay_quality=True)
    rgb = np.indices((8, 8)).sum(axis=0).astype(np.uint8) % 2 * 128 + 64
    rgb = np.repeat(rgb[..., None], 3, axis=2)

    assert producer.submit(
        _packet(rgb=rgb, depth_shape=None, width=8),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED

    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata["replay_quality"] == (1.0, None)


def test_capacity_rejection_skips_opted_in_replay_quality_work(monkeypatch):
    import eligsir.runtime.prepared_source_producer as producer_module

    owner = _TransferOwner()
    producer, _ready = _producer(owner, measure_replay_quality=True)
    assert producer.submit(
        _packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED
    monkeypatch.setattr(
        producer_module,
        "evaluate_frame_quality",
        lambda *_args, **_kwargs: pytest.fail("quality ran after capacity rejection"),
    )

    assert producer.submit(
        _packet(source_ordinal=10),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.DROPPED_READY_OWNER


def test_replay_quality_failure_aborts_reservation_without_transfer(monkeypatch):
    import eligsir.runtime.prepared_source_producer as producer_module

    owner = _TransferOwner()
    producer, ready = _producer(owner, measure_replay_quality=True)
    source_packet = _packet()
    evaluate = producer_module.evaluate_frame_quality
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("quality failure")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(producer_module, "evaluate_frame_quality", fail_once)
    with pytest.raises(RuntimeError, match="quality failure"):
        producer.submit(
            source_packet,
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            ),
        )
    assert owner.transfer_stream.h2d_calls == 0

    assert producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED
    assert ready.claim_consumer("consumer").poll() is not None


def test_replay_quality_is_not_evaluated_when_opt_in_is_off(monkeypatch):
    import eligsir.runtime.prepared_source_producer as producer_module

    owner = _TransferOwner()
    producer, ready = _producer(owner)
    monkeypatch.setattr(
        producer_module,
        "evaluate_frame_quality",
        lambda *_args, **_kwargs: pytest.fail("quality evaluation was not opted in"),
        raising=False,
    )

    assert producer.submit(
        _packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
    ) is PreparedSourceProduceResult.ENQUEUED
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert "replay_quality" not in lease.source_metadata


def test_producer_calibration_bypasses_mutable_generic_prepare_hook():
    owner = _TransferOwner()
    producer, ready = _producer(owner)
    prepare_calls = 0

    def corrupt_calibration(operation):
        nonlocal prepare_calls
        prepare_calls += 1
        payloads, event = operation(), TransferEvent(_Gate())
        c2w, intrinsics = payloads
        altered = c2w.clone()
        altered[0, 3] = 0.75
        return (altered, intrinsics), event

    owner.transfer_stream.prepare = corrupt_calibration

    producer.submit(
        _packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )

    lease = ready.claim_consumer("consumer").poll()
    assert prepare_calls == 0
    assert lease is not None
    assert torch.equal(
        lease.payloads["c2w"].data,
        torch.tensor(
            (
                (1.0, 0.0, 0.0, 0.25),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    assert torch.equal(
        lease.payloads["camera_intrinsics"].data,
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 9.0, 1.0), (0.0, 0.0, 1.0))),
    )


def test_device_calibration_constructor_bypasses_mutable_generic_prepare_hook():
    stream = DeviceTransferStream("cpu")

    def forbidden_prepare(_operation):
        raise AssertionError("sealed calibration entered generic prepare hook")

    stream.prepare = forbidden_prepare
    payloads, event = stream.prepare_ready_calibration(
        pose=(
            (1.0, 0.0, 0.0, 0.25),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        camera_intrinsics=((8.0, 0.0, 1.5), (0.0, 9.0, 1.0), (0.0, 0.0, 1.0)),
    )

    c2w, intrinsics = payloads
    assert event.query()
    assert stream.stats.d2h_ops == 0
    assert stream.stats.d2h_bytes == 0
    assert torch.equal(
        c2w,
        torch.tensor(
            (
                (1.0, 0.0, 0.0, 0.25),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
    )
    assert torch.equal(
        intrinsics,
        torch.tensor(((8.0, 0.0, 1.5), (0.0, 9.0, 1.0), (0.0, 0.0, 1.0))),
    )


def test_producer_ready_event_includes_calibration_preparation_fence():
    metadata_gate = _Gate(False)
    owner = _TransferOwner(metadata_gate=metadata_gate)
    producer, ready = _producer(owner)

    producer.submit(
        _packet(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )
    consumer = ready.claim_consumer("consumer")
    assert consumer.poll() is None

    metadata_gate.ready = True
    lease = consumer.poll()
    assert lease is not None


def test_producer_lease_binds_without_a_second_transfer_or_cold_materialization():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner, measure_replay_quality=True)
    source_packet = _packet()
    token = PreparedIngressControlToken(3, source_packet.source, None, None)

    producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        control_token=token,
    )
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert "replay_quality" in lease.source_metadata
    binding = bind_ready_camera(
        ReadySourceBinding(lease, ReadySourceVersions(1, 2, 3))
    )

    assert binding.control_token is token
    assert binding.camera.original_image.data_ptr() == lease.payloads["rgb"].data.data_ptr()
    assert binding.camera.gsplat.K is lease.payloads["camera_intrinsics"].data
    assert owner.accounting().h2d_ops == 1


@pytest.mark.parametrize(
    "packet_kwargs, expected_error",
    [
        ({"tracking": False}, "pose"),
        ({"width": 4}, "intrinsics"),
        ({"depth_shape": (1, 3)}, "aligned"),
    ],
)
def test_producer_rejects_incomplete_or_mismatched_camera_contract_before_transfer(
    packet_kwargs, expected_error
):
    owner = _TransferOwner()
    producer, _ready = _producer(owner)

    with pytest.raises(ValueError, match=expected_error):
        producer.submit(
            _packet(**packet_kwargs),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
        )

    assert owner.transfer_stream.h2d_calls == 0


def test_rgb_only_prepared_source_binds_with_neutral_depth_and_no_depth_modalities():
    """RGB-only exact work is resident without inventing depth supervision."""

    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, ready = _producer(owner)
    source_packet = _packet()
    source_packet = replace(
        source_packet,
        depth=None,
        intrinsics=replace(source_packet.intrinsics, depth_scale=0.0),
    )
    token = PreparedIngressControlToken(3, source_packet.source, None, None)

    result = producer.submit(
        source_packet,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        control_token=token,
    )

    assert result is PreparedSourceProduceResult.ENQUEUED
    lease = ready.claim_consumer("consumer").poll()
    assert lease is not None
    assert lease.source_metadata["prepared_neutral_depth"] is True
    assert owner.accounting().h2d_ops == 1
    bound = bind_ready_camera(
        ReadySourceBinding(lease, ReadySourceVersions(1, 2, 3))
    )
    assert bound.modality_eligibility == ModalityEligibility(rgb_supervision=True)
    assert torch.count_nonzero(bound.camera._sensor_depth_cache) == 0


def test_rgb_only_prepared_source_rejects_depth_modalities_before_h2d():
    from eligsir.runtime.transfers import DeviceTransferOwner

    owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096)
    producer, _ready = _producer(owner)

    result = producer.submit(
        replace(_packet(), depth=None),
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
        ),
    )

    assert result is PreparedSourceProduceResult.CAPABILITY_UNAVAILABLE
    assert owner.accounting().h2d_ops == 0


@pytest.mark.parametrize("status", [TrackingStatus.INVALID, TrackingStatus.LOST])
def test_owner_prepared_path_rejects_non_usable_pose_before_transfer(status):
    owner = _TransferOwner()
    producer, _ready = _producer(owner)
    source_packet = _packet()
    invalid_tracking = TrackingObservation(
        source=source_packet.source,
        pose=Pose.identity(),
        status=status,
        provider="metadata-test",
    )

    with pytest.raises(ValueError, match="usable"):
        producer.submit(
            replace(source_packet, tracking=invalid_tracking),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
        )
    assert owner.transfer_stream.h2d_calls == 0


def test_legacy_upload_path_preserves_permissive_untracked_packet_behavior():
    from eligsir.runtime.prepared_source_producer import PreparedSourceProduceResult

    class LegacyTransfer:
        device = torch.device("cpu")

        def upload_packed(self, arrays, *, purpose):
            assert purpose == "prepared_source"
            rgb, depth = arrays
            return (
                (
                    torch.from_numpy(np.asarray(rgb)),
                    torch.from_numpy(np.asarray(depth)),
                ),
                TransferEvent(_Gate()),
            )

    class LegacyPool:
        def acquire(self):
            return self

        def stage(self, rgb, depth):
            return rgb, depth

        def release_after(self, _slot, _event):
            pass

    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=LegacyTransfer(),
        frame_pool=LegacyPool(),
        producer_owner="legacy-producer",
    )
    result = producer.submit(
        replace(_packet(), tracking=None),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
    )

    assert result is PreparedSourceProduceResult.ENQUEUED
    lease = owner.claim_consumer("consumer").poll()
    assert lease is not None
    assert set(lease.payloads) == {"rgb", "depth_meters"}
    assert lease.source_metadata["intrinsics"] == (8.0, 9.0, 1.5, 1.0)
