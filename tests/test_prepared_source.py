from __future__ import annotations

import dataclasses

import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_source import (
    PreparedSourceLease,
    PreparedSourceQueue,
    PreparedTierPayload,
)


def identity(n: int = 1) -> SourceIdentity:
    return SourceIdentity("camera", 0, n, n * 1_000)


def lease(n: int = 1, *, payload=None) -> PreparedSourceLease:
    if payload is None:
        payload = BufferRef(
            data=torch.zeros(2),
            spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
            ownership=BufferOwnership.BORROWED,
            owner_token=object(),
        )
    return PreparedSourceLease(
        source_identity=identity(n),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 8, "height": 6},
        tier_preparation_available=("quarter", "full"),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
    )


def test_queue_is_bounded_fifo_and_non_blocking():
    queue = PreparedSourceQueue(max_size=2, max_payload_refs_per_lease=2)
    first, second, third = lease(1), lease(2), lease(3)
    assert queue.offer(first)
    assert queue.try_offer(second)
    assert not queue.offer(third)
    assert queue.poll() is first
    assert queue.try_poll() is second
    assert queue.poll() is None


def test_lease_accepts_unversioned_topology_independent_payloads():
    item = lease()
    assert item.borrowed_device_payloads["rgb"].data.device.type == "cpu"
    assert item.source_metadata["width"] == 8
    assert not any("version" in field.name for field in dataclasses.fields(item))
    assert not hasattr(item, "bind")
    assert not hasattr(item, "finalize")


def test_lease_exposes_immutable_typed_tier_payloads():
    token = object()

    def ref(shape, encoding):
        return BufferRef(
            data=torch.zeros(shape),
            spec=BufferSpec(
                shape,
                "float32",
                MemoryDomain.OTHER_DEVICE,
                encoding,
                units="m" if encoding == "depth_meters" else None,
                device="cpu",
            ),
            ownership=BufferOwnership.BORROWED,
            owner_token=token,
        )

    native = PreparedTierPayload(
        name="native",
        rgb=ref((8, 6, 3), "rgb_float32_hwc"),
        depth=ref((8, 6), "depth_meters"),
        camera_intrinsics=ref((3, 3), "camera_intrinsics"),
        width=6,
        height=8,
    )
    half = PreparedTierPayload(
        name="half",
        rgb=ref((4, 3, 3), "rgb_float32_hwc"),
        depth=ref((4, 3), "depth_meters"),
        camera_intrinsics=ref((3, 3), "camera_intrinsics"),
        width=3,
        height=4,
    )
    item = PreparedSourceLease(
        source_identity=identity(),
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        source_metadata={"width": 6, "height": 8},
        tier_preparation_available=("native", "half"),
        borrowed_device_payloads={"c2w": ref((4, 4), "c2w")},
        device="cpu",
        tier_payloads={"native": native, "half": half},
    )

    assert item.tier_payload("native") is native
    assert item.tier_payload("half").width == 3
    with pytest.raises(KeyError):
        item.tier_payload("quarter")
    with pytest.raises(TypeError):
        item.tier_payloads["quarter"] = native


def test_lease_rejects_tier_payloads_missing_an_advertised_tier():
    item = lease()
    with pytest.raises(ValueError, match="tier payload"):
        dataclasses.replace(
            item,
            tier_preparation_available=("native", "half"),
            tier_payloads={"native": object()},
        )


def test_lease_rejects_empty_identity_and_wrong_payload_device():
    with pytest.raises((TypeError, ValueError)):
        PreparedSourceLease(
            source_identity="",
            modality_eligibility=ModalityEligibility(),
            source_metadata={},
            tier_preparation_available=(),
            borrowed_device_payloads={},
            device="cpu",
        )
    with pytest.raises(ValueError, match="device"):
        lease(payload=BufferRef(
            data=torch.zeros(1),
            spec=BufferSpec((1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="meta"),
            ownership=BufferOwnership.BORROWED,
            owner_token=object(),
        ))


def test_lease_requires_borrowed_buffer_ref_and_lifetime_token():
    with pytest.raises(TypeError, match="BufferRef"):
        lease(payload=torch.zeros(1))
    base = dict(
        data=torch.zeros(1),
        spec=BufferSpec((1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
    )
    with pytest.raises(ValueError, match="borrowed"):
        lease(payload=BufferRef(**base, ownership=BufferOwnership.OWNED, owner_token=object()))
    with pytest.raises(ValueError, match="lifetime"):
        lease(payload=BufferRef(**base, ownership=BufferOwnership.BORROWED))


def test_lease_defensively_copies_and_freezes_mappings():
    metadata = {"width": 8}
    payload = lease().borrowed_device_payloads["rgb"]
    payloads = {"rgb": payload}
    item = PreparedSourceLease(identity(), ModalityEligibility(), metadata, (), payloads, "cpu")
    metadata["width"] = 99
    payloads["other"] = payload
    assert item.source_metadata["width"] == 8
    with pytest.raises(TypeError):
        item.source_metadata["new"] = 1
    with pytest.raises(TypeError):
        item.borrowed_device_payloads["new"] = payload


def test_borrowed_payload_is_transparent_and_zero_copy():
    original = torch.zeros(1)
    item = lease(payload=BufferRef(
        data=original,
        spec=BufferSpec((1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    ))
    view = item.borrowed_device_payloads["rgb"].data
    assert view is original
    assert view.device.type == "cpu"


def test_lease_rejects_mutable_metadata_and_mutable_payload_mapping():
    with pytest.raises(TypeError, match="metadata"):
        PreparedSourceLease(
            source_identity=identity(),
            modality_eligibility=ModalityEligibility(),
            source_metadata={"nested": []},
            tier_preparation_available=(),
            borrowed_device_payloads={"rgb": lease().borrowed_device_payloads["rgb"]},
            device="cpu",
        )


def test_queue_enforces_per_lease_reference_bound_and_bounds():
    item = lease()
    ref = item.borrowed_device_payloads["rgb"]
    two = {"rgb": ref, "depth": ref}
    oversized = dataclasses.replace(item, borrowed_device_payloads=two)
    queue = PreparedSourceQueue(max_size=1, max_payload_refs_per_lease=1)
    assert not queue.offer(oversized)
    with pytest.raises(ValueError):
        PreparedSourceQueue(max_size=0, max_payload_refs_per_lease=1)
    with pytest.raises(ValueError):
        PreparedSourceQueue(max_size=1, max_payload_refs_per_lease=0)
    with pytest.raises(TypeError):
        PreparedSourceQueue(max_size="bad", max_payload_refs_per_lease=1)
