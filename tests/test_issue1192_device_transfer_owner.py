from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.transfers import (
    DeviceTransferOwner,
    TransferEvent,
    get_device_transfer_owner,
)
from eligsir.runtime.packed_staging import layout_numpy_fields


class _Stats:
    def __init__(self) -> None:
        self.h2d_ops = 0
        self.h2d_bytes = 0
        self.d2h_ops = 0
        self.d2h_bytes = 0
        self.pool_waits = 0
        self.pool_drops = 0

    def snapshot(self):
        return {
            "h2d_ops": self.h2d_ops,
            "h2d_bytes": self.h2d_bytes,
            "d2h_ops": self.d2h_ops,
            "d2h_bytes": self.d2h_bytes,
            "pool_waits": self.pool_waits,
            "pool_drops": self.pool_drops,
        }


class _ImmediateTransfer:
    """CPU test double for the existing stream primitive, not another owner."""

    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.stats = _Stats()
        self.handoffs = 0

    def h2d(self, host, *, dtype=None):
        out = host.clone()
        if dtype is not None:
            out = out.to(dtype=dtype)
        self.stats.h2d_ops += 1
        self.stats.h2d_bytes += host.numel() * host.element_size()
        return out, TransferEvent()

    def handoff_to_compute(self, *events):
        self.handoffs += len(events)

    def d2h_into(self, device_tensor, host):
        host.copy_(device_tensor)
        self.stats.d2h_ops += 1
        self.stats.d2h_bytes += device_tensor.numel() * device_tensor.element_size()
        return TransferEvent()


class _PendingTransfer(_ImmediateTransfer):
    class _Gate:
        def query(self):
            return False

    def h2d(self, host, *, dtype=None):
        out, _event = super().h2d(host, dtype=dtype)
        return out, TransferEvent(self._Gate())


class _Cam:
    def __init__(self, shape=(3, 4, 5), dtype=torch.float32, *, with_depth=False):
        self.original_image = torch.zeros(shape, dtype=dtype)
        self._sensor_depth_cache = (
            torch.ones((1, shape[-2], shape[-1]), dtype=torch.float32)
            if with_depth
            else None
        )


def _packed_source_fields(ordinal: int):
    return (
        np.full((2, 3, 3), ordinal, dtype=np.uint8),
        np.full((2, 3), ordinal + 1, dtype=np.uint16),
        np.eye(4, dtype=np.float32) * (ordinal + 1),
        np.eye(3, dtype=np.float32) * (ordinal + 2),
        np.eye(4, dtype=np.float32) * (ordinal + 3),
    )


def test_owner_packs_fields_through_one_stream_and_publishes_frozen_accounting():
    stream = _ImmediateTransfer()
    owner = DeviceTransferOwner(
        "cuda",
        transfer_stream=stream,
        max_inflight=2,
        max_shape_buckets=2,
        max_pinned_bytes=4096,
    )

    fields, event = owner.upload_packed(
        (np.zeros((2, 3), np.float32), np.ones((4,), np.int16)),
        purpose="insertion",
    )

    assert len(fields) == 2
    assert event.query()
    accounting = owner.accounting()
    assert accounting.h2d_ops == 1
    assert accounting.h2d_bytes == 32
    assert accounting.inflight_limit == 2
    assert accounting.shape_bucket_limit == 2
    with pytest.raises(FrozenInstanceError):
        accounting.h2d_ops = 99


def test_prepare_packed_sources_co_packs_calibrated_sources_once_and_preserves_exact_values():
    stream = _ImmediateTransfer()
    owner = DeviceTransferOwner(
        "cuda", transfer_stream=stream, max_inflight=4, max_pinned_bytes=4096
    )
    sources = tuple(_packed_source_fields(ordinal) for ordinal in range(4))

    uploaded, event = owner.prepare_packed_sources(sources, purpose="prepared_cold")

    assert owner.accounting().h2d_ops == 1
    assert owner.accounting().h2d_bytes == layout_numpy_fields(
        field for source in sources for field in source
    )[2]
    assert torch.equal(uploaded[2][0], torch.from_numpy(sources[2][0]))
    assert torch.equal(uploaded[2][2], torch.from_numpy(sources[2][2]))
    assert event.query()


def test_prepare_packed_sources_rejects_valid_work_when_the_only_upload_slot_is_inflight():
    stream = _PendingTransfer()
    owner = DeviceTransferOwner(
        "cuda", transfer_stream=stream, max_inflight=1, max_pinned_bytes=4096
    )
    owner.upload_packed((np.zeros((8,), dtype=np.uint8),), purpose="first")

    with pytest.raises(RuntimeError, match="in-flight transfer depth"):
        owner.prepare_packed_sources((_packed_source_fields(0),), purpose="prepared_cold")

    assert owner.accounting().h2d_ops == 1
    assert owner.accounting().pool_waits == 0


def test_owner_fails_closed_before_exceeding_total_pinned_capacity():
    owner = DeviceTransferOwner(
        "cuda",
        transfer_stream=_ImmediateTransfer(),
        max_inflight=1,
        max_shape_buckets=1,
        max_pinned_bytes=8,
    )

    with pytest.raises(RuntimeError, match="pinned byte limit"):
        owner.upload_packed((np.zeros((4,), np.float32),), purpose="frame")

    accounting = owner.accounting()
    assert accounting.h2d_ops == 0
    assert accounting.pool_drops == 1
    assert accounting.pinned_bytes == 0


def test_owner_bounds_camera_shape_buckets_and_hands_events_to_compute():
    stream = _ImmediateTransfer()
    owner = DeviceTransferOwner(
        "cuda",
        transfer_stream=stream,
        max_inflight=1,
        max_shape_buckets=1,
        max_pinned_bytes=4096,
    )

    assert owner.prefetch_cameras((_Cam(),)) == 1
    assert stream.handoffs == 1
    with pytest.raises(RuntimeError, match="shape bucket limit"):
        owner.prefetch_cameras((_Cam(shape=(3, 2, 2)),))

    accounting = owner.accounting()
    assert accounting.shape_buckets == 1
    assert accounting.pool_drops == 1


def test_owner_prefetches_sensor_depth_through_the_same_transfer_boundary():
    stream = _ImmediateTransfer()
    owner = DeviceTransferOwner(
        "cuda",
        transfer_stream=stream,
        max_inflight=1,
        max_shape_buckets=2,
        max_pinned_bytes=4096,
    )
    camera = _Cam(with_depth=True)

    assert owner.prefetch_cameras((camera,)) == 2
    assert stream.handoffs == 2
    assert torch.equal(camera._sensor_depth_cache, torch.ones((1, 4, 5)))


def test_compatibility_registry_returns_one_owner_per_device():
    assert get_device_transfer_owner("cpu") is get_device_transfer_owner(torch.device("cpu"))


def test_compatibility_registry_honors_and_fences_explicit_pinned_budget():
    owner = get_device_transfer_owner("meta:1192", max_pinned_bytes=4096)

    assert owner.max_pinned_bytes == 4096
    assert get_device_transfer_owner("meta:1192") is owner
    with pytest.raises(RuntimeError, match="already configured"):
        get_device_transfer_owner("meta:1192", max_pinned_bytes=8192)


def test_owner_camera_page_out_uses_its_stream_and_publishes_d2h_accounting(monkeypatch):
    stream = _ImmediateTransfer()
    owner = DeviceTransferOwner(
        "cuda",
        transfer_stream=stream,
        max_inflight=1,
        max_shape_buckets=1,
        max_pinned_bytes=4096,
    )
    assert owner.prefetch_cameras((_Cam(),)) == 1

    class _AcceleratorImage:
        device = torch.device("cuda")
        shape = (3, 4, 5)
        dtype = torch.float32

        def detach(self):
            return self

    cam = _Cam()
    cam.original_image = _AcceleratorImage()

    # Exercise the real bucket/pool/offload path without requiring an accelerator
    # in this ownership test. The source image's device marks it page-out eligible;
    # stacking yields the CPU test-double payload consumed by _ImmediateTransfer.
    monkeypatch.setattr(
        "eligsir.runtime.transfers.torch.stack",
        lambda _images, dim=0: torch.ones((1, 3, 4, 5), dtype=torch.float32),
    )

    assert owner.offload_cameras((cam,)) == 1
    accounting = owner.accounting()
    assert accounting.d2h_ops == 1
    assert accounting.d2h_bytes == 1 * 3 * 4 * 5 * 4
    assert accounting.shape_buckets == 1
    assert accounting.pinned_bytes == accounting.peak_pinned_bytes == 480
    assert cam.original_image.device.type == "cpu"
