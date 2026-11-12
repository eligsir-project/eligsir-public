"""D2H source lifetime contracts for the shared transfer stream."""

from __future__ import annotations

from contextlib import nullcontext
import gc
import weakref

import pytest
import torch

from eligsir.runtime.transfers import (
    CameraBatchOffloader,
    DeviceTransferStream,
    packed_readback,
)


class _CopyStream:
    def __init__(self) -> None:
        self.waited_for = []

    def wait_stream(self, producer) -> None:
        self.waited_for.append(producer)


class _Event:
    def __init__(self, *, blocking: bool) -> None:
        self.blocking = blocking
        self.recorded_on = None

    def record(self, stream) -> None:
        self.recorded_on = stream

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


def _side_stream_transfer(monkeypatch):
    """Exercise the CUDA-side branch with CPU tensors and no accelerator."""
    transfer = DeviceTransferStream("cpu")
    copy_stream = _CopyStream()
    producer_stream = object()
    transfer._stream = copy_stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: producer_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    return transfer, copy_stream, producer_stream


def test_d2h_records_source_on_copy_stream_before_return(monkeypatch):
    """A side-stream D2H keeps its device source allocator-live until its event."""
    transfer, copy_stream, producer_stream = _side_stream_transfer(monkeypatch)
    source = torch.arange(8, dtype=torch.float32)
    host = torch.empty_like(source)
    recorded = []

    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append((tensor, stream)),
    )

    event = transfer.d2h_into(source, host)

    assert copy_stream.waited_for == [producer_stream]
    assert recorded == [(source, copy_stream)]
    assert event._event.recorded_on is copy_stream
    assert torch.equal(host, source)


def test_packed_readbacks_register_each_abandoned_temporary_source(monkeypatch):
    """Multiple packed temporaries are registered even when callers abandon them."""
    transfer, copy_stream, _ = _side_stream_transfer(monkeypatch)
    recorded = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append((tensor.data_ptr(), stream, tensor.numel())),
    )

    first = packed_readback((torch.arange(3), torch.arange(4)), transfer)
    second = packed_readback((torch.arange(5), torch.arange(6)), transfer)
    del first, second

    assert [(stream, count) for _ptr, stream, count in recorded] == [
        (copy_stream, 7),
        (copy_stream, 11),
    ]


def test_camera_offloader_registers_each_shape_bucket_source(monkeypatch):
    """One side-stream owner protects every transient camera bucket before waiting."""
    transfer, copy_stream, _ = _side_stream_transfer(monkeypatch)
    original_stack = torch.stack

    class _DeviceImage:
        def __init__(self, shape) -> None:
            self.device = torch.device("cuda")
            self.shape = shape
            self.dtype = torch.float32

        def detach(self):
            return self

    class _Camera:
        def __init__(self, shape) -> None:
            self.original_image = _DeviceImage(shape)

    def stack(images, *, dim=0):
        values = tuple(images)
        return original_stack([torch.ones(image.shape) for image in values], dim=dim)

    recorded = []
    monkeypatch.setattr(torch, "stack", stack)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append((tuple(tensor.shape), stream)),
    )
    offloader = CameraBatchOffloader(transfer=transfer)
    monkeypatch.setattr(offloader, "_transfer", lambda _device: transfer)

    assert offloader.offload((_Camera((1, 2)), _Camera((1, 3)))) == 2
    assert recorded == [((1, 1, 2), copy_stream), ((1, 1, 3), copy_stream)]


def _queue_pending_stream_delay(stream) -> object:
    """Keep a side stream observably pending without a host synchronization."""
    event = torch.cuda.Event(blocking=False)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000)
        event.record(stream)
    assert not event.query()
    return event


def _allocator_counters(device: torch.device) -> dict[str, int]:
    """Return device-allocation counters when this accelerator exposes them."""
    stats = torch.cuda.memory_stats(device)
    return {
        key: int(stats[key])
        for key in ("num_device_alloc", "num_alloc_retries")
        if key in stats
    }


_ACCELERATOR_DELAY_UNAVAILABLE = not torch.cuda.is_available() or not hasattr(
    torch.cuda, "_sleep"
)


@pytest.mark.skipif(_ACCELERATOR_DELAY_UNAVAILABLE, reason="needs CUDA/HIP stream delay")
@pytest.mark.cuda
def test_d2h_delayed_readback_survives_released_source_allocator_reuse(monkeypatch):
    """A D2H source remains valid while same-size default-stream fills run."""
    device = torch.device("cuda", torch.cuda.current_device())
    transfer = DeviceTransferStream(device)
    warmup = [torch.full((262_144,), -5.0, device=device) for _ in range(16)]
    torch.cuda.current_stream(device).synchronize()
    del warmup
    source = torch.arange(262_144, device=device, dtype=torch.float32)
    host = torch.empty_like(source, device="cpu", pin_memory=True)
    expected = torch.arange(262_144, dtype=torch.float32)
    source_ptr = source.data_ptr()
    original_record_stream = torch.Tensor.record_stream
    recorded = []

    def record_stream(tensor, stream):
        recorded.append((tensor.data_ptr(), stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    delayed = _queue_pending_stream_delay(transfer._stream)
    event = transfer.d2h_into(source, host)
    del source
    allocation_counters = _allocator_counters(device)
    pressure = [torch.full((262_144,), -7.0, device=device) for _ in range(8)]
    torch.cuda.current_stream(device).synchronize()
    assert not delayed.query()
    if allocation_counters:
        assert _allocator_counters(device) == allocation_counters
    del pressure
    event.wait()

    assert recorded == [(source_ptr, transfer._stream)]
    assert torch.equal(host, expected)


@pytest.mark.skipif(_ACCELERATOR_DELAY_UNAVAILABLE, reason="needs CUDA/HIP stream delay")
@pytest.mark.cuda
def test_packed_readbacks_survive_delayed_stream_after_abandonment(monkeypatch):
    """Several temporary packed sources remain correct after their wrappers are dropped."""
    device = torch.device("cuda", torch.cuda.current_device())
    transfer = DeviceTransferStream(device)
    # Warm the exact cat + pinned-D2H operator before timing the side-stream
    # delay. Lazy setup is not part of the source-lifetime overlap being tested.
    warm_sources = (
        torch.zeros((2,), device=device),
        torch.zeros((2,), device=device),
    )
    warmup = packed_readback(warm_sources, transfer)
    warmup.event.wait()
    del warmup, warm_sources
    torch.cuda.current_stream(device).synchronize()
    sources = [
        (torch.full((2,), 1.0, device=device), torch.full((2,), 2.0, device=device)),
        (torch.full((2,), 3.0, device=device), torch.full((2,), 4.0, device=device)),
    ]
    expected = [torch.tensor((1.0, 1.0, 2.0, 2.0)), torch.tensor((3.0, 3.0, 4.0, 4.0))]

    original_d2h_into = transfer.d2h_into
    temporary_sources = []

    def d2h_into(source, host):
        temporary_sources.append(weakref.ref(source))
        return original_d2h_into(source, host)

    monkeypatch.setattr(transfer, "d2h_into", d2h_into)

    delayed = _queue_pending_stream_delay(transfer._stream)
    readbacks = [packed_readback(pair, transfer) for pair in sources]
    pending = [(readback.host, readback.event) for readback in readbacks]
    del readbacks, sources
    gc.collect()
    assert len(temporary_sources) == 2
    assert all(source() is None for source in temporary_sources)
    allocation_counters = _allocator_counters(device)
    pressure = [torch.full((4,), -9.0, device=device) for _ in range(8)]
    torch.cuda.current_stream(device).synchronize()
    assert not delayed.query()
    if allocation_counters:
        assert _allocator_counters(device) == allocation_counters
    del pressure
    for (host, event), value in zip(pending, expected, strict=True):
        event.wait()
        assert torch.equal(host, value)


@pytest.mark.skipif(_ACCELERATOR_DELAY_UNAVAILABLE, reason="needs CUDA/HIP stream delay")
@pytest.mark.cuda
def test_camera_buckets_survive_delayed_stream_after_first_batch_reference_drops(monkeypatch):
    """The first bucket cannot be reused while a later same-size bucket is submitted."""
    device = torch.device("cuda", torch.cuda.current_device())
    transfer = DeviceTransferStream(device)

    class _Camera:
        def __init__(self, image) -> None:
            self.original_image = image

    first = _Camera(torch.full((1, 4), 1.0, device=device))
    second = _Camera(torch.full((2, 2), 2.0, device=device))
    original_stack = torch.stack
    pressure = []
    calls = 0

    def stack(images, *, dim=0):
        nonlocal calls
        result = original_stack(tuple(images), dim=dim)
        calls += 1
        if calls == 2:
            pressure.append(torch.full((4,), -11.0, device=device))
            torch.cuda.current_stream(device).synchronize()
            assert not delayed.query()
        return result

    monkeypatch.setattr(torch, "stack", stack)
    delayed = _queue_pending_stream_delay(transfer._stream)
    assert CameraBatchOffloader(transfer=transfer).offload((first, second)) == 2
    assert torch.equal(first.original_image, torch.ones((1, 4)))
    assert torch.equal(second.original_image, torch.full((2, 2), 2.0))
