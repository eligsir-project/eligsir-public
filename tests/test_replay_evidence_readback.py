"""Compact replay feedback shares transfer capacity without waiting for it."""

import pytest

torch = pytest.importorskip("torch")

from eligsir.runtime.transfers import DeviceTransferOwner, TransferEvent


def _warm_upload_slabs(owner):
    import numpy as np

    for _ in range(2):
        owner.upload_packed((np.zeros(256, dtype=np.uint8),), purpose="test_ingress")


class _DelayedEvent:
    ready = False

    def query(self):
        return self.ready

    def wait(self):
        raise AssertionError("replay feedback must never join the device")


def test_feedback_readback_is_one_bounded_accounted_packet():
    owner = DeviceTransferOwner("cpu", max_inflight=2, max_pinned_bytes=16384)
    import numpy as np
    for _ in range(2):
        owner.upload_packed((np.zeros(1024, dtype=np.uint8),), purpose="test_ingress")
    values = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    before = owner.accounting()
    readback = owner.readback_supervision_evidence(values)
    assert torch.equal(readback.try_take_cpu_tensor().reshape_as(values), values)
    after = owner.accounting()
    assert after.d2h_ops - before.d2h_ops == 1
    assert after.d2h_bytes - before.d2h_bytes == 512
    assert owner.pending_count == 0


@pytest.mark.parametrize("values", [
    torch.empty(0, 4), torch.empty(1025, 4), torch.empty(32, 5),
    torch.empty(128), torch.empty(32, 4, dtype=torch.float64),
    torch.empty(32, 4, device="meta"),
])
def test_feedback_rejects_wrong_shape_dtype_device_before_transfer(values):
    owner = DeviceTransferOwner("cpu")
    with pytest.raises(ValueError, match="supervision evidence"):
        owner.readback_supervision_evidence(values)
    assert owner.accounting().d2h_ops == 0


def test_abandoned_feedback_keeps_pool_slot_until_event_completion(monkeypatch):
    owner = DeviceTransferOwner("cpu", max_inflight=2, max_pinned_bytes=16384)
    _warm_upload_slabs(owner)
    event = _DelayedEvent()
    stream = owner.transfer_stream
    copy = stream.d2h_into

    def delayed_copy(source, host):
        copy(source, host)
        return TransferEvent(event)

    monkeypatch.setattr(stream, "d2h_into", delayed_copy)
    readback = owner.readback_supervision_evidence(torch.ones(4, 4))
    assert readback.try_take_cpu_tensor() is None
    readback.discard()
    readback.discard()  # shutdown may be repeated
    with pytest.raises(RuntimeError, match="consumed"):
        readback.try_take_cpu_tensor()
    assert owner.pending_count == 1
    with pytest.raises(RuntimeError):
        owner.readback_supervision_evidence(torch.zeros(4, 4))
    event.ready = True
    assert owner.pending_count == 0


@pytest.mark.parametrize("slots,budget", ((1, 240), (2, 240)))
def test_feedback_cannot_take_capacity_needed_by_first_source(slots, budget):
    import numpy as np

    owner = DeviceTransferOwner("cpu", max_inflight=slots, max_pinned_bytes=budget)
    with pytest.raises(RuntimeError, match="PinnedByteSlabPool"):
        owner.readback_supervision_evidence(torch.ones(4, 4))
    owner.upload_packed((np.zeros(208, dtype=np.uint8),), purpose="required_source")
    assert owner.accounting().h2d_ops == 1
    assert owner.accounting().d2h_ops == 0


def test_feedback_borrows_small_spare_without_growth_and_keeps_largest_upload_slab():
    import numpy as np

    owner = DeviceTransferOwner("cpu", max_inflight=3, max_pinned_bytes=272)
    owner.upload_packed((np.zeros(64, dtype=np.uint8),), purpose="small_source")
    owner.upload_packed((np.zeros(208, dtype=np.uint8),), purpose="large_source")
    before = owner.accounting().pinned_bytes
    feedback = owner.readback_supervision_evidence(torch.ones(4, 4))
    # Feedback is deliberately not consumed before mandatory ingress proceeds.
    owner.upload_packed((np.ones(208, dtype=np.uint8),), purpose="next_source")
    assert owner.accounting().pinned_bytes == before == 272
    assert owner.accounting().h2d_ops == 3
    assert torch.equal(feedback.try_take_cpu_tensor(), torch.ones(16))
    replacement = owner.readback_supervision_evidence(torch.full((4, 4), 3.0))
    assert torch.equal(replacement.try_take_cpu_tensor(), torch.full((16,), 3.0))
    assert owner.pending_count == 0
