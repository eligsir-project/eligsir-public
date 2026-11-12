"""Bounded, coherent, latest-wins map-output lane for issue #1200."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
import pytest
import torch
from eligsir.map.views import sample_output_from_model
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.viewers.events import MapPreviewSnapshot
from eligsir.viewers.map_output_lane import MapOutputVersion, VersionedMapOutputLane

def _model(n: int, device="cpu"):
    x = torch.arange(n, dtype=torch.float32, device=device)
    return SimpleNamespace(params={
        "means": torch.stack((x, x + 100, x + 200), 1),
        "scales": torch.full((n, 3), -2.0, device=device),
        "quats": torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long, device=device), 4).float(),
        "opacities": torch.full((n,), 1.0, device=device),
        "sh0": torch.zeros((n, 1, 3), device=device),
    })

class _Backend:
    def __init__(self, n=6, device="cpu"):
        self.model, self.content, self.topology = _model(n, device), 1, 1
        self.sample_calls = 0
    def sample_output(self, *, max_gaussians, parameter_version):
        self.sample_calls += 1
        return sample_output_from_model(self.model, max_gaussians=max_gaussians,
            map_content_version=self.content, topology_version=self.topology,
            parameter_version=parameter_version)
    @property
    def map_content_version(self): return self.content
    @property
    def map_version(self): return self.content
    @property
    def topology_version(self): return self.topology

class _Event:
    def __init__(self): self.ready = False
    def query(self): return self.ready

class _Readback:
    def __init__(self, tensors):
        self.event = _Event()
        self._flat = torch.cat([x.reshape(-1) for x in tensors]).clone()
    def numpy(self):
        assert self.event.ready
        return self._flat.numpy()

class _TransferOwner:
    device = torch.device("cpu")
    def __init__(self): self.readbacks = []
    def prepare_output_readback(self, max_bytes): self.prepared = max_bytes
    def readback_packed(self, tensors, *, barrier):
        result = _Readback(tensors); result.barrier = barrier; self.readbacks.append(result)
        return result

def _submit(lane, backend, value, iteration=None):
    backend.content = backend.topology = value
    return lane.try_submit(backend, parameter_version=value,
        iteration=value if iteration is None else iteration)

def test_backend_sampling_is_deterministic_bounded_and_activates_only_selected_rows():
    model = _model(10)
    args = dict(max_gaussians=4, map_content_version=1, topology_version=1, parameter_version=1)
    first, second = sample_output_from_model(model, **args), sample_output_from_model(model, **args)
    assert first.n_total_gaussians == 10
    assert first.positions[:, 0].tolist() == [0.0, 2.0, 5.0, 7.0]
    assert torch.equal(first.positions, second.positions)
    source = inspect.getsource(sample_output_from_model)
    assert "randperm" not in source and "get_scaling" not in source and "get_opacity" not in source

def test_latest_wins_while_one_readback_is_in_flight():
    owner, backend = _TransferOwner(), _Backend(); lane = VersionedMapOutputLane(owner, max_gaussians=4)
    assert _submit(lane, backend, 1, 10); assert lane.poll_completed() is None
    assert _submit(lane, backend, 2, 20) and _submit(lane, backend, 3, 30)
    assert lane.stats().resident_outputs == 2 and lane.stats().coalesced == 1
    owner.readbacks[0].event.ready = True
    assert lane.poll_completed().version == MapOutputVersion(1, 1, 1)
    owner.readbacks[1].event.ready = True
    latest = lane.poll_completed()
    assert latest.version == MapOutputVersion(3, 3, 3) and latest.iteration == 30

def test_duplicate_and_regressed_versions_do_no_output_work():
    owner, backend = _TransferOwner(), _Backend(); lane = VersionedMapOutputLane(owner, max_gaussians=4)
    assert _submit(lane, backend, 2); assert not _submit(lane, backend, 2)
    with pytest.raises(ValueError, match="regressed"): _submit(lane, backend, 1)
    assert len(owner.readbacks) == 0 and lane.stats().duplicates == 1

def test_completed_output_converts_to_existing_native_viser_event():
    owner, backend = _TransferOwner(), _Backend(5); lane = VersionedMapOutputLane(owner, max_gaussians=3)
    backend.content, backend.topology = 7, 4
    lane.try_submit(backend, parameter_version=12, iteration=99); lane.poll_completed(); owner.readbacks[0].event.ready = True
    output = lane.poll_completed(); event = output.to_preview_event(wall_time_s=3.5)
    assert isinstance(event, MapPreviewSnapshot) and event.map_version == 7 and event.n_sampled == 3
    assert output.version == MapOutputVersion(7, 4, 12)

def test_producer_path_has_no_host_sync_renderer_or_read_view():
    source = inspect.getsource(VersionedMapOutputLane.try_submit)
    for forbidden in (".cpu(", ".numpy(", ".item(", "synchronize", "render(", "read_view"):
        assert forbidden not in source

def test_mapper_submission_performs_no_sampling_or_transfer_work():
    owner, backend = _TransferOwner(), _Backend()
    lane = VersionedMapOutputLane(owner, max_gaussians=3)
    assert _submit(lane, backend, 1)
    assert backend.sample_calls == 0 and owner.readbacks == []
    assert lane.poll_completed() is None
    assert backend.sample_calls == 1 and len(owner.readbacks) == 1

def test_stalled_consumer_is_bounded_and_catches_up_to_latest():
    owner, backend = _TransferOwner(), _Backend(5); lane = VersionedMapOutputLane(owner, max_gaussians=2)
    assert _submit(lane, backend, 1); lane.poll_completed()
    for value in range(2, 101): assert _submit(lane, backend, value)
    assert lane.stats().resident_outputs == 2 and lane.stats().coalesced == 98 and len(owner.readbacks) == 1
    owner.readbacks[0].event.ready = True; assert lane.poll_completed().version == MapOutputVersion(1, 1, 1)
    owner.readbacks[1].event.ready = True; assert lane.poll_completed().version == MapOutputVersion(100, 100, 100)

def test_real_transfer_owner_is_capacity_bounded_and_pooled():
    with pytest.raises(RuntimeError, match="pinned byte limit"):
        VersionedMapOutputLane(DeviceTransferOwner("cpu", max_pinned_bytes=1), max_gaussians=3)
    owner = DeviceTransferOwner("cpu", max_inflight=2, max_pinned_bytes=1_000_000)
    lane, backend = VersionedMapOutputLane(owner, max_gaussians=3), _Backend(8)
    assert _submit(lane, backend, 1); assert lane.poll_completed() is None
    output = lane.poll_completed(); assert output.n_sampled == 3
    accounting = owner.accounting()
    assert accounting.d2h_ops == 1 and accounting.d2h_bytes == 3 * 14 * 4
    assert accounting.pinned_bytes == 3 * 14 * 4

def test_producer_lock_contention_drops_instead_of_blocking():
    owner, backend = _TransferOwner(), _Backend(); lane = VersionedMapOutputLane(owner, max_gaussians=3)
    lane._producer_lock.acquire()
    try: assert not lane.try_submit(backend, parameter_version=1, iteration=1)
    finally: lane._producer_lock.release()
    assert lane.stats().dropped == 1

def test_owned_sample_is_stable_across_later_live_map_mutation():
    owner, backend = _TransferOwner(), _Backend(5); lane = VersionedMapOutputLane(owner, max_gaussians=3)
    _submit(lane, backend, 1); lane.poll_completed(); backend.model.params["means"].fill_(999.0); owner.readbacks[0].event.ready = True
    assert lane.poll_completed().positions[:, 0].tolist() == [0.0, 1.0, 3.0]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_real_gpu_output_is_bounded_and_completes_by_event_polling():
    device = torch.device("cuda", torch.cuda.current_device())
    owner = DeviceTransferOwner(device, max_inflight=2, max_pinned_bytes=1_000_000)
    lane, backend = VersionedMapOutputLane(owner, max_gaussians=4), _Backend(32, device)
    assert _submit(lane, backend, 1); assert lane.poll_completed() is None
    # Waiting is permitted on this output consumer thread; try_submit itself
    # remains wait/synchronize-free and mapping never calls this path.
    lane._inflight.readback.event.wait()
    output = lane.poll_completed()
    assert output is not None and output.n_sampled == 4 and owner.accounting().d2h_ops == 1
