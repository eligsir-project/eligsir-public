from types import SimpleNamespace

import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_source import PreparedSourceLease


def _lease(source: SourceIdentity) -> PreparedSourceLease:
    token = object()
    payload = BufferRef(
        data=torch.zeros((1,), dtype=torch.float32),
        spec=BufferSpec(
            shape=(1,),
            dtype="float32",
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding="test",
            device="cpu",
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=token,
    )
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"test": payload},
        device="cpu",
        lifetime_token=token,
    )


def _plan(sources, versions):
    tier = SimpleNamespace(value="native")
    work = tuple(SimpleNamespace(source=source, resolution_tier=tier) for source in sources)
    return SimpleNamespace(
        supervision_work=(),
        replay_work=work,
        supervision_sources=(),
        replay_sources=sources,
        supervision_constellation=SimpleNamespace(realized_batch_size=4, realized_camera_count=4),
        insertion=SimpleNamespace(enabled=False),
        replay_insertion=None,
        map_content_version=versions[0],
        topology_version=versions[1],
        parameter_version=versions[2],
    )


class _Owner:
    def __init__(self, sources, *, reserve_error=None):
        self._leases = tuple(_lease(source) for source in sources)
        self._reserve_error = reserve_error

    def select(self, *_args, **_kwargs):
        return self._leases

    def reserve_return_lanes(self, *_args, **_kwargs):
        if self._reserve_error is not None:
            raise RuntimeError(self._reserve_error)


class _Bridge:
    def __init__(self, owner):
        self.ready_owner = owner
        self.requeues = []

    def requeue_ready_only_sources(self, selection, *, ready_event, lane_by_source=None):
        self.requeues.append((selection, ready_event, lane_by_source))

    def complete_retained_commit_first_supervision(self, _source):
        return True


class _Binding:
    def __init__(self, *, lease, versions, pose_correction=None):
        self.lease = lease
        self.versions = versions
        assert pose_correction is None

    def handoff_to_compute(self, _owner):
        return None


def _adapter(monkeypatch, *, current, planned, reserve_error=None, result=None):
    import eligsir.runtime.ready_b4_streaming_adapter as module

    sources = tuple(SourceIdentity("source", 0, index, index) for index in range(4))
    owner = _Owner(sources, reserve_error=reserve_error)
    bridge = _Bridge(owner)
    runtime = SimpleNamespace(_versions=lambda: current)
    runtime.tick = lambda **_kwargs: (
        result
        if result is not None
        else SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    )
    monkeypatch.setattr(module, "PreparedRuntimeBridge", _Bridge)
    adapter = module.ReadyB4StreamingOptimizationAdapter(
        bridge, runtime=runtime, transfer_owner=object(), warmup_complete=True
    )
    return module, adapter, bridge, _plan(sources, planned)


def test_stale_plan_requeues_without_compute_completion(monkeypatch):
    module, adapter, bridge, plan = _adapter(
        monkeypatch, current=(2, 2, 3), planned=(1, 2, 3)
    )
    monkeypatch.setattr(
        module,
        "_compute_completion",
        lambda: (_ for _ in ()).throw(AssertionError("stale exit allocated completion")),
    )

    outcome = adapter.tick(module.ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is module.ReadyB4StreamingStepState.STALE
    assert bridge.requeues[-1][1] is None


def test_return_lane_failure_requeues_without_compute_completion(monkeypatch):
    module, adapter, bridge, plan = _adapter(
        monkeypatch,
        current=(1, 2, 3),
        planned=(1, 2, 3),
        reserve_error="no return capacity",
    )
    monkeypatch.setattr(
        module,
        "_compute_completion",
        lambda: (_ for _ in ()).throw(AssertionError("capacity exit allocated completion")),
    )

    outcome = adapter.tick(module.ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is module.ReadyB4StreamingStepState.RESERVOIR_CAPACITY_UNAVAILABLE
    assert bridge.requeues[-1][1] is None


def test_compute_entry_still_records_completion(monkeypatch):
    module, adapter, bridge, plan = _adapter(
        monkeypatch, current=(1, 2, 3), planned=(1, 2, 3)
    )
    completion = object()
    monkeypatch.setattr(module, "ReadySourceBinding", _Binding)
    monkeypatch.setattr(
        module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=object(),
        ),
    )
    monkeypatch.setattr(module, "_compute_completion", lambda: completion)

    outcome = adapter.tick(module.ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is module.ReadyB4StreamingStepState.EXECUTED
    assert bridge.requeues[-1][1] is completion
