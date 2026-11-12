"""CPU contracts for the thin canonical READY-B4 adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
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
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import TransferEvent


class _Bridge:
    def __init__(self, owner):
        self.ready_owner = owner
        self.completed = []

    def requeue_ready_only_sources(self, selection, *, ready_event, lane_by_source=None):
        return self.ready_owner.requeue_batch(
            selection.leases,
            ready_event=ready_event,
            lane_by_source=lane_by_source,
        )

    def complete_retained_commit_first_supervision(self, source):
        self.completed.append(source)
        return True


class _Transfer:
    def handoff_to_compute(self, *_args):
        pass


@dataclass(frozen=True)
class _Tier:
    value: str = "native"


def _source(ordinal):
    return SourceIdentity("prepared-test", 0, ordinal, 1_000_000 + ordinal)


def _lease(source, *, tiers=("native",)):
    token = object()
    tensor = torch.zeros((1,), dtype=torch.float32)
    payload = BufferRef(
        data=tensor,
        spec=BufferSpec(
            shape=(1,), dtype="float32", memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding="test", device="cpu",
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=token,
    )
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={},
        tier_preparation_available=tiers,
        borrowed_device_payloads={"test": payload},
        device="cpu",
        lifetime_token=token,
    )


def _plan(
    sources,
    *,
    fresh=(),
    insertion=False,
    versions=(7, 11, 13),
    tier="native",
    batch_size=4,
    requested_batch_size=None,
    camera_count=None,
    requested_camera_count=None,
):
    camera_count = batch_size if camera_count is None else camera_count
    requested_camera_count = (
        batch_size if requested_camera_count is None else requested_camera_count
    )
    work = tuple(
        SimpleNamespace(
            source=source,
            resolution_tier=_Tier(tier),
            modality_eligibility=ModalityEligibility(rgb_supervision=True),
        )
        for source in sources
    )
    return SimpleNamespace(
        supervision_work=tuple(item for item in work if item.source in fresh),
        replay_work=tuple(item for item in work if item.source not in fresh),
        supervision_sources=tuple(fresh),
        replay_sources=tuple(source for source in sources if source not in fresh),
        supervision_constellation=SimpleNamespace(
            realized_batch_size=batch_size,
            realized_camera_count=camera_count,
            requested_batch_size=(
                batch_size if requested_batch_size is None else requested_batch_size
            ),
            requested_camera_count=requested_camera_count,
        ),
        insertion=SimpleNamespace(enabled=insertion),
        replay_insertion=None,
        map_content_version=versions[0],
        topology_version=versions[1],
        parameter_version=versions[2],
    )


def _adapter(
    monkeypatch,
    sources,
    plan,
    *,
    fresh=(),
    result=None,
    database=None,
    available_tiers=None,
    offered_sources=None,
    source_insertion=False,
    trace_ready_state=False,
    timing_scope=None,
    batch_size=4,
):
    import eligsir.runtime.ready_b4_streaming_adapter as module

    monkeypatch.setattr(module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(
        module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=f"camera-{binding.lease.source_identity.source_ordinal}",
        ),
    )
    owner = ReadySourceOwner(
        max_fresh=batch_size, max_replay=max(8, 2 * batch_size), replay_reserve=0
    )
    tiers = tuple(
        sorted(
            {
                item.resolution_tier.value
                for item in tuple(getattr(plan, "supervision_work", ()))
                + tuple(getattr(plan, "replay_work", ()))
            }
        )
    )
    tiers = tiers if available_tiers is None else tuple(available_tiers)
    for source in sources if offered_sources is None else offered_sources:
        assert owner.offer(
            _lease(source, tiers=tiers),
            lane=ReadySourceLane.FRESH if source in fresh else ReadySourceLane.REPLAY,
        )
    runtime = SimpleNamespace(
        _versions=lambda: (7, 11, 13),
        _latest_tick_work=(object(), plan),
        _latest_report_work=None,
        _observation_db=database,
    )
    runtime.tick = lambda *, ready_transaction: (
        result
        if result is not None
        else SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=batch_size)
    )
    bridge = _Bridge(owner)
    adapter_kwargs = dict(
        runtime=runtime,
        transfer_owner=_Transfer(),
        batch_size=batch_size,
        warmup_complete=True,
        source_insertion=source_insertion,
        trace_ready_state=trace_ready_state,
    )
    if timing_scope is not None:
        adapter_kwargs["timing_scope"] = timing_scope
    return module.ReadyB4StreamingOptimizationAdapter(
        bridge,
        **adapter_kwargs,
    ), bridge, runtime


def _canonical_prepared_runtime(
    monkeypatch, plan, *, successful_uid, successor, batch_size=4, realized_batch_size=None
):
    """Run the production B4 tick with a deterministic completed transaction."""

    from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
    from eligsir.runtime import production_application as production
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.telemetry import MapStateTelemetry

    class _Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

    class _Builder:
        def __init__(self, **_kwargs):
            pass

        def add(self, _work):
            pass

        def pop_ready(self, **_kwargs):
            return object()

    monkeypatch.setattr(
        production, "torch", SimpleNamespace(cuda=SimpleNamespace(Event=_Event))
    )
    monkeypatch.setattr(production, "FixedTopologyBatchBuilder", _Builder, raising=False)
    monkeypatch.setattr(
        production,
        "GpuWorkEnvelope",
        lambda **values: SimpleNamespace(**values),
        raising=False,
    )
    monkeypatch.setattr(
        production,
        "BatchExecutionMode",
        SimpleNamespace(TRUE_CAMERA_BATCH="b4", SEQUENTIAL_ACCUMULATION="b1"),
        raising=False,
    )
    realized_batch_size = batch_size if realized_batch_size is None else realized_batch_size
    monkeypatch.setattr(
        production,
        "execute_fixed_topology_transaction",
        lambda _transaction, **_kwargs: SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=realized_batch_size,
            supervised_pixel_count=realized_batch_size,
            depth_valid_sample_count=0,
            rasterizer_call_count=1,
            microbatch_sizes=(realized_batch_size,),
            successful_supervision_source_indices=(successful_uid,),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        production,
        "GpuTelemetryDescriptor",
        lambda **values: SimpleNamespace(**values),
        raising=False,
    )
    monkeypatch.setattr(
        production,
        "TransactionType",
        SimpleNamespace(OPTIMIZATION="optimization"),
        raising=False,
    )
    monkeypatch.setattr(production, "MapStateTelemetry", MapStateTelemetry, raising=False)

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._closed = False
    runtime._control_tick_epoch = 0
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime._last_materialized_work = None
    runtime.initialized = True
    runtime.iteration = 0
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._batch_size = batch_size
    runtime._residency_ledger = SimpleNamespace(release=lambda _camera_ids: None)
    runtime.scene = SimpleNamespace(
        hot_camera_ids=lambda: (), retire_live_admitted_payloads=lambda: None
    )
    runtime._assert_plan_version_fence = lambda _plan: None
    runtime._runtime_policy = SimpleNamespace(
        random_background=False, sh_update_interval=1
    )
    runtime._background = object()
    runtime._depth_loss_policy = SimpleNamespace(enabled=False)
    runtime.gaussians = SimpleNamespace(
        get_xyz=torch.zeros((1, 3)), active_sh_degree=0
    )
    runtime._transfer_owner = SimpleNamespace(accounting=lambda: None)
    runtime._versions = lambda: (7, 11, 13)
    runtime.backend = SimpleNamespace(map_version=7, topology_version=11)
    runtime._engine = SimpleNamespace(session=object())
    runtime.pipe = object()
    runtime._optimization_runtime = None
    runtime.frames_seen = 4
    runtime.config = SimpleNamespace(
        static_options={
            "population_live_enabled": False,
            "population_shadow_enabled": False,
        }
    )
    runtime._begin_optimization_gpu_envelope = lambda *_args, **_kwargs: 0
    runtime._stamp_completed_result_map_version_after = lambda _result: None
    runtime._observe_population_policy_inputs_after_step = lambda *_args: None
    runtime._maybe_raise_sh_degree = lambda **_kwargs: None
    runtime._run_due_lifecycle = lambda *_args, **_kwargs: None
    runtime._observe_replay_evidence = lambda *_args: None
    runtime._post_step_structural_mutation = lambda *_args, **_kwargs: (False, None)
    runtime._record_successful_replay_supervision = lambda *_args, **_kwargs: None
    runtime._sample_primary_training_psnr = lambda *_args: None
    runtime._sample_primary_training_spatial = lambda *_args: None
    runtime._submit_gpu_timing = lambda *_args, **_kwargs: None
    runtime._next_gpu_transaction_id = 0
    runtime._services = SimpleNamespace(poll_map_activity=lambda: None)
    runtime._source_arrival_ns = {}
    runtime._first_supervision_latency_ms = []
    runtime._geometry_commit_latency_ms = []
    runtime._map_clock_ns = 0
    runtime._retire_completed_candidates = lambda *_args: None
    runtime._record_transaction_cost = lambda *_args, **_kwargs: None
    runtime._poll_gpu_telemetry = lambda: None
    runtime._drain_population_selection_telemetry = lambda: ()
    runtime._dispositions_since_tick = deque(maxlen=8)
    runtime._replay_modalities = {}
    runtime._latest_gpu_transaction = None
    runtime.native_geometry_evidence_telemetry = lambda: {}
    runtime._telemetry = SimpleNamespace(
        publish_tick=lambda *_args, **_kwargs: SimpleNamespace(
            map_state=MapStateTelemetry(1, 7, 11, 13)
        )
    )
    runtime._telemetry_inputs = lambda _snapshot: object()
    snapshot = SimpleNamespace(snapshot_id=plan.snapshot_id)
    runtime._prepared_ready_frozen_work = (snapshot, plan)
    runtime._prepared_ready_candidate_work = (snapshot, plan)
    runtime._fresh_idle_snapshot = lambda value: value
    runtime._idle_replay_plan = lambda _snapshot, prior: successor if prior is plan else None
    return runtime


@pytest.mark.parametrize("batch_size", [4, 16, 32])
def test_exact_work_plan_order_enters_canonical_tick(monkeypatch, batch_size):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in reversed(range(batch_size)))
    plan = _plan(sources, batch_size=batch_size)
    snapshot = object()
    adapter, _bridge, runtime = _adapter(
        monkeypatch, sources, plan, batch_size=batch_size
    )
    captured = []
    runtime.tick = lambda *, ready_transaction: (
        captured.append(ready_transaction)
        or SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=batch_size)
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert tuple(captured[0].cameras) == sources
    assert captured[0].plan is plan
    assert captured[0].snapshot is snapshot
    assert runtime._latest_tick_work is None


def test_configured_b32_adapter_executes_exact_authorized_b4_source_service(monkeypatch):
    """Configured B32 must account for the WorkBudget-authorized B4 exactly."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    sources = (fresh, _source(1), _source(2), _source(3))
    plan = _plan(
        sources,
        fresh=(fresh,),
        insertion=True,
        batch_size=4,
        requested_batch_size=32,
    )
    result = SimpleNamespace(
        executed=True,
        optimizer_steps=1,
        supervision_views=4,
        successful_supervision_sources=sources,
    )
    adapter, bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=(fresh,),
        result=result,
        source_insertion=True,
        batch_size=32,
    )
    captured = []
    runtime.tick = lambda *, ready_transaction: captured.append(ready_transaction) or result

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert tuple(captured[0].cameras) == sources
    assert len(outcome.selection.leases) == 4
    assert outcome.selection.fresh_count == 1
    assert outcome.selection.replay_count == 3
    assert not outcome.selection.starved
    assert bridge.completed == [fresh]


@pytest.mark.parametrize("camera_count", (1, 4))
def test_configured_b16_adapter_executes_full_b16_with_existing_camera_microbatches(
    monkeypatch, camera_count
):
    """A full B16 plan reaches the existing C1/C4 execution boundary."""
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(16))
    plan = _plan(
        sources,
        batch_size=16,
        camera_count=camera_count,
        requested_batch_size=16,
        requested_camera_count=camera_count,
    )
    result = SimpleNamespace(
        executed=True,
        optimizer_steps=1,
        supervision_views=16,
        microbatch_sizes=(1,) * 16 if camera_count == 1 else (4,) * 4,
    )
    adapter, _bridge, runtime = _adapter(
        monkeypatch, sources, plan, result=result, batch_size=16
    )
    captured = []
    runtime.tick = lambda *, ready_transaction: captured.append(ready_transaction) or result

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert len(outcome.selection.leases) == 16
    assert tuple(captured[0].cameras) == sources


def test_configured_b16_adapter_executes_fresh_b4_source_service_with_c1(monkeypatch):
    """The narrow fresh B4 service also reaches the existing C1 executor path."""
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    sources = (fresh, *tuple(_source(index) for index in range(1, 4)))
    plan = _plan(
        sources,
        fresh=(fresh,),
        insertion=True,
        batch_size=4,
        camera_count=1,
        requested_batch_size=16,
        requested_camera_count=1,
    )
    result = SimpleNamespace(
        executed=True,
        optimizer_steps=1,
        supervision_views=4,
        successful_supervision_sources=sources,
    )
    adapter, bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=(fresh,),
        result=result,
        source_insertion=True,
        batch_size=16,
    )
    captured = []
    runtime.tick = lambda *, ready_transaction: captured.append(ready_transaction) or result

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert len(outcome.selection.leases) == 4
    assert outcome.selection.fresh_count == 1
    assert outcome.selection.replay_count == 3
    assert bridge.completed == [fresh]


def test_configured_b32_adapter_completes_fresh_from_actual_b4_view_count(monkeypatch):
    """Legacy results without source IDs still use the realized B4 count."""

    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    fresh = _source(0)
    sources = (fresh, _source(1), _source(2), _source(3))
    plan = _plan(
        sources,
        fresh=(fresh,),
        insertion=True,
        batch_size=4,
        requested_batch_size=32,
    )
    result = SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    adapter, bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=(fresh,),
        result=result,
        source_insertion=True,
        batch_size=32,
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.executed
    assert outcome.completed_supervision_sources == sources
    assert bridge.completed == [fresh]


@pytest.mark.parametrize("realized_batch_size", (1, 2, 3))
def test_source_insertion_rejects_unqualified_shortened_service(monkeypatch, realized_batch_size):
    """A missing replay cannot turn a source-depth transaction into B1/B2/B3."""

    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(realized_batch_size))
    plan = _plan(
        sources,
        fresh=sources[:1],
        insertion=True,
        batch_size=realized_batch_size,
        requested_batch_size=32,
    )
    adapter, bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=sources[:1],
        source_insertion=True,
        batch_size=32,
    )
    runtime.tick = lambda **_kwargs: pytest.fail("shortened source-depth plan reached compute")

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert not outcome.executed
    assert outcome.reason == "prepared_authorized_plan_is_not_exact_qualified_batch"
    assert bridge.completed == []


@pytest.mark.parametrize("field", ("realized_batch_size", "realized_camera_count"))
@pytest.mark.parametrize("invalid", (4.0, 4.9, "4", True))
def test_source_insertion_rejects_noninteger_realized_constellation(monkeypatch, field, invalid):
    """READY must not coerce malformed B/C into an executable source-depth batch."""

    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(
        sources,
        fresh=sources[:1],
        insertion=True,
        batch_size=4,
        requested_batch_size=32,
    )
    setattr(plan.supervision_constellation, field, invalid)
    adapter, bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=sources[:1],
        source_insertion=True,
        batch_size=32,
    )
    runtime.tick = lambda **_kwargs: pytest.fail("invalid constellation reached compute")

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert not outcome.executed
    assert outcome.reason == "prepared_authorized_plan_is_not_exact_qualified_batch"
    assert bridge.completed == []


@pytest.mark.parametrize("camera_count", (0, 3, 17))
def test_adapter_rejects_unqualified_or_overlarge_camera_count(monkeypatch, camera_count):
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(16))
    plan = _plan(
        sources,
        batch_size=16,
        camera_count=camera_count,
        requested_batch_size=16,
        requested_camera_count=camera_count,
    )
    adapter, bridge, runtime = _adapter(
        monkeypatch, sources, plan, batch_size=16
    )
    runtime.tick = lambda **_kwargs: pytest.fail("invalid B/C reached compute")

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert not outcome.executed
    assert outcome.reason == "prepared_authorized_plan_is_not_exact_qualified_batch"
    assert bridge.completed == []


@pytest.mark.parametrize(
    ("microbatch_count", "microbatch_sizes"),
    [(5, (4, 4, 4, 4)), (4, (4, 4, 4, 4, 1))],
)
def test_adapter_rejects_wrong_b16_c4_microbatch_partition(
    monkeypatch, microbatch_count, microbatch_sizes
):
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(16))
    plan = _plan(
        sources,
        batch_size=16,
        camera_count=4,
        requested_batch_size=16,
        requested_camera_count=4,
    )
    plan.supervision_constellation.microbatch_count = microbatch_count
    plan.supervision_constellation.microbatch_sizes = microbatch_sizes
    adapter, bridge, runtime = _adapter(
        monkeypatch, sources, plan, batch_size=16
    )
    runtime.tick = lambda **_kwargs: pytest.fail("invalid microbatch partition reached compute")

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert not outcome.executed
    assert outcome.reason == "prepared_authorized_plan_is_not_exact_qualified_batch"
    assert bridge.completed == []


def test_successful_fresh_b4_evicts_idle_replay_before_return_reservation(monkeypatch):
    """A first-supervised fresh source needs one replay-cache slot to return."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    replay = tuple(_source(index) for index in range(1, 4))
    sources = (fresh, *replay)
    plan = _plan(sources, fresh=(fresh,))
    adapter, bridge, _runtime = _adapter(monkeypatch, sources, plan, fresh=(fresh,))
    for ordinal in range(4, 9):
        assert bridge.ready_owner.offer(_lease(_source(ordinal)), lane=ReadySourceLane.REPLAY)

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert bridge.ready_owner.counters.replay_pending == 8


def test_successful_b4_commits_only_the_minimum_required_replay_eviction(monkeypatch):
    """A committed transaction retires exactly the one evicted bystander source."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    replay = tuple(_source(index) for index in range(1, 4))
    sources = (fresh, *replay)
    plan = _plan(sources, fresh=(fresh,))
    adapter, bridge, _runtime = _adapter(monkeypatch, sources, plan, fresh=(fresh,))
    extra = tuple(_source(ordinal) for ordinal in range(4, 9))
    for source in extra:
        assert bridge.ready_owner.offer(_lease(source), lane=ReadySourceLane.REPLAY)

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    records = bridge.ready_owner.terminal_records
    assert len(records) == 1
    assert records[0].source == extra[0]
    assert records[0].reason == "evicted_replay_residency"
    remaining = {lease.source_identity for lease in bridge.ready_owner.pending_leases}
    assert remaining == {fresh, *replay, *extra[1:]}
    assert bridge.ready_owner.counters.replay_pending == 8


def test_nonexecuted_b4_with_required_eviction_restores_exact_pre_attempt_state(monkeypatch):
    """P0-A: a no-execute transaction must not permanently shrink replay residency."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    replay = tuple(_source(index) for index in range(1, 4))
    sources = (fresh, *replay)
    plan = _plan(sources, fresh=(fresh,))
    adapter, bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=(fresh,),
        result=SimpleNamespace(executed=False, reason="canonical_deferred"),
    )
    for ordinal in range(4, 9):
        assert bridge.ready_owner.offer(_lease(_source(ordinal)), lane=ReadySourceLane.REPLAY)

    before = bridge.ready_owner.pending_leases
    before_counters = bridge.ready_owner.counters

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    assert outcome.state is ReadyB4StreamingStepState.DEFERRED
    # The eviction reserved to make return-lane room for this attempt must be
    # rolled back exactly -- same membership, same order, same capacity.
    assert bridge.ready_owner.pending_leases == before
    assert bridge.ready_owner.counters == before_counters
    assert all(lease.ready_event is None for lease in bridge.ready_owner.pending_leases)
    assert bridge.ready_owner.terminal_records == ()


def test_nonexecuted_b4_repeated_eviction_rollback_has_no_residency_drift(monkeypatch):
    """P0-A: 1,000 repeated no-execute attempts must not drift READY residency."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    fresh = _source(0)
    replay = tuple(_source(index) for index in range(1, 4))
    sources = (fresh, *replay)
    plan = _plan(sources, fresh=(fresh,))
    adapter, bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=(fresh,),
        result=SimpleNamespace(executed=False, reason="canonical_deferred"),
    )
    for ordinal in range(4, 9):
        assert bridge.ready_owner.offer(_lease(_source(ordinal)), lane=ReadySourceLane.REPLAY)

    before = bridge.ready_owner.pending_leases
    before_counters = bridge.ready_owner.counters

    for _ in range(1000):
        outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))
        assert outcome.state is ReadyB4StreamingStepState.DEFERRED
        assert bridge.ready_owner.pending_leases == before
        assert bridge.ready_owner.counters == before_counters

    # No source was ever permanently evicted, so nothing could have gone
    # COLD -- a later scheduler page-in would never be demanded by this loop.
    assert bridge.ready_owner.terminal_records == ()


def test_exception_after_dispatch_keeps_selection_fenced_and_restores_eviction(monkeypatch):
    """P0-B: an exception past dispatch cannot prove no device consumption occurred."""

    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    fresh = _source(0)
    replay = tuple(_source(index) for index in range(1, 4))
    sources = (fresh, *replay)
    plan = _plan(sources, fresh=(fresh,))
    adapter, bridge, runtime = _adapter(monkeypatch, sources, plan, fresh=(fresh,))
    extra = tuple(_source(ordinal) for ordinal in range(4, 9))
    for source in extra:
        assert bridge.ready_owner.offer(_lease(source), lane=ReadySourceLane.REPLAY)

    class _SimulatedDeviceFailure(RuntimeError):
        pass

    def _raise(**_kwargs):
        raise _SimulatedDeviceFailure("simulated device failure after submission")

    runtime.tick = _raise

    with pytest.raises(_SimulatedDeviceFailure):
        adapter.tick(ReadyB4StreamingStepInputs(object(), plan))

    # The exception is not a committed transaction: the bystander eviction
    # reserved for return-lane capacity is rolled back, not retired.
    assert bridge.ready_owner.terminal_records == ()
    remaining = {lease.source_identity for lease in bridge.ready_owner.pending_leases}
    assert remaining == {fresh, *replay, *extra}

    # But the selected payloads cannot be proven un-consumed by the device,
    # so they come back leased behind a completion fence rather than
    # immediately reusable.
    fenced = {
        lease.source_identity: lease.ready_event
        for lease in bridge.ready_owner.pending_leases
        if lease.source_identity in {fresh, *replay}
    }
    assert len(fenced) == 4
    assert all(event is not None for event in fenced.values())


def test_homogeneous_half_work_plan_binds_the_exact_requested_tier(monkeypatch):
    from eligsir.runtime import ready_b4_streaming_adapter as module
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources, tier="half")
    requested_tiers = []
    adapter, _bridge, _runtime = _adapter(monkeypatch, sources, plan)
    monkeypatch.setattr(
        module,
        "bind_ready_camera",
        lambda binding, *, tier: (
            requested_tiers.append(tier)
            or SimpleNamespace(source_identity=binding.lease.source_identity, camera=object())
        ),
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.executed
    assert requested_tiers == ["half"] * 4


def test_requested_tier_unavailable_fails_closed_without_native_substitution(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources, tier="quarter")
    adapter, _bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        available_tiers=("native",),
    )
    runtime.tick = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not substitute native"))

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE
    assert outcome.reason == "ready_b4_requested_tier_unavailable"


def test_mixed_tier_b4_fails_closed(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    plan.replay_work[0].resolution_tier = _Tier("half")
    adapter, _bridge, runtime = _adapter(monkeypatch, sources, plan)
    runtime.tick = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch mixed tiers"))

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE
    assert outcome.reason == "ready_b4_requires_one_homogeneous_prepared_tier"


def test_missing_authorized_source_never_substitutes_ready_peer(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    expected = tuple(_source(index) for index in range(4))
    ready = expected[:3] + (_source(99),)
    plan = _plan(expected)
    adapter, _bridge, runtime = _adapter(monkeypatch, ready, plan)
    runtime.tick = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must defer"))

    before = _bridge.ready_owner.pending_leases
    counters = _bridge.ready_owner.counters

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.STARVED
    assert outcome.reason == "prepared_authorized_sources_not_ready"
    assert outcome.retryable
    assert _bridge.ready_owner.pending_leases == before
    assert _bridge.ready_owner.counters == counters


def test_trace_ready_state_explains_exact_b4_absence(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    adapter, _bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        offered_sources=sources[:3],
        trace_ready_state=True,
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.STARVED
    assert outcome.ready_count == 3
    assert outcome.ready_counters == _bridge.ready_owner.counters
    # Check that reason string reports owner-latched READY counts, not occupancy
    assert ";ready_post_select=f0,r3;" in outcome.reason
    assert "lane_occupancy=f0,r3,i0,nonresident=0" in outcome.reason
    assert outcome.reason.startswith(
        "prepared_authorized_sources_not_ready:"
        f"{sources[0].fingerprint()}=READY,"
        f"{sources[1].fingerprint()}=READY,"
        f"{sources[2].fingerprint()}=READY,"
        f"{sources[3].fingerprint()}=ABSENT_REPLAY"
    )


def test_trace_ready_state_distinguishes_occupancy_from_ready(monkeypatch):
    """Verify STARVED reason reports READY counts, not occupancy."""

    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)

    # Create adapter with 3 offered sources
    adapter, bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        offered_sources=sources[:3],
        trace_ready_state=True,
    )

    # Manually add a pending (not ready) replica to one of the offered sources
    # by requeueing it with an un-fired event
    from eligsir.runtime.ready_source import ReadySourceLane

    class PendingEvent:
        def query(self) -> bool:
            return False

    producer = bridge.ready_owner.claim_producer()
    # Add a 4th pending source with un-fired event
    pending_lease = _lease(sources[3], tiers=("native",))
    pending_lease_unfired = replace(pending_lease, ready_event=PendingEvent())
    producer.offer(pending_lease_unfired, lane=ReadySourceLane.REPLAY)

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.STARVED
    # 3 ready (sources 0-2) + 0 pending ready (source 3 has un-fired event)
    assert outcome.ready_count == 3
    # Verify counters distinguish occupancy from ready
    counters = outcome.ready_counters
    assert counters.replay_pending == 4, f"Expected 4 pending (sources 0-3), got {counters.replay_pending}"
    assert counters.replay_ready_count == 3, f"Expected 3 ready (sources 0-2), got {counters.replay_ready_count}"
    # Verify reason string reports the READY count, not occupancy
    assert ";ready_post_select=f0,r3;" in outcome.reason
    assert "lane_occupancy=f0,r4" in outcome.reason


def test_nonexecuted_exact_b4_restores_ready_without_a_completion_fence(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    adapter, bridge, _runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        result=SimpleNamespace(executed=False, reason="canonical_deferred"),
    )
    before = bridge.ready_owner.pending_leases

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.DEFERRED
    assert bridge.ready_owner.pending_leases == before
    assert all(lease.ready_event is None for lease in bridge.ready_owner.pending_leases)


def test_retired_authorized_source_invalidates_frozen_plan(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    expected = tuple(_source(index) for index in range(4))
    plan = _plan(expected)
    database = SimpleNamespace(
        frame_id_for_source=lambda source: None if source == expected[-1] else source.source_ordinal
    )
    adapter, _bridge, runtime = _adapter(
        monkeypatch, expected[:3], plan, database=database
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.DEFERRED
    assert outcome.reason == "prepared_authorized_source_retired"
    assert runtime._latest_tick_work is None


def test_depth_insertion_fails_before_gpu_dispatch(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources, insertion=True)
    adapter, _bridge, runtime = _adapter(monkeypatch, sources, plan)
    runtime.tick = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch"))

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE
    assert outcome.reason == "prepared_depth_insertion_requires_prepared_mapping_frame"


def test_opted_in_source_insertion_dispatches_but_replay_insertion_stays_refused(monkeypatch):
    """READY permits its one source-depth job only through the explicit opt-in."""
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources, fresh=sources[:1], insertion=True)
    adapter, _bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=sources[:1],
        source_insertion=True,
    )
    dispatched = []
    runtime.tick = lambda *, ready_transaction: dispatched.append(ready_transaction) or (
        SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.state is ReadyB4StreamingStepState.EXECUTED
    assert len(dispatched) == 1
    assert dispatched[0].source_insertion is True

    replay_plan = _plan(sources, fresh=sources[:1])
    replay_plan.replay_insertion = object()
    adapter, _bridge, runtime = _adapter(
        monkeypatch,
        sources,
        replay_plan,
        fresh=sources[:1],
        source_insertion=True,
    )
    runtime.tick = lambda **_kwargs: pytest.fail("replay insertion must not dispatch")

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, replay_plan))

    assert outcome.state is ReadyB4StreamingStepState.CAPABILITY_UNAVAILABLE
    assert outcome.reason == "prepared_replay_depth_insertion_unsupported"


@pytest.mark.parametrize("batch_size", [16, 32])
def test_opted_in_source_insertion_preserves_one_newest_fresh_source_at_ready_bn(
    monkeypatch, batch_size
):
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(batch_size))
    plan = _plan(
        sources, fresh=sources[:1], insertion=True, batch_size=batch_size
    )
    dispatched = []
    adapter, _bridge, runtime = _adapter(
        monkeypatch,
        sources,
        plan,
        fresh=sources[:1],
        source_insertion=True,
        batch_size=batch_size,
    )
    runtime.tick = lambda *, ready_transaction: dispatched.append(ready_transaction) or (
        SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=batch_size)
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.executed and len(dispatched) == 1
    assert dispatched[0].source_insertion is True
    assert dispatched[0].bindings[0].source_identity == sources[0]


def _source_insertion_runtime(monkeypatch, *, render_reuse, configured_batch_size=4):
    """Run one fresh B4 plan through the real tick and its mutation boundary."""
    from eligsir.contracts.control import FreshnessMode
    from eligsir.mapping.integration.depth_render_reuse import DepthInsertionReuseReason
    from eligsir.runtime import production_application as production
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingOptimizationAdapter
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

    fresh = (
        SourceIdentity("prepared-a", 3, 17, 1_000_017),
        SourceIdentity("prepared-b", 3, 18, 1_000_018),
    )
    sources = (*fresh, _source(19), _source(20))

    def make_plan(*, insertion, versions, fresh_sources=fresh):
        made = _plan(
            sources,
            fresh=fresh_sources,
            insertion=insertion,
            versions=versions,
            requested_batch_size=configured_batch_size,
        )
        made.insertion.source = fresh[-1]
        made.optimizer_updates = 1
        made.snapshot_id = 3
        made.freshness_mode = FreshnessMode.HEALTHY
        return made

    plan = make_plan(insertion=True, versions=(7, 11, 13))
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=SimpleNamespace(uid=_full_identity_camera_uid(binding.lease.source_identity)),
            metadata=SimpleNamespace(
                intrinsics=SimpleNamespace(
                    fx=8.0, fy=9.0, cx=2.0, cy=1.5,
                    depth_fx=8.0, depth_fy=9.0, depth_cx=2.0, depth_cy=1.5,
                    depth_height=4, depth_width=5,
                ),
                pose_c2w=tuple(
                    tuple(float(row == column) for column in range(4))
                    for row in range(4)
                ),
                depth_shape=(4, 5),
            ),
            lease=binding.lease,
        ),
    )
    owner = ReadySourceOwner(
        max_fresh=configured_batch_size,
        max_replay=2 * configured_batch_size,
        replay_reserve=0,
    )
    for source in sources:
        assert owner.offer(
            _lease(source),
            lane=ReadySourceLane.FRESH if source in fresh else ReadySourceLane.REPLAY,
        )
    runtime = _canonical_prepared_runtime(
        monkeypatch,
        plan,
        successful_uid=_full_identity_camera_uid(fresh[-1]),
        successor=None,
        batch_size=configured_batch_size,
        realized_batch_size=4,
    )
    runtime._versions = lambda: (7, int(runtime.backend.topology_version), 13)
    runtime._assert_plan_version_fence = CanonicalMappingRuntime._assert_plan_version_fence.__get__(runtime)
    runtime._idle_replay_plan = lambda _snapshot, prior: (
        make_plan(insertion=False, versions=runtime._versions(), fresh_sources=fresh[1:])
        if prior is plan
        else None
    )
    for name in ("MAP_MUTATION", "INSERTION_PREP"):
        setattr(production.TransactionType, name, name.lower())
    calls = SimpleNamespace(select=[], prepare=[], commit=[])

    def select(**kwargs):
        calls.select.append(kwargs)
        reason = (
            DepthInsertionReuseReason.REUSED
            if render_reuse is not None
            else DepthInsertionReuseReason.MISSING_RENDER_LEASE
        )
        return render_reuse, reason

    def prepare(camera, frame, source, plan, **kwargs):
        calls.prepare.append((camera, frame, source, plan, kwargs))
        return SimpleNamespace(batch=SimpleNamespace(points=torch.zeros((3, 3))))

    def commit(proposals, *, expected_map_content_version, expected_topology_version):
        calls.commit.append((proposals, expected_map_content_version, expected_topology_version))
        if not proposals:
            return None
        runtime.backend.topology_version += 1
        return SimpleNamespace(
            inserted=3,
            applied=3,
            change=SimpleNamespace(pruned=0, relocated=0, corrected=0, topology_version=runtime.backend.topology_version),
        )

    runtime._select_depth_insertion_render_reuse = select
    runtime._prepare_insertion_proposal = prepare
    runtime._commit_mutation_barrier = commit
    runtime._record_completed_depth_mutation = lambda _commit: None
    monkeypatch.setattr(
        production,
        "run_streaming_optimization_step",
        lambda **_kwargs: pytest.fail("B4 source insertion must use true-camera execution"),
        raising=False,
    )
    return (
        ReadyB4StreamingOptimizationAdapter(
            _Bridge(owner),
            runtime=runtime,
            transfer_owner=_Transfer(),
            batch_size=configured_batch_size,
            warmup_complete=True,
            source_insertion=True,
        ),
        runtime,
        plan,
        fresh,
        calls,
    )


def test_ready_source_insertion_emits_one_live_depth_event_from_the_newest_fresh_source(monkeypatch):
    """The planned newest fresh source alone may mutate after its own B4 render."""
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs
    from eligsir.runtime.ready_insertion_frame import ReadyInsertionFrame

    reuse = object()
    adapter, runtime, plan, fresh, calls = _source_insertion_runtime(monkeypatch, render_reuse=reuse)
    outcome = adapter.tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert len(calls.select) == len(calls.prepare) == 1
    assert calls.select[0]["source"] == fresh[-1]
    assert calls.select[0]["source_lease"].source_identity == fresh[-1]
    _camera, frame, source, seen_plan, kwargs = calls.prepare[0]
    assert isinstance(frame, ReadyInsertionFrame)
    assert frame.mapping_frame.source == source == fresh[-1]
    assert frame._depth_np is None
    assert seen_plan is plan and kwargs["render_reuse"] is reuse
    assert [item[2] for item in calls.commit] == [11]
    events = [event for event in outcome.result.mutation_events if event.kind == "live_depth"]
    assert len(events) == 1 and events[0].source == fresh[-1] and events[0].inserted == 3
    assert outcome.result.prepared_insertion_skip is None


def test_configured_b32_source_insertion_binds_the_authorized_b4_lease(monkeypatch):
    """The adapter forwards the source's authorized B4 lease."""

    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    reuse = object()
    adapter, runtime, plan, fresh, calls = _source_insertion_runtime(
        monkeypatch, render_reuse=reuse, configured_batch_size=32
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert outcome.executed
    assert len(outcome.selection.leases) == 4
    assert calls.select[0]["source_lease"].source_identity == fresh[-1]
    assert calls.prepare[0][4]["render_reuse"] is reuse


def test_ready_source_insertion_missing_lease_skips_without_zero_seed_event(monkeypatch):
    """No lease is a refused insertion, distinct from a real zero-seed event."""
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    adapter, runtime, plan, _fresh, calls = _source_insertion_runtime(monkeypatch, render_reuse=None)
    outcome = adapter.tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert len(calls.select) == 1 and calls.prepare == []
    assert runtime.backend.topology_version == 11
    assert outcome.result.inserted == 0
    assert outcome.result.prepared_insertion_skip == "render_reuse_unavailable:missing_render_lease"
    assert not [event for event in outcome.result.mutation_events if event.kind == "live_depth"]


def test_ready_source_insertion_reused_zero_seed_is_an_executed_live_depth_event(monkeypatch):
    """A valid insertion that selects no seeds is observable, unlike a refusal."""
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    adapter, runtime, plan, fresh, calls = _source_insertion_runtime(
        monkeypatch, render_reuse=object()
    )
    runtime._prepare_insertion_proposal = lambda *_args, **_kwargs: None
    outcome = adapter.tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert len(calls.select) == 1
    assert outcome.result.prepared_insertion_skip is None
    events = [event for event in outcome.result.mutation_events if event.kind == "live_depth"]
    assert len(events) == 1 and events[0].source == fresh[-1] and events[0].inserted == 0


@pytest.mark.parametrize(
    ("configured_batch_size", "realized_batch_size"),
    ((4, 4), (16, 16), (32, 32), (32, 4)),
)
def test_ready_camera_materialize_cache_hits_equal_realized_batch(
    monkeypatch, configured_batch_size, realized_batch_size
):
    """READY cache attribution counts the exact scheduled transaction, not B4."""
    from eligsir.contracts.control import FreshnessMode
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

    sources = tuple(_source(index) for index in range(realized_batch_size))
    plan = _plan(
        sources,
        fresh=sources,
        batch_size=realized_batch_size,
        requested_batch_size=configured_batch_size,
    )
    plan.optimizer_updates = 1
    plan.snapshot_id = 3
    plan.freshness_mode = FreshnessMode.HEALTHY
    runtime = _canonical_prepared_runtime(
        monkeypatch,
        plan,
        successful_uid=_full_identity_camera_uid(sources[-1]),
        successor=None,
        batch_size=configured_batch_size,
        realized_batch_size=realized_batch_size,
    )
    runtime._assert_plan_version_fence = lambda _plan: None
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(adapter_module, "bind_ready_camera", lambda binding, *, tier="native": SimpleNamespace(
        source_identity=binding.lease.source_identity,
        camera=SimpleNamespace(uid=_full_identity_camera_uid(binding.lease.source_identity)),
    ))
    owner = ReadySourceOwner(
        max_fresh=configured_batch_size, max_replay=2 * configured_batch_size
    )
    for source in sources:
        assert owner.offer(_lease(source), lane=ReadySourceLane.FRESH)
    outcome = ReadyB4StreamingOptimizationAdapter(
        _Bridge(owner), runtime=runtime, transfer_owner=_Transfer(), batch_size=configured_batch_size,
        warmup_complete=True,
    ).tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert outcome.executed
    assert outcome.result.camera_materialize_cache_hits == realized_batch_size


@pytest.mark.parametrize("batch_size", [1, 16])
def test_canonical_prepared_tick_records_prepare_endpoint_before_execution(
    monkeypatch, batch_size
):
    """The cost sample's prepare segment ends when real execution begins."""
    from eligsir.contracts.control import FreshnessMode
    from eligsir.runtime import production_application as production
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

    sources = tuple(_source(index) for index in range(batch_size))
    plan = _plan(sources, fresh=sources, batch_size=batch_size)
    plan.optimizer_updates = 1
    plan.snapshot_id = 3
    plan.freshness_mode = FreshnessMode.HEALTHY
    runtime = _canonical_prepared_runtime(
        monkeypatch,
        plan,
        successful_uid=_full_identity_camera_uid(sources[-1]),
        successor=None,
        batch_size=batch_size,
    )
    runtime._depth_loss_policy = SimpleNamespace(enabled=False, weight=0.0)
    runtime._begin_optimization_gpu_envelope = (
        CanonicalMappingRuntime._begin_optimization_gpu_envelope
    )
    recorded = {}
    runtime._record_transaction_cost = lambda _plan, **kwargs: recorded.update(kwargs)
    clock = {"now": 0}
    execution = {}

    def monotonic_ns():
        clock["now"] += 5_000_000
        return clock["now"]

    monkeypatch.setattr(production.time, "monotonic_ns", monotonic_ns)
    start_event_times = []

    class _Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            start_event_times.append(clock["now"])

    production.torch.cuda.Event = _Event

    def completed_transaction(_transaction, **_kwargs):
        execution["started_ns"] = clock["now"]
        clock["now"] += 50_000_000
        execution["completed_ns"] = clock["now"]
        return SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=batch_size,
            supervised_pixel_count=batch_size,
            depth_valid_sample_count=0,
            rasterizer_call_count=1,
            microbatch_sizes=(batch_size,),
            successful_supervision_source_indices=(
                _full_identity_camera_uid(sources[-1]),
            ),
        )

    monkeypatch.setattr(
        production, "execute_fixed_topology_transaction", completed_transaction
    )
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=SimpleNamespace(
                uid=_full_identity_camera_uid(binding.lease.source_identity)
            ),
        ),
    )
    owner = ReadySourceOwner(max_fresh=batch_size, max_replay=2 * batch_size)
    for source in sources:
        assert owner.offer(_lease(source), lane=ReadySourceLane.FRESH)

    outcome = ReadyB4StreamingOptimizationAdapter(
        _Bridge(owner), runtime=runtime, transfer_owner=_Transfer(),
        batch_size=batch_size, warmup_complete=True,
    ).tick(ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan))

    assert outcome.executed
    assert outcome.result.optimizer_steps == 1
    assert outcome.result.supervision_views == batch_size
    assert recorded["prepare_end_ns"] == start_event_times[0]
    assert recorded["prepare_end_ns"] <= execution["started_ns"]
    assert execution["started_ns"] < execution["completed_ns"]
    assert execution["completed_ns"] <= recorded["optimization_end_ns"]
    assert runtime._work_clock_owner.snapshot().mapping_active_ns > 0
    assert runtime._latest_mapping_step_wall_ms > 0


@pytest.mark.parametrize(
    ("batch_size", "override", "depth_insertion", "expected_policy_enabled"),
    (
        (1, 0.2, False, True),
        (4, 0.2, False, True),
        (1, 0.0, True, False),
        (4, 0.0, True, False),
    ),
)
def test_canonical_prepared_tick_applies_tail_depth_policy_to_b1_and_batched_gates(
    monkeypatch, batch_size, override, depth_insertion, expected_policy_enabled
):
    """The tail override reaches each real tick's render gate, not just kwargs.

    A zero loss weight still permits an independently-authorized insertion
    render, so its depth evidence/work accounting is not silently disabled.
    """
    from eligsir.contracts.control import FreshnessMode
    from eligsir.runtime import production_application as production
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.config import DepthLossPolicy
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

    sources = tuple(_source(index) for index in range(batch_size))
    plan = _plan(sources, fresh=sources, batch_size=batch_size)
    for work in (*plan.supervision_work, *plan.replay_work):
        work.modality_eligibility = ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=depth_insertion,
        )
    plan.optimizer_updates = 1
    plan.snapshot_id = 3
    plan.freshness_mode = FreshnessMode.HEALTHY
    runtime = _canonical_prepared_runtime(
        monkeypatch,
        plan,
        successful_uid=_full_identity_camera_uid(sources[-1]),
        successor=None,
        batch_size=batch_size,
    )
    runtime._optimization_policy = SimpleNamespace(
        streaming_refinement_depth_loss_weight=override
    )
    runtime._depth_loss_policy = DepthLossPolicy(enabled=False, weight=0.0)
    runtime._optimization_runtime = SimpleNamespace(
        loss_assembly_kwargs={"depth_loss_weight": 0.0}
    )
    observed = []

    def run_optimization(*_args, **kwargs):
        observed.append(
            (
                kwargs["render_depth"],
                runtime._depth_loss_policy.enabled,
                runtime._optimization_runtime.loss_assembly_kwargs["depth_loss_weight"],
            )
        )
        return SimpleNamespace(
            optimizer_step_count=1,
            supervision_view_count=batch_size,
            supervised_pixel_count=batch_size,
            depth_valid_sample_count=7,
            rasterizer_call_count=1,
            microbatch_sizes=(batch_size,),
            successful_supervision_source_indices=(
                _full_identity_camera_uid(sources[-1]),
            ),
        )

    monkeypatch.setattr(
        production, "run_streaming_optimization_step", run_optimization, raising=False
    )
    monkeypatch.setattr(
        production, "run_streaming_optimization_transaction", run_optimization,
        raising=False,
    )
    monkeypatch.setattr(
        production,
        "RandomStateToken",
        SimpleNamespace(capture=lambda: object()),
        raising=False,
    )
    monkeypatch.setattr(
        production,
        "execute_fixed_topology_transaction",
        lambda _transaction, **kwargs: (
            kwargs["sequential_executor"](())
            if batch_size == 1
            else kwargs["true_batch_executor"](())
        ),
    )
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=SimpleNamespace(
                uid=_full_identity_camera_uid(binding.lease.source_identity)
            ),
        ),
    )
    owner = ReadySourceOwner(max_fresh=batch_size, max_replay=2 * batch_size)
    for source in sources:
        assert owner.offer(_lease(source), lane=ReadySourceLane.FRESH)

    adapter = ReadyB4StreamingOptimizationAdapter(
        _Bridge(owner), runtime=runtime, transfer_owner=_Transfer(),
        batch_size=batch_size, warmup_complete=True,
    )
    with runtime.post_mapping_refinement_loss_context():
        outcome = adapter.tick(
            ReadyB4StreamingStepInputs(runtime._prepared_ready_frozen_work[0], plan)
        )

    assert outcome.executed
    assert observed == [(True, expected_policy_enabled, override)]
    assert runtime._optimization_runtime.loss_assembly_kwargs["depth_loss_weight"] == 0.0
    assert runtime._depth_loss_policy == DepthLossPolicy(enabled=False, weight=0.0)
    if override == 0.0:
        assert runtime._work_clock_owner.snapshot().depth_valid_samples == 7


def test_ready_insertion_drops_borrowed_depth_before_lease_generation_replacement(monkeypatch):
    """A returned READY lease is never temporal state for a later transaction."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_insertion_frame import ReadyInsertionFrame

    _adapter, runtime, plan, fresh, _calls = _source_insertion_runtime(
        monkeypatch, render_reuse=object()
    )
    runtime._prepare_insertion_proposal = (
        CanonicalMappingRuntime._prepare_insertion_proposal.__get__(runtime)
    )
    observed = []
    runtime._prepare_depth_insertion_proposal = lambda *_args, **kwargs: (
        observed.append(kwargs) or None
    )
    runtime._prev_insert_frame = runtime._prev_insert_depth_m = None

    def ready(generation):
        intrinsics = SimpleNamespace(
            fx=8.0, fy=9.0, cx=2.0, cy=1.5,
            depth_fx=8.0, depth_fy=9.0, depth_cx=2.0, depth_cy=1.5,
            depth_height=4, depth_width=5,
        )
        return ReadyInsertionFrame.from_binding(
            SimpleNamespace(
                source_identity=fresh[-1],
                metadata=SimpleNamespace(
                    intrinsics=intrinsics,
                    pose_c2w=tuple(
                        tuple(float(row == column) for column in range(4))
                        for row in range(4)
                    ),
                    depth_shape=(4, 5),
                ),
                lease=SimpleNamespace(
                    generation=generation,
                    payloads={"depth_meters": SimpleNamespace(data=torch.full((4, 5), generation))},
                ),
            )
        )

    runtime._prepare_insertion_proposal(object(), ready(1), fresh[-1], plan)
    assert runtime._prev_insert_frame is runtime._prev_insert_depth_m is None

    runtime._prepare_insertion_proposal(object(), ready(2), fresh[-1], plan)
    assert observed[1]["prev_frame"] is None
    assert observed[1]["prev_depth_meters"] is None


def test_successful_fresh_sources_clear_commit_debt_and_return_to_replay(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(4))
    fresh = sources[:2]
    plan = _plan(sources, fresh=fresh)
    adapter, bridge, runtime = _adapter(monkeypatch, sources, plan, fresh=fresh)
    consumed = []
    runtime._advance_prepared_ready_plan = lambda value: consumed.append(value)
    runtime.tick = lambda **_kwargs: SimpleNamespace(
        executed=True,
        optimizer_steps=1,
        supervision_views=4,
        successful_supervision_sources=(fresh[0],),
    )

    outcome = adapter.tick(ReadyB4StreamingStepInputs(None, plan))

    assert outcome.completed_supervision_sources == (fresh[0],)
    assert bridge.completed == [fresh[0]]
    assert consumed == [plan]
    replay = bridge.ready_owner.select(3, fresh_limit=0)
    assert {lease.source_identity for lease in replay} == set(sources) - {fresh[1]}
    bridge.ready_owner.requeue_batch(replay, ready_event=SimpleNamespace(query=lambda: True))
    fresh_selection = bridge.ready_owner.select(1, fresh_limit=1)
    assert tuple(lease.source_identity for lease in fresh_selection) == (fresh[1],)


def test_canonical_prepared_tick_attributes_full_uid_before_commit_requeue(monkeypatch):
    """One full-identity UID clears and requeues only its exact fresh source."""

    from eligsir.contracts.control import FreshnessMode
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.production_application import MappingTickResult
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

    successful_fresh = SourceIdentity("prepared-a", 3, 17, 1_000_017)
    same_ordinal_fresh = SourceIdentity("prepared-b", 4, 17, 2_000_017)
    replay = (_source(18), _source(19))
    sources = (successful_fresh, same_ordinal_fresh, *replay)
    plan = _plan(sources, fresh=sources[:2])
    plan.optimizer_updates = 1
    plan.snapshot_id = 3
    plan.freshness_mode = FreshnessMode.HEALTHY
    successor = object()

    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier="native": SimpleNamespace(
            source_identity=binding.lease.source_identity,
            camera=SimpleNamespace(
                uid=_full_identity_camera_uid(binding.lease.source_identity)
            ),
        ),
    )
    owner = ReadySourceOwner(max_fresh=4, max_replay=8, replay_reserve=0)
    for source in sources:
        assert owner.offer(
            _lease(source),
            lane=(
                ReadySourceLane.FRESH
                if source in sources[:2]
                else ReadySourceLane.REPLAY
            ),
        )
    runtime = _canonical_prepared_runtime(
        monkeypatch,
        plan,
        successful_uid=_full_identity_camera_uid(successful_fresh),
        successor=successor,
    )
    bridge = _Bridge(owner)
    adapter = ReadyB4StreamingOptimizationAdapter(
        bridge,
        runtime=runtime,
        transfer_owner=_Transfer(),
        warmup_complete=True,
    )

    snapshot = runtime._prepared_ready_frozen_work[0]
    outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))

    assert isinstance(outcome.result, MappingTickResult)
    assert outcome.result.successful_supervision_sources == (successful_fresh,)
    assert outcome.completed_supervision_sources == (successful_fresh,)
    assert bridge.completed == [successful_fresh]
    assert runtime._last_materialized_work[1] is plan
    assert runtime._prepared_ready_candidate_work[1] is successor
    replay_selection = owner.select(3, fresh_limit=0)
    assert {lease.source_identity for lease in replay_selection} == {
        successful_fresh,
        *replay,
    }


@pytest.mark.parametrize("batch_size", (1, 4))
@pytest.mark.parametrize("bypass_feedback", (False, True))
def test_canonical_tick_delivers_every_view_to_replay_feedback(
    monkeypatch, batch_size, bypass_feedback
):
    """The real tick/READY return path drives scoring; bypassing it loses evidence."""
    from eligsir.contracts.control import FreshnessMode
    from eligsir.mapping.scheduler.historical_candidate_index import (
        HistoricalCandidateIndex, HistoricalFrameMetadata,
    )
    from eligsir.mapping.scheduler.hot_cold_challenger import HotColdChallengerArena
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken, SupervisionEvidenceBatch, SupervisionModality,
    )
    from eligsir.runtime import production_application as production
    from eligsir.runtime import ready_b4_streaming_adapter as adapter_module
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid
    from eligsir.runtime.transfers import DeviceTransferOwner

    sources = tuple(_source(i) for i in range(batch_size))
    plan = _plan(sources, batch_size=batch_size)
    plan.optimizer_updates, plan.snapshot_id = 1, 3
    plan.freshness_mode = FreshnessMode.HEALTHY
    runtime = _canonical_prepared_runtime(
        monkeypatch, plan, successful_uid=_full_identity_camera_uid(sources[-1]),
        successor=None, batch_size=batch_size,
    )
    # Keep only CUDA event creation fake; packing/readback use actual CPU Torch.
    monkeypatch.setattr(production.torch, "stack", torch.stack, raising=False)
    monkeypatch.setattr(production, "RandomStateToken", SimpleNamespace(capture=lambda: object()), raising=False)
    runtime._optimization_policy = SimpleNamespace(streaming_historical_usefulness_tracking=True)
    runtime._depth_loss_policy = SimpleNamespace(enabled=False, weight=0.0)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    for _ in range(2):
        runtime._transfer_owner.upload_packed((np.zeros(256, dtype=np.uint8),), purpose="test_ingress")
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._historical_frame_ids_by_source = {source: i for i, source in enumerate(sources)}
    runtime._hot_cold_challenger_arena = HotColdChallengerArena(lease_ticks=0)
    for i in range(batch_size):
        runtime._historical_candidate_index.register(HistoricalFrameMetadata(i, 0))
    if not bypass_feedback:
        runtime._observe_replay_evidence = CanonicalMappingRuntime._observe_replay_evidence.__get__(runtime)
    execute = production.execute_fixed_topology_transaction

    def completed(*args, **kwargs):
        result = execute(*args, **kwargs)
        result.evidence = SupervisionEvidenceBatch(
            snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, runtime.iteration + 1, 0, 0, 0),
            transaction_id=runtime.iteration + 1,
            source_identities=sources,
            modalities=tuple(SupervisionModality() for _ in sources),
            rgb_residual_summary=torch.arange(1, batch_size + 1, dtype=torch.float32),
            depth_residual_summary=torch.zeros(batch_size),
            rgb_valid_pixels=torch.full((batch_size,), 12),
            depth_valid_pixels=torch.zeros(batch_size, dtype=torch.int64),
        )
        return result

    monkeypatch.setattr(production, "execute_fixed_topology_transaction", completed)
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", _Bridge)
    monkeypatch.setattr(adapter_module, "_compute_completion", TransferEvent)
    monkeypatch.setattr(adapter_module, "bind_ready_camera", lambda binding, **_: SimpleNamespace(
        source_identity=binding.lease.source_identity, lease=binding.lease,
        camera=SimpleNamespace(uid=_full_identity_camera_uid(binding.lease.source_identity)),
    ))
    owner = ReadySourceOwner(max_fresh=batch_size, max_replay=max(4, 2 * batch_size), replay_reserve=0)
    for source in sources:
        lease = replace(_lease(source), source_metadata={"replay_quality": (1.0, None)})
        assert owner.offer(lease, lane=ReadySourceLane.REPLAY)
    adapter = adapter_module.ReadyB4StreamingOptimizationAdapter(
        _Bridge(owner), runtime=runtime, transfer_owner=_Transfer(),
        batch_size=batch_size, warmup_complete=True,
    )
    snapshot = SimpleNamespace(snapshot_id=3)
    for _ in range(3):
        runtime._prepared_ready_frozen_work = (snapshot, plan)
        runtime._prepared_ready_candidate_work = (snapshot, plan)
        outcome = adapter.tick(adapter_module.ReadyB4StreamingStepInputs(snapshot, plan))
        assert outcome.executed
        runtime._poll_replay_evidence()
    assert runtime._transfer_owner.accounting().d2h_ops == (0 if bypass_feedback else 3)
    for i in range(batch_size):
        assert runtime._hot_cold_challenger_arena.modality_trusted(i, "rgb") is not bypass_feedback
        weight = runtime._historical_candidate_index.metadata_for(i).base_weight
        assert (weight == 1.0) if bypass_feedback else (weight > 1.0)
    assert set(owner.replay_resident_sources()) == set(sources)


def test_eviction_hold_path_uses_injected_timing_scope(monkeypatch):
    """Issue: throughput-recovery Track 2. The pre-execute return-lane
    eviction retry loop (owner.evict_replay_provisional under capacity
    pressure) must report through the adapter's injected timing_scope so
    its cost can be measured separately from the rest of tick() -- this
    pins that the scope name fires at least once when an eviction actually
    happens, without asserting a specific duration (that's runtime data,
    not a unit-test invariant)."""
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
    )

    sources = tuple(_source(ordinal) for ordinal in range(4))
    plan = _plan(sources, fresh=sources)
    adapter, bridge, runtime = _adapter(monkeypatch, sources, plan, fresh=sources)

    scope_calls = []

    class _RecordingScope:
        def __init__(self, name):
            scope_calls.append(name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    adapter._timing_scope = _RecordingScope  # set directly; see
    # test_constructor_injected_timing_scope_is_invoked_on_eviction_hold
    # below for the positive control that constructor wiring itself works.

    # Force a return-lane capacity failure on the first reservation attempt
    # so the eviction-provisional branch actually runs.
    original_reserve = bridge.ready_owner.reserve_return_lanes
    calls = {"n": 0}

    def flaky_reserve(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ready-source return lane capacity is unavailable")
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(bridge.ready_owner, "reserve_return_lanes", flaky_reserve)

    adapter.tick(ReadyB4StreamingStepInputs(snapshot=None, work_plan=plan))

    assert "ready_b4_eviction_hold" in scope_calls


def test_constructor_injected_timing_scope_is_invoked_on_eviction_hold(monkeypatch):
    """Positive control for the throughput-recovery Track 2 null result.

    The GPU measurement backing "no fix needed" observed zero
    [timing_scope] lines during a real run. That silence is only
    meaningful if the constructor-to-invocation wiring for timing_scope
    actually works -- otherwise "zero lines" is equally consistent with
    the instrumentation being silently unwired. This test constructs the
    real adapter via its normal __init__ (passing timing_scope as a
    constructor argument, not assigning it after construction, unlike
    the sibling test above), forces the same eviction-hold path, and
    asserts the injected callable was actually invoked with the expected
    scope name -- proving the constructor-argument path is live.
    """
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
    )

    sources = tuple(_source(ordinal) for ordinal in range(4))
    plan = _plan(sources, fresh=sources)

    scope_calls = []

    class _RecordingScope:
        def __init__(self, name):
            scope_calls.append(name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    adapter, bridge, runtime = _adapter(
        monkeypatch, sources, plan, fresh=sources, timing_scope=_RecordingScope
    )

    # adapter._timing_scope must be exactly the constructor-injected
    # callable, not the default no-op fallback.
    assert adapter._timing_scope is _RecordingScope

    original_reserve = bridge.ready_owner.reserve_return_lanes
    calls = {"n": 0}

    def flaky_reserve(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ready-source return lane capacity is unavailable")
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(bridge.ready_owner, "reserve_return_lanes", flaky_reserve)

    adapter.tick(ReadyB4StreamingStepInputs(snapshot=None, work_plan=plan))

    assert "ready_b4_eviction_hold" in scope_calls
def test_absent_authorized_replay_source_replans_after_bounded_starvation(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    adapter, _bridge, runtime = _adapter(
        monkeypatch, sources, plan, offered_sources=sources[:3]
    )
    inputs = ReadyB4StreamingStepInputs(None, plan)
    original_latest_tick_work = runtime._latest_tick_work
    threshold = adapter._ABSENT_REPLAN_AFTER_TICKS

    for _ in range(threshold - 1):
        outcome = adapter.tick(inputs)
        assert outcome.state is ReadyB4StreamingStepState.STARVED

    assert runtime._latest_tick_work is original_latest_tick_work
    assert adapter.absent_replans == 0

    outcome = adapter.tick(inputs)

    assert outcome.state is ReadyB4StreamingStepState.DEFERRED
    assert outcome.reason == "prepared_authorized_source_absent"
    assert runtime._latest_tick_work is None
    assert adapter.absent_replans == 1


def test_event_pending_source_never_triggers_absent_replan(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    adapter, bridge, _runtime = _adapter(
        monkeypatch, sources, plan, offered_sources=sources[:3]
    )
    pending_lease = replace(
        _lease(sources[3]), ready_event=SimpleNamespace(query=lambda: False)
    )
    assert bridge.ready_owner.offer(pending_lease, lane=ReadySourceLane.REPLAY)

    inputs = ReadyB4StreamingStepInputs(None, plan)
    for _ in range(adapter._ABSENT_REPLAN_AFTER_TICKS * 2):
        outcome = adapter.tick(inputs)
        assert outcome.state is ReadyB4StreamingStepState.STARVED

    assert adapter.absent_replans == 0


def test_absent_counter_resets_when_the_plan_changes(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources_a = tuple(_source(index) for index in range(4))
    plan_a = _plan(sources_a)
    sources_b = tuple(_source(10 + index) for index in range(4))
    plan_b = _plan(sources_b)

    adapter, bridge, _runtime = _adapter(
        monkeypatch, sources_a, plan_a, offered_sources=sources_a[:3]
    )
    for source in sources_b[:3]:
        assert bridge.ready_owner.offer(_lease(source), lane=ReadySourceLane.REPLAY)

    inputs_a = ReadyB4StreamingStepInputs(None, plan_a)
    inputs_b = ReadyB4StreamingStepInputs(None, plan_b)
    threshold = adapter._ABSENT_REPLAN_AFTER_TICKS

    for _ in range(threshold - 1):
        outcome = adapter.tick(inputs_a)
        assert outcome.state is ReadyB4StreamingStepState.STARVED

    outcome = adapter.tick(inputs_b)
    assert outcome.state is ReadyB4StreamingStepState.STARVED

    for _ in range(threshold - 1):
        outcome = adapter.tick(inputs_a)
        assert outcome.state is ReadyB4StreamingStepState.STARVED

    assert adapter.absent_replans == 0


def test_in_flight_blocks_absent_replan(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )
    from eligsir.runtime.ready_source import ReadySourceOwner

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources)
    adapter, bridge, _runtime = _adapter(
        monkeypatch, sources, plan, offered_sources=sources[:3]
    )

    real_counters = ReadySourceOwner.counters
    monkeypatch.setattr(
        ReadySourceOwner,
        "counters",
        property(lambda self: replace(real_counters.fget(self), in_flight=1)),
    )

    inputs = ReadyB4StreamingStepInputs(None, plan)
    for _ in range(adapter._ABSENT_REPLAN_AFTER_TICKS * 2):
        outcome = adapter.tick(inputs)
        assert outcome.state is ReadyB4StreamingStepState.STARVED

    assert adapter.absent_replans == 0


def test_issue1663_first_supervision_completion_counters_follow_real_execution(monkeypatch):
    """#1663 acceptance counters on the real adapter: an executed fresh B4 admits
    one COMMIT and completes its first supervision exactly once; re-serving the
    same source never double-counts; debt is zero once served."""
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    fresh = _source(0)
    sources = (fresh, _source(1), _source(2), _source(3))
    plan = _plan(sources, fresh=(fresh,), insertion=True, batch_size=4, requested_batch_size=32)
    result = SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    adapter, bridge, _runtime = _adapter(
        monkeypatch, sources, plan, fresh=(fresh,), result=result,
        source_insertion=True, batch_size=32,
    )
    assert (adapter.admitted_commit_count, adapter.first_supervision_completions) == (0, 0)

    outcome = adapter.tick(ReadyB4StreamingStepInputs(object(), plan))
    assert outcome.executed and bridge.completed == [fresh]
    assert adapter.admitted_commit_count == 1
    assert adapter.first_supervision_completions == 1
    assert adapter.admitted_commit_count - adapter.first_supervision_completions == 0

    # Serving the same fresh source again is not a second first-supervision.
    adapter.tick(ReadyB4StreamingStepInputs(object(), plan))
    assert adapter.admitted_commit_count == 1
    assert adapter.first_supervision_completions == 1


def test_issue1663_replay_only_steps_are_counted_only_while_fresh_debt_exists(monkeypatch):
    """A replay-only execution while a fresh source is still pending in the READY
    owner is the starvation signal #1663 forbids; with no fresh debt it is benign."""
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingStepInputs

    sources = tuple(_source(index) for index in range(4))
    replay_only = _plan(sources, fresh=(), batch_size=4)
    result = SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    adapter, bridge, _runtime = _adapter(monkeypatch, sources, replay_only, result=result)

    adapter.tick(ReadyB4StreamingStepInputs(object(), replay_only))
    assert bridge.ready_owner.counters.fresh_pending == 0
    assert adapter.replay_only_steps_while_fresh_pending == 0

    assert bridge.ready_owner.offer(_lease(_source(9)), lane=ReadySourceLane.FRESH)
    assert bridge.ready_owner.counters.fresh_pending == 1
    adapter.tick(ReadyB4StreamingStepInputs(object(), replay_only))
    assert adapter.replay_only_steps_while_fresh_pending == 1


def test_depth_insertion_fails_before_gpu_dispatch(monkeypatch):
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepState,
    )

    sources = tuple(_source(index) for index in range(4))
    plan = _plan(sources, insertion=True)
    adapter, _bridge, runtime = _adapter(monkeypatch, sources, plan)
    runtime.tick = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not dispatch"))
