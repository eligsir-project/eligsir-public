"""Reachability contracts for the slim causal prepared B4 path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

# Every test below imports eligsir.runtime.causal_prepared_ready
# lazily inside its own body (see the module docstring for why: it pulls
# in prepared_runtime_bridge -> prepared_source_producer -> torch), which
# tests/_torch_collect_scan.py's static transitive-import scanner cannot
# see (it only follows module-level imports, not ones nested inside
# function bodies). Without this, the whole file hard-fails with a raw
# ModuleNotFoundError in the torch-less CPU smoke lane instead of being
# skipped like every other torch-dependent test file.
torch = pytest.importorskip("torch")

from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import ResolutionTier


def test_builder_moves_commit_retention_to_pre_handoff_sink(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    captured = {}
    calls = []

    class Bridge:
        def __init__(self, handoff, producer, *, ready_owner, **kwargs):
            self.handoff = handoff
            self.ready_owner = ready_owner
            captured["ready_owner"] = ready_owner
            captured.update(kwargs)

        def drain(self):
            from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult

            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

        def close(self):
            pass

    class Runtime:
        initialized = False
        bootstrap_frames = 5

        def enable_ready_source_owner(self, _owner, *, warmup_complete):
            assert warmup_complete is False

        def retain_prepared_source(self, item, *, lane):
            calls.append(("retain", item, lane))
            return True

        def record_retained_prepared_commit(self, item):
            calls.append(("obligation", item))

        def clear_retained_prepared_commit(self, _item):
            return None

        def mark_source_complete(self):
            return None

    runtime = Runtime()
    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(
        prepared,
        "_transfer_owner_for_runtime",
        lambda _runtime: SimpleNamespace(prepare_source=lambda *_args, **_kwargs: None),
    )
    path = prepared.build_causal_prepared_ready_path(
        runtime=runtime,
        config=SimpleNamespace(
            hard_limits={"optimization_batch_size": 4},
            static_options={
                "optimization_constellation_policy": "fixed",
                "mapping_resolution_tier": "quarter",
                "insert_from_depth": True,
                "prepared_ready_source_insertion": True,
            },
            depth_insertion=SimpleNamespace(
                temporal_check_enabled=False,
                require_temporal_stability=True,
                require_temporal_for_sensor_closer=False,
            ),
        ),
        args=object(),
    )

    disposition = SimpleNamespace(semantic_action=prepared.SemanticAction.COMMIT)
    item = SimpleNamespace(disposition=disposition, packet=SimpleNamespace(depth=object()))
    path.handoff._commit_sink(item)

    assert [call[0] for call in calls] == ["retain", "obligation"]
    assert calls[0][2] is prepared.ReadySourceLane.FRESH
    before = tuple(calls)
    try:
        path.handoff._commit_sink(
            SimpleNamespace(disposition=disposition, packet=SimpleNamespace(depth=None))
        )
    except ValueError as error:
        assert "RGB-D COMMIT" in str(error)
    else:
        raise AssertionError("unsupported prepared COMMIT created durable debt")
    assert tuple(calls) == before
    assert captured["ready_owner"]._max_fresh == 5
    assert captured["ready_owner"]._max_replay == 16
    assert set(captured) - {"ready_owner"} == {
        "consumer_owner", "max_items_per_tick", "retained_commit_clearer"
    }
    path.close()


@pytest.mark.parametrize("tracking_enabled", [False, True])
def test_builder_enables_replay_quality_only_with_historical_usefulness_tracking(
    monkeypatch, tracking_enabled
):
    import eligsir.runtime.causal_prepared_ready as prepared

    captured = {}

    class Bridge:
        def __init__(self, handoff, producer, *, ready_owner, **_kwargs):
            self.handoff = handoff
            captured["producer"] = producer

        def close(self):
            pass

        def drain(self):
            from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult

            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

    class Runtime:
        initialized = False
        bootstrap_frames = 4

        def enable_ready_source_owner(self, _owner, *, warmup_complete):
            assert warmup_complete is False

        def retain_prepared_source(self, _item, *, lane):
            return True

        def record_retained_prepared_commit(self, _item):
            return None

        def clear_retained_prepared_commit(self, _item):
            return None

        def mark_source_complete(self):
            return None

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(
        prepared,
        "_transfer_owner_for_runtime",
        lambda _runtime: SimpleNamespace(prepare_source=lambda *_args, **_kwargs: None),
    )
    path = prepared.build_causal_prepared_ready_path(
        runtime=Runtime(),
        config=SimpleNamespace(
            hard_limits={"optimization_batch_size": 4},
            static_options={
                "optimization_constellation_policy": "fixed",
                "mapping_resolution_tier": "quarter",
                "insert_from_depth": False,
            },
            optimization=SimpleNamespace(
                streaming_historical_usefulness_tracking=tracking_enabled
            ),
        ),
        args=object(),
    )
    try:
        assert captured["producer"]._measure_replay_quality is tracking_enabled
    finally:
        path.close()


def test_historical_usefulness_tracking_reaches_canonical_optimization_policy():
    from eligsir.cli.parser import parse_training_args
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/quality-policy-test"
    assert (
        mapping_runtime_config_from_args(args)
        .optimization.streaming_historical_usefulness_tracking
        is False
    )
    args.streaming_historical_usefulness_tracking = True
    assert (
        mapping_runtime_config_from_args(args)
        .optimization.streaming_historical_usefulness_tracking
        is True
    )


def test_adapter_factory_wires_timing_scope_from_measurement_env_var(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    class Bridge:
        def __init__(self, handoff, producer, *, ready_owner, **kwargs):
            self.handoff = handoff
            self.ready_owner = ready_owner

        def drain(self):
            from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult

            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

        def close(self):
            pass

    class Runtime:
        initialized = False
        bootstrap_frames = 5

        def enable_ready_source_owner(self, _owner, *, warmup_complete):
            return None

        def retain_prepared_source(self, item, *, lane):
            return True

        def record_retained_prepared_commit(self, item):
            return None

        def clear_retained_prepared_commit(self, _item):
            return None

        def mark_source_complete(self):
            return None

    captured_kwargs = {}

    class CapturingAdapter:
        def __init__(self, bridge, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(prepared, "ReadyB4StreamingOptimizationAdapter", CapturingAdapter)
    monkeypatch.setattr(
        prepared,
        "_transfer_owner_for_runtime",
        lambda _runtime: SimpleNamespace(prepare_source=lambda *_args, **_kwargs: None),
    )

    config = SimpleNamespace(
        hard_limits={"optimization_batch_size": 4},
        static_options={
            "optimization_constellation_policy": "fixed",
            "mapping_resolution_tier": "quarter",
            "insert_from_depth": False,
        },
    )

    # (a) env var unset -> timing_scope must be None (byte-identical default behavior).
    monkeypatch.delenv("ELIGSIR_TIME_EVICTION_HOLD", raising=False)
    path = prepared.build_causal_prepared_ready_path(runtime=Runtime(), config=config, args=object())
    path._adapter_factory()
    assert captured_kwargs["timing_scope"] is None
    path.close()

    # (b) env var set -> timing_scope must be a real callable.
    captured_kwargs.clear()
    monkeypatch.setenv("ELIGSIR_TIME_EVICTION_HOLD", "1")
    path = prepared.build_causal_prepared_ready_path(runtime=Runtime(), config=config, args=object())
    path._adapter_factory()
    assert callable(captured_kwargs["timing_scope"])
    path.close()


def test_builder_rejects_only_prepared_b4_capabilities_the_producer_cannot_realize():
    import pytest
    import eligsir.runtime.causal_prepared_ready as prepared

    base = {
        "optimization_constellation_policy": "fixed",
        "mapping_resolution_tier": "native",
        "insert_from_depth": False,
    }
    for overrides, message in (
        ({"optimization_constellation_policy": "adaptive"}, "constellation_policy=fixed"),
        ({"insert_from_depth": True}, "no-streaming_insert_from_depth"),
        ({"replay_depth_max_per_event": 1}, "replay_depth_max_per_event=0"),
        ({"admission_prematerialize": True}, "no-streaming_admission_prematerialize"),
    ):
        with pytest.raises(ValueError, match=message):
            prepared.build_causal_prepared_ready_path(
                runtime=object(),
                config=SimpleNamespace(
                    hard_limits={"optimization_batch_size": 4},
                    static_options=base | overrides,
                ),
                args=object(),
            )


def test_builder_allows_only_opted_in_source_insertion_and_never_replay_insertion():
    import pytest
    import eligsir.runtime.causal_prepared_ready as prepared

    base = {
        "optimization_constellation_policy": "fixed",
        "mapping_resolution_tier": "native",
        "insert_from_depth": True,
        "prepared_ready_source_insertion": True,
    }

    def build(overrides):
        return prepared.build_causal_prepared_ready_path(
            runtime=object(),
            config=SimpleNamespace(
                hard_limits={"optimization_batch_size": 4},
                static_options=base | overrides,
            ),
            args=object(),
        )

    with pytest.raises(ValueError, match="no-streaming_insert_from_depth"):
        build({"prepared_ready_source_insertion": False})
    with pytest.raises(ValueError, match="replay depth insertion is not supported"):
        build({"replay_depth_max_per_event": 1})
    with pytest.raises(ValueError, match="no-streaming_admission_prematerialize"):
        build({"admission_prematerialize": True})


def test_builder_refuses_required_temporal_source_insertion_without_borrowing_depth():
    import pytest
    import eligsir.runtime.causal_prepared_ready as prepared

    with pytest.raises(ValueError, match="required temporal depth masks"):
        prepared.build_causal_prepared_ready_path(
            runtime=object(),
            config=SimpleNamespace(
                hard_limits={"optimization_batch_size": 4},
                static_options={
                    "optimization_constellation_policy": "fixed",
                    "mapping_resolution_tier": "native",
                    "insert_from_depth": True,
                    "prepared_ready_source_insertion": True,
                },
                depth_insertion=SimpleNamespace(
                    temporal_check_enabled=True,
                    require_temporal_stability=True,
                    require_temporal_for_sensor_closer=False,
                ),
            ),
            args=object(),
        )


def test_initialized_tick_returns_adapter_result_directly(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared
    from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepResult,
        ReadyB4StreamingStepState,
    )

    class Bridge:
        handoff = object()

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    selection = SimpleNamespace(leases=())
    canonical = SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)
    plan = object()
    adapter_result = ReadyB4StreamingStepResult(
        ReadyB4StreamingStepState.EXECUTED,
        PreparedRuntimeBridgeResult(0, 0, 0, 0),
        selection,
        result=canonical,
        completed_supervision_sources=("a", "b", "c", "d"),
        work_plan=plan,
    )
    path = prepared.CausalPreparedReadyB4Path(
        runtime=SimpleNamespace(initialized=True, iteration=9),
        bridge=Bridge(),
        adapter_factory=lambda: SimpleNamespace(tick=lambda _inputs: adapter_result),
        step_inputs_factory=lambda: ReadyB4StreamingStepInputs(None, plan),
    )

    outcome = path.tick()
    assert outcome is adapter_result
    assert outcome.optimizer_result is canonical
    assert outcome.optimizer_steps == 1
    assert outcome.supervision_views == 4
    assert outcome.completed_supervision_sources == ("a", "b", "c", "d")
    assert outcome.work_plan is plan


def test_stale_tick_returns_authority_to_runtime_reauthorization(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared
    from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingStepInputs,
        ReadyB4StreamingStepResult,
        ReadyB4StreamingStepState,
    )

    class Bridge:
        handoff = object()

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    plan = object()
    stale = ReadyB4StreamingStepResult(
        ReadyB4StreamingStepState.STALE,
        PreparedRuntimeBridgeResult(0, 0, 0, 0),
        SimpleNamespace(leases=()),
        reason="prepared_work_plan_version_stale",
        work_plan=plan,
    )
    advanced = []
    runtime = SimpleNamespace(
        initialized=True,
        iteration=9,
        _advance_prepared_ready_plan=lambda current: advanced.append(current),
    )
    path = prepared.CausalPreparedReadyB4Path(
        runtime=runtime,
        bridge=Bridge(),
        adapter_factory=lambda: SimpleNamespace(tick=lambda _inputs: stale),
        step_inputs_factory=lambda: ReadyB4StreamingStepInputs(None, plan),
    )

    assert path.tick() is stale
    assert advanced == [plan]


def test_exact_ready_candidate_dispatches_before_promotion_maintenance():
    """Issue #1673: healthy path dispatches before the slow maintenance pass.

    When ``exact_readiness`` already reports a READY candidate, the adapter
    must be entered before the (here artificially slow) promotion-shortlist
    scan and ``schedule_shortlist`` call — not after them.
    """

    import eligsir.runtime.causal_prepared_ready as prepared

    order = []
    sources = tuple(SourceIdentity("healthy", 0, index, 3_000 + index) for index in range(2))
    candidate_plan = SimpleNamespace(supervision_sources=sources[:1], replay_sources=sources[1:])
    candidate_work = (object(), candidate_plan)

    class Owner:
        @staticmethod
        def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
            return tuple((source, "READY") for source in exact_sources)

    def slow_shortlist():
        order.append("shortlist_scan")
        return ()

    def slow_schedule_shortlist(_candidates, **_kwargs):
        time.sleep(0.05)
        order.append("promotion_maintenance")

    runtime = SimpleNamespace(
        initialized=True,
        _prepared_ready_plan_for_execution=lambda: None,
        _prepared_ready_frozen_work=None,
        _prepared_ready_candidate_work=candidate_work,
        _ready_source_owner=Owner(),
        _latest_control_outputs=None,
        prepared_cold_promotion_shortlist=slow_shortlist,
        _observation_db=SimpleNamespace(get_frame=lambda _frame_id: None),
    )
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = runtime
    path._promotion_owner = SimpleNamespace(schedule_shortlist=slow_schedule_shortlist)
    path._adapter = SimpleNamespace(
        tick=lambda _inputs: order.append("adapter_dispatch") or SimpleNamespace(state="deferred")
    )
    path._step_inputs_factory = lambda: None
    path._maybe_complete_source_phase = lambda _outcome: None

    path.tick()

    assert order == ["adapter_dispatch", "shortlist_scan", "promotion_maintenance"]


def test_unready_candidate_dispatches_ready_replay_fallback_before_maintenance(monkeypatch):
    """Issue #1673: a frozen READY fallback is dispatched before COLD upkeep.

    Removing the post-freeze adapter dispatch, or restoring the old
    ``exact_dispatch_ready`` branch, makes the maintenance event precede the
    real READY owner/adapter execution below.
    """

    import eligsir.runtime.causal_prepared_ready as prepared
    import eligsir.runtime.ready_b4_streaming_adapter as adapter_module
    from eligsir.contracts import (
        BufferOwnership,
        BufferRef,
        BufferSpec,
        MemoryDomain,
        ModalityEligibility,
    )
    from eligsir.runtime.prepared_source import PreparedSourceLease
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
    )
    from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner

    order = []
    candidate_sources = tuple(
        SourceIdentity("candidate", 0, index, 5_000 + index) for index in range(4)
    )
    fallback_sources = tuple(
        SourceIdentity("fallback", 0, index, 6_000 + index) for index in range(4)
    )

    def plan(sources):
        work = tuple(
            SimpleNamespace(
                source=source,
                resolution_tier=SimpleNamespace(value="native"),
                modality_eligibility=ModalityEligibility(rgb_supervision=True),
            )
            for source in sources
        )
        return SimpleNamespace(
            supervision_work=(),
            replay_work=work,
            supervision_sources=(),
            replay_sources=sources,
            supervision_constellation=SimpleNamespace(
                realized_batch_size=4,
                realized_camera_count=4,
                microbatch_count=1,
                microbatch_sizes=(4,),
            ),
            insertion=SimpleNamespace(enabled=False),
            replay_insertion=None,
            map_content_version=7,
            topology_version=11,
            parameter_version=13,
        )

    def lease(source):
        token = object()
        payload = BufferRef(
            data=torch.zeros((1,), dtype=torch.float32),
            spec=BufferSpec(
                (1,), "float32", MemoryDomain.OTHER_DEVICE, "test", device="cpu"
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

    class Bridge:
        handoff = object()

        def __init__(self, owner):
            self.ready_owner = owner

        def requeue_ready_only_sources(self, selection, *, ready_event, lane_by_source=None):
            return self.ready_owner.requeue_batch(
                selection.leases, ready_event=ready_event, lane_by_source=lane_by_source
            )

        @staticmethod
        def complete_retained_commit_first_supervision(_source):
            return True

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    for source in fallback_sources:
        assert owner.offer(lease(source), lane=ReadySourceLane.REPLAY)
    bridge = Bridge(owner)
    fallback = plan(fallback_sources)
    candidate = plan(candidate_sources)
    snapshot = object()

    def freeze():
        runtime._prepared_ready_frozen_work = (snapshot, fallback)
        return fallback

    def shortlist():
        order.append("shortlist_scan")
        return ()

    runtime = SimpleNamespace(
        initialized=True,
        _prepared_ready_plan_for_execution=freeze,
        _prepared_ready_frozen_work=None,
        _prepared_ready_candidate_work=(snapshot, candidate),
        _ready_source_owner=owner,
        _latest_control_outputs=None,
        prepared_cold_promotion_shortlist=shortlist,
        _observation_db=SimpleNamespace(get_frame=lambda _frame_id: None),
        _versions=lambda: (7, 11, 13),
    )
    runtime.tick = lambda *, ready_transaction: order.append("adapter_dispatch") or SimpleNamespace(
        executed=True, optimizer_steps=1, supervision_views=4
    )
    monkeypatch.setattr(adapter_module, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(
        adapter_module,
        "bind_ready_camera",
        lambda binding, *, tier: SimpleNamespace(
            source_identity=binding.lease.source_identity, camera=object()
        ),
    )
    adapter = ReadyB4StreamingOptimizationAdapter(
        bridge, runtime=runtime, transfer_owner=SimpleNamespace(handoff_to_compute=lambda *_: None),
        warmup_complete=True,
    )
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = runtime
    scheduled = {}

    def schedule_shortlist(_candidates, **kwargs):
        scheduled.update(kwargs)
        order.append("promotion_maintenance")

    path._promotion_owner = SimpleNamespace(
        schedule_shortlist=schedule_shortlist
    )
    path._adapter = adapter
    path._step_inputs_factory = lambda: prepared._step_inputs_for_runtime(runtime)
    path._maybe_complete_source_phase = lambda _outcome: None

    outcome = path.tick()

    assert outcome.executed
    assert tuple(lease.source_identity for lease in outcome.selection.leases) == fallback_sources
    assert outcome.selection.fallback_to_replay is True
    assert order == ["adapter_dispatch", "shortlist_scan", "promotion_maintenance"]
    assert scheduled["protected_sources"] == frozenset(fallback_sources)


def test_starved_candidate_schedules_promotion_after_deferred_dispatch():
    """Issue #1673: a deferred READY path still schedules future COLD work."""

    import eligsir.runtime.causal_prepared_ready as prepared

    order = []
    sources = tuple(SourceIdentity("starved", 0, index, 4_000 + index) for index in range(2))
    candidate_plan = SimpleNamespace(supervision_sources=sources[:1], replay_sources=sources[1:])
    candidate_work = (object(), candidate_plan)

    class Owner:
        @staticmethod
        def exact_readiness(*, fresh_sources, replay_sources, exact_sources):
            return tuple((source, "EVENT_PENDING") for source in exact_sources)

    def shortlist():
        order.append("shortlist_scan")
        return ()

    def schedule_shortlist(_candidates, **_kwargs):
        order.append("promotion_maintenance")

    runtime = SimpleNamespace(
        initialized=True,
        _prepared_ready_plan_for_execution=lambda: None,
        _prepared_ready_frozen_work=None,
        _prepared_ready_candidate_work=candidate_work,
        _ready_source_owner=Owner(),
        _latest_control_outputs=None,
        prepared_cold_promotion_shortlist=shortlist,
        _observation_db=SimpleNamespace(get_frame=lambda _frame_id: None),
    )
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = runtime
    path._promotion_owner = SimpleNamespace(schedule_shortlist=schedule_shortlist)
    path._adapter = SimpleNamespace(
        tick=lambda _inputs: order.append("adapter_dispatch") or SimpleNamespace(state="deferred")
    )
    path._step_inputs_factory = lambda: None
    path._maybe_complete_source_phase = lambda _outcome: None

    path.tick()

    assert order == ["adapter_dispatch", "shortlist_scan", "promotion_maintenance"]


def test_bootstrap_does_not_schedule_cold_promotion_for_fresh_ingress():
    import eligsir.runtime.causal_prepared_ready as prepared

    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = SimpleNamespace(
        initialized=False,
        _latest_control_outputs=None,
        prepared_cold_promotion_shortlist=lambda: (),
    )
    path._promotion_owner = SimpleNamespace(
        schedule_shortlist=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bootstrap must not page in COLD work")
        )
    )
    expected = SimpleNamespace(state="bootstrap_pending")
    path._bootstrap_tick = lambda: expected
    path._maybe_complete_source_phase = lambda _outcome: None

    assert path.tick() is expected


def test_multistage_bootstrap_clears_all_commit_obligations_only_on_success(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    monkeypatch.setattr(
        prepared,
        "ReadyOnlyTickSelection",
        lambda leases, fresh_count, replay_count, fallback_to_replay, starved: (
            SimpleNamespace(
                leases=leases,
                fresh_count=fresh_count,
                replay_count=replay_count,
                fallback_to_replay=fallback_to_replay,
                starved=starved,
            )
        ),
    )

    sources = tuple(object() for _ in range(5))
    batches = [
        tuple(
            SimpleNamespace(source_identity=source, is_ready=lambda: True)
            for source in sources[:4]
        ),
        (SimpleNamespace(source_identity=sources[4], is_ready=lambda: True),),
    ]

    class Owner:
        @property
        def pending_leases(self):
            return batches[0]

        def select(self, requested):
            batch = batches.pop(0)
            assert len(batch) == requested
            return batch

        @staticmethod
        def selected_lane(_lease):
            return prepared.ReadySourceLane.FRESH

    class Bridge:
        ready_owner = Owner()

        def __init__(self):
            self.completed = []

        def complete_retained_commit_first_supervision(self, source):
            self.completed.append(source)
            return True

        def requeue_ready_only_sources(self, *_args, **_kwargs):
            raise AssertionError("complete bootstrap batches must not be returned early")

    class Runtime:
        initialized = False
        iteration = 0

        def __init__(self):
            self.calls = 0
            self._prepared_ready_candidate_work = (object(), object())
            self.rearmed = []

        def ready_bootstrap_remaining(self):
            return 5 if self.calls == 0 else 1

        def bootstrap_ready_only_selection(self, _selection, *, bootstrap_event):
            self.calls += 1
            assert bootstrap_event is not None
            if self.calls == 2:
                self.initialized = True
                return True
            return False

        def _advance_prepared_ready_plan(self, plan):
            assert set(path.bridge.completed) == set(sources)
            self.rearmed.append(plan)

    completion = SimpleNamespace(record_after_bootstrap=lambda: None)
    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = Runtime()
    path.bridge = Bridge()
    path._prep_worker = None
    path._bootstrap_completion_factory = lambda: completion
    path._bootstrap_fresh_sources = {}

    first = path._bootstrap_tick()
    assert first.state == "bootstrap_pending"
    assert path.bridge.completed == []
    assert tuple(path._bootstrap_fresh_sources) == sources[:4]

    second = path._bootstrap_tick()
    assert second.state == "bootstrapped"
    assert second.executed is False
    assert second.optimizer_steps == 0
    assert second.supervision_views == 0
    assert second.completed_supervision_sources == sources
    assert set(path.bridge.completed) == set(sources)
    assert len(path.bridge.completed) == 5
    assert tuple(path._bootstrap_fresh_sources) == ()
    assert path._runtime.rearmed == [path._runtime._prepared_ready_candidate_work[1]]


@dataclass(frozen=True)
class _Plan:
    source: object
    map_content_version: int
    topology_version: int
    parameter_version: int


def test_delayed_plan_preserves_scheduler_version_provenance():
    import eligsir.runtime.causal_prepared_ready as prepared

    source = object()
    plan = _Plan(source, 1, 2, 3)
    snapshot = object()
    work = (snapshot, plan)
    runtime = SimpleNamespace(
        _latest_tick_work=work,
        _latest_report_work=None,
        _last_materialized_work=None,
        initialized=True,
    )
    inputs = prepared._step_inputs_for_runtime(runtime)
    assert inputs.snapshot is snapshot
    assert inputs.work_plan is plan
    assert (
        inputs.work_plan.map_content_version,
        inputs.work_plan.topology_version,
        inputs.work_plan.parameter_version,
    ) == (1, 2, 3)
    assert runtime._latest_tick_work is work


def test_frozen_prepared_plan_precedes_mutable_tick_slots():
    import eligsir.runtime.causal_prepared_ready as prepared

    frozen = _Plan(object(), 1, 2, 3)
    mutable = _Plan(object(), 5, 6, 7)
    runtime = SimpleNamespace(
        _prepared_ready_plan_for_execution=lambda: frozen,
        _prepared_ready_frozen_work=None,
        _latest_tick_work=(object(), mutable),
        _latest_report_work=None,
        _last_materialized_work=None,
        initialized=True,
    )

    inputs = prepared._step_inputs_for_runtime(runtime)

    assert inputs.work_plan is frozen
    assert runtime._latest_tick_work[1] is mutable


def test_ready_path_never_mints_a_legacy_idle_plan_when_ready_plan_is_pending():
    import eligsir.runtime.causal_prepared_ready as prepared

    runtime = SimpleNamespace(
        _prepared_ready_plan_for_execution=lambda: None,
        _prepared_ready_frozen_work=None,
        _latest_tick_work=None,
        _latest_report_work=None,
        _last_materialized_work=(object(), object()),
        initialized=True,
        _fresh_idle_snapshot=lambda _snapshot: pytest.fail("legacy idle snapshot"),
        _idle_replay_plan=lambda _snapshot, _plan: pytest.fail("legacy idle plan"),
    )

    inputs = prepared._step_inputs_for_runtime(runtime)

    assert inputs.snapshot is None
    assert inputs.work_plan is None


def test_source_complete_after_initialization_never_runs_an_extra_optimizer_tick(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared
    from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult

    class Bridge:
        handoff = object()

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    phase_calls = []
    path = prepared.CausalPreparedReadyB4Path(
        runtime=SimpleNamespace(initialized=True, iteration=23),
        bridge=Bridge(),
        adapter_factory=lambda: (_ for _ in ()).throw(AssertionError("no EOF adapter")),
        step_inputs_factory=lambda: (_ for _ in ()).throw(AssertionError("no EOF inputs")),
        source_complete_phase_hook=lambda: phase_calls.append(True),
    )
    path._take_ingress_result = lambda: PreparedRuntimeBridgeResult(0, 0, 0, 0)
    outcome = path.mark_source_complete()
    assert outcome.executed is False
    assert outcome.state == "bootstrapped"
    assert outcome.iteration == 23
    assert phase_calls == [True]


def test_source_complete_leaves_delayed_bootstrap_retry_to_outer_owner(monkeypatch):
    import eligsir.runtime.causal_prepared_ready as prepared

    path = object.__new__(prepared.CausalPreparedReadyB4Path)
    path._runtime = SimpleNamespace(initialized=False)
    path._source_complete_phase_hook = None
    path._source_complete_phase_marked = False
    path._source_complete_requested = False
    calls = []
    pending = prepared.PreparedBootstrapOutcome(
        "bootstrap_pending",
        prepared.PreparedRuntimeBridgeResult(0, 0, 0, 0),
        prepared.ReadyOnlyTickSelection((), 0, 0, False, True),
    )
    monkeypatch.setattr(path, "_bootstrap_tick", lambda: calls.append(True) or pending)

    outcome = path.mark_source_complete()

    assert calls == [True]
    assert outcome.state == "bootstrap_pending"
    assert outcome.reason == "source_complete_preparation_pending"
    assert path._source_complete_requested is True


@pytest.mark.parametrize(
    ("resolution_tier", "minimum_tier", "expected_tiers"),
    (
        ("native", None, frozenset((ResolutionTier.NATIVE,))),
        (
            "half",
            None,
            frozenset((ResolutionTier.NATIVE, ResolutionTier.HALF)),
        ),
        (
            "quarter",
            None,
            frozenset((ResolutionTier.NATIVE, ResolutionTier.QUARTER)),
        ),
        (
            "adaptive",
            "quarter",
            frozenset(
                (
                    ResolutionTier.NATIVE,
                    ResolutionTier.HALF,
                    ResolutionTier.QUARTER,
                )
            ),
        ),
    ),
)
def test_bootstrap_policy_keeps_native_alongside_work_tiers(
    monkeypatch, resolution_tier, minimum_tier, expected_tiers
):
    """Bootstrap has native geometry while work retains its configured tiers."""
    import eligsir.runtime.causal_prepared_ready as prepared

    class Bridge:
        def __init__(self, handoff, producer, *, ready_owner, **kwargs):
            self.handoff = handoff
            self.ready_owner = ready_owner

        def drain(self):
            from eligsir.runtime.prepared_runtime_bridge import (
                PreparedRuntimeBridgeResult,
            )

            return PreparedRuntimeBridgeResult(0, 0, 0, 0)

        def close(self):
            pass

    class Runtime:
        initialized = False
        bootstrap_frames = 5

        def enable_ready_source_owner(self, _owner, *, warmup_complete):
            pass

        def retain_prepared_source(self, item, *, lane):
            return True

        def record_retained_prepared_commit(self, item):
            pass

        def clear_retained_prepared_commit(self, _item):
            pass

        def mark_source_complete(self):
            pass

    monkeypatch.setattr(prepared, "PreparedRuntimeBridge", Bridge)
    monkeypatch.setattr(
        prepared,
        "_transfer_owner_for_runtime",
        lambda _runtime: SimpleNamespace(prepare_source=lambda *_args, **_kwargs: None),
    )
    static_options = {
        "optimization_constellation_policy": "fixed",
        "mapping_resolution_tier": resolution_tier,
        "insert_from_depth": False,
        "prepared_ready_source_insertion": False,
    }
    if minimum_tier is not None:
        static_options["mapping_resolution_min_tier"] = minimum_tier
    path = prepared.build_causal_prepared_ready_path(
        runtime=Runtime(),
        config=SimpleNamespace(
            hard_limits={"optimization_batch_size": 4},
            static_options=static_options,
        ),
        args=object(),
    )
    try:
        policy = path._runtime._prepared_tier_policy
        assert policy is not None
        assert policy.declared_tiers == expected_tiers
        assert policy.highest() is ResolutionTier.NATIVE
    finally:
        path.close()
