from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest

from eligsir.contracts.control import (
    CompactGpuTelemetry,
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DispositionKind,
    EvidenceFact,
    FreshnessMode,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
)
from eligsir.runtime.config import CanonicalRuntimePolicy, MappingRuntimeConfig
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.telemetry import (
    GpuTelemetryDescriptor,
    GpuTransactionTelemetry,
    TransactionType,
)
from eligsir.cli.parser import parse_training_args
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.batch_contract import validate_supervision_camera_count
from eligsir.mapping.scheduler.work_budget import (
    _AdaptiveConstellationPolicy,
    WorkBudgetOwner,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity(
        source_id="camera",
        stream_epoch=1,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )


def _snapshot(
    *,
    replay_count: int,
    mode: FreshnessMode = FreshnessMode.HEALTHY,
    epoch: int = 0,
    supervision_views_completed: int = 0,
    evidence: tuple[EvidenceFact, ...] = (),
    transaction_budget_ns: int | None = None,
    gpu_transaction: CompactGpuTelemetry | None = None,
) -> ControlSnapshot:
    current = _source(replay_count + 1)
    replay = tuple(_source(i + 1) for i in range(replay_count))
    modality = ModalityEligibility(True, "qualified", False, "missing", False)
    return ControlSnapshot(
        snapshot_id=epoch + 1,
        control_version=1,
        source=current,
        tracking=None,
        pending_sources=(current,),
        newest_frontier=current,
        freshness_mode=mode,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        gaussian_count=10,
        optimizer_steps_completed=epoch,
        supervision_views_completed=supervision_views_completed,
        control_tick_epoch=epoch,
        limits=ControlLimits(
            max_supervision_slots=256,
            max_replay_slots=256,
            max_optimizer_updates=1,
            max_insertion_seeds=0,
            max_mutations=0,
            max_transfer_bytes=1,
            max_optimization_transaction_ns=transaction_budget_ns,
        ),
        supervision_candidates=(current,),
        replay_candidates=replay,
        replay_work_modalities=(modality,) * replay_count,
        latest_gpu_telemetry=gpu_transaction,
        evidence=evidence,
    )


def _completed_optimization_transaction(
    transaction_id: int, *, duration_ns: int
) -> CompactGpuTelemetry:
    return CompactGpuTelemetry(
        transaction_id=transaction_id,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        facts=(
            EvidenceFact("optimization_transaction_duration_ns", duration_ns),
            EvidenceFact("optimizer_updates", 1),
        ),
    )


def _disposition(snapshot: ControlSnapshot) -> ObservationDisposition:
    return ObservationDisposition(
        source=snapshot.source,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="qualified",
        modality_eligibility=ModalityEligibility(
            True, "qualified", False, "missing", False
        ),
    )


def _proposals(batch_size: int) -> tuple[ControlProposal, ...]:
    return (
        ControlProposal("test", "supervision_slots", 1, "current frontier"),
        ControlProposal("test", "replay_slots", max(0, batch_size - 1), "replay fill"),
        ControlProposal("test", "optimizer_updates", 1, "one transaction"),
    )


def _resolve(owner: WorkBudgetOwner, snapshot: ControlSnapshot):
    return owner.resolve(
        snapshot,
        disposition=_disposition(snapshot),
        proposals=_proposals(owner.requested_batch_size),
    )


def test_startup_underfill_realizes_c1_immediately_without_waiting_or_duplication():
    plan = _resolve(WorkBudgetOwner(requested_batch_size=256), _snapshot(replay_count=0))

    constellation = plan.supervision_constellation
    assert constellation.requested_batch_size == 256
    assert constellation.requested_camera_count == 64
    assert constellation.realized_batch_size == 1
    assert constellation.realized_camera_count == 1
    assert constellation.unique_eligible_count == 1
    assert constellation.microbatch_count == 1
    assert constellation.fill_ratio == pytest.approx(1 / 256)
    assert constellation.mode == "single_camera_immediate"
    assert tuple((*plan.supervision_sources, *plan.replay_sources)) == (plan.supervision_sources[0],)


def test_hardware_capped_startup_underfill_still_runs_the_one_available_camera():
    plan = _resolve(
        WorkBudgetOwner(requested_batch_size=64, maximum_camera_count=16),
        _snapshot(replay_count=0),
    )

    constellation = plan.supervision_constellation
    assert constellation.requested_batch_size == 64
    assert constellation.requested_camera_count == 16
    assert constellation.realized_batch_size == 1
    assert constellation.realized_camera_count == 1
    assert constellation.mode == "single_camera_immediate"


def test_b_greater_than_c_is_four_true_c_microbatches_in_one_transaction():
    plan = _resolve(WorkBudgetOwner(requested_batch_size=256), _snapshot(replay_count=255))

    constellation = plan.supervision_constellation
    assert constellation.realized_batch_size == 256
    assert constellation.realized_camera_count == 64
    assert constellation.microbatch_sizes == (64, 64, 64, 64)
    assert constellation.microbatch_count == 4
    identities = (*plan.supervision_sources, *plan.replay_sources)
    assert len(identities) == len(set(identities)) == 256


def test_b3_executes_as_qualified_c2_plus_c1_microbatches():
    plan = _resolve(
        WorkBudgetOwner(
            requested_batch_size=3,
            maximum_camera_count=2,
            constellation_policy="fixed",
        ),
        _snapshot(replay_count=2),
    )

    constellation = plan.supervision_constellation
    assert constellation.requested_batch_size == 3
    assert constellation.requested_camera_count == 2
    assert constellation.realized_batch_size == 3
    assert constellation.realized_camera_count == 2
    assert constellation.microbatch_sizes == (2, 1)
    assert constellation.microbatch_count == 2
    identities = (*plan.supervision_sources, *plan.replay_sources)
    assert len(identities) == len(set(identities)) == 3


def test_underfilled_batch_never_duplicates_camera_identity_to_reach_b_or_c():
    plan = _resolve(WorkBudgetOwner(requested_batch_size=256), _snapshot(replay_count=2))

    identities = (*plan.supervision_sources, *plan.replay_sources)
    assert len(identities) == len(set(identities)) == 3
    assert plan.supervision_constellation.realized_batch_size == 3
    assert plan.supervision_constellation.realized_camera_count == 3
    assert plan.supervision_constellation.microbatch_sizes == (3,)


def test_fixed_constellation_ablation_ignores_latency_and_freshness_shrink():
    owner = WorkBudgetOwner(
        requested_batch_size=4,
        maximum_camera_count=4,
        constellation_policy="fixed",
    )
    snapshot = _snapshot(
        replay_count=3,
        mode=FreshnessMode.CRITICAL,
        epoch=7,
        transaction_budget_ns=10_000_000,
        gpu_transaction=_completed_optimization_transaction(
            7, duration_ns=100_000_000
        ),
        evidence=(EvidenceFact("new_region", True),),
    )

    plan = _resolve(owner, snapshot)

    constellation = plan.supervision_constellation
    assert constellation.requested_batch_size == 4
    assert constellation.requested_camera_count == 4
    assert constellation.realized_batch_size == 4
    assert constellation.realized_camera_count == 4
    assert constellation.microbatch_sizes == (4,)
    assert constellation.reason == "fixed_constellation_ablation"
    assert tuple((*plan.supervision_sources, *plan.replay_sources))
    assert len(set((*plan.supervision_sources, *plan.replay_sources))) == 4


def test_fast_shrink_slow_grow_and_fixed_control_barrier():
    controller = _AdaptiveConstellationPolicy(requested_batch_size=64, grow_dwell=3)
    healthy = _snapshot(replay_count=63, epoch=0)
    configured = controller.decide(healthy, stable_replay=False)
    assert configured.requested_batch_size == 64
    assert configured.requested_camera_count == 64

    pressured = replace(healthy, snapshot_id=2, control_tick_epoch=1,
                        freshness_mode=FreshnessMode.CRITICAL)
    shrunk = controller.decide(pressured, stable_replay=False)
    assert shrunk.requested_batch_size == 2
    assert shrunk.requested_camera_count == 1
    assert shrunk.reason == "critical_backlog_pressure"

    # The same optimizer/control snapshot barrier cannot change constellation.
    same_barrier = replace(pressured, snapshot_id=3,
                           freshness_mode=FreshnessMode.HEALTHY)
    assert controller.decide(same_barrier, stable_replay=True) == shrunk

    for epoch in (2, 3):
        held = controller.decide(
            replace(healthy, snapshot_id=epoch + 2, control_tick_epoch=epoch),
            stable_replay=True,
        )
        assert held.requested_batch_size == 2
        assert held.requested_camera_count == 1
        assert held.hysteresis_state == f"stable_grow_{epoch - 1}/3"
    grown = controller.decide(
        replace(healthy, snapshot_id=5, control_tick_epoch=4),
        stable_replay=True,
    )
    assert grown.requested_batch_size == 4
    assert grown.requested_camera_count == 2
    assert grown.reason == "stable_replay_slow_grow"


def test_worsening_evidence_shrinks_within_one_control_epoch_without_recovery():
    controller = _AdaptiveConstellationPolicy(requested_batch_size=64, grow_dwell=1)
    healthy = _snapshot(replay_count=63, epoch=4)

    configured = controller.decide(healthy, stable_replay=True)
    critical = controller.decide(
        replace(
            healthy,
            snapshot_id=healthy.snapshot_id + 1,
            freshness_mode=FreshnessMode.CRITICAL,
        ),
        stable_replay=False,
    )
    recovered = controller.decide(
        replace(healthy, snapshot_id=healthy.snapshot_id + 2),
        stable_replay=True,
    )

    assert configured.requested_batch_size == 64
    assert critical.requested_batch_size == 2
    assert critical.requested_camera_count == 1
    assert critical.reason == "critical_backlog_pressure"
    assert recovered == critical


def test_stable_replay_does_not_reapply_stale_new_region_frame_evidence():
    controller = _AdaptiveConstellationPolicy(requested_batch_size=8, grow_dwell=1)
    new_region = _snapshot(
        replay_count=7,
        epoch=1,
        evidence=(EvidenceFact("new_region", True),),
    )

    shrunk = controller.decide(new_region, stable_replay=False)
    replay = controller.decide(
        replace(new_region, snapshot_id=3, control_tick_epoch=2),
        stable_replay=True,
    )

    assert shrunk.requested_batch_size == 1
    assert replay.requested_batch_size == 2
    assert replay.reason == "stable_replay_slow_grow"


@pytest.mark.parametrize("fact", ("new_region", "partition_fade", "quality_risk"))
def test_risk_evidence_shrinks_immediately(fact: str):
    controller = _AdaptiveConstellationPolicy(requested_batch_size=64)
    risky = _snapshot(
        replay_count=63,
        epoch=1,
        evidence=(EvidenceFact(fact, True),),
    )

    decision = controller.decide(risky, stable_replay=False)

    assert decision.requested_batch_size == 1
    assert decision.requested_camera_count == 1
    assert decision.hysteresis_state == "risk_shrunk"
    assert fact in decision.reason


def test_critical_backlog_preserves_a_replay_lane_and_publishes_telemetry():
    snapshot = _snapshot(replay_count=3, mode=FreshnessMode.CRITICAL, epoch=1)
    owner = WorkBudgetOwner(requested_batch_size=4)

    plan = _resolve(owner, snapshot)

    assert plan.replay_sources
    assert "critical_bounded_replay_preserved" in plan.reason_trace
    telemetry = plan.supervision_constellation
    assert telemetry.requested_batch_size == 2
    assert telemetry.requested_camera_count == 1
    assert telemetry.unique_eligible_count == 4
    assert telemetry.realized_batch_size == 2
    assert telemetry.realized_camera_count == 1
    assert telemetry.microbatch_count == 2
    assert telemetry.mode == "sequential_accumulation"
    assert telemetry.reason == "critical_backlog_pressure"
    assert telemetry.hysteresis_state == "risk_shrunk"


def test_true_camera_request_outside_qualified_ladder_fails_closed():
    with pytest.raises(ValueError, match="qualified supervision camera count"):
        validate_supervision_camera_count(6)


def test_decision_telemetry_exposes_completed_dual_work_clocks():
    plan = _resolve(
        WorkBudgetOwner(requested_batch_size=8),
        _snapshot(
            replay_count=7,
            epoch=11,
            supervision_views_completed=123,
        ),
    )

    telemetry = plan.supervision_constellation
    assert telemetry.cumulative_realized_supervised_views == 123
    assert telemetry.cumulative_optimizer_updates == 11


def test_live_budget_starts_at_conservative_c_without_waiting_for_fill():
    plan = _resolve(
        WorkBudgetOwner(requested_batch_size=64),
        _snapshot(
            replay_count=7,
            transaction_budget_ns=10_000_000,
        ),
    )

    constellation = plan.supervision_constellation
    assert constellation.requested_batch_size == 1
    assert constellation.requested_camera_count == 1
    assert constellation.realized_batch_size == 1
    assert constellation.realized_camera_count == 1
    assert constellation.unique_eligible_count == 8
    assert constellation.max_optimization_transaction_ns == 10_000_000
    assert constellation.latest_completed_optimization_transaction_ns is None
    assert constellation.p95_optimization_transaction_ns is None
    assert constellation.reason == "latency_unknown_conservative_start"


def test_new_over_budget_latency_shrinks_once_and_stale_packet_cannot_cascade():
    controller = _AdaptiveConstellationPolicy(
        requested_batch_size=64,
        grow_dwell=2,
        latency_window_size=4,
    )
    configured = controller.decide(
        _snapshot(replay_count=63, epoch=0), stable_replay=True
    )
    assert configured.requested_camera_count == 64

    over = controller.decide(
        _snapshot(
            replay_count=63,
            epoch=1,
            transaction_budget_ns=10_000_000,
            gpu_transaction=_completed_optimization_transaction(
                4, duration_ns=12_000_000
            ),
        ),
        stable_replay=True,
    )
    stale = controller.decide(
        _snapshot(
            replay_count=63,
            epoch=2,
            transaction_budget_ns=10_000_000,
            gpu_transaction=_completed_optimization_transaction(
                4, duration_ns=12_000_000
            ),
        ),
        stable_replay=True,
    )

    assert over.requested_camera_count < configured.requested_camera_count
    assert stale.requested_camera_count == over.requested_camera_count
    assert over.latest_completed_transaction_id == 4
    assert over.latest_completed_optimization_transaction_ns == 12_000_000
    assert over.p95_optimization_transaction_ns == 12_000_000
    assert over.reason == "optimization_latency_over_budget"


def test_latency_recovery_requires_fresh_headroom_samples_and_existing_hysteresis():
    controller = _AdaptiveConstellationPolicy(
        requested_batch_size=8,
        grow_dwell=2,
        latency_window_size=4,
    )
    controller.decide(_snapshot(replay_count=7, epoch=0), stable_replay=True)
    shrunk = controller.decide(
        _snapshot(
            replay_count=7,
            epoch=1,
            transaction_budget_ns=10_000_000,
            gpu_transaction=_completed_optimization_transaction(
                1, duration_ns=12_000_000
            ),
        ),
        stable_replay=True,
    )

    decisions = []
    for transaction_id, epoch in zip(range(2, 7), range(2, 7), strict=True):
        decisions.append(
            controller.decide(
                _snapshot(
                    replay_count=7,
                    epoch=epoch,
                    transaction_budget_ns=10_000_000,
                    gpu_transaction=_completed_optimization_transaction(
                        transaction_id, duration_ns=5_000_000
                    ),
                ),
                stable_replay=True,
            )
        )

    assert [decision.requested_camera_count for decision in decisions] == [
        2,
        1,
        1,
        1,
        2,
    ]
    assert decisions[-1].requested_camera_count > decisions[-2].requested_camera_count
    assert decisions[-1].reason == "stable_replay_slow_grow"
    assert decisions[-1].p95_optimization_transaction_ns == 5_000_000


def test_live_work_owner_grows_after_headroom_without_waiting_for_idle_replay():
    owner = WorkBudgetOwner(requested_batch_size=8)
    start = _resolve(
        owner,
        _snapshot(replay_count=7, epoch=0, transaction_budget_ns=10_000_000),
    )

    decisions = [start]
    for transaction_id in range(1, 4):
        decisions.append(
            _resolve(
                owner,
                _snapshot(
                    replay_count=7,
                    epoch=transaction_id,
                    transaction_budget_ns=10_000_000,
                    gpu_transaction=_completed_optimization_transaction(
                        transaction_id, duration_ns=5_000_000
                    ),
                ),
            )
        )

    assert [
        plan.supervision_constellation.requested_camera_count for plan in decisions
    ] == [1, 1, 1, 2]
    assert decisions[-1].supervision_constellation.reason == "latency_headroom_slow_grow"


@pytest.mark.parametrize("duration_ns", [0, -1, True, 1.5])
def test_completed_optimization_transaction_duration_must_be_positive_integer(
    duration_ns,
):
    controller = _AdaptiveConstellationPolicy(requested_batch_size=4)

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        controller.decide(
            _snapshot(
                replay_count=3,
                epoch=1,
                transaction_budget_ns=10_000_000,
                gpu_transaction=_completed_optimization_transaction(
                    1, duration_ns=duration_ns
                ),
            ),
            stable_replay=True,
        )


def test_completed_optimization_gpu_envelope_publishes_typed_transaction_duration(
    monkeypatch,
):
    from eligsir.runtime import production_application

    monkeypatch.setattr(
        production_application, "TransactionType", TransactionType, raising=False
    )
    monkeypatch.setattr(
        production_application,
        "GpuTransactionTelemetry",
        GpuTransactionTelemetry,
        raising=False,
    )
    completed = GpuTransactionTelemetry(
        transaction_id=7,
        snapshot_id=3,
        transaction_type=TransactionType.OPTIMIZATION,
        map_content_version=4,
        topology_version=5,
        parameter_version=6,
        batch_size=4,
        pixels=400,
        rasterizer_calls=1,
        optimizer_updates=1,
        candidates=0,
        duration_ms=12.25,
        mutation_type="none",
        inserted=0,
        pruned=0,
        relocated=0,
        corrected=0,
    )

    class _NoGenericCompletion:
        def submit(self, *_args, **_kwargs):
            return None

        def poll_completed(self):
            return None

    class _StartEvent:
        def elapsed_time(self, _end):
            return completed.duration_ms

    class _EndEvent:
        def query(self):
            return True

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._gpu_event_telemetry = _NoGenericCompletion()
    runtime._latest_gpu_transaction = None
    runtime._latest_optimization_gpu_transaction = None
    runtime._latest_optimization_gpu_ms = None
    runtime._latest_completed_optimization_transaction = None
    runtime._pending_optimization_transaction_timings = deque()
    runtime._submit_gpu_timing(
        _StartEvent(),
        _EndEvent(),
        GpuTelemetryDescriptor(
            transaction_id=7,
            snapshot_id=3,
            transaction_type=TransactionType.OPTIMIZATION,
            map_content_version=4,
            topology_version=5,
            parameter_version=6,
            batch_size=4,
            pixels=400,
            rasterizer_calls=1,
            optimizer_updates=1,
            candidates=0,
            mutation_type="none",
        ),
        optimization_host_prefix_ns=2_750_000,
    )

    runtime._poll_gpu_telemetry()

    packet = runtime._latest_completed_optimization_transaction
    assert packet is not None
    assert packet.transaction_id == 7
    assert dict((fact.name, fact.value) for fact in packet.facts)[
        "optimization_transaction_duration_ns"
    ] == 15_000_000


def test_completed_optimization_gpu_envelope_reads_rocm_events_exactly_once(
    monkeypatch,
):
    """The ordered control lane is the sole owner of optimization timing.

    ROCm does not guarantee that repeated ``elapsed_time`` calls on one event
    pair return identical values.  The generic latest-wins lane must therefore
    not measure the same optimization pair a second time.
    """

    from eligsir.runtime import production_application
    from eligsir.runtime.telemetry import AsyncGpuEventTelemetryPublisher

    monkeypatch.setattr(
        production_application, "TransactionType", TransactionType, raising=False
    )
    monkeypatch.setattr(
        production_application,
        "GpuTransactionTelemetry",
        GpuTransactionTelemetry,
        raising=False,
    )

    class _StartEvent:
        def __init__(self):
            self.elapsed_calls = 0

        def elapsed_time(self, _end):
            self.elapsed_calls += 1
            if self.elapsed_calls > 1:
                return -1.0
            return 2.0

    class _EndEvent:
        def query(self):
            return True

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._gpu_event_telemetry = AsyncGpuEventTelemetryPublisher()
    runtime._latest_gpu_transaction = None
    runtime._latest_optimization_gpu_transaction = None
    runtime._latest_optimization_gpu_ms = None
    runtime._latest_completed_optimization_transaction = None
    runtime._pending_optimization_transaction_timings = deque()
    runtime._invalid_optimization_gpu_timings = 0
    start = _StartEvent()
    runtime._submit_gpu_timing(
        start,
        _EndEvent(),
        GpuTelemetryDescriptor(
            transaction_id=8,
            snapshot_id=3,
            transaction_type=TransactionType.OPTIMIZATION,
            map_content_version=4,
            topology_version=5,
            parameter_version=6,
            batch_size=1,
            pixels=400,
            rasterizer_calls=1,
            optimizer_updates=1,
            candidates=0,
            mutation_type="none",
        ),
        optimization_host_prefix_ns=1_000_000,
    )

    runtime._poll_gpu_telemetry()

    assert start.elapsed_calls == 1
    assert runtime._latest_gpu_transaction is not None
    assert runtime._latest_gpu_transaction.duration_ms == 2.0
    assert runtime._gpu_event_telemetry.pending_count == 0


def test_invalid_completed_optimization_gpu_timing_is_counted_not_published(
    monkeypatch,
):
    from eligsir.runtime import production_application
    from eligsir.runtime.telemetry import AsyncGpuEventTelemetryPublisher

    monkeypatch.setattr(
        production_application, "TransactionType", TransactionType, raising=False
    )
    monkeypatch.setattr(
        production_application,
        "GpuTransactionTelemetry",
        GpuTransactionTelemetry,
        raising=False,
    )

    class _StartEvent:
        def elapsed_time(self, _end):
            return float("nan")

    class _EndEvent:
        def query(self):
            return True

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._gpu_event_telemetry = AsyncGpuEventTelemetryPublisher()
    runtime._latest_gpu_transaction = None
    runtime._latest_optimization_gpu_transaction = None
    runtime._latest_optimization_gpu_ms = None
    runtime._latest_completed_optimization_transaction = None
    runtime._pending_optimization_transaction_timings = deque()
    runtime._invalid_optimization_gpu_timings = 0
    runtime._submit_gpu_timing(
        _StartEvent(),
        _EndEvent(),
        GpuTelemetryDescriptor(
            transaction_id=9,
            snapshot_id=3,
            transaction_type=TransactionType.OPTIMIZATION,
            map_content_version=4,
            topology_version=5,
            parameter_version=6,
            batch_size=1,
            pixels=400,
            rasterizer_calls=1,
            optimizer_updates=1,
            candidates=0,
            mutation_type="none",
        ),
        optimization_host_prefix_ns=1_000_000,
    )

    runtime._poll_gpu_telemetry()

    assert runtime._invalid_optimization_gpu_timings == 1
    assert runtime._latest_gpu_transaction is None
    assert runtime._latest_completed_optimization_transaction is None


def test_host_prefix_closes_before_gpu_envelope_is_recorded(monkeypatch):
    from eligsir.runtime import production_application

    ordering = []

    def _host_prefix_endpoint():
        ordering.append("host_prefix_endpoint")
        return 5_000

    class _StartEvent:
        def record(self):
            ordering.append("gpu_start_event")

    monkeypatch.setattr(
        production_application.time, "monotonic_ns", _host_prefix_endpoint
    )

    prefix_ns = CanonicalMappingRuntime._begin_optimization_gpu_envelope(
        _StartEvent(), transaction_started_ns=2_000
    )

    assert prefix_ns == 3_000
    assert ordering == ["host_prefix_endpoint", "gpu_start_event"]


def test_gpu_kernel_duration_is_not_consumed_as_transaction_freshness():
    kernel_only = CompactGpuTelemetry(
        transaction_id=8,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        facts=(EvidenceFact("gpu_duration_ms", 12.0),),
    )

    decision = _AdaptiveConstellationPolicy(requested_batch_size=8).decide(
        _snapshot(
            replay_count=7,
            transaction_budget_ns=10_000_000,
            gpu_transaction=kernel_only,
        ),
        stable_replay=True,
    )

    assert decision.latest_completed_transaction_id is None
    assert decision.reason == "latency_unknown_conservative_start"


def test_transaction_freshness_budget_is_typed_and_live_mode_requires_it():
    assert CanonicalRuntimePolicy().max_optimization_transaction_ns is None
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError), match="max_optimization_transaction_ns"):
            CanonicalRuntimePolicy(max_optimization_transaction_ns=invalid)

    with pytest.raises(ValueError, match="live.*transaction freshness budget"):
        MappingRuntimeConfig(
            model_layout="gsplat",
            renderer="gsplat",
            numerical_mode="selective_adam",
            static_options={"live_mode": True},
        )

    live = MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="selective_adam",
        runtime=CanonicalRuntimePolicy(
            max_optimization_transaction_ns=20_000_000
        ),
        static_options={"live_mode": True},
    )
    assert live.runtime.max_optimization_transaction_ns == 20_000_000


def test_config_boundary_sets_budget_for_ros_or_online_and_none_for_offline():
    args, _ = parse_training_args(["-s", "/dev/null"])
    args.model_path = "output/task2-config-test"

    offline = mapping_runtime_config_from_args(args)
    args.online_mode = True
    paced = mapping_runtime_config_from_args(args)
    args.online_mode = False
    args.source_path = "ros2://live"
    ros = mapping_runtime_config_from_args(args)

    assert offline.runtime.max_optimization_transaction_ns is None
    assert paced.runtime.max_optimization_transaction_ns == 33_333_333
    assert ros.runtime.max_optimization_transaction_ns == 33_333_333
