from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.control import (
    DispositionKind,
    EvidenceFact,
    FreshnessMode,
    DepthInsertionPlan,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ObservationDisposition,
    ResidencyIntent,
    SupervisionWork,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.contracts.work_clocks import RuntimeWorkClockSnapshot
from eligsir.runtime.telemetry import (
    AsyncGpuEventTelemetryPublisher,
    GpuTelemetryDescriptor,
    GpuTransactionTelemetry,
    LatencySummary,
    MapStateTelemetry,
    QueueStateTelemetry,
    QueueTelemetry,
    RateTelemetry,
    ResidencyTelemetry,
    RuntimeLatencyTelemetry,
    RuntimeTelemetryAdapter,
    RuntimeTelemetryInputs,
    RuntimeTelemetrySnapshot,
    RuntimeTelemetryStore,
    TransactionType,
)
from eligsir.runtime.transfers import TransferAccounting


def _source(ordinal: int = 7) -> SourceIdentity:
    return SourceIdentity("camera", 2, ordinal, ordinal * 1_000_000)


def _disposition(snapshot_id: int = 11) -> ObservationDisposition:
    return ObservationDisposition(
        source=_source(),
        snapshot_id=snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="newest frontier",
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            rgb_quality="qualified",
            depth_supervision=True,
            depth_quality="qualified",
            depth_insertion=True,
        ),
    )


def _plan(snapshot_id: int = 11) -> MappingWorkPlan:
    return MappingWorkPlan(
        plan_id=f"snapshot-{snapshot_id}",
        snapshot_id=snapshot_id,
        control_version=1,
        supervision_sources=(_source(),),
        replay_sources=(),
        resolution_tier="full",
        optimizer_updates=1,
        insertion=DepthInsertionPlan(
            source=_source(),
            enabled=True,
            sample_stride=1,
            candidate_budget=8,
            seed_budget=8,
            voxel_size_m=0.02,
            minimum_confidence=0.0,
        ),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=4,
        topology_version=2,
        parameter_version=8,
        supervision_work=(
            SupervisionWork(_source(), "native", _disposition(snapshot_id).modality_eligibility),
        ),
        reason_trace=("healthy",),
    )


def _transfers() -> TransferAccounting:
    return TransferAccounting(
        h2d_ops=3,
        h2d_bytes=4096,
        d2h_ops=1,
        d2h_bytes=20,
        pool_waits=0,
        pool_drops=0,
        pinned_bytes=8192,
        peak_pinned_bytes=8192,
        inflight_limit=4,
        shape_buckets=1,
        shape_bucket_limit=8,
    )


def _gpu(transaction_id: int = 9) -> GpuTransactionTelemetry:
    return GpuTransactionTelemetry(
        transaction_id=transaction_id,
        snapshot_id=11,
        transaction_type=TransactionType.OPTIMIZATION,
        map_content_version=5,
        topology_version=2,
        parameter_version=9,
        batch_size=2,
        pixels=64 * 64 * 2,
        rasterizer_calls=1,
        optimizer_updates=1,
        candidates=17,
        duration_ms=3.25,
        mutation_type="none",
        inserted=0,
        pruned=0,
        relocated=0,
        corrected=0,
    )


def _snapshot(publication_id: int = 3) -> RuntimeTelemetrySnapshot:
    return RuntimeTelemetrySnapshot(
        publication_id=publication_id,
        snapshot_id=11,
        published_at_ns=50_000_000,
        source_clock_ns=45_000_000,
        admitted_clock_ns=43_000_000,
        map_clock_ns=42_000_000,
        work_clocks=RuntimeWorkClockSnapshot(
            control_tick_epoch=11,
            optimizer_updates=5,
            supervised_rgb_views=9,
            supervised_rgb_pixels=9 * 64 * 64,
            depth_valid_samples=17,
            source_frames_seen=12,
            admitted_observations=7,
            depth_mutation_epochs=3,
            mcmc_mutation_epochs=2,
            mapping_active_ns=8_000_000,
            source_complete=False,
        ),
        freshness=FreshnessMode.HEALTHY,
        rates=RateTelemetry(source_hz=30.0, admitted_hz=10.0, commit_hz=9.0),
        latencies=RuntimeLatencyTelemetry(
            source_to_first_supervision=LatencySummary(5, 2.0, 4.0, 5.0),
            source_to_geometry_commit=LatencySummary(4, 3.0, 5.0, 8.0),
        ),
        queues=QueueTelemetry(
            source=QueueStateTelemetry(1, 1.0),
            candidate=QueueStateTelemetry(2, 2.0),
            commit=QueueStateTelemetry(0, 0.0),
            transfer=QueueStateTelemetry(1, 0.5),
            gpu=QueueStateTelemetry(1, 0.25),
        ),
        observation=_disposition(),
        work_plan=_plan(),
        gpu_transaction=_gpu(),
        transfers=_transfers(),
        map_state=MapStateTelemetry(
            gaussian_count=500,
            map_content_version=5,
            topology_version=2,
            parameter_version=9,
        ),
        residency=ResidencyTelemetry(active=2, replay=1, history=8, hot_gpu=3),
    )


def test_runtime_snapshot_is_immutable_and_scalar_only() -> None:
    snapshot = _snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = 12
    with pytest.raises(FrozenInstanceError):
        snapshot.queues.source.depth = 3

    assert snapshot.observation.source == _source()
    assert snapshot.work_plan.supervision_sources == (_source(),)
    assert snapshot.gpu_transaction.compact().transaction_id == 9
    assert EvidenceFact("candidates", 17) in snapshot.gpu_transaction.compact().facts


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("observation", _disposition(12), "observation.*snapshot_id"),
        ("work_plan", _plan(12), "work_plan.*snapshot_id"),
        (
            "map_state",
            MapStateTelemetry(500, 3, 2, 9),
            "work plan map_content_version",
        ),
        (
            "gpu_transaction",
            replace(_gpu(), topology_version=3),
            "GPU topology_version",
        ),
    ],
)
def test_runtime_snapshot_rejects_incoherent_identity_and_version_joins(
    field: str, value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_snapshot(), **{field: value})


def test_stale_completed_gpu_packet_is_versioned_and_harmless() -> None:
    stale = replace(
        _gpu(),
        transaction_id=4,
        snapshot_id=8,
        map_content_version=3,
        topology_version=1,
        parameter_version=6,
    )
    snapshot = replace(_snapshot(), gpu_transaction=stale)

    assert snapshot.gpu_transaction.transaction_id == 4
    assert snapshot.gpu_transaction.topology_version < snapshot.map_state.topology_version


def test_publication_and_sensor_clocks_may_use_different_epochs() -> None:
    snapshot = replace(
        _snapshot(),
        published_at_ns=50,
        source_clock_ns=1_700_000_000_000_000_000,
        admitted_clock_ns=1_700_000_000_000_000_000,
        map_clock_ns=1_699_999_999_999_000_000,
    )

    assert snapshot.published_at_ns == 50


def test_latest_completed_store_is_capacity_one_and_readers_are_independent() -> None:
    store = RuntimeTelemetryStore()
    viewer = store.subscribe()
    controller = store.subscribe()

    assert store.publish(_snapshot(3))
    assert viewer.read_latest() == _snapshot(3)
    assert controller.read_latest() == _snapshot(3)
    assert viewer.read_latest(only_if_new=True) is None

    assert store.publish(_snapshot(4))
    assert viewer.read_latest(only_if_new=True) == _snapshot(4)
    assert controller.read_latest(only_if_new=True) == _snapshot(4)
    assert not store.publish(_snapshot(2))
    assert store.latest() == _snapshot(4)
    assert store.retained_count == 1


def test_disabled_store_is_a_semantics_free_sink() -> None:
    store = RuntimeTelemetryStore(enabled=False)
    disposition = _snapshot().observation
    plan = _snapshot().work_plan

    assert not store.publish(_snapshot())
    assert store.latest() is None
    assert _snapshot().observation == disposition
    assert _snapshot().work_plan == plan


def test_public_adapter_joins_only_tick_result_fields_and_publishes() -> None:
    adapter = RuntimeTelemetryAdapter()
    template = _snapshot()
    tick = SimpleNamespace(
        snapshot_id=11,
        dispositions=(_disposition(10), _disposition(11)),
        work_plan=_plan(11),
        freshness=FreshnessMode.HEALTHY,
        transfers=_transfers(),
        map_content_version=5,
        topology_version=2,
        parameter_version=9,
        work_clocks=template.work_clocks,
    )
    inputs = RuntimeTelemetryInputs(
        published_at_ns=template.published_at_ns,
        source_clock_ns=template.source_clock_ns,
        admitted_clock_ns=template.admitted_clock_ns,
        map_clock_ns=template.map_clock_ns,
        rates=template.rates,
        latencies=template.latencies,
        queues=template.queues,
        gaussian_count=template.map_state.gaussian_count,
        residency=template.residency,
    )

    published = adapter.publish_tick(tick, inputs, gpu_transaction=_gpu())

    assert published.observation == _disposition(11)
    assert published.work_plan == tick.work_plan
    assert published.map_state.parameter_version == 9
    assert published.work_clocks is tick.work_clocks
    assert adapter.store.latest() == published


def test_runtime_package_exports_the_public_telemetry_seam() -> None:
    from eligsir import runtime

    assert runtime.RuntimeTelemetryAdapter is RuntimeTelemetryAdapter
    assert runtime.RuntimeTelemetrySnapshot is RuntimeTelemetrySnapshot


def _descriptor(transaction_id: int) -> GpuTelemetryDescriptor:
    return GpuTelemetryDescriptor(
        transaction_id=transaction_id,
        snapshot_id=transaction_id,
        transaction_type=TransactionType.OPTIMIZATION,
        map_content_version=transaction_id,
        topology_version=2,
        parameter_version=transaction_id,
        batch_size=2,
        pixels=8192,
        rasterizer_calls=1,
        optimizer_updates=1,
        candidates=17,
        mutation_type="none",
    )


class _TimingEvent:
    def __init__(self, *, complete: bool = False, elapsed_ms: float = 0.0) -> None:
        self.complete = complete
        self.elapsed_ms = elapsed_ms
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.complete

    def elapsed_time(self, end) -> float:
        assert end.complete
        return end.elapsed_ms


def test_event_gpu_publication_waits_for_completion_without_synchronizing() -> None:
    publisher = AsyncGpuEventTelemetryPublisher()
    start = _TimingEvent()
    end = _TimingEvent(elapsed_ms=2.75)

    publisher.submit(start, end, _descriptor(10), inserted=3)

    assert publisher.poll_completed() is None
    assert publisher.latest_completed() is None
    end.complete = True
    completed = publisher.poll_completed()

    assert completed is not None
    assert completed.duration_ms == 2.75
    assert completed.inserted == 3
    assert completed.transaction_type is TransactionType.OPTIMIZATION


def test_event_gpu_publication_retains_truthful_pending_depth() -> None:
    publisher = AsyncGpuEventTelemetryPublisher()
    first_end = _TimingEvent(elapsed_ms=1.0)
    second_end = _TimingEvent(elapsed_ms=2.0)
    third_end = _TimingEvent(elapsed_ms=3.0)

    publisher.submit(_TimingEvent(), first_end, _descriptor(1))
    publisher.submit(_TimingEvent(), second_end, _descriptor(2))
    publisher.submit(_TimingEvent(), third_end, _descriptor(3))

    assert publisher.pending_count == 3
    assert publisher.staged_replacements == 1
    first_end.complete = True
    assert publisher.poll_completed().transaction_id == 1
    third_end.complete = True
    assert publisher.poll_completed().transaction_id == 3
    assert publisher.pending_count == 0


def test_event_gpu_publication_discards_invalid_driver_duration() -> None:
    publisher = AsyncGpuEventTelemetryPublisher()
    end = _TimingEvent(elapsed_ms=float("nan"))

    publisher.submit(_TimingEvent(), end, _descriptor(1))
    end.complete = True

    assert publisher.poll_completed() is None
    assert publisher.latest_completed() is None
    assert publisher.pending_count == 0
    assert publisher.invalid_completed_timings == 1
