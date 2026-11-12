from __future__ import annotations

from collections import deque
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from eligsir.contracts.work_clocks import (
    RuntimeWorkClockOwner,
    RuntimeWorkClockSnapshot,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.application import (
    ComposedMappingApplication,
    MappingApplicationOptions,
)
from eligsir.runtime.telemetry import (
    LatencySummary,
    MapStateTelemetry,
    QueueStateTelemetry,
    QueueTelemetry,
    RateTelemetry,
    ResidencyTelemetry,
    RuntimeLatencyTelemetry,
    RuntimeTelemetryAdapter,
    RuntimeTelemetryInputs,
)


def test_initial_snapshot_has_exact_zeroed_public_contract() -> None:
    snapshot = RuntimeWorkClockOwner().snapshot()

    assert tuple(field.name for field in fields(snapshot)) == (
        "control_tick_epoch",
        "optimizer_updates",
        "supervised_rgb_views",
        "supervised_rgb_pixels",
        "depth_valid_samples",
        "source_frames_seen",
        "admitted_observations",
        "depth_mutation_epochs",
        "mcmc_mutation_epochs",
        "mapping_active_ns",
        "source_complete",
    )
    assert snapshot == RuntimeWorkClockSnapshot(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)


def test_every_committed_event_advances_its_exact_cumulative_field() -> None:
    owner = RuntimeWorkClockOwner()

    assert owner.record_control_tick().control_tick_epoch == 1
    assert owner.record_source_frame().source_frames_seen == 1
    assert owner.record_admission().admitted_observations == 1
    optimized = owner.record_optimization(
        optimizer_updates=2,
        supervised_views=8,
        supervised_pixels=8 * 640 * 480,
        mapping_active_ns=2_000_000,
    )
    assert optimized.optimizer_updates == 2
    assert optimized.supervised_rgb_views == 8
    assert optimized.supervised_rgb_pixels == 2_457_600
    assert optimized.mapping_active_ns == 2_000_000
    depth = owner.record_depth_supervision(depth_valid_samples=37)
    assert depth.depth_valid_samples == 37
    assert depth.depth_mutation_epochs == 0
    assert owner.record_depth_mutation().depth_mutation_epochs == 1
    assert owner.record_mcmc_mutation().mcmc_mutation_epochs == 1
    assert owner.mark_source_complete().source_complete is True


def test_counters_accumulate_monotonically_and_earlier_snapshots_do_not_change() -> None:
    owner = RuntimeWorkClockOwner()
    earlier = owner.record_optimization(
        optimizer_updates=1,
        supervised_views=3,
        supervised_pixels=300,
        mapping_active_ns=11,
    )
    later = owner.record_optimization(
        optimizer_updates=2,
        supervised_views=5,
        supervised_pixels=700,
        mapping_active_ns=13,
    )

    assert earlier == RuntimeWorkClockSnapshot(0, 1, 3, 300, 0, 0, 0, 0, 0, 11, False)
    assert later == RuntimeWorkClockSnapshot(0, 3, 8, 1_000, 0, 0, 0, 0, 0, 24, False)
    with pytest.raises(FrozenInstanceError):
        earlier.optimizer_updates = 99


def test_independent_snapshot_reads_are_equal_non_aliasing_values() -> None:
    owner = RuntimeWorkClockOwner()
    owner.record_control_tick()

    first = owner.snapshot()
    second = owner.snapshot()

    assert first == second
    assert first is not second


@pytest.mark.parametrize("invalid", [True, False, -1, 1.0, "1", None])
@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        (
            "record_optimization",
            {
                "optimizer_updates": 0,
                "supervised_views": 0,
                "supervised_pixels": 0,
                "mapping_active_ns": 0,
            },
        ),
        ("record_depth_supervision", {"depth_valid_samples": 0}),
        ("record_mapping_active", {"mapping_active_ns": 0}),
    ],
)
def test_numeric_inputs_reject_non_exact_or_negative_integers(
    invalid: object, method: str, kwargs: dict[str, object]
) -> None:
    owner = RuntimeWorkClockOwner()

    for name in kwargs:
        invalid_kwargs = dict(kwargs)
        invalid_kwargs[name] = invalid
        with pytest.raises((TypeError, ValueError), match=name):
            getattr(owner, method)(**invalid_kwargs)


@pytest.mark.parametrize(
    ("optimizer_updates", "supervised_views", "supervised_pixels", "mapping_active_ns", "match"),
    [
        (1, 1, 0, 1, "pixel"),
        (1, 0, 1, 1, "view"),
        (0, 1, 1, 0, "optimizer"),
        (0, 0, 0, 1, "optimizer"),
    ],
)
def test_optimization_rejects_inconsistent_or_uncommitted_work(
    optimizer_updates: int,
    supervised_views: int,
    supervised_pixels: int,
    mapping_active_ns: int,
    match: str,
) -> None:
    owner = RuntimeWorkClockOwner()

    with pytest.raises(ValueError, match=match):
        owner.record_optimization(
            optimizer_updates=optimizer_updates,
            supervised_views=supervised_views,
            supervised_pixels=supervised_pixels,
            mapping_active_ns=mapping_active_ns,
        )


def test_zero_work_event_does_not_advance_exposure_or_active_time() -> None:
    owner = RuntimeWorkClockOwner()
    before = owner.snapshot()

    after = owner.record_optimization(
        optimizer_updates=0,
        supervised_views=0,
        supervised_pixels=0,
        mapping_active_ns=0,
    )

    assert after == before


def test_mapping_active_time_advances_for_active_work_without_an_optimizer_update() -> None:
    owner = RuntimeWorkClockOwner()

    snapshot = owner.record_mapping_active(17)

    assert snapshot.mapping_active_ns == 17
    assert snapshot.optimizer_updates == 0


def test_source_completion_is_a_one_way_idempotent_latch() -> None:
    owner = RuntimeWorkClockOwner()

    first = owner.mark_source_complete()
    second = owner.mark_source_complete()

    assert first.source_complete is True
    assert second == first
    assert owner.snapshot().source_complete is True


def test_adaptive_completed_result_advances_only_actual_work_not_requested_bc() -> None:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    completed = SimpleNamespace(
        optimizer_step_count=1,
        supervision_view_count=7,
        supervised_pixel_count=7 * 320 * 240,
        depth_valid_sample_count=113,
        requested_batch_size=256,
        requested_camera_count=64,
        batch_size=256,
        microbatch_sizes=(4, 3),
    )
    padding_and_failed_lanes = SimpleNamespace(
        optimizer_step_count=0,
        supervision_view_count=0,
        supervised_pixel_count=0,
        depth_valid_sample_count=0,
        requested_batch_size=256,
        requested_camera_count=64,
        batch_size=249,
        microbatch_sizes=(),
    )

    first = runtime._record_completed_optimization_result(
        completed, mapping_active_ns=19
    )
    snapshot = runtime._record_completed_optimization_result(
        padding_and_failed_lanes, mapping_active_ns=7
    )

    assert first.optimizer_updates == 1
    assert snapshot.optimizer_updates == 1
    assert snapshot.supervised_rgb_views == 7
    assert snapshot.supervised_rgb_pixels == 537_600
    assert snapshot.depth_valid_samples == 113
    assert snapshot.mapping_active_ns == 26


@pytest.mark.parametrize("chunks", ([1] * 16, [4] * 4, [8, 8], [6, 6, 4]))
def test_matched_realized_work_has_identical_exposure_clocks(
    chunks: list[int],
) -> None:
    owner = RuntimeWorkClockOwner()

    for views in chunks:
        owner.record_optimization(
            optimizer_updates=1,
            supervised_views=views,
            supervised_pixels=views * 320 * 240,
            mapping_active_ns=1,
        )

    snapshot = owner.snapshot()
    assert snapshot.supervised_rgb_views == 16
    assert snapshot.supervised_rgb_pixels == 16 * 320 * 240
    assert snapshot.optimizer_updates == len(chunks)


def test_completed_result_fields_are_required_exact_non_negative_integers() -> None:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    complete = {
        "optimizer_step_count": 1,
        "supervision_view_count": 1,
        "supervised_pixel_count": 4,
        "depth_valid_sample_count": 2,
    }

    for missing in complete:
        with pytest.raises(AttributeError, match=missing):
            runtime._record_completed_optimization_result(
                SimpleNamespace(**{k: v for k, v in complete.items() if k != missing}),
                mapping_active_ns=1,
            )
    for invalid_name in complete:
        invalid = dict(complete)
        invalid[invalid_name] = True
        with pytest.raises(TypeError, match=invalid_name):
            runtime._record_completed_optimization_result(
                SimpleNamespace(**invalid), mapping_active_ns=1
            )


def test_completed_nonfinite_work_still_advances_mapping_active_time() -> None:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._work_clock_owner = RuntimeWorkClockOwner()

    snapshot = runtime._record_completed_optimization_result(
        SimpleNamespace(
            optimizer_step_count=0,
            supervision_view_count=0,
            supervised_pixel_count=0,
            depth_valid_sample_count=0,
        ),
        mapping_active_ns=23,
    )

    assert snapshot.optimizer_updates == 0
    assert snapshot.mapping_active_ns == 23


def test_mutation_epochs_advance_only_for_actual_arbiter_commits() -> None:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._work_clock_owner = RuntimeWorkClockOwner()

    unchanged = runtime._record_completed_depth_mutation(
        SimpleNamespace(applied=0, capacity_dropped=31)
    )
    depth_committed = runtime._record_completed_depth_mutation(
        SimpleNamespace(applied=1, capacity_dropped=0)
    )
    committed = runtime._record_completed_mcmc_commit(SimpleNamespace(applied=1))

    assert unchanged.mcmc_mutation_epochs == 0
    assert unchanged.depth_mutation_epochs == 0
    assert unchanged.depth_valid_samples == 0
    assert depth_committed.depth_mutation_epochs == 1
    assert depth_committed.depth_valid_samples == 0
    assert committed.mcmc_mutation_epochs == 1
    assert committed.depth_mutation_epochs == 1
    assert committed.depth_valid_samples == 0


def _zero_tick_runtime(monkeypatch: pytest.MonkeyPatch) -> CanonicalMappingRuntime:
    from eligsir.runtime import production_application

    monkeypatch.setattr(
        production_application, "MapStateTelemetry", MapStateTelemetry, raising=False
    )
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._closed = False
    runtime._control_tick_epoch = 0
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._latest_tick_work = None
    runtime._latest_report_work = None
    runtime.initialized = False
    runtime.iteration = 0
    runtime.optimizer_steps = 0
    runtime._supervision_views_completed = 0
    runtime._dispositions_since_tick = deque(maxlen=8)
    runtime._transfer_owner = None
    runtime._latest_gpu_transaction = None
    runtime._telemetry = RuntimeTelemetryAdapter()
    runtime._poll_gpu_telemetry = lambda: None
    runtime._versions = lambda: (0, 0, 0)
    runtime._telemetry_inputs = lambda _snapshot: RuntimeTelemetryInputs(
        published_at_ns=1,
        source_clock_ns=0,
        admitted_clock_ns=0,
        map_clock_ns=0,
        rates=RateTelemetry(0.0, 0.0, 0.0),
        latencies=RuntimeLatencyTelemetry(
            LatencySummary(0, 0.0, 0.0, 0.0),
            LatencySummary(0, 0.0, 0.0, 0.0),
        ),
        queues=QueueTelemetry(
            source=QueueStateTelemetry(0, 0.0),
            candidate=QueueStateTelemetry(0, 0.0),
            commit=QueueStateTelemetry(0, 0.0),
            transfer=QueueStateTelemetry(0, 0.0),
            gpu=QueueStateTelemetry(0, 0.0),
        ),
        gaussian_count=0,
        residency=ResidencyTelemetry(0, 0, 0, 0),
    )
    return runtime


def test_zero_frame_tick_publishes_one_post_event_snapshot_without_fabricated_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _zero_tick_runtime(monkeypatch)
    viewer = runtime._telemetry.store.subscribe()
    controller = runtime._telemetry.store.subscribe()

    tick = runtime.tick()

    assert tick.executed is False
    assert tick.reason == "bootstrap_pending"
    assert tick.work_clocks.control_tick_epoch == 1
    assert tick.work_clocks.optimizer_updates == 0
    assert tick.work_clocks.supervised_rgb_views == 0
    assert tick.work_clocks.supervised_rgb_pixels == 0
    assert tick.work_clocks.mapping_active_ns == 0
    assert tick.telemetry.work_clocks is tick.work_clocks
    assert viewer.read_latest().work_clocks is tick.work_clocks
    assert controller.read_latest().work_clocks is tick.work_clocks
    assert runtime._work_clock_owner.snapshot() == tick.work_clocks


def test_source_completion_publishes_a_terminal_snapshot_without_a_second_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _zero_tick_runtime(monkeypatch)
    first = runtime.tick()

    terminal = runtime.mark_source_complete()

    assert terminal.reason == "source_complete"
    assert terminal.work_clocks.source_complete is True
    assert terminal.work_clocks.control_tick_epoch == first.work_clocks.control_tick_epoch
    assert terminal.telemetry.work_clocks is terminal.work_clocks
    assert runtime._telemetry.store.latest().work_clocks is terminal.work_clocks
    assert runtime.mark_source_complete() is terminal
    assert runtime._telemetry.store.latest().work_clocks is terminal.work_clocks


def test_public_application_returns_the_idempotent_terminal_tick_result() -> None:
    terminal = object()
    calls = []
    application = ComposedMappingApplication(
        MappingApplicationOptions(
            mapper_sink=lambda *_args: None,
            optimizer_tick=lambda: None,
            require_tracking=False,
            source_complete_hook=lambda: calls.append(1) or terminal,
        )
    )

    first = application.mark_source_complete()
    second = application.mark_source_complete()

    assert first is terminal
    assert second is terminal
    assert calls == [1]


def test_depth_valid_sample_materialization_counts_the_exact_objective_mask() -> None:
    import torch

    from eligsir.optimization.gaussian.objective import (
        _accepted_depth_valid_sample_count,
        _materialize_depth_valid_sample_count,
    )

    valid = torch.tensor([[True, False, True], [False, True, False]])

    assert _materialize_depth_valid_sample_count(valid) == 3
    assert _accepted_depth_valid_sample_count(3, depth_loss_finite=True) == 3
    assert _accepted_depth_valid_sample_count(3, depth_loss_finite=False) == 0
