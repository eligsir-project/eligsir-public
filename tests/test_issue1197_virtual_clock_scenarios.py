"""Deterministic live-cadence qualification scenarios for issue #1197."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from eligsir.simulation.virtual_clock_scenarios import (
    SCENARIO_NAMES,
    ScheduledArrival,
    build_live_scenario,
    run_live_scenario,
)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_named_live_scenarios_are_repeatable_and_complete(name: str) -> None:
    spec = build_live_scenario(name)

    first = run_live_scenario(spec)
    second = run_live_scenario(spec)

    assert first.signature == second.signature
    assert first.retained_identities == second.retained_identities
    assert first.semantic_skipped_identities == second.semantic_skipped_identities
    assert first.digest["coverage_complete"] is True
    assert first.queue_peak <= spec.candidate_queue_bound
    assert first.transport_queue_peak <= spec.transport_queue_bound
    assert first.materialized_identities == first.retained_identities
    assert set(first.semantic_skipped_identities).isdisjoint(first.materialized_identities)
    assert set(first.transport_dropped_identities).isdisjoint(first.materialized_identities)
    terminal = (
        set(first.retained_identities)
        | set(first.semantic_skipped_identities)
        | set(first.transport_dropped_identities)
    )
    assert len(terminal) == first.source_frames
    assert first.map_version == len(first.retained_identities)
    assert first.parameter_version == first.optimizer_ticks


def test_healthy_30hz_maps_every_frame_with_zero_frame_optimizer_ticks() -> None:
    result = run_live_scenario(build_live_scenario("healthy_30hz"))

    assert result.source_hz == 30
    assert result.semantic_skipped_identities == ()
    assert result.transport_dropped_identities == ()
    assert len(result.retained_identities) == result.source_frames
    assert result.zero_frame_ticks > 0
    assert result.max_frontier_age_ns <= result.target_freshness_ns


@pytest.mark.parametrize("name,source_hz", [("overload_30hz", 30), ("overload_60hz", 60)])
def test_sustained_overload_coalesces_stale_work_but_keeps_mapping(
    name: str, source_hz: int
) -> None:
    result = run_live_scenario(build_live_scenario(name))

    assert result.source_hz == source_hz
    assert result.semantic_skipped_identities
    assert result.retained_identities
    assert result.survival_mode_ticks > 0
    assert result.queue_peak <= result.candidate_queue_bound
    assert result.max_frontier_age_ns <= result.target_freshness_ns


def test_recovery_reduces_freshness_age_after_service_capacity_returns() -> None:
    result = run_live_scenario(build_live_scenario("overload_recovery"))

    assert result.overload_end_age_ns > result.final_frontier_age_ns
    assert result.final_frontier_age_ns <= result.target_freshness_ns
    assert result.recovery_mapped > 0


def test_burst_stall_burst_preserves_order_and_never_resurrects_coalesced_frames() -> None:
    result = run_live_scenario(build_live_scenario("burst_stall_burst"))

    retained_ordinals = tuple(identity.source_ordinal for identity in result.retained_identities)
    skipped_ordinals = tuple(
        identity.source_ordinal for identity in result.semantic_skipped_identities
    )
    assert retained_ordinals == tuple(sorted(retained_ordinals))
    assert set(retained_ordinals).isdisjoint(skipped_ordinals)
    assert result.zero_frame_ticks > 0
    assert result.input_stall_ticks > 0


def test_zero_frame_window_advances_optimizer_without_fabricating_ingress() -> None:
    result = run_live_scenario(build_live_scenario("zero_frame_ticks"))

    assert result.zero_frame_ticks >= 8
    assert result.optimizer_ticks == result.tick_count
    assert result.parameter_version == result.tick_count
    assert result.optimizer_parameter < 1.0
    assert result.map_version == len(result.retained_identities)


def test_harness_has_no_wall_clock_wait_path() -> None:
    import eligsir.simulation.virtual_clock_scenarios as scenarios

    source = inspect.getsource(scenarios)
    assert "time.sleep" not in source
    assert "monotonic" not in source


def test_recovered_capacity_services_the_newest_n_candidates() -> None:
    base = build_live_scenario("zero_frame_ticks")
    recovery = replace(
        base,
        arrivals=tuple(ScheduledArrival(0, ordinal) for ordinal in range(4)),
        service_capacity=(2,) + (0,) * (base.tick_count - 1),
    )

    result = run_live_scenario(recovery)

    assert tuple(source.source_ordinal for source in result.retained_identities) == (2, 3)


def test_read_only_telemetry_consumer_cannot_change_semantic_digest() -> None:
    spec = build_live_scenario("overload_recovery")
    observed = []

    baseline = run_live_scenario(spec)
    connected = run_live_scenario(spec, telemetry_consumer=observed.append)

    assert len(observed) == spec.tick_count
    assert connected.signature == baseline.signature
    assert connected.retained_identities == baseline.retained_identities
    assert connected.semantic_skipped_identities == baseline.semantic_skipped_identities
    assert connected.map_version == baseline.map_version
    assert connected.parameter_version == baseline.parameter_version


def test_disabled_connected_and_stalled_viewers_are_semantically_equivalent() -> None:
    spec = build_live_scenario("overload_recovery")
    connected_events = []

    class StalledLatestOnlyViewer:
        def __init__(self) -> None:
            self.latest = None
            self.replaced = 0

        def __call__(self, event) -> None:
            if self.latest is not None:
                self.replaced += 1
            self.latest = event

    stalled = StalledLatestOnlyViewer()
    disabled = run_live_scenario(spec)
    connected = run_live_scenario(spec, telemetry_consumer=connected_events.append)
    stalled_result = run_live_scenario(spec, telemetry_consumer=stalled)

    for observed in (connected, stalled_result):
        assert observed.signature == disabled.signature
        assert observed.retained_identities == disabled.retained_identities
        assert observed.semantic_skipped_identities == disabled.semantic_skipped_identities
        assert observed.transport_dropped_identities == disabled.transport_dropped_identities
        assert observed.ticks == disabled.ticks
        assert observed.map_version == disabled.map_version
        assert observed.parameter_version == disabled.parameter_version
        assert observed.digest["event_counts"].get("mutation_commit", 0) == disabled.digest[
            "event_counts"
        ].get("mutation_commit", 0)

    assert len(connected_events) == spec.tick_count
    assert stalled.replaced == spec.tick_count - 1
    assert stalled.latest == disabled.telemetry[-1]


def test_source_frontier_age_does_not_grow_during_sensor_or_refinement_stalls() -> None:
    burst = run_live_scenario(build_live_scenario("burst_stall_burst"))
    zero_frames = run_live_scenario(build_live_scenario("zero_frame_ticks"))

    assert burst.final_frontier_age_ns == 0
    assert zero_frames.final_frontier_age_ns == 0


def test_transport_drops_remain_distinct_from_core_semantic_coalescing() -> None:
    base = build_live_scenario("zero_frame_ticks")
    constrained = replace(
        base,
        arrivals=tuple(ScheduledArrival(0, ordinal) for ordinal in range(4)),
        service_capacity=(1,) + (0,) * (base.tick_count - 1),
        transport_queue_bound=2,
    )

    result = run_live_scenario(constrained)

    assert tuple(source.source_ordinal for source in result.retained_identities) == (1,)
    assert tuple(
        source.source_ordinal for source in result.semantic_skipped_identities
    ) == (0,)
    assert tuple(
        source.source_ordinal for source in result.transport_dropped_identities
    ) == (2, 3)
