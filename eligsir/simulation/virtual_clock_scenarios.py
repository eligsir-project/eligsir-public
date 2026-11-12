"""Deterministic live-source cadence scenarios for issue #1197.

This is a software qualification harness, not a throughput benchmark.  A
virtual 60 Hz driver clock feeds tiny timestamped RGB-D packets through the
real :class:`MappingSession`, :func:`drain_tick`, and replay-digest contracts.
Service capacity and arrivals are integer schedules, so overload, recovery,
bursts, stalls, and empty optimizer ticks run without sleeping or consulting a
wall clock.

The built-in mapper and optimizer are deterministic CPU software models.  They
qualify cadence, identity, coalescing, and replay semantics only.  Production
GPU queue, transfer, mutation-arbiter, and Viser compatibility claims require a
separate ``MappingApplication`` integration test; this module intentionally
does not synthesize counters for those owners.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eligsir.contracts.materialize import (
    ColorPolicy,
    DepthPolicy,
    MaterializationPolicy,
    numpy_backend,
)
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.deterministic_digest import DeterministicReplayDigest
from eligsir.runtime.driver import IterationTick, check_cadence_invariant, drain_tick
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import ExternalTrackingProvider, observation_from_pose_matrix


SCENARIO_NAMES = (
    "healthy_30hz",
    "overload_30hz",
    "overload_60hz",
    "overload_recovery",
    "burst_stall_burst",
    "zero_frame_ticks",
)


@dataclass(frozen=True, slots=True)
class ScheduledArrival:
    tick: int
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class LiveScenarioSpec:
    name: str
    source_hz: int
    tick_hz: int
    tick_count: int
    arrivals: tuple[ScheduledArrival, ...]
    service_capacity: tuple[int, ...]
    candidate_queue_bound: int = 4
    transport_queue_bound: int = 16
    target_freshness_ns: int = 100_000_000
    overload_end_tick: int | None = None

    def __post_init__(self) -> None:
        if self.name not in SCENARIO_NAMES:
            raise ValueError(f"unknown live scenario {self.name!r}")
        if self.source_hz <= 0 or self.tick_hz <= 0 or self.tick_count <= 0:
            raise ValueError("source_hz, tick_hz, and tick_count must be positive")
        if len(self.service_capacity) != self.tick_count:
            raise ValueError("service_capacity must contain one value per tick")
        if any(value < 0 for value in self.service_capacity):
            raise ValueError("service capacity must be non-negative")
        if any(arrival.tick < 0 or arrival.tick >= self.tick_count for arrival in self.arrivals):
            raise ValueError("arrival tick lies outside the scenario")
        if tuple(sorted(self.arrivals, key=lambda item: item.tick)) != self.arrivals:
            raise ValueError("arrivals must be ordered by virtual tick")
        for name in (
            "candidate_queue_bound",
            "transport_queue_bound",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TickTelemetry:
    tick: int
    virtual_time_ns: int
    arrivals: int
    submitted: int
    mapped: int
    candidate_depth: int
    frontier_age_ns: int
    map_version: int
    parameter_version: int


@dataclass(slots=True)
class _SoftwareOptimizer:
    """Tiny deterministic optimizer used only to exercise iteration cadence."""

    parameter: float = 1.0
    version: int = 0

    def step(self, tick: IterationTick) -> None:
        # The update deliberately depends on tick work while remaining stable
        # across platforms.  This is executable cadence evidence, not a model-
        # quality or production-GPU claim.
        self.parameter -= 0.001 * (1 + tick.mapped)
        self.version += 1


@dataclass(frozen=True, slots=True)
class LiveScenarioResult:
    schema_version: int
    name: str
    source_hz: int
    source_frames: int
    tick_count: int
    optimizer_ticks: int
    zero_frame_ticks: int
    input_stall_ticks: int
    survival_mode_ticks: int
    recovery_mapped: int
    candidate_queue_bound: int
    transport_queue_bound: int
    queue_peak: int
    transport_queue_peak: int
    target_freshness_ns: int
    max_frontier_age_ns: int
    overload_end_age_ns: int
    final_frontier_age_ns: int
    final_frontier_ordinal: int
    map_version: int
    parameter_version: int
    optimizer_parameter: float
    retained_identities: tuple[SourceIdentity, ...]
    materialized_identities: tuple[SourceIdentity, ...]
    semantic_skipped_identities: tuple[SourceIdentity, ...]
    transport_dropped_identities: tuple[SourceIdentity, ...]
    ticks: tuple[IterationTick, ...]
    telemetry: tuple[TickTelemetry, ...]
    digest: dict
    signature: str


def _periodic_arrivals(
    *, source_hz: int, tick_hz: int, tick_count: int
) -> tuple[ScheduledArrival, ...]:
    return tuple(
        ScheduledArrival(
            tick=tick,
            timestamp_ns=(tick * 1_000_000_000) // tick_hz,
        )
        for tick in range(tick_count)
        if (tick * source_hz) // tick_hz != ((tick - 1) * source_hz) // tick_hz
    )


def build_live_scenario(name: str) -> LiveScenarioSpec:
    """Return one frozen, finite virtual-clock scenario."""

    tick_hz = 60
    if name == "healthy_30hz":
        tick_count, source_hz = 32, 30
        arrivals = _periodic_arrivals(source_hz=source_hz, tick_hz=tick_hz, tick_count=tick_count)
        service = (1,) * tick_count
        overload_end = None
    elif name in {"overload_30hz", "overload_60hz"}:
        tick_count = 60
        source_hz = 30 if name == "overload_30hz" else 60
        arrivals = _periodic_arrivals(source_hz=source_hz, tick_hz=tick_hz, tick_count=tick_count)
        service = tuple(1 if tick % 6 == 0 else 0 for tick in range(tick_count))
        overload_end = tick_count - 1
    elif name == "overload_recovery":
        tick_count, source_hz = 72, 30
        arrivals = _periodic_arrivals(source_hz=source_hz, tick_hz=tick_hz, tick_count=tick_count)
        service = tuple(
            (1 if tick % 6 == 0 else 0) if tick < 36 else 2
            for tick in range(tick_count)
        )
        overload_end = 35
    elif name == "burst_stall_burst":
        tick_count, source_hz = 40, 60
        arrivals = tuple(
            [
                ScheduledArrival(8, (ordinal * 1_000_000_000) // source_hz)
                for ordinal in range(8)
            ]
            + [
                ScheduledArrival(
                    39,
                    ((32 + ordinal) * 1_000_000_000) // tick_hz,
                )
                for ordinal in range(8)
            ]
        )
        service = (1,) * tick_count
        overload_end = None
    elif name == "zero_frame_ticks":
        tick_count, source_hz = 16, 30
        arrivals = (ScheduledArrival(0, 0), ScheduledArrival(1, 1_000_000_000 // tick_hz))
        service = (1,) * tick_count
        overload_end = None
    else:
        raise ValueError(f"unknown live scenario {name!r}; expected one of {SCENARIO_NAMES}")

    return LiveScenarioSpec(
        name=name,
        source_hz=source_hz,
        tick_hz=tick_hz,
        tick_count=tick_count,
        arrivals=arrivals,
        service_capacity=service,
        overload_end_tick=overload_end,
    )


def _packet(ordinal: int, timestamp_ns: int) -> SensorPacket:
    source = SourceIdentity(
        source_id="virtual-rgbd",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=timestamp_ns,
        pair_id=f"virtual-{ordinal:06d}",
    )
    rgb = np.full((4, 4, 3), ordinal % 251, dtype=np.uint8)
    depth = np.full((4, 4), 1.0 + ordinal * 0.001, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = ordinal * 0.01
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=CameraIntrinsics(fx=4.0, fy=4.0, cx=1.5, cy=1.5, width=4, height=4),
        tracking=observation_from_pose_matrix(source, pose, provider="virtual-clock"),
    )


def run_live_scenario(
    spec: LiveScenarioSpec,
    *,
    telemetry_consumer: Callable[[TickTelemetry], None] | None = None,
) -> LiveScenarioResult:
    """Execute ``spec`` synchronously using virtual integer time only."""

    digest = DeterministicReplayDigest()
    transport: deque[SensorPacket] = deque()
    candidates: deque[SensorPacket] = deque()
    semantic_skips: list[SourceIdentity] = []
    transport_drops: list[SourceIdentity] = []
    retained: list[SourceIdentity] = []
    materialized: list[SourceIdentity] = []
    ticks: list[IterationTick] = []
    telemetry: list[TickTelemetry] = []
    arrival_cursor = 0
    queue_peak = transport_peak = 0
    map_version = parameter_version = 0
    optimizer = _SoftwareOptimizer()
    latest_frontier: SourceIdentity | None = None
    latest_source: SourceIdentity | None = None
    max_frontier_age_ns = 0
    overload_end_age_ns = 0
    recovery_mapped = 0
    survival_mode_ticks = 0
    input_stall_ticks = 0
    first_arrival_tick = spec.arrivals[0].tick if spec.arrivals else None
    last_arrival_tick = spec.arrivals[-1].tick if spec.arrivals else None

    def mapper(frame, observation) -> None:
        nonlocal map_version, latest_frontier
        assert frame.source == observation.source
        retained.append(frame.source)
        materialized.append(frame.source)
        latest_frontier = frame.source
        map_version += 1

    session = MappingSession(
        ExternalTrackingProvider("virtual-clock"),
        mapper=mapper,
        materialization_policy=MaterializationPolicy(
            device="cpu",
            color=ColorPolicy.NORMALIZED_FLOAT32,
            depth=DepthPolicy.METERS_FLOAT32,
        ),
        materialization_backend=numpy_backend(),
        deterministic_digest=digest,
    )

    for tick in range(spec.tick_count):
        now_ns = (tick * 1_000_000_000) // spec.tick_hz
        arrivals_this_tick = 0
        while arrival_cursor < len(spec.arrivals) and spec.arrivals[arrival_cursor].tick == tick:
            scheduled = spec.arrivals[arrival_cursor]
            packet = _packet(arrival_cursor, scheduled.timestamp_ns)
            latest_source = packet.source
            arrival_cursor += 1
            arrivals_this_tick += 1
            if len(transport) >= spec.transport_queue_bound:
                transport_drops.append(packet.source)
                digest.record("transport_drop", {"source": packet.source.fingerprint()})
            else:
                transport.append(packet)
                transport_peak = max(transport_peak, len(transport))

        if (
            first_arrival_tick is not None
            and last_arrival_tick is not None
            and first_arrival_tick < tick < last_arrival_tick
            and arrivals_this_tick == 0
        ):
            input_stall_ticks += 1

        while transport:
            packet = transport.popleft()
            if len(candidates) >= spec.candidate_queue_bound:
                skipped = candidates.popleft()
                semantic_skips.append(skipped.source)
                digest.record(
                    "semantic_disposition",
                    {"source": skipped.source.fingerprint(), "decision": "coalesce"},
                )
            candidates.append(packet)
            queue_peak = max(queue_peak, len(candidates))

        capacity = spec.service_capacity[tick]
        overloaded = capacity == 0 or len(candidates) > max(1, capacity)
        if capacity > 0 and len(candidates) > capacity and overloaded:
            while len(candidates) > capacity:
                skipped = candidates.popleft()
                semantic_skips.append(skipped.source)
                digest.record(
                    "semantic_disposition",
                    {"source": skipped.source.fingerprint(), "decision": "coalesce"},
                )
            survival_mode_ticks += 1

        before_mapped = len(retained)
        tick_result = drain_tick(
            session,
            iteration=tick,
            has_ready_frame=lambda: bool(candidates),
            next_packet=candidates.popleft,
            drain_cap=max(1, capacity),
            should_release=(lambda _iteration, _dt: capacity > 0),
            iter_dt=1.0 / spec.tick_hz,
        )
        ticks.append(tick_result)
        mapped_now = len(retained) - before_mapped
        if spec.overload_end_tick is not None and tick > spec.overload_end_tick:
            recovery_mapped += mapped_now

        optimizer.step(tick_result)
        parameter_version = optimizer.version
        digest.record(
            "optimizer_batch",
            {
                "tick": tick,
                "sources": (
                    [source.fingerprint() for source in retained[-mapped_now:]]
                    if mapped_now
                    else []
                ),
                "work_count": 1,
                "parameter_version": parameter_version,
            },
        )
        if mapped_now:
            digest.record(
                "mutation_commit",
                {"tick": tick, "mapped": mapped_now, "map_version": map_version},
            )

        frontier_age_ns = 0
        if latest_source is not None and latest_frontier is not None:
            frontier_age_ns = max(
                0, latest_source.timestamp_ns - latest_frontier.timestamp_ns
            )
        max_frontier_age_ns = max(max_frontier_age_ns, frontier_age_ns)
        if tick == spec.overload_end_tick:
            overload_end_age_ns = frontier_age_ns
        sample = TickTelemetry(
            tick=tick,
            virtual_time_ns=now_ns,
            arrivals=arrivals_this_tick,
            submitted=tick_result.submitted,
            mapped=tick_result.mapped,
            candidate_depth=len(candidates),
            frontier_age_ns=frontier_age_ns,
            map_version=map_version,
            parameter_version=parameter_version,
        )
        telemetry.append(sample)
        if telemetry_consumer is not None:
            telemetry_consumer(sample)

    while candidates:
        skipped = candidates.popleft()
        semantic_skips.append(skipped.source)
        digest.record(
            "semantic_disposition",
            {"source": skipped.source.fingerprint(), "decision": "scenario_closed"},
        )

    check_cadence_invariant(ticks, drain_cap=max(spec.service_capacity + (1,)))
    session.flush()
    session.close()
    snapshot = digest.snapshot()
    final_age = telemetry[-1].frontier_age_ns
    final_ordinal = -1 if latest_frontier is None else latest_frontier.source_ordinal
    return LiveScenarioResult(
        schema_version=1,
        name=spec.name,
        source_hz=spec.source_hz,
        source_frames=len(spec.arrivals),
        tick_count=spec.tick_count,
        optimizer_ticks=spec.tick_count,
        zero_frame_ticks=sum(tick.submitted == 0 for tick in ticks),
        input_stall_ticks=input_stall_ticks,
        survival_mode_ticks=survival_mode_ticks,
        recovery_mapped=recovery_mapped,
        candidate_queue_bound=spec.candidate_queue_bound,
        transport_queue_bound=spec.transport_queue_bound,
        queue_peak=queue_peak,
        transport_queue_peak=transport_peak,
        target_freshness_ns=spec.target_freshness_ns,
        max_frontier_age_ns=max_frontier_age_ns,
        overload_end_age_ns=overload_end_age_ns,
        final_frontier_age_ns=final_age,
        final_frontier_ordinal=final_ordinal,
        map_version=map_version,
        parameter_version=parameter_version,
        optimizer_parameter=optimizer.parameter,
        retained_identities=tuple(retained),
        materialized_identities=tuple(materialized),
        semantic_skipped_identities=tuple(semantic_skips),
        transport_dropped_identities=tuple(transport_drops),
        ticks=tuple(ticks),
        telemetry=tuple(telemetry),
        digest=snapshot,
        signature=snapshot["digest"],
    )


__all__ = [
    "SCENARIO_NAMES",
    "LiveScenarioResult",
    "LiveScenarioSpec",
    "ScheduledArrival",
    "TickTelemetry",
    "build_live_scenario",
    "run_live_scenario",
]
