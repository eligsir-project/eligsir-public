"""Immutable latest-completed telemetry for the canonical live runtime.

The scalar snapshot is deliberately separate from preview images and map
outputs. GPU publishers copy only one fixed, reduced packet through the
canonical output transfer barrier and expose it after its event has completed;
readers never synchronize current mapping work for fresher statistics.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Generic, Optional, TypeVar

from eligsir.contracts.control import (
    CompactGpuTelemetry,
    EvidenceFact,
    FreshnessMode,
    MappingWorkPlan,
    ObservationDisposition,
)
from eligsir.contracts.work_clocks import RuntimeWorkClockSnapshot

from .transfers import TransferAccounting


def _integer(value: Any, name: str) -> int:
    """Validate and return a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    """Validate and return a finite non-negative floating-point value."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class RateTelemetry:
    """Completed source, admission and geometry-commit rates."""

    source_hz: float
    admitted_hz: float
    commit_hz: float

    def __post_init__(self) -> None:
        """Validate all rate values are finite and non-negative."""
        for name in ("source_hz", "admitted_hz", "commit_hz"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Bounded aggregate of one completed source-to-map latency series."""

    samples: int
    p50_ms: float
    p95_ms: float
    max_ms: float

    def __post_init__(self) -> None:
        """Validate latency percentiles are ordered and consistent with sample count."""
        object.__setattr__(self, "samples", _integer(self.samples, "samples"))
        for name in ("p50_ms", "p95_ms", "max_ms"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if self.samples == 0 and any((self.p50_ms, self.p95_ms, self.max_ms)):
            raise ValueError("empty latency summary must contain zero values")
        if not self.p50_ms <= self.p95_ms <= self.max_ms:
            raise ValueError("latency percentiles must satisfy p50 <= p95 <= max")


@dataclass(frozen=True, slots=True)
class RuntimeLatencyTelemetry:
    """Pair of source-to-first and source-to-geometry latency distributions."""
    source_to_first_supervision: LatencySummary
    source_to_geometry_commit: LatencySummary

    def __post_init__(self) -> None:
        """Validate both latency summaries are LatencySummary instances."""
        for name in ("source_to_first_supervision", "source_to_geometry_commit"):
            if not isinstance(getattr(self, name), LatencySummary):
                raise TypeError(f"{name} must be LatencySummary")


@dataclass(frozen=True, slots=True)
class QueueStateTelemetry:
    """Current queue depth and age of oldest pending item."""
    depth: int
    oldest_age_ms: float

    def __post_init__(self) -> None:
        """Validate queue state is consistent between depth and oldest age."""
        object.__setattr__(self, "depth", _integer(self.depth, "depth"))
        object.__setattr__(
            self,
            "oldest_age_ms",
            _finite_nonnegative(self.oldest_age_ms, "oldest_age_ms"),
        )
        if self.depth == 0 and self.oldest_age_ms != 0.0:
            raise ValueError("empty queue must have zero oldest_age_ms")


@dataclass(frozen=True, slots=True)
class QueueTelemetry:
    """Telemetry for all five stages of the mapping pipeline queue."""
    source: QueueStateTelemetry
    candidate: QueueStateTelemetry
    commit: QueueStateTelemetry
    transfer: QueueStateTelemetry
    gpu: QueueStateTelemetry

    def __post_init__(self) -> None:
        """Validate all queue states are QueueStateTelemetry instances."""
        for name in ("source", "candidate", "commit", "transfer", "gpu"):
            if not isinstance(getattr(self, name), QueueStateTelemetry):
                raise TypeError(f"{name} must be QueueStateTelemetry")


class TransactionType(str, Enum):
    """GPU transaction type enumeration for telemetry classification."""
    FRAME_UPLOAD = "frame_upload"
    GPU_EVIDENCE = "gpu_evidence"
    OPTIMIZATION = "optimization"
    INSERTION_PREP = "insertion_prep"
    MAP_MUTATION = "map_mutation"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class GpuTransactionTelemetry:
    """One tiny, completed GPU transaction record owned by CPU memory."""

    transaction_id: int
    snapshot_id: int
    transaction_type: TransactionType
    map_content_version: int
    topology_version: int
    parameter_version: int
    batch_size: int
    pixels: int
    rasterizer_calls: int
    optimizer_updates: int
    candidates: int
    duration_ms: float
    mutation_type: str
    inserted: int
    pruned: int
    relocated: int
    corrected: int

    def __post_init__(self) -> None:
        """Validate transaction telemetry fields and duration value."""
        for name in (
            "transaction_id",
            "snapshot_id",
            "map_content_version",
            "topology_version",
            "parameter_version",
            "batch_size",
            "pixels",
            "rasterizer_calls",
            "optimizer_updates",
            "candidates",
            "inserted",
            "pruned",
            "relocated",
            "corrected",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.transaction_type, TransactionType):
            raise TypeError("transaction_type must be TransactionType")
        if not self.mutation_type:
            raise ValueError("mutation_type must be non-empty")
        object.__setattr__(
            self, "duration_ms", _finite_nonnegative(self.duration_ms, "duration_ms")
        )

    def compact(self) -> CompactGpuTelemetry:
        """Return the completed scalar packet consumed by a later control tick."""

        return CompactGpuTelemetry(
            transaction_id=self.transaction_id,
            map_content_version=self.map_content_version,
            topology_version=self.topology_version,
            parameter_version=self.parameter_version,
            facts=(
                EvidenceFact("gpu_duration_ms", self.duration_ms),
                EvidenceFact("batch_size", self.batch_size),
                EvidenceFact("pixels", self.pixels),
                EvidenceFact("rasterizer_calls", self.rasterizer_calls),
                EvidenceFact("optimizer_updates", self.optimizer_updates),
                EvidenceFact("candidates", self.candidates),
                EvidenceFact(
                    "mutations",
                    self.inserted + self.pruned + self.relocated + self.corrected,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class MapStateTelemetry:
    """Current map geometry count and semantic version information."""
    gaussian_count: int
    map_content_version: int
    topology_version: int
    parameter_version: int

    def __post_init__(self) -> None:
        """Validate all map state fields are non-negative integers."""
        for name in (
            "gaussian_count",
            "map_content_version",
            "topology_version",
            "parameter_version",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class ResidencyTelemetry:
    """GPU cache residency counters for active, replay, history, and hot tiles."""
    active: int
    replay: int
    history: int
    hot_gpu: int

    def __post_init__(self) -> None:
        """Validate residency counts are consistent and non-negative."""
        for name in ("active", "replay", "history", "hot_gpu"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.hot_gpu > self.active + self.replay:
            raise ValueError("hot_gpu cannot exceed active plus replay residency")


@dataclass(frozen=True, slots=True)
class RuntimeTelemetrySnapshot:
    """Coherent immutable scalar state shared by control and presentation."""

    publication_id: int
    snapshot_id: Optional[int]
    published_at_ns: int
    source_clock_ns: int
    admitted_clock_ns: int
    map_clock_ns: int
    work_clocks: RuntimeWorkClockSnapshot
    freshness: FreshnessMode
    rates: RateTelemetry
    latencies: RuntimeLatencyTelemetry
    queues: QueueTelemetry
    observation: Optional[ObservationDisposition]
    work_plan: Optional[MappingWorkPlan]
    gpu_transaction: Optional[GpuTransactionTelemetry]
    transfers: Optional[TransferAccounting]
    map_state: MapStateTelemetry
    residency: ResidencyTelemetry

    def __post_init__(self) -> None:
        """Validate snapshot clocks, types, and cross-field consistency."""
        for name in (
            "publication_id",
            "published_at_ns",
            "source_clock_ns",
            "admitted_clock_ns",
            "map_clock_ns",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.snapshot_id is not None:
            object.__setattr__(self, "snapshot_id", _integer(self.snapshot_id, "snapshot_id"))
        if not self.map_clock_ns <= self.admitted_clock_ns <= self.source_clock_ns:
            raise ValueError("clocks must satisfy map <= admitted <= source")
        if not isinstance(self.work_clocks, RuntimeWorkClockSnapshot):
            raise TypeError("work_clocks must be RuntimeWorkClockSnapshot")
        if not isinstance(self.freshness, FreshnessMode):
            raise TypeError("freshness must be FreshnessMode")
        for name, expected in (
            ("rates", RateTelemetry),
            ("latencies", RuntimeLatencyTelemetry),
            ("queues", QueueTelemetry),
            ("map_state", MapStateTelemetry),
            ("residency", ResidencyTelemetry),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.transfers is not None and not isinstance(
            self.transfers, TransferAccounting
        ):
            raise TypeError("transfers must be TransferAccounting or None")
        self._validate_join()

    def _validate_join(self) -> None:
        """Validate observation and work plan consistency with snapshot."""
        if self.snapshot_id is None and (
            self.observation is not None or self.work_plan is not None
        ):
            raise ValueError("observation/work_plan require a snapshot_id")
        if self.observation is not None:
            if not isinstance(self.observation, ObservationDisposition):
                raise TypeError("observation must be ObservationDisposition or None")
            if self.observation.snapshot_id != self.snapshot_id:
                raise ValueError("observation snapshot_id must match telemetry snapshot_id")
        if self.work_plan is not None:
            if not isinstance(self.work_plan, MappingWorkPlan):
                raise TypeError("work_plan must be MappingWorkPlan or None")
            if self.work_plan.snapshot_id != self.snapshot_id:
                raise ValueError("work_plan snapshot_id must match telemetry snapshot_id")
            self._validate_versions("work plan", self.work_plan)
        if self.gpu_transaction is not None:
            if not isinstance(self.gpu_transaction, GpuTransactionTelemetry):
                raise TypeError(
                    "gpu_transaction must be GpuTransactionTelemetry or None"
                )
            if (
                self.snapshot_id is not None
                and self.gpu_transaction.snapshot_id > self.snapshot_id
            ):
                raise ValueError("completed GPU transaction cannot be from a future snapshot")
            self._validate_versions("GPU", self.gpu_transaction)

    def _validate_versions(self, owner: str, versioned: Any) -> None:
        """Verify versioned object's versions do not exceed map state versions."""
        for name in (
            "map_content_version",
            "topology_version",
            "parameter_version",
        ):
            if getattr(versioned, name) > getattr(self.map_state, name):
                raise ValueError(f"{owner} {name} cannot exceed map state")


T = TypeVar("T")


class _LatestCompletedStore(Generic[T]):
    """Capacity-one publication with non-consuming independent readers."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Initialize capacity-one publication store with optional enable flag."""
        self._enabled = bool(enabled)
        self._latest: Optional[T] = None
        self._lock = Lock()

    def _sequence(self, item: T) -> int:
        """Extract sequence number from item for ordering comparison."""
        raise NotImplementedError

    def publish(self, item: T) -> bool:
        """Publish item if enabled and sequence number is newer than current."""
        if not self._enabled:
            return False
        sequence = self._sequence(item)
        with self._lock:
            if self._latest is not None and sequence <= self._sequence(self._latest):
                return False
            self._latest = item
        return True

    def latest(self) -> Optional[T]:
        """Return the latest published item or None if store is empty."""
        with self._lock:
            return self._latest

    @property
    def retained_count(self) -> int:
        """Return 1 if an item is retained, 0 otherwise."""
        with self._lock:
            return int(self._latest is not None)


class RuntimeTelemetryStore(_LatestCompletedStore[RuntimeTelemetrySnapshot]):
    """Capacity-one store for immutable runtime telemetry snapshots."""
    def _sequence(self, item: RuntimeTelemetrySnapshot) -> int:
        """Extract publication ID from telemetry snapshot."""
        if not isinstance(item, RuntimeTelemetrySnapshot):
            raise TypeError("runtime telemetry store accepts RuntimeTelemetrySnapshot")
        return item.publication_id

    def subscribe(self) -> "RuntimeTelemetrySubscription":
        """Create a new independent subscription cursor to this store."""
        return RuntimeTelemetrySubscription(self)


class RuntimeTelemetrySubscription:
    """Per-consumer cursor over a shared latest-wins publication."""

    def __init__(self, store: RuntimeTelemetryStore) -> None:
        """Initialize subscription with reference to telemetry store."""
        self._store = store
        self._last_publication_id: Optional[int] = None

    def read_latest(
        self, *, only_if_new: bool = False
    ) -> Optional[RuntimeTelemetrySnapshot]:
        """Read the latest snapshot, optionally filtering to new publications."""
        latest = self._store.latest()
        if latest is None:
            return None
        if only_if_new and latest.publication_id == self._last_publication_id:
            return None
        self._last_publication_id = latest.publication_id
        return latest


class _GpuTelemetryStore(_LatestCompletedStore[GpuTransactionTelemetry]):
    def _sequence(self, item: GpuTransactionTelemetry) -> int:
        """Extract transaction ID from GPU telemetry packet."""
        if not isinstance(item, GpuTransactionTelemetry):
            raise TypeError("GPU telemetry store accepts GpuTransactionTelemetry")
        return item.transaction_id


@dataclass(frozen=True, slots=True)
class GpuTelemetryDescriptor:
    """CPU-known transaction facts joined to the fixed reduced GPU vector."""

    transaction_id: int
    snapshot_id: int
    transaction_type: TransactionType
    map_content_version: int
    topology_version: int
    parameter_version: int
    batch_size: int
    pixels: int
    rasterizer_calls: int
    optimizer_updates: int
    candidates: int
    mutation_type: str

    def __post_init__(self) -> None:
        """Validate GPU telemetry descriptor fields and transaction type."""
        for name in (
            "transaction_id",
            "snapshot_id",
            "map_content_version",
            "topology_version",
            "parameter_version",
            "batch_size",
            "pixels",
            "rasterizer_calls",
            "optimizer_updates",
            "candidates",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.transaction_type, TransactionType):
            raise TypeError("transaction_type must be TransactionType")
        if not self.mutation_type:
            raise ValueError("mutation_type must be non-empty")


@dataclass(slots=True)
class _PendingGpuEventTelemetry:
    start_event: Any
    end_event: Any
    descriptor: GpuTelemetryDescriptor
    inserted: int
    pruned: int
    relocated: int
    corrected: int


class AsyncGpuEventTelemetryPublisher:
    """Publish event-timed transactions only after their GPU work completes.

    CUDA events are also the PyTorch timing primitive on HIP. ``query`` is the
    only operation used while work is in flight; elapsed time is resolved only
    after the end event reports completion, so live ticks never synchronize the
    current mapping stream for telemetry.
    """

    def __init__(self) -> None:
        """Initialize GPU event telemetry publisher with pending event tracking."""
        self._store = _GpuTelemetryStore()
        self._active: Optional[_PendingGpuEventTelemetry] = None
        self._active_count = 0
        self._staged: Optional[_PendingGpuEventTelemetry] = None
        self._staged_count = 0
        self._highest_submitted = -1
        self._staged_replacements = 0
        self._invalid_completed_timings = 0

    def submit(
        self,
        start_event: Any,
        end_event: Any,
        descriptor: GpuTelemetryDescriptor,
        *,
        inserted: int = 0,
        pruned: int = 0,
        relocated: int = 0,
        corrected: int = 0,
    ) -> None:
        """Submit GPU event pair and descriptor with optional mutation counts."""
        if not callable(getattr(end_event, "query", None)):
            raise TypeError("end_event must provide query")
        if not callable(getattr(start_event, "elapsed_time", None)):
            raise TypeError("start_event must provide elapsed_time")
        if not isinstance(descriptor, GpuTelemetryDescriptor):
            raise TypeError("descriptor must be GpuTelemetryDescriptor")
        if descriptor.transaction_id <= self._highest_submitted:
            raise ValueError("GPU telemetry transaction_id must increase")
        self._highest_submitted = descriptor.transaction_id
        pending = _PendingGpuEventTelemetry(
            start_event,
            end_event,
            descriptor,
            _integer(inserted, "inserted"),
            _integer(pruned, "pruned"),
            _integer(relocated, "relocated"),
            _integer(corrected, "corrected"),
        )
        self.poll_completed()
        if self._active is None:
            self._active = pending
            self._active_count = 1
        else:
            if self._staged is not None:
                self._staged_replacements += 1
            self._staged = pending
            self._staged_count += 1

    def poll_completed(self) -> Optional[GpuTransactionTelemetry]:
        """Poll for completed GPU events and publish completed transactions."""
        active = self._active
        if active is None or not bool(active.end_event.query()):
            return None
        descriptor = active.descriptor
        duration_ms = float(active.start_event.elapsed_time(active.end_event))
        self._active = self._staged
        self._active_count = self._staged_count
        self._staged = None
        self._staged_count = 0
        if not math.isfinite(duration_ms) or duration_ms < 0.0:
            # HIP may report an invalid completed-event timing.  Observability
            # must never fabricate a duration or terminate mapping; discard the
            # driver sample while keeping strict validation for every packet
            # that is actually published.
            self._invalid_completed_timings += 1
            return None
        packet = GpuTransactionTelemetry(
            transaction_id=descriptor.transaction_id,
            snapshot_id=descriptor.snapshot_id,
            transaction_type=descriptor.transaction_type,
            map_content_version=descriptor.map_content_version,
            topology_version=descriptor.topology_version,
            parameter_version=descriptor.parameter_version,
            batch_size=descriptor.batch_size,
            pixels=descriptor.pixels,
            rasterizer_calls=descriptor.rasterizer_calls,
            optimizer_updates=descriptor.optimizer_updates,
            candidates=descriptor.candidates,
            duration_ms=duration_ms,
            mutation_type=descriptor.mutation_type,
            inserted=active.inserted,
            pruned=active.pruned,
            relocated=active.relocated,
            corrected=active.corrected,
        )
        self._store.publish(packet)
        return packet

    def latest_completed(self) -> Optional[GpuTransactionTelemetry]:
        """Return the latest completed GPU transaction or None."""
        return self._store.latest()

    @property
    def pending_count(self) -> int:
        """Return total count of active and staged pending GPU events."""
        return self._active_count + self._staged_count

    @property
    def staged_replacements(self) -> int:
        """Return count of times staged event was replaced before polling."""
        return self._staged_replacements

    @property
    def invalid_completed_timings(self) -> int:
        """Return count of invalid GPU event timings that were discarded."""
        return self._invalid_completed_timings


@dataclass(frozen=True, slots=True)
class RuntimeTelemetryInputs:
    """Scalar owners supplied to the public tick-result adapter."""

    published_at_ns: int
    source_clock_ns: int
    admitted_clock_ns: int
    map_clock_ns: int
    rates: RateTelemetry
    latencies: RuntimeLatencyTelemetry
    queues: QueueTelemetry
    gaussian_count: int
    residency: ResidencyTelemetry


class RuntimeTelemetryAdapter:
    """Publish a public MappingTickResult without reaching into runtime state."""

    def __init__(self, store: Optional[RuntimeTelemetryStore] = None) -> None:
        """Initialize adapter with optional telemetry store for publishing."""
        self.store = store if store is not None else RuntimeTelemetryStore()
        self._next_publication_id = 0

    def publish_tick(
        self,
        tick_result: Any,
        inputs: RuntimeTelemetryInputs,
        *,
        gpu_transaction: Optional[GpuTransactionTelemetry] = None,
    ) -> RuntimeTelemetrySnapshot:
        """Publish a mapping tick result as a coherent runtime telemetry snapshot."""
        snapshot_id = getattr(tick_result, "snapshot_id")
        dispositions = tuple(getattr(tick_result, "dispositions"))
        observation = next(
            (
                item
                for item in reversed(dispositions)
                if item.snapshot_id == snapshot_id
            ),
            None,
        )
        snapshot = RuntimeTelemetrySnapshot(
            publication_id=self._next_publication_id,
            snapshot_id=snapshot_id,
            published_at_ns=inputs.published_at_ns,
            source_clock_ns=inputs.source_clock_ns,
            admitted_clock_ns=inputs.admitted_clock_ns,
            map_clock_ns=inputs.map_clock_ns,
            work_clocks=getattr(tick_result, "work_clocks"),
            freshness=getattr(tick_result, "freshness"),
            rates=inputs.rates,
            latencies=inputs.latencies,
            queues=inputs.queues,
            observation=observation,
            work_plan=getattr(tick_result, "work_plan"),
            gpu_transaction=gpu_transaction,
            transfers=getattr(tick_result, "transfers"),
            map_state=MapStateTelemetry(
                gaussian_count=inputs.gaussian_count,
                map_content_version=getattr(tick_result, "map_content_version"),
                topology_version=getattr(tick_result, "topology_version"),
                parameter_version=getattr(tick_result, "parameter_version"),
            ),
            residency=inputs.residency,
        )
        self.store.publish(snapshot)
        self._next_publication_id += 1
        return snapshot


__all__ = [
    "AsyncGpuEventTelemetryPublisher",
    "GpuTelemetryDescriptor",
    "GpuTransactionTelemetry",
    "LatencySummary",
    "MapStateTelemetry",
    "QueueStateTelemetry",
    "QueueTelemetry",
    "RateTelemetry",
    "ResidencyTelemetry",
    "RuntimeLatencyTelemetry",
    "RuntimeTelemetryAdapter",
    "RuntimeTelemetryInputs",
    "RuntimeTelemetrySnapshot",
    "RuntimeTelemetryStore",
    "RuntimeTelemetrySubscription",
    "TransactionType",
]
