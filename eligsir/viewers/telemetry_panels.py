"""Pure CPU projection of canonical runtime telemetry into viewer panels.

The assembler only copies facts already published by the latest-completed
runtime snapshot. Unsupported measurements stay ``None``; it never queries a
GPU, model, queue, or clock itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


def _identity(source) -> str:
    return (
        f"{source.source_id}:{source.stream_epoch}:"
        f"{source.source_ordinal}@{source.timestamp_ns}"
    )


@dataclass(frozen=True, slots=True)
class DecisionPanel:
    publication_id: int
    snapshot_id: Optional[int]
    control_version: Optional[int]
    plan_id: Optional[str]
    freshness: str
    disposition: Optional[str]
    reason: Optional[str]
    observe: Optional[bool]
    mapping_candidate: Optional[bool]
    admit_supervision: Optional[bool]
    allow_geometry_insertion: Optional[bool]
    residency_intent: Optional[str]
    supervision_sources: tuple[str, ...]
    replay_sources: tuple[str, ...]
    resolution_tier: Optional[str]
    optimizer_updates: Optional[int]
    insertion_seeds: Optional[int]
    allow_growth: Optional[bool]
    allow_relocation: Optional[bool]
    mutation_budget: Optional[int]


@dataclass(frozen=True, slots=True)
class FreshnessPanel:
    source_hz: float
    admitted_hz: float
    commit_hz: float
    source_to_admitted_age_ms: float
    source_to_map_age_ms: float
    first_supervision_samples: int
    first_supervision_p50_ms: float
    first_supervision_p95_ms: float
    first_supervision_max_ms: float
    geometry_samples: int
    geometry_p50_ms: float
    geometry_p95_ms: float
    geometry_max_ms: float
    queue_depths: tuple[tuple[str, int], ...]
    queue_oldest_age_ms: tuple[tuple[str, float], ...]
    newest_frontier_age_ms: Optional[float] = None


@dataclass(frozen=True, slots=True)
class GpuTransferPanel:
    transaction_id: Optional[int]
    transaction_snapshot_id: Optional[int]
    transaction_type: Optional[str]
    duration_ms: Optional[float]
    batch_size: Optional[int]
    pixels: Optional[int]
    rasterizer_calls: Optional[int]
    optimizer_updates: Optional[int]
    candidates: Optional[int]
    h2d_ops: Optional[int]
    h2d_bytes: Optional[int]
    d2h_ops: Optional[int]
    d2h_bytes: Optional[int]
    pool_waits: Optional[int]
    pool_drops: Optional[int]
    pinned_bytes: Optional[int]
    peak_pinned_bytes: Optional[int]
    inflight_limit: Optional[int]
    shape_buckets: Optional[int]
    shape_bucket_limit: Optional[int]
    batch_mode: Optional[str] = None
    fallback_reason: Optional[str] = None
    unexpected_d2h: Optional[int] = None
    memory_bytes: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ResidencyPanel:
    gaussian_count: int
    map_content_version: int
    topology_version: int
    parameter_version: int
    mutation_type: Optional[str]
    inserted: Optional[int]
    pruned: Optional[int]
    relocated: Optional[int]
    corrected: Optional[int]
    active: int
    replay: int
    history: int
    hot_gpu: int


@dataclass(frozen=True, slots=True)
class ViewerTelemetryPanels:
    decision: DecisionPanel
    freshness: FreshnessPanel
    gpu: GpuTransferPanel
    residency: ResidencyPanel

    def as_wire(self) -> dict:
        return asdict(self)


def assemble_viewer_telemetry(snapshot) -> ViewerTelemetryPanels:
    """Copy one immutable latest-completed runtime snapshot for presentation."""

    observation = snapshot.observation
    plan = snapshot.work_plan
    gpu = snapshot.gpu_transaction
    transfers = snapshot.transfers
    queue_names = ("source", "candidate", "commit", "transfer", "gpu")
    first = snapshot.latencies.source_to_first_supervision
    geometry = snapshot.latencies.source_to_geometry_commit

    decision = DecisionPanel(
        publication_id=snapshot.publication_id,
        snapshot_id=snapshot.snapshot_id,
        control_version=None if plan is None else plan.control_version,
        plan_id=None if plan is None else plan.plan_id,
        freshness=snapshot.freshness.value,
        disposition=None if observation is None else observation.kind.value,
        reason=None if observation is None else observation.reason,
        observe=None if observation is None else observation.observe,
        mapping_candidate=None if observation is None else observation.mapping_candidate,
        admit_supervision=None if observation is None else observation.admit_supervision,
        allow_geometry_insertion=None if observation is None else observation.allow_geometry_insertion,
        residency_intent=None if observation is None else observation.residency.value,
        supervision_sources=() if plan is None else tuple(_identity(x) for x in plan.supervision_sources),
        replay_sources=() if plan is None else tuple(_identity(x) for x in plan.replay_sources),
        resolution_tier=None if plan is None else plan.resolution_tier,
        optimizer_updates=None if plan is None else plan.optimizer_updates,
        insertion_seeds=None if plan is None else plan.insertion.max_seeds,
        allow_growth=None if plan is None else plan.allow_growth,
        allow_relocation=None if plan is None else plan.allow_relocation,
        mutation_budget=None if plan is None else plan.mutation_budget.max_total,
    )
    freshness = FreshnessPanel(
        source_hz=snapshot.rates.source_hz,
        admitted_hz=snapshot.rates.admitted_hz,
        commit_hz=snapshot.rates.commit_hz,
        source_to_admitted_age_ms=(snapshot.source_clock_ns - snapshot.admitted_clock_ns) / 1e6,
        source_to_map_age_ms=(snapshot.source_clock_ns - snapshot.map_clock_ns) / 1e6,
        first_supervision_samples=first.samples,
        first_supervision_p50_ms=first.p50_ms,
        first_supervision_p95_ms=first.p95_ms,
        first_supervision_max_ms=first.max_ms,
        geometry_samples=geometry.samples,
        geometry_p50_ms=geometry.p50_ms,
        geometry_p95_ms=geometry.p95_ms,
        geometry_max_ms=geometry.max_ms,
        queue_depths=tuple((name, getattr(snapshot.queues, name).depth) for name in queue_names),
        queue_oldest_age_ms=tuple((name, getattr(snapshot.queues, name).oldest_age_ms) for name in queue_names),
    )
    gpu_panel = GpuTransferPanel(
        transaction_id=None if gpu is None else gpu.transaction_id,
        transaction_snapshot_id=None if gpu is None else gpu.snapshot_id,
        transaction_type=None if gpu is None else gpu.transaction_type.value,
        duration_ms=None if gpu is None else gpu.duration_ms,
        batch_size=None if gpu is None else gpu.batch_size,
        pixels=None if gpu is None else gpu.pixels,
        rasterizer_calls=None if gpu is None else gpu.rasterizer_calls,
        optimizer_updates=None if gpu is None else gpu.optimizer_updates,
        candidates=None if gpu is None else getattr(gpu, "candidates", 0),
        **{
            name: None if transfers is None else getattr(transfers, name)
            for name in (
                "h2d_ops", "h2d_bytes", "d2h_ops", "d2h_bytes", "pool_waits",
                "pool_drops", "pinned_bytes", "peak_pinned_bytes", "inflight_limit",
                "shape_buckets", "shape_bucket_limit",
            )
        },
    )
    residency = ResidencyPanel(
        gaussian_count=snapshot.map_state.gaussian_count,
        map_content_version=snapshot.map_state.map_content_version,
        topology_version=snapshot.map_state.topology_version,
        parameter_version=snapshot.map_state.parameter_version,
        mutation_type=None if gpu is None else gpu.mutation_type,
        inserted=None if gpu is None else gpu.inserted,
        pruned=None if gpu is None else gpu.pruned,
        relocated=None if gpu is None else gpu.relocated,
        corrected=None if gpu is None else gpu.corrected,
        active=snapshot.residency.active,
        replay=snapshot.residency.replay,
        history=snapshot.residency.history,
        hot_gpu=snapshot.residency.hot_gpu,
    )
    return ViewerTelemetryPanels(decision, freshness, gpu_panel, residency)


def format_panel(panel: dict, title: str) -> str:
    """Format already-assembled scalar facts; ``None`` is rendered honestly."""

    lines = [f"**{title}**", ""]
    for key, value in panel.items():
        shown = "n/a" if value is None else value
        lines.append(f"- {key.replace('_', ' ')}: {shown}")
    return "\n".join(lines)


def runtime_telemetry_preview(snapshot, *, iteration: int, wall_time_s: float):
    """Build the canonical event root composition can publish to PreviewBus."""

    from eligsir.viewers.events import RuntimeTelemetryPreview

    return RuntimeTelemetryPreview(
        iteration=iteration,
        wall_time_s=wall_time_s,
        panels=assemble_viewer_telemetry(snapshot),
    )
