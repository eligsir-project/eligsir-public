"""Sole concrete continual-mapping application composition.

The transport boundary ends at ``SensorPacket``. This module owns the existing
streaming scene, one MappingCoordinator/backend, one GaussianOptimizationEngine,
and the explicit post-optimization MCMC mutation boundary. It creates no queue
and makes no transport/drop decision.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from eligsir.runtime.post_refinement import (
    PostRefinementPhase,
    PostRefinementPhaseOwner,
)
from eligsir.contracts import SourceIdentity
from eligsir.contracts.control import (
    CompactGpuTelemetry,
    ControlProposal,
    EvidenceFact,
    FreshnessMode,
    SemanticAction,
)
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.contracts.work_clocks import (
    RuntimeWorkClockOwner,
    RuntimeWorkClockSnapshot,
)
from eligsir.mapping.scheduler.backpressure import pose_delta_signals
from eligsir.mapping.scheduler.frame_quality import (
    blur_laplacian_score,
    mean_luma,
)
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.mapping.batch_contract import (
    validate_supervision_camera_count,
    validate_supervision_batch_size,
)
from eligsir.cli.config import load_cli_config
from eligsir.cli.parser import parse_training_args
from eligsir.cli.run_config import validate_config
from .config_boundary import (
    depth_insertion_policy_from_args,
    depth_loss_policy_from_args,
    depth_supervision_enabled,
    mapping_runtime_config_from_args,
    motion_admission_evidence,
    scheduled_b1_work,
)
from .config import MappingRuntimeConfig
from .runtime_binding import runtime_binding_from_runtime
from .ready_only_tick import (
    ReadyOnlyTickSelection,
    ReadyOnlyTickSelector,
    ReadySourceBinding,
    ReadySourceVersions,
)
from .ready_source import ReadySourceLane, ReadySourceOwner, ReplayPriority
from .prepared_execution_context import (
    PreparedControlAxes,
    PreparedExecutionContext,
)


def _initialize_train_evwb(record: Any, camera: Any, policy: Any, *, is_train: bool) -> None:
    """Allocate frame-owned train state once; READY rebindings only borrow it."""
    if (
        not is_train
        or not bool(getattr(policy, "streaming_train_evwb_enabled", False))
        or record.source_identity == str(getattr(policy, "streaming_train_evwb_reference_source", ""))
        or record.photometric_evwb is not None
    ):
        return
    import torch as torch_module

    image = getattr(camera, "original_image", None)
    if not isinstance(image, torch_module.Tensor):
        raise RuntimeError("train EV/WB admission requires resident display RGB")
    record.photometric_evwb = torch_module.zeros(
        3, device=image.device, dtype=image.dtype, requires_grad=True
    )


def source_release_rate_ratio_from_clocks(
    clocks: Any,
    *,
    requested_source_fps: float | None,
    cap: float = 2.0,
    now: float | None = None,
) -> float | None:
    """Return realized source release rate over the requested paced source fps.

    ``None`` when the source is not paced (``requested_source_fps`` is None/<=0)
    or no source rate has been measured yet. The denominator is the requested
    SOURCE clock (``--streaming_input_fps_cap`` under dataset_fps /
    wallclock_strict release), never the supervised-views target
    (``replay_target_views_per_s``), which is ~3x larger and would make the
    relaxation reset fire on every tick (#1388).
    """
    if requested_source_fps is None or float(requested_source_fps) <= 0.0:
        return None
    summary = (
        clocks.summary(now=now)
        if callable(getattr(clocks, "summary", None))
        else clocks
    )
    source_rate = 0.0 if summary is None else float(
        getattr(summary, "source_frame_rate_hz", 0.0) or 0.0
    )
    if source_rate <= 0.0:
        return 0.0 if bool(getattr(clocks, "has_established_source_rate", False)) else None
    return min(source_rate / float(requested_source_fps), float(cap))


@dataclass(frozen=True, slots=True)
class _ReadyOnlyB4Transaction:
    """JIT transaction input for the existing B>1 execution/tail."""

    plan: Any
    cameras: dict[Any, Any]
    bindings: tuple[Any, ...]
    source_insertion: bool = False
from .prepared_ingress import PreparedIngressControlToken
from .supervision_work_clocks import SupervisionWorkClocks
from eligsir.utils.ingest_phase_profiler import _timed as _phase_timed

# Test seams are deliberately declared without importing accelerator modules.
render = None


class _TorchStub:
    @staticmethod
    def no_grad():
        return nullcontext()


torch = _TorchStub()


def combine_control_evidence(
    ingress_facts: tuple[EvidenceFact, ...],
    runtime_facts: tuple[EvidenceFact, ...],
) -> tuple[EvidenceFact, ...]:
    """Combine immutable evidence domains, rejecting ambiguous ownership."""

    ingress_facts = tuple(ingress_facts)
    runtime_facts = tuple(runtime_facts)
    if not all(isinstance(fact, EvidenceFact) for fact in ingress_facts + runtime_facts):
        raise TypeError("control evidence domains must contain EvidenceFact values")
    collisions = sorted(
        {fact.name for fact in ingress_facts}
        & {fact.name for fact in runtime_facts}
    )
    if collisions:
        raise ValueError(
            "cross-domain control evidence collision: " + ", ".join(collisions)
        )
    return ingress_facts + runtime_facts


def _scheduled_transaction_work(
    plan: MappingWorkPlan, *, configured_batch_size: int
) -> tuple[Any, ...]:
    """Select one tick's work without changing legacy configured-B1 cadence."""

    configured_batch_size = validate_supervision_batch_size(configured_batch_size)
    if configured_batch_size == 1:
        return scheduled_b1_work(plan)
    constellation = plan.supervision_constellation
    if constellation is None:
        raise RuntimeError("canonical work plan has no supervision constellation")
    scheduled = tuple(plan.supervision_work) + tuple(plan.replay_work)
    return scheduled[: constellation.realized_batch_size]


_LIFECYCLE_NONFINITE_UTILITY_REPORT_LIMIT = 12
_lifecycle_nonfinite_utility_counts: dict[int, int] = {}


def _lifecycle_nonfinite_utility_report(
    n_nonfinite: int, n_total: int, optimizer_updates: int
) -> None:
    """Rate-limited notice that some per-Gaussian lifecycle utility entries were
    non-finite and excluded from this scan (issue #1259). Mirrors
    ``_nan_term_report``'s rate-limiting in
    ``eligsir/mapping/integration/loss_diagnostics.py`` -- keeps the root
    cause visible without spamming the log on every lifecycle scan of a long
    run whose upstream utility keeps producing a few bad rows."""

    key = 0
    n = _lifecycle_nonfinite_utility_counts.get(key, 0) + 1
    _lifecycle_nonfinite_utility_counts[key] = n
    if n <= _LIFECYCLE_NONFINITE_UTILITY_REPORT_LIMIT:
        print(
            f"[lifecycle-guard] optimizer_updates={optimizer_updates} WARNING: "
            f"{n_nonfinite}/{n_total} lifecycle utility entries were non-finite "
            "and excluded from this scan's EMA update/promotion (occurrence "
            f"{n}). See compute_gaussian_utility's own [energy-mcmc] WARNING "
            "for which upstream input produced them.",
            flush=True,
        )


def _planned_optimizer_update_permitted(
    plan: MappingWorkPlan, *, update_ordinal: int
) -> bool:
    """Return exact per-plan Adam permission for one dispatched update."""

    if type(update_ordinal) is not int or update_ordinal < 0:
        raise ValueError("update_ordinal must be a non-negative exact integer")
    return update_ordinal < plan.optimizer_updates


def _adaptive_quality_evidence_facts(
    evidence: tuple[bool, bool] | None,
) -> tuple[EvidenceFact, ...]:
    if evidence is None:
        return ()
    return (
        EvidenceFact("partition_fade", evidence[0]),
        EvidenceFact("quality_risk", evidence[1]),
    )


def _packet_rgb_hwc(packet: Any) -> np.ndarray | None:
    """Return exact RGB in HWC layout for supported declared encodings."""

    rgb_ref = getattr(packet, "rgb", None)
    spec = getattr(rgb_ref, "spec", None)
    encoding = str(getattr(spec, "encoding", "") or "").lower()
    rgb = np.asarray(getattr(rgb_ref, "data", None))
    if encoding == "rgb32f_chw":
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            return None
        return np.transpose(rgb, (1, 2, 0))
    if encoding in {"rgb8", "rgb32f"}:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            return None
        return rgb
    return None


def _packet_depth_meters(packet: Any) -> np.ndarray | None:
    depth_ref = getattr(packet, "depth", None)
    if depth_ref is None:
        return None
    depth = np.asarray(depth_ref.data, dtype=np.float32)
    units = str(getattr(depth_ref.spec, "units", "") or "").lower()
    if units not in {"m", "metres", "meters"}:
        scale = getattr(getattr(packet, "intrinsics", None), "depth_scale", None)
        if scale is None or not np.isfinite(float(scale)) or float(scale) <= 0.0:
            return None
        depth = depth / float(scale)
    return depth


def _packet_parallax_deg(
    packet: Any,
    current_c2w: np.ndarray | None,
    last_admitted_c2w: np.ndarray | None,
    *,
    depth_min: float,
    depth_max: float,
    stride: int,
) -> float | None:
    """Compute median ray parallax from exact packet depth/calibration/poses."""

    if current_c2w is None or last_admitted_c2w is None:
        return None
    depth = _packet_depth_meters(packet)
    intrinsics = getattr(packet, "intrinsics", None)
    if depth is None or intrinsics is None or depth.ndim != 2:
        return None
    step = max(1, int(stride))
    sampled = depth[::step, ::step]
    ys, xs = np.meshgrid(
        np.arange(sampled.shape[0]) * step,
        np.arange(sampled.shape[1]) * step,
        indexing="ij",
    )
    valid = (
        np.isfinite(sampled)
        & (sampled > float(depth_min))
        & (sampled < float(depth_max))
    )
    if int(valid.sum()) <= 10:
        return None
    height, width = depth.shape
    native_width = float(getattr(intrinsics, "depth_width", width) or width)
    native_height = float(getattr(intrinsics, "depth_height", height) or height)
    sx, sy = width / native_width, height / native_height
    fx = float(intrinsics.depth_fx) * sx
    fy = float(intrinsics.depth_fy) * sy
    cx = (float(intrinsics.depth_cx) + 0.5) * sx - 0.5
    cy = (float(intrinsics.depth_cy) + 0.5) * sy - 0.5
    z = sampled[valid]
    x = (xs[valid].astype(np.float32) - cx) / fx * z
    y = (ys[valid].astype(np.float32) - cy) / fy * z
    points_camera = np.stack((x, y, z, np.ones_like(z)), axis=1)
    current = np.asarray(current_c2w, dtype=np.float32)
    previous = np.asarray(last_admitted_c2w, dtype=np.float32)
    points_world = points_camera @ current[:3, :].T
    current_rays = points_world - current[:3, 3][None, :]
    previous_rays = points_world - previous[:3, 3][None, :]
    current_norm = np.linalg.norm(current_rays, axis=1)
    previous_norm = np.linalg.norm(previous_rays, axis=1)
    usable = (current_norm > 1e-6) & (previous_norm > 1e-6)
    if not usable.any():
        return None
    cosine = np.sum(current_rays[usable] * previous_rays[usable], axis=1) / (
        current_norm[usable] * previous_norm[usable]
    )
    return float(np.median(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def spatial_replay_focus_interval_for_batch(interval_views: int, batch_size: int) -> int:
    """Keep spatial-replay quality sampling at its B4 step cadence."""
    return max(1, int(interval_views)) * max(1, int(batch_size) - 1) // 3 or 1


def admission_prefetch_capacity_for_batch(batch_size: int) -> int:
    """Retain two exact READY transactions' protected lookahead sources."""
    return max(8, 2 * int(batch_size))


def _resolve_prematerialize_tier(
    supported_tiers: tuple["ResolutionTier", ...], *, enabled: bool
) -> "ResolutionTier | None":
    """Pure construction-time gate for ``--streaming_admission_prematerialize``.

    Issue #279/#1342 follow-up (2026-09-04 review): admission-time tier
    pre-materialize is only sound when the mapping tier is concrete --
    exactly one supported tier -- because a wrong-tier guess can only miss.
    ``adaptive`` resolves to several supported tiers and is refused.

    Extracted to a module-level pure function (no torch/CUDA access) so the
    refusal condition is testable on CPU CI: previously it lived inline in
    ``CanonicalMappingRuntime.__init__``, whose construction otherwise
    requires an accelerator, and the only compensating CPU test asserted
    ``len(tuple(ResolutionTier)) > 1`` -- a fact about the enum itself, not
    about the filtered ``supported_tiers`` tuple this gate actually reads.

    Returns ``None`` when ``enabled`` is False (the flag-off no-op). Returns
    the single supported tier when exactly one is supported -- the static,
    construction-time answer. Under ``adaptive`` (several supported tiers)
    it also returns ``None``: the tier is then resolved *per admission* by
    :func:`_prematerialize_tier_for` from the work budget's most recent
    selection (see ``CanonicalMappingRuntime._prematerialize_tier_now``).
    A wrong guess there can only miss, which is exactly the flag-off cost,
    so refusing ``adaptive`` outright (the 2026-09-04 behaviour) made the
    optimisation unusable on the canonical profile for no safety gain.
    Zero supported tiers is a real configuration error and still raises.
    """

    if not enabled:
        return None
    if not supported_tiers:
        raise ValueError(
            "streaming_admission_prematerialize requires at least one "
            "supported mapping tier, got none: pass a concrete "
            "--streaming_mapping_resolution_tier (native/half/quarter/eighth) "
            "or a valid --streaming_mapping_resolution_min_tier under adaptive"
        )
    if len(supported_tiers) == 1:
        return supported_tiers[0]
    return None


def _prematerialize_tier_for(
    supported_tiers: tuple["ResolutionTier", ...],
    last_selected: "ResolutionTier | None",
) -> "ResolutionTier":
    """Per-admission tier to pre-materialize under an ``adaptive`` policy.

    Pure (no torch/CUDA) so it is testable on CPU CI. Preference order:

    1. the work budget's most recently *selected* transaction tier, when it
       is one of the supported tiers -- the best available predictor of the
       tier the first draw will actually request;
    2. otherwise the cheapest legal tier (largest downsample factor). Under
       a budget-floor-locked scheduler that is the realized tier for the
       overwhelming majority of transactions, and it is also the cheapest
       thing to build speculatively.
    """

    if not supported_tiers:
        raise ValueError("supported_tiers must not be empty")
    if last_selected is not None and last_selected in supported_tiers:
        return last_selected
    return max(supported_tiers, key=lambda tier: tier.downsample_factor)


def insertion_proposal_from_plan(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("accelerator insertion proposal used before runtime construction")


def normalize_pending_candidates(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("accelerator candidates used before runtime construction")


def insert_gaussians_from_frame_device_native(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("accelerator insertion producer used before runtime construction")


class StaleGpuWorkError(RuntimeError):
    """Torch-free placeholder replaced by the accelerator transaction error."""


@dataclass(frozen=True)
class ProposalKey:
    stream_epoch: int
    source_ordinal: int
    timestamp_ns: int
    source_id: str


def _load_accelerator_dependencies() -> None:
    """Bind accelerator modules only when constructing a causal runtime."""
    global torch, render, GaussianModel, GsplatGaussianModel, CanonicalStreamingScene
    global ControlProposal, ControlLimits, ControlSnapshot, EvidenceFact, FreshnessMode
    global MappingWorkPlan, ObservationDisposition, ResolutionTier, ObservationOwner
    global keyframe_admitter_proposal_provider, KeyframeAdmitterConfig
    global CanonicalMappingServices, CacheHandle, ColdPixelHandle, FrameObservationDB, ProposalKey
    global BatchExecutionMode, FixedTopologyBatchBuilder, GaussianMutationEpoch, GpuWorkEnvelope
    global StaleGpuWorkError, execute_fixed_topology_transaction, insertion_proposal_from_plan
    global validate_accelerator_live_insertion_capabilities, BackpressureOwner, BackpressureConfig
    global pose_delta_signals, WorkBudgetOwner, WeightedReplayCandidateProvider
    global ResolutionPolicyProvider, materialize_resolution_tier
    global BatchConstellationProposalProvider
    global AdaptiveGrowthProposalProvider
    global OptimizerProposalProvider, InsertionProposalProvider, MutationProposalProvider
    global collect_control_proposals
    global TransferAccounting, get_device_transfer_owner, StreamingOptimizationRuntime
    global run_streaming_optimization_step, run_streaming_optimization_transaction
    global supports_streaming_optimization_transaction, RandomStateToken
    global AsyncGpuEventTelemetryPublisher
    global GpuTelemetryDescriptor, GpuTransactionTelemetry, LatencySummary, MapStateTelemetry
    global QueueStateTelemetry, QueueTelemetry, RateTelemetry, ResidencyTelemetry, RuntimeLatencyTelemetry
    global RuntimeTelemetryAdapter, RuntimeTelemetryInputs, RuntimeTelemetrySnapshot, RuntimeTelemetryStore
    global TransactionType, StreamingObservation, reset_sensor_depth_cache_state, set_sensor_depth_cache_bound
    global normalize_pending_candidates, insert_gaussians_from_frame_device_native
    global CameraResidencyLedger, configure_torch_compile, configure_training_ssim
    global MCMCScheduleConfig
    global make_mcmc_strategy
    global PostRefinementPhase, PostRefinementPhaseOwner
    if globals().get("_accelerator_bound", False):
        return
    import torch
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.gsplat_model import GsplatGaussianModel
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene
    from eligsir.contracts.control import ControlProposal, ControlLimits, ControlSnapshot, EvidenceFact, FreshnessMode, MappingWorkPlan, ObservationDisposition, ResolutionTier
    from eligsir.mapping.admission.observation_owner import ObservationOwner
    from eligsir.mapping.admission.keyframe_provider import keyframe_admitter_proposal_provider
    from eligsir.mapping.scheduler.keyframes import KeyframeAdmitterConfig
    from eligsir.mapping.canonical_services import CanonicalMappingServices
    from eligsir.mapping.observation_db import CacheHandle, ColdPixelHandle, FrameObservationDB
    from eligsir.map.arbiter import ProposalKey
    from eligsir.map.gpu_transactions import BatchExecutionMode, FixedTopologyBatchBuilder, GaussianMutationEpoch, GpuWorkEnvelope, StaleGpuWorkError, execute_fixed_topology_transaction, insertion_proposal_from_plan, validate_accelerator_live_insertion_capabilities
    from eligsir.mapping.scheduler.backpressure import BackpressureConfig, BackpressureOwner, pose_delta_signals
    from eligsir.mapping.scheduler.work_budget import (
        AdaptiveGrowthProposalProvider,
        BatchConstellationProposalProvider,
        InsertionProposalProvider,
        MutationProposalProvider,
        OptimizerProposalProvider,
        WeightedReplayCandidateProvider,
        WorkBudgetOwner,
        collect_control_proposals,
    )
    from eligsir.mapping.scheduler.resolution_policy import ResolutionPolicyProvider
    from eligsir.mapping.resolution_materialization import materialize_resolution_tier
    from eligsir.runtime.post_refinement import PostRefinementPhase, PostRefinementPhaseOwner
    from eligsir.runtime.transfers import TransferAccounting, get_device_transfer_owner
    from eligsir.mapping.optimization_runtime import (
        StreamingOptimizationRuntime,
        run_streaming_optimization_step,
        run_streaming_optimization_transaction,
        supports_streaming_optimization_transaction,
    )
    from eligsir.optimization.gaussian.contracts import RandomStateToken
    from eligsir.runtime.telemetry import AsyncGpuEventTelemetryPublisher, GpuTelemetryDescriptor, GpuTransactionTelemetry, LatencySummary, MapStateTelemetry, QueueStateTelemetry, QueueTelemetry, RateTelemetry, ResidencyTelemetry, RuntimeLatencyTelemetry, RuntimeTelemetryAdapter, RuntimeTelemetryInputs, RuntimeTelemetrySnapshot, RuntimeTelemetryStore, TransactionType
    from eligsir.runtime.streaming_observation import StreamingObservation
    from eligsir.mapping.integration.depth_helpers import reset_sensor_depth_cache_state, set_sensor_depth_cache_bound
    from eligsir.mapping.integration.device_candidates import normalize_pending_candidates
    from eligsir.mapping.integration.device_integration_planner import insert_gaussians_from_frame_device_native
    from eligsir.utils.camera_residency import CameraResidencyLedger
    from eligsir.utils.compiled_kernels import configure_torch_compile
    from eligsir.utils.loss_utils import configure_training_ssim
    from eligsir.utils.mcmc_schedule import MCMCScheduleConfig
    from eligsir.utils.strategies import make_mcmc_strategy
    globals()["_accelerator_bound"] = True

DEFAULT_MAPPING_PROFILE = "live/canonical_mapping"


@dataclass(frozen=True, slots=True)
class MutationAttribution:
    """Exact origin and effect of one canonical structural mutation lane."""

    kind: str
    source: Any = None
    candidates: int = 0
    inserted: int = 0
    pruned: int = 0
    relocated: int = 0
    corrected: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"live_depth", "replay_depth", "mcmc"}:
            raise ValueError(f"unsupported mutation attribution kind: {self.kind}")
        if self.kind == "mcmc" and self.source is not None:
            raise ValueError("MCMC mutation attribution has no frame source")
        if self.kind != "mcmc" and self.source is None:
            raise ValueError("depth mutation attribution requires a frame source")
        counts = (
            self.candidates,
            self.inserted,
            self.pruned,
            self.relocated,
            self.corrected,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in counts):
            raise ValueError("mutation attribution counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class MappingTickResult:
    executed: bool
    iteration: int
    work_clocks: RuntimeWorkClockSnapshot
    reason: str = ""
    optimizer_steps: int = 0
    supervision_views: int = 0
    supervised_pixels: int = 0
    unique_supervision_views: int | None = 0
    successful_supervision_sources: tuple[Any, ...] = ()
    microbatch_sizes: tuple[int, ...] = ()
    cumulative_optimizer_updates: int = 0
    cumulative_supervision_views: int = 0
    snapshot_id: int | None = None
    control_snapshot: ControlSnapshot | None = None
    dispositions: tuple[ObservationDisposition, ...] = ()
    retained_catalog_population_post_dispositions: int | None = None
    work_plan: MappingWorkPlan | None = None
    freshness: Any = None
    transfers: TransferAccounting | None = None
    map_content_version: int = 0
    topology_version: int = 0
    parameter_version: int = 0
    telemetry: RuntimeTelemetrySnapshot | None = None
    insertion_candidates: int = 0
    insertion_filter: dict[str, Any] | None = None
    inserted: int = 0
    pruned: int = 0
    relocated: int = 0
    corrected: int = 0
    structural_mutation: bool = False
    mutation_events: tuple[MutationAttribution, ...] = ()
    primary_training_psnr_db: float | None = None
    primary_training_source_id: str | None = None
    primary_training_spatial: Any = None
    primary_training_gaussian_count: int | None = None
    # Issue #279 camera-materialize follow-up: how many of this tick's
    # transaction_work identities hit FrameObservationDB's HOT tier cache
    # (`resident_supervision_for_source`) vs. fell to the miss path
    # (`_restore_camera_payload_from_cold` + `materialize_resolution_tier`).
    # Zero/zero on a non-executed tick (bootstrap/empty-plan/no-camera).
    camera_materialize_cache_hits: int = 0
    camera_materialize_cache_misses: int = 0
    # Trace-only classification: legacy adaptive capacity is independent from
    # the #1365 live population policy's disposition.
    adaptive_growth_enabled: bool = False
    population_policy_disposition: str = "not_requested"
    population_stale_current_mismatches: tuple[tuple[str, int, int], ...] = ()
    population_policy_selection_telemetry: tuple[dict[str, Any], ...] = ()
    native_geometry_evidence_telemetry: dict[str, Any] | None = None
    prepared_insertion_skip: str | None = None
    replay_challenger_trace: tuple[dict[str, Any], ...] = ()
    replay_challenger_trace_truncated: bool = False


@dataclass(frozen=True, slots=True)
class MappingBackendCompatibility:
    """Opaque constructor-only inputs kept outside canonical execution."""

    dataset: Any
    opt: Any
    pipe: Any


class CanonicalMappingRuntime:
    """One owner for bootstrap, frame integration and optimizer/mutation ticks."""

    @staticmethod
    def _make_work_budget_owner(
        config: MappingRuntimeConfig,
        *,
        batch_size: int,
        depth_insertion_policy: Any,
    ) -> Any:
        """Bind the resolved execution cap at the sole production owner seam."""

        camera_count_max = config.hard_limits["optimization_camera_count_max"]
        if isinstance(camera_count_max, bool) or not isinstance(camera_count_max, int):
            raise TypeError("optimization_camera_count_max must be an exact integer")
        # The owner is the sole final resolution authority, so the configured
        # quality floor has to reach it -- not only the evidence provider.
        # Imported locally: this seam is exercised directly by tests that never
        # bind the deferred accelerator globals.
        from eligsir.contracts.control import ResolutionTier
        from eligsir.mapping.scheduler.tier_cost_model import (
            LIVE_TIERS as _LIVE_TIERS,
            resolve_device_identity,
        )

        minimum_resolution_tier = ResolutionTier(
            str(config.static_options.get("mapping_resolution_min_tier", "eighth"))
        )
        resolution_tier_policy = str(config.static_options["mapping_resolution_tier"])
        supported_resolution_tiers = (
            tuple(
                tier
                for tier in _LIVE_TIERS
                if tier.downsample_factor <= minimum_resolution_tier.downsample_factor
            )
            if resolution_tier_policy == "adaptive"
            else (ResolutionTier(resolution_tier_policy),)
        )
        return WorkBudgetOwner(
            supported_resolution_tiers=supported_resolution_tiers,
            device_id=resolve_device_identity(),
            budget_relax_max=float(
                config.static_options.get("resolution_budget_relax_max", 4.0)
            ),
            target_supervised_views_per_s=float(
                config.static_options.get("replay_target_views_per_s", 100.0)
            ),
            requested_batch_size=batch_size,
            maximum_camera_count=(None if camera_count_max == 0 else camera_count_max),
            constellation_policy=str(
                config.static_options["optimization_constellation_policy"]
            ),
            depth_insertion=depth_insertion_policy,
            replay_insertion_voxel_size_m=float(
                config.static_options["replay_depth_voxel_size_m"]
            ),
            sample_stride=max(1, int(config.static_options["depth_stride"])),
            replay_candidate_provider=WeightedReplayCandidateProvider(
                random_seed=int(config.static_options.get("seed", 1)),
                global_weight_floor=float(
                    config.static_options.get("replay_global_weight_floor", 0.05)
                ),
            ),
            refinement_resolution_policy=str(
                config.static_options.get("refinement_resolution_policy", "native")
            ),
            refinement_progressive_switch_frac=float(
                config.static_options.get(
                    "refinement_progressive_switch_frac", 0.5
                )
            ),
        )

    def __init__(
        self,
        config: MappingRuntimeConfig,
        compatibility: MappingBackendCompatibility,
    ) -> None:
        _load_accelerator_dependencies()
        if not isinstance(config, MappingRuntimeConfig):
            raise TypeError("config must be MappingRuntimeConfig")
        if not isinstance(compatibility, MappingBackendCompatibility):
            raise TypeError("compatibility must be MappingBackendCompatibility")
        model_layout = config.model_layout
        # Semantic keyframe selection is owned by the ObservationOwner's
        # keyframe proposal provider (``keyframe_policy`` -> the
        # ``streaming_keyframe_*`` gates), never by a trainer-level admission
        # router. ``hybrid_keyframe`` is therefore accepted as the documented
        # spelling of "keyframe filtering" and means exactly what
        # ``preselected`` means here: transport drops finish before core
        # semantic disposition, and the gates decide. Only the modes that
        # would introduce a second decision owner are refused.
        frame_admission = str(config.static_options["frame_admission"])
        if frame_admission not in ("preselected", "hybrid_keyframe"):
            raise ValueError(
                "canonical MappingApplication requires "
                "streaming_frame_admission=preselected (or its alias "
                f"hybrid_keyframe), got {frame_admission!r}; transport drops "
                "must finish before core semantic disposition"
            )
        self._batch_size = validate_supervision_batch_size(
            config.hard_limits["optimization_batch_size"]
        )
        validate_accelerator_live_insertion_capabilities(config.device_capabilities)

        model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
        self.config = config
        # The legacy optimization namespace is consumed exactly once by the
        # model's optimizer constructor after bootstrap. It never becomes an
        # execution policy; canonical kernels receive the frozen typed policy.
        self._backend_optimizer_setup = compatibility.opt
        self._runtime_policy = config.runtime
        self._optimization_policy = config.optimization
        self._refinement_depth_loss_override()
        self._lifecycle_interval_updates = max(
            1,
            int(self._optimization_policy.streaming_lifecycle_interval_updates),
        )
        self._next_lifecycle_optimizer_update = self._lifecycle_interval_updates
        self._depth_loss_policy = config.depth_loss
        self.opt = config.optimization
        self.pipe = config.pipeline
        self.gaussians = model_cls(compatibility.dataset.sh_degree)
        self._services = CanonicalMappingServices(self.gaussians)
        self.backend = self._services.backend
        self.bootstrap_frames = max(1, int(config.static_options["initial_frames"]))
        self._bootstrap: list[StreamingObservation] = []
        self.scene: CanonicalStreamingScene | None = None
        self.initialized = False
        self.iteration = 0
        self.optimizer_steps = 0
        self._latest_admitted_optimizer_update = 0
        self._sh_degree_thresholds_consumed = 0
        self._supervision_views_completed = 0
        self._work_clock_owner = RuntimeWorkClockOwner()
        self._supervision_work_clocks = SupervisionWorkClocks()
        self._mcmc_cadence_owner: MCMCCadenceOwner | None = None
        self._latest_tick_result = None
        self._source_complete_tick_result = None
        self.frames_seen = 0
        self._prev_insert_frame = None
        self._prev_insert_depth_m = None
        self._requested_source_fps_for_pacing: float | None = None
        self._source_input_exhausted_provider: Callable[[], bool] | None = None
        self._engine = None
        self._optimization_runtime = None
        self._mcmc_strategy = None
        self._mcmc_cfg = None
        self._use_energy_mcmc = False
        # M5 shadow values are owned here rather than inferred from unrelated
        # legacy control states.  Surface evidence/policy currently have no
        # live mutator, so their authoritative initial versions are zero.
        self._population_shadow_surface_evidence_version = 0
        self._population_shadow_policy_version = 0
        self._population_shadow_active_plan = None
        self._population_selection_pending_summaries = deque()
        self._native_geometry_evidence_outcome_trace = []
        self._native_geometry_evidence_lane = None
        if bool(config.static_options.get("population_shadow_enabled", False)):
            from eligsir.mapping.scheduler.native_geometry_evidence import (
                NativeGeometryEvidenceLane,
            )

            self._native_geometry_evidence_lane = NativeGeometryEvidenceLane()
        # M5/M6 live actuation has two deliberately explicit composition
        # adapters.  The context provider captures the registered UID inventory
        # only at the exact mutation fence; the target provider never guesses a
        # pairing from policy rows.  Missing or invalid ownership fails closed
        # (and does *not* fall back to legacy structural owners).
        self._population_live_target_binding_provider = None
        self._population_live_surface_target_provider = None
        self._population_live_context_provider = None
        # One shared bound owns both ControlLimits.max_mutations and the
        # provider's SurfaceTargetBatch capacity.
        self._population_live_max_actions = 1
        if self._population_live_requested():
            from eligsir.mapping.evidence.production_surface_targets import (
                ProductionSurfaceTargetProvider,
                bind_policy_surface_targets,
                reduce_native_depth_surface_evidence,
            )

            provider = ProductionSurfaceTargetProvider(
                reducer=reduce_native_depth_surface_evidence,
                binder=bind_policy_surface_targets,
                max_targets=self._population_live_max_actions,
            )
            self._population_live_target_binding_provider = provider
            self._population_live_surface_target_provider = provider
            self._population_live_context_provider = self._population_live_plan_context
        self._closed = False
        self._control_snapshot_id = 0
        self._control_tick_epoch = 0
        self._telemetry = RuntimeTelemetryAdapter(RuntimeTelemetryStore())
        self._next_gpu_transaction_id = 0
        self._gpu_event_telemetry = AsyncGpuEventTelemetryPublisher()
        self._latest_gpu_transaction: GpuTransactionTelemetry | None = None
        self._latest_optimization_gpu_transaction: GpuTransactionTelemetry | None = None
        self._latest_completed_optimization_transaction: CompactGpuTelemetry | None = None
        self._pending_optimization_transaction_timings: deque[tuple[Any, ...]] = deque()
        self._invalid_optimization_gpu_timings = 0
        self._latest_optimization_gpu_ms: float | None = None
        self._latest_mapping_step_wall_ms: float | None = None
        self._source_clock_ns = 0
        self._admitted_clock_ns = 0
        self._map_clock_ns = 0
        self._admitted_observations_total = 0
        self._last_frontier_age_ms = 0.0
        self._last_frontier_sample_ns: int | None = None
        self._source_arrival_ns: dict[str, int] = {}
        self._first_supervision_latency_ms: deque[float] = deque(maxlen=256)
        self._geometry_commit_latency_ms: deque[float] = deque(maxlen=256)
        self._latest_control_snapshot: ControlSnapshot | None = None
        self._latest_control_outputs: tuple[ControlSnapshot, tuple[ControlProposal, ...]] | None = None
        # submit() is synchronous through the terminal disposition sink. Keep
        # exactly one packet only across that boundary so DEFER can publish a
        # cold handle without creating a second application queue.
        self._latest_control_packet: Any | None = None
        self._last_admitted_pose: np.ndarray | None = None
        self._dispositions_since_tick: deque[ObservationDisposition] = deque(maxlen=4096)
        self._replay_modalities: OrderedDict[Any, Any] = OrderedDict()
        self._replay_weights: dict[Any, float] = {}
        self._replay_focus_sources: tuple[Any, ...] = ()
        self._replay_focus_frame_weights: tuple[tuple[int, float], ...] = ()
        from eligsir.mapping.scheduler.historical_candidate_index import (
            HistoricalCandidateIndex,
            HistoricalCandidateConfig,
        )

        candidate_config = HistoricalCandidateConfig.for_batch_size(self._batch_size)
        self._historical_candidate_index = HistoricalCandidateIndex(
            config=candidate_config,
            seed=int(config.static_options.get("seed", 1))
        )
        from eligsir.mapping.scheduler.hot_cold_challenger import (
            HotColdChallengerArena,
        )

        # Freshness leases and modality EMAs share one historical identity owner.
        # Optional same-render feedback does not grant physical existence confidence.
        self._hot_cold_challenger_arena = HotColdChallengerArena(
            lease_ticks=int(
                getattr(
                    self._optimization_policy,
                    "streaming_replay_challenger_lease_ticks",
                    100,
                )
            )
        )
        self._historical_sources_by_frame_id: dict[int, Any] = {}
        self._historical_frame_ids_by_source: dict[Any, int] = {}
        # Issue #1308 wiring-verification counters. A null result from a run
        # with --streaming_historical_usefulness_tracking must be
        # distinguishable from "the producer never fired" -- see
        # historical_usefulness_telemetry() below. Cheap unconditional
        # increments; no behavior depends on these values.
        self._historical_usefulness_observations_total = 0
        self._historical_usefulness_trusted_updates_total = 0
        self._replay_challenger_realized_demotions_total = 0
        self._replay_challenger_trace_events: list[dict[str, Any]] = []
        self._replay_challenger_trace_truncated = False
        self._pending_supervision_modalities: OrderedDict[Any, Any] = OrderedDict()
        self._pending_supervision_capacity = max(1, self._batch_size - 1)
        self._latest_tick_work: tuple[ControlSnapshot, MappingWorkPlan] | None = None
        self._latest_report_work: tuple[ControlSnapshot, MappingWorkPlan] | None = None
        self._last_admitted_control: tuple[Any, tuple[Any, ...], Any, Any] | None = None
        # Prepared ingress cannot use the legacy single latest-work slots: a
        # later ready lease must retain its own terminal control decision.
        self._prepared_execution_contexts: OrderedDict[
            tuple[int, Any, int | None], PreparedExecutionContext
        ] = OrderedDict()
        self._prepared_execution_contexts_by_source: OrderedDict[
            Any, PreparedExecutionContext
        ] = OrderedDict()
        # A COMMIT whose bounded FRESH residency was full remains admitted
        # first-supervision work.  Its rebuild obligation is attached to the
        # corresponding durable FrameObservationDB record, rather than kept in
        # this bounded runtime context cache.
        self._prepared_execution_context_capacity = 4096
        # The prepared adapter freezes exactly one WorkBudget allocation.  It
        # owns neither plan construction nor a second queue: this pair merely
        # preserves source identity until the canonical tick consumes it.
        self._prepared_ready_candidate_work: tuple[
            ControlSnapshot, MappingWorkPlan
        ] | None = None
        self._prepared_ready_frozen_work: tuple[
            ControlSnapshot, MappingWorkPlan
        ] | None = None
        self._active_materialization_work: tuple[ControlSnapshot, MappingWorkPlan] | None = None
        self._last_materialized_work: tuple[ControlSnapshot, MappingWorkPlan] | None = None
        self._source_complete = False
        # Optional P0 ready-source seam.  It is deliberately not auto-created:
        # a producer must explicitly provide the bounded owner before this
        # runtime can select prepared work.  The legacy plan/camera path is
        # therefore byte-for-byte untouched when the owner is absent.
        self._ready_source_owner: ReadySourceOwner | None = None
        self._ready_only_tick_selector: ReadyOnlyTickSelector | None = None
        self._post_refinement_phase_owner = PostRefinementPhaseOwner()
        self._spatial_replay_focus_enabled = bool(
            config.static_options.get("spatial_replay_focus", False)
        )
        self._partition_replay_focus_enabled = bool(
            config.static_options.get("partition_replay_focus", False)
        )
        self._partition_focus_actuation_reaches_active_sources = bool(
            config.static_options.get(
                "partition_focus_actuation_reaches_active_sources", False
            )
        )
        self._spatial_replay_focus_interval = spatial_replay_focus_interval_for_batch(
            int(config.static_options.get("spatial_replay_focus_interval", 25)),
            int(self._batch_size),
        )
        self._next_spatial_quality_replay_view = self._spatial_replay_focus_interval
        self._primary_psnr_interval = max(
            1, int(getattr(self._optimization_policy, "scalar_log_interval", 100))
        )
        self._next_primary_psnr_optimizer_update = self._primary_psnr_interval
        self._spatial_quality_tracker = None
        if self._spatial_replay_focus_enabled:
            from eligsir.utils.local_quality import LocalQualityTracker

            self._spatial_quality_tracker = LocalQualityTracker(
                cell_size=float(
                    config.static_options["spatial_replay_focus_cell_size_m"]
                )
            )
        self._replay_depth_max_per_event = max(
            0, int(config.static_options["replay_depth_max_per_event"])
        )
        self._replay_depth_interval = max(
            1, int(config.static_options["replay_depth_interval"])
        )
        self._replay_supervision_views_completed = 0
        self._next_replay_depth_supervision_view = self._replay_depth_interval
        self._pending_integration: tuple[Any, StreamingObservation, Any] | None = None
        self._pending_integration_since_ns: int | None = None
        self._candidate_oldest_age_ns = 0
        self._commit_oldest_age_ns = 0
        self._source_identities_by_ordinal: dict[int, Any] = {}
        self._depth_insertion_policy = config.depth_insertion
        # M6 staged accounting.  Values are updated only at the explicit
        # insertion-preparation boundary, so they cannot be mistaken for a
        # render or transfer performed by normal RGB-D supervision.
        self._depth_insertion_render_reuse_telemetry = {
            "insertion_render_reused": 0,
            "source_lease_hit": 0,
            "source_h2d_ops": 0,
            "source_d2h_ops": 0,
            "extra_rasterizer_calls": 0,
            "fallback": "not_attempted",
        }
        # #1259 far-region-starvation follow-up: the post-source-complete
        # replay-depth-insertion pass below may use a relaxed occupancy/range
        # policy distinct from the live path's (see
        # ``relaxed_depth_insertion_policy``'s docstring). Sentinels in
        # ``config.static_options`` make this identical to
        # ``self._depth_insertion_policy`` unless explicitly overridden.
        from eligsir.runtime.config import relaxed_depth_insertion_policy

        self._replay_depth_insertion_policy = relaxed_depth_insertion_policy(
            self._depth_insertion_policy,
            alpha_threshold=float(
                config.static_options.get(
                    "replay_depth_relaxed_alpha_threshold", -1.0
                )
            ),
            max_depth_m=float(
                config.static_options.get(
                    "replay_depth_relaxed_max_depth_m", 0.0
                )
            ),
        )
        self._work_budget_owner = self._make_work_budget_owner(
            config,
            batch_size=self._batch_size,
            depth_insertion_policy=self._depth_insertion_policy,
        )
        self._motion_admission_thresholds = (
            float(config.static_options["keyframe_min_translation_m"]),
            float(config.static_options["keyframe_min_rotation_deg"]),
        )
        resolution_tier_policy = str(
            config.static_options["mapping_resolution_tier"]
        )
        minimum_resolution_tier = ResolutionTier(
            str(config.static_options.get("mapping_resolution_min_tier", "eighth"))
        )
        supported_resolution_tiers = (
            tuple(
                tier
                for tier in ResolutionTier
                if tier.downsample_factor <= minimum_resolution_tier.downsample_factor
            )
            if resolution_tier_policy == "adaptive"
            else (ResolutionTier(resolution_tier_policy),)
        )
        # Issue #279/#1342 follow-up: admission-time tier pre-materialize is
        # only sound when the mapping tier is concrete (exactly one supported
        # tier), because a wrong-tier guess can only miss. Gate extracted to
        # the module-level `_resolve_prematerialize_tier` so the refusal
        # condition is testable on CPU CI (2026-09-04 review).
        self._admission_prematerialize_enabled = bool(
            getattr(config.runtime, "admission_prematerialize_enabled", False)
        )
        self._admission_prematerialize_tier = _resolve_prematerialize_tier(
            supported_resolution_tiers, enabled=self._admission_prematerialize_enabled
        )
        # Under ``adaptive`` the static tier above is None and the tier is
        # resolved per admission from the work budget's latest selection
        # (``_prematerialize_tier_now``); keep the supported set for that.
        self._admission_prematerialize_supported_tiers = tuple(
            supported_resolution_tiers
        )
        self._last_transaction_resolution_tier: "ResolutionTier | None" = None
        self._admission_prematerialize_total = 0
        self._admission_prematerialize_failed = 0
        self._admission_prematerialize_warned = False
        self._resolution_policy_provider = ResolutionPolicyProvider(
            supported_tiers=supported_resolution_tiers,
            target_views_per_s=float(
                config.static_options.get("replay_target_views_per_s", 100.0)
            ),
            batch_size=self._batch_size,
        )
        self._mutation_policy_provider = MutationProposalProvider(
            live_population_authorization=self._population_live_requested()
        )
        self._adaptive_growth_provider = AdaptiveGrowthProposalProvider(
            enabled=bool(config.static_options.get("adaptive_growth", False)),
            marginal_utility_brake=float(
                config.static_options.get("adaptive_growth_u_brake", -1.0e-5)
            ),
            marginal_utility_release=float(
                config.static_options.get("adaptive_growth_u_release", 1.0e-5)
            ),
            hysteresis=int(
                config.static_options.get("adaptive_growth_hysteresis", 1)
            ),
            minimum_population_delta=int(
                config.static_options.get(
                    "adaptive_growth_min_population_delta", 200
                )
            ),
            utility_ema=float(
                config.static_options.get("adaptive_growth_ema", 0.5)
            ),
            regression_tolerance_db=float(
                config.static_options.get(
                    "adaptive_growth_regression_tolerance_db", 0.25
                )
            ),
        )
        self._adaptive_quality_risk_tolerance_db = float(
            config.static_options.get(
                "adaptive_growth_regression_tolerance_db", 0.25
            )
        )
        self._adaptive_supervision_quality_evidence: tuple[bool, bool] | None = None
        self._control_proposal_providers = (
            self._resolution_policy_provider,
            BatchConstellationProposalProvider(
                batch_size=self._batch_size,
                pressure_dwell=3,
            ),
            OptimizerProposalProvider(),
            InsertionProposalProvider(
                max(0, int(config.runtime.max_new_gaussians_per_frame)),
                enabled=bool(config.static_options["insert_from_depth"]),
            ),
            self._mutation_policy_provider,
        )
        self._observation_db = FrameObservationDB(
            device="cuda",
            max_active_supervision_frames=config.runtime.max_active_supervision_frames,
            max_cold_payload_frames=config.runtime.cold_payload_max_frames,
            max_cold_payload_bytes=config.runtime.cold_payload_max_bytes,
            # 0 == unbounded: the count-based active-supervision policy alone
            # governs HOT residency, which is the pre-existing behaviour and
            # today's default. See RuntimeConfig.hot_tier_cache_max_bytes.
            max_hot_tier_bytes=(
                int(config.runtime.hot_tier_cache_max_bytes) or None
            ),
            max_tiers_per_frame=int(config.runtime.max_tiers_per_frame),
            tier_prepare_cost_provider=self._tier_prepare_cost_ms,
        )
        self._observation_db.add_payload_eviction_observer(
            self._retire_historical_identity
        )
        self._retired_unmaterializable_total = 0
        self._retired_unmaterializable_warned = False
        self._retired_rgb_ineligible_total = 0
        self._transfer_owner = None
        self._background = None
        # Canonical production owns the same bounded depth-cache policy as the
        # streaming trainer.  Without this wiring, every admitted live camera
        # can retain its sensor-depth tensor indefinitely.
        self._sensor_depth_cache_bound = max(
            0, int(config.runtime.sensor_depth_cache_max_frames)
        )
        # The depth helper registry is process-global for historical trainer
        # callers. Reset it at each canonical runtime boundary so a closed run
        # cannot evict or retain cameras belonging to a new run.
        reset_sensor_depth_cache_state()
        set_sensor_depth_cache_bound(self._sensor_depth_cache_bound)
        # Replay page-ins remain live until the next tick boundary, when every
        # B=1 consumer has returned.  The trainer has this ledger already; the
        # canonical MappingApplication must own it too.
        self._residency_ledger = CameraResidencyLedger(
            enabled=config.runtime.offload_camera_images
        )
        self.observation_owner = ObservationOwner.from_backpressure(
            BackpressureOwner(BackpressureConfig(**dict(config.backpressure_policy))),
            representative_lookup=self._representative_identity,
            semantic_providers=(
                keyframe_admitter_proposal_provider(
                    KeyframeAdmitterConfig(**dict(config.keyframe_policy))
                ),
            ),
        )
        # Issue #279/#1259/#1342 admission-time prefetch. The camera-materialize
        # stage (`m_tick_camera_materialize`) pays a cold decode
        # (`_decode_reload_token`: file I/O + PIL/cv2 image decode) for the
        # FIRST tick that draws a newly admitted frame, because admission
        # (`_record_scene_candidate`, called from `map_frame`) and the tick
        # that later materializes that same frame for supervision
        # (`execute_fixed_topology_transaction`'s `m_tick_camera_materialize`
        # loop) are always separated by at least one full tick -- frequently
        # many ticks, at `--streaming_steps_per_frame` cadence. Nothing
        # previously used that gap: `record_candidate` performs no prefetch at
        # all.
        #
        # This decodes ahead, on a single dedicated background thread, purely
        # from the immutable `PixelReloadToken` captured on the MAIN thread at
        # admission time (see `_record_scene_candidate`) -- `_decode_reload_token`
        # is a `@staticmethod`: pure file I/O + numpy/PIL decode with no access
        # to `self._observation_db` or any other mutable shared state (verified:
        # it builds its own throwaway `SimpleNamespace` with no `_cache_pos`, so
        # it never touches the unrelated global `_DECODE_CACHE` from
        # `eligsir/io/sources/decode_cache.py` either). The ONLY state
        # shared between the background thread and the main thread is
        # `self._admission_prefetch_cache`, a plain dict keyed by `frame_id`,
        # guarded by `self._admission_prefetch_lock` -- the background thread
        # only ever writes one entry at job completion, and the main thread
        # only ever pops one entry (`_pop_admission_prefetch`, called from
        # `_replay_mapping_frame`, which already runs on the same thread as
        # every existing observation_db call). No other code path is touched:
        # the consumer of a prefetch hit is `_replay_mapping_frame`'s existing
        # `_decode_reload_token(reload_token)` call site, which now checks the
        # cache first and falls back to a normal cold decode on a miss (never
        # materialized speculatively into `resident_supervision_for_source`'s
        # HOT cache -- only the raw native RGB/depth numpy arrays are
        # prefetched; tier derivation/H2D staging/`activate_supervision`
        # remain exactly where they were, on the main thread, in the same
        # order, byte-identical). PR: implements the #1342 finding referenced
        # by issues #279 and #1259.
        self._admission_prefetch_enabled = bool(
            getattr(config.runtime, "admission_prefetch_enabled", True)
        )
        self._admission_prefetch_requested = bool(
            getattr(config.runtime, "admission_prefetch_requested", False)
        )
        self._admission_prefetch_reason = str(
            getattr(config.runtime, "admission_prefetch_reason", "disabled")
        )
        self._admission_prefetch_lock = threading.Lock()
        self._admission_prefetch_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._admission_prefetch_pending: set[int] = set()
        self._admission_prefetch_terminal_failures: OrderedDict[int, str] = OrderedDict()
        self._admission_prefetch_protected_frame_ids: set[int] = set()
        # Bounded eviction: a prefetched-but-never-drawn frame (evicted from
        # the local window, or simply never selected) would otherwise retain
        # its full decoded native RGB+depth arrays until `close()` -- an
        # unbounded host-memory leak proportional to admitted-but-undrawn
        # frame count x resolution on any ingestion path where this cache is
        # actually populated (today: none, see the module-level comment
        # above -- but the bound must exist regardless of what's live today).
        # Small on purpose: this is a same-tick-or-next-few-ticks lookahead,
        # not a general cache -- FIFO eviction of the oldest unconsumed entry
        # is the right policy (a stale prefetch nobody drew yet is also the
        # least likely to still be drawn soon).
        # Check if residency budget resolved prefetch entries; otherwise use default formula
        resolved_prefetch = config.static_options.get("_resolved_prefetch_entries", None)
        if resolved_prefetch is not None:
            self._admission_prefetch_cache_max_entries = int(resolved_prefetch)
        else:
            self._admission_prefetch_cache_max_entries = admission_prefetch_capacity_for_batch(
                int(self._batch_size)
            )
        self._admission_prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="admission-prefetch")
            if self._admission_prefetch_enabled
            else None
        )
        self._admission_prefetch_scheduled_total = 0
        self._admission_prefetch_completed_total = 0
        self._admission_prefetch_failed_total = 0
        self._admission_prefetch_terminal_failed_total = 0
        self._admission_prefetch_hits_total = 0

    def enable_ready_source_owner(
        self,
        owner: ReadySourceOwner,
        *,
        warmup_complete: bool = False,
    ) -> None:
        """Explicitly enable the ready-only GPU-owner hook before use."""

        if not isinstance(owner, ReadySourceOwner):
            raise TypeError("owner must be ReadySourceOwner")
        if type(warmup_complete) is not bool:
            raise TypeError("warmup_complete must be bool")
        if self._ready_only_tick_selector is not None:
            raise RuntimeError("ready-only source mode is already enabled")
        self._ready_source_owner = owner
        self._ready_only_tick_selector = ReadyOnlyTickSelector(
            owner,
            batch_size=self._batch_size,
            warmup_complete=warmup_complete,
        )

    @property
    def ready_source_owner(self) -> ReadySourceOwner | None:
        """Return the explicitly injected prepared-source owner, if enabled."""

        return self._ready_source_owner

    @property
    def ready_only_tick_diagnostics(self) -> Any | None:
        """Return bounded ready-only selection counters, when enabled."""

        selector = self._ready_only_tick_selector
        return None if selector is None else selector.diagnostics

    def select_ready_only_sources(
        self,
        *,
        fresh_sources: tuple[Any, ...] = (),
        replay_sources: tuple[Any, ...] = (),
        demanded_replay_sources: tuple[Any, ...] = (),
    ) -> ReadyOnlyTickSelection:
        """Poll only complete prepared leases without waiting for a source.

        This is the GPU-owner hook for the staged ready-source path.  The
        canonical plan/camera path remains the default and is not consulted
        unless a caller explicitly injected ``ready_source_owner``.
        """

        selector = self._ready_only_tick_selector
        if selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        return selector.select(
            fresh_sources=fresh_sources,
            replay_sources=replay_sources,
            demanded_replay_sources=demanded_replay_sources,
        )

    def mark_ready_source_warmup_complete(self) -> None:
        """Start starvation accounting after the caller establishes bootstrap."""

        selector = self._ready_only_tick_selector
        if selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        selector.mark_warmup_complete()

    def bind_ready_only_sources(
        self,
        selection: ReadyOnlyTickSelection,
    ) -> tuple[ReadySourceBinding, ...]:
        """JIT-bind selected leases to the current map/version fence."""

        if self._ready_only_tick_selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        map_content, topology, parameter = self._versions()
        versions = ReadySourceVersions(
            map_content_version=map_content,
            topology_version=topology,
            parameter_version=parameter,
        )
        return ReadyOnlyTickSelector.bind_current_versions(selection, versions)

    def bind_ready_only_cameras(
        self,
        selection: ReadyOnlyTickSelection,
    ) -> tuple[Any, ...]:
        """JIT-create canonical cameras from selected ready leases.

        This explicit opt-in hook deliberately stops before dispatch and does
        not update scene/replay/observation ownership.  Its source bindings
        are sampled above only after ready selection, so no stale work envelope
        can survive from source preparation into a later optimizer transaction.
        """

        from eligsir.runtime.ready_camera_binder import bind_ready_camera

        return tuple(
            bind_ready_camera(binding)
            for binding in self.bind_ready_only_sources(selection)
        )

    def _ensure_ready_transfer_owner(self):
        owner = getattr(self, "_transfer_owner", None)
        if owner is None:
            owner = get_device_transfer_owner(
                torch.device("cuda", torch.cuda.current_device()),
                max_pinned_bytes=self._runtime_policy.max_transfer_bytes,
            )
            self._transfer_owner = owner
        return owner

    def _bind_ready_bootstrap_cameras(
        self, selection: ReadyOnlyTickSelection
    ) -> tuple[Any, ...]:
        """Handoff every selected source before reading its device payloads.

        Binds cameras at the highest declared tier from the prepared-tier policy.
        """

        from eligsir.runtime.ready_camera_binder import bind_ready_camera

        owner = self._ensure_ready_transfer_owner()
        bindings = self.bind_ready_only_sources(selection)
        prepared = []
        # Determine bootstrap tier: use policy's highest tier if provided, else native
        bootstrap_tier = "native"
        prepared_tier_policy = getattr(self, "_prepared_tier_policy", None)
        if prepared_tier_policy is not None:
            bootstrap_tier = prepared_tier_policy.highest().value
        for binding in bindings:
            binding.handoff_to_compute(owner)
            prepared.append(bind_ready_camera(binding, tier=bootstrap_tier))
        return tuple(prepared)

    def bootstrap_ready_only_selection(
        self,
        selection: ReadyOnlyTickSelection,
        *,
        bootstrap_event: Any,
    ) -> bool:
        """Initialize from selected READY leases and return them to replay.

        Binds cameras at the highest declared tier from the prepared-tier policy.
        This is an explicit prepared-camera entry point.  It never calls the
        legacy MappingFrame mapper; callers must pass the compute event that
        fences the device bootstrap before the leases become replay-eligible.
        """

        selector = self._ready_only_tick_selector
        if selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        if self.initialized:
            raise RuntimeError("ready bootstrap is only valid before initialization")
        if not isinstance(selection, ReadyOnlyTickSelection):
            raise TypeError("selection must be ReadyOnlyTickSelection")
        if not callable(getattr(bootstrap_event, "query", None)):
            raise TypeError("bootstrap_event must provide a callable query() method")
        bindings = self._bind_ready_bootstrap_cameras(selection)
        pending = getattr(self, "_ready_bootstrap_bindings", None)
        if pending is None:
            pending = []
            self._ready_bootstrap_bindings = pending
        selections = getattr(self, "_ready_bootstrap_selections", None)
        if selections is None:
            selections = []
            self._ready_bootstrap_selections = selections
        known = {binding.source_identity for binding in pending}
        if any(binding.source_identity in known for binding in bindings):
            raise RuntimeError("ready bootstrap selected a duplicate source identity")
        prospective_bindings = tuple(pending) + tuple(bindings)
        prospective_selections = tuple(selections) + (selection,)
        if len(prospective_bindings) < self.bootstrap_frames:
            pending.extend(bindings)
            selections.append(selection)
            return False
        replay_leases = tuple(
            lease for prior_selection in prospective_selections
            for lease in prior_selection.leases
        )
        # This owner operation validates the entire target-lane batch under one
        # lock before changing any lease.  It runs before initialization so a
        # saturated replay lane cannot leave an initialized map with selected
        # bootstrap sources stranded in flight.
        selector.requeue_batch(
            replay_leases,
            ready_event=bootstrap_event,
            lane=ReadySourceLane.REPLAY,
        )
        # The prepared causal path requests exactly the remaining bootstrap
        # cardinality.  Keep this trim as a defensive invariant for other
        # callers too: B4 selection must never silently change a configured
        # five-source bootstrap into an eight-source initialization.
        self._initialize(prospective_bindings[: self.bootstrap_frames])
        pending.clear()
        selections.clear()
        self.mark_ready_source_warmup_complete()
        return True

    def ready_bootstrap_remaining(self) -> int:
        """Return the exact number of source bindings still needed to initialize.

        This is intentionally a cardinality-only query for the prepared
        causal boundary.  It prevents the B4 optimizer size from rounding a
        configured five-source bootstrap up to eight selected bindings.
        """

        pending = getattr(self, "_ready_bootstrap_bindings", None)
        if pending is None:
            pending = []
            self._ready_bootstrap_bindings = pending
        if not isinstance(pending, list):
            raise TypeError("ready bootstrap bindings must be a list")
        return max(0, int(self.bootstrap_frames) - len(pending))

    def release_ready_only_sources(
        self,
        selection: ReadyOnlyTickSelection,
    ) -> None:
        """Release GPU-owned ready leases after the bound transaction completes."""

        selector = self._ready_only_tick_selector
        if selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        selector.release(selection)

    def requeue_ready_only_sources(self, selection, *, ready_event: Any) -> None:
        """Return selected ready leases without ending their payload lifetime."""

        selector = self._ready_only_tick_selector
        if selector is None:
            raise RuntimeError("ready-only source mode is not enabled")
        selector.requeue(selection, ready_event=ready_event)

    def _prepared_ready_plan_for_execution(self) -> MappingWorkPlan | None:
        """Freeze only an executable READY WorkBudget plan.

        A candidate with COLD members remains scheduler-authored page-in demand;
        it must not pin the compute owner while an independently authored,
        READY-only replay plan can execute.
        """

        frozen = self._prepared_ready_frozen_work
        candidate = frozen or getattr(self, "_prepared_ready_candidate_work", None)
        if candidate is None:
            return None
        snapshot, plan = candidate

        def exact(work: MappingWorkPlan) -> tuple[SourceIdentity, ...] | None:
            sources = (*work.supervision_sources, *work.replay_sources)
            constellation = work.supervision_constellation
            raw_batch_size = getattr(constellation, "realized_batch_size", None)
            raw_camera_count = getattr(constellation, "realized_camera_count", None)
            if (
                constellation is None
                or type(raw_batch_size) is not int
                or type(raw_camera_count) is not int
                or len(set(sources)) != len(sources)
            ):
                return None
            try:
                batch_size = validate_supervision_batch_size(raw_batch_size)
                camera_count = validate_supervision_camera_count(raw_camera_count)
            except (TypeError, ValueError):
                return None
            if batch_size > int(self._batch_size) or camera_count > batch_size:
                return None
            full, remainder = divmod(batch_size, camera_count)
            expected_microbatch_sizes = (camera_count,) * full + (
                (remainder,) if remainder else ()
            )
            if (
                getattr(constellation, "microbatch_count", len(expected_microbatch_sizes))
                != len(expected_microbatch_sizes)
            ):
                return None
            declared_microbatch_sizes = getattr(
                constellation, "microbatch_sizes", expected_microbatch_sizes
            )
            if (
                declared_microbatch_sizes is None
                or tuple(declared_microbatch_sizes) != expected_microbatch_sizes
            ):
                return None
            if batch_size == int(self._batch_size) and len(sources) == batch_size:
                return tuple(sources)
            if not (
                bool(
                    getattr(getattr(self, "config", None), "static_options", {}).get(
                        "prepared_ready_source_insertion", False
                    )
                )
                and batch_size == 4
                and len(work.supervision_sources) == 1
                and len(work.replay_sources) == 3
            ):
                return None
            source = work.supervision_sources[0]
            stored = getattr(
                getattr(self, "_observation_db", None),
                "prepared_commit_obligation_for_source",
                lambda _: None,
            )(source)
            if not (isinstance(stored, tuple) and len(stored) == 4):
                return None
            context = stored[3]
            if (
                getattr(context, "lane", None) is not ReadySourceLane.FRESH
                or getattr(getattr(context, "disposition", None), "semantic_action", None)
                is not SemanticAction.COMMIT
                or self._native_geometry_evidence_purpose() is not None
            ):
                return None
            from eligsir.mapping.scheduler.work_budget import WorkBudgetActuator, _boolean_request

            if _boolean_request(
                tuple(getattr(getattr(context, "control_axes", None), "proposals", ())),
                WorkBudgetActuator.LIVE_POPULATION_AUTHORIZATION,
            ):
                return None
            return tuple(sources)

        sources = exact(plan)
        if sources is None and frozen is None:
            rearmed = self._rearm_prepared_pending_supervision()
            if rearmed is not None:
                candidate = self._prepared_ready_candidate_work = rearmed
                snapshot, plan = candidate
                sources = exact(plan)
            elif (
                not plan.supervision_sources
                and getattr(self, "_ready_source_owner", None) is not None
            ):
                snapshot = self._fresh_idle_snapshot(snapshot)
                candidate = self._prepared_ready_candidate_work = (
                    snapshot,
                    self._idle_replay_plan(snapshot, plan),
                )
                plan = candidate[1]
                sources = exact(plan)
        if sources is None:
            return None
        owner = getattr(self, "_ready_source_owner", None)
        if owner is None:
            self._prepared_ready_frozen_work = candidate
            return plan
        fresh = frozenset(plan.supervision_sources)
        replay = frozenset(plan.replay_sources)
        readiness = owner.exact_readiness(
            fresh_sources=fresh,
            replay_sources=replay,
            exact_sources=sources,
        )
        if all(state in ("READY", "EVENT_PENDING") for _source, state in readiness):
            self._prepared_ready_frozen_work = candidate
            return plan
        self._retain_pending_replay_trial_sources(
            tuple(getattr(plan, "replay_trial_sources", ())),
            limit=len(plan.replay_sources),
        )
        self._prepared_ready_frozen_work = None
        ready_replay_sources = frozenset(owner.ready_replay_sources())
        fallback_replay_sources = ready_replay_sources
        if len(fallback_replay_sources) < int(self._batch_size):
            replay_resident_sources = getattr(owner, "replay_resident_sources", None)
            if callable(replay_resident_sources):
                fallback_replay_sources = frozenset(replay_resident_sources())
        if len(fallback_replay_sources) < int(self._batch_size):
            return None
        fallback_snapshot = self._fresh_idle_snapshot(snapshot)
        fallback = self._idle_replay_plan(
            fallback_snapshot,
            plan,
            ready_replay_sources=fallback_replay_sources,
        )
        if (
            getattr(fallback.supervision_constellation, "realized_batch_size", None)
            != int(self._batch_size)
            or exact(fallback) is None
        ):
            return None
        self._prepared_ready_frozen_work = (fallback_snapshot, fallback)
        return fallback

    def _retain_pending_replay_trial_sources(
        self, sources: tuple[SourceIdentity, ...], *, limit: int
    ) -> None:
        """Keep only bounded, source-only trial demand across READY fallback."""

        if limit < 1 or not sources:
            return
        pending = tuple(getattr(self, "_pending_replay_trial_sources", ()))
        self._pending_replay_trial_sources = tuple(
            dict.fromkeys((*pending, *sources))
        )[:limit]

    def _prune_pending_replay_trial_sources(self) -> tuple[SourceIdentity, ...]:
        """Drop demand that no longer has durable, RGB-authorized page-in authority."""

        pending = tuple(getattr(self, "_pending_replay_trial_sources", ()))
        replay_modalities = getattr(self, "_replay_modalities", {})
        retired = getattr(self, "_retired_replay_sources", {})
        frame_ids = getattr(self, "_historical_frame_ids_by_source", {})
        database = getattr(self, "_observation_db", None)
        retained = tuple(
            source
            for source in pending
            if source in replay_modalities
            and source not in retired
            and bool(getattr(replay_modalities[source], "rgb_supervision", False))
            and source in frame_ids
            and database is not None
            and database.prepared_commit_obligation_for_source(source) is None
            and self._prepared_cold_pagein_capable(database.get_frame(frame_ids[source]))
        )
        self._pending_replay_trial_sources = retained
        return retained

    def _advance_prepared_ready_plan(self, plan: MappingWorkPlan) -> None:
        """Replace the current prepared authority with its replay successor."""

        frozen = getattr(self, "_prepared_ready_frozen_work", None)
        if frozen is None or frozen[1] is not plan:
            return
        self._prepared_ready_frozen_work = None
        if (rearmed := self._rearm_prepared_pending_supervision()) is not None:
            self._prepared_ready_candidate_work = rearmed
            return
        snapshot = self._fresh_idle_snapshot(frozen[0])
        self._prepared_ready_candidate_work = (
            snapshot,
            self._idle_replay_plan(snapshot, plan),
        )

    def _rearm_prepared_pending_supervision(
        self,
    ) -> tuple[ControlSnapshot, MappingWorkPlan] | None:
        """Resolve the oldest retained COMMIT against eligible resident replay."""
        pending = getattr(self, "_pending_supervision_modalities", ())
        observation_db = getattr(self, "_observation_db", None)
        next_obligation = getattr(
            observation_db, "next_prepared_commit_obligation_source", None
        )
        obligation_for = getattr(
            observation_db, "prepared_commit_obligation_for_source", None
        )
        if not callable(next_obligation) or not callable(obligation_for):
            return None
        source = next_obligation()
        if source is None:
            return None
        stored = obligation_for(source)
        if not (isinstance(stored, tuple) and len(stored) == 4):
            raise RuntimeError("prepared COMMIT obligation lacks execution context")
        _tracking, _disposition, _token, context = stored
        if (
            context is None
            or context.lane is not ReadySourceLane.FRESH
            or context.disposition.semantic_action is not SemanticAction.COMMIT
        ):
            raise RuntimeError("prepared COMMIT service context is invalid")
        prior = context.snapshot
        replay_modalities, reserved = self._shortlisted_replay_view(
            snapshot_id=int(prior.snapshot_id),
            control_tick_epoch=int(getattr(self, "_control_tick_epoch", 0)),
        )
        replay_resident_sources = getattr(
            getattr(self, "_ready_source_owner", None),
            "replay_resident_sources",
            None,
        )
        if callable(replay_resident_sources):
            resident = frozenset(replay_resident_sources())
            replay_modalities = OrderedDict(
                (candidate, modality)
                for candidate, modality in replay_modalities.items()
                if candidate in resident
            )
            if reserved not in replay_modalities:
                reserved = None
        purpose = self._native_geometry_evidence_purpose()
        source_insertion = bool(
            getattr(getattr(self, "config", None), "static_options", {}).get(
                "prepared_ready_source_insertion", False
            )
        )
        from eligsir.mapping.scheduler.work_budget import WorkBudgetActuator, _boolean_request

        live_population_authorization = _boolean_request(
            tuple(context.control_axes.proposals),
            WorkBudgetActuator.LIVE_POPULATION_AUTHORIZATION,
        )
        bounded_source_service = (
            source_insertion
            and int(getattr(self, "_batch_size", 1)) > 4
            and purpose is None
            and not live_population_authorization
        )
        if bounded_source_service and len(replay_modalities) < 3:
            return None
        if bounded_source_service:
            selected = tuple(replay_modalities)[:3]
            if reserved is not None and reserved not in selected:
                selected = (*selected[:2], reserved)
            replay_modalities = OrderedDict(
                (candidate, modality)
                for candidate, modality in replay_modalities.items()
                if candidate in selected
            )
        content, topology, parameters = self._versions()
        replay_weights = getattr(self, "_replay_weights", {})
        snapshot = replace(
            prior,
            pending_sources=(source,),
            newest_frontier=source,
            candidate_queue_depth=len(pending),
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            gaussian_count=(
                int(self.gaussians.get_xyz.shape[0])
                if bool(getattr(self, "initialized", False))
                else 0
            ),
            optimizer_steps_completed=int(getattr(self, "optimizer_steps", 0)),
            supervision_views_completed=int(
                getattr(self, "_supervision_views_completed", 0)
            ),
            control_tick_epoch=int(getattr(self, "_control_tick_epoch", 0)),
            supervision_candidates=(source,),
            replay_candidates=tuple(replay_modalities),
            replay_work_modalities=tuple(replay_modalities.values()),
            replay_candidate_weights=tuple(
                float(replay_weights.get(candidate, 1.0))
                for candidate in replay_modalities
            ),
            reserved_global_replay_candidate=reserved,
        )
        return snapshot, self._work_budget_owner.resolve(
            snapshot,
            disposition=context.disposition,
            proposals=context.control_axes.proposals,
            native_geometry_evidence_purpose=purpose is not None,
            source_depth_requires_native=source_insertion,
        )

    def _invalidate_prepared_ready_plan_for_source(self, source: Any) -> None:
        """Replan only after terminal retirement invalidates an exact identity."""

        frozen = getattr(self, "_prepared_ready_frozen_work", None)
        candidate = getattr(self, "_prepared_ready_candidate_work", None)
        active = frozen or candidate
        if active is None or source not in (
            *active[1].supervision_sources,
            *active[1].replay_sources,
        ):
            return
        self._prepared_ready_frozen_work = None
        snapshot = self._fresh_idle_snapshot(active[0])
        self._prepared_ready_candidate_work = (
            snapshot,
            self._idle_replay_plan(snapshot, active[1]),
        )

    @property
    def background(self) -> Any:
        """Return the standard render-background tensor for this run.

        ``None`` until the first bootstrap tick sets it (mirrors ``self.pipe``/
        ``self.gaussians`` which are always set, but this cannot be computed
        before ``self._runtime_policy.white_background`` is resolved against a
        real device at bootstrap). Read-only external accessor for issue
        #1302's live/ROS2 eval-report path -- see
        ``eligsir.runtime.live_eval_holdout``.
        """

        return self._background

    @property
    def render_fn(self) -> Any:
        """Return the project's standard ``render()`` callable.

        Exactly the ``gaussian_renderer.render`` this runtime itself calls
        for every tick (bound into this module's globals by
        ``_load_accelerator_dependencies()`` during ``__init__``, so it is
        always available once a ``CanonicalMappingRuntime`` exists). Exposed
        so an external caller with only a handle to this runtime (e.g. via
        ``ComposedMappingApplication.runtime``) can render held-out cameras
        against it without importing ``gaussian_renderer`` a second time.
        """

        return render

    def _record_scene_candidate(
        self, cam: Any, frame: StreamingObservation, is_train: bool, is_holdout: bool
    ) -> None:
        """Record one scene candidate through the shared history façade."""

        mapping_frame = getattr(frame, "mapping_frame", None)
        source = getattr(mapping_frame, "source", None)
        if source is None:
            raise RuntimeError("canonical scene candidate has no SourceIdentity")
        cam._source_identity = source
        record = self._observation_db.record_candidate(
            cam,
            frame,
            source_identity=source,
            is_train=is_train,
            is_holdout=is_holdout,
            visible_uids=getattr(cam, "_observed_gaussian_uids", None),
            submap_id=0,
            historical_eligible=bool(is_train),
            modality_eligibility=getattr(self, "_replay_modalities", {}).get(source),
        )
        _initialize_train_evwb(
            record, cam, getattr(self, "_optimization_policy", None), is_train=is_train
        )
        if is_train:
            self._register_historical_record(
                record,
                source,
                protected=bool(getattr(cam, "_is_bootstrap", False)),
            )
        cold_pixels = self._observation_db.get_cold_pixels(record.frame_id)
        cold_handle_type = globals().get("ColdPixelHandle")
        if (
            cold_pixels is not None
            and cold_handle_type is not None
            and not isinstance(cold_pixels, cold_handle_type)
        ):
            raise RuntimeError("observation DB published an invalid cold pixel owner")
        if is_train:
            self._maybe_schedule_admission_prefetch(record.frame_id, cold_pixels)
            self._maybe_prematerialize_admitted_supervision(cam, record.frame_id)

    def _prematerialize_supervision_tier(
        self, cam: Any, frame_id: int, *, register_residency: bool = True
    ) -> bool:
        """Run the tick's own miss path for one camera now, so its first draw hits.

        Issue #279/#1342 follow-up (2026-09-04). ``record_candidate`` /
        ``record_historical_candidate`` build no tier and
        ``activate_supervision`` only runs inside
        ``execute_fixed_topology_transaction`` AFTER a view already missed, so
        every admitted frame's first supervision draw is a guaranteed miss --
        measured at ~54% of all ``m_tick_camera_materialize`` misses on the
        fr3 legacy recipe (a ~25% floor that neither a larger HOT window nor
        COLD promotion can touch, since the fresh view is not a replay
        candidate). This runs the exact same materialize -> H2D -> prepare ->
        write-through sequence the tick's miss path runs, just at admission,
        so the first draw is a HOT hit (``_bind_resident_supervision`` for a
        keyframe camera; ``promote_resident_historical_frame`` for a
        packet-only source, which builds its transaction camera around the
        DB-owned HOT tensors and needs no pre-built camera at all).

        Sound only for a concrete tier (gated in ``__init__``); a no-op unless
        ``--streaming_admission_prematerialize`` is set. Best-effort: any
        failure leaves the frame exactly as it was (the miss path still
        works) -- the tick never depends on this having run. Returns whether
        the tier was published.

        ``register_residency`` is False for the packet-only path's throwaway
        proxy camera (issue review, 2026-09-04): that proxy is never in
        ``scene.hot_camera_ids()``, so registering it with the residency
        ledger makes the NEXT tick's ``release()`` see it as a stale batchable
        camera and submit a real D2H for a tensor nothing owns (the tier
        tensors belong to the DB's ``CacheHandle``) -- wasted transfer-pool
        contention with zero benefit, and it also skews
        ``cumulative_paged_in``/``paged_out``. For the same reason, skip
        ``prepare_camera_for_render``: it runs ``make_gsplat_camera_tensors``
        from the proxy's fabricated pose/intrinsics, which
        ``promote_resident_historical_frame`` discards and rebuilds from the
        resident record anyway.
        """

        tier = self._prematerialize_tier_now()
        if tier is None or self._transfer_owner is None:
            return False
        if getattr(cam, "original_image", None) is None:
            return False
        # Import directly rather than through this module's lazily-bound
        # globals (`_lazy_imports`): those are None until `__init__` ran, and
        # this method is also exercised on `object.__new__` runtimes in tests.
        from scene.cameras import prepare_camera_for_render
        from eligsir.mapping.observation_db import CacheHandle as _CacheHandle
        from eligsir.mapping.resolution_materialization import (
            materialize_resolution_tier as _materialize_resolution_tier,
        )

        try:
            _materialize_resolution_tier(
                cam,
                tier,
                resampling_mode=str(
                    self.config.static_options.get("resolution_resampling_mode", "legacy")
                ),
                discontinuity_abs_floor_m=float(
                    self.config.static_options.get("resolution_discontinuity_abs_floor_m", 0.05)
                ),
                discontinuity_rel_frac=float(
                    self.config.static_options.get("resolution_discontinuity_rel_frac", 0.02)
                ),
                depth_downsample=str(
                    self.config.static_options.get("depth_loss_downsample", "nearest")
                ),
            )
            self._transfer_owner.prefetch_cameras((cam,))
            if register_residency:
                # The transfer owner's device is valid from composition time;
                # `self.gaussians.get_xyz` is not until the model is
                # initialized (bootstrap frames arrive first and would
                # AttributeError).
                prepare_camera_for_render(cam, device=str(self._transfer_owner.device))
                self._residency_ledger.note_paged_in(cam)
            self._observation_db.activate_supervision(
                frame_id,
                {
                    tier.value: _CacheHandle(
                        rgb=cam.original_image,
                        depth=getattr(cam, "_sensor_depth_cache", None),
                        intrinsics=(
                            int(cam.image_width), int(cam.image_height),
                            getattr(cam, "fx", None), getattr(cam, "fy", None),
                            getattr(cam, "cx", None), getattr(cam, "cy", None),
                        ),
                        score_provenance=getattr(cam, "score_provenance", None),
                    )
                },
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; the miss path remains authoritative
            self._admission_prematerialize_failed += 1
            if not self._admission_prematerialize_warned:
                self._admission_prematerialize_warned = True
                print(
                    f"[admission-prefetch] prematerialize failed (first of run): {exc!r}",
                    flush=True,
                )
            return False
        self._admission_prematerialize_total += 1
        return True

    def _maybe_prematerialize_admitted_supervision(self, cam: Any, frame_id: int) -> None:
        """Keyframe path: the admitted ``Camera`` already carries native pixels."""

        self._prematerialize_supervision_tier(cam, frame_id)

    def _maybe_prematerialize_retained_supervision(
        self, record: Any, handle: Any, *, packet: Any | None = None
    ) -> None:
        """Packet-only path: a DEFER/COMMIT source retained as CPU history.

        These are the ~5 of 6 fr3 sources that never become keyframes yet
        still arrive as the B4 fresh view (``newest_frontier`` is every
        submitted packet). They own no ``Camera`` until their first draw
        builds one, so the tier is built on a throwaway proxy carrying exactly
        the fields ``materialize_resolution_tier`` reads; on the first draw
        ``_camera_for_source`` finds the HOT tier and takes the
        ``promote_resident_historical_frame`` branch (no cold restore).
        Pixels come from the PACKET, not the retained handle: on the
        causal_replay TUM recipe every packet carries a ``reload_token``
        (`replay.py::packet_from_streaming_frame`) AND fully resident rgb/depth,
        but the retention sink stores ``rgb=None`` in the handle whenever a
        token exists -- so reading the handle silently skipped all 2101
        non-keyframes per fr3 run (measured 2026-09-04, diag
        ``skip=no_resident_rgb reload_token=True`` x2101). Skipped only when
        neither the packet nor the handle holds resident pixels.

        Unlike the canonical insertion path
        (``contracts/streaming_bridge.py::mapping_frame_source_arrays``),
        this hook never runs through ``_cpu_numpy`` validation -- the handle
        and packet arrays are consumed directly. Reviewer finding
        (2026-09-04): validate depth units and RGB/depth shape agreement
        before building the proxy, same as the canonical path, rather than
        assuming HWC and metric depth.
        """

        if self._prematerialize_tier_now() is None:
            return
        rgb = getattr(handle, "rgb", None)
        depth = getattr(handle, "depth", None)
        if not isinstance(rgb, np.ndarray) and packet is not None:
            packet_rgb = getattr(getattr(packet, "rgb", None), "data", None)
            if packet_rgb is not None:
                rgb = np.asarray(packet_rgb)
                packet_depth = getattr(getattr(packet, "depth", None), "data", None)
                depth = None if packet_depth is None else np.asarray(packet_depth)
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
            return
        # `mapping_frame_source_arrays` moves a CHW rgb array to HWC before
        # any shape comparison; the proxy assumed HWC unconditionally, so
        # match that here first.
        if rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        if isinstance(depth, np.ndarray):
            depth_spec = getattr(getattr(packet, "depth", None), "spec", None) if packet is not None else None
            units = (
                getattr(depth_spec, "units", None)
                if depth_spec is not None
                else getattr(handle, "depth_units", None)
            )
            if str(units or "").lower() not in {"m", "meter", "meters", "metre", "metres"}:
                # Non-metric (or unknown-unit) depth: the tick's real path
                # (which does go through the canonical validated arrays)
                # builds this frame's supervision later -- neither a success
                # nor a crash, just declined here.
                return
            if depth.shape[:2] != rgb.shape[:2]:
                return
        try:
            import torch as torch_module
            from types import SimpleNamespace
            from eligsir.utils.score_contract import declare_camera_score_provenance

            # The retained handle carries the canonical CameraIntrinsics
            # object (attrs), not the tick-path CacheHandle's 6-tuple.
            intr = handle.intrinsics
            if isinstance(intr, (tuple, list)):
                width, height, fx, fy, cx, cy = tuple(intr)[:6]
            else:
                width, height = int(intr.width), int(intr.height)
                fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
            # Same integer->[0,1] float conversion as the canonical cold path
            # (`contracts/streaming_bridge.py::mapping_frame_source_arrays`),
            # so the pre-materialized tier is byte-identical to a tick-built one
            # and `declare_camera_score_provenance` accepts it (it refuses
            # integer "presentation" dtypes).
            rgb_hwc = rgb[..., :3]
            if np.issubdtype(rgb_hwc.dtype, np.integer):
                rgb_hwc = rgb_hwc.astype(np.float32) / float(np.iinfo(rgb_hwc.dtype).max)
            else:
                rgb_hwc = rgb_hwc.astype(np.float32, copy=False)
            proxy = SimpleNamespace(
                original_image=torch_module.from_numpy(np.ascontiguousarray(rgb_hwc.transpose(2, 0, 1))),
                _sensor_depth_cache=(
                    None if not isinstance(depth, np.ndarray)
                    else torch_module.from_numpy(np.ascontiguousarray(depth))
                ),
                image_width=int(width), width=int(width),
                image_height=int(height), height=int(height),
                fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                gsplat=None,
                _canonical_native_resolution_state=None,
                _canonical_resolution_binding_native_state=None,
                _canonical_resolution_tier=None,
                has_pending_image=False,
                R=np.eye(3, dtype=np.float32), T=np.zeros(3, dtype=np.float32),
                FoVx=1.0, FoVy=1.0, znear=0.01, zfar=100.0,
                world_view_transform=torch_module.eye(4), full_proj_transform=torch_module.eye(4),
                camera_center=torch_module.zeros(3), uid=int(record.frame_id),
            )
            declare_camera_score_provenance(proxy, context="admission_prematerialize")
        except Exception:  # noqa: BLE001 - best-effort
            return
        self._prematerialize_supervision_tier(
            proxy, int(record.frame_id), register_residency=False
        )

    def _record_ready_scene_candidate(
        self, cam: Any, binding: Any, is_train: bool, is_holdout: bool
    ) -> None:
        """Attach an attested READY lease to its transient or durable record."""

        if is_holdout:
            raise ValueError("prepared bootstrap cameras cannot be holdout views")
        metadata = getattr(binding, "metadata", None)
        source = getattr(binding, "source_identity", None)
        if metadata is None or source is None or metadata.source_identity is not source:
            raise RuntimeError("prepared bootstrap binding lost authoritative source metadata")
        cam._source_identity = source
        record = self._observation_db.record_prepared_candidate(
            source_identity=source,
            pose_c2w=metadata.pose_c2w,
            timestamp=source.timestamp_seconds,
            intrinsics=metadata.intrinsics,
            ready_source_reference=binding,
            modality_eligibility=binding.modality_eligibility,
            is_train=is_train,
            submap_id=0,
        )
        _initialize_train_evwb(
            record, cam, getattr(self, "_optimization_policy", None), is_train=is_train
        )
        if record.historical_eligible:
            self._register_historical_record(record, source, protected=True)
            return
        self._prepared_device_transient_retired_total = int(
            getattr(self, "_prepared_device_transient_retired_total", 0)
        ) + 1
        if self._observation_db.get_cold_pixels(record.frame_id) is not None:
            raise RuntimeError("prepared bootstrap unexpectedly registered cold pixels")

    def _maybe_schedule_admission_prefetch(
        self, frame_id: int, cold_pixels: Any | None
    ) -> None:
        """Kick off a background native RGB/depth decode for a just-admitted frame.

        Narrow by construction: only fires when there is real decode work to
        move off the critical path (a `PixelReloadToken`-backed cold handle
        whose arrays are not already resident in memory) and only ever reads
        the immutable token captured here, on the main thread, right now --
        the background job touches nothing else. See the `__init__` comment
        above `self._admission_prefetch_executor` for the full invariant.
        """

        executor = getattr(self, "_admission_prefetch_executor", None)
        if executor is None or cold_pixels is None:
            return
        if getattr(cold_pixels, "rgb", None) is not None and (
            getattr(cold_pixels, "depth", None) is not None
        ):
            # Already resident (e.g. live/eager-decoded sources) -- nothing to
            # prefetch, decoding again would be pure waste.
            return
        reload_token = getattr(cold_pixels, "reload_token", None)
        if reload_token is None:
            return
        with self._admission_prefetch_lock:
            if (
                frame_id in self._admission_prefetch_cache
                or frame_id in self._admission_prefetch_pending
            ):
                return
            self._admission_prefetch_pending.add(frame_id)
        try:
            executor.submit(self._prefetch_decode_worker, frame_id, reload_token)
        except Exception:
            with self._admission_prefetch_lock:
                self._admission_prefetch_pending.discard(frame_id)
            raise
        self._admission_prefetch_scheduled_total += 1

    def _set_admission_prefetch_protected_sources(
        self, sources: frozenset[SourceIdentity]
    ) -> None:
        """Protect frozen exact-plan decodes until their resolver claims them."""

        frame_ids = {
            int(frame_id)
            for source in sources
            if (frame_id := self._historical_frame_ids_by_source.get(source)) is not None
        }
        max_entries = int(getattr(self, "_admission_prefetch_cache_max_entries", 8))
        if len(frame_ids) > max_entries:
            raise ValueError("exact prepared demand exceeds admission-prefetch capacity")
        with self._admission_prefetch_lock:
            self._admission_prefetch_protected_frame_ids = frame_ids

    def _prefetch_decode_worker(self, frame_id: int, reload_token: Any) -> None:
        """Background-thread body: pure decode, never touches the observation DB.

        Must never raise into the executor -- a prefetch is a best-effort
        optimization, not a correctness dependency (the miss path it
        shortcuts remains fully functional and is exercised on any failure,
        cache eviction, or disabled config).
        """

        try:
            if getattr(reload_token, "provider", None) == "prepared_cold_disk_rgbd_v1":
                rgb, depth = self._observation_db.load_cold_payload_from_token(reload_token)
            else:
                rgb, depth = self._decode_reload_token(reload_token)
        except Exception as error:
            self._admission_prefetch_failed_total += 1
            if getattr(reload_token, "provider", None) == "prepared_cold_disk_rgbd_v1":
                with self._admission_prefetch_lock:
                    self._admission_prefetch_terminal_failures[int(frame_id)] = (
                        f"{type(error).__name__}: {error}"
                    )
            return
        else:
            with self._admission_prefetch_lock:
                cache = self._admission_prefetch_cache
                max_entries = getattr(self, "_admission_prefetch_cache_max_entries", 8)
                protected = self._admission_prefetch_protected_frame_ids
                while len(cache) >= max_entries and frame_id not in cache:
                    victim = next(
                        (candidate for candidate in cache if candidate not in protected),
                        None,
                    )
                    if victim is None:
                        break
                    cache.pop(victim)
                if len(cache) < max_entries or frame_id in cache:
                    cache[frame_id] = (rgb, depth)
            self._admission_prefetch_completed_total += 1
        finally:
            with self._admission_prefetch_lock:
                self._admission_prefetch_pending.discard(frame_id)

    def _pop_admission_prefetch(
        self, frame_id: int | None
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Consume (at most once) a completed prefetch for *frame_id*, if any."""

        if frame_id is None:
            return None
        lock = getattr(self, "_admission_prefetch_lock", None)
        if lock is None:
            # Not initialized (e.g. a test double built via `object.__new__`
            # that never ran `__init__`) -- behave exactly like a miss.
            return None
        with lock:
            result = self._admission_prefetch_cache.pop(frame_id, None)
        if result is not None:
            self._admission_prefetch_hits_total += 1
        return result

    def admission_prefetch_telemetry(self) -> dict[str, int | bool | str]:
        """Read-only snapshot of admission-prefetch state, for reports/tests.

        ``requested``/``reason`` (issue #1543 finding A) distinguish what the
        user actually asked for from the effective ``enabled`` value, which
        ``causal_prepared_ready_b4`` can force on independently of the raw
        ``--streaming_admission_prefetch`` flag.
        """

        return {
            "requested": self._admission_prefetch_requested,
            "enabled": self._admission_prefetch_enabled,
            "reason": self._admission_prefetch_reason,
            "scheduled": self._admission_prefetch_scheduled_total,
            "completed": self._admission_prefetch_completed_total,
            "failed": self._admission_prefetch_failed_total,
            "terminal_failed": int(
                getattr(self, "_admission_prefetch_terminal_failed_total", 0)
            ),
            "hits": self._admission_prefetch_hits_total,
            # Admission-time tier pre-materialize (sibling mechanism, same
            # close-out line): 0 on a run self-diagnoses "flag off or never
            # fired" without source-archaeology -- the exact ambiguity the
            # 2026-08-31 prefetch attempts left unresolved.
            "prematerialized": int(getattr(self, "_admission_prematerialize_total", 0)),
            # Distinguishes "declined" (flag off, no pixels yet, tier
            # unresolved -- an expected no-op) from "attempted and crashed
            # inside the best-effort except" (issue review, 2026-09-04): the
            # broad `except Exception` in `_prematerialize_supervision_tier`
            # must stay broad (it also covers torch.cuda.OutOfMemoryError),
            # so without this counter a silently-always-failing prematerialize
            # path was indistinguishable from one that never fires.
            "prematerialize_failed": int(getattr(self, "_admission_prematerialize_failed", 0)),
        }

    def _retain_deferred_observation(
        self,
        disposition: Any,
        permit: Any,
        packet: Any,
        observation: Any,
    ) -> Any:
        """Retain one DEFER as CPU/reload history without mapping authority."""

        from eligsir.mapping.admission.observation_owner import ColdRetentionPermit
        from eligsir.mapping.observation_db import ColdPixelHandle

        if not isinstance(permit, ColdRetentionPermit):
            raise TypeError("deferred retention requires ColdRetentionPermit")
        if permit.source != disposition.source or permit.snapshot_id != disposition.snapshot_id:
            raise ValueError("cold retention permit does not match disposition")
        if packet.source != disposition.source or observation.source != disposition.source:
            raise ValueError("deferred packet/tracking identity mismatch")
        pose = getattr(observation, "pose", None)
        if pose is None:
            raise ValueError("deferred historical observation requires a pose")
        source_key = packet.source.fingerprint()
        cache_owned = self._observation_db.cold_payload_disk_enabled
        reload_token = None if cache_owned else getattr(packet, "reload_token", None)
        rgb = None
        depth = None
        if cache_owned or reload_token is None:
            rgb = self._cache_owned_cpu_payload(packet.rgb)
            if packet.depth is not None:
                depth = self._cache_owned_cpu_payload(packet.depth)
        handle = ColdPixelHandle(
            source_identity=source_key,
            rgb=rgb,
            depth=depth,
            intrinsics=packet.intrinsics,
            reload_token=reload_token,
            rgb_encoding=str(packet.rgb.spec.encoding),
            depth_encoding=(
                None if packet.depth is None else str(packet.depth.spec.encoding)
            ),
            depth_units=(
                None if packet.depth is None else packet.depth.spec.units
            ),
        )
        record = self._observation_db.record_historical_candidate(
            source_identity=packet.source,
            pose_c2w=pose.matrix,
            timestamp=packet.source.timestamp_seconds,
            intrinsics=packet.intrinsics,
            cold_pixels=handle,
            modality_eligibility=disposition.modality_eligibility,
            disposition_snapshot_id=disposition.snapshot_id,
            semantic_action=disposition.semantic_action.value,
        )
        self._register_historical_record(record, disposition.source)
        self._maybe_prematerialize_retained_supervision(record, handle, packet=packet)
        return record

    def configure_prepared_cold_payload_spill(
        self, root: object, *, max_bytes: int | None = None
    ) -> None:
        """Configure source-independent COLD backing during composition setup."""
        self._observation_db.configure_cold_payload_spill(root, max_bytes=max_bytes)
        if getattr(self, "_admission_prefetch_executor", None) is None:
            self._admission_prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="admission-prefetch"
            )

    def can_accept_prepared_cold_payload(self, packet: Any) -> bool:
        """Packet-only cache capacity check before tracking or semantic commit."""
        from eligsir.contracts import MemoryDomain, SensorPacket

        if not isinstance(packet, SensorPacket):
            raise TypeError("prepared cold capacity preflight requires SensorPacket")
        payloads = (packet.rgb,) if packet.depth is None else (packet.rgb, packet.depth)
        if any(
            payload.spec.memory_domain not in {MemoryDomain.HOST, MemoryDomain.PINNED_HOST}
            or not isinstance(payload.data, np.ndarray)
            for payload in payloads
        ):
            raise ValueError("prepared cold backing accepts only CPU ndarray packets")
        return self._observation_db.can_accept_prepared_cold_payload(
            payload_bytes=sum(int(payload.data.nbytes) for payload in payloads)
        )

    def capture_packet_for_history(self, packet: Any) -> Any:
        """Return a source-independent host packet before transport reuse.

        The common session invokes this before tracking and the source-consumed
        callback.  It is deliberately packet-only: semantic retention remains
        owned by the later disposition sink.
        """
        from eligsir.contracts import BufferOwnership, BufferRef, SensorPacket
        from eligsir.contracts.control import _owned_tracking_evidence
        from eligsir.runtime.prepared_ingress import PreparedIngressDeferred

        if not isinstance(packet, SensorPacket):
            raise TypeError("history capture requires SensorPacket")
        if not self._observation_db.cold_payload_disk_enabled:
            return packet
        if not self.can_accept_prepared_cold_payload(packet):
            raise PreparedIngressDeferred(
                "cache-owned history capacity is unavailable; source remains pending"
            )

        def owned(payload: Any) -> BufferRef:
            array = self._cache_owned_cpu_payload(payload)
            return BufferRef(
                data=array,
                spec=replace(payload.spec, contiguous=bool(array.flags.c_contiguous)),
                ownership=BufferOwnership.OWNED,
            )

        return replace(
            packet,
            rgb=owned(packet.rgb),
            depth=None if packet.depth is None else owned(packet.depth),
            tracking=(
                None
                if packet.tracking is None
                else _owned_tracking_evidence(packet.tracking)
            ),
            reload_token=None,
        )

    def cache_causal_bypass_packet(self, packet: Any, *, role: str) -> Any:
        """Capture a held-out/guard CPU packet before source consumption."""
        if role not in {"holdout", "guard"}:
            raise ValueError("causal bypass cache role must be holdout or guard")
        packet = self.capture_packet_for_history(packet)
        self.cache_nontraining_packet(packet, role=role)
        return packet

    def cache_nontraining_packet(self, packet: Any, *, role: str) -> None:
        """Catalog one captured source without granting it mapping eligibility."""
        from eligsir.contracts import BufferOwnership, SensorPacket
        from eligsir.mapping.observation_db import ColdPixelHandle

        if role not in {"holdout", "guard", "rejected", "terminal"}:
            raise ValueError("non-training cache role is invalid")
        if not isinstance(packet, SensorPacket):
            raise TypeError("non-training cache requires SensorPacket")
        if not self._observation_db.cold_payload_disk_enabled:
            return
        payloads = (packet.rgb,) + (() if packet.depth is None else (packet.depth,))
        if any(payload.ownership is not BufferOwnership.OWNED for payload in payloads):
            raise ValueError("non-training cache requires a captured packet")
        pose = getattr(getattr(packet, "tracking", None), "pose", None)
        if pose is None and role != "terminal":
            raise ValueError("non-training cache requires an accepted packet pose")
        self._observation_db.record_nontraining_cache(
            source_identity=packet.source,
            pose_c2w=None if pose is None else pose.matrix,
            timestamp=packet.source.timestamp_seconds,
            intrinsics=packet.intrinsics,
            cold_pixels=ColdPixelHandle(
                source_identity=packet.source.fingerprint(),
                rgb=packet.rgb.data,
                depth=None if packet.depth is None else packet.depth.data,
                intrinsics=packet.intrinsics,
                rgb_encoding=str(packet.rgb.spec.encoding),
                depth_encoding=(
                    None if packet.depth is None else str(packet.depth.spec.encoding)
                ),
                depth_units=(
                    None if packet.depth is None else packet.depth.spec.units
                ),
            ),
            role=role,
        )

    def cache_terminal_packet(self, packet: Any, outcome: Any) -> None:
        """Retain a captured terminal source without inventing tracking or replay."""
        if getattr(outcome, "accepted", True) or getattr(outcome, "disposition", None) is not None:
            return
        if not self._observation_db.cold_payload_disk_enabled:
            return
        if self._observation_db.frame_id_for_source(packet.source) is None:
            self.cache_nontraining_packet(packet, role="terminal")

    def _poll_cold_payload_spill(self) -> None:
        """Publish one completed spill group and launch no-fill pending work."""
        future = getattr(self, "_cold_payload_spill_future", None)
        if future is not None:
            if not future.done():
                return
            self._cold_payload_spill_future = None
            receipts = future.result()
            if receipts:
                self._observation_db.commit_cold_payload_spill_receipts(receipts)
        if self._observation_db.pending_cold_payload_spill_count == 0:
            return
        executor = getattr(self, "_admission_prefetch_executor", None)
        if executor is None:
            raise RuntimeError("cold payload backing requires the admission-prefetch executor")
        self._cold_payload_spill_future = executor.submit(
            self._observation_db.write_pending_cold_payload_spills,
            max_items=max(1, int(getattr(self, "_batch_size", 1))),
        )

    def _commit_prepared_cold_payload_spill_receipts(self) -> None:
        """Publish cache-worker receipts before the next mapping selection."""
        bridge = getattr(self, "_prepared_cold_payload_spill_bridge", None)
        if bridge is not None:
            receipts = bridge.take_cold_payload_spill_receipts()
            if receipts:
                self._observation_db.commit_cold_payload_spill_receipts(receipts)
        self._poll_cold_payload_spill()
        failures = self._observation_db.take_cold_payload_spill_failures()
        if failures:
            raise RuntimeError(f"cold payload spill failed: {failures[0].error}")
        lock = getattr(self, "_admission_prefetch_lock", None)
        if lock is None:
            return
        with lock:
            failures = tuple(self._admission_prefetch_terminal_failures.items())
            self._admission_prefetch_terminal_failures.clear()
        for frame_id, _error in failures:
            self._observation_db.set_historical_eligibility(frame_id, eligible=False)
            self._retire_historical_identity(frame_id, reason="cache_backing_failure")
            self._admission_prefetch_terminal_failed_total += 1

    @staticmethod
    def _cache_owned_cpu_payload(payload: Any) -> np.ndarray:
        """Retain a CPU buffer without implicit device readback or source aliasing."""
        from eligsir.contracts import BufferOwnership, MemoryDomain

        if payload.spec.memory_domain not in {MemoryDomain.HOST, MemoryDomain.PINNED_HOST}:
            raise ValueError("prepared cold backing accepts only already-owned CPU arrays")
        if not isinstance(payload.data, np.ndarray):
            raise TypeError("prepared cold backing requires an ndarray payload")
        array = payload.data
        if payload.ownership is BufferOwnership.BORROWED:
            array = np.array(array, copy=True)
        elif payload.ownership not in {
            BufferOwnership.OWNED,
            BufferOwnership.SHARED_IMMUTABLE,
        }:
            raise ValueError("prepared cold backing requires owned or immutable CPU buffers")
        if array.flags.writeable:
            array.setflags(write=False)
        return array

    def retain_prepared_source(
        self,
        item: Any,
        *,
        lane: ReadySourceLane,
        promotion_permit: Any | None = None,
    ) -> bool:
        """Publish durable COLD ownership before a prepared READY-cache offer.

        READY is a bounded device-residency cache. A terminal COMMIT, or a
        DEFER supplied through a current promotion permit, first retains its
        CPU/reload route in the existing ``FrameObservationDB`` catalog. This
        method creates no camera, device payload, or page-in work.
        """

        from eligsir.mapping.observation_db import ColdPixelHandle
        from eligsir.runtime.prepared_ingress import PreparedIngressItem

        if not isinstance(item, PreparedIngressItem):
            raise TypeError("item must be PreparedIngressItem")
        if not isinstance(lane, ReadySourceLane):
            raise TypeError("lane must be ReadySourceLane")
        disposition = item.disposition
        token = item.control_token
        if disposition is None or token is None:
            raise ValueError("prepared durable retention requires terminal control")
        if token.source != item.packet.source or disposition.source != item.packet.source:
            raise ValueError("prepared durable retention source identity mismatch")
        action = disposition.semantic_action
        if action is SemanticAction.COMMIT:
            if lane is not ReadySourceLane.FRESH or not disposition.admit_supervision:
                raise ValueError("prepared COMMIT retention requires a fresh admitted source")
        elif action is SemanticAction.DEFER:
            if lane is not ReadySourceLane.REPLAY or promotion_permit is None:
                raise ValueError("prepared DEFER retention requires a replay promotion permit")
        else:
            raise ValueError("prepared durable retention requires COMMIT or DEFER")

        existing = self._observation_db.historical_candidate_for_source(
            item.packet.source, materializable_only=False
        )
        if existing is not None:
            handle = self._observation_db.get_cold_pixels(existing.frame_id)
            if handle is None or not handle.can_materialize:
                raise RuntimeError("prepared source lost its durable materialization route")
            return True

        tracking = item.observation
        pose = getattr(tracking, "pose", None)
        if pose is None:
            raise ValueError("prepared durable retention requires a tracked pose")
        packet = item.packet
        cache_owned = self._observation_db.cold_payload_disk_enabled
        reload_token = None if cache_owned else packet.reload_token
        rgb = None
        depth = None
        if cache_owned or reload_token is None:
            rgb = self._cache_owned_cpu_payload(packet.rgb)
            depth = None if packet.depth is None else self._cache_owned_cpu_payload(packet.depth)
        handle = ColdPixelHandle(
            source_identity=packet.source.fingerprint(),
            rgb=rgb,
            depth=depth,
            intrinsics=packet.intrinsics,
            reload_token=reload_token,
            rgb_encoding=str(packet.rgb.spec.encoding),
            depth_encoding=(
                None if packet.depth is None else str(packet.depth.spec.encoding)
            ),
            depth_units=(
                None if packet.depth is None else packet.depth.spec.units
            ),
        )
        record = self._observation_db.record_prepared_historical_candidate(
            source_identity=packet.source,
            pose_c2w=pose.matrix,
            timestamp=packet.source.timestamp_seconds,
            intrinsics=packet.intrinsics,
            cold_pixels=handle,
            modality_eligibility=disposition.modality_eligibility,
            disposition_snapshot_id=disposition.snapshot_id,
            semantic_action=action.value,
        )
        self._register_historical_record(record, packet.source)
        retained = self._observation_db.historical_candidate_for_source(
            packet.source, materializable_only=True
        )
        if retained is None:
            raise RuntimeError("prepared source durable retention did not publish a catalog route")
        return True

    def record_retained_prepared_commit(self, item: Any) -> None:
        """Keep one READY-cache-full COMMIT eligible for fresh page-in."""

        from eligsir.runtime.prepared_ingress import PreparedIngressItem

        if not isinstance(item, PreparedIngressItem):
            raise TypeError("item must be PreparedIngressItem")
        disposition = item.disposition
        token = item.control_token
        if (
            disposition is None
            or token is None
            or disposition.semantic_action is not SemanticAction.COMMIT
            or not disposition.admit_supervision
        ):
            raise ValueError("retained prepared source must be an admitted COMMIT")
        if token.source != item.packet.source or disposition.source != item.packet.source:
            raise ValueError("retained prepared COMMIT source identity mismatch")
        context = getattr(self, "_prepared_execution_contexts_by_source", {}).get(
            item.packet.source
        )
        if context is None:
            raise RuntimeError("prepared COMMIT requires frozen source execution context")
        value = (item.observation, disposition, token, context)
        self._observation_db.record_prepared_commit_obligation(
            item.packet.source, value
        )

    def clear_retained_prepared_commit(self, item: Any) -> None:
        """Clear only the special fresh-page-in marker after READY re-entry."""

        from eligsir.runtime.prepared_ingress import PreparedIngressItem

        if not isinstance(item, PreparedIngressItem):
            raise TypeError("item must be PreparedIngressItem")
        stored = self._observation_db.prepared_commit_obligation_for_source(
            item.packet.source
        )
        if stored is None:
            return
        _tracking, disposition, token, _context = stored
        if disposition != item.disposition or token != item.control_token:
            raise ValueError("retained prepared COMMIT context differs from re-entry item")
        self._observation_db.clear_prepared_commit_obligation(
            item.packet.source, expected=stored
        )

    def _register_historical_record(
        self, record: Any, source: Any, *, protected: bool = False
    ) -> None:
        """Publish one durable DB identity into bounded shortlist lanes."""

        from eligsir.mapping.scheduler.historical_candidate_index import (
            HistoricalFrameMetadata,
        )

        frame_id = int(record.frame_id)
        sources = getattr(self, "_historical_sources_by_frame_id", None)
        if sources is None:
            sources = self._historical_sources_by_frame_id = {}
        candidate_index = getattr(self, "_historical_candidate_index", None)
        if candidate_index is None:
            from eligsir.mapping.scheduler.historical_candidate_index import (
                HistoricalCandidateIndex,
                HistoricalCandidateConfig,
            )

            # Partially constructed runtimes (packet-only / deferred wiring tests)
            # reach this lazy path before the supervision batch size exists; they
            # keep the default lanes exactly as before this sizing was added.
            if hasattr(self, "_batch_size"):
                candidate_config = HistoricalCandidateConfig.for_batch_size(self._batch_size)
            else:
                candidate_config = HistoricalCandidateConfig()
            candidate_index = self._historical_candidate_index = (
                HistoricalCandidateIndex(config=candidate_config)
            )
        prior = sources.get(frame_id)
        if prior is not None:
            if prior != source:
                raise RuntimeError("historical frame id changed source identity")
            return
        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", None)
        if frame_ids_by_source is None:
            frame_ids_by_source = self._historical_frame_ids_by_source = {}
        prior_frame_id = frame_ids_by_source.get(source)
        if prior_frame_id is not None and prior_frame_id != frame_id:
            raise RuntimeError("historical source identity changed frame id")
        candidate_index.register(
            HistoricalFrameMetadata(
                frame_id=frame_id,
                region_id=int(record.submap_id),
                base_weight=1.0,
                protected=bool(protected),
            )
        )
        sources[frame_id] = source
        frame_ids_by_source[source] = frame_id
        # #1322: every newly-registered historical record starts life as a
        # protected HOT incumbent (design doc section 8.1) -- this is what
        # stops a mature historical pool from immediately displacing genuinely
        # new frontier evidence. Real, unconditional production caller.
        arena = getattr(self, "_hot_cold_challenger_arena", None)
        if arena is not None:
            arena.grant_lease(frame_id, int(self.iteration))

    def historical_usefulness_telemetry(self) -> "dict[str, int]":
        """Completed feedback rows, trusted index updates, and bounded deferrals."""
        owner = getattr(self, "_ready_source_owner", None)
        return {
            "observations_total": int(
                getattr(self, "_historical_usefulness_observations_total", 0)
            ),
            "trusted_updates_total": int(
                getattr(self, "_historical_usefulness_trusted_updates_total", 0)
            ),
            "feedback_busy": int(getattr(self, "_replay_feedback_busy", 0)),
            "replay_challenger_trials_completed": int(
                getattr(self, "_replay_challenger_completed_trials_total", 0)
            ),
            "replay_challenger_trials_authorized": int(getattr(self, "_replay_challenger_authorized_trials_total", 0)),
            "replay_challenger_demotions": int(
                getattr(self, "_replay_challenger_realized_demotions_total", 0)
            ),
            "replay_challenger_turnover_blocked": int(getattr(self, "_replay_challenger_turnover_blocked_total", 0)),
            "replay_challenger_no_ready": int(getattr(self, "_replay_challenger_no_ready_total", 0)),
            # This is the owner's ALL-replay-eviction byte count, not an
            # attribution claim for challenger decisions alone.
            "ready_replay_evicted_bytes": owner.counters.replay_evicted_bytes if owner is not None else 0,
        }

    def historical_usefulness_by_source_ordinal(self) -> "dict[int, float]":
        """Export the current train-only replay priors to historical repair.

        Missing ordinals/retired frames abstain; defaults-off exports nothing."""

        if not bool(
            getattr(
                self._optimization_policy,
                "streaming_historical_usefulness_tracking",
                False,
            )
        ):
            return {}
        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", None)
        candidate_index = getattr(self, "_historical_candidate_index", None)
        if not frame_ids_by_source or candidate_index is None:
            return {}
        out: dict[int, float] = {}
        for source, frame_id in frame_ids_by_source.items():
            ordinal = getattr(source, "source_ordinal", None)
            if ordinal is None:
                continue
            try:
                metadata = candidate_index.metadata_for(frame_id)
            except KeyError:
                continue
            out[int(ordinal)] = float(metadata.base_weight)
        return out

    def record_modality_usefulness_observation(
        self, source: Any, modality: str, raw_weight: float
    ) -> bool:
        """Update one modality EMA from measured train-only feedback.

        RGB, depth and insertion evidence remain separate. Unknown/retired
        sources abstain; the arena requires repeated measurements before trust."""

        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", None)
        if not frame_ids_by_source:
            return False
        frame_id = frame_ids_by_source.get(source)
        if frame_id is None:
            return False
        arena = getattr(self, "_hot_cold_challenger_arena", None)
        if arena is None:
            return False
        observation = arena.record_modality_observation(frame_id, modality, raw_weight)
        return bool(observation.trusted)

    def _record_replay_challenger_trace(self, event: Mapping[str, Any]) -> None:
        """Retain a bounded transition receipt only for the opt-in decision trace."""

        if not bool(getattr(self, "_population_selection_trace_enabled", False)):
            return
        if not isinstance(event, Mapping):
            raise TypeError("replay challenger trace events must be mappings")
        cap = max(1, int(getattr(self, "_batch_size", 1)) * 4)
        events = list(getattr(self, "_replay_challenger_trace_events", ()))
        if len(events) >= cap:
            events.pop(0)
            self._replay_challenger_trace_truncated = True
        events.append(dict(event))
        self._replay_challenger_trace_events = events

    @staticmethod
    def _replay_challenger_source_trace(source: Any) -> dict[str, Any]:
        """Serialize only stable source identity fields for the audit receipt."""

        if not isinstance(source, SourceIdentity):
            return {"source": None}
        return {
            "source": source.fingerprint(),
            "source_ordinal": int(source.source_ordinal),
            "timestamp_ns": int(source.timestamp_ns),
        }

    @staticmethod
    def _replay_challenger_snapshot_trace(token: Any) -> dict[str, int] | None:
        """Expose available evidence versions without inventing unavailable ones."""

        if token is None:
            return None
        names = (
            "map_content",
            "topology",
            "coordinate_frame",
            "schema",
            "optimizer_update",
            "surface_evidence",
            "policy",
            "parameter",
        )
        values = {
            name: int(getattr(token, name))
            for name in names
            if getattr(token, name, None) is not None
        }
        return values or None

    def _record_replay_challenger_completion(
        self,
        source: SourceIdentity,
        *,
        completion: str,
        reason: str,
        modality: str | None = None,
        lease_generation: int | None = None,
        resolution_tier: str | None = None,
        snapshot: Any = None,
        **scores: Any,
    ) -> None:
        event = {
            "stage": "completion",
            **self._replay_challenger_source_trace(source),
            "completion": completion,
            "reason": reason,
            "lease_generation": lease_generation,
            "resolution_tier": resolution_tier,
            "snapshot": (
                snapshot
                if isinstance(snapshot, dict)
                else self._replay_challenger_snapshot_trace(snapshot)
            ),
        }
        if modality is not None:
            event["modality"] = modality
        event.update(scores)
        self._record_replay_challenger_trace(event)

    def evaluate_hot_cold_challenge(
        self,
        incumbent_source: Any,
        challenger_source: Any,
        modality: str,
        *,
        audit: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate trusted modality evidence and update the winner's prior.

        The arena owns hysteresis and freshness leases. This decision alone is
        not a physical promotion: the caller must complete protected READY
        turnover, and reports realized eviction separately from arena decisions.
        """

        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", None)
        if not frame_ids_by_source:
            if audit is not None:
                audit.update({"arena_reason": "source_index_unavailable", "arena_outcome": "retain"})
            return False
        incumbent_id = frame_ids_by_source.get(incumbent_source)
        challenger_id = frame_ids_by_source.get(challenger_source)
        if incumbent_id is None or challenger_id is None:
            if audit is not None:
                audit.update({"arena_reason": "source_identity_unavailable", "arena_outcome": "retain"})
            return False
        arena = getattr(self, "_hot_cold_challenger_arena", None)
        candidate_index = getattr(self, "_historical_candidate_index", None)
        if arena is None or candidate_index is None:
            if audit is not None:
                audit.update({"arena_reason": "arena_unavailable", "arena_outcome": "retain"})
            return False
        decision = arena.evaluate_challenge(
            incumbent_id, challenger_id, modality, int(self.iteration)
        )
        if audit is not None:
            audit.update(
                {
                    "arena_reason": decision.reason,
                    "arena_outcome": "promote" if decision.promote else "retain",
                    "incumbent_weight": float(decision.incumbent_weight),
                    "challenger_weight": float(decision.challenger_weight),
                }
            )
        if not decision.promote:
            return False
        updated = candidate_index.update_base_weight(
            challenger_id, decision.challenger_weight
        )
        if audit is not None and not updated:
            audit.update(
                {"arena_reason": "candidate_update_rejected", "arena_outcome": "retain"}
            )
        return updated

    def _actuate_replay_challenger_turnover(
        self,
        trusted_sources: tuple[SourceIdentity, ...],
        *,
        observation_audit: Mapping[tuple[SourceIdentity, str], Mapping[str, Any]] | None = None,
        evidence_snapshot: Any = None,
    ) -> tuple[SourceIdentity, ...]:
        """Turn trusted COLD evidence into bounded physical READY turnover."""

        policy = getattr(self, "_optimization_policy", None)
        fraction = float(getattr(policy, "streaming_replay_challenger_fraction", 0.0))
        cadence = int(getattr(policy, "streaming_replay_challenger_demotion_cadence", 0))
        limit = int(getattr(policy, "streaming_replay_challenger_demotion_limit", 0))
        owner = getattr(self, "_ready_source_owner", None)
        phase_owner = getattr(self, "_post_refinement_phase_owner", None)
        blocked_reason = None
        if fraction <= 0.0:
            blocked_reason = "feature_disabled"
        elif cadence <= 0:
            blocked_reason = "cadence_disabled"
        elif limit <= 0:
            blocked_reason = "demotion_limit_disabled"
        elif owner is None:
            blocked_reason = "ready_owner_unavailable"
        elif bool(getattr(self, "_closed", False)):
            blocked_reason = "runtime_closed"
        elif phase_owner is not None and phase_owner.phase is PostRefinementPhase.REFINEMENT:
            blocked_reason = "refinement_phase"
        elif int(getattr(self, "optimizer_steps", 0)) % cadence:
            blocked_reason = "cadence_not_due"
        if blocked_reason is not None:
            if bool(getattr(self, "_population_selection_trace_enabled", False)):
                self._record_replay_challenger_trace(
                    {
                        "stage": "turnover",
                        "outcome": "not_attempted",
                        "reason": blocked_reason,
                        "sources": [
                            source.fingerprint()
                            for source in trusted_sources
                            if isinstance(source, SourceIdentity)
                        ],
                    }
                )
            return ()
        if not isinstance(trusted_sources, tuple) or not all(
            isinstance(source, SourceIdentity) for source in trusted_sources
        ):
            raise TypeError("trusted_sources must be a tuple of SourceIdentity")
        residents = tuple(owner.ready_replay_sources())
        if len(residents) <= max(1, int(owner.replay_reserve_target)):
            self._replay_challenger_no_ready_total = int(getattr(self, "_replay_challenger_no_ready_total", 0)) + 1
            self._record_replay_challenger_trace(
                {
                    "stage": "turnover",
                    "outcome": "not_attempted",
                    "reason": "ready_reserve_only",
                    "sources": [
                        source.fingerprint()
                        for source in trusted_sources
                        if isinstance(source, SourceIdentity)
                    ],
                }
            )
            return ()
        protected_work = (
            getattr(self, "_prepared_ready_frozen_work", None),
            getattr(self, "_prepared_ready_candidate_work", None),
        )
        frozen_sources = frozenset(
            source
            for work in protected_work
            if work is not None
            for source in (*work[1].supervision_sources, *work[1].replay_sources)
        )
        snapshot = getattr(self, "_latest_control_snapshot", None)
        global_source = getattr(snapshot, "reserved_global_replay_candidate", None)
        protected_base = frozen_sources | frozenset(trusted_sources) | (
            {global_source} if isinstance(global_source, SourceIdentity) else set()
        )
        # COMMIT debt survives residency changes; a replay-lane label alone is
        # not proof that a fresh source has received its first service.
        database = self._observation_db
        protected_base |= {
            source for source in residents
            if database.prepared_commit_obligation_for_source(source) is not None
        }
        ready_generations = (
            {
                lease.source_identity: int(lease.generation)
                for lease in getattr(owner, "pending_leases", ())
            }
            if bool(getattr(self, "_population_selection_trace_enabled", False))
            else {}
        )
        evicted: list[SourceIdentity] = []
        arena = self._hot_cold_challenger_arena
        for challenger in dict.fromkeys(trusted_sources):
            if len(evicted) >= limit or len(owner.ready_replay_sources()) <= max(
                1, int(owner.replay_reserve_target)
            ):
                break
            challenger_id = self._historical_frame_ids_by_source.get(challenger)
            if challenger_id is None:
                continue
            if challenger not in frozenset(owner.replay_resident_sources()):
                continue
            modalities = tuple(
                modality
                for modality in ("rgb", "depth")
                if arena.modality_trusted(challenger_id, modality)
                and arena.modality_weight(challenger_id, modality) > 0
            )
            for incumbent in residents:
                if incumbent == challenger or incumbent in protected_base or incumbent in evicted:
                    continue
                incumbent_id = self._historical_frame_ids_by_source.get(incumbent)
                record = database.get_frame(incumbent_id) if incumbent_id is not None else None
                if not self._prepared_cold_pagein_capable(record):
                    continue
                # A stronger RGB repair signal must not displace independently
                # useful depth (or vice versa). Unknown challenger support abstains.
                if any(
                    arena.modality_trusted(incumbent_id, modality)
                    and (modality not in modalities or arena.modality_weight(challenger_id, modality)
                         < arena.modality_weight(incumbent_id, modality))
                    for modality in ("rgb", "depth")
                ):
                    continue
                for modality in modalities:
                    decision_audit: dict[str, Any] = {}
                    promoted = self.evaluate_hot_cold_challenge(
                        incumbent,
                        challenger,
                        modality,
                        audit=decision_audit,
                    )
                    observation = None if observation_audit is None else observation_audit.get(
                        (challenger, modality)
                    )
                    event = {
                        "stage": "arena",
                        **self._replay_challenger_source_trace(challenger),
                        "incumbent": incumbent.fingerprint(),
                        "modality": modality,
                        "lease_generation": ready_generations.get(challenger),
                        "incumbent_lease_generation": ready_generations.get(incumbent),
                        "snapshot": self._replay_challenger_snapshot_trace(evidence_snapshot),
                        "versions": None,
                        "residency_outcome": "retained",
                    }
                    version_reader = getattr(self, "_versions", None)
                    if bool(getattr(self, "_population_selection_trace_enabled", False)) and callable(version_reader):
                        try:
                            content, topology, parameter = version_reader()
                            event["versions"] = {
                                "map_content": int(content),
                                "topology": int(topology),
                                "parameter": int(parameter),
                            }
                        except (TypeError, ValueError, RuntimeError):
                            pass
                    if observation is not None:
                        event.update(dict(observation))
                    event.update(decision_audit)
                    if not promoted:
                        self._record_replay_challenger_trace(event)
                        continue
                    removed = owner.evict_replay(
                        target_source=incumbent, protected_sources=frozenset(protected_base),
                    )
                    if removed and removed[0].source_identity == incumbent:
                        event["residency_outcome"] = "demote"
                        evicted.append(incumbent)
                        self._replay_challenger_realized_demotions_total = int(
                            getattr(self, "_replay_challenger_realized_demotions_total", 0)
                        ) + 1
                    else:
                        event["residency_outcome"] = "blocked"
                    self._record_replay_challenger_trace(event)
                    break
                if incumbent in evicted:
                    break
        if not evicted:
            self._replay_challenger_turnover_blocked_total = int(getattr(self, "_replay_challenger_turnover_blocked_total", 0)) + 1
        return tuple(evicted)

    def _retire_historical_identity(
        self, frame_id: int, *, reason: str = "unmaterializable"
    ) -> bool:
        """Withdraw one identity from every scheduler lane.

        The FrameObservationDB is authoritative for historical eligibility, but
        the runtime keeps its own bounded replay lanes keyed by source identity.
        Leaving a dead identity in these lanes lets the work plan select a
        supervision identity for which no camera can ever be built, which used
        to abort the whole mapping worker (issue #1289). Retire it everywhere,
        in the same step, so the planner simply stops proposing it -- a
        graceful skip alone would leave the planner reproposing it forever.

        ``reason`` distinguishes accounting-distinct causes (issue #1405):

        - ``"unmaterializable"`` (default): a live (non-reloadable) source was
          evicted from the bounded cold payload store with no route back to
          real pixels -- the original #1289 case. Counted in
          ``_retired_unmaterializable_total`` with its one-time operator
          warning.
        - ``"rgb_supervision_ineligible"``: the identity is fully
          materializable but its frozen ``ModalityEligibility`` can never
          satisfy the RGB-supervision promotion query (issue #1302). Its
          pixels are not gone, so it must not increment the unmaterializable
          counter or emit that warning; it is counted separately in
          ``_retired_rgb_ineligible_total`` instead.
        - ``"cache_backing_failure"``: a local source-independent cache token
          failed validation or page-in.  This is terminal for that identity;
          leaving it pending would repeatedly schedule a known-corrupt token.

        Either reason still withdraws the identity from every lane below,
        including depth-only replay/candidate eligibility: this method does
        not yet preserve a retired ``rgb_supervision_ineligible`` identity's
        depth/depth-insertion eligibility for future modality-aware
        scheduling (issue #1405's item 3, and #1365/#1399's direction) --
        only its accounting is reason-aware today.
        """

        frame_id = int(frame_id)
        sources = getattr(self, "_historical_sources_by_frame_id", None) or {}
        source = sources.pop(frame_id, None)
        candidate_index = getattr(self, "_historical_candidate_index", None)
        if candidate_index is not None:
            candidate_index.retire(frame_id)
        arena = getattr(self, "_hot_cold_challenger_arena", None)
        if arena is not None:
            arena.revoke_lease(frame_id)
            arena.forget_modality(frame_id)
        if source is None:
            return False
        self._observation_db.clear_prepared_commit_obligation(source)
        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", None)
        if frame_ids_by_source is not None:
            frame_ids_by_source.pop(source, None)
        # `shortlist()` rejects focus weights naming an unregistered frame, so
        # the retained focus tuple has to drop the retired identity too.
        focus_weights = getattr(self, "_replay_focus_frame_weights", ())
        if any(int(item[0]) == frame_id for item in focus_weights):
            self._replay_focus_frame_weights = tuple(
                item for item in focus_weights if int(item[0]) != frame_id
            )
        focus_sources = getattr(self, "_replay_focus_sources", ())
        if source in focus_sources:
            self._replay_focus_sources = tuple(
                item for item in focus_sources if item != source
            )
        for lane_name in (
            "_replay_modalities",
            "_replay_weights",
            "_pending_supervision_modalities",
        ):
            lane = getattr(self, lane_name, None)
            if lane is not None:
                lane.pop(source, None)
        # `_idle_replay_plan`'s last-resort fallback re-mints work from the
        # previous plan's sources; without this bounded tombstone set it would
        # resurrect an identity that was just retired.
        retired = getattr(self, "_retired_replay_sources", None)
        if retired is None:
            retired = self._retired_replay_sources = OrderedDict()
        retired[source] = True
        while len(retired) > 1024:
            retired.popitem(last=False)
        self._pending_replay_trial_sources = tuple(
            pending
            for pending in getattr(self, "_pending_replay_trial_sources", ())
            if pending != source
        )
        self._invalidate_prepared_ready_plan_for_source(source)
        if reason == "rgb_supervision_ineligible":
            self._retired_rgb_ineligible_total = int(
                getattr(self, "_retired_rgb_ineligible_total", 0)
            ) + 1
        elif reason == "cache_backing_failure":
            self._retired_cache_backing_failure_total = int(
                getattr(self, "_retired_cache_backing_failure_total", 0)
            ) + 1
        else:
            self._retired_unmaterializable_total = int(
                getattr(self, "_retired_unmaterializable_total", 0)
            ) + 1
            if not getattr(self, "_retired_unmaterializable_warned", False):
                self._retired_unmaterializable_warned = True
                print(
                    "[observation_db] WARNING: a deferred live observation was evicted "
                    "from the bounded cold payload store with no reload token; its "
                    "pixels are gone, so it is now permanently ineligible for replay. "
                    "Raise --streaming_live_cold_payload_max_frames / "
                    "--streaming_live_cold_payload_max_bytes to retain more history.",
                    flush=True,
                )
        return True

    def _shortlisted_replay_modalities(
        self, *, snapshot_id: int, control_tick_epoch: int
    ) -> OrderedDict[Any, Any]:
        """Return bounded sampler inputs while retaining the full identity DB."""

        modalities, _reserved_source = self._shortlisted_replay_view(
            snapshot_id=snapshot_id,
            control_tick_epoch=control_tick_epoch,
        )
        return modalities

    def _shortlisted_replay_view(
        self,
        *,
        snapshot_id: int,
        control_tick_epoch: int,
        reserved_global_epoch: int | None = None,
    ) -> tuple[OrderedDict[Any, Any], Any | None]:
        """Return one bounded identity view plus its global reservation."""

        replay_modalities = getattr(self, "_replay_modalities", OrderedDict())
        observation_db = getattr(self, "_observation_db", None)
        candidate_index = getattr(self, "_historical_candidate_index", None)
        if candidate_index is None or candidate_index.history_size == 0:
            if observation_db is not None:
                replay_modalities = OrderedDict(
                    (source, modality)
                    for source, modality in replay_modalities.items()
                    if observation_db.prepared_commit_obligation_for_source(source) is None
                )
            return OrderedDict(
                replay_modalities
                if not hasattr(self, "initialized") or self.initialized
                else getattr(self, "_pending_supervision_modalities", OrderedDict())
            ), None

        def replay_eligible(source: Any) -> bool:
            return source in replay_modalities and (
                observation_db is None
                or observation_db.prepared_commit_obligation_for_source(source) is None
            )

        from eligsir.utils.partition_control import _deterministic_probe_indices

        owner = getattr(self, "_work_budget_owner", None)
        optimizer_steps = int(getattr(self, "optimizer_steps", 0))
        if reserved_global_epoch is None:
            reservation_epoch = (
                owner.global_history_reservation_epoch(optimizer_steps)
                if owner is not None
                else optimizer_steps // 8
            )
        else:
            reservation_epoch = int(reserved_global_epoch)
        shortlist = candidate_index.shortlist(
            snapshot_id=int(snapshot_id),
            epoch=int(control_tick_epoch),
            focus_identity_weights=getattr(
                self, "_replay_focus_frame_weights", ()
            ),
            reserved_global_epoch=reservation_epoch,
        )
        sources = getattr(self, "_historical_sources_by_frame_id", {})
        modalities = OrderedDict(
            (source, replay_modalities[source])
            for candidate in shortlist
            if replay_eligible(source := sources.get(candidate.frame_id))
        )
        reserved_source = next(
            (
                sources.get(candidate.frame_id)
                for candidate in shortlist
                if "reserved_global" in candidate.evidence.lanes
                and sources.get(candidate.frame_id) in modalities
            ),
            None,
        )

        # #1723 keeps historical priority, then optionally unions a bounded,
        # deterministic active sample after COMMIT eligibility filtering.
        # With the flag off, replay remains historical-only.
        if getattr(self, "_partition_focus_actuation_reaches_active_sources", False):
            MAX_SHORTLIST = 128
            active_sources = tuple(
                source
                for camera in getattr(self.scene, "train_cameras", ())
                if (source := getattr(camera, "_source_identity", None)) is not None
                and replay_eligible(source)
            )
            # active_budget <= MAX_SHORTLIST - len(modalities) always holds,
            # so the union below can never push len(modalities) past
            # MAX_SHORTLIST -- no post-hoc trim is needed or reachable.
            active_budget = max(0, min(MAX_SHORTLIST // 2, MAX_SHORTLIST - len(modalities)))
            if active_sources and active_budget > 0:
                ordered_active = tuple(sorted(active_sources, key=lambda s: s.source_ordinal))
                picked_positions = _deterministic_probe_indices(
                    tuple(range(len(ordered_active))), active_budget
                )
                for position in picked_positions:
                    source = ordered_active[position]
                    if source not in modalities:
                        modalities[source] = replay_modalities[source]

        return modalities, reserved_source

    def _representative_identity(self, snapshot: ControlSnapshot, ordinal: int):
        representative = self._source_identities_by_ordinal.get(int(ordinal))
        if representative is None:
            raise ValueError(f"unknown backpressure representative ordinal {ordinal}")
        source = snapshot.source
        if (
            source is None
            or representative.source_id != source.source_id
            or representative.stream_epoch != source.stream_epoch
        ):
            raise ValueError("backpressure representative crossed a source epoch")
        return representative

    def _versions(self) -> tuple[int, int, int]:
        content = int(
            getattr(self.backend, "map_content_version", getattr(self.backend, "map_version", 0))
        )
        topology = int(getattr(self.backend, "topology_version", 0))
        optimization_session = getattr(getattr(self, "_engine", None), "session", None)
        parameters = int(getattr(optimization_session, "parameter_version", 0))
        return content, topology, parameters

    def _supervision_work_clocks_or_create(self) -> Any:
        """Lazy-create SupervisionWorkClocks if not initialized.

        Resilient to runtimes created via object.__new__() (test seams) that
        never run __init__. Issue #703.
        """
        clocks = getattr(self, "_supervision_work_clocks", None)
        if clocks is None:
            from eligsir.runtime.supervision_work_clocks import SupervisionWorkClocks
            clocks = SupervisionWorkClocks()
            self._supervision_work_clocks = clocks
        return clocks

    def _stamp_completed_result_map_version_after(self, result: Any) -> None:
        """Bind a completed fixed-topology result to Adam's map-content commit.

        The engine receives ``map_version`` before the optimizer session steps,
        while the canonical backend advances ``map_content_version`` inside that
        step.  Capture the authoritative post-Adam value immediately after the
        transaction returns, before deferred shadow publication can compare it
        against the current version vector.  The pre-render/E0 provenance and
        ``map_version_before`` remain untouched.
        """

        before = getattr(result, "map_version_before", None)
        after = getattr(result, "map_version_after", None)
        if type(before) is not int or type(after) is not int:
            raise TypeError("completed optimization result must publish exact map versions")
        committed = self._population_shadow_exact_version(
            getattr(self.backend, "map_content_version", None)
        )
        if committed is None:
            raise RuntimeError(
                "canonical backend must publish map_content_version after optimization"
            )
        result.map_version_after = committed

    @staticmethod
    def _population_shadow_exact_version(value: Any) -> int | None:
        """Accept only an explicitly published non-negative version."""

        return value if type(value) is int and value >= 0 else None

    def _population_shadow_evidence_versions(self):
        """Read the exact pre-render versions needed to stamp E0.

        Missing ownership is not replaced by a convenient zero: the true batch
        executor simply omits evidence and the shadow bridge sees no event.
        """

        from eligsir.mapping.population_shadow_runtime import PopulationShadowVersions

        backend = getattr(self, "backend", None)
        session = getattr(getattr(self, "_engine", None), "session", None)
        values = (
            self._population_shadow_exact_version(
                getattr(backend, "map_content_version", getattr(backend, "map_version", None))
            ),
            self._population_shadow_exact_version(getattr(backend, "topology_version", None)),
            self._population_shadow_exact_version(
                getattr(backend, "coordinate_frame_version", None)
            ),
            self._population_shadow_exact_version(getattr(backend, "schema_version", None)),
            # The context applies the exact planned update id to this field;
            # use the current work-clock here only to prove the owner exists.
            self._population_shadow_exact_version(
                getattr(getattr(self, "_work_clock_owner", None).snapshot(), "optimizer_updates", None)
                if getattr(self, "_work_clock_owner", None) is not None
                else None
            ),
            self._population_shadow_exact_version(
                getattr(self, "_population_shadow_surface_evidence_version", None)
            ),
            self._population_shadow_exact_version(
                getattr(self, "_population_shadow_policy_version", None)
            ),
            self._population_shadow_exact_version(getattr(session, "parameter_version", None)),
        )
        if any(value is None for value in values):
            return None
        return PopulationShadowVersions(
            map_content=values[0],
            topology=values[1],
            coordinate_frame=values[2],
            schema=values[3],
            optimizer_update=values[4],
            surface_evidence=values[5],
            policy=values[6],
            parameter=values[7],
        )

    def _population_shadow_work_token(self, _result: Any, context: Any):
        """Project only scheduler authorization into a neutral work token."""

        from eligsir.mapping.population_policy import MutationWorkToken

        plan = getattr(self, "_population_shadow_active_plan", None)
        budget = getattr(getattr(plan, "mutation_budget", None), "max_total", None)
        epoch = getattr(context, "optimizer_update_id", None)
        current = self._population_shadow_evidence_versions()
        if (
            type(budget) is not int
            or budget < 0
            or type(epoch) is not int
            or epoch < 0
            or current is None
            or current.optimizer_update != epoch
        ):
            return None
        # The compatibility/shadow path retains its pre-existing plan-budget
        # projection.  Once live policy owns structural action selection, the
        # cadence owner authorizes only neutral work units instead of exposing
        # legacy grow/relocate lanes to M5.
        if not self._population_live_requested():
            return MutationWorkToken(epoch=epoch, units=budget)
        depth_valid_samples = getattr(_result, "depth_valid_sample_count", None)
        geometry_opportunity = (
            type(depth_valid_samples) is int and depth_valid_samples > 0
        )
        from eligsir.optimization.gaussian.supervision_evidence import (
            SupervisionEvidenceBatch,
        )

        completed_b4_c4 = bool(
            type(getattr(_result, "realized_batch_size", None)) is int
            and _result.realized_batch_size == 4
            and type(getattr(_result, "realized_camera_count", None)) is int
            and _result.realized_camera_count == 4
            and type(getattr(_result, "supervision_view_count", None)) is int
            and _result.supervision_view_count == 4
            and type(getattr(_result, "optimizer_step_count", None)) is int
            and _result.optimizer_step_count == 1
            and isinstance(
                getattr(_result, "evidence", None), SupervisionEvidenceBatch
            )
        )
        if not completed_b4_c4:
            return MutationWorkToken(
                epoch=epoch,
                units=0,
                geometry_opportunity=geometry_opportunity,
            )
        cadence_owner = getattr(self, "_mcmc_cadence_owner", None)
        if cadence_owner is None:
            # Live policy must observe an explicit scheduler abstention, not
            # turn an unavailable owner into a silent missing-token drop.
            return MutationWorkToken(
                epoch=epoch,
                units=0,
                geometry_opportunity=geometry_opportunity,
            )
        return cadence_owner.authorize_population_mutation(
            self._work_clock_owner.snapshot(),
            topology_version=current.topology,
            max_actions=budget,
            geometry_opportunity=geometry_opportunity,
        )

    @staticmethod
    def _population_shadow_belief_from_rows(gaussians: Any):
        """Return a strict, zero-copy belief view over registered row state.

        This production path intentionally does *not* call
        ``GaussianBelief.from_gaussians`` because that compatibility accessor
        calls ``ensure_belief_buffers``.  A shadow observer must never repair,
        allocate, resize, or otherwise mutate a live model merely to observe
        it.  Missing or stale rows therefore return ``None`` and the bridge
        publishes its existing ``MISSING_BELIEF`` drop reason.
        """

        from eligsir.map.row_state import GaussianRowStateRegistry
        import torch as torch_module
        from eligsir.utils.gaussian_belief import (
            ExistenceBelief,
            GaussianBelief,
            GeometryBelief,
            ObservabilityBelief,
            ProvenanceBelief,
            RepresentationBelief,
            TemporalBelief,
        )

        registry_builder = getattr(gaussians, "build_row_state_registry", None)
        if not callable(registry_builder):
            return None
        try:
            registry = registry_builder()
        except Exception:
            return None
        if not isinstance(registry, GaussianRowStateRegistry):
            return None
        xyz = getattr(gaussians, "get_xyz", None)
        if not isinstance(xyz, torch_module.Tensor) or xyz.ndim != 2:
            return None
        n, device = xyz.shape[0], xyz.device
        fields = {
            "gaussian_uid": ("provenance.gaussian_uid", ()),
            "birth_iter": ("provenance.birth_iter", ()),
            "birth_method": ("provenance.birth_method", ()),
            "parent_uid": ("provenance.parent_uid", ()),
            "root_uid": ("provenance.root_uid", ()),
            "generation": ("provenance.generation", ()),
            "birth_anchor_xyz": ("belief.birth_anchor_xyz", (3,)),
            "population_policy_support_count": (
                "belief.population_policy_support_count", ()
            ),
            "population_policy_exposure_count": (
                "belief.population_policy_exposure_count", ()
            ),
            "baseline_diversity": ("belief.baseline_diversity", ()),
            "last_exposed_update": ("belief.last_exposed_update", ()),
            "support_evidence": ("belief.support_evidence", ()),
            "contradiction_evidence": ("belief.contradiction_evidence", ()),
            "p_exist": ("belief.p_exist", ()),
            "surface_anchor_id": ("belief.surface_anchor_id", ()),
            "anchor_confidence": ("surface.anchor_confidence", ()),
            "surface_position_uncertainty": (
                "belief.surface_position_uncertainty", ()
            ),
            "surface_normal_uncertainty": (
                "belief.surface_normal_uncertainty", ()
            ),
            "visibility_ema": ("render.visibility_ema", ()),
            "rgb_residual_ema": ("belief.rgb_residual_ema", ()),
            "depth_residual_ema": ("belief.depth_residual_ema", ()),
            "utility_ema": ("provenance.utility_ema", ()),
            "redundancy": ("belief.redundancy", ()),
            "expected_marginal_gain": ("belief.expected_marginal_gain", ()),
            "evidence_recency": ("belief.evidence_recency", ()),
            "change_probability": ("belief.change_probability", ()),
            "confidence_decay": ("belief.confidence_decay", ()),
        }
        if not {registered for registered, _suffix in fields.values()} <= registry.names:
            return None
        descriptors = {descriptor.name: descriptor for descriptor in registry.descriptors}
        values: dict[str, torch_module.Tensor] = {}
        for name, (registered, suffix) in fields.items():
            value = getattr(gaussians, name, None)
            if name == "visibility_ema":
                actual_suffix = (
                    tuple(value.shape[1:])
                    if isinstance(value, torch_module.Tensor) and value.ndim >= 1
                    else None
                )
                descriptor = descriptors.get(registered)
                if (
                    not isinstance(value, torch_module.Tensor)
                    or value.ndim not in (1, 2)
                    or value.shape[0] != n
                    or value.device != device
                    or actual_suffix not in ((), (1,))
                    or descriptor is None
                    or descriptor.shape_suffix != actual_suffix
                    or descriptor.dtype != value.dtype
                ):
                    return None
                values[name] = value if actual_suffix == () else value.squeeze(1)
                continue
            if (
                not isinstance(value, torch_module.Tensor)
                or value.ndim != 1 + len(suffix)
                or value.shape[0] != n
                or tuple(value.shape[1:]) != suffix
                or value.device != device
            ):
                return None
            values[name] = value
        # Exposure metadata is canonical when allocated, but remains optional
        # while the old compatibility feature flags are still active.  Its
        # absence means unknown measurement sufficiency, not zero evidence;
        # a present-but-stale/unregistered row fails closed like every other
        # registered population field.
        optional_fields = {
            "supervised_tier_high_water": (
                "compatibility.supervised_tier_high_water", ()
            ),
            "last_supervised_iter": ("compatibility.last_supervised_iter", ()),
        }
        optional_values: dict[str, torch_module.Tensor | None] = {}
        for name, (registered, suffix) in optional_fields.items():
            value = getattr(gaussians, name, None)
            registered_present = registered in registry.names
            if not registered_present:
                if value is not None:
                    return None
                optional_values[name] = None
                continue
            if (
                not isinstance(value, torch_module.Tensor)
                or value.ndim != 1 + len(suffix)
                or value.shape[0] != n
                or tuple(value.shape[1:]) != suffix
                or value.device != device
            ):
                return None
            optional_values[name] = value
        return GaussianBelief(
            provenance=ProvenanceBelief(
                gaussian_uid=values["gaussian_uid"],
                birth_update=values["birth_iter"],
                birth_method=values["birth_method"],
                parent_uid=values["parent_uid"],
                root_uid=values["root_uid"],
                generation=values["generation"],
                birth_anchor_xyz=values["birth_anchor_xyz"],
            ),
            observability=ObservabilityBelief(
                meaningful_exposure_count=values["population_policy_exposure_count"],
                independent_support_views=values["population_policy_support_count"],
                baseline_diversity=values["baseline_diversity"],
                highest_supervised_tier=optional_values["supervised_tier_high_water"],
                last_exposed_update=values["last_exposed_update"],
                last_supported_update=optional_values["last_supervised_iter"],
            ),
            existence=ExistenceBelief(
                support_evidence=values["support_evidence"],
                contradiction_evidence=values["contradiction_evidence"],
                p_exist=values["p_exist"],
            ),
            geometry=GeometryBelief(
                surface_anchor_id=values["surface_anchor_id"],
                p_surface_consistent=values["anchor_confidence"],
                surface_position_uncertainty=values["surface_position_uncertainty"],
                surface_normal_uncertainty=values["surface_normal_uncertainty"],
            ),
            representation=RepresentationBelief(
                render_contribution=values["visibility_ema"],
                rgb_residual_ema=values["rgb_residual_ema"],
                depth_residual_ema=values["depth_residual_ema"],
                gradient_utility_evidence=values["utility_ema"],
                redundancy=values["redundancy"],
                expected_marginal_gain=values["expected_marginal_gain"],
            ),
            temporal=TemporalBelief(
                evidence_recency=values["evidence_recency"],
                change_probability=values["change_probability"],
                confidence_decay=values["confidence_decay"],
            ),
            n=n,
        )

    def _population_shadow_current_versions(self, _result: Any, context: Any):
        """Read the post-Adam snapshot and fail closed on a clock mismatch."""

        versions = self._population_shadow_evidence_versions()
        epoch = getattr(context, "optimizer_update_id", None)
        if versions is None or type(epoch) is not int or versions.optimizer_update != epoch:
            return None
        return versions

    def _population_shadow_runtime_kwargs(self) -> dict[str, Any]:
        """Build disabled-by-default, observational-only runtime adapters."""

        enabled = self.config.static_options.get("population_shadow_enabled", False)
        if type(enabled) is not bool:
            raise TypeError("population_shadow_enabled must be an exact bool")
        return {
            "population_shadow_enabled": enabled,
            "population_shadow_work_token_provider": self._population_shadow_work_token,
            "population_shadow_versions_provider": self._population_shadow_current_versions,
            "population_shadow_belief_provider": self._population_shadow_belief_from_rows,
            "population_shadow_evidence_versions_provider": self._population_shadow_evidence_versions,
            "population_shadow_capacity": self._runtime_policy.cap_max,
            "population_shadow_defer_until_post_step_commit": enabled,
        }

    def _observe_population_shadow_after_step(
        self, runtime: Any, result: Any, *, native_targets: Any = None
    ):
        """Observe after the canonical work-clock commit, before any mutation."""

        if runtime is None or not bool(getattr(runtime, "population_shadow_enabled", False)):
            return None
        return runtime.observe_population_shadow_after_post_step_commit(
            result, native_targets=native_targets
        )

    def set_population_live_adapters(
        self,
        *,
        target_binding_provider: Any,
        context_provider: Any,
    ) -> None:
        """Install the two authoritative, bounded M5/M6 composition adapters.

        ``target_binding_provider`` owns device-side pairing of the policy's
        bounded structural envelope to an exact ``SurfaceTargetBatch``.
        ``context_provider`` owns the immutable row registry/identity
        inventory and proposal key.  Keeping both explicit prevents the
        production loop from silently reconstructing stale sidecars or doing
        an unbounded UID readback merely to activate a feature flag.
        """

        if not callable(target_binding_provider):
            raise TypeError("target_binding_provider must be callable")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")
        self._population_live_target_binding_provider = target_binding_provider
        self._population_live_context_provider = context_provider
        # A production surface-target provider is still the exact callable
        # consumed by live activation.  Retain its observation side explicitly
        # so the transaction can give it the same E0/render/source borrows
        # before the later policy fence asks it to bind target IDs.
        observe = getattr(target_binding_provider, "observe_evidence", None)
        self._population_live_surface_target_provider = (
            target_binding_provider if callable(observe) else None
        )

    def _observe_population_surface_targets_after_step(
        self,
        result: Any,
        source_leases: Any,
    ):
        """Pass the completed B1/B4 E0 object to the installed M6 provider.

        The method has no fallback construction: a missing evidence object,
        malformed provider, or unavailable source lease is delegated to the
        provider's fail-closed cache clear.  In particular, this boundary
        never asks the renderer to reproduce depth or asks the transfer owner
        to upload source pixels.
        """

        provider = getattr(self, "_population_live_surface_target_provider", None)
        observe = getattr(provider, "observe_evidence", None)
        if not callable(observe):
            return None
        return observe(
            evidence=getattr(result, "evidence", None),
            source_leases=source_leases,
        )

    def _observe_population_policy_inputs_after_step(
        self,
        runtime: Any,
        result: Any,
        source_leases: Any,
        native_geometry_actual_plan: MappingWorkPlan | None = None,
    ) -> tuple[Any, Any]:
        """Observe one completed step through the normal live-policy inputs.

        The deferred shadow bridge consumes the post-work-clock belief and E0
        snapshot first.  Live mode then gives the installed target provider
        the exact same completed E0 object plus already-resident source
        borrows.  Missing or invalid evidence remains fail-closed in either
        owner; this helper does not manufacture policy evidence or a target.
        """

        # M5 shadow is observational.  Updating ``utility_ema`` there would
        # change a legacy lifecycle input even though policy actuation is off;
        # only the explicitly enabled live-policy path may fold E0 into row
        # belief before making its own decision.
        if (
            runtime is not None
            and bool(getattr(runtime, "population_shadow_enabled", False))
            and self._population_live_requested()
        ):
            from eligsir.mapping.gaussian_belief_accumulator import (
                accumulate_true_b4_belief,
            )

            accumulation_status = accumulate_true_b4_belief(
                self.gaussians,
                result,
                current_versions=self._population_shadow_evidence_versions(),
            )
            # Compact status-only provenance for the asynchronous policy
            # summary.  It neither changes row state nor makes a topology
            # decision.
            setattr(
                result,
                "population_policy_e0_accumulator_status",
                accumulation_status.value,
            )
        targets = None
        if self._population_live_requested():
            targets = self._observe_population_surface_targets_after_step(
                result, source_leases
            )
        shadow_drop = self._observe_population_shadow_after_step(
            runtime, result, native_targets=targets
        )
        self._publish_native_geometry_evidence_demands(runtime)
        if native_geometry_actual_plan is not None:
            self._bind_native_geometry_evidence_after_success(
                native_geometry_actual_plan
            )
        return shadow_drop, targets

    def _publish_native_geometry_evidence_demands(self, runtime: Any) -> None:
        """Move compact shadow demand descriptors into the bounded lane.

        This is after the fixed-topology transaction returned.  It handles no
        tensor and deliberately retains the lane's first outstanding demand;
        later policy descriptors cannot replace an unresolved epoch.
        """

        lane = getattr(self, "_native_geometry_evidence_lane", None)
        bridge = getattr(runtime, "population_shadow_runtime", None)
        take = getattr(bridge, "take_latest_evidence_demands", None)
        if lane is None or not callable(take):
            return
        from eligsir.mapping.evidence.evidence_demand import EvidenceDemandKind

        for demand in take():
            if demand.kind is not EvidenceDemandKind.DEPTH_SURFACE:
                continue
            if not lane.publish(demand):
                break

    def _native_geometry_evidence_purpose(self):
        """Return one version-current scheduler purpose without choosing work."""

        lane = getattr(self, "_native_geometry_evidence_lane", None)
        versions = self._population_shadow_evidence_versions()
        if lane is None or versions is None:
            return None
        from eligsir.mapping.evidence.evidence_demand_scheduler import (
            EvidenceDemandCurrentVersions,
        )

        return lane.next_purpose(
            current_versions=EvidenceDemandCurrentVersions(
                map=versions.map_content,
                topology=versions.topology,
                coordinate=versions.coordinate_frame,
                schema=versions.schema,
                optimizer=versions.optimizer_update,
                surface=versions.surface_evidence,
                policy=versions.policy,
            ),
            epoch=versions.optimizer_update,
        )

    def native_geometry_evidence_telemetry(self) -> dict[str, Any]:
        """Expose serializable lane state and its bounded terminal trace."""

        lane = getattr(self, "_native_geometry_evidence_lane", None)
        telemetry = None if lane is None else lane.telemetry
        return {
            "state": (
                "disabled"
                if lane is None
                else "in_flight"
                if lane.in_flight is not None
                else "pending"
                if lane.pending is not None
                else "idle"
            ),
            "counters": {
                name: 0 if telemetry is None else int(getattr(telemetry, name))
                for name in (
                    "published", "queue_full", "expired", "stale", "deferred_unready",
                    "deferred_budget", "scheduled", "incomplete", "satisfied",
                )
            },
            "outcome_trace": [
                dict(item)
                for item in getattr(self, "_native_geometry_evidence_outcome_trace", ())
            ],
        }

    def _record_native_geometry_evidence_outcome(
        self,
        plan: MappingWorkPlan,
        *,
        phase: str,
        reason: Any,
        attestation: Any = None,
    ) -> None:
        """Retain a bounded CPU trace that joins terminal lane state to a plan."""

        trace = getattr(self, "_native_geometry_evidence_outcome_trace", None)
        if trace is None:
            trace = self._native_geometry_evidence_outcome_trace = []
        item = {
            "plan_id": plan.plan_id,
            "phase": phase,
            "reason": str(getattr(reason, "value", reason)),
        }
        trace_attestation = getattr(attestation, "trace", None)
        if phase == "completion":
            item["attestation"] = (
                trace_attestation() if callable(trace_attestation) else None
            )
        trace.append(item)
        del trace[:-16]

    def _bind_native_geometry_evidence_after_success(
        self, plan: MappingWorkPlan
    ) -> None:
        """Bind a pending purpose only after its exact B>1 plan returned."""

        lane = getattr(self, "_native_geometry_evidence_lane", None)
        if lane is None or not plan.native_geometry_evidence_purpose:
            return
        active_plan = getattr(self, "_population_shadow_active_plan", None)
        if active_plan is not plan:
            return
        purpose = lane.pending
        if purpose is None:
            return
        reason = lane.observe_scheduler_outcome(plan, purpose=purpose)
        self._record_native_geometry_evidence_outcome(
            plan, phase="schedule", reason=reason
        )

    def _complete_native_geometry_evidence(
        self, result: Any, attestation: Any, plan: MappingWorkPlan | None
    ) -> None:
        """Complete a lane only from the post-fence target attestation."""

        lane = getattr(self, "_native_geometry_evidence_lane", None)
        if lane is None or getattr(lane, "in_flight", None) is None:
            return
        if plan is not getattr(self, "_population_shadow_active_plan", None):
            reason = lane.discard_stale_completion()
            if plan is not None:
                self._record_native_geometry_evidence_outcome(
                    plan, phase="completion", reason=reason, attestation=None
                )
            return
        from eligsir.mapping.scheduler.native_geometry_evidence import (
            NativeGeometryEvidenceOutcome,
        )
        from eligsir.mapping.evidence.surface_target_attestation import (
            SurfaceTargetQualificationAttestation,
        )
        from eligsir.optimization.gaussian.supervision_evidence import (
            SupervisionEvidenceBatch,
        )

        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, SupervisionEvidenceBatch):
            # This terminal transaction cannot produce its scheduled E0
            # target. Retire the in-flight slot as incomplete so a later,
            # current policy demand can publish; do not retain a snapshot
            # whose exact post-Adam versions can no longer be recreated.
            demand = lane.in_flight
            reason = lane.complete(
                NativeGeometryEvidenceOutcome(
                    map_version=demand.map_version,
                    topology_version=demand.topology_version,
                    coordinate_version=demand.coordinate_version,
                    schema_version=demand.schema_version,
                    optimizer_version=demand.optimizer_version,
                    surface_version=demand.surface_version,
                    policy_version=demand.policy_version,
                    parameter_version=0,
                    realized_batch_size=int(getattr(result, "realized_batch_size", 0)),
                    realized_camera_count=int(
                        getattr(result, "realized_camera_count", 0)
                    ),
                    resolution_tier="non_native",
                    native_geometry_leases=0,
                    target_attestation=None,
                )
            )
            if plan is not None:
                self._record_native_geometry_evidence_outcome(
                    plan, phase="completion", reason=reason, attestation=None
                )
            return
        token = evidence.snapshot_token
        leases = evidence.geometry_render_leases
        outcome = NativeGeometryEvidenceOutcome(
            map_version=token.map_content,
            topology_version=token.topology,
            coordinate_version=token.coordinate_frame,
            schema_version=token.schema,
            optimizer_version=token.optimizer_update,
            surface_version=token.surface_evidence,
            policy_version=token.policy,
            parameter_version=token.parameter,
            realized_batch_size=int(getattr(result, "realized_batch_size", 0)),
            realized_camera_count=int(getattr(result, "realized_camera_count", 0)),
            resolution_tier=(
                "native"
                if len(leases) == int(getattr(self, "_batch_size", 4))
                and all(getattr(lease, "resolution_tier", None) == "native" for lease in leases)
                else "non_native"
            ),
            native_geometry_leases=sum(
                getattr(lease, "resolution_tier", None) == "native" for lease in leases
            ),
            target_attestation=(
                attestation
                if isinstance(attestation, SurfaceTargetQualificationAttestation)
                else None
            ),
        )
        reason = lane.complete(outcome)
        if plan is not None:
            self._record_native_geometry_evidence_outcome(
                plan,
                phase="completion",
                reason=reason,
                attestation=outcome.target_attestation,
            )

    def _population_live_surface_source_leases(
        self,
        result: Any,
        cameras_by_source: Any,
    ) -> dict[Any, Any]:
        """Borrow native source leases for the exact all-view E0 geometry set.

        This does not materialize a source, issue an upload, or recover an
        absent frame.  The M6 provider sees an empty mapping on any mismatch
        and clears its target cache, which makes the later live policy
        ABSTAIN.  A source only appears when it is represented by exactly the
        completed supervision evidence and its existing camera payload is
        still eligible for native geometry evidence.
        """

        from eligsir.mapping.integration.depth_render_reuse import (
            GeometryRenderLease,
        )
        from eligsir.optimization.gaussian.supervision_evidence import (
            SupervisionEvidenceBatch,
        )

        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, SupervisionEvidenceBatch):
            return {}
        if not isinstance(cameras_by_source, dict):
            return {}
        leases: dict[Any, Any] = {}
        for render_lease in evidence.geometry_render_leases:
            if not isinstance(render_lease, GeometryRenderLease):
                return {}
            source = render_lease.source_identity
            if source in leases:
                return {}
            camera = cameras_by_source.get(source)
            if camera is None:
                return {}
            source_lease = self._native_prepared_source_lease(
                camera, source, render_lease.modality_eligibility
            )
            if source_lease is None:
                return {}
            leases[source] = source_lease
        return leases

    def _population_live_requested(self) -> bool:
        """Return the explicit default-false live-policy request bit."""

        options = getattr(getattr(self, "config", None), "static_options", {})
        value = options.get("population_live_enabled", False)
        shadow = options.get("population_shadow_enabled", False)
        if type(value) is not bool:
            raise TypeError("population_live_enabled must be an exact bool")
        if type(shadow) is not bool:
            raise TypeError("population_shadow_enabled must be an exact bool")
        if value and not shadow:
            raise RuntimeError(
                "population live policy requires the enabled shadow evidence bridge"
            )
        return value

    def _fixed_topology_mcmc_runtime_options(self) -> tuple[Any | None, bool]:
        """Expose legacy MCMC hooks only while legacy population owns mutations."""

        if self._population_live_requested():
            return None, False
        return self._mcmc_strategy, True

    @staticmethod
    def _population_live_snapshot(versions: Any):
        """Convert one exact post-Adam version vector into a fence token."""

        from eligsir.mapping.population_shadow_runtime import PopulationShadowVersions
        from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken

        if not isinstance(versions, PopulationShadowVersions):
            return None
        return EvidenceSnapshotToken(
            map_content=versions.map_content,
            topology=versions.topology,
            coordinate_frame=versions.coordinate_frame,
            schema=versions.schema,
            optimizer_update=versions.optimizer_update,
            surface_evidence=versions.surface_evidence,
            policy=versions.policy,
            parameter=versions.parameter,
        )

    def _population_live_plan_context(self, current: Any, work: Any, plan: Any):
        """Capture the authoritative live identity/context at one plan fence."""

        import torch as torch_module

        from eligsir.contracts import SourceIdentity
        from eligsir.map.arbiter import ProposalKey as PopulationProposalKey
        from eligsir.map.row_state import GaussianRowStateRegistry
        from eligsir.mapping.population_plan_adapter import (
            PopulationPlanAdapterContext,
        )
        from eligsir.mapping.population_policy import MutationWorkToken
        from eligsir.optimization.gaussian.supervision_evidence import (
            EvidenceSnapshotToken,
        )
        from eligsir.runtime.transfers import DeviceTransferOwner

        if not self._population_live_requested():
            raise RuntimeError("population-plan context requires enabled live policy")
        if not isinstance(current, EvidenceSnapshotToken):
            raise TypeError("population-plan current snapshot is not canonical")
        if not isinstance(work, MutationWorkToken):
            raise TypeError("population-plan work token is not canonical")
        max_actions = getattr(self, "_population_live_max_actions", None)
        budget = getattr(getattr(plan, "mutation_budget", None), "max_total", None)
        snapshot_id = getattr(plan, "snapshot_id", None)
        if (
            type(max_actions) is not int
            or max_actions <= 0
            or type(budget) is not int
            or budget != work.units
            or work.units <= 0
            or work.units > max_actions
            or type(snapshot_id) is not int
            or snapshot_id < 0
            or current.optimizer_update != work.epoch
        ):
            raise ValueError("population-plan authorization is malformed or stale")

        backend = getattr(self, "backend", None)
        gaussians = getattr(self, "gaussians", None)
        versions = self._versions()
        if (
            (current.map_content, current.topology, current.parameter) != versions
            or current.coordinate_frame
            != getattr(backend, "coordinate_frame_version", None)
            or current.schema != getattr(backend, "schema_version", None)
        ):
            raise ValueError("population-plan snapshot is stale")

        transfer_owner = getattr(self, "_transfer_owner", None)
        means = getattr(gaussians, "get_xyz", None)
        if (
            not isinstance(transfer_owner, DeviceTransferOwner)
            or not isinstance(means, torch_module.Tensor)
            or means.ndim != 2
            or transfer_owner.device != means.device
        ):
            raise ValueError("population-plan transfer ownership is invalid")
        capacity = getattr(getattr(self, "_runtime_policy", None), "cap_max", None)
        if type(capacity) is not int or capacity < means.shape[0]:
            raise ValueError("population-plan capacity is invalid")

        identity_provider = getattr(
            backend, "current_population_plan_identity_snapshot", None
        )
        if not callable(identity_provider):
            raise TypeError("backend has no population-plan identity provider")
        registry, hypothesis_uids, next_uid = identity_provider()
        if (
            not isinstance(registry, GaussianRowStateRegistry)
            or not isinstance(hypothesis_uids, tuple)
            or len(hypothesis_uids) != means.shape[0]
            or type(next_uid) is not int
        ):
            raise ValueError("population-plan identity snapshot is malformed")

        supervision = getattr(plan, "supervision_sources", None)
        replay = getattr(plan, "replay_sources", None)
        if (
            not isinstance(supervision, tuple)
            or not isinstance(replay, tuple)
            or not supervision + replay
            or not all(isinstance(source, SourceIdentity) for source in supervision + replay)
            or len(set(supervision + replay)) != len(supervision + replay)
        ):
            raise ValueError("population-plan source identity is malformed")
        source = (supervision + replay)[0]
        context = PopulationPlanAdapterContext(
            current_snapshot=current,
            decision_snapshot=current,
            capacity=capacity,
            hypothesis_uids=hypothesis_uids,
            next_hypothesis_uid=next_uid,
            row_state_registry=registry,
            proposal_key=PopulationProposalKey(
                stream_epoch=source.stream_epoch,
                source_ordinal=source.source_ordinal,
                timestamp_ns=source.timestamp_ns,
                source_id=source.source_id,
                local_ordinal=snapshot_id,
            ),
            max_actions=work.units,
            transfer_owner=transfer_owner,
        )
        backend.install_population_plan_snapshot(current)
        return context

    @staticmethod
    def _is_population_target_binding_derivative(
        original: Any, bound: Any
    ) -> bool:
        """Accept only a target-bound derivative of this exact decision.

        A target matcher is allowed to add concrete surface-target IDs and
        the corresponding target-index tensor.  It is not a second policy:
        action codes, their bounded source/action envelope, demand objects,
        and both policy attestations remain the original borrowed objects.
        Identity checks are metadata-only and therefore add no device sync.
        """

        from eligsir.mapping.population_policy import PopulationDecision

        if not isinstance(original, PopulationDecision) or not isinstance(
            bound, PopulationDecision
        ):
            return False
        if not (
            original.has_authentic_structural_selection()
            and bound.has_authentic_structural_selection()
            and original.structural_selection is not None
            and bound.structural_selection is not None
        ):
            return False
        before = original.structural_selection
        after = bound.structural_selection
        return (
            bound.actions is original.actions
            and bound.evidence_demands is original.evidence_demands
            and bound.demand_masks is original.demand_masks
            and after.source_rows is before.source_rows
            and after.action_codes is before.action_codes
            and after.valid_mask is before.valid_mask
            and after.max_actions == before.max_actions
            and after._policy_actions is original.actions
            and after._policy_target_indices is bound.target_indices
        )

    def _maybe_actuate_population_policy_after_step(self, result: Any, plan: Any):
        """Commit at most one complete population plan after a transaction.

        Returning ``(requested, commit)`` makes the caller's ownership split
        explicit: when live mode is requested, missing evidence, stale
        versions, zero scheduler budget, incomplete adapters, or an ABSTAIN
        decision all leave the map unchanged *without* re-enabling lifecycle,
        delegated MCMC, or direct depth insertion for that transaction.
        """

        self._native_geometry_surface_target_attestation = None
        if not self._population_live_requested():
            return False, None
        self._population_live_activation_disposition = None

        runtime = getattr(self, "_optimization_runtime", None)
        bridge = getattr(runtime, "population_shadow_runtime", None)
        take = getattr(bridge, "take_latest_activation_observation", None)
        target_provider = getattr(self, "_population_live_target_binding_provider", None)
        context_provider = getattr(self, "_population_live_context_provider", None)
        if not (
            bool(getattr(runtime, "population_shadow_enabled", False))
            and callable(take)
            and callable(target_provider)
            and callable(context_provider)
            and getattr(self, "_transfer_owner", None) is not None
        ):
            self._population_live_activation_disposition = (
                "target_binding_rejected"
                if not callable(target_provider)
                else "population_context_rejected"
                if not callable(context_provider)
                else "population_adapter_rejected"
            )
            return True, None
        observation = take()
        if observation is None or observation.token.units <= 0:
            return True, None

        def record_adapter_rejection(disposition: str) -> None:
            self._population_live_activation_disposition = disposition
            record = getattr(bridge, "record_adapter_rejection", None)
            if callable(record):
                record(observation)

        origin = getattr(observation, "origin_snapshot", None)
        from eligsir.optimization.gaussian.supervision_evidence import (
            EvidenceSnapshotToken,
        )

        if not isinstance(origin, EvidenceSnapshotToken):
            record_adapter_rejection("population_adapter_rejected")
            return True, None
        current = self._population_live_snapshot(observation.versions)
        if current is None:
            record_adapter_rejection("population_adapter_rejected")
            return True, None
        versions = self._versions()
        if (
            (current.map_content, current.topology, current.parameter) != versions
            or current.optimizer_update != self._work_clock_owner.snapshot().optimizer_updates
            or int(getattr(getattr(plan, "mutation_budget", None), "max_total", -1))
            != observation.token.units
        ):
            record_adapter_rejection("population_adapter_rejected")
            return True, None
        try:
            bound = target_provider(observation.decision, origin, current, observation.token)
        except Exception:
            record_adapter_rejection("target_binding_rejected")
            return True, None
        if not isinstance(bound, tuple) or len(bound) != 2:
            record_adapter_rejection("target_binding_rejected")
            return True, None
        try:
            context = context_provider(current, observation.token, plan)
        except Exception:
            record_adapter_rejection("population_context_rejected")
            return True, None
        decision, targets = bound
        from eligsir.mapping.evidence.surface_targets import SurfaceTargetBatch
        from eligsir.mapping.population_plan_adapter import (
            PopulationPlanAdapterContext,
            PopulationPlanDropReason,
            PopulationPolicyPlanAdapter,
        )
        from eligsir.mapping.population_policy import PopulationDecision

        if (
            not isinstance(decision, PopulationDecision)
            or not self._is_population_target_binding_derivative(
                observation.decision, decision
            )
            or (targets is not None and not isinstance(targets, SurfaceTargetBatch))
        ):
            record_adapter_rejection("target_binding_rejected")
            return True, None
        if (
            not isinstance(context, PopulationPlanAdapterContext)
            or context.current_snapshot != current
            or context.decision_snapshot != current
            or context.max_actions != observation.token.units
            or context.capacity != self._runtime_policy.cap_max
            or context.transfer_owner is not self._transfer_owner
        ):
            record_adapter_rejection("population_context_rejected")
            return True, None
        built = PopulationPolicyPlanAdapter().build(
            decision=decision, targets=targets, context=context
        )
        self._native_geometry_surface_target_attestation = (
            built.target_attestation
        )
        proposal = built.proposal
        if proposal is None:
            # Policy ABSTAIN / no selected structural action is normal
            # selection telemetry, not an adapter failure.
            if built.reason is PopulationPlanDropReason.NO_STRUCTURAL_ACTION:
                self._population_live_activation_disposition = "no_structural_action"
            else:
                self._population_live_activation_disposition = "population_plan_rejected"
                record = getattr(bridge, "record_adapter_rejection", None)
                if callable(record):
                    record(observation, plan_drop_reason=built.reason)
                else:
                    record_adapter_rejection("population_plan_rejected")
            return True, None
        # The arbiter rejects any mixed plan.  Do not pass a legacy proposal
        # alongside this one: this is the sole MUTATION_CARDINALITY boundary.
        commit = self._commit_mutation_barrier(
            (proposal,),
            expected_map_content_version=current.map_content,
            expected_topology_version=current.topology,
        )
        return True, commit

    def _record_completed_population_policy_commit(self, commit: Any | None):
        """Consume neutral cadence after an applied live-policy plan.

        This is intentionally not ``record_mcmc_mutation``: MCMC no longer
        decided the action.  The retained cadence owner records the topology
        epoch plus neutral RGB/depth baselines, while its historical MCMC
        grow/relocate exposure debt remains untouched.
        """

        if commit is None or self._required_result_count(commit, "applied") <= 0:
            return self._work_clock_owner.snapshot()
        change = getattr(commit, "change", None)
        topology_after = getattr(change, "topology_version", None)
        if type(topology_after) is not int or topology_after < 0:
            raise TypeError(
                "population-policy commit must publish an exact topology version"
            )
        snapshot = self._work_clock_owner.snapshot()
        cadence_owner = getattr(self, "_mcmc_cadence_owner", None)
        if cadence_owner is not None:
            cadence_owner.record_population_mutation(
                snapshot,
                topology_version_after=topology_after,
            )
        return snapshot

    def _post_step_structural_mutation(
        self,
        result: Any,
        plan: Any,
        *,
        post_step_utility: Any | None,
    ) -> tuple[bool, Any | None]:
        """Select exactly one structural owner for a completed transaction."""

        if self._population_live_requested():
            _requested, commit = self._maybe_actuate_population_policy_after_step(
                result, plan
            )
            self._complete_native_geometry_evidence(
                result,
                getattr(self, "_native_geometry_surface_target_attestation", None),
                plan,
            )
            self._record_completed_population_policy_commit(commit)
            return True, commit
        return False, self._mutate_after_step(
            result, plan, post_step_utility=post_step_utility
        )

    def _assert_plan_version_fence(self, plan: MappingWorkPlan) -> None:
        current_versions = self._versions()
        planned_versions = (
            plan.map_content_version,
            plan.topology_version,
            plan.parameter_version,
        )
        if current_versions != planned_versions:
            raise StaleGpuWorkError(
                "stale mapping work plan version fence: "
                f"planned={planned_versions} current={current_versions}"
            )

    def _idle_replay_plan(
        self,
        snapshot: ControlSnapshot,
        active_plan: MappingWorkPlan,
        *,
        ready_replay_sources: frozenset[SourceIdentity] | None = None,
    ) -> MappingWorkPlan:
        """Mint replay-only work from already admitted identity-bound views."""
        phase_owner = getattr(self, "_post_refinement_phase_owner", None)
        if phase_owner is None:
            if hasattr(self, "config"):
                raise RuntimeError("canonical runtime is missing post-refinement phase ownership")
            post_refinement = False
        else:
            post_refinement = phase_owner.phase is PostRefinementPhase.REFINEMENT
        if post_refinement:
            replay_modalities = self._post_refinement_replay_catalog()
            reserved_global_source = None
        else:
            replay_modalities, reserved_global_source = self._shortlisted_replay_view(
                snapshot_id=snapshot.snapshot_id,
                control_tick_epoch=snapshot.control_tick_epoch,
            )
        if not replay_modalities and not post_refinement:
            # Retired identities (unmaterializable after bounded-cold eviction)
            # must not be resurrected through the previous plan, or idle replay
            # would repropose a source no camera can ever be built for.
            retired = getattr(self, "_retired_replay_sources", None) or {}
            replay_modalities = OrderedDict(
                (work.source, work.modality_eligibility)
                for work in (*active_plan.replay_work, *active_plan.supervision_work)
                if work.source not in retired
            )
        if ready_replay_sources is not None:
            if not isinstance(ready_replay_sources, frozenset) or not all(
                isinstance(source, SourceIdentity) for source in ready_replay_sources
            ):
                raise TypeError("ready_replay_sources must be a frozenset of SourceIdentity")
            all_replay_modalities = getattr(self, "_replay_modalities", {})
            replay_modalities = OrderedDict(
                (source, all_replay_modalities[source])
                for source in sorted(ready_replay_sources, key=SourceIdentity.ordering_key)
                if source in all_replay_modalities
            )
        if reserved_global_source not in replay_modalities:
            reserved_global_source = None
        replay_weights = getattr(self, "_replay_weights", {})
        force_reserved_global = (
            post_refinement
            and ready_replay_sources is None
            and any(
                float(replay_weights.get(source, 1.0)) > 1.0
                for source in replay_modalities
            )
        )
        if force_reserved_global:
            _shortlist, reserved_global_source = self._shortlisted_replay_view(
                snapshot_id=snapshot.snapshot_id,
                control_tick_epoch=snapshot.control_tick_epoch,
                reserved_global_epoch=int(
                    getattr(self, "optimizer_steps", snapshot.optimizer_steps_completed)
                ),
            )
            if reserved_global_source not in replay_modalities:
                reserved_global_source = None
        snapshot = replace(
            snapshot,
            replay_candidates=tuple(replay_modalities),
            replay_work_modalities=tuple(replay_modalities.values()),
            replay_candidate_weights=tuple(
                float(replay_weights.get(source, 1.0))
                for source in replay_modalities
            ),
            reserved_global_replay_candidate=reserved_global_source,
        )
        owner = getattr(self, "_work_budget_owner", None)
        if owner is None:
            from eligsir.mapping.scheduler.work_budget import WorkBudgetOwner

            owner = WorkBudgetOwner()
        # issue #1346: WorkBudgetOwner._resolve_transaction_resolution only
        # honors a ResolutionPolicyProvider proposal outside its native-
        # forcing branch (see that method) -- forcing native already ignores
        # any ceiling this provider proposes.  Skipping the provider call
        # there is a pure evidence-shaped optimization (its proposal would be
        # discarded anyway); it must NOT be skipped whenever the owner is NOT
        # currently forcing native (e.g. "bounded" always, or "progressive"
        # before its switch point), where the proposal is the ceiling.
        refinement_progress_fraction = (
            getattr(self, "_refinement_progress_fraction", None)
            if post_refinement
            else None
        )
        forces_native = post_refinement and owner.forces_native_post_refinement(
            refinement_progress_fraction
        )
        resolution_provider = (
            None if forces_native else getattr(self, "_resolution_policy_provider", None)
        )
        propose_idle = getattr(resolution_provider, "propose_idle", None)
        proposals = (
            (
                propose_idle(
                    snapshot,
                    optimizer_steps_completed=int(
                        getattr(
                            self,
                            "optimizer_steps",
                            snapshot.optimizer_steps_completed,
                        )
                    ),
                    mapping_cost_ms=getattr(
                        self,
                        "_latest_mapping_step_wall_ms",
                        getattr(self, "_latest_optimization_gpu_ms", None),
                    ),
                    batch_size=self._batch_size,
                    realtime_constraints=True,
                ),
            )
            if callable(propose_idle)
            else ()
        )
        mutation_provider = getattr(self, "_mutation_policy_provider", None)
        propose_idle_mutation = getattr(mutation_provider, "propose_idle", None)
        if callable(propose_idle_mutation) and not post_refinement:
            proposals = (*proposals, *propose_idle_mutation(snapshot))
        adaptive_growth = getattr(self, "_adaptive_growth_provider", None)
        propose_idle_growth = getattr(adaptive_growth, "propose_idle", None)
        if callable(propose_idle_growth):
            proposals = (*proposals, propose_idle_growth(snapshot))
        if self._replay_depth_due():
            proposals = (
                *proposals,
                ControlProposal(
                    "ReplayDepthTailProvider",
                    "replay_insertion_seeds",
                    self._replay_depth_max_per_event,
                    "bounded settled replay depth at fixed tail cadence",
                ),
            )
        pending_trials = self._prune_pending_replay_trial_sources()
        challenger_fraction = float(
            getattr(
                self._optimization_policy,
                "streaming_replay_challenger_fraction",
                0.0,
            )
        )
        ready_owner = getattr(self, "_ready_source_owner", None)
        ready_pending = frozenset(
            pending_trials
        ).intersection(ready_owner.ready_replay_sources() if ready_owner is not None else ())
        cold_challengers = ()
        if ready_replay_sources is not None:
            cold_challengers = tuple(
                source
                for source in pending_trials
                if source in ready_pending and source in ready_replay_sources
            )
        elif challenger_fraction > 0.0 and not post_refinement and ready_owner is not None:
            residents = frozenset(ready_owner.replay_resident_sources())
            cold_challengers = tuple(
                source
                for source in replay_modalities
                if source in ready_pending
                or (
                    source not in residents
                    and self._prepared_cold_pagein_capable(
                        self._observation_db.get_frame(
                            self._historical_frame_ids_by_source.get(source, -1)
                        )
                    )
                )
            )
        plan = owner.resolve_idle_replay(
            snapshot,
            active_plan,
            replay_modalities=replay_modalities,
            replay_weights=replay_weights,
            batch_size=self._batch_size,
            optimizer_steps_completed=int(
                getattr(self, "optimizer_steps", snapshot.optimizer_steps_completed)
            ),
            versions=self._versions(),
            proposals=proposals,
            post_refinement=post_refinement,
            refinement_progress_fraction=refinement_progress_fraction,
            force_reserved_global=force_reserved_global,
            cold_challenger_sources=cold_challengers,
            challenger_fraction=(
                1.0
                if ready_replay_sources is not None and cold_challengers
                else challenger_fraction if cold_challengers else 0.0
            ),
        )
        if ready_replay_sources is None:
            self._retain_pending_replay_trial_sources(
                plan.replay_trial_sources,
                limit=len(plan.replay_sources),
            )
        return plan

    def _tier_prepare_cost_ms(self, tier_value: str) -> float | None:
        """``T_prepare_saved`` for HOT retention, from the allocator's evidence.

        Deliberately a read of ``WorkBudgetOwner``'s own bounded rolling
        statistics rather than a second cost model: retention must price a tier
        with the same measured numbers, in the same calibration bucket, that
        selection prices it with.  ``None`` means "not measured yet", which the
        retention policy treats as unknown (neutral), never as zero.

        This is a cost read only.  Nothing here influences which tier is
        chosen -- ``WorkBudgetOwner`` remains the single owner of that.
        """

        owner = getattr(self, "_work_budget_owner", None)
        reader = getattr(owner, "tier_prepare_saving_ms", None)
        if not callable(reader):
            return None
        try:
            gaussian_count = (
                0 if not self.initialized else int(self.gaussians.get_xyz.shape[0])
            )
        except Exception:
            gaussian_count = 0
        try:
            return reader(
                tier_value,
                gaussian_count=gaussian_count,
                camera_count=max(1, int(self._batch_size)),
            )
        except Exception:
            return None

    def _prematerialize_tier_now(self) -> "ResolutionTier | None":
        """Tier to pre-materialize for the frame being admitted right now.

        ``None`` when ``--streaming_admission_prematerialize`` is off. The
        static construction-time tier when the mapping policy is concrete.
        Under ``adaptive`` -- where ``_resolve_prematerialize_tier`` yields
        no static answer -- the work budget's most recently selected
        transaction tier, falling back to the cheapest legal tier before the
        first transaction has run (see ``_prematerialize_tier_for``).
        ``getattr`` defaults keep the bare ``object.__new__`` test fixtures,
        which bypass ``__init__``, on the static-attribute path.
        """

        static_tier = getattr(self, "_admission_prematerialize_tier", None)
        if static_tier is not None:
            return static_tier
        if not getattr(self, "_admission_prematerialize_enabled", False):
            return None
        supported = getattr(self, "_admission_prematerialize_supported_tiers", ())
        if not supported:
            return None
        return _prematerialize_tier_for(
            tuple(supported),
            getattr(self, "_last_transaction_resolution_tier", None),
        )

    def _record_transaction_cost(
        self,
        plan: MappingWorkPlan,
        *,
        tick_started_ns: int,
        prepare_end_ns: int,
        optimization_end_ns: int,
        transaction_wall_ms: float,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ) -> None:
        """Record host preparation, execution, and whole-transaction costs.

        Preparation ends at the existing pre-dispatch host boundary, not the
        work clock's rolling checkpoint. Execution is unsynchronized host wall
        time; optional GPU-event telemetry remains separate. The whole wall
        cost, not either component, is the resolution selector's cost sensor.

        ``cache_hits``/``cache_misses`` count how many of this transaction's
        supervision identities were already HOT at the chosen tier.  They split
        the recorded preparation evidence by residency, so a first-use
        materialization and a steady-state HOT hit are two distinguishable cost
        samples rather than one blended average.
        """

        from eligsir.mapping.scheduler.tier_cost_model import FORWARD_BACKWARD

        owner = getattr(self, "_work_budget_owner", None)
        record = getattr(owner, "record_completed_transaction", None)
        if not callable(record):
            return
        views = len(plan.supervision_work) + len(plan.replay_work)
        if views <= 0:
            return
        prepare_ms = max(0.0, float(prepare_end_ns - tick_started_ns) / 1e6)
        optimization_ms = max(
            0.0, float(optimization_end_ns - prepare_end_ns) / 1e6
        )
        constellation = plan.supervision_constellation
        record(
            transaction_class=plan.transaction_class,
            tier=plan.resolution_tier,
            gaussian_count=(
                0 if not self.initialized else int(self.gaussians.get_xyz.shape[0])
            ),
            realized_camera_count=(
                views
                if constellation is None
                else int(constellation.realized_camera_count)
            ),
            prepare_ms=prepare_ms,
            optimization_ms=optimization_ms,
            transaction_wall_ms=float(transaction_wall_ms),
            kernel_ms=getattr(self, "_latest_optimization_gpu_ms", None),
            cache_hits=max(0, int(cache_hits)),
            cache_misses=max(0, int(cache_misses)),
            pass_mode=FORWARD_BACKWARD,
        )
        model_path = getattr(getattr(self, "_runtime_policy", None), "model_path", None)
        if model_path:
            from eligsir.mapping.scheduler.resolution_tier_usage import (
                record_resolution_tier_usage,
            )

            record_resolution_tier_usage(
                model_path,
                iteration=int(self.iteration),
                tier=plan.resolution_tier,
                transaction_class=plan.transaction_class,
            )
        # Feed the realized tier to admission-time pre-materialize under an
        # ``adaptive`` policy (``_prematerialize_tier_now``): the most recent
        # selection is the best predictor of what the next first draw asks.
        self._last_transaction_resolution_tier = plan.resolution_tier

    def _post_refinement_replay_catalog(self) -> OrderedDict[Any, Any]:
        """Resolve the full retained eligible DB catalog, never the live shortlist."""

        database = getattr(self, "_observation_db", None)
        if database is None:
            raise RuntimeError("post-refinement requires FrameObservationDB ownership")
        generation = getattr(database, "historical_rgb_materializable_generation", None)
        cached = getattr(self, "_post_refinement_replay_catalog_cache", None)
        if (
            generation is not None
            and cached is not None
            and getattr(self, "_post_refinement_replay_catalog_generation", None)
            == generation
        ):
            return cached
        candidates = database.historical_candidates(
            require_rgb=True,
            materializable_only=True,
        )
        sources = getattr(self, "_historical_sources_by_frame_id", {})
        modalities = getattr(self, "_replay_modalities", {})
        catalog: OrderedDict[Any, Any] = OrderedDict()
        for candidate in candidates:
            source = sources.get(candidate.frame_id)
            modality = modalities.get(source)
            if source is None or modality is None or not modality.rgb_supervision:
                raise RuntimeError("eligible FrameObservationDB record lacks replay identity")
            if (
                getattr(self, "_ready_source_owner", None) is not None
                and not self._prepared_cold_pagein_capable(
                    database.get_frame(int(candidate.frame_id))
                )
            ):
                continue
            catalog[source] = modality
        print(
            f"[DIAG-no_training_camera] _post_refinement_replay_catalog: "
            f"db_candidates={len(candidates)} catalog_size={len(catalog)} "
            f"ready_source_owner={getattr(self, '_ready_source_owner', None) is not None}",
            flush=True,
        )
        if generation is not None:
            self._post_refinement_replay_catalog_cache = catalog
            self._post_refinement_replay_catalog_generation = generation
        return catalog

    def _fresh_idle_snapshot(self, previous: ControlSnapshot) -> ControlSnapshot:
        """Fence replay-only work to a newly published control epoch."""

        content, topology, parameters = self._versions()
        risk_names = {"partition_fade", "quality_risk"}
        evidence = tuple(
            fact for fact in previous.evidence if fact.name not in risk_names
        ) + _adaptive_quality_evidence_facts(
            getattr(self, "_adaptive_supervision_quality_evidence", None)
        )
        snapshot = replace(
            previous,
            snapshot_id=self._control_snapshot_id,
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            optimizer_steps_completed=self.optimizer_steps,
            supervision_views_completed=self._supervision_views_completed,
            control_tick_epoch=self._control_tick_epoch,
            evidence=evidence,
            source_release_rate_ratio=self._source_rate_control_ratio(),
        )
        self._control_snapshot_id += 1
        return snapshot

    def control_snapshot(self, packet, observation, ingress_telemetry) -> ControlSnapshot:
        """Capture cheap, coherent CPU state after tracking and before pixels."""

        self._poll_gpu_telemetry()
        self._work_clock_owner.record_source_frame()
        self._source_clock_ns = max(self._source_clock_ns, int(packet.source.timestamp_ns))
        self._source_arrival_ns[packet.source.fingerprint()] = time.monotonic_ns()
        if len(self._source_arrival_ns) > 4096:
            oldest = next(iter(self._source_arrival_ns))
            del self._source_arrival_ns[oldest]
        self._source_identities_by_ordinal[packet.source.source_ordinal] = packet.source
        if len(self._source_identities_by_ordinal) > 4096:
            oldest = next(iter(self._source_identities_by_ordinal))
            del self._source_identities_by_ordinal[oldest]
        snapshot_id = self._control_snapshot_id
        self._control_snapshot_id += 1
        map_content_version, topology_version, parameter_version = self._versions()
        gaussian_count = 0 if not self.initialized else int(self.gaussians.get_xyz.shape[0])
        now_ns = time.monotonic_ns()
        pending_supervision = getattr(
            self, "_pending_supervision_modalities", OrderedDict()
        )
        candidate_depth = len(pending_supervision)
        candidate_since = None
        if pending_supervision:
            candidate_source = next(iter(pending_supervision))
            candidate_since = self._source_arrival_ns.get(
                candidate_source.fingerprint()
            )
        self._candidate_oldest_age_ns = (
            0 if candidate_since is None else max(0, now_ns - candidate_since)
        )
        self._commit_oldest_age_ns = (
            0 if self._pending_integration_since_ns is None
            else max(0, now_ns - self._pending_integration_since_ns)
        )
        transfer_depth = (
            0 if self._transfer_owner is None else self._transfer_owner.pending_count
        )
        gpu_depth = (
            self._gpu_event_telemetry.pending_count
            + len(self._pending_optimization_transaction_timings)
        )
        replay_view, reserved_global_source = self._shortlisted_replay_view(
            snapshot_id=snapshot_id,
            control_tick_epoch=self._control_tick_epoch,
        )
        replay_history = tuple(replay_view.items())
        replay_weights = getattr(self, "_replay_weights", {})
        ingress_facts = tuple(ingress_telemetry.evidence)
        ingress_by_name = {fact.name: fact.value for fact in ingress_facts}
        ingress_depth = int(
            ingress_by_name.get(
                "ingress_queue_depth",
                ingress_by_name.get("transport.queue_depth", 0),
            )
        )
        # Budget-relaxation reset signal (#1388): realized source release rate vs
        # the requested paced source clock. None for unpaced/live sources.
        source_release_rate_ratio = self._source_rate_control_ratio()
        control_evidence = self._control_evidence(
            packet,
            observation,
            ingress_depth=ingress_depth,
            candidate_depth=candidate_depth,
            transfer_depth=transfer_depth,
            gpu_depth=gpu_depth,
        )
        evidence = combine_control_evidence(ingress_facts, control_evidence)
        pressure_state = str(ingress_by_name.get("pressure_state", "low"))
        freshness = (
            FreshnessMode.CRITICAL
            if pressure_state == "critical"
            else FreshnessMode.CATCH_UP
            if pressure_state in {"rising", "high"}
            else FreshnessMode.HEALTHY
        )
        pending_sources = tuple(pending_supervision)
        if packet.source not in pending_supervision:
            pending_sources = (*pending_sources, packet.source)
        snapshot = ControlSnapshot(
            snapshot_id=snapshot_id,
            control_version=1,
            source=packet.source,
            tracking=observation,
            pending_sources=pending_sources,
            newest_frontier=packet.source,
            freshness_mode=freshness,
            pending_oldest_age_ns=max(
                self._candidate_oldest_age_ns, self._commit_oldest_age_ns
            ),
            ingress_queue_depth=ingress_depth,
            candidate_queue_depth=candidate_depth,
            transfer_queue_depth=transfer_depth,
            gpu_queue_depth=gpu_depth,
            map_content_version=map_content_version,
            topology_version=topology_version,
            parameter_version=parameter_version,
            gaussian_count=gaussian_count,
            optimizer_steps_completed=self.optimizer_steps,
            supervision_views_completed=self._supervision_views_completed,
            control_tick_epoch=self._control_tick_epoch,
            limits=ControlLimits(
                max_supervision_slots=max(1, self._batch_size - 1),
                max_replay_slots=(
                    1 if self._batch_size == 1 else self._batch_size - 1
                ),
                max_optimizer_updates=2 if self._batch_size == 1 else 1,
                max_insertion_seeds=max(
                    0, int(self._runtime_policy.max_new_gaussians_per_frame)
                ),
                max_mutations=self._population_live_max_actions,
                max_transfer_bytes=self._runtime_policy.max_transfer_bytes,
                max_replay_insertion_seeds=self._replay_depth_max_per_event,
                max_optimization_transaction_ns=(
                    self._runtime_policy.max_optimization_transaction_ns
                ),
            ),
            supervision_candidates=pending_sources,
            replay_candidates=tuple(source for source, _ in replay_history),
            replay_work_modalities=tuple(modalities for _, modalities in replay_history),
            replay_candidate_weights=tuple(
                float(replay_weights.get(source, 1.0)) for source, _ in replay_history
            ),
            reserved_global_replay_candidate=reserved_global_source,
            latest_gpu_telemetry=(
                self._latest_completed_optimization_transaction
            ),
            evidence=evidence,
            source_release_rate_ratio=source_release_rate_ratio,
        )
        proposals = collect_control_proposals(
            snapshot, self._control_proposal_providers
        )
        self._latest_control_snapshot = snapshot
        self._latest_control_outputs = (snapshot, proposals)
        if getattr(self, "_latest_control_packet", None) is not None:
            raise RuntimeError("previous control packet has no terminal disposition")
        self._latest_control_packet = packet
        return snapshot

    def record_source_consumed(self, packet: Any) -> None:
        """Record one consumed ingress attempt in the canonical source clock."""
        self._services.four_clocks.record_source_frame(
            frame_timestamp=packet.source.timestamp_seconds
        )

    def set_requested_source_fps(self, source_fps: float | None) -> None:
        """Declare the paced source clock (Hz) used for the budget-relaxation reset signal.

        Pass ``None`` for unpaced / live sources; the relaxation then ignores
        source rate (behaviour identical to before #1388).
        """
        self._requested_source_fps_for_pacing = (
            None if source_fps is None or float(source_fps) <= 0.0 else float(source_fps)
        )

    def set_source_input_exhausted_provider(
        self, provider: Callable[[], bool] | None
    ) -> None:
        """Read the paced producer's terminal input status without owning it."""
        if provider is not None and not callable(provider):
            raise TypeError("source input exhausted provider must be callable or None")
        self._source_input_exhausted_provider = provider

    def _source_rate_control_ratio(self) -> float | None:
        provider = getattr(self, "_source_input_exhausted_provider", None)
        if provider is not None and provider():
            return None
        return source_release_rate_ratio_from_clocks(
            getattr(getattr(self, "_services", None), "four_clocks", None),
            requested_source_fps=self._requested_source_fps_for_pacing,
        )

    def _control_evidence(
        self,
        packet: Any,
        observation: Any,
        *,
        ingress_depth: int = 0,
        candidate_depth: int = 0,
        transfer_depth: int = 0,
        gpu_depth: int = 0,
    ) -> tuple[EvidenceFact, ...]:
        """Expose immutable tracking and ingress modality evidence to the owner."""

        pose = getattr(observation, "pose", None)
        current_pose = None if pose is None else np.asarray(pose.matrix)
        translation_m, rotation_deg = pose_delta_signals(
            self._last_admitted_pose, current_pose
        )
        confidence = getattr(observation, "confidence", None)
        diagnostics = getattr(observation, "diagnostics", {})
        novelty = float(diagnostics.get("novelty", 0.0))
        coverage_gain = (
            None
            if diagnostics.get("coverage_gain") is None
            else float(diagnostics["coverage_gain"])
        )
        new_region = bool(
            (coverage_gain is not None and coverage_gain > 0.0)
            or novelty >= 0.5
        )
        quality_evidence = getattr(
            self, "_adaptive_supervision_quality_evidence", None
        )
        quality_facts = _adaptive_quality_evidence_facts(quality_evidence)
        config = getattr(self, "config", None)
        keyframe_policy = {} if config is None else config.keyframe_policy
        quality_enabled = bool(keyframe_policy.get("quality_enabled", False))
        exposure_gate_enabled = bool(
            keyframe_policy.get("exposure_gate_enabled", False)
        )
        blur_gate_enabled = bool(
            keyframe_policy.get("blur_gate_enabled", False)
        )
        rgb_hwc = _packet_rgb_hwc(packet)
        rgb = np.empty((0, 0, 3), dtype=np.float32) if rgb_hwc is None else rgb_hwc
        rgb_ok = bool(rgb_hwc is not None and rgb.size and np.isfinite(rgb).all())
        rgb_blur = (
            blur_laplacian_score(rgb)
            if rgb_ok and quality_enabled
            else None
        )
        standalone_blur = (
            None
            if not rgb_ok or not blur_gate_enabled
            else blur_laplacian_score(
                rgb,
                downsample=max(
                    1,
                    int(
                            getattr(
                                getattr(self, "_runtime_policy", None),
                                "blur_downsample",
                                4,
                            )
                    ),
                ),
            )
        )
        rgb_luma = (
            mean_luma(rgb)
            if rgb_ok and exposure_gate_enabled
            else None
        )
        rgb_clip_ratio = (
            float(np.mean((rgb <= 0) | (rgb >= 255)))
            if rgb_ok and quality_enabled
            else None
        )
        depth_ref = getattr(packet, "depth", None)
        depth_ok = False
        depth_valid_ratio = 0.0
        motion_thresholds = getattr(self, "_motion_admission_thresholds", (0.05, 5.0))
        depth_quality = "missing"
        if depth_ref is not None:
            depth = np.asarray(depth_ref.data, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.0)
            depth_valid_ratio = 0.0 if not depth.size else float(valid.mean())
            depth_ok = bool(depth.size and depth_valid_ratio >= 0.10)
            depth_quality = "qualified" if depth_ok else "insufficient_valid_depth"
        clocks_owner = getattr(getattr(self, "_services", None), "four_clocks", None)
        clocks = None if clocks_owner is None else clocks_owner.summary()
        source_rate = 0.0 if clocks is None else float(clocks.source_frame_rate_hz)
        admitted_rate = 0.0 if clocks is None else float(clocks.admitted_keyframe_rate_hz)
        commit_rate = 0.0 if clocks is None else float(clocks.map_commit_rate_hz)
        mapping_service_ratio = 1.0 if source_rate <= 0.0 else commit_rate / source_rate
        admission_ratio = 1.0 if source_rate <= 0.0 else admitted_rate / source_rate
        source_clock_ns = int(getattr(self, "_source_clock_ns", 0))
        map_clock_ns = int(getattr(self, "_map_clock_ns", 0))
        frontier_age_ms = (
            0.0
            if map_clock_ns <= 0
            else max(0.0, float(source_clock_ns - map_clock_ns) / 1e6)
        )
        sample_ns = time.monotonic_ns()
        prior_sample_ns = getattr(self, "_last_frontier_sample_ns", None)
        slope = 0.0
        if prior_sample_ns is not None and sample_ns > prior_sample_ns:
            slope = (frontier_age_ms - getattr(self, "_last_frontier_age_ms", 0.0)) / (
                float(sample_ns - prior_sample_ns) / 1e9
            )
        self._last_frontier_sample_ns = sample_ns
        self._last_frontier_age_ms = frontier_age_ms
        gpu_cost_ms = (
            None
            if getattr(self, "_latest_gpu_transaction", None) is None
            else float(self._latest_gpu_transaction.duration_ms)
        )
        gpu_updates = (
            0
            if getattr(self, "_latest_gpu_transaction", None) is None
            else int(self._latest_gpu_transaction.optimizer_updates)
        )
        observation_value = float(
            diagnostics.get(
                "observation_value",
                0.0 if confidence is None else confidence,
            )
        )
        candidate_index = getattr(self, "_historical_candidate_index", None)
        candidate_config = getattr(candidate_index, "config", None)
        replay_shortlist_configured_limit = getattr(
            candidate_config, "shortlist_size", None
        )
        parallax_enabled = bool(keyframe_policy.get("parallax_gate", False))
        parallax_deg = None
        if parallax_enabled:
            parallax_deg = _packet_parallax_deg(
                packet,
                current_pose,
                getattr(self, "_last_admitted_pose", None),
                depth_min=self._depth_insertion_policy.min_depth_m,
                depth_max=self._depth_insertion_policy.max_depth_m,
                stride=int(
                    keyframe_policy.get(
                        "coverage_gain_depth_stride", self.config.bootstrap.depth_stride
                    )
                ),
            )
        if (
            parallax_enabled
            and getattr(self, "_last_admitted_pose", None) is not None
            and parallax_deg is None
        ):
            raise RuntimeError(
                "configured parallax gate requires exact packet depth, intrinsics, "
                "and current/last-admitted poses"
            )
        return (
            EvidenceFact("source_rate_hz", source_rate),
            EvidenceFact("admitted_rate_hz", admitted_rate),
            EvidenceFact("commit_rate_hz", commit_rate),
            EvidenceFact("mapping_service_ratio", mapping_service_ratio),
            EvidenceFact("admission_ratio", admission_ratio),
            EvidenceFact(
                "replay_ratio",
                min(
                    1.0,
                    len(getattr(self, "_replay_modalities", ()))
                    / max(1, int(getattr(self, "_admitted_observations_total", 0))),
                ),
            ),
            EvidenceFact(
                "replay_catalog_population",
                len(getattr(self, "_replay_modalities", ())),
            ),
            EvidenceFact(
                "replay_shortlist_configured_limit",
                replay_shortlist_configured_limit,
            ),
            EvidenceFact("frontier_age_ms", frontier_age_ms),
            EvidenceFact("frontier_age_slope_ms_per_s", slope),
            EvidenceFact("ingress_queue_depth", ingress_depth),
            EvidenceFact("candidate_queue_depth", candidate_depth),
            EvidenceFact("transfer_queue_depth", transfer_depth),
            EvidenceFact("gpu_queue_depth", gpu_depth),
            EvidenceFact("gpu_cost_ms", gpu_cost_ms),
            EvidenceFact(
                "gpu_cost_per_update_ms",
                None if gpu_cost_ms is None or gpu_updates <= 0 else gpu_cost_ms / gpu_updates,
            ),
            EvidenceFact("observation_value", observation_value),
            EvidenceFact(
                "admitted_observations_total",
                int(getattr(self, "_admitted_observations_total", 0)),
            ),
            EvidenceFact("translation_m", translation_m),
            EvidenceFact("rotation_deg", rotation_deg),
            EvidenceFact("motion", max(translation_m, rotation_deg / 180.0)),
            EvidenceFact("quality", 0.0 if confidence is None else float(confidence)),
            EvidenceFact("novelty", novelty),
            EvidenceFact(
                "overlap",
                None
                if diagnostics.get("overlap") is None
                else float(diagnostics["overlap"]),
            ),
            EvidenceFact(
                "coverage_gain",
                coverage_gain,
            ),
            EvidenceFact("new_region", new_region),
            *quality_facts,
            EvidenceFact("reconstruction_residual", float(diagnostics.get("reconstruction_residual", 0.0))),
            EvidenceFact(
                "rgb_quality_score",
                float(
                    diagnostics.get(
                        "rgb_quality", 1.0 if confidence is None else confidence
                    )
                ),
            ),
            EvidenceFact(
                "depth_quality_score",
                float(diagnostics.get("depth_quality_score", depth_valid_ratio)),
            ),
            EvidenceFact(
                "rgb_blur_laplacian",
                rgb_blur,
            ),
            EvidenceFact("rgb_standalone_blur_laplacian", standalone_blur),
            EvidenceFact(
                "rgb_luma",
                rgb_luma,
            ),
            EvidenceFact(
                "parallax_deg",
                parallax_deg,
            ),
            EvidenceFact(
                "rgb_clip_ratio",
                rgb_clip_ratio,
            ),
            EvidenceFact("depth_valid_ratio", depth_valid_ratio),
            EvidenceFact(
                "cheap_keyframe",
                motion_admission_evidence(
                    translation_m,
                    rotation_deg,
                    min_translation_m=motion_thresholds[0],
                    min_rotation_deg=motion_thresholds[1],
                ),
            ),
            EvidenceFact("rgb_supervision_eligible", rgb_ok),
            EvidenceFact("rgb_quality", "qualified" if rgb_ok else "invalid"),
            EvidenceFact("depth_supervision_eligible", depth_ok),
            EvidenceFact("depth_quality", depth_quality),
            EvidenceFact("depth_insertion_eligible", depth_ok),
            EvidenceFact("geometry_population_target_eligible", depth_ok),
        )

    def record_disposition(self, disposition: ObservationDisposition) -> None:
        """Resolve executable work from the exact snapshot that was decided."""

        packet = getattr(self, "_latest_control_packet", None)
        self._latest_control_packet = None
        if packet is not None and packet.source != disposition.source:
            raise ValueError("terminal disposition packet identity mismatch")
        outputs = self._latest_control_outputs
        if outputs is None or outputs[0].snapshot_id != disposition.snapshot_id:
            raise RuntimeError("terminal disposition has no matching control snapshot")
        snapshot, proposals = outputs
        if snapshot.source != disposition.source:
            raise ValueError("terminal disposition source must match control snapshot")
        purpose = self._native_geometry_evidence_purpose()
        plan = self._work_budget_owner.resolve(
            snapshot,
            disposition=disposition,
            proposals=proposals,
            native_geometry_evidence_purpose=purpose is not None,
        )
        self._dispositions_since_tick.append(disposition)
        self._latest_report_work = (snapshot, plan)
        if (
            disposition.semantic_action is SemanticAction.REJECT
            and self._observation_db.cold_payload_disk_enabled
        ):
            if packet is None:
                raise RuntimeError("rejected disposition has no synchronous source packet")
            self.cache_nontraining_packet(packet, role="rejected")
        if disposition.cold_retention_permitted:
            if packet is None:
                raise RuntimeError("deferred disposition has no synchronous source packet")
            tracking = snapshot.tracking
            if tracking is None:
                raise RuntimeError("deferred historical observation requires tracking")
            permit = self.observation_owner.cold_retention_permit(disposition)
            if permit is None:
                raise RuntimeError("deferred disposition did not mint cold retention")
            self._retain_deferred_observation(
                disposition,
                permit,
                packet,
                tracking,
            )
            replay_modalities = getattr(self, "_replay_modalities", None)
            if replay_modalities is None:
                replay_modalities = self._replay_modalities = OrderedDict()
            replay_modalities[disposition.source] = disposition.modality_eligibility
            replay_modalities.move_to_end(disposition.source)
            replay_weights = getattr(self, "_replay_weights", None)
            if replay_weights is None:
                replay_weights = self._replay_weights = {}
            replay_weights.setdefault(disposition.source, 1.0)
        if disposition.admit_supervision:
            work_clock_owner = self._work_clock_owner_if_available()
            if work_clock_owner is not None:
                work_clock_owner.record_admission()
                self._latest_admitted_optimizer_update = (
                    work_clock_owner.snapshot().optimizer_updates
                )
                optimization_runtime = getattr(
                    self, "_optimization_runtime", None
                )
                if optimization_runtime is not None:
                    optimization_runtime.note_admitted_source(
                        optimizer_updates_completed=(
                            self._latest_admitted_optimizer_update
                        )
                    )
            self._last_admitted_control = (snapshot, proposals, disposition, plan)
            self._admitted_observations_total = int(
                getattr(self, "_admitted_observations_total", 0)
            ) + 1
            clocks = getattr(getattr(self, "_services", None), "four_clocks", None)
            if clocks is not None:
                clocks.record_admitted_keyframe()
            tracking = snapshot.tracking
            pose = None if tracking is None else tracking.pose
            self._last_admitted_pose = (
                None if pose is None else np.array(pose.matrix, copy=True)
            )
            self._admitted_clock_ns = max(
                int(getattr(self, "_admitted_clock_ns", 0)),
                int(disposition.source.timestamp_ns),
            )
            work = (snapshot, plan)
            replay_modalities = getattr(self, "_replay_modalities", None)
            if replay_modalities is None:
                replay_modalities = self._replay_modalities = OrderedDict()
            replay_modalities[disposition.source] = disposition.modality_eligibility
            pending_supervision = getattr(
                self, "_pending_supervision_modalities", None
            )
            if pending_supervision is None:
                pending_supervision = self._pending_supervision_modalities = OrderedDict()
            pending_supervision[disposition.source] = disposition.modality_eligibility
            pending_supervision.move_to_end(disposition.source)
            pending_capacity = int(
                getattr(self, "_pending_supervision_capacity", 1)
            )
            while len(pending_supervision) > pending_capacity:
                # A superseded pending identity remains in the full replay
                # cache and can still receive weighted supervision. It merely
                # stops consuming the bounded new-frame lane.
                pending_supervision.popitem(last=False)
            replay_weights = getattr(self, "_replay_weights", None)
            if replay_weights is None:
                replay_weights = self._replay_weights = {}
            replay_weights.setdefault(disposition.source, 1.0)
            replay_modalities.move_to_end(disposition.source)
            # The bounded candidate lane preserves admitted-but-not-yet-
            # optimized identities until one canonical transaction retires
            # them. The CPU-only replay identity catalog retains every admitted
            # cold record; heavy camera/GPU residency remains bounded elsewhere.
            self._latest_tick_work = work
            self._active_materialization_work = work

    def record_prepared_disposition(self, disposition: ObservationDisposition) -> None:
        """Acknowledge one staged control decision without creating work slots.

        Prepared ingress retains this immutable token with its source.  In
        particular, it must not populate ``_latest_tick_work`` or
        ``_active_materialization_work``: a later bridge owns camera and
        transaction construction, so overwriting a global slot here would
        corrupt multiple accepted prepared sources.
        """

        packet = getattr(self, "_latest_control_packet", None)
        self._latest_control_packet = None
        if packet is not None and packet.source != disposition.source:
            raise ValueError("terminal disposition packet identity mismatch")
        outputs = self._latest_control_outputs
        if outputs is None or outputs[0].snapshot_id != disposition.snapshot_id:
            raise RuntimeError("terminal disposition has no matching control snapshot")
        snapshot, _proposals = outputs
        if snapshot.source != disposition.source:
            raise ValueError("terminal disposition source must match control snapshot")
        work_budget_owner = getattr(self, "_work_budget_owner", None)
        if work_budget_owner is not None:
            plan = work_budget_owner.resolve(
                snapshot,
                disposition=disposition,
                proposals=_proposals,
                native_geometry_evidence_purpose=(
                    self._native_geometry_evidence_purpose() is not None
                ),
                source_depth_requires_native=bool(
                    getattr(getattr(self, "config", None), "static_options", {}).get(
                        "prepared_ready_source_insertion", False
                    )
                ),
            )
            # A later disposition may improve an incomplete candidate, but
            # must first freeze an executable B4 before replacing it.
            if self._prepared_ready_plan_for_execution() is None:
                self._prepared_ready_candidate_work = (snapshot, plan)
        contexts_by_source = getattr(
            self, "_prepared_execution_contexts_by_source", None
        )
        if contexts_by_source is None:
            contexts_by_source = self._prepared_execution_contexts_by_source = OrderedDict()
        previous = contexts_by_source.get(disposition.source)
        if previous is not None:
            if previous.disposition != disposition:
                raise ValueError("prepared source disposition cannot change")
            return
        axes = PreparedControlAxes(
            snapshot_id=snapshot.snapshot_id,
            control_version=snapshot.control_version,
            freshness_mode=snapshot.freshness_mode,
            limits=snapshot.limits,
            proposals=tuple(_proposals),
        )
        # The token is completed by MappingSession after this terminal sink;
        # keep a source-local placeholder and verify the immutable token at
        # the bridge lookup rather than introducing a latest-work slot.
        placeholder = PreparedIngressControlToken(
            sequence=int(disposition.source.source_ordinal),
            source=disposition.source,
            snapshot_id=disposition.snapshot_id,
            disposition=disposition,
        )
        lane = (
            ReadySourceLane.FRESH
            if disposition.semantic_action is SemanticAction.COMMIT
            else ReadySourceLane.REPLAY
        )
        context = PreparedExecutionContext(
            source=disposition.source,
            control_token=placeholder,
            disposition=disposition,
            control_axes=axes,
            snapshot=snapshot,
            lane=lane,
            replay_priority=(
                ReplayPriority.NORMAL
                if lane is ReadySourceLane.FRESH
                else ReplayPriority.DEMANDED
            ),
        )
        contexts_by_source[disposition.source] = context
        move_to_end = getattr(contexts_by_source, "move_to_end", None)
        if callable(move_to_end):
            move_to_end(disposition.source)
        capacity = int(getattr(self, "_prepared_execution_context_capacity", 4096))
        while len(contexts_by_source) > capacity:
            popitem = getattr(contexts_by_source, "popitem", None)
            if not callable(popitem):
                break
            popitem(last=False)
        self._dispositions_since_tick.append(disposition)
        if disposition.cold_retention_permitted and packet is not None:
            tracking = (
                snapshot.tracking if snapshot.tracking is not None else packet.tracking
            )
            if tracking is None:
                raise RuntimeError("deferred historical observation requires tracking")
            permit = self.observation_owner.cold_retention_permit(disposition)
            if permit is None:
                raise RuntimeError("deferred disposition did not mint cold retention")
            self._retain_deferred_observation(disposition, permit, packet, tracking)
            self._observation_db.record_prepared_defer_context(
                disposition.source,
                (tracking, disposition, placeholder),
            )
        if disposition.cold_retention_permitted or disposition.admit_supervision:
            replay_modalities = getattr(self, "_replay_modalities", None)
            if replay_modalities is None:
                replay_modalities = self._replay_modalities = OrderedDict()
            replay_modalities[disposition.source] = disposition.modality_eligibility
            replay_modalities.move_to_end(disposition.source)
            replay_weights = getattr(self, "_replay_weights", None)
            if replay_weights is None:
                replay_weights = self._replay_weights = {}
            replay_weights.setdefault(disposition.source, 1.0)
        if not disposition.admit_supervision:
            return
        work_clock_owner = self._work_clock_owner_if_available()
        if work_clock_owner is not None:
            work_clock_owner.record_admission()
            self._latest_admitted_optimizer_update = (
                work_clock_owner.snapshot().optimizer_updates
            )
            optimization_runtime = getattr(self, "_optimization_runtime", None)
            if optimization_runtime is not None:
                optimization_runtime.note_admitted_source(
                    optimizer_updates_completed=self._latest_admitted_optimizer_update
                )
        self._admitted_observations_total = int(
            getattr(self, "_admitted_observations_total", 0)
        ) + 1
        clocks = getattr(getattr(self, "_services", None), "four_clocks", None)
        if clocks is not None:
            clocks.record_admitted_keyframe()
        self._admitted_clock_ns = max(
            int(getattr(self, "_admitted_clock_ns", 0)),
            int(disposition.source.timestamp_ns),
        )
        pending_supervision = getattr(self, "_pending_supervision_modalities", None)
        if pending_supervision is None:
            pending_supervision = self._pending_supervision_modalities = OrderedDict()
        pending_supervision[disposition.source] = disposition.modality_eligibility
        pending_supervision.move_to_end(disposition.source)
        pending_capacity = int(getattr(self, "_pending_supervision_capacity", 1))
        while len(pending_supervision) > pending_capacity:
            pending_supervision.popitem(last=False)

    def prepared_execution_context_for(
        self,
        control_token: PreparedIngressControlToken,
        *,
        promotion_permit: Any | None = None,
    ) -> PreparedExecutionContext | None:
        """Return one immutable source context, never an admission-time work plan."""

        if not isinstance(control_token, PreparedIngressControlToken):
            raise TypeError("control_token must be PreparedIngressControlToken")
        key = (
            control_token.sequence,
            control_token.source,
            control_token.snapshot_id,
        )
        contexts = getattr(self, "_prepared_execution_contexts", None)
        if contexts is None:
            contexts = self._prepared_execution_contexts = OrderedDict()
        context = contexts.get(key)
        durable_context = False
        if context is None:
            contexts_by_source = getattr(
                self, "_prepared_execution_contexts_by_source", {}
            )
            candidate = contexts_by_source.get(control_token.source)
            if candidate is None:
                stored = self._observation_db.prepared_commit_obligation_for_source(
                    control_token.source
                )
                if stored is None:
                    return None
                if not (isinstance(stored, tuple) and len(stored) == 4):
                    raise RuntimeError("prepared COMMIT obligation lacks execution context")
                _tracking, disposition, token, candidate = stored
                if disposition != control_token.disposition or token != control_token:
                    raise ValueError("prepared control token disposition differs")
                if candidate is None:
                    raise RuntimeError("prepared COMMIT obligation lacks execution context")
                durable_context = True
            if control_token.disposition != candidate.disposition:
                raise ValueError("prepared control token disposition differs")
            existing_key = next(
                (
                    recorded_key
                    for recorded_key, recorded in contexts.items()
                    if recorded.source == control_token.source
                ),
                None,
            )
            if existing_key is not None and existing_key != key:
                raise ValueError("prepared control token differs for source")
            context = replace(candidate, control_token=control_token)
            if not durable_context:
                contexts[key] = context
        if control_token != context.control_token:
            raise ValueError("prepared control token disposition differs")
        if context.disposition.semantic_action is SemanticAction.REJECT:
            return None
        if context.disposition.semantic_action is SemanticAction.DEFER:
            if promotion_permit is None:
                return None
            if not isinstance(promotion_permit, HistoricalPromotionPermit):
                raise TypeError("promotion_permit must be HistoricalPromotionPermit")
            if (
                promotion_permit.source != context.source
                or promotion_permit.disposition_snapshot_id
                != context.disposition.snapshot_id
            ):
                raise ValueError("promotion permit does not match deferred source")
            return replace(
                context,
                control_token=control_token,
                promotion_permit=promotion_permit,
            )
        return replace(context, control_token=control_token)

    def _activate_successful_prepared_deferred_sources(
        self, sources: Iterable[SourceIdentity]
    ) -> None:
        """Advance DEFER sources only when a prepared transaction supervised them."""

        contexts_by_source = getattr(self, "_prepared_execution_contexts_by_source", {})
        contexts = getattr(self, "_prepared_execution_contexts", {})
        for source in sources:
            context = contexts_by_source.get(source)
            if (
                context is None
                or context.disposition.semantic_action is not SemanticAction.DEFER
            ):
                continue
            contexts_by_source.pop(source)
            for key, recorded in tuple(contexts.items()):
                if recorded.source == source:
                    del contexts[key]
            self._admitted_clock_ns = max(
                int(getattr(self, "_admitted_clock_ns", 0)),
                int(source.timestamp_ns),
            )
            clocks = getattr(getattr(self, "_services", None), "four_clocks", None)
            if clocks is not None:
                clocks.record_admitted_keyframe()
            work_clock_owner = self._work_clock_owner_if_available()
            if work_clock_owner is not None:
                self._latest_admitted_optimizer_update = (
                    work_clock_owner.snapshot().optimizer_updates
                )
                optimization_runtime = getattr(self, "_optimization_runtime", None)
                if optimization_runtime is not None:
                    optimization_runtime.note_admitted_source(
                        optimizer_updates_completed=self._latest_admitted_optimizer_update
                    )

    def _prepared_cold_promotion_snapshot(self):
        """Return the current prepared-work authority for COLD page-in."""

        active_work = (
            getattr(self, "_prepared_ready_candidate_work", None)
            or getattr(self, "_prepared_ready_frozen_work", None)
        )
        if active_work is not None:
            return active_work[0]
        outputs = getattr(self, "_latest_control_outputs", None)
        return None if outputs is None else outputs[0]

    def prepared_cold_promotion_context_for(self, source_key: object):
        """Return one durable DEFER or retained-COMMIT resolver context.

        This method only reads bounded source-local control state and the
        exact DB record.  It never decodes, stages, transfers, or binds a
        camera; those operations remain on ``PreparedSourcePrepWorker``.
        """

        from eligsir.runtime.prepared_cold_promotion import (
            PreparedColdPromotionContext,
        )

        stored = self._observation_db.prepared_defer_context_for_source(source_key)
        retained_commit = False
        completed_commit = False
        if stored is None:
            stored = self._observation_db.prepared_commit_obligation_for_source(
                source_key
            )
            retained_commit = stored is not None
        if stored is None:
            stored = self._observation_db.prepared_commit_context_for_source(
                source_key
            )
            completed_commit = stored is not None
        if stored is None:
            return None
        if retained_commit or completed_commit:
            tracking, disposition, token, _context = stored
        else:
            tracking, disposition, token = stored
        if retained_commit:
            if disposition.semantic_action is not SemanticAction.COMMIT:
                raise RuntimeError("retained fresh context is not a COMMIT")
            candidate = self._observation_db.historical_candidate_for_source(
                disposition.source, materializable_only=False
            )
            if candidate is None or candidate.semantic_action != SemanticAction.COMMIT.value:
                return None
            return PreparedColdPromotionContext(
                observation=tracking,
                disposition=disposition,
                control_token=token,
                initial_fresh=True,
            )
        if completed_commit:
            if disposition.semantic_action is not SemanticAction.COMMIT:
                raise RuntimeError("completed prepared context is not a COMMIT")
            candidate = self._observation_db.historical_candidate_for_source(
                disposition.source, materializable_only=False
            )
            if candidate is None or candidate.semantic_action != SemanticAction.COMMIT.value:
                return None
            return PreparedColdPromotionContext(
                observation=tracking,
                disposition=disposition,
                control_token=token,
                initial_fresh=False,
            )
        current_snapshot = self._prepared_cold_promotion_snapshot()
        if current_snapshot is None:
            return None
        candidate = self._observation_db.historical_candidate_for_source(
            disposition.source, materializable_only=False
        )
        if candidate is None:
            return None
        permit = None if current_snapshot.snapshot_id <= candidate.disposition_snapshot_id else self.observation_owner.promotion_permit_from_retained(
            source=disposition.source,
            disposition_snapshot_id=int(candidate.disposition_snapshot_id),
            semantic_action=str(candidate.semantic_action),
            current_snapshot=current_snapshot,
        )
        if permit is None:
            return None
        return PreparedColdPromotionContext(
            observation=tracking,
            disposition=disposition,
            control_token=token,
            promotion_permit=permit,
        )

    @staticmethod
    def _prepared_cold_pagein_capable(record: Any | None) -> bool:
        """Whether durable history carries immutable prepared page-in authority."""

        return bool(
            record is not None
            and (
                record.semantic_action == SemanticAction.DEFER.value
                and getattr(record, "prepared_defer_context", None) is not None
                or record.semantic_action == SemanticAction.COMMIT.value
                and (
                    getattr(record, "prepared_commit_context", None) is not None
                    or getattr(record, "prepared_commit_obligation", None)
                    is not None
                )
            )
        )

    def prepared_cold_promotion_shortlist(self):
        """Return bounded DEFER and retained-COMMIT COLD page-in candidates."""

        candidate_index = getattr(self, "_historical_candidate_index", None)
        snapshot = self._prepared_cold_promotion_snapshot()
        if candidate_index is None or snapshot is None:
            return ()
        pending = tuple(getattr(self, "_pending_replay_trial_sources", ()))
        active_work = (
            getattr(self, "_prepared_ready_candidate_work", None)
            or getattr(self, "_prepared_ready_frozen_work", None)
        )
        plan = None if active_work is None else active_work[1]
        frame_ids_by_source = getattr(self, "_historical_frame_ids_by_source", {})
        active_sources = (
            tuple(getattr(plan, "supervision_sources", ()))
            + tuple(getattr(plan, "replay_sources", ()))
        )
        active_sources = tuple(dict.fromkeys(active_sources))
        active_ids = tuple(
            frame_ids_by_source[source]
            for source in active_sources
            if source in frame_ids_by_source
        )
        lane_size = getattr(
            getattr(candidate_index, "config", None), "recent_lane_size", len(active_ids)
        )
        local_ids = active_ids[:lane_size]
        pending_sources = tuple(
            source for source in dict.fromkeys(pending) if source not in active_sources
        )
        pending_ids = tuple(
            frame_ids_by_source[source]
            for source in pending_sources
            if source in frame_ids_by_source
        )
        shortlist = candidate_index.shortlist(
            snapshot_id=snapshot.snapshot_id,
            epoch=int(snapshot.control_version),
            local_ids=local_ids,
        )
        if local_ids or pending_ids:
            requested = frozenset(local_ids)
            shortlist = tuple(
                candidate for candidate in shortlist if candidate.frame_id in requested
            )
            if pending_sources:
                from eligsir.mapping.scheduler.historical_candidate_index import (
                    HistoricalCandidateEvidence,
                    HistoricalReplayCandidate,
                )

                present = {candidate.frame_id for candidate in shortlist}
                for source in pending_sources:
                    frame_id = frame_ids_by_source.get(source)
                    if frame_id is None or frame_id in present:
                        continue
                    try:
                        metadata = candidate_index.metadata_for(frame_id)
                    except (AttributeError, KeyError):
                        continue
                    shortlist += (
                        HistoricalReplayCandidate(
                            frame_id=frame_id,
                            weight=float(metadata.base_weight),
                            evidence=HistoricalCandidateEvidence(
                                lanes=("pending_trial",),
                                region_id=metadata.region_id,
                                focus_weight=0.0,
                                snapshot_id=snapshot.snapshot_id,
                                epoch=int(snapshot.control_version),
                            ),
                        ),
                    )
                    present.add(frame_id)
        selected = []
        for candidate in shortlist:
            record = self._observation_db.get_frame(int(candidate.frame_id))
            if record is None:
                continue
            if self._prepared_cold_pagein_capable(record):
                selected.append(candidate)
        return tuple(selected[: len(local_ids) + len(pending_ids)]) if pending else tuple(selected)

    def rearm_post_mapping_work(
        self,
        *,
        optimizer_updates: int | None = None,
        refinement_progress_fraction: float | None = None,
    ) -> None:
        """Prepare fresh weighted replay-only work without replaying admission."""

        phase_owner = getattr(self, "_post_refinement_phase_owner", None)
        if phase_owner is None or phase_owner.phase is not PostRefinementPhase.REFINEMENT:
            raise RuntimeError(
                "post-mapping refinement requires a durable post-refinement phase"
            )
        if refinement_progress_fraction is not None and (
            isinstance(refinement_progress_fraction, bool)
            or not math.isfinite(float(refinement_progress_fraction))
        ):
            raise TypeError(
                "refinement_progress_fraction must be None or a finite number"
            )
        # Read by ``_idle_replay_plan`` (via ``self``) on the very next tick
        # this rearm enables -- the "progressive" refinement-resolution
        # policy's only source of run progress. See issue #1346 follow-up.
        self._refinement_progress_fraction = (
            None
            if refinement_progress_fraction is None
            else float(refinement_progress_fraction)
        )
        retained_work = getattr(self, "_last_materialized_work", None)
        if retained_work is None:
            retained = self._last_admitted_control
            if retained is None:
                raise RuntimeError(
                    "post-mapping refinement has no admitted work to re-arm"
                )
            previous, _proposals, _disposition, previous_plan = retained
        else:
            previous, previous_plan = retained_work
        content, topology, parameters = self._versions()
        if hasattr(self, "_supervision_views_completed"):
            supervision_views_completed = self._supervision_views_completed
        elif hasattr(self, "config"):
            raise RuntimeError(
                "initialized runtime is missing its supervision work clock"
            )
        else:
            # ``__new__``-constructed CPU test seams predate the runtime-owned
            # counter. Their retained immutable snapshot is authoritative.
            supervision_views_completed = previous.supervision_views_completed
        snapshot = replace(
            previous,
            snapshot_id=self._control_snapshot_id,
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            optimizer_steps_completed=self.optimizer_steps,
            supervision_views_completed=supervision_views_completed,
            control_tick_epoch=self._control_tick_epoch,
            source_release_rate_ratio=self._source_rate_control_ratio(),
        )
        self._control_snapshot_id += 1
        replay_plan = self._idle_replay_plan(snapshot, previous_plan)
        retained_updates = max(1, int(replay_plan.optimizer_updates))
        updates = (
            retained_updates
            if optimizer_updates is None
            else min(retained_updates, max(1, int(optimizer_updates)))
        )
        supervision_work = tuple(replay_plan.supervision_work)
        replay_work = tuple(replay_plan.replay_work)
        # In B=1, optimizer_updates is also the number of sequential views and
        # the final refinement tick may need a shorter work list.  In true-C
        # mode one optimizer update consumes the complete camera batch, so
        # trimming by update count would silently collapse C>1 to C=1.
        if (
            int(getattr(self, "_batch_size", 1)) == 1
            and updates < len(supervision_work) + len(replay_work)
        ):
            supervision_work = supervision_work[:updates]
            replay_work = replay_work[: max(0, updates - len(supervision_work))]
        replay_weights_by_source = dict(
            zip(
                replay_plan.replay_sources,
                replay_plan.replay_selected_weights,
                strict=bool(replay_plan.replay_selected_weights),
            )
        )
        replay_selected_weights = (
            tuple(replay_weights_by_source[item.source] for item in replay_work)
            if replay_plan.replay_selected_weights
            else ()
        )
        replay_insertion = replay_plan.replay_insertion
        if (
            replay_insertion is not None
            and replay_insertion.source not in {item.source for item in replay_work}
        ):
            replay_insertion = None
        plan = replace(
            replay_plan,
            plan_id=(
                f"refinement-{snapshot.snapshot_id}-m{content}-t{topology}-p{parameters}"
            ),
            snapshot_id=snapshot.snapshot_id,
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            optimizer_updates=updates,
            supervision_sources=tuple(item.source for item in supervision_work),
            replay_sources=tuple(item.source for item in replay_work),
            supervision_work=supervision_work,
            replay_work=replay_work,
            replay_selected_weights=replay_selected_weights,
            replay_insertion=replay_insertion,
        )
        self._latest_report_work = (snapshot, plan)
        self._latest_tick_work = (snapshot, plan)
        if (
            getattr(self, "_prepared_ready_frozen_work", None) is not None
            or getattr(self, "_prepared_ready_candidate_work", None) is not None
        ):
            self._prepared_ready_frozen_work = None
            self._prepared_ready_candidate_work = (snapshot, plan)

    def _retire_completed_candidates(self, successful_sources: set[Any]) -> None:
        pending = getattr(self, "_pending_supervision_modalities", None)
        if pending:
            for source in successful_sources:
                pending.pop(source, None)

    def publish_replay_focus_weights(self, weights) -> None:
        """Publish CPU-only per-identity focus evidence for the next plan.

        Missing identities keep neutral weight 1.0. A zero requested weight is
        still sampled at the provider's configured global floor, so regional
        focus cannot erase trajectory-wide replay coverage.
        """

        from collections.abc import Mapping
        from eligsir.mapping.scheduler.historical_candidate_index import (
            MAX_REPLAY_IDENTITY_FOCUS_UPDATES,
        )

        if not isinstance(weights, Mapping):
            raise TypeError("replay focus publication must be a bounded mapping")
        if len(weights) > MAX_REPLAY_IDENTITY_FOCUS_UPDATES:
            raise ValueError(
                "replay focus publication accepts at most "
                f"{MAX_REPLAY_IDENTITY_FOCUS_UPDATES} identities"
            )
        updates = tuple(weights.items())
        eligible = getattr(self, "_replay_modalities", ())
        unknown = tuple(source for source, _weight in updates if source not in eligible)
        if unknown:
            raise ValueError("replay focus weights contain identities outside the cache")
        validated = {}
        for source, weight in updates:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not np.isfinite(float(weight))
                or float(weight) < 0.0
            ):
                raise ValueError("replay focus weights must be finite and non-negative")
            validated[source] = float(weight)
        replay_weights = getattr(self, "_replay_weights", None)
        if replay_weights is None:
            replay_weights = self._replay_weights = {}
        for source in getattr(self, "_replay_focus_sources", ()):
            if source in replay_weights:
                replay_weights[source] = 1.0
        for source, weight in validated.items():
            replay_weights[source] = weight
        self._replay_focus_sources = tuple(validated)
        frame_id_by_source = getattr(self, "_historical_frame_ids_by_source", {})
        ranked_focus = sorted(
            (
                (-weight, source.ordering_key(), frame_id_by_source[source], weight)
                for source, weight in validated.items()
                if weight > 1.0 and source in frame_id_by_source
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )[:32]
        self._replay_focus_frame_weights = tuple(
            # HistoricalCandidateIndex stores a focus *boost* over the
            # metadata base weight, while runtime replay weights are absolute.
            (frame_id, weight - 1.0)
            for _negative_weight, _source_key, frame_id, weight in ranked_focus
        )

    def publish_growth_quality_evidence(
        self,
        *,
        iteration: int,
        aggregate_psnr_db: float,
        gaussian_count: int,
        worst_drop_from_peak_db: float,
    ) -> None:
        """Publish training-only spatial evidence to the growth proposer."""

        provider = getattr(self, "_adaptive_growth_provider", None)
        if provider is None:
            raise RuntimeError("adaptive growth proposal provider is unavailable")
        provider.observe_training_spatial_quality(
            iteration=iteration,
            aggregate_psnr_db=aggregate_psnr_db,
            gaussian_count=gaussian_count,
            worst_drop_from_peak_db=worst_drop_from_peak_db,
        )
        partition_fade = bool(
            float(worst_drop_from_peak_db)
            > float(self._adaptive_quality_risk_tolerance_db)
        )
        self._adaptive_supervision_quality_evidence = (
            partition_fade,
            partition_fade,
        )

    def replay_candidate_identities(self) -> tuple[Any, ...]:
        """Return the immutable CPU identity set eligible for replay weighting."""

        return tuple(getattr(self, "_replay_modalities", ()))

    def covered_cell_keys_for_sources(
        self, sources: "Iterable[Any]"
    ) -> dict[Any, tuple[int, ...]]:
        """Geometry-only world-microcell coverage for arbitrary sources (#1723).

        Cell *membership* only needs a camera's fixed pose and sensor depth
        (see ``spatial_microcell_quality_from_render``'s docstring) -- never a
        render or the current map -- so this can answer for a cached/evicted
        source with no recent training pass, which is exactly the gap
        ``SpatialMicrocellReplayFocusController.active_source_cell_keys``
        cannot fill on its own.

        Deliberately uses ``self.scene.admitted_camera_for_source`` directly
        rather than ``self._camera_for_source`` -- the latter's
        never-admitted/deferred fallback path does DB lookups and can raise
        without a ``current_snapshot``, which is unnecessary risk and cost for
        a best-effort, read-only lookup. A source with no admitted camera (or
        no resolvable depth) is silently omitted from the result rather than
        raising: callers must already treat "no region_ids for this source"
        as "not a promote/demote candidate this tick", not an error.
        """
        tracker = getattr(self, "_spatial_quality_tracker", None)
        scene = getattr(self, "scene", None)
        if tracker is None or scene is None:
            return {}
        admitted_lookup = getattr(scene, "admitted_camera_for_source", None)
        if not callable(admitted_lookup):
            return {}
        from eligsir.mapping.scheduler.partition_replay_focus import (
            world_cell_incidence_from_pose_depth,
        )
        from eligsir.utils.comparison_report import _get_sensor_depth

        result: dict[Any, tuple[int, ...]] = {}
        for source in sources:
            camera = admitted_lookup(source)
            if camera is None:
                continue
            height = getattr(camera, "image_height", None)
            width = getattr(camera, "image_width", None)
            if not height or not width:
                continue
            try:
                sensor_depth = _get_sensor_depth(camera, int(height), int(width))
            except Exception:
                continue
            sample = world_cell_incidence_from_pose_depth(
                source=source,
                tracker=tracker,
                camera=camera,
                sensor_depth=sensor_depth,
            )
            if sample is None:
                continue
            result[source] = sample.cell_keys
        return result

    def replay_catalog_population(self) -> int:
        """Return the O(1) retained COMMIT+DEFER identity population."""

        candidate_index = getattr(self, "_historical_candidate_index", None)
        if candidate_index is not None:
            return int(candidate_index.history_size)
        return len(getattr(self, "_replay_modalities", ()))

    def replay_control_candidate_identities(self) -> tuple[Any, ...]:
        """Expose one bounded scheduler shortlist to low-cadence control."""

        modalities, _reserved_source = self._shortlisted_replay_view(
            snapshot_id=int(getattr(self, "_control_snapshot_id", 0)),
            control_tick_epoch=int(getattr(self, "_control_tick_epoch", 0)),
        )
        candidates = tuple(modalities)
        if len(candidates) > 128:
            raise RuntimeError("replay control shortlist exceeded its fixed bound")
        return candidates

    def source_identity_for_fingerprint(self, fingerprint: str) -> Any | None:
        """Resolve telemetry identity only within the admitted replay cache."""

        matches = tuple(
            source
            for source in getattr(self, "_replay_modalities", ())
            if source.fingerprint() == str(fingerprint)
        )
        if len(matches) > 1:
            raise RuntimeError("replay cache contains duplicate source fingerprints")
        return matches[0] if matches else None

    def mark_source_complete(self) -> MappingTickResult | None:
        """Record finite-source EOF; endpoint publication gates refinement."""

        terminal = getattr(self, "_source_complete_tick_result", None)
        if terminal is not None:
            return terminal
        self._source_complete = True
        phase_owner = getattr(self, "_post_refinement_phase_owner", None)
        if phase_owner is None:
            if hasattr(self, "config"):
                raise RuntimeError("canonical runtime is missing post-refinement phase ownership")
        else:
            phase_owner.mark_source_complete()
        work_clock_owner = self._work_clock_owner_if_available()
        if work_clock_owner is not None:
            work_clock_owner.mark_source_complete()
        if int(getattr(self, "_replay_depth_max_per_event", 0)) > 0:
            self._next_replay_depth_supervision_view = (
                int(getattr(self, "_replay_supervision_views_completed", 0))
                + max(1, int(self._replay_depth_interval))
            )
        self._next_spatial_quality_replay_view = (
            int(getattr(self, "_replay_supervision_views_completed", 0))
            + max(1, int(getattr(self, "_spatial_replay_focus_interval", 25)))
        )
        if work_clock_owner is None:
            return None
        previous = getattr(self, "_latest_tick_result", None)
        terminal = self._tick_result(
            False,
            "source_complete",
            snapshot=(None if previous is None else previous.control_snapshot),
            plan=(None if previous is None else previous.work_plan),
        )
        self._source_complete_tick_result = terminal
        return terminal

    def enter_post_mapping_refinement(self, mapping_endpoint: Mapping[str, object]) -> None:
        """Enter the typed refinement phase after durable endpoint publication."""

        phase_owner = getattr(self, "_post_refinement_phase_owner", None)
        if phase_owner is None:
            raise RuntimeError("post-refinement phase ownership is unavailable")
        phase_owner.enter_post_refinement(mapping_endpoint)

    @contextmanager
    def post_mapping_refinement_loss_context(self):
        """Apply the optional depth-loss override only while a tail tick runs."""

        override = self._refinement_depth_loss_override()
        if override is None:
            yield
            return
        kwargs = getattr(self._optimization_runtime, "loss_assembly_kwargs", None)
        if not isinstance(kwargs, dict):
            raise RuntimeError(
                "post-mapping depth-loss override requires loss assembly kwargs"
            )
        previous = kwargs["depth_loss_weight"]
        previous_policy = self._depth_loss_policy
        kwargs["depth_loss_weight"] = override
        self._depth_loss_policy = replace(
            previous_policy, enabled=override > 0.0, weight=override
        )
        try:
            yield
        finally:
            kwargs["depth_loss_weight"] = previous
            self._depth_loss_policy = previous_policy

    def _refinement_depth_loss_override(self) -> float | None:
        """Return the finite tail-only override, or ``None`` for the -1 sentinel."""

        override = float(
            getattr(
                getattr(self, "_optimization_policy", None),
                "streaming_refinement_depth_loss_weight",
                -1.0,
            )
        )
        if not math.isfinite(override) or (override < 0.0 and override != -1.0):
            raise ValueError(
                "refinement depth-loss weight must be -1 or a finite non-negative value"
            )
        return None if override == -1.0 else override

    def _capture_accepted_map_branch_state(self) -> dict[str, Any]:
        """Capture the numerical state that defines an accepted refinement map."""

        session = getattr(getattr(self, "_engine", None), "session", None)
        capture_session = getattr(session, "capture_branch_state", None)
        if not callable(capture_session):
            raise RuntimeError("accepted-map branch requires optimizer-session state")
        work_clock = self._work_clock_owner_if_available()
        if work_clock is None:
            raise RuntimeError("accepted-map branch requires semantic work clocks")
        return {
            "iteration": int(self.iteration),
            "optimizer_steps": int(self.optimizer_steps),
            "supervision_views_completed": int(self._supervision_views_completed),
            "latest_admitted_optimizer_update": int(
                self._latest_admitted_optimizer_update
            ),
            "control_tick_epoch": int(self._control_tick_epoch),
            "sh_degree_thresholds_consumed": int(
                self._sh_degree_thresholds_consumed
            ),
            "next_lifecycle_optimizer_update": int(
                self._next_lifecycle_optimizer_update
            ),
            "optimizer_session": capture_session(),
            "work_clock": work_clock.snapshot(),
        }

    def _restore_accepted_map_branch_state(
        self, state: Mapping[str, Any], *, live: bool = False
    ) -> None:
        """Restore only map-accepted numerical state after a rejected chunk."""

        required = {
            "iteration",
            "optimizer_steps",
            "supervision_views_completed",
            "latest_admitted_optimizer_update",
            "control_tick_epoch",
            "sh_degree_thresholds_consumed",
            "next_lifecycle_optimizer_update",
            "optimizer_session",
            "work_clock",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("accepted-map branch state layout mismatch")
        values = {
            name: state[name]
            for name in (
                "iteration",
                "optimizer_steps",
                "supervision_views_completed",
                "latest_admitted_optimizer_update",
                "control_tick_epoch",
                "next_lifecycle_optimizer_update",
            )
        }
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise TypeError("accepted-map branch counters must be non-negative integers")
        thresholds = state["sh_degree_thresholds_consumed"]
        if type(thresholds) is not int or thresholds < 0:
            raise TypeError("accepted-map branch SH threshold count must be a non-negative integer")
        session = getattr(getattr(self, "_engine", None), "session", None)
        restore_session = getattr(session, "restore_branch_state", None)
        if not callable(restore_session):
            raise RuntimeError("accepted-map branch requires optimizer-session restore")
        work_clock = self._work_clock_owner_if_available()
        restore_clock = getattr(work_clock, "restore_branch_state", None)
        if not callable(restore_clock):
            raise RuntimeError("accepted-map branch requires semantic work-clock restore")
        self.iteration = values["iteration"]
        self.optimizer_steps = values["optimizer_steps"]
        self._supervision_views_completed = values["supervision_views_completed"]
        self._latest_admitted_optimizer_update = values[
            "latest_admitted_optimizer_update"
        ]
        self._control_tick_epoch = values["control_tick_epoch"]
        self._sh_degree_thresholds_consumed = thresholds
        self._next_lifecycle_optimizer_update = values[
            "next_lifecycle_optimizer_update"
        ]
        if live:
            restore_live_session = getattr(session, "restore_live_branch_state", None)
            if not callable(restore_live_session):
                raise RuntimeError("accepted-map branch requires live optimizer-session restore")
            restore_live_session(state["optimizer_session"])
        else:
            restore_session(state["optimizer_session"])
        restore_clock(state["work_clock"])

    @staticmethod
    def _reject_post_refinement_branch_ingest(_observation: Any) -> None:
        raise RuntimeError("post-source refinement branch cannot ingest observations")

    def refinement_branch_runtime(self, *, evaluate_fn: Any):
        """Expose a bounded accepted-map branch, not full async continuation.

        READY residency, WorkBudget, promotion, and telemetry remain
        attempt-monotonic.  The branch restores the map/optimizer numerical
        boundary only; callers must publish external control effects solely
        after acceptance.
        """

        if not bool(getattr(self, "_source_complete", False)):
            raise RuntimeError("accepted-map branch requires source completion")
        if not callable(evaluate_fn):
            raise TypeError("accepted-map branch requires a fixed-view evaluator")
        training_args = self._backend_optimizer_setup
        if training_args is None:
            raise RuntimeError("accepted-map branch requires optimizer setup")
        from eligsir.mapping.branching import (
            BranchableMapperRuntime,
            CallableStateProvider,
        )

        return BranchableMapperRuntime(
            backend=self.backend,
            training_args=training_args,
            ingest_fn=self._reject_post_refinement_branch_ingest,
            evaluate_fn=evaluate_fn,
            state_providers=(
                CallableStateProvider(
                    provider_name="accepted_map_numerical_state",
                    capture_fn=self._capture_accepted_map_branch_state,
                    restore_fn=self._restore_accepted_map_branch_state,
                    restore_live_fn=lambda state: self._restore_accepted_map_branch_state(
                        state, live=True
                    ),
                ),
            ),
            evaluation_id="causal_refinement_control_probe",
            metadata={
                "scope": "accepted_map_refinement",
                "attempt_monotonic": "ready_work_budget_promotion_telemetry",
            },
        )

    def _replay_depth_due(self) -> bool:
        return bool(
            getattr(self, "_source_complete", False)
            and int(getattr(self, "_replay_depth_max_per_event", 0)) > 0
            and int(getattr(self, "_replay_supervision_views_completed", 0))
            >= int(getattr(self, "_next_replay_depth_supervision_view", 0))
        )

    def _record_successful_replay_supervision(
        self,
        plan: MappingWorkPlan,
        *,
        successful_sources: set[Any],
        unique_sources_known: bool,
    ) -> None:
        """Advance replay-depth cadence only from attributed successful views."""

        if not unique_sources_known:
            return
        replay_sources = set(plan.replay_sources)
        completed = sum(source in replay_sources for source in successful_sources)
        self._replay_supervision_views_completed = int(
            getattr(self, "_replay_supervision_views_completed", 0)
        ) + int(completed)

    def _consume_replay_depth_deadline(self) -> None:
        """Arm one future replay-view deadline without catch-up bursts."""

        interval = max(1, int(self._replay_depth_interval))
        self._next_replay_depth_supervision_view = (
            int(getattr(self, "_replay_supervision_views_completed", 0))
            + interval
        )

    def _primary_psnr_due(self) -> bool:
        return bool(
            int(getattr(self, "optimizer_steps", 0))
            >= int(getattr(self, "_next_primary_psnr_optimizer_update", 0))
        )

    def _consume_primary_psnr_deadline(self) -> None:
        self._next_primary_psnr_optimizer_update = (
            int(getattr(self, "optimizer_steps", 0))
            + max(1, int(self._primary_psnr_interval))
        )

    def _spatial_quality_due(self) -> bool:
        return bool(
            getattr(self, "_spatial_replay_focus_enabled", False)
            and int(getattr(self, "_replay_supervision_views_completed", 0))
            >= int(getattr(self, "_next_spatial_quality_replay_view", 0))
        )

    def _consume_spatial_quality_deadline(self) -> None:
        self._next_spatial_quality_replay_view = (
            int(getattr(self, "_replay_supervision_views_completed", 0))
            + max(1, int(self._spatial_replay_focus_interval))
        )

    @staticmethod
    def _summary_cpu_values(summary: Any) -> dict[str, Any]:
        """Serialize only compact CPU copies already produced by the bridge."""

        packed = getattr(summary, "compact_packed_readback", None)
        if packed is not None:
            try:
                payload = packed.try_take_cpu_tensor()
            except (RuntimeError, TypeError, ValueError, AttributeError):
                payload = None
            try:
                payload_valid = (
                    payload is not None
                    and getattr(getattr(payload, "device", None), "type", None) == "cpu"
                    and payload.ndim == 1
                    and payload.numel() >= 9
                )
            except (AttributeError, TypeError, ValueError):
                payload_valid = False
            if not payload_valid:
                return {
                    "transaction_id": getattr(summary, "transaction_id", None),
                    "epoch": getattr(summary, "epoch", None),
                    "units": getattr(summary, "units", None),
                    "decision_token": getattr(summary, "decision_token", None),
                    "drop_reason": getattr(
                        getattr(summary, "drop_reason", None), "value", None
                    ),
                    "native_target_count": getattr(summary, "native_target_count", None),
                    "native_target_status": None,
                    "selected_action_histogram": None,
                    "adapter_plan_drop_reason": getattr(summary, "adapter_plan_drop_reason", None),
                    "adapter_rejected": getattr(summary, "adapter_rejected", None),
                }
            return {
                "transaction_id": getattr(summary, "transaction_id", None),
                "epoch": getattr(summary, "epoch", None),
                "units": getattr(summary, "units", None),
                "decision_token": getattr(summary, "decision_token", None),
                "drop_reason": getattr(getattr(summary, "drop_reason", None), "value", None),
                "native_target_count": getattr(summary, "native_target_count", None),
                "native_target_status": [int(payload[7])],
                "selected_action_histogram": payload[:7].tolist(),
                "adapter_plan_drop_reason": getattr(summary, "adapter_plan_drop_reason", None),
                "adapter_rejected": getattr(summary, "adapter_rejected", None),
            }

        def compact(value: Any) -> Any:
            device = getattr(value, "device", None)
            if value is None or device is None or getattr(device, "type", None) != "cpu":
                return None
            # The bridge performs the only device-to-host transfer.  This
            # consumer reads its bounded CPU result; it never touches a
            # borrowed decision/target tensor.
            return value.tolist()

        return {
            "transaction_id": getattr(summary, "transaction_id", None),
            "epoch": getattr(summary, "epoch", None),
            "units": getattr(summary, "units", None),
            "decision_token": getattr(summary, "decision_token", None),
            "drop_reason": getattr(
                getattr(summary, "drop_reason", None), "value", None
            ),
            "native_target_count": (
                None
                if getattr(summary, "native_target_count", 0) is None
                else int(getattr(summary, "native_target_count", 0))
            ),
            "native_target_status": compact(
                getattr(summary, "native_target_status_cpu", None)
            ),
            "selected_action_histogram": compact(
                getattr(summary, "selected_action_histogram_cpu", None)
            ),
            "adapter_plan_drop_reason": getattr(
                summary, "adapter_plan_drop_reason", None
            ),
            "adapter_rejected": getattr(summary, "adapter_rejected", None),
        }

    def _drain_population_selection_telemetry(self) -> tuple[dict[str, Any], ...]:
        """Drain the bridge's bounded async summaries at the trace boundary."""

        if not bool(getattr(self, "_population_selection_trace_enabled", False)):
            return ()

        runtime = getattr(self, "_optimization_runtime", None)
        bridge = getattr(runtime, "population_shadow_runtime", None)
        drain = getattr(bridge, "drain_async_summaries", None)
        if not callable(drain):
            return ()
        pending_summaries = getattr(
            self, "_population_selection_pending_summaries", None
        )
        if pending_summaries is None:
            pending_summaries = self._population_selection_pending_summaries = deque()
        incoming = tuple(pending_summaries)
        pending_summaries.clear()
        records = []
        for summary in incoming:
            packed = getattr(summary, "compact_packed_readback", None)
            if packed is not None:
                try:
                    ready = packed.event.query()
                except (RuntimeError, TypeError, AttributeError):
                    ready = False
                if not ready:
                    pending_summaries.append(summary)
                    continue
            records.append(self._summary_cpu_values(summary))

        # Apply capacity before submitting another optional packed readback.
        # A pending in-flight slab therefore cannot be evicted into cleanup.
        if pending_summaries:
            return tuple(records)
        incoming = tuple(drain())
        for summary in incoming:
            packed = getattr(summary, "compact_packed_readback", None)
            if packed is not None:
                try:
                    if not packed.event.query():
                        pending_summaries.append(summary)
                        continue
                except (RuntimeError, TypeError, AttributeError):
                    pending_summaries.append(summary)
                    continue
            records.append(self._summary_cpu_values(summary))
        return tuple(records)

    def record_successful_supervision_sources(self, sources: Iterable[Any]) -> None:
        """Accumulate completed source uses for post-run causal reporting.

        Keyed by the full ``SourceIdentity`` (issue #1688), not just
        ``source_ordinal``: a reconnected stream epoch or another source
        namespace can legally reuse the same ordinal, and a ledger keyed by
        ordinal alone would silently merge those physically distinct sources.
        """

        uses = getattr(self, "_successful_supervision_uses_by_identity", None)
        if uses is None:
            uses = self._successful_supervision_uses_by_identity = {}
        for source in sources:
            if getattr(source, "source_ordinal", None) is None:
                continue
            uses[source] = int(uses.get(source, 0)) + 1

    def successful_supervision_uses_by_identity(self) -> dict[SourceIdentity, int]:
        """Return the canonical, collision-free supervision-use ledger."""

        return dict(getattr(self, "_successful_supervision_uses_by_identity", {}))

    def successful_supervision_uses_by_ordinal(self) -> dict[int, int]:
        """Return ordinal-aggregated reporting counts (chart/compat view only).

        Sums use counts across any distinct identities that share an ordinal.
        Report-camera *selection* must not use this aggregate to decide which
        physical record was supervised; use
        :meth:`successful_supervision_uses_by_identity` for that.
        """

        totals: dict[int, int] = {}
        for identity, count in self.successful_supervision_uses_by_identity().items():
            ordinal = int(identity.source_ordinal)
            totals[ordinal] = totals.get(ordinal, 0) + count
        return totals

    def _tick_result(
        self,
        executed: bool,
        reason: str,
        *,
        snapshot: ControlSnapshot | None,
        plan: MappingWorkPlan | None,
        optimizer_steps: int = 0,
        supervision_views: int = 0,
        supervised_pixels: int = 0,
        unique_supervision_views: int | None = 0,
        successful_supervision_sources: tuple[Any, ...] = (),
        microbatch_sizes: tuple[int, ...] = (),
        insertion_candidates: int = 0,
        inserted: int = 0,
        pruned: int = 0,
        relocated: int = 0,
        corrected: int = 0,
        mutation_events: tuple[MutationAttribution, ...] = (),
        primary_training_psnr_db: float | None = None,
        primary_training_source_id: str | None = None,
        primary_training_spatial: Any = None,
        primary_training_gaussian_count: int | None = None,
        camera_materialize_cache_hits: int = 0,
        camera_materialize_cache_misses: int = 0,
        prepared_insertion_skip: str | None = None,
    ) -> MappingTickResult:
        self._poll_gpu_telemetry()
        population_policy_selection_telemetry = self._drain_population_selection_telemetry()
        dispositions = tuple(self._dispositions_since_tick)
        self._dispositions_since_tick.clear()
        trace_enabled = bool(getattr(self, "_population_selection_trace_enabled", False))
        if trace_enabled:
            replay_challenger_trace = tuple(
                getattr(self, "_replay_challenger_trace_events", ())
            )
            replay_challenger_trace_truncated = bool(
                getattr(self, "_replay_challenger_trace_truncated", False)
            )
            self._replay_challenger_trace_events = []
            self._replay_challenger_trace_truncated = False
        else:
            replay_challenger_trace = ()
            replay_challenger_trace_truncated = False
        content, topology, parameters = self._versions()
        work_clocks = self._work_clock_owner.snapshot()
        result = MappingTickResult(
            executed=executed,
            iteration=self.iteration,
            work_clocks=work_clocks,
            reason=reason,
            optimizer_steps=optimizer_steps,
            supervision_views=supervision_views,
            supervised_pixels=supervised_pixels,
            unique_supervision_views=unique_supervision_views,
            successful_supervision_sources=tuple(successful_supervision_sources),
            microbatch_sizes=tuple(microbatch_sizes),
            cumulative_optimizer_updates=self.optimizer_steps,
            cumulative_supervision_views=self._supervision_views_completed,
            snapshot_id=None if snapshot is None else snapshot.snapshot_id,
            control_snapshot=snapshot,
            dispositions=dispositions,
            retained_catalog_population_post_dispositions=len(
                getattr(self, "_replay_modalities", ())
            ),
            work_plan=plan,
            freshness=FreshnessMode.HEALTHY if plan is None else plan.freshness_mode,
            transfers=(
                None if self._transfer_owner is None else self._transfer_owner.accounting()
            ),
            map_content_version=content,
            topology_version=topology,
            parameter_version=parameters,
            insertion_candidates=insertion_candidates,
            insertion_filter=(
                getattr(self, "_decision_trace_insertion_filter", None)
                if bool(getattr(self, "_population_selection_trace_enabled", False))
                else None
            ),
            inserted=inserted,
            pruned=pruned,
            relocated=relocated,
            corrected=corrected,
            structural_mutation=bool(inserted + pruned + relocated + corrected),
            mutation_events=tuple(mutation_events),
            primary_training_psnr_db=primary_training_psnr_db,
            primary_training_source_id=primary_training_source_id,
            primary_training_spatial=primary_training_spatial,
            primary_training_gaussian_count=primary_training_gaussian_count,
            camera_materialize_cache_hits=max(0, int(camera_materialize_cache_hits)),
            camera_materialize_cache_misses=max(0, int(camera_materialize_cache_misses)),
            prepared_insertion_skip=prepared_insertion_skip,
            adaptive_growth_enabled=bool(
                getattr(getattr(self, "_adaptive_growth_provider", None), "_enabled", False)
            ),
            population_policy_disposition=(
                str(
                    getattr(
                        getattr(self, "_optimization_runtime", None),
                        "last_population_shadow_drop_reason",
                        None,
                    ).value
                )
                if getattr(
                    getattr(self, "_optimization_runtime", None),
                    "last_population_shadow_drop_reason",
                    None,
                )
                is not None
                else (
                    getattr(self, "_population_live_activation_disposition", None)
                    or (
                    "structural_action"
                    if any(
                        event.kind == "population_policy"
                        and int(event.inserted) + int(event.pruned)
                        + int(event.relocated) + int(event.corrected) > 0
                        for event in mutation_events
                    )
                    else (
                        "no_structural_action"
                        if self._population_live_requested()
                        else "not_requested"
                    )
                    )
                )
            ),
            population_stale_current_mismatches=tuple(
                getattr(
                    getattr(self, "_optimization_runtime", None),
                    "last_population_shadow_stale_current_mismatches",
                    (),
                )
                or ()
            ),
            population_policy_selection_telemetry=population_policy_selection_telemetry,
            native_geometry_evidence_telemetry=self.native_geometry_evidence_telemetry(),
            replay_challenger_trace=replay_challenger_trace,
            replay_challenger_trace_truncated=replay_challenger_trace_truncated,
        )
        inputs = self._telemetry_inputs(snapshot)
        gpu_transaction = self._latest_gpu_transaction
        if (
            gpu_transaction is not None
            and (
                (snapshot is not None and gpu_transaction.snapshot_id > snapshot.snapshot_id)
                or (
                    gpu_transaction.map_content_version != content
                    or gpu_transaction.topology_version != topology
                    or gpu_transaction.parameter_version != parameters
                )
            )
        ):
            gpu_transaction = None
        telemetry = self._telemetry.publish_tick(
            result,
            inputs,
            gpu_transaction=gpu_transaction,
        )
        if not isinstance(telemetry.map_state, MapStateTelemetry):
            raise RuntimeError("telemetry adapter published invalid canonical map state")
        completed = replace(result, telemetry=telemetry)
        self._latest_tick_result = completed
        return completed

    def _poll_gpu_telemetry(self) -> None:
        """Publish only transactions whose HIP/CUDA end event has completed."""

        # Controller-critical optimization timings use their own ordered lane.
        # The generic telemetry publisher is latest-wins and may coalesce staged
        # transactions; freshness control must consume every optimization
        # completion exactly once.
        while self._pending_optimization_transaction_timings:
            (
                start_event,
                end_event,
                descriptor,
                host_prefix_ns,
                inserted,
                pruned,
                relocated,
                corrected,
            ) = self._pending_optimization_transaction_timings[0]
            if not bool(end_event.query()):
                break
            self._pending_optimization_transaction_timings.popleft()
            duration_ms = float(start_event.elapsed_time(end_event))
            if not np.isfinite(duration_ms) or duration_ms < 0.0:
                self._invalid_optimization_gpu_timings = int(
                    getattr(self, "_invalid_optimization_gpu_timings", 0)
                ) + 1
                continue
            completed_optimization = GpuTransactionTelemetry(
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
                inserted=inserted,
                pruned=pruned,
                relocated=relocated,
                corrected=corrected,
            )
            self._publish_completed_optimization_timing(
                completed_optimization,
                host_prefix_ns=host_prefix_ns,
            )
            self._latest_gpu_transaction = completed_optimization

        while True:
            completed = self._gpu_event_telemetry.poll_completed()
            if completed is None:
                return
            self._latest_gpu_transaction = completed
            if completed.transaction_type is TransactionType.OPTIMIZATION:
                self._latest_optimization_gpu_transaction = completed
                self._latest_optimization_gpu_ms = float(completed.duration_ms)

    def _publish_completed_optimization_timing(
        self,
        completed: GpuTransactionTelemetry,
        *,
        host_prefix_ns: int,
    ) -> None:
        self._latest_optimization_gpu_transaction = completed
        self._latest_optimization_gpu_ms = float(completed.duration_ms)
        gpu_envelope_ns = int(round(completed.duration_ms * 1_000_000.0))
        compact = completed.compact()
        self._latest_completed_optimization_transaction = CompactGpuTelemetry(
            transaction_id=compact.transaction_id,
            map_content_version=compact.map_content_version,
            topology_version=compact.topology_version,
            parameter_version=compact.parameter_version,
            facts=compact.facts
            + (
                EvidenceFact(
                    "optimization_transaction_duration_ns",
                    host_prefix_ns + gpu_envelope_ns,
                ),
            ),
        )

    @staticmethod
    def _begin_optimization_gpu_envelope(
        start_event: Any,
        *,
        transaction_started_ns: int,
    ) -> int:
        """Close the host prefix before recording the non-overlapping GPU span."""

        host_prefix_endpoint_ns = time.monotonic_ns()
        start_event.record()
        return max(1, host_prefix_endpoint_ns - transaction_started_ns)

    def _submit_gpu_timing(
        self,
        start_event: Any,
        end_event: Any,
        descriptor: GpuTelemetryDescriptor,
        *,
        inserted: int = 0,
        pruned: int = 0,
        relocated: int = 0,
        corrected: int = 0,
        optimization_host_prefix_ns: int | None = None,
    ) -> None:
        if descriptor.transaction_type is TransactionType.OPTIMIZATION:
            if (
                isinstance(optimization_host_prefix_ns, bool)
                or not isinstance(optimization_host_prefix_ns, int)
                or optimization_host_prefix_ns < 1
            ):
                raise ValueError(
                    "optimization timing requires positive host prefix duration"
                )
        elif optimization_host_prefix_ns is not None:
            raise ValueError(
                "host prefix duration is valid only for optimization telemetry"
            )
        if optimization_host_prefix_ns is not None:
            self._pending_optimization_transaction_timings.append(
                (
                    start_event,
                    end_event,
                    descriptor,
                    optimization_host_prefix_ns,
                    inserted,
                    pruned,
                    relocated,
                    corrected,
                )
            )
        else:
            self._gpu_event_telemetry.submit(
                start_event,
                end_event,
                descriptor,
                inserted=inserted,
                pruned=pruned,
                relocated=relocated,
                corrected=corrected,
            )

    @staticmethod
    def _latency_summary(values: deque[float]) -> LatencySummary:
        """Compute latency summary from historical values."""
        if not values:
            return LatencySummary(0, 0.0, 0.0, 0.0)
        samples = np.asarray(tuple(values), dtype=np.float64)
        return LatencySummary(
            len(samples),
            float(np.percentile(samples, 50)),
            float(np.percentile(samples, 95)),
            float(samples.max()),
        )

    def _telemetry_inputs(
        self, snapshot: ControlSnapshot | None
    ) -> RuntimeTelemetryInputs:
        """Construct telemetry inputs from current runtime state."""
        clocks = self._services.four_clocks.summary()
        active = self._observation_db.active_supervision_occupancy
        history = max(0, len(self._observation_db) - active)

        def queue(depth: int, oldest_age_ns: int = 0) -> QueueStateTelemetry:
            """Format queue telemetry from depth and age."""
            return QueueStateTelemetry(
                max(0, int(depth)),
                0.0 if int(depth) == 0 else max(0.0, float(oldest_age_ns) / 1e6),
            )

        return RuntimeTelemetryInputs(
            published_at_ns=time.monotonic_ns(),
            source_clock_ns=self._source_clock_ns,
            admitted_clock_ns=self._admitted_clock_ns,
            map_clock_ns=self._map_clock_ns,
            rates=RateTelemetry(
                clocks.source_frame_rate_hz,
                clocks.admitted_keyframe_rate_hz,
                clocks.map_commit_rate_hz,
            ),
            latencies=RuntimeLatencyTelemetry(
                self._latency_summary(self._first_supervision_latency_ms),
                self._latency_summary(self._geometry_commit_latency_ms),
            ),
            queues=QueueTelemetry(
                source=queue(0 if snapshot is None else snapshot.ingress_queue_depth,
                             0 if snapshot is None else snapshot.pending_oldest_age_ns),
                candidate=queue(
                    0 if snapshot is None else snapshot.candidate_queue_depth,
                    self._candidate_oldest_age_ns,
                ),
                commit=queue(
                    int(self._pending_integration is not None),
                    self._commit_oldest_age_ns,
                ),
                transfer=queue(0 if snapshot is None else snapshot.transfer_queue_depth),
                gpu=queue(0 if snapshot is None else snapshot.gpu_queue_depth),
            ),
            gaussian_count=0 if not self.initialized else int(self.gaussians.get_xyz.shape[0]),
            residency=ResidencyTelemetry(
                active=active,
                replay=0,
                history=history,
                hot_gpu=active,
            ),
        )

    def map_frame(self, frame, _observation) -> Any:
        """Record a materialized frame in the mapping."""
        if self._closed:
            raise RuntimeError("mapping frame after runtime close")
        local = StreamingObservation.from_mapping_frame(frame)
        work = self._active_materialization_work
        self._active_materialization_work = None
        self._last_materialized_work = work
        plan = None if work is None else work[1]
        if plan is None or frame.source not in plan.materialization_sources:
            raise RuntimeError("materialized source has no authoritative work plan")
        self.frames_seen += 1

        if not self.initialized:
            self._bootstrap.append(local)
            if len(self._bootstrap) < self.bootstrap_frames:
                return {"stage": "bootstrap", "ready": False}
            # Bootstrap is executed by tick under this plan's version fence;
            # submit never mutates the map behind a frozen plan.
            return {"stage": "bootstrap", "ready": False}

        assert self.scene is not None
        result = self.scene.ingest_canonical_frame(
            local, record_fn=self._record_scene_candidate
        )
        if result is None:
            raise RuntimeError("canonical push ingest unexpectedly returned source-exhausted")
        camera, streamed, is_train = result
        if is_train:
            # Geometry insertion is part of tick execution, after the plan's
            # version precondition is checked. Keep only the newest job.
            self._pending_integration = (camera, streamed, frame.source)
            self._pending_integration_since_ns = time.monotonic_ns()
        # ``preselected`` keeps pose-bearing cameras for coverage/accounting, but
        # old cameras outside every replay/current pool must not retain live
        # RGB-D payloads or native loader closures.
        self.scene.retire_live_admitted_payloads()
        return result

    def _initialize(self, ready_bindings=None) -> None:
        ready_bindings = None if ready_bindings is None else tuple(ready_bindings)
        self.scene = CanonicalStreamingScene(
            self._runtime_policy,
            self.config.bootstrap,
            self.gaussians,
            () if ready_bindings is not None else tuple(self._bootstrap),
        )
        self._transfer_owner = self._ensure_ready_transfer_owner()
        self.scene.bind_device_transfer_owner(self._transfer_owner)
        self.scene.set_observation_db(self._observation_db)
        self._residency_ledger.bind_transfer_owner(self._transfer_owner)
        if ready_bindings is None:
            self.scene.initialize_from_canonical_frames(
                record_fn=self._record_scene_candidate
            )
        else:
            self.scene.initialize_from_ready_bindings(
                ready_bindings, record_fn=self._record_ready_scene_candidate
            )
        self.gaussians.training_setup(self._backend_optimizer_setup)
        self.gaussians.current_frame_idx = (
            self.bootstrap_frames if ready_bindings is None else len(ready_bindings)
        )
        if self._population_live_requested():
            self.backend.install_population_identity_snapshot_from_bootstrap()

        device = self.gaussians.get_xyz.device
        if self._transfer_owner.device != device:
            raise RuntimeError("production transfer owner device changed after bootstrap")
        bg = [1.0, 1.0, 1.0] if self._runtime_policy.white_background else [0.0, 0.0, 0.0]
        self._background = torch.tensor(bg, dtype=torch.float32, device=device)

        strategy_name = self._runtime_policy.densification_strategy
        self._mcmc_strategy = make_mcmc_strategy(
            strategy_name,
            gaussians=self.gaussians,
            args=self._optimization_policy,
            semantic_cadence=True,
        )
        self._mcmc_strategy.initialize_state(
            gaussians=self.gaussians,
            args=self._optimization_policy,
            scene_scale=float(self.scene.cameras_extent),
        )
        configure_torch_compile(self._optimization_policy)
        configure_training_ssim(self._optimization_policy)

        self._mcmc_cadence_owner = MCMCCadenceOwner(
            grow_interval_pixels=self._runtime_policy.mcmc_grow_interval_pixels,
            post_ingestion_grow_interval_pixels=(
                self._runtime_policy.mcmc_post_ingestion_grow_interval_pixels
            ),
            relocate_interval_pixels=(
                self._runtime_policy.mcmc_relocate_interval_pixels
            ),
            post_ingestion_relocate_interval_pixels=(
                self._runtime_policy.mcmc_post_ingestion_relocate_interval_pixels
            ),
            min_optimizer_updates_after_structural_mutation=(
                self._runtime_policy.mcmc_min_optimizer_updates_after_structural_mutation
            ),
        )

        self._engine = self._services.engine
        optimizer_type = self.config.numerical_mode.lower()
        sparse_active_set = optimizer_type == "selective_adam" and not bool(
            self._runtime_policy.selective_adam_allow_dense_grads
        )
        if bool(self.pipe.gsplat_sparse_grad) != self._runtime_policy.gsplat_sparse_grad:
            raise RuntimeError("canonical renderer sparse-gradient policy disagrees with runtime")
        self._use_energy_mcmc = bool(getattr(self._optimization_policy, "energy_mcmc", True))
        fixed_topology_mcmc_strategy, fixed_shape_mcmc = (
            self._fixed_topology_mcmc_runtime_options()
        )
        self._optimization_runtime = StreamingOptimizationRuntime(
            session=self._engine.session,
            gaussians=self.gaussians,
            args=self._optimization_policy,
            opt=self.opt,
            pipe=self.pipe,
            mcmc_strategy=fixed_topology_mcmc_strategy,
            tb_writer=None,
            # Issue #279: `_phase_timed` (utils.ingest_phase_profiler._timed) is
            # zero-overhead when SPLAT_PROFILE_PHASES is unset (no sync, no
            # dict writes -- see that class's own docstring) and was
            # previously never threaded through here, so engine.py's own
            # per-phase brackets (m_grad_policies/m_loss_assembly/render/
            # backward/optimizer -- see engine.py's run_step()) always fell
            # back to `nullcontext()` on this path regardless of the env var,
            # making causal_mapping runs unprofilable by this mechanism.
            timing_scope=_phase_timed,
            optimizer_type=optimizer_type,
            sparse_active_set=sparse_active_set,
            use_energy_mcmc=self._use_energy_mcmc,
            fixed_shape_mcmc=fixed_shape_mcmc,
            mcmc_local=bool(getattr(self._optimization_policy, "streaming_mcmc_local_only", True)),
            loss_assembly_kwargs={
                "streaming_scene": self.scene,
                "depth_loss_type": self._depth_loss_policy.loss_type,
                "depth_loss_weight": self._depth_loss_policy.weight,
                "scalar_log_interval": max(1, int(getattr(self._optimization_policy, "scalar_log_interval", 100))),
                "_anchor_loss_weight": float(getattr(self._optimization_policy, "streaming_anchor_loss_weight", 0.0)),
                "_anchor_decay_steps": max(1, int(getattr(self._optimization_policy, "streaming_anchor_decay_steps", 500))),
                "_bootstrap_anchor_weight": float(getattr(self._optimization_policy, "streaming_bootstrap_anchor_weight", 1.0)),
                "_bootstrap_anchor_decay": max(1, int(getattr(self._optimization_policy, "streaming_bootstrap_anchor_decay_steps", 2000))),
                "_insertion_anchor_weight": float(getattr(self._optimization_policy, "streaming_insertion_anchor_weight", 1.0)),
                "_insertion_anchor_decay": max(1, int(getattr(self._optimization_policy, "streaming_insertion_anchor_decay_steps", 200))),
                "_lifecycle_enabled": bool(getattr(self._optimization_policy, "streaming_lifecycle_enabled", False)),
                "_mature_anchor_xyz_w": float(getattr(self._optimization_policy, "streaming_mature_anchor_xyz_weight", 0.0)),
                "_mature_anchor_scale_w": float(getattr(self._optimization_policy, "streaming_mature_anchor_scale_weight", 0.0)),
                "_mature_anchor_opacity_w": float(getattr(self._optimization_policy, "streaming_mature_anchor_opacity_weight", 0.0)),
                "nan_skip_count_holder": [0],
            },
            iter_since_new_frame=0,
            camera_transfer_owner=self._transfer_owner,
            latest_admitted_optimizer_update=int(
                getattr(self, "_latest_admitted_optimizer_update", 0)
            ),
            **self._population_shadow_runtime_kwargs(),
        )
        self._mcmc_cfg = MCMCScheduleConfig(
            # Canonical due authority is the exposure owner above. These
            # compatibility fields shape non-cadence schedule values only and
            # cannot impose an iteration start/stop or interval.
            start_iter=0,
            stop_growth_iter=2**63 - 1,
            stop_reloc_iter=2**63 - 1,
            relocate_interval_min=1,
            relocate_interval_max=1,
            relocate_tau=1.0,
            grow_interval_min=1,
            grow_interval_max=1,
            grow_tau=1.0,
            growth_factor_start=self._runtime_policy.mcmc_growth_factor_start,
            growth_factor_min=self._optimization_policy.mcmc_growth_factor_min,
            growth_factor_tau=1.0,
            post_ingestion_growth_factor_start=(
                self._optimization_policy.streaming_mcmc_post_ingestion_growth_factor_start
            ),
            cap_growth_power=self._optimization_policy.mcmc_cap_growth_power,
            cap_interval_strength=0.0,
            cap_interval_power=1.0,
            cap_stop_ratio=self._optimization_policy.mcmc_cap_stop_ratio,
            dead_opacity_start=self._optimization_policy.mcmc_dead_opacity_start,
            dead_opacity_end=self._optimization_policy.mcmc_dead_opacity_end,
            dead_opacity_power=1.0,
            target_capacity=self._runtime_policy.target_capacity,
            use_target_deficit=self._optimization_policy.mcmc_use_target_deficit,
            target_splat_end=self._optimization_policy.mcmc_target_splat_end,
            target_q_start=0.0,
            target_q_end=1.0,
            target_tau=1.0,
        )
        self.initialized = True

    def _record_depth_insertion_render_reuse(self, stats: dict[str, Any]) -> None:
        """Publish compact attribution for exactly one insertion preparation."""

        required = (
            "insertion_render_reused",
            "source_lease_hit",
            "source_h2d_ops",
            "source_d2h_ops",
            "extra_rasterizer_calls",
            "fallback",
        )
        if not isinstance(stats, dict) or any(name not in stats for name in required):
            raise ValueError("depth insertion reuse telemetry is incomplete")
        if type(stats["insertion_render_reused"]) is not bool:
            raise TypeError("insertion_render_reused must be bool")
        if type(stats["source_lease_hit"]) is not bool:
            raise TypeError("source_lease_hit must be bool")
        for name in ("source_h2d_ops", "source_d2h_ops", "extra_rasterizer_calls"):
            if type(stats[name]) is not int or stats[name] < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if not isinstance(stats["fallback"], str) or not stats["fallback"]:
            raise ValueError("fallback must be a non-empty string")
        current = getattr(self, "_depth_insertion_render_reuse_telemetry", None)
        if current is None:
            current = {
                "insertion_render_reused": 0,
                "source_lease_hit": 0,
                "source_h2d_ops": 0,
                "source_d2h_ops": 0,
                "extra_rasterizer_calls": 0,
                "fallback": "not_attempted",
            }
            self._depth_insertion_render_reuse_telemetry = current
        current["insertion_render_reused"] = int(stats["insertion_render_reused"])
        current["source_lease_hit"] = int(stats["source_lease_hit"])
        current["source_h2d_ops"] = int(stats["source_h2d_ops"])
        current["source_d2h_ops"] = int(stats["source_d2h_ops"])
        current["extra_rasterizer_calls"] = int(stats["extra_rasterizer_calls"])
        current["fallback"] = stats["fallback"]

    def _record_decision_trace_insertion_filter(
        self, source: SourceIdentity, stats: Mapping[str, Any], *, debug: bool
    ) -> None:
        if not (
            debug
            and bool(getattr(self, "_population_selection_trace_enabled", False))
        ):
            return
        raw_candidates = stats.get("raw_candidates")
        if type(raw_candidates) is not int or raw_candidates < 0:
            raise ValueError("debug insertion receipt lacks raw candidate count")
        fields = (
            "raw_candidates",
            "after_stable_depth_filter",
            "rejected_unstable_depth",
            "after_alpha_depth_mask",
            "insert_path_low_alpha",
            "insert_path_no_depth",
            "insert_path_sensor_closer",
            "after_temporal_depth_filter",
            "rejected_temporal_depth",
            "after_edge_filter",
            "after_grazing_filter",
            "sparse_seed_input",
            "sparse_seed_reliable",
            "sparse_seed_voxel",
            "sparse_seed_output",
            "device_native_candidates",
        )
        compact = {
            name: value
            for name in fields
            if (value := stats.get(name)) is not None
            and type(value) is int
            and value >= 0
        }
        self._decision_trace_insertion_filter = {
            "source": source.fingerprint(),
            "stats": {
                **({"producer": stats["producer"]} if isinstance(stats.get("producer"), str) else {}),
                **compact,
            },
        }

    def depth_insertion_render_reuse_telemetry(self) -> dict[str, Any]:
        """Return a bounded copy of the latest insertion reuse attribution."""

        return dict(getattr(self, "_depth_insertion_render_reuse_telemetry", {}))

    @staticmethod
    def _native_prepared_source_lease(
        camera: Any, source: Any, modality_eligibility: Any | None = None
    ):
        """Borrow already-resident native RGB-D/pose values without H2D.

        This is deliberately a strict bridge, not a camera materializer.  A
        CPU resident value, non-native camera, missing pose matrix, or absent
        source identity simply returns ``None`` and lets the established path
        run.  The returned lease contains no map/version state.
        """

        from eligsir.contracts import (
            BufferOwnership,
            BufferRef,
            BufferSpec,
            MemoryDomain,
            ModalityEligibility,
            SourceIdentity,
        )
        from eligsir.runtime.prepared_source import PreparedSourceLease

        if not isinstance(source, SourceIdentity):
            return None
        tier = getattr(camera, "_canonical_resolution_tier", None)
        if getattr(tier, "value", tier) != "native":
            return None
        native = getattr(camera, "_canonical_native_resolution_state", None)
        rgb_chw = getattr(native, "rgb", getattr(camera, "original_image", None))
        depth = getattr(native, "depth", getattr(camera, "_sensor_depth_cache", None))
        w2c = getattr(camera, "world_view_transform", None)
        camera_intrinsics = getattr(getattr(camera, "gsplat", None), "K", None)
        if not all(
            hasattr(value, "device")
            for value in (rgb_chw, depth, w2c, camera_intrinsics)
        ):
            return None
        if rgb_chw.ndim != 3 or depth.ndim != 2 or tuple(rgb_chw.shape[1:]) != tuple(depth.shape):
            return None
        if (
            rgb_chw.device != depth.device
            or rgb_chw.device != w2c.device
            or rgb_chw.device != camera_intrinsics.device
            or camera_intrinsics.shape != (3, 3)
        ):
            return None
        if getattr(rgb_chw.device, "type", "cpu") == "cpu":
            return None
        try:
            import torch as torch_module

            with torch_module.no_grad():
                # Camera matrices are world->camera stored transposed.  This
                # mirrors objective.py and remains a device-only operation.
                c2w = torch_module.inverse(w2c.transpose(0, 1)).detach()
            rgb_hwc = rgb_chw.permute(1, 2, 0).detach()
        except Exception:
            return None
        if c2w.shape != (4, 4):
            return None

        def ref(value: Any, encoding: str) -> BufferRef:
            return BufferRef(
                data=value,
                spec=BufferSpec(
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(value.dtype).removeprefix("torch."),
                    memory_domain=MemoryDomain.OTHER_DEVICE,
                    encoding=encoding,
                    device=str(value.device),
                    contiguous=bool(value.is_contiguous()),
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=native if native is not None else camera,
            )

        try:
            return PreparedSourceLease(
                source_identity=source,
                modality_eligibility=modality_eligibility
                or getattr(camera, "modality_eligibility", None)
                or ModalityEligibility(
                    rgb_supervision=True,
                    depth_supervision=True,
                    depth_insertion=True,
                ),
                source_metadata={
                    "native_height": int(depth.shape[0]),
                    "native_width": int(depth.shape[1]),
                },
                tier_preparation_available=("native",),
                borrowed_device_payloads={
                    "depth_meters": ref(depth.detach(), "depth_meters"),
                    "rgb": ref(rgb_hwc, "rgb_hwc"),
                    "c2w": ref(c2w, "c2w"),
                    "camera_intrinsics": ref(
                        camera_intrinsics.detach(), "camera_intrinsics"
                    ),
                },
                device=str(depth.device),
            )
        except (TypeError, ValueError):
            return None

    def _select_depth_insertion_render_reuse(
        self,
        *,
        result: Any,
        camera: Any,
        source: Any,
        plan: MappingWorkPlan,
        source_lease: Any | None = None,
    ):
        """Select one exact B1/B4 native geometry lease without rerendering."""

        from eligsir.mapping.integration.depth_render_reuse import (
            DepthInsertionReuseReason,
            bind_depth_insertion_render_reuse,
        )
        from eligsir.optimization.gaussian.supervision_evidence import (
            SupervisionEvidenceBatch,
        )

        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, SupervisionEvidenceBatch):
            return None, DepthInsertionReuseReason.MISSING_RENDER_LEASE
        matching = tuple(
            item
            for item in evidence.geometry_render_leases
            if item.source_identity == source
        )
        if len(matching) != 1:
            return None, DepthInsertionReuseReason.MISSING_RENDER_LEASE
        render_lease = matching[0]
        parameter_before = getattr(result, "parameter_version_before", None)
        if (
            type(parameter_before) is not int
            or type(getattr(plan, "parameter_version", None)) is not int
            or int(plan.parameter_version) != parameter_before
        ):
            return None, DepthInsertionReuseReason.STALE_SNAPSHOT
        expected = replace(
            render_lease.snapshot_token,
            map_content=int(plan.map_content_version),
            topology=int(plan.topology_version),
            parameter=parameter_before,
        )
        if source_lease is None:
            source_lease = self._native_prepared_source_lease(
                camera,
                source,
                render_lease.modality_eligibility,
            )
        return bind_depth_insertion_render_reuse(
            render_lease=render_lease,
            source_lease=source_lease,
            source=source,
            expected_snapshot=expected,
            expected_resolution_tier="native",
        )

    def _prepare_insertion_proposal(
        self,
        camera: Any,
        frame: StreamingObservation,
        source: Any,
        plan: MappingWorkPlan,
        *,
        render_reuse: Any | None = None,
        render_reuse_fallback: str = "legacy_native_render",
    ) -> Any | None:
        if not plan.insertion.enabled:
            return None
        primary_work = next(
            (
                work
                for work in plan.supervision_work
                if work.source == source
            ),
            None,
        )
        if primary_work is None:
            raise RuntimeError("depth insertion source has no planned supervision work")
        proposal = self._prepare_depth_insertion_proposal(
            camera,
            frame,
            source,
            insertion_plan=plan.insertion,
            modality_eligibility=primary_work.modality_eligibility,
            prev_frame=self._prev_insert_frame,
            prev_depth_meters=self._prev_insert_depth_m,
            render_reuse=render_reuse,
            render_reuse_fallback=render_reuse_fallback,
        )
        from eligsir.runtime.ready_insertion_frame import ReadyInsertionFrame

        # READY leases return to their owner when the transaction finishes;
        # retaining their borrowed depth across ticks could outlive eviction or
        # generation replacement.  Only legacy frames own cross-tick depth.
        if isinstance(frame, ReadyInsertionFrame):
            self._prev_insert_frame = self._prev_insert_depth_m = None
        else:
            self._prev_insert_frame = frame
            self._prev_insert_depth_m = (
                None if frame._depth_np is None else np.asarray(frame._depth_np, dtype=np.float32)
            )
        return proposal

    def _prepare_depth_insertion_proposal(
        self,
        camera: Any,
        frame: StreamingObservation,
        source: Any,
        *,
        insertion_plan: Any,
        modality_eligibility: Any,
        prev_frame: StreamingObservation | None,
        prev_depth_meters: np.ndarray | None,
        insertion_policy: Any | None = None,
        render_reuse: Any | None = None,
        render_reuse_fallback: str = "legacy_native_render",
    ) -> Any | None:
        """Prepare one authorized device-native proposal without committing it.

        ``insertion_policy`` defaults to the live ``self._depth_insertion_policy``;
        callers driving the post-source-complete replay-depth pass pass
        ``self._replay_depth_insertion_policy`` instead (see #1259 far-region
        starvation follow-up).

        Renders ``camera`` at NATIVE resolution for the occupancy-gate check
        below, regardless of whatever supervision tier ``camera`` currently
        carries (issue #1259). ``camera`` is the same, long-lived object
        ``execute_fixed_topology_transaction``'s ``m_tick_camera_materialize``
        stage rebinds in place via ``materialize_resolution_tier`` earlier in
        this same ``tick()`` -- so without this, a mapping run pinned (or
        adaptively dropped) to a coarser tier renders this gate's
        ``render_alpha``/``render_depth`` at that coarser resolution while the
        insertion candidates it gates remain native. gsplat's minimum-radius/
        antialias dilation is not scale-invariant across tiers, so a
        fixed-world-size Gaussian occupies a larger share of a coarser pixel:
        coverage reads as falsely denser and the gate over-suppresses
        insertion. Measured on Replica room0 (dense, valid synthetic depth):
        36.8 -> 7.5 candidates/frame at quarter tier, an 80% loss, vs. a
        barely-visible drop on TUM fr3 (real scene, already-marginal
        insertion) -- see
        docs/architecture/2026-08-29-replica-quarter-tier-anomaly-investigation.md.
        ``materialize_resolution_tier`` always derives a tier from the
        camera's own immutable ``_canonical_native_resolution_state``, never
        chaining tier-off-tier, so binding to NATIVE here for one render and
        then re-binding back is exactly as cheap and side-effect-free as any
        other tier switch this camera already undergoes across ticks --
        the memoization in ``materialize_resolution_tier`` makes the
        restore a no-op if nothing else touches this camera meanwhile. It
        also means this bypasses (rather than reintroduces) whatever the
        ``masked``/``ray_preserving`` ``resampling_mode``s guard against:
        those concern how a DERIVED tier is resampled from native, and NATIVE
        is the canonical state itself, so the gate render never resamples at
        all.
        """

        assert self.scene is not None
        active_policy = (
            self._depth_insertion_policy if insertion_policy is None else insertion_policy
        )
        if render_reuse is None:
            _prior_tier = getattr(camera, "_canonical_resolution_tier", None)
            _needs_native_gate_render = (
                _prior_tier is not None and _prior_tier is not ResolutionTier.NATIVE
            )
            if _needs_native_gate_render:
                # Only touch self.config (absent on minimal test doubles that
                # construct this runtime without going through __init__) when a
                # tier switch is actually required.
                _resampling_mode = str(
                    self.config.static_options.get("resolution_resampling_mode", "legacy")
                )
                _depth_downsample = str(
                    self.config.static_options.get("depth_loss_downsample", "nearest")
                )
                materialize_resolution_tier(
                    camera,
                    ResolutionTier.NATIVE,
                    resampling_mode=_resampling_mode,
                    depth_downsample=_depth_downsample,
                )
            try:
                with torch.no_grad():
                    pkg = render(
                        camera,
                        self.gaussians,
                        self.pipe,
                        self._background,
                        render_depth=True,
                    )
            finally:
                if _needs_native_gate_render:
                    materialize_resolution_tier(
                        camera,
                        _prior_tier,
                        resampling_mode=_resampling_mode,
                        depth_downsample=_depth_downsample,
                    )
            render_alpha = pkg.get("alpha")
            rendered_depth = pkg.get("rendered_depth")
            extra_rasterizer_calls = 1
        else:
            render_alpha = render_reuse.alpha
            rendered_depth = render_reuse.rendered_depth
            extra_rasterizer_calls = 0
        candidates, _stats = insert_gaussians_from_frame_device_native(
            self.gaussians,
            frame,
            self.scene,
            prev_frame=prev_frame,
            prev_depth_meters=prev_depth_meters,
            render_alpha=render_alpha,
            render_depth=rendered_depth,
            current_frame_idx=self.frames_seen,
            insertion_plan=insertion_plan,
            insertion_policy=active_policy,
            modality_eligibility=modality_eligibility,
            render_reuse=render_reuse,
        )
        # The real device producer returns a dict.  Keep narrow legacy/test
        # seams that returned an opaque stats placeholder behaviorally intact:
        # telemetry still reports the known render/reuse facts, without
        # making candidate production depend on a diagnostic shape.
        _stats = dict(_stats) if isinstance(_stats, dict) else {}
        _stats.setdefault("insertion_render_reused", render_reuse is not None)
        _stats.setdefault("source_lease_hit", render_reuse is not None)
        _stats.setdefault("source_h2d_ops", 0 if render_reuse is not None else 1)
        _stats.setdefault("source_d2h_ops", 0)
        _stats["extra_rasterizer_calls"] = extra_rasterizer_calls
        _stats["fallback"] = "reused" if render_reuse is not None else str(render_reuse_fallback)
        self._record_depth_insertion_render_reuse(
            {
                "insertion_render_reused": bool(_stats["insertion_render_reused"]),
                "source_lease_hit": bool(_stats["source_lease_hit"]),
                "source_h2d_ops": int(_stats["source_h2d_ops"]),
                "source_d2h_ops": int(_stats["source_d2h_ops"]),
                "extra_rasterizer_calls": int(_stats["extra_rasterizer_calls"]),
                "fallback": str(_stats["fallback"]),
            }
        )
        self._record_decision_trace_insertion_filter(
            source,
            _stats,
            debug=bool(getattr(active_policy, "debug_statistics", False)),
        )
        if candidates is None or len(candidates) == 0:
            return None
        candidates = normalize_pending_candidates(
            (candidates,),
            device=candidates.pts.device,
            init_opacity=float(active_policy.initial_opacity),
            transfer_owner=self._transfer_owner,
        )
        return insertion_proposal_from_plan(
            insertion_plan,
            candidates,
            key=ProposalKey(
                stream_epoch=int(source.stream_epoch),
                source_ordinal=int(source.source_ordinal),
                timestamp_ns=int(source.timestamp_ns),
                source_id=str(source.source_id),
            ),
            # Shared insertion storage retains the historical ``birth_iter``
            # field name, but canonical rows are stamped in optimizer-update
            # units. Artifact attempt identity never enters Gaussian age.
            iteration=self._work_clock_owner.snapshot().optimizer_updates,
        )

    @staticmethod
    def _decode_reload_token(token: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Decode one deterministic file-backed cold handle on selection."""

        from types import SimpleNamespace

        from eligsir.contracts.runtime import PixelReloadToken
        from eligsir.io.sources.decode_cache import (
            load_frame_depth_np,
            load_frame_rgb,
        )
        from eligsir.utils.rgbd_frames import depth_to_meters

        if not isinstance(token, PixelReloadToken):
            raise TypeError("cold reload token must be PixelReloadToken")
        if token.provider != "rgbd_files_v1":
            raise ValueError(f"unsupported cold reload provider: {token.provider}")
        if len(token.locators) != 2:
            raise ValueError("rgbd_files_v1 requires RGB and depth locators")
        parameters = dict(token.parameters)
        if "depth_scale" not in parameters:
            raise ValueError("rgbd_files_v1 requires depth_scale")
        frame = SimpleNamespace(
            rgb_path=token.locators[0],
            depth_path=token.locators[1] or None,
            _rgb_np=None,
            _rgb_bytes=None,
            _depth_np=None,
            _depth_bytes=None,
            _sens_header=None,
        )
        rgb = np.asarray(load_frame_rgb(frame), dtype=np.uint8)
        raw_depth = load_frame_depth_np(frame)
        depth = None
        if raw_depth is not None:
            depth = depth_to_meters(
                raw_depth, float(parameters["depth_scale"])
            ).astype(np.float32, copy=False)
        return np.ascontiguousarray(rgb), (
            None if depth is None else np.ascontiguousarray(depth)
        )

    def _replay_mapping_frame(self, source: Any, camera: Any | None = None) -> Any | None:
        """Resolve CPU/cold pixels for exactly one already-planned replay view."""

        mapping = getattr(camera, "_mapping_frame", None)
        if mapping is None:
            mapping = getattr(getattr(camera, "_streaming_frame", None), "mapping_frame", None)
        if mapping is not None:
            if mapping.source != source:
                raise RuntimeError("replay camera MappingFrame identity mismatch")
            return mapping
        frame_id = self._observation_db.frame_id_for_source(source)
        if frame_id is None:
            return None
        cold = self._observation_db.get_cold_pixels(frame_id)
        metadata = self._observation_db.get_metadata(frame_id)
        if cold is None or metadata is None:
            return None
        rgb = cold.rgb
        depth = cold.depth
        reload_token = getattr(cold, "reload_token", None)
        if reload_token is not None and (rgb is None or depth is None):
            # Issue #279/#1259/#1342: a background job scheduled at admission
            # time (`_maybe_schedule_admission_prefetch`) may already have
            # decoded this exact token -- same deterministic decode, so a
            # cache hit here is byte-identical to calling
            # `_decode_reload_token` fresh, just without paying for it on
            # this tick. A miss (not yet scheduled, still in flight, disabled,
            # or already consumed) falls through to the original cold decode
            # unchanged.
            prefetched = self._pop_admission_prefetch(frame_id)
            if prefetched is not None:
                rgb, depth = prefetched
            elif getattr(reload_token, "provider", None) == "prepared_cold_disk_rgbd_v1":
                self._maybe_schedule_admission_prefetch(frame_id, cold)
                return None
            else:
                rgb, depth = self._decode_reload_token(reload_token)
        if not isinstance(rgb, np.ndarray) or not isinstance(depth, np.ndarray):
            return None
        from eligsir.contracts.runtime import (
            BufferOwnership,
            BufferRef,
            MappingFrame,
        )
        from eligsir.contracts.types import CameraIntrinsics, Pose

        intrinsics = cold.intrinsics
        if not isinstance(intrinsics, CameraIntrinsics):
            return None
        return MappingFrame(
            source=source,
            rgb=BufferRef.from_numpy(
                rgb,
                encoding="rgb8",
                ownership=BufferOwnership.SHARED_IMMUTABLE,
            ),
            depth=BufferRef.from_numpy(
                depth,
                encoding="32FC1",
                units="m",
                ownership=BufferOwnership.SHARED_IMMUTABLE,
            ),
            intrinsics=intrinsics,
            device="cpu",
            pose=Pose(np.asarray(metadata.pose_c2w, dtype=np.float32)),
            reload_token=reload_token,
        )

    def _restore_camera_payload_from_cold(self, source: Any, camera: Any) -> bool:
        """Rebuild one retired planned camera on CPU before owned H2D paging.

        Returns True whenever *camera* ends this call with a usable payload
        -- including the no-op case where it already had one. The caller
        (``tick``'s camera-materialize loop) reads any False return as
        "still pending, retry next tick"; returning False here for an
        already-ready camera made that camera's tick retry forever with no
        prefetch ever scheduled (`last_tick_reason=cold_payload_pending`,
        `_admission_prefetch_scheduled_total` staying 0), discovered via a
        live causal_replay/replica GPU run pinned to a `dataset_fps`-paced,
        `hybrid_keyframe`-admitted recipe (2026-09-09) where a source can
        reach this call while its camera is already resident.
        """

        from eligsir.contracts.streaming_bridge import mapping_frame_source_arrays
        from eligsir.runtime.payload_residency import camera_payload_ready
        from eligsir.runtime.streaming_observation import StreamingObservation
        import torch as torch_module

        if camera_payload_ready(camera):
            return True
        mapping = self._replay_mapping_frame(source, camera)
        if mapping is None:
            frame_id = self._observation_db.frame_id_for_source(source)
            cold = None if frame_id is None else self._observation_db.get_cold_pixels(frame_id)
            token = None if cold is None else getattr(cold, "reload_token", None)
            if getattr(token, "provider", None) == "prepared_cold_disk_rgbd_v1":
                with self._admission_prefetch_lock:
                    if frame_id in self._admission_prefetch_pending:
                        return False
                # The worker can complete between the nonblocking prefetch pop
                # above and this check. Retry once before treating the token as
                # terminally unavailable; this never waits or decodes here.
                mapping = self._replay_mapping_frame(source, camera)
                if mapping is None:
                    raise RuntimeError(
                        "planned cold replay cache token completed without pixels: "
                        f"{source.fingerprint()}"
                    )
            else:
                raise RuntimeError(
                    "planned cold replay identity has no reloadable RGB-D ownership: "
                    f"{source.fingerprint()}"
                )
        observation = StreamingObservation.from_mapping_frame(mapping)
        depth, rgb = mapping_frame_source_arrays(observation)
        intrinsics = mapping.intrinsics
        camera.original_image = torch_module.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        )
        camera._sensor_depth_cache = torch_module.from_numpy(
            np.ascontiguousarray(depth)
        )
        camera._mapping_frame = mapping
        camera._streaming_frame = observation
        camera._canonical_native_resolution_state = None
        camera._canonical_resolution_binding_native_state = None
        camera._canonical_resolution_tier = None
        # `camera.original_image` above is a brand-new tensor (fresh cold
        # reload) at this frame's NATIVE resolution/dtype. `camera.score_provenance`
        # is a camera-level attribute (utils/score_contract.py's
        # `camera_provenance`/`score_tensor_from_camera`), not tensor-attached, so
        # it survives this reassignment untouched unless explicitly refreshed here
        # -- and `_native_state()` in resolution_materialization.py later reads
        # this exact attribute and pairs it with the new `original_image` as if it
        # described it. Left stale (e.g. a prior tier's resolution/dtype from
        # before this camera object was retired/evicted), that stale pairing
        # survives every `factor != 1` resolution_materialization call (which
        # only rewrites the `resolution` field of an existing provenance, not its
        # dtype/domain/history) and is passed through completely unmodified on
        # the very first `factor == 1` (native-tier) call -- exactly the
        # mapping-at-a-derived-tier -> native-tier-refinement transition, which
        # raised `ValueError: canonical score provenance does not match tensor
        # dtype/domain/resolution` on a real ScanNet++ run (issue #1259).
        # Re-declaring fresh provenance for these exact new native pixels here
        # closes the gap at its source.
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(camera, context="cold_replay_restore")
        camera.image_width = camera.width = int(intrinsics.width)
        camera.image_height = camera.height = int(intrinsics.height)
        camera.fx = float(intrinsics.fx)
        camera.fy = float(intrinsics.fy)
        camera.cx = float(intrinsics.cx)
        camera.cy = float(intrinsics.cy)
        camera.gsplat = None
        camera._live_payload_retired = False
        from eligsir.runtime.payload_residency import mark_payload_residency_dirty

        scene = getattr(self, "scene", None)
        if scene is not None:
            mark_payload_residency_dirty(scene)
        return True

    def _prepare_replay_insertion_proposal(
        self,
        camera: Any,
        source: Any,
        plan: MappingWorkPlan,
        *,
        render_reuse: Any | None = None,
        render_reuse_fallback: str = "legacy_native_render",
    ) -> Any | None:
        insertion = plan.replay_insertion
        if insertion is None:
            return None
        replay_work = next(
            (work for work in plan.replay_work if work.source == source), None
        )
        if replay_work is None:
            raise RuntimeError("replay depth source has no planned replay work")
        mapping = self._replay_mapping_frame(source, camera)
        if mapping is None:
            return None
        frame = StreamingObservation.from_mapping_frame(mapping)
        # Random full-cache replay identities are not temporal neighbours. Do
        # not borrow the live frontier pair state or corrupt it for the next
        # live insertion; all other stability/occupancy gates remain active.
        return self._prepare_depth_insertion_proposal(
            camera,
            frame,
            source,
            insertion_plan=insertion,
            modality_eligibility=replay_work.modality_eligibility,
            prev_frame=None,
            prev_depth_meters=None,
            insertion_policy=self._replay_depth_insertion_policy,
            render_reuse=render_reuse,
            render_reuse_fallback=render_reuse_fallback,
        )

    @staticmethod
    def _prepared_ready_pending_integration(
        ready_transaction: Any, plan: MappingWorkPlan
    ) -> tuple[tuple[Any, Any, Any] | None, str | None]:
        """Bind exactly the plan's newest fresh insertion source, if present."""
        if not getattr(ready_transaction, "source_insertion", False) or not bool(
            getattr(plan.insertion, "enabled", False)
        ):
            return None, None
        source = plan.insertion.source
        if not plan.supervision_sources or source != max(
            plan.supervision_sources, key=SourceIdentity.ordering_key
        ):
            return None, "insertion_source_is_not_the_newest_fresh_source"
        binding = next(
            (
                item
                for item in ready_transaction.bindings
                if getattr(item, "source_identity", None) == source
            ),
            None,
        )
        if binding is None:
            return None, "insertion_source_not_bound"
        from eligsir.runtime.ready_insertion_frame import ReadyInsertionFrame

        return (binding.camera, ReadyInsertionFrame.from_binding(binding), source), None

    def _commit_mutation_barrier(
        self,
        proposals: tuple[Any, ...],
        *,
        expected_map_content_version: int,
        expected_topology_version: int,
        allow_delegated: bool = False,
    ):
        """Commit all typed structural proposals once after fixed-topology work."""

        if not proposals:
            return None
        epoch = GaussianMutationEpoch(
            self.backend,
            expected_map_content_version=expected_map_content_version,
            expected_topology_version=expected_topology_version,
            max_population=self._runtime_policy.cap_max,
            allow_delegated=allow_delegated,
        )
        result = epoch.commit(proposals)
        inserted = int(result.inserted)
        if inserted > 0:
            assert self.scene is not None
            new_xyz = self.gaussians.get_xyz[-inserted:]
            self.scene.add_to_occupancy_hash(new_xyz)
            self._services.four_clocks.record_map_commit(n_inserted=inserted)
        return result

    def _work_clock_owner_if_available(self) -> RuntimeWorkClockOwner | None:
        """Permit legacy ``__new__`` CPU seams, never an ownerless real runtime."""

        owner = getattr(self, "_work_clock_owner", None)
        if owner is None and hasattr(self, "config"):
            raise RuntimeError("canonical runtime is missing its semantic work-clock owner")
        return owner

    @staticmethod
    def _required_result_count(result: Any, name: str) -> int:
        value = object.__getattribute__(result, name)
        if type(value) is not int:
            raise TypeError(f"completed optimization {name} must be an exact integer")
        if value < 0:
            raise ValueError(f"completed optimization {name} must be non-negative")
        return value

    def _record_completed_optimization_result(
        self, result: Any, *, mapping_active_ns: int
    ) -> RuntimeWorkClockSnapshot:
        """Commit one result before any later lifecycle or mutation work."""

        optimizer_updates = self._required_result_count(
            result, "optimizer_step_count"
        )
        supervised_views = self._required_result_count(
            result, "supervision_view_count"
        )
        supervised_pixels = self._required_result_count(
            result, "supervised_pixel_count"
        )
        depth_valid_samples = self._required_result_count(
            result, "depth_valid_sample_count"
        )
        if type(mapping_active_ns) is not int:
            raise TypeError("mapping_active_ns must be an exact integer")
        if mapping_active_ns < 0:
            raise ValueError("mapping_active_ns must be non-negative")
        if depth_valid_samples > 0 and optimizer_updates == 0:
            raise ValueError(
                "positive depth_valid_sample_count requires optimizer_step_count"
            )
        self._work_clock_owner.record_optimization(
            optimizer_updates=optimizer_updates,
            supervised_views=supervised_views,
            supervised_pixels=supervised_pixels,
            mapping_active_ns=0,
        )
        self._work_clock_owner.record_depth_supervision(depth_valid_samples)
        return self._work_clock_owner.record_mapping_active(mapping_active_ns)

    def _record_completed_mcmc_commit(
        self, commit: Any | None
    ) -> RuntimeWorkClockSnapshot:
        """Advance once immediately after an applied MCMC arbiter commit."""

        if commit is not None and self._required_result_count(commit, "applied") > 0:
            self._work_clock_owner.record_mcmc_mutation()
        return self._work_clock_owner.snapshot()

    def _record_completed_depth_mutation(
        self, commit: Any | None
    ) -> RuntimeWorkClockSnapshot:
        """Advance only when a depth proposal actually applied."""

        if commit is not None and self._required_result_count(commit, "applied") > 0:
            snapshot = self._work_clock_owner.record_depth_mutation()
            cadence_owner = getattr(self, "_mcmc_cadence_owner", None)
            if cadence_owner is not None:
                cadence_owner.record_structural_mutation(
                    snapshot,
                    topology_version_after=int(commit.change.topology_version),
                )
        return self._work_clock_owner.snapshot()

    def tick(
        self, *, ready_transaction: _ReadyOnlyB4Transaction | None = None
    ) -> MappingTickResult:
        tick_started_ns = time.monotonic_ns()
        with _phase_timed("m_tick_prologue"):
            if self._closed:
                raise RuntimeError("tick after runtime close")
            self._commit_prepared_cold_payload_spill_receipts()
            if bool(getattr(self, "_population_selection_trace_enabled", False)):
                self._decision_trace_insertion_filter = None
            # Issue #279 follow-up (2026-08-31): sub-brackets below break down
            # m_tick_prologue's own 6.8ms/tick into its constituent stages --
            # see docs/architecture/2026-08-31-issue279-tick-attribution.md.
            with _phase_timed("m_tick_prologue_epoch_bookkeeping"):
                # Every packet submitted before this call observed the same frozen
                # allocation epoch. Advance once here so the next drain cannot mutate
                # hysteresis within a true-camera transaction.
                self._control_tick_epoch = int(
                    getattr(self, "_control_tick_epoch", 0)
                ) + 1
                self._work_clock_owner.record_control_tick()
                work = (
                    (ready_transaction.snapshot, ready_transaction.plan)
                    if ready_transaction is not None
                    else self._latest_tick_work or self._latest_report_work
                )
                if ready_transaction is None:
                    self._latest_tick_work = None
                    self._latest_report_work = None
                if work is None and self.initialized and self._last_materialized_work is not None:
                    active_snapshot, active_plan = self._last_materialized_work
                    idle_snapshot = self._fresh_idle_snapshot(active_snapshot)
                    work = (
                        idle_snapshot,
                        self._idle_replay_plan(idle_snapshot, active_plan),
                    )
                snapshot, plan = (None, None) if work is None else work
            with _phase_timed("m_tick_prologue_work_plan_resolution"):
                if not self.initialized:
                    if plan is None or len(self._bootstrap) < self.bootstrap_frames:
                        return self._tick_result(
                            False, "bootstrap_pending", snapshot=snapshot, plan=plan
                        )
                    with _phase_timed("m_tick_prologue_fence_assert"):
                        self._assert_plan_version_fence(plan)
                    self._initialize()
                if (
                    plan is None
                    or plan.optimizer_updates <= 0
                    or not (plan.supervision_sources or plan.replay_sources)
                ):
                    return self._tick_result(
                        False, "work_plan_empty", snapshot=snapshot, plan=plan
                    )

                assert self.scene is not None
            with _phase_timed("m_tick_prologue_fence_assert"):
                self._assert_plan_version_fence(plan)
            with _phase_timed("m_tick_prologue_scheduled_transaction_work"):
                self._residency_ledger.release(self.scene.hot_camera_ids())
                constellation = plan.supervision_constellation
                transaction_work = _scheduled_transaction_work(
                    plan, configured_batch_size=self._batch_size
                )
                if not transaction_work:
                    return self._tick_result(False, "work_plan_empty", snapshot=snapshot, plan=plan)
                if self._batch_size > 1:
                    if len(transaction_work) != constellation.realized_batch_size:
                        raise RuntimeError("work plan does not contain its realized B identities")
                    identities = tuple(work.source for work in transaction_work)
                    if len(identities) != len(set(identities)):
                        raise RuntimeError("work plan duplicated a supervision identity")
                if len({work.resolution_tier for work in transaction_work}) != 1:
                    raise RuntimeError(
                        "adaptive supervision transaction requires one resolution tier; "
                        "refusing fallback"
                    )
                # This identity-bearing plan is the scheduler's sole source
                # of neutral mutation affordability for M5.  The shadow
                # policy never consumes the legacy grow/relocate due flags.
                self._population_shadow_active_plan = plan
            with _phase_timed("m_tick_prologue_cuda_event_construction"):
                # Host-monotonic active time begins at camera lookup/materialization and
                # transfer dispatch. It intentionally measures mapper-side execution
                # without synchronizing the asynchronous GPU stream.
                mapping_active_checkpoint_ns = time.monotonic_ns()
                optimization_transaction_started_ns = mapping_active_checkpoint_ns
                optimization_start = torch.cuda.Event(enable_timing=True)
                optimization_end = torch.cuda.Event(enable_timing=True)
                optimization_host_prefix_ns = None
                prepared_insertion_skip = None
                if ready_transaction is not None:
                    pending_integration, prepared_insertion_skip = (
                        self._prepared_ready_pending_integration(ready_transaction, plan)
                    )
                else:
                    pending_integration = self._pending_integration
                if ready_transaction is None:
                    self._pending_integration = None
                    self._pending_integration_since_ns = None

            from scene.cameras import prepare_camera_for_render

            device = self.gaussians.get_xyz.device
            assert self._transfer_owner is not None
            cameras = (
                dict(ready_transaction.cameras)
                if ready_transaction is not None else {}
            )
            # HOT-hit vs first-use-miss accounting for this transaction. A tier
            # that was already resident renders without paying any preparation
            # cost (~0.02ms measured); a miss pays reload/resize/stage/H2D
            # (5-12ms at 640x480, 36-102ms at 1920x1440, PR #1252). Blending the
            # two into one per-tier average would make a tier look expensive when
            # what was really observed is a one-time miss that amortizes away.
            supervision_cache_hits = len(transaction_work) if ready_transaction is not None else 0
            supervision_cache_misses = 0
        with _phase_timed("m_tick_camera_materialize"):
            for view_work in (() if ready_transaction is not None else transaction_work):
                resident_supervision = (
                    self._observation_db.resident_supervision_for_source(
                        view_work.source, view_work.resolution_tier.value
                    )
                )
                camera = self._camera_for_source(
                    view_work.source,
                    current_snapshot=snapshot,
                    resident_supervision=resident_supervision,
                )
                if camera is None:
                    # Abandoning the transaction must not silently drop the
                    # depth-insertion job this tick had already taken ownership of
                    # (issue #1289 made this branch reachable for the first time:
                    # `_camera_for_source` previously either returned a camera or
                    # raised). Hand it back so the next tick still commits it.
                    self._pending_integration = pending_integration
                    self._pending_integration_since_ns = (
                        None if pending_integration is None else time.monotonic_ns()
                    )
                    return self._tick_result(False, "no_training_camera", snapshot=snapshot, plan=plan)
                if resident_supervision is not None:
                    supervision_cache_hits += 1
                    self._bind_resident_supervision(
                        camera,
                        view_work.source,
                        view_work.resolution_tier,
                        resident_supervision,
                    )
                else:
                    supervision_cache_misses += 1
                    if not self._restore_camera_payload_from_cold(view_work.source, camera):
                        self._pending_integration = pending_integration
                        self._pending_integration_since_ns = (
                            None if pending_integration is None else time.monotonic_ns()
                        )
                        if ready_transaction is None:
                            # A cache-owned page-in is producer work.  Keep the
                            # exact frozen legacy plan for its nonblocking retry
                            # instead of dropping the selected identity.
                            self._latest_tick_work = (snapshot, plan)
                        return self._tick_result(
                            False,
                            "cold_payload_pending",
                            snapshot=snapshot,
                            plan=plan,
                        )
                    if getattr(camera, "has_pending_image", False):
                        camera.materialize_pending_image()
                    # Every tier is derived from the camera's canonical NATIVE
                    # state, never chained off another materialized tier -- see
                    # `materialize_resolution_tier`. Load-bearing: chaining would
                    # compound area/nearest filtering artifacts, and deriving from
                    # native is also 14-72x cheaper when native is resident.
                    materialize_resolution_tier(
                        camera,
                        view_work.resolution_tier,
                        resampling_mode=str(
                            self.config.static_options.get(
                                "resolution_resampling_mode", "legacy"
                            )
                        ),
                        discontinuity_abs_floor_m=float(
                            self.config.static_options.get(
                                "resolution_discontinuity_abs_floor_m", 0.05
                            )
                        ),
                        discontinuity_rel_frac=float(
                            self.config.static_options.get(
                                "resolution_discontinuity_rel_frac", 0.02
                            )
                        ),
                        depth_downsample=str(
                            self.config.static_options.get(
                                "depth_loss_downsample", "nearest"
                            )
                        ),
                    )
                cameras[view_work.source] = camera
        with _phase_timed("m_tick_camera_prefetch"):
            if ready_transaction is None:
                self._transfer_owner.prefetch_cameras(tuple(cameras.values()))
        with _phase_timed("m_tick_camera_prepare"):
            for view_work in (() if ready_transaction is not None else transaction_work):
                camera = cameras[view_work.source]
                prepare_camera_for_render(camera, device=str(device))
                self._note_scene_camera_page_in(view_work.source, camera)
                frame_id = self._observation_db.frame_id_for_source(view_work.source)
                if frame_id is None:
                    raise RuntimeError("planned supervision identity is absent from observation DB")
                self._observation_db.activate_supervision(
                    frame_id,
                    {
                        view_work.resolution_tier.value: CacheHandle(
                            rgb=camera.original_image,
                            depth=getattr(camera, "_sensor_depth_cache", None),
                            intrinsics=(
                                int(camera.image_width), int(camera.image_height),
                                getattr(camera, "fx", None), getattr(camera, "fy", None),
                                getattr(camera, "cx", None), getattr(camera, "cy", None),
                            ),
                            score_provenance=getattr(camera, "score_provenance", None),
                        )
                    },
                )
        optimizer_steps = 0
        supervision_views = 0
        rasterizer_calls = 0
        supervised_pixels = 0
        successful_supervision_sources: set[Any] = set()
        unique_supervision_views_known = True
        realized_microbatch_sizes: list[int] = []
        training_result = None
        training_camera = None
        training_source_id = None
        training_source = None
        primary_training_gaussian_count = None
        optimization_mutation = None
        population_live_requested = self._population_live_requested()
        if self._batch_size > 1:
            work_clock = self._work_clock_owner.snapshot()
            optimizer_update_id = work_clock.optimizer_updates + 1
            optimizer_update_permitted = _planned_optimizer_update_permitted(
                plan, update_ordinal=0
            )
            # `self.iteration` must never advance ahead of the work-clock
            # owner's `optimizer_updates` (see `flush()`'s invariant check
            # below): the two are required to stay in lockstep. The GPU
            # transaction dispatched by `execute_fixed_topology_transaction`
            # below can raise (a real CUDA OOM during backward has been
            # observed in production) after this point but before the
            # optimizer step is recorded via
            # `_record_completed_optimization_result`. Incrementing
            # `self.iteration` here, eagerly, before that transaction
            # completes, previously left it one ahead of the work clock
            # forever whenever the transaction failed -- silently poisoning
            # every later `flush()`/`close()` call for the rest of the
            # runtime's lifetime with a confusing, unrelated
            # "optimizer clock disagrees with runtime iteration" error that
            # masked the real (OOM) failure. Compute the attempted iteration
            # id for use inside this transaction, but only commit it to
            # `self.iteration` once the transaction has actually returned.
            attempted_iteration = self.iteration + 1
            background = (
                torch.rand((3,), device=device)
                if self._runtime_policy.random_background
                else self._background
            )
            sh_interval = max(1, self._runtime_policy.sh_update_interval)
            update_sh_rest = (
                self.gaussians.active_sh_degree == 0
                or sh_interval <= 1
                or optimizer_update_id % sh_interval == 0
            )
            render_depth = any(
                depth_supervision_enabled(
                    work, depth_loss_configured=self._depth_loss_policy.enabled
                )
                or bool(getattr(work.modality_eligibility, "depth_insertion", False))
                for work in transaction_work
            )
            ready_ns = time.monotonic_ns()
            with _phase_timed("m_tick_envelope_build"):
                content_version, topology_version, parameter_version = self._versions()
                builder = FixedTopologyBatchBuilder(
                    max_views=len(transaction_work), max_pending=len(transaction_work)
                )
                execution_mode = (
                    BatchExecutionMode.TRUE_CAMERA_BATCH
                    if constellation.realized_camera_count > 1
                    else BatchExecutionMode.SEQUENTIAL_ACCUMULATION
                )
                for active_work in transaction_work:
                    builder.add(
                        GpuWorkEnvelope(
                            payload=cameras[active_work.source],
                            source_id=active_work.source.fingerprint(),
                            snapshot_id=plan.snapshot_id,
                            map_content_version=content_version,
                            topology_version=topology_version,
                            parameter_version=parameter_version,
                            resolution_bucket=active_work.resolution_tier.value,
                            render_depth=render_depth,
                            update_sh_rest=update_sh_rest,
                            ready_ns=ready_ns,
                            deadline_ns=ready_ns,
                            execution_mode=execution_mode,
                        )
                    )
                transaction = builder.pop_ready(now_ns=ready_ns, force=True)
                if transaction is None:
                    raise RuntimeError("true-camera transaction did not become ready")
                batch_cameras = tuple(
                    cameras[active_work.source] for active_work in transaction_work
                )

            def _run_adaptive_transaction():
                return run_streaming_optimization_transaction(
                    self._optimization_runtime,
                    self._engine,
                    cameras=batch_cameras,
                    requested_batch_size=constellation.requested_batch_size,
                    requested_camera_count=constellation.requested_camera_count,
                    modality_eligibilities=tuple(
                        work.modality_eligibility for work in transaction_work
                    ),
                    background=background,
                    iteration=attempted_iteration,
                    map_version=self.backend.map_version,
                    topology_version=self.backend.topology_version,
                    resolution_bucket=transaction_work[0].resolution_tier.value,
                    update_sh_rest=update_sh_rest,
                    render_depth=render_depth,
                    n_frames_ingested=self.frames_seen,
                    visible_mask=None,
                    random_state_token=RandomStateToken.capture(),
                    optimizer_update_id=optimizer_update_id,
                    optimizer_update_permitted=optimizer_update_permitted,
                )

            def _sequential_executor(_payloads):
                return _run_adaptive_transaction()

            def _true_batch_executor(_payloads):
                return _run_adaptive_transaction()

            with _phase_timed("m_optimization_transaction"):
                optimization_host_prefix_ns = self._begin_optimization_gpu_envelope(
                    optimization_start,
                    transaction_started_ns=optimization_transaction_started_ns,
                )
                result = execute_fixed_topology_transaction(
                    transaction,
                    backend=self.backend,
                    optimization_session=self._engine.session,
                    pipe=self.pipe,
                    cameras=batch_cameras,
                    sequential_executor=_sequential_executor,
                    true_batch_executor=_true_batch_executor,
                    true_batch_support=lambda pipe, selected: (
                        supports_streaming_optimization_transaction(
                            pipe,
                            selected,
                            requested_camera_count=constellation.requested_camera_count,
                        )
                    ),
                )
                self._stamp_completed_result_map_version_after(result)
            # The transaction above completed without raising -- only now is
            # it safe to commit the attempted iteration id, atomically with
            # the work-clock advance immediately below. See the comment at
            # `attempted_iteration`'s assignment for why this must not
            # happen any earlier.
            with _phase_timed("m_tick_post_step"):
                self.iteration = attempted_iteration
                result_completed_ns = time.monotonic_ns()
                completed_clock = self._record_completed_optimization_result(
                    result,
                    mapping_active_ns=max(
                        0, result_completed_ns - mapping_active_checkpoint_ns
                    ),
                )
                self._observe_population_policy_inputs_after_step(
                    self._optimization_runtime,
                    result,
                    self._population_live_surface_source_leases(
                        result,
                        {
                            work.source: cameras[work.source]
                            for work in transaction_work
                        },
                    )
                    if population_live_requested
                    else {},
                    plan,
                )
                post_step_utility = None
                if int(result.optimizer_step_count) > 0:
                    self._maybe_raise_sh_degree(
                        supervised_rgb_pixels=completed_clock.supervised_rgb_pixels
                    )
                    if not population_live_requested:
                        post_step_utility = self._run_due_lifecycle(
                            result,
                            optimizer_updates=completed_clock.optimizer_updates,
                            expected_topology_version=topology_version,
                        )
            with _phase_timed("m_tick_result_bookkeeping"):
                mapping_active_checkpoint_ns = result_completed_ns
                training_result = result
                training_camera = batch_cameras[0]
                training_source = transaction_work[0].source
                training_source_id = transaction_work[0].source.fingerprint()
                self._observe_replay_evidence(result, ready_transaction)
                # Issue #703: accumulate execution-shape supervision counters
                self._supervision_work_clocks_or_create().observe(result)
                result_optimizer_steps = result.optimizer_step_count
                result_supervision_views = result.supervision_view_count
                optimizer_steps += result_optimizer_steps
                supervision_views += result_supervision_views
                rasterizer_calls += int(getattr(result, "rasterizer_call_count", 0))
                supervised_pixels += result.supervised_pixel_count
                realized_microbatch_sizes.extend(
                    int(size) for size in getattr(result, "microbatch_sizes", ())
                )
                successful_indices = tuple(
                    int(index)
                    for index in getattr(
                        result, "successful_supervision_source_indices", ()
                    )
                )
                if successful_indices:
                    successful_index_set = set(successful_indices)
                    if ready_transaction is not None:
                        source_by_camera_uid = {
                            int(camera.uid): source
                            for source, camera in ready_transaction.cameras.items()
                        }
                        unknown_uids = successful_index_set - set(source_by_camera_uid)
                        if unknown_uids:
                            raise RuntimeError(
                                "prepared optimization completed an unknown camera UID"
                            )
                        successful_supervision_sources.update(
                            source_by_camera_uid[uid] for uid in successful_indices
                        )
                    else:
                        successful_supervision_sources.update(
                            active_work.source
                            for active_work in transaction_work
                            if active_work.source.source_ordinal in successful_index_set
                        )
                elif result_supervision_views == len(transaction_work):
                    successful_supervision_sources.update(
                        active_work.source for active_work in transaction_work
                    )
                elif result_supervision_views > 0:
                    unique_supervision_views_known = False
                primary_training_gaussian_count = int(self.gaussians.get_xyz.shape[0])
            with _phase_timed("m_tick_mutation"):
                used_live_population, optimization_mutation = (
                    self._post_step_structural_mutation(
                        result, plan, post_step_utility=post_step_utility
                    )
                )
                if not used_live_population:
                    self._record_completed_mcmc_commit(optimization_mutation)
        else:
            with _phase_timed("m_tick_b1_branch_loop"):
                for update_ordinal, active_work in enumerate(transaction_work):
                    camera = cameras[active_work.source]
                    work_clock = self._work_clock_owner.snapshot()
                    optimizer_update_id = work_clock.optimizer_updates + 1
                    optimizer_update_permitted = _planned_optimizer_update_permitted(
                        plan, update_ordinal=update_ordinal
                    )
                    # See the matching comment in the batch_size > 1 branch above:
                    # commit only after the transaction below returns without
                    # raising, so `self.iteration` can never outrun the
                    # work-clock owner's `optimizer_updates` when a per-view GPU
                    # transaction fails (e.g. a CUDA OOM during backward).
                    attempted_iteration = self.iteration + 1
                    background = (
                        torch.rand((3,), device=device)
                        if self._runtime_policy.random_background
                        else self._background
                    )
                    sh_interval = max(1, self._runtime_policy.sh_update_interval)
                    update_sh_rest = (
                        self.gaussians.active_sh_degree == 0
                        or sh_interval <= 1
                        or optimizer_update_id % sh_interval == 0
                    )
                    depth_weight = self._depth_loss_policy.weight
                    depth_weight = depth_weight if depth_supervision_enabled(
                        active_work,
                        depth_loss_configured=self._depth_loss_policy.enabled,
                    ) else 0.0
                    ready_ns = time.monotonic_ns()
                    content_version, topology_version, parameter_version = self._versions()
                    work = GpuWorkEnvelope(
                        payload=camera,
                        source_id=active_work.source.fingerprint(),
                        snapshot_id=plan.snapshot_id,
                        map_content_version=content_version,
                        topology_version=topology_version,
                        parameter_version=parameter_version,
                        resolution_bucket=active_work.resolution_tier.value,
                        render_depth=(
                            depth_weight > 0.0
                            or bool(getattr(active_work.modality_eligibility, "depth_insertion", False))
                        ),
                        update_sh_rest=update_sh_rest,
                        ready_ns=ready_ns,
                        deadline_ns=ready_ns,
                        execution_mode=BatchExecutionMode.SEQUENTIAL_ACCUMULATION,
                    )
                    builder = FixedTopologyBatchBuilder(max_views=1, max_pending=1)
                    builder.add(work)
                    transaction = builder.pop_ready(now_ns=ready_ns, force=True)
                    if transaction is None:
                        raise RuntimeError("B=1 fixed-topology transaction did not become ready")

                    def _sequential_executor(_payloads):
                        return run_streaming_optimization_step(
                            self._optimization_runtime,
                            self._engine,
                            camera=camera,
                            background=background,
                            iteration=attempted_iteration,
                            map_version=self.backend.map_version,
                            topology_version=self.backend.topology_version,
                            resolution_bucket=active_work.resolution_tier.value,
                            update_sh_rest=update_sh_rest,
                            render_depth=(
                                depth_weight > 0.0
                                or bool(getattr(active_work.modality_eligibility, "depth_insertion", False))
                            ),
                            modality_eligibility=active_work.modality_eligibility,
                            n_frames_ingested=self.frames_seen,
                            visible_mask=None,
                            random_state_token=RandomStateToken.capture(),
                            optimizer_update_id=optimizer_update_id,
                            optimizer_update_permitted=optimizer_update_permitted,
                        )

                    def _true_batch_executor(_payloads):
                        raise RuntimeError("TRUE_CAMERA_BATCH is unavailable for canonical B=1")

                    if optimization_host_prefix_ns is None:
                        optimization_host_prefix_ns = self._begin_optimization_gpu_envelope(
                            optimization_start,
                            transaction_started_ns=optimization_transaction_started_ns,
                        )
                    result = execute_fixed_topology_transaction(
                        transaction,
                        backend=self.backend,
                        optimization_session=self._engine.session,
                        pipe=self.pipe,
                        cameras=(camera,),
                        sequential_executor=_sequential_executor,
                        true_batch_executor=_true_batch_executor,
                    )
                    self._stamp_completed_result_map_version_after(result)
                    # The transaction above completed without raising -- only now
                    # is it safe to commit the attempted iteration id.
                    self.iteration = attempted_iteration
                    result_completed_ns = time.monotonic_ns()
                    completed_clock = self._record_completed_optimization_result(
                        result,
                        mapping_active_ns=max(
                            0, result_completed_ns - mapping_active_checkpoint_ns
                        ),
                    )
                    self._observe_population_policy_inputs_after_step(
                        self._optimization_runtime,
                        result,
                        self._population_live_surface_source_leases(
                            result, {active_work.source: camera}
                        )
                        if population_live_requested
                        else {},
                    )
                    post_step_utility = None
                    if int(result.optimizer_step_count) > 0:
                        self._maybe_raise_sh_degree(
                            supervised_rgb_pixels=completed_clock.supervised_rgb_pixels
                        )
                        if not population_live_requested:
                            post_step_utility = self._run_due_lifecycle(
                                result,
                                optimizer_updates=completed_clock.optimizer_updates,
                                expected_topology_version=topology_version,
                            )
                    mapping_active_checkpoint_ns = result_completed_ns
                    training_result = result
                    training_camera = camera
                    training_source = active_work.source
                    training_source_id = active_work.source.fingerprint()
                    self._observe_replay_evidence(result, ready_transaction)
                    # Issue #703: accumulate execution-shape supervision counters
                    self._supervision_work_clocks_or_create().observe(result)
                    result_optimizer_steps = result.optimizer_step_count
                    result_supervision_views = result.supervision_view_count
                    optimizer_steps += result_optimizer_steps
                    supervision_views += result_supervision_views
                    rasterizer_calls += int(getattr(result, "rasterizer_call_count", 0))
                    supervised_pixels += result.supervised_pixel_count
                    if result_supervision_views > 0:
                        successful_supervision_sources.add(active_work.source)
                        realized_microbatch_sizes.append(result_supervision_views)
            # A B=1 plan may schedule multiple sequential optimizer updates.
            # MutationBudget is per plan/tick, so the sole structural tail runs
            # once after the final fixed-topology update, never once per view.
            with _phase_timed("m_tick_mutation"):
                if training_result is not None:
                    primary_training_gaussian_count = int(
                        self.gaussians.get_xyz.shape[0]
                    )
                    used_live_population, optimization_mutation = (
                        self._post_step_structural_mutation(
                            training_result, plan,
                            post_step_utility=post_step_utility,
                        )
                    )
                    if not used_live_population:
                        self._record_completed_mcmc_commit(optimization_mutation)
        with _phase_timed("m_tick_post_optimization_tail"):
            # Issue #279 follow-up (2026-08-31): sub-brackets below break down
            # m_tick_post_optimization_tail's own 16.3ms/tick into its
            # constituent stages -- see
            # docs/architecture/2026-08-31-issue279-tick-attribution.md.
            with _phase_timed("m_tick_post_optimization_tail_gpu_record_bookkeeping"):
                optimization_end.record()
                optimization_completed_ns = time.monotonic_ns()
                if optimization_host_prefix_ns is None:
                    raise RuntimeError("optimization transaction never entered GPU execution")
                self._work_clock_owner.record_mapping_active(
                    max(0, optimization_completed_ns - mapping_active_checkpoint_ns)
                )
                self.optimizer_steps += optimizer_steps
                self._supervision_views_completed += supervision_views
                self._record_successful_replay_supervision(
                    plan,
                    successful_sources=successful_supervision_sources,
                    unique_sources_known=unique_supervision_views_known,
                )
                if ready_transaction is not None:
                    self._activate_successful_prepared_deferred_sources(
                        successful_supervision_sources
                    )
                if ready_transaction is not None and optimizer_steps > 0 and snapshot is not None:
                    self._last_materialized_work = (snapshot, plan)
            with _phase_timed("m_tick_post_optimization_tail_psnr_sample"):
                primary_training_psnr_db = self._sample_primary_training_psnr(
                    training_result, training_camera
                )
            with _phase_timed("m_tick_post_optimization_tail_spatial_sample"):
                primary_training_spatial = self._sample_primary_training_spatial(
                    training_result, training_camera, training_source
                )
            with _phase_timed("m_tick_post_optimization_tail_gpu_timing_submit_optimization"):
                content, topology, parameters = self._versions()
                optimization_change = (
                    None if optimization_mutation is None else optimization_mutation.change
                )
                optimization_inserted = (
                    0 if optimization_mutation is None else int(optimization_mutation.inserted)
                )
                optimization_pruned = (
                    0 if optimization_change is None else int(optimization_change.pruned)
                )
                optimization_relocated = (
                    0 if optimization_change is None else int(optimization_change.relocated)
                )
                optimization_corrected = (
                    0 if optimization_change is None else int(optimization_change.corrected)
                )
                self._submit_gpu_timing(
                    optimization_start,
                    optimization_end,
                    GpuTelemetryDescriptor(
                        transaction_id=self._next_gpu_transaction_id,
                        snapshot_id=plan.snapshot_id,
                        transaction_type=TransactionType.OPTIMIZATION,
                        map_content_version=content,
                        topology_version=topology,
                        parameter_version=parameters,
                        batch_size=supervision_views,
                        pixels=supervised_pixels,
                        rasterizer_calls=rasterizer_calls,
                        optimizer_updates=optimizer_steps,
                        candidates=0,
                        mutation_type=(
                            (
                                "population_policy"
                                if population_live_requested
                                else "delegated_mcmc"
                            )
                            if optimization_mutation is not None else "none"
                        ),
                    ),
                    inserted=optimization_inserted,
                    pruned=optimization_pruned,
                    relocated=optimization_relocated,
                    corrected=optimization_corrected,
                    optimization_host_prefix_ns=optimization_host_prefix_ns,
                )
                self._next_gpu_transaction_id += 1
            with _phase_timed("m_tick_post_optimization_tail_depth_insertion_proposal"):
                depth_active_started_ns = time.monotonic_ns()
                mutation_proposals = []
                insertion_versions = self._versions()
                insertion_start = None
                insertion_end = None
                candidate_count = 0
                insertion_mutation_type = "surface_insertion"
                depth_mutation_kind = None
                depth_mutation_source = None
                # A true-camera optimization produces render leases for every
                # eligible view, but live depth insertion intentionally
                # remains one proposal for the transaction's planned newest
                # fresh source.  The source fence keeps the single mutation
                # barrier and its transfer attribution unambiguous; other
                # other leases are available only when their own planned
                # insertion boundary is reached.
                if (
                    not population_live_requested
                    and
                    pending_integration is not None
                    and plan.supervision_sources
                    and pending_integration[2]
                    == max(plan.supervision_sources, key=SourceIdentity.ordering_key)
                ):
                    if insertion_versions[1] != plan.topology_version:
                        raise RuntimeError(
                            "topology changed before insertion proposal compute: "
                            f"planned={plan.topology_version} current={insertion_versions[1]}"
                        )
                    insertion_start = torch.cuda.Event(enable_timing=True)
                    insertion_end = torch.cuda.Event(enable_timing=True)
                    insertion_start.record()
                    with _phase_timed(
                        "m_tick_post_optimization_tail_depth_insertion_proposal_live"
                    ):
                        render_reuse, reuse_reason = (
                            self._select_depth_insertion_render_reuse(
                                result=training_result,
                                camera=pending_integration[0],
                                source=pending_integration[2],
                                plan=plan,
                                source_lease=getattr(pending_integration[1], "lease", None),
                            )
                        )
                        if ready_transaction is not None and render_reuse is None:
                            proposal = None
                            prepared_insertion_skip = (
                                f"render_reuse_unavailable:{reuse_reason.value}"
                            )
                        else:
                            depth_mutation_kind = "live_depth"
                            depth_mutation_source = pending_integration[2]
                            proposal = self._prepare_insertion_proposal(
                                pending_integration[0],
                                pending_integration[1],
                                pending_integration[2],
                                plan,
                                render_reuse=render_reuse,
                                render_reuse_fallback=reuse_reason.value,
                            )
                    if proposal is not None:
                        candidate_count = int(proposal.batch.points.shape[0])
                        mutation_proposals.append(proposal)
                if not population_live_requested and plan.replay_insertion is not None:
                    if depth_mutation_kind is not None:
                        raise RuntimeError(
                            "live and replay depth cannot share one attributed mutation barrier"
                        )
                    replay_source = plan.replay_insertion.source
                    if replay_source not in cameras:
                        raise RuntimeError("replay insertion source was not materialized")
                    if insertion_versions[1] != plan.topology_version:
                        raise RuntimeError(
                            "topology changed before replay insertion proposal compute: "
                            f"planned={plan.topology_version} current={insertion_versions[1]}"
                        )
                    insertion_start = torch.cuda.Event(enable_timing=True)
                    insertion_end = torch.cuda.Event(enable_timing=True)
                    insertion_start.record()
                    depth_mutation_kind = "replay_depth"
                    depth_mutation_source = replay_source
                    with _phase_timed(
                        "m_tick_post_optimization_tail_depth_insertion_proposal_replay"
                    ):
                        render_reuse, reuse_reason = (
                            self._select_depth_insertion_render_reuse(
                                result=training_result,
                                camera=cameras[replay_source],
                                source=replay_source,
                                plan=plan,
                            )
                        )
                        proposal = self._prepare_replay_insertion_proposal(
                            cameras[replay_source],
                            replay_source,
                            plan,
                            render_reuse=render_reuse,
                            render_reuse_fallback=reuse_reason.value,
                        )
                    if proposal is not None:
                        candidate_count = int(proposal.batch.points.shape[0])
                        mutation_proposals.append(proposal)
                    insertion_mutation_type = "replay_surface_insertion"
                    self._consume_replay_depth_deadline()
            with _phase_timed("m_tick_post_optimization_tail_mutation_barrier_commit"):
                mutation = self._commit_mutation_barrier(
                    tuple(mutation_proposals),
                    expected_map_content_version=insertion_versions[0],
                    expected_topology_version=insertion_versions[1],
                )
                self._record_completed_depth_mutation(mutation)
                self._work_clock_owner.record_mapping_active(
                    max(0, time.monotonic_ns() - depth_active_started_ns)
                )
            with _phase_timed("m_tick_post_optimization_tail_mutation_events_bookkeeping"):
                mutation_results = tuple(
                    item for item in (optimization_mutation, mutation) if item is not None
                )
                inserted = sum(int(item.inserted) for item in mutation_results)
                pruned = sum(int(item.change.pruned) for item in mutation_results)
                relocated = sum(int(item.change.relocated) for item in mutation_results)
                corrected = sum(int(item.change.corrected) for item in mutation_results)
                mutation_events = []
                if optimization_mutation is not None:
                    mutation_events.append(
                        MutationAttribution(
                            kind=(
                                "population_policy"
                                if population_live_requested else "mcmc"
                            ),
                            inserted=optimization_inserted,
                            pruned=optimization_pruned,
                            relocated=optimization_relocated,
                            corrected=optimization_corrected,
                        )
                    )
                if depth_mutation_kind is not None:
                    depth_change = None if mutation is None else mutation.change
                    mutation_events.append(
                        MutationAttribution(
                            kind=depth_mutation_kind,
                            source=depth_mutation_source,
                            candidates=candidate_count,
                            inserted=0 if mutation is None else int(mutation.inserted),
                            pruned=0 if depth_change is None else int(depth_change.pruned),
                            relocated=(
                                0 if depth_change is None else int(depth_change.relocated)
                            ),
                            corrected=(
                                0 if depth_change is None else int(depth_change.corrected)
                            ),
                        )
                    )
            with _phase_timed("m_tick_post_optimization_tail_gpu_timing_submit_insertion"):
                if insertion_end is not None:
                    insertion_end.record()
                    content, topology, parameters = self._versions()
                    self._submit_gpu_timing(
                        insertion_start,
                        insertion_end,
                        GpuTelemetryDescriptor(
                            transaction_id=self._next_gpu_transaction_id,
                            snapshot_id=plan.snapshot_id,
                            transaction_type=(
                                TransactionType.MAP_MUTATION
                                if mutation is not None and (
                                    int(mutation.inserted)
                                    + int(mutation.change.pruned)
                                    + int(mutation.change.relocated)
                                    + int(mutation.change.corrected)
                                    > 0
                                )
                                else TransactionType.INSERTION_PREP
                            ),
                            map_content_version=content,
                            topology_version=topology,
                            parameter_version=parameters,
                            batch_size=1,
                            pixels=0,
                            rasterizer_calls=int(
                                self.depth_insertion_render_reuse_telemetry().get(
                                    "extra_rasterizer_calls", 1
                                )
                            ),
                            optimizer_updates=0,
                            candidates=candidate_count,
                            mutation_type=(
                                insertion_mutation_type
                                if mutation is not None and int(mutation.inserted) > 0
                                else "none"
                            ),
                        ),
                        inserted=0 if mutation is None else int(mutation.inserted),
                        pruned=0 if mutation is None else int(mutation.change.pruned),
                        relocated=0 if mutation is None else int(mutation.change.relocated),
                        corrected=0 if mutation is None else int(mutation.change.corrected),
                    )
                    self._next_gpu_transaction_id += 1
            with _phase_timed("m_tick_post_optimization_tail_poll_map_activity"):
                self._services.poll_map_activity()
            with _phase_timed("m_tick_post_optimization_tail_latency_bookkeeping"):
                completed_ns = time.monotonic_ns()
                source = transaction_work[0].source
                arrival_ns = self._source_arrival_ns.pop(source.fingerprint(), None)
                if arrival_ns is not None and supervision_views > 0:
                    self._first_supervision_latency_ms.append(
                        max(0.0, float(completed_ns - arrival_ns) / 1e6)
                    )
                if arrival_ns is not None and inserted > 0:
                    self._geometry_commit_latency_ms.append(
                        max(0.0, float(completed_ns - arrival_ns) / 1e6)
                    )
                if optimizer_steps > 0 or inserted + pruned + relocated + corrected > 0:
                    self._map_clock_ns = max(self._map_clock_ns, int(source.timestamp_ns))
            with _phase_timed("m_tick_post_optimization_tail_retire_live_admitted_payloads"):
                self.scene.retire_live_admitted_payloads()
            self.record_successful_supervision_sources(successful_supervision_sources)
            with _phase_timed("m_tick_post_optimization_tail_retire_completed_candidates"):
                self._retire_completed_candidates(successful_supervision_sources)
            self._latest_mapping_step_wall_ms = max(
                0.0, float(time.monotonic_ns() - tick_started_ns) / 1e6
            )
            with _phase_timed("m_tick_post_optimization_tail_record_transaction_cost"):
                self._record_transaction_cost(
                    plan,
                    tick_started_ns=tick_started_ns,
                    prepare_end_ns=optimization_transaction_started_ns + optimization_host_prefix_ns,
                    optimization_end_ns=optimization_completed_ns,
                    transaction_wall_ms=self._latest_mapping_step_wall_ms,
                    cache_hits=supervision_cache_hits,
                    cache_misses=supervision_cache_misses,
                )
            return self._tick_result(
                True,
                "",
                snapshot=snapshot,
                plan=plan,
                optimizer_steps=optimizer_steps,
                supervision_views=supervision_views,
                supervised_pixels=supervised_pixels,
                unique_supervision_views=(
                    len(successful_supervision_sources)
                    if unique_supervision_views_known
                    else None
                ),
                successful_supervision_sources=tuple(successful_supervision_sources),
                microbatch_sizes=tuple(realized_microbatch_sizes),
                insertion_candidates=candidate_count,
                inserted=inserted,
                pruned=pruned,
                relocated=relocated,
                corrected=corrected,
                mutation_events=tuple(mutation_events),
                primary_training_psnr_db=primary_training_psnr_db,
                primary_training_source_id=(
                    training_source_id if primary_training_psnr_db is not None else None
                ),
                primary_training_spatial=primary_training_spatial,
                primary_training_gaussian_count=(
                    primary_training_gaussian_count
                    if primary_training_spatial is not None
                    else None
                ),
                camera_materialize_cache_hits=supervision_cache_hits,
                camera_materialize_cache_misses=supervision_cache_misses,
                prepared_insertion_skip=prepared_insertion_skip,
            )

    def _observe_replay_evidence(self, result: Any, ready_transaction: Any = None) -> None:
        """Stage same-render training feedback; never wait for a ranking decision."""
        if not getattr(self._optimization_policy, "streaming_historical_usefulness_tracking", False):
            return
        self._poll_replay_evidence()
        trace_enabled = bool(getattr(self, "_population_selection_trace_enabled", False))
        plan = getattr(ready_transaction, "plan", None)
        planned_trials = tuple(getattr(plan, "replay_trial_sources", ())) if trace_enabled else ()
        lease_generations = {
            binding.source_identity: getattr(binding.lease, "generation", None)
            for binding in getattr(ready_transaction, "bindings", ())
            if getattr(binding, "source_identity", None) is not None
        } if trace_enabled else {}
        resolution_tier = None
        if trace_enabled:
            resolution_tier = getattr(getattr(plan, "resolution_tier", None), "value", None)
            if resolution_tier is None and plan is not None:
                resolution_tier = str(getattr(plan, "resolution_tier", "")) or None

        def record_completion(
            source: SourceIdentity,
            status: str,
            reason: str,
            snapshot: Any = None,
        ) -> None:
            self._record_replay_challenger_completion(
                source,
                completion=status,
                reason=reason,
                lease_generation=lease_generations.get(source),
                resolution_tier=resolution_tier,
                snapshot=snapshot,
            )

        if int(getattr(result, "optimizer_step_count", 0)) <= 0:
            for source in planned_trials:
                record_completion(source, "not_attempted", "no_optimizer_step")
            return
        from eligsir.optimization.gaussian.supervision_evidence import SupervisionEvidenceBatch

        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, SupervisionEvidenceBatch) or not evidence.source_identities:
            for source in planned_trials:
                record_completion(source, "unavailable", "evidence_missing")
            return
        trial_sources = frozenset(getattr(getattr(ready_transaction, "plan", None), "replay_trial_sources", ()))
        authorized_trials = trial_sources.intersection(evidence.source_identities)
        if authorized_trials:
            from eligsir.runtime.ready_camera_binder import _full_identity_camera_uid

            successful = set(getattr(result, "successful_supervision_source_indices", ()) or ())
            if successful:
                completed = {source for source in evidence.source_identities if _full_identity_camera_uid(source) in successful}
            elif getattr(result, "supervision_view_count", 0) == len(evidence.source_identities):
                completed = set(evidence.source_identities)
            else:
                completed = set()  # Partial execution without attribution abstains.
            for source in sorted(authorized_trials - completed, key=SourceIdentity.ordering_key):
                record_completion(
                    source,
                    "partial",
                    "completion_unattributed",
                    snapshot=evidence.snapshot_token,
                )
            trial_sources = authorized_trials.intersection(completed)
            self._replay_challenger_authorized_trials_total = int(getattr(self, "_replay_challenger_authorized_trials_total", 0)) + len(authorized_trials)
            self._replay_challenger_completed_trials_total = int(getattr(self, "_replay_challenger_completed_trials_total", 0)) + len(trial_sources)
            if trial_sources:
                self._pending_replay_trial_sources = tuple(
                    source
                    for source in getattr(self, "_pending_replay_trial_sources", ())
                    if source not in trial_sources
                )
        if getattr(self, "_pending_replay_evidence", None) is not None:
            self._replay_feedback_busy = int(getattr(self, "_replay_feedback_busy", 0)) + 1
            return
        fields = (
            evidence.rgb_residual_summary, evidence.depth_residual_summary,
            evidence.rgb_valid_pixels, evidence.depth_valid_pixels,
        )
        if any(value is None for value in fields):
            return
        quality_by_source = {
            binding.source_identity: binding.lease.source_metadata.get("replay_quality")
            for binding in getattr(ready_transaction, "bindings", ())
        }
        if not any(quality_by_source.get(source) is not None for source in evidence.source_identities):
            for source in sorted(trial_sources, key=SourceIdentity.ordering_key):
                record_completion(source, "unavailable", "replay_quality_missing")
            return
        # Only immutable CPU metadata is retained. No source pixels or loss graph
        # outlive this call; the detached E0 packet shares the existing slab pool.
        rows = tuple(
            (source, eligibility, quality_by_source.get(source), lease_generations.get(source))
            for source, eligibility in zip(evidence.source_identities, evidence.modality_eligibility)
        )
        packed = torch.stack(tuple(value.detach().float() for value in fields), dim=1)
        try:
            readback = self._transfer_owner.readback_supervision_evidence(packed)
        except RuntimeError as exc:
            if not any(reason in str(exc) for reason in ("PinnedByteSlabPool", "pinned byte limit exceeded")):
                raise
            self._replay_feedback_busy = int(getattr(self, "_replay_feedback_busy", 0)) + 1
            return
        self._pending_replay_evidence = (
            readback,
            rows,
            trial_sources,
            evidence.snapshot_token,
            resolution_tier,
        )

    def _poll_replay_evidence(self, *, allow_turnover: bool = True) -> None:
        """Consume a completed compact packet into the existing modality EMAs."""
        pending = getattr(self, "_pending_replay_evidence", None)
        if pending is None:
            return
        readback, rows, trial_sources = pending[:3]
        evidence_snapshot = pending[3] if len(pending) > 3 else None
        resolution_tier = pending[4] if len(pending) > 4 else None
        values = readback.try_take_cpu_tensor()
        if values is None:
            return
        self._pending_replay_evidence = None
        from eligsir.mapping.scheduler.historical_usefulness import usefulness_weight_from_loss

        loss_scale = float(getattr(self._optimization_policy, "streaming_historical_usefulness_loss_scale", 1.0))
        arena = self._hot_cold_challenger_arena
        trace_enabled = bool(getattr(self, "_population_selection_trace_enabled", False))
        trusted_trials = []
        observation_audit: dict[tuple[SourceIdentity, str], dict[str, Any]] | None = (
            {} if trace_enabled else None
        )
        for item, row in zip(rows, values.reshape(-1, 4).tolist()):
            source, eligibility, quality = item[:3]
            lease_generation = item[3] if len(item) > 3 else None
            valid_modalities = tuple(
                getattr(eligibility, f"{modality}_supervision")
                and math.isfinite(row[index]) and row[index] >= 0
                and math.isfinite(row[index + 2]) and row[index + 2] > 0
                for index, modality in enumerate(("rgb", "depth"))
            )
            frame_id = self._historical_frame_ids_by_source.get(source)
            if frame_id is None or not isinstance(quality, tuple) or len(quality) != 2:
                if trace_enabled and source in trial_sources:
                    self._record_replay_challenger_completion(
                        source,
                        completion="unavailable",
                        reason="source_or_quality_unavailable",
                        lease_generation=lease_generation,
                        resolution_tier=resolution_tier,
                        snapshot=evidence_snapshot,
                    )
                continue
            observed = False
            score_records: dict[str, dict[str, Any]] = {}
            for index, modality in enumerate(("rgb", "depth")):
                q = quality[index]
                residual = row[index]
                if (
                    not valid_modalities[index]
                    or isinstance(q, bool) or not isinstance(q, (int, float))
                    or not math.isfinite(q) or not 0 <= q <= 1
                ):
                    if trace_enabled and source in trial_sources:
                        score_records[modality] = {
                            "completion": "unavailable",
                            "reason": "invalid_quality_or_residual",
                            "q": q if isinstance(q, (int, float)) and not isinstance(q, bool) else None,
                            "residual": residual if math.isfinite(residual) else None,
                            "valid_pixels": int(row[index + 2]) if math.isfinite(row[index + 2]) else None,
                            "trusted": False,
                        }
                    continue
                # Reliability and current-map need are distinct. Unusable measured
                # sources can be demoted but high residual cannot make them win.
                weight = 0.0 if q < 0.5 else q * usefulness_weight_from_loss(residual, loss_scale=loss_scale)
                self.record_modality_usefulness_observation(source, modality, weight)
                observed = True
                if trace_enabled and source in trial_sources:
                    assert observation_audit is not None
                    trusted = bool(arena.modality_trusted(frame_id, modality))
                    trusted_weight = arena.modality_weight(
                        frame_id, modality, default=math.nan
                    )
                    observation_audit[(source, modality)] = {
                        "q": float(q),
                        "residual": float(residual),
                        "valid_pixels": int(row[index + 2]),
                        "trusted": trusted,
                        "trusted_weight": (
                            float(trusted_weight) if trusted and math.isfinite(trusted_weight) else None
                        ),
                        "lease_generation": lease_generation,
                        "resolution_tier": resolution_tier,
                        "snapshot": self._replay_challenger_snapshot_trace(evidence_snapshot),
                    }
                    score_records[modality] = {
                        "completion": "completed",
                        "reason": "same_render_feedback",
                        **observation_audit[(source, modality)],
                    }
            if trace_enabled and source in trial_sources and score_records:
                self._record_replay_challenger_completion(
                    source,
                    completion="completed" if observed else "unavailable",
                    reason="same_render_feedback" if observed else "invalid_quality_or_residual",
                    lease_generation=lease_generation,
                    resolution_tier=resolution_tier,
                    snapshot=evidence_snapshot,
                    scores=score_records,
                )
            if not observed:
                continue
            self._historical_usefulness_observations_total = int(getattr(self, "_historical_usefulness_observations_total", 0)) + 1
            trusted = [
                arena.modality_weight(frame_id, modality)
                for modality in ("rgb", "depth")
                if arena.modality_trusted(frame_id, modality)
            ]
            if trusted and self._historical_candidate_index.update_base_weight(frame_id, max(0.05, max(trusted))):
                self._historical_usefulness_trusted_updates_total = int(getattr(self, "_historical_usefulness_trusted_updates_total", 0)) + 1
            if trusted and source in trial_sources:
                trusted_trials.append(source)
        if allow_turnover and trusted_trials:
            if bool(getattr(self, "_population_selection_trace_enabled", False)):
                self._actuate_replay_challenger_turnover(
                    tuple(trusted_trials),
                    observation_audit=observation_audit,
                    evidence_snapshot=evidence_snapshot,
                )
            else:
                self._actuate_replay_challenger_turnover(tuple(trusted_trials))

    def _sample_primary_training_psnr(
        self, result: Any, camera: Any
    ) -> float | None:
        """Materialize one exact pre-update primary-view PSNR at low cadence."""

        if not (
            bool(
                getattr(
                    self._optimization_policy,
                    "streaming_track_primary_psnr",
                    False,
                )
            )
            or self._partition_replay_focus_enabled
        ):
            return None
        if not self._primary_psnr_due() or result is None or camera is None:
            return None
        self._consume_primary_psnr_deadline()
        render_outputs = getattr(result, "render_outputs", None)
        payload = getattr(render_outputs, "payload", None)
        if not isinstance(payload, dict) or "render" not in payload:
            raise RuntimeError(
                "primary training PSNR requested but optimization returned no render"
            )
        from eligsir.utils.score_contract import (
            apply_declared_score_clamp,
            require_canonical_score_tensor,
            score_tensor_from_camera,
        )

        with torch.no_grad():
            render_image = payload["render"].detach()
            gt_image = score_tensor_from_camera(
                camera, device=render_image.device
            )
            render_score, _ = apply_declared_score_clamp(
                render_image,
                context="canonical_primary_training_psnr",
            )
            gt_score, _ = require_canonical_score_tensor(
                gt_image, context="canonical_primary_training_psnr:gt"
            )
            if render_score.shape != gt_score.shape:
                raise RuntimeError(
                    "primary training render/GT shapes disagree: "
                    f"{tuple(render_score.shape)} != {tuple(gt_score.shape)}"
                )
            # This is deliberately a diagnostic score on the realized
            # supervision grid, not a frozen-holdout scientific metric. GT can
            # be area-resampled while the renderer rasterizes directly at that
            # grid, so their transform lineages are intentionally different.
            mse = ((render_score - gt_score) ** 2).reshape(
                render_score.shape[0], -1
            ).mean(dim=1)
            return float((20.0 * torch.log10(1.0 / torch.sqrt(mse))).mean().item())

    def _sample_primary_training_spatial(
        self, result: Any, camera: Any, source: Any
    ) -> Any:
        """Attach training RGB error to fixed sensor-depth world microcells."""

        if (
            not self._spatial_quality_due()
            or result is None
            or camera is None
            or source is None
        ):
            return None
        self._consume_spatial_quality_deadline()
        render_outputs = getattr(result, "render_outputs", None)
        payload = getattr(render_outputs, "payload", None)
        if not isinstance(payload, dict) or "render" not in payload:
            raise RuntimeError(
                "spatial replay focus requested but optimization returned no render"
            )
        from eligsir.mapping.scheduler.partition_replay_focus import (
            spatial_microcell_quality_from_render,
        )
        from eligsir.utils.comparison_report import _get_sensor_depth
        from eligsir.utils.score_contract import (
            apply_declared_score_clamp,
            require_canonical_score_tensor,
            score_tensor_from_camera,
        )

        with torch.no_grad():
            render_image = payload["render"].detach()
            gt_image = score_tensor_from_camera(camera, device=render_image.device)
            render_score, _ = apply_declared_score_clamp(
                render_image,
                context="canonical_primary_training_spatial",
            )
            gt_score, _ = require_canonical_score_tensor(
                gt_image, context="canonical_primary_training_spatial:gt"
            )
            if render_score.shape != gt_score.shape:
                raise RuntimeError(
                    "spatial training render/GT shapes disagree: "
                    f"{tuple(render_score.shape)} != {tuple(gt_score.shape)}"
                )
            sensor_depth = _get_sensor_depth(
                camera, int(render_score.shape[-2]), int(render_score.shape[-1])
            )
            if sensor_depth is None:
                return None
            return spatial_microcell_quality_from_render(
                source=source,
                camera=camera,
                tracker=self._spatial_quality_tracker,
                render_image=render_score,
                gt_image=gt_score,
                sensor_depth=sensor_depth,
            )

    def _camera_for_source(
        self,
        source,
        *,
        current_snapshot: ControlSnapshot | None = None,
        resident_supervision: Any | None = None,
    ) -> Any:
        """Resolve the exact planned identity; never substitute a random view."""

        from eligsir.utils.partition_eval import camera_source_index

        assert self.scene is not None
        admitted_lookup = getattr(self.scene, "admitted_camera_for_source", None)
        if callable(admitted_lookup):
            camera = admitted_lookup(source)
            matches = [] if camera is None else [camera]
        else:
            cameras = list(self.scene.train_cameras)
            current = getattr(self.scene, "current_camera", None)
            if current is not None and current not in cameras:
                cameras.append(current)
            matches = []
            for camera in cameras:
                attached = getattr(getattr(camera, "_mapping_frame", None), "source", None)
                retained = getattr(camera, "_source_identity", None)
                source_index = camera_source_index(camera)
                registered = (
                    None
                    if source_index is None
                    else self._source_identities_by_ordinal.get(source_index)
                )
                if (
                    attached == source
                    or retained == source
                    or (attached is None and registered == source)
                ):
                    matches.append(camera)
        if not matches:
            # Deferred observations have no camera until a current work plan
            # selects them. The DB record is the durable semantic authority:
            # REJECT never creates an eligible record, so this cannot
            # resurrect a terminally discarded source after the owner's
            # bounded idempotency ledger has rotated.
            observation_db = getattr(self, "_observation_db", None)
            candidate = (
                None
                if observation_db is None
                else observation_db.historical_candidate_for_source(
                    source, require_rgb=True
                )
            )
            print(
                f"[DIAG-no_training_camera] _camera_for_source: source={source!r} "
                f"admitted_lookup_hit=False observation_db_present={observation_db is not None} "
                f"historical_candidate_found={candidate is not None} "
                f"resident_supervision_present={resident_supervision is not None}",
                flush=True,
            )
            if candidate is not None:
                if current_snapshot is None:
                    raise RuntimeError(
                        "historical promotion requires the current control snapshot"
                    )
                disposition_snapshot_id = candidate.disposition_snapshot_id
                if disposition_snapshot_id is None:
                    raise RuntimeError(
                        "historical source has no durable disposition provenance"
                    )
                permit = self.observation_owner.promotion_permit_from_retained(
                    source=source,
                    disposition_snapshot_id=disposition_snapshot_id,
                    semantic_action=str(candidate.semantic_action),
                    current_snapshot=current_snapshot,
                )
                if permit is None:
                    raise RuntimeError("historical source is not promotion eligible")
                if (
                    permit.source != source
                    or permit.promotion_snapshot_id != current_snapshot.snapshot_id
                    or permit.topology_version != current_snapshot.topology_version
                ):
                    raise RuntimeError("historical promotion permit fence mismatch")
                if resident_supervision is not None:
                    if (
                        resident_supervision.source_identity != source.fingerprint()
                    ):
                        raise RuntimeError(
                            "resident supervision source identity mismatch"
                        )
                    camera = self.scene.promote_resident_historical_frame(
                        resident_supervision
                    )
                    camera._source_identity = source
                else:
                    if observation_db.prepared_source_reference_for_source(source) is not None:
                        raise RuntimeError(
                            "prepared historical source lost its admitted ready camera; "
                            "cold materialization is forbidden"
                        )
                    mapping = self._replay_mapping_frame(source)
                    if mapping is None:
                        # _replay_mapping_frame's own "still decoding" branch
                        # (a `prepared_cold_disk_rgbd_v1` reload token whose
                        # background admission-prefetch is in flight) returns
                        # None as a SOFT "not ready yet" signal, exactly like
                        # _restore_camera_payload_from_cold's identical check
                        # a few hundred lines up -- that call site retries via
                        # cold_payload_pending. This one instead raised
                        # unconditionally, turning the very first deferred
                        # promotion whose prefetch simply hadn't completed
                        # yet into a hard crash (observed live on
                        # causal_replay/replica, 2026-09-09: source
                        # 9b36e110344b33dc76300cbd77e2c99a, 9 optimizer steps
                        # in). Mirror the other call site's retry-once
                        # pattern before treating this as terminal.
                        frame_id = self._observation_db.frame_id_for_source(source)
                        cold = (
                            None if frame_id is None
                            else self._observation_db.get_cold_pixels(frame_id)
                        )
                        token = (
                            None if cold is None
                            else getattr(cold, "reload_token", None)
                        )
                        if getattr(token, "provider", None) == "prepared_cold_disk_rgbd_v1":
                            with self._admission_prefetch_lock:
                                prefetch_in_flight = (
                                    frame_id in self._admission_prefetch_pending
                                )
                            if prefetch_in_flight:
                                return None
                            mapping = self._replay_mapping_frame(source)
                        if mapping is None:
                            raise RuntimeError(
                                "selected deferred source has no materializable cold pixels: "
                                f"{source.fingerprint()}"
                            )
                    from eligsir.runtime.streaming_observation import (
                        StreamingObservation,
                    )

                    camera = self.scene.promote_historical_frame(
                        StreamingObservation.from_mapping_frame(mapping)
                    )
                # DEFER is not immediate mapping admission, but its explicit
                # later promotion is the transition into ACTIVE supervision.
                # Advance the admission frontier exactly once here so replay
                # work can never publish map > admitted telemetry.
                self._admitted_clock_ns = max(
                    int(getattr(self, "_admitted_clock_ns", 0)),
                    int(source.timestamp_ns),
                )
                clocks = getattr(
                    getattr(self, "_services", None), "four_clocks", None
                )
                if clocks is not None:
                    clocks.record_admitted_keyframe()
                completed_updates = (
                    self._work_clock_owner.snapshot().optimizer_updates
                )
                self._latest_admitted_optimizer_update = completed_updates
                self._optimization_runtime.note_admitted_source(
                    optimizer_updates_completed=completed_updates
                )
                matches.append(camera)
            elif observation_db is not None:
                # `candidate` above (require_rgb=True) found nothing, but the
                # DB still knows this frame_id -- its pixels are not gone,
                # its frozen modality just cannot satisfy RGB supervision
                # (issue #1302/#1405, not #1289's eviction case below).
                # Retiring it here is what keeps this from becoming a
                # permanent no-progress stall instead of aborting the worker.
                frame_id = observation_db.frame_id_for_source(source)
                if frame_id is not None:
                    # `candidate` above already ran the exact require_rgb=True
                    # query this function's only promotion path can ever use
                    # (a few lines up) and found nothing -- so `candidate is
                    # None` here. issue #1302 (kitchen1): a durably RGB-
                    # ineligible-but-depth-eligible historical record (a
                    # no-motion DEFER whose ModalityEligibility is frozen at
                    # record time -- see ObservationDisposition/FrameRecord,
                    # never mutated afterward) used to survive here forever,
                    # because the retirement check below queried
                    # historical_candidate_for_source WITHOUT require_rgb and
                    # found "some" usable modality (depth), so it never
                    # retired. But depth-only eligibility can never satisfy
                    # this function's hardcoded require_rgb=True promotion
                    # query above -- there is no depth-only promotion path --
                    # so that identity was permanently un-promotable AND
                    # permanently un-retired: the planner kept reproposing the
                    # exact same doomed identity every tick, aborting the
                    # supervision transaction with "no_training_camera"
                    # forever (reproduced on a real kitchen1 bag replay:
                    # optimizer_steps stuck at 0 for the entire run). Retire
                    # whenever the SAME criterion promotion needs (require_rgb)
                    # fails, not a laxer one -- modality eligibility is frozen,
                    # so a require_rgb=True miss now is a miss for all time.
                    self._retire_historical_identity(
                        frame_id, reason="rgb_supervision_ineligible"
                    )
                    return None
                # issue #1302: the planner named an identity the observation
                # DB has never recorded at all (frame_id_for_source returns
                # None) -- a genuinely different gap from #1289's eviction
                # case above, where the DB at least *knew* about the frame.
                # Observed on a real floor2 bag replay under pathologically
                # sparse sync-node admission (11 pairs matched over a 145s
                # bag): a stale/speculative plan can name a source that was
                # never (or not yet) registered via
                # `_register_historical_record`. There is no frame_id to
                # retire, so there is nothing durable to do beyond skipping
                # this transaction -- but skipping, not raising, is what
                # keeps a single unlucky plan entry from aborting the whole
                # mapping worker.
                return None
        if len(matches) != 1:
            raise RuntimeError(
                "planned supervision identity did not resolve to exactly one camera: "
                f"{source.fingerprint()}"
            )
        return matches[0]

    def _note_scene_camera_page_in(self, source: Any, camera: Any) -> None:
        """Schedule page-out only for cameras whose pixels the scene owns.

        Transaction-local historical cameras borrow device tensors owned by
        ``FrameObservationDB``. Paging those temporary wrappers out copies the
        tensor to host without releasing the durable HOT cache owner.
        """
        assert self.scene is not None
        if self.scene.admitted_camera_for_source(source) is camera:
            self._residency_ledger.note_paged_in(camera)

    @staticmethod
    def _bind_resident_supervision(camera, source, tier, resident) -> None:
        """Project one DB-validated HOT tier onto a transaction camera."""

        if resident.source_identity != source.fingerprint():
            raise RuntimeError("resident supervision source identity mismatch")
        if resident.tier != tier.value:
            raise RuntimeError("resident supervision tier mismatch")
        retained = getattr(camera, "_source_identity", None)
        if retained is not None and retained != source:
            raise RuntimeError("resident supervision camera identity mismatch")
        canonical_native = getattr(
            camera, "_canonical_native_resolution_state", None
        )
        width, height, fx, fy, cx, cy = tuple(resident.intrinsics)
        camera._source_identity = source
        camera.original_image = resident.rgb
        camera._sensor_depth_cache = resident.depth
        camera.image_width = camera.width = int(width)
        camera.image_height = camera.height = int(height)
        camera.fx = float(fx)
        camera.fy = float(fy)
        camera.cx = float(cx)
        camera.cy = float(cy)
        camera._canonical_native_resolution_state = canonical_native
        camera._canonical_resolution_binding_native_state = canonical_native
        camera._canonical_resolution_tier = tier
        camera.gsplat = None
        camera._live_payload_retired = False
        if resident.score_provenance is not None:
            from eligsir.utils.score_contract import attach_score_provenance

            camera.score_provenance = resident.score_provenance
            attach_score_provenance(camera.original_image, resident.score_provenance)

    def _maybe_raise_sh_degree(self, *, supervised_rgb_pixels: int) -> None:
        """Consume every completed RGB-exposure threshold exactly once."""

        if type(supervised_rgb_pixels) is not int or supervised_rgb_pixels < 0:
            raise ValueError("supervised_rgb_pixels must be a non-negative exact integer")
        schedule = self._runtime_policy.sh_degree_schedule_rgb_pixels
        consumed = int(getattr(self, "_sh_degree_thresholds_consumed", 0))
        while consumed < len(schedule) and supervised_rgb_pixels >= schedule[consumed]:
            self.gaussians.oneupSHdegree()
            consumed += 1
        self._sh_degree_thresholds_consumed = consumed

    def _compute_post_step_utility(
        self,
        result: Any,
        *,
        optimizer_updates: int,
        expected_topology_version: int,
    ) -> Any | None:
        """Compute lifecycle/MCMC utility behind the completed-update fence."""

        if type(optimizer_updates) is not int or optimizer_updates <= 0:
            raise ValueError("post-step utility requires a positive optimizer-update clock")
        if type(expected_topology_version) is not int or expected_topology_version < 0:
            raise ValueError("expected_topology_version must be a non-negative exact integer")
        current_topology_version = int(self._versions()[1])
        if current_topology_version != expected_topology_version:
            raise RuntimeError(
                "stale topology before lifecycle utility compute: "
                f"planned={expected_topology_version} current={current_topology_version}"
            )
        if not self._use_energy_mcmc:
            return None

        args = self._optimization_policy
        depth_weight = float(getattr(args, "energy_w_depth_residual", 0.0))
        if depth_weight != 0.0:
            raise RuntimeError(
                "canonical post-step energy MCMC has not qualified depth-residual utility"
            )
        render_outputs = getattr(result, "render_outputs", None)
        render_pkg = getattr(render_outputs, "payload", None)
        if not isinstance(render_pkg, dict):
            render_pkg = {}
        from eligsir.utils.energy_mcmc import compute_gaussian_utility

        evidence = getattr(result, "evidence", None)
        return compute_gaussian_utility(
            gaussians=self.gaussians,
            render_pkg=render_pkg,
            iteration=optimizer_updates,
            grad_overrides=getattr(evidence, "payload", None),
            w_alpha=args.energy_w_alpha,
            w_vis=args.energy_w_vis,
            w_grad=args.energy_w_grad,
            w_scale=args.energy_w_scale,
            w_dead=args.energy_w_dead,
            w_support=getattr(args, "energy_w_support", 2.0),
            beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
            beta_scale=getattr(args, "energy_beta_scale", 0.5),
            alpha_dead=args.energy_alpha_dead,
            w_aniso_utility=float(getattr(args, "utility_aniso_weight", 0.1)),
            aniso_soft_limit=float(
                getattr(args, "utility_aniso_soft_limit", 10.0)
            ),
            args=args,
            w_depth_residual=0.0,
            depth_residual_scalar=0.0,
        )

    def _lifecycle_due(self, optimizer_updates: int) -> bool:
        """Return whether one bounded full-map lifecycle scan is due."""

        if type(optimizer_updates) is not int or optimizer_updates < 0:
            raise ValueError("optimizer_updates must be a non-negative exact integer")
        return optimizer_updates >= int(self._next_lifecycle_optimizer_update)

    def _consume_lifecycle_deadline(self, optimizer_updates: int) -> None:
        """Arm one future optimizer-update deadline without catch-up bursts."""

        if type(optimizer_updates) is not int or optimizer_updates < 0:
            raise ValueError("optimizer_updates must be a non-negative exact integer")
        self._next_lifecycle_optimizer_update = (
            optimizer_updates + int(self._lifecycle_interval_updates)
        )

    def _run_due_lifecycle(
        self,
        result: Any,
        *,
        optimizer_updates: int,
        expected_topology_version: int,
    ) -> Any | None:
        """Run at most one due lifecycle scan and return reusable utility."""

        policy = self._optimization_policy
        if not bool(policy.streaming_lifecycle_enabled):
            return None
        if not self._lifecycle_due(optimizer_updates):
            return None
        utility = None
        if float(policy.streaming_mature_min_utility) > 0.0:
            utility = self._compute_post_step_utility(
                result,
                optimizer_updates=optimizer_updates,
                expected_topology_version=expected_topology_version,
            )
        self._advance_lifecycle(
            utility=utility,
            optimizer_updates=optimizer_updates,
            expected_topology_version=expected_topology_version,
        )
        self._consume_lifecycle_deadline(optimizer_updates)
        return utility

    def _advance_lifecycle(
        self,
        *,
        utility: Any | None,
        optimizer_updates: int,
        expected_topology_version: int,
    ) -> None:
        """Promote supported geometry on completed optimizer-update age."""

        policy = self._optimization_policy
        if not bool(getattr(policy, "streaming_lifecycle_enabled", True)):
            return
        if type(optimizer_updates) is not int or optimizer_updates < 0:
            raise ValueError("optimizer_updates must be a non-negative exact integer")
        if type(expected_topology_version) is not int or expected_topology_version < 0:
            raise ValueError("expected_topology_version must be a non-negative exact integer")
        current_topology_version = int(self._versions()[1])
        if current_topology_version != expected_topology_version:
            raise RuntimeError(
                "stale topology before lifecycle: "
                f"planned={expected_topology_version} current={current_topology_version}"
            )
        self._lifecycle_transition_calls = int(
            getattr(self, "_lifecycle_transition_calls", 0)
        ) + 1
        gaussians = self.gaussians
        if not all(
            hasattr(gaussians, name)
            for name in ("lifecycle_state", "birth_iter", "provisional")
        ):
            raise RuntimeError(
                "canonical lifecycle requires lifecycle_state, birth_iter, and provisional buffers"
            )
        lifecycle = gaussians.lifecycle_state
        if lifecycle.numel() == 0:
            return
        if lifecycle.shape != gaussians.birth_iter.shape:
            raise RuntimeError("canonical lifecycle birth_iter shape mismatch")
        if lifecycle.shape != gaussians.provisional.shape:
            raise RuntimeError("canonical lifecycle provisional shape mismatch")

        utility_ema = getattr(gaussians, "utility_ema", None)
        minimum_utility = float(
            getattr(policy, "streaming_mature_min_utility", 0.1)
        )
        finite_utility_mask = None
        if utility is not None:
            if not hasattr(utility, "isfinite") or utility.shape != lifecycle.shape:
                raise ValueError("lifecycle utility must be a finite per-Gaussian tensor")
            # issue #1259: a rare upstream non-finite utility entry (e.g. a
            # squared scale-penalty or aniso-ratio overflow on one degenerate
            # Gaussian -- see compute_gaussian_utility's own diagnostics for
            # which) used to hard-crash the whole run here. That is
            # over-broad: one bad Gaussian is a per-Gaussian data problem, not
            # evidence the WHOLE population's lifecycle state is corrupt.
            # Exclude non-finite entries from this scan (EMA update +
            # promotion) instead of raising, mirroring the existing
            # _add_guarded/_nan_term_report pattern for loss terms
            # (eligsir/mapping/integration/loss_diagnostics.py). Excluded
            # entries simply keep their prior utility_ema/lifecycle state and
            # are reconsidered on the next scan -- never silently promoted
            # into a wrong value. Only raise when EVERY entry is non-finite,
            # which is real, whole-population corruption a per-Gaussian mask
            # cannot meaningfully repair.
            finite_utility_mask = utility.isfinite()
            n_total = int(finite_utility_mask.numel())
            n_nonfinite = n_total - int(finite_utility_mask.sum().item())
            if n_nonfinite:
                if n_nonfinite == n_total:
                    raise ValueError(
                        "lifecycle utility must be a finite per-Gaussian tensor"
                    )
                _lifecycle_nonfinite_utility_report(
                    n_nonfinite, n_total, optimizer_updates
                )
            if utility_ema is None or utility_ema.shape != lifecycle.shape:
                raise RuntimeError("canonical lifecycle utility_ema shape mismatch")
        elif minimum_utility > 0.0:
            raise RuntimeError(
                "positive lifecycle utility threshold requires post-step utility"
            )

        with torch.no_grad():
            if utility is not None:
                beta = float(getattr(policy, "streaming_utility_ema_beta", 0.95))
                safe_utility = utility.detach().to(utility_ema).clamp(min=0.0)
                if finite_utility_mask is not None and not bool(
                    finite_utility_mask.all()
                ):
                    update_mask = finite_utility_mask.to(device=utility_ema.device)
                    utility_ema[update_mask] = utility_ema[update_mask].mul(beta).add(
                        safe_utility[update_mask], alpha=1.0 - beta
                    )
                else:
                    utility_ema.mul_(beta).add_(safe_utility, alpha=1.0 - beta)

            support_count = getattr(gaussians, "support_count", None)
            if support_count is not None and support_count.shape == lifecycle.shape:
                confirmed = gaussians.provisional & (
                    support_count
                    >= int(getattr(policy, "streaming_min_support_views", 2))
                )
                gaussians.provisional[confirmed] = False
                lifecycle[confirmed] = 1

            newly_young = (~gaussians.provisional) & (lifecycle == 0)
            lifecycle[newly_young] = 1
            update_age = (
                optimizer_updates - gaussians.birth_iter.long()
            ).clamp(min=0)
            young = lifecycle == 1
            if utility_ema is None or minimum_utility <= 0.0:
                utility_ok = young | ~young
            else:
                utility_ok = utility_ema >= minimum_utility
            mature = young & (
                update_age
                >= int(getattr(policy, "streaming_mature_age_frames", 15))
            ) & utility_ok
            lifecycle[mature] = 2

            freeze_age = int(getattr(policy, "streaming_freeze_age_frames", -1))
            if freeze_age >= 0:
                frozen = (lifecycle == 2) & (update_age >= freeze_age)
                lifecycle[frozen] = 3

    def _mutate_after_step(
        self,
        result: Any,
        plan: MappingWorkPlan,
        *,
        post_step_utility: Any | None = None,
    ):
        if (
            not bool(getattr(result, "backward_ran", False))
            or plan.mutation_budget.max_total <= 0
            or not (plan.allow_growth or plan.allow_relocation)
        ):
            return
        if (
            plan.supervision_sources
            or plan.insertion.enabled
            or plan.replay_insertion is not None
        ):
            raise RuntimeError(
                "adaptive MCMC is replay-only and cannot share a depth-insertion mutation plan"
            )
        if plan.mutation_budget.max_total != 1:
            raise RuntimeError("canonical MCMC permits exactly one mutation epoch")
        from eligsir.utils.mcmc_schedule import get_mcmc_schedule

        versions = self._versions()
        source_complete = getattr(self, "_source_complete", False)
        if type(source_complete) is not bool:
            raise TypeError("source_complete phase authority must be exact bool")
        cadence_owner = getattr(self, "_mcmc_cadence_owner", None)
        cadence_decision = None
        if cadence_owner is not None:
            if versions[1] != plan.topology_version:
                raise RuntimeError(
                    "stale topology before MCMC utility compute: "
                    f"planned={plan.topology_version} current={versions[1]}"
                )
            work_clocks = self._work_clock_owner.snapshot()
            behavior_update_id = work_clocks.optimizer_updates
            cadence_decision = cadence_owner.decide(
                work_clocks,
                topology_version=versions[1],
                growth_allowed=plan.allow_growth,
                relocation_allowed=plan.allow_relocation,
            )
            schedule = get_mcmc_schedule(
                work_clocks.optimizer_updates,
                int(self.gaussians.get_xyz.shape[0]),
                self._runtime_policy.cap_max,
                self._mcmc_cfg,
                ingestion_complete=source_complete,
                grow_due=cadence_decision.grow_due,
                relocate_due=cadence_decision.relocate_due,
                target_deficit_enabled=source_complete,
            )
        else:
            # Narrow CPU test/legacy seam. A composed canonical runtime always
            # constructs the cadence owner during initialization.
            if hasattr(self, "config"):
                raise RuntimeError("canonical runtime is missing its MCMC cadence owner")
            behavior_update_id = self.iteration
            schedule = get_mcmc_schedule(
                self.iteration,
                int(self.gaussians.get_xyz.shape[0]),
                self._runtime_policy.cap_max,
                self._mcmc_cfg,
                ingestion_complete=source_complete,
                target_deficit_enabled=source_complete,
            )
        schedule["allow_growth"] = bool(
            schedule["allow_growth"] and plan.allow_growth
        )
        schedule["allow_relocation"] = bool(
            schedule["allow_relocation"] and plan.allow_relocation
        )
        # Position noise is a global content mutation, not a topology-cleanup
        # operation.  During live ingestion it can perturb a large provisional
        # population immediately before the mapping endpoint.  The finite
        # source boundary explicitly opens the unconstrained refinement phase.
        schedule["allow_position_noise"] = source_complete
        if not (schedule["allow_growth"] or schedule["allow_relocation"]):
            self._vram_growth_braked = False
            return None
        vram_braked = False
        vram_fraction = float(
            getattr(
                self._optimization_policy,
                "streaming_vram_brake_fraction",
                0.0,
            )
            or 0.0
        )
        if (
            schedule["allow_growth"]
            and vram_fraction > 0.0
            and torch.cuda.is_available()
        ):
            reserved = int(torch.cuda.memory_reserved())
            total = int(torch.cuda.get_device_properties(0).total_memory)
            vram_braked = reserved / max(1, total) >= vram_fraction
            if vram_braked:
                schedule["allow_growth"] = False
        self._vram_growth_braked = vram_braked
        if not (schedule["allow_growth"] or schedule["allow_relocation"]):
            return None

        utility = post_step_utility
        temperature = 1.0
        render_outputs = getattr(result, "render_outputs", None)
        render_pkg = getattr(render_outputs, "payload", None)
        if not isinstance(render_pkg, dict):
            render_pkg = {}
        if self._use_energy_mcmc and utility is None:
            utility = self._compute_post_step_utility(
                result,
                optimizer_updates=behavior_update_id,
                expected_topology_version=versions[1],
            )
        if self._use_energy_mcmc:
            args = self._optimization_policy
            from eligsir.utils.strategies.mcmc.proposal import (
                temperature_from_optimizer_updates,
            )

            temperature = temperature_from_optimizer_updates(
                args, optimizer_updates=behavior_update_id
            )

        builder = getattr(
            self._mcmc_strategy, "delegated_post_backward_proposal", None
        )
        if not callable(builder):
            raise TypeError(
                "canonical MCMC strategy must build a delegated mutation proposal"
            )
        proposal = builder(
            gaussians=self.gaussians,
            args=self._optimization_policy,
            sched=schedule,
            iteration=behavior_update_id,
            optimizer_updates=behavior_update_id,
            utility=utility,
            temperature=temperature,
            use_energy_mcmc=self._use_energy_mcmc,
            tb_writer=None,
            should_log_strategy=lambda _iteration: False,
            render_pkg=render_pkg,
            lr=getattr(result, "xyz_lr", None),
        )
        commit = self._commit_mutation_barrier(
            (proposal,),
            expected_map_content_version=versions[0],
            expected_topology_version=versions[1],
            allow_delegated=True,
        )
        if (
            cadence_owner is not None
            and cadence_decision is not None
            and commit is not None
            and int(commit.change.topology_version) > versions[1]
        ):
            cadence_owner.record_structural_mutation(
                self._work_clock_owner.snapshot(),
                topology_version_after=int(commit.change.topology_version),
                grew=int(commit.inserted) > 0,
                relocated=int(commit.change.relocated) > 0,
            )
        return commit

    def flush(self) -> None:
        if not self.initialized:
            return None
        from eligsir.runtime.artifact_paths import (
            point_cloud_artifact_paths,
            publish_legacy_iteration_alias,
        )

        work = self._work_clock_owner.snapshot()
        if int(work.optimizer_updates) != int(self.iteration):
            raise RuntimeError(
                "point-cloud snapshot optimizer clock disagrees with runtime iteration"
            )
        paths = point_cloud_artifact_paths(
            self._runtime_policy.model_path,
            optimizer_updates=int(work.optimizer_updates),
            supervised_views=int(work.supervised_rgb_views),
        )
        self.gaussians.save_ply(str(paths.point_cloud))
        publish_legacy_iteration_alias(paths)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._poll_replay_evidence(allow_turnover=False)
        pending = getattr(self, "_pending_replay_evidence", None)
        if pending is not None:
            pending[0].discard()
            self._pending_replay_evidence = None
        self._pending_replay_trial_sources = ()
        # Issue #1664: resolution-tier usage telemetry is buffered in memory
        # (see resolution_tier_usage.py) rather than appended to disk on
        # every completed transaction; flush it unconditionally here so a
        # normal close (or an exception path that reaches this via
        # ``MappingApplication.close()``'s cleanup chain) never drops rows
        # accumulated since the last cadence flush. Best-effort: mirrors
        # ``record_resolution_tier_usage``'s own silent-on-failure contract.
        model_path = getattr(getattr(self, "_runtime_policy", None), "model_path", None)
        if model_path:
            from eligsir.mapping.scheduler.resolution_tier_usage import (
                flush_resolution_tier_usage,
            )

            flush_resolution_tier_usage(model_path)
        if bool(
            getattr(
                self._optimization_policy,
                "streaming_historical_usefulness_tracking",
                False,
            )
        ):
            # Issue #1308: loud, grep-able wiring-verification line -- see
            # historical_usefulness_telemetry()'s docstring for why
            # observations_total==0 or trusted_updates_total==0 matters.
            telemetry = self.historical_usefulness_telemetry()
            print(
                "[historical-usefulness] "
                f"observations_total={telemetry['observations_total']} "
                f"trusted_updates_total={telemetry['trusted_updates_total']}",
                flush=True,
            )
        self._closed = True
        if self._admission_prefetch_executor is not None:
            # Bounded, single-worker, tiny-job pool -- waiting here is a few
            # milliseconds at most and guarantees no decode thread outlives
            # this runtime (avoids interpreter-shutdown warnings and a
            # dangling reference into a cold-payload cache that is about to
            # be torn down).
            self._admission_prefetch_executor.shutdown(wait=True)
            with self._admission_prefetch_lock:
                self._admission_prefetch_cache.clear()
        # Grep-able, ALWAYS-printed line (same idiom as the
        # `[historical-usefulness]` line above): `scheduled=0` on a run
        # self-diagnoses that the prefetch never had anything to do on that
        # ingestion path (e.g. causal_replay/tum_rgbd, whose cold pixels are
        # already resident by admission time -- see the `__init__` comment
        # above `_admission_prefetch_enabled`), and `prematerialized=N` does
        # the same for admission-time tier pre-materialize. This used to sit
        # inside the executor guard above, so with prefetch disabled (the
        # default on that very recipe) it never printed at all -- exactly
        # the "flag off vs silently no-op" ambiguity it exists to remove.
        print(
            "[admission-prefetch] "
            + " ".join(
                f"{k}={v}" for k, v in self.admission_prefetch_telemetry().items()
            ),
            flush=True,
        )
        if self.scene is not None:
            self._residency_ledger.release(self.scene.hot_camera_ids())
        for frame_id in self._observation_db.active_supervision_frame_ids:
            self._observation_db.demote_supervision(frame_id)
        if self.scene is not None:
            self.scene.close_decode_cache()
        # These insertion handoff fields are the remaining runtime-owned
        # references to the most recent RGB-D payload outside camera pools.
        self._prev_insert_frame = None
        self._prev_insert_depth_m = None
        self._bootstrap.clear()
        self._latest_tick_work = None
        self._latest_report_work = None
        self._active_materialization_work = None
        self._last_materialized_work = None
        self._prepared_ready_candidate_work = None
        self._prepared_ready_frozen_work = None
        self._pending_supervision_modalities.clear()
        self._pending_integration = None
        self._pending_integration_since_ns = None
        self._source_identities_by_ordinal.clear()
        reset_sensor_depth_cache_state()
        self._services.close()


def _resolve_parameters(
    profile: str,
    model_path: str | None,
    *,
    overrides: dict[str, Any] | None = None,
):
    from eligsir.cli.config import ConfigError

    args, bundle = parse_training_args([], description="Canonical mapping application")
    merged = load_cli_config(run_profile=profile)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("production mapping overrides must be a dict")
        merged = {**merged, **overrides}
    retired = "streaming_optimization_engine"
    if retired in merged:
        raise ConfigError(
            f"canonical mapping profile contains retired composition key: {retired}"
        )
    merged = validate_config(merged)
    for key, value in merged.items():
        setattr(args, key, value)
    if not getattr(args, "model_path", ""):
        args.model_path = model_path or os.path.join("output", f"live_{uuid.uuid4().hex[:10]}")
    elif model_path:
        args.model_path = model_path
    os.makedirs(args.model_path, exist_ok=True)
    dataset = bundle.model_params.extract(args)
    opt = bundle.optimization_params.extract(args)
    pipe = bundle.pipeline_params.extract(args)
    return args, dataset, opt, pipe


def create_production_mapping_application(
    *,
    profile: str = DEFAULT_MAPPING_PROFILE,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    ready_source_owner: ReadySourceOwner | None = None,
    ready_source_warmup_complete: bool = False,
    prepared_runtime_bridge: Any | None = None,
    **_ignored_transport_kwargs: Any,
):
    """Build the one standalone mapper used by ROS2 and other producers."""
    if prepared_runtime_bridge is not None:
        from .prepared_runtime_bridge import PreparedRuntimeBridge

        if not isinstance(prepared_runtime_bridge, PreparedRuntimeBridge):
            raise TypeError("prepared_runtime_bridge must be PreparedRuntimeBridge")
        bridge_owner = prepared_runtime_bridge.ready_owner
        if ready_source_owner is not None and ready_source_owner is not bridge_owner:
            raise ValueError(
                "prepared runtime bridge and ready_source_owner must share one owner"
            )
        ready_source_owner = bridge_owner
    args, dataset, opt, pipe = _resolve_parameters(
        profile, model_path, overrides=overrides
    )
    config = mapping_runtime_config_from_args(args)
    runtime = CanonicalMappingRuntime(
        config,
        MappingBackendCompatibility(dataset=dataset, opt=opt, pipe=pipe),
    )
    if ready_source_owner is not None:
        runtime.enable_ready_source_owner(
            ready_source_owner,
            warmup_complete=ready_source_warmup_complete,
        )
    if prepared_runtime_bridge is not None:
        prepared_runtime_bridge.attach_runtime(runtime)
    binding = runtime_binding_from_runtime(runtime, args=args)
    from .application import ComposedMappingApplication, MappingApplicationOptions

    return ComposedMappingApplication(
        MappingApplicationOptions(
            mapper_sink=(
                None if prepared_runtime_bridge is not None else binding.mapper_sink
            ),
            optimizer_tick=binding.optimizer_tick,
            require_tracking=True,
            materialization_policy=binding.materialization_policy,
            materialization_backend=binding.materialization_backend,
            source_complete_hook=binding.source_complete_hook,
            flush_hook=binding.flush_hook,
            close_hook=binding.close_hook,
            observation_owner=binding.observation_owner,
            control_snapshot_provider=binding.control_snapshot_provider,
            packet_capture=binding.packet_capture,
            terminal_packet_sink=binding.terminal_packet_sink,
            source_consumed_observer=binding.source_consumed_observer,
            disposition_sink=(
                runtime.record_prepared_disposition
                if prepared_runtime_bridge is not None and binding.observation_owner is not None
                else binding.disposition_sink
            ),
            runtime=runtime,
            prepared_mode=prepared_runtime_bridge is not None,
            prepared_ingress_handoff=(
                None
                if prepared_runtime_bridge is None
                else prepared_runtime_bridge.handoff
            ),
            prepared_runtime_bridge=prepared_runtime_bridge,
        )
    )


__all__ = [
    "CanonicalMappingRuntime",
    "DEFAULT_MAPPING_PROFILE",
    "MappingBackendCompatibility",
    "MappingTickResult",
    "create_production_mapping_application",
    "runtime_binding_from_runtime",
]
