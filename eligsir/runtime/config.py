"""Immutable composition contracts for the transport-free mapping runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from eligsir.contracts.materialize import MaterializationPolicy, TensorBackend
from eligsir.contracts.control import EvidenceFact
from eligsir.contracts.runtime import MappingFrame, SensorPacket, TrackingObservation
from eligsir.contracts.tracking import TrackingProvider


Scalar = bool | int | float | str | None
MapperSink = Callable[[MappingFrame, TrackingObservation], Any]
BackendFactory = Callable[["MappingRuntimeConfig"], MapperSink]

# --- CanonicalRuntimePolicy.admission_prefetch_reason -----------------------
# Named the way eligsir/mapping/resolution_provenance.py names its own
# requested-vs-effective override reasons (issue #1543 finding A is the same
# defect class as that module's startup_override_reason: a value silently
# forced by another mechanism, invisible to comparisons between runs).

ADMISSION_PREFETCH_REASON_EXPLICIT = "explicit"
"""``--streaming_admission_prefetch`` was set; the effective value matches it."""

ADMISSION_PREFETCH_REASON_PREPARED_B4_REQUIRED = "prepared_b4_required"
"""``causal_prepared_ready_b4`` forced prefetch on though it was not requested."""

ADMISSION_PREFETCH_REASON_DISABLED = "disabled"
"""Neither the explicit flag nor a forcing mechanism enabled prefetch."""

ADMISSION_PREFETCH_REASONS = (
    ADMISSION_PREFETCH_REASON_EXPLICIT,
    ADMISSION_PREFETCH_REASON_PREPARED_B4_REQUIRED,
    ADMISSION_PREFETCH_REASON_DISABLED,
)

@dataclass(frozen=True, slots=True)
class IngressTelemetrySnapshot:
    """Immutable transport-owned telemetry available to core control.

    Sources own transport-loss and queue accounting.  Core may consume only
    compact scalar evidence, never a mutable source object or packet buffer.
    """

    evidence: tuple[EvidenceFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not all(isinstance(fact, EvidenceFact) for fact in self.evidence):
            raise TypeError("ingress telemetry evidence must be EvidenceFact values")
        names = [fact.name for fact in self.evidence]
        if len(names) != len(set(names)):
            raise ValueError("ingress telemetry evidence names must be unique")


IngressTelemetryProvider = Callable[
    [SensorPacket, TrackingObservation], IngressTelemetrySnapshot
]


def _freeze_scalars(value: Any, *, name: str) -> Any:
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_scalars(item, name=name) for key, item in value.items()}
        )
    if isinstance(value, tuple):
        return tuple(_freeze_scalars(item, name=name) for item in value)
    raise TypeError(f"{name} must contain scalar values only")


@dataclass(frozen=True, slots=True)
class DepthLossPolicy:
    """Static numerical policy for depth supervision."""

    enabled: bool = True
    loss_type: str = "l1"
    weight: float = 0.0
    robust_delta: float | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.loss_type not in {"l1", "huber"}:
            raise ValueError("loss_type must be l1 or huber")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise TypeError("weight must be numeric")
        if float(self.weight) < 0.0:
            raise ValueError("weight must be non-negative")
        object.__setattr__(self, "weight", float(self.weight))
        if self.robust_delta is not None:
            if not isinstance(self.robust_delta, (int, float)) or isinstance(self.robust_delta, bool):
                raise TypeError("robust_delta must be numeric or None")
            if float(self.robust_delta) <= 0.0:
                raise ValueError("robust_delta must be positive")
            object.__setattr__(self, "robust_delta", float(self.robust_delta))
        elif self.loss_type == "huber":
            raise ValueError("huber depth loss requires robust_delta")


@dataclass(frozen=True, slots=True)
class DepthInsertionPolicy:
    """Static geometry semantics; per-tick budgets belong in work plans."""

    min_depth_m: float = 0.1
    max_depth_m: float = 10.0
    temporal_tolerance_m: float = 0.02
    temporal_relative_tolerance: float = 0.05
    edge_threshold: float = 0.2
    max_view_angle_deg: float = 75.0
    occupancy_distance_m: float = 0.02
    voxel_size_m: float = 0.02
    minimum_confidence: float = 0.0
    require_temporal_stability: bool = True
    require_new_surface: bool = True
    reject_edges: bool = True
    reject_grazing: bool = True
    confidence_interpretation: str = "metric"
    scale_mode: str = "knn"
    stability_filter_enabled: bool = False
    stability_erode_px: int = 2
    stability_window_px: int = 5
    stability_min_valid_ratio: float = 0.75
    stability_max_range_m: float = 0.08
    stability_median_threshold_m: float = 0.04
    stability_bottom_margin: float = 0.0
    temporal_check_enabled: bool = False
    require_temporal_for_sensor_closer: bool = False
    alpha_threshold: float = 0.3
    score_mode: str = "boolean_or"
    surface_fusion_enabled: bool = False
    surface_fusion_neighbors: int = 8
    surface_fusion_radius_multiplier: float = 2.0
    knn_dedup_enabled: bool = False
    knn_normal_aware: bool = False
    knn_max_existing: int = 200_000
    knn_radius_factor: float = 0.005
    knn_normal_cosine: float = 0.85
    scale_multiplier: float = 0.5
    scale_max_m: float = 0.05
    normal_scale_ratio: float = 0.15
    distance_scale_enabled: bool = False
    distance_scale_coefficient: float = 1.0
    isotropic_scale: bool = False
    score_alpha_weight: float = 1.0
    score_no_depth_weight: float = 1.0
    score_sensor_closer_weight: float = 2.0
    initial_opacity: float = 0.05
    normal_estimation: str = "finite_diff"
    debug_statistics: bool = False
    # Sparse-to-dense depth-insertion density schedule (issue #1279's
    # device-native follow-up; see utils/depth_sparsity_schedule.py). Mirrors
    # --streaming_depth_sparsity_schedule/_start_stride/_end_stride/
    # _ramp_frames. "fixed" (the default) makes
    # utils.depth_sparsity_schedule.config_from_policy() return None, which
    # keeps the device-native insertion path byte-for-byte identical to
    # before this schedule existed.
    depth_sparsity_schedule_mode: str = "fixed"
    depth_sparsity_start_stride: int = 16
    depth_sparsity_end_stride: int = 0
    depth_sparsity_ramp_frames: int = 500

    def __post_init__(self) -> None:
        for name in (
            "min_depth_m", "max_depth_m", "temporal_tolerance_m",
            "temporal_relative_tolerance", "edge_threshold", "max_view_angle_deg",
            "occupancy_distance_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
        if self.min_depth_m < 0.0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth insertion range must satisfy 0 <= min < max")
        if any(
            value < 0.0
            for value in (
                self.temporal_tolerance_m,
                self.temporal_relative_tolerance,
                self.edge_threshold,
                self.occupancy_distance_m,
            )
        ) or not 0.0 < self.max_view_angle_deg <= 90.0:
            raise ValueError("depth insertion thresholds are outside valid bounds")
        for name in ("voxel_size_m", "minimum_confidence"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
        if self.voxel_size_m <= 0.0 or self.minimum_confidence < 0.0:
            raise ValueError("voxel_size_m must be positive and minimum_confidence non-negative")
        for name in (
            "require_temporal_stability",
            "require_new_surface",
            "reject_edges",
            "reject_grazing",
            "stability_filter_enabled",
            "temporal_check_enabled",
            "require_temporal_for_sensor_closer",
            "surface_fusion_enabled", "knn_dedup_enabled", "knn_normal_aware",
            "distance_scale_enabled", "isotropic_scale",
            "debug_statistics",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in ("stability_erode_px", "stability_window_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if isinstance(self.surface_fusion_neighbors, bool) or int(self.surface_fusion_neighbors) < 1:
            raise ValueError("surface_fusion_neighbors must be positive")
        object.__setattr__(self, "surface_fusion_neighbors", int(self.surface_fusion_neighbors))
        if isinstance(self.knn_max_existing, bool) or int(self.knn_max_existing) < 1:
            raise ValueError("knn_max_existing must be positive")
        object.__setattr__(self, "knn_max_existing", int(self.knn_max_existing))
        for name in (
            "stability_min_valid_ratio", "stability_max_range_m",
            "stability_median_threshold_m", "stability_bottom_margin", "alpha_threshold",
            "surface_fusion_radius_multiplier", "knn_radius_factor", "knn_normal_cosine",
            "scale_multiplier", "scale_max_m", "normal_scale_ratio", "distance_scale_coefficient",
            "score_alpha_weight", "score_no_depth_weight", "score_sensor_closer_weight",
            "initial_opacity",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, float(value))
        if self.stability_min_valid_ratio > 1.0 or self.stability_bottom_margin >= 1.0:
            raise ValueError("stability ratios must be within [0, 1)")
        if self.score_mode not in {"boolean_or", "alpha_only", "depth_gap_only", "sensor_closer_only", "weighted_topk"}:
            raise ValueError("unsupported insertion score mode")
        if self.normal_estimation not in {"finite_diff", "kornia"}:
            raise ValueError("normal_estimation must be finite_diff or kornia")
        if not isinstance(self.confidence_interpretation, str) or not self.confidence_interpretation:
            raise ValueError("confidence_interpretation must be non-empty")
        if self.scale_mode not in {"knn", "footprint", "hybrid"}:
            raise ValueError("scale_mode must be knn, footprint, or hybrid")
        if self.depth_sparsity_schedule_mode not in {"fixed", "coarse_to_fine"}:
            raise ValueError(
                "depth_sparsity_schedule_mode must be fixed or coarse_to_fine"
            )
        for name in (
            "depth_sparsity_start_stride",
            "depth_sparsity_end_stride",
            "depth_sparsity_ramp_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))


def relaxed_depth_insertion_policy(
    base: "DepthInsertionPolicy",
    *,
    alpha_threshold: float,
    max_depth_m: float,
) -> "DepthInsertionPolicy":
    """Return a policy for the post-source-complete replay-depth pass.

    ``alpha_threshold < 0.0`` and ``max_depth_m <= 0.0`` are the "unset"
    sentinels (see ``streaming_replay_depth_relaxed_alpha_threshold`` /
    ``streaming_replay_depth_relaxed_max_depth`` in ``arguments/__init__.py``)
    -- either or both being unset means that field is inherited verbatim from
    ``base``, so the default call site is byte-identical to using ``base``
    directly. Raising ``alpha_threshold`` widens the occupancy gate's
    "already covered" bar (fewer pixels get treated as covered, so more of
    them remain insertion candidates); raising ``max_depth_m`` admits farther
    sensor returns. This is the root-cause mitigation for #1259's
    bootstrap-vs-ongoing-insertion far-region asymmetry: see
    ``docs/architecture/2026-08-30-issue1259-far-region-insertion-asymmetry.md``.
    """

    overrides: dict[str, float] = {}
    if alpha_threshold >= 0.0:
        overrides["alpha_threshold"] = float(alpha_threshold)
    if max_depth_m > 0.0:
        overrides["max_depth_m"] = float(max_depth_m)
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)


@dataclass(frozen=True, slots=True)
class CanonicalDeviceCapabilities:
    """Accelerator facts resolved once at the outer composition boundary."""

    accelerator: str = "cuda"
    gpu_frame_prep: bool = False
    gpu_frame_resident: bool = False
    tensor_candidate_path: bool = False

    def __post_init__(self) -> None:
        if self.accelerator not in {"cuda", "rocm"}:
            raise ValueError("canonical accelerator must be cuda or rocm")
        for name in (
            "gpu_frame_prep",
            "gpu_frame_resident",
            "tensor_candidate_path",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class CanonicalBootstrapPolicy:
    """Strict metric RGB-D bootstrap semantics for canonical MappingFrames."""

    depth_stride: int = 8
    min_depth_m: float = 0.1
    max_depth_m: float = 8.0
    max_points: int = 250_000
    seed_voxel_size_m: float = 0.02
    init_scale_mode: str = "knn"
    init_scale: float = 0.01
    model_voxel_size_m: float = 0.02
    occupancy_voxel_size_m: float = 0.02
    temporal_check_enabled: bool = False
    temporal_tolerance_m: float = 0.07
    require_temporal_for_seed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.depth_stride, bool) or int(self.depth_stride) < 1:
            raise ValueError("depth_stride must be positive")
        if isinstance(self.max_points, bool) or int(self.max_points) < 1:
            raise ValueError("max_points must be positive")
        object.__setattr__(self, "depth_stride", int(self.depth_stride))
        object.__setattr__(self, "max_points", int(self.max_points))
        for name in (
            "min_depth_m",
            "max_depth_m",
            "seed_voxel_size_m",
            "init_scale",
            "model_voxel_size_m",
            "occupancy_voxel_size_m",
            "temporal_tolerance_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            object.__setattr__(self, name, float(value))
        if self.min_depth_m < 0.0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("bootstrap depth range must satisfy 0 <= min < max")
        if min(
            self.seed_voxel_size_m,
            self.init_scale,
            self.model_voxel_size_m,
            self.occupancy_voxel_size_m,
        ) < 0.0:
            raise ValueError("bootstrap scale and voxel values must be non-negative")
        if self.init_scale_mode not in {"knn", "fixed", "voxel", "constant"}:
            raise ValueError("unsupported bootstrap init_scale_mode")
        for name in ("temporal_check_enabled", "require_temporal_for_seed"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.require_temporal_for_seed and not self.temporal_check_enabled:
            raise ValueError(
                "required temporal bootstrap support needs temporal checking enabled"
            )
        if self.temporal_check_enabled and self.temporal_tolerance_m <= 0.0:
            raise ValueError("temporal bootstrap tolerance must be positive")


@dataclass(frozen=True, slots=True)
class CanonicalRuntimePolicy:
    """Canonical runtime controls resolved before accelerator construction."""

    model_path: str = "output/canonical"
    cap_max: int | None = None
    max_new_gaussians_per_frame: int = 0
    max_active_supervision_frames: int = 120
    sensor_depth_cache_max_frames: int = 0
    live_payload_max_frames: int = 512
    cold_payload_max_frames: int = 128
    cold_payload_max_bytes: int = 2_147_483_648
    # Byte ceiling for multi-tier HOT supervision residency
    # (``eligsir/mapping/hot_tier_cache.py``).  ``0`` means unbounded, i.e.
    # the count-based ``max_active_supervision_frames`` policy alone governs
    # HOT residency, which is the pre-existing behaviour and the default here.
    #
    # Frame-count bounding is NOT a memory bound: the same 400-frame cap holds
    # 1.83 GiB at TUM resolution and 16.5 GiB at ScanNet++ resolution (PR
    # #1252).  Setting this to a positive value makes the byte budget the
    # binding constraint and can evict a frame's only retained tier, so it
    # changes production residency behaviour and is deliberately opt-in until
    # that has its own measured validation.
    hot_tier_cache_max_bytes: int = 0
    # How many distinct resolution tiers one HOT frame may retain at once
    # (``eligsir/mapping/hot_tier_cache.py``). Was previously a hardcoded
    # module constant with no CLI path (issue #279 follow-up); default
    # matches that pre-existing constant so this field changes nothing
    # unless explicitly overridden.
    max_tiers_per_frame: int = 2
    keyframe_window: int = 120
    offload_camera_images: bool = True
    # Issue #279/#1259/#1342: decode a just-admitted frame's native RGB/depth
    # on a background thread ahead of the tick that first draws it for
    # supervision, instead of paying that decode cold on the critical path.
    # Report-only telemetry aside, this changes WHEN pixels are decoded, never
    # WHAT gets decoded or rendered -- see production_application.py's
    # `_maybe_schedule_admission_prefetch`/`_prefetch_decode_worker`. Default
    # OFF (supervisor decision, issue #279/#1259 throughput cluster wrap-up):
    # measured a correct no-op on the causal_replay/tum_rgbd recipe this
    # session's goal scenarios actually use -- see arguments/__init__.py's
    # `streaming_admission_prefetch` for the full rationale. Enabling
    # reproduces the prefetch path exactly for any workload confirmed to
    # build PixelReloadToken handles at admission time.
    # Issue #1543 finding A: `admission_prefetch_requested` preserves what the
    # raw CLI/profile flag actually asked for; `admission_prefetch_reason`
    # (one of "explicit"/"prepared_b4_required"/"disabled") records why the
    # effective decision may differ, since `causal_prepared_ready_b4` can
    # force prefetch on even when it was never requested. `enabled` (below)
    # is derived from `reason` rather than stored separately, so the two
    # can never drift out of sync.
    admission_prefetch_requested: bool = False
    admission_prefetch_reason: str = ADMISSION_PREFETCH_REASON_DISABLED
    admission_prematerialize_enabled: bool = False
    blur_downsample: int = 4
    target_capacity: int = 0
    # Raised 2026-08-29 (issue #1259) to match arguments/__init__.py's
    # streaming_max_transfer_bytes default -- see that field's comment.
    max_transfer_bytes: int = 2 * 1024 * 1024 * 1024
    sh_degree_schedule_rgb_pixels: tuple[int, ...] = (
        307_200_000,
        614_400_000,
        921_600_000,
    )
    selective_adam_allow_dense_grads: bool = False
    random_background: bool = False
    white_background: bool = False
    gsplat_sparse_grad: bool = False
    sh_update_interval: int = 1
    densification_strategy: str = "gsplat_energy_mcmc"
    densify_from_iter: int = 500
    mcmc_stop_growth_iter: int = 12_000
    densify_until_iter: int = 25_000
    mcmc_grow_interval_pixels: int = 30_720_000
    mcmc_post_ingestion_grow_interval_pixels: int = 0
    mcmc_relocate_interval_pixels: int = 15_360_000
    mcmc_post_ingestion_relocate_interval_pixels: int = 0
    mcmc_min_optimizer_updates_after_structural_mutation: int = 2
    mcmc_noise_stop_optimizer_updates: int = 30_000
    mcmc_temperature_horizon_optimizer_updates: int = 30_000
    mcmc_growth_factor_start: float = 1.05
    max_optimization_transaction_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be non-empty")
        if self.cap_max is not None and (
            isinstance(self.cap_max, bool) or int(self.cap_max) < 0
        ):
            raise ValueError("cap_max must be non-negative or None")
        if self.cap_max is not None:
            object.__setattr__(self, "cap_max", int(self.cap_max))
        if self.max_optimization_transaction_ns is not None:
            value = self.max_optimization_transaction_ns
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("max_optimization_transaction_ns must be an integer or None")
            if value <= 0:
                raise ValueError("max_optimization_transaction_ns must be positive")
        for name in (
            "mcmc_grow_interval_pixels",
            "mcmc_post_ingestion_grow_interval_pixels",
            "mcmc_relocate_interval_pixels",
            "mcmc_post_ingestion_relocate_interval_pixels",
            "mcmc_min_optimizer_updates_after_structural_mutation",
            "mcmc_noise_stop_optimizer_updates",
            "mcmc_temperature_horizon_optimizer_updates",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name in (
            "max_new_gaussians_per_frame",
            "max_active_supervision_frames",
            "sensor_depth_cache_max_frames",
            "live_payload_max_frames",
            "cold_payload_max_frames",
            "cold_payload_max_bytes",
            "hot_tier_cache_max_bytes",
            "max_tiers_per_frame",
            "keyframe_window",
            "blur_downsample",
            "target_capacity",
            "max_transfer_bytes",
            "sh_update_interval",
            "densify_from_iter",
            "mcmc_stop_growth_iter",
            "densify_until_iter",
            "mcmc_grow_interval_pixels",
            "mcmc_post_ingestion_grow_interval_pixels",
            "mcmc_relocate_interval_pixels",
            "mcmc_post_ingestion_relocate_interval_pixels",
            "mcmc_min_optimizer_updates_after_structural_mutation",
            "mcmc_noise_stop_optimizer_updates",
            "mcmc_temperature_horizon_optimizer_updates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        if self.max_active_supervision_frames < 1 or self.keyframe_window < 1:
            raise ValueError("runtime frame bounds must be positive")
        if self.max_tiers_per_frame < 1:
            raise ValueError("max_tiers_per_frame must be positive")
        if self.mcmc_grow_interval_pixels < 1 or self.mcmc_relocate_interval_pixels < 1:
            raise ValueError("canonical MCMC exposure intervals must be positive")
        if self.mcmc_temperature_horizon_optimizer_updates < 1:
            raise ValueError("MCMC temperature optimizer-update horizon must be positive")
        if self.admission_prefetch_reason not in ADMISSION_PREFETCH_REASONS:
            raise ValueError(
                f"admission_prefetch_reason must be one of {ADMISSION_PREFETCH_REASONS}"
            )
        for name in (
            "offload_camera_images",
            "admission_prefetch_requested",
            "admission_prematerialize_enabled",
            "selective_adam_allow_dense_grads",
            "random_background",
            "white_background",
            "gsplat_sparse_grad",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        # Issue #1259: originally hard-gated to "gsplat_energy_mcmc" only, with
        # no comment explaining why. Audited 2026-08-31: `make_mcmc_strategy`
        # (utils/strategies/mcmc/registry.py) dispatches "gsplat_mcmc" to
        # `UpstreamGsplatMCMCStrategy` on a gsplat-layout model, uniformly
        # wrapped by `ensure_arbiter_owned`/`ArbiterOwnedMCMCStrategy`, which
        # is what actually supplies `delegated_post_backward_proposal` to
        # `production_application.py`'s mutation-commit path. That wrapper
        # passes `utility`/`use_energy_mcmc` through as plain kwargs
        # regardless of strategy; `UpstreamGsplatMCMCStrategy` never reads
        # them, since it uses no externally-supplied utility score.
        # Separately, `self._use_energy_mcmc` (production_application.py) is
        # already independent of `densification_strategy` -- it comes from
        # `args.energy_mcmc` -- and already handles `utility=None` cleanly.
        #
        # A real canonical-runtime run of "gsplat_mcmc" surfaced THREE further
        # real crashes past that mutation-layer audit, all now fixed and
        # covered by tests/test_issue1259_gsplat_mcmc_unlock.py:
        #  1. `AttributeError: mcmc_stop_growth_iter` -- config_boundary.py's
        #     `mapping_runtime_config_from_args` deliberately pops
        #     fixed-iteration-cadence fields (the canonical runtime is
        #     iteration-index-agnostic; `gsplat_energy_mcmc` uses pixel/
        #     exposure-based cadence instead), but
        #     `UpstreamGsplatMCMCStrategy.initialize_state` read it unguarded.
        #     Fixed via a getattr fallback in utils/strategies/mcmc/baseline.py.
        #  2. `RuntimeError: tensor size mismatch` in objective.py's
        #     `_visible | _provisional` mask assembly, one growth event later
        #     -- `UpstreamGsplatMCMCStrategy` delegates straight into gsplat's
        #     own upstream `MCMCStrategy`, which mutates only
        #     `gaussians.params`/`.optimizers` and has zero awareness of this
        #     repo's ~25 parallel per-Gaussian lifecycle buffers, unlike
        #     `GsplatEnergyMCMCStrategy` which keeps them in sync itself.
        #  3. `RuntimeError: canonical lifecycle birth_iter shape mismatch`
        #     -- the fix for (2) reused `train_streaming.py`'s
        #     `_sync_streaming_state_lengths`, whose own hand-picked buffer
        #     list had silently drifted out of sync with
        #     `GsplatGaussianModel._STREAMING_BUFFER_NAMES` and never covered
        #     `birth_iter` or any lineage buffer.
        # Fixed by `utils/strategies/mcmc/arbiter_owned.py`'s
        # `_resync_gaussian_lifecycle_buffers`, which pads/truncates the FULL
        # buffer list after every delegated strategy's mutation, applied
        # unconditionally so it holds for any future strategy too.
        #
        # "gsplat_mcmc" is genuinely unlocked here, verified via a real,
        # crash-free, full end-to-end fr3 comparison run (issue #1259).
        #
        # "gsplat_default" (gsplat's own gradient-threshold clone/split/prune
        # ADC, the ORIGINAL non-MCMC 3DGS densification approach) was audited
        # the same way and looked safe by the same reasoning -- its own
        # `initialize_state` already uses getattr fallbacks throughout, per
        # its own #368 comment -- but a real run hit a FOURTH, materially
        # different and deeper crash: `AttributeError: 'NoneType' object has
        # no attribute 'clone'` in gsplat's `DefaultStrategy._update_state`,
        # because ADC's clone/split heuristic needs `.grad` populated on the
        # 2D screen-space means tensor post-backward
        # (`means2d.retain_grad()`), which this render pipeline evidently
        # never enables since the MCMC-based strategies (the only ones this
        # runtime has run until now) don't need it. That is a render-pipeline
        # change, not a bookkeeping fix, and was judged too large a scope
        # increase to chase in the same pass as the above three -- NOT
        # unlocked here. Left gated pending that follow-up -- see issue
        # #1259.
        #
        # Legacy "mcmc"/"hybrid" (confirmed by reading `ScheduledMCMCStrategy`
        # -- both ARE MCMC-based, not ADC, despite the "legacy" name) use a
        # different model layout (`GaussianModel`, not `GsplatGaussianModel`)
        # that this runtime's own bootstrap can in principle construct via
        # `model_layout="legacy"`, but that combination has not been
        # constructed or run end-to-end on the canonical path and is left
        # gated pending that verification -- see issue #1259.
        if self.densification_strategy not in (
            "gsplat_energy_mcmc",
            "gsplat_mcmc",
        ):
            raise ValueError(
                "canonical runtime supports gsplat_energy_mcmc or gsplat_mcmc "
                "only"
            )
        if not isinstance(self.mcmc_growth_factor_start, (int, float)) or isinstance(
            self.mcmc_growth_factor_start, bool
        ):
            raise TypeError("mcmc_growth_factor_start must be numeric")
        object.__setattr__(
            self, "mcmc_growth_factor_start", float(self.mcmc_growth_factor_start)
        )
        sh_schedule = tuple(
            int(value) for value in self.sh_degree_schedule_rgb_pixels
        )
        if any(value < 1 for value in sh_schedule):
            raise ValueError("SH degree RGB-pixel thresholds must be positive")
        object.__setattr__(self, "sh_degree_schedule_rgb_pixels", sh_schedule)

    @property
    def admission_prefetch_enabled(self) -> bool:
        """Effective admission-prefetch decision, derived from `reason`.

        Not a stored field: keeping this as a computed property (issue
        #1543 finding A) makes it impossible for `enabled` to disagree with
        `reason`/`requested`, which a third independently-set boolean field
        could otherwise drift from.
        """

        return self.admission_prefetch_reason != ADMISSION_PREFETCH_REASON_DISABLED


@dataclass(frozen=True, slots=True)
class CanonicalOptimizationPolicy:
    """Read-only adapter for characterized optimizer/MCMC scalar controls.

    Older optimizer kernels still consume attribute-shaped policy inputs.  The
    outer resolver selects and freezes only numerical policy fields here; the
    raw parser namespace never crosses into the canonical runtime.
    """

    values: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_scalars(self.values, name="values"))

    def __getattr__(self, name: str) -> Scalar:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class CanonicalPipelinePolicy:
    """Read-only renderer policy resolved before runtime construction."""

    values: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_scalars(self.values, name="values"))

    def __getattr__(self, name: str) -> Scalar:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class MappingRuntimeConfig:
    """Dataset-free, immutable runtime method and policy configuration."""

    model_layout: str
    renderer: str
    numerical_mode: str
    hard_limits: Mapping[str, Scalar] = field(default_factory=dict)
    residency: Mapping[str, Scalar] = field(default_factory=dict)
    depth_loss: DepthLossPolicy = field(default_factory=DepthLossPolicy)
    depth_insertion: DepthInsertionPolicy = field(default_factory=DepthInsertionPolicy)
    device_capabilities: CanonicalDeviceCapabilities = field(
        default_factory=CanonicalDeviceCapabilities
    )
    bootstrap: CanonicalBootstrapPolicy = field(default_factory=CanonicalBootstrapPolicy)
    runtime: CanonicalRuntimePolicy = field(default_factory=CanonicalRuntimePolicy)
    optimization: CanonicalOptimizationPolicy = field(
        default_factory=CanonicalOptimizationPolicy
    )
    pipeline: CanonicalPipelinePolicy = field(default_factory=CanonicalPipelinePolicy)
    backpressure_policy: Mapping[str, Scalar] = field(default_factory=dict)
    keyframe_policy: Mapping[str, Scalar] = field(default_factory=dict)
    static_options: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_layout", "renderer", "numerical_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.model_layout not in {"gsplat", "legacy"}:
            raise ValueError("model_layout must be gsplat or legacy")
        if not isinstance(self.depth_loss, DepthLossPolicy):
            raise TypeError("depth_loss must be DepthLossPolicy")
        if not isinstance(self.depth_insertion, DepthInsertionPolicy):
            raise TypeError("depth_insertion must be DepthInsertionPolicy")
        for name, expected in (
            ("device_capabilities", CanonicalDeviceCapabilities),
            ("bootstrap", CanonicalBootstrapPolicy),
            ("runtime", CanonicalRuntimePolicy),
            ("optimization", CanonicalOptimizationPolicy),
            ("pipeline", CanonicalPipelinePolicy),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        challenger_fraction = self.optimization.values.get(
            "streaming_replay_challenger_fraction", 0.0
        )
        if (
            isinstance(challenger_fraction, bool)
            or not isinstance(challenger_fraction, (int, float))
            or not isfinite(challenger_fraction)
            or not 0.0 <= challenger_fraction <= 1.0
        ):
            raise ValueError(
                "streaming_replay_challenger_fraction must be finite and within [0, 1]"
            )
        for name, default in (
            ("streaming_replay_challenger_demotion_cadence", 0),
            ("streaming_replay_challenger_demotion_limit", 0),
            ("streaming_replay_challenger_lease_ticks", 100),
        ):
            value = self.optimization.values.get(name, default)
            if type(value) is not int:
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "hard_limits",
            "residency",
            "backpressure_policy",
            "keyframe_policy",
            "static_options",
        ):
            object.__setattr__(self, name, _freeze_scalars(getattr(self, name), name=name))
        population_shadow = self.static_options.get("population_shadow_enabled", False)
        population_live = self.static_options.get("population_live_enabled", False)
        if type(population_shadow) is not bool:
            raise TypeError("population_shadow_enabled must be an exact bool")
        if type(population_live) is not bool:
            raise TypeError("population_live_enabled must be an exact bool")
        if population_live and not population_shadow:
            raise ValueError(
                "population_live_enabled requires population_shadow_enabled"
            )
        if population_live:
            ownership_strategy = self.static_options.get(
                "population_ownership_strategy"
            )
            if ownership_strategy != "gsplat_energy_mcmc":
                raise ValueError(
                    "population_live_enabled requires unified population ownership "
                    "through --densification-strategy gsplat_energy_mcmc"
                )
            conflicting_controls = (
                (
                    "legacy_lifecycle_enabled",
                    "--streaming-lifecycle-enabled",
                ),
                (
                    "legacy_post_mapping_mature_mobility",
                    "--streaming-post-mapping-mature-mobility",
                ),
                (
                    "legacy_depth_lock_centers",
                    "--streaming-depth-lock-centers",
                ),
                (
                    "legacy_depth_lock_bootstrap",
                    "--streaming-depth-lock-bootstrap",
                ),
                (
                    "legacy_depth_lock_insertions",
                    "--streaming-depth-lock-insertions",
                ),
                (
                    "legacy_depth_lock_prune_conflicts",
                    "--streaming-depth-lock-prune-conflicts",
                ),
                (
                    "legacy_insert_from_depth",
                    "--streaming-insert-from-depth",
                ),
                ("legacy_delegated_mcmc", "--densification-strategy gsplat_mcmc"),
            )
            enabled_controls = [
                option
                for name, option in conflicting_controls
                if self.static_options.get(name) is not False
            ]
            if enabled_controls:
                raise ValueError(
                    "population_live_enabled requires unified population ownership; "
                    "disable competing legacy controls: "
                    + ", ".join(enabled_controls)
                )
        if (
            self.static_options.get("live_mode") is True
            and self.runtime.max_optimization_transaction_ns is None
        ):
            raise ValueError(
                "live online mode requires a positive transaction freshness budget"
            )


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    """All runtime actuators obtained from one composed runtime instance."""

    mapper_sink: MapperSink
    optimizer_tick: Callable[[], Any]
    source_complete_hook: Optional[Callable[[], Any]] = None
    flush_hook: Optional[Callable[[], Any]] = None
    close_hook: Optional[Callable[[], Any]] = None
    observation_owner: Any = None
    control_snapshot_provider: Any = None
    disposition_sink: Any = None
    packet_capture: Any = None
    terminal_packet_sink: Any = None
    source_consumed_observer: Any = None
    materialization_policy: Optional[MaterializationPolicy] = None
    materialization_backend: Optional[TensorBackend] = None

    def __post_init__(self) -> None:
        for name in ("mapper_sink", "optimizer_tick"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        for name in (
            "source_complete_hook",
            "flush_hook",
            "close_hook",
            "control_snapshot_provider",
            "disposition_sink",
            "packet_capture",
            "terminal_packet_sink",
            "source_consumed_observer",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable when provided")
        if (self.observation_owner is None) != (self.control_snapshot_provider is None):
            raise ValueError(
                "observation_owner and control_snapshot_provider must be supplied together"
            )
        if self.disposition_sink is not None and self.observation_owner is None:
            raise ValueError("disposition_sink requires an observation_owner")


RuntimeFactory = Callable[["MappingRuntimeConfig"], RuntimeBinding]


@dataclass(frozen=True, slots=True)
class MappingRuntimeDependencies:
    """Process-owned factories/providers/sinks required to run one application."""

    runtime_factory: RuntimeFactory
    tracking_provider: TrackingProvider
    ingress_telemetry_provider: IngressTelemetryProvider

    def __post_init__(self) -> None:
        for name in (
            "runtime_factory",
            "ingress_telemetry_provider",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if not callable(getattr(self.tracking_provider, "observe", None)):
            raise TypeError("tracking_provider must implement observe(packet)")


__all__ = [
    "CanonicalBootstrapPolicy",
    "CanonicalDeviceCapabilities",
    "CanonicalOptimizationPolicy",
    "CanonicalPipelinePolicy",
    "CanonicalRuntimePolicy",
    "DepthInsertionPolicy",
    "DepthLossPolicy",
    "IngressTelemetrySnapshot",
    "MappingRuntimeConfig",
    "MappingRuntimeDependencies",
    "RuntimeBinding",
]
