"""Typed configuration resolved once before canonical runtime construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from eligsir.contracts.control import MappingWorkPlan
from eligsir.mapping.batch_contract import validate_supervision_camera_count
from eligsir.mapping.hot_tier_cache import MAX_RETAINED_TIERS_PER_FRAME
from eligsir.population_control_deprecation import (
    LEGACY_POPULATION_CONTROLS_STATUS,
    legacy_population_control_values,
)
from eligsir.runtime.config import (
    ADMISSION_PREFETCH_REASON_DISABLED,
    ADMISSION_PREFETCH_REASON_EXPLICIT,
    ADMISSION_PREFETCH_REASON_PREPARED_B4_REQUIRED,
    CanonicalBootstrapPolicy,
    CanonicalDeviceCapabilities,
    CanonicalOptimizationPolicy,
    CanonicalPipelinePolicy,
    CanonicalRuntimePolicy,
    DepthInsertionPolicy,
    DepthLossPolicy,
    MappingRuntimeConfig,
)


LIVE_OPTIMIZATION_TRANSACTION_FRESHNESS_NS = 33_333_333
SUPPORTED_RASTERIZE_MODES = frozenset({"classic", "antialiased"})

# 2026-09-07 (post-mapping refinement "all"-mode catalog collapse, floor3
# r9700 `psnr26v12h_orb12_floor3_plain_sampleall_h`): the LIVE mapper's
# bounded cold-payload store (`--streaming_live_cold_payload_max_frames`,
# default 128) evicts non-reloadable (in-memory bag source, no
# `rgb_path`/`reload_token`) deferred frame payloads once that FRAME-COUNT
# bound is exceeded -- observed in that run's own `train.log`:
# "[observation_db] WARNING: a deferred live observation was evicted from
# the bounded cold payload store with no reload token; its pixels are gone,
# so it is now permanently ineligible for replay." Eviction fires
# `FrameObservationDB`'s payload-eviction observer
# (`ProductionApplication._retire_historical_identity`), which pops the
# frame from `_historical_sources_by_frame_id` -- permanently removing it
# from `build_all_mode_replay_catalog`'s population. 128 is sized for the
# LIVE mapper's own bounded backpressure/replay needs, not for
# `--streaming_post_mapping_refinement_sampling all`'s explicit goal of
# covering the WHOLE trajectory; on that run the catalog collapsed to
# exactly 128 unique frames (matching the default one-for-one), so "all"
# mode never diverged from the "pool" baseline it was meant to fix.
#
# `--streaming_observation_db_retain_all` (default off, auto-enabled by
# `causal_entry.py` whenever `streaming_post_mapping_refinement_sampling !=
# "pool"`) removes ONLY the frame-COUNT cap (passes `max_frames=None` to
# `BoundedColdPayloadStore`, its documented "legacy unbounded" value) --
# the frame-BYTES cap remains a real, finite safety bound throughout,
# raised only to `--streaming_observation_db_retain_all_max_bytes` (never
# below whatever `--streaming_live_cold_payload_max_bytes` was already
# explicitly set to). This is deliberately NOT "retain literally
# everything, unbounded": a non-reloadable in-memory bag source (Orbbec/
# RealSense ROS2 bags -- no on-disk per-frame path) still permanently loses
# any frame evicted once that byte ceiling is exceeded; there is no cheap
# "lightweight pose/timestamp-only" placeholder that can later re-decode
# such a frame's pixels, so covering a multi-thousand-frame trajectory's
# full RGB+depth in this store costs real host RAM (order single-digit GB)
# by design, not by bug.
STREAMING_OBSERVATION_DB_RETAIN_ALL_DEFAULT_MAX_BYTES = 8_589_934_592  # 8 GiB

# `CanonicalRuntimePolicy.cold_payload_max_frames` is a strictly-typed
# `int` field (`__post_init__` rejects anything whose `type(...) is not
# int`, including `None`), so "count-unbounded" cannot be spelled as `None`
# at that layer the way `BoundedColdPayloadStore` itself allows -- this
# sentinel (larger than any realistic trajectory) stands in for it instead;
# the real, finite bound retain-all mode still enforces is the byte cap
# below.
_RETAIN_ALL_UNBOUNDED_FRAME_SENTINEL = 2_000_000_000


def source_native_resolution(source: Any) -> tuple[tuple[int, int], str] | tuple[None, str]:
    """Derive native resolution from a frame source, trying multiple fallback paths.

    Tries in order:
    1. source.native_resolution (for RealityScan-like sources with explicit attribute)
    2. (source.width, source.height) if both present as attributes on the source
    3. First frame record's width/height from source._frames[0] (ScanNet, TUM, Replica, etc.)
    4. First frame's colour image header via PIL (if rgb_path exists and is readable)

    Args:
        source: A frame source object (ScanNetFrameSource, TUMFrameSource, etc.)

    Returns:
        ((width, height), method_name) on success, (None, "unknown") on all fallbacks exhausted.
        method_name describes which fallback resolved it (e.g. "native_resolution_attr",
        "source_width_height", "first_frame_record", "pil_image_header").
    """
    try:
        # Attempt 1: explicit native_resolution attribute
        native_res = getattr(source, "native_resolution", None)
        if native_res is not None:
            if isinstance(native_res, (tuple, list)) and len(native_res) >= 2:
                return (int(native_res[0]), int(native_res[1])), "native_resolution_attr"
    except Exception:
        pass

    try:
        # Attempt 2: source.width and source.height attributes
        source_width = getattr(source, "width", None)
        source_height = getattr(source, "height", None)
        if source_width is not None and source_height is not None:
            return (int(source_width), int(source_height)), "source_width_height"
    except Exception:
        pass

    try:
        # Attempt 3: first frame record in source._frames[0]
        frames = getattr(source, "_frames", None)
        if frames is not None and isinstance(frames, (list, tuple)) and len(frames) > 0:
            first_frame = frames[0]
            frame_width = getattr(first_frame, "width", None)
            frame_height = getattr(first_frame, "height", None)
            if frame_width is not None and frame_height is not None:
                return (int(frame_width), int(frame_height)), "first_frame_record"
    except Exception:
        pass

    try:
        # Attempt 4: PIL image header from first frame's rgb_path
        frames = getattr(source, "_frames", None)
        if frames is not None and isinstance(frames, (list, tuple)) and len(frames) > 0:
            first_frame = frames[0]
            rgb_path = getattr(first_frame, "rgb_path", None)
            if rgb_path and isinstance(rgb_path, str) and len(rgb_path) > 0:
                from PIL import Image
                img = Image.open(rgb_path)
                width, height = img.size
                return (width, height), "pil_image_header"
    except Exception:
        pass

    return None, "unknown"


def _probe_device_total_memory() -> int | None:
    """Probe CUDA device memory, returning total bytes or None if unavailable.

    Returns:
        Total device memory in bytes, or None if CUDA is not available.

    Raises:
        Exception: On torch import or device query failures (caught by caller).
    """
    import torch
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory)
    return None


def resolve_cold_payload_bounds(args: Any) -> tuple[int, int]:
    """Return the effective ``(cold_payload_max_frames, cold_payload_max_bytes)``.

    ``cold_payload_max_frames`` is raised to a count-unbounded sentinel
    (see :class:`~eligsir.mapping.stores.cold_payload_store.
    BoundedColdPayloadStore`'s own "``None`` preserves legacy unbounded
    behavior" docstring -- this sentinel has the same effect once it is
    itself far larger than any real trajectory) only when
    ``--streaming_observation_db_retain_all`` is active; otherwise the raw
    ``--streaming_live_cold_payload_max_frames`` value passes through
    unchanged, byte-identical to before this flag existed. The byte cap
    remains real and finite in both cases.

    When policy is "residency", the explicit flags still win if set; otherwise
    this function returns the legacy defaults (retain_all is still honored).
    """

    configured_bytes = int(args.streaming_live_cold_payload_max_bytes)
    if not bool(getattr(args, "streaming_observation_db_retain_all", False)):
        return int(args.streaming_live_cold_payload_max_frames), configured_bytes
    retain_all_bytes = int(
        getattr(
            args,
            "streaming_observation_db_retain_all_max_bytes",
            STREAMING_OBSERVATION_DB_RETAIN_ALL_DEFAULT_MAX_BYTES,
        )
    )
    return _RETAIN_ALL_UNBOUNDED_FRAME_SENTINEL, max(configured_bytes, retain_all_bytes)


def resolve_admitted_payload_bounds(
    args: Any,
    native_width: int | None = None,
    native_height: int | None = None,
    probe_device_memory=None,
    probe_supply_fn=None,
) -> tuple[int, int, dict]:
    """Return the effective ``(admitted_payload_max_frames, hot_tier_max_bytes, metadata)``.

    When ``--streaming_payload_budget_policy`` is "resolution_scaled" (default),
    derives frame and byte bounds from the source's native resolution and device
    memory. When policy is "legacy", uses hardcoded defaults (512 frames, 0 bytes).
    When policy is "residency", computes bounds from residency budget resolver.

    Args:
        args: Configuration namespace.
        native_width: Source native width in pixels (optional).
        native_height: Source native height in pixels (optional).
        probe_device_memory: Optional callable that returns device memory in bytes or None
            if CUDA unavailable. If not provided, uses real device probe. For testing,
            pass `lambda: <bytes>` to inject a known value or `lambda: None` to simulate
            no CUDA. Production path (None) calls :func:`_probe_device_total_memory`.
        probe_supply_fn: Optional callable that returns ResidencySupply. For testing,
            pass a lambda that returns ResidencySupply. Production path (None) calls
            :func:`probe_supply` from payload_budget module.

    Returns:
        ``(admitted_payload_max_frames, hot_tier_max_bytes, metadata_dict)`` where
        metadata contains policy, native resolution, and resolved values for logging.
    """
    from eligsir.runtime.payload_budget import (
        ResidencyDemand,
        probe_supply,
        resolve_residency_budget,
    )

    policy = str(getattr(args, "streaming_payload_budget_policy", "resolution_scaled"))
    metadata = {"policy": policy}

    # Defaults for legacy mode
    legacy_frames = 512
    legacy_hot_bytes = 0

    # Read explicit flags if provided (sentinel is None)
    explicit_frames = getattr(args, "streaming_live_admitted_payload_max_frames", None)
    explicit_hot_bytes = getattr(args, "streaming_hot_tier_max_bytes", None)

    # Special handling for residency policy
    if policy == "residency" and explicit_frames is None:
        # Build ResidencyDemand from args
        if not native_width or not native_height:
            raise ValueError(
                "residency policy requires native_width and native_height to be provided"
            )

        # Get tier divisor from mapping resolution tier
        mapping_tier = str(args.streaming_mapping_resolution_tier).lower()
        tier_divisor = {"native": 1, "half": 2, "quarter": 4, "eighth": 8}.get(mapping_tier, 2)

        window_frames = int(args.streaming_keyframe_window)
        batch_size = int(args.streaming_optimization_batch_size)

        demand = ResidencyDemand(
            native_width=native_width,
            native_height=native_height,
            tier_divisor=tier_divisor,
            window_frames=window_frames,
            batch_size=batch_size,
        )

        # Probe supply
        probe_fn = probe_supply_fn or probe_supply
        supply = probe_fn()

        # Get headroom settings (real parser flags, not getattr defaults)
        device_headroom_frac = float(args.streaming_payload_budget_device_headroom_frac)
        host_headroom_frac = float(args.streaming_payload_budget_host_headroom_frac)

        # Resolve budget
        budget = resolve_residency_budget(
            demand,
            supply,
            device_headroom_frac=device_headroom_frac,
            host_headroom_frac=host_headroom_frac,
        )

        # Store budget in metadata for later use
        from dataclasses import asdict
        metadata["residency"] = budget
        metadata["source"] = "residency_policy"
        metadata["native_resolution"] = f"{native_width}x{native_height}"

        frames = budget.admitted_payload_max_frames
        hot_bytes = budget.hot_tier_max_bytes
        metadata["admitted_payload_max_frames"] = frames
        metadata["hot_tier_max_bytes"] = hot_bytes
        return frames, hot_bytes, metadata

    # If user explicitly passed flags, use them regardless of policy
    if explicit_frames is not None:
        frames = int(explicit_frames)
        metadata["source"] = "explicit_flag"
    elif policy == "legacy":
        frames = legacy_frames
        metadata["source"] = "legacy_policy_default"
    else:  # resolution_scaled
        # Derive from native resolution using the measured law:
        # 512 frames @ 640x480 baseline; scale with 1/pixels
        if native_width and native_height:
            native_pixels = native_width * native_height
            baseline_pixels = 640 * 480
            frames = max(64, min(512, round(512 * baseline_pixels / native_pixels)))
            metadata["native_resolution"] = f"{native_width}x{native_height}"
            metadata["source"] = "resolution_scaled"
        else:
            # Fallback if resolution not available yet
            frames = legacy_frames
            metadata["source"] = "unknown_resolution_fallback"

    metadata["admitted_payload_max_frames"] = frames

    # Resolve hot tier bytes
    if explicit_hot_bytes is not None:
        hot_bytes = int(explicit_hot_bytes)
        metadata["hot_tier_source"] = "explicit_flag"
    elif policy == "legacy":
        hot_bytes = legacy_hot_bytes
        metadata["hot_tier_source"] = "legacy_policy_default"
    else:  # resolution_scaled
        # 20% of device memory
        probe = probe_device_memory or _probe_device_total_memory
        try:
            total_memory = probe()
        except Exception as e:
            print(
                f"[payload-budget] Warning: failed to query CUDA device memory: {e}; "
                f"hot tier remains unbounded",
                flush=True,
            )
            hot_bytes = legacy_hot_bytes
            metadata["hot_tier_source"] = "cuda_query_failed"
        else:
            if total_memory is None:
                hot_bytes = legacy_hot_bytes
                metadata["hot_tier_source"] = "no_cuda_fallback"
            else:
                hot_bytes = int(0.20 * total_memory)
                metadata["device_total_memory"] = total_memory
                metadata["hot_tier_source"] = "20pct_device_memory"

    metadata["hot_tier_max_bytes"] = hot_bytes
    return frames, hot_bytes, metadata


def _tracker_fold_mode(args: Any) -> str:
    """Shared parsing of ``--streaming_tracker_discontinuity_fold`` --
    see ``eligsir/mapping/tracker_discontinuity_fold.py::
    normalize_tracker_fold_mode`` (imported lazily to avoid this module
    paying that one's torch-adjacent import cost for callers that never
    touch tracker-fold config)."""

    from eligsir.mapping.tracker_discontinuity_fold import (
        normalize_tracker_fold_mode,
    )

    return normalize_tracker_fold_mode(
        getattr(args, "streaming_tracker_discontinuity_fold", "off")
    )


def _map_deformation_mode(args: Any) -> str:
    """Shared parsing of ``--streaming_map_deformation`` -- see
    ``eligsir/mapping/anchored_map_deformation.py::
    normalize_map_deformation_mode`` (imported lazily, same rationale as
    ``_tracker_fold_mode`` above)."""

    from eligsir.mapping.anchored_map_deformation import (
        normalize_map_deformation_mode,
    )

    return normalize_map_deformation_mode(
        getattr(args, "streaming_map_deformation", "off")
    )


def depth_insertion_policy_from_args(args: Any) -> DepthInsertionPolicy:
    """Materialize every insertion control from the full parser namespace."""
    # Direct unit callers may supply only the controls under test; production
    # always passes the full parsed namespace.  Complete test namespaces from
    # the parser once here, never with per-field literal fallbacks.
    from eligsir.cli.parser import parse_training_args
    parser_args, _ = parse_training_args(["-s", "/dev/null"])
    args = SimpleNamespace(**(vars(parser_args) | vars(args)))
    return DepthInsertionPolicy(
        min_depth_m=float(args.streaming_min_depth), max_depth_m=float(args.streaming_max_depth),
        voxel_size_m=float(args.streaming_sparse_seed_voxel_size),
        minimum_confidence=float(args.streaming_sparse_seed_min_confidence),
        edge_threshold=float(args.streaming_depth_edge_threshold), max_view_angle_deg=float(args.streaming_max_view_angle),
        stability_filter_enabled=bool(args.streaming_depth_filter_enabled), stability_erode_px=int(args.streaming_depth_filter_erode_px),
        stability_window_px=int(args.streaming_depth_filter_window), stability_min_valid_ratio=float(args.streaming_depth_filter_min_valid_ratio),
        stability_max_range_m=float(args.streaming_depth_filter_max_range), stability_median_threshold_m=float(args.streaming_depth_filter_median_thresh),
        stability_bottom_margin=float(args.streaming_depth_filter_bottom_margin), temporal_check_enabled=bool(args.streaming_depth_temporal_check),
        temporal_tolerance_m=float(args.streaming_depth_temporal_thresh), require_temporal_stability=bool(args.streaming_depth_temporal_require_for_insert),
        require_temporal_for_sensor_closer=bool(args.streaming_depth_temporal_require_for_sensor_closer), occupancy_distance_m=float(args.streaming_depth_consistency_thresh),
        alpha_threshold=float(args.streaming_insert_alpha_threshold), score_mode=str(args.streaming_insert_score_mode),
        surface_fusion_enabled=bool(args.streaming_insert_surface_fusion), surface_fusion_neighbors=int(args.streaming_insert_surface_fusion_k),
        surface_fusion_radius_multiplier=float(args.streaming_insert_surface_fusion_max_radius_mult), knn_dedup_enabled=bool(args.streaming_insert_knn_dedup),
        knn_normal_aware=bool(args.streaming_insert_knn_dedup_normal_aware), knn_max_existing=int(args.streaming_insert_knn_max_existing),
        knn_radius_factor=float(args.streaming_insert_knn_radius_factor), knn_normal_cosine=float(args.streaming_insert_knn_dedup_normal_cos),
        scale_multiplier=float(args.streaming_insert_scale_mult), scale_max_m=float(args.streaming_insert_scale_max),
        normal_scale_ratio=float(args.streaming_insert_normal_scale_ratio), distance_scale_enabled=bool(args.streaming_insert_distance_scale),
        distance_scale_coefficient=float(args.streaming_insert_distance_scale_coeff), isotropic_scale=bool(args.streaming_insert_isotropic_scale),
        score_alpha_weight=float(args.streaming_insert_score_alpha_weight), score_no_depth_weight=float(args.streaming_insert_score_no_depth_weight),
        score_sensor_closer_weight=float(args.streaming_insert_score_sensor_closer_weight), initial_opacity=float(args.streaming_insert_opacity),
        normal_estimation=str(args.gpu_frame_prep_normals), debug_statistics=bool(args.streaming_insertion_debug),
        depth_sparsity_schedule_mode=str(args.streaming_depth_sparsity_schedule),
        depth_sparsity_start_stride=int(args.streaming_depth_sparsity_start_stride),
        depth_sparsity_end_stride=int(args.streaming_depth_sparsity_end_stride),
        depth_sparsity_ramp_frames=int(args.streaming_depth_sparsity_ramp_frames),
    )


def depth_loss_policy_from_args(args: Any) -> DepthLossPolicy:
    weight = float(args.streaming_depth_loss_weight)
    loss_type = str(args.streaming_depth_loss_type)
    return DepthLossPolicy(enabled=weight > 0.0, loss_type=loss_type, weight=weight,
                           robust_delta=0.1 if loss_type == "huber" else None)


def mapping_runtime_config_from_args(args: Any) -> MappingRuntimeConfig:
    """Freeze the canonical method contract at the composition root."""

    from dataclasses import fields
    from eligsir.cli.parser import parse_training_args
    from eligsir.mapping.scheduler.backpressure import backpressure_config_from_args
    from eligsir.mapping.scheduler.keyframes import keyframe_admitter_config_from_args

    backpressure = backpressure_config_from_args(args)
    keyframe = keyframe_admitter_config_from_args(args)
    backpressure_values = {
        item.name: getattr(backpressure, item.name) for item in fields(backpressure)
    }
    keyframe_values = {
        item.name: getattr(keyframe, item.name)
        for item in fields(keyframe)
        if item.name != "config_warnings"
    }
    tensor_candidate_path = not bool(args.streaming_surface_cell_dedup) and not bool(
        args.streaming_insert_confidence_opacity
    )
    schedule_rgb_pixels = args.sh_degree_schedule_rgb_pixels
    if isinstance(schedule_rgb_pixels, str):
        schedule_rgb_pixels = tuple(
            int(value) for value in schedule_rgb_pixels.split(",") if value.strip()
        )
    else:
        schedule_rgb_pixels = tuple(int(value) for value in schedule_rgb_pixels)
    # Resolve the legacy backend parameter groups here.  The raw extracted
    # objects remain constructor-only compatibility inputs; execution receives
    # these immutable scalar projections instead.
    _, parameter_bundle = parse_training_args(["-s", "/dev/null"])
    optimizer_values = vars(parameter_bundle.optimization_params.extract(args))
    pipeline_values = vars(parameter_bundle.pipeline_params.extract(args))
    rasterize_mode = pipeline_values.get("rasterize_mode")
    if not isinstance(rasterize_mode, str):
        raise TypeError("rasterize_mode must be a string")
    if rasterize_mode not in SUPPORTED_RASTERIZE_MODES:
        raise ValueError(
            "rasterize_mode must be exactly one of: antialiased, classic"
        )

    # The characterized optimizer/MCMC implementation still accepts an
    # attribute-shaped policy object.  Select scalar numerical controls only;
    # process/source/composition keys and the retired engine switch never enter
    # this immutable adapter.
    optimization_values = dict(optimizer_values)
    # This typed non-scalar is projected into CanonicalRuntimePolicy below;
    # it must not leak into the scalar-only legacy optimizer adapter.
    optimization_values.pop("sh_degree_schedule_rgb_pixels", None)
    optimization_values.update({
        str(name): value
        for name, value in vars(args).items()
        if isinstance(value, (bool, int, float, str, type(None)))
        and name != "streaming_optimization_engine"
        and (
            name.startswith(("streaming_", "energy_", "mcmc_", "compile_"))
            or name
            in {
                "cap_max", "densify_from_iter", "densify_until_iter",
                "densification_interval", "iterations", "noise_lr",
                "noise_max_scale", "noise_max_step", "online_mode",
                "opacity_reset_interval", "scale_moderate_threshold",
                "scalar_log_interval", "soft_lifecycle_enabled",
                "surface_mcmc_normal_scale", "surface_mcmc_tangent_scale",
                "surface_mcmc_provisional_normal_scale", "support_threshold",
                "gsplat_mcmc_noise_max_scale", "gsplat_mcmc_noise_max_step",
            }
        )
    })
    legacy_iteration_cadence = {
        "densification_interval",
        "densify_from_iter",
        "densify_until_iter",
        "mcmc_stop_growth_iter",
        "mcmc_noise_stop_iter",
        "mcmc_grow_interval_min",
        "mcmc_grow_interval_max",
        "mcmc_relocate_interval_min",
        "mcmc_relocate_interval_max",
        "mcmc_grow_tau",
        "mcmc_relocate_tau",
        "mcmc_growth_factor_tau",
        "mcmc_dead_opacity_power",
        "mcmc_cap_interval_strength",
        "mcmc_cap_interval_power",
        "mcmc_target_q_start",
        "mcmc_target_q_end",
        "mcmc_target_tau",
        "streaming_growth_stop_horizon",
        "streaming_noise_stop_horizon",
    }
    for name in legacy_iteration_cadence:
        optimization_values.pop(name, None)
    pipeline_values.update(
        {
            "gsplat_sparse_grad": bool(args.gsplat_sparse_grad),
            "sparse_mode_detach_sh_dir": bool(args.sparse_mode_detach_sh_dir),
            "compile_sh": bool(args.compile_sh),
        }
    )

    # Issue #1543 finding A: `causal_prepared_ready_b4` silently forces
    # admission prefetch on even when the user never asked for it via
    # `--streaming_admission_prefetch`. Preserve both the raw request and an
    # explicit reason for the effective value alongside it, rather than
    # collapsing them into one boolean that hides why prefetch is (or isn't)
    # actually running. `CanonicalRuntimePolicy.admission_prefetch_enabled`
    # is derived from `admission_prefetch_reason`, so only the two
    # independent facts -- requested, and why the effective value may
    # differ -- are computed and passed here.
    admission_prefetch_requested = bool(getattr(args, "streaming_admission_prefetch", False))
    admission_prefetch_forced = bool(getattr(args, "causal_prepared_ready_b4", False))
    admission_prefetch_reason = (
        ADMISSION_PREFETCH_REASON_EXPLICIT
        if admission_prefetch_requested
        else ADMISSION_PREFETCH_REASON_PREPARED_B4_REQUIRED
        if admission_prefetch_forced
        else ADMISSION_PREFETCH_REASON_DISABLED
    )

    depth_insertion = depth_insertion_policy_from_args(args)
    replay_depth_voxel_size_m = float(args.streaming_replay_depth_voxel_size)
    if replay_depth_voxel_size_m <= 0.0:
        replay_depth_voxel_size_m = depth_insertion.voxel_size_m
    spatial_cell_size_m = float(args.streaming_spatial_replay_focus_cell_size)
    spatial_interval = int(args.streaming_spatial_replay_focus_interval)
    if spatial_cell_size_m <= 0.0:
        raise ValueError("streaming_spatial_replay_focus_cell_size must be positive")
    if spatial_interval <= 0:
        raise ValueError("streaming_spatial_replay_focus_interval must be positive")
    mapping_resolution_tier = str(args.streaming_mapping_resolution_tier).lower()
    if mapping_resolution_tier not in {
        "adaptive", "native", "half", "quarter", "eighth"
    }:
        raise ValueError(
            "streaming_mapping_resolution_tier must be adaptive, native, half, "
            "quarter, or eighth"
        )
    refinement_resolution_policy = str(
        getattr(args, "streaming_refinement_resolution_policy", "native")
    ).lower()
    if refinement_resolution_policy not in {"native", "bounded", "progressive"}:
        raise ValueError(
            "streaming_refinement_resolution_policy must be native, bounded, "
            "or progressive"
        )
    refinement_progressive_switch_frac = float(
        getattr(args, "streaming_refinement_progressive_switch_frac", 0.5)
    )
    if not (0.0 <= refinement_progressive_switch_frac <= 1.0):
        raise ValueError(
            "streaming_refinement_progressive_switch_frac must be in [0, 1]"
        )
    mapping_resolution_min_tier = str(
        args.streaming_mapping_resolution_min_tier
    ).lower()
    if mapping_resolution_min_tier not in {"native", "half", "quarter", "eighth"}:
        raise ValueError(
            "streaming_mapping_resolution_min_tier must be native, half, "
            "quarter, or eighth"
        )
    _resolution_factor = {"native": 1, "half": 2, "quarter": 4, "eighth": 8}
    if (
        mapping_resolution_tier != "adaptive"
        and _resolution_factor[mapping_resolution_tier]
        > _resolution_factor[mapping_resolution_min_tier]
    ):
        raise ValueError(
            "streaming_mapping_resolution_tier cannot be below "
            "streaming_mapping_resolution_min_tier"
        )
    adaptive_growth = bool(args.streaming_adaptive_capacity)
    if adaptive_growth and not bool(args.streaming_spatial_replay_focus):
        raise ValueError(
            "canonical adaptive capacity requires training-only spatial replay focus"
        )
    live_mode = bool(args.online_mode) or str(args.source_path).startswith("ros2://")
    raw_camera_count_max = args.streaming_optimization_camera_count_max
    if type(raw_camera_count_max) is not int:
        raise TypeError(
            "streaming_optimization_camera_count_max must be an exact integer"
        )
    camera_count_max = int(raw_camera_count_max)
    if camera_count_max < 0:
        raise ValueError("streaming_optimization_camera_count_max must be non-negative")
    if camera_count_max > 0:
        validate_supervision_camera_count(camera_count_max)
    raw_transaction_budget_ms = args.streaming_optimization_transaction_budget_ms
    if type(raw_transaction_budget_ms) is not int:
        raise TypeError(
            "streaming_optimization_transaction_budget_ms must be an exact integer"
        )
    transaction_budget_ms = int(raw_transaction_budget_ms)
    if transaction_budget_ms < 0:
        raise ValueError(
            "streaming_optimization_transaction_budget_ms must be non-negative"
        )
    transaction_budget_ns = (
        transaction_budget_ms * 1_000_000
        if transaction_budget_ms > 0
        else (
            LIVE_OPTIMIZATION_TRANSACTION_FRESHNESS_NS
            if live_mode
            else None
        )
    )
    constellation_policy = args.streaming_optimization_constellation_policy
    if not isinstance(constellation_policy, str):
        raise TypeError(
            "streaming_optimization_constellation_policy must be a string"
        )
    if constellation_policy not in {"adaptive", "fixed"}:
        raise ValueError(
            "streaming_optimization_constellation_policy must be adaptive or fixed"
        )

    legacy_population_controls = legacy_population_control_values(args)
    resolved_cold_payload_max_frames, resolved_cold_payload_max_bytes = (
        resolve_cold_payload_bounds(args)
    )
    # Resolve payload budget: admitted frames and hot-tier bytes
    resolved_admitted_payload_frames, resolved_hot_tier_bytes, _payload_budget_metadata = (
        resolve_admitted_payload_bounds(
            args,
            native_width=getattr(args, "_resolved_native_width", None),
            native_height=getattr(args, "_resolved_native_height", None),
        )
    )
    return MappingRuntimeConfig(
        model_layout=str(args.model_layout).lower(),
        renderer="gsplat",
        numerical_mode=str(args.optimizer_type),
        hard_limits={
            "cap_max": None if args.cap_max is None else int(args.cap_max),
            "optimization_batch_size": int(args.streaming_optimization_batch_size),
            "optimization_camera_count_max": camera_count_max,
            "max_new_gaussians_per_frame": int(
                args.streaming_max_new_gaussians_per_frame
            ),
        },
        residency={
            "live_payload_max_frames": resolved_admitted_payload_frames,
            "cold_payload_max_frames": resolved_cold_payload_max_frames,
            "cold_payload_max_bytes": resolved_cold_payload_max_bytes,
            "sensor_depth_cache_max_frames": int(
                args.streaming_sensor_depth_cache_max_frames
            ),
        },
        depth_loss=depth_loss_policy_from_args(args),
        depth_insertion=depth_insertion,
        device_capabilities=CanonicalDeviceCapabilities(
            accelerator="cuda",
            gpu_frame_prep=bool(args.gpu_frame_prep),
            gpu_frame_resident=bool(args.gpu_frame_resident),
            tensor_candidate_path=tensor_candidate_path,
        ),
        bootstrap=CanonicalBootstrapPolicy(
            depth_stride=int(args.streaming_depth_stride),
            min_depth_m=float(args.streaming_min_depth),
            max_depth_m=float(args.streaming_max_depth),
            max_points=int(args.rgbd_max_init_points),
            seed_voxel_size_m=float(args.streaming_seed_voxel_size),
            init_scale_mode=str(args.init_scale_mode),
            init_scale=float(args.init_scale),
            model_voxel_size_m=float(args.pcd_voxel_size),
            occupancy_voxel_size_m=float(args.streaming_insert_voxel_size),
            temporal_check_enabled=bool(args.streaming_depth_temporal_check),
            temporal_tolerance_m=float(args.streaming_depth_temporal_thresh),
            require_temporal_for_seed=bool(
                args.streaming_depth_temporal_require_for_seed
            ),
        ),
        runtime=CanonicalRuntimePolicy(
            model_path=str(args.model_path),
            cap_max=None if args.cap_max is None else int(args.cap_max),
            max_new_gaussians_per_frame=int(args.streaming_max_new_gaussians_per_frame),
            max_active_supervision_frames=max(
                1,
                min(
                    int(args.streaming_resolution_cache_max_frames),
                    int(args.streaming_keyframe_window),
                ),
            ),
            hot_tier_cache_max_bytes=resolved_hot_tier_bytes,
            max_tiers_per_frame=max(
                1,
                int(
                    getattr(
                        args,
                        "streaming_hot_tier_max_tiers_per_frame",
                        MAX_RETAINED_TIERS_PER_FRAME,
                    )
                ),
            ),
            sensor_depth_cache_max_frames=int(args.streaming_sensor_depth_cache_max_frames),
            live_payload_max_frames=resolved_admitted_payload_frames,
            cold_payload_max_frames=resolved_cold_payload_max_frames,
            cold_payload_max_bytes=resolved_cold_payload_max_bytes,
            keyframe_window=int(args.streaming_keyframe_window),
            offload_camera_images=bool(args.streaming_offload_camera_images),
            admission_prefetch_requested=admission_prefetch_requested,
            admission_prefetch_reason=admission_prefetch_reason,
            admission_prematerialize_enabled=bool(getattr(args, "streaming_admission_prematerialize", False)),
            blur_downsample=int(args.streaming_keyframe_blur_downsample),
            target_capacity=int(args.streaming_target_capacity),
            max_transfer_bytes=int(args.streaming_max_transfer_bytes),
            sh_degree_schedule_rgb_pixels=schedule_rgb_pixels,
            selective_adam_allow_dense_grads=bool(args.selective_adam_allow_dense_grads),
            random_background=bool(args.random_background),
            white_background=bool(args.white_background),
            gsplat_sparse_grad=bool(args.gsplat_sparse_grad),
            sh_update_interval=int(args.sh_update_interval),
            densification_strategy=str(args.densification_strategy).lower(),
            densify_from_iter=int(args.densify_from_iter),
            # Legacy artifact horizon retained for compatibility labels only.
            # Canonical mutation actuation uses the semantic cadence owner.
            mcmc_stop_growth_iter=int(args.mcmc_stop_growth_iter),
            densify_until_iter=int(args.densify_until_iter),
            mcmc_grow_interval_pixels=int(args.mcmc_grow_interval_pixels),
            mcmc_post_ingestion_grow_interval_pixels=int(
                args.streaming_mcmc_post_ingestion_grow_interval_pixels
            ),
            mcmc_relocate_interval_pixels=int(args.mcmc_relocate_interval_pixels),
            mcmc_post_ingestion_relocate_interval_pixels=int(
                args.streaming_mcmc_post_ingestion_relocate_interval_pixels
            ),
            mcmc_min_optimizer_updates_after_structural_mutation=int(
                args.mcmc_min_optimizer_updates_after_structural_mutation
            ),
            mcmc_noise_stop_optimizer_updates=int(
                args.mcmc_noise_stop_optimizer_updates
            ),
            mcmc_temperature_horizon_optimizer_updates=int(
                args.mcmc_temperature_horizon_optimizer_updates
            ),
            mcmc_growth_factor_start=float(args.mcmc_growth_factor_start),
            max_optimization_transaction_ns=transaction_budget_ns,
        ),
        optimization=CanonicalOptimizationPolicy(optimization_values),
        pipeline=CanonicalPipelinePolicy(pipeline_values),
        backpressure_policy=backpressure_values,
        keyframe_policy=keyframe_values,
        static_options={
            "optimization_constellation_policy": constellation_policy,
            "online_mode": bool(args.online_mode),
            "live_mode": live_mode,
            "frame_admission": str(args.streaming_frame_admission),
            "initial_frames": int(args.streaming_initial_frames),
            "depth_stride": int(args.streaming_depth_stride),
            "insert_from_depth": bool(args.streaming_insert_from_depth),
            "prepared_ready_source_insertion": bool(
                getattr(args, "causal_prepared_ready_source_insertion", False)
            ),
            "admission_prematerialize": bool(
                getattr(args, "streaming_admission_prematerialize", False)
            ),
            "seed": int(args.seed),
            "replay_buffer": int(args.streaming_replay_buffer),
            "replay_global_weight_floor": float(
                args.streaming_replay_global_weight_floor
            ),
            "replay_target_views_per_s": float(
                args.streaming_replay_target_views_per_s
            ),
            "resolution_budget_relax_max": float(
                args.streaming_resolution_budget_relax_max
            ),
            "mapping_resolution_tier": mapping_resolution_tier,
            "mapping_resolution_min_tier": mapping_resolution_min_tier,
            "refinement_resolution_policy": refinement_resolution_policy,
            "refinement_progressive_switch_frac": refinement_progressive_switch_frac,
            "replay_depth_max_per_event": int(
                args.streaming_replay_depth_max_per_event
            ),
            "replay_depth_interval": int(args.streaming_replay_depth_interval),
            "replay_depth_voxel_size_m": replay_depth_voxel_size_m,
            "replay_depth_relaxed_alpha_threshold": float(
                args.streaming_replay_depth_relaxed_alpha_threshold
            ),
            "replay_depth_relaxed_max_depth_m": float(
                args.streaming_replay_depth_relaxed_max_depth
            ),
            "spatial_replay_focus": bool(args.streaming_spatial_replay_focus),
            "partition_replay_focus": bool(args.streaming_partition_replay_focus),
            "partition_focus_actuation_reaches_active_sources": bool(
                getattr(args, "streaming_partition_focus_spatial_substrate", False)
            ),
            "spatial_replay_focus_cell_size_m": spatial_cell_size_m,
            "spatial_replay_focus_interval": spatial_interval,
            "adaptive_growth": adaptive_growth,
            "adaptive_growth_u_brake": float(args.streaming_adcap_u_brake),
            "adaptive_growth_u_release": float(args.streaming_adcap_u_release),
            "adaptive_growth_hysteresis": int(args.streaming_adcap_hysteresis),
            "adaptive_growth_min_population_delta": int(
                args.streaming_adcap_min_dN
            ),
            "adaptive_growth_ema": float(args.streaming_adcap_ema),
            "adaptive_growth_regression_tolerance_db": float(
                args.streaming_partition_replay_focus_regression_tolerance_db
            ),
            "population_shadow_enabled": bool(
                args.streaming_population_shadow_enabled
            ),
            "population_live_enabled": bool(
                args.streaming_population_live_enabled
            ),
            # Live population actuation is only valid when the canonical
            # compatibility strategy remains the sole ordinary mutation route.
            # Retain the legacy controls in the immutable contract so its
            # fail-closed boundary cannot be bypassed after argument parsing.
            "population_ownership_strategy": str(args.densification_strategy).lower(),
            "population_control_mode": (
                "live_probabilistic"
                if bool(args.streaming_population_live_enabled)
                else "legacy_compatibility"
            ),
            "legacy_population_controls_status": LEGACY_POPULATION_CONTROLS_STATUS,
            **legacy_population_controls,
            "keyframe_min_translation_m": float(
                args.streaming_keyframe_min_translation
            ),
            "keyframe_min_rotation_deg": float(
                args.streaming_keyframe_min_rotation_deg
            ),
            "resolution_resampling_mode": str(
                getattr(args, "streaming_resolution_resampling_mode", "legacy")
            ),
            "resolution_discontinuity_abs_floor_m": float(
                getattr(args, "streaming_resolution_discontinuity_abs_floor_m", 0.05)
            ),
            "resolution_discontinuity_rel_frac": float(
                getattr(args, "streaming_resolution_discontinuity_rel_frac", 0.02)
            ),
            "depth_loss_downsample": str(
                getattr(args, "streaming_depth_loss_downsample", "nearest")
            ),
            # Absence of _resolved_prefetch_entries means the residency budget
            # policy was not resolved (policy is not "residency" or explicit
            # frames were already set). This is an internal sentinel, not a flag.
            "_resolved_prefetch_entries": (
                args._resolved_prefetch_entries if hasattr(args, "_resolved_prefetch_entries") else None
            ),
        },
    )


def depth_supervision_enabled(work: Any, *, depth_loss_configured: bool) -> bool:
    return bool(depth_loss_configured and work.modality_eligibility.depth_supervision)


def motion_admission_evidence(translation_m: float, rotation_deg: float, *, min_translation_m: float, min_rotation_deg: float) -> bool:
    return bool(float(translation_m) >= float(min_translation_m) or float(rotation_deg) >= float(min_rotation_deg))


def scheduled_b1_work(plan: MappingWorkPlan) -> tuple[Any, ...]:
    work = tuple(plan.supervision_work) + tuple(plan.replay_work)
    if not work:
        return ()
    if len(work) > plan.optimizer_updates:
        raise ValueError("work plan schedules more B=1 views than optimizer updates")
    return tuple(work[index % len(work)] for index in range(plan.optimizer_updates))
