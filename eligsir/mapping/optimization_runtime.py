"""Streaming composition for one fixed-topology optimization step.

Selected camera images are prefetched through the composition-owned transfer
path. Every B>1 request carries an explicit execution mode: sequential
gradient accumulation or true C-camera gsplat rasterization. A planned true
batch fails closed when unsupported. This boundary publishes continuous
assimilation support and a control-plane digest record for every optimizer
batch without hashing tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch

from eligsir.map.gpu_transactions import BatchExecutionMode
from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.batch_contract import (
    validate_supervision_batch_size,
    validate_supervision_camera_count,
)
from eligsir.mapping.population_shadow_runtime import (
    PopulationShadowRuntime,
    PopulationShadowDropReason,
    PopulationShadowVersions,
)
from eligsir.optimization.gaussian.contracts import (
    ActiveSetState,
    AnchorScheduleState,
    AnchorState,
    BubbleState,
    DepthLockState,
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationView,
    RandomStateToken,
    resolve_supervision_modality,
    StableGeometryState,
)
from eligsir.runtime.deterministic_digest import record_global
from eligsir.utils.ingest_phase_profiler import _timed as _phase_timed
from eligsir.utils.streaming_depth_lock import depth_lock_enabled
from eligsir.utils.partition_eval import camera_source_index


class CameraTransferOwner(Protocol):
    """Canonical camera-pixel transfer path injected by composition."""

    def prefetch_cameras(self, cameras: Sequence[Any]) -> int:
        """Submit camera pixels and bind their completion to compute."""


@dataclass
class StreamingOptimizationRuntime:
    """Inputs and mutable bookkeeping for one streaming optimizer owner."""

    session: Any
    gaussians: Any
    args: Any
    opt: Any
    pipe: Any
    mcmc_strategy: Any
    tb_writer: Any
    optimizer_type: str
    sparse_active_set: bool
    use_energy_mcmc: bool
    fixed_shape_mcmc: bool
    mcmc_local: bool
    loss_assembly_kwargs: dict[str, Any]
    iter_since_new_frame: int
    post_render_hook: Any = None
    placement_only: bool = False
    random_state_token: RandomStateToken | None = None
    post_optimizer_hook: Any = None
    post_mcmc_hook: Any = None
    timing_scope: Any = None
    camera_residency_ledger: Any = None
    camera_transfer_owner: CameraTransferOwner | None = None
    # Optional loop-owned optimizer ceiling. None preserves the historical
    # opt.iterations ceiling; an explicit post-mapping tail can extend Adam
    # without changing opt.iterations-derived mapping schedules.
    optimizer_iteration_limit: int | None = None
    latest_admitted_optimizer_update: int | None = None
    # M5: read-only population-policy observation.  The scheduler must inject
    # a neutral token provider; this runtime deliberately has no grow/relocate
    # cadence interpretation and does not adapt a legacy MCMC due signal.
    population_shadow_enabled: bool = False
    population_shadow_work_token_provider: Any = None
    population_shadow_versions_provider: Any = None
    population_shadow_belief_provider: Any = None
    # Optional pre-render source of the five non-map snapshot values used by
    # E0.  Production supplies it only for the explicitly enabled M5 shadow
    # bridge; absent or invalid values deliberately leave evidence unavailable.
    population_shadow_evidence_versions_provider: Any = None
    population_shadow_capacity: int | None = None
    population_shadow_runtime: PopulationShadowRuntime | None = None
    # Canonical production commits work clocks after the engine transaction.
    # Defer observation until that commit so its optimizer version is current.
    population_shadow_defer_until_post_step_commit: bool = False
    last_population_shadow_drop_reason: PopulationShadowDropReason | None = field(
        default=None, init=False
    )
    last_population_shadow_stale_current_mismatches: tuple[tuple[str, int, int], ...] = field(
        default=(), init=False
    )
    _population_shadow_completed_context: OptimizationRuntimeContext | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if type(self.population_shadow_enabled) is not bool:
            raise TypeError("population_shadow_enabled must be bool")
        if type(self.population_shadow_defer_until_post_step_commit) is not bool:
            raise TypeError("population_shadow_defer_until_post_step_commit must be bool")
        if self.population_shadow_capacity is not None and (
            type(self.population_shadow_capacity) is not int
            or self.population_shadow_capacity < 0
        ):
            raise ValueError("population_shadow_capacity must be a non-negative exact integer")
        if self.population_shadow_runtime is not None and not isinstance(
            self.population_shadow_runtime, PopulationShadowRuntime
        ):
            raise TypeError("population_shadow_runtime must be PopulationShadowRuntime or None")
        if self.population_shadow_enabled:
            # Shadow and live policy consume the same strict belief view.  The
            # dedicated E0 buffers must therefore exist before either mode
            # asks the provider for row evidence.
            from eligsir.mapping.gaussian_belief_accumulator import (
                ensure_population_policy_e0_buffers,
            )

            # Lightweight unit/shadow adapters may provide belief directly and
            # deliberately have no Gaussian row owner.  Real row owners are
            # initialized here in both shadow-only and live configurations.
            if isinstance(getattr(self.gaussians, "get_xyz", None), torch.Tensor):
                ensure_population_policy_e0_buffers(self.gaussians)
        if self.population_shadow_runtime is None and self.population_shadow_enabled:
            self.population_shadow_runtime = PopulationShadowRuntime(
                enabled=True,
                transfer_owner=self.camera_transfer_owner,
            )
        elif self.population_shadow_runtime is not None and (
            self.population_shadow_runtime.enabled != self.population_shadow_enabled
        ):
            raise ValueError("population shadow runtime enabled state must match configuration")
        elif self.population_shadow_runtime is not None and self.camera_transfer_owner is not None:
            self.population_shadow_runtime.transfer_owner = self.camera_transfer_owner

    def _evidence_versions(self) -> PopulationShadowVersions | None:
        provider = self.population_shadow_evidence_versions_provider
        try:
            candidate = provider() if callable(provider) else None
        except Exception:
            candidate = None
        return candidate if isinstance(candidate, PopulationShadowVersions) else None

    def build_step(
        self,
        *,
        camera: Any,
        background: Any,
        iteration: int,
        map_version: int,
        topology_version: int,
        resolution_bucket: str,
        update_sh_rest: bool,
        render_depth: bool,
        n_frames_ingested: int,
        visible_mask: Any,
        random_state_token: RandomStateToken,
        optimizer_update_id: int | None = None,
        optimizer_update_permitted: bool | None = None,
        modality_eligibility: ModalityEligibility | None = None,
        _shared_context: OptimizationRuntimeContext | None = None,
    ) -> tuple[OptimizationStepRequest, OptimizationRuntimeContext]:
        """Build the typed request and policy context for one camera step."""

        if optimizer_update_id is not None:
            if type(optimizer_update_id) is not int or optimizer_update_id < 1:
                raise ValueError("optimizer_update_id must be a positive exact integer")
            if optimizer_update_permitted is not None and type(optimizer_update_permitted) is not bool:
                raise TypeError("optimizer_update_permitted must be an exact bool or None")
            resolved_optimizer_permission = optimizer_update_permitted is True
            admitted_at = self.latest_admitted_optimizer_update
            if admitted_at is None:
                admitted_at = max(0, optimizer_update_id - self.iter_since_new_frame)
            updates_since_admission = max(0, optimizer_update_id - admitted_at)
        else:
            resolved_optimizer_permission = optimizer_update_permitted
            updates_since_admission = None
        policies = getattr(_shared_context, "gradient_policy_context", None) or GradientPolicyContext(
            gaussians=self.gaussians,
            args=self.args,
            iteration=iteration,
            n_frames_ingested=n_frames_ingested,
            optimizer_update_id=optimizer_update_id,
            active_set=ActiveSetState(
                sparse_active_set=self.sparse_active_set,
                zero_invisible_grads=bool(
                    getattr(self.args, "selective_adam_zero_invisible_grads", True)
                ),
                visible_mask=visible_mask,
            ),
            stable_geometry=StableGeometryState(
                freeze_old_geometry=bool(
                    getattr(self.args, "streaming_freeze_old_geometry", False)
                ),
                freeze_new_frame_steps=max(
                    0, int(getattr(self.args, "streaming_freeze_new_frame_steps", 50))
                ),
                iter_since_new_frame=self.iter_since_new_frame,
                successful_updates_since_admission=updates_since_admission,
                young_age_frames=max(
                    0, int(getattr(self.args, "streaming_young_age_frames", 5))
                ),
                lifecycle_enabled=bool(
                    getattr(self.args, "streaming_lifecycle_enabled", True)
                ),
                bubble=BubbleState(
                    enabled=bool(getattr(self.args, "streaming_spatial_bubble_enabled", False))
                ),
            ),
            anchor=AnchorState(
                bootstrap=AnchorScheduleState(
                    weight=float(getattr(self.args, "streaming_bootstrap_anchor_weight", 1.0)),
                    decay_steps=max(
                        1,
                        int(getattr(self.args, "streaming_bootstrap_anchor_decay_steps", 2000)),
                    ),
                    full_n_draw=bool(getattr(self.args, "streaming_anchor_fullN_draw", True)),
                ),
                insertion=AnchorScheduleState(
                    weight=float(getattr(self.args, "streaming_insertion_anchor_weight", 1.0)),
                    decay_steps=max(
                        1,
                        int(getattr(self.args, "streaming_insertion_anchor_decay_steps", 200)),
                    ),
                    full_n_draw=bool(getattr(self.args, "streaming_anchor_fullN_draw", True)),
                ),
            ),
            depth_lock=DepthLockState(active=depth_lock_enabled(self.args)),
        )
        frame_value_e0_requested = bool(
            getattr(self.args, "streaming_historical_usefulness_tracking", False)
        )
        evidence_versions = self._evidence_versions() if (
            self.population_shadow_enabled or frame_value_e0_requested
        ) else None
        request = OptimizationStepRequest(
            view=OptimizationView(
                camera=camera,
                source_index=camera_source_index(camera),
                source_identity=getattr(
                    camera,
                    "_source_identity",
                    getattr(getattr(camera, "_mapping_frame", None), "source", None),
                ),
                background=background,
                update_sh_rest=update_sh_rest,
                render_depth=render_depth,
                modality_eligibility=(
                    modality_eligibility
                    if modality_eligibility is not None
                    else getattr(camera, "modality_eligibility", None)
                ),
                resolution_bucket=resolution_bucket,
            ),
            iteration=iteration,
            map_version=map_version,
            topology_version=topology_version,
            resolution_bucket=resolution_bucket,
            evidence_requirements=EvidenceRequirements(
                want_energy_grad_snapshot=self.use_energy_mcmc,
                want_supervision_evidence=(
                    evidence_versions is not None
                ),
            ),
            random_state_token=random_state_token,
        )
        return request, (_shared_context or OptimizationRuntimeContext(
            session=self.session,
            gaussians=self.gaussians,
            args=self.args,
            opt=self.opt,
            pipe=self.pipe,
            mcmc_strategy=self.mcmc_strategy,
            gradient_policy_context=policies,
            n_frames_ingested=n_frames_ingested,
            use_energy_mcmc=self.use_energy_mcmc,
            sparse_active_set=self.sparse_active_set,
            optimizer_type=self.optimizer_type,
            fixed_shape_mcmc=self.fixed_shape_mcmc,
            mcmc_local=self.mcmc_local,
            placement_only=self.placement_only,
            tb_writer=self.tb_writer,
            post_render_hook=self.post_render_hook,
            post_optimizer_hook=self.post_optimizer_hook,
            post_mcmc_hook=self.post_mcmc_hook,
            timing_scope=self.timing_scope,
            loss_assembly_kwargs=dict(self.loss_assembly_kwargs),
            optimizer_iteration_limit=self.optimizer_iteration_limit,
            optimizer_update_id=optimizer_update_id,
            optimizer_update_permitted=resolved_optimizer_permission,
            evidence_transaction_id=(
                optimizer_update_id
                if evidence_versions is not None and optimizer_update_id is not None
                else None
            ),
            coordinate_frame_version=(
                None if evidence_versions is None else evidence_versions.coordinate_frame
            ),
            schema_version=(None if evidence_versions is None else evidence_versions.schema),
            surface_evidence_version=(
                None if evidence_versions is None else evidence_versions.surface_evidence
            ),
            policy_version=(None if evidence_versions is None else evidence_versions.policy),
        ))

    def note_admitted_source(self, *, optimizer_updates_completed: int) -> None:
        """Reset canonical stable-geometry age at a real admission event."""

        if type(optimizer_updates_completed) is not int or optimizer_updates_completed < 0:
            raise ValueError("optimizer_updates_completed must be a non-negative exact integer")
        self.latest_admitted_optimizer_update = optimizer_updates_completed
        self.iter_since_new_frame = 0

    def prefetch_selected_cameras(self, cameras: list[Any]) -> int:
        """Move selected camera images to the Gaussian device by shape bucket."""

        if not cameras:
            return 0
        try:
            device = self.gaussians.get_xyz.device
        except Exception:
            return 0
        if torch.device(device).type == "cpu":
            return 0
        transfer_owner = self.camera_transfer_owner
        if transfer_owner is None or not callable(
            getattr(transfer_owner, "prefetch_cameras", None)
        ):
            raise RuntimeError(
                "accelerator streaming optimization requires an injected canonical "
                "camera_transfer_owner with prefetch_cameras(cameras)"
            )
        newly_resident = []
        for cam in cameras:
            image = getattr(cam, "original_image", None)
            if image is None or getattr(getattr(image, "device", None), "type", None) == "cpu":
                newly_resident.append(cam)
        n_buckets = int(transfer_owner.prefetch_cameras(cameras))
        if self.camera_residency_ledger is not None:
            for cam in newly_resident:
                image = getattr(cam, "original_image", None)
                if image is not None and getattr(image.device, "type", None) != "cpu":
                    self.camera_residency_ledger.note_paged_in(cam)
        return n_buckets

    def publish_assimilation_support(
        self,
        iteration: int,
        *,
        optimizer_updates: int = 0,
        topology_version: int = 0,
        parameter_version: int = 0,
    ) -> None:
        """Publish continuous support-debt telemetry when support is available.

        Production: calls publish_support_counts() to reduce and latch the
        summary with provenance. Formatting: reads the latched summary and writes
        TensorBoard scalars, performing zero additional device operations.

        Args:
            iteration: optimizer iteration number
            optimizer_updates: current optimizer update count (for provenance)
            topology_version: current map topology version (for staleness detection)
            parameter_version: current parameter version (for provenance)
        """

        from eligsir.utils.assimilation_pressure import assimilation_control_enabled

        if not assimilation_control_enabled() and (
            self.tb_writer is None
            or int(iteration) % max(
                1, int(self.loss_assembly_kwargs.get("scalar_log_interval", 1))
            )
        ):
            return

        support = getattr(self.gaussians, "support_count", None)
        if support is None or getattr(support, "ndim", 0) == 0:
            return
        if int(support.shape[0]) != int(self.gaussians.get_xyz.shape[0]):
            return
        from eligsir.utils.assimilation_pressure import (
            latest_assimilation_support_summary,
            publish_support_counts,
        )
        min_support = int(getattr(self.args, "streaming_min_support_views", 2))
        if min_support <= 0:
            return
        # Attach provenance (population_size is the current support count tensor size)
        population_size = int(support.shape[0])
        debt = publish_support_counts(
            support,
            min_support_views=min_support,
            optimizer_updates=optimizer_updates,
            topology_version=topology_version,
            parameter_version=parameter_version,
            population_size=population_size,
        )
        # Format using latched summary (zero device ops)
        if self.tb_writer is not None:
            summary = latest_assimilation_support_summary()
            if summary is not None:
                self.tb_writer.add_scalar(
                    "streaming/assimilation_support_debt_continuous", float(summary.debt), int(iteration)
                )
                self.tb_writer.add_scalar(
                    "streaming/assimilation_support_hist/support_eq_0", float(summary.support_eq_0), int(iteration)
                )
                self.tb_writer.add_scalar(
                    "streaming/assimilation_support_hist/support_eq_1", float(summary.support_eq_1), int(iteration)
                )
                self.tb_writer.add_scalar(
                    "streaming/assimilation_support_hist/support_ge_target", float(summary.support_ge_target), int(iteration)
                )
                self.tb_writer.add_scalar(
                    "streaming/assimilation_support_hist/support_pool_n", float(summary.support_pool_n), int(iteration)
                )

    def commit_step(self, context: OptimizationRuntimeContext, result: Any = None) -> None:
        """Persist runtime counters updated through the completed step context."""

        semantic_age = (
            context.gradient_policy_context.stable_geometry
            .successful_updates_since_admission
        )
        if semantic_age is None:
            self.iter_since_new_frame = (
                context.gradient_policy_context.stable_geometry.iter_since_new_frame
            )
        elif result is not None and int(getattr(result, "optimizer_step_count", 0)) > 0:
            self.iter_since_new_frame = int(semantic_age)
        if result is not None:
            if self.population_shadow_defer_until_post_step_commit:
                self._population_shadow_completed_context = context
            else:
                self.observe_population_shadow(context, result)

    def observe_population_shadow_after_post_step_commit(
        self, result: Any, *, native_targets: Any = None
    ) -> PopulationShadowDropReason | None:
        """Publish deferred M5 telemetry after production commits work clocks."""

        context = self._population_shadow_completed_context
        self._population_shadow_completed_context = None
        if context is None:
            return PopulationShadowDropReason.MISSING_CURRENT_VERSIONS
        return self.observe_population_shadow(
            context, result, native_targets=native_targets
        )

    def observe_population_shadow(
        self,
        context: OptimizationRuntimeContext,
        result: Any,
        *,
        native_targets: Any = None,
    ) -> PopulationShadowDropReason | None:
        """Publish an M5 shadow policy observation after a completed step.

        This method is intentionally downstream-only: it neither changes the
        completed result nor forwards ``PopulationDecision``/``EvidenceDemand``
        to the live scheduler, mutation arbiter, MCMC strategy, lifecycle, or
        depth insertion owner.  The injected scheduler adapters supply the
        neutral budget and exact versions; absent adapters fail closed as a
        telemetry drop rather than inventing a semantic decision.
        """

        bridge = self.population_shadow_runtime
        if bridge is None:
            return None
        token_provider = self.population_shadow_work_token_provider
        versions_provider = self.population_shadow_versions_provider
        belief_provider = self.population_shadow_belief_provider
        try:
            token = (
                token_provider(result, context)
                if callable(token_provider)
                else None
            )
        except Exception:
            token = None
        try:
            versions = (
                versions_provider(result, context)
                if callable(versions_provider)
                else None
            )
        except Exception:
            versions = None
        try:
            belief = (
                belief_provider(self.gaussians)
                if callable(belief_provider)
                else None
            )
        except Exception:
            belief = None
        drop_reason = bridge.observe(
            result=result,
            belief=belief,
            token=token,
            current_versions=(
                versions if isinstance(versions, PopulationShadowVersions) else None
            ),
            capacity=self.population_shadow_capacity,
            native_targets=native_targets,
        )
        self.last_population_shadow_drop_reason = drop_reason
        self.last_population_shadow_stale_current_mismatches = (
            bridge.last_stale_current_mismatches
        )
        return drop_reason


def _record_optimization_digest(
    request,
    result,
    selected_cameras,
    skipped_extra_view_count: int,
    *,
    requested_batch_size: int | None = None,
) -> None:
    successful = getattr(result, "successful_supervision_source_indices", None)
    if successful is None:
        successful = tuple(
            idx for idx in (camera_source_index(cam) for cam in selected_cameras) if idx is not None
        )
    record_global(
        "optimizer_batch",
        iteration=int(request.iteration),
        requested_source_indices=[
            camera_source_index(cam) for cam in selected_cameras
        ],
        successful_source_indices=[int(x) for x in successful],
        requested_batch_size=(
            len(selected_cameras) + int(skipped_extra_view_count)
            if requested_batch_size is None
            else int(requested_batch_size)
        ),
        realized_supervision_views=int(getattr(result, "supervision_view_count", 0) or 0),
        skipped_views=int(getattr(result, "skipped_view_count", skipped_extra_view_count) or 0),
        rasterizer_call_count=int(getattr(result, "rasterizer_call_count", 1) or 0),
        supervised_pixel_count=int(getattr(result, "supervised_pixel_count", 0) or 0),
        optimizer_step_count=int(getattr(result, "optimizer_step_count", 0) or 0),
        true_camera_batch=bool(getattr(result, "true_camera_batch", False)),
        true_camera_batch_fallback_reason=getattr(result, "true_camera_batch_fallback_reason", None),
        step_allowed=bool(getattr(result, "step_allowed", False)),
        skip_reason=getattr(result, "skip_reason", None),
        map_version_before=int(getattr(result, "map_version_before", request.map_version)),
        map_version_after=int(getattr(result, "map_version_after", request.map_version)),
        topology_version=int(getattr(result, "topology_version", request.topology_version)),
        resolution_bucket=str(request.resolution_bucket),
    )


def run_streaming_optimization_step(
    runtime: StreamingOptimizationRuntime,
    engine: Any,
    *,
    extra_cameras: Any = None,
    skipped_extra_view_count: int = 0,
    execution_mode: BatchExecutionMode = BatchExecutionMode.SEQUENTIAL_ACCUMULATION,
    **step_kwargs: Any,
) -> Any:
    """Run one B=1 or batched streaming optimization transaction."""

    if not isinstance(execution_mode, BatchExecutionMode):
        raise TypeError("execution_mode must be BatchExecutionMode")
    skipped_extra_view_count = int(skipped_extra_view_count or 0)
    extras = list(extra_cameras or ())
    iteration = step_kwargs.get("iteration")
    if iteration is not None and hasattr(runtime, "publish_assimilation_support"):
        # Issue #279 follow-up (2026-08-31): this call was previously
        # unbracketed on the B=1 path (unlike its batched-path sibling
        # below, which already carries "m_publish_assimilation_support"),
        # so its cost silently fell into m_optimization_transaction's
        # unitemized residual on every ordinary B=1 fr3-style run.
        with _phase_timed("m_publish_assimilation_support"):
            runtime.publish_assimilation_support(
                int(iteration),
                optimizer_updates=step_kwargs.get("optimizer_update_id", 0),
                topology_version=step_kwargs.get("topology_version", 0),
            )

    selected_cameras = [step_kwargs["camera"], *extras]
    if (
        execution_mode is BatchExecutionMode.TRUE_CAMERA_BATCH
        and len(selected_cameras) < 2
    ):
        raise RuntimeError(
            "planned TRUE_CAMERA_BATCH is unavailable; refusing sequential fallback: "
            "requires_at_least_two_materialized_views"
        )
    if hasattr(runtime, "prefetch_selected_cameras"):
        with _phase_timed("m_prefetch_selected_cameras"):
            runtime.prefetch_selected_cameras(selected_cameras)
    with _phase_timed("m_build_step"):
        request, context = runtime.build_step(**step_kwargs)
    post_render_hook = context.post_render_hook

    def _bind_visible_mask(render_outputs: Any) -> None:
        context.gradient_policy_context.active_set.visible_mask = (
            render_outputs.payload["visibility_filter"].detach()
            .to(dtype=torch.bool)
            .contiguous()
        )
        if post_render_hook is not None:
            post_render_hook(render_outputs)

    context.post_render_hook = _bind_visible_mask
    if extras or skipped_extra_view_count:
        requests = [request]
        with _phase_timed("m_build_step"):
            for cam in extras:
                extra_request, _ = runtime.build_step(
                    **dict(step_kwargs, camera=cam, _shared_context=context)
                )
                requests.append(extra_request)

        result = None
        if execution_mode is BatchExecutionMode.TRUE_CAMERA_BATCH:
            from eligsir.mapping.renderer.true_camera_batch import supports_true_camera_batch

            supported, true_batch_reason = supports_true_camera_batch(
                runtime.pipe, [req.view.camera for req in requests]
            )
            if not supported:
                raise RuntimeError(
                    "planned TRUE_CAMERA_BATCH is unavailable; refusing sequential fallback: "
                    f"{true_batch_reason}"
                )
            from eligsir.optimization.gaussian.true_batch_executor import (
                run_true_camera_batched_step,
            )

            result = run_true_camera_batched_step(
                engine,
                requests,
                context,
                skipped_view_count=skipped_extra_view_count,
            )
            _set_result_metadata(
                result,
                true_camera_batch=True,
                true_camera_batch_fallback_reason=None,
            )

        if result is None:
            result = engine.run_batched_step(
                requests,
                context,
                skipped_view_count=skipped_extra_view_count,
            )
            _set_result_metadata(
                result,
                true_camera_batch=False,
                true_camera_batch_fallback_reason="planned_sequential_accumulation",
            )
    else:
        result = engine.run_step(request, context)
        _attach_b1_supervision_evidence(result, request, context)
        _set_result_metadata(result, true_camera_batch=False,
                             true_camera_batch_fallback_reason="B_equals_1")
    with _phase_timed("m_commit_step"):
        runtime.commit_step(context, result)
    runtime.random_state_token = getattr(result, "random_state_token", None)
    with _phase_timed("m_record_digest"):
        _record_optimization_digest(request, result, selected_cameras, skipped_extra_view_count)
    return result


def run_streaming_optimization_transaction(
    runtime: StreamingOptimizationRuntime,
    engine: Any,
    *,
    cameras: Sequence[Any],
    requested_batch_size: int | None = None,
    requested_camera_count: int,
    modality_eligibilities: Sequence[Any] | None = None,
    **step_kwargs: Any,
) -> Any:
    """Run one adaptive B/C transaction without waiting for requested fill.

    The caller supplies the already-selected unique cameras for realized B.
    Requested C must be on the qualified ladder; realized C is immediately
    underfilled to the available camera count. B>C is partitioned into
    C-sized render microbatches and accumulated before one optimizer step.
    """

    selected_cameras = tuple(cameras)
    if not selected_cameras:
        raise ValueError("adaptive optimization transaction requires cameras")
    if modality_eligibilities is not None and len(modality_eligibilities) != len(selected_cameras):
        raise ValueError("modality_eligibilities must align with cameras")
    requested_b = (
        len(selected_cameras)
        if requested_batch_size is None
        else validate_supervision_batch_size(requested_batch_size)
    )
    if requested_b < len(selected_cameras):
        raise ValueError("requested B cannot be smaller than realized camera work")
    requested_c = validate_supervision_camera_count(requested_camera_count)
    identity_keys = tuple(
        (
            "source",
            int(source_index),
        )
        if (source_index := camera_source_index(camera)) is not None
        else ("object", id(camera))
        for camera in selected_cameras
    )
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("adaptive optimization transaction requires unique camera identities")

    iteration = step_kwargs.get("iteration")
    if iteration is not None and hasattr(runtime, "publish_assimilation_support"):
        with _phase_timed("m_publish_assimilation_support"):
            runtime.publish_assimilation_support(
                int(iteration),
                optimizer_updates=step_kwargs.get("optimizer_update_id", 0),
                topology_version=step_kwargs.get("topology_version", 0),
            )
    if hasattr(runtime, "prefetch_selected_cameras"):
        with _phase_timed("m_prefetch_selected_cameras"):
            runtime.prefetch_selected_cameras(list(selected_cameras))

    requests = []
    context = None
    with _phase_timed("m_build_step"):
        for index, camera in enumerate(selected_cameras):
            request_kwargs = dict(step_kwargs, camera=camera, _shared_context=context)
            if modality_eligibilities is not None:
                request_kwargs["modality_eligibility"] = modality_eligibilities[index]
            request, request_context = runtime.build_step(**request_kwargs)
            if context is None:
                context = request_context
            requests.append(request)
    assert context is not None
    if any(
        bool(getattr(request.view.modality_eligibility, "depth_insertion", False))
        for request in requests
    ):
        # The true-camera executor receives only this batch context, while the
        # source eligible for reuse need not be the first selected camera.
        if (versions := runtime._evidence_versions()) is not None:
            context.evidence_transaction_id = context.optimizer_update_id
            context.coordinate_frame_version = versions.coordinate_frame
            context.schema_version = versions.schema
            context.surface_evidence_version = versions.surface_evidence
            context.policy_version = versions.policy
    post_render_hook = context.post_render_hook

    def _bind_visible_mask(render_outputs: Any) -> None:
        context.gradient_policy_context.active_set.visible_mask = (
            render_outputs.payload["visibility_filter"].detach()
            .to(dtype=torch.bool)
            .contiguous()
        )
        if post_render_hook is not None:
            post_render_hook(render_outputs)

    context.post_render_hook = _bind_visible_mask
    realized_c = min(requested_c, len(requests))
    request_groups = tuple(
        tuple(requests[index : index + realized_c])
        for index in range(0, len(requests), realized_c)
    )
    if realized_c > 1:
        from eligsir.optimization.gaussian.true_batch_executor import (
            run_true_camera_batched_step,
        )

        result = run_true_camera_batched_step(
            engine,
            requests,
            context,
            camera_microbatch_size=realized_c,
        )
        fallback_reason = None
    elif len(requests) == 1:
        result = engine.run_step(requests[0], context)
        _attach_b1_supervision_evidence(result, requests[0], context)
        fallback_reason = "realized_C_equals_1"
    else:
        result = engine.run_batched_step(requests, context, skipped_view_count=0)
        # C1 keeps the legacy sequential accumulation engine for B>1.  Its
        # primary render is already in the result; attach only that source's
        # same-render lease before the single commit boundary.
        _attach_b1_supervision_evidence(result, requests[0], context)
        fallback_reason = "requested_C_shrunk_to_1"

    _set_result_metadata(
        result,
        true_camera_batch=realized_c > 1,
        true_camera_batch_fallback_reason=fallback_reason,
        requested_camera_count=requested_c,
        realized_camera_count=realized_c,
        microbatch_count=len(request_groups),
        microbatch_sizes=tuple(len(group) for group in request_groups),
        requested_batch_size=requested_b,
        realized_batch_size=len(requests),
    )
    with _phase_timed("m_commit_step"):
        runtime.commit_step(context, result)
    runtime.random_state_token = getattr(result, "random_state_token", None)
    with _phase_timed("m_record_digest"):
        _record_optimization_digest(
            requests[0],
            result,
            selected_cameras,
            0,
            requested_batch_size=requested_b,
        )
    return result


def _attach_b1_supervision_evidence(
    result: Any,
    request: OptimizationStepRequest,
    context: OptimizationRuntimeContext,
) -> None:
    """Attach the same-render B1/C1 observation bridge without a probe render.

    B1 and shortened-C1 transactions use the legacy engine's general step
    implementation, so they have no true-batch reducer to construct
    ``SupervisionEvidenceBatch``.  This adapter consumes only the primary
    already-returned render payload and exact pre-step result versions.
    Missing ownership values fail closed and leave the original result
    evidence untouched.
    """

    # Source-depth reuse is an explicit evidence consumer even with all-view E0 off.
    if not (
        bool(request.evidence_requirements.want_supervision_evidence)
        or bool(getattr(request.view.modality_eligibility, "depth_insertion", False))
    ):
        return
    if getattr(result, "optimizer_step_count", None) == 0:
        result.evidence = None
        return
    from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken,
        SupervisionEvidenceBatch,
        SupervisionModality,
    )

    transaction_id = context.evidence_transaction_id
    values = (
        context.coordinate_frame_version,
        context.schema_version,
        context.optimizer_update_id,
        context.surface_evidence_version,
        context.policy_version,
        getattr(result, "parameter_version_before", None),
    )
    if not isinstance(transaction_id, (int, str)) or any(
        type(value) is not int or value < 0 for value in values
    ):
        return
    source_identity = request.view.source_identity
    if not isinstance(source_identity, SourceIdentity):
        return
    try:
        token = EvidenceSnapshotToken(
            map_content=int(request.map_version),
            topology=int(request.topology_version),
            coordinate_frame=values[0],
            schema=values[1],
            optimizer_update=values[2],
            surface_evidence=values[3],
            policy=values[4],
            parameter=values[5],
        )
    except (TypeError, ValueError):
        return
    eligibility = request.view.modality_eligibility or ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=bool(request.view.render_depth),
        depth_insertion=False,
        geometry_population_target=bool(request.view.render_depth),
    )
    modality = resolve_supervision_modality(request.view)
    payload = getattr(getattr(result, "render_outputs", None), "payload", {})
    alpha = payload.get("alpha") if isinstance(payload, dict) else None
    rendered_depth = payload.get("rendered_depth") if isinstance(payload, dict) else None
    leases: tuple[GeometryRenderLease, ...] = ()
    # A shortened B4 source-service transaction may request one native
    # depth-insertion lease while deliberately disabling all-view population
    # geometry evidence.  Keep that lease narrow: completeness consumers still
    # require one lease per realized view, while insertion binds only its exact
    # fresh source below.
    if (
        (modality.geometry_evidence_enabled or bool(eligibility.depth_insertion))
        and isinstance(alpha, torch.Tensor)
        and isinstance(rendered_depth, torch.Tensor)
    ):
        leases = (
            GeometryRenderLease(
                source_identity=source_identity,
                snapshot_token=token,
                resolution_tier=request.resolution_bucket,
                modality_eligibility=eligibility,
                alpha=alpha,
                rendered_depth=rendered_depth,
            ),
        )
    previous = getattr(result, "evidence", None)
    breakdown = getattr(result, "loss_breakdown", None)
    components = getattr(breakdown, "components", {})
    rgb_loss = components.get("Ll1") if isinstance(components, dict) else None
    depth_loss = components.get("depth") if isinstance(components, dict) else None
    reference = (
        rgb_loss
        if torch.is_tensor(rgb_loss)
        else depth_loss if torch.is_tensor(depth_loss) else None
    )
    rgb_available = (
        modality.rgb_evidence_enabled
        and torch.is_tensor(rgb_loss)
    )
    depth_available = (
        modality.depth_loss_enabled
        and modality.depth_evidence_enabled
        and torch.is_tensor(depth_loss)
    )
    image = payload.get("render") if isinstance(payload, dict) else None
    rgb_pixels = (
        int(image.shape[-2]) * int(image.shape[-1])
        if isinstance(image, torch.Tensor) and rgb_available
        else 0
    )
    depth_pixels = (
        int(getattr(breakdown, "depth_valid_sample_count", 0) or 0)
        if depth_available
        else 0
    )
    result.evidence = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id=transaction_id,
        modalities=(modality,),
        source_identities=(source_identity,),
        modality_eligibility=(eligibility,),
        rgb_residual_summary=(
            rgb_loss.reshape(1)
            if rgb_available
            else torch.zeros_like(reference).reshape(1)
            if reference is not None
            else None
        ),
        # ``depth`` is loss assembly's accepted unweighted masked depth-data
        # term.  It excludes depth weighting and every non-data regularizer.
        depth_residual_summary=(
            depth_loss.reshape(1)
            if depth_available
            else torch.zeros_like(reference).reshape(1)
            if reference is not None
            else None
        ),
        rgb_valid_pixels=(
            torch.full_like(reference, rgb_pixels, dtype=torch.int64).reshape(1)
            if reference is not None
            else None
        ),
        depth_valid_pixels=(
            torch.full_like(reference, depth_pixels, dtype=torch.int64).reshape(1)
            if reference is not None
            else None
        ),
        optimization_payload=getattr(previous, "payload", None),
        geometry_render_leases=leases,
    )


def supports_streaming_optimization_transaction(
    pipe: Any,
    cameras: Sequence[Any],
    *,
    requested_camera_count: int,
) -> tuple[bool, str]:
    """Fail-closed capability preflight for every realized true-C group."""

    requested_c = validate_supervision_camera_count(requested_camera_count)
    cameras = tuple(cameras)
    if not cameras:
        return False, "requires_at_least_one_materialized_view"
    realized_c = min(requested_c, len(cameras))
    from eligsir.mapping.renderer.true_camera_batch import (
        supports_true_camera_batch,
    )

    for index in range(0, len(cameras), realized_c):
        group = cameras[index : index + realized_c]
        if len(group) == 1:
            continue
        supported, reason = supports_true_camera_batch(pipe, group)
        if not supported:
            return False, f"microbatch_{index // realized_c}:{reason}"
    return True, "qualified_camera_microbatches"


def _set_result_metadata(result: Any, **values: Any) -> None:
    """Attach optional telemetry without narrowing the engine result contract."""
    for name, value in values.items():
        try:
            setattr(result, name, value)
        except (AttributeError, TypeError):
            pass


def merge_deferred_prune_masks(stale_mask: Any, conflict_mask: Any) -> Any:
    """Combine compatible stale/conflict masks and fail visibly on mismatch."""

    if stale_mask is None:
        return conflict_mask
    if stale_mask.shape[0] == conflict_mask.shape[0]:
        return stale_mask | conflict_mask
    print(
        "[streaming-depth-lock] WARNING: conflict prune mask shape mismatch "
        f"({stale_mask.shape[0]} vs {conflict_mask.shape[0]}), skipping conflicts",
        flush=True,
    )
    return stale_mask
