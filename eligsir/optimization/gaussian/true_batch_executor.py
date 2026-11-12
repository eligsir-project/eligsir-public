"""True C>1 fixed-topology optimization for issue #106.

One gsplat rasterization call supervises C homogeneous cameras. The executor
preserves the existing batched-engine contracts: requested-B loss scaling,
per-view full objectives, one backward/optimizer update, union visibility,
primary-only streaming bookkeeping, deterministic secondary non-finite skips,
and explicit work accounting.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Sequence

import torch

from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity

from eligsir.mapping.integration.depth_helpers import _get_sensor_depth
from eligsir.mapping.integration.loss_diagnostics import (
    _nan_term_report,
    batched_status_packet,
)
from eligsir.mapping.renderer.true_camera_batch import (
    TrueCameraBatchResult,
    render_true_camera_batch,
    supports_true_camera_batch,
)
from eligsir.optimization.gaussian import objective as _objective
from eligsir.optimization.gaussian import losses as _losses
from eligsir.optimization.gaussian.batched_pixel_terms import (
    PixelTermsView,
    assemble_batched_pixel_terms,
)
from eligsir.optimization.gaussian.depth_loss_weighting import DEPTH_LOSS_QUANTITIES
from eligsir.optimization.gaussian.contracts import (
    GradientPolicyContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    RenderOutputs,
    finalize_optimization_work_accounting,
    resolve_supervision_modality,
    validate_optimizer_clock_contract,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.engine import (
    _evwb_gradient_transaction,
    _commit_evwb_states,
    record_completed_supervision,
)
from eligsir.optimization.gaussian.objective import LossAssemblyInputs
from eligsir.optimization.gaussian.render_port import invoke_render
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    RenderRequirement,
    SupervisionEvidenceBatch,
    SupervisionModality,
    group_render_requirement,
)


def _per_camera_visibility(meta: dict, *, camera_count: int, population: int, device) -> torch.Tensor:
    out = torch.zeros((camera_count, population), dtype=torch.bool, device=device)
    camera_ids = meta.get("camera_ids")
    gaussian_ids = meta.get("gaussian_ids")
    if camera_ids is not None and gaussian_ids is not None:
        if camera_ids.numel() and gaussian_ids.numel():
            out[camera_ids.long(), gaussian_ids.long()] = True
        return out
    radii = meta.get("radii")
    if radii is not None and getattr(radii, "ndim", 0) >= 2 and radii.shape[0] == camera_count:
        dense = radii.amax(dim=-1) if radii.ndim > 2 else radii
        if dense.shape[1] == population:
            return dense > 0
    raise RuntimeError(
        "TRUE_CAMERA_BATCH requires gsplat camera_ids+gaussian_ids or dense [C,N] radii metadata"
    )


def _group_meta_by_camera(meta: dict, *, camera_count: int) -> list[dict]:
    """Partition packed per-hit metadata into one dict per camera.

    This performs one grouping/offset pass over the packed tensors (a
    stable sort by ``camera_ids`` plus a single gather, per issue #1580)
    instead of the previous ``_slice_meta``, which built a fresh
    ``camera_ids == camera_index`` boolean mask and boolean-indexed every
    packed metadata tensor once *per camera* -- ``camera_count`` repeated
    O(P) scans/compactions of the same packed data. Output is identical to
    the old per-camera boolean-mask slice: same per-camera element order
    (the sort is stable), same untouched non-packed keys, and the same
    ``camera_ids`` zeroed-to-local convention.
    """
    camera_ids = meta.get("camera_ids")
    if camera_ids is None:
        return [dict(meta) for _ in range(camera_count)]
    camera_ids_long = camera_ids.long()
    packed_n = int(camera_ids_long.shape[0])
    packed_keys = tuple(
        key
        for key, value in meta.items()
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == packed_n
    )
    order = torch.argsort(camera_ids_long, stable=True)
    counts = torch.bincount(camera_ids_long, minlength=camera_count)
    # One small (camera_count-length) device->host sync for the split
    # sizes, replacing the `camera_count * len(packed_keys)` boolean-index
    # compactions the old per-camera loop paid for (each of which forces
    # its own equivalent host sync to size its output).
    sizes = counts.tolist()
    splits = {
        key: torch.split(meta[key].index_select(0, order), sizes) for key in packed_keys
    }
    outs: list[dict] = []
    for camera_index in range(camera_count):
        out = dict(meta)
        for key in packed_keys:
            out[key] = splits[key][camera_index]
        if torch.is_tensor(out.get("camera_ids")):
            out["camera_ids"] = torch.zeros_like(out["camera_ids"])
        outs.append(out)
    return outs


def _payload_for_view(
    batch: TrueCameraBatchResult,
    *,
    camera_index: int,
    visibility: torch.Tensor,
    meta_i: dict,
) -> RenderOutputs:
    rgb = batch.renders[camera_index]
    alpha_chw = batch.alphas[camera_index]
    alpha_hwc = alpha_chw.permute(1, 2, 0).contiguous()

    # No dummy `viewspace_points[N,3]` and no dense per-view `radii[N]`
    # reconstruction here: the consumer audit for issue #1580 traced every
    # reader of these two keys on the prepared true-camera-batch path and
    # found none -- gsplat_backend.py documents `viewspace_points` as dummy
    # API-parity state disconnected from rasterization even on the
    # single-camera path, and the one theoretical `radii`-consuming hook
    # (`EvidenceBuilder.build_gaussian_evidence(projected_radius=...)`) is
    # never actually called with it. `visibility_filter` and `meta` (the
    # packed per-camera metadata gsplat's own MCMC strategy reads via
    # `render_pkg["meta"]`) remain: both have real objective/topology
    # readers and must not be removed.
    payload: dict[str, Any] = {
        "render": rgb,
        "visibility_filter": visibility,
        "is_used": visibility,
        "alpha": alpha_hwc,
        "meta": meta_i,
        "sparse_grad": bool(batch.sparse_grad),
        "true_camera_batch": True,
        "true_camera_batch_camera_index": int(camera_index),
        "true_camera_batch_camera_count": int(batch.camera_count),
    }
    if batch.rendered_depth is not None:
        payload["rendered_depth"] = batch.rendered_depth[camera_index]
    if batch.rendered_depth_ed is not None:
        payload["rendered_depth_ed"] = batch.rendered_depth_ed[camera_index]
    return RenderOutputs(payload=payload)


def _aggregate_loss(
    breakdowns: Sequence[LossBreakdown],
    scale: float,
    *,
    detach: bool = False,
) -> LossBreakdown:
    if not breakdowns:
        return LossBreakdown(total=0.0, total_tensor=None, components={}, loss_finite=False)
    losses = [
        breakdown.total.detach() if detach else breakdown.total_tensor
        for breakdown in breakdowns
    ]
    total_tensor = losses[0] * 0.0
    for loss in losses:
        total_tensor = total_tensor + loss
    total_tensor = total_tensor * float(scale)
    components: dict[str, Any] = {}
    for key in sorted(set().union(*(b.components.keys() for b in breakdowns))):
        values = [b.components[key] for b in breakdowns if key in b.components]
        if key in ("Ll1", "ssim") and len(values) != len(breakdowns):
            continue  # Missing RGB measurements are not zero-padded means.
        if values:
            acc = values[0] * 0.0
            for value in values:
                acc = acc + value
            components[key] = acc * float(scale)
    return LossBreakdown(
        total=total_tensor.detach(),
        total_tensor=total_tensor,
        components=components,
        loss_finite=all(bool(b.loss_finite) for b in breakdowns),
        depth_valid_sample_count=sum(
            b.depth_valid_sample_count for b in breakdowns
        ),
    )


def _combine_group_aggregates(group_aggregates: Sequence[LossBreakdown]) -> LossBreakdown:
    """Combine already globally scaled group totals and components."""
    totals = [g.total_tensor.detach() for g in group_aggregates]
    total_tensor = totals[0] * 0.0
    for loss in totals:
        total_tensor = total_tensor + loss
    components: dict[str, Any] = {}
    for key in sorted(set().union(*(g.components.keys() for g in group_aggregates))):
        values = [g.components[key] for g in group_aggregates if key in g.components]
        if key in ("Ll1", "ssim") and len(values) != len(group_aggregates):
            continue
        if values:
            acc = values[0] * 0.0
            for value in values:
                acc = acc + value
            components[key] = acc
    return LossBreakdown(
        total=total_tensor.detach(),
        total_tensor=total_tensor,
        components=components,
        loss_finite=all(bool(g.loss_finite) for g in group_aggregates),
        depth_valid_sample_count=sum(g.depth_valid_sample_count for g in group_aggregates),
    )


def _base_result(*, primary, breakdown, render_outputs, step_allowed: bool, skip_reason: str | None):
    return OptimizationStepResult(
        loss_breakdown=breakdown,
        render_outputs=render_outputs,
        evidence=None,
        optimization_gpu_ms=None,
        map_version_before=primary.map_version,
        map_version_after=primary.map_version,
        topology_version=primary.topology_version,
        step_allowed=step_allowed,
        skip_reason=skip_reason,
    )


def _prepare_evidence_snapshot(
    *, primary: OptimizationStepRequest, runtime: Any, session: Any, optimizer_update: int
) -> tuple[EvidenceSnapshotToken, int | str] | None:
    """Capture the immutable transaction snapshot before the first render."""

    transaction_id = getattr(runtime, "evidence_transaction_id", None)
    values = (
        getattr(runtime, "coordinate_frame_version", None)
        if getattr(runtime, "coordinate_frame_version", None) is not None
        else getattr(session, "coordinate_frame_version", None),
        getattr(runtime, "schema_version", None)
        if getattr(runtime, "schema_version", None) is not None
        else getattr(session, "schema_version", None),
        getattr(session, "parameter_version", None),
        getattr(runtime, "surface_evidence_version", None),
        getattr(runtime, "policy_version", None),
    )
    if not isinstance(transaction_id, (int, str)) or any(value is None for value in values):
        return None
    coordinate, schema, parameter, surface, policy = values
    try:
        return (
            EvidenceSnapshotToken(
                map_content=primary.map_version, topology=primary.topology_version,
                coordinate_frame=coordinate, schema=schema, parameter=parameter,
                optimizer_update=optimizer_update, surface_evidence=surface, policy=policy,
            ),
            transaction_id,
        )
    except (TypeError, ValueError):
        return None


def _eligibility(request: OptimizationStepRequest, modality: SupervisionModality):
    eligibility = request.view.modality_eligibility
    if eligibility is not None:
        return eligibility
    return ModalityEligibility(
        rgb_supervision=modality.rgb_loss_enabled,
        depth_supervision=modality.depth_loss_enabled,
        geometry_population_target=modality.geometry_evidence_enabled,
    )


def _resolve_batched_pixel_terms(
    *,
    args: Any,
    group: Sequence[OptimizationStepRequest],
    group_modalities: Sequence[SupervisionModality],
    batch: TrueCameraBatchResult,
    iteration: int,
    tb_writer: Any,
    scalar_log_interval: int,
    depth_loss_type: str,
    opt: Any,
    rgb_measurements: tuple[bool, bool],
    streaming_scene: Any = None,
) -> tuple[PixelTermsView, ...] | None:
    """Return batched terms only for homogeneous, already-resident inputs."""

    if not bool(args.streaming_objective_batched_pixel_terms):
        return None
    if len(group) <= 1:
        return None
    if not all(bool(modality.rgb_loss_enabled) for modality in group_modalities):
        return None
    if float(getattr(args, "streaming_free_space_loss_weight", 0.0)) > 0.0:
        return None
    if bool(getattr(args, "streaming_appearance_only_enabled", False)):
        return None
    if int(getattr(args, "supervision_tile_probe_grid", 0) or 0) > 0:
        return None
    if tb_writer is not None and iteration % max(1, int(scalar_log_interval)) == 0:
        return None

    renders = batch.renders
    cameras = tuple(req.view.camera for req in group)
    raw_gts = tuple(camera.original_image for camera in cameras)
    if not all(
        isinstance(gt, torch.Tensor)
        and gt.device == renders.device
        and gt.dtype == renders.dtype
        and gt.shape == renders.shape[1:]
        for gt in raw_gts
    ):
        return None
    evwb = _objective._evwb_config(args)
    try:
        gts = torch.stack(
            list(raw_gts) if evwb is not None else [_objective._gt_image_validated(cam, gt) for cam, gt in zip(cameras, raw_gts, strict=True)]
        )
    except RuntimeError:
        return None
    if gts.device != renders.device or gts.dtype != renders.dtype or gts.shape != renders.shape:
        return None

    rendered_depth = None
    sensor_depth = None
    alpha = None
    weight_maps: list[torch.Tensor] = []
    depth_quantity = "d"
    alpha_gate = 0.0
    depth_eligible = all(
        bool(modality.depth_loss_enabled) for modality in group_modalities
    ) and not bool(getattr(args, "surface_confidence_depth_loss", False))
    if depth_eligible:
        depth_quantity = str(getattr(args, "streaming_depth_loss_quantity", "d")).lower()
        alpha_gate = float(getattr(args, "streaming_depth_loss_alpha_gate", 0.0))
        valid_config = (
            depth_quantity in DEPTH_LOSS_QUANTITIES
            and math.isfinite(alpha_gate)
            and 0.0 <= alpha_gate < 1.0
        )
        depth_source = (
            batch.rendered_depth_ed
            if depth_quantity in ("ed", "ed_alpha_weighted")
            else batch.rendered_depth
        )
        if valid_config and depth_source is not None:
            h, w = int(gts.shape[-2]), int(gts.shape[-1])
            try:
                sensor_list = [_get_sensor_depth(cam, h, w) for cam in cameras]
            except Exception:
                sensor_list = None
            if sensor_list is not None and all(
                t is not None and t.device == renders.device for t in sensor_list
            ):
                try:
                    sensor_stack = torch.stack(sensor_list)
                except RuntimeError:
                    sensor_stack = None
                if sensor_stack is not None and sensor_stack.shape == depth_source.shape:
                    rendered_depth = depth_source
                    sensor_depth = sensor_stack
                    # ed_alpha_weighted (#1833) needs alpha for its per-pixel
                    # weighting even when alpha_gate == 0.0 (they're
                    # orthogonal knobs -- see batched_pixel_terms.py).
                    if alpha_gate > 0.0 or depth_quantity == "ed_alpha_weighted":
                        alpha = batch.alphas
                    disc_list = [
                        getattr(cam, "_resolution_depth_confidence_weight", None)
                        for cam in cameras
                    ]
                    if all(t is not None for t in disc_list):
                        if all(t.device == renders.device for t in disc_list):
                            stacked = [t.unsqueeze(0) if t.dim() == 2 else t for t in disc_list]
                            weight_maps.append(torch.stack(stacked))
                        else:
                            rendered_depth = None
                            sensor_depth = None
                    elif any(t is not None for t in disc_list):
                        # Non-uniform across the group (some views carry a
                        # per-camera resolution-confidence weight, others
                        # don't) -- cannot stack; skip depth batching for
                        # this group entirely, RGB stays batched.
                        rendered_depth = None
                        sensor_depth = None

    states = priors = None
    if evwb is not None:
        states = tuple(
            _objective._evwb_state_for_camera(streaming_scene, cam, renders[index], evwb)
            for index, cam in enumerate(cameras)
        )
        corrected = []
        prior_values = []
        for index, state in enumerate(states):
            if state is None:
                corrected.append(renders[index])
                prior_values.append(None)
            else:
                image, prior = _objective._display_referred_evwb(
                    renders[index], gts[index], state, display_margin=evwb[0], prior_weight=evwb[1]
                )
                corrected.append(image)
                prior_values.append(prior)
        renders = torch.stack(corrected)
        priors = tuple(prior_values)
    terms = assemble_batched_pixel_terms(
        renders=renders,
        gts=gts,
        rendered_depth=rendered_depth,
        sensor_depth=sensor_depth,
        alpha=alpha,
        alpha_gate=alpha_gate,
        depth_quantity=depth_quantity,
        depth_loss_type=depth_loss_type,
        weight_maps=tuple(weight_maps) if weight_maps else None,
        required_rgb=_objective.required_rgb_terms(opt, True, rgb_measurements),
    )
    if evwb is None:
        return terms
    assert states is not None and priors is not None
    return tuple(
        PixelTermsView(
            Ll1=term.Ll1, ssim=term.ssim, depth=term.depth,
            depth_valid_count=term.depth_valid_count,
            photometric_evwb_state=states[index], photometric_evwb_prior=priors[index],
        )
        for index, term in enumerate(terms)
    )


def run_true_camera_batched_step(
    engine: Any,
    requests: Sequence[OptimizationStepRequest],
    runtime: Any,
    *,
    skipped_view_count: int = 0,
    camera_microbatch_size: int | None = None,
) -> OptimizationStepResult:
    requests = tuple(requests)
    if len(requests) < 2:
        raise ValueError("true camera batching requires at least two requests")
    validate_optimizer_clock_contract(runtime)
    skipped_view_count = int(skipped_view_count or 0)
    if skipped_view_count < 0:
        raise ValueError("skipped_view_count must be >= 0")
    primary = requests[0]
    modalities = tuple(resolve_supervision_modality(request.view) for request in requests)
    eligibility = tuple(
        _eligibility(request, modality) for request, modality in zip(requests, modalities, strict=True)
    )
    modalities = tuple(
        SupervisionModality(
            rgb_loss_enabled=modality.rgb_loss_enabled,
            depth_loss_enabled=modality.depth_loss_enabled,
            rgb_evidence_enabled=modality.rgb_evidence_enabled,
            depth_evidence_enabled=(
                modality.depth_evidence_enabled and item.depth_supervision
            ),
            geometry_evidence_enabled=item.geometry_population_target,
        )
        for modality, item in zip(modalities, eligibility, strict=True)
    )
    evidence_requested = tuple(
        bool(request.evidence_requirements.want_supervision_evidence)
        for request in requests
    )
    supervision_evidence_requested = all(evidence_requested)
    if any(evidence_requested) and not supervision_evidence_requested:
        raise ValueError("true camera batch requires homogeneous supervision evidence requests")
    geometry_render_reuse_requested = any(
        item.depth_insertion or item.geometry_population_target
        for item in eligibility
    )
    for extra in requests[1:]:
        if (
            extra.iteration != primary.iteration
            or extra.map_version != primary.map_version
            or extra.topology_version != primary.topology_version
            or extra.resolution_bucket != primary.resolution_bucket
        ):
            raise ValueError("true camera batch requires one iteration/map/topology/resolution bucket")
        if bool(extra.view.update_sh_rest) != bool(primary.view.update_sh_rest):
            raise ValueError("true camera batch requires homogeneous update_sh_rest semantics")

    if camera_microbatch_size is None:
        camera_microbatch_size = len(requests)
    if isinstance(camera_microbatch_size, bool) or int(camera_microbatch_size) < 1:
        raise ValueError("camera_microbatch_size must be a positive integer")
    camera_microbatch_size = min(int(camera_microbatch_size), len(requests))
    request_groups = tuple(
        requests[index : index + camera_microbatch_size]
        for index in range(0, len(requests), camera_microbatch_size)
    )
    for group in request_groups:
        if len(group) == 1:
            continue
        ok, reason = supports_true_camera_batch(
            runtime.pipe, tuple(req.view.camera for req in group)
        )
        if not ok:
            raise RuntimeError(f"TRUE_CAMERA_BATCH unavailable: {reason}")

    session = engine.session
    gaussians = runtime.gaussians or getattr(session, "model", None)
    if gaussians is None:
        raise ValueError("true camera batch requires the engine session model")
    if getattr(session, "model", gaussians) is not gaussians:
        raise ValueError("runtime.gaussians must match engine session.model")

    context = runtime.gradient_policy_context
    if context is None:
        context = GradientPolicyContext(
            gaussians=gaussians,
            args=runtime.args,
            iteration=primary.iteration,
            n_frames_ingested=runtime.n_frames_ingested,
            optimizer_update_id=runtime.optimizer_update_id,
        )
    elif context.gaussians is not gaussians:
        raise ValueError("GradientPolicyContext.gaussians must match session.model")
    elif context.optimizer_update_id != runtime.optimizer_update_id:
        raise ValueError(
            "GradientPolicyContext.optimizer_update_id must match "
            "OptimizationRuntimeContext.optimizer_update_id"
        )

    timed = runtime.timing_scope or (lambda _name: nullcontext())
    requested_b = len(requests) + skipped_view_count
    batch_scale = 1.0 / float(requested_b)
    parameter_before = int(getattr(session, "parameter_version", 0) or 0)
    optimizer_update_for_evidence = (
        primary.iteration if runtime.optimizer_update_id is None else runtime.optimizer_update_id
    )
    evidence_snapshot = (
        _prepare_evidence_snapshot(
            primary=primary,
            runtime=runtime,
            session=session,
            optimizer_update=optimizer_update_for_evidence,
        )
        if supervision_evidence_requested or geometry_render_reuse_requested
        else None
    )
    legacy_rgb_metrics = runtime.optimizer_update_id is None
    rgb_measurements = (
        legacy_rgb_metrics or (evidence_snapshot is not None and supervision_evidence_requested),
        legacy_rgb_metrics,
    )
    opt_steps_before = int(getattr(engine, "_optimizer_step_calls", 0))
    result = None
    xyz_lr = None
    supervised_pixels = 0
    depth_valid_samples = 0
    skipped = skipped_view_count
    successful_indices: list[int] = []
    breakdowns: list[LossBreakdown] = []
    outputs: list[RenderOutputs] = []
    rasterizer_calls = 0
    true_camera_rasterizer_calls = 0
    renderer_camera_batch_sizes: list[int] = []
    loss_objective_call_count = 0
    loss_views_per_call: list[int] = []
    backward_call_count = 0
    evwb_enabled = bool(getattr(runtime.args, "streaming_train_evwb_enabled", False))
    evwb_states: dict[int, torch.Tensor] | None = {} if evwb_enabled else None
    e0_rgb_losses: list[torch.Tensor] = []
    e0_depth_losses: list[torch.Tensor] = []
    e0_rgb_valid_pixels: list[torch.Tensor] = []
    e0_depth_valid_pixels: list[torch.Tensor] = []
    geometry_render_leases: list[Any] = []
    # E0 is produced from the normal B4 render/backward transaction.  These
    # are short-lived, full-row device tensors; they are not persistent
    # [B, N] state and are never materialized on the host.
    e0_view_incidence: torch.Tensor | None = None
    e0_meaningful_opportunity: torch.Tensor | None = None
    e0_render_contribution: torch.Tensor | None = None
    e0_gradient_utility: torch.Tensor | None = None

    with primary.random_state_token.continuous() as rng_scope, _evwb_gradient_transaction(evwb_states):
        optimizer_update_id = runtime.optimizer_update_id
        behavior_update_id = (
            primary.iteration
            if optimizer_update_id is None
            else optimizer_update_id
        )
        canonical_update_denied = runtime.optimizer_update_permitted is False
        xyz_lr = (
            session.update_learning_rate(primary.iteration)
            if optimizer_update_id is None
            else None
        )
        population = int(gaussians.get_xyz.shape[0])
        visible_union = torch.zeros(
            population,
            dtype=torch.bool,
            device=gaussians.get_xyz.device,
        )
        primary_failed = False
        global_index = 0
        group_aggregates: list[LossBreakdown] = []
        group_offset = 0
        # Issue #1670 Phase A (`--streaming_objective_step_regularizers`):
        # once-per-step per-Gaussian regularizers instead of once per view.
        # Eligibility (else fall back to assemble_loss's own per-view loop,
        # unchanged): the flag is on, sparse active-set is in effect, and
        # the differentiable-energy-loss path is not (that path replaces
        # the reg call sites entirely -- see objective.py's
        # `_energy_loss_enabled_main`). `len(group) > 1` is checked per
        # group below (a B=1 group always falls back).
        _step_regs_flag = bool(runtime.args.streaming_objective_step_regularizers)
        _step_regs_eligible = (
            _step_regs_flag
            and bool(runtime.sparse_active_set)
            and not bool(getattr(runtime.args, "energy_loss_enabled", False))
        )
        for group in request_groups:
            group_modalities = modalities[group_offset : group_offset + len(group)]
            group_render_depth = (
                group_render_requirement(group_modalities) is RenderRequirement.RGB_DEPTH
            )
            reg_step = None
            pixel_terms: tuple[PixelTermsView, ...] | None = None
            if len(group) > 1:
                with timed("m_render"):
                    batch = render_true_camera_batch(
                        tuple(req.view.camera for req in group),
                        gaussians,
                        runtime.pipe,
                        group[0].view.background,
                        update_sh_rest=group[0].view.update_sh_rest,
                        render_depth=group_render_depth,
                    )
                renderer_camera_batch_sizes.append(len(group))
                visibility_by_camera = _per_camera_visibility(
                    batch.meta,
                    camera_count=len(group),
                    population=population,
                    device=gaussians.get_xyz.device,
                )
                if _step_regs_eligible:
                    with timed("m_loss_regs_gaussian"):
                        masks = visibility_by_camera | gaussians.provisional.detach()[None]
                        n_v = masks.sum(dim=1).clamp_min(1).to(torch.float32)
                        weights = (masks.to(torch.float32) / n_v[:, None]).sum(dim=0)
                        # `batch_size` MUST be the call's true global divisor
                        # (`requested_b`), not `len(group)`: under the
                        # adaptive B/C microbatch path or a non-zero
                        # `skipped_view_count`, `len(group)` can be strictly
                        # less than `requested_b`, and
                        # `assemble_step_regularizers` divides by exactly
                        # what it is given (see its docstring's RETRACTED
                        # note -- reviewer reproduced a 2.0000x over-weight
                        # from this exact mismatch).
                        reg_step = _losses.assemble_step_regularizers(
                            gaussians,
                            runtime.args,
                            behavior_update_id,
                            weights,
                            batch_size=requested_b,
                        )
                # The packed gsplat metadata has already been reduced into
                # per-camera visibility for the optimizer.  Reuse that exact
                # device result for B4 supervision evidence rather than
                # issuing an attribution render or reconstructing it on CPU.
                # Per-view RGB eligibility deliberately gates measurement
                # opportunity: raster visibility by itself is not evidence.
                if (
                    evidence_snapshot is not None
                    and supervision_evidence_requested
                    and len(request_groups) == 1
                    and len(group) == 4
                    and len(requests) == 4
                ):
                    incidence = visibility_by_camera.transpose(0, 1)
                    meaningful_views = torch.tensor(
                        [
                            modality.rgb_evidence_enabled
                            for modality in group_modalities
                        ],
                        dtype=incidence.dtype,
                        device=incidence.device,
                    )
                    opportunity = incidence.to(dtype=torch.float32).matmul(
                        meaningful_views.to(dtype=torch.float32)
                    )
                    e0_view_incidence = incidence
                    e0_meaningful_opportunity = opportunity
                    e0_render_contribution = opportunity / float(len(group))
                grouped_meta = _group_meta_by_camera(batch.meta, camera_count=len(group))
                group_outputs = tuple(
                    _payload_for_view(
                        batch,
                        camera_index=index,
                        visibility=visibility_by_camera[index],
                        meta_i=grouped_meta[index],
                    )
                    for index in range(len(group))
                )
                pixel_terms = _resolve_batched_pixel_terms(
                    args=runtime.args,
                    group=group,
                    group_modalities=group_modalities,
                    opt=runtime.opt,
                    rgb_measurements=rgb_measurements,
                    batch=batch,
                    iteration=primary.iteration,
                    tb_writer=runtime.tb_writer,
                    scalar_log_interval=int(
                        runtime.loss_assembly_kwargs.get("scalar_log_interval", 10)
                    ),
                    # Same dict `LossAssemblyInputs` is built from below via
                    # `**runtime.loss_assembly_kwargs` -- NOT `args` directly
                    # (its default disagrees with `LossAssemblyInputs.
                    # depth_loss_type`'s default; see
                    # `_resolve_batched_pixel_terms`'s docstring).
                    depth_loss_type=str(
                        runtime.loss_assembly_kwargs.get("depth_loss_type", "l1")
                    ),
                    streaming_scene=runtime.loss_assembly_kwargs.get("streaming_scene"),
                )
                true_camera_rasterizer_calls += 1
            else:
                req = group[0]
                with timed("m_render"):
                    out = invoke_render(
                        gaussians,
                        req.view.camera,
                        runtime.pipe,
                        req.view.background,
                        update_sh_rest=req.view.update_sh_rest,
                    render_depth=group_render_depth,
                    )
                renderer_camera_batch_sizes.append(1)
                group_outputs = (out,)
            rasterizer_calls += 1
            group_breakdowns: list[LossBreakdown] = []
            group_surviving: list[tuple[OptimizationStepRequest, RenderOutputs]] = []
            pending_losses = []
            for view_index, (req, modality, out) in enumerate(
                zip(group, group_modalities, group_outputs, strict=True)
            ):
                outputs.append(out)
                if (
                    group_offset == 0 and view_index == 0
                    and not canonical_update_denied
                    and runtime.post_render_hook is not None
                ):
                    with timed("m_post_render"):
                        runtime.post_render_hook(out)
                    primary_group_visible = (
                        visibility_by_camera.any(dim=0)
                        if len(group) > 1
                        else out.payload.get("visibility_filter")
                    )
                    context.active_set.visible_mask = (
                        primary_group_visible.detach().to(dtype=torch.bool).contiguous()
                        if primary_group_visible is not None
                        else torch.zeros_like(visible_union)
                    )
                image = out.payload["render"]
                inputs = LossAssemblyInputs(
                    gaussians=gaussians,
                    viewpoint_cam=req.view.camera,
                    args=runtime.args,
                    opt=runtime.opt,
                    pipe=runtime.pipe,
                    render_outputs=out,
                    image=image,
                    iteration=primary.iteration,
                    n_frames_ingested=runtime.n_frames_ingested,
                    optimizer_update_id=optimizer_update_id,
                    sparse_active_set=runtime.sparse_active_set,
                    rgb_loss_enabled=modality.rgb_loss_enabled,
                    rgb_measurements=rgb_measurements,
                    use_depth_loss=modality.depth_loss_enabled,
                    use_energy_mcmc=runtime.use_energy_mcmc,
                    background=req.view.background,
                    skip_gaussian_regularizers=reg_step is not None,
                    precomputed_pixel_terms=(pixel_terms[view_index] if pixel_terms is not None else None),
                    **runtime.loss_assembly_kwargs,
                )
                with timed("m_loss_assembly"):
                    pending_losses.append(_objective.prepare_loss(inputs))
                loss_objective_call_count += 1
                loss_views_per_call.append(1)
            # One detached status packet for this render group, including the
            # optional step regularizer. Rejected terms never enter autograd.
            with timed("m_loss_numerical_gate"):
                packet = batched_status_packet([(acc, loss) for acc, loss, _ in pending_losses])
                if reg_step is not None:
                    packet = torch.cat([packet, torch.isfinite(reg_step.detach()).to(torch.int64).reshape(1)])
                status = iter(packet.tolist())
            for (req, modality, out), (acc, loss, finalize) in zip(
                zip(group, group_modalities, group_outputs, strict=True), pending_losses, strict=True
            ):
                with timed("m_loss_assembly"):
                    breakdown = finalize(*acc.resolve(loss, primary.iteration, status=status))
                if not breakdown.loss_finite:
                    # The once-per-group regularizer already contains every
                    # view's mask.  A secondary loss cannot be skipped after
                    # that reduction without changing its weight, so reject
                    # this exceptional transaction rather than stepping a
                    # mathematically different objective.
                    if global_index == 0 or reg_step is not None:
                        result = _base_result(
                            primary=primary,
                            breakdown=breakdown,
                            render_outputs=out,
                            step_allowed=False,
                            skip_reason="non_finite_loss",
                        )
                        primary_failed = True
                        break
                    if evidence_snapshot is not None and supervision_evidence_requested:
                        rejected_reference = out.payload["render"].new_zeros(())
                        e0_rgb_losses.append(rejected_reference)
                        e0_depth_losses.append(rejected_reference)
                        e0_rgb_valid_pixels.append(
                            torch.zeros_like(rejected_reference, dtype=torch.int64)
                        )
                        e0_depth_valid_pixels.append(
                            torch.zeros_like(rejected_reference, dtype=torch.int64)
                        )
                    skipped += 1
                    global_index += 1
                    continue
                breakdowns.append(breakdown)
                rgb_loss = breakdown.components.get("Ll1")
                if (
                    evidence_snapshot is not None
                    and supervision_evidence_requested
                ):
                    # ``Ll1`` and ``depth`` are the ordinary accepted data
                    # terms already detached by loss assembly.  Keep their
                    # scalar summaries and exact support device-resident;
                    # no render, loss, or host readback is introduced here.
                    depth_loss = breakdown.components.get("depth")
                    reference = (
                        rgb_loss
                        if torch.is_tensor(rgb_loss)
                        else depth_loss if torch.is_tensor(depth_loss) else None
                    )
                    if reference is not None:
                        rgb_available = (
                            modality.rgb_evidence_enabled
                            and torch.is_tensor(rgb_loss)
                        )
                        depth_available = (
                            modality.depth_loss_enabled
                            and modality.depth_evidence_enabled
                            and torch.is_tensor(depth_loss)
                        )
                        e0_rgb_losses.append(
                            rgb_loss if rgb_available else torch.zeros_like(reference)
                        )
                        e0_depth_losses.append(
                            depth_loss if depth_available else torch.zeros_like(reference)
                        )
                        pixels = int(image.shape[-2]) * int(image.shape[-1])
                        e0_rgb_valid_pixels.append(
                            torch.full_like(
                                reference, pixels if rgb_available else 0, dtype=torch.int64
                            )
                        )
                        e0_depth_valid_pixels.append(
                            torch.full_like(
                                reference,
                                breakdown.depth_valid_sample_count if depth_available else 0,
                                dtype=torch.int64,
                            )
                        )
                # An all-view, short-lived geometry lane.  This must not use
                # the primary-only legacy post-render hook or union
                # visibility: every surviving camera retains its own
                # already-rendered alpha/depth tensors without a second call.
                # The shared lease lane independently serves insertion and B4 E0 evidence.
                if (
                    evidence_snapshot is not None
                    and geometry_render_reuse_requested
                    and (
                        eligibility[global_index].depth_insertion
                        or eligibility[global_index].geometry_population_target
                    )
                    and isinstance(req.view.source_identity, SourceIdentity)
                ):
                    alpha = out.payload.get("alpha")
                    rendered_depth = out.payload.get("rendered_depth")
                    if isinstance(alpha, torch.Tensor) and isinstance(
                        rendered_depth, torch.Tensor
                    ):
                        from eligsir.mapping.integration.depth_render_reuse import (
                            GeometryRenderLease,
                        )

                        snapshot_token, _transaction_id = evidence_snapshot
                        geometry_render_leases.append(
                            GeometryRenderLease(
                                source_identity=req.view.source_identity,
                                snapshot_token=snapshot_token,
                                resolution_tier=req.resolution_bucket,
                                modality_eligibility=eligibility[global_index],
                                alpha=alpha,
                                rendered_depth=rendered_depth,
                            )
                        )
                group_breakdowns.append(breakdown)
                group_surviving.append((req, out))
                completed_visible = out.payload.get("visibility_filter")
                if completed_visible is not None:
                    visible_union |= completed_visible.detach().to(dtype=torch.bool)
                image = out.payload["render"]
                supervised_pixels += int(image.shape[-2]) * int(image.shape[-1])
                if modality.depth_loss_enabled:
                    depth_valid_samples += breakdown.depth_valid_sample_count
                if req.view.source_index is not None:
                    successful_indices.append(int(req.view.source_index))
                global_index += 1
            pending_losses.clear()
            del acc, loss, finalize, inputs, packet
            if primary_failed:
                break
            group_offset += len(group)
            if (
                not runtime.placement_only
                and not canonical_update_denied
                and group_breakdowns
            ):
                group_aggregate = _aggregate_loss(group_breakdowns, batch_scale)
                if reg_step is not None:
                    reg_step_finite = bool(next(status))
                    if reg_step_finite:
                        group_aggregate.total_tensor = group_aggregate.total_tensor + reg_step
                        group_aggregate.total = group_aggregate.total_tensor.detach()
                        group_aggregate.components["active_reg+young"] = reg_step.detach()
                    else:
                        _nan_term_report("active_reg+young", primary.iteration)
                group_aggregates.append(group_aggregate)
                if runtime.mcmc_strategy is not None:
                    for req, out in group_surviving:
                        runtime.mcmc_strategy.step_pre_backward(
                            gaussians=gaussians,
                            args=runtime.args,
                            iteration=behavior_update_id,
                            render_pkg=out.payload,
                            loss=group_aggregate.total_tensor,
                        )
                if evwb_states is not None:
                    for breakdown in group_breakdowns:
                        state = breakdown.photometric_evwb_state
                        if state is not None:
                            evwb_states[id(state)] = state
                with timed("m_backward"):
                    group_aggregate.total_tensor.backward()
                backward_call_count += 1

        if not primary_failed:
            if len(group_aggregates) == 1:
                aggregate = group_aggregates[0]
            elif group_aggregates:
                # More than one group in this call (microbatching, or
                # multiple B4 groups) -- recombine from the already-scaled
                # group aggregates, not the flat per-view `breakdowns` list,
                # so a step-regularizer term added to an individual group
                # (see `reg_step` above) is not silently dropped from the
                # final reported total/components. See
                # `_combine_group_aggregates`'s docstring.
                aggregate = _combine_group_aggregates(group_aggregates)
            else:
                aggregate = _aggregate_loss(breakdowns, batch_scale, detach=True)
            if runtime.placement_only:
                aggregate = _aggregate_loss(breakdowns, batch_scale)
                result = _base_result(
                    primary=primary,
                    breakdown=aggregate,
                    render_outputs=outputs[0],
                    step_allowed=False,
                    skip_reason="placement_only",
                )
            else:
                if runtime.optimizer_update_permitted is None:
                    iteration_limit = (
                        runtime.optimizer_iteration_limit
                        if runtime.optimizer_iteration_limit is not None
                        else getattr(runtime.opt, "iterations", float("inf"))
                    )
                    optimizer_update_denied = primary.iteration >= iteration_limit
                    endpoint_reason = "iteration_limit"
                else:
                    optimizer_update_denied = not runtime.optimizer_update_permitted
                    endpoint_reason = "work_plan_endpoint"
                step_mask = runtime.step_mask
                if step_mask is None:
                    step_mask = visible_union | gaussians.provisional.detach()
                context.active_set.visible_mask = visible_union
                if optimizer_update_denied:
                    if runtime.optimizer_update_permitted is None:
                        engine.run_gradient_policies(context)
                    result = _base_result(
                        primary=primary,
                        breakdown=aggregate,
                        render_outputs=outputs[0],
                        step_allowed=False,
                        skip_reason=endpoint_reason,
                    )
                    result.backward_ran = (
                        runtime.optimizer_update_permitted is None
                    )
                else:
                    # Snapshot the already-computed position gradient before
                    # the optimizer clears it.  This is a device reduction
                    # over the normal backward result, not an additional loss
                    # or render; absent/incompatible gradients remain unknown.
                    if (
                        evidence_snapshot is not None
                        and supervision_evidence_requested
                        and e0_view_incidence is not None
                        and len(request_groups) == 1
                        and len(requests) == 4
                    ):
                        xyz = getattr(gaussians, "get_xyz", None)
                        gradient = getattr(xyz, "grad", None)
                        if (
                            isinstance(gradient, torch.Tensor)
                            and gradient.ndim == 2
                            and gradient.shape[0] == population
                            and gradient.device == gaussians.get_xyz.device
                        ):
                            e0_gradient_utility = torch.linalg.vector_norm(
                                gradient.detach(), dim=1
                            )
                    if optimizer_update_id is not None:
                        xyz_lr = session.update_learning_rate(optimizer_update_id)
                    result = engine.run_post_backward_step(
                        gaussians,
                        runtime.args,
                        runtime.pipe,
                        context,
                        optimizer_type=runtime.optimizer_type,
                        sparse_active_set=runtime.sparse_active_set,
                        step_mask=step_mask,
                        visible=visible_union,
                        use_energy_mcmc=runtime.use_energy_mcmc,
                        loss_breakdown=aggregate,
                        render_outputs=outputs[0],
                        map_version=primary.map_version,
                        topology_version=primary.topology_version,
                        timing_scope=runtime.timing_scope,
                    )
                    result.backward_ran = bool(result.step_allowed)
                    result.optimizer_step_ran = bool(result.step_allowed)
                    if result.step_allowed and evwb_states:
                        _commit_evwb_states(evwb_states, runtime.args)

                if result.step_allowed and runtime.post_optimizer_hook is not None:
                    runtime.post_optimizer_hook(result)
                if result.step_allowed and runtime.fixed_shape_mcmc:
                    if runtime.mcmc_strategy is None:
                        raise ValueError("fixed_shape_mcmc requires an MCMC strategy")
                    with timed("m_noise"):
                        runtime.mcmc_strategy.inject_noise(
                            gaussians=gaussians,
                            args=runtime.args,
                            xyz_lr=xyz_lr,
                            visible=(visible_union if runtime.mcmc_local and runtime.sparse_active_set else None),
                            sparse_active_set=runtime.mcmc_local and runtime.sparse_active_set,
                            iteration=behavior_update_id,
                            tb_writer=runtime.tb_writer,
                        )
                    result.mcmc_noise_ran = True
                if result.step_allowed and runtime.post_mcmc_hook is not None:
                    runtime.post_mcmc_hook(result)

    if result is None:
        raise RuntimeError("true camera batch produced no optimization result")
    result.random_state_token = rng_scope.returned
    result.xyz_lr = xyz_lr
    result.parameter_version_before = parameter_before
    result.parameter_version_after = int(
        getattr(session, "parameter_version", parameter_before) or parameter_before
    )
    result.batch_size = requested_b
    result.supervision_view_count = len(breakdowns) if not runtime.placement_only else 0
    result.skipped_view_count = skipped
    result.batch_loss_scale = batch_scale
    result.rasterizer_call_count = rasterizer_calls
    result.supervised_pixel_count = supervised_pixels
    result.renderer_camera_batch_sizes = tuple(renderer_camera_batch_sizes)
    result.loss_objective_call_count = loss_objective_call_count
    result.loss_views_per_call = tuple(loss_views_per_call)
    result.backward_call_count = backward_call_count
    result.optimizer_step_count = int(getattr(engine, "_optimizer_step_calls", 0)) - opt_steps_before
    if result.optimizer_step_count > 0:
        record_completed_supervision(
            gaussians,
            visible_union,
            runtime=runtime,
        )
    result.depth_valid_sample_count = (
        depth_valid_samples if result.optimizer_step_count > 0 else 0
    )
    result.successful_supervision_source_indices = tuple(successful_indices)
    if evidence_snapshot is not None and result.optimizer_step_count > 0:
        snapshot_token, transaction_id = evidence_snapshot
        candidate_source_identities = tuple(
            request.view.source_identity for request in requests
        )
        # Source identity is a new M6 provenance inventory, not a precondition
        # for ordinary E0 RGB/depth evidence.  Preserve compatibility for an
        # older caller that supplies ``None`` or duplicate identities; the M6
        # target provider requires this inventory and therefore ABSTAINS.
        source_identities = (
            candidate_source_identities
            if (
                all(
                    isinstance(source, SourceIdentity)
                    for source in candidate_source_identities
                )
                and len(set(candidate_source_identities)) == len(requests)
            )
            else ()
        )
        result.evidence = SupervisionEvidenceBatch(
            snapshot_token=snapshot_token,
            transaction_id=transaction_id,
            modalities=modalities,
            source_identities=source_identities,
            modality_eligibility=eligibility,
            rgb_residual_summary=(
                torch.stack(e0_rgb_losses)
                if len(e0_rgb_losses) == len(requests)
                else None
            ),
            # ``depth`` is the accepted unweighted masked depth-data term
            # from loss assembly; regularizers and its objective weight are
            # intentionally excluded from this per-view residual summary.
            depth_residual_summary=(
                torch.stack(e0_depth_losses)
                if len(e0_depth_losses) == len(requests)
                else None
            ),
            rgb_valid_pixels=(
                torch.stack(e0_rgb_valid_pixels)
                if len(e0_rgb_valid_pixels) == len(requests)
                else None
            ),
            depth_valid_pixels=(
                torch.stack(e0_depth_valid_pixels)
                if len(e0_depth_valid_pixels) == len(requests)
                else None
            ),
            view_incidence=e0_view_incidence,
            meaningful_measurement_opportunity=e0_meaningful_opportunity,
            render_contribution=e0_render_contribution,
            gradient_utility_evidence=e0_gradient_utility,
            geometry_render_leases=tuple(geometry_render_leases),
        )
    else:
        result.evidence = None
    setattr(result, "true_camera_batch", True)
    setattr(result, "camera_batch_rasterizer_calls", true_camera_rasterizer_calls)
    setattr(result, "microbatch_count", len(request_groups))
    setattr(result, "microbatch_sizes", tuple(len(group) for group in request_groups))
    return finalize_optimization_work_accounting(result)


__all__ = ["run_true_camera_batched_step"]
