"""CPU-Torch contract tests for the gsplat atomic population-plan fence."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scene.gsplat_model import GsplatGaussianModel, OptimizerDictProxy
from eligsir.cli.parser import parse_training_args
from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.map.arbiter import GaussianMutationArbiter, PopulationMutationProposal, ProposalKey
from eligsir.map.mutation import RelocationPlan
from eligsir.map.population_mutation_plan import PopulationMutationPlan
from eligsir.map.population_target_payload import PopulationTargetPayloadBatch, PopulationTargetPayloadSchema
from eligsir.map.row_state import PopulationRowCreation, PopulationRowRelocation, PopulationRowTransform
from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease
from eligsir.mapping.gaussian_belief_accumulator import (
    GaussianBeliefAccumulationStatus,
    accumulate_true_b4_belief,
    ensure_population_policy_e0_buffers,
)
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime
from eligsir.mapping.population_policy import MutationWorkToken, PopulationAction
from eligsir.mapping.population_shadow_runtime import PopulationShadowVersions
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.optimization.gaussian.contracts import OptimizationRuntimeContext
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.runtime.production_application import (
    CanonicalMappingRuntime,
    MappingBackendCompatibility,
)
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.utils.sh_utils import RGB2SH


requires_gsplat_device = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "normal live-runtime B4 bridge constructs GsplatGaussianModel, whose "
        "production schema is device-resident; run inside the ROCm/CUDA "
        "training container with GPU passthrough"
    ),
)


def _token(**overrides) -> EvidenceSnapshotToken:
    values = dict(
        map_content=0,
        topology=0,
        coordinate_frame=0,
        schema=0,
        optimizer_update=0,
        surface_evidence=0,
        policy=0,
        parameter=0,
    )
    values.update(overrides)
    return EvidenceSnapshotToken(**values)


def _cpu_model(rows: int = 3, *, cls=GsplatGaussianModel) -> GsplatGaussianModel:
    """Construct the real gsplat descriptor schema without CUDA-only init."""

    model = object.__new__(cls)
    model.max_sh_degree = 0
    params = {
        "means": nn.Parameter(torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)),
        "scales": nn.Parameter(torch.full((rows, 3), 2.0)),
        "quats": nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1)),
        "opacities": nn.Parameter(torch.arange(rows, dtype=torch.float32)),
        "sh0": nn.Parameter(torch.zeros((rows, 1, 3))),
        "shN": nn.Parameter(torch.empty((rows, 0, 3))),
    }
    model.params = nn.ParameterDict(params)
    model.optimizers = {}
    for name, param in model.params.items():
        optimizer = torch.optim.Adam([param], lr=0.01)
        optimizer.state[param] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.full_like(param, float(10 + len(model.optimizers))),
            "exp_avg_sq": torch.full_like(param, float(20 + len(model.optimizers))),
        }
        model.optimizers[name] = optimizer
    model.optimizer = OptimizerDictProxy(model.optimizers)
    model._next_gaussian_uid = 10 + rows
    model._seed_uid_range = (10, 10 + rows)
    for name in GsplatGaussianModel._ROW_STATE_REQUIRED_BUFFER_NAMES:
        metadata = GsplatGaussianModel._ROW_STATE_BUFFER_DESCRIPTOR_TABLE[name]
        family = metadata[0]
        if name in {"anchor_xyz", "birth_anchor_xyz"}:
            value = model.params["means"].detach().clone()
        elif name == "anchor_normal":
            value = torch.tensor([[0.0, 0.0, 1.0]]).repeat(rows, 1)
        elif name == "anchor_scale_log":
            value = model.params["scales"].detach().clone()
        elif name == "anchor_opacity_logit":
            value = model.params["opacities"].detach().clone()
        elif name == "gaussian_uid":
            value = torch.arange(10, 10 + rows, dtype=torch.int64)
        elif name == "root_uid":
            value = torch.arange(10, 10 + rows, dtype=torch.int64)
        elif name == "parent_uid":
            value = torch.full((rows,), -1, dtype=torch.int64)
        elif name == "p_exist":
            value = torch.full((rows,), 0.25, dtype=torch.float16)
        elif name in {"provisional"}:
            value = torch.ones(rows, dtype=torch.bool)
        elif name in {"max_radii2D"}:
            value = torch.arange(rows, dtype=torch.float32)
        elif name in {"xyz_gradient_accum", "denom", "visibility_ema"}:
            value = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
        else:
            descriptor_dtype = {
                "render": torch.float32,
                "surface": torch.float32,
                "belief": torch.float32,
                "provenance": torch.int32,
                "compatibility": torch.int32,
            }[family]
            if name in {"anchor_confidence", "utility_ema", "baseline_diversity", "support_evidence", "contradiction_evidence", "surface_position_uncertainty", "surface_normal_uncertainty", "rgb_residual_ema", "depth_residual_ema", "redundancy", "expected_marginal_gain", "change_probability", "confidence_decay"}:
                descriptor_dtype = torch.float32
            value = torch.arange(rows, dtype=descriptor_dtype)
        setattr(model, name, value)
    return model


def _backend(model: GsplatGaussianModel) -> GsplatMapBackend:
    backend = GsplatMapBackend(model)
    backend.install_population_identity_snapshot_from_bootstrap()
    return backend


def _normal_live_runtime() -> CanonicalMappingRuntime:
    args, _ = parse_training_args(["-s", "/dev/null", "-m", "output/test"])
    args.streaming_frame_admission = "preselected"
    args.gpu_frame_prep = True
    args.gpu_frame_resident = True
    args.streaming_population_live_enabled = True
    args.streaming_population_shadow_enabled = True
    args.streaming_lifecycle_enabled = False
    args.streaming_post_mapping_mature_mobility = False
    args.streaming_depth_lock_centers = False
    args.streaming_insert_from_depth = False
    config = mapping_runtime_config_from_args(args)
    return CanonicalMappingRuntime(
        config,
        MappingBackendCompatibility(
            dataset=SimpleNamespace(sh_degree=3),
            opt=SimpleNamespace(),
            pipe=SimpleNamespace(),
        ),
    )


def _payload() -> PopulationTargetPayloadBatch:
    return PopulationTargetPayloadBatch(
        schema=PopulationTargetPayloadSchema(),
        positions=torch.tensor([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]]),
        colors=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        raw_scales=torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        opacity_logits=torch.tensor([[5.0], [6.0]]),
        sh_rest=torch.empty((2, 0, 3)),
        opaque_values=(),
        max_targets=2,
    )


def _replace_plan(model: GsplatGaussianModel, *, retire_only: bool = False) -> tuple[PopulationMutationPlan, PopulationTargetPayloadBatch]:
    registry = model.build_row_state_registry()
    before = _token()
    after = _token(map_content=1, topology=1, optimizer_update=1, parameter=1)
    if retire_only:
        transform = PopulationRowTransform(
            old_population=3,
            new_population=2,
            survivor_old_indices=torch.tensor([1, 2]),
            retired_old_indices=torch.tensor([0]),
            created_rows=(),
            relocated_rows=(),
            covered_fields=registry.names,
        )
        payload = PopulationTargetPayloadBatch(
            schema=PopulationTargetPayloadSchema(),
            positions=torch.empty((0, 3)), colors=torch.empty((0, 3)), raw_scales=torch.empty((0, 3)),
            quaternions=torch.empty((0, 4)), opacity_logits=torch.empty((0, 1)), sh_rest=torch.empty((0, 0, 3)),
            opaque_values=(), max_targets=0,
        )
        return PopulationMutationPlan(
            snapshot_token=before, next_snapshot_token=after, row_transform=transform,
            survivor_hypothesis_uids=("11", "12"),
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=before,
            capacity=3, old_population=3, new_population=2,
            retire_rows=torch.tensor([0]), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.empty(0, dtype=torch.int64),
            target_payload=torch.empty((0, 3)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(), created_hypothesis_uids=(),
            reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=1,
            row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(registry),
        ), payload
    payload = _payload()
    transform = PopulationRowTransform(
        old_population=3,
        new_population=3,
        survivor_old_indices=torch.tensor([1]),
        retired_old_indices=torch.tensor([0, 2]),
        created_rows=(PopulationRowCreation(row_index=2, hypothesis_uid="101"),),
        relocated_rows=(PopulationRowRelocation(row_index=1, donor_row_index=0, old_hypothesis_uid="10", new_hypothesis_uid="100"),),
        covered_fields=registry.names,
    )
    return PopulationMutationPlan(
        snapshot_token=before, next_snapshot_token=after, row_transform=transform,
        survivor_hypothesis_uids=("11",),
        mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=before,
        capacity=3, old_population=3, new_population=3,
        retire_rows=torch.tensor([2]), relocate_donor_rows=torch.tensor([0]), relocate_target_rows=torch.tensor([1]), create_rows=torch.tensor([2]),
        target_payload=payload.positions.clone(), donor_hypothesis_uids=("10",),
        relocated_hypothesis_uids=("100",), created_hypothesis_uids=("101",),
        reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=3,
        row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(registry),
    ), payload


def _single_payload() -> PopulationTargetPayloadBatch:
    payload = _payload()
    return PopulationTargetPayloadBatch(
        schema=payload.schema,
        positions=payload.positions[:1],
        colors=payload.colors[:1],
        raw_scales=payload.raw_scales[:1],
        quaternions=payload.quaternions[:1],
        opacity_logits=payload.opacity_logits[:1],
        sh_rest=payload.sh_rest[:1],
        opaque_values=(),
        max_targets=1,
    )


def _single_action_plan(
    model: GsplatGaussianModel, action: str,
) -> tuple[PopulationMutationPlan, PopulationTargetPayloadBatch]:
    if action == "retire":
        return _replace_plan(model, retire_only=True)

    registry = model.build_row_state_registry()
    before = _token()
    after = _token(map_content=1, topology=1, optimizer_update=1, parameter=1)
    payload = _single_payload()
    if action == "create":
        transform = PopulationRowTransform(
            old_population=3,
            new_population=4,
            survivor_old_indices=torch.tensor([0, 1, 2]),
            retired_old_indices=torch.empty(0, dtype=torch.int64),
            created_rows=(PopulationRowCreation(row_index=3, hypothesis_uid="100"),),
            relocated_rows=(),
            covered_fields=registry.names,
        )
        return PopulationMutationPlan(
            snapshot_token=before, next_snapshot_token=after, row_transform=transform,
            survivor_hypothesis_uids=("10", "11", "12"),
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=before,
            capacity=4, old_population=3, new_population=4,
            retire_rows=torch.empty(0, dtype=torch.int64),
            relocate_donor_rows=torch.empty(0, dtype=torch.int64),
            relocate_target_rows=torch.empty(0, dtype=torch.int64),
            create_rows=torch.tensor([3]), target_payload=payload.positions.clone(),
            donor_hypothesis_uids=(), relocated_hypothesis_uids=(), created_hypothesis_uids=("100",),
            reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=1,
            row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(registry),
        ), payload
    if action == "relocate":
        transform = PopulationRowTransform(
            old_population=3,
            new_population=3,
            survivor_old_indices=torch.tensor([1, 2]),
            retired_old_indices=torch.tensor([0]),
            created_rows=(),
            relocated_rows=(
                PopulationRowRelocation(
                    row_index=2, donor_row_index=0, old_hypothesis_uid="10", new_hypothesis_uid="100",
                ),
            ),
            covered_fields=registry.names,
        )
        return PopulationMutationPlan(
            snapshot_token=before, next_snapshot_token=after, row_transform=transform,
            survivor_hypothesis_uids=("11", "12"),
            mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=before,
            capacity=3, old_population=3, new_population=3,
            retire_rows=torch.empty(0, dtype=torch.int64),
            relocate_donor_rows=torch.tensor([0]), relocate_target_rows=torch.tensor([2]),
            create_rows=torch.empty(0, dtype=torch.int64), target_payload=payload.positions.clone(),
            donor_hypothesis_uids=("10",), relocated_hypothesis_uids=("100",), created_hypothesis_uids=(),
            reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=1,
            row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(registry),
        ), payload
    raise ValueError(f"unsupported action {action!r}")


def _proposal(plan: PopulationMutationPlan, payload: PopulationTargetPayloadBatch) -> PopulationMutationProposal:
    return PopulationMutationProposal(
        key=ProposalKey(1, 1, 1, "population-policy"), plan=plan, target_payload=payload,
        row_state_registry_snapshot=plan.row_state_registry_snapshot,
    )


def _buffer(value: torch.Tensor) -> BufferRef:
    return BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape), dtype=str(value.dtype).removeprefix("torch."),
            memory_domain=MemoryDomain.HOST, encoding="test", device="cpu",
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )


def _b4_source_lease(source: SourceIdentity) -> PreparedSourceLease:
    eligibility = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=False,
        geometry_population_target=True,
    )
    c2w = torch.eye(4)
    c2w[0, 3] = 0.2 * source.source_ordinal
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=eligibility,
        source_metadata={"native_height": 3, "native_width": 9},
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "depth_meters": _buffer(torch.full((3, 9), 2.0)),
            "rgb": _buffer(torch.zeros(3, 9, 3)),
            "c2w": _buffer(c2w),
            "camera_intrinsics": _buffer(
                torch.tensor(
                    [[10.0, 0.0, 4.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]
                )
            ),
        },
        device="cpu",
    )


def _b4_evidence(
    sources: tuple[SourceIdentity, ...], token: EvidenceSnapshotToken, *, measured: bool = True,
) -> SupervisionEvidenceBatch:
    eligibility = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=False,
        geometry_population_target=True,
    )
    leases = []
    for index, source in enumerate(sources):
        alpha = torch.ones(3, 9, 1)
        alpha[1, 4 if index == 0 else 8] = 0.0
        leases.append(
            GeometryRenderLease(
                source_identity=source,
                snapshot_token=token,
                resolution_tier="native",
                modality_eligibility=eligibility,
                alpha=alpha,
                rendered_depth=torch.full((3, 9, 1), 2.0),
            )
        )
    leases = tuple(leases)
    return SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="b4-live-context",
        source_identities=sources,
        modalities=tuple(
            SupervisionModality(
                depth_loss_enabled=True,
                depth_evidence_enabled=True,
                geometry_evidence_enabled=True,
            )
            for _ in sources
        ),
        modality_eligibility=tuple(eligibility for _ in sources),
        view_incidence=(
            torch.tensor(
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, True, True],
                ]
            )
            if measured
            else torch.zeros((3, 4), dtype=torch.bool)
        ),
        meaningful_measurement_opportunity=(
            torch.tensor([1.0, 1.0, 2.0]) if measured else torch.zeros(3)
        ),
        render_contribution=(
            torch.tensor([0.25, 0.25, 0.5]) if measured else torch.zeros(3)
        ),
        gradient_utility_evidence=(
            torch.tensor([0.1, 0.2, 1.0]) if measured else torch.zeros(3)
        ),
        geometry_render_leases=leases,
    )


def _normal_live_b4_activation(*, favorable_evidence: bool):
    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.support_evidence.zero_()
    # This is a positive control for native E0 creation, not a retirement
    # precedence test.
    model.p_exist.fill_(1)
    model.visibility_ema.zero_()
    model.utility_ema.zero_()
    model.surface_position_uncertainty.zero_()
    model.depth_residual_ema.zero_()
    model.contradiction_evidence.zero_()
    model.expected_marginal_gain.zero_()
    backend = GsplatMapBackend(model)
    backend.install_population_identity_snapshot_from_bootstrap()
    runtime = _normal_live_runtime()
    runtime.gaussians = model
    runtime.backend = backend
    runtime._engine = runtime._services.engine
    runtime._engine.session._parameter_version = 1
    runtime._runtime_policy = replace(runtime._runtime_policy, cap_max=4)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    runtime.scene = SimpleNamespace(
        add_to_occupancy_hash=lambda _xyz: None,
        hot_camera_ids=lambda: (),
        close_decode_cache=lambda: None,
    )
    runtime._mcmc_cadence_owner = MCMCCadenceOwner(
        grow_interval_pixels=400,
        relocate_interval_pixels=400,
        min_optimizer_updates_after_structural_mutation=1,
    )
    sources = tuple(
        SourceIdentity("replay", 5, index, 1_700_000_000 + index)
        for index in range(4)
    )
    plan = SimpleNamespace(
        plan_id="b4-live-context-plan",
        snapshot_id=23,
        control_version=29,
        supervision_sources=sources,
        replay_sources=(),
        mutation_budget=SimpleNamespace(max_total=1),
    )
    runtime._population_shadow_active_plan = plan
    runtime._optimization_runtime = StreamingOptimizationRuntime(
        session=runtime._engine.session,
        gaussians=model,
        args=runtime._optimization_policy,
        opt=runtime.opt,
        pipe=runtime.pipe,
        mcmc_strategy=None,
        tb_writer=None,
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={},
        iter_since_new_frame=0,
        **runtime._population_shadow_runtime_kwargs(),
    )
    current = _token(optimizer_update=1)
    evidence = _b4_evidence(sources, current, measured=favorable_evidence)
    result = SimpleNamespace(
        evidence=evidence,
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        supervised_pixel_count=80,
        depth_valid_sample_count=80,
        realized_batch_size=4,
        realized_camera_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    context = OptimizationRuntimeContext(
        optimizer_update_id=1,
        optimizer_update_permitted=True,
        gradient_policy_context=SimpleNamespace(
            stable_geometry=SimpleNamespace(
                successful_updates_since_admission=1,
                iter_since_new_frame=0,
            )
        ),
    )
    runtime._optimization_runtime.commit_step(context, result)
    runtime._record_completed_optimization_result(result, mapping_active_ns=0)
    return runtime, model, backend, sources, plan, result


def test_true_b4_accumulator_rejects_stale_snapshot_without_any_row_write():
    model = _cpu_model()
    model.exposure_count = torch.zeros(3, dtype=torch.int32)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    sources = tuple(SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4))
    evidence = _b4_evidence(sources, _token(topology=1))
    result = SimpleNamespace(
        evidence=evidence,
        map_version_before=0,
        map_version_after=0,
        topology_version=1,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    before = {
        name: getattr(model, name).clone()
        for name in (
            "exposure_count", "last_exposed_update", "visibility_ema",
            "utility_ema", "expected_marginal_gain",
        )
    }
    support_before = model.support_evidence.clone()

    status = accumulate_true_b4_belief(
        model,
        result,
        current_versions=PopulationShadowVersions(
            map_content=0,
            topology=1,
            coordinate_frame=1,
            schema=0,
            optimizer_update=0,
            surface_evidence=0,
            policy=0,
            parameter=1,
        ),
    )

    assert status is GaussianBeliefAccumulationStatus.STALE_SNAPSHOT
    for name, value in before.items():
        assert torch.equal(getattr(model, name), value), name
    assert torch.equal(model.support_evidence, support_before)


def test_completed_true_b4_post_step_content_version_updates_live_policy():
    """A valid native B4 survives its expected fixed-topology content update."""
    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.p_exist.fill_(1)
    model.anchor_confidence.fill_(1)
    model.surface_position_uncertainty.zero_()
    model.depth_residual_ema.zero_()
    model.contradiction_evidence.zero_()
    model.support_evidence.zero_()
    model.expected_marginal_gain.zero_()
    model.utility_ema.zero_()
    sources = tuple(SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4))
    evidence = _b4_evidence(sources, _token(optimizer_update=1))
    result = SimpleNamespace(
        evidence=evidence,
        map_version_before=0,
        # Gsplat advances visible map content for the optimizer step while
        # preserving topology; this is the production transition from the
        # FR3 trace, not an out-of-band structural mutation.
        map_version_after=1,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        supervised_pixel_count=80,
        depth_valid_sample_count=0,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=1, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=model, parameter_version=1), gaussians=model, args=SimpleNamespace(),
        opt=SimpleNamespace(), pipe=SimpleNamespace(), mcmc_strategy=None, tb_writer=None,
        optimizer_type="adam", sparse_active_set=False, use_energy_mcmc=False,
        fixed_shape_mcmc=False, mcmc_local=False, loss_assembly_kwargs={},
        iter_since_new_frame=0, population_shadow_enabled=True,
        population_shadow_work_token_provider=lambda _result, _context: MutationWorkToken(1, 1),
        population_shadow_versions_provider=lambda _result, _context: versions,
        population_shadow_belief_provider=CanonicalMappingRuntime._population_shadow_belief_from_rows,
        population_shadow_capacity=4,
        population_shadow_defer_until_post_step_commit=True,
    )
    context = OptimizationRuntimeContext(
        optimizer_update_id=1,
        optimizer_update_permitted=True,
        gradient_policy_context=SimpleNamespace(
            stable_geometry=SimpleNamespace(
                successful_updates_since_admission=1, iter_since_new_frame=0,
            )
        ),
    )
    runtime.commit_step(context, result)
    production = object.__new__(CanonicalMappingRuntime)
    production.gaussians = model
    production.config = SimpleNamespace(
        static_options={"population_live_enabled": True, "population_shadow_enabled": True}
    )
    production._population_shadow_evidence_versions = lambda: versions

    drop, targets = production._observe_population_policy_inputs_after_step(
        runtime, result, {}
    )
    observation = runtime.population_shadow_runtime.take_latest_activation_observation()

    assert drop is None
    assert targets is None
    assert observation is not None
    assert torch.any(observation.decision.actions == PopulationAction.CREATE.code)
    assert torch.equal(model.population_policy_exposure_count, torch.tensor([1, 1, 2]))
    assert torch.equal(
        model.population_policy_support_count,
        torch.tensor([1, 1, 2], dtype=model.population_policy_support_count.dtype),
    )
    assert torch.count_nonzero(model.support_evidence) == 0


@pytest.mark.parametrize(
    ("name", "result_updates", "current_map", "expected"),
    (
        (
            "stale_current_content",
            {},
            2,
            GaussianBeliefAccumulationStatus.STALE_SNAPSHOT,
        ),
        (
            "incomplete_batch",
            {"supervision_view_count": 3},
            1,
            GaussianBeliefAccumulationStatus.NOT_COMPLETED_TRUE_B4,
        ),
        (
            "non_b4_batch",
            {"batch_size": 1},
            1,
            GaussianBeliefAccumulationStatus.NOT_COMPLETED_TRUE_B4,
        ),
    ),
)
def test_true_b4_accumulator_rejects_invalid_live_policy_epochs(
    name, result_updates, current_map, expected,
):
    """Only the exact completed B4 post-step version transition is admitted."""

    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    sources = tuple(
        SourceIdentity("replay", 5, index, 1_700_000_000 + index)
        for index in range(4)
    )
    result = SimpleNamespace(
        evidence=_b4_evidence(sources, _token(optimizer_update=1)),
        map_version_before=0,
        map_version_after=1,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    for field, value in result_updates.items():
        setattr(result, field, value)

    status = accumulate_true_b4_belief(
        model,
        result,
        current_versions=PopulationShadowVersions(
            map_content=current_map,
            topology=0,
            coordinate_frame=0,
            schema=0,
            optimizer_update=1,
            surface_evidence=0,
            policy=0,
            parameter=1,
        ),
    )

    assert status is expected, name
    assert torch.count_nonzero(model.population_policy_exposure_count) == 0
    assert torch.count_nonzero(model.population_policy_support_count) == 0


def test_shadow_only_observation_never_mutates_registered_belief_rows():
    """M5 telemetry cannot silently become a legacy lifecycle controller."""

    model = _cpu_model()
    model.exposure_count = torch.zeros(3, dtype=torch.int32)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    sources = tuple(SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4))
    token = _token(optimizer_update=1)
    result = SimpleNamespace(
        evidence=_b4_evidence(sources, token),
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=0, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=model, parameter_version=1), gaussians=model,
        args=SimpleNamespace(), opt=SimpleNamespace(), pipe=SimpleNamespace(),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0, population_shadow_enabled=True,
        population_shadow_work_token_provider=lambda _result, _context: MutationWorkToken(1, 1),
        population_shadow_versions_provider=lambda _result, _context: versions,
        population_shadow_belief_provider=CanonicalMappingRuntime._population_shadow_belief_from_rows,
        population_shadow_capacity=4, population_shadow_defer_until_post_step_commit=True,
    )
    runtime.commit_step(
        OptimizationRuntimeContext(
            optimizer_update_id=1,
            optimizer_update_permitted=True,
            gradient_policy_context=SimpleNamespace(
                stable_geometry=SimpleNamespace(
                    successful_updates_since_admission=1, iter_since_new_frame=0,
                )
            ),
        ),
        result,
    )
    production = object.__new__(CanonicalMappingRuntime)
    production.gaussians = model
    production.config = SimpleNamespace(
        static_options={"population_live_enabled": False, "population_shadow_enabled": True}
    )
    production._population_shadow_evidence_versions = lambda: versions
    before = tuple(
        getattr(model, name).clone()
        for name in ("exposure_count", "last_exposed_update", "visibility_ema", "utility_ema", "expected_marginal_gain")
    )

    production._observe_population_policy_inputs_after_step(runtime, result, {})

    for name, expected in zip(
        ("exposure_count", "last_exposed_update", "visibility_ema", "utility_ema", "expected_marginal_gain"),
        before,
        strict=True,
    ):
        assert torch.equal(getattr(model, name), expected), name


def test_true_b4_accumulator_masks_nonfinite_or_inconsistent_rows_without_writes():
    """Malformed device evidence cannot be clamped into Gaussian belief."""

    model = _cpu_model()
    model.exposure_count = torch.zeros(3, dtype=torch.int32)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.visibility_ema.zero_()
    model.utility_ema.zero_()
    model.expected_marginal_gain.zero_()
    sources = tuple(SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4))
    token = _token(optimizer_update=1)
    evidence = replace(
        _b4_evidence(sources, token),
        # Opportunity/contribution are exact, so the negative finite gradient
        # itself must invalidate row zero rather than being hidden by another
        # malformed field.  Rows one/two exercise NaN and Inf rejection.
        meaningful_measurement_opportunity=torch.tensor([1.0, 1.0, 2.0]),
        render_contribution=torch.tensor([0.25, 0.25, 0.5]),
        gradient_utility_evidence=torch.tensor([-0.1, float("nan"), float("inf")]),
    )
    result = SimpleNamespace(
        evidence=evidence,
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    before = tuple(
        getattr(model, name).clone()
        for name in ("exposure_count", "last_exposed_update", "visibility_ema", "utility_ema", "expected_marginal_gain")
    )

    accumulate_true_b4_belief(
        model,
        result,
        current_versions=PopulationShadowVersions(
            map_content=0,
            topology=0,
            coordinate_frame=0,
            schema=0,
            optimizer_update=1,
            surface_evidence=0,
            policy=0,
            parameter=1,
        ),
    )

    for name, expected in zip(
        ("exposure_count", "last_exposed_update", "visibility_ema", "utility_ema", "expected_marginal_gain"),
        before,
        strict=True,
    ):
        assert torch.equal(getattr(model, name), expected), name


def test_true_b4_accumulator_uses_dedicated_policy_counts_from_distinct_sources_only():
    """The live policy must not inherit resolution-veto or lifecycle counts."""

    model = _cpu_model()
    model.exposure_count = torch.full((3,), 41, dtype=torch.int32)
    model.support_count = torch.full((3,), 37, dtype=torch.int16)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    ensure_population_policy_e0_buffers(model)
    token = _token(optimizer_update=1)
    sources = tuple(SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4))
    result = SimpleNamespace(
        evidence=_b4_evidence(sources, token),
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=0, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )

    assert accumulate_true_b4_belief(model, result, current_versions=versions) is GaussianBeliefAccumulationStatus.UPDATED
    assert model.population_policy_exposure_count.tolist() == [1, 1, 2]
    assert model.population_policy_support_count.tolist() == [1, 1, 2]
    assert model.exposure_count.tolist() == [41, 41, 41]
    assert model.support_count.tolist() == [37, 37, 37]


def test_true_b4_accumulator_deduplicates_replayed_source_identities_across_epochs():
    """A replay updates exposure but cannot add physical-view support twice."""

    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    token = _token(optimizer_update=1)
    sources = tuple(
        SourceIdentity("replay", 5, index, 1_700_000_000 + index)
        for index in range(4)
    )
    result = SimpleNamespace(
        evidence=_b4_evidence(sources, token),
        map_version_before=0, map_version_after=0, topology_version=0,
        parameter_version_before=0, parameter_version_after=1,
        optimizer_step_count=1, supervision_view_count=4, batch_size=4,
        skipped_view_count=0, true_camera_batch=True,
        camera_batch_rasterizer_calls=1, step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=0, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )

    assert accumulate_true_b4_belief(model, result, current_versions=versions) is GaussianBeliefAccumulationStatus.UPDATED
    before = model.population_policy_support_count.clone()
    assert accumulate_true_b4_belief(model, result, current_versions=versions) is GaussianBeliefAccumulationStatus.UPDATED
    assert torch.equal(model.population_policy_support_count, before)


def test_true_b4_accumulator_saturates_dedicated_support_before_int16_wrap():
    """Unique physical sources cannot wrap a bounded policy support counter."""

    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    model.last_exposed_update = torch.full((3,), -1, dtype=torch.int32)
    model.population_policy_support_count.fill_(torch.iinfo(torch.int16).max)
    token = _token(optimizer_update=1)
    sources = tuple(
        SourceIdentity("replay", 6, index, 1_700_000_100 + index)
        for index in range(4)
    )
    result = SimpleNamespace(
        evidence=_b4_evidence(sources, token),
        map_version_before=0, map_version_after=0, topology_version=0,
        parameter_version_before=0, parameter_version_after=1,
        optimizer_step_count=1, supervision_view_count=4, batch_size=4,
        skipped_view_count=0, camera_batch_rasterizer_calls=1,
        true_camera_batch=True, step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=0, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )

    assert accumulate_true_b4_belief(model, result, current_versions=versions) is GaussianBeliefAccumulationStatus.UPDATED
    assert torch.equal(
        model.population_policy_support_count,
        torch.full((3,), torch.iinfo(torch.int16).max, dtype=torch.int16),
    )


@pytest.mark.parametrize("source_count", [1, 4])
def test_true_b4_accumulator_rejects_nonindependent_physical_source_evidence(source_count):
    """Repeated or singleton SourceIdentity evidence cannot satisfy E0 support."""

    model = _cpu_model()
    ensure_population_policy_e0_buffers(model)
    token = _token(optimizer_update=1)
    first = SourceIdentity("replay", 5, 1, 1_700_000_001)
    sources = (first,) if source_count == 1 else (first,) * 4
    distinct = tuple(
        SourceIdentity("replay", 5, i, 1_700_000_000 + i) for i in range(4)
    )
    evidence = _b4_evidence(distinct, token)
    object.__setattr__(evidence, "source_identities", sources)
    result = SimpleNamespace(
        evidence=evidence,
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
        parameter_version_before=0,
        parameter_version_after=1,
        optimizer_step_count=1,
        supervision_view_count=4,
        batch_size=4,
        skipped_view_count=0,
        true_camera_batch=True,
        camera_batch_rasterizer_calls=1,
        step_allowed=True,
    )
    versions = PopulationShadowVersions(
        map_content=0, topology=0, coordinate_frame=0, schema=0,
        optimizer_update=1, surface_evidence=0, policy=0, parameter=1,
    )

    assert accumulate_true_b4_belief(model, result, current_versions=versions) is GaussianBeliefAccumulationStatus.INVALID_EVIDENCE
    assert torch.count_nonzero(model.population_policy_exposure_count) == 0
    assert torch.count_nonzero(model.population_policy_support_count) == 0


@requires_gsplat_device
def test_normal_live_runtime_carries_completed_b4_to_typed_plan_and_real_arbiter(monkeypatch):
    """Breaking the normal post-step bridge must prevent the atomic CREATE."""

    runtime, model, backend, sources, plan, result = _normal_live_b4_activation(
        favorable_evidence=True
    )

    uid_storage = model.gaussian_uid.untyped_storage().data_ptr()
    original_to = torch.Tensor.to
    original_cpu = torch.Tensor.cpu
    original_tolist = torch.Tensor.tolist

    def _is_uid_storage(tensor):
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.numel() == 3
            and tensor.untyped_storage().data_ptr() == uid_storage
        )

    def guarded_to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if _is_uid_storage(tensor) and target is not None and torch.device(target).type == "cpu":
            pytest.fail("gaussian_uid crossed an unowned device-to-host boundary")
        return original_to(tensor, *args, **kwargs)

    def guarded_cpu(tensor, *args, **kwargs):
        if _is_uid_storage(tensor):
            pytest.fail("gaussian_uid crossed an unowned device-to-host boundary")
        return original_cpu(tensor, *args, **kwargs)

    def guarded_tolist(tensor, *args, **kwargs):
        if _is_uid_storage(tensor):
            pytest.fail("gaussian_uid was synchronously materialized on host")
        return original_tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guarded_to)
    monkeypatch.setattr(torch.Tensor, "cpu", guarded_cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", guarded_tolist)

    captured = []
    real_arbiter_commit = GaussianMutationArbiter.commit

    def capture_real_commit(arbiter, proposals):
        captured.extend(proposals)
        return real_arbiter_commit(arbiter, proposals)

    monkeypatch.setattr(GaussianMutationArbiter, "commit", capture_real_commit)

    try:
        provider = runtime._population_live_target_binding_provider
        shadow_drop, targets = runtime._observe_population_policy_inputs_after_step(
            runtime._optimization_runtime,
            result,
            {source: _b4_source_lease(source) for source in sources},
        )
        # The measured source row has gain before a CREATE consumes it.  The
        # atomic plan correctly resets the newly created row afterwards.
        torch.testing.assert_close(
            model.expected_marginal_gain[-1], torch.tensor(0.5), atol=1e-6, rtol=0
        )
        assert torch.equal(
            model.population_policy_exposure_count, torch.tensor([1, 1, 2])
        )
        assert torch.equal(
            model.population_policy_support_count,
            torch.tensor(
                [1, 1, 2], dtype=model.population_policy_support_count.dtype
            ),
        )
        assert torch.equal(model.p_exist, torch.ones(3, dtype=model.p_exist.dtype))
        torch.testing.assert_close(
            targets.support_evidence / targets.independent_view_count, torch.tensor([1.0])
        )
        policy = runtime._optimization_runtime.population_shadow_runtime.policy
        assert (policy.min_meaningful_exposures, policy.min_independent_views) == (1, 1)
        decision = runtime._optimization_runtime.population_shadow_runtime._activation_pending[-1].decision
        assert torch.any(decision.actions == PopulationAction.CREATE.code), decision.actions
        used_live, commit = runtime._post_step_structural_mutation(
            result, plan, post_step_utility=None
        )

        assert shadow_drop is None
        assert targets is not None and targets.count == 1
        assert provider.last_observation.source_count == 4
        assert used_live and commit is not None
        assert len(captured) == 1
        assert isinstance(captured[0], PopulationMutationProposal)
        assert isinstance(captured[0].plan, PopulationMutationPlan)
        assert captured[0].plan.create_rows.numel() == 1
        assert commit.applied == 1
        assert commit.inserted == 1
        assert backend.population_count == 4
        assert torch.equal(model.gaussian_uid, torch.tensor([10, 11, 12, 13]))
        _registry, host_uids, next_uid = backend.current_population_plan_identity_snapshot()
        assert host_uids == ("10", "11", "12", "13")
        assert next_uid == 14
        committed = backend.current_population_plan_snapshot()
        assert committed is not None
        assert committed.map_content == 1
        assert committed.topology == 1
        assert committed.optimizer_update == 2
        assert committed.parameter == 2
    finally:
        runtime.close()


@requires_gsplat_device
def test_normal_live_b4_at_capacity_binds_and_applies_one_paired_relocation():
    """The real B4 bridge must reach attested constant-N RELOCATE atomically."""

    runtime, model, backend, sources, plan, result = _normal_live_b4_activation(
        favorable_evidence=True
    )
    runtime._runtime_policy = replace(runtime._runtime_policy, cap_max=3)
    runtime._optimization_runtime.population_shadow_capacity = 3
    model.redundancy[0] = 1.0

    try:
        shadow_drop, targets = runtime._observe_population_policy_inputs_after_step(
            runtime._optimization_runtime,
            result,
            {source: _b4_source_lease(source) for source in sources},
        )
        pending = runtime._optimization_runtime.population_shadow_runtime._activation_pending[-1]
        assert torch.count_nonzero(
            pending.decision.actions == PopulationAction.RELOCATE.code
        ) == 1

        used_live, commit = runtime._post_step_structural_mutation(
            result, plan, post_step_utility=None
        )

        assert shadow_drop is None
        assert targets is not None and targets.count == 1
        assert used_live and commit is not None
        assert commit.applied == 1
        assert commit.change.relocated == 1
        assert commit.inserted == 0
        assert backend.population_count == 3
        assert runtime._native_geometry_surface_target_attestation is not None
    finally:
        runtime.close()


@requires_gsplat_device
def test_normal_live_runtime_abstains_without_meaningful_measurement_evidence(monkeypatch):
    """A valid B4 target cannot bypass the policy measurement-evidence gate."""

    runtime, model, backend, sources, plan, result = _normal_live_b4_activation(
        favorable_evidence=False
    )
    calls = []
    real_arbiter_commit = GaussianMutationArbiter.commit

    def capture_real_commit(arbiter, proposals):
        calls.append(tuple(proposals))
        return real_arbiter_commit(arbiter, proposals)

    monkeypatch.setattr(GaussianMutationArbiter, "commit", capture_real_commit)

    try:
        shadow_drop, targets = runtime._observe_population_policy_inputs_after_step(
            runtime._optimization_runtime,
            result,
            {source: _b4_source_lease(source) for source in sources},
        )
        used_live, commit = runtime._post_step_structural_mutation(
            result, plan, post_step_utility=None
        )

        assert shadow_drop is None
        assert targets is not None and targets.count == 1
        assert used_live and commit is None
        assert runtime._population_live_activation_disposition == "no_structural_action"
        assert calls == []
        assert backend.population_count == 3
        assert torch.equal(model.gaussian_uid, torch.tensor([10, 11, 12]))
        assert runtime._versions() == (0, 0, 1)
        _registry, host_uids, next_uid = backend.current_population_plan_identity_snapshot()
        assert host_uids == ("10", "11", "12")
        assert next_uid == 13
    finally:
        runtime.close()


def test_production_context_fails_closed_before_installing_duplicate_uid_inventory():
    model = _cpu_model()
    backend = _backend(model)
    backend._population_hypothesis_uids = ("10", "10", "12")
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.config = SimpleNamespace(
        static_options={
            "population_live_enabled": True,
            "population_shadow_enabled": True,
        }
    )
    runtime.gaussians = model
    runtime.backend = backend
    runtime._engine = SimpleNamespace(session=SimpleNamespace(parameter_version=0))
    runtime._runtime_policy = SimpleNamespace(cap_max=4)
    runtime._transfer_owner = DeviceTransferOwner("cpu")
    runtime._population_live_max_actions = 1
    source = SourceIdentity("replay", 5, 0, 1_700_000_000)
    plan_identity = SimpleNamespace(
        snapshot_id=23,
        supervision_sources=(source,),
        replay_sources=(),
        mutation_budget=SimpleNamespace(max_total=1),
    )

    with pytest.raises(ValueError, match="UID ledger"):
        runtime._population_live_plan_context(
            _token(), MutationWorkToken(epoch=0, units=1), plan_identity,
        )
    with pytest.raises(RuntimeError, match="has not been installed"):
        backend.current_population_plan_snapshot()


def test_atomic_population_plan_replaces_params_optimizer_belief_lineage_and_anchors():
    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _replace_plan(model)

    result = GaussianMutationArbiter(backend).commit((_proposal(plan, payload),))

    assert result.change.map_content_version == 1
    assert result.change.topology_version == 1
    assert backend.current_population_plan_snapshot() == plan.next_snapshot_token
    assert torch.equal(model.params["means"].detach()[0], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(model.params["means"].detach()[1:], payload.positions)
    assert torch.equal(model.params["sh0"].detach()[1:, 0], RGB2SH(payload.colors))
    assert torch.equal(model.gaussian_uid, torch.tensor([11, 100, 101]))
    assert torch.equal(model.root_uid, torch.tensor([11, 100, 101]))
    assert torch.equal(model.parent_uid, torch.tensor([-1, -1, -1]))
    assert torch.equal(model.p_exist, torch.tensor([0.25, 1.0, 1.0], dtype=torch.float16))
    assert torch.equal(model.anchor_xyz, model.params["means"].detach())
    for name, optimizer in model.optimizers.items():
        state = optimizer.state[model.params[name]]
        assert torch.equal(state["exp_avg"][0], torch.full_like(state["exp_avg"][0], float(10 + list(model.optimizers).index(name))))
        assert torch.count_nonzero(state["exp_avg"][1:]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][1:]) == 0
    assert {descriptor.name for descriptor in backend.current_population_plan_registry_snapshot()} == set(plan.row_transform.covered_fields)
    _registry, host_uids, next_uid = backend.current_population_plan_identity_snapshot()
    assert host_uids == ("11", "100", "101")
    assert next_uid == 102


def test_atomic_population_plan_retirement_compacts_all_registered_rows():
    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _replace_plan(model, retire_only=True)

    GaussianMutationArbiter(backend).commit((_proposal(plan, payload),))

    registry = model.build_row_state_registry()
    assert backend.population_count == 2
    for descriptor in registry.descriptors:
        if descriptor.name.startswith("params."):
            value = model.params[descriptor.name.removeprefix("params.")]
        elif descriptor.name.startswith("optimizer."):
            _, parameter_name, state_name = descriptor.name.split(".", 2)
            value = model.optimizers[parameter_name].state[model.params[parameter_name]][state_name]
        else:
            _, attribute = descriptor.name.split(".", 1)
            value = getattr(model, attribute)
        assert value.shape[0] == 2, descriptor.name
    assert torch.equal(model.gaussian_uid, torch.tensor([11, 12]))


def test_invalid_registry_attestation_fails_before_any_model_write():
    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _replace_plan(model)
    bad_plan = replace(plan, row_state_registry_snapshot=tuple(reversed(plan.row_state_registry_snapshot)))
    before_means = model.params["means"].detach().clone()
    before_uid = model.gaussian_uid.clone()

    with pytest.raises(ValueError, match="registry snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(bad_plan, payload),))

    assert torch.equal(model.params["means"].detach(), before_means)
    assert torch.equal(model.gaussian_uid, before_uid)
    assert backend.current_population_plan_snapshot() == _token()


def test_atomic_population_plan_scatters_unsorted_target_rows_and_exact_plan_identities():
    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    registry = model.build_row_state_registry()
    payload = _payload()
    plan = PopulationMutationPlan(
        snapshot_token=_token(), next_snapshot_token=_token(map_content=1, topology=1, optimizer_update=1, parameter=1),
        row_transform=PopulationRowTransform(
            old_population=3, new_population=3,
            survivor_old_indices=torch.tensor([1]), retired_old_indices=torch.tensor([0, 2]),
            created_rows=(PopulationRowCreation(row_index=1, hypothesis_uid="301"),),
            relocated_rows=(PopulationRowRelocation(row_index=2, donor_row_index=0, old_hypothesis_uid="10", new_hypothesis_uid="300"),),
            covered_fields=registry.names,
        ),
        survivor_hypothesis_uids=("11",), mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=_token(),
        capacity=3, old_population=3, new_population=3,
        retire_rows=torch.tensor([2]), relocate_donor_rows=torch.tensor([0]), relocate_target_rows=torch.tensor([2]), create_rows=torch.tensor([1]),
        target_payload=payload.positions.clone(), donor_hypothesis_uids=("10",), relocated_hypothesis_uids=("300",), created_hypothesis_uids=("301",),
        reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS), max_actions=3,
        row_state_registry_snapshot=PopulationMutationPlan.snapshot_registry(registry),
    )

    GaussianMutationArbiter(backend).commit((_proposal(plan, payload),))

    # Payload order is relocation then creation, not output-row order.
    assert torch.equal(model.params["means"].detach(), torch.tensor([[3.0, 4.0, 5.0], [40.0, 41.0, 42.0], [30.0, 31.0, 32.0]]))
    assert torch.equal(model.gaussian_uid, torch.tensor([11, 301, 300]))
    assert model._next_gaussian_uid == 302


def test_plan_donor_identity_mismatch_fails_before_any_model_write():
    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _replace_plan(model)
    bad_transform = replace(
        plan.row_transform,
        relocated_rows=(PopulationRowRelocation(row_index=1, donor_row_index=0, old_hypothesis_uid="999", new_hypothesis_uid="100"),),
    )
    bad_plan = replace(plan, donor_hypothesis_uids=("999",), row_transform=bad_transform)
    before = model.params["means"].detach().clone()

    with pytest.raises(ValueError, match="donor hypothesis identity"):
        GaussianMutationArbiter(backend).commit((_proposal(bad_plan, payload),))

    assert torch.equal(model.params["means"].detach(), before)


def test_bootstrap_identity_ledger_rejects_stale_uid_counter_without_device_read():
    model = _cpu_model()
    backend = GsplatMapBackend(model)
    backend.install_population_identity_snapshot_from_bootstrap()
    model._next_gaussian_uid = 14

    with pytest.raises(ValueError, match="watermark is stale"):
        backend.current_population_plan_identity_snapshot()


def test_non_plan_topology_mutation_invalidates_population_identity_ledger():
    backend = _backend(_cpu_model())

    backend.record_external_mutation(inserted=1)

    with pytest.raises(ValueError, match="UID ledger is unavailable"):
        backend.current_population_plan_identity_snapshot()


class _FailOnceGsplatModel(GsplatGaussianModel):
    def __setattr__(self, name, value):
        if name == "anchor_xyz" and getattr(self, "_fail_next_anchor_assignment", False):
            object.__setattr__(self, "_fail_next_anchor_assignment", False)
            raise RuntimeError("injected anchor assignment failure")
        object.__setattr__(self, name, value)


def test_atomic_population_plan_rolls_back_every_registry_family_after_injected_write_failure():
    model = _cpu_model(cls=_FailOnceGsplatModel)
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _replace_plan(model)
    registry = model.build_row_state_registry()
    before_params = {name: param for name, param in model.params.items()}
    before_optimizer_state = {
        name: model.optimizers[name].state[param]
        for name, param in before_params.items()
    }
    before_buffers = {
        descriptor.name: (
            getattr(model, descriptor.name.split(".", 1)[1])
            if not descriptor.name.startswith(("params.", "optimizer."))
            else None
        )
        for descriptor in registry.descriptors
    }
    before_next_uid = model._next_gaussian_uid
    model._fail_next_anchor_assignment = True

    with pytest.raises(RuntimeError, match="injected anchor"):
        GaussianMutationArbiter(backend).commit((_proposal(plan, payload),))

    for name, param in before_params.items():
        assert model.params[name] is param
        assert model.optimizers[name].param_groups[0]["params"][0] is param
        assert model.optimizers[name].state[param] is before_optimizer_state[name]
    for descriptor in registry.descriptors:
        if descriptor.name.startswith(("params.", "optimizer.")):
            continue
        assert getattr(model, descriptor.name.split(".", 1)[1]) is before_buffers[descriptor.name]
    assert model._next_gaussian_uid == before_next_uid
    assert backend.current_population_plan_snapshot() == _token()


class _PostCommitRowCountCorruptionModel(GsplatGaussianModel):
    """Corrupt one registered anchor only after plan preflight has completed."""

    def __setattr__(self, name, value):
        if name == "anchor_xyz" and getattr(self, "_corrupt_plan_commit", False):
            value = value[:-1]
        object.__setattr__(self, name, value)


@pytest.mark.parametrize("action", ("create", "relocate", "retire"))
def test_each_population_plan_action_rejects_a_post_commit_registry_row_mismatch(action):
    """CREATE/RELOCATE/RETIRE must not publish a partial row-state transform."""

    model = _cpu_model(cls=_PostCommitRowCountCorruptionModel)
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    plan, payload = _single_action_plan(model, action)
    model._corrupt_plan_commit = True

    with pytest.raises(ValueError, match="live population row attestation"):
        GaussianMutationArbiter(backend).commit((_proposal(plan, payload),))


def test_live_population_ownership_rejects_direct_legacy_add_and_prune_writers():
    """A live plan owner makes model-level append/prune writes unavailable."""

    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())

    with pytest.raises(RuntimeError, match="live population ownership"):
        model.add_points_as_gaussians(torch.zeros((1, 3)), torch.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="live population ownership"):
        model.prune_points(torch.tensor([True, False, False]))


def test_live_population_ownership_rejects_public_optimizer_tensor_append():
    """The public optimizer append primitive cannot bypass the row transform."""

    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    before = {name: parameter.detach().clone() for name, parameter in model.params.items()}
    append = {name: torch.zeros_like(parameter[:1]) for name, parameter in model.params.items()}

    with pytest.raises(RuntimeError, match="live population ownership"):
        model.cat_tensors_to_optimizer(append)

    for name, parameter in model.params.items():
        assert torch.equal(parameter.detach(), before[name]), name


def test_live_population_ownership_rejects_legacy_transaction_relocation_reset():
    """In-place legacy relocation cannot bypass the complete plan reset fields."""

    model = _cpu_model()
    backend = _backend(model)
    backend.install_population_plan_snapshot(_token())
    transaction = backend.begin_mutation()

    with pytest.raises(RuntimeError, match="live population ownership"):
        transaction.relocate(
            RelocationPlan(
                mask=torch.tensor([True, False, False]),
                new_means=torch.tensor([[20.0, 21.0, 22.0]]),
            )
        )
