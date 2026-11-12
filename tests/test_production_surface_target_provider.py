"""Production M6 binding of same-transaction geometry evidence to targets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.cli.parser import parse_training_args
from eligsir.mapping.evidence.depth_surface_adapter import DepthSurfaceEvidenceInput
from eligsir.mapping.evidence.evidence_demand import EvidenceDemandBatch
from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease
from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    MutationWorkToken,
    PopulationAction,
    PopulationDecision,
)
from eligsir.mapping.scheduler.mcmc_cadence import MCMCCadenceOwner
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.optimization.gaussian.contracts import OptimizationEvidence
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
from eligsir.runtime.production_application import (
    CanonicalMappingRuntime,
    MappingBackendCompatibility,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("replay", 5, ordinal, 1_700_000_000 + ordinal)


def _token(
    *, map_content: int = 3, surface_evidence: int = 8, parameter: int = 10
) -> EvidenceSnapshotToken:
    return EvidenceSnapshotToken(
        map_content=map_content,
        topology=4,
        coordinate_frame=5,
        schema=6,
        optimizer_update=7,
        surface_evidence=surface_evidence,
        policy=9,
        parameter=parameter,
    )


def _buffer(value: torch.Tensor) -> BufferRef:
    return BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding="test",
            device=str(value.device),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )


def _source_lease(
    source: SourceIdentity,
    *,
    depth_meters: torch.Tensor | None = None,
    c2w: torch.Tensor | None = None,
    camera_intrinsics: torch.Tensor | None = None,
) -> PreparedSourceLease:
    depth_meters = (
        torch.ones(4, 5) if depth_meters is None else depth_meters
    )
    c2w = torch.eye(4) if c2w is None else c2w
    camera_intrinsics = (
        torch.tensor(
            [[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]
        )
        if camera_intrinsics is None
        else camera_intrinsics
    )
    height, width = depth_meters.shape
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=False,
            geometry_population_target=True,
        ),
        source_metadata={"native_height": height, "native_width": width},
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "depth_meters": _buffer(depth_meters),
            "rgb": _buffer(torch.zeros(height, width, 3)),
            "c2w": _buffer(c2w),
            "camera_intrinsics": _buffer(camera_intrinsics),
        },
        device="cpu",
    )


def _geometry(source: SourceIdentity, *, token: EvidenceSnapshotToken, tier: str = "native"):
    return GeometryRenderLease(
        source_identity=source,
        snapshot_token=token,
        resolution_tier=tier,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=False,
            geometry_population_target=True,
        ),
        alpha=torch.ones(4, 5, 1),
        rendered_depth=torch.full((4, 5, 1), 2.0),
    )


def _perspective_plane_depth(
    normal: torch.Tensor,
    *,
    height: int,
    width: int,
    intrinsics: torch.Tensor,
    offset: float = 2.0,
) -> torch.Tensor:
    """Analytic plane oracle: ``normal · (depth * ray) == offset``."""

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    rays = torch.stack(
        (
            (xs - intrinsics[0, 2]) / intrinsics[0, 0],
            (ys - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xs),
        ),
        dim=-1,
    )
    return offset / (rays @ normal)


def _tilted_plane() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[13.0, 0.0, 4.0], [0.0, 11.0, 3.0], [0.0, 0.0, 1.0]]
    )
    normal = torch.nn.functional.normalize(
        torch.tensor([0.3, -0.2, 1.0]), dim=0
    )
    return normal, _perspective_plane_depth(
        normal, height=7, width=9, intrinsics=intrinsics
    ), intrinsics


def _reduce_native_depth(
    depth: torch.Tensor,
    *,
    target: tuple[int, int],
    intrinsics: torch.Tensor,
    c2w: torch.Tensor | None = None,
):
    """Run the real reducer with one deliberately uncovered target pixel."""

    from eligsir.mapping.evidence.production_surface_targets import (
        reduce_native_depth_surface_evidence,
    )
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionRenderReuse,
    )

    height, width = depth.shape
    source = _source(97)
    token = _token()
    alpha = torch.ones(height, width, 1)
    alpha[target[1], target[0], 0] = 0.0
    geometry = GeometryRenderLease(
        source_identity=source,
        snapshot_token=token,
        resolution_tier="native",
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=False,
            geometry_population_target=True,
        ),
        alpha=alpha,
        rendered_depth=torch.ones(height, width, 1),
    )
    reuse = DepthInsertionRenderReuse(
        alpha=alpha,
        rendered_depth=geometry.rendered_depth,
        depth_meters=depth,
        rgb=torch.zeros(height, width, 3),
        c2w=torch.eye(4) if c2w is None else c2w,
        camera_intrinsics=intrinsics,
    )
    return reduce_native_depth_surface_evidence(reuse, _evidence(geometry), source)


def _evidence(*leases: GeometryRenderLease) -> SupervisionEvidenceBatch:
    token = leases[0].snapshot_token
    return SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="b4-evidence",
        source_identities=tuple(lease.source_identity for lease in leases),
        modalities=tuple(
            SupervisionModality(
                depth_loss_enabled=True,
                depth_evidence_enabled=True,
                geometry_evidence_enabled=True,
            )
            for _ in leases
        ),
        modality_eligibility=tuple(lease.modality_eligibility for lease in leases),
        geometry_render_leases=leases,
    )


def _reduced(target_id: int) -> DepthSurfaceEvidenceInput:
    return DepthSurfaceEvidenceInput(
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.02),
        support_evidence=torch.ones(1),
        conflict_evidence=torch.zeros(1),
        independent_view_count=torch.ones(1, dtype=torch.int32),
        repeated_exposure_count=torch.ones(1, dtype=torch.int32),
        target_ids=torch.tensor([target_id], dtype=torch.int64),
    )


def _create_decision(count: int) -> PopulationDecision:
    actions = torch.full((count,), PopulationAction.CREATE.code, dtype=torch.int8)
    targets = torch.full((count,), -1, dtype=torch.int64)
    selection = GaussianPopulationPolicy._structural_selection(
        actions, torch.ones(count), targets, max_actions=count,
    )
    return PopulationDecision.policy_output(
        actions=actions,
        target_indices=targets,
        evidence_demands=EvidenceDemandBatch(()),
        demand_masks=(),
        structural_selection=selection,
    )


def _structural_decision(action: PopulationAction) -> PopulationDecision:
    actions = torch.tensor([action.code], dtype=torch.int8)
    target_indices = torch.tensor(
        [0 if action is PopulationAction.RELOCATE else -1], dtype=torch.int64
    )
    selection = GaussianPopulationPolicy._structural_selection(
        actions, torch.ones(1), target_indices, max_actions=1,
    )
    return PopulationDecision.policy_output(
        actions=actions,
        target_indices=target_indices,
        evidence_demands=EvidenceDemandBatch(()),
        demand_masks=(),
        structural_selection=selection,
    )


def _binder(decision: PopulationDecision, targets, _current, _work):
    return decision.bind_surface_target_ids(
        targets.target_ids,
        target_snapshot=targets.snapshot,
        structural_action=PopulationAction.CREATE,
    ), targets


def _live_runtime(
    *, batch_size: int = 1, monkeypatch: pytest.MonkeyPatch | None = None
) -> CanonicalMappingRuntime:
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
    args.streaming_optimization_batch_size = batch_size
    config = mapping_runtime_config_from_args(args)
    if monkeypatch is not None:
        # These are construction/composition tests.  They must run in the
        # CPU-Torch simulation lane, whose contract deliberately has no CUDA
        # device.  Replace only the CUDA-owning model/services boundary after
        # production_application has bound its lazy imports; the runtime's
        # scheduler/policy wiring remains real.
        import eligsir.runtime.production_application as application

        application._load_accelerator_dependencies()

        class _CpuModel:
            def __init__(self, _sh_degree):
                pass

        class _CpuServices:
            def __init__(self, _model):
                self.backend = SimpleNamespace(
                    map_content_version=0,
                    topology_version=0,
                    coordinate_frame_version=0,
                    schema_version=0,
                )

            def close(self):
                pass

        monkeypatch.setattr(application, "GsplatGaussianModel", _CpuModel)
        monkeypatch.setattr(application, "CanonicalMappingServices", _CpuServices)
    return CanonicalMappingRuntime(
        config,
        MappingBackendCompatibility(
            dataset=SimpleNamespace(sh_degree=3),
            opt=SimpleNamespace(),
            pipe=SimpleNamespace(),
        ),
    )


@pytest.mark.parametrize(
    ("camera_normal", "rotation"),
    (
        pytest.param(
            torch.tensor([0.0, 0.0, 1.0]), torch.eye(3), id="frontoparallel"
        ),
        pytest.param(
            torch.tensor([0.3, -0.2, 1.0]), torch.eye(3), id="tilted"
        ),
        pytest.param(
            torch.tensor([0.3, -0.2, 1.0]),
            torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            id="world_rotated",
        ),
    ),
)
def test_native_depth_reducer_normal_matches_analytic_perspective_plane(
    camera_normal: torch.Tensor, rotation: torch.Tensor
):
    """Catch image-depth finite differences on a perspective plane."""

    height, width = 7, 9
    intrinsics = torch.tensor(
        [[13.0, 0.0, 4.0], [0.0, 11.0, 3.0], [0.0, 0.0, 1.0]]
    )
    unit_camera_normal = torch.nn.functional.normalize(camera_normal, dim=0)
    depth = _perspective_plane_depth(
        unit_camera_normal,
        height=height,
        width=width,
        intrinsics=intrinsics,
    )
    c2w = torch.eye(4)
    c2w[:3, :3] = rotation

    reduced = _reduce_native_depth(
        depth, target=(4, 3), intrinsics=intrinsics, c2w=c2w
    )

    assert reduced is not None
    torch.testing.assert_close(reduced.normals[0], rotation @ unit_camera_normal)
    assert reduced.independent_view_count.tolist() == [1]


def test_native_depth_reducer_keeps_one_sided_stencil_when_other_neighbor_is_nan():
    """A valid left/down stencil must survive an unrelated missing neighbor."""

    camera_normal, depth, intrinsics = _tilted_plane()
    depth[3, 5] = float("nan")

    reduced = _reduce_native_depth(depth, target=(4, 3), intrinsics=intrinsics)

    assert reduced is not None
    torch.testing.assert_close(reduced.normals[0], camera_normal)
    assert reduced.independent_view_count.tolist() == [1]
    assert reduced.support_evidence.tolist() == [1.0]


@pytest.mark.parametrize("target", ((0, 0), (8, 6)), ids=("top_left", "bottom_right"))
def test_native_depth_reducer_uses_one_sided_boundary_stencil(
    target: tuple[int, int],
):
    """A boundary target has distinct rays even without centered differences."""

    camera_normal, depth, intrinsics = _tilted_plane()

    reduced = _reduce_native_depth(depth, target=target, intrinsics=intrinsics)

    assert reduced is not None
    torch.testing.assert_close(reduced.normals[0], camera_normal)
    assert reduced.independent_view_count.tolist() == [1]


@pytest.mark.parametrize(
    ("depth", "focal"),
    (
        pytest.param(2.0, 1_000_000.0, id="high_focal_close_plane"),
        pytest.param(torch.finfo(torch.float32).max, 13.0, id="large_depth_cross_overflow"),
    ),
)
def test_native_depth_reducer_normalizes_valid_extreme_scale_stencils(
    depth: float, focal: float
):
    """Finite, distinct rays remain valid even when an unscaled cross is extreme."""

    intrinsics = torch.tensor(
        [[focal, 0.0, 2.0], [0.0, focal, 2.0], [0.0, 0.0, 1.0]]
    )
    reduced = _reduce_native_depth(
        torch.full((5, 5), depth), target=(2, 2), intrinsics=intrinsics
    )

    assert reduced is not None
    torch.testing.assert_close(reduced.normals[0], torch.tensor([0.0, 0.0, 1.0]))
    assert reduced.independent_view_count.tolist() == [1]
    assert reduced.support_evidence.tolist() == [1.0]


def test_native_depth_reducer_invalidates_representationally_zero_tangents():
    """Underflowed distinct pixels cannot fabricate a normalized surface normal."""

    intrinsics = torch.tensor(
        [[13.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]]
    )
    reduced = _reduce_native_depth(
        torch.full((5, 5), 1e-45), target=(2, 2), intrinsics=intrinsics
    )

    assert reduced is not None
    assert reduced.independent_view_count.tolist() == [0]
    assert reduced.support_evidence.tolist() == [0.0]
    assert not torch.isfinite(reduced.normals).all()


@pytest.mark.parametrize(
    "invalid_neighbor", (float("nan"), float("inf"), 0.0), ids=("nan", "inf", "zero")
)
def test_native_depth_reducer_invalidates_target_without_a_valid_horizontal_stencil(
    invalid_neighbor: float,
):
    """Catch positive support from an invalid selected-pixel neighborhood."""

    intrinsics = torch.tensor(
        [[13.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]]
    )
    depth = torch.full((5, 5), 2.0)
    depth[2, 1] = invalid_neighbor
    depth[2, 3] = invalid_neighbor

    reduced = _reduce_native_depth(depth, target=(2, 2), intrinsics=intrinsics)

    assert reduced is not None
    assert reduced.independent_view_count.tolist() == [0]
    assert reduced.repeated_exposure_count.tolist() == [0]
    assert reduced.support_evidence.tolist() == [0.0]
    assert not torch.isfinite(reduced.normals).all()


@pytest.mark.parametrize(
    ("shape", "target"),
    (
        pytest.param((1, 5), (2, 0), id="one_row"),
        pytest.param((5, 1), (0, 2), id="one_column"),
        pytest.param((1, 1), (0, 0), id="one_pixel"),
    ),
)
def test_native_depth_reducer_invalidates_tiny_images_without_distinct_neighbor_rays(
    shape: tuple[int, int], target: tuple[int, int]
):
    """A clamped neighbor must never fabricate a normal or positive support."""

    height, width = shape
    intrinsics = torch.tensor(
        [[13.0, 0.0, width / 2.0], [0.0, 11.0, height / 2.0], [0.0, 0.0, 1.0]]
    )
    reduced = _reduce_native_depth(
        torch.full(shape, 2.0), target=target, intrinsics=intrinsics
    )

    assert reduced is not None
    assert reduced.independent_view_count.tolist() == [0]
    assert reduced.support_evidence.tolist() == [0.0]
    assert not torch.isfinite(reduced.positions).all()


def test_live_b4_constructor_mints_only_completed_true_c4_cadence_token(monkeypatch):
    runtime = _live_runtime(batch_size=4, monkeypatch=monkeypatch)
    try:
        runtime._engine = SimpleNamespace(
            session=SimpleNamespace(parameter_version=0)
        )
        runtime._population_shadow_active_plan = SimpleNamespace(
            mutation_budget=SimpleNamespace(max_total=1)
        )
        runtime._mcmc_cadence_owner = MCMCCadenceOwner(
            grow_interval_pixels=400,
            relocate_interval_pixels=400,
            min_optimizer_updates_after_structural_mutation=1,
        )
        runtime._work_clock_owner.record_optimization(
            optimizer_updates=1,
            supervised_views=4,
            supervised_pixels=400,
            mapping_active_ns=0,
        )
        runtime._work_clock_owner.record_depth_supervision(16)
        context = SimpleNamespace(optimizer_update_id=1)
        completed_b4_c4 = SimpleNamespace(
            depth_valid_sample_count=16,
            realized_batch_size=4,
            realized_camera_count=4,
            supervision_view_count=4,
            optimizer_step_count=1,
            evidence=_evidence(_geometry(_source(1), token=_token())),
        )

        wrong_type = SimpleNamespace(**vars(completed_b4_c4))
        wrong_type.evidence = OptimizationEvidence(payload=object())
        assert runtime._population_shadow_work_token(wrong_type, context).units == 0

        token = runtime._population_shadow_work_token(completed_b4_c4, context)

        assert token.units == 1
        assert token.geometry_opportunity is True
        incomplete_b2 = SimpleNamespace(
            **vars(completed_b4_c4),
        )
        incomplete_b2.realized_batch_size = 2
        incomplete_b2.realized_camera_count = 2
        incomplete_b2.supervision_view_count = 2
        assert runtime._population_shadow_work_token(incomplete_b2, context).units == 0
    finally:
        runtime.close()


def test_live_runtime_constructs_surface_provider_and_binds_only_current_completed_evidence(monkeypatch):
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    runtime = _live_runtime(monkeypatch=monkeypatch)
    try:
        assert runtime.config.static_options["insert_from_depth"] is False
        provider = runtime._population_live_target_binding_provider
        assert isinstance(provider, ProductionSurfaceTargetProvider)
        assert runtime._population_live_surface_target_provider is provider
        assert callable(runtime._population_live_context_provider)

        sources = (_source(1), _source(2))
        token = _token()
        evidence = _evidence(*(_geometry(source, token=token) for source in sources))
        assert all(
            not lease.modality_eligibility.depth_insertion
            and lease.modality_eligibility.geometry_population_target
            for lease in evidence.geometry_render_leases
        )
        targets = runtime._observe_population_surface_targets_after_step(
            SimpleNamespace(evidence=evidence),
            {source: _source_lease(source) for source in sources},
        )

        assert targets is not None
        assert targets.count == 1
        assert provider.last_observation.source_h2d_ops == 0
        assert provider.last_observation.extra_rasterizer_calls == 0
        action_fence = _token(parameter=token.parameter + 1)
        bound = provider(
            _create_decision(1), token, action_fence, MutationWorkToken(epoch=7, units=1)
        )
        assert bound is not None
        bound_decision, bound_targets = bound
        assert bound_targets is not targets
        assert bound_targets.positions is targets.positions
        assert bound_targets.snapshot == targets.snapshot
        assert bound_decision.structural_selection is not None
        assert torch.equal(
            bound_decision.structural_selection.surface_target_ids,
            targets.target_ids,
        )

        assert runtime._observe_population_surface_targets_after_step(
            SimpleNamespace(evidence=None), {}
        ) is None
        assert provider(
            _create_decision(1), token, token, MutationWorkToken(epoch=7, units=1)
        ) is None

        assert runtime._observe_population_surface_targets_after_step(
            SimpleNamespace(evidence=evidence),
            {source: _source_lease(source) for source in sources},
        ) is not None
        assert provider(
            _create_decision(1),
            _token(map_content=token.map_content + 1),
            _token(map_content=token.map_content + 1),
            MutationWorkToken(epoch=7, units=1),
        ) is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("action", "expected_target_id"),
    [
        (PopulationAction.CREATE, 101),
        (PopulationAction.RELOCATE, 101),
        (PopulationAction.RETIRE, -1),
    ],
)
def test_production_target_binder_preserves_arbitrary_exact_policy_action(
    action: PopulationAction, expected_target_id: int
):
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
        bind_policy_surface_targets,
    )

    sources = (_source(1), _source(2))
    token = _token()
    provider_targets = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101),
        binder=bind_policy_surface_targets,
        max_targets=1,
    ).observe_evidence(
        evidence=_evidence(*(_geometry(source, token=token) for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    )
    assert provider_targets is not None
    original = _structural_decision(action)

    bound = bind_policy_surface_targets(
        original,
        provider_targets,
        token,
        MutationWorkToken(epoch=7, units=1),
    )

    assert bound is not None
    decision, targets = bound
    assert targets is provider_targets
    assert decision.actions is original.actions
    assert decision.evidence_demands is original.evidence_demands
    assert decision.structural_selection is not None
    assert decision.structural_selection.action_codes.tolist() == [action.code]
    assert decision.structural_selection.surface_target_ids.tolist() == [
        expected_target_id
    ]
    assert decision.target_indices.tolist() == [expected_target_id]


def test_b1_target_provider_abstains_without_calling_the_reducer():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    source = _source(1)
    lease = _geometry(source, token=_token())
    evidence = _evidence(lease)
    resident = _source_lease(source)
    seen = []

    def reducer(reuse, observed_evidence, observed_source):
        seen.append((reuse, observed_evidence, observed_source))
        return _reduced(101)

    provider = ProductionSurfaceTargetProvider(reducer=reducer, binder=_binder, max_targets=1)
    targets = provider.observe_evidence(evidence=evidence, source_leases={source: resident})

    assert targets is None
    assert seen == []


def test_b4_target_provider_forms_one_depth_consistent_target_from_distinct_views():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    token = _token()
    sources = (_source(1), _source(2))
    evidence = _evidence(*(_geometry(source, token=token) for source in sources))
    calls = []

    def reducer(reuse, observed_evidence, source):
        calls.append((reuse, observed_evidence, source))
        # The distinct source identities observe the same physical surface.
        # Their per-view target IDs need not become separate population rows.
        return _reduced(101)

    provider = ProductionSurfaceTargetProvider(
        reducer=reducer, binder=_binder, max_targets=1,
    )
    targets = provider.observe_evidence(
        evidence=evidence, source_leases={source: _source_lease(source) for source in sources},
    )
    decision = _create_decision(1)
    bound = provider(
        decision,
        token,
        _token(parameter=token.parameter + 1),
        MutationWorkToken(epoch=7, units=1),
    )

    assert targets is not None
    assert len(calls) == 2
    assert all(observed is evidence for _reuse, observed, _source_value in calls)
    assert [source.source_ordinal for _reuse, _observed, source in calls] == [1, 2]
    assert bound is not None
    bound_decision, bound_targets = bound
    assert bound_targets is not targets
    assert bound_targets.positions is targets.positions
    assert bound_targets.target_ids is targets.target_ids
    assert bound_targets.count == 1
    assert bound_targets.target_ids.tolist() == [101]
    assert bound_targets.independent_view_count.tolist() == [2]
    assert bound_targets.repeated_exposure_count.tolist() == [2]
    assert bound_targets.support_evidence.tolist() == [2.0]
    assert bound_decision.structural_selection is not None
    assert bound_decision.structural_selection.surface_target_ids.tolist() == [101]
    assert bound_decision.target_indices.tolist() == [101]


def test_b4_target_provider_ignores_insertion_only_leases_for_exact_target_set():
    """The shared render lane retains insertion leases without polluting B4 targets."""

    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    token = _token()
    sources = tuple(_source(index) for index in range(4))
    target_only = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=False,
        geometry_population_target=True,
    )
    insertion_only = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=True,
        geometry_population_target=False,
    )
    eligibilities = (target_only, insertion_only, target_only, insertion_only)
    leases = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=eligibility,
            alpha=torch.ones(4, 5, 1),
            rendered_depth=torch.full((4, 5, 1), 2.0),
        )
        for source, eligibility in zip(sources, eligibilities, strict=True)
    )
    evidence = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="b4-mixed-leases",
        source_identities=sources,
        modalities=tuple(
            SupervisionModality(
                depth_loss_enabled=True,
                depth_evidence_enabled=True,
                geometry_evidence_enabled=eligibility.geometry_population_target,
            )
            for eligibility in eligibilities
        ),
        modality_eligibility=eligibilities,
        geometry_render_leases=leases,
    )
    observed = []
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda _reuse, _evidence, source: observed.append(source) or _reduced(101),
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=evidence,
        source_leases={source: _source_lease(source) for source in (sources[0], sources[2])},
    )

    assert targets is not None
    assert observed == [sources[0], sources[2]]


def test_uncovered_b4_depth_uses_valid_sensor_observations_as_positive_support():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
        reduce_native_depth_surface_evidence,
    )

    token = _token()
    sources = (_source(1), _source(2))
    geometry = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=False,
                geometry_population_target=True,
            ),
            alpha=torch.zeros(4, 5, 1),
            rendered_depth=torch.full((4, 5, 1), 2.0),
        )
        for source in sources
    )
    provider = ProductionSurfaceTargetProvider(
        reducer=reduce_native_depth_surface_evidence,
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=_evidence(*geometry),
        source_leases={source: _source_lease(source) for source in sources},
    )

    assert targets is not None
    assert targets.independent_view_count.tolist() == [2]
    assert targets.support_evidence.tolist() == [2.0]


def test_b4_target_reprojects_one_anchor_across_distinct_camera_poses():
    """Independent per-camera top-1 pixels are not physical correspondences."""

    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
        reduce_native_depth_surface_evidence,
    )

    token = _token()
    sources = (_source(1), _source(2))
    height, width = 3, 9
    intrinsics = torch.tensor(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]
    )
    poses = (torch.eye(4), torch.eye(4))
    poses[1][0, 3] = 0.2
    depths = tuple(torch.full((height, width), 2.0) for _ in sources)
    alphas = tuple(torch.ones(height, width, 1) for _ in sources)
    alphas[0][1, 4] = 0.0
    # This camera's own top-1 opportunity is a different world point.  The
    # shared physical point from camera zero instead reprojects to pixel x=3.
    alphas[1][1, 8] = 0.0
    geometry = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=False,
                geometry_population_target=True,
            ),
            alpha=alpha,
            rendered_depth=torch.full((height, width, 1), 2.0),
        )
        for source, alpha in zip(sources, alphas, strict=True)
    )
    provider = ProductionSurfaceTargetProvider(
        reducer=reduce_native_depth_surface_evidence,
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=_evidence(*geometry),
        source_leases={
            source: _source_lease(
                source,
                depth_meters=depth,
                c2w=pose,
                camera_intrinsics=intrinsics,
            )
            for source, depth, pose in zip(sources, depths, poses, strict=True)
        },
    )

    assert targets is not None
    torch.testing.assert_close(targets.positions, torch.tensor([[0.0, 0.0, 2.0]]))
    assert targets.independent_view_count.tolist() == [2]
    assert targets.support_evidence.tolist() == [2.0]
    assert provider.last_observation.source_h2d_ops == 0
    assert provider.last_observation.extra_rasterizer_calls == 0


def test_b4_target_counts_only_distinct_views_that_reproject_to_the_anchor():
    """A complete B4 may qualify from two views without fabricating four."""

    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
        reduce_native_depth_surface_evidence,
    )

    token = _token()
    sources = tuple(_source(index) for index in range(4))
    height, width = 3, 9
    intrinsics = torch.tensor(
        [[10.0, 0.0, 4.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]]
    )
    poses = tuple(torch.eye(4) for _ in sources)
    for index, pose in enumerate(poses):
        pose[0, 3] = 0.2 * index
    depths = tuple(
        torch.full((height, width), 2.0 if index < 2 else 3.0)
        for index in range(4)
    )
    alphas = tuple(torch.ones(height, width, 1) for _ in sources)
    alphas[0][1, 4] = 0.0
    geometry = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=False,
                geometry_population_target=True,
            ),
            alpha=alpha,
            rendered_depth=depth.unsqueeze(2),
        )
        for source, alpha, depth in zip(sources, alphas, depths, strict=True)
    )
    provider = ProductionSurfaceTargetProvider(
        reducer=reduce_native_depth_surface_evidence,
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=_evidence(*geometry),
        source_leases={
            source: _source_lease(
                source,
                depth_meters=depth,
                c2w=pose,
                camera_intrinsics=intrinsics,
            )
            for source, depth, pose in zip(sources, depths, poses, strict=True)
        },
    )

    assert targets is not None
    torch.testing.assert_close(targets.positions, torch.tensor([[0.0, 0.0, 2.0]]))
    assert targets.independent_view_count.tolist() == [2]
    assert targets.repeated_exposure_count.tolist() == [2]
    assert targets.support_evidence.tolist() == [2.0]


def test_b4_target_provider_replaying_the_same_sources_does_not_accumulate_independent_support():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    token = _token()
    sources = (_source(1), _source(2))
    evidence = _evidence(*(_geometry(source, token=token) for source in sources))
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101), binder=_binder, max_targets=1,
    )

    first = provider.observe_evidence(
        evidence=evidence,
        source_leases={source: _source_lease(source) for source in sources},
    )
    replay = provider.observe_evidence(
        evidence=evidence,
        source_leases={source: _source_lease(source) for source in sources},
    )

    assert first is not None
    assert replay is not None
    assert first.independent_view_count.tolist() == [2]
    assert replay.independent_view_count.tolist() == [2]


def test_occluded_b4_depth_produces_an_unqualified_target_without_a_rerender_or_upload():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
        reduce_native_depth_surface_evidence,
    )

    token = _token()
    sources = (_source(1), _source(2))
    geometry = tuple(
        GeometryRenderLease(
            source_identity=source,
            snapshot_token=token,
            resolution_tier="native",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=False,
                geometry_population_target=True,
            ),
            alpha=torch.ones(4, 5, 1),
            # The map surface is already depth-consistent with the sensor, so
            # it is not a new population target.
            rendered_depth=torch.ones(4, 5, 1),
        )
        for source in sources
    )
    provider = ProductionSurfaceTargetProvider(
        reducer=reduce_native_depth_surface_evidence,
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=_evidence(*geometry),
        source_leases={source: _source_lease(source) for source in sources},
    )

    assert targets is not None
    assert targets.independent_view_count.tolist() == [0]
    assert targets.support_evidence.tolist() == [0.0]
    assert provider.last_observation.source_h2d_ops == 0
    assert provider.last_observation.extra_rasterizer_calls == 0


def test_partial_or_duplicate_b4_geometry_leases_fail_closed_before_any_reduction():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    token = _token()
    sources = (_source(1), _source(2))
    complete = tuple(_geometry(source, token=token) for source in sources)
    resident = {source: _source_lease(source) for source in sources}
    calls = []
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: calls.append("reduced") or _reduced(101),
        binder=_binder,
        max_targets=2,
    )
    partial = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="b4-partial",
        source_identities=sources,
        modalities=tuple(
            SupervisionModality(geometry_evidence_enabled=True) for _ in sources
        ),
        modality_eligibility=tuple(lease.modality_eligibility for lease in complete),
        geometry_render_leases=(complete[0],),
    )
    duplicate = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id="b4-duplicate",
        source_identities=sources,
        modalities=tuple(
            SupervisionModality(geometry_evidence_enabled=True) for _ in sources
        ),
        modality_eligibility=tuple(lease.modality_eligibility for lease in complete),
        geometry_render_leases=(complete[0], complete[0]),
    )

    assert provider.observe_evidence(evidence=partial, source_leases=resident) is None
    assert provider.observe_evidence(evidence=duplicate, source_leases=resident) is None
    assert calls == []


def test_more_than_b4_geometry_leases_fail_closed_before_any_reduction():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    token = _token()
    sources = tuple(_source(index) for index in range(5))
    evidence = _evidence(*(_geometry(source, token=token) for source in sources))
    calls = []
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: calls.append("reduced") or _reduced(101),
        binder=_binder,
        max_targets=1,
    )

    targets = provider.observe_evidence(
        evidence=evidence,
        source_leases={source: _source_lease(source) for source in sources},
    )

    assert targets is None
    assert calls == []


def test_stale_or_non_native_evidence_clears_cached_target_and_fails_closed():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    sources = (_source(1), _source(2))
    token = _token()
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101), binder=_binder, max_targets=1,
    )
    assert provider.observe_evidence(
        evidence=_evidence(*(_geometry(source, token=token) for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    ) is not None

    assert provider(
        _create_decision(1), token, _token(map_content=99), MutationWorkToken(7, 1)
    ) is None
    assert provider.observe_evidence(
        evidence=_evidence(*(_geometry(source, token=token, tier="half") for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    ) is None
    assert provider(_create_decision(1), token, token, MutationWorkToken(7, 1)) is None


def test_target_binding_rejects_a_later_optimizer_epoch_even_if_surface_versions_match():
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    sources = (_source(1), _source(2))
    token = _token()
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101), binder=_binder, max_targets=1,
    )
    assert provider.observe_evidence(
        evidence=_evidence(*(_geometry(source, token=token) for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    ) is not None
    later = EvidenceSnapshotToken(
        map_content=token.map_content,
        topology=token.topology,
        coordinate_frame=token.coordinate_frame,
        schema=token.schema,
        optimizer_update=token.optimizer_update + 2,
        surface_evidence=token.surface_evidence,
        policy=token.policy,
        parameter=token.parameter + 2,
    )

    assert provider(_create_decision(1), token, later, MutationWorkToken(8, 1)) is None


@pytest.mark.parametrize("map_advance", (0, 1))
def test_target_binding_rebinds_the_exact_e0_origin_to_its_post_adam_action_fence(
    map_advance: int,
):
    """Catch implicit pre-/post-Adam snapshot matching at live activation."""

    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    origin = _token(map_content=11, parameter=11)
    action_fence = EvidenceSnapshotToken(
        map_content=origin.map_content + map_advance,
        topology=origin.topology,
        coordinate_frame=origin.coordinate_frame,
        schema=origin.schema,
        optimizer_update=origin.optimizer_update,
        surface_evidence=origin.surface_evidence,
        policy=origin.policy,
        parameter=origin.parameter + 1,
    )
    sources = (_source(1), _source(2))
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101), binder=_binder, max_targets=1,
    )
    targets = provider.observe_evidence(
        evidence=_evidence(*(_geometry(source, token=origin) for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    )

    assert targets is not None
    target_ids = targets.target_ids
    bound = provider(
        _create_decision(1), origin, action_fence, MutationWorkToken(7, 1)
    )

    assert bound is not None
    _decision, bound_targets = bound
    assert bound_targets is not targets
    assert bound_targets.snapshot == ProductionSurfaceTargetProvider._surface_snapshot(action_fence)
    assert bound_targets.positions is targets.positions
    assert bound_targets.normals is targets.normals
    assert bound_targets.uncertainty is targets.uncertainty
    assert bound_targets.target_ids is target_ids
    assert provider.last_observation.source_h2d_ops == 0
    assert provider.last_observation.extra_rasterizer_calls == 0


@pytest.mark.parametrize(
    "drift",
    (
        {"map_content": 2},
        {"topology": 1},
        {"coordinate_frame": 1},
        {"schema": 1},
        {"surface_evidence": 1},
        {"policy": 1},
        {"optimizer_update": 1},
        {"parameter": 0},
        {"parameter": 2},
    ),
)
def test_target_binding_rejects_any_transition_other_than_one_post_adam_step(drift):
    from eligsir.mapping.evidence.production_surface_targets import (
        ProductionSurfaceTargetProvider,
    )

    origin = _token(map_content=11, parameter=11)
    sources = (_source(1), _source(2))
    provider = ProductionSurfaceTargetProvider(
        reducer=lambda *_args: _reduced(101), binder=_binder, max_targets=1,
    )
    assert provider.observe_evidence(
        evidence=_evidence(*(_geometry(source, token=origin) for source in sources)),
        source_leases={source: _source_lease(source) for source in sources},
    ) is not None
    values = dict(
        map_content=origin.map_content + 1,
        topology=origin.topology,
        coordinate_frame=origin.coordinate_frame,
        schema=origin.schema,
        optimizer_update=origin.optimizer_update,
        surface_evidence=origin.surface_evidence,
        policy=origin.policy,
        parameter=origin.parameter + 1,
    )
    for axis, offset in drift.items():
        values[axis] = getattr(origin, axis) + offset

    assert provider(
        _create_decision(1), origin, EvidenceSnapshotToken(**values), MutationWorkToken(8, 1)
    ) is None


def test_production_observer_passes_exact_result_evidence_and_resident_leases_to_surface_provider():
    runtime = object.__new__(CanonicalMappingRuntime)
    source = _source(1)
    evidence = _evidence(_geometry(source, token=_token()))
    calls = []

    class Provider:
        def observe_evidence(self, *, evidence, source_leases):
            calls.append((evidence, source_leases))
            return None

    runtime._population_live_surface_target_provider = Provider()
    result = SimpleNamespace(evidence=evidence)
    leases = {source: _source_lease(source)}

    assert runtime._observe_population_surface_targets_after_step(result, leases) is None
    assert calls == [(evidence, leases)]


def test_production_source_lease_builder_only_borrows_geometry_eligible_native_sources():
    runtime = object.__new__(CanonicalMappingRuntime)
    source = _source(1)
    token = _token()
    evidence = _evidence(_geometry(source, token=token))
    camera = object()
    resident = _source_lease(source)
    calls = []

    runtime._native_prepared_source_lease = lambda value, identity, eligibility: (
        calls.append((value, identity, eligibility)) or resident
    )

    leases = runtime._population_live_surface_source_leases(
        SimpleNamespace(evidence=evidence), {source: camera}
    )

    assert leases == {source: resident}
    assert calls == [(camera, source, evidence.geometry_render_leases[0].modality_eligibility)]


def test_live_adapter_registration_keeps_observing_provider_as_the_activation_binder():
    runtime = object.__new__(CanonicalMappingRuntime)

    class Provider:
        def __call__(self, *_args):
            return None

        def observe_evidence(self, **_kwargs):
            return None

    provider = Provider()
    runtime.set_population_live_adapters(
        target_binding_provider=provider,
        context_provider=lambda *_args: None,
    )

    assert runtime._population_live_target_binding_provider is provider
    assert runtime._population_live_surface_target_provider is provider
