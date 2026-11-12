"""CPU contract harness for B4 RGB+ED evidence transfer attribution.

This is deliberately not a GPU qualification: it exercises the production
true-camera executor and insertion-reuse boundaries with CPU tensors.  A
separate ROCm run must still establish actual DMA and synchronization behavior.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    CameraIntrinsics,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.contracts.control import (
    ControlLimits,
    ControlProposal,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
    ObservationDisposition,
    ResolutionTier,
    ResidencyIntent,
)
from eligsir.mapping.scheduler import WorkBudgetOwner
from eligsir.mapping.optimization_runtime import (
    StreamingOptimizationRuntime,
    run_streaming_optimization_transaction,
)
from eligsir.mapping.population_shadow_runtime import PopulationShadowVersions
from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RandomStateToken,
    RenderOutputs,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality
from eligsir.optimization.gaussian.true_batch_executor import run_true_camera_batched_step
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken


class _Camera:
    image_width = 5
    image_height = 4

    def __init__(self, uid: int) -> None:
        self.uid = uid


class _Model:
    def __init__(self) -> None:
        self.get_xyz = torch.zeros((3, 3))
        self.provisional = torch.zeros(3, dtype=torch.bool)


class _Session:
    parameter_version = 9
    coordinate_frame_version = 5
    schema_version = 6

    def __init__(self, model: _Model) -> None:
        self.model = model

    def update_learning_rate(self, _iteration: int) -> float:
        return 0.01


class _Engine:
    def __init__(self, model: _Model) -> None:
        self.session = _Session(model)
        self._optimizer_step_calls = 0

    def run_post_backward_step(self, *_args, loss_breakdown, render_outputs, **_kwargs):
        self._optimizer_step_calls += 1
        self.session.parameter_version += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=3,
            map_version_after=3,
            topology_version=4,
        )


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("b4-telemetry", 2, ordinal, 1_000_000 + ordinal)


def _request(
    camera: _Camera, *, geometry_population_target: bool = True
) -> OptimizationStepRequest:
    source = _source(camera.uid)
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            source_identity=source,
            resolution_bucket="native",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=True,
                geometry_population_target=geometry_population_target,
            ),
            supervision_modality=SupervisionModality(
                depth_loss_enabled=True,
                depth_evidence_enabled=True,
                geometry_evidence_enabled=geometry_population_target,
            ),
        ),
        iteration=7,
        map_version=3,
        topology_version=4,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(want_supervision_evidence=True),
        random_state_token=RandomStateToken.capture(),
    )


def _runtime(engine: _Engine, model: _Model) -> OptimizationRuntimeContext:
    return OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(
            streaming_objective_step_regularizers=False,
            streaming_objective_batched_pixel_terms=False,
        ),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        gradient_policy_context=GradientPolicyContext(
            gaussians=model,
            args=SimpleNamespace(
                streaming_objective_step_regularizers=False,
                streaming_objective_batched_pixel_terms=False,
            ),
            iteration=7,
            n_frames_ingested=4,
            optimizer_update_id=7,
        ),
        n_frames_ingested=4,
        optimizer_update_id=7,
        optimizer_update_permitted=True,
        surface_evidence_version=8,
        policy_version=9,
        evidence_transaction_id=7,
    )


def _buffer(value: torch.Tensor, *, encoding: str = "tensor") -> BufferRef:
    return BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding=encoding,
            device=str(value.device),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )


def _prepared_source(source: SourceIdentity) -> PreparedSourceLease:
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=5, height=4, depth_scale=1000.0
    )
    pose = tuple(tuple(float(value) for value in row) for row in torch.eye(4).tolist())
    camera_intrinsics = (
        (intrinsics.fx, 0.0, intrinsics.cx),
        (0.0, intrinsics.fy, intrinsics.cy),
        (0.0, 0.0, 1.0),
    )
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=1,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="test",
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True, depth_supervision=True, depth_insertion=True
        ),
    )
    metadata = {
        "source_id": source.source_id,
        "stream_epoch": source.stream_epoch,
        "source_ordinal": source.source_ordinal,
        "timestamp_ns": source.timestamp_ns,
        "rgb_shape": (4, 5, 3),
        "depth_shape": (4, 5),
        "depth_units": "m",
        "depth_aligned_to_rgb": True,
        "camera_intrinsics": tuple(getattr(intrinsics, field.name) for field in fields(intrinsics)),
        "pose_c2w": pose,
        "calibration_attestation": ("prepared-calibration-v1", pose, camera_intrinsics),
    }
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=True,
            geometry_population_target=True,
        ),
        source_metadata=metadata,
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "depth_meters": _buffer(torch.ones(4, 5), encoding="depth_meters"),
            "rgb": _buffer(torch.zeros(4, 5, 3), encoding="rgb_float32_hwc"),
            "c2w": _buffer(torch.eye(4), encoding="c2w"),
            "camera_intrinsics": _buffer(
                torch.tensor(camera_intrinsics), encoding="camera_intrinsics"
            ),
        },
        device="cpu",
        control_token=PreparedIngressControlToken(1, source, 1, disposition),
    )


def test_b4_evidence_reuses_prepared_sources_and_emits_zero_transfer_attribution(monkeypatch):
    """One B4 render supplies all RGB+ED evidence without a sync or re-render.

    The CPU fallback proves composition and attribution only.  It intentionally
    does not claim that a particular CUDA/HIP driver issued zero DMA commands
    or that it avoided host synchronization.
    """
    from eligsir.optimization.gaussian import true_batch_executor as executor
    from eligsir.runtime import production_application as app
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    render_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_args: (True, "ok"))

    def render_batch(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        assert _kwargs["render_depth"] is True
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 4, 5)),
            alphas=torch.ones((4, 1, 4, 5)),
            rendered_depth=torch.ones((4, 1, 4, 5)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 1, 2, 3]),
                "gaussian_ids": torch.tensor([0, 0, 1, 2]),
            },
            camera_count=4,
            sparse_grad=True,
        )

    def assemble_loss(inputs):
        loss = torch.tensor(float(inputs.viewpoint_cam.uid + 1), requires_grad=True)
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={"Ll1": loss}, loss_finite=True
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    patch_assemble_loss(monkeypatch, executor._objective, assemble_loss)
    model = _Model()
    engine = _Engine(model)
    cameras = tuple(_Camera(index) for index in range(4))
    result = run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), _runtime(engine, model)
    )

    assert render_calls == [(0, 1, 2, 3)]
    assert result.rasterizer_call_count == 1
    assert result.camera_batch_rasterizer_calls == 1
    assert result.evidence is not None
    assert result.evidence.rgb_residual_summary.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert [lease.source_identity for lease in result.evidence.geometry_render_leases] == [
        _source(index) for index in range(4)
    ]

    prepared = {_source(index): _prepared_source(_source(index)) for index in range(4)}
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    monkeypatch.setattr(
        runtime,
        "_native_prepared_source_lease",
        lambda _camera, source, _eligibility: prepared[source],
    )
    plan = SimpleNamespace(map_content_version=3, topology_version=4, parameter_version=9)
    reuses = [
        runtime._select_depth_insertion_render_reuse(
            result=result, camera=camera, source=_source(camera.uid), plan=plan
        )
        for camera in cameras
    ]
    assert [reason.value for _reuse, reason in reuses] == ["reused"] * 4
    assert all(reuse is not None for reuse, _reason in reuses)
    assert all(reuse.source_h2d_ops == 0 for reuse, _reason in reuses)
    assert all(reuse.extra_rasterizer_calls == 0 for reuse, _reason in reuses)

    def producer(*_args, **kwargs):
        assert kwargs["render_reuse"] is reuses[0][0]
        return None, {
            "insertion_render_reused": True,
                "source_lease_hit": True,
                "source_h2d_ops": 0,
                "source_d2h_ops": 0,
                "extra_rasterizer_calls": 0,
        }

    monkeypatch.setattr(app, "insert_gaussians_from_frame_device_native", producer)
    monkeypatch.setattr(
        app, "render", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rerender"))
    )
    runtime.scene = object()
    runtime.gaussians = object()
    runtime.pipe = object()
    runtime._background = object()
    runtime._depth_insertion_policy = SimpleNamespace(initial_opacity=0.1)
    runtime.frames_seen = 0
    runtime._transfer_owner = object()
    assert runtime._prepare_depth_insertion_proposal(
        cameras[0],
        SimpleNamespace(),
        _source(0),
        insertion_plan=object(),
        modality_eligibility=ModalityEligibility(depth_insertion=True),
        prev_frame=None,
        prev_depth_meters=None,
        insertion_policy=runtime._depth_insertion_policy,
        render_reuse=reuses[0][0],
    ) is None
    assert runtime.depth_insertion_render_reuse_telemetry() == {
        "insertion_render_reused": 1,
        "source_lease_hit": 1,
        "source_h2d_ops": 0,
        "source_d2h_ops": 0,
        "extra_rasterizer_calls": 0,
        "fallback": "reused",
    }


def test_depth_insertion_gets_its_existing_b4_render_without_population_geometry(monkeypatch):
    """Depth insertion owns its lease eligibility, not population geometry policy."""
    from eligsir.mapping.integration.depth_render_reuse import DepthInsertionReuseReason
    from eligsir.optimization.gaussian import true_batch_executor as executor
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    render_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_args: (True, "ok"))

    def render_batch(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 4, 5)),
            alphas=torch.ones((4, 1, 4, 5)),
            rendered_depth=torch.ones((4, 1, 4, 5)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 1, 2, 3]),
                "gaussian_ids": torch.tensor([0, 0, 1, 2]),
            },
            camera_count=4,
            sparse_grad=True,
        )

    def assemble_loss(_inputs):
        loss = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={"Ll1": loss}, loss_finite=True
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    patch_assemble_loss(monkeypatch, executor._objective, assemble_loss)
    model = _Model()
    engine = _Engine(model)
    cameras = tuple(_Camera(index) for index in range(4))
    result = run_true_camera_batched_step(
        engine,
        tuple(
            _request(camera, geometry_population_target=False) for camera in cameras
        ),
        _runtime(engine, model),
    )

    assert render_calls == [(0, 1, 2, 3)]
    assert result.rasterizer_call_count == 1
    assert result.evidence is not None
    assert [lease.source_identity for lease in result.evidence.geometry_render_leases] == [
        _source(index) for index in range(4)
    ]
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    source = _source(0)
    reuse, reason = runtime._select_depth_insertion_render_reuse(
        result=result,
        camera=cameras[0],
        source=source,
        source_lease=_prepared_source(source),
        plan=SimpleNamespace(map_content_version=3, topology_version=4, parameter_version=9),
    )
    assert reason is DepthInsertionReuseReason.REUSED
    assert reuse is not None and reuse.source_h2d_ops == reuse.extra_rasterizer_calls == 0
    stale, reason = runtime._select_depth_insertion_render_reuse(
        result=result,
        camera=cameras[0],
        source=source,
        source_lease=_prepared_source(source),
        plan=SimpleNamespace(map_content_version=3, topology_version=4, parameter_version=8),
    )
    assert stale is None and reason is DepthInsertionReuseReason.STALE_SNAPSHOT


def test_configured_b32_source_insertion_uses_actual_native_b4_true_camera_lease(monkeypatch):
    """A realized B4 uses one true-camera render even when the plan requested B32."""
    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls: list[tuple[int, ...]] = []
    rendered_batches: list[TrueCameraBatchResult] = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_args: (True, "ok"))

    def render_batch(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        batch = TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 4, 5)),
            alphas=torch.ones((4, 1, 4, 5)),
            rendered_depth=torch.ones((4, 1, 4, 5)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 1, 2, 3]),
                "gaussian_ids": torch.tensor([0, 0, 1, 2]),
            },
            camera_count=4,
            sparse_grad=True,
        )
        rendered_batches.append(batch)
        return batch

    def assemble_loss(_inputs):
        loss = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={"Ll1": loss}, loss_finite=True
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    patch_assemble_loss(monkeypatch, executor._objective, assemble_loss)
    model = _Model()
    engine = _Engine(model)
    engine.run_step = lambda *_args, **_kwargs: pytest.fail("sequential fallback")
    cameras = tuple(_Camera(index) for index in range(4))
    for camera in cameras:
        camera._source_identity = _source(camera.uid)

    runtime = StreamingOptimizationRuntime(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(
            streaming_objective_step_regularizers=False,
            streaming_objective_batched_pixel_terms=False,
        ),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=None,
        tb_writer=None,
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={},
        iter_since_new_frame=0,
        population_shadow_enabled=False,
        population_shadow_evidence_versions_provider=lambda: PopulationShadowVersions(
            map_content=3,
            topology=4,
            coordinate_frame=5,
            schema=6,
            optimizer_update=7,
            surface_evidence=8,
            policy=9,
            parameter=9,
        ),
    )
    result = run_streaming_optimization_transaction(
        runtime,
        engine,
        cameras=cameras,
        requested_batch_size=32,
        requested_camera_count=32,
        modality_eligibilities=tuple(
            ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=index == 0,
            )
            for index, _camera in enumerate(cameras)
        ),
        background=None,
        iteration=7,
        map_version=3,
        topology_version=4,
        resolution_bucket="native",
        update_sh_rest=False,
        render_depth=True,
        n_frames_ingested=4,
        visible_mask=None,
        random_state_token=RandomStateToken.capture(),
        optimizer_update_id=7,
        optimizer_update_permitted=True,
    )

    assert render_calls == [(0, 1, 2, 3)]
    assert result.batch_size == result.supervision_view_count == 4
    assert result.requested_batch_size == result.requested_camera_count == 32
    assert result.realized_batch_size == result.realized_camera_count == 4
    leases = result.evidence.geometry_render_leases
    assert [lease.source_identity for lease in leases] == [_source(0)]
    lease = leases[0]
    assert lease.resolution_tier == "native"
    assert lease.alpha.device.type == lease.rendered_depth.device.type == "cpu"
    assert lease.rendered_depth.shape == (1, 4, 5)
    assert lease.alpha.data_ptr() == rendered_batches[0].alphas[0].data_ptr()
    assert lease.rendered_depth.data_ptr() == rendered_batches[0].rendered_depth[0].data_ptr()
    assert (lease.snapshot_token.map_content, lease.snapshot_token.topology, lease.snapshot_token.parameter) == (
        3,
        4,
        9,
    )


@pytest.mark.parametrize(
    ("population_shadow_enabled", "depth_insertion_enabled", "optimizer_step_count"),
    [(False, True, 1), (True, True, 1), (False, False, 1), (False, True, 0)],
)
def test_b4_c1_production_engine_path_attaches_primary_native_lease(
    monkeypatch, population_shadow_enabled, depth_insertion_enabled, optimizer_step_count
):
    """The real B4/C1 engine branch reuses its primary render for insertion."""
    from eligsir.mapping.integration.depth_render_reuse import DepthInsertionReuseReason
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    model = _Model()
    engine = _Engine(model)
    cameras = tuple(_Camera(index) for index in range(4))
    for camera in cameras:
        camera._source_identity = _source(camera.uid)
    render_outputs = RenderOutputs(
        payload={
            "render": torch.zeros((1, 3, 4, 5)),
            "alpha": torch.ones((1, 4, 5)),
            "rendered_depth": torch.ones((1, 4, 5)),
        }
    )
    loss = torch.tensor(1.0)
    calls = []

    def run_batched_step(requests, context, *, skipped_view_count):
        calls.append((tuple(request.view.source_identity for request in requests), context))
        assert skipped_view_count == 0
        return OptimizationStepResult(
            loss_breakdown=LossBreakdown(
                total=loss,
                total_tensor=loss,
                components={"Ll1": loss, "depth": loss},
                loss_finite=True,
                depth_valid_sample_count=20,
            ),
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=3,
            map_version_after=3,
            topology_version=4,
            parameter_version_before=9,
            parameter_version_after=10,
            optimizer_step_count=optimizer_step_count,
            supervision_view_count=4,
            rasterizer_call_count=4,
        )

    engine.run_batched_step = run_batched_step
    evidence_version_calls = []

    def evidence_versions():
        evidence_version_calls.append(True)
        return PopulationShadowVersions(
            map_content=3,
            topology=4,
            coordinate_frame=5,
            schema=6,
            optimizer_update=7,
            surface_evidence=8,
            policy=9,
            parameter=9,
        )

    runtime = StreamingOptimizationRuntime(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=None,
        tb_writer=None,
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={},
        iter_since_new_frame=0,
        population_shadow_enabled=population_shadow_enabled,
        population_shadow_evidence_versions_provider=evidence_versions,
    )
    commit_calls = []
    original_commit = runtime.commit_step

    def commit_once(context, result):
        commit_calls.append(result)
        return original_commit(context, result)

    monkeypatch.setattr(runtime, "commit_step", commit_once)
    monkeypatch.setattr(
        "eligsir.optimization.gaussian.true_batch_executor.run_true_camera_batched_step",
        lambda *_args, **_kwargs: pytest.fail("B4/C1 must stay on engine.run_batched_step"),
    )
    result = run_streaming_optimization_transaction(
        runtime,
        engine,
        cameras=cameras,
        requested_batch_size=4,
        requested_camera_count=1,
        modality_eligibilities=tuple(
            ModalityEligibility(
                rgb_supervision=True,
                depth_supervision=True,
                depth_insertion=depth_insertion_enabled and index == 0,
                geometry_population_target=False,
            )
            for index in range(4)
        ),
        background=None,
        iteration=7,
        map_version=3,
        topology_version=4,
        resolution_bucket="native",
        update_sh_rest=False,
        render_depth=True,
        n_frames_ingested=4,
        visible_mask=None,
        random_state_token=RandomStateToken.capture(),
        optimizer_update_id=7,
        optimizer_update_permitted=True,
    )

    assert [source for source, _context in calls] == [tuple(_source(index) for index in range(4))]
    assert len(commit_calls) == 1
    assert result.optimizer_step_count == optimizer_step_count
    assert bool(evidence_version_calls) is depth_insertion_enabled
    if not depth_insertion_enabled or optimizer_step_count == 0:
        assert result.evidence is None
        return
    assert len(result.evidence.geometry_render_leases) == 1
    lease = result.evidence.geometry_render_leases[0]
    assert lease.source_identity == _source(0)
    assert lease.modality_eligibility.geometry_population_target is False
    assert result.evidence.modalities[0].geometry_evidence_enabled is False
    assert lease.alpha.data_ptr() == render_outputs.payload["alpha"].data_ptr()
    assert lease.rendered_depth.data_ptr() == render_outputs.payload["rendered_depth"].data_ptr()
    assert (
        lease.snapshot_token.map_content,
        lease.snapshot_token.topology,
        lease.snapshot_token.parameter,
    ) == (3, 4, 9)

    reuse, reason = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)._select_depth_insertion_render_reuse(
        result=result,
        camera=cameras[0],
        source=_source(0),
        source_lease=_prepared_source(_source(0)),
        plan=SimpleNamespace(map_content_version=3, topology_version=4, parameter_version=9),
    )
    assert reason is DepthInsertionReuseReason.REUSED
    assert reuse is not None
    assert reuse.source_h2d_ops == 0
    assert reuse.extra_rasterizer_calls == 0
    assert len(result.evidence.geometry_render_leases) < result.realized_batch_size


@pytest.mark.parametrize("batch_size", [16, 32])
def test_depth_insertion_lease_uses_ready_bn_render_without_b4_population_evidence(monkeypatch, batch_size):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls = []
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_args: (True, "ok"))
    monkeypatch.setattr(
        executor,
        "render_true_camera_batch",
        lambda cameras, *_args, **_kwargs: (
            render_calls.append(len(cameras))
            or TrueCameraBatchResult(
                renders=torch.zeros((batch_size, 3, 4, 5)),
                alphas=torch.ones((batch_size, 1, 4, 5)),
                rendered_depth=torch.ones((batch_size, 1, 4, 5)),
                rendered_depth_ed=None,
                meta={
                    "camera_ids": torch.arange(batch_size),
                    "gaussian_ids": torch.zeros(batch_size, dtype=torch.long),
                },
                camera_count=batch_size,
                sparse_grad=True,
            )
        ),
    )

    def assemble_loss(_inputs):
        loss = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={"Ll1": loss}, loss_finite=True
        )

    patch_assemble_loss(monkeypatch, executor._objective, assemble_loss)
    model = _Model()
    engine = _Engine(model)
    cameras = tuple(_Camera(index) for index in range(batch_size))
    result = run_true_camera_batched_step(
        engine,
        tuple(_request(camera, geometry_population_target=False) for camera in cameras),
        _runtime(engine, model),
    )

    assert render_calls == [batch_size]
    assert result.rasterizer_call_count == 1
    assert result.evidence is not None
    assert len(result.evidence.geometry_render_leases) == batch_size
    assert result.evidence.view_incidence is None
    assert result.evidence.meaningful_measurement_opportunity is None
    assert result.evidence.render_contribution is None
    assert result.evidence.gradient_utility_evidence is None


def test_actual_device_insertion_reuse_branch_skips_source_transfer_and_reports_d2h(monkeypatch):
    """The B4 insertion owner prepares only the first source per transaction.

    B4 produces four geometry leases, but the existing transaction tail chooses
    its primary source for one insertion proposal.  This CPU harness drives
    that *actual* producer reuse branch to its early empty-mask exit while
    making any legacy packed H2D call an immediate failure.
    """
    from eligsir.mapping.integration import device_integration_planner as planner
    from eligsir.mapping.integration.depth_render_reuse import DepthInsertionRenderReuse

    class _CpuCompatibleAccelerator:
        type = "cuda"

        def __eq__(self, value):
            return isinstance(value, torch.device) and value.type == "cpu"

    source = _source(0)
    prepared = _prepared_source(source)
    reuse = DepthInsertionRenderReuse(
        alpha=torch.ones(4, 5, 1),
        rendered_depth=torch.ones(4, 5, 1),
        depth_meters=prepared.payloads["depth_meters"].data,
        rgb=prepared.payloads["rgb"].data,
        c2w=prepared.payloads["c2w"].data,
    )
    monkeypatch.setattr(planner, "_validate_insertion_authority", lambda *_args: None)
    monkeypatch.setattr(planner, "_target_device", lambda *_args: _CpuCompatibleAccelerator())
    monkeypatch.setattr(
        planner,
        "_packed_source_to_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy full-frame H2D")),
    )
    monkeypatch.setattr(planner, "_intrinsics", lambda *_args: (1.0, 1.0, 0.0, 0.0))
    monkeypatch.setattr(
        planner, "_grid", lambda *_args: (torch.empty((0,), dtype=torch.long),) * 2
    )
    monkeypatch.setattr(
        planner,
        "_gpu_render_occupancy_masks",
        lambda *_args: (torch.empty((0,), dtype=torch.bool),) * 3
        + (torch.empty((0,), dtype=torch.float32),),
    )
    monkeypatch.setattr(
        "eligsir.utils.depth_sparsity_schedule.config_from_policy", lambda *_args: None
    )
    monkeypatch.setattr(
        "eligsir.utils.depth_sparsity_schedule.resolve_effective_depth_stride", lambda *_args: 1
    )
    policy = SimpleNamespace(
        debug_statistics=False,
        min_depth_m=0.1,
        max_depth_m=5.0,
        initial_opacity=0.1,
        stability_filter_enabled=False,
        temporal_check_enabled=False,
        edge_threshold=0.0,
        max_view_angle_deg=80.0,
        score_mode="alpha_only",
    )
    plan = SimpleNamespace(voxel_size_m=0.01, candidate_budget=10, sample_stride=1)

    candidates, stats = planner.insert_gaussians_from_frame_device_native(
        SimpleNamespace(),
        SimpleNamespace(),
        object(),
        render_alpha=reuse.alpha,
        render_depth=reuse.rendered_depth,
        insertion_plan=plan,
        insertion_policy=policy,
        modality_eligibility=ModalityEligibility(depth_insertion=True),
        render_reuse=reuse,
    )

    assert candidates is None
    assert stats["insertion_render_reused"] is True
    assert stats["source_lease_hit"] is True
    assert stats["source_h2d_ops"] == 0
    assert stats["source_d2h_ops"] == 0
    assert stats["candidate_d2h_ops"] == 0


@pytest.mark.parametrize("mutation_budget", [0, 1])
def test_scheduler_native_depth_plan_reaches_ready_camera_binding(monkeypatch, mutation_budget):
    """The real scheduler, adapter and camera binder agree on the exact tier.

    The tick is a capture stub; render reuse and insertion have their own
    tests above. This test does not claim device execution or COMMIT service.
    """
    source = _source(15)
    replay = tuple(_source(index) for index in range(15))
    modality = ModalityEligibility(
        rgb_supervision=True,
        depth_supervision=True,
        depth_insertion=True,
        geometry_population_target=False,
    )
    snapshot = ControlSnapshot(
        snapshot_id=17,
        control_version=3,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=3,
        topology_version=4,
        parameter_version=9,
        gaussian_count=100,
        optimizer_steps_completed=40,
        limits=ControlLimits(1, 15, 1, 5, 1, 1_000_000),
        supervision_candidates=(source,),
        replay_candidates=replay,
        replay_work_modalities=(modality,) * len(replay),
    )
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=snapshot.snapshot_id,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="depth insertion",
        modality_eligibility=modality,
    )
    proposals = tuple(
        ControlProposal("test", actuator, value, "READY depth")
        for actuator, value in (
            ("resolution_tier", "quarter"),
            ("supervision_slots", 1),
            ("replay_slots", 15),
            ("optimizer_updates", 1),
            ("insertion_enabled", True),
            ("insertion_seeds", 5),
            ("mutation_budget", mutation_budget),
        )
    )
    owner = WorkBudgetOwner(
        requested_batch_size=16,
        maximum_camera_count=16,
        constellation_policy="fixed",
        target_supervised_views_per_s=200.0,
        native_geometry_evidence_allowance_ms=0.0,
    )
    for tier, cost in (
        (ResolutionTier.QUARTER, 10.0),
        (ResolutionTier.HALF, 100.0),
        (ResolutionTier.NATIVE, 400.0),
    ):
        owner.record_completed_transaction(
            transaction_class=None,
            tier=tier,
            gaussian_count=100,
            realized_camera_count=16,
            prepare_ms=cost,
            optimization_ms=0.0,
            transaction_wall_ms=cost,
        )
    plan = owner.resolve(
        snapshot,
        disposition=disposition,
        proposals=proposals,
        source_depth_requires_native=True,
    )
    assert plan.resolution_tier is ResolutionTier.NATIVE
    assert plan.insertion.enabled and plan.insertion.source == source
    assert plan.mutation_budget.max_total == mutation_budget
    assert all(work.resolution_tier is ResolutionTier.NATIVE for work in plan.supervision_work)

    from eligsir.runtime import ready_b4_streaming_adapter as ready_adapter
    from eligsir.runtime.ready_b4_streaming_adapter import (
        ReadyB4StreamingOptimizationAdapter,
        ReadyB4StreamingStepInputs,
    )
    from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner

    expected = tuple(
        work.source for work in (*plan.supervision_work, *plan.replay_work)
    )
    ready_owner = ReadySourceOwner(max_fresh=16, max_replay=16, replay_reserve=0)
    producer = ready_owner.claim_producer("scheduler-native-depth")
    for item in plan.supervision_work:
        assert producer.offer(
            _prepared_source(item.source), lane=ReadySourceLane.FRESH
        )
    for item in plan.replay_work:
        assert producer.offer(
            _prepared_source(item.source), lane=ReadySourceLane.REPLAY
        )
    captured_transaction = []

    class _Bridge:
        def __init__(self):
            self.ready_owner = ready_owner

        def requeue_ready_only_sources(self, selection, *, ready_event, lane_by_source=None):
            return ready_owner.requeue_batch(
                selection.leases, ready_event=ready_event, lane_by_source=lane_by_source
            )

        def complete_retained_commit_first_supervision(self, completed_source):
            assert completed_source == source

    monkeypatch.setattr(ready_adapter, "PreparedRuntimeBridge", _Bridge)

    runtime_adapter = SimpleNamespace(
        _versions=lambda: (3, 4, 9),
        _advance_prepared_ready_plan=lambda _plan: None,
    )

    def _ready_tick(*, ready_transaction):
        captured_transaction.append(ready_transaction)
        return SimpleNamespace(
            executed=True,
            optimizer_steps=1,
            supervision_views=16,
            successful_supervision_sources=expected,
        )

    runtime_adapter.tick = _ready_tick
    adapter = ReadyB4StreamingOptimizationAdapter(
        _Bridge(),
        runtime=runtime_adapter,
        transfer_owner=SimpleNamespace(handoff_to_compute=lambda *_events: None),
        batch_size=16,
        source_insertion=True,
        warmup_complete=True,
    )
    ready_outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot, plan))
    assert ready_outcome.executed
    assert len(captured_transaction) == 1
    assert tuple(
        binding.prepared_tier for binding in captured_transaction[0].bindings
    ) == ("native",) * 16
    assert tuple(captured_transaction[0].cameras) == expected

    assert captured_transaction[0].source_insertion is True


def test_reuse_attribution_rejects_missing_or_negative_transfer_facts():
    """The emitted reuse telemetry remains a complete, bounded attribution."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    with pytest.raises(ValueError, match="incomplete"):
        runtime._record_depth_insertion_render_reuse({})
    with pytest.raises(ValueError, match="source_h2d_ops"):
        runtime._record_depth_insertion_render_reuse(
            {
                "insertion_render_reused": True,
                "source_lease_hit": True,
                "source_h2d_ops": -1,
                "source_d2h_ops": 0,
                "extra_rasterizer_calls": 0,
                "fallback": "reused",
            }
        )
    with pytest.raises(ValueError, match="source_d2h_ops"):
        runtime._record_depth_insertion_render_reuse(
            {
                "insertion_render_reused": True,
                "source_lease_hit": True,
                "source_h2d_ops": 0,
                "source_d2h_ops": -1,
                "extra_rasterizer_calls": 0,
                "fallback": "reused",
            }
        )
