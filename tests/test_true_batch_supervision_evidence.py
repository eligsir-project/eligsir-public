"""True-camera batch supervision-evidence handoff contracts."""

from __future__ import annotations

from dataclasses import replace
import typing
from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RandomStateToken,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.supervision_evidence import (
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.optimization.gaussian.true_batch_executor import (
    run_true_camera_batched_step,
)


class _Camera:
    image_width = 2
    image_height = 2

    def __init__(self, uid: int):
        self.uid = uid


class _Model:
    def __init__(self):
        self.get_xyz = torch.zeros((3, 3))
        self.provisional = torch.zeros(3, dtype=torch.bool)


class _Session:
    parameter_version = 7
    coordinate_frame_version = 2
    schema_version = 3

    def __init__(self, model):
        self.model = model

    def update_learning_rate(self, _iteration):
        return 0.01


class _Engine:
    def __init__(self, model):
        self.session = _Session(model)
        self._optimizer_step_calls = 0

    def run_post_backward_step(self, *_args, loss_breakdown, render_outputs, **_kwargs):
        self._optimizer_step_calls += 1
        self.session.parameter_version += 1
        if hasattr(self.session, "coordinate_frame_version"):
            self.session.coordinate_frame_version += 1
        if hasattr(self.session, "schema_version"):
            self.session.schema_version += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=11,
            map_version_after=11,
            topology_version=13,
        )


def _request(
    camera: _Camera, *, evidence: bool, modality: SupervisionModality | None = None,
    eligibility: ModalityEligibility | None = None,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            source_identity=SourceIdentity("test", 0, camera.uid, camera.uid),
            resolution_bucket="native",
            supervision_modality=modality,
            modality_eligibility=eligibility,
        ),
        iteration=17,
        map_version=11,
        topology_version=13,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(
            want_supervision_evidence=evidence,
        ),
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
            iteration=17,
            n_frames_ingested=2,
            optimizer_update_id=19,
        ),
        n_frames_ingested=2,
        optimizer_update_id=19,
        optimizer_update_permitted=True,
        surface_evidence_version=23,
        policy_version=29,
        evidence_transaction_id=31,
    )


def _patch_true_batch(monkeypatch):
    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls = []
    render_depths = []
    monkeypatch.setattr(executor, "_e0_test_render_depths", render_depths, raising=False)
    monkeypatch.setattr(
        executor, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    def _render(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        render_depths.append(_kwargs["render_depth"])
        return TrueCameraBatchResult(
            renders=torch.zeros((2, 3, 2, 2)),
            alphas=torch.ones((2, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 0, 1]),
                "gaussian_ids": torch.tensor([0, 2, 1]),
            },
            camera_count=2,
            sparse_grad=True,
        )

    def _loss(inputs):
        loss = torch.tensor(float(inputs.viewpoint_cam.uid + 1), requires_grad=True)
        return LossBreakdown(
            total=loss.detach(),
            total_tensor=loss,
            components={"Ll1": loss.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    return render_calls


def test_true_batch_returns_device_resident_observational_evidence(monkeypatch):
    """Removing the handoff must lose the same-render batch facts."""

    render_calls = _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)

    result = run_true_camera_batched_step(
        engine,
        (_request(_Camera(0), evidence=True), _request(_Camera(1), evidence=True)),
        _runtime(engine, model),
    )

    assert render_calls == [(0, 1)]
    assert isinstance(result.evidence, SupervisionEvidenceBatch)
    assert result.evidence.snapshot_token.map_content == 11
    assert result.evidence.snapshot_token.topology == 13
    assert result.evidence.snapshot_token.coordinate_frame == 2
    assert result.evidence.snapshot_token.schema == 3
    assert result.evidence.snapshot_token.optimizer_update == 19
    assert result.evidence.snapshot_token.surface_evidence == 23
    assert result.evidence.snapshot_token.policy == 29
    assert result.evidence.snapshot_token.parameter == 7
    assert result.evidence.transaction_id == 31
    assert result.evidence.source_identities == (
        SourceIdentity("test", 0, 0, 0),
        SourceIdentity("test", 0, 1, 1),
    )
    assert torch.equal(result.evidence.rgb_residual_summary, torch.tensor([1.0, 2.0]))
    assert torch.equal(result.evidence.rgb_valid_pixels, torch.tensor([4, 4]))
    assert result.evidence.view_incidence is None
    assert torch.equal(result.evidence.depth_residual_summary, torch.tensor([0.0, 0.0]))
    assert torch.equal(result.evidence.depth_valid_pixels, torch.tensor([0, 0]))
    assert result.evidence.alpha_support_summary is None
    assert result.evidence.rgb_residual_summary.device == model.get_xyz.device


@pytest.mark.parametrize("camera_microbatch_size, expected_renders", [(4, [(0, 1, 2, 3)]), (2, [(0, 1), (2, 3)])])
def test_true_batch_e0_records_accepted_per_view_rgb_and_depth_terms(
    monkeypatch, camera_microbatch_size, expected_renders
):
    """Removing E0's normal-objective reduction loses per-view frame values."""

    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls = []
    monkeypatch.setattr(
        executor, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    def _render(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        count = len(cameras)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=torch.ones((count, 1, 2, 2)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.arange(count),
                "gaussian_ids": torch.arange(count),
            },
            camera_count=count,
            sparse_grad=True,
        )

    def _loss(inputs):
        uid = inputs.viewpoint_cam.uid
        total = inputs.gaussians.get_xyz.sum() + float(uid + 1)
        return LossBreakdown(
            total=total.detach(),
            total_tensor=total,
            components={
                "Ll1": torch.tensor(float(uid + 1)),
                "depth": torch.tensor(float(10 + uid)),
            },
            depth_valid_sample_count=3 + uid,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    model = _Model()
    model.get_xyz = torch.nn.Parameter(torch.zeros((4, 3)))
    model.provisional = torch.zeros(4, dtype=torch.bool)
    engine = _Engine(model)
    depth = SupervisionModality(depth_loss_enabled=True, depth_evidence_enabled=True)
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)

    result = run_true_camera_batched_step(
        engine,
        tuple(
            _request(_Camera(index), evidence=True, modality=depth, eligibility=eligibility)
            for index in range(4)
        ),
        _runtime(engine, model),
        camera_microbatch_size=camera_microbatch_size,
    )

    assert render_calls == expected_renders
    assert result.optimizer_step_count == 1
    assert result.evidence.source_identities == tuple(
        SourceIdentity("test", 0, index, index) for index in range(4)
    )
    assert torch.equal(result.evidence.rgb_residual_summary, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(result.evidence.depth_residual_summary, torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert torch.equal(result.evidence.rgb_valid_pixels, torch.tensor([4, 4, 4, 4]))
    assert torch.equal(result.evidence.depth_valid_pixels, torch.tensor([3, 4, 5, 6]))


def test_true_batch_e0_is_absent_when_no_optimizer_update_is_accepted(monkeypatch):
    """A rejected optimizer update must not publish its rendered rows as evidence."""

    from eligsir.optimization.gaussian import true_batch_executor as executor

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    runtime = _runtime(engine, model)
    runtime.optimizer_update_permitted = False

    result = run_true_camera_batched_step(
        engine,
        (_request(_Camera(0), evidence=True), _request(_Camera(1), evidence=True)),
        runtime,
    )

    assert result.optimizer_step_count == 0
    assert result.evidence is None


def test_true_batch_e0_zeroes_a_rejected_secondary_row_without_losing_accepted_rows(monkeypatch):
    """A rejected secondary must not erase its peers' accepted E0 measurements."""

    from eligsir.optimization.gaussian import true_batch_executor as executor

    _patch_true_batch(monkeypatch)

    def _loss(inputs):
        uid = inputs.viewpoint_cam.uid
        total = torch.tensor(float(uid + 1), requires_grad=True)
        return LossBreakdown(
            total=total.detach(),
            total_tensor=total,
            components={
                "Ll1": torch.tensor(float(uid + 1)),
                "depth": torch.tensor(float(10 + uid)),
            },
            loss_finite=uid != 1,
            depth_valid_sample_count=5 + uid,
        )

    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    model = _Model()
    engine = _Engine(model)
    result = run_true_camera_batched_step(
        engine,
        tuple(
            _request(
                _Camera(index),
                evidence=True,
                modality=SupervisionModality(
                    rgb_loss_enabled=True,
                    rgb_evidence_enabled=True,
                    depth_loss_enabled=True,
                    depth_evidence_enabled=True,
                ),
                eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
            )
            for index in range(2)
        ),
        _runtime(engine, model),
    )

    assert result.optimizer_step_count == 1
    assert result.supervision_view_count == 1
    assert torch.equal(result.evidence.rgb_residual_summary, torch.tensor([1.0, 0.0]))
    assert torch.equal(result.evidence.rgb_valid_pixels, torch.tensor([4, 0]))
    assert torch.equal(result.evidence.depth_residual_summary, torch.tensor([10.0, 0.0]))
    assert torch.equal(result.evidence.depth_valid_pixels, torch.tensor([5, 0]))


def test_true_b4_emits_same_render_full_row_attribution_and_gradient_utility(monkeypatch):
    """B4 E0 reuses packed render attribution and the normal backward gradient."""

    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls = []
    monkeypatch.setattr(
        executor, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    def _render(cameras, *_args, **_kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 2, 2)),
            alphas=torch.ones((4, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 0, 1, 2, 2, 3]),
                "gaussian_ids": torch.tensor([0, 2, 1, 2, 1, 2]),
            },
            camera_count=4,
            sparse_grad=True,
        )

    def _loss(inputs):
        # This is the ordinary backward source for the test; no evidence path
        # injects a synthetic utility tensor.
        loss = inputs.gaussians.get_xyz.sum()
        return LossBreakdown(
            total=loss.detach(),
            total_tensor=loss,
            components={"Ll1": loss.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    model = _Model()
    model.get_xyz = torch.nn.Parameter(torch.zeros((3, 3)))
    engine = _Engine(model)
    result = run_true_camera_batched_step(
        engine,
        tuple(_request(_Camera(index), evidence=True) for index in range(4)),
        _runtime(engine, model),
    )

    assert render_calls == [(0, 1, 2, 3)]
    assert isinstance(result.evidence, SupervisionEvidenceBatch)
    assert torch.equal(
        result.evidence.view_incidence,
        torch.tensor(
            [[True, False, False, False], [False, True, True, False], [True, False, True, True]]
        ),
    )
    assert torch.equal(
        result.evidence.meaningful_measurement_opportunity,
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        result.evidence.render_contribution,
        torch.tensor([0.25, 0.5, 0.75]),
    )
    assert torch.allclose(
        result.evidence.gradient_utility_evidence,
        torch.full((3,), 3.0**0.5),
    )


def test_true_batch_returns_no_evidence_only_when_collection_is_disabled(monkeypatch):
    """Disabled collection remains the explicit no-allocation path."""

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)

    result = run_true_camera_batched_step(
        engine,
        (_request(_Camera(0), evidence=False), _request(_Camera(1), evidence=False)),
        _runtime(engine, model),
    )

    assert result.evidence is None


def test_true_batch_keeps_ordinary_evidence_when_source_inventory_is_absent_or_duplicate(monkeypatch):
    """M6 abstains, but an older non-geometry consumer still receives E0."""

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    first = _request(_Camera(0), evidence=True)
    second = _request(_Camera(1), evidence=True)
    duplicate = first.view.source_identity

    for source_identity in (None, duplicate):
        requests = (
            first,
            replace(second, view=replace(second.view, source_identity=source_identity)),
        )
        result = run_true_camera_batched_step(engine, requests, _runtime(engine, model))

        assert isinstance(result.evidence, SupervisionEvidenceBatch)
        assert result.evidence.source_identities == ()


def test_true_batch_marks_evidence_unavailable_without_an_exact_snapshot(monkeypatch):
    """A missing surface/policy version must not be silently invented as zero."""

    render_calls = _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    runtime = _runtime(engine, model)
    runtime.surface_evidence_version = None
    runtime.policy_version = None

    result = run_true_camera_batched_step(
        engine,
        (_request(_Camera(0), evidence=True), _request(_Camera(1), evidence=True)),
        runtime,
    )

    assert result.evidence is None
    assert render_calls == [(0, 1)]


def test_optimization_result_type_hints_resolve_supervision_evidence():
    assert typing.get_type_hints(OptimizationStepResult)["evidence"]


def test_true_batch_uses_one_depth_render_but_keeps_per_view_depth_intent(monkeypatch):
    """The RGB-only lane must not acquire depth loss merely because its peer needs ED."""

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    from eligsir.optimization.gaussian import true_batch_executor as executor

    depth_loss_flags = []
    original_loss = executor._objective.assemble_loss

    def _loss(inputs):
        depth_loss_flags.append(inputs.use_depth_loss)
        return original_loss(inputs)

    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    result = run_true_camera_batched_step(
        engine,
        (
            _request(_Camera(0), evidence=True, modality=SupervisionModality()),
            _request(
                _Camera(1),
                evidence=True,
                modality=SupervisionModality(depth_loss_enabled=True),
            ),
        ),
        _runtime(engine, model),
    )

    assert executor._e0_test_render_depths == [True]
    assert depth_loss_flags == [False, True]
    assert result.evidence.modalities[0].depth_loss_enabled is False
    assert result.evidence.modalities[1].depth_loss_enabled is True


@pytest.mark.parametrize("secondary_rgb_loss", [False, True])
def test_true_batch_gates_rgb_evidence_independently_of_rgb_loss(monkeypatch, secondary_rgb_loss):
    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    from eligsir.optimization.gaussian import true_batch_executor as executor

    rgb_loss_flags = []
    original_loss = executor._objective.assemble_loss

    def _loss(inputs):
        rgb_loss_flags.append(inputs.rgb_loss_enabled)
        return original_loss(inputs)

    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    result = run_true_camera_batched_step(
        engine,
        (
            _request(_Camera(0), evidence=True, modality=SupervisionModality()),
            _request(
                _Camera(1),
                evidence=True,
                modality=SupervisionModality(
                    rgb_loss_enabled=secondary_rgb_loss,
                    depth_loss_enabled=True,
                    rgb_evidence_enabled=False,
                    depth_evidence_enabled=True,
                ),
                eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
            ),
        ),
        _runtime(engine, model),
    )

    assert rgb_loss_flags == [True, secondary_rgb_loss]
    assert result.optimizer_step_count == 1
    assert result.supervision_view_count == 2
    assert torch.equal(
        result.evidence.rgb_residual_summary,
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(result.evidence.rgb_valid_pixels, torch.tensor([4, 0]))
    assert result.evidence.modalities[1].rgb_loss_enabled is secondary_rgb_loss
    assert result.evidence.modalities[1].rgb_evidence_enabled is False


def test_true_batch_preserves_scheduler_eligibility_and_gates_geometry_evidence(monkeypatch):
    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    eligibility = ModalityEligibility(
        rgb_supervision=True,
        rgb_quality="qualified",
        depth_supervision=True,
        depth_quality="qualified",
        depth_insertion=False,
    )
    result = run_true_camera_batched_step(
        engine,
        (
            _request(_Camera(0), evidence=True),
            _request(
                _Camera(1),
                evidence=True,
                modality=SupervisionModality(
                    depth_loss_enabled=True, depth_evidence_enabled=True
                ),
                eligibility=eligibility,
            ),
        ),
        _runtime(engine, model),
    )

    assert result.evidence.modality_eligibility[1] == eligibility
    assert result.evidence.modalities[1].depth_loss_enabled is True
    assert result.evidence.modalities[1].depth_evidence_enabled is True


def test_true_batch_uses_runtime_coordinate_and_schema_snapshot_versions(monkeypatch):
    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    monkeypatch.delattr(_Session, "coordinate_frame_version")
    monkeypatch.delattr(_Session, "schema_version")
    runtime = _runtime(engine, model)
    runtime.coordinate_frame_version = 41
    runtime.schema_version = 43

    result = run_true_camera_batched_step(
        engine,
        (_request(_Camera(0), evidence=True), _request(_Camera(1), evidence=True)),
        runtime,
    )

    assert result.evidence.snapshot_token.coordinate_frame == 41
    assert result.evidence.snapshot_token.schema == 43


def test_geometry_eligibility_requests_depth_render_without_depth_loss(monkeypatch):
    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    from eligsir.optimization.gaussian import true_batch_executor as executor

    result = run_true_camera_batched_step(
        engine,
        (
            _request(_Camera(0), evidence=True),
            _request(
                _Camera(1),
                evidence=True,
                modality=SupervisionModality(
                    depth_loss_enabled=False, depth_evidence_enabled=False
                ),
                eligibility=ModalityEligibility(
                    depth_insertion=True,
                    geometry_population_target=True,
                ),
            ),
        ),
        _runtime(engine, model),
    )

    assert executor._e0_test_render_depths == [True]
    assert result.evidence.modalities[1].depth_loss_enabled is False
    assert result.evidence.modalities[1].geometry_evidence_enabled is True


def test_geometry_render_reuse_lane_keeps_each_eligible_b4_view_not_primary_hook(monkeypatch):
    """The insertion bridge gets borrowed ED for both eligible cameras.

    ``post_render_hook`` remains a primary-only legacy hook; this regression
    proves it cannot be the source of the B4 geometry lease lane.
    """

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    from eligsir.optimization.gaussian import true_batch_executor as executor

    def _render(cameras, *_args, **_kwargs):
        assert _kwargs["render_depth"] is True
        return TrueCameraBatchResult(
            renders=torch.zeros((2, 3, 2, 2)),
            alphas=torch.stack((torch.ones((1, 2, 2)), torch.full((1, 2, 2), 2.0))),
            rendered_depth=torch.stack((torch.ones((1, 2, 2)), torch.full((1, 2, 2), 3.0))),
            rendered_depth_ed=None,
            meta={"camera_ids": torch.tensor([0, 1]), "gaussian_ids": torch.tensor([0, 1])},
            camera_count=2,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    hook_calls = []
    runtime = _runtime(engine, model)
    runtime.post_render_hook = lambda _outputs: hook_calls.append("primary")
    eligible = ModalityEligibility(
        depth_insertion=True,
        geometry_population_target=True,
    )
    result = run_true_camera_batched_step(
        engine,
        (
            _request(_Camera(0), evidence=True, eligibility=eligible),
            _request(_Camera(1), evidence=True, eligibility=eligible),
        ),
        runtime,
    )

    assert hook_calls == ["primary"]
    leases = result.evidence.geometry_render_leases
    assert [lease.source_identity.source_ordinal for lease in leases] == [0, 1]
    assert len(leases) == 2
    assert leases[0].rendered_depth.data_ptr() != leases[1].rendered_depth.data_ptr()


def test_population_target_only_b4_keeps_all_geometry_leases(monkeypatch):
    """Population target evidence has a lease lane independent of insertion."""

    _patch_true_batch(monkeypatch)
    model = _Model()
    engine = _Engine(model)
    from eligsir.optimization.gaussian import true_batch_executor as executor

    def _render(cameras, *_args, **_kwargs):
        count = len(cameras)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=torch.ones((count, 1, 2, 2)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.arange(count),
                "gaussian_ids": torch.arange(count) % 3,
            },
            camera_count=count,
            sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    target_only = ModalityEligibility(
        depth_supervision=True,
        depth_insertion=False,
        geometry_population_target=True,
    )

    result = run_true_camera_batched_step(
        engine,
        tuple(
            _request(_Camera(index), evidence=True, eligibility=target_only)
            for index in range(4)
        ),
        _runtime(engine, model),
    )

    assert [
        lease.source_identity.source_ordinal
        for lease in result.evidence.geometry_render_leases
    ] == [0, 1, 2, 3]


def test_depth_insertion_eligibility_emits_one_b4_lease_without_e0_proxies(monkeypatch):
    """READY insertion borrows its selected render without enabling population E0."""

    from eligsir.optimization.gaussian import true_batch_executor as executor

    render_calls = []
    monkeypatch.setattr(
        executor, "supports_true_camera_batch", lambda _pipe, _cameras: (True, "ok")
    )

    def _render(cameras, *_args, **kwargs):
        render_calls.append(tuple(camera.uid for camera in cameras))
        assert kwargs["render_depth"] is True
        return TrueCameraBatchResult(
            renders=torch.zeros((4, 3, 2, 2)),
            alphas=torch.ones((4, 1, 2, 2)),
            rendered_depth=torch.ones((4, 1, 2, 2)),
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.tensor([0, 1, 2, 3]),
                "gaussian_ids": torch.tensor([0, 1, 2, 0]),
            },
            camera_count=4,
            sparse_grad=True,
        )

    def _loss(inputs):
        loss = torch.tensor(1.0, requires_grad=True)
        return LossBreakdown(
            total=loss.detach(),
            total_tensor=loss,
            components={"Ll1": loss.detach()},
            loss_finite=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)
    model = _Model()
    engine = _Engine(model)
    source_insertion = ModalityEligibility(
        depth_supervision=True,
        depth_insertion=True,
        geometry_population_target=False,
    )

    def _source_request(index):
        return _request(
            _Camera(index),
            evidence=False,
            modality=SupervisionModality(
                depth_loss_enabled=True, depth_evidence_enabled=True
            ),
            eligibility=(
                source_insertion
                if index == 2
                else ModalityEligibility(depth_supervision=True)
            ),
        )

    result = run_true_camera_batched_step(
        engine,
        tuple(_source_request(index) for index in range(4)),
        _runtime(engine, model),
    )

    assert render_calls == [(0, 1, 2, 3)]
    assert [lease.source_identity.source_ordinal for lease in result.evidence.geometry_render_leases] == [2]
    assert result.evidence.rgb_residual_summary is None
    assert result.evidence.view_incidence is None
    assert result.evidence.meaningful_measurement_opportunity is None
    assert result.evidence.render_contribution is None
    assert result.evidence.gradient_utility_evidence is None
