"""CPU transaction coverage for train-frame display-referred EV/WB."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts import CameraIntrinsics, ModalityEligibility, SourceIdentity
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.mapping.resolution_cache import build_resolution_cache
from eligsir.mapping.tier_bind import bind_resolution_tier
from eligsir.optimization.gaussian import engine as gaussian_engine
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RandomStateToken,
)
from eligsir.optimization.gaussian.true_batch_executor import run_true_camera_batched_step
from eligsir.runtime.ready_camera_binder import PreparedCameraMetadata, ReadyCameraBinding
from eligsir.runtime.ready_only_tick import ReadySourceVersions


def _binding(ordinal: int, image: torch.Tensor) -> ReadyCameraBinding:
    source = SourceIdentity("evwb-transaction", 1, ordinal, 1_700_000_000 + ordinal)
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = float(ordinal)
    intrinsics = CameraIntrinsics(
        fx=4.0, fy=4.0, cx=1.5, cy=1.5, width=4, height=4, depth_scale=1.0,
    )
    camera = SimpleNamespace(
        original_image=image.clone(), image_height=4, image_width=4,
        width=4, height=4, fx=4.0, fy=4.0, cx=1.5, cy=1.5,
        c2w=pose.clone(), _source_identity=source,
    )
    token = SimpleNamespace(source=source)
    metadata = PreparedCameraMetadata(
        source_identity=source,
        intrinsics=intrinsics,
        pose_c2w=tuple(tuple(float(value) for value in row) for row in pose.tolist()),
        rgb_shape=(4, 4, 3), depth_shape=(4, 4), depth_units="m",
        depth_aligned_to_rgb=True,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        tier="native", control_token=token,
    )
    return ReadyCameraBinding(
        camera=camera, source_identity=source,
        modality_eligibility=metadata.modality_eligibility, control_token=token,
        versions=ReadySourceVersions(1, 1, 1), prepared_tier="native",
        metadata=metadata, lease=SimpleNamespace(),
    )


def _prepared_evwb_admission(db: FrameObservationDB, binding: ReadyCameraBinding, reference: str):
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._optimization_policy = SimpleNamespace(
        streaming_train_evwb_enabled=True,
        streaming_train_evwb_reference_source=reference,
    )
    runtime._register_historical_record = lambda *_args, **_kwargs: None
    runtime._record_ready_scene_candidate(binding.camera, binding, is_train=True, is_holdout=False)
    return db.get_frame(db.frame_id_for_source(binding.source_identity))


def _durable_record(db: FrameObservationDB, binding: ReadyCameraBinding):
    image = binding.camera.original_image
    rgb = image.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
    depth = np.ones((4, 4), dtype=np.float32)
    source = binding.source_identity
    record = db.record_historical_candidate(
        source_identity=source, pose_c2w=binding.metadata.pose_c2w,
        timestamp=source.timestamp_seconds, intrinsics=binding.metadata.intrinsics,
        cold_pixels=ColdPixelHandle(
            source.fingerprint(), rgb, depth, binding.metadata.intrinsics,
            rgb_encoding="rgb8", depth_encoding="depth_float32", depth_units="m",
        ),
        modality_eligibility=binding.modality_eligibility, disposition_snapshot_id=1,
    )
    record.resolution_cache = build_resolution_cache(
        rgb, depth, binding.metadata.intrinsics, low_res_divisor=2, device="cpu",
    )
    return record


def _cold_camera(record, image: torch.Tensor):
    return SimpleNamespace(
        original_image=image.clone(), image_height=4, image_width=4, width=4, height=4,
        fx=4.0, fy=4.0, cx=1.5, cy=1.5, _source_identity=record.source_identity,
    )


class _Session:
    parameter_version = 0

    def __init__(self, model):
        self.model = model

    def update_learning_rate(self, _iteration):
        return 0.1


class _SteppingEngine:
    """The executor owns grouping/backward/commit; this owns the real SGD step."""

    def __init__(self, model):
        self.session = _Session(model)
        self._optimizer_step_calls = 0

    def run_post_backward_step(self, _gaussians, _args, _pipe, _context, *, loss_breakdown, render_outputs, **_kwargs):
        self.session.model.optimizer.step()
        self.session.model.optimizer.zero_grad(set_to_none=True)
        self.session.parameter_version += 1
        self._optimizer_step_calls += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown, render_outputs=render_outputs, evidence=None,
            optimization_gpu_ms=None, map_version_before=0, map_version_after=1,
            topology_version=0,
        )


def test_cold_rebind_keeps_learned_evwb_state_through_real_ready_registration():
    """Would catch a READY callback that replaces durable COLD train state."""
    image = torch.full((3, 4, 4), 0.2)
    binding = _binding(20, image)
    db = FrameObservationDB(device="cpu")
    record = _durable_record(db, binding)
    reference = "another-source"
    assert _prepared_evwb_admission(db, binding, reference) is record
    state = record.photometric_evwb
    assert state is not None
    with torch.no_grad():
        state.copy_(torch.tensor([0.25, -0.10, 0.05]))

    cold = _cold_camera(record, image)
    assert bind_resolution_tier(cold, record, "low").ok
    assert bind_resolution_tier(cold, record, "high").ok
    assert _prepared_evwb_admission(db, binding, reference) is record

    assert record.photometric_evwb is state
    torch.testing.assert_close(state, torch.tensor([0.25, -0.10, 0.05]))


def _request(camera, source_index: int, *, rgb_loss_enabled: bool = True) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera, source_index=source_index, background=None,
            update_sh_rest=True, render_depth=False, resolution_bucket="native",
            # Match StreamingOptimizationRuntime.build_step: eligibility is
            # authoritative there and no legacy supervision modality is set.
            modality_eligibility=ModalityEligibility(rgb_supervision=rgb_loss_enabled),
        ),
        iteration=1, map_version=0, topology_version=0, resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(), random_state_token=RandomStateToken.capture(),
    )


@pytest.mark.parametrize(
    ("camera_count", "include_nontrain"),
    [(4, False), (2, True)],
    ids=["b_equals_c", "b_greater_than_c"],
)
def test_true_camera_transaction_updates_each_eligible_evwb_state_once_and_preserves_cold_binding(
    monkeypatch, camera_count, include_nontrain,
):
    """Would catch detached batched renders, per-microbatch EV/WB commits, or tier resets."""
    from test_objective_precomputed_pixel_terms_integration import _fixture
    from eligsir.optimization.gaussian import true_batch_executor as executor

    accepted_image = torch.full((3, 4, 4), 0.20)
    unsafe_image = torch.zeros((3, 4, 4))
    reference_image = torch.full((3, 4, 4), 0.35)
    db = FrameObservationDB(device="cpu")
    accepted_binding = _binding(10, accepted_image)
    unsafe_binding = _binding(11, unsafe_image)
    reference_binding = _binding(12, reference_image)
    reference = reference_binding.source_identity.fingerprint()
    accepted_record = _prepared_evwb_admission(db, accepted_binding, reference)
    unsafe_record = _prepared_evwb_admission(db, unsafe_binding, reference)
    reference_record = _prepared_evwb_admission(db, reference_binding, reference)
    assert accepted_record is not None and not accepted_record.historical_eligible
    assert unsafe_record is not None and not unsafe_record.historical_eligible
    assert reference_record is not None and not reference_record.historical_eligible
    assert accepted_record.photometric_evwb is not None
    assert unsafe_record.photometric_evwb is not None
    assert reference_record.photometric_evwb is None

    # Exercise the normal native tier bind after a detached camera reconstruction.
    # The READY record deliberately remains non-historical; this is the production
    # state that must nevertheless own its train-frame EV/WB value.
    ready_gt = accepted_binding.camera.original_image.clone()
    ready_pose = accepted_binding.camera.c2w.clone()
    cold_accepted = _cold_camera(accepted_record, accepted_image)
    assert bind_resolution_tier(cold_accepted, accepted_record, "high").ok
    assert accepted_record.photometric_evwb is not None
    torch.testing.assert_close(accepted_binding.camera.original_image, ready_gt)
    torch.testing.assert_close(accepted_binding.camera.c2w, ready_pose)
    np.testing.assert_array_equal(accepted_record.pose_c2w, np.asarray(accepted_binding.metadata.pose_c2w))

    rejected_binding = _binding(13, reference_image)
    holdout_binding = _binding(14, reference_image)
    rejected_record = db.record_prepared_candidate(
        source_identity=rejected_binding.source_identity, pose_c2w=rejected_binding.metadata.pose_c2w,
        timestamp=rejected_binding.source_identity.timestamp_seconds,
        intrinsics=rejected_binding.metadata.intrinsics, ready_source_reference=rejected_binding,
        modality_eligibility=rejected_binding.modality_eligibility, is_train=True,
    )
    holdout_record = db.record_prepared_candidate(
        source_identity=holdout_binding.source_identity, pose_c2w=holdout_binding.metadata.pose_c2w,
        timestamp=holdout_binding.source_identity.timestamp_seconds,
        intrinsics=holdout_binding.metadata.intrinsics, ready_source_reference=holdout_binding,
        modality_eligibility=holdout_binding.modality_eligibility, is_train=False,
    )
    # No READY train admission created a state for either source.  The executor
    # must never manufacture or update one merely because those views appear
    # beside an admitted train frame in the same true-camera transaction.
    assert rejected_record.photometric_evwb is None
    assert holdout_record.photometric_evwb is None

    args, opt, pipe, _camera, model, _render = _fixture()
    # The fixture's regularizer tensors are derived leaves.  The executor
    # legitimately backpropagates each microbatch separately, so use fresh
    # leaves here rather than retaining a fixture-only construction graph.
    model.get_scaling = model.get_scaling.detach().requires_grad_()
    model.get_opacity = model.get_opacity.detach().requires_grad_()
    args.opacity_reg = args.scale_reg = 0.0
    args.streaming_train_evwb_enabled = True
    args.streaming_train_evwb_reference_source = reference
    args.streaming_train_evwb_learning_rate = 0.05
    args.streaming_train_evwb_prior_weight = 0.0
    args.streaming_objective_batched_pixel_terms = True
    args.streaming_free_space_loss_weight = 0.0
    args.streaming_appearance_only_enabled = False
    args.supervision_tile_probe_grid = 0
    args.surface_confidence_depth_loss = False
    render = torch.nn.Parameter(torch.full((6, 3, 4, 4), 0.60))
    model.optimizer = torch.optim.SGD([render], lr=0.1)
    engine = _SteppingEngine(model)
    runtime = OptimizationRuntimeContext(
        session=engine.session, gaussians=model, args=args, opt=opt, pipe=pipe,
        n_frames_ingested=6, loss_assembly_kwargs={"streaming_scene": SimpleNamespace(_observation_db=db)},
        gradient_policy_context=GradientPolicyContext(model, args, 1, 6),
    )
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_: (True, "cpu fixture"))

    def render_batch(cameras, *_args, **_kwargs):
        indices = [camera._test_index for camera in cameras]
        return TrueCameraBatchResult(
            renders=render[indices], alphas=torch.ones(len(indices), 1, 4, 4),
            rendered_depth=None, rendered_depth_ed=None,
            meta={"radii": torch.ones(len(indices), model.get_xyz.shape[0])},
            camera_count=len(indices), sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    commits = []
    real_commit = gaussian_engine._commit_evwb_states

    def commit(states, commit_args):
        commits.append({key: value.grad.detach().clone() if value.grad is not None else None for key, value in states.items()})
        real_commit(states, commit_args)

    monkeypatch.setattr(executor, "_commit_evwb_states", commit)
    cameras = [cold_accepted, cold_accepted, reference_binding.camera, unsafe_binding.camera]
    if include_nontrain:
        cameras.extend((rejected_binding.camera, holdout_binding.camera))
    for index, camera in enumerate(cameras):
        camera._test_index = index
    before = accepted_record.photometric_evwb.detach().clone()
    result = run_true_camera_batched_step(
        engine, tuple(_request(camera, index) for index, camera in enumerate(cameras)), runtime,
        camera_microbatch_size=camera_count,
    )

    assert result.optimizer_step_count == 1
    assert result.backward_call_count == len(cameras) // camera_count
    assert not torch.equal(render.detach(), torch.full_like(render, 0.60))
    assert len(commits) == 1
    accepted_key = id(accepted_record.photometric_evwb)
    assert set(commits[0]) == {accepted_key, id(unsafe_record.photometric_evwb)}
    assert commits[0][accepted_key] is not None
    assert torch.count_nonzero(commits[0][accepted_key]) > 0
    assert torch.count_nonzero(commits[0][id(unsafe_record.photometric_evwb)]) == 0
    assert not torch.equal(accepted_record.photometric_evwb.detach(), before)
    torch.testing.assert_close(unsafe_record.photometric_evwb, torch.zeros(3))
    assert rejected_record.photometric_evwb is None
    assert holdout_record.photometric_evwb is None


@pytest.mark.parametrize(
    ("camera_microbatch_size", "repetitions"),
    [(2, 1), (2, 2)],
    ids=["b_equals_c", "b_greater_than_c"],
)
def test_accepted_no_rgb_transaction_excludes_evwb_prior_state_and_keeps_rgb_control(
    monkeypatch, camera_microbatch_size, repetitions,
):
    """Would catch an EV/WB prior moving a frame with RGB supervision disabled."""
    from test_objective_precomputed_pixel_terms_integration import _fixture
    from eligsir.optimization.gaussian import true_batch_executor as executor

    no_rgb_image = torch.full((3, 4, 4), 0.20)
    rgb_image = torch.full((3, 4, 4), 0.25)
    db = FrameObservationDB(device="cpu")
    no_rgb_binding = _binding(30, no_rgb_image)
    rgb_binding = _binding(31, rgb_image)
    reference = "another-source"
    no_rgb_record = _prepared_evwb_admission(db, no_rgb_binding, reference)
    rgb_record = _prepared_evwb_admission(db, rgb_binding, reference)
    assert no_rgb_record is not None and rgb_record is not None
    assert no_rgb_record.photometric_evwb is not None
    assert rgb_record.photometric_evwb is not None
    with torch.no_grad():
        no_rgb_record.photometric_evwb.copy_(torch.tensor([0.25, -0.10, 0.05]))
        rgb_record.photometric_evwb.copy_(torch.tensor([-0.20, 0.08, -0.04]))
    no_rgb_before = no_rgb_record.photometric_evwb.detach().clone()
    rgb_before = rgb_record.photometric_evwb.detach().clone()

    args, opt, pipe, _camera, model, _render = _fixture()
    model.get_scaling = model.get_scaling.detach().requires_grad_()
    model.get_opacity = model.get_opacity.detach().requires_grad_()
    args.opacity_reg = args.scale_reg = 0.0
    args.streaming_train_evwb_enabled = True
    args.streaming_train_evwb_reference_source = reference
    args.streaming_train_evwb_learning_rate = 0.05
    args.streaming_train_evwb_prior_weight = 0.25
    args.streaming_objective_batched_pixel_terms = True
    args.streaming_free_space_loss_weight = 0.0
    args.streaming_appearance_only_enabled = False
    args.supervision_tile_probe_grid = 0
    args.surface_confidence_depth_loss = False
    render = torch.nn.Parameter(torch.full((2 * repetitions, 3, 4, 4), 0.60))
    model.optimizer = torch.optim.SGD([render], lr=0.1)
    engine = _SteppingEngine(model)
    runtime = OptimizationRuntimeContext(
        session=engine.session, gaussians=model, args=args, opt=opt, pipe=pipe,
        n_frames_ingested=2 * repetitions,
        loss_assembly_kwargs={"streaming_scene": SimpleNamespace(_observation_db=db)},
        gradient_policy_context=GradientPolicyContext(model, args, 1, 2 * repetitions),
    )
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_: (True, "cpu fixture"))

    def render_batch(cameras, *_args, **_kwargs):
        indices = [camera._test_index for camera in cameras]
        return TrueCameraBatchResult(
            renders=render[indices], alphas=torch.ones(len(indices), 1, 4, 4),
            rendered_depth=None, rendered_depth_ed=None,
            meta={"radii": torch.ones(len(indices), model.get_xyz.shape[0])},
            camera_count=len(indices), sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    commits = []
    real_commit = gaussian_engine._commit_evwb_states

    def commit(states, commit_args):
        commits.append({key: value.grad.detach().clone() if value.grad is not None else None for key, value in states.items()})
        real_commit(states, commit_args)

    monkeypatch.setattr(executor, "_commit_evwb_states", commit)
    cameras = []
    rgb_enabled = []
    for _ in range(repetitions):
        cameras.extend((_cold_camera(no_rgb_record, no_rgb_image), _cold_camera(rgb_record, rgb_image)))
        rgb_enabled.extend((False, True))
    for index, camera in enumerate(cameras):
        camera._test_index = index

    result = run_true_camera_batched_step(
        engine,
        tuple(
            _request(camera, index, rgb_loss_enabled=enabled)
            for index, (camera, enabled) in enumerate(zip(cameras, rgb_enabled, strict=True))
        ),
        runtime,
        camera_microbatch_size=camera_microbatch_size,
    )

    assert result.optimizer_step_count == 1
    assert len(commits) == 1
    rgb_key = id(rgb_record.photometric_evwb)
    assert set(commits[0]) == {rgb_key}
    assert commits[0][rgb_key] is not None
    assert torch.count_nonzero(commits[0][rgb_key]) > 0
    assert no_rgb_record.photometric_evwb.grad is None
    torch.testing.assert_close(no_rgb_record.photometric_evwb, no_rgb_before)
    assert not torch.equal(rgb_record.photometric_evwb.detach(), rgb_before)


def test_rejected_after_backward_does_not_leak_evwb_gradient_into_next_accepted_transaction(
    monkeypatch,
):
    """Would catch a denied step leaving frame-owned EV/WB gradients to accumulate."""
    from test_objective_precomputed_pixel_terms_integration import _fixture
    from eligsir.optimization.gaussian import true_batch_executor as executor

    class _PostBackwardDeniedEngine(_SteppingEngine):
        def __init__(self, model):
            super().__init__(model)
            self.deny_next_step = True

        def run_post_backward_step(
            self, _gaussians, _args, _pipe, _context, *, loss_breakdown, render_outputs, **_kwargs,
        ):
            if not self.deny_next_step:
                return super().run_post_backward_step(
                    _gaussians, _args, _pipe, _context,
                    loss_breakdown=loss_breakdown, render_outputs=render_outputs,
                )
            # This is the production boundary under test: backward has already
            # run, but the policy rejects the optimizer update.
            self.session.model.optimizer.zero_grad(set_to_none=True)
            return OptimizationStepResult(
                loss_breakdown=loss_breakdown, render_outputs=render_outputs, evidence=None,
                optimization_gpu_ms=None, map_version_before=0, map_version_after=0,
                topology_version=0, step_allowed=False,
            )

    def make_case(ordinal: int, engine_type):
        image = torch.full((3, 4, 4), 0.20)
        binding = _binding(ordinal, image)
        db = FrameObservationDB(device="cpu")
        record = _prepared_evwb_admission(db, binding, "another-source")
        assert record is not None and record.photometric_evwb is not None
        with torch.no_grad():
            record.photometric_evwb.copy_(torch.tensor([0.25, -0.10, 0.05]))
        args, opt, pipe, _camera, model, _render = _fixture()
        model.get_scaling = model.get_scaling.detach().requires_grad_()
        model.get_opacity = model.get_opacity.detach().requires_grad_()
        args.opacity_reg = args.scale_reg = 0.0
        args.streaming_train_evwb_enabled = True
        args.streaming_train_evwb_reference_source = "another-source"
        args.streaming_train_evwb_learning_rate = 0.05
        args.streaming_train_evwb_prior_weight = 0.25
        args.streaming_objective_batched_pixel_terms = True
        args.streaming_free_space_loss_weight = 0.0
        args.streaming_appearance_only_enabled = False
        args.supervision_tile_probe_grid = 0
        args.surface_confidence_depth_loss = False
        render = torch.nn.Parameter(torch.full((2, 3, 4, 4), 0.60))
        model.optimizer = torch.optim.SGD([render], lr=0.1)
        engine = engine_type(model)
        runtime = OptimizationRuntimeContext(
            session=engine.session, gaussians=model, args=args, opt=opt, pipe=pipe,
            n_frames_ingested=2,
            loss_assembly_kwargs={"streaming_scene": SimpleNamespace(_observation_db=db)},
            gradient_policy_context=GradientPolicyContext(model, args, 1, 2),
        )
        cameras = (_cold_camera(record, image), _cold_camera(record, image))
        for index, camera in enumerate(cameras):
            camera._test_render = render[index]
            camera._test_population = model.get_xyz.shape[0]
        return record, engine, runtime, cameras

    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_: (True, "cpu fixture"))

    def render_batch(cameras, _model, _pipe, _background, **_kwargs):
        renders = torch.stack([camera._test_render for camera in cameras])
        return TrueCameraBatchResult(
            renders=renders, alphas=torch.ones(len(cameras), 1, 4, 4),
            rendered_depth=None, rendered_depth_ed=None,
            meta={"radii": torch.ones(len(cameras), cameras[0]._test_population)},
            camera_count=len(cameras), sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render_batch)
    stale_record, denied_engine, denied_runtime, denied_cameras = make_case(40, _PostBackwardDeniedEngine)
    stale_before = stale_record.photometric_evwb.detach().clone()
    denied = run_true_camera_batched_step(
        denied_engine,
        tuple(_request(camera, index) for index, camera in enumerate(denied_cameras)),
        denied_runtime,
        camera_microbatch_size=2,
    )
    assert denied.optimizer_step_count == 0
    assert denied.backward_call_count == 1
    assert stale_record.photometric_evwb.grad is None
    torch.testing.assert_close(stale_record.photometric_evwb, stale_before)
    denied_engine.deny_next_step = False

    clean_record, clean_engine, clean_runtime, clean_cameras = make_case(41, _SteppingEngine)
    accepted_stale = run_true_camera_batched_step(
        denied_engine,
        tuple(_request(camera, index) for index, camera in enumerate(denied_cameras)),
        denied_runtime,
        camera_microbatch_size=2,
    )
    accepted_clean = run_true_camera_batched_step(
        clean_engine,
        tuple(_request(camera, index) for index, camera in enumerate(clean_cameras)),
        clean_runtime,
        camera_microbatch_size=2,
    )

    assert accepted_stale.optimizer_step_count == accepted_clean.optimizer_step_count == 1
    torch.testing.assert_close(stale_record.photometric_evwb, clean_record.photometric_evwb)
    assert stale_record.photometric_evwb.grad is None
    assert clean_record.photometric_evwb.grad is None
