from types import SimpleNamespace
from argparse import ArgumentParser

import pytest
import torch

from arguments import StreamingParams
from eligsir.contracts.control import ModalityEligibility
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime
from eligsir.map.gpu_transactions import BatchExecutionMode
from eligsir.optimization.gaussian.contracts import (
    RandomStateToken,
)


def test_streaming_runtime_builds_typed_fixed_topology_engine_inputs(monkeypatch):
    """The streaming composition root owns config translation, not the engine."""
    gaussians = object()
    session = SimpleNamespace(model=gaussians)
    args = SimpleNamespace(
        streaming_freeze_old_geometry=True,
        streaming_freeze_new_frame_steps=7,
        streaming_young_age_frames=3,
        streaming_lifecycle_enabled=True,
        streaming_spatial_bubble_enabled=True,
        streaming_anchor_fullN_draw=False,
        streaming_bootstrap_anchor_weight=0.4,
        streaming_bootstrap_anchor_decay_steps=12,
        streaming_insertion_anchor_weight=0.6,
        streaming_insertion_anchor_decay_steps=18,
        selective_adam_zero_invisible_grads=False,
        streaming_depth_lock_centers=True,
    )
    visible = object()
    post_render_hook = object()
    runtime = StreamingOptimizationRuntime(
        session=session,
        gaussians=gaussians,
        args=args,
        opt=object(),
        pipe=object(),
        mcmc_strategy=object(),
        tb_writer=object(),
        optimizer_type="selective_adam",
        sparse_active_set=True,
        use_energy_mcmc=True,
        fixed_shape_mcmc=True,
        mcmc_local=True,
        loss_assembly_kwargs={"scalar_log_interval": 9},
        iter_since_new_frame=2,
        post_render_hook=post_render_hook,
        optimizer_iteration_limit=77,
    )
    token = RandomStateToken(torch_rng_state=b"test")

    request, context = runtime.build_step(
        camera="camera",
        background="background",
        iteration=11,
        map_version=4,
        topology_version=9,
        resolution_bucket="high",
        update_sh_rest=False,
        render_depth=True,
        n_frames_ingested=27,
        visible_mask=visible,
        random_state_token=token,
        optimizer_update_id=7,
    )

    assert request.view.camera == "camera"
    assert request.view.background == "background"
    assert request.view.render_depth is True
    assert request.random_state_token is token
    assert request.map_version == 4
    assert request.topology_version == 9
    assert context.session is session
    assert context.gaussians is gaussians
    assert context.fixed_shape_mcmc is True
    assert context.mcmc_local is True
    assert context.loss_assembly_kwargs == {"scalar_log_interval": 9}
    assert context.post_render_hook is post_render_hook
    assert context.optimizer_iteration_limit == 77
    assert context.optimizer_update_id == 7
    assert context.optimizer_update_permitted is False
    policies = context.gradient_policy_context
    assert policies.active_set.visible_mask is visible
    assert policies.active_set.zero_invisible_grads is False
    assert policies.stable_geometry.freeze_old_geometry is True
    assert policies.stable_geometry.successful_updates_since_admission == 2
    assert policies.stable_geometry.bubble.enabled is True
    assert policies.anchor.bootstrap.weight == 0.4
    assert policies.anchor.bootstrap.decay_steps == 12
    assert policies.anchor.bootstrap.full_n_draw is False
    assert policies.anchor.insertion.weight == 0.6
    assert policies.anchor.insertion.decay_steps == 18
    assert policies.depth_lock.active is True


def test_canonical_build_step_requires_permission_for_each_planned_update():
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    kwargs = dict(
        camera=None, background=None, iteration=1000, map_version=0,
        topology_version=0, resolution_bucket="default", update_sh_rest=True,
        render_depth=False, n_frames_ingested=1, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
        optimizer_update_id=13,
    )

    _, endpoint = runtime.build_step(**kwargs)
    _, planned = runtime.build_step(**kwargs, optimizer_update_permitted=True)

    assert endpoint.optimizer_update_permitted is False
    assert planned.optimizer_update_permitted is True


def test_admission_resets_stable_geometry_age_to_successful_update_clock():
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=99,
    )
    runtime.note_admitted_source(optimizer_updates_completed=12)

    _, context = runtime.build_step(
        camera=None, background=None, iteration=1000, map_version=0,
        topology_version=0, resolution_bucket="default", update_sh_rest=True,
        render_depth=False, n_frames_ingested=1, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
        optimizer_update_id=13,
    )

    assert context.gradient_policy_context.stable_geometry.successful_updates_since_admission == 1
    assert runtime.iter_since_new_frame == 0


def test_streaming_runtime_carries_gradient_clock_forward_after_engine_step():
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=object(), args=SimpleNamespace(), opt=object(), pipe=object(),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=3,
        post_render_hook=None,
    )
    _, context = runtime.build_step(
        camera=None, background=None, iteration=1, map_version=0,
        topology_version=0, resolution_bucket="default", update_sh_rest=True,
        render_depth=False, n_frames_ingested=1, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )
    context.gradient_policy_context.stable_geometry.iter_since_new_frame = 4

    runtime.commit_step(context)

    assert runtime.iter_since_new_frame == 4


def test_streaming_runtime_stamps_the_camera_source_index_on_the_typed_view():
    """The engine must receive a stable identity, never infer one after a skip."""
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    camera = SimpleNamespace(uid=73)

    request, _ = runtime.build_step(
        camera=camera, background=None, iteration=1, map_version=0,
        topology_version=0, resolution_bucket="default", update_sh_rest=True,
        render_depth=False, n_frames_ingested=1, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert request.view.source_index == 73


def test_disabled_assimilation_support_only_samples_on_existing_scalar_cadence(
    monkeypatch,
):
    """A disabled controller must not force a tensor-to-host summary each step."""
    import torch
    import eligsir.utils.assimilation_pressure as pressure

    published = []
    monkeypatch.setattr(pressure, "assimilation_control_enabled", lambda: False)
    monkeypatch.setattr(
        pressure,
        "publish_support_counts",
        lambda *_args, **_kwargs: published.append("support") or 0.25,
    )
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(
            get_xyz=torch.zeros((2, 3)),
            support_count=torch.ones(2),
        ),
        args=SimpleNamespace(streaming_min_support_views=2),
        opt=object(),
        pipe=object(),
        mcmc_strategy=None,
        tb_writer=SimpleNamespace(add_scalar=lambda *_args: None),
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={"scalar_log_interval": 5},
        iter_since_new_frame=0,
    )

    runtime.publish_assimilation_support(4)
    assert published == []

    runtime.publish_assimilation_support(5)
    assert published == ["support"]


def test_streaming_runtime_dispatches_one_typed_engine_step_and_commits_clock():
    """The composition root uses the typed request/context boundary end to end."""
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    gaussians = object()
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=gaussians),
        gaussians=gaussians, args=SimpleNamespace(), opt=object(), pipe=object(),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=3,
    )

    class Engine:
        def run_step(self, request, context):
            self.request = request
            self.context = context
            context.gradient_policy_context.stable_geometry.iter_since_new_frame = 4
            return "step-result"

    engine = Engine()
    result = run_streaming_optimization_step(
        runtime,
        engine,
        camera="camera",
        background="background",
        iteration=11,
        map_version=4,
        topology_version=9,
        resolution_bucket="high",
        update_sh_rest=False,
        render_depth=True,
        n_frames_ingested=27,
        visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert result == "step-result"
    assert engine.request.iteration == 11
    assert engine.request.map_version == 4
    assert engine.request.topology_version == 9
    assert engine.context.gaussians is gaussians
    assert runtime.iter_since_new_frame == 4


def test_batched_depth_insertion_borrows_versions_without_population_shadow(monkeypatch):
    """The selected insertion source gets a reuse fence, not E0 demand."""

    from eligsir.mapping.optimization_runtime import (
        run_streaming_optimization_transaction,
    )
    from eligsir.mapping.population_shadow_runtime import PopulationShadowVersions
    from eligsir.runtime.config import (
        CanonicalOptimizationPolicy,
        MappingRuntimeConfig,
    )
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    from eligsir.optimization.gaussian import true_batch_executor

    captured = {}

    def _batched_step(_engine, requests, context, **_kwargs):
        captured["requests"] = requests
        captured["context"] = context
        return SimpleNamespace(
            random_state_token=None,
            optimizer_step_count=0,
            supervision_view_count=0,
            skipped_view_count=0,
            rasterizer_call_count=0,
            supervised_pixel_count=0,
            map_version_before=4,
            map_version_after=4,
            topology_version=9,
        )

    monkeypatch.setattr(true_batch_executor, "run_true_camera_batched_step", _batched_step)
    versions = PopulationShadowVersions(
        map_content=4,
        topology=9,
        coordinate_frame=2,
        schema=3,
        optimizer_update=7,
        surface_evidence=11,
        policy=13,
        parameter=5,
    )
    production = object.__new__(CanonicalMappingRuntime)
    production.config = MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="default",
        static_options={"population_shadow_enabled": False}
    )
    production._runtime_policy = SimpleNamespace(cap_max=0)
    production._population_shadow_evidence_versions = lambda: versions
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        args=CanonicalOptimizationPolicy(),
        opt=object(),
        pipe=object(),
        mcmc_strategy=None,
        tb_writer=None,
        optimizer_type="adam",
        sparse_active_set=False,
        use_energy_mcmc=False,
        fixed_shape_mcmc=False,
        mcmc_local=False,
        loss_assembly_kwargs={},
        iter_since_new_frame=0,
        **production._population_shadow_runtime_kwargs(),
    )

    run_streaming_optimization_transaction(
        runtime,
        object(),
        cameras=(object(), object()),
        requested_batch_size=2,
        requested_camera_count=2,
        modality_eligibilities=(
            ModalityEligibility(depth_insertion=True),
            ModalityEligibility(),
        ),
        background=None,
        iteration=7,
        map_version=4,
        topology_version=9,
        resolution_bucket="native",
        update_sh_rest=True,
        render_depth=True,
        n_frames_ingested=2,
        visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
        optimizer_update_id=7,
        optimizer_update_permitted=True,
    )

    assert all(
        not request.evidence_requirements.want_supervision_evidence
        for request in captured["requests"]
    )
    assert captured["context"].evidence_transaction_id == 7
    assert captured["context"].coordinate_frame_version == 2
    assert captured["context"].schema_version == 3
    assert captured["context"].surface_evidence_version == 11
    assert captured["context"].policy_version == 13


def test_streaming_runtime_dispatch_binds_render_visibility_before_engine_policies():
    import torch

    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    visible = torch.tensor([True, False])
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="selective_adam", sparse_active_set=True,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    class Engine:
        def run_step(self, request, context):
            context.post_render_hook(SimpleNamespace(payload={"visibility_filter": visible}))
            assert torch.equal(context.gradient_policy_context.active_set.visible_mask, visible)
            return object()

    run_streaming_optimization_step(
        runtime, Engine(), camera=None, background=None, iteration=1,
        map_version=0, topology_version=0, resolution_bucket="default",
        update_sh_rest=True, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )


def test_streaming_runtime_dispatch_carries_the_engine_rng_token_forward():
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=object(),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    returned = RandomStateToken(torch_rng_state=b"next")

    class Engine:
        def run_step(self, request, context):
            return SimpleNamespace(random_state_token=returned)

    run_streaming_optimization_step(
        runtime, Engine(), camera=None, background=None, iteration=1,
        map_version=0, topology_version=0, resolution_bucket="default",
        update_sh_rest=True, render_depth=False, n_frames_ingested=1,
        visible_mask=None, random_state_token=RandomStateToken(torch_rng_state=b"start"),
    )

    assert runtime.random_state_token is returned


def test_streaming_optimization_engine_is_promoted_with_explicit_compatibility_off_switch():
    parser = ArgumentParser()
    StreamingParams(parser)

    assert parser.parse_args([]).streaming_optimization_engine is True
    assert parser.parse_args(["--streaming_optimization_engine"]).streaming_optimization_engine is True
    assert parser.parse_args(["--no-streaming_optimization_engine"]).streaming_optimization_engine is False


def test_prune_mask_shape_mismatch_warns_and_preserves_the_stale_mask(capsys):
    import torch

    from eligsir.mapping.optimization_runtime import merge_deferred_prune_masks

    stale = torch.tensor([True, False])
    conflict = torch.tensor([False, True, True])

    merged = merge_deferred_prune_masks(stale, conflict)

    assert torch.equal(merged, stale)
    assert "conflict prune mask shape mismatch (2 vs 3), skipping conflicts" in capsys.readouterr().out


def test_streaming_runtime_requires_an_injected_transfer_owner_on_accelerator():
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cuda")),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    with pytest.raises(RuntimeError, match="injected canonical camera_transfer_owner"):
        runtime.prefetch_selected_cameras([object()])


def test_streaming_runtime_uses_the_injected_canonical_transfer_owner():
    calls = []

    class TransferOwner:
        def prefetch_cameras(self, cameras):
            calls.append(tuple(cameras))
            return 2

    camera = object()
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cuda")),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
        camera_transfer_owner=TransferOwner(),
    )

    assert runtime.prefetch_selected_cameras([camera]) == 2
    assert calls == [(camera,)]


def test_streaming_runtime_keeps_cpu_harnesses_transfer_owner_free():
    import torch

    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device=torch.device("cpu"))),
        args=SimpleNamespace(), opt=object(), pipe=object(), mcmc_strategy=None,
        tb_writer=None, optimizer_type="adam", sparse_active_set=False,
        use_energy_mcmc=False, fixed_shape_mcmc=False, mcmc_local=False,
        loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    assert runtime.prefetch_selected_cameras([object()]) == 0


def test_requested_true_camera_batch_refuses_unqualified_renderer():
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    class Camera:
        image_width = 2
        image_height = 2

    primary, secondary = Camera(), Camera()
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=False),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    with pytest.raises(RuntimeError, match="planned TRUE_CAMERA_BATCH is unavailable"):
        run_streaming_optimization_step(
            runtime,
            object(),
            camera=primary,
            extra_cameras=[secondary],
            background=None,
            iteration=1,
            map_version=0,
            topology_version=0,
            resolution_bucket="default",
            update_sh_rest=True,
            render_depth=False,
            n_frames_ingested=1,
            visible_mask=None,
            random_state_token=RandomStateToken(torch_rng_state=b"test"),
            execution_mode=BatchExecutionMode.TRUE_CAMERA_BATCH,
        )


def test_requested_true_camera_batch_refuses_a_single_view_request():
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    with pytest.raises(RuntimeError, match="requires_at_least_two_materialized_views"):
        run_streaming_optimization_step(
            runtime,
            object(),
            camera=SimpleNamespace(image_width=2, image_height=2),
            background=None,
            iteration=1,
            map_version=0,
            topology_version=0,
            resolution_bucket="default",
            update_sh_rest=True,
            render_depth=False,
            n_frames_ingested=1,
            visible_mask=None,
            random_state_token=RandomStateToken(torch_rng_state=b"test"),
            execution_mode=BatchExecutionMode.TRUE_CAMERA_BATCH,
        )


def test_default_batched_dispatch_stays_on_the_planned_sequential_path(monkeypatch):
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step
    from eligsir.optimization.gaussian import true_batch_executor

    class Camera:
        image_width = 2
        image_height = 2

    primary, secondary = Camera(), Camera()
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    def fail_if_true_batch_is_inferred(*_args, **_kwargs):
        pytest.fail("sequential work must not infer TRUE_CAMERA_BATCH")

    monkeypatch.setattr(
        true_batch_executor, "run_true_camera_batched_step", fail_if_true_batch_is_inferred
    )

    class Engine:
        def run_batched_step(self, requests, _context, *, skipped_view_count):
            assert [request.view.camera for request in requests] == [primary, secondary]
            assert skipped_view_count == 0
            return SimpleNamespace(random_state_token=None)

    result = run_streaming_optimization_step(
        runtime,
        Engine(),
        camera=primary,
        extra_cameras=[secondary],
        background=None,
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="default",
        update_sh_rest=True,
        render_depth=False,
        n_frames_ingested=1,
        visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert result.true_camera_batch is False
    assert result.true_camera_batch_fallback_reason == "planned_sequential_accumulation"


def test_adaptive_transaction_partitions_b_greater_than_c_and_steps_once(monkeypatch):
    from eligsir.mapping.optimization_runtime import (
        run_streaming_optimization_transaction,
    )
    from eligsir.optimization.gaussian import true_batch_executor

    cameras = [SimpleNamespace(uid=index) for index in range(8)]
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=SimpleNamespace(
            get_xyz=SimpleNamespace(device="cpu")
        ),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    captured = {}

    def _microbatched(
        _engine, requests, _context, *, camera_microbatch_size, skipped_view_count=0
    ):
        captured["groups"] = tuple(
            tuple(request.view.source_index for request in requests[index : index + camera_microbatch_size])
            for index in range(0, len(requests), camera_microbatch_size)
        )
        assert skipped_view_count == 0
        return SimpleNamespace(
            random_state_token=None,
            optimizer_step_count=1,
            supervision_view_count=8,
            rasterizer_call_count=4,
        )

    monkeypatch.setattr(
        true_batch_executor, "run_true_camera_batched_step", _microbatched
    )

    result = run_streaming_optimization_transaction(
        runtime,
        object(),
        cameras=cameras,
        requested_camera_count=2,
        background=None,
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="default",
        update_sh_rest=True,
        render_depth=False,
        n_frames_ingested=8,
        visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert captured["groups"] == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert result.optimizer_step_count == 1
    assert result.microbatch_count == 4
    assert result.requested_camera_count == 2
    assert result.realized_camera_count == 2


def test_batched_build_reuses_one_runtime_context_for_all_camera_requests(monkeypatch):
    """B requests are view-specific; policy context is transaction-shared."""
    import eligsir.mapping.optimization_runtime as optimization_runtime
    from eligsir.mapping.optimization_runtime import (
        run_streaming_optimization_transaction,
    )
    from eligsir.optimization.gaussian import true_batch_executor

    context_type = optimization_runtime.OptimizationRuntimeContext
    context_calls = []
    policy_type = optimization_runtime.GradientPolicyContext
    policy_calls = []

    def _counted_context(*args, **kwargs):
        context = context_type(*args, **kwargs)
        context_calls.append(context)
        return context

    def _counted_policy(*args, **kwargs):
        policy = policy_type(*args, **kwargs)
        policy_calls.append(policy)
        return policy

    monkeypatch.setattr(optimization_runtime, "OptimizationRuntimeContext", _counted_context)
    monkeypatch.setattr(optimization_runtime, "GradientPolicyContext", _counted_policy)
    cameras = [SimpleNamespace(uid=index) for index in range(3)]
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    captured = {}

    def _microbatched(_engine, requests, context, *, camera_microbatch_size, **_kwargs):
        captured["requests"] = tuple(requests)
        captured["context"] = context
        return SimpleNamespace(
            random_state_token=None, optimizer_step_count=1,
            supervision_view_count=3, rasterizer_call_count=1,
        )

    monkeypatch.setattr(true_batch_executor, "run_true_camera_batched_step", _microbatched)
    run_streaming_optimization_transaction(
        runtime, object(), cameras=cameras, requested_camera_count=2,
        background=None, iteration=1, map_version=0, topology_version=0,
        resolution_bucket="default", update_sh_rest=True, render_depth=False,
        n_frames_ingested=3, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert len(context_calls) == 1
    assert len(policy_calls) == 1
    assert captured["context"] is context_calls[0]
    assert tuple(request.view.camera for request in captured["requests"]) == tuple(cameras)


def test_legacy_extra_camera_build_reuses_first_runtime_context(monkeypatch):
    import eligsir.mapping.optimization_runtime as optimization_runtime
    from eligsir.mapping.optimization_runtime import run_streaming_optimization_step

    context_type = optimization_runtime.OptimizationRuntimeContext
    context_calls = []

    def _counted_context(*args, **kwargs):
        context = context_type(*args, **kwargs)
        context_calls.append(context)
        return context

    monkeypatch.setattr(optimization_runtime, "OptimizationRuntimeContext", _counted_context)
    cameras = [SimpleNamespace(uid=index) for index in range(3)]
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        args=SimpleNamespace(), opt=object(), pipe=SimpleNamespace(),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    captured = {}

    class Engine:
        def run_batched_step(self, requests, context, **_kwargs):
            captured["requests"] = tuple(requests)
            captured["context"] = context
            return SimpleNamespace(
                random_state_token=None, optimizer_step_count=1,
                supervision_view_count=3, skipped_view_count=0,
            )

    run_streaming_optimization_step(
        runtime, Engine(), extra_cameras=cameras[1:], camera=cameras[0],
        background=None, iteration=1, map_version=0, topology_version=0,
        resolution_bucket="default", update_sh_rest=True, render_depth=False,
        n_frames_ingested=3, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert len(context_calls) == 1
    assert captured["context"] is context_calls[0]
    assert tuple(request.view.camera for request in captured["requests"]) == tuple(cameras)


def test_adaptive_transaction_runs_single_underfilled_camera_immediately(monkeypatch):
    from eligsir.mapping.optimization_runtime import (
        run_streaming_optimization_transaction,
    )
    from eligsir.optimization.gaussian import true_batch_executor

    camera = SimpleNamespace(uid=7)
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=SimpleNamespace(
            get_xyz=SimpleNamespace(device="cpu")
        ),
        args=SimpleNamespace(), opt=object(),
        pipe=SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )

    monkeypatch.setattr(
        true_batch_executor,
        "run_true_camera_batched_step",
        lambda *_args, **_kwargs: pytest.fail("C1 underfill must not request true C"),
    )

    class Engine:
        def run_step(self, request, _context):
            assert request.view.camera is camera
            return SimpleNamespace(
                random_state_token=None,
                optimizer_step_count=1,
                supervision_view_count=1,
            )

    result = run_streaming_optimization_transaction(
        runtime,
        Engine(),
        cameras=(camera,),
        requested_batch_size=256,
        requested_camera_count=64,
        background=None,
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="default",
        update_sh_rest=True,
        render_depth=False,
        n_frames_ingested=1,
        visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    assert result.optimizer_step_count == 1
    assert result.microbatch_count == 1
    assert result.requested_batch_size == 256
    assert result.realized_batch_size == 1
    assert result.realized_camera_count == 1


def test_adaptive_transaction_refuses_identity_duplication_and_unqualified_target_c():
    from eligsir.mapping.optimization_runtime import (
        run_streaming_optimization_transaction,
    )

    camera = SimpleNamespace(uid=7)
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()), gaussians=SimpleNamespace(
            get_xyz=SimpleNamespace(device="cpu")
        ),
        args=SimpleNamespace(), opt=object(), pipe=SimpleNamespace(),
        mcmc_strategy=None, tb_writer=None, optimizer_type="adam",
        sparse_active_set=False, use_energy_mcmc=False, fixed_shape_mcmc=False,
        mcmc_local=False, loss_assembly_kwargs={}, iter_since_new_frame=0,
    )
    kwargs = dict(
        background=None, iteration=1, map_version=0, topology_version=0,
        resolution_bucket="default", update_sh_rest=True, render_depth=False,
        n_frames_ingested=1, visible_mask=None,
        random_state_token=RandomStateToken(torch_rng_state=b"test"),
    )

    with pytest.raises(ValueError, match="unique camera identities"):
        run_streaming_optimization_transaction(
            runtime, object(), cameras=(camera, camera), requested_camera_count=2,
            **kwargs,
        )
    with pytest.raises(ValueError, match="qualified supervision camera count"):
        run_streaming_optimization_transaction(
            runtime, object(), cameras=(camera,), requested_camera_count=6,
            **kwargs,
        )
