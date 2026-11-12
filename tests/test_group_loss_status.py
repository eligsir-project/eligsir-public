"""One numerical-status readback per real objective render group."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian import true_batch_executor as executor
from test_objective_batched_pixel_terms_wiring import _Engine, _request, _runtime
from test_objective_precomputed_pixel_terms_integration import _fixture


def _case(monkeypatch, *, batch_size=32, regularizers=False):
    args, opt, pipe, cam, model, _ = _fixture()
    # Fresh leaf parameters avoid fixture-only graph sharing across backwards.
    model.get_scaling = model.get_scaling.detach().requires_grad_()
    model.get_opacity = model.get_opacity.detach().requires_grad_()
    engine = _Engine(model)
    runtime = _runtime(engine, model, flag=False)
    runtime.args, runtime.opt, runtime.pipe = args, opt, pipe
    args.streaming_objective_step_regularizers = regularizers
    runtime.sparse_active_set = regularizers
    cameras = []
    for i in range(batch_size):
        camera = SimpleNamespace(
            uid=i, original_image=cam.original_image, image_height=4, image_width=4,
            _source_identity=object(), _prepared_score_provenance=True,
        )
        cameras.append(camera)
    renders = torch.rand(batch_size, 3, 4, 4, requires_grad=True)
    model.optimizer = torch.optim.SGD([renders, model.get_scaling, model.get_opacity], lr=0.0)
    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda *_: (True, "ok"))

    def render(group, *_args, **_kwargs):
        indices = [camera.uid for camera in group]
        return TrueCameraBatchResult(
            renders=renders[indices], alphas=torch.ones(len(group), 1, 4, 4),
            rendered_depth=None, rendered_depth_ed=None,
            meta={"radii": torch.ones(len(group), model.get_xyz.shape[0])},
            camera_count=len(group), sparse_grad=True,
        )

    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    return engine, runtime, cameras, renders


@pytest.mark.parametrize("camera_count", [32, 8])
@pytest.mark.parametrize("regularizers", [False, True])
def test_real_objective_has_one_status_readback_per_render_group(monkeypatch, camera_count, regularizers):
    engine, runtime, cameras, renders = _case(monkeypatch, regularizers=regularizers)
    readbacks = []
    original = torch.Tensor.tolist

    def tolist(tensor):
        readbacks.append(tuple(tensor.shape))
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    def forbid_item(_tensor, *_args):
        raise AssertionError("per-view scalar status readback")
    monkeypatch.setattr(torch.Tensor, "item", forbid_item)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    assert result.backward_ran
    assert renders.grad is not None and torch.isfinite(renders.grad).all()
    assert len(readbacks) == 32 // camera_count
    packet_size = 2 * camera_count + 1 if regularizers else 3 * camera_count
    assert readbacks == [(packet_size,)] * (32 // camera_count)


@pytest.mark.parametrize("camera_count", [32, 8])
@pytest.mark.parametrize("regularizers", [False, True])
def test_executor_real_objective_forward_and_gradients_match_sequential_reference(monkeypatch, camera_count, regularizers):
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs, assemble_loss
    from eligsir.optimization.gaussian.contracts import RenderOutputs

    engine, runtime, cameras, renders = _case(monkeypatch, regularizers=regularizers)
    model = runtime.gaussians
    expected_groups = []
    for start in range(0, 32, camera_count):
        breakdowns = []
        for i in range(start, start + camera_count):
            image = renders[i]
            breakdowns.append(assemble_loss(LossAssemblyInputs(
                gaussians=model, viewpoint_cam=cameras[i], args=runtime.args,
                opt=runtime.opt, pipe=runtime.pipe, image=image,
                render_outputs=RenderOutputs(payload={
                    "render": image, "visibility_filter": torch.ones(5, dtype=torch.bool),
                }), iteration=1, n_frames_ingested=4,
                sparse_active_set=regularizers, skip_gaussian_regularizers=regularizers,
            )))
        aggregate = executor._aggregate_loss(breakdowns, 1 / 32)
        if regularizers:
            # Preserve the original per-view mask reduction, not its rounded
            # closed-form C/N replacement, for bit-exact regularizer gradients.
            masks = torch.ones(camera_count, 5)
            weights = (masks / masks.sum(dim=1)[:, None]).sum(dim=0)
            reg_step = executor._losses.assemble_step_regularizers(
                model, runtime.args, 1, weights, batch_size=32,
            )
            aggregate.total_tensor = aggregate.total_tensor + reg_step
            aggregate.total = aggregate.total_tensor.detach()
        expected_groups.append(aggregate.total)
        aggregate.total_tensor.backward()
    expected_total = sum(expected_groups)
    params = (renders, model.get_scaling, model.get_opacity)
    expected_gradients = [p.grad.clone() for p in params]
    model.optimizer.zero_grad(set_to_none=True)

    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    torch.testing.assert_close(result.loss_breakdown.total, expected_total, rtol=0, atol=0)
    for param, expected in zip(params, expected_gradients):
        torch.testing.assert_close(param.grad, expected, rtol=0, atol=0)
    assert result.batch_loss_scale == 1 / 32
    assert result.backward_call_count == 32 // camera_count


@pytest.mark.parametrize("bad_index", [0, 2])
@pytest.mark.parametrize("regularizers", [False, True])
def test_actual_nonfinite_view_preserves_abort_skip_and_callback_order(monkeypatch, bad_index, regularizers):
    engine, runtime, cameras, renders = _case(monkeypatch, batch_size=4, regularizers=regularizers)
    skips = [0]
    runtime.loss_assembly_kwargs = {"nan_skip_count_holder": skips}
    with torch.no_grad():
        renders[bad_index].fill_(float("nan"))
        if bad_index == 0 or regularizers:
            renders[bad_index:].fill_(float("nan"))
    finalized = []
    original = executor._objective.prepare_loss

    def prepare(inputs):
        accumulator, loss, finalize = original(inputs)
        def finish(*resolved):
            finalized.append(inputs.viewpoint_cam.uid)
            return finalize(*resolved)
        return accumulator, loss, finish

    monkeypatch.setattr(executor._objective, "prepare_loss", prepare)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(camera) for camera in cameras), runtime,
    )
    abort = bad_index == 0 or regularizers
    assert result.optimizer_step_count == (0 if abort else 1)
    assert result.backward_call_count == (0 if abort else 1)
    assert finalized == (list(range(bad_index + 1)) if abort else [0, 1, 2, 3])
    assert skips == [1]
    if abort:
        assert result.skip_reason == "non_finite_loss"
        assert renders.grad is None
    else:
        assert result.skipped_view_count == 1
        assert result.successful_supervision_source_indices == (0, 1, 3)
        assert result.batch_loss_scale == 0.25


def test_nonfinite_step_regularizer_is_dropped_not_transaction_rejected(monkeypatch):
    engine, runtime, cameras, renders = _case(monkeypatch, batch_size=4, regularizers=True)
    bad = torch.tensor(1.0, requires_grad=True)
    monkeypatch.setattr(executor._losses, "assemble_step_regularizers", lambda *_a, **_kw: bad / 0)
    reports = []
    monkeypatch.setattr(executor, "_nan_term_report", lambda name, _it: reports.append(name))
    result = executor.run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), runtime)
    assert result.optimizer_step_count == result.backward_call_count == 1
    assert reports == ["active_reg+young"]
    assert bad.grad is None
    assert torch.isfinite(renders.grad).all()


def test_pending_finalization_captures_are_released_before_next_render(monkeypatch):
    import weakref

    engine, runtime, cameras, _ = _case(monkeypatch, batch_size=8)
    original_prepare = executor._objective.prepare_loss
    original_render = executor.render_true_camera_batch
    captures = []
    saved = []

    class SavedTensor:
        def __init__(self, tensor):
            self.tensor = tensor

    def pack(tensor):
        value = SavedTensor(tensor)
        saved.append(weakref.ref(value))
        return value

    def prepare(inputs):
        result = original_prepare(inputs)
        captures.append(weakref.ref(result[2]))
        return result

    def render(*args, **kwargs):
        assert all(ref() is None for ref in captures)
        assert all(ref() is None for ref in saved)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(executor._objective, "prepare_loss", prepare)
    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value.tensor):
        executor.run_true_camera_batched_step(
            engine, tuple(_request(c) for c in cameras), runtime, camera_microbatch_size=4,
        )
    assert len(captures) == 8 and all(ref() is None for ref in captures)
    assert saved and all(ref() is None for ref in saved)


@pytest.mark.parametrize("empty", [False, True])
def test_group_packet_exact_gradients_callbacks_and_int64_depth_counts(monkeypatch, empty):
    from eligsir.mapping.integration.loss_diagnostics import GuardedLossAccumulator

    def build():
        parameters, pending, calls = [], [], []
        for view in range(4):
            p = torch.tensor([1.25, 2.5, 3.75], requires_grad=True)
            acc = GuardedLossAccumulator()
            if not empty:
                term = p[1] / 0 if view == 1 else p[1].square()
                acc.depth_valid_sample_count_device = torch.tensor(2**40 + view)
                acc.add(term, gates_depth_valid_sample_count=True,
                        on_resolved=lambda finite, a=acc, v=view: calls.append(
                            (v, finite, a.resolved_depth_valid_sample_count)))
                acc.add_unconditional(p[2].square())
                acc.add(p[0] * 0.25)
            pending.append((acc, p[0].square()))
            parameters.append(p)
        return parameters, pending, calls

    old_params, old_pending, old_calls = build()
    expected = [acc.resolve(loss, 1) for acc, loss in old_pending]
    sum(loss for loss, _ in expected).backward()
    params, pending, calls = build()
    reads = []
    original = torch.Tensor.tolist
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", lambda t: (reads.append(t.dtype), original(t))[1])
        patch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("separate item readback"))
        status = iter(torch.cat([acc.status_tensor(loss) for acc, loss in pending]).tolist())
        actual = [acc.resolve(loss, 1, status=status) for acc, loss in pending]
    sum(loss for loss, _ in actual).backward()
    assert reads == [torch.int64]
    assert calls == old_calls
    if not empty:
        assert calls == [(0, True, 2**40), (1, False, 0), (2, True, 2**40 + 2), (3, True, 2**40 + 3)]
        assert params[1].grad[1] == 0
    for (loss, finite), (reference, reference_finite), p, old_p in zip(actual, expected, params, old_params):
        assert finite == reference_finite
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, old_p.grad, rtol=0, atol=0)


@pytest.mark.parametrize("camera_count", [4, 2])
@pytest.mark.parametrize("batched_pixels", [False, True])
def test_real_executor_depth_guard_accepts_exact_counts_and_drops_inf_term(monkeypatch, camera_count, batched_pixels):
    engine, runtime, cameras, _ = _case(monkeypatch, batch_size=4)
    for camera in cameras:
        camera._sensor_depth_cache = torch.ones(1, 4, 4)
    runtime.args.surface_confidence_depth_loss = False
    runtime.args.streaming_objective_batched_pixel_terms = batched_pixels
    runtime.loss_assembly_kwargs = {"depth_loss_weight": 0.25}
    depths = torch.full((4, 1, 4, 4), 2.0, requires_grad=True)
    with torch.no_grad():
        depths[1].fill_(float("inf"))
    original_render = executor.render_true_camera_batch

    def render(group, *args, **kwargs):
        batch = original_render(group, *args, **kwargs)
        return replace(batch, rendered_depth=depths[[camera.uid for camera in group]])

    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(c, depth_loss_enabled=True) for c in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    assert result.optimizer_step_count == 1
    assert result.depth_valid_sample_count == 3 * 16
    assert result.supervision_view_count == 4
    assert result.batch_loss_scale == 0.25
    assert torch.count_nonzero(depths.grad[1]) == 0
    torch.testing.assert_close(depths.grad[[0, 2, 3]], torch.full((3, 1, 4, 4), 0.25 / 4 / 16))


@pytest.mark.parametrize("camera_count", [4, 2])
def test_real_depth_e0_survives_without_tensorboard_logging(monkeypatch, camera_count):
    """E0 must receive accepted raw depth data outside the logging cadence."""

    from eligsir.contracts.runtime import SourceIdentity
    from eligsir.optimization.gaussian.contracts import EvidenceRequirements
    from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality

    engine, runtime, cameras, _ = _case(monkeypatch, batch_size=4)
    for camera in cameras:
        camera._sensor_depth_cache = torch.ones(1, 4, 4)
        camera._source_identity = SourceIdentity("test", 0, camera.uid, camera.uid)
    runtime.evidence_transaction_id = 31
    runtime.coordinate_frame_version = runtime.schema_version = 0
    runtime.surface_evidence_version = runtime.policy_version = 0
    engine.session.parameter_version = 0
    runtime.args.surface_confidence_depth_loss = False
    assert runtime.tb_writer is None
    depths = torch.stack(
        [torch.full((1, 4, 4), 1.5 + camera.uid) for camera in cameras]
    ).requires_grad_()
    original_render = executor.render_true_camera_batch

    def render(group, *args, **kwargs):
        batch = original_render(group, *args, **kwargs)
        return replace(batch, rendered_depth=depths[[camera.uid for camera in group]])

    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    requests = []
    for camera in cameras:
        request = _request(camera, depth_loss_enabled=True)
        requests.append(
            replace(
                request,
                evidence_requirements=EvidenceRequirements(want_supervision_evidence=True),
                view=replace(
                    request.view,
                    source_identity=camera._source_identity,
                    supervision_modality=SupervisionModality(
                        depth_loss_enabled=True, depth_evidence_enabled=True
                    ),
                ),
            )
        )

    result = executor.run_true_camera_batched_step(
        engine, tuple(requests), runtime, camera_microbatch_size=camera_count
    )

    assert result.optimizer_step_count == 1
    assert result.evidence.source_identities == tuple(
        camera._source_identity for camera in cameras
    )
    assert torch.equal(
        result.evidence.depth_residual_summary, torch.tensor([0.5, 1.5, 2.5, 3.5])
    )
    assert torch.equal(result.evidence.depth_valid_pixels, torch.tensor([16, 16, 16, 16]))


@pytest.mark.parametrize("iteration", [1, 10])
@pytest.mark.parametrize("primary_failed", [False, True])
def test_real_depth_diagnostics_keep_cadence_and_stop_at_primary_failure(monkeypatch, iteration, primary_failed):
    engine, runtime, cameras, renders = _case(monkeypatch, batch_size=4)
    for camera in cameras:
        camera._sensor_depth_cache = torch.ones(1, 4, 4)
    if primary_failed:
        with torch.no_grad():
            renders.fill_(float("nan"))
    runtime.args.surface_confidence_depth_loss = False
    logs = []
    writer = SimpleNamespace(add_scalar=lambda name, value, tick: logs.append((name, value, tick)))
    runtime.loss_assembly_kwargs = {"depth_loss_weight": 0.25, "tb_writer": writer}
    original_render = executor.render_true_camera_batch

    def render(group, *args, **kwargs):
        batch = original_render(group, *args, **kwargs)
        return replace(batch, rendered_depth=torch.stack([
            torch.full((1, 4, 4), 1.5 + camera.uid) for camera in group
        ]))

    monkeypatch.setattr(executor, "render_true_camera_batch", render)
    executor.run_true_camera_batched_step(
        engine, tuple(replace(_request(c, depth_loss_enabled=True), iteration=iteration) for c in cameras), runtime,
    )
    depth_logs = [(value, tick) for name, value, tick in logs if name == "streaming/depth_loss_raw_magnitude"]
    expected = [0.5] if primary_failed else [0.5, 1.5, 2.5, 3.5]
    assert depth_logs == ([(value, iteration) for value in expected] if iteration == 10 else [])
    assert sum(name == "streaming/nan_skip" for name, _, _ in logs) == int(primary_failed)


@pytest.mark.parametrize("camera_count", [8, 4])
def test_group_construction_packet_and_finalization_have_exact_timing_scopes(monkeypatch, camera_count):
    from contextlib import contextmanager
    from eligsir.mapping.integration.loss_diagnostics import GuardedLossAccumulator

    engine, runtime, cameras, _ = _case(monkeypatch, batch_size=8)
    stack, calls, scopes = [], [], []

    @contextmanager
    def timed(name):
        stack.append(name)
        scopes.append(name)
        try:
            yield
        finally:
            assert stack.pop() == name

    runtime.timing_scope = timed
    original_prepare = executor._objective.prepare_loss
    original_status = GuardedLossAccumulator.status_tensor
    original_resolve = GuardedLossAccumulator.resolve
    original_tolist = torch.Tensor.tolist

    def prepare(inputs):
        assert stack[-1:] == ["m_loss_assembly"]
        calls.append(("construct", inputs.viewpoint_cam.uid))
        acc, loss, finalize = original_prepare(inputs)
        def finish(*resolved):
            assert stack[-1:] == ["m_loss_assembly"]
            calls.append(("finalize", inputs.viewpoint_cam.uid))
            return finalize(*resolved)
        return acc, loss, finish

    def status_tensor(acc, loss):
        raise AssertionError("per-view status tensors must not be built inside a render group")

    original_packet = executor.batched_status_packet

    def batched_packet(entries):
        assert stack[-1:] == ["m_loss_numerical_gate"]
        calls.append(("packet", None))
        return original_packet(entries)

    def resolve(acc, loss, iteration, *, status=None):
        assert stack[-1:] == ["m_loss_assembly"]
        assert status is not None
        calls.append(("replay", None))
        return original_resolve(acc, loss, iteration, status=status)

    def tolist(tensor):
        assert stack[-1:] == ["m_loss_numerical_gate"]
        calls.append(("readback", None))
        return original_tolist(tensor)

    monkeypatch.setattr(executor._objective, "prepare_loss", prepare)
    monkeypatch.setattr(GuardedLossAccumulator, "status_tensor", status_tensor)
    monkeypatch.setattr(executor, "batched_status_packet", batched_packet)
    monkeypatch.setattr(GuardedLossAccumulator, "resolve", resolve)
    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    result = executor.run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), runtime,
        camera_microbatch_size=camera_count,
    )
    assert result.optimizer_step_count == 1
    for phase in ("construct", "finalize"):
        assert [view for name, view in calls if name == phase] == list(range(8))
    assert sum(name == "packet" for name, _ in calls) == 8 // camera_count
    assert sum(name == "replay" for name, _ in calls) == 8
    assert sum(name == "readback" for name, _ in calls) == 8 // camera_count
    assert scopes.count("m_loss_assembly") == 16
    assert scopes.count("m_loss_numerical_gate") == 8 // camera_count
    assert stack == []
