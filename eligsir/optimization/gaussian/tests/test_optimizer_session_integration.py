"""Path C parity: ``GaussianOptimizationEngine``'s orchestration methods.

Wave 3, PR W3-02, Subagent D scope. Proves that driving the SAME toy
fixture used by ``test_gpu_parity_harness.py`` through
``GaussianOptimizationEngine.run_gradient_policies`` +
``GaussianOptimizationEngine.run_optimizer_step`` (equivalently,
``run_post_backward_step``) -- i.e. going through the engine's own
post-backward orchestration rather than calling the gradient policies and
``gaussians.optimizer.step()``/``normalize_rotation_params()``/
``zero_grad()`` directly, as ``test_gpu_parity_harness.py``'s "Path B" does
-- produces IDENTICAL numerics (tensors, gradients, optimizer state,
parameters after the step) to "Path A" (the hand-transcribed reference) and
"Path B" (the extracted-but-manually-sequenced pieces), within the same
tolerances already established by ``test_gpu_parity_harness.py``.

Reuses the exact fixture-construction helpers from that module (same
package) rather than duplicating them.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import gsplat  # noqa: F401
    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "test_optimizer_session_integration requires CUDA/ROCm + gsplat (no "
        "CPU rendering path exists in this codebase); run inside the "
        "training container with GPU passthrough per CLAUDE.md's Running "
        "Commands section: docker compose run --rm train python -m pytest "
        "eligsir/optimization/gaussian/tests/ -q"
    ),
)

from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (  # noqa: E402
    _bg,
    _build_args,
    _build_camera,
    _build_gaussians,
    _run_path_a,
    _snapshot_adam_state,
    _snapshot_grads,
    _snapshot_params,
)


def _run_path_c(gaussians, cam, args, opt, pipe, iteration: int, gs_rand_seed: int):
    """Drives the fixture through ``GaussianOptimizationEngine``'s own
    orchestration methods (``run_gradient_policies`` + ``run_optimizer_step``)
    instead of calling the gradient policies / optimizer.step() /
    normalize_rotation_params() / zero_grad() directly (that's Path B).
    """
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.optimization.gaussian.contracts import (
        ActiveSetState,
        AnchorScheduleState,
        AnchorState,
        BubbleState,
        DepthLockState,
        GradientPolicyContext,
        StableGeometryState,
    )
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs, assemble_loss
    from eligsir.optimization.gaussian.render_port import invoke_render

    backend = GsplatMapBackend(gaussians)
    engine = GaussianOptimizationEngine(backend)

    render_outputs = invoke_render(gaussians, cam, pipe, _bg(), update_sh_rest=True, render_depth=False)
    image = render_outputs.payload["render"]

    inputs = LossAssemblyInputs(
        gaussians=gaussians, viewpoint_cam=cam, args=args, opt=opt, pipe=pipe,
        render_outputs=render_outputs, image=image, iteration=iteration,
        n_frames_ingested=0,
        sparse_active_set=True,
        use_depth_loss=False, use_energy_mcmc=False,
    )
    loss_breakdown = assemble_loss(inputs)
    loss_breakdown.total_tensor.backward()

    grads_before_masking = _snapshot_grads(gaussians)

    visible = render_outputs.payload["visibility_filter"].detach()
    context = GradientPolicyContext(
        gaussians=gaussians, args=args, iteration=iteration, n_frames_ingested=0,
        active_set=ActiveSetState(sparse_active_set=True, zero_invisible_grads=True, visible_mask=visible),
        stable_geometry=StableGeometryState(freeze_old_geometry=False, lifecycle_enabled=False, bubble=BubbleState(enabled=False)),
        anchor=AnchorState(
            bootstrap=AnchorScheduleState(weight=1.0, decay_steps=100, full_n_draw=True),
            insertion=AnchorScheduleState(weight=0.0, decay_steps=100, full_n_draw=True),
        ),
        depth_lock=DepthLockState(active=False),
    )

    torch.manual_seed(gs_rand_seed)  # same seed, same call position as Path A/B's torch.rand draw
    engine.run_gradient_policies(context)

    grads_after_masking = _snapshot_grads(gaussians)

    step_mask = visible | gaussians.provisional.detach()
    engine.run_optimizer_step(
        gaussians, args, pipe, context,
        optimizer_type="adam",
        sparse_active_set=True,
        step_mask=step_mask,
        visible=visible,
        use_energy_mcmc=False,
    )

    params_after_step = _snapshot_params(gaussians)
    return {
        # Materialization is deliberately at this test/report boundary, not
        # inside assemble_loss()'s hot forward path.
        "total_before_backward": loss_breakdown.materialized_total(),
        "grads_before_masking": grads_before_masking,
        "grads_after_masking": grads_after_masking,
        "params_after_step": params_after_step,
        "components": {
            # PR #39 review fix (P1): loss_breakdown.components now holds
            # detached tensors (not pre-materialized floats) to avoid a host
            # sync inside assemble_loss() itself -- materialize explicitly
            # here at the test's comparison boundary.
            "Ll1": loss_breakdown.materialized_components()["Ll1"],
            "ssim": loss_breakdown.materialized_components()["ssim"],
        },
    }


def test_engine_orchestration_matches_reference_tensors_grads_optimizer_state():
    """Path C (engine orchestration) vs. Path A (hand-transcribed
    reference), same tolerances as
    test_gpu_parity_harness.test_extracted_step_matches_reference_tensors_grads_optimizer_state.
    """
    args, opt, pipe, _ns = _build_args()
    iteration = 0
    gs_rand_seed = 42

    torch.manual_seed(1234)
    g_a = _build_gaussians(seed=0)
    g_a.training_setup(opt)
    cam_a = _build_camera(seed=1)
    result_a = _run_path_a(g_a, cam_a, args, opt, pipe, iteration=iteration, gs_rand_seed=gs_rand_seed)

    torch.manual_seed(1234)
    g_c = _build_gaussians(seed=0)
    g_c.training_setup(opt)
    cam_c = _build_camera(seed=1)
    result_c = _run_path_c(g_c, cam_c, args, opt, pipe, iteration=iteration, gs_rand_seed=gs_rand_seed)

    LOSS_TOL = 5e-4
    TENSOR_ATOL = 2e-4
    TENSOR_RTOL = 1e-3

    assert result_a["total_before_backward"] == pytest.approx(
        result_c["total_before_backward"], abs=LOSS_TOL
    ), "loss totals diverge beyond documented tolerance (Path A vs Path C)"
    for key in result_a["components"]:
        assert result_a["components"][key] == pytest.approx(
            result_c["components"][key], abs=LOSS_TOL
        ), f"loss component {key!r} diverges (Path A vs Path C)"

    for name in result_a["grads_before_masking"]:
        ga = result_a["grads_before_masking"][name]
        gc = result_c["grads_before_masking"][name]
        assert (ga is None) == (gc is None), f"grad presence mismatch for {name!r}"
        if ga is not None:
            torch.testing.assert_close(
                ga, gc, atol=TENSOR_ATOL, rtol=TENSOR_RTOL,
                msg=f"pre-masking grad mismatch for param {name!r} (Path A vs Path C)",
            )

    for name in result_a["grads_after_masking"]:
        ga = result_a["grads_after_masking"][name]
        gc = result_c["grads_after_masking"][name]
        assert (ga is None) == (gc is None), f"post-mask grad presence mismatch for {name!r}"
        if ga is not None:
            torch.testing.assert_close(
                ga, gc, atol=TENSOR_ATOL, rtol=TENSOR_RTOL,
                msg=f"post-masking grad mismatch for param {name!r} (Path A vs Path C)",
            )

    for name in result_a["params_after_step"]:
        pa = result_a["params_after_step"][name]
        pc = result_c["params_after_step"][name]
        torch.testing.assert_close(
            pa, pc, atol=TENSOR_ATOL, rtol=TENSOR_RTOL,
            msg=f"post-step parameter mismatch for param {name!r} (Path A vs Path C)",
        )

    state_a = _snapshot_adam_state(g_a)
    state_c = _snapshot_adam_state(g_c)
    for name in state_a:
        for key in ("exp_avg", "exp_avg_sq"):
            va = state_a[name][key]
            vc = state_c[name][key]
            assert (va is None) == (vc is None), f"optimizer state presence mismatch {name}/{key}"
            if va is not None:
                torch.testing.assert_close(
                    va, vc, atol=TENSOR_ATOL, rtol=TENSOR_RTOL,
                    msg=f"optimizer state {key!r} mismatch for param {name!r} (Path A vs Path C)",
                )
        assert state_a[name]["step"] == state_c[name]["step"], (
            f"optimizer step-count mismatch for {name!r} (Path A vs Path C)"
        )


def test_run_post_backward_step_returns_populated_result():
    """Smoke test for the higher-level ``run_post_backward_step`` wrapper:
    it should run without error and return an ``OptimizationStepResult``
    with the loss/render objects the caller passed through, unmutated
    map/topology versions (no mutation happens in this engine), and
    ``optimization_gpu_ms`` left as ``None`` (GPU timing wiring is a later
    PR's job per plan section 7's non-goals for this PR)."""
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.optimization.gaussian.contracts import (
        ActiveSetState,
        AnchorState,
        DepthLockState,
        GradientPolicyContext,
        StableGeometryState,
    )
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs, assemble_loss
    from eligsir.optimization.gaussian.render_port import invoke_render

    args, opt, pipe, _ns = _build_args()
    torch.manual_seed(1234)
    gaussians = _build_gaussians(seed=0)
    gaussians.training_setup(opt)
    cam = _build_camera(seed=1)

    backend = GsplatMapBackend(gaussians)
    engine = GaussianOptimizationEngine(backend)

    render_outputs = invoke_render(gaussians, cam, pipe, _bg(), update_sh_rest=True, render_depth=False)
    image = render_outputs.payload["render"]
    inputs = LossAssemblyInputs(
        gaussians=gaussians, viewpoint_cam=cam, args=args, opt=opt, pipe=pipe,
        render_outputs=render_outputs, image=image, iteration=0, n_frames_ingested=0,
        sparse_active_set=False, use_depth_loss=False, use_energy_mcmc=False,
    )
    loss_breakdown = assemble_loss(inputs)
    loss_breakdown.total_tensor.backward()

    visible = render_outputs.payload["visibility_filter"].detach()
    context = GradientPolicyContext(
        gaussians=gaussians, args=args, iteration=0, n_frames_ingested=0,
        active_set=ActiveSetState(sparse_active_set=False, visible_mask=visible),
        stable_geometry=StableGeometryState(),
        anchor=AnchorState(),
        depth_lock=DepthLockState(active=False),
    )

    result = engine.run_post_backward_step(
        gaussians, args, pipe, context,
        optimizer_type="adam", sparse_active_set=False, visible=visible,
        use_energy_mcmc=False,
        loss_breakdown=loss_breakdown, render_outputs=render_outputs,
        map_version=3, topology_version=7,
    )

    assert result.loss_breakdown is loss_breakdown
    assert result.render_outputs is render_outputs
    assert result.evidence is None
    assert result.optimization_gpu_ms is None
    assert result.map_version_before == 3
    assert result.map_version_after == 3
    assert result.topology_version == 7
