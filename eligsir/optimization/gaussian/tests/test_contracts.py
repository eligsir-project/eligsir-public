"""Smoke tests for eligsir.optimization.gaussian.contracts.

Not exhaustive — just asserts the dataclass shapes are constructible and
carry the fields the module docstring/plan section 7.3 promise. Real
behavioral tests (parity, gradient-policy ordering) belong to later
subagents in PR W3-02 per the plan's section 7.5/7.6.
"""

from eligsir.optimization.gaussian.contracts import (
    ActiveSetState,
    AnchorScheduleState,
    AnchorState,
    BubbleState,
    DepthLockState,
    EvidenceRequirements,
    GradientPolicy,
    GradientPolicyBase,
    GradientPolicyContext,
    GradientPolicyResult,
    LossBreakdown,
    OptimizationEvidence,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationRuntimeContext,
    OptimizationView,
    RandomStateToken,
    RenderOutputs,
    StableGeometryState,
)


def test_random_state_token_scopes_and_returns_cpu_rng_state():
    """The engine must not leak its stochastic step into process-global RNG."""
    import torch

    torch.manual_seed(17)
    token = RandomStateToken.capture()
    expected = torch.rand(3)
    torch.manual_seed(999)
    with token.scoped() as scope:
        assert torch.equal(torch.rand(3), expected)
        torch.rand(2)
        expected_returned_state = torch.get_rng_state().clone()
    outer = RandomStateToken.capture()
    scope.returned.restore()
    assert torch.equal(torch.get_rng_state(), expected_returned_state)
    outer.restore()


def test_view_and_runtime_context_own_the_engine_inputs():
    view = OptimizationView(
        camera="camera", background="background", update_sh_rest=False,
        render_depth=True, resolution_bucket="half",
    )
    runtime = OptimizationRuntimeContext(
        gaussians="model", args="args", opt="opt", pipe="pipe",
        mcmc_strategy="mcmc", gradient_policy_context="policies",
    )
    assert view.camera == "camera"
    assert runtime.gaussians == "model"


def test_engine_rejects_runtime_session_or_model_divergence():
    """A typed step must not backprop one model and step another session."""
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    class Session:
        def __init__(self, model):
            self.model = model

        def zero_grad(self): pass
        def step(self, visibility=None): pass
        def update_learning_rate(self, iteration): pass

    owner = Session("owner-model")
    engine = GaussianOptimizationEngine(type("Backend", (), {
        "begin_optimization": lambda self: owner,
    })())
    runtime = OptimizationRuntimeContext(session=Session("other-model"))
    try:
        engine._resolve_runtime(runtime)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_step_applies_fixed_shape_noise_only_after_an_allowed_step(monkeypatch):
    """A fixed-topology engine step owns MCMC noise and uses its session LR.

    Regression target: leaving noise in the streaming loop means a caller can
    run the typed optimizer without the matching stochastic update, or inject
    noise after a non-finite loss that deliberately skipped Adam.
    """
    import torch

    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.contracts import (
        EvidenceRequirements,
        OptimizationRuntimeContext,
        OptimizationStepRequest,
        OptimizationView,
        RandomStateToken,
    )
    from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
    from eligsir.optimization.gaussian.contracts import RenderOutputs

    class Model:
        provisional = torch.tensor([False])

    model = Model()

    class Session:
        def __init__(self, model):
            self.model = model
            self.parameter_version = 0

        def update_learning_rate(self, _iteration):
            return 0.125

        def step(self, visibility=None):
            self.parameter_version += 1

        def zero_grad(self):
            pass

    session = Session(model)
    engine = GaussianOptimizationEngine(type("Backend", (), {
        "begin_optimization": lambda self: session,
    })())

    monkeypatch.setattr(
        "eligsir.optimization.gaussian.engine.invoke_render",
        lambda *args, **kwargs: RenderOutputs(payload={"render": torch.tensor(1.0), "visibility_filter": torch.tensor([True])}),
    )
    monkeypatch.setattr(
        "eligsir.optimization.gaussian.engine._objective.assemble_loss",
        lambda _inputs: LossBreakdown(total=torch.tensor(1.0), total_tensor=torch.tensor(1.0, requires_grad=True)),
    )
    monkeypatch.setattr(engine, "render_and_backward", lambda *args, **kwargs: None)

    from eligsir.optimization.gaussian.contracts import OptimizationStepResult
    monkeypatch.setattr(
        engine,
        "run_post_backward_step",
        lambda *args, **kwargs: OptimizationStepResult(
            loss_breakdown=kwargs["loss_breakdown"], render_outputs=kwargs["render_outputs"],
            evidence=None, optimization_gpu_ms=None, map_version_before=0,
            map_version_after=0, topology_version=0,
        ),
    )

    class Strategy:
        def __init__(self):
            self.noise_lrs = []

        def inject_noise(self, **kwargs):
            self.noise_lrs.append(kwargs["xyz_lr"])

    strategy = Strategy()
    post_mcmc_results = []
    request = OptimizationStepRequest(
        view=OptimizationView(camera=object(), background=object()), iteration=7,
        map_version=0, topology_version=0, resolution_bucket="default",
        evidence_requirements=EvidenceRequirements(), random_state_token=RandomStateToken.capture(),
    )
    result = engine.run_step(request, OptimizationRuntimeContext(
        gaussians=model, args=object(), opt=object(), pipe=object(),
        mcmc_strategy=strategy, fixed_shape_mcmc=True,
        post_mcmc_hook=post_mcmc_results.append,
    ))

    assert strategy.noise_lrs == [0.125]
    assert result.backward_ran is True
    assert result.optimizer_step_ran is True
    assert result.mcmc_noise_ran is True
    assert post_mcmc_results == [result]


def test_run_step_does_not_step_past_the_configured_iteration_limit(monkeypatch):
    """Terminal backward runs, while the guarded optimizer step does not."""
    from types import SimpleNamespace

    from eligsir.optimization.gaussian import objective as objective_mod
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    class Model:
        provisional = False

    model = Model()

    class Session:
        def __init__(self):
            self.model = model
            self.parameter_version = 3

        def update_learning_rate(self, iteration):
            return 0.1

        def step(self, visibility=None):
            raise AssertionError("optimizer step must not run past the iteration limit")

        def zero_grad(self):
            raise AssertionError("zero_grad must not run past the iteration limit")

    class Backend:
        def begin_optimization(self):
            return Session()

    engine = GaussianOptimizationEngine(Backend())
    rendered = {"render": object(), "visibility_filter": None}
    monkeypatch.setattr(
        "eligsir.optimization.gaussian.engine.invoke_render",
        lambda *args, **kwargs: RenderOutputs(payload=rendered),
    )
    monkeypatch.setattr(
        objective_mod,
        "assemble_loss",
        lambda inputs: LossBreakdown(
            total=1.0, components={}, total_tensor=object(), loss_finite=True,
        ),
    )
    calls = []
    monkeypatch.setattr(engine, "render_and_backward", lambda *args, **kwargs: calls.append("backward"))
    monkeypatch.setattr(engine, "run_gradient_policies", lambda *args, **kwargs: calls.append("policies"))

    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=5, map_version=2, topology_version=4,
            resolution_bucket="default", evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            session=engine._session, gaussians=model, args=SimpleNamespace(),
            opt=SimpleNamespace(iterations=5), pipe=SimpleNamespace(),
            mcmc_strategy=None,
        ),
    )

    assert result.step_allowed is False
    assert result.skip_reason == "iteration_limit"
    assert result.backward_ran is True
    assert result.optimizer_step_ran is False
    assert calls == ["backward", "policies"]
    assert result.parameter_version_before == result.parameter_version_after == 3


def test_run_step_runs_the_streaming_post_render_hook_before_loss_assembly(monkeypatch):
    """Render-derived streaming state must be ready before the objective reads it."""
    from types import SimpleNamespace

    from eligsir.optimization.gaussian import objective as objective_mod
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    class Model:
        provisional = False

    model = Model()

    class Session:
        def __init__(self):
            self.model = model
            self.parameter_version = 0

        def update_learning_rate(self, iteration):
            return 0.1

    session = Session()
    engine = GaussianOptimizationEngine(type("Backend", (), {
        "begin_optimization": lambda self: session,
    })())
    events = []
    rendered = {"render": object(), "visibility_filter": None}
    monkeypatch.setattr(
        "eligsir.optimization.gaussian.engine.invoke_render",
        lambda *args, **kwargs: RenderOutputs(payload=rendered),
    )
    monkeypatch.setattr(
        objective_mod,
        "assemble_loss",
        lambda inputs: (
            events.append("loss"),
            LossBreakdown(total=1.0, components={}, total_tensor=None, loss_finite=False),
        )[1],
    )

    engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()), iteration=1,
            map_version=0, topology_version=0, resolution_bucket="default",
            evidence_requirements=EvidenceRequirements(), random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            session=session, gaussians=model, args=SimpleNamespace(),
            opt=SimpleNamespace(iterations=2), pipe=SimpleNamespace(),
            mcmc_strategy=None,
            post_render_hook=lambda outputs: events.append(("post_render", outputs.payload)),
        ),
    )

    assert events == [("post_render", rendered), "loss"]


def test_engine_exposes_its_backend_optimization_session():
    """Composition roots must not reach through a private engine field."""
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    session = object()
    engine = GaussianOptimizationEngine(type("Backend", (), {
        "begin_optimization": lambda self: session,
    })())

    assert engine.session is session


def test_optimization_step_request_constructs():
    req = OptimizationStepRequest(
        view=OptimizationView(payload="fake_view"),
        iteration=42,
        map_version=1,
        topology_version=1,
        resolution_bucket="full",
        evidence_requirements=EvidenceRequirements(want_energy_grad_snapshot=True),
        random_state_token=RandomStateToken(torch_rng_state=b"abc"),
    )
    assert req.iteration == 42
    assert req.evidence_requirements.want_energy_grad_snapshot is True
    # frozen dataclass: mutation should fail
    try:
        req.iteration = 43  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_optimization_step_result_constructs():
    result = OptimizationStepResult(
        loss_breakdown=LossBreakdown(total=1.23, components={"l1": 1.0, "ssim": 0.23}),
        render_outputs=RenderOutputs(payload="fake_render_pkg"),
        evidence=OptimizationEvidence(payload={"energy_grad": None}),
        optimization_gpu_ms=12.5,
        map_version_before=1,
        map_version_after=2,
        topology_version=1,
    )
    assert result.map_version_after == result.map_version_before + 1
    assert result.loss_breakdown.components["l1"] == 1.0


def test_optimization_step_result_evidence_may_be_none():
    result = OptimizationStepResult(
        loss_breakdown=LossBreakdown(),
        render_outputs=RenderOutputs(),
        evidence=None,
        optimization_gpu_ms=None,
        map_version_before=0,
        map_version_after=0,
        topology_version=0,
    )
    assert result.evidence is None


def test_gradient_policy_context_default_construction():
    context = GradientPolicyContext(
        gaussians="fake_gaussians",
        args="fake_args",
        iteration=100,
        n_frames_ingested=10,
    )
    # Every sub-state defaults to a disabled/no-op shape.
    assert context.active_set.sparse_active_set is False
    assert context.stable_geometry.freeze_old_geometry is False
    assert context.stable_geometry.bubble.enabled is False
    assert context.anchor.bootstrap.weight == 0.0
    assert context.anchor.insertion.weight == 0.0
    assert context.depth_lock.active is False


def test_gradient_policy_context_representative_construction():
    """Construct with representative non-default values matching a real
    train_streaming.py cascade config, to sanity-check the field grouping
    is actually usable end-to-end.
    """
    context = GradientPolicyContext(
        gaussians="fake_gaussians",
        args="fake_args",
        iteration=5000,
        n_frames_ingested=120,
        active_set=ActiveSetState(
            sparse_active_set=True,
            zero_invisible_grads=True,
            visible_mask="fake_bool_tensor",
        ),
        stable_geometry=StableGeometryState(
            freeze_old_geometry=True,
            freeze_new_frame_steps=30,
            iter_since_new_frame=5,
            young_age_frames=15,
            lifecycle_enabled=True,
            bubble=BubbleState(enabled=True),
        ),
        anchor=AnchorState(
            bootstrap=AnchorScheduleState(weight=0.5, decay_steps=2000, full_n_draw=True),
            insertion=AnchorScheduleState(weight=1.0, decay_steps=200, full_n_draw=True),
        ),
        depth_lock=DepthLockState(active=True),
    )
    assert context.active_set.sparse_active_set is True
    assert context.stable_geometry.bubble.enabled is True
    assert context.anchor.bootstrap.decay_steps == 2000
    assert context.anchor.insertion.decay_steps == 200
    assert context.depth_lock.active is True


def test_gradient_policy_result_shape():
    result = GradientPolicyResult(policy_name="stable_geometry_freeze", applied=True, notes="froze 42 rows")
    assert result.policy_name == "stable_geometry_freeze"
    assert result.applied is True


class _FakePolicy(GradientPolicyBase):
    name = "fake_policy"

    def apply(self, context: GradientPolicyContext) -> GradientPolicyResult:
        return GradientPolicyResult(policy_name=self.name, applied=True)


def test_gradient_policy_base_subclass_satisfies_protocol():
    policy = _FakePolicy()
    assert isinstance(policy, GradientPolicy)
    context = GradientPolicyContext(
        gaussians=None, args=None, iteration=0, n_frames_ingested=0,
    )
    result = policy.apply(context)
    assert result.applied is True
