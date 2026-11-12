"""CPU-safe deterministic contract for one fixed-topology engine step.

The real renderer parity test is necessarily GPU-gated because gsplat has no
CPU rasterizer.  This companion harness keeps the engine's sequencing
contract executable in CPU-only CI: an inline-compatible Adam update and
``GaussianOptimizationEngine.run_step`` receive identical deterministic
render/loss inputs and must agree on loss, gradients, parameters, optimizer
state, returned RNG state, and fixed-topology version reporting.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from eligsir.optimization.gaussian import engine as engine_module
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationView,
    RandomStateToken,
    RenderOutputs,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown


class _ToyGaussians:
    """Minimal fixed-shape model with the fields the engine reads."""

    def __init__(self) -> None:
        self.params = torch.nn.ParameterDict(
            {"means": torch.nn.Parameter(torch.tensor([[0.25], [-0.5]]))}
        )
        self.provisional = torch.zeros(2, dtype=torch.bool)
        self.optimizer = torch.optim.Adam(self.params.values(), lr=0.1)


class _Session:
    def __init__(self, model: _ToyGaussians) -> None:
        self.model = model
        self.parameter_version = 7
        self.lr_updates: list[int] = []
        self.pre_step_gradient: torch.Tensor | None = None

    def update_learning_rate(self, iteration: int) -> float:
        self.lr_updates.append(iteration)
        return 0.1

    def step(self, visibility=None) -> None:
        assert visibility is None
        self.pre_step_gradient = self.model.params["means"].grad.detach().clone()
        self.model.optimizer.step()
        self.parameter_version += 1

    def zero_grad(self) -> None:
        self.model.optimizer.zero_grad(set_to_none=True)


class _Backend:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def begin_optimization(self) -> _Session:
        return self._session


@dataclass
class _Strategy:
    random_draws: list[float]

    def step_pre_backward(self, **_kwargs) -> None:
        self.random_draws.append(float(torch.rand(())))


class _ContinuousStrategy:
    def __init__(self) -> None:
        self.draws: dict[str, float] = {}

    def step_pre_backward(self, **_kwargs) -> None:
        self.draws["pre_backward"] = float(torch.rand(()))

    def inject_noise(self, **_kwargs) -> None:
        self.draws["engine_noise"] = float(torch.rand(()))

    def step_post_backward(self) -> None:
        self.draws["shared_tail"] = float(torch.rand(()))


def _render(model, *_args, **_kwargs) -> RenderOutputs:
    return RenderOutputs(
        payload={
            "render": model.params["means"],
            "visibility_filter": torch.tensor([True, True]),
        }
    )


def _assemble_loss(inputs) -> LossBreakdown:
    # Deliberately consume randomness before backward.  A broken RNG scope or
    # a reordered step therefore changes both the loss and the returned token.
    coefficient = torch.rand(()) + 0.5
    total = ((inputs.image * coefficient - 0.125) ** 2).mean()
    return LossBreakdown(
        total=total.detach(),
        components={"coefficient": coefficient.detach()},
        total_tensor=total,
        loss_finite=True,
    )


def _optimizer_state(model: _ToyGaussians) -> dict[str, torch.Tensor | None | int]:
    parameter = model.params["means"]
    state = model.optimizer.state.get(parameter, {})
    return {
        "exp_avg": state.get("exp_avg").detach().clone() if "exp_avg" in state else None,
        "exp_avg_sq": state.get("exp_avg_sq").detach().clone() if "exp_avg_sq" in state else None,
        "step": int(state["step"].item()) if "step" in state else 0,
    }


def _parameter_topology(model: _ToyGaussians) -> dict[str, tuple[int, tuple[int, ...], int]]:
    """Snapshot the fixed-shape facts mutation would change."""
    return {
        name: (id(parameter), tuple(parameter.shape), parameter.numel())
        for name, parameter in model.params.items()
    }


def _same_token(left: RandomStateToken, right: RandomStateToken) -> bool:
    return (
        left.torch_rng_state == right.torch_rng_state
        and left.cuda_rng_state == right.cuda_rng_state
    )


def _run_inline_compatible_step(
    session: _Session, token: RandomStateToken, strategy: _Strategy, iteration: int,
):
    """Independent reference for the fixed-topology ordering in the plan."""
    with token.scoped() as scope:
        session.update_learning_rate(iteration)
        rendered = _render(session.model)
        loss = _assemble_loss(
            type("Inputs", (), {"image": rendered.payload["render"]})()
        )
        strategy.step_pre_backward(
            gaussians=session.model,
            args=None,
            iteration=iteration,
            render_pkg=rendered.payload,
            loss=loss.total_tensor,
        )
        loss.total_tensor.backward()
        grads = session.model.params["means"].grad.detach().clone()
        session.step()
        session.zero_grad()
    return loss, grads, scope.returned


def test_cpu_engine_step_matches_inline_loss_grad_parameters_optimizer_rng_and_versions(monkeypatch):
    """The typed engine preserves the inline fixed-topology update contract."""
    token_source_seed = 1234
    torch.manual_seed(token_source_seed)
    reference_token = RandomStateToken.capture()
    engine_token = RandomStateToken(
        torch_rng_state=reference_token.torch_rng_state,
        cuda_rng_state=reference_token.cuda_rng_state,
    )
    torch.manual_seed(9999)
    reference_model = _ToyGaussians()
    reference_session = _Session(reference_model)
    reference_strategy = _Strategy([])
    reference_loss, reference_grads, reference_returned_token = _run_inline_compatible_step(
        reference_session, reference_token, reference_strategy, iteration=3,
    )
    reference_params = reference_model.params["means"].detach().clone()
    reference_optimizer_state = _optimizer_state(reference_model)

    engine_model = _ToyGaussians()
    engine_session = _Session(engine_model)
    engine = engine_module.GaussianOptimizationEngine(_Backend(engine_session))
    engine_strategy = _Strategy([])
    engine_topology_before = _parameter_topology(engine_model)
    monkeypatch.setattr(engine_module, "invoke_render", _render)
    monkeypatch.setattr(engine_module._objective, "assemble_loss", _assemble_loss)

    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=3,
            map_version=41,
            topology_version=9,
            resolution_bucket="cpu-contract",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=engine_token,
        ),
        OptimizationRuntimeContext(
            gaussians=engine_model,
            args=object(),
            opt=type("Opt", (), {"iterations": 10})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=engine_strategy,
        ),
    )

    assert _same_token(RandomStateToken.capture(), result.random_state_token)
    assert result.loss_breakdown.materialized_total() == reference_loss.materialized_total()
    assert result.loss_breakdown.materialized_components() == reference_loss.materialized_components()
    assert engine_strategy.random_draws == reference_strategy.random_draws
    assert result.random_state_token is not None
    assert _same_token(result.random_state_token, reference_returned_token)

    # ``zero_grad(set_to_none=True)`` is part of the engine contract.  The
    # session records the gradient just before Adam so the comparison remains
    # meaningful after the intentionally cleared live ``.grad`` fields.
    assert not torch.equal(reference_grads, torch.zeros_like(reference_grads))
    assert engine_session.pre_step_gradient is not None
    assert torch.equal(engine_session.pre_step_gradient, reference_grads)
    assert torch.equal(engine_model.params["means"].detach(), reference_params)
    engine_optimizer_state = _optimizer_state(engine_model)
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(engine_optimizer_state[key], reference_optimizer_state[key])
    assert engine_optimizer_state["step"] == reference_optimizer_state["step"] == 1
    assert engine_model.params["means"].grad is None

    # This is the runtime fixed-topology contract: not merely unchanged map
    # metadata, but the same parameter objects with the same shapes and
    # element counts after the engine's complete numerical update.
    assert _parameter_topology(engine_model) == engine_topology_before
    assert result.map_version_before == result.map_version_after == 41
    assert result.topology_version == 9
    assert result.parameter_version_before == 7
    assert result.parameter_version_after == 8
    assert engine_session.lr_updates == reference_session.lr_updates == [3]


def test_canonical_denied_then_permitted_matches_clean_permitted_update(monkeypatch):
    """A denied canonical attempt cannot leave gradients for its retry."""

    monkeypatch.setattr(engine_module, "invoke_render", _render)
    monkeypatch.setattr(engine_module._objective, "assemble_loss", _assemble_loss)
    torch.manual_seed(123)
    token = RandomStateToken.capture()

    def _request(iteration):
        return OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=iteration,
            map_version=0,
            topology_version=0,
            resolution_bucket="cpu-contract",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken(
                torch_rng_state=token.torch_rng_state,
                cuda_rng_state=token.cuda_rng_state,
            ),
        )

    def _runtime(session, *, permitted):
        return OptimizationRuntimeContext(
            session=session,
            gaussians=session.model,
            args=object(),
            opt=type("Opt", (), {"iterations": 1})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=_Strategy([]),
            optimizer_update_id=1,
            optimizer_update_permitted=permitted,
        )

    retried_model = _ToyGaussians()
    retried_session = _Session(retried_model)
    retried_engine = engine_module.GaussianOptimizationEngine(_Backend(retried_session))
    denied = retried_engine.run_batched_step(
        (_request(100), _request(100)),
        _runtime(retried_session, permitted=False),
    )
    retried = retried_engine.run_batched_step(
        (_request(101), _request(101)),
        _runtime(retried_session, permitted=True),
    )

    clean_model = _ToyGaussians()
    clean_session = _Session(clean_model)
    clean_engine = engine_module.GaussianOptimizationEngine(_Backend(clean_session))
    clean = clean_engine.run_batched_step(
        (_request(101), _request(101)),
        _runtime(clean_session, permitted=True),
    )

    assert denied.optimizer_step_count == 0
    assert denied.backward_ran is False
    assert retried.optimizer_step_count == clean.optimizer_step_count == 1
    torch.testing.assert_close(
        retried_session.pre_step_gradient,
        clean_session.pre_step_gradient,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        retried_model.params["means"], clean_model.params["means"], rtol=0, atol=0
    )


def test_engine_and_inline_share_one_continuous_iteration_rng_stream(monkeypatch):
    """Background, engine work, and the shared tail consume one sequence."""
    monkeypatch.setattr(engine_module, "invoke_render", _render)
    monkeypatch.setattr(engine_module._objective, "assemble_loss", _assemble_loss)

    torch.manual_seed(2026)
    inline_model = _ToyGaussians()
    inline_session = _Session(inline_model)
    inline_strategy = _ContinuousStrategy()
    inline_draws = {"background": float(torch.rand(()))}
    inline_session.update_learning_rate(3)
    inline_render = _render(inline_model)
    inline_loss = _assemble_loss(
        type("Inputs", (), {"image": inline_render.payload["render"]})()
    )
    inline_draws["loss"] = float(inline_loss.components["coefficient"])
    inline_strategy.step_pre_backward()
    inline_loss.total_tensor.backward()
    inline_session.step()
    inline_session.zero_grad()
    inline_strategy.inject_noise()
    inline_token_before_tail = RandomStateToken.capture()
    inline_strategy.step_post_backward()
    inline_draws.update(inline_strategy.draws)
    inline_params = inline_model.params["means"].detach().clone()

    torch.manual_seed(2026)
    engine_model = _ToyGaussians()
    engine_session = _Session(engine_model)
    engine = engine_module.GaussianOptimizationEngine(_Backend(engine_session))
    engine_strategy = _ContinuousStrategy()
    engine_draws = {"background": float(torch.rand(()))}
    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=3,
            map_version=0,
            topology_version=0,
            resolution_bucket="cpu-contract",
            evidence_requirements=EvidenceRequirements(),
            # The composition root snapshots after background selection.
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            gaussians=engine_model,
            args=object(),
            opt=type("Opt", (), {"iterations": 10})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=engine_strategy,
            fixed_shape_mcmc=True,
        ),
    )
    engine_draws["loss"] = float(result.loss_breakdown.components["coefficient"])
    engine_strategy.step_post_backward()
    engine_draws.update(engine_strategy.draws)

    assert engine_draws == inline_draws
    assert torch.equal(engine_model.params["means"].detach(), inline_params)
    assert _same_token(result.random_state_token, inline_token_before_tail)


def test_terminal_iteration_runs_backward_without_adam_or_fixed_shape_noise(monkeypatch):
    monkeypatch.setattr(engine_module, "invoke_render", _render)
    monkeypatch.setattr(engine_module._objective, "assemble_loss", _assemble_loss)
    model = _ToyGaussians()
    before = model.params["means"].detach().clone()
    session = _Session(model)
    strategy = _ContinuousStrategy()
    engine = engine_module.GaussianOptimizationEngine(_Backend(session))

    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=10,
            map_version=0,
            topology_version=0,
            resolution_bucket="terminal",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            gaussians=model,
            args=object(),
            opt=type("Opt", (), {"iterations": 10})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=strategy,
            fixed_shape_mcmc=True,
        ),
    )

    assert result.step_allowed is False
    assert result.skip_reason == "iteration_limit"
    assert result.backward_ran is True
    assert result.optimizer_step_ran is False
    assert result.mcmc_noise_ran is False
    assert "pre_backward" in strategy.draws
    assert "engine_noise" not in strategy.draws
    assert model.params["means"].grad is not None
    assert torch.equal(model.params["means"].detach(), before)
    assert session.parameter_version == 7


@pytest.mark.parametrize(
    ("skip_reason", "placement_only", "iteration", "loss_finite"),
    [
        ("placement_only", True, 1, True),
        ("non_finite_loss", False, 1, False),
        ("iteration_limit", False, 10, True),
    ],
)
def test_skipped_engine_results_do_not_run_the_mutating_post_mcmc_hook(
    monkeypatch, skip_reason, placement_only, iteration, loss_finite,
):
    monkeypatch.setattr(engine_module, "invoke_render", _render)

    def assemble(inputs):
        total = inputs.image.square().mean()
        if not loss_finite:
            total = total * float("nan")
        return LossBreakdown(
            total=total.detach(), components={}, total_tensor=total,
            loss_finite=loss_finite,
        )

    monkeypatch.setattr(engine_module._objective, "assemble_loss", assemble)
    model = _ToyGaussians()
    before = model.params["means"].detach().clone()
    engine = engine_module.GaussianOptimizationEngine(_Backend(_Session(model)))

    def mutating_sanitizer(_result) -> None:
        with torch.no_grad():
            model.params["means"].fill_(123.0)

    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=iteration,
            map_version=0,
            topology_version=0,
            resolution_bucket="skip",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            gaussians=model,
            args=object(),
            opt=type("Opt", (), {"iterations": 10})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=_ContinuousStrategy(),
            placement_only=placement_only,
            post_mcmc_hook=mutating_sanitizer,
        ),
    )

    assert result.skip_reason == skip_reason
    assert torch.equal(model.params["means"].detach(), before)


def test_engine_reports_granular_backward_optimizer_and_noise_timings(monkeypatch):
    monkeypatch.setattr(engine_module, "invoke_render", _render)
    monkeypatch.setattr(engine_module._objective, "assemble_loss", _assemble_loss)
    phases: list[str] = []

    @contextmanager
    def timed(name: str):
        phases.append(name)
        yield

    model = _ToyGaussians()
    engine = engine_module.GaussianOptimizationEngine(_Backend(_Session(model)))
    engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=object(), background=object()),
            iteration=1,
            map_version=0,
            topology_version=0,
            resolution_bucket="diagnostics",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            gaussians=model,
            args=object(),
            opt=type("Opt", (), {"iterations": 10})(),
            pipe=type("Pipe", (), {"gsplat_sparse_grad": False})(),
            mcmc_strategy=_ContinuousStrategy(),
            fixed_shape_mcmc=True,
            timing_scope=timed,
        ),
    )

    assert phases == [
        "m_render",
        "m_post_render",
        "m_loss_assembly",
        "m_backward",
        "m_grad_policies",
        "m_optstep",
        "m_noise",
    ]
