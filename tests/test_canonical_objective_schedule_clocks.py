from types import SimpleNamespace

import pytest
import torch

from eligsir.cli.parser import parse_training_args
from eligsir.optimization.gaussian.contracts import RenderOutputs
from eligsir.optimization.gaussian.objective import LossAssemblyInputs, assemble_loss


class _Gaussians:
    def __init__(self, n: int = 3):
        self._opacity = torch.zeros((n, 1), requires_grad=True)
        self.get_opacity = torch.sigmoid(self._opacity)
        self.get_scaling = torch.ones((n, 3), requires_grad=True)
        self.get_xyz = torch.zeros((n, 3))
        self.provisional = torch.zeros(n, dtype=torch.bool)


def _inputs(*, artifact_iteration: int, optimizer_update_id: int, gaussians=None):
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0
    camera = SimpleNamespace(original_image=torch.zeros((3, 16, 16)))
    model = _Gaussians() if gaussians is None else gaussians
    return LossAssemblyInputs(
        gaussians=model,
        viewpoint_cam=camera,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": torch.zeros((3, 16, 16), requires_grad=True),
                "visibility_filter": torch.ones(model.get_xyz.shape[0], dtype=torch.bool),
            }
        ),
        image=torch.zeros((3, 16, 16), requires_grad=True),
        iteration=artifact_iteration,
        optimizer_update_id=optimizer_update_id,
        n_frames_ingested=1,
    )


@pytest.mark.parametrize("artifact_iteration", [9, 10, 11])
def test_sparse_energy_and_effective_count_hold_semantic_clock_fixed(
    monkeypatch, artifact_iteration
):
    import eligsir.utils.compiled_kernels as kernels
    import eligsir.utils.energy_mcmc as energy

    inputs = _inputs(artifact_iteration=artifact_iteration, optimizer_update_id=7)
    inputs.sparse_active_set = True
    inputs.use_energy_mcmc = True
    inputs.args.sparse_energy_global_interval = 10
    calls = []
    monkeypatch.setattr(kernels, "active_reg_core", lambda *_args, **_kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        energy,
        "compute_effective_count_loss",
        lambda *_args, iteration, **_kwargs: (
            calls.append(iteration) or torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
    )
    monkeypatch.setattr(energy, "compute_opacity_entropy_loss", lambda *_args: torch.tensor(0.0))

    assemble_loss(inputs)

    assert calls == []


@pytest.mark.parametrize("artifact_iteration", [9, 11])
def test_effective_count_anneal_receives_semantic_update_id(
    monkeypatch, artifact_iteration
):
    import eligsir.utils.compiled_kernels as kernels
    import eligsir.utils.energy_mcmc as energy

    inputs = _inputs(artifact_iteration=artifact_iteration, optimizer_update_id=10)
    inputs.sparse_active_set = True
    inputs.use_energy_mcmc = True
    inputs.args.sparse_energy_global_interval = 10
    calls = []
    monkeypatch.setattr(kernels, "active_reg_core", lambda *_args, **_kwargs: torch.tensor(0.0))
    monkeypatch.setattr(
        energy,
        "compute_effective_count_loss",
        lambda *_args, iteration, **_kwargs: (
            calls.append(iteration) or torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ),
    )
    monkeypatch.setattr(energy, "compute_opacity_entropy_loss", lambda *_args: torch.tensor(0.0))

    assemble_loss(inputs)

    assert calls == [10]


@pytest.mark.parametrize("artifact_iteration", [19, 20, 21])
def test_appearance_only_cadence_holds_semantic_clock_fixed(
    monkeypatch, artifact_iteration
):
    from eligsir.optimization.gaussian import objective

    inputs = _inputs(artifact_iteration=artifact_iteration, optimizer_update_id=7)
    inputs.background = torch.zeros(3)
    inputs.args.streaming_appearance_only_enabled = True
    inputs.args.streaming_appearance_only_interval = 20
    inputs.args.streaming_appearance_only_weight = 1.0
    fake_depth = SimpleNamespace(
        to=lambda *_args, **_kwargs: fake_depth,
        squeeze=lambda *_args, **_kwargs: torch.ones((16, 16)),
    )
    monkeypatch.setattr(objective, "_get_sensor_depth", lambda *_args, **_kwargs: fake_depth)
    monkeypatch.setattr(
        objective,
        "_classify_view_rays",
        lambda *_args, **_kwargs: SimpleNamespace(kind_map=torch.zeros((16, 16), dtype=torch.int64)),
    )
    monkeypatch.setattr(objective, "geometry_reliability_map", lambda *_args, **_kwargs: torch.ones((16, 16)))
    monkeypatch.setattr(objective, "material_risk_mask", lambda *_args, **_kwargs: torch.ones((16, 16), dtype=torch.bool))
    calls = []
    monkeypatch.setattr(
        objective,
        "run_appearance_only_update",
        lambda *_args, **_kwargs: calls.append(True) or torch.tensor(0.0),
    )

    assemble_loss(inputs)

    assert calls == []


@pytest.mark.parametrize("artifact_iteration", [19, 21])
def test_appearance_only_cadence_runs_at_semantic_boundary(
    monkeypatch, artifact_iteration
):
    from eligsir.optimization.gaussian import objective

    inputs = _inputs(artifact_iteration=artifact_iteration, optimizer_update_id=20)
    inputs.background = torch.zeros(3)
    inputs.args.streaming_appearance_only_enabled = True
    inputs.args.streaming_appearance_only_interval = 20
    inputs.args.streaming_appearance_only_weight = 1.0
    fake_depth = SimpleNamespace(
        to=lambda *_args, **_kwargs: fake_depth,
        squeeze=lambda *_args, **_kwargs: torch.ones((16, 16)),
    )
    monkeypatch.setattr(objective, "_get_sensor_depth", lambda *_args, **_kwargs: fake_depth)
    monkeypatch.setattr(
        objective,
        "_classify_view_rays",
        lambda *_args, **_kwargs: SimpleNamespace(kind_map=torch.zeros((16, 16), dtype=torch.int64)),
    )
    monkeypatch.setattr(objective, "geometry_reliability_map", lambda *_args, **_kwargs: torch.ones((16, 16)))
    monkeypatch.setattr(objective, "material_risk_mask", lambda *_args, **_kwargs: torch.ones((16, 16), dtype=torch.bool))
    calls = []
    monkeypatch.setattr(
        objective,
        "run_appearance_only_update",
        lambda *_args, **_kwargs: calls.append(True) or torch.tensor(0.0),
    )

    assemble_loss(inputs)

    assert calls == [True]


@pytest.mark.parametrize("artifact_iteration", [99, 100, 101])
def test_knn_refresh_holds_semantic_clock_fixed(monkeypatch, artifact_iteration):
    import eligsir.utils.rocm_knn_fallback as knn

    model = _Gaussians(n=3)
    model._knn_reg_idx = torch.zeros((3, 1), dtype=torch.long)
    model._knn_reg_n = 3
    model._knn_reg_iter = 5
    inputs = _inputs(
        artifact_iteration=artifact_iteration,
        optimizer_update_id=7,
        gaussians=model,
    )
    inputs.args.knn_reg_weight = 1.0
    inputs.args.knn_reg_k = 1
    inputs.args.knn_reg_interval = 3
    calls = []
    monkeypatch.setattr(
        knn,
        "knn_self",
        lambda means, _k: (
            calls.append(True) or torch.zeros((means.shape[0], 1)),
            torch.zeros((means.shape[0], 1), dtype=torch.long),
        ),
    )

    assemble_loss(inputs)

    assert calls == []
    assert model._knn_reg_iter == 5


@pytest.mark.parametrize("artifact_iteration", [99, 101])
def test_knn_refresh_runs_at_semantic_boundary(monkeypatch, artifact_iteration):
    import eligsir.utils.rocm_knn_fallback as knn

    model = _Gaussians(n=3)
    model._knn_reg_idx = torch.zeros((3, 1), dtype=torch.long)
    model._knn_reg_n = 3
    model._knn_reg_iter = 5
    inputs = _inputs(
        artifact_iteration=artifact_iteration,
        optimizer_update_id=8,
        gaussians=model,
    )
    inputs.args.knn_reg_weight = 1.0
    inputs.args.knn_reg_k = 1
    inputs.args.knn_reg_interval = 3
    calls = []
    monkeypatch.setattr(
        knn,
        "knn_self",
        lambda means, _k: (
            calls.append(True) or torch.zeros((means.shape[0], 1)),
            torch.zeros((means.shape[0], 1), dtype=torch.long),
        ),
    )

    assemble_loss(inputs)

    assert calls == [True]
    assert model._knn_reg_iter == 8
