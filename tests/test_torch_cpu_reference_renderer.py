import types

import pytest

torch = pytest.importorskip("torch")

from gaussian_renderer import render


_SH_C0 = 0.28209479177387814


class _Camera:
    image_width = 16
    image_height = 16
    fx = 20.0
    fy = 20.0
    cx = 8.0
    cy = 8.0
    world_view_transform = torch.eye(4, dtype=torch.float32)


class _Gaussians:
    active_sh_degree = 0

    def __init__(self, means):
        means = torch.tensor(means, dtype=torch.float32)
        n = means.shape[0]
        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(torch.full((n, 3), -2.0))
        self.quats = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(n, 1)
        )
        self.opacity_logits = torch.nn.Parameter(torch.full((n,), 2.0))
        rgb = torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float32).repeat(n, 1)
        self.sh0 = torch.nn.Parameter(((rgb - 0.5) / _SH_C0)[:, None, :])

    @property
    def get_xyz(self):
        return self.means

    @property
    def get_scaling(self):
        return torch.exp(self.log_scales)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self.opacity_logits)[:, None]

    @property
    def _features_dc(self):
        return self.sh0


def _pipe(**kwargs):
    values = {
        "renderer_backend": "auto",
        "near_plane": 0.01,
        "far_plane": 10.0,
        "eps2d": 0.3,
        "cpu_reference_max_interactions": 1_000_000,
    }
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def test_auto_dispatch_renders_cpu_gaussian_without_importing_gpu_backend():
    gaussians = _Gaussians([[0.0, 0.0, 2.0]])
    # Pass scaling_modifier positionally to lock the legacy renderer API.
    out = render(
        _Camera(),
        gaussians,
        _pipe(),
        torch.zeros(3),
        1.0,
        render_depth=True,
    )

    assert out["render"].shape == (3, 16, 16)
    assert out["alpha"].shape == (1, 16, 16)
    assert out["rendered_depth"].shape == (1, 16, 16)
    assert torch.isfinite(out["render"]).all()
    assert torch.isfinite(out["rendered_depth"]).all()
    assert out["alpha"][0, 8, 8] > 0.1
    assert out["render"][0, 8, 8] > out["render"][1, 8, 8]
    assert bool(out["visibility_filter"][0])


def test_reference_renderer_is_differentiable_on_cpu():
    gaussians = _Gaussians([[0.10, 0.0, 2.0]])
    out = render(
        _Camera(),
        gaussians,
        _pipe(renderer_backend="torch_cpu_reference"),
        torch.zeros(3),
        override_color=torch.tensor([[1.0, 0.0, 0.0]]),
    )

    loss = -out["render"][0, 8, 10]
    loss.backward()

    assert gaussians.means.grad is not None
    assert gaussians.opacity_logits.grad is not None
    assert torch.isfinite(gaussians.means.grad).all()
    assert torch.isfinite(gaussians.opacity_logits.grad).all()
    assert gaussians.means.grad.abs().sum() > 0
    assert gaussians.opacity_logits.grad.abs().sum() > 0


def test_expected_depth_stays_between_two_overlapping_gaussians():
    gaussians = _Gaussians([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    out = render(
        _Camera(),
        gaussians,
        _pipe(),
        torch.zeros(3),
        render_depth=True,
    )

    depth = float(out["rendered_depth"][0, 8, 8])
    assert 2.0 <= depth <= 4.0


def test_reference_backend_refuses_large_accidental_workloads():
    gaussians = _Gaussians([[0.0, 0.0, 2.0]])
    with pytest.raises(RuntimeError, match="tiny correctness backend"):
        render(
            _Camera(),
            gaussians,
            _pipe(cpu_reference_max_interactions=10),
            torch.zeros(3),
        )
