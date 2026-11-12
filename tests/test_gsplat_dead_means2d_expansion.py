"""Regression for dead dense means2d expansion after gsplat rasterization."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch


class _Camera:
    image_width = 3
    image_height = 2

    def __init__(self) -> None:
        self.world_view_transform = torch.eye(4)
        self.projection_matrix = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.gsplat = SimpleNamespace(
            viewmat=torch.eye(4),
            K=torch.eye(3),
            camera_center=torch.zeros(1, 3),
            width=self.image_width,
            height=self.image_height,
        )


class _Gaussians:
    active_sh_degree = 0

    def __init__(self, count: int = 5) -> None:
        self._xyz = torch.zeros(count, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.ones(self._xyz.shape[0], 3)

    @property
    def get_rotation(self):
        values = torch.zeros(self._xyz.shape[0], 4)
        values[:, 0] = 1.0
        return values

    @property
    def get_opacity(self):
        return torch.ones(self._xyz.shape[0], 1)


def _install_fake_rasterization(monkeypatch):
    def rasterization(**kwargs):
        colors = kwargs["viewmats"].sum().expand(1, 2, 3, 3)
        alphas = torch.ones(1, 2, 3, 1)
        return colors, alphas, {
            "gaussian_ids": torch.tensor([0, 3]),
            "radii": torch.tensor([[2, 1], [3, 4]]),
            "means2d": torch.tensor([[0.5, 0.5], [1.5, 0.5]]),
        }

    rendering = ModuleType("gsplat.rendering")
    rendering.rasterization = rasterization
    gsplat = ModuleType("gsplat")
    gsplat.rendering = rendering
    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)


def test_default_render_preserves_outputs_and_pose_gradients_without_dense_means2d(monkeypatch):
    """Fail if the unreturned ``means2d[N,2]`` compatibility expansion returns."""
    from eligsir.mapping.renderer import gsplat_backend
    from eligsir.pose_opt import apply_pose_delta, se3_delta_matrix

    _install_fake_rasterization(monkeypatch)
    zero_shapes: list[tuple[int, ...]] = []
    real_zeros = torch.zeros

    def tracked_zeros(*shape, **kwargs):
        zero_shapes.append(tuple(shape))
        return real_zeros(*shape, **kwargs)

    monkeypatch.setattr(gsplat_backend.torch, "zeros", tracked_zeros)
    camera, gaussians = _Camera(), _Gaussians()
    delta6 = torch.zeros(6, requires_grad=True)
    adjusted = apply_pose_delta(camera, se3_delta_matrix(delta6))
    result = gsplat_backend.render(
        adjusted,
        gaussians,
        SimpleNamespace(sh_impl="python", render_mode="RGB"),
        torch.zeros(3),
        override_color=torch.ones(5, 3),
    )

    assert {"render", "render_full", "radii", "is_used", "visibility_filter", "viewspace_points", "meta"} <= result.keys()
    assert result["radii"].tolist() == [2, 0, 0, 4, 0]
    assert result["is_used"].tolist() == [True, False, False, True, False]
    assert torch.equal(result["meta"]["means2d"], torch.tensor([[0.5, 0.5], [1.5, 0.5]]))
    torch.testing.assert_close(result["render"], torch.full((3, 2, 3), 4.0))
    result["render"].sum().backward()
    assert delta6.grad is not None and torch.isfinite(delta6.grad).all()
    assert (5, 2) not in zero_shapes
