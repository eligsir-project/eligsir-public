"""CPU parity contracts for fixed-base pose-fit geometry reuse."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _Camera:
    image_width = image_height = 2
    fx = fy = cx = cy = 1.0

    def __init__(self) -> None:
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        c2w[:3, 3] = torch.tensor((0.4, -0.3, 0.2))
        self.world_view_transform = torch.inverse(c2w).transpose(0, 1).contiguous()
        self.projection_matrix = torch.eye(4)
        self.camera_center = c2w[:3, 3].clone()
        self.gsplat = SimpleNamespace(K=torch.eye(3), width=2, height=2)


class _Gaussians:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.Adam((self.parameter,), lr=1e-3)


def _render(camera, *_args):
    values = torch.cat((camera.camera_center, camera.world_view_transform[:3, 0]))
    return {"render": values.view(6, 1, 1)}


def _fit(camera, gaussians, *, loss_out):
    from eligsir.utils.eval_pose_align import _fit_pose_delta, _frozen_gaussians

    frozen = []

    def render(camera, *args):
        frozen.append(gaussians.parameter.requires_grad)
        return _render(camera, *args)

    with _frozen_gaussians(gaussians):
        delta = _fit_pose_delta(
            camera, gaussians, render, object(), None, torch.zeros(6, 1, 1),
            lambda *_args: torch.zeros(()), n_steps=50, lr=1e-2, loss_out=loss_out,
        )
    assert frozen and not any(frozen)
    return delta


def test_fifty_step_fit_reuses_the_fixed_base_camera_inverse(monkeypatch):
    import eligsir.utils.eval_pose_align as fit

    camera = _Camera()
    gaussians = _Gaussians()
    original_inverse = torch.inverse
    base_storage = camera.world_view_transform.untyped_storage().data_ptr()
    base_inverse_calls = 0

    def counted_inverse(value):
        nonlocal base_inverse_calls
        if value.untyped_storage().data_ptr() == base_storage:
            base_inverse_calls += 1
        return original_inverse(value)

    monkeypatch.setattr(torch, "inverse", counted_inverse)
    _fit(camera, gaussians, loss_out={})

    assert base_inverse_calls == 1


def test_inference_mode_base_geometry_falls_back_to_uncached_fit():
    camera = _Camera()
    with torch.inference_mode():
        camera.world_view_transform = camera.world_view_transform.clone()

    delta = _fit(camera, _Gaussians(), loss_out={})

    assert torch.isfinite(delta).all()


def test_noninference_geometry_metadata_error_propagates():
    from eligsir.utils.eval_pose_align import _fixed_camera_geometry

    class _BrokenGeometry:
        def is_inference(self):
            return False

        def data_ptr(self):
            return 1

        @property
        def _version(self):
            raise RuntimeError("unexpected geometry metadata failure")

        def transpose(self, *_args):
            return torch.eye(4)

    with pytest.raises(RuntimeError, match="unexpected geometry metadata failure"):
        _fixed_camera_geometry(SimpleNamespace(world_view_transform=_BrokenGeometry()))


def test_cached_geometry_matches_rebuilding_each_step_for_losses_and_final_delta(monkeypatch):
    import eligsir.utils.eval_pose_align as fit

    cached_losses, rebuilt_losses = {}, {}
    cached = _fit(_Camera(), _Gaussians(), loss_out=cached_losses)
    original_geometry = fit._fixed_camera_geometry
    monkeypatch.setattr(fit, "_fixed_camera_geometry", lambda camera, _cached=None: original_geometry(camera))
    rebuilt = _fit(_Camera(), _Gaussians(), loss_out=rebuilt_losses)

    torch.testing.assert_close(cached, rebuilt, rtol=0, atol=0)
    assert cached_losses == rebuilt_losses


def test_fixed_geometry_invalidates_only_on_base_pose_mutation_and_keeps_live_calibration():
    from eligsir.pose_opt import apply_pose_delta, se3_delta_matrix
    from eligsir.utils.eval_pose_align import _fixed_camera_geometry

    camera = _Camera()
    geometry = _fixed_camera_geometry(camera)
    delta6 = torch.tensor((0.0, 0.0, 0.2, 0.1, -0.05, 0.02), requires_grad=True)
    adjusted = apply_pose_delta(camera, se3_delta_matrix(delta6), base_c2w=geometry[1])
    (adjusted.camera_center.square().sum() + adjusted.full_proj_transform.square().sum()).backward()
    assert delta6.grad is not None and torch.isfinite(delta6.grad).all()

    with torch.no_grad():
        camera.projection_matrix[0, 0] = 2.0
    assert _fixed_camera_geometry(camera, geometry) is geometry
    camera.gsplat.K[0, 0] += 1.0
    camera.image_width = 3
    adjusted = apply_pose_delta(camera, se3_delta_matrix(delta6.detach()), base_c2w=geometry[1])
    torch.testing.assert_close(
        adjusted.full_proj_transform,
        adjusted.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0)).squeeze(0),
    )
    torch.testing.assert_close(adjusted.gsplat.K, camera.gsplat.K)
    with torch.no_grad():
        camera.world_view_transform[0, 0] += 0.1
    assert _fixed_camera_geometry(camera, geometry) is not geometry
