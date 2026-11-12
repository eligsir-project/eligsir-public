"""Reference-only contracts for the audited post-mapping pose implementation.

The production pose-align feature is intentionally absent from this main-based
checkout.  These tests load the immutable audited implementation directly from
its pinned Git object to preserve the geometry and warm-start oracle while the
owned-payload integration prerequisite is reviewed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


_PIN = "8064aca93886dd07ad6d890bccbdcda960a9a2af"
_FIXTURES = Path(__file__).with_name("fixtures")


def _oracle_module(path: str, name: str) -> types.ModuleType:
    source = (_FIXTURES / path).read_text()
    module = types.ModuleType(name)
    module.__file__ = f"git:{_PIN}:{path}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class _Camera:
    def __init__(self) -> None:
        self.world_view_transform = torch.tensor(
            [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.0, 1.0]]
        )
        self.projection_matrix = torch.eye(4)
        self.camera_center = torch.tensor([1.0, 2.0, 3.0])
        self.image_width = self.image_height = 2


def _render_camera_center(camera, *_args):
    return {"render": camera.camera_center.view(3, 1, 1).expand(3, 2, 2)}


def test_oracle_apply_pose_delta_matches_right_composed_nontrivial_transform():
    pose = _oracle_module("pose_opt_8064_oracle.py", "_legacy_pose_opt_geometry")
    camera = _Camera()
    delta6 = torch.tensor([0.0, 0.0, 0.3, 0.2, -0.4, 0.1])
    delta = pose.se3_delta_matrix(delta6)

    adjusted = pose.apply_pose_delta(camera, delta)

    base_w2c = camera.world_view_transform.transpose(0, 1)
    expected_c2w = torch.inverse(base_w2c) @ delta
    expected_w2c = torch.inverse(expected_c2w)
    torch.testing.assert_close(adjusted.world_view_transform, expected_w2c.transpose(0, 1))
    torch.testing.assert_close(adjusted.camera_center, expected_c2w[:3, 3])
    torch.testing.assert_close(
        adjusted.full_proj_transform,
        expected_w2c.transpose(0, 1).unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0)).squeeze(0),
    )


def test_shared_pose_core_matches_the_immutable_geometry_oracle():
    from eligsir.pose_opt import apply_pose_delta, se3_delta_matrix

    camera = _Camera()
    delta6 = torch.tensor([0.0, 0.0, 0.3, 0.2, -0.4, 0.1])
    adjusted = apply_pose_delta(camera, se3_delta_matrix(delta6))

    base_w2c = camera.world_view_transform.transpose(0, 1)
    expected_c2w = torch.inverse(base_w2c) @ _oracle_module(
        "pose_opt_8064_oracle.py", "_legacy_pose_opt_parity"
    ).se3_delta_matrix(delta6)
    torch.testing.assert_close(adjusted.world_view_transform, torch.inverse(expected_c2w).transpose(0, 1))
    torch.testing.assert_close(adjusted.camera_center, expected_c2w[:3, 3])


def test_shared_pose_core_masks_stale_cached_extrinsics_and_bakes_writeback():
    from eligsir.pose_opt import apply_pose_delta, bake_pose_delta_into_camera, se3_delta_matrix

    camera = _Camera()
    camera.c2w = torch.inverse(camera.world_view_transform.transpose(0, 1))
    camera._gsplat_viewmat = torch.zeros(4, 4)
    delta = se3_delta_matrix(torch.tensor([0.0, 0.0, 0.2, 0.1, 0.0, 0.0]))

    proxy = apply_pose_delta(camera, delta)
    with pytest.raises(AttributeError):
        proxy._gsplat_viewmat

    bake_pose_delta_into_camera(camera, delta)
    assert not hasattr(camera, "_gsplat_viewmat")
    torch.testing.assert_close(camera.camera_center, camera.c2w[:3, 3])


def test_oracle_fit_uses_provided_warm_start_before_first_of_fixed_steps(monkeypatch):
    pose = _oracle_module("pose_opt_8064_oracle.py", "_legacy_pose_opt_warm_start")
    original = sys.modules.get("eligsir.pose_opt")
    sys.modules["eligsir.pose_opt"] = pose
    try:
        fit = _oracle_module("eval_pose_align_8064_oracle.py", "_legacy_eval_pose_align")
    finally:
        if original is None:
            del sys.modules["eligsir.pose_opt"]
        else:
            sys.modules["eligsir.pose_opt"] = original

    camera = _Camera()
    cold_loss: dict[str, float] = {}
    warm_loss: dict[str, float] = {}
    common = dict(
        cam=camera,
        gaussians=object(),
        render_fn=_render_camera_center,
        pipe=object(),
        background=None,
        gt=torch.zeros(3, 2, 2),
        ssim_fn=lambda *_args: torch.zeros((), dtype=torch.float32),
        n_steps=2,
        lr=1e-2,
    )
    fit._fit_pose_delta(**common, loss_out=cold_loss)
    fit._fit_pose_delta(
        **common,
        init_delta6=torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
        loss_out=warm_loss,
    )

    assert cold_loss["initial"] != warm_loss["initial"]
    assert set(cold_loss) == {"initial", "final"}
    assert set(warm_loss) == {"initial", "final"}
