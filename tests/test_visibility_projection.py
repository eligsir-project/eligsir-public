"""Tests for fast GPU frustum-projection visibility (task 3)."""
from __future__ import annotations

import numpy as np
import torch

from eligsir.mapping.visibility import project_visible_gaussian_uids


class _FakeModel:
    def __init__(self, xyz: torch.Tensor, uids: torch.Tensor) -> None:
        self._xyz = xyz
        self.gaussian_uid = uids

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz


class _FakeCam:
    def __init__(self, R: np.ndarray, T: np.ndarray) -> None:
        self.R = R.astype(np.float32)
        self.T = T.astype(np.float32)
        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.image_width, self.image_height = 640, 480


def _identity_cam() -> _FakeCam:
    # Camera at world origin looking down +Z (R_c2w = I means camera axes
    # align with world axes; t_w2c = 0 means camera is at the world origin).
    return _FakeCam(np.eye(3), np.zeros(3))


def test_points_in_front_and_in_frame_are_visible():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], device="cpu")
    uids = torch.tensor([100, 101], dtype=torch.int64)
    model = _FakeModel(xyz, uids)
    visible = project_visible_gaussian_uids(model, _identity_cam())
    assert set(visible.tolist()) == {100, 101}


def test_point_behind_camera_is_excluded():
    xyz = torch.tensor([[0.0, 0.0, -2.0]], device="cpu")
    uids = torch.tensor([200], dtype=torch.int64)
    model = _FakeModel(xyz, uids)
    visible = project_visible_gaussian_uids(model, _identity_cam())
    assert visible.numel() == 0


def test_point_outside_image_bounds_is_excluded():
    # Far off to the side: at z=2, x=100 projects way outside a 640-wide image.
    xyz = torch.tensor([[100.0, 0.0, 2.0]], device="cpu")
    uids = torch.tensor([300], dtype=torch.int64)
    model = _FakeModel(xyz, uids)
    visible = project_visible_gaussian_uids(model, _identity_cam())
    assert visible.numel() == 0


def test_point_beyond_far_plane_is_excluded():
    xyz = torch.tensor([[0.0, 0.0, 500.0]], device="cpu")
    uids = torch.tensor([400], dtype=torch.int64)
    model = _FakeModel(xyz, uids)
    visible = project_visible_gaussian_uids(model, _identity_cam(), far=20.0)
    assert visible.numel() == 0


def test_rotated_translated_camera_transforms_correctly():
    # Camera rotated 90 degrees about Y (so its forward axis, world -Z in
    # camera-local terms, now points along world +X) with its center at
    # world (5, 0, 0). Under R_c2w = I / t_w2c = 0 (every other test here),
    # several plausible transpose/sign bugs (R_c2w vs R_c2w.T, t_c2w vs
    # t_w2c) would produce results indistinguishable from the correct
    # formula -- this fixture actually exercises the rotation/translation.
    R_c2w = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    t_c2w = np.array([5.0, 0.0, 0.0])
    t_w2c = -R_c2w.T @ t_c2w
    cam = _FakeCam(R_c2w, t_w2c)

    # 2m ahead of the camera along its forward axis (world +X from (5,0,0)).
    ahead_point = torch.tensor([[7.0, 0.0, 0.0]], device="cpu")
    # 2m behind the camera (world -X from (5,0,0)) -- must be excluded.
    behind_point = torch.tensor([[3.0, 0.0, 0.0]], device="cpu")

    visible_ahead = project_visible_gaussian_uids(
        _FakeModel(ahead_point, torch.tensor([500], dtype=torch.int64)), cam
    )
    visible_behind = project_visible_gaussian_uids(
        _FakeModel(behind_point, torch.tensor([600], dtype=torch.int64)), cam
    )
    assert visible_ahead.tolist() == [500]
    assert visible_behind.numel() == 0
