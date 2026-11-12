"""Required excerpt of eligsir/pose_opt.py at 8064aca93886dd07ad6d890bccbdcda960a9a2af.

This test fixture is intentionally independent of the current main checkout.
It contains the audited fitting geometry verbatim, excluding only unrelated
bake/writeback code not exercised by the reference contracts.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    small = theta < eps
    sin_theta_over_theta = torch.where(
        small, torch.ones_like(theta), torch.sin(theta) / theta.clamp_min(eps)
    )
    one_minus_cos_over_theta2 = torch.where(
        small,
        torch.full_like(theta, 0.5),
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps),
    )
    K = skew_symmetric(omega)
    K2 = K @ K
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    A = sin_theta_over_theta.unsqueeze(-1)
    B = one_minus_cos_over_theta2.unsqueeze(-1)
    return eye + A * K + B * K2


def se3_delta_matrix(delta6: torch.Tensor) -> torch.Tensor:
    if delta6.shape[-1] != 6:
        raise ValueError(f"expected a 6-vector [omega(3), t(3)], got shape {tuple(delta6.shape)}")
    omega, t = delta6[..., :3], delta6[..., 3:]
    R = so3_exp_map(omega)
    transform = torch.eye(4, device=delta6.device, dtype=delta6.dtype)
    transform = transform.clone()
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


class CameraPoseOptModule(nn.Module):
    def __init__(self, num_cameras: int, device: Any = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.deltas = nn.Parameter(torch.zeros(int(num_cameras), 6, device=device, dtype=dtype))

    def delta_matrix(self, idx: int) -> torch.Tensor:
        return se3_delta_matrix(self.deltas[idx])


_MASKED_GRAPH_CAPTURE_ATTRS = frozenset(
    {"_gsplat_viewmat", "_gsplat_K", "_gsplat_camera_center"}
)


class _PoseAdjustedCamera:
    __slots__ = ("_base", "world_view_transform", "full_proj_transform", "camera_center", "gsplat")

    def __init__(self, base, world_view_transform, full_proj_transform, camera_center):
        self._base = base
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.gsplat = None

    def __getattr__(self, name):
        if name in _MASKED_GRAPH_CAPTURE_ATTRS:
            raise AttributeError(name)
        return getattr(self._base, name)


def apply_pose_delta(camera, delta_transform: torch.Tensor) -> _PoseAdjustedCamera:
    base_wvt = camera.world_view_transform
    delta_transform = delta_transform.to(device=base_wvt.device, dtype=base_wvt.dtype)
    w2c = base_wvt.transpose(0, 1)
    c2w = torch.inverse(w2c)
    c2w_new = c2w @ delta_transform
    w2c_new = torch.inverse(c2w_new)
    world_view_transform_new = w2c_new.transpose(0, 1).contiguous()
    camera_center_new = c2w_new[:3, 3].contiguous()
    full_proj_transform_new = (
        world_view_transform_new.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return _PoseAdjustedCamera(
        camera, world_view_transform_new, full_proj_transform_new, camera_center_new
    )
