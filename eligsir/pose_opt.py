"""Shared differentiable camera-local SE(3) geometry for post-mapping alignment."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


_TIER_FACTOR = {"native": 1, "half": 2, "quarter": 4, "eighth": 8}


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack((
        torch.stack((zeros, -z, y), dim=-1),
        torch.stack((z, zeros, -x), dim=-1),
        torch.stack((-y, x, zeros), dim=-1),
    ), dim=-2)


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    small = theta < eps
    a = torch.where(small, torch.ones_like(theta), torch.sin(theta) / theta.clamp_min(eps))
    b = torch.where(
        small, torch.full_like(theta, 0.5),
        (1.0 - torch.cos(theta)) / (theta * theta).clamp_min(eps * eps),
    )
    k = skew_symmetric(omega)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def se3_delta_matrix(delta6: torch.Tensor) -> torch.Tensor:
    if delta6.shape[-1] != 6:
        raise ValueError(f"expected a 6-vector [omega(3), t(3)], got shape {tuple(delta6.shape)}")
    result = torch.eye(4, device=delta6.device, dtype=delta6.dtype)
    result[:3, :3] = so3_exp_map(delta6[..., :3])
    result[:3, 3] = delta6[..., 3:]
    return result


class CameraPoseOptModule(nn.Module):
    def __init__(self, num_cameras: int, device: Any = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.deltas = nn.Parameter(torch.zeros(int(num_cameras), 6, device=device, dtype=dtype))

    def delta_matrix(self, idx: int) -> torch.Tensor:
        return se3_delta_matrix(self.deltas[idx])

class _TieredCamera:
    """Read-through camera proxy using the established decimated tier geometry."""

    __slots__ = ("_base", "image_width", "image_height", "fx", "fy", "cx", "cy", "gsplat")

    def __init__(self, base: Any, tier: str) -> None:
        factor = _TIER_FACTOR[tier]
        self._base = base
        self.image_width, self.image_height = max(1, int(base.image_width) // factor), max(1, int(base.image_height) // factor)
        self.fx = float(base.fx) / factor if getattr(base, "fx", None) is not None else None
        self.fy = float(base.fy) / factor if getattr(base, "fy", None) is not None else None
        self.cx = float(base.cx) / factor if getattr(base, "cx", None) is not None else None
        self.cy = float(base.cy) / factor if getattr(base, "cy", None) is not None else None
        base_gsplat = getattr(base, "gsplat", None)
        if base_gsplat is None or factor == 1:
            self.gsplat = base_gsplat
        else:
            from scene.cameras import GsplatCameraTensors

            K = base_gsplat.K.clone()
            K[0, 0] /= factor
            K[1, 1] /= factor
            K[0, 2] /= factor
            K[1, 2] /= factor
            self.gsplat = GsplatCameraTensors(
                viewmat=base_gsplat.viewmat,
                K=K,
                camera_center=base_gsplat.camera_center,
                width=self.image_width,
                height=self.image_height,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _tiered_gt(original_image: torch.Tensor, tier: str) -> torch.Tensor:
    factor = _TIER_FACTOR[tier]
    if factor == 1:
        return original_image
    _, height, width = original_image.shape
    return original_image[:, ::factor, ::factor][:, :max(1, height // factor), :max(1, width // factor)].contiguous()


def _clamp_delta(delta6: torch.Tensor, max_deg: float, max_mm: float) -> tuple[torch.Tensor, bool, bool]:
    """Rescale, never element-wise clip, the local rotation/translation blocks."""
    omega, translation = delta6[:3].clone(), delta6[3:].clone()
    max_rad = math.radians(max_deg)
    theta = float(torch.linalg.norm(omega).item())
    clamped_rotation = theta > max_rad > 0.0
    if clamped_rotation:
        omega *= max_rad / theta
    max_m = max_mm / 1000.0
    magnitude = float(torch.linalg.norm(translation).item())
    clamped_translation = magnitude > max_m > 0.0
    if clamped_translation:
        translation *= max_m / magnitude
    return torch.cat([omega, translation]), clamped_rotation, clamped_translation


def _delta_deg_mm(delta6: torch.Tensor) -> tuple[float, float]:
    return math.degrees(float(torch.linalg.norm(delta6[:3]).item())), float(torch.linalg.norm(delta6[3:]).item()) * 1000.0


_MASKED_GRAPH_CAPTURE_ATTRS = frozenset({"_gsplat_viewmat", "_gsplat_K", "_gsplat_camera_center"})


def _corrected_gsplat(base_gsplat: Any, world_view_transform: torch.Tensor, camera_center: torch.Tensor) -> Any:
    if base_gsplat is None:
        return None
    from scene.cameras import GsplatCameraTensors

    return GsplatCameraTensors(
        viewmat=world_view_transform.transpose(0, 1).contiguous(), K=base_gsplat.K,
        camera_center=camera_center.view(1, 3).contiguous(), width=base_gsplat.width, height=base_gsplat.height,
    )


class _PoseAdjustedCamera:
    __slots__ = ("_base", "world_view_transform", "full_proj_transform", "camera_center", "gsplat")

    def __init__(self, base: object, world_view_transform: torch.Tensor, full_proj_transform: torch.Tensor, camera_center: torch.Tensor):
        self._base = base
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.gsplat = _corrected_gsplat(
            getattr(base, "gsplat", None), world_view_transform, camera_center
        )

    def __getattr__(self, name: str) -> Any:
        if name in _MASKED_GRAPH_CAPTURE_ATTRS:
            raise AttributeError(name)
        return getattr(self._base, name)


def apply_pose_delta(camera: object, delta_transform: torch.Tensor, *, base_c2w: torch.Tensor | None = None) -> _PoseAdjustedCamera:
    base_wvt = camera.world_view_transform
    delta = delta_transform.to(device=base_wvt.device, dtype=base_wvt.dtype)
    c2w = base_c2w if base_c2w is not None else torch.inverse(base_wvt.transpose(0, 1))
    corrected_c2w = c2w @ delta
    corrected_wvt = torch.inverse(corrected_c2w).transpose(0, 1).contiguous()
    center = corrected_c2w[:3, 3].contiguous()
    projection = corrected_wvt.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0)).squeeze(0)
    return _PoseAdjustedCamera(camera, corrected_wvt, projection, center)


def bake_pose_delta_into_camera(camera: object, delta_transform: torch.Tensor) -> None:
    """Apply the audited local-frame delta to one mutable camera in place."""
    base_wvt = camera.world_view_transform
    delta = delta_transform.to(device=base_wvt.device, dtype=base_wvt.dtype)
    c2w = torch.inverse(base_wvt.transpose(0, 1)) @ delta
    camera.world_view_transform = torch.inverse(c2w).transpose(0, 1).contiguous()
    camera.camera_center = c2w[:3, 3].contiguous()
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0)
        .bmm(camera.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    if hasattr(camera, "c2w") and camera.c2w is not None:
        camera.c2w = c2w.contiguous()
    gsplat = _corrected_gsplat(
        getattr(camera, "gsplat", None), camera.world_view_transform, camera.camera_center
    )
    if gsplat is not None:
        camera.gsplat = gsplat
    for name in _MASKED_GRAPH_CAPTURE_ATTRS:
        try:
            if name in vars(camera):
                delattr(camera, name)
        except TypeError:
            pass
    if getattr(camera, "R", None) is not None and getattr(camera, "T", None) is not None:
        from eligsir.utils.rgbd_frames import c2w_to_camera_rt

        camera.R, camera.T = c2w_to_camera_rt(c2w.detach().cpu().numpy())
