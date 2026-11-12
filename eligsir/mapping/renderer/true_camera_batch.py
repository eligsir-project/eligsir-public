"""True C>1 gsplat rasterization adapter.

Unlike sequential gradient accumulation, this invokes gsplat.rasterization
exactly once with C camera matrices. It is capability-gated and fail-closed:
all cameras must share resolution and the native gsplat SH path is required
until per-camera Python-SH colors and dense C>1 gradients are qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from scene.cameras import make_gsplat_camera_batch


@dataclass(frozen=True, slots=True)
class TrueCameraBatchResult:
    renders: torch.Tensor       # [C,3,H,W]
    alphas: torch.Tensor        # [C,1,H,W]
    rendered_depth: torch.Tensor | None
    rendered_depth_ed: torch.Tensor | None
    meta: dict
    camera_count: int
    sparse_grad: bool
    rasterizer_call_count: int = 1


def supports_true_camera_batch(pipe, cameras: Sequence[Any]) -> tuple[bool, str]:
    cams = tuple(cameras)
    if len(cams) < 2:
        return False, "requires_C_gt_1"
    if len({(int(c.image_width), int(c.image_height)) for c in cams}) != 1:
        return False, "mixed_resolution_batch_not_supported"
    if str(getattr(pipe, "sh_impl", "python")).lower() != "gsplat":
        return False, "requires_native_gsplat_sh_until_python_sh_batch_is_qualified"
    if not bool(getattr(pipe, "gsplat_sparse_grad", False)):
        return False, "requires_sparse_grad_until_dense_C_gt_1_is_qualified"
    return True, "supported"


def render_true_camera_batch(
    cameras: Sequence[Any],
    pc: Any,
    pipe: Any,
    bg_color: torch.Tensor | None,
    *,
    update_sh_rest: bool = True,
    render_depth: bool = False,
) -> TrueCameraBatchResult:
    ok, reason = supports_true_camera_batch(pipe, cameras)
    if not ok:
        raise RuntimeError(f"TRUE_CAMERA_BATCH unavailable: {reason}")
    from gsplat.rendering import rasterization
    from eligsir.mapping.renderer.gsplat_backend import (
        _gsplat_sh_coeffs,
        validate_rasterize_mode,
    )

    cams = tuple(cameras)
    device = pc.get_xyz.device
    batch = make_gsplat_camera_batch(cams, device=device)
    width, height = batch.widths[0], batch.heights[0]

    means = pc.get_xyz.contiguous()
    scales = pc.get_scaling.contiguous()
    sparse_grad = bool(getattr(pipe, "gsplat_sparse_grad", False))
    quats = (pc._rotation if sparse_grad else pc.get_rotation).contiguous()
    opacities = pc.get_opacity.squeeze(-1).contiguous()
    colors = _gsplat_sh_coeffs(pc, update_sh_rest=update_sh_rest)
    sh_degree = pc.active_sh_degree

    if render_depth:
        mode = "RGB+ED"
        backgrounds = None
    else:
        mode = getattr(pipe, "render_mode", "RGB")
        if bg_color is None:
            backgrounds = None
        else:
            # gsplat's packed rasterizer removes the camera dimension from the
            # packed means tensor and consequently requires one flat shared
            # background.  This adapter accepts one shared background already;
            # expanding it to [C, channels] makes real C>1 packed rendering fail.
            backgrounds = bg_color.reshape(-1).contiguous()

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=batch.viewmats.contiguous(),
        Ks=batch.Ks.contiguous(),
        width=int(width),
        height=int(height),
        near_plane=getattr(pipe, "near_plane", 0.01),
        far_plane=getattr(pipe, "far_plane", 1e10),
        radius_clip=getattr(pipe, "radius_clip", 0.0),
        eps2d=getattr(pipe, "eps2d", 0.3),
        sh_degree=sh_degree,
        packed=True,
        tile_size=getattr(pipe, "tile_size", 16),
        backgrounds=backgrounds,
        render_mode=mode,
        sparse_grad=sparse_grad,
        absgrad=getattr(pipe, "absgrad", False),
        rasterize_mode=validate_rasterize_mode(
            getattr(pipe, "rasterize_mode", "classic")
        ),
    )
    rendered = render_colors.permute(0, 3, 1, 2).contiguous()
    rgb = rendered[:, :3]
    alpha = render_alphas.permute(0, 3, 1, 2).contiguous()
    depth = None
    depth_ed = None
    if render_depth:
        depth_ed = rendered[:, 3:4]
        depth = depth_ed * alpha
        if bg_color is not None:
            rgb = rgb + bg_color.reshape(1, 3, 1, 1) * (1.0 - alpha)
    return TrueCameraBatchResult(
        renders=rgb,
        alphas=alpha,
        rendered_depth=depth,
        rendered_depth_ed=depth_ed,
        meta=meta,
        camera_count=len(cams),
        sparse_grad=sparse_grad,
    )
