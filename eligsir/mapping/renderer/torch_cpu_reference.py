"""Small differentiable CPU reference renderer for Gaussian splats.

This backend exists for correctness tests and tiny synthetic CI scenes.  It is
intentionally not a production rasterizer: it evaluates every Gaussian against
every pixel with ordinary PyTorch operations, then composites front-to-back.
That keeps the implementation independent of gsplat's CUDA/HIP kernels and
lets autograd provide the backward pass on CPU.

Supported scope:
- CPU float32 tensors only
- one pinhole camera
- anisotropic 3D Gaussians
- SH degree 0 (or explicit ``override_color``)
- RGB, alpha, and expected-depth rendering

Unsupported production features fail explicitly instead of silently changing
semantics (packed/sparse metadata, normals, taming statistics, pixel weights,
and SH degree > 0).
"""

from __future__ import annotations

import math
from typing import Any

import torch

_SH_C0 = 0.28209479177387814
_DEFAULT_MAX_INTERACTIONS = 5_000_000


def _cpu_f32(value: Any, name: str) -> torch.Tensor:
    """Return ``value`` as CPU float32 without detaching autograd."""
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(
                f"torch_cpu_reference requires CPU tensors; {name} is on {value.device}"
            )
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device="cpu")


def _model_value(model: Any, name: str) -> Any:
    value = getattr(model, name)
    return value() if callable(value) else value


def _fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(float(fov) / 2.0))


def _camera_tensors(viewpoint_camera: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the actual world-to-camera matrix and K on CPU."""
    cached = getattr(viewpoint_camera, "gsplat", None)
    if cached is not None and hasattr(cached, "viewmat") and hasattr(cached, "K"):
        return (
            _cpu_f32(cached.viewmat, "camera.viewmat").contiguous(),
            _cpu_f32(cached.K, "camera.K").contiguous(),
        )

    if hasattr(viewpoint_camera, "_gsplat_viewmat") and hasattr(viewpoint_camera, "_gsplat_K"):
        return (
            _cpu_f32(viewpoint_camera._gsplat_viewmat, "camera._gsplat_viewmat").contiguous(),
            _cpu_f32(viewpoint_camera._gsplat_K, "camera._gsplat_K").contiguous(),
        )

    if not hasattr(viewpoint_camera, "world_view_transform"):
        raise ValueError(
            "camera must provide world_view_transform or cached gsplat viewmat/K tensors"
        )

    # EliGSiR/Inria cameras store world_view_transform transposed; gsplat
    # and this backend consume the actual world-to-camera matrix.
    viewmat = _cpu_f32(
        viewpoint_camera.world_view_transform, "camera.world_view_transform"
    ).transpose(0, 1).contiguous()

    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    fx = getattr(viewpoint_camera, "fx", None)
    fy = getattr(viewpoint_camera, "fy", None)
    cx = getattr(viewpoint_camera, "cx", None)
    cy = getattr(viewpoint_camera, "cy", None)
    if fx is None:
        fx = _fov2focal(viewpoint_camera.FoVx, width)
    if fy is None:
        fy = _fov2focal(viewpoint_camera.FoVy, height)
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    K = torch.tensor(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return viewmat, K


def _quat_to_rotmat_wxyz(quats: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _sh0_colors(model: Any) -> torch.Tensor:
    if int(getattr(model, "active_sh_degree", 0)) != 0:
        raise NotImplementedError("torch_cpu_reference supports SH degree 0 only")
    sh0 = _cpu_f32(getattr(model, "_features_dc"), "gaussians._features_dc")
    if sh0.ndim == 3 and sh0.shape[1] == 1:
        sh0 = sh0[:, 0, :]
    if sh0.ndim != 2 or sh0.shape[-1] != 3:
        raise ValueError(f"expected SH0 shape [N,3] or [N,1,3], got {tuple(sh0.shape)}")
    # Standard 3DGS degree-0 SH evaluation.
    return (0.5 + _SH_C0 * sh0).clamp(0.0, 1.0)


def _empty_result(
    height: int,
    width: int,
    background: torch.Tensor,
    *,
    render_depth: bool,
) -> dict[str, Any]:
    image = background[:, None, None].expand(3, height, width).clone()
    alpha = torch.zeros(1, height, width, dtype=torch.float32)
    result: dict[str, Any] = {
        "render": image,
        "render_full": image,
        "viewspace_points": torch.empty(0, 3, dtype=torch.float32),
        "visibility_filter": torch.empty(0, dtype=torch.bool),
        "radii": torch.empty(0, dtype=torch.int32),
        "is_used": torch.empty(0, dtype=torch.bool),
        "alpha": alpha,
        "meta": {},
        "sparse_grad": False,
    }
    if render_depth:
        depth = torch.zeros(1, height, width, dtype=torch.float32)
        result["rendered_depth"] = depth
        result["render_full"] = torch.cat((image, depth), dim=0)
    return result


def render(
    viewpoint_camera: Any,
    pc: Any,
    pipe: Any,
    bg_color: Any,
    scaling_modifier: float = 1.0,
    override_color: Any = None,
    pixel_weights: Any = None,
    return_taming_stats: bool = False,
    update_sh_rest: bool = True,
    render_depth: bool = False,
    render_normals: bool = False,
) -> dict[str, Any]:
    """Render a tiny Gaussian scene with plain differentiable PyTorch ops."""
    del update_sh_rest  # SH>0 is intentionally outside this reference backend.
    if pixel_weights is not None:
        raise NotImplementedError("torch_cpu_reference does not support pixel weights")
    if return_taming_stats:
        raise NotImplementedError("torch_cpu_reference does not support taming statistics")
    if render_normals:
        raise NotImplementedError("torch_cpu_reference does not support normal rendering")

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    if height <= 0 or width <= 0:
        raise ValueError("camera image dimensions must be positive")

    if bg_color is None:
        background = torch.zeros(3, dtype=torch.float32)
    else:
        background = _cpu_f32(bg_color, "bg_color").reshape(-1)
        if background.numel() != 3:
            raise ValueError(f"bg_color must contain exactly 3 values, got {background.numel()}")

    means_world = _cpu_f32(_model_value(pc, "get_xyz"), "gaussians.get_xyz")
    if means_world.ndim != 2 or means_world.shape[-1] != 3:
        raise ValueError(f"expected means [N,3], got {tuple(means_world.shape)}")
    n_gaussians = int(means_world.shape[0])
    if n_gaussians == 0:
        return _empty_result(height, width, background, render_depth=render_depth)

    max_interactions = int(
        getattr(pipe, "cpu_reference_max_interactions", _DEFAULT_MAX_INTERACTIONS)
    )
    interactions = n_gaussians * height * width
    if interactions > max_interactions:
        raise RuntimeError(
            "torch_cpu_reference is a tiny correctness backend, not a production rasterizer: "
            f"requested {interactions:,} Gaussian-pixel interactions, limit is "
            f"{max_interactions:,}"
        )

    scales = _cpu_f32(_model_value(pc, "get_scaling"), "gaussians.get_scaling")
    quats = _cpu_f32(_model_value(pc, "get_rotation"), "gaussians.get_rotation")
    opacities = _cpu_f32(_model_value(pc, "get_opacity"), "gaussians.get_opacity").reshape(-1)
    colors = (
        _cpu_f32(override_color, "override_color")
        if override_color is not None
        else _sh0_colors(pc)
    )

    if scales.shape != (n_gaussians, 3):
        raise ValueError(f"expected scales [N,3], got {tuple(scales.shape)}")
    if quats.shape != (n_gaussians, 4):
        raise ValueError(f"expected quaternions [N,4], got {tuple(quats.shape)}")
    if opacities.shape != (n_gaussians,):
        raise ValueError(f"expected opacities [N], got {tuple(opacities.shape)}")
    if colors.shape != (n_gaussians, 3):
        raise ValueError(f"expected colors [N,3], got {tuple(colors.shape)}")

    for name, tensor in (
        ("means", means_world),
        ("scales", scales),
        ("quaternions", quats),
        ("opacities", opacities),
        ("colors", colors),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
    if bool((scales <= 0).any()):
        raise ValueError("Gaussian scales must be strictly positive")

    scales = scales * float(scaling_modifier)
    opacities = opacities.clamp(0.0, 1.0)
    colors = colors.clamp(0.0, 1.0)
    viewmat, K = _camera_tensors(viewpoint_camera)

    rotation = _quat_to_rotmat_wxyz(quats)
    covariance_world = rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)

    R_w2c = viewmat[:3, :3]
    t_w2c = viewmat[:3, 3]
    means_cam = means_world @ R_w2c.transpose(0, 1) + t_w2c
    covariance_cam = R_w2c[None] @ covariance_world @ R_w2c.transpose(0, 1)[None]

    x, y, z = means_cam.unbind(dim=-1)
    near_plane = float(getattr(pipe, "near_plane", 0.01))
    far_plane = float(getattr(pipe, "far_plane", 1e10))
    valid = (z > near_plane) & (z < far_plane)
    z_safe = z.clamp_min(max(near_plane, 1e-6))

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    means2d = torch.stack((u, v), dim=-1)

    zeros = torch.zeros_like(z)
    J = torch.stack(
        (
            torch.stack((fx / z_safe, zeros, -fx * x / z_safe.square()), dim=-1),
            torch.stack((zeros, fy / z_safe, -fy * y / z_safe.square()), dim=-1),
        ),
        dim=1,
    )
    eps2d = float(getattr(pipe, "eps2d", 0.3))
    covariance_2d = J @ covariance_cam @ J.transpose(-1, -2)
    covariance_2d = covariance_2d + torch.eye(2, dtype=torch.float32)[None] * eps2d
    inverse_covariance_2d = torch.linalg.inv(covariance_2d)

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), dim=-1)
    delta = pixels[None] - means2d[:, None, None, :]
    mahalanobis = torch.einsum(
        "nhwi,nij,nhwj->nhw", delta, inverse_covariance_2d, delta
    )

    sigma_extent = float(getattr(pipe, "cpu_reference_sigma_extent", 3.0))
    footprint = torch.exp(-0.5 * mahalanobis)
    footprint = torch.where(
        mahalanobis <= sigma_extent * sigma_extent,
        footprint,
        torch.zeros_like(footprint),
    )
    alphas = opacities[:, None, None] * footprint
    alphas = torch.where(valid[:, None, None], alphas, torch.zeros_like(alphas))
    alphas = alphas.clamp(0.0, 0.999)

    # One global depth order is enough for this tiny correctness backend.  It is
    # deliberately simpler than gsplat's per-tile ordering and adequate for the
    # small CI scenes this module is allowed to render.
    order = torch.argsort(z)
    alpha_sorted = alphas[order]
    colors_sorted = colors[order]
    z_sorted = z[order]

    survival = (1.0 - alpha_sorted).clamp_min(1e-6)
    survival_prefix = torch.cumprod(survival, dim=0)
    transmittance_before = torch.cat(
        (torch.ones_like(survival_prefix[:1]), survival_prefix[:-1]), dim=0
    )
    weights = transmittance_before * alpha_sorted
    final_transmittance = survival_prefix[-1]

    rgb_hwc = (weights[..., None] * colors_sorted[:, None, None, :]).sum(dim=0)
    rgb_hwc = rgb_hwc + final_transmittance[..., None] * background[None, None, :]
    image = rgb_hwc.permute(2, 0, 1).contiguous()

    alpha_image_hw = weights.sum(dim=0).clamp(0.0, 1.0)
    alpha_image = alpha_image_hw[None].contiguous()
    depth_numerator = (weights * z_sorted[:, None, None]).sum(dim=0)
    expected_depth_hw = torch.where(
        alpha_image_hw > 1e-8,
        depth_numerator / alpha_image_hw.clamp_min(1e-8),
        torch.zeros_like(depth_numerator),
    )

    eigenvalues = torch.linalg.eigvalsh(covariance_2d).clamp_min(0.0)
    radii = torch.ceil(sigma_extent * eigenvalues.sqrt().amax(dim=-1)).to(torch.int32)
    radii = torch.where(valid, radii, torch.zeros_like(radii))
    on_screen = (
        (u + radii.to(torch.float32) >= 0)
        & (u - radii.to(torch.float32) < width)
        & (v + radii.to(torch.float32) >= 0)
        & (v - radii.to(torch.float32) < height)
    )
    is_used = valid & on_screen & (alphas.flatten(1).amax(dim=1) > 1e-6)
    viewspace_points = torch.stack((u, v, z), dim=-1)

    result: dict[str, Any] = {
        "render": image,
        "render_full": image,
        "viewspace_points": viewspace_points,
        "visibility_filter": is_used,
        "radii": radii,
        "is_used": is_used,
        "alpha": alpha_image,
        "meta": {"means2d": means2d, "depths": z},
        "sparse_grad": False,
    }
    if render_depth:
        rendered_depth = expected_depth_hw[None].contiguous()
        result["rendered_depth"] = rendered_depth
        result["render_full"] = torch.cat((image, rendered_depth), dim=0)
    return result


__all__ = ["render"]
