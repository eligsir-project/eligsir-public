"""Pure selected-tier motion confidence reference for depth supervision."""

import math
from numbers import Real

import torch


def _valid_scalar(value: float, *, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a host numeric scalar")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return value


def motion_depth_confidence(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_to_rgb_capture: torch.Tensor,
    *,
    tau_px: float | tuple[float, float] = 1.0,
    w_min: float = 0.05,
) -> torch.Tensor:
    """Return per-pixel confidence from selected-tier capture-to-capture motion.

    Structural errors raise; invalid depth, projection, or matrix content is
    deliberately neutral so callers can retain an otherwise admitted source.
    """
    if not all(isinstance(value, torch.Tensor) for value in (depth, intrinsics, depth_to_rgb_capture)):
        raise ValueError("depth, intrinsics, and depth_to_rgb_capture must be tensors")
    if depth.ndim != 2 or intrinsics.shape != (3, 3) or depth_to_rgb_capture.shape != (4, 4):
        raise ValueError("expected depth [H,W], intrinsics [3,3], and transform [4,4]")
    if not depth.is_floating_point():
        raise ValueError("depth must have floating dtype")
    if any(value.device != depth.device or value.dtype != depth.dtype for value in (intrinsics, depth_to_rgb_capture)):
        raise ValueError("depth, intrinsics, and transform must share device and dtype")

    dtype_limits = torch.finfo(depth.dtype)
    if isinstance(tau_px, tuple):
        if len(tau_px) != 2:
            raise ValueError("tau_px tuple must contain x and y values")
        tau_x, tau_y = (
            _valid_scalar(value, name=f"tau_px_{axis}", lower=dtype_limits.tiny, upper=dtype_limits.max)
            for axis, value in zip(("x", "y"), tau_px, strict=True)
        )
    else:
        tau_x = tau_y = _valid_scalar(
            tau_px, name="tau_px", lower=dtype_limits.tiny, upper=dtype_limits.max
        )
    w_min = _valid_scalar(w_min, name="w_min", lower=0.0, upper=1.0)
    if depth.numel() == 0:
        return torch.ones_like(depth)

    matrix_finite = torch.isfinite(intrinsics).all() & torch.isfinite(depth_to_rgb_capture).all()
    safe_intrinsics = torch.where(torch.isfinite(intrinsics), intrinsics, torch.zeros_like(intrinsics))
    safe_transform = torch.where(
        torch.isfinite(depth_to_rgb_capture), depth_to_rgb_capture, torch.zeros_like(depth_to_rgb_capture)
    )
    fx, fy = safe_intrinsics[0, 0], safe_intrinsics[1, 1]
    focal_valid = (fx > 0) & (fy > 0)
    safe_fx = torch.where(focal_valid, fx, torch.ones_like(fx))
    safe_fy = torch.where(focal_valid, fy, torch.ones_like(fy))

    height, width = depth.shape
    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    source_valid = torch.isfinite(depth) & (depth > 0)
    z = torch.where(source_valid, depth, torch.ones_like(depth))
    points = torch.stack(
        ((u - safe_intrinsics[0, 2]) * z / safe_fx, (v - safe_intrinsics[1, 2]) * z / safe_fy, z),
        dim=-1,
    )
    rgb_points = points @ safe_transform[:3, :3].transpose(0, 1) + safe_transform[:3, 3]
    projected_depth = rgb_points[..., 2]
    projected_finite = torch.isfinite(rgb_points).all(dim=-1)
    projection_valid = projected_finite & (projected_depth > 0)
    safe_projected_depth = torch.where(
        projection_valid, projected_depth, torch.ones_like(projected_depth)
    )
    projected_u = safe_intrinsics[0, 0] * rgb_points[..., 0] / safe_projected_depth + safe_intrinsics[0, 2]
    projected_v = safe_intrinsics[1, 1] * rgb_points[..., 1] / safe_projected_depth + safe_intrinsics[1, 2]
    reprojected_finite = torch.isfinite(projected_u) & torch.isfinite(projected_v)
    valid = source_valid & projection_valid & reprojected_finite & focal_valid & matrix_finite
    safe_u = torch.where(valid, projected_u, u)
    safe_v = torch.where(valid, projected_v, v)
    normalized_displacement_squared = ((safe_u - u) / tau_x).square() + ((safe_v - v) / tau_y).square()
    confidence = torch.clamp(1.0 / (1.0 + normalized_displacement_squared), min=w_min, max=1.0)
    return torch.where(valid, confidence, torch.ones_like(confidence))
