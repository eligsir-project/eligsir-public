"""Resolution-aware image metrics for classic/antialiased live A/B runs.

These metrics are observational: they never influence mapping policy.  Edge
PSNR is computed in a normalized Sobel-gradient domain at the exact evaluated
resolution.  Temporal flicker compares render residuals after an explicitly
provided motion-compensation grid, avoiding the invalid shortcut of comparing
different camera views pixel-for-pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


_SOBEL_NORMALIZER = 32.0**0.5


def _image_chw(image: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError(f"{name} must have shape [1|3,H,W]")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite samples")
    return image


def _normalized_sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    image = _image_chw(image, name="image")
    gray = image.mean(dim=0, keepdim=True)[None]
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square()).div(_SOBEL_NORMALIZER).clamp(0, 1)


def edge_psnr(render: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """Return scalar Sobel-domain PSNR at the inputs' existing resolution."""

    render = _image_chw(render, name="render")
    ground_truth = _image_chw(ground_truth, name="ground_truth")
    if render.shape != ground_truth.shape:
        raise ValueError("edge PSNR requires render and ground_truth shape equality")
    mse = F.mse_loss(
        _normalized_sobel_magnitude(render),
        _normalized_sobel_magnitude(ground_truth),
    )
    return torch.where(
        mse > 0,
        -10.0 * torch.log10(mse),
        mse.new_tensor(float("inf")),
    )


@dataclass(frozen=True, slots=True)
class TemporalFlickerResult:
    """Sufficient statistics for motion-compensated RGB residual change.

    Keeping the reduction on-device lets a whole evaluation tick aggregate
    pair statistics before one host transfer.  ``valid_pixels`` is spatial;
    the squared-error denominator also includes ``channels``.
    """

    squared_error_sum: torch.Tensor
    valid_pixels_tensor: torch.Tensor
    channels: int = 3

    @classmethod
    def from_sufficient_statistics(
        cls,
        *,
        squared_error_sum: torch.Tensor,
        valid_pixels: torch.Tensor,
        channels: int = 3,
    ) -> "TemporalFlickerResult":
        if squared_error_sum.ndim != 0 or valid_pixels.ndim != 0:
            raise ValueError("temporal flicker sufficient statistics must be scalar")
        if channels < 1:
            raise ValueError("temporal flicker channels must be positive")
        return cls(squared_error_sum, valid_pixels, channels)

    @property
    def valid_pixels(self) -> int:
        return int(self.valid_pixels_tensor.item())

    @property
    def rms(self) -> float:
        if self.valid_pixels < 1:
            raise ValueError("temporal flicker requires non-empty valid overlap")
        mse = float(self.squared_error_sum.item()) / (
            self.channels * self.valid_pixels
        )
        return math.sqrt(max(0.0, mse))

    @property
    def psnr_db(self) -> float:
        rms = self.rms
        return float("inf") if rms == 0.0 else -20.0 * math.log10(rms)


def summarize_temporal_flicker(
    values: list[TemporalFlickerResult],
) -> tuple[float | None, float | None, int]:
    """Derive one sample-weighted tick metric with one device-to-host copy."""

    if not values:
        return None, None, 0
    devices = {value.squared_error_sum.device for value in values}
    if len(devices) != 1:
        raise ValueError("temporal flicker statistics must share one device")
    dtype = values[0].squared_error_sum.dtype
    total_sse = torch.stack(
        [value.squared_error_sum.to(dtype=dtype) for value in values]
    ).sum()
    total_samples = torch.stack(
        [
            value.valid_pixels_tensor.to(device=total_sse.device, dtype=dtype)
            * value.channels
            for value in values
        ]
    ).sum()
    total_sse_value, total_samples_value = torch.stack(
        (total_sse, total_samples)
    ).detach().cpu().tolist()
    if not math.isfinite(total_sse_value) or not math.isfinite(total_samples_value):
        raise ValueError("temporal flicker aggregate is non-finite")
    if total_sse_value < 0:
        raise ValueError("temporal flicker squared error cannot be negative")
    if total_samples_value <= 0:
        return None, None, 0
    mse = max(0.0, total_sse_value / total_samples_value)
    rms = math.sqrt(mse)
    # Exact matches have infinite PSNR.  Preserve the valid RMS but emit null
    # for the dB field so canonical JSON never contains a nonfinite number.
    psnr_db = None if rms == 0.0 else -20.0 * math.log10(rms)
    return rms, psnr_db, len(values)


def _camera_intrinsics(cam, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    width = int(cam.image_width)
    height = int(cam.image_height)
    fx = getattr(cam, "fx", None)
    fy = getattr(cam, "fy", None)
    cx = getattr(cam, "cx", None)
    cy = getattr(cam, "cy", None)
    fx = float(fx) if fx is not None else width / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = float(fy) if fy is not None else height / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    cx = float(cx) if cx is not None else width / 2.0
    cy = float(cy) if cy is not None else height / 2.0
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )


def reprojection_grid_a_for_b(
    depth_b: torch.Tensor,
    intrinsics_a: torch.Tensor,
    world_to_camera_a: torch.Tensor,
    intrinsics_b: torch.Tensor,
    world_to_camera_b: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project target-B pixels through sensor depth into source camera A."""

    if depth_b.ndim == 3 and depth_b.shape[0] == 1:
        depth_b = depth_b[0]
    if depth_b.ndim != 2:
        raise ValueError("depth_b must have shape [H,W] or [1,H,W]")
    height, width = depth_b.shape
    device, dtype = depth_b.device, depth_b.dtype
    for name, matrix, shape in (
        ("intrinsics_a", intrinsics_a, (3, 3)),
        ("intrinsics_b", intrinsics_b, (3, 3)),
        ("world_to_camera_a", world_to_camera_a, (4, 4)),
        ("world_to_camera_b", world_to_camera_b, (4, 4)),
    ):
        if matrix.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        if matrix.device != device:
            raise ValueError(f"{name} must share the depth device")
    vv, uu = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels_b = torch.stack((uu, vv, torch.ones_like(uu)), dim=0).reshape(3, -1)
    xyz_b = torch.linalg.solve(intrinsics_b, pixels_b) * depth_b.reshape(1, -1)
    xyz_b_h = torch.cat((xyz_b, torch.ones(1, xyz_b.shape[1], device=device, dtype=dtype)))
    world = torch.linalg.solve(world_to_camera_b, xyz_b_h)
    xyz_a = (world_to_camera_a @ world)[:3]
    projected = intrinsics_a @ xyz_a
    z_a = xyz_a[2].reshape(height, width)
    denom = projected[2].clamp_min(torch.finfo(dtype).eps)
    u_a = (projected[0] / denom).reshape(height, width)
    v_a = (projected[1] / denom).reshape(height, width)
    if isinstance(source_height, bool) or not isinstance(source_height, int) or source_height < 1:
        raise ValueError("source_height must be a positive integer")
    if isinstance(source_width, bool) or not isinstance(source_width, int) or source_width < 1:
        raise ValueError("source_width must be a positive integer")
    x_norm = torch.zeros_like(u_a) if source_width == 1 else 2.0 * u_a / (source_width - 1) - 1.0
    y_norm = torch.zeros_like(v_a) if source_height == 1 else 2.0 * v_a / (source_height - 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1)
    valid = (
        torch.isfinite(depth_b)
        & (depth_b > 0)
        & torch.isfinite(z_a)
        & (z_a > 0)
        & (u_a >= 0)
        & (u_a <= source_width - 1)
        & (v_a >= 0)
        & (v_a <= source_height - 1)
    )
    return grid, valid[None], z_a


def camera_pair_temporal_flicker(
    cam_a,
    render_a: torch.Tensor,
    ground_truth_a: torch.Tensor,
    cam_b,
    render_b: torch.Tensor,
    ground_truth_b: torch.Tensor,
) -> TemporalFlickerResult:
    """Motion-compensated residual flicker for two pose/depth-bearing cameras."""

    from eligsir.mapping.integration.depth_helpers import _get_sensor_depth

    device, dtype = render_b.device, render_b.dtype
    depth_b = _get_sensor_depth(cam_b, *render_b.shape[-2:])
    if depth_b is None:
        raise ValueError("target camera has no sensor depth for temporal flicker")
    depth_a = _get_sensor_depth(cam_a, *render_a.shape[-2:])
    if depth_a is None:
        raise ValueError("source camera has no sensor depth for temporal flicker")
    depth_b = depth_b.to(device=device, dtype=dtype)
    depth_a = depth_a.to(device=device, dtype=dtype)
    k_a = _camera_intrinsics(cam_a, device=device, dtype=dtype)
    k_b = _camera_intrinsics(cam_b, device=device, dtype=dtype)
    w2c_a = cam_a.world_view_transform.transpose(0, 1).to(device=device, dtype=dtype)
    w2c_b = cam_b.world_view_transform.transpose(0, 1).to(device=device, dtype=dtype)
    grid, valid, projected_depth_a = reprojection_grid_a_for_b(
        depth_b,
        k_a,
        w2c_a,
        k_b,
        w2c_b,
        source_height=render_a.shape[-2],
        source_width=render_a.shape[-1],
    )
    sampled_depth_a = F.grid_sample(
        depth_a[None],
        grid[None],
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    depth_tolerance = torch.maximum(
        sampled_depth_a.new_tensor(0.05), sampled_depth_a.abs() * 0.05
    )
    valid &= (
        (sampled_depth_a > 0)
        & torch.isfinite(sampled_depth_a)
        & ((sampled_depth_a - projected_depth_a).abs() <= depth_tolerance)
    )[None]
    return motion_compensated_temporal_flicker(
        render_a, ground_truth_a, render_b, ground_truth_b, grid, valid
    )


def motion_compensated_temporal_flicker(
    render_a: torch.Tensor,
    ground_truth_a: torch.Tensor,
    render_b: torch.Tensor,
    ground_truth_b: torch.Tensor,
    sample_grid_a_for_b: torch.Tensor,
    valid_overlap: torch.Tensor,
) -> TemporalFlickerResult:
    """Compare residual A warped into B against residual B.

    ``sample_grid_a_for_b`` follows ``torch.grid_sample`` normalized
    coordinates and must have shape ``[H_B,W_B,2]``. ``valid_overlap`` is an
    explicit visibility/depth-consistency mask; an empty overlap fails closed.
    """

    render_a = _image_chw(render_a, name="render_a")
    ground_truth_a = _image_chw(ground_truth_a, name="ground_truth_a")
    render_b = _image_chw(render_b, name="render_b")
    ground_truth_b = _image_chw(ground_truth_b, name="ground_truth_b")
    if render_a.shape != ground_truth_a.shape:
        raise ValueError("frame A render/ground truth shapes differ")
    if render_b.shape != ground_truth_b.shape:
        raise ValueError("frame B render/ground truth shapes differ")
    height, width = render_b.shape[-2:]
    if sample_grid_a_for_b.shape != (height, width, 2):
        raise ValueError("motion-compensation grid must have shape [H_B,W_B,2]")
    if valid_overlap.shape not in {(height, width), (1, height, width)}:
        raise ValueError("valid_overlap must have shape [H_B,W_B] or [1,H_B,W_B]")
    if sample_grid_a_for_b.device != render_a.device:
        raise ValueError("motion-compensation grid must share the image device")
    mask = valid_overlap.reshape(height, width).to(device=render_a.device, dtype=torch.bool)
    valid_pixels = mask.sum()
    if int(valid_pixels.item()) == 0:
        raise ValueError("temporal flicker requires non-empty valid overlap")
    residual_a = render_a - ground_truth_a
    residual_b = render_b - ground_truth_b
    warped_a = F.grid_sample(
        residual_a[None],
        sample_grid_a_for_b[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]
    delta = (warped_a - residual_b)[:, mask]
    return TemporalFlickerResult.from_sufficient_statistics(
        squared_error_sum=delta.square().sum(),
        valid_pixels=valid_pixels,
        channels=int(delta.shape[0]),
    )


__all__ = [
    "TemporalFlickerResult",
    "camera_pair_temporal_flicker",
    "edge_psnr",
    "motion_compensated_temporal_flicker",
    "reprojection_grid_a_for_b",
    "summarize_temporal_flicker",
]
