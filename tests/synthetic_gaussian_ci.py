"""Tiny independent synthetic RGB-D scene and CPU reconstruction harness.

This test-owned module is intentionally not a second production mapper.  It
provides one deterministic closed loop for CI:

analytic plane+sphere scene -> RGB-D + exact pose -> SensorPacket ->
MappingSession -> sparse depth-seeded Gaussians -> CPU PyTorch optimization ->
held-out render comparison.

The input images come from a NumPy analytic ray caster, so the ground truth is
independent of the Gaussian renderer being tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from gaussian_renderer import render as render_gaussians
from eligsir.contracts.runtime import (
    BufferRef,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.types import CameraIntrinsics, Pose

WIDTH = 16
HEIGHT = 12
FX = 18.0
FY = 18.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0
_SH_C0 = 0.28209479177387814


@dataclass(frozen=True, slots=True)
class SyntheticView:
    index: int
    camera_x: float
    rgb: np.ndarray
    depth: np.ndarray
    pose: Pose
    intrinsics: CameraIntrinsics


@dataclass(frozen=True, slots=True)
class ReconstructionStats:
    gaussian_count: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mse: float
    psnr_db: float


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        width=WIDTH,
        height=HEIGHT,
        depth_scale=1.0,
        camera_frame="synthetic_camera",
        world_frame="world",
        intrinsics_source="synthetic_ci",
    )


def render_analytic_view(index: int, camera_x: float) -> SyntheticView:
    """Render a plane plus an occluding sphere with an analytic NumPy ray caster."""
    vv, uu = np.meshgrid(
        np.arange(HEIGHT, dtype=np.float32),
        np.arange(WIDTH, dtype=np.float32),
        indexing="ij",
    )
    directions = np.stack(
        ((uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu)), axis=-1
    ).astype(np.float32)
    origin = np.array([camera_x, 0.0, 0.0], dtype=np.float32)

    # Red sphere in front of an infinite textured plane at z=3 m.  Because the
    # camera only translates along x and ray z is exactly 1, the ray parameter
    # is also camera-space z-depth, matching the RGB-D contract directly.
    sphere_center = np.array([0.0, 0.0, 2.2], dtype=np.float32)
    sphere_radius = np.float32(0.38)
    oc = origin - sphere_center
    qa = np.sum(directions * directions, axis=-1)
    qb = 2.0 * np.sum(directions * oc, axis=-1)
    qc = float(np.dot(oc, oc) - sphere_radius * sphere_radius)
    discriminant = qb * qb - 4.0 * qa * qc
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
    t_near = (-qb - sqrt_discriminant) / (2.0 * qa)
    t_far = (-qb + sqrt_discriminant) / (2.0 * qa)
    sphere_t = np.where(
        discriminant >= 0.0,
        np.where(t_near > 0.0, t_near, np.where(t_far > 0.0, t_far, np.inf)),
        np.inf,
    )

    plane_t = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
    sphere_visible = sphere_t < plane_t
    depth = np.where(sphere_visible, sphere_t, plane_t).astype(np.float32)
    points_world = origin[None, None, :] + directions * depth[..., None]

    checker = (
        (
            np.floor((points_world[..., 0] + 1.5) * 3.0)
            + np.floor((points_world[..., 1] + 1.0) * 3.0)
        ).astype(np.int32)
        % 2
    )
    plane_rgb = np.stack(
        (
            0.15 + 0.55 * checker,
            0.25 + 0.25 * (1 - checker),
            0.55 - 0.25 * checker,
        ),
        axis=-1,
    ).astype(np.float32)
    sphere_rgb = np.empty_like(plane_rgb)
    sphere_rgb[..., 0] = 0.85
    sphere_rgb[..., 1] = 0.12
    sphere_rgb[..., 2] = 0.08
    rgb = np.where(sphere_visible[..., None], sphere_rgb, plane_rgb).astype(np.float32)

    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = np.float32(camera_x)
    return SyntheticView(
        index=index,
        camera_x=float(camera_x),
        rgb=np.ascontiguousarray(rgb),
        depth=np.ascontiguousarray(depth),
        pose=Pose(c2w),
        intrinsics=_intrinsics(),
    )


def packet_from_view(view: SyntheticView) -> SensorPacket:
    source = SourceIdentity(
        source_id="synthetic-gaussian-ci",
        stream_epoch=0,
        source_ordinal=view.index,
        timestamp_ns=view.index * 33_333_333,
    )
    tracking = TrackingObservation(
        source=source,
        pose=view.pose,
        status=TrackingStatus.VALID,
        provider="synthetic_exact",
        confidence=1.0,
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(view.rgb, encoding="rgb32f"),
        depth=BufferRef.from_numpy(view.depth, encoding="32FC1", units="m"),
        intrinsics=view.intrinsics,
        tracking=tracking,
    )


class ExactPacketTrackingProvider:
    """Pass through the exact pose attached to each synthetic packet."""

    provider_name = "synthetic_exact"

    def __init__(self) -> None:
        self.closed = False

    def observe(self, packet: SensorPacket, *, predicted_pose: Pose | None = None) -> TrackingObservation:
        del predicted_pose
        if packet.tracking is not None:
            return packet.tracking
        return TrackingObservation(
            source=packet.source,
            pose=None,
            status=TrackingStatus.INVALID,
            provider=self.provider_name,
            confidence=0.0,
        )

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FrameRecord:
    rgb: np.ndarray
    depth: np.ndarray
    pose: Pose
    intrinsics: CameraIntrinsics


class _TinyTorchCamera:
    def __init__(self, record: _FrameRecord) -> None:
        self.image_width = int(record.intrinsics.width)
        self.image_height = int(record.intrinsics.height)
        self.fx = float(record.intrinsics.fx)
        self.fy = float(record.intrinsics.fy)
        self.cx = float(record.intrinsics.cx)
        self.cy = float(record.intrinsics.cy)
        # Match the existing Inria/EliGSiR camera storage convention.
        self.world_view_transform = torch.from_numpy(record.pose.w2c.T.copy())


class _TinyGaussianModel:
    active_sh_degree = 0

    def __init__(self, points: np.ndarray, colors: np.ndarray, scale: float) -> None:
        self.means = torch.nn.Parameter(torch.from_numpy(points.copy()), requires_grad=False)
        self.log_scales = torch.full_like(self.means, math.log(scale))
        self.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(
            points.shape[0], 1
        )
        self.opacity_logits = torch.nn.Parameter(torch.full((points.shape[0],), -1.5))
        sh0 = (torch.from_numpy(colors.copy()) - 0.5) / _SH_C0
        self.sh0 = torch.nn.Parameter(sh0[:, None, :])

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.means

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)[:, None]

    @property
    def _features_dc(self) -> torch.Tensor:
        return self.sh0


class TinyCpuGaussianReconstructor:
    """Depth-seed a tiny map, then optimize SH0 and opacity on CPU."""

    def __init__(self, *, sample_stride: int = 3, voxel_size: float = 0.14) -> None:
        self.sample_stride = int(sample_stride)
        self.voxel_size = float(voxel_size)
        self._frames: list[_FrameRecord] = []
        self._voxels: dict[tuple[int, int, int], list[Any]] = {}
        self.model: _TinyGaussianModel | None = None
        self._pipe = SimpleNamespace(
            renderer_backend="torch_cpu_reference",
            near_plane=0.01,
            far_plane=10.0,
            eps2d=0.3,
            cpu_reference_max_interactions=1_000_000,
        )

    def __call__(self, frame: Any, observation: TrackingObservation) -> None:
        if frame.device != "cpu":
            raise ValueError(f"synthetic CPU mapper received frame on {frame.device!r}")
        if frame.depth is None or observation.pose is None:
            raise ValueError("synthetic CPU mapper requires RGB, depth, and an exact pose")

        rgb = np.asarray(frame.rgb.data, dtype=np.float32)
        depth = np.asarray(frame.depth.data, dtype=np.float32)
        record = _FrameRecord(rgb=rgb, depth=depth, pose=observation.pose, intrinsics=frame.intrinsics)
        self._frames.append(record)
        self._accumulate_depth_seeds(record)

    def _accumulate_depth_seeds(self, record: _FrameRecord) -> None:
        K = record.intrinsics
        c2w = record.pose.c2w
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        height, width = record.depth.shape
        for v in range(1, height, self.sample_stride):
            for u in range(1, width, self.sample_stride):
                z = float(record.depth[v, u])
                if not math.isfinite(z) or z <= 0.0:
                    continue
                point_camera = np.array(
                    [
                        (u - float(K.cx)) * z / float(K.fx),
                        (v - float(K.cy)) * z / float(K.fy),
                        z,
                    ],
                    dtype=np.float32,
                )
                point_world = R @ point_camera + t
                key = tuple(np.floor(point_world / self.voxel_size + 0.5).astype(np.int32))
                color = record.rgb[v, u]
                entry = self._voxels.get(key)
                if entry is None:
                    self._voxels[key] = [point_world.copy(), color.copy(), 1]
                else:
                    entry[0] += point_world
                    entry[1] += color
                    entry[2] += 1

    def _seed_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._voxels:
            raise RuntimeError("no valid synthetic depth seeds were accumulated")
        points = []
        colors = []
        for point_sum, color_sum, count in self._voxels.values():
            points.append(point_sum / count)
            colors.append(color_sum / count)
        return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)

    def _training_loss(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("reconstructor has not been finalized")
        losses = []
        background = torch.zeros(3, dtype=torch.float32)
        for record in self._frames:
            prediction = render_gaussians(
                _TinyTorchCamera(record), self.model, self._pipe, background
            )["render"]
            target = torch.from_numpy(record.rgb).permute(2, 0, 1)
            losses.append(torch.mean((prediction - target) ** 2))
        return torch.stack(losses).mean()

    def finalize(self, *, optimizer_steps: int = 10) -> ReconstructionStats:
        points, colors = self._seed_arrays()
        self.model = _TinyGaussianModel(points, colors, scale=0.11)
        optimizer = torch.optim.Adam(
            [self.model.sh0, self.model.opacity_logits], lr=0.08
        )

        initial_loss = float(self._training_loss().detach())
        for _ in range(int(optimizer_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = self._training_loss()
            loss.backward()
            optimizer.step()
        final_loss = float(self._training_loss().detach())

        return ReconstructionStats(
            gaussian_count=int(points.shape[0]),
            optimizer_steps=int(optimizer_steps),
            initial_loss=initial_loss,
            final_loss=final_loss,
        )

    def evaluate(self, view: SyntheticView) -> EvaluationMetrics:
        if self.model is None:
            raise RuntimeError("call finalize() before evaluate()")
        record = _FrameRecord(
            rgb=view.rgb, depth=view.depth, pose=view.pose, intrinsics=view.intrinsics
        )
        prediction = render_gaussians(
            _TinyTorchCamera(record),
            self.model,
            self._pipe,
            torch.zeros(3, dtype=torch.float32),
        )["render"]
        target = torch.from_numpy(view.rgb).permute(2, 0, 1)
        mse = float(torch.mean((prediction - target) ** 2).detach())
        psnr_db = -10.0 * math.log10(max(mse, 1e-12))
        return EvaluationMetrics(mse=mse, psnr_db=psnr_db)


__all__ = [
    "EvaluationMetrics",
    "ExactPacketTrackingProvider",
    "ReconstructionStats",
    "SyntheticView",
    "TinyCpuGaussianReconstructor",
    "packet_from_view",
    "render_analytic_view",
]
