#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from eligsir.utils.graphics_utils import getProjectionMatrix, getWorld2View2


@dataclass
class GsplatCameraTensors:
    viewmat: torch.Tensor
    K: torch.Tensor
    camera_center: torch.Tensor
    width: int
    height: int


@dataclass
class GsplatCameraBatch:
    """Shape-compatible camera batch prepared in one tensor operation."""

    viewmats: torch.Tensor
    Ks: torch.Tensor
    camera_centers: torch.Tensor
    widths: tuple[int, ...]
    heights: tuple[int, ...]


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device=None,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        width=None,
        height=None,
        pending_image_loader=None,
    ):
        super().__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        if data_device is None:
            if image is not None:
                data_device = image.device
            else:
                raise ValueError(
                    "Camera: data_device must be specified explicitly when image=None"
                )
        self.data_device = torch.device(data_device)

        if image is None:
            if pending_image_loader is None:
                raise ValueError("Camera: image=None requires pending_image_loader")
            if width is None or height is None:
                raise ValueError("Camera: image=None requires explicit width/height")
            self._pending_image_loader = pending_image_loader
            self._native_image_loader = pending_image_loader
            self.image_width = int(width)
            self.image_height = int(height)
            self.width = self.image_width
            self.height = self.image_height
        else:
            self._pending_image_loader = None
            self._native_image_loader = None
            if image.device.type != "cpu":
                raise ValueError("Camera image producer must hand off CPU-owned pixels")
            self.original_image = image.clamp(0.0, 1.0)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            self.width = self.image_width
            self.height = self.image_height
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.original_image.device)
            self._declare_score_provenance()

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.as_tensor(getWorld2View2(R, T, trans, scale), device=self.data_device)
            .transpose(0, 1)
            .contiguous()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .to(self.data_device)
            .transpose(0, 1)
            .contiguous()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def has_pending_image(self) -> bool:
        return self._pending_image_loader is not None

    def set_image_loader(self, loader):
        self._pending_image_loader = loader
        self._native_image_loader = loader

    def _declare_score_provenance(self) -> None:
        """Declare lineage where this camera owns canonical GT materialization."""
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(
            self,
            context=f"camera:{self.image_name}",
        )

    def _declare_prepared_score_provenance(self) -> None:
        """Attach lineage to an already-attested prepared RGB tensor without D2H.

        Prepared RGB reaches this constructor only after the strict source
        contract has accepted the raw source and the preparation owner has
        converted/resampled it on-device. Re-running score-domain reductions
        here would turn every READY replay bind into several device-to-Python
        synchronization points. Scientific scoring remains free to validate at
        its explicit reporting boundary; the mapping hot path only carries the
        immutable provenance established by this trusted preparation boundary.
        """
        from eligsir.utils.image_provenance import provenance_for_canonical_float
        from eligsir.utils.score_contract import attach_score_provenance

        tensor = self.original_image
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = tuple(int(value) for value in tensor.shape[-3:])
        resolution = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        provenance = provenance_for_canonical_float(dtype, shape, resolution)
        self.score_provenance = provenance
        attach_score_provenance(tensor, provenance)
        # Objective.py may use this immutable preparation attestation to avoid
        # repeating a full-image finite reduction on every replay of the same
        # prepared generation. Legacy/lazy cameras do not receive this marker.
        # `_version` is included because object id alone survives an in-place
        # mutation of the same tensor (issue #1671): a torch in-place op keeps
        # `id(tensor)` unchanged but increments `Tensor._version`, so the pair
        # is what actually proves this exact sealed generation is unmodified.
        self._gt_finite_validated = True
        self._gt_finite_validated_id = id(tensor)
        self._gt_finite_validated_version = int(tensor._version)

    def _reattach_score_provenance(self) -> None:
        """Carry the camera's declared lineage across a device/copy boundary."""
        from eligsir.utils.score_contract import reattach_camera_score_provenance

        reattach_camera_score_provenance(self)

    def _canonicalize_streaming_materialization(self, image: torch.Tensor) -> torch.Tensor:
        frame = getattr(self, "_streaming_frame", None)
        if frame is None or image.device.type != "cpu":
            return image
        existing = getattr(frame, "mapping_frame", None)
        if existing is None:
            existing = getattr(frame, "_mapping_frame", None)
        if existing is not None:
            self._mapping_frame = existing
            return image
        try:
            depth_raw = getattr(frame, "_depth_np", None)
            if depth_raw is None:
                from eligsir.utils.streaming_frames import load_frame_depth_np
                depth_raw = load_frame_depth_np(frame)
                if depth_raw is not None:
                    frame._depth_np = depth_raw
            from eligsir.contracts.streaming_bridge import mapping_frame_from_streaming_frame
            mapping_frame = mapping_frame_from_streaming_frame(frame, image, depth_raw)
            frame._mapping_frame = mapping_frame
            self._mapping_frame = mapping_frame
            return mapping_frame.rgb.data
        except Exception as exc:
            raise RuntimeError(
                f"canonical streaming frame materialization failed for camera {self.image_name!r}"
            ) from exc

    def materialize_pending_image(self):
        if self._pending_image_loader is None:
            return
        image, gt_alpha_mask = self._pending_image_loader()
        self._pending_image_loader = None
        materialized = image.clamp(0.0, 1.0)
        if materialized.device.type != "cpu":
            raise ValueError("lazy image loader must return CPU-owned pixels")
        if gt_alpha_mask is not None:
            materialized = materialized * gt_alpha_mask.to(materialized.device)
        materialized = Camera._canonicalize_streaming_materialization(self, materialized)
        self.original_image = materialized
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(self, context="camera_lazy_materialization")
        if (
            self.original_image.shape[2] != self.image_width
            or self.original_image.shape[1] != self.image_height
        ):
            raise ValueError(
                "Camera.materialize_pending_image: decoded image size "
                f"{(self.original_image.shape[2], self.original_image.shape[1])} does not match "
                f"geometry-derived size {(self.image_width, self.image_height)}"
            )

    def rematerialize_native_image(self):
        """Restore native pixels after tier eviction using the retained loader."""
        return rematerialize_native_image(self)

    @classmethod
    def from_prepared_device_payloads(
        cls,
        *,
        rgb_hwc,
        depth_meters,
        c2w,
        camera_intrinsics,
        fx,
        fy,
        cx,
        cy,
        uid,
        world_view_transform=None,
        camera_center=None,
        projection_matrix=None,
        full_proj_transform=None,
        gsplat_viewmat=None,
        gsplat_camera_center=None,
    ):
        """Create a native camera directly around already-device-resident data.

        This is a deliberately narrow alternate constructor for the ready-source
        binder.  It does not accept CPU pixels, invoke ``Camera.__init__``, or
        call ``Tensor.to``: RGB/depth/pose/K must already share one device.
        Legacy camera construction remains unchanged.
        """

        tensors = (rgb_hwc, depth_meters, c2w, camera_intrinsics)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("prepared camera payloads must be torch tensors")
        if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
            raise ValueError("prepared RGB payload must be HWC with three channels")
        if depth_meters.ndim != 2 or tuple(rgb_hwc.shape[:2]) != tuple(depth_meters.shape):
            raise ValueError("prepared depth payload must match RGB height and width")
        if c2w.shape != (4, 4) or camera_intrinsics.shape != (3, 3):
            raise ValueError("prepared pose/K payload shapes must be 4x4 and 3x3")
        device = rgb_hwc.device
        if any(value.device != device for value in tensors):
            raise ValueError("prepared camera payloads must share one device")
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("prepared camera payloads must be float32")
        height, width = (int(rgb_hwc.shape[0]), int(rgb_hwc.shape[1]))
        camera = cls.__new__(cls)
        nn.Module.__init__(camera)
        camera.uid = int(uid)
        camera.colmap_id = int(uid)
        camera.R = None
        camera.T = None
        camera.image_name = f"{int(uid):06d}"
        camera.data_device = device
        camera.original_image = rgb_hwc.permute(2, 0, 1)
        camera.image_width = width
        camera.image_height = height
        camera.width = width
        camera.height = height
        camera.fx = float(fx)
        camera.fy = float(fy)
        camera.cx = float(cx)
        camera.cy = float(cy)
        camera.FoVx = 2.0 * math.atan(width / (2.0 * camera.fx))
        camera.FoVy = 2.0 * math.atan(height / (2.0 * camera.fy))
        camera.zfar = 100.0
        camera.znear = 0.01
        camera.trans = np.array([0.0, 0.0, 0.0])
        camera.scale = 1.0
        camera._pending_image_loader = None
        camera._native_image_loader = None
        camera._sensor_depth_cache = depth_meters
        camera.c2w = c2w
        precomputed = (
            world_view_transform,
            camera_center,
            projection_matrix,
            full_proj_transform,
            gsplat_viewmat,
            gsplat_camera_center,
        )
        if any(value is not None for value in precomputed):
            if not all(isinstance(value, torch.Tensor) for value in precomputed):
                raise ValueError("prepared camera geometry must be complete device tensors")
            expected = ((4, 4), (3,), (4, 4), (4, 4), (4, 4), (1, 3))
            if any(
                tuple(value.shape) != shape or value.device != device or value.dtype != torch.float32
                for value, shape in zip(precomputed, expected)
            ):
                raise ValueError("prepared camera geometry does not match the camera device contract")
            (
                camera.world_view_transform,
                camera.camera_center,
                camera.projection_matrix,
                camera.full_proj_transform,
                gsplat_viewmat,
                gsplat_camera_center,
            ) = precomputed
        else:
            camera.world_view_transform = torch.linalg.inv(camera.c2w).transpose(0, 1).contiguous()
            camera.camera_center = camera.c2w[:3, 3]
            projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
            projection[0, 0] = 2.0 * camera_intrinsics[0, 0] / width
            projection[1, 1] = 2.0 * camera_intrinsics[1, 1] / height
            projection[0, 2] = 2.0 * camera_intrinsics[0, 2] / width - 1.0
            projection[1, 2] = 2.0 * camera_intrinsics[1, 2] / height - 1.0
            projection[3, 2] = 1.0
            projection[2, 2] = camera.zfar / (camera.zfar - camera.znear)
            projection[2, 3] = -(camera.zfar * camera.znear) / (camera.zfar - camera.znear)
            camera.projection_matrix = projection.transpose(0, 1).contiguous()
            camera.full_proj_transform = (
                camera.world_view_transform.unsqueeze(0)
                .bmm(camera.projection_matrix.unsqueeze(0))
                .squeeze(0)
            )
            gsplat_viewmat = camera.world_view_transform.transpose(0, 1).contiguous()
            gsplat_camera_center = camera.camera_center.view(1, 3).contiguous()
        camera.gsplat = GsplatCameraTensors(
            viewmat=gsplat_viewmat,
            K=camera_intrinsics,
            camera_center=gsplat_camera_center,
            width=width,
            height=height,
        )
        camera._declare_prepared_score_provenance()
        return camera


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3][:3]


def _camera_device(cam, device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    owned = getattr(cam, "data_device", None)
    if owned is not None:
        return torch.device(owned)
    tensor = getattr(cam, "world_view_transform", None)
    if tensor is None:
        raise ValueError("camera has no owned device and no world_view_transform")
    return tensor.device


def _device_matches(actual: torch.device, target: torch.device) -> bool:
    """Treat an unindexed device (for example ``cuda``) as its active index."""
    if actual == target:
        return True
    try:
        actual = torch.device(actual)
        target = torch.device(target)
    except (TypeError, ValueError):
        return False
    if actual.type != target.type:
        return False
    return actual.index is None or target.index is None or actual.index == target.index


def _camera_intrinsics(cam) -> tuple[float, float, float, float]:
    width = int(cam.image_width)
    height = int(cam.image_height)
    fx = getattr(cam, "fx", None)
    fy = getattr(cam, "fy", None)
    cx = getattr(cam, "cx", None)
    cy = getattr(cam, "cy", None)
    if fx is None:
        fx = width / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    if fy is None:
        fy = height / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def make_gsplat_camera_tensors(cam, device=None) -> GsplatCameraTensors:
    target = _camera_device(cam, device)
    width = int(cam.image_width)
    height = int(cam.image_height)
    viewmat = (
        cam.world_view_transform.transpose(0, 1)
        .to(device=target, dtype=torch.float32)
        .contiguous()
    )
    fx, fy, cx, cy = _camera_intrinsics(cam)
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=target,
        dtype=torch.float32,
    )
    center = cam.camera_center.to(
        device=target, dtype=torch.float32
    ).view(1, 3).contiguous()
    return GsplatCameraTensors(
        viewmat=viewmat,
        K=K.contiguous(),
        camera_center=center,
        width=width,
        height=height,
    )


def make_gsplat_camera_batch(cams, device=None) -> GsplatCameraBatch:
    """Prepare a shape-compatible batch without rebuilding sealed camera K.

    READY cameras already own device-resident ``GsplatCameraTensors``. Reusing
    those tensors avoids constructing a fresh host-derived K tensor on every
    true-camera transaction. Legacy/stale cameras retain the established
    materialization fallback.
    """
    cams = tuple(cams)
    if not cams:
        raise ValueError("camera batch must be non-empty")
    target = _camera_device(cams[0], device)
    prepared = tuple(getattr(cam, "gsplat", None) for cam in cams)
    reuse = all(
        item is not None
        and _device_matches(item.viewmat.device, target)
        and _device_matches(item.K.device, target)
        and _device_matches(item.camera_center.device, target)
        and item.viewmat.dtype == torch.float32
        and item.K.dtype == torch.float32
        and item.camera_center.dtype == torch.float32
        for item in prepared
    )
    if reuse:
        viewmats = torch.stack([item.viewmat for item in prepared], dim=0).contiguous()
        Ks = torch.stack([item.K for item in prepared], dim=0).contiguous()
        centers = torch.cat([item.camera_center for item in prepared], dim=0).contiguous()
        widths = tuple(int(item.width) for item in prepared)
        heights = tuple(int(item.height) for item in prepared)
    else:
        viewmats = torch.stack(
            [c.world_view_transform.transpose(0, 1) for c in cams], dim=0
        ).to(device=target, dtype=torch.float32).contiguous()
        intrinsics = [_camera_intrinsics(c) for c in cams]
        Ks = torch.tensor(
            [
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
                for fx, fy, cx, cy in intrinsics
            ],
            device=target,
            dtype=torch.float32,
        ).contiguous()
        centers = torch.stack([c.camera_center for c in cams], dim=0).to(
            device=target, dtype=torch.float32
        ).contiguous()
        widths = tuple(int(c.image_width) for c in cams)
        heights = tuple(int(c.image_height) for c in cams)
    return GsplatCameraBatch(
        viewmats=viewmats,
        Ks=Ks,
        camera_centers=centers,
        widths=widths,
        heights=heights,
    )


def rematerialize_native_image(cam) -> bool:
    """Re-decode native pixels without changing tier-owned camera geometry."""
    loader = getattr(cam, "_native_image_loader", None)
    if loader is None:
        return False
    try:
        image, gt_alpha_mask = loader()
    except Exception:
        return False
    if image is None:
        return False
    restored = image.clamp(0.0, 1.0)
    if restored.device.type != "cpu":
        return False
    if gt_alpha_mask is not None:
        restored = restored * gt_alpha_mask.to(restored.device)
    canonicalize = getattr(cam, "_canonicalize_streaming_materialization", None)
    if callable(canonicalize):
        restored = canonicalize(restored)
    cam.original_image = restored
    from eligsir.utils.score_contract import declare_camera_score_provenance

    declare_camera_score_provenance(cam, context="camera_native_rematerialization")
    return True


def prepare_camera_for_render(cam, device=None):
    target = _camera_device(cam, device)
    if getattr(cam, "has_pending_image", False):
        cam.materialize_pending_image()
    gsplat = getattr(cam, "gsplat", None)
    if gsplat is None or not _device_matches(gsplat.viewmat.device, target):
        cam.gsplat = make_gsplat_camera_tensors(cam, device=target)
    if getattr(cam, "original_image", None) is not None and cam.original_image.device != target:
        is_pinned = getattr(cam.original_image, "is_pinned", None)
        non_blocking = bool(is_pinned()) if callable(is_pinned) else False
        cam.original_image = cam.original_image.to(
            target, non_blocking=non_blocking
        ).contiguous()
        from eligsir.utils.score_contract import reattach_camera_score_provenance

        reattach_camera_score_provenance(cam)
    return cam


def prepare_camera_batch_for_render(cams, device=None) -> None:
    """Prepare cold camera tensors together while preserving warm caches."""
    cams = list(cams)
    if not cams:
        return
    target = _camera_device(cams[0], device)
    for cam in cams:
        if getattr(cam, "has_pending_image", False):
            cam.materialize_pending_image()
    stale = [
        cam
        for cam in cams
        if getattr(cam, "gsplat", None) is None
        or not _device_matches(cam.gsplat.viewmat.device, target)
    ]
    if stale:
        batch = make_gsplat_camera_batch(stale, device=target)
        for index, cam in enumerate(stale):
            cam.gsplat = GsplatCameraTensors(
                viewmat=batch.viewmats[index],
                K=batch.Ks[index],
                camera_center=batch.camera_centers[index : index + 1],
                width=batch.widths[index],
                height=batch.heights[index],
            )
    for cam in cams:
        image = getattr(cam, "original_image", None)
        if image is not None and image.device != target:
            is_pinned = getattr(image, "is_pinned", None)
            non_blocking = bool(is_pinned()) if callable(is_pinned) else False
            cam.original_image = image.to(
                target, non_blocking=non_blocking
            ).contiguous()
            from eligsir.utils.score_contract import reattach_camera_score_provenance

            reattach_camera_score_provenance(cam)


def offload_camera_image(cam) -> bool:
    image = getattr(cam, "original_image", None)
    if image is None:
        return False
    if image.device.type != "cpu":
        cam.original_image = image.detach().to(device="cpu").contiguous()
        from eligsir.utils.score_contract import reattach_camera_score_provenance

        reattach_camera_score_provenance(cam)
    return True


def gpu_resident_image_count(cams) -> int:
    return sum(
        1
        for cam in cams or ()
        if getattr(getattr(cam, "original_image", None), "device", None) is not None
        and cam.original_image.device.type != "cpu"
    )


def gpu_resident_image_bytes(cams) -> int:
    total = 0
    for cam in cams or ():
        image = getattr(cam, "original_image", None)
        if image is not None and getattr(image, "device", None) is not None:
            if image.device.type != "cpu":
                total += int(image.numel()) * int(image.element_size())
    return total


def offload_camera_images(cams) -> int:
    moved = 0
    for cam in cams:
        image = getattr(cam, "original_image", None)
        was_gpu = (
            image is not None
            and getattr(image, "device", None) is not None
            and image.device.type != "cpu"
        )
        if offload_camera_image(cam) and was_gpu:
            moved += 1
    return moved
