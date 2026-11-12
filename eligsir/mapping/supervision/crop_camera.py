"""Exact crop-camera geometry for spatially selective supervision (issue #310).

**Why this can be small.** The rasterizer already takes a real 3x3 intrinsics
matrix: ``eligsir/mapping/renderer/gsplat_backend.py`` builds
``K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`` from ``camera.fx/fy/cx/cy`` (falling back
to FoV-derived focals and a centred principal point only when those are unset)
and hands it to ``gsplat.rendering.rasterization(Ks=...)`` with ``width`` and
``height`` passed separately. An off-centre principal point is therefore
already supported end to end, and a crop camera is exactly "shift ``cx``/``cy``
by the crop origin, scale by the resize factor, shrink ``image_width``/
``image_height``". No renderer change is required.

**The one gap this module closes.** ``Camera.projection_matrix`` /
``full_proj_transform`` (``scene/cameras.py``) are built from
``getProjectionMatrix(znear, zfar, FoVx, FoVy)``, whose frustum is symmetric by
construction and cannot represent an off-centre principal point. For a centred
camera that is harmless (the rasterizer uses ``K``); for a crop camera it would
silently disagree with the render in every consumer of those fields. The crop
camera built here uses
``utils.graphics_utils.getProjectionMatrixFromIntrinsics`` instead.

**Resize convention.** The resize half of a crop transform reuses, verbatim, the
half-pixel convention already established by the resolution-tier materialization
boundary (``eligsir/mapping/resolution_materialization.py``)::

    fx_new = fx * scale_x
    cx_new = (cx + 0.5) * scale_x - 0.5

Deviating from it would make "crop then resize" and "resize then crop" disagree
by half a pixel, and the parity tests would then encode the bug rather than
catch it.

Torch is imported lazily so the pure intrinsics/pixel math in this module stays
importable (and CI-testable) in the torch-free CPU image.

Scope: geometry only. Nothing here decides *which* crop to take. See
``docs/architecture/2026-08-29-tile-crop-supervision-implementation-plan.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CropTransform:
    """A deterministic, serialisable crop-then-resize of a parent image.

    ``x0``/``y0``/``width``/``height`` are the crop rectangle in **parent pixel
    coordinates** (integers; the crop must lie fully inside the parent).
    ``out_width``/``out_height`` are the supervision resolution the crop is
    resampled to — equal to ``width``/``height`` for a pure crop with no
    rescale, which is the native-detail-promotion case #309 is about.

    ``parent_width``/``parent_height`` are carried so the transform is
    self-describing: a run manifest recording only this object can state
    exactly what was cropped from what, per #310 item 6 (crop lineage).
    """

    x0: int
    y0: int
    width: int
    height: int
    out_width: int
    out_height: int
    parent_width: int
    parent_height: int

    def __post_init__(self) -> None:
        for name in ("width", "height", "out_width", "out_height"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"CropTransform.{name} must be positive, got {getattr(self, name)!r}")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"CropTransform origin must be non-negative, got ({self.x0}, {self.y0})")
        if self.x0 + self.width > self.parent_width or self.y0 + self.height > self.parent_height:
            raise ValueError(
                f"crop ({self.x0},{self.y0},{self.width},{self.height}) does not fit inside "
                f"parent {self.parent_width}x{self.parent_height}"
            )

    @classmethod
    def full_frame(cls, width: int, height: int) -> "CropTransform":
        """The identity transform — used as the parity reference in tests."""
        return cls(
            x0=0, y0=0, width=int(width), height=int(height),
            out_width=int(width), out_height=int(height),
            parent_width=int(width), parent_height=int(height),
        )

    @property
    def scale_x(self) -> float:
        return float(self.out_width) / float(self.width)

    @property
    def scale_y(self) -> float:
        return float(self.out_height) / float(self.height)

    @property
    def is_identity(self) -> bool:
        return (
            self.x0 == 0
            and self.y0 == 0
            and self.width == self.parent_width
            and self.height == self.parent_height
            and self.out_width == self.width
            and self.out_height == self.height
        )

    @property
    def supervision_pixels(self) -> int:
        """Pixels this crop contributes to the objective (#310 telemetry)."""
        return int(self.out_width) * int(self.out_height)

    def as_dict(self) -> dict:
        return {
            "x0": int(self.x0),
            "y0": int(self.y0),
            "width": int(self.width),
            "height": int(self.height),
            "out_width": int(self.out_width),
            "out_height": int(self.out_height),
            "parent_width": int(self.parent_width),
            "parent_height": int(self.parent_height),
        }


@dataclass(frozen=True)
class CropIntrinsics:
    """Pinhole intrinsics of a crop camera, in crop pixel coordinates."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def fov_x(self) -> float:
        """Total horizontal FoV. Only a faithful summary for a centred crop.

        Retained because ``Camera`` carries ``FoVx``/``FoVy`` and several
        non-rendering consumers read them. The *rendering* path uses ``fx``/
        ``cx`` directly, so an off-centre crop is still exact where it matters.
        """
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fov_y(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))


def parent_intrinsics(camera) -> CropIntrinsics:
    """Resolve a parent camera's intrinsics, filling FoV-only cameras in.

    Mirrors the exact fallback ladder the renderer itself uses
    (``gsplat_backend._camera_tensors``): explicit ``fx/fy/cx/cy`` when
    present, else focals derived from ``FoVx``/``FoVy`` and a centred principal
    point. Resolving it the same way here is what guarantees a crop derived
    from a FoV-only parent still lands where the parent's own render put it.
    """
    width = int(getattr(camera, "image_width", 0) or getattr(camera, "width", 0) or 0)
    height = int(getattr(camera, "image_height", 0) or getattr(camera, "height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("parent camera has no usable image_width/image_height")
    fx = getattr(camera, "fx", None)
    fy = getattr(camera, "fy", None)
    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)
    if fx is None:
        fov_x = getattr(camera, "FoVx", None)
        if fov_x is None:
            raise ValueError("parent camera has neither fx nor FoVx")
        fx = width / (2.0 * math.tan(float(fov_x) / 2.0))
    if fy is None:
        fov_y = getattr(camera, "FoVy", None)
        fy = (
            height / (2.0 * math.tan(float(fov_y) / 2.0))
            if fov_y is not None
            else float(fx)
        )
    return CropIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx) if cx is not None else width / 2.0,
        cy=float(cy) if cy is not None else height / 2.0,
        width=width,
        height=height,
    )


def derive_crop_intrinsics(parent: CropIntrinsics, transform: CropTransform) -> CropIntrinsics:
    """Exact intrinsics of ``transform`` applied to ``parent``.

    Crop shifts the principal point by the crop origin and leaves the focal
    length alone (a crop changes the field of view, not the pixel scale).
    Resize then scales both, using the resolution-materialization half-pixel
    convention (see the module docstring).
    """
    if int(parent.width) != int(transform.parent_width) or int(parent.height) != int(transform.parent_height):
        raise ValueError(
            f"transform declares parent {transform.parent_width}x{transform.parent_height} "
            f"but intrinsics are for {parent.width}x{parent.height}"
        )
    sx = transform.scale_x
    sy = transform.scale_y
    return CropIntrinsics(
        fx=parent.fx * sx,
        fy=parent.fy * sy,
        cx=(parent.cx - transform.x0 + 0.5) * sx - 0.5,
        cy=(parent.cy - transform.y0 + 0.5) * sy - 0.5,
        width=int(transform.out_width),
        height=int(transform.out_height),
    )


def project_to_crop_pixel(
    parent_pixel: Tuple[float, float], transform: CropTransform
) -> Tuple[float, float]:
    """Map a parent-image pixel to the corresponding crop-image pixel.

    Same half-pixel convention as :func:`derive_crop_intrinsics`, which is what
    makes the #310 parity test meaningful: projecting a 3D point through the
    parent camera and then through this map must land on the same pixel as
    projecting it directly through the crop camera.
    """
    u, v = float(parent_pixel[0]), float(parent_pixel[1])
    return (
        (u - transform.x0 + 0.5) * transform.scale_x - 0.5,
        (v - transform.y0 + 0.5) * transform.scale_y - 0.5,
    )


def project_point(intrinsics: CropIntrinsics, point_camera) -> Tuple[float, float]:
    """Pinhole-project a camera-space point ``(x, y, z)`` to a pixel."""
    x, y, z = (float(point_camera[0]), float(point_camera[1]), float(point_camera[2]))
    if z <= 0.0:
        raise ValueError(f"point is not in front of the camera (z={z})")
    return (intrinsics.fx * x / z + intrinsics.cx, intrinsics.fy * y / z + intrinsics.cy)


class CropCamera:
    """A duck-typed camera view of one crop of a parent camera.

    Deliberately NOT a ``scene.cameras.Camera`` subclass: it shares the
    parent's pose tensors by reference (a crop never moves the camera) and
    exposes exactly the attribute surface the renderer reads. The parent's
    cached ``_gsplat_K``/``gsplat`` tensors are *not* inherited — inheriting a
    stale full-frame ``K`` is the one way this abstraction could silently
    render the wrong thing.
    """

    def __init__(self, parent, transform: CropTransform, intrinsics: CropIntrinsics, image=None):
        from eligsir.utils.graphics_utils import getProjectionMatrixFromIntrinsics

        self.parent_camera = parent
        self.crop_transform = transform
        self.uid = getattr(parent, "uid", None)
        self.colmap_id = getattr(parent, "colmap_id", None)
        self.image_name = f"{getattr(parent, 'image_name', 'frame')}#crop{transform.x0}_{transform.y0}"

        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.cx
        self.cy = intrinsics.cy
        self.FoVx = intrinsics.fov_x
        self.FoVy = intrinsics.fov_y
        self.image_width = int(intrinsics.width)
        self.image_height = int(intrinsics.height)
        self.width = self.image_width
        self.height = self.image_height

        self.znear = float(getattr(parent, "znear", 0.01))
        self.zfar = float(getattr(parent, "zfar", 100.0))
        self.data_device = getattr(parent, "data_device", None)

        # Pose is shared by reference: a crop is a different sensor model on the
        # same physical viewpoint.
        self.world_view_transform = parent.world_view_transform
        self.camera_center = parent.camera_center
        self.R = getattr(parent, "R", None)
        self.T = getattr(parent, "T", None)

        self.projection_matrix = (
            getProjectionMatrixFromIntrinsics(
                self.znear, self.zfar,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                intrinsics.width, intrinsics.height,
            )
            .to(self.world_view_transform.device)
            .transpose(0, 1)
            .contiguous()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        if image is not None:
            self.original_image = image

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        t = self.crop_transform
        return (
            f"CropCamera(parent={getattr(self.parent_camera, 'uid', None)}, "
            f"rect=({t.x0},{t.y0},{t.width},{t.height}), out={t.out_width}x{t.out_height})"
        )


def crop_image(image, transform: CropTransform):
    """Crop (and resize, when the transform rescales) a ``[C, H, W]`` tensor.

    The RGB resample uses ``mode="area"``, matching the ``legacy``/``masked``
    resolution-tier resampling path so a crop resized to a tier's density is
    photometrically comparable with that tier's own full-frame supervision.
    """
    import torch.nn.functional as F

    cropped = image[
        ...,
        transform.y0 : transform.y0 + transform.height,
        transform.x0 : transform.x0 + transform.width,
    ]
    if transform.out_width == transform.width and transform.out_height == transform.height:
        return cropped
    return F.interpolate(
        cropped.unsqueeze(0).float(),
        size=(transform.out_height, transform.out_width),
        mode="area",
    ).squeeze(0)


def crop_camera_from(
    camera, transform: CropTransform, *, image: Optional[object] = None, crop_parent_image: bool = False
) -> CropCamera:
    """Build an exact crop camera for ``transform`` over ``camera``.

    ``image`` supplies the crop's ground truth explicitly. With
    ``crop_parent_image=True`` and no explicit ``image``, the parent's
    ``original_image`` is cropped here instead. Neither is required — the
    geometry is usable on its own (e.g. to render a crop for diagnostics).
    """
    parent = parent_intrinsics(camera)
    intrinsics = derive_crop_intrinsics(parent, transform)
    if image is None and crop_parent_image:
        parent_image = getattr(camera, "original_image", None)
        if parent_image is not None:
            image = crop_image(parent_image, transform)
    return CropCamera(camera, transform, intrinsics, image=image)
