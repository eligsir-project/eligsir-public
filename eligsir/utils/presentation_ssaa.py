"""Presentation-only supersampling antialiasing (SSAA) + unsharp-mask sharpening.

STRICTLY for human-facing presentation outputs (issue #1323). This module must
NEVER be wired into anything that feeds a scored numeric report field
(``mean_test_psnr``/``mean_train_psnr``/SSIM/LPIPS in ``utils/comparison_report.py``'s
``_compute_metrics``/``_render_image``/``_gt_image``, or the ``_visual_cam_render_cache``
that the TEST split's contact sheets share with those metrics). See CLAUDE.md's #1259
campaign isolation note for why: this project depends on those numbers being stable and
byte-identical across a long-running multi-arm PSNR campaign, and this feature is a
purely cosmetic, presentation-side add-on that must not be able to move them even
accidentally.

Full SSAA (higher-resolution re-render + downsample) is wired into (utils/comparison_report.py):
  - ``render_camera_path`` (``trajectory.mp4``) -- independent render pass, no scoring tie.
  - ``_write_visual_sheet`` (``train_contact_sheet.png``) -- independent render pass, no
    scoring tie; the GT comparison stays correct because the render is downsampled back to
    native resolution with ``downsample_tensor_chw`` BEFORE anything is compared against GT.

Sharpening-ONLY (no re-render, so no shared-render risk at all) is additionally applied to
the TEST-split contact sheets (``test_contact_sheet.png`` and friends, inside
``write_post_training_report``'s ``visual_test_cams`` loop): the unsharp-mask runs on the
already-quantized uint8 display copy (``r_u8``, built by ``_to_uint8_hwc`` from the SAME
``full_out["render"]`` tensor ``_compute_metrics`` later reads out of
``_visual_cam_render_cache``) -- the float tensor and the cache entry are never touched, so
this cannot move ``mean_test_psnr``/SSIM/LPIPS even though the pixels it feeds
(``test_contact_sheet.png``, ``normals_contact_sheet.png``, ``surf_normals_contact_sheet.png``,
``centers_contact_sheet.png``, ``rings_contact_sheet.png``) share a render pass with scoring.

Deliberately NOT wired into ``train_depth_timeseries.mp4`` or the overview sheets
(``centers_overview_sheet.png``/``rings_overview_sheet.png``) -- the former overlays sensor
depth / GT panels where a resolution mismatch from a separate high-res render pass is not
worth the added risk, and the latter draws per-Gaussian marker overlays where a downsample
would blur/misalign discrete markers. Both are candidates for a follow-up, not attempted
here to keep this change's scope and GPU-cost auditable.

Never call any of this from the live streaming/mapping hot path (``train_streaming.py``'s
per-iteration loop) -- it belongs exclusively to the offline post-training report-
generation phase.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def make_supersampled_camera(cam, factor: int):
    """Return a shallow copy of ``cam`` with ``image_width``/``image_height`` scaled by
    ``factor``, same FoV/pose/projection -- i.e. the same view frustum rendered at more
    pixels (classic SSAA). ``factor <= 1`` returns ``cam`` unchanged (no copy made).

    Works for both ``scene.cameras.MiniCam`` and ``scene.cameras.Camera`` because neither
    ``world_view_transform`` nor ``full_proj_transform`` depends on pixel resolution, only
    on FoV/near/far -- so nothing else needs to change for the render to come out at the
    higher resolution. Any renderer-side cached per-resolution tensors (e.g. a
    ``.gsplat`` attribute set by ``prepare_camera_for_render``) are dropped on the copy so
    a fresh supersampled pass is actually computed rather than silently reusing the
    original-resolution cache.

    Deliberately does NOT touch ``original_image``/GT-bearing attributes -- this is only
    ever used to build a throwaway render-only camera copy for a presentation image; nothing
    reads GT off the returned object.

    Cameras constructed with explicit pixel-space intrinsics (``fx``/``fy``/``cx``/``cy`` --
    e.g. TUM RGB-D cameras) must have those intrinsics scaled by the same ``factor`` as the
    resolution, or the K matrix desyncs from the enlarged canvas: the same focal length in
    pixels now maps onto a canvas that's ``factor``x larger, widening the effective FoV
    ("zoomed out") and shifting the principal point off-center (issue #1330). Cameras built
    from ``FoVx``/``FoVy`` only (``fx``/``fy``/``cx``/``cy`` left ``None``) are unaffected --
    ``scene/cameras.py``'s ``_camera_intrinsics`` derives them fresh from FoV + the
    already-scaled width/height whenever the explicit values are ``None``.
    """
    if factor is None or int(factor) <= 1:
        return cam
    factor = int(factor)
    cam2 = copy.copy(cam)
    cam2.image_width = int(round(cam.image_width * factor))
    cam2.image_height = int(round(cam.image_height * factor))
    if getattr(cam2, "width", None) is not None:
        cam2.width = cam2.image_width
    if getattr(cam2, "height", None) is not None:
        cam2.height = cam2.image_height
    for _attr in ("fx", "fy", "cx", "cy"):
        _val = getattr(cam2, _attr, None)
        if _val is not None:
            setattr(cam2, _attr, float(_val) * factor)
    if hasattr(cam2, "gsplat"):
        cam2.gsplat = None
    return cam2


def downsample_tensor_chw(img_chw: torch.Tensor, factor: int, target_hw: Optional[tuple] = None) -> torch.Tensor:
    """Area-downsample a rendered ``(C, H, W)`` float tensor back to its native
    resolution (or an explicit ``target_hw=(h, w)``). Used when the downsampled result
    must go on to be compared against/concatenated with a native-resolution GT tensor
    (e.g. ``_write_visual_sheet``'s train contact sheet) -- doing this at the tensor
    level, before ``_to_uint8_hwc``, keeps that downstream comparison shape-correct.
    No-op (returns ``img_chw`` unchanged) for ``factor <= 1`` and no explicit target.
    """
    if target_hw is None and (factor is None or int(factor) <= 1):
        return img_chw
    if target_hw is not None:
        out_h, out_w = int(target_hw[0]), int(target_hw[1])
    else:
        factor = int(factor)
        out_h = max(1, int(round(img_chw.shape[-2] / factor)))
        out_w = max(1, int(round(img_chw.shape[-1] / factor)))
    if (out_h, out_w) == (img_chw.shape[-2], img_chw.shape[-1]):
        return img_chw
    return F.interpolate(img_chw.unsqueeze(0), size=(out_h, out_w), mode="area").squeeze(0)


def downsample_area(img_hwc_uint8: np.ndarray, factor: int) -> np.ndarray:
    """Area-filter downsample by ``factor`` (the second half of SSAA). No-op for
    ``factor <= 1``."""
    if factor is None or int(factor) <= 1:
        return img_hwc_uint8
    factor = int(factor)
    import cv2

    h, w = img_hwc_uint8.shape[:2]
    out_w = max(1, int(round(w / factor)))
    out_h = max(1, int(round(h / factor)))
    return cv2.resize(img_hwc_uint8, (out_w, out_h), interpolation=cv2.INTER_AREA)


def unsharp_mask(img_hwc_uint8: np.ndarray, strength: float, radius: float = 1.5) -> np.ndarray:
    """Mild unsharp-mask sharpening. ``strength <= 0`` is a no-op. Keep ``strength``
    small (e.g. 0.1-0.3) -- over-sharpening produces visible halos/artifacts, which look
    worse than doing nothing (see CLAUDE.md issue #1323)."""
    if strength is None or float(strength) <= 0:
        return img_hwc_uint8
    import cv2

    strength = float(strength)
    blurred = cv2.GaussianBlur(img_hwc_uint8, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(img_hwc_uint8, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(img_hwc_uint8.dtype)


def presentation_postprocess(img_hwc_uint8: np.ndarray, *, supersample_factor: int = 1,
                              sharpen_strength: float = 0.0) -> np.ndarray:
    """Downsample (if supersampled) then sharpen an already-rendered uint8 HWC image.
    Pure post-process on a saved-image copy -- never call this on a tensor that also
    feeds a numeric report field."""
    out = downsample_area(img_hwc_uint8, supersample_factor)
    out = unsharp_mask(out, sharpen_strength)
    return out


class PresentationSSAAConfig:
    """Small value object bundling the three CLI-facing knobs so call sites don't have to
    thread three separate scalars through several layers of report-generation code."""

    __slots__ = ("enabled", "supersample_factor", "sharpen_strength")

    def __init__(self, enabled: bool = True, supersample_factor: int = 8,
                 sharpen_strength: float = 0.15):
        self.enabled = bool(enabled)
        self.supersample_factor = max(1, int(supersample_factor))
        self.sharpen_strength = max(0.0, float(sharpen_strength))

    @property
    def effective_factor(self) -> int:
        return self.supersample_factor if self.enabled else 1

    @property
    def effective_sharpen(self) -> float:
        return self.sharpen_strength if self.enabled else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return (f"PresentationSSAAConfig(enabled={self.enabled}, "
                f"supersample_factor={self.supersample_factor}, "
                f"sharpen_strength={self.sharpen_strength})")


def presentation_ssaa_kwargs_from_args(args) -> dict:
    """``**``-splat this at every ``write_post_training_report(...)`` call site to wire
    the ``--presentation_supersample_enabled``/``--presentation_supersample_factor``/
    ``--presentation_sharpen_strength`` CLI flags (``arguments/__init__.py``) through to
    its ``presentation_supersample_factor``/``presentation_sharpen_strength`` kwargs.
    Missing attributes (an ``args`` namespace built before this feature existed) fall
    back to this module's own defaults, matching every other ``*_from_args`` helper in
    this codebase (e.g. ``git_provenance_from_args``)."""
    cfg = config_from_args(args)
    return {
        "presentation_supersample_factor": cfg.effective_factor,
        "presentation_sharpen_strength": cfg.effective_sharpen,
    }


def config_from_args(args) -> PresentationSSAAConfig:
    """Build a ``PresentationSSAAConfig`` from an argparse namespace carrying the
    ``--presentation_supersample_enabled``/``--presentation_supersample_factor``/
    ``--presentation_sharpen_strength`` flags (``arguments/__init__.py``). Missing
    attributes fall back to this module's own defaults so callers that construct a bare
    ``argparse.Namespace()`` in a test don't need to set all three."""
    return PresentationSSAAConfig(
        enabled=getattr(args, "presentation_supersample_enabled", True),
        supersample_factor=getattr(args, "presentation_supersample_factor", 8),
        sharpen_strength=getattr(args, "presentation_sharpen_strength", 0.15),
    )
