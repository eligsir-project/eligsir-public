#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import torch
from eligsir.utils.general_utils import PILtoTorch
from eligsir.utils.graphics_utils import fov2focal

WARNED = False


def _scale_intrinsics(cam_info, resolution, orig_w, orig_h):
    fx = getattr(cam_info, "fx", None)
    fy = getattr(cam_info, "fy", None)
    cx = getattr(cam_info, "cx", None)
    cy = getattr(cam_info, "cy", None)
    if fx is None or fy is None or cx is None or cy is None:
        return None, None, None, None

    sx = float(resolution[0]) / float(orig_w)
    sy = float(resolution[1]) / float(orig_h)
    return float(fx) * sx, float(fy) * sy, float(cx) * sx, float(cy) * sy


def _resolve_geometry(args, cam_info, resolution_scale, rgb_uint8=None):
    """Pure-metadata resolution: figure out orig/target (w, h) and the
    scaled intrinsics without touching any pixel data. Split out of
    ``loadCam`` so the lazy (``build_image=False``) path can compute a
    camera's geometry (needed for admission / visibility checks) without
    paying for the RGB decode + H2D upload below.
    """
    # cam_info.image may be None when the streaming fast path skipped building a
    # PIL (geometry was derived from the decoded array shape). Fall back to the
    # explicit width/height (always set), then to the rgb_uint8 shape.
    if cam_info.image is not None:
        orig_w, orig_h = cam_info.image.size
    elif getattr(cam_info, "width", None) and getattr(cam_info, "height", None):
        orig_w, orig_h = int(cam_info.width), int(cam_info.height)
    elif rgb_uint8 is not None:
        orig_h, orig_w = rgb_uint8.shape[:2]
    else:
        raise ValueError("loadCam: cam_info has no image, width/height, or rgb_uint8")

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    fx, fy, cx, cy = _scale_intrinsics(cam_info, resolution, orig_w, orig_h)
    return orig_w, orig_h, resolution, (fx, fy, cx, cy)


def _decode_image_tensor(args, cam_info, orig_w, orig_h, resolution, rgb_uint8=None):
    """The expensive part: decode/rescale RGB and (for the streaming fast
    path) do the host->device upload. Returns (gt_image, loaded_mask), both
    still un-clamped / not yet multiplied by the alpha mask -- Camera
    finishes that (see ``Camera.attach_image``).
    """
    # Fast path (live streaming): when a decoded uint8 HxWx{3,4} array is supplied
    # AND no rescale is needed (target == native), skip the numpy->PIL->numpy
    # round-trip in PILtoTorch. PIL.resize to the same size is a verified no-op, so
    # this is byte-identical to PILtoTorch(cam_info.image, resolution) while cutting
    # the conversion from ~4.6ms to ~1.4ms on a 720p frame. Any actual downscale
    # falls back to the PIL path (cv2/torch resize != PIL resampling).
    if (
        rgb_uint8 is not None
        and getattr(rgb_uint8, "dtype", None) == np.uint8
        and rgb_uint8.ndim == 3
        and rgb_uint8.shape[1::-1] == (orig_w, orig_h)  # (W,H) match
        and resolution == (orig_w, orig_h)
    ):
        # Keep decoded pixels CPU-owned. DeviceTransferOwner performs the sole
        # first-use H2D after semantic admission, in a pinned shape bucket.
        # issue #617 (same defect shape as #593, eligsir/mapping/resolution_cache.py):
        # np.ascontiguousarray guarantees C-contiguity but NOT writability -- an
        # already-contiguous read-only array (e.g. a ROS2 Image message decoded via
        # cv_bridge.imgmsg_to_cv2, which wraps the message's immutable `.data` buffer
        # in-place without copying when the underlying RMW returns `bytes`, or a PIL
        # `np.asarray(...)`-backed frame) would pass this guard unchanged, and the
        # resulting torch.from_numpy tensor would alias non-writable memory -- formally
        # undefined behavior on any later in-place write. Check both guarantees and
        # copy only when either is actually missing.
        if not (rgb_uint8.flags["C_CONTIGUOUS"] and rgb_uint8.flags["WRITEABLE"]):
            rgb_uint8 = np.array(rgb_uint8, copy=True, order="C")
        _t = torch.from_numpy(rgb_uint8)  # [H,W,C] uint8 on CPU
        resized_image_rgb = (_t.permute(2, 0, 1).to(torch.float32) / 255.0)
    else:
        # Slow path (an actual downscale, or no uint8 array). cam_info.image may be
        # None when the streaming fast path skipped building a PIL (geometry came
        # from the array shape); rebuild it from rgb_uint8 here, only when actually
        # needed. Byte-identical to the prior load_frame_rgb -> PILtoTorch path.
        _src = cam_info.image
        if _src is None and rgb_uint8 is not None:
            from PIL import Image as _Image
            # issue #643: do NOT slice `[..., :3]` here. That silently dropped a
            # 4th (alpha) channel on this path only, while the uint8 fast path
            # above keeps all channels -- so the same HxWx4 array produced an
            # alpha mask on one path and not the other. Keep every channel and
            # let the single alpha check below decide, on both paths.
            if rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 4:
                _src = _Image.fromarray(rgb_uint8, "RGBA")
            else:
                _src = _Image.fromarray(rgb_uint8)
        resized_image_rgb = PILtoTorch(_src, resolution)

    # `resized_image_rgb` is a 3-D CHW tensor on BOTH paths above: the fast path
    # explicitly does `.permute(2, 0, 1)` on an HWC array, and PILtoTorch does
    # the same (and unsqueezes single-channel input to CHW). So the CHANNEL
    # dimension is shape[0]; shape[1] is H.
    #
    # issue #643 (fix): this test used to read `resized_image_rgb.shape[1] == 4`
    # (inherited verbatim from upstream INRIA 3DGS). That is the HEIGHT, so it
    # was wrong in both directions:
    #   * a real RGBA image (C == 4) never produced a `loaded_mask`, and
    #   * any image exactly 4 pixels tall produced a bogus mask -- `[3:4, ...]`
    #     on a 3-channel tensor is an EMPTY (0, H, W) tensor, which then blows up
    #     in `Camera.__init__`'s `original_image *= gt_alpha_mask`.
    # An explicit `ndim` guard pins the rank so a future 4-D (B, C, H, W) change
    # cannot silently re-alias the dimensions.
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    if resized_image_rgb.ndim == 3 and resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    return gt_image, loaded_mask


def loadCam(args, id, cam_info, resolution_scale, rgb_uint8=None, build_image=True):
    """Build a :class:`~scene.cameras.Camera`.

    ``build_image=False`` (lazy mode) skips the RGB decode + host->device
    upload entirely and returns a Camera with geometry only (R/T/FoV/
    intrinsics/width/height) plus a ``_pending_image`` closure that a
    caller must invoke (``Camera.materialize_pending_image()``) before the
    camera's ``original_image`` is actually used (e.g. for a training
    loss). Used by streaming keyframe admission (see
    ``scene/streaming_scene.py:_frame_to_camera``) so candidate frames
    that end up REJECTED never pay the image upload cost — see
    ``docs/superpowers/plans/...`` D2 in the Wave 2 consolidation plan.
    """
    orig_w, orig_h, resolution, (fx, fy, cx, cy) = _resolve_geometry(
        args, cam_info, resolution_scale, rgb_uint8=rgb_uint8
    )

    if not build_image:
        def _pending_image_loader():
            return _decode_image_tensor(
                args, cam_info, orig_w, orig_h, resolution, rgb_uint8=rgb_uint8
            )

        return Camera(
            colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
            FoVx=cam_info.FovX, FoVy=cam_info.FovY,
            image=None, gt_alpha_mask=None,
            image_name=cam_info.image_name, uid=id, data_device=args.data_device,
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=resolution[0], height=resolution[1],
            pending_image_loader=_pending_image_loader,
        )

    gt_image, loaded_mask = _decode_image_tensor(
        args, cam_info, orig_w, orig_h, resolution, rgb_uint8=rgb_uint8
    )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  fx=fx, fy=fy, cx=cx, cy=cy)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : camera.fy if camera.fy is not None else fov2focal(camera.FovY, camera.height),
        'fx' : camera.fx if camera.fx is not None else fov2focal(camera.FovX, camera.width),
        'cx' : camera.cx if camera.cx is not None else camera.width / 2.0,
        'cy' : camera.cy if camera.cy is not None else camera.height / 2.0,
    }
    return camera_entry
