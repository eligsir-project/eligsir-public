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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixFromIntrinsics(znear, zfar, fx, fy, cx, cy, width, height):
    """Asymmetric-frustum projection matrix for an off-centre principal point.

    ``getProjectionMatrix`` above derives a *symmetric* frustum from FoV alone
    (``left = -right``, ``bottom = -top``), so it cannot represent a camera
    whose principal point is not at the image centre — which is exactly what a
    supervision crop is. The rasterizer itself does not care (it is handed a
    real 3x3 ``K``; see ``eligsir/mapping/renderer/gsplat_backend.py``), but
    ``Camera.projection_matrix`` / ``full_proj_transform`` are consumed
    elsewhere (frustum culling, viewers) and would silently disagree with the
    render for an off-centre camera.

    This function computes the same matrix layout from full pinhole intrinsics.
    For a centred camera (``cx = width/2``, ``cy = height/2``) it reproduces
    ``getProjectionMatrix(znear, zfar, focal2fov(fx, width), focal2fov(fy,
    height))`` exactly. No existing caller is changed; this is additive.

    Issue #310 (exact crop-camera geometry). See
    ``docs/architecture/2026-08-29-tile-crop-supervision-implementation-plan.md``
    section 4.2.
    """
    # Derived directly from the pixel mapping the rasterizer's own K performs,
    # rather than from OpenGL left/right/bottom/top. The two differ in the SIGN
    # of the shear terms here: `getProjectionMatrix` sets `z_sign = +1` (camera
    # looks down +z, COLMAP convention), whereas the textbook `(r+l)/(r-l)`
    # form assumes OpenGL's -z. For a symmetric frustum both shear terms are
    # zero, so the existing function never exercised the difference and gives
    # no guidance; getting it backwards mirrors the image about its centre.
    #
    #   u      = fx * x/z + cx
    #   ndc_x  = 2u/width - 1
    #          = (2*fx/width) * (x/z) + (2*cx/width - 1)
    #   clip_x = (2*fx/width) * x + (2*cx/width - 1) * z        [w = z]
    #
    # and symmetrically for y. Camera-space +y runs the same direction as the
    # pixel row index, so no axis flip is involved.
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[0, 2] = 2.0 * cx / width - 1.0
    P[1, 2] = 2.0 * cy / height - 1.0
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))