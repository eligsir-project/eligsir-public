"""Fast (no-rasterization) frustum-projection visibility for the observation DB.

Must run on EVERY incoming candidate frame before the admission decision,
not just admitted ones (see docs/superpowers/specs/2026-07-25-keyframe-
database-design.md section 3.1) -- a real differentiable render pass is far
too slow to run unconditionally, so this is a deliberately approximate
projection + frustum cull, no depth/occlusion test. (A future refinement
could depth-test against the existing streaming_seed_voxel_size occupancy
grid for tighter occlusion; out of scope for this plan.)
"""
from __future__ import annotations

import torch


def project_visible_gaussian_uids(
    gaussians,
    cam,
    near: float = 0.05,
    far: float = 20.0,
) -> torch.Tensor:
    """Return the gaussian_uid values whose means project inside cam's frustum.

    Args:
        gaussians: model exposing .get_xyz ([N,3]) and .gaussian_uid ([N] int64).
        cam: camera exposing .R (R_c2w), .T (t_w2c), .fx/.fy/.cx/.cy,
            .image_width/.image_height -- the same convention as
            scene/cameras.py's Camera and tests/test_pose_corrections.py's
            _FakeCamera.
        near: near-plane distance (metres); points closer are excluded.
        far: far-plane distance (metres); points farther are excluded.

    Returns:
        int64 tensor of visible gaussian_uid values (possibly empty).
    """
    xyz = gaussians.get_xyz
    device = xyz.device
    R_c2w = torch.as_tensor(cam.R, dtype=torch.float32, device=device)
    t_w2c = torch.as_tensor(cam.T, dtype=torch.float32, device=device)

    # World -> camera: R_w2c = R_c2w^T; t_w2c already given directly.
    R_w2c = R_c2w.T
    xyz_cam = xyz @ R_w2c.T + t_w2c  # [N, 3]

    z = xyz_cam[:, 2]
    in_depth_range = (z > near) & (z < far)

    # Avoid div-by-zero for excluded points; guard before projecting.
    safe_z = torch.where(in_depth_range, z, torch.ones_like(z))
    u = cam.fx * (xyz_cam[:, 0] / safe_z) + cam.cx
    v = cam.fy * (xyz_cam[:, 1] / safe_z) + cam.cy

    in_bounds = (
        (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height)
    )
    visible_mask = in_depth_range & in_bounds
    return gaussians.gaussian_uid[visible_mask]
