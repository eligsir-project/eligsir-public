"""Heavy (torch/CUDA) render helpers for the comparison library.

Imported lazily by ``compare.fixed_view_strip`` so the pure-data comparison
functions stay importable without torch. Renders a saved checkpoint PLY from a
camera reconstructed out of the run's ``cameras.json`` (no GT image needed).
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from arguments import get_default_pipeline_params


def _parse_cfg_scalar(run_dir: str, key: str, default):
    from eligsir.eval.compare import parse_cfg_args, _find_cfg
    cfg = _find_cfg(run_dir)
    if not cfg:
        return default
    v = parse_cfg_args(cfg).get(key)
    if v is None:
        return default
    return v.strip("'\"")


def _build_minicam(entry: dict):
    """Reconstruct a MiniCam from a cameras.json entry (inverse of camera_to_JSON)."""
    import torch
    from scene.cameras import MiniCam
    from eligsir.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

    W = int(entry["width"]); H = int(entry["height"])
    fx = float(entry["fx"]); fy = float(entry["fy"])
    fovx = focal2fov(fx, W); fovy = focal2fov(fy, H)

    # camera_to_JSON wrote: W2C = inv([[R.T|T],[0,1]]); position=W2C[:3,3], rotation=W2C[:3,:3].
    # Invert that back to the (R, T) that getWorld2View2 consumes.
    rot = np.array(entry["rotation"], dtype=np.float64)      # W2C rotation
    pos = np.array(entry["position"], dtype=np.float64)      # W2C translation
    W2C = np.eye(4); W2C[:3, :3] = rot; W2C[:3, 3] = pos
    C2W = np.linalg.inv(W2C)
    R = C2W[:3, :3].transpose()   # undo the .transpose() in camera_to_JSON's Rt[:3,:3]=R.T
    T = W2C[:3, 3]

    znear, zfar = 0.01, 100.0
    world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full_proj)


def render_checkpoint_view(run_dir: str, ply_path: str, cam_index: int) -> Optional[np.ndarray]:
    """Render one camera from a checkpoint PLY. Returns HxWx3 uint8 or None.

    Requires torch/CUDA + the project deps; intended to run inside the container.
    """
    import torch
    from gaussian_renderer import render

    cams_path = os.path.join(run_dir, "cameras.json")
    if not os.path.isfile(cams_path) or not os.path.isfile(ply_path):
        return None
    cams = json.load(open(cams_path))
    if not cams:
        return None
    entry = cams[min(cam_index, len(cams) - 1)]

    layout = str(_parse_cfg_scalar(run_dir, "model_layout", "gsplat"))
    sh_degree = int(float(_parse_cfg_scalar(run_dir, "sh_degree", 3)))
    if layout == "gsplat":
        from scene.gsplat_model import GsplatGaussianModel as Model
    else:
        from scene.gaussian_model import GaussianModel as Model
    gaussians = Model(sh_degree)
    gaussians.load_ply(ply_path)

    cam = _build_minicam(entry)
    pipe = get_default_pipeline_params()
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        out = render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)
    img = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img
