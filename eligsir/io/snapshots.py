"""Snapshot helpers shared by offline and streaming save paths."""

from __future__ import annotations

import os
import types
from collections import OrderedDict
from typing import Any, Callable


def _torch():
    import torch

    return torch


def cpu_deep(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_deep(v) for v in value)
    return value


def capture_checkpoint(gaussians) -> Any:
    """Snapshot Gaussian model state as CPU tensors for async checkpointing."""
    state = gaussians.capture()
    if isinstance(state, dict):
        return {k: cpu_deep(v) for k, v in state.items()}

    torch = _torch()
    return tuple(
        v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for v in state
    )


def mutate_and_capture_iteration_outputs(
    gaussians,
    *,
    mutate: Callable[[], None],
    capture_ply: bool,
    capture_checkpoint_state: bool,
) -> dict[str, Any]:
    """Run final population mutations before capturing persistence state."""
    mutate()
    return {
        "ply": snapshot_gaussians_for_ply(gaussians) if capture_ply else None,
        "checkpoint": capture_checkpoint(gaussians) if capture_checkpoint_state else None,
    }


def snapshot_gaussians_for_ply(gaussians) -> dict[str, Any]:
    """Capture Gaussian parameters as CPU tensors for async PLY writing.

    shN/_features_rest is truncated to the model's current
    active_sh_degree band count here (at capture time, synchronously on
    the live model), mirroring save_ply()'s own truncation -- the write
    happens later, asynchronously, by which point active_sh_degree may
    have changed, so the degree and the correctly-truncated data must both
    be captured together now.
    """
    n_active_bands = (gaussians.active_sh_degree + 1) ** 2 - 1

    if hasattr(gaussians, "params") and isinstance(gaussians.params, (dict, OrderedDict)):
        params = gaussians.params
        return {
            "attrs": gaussians.construct_list_of_attributes(),
            "active_sh_degree": gaussians.active_sh_degree,
            "means": params["means"].detach().contiguous().cpu(),
            "sh0": params["sh0"].detach().contiguous().cpu(),
            "shN": params["shN"][:, :n_active_bands, :].detach().contiguous().cpu(),
            "opacities": params["opacities"].detach().unsqueeze(-1).contiguous().cpu(),
            "scales": params["scales"].detach().contiguous().cpu(),
            "quats": params["quats"].detach().contiguous().cpu(),
        }

    return {
        "attrs": gaussians.construct_list_of_attributes(),
        "active_sh_degree": gaussians.active_sh_degree,
        "means": gaussians._xyz.detach().contiguous().cpu(),
        "sh0": gaussians._features_dc.detach().contiguous().cpu(),
        "shN": gaussians._features_rest[:, :n_active_bands, :].detach().contiguous().cpu(),
        "opacities": gaussians._opacity.detach().contiguous().cpu(),
        "scales": gaussians._scaling.detach().contiguous().cpu(),
        "quats": gaussians._rotation.detach().contiguous().cpu(),
    }


def write_ply_from_snapshot(snapshot: dict[str, Any], out_path: str) -> None:
    """Build and write a PLY file from a CPU snapshot."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    xyz = snapshot["means"].numpy()
    normals = np.zeros_like(xyz)
    f_dc = snapshot["sh0"].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    f_rest = snapshot["shN"].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    opacities = snapshot["opacities"].numpy()
    scale = snapshot["scales"].numpy()
    rotation = snapshot["quats"].numpy()

    dtype_full = [(attr, "f4") for attr in snapshot["attrs"]]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData(
        [PlyElement.describe(elements, "vertex")],
        comments=[f"active_sh_degree {snapshot['active_sh_degree']}"],
    ).write(out_path)


def subset_ply_snapshot(snapshot: dict[str, Any], mask) -> dict[str, Any]:
    """Return a masked PLY snapshot using a CPU bool mask."""
    mask_cpu = mask.detach().to(device="cpu", dtype=_torch().bool)
    return {
        "attrs": snapshot["attrs"],
        "active_sh_degree": snapshot["active_sh_degree"],
        "means": snapshot["means"][mask_cpu],
        "sh0": snapshot["sh0"][mask_cpu],
        "shN": snapshot["shN"][mask_cpu],
        "opacities": snapshot["opacities"][mask_cpu],
        "scales": snapshot["scales"][mask_cpu],
        "quats": snapshot["quats"][mask_cpu],
    }


def snapshot_gaussians_for_render(gaussians):
    """Detached, render-capable snapshot of the live Gaussian model.

    Safe to hand to a background thread while the main thread continues
    mutating the live model in-place (optimizer steps, MCMC relocation).
    Unlike snapshot_gaussians_for_ply, this keeps tensors on their current
    device (no .cpu() move) and preserves full SH-band width (no
    active_sh_degree truncation) since it's read directly by
    render()/render_batch() (eligsir/mapping/renderer/gsplat_backend.py),
    not written to a PLY file.

    The renderer reads exactly these attributes from its `pc` parameter:
    get_xyz, get_scaling, get_rotation, _rotation (read directly instead of
    get_rotation when sparse_grad is active), get_opacity, _features_dc,
    _features_rest, active_sh_degree. Since a snapshot has no gradients and
    is never optimized, get_rotation and _rotation are set to the same
    already-activated tensor -- the sparse/dense distinction only matters
    for gradient flow, which a frozen snapshot has none of.
    """
    if hasattr(gaussians, "params") and isinstance(gaussians.params, (dict, OrderedDict)):
        params = gaussians.params
        xyz = params["means"].detach().clone()
        scales = torch.exp(params["scales"]).detach().clone()
        quats = torch.nn.functional.normalize(params["quats"], dim=-1).detach().clone()
        opacity = torch.sigmoid(params["opacities"]).unsqueeze(-1).detach().clone()
        features_dc = params["sh0"].detach().clone()
        features_rest = params["shN"].detach().clone()
    else:
        xyz = gaussians.get_xyz.detach().clone()
        scales = gaussians.get_scaling.detach().clone()
        quats = gaussians.get_rotation.detach().clone()
        opacity = gaussians.get_opacity.detach().clone()
        features_dc = gaussians._features_dc.detach().clone()
        features_rest = gaussians._features_rest.detach().clone()

    return types.SimpleNamespace(
        get_xyz=xyz,
        get_scaling=scales,
        get_rotation=quats,
        _rotation=quats,
        get_opacity=opacity,
        _features_dc=features_dc,
        _features_rest=features_rest,
        active_sh_degree=int(gaussians.active_sh_degree),
    )
