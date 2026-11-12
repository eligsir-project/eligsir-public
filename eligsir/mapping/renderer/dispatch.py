"""Lazy renderer backend dispatch.

GPU Gaussian states keep using the production gsplat backend.  CPU Gaussian
states use the tiny differentiable PyTorch reference backend, which is intended
only for correctness tests and synthetic CI fixtures.
"""

from __future__ import annotations

from typing import Any


def _resolved_backend(pc: Any, pipe: Any) -> str:
    requested = str(getattr(pipe, "renderer_backend", "auto")).lower()
    if requested != "auto":
        return requested

    xyz = getattr(pc, "get_xyz")
    xyz = xyz() if callable(xyz) else xyz
    device = getattr(xyz, "device", None)
    device_type = getattr(device, "type", None)
    if device_type == "cpu" or str(device) == "cpu":
        return "torch_cpu_reference"
    return "gsplat"


def render(
    viewpoint_camera: Any,
    pc: Any,
    pipe: Any,
    bg_color: Any,
    *args: Any,
    **kwargs: Any,
):
    """Dispatch while preserving the existing renderer positional API."""
    backend = _resolved_backend(pc, pipe)
    if backend == "torch_cpu_reference":
        from .torch_cpu_reference import render as backend_render
    elif backend == "gsplat":
        from .gsplat_backend import render as backend_render
    else:
        raise ValueError(
            f"unsupported renderer_backend {backend!r}; expected "
            "'auto', 'gsplat', or 'torch_cpu_reference'"
        )
    return backend_render(viewpoint_camera, pc, pipe, bg_color, *args, **kwargs)


__all__ = ["render"]
