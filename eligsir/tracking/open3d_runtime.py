"""Shared Open3D runtime helpers for CPU, CUDA and ROCm/HIP tracking frontends.

Engine/device resolution and build provenance only. Nothing here imports the
retired ``SLAMBackend`` registry, so it is reusable by any future
``TrackingProvider`` implementation backed by Open3D.
"""

from __future__ import annotations

import os
from typing import Any


def import_open3d():
    """Import Open3D lazily and preserve non-TTY worker compatibility."""
    import sys as _sys

    if not hasattr(_sys.stdout, "isatty"):
        _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
    if not hasattr(_sys.stderr, "isatty"):
        _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D SLAM frontends require Open3D. "
            "Install it in the container: pip install open3d"
        ) from exc
    return o3d


def tensor_device(o3d: Any, requested: str = "auto"):
    """Resolve an Open3D Tensor device without conflating API spelling and hardware.

    The upstream ROCm port currently reuses Open3D's historical CUDA-facing
    device API, so an AMD HIP build exposes the accelerator as ``CUDA:0``.
    ``physical_accelerator()`` records the actual backend separately.
    """
    if requested != "auto":
        return o3d.core.Device(requested)

    build_config = getattr(o3d, "_build_config", {}) or {}
    if bool(build_config.get("BUILD_CUDA_MODULE", False)):
        try:
            if o3d.core.cuda.is_available():
                return o3d.core.Device("CUDA:0")
        except (AttributeError, RuntimeError):
            pass
    return o3d.core.Device("CPU:0")


def is_accelerator_device(device: Any) -> bool:
    """Return whether *device* names an Open3D GPU accelerator."""
    text = str(device).upper()
    return text.startswith("CUDA:") or text.startswith("SYCL:")


def physical_accelerator(device: Any) -> str:
    """Describe physical accelerator independently of Open3D logical naming.

    EliGSiR ROCm images set ``ELIGSIR_OPEN3D_ACCELERATOR=rocm``. The
    environment fallbacks keep provenance truthful in ad-hoc ROCm containers
    even though upstream Open3D #7509 still calls its HIP device ``CUDA:0``.
    """
    explicit = os.environ.get("ELIGSIR_OPEN3D_ACCELERATOR", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
        return "rocm" if is_accelerator_device(device) else "cpu"
    text = str(device).upper()
    if text.startswith("CUDA:"):
        return "cuda"
    if text.startswith("SYCL:"):
        return "sycl"
    return "cpu"


def open3d_build_identity() -> dict[str, str]:
    """Return image-provided upstream Open3D build provenance when available."""
    return {
        "open3d_upstream_sha": os.environ.get("ELIGSIR_OPEN3D_UPSTREAM_SHA", "unknown"),
        "open3d_upstream_pr": os.environ.get("ELIGSIR_OPEN3D_UPSTREAM_PR", "unknown"),
        "open3d_rocm_arch": os.environ.get("ELIGSIR_OPEN3D_ROCM_ARCH", "unknown"),
    }
