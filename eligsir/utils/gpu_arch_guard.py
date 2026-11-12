"""Detect a mismatch between the ROCm arch this image was built for and the
GPU actually visible at runtime (issue #1392).

The core images are single-architecture: ``Dockerfile``'s
``rocm-python-base-scratch`` stage prunes the hipBLASLt/Tensile library down to
only the one ``PYTORCH_ROCM_ARCH`` it was built for. ``docker-compose.yml``'s
``ELIGSIR_IMAGE_TAG``/``PYTORCH_ROCM_ARCH`` both default to ``gfx1151``, so
a bare ``docker compose run --rm train ...`` on a different-arch host (e.g. the
gfx1201 R9700 box) silently selects the wrong-arch image instead of refusing.
That failure then surfaces deep inside hipBLASLt as an opaque
``rocblaslt error: Cannot read ".../TensileLibrary_lazy_<arch>.dat"`` /
``HIPBLAS_STATUS_INVALID_VALUE`` on the very first matmul -- which reads exactly
like a build defect (a mislabeled/cross-contaminated image) even though it is
not one. Issue #1392 was filed as exactly that misreading: both
``latest-gfx1151`` and ``latest-gfx1201`` were independently verified (two
digests each) to contain only their own architecture's Tensile binaries, zero
cross-contamination. The real failure mode is an operator/host running the
wrong tag for their hardware.

This module turns that opaque, hard-to-diagnose failure into an actionable one
at process startup, before the first real matmul, so the same confusion cannot
silently reoccur as a fresh report of a non-existent image bug.
"""

from __future__ import annotations

import os


class GPUArchMismatchError(RuntimeError):
    """Raised when the visible GPU's ROCm arch differs from this image's baked arch."""


def _base_arch(arch: str) -> str:
    # torch's gcnArchName can carry a feature suffix, e.g. "gfx1151:xnack+".
    return arch.split(":")[0].strip().lower()


def check_rocm_arch_match() -> None:
    """Raise ``GPUArchMismatchError`` if the visible GPU's arch != this image's
    baked ``PYTORCH_ROCM_ARCH``.

    Deliberately a narrow, best-effort guard against one specific,
    already-observed operator mistake -- not a general hardware-compatibility
    gate. It no-ops (never raises) whenever there is nothing conclusive to
    compare: no baked arch recorded in the environment, torch not importable,
    no CUDA/HIP device visible (CPU-only hosts, CPU CI images), or the
    device-properties query itself fails for any reason. A run that would
    otherwise work must never be turned into a hard failure by this check.
    """
    baked_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    if not baked_arch:
        return

    try:
        import torch
    except Exception:
        return

    try:
        if not torch.cuda.is_available():
            return
        detected_arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return

    baked_base = _base_arch(baked_arch)
    detected_base = _base_arch(detected_arch)
    if baked_base == detected_base:
        return

    raise GPUArchMismatchError(
        "GPU architecture mismatch: this image/environment was built for "
        f"PYTORCH_ROCM_ARCH='{baked_base}', but the visible GPU reports "
        f"'{detected_base}' (torch.cuda.get_device_properties(0).gcnArchName="
        f"'{detected_arch}'). This image's hipBLASLt/Tensile library only ships "
        f"'{baked_base}' binaries (per-arch pruning, see Dockerfile's "
        "rocm-python-base-scratch stage), so any matmul will fail deep inside "
        f"hipBLASLt with an opaque \"Cannot read ... TensileLibrary_lazy_"
        f"{detected_base}.dat\" error that looks like a mislabeled/cross-"
        "contaminated image but is not (see issue #1392). Pull/build the "
        f"matching tag instead: ELIGSIR_IMAGE_TAG=latest-{detected_base} "
        f"PYTORCH_ROCM_ARCH={detected_base} docker compose run --rm train ..."
    )
