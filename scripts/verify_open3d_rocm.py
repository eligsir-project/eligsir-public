#!/usr/bin/env python3
"""Verify that the pinned Open3D build executes Tensor ICP on an AMD GPU.

Run this inside ``docker/open3d_rocm.Dockerfile``'s final image with /dev/kfd
and /dev/dri exposed.  Import success alone is not sufficient: this script
allocates tensors on the Open3D accelerator and executes Tensor ICP.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


def _fail(message: str) -> None:
    print(f"open3d-rocm verification failed: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    import open3d as o3d

    expected_backend = os.environ.get("ELIGSIR_OPEN3D_ACCELERATOR", "")
    expected_sha = os.environ.get("ELIGSIR_OPEN3D_UPSTREAM_SHA", "")
    expected_arch = os.environ.get("ELIGSIR_OPEN3D_ROCM_ARCH", "")
    if expected_backend != "rocm":
        _fail(
            "ELIGSIR_OPEN3D_ACCELERATOR must be 'rocm'; "
            f"got {expected_backend!r}"
        )
    if not expected_sha:
        _fail("missing ELIGSIR_OPEN3D_UPSTREAM_SHA provenance")
    if not expected_arch:
        _fail("missing ELIGSIR_OPEN3D_ROCM_ARCH provenance")

    build_config = dict(getattr(o3d, "_build_config", {}) or {})
    if not build_config.get("BUILD_CUDA_MODULE", False):
        _fail("Open3D GPU module is not compiled")
    try:
        available = bool(o3d.core.cuda.is_available())
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        _fail(f"Open3D accelerator availability check raised: {exc!r}")
    if not available:
        _fail("Open3D reports no accelerator; check /dev/kfd, ROCm runtime, and arch")

    # Upstream #7509 preserves the logical CUDA device/API spelling while the
    # underlying translation units execute through HIP.  Keep both facts in the
    # verification record instead of renaming the logical device locally.
    device = o3d.core.Device("CUDA:0")
    probe = o3d.core.Tensor.ones((1024,), dtype=o3d.core.float32, device=device)
    probe_sum = float(probe.sum().cpu().item())
    if probe_sum != 1024.0:
        _fail(f"unexpected accelerator tensor result: {probe_sum}")

    rng = np.random.default_rng(7)
    target_points = rng.uniform(-0.5, 0.5, size=(4096, 3)).astype(np.float32)
    offset = np.array([0.06, -0.025, 0.04], dtype=np.float32)
    source_points = target_points - offset

    source = o3d.t.geometry.PointCloud(device)
    target = o3d.t.geometry.PointCloud(device)
    source.point.positions = o3d.core.Tensor(
        source_points, dtype=o3d.core.float32, device=device
    )
    target.point.positions = o3d.core.Tensor(
        target_points, dtype=o3d.core.float32, device=device
    )

    result = o3d.t.pipelines.registration.icp(
        source,
        target,
        0.25,
        o3d.core.Tensor.eye(4, o3d.core.float64, device),
        o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    o3d.core.cuda.synchronize()

    transform = np.asarray(result.transformation.cpu().numpy(), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        _fail(f"invalid Tensor ICP transform: {transform!r}")
    translation_error = float(np.linalg.norm(transform[:3, 3] - offset.astype(np.float64)))
    if float(result.fitness) < 0.99:
        _fail(f"Tensor ICP fitness too low: {float(result.fitness):.6f}")
    if translation_error > 0.015:
        _fail(f"Tensor ICP translation error too high: {translation_error:.6f} m")

    record = {
        "status": "PASS",
        "open3d_version": getattr(o3d, "__version__", "unknown"),
        "open3d_upstream_sha": expected_sha,
        "open3d_upstream_pr": os.environ.get("ELIGSIR_OPEN3D_UPSTREAM_PR", "unknown"),
        "rocm_arch": expected_arch,
        "physical_accelerator": "rocm",
        "open3d_logical_device": str(device),
        "build_cuda_module_compat": bool(build_config.get("BUILD_CUDA_MODULE", False)),
        "tensor_probe_sum": probe_sum,
        "icp_fitness": float(result.fitness),
        "icp_inlier_rmse": float(result.inlier_rmse),
        "icp_translation_error_m": translation_error,
        "icp_transform": transform.tolist(),
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
