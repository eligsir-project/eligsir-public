#!/usr/bin/env python3
"""Minimal real-GPU smoke for a published EliGSiR ROCm image."""
from __future__ import annotations

import json

import torch
import gsplat
import fused_ssim  # noqa: F401
from simple_knn._C import distCUDA2

assert torch.cuda.is_available(), "ROCm GPU is not visible to PyTorch"
assert torch.version.hip, "expected ROCm/HIP PyTorch"

# Exercise a core ROCm/Torch kernel.
device = torch.device("cuda:0")
a = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
b = a.T.contiguous()
c = a @ b
assert c.is_cuda and torch.isfinite(c).all()

# Exercise EliGSiR's exact native KNN extension used for Gaussian scale init.
pts = torch.tensor(
    [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
    dtype=torch.float32,
    device=device,
)
d2 = distCUDA2(pts)
assert d2.is_cuda and torch.isfinite(d2).all()

print(
    json.dumps(
        {
            "backend": "rocm",
            "hip": torch.version.hip,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "gsplat": getattr(gsplat, "__version__", "unknown"),
            "capabilities": [
                "torch_rocm_kernel",
                "simple_knn_gpu",
                "gsplat_import",
                "fused_ssim_import",
            ],
        },
        sort_keys=True,
    )
)
