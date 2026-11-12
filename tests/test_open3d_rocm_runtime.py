"""Software-only tests for Open3D ROCm/Tensor runtime semantics."""

from __future__ import annotations

import numpy as np

from eligsir.contracts.types import CameraIntrinsics
from eligsir.tracking.open3d_runtime import (
    is_accelerator_device,
    physical_accelerator,
    tensor_device,
)
from eligsir.tracking.open3d_icp import Open3DICPTracker, _depth_to_xyz


class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCore:
    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)

    @staticmethod
    def Device(name: str) -> _FakeDevice:
        return _FakeDevice(name)


class _FakeOpen3D:
    def __init__(self, *, compiled: bool, available: bool) -> None:
        self._build_config = {"BUILD_CUDA_MODULE": compiled}
        self.core = _FakeCore(available)


def test_auto_tensor_device_uses_accelerator_when_gpu_module_is_available() -> None:
    o3d = _FakeOpen3D(compiled=True, available=True)
    device = tensor_device(o3d, "auto")
    assert str(device) == "CUDA:0"
    assert is_accelerator_device(device)


def test_auto_tensor_device_falls_back_to_cpu_without_gpu() -> None:
    o3d = _FakeOpen3D(compiled=True, available=False)
    device = tensor_device(o3d, "auto")
    assert str(device) == "CPU:0"
    assert not is_accelerator_device(device)


def test_rocm_physical_backend_is_separate_from_open3d_cuda_label(monkeypatch) -> None:
    monkeypatch.setenv("ELIGSIR_OPEN3D_ACCELERATOR", "rocm")
    assert physical_accelerator(_FakeDevice("CUDA:0")) == "rocm"


def test_icp_auto_engine_selects_tensor_on_rocm_compat_device(monkeypatch) -> None:
    monkeypatch.setenv("ELIGSIR_OPEN3D_ACCELERATOR", "rocm")
    tracker = Open3DICPTracker(engine="auto", device="auto")
    engine, device = tracker._resolve_runtime(_FakeOpen3D(compiled=True, available=True))
    assert engine == "tensor"
    assert str(device) == "CUDA:0"
    assert tracker._logical_device == "CUDA:0"
    assert tracker._physical_accelerator == "rocm"


def test_icp_auto_engine_uses_legacy_on_cpu() -> None:
    tracker = Open3DICPTracker(engine="auto", device="auto")
    engine, device = tracker._resolve_runtime(_FakeOpen3D(compiled=False, available=False))
    assert engine == "legacy"
    assert str(device) == "CPU:0"
    assert tracker._logical_device == "CPU:0"
    assert tracker._physical_accelerator == "cpu"


def test_legacy_engine_rejects_accelerator_device() -> None:
    try:
        Open3DICPTracker(engine="legacy", device="CUDA:0")
    except ValueError as exc:
        assert "CPU-only" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("legacy engine accepted a GPU device")


def test_depth_to_xyz_uses_canonical_opencv_camera_coordinates() -> None:
    intrinsics = CameraIntrinsics(
        fx=2.0,
        fy=2.0,
        cx=0.0,
        cy=0.0,
        width=2,
        height=2,
    )
    depth = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
    points = _depth_to_xyz(depth, intrinsics, 0.1, 8.0)
    np.testing.assert_allclose(
        points,
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32),
    )
