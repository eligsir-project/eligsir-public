"""Priority-1 GPU-native depth backend (issue #1348).

docs/architecture/2026-08-31-gpu-native-decode-and-compressed-ingress.md's
priority 1: raw depth transferred in its native on-wire dtype, with the
depth_scale/metric conversion executed on-device rather than via NumPy on the
CPU. ``torch_gpu_depth_backend`` is a drop-in ``NumpyBackend`` replacement --
these tests assert it produces bit-identical output to the existing CPU path
for every case ``NumpyBackend`` already covers, and delegates RGB/unspecified-
kind conversions unchanged (RGB/JPEG decode is priority 2, out of scope here).

No CUDA is required: these tests exercise the backend with ``device="cpu"``,
which runs the exact same torch code path (pin/H2D/convert/D2H all become
no-ops or trivial copies on a CPU-only host) and is sufficient to prove
numerical equivalence -- the only thing this backend changes relative to
NumpyBackend is *where* the divide executes, never its numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.contracts import (
    CameraIntrinsics,
    DepthPolicy,
    MaterializationPolicy,
    SensorPacket,
    SourceIdentity,
    materialize_mapping_frame,
    numpy_backend,
    torch_gpu_depth_backend,
)
from eligsir.contracts.runtime import BufferRef


def _intrinsics(depth_scale: float = 5000.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=64, height=48, depth_scale=depth_scale
    )


def _packet(raw_depth: np.ndarray, rgb: np.ndarray | None = None) -> SensorPacket:
    if rgb is None:
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    return SensorPacket(
        SourceIdentity("camera", 0, 1, 1_699_999_999_123_456_789),
        BufferRef.from_numpy(rgb, encoding="rgb8"),
        BufferRef.from_numpy(raw_depth, encoding="16UC1", units="raw_counts"),
        _intrinsics(),
    )


def _raw_depth_with_invalid_speckle(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 65535, size=(48, 64), dtype=np.uint16)
    raw[::7, ::11] = 0  # invalid-depth speckle, matches sensor convention
    return raw


def test_gpu_backend_to_float32_bit_identical_to_numpy_for_depth():
    raw = _raw_depth_with_invalid_speckle()
    cpu_out = numpy_backend().to_float32(raw, 5000.0, kind="depth")
    gpu_out = torch_gpu_depth_backend(device="cpu").to_float32(raw, 5000.0, kind="depth")
    assert cpu_out.dtype == gpu_out.dtype == np.float32
    assert np.array_equal(cpu_out, gpu_out)


def test_gpu_backend_delegates_color_kind_unchanged():
    rgb = np.random.default_rng(2).integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    cpu_out = numpy_backend().to_float32(rgb, 255.0, kind="color")
    gpu_out = torch_gpu_depth_backend(device="cpu").to_float32(rgb, 255.0, kind="color")
    assert np.array_equal(cpu_out, gpu_out)


def test_gpu_backend_delegates_unspecified_kind_unchanged():
    raw = _raw_depth_with_invalid_speckle(seed=3)
    cpu_out = numpy_backend().to_float32(raw, 5000.0)
    gpu_out = torch_gpu_depth_backend(device="cpu").to_float32(raw, 5000.0)
    assert np.array_equal(cpu_out, gpu_out)


def test_gpu_backend_preserves_validity_convention():
    """depth==0 (invalid) must map to exactly 0.0, matching the CPU path's
    isfinite(depth) & depth>0 validity predicate used everywhere downstream."""
    raw = np.array([[0, 5000, 65535]], dtype=np.uint16)
    out = torch_gpu_depth_backend(device="cpu").to_float32(raw, 5000.0, kind="depth")
    assert out[0, 0] == 0.0
    assert out[0, 1] == pytest.approx(1.0)
    np.testing.assert_allclose(out[0, 2], 65535.0 / 5000.0, rtol=0, atol=1e-6)


def test_end_to_end_materialize_mapping_frame_matches_cpu_backend():
    raw = _raw_depth_with_invalid_speckle(seed=7)
    rgb = np.random.default_rng(7).integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    packet = _packet(raw, rgb)
    policy = MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32)

    frame_cpu = materialize_mapping_frame(packet, policy, backend=numpy_backend())
    frame_gpu = materialize_mapping_frame(
        packet, policy, backend=torch_gpu_depth_backend(device="cpu")
    )

    assert frame_gpu.device == "cpu", "GPU-assisted backend must keep MappingFrame CPU-owned"
    assert frame_cpu.depth.spec.units == frame_gpu.depth.spec.units == "meters"
    assert np.array_equal(frame_cpu.depth.data, frame_gpu.depth.data)
    assert np.array_equal(frame_cpu.rgb.data, frame_gpu.rgb.data)


def test_runtime_binding_selects_gpu_backend_only_when_flag_set(monkeypatch):
    from eligsir.runtime.runtime_binding import runtime_binding_from_runtime

    class _FakeRuntime:
        map_frame = staticmethod(lambda *a, **k: None)
        tick = staticmethod(lambda *a, **k: None)
        mark_source_complete = staticmethod(lambda *a, **k: None)
        flush = staticmethod(lambda *a, **k: None)
        close = staticmethod(lambda *a, **k: None)
        observation_owner = None
        control_snapshot = None
        record_disposition = None

    class _Args:
        streaming_gpu_native_depth = False

    runtime = _FakeRuntime()

    binding_off = runtime_binding_from_runtime(runtime, args=_Args())
    # The binding is also the read-only description of production packet
    # materialization for callers that build an out-of-band holdout camera.
    # The normal production session has always resolved a missing backend to
    # numpy_backend(); hiding that effective backend here lets that caller
    # silently diverge from the mapper.  CPU materialization must therefore
    # be explicit, not represented by ``None``.
    assert isinstance(binding_off.materialization_backend, type(numpy_backend()))

    args_on = _Args()
    args_on.streaming_gpu_native_depth = True
    binding_on = runtime_binding_from_runtime(runtime, args=args_on)
    assert binding_on.materialization_backend is not None

    binding_default = runtime_binding_from_runtime(runtime)
    assert isinstance(binding_default.materialization_backend, type(numpy_backend()))
