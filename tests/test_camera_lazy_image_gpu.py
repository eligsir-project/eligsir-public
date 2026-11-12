"""GPU-gated lazy-camera ownership tests.

Lazy camera geometry is placed on the configured accelerator immediately, but
pixel decode/materialization remains CPU-owned. ``_CameraBatchPrefetcher`` is the
single first-use H2D owner and binds the uploaded image only after its transfer
event has been handed to the compute stream.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from scene.cameras import Camera
from eligsir.utils.camera_utils import _decode_image_tensor
from eligsir.runtime.transfers import (
    _CameraBatchPrefetcher,
    DeviceTransferStream,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _make_eager_and_lazy(image: torch.Tensor, data_device: str = "cuda"):
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    kwargs = dict(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=0.9,
        FoVy=0.7,
        image_name="f",
        uid=0,
        fx=50.0,
        fy=50.0,
        cx=32.0,
        cy=24.0,
        data_device=data_device,
    )
    eager = Camera(image=image, gt_alpha_mask=None, **kwargs)

    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return image, None

    lazy = Camera(
        image=None,
        gt_alpha_mask=None,
        width=image.shape[2],
        height=image.shape[1],
        pending_image_loader=loader,
        **kwargs,
    )
    return eager, lazy, calls


def test_lazy_camera_has_no_original_image_before_materialize():
    image = torch.rand(3, 16, 16)
    _eager, lazy, calls = _make_eager_and_lazy(image)
    assert lazy.has_pending_image is True
    assert not hasattr(lazy, "original_image")
    assert calls["n"] == 0


def test_lazy_camera_geometry_matches_eager_before_materialize():
    image = torch.rand(3, 16, 16)
    eager, lazy, _ = _make_eager_and_lazy(image)
    assert torch.allclose(eager.world_view_transform, lazy.world_view_transform)
    assert torch.allclose(eager.projection_matrix, lazy.projection_matrix)
    assert torch.allclose(eager.camera_center, lazy.camera_center)
    assert lazy.image_width == eager.image_width
    assert lazy.image_height == eager.image_height


def test_lazy_materialization_is_idempotent_and_batch_prefetch_owns_one_h2d():
    """What holds TODAY: one decode, and _CameraBatchPrefetcher issues exactly
    one grouped H2D for a CPU-resident camera.

    ``materialize_pending_image`` leaves pixels on CPU so the prefetcher is the
    sole uploader.
    """
    image = torch.rand(3, 16, 16)
    eager, lazy, calls = _make_eager_and_lazy(image)

    lazy.materialize_pending_image()
    assert calls["n"] == 1
    assert lazy.has_pending_image is False

    lazy.materialize_pending_image()
    assert calls["n"] == 1

    # A CPU-resident lazy camera is what the prefetcher is built to take, so
    # build one explicitly rather than assuming materialization produced it.
    _eager_cpu, lazy_cpu, _ = _make_eager_and_lazy(image, data_device="cpu")
    lazy_cpu.materialize_pending_image()
    assert lazy_cpu.original_image.device.type == "cpu"

    transfer = DeviceTransferStream(torch.device("cuda"))
    prefetcher = _CameraBatchPrefetcher(transfer)
    assert prefetcher.prefetch_and_bind([lazy_cpu]) == 1
    assert transfer.stats.h2d_ops == 1
    assert lazy_cpu.original_image.device.type == "cuda"
    assert eager.original_image.device.type == "cpu"
    assert torch.allclose(
        eager.original_image.to(lazy_cpu.original_image.device),
        lazy_cpu.original_image,
    )


def test_lazy_materialization_leaves_pixels_cpu_owned():
    image = torch.rand(3, 16, 16)
    _eager, lazy, _calls = _make_eager_and_lazy(image)
    lazy.materialize_pending_image()
    assert lazy.original_image.device.type == "cpu"


def test_production_uint8_decoder_keeps_pixels_cpu_owned_for_cuda_geometry():
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    args = SimpleNamespace(data_device="cuda")
    cam_info = SimpleNamespace(image=None)

    image, mask = _decode_image_tensor(args, cam_info, 12, 8, (12, 8), rgb)

    assert image.device.type == "cpu"
    assert mask is None
