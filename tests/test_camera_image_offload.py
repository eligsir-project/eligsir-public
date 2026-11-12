"""Tests for offload_camera_image / prepare_camera_for_render CPU<->GPU paging.

Regression coverage for the ScanNet++ GTT-exhaustion bug (2026-07-07): streaming
replay kept every admitted keyframe's full-resolution image tensor GPU-resident
forever (train_cameras/replay_buffer/global_reservoir never evicted Camera
objects, and no code ever moved `original_image` off the GPU). Memory scaled as
(admitted-frame-count x width x height), independent of --cap_max -- fine for
TUM's small low-res trajectories, catastrophic for ScanNet++'s high-resolution
iPhone-LiDAR captures over thousands of frames.
"""
import pytest
import torch

from scene.cameras import offload_camera_image, prepare_camera_for_render


class _FakeCam:
    """Minimal stand-in exposing only what offload/prepare touch."""

    def __init__(self, image: torch.Tensor):
        self.original_image = image


def test_offload_camera_image_moves_to_cpu():
    cam = _FakeCam(torch.rand(3, 8, 8))
    assert cam.original_image.device.type == "cpu"
    moved = offload_camera_image(cam)
    assert moved is True
    assert cam.original_image.device.type == "cpu"


def test_offload_camera_image_no_image_returns_false():
    class _NoImage:
        pass

    assert offload_camera_image(_NoImage()) is False


def test_offload_camera_image_is_idempotent():
    cam = _FakeCam(torch.rand(3, 4, 4))
    offload_camera_image(cam)
    ref = cam.original_image
    offload_camera_image(cam)
    # Already on CPU: second call should be a no-op (same tensor identity),
    # not a needless clone/copy on every call.
    assert cam.original_image is ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_offload_then_prepare_round_trips_through_gpu():
    from scene.cameras import GsplatCameraTensors

    cam = _FakeCam(torch.rand(3, 8, 8, device="cuda"))
    # Pre-populate `gsplat` (already on the target device) so
    # prepare_camera_for_render only exercises the image-paging branch under test.
    cam.gsplat = GsplatCameraTensors(
        viewmat=torch.eye(4, device="cuda"),
        K=torch.eye(3, device="cuda"),
        camera_center=torch.zeros(1, 3, device="cuda"),
        width=8,
        height=8,
    )

    offload_camera_image(cam)
    assert cam.original_image.device.type == "cpu"

    prepare_camera_for_render(cam, device="cuda")
    assert cam.original_image.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_prepare_camera_skips_rebuild_when_gsplat_already_cached():
    """Regression: torch.device("cuda") != torch.device("cuda:0") is True, so
    calling prepare_camera_for_render(cam, device="cuda") (bare string, no
    index) against a gsplat cache built on an indexed cuda device previously
    NEVER matched -- the cache-skip check silently rebuilt gsplat tensors on
    every single call, defeating the caching this function exists to
    provide. Assert the cached GsplatCameraTensors object identity survives
    a second prepare_camera_for_render call (i.e., no rebuild happened)."""
    from scene.cameras import GsplatCameraTensors

    cam = _FakeCam(torch.rand(3, 8, 8, device="cuda"))
    cached = GsplatCameraTensors(
        viewmat=torch.eye(4, device="cuda"),
        K=torch.eye(3, device="cuda"),
        camera_center=torch.zeros(1, 3, device="cuda"),
        width=8,
        height=8,
    )
    cam.gsplat = cached

    prepare_camera_for_render(cam, device="cuda")
    assert cam.gsplat is cached, "gsplat was rebuilt despite already being cached on-device"
