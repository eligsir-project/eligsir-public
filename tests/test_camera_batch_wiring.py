"""Numerical-equivalence tests for `prepare_camera_batch_for_render` (issue #16b
camera-batching wiring audit).

`scene/cameras.py` has long had a real, working batched camera-tensor builder
(`make_gsplat_camera_batch`), but until this change no production call site
used it -- only `tests/test_owner_transfer_sequence.py` exercised it. This
file verifies the new production-facing helper, `prepare_camera_batch_for_render`,
produces bit-identical `cam.gsplat` tensors to calling `prepare_camera_for_render`
on each camera individually (the pre-existing per-camera path still used
everywhere else), and that it is a true no-op for already-warm cameras (the
steady-state case at the one production call site this was wired into,
`train_streaming.py`'s B>1 `--streaming_optimization_batch_size` engine path).

CPU-only by design: `render()` (`eligsir/mapping/renderer/gsplat_backend.py`)
consumes `cam.gsplat.viewmat`/`.K`/`.camera_center` directly and is itself
single-camera-only (`viewmats = viewmat[None]`), so bit-identical camera
tensors are sufficient to guarantee bit-identical rasterizer input -- there is
nothing further a GPU comparison would additionally verify.
"""
from __future__ import annotations

import copy

import torch

from scene.cameras import (
    GsplatCameraTensors,
    prepare_camera_batch_for_render,
    prepare_camera_for_render,
)


class _Cam:
    """Same fake-camera shape as tests/test_owner_transfer_sequence.py's _Cam,
    extended with an `original_image` so image-residency behavior is also
    covered."""

    def __init__(self, tx, fx=4.0, fy=5.0, cx=3.5, cy=2.5, fovx=1.0, fovy=1.0):
        self.data_device = torch.device("cpu")
        self.world_view_transform = torch.eye(4)
        self.world_view_transform[3, 0] = tx
        self.camera_center = torch.tensor([tx, 0.0, 0.0])
        self.image_width = 8
        self.image_height = 6
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.original_image = torch.rand(3, 6, 8)


def _assert_tensors_equal(a: GsplatCameraTensors, b: GsplatCameraTensors) -> None:
    assert torch.equal(a.viewmat, b.viewmat)
    assert torch.equal(a.K, b.K)
    assert torch.equal(a.camera_center, b.camera_center)
    assert a.width == b.width
    assert a.height == b.height
    assert a.viewmat.shape == b.viewmat.shape
    assert a.K.shape == b.K.shape
    assert a.camera_center.shape == b.camera_center.shape
    assert a.viewmat.dtype == b.viewmat.dtype == torch.float32
    assert a.viewmat.device == b.viewmat.device
    assert a.viewmat.is_contiguous()
    assert a.K.is_contiguous()
    assert a.camera_center.is_contiguous()


def test_batch_helper_matches_per_camera_path_all_cold():
    """All-cold batch: the batched builder must produce bit-identical tensors
    to the pre-existing per-camera prepare_camera_for_render loop."""
    cams_individual = [_Cam(0.0), _Cam(1.0), _Cam(2.0)]
    cams_batched = copy.deepcopy(cams_individual)

    for c in cams_individual:
        prepare_camera_for_render(c, device="cpu")

    prepare_camera_batch_for_render(cams_batched, device="cpu")

    for ci, cb in zip(cams_individual, cams_batched):
        _assert_tensors_equal(ci.gsplat, cb.gsplat)
        assert torch.equal(ci.original_image, cb.original_image)


def test_batch_helper_matches_per_camera_path_fov_derived_intrinsics():
    """Cameras with no explicit fx/fy/cx/cy (FoV-derived intrinsics branch of
    _camera_intrinsics) must also match bit-exactly between the two paths."""
    cams_individual = [
        _Cam(0.0, fx=None, fy=None, cx=None, cy=None, fovx=0.9, fovy=0.7),
        _Cam(1.5, fx=None, fy=None, cx=None, cy=None, fovx=1.1, fovy=0.8),
    ]
    cams_batched = copy.deepcopy(cams_individual)

    for c in cams_individual:
        prepare_camera_for_render(c, device="cpu")
    prepare_camera_batch_for_render(cams_batched, device="cpu")

    for ci, cb in zip(cams_individual, cams_batched):
        _assert_tensors_equal(ci.gsplat, cb.gsplat)


def test_batch_helper_matches_per_camera_path_negative_translation():
    """A second, independent all-cold cross-check (distinct camera values,
    including a negative translation component) against the pre-existing
    per-camera prepare_camera_for_render path."""
    cams_individual = [_Cam(0.0), _Cam(-2.5)]
    cams_batched = copy.deepcopy(cams_individual)

    for c in cams_individual:
        prepare_camera_for_render(c, device="cpu")
    prepare_camera_batch_for_render(cams_batched, device="cpu")

    for ci, cb in zip(cams_individual, cams_batched):
        _assert_tensors_equal(ci.gsplat, cb.gsplat)


def test_batch_helper_is_noop_for_already_warm_cameras():
    """Steady-state case at the wired call site: every camera in the batch
    already has a fresh cam.gsplat (built individually at collection time by
    the existing per-camera prepare_camera_for_render calls). The batch
    helper must not rebuild or replace it."""
    cams = [_Cam(0.0), _Cam(1.0), _Cam(2.0)]
    for c in cams:
        prepare_camera_for_render(c, device="cpu")
    cached = [c.gsplat for c in cams]

    prepare_camera_batch_for_render(cams, device="cpu")

    for c, ref in zip(cams, cached):
        assert c.gsplat is ref, "warm camera's gsplat was rebuilt (not a no-op)"


def test_batch_helper_mixed_warm_and_cold():
    """Some cameras already warm, some cold (e.g. the primary view was
    already prepared earlier while an extra sampled view is brand new) --
    only the cold ones should be rebuilt, and both classes must end up
    correct."""
    cams = [_Cam(0.0), _Cam(1.0), _Cam(2.0)]
    prepare_camera_for_render(cams[0], device="cpu")
    prepare_camera_for_render(cams[2], device="cpu")
    warm_refs = {0: cams[0].gsplat, 2: cams[2].gsplat}
    assert getattr(cams[1], "gsplat", None) is None

    expected_cold_cam = copy.deepcopy(cams[1])
    prepare_camera_for_render(expected_cold_cam, device="cpu")

    prepare_camera_batch_for_render(cams, device="cpu")

    assert cams[0].gsplat is warm_refs[0]
    assert cams[2].gsplat is warm_refs[2]
    _assert_tensors_equal(expected_cold_cam.gsplat, cams[1].gsplat)


def test_batch_helper_moves_images_to_target_device():
    cams = [_Cam(0.0), _Cam(1.0)]
    assert all(c.original_image.device.type == "cpu" for c in cams)
    prepare_camera_batch_for_render(cams, device="cpu")
    assert all(c.original_image.device.type == "cpu" for c in cams)


def test_batch_helper_empty_list_is_noop():
    prepare_camera_batch_for_render([], device="cpu")
