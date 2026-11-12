"""Regression guard for the `holdout_gt/<id>_depth.png` ENCODING contract (#358).

The defect class this pins is *silent*: nothing crashes, the geometry is just
wrong. Issue #358 published Level A and Level B fr3 figures and then had to
**retract** them because `scripts/local_quality_offline_demo.py` divided this
file by the dataset's TUM `depth_scale` (5000) instead of 1000. Every
backprojected point sat 5x too close to its camera, so the microcell lattice
measured a 5x-shrunk scene: 65 observed cells reported instead of 617.

The invariant is NOT "the dataset depth_scale is irrelevant" -- it is very
relevant, and `utils.comparison_report._get_sensor_depth` correctly consumes it
(pinned separately by `tests/test_holdout_sensor_depth_producer.py`). The
invariant is one step later::

    holdout_gt/<id>_depth.png == round(clip(metric_depth_m * 1000, 0, 65535))

i.e. millimetres UNCONDITIONALLY, because by the time a value reaches this file
the source scale has already been divided out.

So the discriminating test here (`test_png_is_millimetres_whatever_the_dataset_
scale_was`) feeds two cameras whose RAW depth arrays differ and whose
`_streaming_depth_scale` differs (5000 vs 1000) but which mean the SAME metric
depth, and asserts the two PNGs come out byte-identical. Feeding the same raw
array at two scales and asserting the same PNG would assert something false.
"""
import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2", reason="holdout_gt PNGs are written via cv2")

from eligsir.io.holdout_gt_depth import (  # noqa: E402
    HOLDOUT_GT_DEPTH_MAX_MM,
    HOLDOUT_GT_DEPTH_SCALE,
    decode_holdout_depth_m,
    encode_holdout_depth_mm,
)

def _import_checkpoint_io():
    """Import the REAL producer module.

    `checkpoint_io` -> `frame_preparation` -> `torchvision.utils.save_image`, and
    torchvision's compiled ops are absent on the bare host (the repo's known ~38
    host collection errors), which would otherwise force this whole suite to skip
    on CPU -- i.e. exactly where it is meant to run. torchvision is irrelevant to
    the encoding contract, so stub it ONLY if the real import fails, never
    preemptively; in-container the real one is used.

    The stub is torn down again as soon as the import returns. pytest shares
    `sys.modules` for the whole session, so a stub left installed is inherited
    by every later-collected test -- verified concretely: leaving it in made
    `tests/test_issue162_depth_color_resolution_mismatch.py` collect past its
    own documented torchvision failure and into a different one. Changing an
    unrelated test's outcome is not an acceptable price for this suite running
    on a broken host.
    """
    try:
        import eligsir.mapping.integration.checkpoint_io as _m
        return _m
    except Exception:
        pass

    import sys as _sys
    import types as _types
    before = set(_sys.modules)
    installed = []
    if "torchvision" not in _sys.modules:
        _tv = _types.ModuleType("torchvision")
        _tv_utils = _types.ModuleType("torchvision.utils")
        _tv_utils.save_image = lambda *a, **k: None
        _tv.utils = _tv_utils
        _sys.modules["torchvision"] = _tv
        _sys.modules["torchvision.utils"] = _tv_utils
        installed = ["torchvision", "torchvision.utils"]
    try:
        import eligsir.mapping.integration.checkpoint_io as _m
        return _m
    finally:
        # Remove EVERY module this stubbed import cached, not just the stub
        # itself: `eligsir.mapping.integration.*` and friends would
        # otherwise stay in sys.modules holding the fake `save_image`, and a
        # later test importing them would silently inherit it. Measured: with
        # only the stub popped, an unrelated host test flipped fail->pass. The
        # returned module object keeps working -- its own imports are already
        # bound -- while other tests re-import these modules cleanly.
        for name in set(_sys.modules) - before:
            _sys.modules.pop(name, None)
        for name in installed:
            _sys.modules.pop(name, None)


checkpoint_io = _import_checkpoint_io()  # noqa: E402


# --------------------------------------------------------------------------
# fixtures / doubles
# --------------------------------------------------------------------------

H, W = 4, 6


class _Cam:
    """Minimal holdout camera. Depth is supplied by the patched producer."""

    def __init__(self, raw_depth: np.ndarray, depth_scale: float):
        self.image_height, self.image_width = H, W
        self._raw_depth = raw_depth
        self._streaming_depth_scale = depth_scale


def _fake_get_sensor_depth(cam, h, w):
    """Stand-in for `utils.comparison_report._get_sensor_depth`.

    Reproduces the ONE property of the real function this contract depends on:
    it returns METRES, having already divided the dataset's own depth_scale out.
    """
    metres = cam._raw_depth.astype(np.float32) / float(cam._streaming_depth_scale)
    return torch.from_numpy(metres).reshape(1, h, w)


@pytest.fixture(autouse=True)
def _clear_saved_ids():
    """`_HOLDOUT_GT_SAVED_IDS` is module-level mutable state -- a second call
    with the same camera indices returns early and writes NOTHING, which would
    let every assertion below pass vacuously."""
    checkpoint_io._HOLDOUT_GT_SAVED_IDS.clear()
    yield
    checkpoint_io._HOLDOUT_GT_SAVED_IDS.clear()


@pytest.fixture
def patched_producer(monkeypatch):
    """`_get_sensor_depth` is imported INSIDE `_do()` from eligsir.utils.comparison_report,
    so that is where it must be patched."""
    import eligsir.utils.comparison_report as cr
    monkeypatch.setattr(cr, "_get_sensor_depth", _fake_get_sensor_depth, raising=True)
    # _gt_image is irrelevant here; make it cheap and non-throwing.
    monkeypatch.setattr(
        cr, "_gt_image", lambda cam: torch.zeros(3, H, W), raising=True
    )
    return None


def _save(tmp_path, cams):
    """Run the real producer synchronously (save_worker=None runs _do() inline)."""
    checkpoint_io._save_holdout_gt(None, str(tmp_path), cams)


def _read_png(tmp_path, idx):
    p = tmp_path / "holdout_gt" / f"{idx}_depth.png"
    assert p.exists(), (
        f"{p} was not written -- the producer swallows every exception into a "
        f"print, so a missing file is the failure mode this suite guards"
    )
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


# --------------------------------------------------------------------------
# the contract itself
# --------------------------------------------------------------------------

def test_encode_decode_round_trip_is_millimetres():
    depth_m = np.array([[0.0, 0.5, 1.0], [2.4, 3.75, 8.0]], dtype=np.float32)
    mm = encode_holdout_depth_mm(depth_m)
    assert mm.dtype == np.uint16
    np.testing.assert_array_equal(mm, np.array([[0, 500, 1000], [2400, 3750, 8000]],
                                               dtype=np.uint16))
    np.testing.assert_allclose(decode_holdout_depth_m(mm), depth_m, atol=1e-6)


def test_encode_clips_rather_than_wraps():
    """A wrapped uint16 would decode to a plausible near-field depth -- the
    worst possible failure for a metric that backprojects it into a lattice."""
    mm = encode_holdout_depth_mm(np.array([70.0, -1.0], dtype=np.float32))
    assert mm[0] == HOLDOUT_GT_DEPTH_MAX_MM
    assert mm[1] == 0


def test_scale_constant_is_1000_not_a_dataset_scale():
    """Pinned as a value, not a grep. TUM's 5000 is the divisor that caused the
    #358 retraction; ScanNet's 1000 coincides and must not be read as licence
    to use the dataset scale here."""
    assert HOLDOUT_GT_DEPTH_SCALE == 1000.0


def test_png_is_millimetres_whatever_the_dataset_scale_was(tmp_path, patched_producer):
    """THE regression guard for the #358 retraction.

    Two cameras, DIFFERENT raw arrays and DIFFERENT dataset depth_scale, but
    identical metric depth -> byte-identical PNGs, both decoding to metres.
    """
    metres = (np.arange(H * W, dtype=np.float32).reshape(H, W) * 0.05 + 1.0)
    tum = _Cam((metres * 5000.0).astype(np.uint16), depth_scale=5000.0)   # TUM
    scannet = _Cam((metres * 1000.0).astype(np.uint16), depth_scale=1000.0)

    _save(tmp_path, [tum, scannet])

    png_tum, png_scannet = _read_png(tmp_path, 0), _read_png(tmp_path, 1)
    np.testing.assert_array_equal(
        png_tum, png_scannet,
        err_msg="holdout_gt depth PNG must not depend on the source dataset scale",
    )
    np.testing.assert_array_equal(png_tum, encode_holdout_depth_mm(metres))
    np.testing.assert_allclose(decode_holdout_depth_m(png_tum), metres, atol=1e-3)


def test_reading_the_png_with_the_dataset_scale_is_the_retracted_bug(tmp_path,
                                                                    patched_producer):
    """Demonstrates the defect explicitly so its signature stays recorded: the
    wrong divisor does not raise, it just shrinks the scene by exactly 5x."""
    metres = np.full((H, W), 2.0, dtype=np.float32)
    _save(tmp_path, [_Cam((metres * 5000.0).astype(np.uint16), depth_scale=5000.0)])
    png = _read_png(tmp_path, 0)

    correct = decode_holdout_depth_m(png)
    wrong = png.astype(np.float32) / 5000.0          # the retracted divisor
    np.testing.assert_allclose(correct, 2.0, atol=1e-3)
    np.testing.assert_allclose(wrong, 0.4, atol=1e-3)
    assert np.allclose(correct / wrong, 5.0), "shrink factor must be exactly 5x"


def test_producer_actually_wrote_a_file(tmp_path, patched_producer):
    """Guards against the whole suite passing on a silent no-write: the
    producer's body sits inside a bare try/except that only prints."""
    _save(tmp_path, [_Cam(np.full((H, W), 1500, np.uint16), 1000.0)])
    assert (tmp_path / "holdout_gt" / "0_depth.png").exists()


def test_no_depth_available_writes_no_png(tmp_path, monkeypatch):
    """The negative case must stay negative: no sensor depth -> no file, rather
    than a zero-filled PNG that would read back as 'the whole scene is at 0 m'."""
    import eligsir.utils.comparison_report as cr
    monkeypatch.setattr(cr, "_get_sensor_depth", lambda cam, h, w: None, raising=True)
    monkeypatch.setattr(cr, "_gt_image", lambda cam: torch.zeros(3, H, W), raising=True)
    _save(tmp_path, [_Cam(np.zeros((H, W), np.uint16), 1000.0)])
    assert not (tmp_path / "holdout_gt" / "0_depth.png").exists()


# --------------------------------------------------------------------------
# every reader must use the shared constant, asserted by value
# --------------------------------------------------------------------------

def test_render_report_reader_round_trips_through_the_contract(tmp_path,
                                                               patched_producer,
                                                               monkeypatch):
    """`render_report._attach_holdout_gt` is the other on-disk reader. It calls
    `.cuda()`, so patch that to identity rather than skipping on a CPU box --
    a skipped test guards nothing here."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self, raising=False)
    import render_report

    metres = np.full((H, W), 3.25, dtype=np.float32)
    _save(tmp_path, [_Cam((metres * 5000.0).astype(np.uint16), depth_scale=5000.0)])

    class _MiniCam:
        pass

    cam = _MiniCam()
    # device="cpu" (issue #639 made the device move a parameter): this test
    # asserts on .numpy(), so it failed on any GPU-present surface before.
    render_report._attach_holdout_gt(cam, str(tmp_path / "holdout_gt"), 0, device="cpu")
    assert cam._streaming_depth_scale == HOLDOUT_GT_DEPTH_SCALE
    np.testing.assert_allclose(
        cam._sensor_depth_cache.numpy().reshape(H, W), metres, atol=1e-3
    )


def test_local_quality_from_run_default_divisor_is_the_contract():
    """The Level A offline driver's default must be the contract constant, not
    the run's own dataset depth_scale (which is what it was fixed away from)."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "_lqfr",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "local_quality_from_run.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HOLDOUT_GT_DEPTH_SCALE == HOLDOUT_GT_DEPTH_SCALE


def test_offline_demo_divisor_is_the_contract():
    """`scripts/local_quality_offline_demo.py` is the exact site of the #358
    retraction; its constant is pinned so a 5000 cannot come back."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "_lqod",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "local_quality_offline_demo.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEPTH_SCALE == HOLDOUT_GT_DEPTH_SCALE
