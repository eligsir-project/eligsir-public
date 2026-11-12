"""Regression tests for issue #1326.

Root cause (confirmed on real GPU hardware, r9700/gfx1201): gsplat's packed=True
forward rasterization kernel non-deterministically corrupts every camera slot
beyond index 0 in a C>1 batch whenever `colors` is a per-camera-varying
(C, N, 3) tensor with sh_degree=None -- exactly the shape
render_batch()'s sh_impl="python" branch builds. Repeated calls with
byte-identical inputs produced different (sometimes wildly out-of-range,
e.g. >100 pre-clamp) results for the same camera slot -- consistent with the
kernel reading uninitialized/reused GPU memory, not a deterministic indexing
bug. Switching to packed=False reproduced the correct reference render
across many repeated attempts with no corruption, and costs nothing here
because render_batch() always calls with sparse_grad=False (sparse
gradients are gsplat-documented as "only effective when packed is True", so
disabling packed mode does not disable any feature this function actually
uses).

These tests are CPU-only wiring checks (gsplat.rendering.rasterization is
monkeypatched) -- they cannot exercise the real GPU kernel bug itself (that
requires real GPU hardware; see the manual verification recorded in the
issue). What they DO guard against:
  1. A regression back to packed=True.
  2. Any regression in render_batch()'s OWN per-camera output assembly
     (render_colors[c_idx] -> results list), at EVERY chunk-relative index,
     not just the specific index (2) originally reported -- reproducing the
     exact chunk_size=8 / chunk_size=7 isolation pattern from the issue.
"""
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

import torch

from eligsir.mapping.renderer.gsplat_backend import render_batch


@dataclass
class _FakeGsplatCam:
    viewmat: torch.Tensor
    K: torch.Tensor
    camera_center: torch.Tensor


class _FakeCam:
    """Minimal camera stand-in: pre-populated `.gsplat` short-circuits
    _camera_tensors() so no real Camera/scene machinery is needed."""

    def __init__(self, idx, width=8, height=6):
        self.image_width = width
        self.image_height = height
        self.gsplat = _FakeGsplatCam(
            viewmat=torch.eye(4, dtype=torch.float32),
            K=torch.eye(3, dtype=torch.float32),
            camera_center=torch.zeros(1, 3, dtype=torch.float32),
        )
        self.idx = idx


class _FakePC:
    def __init__(self, n=5):
        self.active_sh_degree = 0
        self._xyz = torch.zeros(n, 3)
        self._features_dc = torch.zeros(n, 1, 3)
        self._features_rest = torch.zeros(n, 0, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.ones(self._xyz.shape[0], 3)

    @property
    def get_rotation(self):
        q = torch.zeros(self._xyz.shape[0], 4)
        q[:, 0] = 1.0
        return q

    @property
    def get_opacity(self):
        return torch.ones(self._xyz.shape[0], 1)


def _install_fake_rasterization(monkeypatch, captured_calls, width=8, height=6):
    def fake_rasterization(**kwargs):
        captured_calls.append(kwargs)
        C = kwargs["viewmats"].shape[0]
        # Fill each camera's output with its own batch-slot index (as a
        # constant color) so the test can verify render_batch() attributes
        # render_colors[c_idx] to the correct output position.
        render_colors = torch.zeros(C, height, width, 3)
        for c_idx in range(C):
            # Scaled to stay within render_batch()'s final clamp(0, 1) so
            # every chunk-relative index remains distinguishable after
            # clamping, not just index 0/1.
            render_colors[c_idx] = float(c_idx) / 1000.0
        render_alphas = torch.ones(C, height, width, 1)
        return render_colors, render_alphas, {}

    # The assertions below are CPU wiring tests. Give the lazy renderer import
    # the one upstream symbol it needs without requiring the GPU-only package.
    rendering = ModuleType("gsplat.rendering")
    rendering.rasterization = fake_rasterization
    gsplat = ModuleType("gsplat")
    gsplat.rendering = rendering
    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)


def test_render_batch_uses_packed_false(monkeypatch):
    captured = []
    _install_fake_rasterization(monkeypatch, captured)
    cams = [_FakeCam(i) for i in range(8)]
    pc = _FakePC()
    pipe = SimpleNamespace(sh_impl="python")
    bg = torch.zeros(3)

    render_batch(cams, pc, pipe, bg, chunk_size=8)

    assert captured, "rasterization was never called"
    for call in captured:
        assert call["packed"] is False, (
            "render_batch must call gsplat rasterization with packed=False "
            "(issue #1326) -- packed=True corrupts camera slots beyond "
            "index 0 in a C>1 batch"
        )


def test_render_batch_uses_packed_false_for_gsplat_sh_impl(monkeypatch):
    captured = []
    _install_fake_rasterization(monkeypatch, captured)
    cams = [_FakeCam(i) for i in range(5)]
    pc = _FakePC()
    pipe = SimpleNamespace(sh_impl="gsplat")
    bg = torch.zeros(3)

    render_batch(cams, pc, pipe, bg, chunk_size=5)

    assert captured
    for call in captured:
        assert call["packed"] is False


def test_render_batch_background_is_per_camera_shaped_for_unpacked_mode(monkeypatch):
    captured = []
    _install_fake_rasterization(monkeypatch, captured)
    cams = [_FakeCam(i) for i in range(4)]
    pc = _FakePC()
    pipe = SimpleNamespace(sh_impl="python")
    bg = torch.tensor([0.1, 0.2, 0.3])

    render_batch(cams, pc, pipe, bg, chunk_size=4)

    call = captured[0]
    backgrounds = call["backgrounds"]
    assert backgrounds is not None
    assert backgrounds.shape == (4, 3), (
        "packed=False requires a per-camera (C, 3) background, not the flat "
        "(3,) tensor packed=True mode accepted"
    )
    for c_idx in range(4):
        assert torch.equal(backgrounds[c_idx], bg)


def test_render_batch_no_corruption_at_any_chunk_relative_index(monkeypatch):
    """Reproduces the exact isolation pattern from issue #1326: 60 cameras,
    chunk_size=8 and chunk_size=7, checking EVERY camera at EVERY
    chunk-relative index -- not just spot-checking the originally-reported
    index 2."""
    n_cams = 60
    pc = _FakePC()
    pipe = SimpleNamespace(sh_impl="python")
    bg = torch.zeros(3)

    for chunk_size in (8, 7, 16, 32):
        captured = []
        _install_fake_rasterization(monkeypatch, captured)
        cams = [_FakeCam(i) for i in range(n_cams)]

        results = render_batch(cams, pc, pipe, bg, chunk_size=chunk_size)

        assert len(results) == n_cams
        for abs_idx, img in enumerate(results):
            chunk_rel = abs_idx % chunk_size
            # Every pixel of every channel must equal this camera's own
            # batch-slot index within its chunk (see fake_rasterization);
            # any deviation means render_batch mis-attributed a different
            # camera's rendered output to this position.
            # The fake fills render_colors[c_idx] = c_idx / 1000, where c_idx
            # is the position WITHIN the chunk -- i.e. chunk_rel.
            expected = float(chunk_rel) / 1000.0
            assert torch.allclose(img, torch.full_like(img, expected)), (
                f"chunk_size={chunk_size} abs_idx={abs_idx} chunk_rel={chunk_rel}: "
                f"expected constant {expected}, got mean {img.mean().item()} "
                "-- render_batch mis-assembled per-camera output"
            )
