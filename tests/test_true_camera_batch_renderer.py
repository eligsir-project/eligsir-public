from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from eligsir.mapping.renderer import gsplat_backend, true_camera_batch


def test_packed_true_camera_batch_passes_one_shared_flat_background(monkeypatch):
    camera_count = 4
    width = 8
    height = 6
    captured: dict[str, object] = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros(camera_count, height, width, 3),
            torch.zeros(camera_count, height, width, 1),
            {},
        )

    # This test covers our C>1 adapter's call contract, not a native kernel.
    rendering = ModuleType("gsplat.rendering")
    rendering.rasterization = fake_rasterization
    gsplat = ModuleType("gsplat")
    gsplat.rendering = rendering
    monkeypatch.setitem(sys.modules, "gsplat", gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)
    monkeypatch.setattr(
        true_camera_batch,
        "make_gsplat_camera_batch",
        lambda _cameras, *, device: SimpleNamespace(
            widths=(width,) * camera_count,
            heights=(height,) * camera_count,
            viewmats=torch.eye(4, device=device).repeat(camera_count, 1, 1),
            Ks=torch.eye(3, device=device).repeat(camera_count, 1, 1),
        ),
    )
    monkeypatch.setattr(
        gsplat_backend,
        "_gsplat_sh_coeffs",
        lambda _pc, *, update_sh_rest: torch.zeros(3, 1, 3),
    )

    cameras = tuple(
        SimpleNamespace(image_width=width, image_height=height)
        for _ in range(camera_count)
    )
    pc = SimpleNamespace(
        get_xyz=torch.zeros(3, 3),
        get_scaling=torch.ones(3, 3),
        _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        get_opacity=torch.ones(3, 1),
        active_sh_degree=0,
    )
    pipe = SimpleNamespace(
        sh_impl="gsplat", gsplat_sparse_grad=True, rasterize_mode="antialiased"
    )
    background = torch.tensor([0.1, 0.2, 0.3])

    true_camera_batch.render_true_camera_batch(
        cameras,
        pc,
        pipe,
        background,
    )

    assert captured["packed"] is True
    assert captured["rasterize_mode"] == "antialiased"
    assert captured["backgrounds"].shape == (3,)
    assert torch.equal(captured["backgrounds"], background)
