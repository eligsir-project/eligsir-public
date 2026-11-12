"""Verifies the D-reconstruction identity added for the D-vs-ED depth-loss
experiment: result["rendered_depth"] (D) must be reconstructable from the
new RGB+ED render mode as D = ED * alpha, matching pre-change D values
exactly (float32 precision) for every existing consumer.

Fixture note: the second test below reuses this codebase's existing
render()-fixture pattern verbatim (real ``GsplatGaussianModel`` built via
``create_from_pcd``, real ``scene.cameras.Camera``, real CLI-parser-derived
``args``/``opt``/``pipe``) from
``eligsir/optimization/gaussian/tests/test_gpu_parity_harness.py``'s
``_build_gaussians``/``_build_camera``/``_build_args``/``_bg`` helpers,
rather than hand-rolling a new fixture, per this codebase's established
convention (see also ``tests/regression/test_golden_render.py``, the
original source of that recipe).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import gsplat  # noqa: F401
    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "test_rendered_depth_ed_reconstruction requires CUDA/ROCm + gsplat "
        "(no CPU rendering path exists in this codebase); run inside the "
        "training container with GPU passthrough per CLAUDE.md's Running "
        "Commands section: docker compose run --rm train python -m pytest "
        "eligsir/mapping/renderer/tests/ -q"
    ),
)


@pytest.mark.cuda
def test_rendered_depth_reconstruction_matches_direct_d():
    """result["rendered_depth"] (reconstructed via D = ED * alpha) must match
    a direct RGB+D render of the identical scene, within float32 tolerance."""
    from gsplat.rendering import rasterization

    device = "cuda"
    torch.manual_seed(0)
    means = torch.tensor([[0.0, 0.0, 2.0], [0.3, -0.2, 2.5]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
    scales = torch.full((2, 3), 0.1, device=device)
    opacities = torch.tensor([0.8, 0.3], device=device)
    colors = torch.rand(2, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]], device=device
    )

    render_colors_ed, render_alphas_ed, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks, width=64, height=64, sh_degree=None,
        render_mode="RGB+ED", packed=True, tile_size=16,
    )
    ed = render_colors_ed[..., 3:4]
    alpha = render_alphas_ed
    d_reconstructed = ed * alpha

    render_colors_d, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks, width=64, height=64, sh_degree=None,
        render_mode="RGB+D", packed=True, tile_size=16,
    )
    d_direct = render_colors_d[..., 3:4]

    assert torch.allclose(d_reconstructed, d_direct, atol=1e-4, rtol=1e-4), (
        f"max abs diff: {(d_reconstructed - d_direct).abs().max().item()}"
    )


@pytest.mark.cuda
def test_render_function_rendered_depth_key_unchanged_by_this_change():
    """The public render() function's result["rendered_depth"] must be
    value-identical whether obtained via the old RGB+D path or the new
    RGB+ED-then-reconstruct path -- i.e. this task's change is a no-op for
    any existing caller of render(render_depth=True). Also checks that the
    new "rendered_depth_ed" key is a real, present, correctly-shaped
    by-product of the same change."""
    from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
        _bg,
        _build_args,
        _build_camera,
        _build_gaussians,
    )

    from gaussian_renderer import render

    gaussians = _build_gaussians()
    cam = _build_camera()
    args, opt, pipe, ns = _build_args()

    render_pkg = render(cam, gaussians, pipe, _bg(), update_sh_rest=False, render_depth=True)

    assert "rendered_depth" in render_pkg
    assert "rendered_depth_ed" in render_pkg
    d = render_pkg["rendered_depth"]
    ed = render_pkg["rendered_depth_ed"]
    alpha = render_pkg["alpha"]
    assert d.shape == ed.shape == (1, cam.image_height, cam.image_width)

    # Independently reconstruct D from ED using this file's own permute
    # logic (mirrors gsplat_backend.py's fix: render_pkg["alpha"] is
    # [H, W, 1], not [1, H, W] -- confirmed via a real-kernel probe
    # 2026-08-11 -- so it must be permuted, not multiplied in directly).
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha_hw = alpha.permute(2, 0, 1)
    else:
        alpha_hw = alpha
    d_reconstructed_from_pkg = ed * alpha_hw

    assert torch.equal(d, d_reconstructed_from_pkg), (
        "render()'s rendered_depth must equal rendered_depth_ed * alpha exactly "
        "(same tensor, same op, no extra rounding introduced between the "
        "renderer's internal reconstruction and this external check)"
    )

    # Direct RGB+D render of the exact same scene, as an independent
    # reference for "unchanged by this task's change" (not just internally
    # self-consistent).
    from eligsir.mapping.renderer.gsplat_backend import _camera_tensors

    viewmat, K, cam_center = _camera_tensors(cam, gaussians.get_xyz.device)
    from gsplat.rendering import rasterization

    means = gaussians.get_xyz
    quats = gaussians.get_rotation
    scales = gaussians.get_scaling
    opacities = gaussians.get_opacity.squeeze(-1)
    from eligsir.mapping.renderer.gsplat_backend import _python_sh_colors

    colors = _python_sh_colors(
        gaussians, means, cam_center, False, pipe, False, compiled=False
    )
    render_colors_d, _, _ = rasterization(
        means=means.detach(), quats=quats.detach(), scales=scales.detach(),
        opacities=opacities.detach(), colors=colors.detach(),
        viewmats=viewmat[None].contiguous(), Ks=K[None].contiguous(),
        width=cam.image_width, height=cam.image_height, sh_degree=None,
        packed=True, tile_size=16, render_mode="RGB+D",
    )
    d_direct = render_colors_d[0].permute(2, 0, 1)[3:4]

    assert torch.allclose(d.detach(), d_direct, atol=1e-4, rtol=1e-4), (
        f"max abs diff vs independent direct RGB+D render: "
        f"{(d.detach() - d_direct).abs().max().item()}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("opacities_init", [(0.8, 0.3), (0.02, 0.015)])
def test_ed_reconstruction_gradient_parity_with_direct_d(opacities_init):
    """Gradient-parity check for the D-vs-ED reconstruction (2026-08-11 review
    finding #1): D = ED * alpha must produce IDENTICAL gradients w.r.t.
    means/scales/quats/opacities as a direct RGB+D render of the same scene,
    not just matching forward values. This is expected to hold exactly
    (up to float32 rounding) because gsplat's ED is a plain Python-level
    post-kernel divide on the SAME kernel outputs (D_raw, alpha_raw) --
    verified via source inspection 2026-08-11 -- so by the ordinary quotient
    rule, d/dtheta(alpha_raw * (D_raw / alpha_raw)) collapses algebraically
    to dD_raw/dtheta whenever the divide's eps-clamp doesn't engage.

    Parametrized over a normal-alpha regime (opacities ~0.3-0.8, clamp far
    away) and a low-alpha regime (opacities ~0.015-0.02, alpha per-pixel
    much closer to -- though per gsplat_backend.py's comment, still not
    reaching -- the 1e-10 clamp floor) to cover the case this review flagged
    as the riskiest: near-clamp low-alpha pixels.
    """
    from gsplat.rendering import rasterization

    device = "cuda"

    def _make_params():
        torch.manual_seed(0)
        means = torch.tensor(
            [[0.0, 0.0, 2.0], [0.3, -0.2, 2.5]], device=device, requires_grad=True
        )
        quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2, device=device, requires_grad=True
        )
        scales = torch.full((2, 3), 0.1, device=device, requires_grad=True)
        opacities = torch.tensor(list(opacities_init), device=device, requires_grad=True)
        colors = torch.rand(2, 3, device=device, requires_grad=True)
        return means, quats, scales, opacities, colors

    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]], device=device
    )
    # Non-uniform per-pixel weights so the loss isn't a trivial global sum
    # (which could mask a gradient-shape bug that only shows up per-pixel).
    torch.manual_seed(1)
    pixel_weights = torch.rand(1, 64, 64, device=device)

    # -- Path A: direct RGB+D --
    means_a, quats_a, scales_a, opacities_a, colors_a = _make_params()
    render_colors_d, _, _ = rasterization(
        means=means_a, quats=quats_a, scales=scales_a, opacities=opacities_a,
        colors=colors_a, viewmats=viewmats, Ks=Ks, width=64, height=64,
        sh_degree=None, render_mode="RGB+D", packed=True, tile_size=16,
    )
    d_direct = render_colors_d[..., 3:4].squeeze(-1)
    loss_a = (d_direct * pixel_weights).sum()
    loss_a.backward()

    # -- Path B: RGB+ED, reconstruct D = ED * alpha --
    means_b, quats_b, scales_b, opacities_b, colors_b = _make_params()
    render_colors_ed, render_alphas_ed, _ = rasterization(
        means=means_b, quats=quats_b, scales=scales_b, opacities=opacities_b,
        colors=colors_b, viewmats=viewmats, Ks=Ks, width=64, height=64,
        sh_degree=None, render_mode="RGB+ED", packed=True, tile_size=16,
    )
    ed = render_colors_ed[..., 3:4]
    alpha = render_alphas_ed
    d_reconstructed = (ed * alpha).squeeze(-1)
    loss_b = (d_reconstructed * pixel_weights).sum()
    loss_b.backward()

    # Forward values must still match (sanity check, already covered above
    # for the non-parametrized case, but confirms the low-alpha regime too).
    assert torch.allclose(d_direct, d_reconstructed, atol=1e-4, rtol=1e-4), (
        f"forward value mismatch at opacities_init={opacities_init}: "
        f"max abs diff {(d_direct - d_reconstructed).abs().max().item()}"
    )

    for name, grad_a, grad_b in [
        ("means", means_a.grad, means_b.grad),
        ("quats", quats_a.grad, quats_b.grad),
        ("scales", scales_a.grad, scales_b.grad),
        ("opacities", opacities_a.grad, opacities_b.grad),
    ]:
        assert grad_a is not None and grad_b is not None, (
            f"{name}.grad is None on one of the two paths (opacities_init={opacities_init})"
        )
        assert torch.allclose(grad_a, grad_b, atol=1e-3, rtol=1e-2), (
            f"gradient mismatch for {name} at opacities_init={opacities_init}: "
            f"direct-D grad={grad_a}, ED-reconstructed grad={grad_b}, "
            f"max abs diff={(grad_a - grad_b).abs().max().item()}"
        )
