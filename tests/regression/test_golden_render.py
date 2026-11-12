"""Golden-image regression test (graphics-test pattern).

Renders a small committed fixture PLY from fixed cameras and asserts the output
matches committed golden PNGs within pixel + perceptual (LPIPS) tolerance. Guards
the render pipeline against silent regressions.

Determinism: renders from a FIXED committed PLY (not a fresh train), seed-free
because rasterization of a fixed scene/camera is deterministic.

Regenerate goldens after an intentional render change:
    SPLAT_REGEN_GOLDEN=1 python -m pytest tests/regression/test_golden_render.py
(run inside the container — needs torch/CUDA + project deps).
"""
import glob
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")
FIXTURE_PLY = os.path.join(GOLDEN_DIR, "fixture.ply")
FIXTURE_CAMS = os.path.join(GOLDEN_DIR, "cameras.json")

PIXEL_MSE_TOL = 2.0      # mean squared error on 0-255 uint8 (tiny GPU nondeterminism)
LPIPS_TOL = 0.01         # perceptual tolerance


def _have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_cuda() or not os.path.isfile(FIXTURE_PLY),
    reason="golden-image test needs CUDA + a committed fixture (run in container)",
)


def _render_fixture(cam_index):
    """Render the fixture PLY from its own cameras.json at cam_index -> uint8 HxWx3."""
    import json
    import numpy as np
    from eligsir.eval._render import _build_minicam
    import torch
    from gaussian_renderer import render
    from scene.gsplat_model import GsplatGaussianModel
    from types import SimpleNamespace

    g = GsplatGaussianModel(3)
    g.load_ply(FIXTURE_PLY)
    entry = json.load(open(FIXTURE_CAMS))[cam_index]
    cam = _build_minicam(entry)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        out = render(cam, g, pipe, bg)["render"].clamp(0, 1)
    return (out.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")


def _golden_path(cam_index):
    return os.path.join(GOLDEN_DIR, f"golden_cam{cam_index}.png")


@pytest.mark.parametrize("cam_index", [0, 1])
def test_render_matches_golden(cam_index):
    import json
    import numpy as np
    from PIL import Image

    n_cams = len(json.load(open(FIXTURE_CAMS)))
    if cam_index >= n_cams:
        pytest.skip(f"fixture has only {n_cams} cameras")

    img = _render_fixture(cam_index)
    gp = _golden_path(cam_index)

    if os.environ.get("SPLAT_REGEN_GOLDEN") == "1":
        Image.fromarray(img).save(gp)
        pytest.skip(f"regenerated golden {gp}")

    assert os.path.isfile(gp), f"missing golden {gp}; run with SPLAT_REGEN_GOLDEN=1"
    golden = np.asarray(Image.open(gp).convert("RGB"))
    assert golden.shape == img.shape

    mse = float(((img.astype("float32") - golden.astype("float32")) ** 2).mean())
    assert mse <= PIXEL_MSE_TOL, f"pixel MSE {mse:.3f} > {PIXEL_MSE_TOL}"

    # Perceptual gate (LPIPS); cached net.
    import torch
    from lpipsPyTorch import lpips
    a = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
    b = torch.from_numpy(golden).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
    with torch.no_grad():
        d = float(lpips(a, b, net_type="vgg").item())
    assert d <= LPIPS_TOL, f"LPIPS {d:.4f} > {LPIPS_TOL}"
