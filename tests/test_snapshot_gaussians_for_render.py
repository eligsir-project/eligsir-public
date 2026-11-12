"""Regression test: snapshot_gaussians_for_render() must fully decouple
from the live Gaussian model, so a background render thread reading the
snapshot is unaffected by the main thread continuing to mutate the live
model in-place (in-place mutation is exactly what training does every
iteration via optimizer steps and MCMC relocation).
"""
import pytest
import torch

from arguments import ModelParams
from scene.gsplat_model import GsplatGaussianModel
from scene.gaussian_model import GaussianModel
from eligsir.io.snapshots import snapshot_gaussians_for_render

# Issue #555: GPU-only (model construction moves params to CUDA). Without
# device passthrough these fail with `No HIP GPUs are available` -- a false
# RED, not a defect. Capability probe, never an image tag. See
# tests/test_gaussian_model_ply_degree.py for the full rationale.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def _make_gsplat_model(n=8, sh_degree=1):
    m = GsplatGaussianModel(sh_degree)
    m.params["means"] = torch.nn.Parameter(torch.randn(n, 3, device="cuda" if torch.cuda.is_available() else "cpu"))
    m.params["sh0"] = torch.nn.Parameter(torch.randn(n, 1, 3, device=m.params["means"].device))
    n_rest = (sh_degree + 1) ** 2 - 1
    m.params["shN"] = torch.nn.Parameter(torch.randn(n, n_rest, 3, device=m.params["means"].device))
    m.params["opacities"] = torch.nn.Parameter(torch.randn(n, device=m.params["means"].device))
    m.params["scales"] = torch.nn.Parameter(torch.randn(n, 3, device=m.params["means"].device))
    m.params["quats"] = torch.nn.Parameter(torch.randn(n, 4, device=m.params["means"].device))
    m.active_sh_degree = sh_degree
    return m


def test_gsplat_layout_snapshot_is_isolated_from_live_mutation():
    model = _make_gsplat_model()
    snap = snapshot_gaussians_for_render(model)

    xyz_before = snap.get_xyz.clone()
    with torch.no_grad():
        model.params["means"].add_(100.0)  # in-place mutation, like training does

    assert torch.equal(snap.get_xyz, xyz_before), "snapshot must not see live in-place mutation"
    assert snap.active_sh_degree == 1


def test_legacy_layout_snapshot_is_isolated_from_live_mutation():
    model = GaussianModel(sh_degree=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8
    model._xyz = torch.nn.Parameter(torch.randn(n, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.randn(n, 1, 3, device=device))
    model._features_rest = torch.nn.Parameter(torch.randn(n, 3, 3, device=device))
    model._opacity = torch.nn.Parameter(torch.randn(n, 1, device=device))
    model._scaling = torch.nn.Parameter(torch.randn(n, 3, device=device))
    model._rotation = torch.nn.Parameter(torch.randn(n, 4, device=device))
    model.active_sh_degree = 1

    snap = snapshot_gaussians_for_render(model)
    xyz_before = snap.get_xyz.clone()
    with torch.no_grad():
        model._xyz.add_(100.0)

    assert torch.equal(snap.get_xyz, xyz_before)
    assert snap.active_sh_degree == 1
