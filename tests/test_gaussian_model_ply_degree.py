import os
import tempfile

import pytest
import torch
from plyfile import PlyData

from scene.gaussian_model import GaussianModel

# Issue #555. GaussianModel/GsplatGaussianModel move parameters to CUDA on
# construction, so on a container WITHOUT device passthrough
# (`training-nogpu`, per scripts/run_test_surface.sh) every test here dies
# with `RuntimeError: No HIP GPUs are available` -- a false RED that says
# nothing about the code under test. These pass on `training-gpu`. Gated on
# the real device probe, never on an image tag (PR #554 showed tag-based
# surface detection silently lies).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def _make_model(max_sh_degree=3, active_sh_degree=1, n=5):
    model = GaussianModel(max_sh_degree)
    model.active_sh_degree = active_sh_degree
    model._xyz = torch.nn.Parameter(torch.randn(n, 3))
    model._features_dc = torch.nn.Parameter(torch.randn(n, 1, 3))
    model._features_rest = torch.nn.Parameter(
        torch.randn(n, (max_sh_degree + 1) ** 2 - 1, 3)
    )
    model._opacity = torch.nn.Parameter(torch.randn(n, 1))
    model._scaling = torch.nn.Parameter(torch.randn(n, 3))
    model._rotation = torch.nn.Parameter(torch.randn(n, 4))
    return model


def test_save_load_round_trip_preserves_trained_degree_and_zero_pads_rest():
    src = _make_model(max_sh_degree=3, active_sh_degree=1, n=5)
    n_active_bands = (1 + 1) ** 2 - 1  # 3
    max_bands = (3 + 1) ** 2 - 1  # 15

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "point_cloud.ply")
        src.save_ply(path)

        plydata = PlyData.read(path)
        assert any(
            c.strip() == "active_sh_degree 1" for c in plydata.comments
        ), f"expected 'active_sh_degree 1' comment, got {plydata.comments}"
        n_rest_props = sum(
            1 for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
        )
        assert n_rest_props == 3 * n_active_bands

        dst = GaussianModel(3)
        dst.load_ply(path)

        assert dst.active_sh_degree == 1
        assert dst._features_rest.shape == (5, max_bands, 3)
        # trained bands round-trip exactly
        assert torch.allclose(
            dst._features_rest[:, :n_active_bands, :].cpu(),
            src._features_rest[:, :n_active_bands, :],
            atol=1e-5,
        )
        # untrained bands are zero
        assert torch.all(dst._features_rest[:, n_active_bands:, :].cpu() == 0)


def test_load_ply_without_comment_falls_back_to_max_degree():
    src = _make_model(max_sh_degree=3, active_sh_degree=3, n=4)  # saved at max degree already
    with tempfile.TemporaryDirectory() as tmp:
        # First, save the model
        path1 = os.path.join(tmp, "point_cloud_new.ply")
        src.save_ply(path1)

        # Now load it back and verify it has the comment
        plydata = PlyData.read(path1)
        has_comment = any(
            c.strip().startswith("active_sh_degree") for c in plydata.comments
        )
        assert has_comment, f"Expected active_sh_degree comment, got {plydata.comments}"

        # To test the fallback, we manually create a PLY without the comment
        # by reading properties directly and reconstructing without comments
        path_old_format = os.path.join(tmp, "point_cloud_old.ply")

        # Create PLY data without active_sh_degree comment
        from plyfile import PlyElement
        el = PlyElement.describe(plydata.elements[0].data, 'vertex')
        from plyfile import PlyData as PlyDataClass
        PlyDataClass([el]).write(path_old_format)

        # Load without comment should fallback to max_sh_degree
        dst = GaussianModel(3)
        dst.load_ply(path_old_format)

        assert dst.active_sh_degree == 3
        assert dst._features_rest.shape == (4, 15, 3)
