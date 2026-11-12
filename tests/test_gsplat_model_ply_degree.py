import os
import tempfile

import pytest
import torch
from plyfile import PlyData

from scene.gsplat_model import GsplatGaussianModel

# Issue #555: GPU-only (model construction moves params to CUDA). Without
# device passthrough these fail with `No HIP GPUs are available` -- a false
# RED, not a defect. Capability probe, never an image tag. See
# tests/test_gaussian_model_ply_degree.py for the full rationale.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def _make_model(max_sh_degree=3, active_sh_degree=1, n=5):
    model = GsplatGaussianModel(max_sh_degree)
    model.active_sh_degree = active_sh_degree
    model.params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3)),
        "scales": torch.nn.Parameter(torch.randn(n, 3)),
        "quats": torch.nn.Parameter(torch.randn(n, 4)),
        "opacities": torch.nn.Parameter(torch.randn(n)),
        "sh0": torch.nn.Parameter(torch.randn(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.randn(n, (max_sh_degree + 1) ** 2 - 1, 3)),
    })
    return model


def test_save_load_round_trip_preserves_trained_degree_and_zero_pads_rest():
    src = _make_model(max_sh_degree=3, active_sh_degree=1, n=5)
    # Move source model to cuda to match load_ply behavior
    for key in src.params:
        src.params[key] = torch.nn.Parameter(src.params[key].cuda())

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

        dst = GsplatGaussianModel(sh_degree=3)
        dst.load_ply(path)

        assert dst.active_sh_degree == 1
        assert dst.params["shN"].shape == (5, max_bands, 3)
        assert torch.allclose(
            dst.params["shN"][:, :n_active_bands, :],
            src.params["shN"][:, :n_active_bands, :],
            atol=1e-5,
        )
        assert torch.all(dst.params["shN"][:, n_active_bands:, :] == 0)


def test_load_ply_without_comment_falls_back_to_max_degree():
    """Test that load_ply falls back to max_sh_degree when comment is missing."""
    src = _make_model(max_sh_degree=3, active_sh_degree=3, n=4)
    # Move source model to cuda to match load_ply behavior
    for key in src.params:
        src.params[key] = torch.nn.Parameter(src.params[key].cuda())

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "point_cloud.ply")
        src.save_ply(path)

        # Read the saved PLY, strip the comment, and save to a new file manually
        plydata = PlyData.read(path)
        plydata_no_comment = PlyData(
            elements=plydata.elements,
            comments=[],  # No comments
            text=plydata.text,
        )
        # Write to a different path to avoid the plyfile bus error
        path_no_comment = os.path.join(tmp, "point_cloud_no_comment.ply")
        plydata_no_comment.write(path_no_comment)

        dst = GsplatGaussianModel(sh_degree=3)
        dst.load_ply(path_no_comment)

        assert dst.active_sh_degree == 3
        assert dst.params["shN"].shape == (4, 15, 3)
