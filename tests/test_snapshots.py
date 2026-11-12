import pytest
import torch
import torch.nn as nn
from plyfile import PlyData

from eligsir.io.snapshots import snapshot_gaussians_for_ply, write_ply_from_snapshot
from scene.gsplat_model import GsplatGaussianModel

# Issue #555: GPU-only (model construction moves params to CUDA). Without
# device passthrough these fail with `No HIP GPUs are available` -- a false
# RED, not a defect. Capability probe, never an image tag. See
# tests/test_gaussian_model_ply_degree.py for the full rationale.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def _make_model(n=6, max_sh_degree=3, active_sh_degree=1):
    model = GsplatGaussianModel(max_sh_degree)
    model.active_sh_degree = active_sh_degree
    model.params = nn.ParameterDict({
        "means": nn.Parameter(torch.randn(n, 3)),
        "scales": nn.Parameter(torch.randn(n, 3)),
        "quats": nn.Parameter(torch.randn(n, 4)),
        "opacities": nn.Parameter(torch.randn(n)),
        "sh0": nn.Parameter(torch.randn(n, 1, 3)),
        "shN": nn.Parameter(torch.randn(n, (max_sh_degree + 1) ** 2 - 1, 3)),
    })
    return model


def test_snapshot_and_write_at_sub_max_degree_succeeds(tmp_path):
    model = _make_model(n=6, max_sh_degree=3, active_sh_degree=1)

    snapshot = snapshot_gaussians_for_ply(model)
    out_path = str(tmp_path / "snapshot.ply")
    write_ply_from_snapshot(snapshot, out_path)  # must not raise ValueError

    plydata = PlyData.read(out_path)
    n_active_bands = (1 + 1) ** 2 - 1  # 3
    n_rest_props = sum(
        1 for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    )
    assert n_rest_props == 3 * n_active_bands
    assert any(c.strip() == "active_sh_degree 1" for c in plydata.comments)


def test_write_ply_from_snapshot_round_trips_through_load_ply(tmp_path):
    model = _make_model(n=6, max_sh_degree=3, active_sh_degree=1)
    n_active_bands = (1 + 1) ** 2 - 1

    snapshot = snapshot_gaussians_for_ply(model)
    out_path = str(tmp_path / "snapshot.ply")
    write_ply_from_snapshot(snapshot, out_path)

    reloaded = GsplatGaussianModel(3)
    reloaded.load_ply(out_path)

    assert reloaded.active_sh_degree == 1
    max_bands = (3 + 1) ** 2 - 1
    assert reloaded.params["shN"].shape == (6, max_bands, 3)
    assert torch.allclose(
        reloaded.params["shN"][:, :n_active_bands, :].cpu(),
        model.params["shN"][:, :n_active_bands, :].cpu(),
        atol=1e-5,
    )
    assert torch.all(reloaded.params["shN"][:, n_active_bands:, :].cpu() == 0)


def test_snapshot_at_max_degree_still_works(tmp_path):
    model = _make_model(n=4, max_sh_degree=3, active_sh_degree=3)

    snapshot = snapshot_gaussians_for_ply(model)
    out_path = str(tmp_path / "snapshot_max.ply")
    write_ply_from_snapshot(snapshot, out_path)  # must not raise

    plydata = PlyData.read(out_path)
    max_bands = (3 + 1) ** 2 - 1
    n_rest_props = sum(
        1 for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    )
    assert n_rest_props == 3 * max_bands
    assert any(c.strip() == "active_sh_degree 3" for c in plydata.comments)


def test_subset_ply_snapshot_preserves_active_sh_degree(tmp_path):
    from eligsir.io.snapshots import subset_ply_snapshot

    model = _make_model(n=6, max_sh_degree=3, active_sh_degree=1)
    snapshot = snapshot_gaussians_for_ply(model)
    mask = torch.tensor([True, False, True, True, False, False])

    subset = subset_ply_snapshot(snapshot, mask)

    assert subset["active_sh_degree"] == 1
    out_path = str(tmp_path / "subset.ply")
    write_ply_from_snapshot(subset, out_path)  # must not raise

    plydata = PlyData.read(out_path)
    assert any(c.strip() == "active_sh_degree 1" for c in plydata.comments)
