import torch
import pytest
from argparse import Namespace
from scene.gsplat_model import GsplatGaussianModel
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP, BIRTH_METHOD_SOURCE_DEPTH

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GsplatGaussianModel is CUDA-only")


def _make_training_args():
    """Minimal training args namespace for model.training_setup(), matching the
    pattern used by tests/test_gaussian_uid.py's _make_model() helper -- needed
    so add_points_as_gaussians()'s optimizer-extension path has real optimizers
    to extend."""
    return Namespace(
        iterations=100,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=20000,
        feature_lr=0.0025,
        feature_lr_final=0.00025,
        opacity_lr=0.05,
        opacity_lr_final=0.001,
        scaling_lr=0.005,
        scaling_lr_final=0.0005,
        rotation_lr=0.001,
        rotation_lr_final=0.0001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=30000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        random_background=False,
    )


def _bootstrap_model(n=4):
    model = GsplatGaussianModel(sh_degree=0)
    pts = torch.randn(n, 3)
    cols = torch.rand(n, 3)
    from eligsir.utils.graphics_utils import BasicPointCloud
    pcd = BasicPointCloud(points=pts.numpy(), colors=cols.numpy(), normals=torch.zeros(n, 3).numpy())
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_make_training_args())
    return model


def test_bootstrap_seeds_lineage_as_self_root():
    model = _bootstrap_model(4)
    assert model.parent_uid.shape[0] == 4
    # model buffers live on cuda; compare via .cpu() (brief's literal comparison
    # tensors omit device="cuda", which errors when compared directly against a
    # cuda tensor -- see task-2-report.md's "Deviations" section).
    assert torch.equal(model.parent_uid.cpu(), torch.full((4,), -1, dtype=torch.int64))
    assert torch.equal(model.root_uid, model.gaussian_uid)
    assert torch.equal(model.generation.cpu(), torch.zeros(4, dtype=torch.int32))
    assert torch.equal(model.birth_method.cpu(), torch.full((4,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8))
    assert torch.equal(model.last_reseed_iter.cpu(), torch.full((4,), -1, dtype=torch.int32))


def test_insert_seeds_lineage_as_new_source_depth_root_with_birth_iter():
    model = _bootstrap_model(2)
    pts = torch.randn(3, 3).cuda()
    cols = torch.rand(3, 3).cuda()
    added = model.add_points_as_gaussians(pts, cols, birth_iter=30000)
    assert added == 3
    new_uid = model.gaussian_uid[-3:]
    assert torch.equal(model.root_uid[-3:], new_uid)
    assert torch.equal(model.parent_uid[-3:], torch.full((3,), -1, dtype=torch.int64, device="cuda"))
    assert torch.equal(model.birth_method[-3:], torch.full((3,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8, device="cuda"))
    assert torch.equal(model.birth_iter[-3:].cpu(), torch.full((3,), 30000, dtype=torch.int32))
    assert torch.equal(model.anchor_iter[-3:].cpu(), torch.full((3,), 30000, dtype=torch.int32))


def test_prune_points_reindexes_lineage_buffers():
    model = _bootstrap_model(4)
    mask = torch.tensor([True, False, True, False], device="cuda")  # prune indices 0 and 2
    model.prune_points(mask)
    assert model.parent_uid.shape[0] == 2
    assert model.root_uid.shape[0] == 2
    assert model.generation.shape[0] == 2
    assert model.birth_method.shape[0] == 2
    assert model.last_reseed_iter.shape[0] == 2
