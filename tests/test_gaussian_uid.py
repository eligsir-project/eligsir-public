"""Tests for the persistent gaussian_uid buffer (Frame/Gaussian observation DB, task 1)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from argparse import Namespace

from scene.gsplat_model import GsplatGaussianModel

# Issue #555: GPU-only (model construction moves params to CUDA). Without
# device passthrough these fail with `No HIP GPUs are available` -- a false
# RED, not a defect. Capability probe, never an image tag. See
# tests/test_gaussian_model_ply_degree.py for the full rationale.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def _make_training_args():
    """Create a minimal training args namespace for model setup."""
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


def _make_model(n_points: int = 5) -> GsplatGaussianModel:
    model = GsplatGaussianModel(sh_degree=0)
    points = torch.rand(n_points, 3, device="cuda")
    colors = torch.rand(n_points, 3, device="cuda")
    from eligsir.utils.graphics_utils import BasicPointCloud
    pcd = BasicPointCloud(points=points.cpu().numpy(), colors=colors.cpu().numpy(), normals=np.zeros((n_points, 3)))
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_make_training_args())
    return model


@pytest.mark.cuda
def test_bootstrap_assigns_unique_sequential_uids():
    model = _make_model(n_points=5)
    assert model.gaussian_uid.shape[0] == 5
    assert model.gaussian_uid.dtype == torch.int64
    uids = model.gaussian_uid.cpu().tolist()
    assert len(set(uids)) == 5  # all unique
    assert uids == sorted(uids)  # birth-order monotonic at bootstrap
    # _make_model calls create_from_pcd AND training_setup, and both already
    # (pre-existing behavior) call _reset_running_state() -- so the counter
    # advances by `count` on EACH call, not just once. The buffer itself is
    # still correct (unique, monotonic, right shape); only the counter's
    # absolute value reflects both calls, so assert the invariant that
    # actually matters (counter is always exactly one-past the max live UID)
    # rather than a hardcoded value that assumes a single call.
    assert model._next_gaussian_uid == max(uids) + 1


@pytest.mark.cuda
def test_insertion_extends_uids_without_reuse():
    model = _make_model(n_points=3)
    before = set(model.gaussian_uid.cpu().tolist())
    new_points = torch.rand(2, 3, device="cuda")
    new_colors = torch.rand(2, 3, device="cuda")
    model.add_points_as_gaussians(new_points, new_colors)
    after = set(model.gaussian_uid.cpu().tolist())
    assert len(after) == 5
    assert before.issubset(after)
    # See test_bootstrap_assigns_unique_sequential_uids for why this isn't a
    # hardcoded count: _make_model's bootstrap already advances the counter
    # via two _reset_running_state() calls before any insertion happens.
    assert model._next_gaussian_uid == max(after) + 1


@pytest.mark.cuda
def test_prune_keeps_surviving_uids_aligned():
    model = _make_model(n_points=4)
    uids_before = model.gaussian_uid.clone()
    mask = torch.tensor([True, False, True, False], device="cuda")  # prune indices 0, 2
    model.prune_points(mask)
    assert model.gaussian_uid.shape[0] == 2
    assert torch.equal(model.gaussian_uid, uids_before[~mask])


@pytest.mark.cuda
def test_uid_survives_capture_restore_roundtrip():
    model = _make_model(n_points=3)
    ckpt = model.capture()
    restored = GsplatGaussianModel(sh_degree=0)
    restored.restore(ckpt, _make_training_args())
    assert torch.equal(restored.gaussian_uid.cpu(), model.gaussian_uid.cpu())
    assert restored._next_gaussian_uid == model._next_gaussian_uid
