"""M3/M4: lineage-buffer class invariants on ``GsplatGaussianModel``.

M3 -- the five lineage buffers (``parent_uid``/``root_uid``/``generation``/
``birth_method``/``last_reseed_iter``) must exist as empty tensors straight out
of ``__init__``, like ``gaussian_uid`` does. ``_commit_new_gaussians`` reads
``getattr(self, name).shape[0]`` with no default, so a model that never ran
``_reset_running_state`` would raise AttributeError there.

M4 -- restoring a PRE-lineage-fix checkpoint (whose ``streaming_state`` has
``gaussian_uid`` but no lineage keys) must re-seed lineage from the RESTORED
uids. Otherwise ``_reset_running_state``'s throwaway fresh ``arange`` block
leaves ``root_uid`` pointing at uids that exist nowhere in the model -- shapes
match, so nothing crashes, the data is just silently wrong.

GPU required: ``GsplatGaussianModel.__init__`` allocates ``device="cuda"``
tensors eagerly.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from scene.gsplat_model import GsplatGaussianModel
from eligsir.mapping.gaussian_belief_accumulator import (
    ensure_population_policy_e0_buffers,
)
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP
from eligsir.utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

_LINEAGE = ("parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter")


def _training_args():
    return Namespace(
        iterations=100,
        position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=20000,
        feature_lr=0.0025, feature_lr_final=0.00025,
        opacity_lr=0.05, opacity_lr_final=0.001,
        scaling_lr=0.005, scaling_lr_final=0.0005,
        rotation_lr=0.001, rotation_lr_final=0.0001,
        percent_dense=0.01, lambda_dssim=0.2,
        densification_interval=100, opacity_reset_interval=30000,
        densify_from_iter=500, densify_until_iter=15000,
        densify_grad_threshold=0.0002, random_background=False,
    )


def _make_model(n=4):
    torch.manual_seed(0)
    model = GsplatGaussianModel(sh_degree=0)
    pcd = BasicPointCloud(
        points=np.random.rand(n, 3).astype(np.float32),
        colors=np.random.rand(n, 3).astype(np.float32),
        normals=np.zeros((n, 3), dtype=np.float32),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_training_args())
    return model


# --------------------------------------------------------------------- M3


def test_init_allocates_empty_lineage_buffers():
    model = GsplatGaussianModel(sh_degree=0)
    for name in _LINEAGE:
        buf = getattr(model, name, None)
        assert buf is not None, f"{name} must exist right after __init__ (like gaussian_uid)"
        assert buf.shape[0] == 0
        assert buf.device.type == "cuda"
    assert model.parent_uid.dtype == torch.int64
    assert model.root_uid.dtype == torch.int64
    assert model.generation.dtype == torch.int32
    assert model.birth_method.dtype == torch.int8
    assert model.last_reseed_iter.dtype == torch.int32


# --------------------------------------------------------------------- M4


def _prefix_checkpoint(model, n_restore=4, uid_offset=1000):
    """A checkpoint dict shaped like a PRE-lineage-fix save: streaming_state
    carries gaussian_uid (deliberately OFFSET, so a stale root_uid derived from
    a fresh arange is detectably wrong) and no lineage keys at all.
    """
    ckpt = model.capture()
    streaming_state = {
        k: v for k, v in ckpt["streaming_state"].items() if k not in _LINEAGE
    }
    streaming_state["gaussian_uid"] = torch.arange(
        uid_offset, uid_offset + n_restore, dtype=torch.int64
    )
    ckpt["streaming_state"] = streaming_state
    ckpt.pop("next_gaussian_uid", None)
    return ckpt


def test_restore_pre_fix_checkpoint_reseeds_lineage_from_restored_uids():
    model = _make_model(4)
    ckpt = _prefix_checkpoint(model, n_restore=4, uid_offset=1000)

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())

    n = fresh.get_xyz.shape[0]
    assert n == 4
    assert torch.equal(fresh.gaussian_uid.cpu(), torch.arange(1000, 1004, dtype=torch.int64))
    # The decisive assertion: root_uid must track the RESTORED uids, not the
    # throwaway 0..N-1 arange _reset_running_state seeded during training_setup.
    assert torch.equal(fresh.root_uid.cpu(), fresh.gaussian_uid.cpu())
    assert torch.equal(fresh.parent_uid.cpu(), torch.full((n,), -1, dtype=torch.int64))
    assert torch.equal(fresh.generation.cpu(), torch.zeros(n, dtype=torch.int32))
    assert torch.equal(
        fresh.birth_method.cpu(), torch.full((n,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8)
    )
    assert torch.equal(fresh.last_reseed_iter.cpu(), torch.full((n,), -1, dtype=torch.int32))


def test_restore_clears_e0_bitmap_without_process_local_source_slot_mapping():
    """A resumed model must never reuse checkpointed bitmap slots blindly."""

    model = _make_model(4)
    ensure_population_policy_e0_buffers(model)
    model.population_policy_exposure_count.fill_(9)
    model.population_policy_support_count.fill_(8)
    model.population_policy_source_seen[:, 0] = True
    model._population_policy_source_slot_by_fingerprint["old-source"] = 0
    model._population_policy_source_identity_by_fingerprint["old-source"] = object()
    checkpoint = model.capture()

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(checkpoint, _training_args())
    ensure_population_policy_e0_buffers(fresh)

    assert torch.count_nonzero(fresh.population_policy_exposure_count) == 0
    assert torch.count_nonzero(fresh.population_policy_support_count) == 0
    assert not torch.any(fresh.population_policy_source_seen)
    assert fresh._population_policy_source_slot_by_fingerprint == {}
    assert fresh._population_policy_source_identity_by_fingerprint == {}


def test_restore_partial_lineage_checkpoint_reseeds_all_five_consistently():
    """A half-written checkpoint (some lineage keys, not all) must be treated as
    pre-fix and re-seeded across all five, not left as a stale/fresh mix."""
    model = _make_model(4)
    ckpt = _prefix_checkpoint(model, n_restore=4, uid_offset=1000)
    # Only root_uid present, and deliberately WRONG (a stale 0..N-1 block).
    ckpt["streaming_state"]["root_uid"] = torch.arange(4, dtype=torch.int64)

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())

    assert torch.equal(fresh.root_uid.cpu(), fresh.gaussian_uid.cpu())
    assert torch.equal(fresh.parent_uid.cpu(), torch.full((4,), -1, dtype=torch.int64))


def test_restore_post_fix_checkpoint_preserves_saved_lineage():
    """A checkpoint that DOES carry lineage must be restored verbatim, never
    clobbered by the pre-fix re-seed path."""
    model = _make_model(4)
    model.parent_uid = torch.tensor([-1, 0, 0, 1], dtype=torch.int64, device="cuda")
    model.root_uid = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device="cuda")
    model.generation = torch.tensor([0, 1, 1, 2], dtype=torch.int32, device="cuda")
    ckpt = model.capture()

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())

    assert torch.equal(fresh.parent_uid.cpu(), torch.tensor([-1, 0, 0, 1], dtype=torch.int64))
    assert torch.equal(fresh.root_uid.cpu(), torch.tensor([0, 0, 0, 0], dtype=torch.int64))
    assert torch.equal(fresh.generation.cpu(), torch.tensor([0, 1, 1, 2], dtype=torch.int32))


def test_restore_stale_length_lineage_checkpoint_reseeds():
    """Hardening beyond M4's letter: lineage keys all present but the WRONG
    length restore as silently-mismatched data. Treated as pre-fix."""
    model = _make_model(4)
    ckpt = model.capture()
    for name in _LINEAGE:
        ckpt["streaming_state"][name] = ckpt["streaming_state"][name][:2].clone()

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())

    n = fresh.get_xyz.shape[0]
    assert n == 4
    for name in _LINEAGE:
        assert getattr(fresh, name).shape[0] == n
    assert torch.equal(fresh.root_uid.cpu(), fresh.gaussian_uid.cpu())


# ------------------------------------------------ external-growth desync (W3/pixel10a)


def test_restore_checkpoint_grown_by_external_strategy_does_not_crash():
    """Real-world bug: an offline checkpoint whose N grew via a strategy that
    bypasses this project's lineage-aware growth helpers (e.g. gsplat's own
    DefaultStrategy clone/split ADC) has params sized to the NEW N but its
    streaming_state buffers still sized to some OLDER N -- and different
    buffers can be frozen at DIFFERENT stale sizes (mirrors the real crash:
    gaussian_uid=97862, birth_iter=30681, both vs. expected N=854830).

    restore() must never apply a stale-sized buffer; every _STREAMING_BUFFER_NAMES
    buffer present on the model afterward must be sized to the current N, and
    assert_lineage_consistent must pass cleanly.
    """
    from eligsir.utils.gaussian_lineage import assert_lineage_consistent

    model = _make_model(4)
    ckpt = model.capture()
    # Simulate params growing to N=10 via an external ADC path (params only --
    # exactly what a non-lineage-aware clone/split strategy would do).
    new_n = 10
    for name, tensor in ckpt["params"].items():
        reps = [new_n] + [1] * (tensor.dim() - 1)
        ckpt["params"][name] = tensor[:1].repeat(*reps).clone()
    # Different streaming buffers desync at different stale sizes (matches the
    # real bug: not every buffer freezes at the same point).
    ckpt["streaming_state"]["gaussian_uid"] = ckpt["streaming_state"]["gaussian_uid"][:1].clone()
    ckpt["streaming_state"]["birth_iter"] = ckpt["streaming_state"]["birth_iter"][:2].clone()
    for name in _LINEAGE:
        ckpt["streaming_state"][name] = ckpt["streaming_state"][name][:1].clone()

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())  # must not raise

    assert fresh.get_xyz.shape[0] == new_n
    for name in fresh._STREAMING_BUFFER_NAMES:
        buf = getattr(fresh, name, None)
        if buf is not None:
            assert buf.shape[0] == new_n, f"{name} not resized to {new_n}"
    assert_lineage_consistent(fresh, context="test_restore_checkpoint_grown_by_external_strategy")


def test_bootstrap_anchor_keyframe_ids_are_population_aligned():
    model = _make_model(4)

    assert model.anchor_keyframe_id.shape == (4,)
    assert torch.count_nonzero(model.anchor_keyframe_id) == 0


def test_restore_matching_size_checkpoint_still_uses_saved_values():
    """Guard against the fix being too aggressive: when the checkpoint's
    buffers DO match the current N (the common/normal case), restore() must
    still use the real saved values, not silently discard them."""
    model = _make_model(4)
    model.birth_iter = torch.tensor([5, 6, 7, 8], dtype=torch.int32, device="cuda")
    ckpt = model.capture()

    fresh = GsplatGaussianModel(sh_degree=0)
    fresh.restore(ckpt, _training_args())

    assert torch.equal(fresh.birth_iter.cpu(), torch.tensor([5, 6, 7, 8], dtype=torch.int32))
