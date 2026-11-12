"""Issue #283: the exact, immutable SEED (bootstrap) cohort.

Mechanism under test: a scalar ``_seed_uid_range`` (the contiguous
``gaussian_uid`` block allocated to the bootstrap population) combined with the
EXISTING lineage buffers:

    seed = (uid in [lo, hi)) and root_uid == uid and birth_method == BOOTSTRAP

The CPU tests here drive the REAL mutation functions
(``utils.strategies.mcmc.proposal.reset_relocated_state`` /
``append_running_state``) against a CPU stand-in model, because those functions
are device-agnostic. The model-level tests (fresh bootstrap, insertion, prune,
capture/restore, PLY) need ``GsplatGaussianModel``, whose ``__init__``
allocates ``device="cuda"`` tensors eagerly, so they are GPU-gated exactly like
``tests/test_gsplat_model_lineage_init_and_restore.py``.
"""

from __future__ import annotations

import pathlib
import re
from argparse import Namespace

import numpy as np
import pytest
import torch

from eligsir.utils.gaussian_lineage import (
    BIRTH_METHOD_BOOTSTRAP,
    BIRTH_METHOD_MCMC_GROW,
    BIRTH_METHOD_SOURCE_DEPTH,
    SeedCohortUnavailable,
    seed_cohort_mask,
    seed_cohort_summary,
    seed_lineage_mask,
    seed_uid_range,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# CPU stand-in: holds exactly the buffers the real mutation helpers touch.
# --------------------------------------------------------------------------


class _CpuModel:
    """Minimal duck-type of GsplatGaussianModel for the device-agnostic helpers."""

    def __init__(self, n_seed: int, next_uid_start: int = 0):
        lo = next_uid_start
        hi = lo + n_seed
        self._xyz = torch.zeros(n_seed, 3)
        self.gaussian_uid = torch.arange(lo, hi, dtype=torch.int64)
        self.parent_uid = torch.full((n_seed,), -1, dtype=torch.int64)
        self.root_uid = self.gaussian_uid.clone()
        self.generation = torch.zeros(n_seed, dtype=torch.int32)
        self.birth_method = torch.full((n_seed,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8)
        self.last_reseed_iter = torch.full((n_seed,), -1, dtype=torch.int32)
        self.birth_iter = torch.zeros(n_seed, dtype=torch.int32)
        self.birth_frame = torch.zeros(n_seed, dtype=torch.int32)
        self.provisional = torch.zeros(n_seed, dtype=torch.bool)
        self.support_count = torch.zeros(n_seed, dtype=torch.int16)
        self.visibility_ema = torch.zeros(n_seed, 1)
        self._next_gaussian_uid = hi
        self._seed_uid_range = (lo, hi)
        self._seed_uid_range_reason = ""
        self._seed_uid_range_invalidated = False

    @property
    def get_xyz(self):
        return self._xyz

    # -- mutations mirroring the model's own row semantics -----------------

    def _row_buffers(self):
        return [
            "gaussian_uid", "parent_uid", "root_uid", "generation", "birth_method",
            "last_reseed_iter", "birth_iter", "birth_frame", "provisional",
            "support_count", "visibility_ema",
        ]

    def prune(self, kill_mask: torch.Tensor):
        keep = ~kill_mask
        self._xyz = self._xyz[keep]
        for name in self._row_buffers():
            setattr(self, name, getattr(self, name)[keep])

    def append_rows(self, n: int):
        """What add_points_as_gaussians does: append n fresh SOURCE_DEPTH roots."""
        lo = self._next_gaussian_uid
        uid = torch.arange(lo, lo + n, dtype=torch.int64)
        self._next_gaussian_uid = lo + n
        self._xyz = torch.cat([self._xyz, torch.zeros(n, 3)], dim=0)
        self.gaussian_uid = torch.cat([self.gaussian_uid, uid])
        self.parent_uid = torch.cat([self.parent_uid, torch.full((n,), -1, dtype=torch.int64)])
        self.root_uid = torch.cat([self.root_uid, uid.clone()])
        self.generation = torch.cat([self.generation, torch.zeros(n, dtype=torch.int32)])
        self.birth_method = torch.cat(
            [self.birth_method, torch.full((n,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8)]
        )
        self.last_reseed_iter = torch.cat([self.last_reseed_iter, torch.full((n,), -1, dtype=torch.int32)])
        self.birth_iter = torch.cat([self.birth_iter, torch.zeros(n, dtype=torch.int32)])
        self.birth_frame = torch.cat([self.birth_frame, torch.zeros(n, dtype=torch.int32)])
        self.provisional = torch.cat([self.provisional, torch.zeros(n, dtype=torch.bool)])
        self.support_count = torch.cat([self.support_count, torch.zeros(n, dtype=torch.int16)])
        self.visibility_ema = torch.cat([self.visibility_ema, torch.zeros(n, 1)])


# --------------------------------------------------------------------------
# 1. Base predicate
# --------------------------------------------------------------------------


def test_fresh_bootstrap_cohort_is_the_whole_population():
    m = _CpuModel(n_seed=6, next_uid_start=6)  # non-zero start: uids are NOT 0..N-1
    assert seed_uid_range(m) == (6, 12)
    assert seed_cohort_mask(m).all()
    assert seed_lineage_mask(m).all()
    assert seed_cohort_summary(m)["n_seed_alive"] == 6


def test_uid_lt_n_bootstrap_heuristic_would_be_wrong():
    """The rejected candidate fix: bootstrap uids are not [0, n_bootstrap)."""
    m = _CpuModel(n_seed=6, next_uid_start=6)
    naive = m.gaussian_uid < 6
    assert not naive.any()          # would report an EMPTY seed cohort
    assert seed_cohort_mask(m).all()  # the real cohort is all six rows


# --------------------------------------------------------------------------
# 2. Index-shift safety: append (depth insertion), prune, growth, relocation
# --------------------------------------------------------------------------


def test_cohort_survives_depth_insertion_append():
    m = _CpuModel(n_seed=4, next_uid_start=4)
    m.append_rows(5)
    mask = seed_cohort_mask(m)
    assert mask.shape[0] == 9
    assert mask[:4].all() and not mask[4:].any()


def test_cohort_survives_prune_index_shift():
    m = _CpuModel(n_seed=4, next_uid_start=4)
    m.append_rows(4)                     # rows 0-3 seed, 4-7 inserted
    kill = torch.zeros(8, dtype=torch.bool)
    kill[[0, 2, 5]] = True               # prune two seeds and one insert
    surviving_seed_uids = m.gaussian_uid[seed_cohort_mask(m) & ~kill].tolist()
    m.prune(kill)
    mask = seed_cohort_mask(m)
    assert m.gaussian_uid[mask].tolist() == surviving_seed_uids == [5, 7]
    # positional indices moved (5 and 7 are now rows 1 and 2), the cohort did not
    assert mask.nonzero().flatten().tolist() == [0, 1]


def test_mcmc_growth_clone_is_descendant_not_seed():
    """Drives the REAL append_running_state growth-lineage path."""
    from eligsir.utils.strategies.mcmc.proposal import append_running_state

    m = _CpuModel(n_seed=4, next_uid_start=4)
    old_n = 4
    added = 2
    # growth cats params first; mirror that on our stand-in
    m._xyz = torch.cat([m._xyz, torch.zeros(added, 3)], dim=0)
    append_running_state(m, old_n, added, iteration=100, parent_idxs=torch.tensor([0, 1]))

    assert m.birth_method[-added:].tolist() == [BIRTH_METHOD_MCMC_GROW] * added
    cohort = seed_cohort_mask(m)
    assert cohort[:old_n].all() and not cohort[old_n:].any()
    # descendants of seeds ARE in the lineage closure
    assert seed_lineage_mask(m).all()


def test_relocated_seed_slot_leaves_the_cohort():
    """Drives the REAL reset_relocated_state. A relocated slot KEEPS its
    gaussian_uid, so a uid-range-only predicate would silently keep counting it.
    """
    from eligsir.utils.strategies.mcmc.proposal import reset_relocated_state

    m = _CpuModel(n_seed=4, next_uid_start=4)
    m.append_rows(2)                     # donors: rows 4,5 (SOURCE_DEPTH roots)
    dead = torch.tensor([1])             # seed row 1 dies and is revived
    donor = torch.tensor([4])
    reset_relocated_state(m, dead, donor, iteration=500)

    # uid is deliberately preserved by the production code:
    assert int(m.gaussian_uid[1]) == 5
    uid_range_only = (m.gaussian_uid >= 4) & (m.gaussian_uid < 8)
    assert bool(uid_range_only[1])       # the trap: naive predicate still says "seed"
    assert not bool(seed_cohort_mask(m)[1])   # the real cohort drops it
    assert seed_cohort_mask(m).nonzero().flatten().tolist() == [0, 2, 3]
    assert seed_cohort_summary(m)["n_seed_alive"] == 3
    assert seed_cohort_summary(m)["n_seed_born"] == 4


def test_relocation_of_a_seed_onto_a_seed_donor_stays_in_lineage():
    from eligsir.utils.strategies.mcmc.proposal import reset_relocated_state

    m = _CpuModel(n_seed=4, next_uid_start=4)
    reset_relocated_state(m, torch.tensor([0]), torch.tensor([3]), iteration=7)
    assert not bool(seed_cohort_mask(m)[0])   # no longer the ORIGINAL seed
    assert bool(seed_lineage_mask(m)[0])      # but still seed-rooted


# --------------------------------------------------------------------------
# 3. Fails loudly, never silently
# --------------------------------------------------------------------------


def test_undetermined_range_raises():
    m = _CpuModel(n_seed=3)
    m._seed_uid_range = None
    m._seed_uid_range_reason = "loaded from PLY"
    with pytest.raises(SeedCohortUnavailable, match="loaded from PLY"):
        seed_cohort_mask(m)
    with pytest.raises(SeedCohortUnavailable):
        seed_lineage_mask(m)
    with pytest.raises(SeedCohortUnavailable):
        seed_cohort_summary(m)


def test_model_without_the_attribute_at_all_raises():
    """e.g. the legacy GaussianModel layout, which has no lineage buffers."""
    class _Legacy:
        get_xyz = torch.zeros(3, 3)

    with pytest.raises(SeedCohortUnavailable, match="legacy"):
        seed_cohort_mask(_Legacy())


@pytest.mark.parametrize("missing", ["gaussian_uid", "root_uid", "birth_method"])
def test_missing_lineage_buffer_raises(missing):
    m = _CpuModel(n_seed=3)
    setattr(m, missing, None)
    with pytest.raises(SeedCohortUnavailable, match=missing):
        seed_cohort_mask(m)


def test_seed_lineage_mask_raises_on_missing_root_uid():
    m = _CpuModel(n_seed=3)
    m.root_uid = None
    with pytest.raises(SeedCohortUnavailable, match="root_uid"):
        seed_lineage_mask(m)


@pytest.mark.parametrize("stale", ["gaussian_uid", "root_uid", "birth_method"])
def test_stale_length_lineage_buffer_raises(stale):
    m = _CpuModel(n_seed=3)
    setattr(m, stale, getattr(m, stale)[:2])
    with pytest.raises(SeedCohortUnavailable, match="lockstep"):
        seed_cohort_mask(m)


# --------------------------------------------------------------------------
# 4. Behaviour neutrality / purity
# --------------------------------------------------------------------------


def test_accessors_do_not_mutate_the_model():
    m = _CpuModel(n_seed=5, next_uid_start=5)
    before = {n: getattr(m, n).clone() for n in m._row_buffers()}
    seed_cohort_mask(m)
    seed_lineage_mask(m)
    seed_cohort_summary(m)
    for name, val in before.items():
        assert torch.equal(getattr(m, name), val), name
    assert m._seed_uid_range == (5, 10)


def test_seed_cohort_module_touches_no_training_state():
    """Lineage bookkeeping only: no loss / lambda / optimizer / noise anywhere."""
    import ast

    src = (REPO / "utils" / "gaussian_lineage.py").read_text()
    new_symbols = {
        "SeedCohortUnavailable", "seed_uid_range", "_require_buffer",
        "seed_cohort_mask", "seed_lineage_mask", "seed_cohort_summary",
    }
    tree = ast.parse(src)
    forbidden = {
        "loss", "optimizer", "optimizers", "noise", "backward", "grad",
        "requires_grad", "params", "opacities", "lambda_dssim", "step",
    }
    seen = set()
    for node in tree.body:
        if getattr(node, "name", None) not in new_symbols:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                seen.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                seen.add(sub.attr)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                continue  # docstrings/messages are prose, not behaviour
    assert seen, "new symbols not found -- test is not actually checking anything"
    assert not (seen & forbidden), seen & forbidden


# --------------------------------------------------------------------------
# 5. Structural guarantees read off the real model source (no GPU needed)
# --------------------------------------------------------------------------


def _gsplat_source() -> str:
    return (REPO / "scene" / "gsplat_model.py").read_text()


def test_prune_points_masks_every_buffer_the_predicate_reads():
    src = _gsplat_source()
    body = src[src.index("    def prune_points(self, mask):"):src.index("    def add_points_as_gaussians(")]
    assert "self.gaussian_uid = self.gaussian_uid[valid_points_mask]" in body
    # root_uid / birth_method are masked via the lineage loop
    assert re.search(
        r'for _name in \("parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter"\):',
        body,
    )
    assert "setattr(self, _name, _t[valid_points_mask])" in body


def test_gaussian_uid_is_never_remapped_in_place():
    """The uid-range predicate is only index-shift-safe because uids are created,
    mask-selected or concatenated -- never rewritten for an existing row."""
    hits = []
    for path in (REPO / "scene").rglob("*.py"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"self\.gaussian_uid\s*(\[[^\]]*\])?\s*=", line):
                hits.append((path.name, i, line.strip()))
    assert [h[2] for h in hits] == [
        'self.gaussian_uid = torch.empty(0, dtype=torch.int64, device="cuda")',
        "self.gaussian_uid = torch.arange(",
        "self.gaussian_uid = self.gaussian_uid[valid_points_mask]",
    ], hits


# --------------------------------------------------------------------------
# 6. Model-level (GPU-gated: GsplatGaussianModel.__init__ allocates on cuda)
# --------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _training_args():
    return Namespace(
        iterations=100, position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=20000,
        feature_lr=0.0025, feature_lr_final=0.00025, opacity_lr=0.05,
        opacity_lr_final=0.001, scaling_lr=0.005, scaling_lr_final=0.0005,
        rotation_lr=0.001, rotation_lr_final=0.0001, percent_dense=0.01,
        lambda_dssim=0.2, densification_interval=100, opacity_reset_interval=30000,
        densify_from_iter=500, densify_until_iter=15000,
        densify_grad_threshold=0.0002, random_background=False,
    )


def _make_model(n=6):
    from scene.gsplat_model import GsplatGaussianModel
    from eligsir.utils.graphics_utils import BasicPointCloud

    model = GsplatGaussianModel(sh_degree=0)
    pcd = BasicPointCloud(
        points=np.random.rand(n, 3).astype(np.float32),
        colors=np.random.rand(n, 3).astype(np.float32),
        normals=np.zeros((n, 3), dtype=np.float32),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_training_args())
    return model


@cuda_only
def test_gpu_fresh_run_range_matches_live_uids():
    """create_from_pcd + training_setup reset TWICE; the recorded range must be
    the SECOND block, which is what the live gaussian_uid buffer holds."""
    m = _make_model(n=6)
    lo, hi = seed_uid_range(m)
    assert (lo, hi) == (6, 12)          # not (0, 6): the first block is thrown away
    assert m.gaussian_uid.min().item() == lo and m.gaussian_uid.max().item() == hi - 1
    assert seed_cohort_mask(m).all()


@cuda_only
def test_gpu_cohort_after_insert_and_prune():
    m = _make_model(n=6)
    m.add_points_as_gaussians(
        torch.rand(4, 3, device="cuda"), torch.rand(4, 3, device="cuda"), birth_iter=10,
    )
    assert seed_cohort_mask(m).tolist() == [True] * 6 + [False] * 4
    kill = torch.zeros(10, dtype=torch.bool, device="cuda")
    kill[[0, 7]] = True
    m.prune_points(kill)
    assert seed_cohort_mask(m).tolist() == [True] * 5 + [False] * 3


@cuda_only
def test_gpu_capture_restore_roundtrip_preserves_range():
    m = _make_model(n=6)
    m.add_points_as_gaussians(
        torch.rand(3, 3, device="cuda"), torch.rand(3, 3, device="cuda"), birth_iter=10,
    )
    state = m.capture()
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    m2.restore(state, _training_args())
    assert seed_uid_range(m2) == seed_uid_range(m)
    assert seed_cohort_mask(m2).tolist() == seed_cohort_mask(m).tolist()


@cuda_only
def test_gpu_checkpoint_without_seed_range_is_unavailable():
    """Every checkpoint written before #283 is this case."""
    m = _make_model(n=6)
    state = m.capture()
    state.pop("seed_uid_range")
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    m2.restore(state, _training_args())
    with pytest.raises(SeedCohortUnavailable, match="predates the SEED uid range"):
        seed_cohort_mask(m2)


@cuda_only
def test_gpu_pre_lineage_checkpoint_is_unavailable():
    m = _make_model(n=6)
    state = m.capture()
    for name in ("parent_uid", "root_uid", "generation", "birth_method", "last_reseed_iter"):
        state["streaming_state"].pop(name, None)
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    m2.restore(state, _training_args())
    with pytest.raises(SeedCohortUnavailable, match="re-seeded"):
        seed_cohort_mask(m2)


@cuda_only
def test_gpu_uid_buffer_skipped_on_size_mismatch_is_unavailable():
    m = _make_model(n=6)
    state = m.capture()
    state["streaming_state"]["gaussian_uid"] = state["streaming_state"]["gaussian_uid"][:3]
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    m2.restore(state, _training_args())
    with pytest.raises(SeedCohortUnavailable):
        seed_cohort_mask(m2)


@cuda_only
def test_gpu_ply_loaded_model_is_unavailable(tmp_path):
    m = _make_model(n=6)
    path = str(tmp_path / "point_cloud.ply")
    m.save_ply(path)
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    m2.load_ply(path)
    with pytest.raises(SeedCohortUnavailable, match="PLY"):
        seed_cohort_mask(m2)


@cuda_only
def test_gpu_restore_overrides_a_previous_ply_invalidation():
    """The sticky invalidation must not outlive an authoritative restore()."""
    m = _make_model(n=6)
    state = m.capture()
    from scene.gsplat_model import GsplatGaussianModel

    m2 = GsplatGaussianModel(sh_degree=0)
    tmp = pathlib.Path(__file__).parent / "_seed_cohort_tmp.ply"
    try:
        m.save_ply(str(tmp))
        m2.load_ply(str(tmp))
        with pytest.raises(SeedCohortUnavailable):
            seed_cohort_mask(m2)
        m2.restore(state, _training_args())
        assert seed_uid_range(m2) == seed_uid_range(m)
        assert seed_cohort_mask(m2).all()
    finally:
        tmp.unlink(missing_ok=True)


@cuda_only
def test_gpu_mid_run_training_setup_rebootstrap_is_unavailable():
    """rolling_seed_application / submap_stitch_application call training_setup()
    AFTER inserting points; that reset destroys the original seed identity."""
    m = _make_model(n=6)
    m.add_points_as_gaussians(
        torch.rand(3, 3, device="cuda"), torch.rand(3, 3, device="cuda"), birth_iter=10,
    )
    m.training_setup(_training_args())
    with pytest.raises(SeedCohortUnavailable, match="re-bootstrapped"):
        seed_cohort_mask(m)
