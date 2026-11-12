"""Issue #908: ``lifecycle_state`` must agree with ``provisional`` at insertion time.

``lifecycle_state`` is an int8 enum (0=PROVISIONAL, 1=YOUNG, 2=MATURE, 3=FROZEN).
Every Gaussian-creation path used to pad/init it to 0 unconditionally, even for
insertions explicitly marked ``is_provisional=False`` -- while the parallel
``provisional`` bool buffer was correctly set to ``False`` for the same rows.
The two buffers then asserted opposite things about the same Gaussian until a
periodic repair pass (``train_streaming.py``'s lifecycle block,
``newly_promoted = (~provisional) & (lifecycle_state == 0)``) fixed it up.

This file exercises the two real Gaussian-creation call sites the fix touches:
  1. ``GsplatGaussianModel.add_points_as_gaussians`` (``scene/gsplat_model.py``)
  2. ``utils.strategies.mcmc.proposal.append_running_state`` (MCMC growth clones)

and asserts, for each, that ``lifecycle_state`` agrees with ``provisional``
immediately -- i.e. the invariant the repair pass exists to restore now holds
by construction, with no repair pass involved.

Runs entirely on CPU via the same ``device="cuda"`` -> ``cpu`` shim
``tests/test_training_setup_preserve_lineage.py`` / ``test_staged_insert_parity.py``
use (``GsplatGaussianModel`` hard-codes the device in its tensor constructors).
"""

from __future__ import annotations

import contextlib
from argparse import Namespace

import torch
from torch import nn

from scene.gsplat_model import GsplatGaussianModel, OptimizerDictProxy
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_SOURCE_DEPTH
from eligsir.utils.strategies.mcmc.proposal import append_running_state


# --------------------------------------------------------------------------- #
# CPU device shim
# --------------------------------------------------------------------------- #
def _to_cpu_device(device):
    if device is None:
        return device
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.device("cpu")
    return device


@contextlib.contextmanager
def force_cpu():
    """Rewrite every ``device="cuda"`` request to ``cpu`` while active."""
    orig = {name: getattr(torch, name) for name in
            ("zeros", "ones", "full", "tensor", "arange", "empty", "randn", "from_numpy")}
    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda

    def wrap_factory(fn):
        def wrapped(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _to_cpu_device(kwargs["device"])
            return fn(*args, **kwargs)
        return wrapped

    def to_cpu(self, *args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, (str, torch.device)):
                try:
                    new_args.append(_to_cpu_device(a))
                    continue
                except (RuntimeError, TypeError):
                    pass
            new_args.append(a)
        if "device" in kwargs:
            kwargs["device"] = _to_cpu_device(kwargs["device"])
        return orig_to(self, *new_args, **kwargs)

    def cuda_cpu(self, *args, **kwargs):
        return self  # already CPU

    try:
        for name, fn in orig.items():
            if name == "from_numpy":
                continue  # from_numpy takes no device kwarg
            setattr(torch, name, wrap_factory(fn))
        torch.Tensor.to = to_cpu
        torch.Tensor.cuda = cuda_cpu
        yield
    finally:
        for name, fn in orig.items():
            setattr(torch, name, fn)
        torch.Tensor.to = orig_to
        torch.Tensor.cuda = orig_cuda


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _training_args():
    return Namespace(
        iterations=100,
        position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=20000,
        feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001,
        percent_dense=0.01, optimizer_type="adam",
    )


def _bootstrap_model(n_init: int = 6, seed: int = 0) -> GsplatGaussianModel:
    """Fresh model + a real ``training_setup()``, exactly like a run's bootstrap."""
    torch.manual_seed(seed)
    with force_cpu():
        g = GsplatGaussianModel(sh_degree=1)
        g.spatial_lr_scale = 1.0
        num_sh = (g.max_sh_degree + 1) ** 2
        g.params = nn.ParameterDict({
            "means": nn.Parameter(torch.randn(n_init, 3)),
            "scales": nn.Parameter(torch.randn(n_init, 3)),
            "quats": nn.Parameter(torch.randn(n_init, 4)),
            "opacities": nn.Parameter(torch.randn(n_init)),
            "sh0": nn.Parameter(torch.randn(n_init, 1, 3)),
            "shN": nn.Parameter(torch.randn(n_init, num_sh - 1, 3)),
        })
        g.optimizers = {}
        for name in g.params:
            g.optimizers[name] = torch.optim.Adam(
                [{"params": [g.params[name]], "lr": 0.0, "name": name}], lr=0.0, eps=1e-15)
        g.optimizer = OptimizerDictProxy(g.optimizers)
        g._reset_running_state()          # create_from_pcd's reset
        g.training_setup(_training_args())  # bootstrap's reset (default path)
    return g


def _assert_no_disagreement(provisional, lifecycle_state):
    """The repair-pass invariant: no row is (~provisional) & (lifecycle_state==0).

    Scoped to the rows under test (the newly-inserted/appended slice), not a
    model's whole population: bootstrap init (``_reset_running_state``) has
    its own separate, pre-existing provisional=False/lifecycle_state=0 rows
    that are out of scope for issue #908's four named call sites (see this
    file's docstring and the PR description) -- asserting the invariant over
    the whole model would spuriously fail on that unrelated bootstrap state.
    """
    disagreement = (~provisional) & (lifecycle_state == 0)
    assert not disagreement.any(), (
        f"{int(disagreement.sum())} row(s) have provisional=False but "
        "lifecycle_state=PROVISIONAL(0) -- the two buffers disagree"
    )


# --------------------------------------------------------------------------- #
# Site 0: GsplatGaussianModel._reset_running_state (bootstrap/fallback)
# --------------------------------------------------------------------------- #
def test_bootstrap_reset_marks_non_provisional_rows_young():
    """A reset population is non-provisional, so it must not start PROVISIONAL."""
    g = _bootstrap_model()

    assert torch.equal(g.provisional, torch.zeros_like(g.provisional))
    assert torch.equal(g.lifecycle_state, torch.ones_like(g.lifecycle_state))
    _assert_no_disagreement(g.provisional, g.lifecycle_state)


def test_length_resync_pads_non_provisional_rows_young():
    """The generic streaming safety-net must not synthesize provisional rows.

    Moved (issue #1364 audit, 2026-08-31): ``_sync_streaming_state_lengths``
    now lives in ``utils/streaming_state_sync.py``, not as a bare
    ``ast.FunctionDef`` in ``train_streaming.py`` -- it was relocated because
    a canonical-side caller (``utils/strategies/mcmc/arbiter_owned.py``)
    importing it directly from ``train_streaming`` pulled the entire
    confirmed-dead legacy module graph into
    ``check_canonical_mapping_hard_cut.py``'s canonical dependency cone,
    breaking that gate. Import it directly rather than exec'ing it out of
    ``train_streaming.py``'s AST -- the whole point of the move is that it no
    longer needs to live there.
    """
    from eligsir.utils.streaming_state_sync import _sync_streaming_state_lengths

    class _Gaussians:
        get_xyz = torch.zeros(4, 3)
        provisional = torch.zeros(2, dtype=torch.bool)
        lifecycle_state = torch.ones(2, dtype=torch.int8)

    gaussians = _Gaussians()
    _sync_streaming_state_lengths(gaussians)

    assert torch.equal(gaussians.provisional, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(gaussians.lifecycle_state, torch.ones(4, dtype=torch.int8))
    _assert_no_disagreement(gaussians.provisional, gaussians.lifecycle_state)


# --------------------------------------------------------------------------- #
# Site 2: GsplatGaussianModel.add_points_as_gaussians
# --------------------------------------------------------------------------- #
def test_add_points_non_provisional_gets_young_lifecycle_state():
    g = _bootstrap_model()
    gen = torch.Generator().manual_seed(11)
    n_new = 4
    with force_cpu():
        g.add_points_as_gaussians(
            points=torch.randn(n_new, 3, generator=gen),
            colors=torch.rand(n_new, 3, generator=gen),
            scales=torch.randn(n_new, 3, generator=gen),
            rotations=nn.functional.normalize(torch.randn(n_new, 4, generator=gen), dim=1),
            is_provisional=False, birth_frame=4, birth_iter=1500,
            birth_method=BIRTH_METHOD_SOURCE_DEPTH,
        )
    new_provisional = g.provisional[-n_new:]
    new_lifecycle = g.lifecycle_state[-n_new:]
    assert torch.equal(new_provisional, torch.zeros(n_new, dtype=torch.bool))
    assert torch.equal(new_lifecycle, torch.ones(n_new, dtype=torch.int8))  # YOUNG
    _assert_no_disagreement(new_provisional, new_lifecycle)


def test_add_points_provisional_gets_provisional_lifecycle_state():
    g = _bootstrap_model()
    gen = torch.Generator().manual_seed(12)
    n_new = 3
    with force_cpu():
        g.add_points_as_gaussians(
            points=torch.randn(n_new, 3, generator=gen),
            colors=torch.rand(n_new, 3, generator=gen),
            scales=torch.randn(n_new, 3, generator=gen),
            rotations=nn.functional.normalize(torch.randn(n_new, 4, generator=gen), dim=1),
            is_provisional=True, birth_frame=4, birth_iter=1600,
            birth_method=BIRTH_METHOD_SOURCE_DEPTH,
        )
    new_provisional = g.provisional[-n_new:]
    new_lifecycle = g.lifecycle_state[-n_new:]
    assert torch.equal(new_provisional, torch.ones(n_new, dtype=torch.bool))
    assert torch.equal(new_lifecycle, torch.zeros(n_new, dtype=torch.int8))  # PROVISIONAL
    # provisional=True rows are expected/allowed to have lifecycle_state==0;
    # the invariant only forbids (~provisional) & (lifecycle_state==0).
    _assert_no_disagreement(new_provisional, new_lifecycle)


# --------------------------------------------------------------------------- #
# Site 1: utils.strategies.mcmc.proposal.append_running_state (MCMC growth)
# --------------------------------------------------------------------------- #
class _FakeGaussians:
    def __init__(self, n):
        self.gaussian_uid = torch.arange(n, dtype=torch.int64)
        self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
        self.root_uid = self.gaussian_uid.clone()
        self.generation = torch.zeros(n, dtype=torch.int32)
        self.birth_method = torch.zeros(n, dtype=torch.int8)
        self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
        self.birth_iter = torch.zeros(n, dtype=torch.int32)
        self.birth_frame = torch.zeros(n, dtype=torch.int32)
        self._next_gaussian_uid = n
        self.get_xyz = torch.zeros(n, 3)
        # provisional buffer: all False, matching a real running population
        # where MCMC-grown clones are never provisional.
        self.provisional = torch.zeros(n, dtype=torch.bool)
        # lifecycle_state buffer: pre-fix rows are a mix of YOUNG/MATURE,
        # never 0 for an already-agreeing population.
        self.lifecycle_state = torch.ones(n, dtype=torch.int8)
        for name in ("visibility_ema", "xyz_gradient_accum", "denom", "support_count",
                     "anchor_iter", "depth_conflict_count", "utility_ema"):
            setattr(self, name, torch.zeros(n))
        self.max_radii2D = torch.zeros(n)


def test_mcmc_growth_append_gets_young_not_provisional_lifecycle_state():
    g = _FakeGaussians(5)
    parent_idxs = torch.tensor([2, 4], dtype=torch.int64)  # 2 clones
    append_running_state(g, old_count=5, added_count=2, parent_idxs=parent_idxs, iteration=12300)

    assert g.lifecycle_state.shape[0] == 7
    assert g.provisional.shape[0] == 7
    new_lifecycle = g.lifecycle_state[-2:]
    new_provisional = g.provisional[-2:]
    # MCMC growth clones are never provisional -- padded provisional=False.
    assert torch.equal(new_provisional, torch.zeros(2, dtype=torch.bool))
    # ... so they must be padded YOUNG (1), not PROVISIONAL (0).
    assert torch.equal(new_lifecycle, torch.ones(2, dtype=torch.int8))
    _assert_no_disagreement(new_provisional, new_lifecycle)
