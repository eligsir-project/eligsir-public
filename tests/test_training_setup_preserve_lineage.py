"""Issue #374: ``training_setup(preserve_lineage=True)`` must not destroy lineage.

``GsplatGaussianModel.training_setup()`` calls ``_reset_running_state()``, which
re-stamps EVERY row as a freshly-born BOOTSTRAP root and hands out a new
``gaussian_uid`` block. That is harmless on the fresh-bootstrap path, but
``rolling_seed_application`` / ``submap_stitch_application`` call
``training_setup()`` mid-run on an already-populated model purely to rebuild the
optimizers -- there it silently destroys the provenance the #283/#288 cohort
machinery is defined in terms of.

Three tests:
  1. NEGATIVE CONTROL -- default (``preserve_lineage=False``) still wipes lineage,
     i.e. the fresh-run path is unchanged by this fix.
  2. POSITIVE -- ``preserve_lineage=True`` keeps every provenance buffer verbatim,
     including non-BOOTSTRAP ``birth_method`` rows.
  3. COLLISION -- a subsequent ``add_points_as_gaussians`` after a preserved
     setup still produces globally-unique uids.

Runs entirely on CPU via the same ``device="cuda"`` -> ``cpu`` shim
``tests/test_staged_insert_parity.py`` uses (``GsplatGaussianModel`` hard-codes
the device in its tensor constructors).
"""

from __future__ import annotations

import contextlib
from argparse import Namespace

import torch
from torch import nn

from scene.gsplat_model import GsplatGaussianModel, OptimizerDictProxy
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP, BIRTH_METHOD_SOURCE_DEPTH


# --------------------------------------------------------------------------- #
# CPU device shim (mirrors tests/test_staged_insert_parity.py)
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
_PRESERVED = GsplatGaussianModel._LINEAGE_PRESERVE_NAMES


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


def _grow(g, n: int, seed: int, birth_iter: int, submap_id: int = 3):
    """Insert real, non-BOOTSTRAP rows so lineage is a genuine mix."""
    gen = torch.Generator().manual_seed(seed)
    with force_cpu():
        g._current_submap_id = submap_id
        g.add_points_as_gaussians(
            points=torch.randn(n, 3, generator=gen),
            colors=torch.rand(n, 3, generator=gen),
            scales=torch.randn(n, 3, generator=gen),
            rotations=nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=1),
            is_provisional=False, birth_frame=4, birth_iter=birth_iter,
            birth_method=BIRTH_METHOD_SOURCE_DEPTH,
        )


def _snap(g) -> dict:
    return {name: getattr(g, name).clone() for name in _PRESERVED
            if getattr(g, name, None) is not None}


# --------------------------------------------------------------------------- #
# 1. Negative control: default path unchanged (still wipes lineage)
# --------------------------------------------------------------------------- #
def test_default_training_setup_still_rebootstraps_lineage():
    g = _bootstrap_model()
    _grow(g, 4, seed=11, birth_iter=1500)
    before = _snap(g)
    n = g.get_xyz.shape[0]
    uid_before_max = int(g.gaussian_uid.max().item())

    with force_cpu():
        g.training_setup(_training_args())  # default: preserve_lineage=False

    # Every row re-stamped as a fresh BOOTSTRAP root with a brand-new uid block.
    assert torch.equal(g.birth_method, torch.full((n,), BIRTH_METHOD_BOOTSTRAP,
                                                  dtype=g.birth_method.dtype))
    assert not torch.equal(g.birth_iter, before["birth_iter"])
    assert int(g.gaussian_uid.min().item()) > uid_before_max
    assert torch.equal(g.root_uid, g.gaussian_uid)
    assert getattr(g, "_lineage_preserving_setup_count", 0) == 0
    # Issue #374 item 3: the default path must record no lineage-preserving
    # event at all, not just leave the count at zero.
    assert getattr(g, "_lineage_preserving_setup_events", []) == []


# --------------------------------------------------------------------------- #
# 2. Positive: preserve_lineage keeps provenance verbatim
# --------------------------------------------------------------------------- #
def test_preserve_lineage_keeps_every_provenance_buffer():
    g = _bootstrap_model()
    _grow(g, 4, seed=11, birth_iter=1500)
    _grow(g, 3, seed=12, birth_iter=2700, submap_id=5)
    n = g.get_xyz.shape[0]
    before = _snap(g)
    next_uid_before = int(g._next_gaussian_uid)

    # Precondition: lineage really is a mix, so the assertions below have teeth.
    assert (before["birth_method"] == BIRTH_METHOD_BOOTSTRAP).any()
    assert (before["birth_method"] != BIRTH_METHOD_BOOTSTRAP).any()
    assert len(set(before["birth_iter"].tolist())) > 1

    with force_cpu():
        g.training_setup(_training_args(), preserve_lineage=True)

    for name in _PRESERVED:
        assert torch.equal(getattr(g, name), before[name]), f"{name} not preserved"
    assert g.get_xyz.shape[0] == n
    assert int(g._next_gaussian_uid) == next_uid_before
    assert g._lineage_preserving_setup_count == 1
    # The optimizers really were rebuilt (that is why these sites call it at all).
    assert set(g.optimizers) == set(g.params)
    for name, opt in g.optimizers.items():
        assert opt.param_groups[0]["params"][0] is g.params[name]
    # Non-provenance running state is still reset, as before this fix.
    assert torch.count_nonzero(g.visibility_ema) == 0
    assert torch.equal(g.lifecycle_state, torch.ones_like(g.lifecycle_state))
    # Issue #374 item 3: an event is recorded even without an explicit
    # iteration/site (both optional, default None/"").
    assert len(g._lineage_preserving_setup_events) == 1
    ev = g._lineage_preserving_setup_events[0]
    assert ev["setup_index"] == 1
    assert ev["iteration"] is None
    assert ev["site"] is None
    assert ev["n_gaussians"] == n
    assert set(ev["restored"]) == set(_PRESERVED)
    assert ev["skipped"] == []


# --------------------------------------------------------------------------- #
# 2b. iteration/site are recorded verbatim when the caller supplies them
#     (rolling_seed_application/submap_stitch_application both do, with
#     iteration=0 -- see those files' call sites).
# --------------------------------------------------------------------------- #
def test_preserve_lineage_records_caller_supplied_iteration_and_site():
    g = _bootstrap_model()
    _grow(g, 4, seed=11, birth_iter=1500)

    with force_cpu():
        g.training_setup(
            _training_args(), preserve_lineage=True,
            lineage_site="rolling_seed:pre_global_refine", iteration=0,
        )
        g.training_setup(
            _training_args(), preserve_lineage=True,
            lineage_site="submap_stitch:pre_global_refine", iteration=0,
        )

    assert g._lineage_preserving_setup_count == 2
    events = g._lineage_preserving_setup_events
    assert [e["setup_index"] for e in events] == [1, 2]
    assert [e["iteration"] for e in events] == [0, 0]
    assert events[0]["site"] == "rolling_seed:pre_global_refine"
    assert events[1]["site"] == "submap_stitch:pre_global_refine"

    from eligsir.utils.comparison_report import lineage_preserving_setup_report_block

    block = lineage_preserving_setup_report_block(events)
    assert block["count"] == 2
    assert block["iterations"] == [0, 0]
    assert block["sites"] == [
        "rolling_seed:pre_global_refine", "submap_stitch:pre_global_refine",
    ]
    assert block["events"] == events

    # No events -> no block, so a normal run's report.json is byte-identical
    # to before this field existed (same optional-gate convention as
    # depth_insertion_report_block).
    assert lineage_preserving_setup_report_block([]) is None
    assert lineage_preserving_setup_report_block(None) is None


# --------------------------------------------------------------------------- #
# 3. Collision: later inserts still get globally-unique uids
# --------------------------------------------------------------------------- #
def test_preserve_lineage_leaves_no_uid_collisions():
    g = _bootstrap_model()
    _grow(g, 4, seed=11, birth_iter=1500)
    with force_cpu():
        g.training_setup(_training_args(), preserve_lineage=True)
    _grow(g, 5, seed=13, birth_iter=3300)

    uids = g.gaussian_uid.tolist()
    assert len(set(uids)) == len(uids), "duplicate gaussian_uid after preserved setup"
    assert int(g.gaussian_uid.max().item()) < int(g._next_gaussian_uid)
    # Parent/root pointers still resolve inside the live uid space where they are
    # roots of their own (a bootstrap row's root_uid is its own uid).
    boot = g.birth_method == BIRTH_METHOD_BOOTSTRAP
    assert torch.equal(g.root_uid[boot], g.gaussian_uid[boot])


# --------------------------------------------------------------------------- #
# 4. Atomicity: a stale-length gaussian_uid must withhold the WHOLE uid group
# --------------------------------------------------------------------------- #
def test_stale_gaussian_uid_withholds_the_whole_lineage_group():
    """Restoring root_uid/parent_uid while gaussian_uid is the reset's throwaway
    arange block would leave them pointing at uids present nowhere in the model
    -- shapes match, nothing raises. Same corruption class
    ``_ensure_lineage_after_restore`` refuses; refuse it here too."""
    g = _bootstrap_model()
    _grow(g, 4, seed=11, birth_iter=1500)
    n = g.get_xyz.shape[0]
    birth_iter_before = g.birth_iter.clone()

    # Desync gaussian_uid only (prune_points itself guards buffers with
    # `shape[0] == mask.shape[0]` precisely because this drift is possible).
    g.gaussian_uid = g.gaussian_uid[:-2].clone()

    with force_cpu():
        g.training_setup(_training_args(), preserve_lineage=True)

    assert g.gaussian_uid.shape[0] == n, "must fall back to the reset's fresh uids"
    # Fell back to the reset's internally-coherent lineage, not a mixed one.
    assert torch.equal(g.root_uid, g.gaussian_uid)
    assert torch.equal(g.parent_uid, torch.full_like(g.parent_uid, -1))
    assert torch.equal(g.birth_method,
                       torch.full((n,), BIRTH_METHOD_BOOTSTRAP, dtype=g.birth_method.dtype))
    assert int(g.gaussian_uid.max().item()) < int(g._next_gaussian_uid)
    # birth_iter/birth_frame/submap_id do not reference uid space, so they are
    # still preserved independently of the withheld group.
    assert torch.equal(g.birth_iter, birth_iter_before)
