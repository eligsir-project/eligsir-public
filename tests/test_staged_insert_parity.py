"""CPU parity test for P3 staged Gaussian insertion (perf/p3-amortized-growth).

Verifies that running a sequence of depth-insertions through the staged path
(``staged_gaussian_insert=True`` + a begin/flush session) produces a model that is
bit-identical to the legacy per-insert cat path:

  - identical final Gaussian count,
  - identical ``params[name]`` tensors (allclose),
  - identical Adam exp_avg / exp_avg_sq for the inserted rows (zeros),
  - identical contents for every ``_STREAMING_BUFFER_NAMES`` buffer.

Runs entirely on CPU. ``GsplatGaussianModel`` hard-codes ``device="cuda"`` in its
tensor constructors, so we install a small shim that rewrites the ``cuda`` device to
``cpu`` for the duration of each model build. This exercises the real model code
(``add_points_as_gaussians`` / ``cat_tensors_to_optimizer`` / staging) without a GPU.
"""

import contextlib

import pytest
import torch
from torch import nn

from scene.gsplat_model import GsplatGaussianModel


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
        def inner(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _to_cpu_device(kwargs["device"])
            return fn(*args, **kwargs)
        return inner

    def to_cpu(self, *args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _to_cpu_device(kwargs["device"])
        new_args = []
        for a in args:
            if isinstance(a, (str, torch.device)):
                try:
                    new_args.append(_to_cpu_device(a))
                    continue
                except (RuntimeError, TypeError):
                    pass
            new_args.append(a)
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
def _build_model(n_init: int, seed: int = 0):
    """Build a small GsplatGaussianModel with Adam optimizers, on CPU."""
    torch.manual_seed(seed)
    with force_cpu():
        g = GsplatGaussianModel(sh_degree=1)  # num_sh = 4 -> shN has 3 entries
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
        # Plain Adam optimizers, one group per param (mirrors training_setup).
        g.optimizers = {}
        for name in g.params:
            grp = {"params": [g.params[name]], "lr": 0.0, "name": name}
            g.optimizers[name] = torch.optim.Adam([grp], lr=0.0, eps=1e-15)
        from scene.gsplat_model import OptimizerDictProxy
        g.optimizer = OptimizerDictProxy(g.optimizers)
        g._reset_running_state()
        # Materialize Adam momentum state so the inserted rows must be zero-extended
        # (a non-trivial parity check on exp_avg / exp_avg_sq).
        for name, p in g.params.items():
            p.grad = torch.ones_like(p)
        for opt in g.optimizers.values():
            opt.step()
        for p in g.params.values():
            p.grad = None
    return g


def _make_insert(n: int, seed: int):
    """Deterministic insert payload (points, colors, scales, quats, normals, ...)."""
    gen = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, 3, generator=gen)
    cols = torch.rand(n, 3, generator=gen)
    scales = torch.randn(n, 3, generator=gen)
    quats = torch.randn(n, 4, generator=gen)
    quats = quats / quats.norm(dim=1, keepdim=True)
    normals = torch.randn(n, 3, generator=gen)
    confidence = torch.rand(n, generator=gen)
    source = torch.zeros(n, dtype=torch.int16)
    return dict(
        points=pts, colors=cols, scales=scales, rotations=quats,
        normals=normals, confidence=confidence, source=source,
        is_provisional=True, birth_frame=7, birth_iter=11,
    )


def _run_inserts(g, inserts, staged: bool):
    with force_cpu():
        if staged:
            g.begin_staged_inserts()
        for ins in inserts:
            g.add_points_as_gaussians(**ins)
        if staged:
            g.flush_staged_inserts()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_staged_insert_matches_per_frame_cat():
    n_init = 12
    insert_sizes = [5, 3, 8, 1, 6]
    inserts = [_make_insert(n, seed=100 + i) for i, n in enumerate(insert_sizes)]

    g_base = _build_model(n_init, seed=0)
    g_base.staged_gaussian_insert = False
    _run_inserts(g_base, inserts, staged=False)

    g_stag = _build_model(n_init, seed=0)
    g_stag.staged_gaussian_insert = True
    _run_inserts(g_stag, inserts, staged=True)

    total = n_init + sum(insert_sizes)

    # 1) identical final count
    assert g_base.params["means"].shape[0] == total
    assert g_stag.params["means"].shape[0] == total

    # 2) identical params
    for name in g_base.params:
        a = g_base.params[name].detach()
        b = g_stag.params[name].detach()
        assert a.shape == b.shape, name
        assert torch.allclose(a, b, atol=0, rtol=0), f"param mismatch: {name}"

    # 3) identical Adam momentum, and inserted rows are zero
    for name in g_base.params:
        sa = g_base.optimizers[name].state[g_base.params[name]]
        sb = g_stag.optimizers[name].state[g_stag.params[name]]
        for key in ("exp_avg", "exp_avg_sq"):
            ta, tb = sa[key], sb[key]
            assert ta.shape[0] == total and tb.shape[0] == total, (name, key)
            assert torch.allclose(ta, tb, atol=0, rtol=0), f"momentum mismatch: {name}/{key}"
            inserted = tb[n_init:]
            assert torch.count_nonzero(inserted) == 0, f"inserted momentum not zero: {name}/{key}"

    # 4) identical streaming buffers (and length-consistent).
    for name in GsplatGaussianModel._STREAMING_BUFFER_NAMES:
        ba = getattr(g_base, name, None)
        bb = getattr(g_stag, name, None)
        assert (ba is None) == (bb is None), f"buffer presence mismatch: {name}"
        if ba is None:
            continue
        assert ba.shape == bb.shape, f"buffer shape mismatch: {name}"
        assert ba.shape[0] == total, f"buffer length mismatch: {name}"
        if ba.dtype.is_floating_point:
            assert torch.equal(torch.nan_to_num(ba), torch.nan_to_num(bb)), f"buffer mismatch: {name}"
        else:
            assert torch.equal(ba, bb), f"buffer mismatch: {name}"


def test_staged_buffer_presence_parity_with_consensus_buffers():
    """When optional consensus buffers are allocated, both paths extend them alike."""
    n_init = 9
    insert_sizes = [4, 2, 5]
    inserts = [_make_insert(n, seed=200 + i) for i, n in enumerate(insert_sizes)]

    def build_with_consensus(seed_staged):
        g = _build_model(n_init, seed=0)
        with force_cpu():
            g.seen_count = torch.zeros(n_init, dtype=torch.int32)
            g.consensus_support_count = torch.zeros(n_init, dtype=torch.int32)
            g.consensus_conflict_count = torch.zeros(n_init, dtype=torch.int32)
            g.consensus_last_seen_uid = torch.full((n_init,), -1, dtype=torch.int32)
            g.consensus_last_voter_center = torch.full((n_init, 3), float("nan"))
        g.staged_gaussian_insert = seed_staged
        return g

    g_base = build_with_consensus(False)
    _run_inserts(g_base, inserts, staged=False)
    g_stag = build_with_consensus(True)
    _run_inserts(g_stag, inserts, staged=True)

    total = n_init + sum(insert_sizes)
    for name in ("seen_count", "consensus_support_count", "consensus_conflict_count",
                 "consensus_last_seen_uid", "consensus_last_voter_center"):
        ba, bb = getattr(g_base, name), getattr(g_stag, name)
        assert ba.shape[0] == total and bb.shape[0] == total, name
        assert torch.equal(torch.nan_to_num(ba.float()), torch.nan_to_num(bb.float())), name


def test_flush_is_noop_without_staging():
    """Flush with no open session / nothing staged returns 0 and leaves model intact."""
    g = _build_model(6, seed=3)
    before = g.params["means"].shape[0]
    with force_cpu():
        assert g.flush_staged_inserts() == 0
    assert g.params["means"].shape[0] == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
