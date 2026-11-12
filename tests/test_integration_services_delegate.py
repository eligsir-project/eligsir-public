"""Real-insertion regression test for MappingCoordinator.step().

Historical note: this test used to compare MappingCoordinator.step() against
calling train_streaming.insert_gaussians_from_frame() directly, using
_collect_only=True (no real commit) and fake Gaussians. Since Wave 1
consolidation (docs/architecture/mapping_coordinator_plan.md, "Scope
correction 2"), MappingCoordinator.step() is a thin lazy-import wrapper that
calls that exact same train_streaming function -- there is no longer a
second, independent implementation to compare against, so that comparison
could no longer catch a real divergence (it was, structurally, comparing an
implementation against itself). It also only exercised the collect-only
(no-commit) path: no real add_points_as_gaussians call, no optimizer-state
extension, no occupancy update.

This test instead runs the SAME real insertion scenario, through the SAME
production entry point (MappingCoordinator.step(), _collect_only=False, a
real commit), twice against freshly built but identical GsplatGaussianModel
instances, on CPU (mirrors the CPU shim in test_staged_insert_parity.py --
GsplatGaussianModel hard-codes device="cuda" in its tensor constructors).
It asserts:

  1. determinism/reproducibility -- the two runs produce byte-identical
     results (params, Adam momentum, streaming buffers, stats, n_inserted);
  2. specific expected properties of a real commit that a regression could
     silently break -- non-zero insertion count, correct grown tensor
     shapes, Adam state populated (and zero-initialized) for the newly
     inserted rows, occupancy-hash update fired exactly once per inserted
     batch.

This does not restore independence between two implementations (there is
only one now), but it does exercise the real commit path end-to-end and
will catch a real regression in insertion math, optimizer-state extension,
or buffer bookkeeping -- unlike the old _collect_only=True comparison.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scene.gsplat_model import GsplatGaussianModel
from eligsir.mapping.coordinator import MappingCoordinator
from eligsir.utils.streaming_frames import StreamingRGBDFrame


# --------------------------------------------------------------------------- #
# CPU device shim (verbatim pattern from tests/test_staged_insert_parity.py)
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


def _build_model(n_init: int, seed: int = 0) -> GsplatGaussianModel:
    """Build a small GsplatGaussianModel with real Adam optimizers, on CPU."""
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
        g.optimizers = {}
        for name in g.params:
            grp = {"params": [g.params[name]], "lr": 0.0, "name": name}
            g.optimizers[name] = torch.optim.Adam([grp], lr=0.0, eps=1e-15)
        from scene.gsplat_model import OptimizerDictProxy
        g.optimizer = OptimizerDictProxy(g.optimizers)
        g._reset_running_state()
        # Materialize Adam momentum state so inserted rows must be zero-extended.
        for name, p in g.params.items():
            p.grad = torch.ones_like(p)
        for opt in g.optimizers.values():
            opt.step()
        for p in g.params.values():
            p.grad = None
    return g


class _RecordingStreamingScene:
    """Minimal real-ish StreamingScene stand-in: nothing is pre-occupied, and
    every occupancy-hash update from a real commit is recorded (not silently
    dropped) so the test can assert it fired."""

    def __init__(self):
        self.occupancy_hash_calls: list[int] = []
        self.surface_cell_calls: list[int] = []

    def check_occupancy(self, xyz_np, voxel_size, check_neighbors=True):
        return np.zeros(xyz_np.shape[0], dtype=bool)

    def check_surface_cells(self, xyz_np, normals_np):
        return np.zeros(xyz_np.shape[0], dtype=bool)

    def add_to_occupancy_hash(self, xyz):
        self.occupancy_hash_calls.append(int(xyz.shape[0]))

    def add_to_surface_cells(self, xyz, normals):
        self.surface_cell_calls.append(int(xyz.shape[0]))


def _make_frame(index, c2w, h, w, rgb_np, depth_np, *, fx=500.0, fy=500.0):
    return StreamingRGBDFrame(
        index=index, timestamp=float(index), rgb_path="", depth_path="dummy.png",
        c2w=c2w, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0, width=w, height=h,
        depth_scale=1000.0, _rgb_np=rgb_np, _depth_np=depth_np,
    )


class _Args(SimpleNamespace):
    def __getattr__(self, name):
        raise AttributeError(name)


def _build_args() -> _Args:
    return _Args(
        streaming_depth_stride=4,
        streaming_min_depth=0.1,
        streaming_max_depth=8.0,
        streaming_insert_voxel_size=0.02,
        streaming_max_new_gaussians_per_frame=2000,
        streaming_insert_opacity=0.05,
        streaming_depth_edge_threshold=0.02,
        streaming_max_view_angle=70.0,
        streaming_insert_knn_scale=False,
        streaming_insertion_debug=True,
        gpu_frame_prep=False,
        gpu_frame_resident=False,
        streaming_depth_filter_enabled=False,
        streaming_depth_temporal_check=False,
        # issue #408: integration_planner reads these two bare (no getattr
        # fallback). They are only *read* when streaming_depth_temporal_check
        # is on -- which it is not here -- so declaring them at the real
        # parser default (False, arguments/__init__.py:1049-1050) is
        # behaviour-neutral for this fixture and keeps it robust if a future
        # edit flips temporal_check on.
        streaming_depth_temporal_require_for_insert=False,
        streaming_depth_temporal_require_for_sensor_closer=False,
        # issue #408: bare-accessed by integration_planner since PR #394/#404
        # (_GUARDED). Its absence here was a real, currently-red regression on
        # main -- test_real_commit_* both failed with AttributeError. Pinned to
        # "footprint", the exact literal that PR removed from the getattr
        # (commit ab9834f), NOT the parser default "knn" -- "knn" takes the
        # distCUDA2 branch and would change this fixture's scale math.
        streaming_insert_scale_mode="footprint",
        streaming_surface_cell_dedup=False,
        streaming_carve_invalid_depth=False,
        streaming_insert_confidence_opacity=False,
    )


def _build_frame():
    h, w = 48, 64
    rng = np.random.default_rng(0)
    # A smooth (gently sloped) synthetic depth plane, not per-pixel noise: the
    # real insertion path rejects high local depth-gradient pixels as "edges",
    # so independent per-pixel noise (as an earlier version of this fixture
    # used) gets filtered down to zero candidates everywhere. A smooth plane
    # with a small amount of low-frequency variation gives a realistic surface
    # that survives the edge filter, matching what a real sensor sees.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth_m = 2.0 + 0.002 * xx + 0.001 * yy
    rgb_np = (rng.uniform(0, 255, size=(h, w, 3))).astype(np.uint8)
    # depth_to_meters() treats a float32 array as already-in-meters (no
    # depth_scale division applied) -- pass meters directly, not mm.
    depth_np = depth_m.astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    return _make_frame(0, c2w, h, w, rgb_np, depth_np)


def _run_real_commit(n_init: int, model_seed: int):
    """Build a fresh real model + scene and run one real coordinator.step()
    commit against it. Returns (gaussians, scene, n_inserted, stats)."""
    gaussians = _build_model(n_init, seed=model_seed)
    scene = _RecordingStreamingScene()
    args = _build_args()
    frame = _build_frame()
    coordinator = MappingCoordinator()

    torch.manual_seed(0)  # any RNG draws inside insertion (e.g. jitter) are deterministic
    with force_cpu():
        # MappingCoordinator.step() returns a typed MappingCycleResult
        # (E2 fix), not a raw (n, stats) tuple -- unpack accordingly.
        cycle_result = coordinator.step(
            gaussians, frame, args, scene, current_frame_idx=3, _collect_only=False
        )
    return gaussians, scene, cycle_result.n_inserted, cycle_result.stats


def test_real_commit_produces_expected_properties():
    """A real (non-collect-only) commit through the production entry point
    must actually insert Gaussians and extend all bookkeeping consistently."""
    n_init = 10
    gaussians, scene, n_inserted, stats = _run_real_commit(n_init, model_seed=0)

    assert isinstance(n_inserted, int)
    assert n_inserted > 0, "expected the synthetic depth frame to insert at least one Gaussian"

    total = n_init + n_inserted
    for name, p in gaussians.params.items():
        assert p.shape[0] == total, f"param {name} not grown to expected count"

    # Adam momentum extended for every param, inserted rows zero-initialized.
    for name, p in gaussians.params.items():
        state = gaussians.optimizers[name].state[p]
        for key in ("exp_avg", "exp_avg_sq"):
            t = state[key]
            assert t.shape[0] == total, f"{name}/{key} not extended to expected count"
            assert torch.count_nonzero(t[n_init:]) == 0, f"{name}/{key} inserted rows not zero"

    # Occupancy hash updated exactly once, with exactly the inserted count.
    assert scene.occupancy_hash_calls == [n_inserted]

    if stats:
        assert stats.get("inserted") == n_inserted


def test_real_commit_is_deterministic_across_runs():
    """Running the identical scenario twice through the same production path
    (MappingCoordinator.step -> insert_gaussians_from_frame, real commit)
    must produce byte-identical results."""
    g1, scene1, n1, stats1 = _run_real_commit(10, model_seed=0)
    g2, scene2, n2, stats2 = _run_real_commit(10, model_seed=0)

    assert n1 == n2
    assert stats1 == stats2
    assert scene1.occupancy_hash_calls == scene2.occupancy_hash_calls
    assert scene1.surface_cell_calls == scene2.surface_cell_calls

    assert set(g1.params.keys()) == set(g2.params.keys())
    for name in g1.params:
        a = g1.params[name].detach()
        b = g2.params[name].detach()
        assert a.shape == b.shape, name
        assert torch.allclose(a, b, atol=0, rtol=0), f"param mismatch: {name}"

    for name in g1.params:
        sa = g1.optimizers[name].state[g1.params[name]]
        sb = g2.optimizers[name].state[g2.params[name]]
        for key in ("exp_avg", "exp_avg_sq"):
            ta, tb = sa[key], sb[key]
            assert ta.shape == tb.shape, (name, key)
            assert torch.allclose(ta, tb, atol=0, rtol=0), f"momentum mismatch: {name}/{key}"

    for name in GsplatGaussianModel._STREAMING_BUFFER_NAMES:
        ba = getattr(g1, name, None)
        bb = getattr(g2, name, None)
        assert (ba is None) == (bb is None), f"buffer presence mismatch: {name}"
        if ba is None:
            continue
        assert ba.shape == bb.shape, f"buffer shape mismatch: {name}"
        if ba.dtype.is_floating_point:
            assert torch.equal(torch.nan_to_num(ba), torch.nan_to_num(bb)), f"buffer mismatch: {name}"
        else:
            assert torch.equal(ba, bb), f"buffer mismatch: {name}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
