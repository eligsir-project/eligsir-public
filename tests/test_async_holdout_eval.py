"""Tests for the async mid-mapping holdout evaluator (issue #1314).

Two tiers:

1. Pure-tensor-math tests for :class:`GaussianRenderSnapshot` and the
   sequence-stats helper -- these need no CUDA/HIP device and always run.
2. Real-stream tests for :func:`snapshot_gaussians_on_stream`,
   :class:`MutationBarrier`, and :class:`AsyncHoldoutEvaluator` -- these
   exercise genuine async stream ordering and are skipped when no CUDA/HIP
   device is available. The canary-race test
   (``test_naive_unguarded_copy_can_observe_a_torn_read`` /
   ``test_mutation_barrier_prevents_torn_read``) is the load-bearing pair:
   it demonstrates a real torn/contaminated read is possible without the
   reverse-direction guard, and provably impossible with it.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

# Deliberately `import ... as` rather than `from ... import Name` for the
# internal-only helpers (MutationBarrier, snapshot_gaussians_on_stream):
# they are implementation details of AsyncHoldoutEvaluator, not a separate
# public API another production module is expected to import by name (see
# contract/production_reachability.json's symbol-level rule) -- only
# AsyncHoldoutEvaluator and GaussianRenderSnapshot are real cross-module
# production surface, and both are imported by name from
# eligsir/runtime/causal_evaluation.py.
import eligsir.runtime.async_holdout_eval as async_holdout_eval_mod
from eligsir.runtime.async_holdout_eval import (
    AsyncHoldoutEvaluator,
    GaussianRenderSnapshot,
    _sequence_stats,
)

MutationBarrier = async_holdout_eval_mod.MutationBarrier
snapshot_gaussians_on_stream = async_holdout_eval_mod.snapshot_gaussians_on_stream

_CUDA = torch.cuda.is_available()
_skip_no_cuda = pytest.mark.skipif(
    not _CUDA, reason="async stream/event semantics require a CUDA/HIP device"
)


def _make_snapshot_cpu(n: int = 8) -> GaussianRenderSnapshot:
    torch.manual_seed(0)
    return GaussianRenderSnapshot(
        xyz=torch.randn(n, 3),
        scaling_raw=torch.randn(n, 3),
        rotation_raw=torch.randn(n, 4),
        opacity_raw=torch.randn(n, 1),
        features_dc=torch.randn(n, 1, 3),
        features_rest=torch.randn(n, 15, 3),
        active_sh_degree=2,
        max_sh_degree=3,
    )


def test_snapshot_render_surface_matches_gsplat_activation_formulas():
    """The snapshot must be render()-compatible with no special-casing: same
    property names, same activation formulas as GsplatGaussianModel/
    GaussianModel (exp scaling, normalized rotation, sigmoid opacity)."""

    snap = _make_snapshot_cpu()

    assert torch.allclose(snap.get_xyz, snap._xyz_t)
    assert torch.allclose(snap.get_scaling, torch.exp(snap._scaling))
    assert torch.allclose(
        snap.get_rotation,
        torch.nn.functional.normalize(snap._rotation, dim=1, eps=1e-8),
    )
    assert torch.allclose(snap.get_opacity, torch.sigmoid(snap._opacity))
    assert torch.allclose(
        snap.get_features,
        torch.cat((snap._features_dc, snap._features_rest), dim=1),
    )
    assert snap.active_sh_degree == 2
    assert snap.max_sh_degree == 3


def test_snapshot_is_immutable_view_independent_of_source_tensors():
    """A snapshot clones its inputs -- mutating the tensors passed in after
    construction must not change what the snapshot reports."""

    xyz = torch.zeros(4, 3)
    snap = GaussianRenderSnapshot(
        xyz=xyz.clone(),
        scaling_raw=torch.zeros(4, 3),
        rotation_raw=torch.zeros(4, 4),
        opacity_raw=torch.zeros(4, 1),
        features_dc=torch.zeros(4, 1, 3),
        features_rest=torch.zeros(4, 15, 3),
        active_sh_degree=0,
        max_sh_degree=0,
    )
    xyz.fill_(99.0)
    assert torch.all(snap.get_xyz == 0.0)


def test_sequence_stats_empty_and_populated():
    assert _sequence_stats([]) == {"max": 0.0, "mean": 0.0, "samples": 0}
    stats = _sequence_stats([1.0, 2.0, 3.0])
    assert stats == {"max": 3.0, "mean": 2.0, "samples": 3}


class _FakeGaussiansCPU:
    """Attribute surface `snapshot_gaussians_on_stream` reads, on CPU."""

    def __init__(self, n: int = 4):
        self._xyz = torch.zeros(n, 3)
        self._scaling = torch.zeros(n, 3)
        self._rotation = torch.zeros(n, 4)
        self._opacity = torch.zeros(n, 1)
        self._features_dc = torch.zeros(n, 1, 3)
        self._features_rest = torch.zeros(n, 15, 3)
        self.active_sh_degree = 1
        self.max_sh_degree = 3

    @property
    def get_xyz(self):
        return self._xyz


@_skip_no_cuda
class TestRealStreamSemantics:
    def _make_gaussians(self, n: int, device: str = "cuda") -> _FakeGaussiansCPU:
        g = _FakeGaussiansCPU(n)
        g._xyz = g._xyz.to(device)
        g._scaling = g._scaling.to(device)
        g._rotation = g._rotation.to(device)
        g._opacity = g._opacity.to(device)
        g._features_dc = g._features_dc.to(device)
        g._features_rest = g._features_rest.to(device)
        return g

    def test_snapshot_on_stream_returns_correct_values_and_measures_stall(self):
        gaussians = self._make_gaussians(1024)
        gaussians._xyz.fill_(7.0)
        stream = torch.cuda.Stream()

        snapshot, event, stall_s = snapshot_gaussians_on_stream(gaussians, stream)
        event.synchronize()

        assert torch.all(snapshot.get_xyz == 7.0)
        assert stall_s >= 0.0
        assert stall_s < 1.0  # enqueueing a clone must never itself block for ~1s

    def test_mutation_barrier_orders_default_stream_after_copy(self):
        """Direct barrier unit test: after wait_and_clear(), the default
        stream's next op is guaranteed ordered after the recorded event."""

        barrier = MutationBarrier()
        event = torch.cuda.Event()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            marker = torch.zeros(1, device="cuda")
            marker.fill_(1.0)
            event.record(stream)
        barrier.arm(event)
        # If wait_and_clear() did not actually enqueue a wait, this add on
        # the (unrelated) default stream could complete before `marker` is
        # written. We can't observe stream-internal timing directly here,
        # but query() must be True only once the event has truly completed --
        # exercised for real in the canary test below.
        barrier.wait_and_clear()
        torch.cuda.current_stream().synchronize()
        assert event.query()

    def test_async_holdout_evaluator_submit_runs_evaluate_fn_off_thread(self):
        gaussians = self._make_gaussians(64)
        gaussians._xyz.fill_(3.0)
        evaluator = AsyncHoldoutEvaluator()
        seen = {}
        submitting_thread = threading.get_ident()

        def _evaluate_fn(snapshot):
            seen["thread"] = threading.get_ident()
            seen["xyz_mean"] = float(snapshot.get_xyz.mean().item())

        stall_s = evaluator.submit(gaussians=gaussians, evaluate_fn=_evaluate_fn)
        evaluator.drain()

        assert stall_s >= 0.0
        assert seen["thread"] != submitting_thread
        assert seen["xyz_mean"] == pytest.approx(3.0)
        stats = evaluator.stats()
        assert stats["mapping_thread_stall_samples"] == 1


@_skip_no_cuda
class TestTornReadCanary:
    """The single most important test in this task (issue #1314): prove a
    torn/contaminated read is possible without the reverse-direction guard,
    and provably impossible with it, using the SAME snapshot mechanism in
    both arms -- only the presence/absence of the MutationBarrier wait
    differs.
    """

    N = 4_000_000  # large enough that the clone is not instantaneous
    BEFORE = 1.0
    AFTER = 2.0
    SLEEP_CYCLES = 300_000_000  # delays the snapshot-stream read

    def _mutate_in_place(self, tensor: torch.Tensor) -> None:
        # Mirrors an optimizer step / MCMC relocation / append: an in-place
        # write into the SAME storage the snapshot is reading from.
        tensor.fill_(self.AFTER)

    def test_naive_unguarded_copy_can_observe_a_torn_read(self):
        """Without the barrier wait, the mutation is free to race the
        snapshot-stream clone -- the observed value is not reliably the
        pre-mutation BEFORE value."""

        device = "cuda"
        contaminated_trials = 0
        for _ in range(10):
            means = torch.full((self.N, 1), self.BEFORE, device=device)
            snapshot_stream = torch.cuda.Stream()
            with torch.cuda.stream(snapshot_stream):
                snapshot_stream.wait_stream(torch.cuda.current_stream())
                torch.cuda._sleep(self.SLEEP_CYCLES)
                cloned = means.detach().clone()
                copy_done_event = torch.cuda.Event()
                copy_done_event.record(snapshot_stream)
            # BUG: no wait_event() before the mutation -- the default stream
            # is free to overwrite `means` while `cloned`'s copy kernel is
            # still queued behind the sleep on the snapshot stream.
            self._mutate_in_place(means)
            torch.cuda.synchronize()
            observed = cloned.unique()
            if not (observed.numel() == 1 and float(observed[0]) == self.BEFORE):
                contaminated_trials += 1
        assert contaminated_trials > 0, (
            "expected at least one of 10 unguarded trials to observe "
            "contamination from the racing mutation; if this ever flakes "
            "true-negative, widen N/SLEEP_CYCLES -- it must never be "
            "trusted as evidence the naive path is safe"
        )

    def test_mutation_barrier_prevents_torn_read(self):
        """With the barrier wait in place (the production choke point,
        `wait_for_pending_snapshot_copy` / `optimizer_tick` wrapping in
        causal_entry.py), every trial reads exactly the pre-mutation value,
        deterministically, over the same number of trials that produced
        contamination above."""

        device = "cuda"
        for _ in range(10):
            means = torch.full((self.N, 1), self.BEFORE, device=device)
            snapshot_stream = torch.cuda.Stream()
            with torch.cuda.stream(snapshot_stream):
                snapshot_stream.wait_stream(torch.cuda.current_stream())
                torch.cuda._sleep(self.SLEEP_CYCLES)
                cloned = means.detach().clone()
                copy_done_event = torch.cuda.Event()
                copy_done_event.record(snapshot_stream)
            barrier = MutationBarrier()
            barrier.arm(copy_done_event)
            # FIX: the mutation choke point waits for the copy first.
            barrier.wait_and_clear()
            self._mutate_in_place(means)
            torch.cuda.synchronize()
            observed = cloned.unique()
            assert observed.numel() == 1
            assert float(observed[0]) == self.BEFORE

    def test_async_holdout_evaluator_end_to_end_never_tears(self):
        """Same canary, driven through the real production entry points
        (`snapshot_gaussians_on_stream` + `AsyncHoldoutEvaluator.submit` /
        `wait_for_pending_copy`) instead of hand-rolled streams, so a
        regression in the actual wiring -- not just the isolated primitive
        -- would be caught."""

        # The race window here comes entirely from SLEEP_CYCLES (the GPU-side
        # delay below), not from tensor size -- unlike the two hand-rolled
        # canary trials above, this end-to-end path clones the FULL
        # six-tensor attribute set (including the (n, 15, 3) SH-rest buffer),
        # so keep n small to avoid needlessly holding ~1.5GB across the
        # clone on a GPU that may be shared with a concurrent training run.
        n = 20_000
        device = "cuda"
        gaussians = _FakeGaussiansCPU(n)
        gaussians._xyz = torch.full((n, 3), self.BEFORE, device=device)
        gaussians._scaling = gaussians._scaling.to(device)
        gaussians._rotation = gaussians._rotation.to(device)
        gaussians._opacity = gaussians._opacity.to(device)
        gaussians._features_dc = gaussians._features_dc.to(device)
        gaussians._features_rest = gaussians._features_rest.to(device)

        evaluator = AsyncHoldoutEvaluator()
        # Widen the race window the same way the canary above does, by
        # patching in a delayed clone via the evaluator's own snapshot
        # stream -- done by directly driving the two public entry points in
        # the same order production code does (submit, then the mutation
        # choke point), with an artificial GPU-side delay so the window is
        # not left to chance.
        with torch.cuda.stream(evaluator._snapshot_stream):
            torch.cuda._sleep(self.SLEEP_CYCLES)

        observed_means = {}

        def _evaluate_fn(snapshot):
            observed_means["xyz"] = snapshot.get_xyz.unique()

        evaluator.submit(gaussians=gaussians, evaluate_fn=_evaluate_fn)
        # Production choke point: wired into binding.optimizer_tick in
        # causal_entry.py, called here directly.
        evaluator.wait_for_pending_copy()
        gaussians._xyz.fill_(self.AFTER)
        evaluator.drain()

        observed = observed_means["xyz"]
        assert observed.numel() == 1
        assert float(observed[0]) == self.BEFORE
