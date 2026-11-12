"""Wave 3, PR W3-12: deliverable 2 -- "graph failure cannot corrupt the
map; it falls back before applying a partial step."

This is the single most safety-critical property in this PR. This test
does NOT accept "should be safe by construction": it induces a real
replay failure (monkeypatching the captured ``torch.cuda.CUDAGraph``
instance's own ``.replay`` method to raise) and then checks, on real GPU
tensors, that:

1. the exception never propagates out of ``run_step``;
2. ``StepOutcome.fallback_reason`` is set (the failure is reported, not
   silently swallowed);
3. **the single-step state transition produced by the failed-replay-then-
   fallback call is bit-identical to what one plain eager ``step_fn()``
   call, starting from the exact same pre-step (param, exp_avg,
   exp_avg_sq, grad) snapshot, would have produced.**

Point 3 is deliberately framed as a *single-step, snapshot-based*
comparison rather than "match a multi-step eager replay of the whole
run's history": ``TorchAcceleratorGraphRuntime``'s warmup-before-capture
(standard PyTorch capture guidance) genuinely performs extra real
Adam-family updates whose effect on (exp_avg, exp_avg_sq) depends on
whatever those buffers already contained at the time -- reconstructing an
independent multi-call eager trace that reproduces that exact history
bit-for-bit is a test-harness modeling problem, not part of the property
this PR needs to prove. Snapshotting the state immediately before the
induced failure and diffing forward from there in two parallel branches
(graph-with-injected-failure vs. plain eager) isolates exactly the
property deliverable 2 asks for, with no such modeling risk: if the
failed replay had left ANY partial/torn mutation (e.g. optimizer moments
updated but the parameter not, or a stale half-written buffer), the two
branches would diverge, because they'd no longer be applying the
identical "one full Adam update from this exact starting state" operation.

GPU-required (CUDAGraph capture needs a CUDA/HIP device).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsplat.optimizers import SelectiveAdam  # noqa: E402

from eligsir.optimization.gaussian.graph_runtime.runtime import (  # noqa: E402
    GraphRuntime,
    TorchAcceleratorGraphRuntime,
    zero_grad_in_place,
)
from eligsir.optimization.gaussian.graph_runtime.signature import (  # noqa: E402
    GraphSignature,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

_DEVICE = "cuda"
_N = 512


def _signature() -> GraphSignature:
    return GraphSignature(
        device_backend="rocm-test",
        capacity_bucket=_N,
        active_shape=(_N,),
        resolution_bucket=(0, 0),
        sh_degree=0,
        renderer_mode="none",
        packed_mode=False,
        sparse_grad_mode=True,
        optimizer_mode="selective_adam",
        loss_policy_id="test",
        gradient_policy_id="test",
    )


class _Harness:
    """A stand-in for the engine's post-backward tensor-only tail:
    gradient already populated on ``.grad`` (set directly here rather than
    via a real render+backward, matching this PR's capture-scope decision
    that backward runs eagerly outside the graph) -> SelectiveAdam step ->
    in-place grad zeroing.
    """

    def __init__(self, seed: int):
        g = torch.Generator(device=_DEVICE).manual_seed(seed)
        init = torch.randn(_N, 4, device=_DEVICE, generator=g)
        self.param = torch.nn.Parameter(init.clone())
        self.optimizer = SelectiveAdam([self.param], eps=1e-15, betas=(0.9, 0.999))
        self.visibility = torch.ones(_N, dtype=torch.bool, device=_DEVICE)
        self._grad_gen = torch.Generator(device=_DEVICE).manual_seed(seed + 1)
        # Fixed grad buffer: graph capture requires a stable address for
        # any tensor the captured kernels read, so .grad is assigned ONCE
        # here and only ever mutated in place afterward (see
        # runtime.py's "set_to_none pointer-stability" note).
        self.param.grad = torch.zeros_like(self.param)

    def set_next_grad(self) -> None:
        grad = torch.randn(_N, 4, device=_DEVICE, generator=self._grad_gen) * 0.01
        self.param.grad.copy_(grad)

    def step_fn(self) -> None:
        self.optimizer.step(self.visibility)
        zero_grad_in_place([self.param])

    def optimizer_state(self):
        state = self.optimizer.state[self.param]
        return state["exp_avg"].clone(), state["exp_avg_sq"].clone()


def _eager_step_from_snapshot(
    param_snapshot: torch.Tensor,
    exp_avg_snapshot: torch.Tensor,
    exp_avg_sq_snapshot: torch.Tensor,
    grad_snapshot: torch.Tensor,
) -> torch.Tensor:
    """Build a fresh, independent SelectiveAdam optimizer state matching
    the given snapshot exactly, apply exactly ONE plain eager step, and
    return the resulting parameter value. No graph runtime involved at
    all -- pure baseline for the single-step comparison.
    """
    param = torch.nn.Parameter(param_snapshot.clone())
    optimizer = SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
    visibility = torch.ones(_N, dtype=torch.bool, device=_DEVICE)
    # Prime the lazy state dict with the exact snapshot (matches
    # SelectiveAdam.step's own init: {"step": ..., "exp_avg": ...,
    # "exp_avg_sq": ...}).
    optimizer.state[param] = {
        "step": torch.tensor(0.0, dtype=torch.float32),
        "exp_avg": exp_avg_snapshot.clone(),
        "exp_avg_sq": exp_avg_sq_snapshot.clone(),
    }
    param.grad = grad_snapshot.clone()
    optimizer.step(visibility)
    return param.detach().clone()


def test_forced_replay_failure_falls_back_with_no_partial_mutation():
    seed = 42
    h = _Harness(seed)
    runtime: GraphRuntime = TorchAcceleratorGraphRuntime()
    sig = _signature()

    # Get to a real, warmed-up, arbitrary-but-real state: capture, then a
    # couple of successful replays. Nothing about the comparison below
    # depends on reproducing this history -- only the state immediately
    # before the induced failure matters (see module docstring).
    h.set_next_grad()
    outcome = runtime.run_step(sig, h.step_fn)
    assert outcome.first_capture is True

    for _ in range(2):
        h.set_next_grad()
        outcome = runtime.run_step(sig, h.step_fn)
        assert outcome.used_graph is True
        assert outcome.fallback_reason is None

    # Snapshot the exact pre-failure state.
    param_before = h.param.detach().clone()
    exp_avg_before, exp_avg_sq_before = h.optimizer_state()
    h.set_next_grad()
    grad_before = h.param.grad.detach().clone()

    # INDUCE a real replay failure: patch the captured graph's own
    # .replay to raise, simulating a hardware/driver-level replay fault.
    assert runtime.is_captured
    captured_graph = runtime._graph  # noqa: SLF001 - intentional white-box probe
    real_replay = captured_graph.replay

    def _boom():
        raise RuntimeError("INJECTED replay failure for W3-12 safety test")

    captured_graph.replay = _boom  # type: ignore[method-assign]

    outcome = runtime.run_step(sig, h.step_fn)  # must NOT raise
    torch.cuda.synchronize()

    assert outcome.used_graph is False
    assert outcome.fallback_reason is not None
    assert "INJECTED" in outcome.fallback_reason
    assert not runtime.is_captured, "a failed replay must invalidate the stale graph"

    after_fallback = h.param.detach().clone()

    # The critical assertion: the fallback-produced state transition is
    # bit-identical to one plain eager step from the exact same starting
    # point. The failed replay attempt (which never actually enqueued a
    # single kernel, since .replay was completely replaced) contributed
    # exactly zero partial mutation -- if it had, this equality would not
    # hold, because the two branches would no longer represent the same
    # "apply one full update from this state" operation.
    after_eager = _eager_step_from_snapshot(
        param_before, exp_avg_before, exp_avg_sq_before, grad_before
    )
    assert torch.equal(after_fallback, after_eager), (
        "post-failure parameter state does not match a single plain-eager "
        "step from the identical pre-failure snapshot -- the failed "
        "replay left a partial/torn mutation"
    )

    # Restore (avoid leaking a broken bound method on a captured graph
    # that CUDA-graph internals may try to free/inspect on teardown).
    captured_graph.replay = real_replay  # type: ignore[method-assign]

    # The runtime must recover: a fresh capture on the next call, and it
    # must not raise.
    h.set_next_grad()
    outcome = runtime.run_step(sig, h.step_fn)
    assert outcome.first_capture is True

    h.set_next_grad()
    outcome = runtime.run_step(sig, h.step_fn)
    assert outcome.used_graph is True
    assert outcome.fallback_reason is None


def test_forced_capture_failure_falls_back_and_step_still_completes():
    """Same property, but the induced failure is during the initial
    capture attempt itself (not a later replay) -- exercised by asking
    the runtime to capture a step_fn that raises mid-capture."""
    seed = 7
    h = _Harness(seed)
    runtime: GraphRuntime = TorchAcceleratorGraphRuntime()
    sig = _signature()

    calls = {"n": 0}

    def flaky_step_fn():
        calls["n"] += 1
        # Fail only during the graph-capture context itself (the 4th
        # call: 3 warmup iters succeed for real, the capture-region call
        # raises) -- torch.cuda.graph capture failing must not corrupt
        # anything either.
        if calls["n"] == 4:
            raise RuntimeError("INJECTED capture-time failure")
        h.set_next_grad()
        h.step_fn()

    outcome = runtime.run_step(sig, flaky_step_fn)
    assert outcome.used_graph is False
    assert outcome.fallback_reason is not None
    assert "INJECTED" in outcome.fallback_reason
    assert not runtime.is_captured

    # The runtime must have run step_fn eagerly as the fallback for THIS
    # call so the step still completed -- confirmed by the call counter
    # advancing past the warmup+failed-capture attempt.
    assert calls["n"] >= 5
