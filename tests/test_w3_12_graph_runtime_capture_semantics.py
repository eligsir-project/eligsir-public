"""Wave 3, PR W3-12: pins a real, verified, and initially-mis-described
correctness property of ``TorchAcceleratorGraphRuntime``'s capture
semantics -- caught by review, not found unprompted.

``torch.cuda.graph(...)``'s capture context **records** the enclosed ops;
it does not execute them. An earlier draft of ``runtime.py`` incorrectly
claimed "this call's real numerical effect has already happened by the
time we return" for the call that triggers capture. That was never
verified and is false: verified here with both a minimal in-place-add
probe (isolating the claim from any optimizer/autograd complexity) and
against the real ``TorchAcceleratorGraphRuntime.run_step`` path with
``warmup_iters=0`` (which removes the confound of warmup's OWN real,
eager, side-stream applications of ``step_fn`` -- those DO have a real
effect, and were the likely source of the original, incorrect,
unverified belief that "the capturing call" as a whole has a guaranteed
real effect).

Practical consequence pinned by this test: the call that triggers
capture must not be relied upon to have applied ITS OWN specific
step_fn input via one correct execution. Callers must treat it as setup
and call ``run_step`` again (a cheap replay) to get a real, correctly-
applied step from freshly-set input. Every other test in this package
that needs correctness (parity, fallback-safety) already does this.

GPU-required.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.graph_runtime.runtime import (  # noqa: E402
    TorchAcceleratorGraphRuntime,
)
from eligsir.optimization.gaussian.graph_runtime.signature import (  # noqa: E402
    GraphSignature,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

_DEVICE = "cuda"


def test_minimal_capture_context_records_but_does_not_execute():
    """Isolates the claim from any optimizer/autograd machinery: a bare
    in-place add inside ``torch.cuda.graph(...)`` does not change the
    tensor until ``.replay()`` is called.
    """
    x = torch.zeros(4, device=_DEVICE)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.current_stream().wait_stream(stream)  # no warmup at all

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.add_(1.0)
    torch.cuda.synchronize()

    assert torch.equal(x, torch.zeros(4, device=_DEVICE)), (
        "capture-region ops must not execute during capture itself"
    )

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.ones(4, device=_DEVICE)), (
        "the recorded op must execute on the first explicit replay"
    )


def test_capturing_run_step_call_with_zero_warmup_applies_nothing():
    """Same property through the real runtime, with warmup_iters=0 to
    remove the confound of warmup's own real applications.
    """
    x = torch.zeros(4, device=_DEVICE)

    def step_fn():
        x.add_(1.0)

    sig = GraphSignature(
        device_backend="test",
        capacity_bucket=4,
        active_shape=(4,),
        resolution_bucket=(0, 0),
        sh_degree=0,
        renderer_mode="none",
        packed_mode=False,
        sparse_grad_mode=False,
        optimizer_mode="raw_add",
        loss_policy_id="test",
        gradient_policy_id="test",
    )
    runtime = TorchAcceleratorGraphRuntime(warmup_iters=0)
    outcome = runtime.run_step(sig, step_fn)
    torch.cuda.synchronize()

    assert outcome.first_capture is True
    assert torch.equal(x, torch.zeros(4, device=_DEVICE)), (
        "the capturing call itself (zero warmup) must not have applied "
        "step_fn's effect for real -- only .replay() does"
    )

    # The very next call (matching signature) is a real replay and DOES
    # apply the recorded op.
    outcome2 = runtime.run_step(sig, step_fn)
    torch.cuda.synchronize()
    assert outcome2.used_graph is True
    assert torch.equal(x, torch.ones(4, device=_DEVICE))


def test_capturing_run_step_call_with_warmup_applies_only_warmup_effect():
    """With the default nonzero warmup, the capturing call's visible
    effect comes entirely from the warmup_iters real, eager applications
    of step_fn (which reuse whatever inputs were set before the call --
    nothing refreshes them between warmup iterations) -- NOT from the
    capture-region call inside ``torch.cuda.graph(...)`` itself, which
    still applies nothing (consistent with the two tests above).
    """
    x = torch.zeros(4, device=_DEVICE)
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1
        x.add_(1.0)

    sig = GraphSignature(
        device_backend="test",
        capacity_bucket=4,
        active_shape=(4,),
        resolution_bucket=(0, 0),
        sh_degree=0,
        renderer_mode="none",
        packed_mode=False,
        sparse_grad_mode=False,
        optimizer_mode="raw_add",
        loss_policy_id="test",
        gradient_policy_id="test",
    )
    warmup_iters = 3
    runtime = TorchAcceleratorGraphRuntime(warmup_iters=warmup_iters)
    runtime.run_step(sig, step_fn)
    torch.cuda.synchronize()

    # step_fn was called warmup_iters (real, eager) + 1 (capture-region,
    # recorded but not executed) times = warmup_iters + 1 total Python
    # calls, but x only reflects warmup_iters real increments.
    assert calls["n"] == warmup_iters + 1
    assert torch.equal(x, torch.full((4,), float(warmup_iters), device=_DEVICE))
