"""Wave 3, PR W3-12: minimal, direct coverage of ``EagerRuntime``.

Deliverable 1 ("eager fallback is always available") was previously
claimed MET on the class's mere existence, with no test instantiating it
directly (every other test exercises eager behavior only indirectly, via
``TorchAcceleratorGraphRuntime``'s fallback paths). This closes that gap.

CPU-only: ``EagerRuntime`` has no CUDA/HIP dependency at all -- it just
calls the given callable every time, which is the entire point of it
being the always-available default.
"""

from __future__ import annotations

from eligsir.optimization.gaussian.graph_runtime.runtime import EagerRuntime
from eligsir.optimization.gaussian.graph_runtime.signature import GraphSignature


def _sig() -> GraphSignature:
    return GraphSignature(
        device_backend="cpu",
        capacity_bucket=1,
        active_shape=(1,),
        resolution_bucket=(0, 0),
        sh_degree=0,
        renderer_mode="none",
        packed_mode=False,
        sparse_grad_mode=False,
        optimizer_mode="none",
        loss_policy_id="test",
        gradient_policy_id="test",
    )


def test_eager_runtime_calls_step_fn_every_time():
    runtime = EagerRuntime()
    sig = _sig()
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1

    for _ in range(5):
        outcome = runtime.run_step(sig, step_fn)
        assert outcome.used_graph is False
        assert outcome.first_capture is False
        assert outcome.fallback_reason is None
        assert outcome.signature == sig

    assert calls["n"] == 5


def test_eager_runtime_ignores_signature_changes():
    """Unlike TorchAcceleratorGraphRuntime, EagerRuntime has no capture
    state to invalidate -- a changing signature must not affect behavior
    at all (it always just calls step_fn)."""
    runtime = EagerRuntime()
    calls = {"n": 0}

    def step_fn():
        calls["n"] += 1

    for capacity in (1, 2, 4, 8):
        sig = GraphSignature(
            device_backend="cpu",
            capacity_bucket=capacity,
            active_shape=(capacity,),
            resolution_bucket=(0, 0),
            sh_degree=0,
            renderer_mode="none",
            packed_mode=False,
            sparse_grad_mode=False,
            optimizer_mode="none",
            loss_policy_id="test",
            gradient_policy_id="test",
        )
        runtime.run_step(sig, step_fn)

    assert calls["n"] == 4


def test_eager_runtime_propagates_step_fn_exceptions():
    """EagerRuntime does no fallback of its own -- if step_fn raises,
    that's the caller's problem, not something to swallow. (Only
    TorchAcceleratorGraphRuntime has fallback semantics to test, and
    those are covered in test_w3_12_graph_runtime_fallback_safety.py.)
    """
    runtime = EagerRuntime()
    sig = _sig()

    def flaky():
        raise RuntimeError("boom")

    try:
        runtime.run_step(sig, flaky)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
