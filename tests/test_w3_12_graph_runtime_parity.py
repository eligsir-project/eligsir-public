"""Wave 3, PR W3-12: deliverable 6 ("graph mode must produce parity within
the declared numerical tolerance") and deliverable 4 ("capture invalidation
rules are documented and tested"), exercised at the ``GraphRuntime`` level
against real ``SelectiveAdam``/``torch.optim.Adam`` steps.

Also demonstrates the Subagent-B "static buffer pointer stability" finding
documented in ``runtime.py``: ``zero_grad(set_to_none=True)`` breaks graph
replay parity (a real, reproduced bug), while the graph-safe in-place
``zero_grad_in_place`` does not.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gsplat.optimizers import SelectiveAdam  # noqa: E402

from eligsir.optimization.gaussian.graph_runtime.runtime import (  # noqa: E402
    TorchAcceleratorGraphRuntime,
    zero_grad_in_place,
)
from eligsir.optimization.gaussian.graph_runtime.signature import (  # noqa: E402
    GraphSignature,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

_DEVICE = "cuda"
_N = 1024


def _sig(**overrides) -> GraphSignature:
    base = dict(
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
    base.update(overrides)
    return GraphSignature(**base)


def _make_param_and_optimizer(seed: int):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(_N, 6, device=_DEVICE, generator=g))
    optimizer = SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
    visibility = torch.ones(_N, dtype=torch.bool, device=_DEVICE)
    return param, optimizer, visibility


def test_graph_replay_bit_exact_parity_with_eager_over_many_steps():
    """Bit-exact parity, accounting honestly for graph-capture warmup.

    ``TorchAcceleratorGraphRuntime`` runs ``step_fn`` for real
    ``warmup_iters`` extra times (standard PyTorch capture guidance,
    documented in ``runtime.py``) before the capture-region call itself
    also runs it once more for real -- ``warmup_iters + 1`` genuine
    applications happen before the first *replay*. To make a fair
    comparison against a plain eager reference without re-deriving that
    internal detail from first principles in the assertion, this test
    primes BOTH the graph-driven optimizer and the eager reference
    optimizer with that same number of *zero-gradient* applications
    first. For SelectiveAdam (no bias-correction/step-count term, verified
    by reading ``gsplat.optimizers.SelectiveAdam.step``'s source), a
    zero-gradient Adam-family update is an exact mathematical no-op from a
    zero initial (exp_avg, exp_avg_sq) state, so this priming changes
    nothing about either side's param/optimizer-state values -- it only
    establishes, honestly and without assuming it, that both sides have
    been driven through the identical number of real kernel-level step
    applications before the real-gradient comparison window starts.
    """
    seed = 3
    n_steps = 20
    warmup_iters = 3

    param_g, opt_g, vis_g = _make_param_and_optimizer(seed)
    param_e, opt_e, vis_e = _make_param_and_optimizer(seed)

    runtime = TorchAcceleratorGraphRuntime(warmup_iters=warmup_iters)
    sig = _sig()

    def step_fn_graph():
        opt_g.step(vis_g)
        zero_grad_in_place([param_g])

    def step_fn_eager():
        opt_e.step(vis_e)
        zero_grad_in_place([param_e])

    # Zero-grad priming (see docstring): triggers capture on the graph
    # side (warmup_iters real applications + 1 capture-region
    # application), matched by the same call count on the eager side.
    param_g.grad = torch.zeros(_N, 6, device=_DEVICE)
    outcome0 = runtime.run_step(sig, step_fn_graph)
    assert outcome0.first_capture is True
    for _ in range(warmup_iters + 1):
        param_e.grad = torch.zeros(_N, 6, device=_DEVICE)
        step_fn_eager()

    grad_gen = torch.Generator(device=_DEVICE).manual_seed(seed + 1)
    for _ in range(n_steps):
        grad = torch.randn(_N, 6, device=_DEVICE, generator=grad_gen) * 0.01
        # Write into the SAME static buffer object the graph captured
        # (param_g.grad's identity is stable across steps because
        # zero_grad_in_place zeros in place rather than reassigning) --
        # an in-place copy_, not a reassignment, is what makes this a
        # legitimate captured-graph input update.
        param_g.grad.copy_(grad)
        outcome = runtime.run_step(sig, step_fn_graph)
        assert outcome.used_graph is True, "expected pure replay once captured"

        param_e.grad = grad.clone()
        step_fn_eager()

    torch.cuda.synchronize()
    assert torch.equal(param_g.detach(), param_e.detach())
    assert runtime.counters["replay_count"] == n_steps
    assert runtime.counters["fallback_count"] == 0


def test_signature_change_forces_recapture_not_stale_replay():
    seed = 5
    param, optimizer, visibility = _make_param_and_optimizer(seed)
    runtime = TorchAcceleratorGraphRuntime()
    sig_a = _sig(capacity_bucket=_N)

    def step_fn():
        optimizer.step(visibility)
        zero_grad_in_place([param])

    param.grad = torch.zeros(_N, 6, device=_DEVICE)
    outcome1 = runtime.run_step(sig_a, step_fn)
    assert outcome1.first_capture is True
    sig_after_first_capture = runtime.captured_signature

    param.grad = torch.zeros(_N, 6, device=_DEVICE)
    outcome2 = runtime.run_step(sig_a, step_fn)
    assert outcome2.used_graph is True
    assert runtime.captured_signature == sig_after_first_capture

    # Now change the signature (a different capacity bucket -- exactly
    # the kind of topology change a real growth event would trigger).
    sig_b = _sig(capacity_bucket=_N * 2)
    param.grad = torch.zeros(_N, 6, device=_DEVICE)
    outcome3 = runtime.run_step(sig_b, step_fn)
    assert outcome3.first_capture is True, "signature change must force a fresh capture"
    assert runtime.captured_signature == sig_b
    assert runtime.counters["capture_attempts"] == 2


def test_set_to_none_true_breaks_replay_parity_demonstrating_the_bug():
    """Real, reproduced Subagent-B finding, and the ACTUAL verified
    mechanism (an earlier draft of this test/docstring incorrectly
    guessed "the graph reads a stale old buffer forever"; that guess was
    never checked and was wrong -- corrected after review caught that the
    test's assertions didn't actually distinguish "stale buffer" from
    "empty graph").

    ``optimizer.zero_grad(set_to_none=True)`` sets ``.grad`` to ``None``.
    ``SelectiveAdam.step`` skips any parameter whose ``.grad`` is
    ``None`` (``if param.grad is None: continue`` -- see
    ``gsplat.optimizers.SelectiveAdam.step``'s source). Since nothing
    inside the captured region ever re-populates ``.grad`` from ``None``,
    ``optimizer.step(visibility)`` records ZERO operations for the entire
    capture window -- PyTorch itself detects this and warns ("The CUDA
    Graph is empty..."). This test asserts on that warning directly
    (not just on a downstream numeric divergence, which could have other
    causes) via ``warnings.catch_warnings``, then proves every
    subsequent ``.replay()`` is consequently a genuine, silent, permanent
    no-op regardless of what new gradient is set -- and finally proves
    the graph-safe ``zero_grad_in_place`` alternative does not have
    either problem (captures a non-empty graph, replay is not a no-op).
    """
    import warnings

    seed = 9
    param, optimizer, visibility = _make_param_and_optimizer(seed)
    runtime = TorchAcceleratorGraphRuntime()
    sig = _sig()

    def step_fn_broken():
        optimizer.step(visibility)
        optimizer.zero_grad(set_to_none=True)  # the documented anti-pattern

    param.grad = torch.ones(_N, 6, device=_DEVICE) * 0.1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runtime.run_step(sig, step_fn_broken)  # capture
    assert any("CUDA Graph is empty" in str(w.message) for w in caught), (
        "expected PyTorch's own empty-graph warning -- if this no longer "
        "fires, set_to_none=True's effect on capture may have changed "
        "upstream and this test's mechanism claim should be revisited"
    )

    before = param.detach().clone()
    param.grad = torch.ones(_N, 6, device=_DEVICE) * 999.0  # a real, large, new gradient
    outcome = runtime.run_step(sig, step_fn_broken)
    torch.cuda.synchronize()
    after_first_replay = param.detach().clone()

    assert outcome.used_graph is True  # replay "succeeds" (raises nothing)...
    assert torch.equal(before, after_first_replay), (
        "expected a true no-op: an empty captured graph must not move "
        "the parameter at all, even given a large new gradient"
    )

    # Prove this is PERMANENT, not a one-off: set another large gradient
    # and replay again -- still nothing happens.
    param.grad = torch.ones(_N, 6, device=_DEVICE) * -12345.0
    runtime.run_step(sig, step_fn_broken)
    torch.cuda.synchronize()
    assert torch.equal(before, param.detach())

    # Contrast: the graph-safe alternative captures a NON-empty graph and
    # DOES apply real updates on replay.
    param2, optimizer2, visibility2 = _make_param_and_optimizer(seed)

    def step_fn_safe():
        optimizer2.step(visibility2)
        zero_grad_in_place([param2])

    param2.grad = torch.ones(_N, 6, device=_DEVICE) * 0.1
    runtime2 = TorchAcceleratorGraphRuntime()
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        runtime2.run_step(sig, step_fn_safe)
    assert not any("CUDA Graph is empty" in str(w.message) for w in caught2)

    before2 = param2.detach().clone()
    param2.grad = torch.ones(_N, 6, device=_DEVICE) * 999.0
    runtime2.run_step(sig, step_fn_safe)
    torch.cuda.synchronize()
    assert not torch.equal(before2, param2.detach()), (
        "graph-safe zero_grad_in_place path must actually apply updates "
        "on replay, unlike the set_to_none=True path above"
    )
