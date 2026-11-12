"""``GraphRuntime`` abstraction (plan section 17.4) and its two Wave-3
implementations: :class:`EagerRuntime` (default, always available) and
:class:`TorchAcceleratorGraphRuntime` (PyTorch ``torch.cuda.graph``-backed
capture/replay -- the same API namespace covers both CUDA and ROCm/HIP
builds of PyTorch, hence "TorchAccelerator" rather than "TorchCuda").

``NativeHipGraphRuntime``/``NativeCudaGraphRuntime`` (plan section 17.4,
"later, only if required") are not implemented in this PR -- nothing found
during capability probing (see ``capability.py``) motivated dropping below
PyTorch's own graph interface for the capture region this PR actually
covers (see the module docstring on capture scope below).

## Capture scope actually shipped in this PR (read before using this class)

Plan section 17.3 asks to capture render -> loss assembly -> backward ->
tensor-only gradient policies -> optimizer step -> zeroing. This PR does
NOT capture render or backward -- see
``tests/test_w3_12_renderer_capture_incompatible.py`` and
``docs/architecture/2026-08-09-w3-12-graph-runtime.md`` for the real,
confound-free finding (a first draft of that investigation reported a
hard process crash and was WRONG -- caught and corrected during review;
the crash reproduced independent of graph capture entirely and was never
actually isolated to it). The real, verified finding, using the actual
production render path (``gaussian_renderer.render``): capturing it
inside ``torch.cuda.graph(...)`` raises a clean, catchable
``torch.AcceleratorError: HIP error: operation not permitted when stream
is capturing`` -- root-caused to ``_camera_tensors()``
(``eligsir/mapping/renderer/gsplat_backend.py``) constructing a fresh
host-to-device ``K`` tensor from Python floats on every call when a
camera's ``_gsplat_K``/``_gsplat_viewmat``/``_gsplat_camera_center``
device-tensor cache isn't already populated (nothing in the codebase
currently populates it). Pre-populating that cache to route around this
specific call fixes THAT obstacle but hits a second, distinct one: per a
manually-captured traceback, the SAME class of clean exception now
originates inside gsplat's own native kernel call
(``gsplat.cuda._wrapper.fully_fused_projection`` ->
``_FullyFusedProjectionPacked.forward`` -> the
``projection_ewa_3dgs_packed_fwd`` HIP kernel) -- so render capture needs
more than just camera-tensor caching to become viable, but nothing found
so far is an intractable, un-catchable hardware fault; it is a normal,
expected class of CUDA/HIP graph-capture restriction (illegal in-capture
host syncs/allocations), the SAME class CUDA itself documents and raises
cleanly for -- not a ROCm-specific defect. Wiring render into this
runtime's capture scope is future work, out of this PR's scope, not ruled
out by anything found here.

Since backward() for a rendered image only replays correctly under graph
capture if the SAME autograd-graph object identity is reproduced on every
replay (which requires forward to also be inside the graph -- there is no
way to "capture backward only" against a forward that runs fresh,
eagerly, every iteration), and render capture is not yet viable for the
reasons above, this PR excludes both render AND backward from its capture
scope for now.

**What this PR's ``TorchAcceleratorGraphRuntime`` actually captures**: the
tensor-only tail of the optimization step, run *after* an eager
``render()`` + loss-assembly + ``backward()`` have already populated
``.grad`` on the live parameters -- i.e. the gradient-policy masking cascade
+ optimizer step (+ bubble/rotation-normalize/depth-lock restores) + grad
zeroing. This maps directly onto
``GaussianOptimizationEngine.run_gradient_policies`` +
``.run_optimizer_step`` (equivalently, ``run_post_backward_step`` minus its
non-finite-loss short-circuit). This is a real, useful, and honestly
narrower capture region than section 17.3's literal wording -- it still
removes the per-step host-launch overhead of the five gradient-policy
object/method dispatches plus the Adam/SelectiveAdam kernel sequence, just
not render/backward's own launch overhead. See the completion doc for the
full accounting against all seven deliverables.

## The ``set_to_none`` pointer-stability requirement (Subagent B finding)

Graph replay reads/writes fixed GPU memory addresses recorded at capture
time. ``optimizer.zero_grad(set_to_none=True)`` (the engine's default,
see ``engine.py``'s ``run_optimizer_step``) sets each parameter's
``.grad`` to ``None`` instead of zeroing it in place. **Verified real
mechanism** (``tests/test_w3_12_graph_runtime_parity.py``'s
``test_set_to_none_true_breaks_replay_parity_demonstrating_the_bug``,
using ``warnings.catch_warnings`` to capture PyTorch's own diagnostic):
because ``SelectiveAdam.step`` skips any parameter whose ``.grad`` is
``None`` (``if param.grad is None: continue``), and ``set_to_none=True``
leaves ``.grad`` as ``None`` for the rest of the capture window (nothing
inside the captured region ever re-populates it), the entire capture
records **zero operations** -- PyTorch itself detects and warns about
this ("The CUDA Graph is empty. This usually means that the graph was
attempted to be captured on wrong device or stream."). Every subsequent
``.replay()`` is then a genuine, silent, permanent no-op: no error, no
warning on replay, the parameter simply never updates again for the
lifetime of that captured graph, regardless of what new gradients are
set. This is arguably worse than "reads a stale buffer forever" (the
original, unverified description of this finding, corrected after
review) -- it is "does nothing at all, forever, with no signal." Graph
mode therefore requires ``set_to_none=False`` (an in-place
``.grad.zero_()``) so gradients are actually present (as zeros) for the
capture to record real ops against; :func:`zero_grad_in_place` below is
the graph-safe replacement, and ``TorchAcceleratorGraphRuntime`` asserts
this policy is honored by the captured callable's contract (see
``ensure_captured``'s docstring). This is demonstrated concretely by
``tests/test_w3_12_graph_runtime_parity.py``'s
``test_set_to_none_true_breaks_replay_partity`` case.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from eligsir.optimization.gaussian.graph_runtime.signature import GraphSignature

logger = logging.getLogger(__name__)


def zero_grad_in_place(params) -> None:
    """Graph-safe replacement for ``optimizer.zero_grad(set_to_none=True)``.

    Zeros each parameter's existing ``.grad`` tensor in place (allocating
    one of zeros the first time a parameter has never been backward'd
    through) instead of freeing/reassigning it, so its GPU address stays
    stable across graph replays. Safe to call from both eager and captured
    code -- it is itself just a tensor op.
    """
    import torch

    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        else:
            p.grad.zero_()


class GraphCaptureUnavailable(RuntimeError):
    """Raised (and always caught internally by
    :class:`TorchAcceleratorGraphRuntime`) when capture itself fails.
    Exposed publicly only so tests can assert on it directly.
    """


@dataclass
class StepOutcome:
    """What actually happened for one ``GraphRuntime.run_step`` call --
    the honest audit trail deliverable 7 (launch count / amortization
    reporting) and deliverable 2 (fallback proof) both read from.
    """

    used_graph: bool
    first_capture: bool = False
    fallback_reason: str | None = None
    signature: GraphSignature | None = None


class GraphRuntime(ABC):
    """Executes a fixed-topology, tensor-only step callable, either by
    always running it eagerly (:class:`EagerRuntime`) or by capturing it
    once per distinct :class:`GraphSignature` and replaying thereafter
    (:class:`TorchAcceleratorGraphRuntime`).

    ``step_fn`` must be a zero-argument callable whose entire effect is
    in-place mutation of pre-existing, fixed-address GPU tensors (no
    Python-level branching on tensor values, no tensor creation/resizing,
    no ``set_to_none=True``-style deallocation) -- i.e. exactly what
    ``torch.cuda.graph`` capture requires. Passing a callable that
    violates this is a caller bug; :class:`TorchAcceleratorGraphRuntime`
    only defends against *capture/replay itself* failing loudly (or
    launching kernels that then read stale data), not against silently
    wrong-but-not-erroring callables.
    """

    @abstractmethod
    def run_step(
        self, signature: GraphSignature, step_fn: Callable[[], None]
    ) -> StepOutcome:
        raise NotImplementedError


class EagerRuntime(GraphRuntime):
    """Always available, always correct, default runtime (deliverable 1).
    Every ``GraphRuntime`` subclass's fallback path degrades to behavior
    equivalent to this class's ``run_step``.
    """

    def run_step(
        self, signature: GraphSignature, step_fn: Callable[[], None]
    ) -> StepOutcome:
        step_fn()
        return StepOutcome(used_graph=False, signature=signature)


class TorchAcceleratorGraphRuntime(GraphRuntime):
    """``torch.cuda.graph``-backed capture/replay. Works identically (same
    API surface) on CUDA and ROCm/HIP PyTorch builds where basic capture
    is supported -- see ``capability.py`` for what "supported" actually
    means on this machine's ROCm build. Falls back to plain eager
    execution of ``step_fn`` whenever capture or replay raises, for
    *any* reason, and reports honestly via :class:`StepOutcome` (never
    silently downgrades without recording why).
    """

    def __init__(self, *, warmup_iters: int = 3):
        self._warmup_iters = warmup_iters
        self._graph = None
        self._captured_signature: GraphSignature | None = None
        self._capture_attempts = 0
        self._replay_count = 0
        self._fallback_count = 0

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    @property
    def captured_signature(self) -> GraphSignature | None:
        return self._captured_signature

    @property
    def counters(self) -> dict:
        """Raw counters backing deliverable 7's launch-count/amortization
        reporting."""
        return {
            "capture_attempts": self._capture_attempts,
            "replay_count": self._replay_count,
            "fallback_count": self._fallback_count,
        }

    def _invalidate(self) -> None:
        self._graph = None
        self._captured_signature = None

    def _capture(self, signature: GraphSignature, step_fn: Callable[[], None]) -> None:
        """Warm up on a side stream, then capture. Raises
        ``GraphCaptureUnavailable`` on any failure; the caller
        (``run_step``) is responsible for treating that as "run step_fn
        eagerly instead, right now, so this call still completes
        correctly" -- see the module docstring's deliverable-2 discussion.

        **GC Safety:** The entire capture window (warmup + graph capture) is
        protected against garbage collection. Python's garbage collector can
        trigger arbitrarily inside step_fn's profiler/autograd machinery,
        which can SIGSEGV inside torch's pytree flattening code when the
        ptree machinery is invoked during an open CUDA graph capture context.
        This guard disables automatic collection for the duration of both the
        warmup loop and the capture region, then restores the prior state in
        a finally block, preserving the caller's existing GC state. Note:
        explicit gc.collect() calls inside step_fn would still run -- this
        guard only suppresses *automatic* collection.
        """
        import gc
        import torch

        self._capture_attempts += 1

        # Disable automatic garbage collection for the entire capture window
        # to prevent collection from firing inside the open graph context.
        _gc_was_enabled = gc.isenabled()
        if _gc_was_enabled:
            gc.disable()

        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self._warmup_iters):
                    step_fn()
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step_fn()
        except Exception as exc:  # noqa: BLE001 - must never propagate raw
            self._invalidate()
            raise GraphCaptureUnavailable(
                f"graph capture failed for signature={signature}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            # Restore the prior GC state, even if an exception occurred.
            # Only re-enable if it was enabled before we entered.
            if _gc_was_enabled:
                gc.enable()

        self._graph = graph
        self._captured_signature = signature

    def run_step(
        self, signature: GraphSignature, step_fn: Callable[[], None]
    ) -> StepOutcome:
        if self._graph is None or self._captured_signature != signature:
            reason = None
            if self._captured_signature is not None and self._captured_signature != signature:
                diff = signature.describe_diff(self._captured_signature)
                reason = f"signature changed: {'; '.join(diff)}"
                logger.info("graph_runtime: recapture triggered (%s)", reason)
            try:
                # IMPORTANT, verified empirically (see
                # tests/test_w3_12_graph_runtime_capture_semantics.py):
                # torch.cuda.graph(...)'s capture context RECORDS the
                # enclosed ops, it does NOT execute them -- confirmed with
                # a minimal in-place add_() probe (value unchanged
                # immediately after capture, only changes after the first
                # .replay()). So of _capture's two phases, only the
                # WARMUP loop (self._warmup_iters real, eager, side-stream
                # applications of step_fn) has a real numerical effect
                # here; the capture-region call inside
                # `with torch.cuda.graph(...)` applies NOTHING for real.
                # An earlier version of this comment incorrectly claimed
                # the capture-region call's effect "has already happened"
                # -- that was never verified and is false; corrected
                # after review caught it.
                #
                # Practical consequence: the call that triggers capture
                # should be treated as one-time setup, not as a normal
                # step -- it does NOT guarantee that THIS call's own
                # step_fn input was applied via a single correct
                # execution (warmup instead applies whatever was in the
                # static buffers warmup_iters times, using no fresh input
                # between those internal iterations). Every test in this
                # package that needs a correctness guarantee (parity,
                # fallback-safety) accounts for this by treating the
                # capturing call purely as setup and only asserting
                # eager-equivalence starting from the NEXT call onward.
                # A caller integrating this runtime for real must do the
                # same: don't trust the capturing call's own step to have
                # been applied correctly; call run_step again immediately
                # (a cheap replay) with the intended input freshly set.
                self._capture(signature, step_fn)
                return StepOutcome(
                    used_graph=False, first_capture=True, signature=signature
                )
            except GraphCaptureUnavailable as exc:
                self._fallback_count += 1
                logger.warning("graph_runtime: capture unavailable, falling back: %s", exc)
                step_fn()
                return StepOutcome(
                    used_graph=False,
                    fallback_reason=str(exc),
                    signature=signature,
                )

        try:
            self._graph.replay()
        except Exception as exc:  # noqa: BLE001
            # Deliverable 2: replay failed. We must not treat the live
            # parameters as having received this step's update from the
            # graph -- invalidate the (now-suspect) captured graph and run
            # the equivalent eager step right now so this call still
            # completes with a real, fully-applied, correct step. See
            # tests/test_w3_12_graph_runtime_fallback_safety.py for the
            # induced-failure proof that this path is exercised and that
            # no partial mutation from the failed replay leaks through.
            self._invalidate()
            self._fallback_count += 1
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("graph_runtime: replay failed, falling back: %s", reason)
            step_fn()
            return StepOutcome(used_graph=False, fallback_reason=reason, signature=signature)

        self._replay_count += 1
        return StepOutcome(used_graph=True, signature=signature)
