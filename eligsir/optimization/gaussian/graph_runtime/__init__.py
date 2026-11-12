"""CUDA/HIP graph runtime for the fixed-topology Gaussian optimization step.

Wave 3, PR W3-12 (plan section 17). See
``docs/architecture/2026-08-09-w3-12-graph-runtime.md`` for the completion
doc covering what was actually captured on this ROCm hardware, why, and
which of the two W3-11-integration paths this PR took.

Public surface:

- :class:`~eligsir.optimization.gaussian.graph_runtime.signature.GraphSignature`
  — the capture-invalidation key (plan section 17.5).
- :class:`~eligsir.optimization.gaussian.graph_runtime.runtime.GraphRuntime`
  (ABC), :class:`~...runtime.EagerRuntime` (default, always available),
  :class:`~...runtime.TorchAcceleratorGraphRuntime` (PyTorch
  ``torch.cuda.graph``-backed capture/replay for CUDA *and* ROCm/HIP —
  both go through the same ``torch.cuda`` namespace on a ROCm build).
- :mod:`~eligsir.optimization.gaussian.graph_runtime.capability` —
  ``probe_graph_capability()``, a live, non-destructive probe of what this
  device/PyTorch build actually supports (do not assume ROCm mirrors CUDA).
- :mod:`~eligsir.optimization.gaussian.graph_runtime.noise` —
  ``GraphSafeNoise``, the explicit-RNG mechanism graph replay requires
  (in-graph ``torch.randn(generator=...)`` is confirmed NOT graph-safe on
  this ROCm/PyTorch build — see the capability doc).

Graph mode is opt-in and defaults OFF everywhere in this module and every
caller of it (plan section 2.4.2). Nothing in ``train_streaming.py`` or
``engine.py`` calls into this package yet outside of tests — this PR ships
the runtime as tested, standalone infrastructure, matching W3-11's
precedent (see that PR's completion doc) of not force-fitting a live-loop
wiring change into the same PR as the new mechanism itself.
"""
