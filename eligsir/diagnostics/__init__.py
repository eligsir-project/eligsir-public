"""Runtime diagnostics that are OFF by default and cost nothing when off.

Nothing in this package may be imported eagerly by a training-loop hot path.
The convention every module here follows:

    from eligsir.diagnostics.gaussian_op_trace import get_tracer
    _tracer = get_tracer()          # module-level, resolved ONCE at startup
    ...
    if _tracer is not None:         # the ONLY cost on the off path
        ... every tensor touch lives here ...

``gaussian_op_trace`` and ``gaussian_op_conflicts`` are deliberately
**torch-free** so their tests collect and execute in the lean CPU CI image
(``Dockerfile.ci-cpu``; see ``tests/conftest.py`` and issue #423). Anything
that must touch a tensor lives in ``gaussian_op_probe``, which is imported
lazily from inside a guard and never on the off path.
"""
