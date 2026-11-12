"""Torch-free stand-in for the alpha tensor ``KeyframeAdmitter`` consumes.

Why this exists (issue #794, PR #927)
-------------------------------------
``tests/test_issue794_precedence_behavior.py`` and
``tests/test_issue794_admission_precedence.py`` each drive the render-overlap
gate through a stub ``render_fn`` that returns ``{"alpha": <tensor>}``. Both
stubs originally built that value with ``import torch`` *inside* the function
body, which made the dependency invisible at module scope.

The CPU smoke image has no torch. ``keyframes.py``'s render block wraps the
overlap computation in a bare ``except Exception: render_failed = True``, so the
``ModuleNotFoundError`` was swallowed: the gate silently produced no overlap,
nothing rejected the frame, and it fell through to the ``pose_delta`` owner.
Four tests asserting ``low_overlap`` / ``low_overlap_force_admit`` therefore
failed -- but ONLY in CI, and only once PR #927's commit ``875f64f6`` (moving
``precedence.py`` out of the torch-heavy admission package) let these two files
be collected in that image at all. Before that they were on the CI exclusion
list and had never run there.

Production is unaffected: it always has torch. This was a harness gap, not a
defect in the precedence contract.

What it must support
--------------------
``keyframes.py`` computes overlap as::

    float((alpha.detach() > cfg.coverage_alpha).float().mean().item())

so the object needs ``detach()``, ``__gt__``, ``float()``, ``mean()`` and
``item()``. That is the whole contract -- deliberately no more, so this cannot
drift into a general fake tensor.

It is used UNCONDITIONALLY rather than only as a no-torch fallback: a single
code path means the tests exercise identical arithmetic locally and in CI,
instead of silently testing a different thing in each. Equivalence to the real
torch path is verified by the fact that these suites passed with torch before
this module existed and still pass with it after.
"""

from __future__ import annotations

import numpy as np


class FakeAlpha:
    """Minimal duck-typed stand-in for a torch alpha tensor.

    Supports exactly the ``detach() -> __gt__ -> float() -> mean() -> item()``
    chain that ``KeyframeAdmitter`` uses to turn an alpha buffer into a scalar
    overlap fraction. Nothing else.
    """

    __slots__ = ("_a",)

    def __init__(self, values) -> None:
        self._a = np.asarray(values, dtype=np.float64)

    # keyframes.py calls .detach() before comparing.
    def detach(self) -> "FakeAlpha":
        return self

    # (alpha > threshold) -> boolean mask, kept as 0.0/1.0 floats.
    def __gt__(self, other) -> "FakeAlpha":
        return FakeAlpha((self._a > other).astype(np.float64))

    # .float() is a no-op here: the mask is already float-valued.
    def float(self) -> "FakeAlpha":
        return self

    def mean(self) -> "FakeAlpha":
        return FakeAlpha(self._a.mean())

    def item(self) -> float:
        return float(self._a)


def stub_render(overlap_value: float, n: int = 100):
    """Return a ``render_fn`` whose alpha yields ``overlap_value`` overlap.

    The alpha is ``k`` ones followed by ``n - k`` zeros with
    ``k = round(overlap_value * n)``, so
    ``(alpha > coverage_alpha).float().mean() == k / n`` for any threshold
    strictly between 0 and 1 -- which is what both callers relied on.
    """
    k = int(round(overlap_value * n))

    def _fn(cam, *a, **kw):
        return {"alpha": FakeAlpha([[1.0]] * k + [[0.0]] * (n - k))}

    return _fn
