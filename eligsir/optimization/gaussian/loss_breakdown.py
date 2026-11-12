"""Authoritative ``LossBreakdown`` definition.

Wave 3, PR W3-02, Subagent B scope. Per the plan section 7.5 file list,
``loss_breakdown.py`` is a separate file from ``contracts.py``; this module
is where the real (non-placeholder) shape lives once Subagent B's extraction
work needs one.

## Why this extends (not just re-exports) ``contracts.LossBreakdown``

``contracts.LossBreakdown`` (Subagent A, landed commit 96e21ec) is:

.. code-block:: python

    @dataclass
    class LossBreakdown:
        total: float = 0.0
        components: dict[str, float] = field(default_factory=dict)

That shape is correct for logging/JSON export but cannot carry the actual
optimization loss through ``.backward()`` — ``total`` is typed ``float``,
and a Python float has no autograd graph. The ~250-line inline loss
assembly this PR moves into ``objective.assemble_loss`` builds up a live
``torch.Tensor`` (``loss = (1-lambda)*Ll1 + lambda*(1-ssim) + ...``) that
must still be a tensor when it reaches ``loss.backward()`` in the (not yet
extracted — Subagent D's scope) optimizer-sequencing step.

This is flagged in the task brief as a legitimate parity-blocking gap in
the placeholder contract. This module extends ``LossBreakdown`` with a live
``total_tensor`` and keeps ``total`` as a detached device tensor. Python
floats are materialized only at an explicit logging/JSON boundary.
``contracts.py`` is amended
in place (not left as a stale duplicate) so there is exactly one
``LossBreakdown`` definition in the codebase; this module holds the
authoritative source and ``contracts.py`` imports it.

**Flag for the integration owner**: this is a real amendment to
``contracts.py`` (Subagent A's file), not just a consumption of it. Subagent
C (gradient policies) and Subagent D (optimizer sequencing) should not need
``total_tensor`` themselves — gradient policies read `.grad` off
``gaussians`` parameters post-``backward()``, and Subagent D's
``loss.backward()`` call should read ``loss_breakdown.total_tensor``
directly rather than reconstructing the tensor itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LossBreakdown:
    """Typed decomposition of the streaming-training loss-assembly block.

    ``components`` keys use the EXACT variable names the original inline
    code in ``train_streaming.streaming_training()`` already used for each
    term (e.g. ``"Ll1"``, ``"ssim"``, ``"depth"``, ``"free_space"``,
    ``"mature_anchor"``, ``"soft_anchor"``, ``"surface"``,
    ``"knn_smoothness"``, ``"young_isotropy"``, ``"young_opacity"``, ...),
    for traceability back to the original — see ``objective.assemble_loss``
    for the full mapping.

    ``total`` is a detached device tensor and ``total_tensor`` is the live
    tensor the inline block accumulates.  Keeping both tensors avoids a
    GPU-to-host synchronization during every optimization step.  Reporting
    code must explicitly call :meth:`materialized_total` at its logging or
    JSON boundary; the optimizer must use ``total_tensor``.

    PR #39 review fix (P1): ``components`` holds detached ``torch.Tensor``
    values, NOT Python ``float``s. ``objective.assemble_loss()`` used to
    call ``float(tensor.detach())`` on every component eagerly, mid-forward
    -- a GPU->CPU host sync on every single one, every step, whether or not
    anything actually reads the float that step. That is a new host
    synchronization the original inline ``train_streaming.py`` code never
    had at this point (it deferred all ``.item()``/``float()`` conversion
    until after ``.backward()``), violating the plan's "no new host
    synchronization in the hot step" acceptance gate. Call
    :meth:`materialized_components` at the actual reporting/logging/JSON
    boundary to opt into that sync explicitly, only when a caller actually
    needs Python floats.
    """

    total: Any = 0.0
    components: dict[str, Any] = field(default_factory=dict)
    total_tensor: Any = None  # torch.Tensor; see module docstring
    # Review fix (PR #39, P0): the original inline train_streaming.py code
    # guards backward()/optimizer.step() on `if not _placement_only and
    # _loss_finite:` -- a non-finite loss (one bad live frame) must SKIP
    # backward/step entirely, not just get logged. assemble_loss() already
    # detects this and zero_grads defensively, but a first version of the
    # extracted engine.run_step() ignored this signal and unconditionally
    # called render_and_backward()+run_post_backward_step() on a non-finite
    # tensor -- turning "skip" into "backward(NaN) -> optimizer step",
    # exactly the corruption the original guard exists to prevent. Callers
    # (GaussianOptimizationEngine.run_step()) MUST check this field and
    # skip backward/optimizer-step when False.
    loss_finite: bool = True
    # Exact number of pixels in the depth mask used by this view's depth
    # objective. The engine only promotes this to completed-work telemetry
    # when the view belongs to an accepted optimizer update.
    depth_valid_sample_count: int = 0
    # A train-frame-owned EV/WB tensor whose gradient was included in this
    # loss.  The engine applies its plain SGD update once at the accepted
    # transaction boundary; keeping it here avoids a second optimizer/owner.
    photometric_evwb_state: Any = None

    def materialized_total(self) -> float:
        """Return ``total`` as a Python float at an explicit report boundary."""
        detach = getattr(self.total, "detach", None)
        return float(detach()) if detach is not None else float(self.total)

    def materialized_components(self) -> dict[str, float]:
        """Convert ``components`` to plain Python floats, forcing a
        GPU->CPU host sync per entry.

        Call this ONLY at the point a caller actually needs floats (TB
        logging, JSON export, human-readable printing) -- never inside the
        hot forward/backward step. Values that are already plain floats
        (e.g. from a caller that pre-materialized before constructing this
        object) pass through unchanged; anything with a ``.detach()``
        method (a ``torch.Tensor``) is detached and converted.
        """
        out: dict[str, float] = {}
        for key, value in self.components.items():
            detach = getattr(value, "detach", None)
            out[key] = float(detach()) if detach is not None else float(value)
        return out
