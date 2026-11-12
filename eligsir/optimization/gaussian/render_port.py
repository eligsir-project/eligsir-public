"""Narrow render adapter for the Gaussian optimization step.

Wave 3, PR W3-02 (see
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
section 7), Subagent B scope: **move the render invocation behind a typed
adapter, verbatim**. Per plan section 7.4: "The renderer is called through a
narrow adapter; it is not internally split." This module does not
reimplement or restructure the gsplat rendering path
(``eligsir/mapping/renderer/gsplat_backend.py``'s ``render()``) — it
just wraps the single existing call site that used to live inline in
``train_streaming.streaming_training()`` (~line 3931-3932 as of this PR):

.. code-block:: python

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                         update_sh_rest=update_sh_rest,
                         render_depth=use_depth_loss)
    image = render_pkg["render"]

``render`` is re-exported from ``gaussian_renderer`` (``train_streaming.py``
imports it as ``from gaussian_renderer import render`` at line 46); the real
implementation lives in ``eligsir/mapping/renderer/gsplat_backend.py``.

## Known ``render_pkg`` keys (read from ``gsplat_backend.render``'s return
dict, lines ~355-367 as of this PR)

- ``"render"`` — rendered RGB image tensor, ``[3, H, W]``.
- ``"viewspace_points"`` — screen-space points tensor used to accumulate
  positional gradients for densification.
- ``"visibility_filter"`` — bool tensor, ``[N]``, which Gaussians were
  visible/rasterized this call.
- ``"radii"`` — per-Gaussian screen-space radii, ``[N]``.
- ``"alpha"`` — rendered alpha channel, ``[1, H, W]`` (gsplat's packed
  ``render_alphas[0]``).
- ``"rendered_depth"`` — present only when ``render_depth=True`` was
  passed; ``[1, H, W]``.

Downstream loss-assembly code (``objective.py``) accesses these exact key
names off ``RenderOutputs.payload`` (the ``render_pkg`` dict, unwrapped),
preserving full backward-compatible access.
"""

from __future__ import annotations

from typing import Any

from gaussian_renderer import render as _gsplat_render

from eligsir.optimization.gaussian.contracts import RenderOutputs


def invoke_render(
    gaussians: Any,
    viewpoint_cam: Any,
    pipe: Any,
    bg: Any,
    *,
    update_sh_rest: bool = True,
    render_depth: bool = False,
) -> RenderOutputs:
    """Thin, non-splitting adapter around ``gaussian_renderer.render``.

    Mirrors the exact positional/keyword call that used to live inline in
    ``train_streaming.streaming_training()``:

    ``render(viewpoint_cam, gaussians, pipe, bg, update_sh_rest=..., render_depth=...)``

    Returns a :class:`RenderOutputs` wrapping the unmodified ``render_pkg``
    dict as ``.payload`` — callers that need the original dict (e.g. to
    match every existing ``render_pkg["..."]`` access verbatim) should read
    ``result.payload``.
    """
    render_pkg = _gsplat_render(
        viewpoint_cam,
        gaussians,
        pipe,
        bg,
        update_sh_rest=update_sh_rest,
        render_depth=render_depth,
    )
    return RenderOutputs(payload=render_pkg)
