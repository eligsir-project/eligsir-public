"""Auxiliary appearance-only update for uncertain/specular regions.

Plan section 13.5 ("Appearance-only handling") and 13.6 Subagent D. For
pixels classified ``UNKNOWN`` by
:func:`eligsir.mapping.geometry.ray_classifier.classify_rays` (ordinary
invalid depth, or flagged as material-risk by
:func:`eligsir.mapping.geometry.material_risk.material_risk_mask`), the
main render's depth/geometry supervision must not run (deliverable: no
geometry insertion, no free-space carving, no depth-geometry gradients for
UNKNOWN rays — see ``ray_semantics.py``). Appearance (color) supervision may
still run there — glass/specular regions usually *do* have a coherent
photometric appearance even though their depth return is untrustworthy.

``run_appearance_only_update`` implements this as a genuine second forward
pass with geometry parameters (``means``/``scales``/``rotation``/``opacity``)
detached via :class:`DetachedGeometryView` — SH/color parameters stay live so
only they receive gradient from this loss term. This is deliberately
independent of ``objective.py``'s main geometry-supervised loss, satisfying
deliverable 6 ("geometry and appearance supervision can be weighted
independently": ``streaming_depth_loss_weight`` vs.
``streaming_appearance_only_weight``, two unrelated CLI flags).

**Scope note (honest limitation):** plan section 13.5 also asks for
"reduced-resolution" as a cost control. What's implemented here is a
correct geometry-detached second forward restricted to the (usually sparse)
material-risk mask, throttled to run only every
``streaming_appearance_only_interval`` iterations — not an actual
sub-sampled/low-res rasterization pass (that would need renderer-level
support for a scaled intrinsics/resolution camera per call site, which is
out of this PR's scope; see the module's TODO-equivalent note in
``docs/architecture/2026-08-08-tri-state-ray-semantics.md``). The interval
throttle is the primary cost control actually enforced: "do not add a second
full-resolution render to every step" is satisfied by *not running this
every step*, rather than by rendering at lower resolution.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class DetachedGeometryView:
    """Duck-typed proxy: geometry properties detached, everything else
    (including appearance/SH parameters) delegates straight through to the
    wrapped Gaussian model.

    Works uniformly across both model layouts (``GsplatGaussianModel`` and
    the legacy ``GaussianModel``) because both already expose the same
    property names the renderer reads (``get_xyz``, ``get_scaling``,
    ``get_rotation``, ``_rotation``, ``get_opacity``) — this class only
    needs to shadow those five; ``__getattr__`` covers everything else
    (``_features_dc``, ``_features_rest``, ``active_sh_degree``,
    ``max_sh_degree``, etc.) without needing to know the concrete layout.
    """

    def __init__(self, gaussians: Any) -> None:
        self._g = gaussians

    def __getattr__(self, name: str) -> Any:
        return getattr(self._g, name)

    @property
    def get_xyz(self) -> torch.Tensor:
        """Detached Gaussian center positions (no gradients)."""
        return self._g.get_xyz.detach()

    @property
    def get_scaling(self) -> torch.Tensor:
        """Detached Gaussian scales (no gradients)."""
        return self._g.get_scaling.detach()

    @property
    def get_rotation(self) -> torch.Tensor:
        """Detached Gaussian rotations (no gradients)."""
        return self._g.get_rotation.detach()

    @property
    def _rotation(self) -> torch.Tensor:
        """Detached raw rotation quaternions (no gradients)."""
        return self._g._rotation.detach()

    @property
    def get_opacity(self) -> torch.Tensor:
        """Detached Gaussian opacities (no gradients)."""
        return self._g.get_opacity.detach()


def run_appearance_only_update(
    render_fn: Callable[..., dict],
    gaussians: Any,
    viewpoint_cam: Any,
    pipe: Any,
    bg_color: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight: float,
) -> Optional[torch.Tensor]:
    """Run one geometry-detached appearance-only forward+loss.

    Parameters
    ----------
    render_fn : the renderer's ``render(viewpoint_camera, pc, pipe,
        bg_color, ...)`` entry point (``eligsir.mapping.renderer
        .gsplat_backend.render`` in production; a stub in unit tests).
    mask : ``[H, W]`` bool tensor — restrict the loss to this region
        (material-risk / UNKNOWN pixels only). Returns ``None`` and does no
        work when the mask is empty.
    weight : loss weight, independent of the main depth-loss weight
        (deliverable 6).
    """
    if not bool(mask.any()):
        return None
    detached_pc = DetachedGeometryView(gaussians)
    render_pkg = render_fn(viewpoint_cam, detached_pc, pipe, bg_color)
    rendered = render_pkg["render"] if "render" in render_pkg else render_pkg
    m = mask.unsqueeze(0) if mask.dim() == 2 else mask
    diff = (rendered - gt_image).abs()
    masked = diff[m.expand_as(diff)]
    if masked.numel() == 0:
        return None
    return float(weight) * masked.mean()
