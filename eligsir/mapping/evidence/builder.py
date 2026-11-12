"""Assembles evidence dataclasses from values the mapper already computes
(plan section 10, Subagent B "render/gradient evidence extraction").

**This module introduces no new computation that changes what gets
rendered/backpropagated/decided.** Every field below is either:

1. read directly off an object the pipeline already builds
   (``utils.strategies.mcmc.energy_policy.GaussianEnergyTerms`` for
   per-Gaussian state — see that module's docstring: it already carries
   ``alpha``, ``support``, ``xyz_grad_norm``, ``opacity_grad_abs``,
   ``visibility``, ``age_frac``, ``geometry_confidence``,
   ``free_space_conflict``, ``render_contribution_ema``; and the
   ``render_pkg`` dict documented in
   ``eligsir/optimization/gaussian/render_port.py`` for render-space
   state — ``"render"``, ``"alpha"``, ``"rendered_depth"``, ``"radii"``), or
2. a trivial, non-differentiable derived value with no side effects
   (e.g. ``rgb_residual = (render - gt_image).detach()``,
   ``transmittance = (1 - alpha).detach()``) — a subtraction/complement of
   tensors that already exist, computed only when explicitly requested and
   never fed back into the loss, optimizer, or any mutation decision.

Every builder method is demand-driven: it takes an
:class:`eligsir.mapping.evidence.requirements.EvidenceRequirements`
and returns ``None`` immediately (no tensor touched at all) when the
relevant flags are unset. This is what makes evidence collection
near-zero-cost when nothing has requested it (plan section 10.7,
deliverable 2/4).

Nothing in this module calls into the training loop, the optimizer, or any
mutation/admission code path — it is only ever invoked *by* a caller that
already has these values in hand, never the other way around. No existing
call site in ``train_streaming.py``/``engine.py``/``energy_policy.py`` is
modified to call this module as part of this PR (see
``docs/architecture/evidence_duplicate_computations.md`` for the explicit
audit of which existing inline computations this layer could eventually
replace, and why removal is out of scope here).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from eligsir.mapping.evidence.contracts import (
    EvidenceMeta,
    EvidenceSummary,
    GaussianEvidence,
    InputEvidence,
    RenderEvidence,
)
from eligsir.mapping.evidence.gaussian_history import GaussianEvidenceHistory
from eligsir.mapping.evidence.reducers import (
    ema_update,
    scalar_fraction,
    scalar_mean,
    under_supported_mass,
)
from eligsir.mapping.evidence.requirements import EvidenceRequirements


class EvidenceBuilder:
    """Demand-driven assembler for the four evidence dataclasses.

    Holds no state of its own beyond an optional
    :class:`GaussianEvidenceHistory` for EMA continuity across calls
    (``xyz_grad_ema``/``opacity_grad_ema`` need a running average, not a
    single-frame snapshot — the raw per-step signal already exists as
    ``GaussianEnergyTerms.xyz_grad_norm``/``.opacity_grad_abs``, but the EMA
    itself must be reduced across steps, which is exactly what
    ``reducers.ema_update`` + this history object do).
    """

    def __init__(
        self,
        requirements: EvidenceRequirements,
        history: Optional[GaussianEvidenceHistory] = None,
        grad_ema_decay: float = 0.9,
    ) -> None:
        self.requirements = requirements
        self.history = history if history is not None else GaussianEvidenceHistory()
        self.grad_ema_decay = grad_ema_decay

    # ------------------------------------------------------------------
    # InputEvidence
    # ------------------------------------------------------------------
    def build_input_evidence(
        self,
        *,
        pose_confidence: float,
        image_quality: float,
        depth_valid_fraction: float,
        geometry_reliability_mean: float,
        meta: Optional[EvidenceMeta] = None,
    ) -> Optional[InputEvidence]:
        if not self.requirements.want_input_evidence:
            return None
        return InputEvidence(
            pose_confidence=pose_confidence,
            image_quality=image_quality,
            depth_valid_fraction=depth_valid_fraction,
            geometry_reliability_mean=geometry_reliability_mean,
            meta=meta or EvidenceMeta(),
        )

    # ------------------------------------------------------------------
    # RenderEvidence
    # ------------------------------------------------------------------
    def build_render_evidence(
        self,
        render_pkg: dict,
        *,
        gt_image: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        meta: Optional[EvidenceMeta] = None,
    ) -> Optional[RenderEvidence]:
        req = self.requirements
        if not req.want_render_evidence:
            return None

        rgb_residual = None
        if req.want_rgb_residual and gt_image is not None and "render" in render_pkg:
            rgb_residual = (render_pkg["render"].detach() - gt_image.detach()).detach()

        depth_residual = None
        if req.want_depth_residual and gt_depth is not None and render_pkg.get("rendered_depth") is not None:
            depth_residual = (render_pkg["rendered_depth"].detach() - gt_depth.detach()).detach()

        alpha = None
        if req.want_alpha:
            alpha = render_pkg.get("alpha")
            alpha = alpha.detach().clone() if alpha is not None else None

        rendered_depth = None
        if req.want_rendered_depth:
            rendered_depth = render_pkg.get("rendered_depth")
            rendered_depth = rendered_depth.detach().clone() if rendered_depth is not None else None

        transmittance = None
        if req.want_transmittance:
            raw_alpha = render_pkg.get("alpha")
            if raw_alpha is not None:
                transmittance = (1.0 - raw_alpha.detach()).detach()

        return RenderEvidence(
            rgb_residual=rgb_residual,
            depth_residual=depth_residual,
            alpha=alpha,
            rendered_depth=rendered_depth,
            transmittance=transmittance,
            meta=meta or EvidenceMeta(),
        )

    # ------------------------------------------------------------------
    # GaussianEvidence
    # ------------------------------------------------------------------
    def build_gaussian_evidence(
        self,
        energy_terms: Any,
        *,
        projected_radius: Optional[torch.Tensor] = None,
        topology_version: int = 0,
        map_version: int = 0,
        meta: Optional[EvidenceMeta] = None,
    ) -> Optional[GaussianEvidence]:
        """``energy_terms`` is expected to duck-type
        ``utils.strategies.mcmc.energy_policy.GaussianEnergyTerms``
        (``alpha``, ``support``, ``xyz_grad_norm``, ``opacity_grad_abs``,
        ``visibility``, ``age_frac``, ``geometry_confidence``,
        ``free_space_conflict``, ``render_contribution_ema``) — imported
        loosely (``Any``) rather than a hard dependency, so this module
        does not force an import-order coupling onto
        ``utils/strategies/mcmc``.

        ``projected_radius`` is passed separately since
        ``GaussianEnergyTerms`` does not carry it; the natural existing
        source is ``render_pkg["radii"]``/``render_pkg["gaussian_radii"]``
        (see ``eligsir/mapping/renderer/gsplat_backend.py``), which the
        caller already has.
        """
        req = self.requirements
        if not req.want_gaussian_evidence:
            return None

        device_ref = energy_terms.alpha
        zeros = torch.zeros_like(device_ref)

        # NOTE: every tensor handed back here is `.detach().clone()`-ed, not
        # just `.detach()`-ed. `.detach()` alone still shares storage with
        # the source tensor -- a downstream evidence consumer mutating the
        # returned `GaussianEvidence` in place (e.g. an in-place clamp)
        # would otherwise silently corrupt the live MCMC/energy-policy
        # state this evidence was read from. Evidence must be a read-only
        # observation, never a live alias.
        visibility = energy_terms.visibility.detach().clone() if req.want_visibility else zeros
        contribution = (
            energy_terms.render_contribution_ema.detach().clone()
            if req.want_contribution and energy_terms.render_contribution_ema is not None
            else zeros
        )
        radius = (
            projected_radius.detach().clone()
            if req.want_projected_radius and projected_radius is not None
            else zeros
        )
        support = energy_terms.support.detach().clone() if req.want_support else zeros
        geometry_confidence = (
            energy_terms.geometry_confidence.detach().clone()
            if req.want_geometry_confidence and energy_terms.geometry_confidence is not None
            else zeros
        )
        free_space_conflict = (
            energy_terms.free_space_conflict.detach().clone()
            if req.want_free_space_conflict and energy_terms.free_space_conflict is not None
            else zeros
        )
        age = (
            energy_terms.age_frac.detach().clone()
            if req.want_age and energy_terms.age_frac is not None
            else zeros
        )

        # begin_step() must run before any EMA state is read, so a
        # topology-version change cold-starts the EMA *before* we blend in
        # this step's observation -- see GaussianEvidenceHistory.begin_step.
        self.history.begin_step(topology_version, map_version)

        xyz_grad_ema = zeros
        opacity_grad_ema = zeros
        if req.want_xyz_grad_ema or req.want_opacity_grad_ema:
            prev_xyz_ema, prev_opacity_ema = self.history.ema_state()
            new_xyz_ema = prev_xyz_ema
            new_opacity_ema = prev_opacity_ema
            if req.want_xyz_grad_ema:
                new_xyz_ema = ema_update(prev_xyz_ema, energy_terms.xyz_grad_norm, self.grad_ema_decay)
                xyz_grad_ema = new_xyz_ema
            if req.want_opacity_grad_ema:
                new_opacity_ema = ema_update(prev_opacity_ema, energy_terms.opacity_grad_abs, self.grad_ema_decay)
                opacity_grad_ema = new_opacity_ema
            self.history.set_ema_state(new_xyz_ema, new_opacity_ema)

        evidence = GaussianEvidence(
            visibility=visibility,
            projected_radius=radius,
            contribution=contribution,
            xyz_grad_ema=xyz_grad_ema,
            opacity_grad_ema=opacity_grad_ema,
            support=support,
            geometry_confidence=geometry_confidence,
            free_space_conflict=free_space_conflict,
            age=age,
            meta=meta or EvidenceMeta(topology_version=topology_version, map_version=map_version),
        )
        self.history.update(evidence, topology_version, map_version)
        return evidence

    # ------------------------------------------------------------------
    # EvidenceSummary
    # ------------------------------------------------------------------
    def build_summary(
        self,
        energy_terms: Any,
        *,
        new_surface_ratio: float = 0.0,
        persistent_residual: float = 0.0,
        predicted_cost_ms: float = 0.0,
        support_threshold: float = 0.01,
        meta: Optional[EvidenceMeta] = None,
    ) -> Optional[EvidenceSummary]:
        if not self.requirements.want_summary:
            return None
        under_supported = under_supported_mass(energy_terms.support, energy_terms.alpha, support_threshold)
        expected_gradient_value = scalar_mean(energy_terms.xyz_grad_norm)
        return EvidenceSummary(
            new_surface_ratio=new_surface_ratio,
            persistent_residual=persistent_residual,
            under_supported_mass=under_supported,
            expected_gradient_value=expected_gradient_value,
            predicted_cost_ms=predicted_cost_ms,
            meta=meta or EvidenceMeta(),
        )
