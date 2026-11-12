"""GaussianEnergyPolicy — unified per-Gaussian energy term logic.

Consolidates the term computation currently split between
``utils.energy_mcmc`` and ``utils.strategies.mcmc_strategy`` into one place.

Phase 1 (current): wraps the existing API and delegates to it; the contract
surface is established but no numerical change occurs. Ablation gate: the
original ``compute_gaussian_utility`` / ``compute_dead_mask`` remain in
``utils.energy_mcmc`` and are called by default paths. Callers that switch
to ``GaussianEnergyPolicy`` get numerically equivalent results.

Phase 3+: ``GaussianEnergyPolicy.compute_population_energy()`` will be
added here, gated by ``energy_mode="population_energy"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GaussianEnergyTerms:
    """Per-Gaussian energy components, all shape ``[N]``.

    All history/state tensors (``support``, ``age_frac``, ``has_anchor``,
    ``has_depth_conflict``) are **detached** when ``detach_state=True``
    (the default) so they contribute no gradient.

    Attributes:
        alpha: Activated opacity in [0, 1].
        support: Multi-view visibility EMA.
        max_scale: Maximum scale per Gaussian.
        min_scale: Minimum scale per Gaussian (>0).
        anisotropy: max_scale / min_scale ratio.
        xyz_grad_norm: L2 norm of position gradient (after backward).
        opacity_grad_abs: Absolute opacity gradient (after backward).
        scale_grad_norm: L2 norm of scale gradient (after backward).
        visibility: Single-frame visibility mask (bool).
        age_frac: Normalised age in [0, 1] (birth_iter / current_iteration).
        has_anchor: Whether each Gaussian has a registered anchor.
        has_depth_conflict: Whether a depth conflict was recorded.
        provisional: Whether each Gaussian is in the provisional lifecycle state.
    """

    alpha: torch.Tensor
    support: torch.Tensor
    max_scale: torch.Tensor
    min_scale: torch.Tensor
    anisotropy: torch.Tensor
    xyz_grad_norm: torch.Tensor
    opacity_grad_abs: torch.Tensor
    scale_grad_norm: torch.Tensor
    visibility: torch.Tensor
    age_frac: Optional[torch.Tensor] = None
    has_anchor: Optional[torch.Tensor] = None
    has_depth_conflict: Optional[torch.Tensor] = None
    provisional: Optional[torch.Tensor] = None
    geometry_confidence: Optional[torch.Tensor] = None
    surface_state: Optional[torch.Tensor] = None
    normal_drift: Optional[torch.Tensor] = None
    duplicate_mass: Optional[torch.Tensor] = None
    free_space_conflict: Optional[torch.Tensor] = None
    surface_cell_overpopulation: Optional[torch.Tensor] = None
    render_contribution_ema: Optional[torch.Tensor] = None


@dataclass
class GaussianEnergyConfig:
    """Hyperparameters controlling energy term computation.

    Attributes:
        w_alpha: Weight for opacity term in utility.
        w_vis: Weight for single-frame visibility.
        w_support: Weight for multi-view support EMA.
        w_grad: Weight for position gradient norm.
        w_scale: Weight for scale penalty.
        w_dead: Weight for dead penalty.
        beta_opacity: Opacity sigmoid gain in utility_core.
        beta_scale: Scale penalty gain in utility_core.
        alpha_dead: Opacity death threshold.
        support_threshold: Support EMA death threshold.
        w_aniso_utility: Anisotropy penalty weight in utility score.
        aniso_soft_limit: Anisotropy ratio above which penalty activates.
        scale_kill_threshold: Max scale above which low-support → dead.
        scale_moderate_threshold: Mid-scale threshold for soft kills.
        scale_moderate_opacity_mult: Opacity multiplier for mid-scale kills.
        scale_dead_support_mult: Support multiplier for scale/aniso clauses.
        anisotropy_kill_ratio: s_max/s_min ratio above which needle → dead.
        utility_quantile: Bottom quantile for utility-based death.
        use_utility_quantile: Enable utility quantile death clause.
        detach_state: Detach history buffers from autograd graph.
    """

    w_alpha: float = 1.0
    w_vis: float = 2.0
    w_support: float = 2.0
    w_grad: float = 3.0
    w_scale: float = 0.5
    w_dead: float = 1.0
    beta_opacity: float = 1.0
    beta_scale: float = 0.5
    alpha_dead: float = 0.005
    support_threshold: float = 0.01
    w_aniso_utility: float = 0.1
    aniso_soft_limit: float = 10.0
    scale_kill_threshold: float = 0.20
    scale_moderate_threshold: float = 0.05
    scale_moderate_opacity_mult: float = 4.0
    scale_dead_support_mult: float = 1.0
    anisotropy_kill_ratio: float = 20.0
    utility_quantile: float = 0.05
    use_utility_quantile: bool = True
    detach_state: bool = True
    # Phase 3: population energy weights (used when energy_mode="population_energy")
    energy_mode: str = "legacy_utility"  # "legacy_utility" | "population_energy"
    energy_w_opacity: float = 1.0
    energy_w_support: float = 2.0
    energy_w_scale: float = 0.5
    energy_w_anisotropy: float = 0.3
    energy_w_dead: float = 1.0
    energy_w_grad: float = 3.0
    energy_w_conflict: float = 0.5
    energy_w_age: float = 0.2
    energy_w_geometry_support: float = 0.0
    energy_w_duplicate: float = 0.0
    energy_w_surface_drift: float = 0.0
    energy_w_free_space: float = 0.0
    energy_w_budget: float = 0.0
    # Simplified per-frame-scalar depth-residual gate on growth-utility gradient
    # term (see utils/energy_mcmc.compute_gaussian_utility docstring). 0.0 = off.
    energy_w_depth_residual: float = 0.0
    # Phase 4: energy-based death mode
    energy_dead_mode: str = "legacy_clauses"  # "legacy_clauses" | "energy_quantile" | "energy_sigmoid"
    energy_dead_quantile: float = 0.05
    energy_dead_temperature: float = 10.0
    # Phase 6: trust-region MCMC proposal
    mcmc_proposal_mode: str = "legacy_clamped"  # "legacy_clamped" | "trust_region"
    mcmc_trust_radius: float = 0.05
    mcmc_support_power: float = 0.5
    mcmc_opacity_power: float = 0.5
    # Campaign C3: retarget the anisotropy dead-mask clause from s_max/s_min ("s_max_min",
    # default — conflates flat disks with needles) to the needle ratio s2/s1 ("s2_s1",
    # kills when s2/s1 < needle_kill_ratio). Default preserves legacy parity.
    aniso_kill_mode: str = "s_max_min"
    needle_kill_ratio: float = 0.3
    # Phase 10: soft lifecycle
    soft_lifecycle_enabled: bool = False
    soft_lifecycle_support_k: float = 10.0
    soft_lifecycle_age_k: float = 5.0


# ---------------------------------------------------------------------------
# CLI -> config wiring (issue #385)
# ---------------------------------------------------------------------------
#
# Every ``GaussianEnergyConfig`` field that is meant to be tunable, mapped to
# the argparse flag that carries it. Before this map existed, each of the nine
# production construction sites hand-listed a different subset of fields, so
# any field a given site forgot was silently pinned at its dataclass default no
# matter what the CLI (or a spec's ``extra_args``) said. The worst of those was
# ``utils/strategies/mcmc/energy.py``'s ``_pcfg``, which drives
# ``compute_legacy_dead_mask`` -- i.e. real Gaussian birth/death topology on
# the default ``gsplat_energy_mcmc`` strategy -- while pinning five of the nine
# fields that method reads.
#
# Two deliberate omissions from this map:
#   * ``alpha_dead`` -- a REQUIRED keyword argument of
#     :func:`energy_config_from_args` with no args fallback, because its correct
#     source genuinely differs per call site. The strategy sites take the
#     iteration-varying ``sched["dead_opacity_threshold"]`` from
#     ``MCMCScheduleConfig``; the utility/telemetry sites take the static
#     ``args.energy_alpha_dead``. These are not interchangeable (the parser
#     default 0.05 is 10x the dataclass default 0.005), so the factory refuses
#     to guess.
#   * ``detach_state`` -- intentionally internal. It controls whether history
#     buffers are cut out of the autograd graph, which is a correctness
#     property of the policy, not a hyperparameter to sweep.
#
# The fallback for each entry is read from a ``GaussianEnergyConfig()`` instance
# rather than being restated as a literal, so this map cannot develop the stale
# ``getattr(args, "<flag>", <literal>)`` drift issue #368 / PR #380 added a
# repo-wide CI guard against -- there is no literal here to drift.
# ``tests/test_energy_config_cli_wiring.py`` additionally asserts every parser
# default equals the corresponding dataclass default, so the fallback and the
# real parser default can never disagree either.
_ARGS_FIELD_MAP: dict[str, str] = {
    # field name                     : argparse flag name
    "w_alpha": "energy_w_alpha",
    "w_vis": "energy_w_vis",
    "w_support": "energy_w_support",
    "w_grad": "energy_w_grad",
    "w_scale": "energy_w_scale",
    "w_dead": "energy_w_dead",
    "beta_opacity": "energy_beta_opacity",
    "beta_scale": "energy_beta_scale",
    "support_threshold": "energy_support_threshold",
    "w_aniso_utility": "utility_aniso_weight",
    "aniso_soft_limit": "utility_aniso_soft_limit",
    "scale_kill_threshold": "scale_kill_threshold",
    "scale_moderate_threshold": "scale_moderate_threshold",
    "scale_moderate_opacity_mult": "energy_scale_moderate_opacity_mult",
    "scale_dead_support_mult": "energy_scale_dead_support_mult",
    "anisotropy_kill_ratio": "anisotropy_kill_ratio",
    "utility_quantile": "energy_utility_quantile",
    "use_utility_quantile": "energy_use_utility_quantile",
    "energy_mode": "energy_mode",
    "energy_w_opacity": "energy_w_opacity",
    "energy_w_support": "energy_w_support",
    "energy_w_scale": "energy_w_scale",
    "energy_w_anisotropy": "energy_w_anisotropy",
    "energy_w_dead": "energy_w_dead",
    "energy_w_grad": "energy_w_grad",
    "energy_w_conflict": "energy_w_conflict",
    "energy_w_age": "energy_w_age",
    "energy_w_geometry_support": "energy_w_geometry_support",
    "energy_w_duplicate": "energy_w_duplicate",
    "energy_w_surface_drift": "energy_w_surface_drift",
    "energy_w_free_space": "energy_w_free_space",
    "energy_w_budget": "energy_w_budget",
    "energy_w_depth_residual": "energy_w_depth_residual",
    "energy_dead_mode": "energy_dead_mode",
    "energy_dead_quantile": "energy_dead_quantile",
    "energy_dead_temperature": "energy_dead_temperature",
    "mcmc_proposal_mode": "mcmc_proposal_mode",
    "mcmc_trust_radius": "mcmc_trust_radius",
    "mcmc_support_power": "mcmc_trust_support_power",
    "mcmc_opacity_power": "mcmc_trust_opacity_power",
    "aniso_kill_mode": "aniso_kill_mode",
    "needle_kill_ratio": "needle_reg_threshold",
    "soft_lifecycle_enabled": "soft_lifecycle_enabled",
    "soft_lifecycle_support_k": "soft_lifecycle_support_k",
    "soft_lifecycle_age_k": "soft_lifecycle_age_k",
}

# Fields deliberately NOT sourced from args by the factory (see above).
_ARGS_FIELD_MAP_EXEMPT: frozenset[str] = frozenset({"alpha_dead", "detach_state"})


def energy_config_from_args(args, *, alpha_dead: float, **overrides) -> GaussianEnergyConfig:
    """Build a :class:`GaussianEnergyConfig` from parsed CLI args.

    This is the single place the args -> config mapping is defined. Every
    production construction site goes through it, so a new tunable field is
    reachable from the CLI as soon as it is added to :data:`_ARGS_FIELD_MAP` --
    it can no longer be reachable at one site and pinned at another.

    Args:
        args: Parsed trainer args (any object; missing attributes fall back to
            the ``GaussianEnergyConfig`` dataclass default for that field, which
            the wiring test pins equal to the real parser default).
        alpha_dead: Opacity death threshold. Required keyword-only, with no
            args fallback, because the correct source differs per call site:
            MCMC strategy sites pass the iteration-varying
            ``sched["dead_opacity_threshold"]``, utility/telemetry sites pass
            ``args.energy_alpha_dead``. See ``_ARGS_FIELD_MAP``'s note.
        **overrides: Fields to force regardless of ``args`` -- for call sites
            that are already inside a branch gated on that mode (e.g.
            ``energy_mode="population_energy"``).

    Returns:
        A fully-populated config.
    """
    defaults = GaussianEnergyConfig()
    kwargs: dict = {"alpha_dead": float(alpha_dead)}
    for field_name, flag in _ARGS_FIELD_MAP.items():
        default = getattr(defaults, field_name)
        value = getattr(args, flag, default)
        # Coerce to the dataclass field's declared type so a str-typed CLI
        # value (or a numpy scalar out of a config file) can't leak through.
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, float):
            value = float(value)
        elif isinstance(default, str):
            value = str(value)
        kwargs[field_name] = value
    kwargs.update(overrides)
    return GaussianEnergyConfig(**kwargs)


def _dense_grad(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Densify a sparse gradient or return it unchanged.

    Args:
        grad: Gradient tensor, possibly sparse.

    Returns:
        Dense gradient tensor or None.
    """
    if grad is None:
        return None
    if getattr(grad, "layout", torch.strided) != torch.strided:
        return grad.to_dense()
    return grad


def _named_grad(
    gaussians, legacy_attr: str, gsplat_key: Optional[str] = None
) -> Optional[torch.Tensor]:
    """Fetch gradient by name, supporting both gsplat and legacy model layouts.

    Args:
        gaussians: Gaussian model (any layout).
        legacy_attr: Attribute name on the legacy ``GaussianModel``.
        gsplat_key: Key in ``gaussians.params`` for the gsplat layout.

    Returns:
        Dense gradient tensor or None.
    """
    params = getattr(gaussians, "params", None)
    if params is not None:
        # gsplat layout: read the leaf Parameter's grad. The legacy attribute
        # (e.g. ``_xyz``) is a non-leaf property here, so touching ``.grad`` on
        # it both returns None and spams a UserWarning every call — avoid it.
        if gsplat_key is not None and gsplat_key in params:
            return _dense_grad(params[gsplat_key].grad)
        return None
    return _dense_grad(getattr(gaussians, legacy_attr).grad)


@dataclass
class PopulationEnergyResult:
    """Output of :meth:`GaussianEnergyPolicy.compute_population_energy`.

    All per-component tensors have shape ``[N]``. Scalar means are
    pre-computed for logging.

    Attributes:
        total: Total per-Gaussian energy ``[N]``.
        E_opacity: Opacity component.
        E_support: Support component.
        E_scale: Scale component.
        E_anisotropy: Anisotropy component.
        E_dead: Death-proximity component.
        E_grad: Gradient-signal component.
        E_conflict: Depth-conflict component.
        E_age: Age/maturity component.
        population_mean: Mean total energy.
        population_p95: 95th-percentile total energy.
    """

    total: torch.Tensor
    E_opacity: torch.Tensor
    E_support: torch.Tensor
    E_scale: torch.Tensor
    E_anisotropy: torch.Tensor
    E_dead: torch.Tensor
    E_grad: torch.Tensor
    E_conflict: torch.Tensor
    E_age: torch.Tensor
    population_mean: float
    population_p95: float

    def to_tensorboard_dict(self) -> dict:
        """Return scalar summaries for TensorBoard logging.

        Returns:
            Dict of metric name → float.
        """
        return {
            "energy/population_mean": self.population_mean,
            "energy/population_p95": self.population_p95,
            "energy/E_opacity_mean": float(self.E_opacity.mean()),
            "energy/E_support_mean": float(self.E_support.mean()),
            "energy/E_scale_mean": float(self.E_scale.mean()),
            "energy/E_anisotropy_mean": float(self.E_anisotropy.mean()),
            "energy/E_dead_mean": float(self.E_dead.mean()),
            "energy/E_grad_mean": float(self.E_grad.mean()),
            "energy/E_age_mean": float(self.E_age.mean()),
        }


@dataclass
class DeadMaskResult:
    """Dead-mask with per-clause breakdown for TB logging.

    Attributes:
        mask: ``[N]`` bool dead mask.
        stats: Dict of clause name → dead count (non-exclusive).
    """

    mask: torch.Tensor
    stats: dict


class GaussianEnergyPolicy:
    """Compute per-Gaussian energy terms and drive death decisions.

    Wraps the existing ``compute_gaussian_utility`` and ``compute_dead_mask``
    functions in ``utils.energy_mcmc`` during Phase 1. Callers get the same
    numerical results but through the canonical interface.

    Args:
        config: Energy hyperparameters. Uses defaults if not provided.
    """

    def __init__(self, config: Optional[GaussianEnergyConfig] = None) -> None:
        self.config = config or GaussianEnergyConfig()

    def compute_terms(
        self, gaussians, render_pkg: dict, iteration: int
    ) -> GaussianEnergyTerms:
        """Extract per-Gaussian energy terms from the model and render output.

        Args:
            gaussians: Gaussian model (``GsplatGaussianModel`` or ``GaussianModel``).
            render_pkg: Dict returned by the renderer (contains
                ``visibility_filter``).
            iteration: Current training iteration (used for age normalisation).

        Returns:
            :class:`GaussianEnergyTerms` with all terms populated.
        """
        cfg = self.config
        device = gaussians.get_xyz.device
        N = gaussians.get_xyz.shape[0]

        alpha = gaussians.get_opacity.squeeze(-1)

        visibility = render_pkg.get(
            "visibility_filter", torch.ones(N, device=device, dtype=torch.bool)
        )

        # Guard against a stale visibility_ema left over from before an
        # external growth event (e.g. GsplatDefaultStrategy's clone/split,
        # which doesn't extend this MCMC-strategy-specific buffer the way
        # this project's own add/relocate path does) -- same defensive
        # pattern already used below for anchor_confidence/surface_state/
        # anchor_normal in this function.
        support = (
            gaussians.visibility_ema.squeeze(-1)
            if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == N
            else visibility.float()
        )
        if cfg.detach_state:
            support = support.detach()

        s = gaussians.get_scaling
        # Cap max_scale at a large-but-finite numerical safety ceiling, not a
        # scene-scale parameter -- purely to keep `anisotropy`/energy terms
        # derived from it finite. Many compounding MCMC relocation cycles on
        # the same surviving Gaussian can otherwise drive a scale toward Inf
        # (found live 2026-08-13, see _norm's comment in
        # compute_population_energy for the full failure chain).
        max_scale = s.max(dim=1).values.clamp_max(1e6)
        min_scale = s.min(dim=1).values.clamp_min(1e-6)
        anisotropy = max_scale / min_scale

        xyz_grad = torch.zeros(N, device=device)
        g = _named_grad(gaussians, "_xyz", "means")
        if g is not None:
            xyz_grad = g.norm(dim=1)

        opacity_grad = torch.zeros(N, device=device)
        g = _named_grad(gaussians, "_opacity", "opacities")
        if g is not None:
            opacity_grad = g.abs().flatten()

        scale_grad = torch.zeros(N, device=device)
        g = _named_grad(gaussians, "_scaling", "scales")
        if g is not None:
            scale_grad = g.norm(dim=1)

        age_frac = None
        if hasattr(gaussians, "birth_iter") and iteration > 0 and gaussians.birth_iter.shape[0] == N:
            bf = gaussians.birth_iter.float()
            if cfg.detach_state:
                bf = bf.detach()
            age_frac = (iteration - bf).clamp_min(0) / max(iteration, 1)

        has_anchor = None
        if hasattr(gaussians, "anchor_xyz") and gaussians.anchor_xyz.shape[0] == N:
            has_anchor = (gaussians.anchor_xyz.abs().sum(dim=-1) > 0).detach()

        has_depth_conflict = None
        if hasattr(gaussians, "depth_conflict_count") and gaussians.depth_conflict_count.shape[0] == N:
            has_depth_conflict = (gaussians.depth_conflict_count > 0).detach()

        provisional = None
        if hasattr(gaussians, "provisional") and gaussians.provisional.shape[0] == N:
            provisional = gaussians.provisional.detach()

        geometry_confidence = None
        if hasattr(gaussians, "anchor_confidence") and gaussians.anchor_confidence.shape[0] == N:
            geometry_confidence = gaussians.anchor_confidence.detach() if cfg.detach_state else gaussians.anchor_confidence

        surface_state = None
        if hasattr(gaussians, "surface_state") and gaussians.surface_state.shape[0] == N:
            surface_state = gaussians.surface_state.detach()

        normal_drift = None
        if (
            hasattr(gaussians, "anchor_normal")
            and hasattr(gaussians, "anchor_xyz")
            and gaussians.anchor_normal.shape[0] == N
            and gaussians.anchor_xyz.shape[0] == N
        ):
            normal = torch.nn.functional.normalize(gaussians.anchor_normal.detach(), dim=-1)
            normal_drift = ((gaussians.get_xyz - gaussians.anchor_xyz.detach()) * normal).sum(dim=-1).abs()

        return GaussianEnergyTerms(
            alpha=alpha,
            support=support,
            max_scale=max_scale,
            min_scale=min_scale,
            anisotropy=anisotropy,
            xyz_grad_norm=xyz_grad,
            opacity_grad_abs=opacity_grad,
            scale_grad_norm=scale_grad,
            visibility=visibility,
            age_frac=age_frac,
            has_anchor=has_anchor,
            has_depth_conflict=has_depth_conflict,
            provisional=provisional,
            geometry_confidence=geometry_confidence,
            surface_state=surface_state,
            normal_drift=normal_drift,
        )

    def compute_utility(
        self, gaussians, render_pkg: dict, iteration: int
    ) -> torch.Tensor:
        """Compute per-Gaussian utility score, numerically equivalent to legacy.

        Delegates to ``utils.energy_mcmc.compute_gaussian_utility`` so Phase 1
        produces identical values to the existing code.

        Args:
            gaussians: Gaussian model.
            render_pkg: Renderer output dict.
            iteration: Training iteration.

        Returns:
            ``[N]`` utility score tensor.
        """
        from eligsir.utils.energy_mcmc import compute_gaussian_utility

        cfg = self.config
        return compute_gaussian_utility(
            gaussians,
            render_pkg,
            iteration,
            w_alpha=cfg.w_alpha,
            w_vis=cfg.w_vis,
            w_grad=cfg.w_grad,
            w_scale=cfg.w_scale,
            w_dead=cfg.w_dead,
            w_support=cfg.w_support,
            beta_opacity=cfg.beta_opacity,
            beta_scale=cfg.beta_scale,
            alpha_dead=cfg.alpha_dead,
            w_aniso_utility=cfg.w_aniso_utility,
            aniso_soft_limit=cfg.aniso_soft_limit,
        )

    def compute_legacy_dead_mask(
        self,
        gaussians,
        utility: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
        young_prot_iters: int = 0,
        active_mask: Optional[torch.Tensor] = None,
    ) -> DeadMaskResult:
        """Compute dead mask with per-clause breakdown (Phase 2).

        Reproduces all six clauses from ``compute_dead_mask`` and returns
        a :class:`DeadMaskResult` so TB counts match the actual mask.
        Replaces ``_dead_clause_breakdown()`` in ``mcmc_strategy.py``.

        Args:
            gaussians: Gaussian model.
            utility: Optional ``[N]`` utility scores.
            active_mask: Optional ``[N]`` bool mask (bubble-as-local-bounds);
                the final dead mask is intersected with it so only in-bubble
                Gaussians can be declared dead. Clause logic is unchanged.

        Returns:
            :class:`DeadMaskResult` with mask and per-clause stats dict.
        """
        cfg = self.config
        alpha = gaussians.get_opacity.squeeze(-1)
        support = (
            gaussians.visibility_ema.squeeze(-1)
            if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == alpha.shape[0]
            else torch.ones_like(alpha)
        )

        thresh = cfg.alpha_dead
        sup_thresh = cfg.support_threshold
        eff_support = sup_thresh * cfg.scale_dead_support_mult
        very_low = thresh * 0.1

        stats = {}

        dead = alpha < very_low
        stats["very_low_opacity"] = int(dead.sum())

        clause2 = (alpha < thresh) & (support < sup_thresh)
        dead = dead | clause2
        stats["low_opacity_low_support"] = int(clause2.sum())

        stats["large_low_support"] = 0
        stats["moderate_transparent"] = 0
        stats["anisotropic"] = 0

        if hasattr(gaussians, "get_scaling"):
            s = gaussians.get_scaling
            max_s = s.max(dim=-1).values
            min_s = s.min(dim=-1).values.clamp_min(1e-6)
            aniso = max_s / min_s

            if cfg.scale_kill_threshold > 0:
                c3 = (max_s > cfg.scale_kill_threshold) & (support < eff_support)
                dead = dead | c3
                stats["large_low_support"] = int(c3.sum())

            if cfg.scale_moderate_threshold > 0:
                c4 = (
                    (max_s > cfg.scale_moderate_threshold)
                    & (alpha < thresh * cfg.scale_moderate_opacity_mult)
                    & (support < eff_support)
                )
                dead = dead | c4
                stats["moderate_transparent"] = int(c4.sum())

            if cfg.anisotropy_kill_ratio > 0:
                # C3: choose the anisotropy criterion. "s2_s1" targets needles (second-axis
                # collapse) without flagging healthy flat disks; default keeps s_max/s_min.
                if getattr(cfg, "aniso_kill_mode", "s_max_min") == "s2_s1":
                    top2 = torch.topk(s, k=min(2, s.shape[-1]), dim=-1).values
                    needle_ratio = top2[:, 1] / top2[:, 0].clamp_min(1e-6)
                    aniso_clause = needle_ratio < getattr(cfg, "needle_kill_ratio", 0.3)
                else:
                    aniso_clause = aniso > cfg.anisotropy_kill_ratio
                c5 = (
                    aniso_clause
                    & (alpha < thresh * cfg.scale_moderate_opacity_mult)
                    & (support < eff_support)
                )
                dead = dead | c5
                stats["anisotropic"] = int(c5.sum())

        if cfg.use_utility_quantile and utility is not None and utility.numel() > 0:
            q_val = torch.quantile(utility, cfg.utility_quantile)
            c6 = (alpha < thresh) & (utility < q_val)
            dead = dead | c6
            stats["low_utility_low_opacity"] = int(c6.sum())

        if young_prot_iters > 0 and current_iter is not None:
            birth_iter = getattr(gaussians, "birth_iter", None)
            if birth_iter is not None and birth_iter.shape[0] == dead.shape[0]:
                young = (current_iter - birth_iter.to(dtype=torch.int64)) < young_prot_iters
                from eligsir.utils.energy_mcmc import _or_exposure_young
                young = _or_exposure_young(gaussians, young)
                dead = dead & ~young
                stats["young_protected"] = int(young.sum())
            elif birth_iter is not None:
                from eligsir.utils.gaussian_lineage import warn_young_prot_mismatch_once
                warn_young_prot_mismatch_once(
                    gaussians,
                    site="energy_policy.compute_legacy_dead_mask",
                    warn_attr="_young_prot_mismatch_warned_legacy_dead_mask",
                    birth_len=int(birth_iter.shape[0]),
                    mask_len=int(dead.shape[0]),
                )

        if active_mask is not None and active_mask.shape[0] == dead.shape[0]:
            dead = dead & active_mask.to(dead.device)

        stats["total"] = int(dead.sum())

        # Mirror into utils.energy_mcmc's module-level pruning-stats accumulator
        # (report.json's geometry.pruning/pruning_cumulative blocks read this) --
        # this method is a separate implementation from
        # utils.energy_mcmc.compute_dead_mask and never touched that
        # accumulator, so total_pruned stayed 0 for every --energy_dead_mode
        # legacy_clauses run (the default) regardless of any threshold, even
        # though this method's own dead mask WAS being computed and applied.
        # Found 2026-07-06 tracing why total_pruned stayed 0 after two
        # threshold fixes that don't apply to this code path at all.
        #
        # issue #1286: the original 2026-07-06 fix mutated the module-level
        # `_pruning_stats` dict directly, which only updates the WINDOWED
        # counters (`pop_pruning_stats()`) -- it bypasses `.bump()`, the only
        # thing that also updates the CUMULATIVE counters
        # (`get_cumulative_pruning_stats()`), so `pruning_cumulative` stayed
        # permanently 0 for the entire run on this code path regardless of
        # how much pruning actually happened. Route through the same
        # `_bump_pruning_stat()` helper `energy_mcmc.compute_dead_mask()`
        # itself uses per-clause so both counters stay consistent.
        from eligsir.utils.energy_mcmc import _bump_pruning_stat
        _bump_pruning_stat("dead_opacity", stats.get("very_low_opacity", 0) + stats.get("low_opacity_low_support", 0))
        _bump_pruning_stat("scale_moderate", stats.get("large_low_support", 0) + stats.get("moderate_transparent", 0))
        _bump_pruning_stat("anisotropy_kill", stats.get("anisotropic", 0))
        _bump_pruning_stat("utility_quantile", stats.get("low_utility_low_opacity", 0))
        _bump_pruning_stat("young_protected_skip", stats.get("young_protected", 0))
        _bump_pruning_stat("total_pruned", stats["total"])

        return DeadMaskResult(mask=dead, stats=stats)

    def compute_dead_mask(
        self,
        gaussians,
        utility: Optional[torch.Tensor] = None,
        current_iter: Optional[int] = None,
        young_prot_iters: int = 0,
    ) -> torch.Tensor:
        """Compute the dead-Gaussian mask, numerically equivalent to legacy.

        Delegates to ``utils.energy_mcmc.compute_dead_mask``.

        Args:
            gaussians: Gaussian model.
            utility: Optional ``[N]`` utility scores (used by clause 6).

        Returns:
            ``[N]`` bool dead mask.
        """
        from eligsir.utils.energy_mcmc import compute_dead_mask

        cfg = self.config
        return compute_dead_mask(
            gaussians,
            utility=utility,
            opacity_threshold=cfg.alpha_dead,
            support_threshold=cfg.support_threshold,
            utility_quantile=cfg.utility_quantile,
            use_utility_quantile=cfg.use_utility_quantile,
            scale_kill_threshold=cfg.scale_kill_threshold,
            scale_moderate_threshold=cfg.scale_moderate_threshold,
            scale_moderate_opacity_mult=cfg.scale_moderate_opacity_mult,
            scale_dead_support_mult=cfg.scale_dead_support_mult,
            anisotropy_kill_ratio=cfg.anisotropy_kill_ratio,
            current_iter=current_iter,
            young_prot_iters=young_prot_iters,
        )

    def compute_population_energy(
        self, terms: GaussianEnergyTerms
    ) -> PopulationEnergyResult:
        """Compute per-Gaussian population energy from extracted terms (Phase 3).

        All components are normalised to [0, 1] before weighting so that
        coefficients are scale-invariant across sequences.

        Args:
            terms: Pre-computed :class:`GaussianEnergyTerms`.

        Returns:
            :class:`PopulationEnergyResult` with per-component and total energy.
        """
        cfg = self.config

        def _norm(t: torch.Tensor) -> torch.Tensor:
            # A single non-finite entry (e.g. `max_scale` overflowing to Inf
            # after many compounding MCMC relocation cycles on the same
            # surviving Gaussian) previously poisoned the WHOLE population:
            # vmax became Inf, every finite entry divided to 0, and the
            # offending entry itself became Inf/Inf=NaN -- then any
            # downstream torch.quantile() over `total` (which sums this
            # term in) returns NaN for the entire tensor, permanently and
            # silently disabling energy_quantile/energy_sigmoid dead-mask
            # selection (found live 2026-08-13: relocation ran for ~51k
            # iterations then silently stopped finding any dead Gaussians
            # for the remaining ~249k iterations of a 300k-iteration run).
            # Sanitize at the source so one bad Gaussian can't take down
            # every other Gaussian's death-energy score.
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            vmax = t.max().clamp_min(1e-6)
            return t / vmax

        E_opacity = cfg.energy_w_opacity * (1.0 - terms.alpha)
        E_support = cfg.energy_w_support * (1.0 - terms.support.clamp(0, 1))
        E_scale = cfg.energy_w_scale * _norm(terms.max_scale)
        E_anisotropy = cfg.energy_w_anisotropy * _norm((terms.anisotropy - 1.0).clamp_min(0))
        E_dead = cfg.energy_w_dead * (terms.alpha < cfg.alpha_dead).float()
        E_grad = cfg.energy_w_grad * (1.0 - _norm(terms.xyz_grad_norm))
        E_conflict = torch.zeros_like(terms.alpha)
        if terms.has_depth_conflict is not None:
            E_conflict = cfg.energy_w_conflict * terms.has_depth_conflict.float()
        E_geometry_support = torch.zeros_like(terms.alpha)
        if terms.geometry_confidence is not None:
            # Trusted geometry protects splats, so it subtracts from death energy.
            E_geometry_support = -cfg.energy_w_geometry_support * terms.geometry_confidence.clamp(0, 1)
        E_surface_drift = torch.zeros_like(terms.alpha)
        if terms.normal_drift is not None:
            E_surface_drift = cfg.energy_w_surface_drift * _norm(terms.normal_drift)
        E_duplicate = torch.zeros_like(terms.alpha)
        if terms.duplicate_mass is not None:
            E_duplicate = cfg.energy_w_duplicate * _norm(terms.duplicate_mass)
        E_free = torch.zeros_like(terms.alpha)
        if terms.free_space_conflict is not None:
            E_free = cfg.energy_w_free_space * _norm(terms.free_space_conflict)
        E_budget = torch.zeros_like(terms.alpha)
        if cfg.energy_w_budget > 0 and terms.alpha.numel() > 0:
            E_budget = torch.full_like(terms.alpha, cfg.energy_w_budget)
        E_age = torch.zeros_like(terms.alpha)
        if terms.age_frac is not None:
            E_age = cfg.energy_w_age * (1.0 - terms.age_frac.clamp(0, 1))

        total = (
            E_opacity + E_support + E_scale + E_anisotropy + E_dead + E_grad
            + E_conflict + E_age + E_geometry_support + E_surface_drift
            + E_duplicate + E_free + E_budget
        )
        # Defense in depth (see _norm's comment above): a non-finite `total`
        # entry from any future term poisons torch.quantile/mean for the
        # WHOLE population, not just the offending Gaussian. Sanitize here
        # too so this can't silently deadlock dead-mask selection again.
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        pop_mean = float(total.mean())
        pop_p95 = float(torch.quantile(total, 0.95))

        return PopulationEnergyResult(
            total=total,
            E_opacity=E_opacity,
            E_support=E_support,
            E_scale=E_scale,
            E_anisotropy=E_anisotropy,
            E_dead=E_dead,
            E_grad=E_grad,
            E_conflict=E_conflict,
            E_age=E_age,
            population_mean=pop_mean,
            population_p95=pop_p95,
        )

    def compute_dead_mask_from_energy(
        self,
        energy: PopulationEnergyResult,
        gaussians,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dead mask from population energy (Phase 4).

        Args:
            energy: Result of :meth:`compute_population_energy`.
            gaussians: Gaussian model (used for opacity gate).
            active_mask: Optional ``[N]`` bool mask restricting the POPULATION
                STATISTIC (quantile / mean) to the active local region
                (bubble-as-local-bounds). Without it, out-of-bubble/frozen
                Gaussians shift the thresholds applied to active ones as the
                frozen fraction grows. Dead is also constrained to the mask.

        Returns:
            ``[N]`` bool dead mask.
        """
        cfg = self.config
        alpha = gaussians.get_opacity.squeeze(-1)
        total = energy.total
        stat_pool = total[active_mask] if (active_mask is not None and bool(active_mask.any())) else total

        if cfg.energy_dead_mode == "energy_quantile":
            # `total` is a DEATH-energy score: higher = more deserving of death
            # (E_opacity rises as opacity falls, E_support rises as support
            # falls, geometry support is subtractive/protective -- see
            # test_geometry_support_is_protective). The dead quantile must
            # therefore select the TOP `energy_dead_quantile` fraction (the
            # worst-scoring tail), not the bottom -- selecting the bottom
            # picks the HEALTHIEST Gaussians, which almost never also have
            # low opacity, so the AND below silently produced an empty mask.
            q_val = torch.quantile(stat_pool, 1.0 - cfg.energy_dead_quantile)
            dead = (total > q_val) & (alpha < cfg.alpha_dead * cfg.scale_moderate_opacity_mult)
        elif cfg.energy_dead_mode == "energy_sigmoid":
            # Same direction fix: dead probability must rise with `total`
            # (higher death-energy), not fall with it.
            pop_mean = stat_pool.mean() if active_mask is not None else energy.population_mean
            prob = torch.sigmoid(cfg.energy_dead_temperature * (total - pop_mean))
            dead = (prob > 0.5) & (alpha < cfg.alpha_dead * cfg.scale_moderate_opacity_mult)
        else:
            dead = self.compute_dead_mask(gaussians)
        if active_mask is not None:
            dead = dead & active_mask
        return dead

    def sample_trust_region_noise(
        self,
        gaussians,
        xyz_lr: float,
        noise_lr: float,
        visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample trust-region MCMC noise proposals (Phase 6).

        Replaces the manual ``noise_max_scale`` / ``noise_max_step`` clamps in
        ``ScheduledMCMCStrategy.inject_noise`` with per-Gaussian trust radii
        proportional to confidence (opacity × support).

        Args:
            gaussians: Gaussian model.
            xyz_lr: Position learning rate at this step.
            noise_lr: Noise lr scale (from ``args.noise_lr``).
            visible: Optional bool visibility mask; if provided, only visible
                Gaussians are given non-zero noise.

        Returns:
            ``[N, 3]`` noise tensor (zero for non-visible / frozen Gaussians).
        """
        from eligsir.utils.general_utils import build_scaling_rotation

        cfg = self.config
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device

        alpha = gaussians.get_opacity.squeeze(-1).detach()
        if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.shape[0] == N:
            _vis = gaussians.visibility_ema.squeeze(-1).detach()
            # Fall back to uniform support when EMA has not yet been warmed up
            # (train_streaming.py doesn't call update_visibility_ema yet)
            support = _vis if _vis.sum() > 0.5 else torch.ones(N, device=device)
        else:
            support = torch.ones(N, device=device)

        # Per-Gaussian trust radius: shrink near low-opacity / low-support Gaussians
        radius = (
            cfg.mcmc_trust_radius
            * alpha.clamp(0, 1) ** cfg.mcmc_opacity_power
            * support.clamp(0, 1) ** cfg.mcmc_support_power
        )

        scales = gaussians.get_scaling.detach().clamp(max=cfg.mcmc_trust_radius)
        L = build_scaling_rotation(scales, gaussians.get_rotation.detach())
        cov = L @ L.transpose(1, 2)

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        base_noise = torch.randn(N, 3, device=device)
        base_noise = base_noise * op_sigmoid(1 - alpha).unsqueeze(-1) * noise_lr * xyz_lr
        noise = torch.bmm(cov, base_noise.unsqueeze(-1)).squeeze(-1)

        # Scale to trust radius
        step_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        noise = noise * (radius.unsqueeze(-1) / step_norm).clamp(max=1.0)

        if visible is not None:
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[visible] = True
            noise = noise * mask.unsqueeze(-1).float()

        return noise

    def compute_maturity_logit(self, terms: GaussianEnergyTerms) -> torch.Tensor:
        """Compute per-Gaussian maturity logit (Phase 10).

        Used as a **detached weight** only — no autograd through the lifecycle.
        High maturity = well-supported, old, conflict-free Gaussian.

        Args:
            terms: Pre-computed :class:`GaussianEnergyTerms`.

        Returns:
            ``[N]`` maturity logit in (-inf, +inf); apply sigmoid for probability.
        """
        cfg = self.config
        logit = torch.zeros_like(terms.alpha)
        logit = logit + cfg.soft_lifecycle_support_k * (terms.support.clamp(0, 1) - 0.5)
        if terms.age_frac is not None:
            logit = logit + cfg.soft_lifecycle_age_k * (terms.age_frac.clamp(0, 1) - 0.5)
        if terms.has_depth_conflict is not None:
            logit = logit - cfg.soft_lifecycle_support_k * terms.has_depth_conflict.float()
        return logit
