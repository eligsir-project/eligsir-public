"""Strategy-name dispatch under the single-owner Gaussian mutation contract."""

from .baseline import GsplatDefaultStrategy, GsplatMCMCBaselineStrategy, UpstreamGsplatMCMCStrategy
from .energy import GsplatEnergyMCMCStrategy
from .legacy import ScheduledMCMCStrategy
from .arbiter_owned import ensure_arbiter_owned


_COMPETING_LIVE_PHASE_FLAGS = (
    "streaming_live_strategy_phase_b_retirement",
    "streaming_live_strategy_phase_c_reseeding",
    "streaming_live_strategy_phase_d_cell_budgets",
    "streaming_live_strategy_phase_e_constrained_movement",
)


def _reject_competing_live_phase_flags(args) -> None:
    enabled = [name for name in _COMPETING_LIVE_PHASE_FLAGS if bool(getattr(args, name, False))]
    if enabled:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in enabled)
        raise ValueError(
            "live-strategy activation phases B-E are retired by the single-owner Gaussian "
            f"control hard cut (#331). Enabled: {rendered}. "
            "Use gsplat_live_strategy with phase A only for lifecycle evidence/telemetry."
        )


def _owned(strategy, gaussians, args):
    if gaussians is None:
        raise ValueError("production MCMC construction requires the live Gaussian model")
    return ensure_arbiter_owned(strategy, gaussians, args)


def make_mcmc_strategy(
    name,
    gaussians=None,
    args=None,
    *,
    semantic_cadence: bool = False,
):
    """Construct the selected strategy with one mutation-policy owner.

    Raw strategy objects are no longer returned from production dispatch.
    Their mutating ``step_post_backward`` tail is always serialized through
    :class:`GaussianMutationArbiter`; non-mutating pre-backward/fixed-shape
    hooks delegate unchanged.
    """
    if type(semantic_cadence) is not bool:
        raise TypeError("semantic_cadence must be exact bool")
    if name == "gsplat_live_strategy":
        if not getattr(gaussians, "uses_gsplat_layout", False):
            raise ValueError("--densification_strategy gsplat_live_strategy requires --model_layout gsplat.")
        _reject_competing_live_phase_flags(args)
        from .live_wrapped import make_live_wrapped_strategy
        return _owned(
            make_live_wrapped_strategy(
                gaussians=gaussians,
                args=args,
                semantic_cadence=semantic_cadence,
            ),
            gaussians,
            args,
        )
    if name == "gsplat_energy_mcmc" and getattr(gaussians, "uses_gsplat_layout", False):
        return _owned(
            GsplatEnergyMCMCStrategy(semantic_cadence=semantic_cadence),
            gaussians,
            args,
        )
    if name == "gsplat_energy_mcmc":
        raise ValueError("--densification_strategy gsplat_energy_mcmc requires --model_layout gsplat.")
    if name == "gsplat_mcmc" and getattr(gaussians, "uses_gsplat_layout", False):
        return _owned(
            UpstreamGsplatMCMCStrategy(semantic_cadence=semantic_cadence),
            gaussians,
            args,
        )
    if name == "gsplat_mcmc":
        return _owned(GsplatMCMCBaselineStrategy(), gaussians, args)
    if name == "gsplat_default":
        if not getattr(gaussians, "uses_gsplat_layout", False):
            raise ValueError("--densification_strategy gsplat_default requires --model_layout gsplat.")
        return _owned(GsplatDefaultStrategy(), gaussians, args)
    return _owned(ScheduledMCMCStrategy(name), gaussians, args)


__all__ = ["make_mcmc_strategy"]
