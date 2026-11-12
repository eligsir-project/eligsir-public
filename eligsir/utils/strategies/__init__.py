"""Strategy metadata and lazy compatibility exports."""

from __future__ import annotations

from eligsir.mapping.strategies.registry import (
    STRATEGIES,
    StrategyInfo,
    list_strategies,
    strategy_info,
    validate_strategy_for_layout,
)

_MCMC_EXPORTS = {
    "GsplatEnergyMCMCStrategy",
    "GsplatMCMCBaselineStrategy",
    "ScheduledMCMCStrategy",
    "UpstreamGsplatMCMCStrategy",
    "make_mcmc_strategy",
}


def __getattr__(name: str):
    if name in _MCMC_EXPORTS:
        from eligsir.utils.strategies import mcmc_strategy

        return getattr(mcmc_strategy, name)
    raise AttributeError(name)


__all__ = [
    "STRATEGIES",
    "StrategyInfo",
    "list_strategies",
    "strategy_info",
    "validate_strategy_for_layout",
    *_MCMC_EXPORTS,
]
