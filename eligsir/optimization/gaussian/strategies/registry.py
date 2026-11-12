"""Population-strategy implementation registry (Wave 3, PR W3-09, plan
section 14.3).

Distinct from :mod:`eligsir.mapping.strategies.registry` (the
pre-existing ``--densification_strategy`` CLI metadata table, e.g.
``"gsplat_energy_mcmc"``/``"gsplat_mcmc"``) — that registry governs what a
*training run* may select via config/CLI. This module is a lighter-weight
constructor registry for the population-strategy *classes* introduced in
``eligsir/optimization/gaussian/strategies/`` (currently just
``EliGSiRLiveGaussianStrategy``), used by tests/tooling
(``scripts/w3_09_live_strategy_dry_run.py``) that want to construct one by
name without importing the class path directly.

**Not** wired into ``eligsir.mapping.strategies.registry.STRATEGIES``
or ``utils.strategies.mcmc.registry.make_mcmc_strategy`` — a name being
registered here does not make it selectable via ``--densification_strategy``.

**Update (W3-17, 2026-08):** ``EliGSiRLiveGaussianStrategy`` itself is
now separately reachable via ``--densification_strategy gsplat_live_strategy``
(see ``utils/strategies/mcmc/live_wrapped.py``), but that path does **not**
go through this module's ``_REGISTRY``/``make_population_strategy`` at all
-- ``LiveWrappedMCMCStrategy`` constructs ``EliGSiRLiveGaussianStrategy``
directly. This module remains exactly what it always was: a separate,
lighter-weight constructor registry for tests/tooling
(``scripts/w3_09_live_strategy_dry_run.py``), still with its own
``"eligsir_live_observe_only"`` key, still unrelated to
``--densification_strategy``.
"""

from __future__ import annotations

from typing import Callable

from eligsir.optimization.gaussian.strategies.live.strategy import (
    LiveStrategyConfig,
    EliGSiRLiveGaussianStrategy,
)

_REGISTRY: dict[str, Callable[[], EliGSiRLiveGaussianStrategy]] = {
    "eligsir_live_observe_only": lambda: EliGSiRLiveGaussianStrategy(LiveStrategyConfig()),
}


def list_population_strategies() -> list[str]:
    """Get the list of registered population strategy names.

    Returns:
        A sorted list of strategy names available via make_population_strategy().
    """
    return sorted(_REGISTRY.keys())


def make_population_strategy(name: str) -> EliGSiRLiveGaussianStrategy:
    """Construct a population strategy instance by name.

    Args:
        name: The strategy name (e.g., "eligsir_live_observe_only").

    Returns:
        An initialized EliGSiRLiveGaussianStrategy instance.

    Raises:
        KeyError: If the strategy name is not registered.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown population strategy '{name}'. Available: {list_population_strategies()}"
        ) from exc
    return factory()
