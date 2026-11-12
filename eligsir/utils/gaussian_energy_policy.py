"""Compatibility shim.

``utils/gaussian_energy_policy.py`` moved (Wave 2 Task 5, 2026-08-05) to
``utils/strategies/mcmc/energy_policy.py`` — it is heavily used by, and
tightly coupled to, ``utils/strategies/mcmc/{energy,legacy}.py`` (every
call site does a local ``from eligsir.utils.gaussian_energy_policy import ...``
inside those two modules), so it belongs alongside them as a sibling in
the ``mcmc`` package rather than staying in the flat ``utils/`` directory.
This module re-exports the same public names so existing callers/tests
keep working unchanged — matches this project's existing compat-shim
convention (see ``gaussian_renderer/gsplat_backend.py``,
``utils/strategies/mcmc_strategy.py``).
"""

from eligsir.utils.strategies.mcmc.energy_policy import (
    DeadMaskResult,
    GaussianEnergyConfig,
    GaussianEnergyPolicy,
    GaussianEnergyTerms,
    PopulationEnergyResult,
    energy_config_from_args,
)

__all__ = [
    "DeadMaskResult",
    "GaussianEnergyConfig",
    "GaussianEnergyPolicy",
    "GaussianEnergyTerms",
    "PopulationEnergyResult",
    "energy_config_from_args",
]
