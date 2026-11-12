"""Compatibility shim.

``utils/strategies/mcmc_strategy.py`` was split (Wave 2 Task 4, 2026-08-05)
into ``utils/strategies/mcmc/{registry,base,legacy,baseline,energy,proposal}.py``.
This module re-exports the same public (and a few historically-imported
private) names so existing callers/tests keep working unchanged — matches
this project's existing compat-shim convention (see
``gaussian_renderer/gsplat_backend.py``).
"""

from eligsir.utils.strategies.mcmc.base import (
    _BUBBLE_MIN_ACTIVE,
    _augment_dead_mask_knn_isolation,
    _bubble_active_mask,
    _bubble_feature_enabled,
    _effective_cap_max,
    _exclude_locked_from_mask,
    _gate_dead_mask_to_visible,
    _locked_depth_mask,
)
from eligsir.utils.strategies.mcmc.baseline import (
    GsplatDefaultStrategy,
    GsplatMCMCBaselineStrategy,
    UpstreamGsplatMCMCStrategy,
)
from eligsir.utils.strategies.mcmc.energy import GsplatEnergyMCMCStrategy
from eligsir.utils.strategies.mcmc.legacy import ScheduledMCMCStrategy
from eligsir.utils.strategies.mcmc.registry import make_mcmc_strategy

__all__ = [
    "_BUBBLE_MIN_ACTIVE",
    "_augment_dead_mask_knn_isolation",
    "_bubble_active_mask",
    "_bubble_feature_enabled",
    "_effective_cap_max",
    "_exclude_locked_from_mask",
    "_gate_dead_mask_to_visible",
    "_locked_depth_mask",
    "GsplatDefaultStrategy",
    "GsplatEnergyMCMCStrategy",
    "GsplatMCMCBaselineStrategy",
    "ScheduledMCMCStrategy",
    "UpstreamGsplatMCMCStrategy",
    "make_mcmc_strategy",
]
