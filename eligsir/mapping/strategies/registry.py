"""Strategy metadata registry.

This module is intentionally lightweight: it does not import torch or the large
strategy implementations. Construction still lives in ``utils.strategies`` until
the strategy files are split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    status: str
    model_layouts: FrozenSet[str]
    online_safe: bool
    offline_only: bool = False


STRATEGIES: dict[str, StrategyInfo] = {
    "gsplat_energy_mcmc": StrategyInfo(
        name="gsplat_energy_mcmc",
        status="stable",
        model_layouts=frozenset({"gsplat"}),
        online_safe=True,
    ),
    "gsplat_mcmc": StrategyInfo(
        name="gsplat_mcmc",
        status="experimental",
        model_layouts=frozenset({"gsplat", "legacy"}),
        online_safe=True,
    ),
    "gsplat_default": StrategyInfo(
        name="gsplat_default",
        status="experimental",
        model_layouts=frozenset({"gsplat"}),
        online_safe=False,
        offline_only=True,
    ),
    "gsplat_live_strategy": StrategyInfo(
        # W3-17 wiring (2026-08): wraps gsplat_energy_mcmc for all real
        # mutation and additionally runs the Wave 3 (W3-09/W3-10) live
        # population-strategy class as a shadow observer. With every
        # streaming_live_strategy_phase_* flag off (the default), behavior
        # is identical to gsplat_energy_mcmc -- see
        # utils/strategies/mcmc/live_wrapped.py.
        name="gsplat_live_strategy",
        status="experimental",
        model_layouts=frozenset({"gsplat"}),
        online_safe=True,
    ),
    "mcmc": StrategyInfo(
        name="mcmc",
        status="legacy",
        model_layouts=frozenset({"legacy"}),
        online_safe=False,
    ),
    "hybrid": StrategyInfo(
        name="hybrid",
        status="legacy",
        model_layouts=frozenset({"legacy"}),
        online_safe=False,
    ),
    "taming": StrategyInfo(
        name="taming",
        status="legacy",
        model_layouts=frozenset({"legacy"}),
        online_safe=False,
        offline_only=True,
    ),
}


def list_strategies(*, include_legacy: bool = True) -> list[str]:
    """Return registered strategy names."""

    names = [
        name
        for name, info in STRATEGIES.items()
        if include_legacy or info.status != "legacy"
    ]
    return sorted(names)


def strategy_info(name: str) -> StrategyInfo:
    """Return metadata for a strategy name."""

    key = str(name)
    try:
        return STRATEGIES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown strategy '{key}'. Available: {list_strategies()}") from exc


def validate_strategy_for_layout(name: str, model_layout: str) -> StrategyInfo:
    """Validate strategy compatibility with a model layout."""

    info = strategy_info(name)
    layout = str(model_layout)
    if layout not in info.model_layouts:
        allowed = ", ".join(sorted(info.model_layouts))
        raise ValueError(f"Strategy '{name}' requires model_layout in {{{allowed}}}; got '{layout}'.")
    return info
