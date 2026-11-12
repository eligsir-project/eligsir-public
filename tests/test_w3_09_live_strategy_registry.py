"""Wave 3, PR W3-09: registry + adapter + ``utils.strategies`` remains
compat-construction-only (deliverable 2)."""

from __future__ import annotations

import torch

from eligsir.optimization.gaussian.strategies.gsplat_mcmc_adapter import overlap_report, shadow_dead_mask
from eligsir.optimization.gaussian.strategies.live.strategy import EliGSiRLiveGaussianStrategy
from eligsir.optimization.gaussian.strategies.registry import (
    list_population_strategies,
    make_population_strategy,
)


def test_registry_constructs_live_strategy():
    assert "eligsir_live_observe_only" in list_population_strategies()
    strat = make_population_strategy("eligsir_live_observe_only")
    assert isinstance(strat, EliGSiRLiveGaussianStrategy)


def test_registry_unknown_name_raises():
    import pytest

    with pytest.raises(KeyError):
        make_population_strategy("does_not_exist")


def test_utils_strategies_does_not_import_live_package():
    """Deliverable 2: ``utils.strategies`` stays compatibility-construction
    only -- it must not gain a new implementation import pointing at the
    new live-strategy package."""
    import eligsir.utils.strategies as compat_pkg
    import eligsir.utils.strategies.mcmc_strategy as compat_mcmc

    for mod in (compat_pkg, compat_mcmc):
        assert "eligsir.optimization.gaussian.strategies" not in getattr(mod, "__dict__", {})
        for value in vars(mod).values():
            module_name = getattr(value, "__module__", "") or ""
            assert not module_name.startswith("eligsir.optimization.gaussian.strategies"), (
                f"{mod.__name__} re-exports a name from the new live-strategy package: {value!r}"
            )


class _StubGaussiansForAdapter:
    def __init__(self, opacities):
        self._opacity = torch.tensor(opacities).unsqueeze(-1)

    @property
    def get_opacity(self):
        return self._opacity


class _StubArgs:
    mcmc_dead_opacity_end = 0.01


def test_shadow_dead_mask_replicates_threshold_and_never_mutates():
    gaussians = _StubGaussiansForAdapter([0.001, 0.5, 0.005, 0.9])
    mask = shadow_dead_mask(gaussians, _StubArgs())
    assert mask.tolist() == [True, False, True, False]


def test_shadow_dead_mask_fails_open_on_bad_input():
    class _BrokenGaussians:
        pass

    mask = shadow_dead_mask(_BrokenGaussians(), _StubArgs())
    assert mask.numel() == 0


def test_overlap_report_counts():
    shadow = torch.tensor([True, False, True, False, True])
    proposed = torch.tensor([0, 1, 4], dtype=torch.int64)
    report = overlap_report(proposed, shadow)
    assert report["proposed_count"] == 3
    assert report["shadow_count"] == 3
    assert report["overlap_count"] == 2  # indices 0 and 4
    assert report["proposed_only_count"] == 1
    assert report["shadow_only_count"] == 1
