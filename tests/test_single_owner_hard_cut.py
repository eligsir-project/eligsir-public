"""Regression tests for the #331 single-owner hard cut."""

from types import SimpleNamespace

import pytest

from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend
from eligsir.utils.strategies.mcmc.registry import make_mcmc_strategy


class _Model:
    pass


class _GsplatSelection:
    uses_gsplat_layout = True


@pytest.mark.parametrize("backend_cls", [GsplatMapBackend, LegacyGraphdecoMapBackend])
def test_map_backend_is_canonical_per_live_model(backend_cls):
    model = _Model()
    first = backend_cls(model)
    first._map_content_version = 7
    first._topology_version = 3

    second = backend_cls(model)

    assert second is first
    assert second.map_content_version == 7
    assert second.topology_version == 3


@pytest.mark.parametrize(
    "flag",
    [
        "streaming_live_strategy_phase_b_retirement",
        "streaming_live_strategy_phase_c_reseeding",
        "streaming_live_strategy_phase_d_cell_budgets",
        "streaming_live_strategy_phase_e_constrained_movement",
    ],
)
def test_production_factory_rejects_competing_live_mutation_phase(flag):
    args = SimpleNamespace(**{flag: True})

    with pytest.raises(ValueError, match="single-owner Gaussian control"):
        make_mcmc_strategy(
            "gsplat_live_strategy",
            gaussians=_GsplatSelection(),
            args=args,
        )


def test_unrelated_strategy_does_not_consult_retired_live_phase_flags():
    strategy = make_mcmc_strategy(
        "gsplat_energy_mcmc",
        gaussians=_GsplatSelection(),
        args=SimpleNamespace(streaming_live_strategy_phase_b_retirement=True),
    )
    assert strategy.name == "gsplat_energy_mcmc"
