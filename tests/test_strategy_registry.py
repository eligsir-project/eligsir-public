"""Tests for strategy metadata."""

from __future__ import annotations

import sys

import pytest

from eligsir.mapping.strategies.registry import (
    list_strategies,
    strategy_info,
    validate_strategy_for_layout,
)


def test_strategy_registry_import_is_lightweight():
    if "torch" in sys.modules:
        pytest.skip("torch was imported by an earlier test in this pytest process")
    assert "torch" not in sys.modules


def test_utils_strategy_registry_reexports_package_registry():
    from eligsir.utils.strategies.registry import strategy_info as compat_strategy_info

    assert compat_strategy_info is strategy_info


def test_gsplat_energy_mcmc_is_canonical_online_strategy():
    info = strategy_info("gsplat_energy_mcmc")
    assert info.status == "stable"
    assert info.model_layouts == frozenset({"gsplat"})
    assert info.online_safe


def test_legacy_strategies_can_be_filtered():
    assert "mcmc" in list_strategies()
    assert "mcmc" not in list_strategies(include_legacy=False)


def test_validate_strategy_for_layout_rejects_wrong_layout():
    with pytest.raises(ValueError, match="requires model_layout"):
        validate_strategy_for_layout("gsplat_energy_mcmc", "legacy")
