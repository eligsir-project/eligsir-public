"""Wave 3, PR W3-10: pure-CPU tests for the Phase D tiered cell budget and
the Phase A-E ``ActivationPhaseConfig`` defaults. No GPU/gsplat needed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.strategies.contracts import SurfaceSeedRequest  # noqa: E402
from eligsir.optimization.gaussian.strategies.live.backend import ActivationPhaseConfig  # noqa: E402
from eligsir.optimization.gaussian.strategies.live.cell_budget import TieredCellBudget  # noqa: E402


def test_activation_phase_config_defaults_all_off():
    cfg = ActivationPhaseConfig()
    assert cfg.phase_a_lifecycle is False
    assert cfg.phase_b_retirement is False
    assert cfg.phase_c_reseeding is False
    assert cfg.phase_d_cell_budgets is False
    assert cfg.phase_e_constrained_movement is False
    assert cfg.any_active() is False


def test_activation_phase_config_any_active_true_when_one_set():
    cfg = ActivationPhaseConfig(phase_b_retirement=True)
    assert cfg.any_active() is True


def _req(cell_id: int, utility: float, reason: str = "x") -> SurfaceSeedRequest:
    return SurfaceSeedRequest(cell_id=cell_id, utility=utility, reason=reason)


def test_tiered_cell_budget_admits_up_to_total_per_cell():
    budget = TieredCellBudget(minimum=1, normal=1, temporary=1, reserve=1)  # total 4
    requests = tuple(_req(0, float(i)) for i in range(10))
    admitted, rejections = budget.admit(requests)
    assert len(admitted) == 4
    assert rejections.get("cell_budget:tiered_per_cell_cap") == 6


def test_tiered_cell_budget_ranks_by_utility_within_cell():
    budget = TieredCellBudget(minimum=1, normal=0, temporary=0, reserve=0)  # total 1
    requests = (_req(0, 1.0), _req(0, 9.0), _req(0, 5.0))
    admitted, _ = budget.admit(requests)
    assert len(admitted) == 1
    assert admitted[0].utility == 9.0, "highest-utility request must win the single slot"


def test_tiered_cell_budget_respects_max_total_requests_across_cells():
    budget = TieredCellBudget(minimum=5, normal=5, temporary=5, reserve=5, max_total_requests=2)
    requests = (_req(0, 1.0), _req(1, 2.0), _req(2, 3.0))
    admitted, rejections = budget.admit(requests)
    assert len(admitted) == 2
    assert rejections.get("cell_budget:max_total_requests") == 1


def test_tiered_cell_budget_isolates_cells():
    budget = TieredCellBudget(minimum=1, normal=0, temporary=0, reserve=0)
    requests = (_req(0, 1.0), _req(1, 1.0), _req(2, 1.0))
    admitted, rejections = budget.admit(requests)
    assert len(admitted) == 3, "each cell gets its own independent tier budget"
    assert rejections == {}
