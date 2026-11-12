"""Wave 3, PR W3-09: scoring, cell budgets, free-space-consensus wiring
(deliverable 6: fuses stay separate from scoring)."""

from __future__ import annotations

import inspect

import torch

from eligsir.optimization.gaussian.strategies.contracts import SurfaceSeedRequest
from eligsir.optimization.gaussian.strategies.live import mutation_plan as mp_module
from eligsir.optimization.gaussian.strategies.live import scoring as scoring_module
from eligsir.optimization.gaussian.strategies.live.cell_budget import CellBudgetTracker
from eligsir.optimization.gaussian.strategies.live.mutation_plan import SafetyFuses
from eligsir.optimization.gaussian.strategies.live.scoring import (
    free_space_conflict_from_consensus,
    prune_score,
    retire_score,
)


def test_retire_score_penalizes_conflict_and_low_support_rewards_contribution():
    conflict = torch.tensor([0.0, 0.9])
    contribution = torch.tensor([0.9, 0.0])
    support = torch.tensor([10.0, 0.0])
    scores = retire_score(conflict_ema=conflict, contribution_ema=contribution, support_count=support)
    assert scores[1] > scores[0]


def test_prune_score_needs_both_low_confidence_and_high_conflict():
    conf = torch.tensor([0.9, 0.1])
    conflict = torch.tensor([0.9, 0.9])
    scores = prune_score(geometry_confidence=conf, conflict_ema=conflict)
    assert scores[1] > scores[0]


def test_cell_budget_caps_per_cell_and_ranks_by_utility():
    reqs = tuple(
        SurfaceSeedRequest(cell_id=1, utility=float(i), reason="test") for i in range(10)
    )
    tracker = CellBudgetTracker(budget_per_cell=3, max_total_requests=100)
    admitted, rejections = tracker.admit(reqs)
    assert len(admitted) == 3
    assert {r.utility for r in admitted} == {9.0, 8.0, 7.0}
    assert rejections.get("cell_budget:per_cell_cap") == 7


def test_cell_budget_caps_total():
    reqs = tuple(SurfaceSeedRequest(cell_id=i, utility=1.0, reason="t") for i in range(20))
    tracker = CellBudgetTracker(budget_per_cell=100, max_total_requests=5)
    admitted, rejections = tracker.admit(reqs)
    assert len(admitted) == 5
    assert rejections.get("cell_budget:max_total_requests") == 15


def test_free_space_conflict_wired_to_w3_08_consensus_index():
    from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex

    index = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=0.05)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    # 3 distinct, well-separated views vote the same voxel free.
    index.vote_free(pts, view_uid=1, camera_center=torch.tensor([0.0, 0.0, -1.0]))
    index.vote_free(pts, view_uid=2, camera_center=torch.tensor([1.0, 0.0, -1.0]))
    index.vote_free(pts, view_uid=3, camera_center=torch.tensor([0.0, 1.0, -1.0]))

    positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    conflict = free_space_conflict_from_consensus(positions, index, min_views=3)
    assert conflict[0].item() == 1.0  # certified free space -> conflict
    assert conflict[1].item() == 0.0  # never voted -> no conflict


def test_safety_fuses_are_a_separate_module_from_scoring():
    """Deliverable 6, structural check: the SafetyFuses class body itself
    (not the orchestration code that calls both) never references a
    scoring-module function name, and scoring.py never references a fuse
    threshold field name."""
    import re

    fuses_src = inspect.getsource(SafetyFuses)
    for scoring_symbol in ("retire_score", "prune_score", "reseed_utility"):
        pattern = re.compile(rf"\b{re.escape(scoring_symbol)}\s*\(")  # a *call*, not a field name like retire_scores
        assert not pattern.search(fuses_src), f"SafetyFuses calls scoring function {scoring_symbol!r}"
    scoring_src = inspect.getsource(scoring_module)
    for fuse_field in ("max_retire_fraction", "max_prune_fraction", "max_reseed_requests", "max_growth_requests"):
        assert fuse_field not in scoring_src, f"scoring.py references fuse field {fuse_field!r}"


def test_mutation_plan_module_only_calls_fuses_apply_after_scoring():
    """Ordering check: SafetyFuses.apply is called exactly once in
    build_mutation_plan, and it is the last thing called before assembling
    the reasons/plan (i.e. after scoring and cell-budget admission)."""
    source = inspect.getsource(mp_module.build_mutation_plan)
    fuse_call_idx = source.index("fuses.apply(")
    retire_score_idx = source.index("retire_score(")
    cell_budget_idx = source.index(".admit(")
    assert retire_score_idx < fuse_call_idx
    assert cell_budget_idx < fuse_call_idx
