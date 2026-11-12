"""Wave 3, PR W3-09: typed/versioned plan contracts (deliverables 3, 4)."""

from __future__ import annotations

import pytest
import torch

from eligsir.optimization.gaussian.strategies.contracts import (
    PLAN_CONTRACT_VERSION,
    LifecycleUpdateBatch,
    MutationReasonSummary,
    PopulationMutationPlan,
    SurfaceSeedRequest,
)


def _empty_batch():
    empty_long = torch.empty(0, dtype=torch.int64)
    empty_i8 = torch.empty(0, dtype=torch.int8)
    return LifecycleUpdateBatch(
        geometry_indices=empty_long, geometry_old_state=empty_i8, geometry_new_state=empty_i8,
        geometry_reasons=(),
        appearance_indices=empty_long, appearance_old_state=empty_i8, appearance_new_state=empty_i8,
        appearance_reasons=(),
    )


def test_plan_is_versioned():
    plan = PopulationMutationPlan(
        retire_indices=torch.empty(0, dtype=torch.int64),
        prune_indices=torch.empty(0, dtype=torch.int64),
        reseed_requests=(),
        growth_requests=(),
        lifecycle_updates=_empty_batch(),
        reasons=MutationReasonSummary(),
    )
    assert plan.version == PLAN_CONTRACT_VERSION
    assert plan.observe_only is True


def test_plan_rejects_unknown_version():
    with pytest.raises(ValueError):
        PopulationMutationPlan(
            retire_indices=torch.empty(0, dtype=torch.int64),
            prune_indices=torch.empty(0, dtype=torch.int64),
            reseed_requests=(), growth_requests=(),
            lifecycle_updates=_empty_batch(), reasons=MutationReasonSummary(),
            version="99.0",
        )


def test_lifecycle_update_batch_requires_one_reason_per_transition():
    idx = torch.tensor([0, 1], dtype=torch.int64)
    codes = torch.tensor([0, 0], dtype=torch.int8)
    with pytest.raises(ValueError):
        LifecycleUpdateBatch(
            geometry_indices=idx, geometry_old_state=codes, geometry_new_state=codes,
            geometry_reasons=("only_one_reason",),
            appearance_indices=torch.empty(0, dtype=torch.int64),
            appearance_old_state=torch.empty(0, dtype=torch.int8),
            appearance_new_state=torch.empty(0, dtype=torch.int8),
            appearance_reasons=(),
        )


def test_surface_seed_request_carries_a_reason():
    req = SurfaceSeedRequest(cell_id=3, utility=0.5, reason="under_supported_mass")
    assert req.reason == "under_supported_mass"


def test_reason_summary_keeps_fuse_triggers_separate_from_scoring_reasons():
    summary = MutationReasonSummary(
        retire_reasons={"free_space_conflict_or_low_contribution": 3},
        safety_fuse_triggers={"safety_fuse:max_retire_fraction": 1},
    )
    d = summary.as_dict()
    assert "safety_fuse:max_retire_fraction" in d["safety_fuse_triggers"]
    assert "safety_fuse:max_retire_fraction" not in d["retire_reasons"]
