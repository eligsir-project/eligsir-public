"""Decision-trace provenance for realized replay trials."""

import json
from types import SimpleNamespace

from eligsir.contracts.control import (
    DepthInsertionPlan,
    FreshnessMode,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ResolutionTier,
    SupervisionWork,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.runtime.causal_replay import decision_trace_row


def _insertion() -> DepthInsertionPlan:
    return DepthInsertionPlan(None, False, 1, 0, 0, 0.02, 0.0)


def test_decision_trace_preserves_namespaced_realized_replay_trial_identity():
    trial = SourceIdentity("trial-camera", 7, 0, 11)
    other = SourceIdentity("other-camera", 7, 0, 11)
    modality = ModalityEligibility(rgb_supervision=True)
    plan = MappingWorkPlan(
        plan_id="trial-plan",
        snapshot_id=1,
        control_version=1,
        supervision_sources=(),
        replay_sources=(trial, other),
        resolution_tier=ResolutionTier.NATIVE,
        optimizer_updates=1,
        insertion=_insertion(),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=1,
        topology_version=1,
        parameter_version=1,
        supervision_work=(),
        replay_work=tuple(SupervisionWork(source, ResolutionTier.NATIVE, modality) for source in (trial, other)),
        replay_trial_sources=(trial,),
    )

    row = json.loads(json.dumps(decision_trace_row(SimpleNamespace(work_plan=plan))))

    assert row["replay_trial_sources"] == [trial.fingerprint()]
    assert row["replay_trial_source_records"] == [
        {
            "fingerprint": trial.fingerprint(),
            "source_ordinal": 0,
            "timestamp_ns": 11,
        }
    ]
    assert row["replay_trial_sources"] != [other.fingerprint()]


def test_decision_trace_keeps_the_legacy_empty_trial_default():
    row = decision_trace_row(SimpleNamespace(work_plan=None))

    assert row["replay_trial_sources"] == []
    assert row["replay_trial_source_records"] == []
    assert "replay_challenger_trace" not in row
