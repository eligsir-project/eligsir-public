from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from eligsir.map.arbiter import ProposalKey
from eligsir.map.row_state import GaussianRowStateRegistry, RowStateDescriptor
from eligsir.mapping.evidence.population_transaction import (
    PopulationEvidenceTransaction,
    PopulationEvidenceTransactionResult,
)
from eligsir.mapping.evidence.surface_targets import SurfaceTargetBatch, SurfaceTargetSnapshot
from eligsir.mapping.population_plan_adapter import PopulationPlanAdapterContext
from eligsir.mapping.population_policy import GaussianPopulationPolicy, PopulationEvidence, MutationWorkToken
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.utils.gaussian_belief import (
    ExistenceBelief, GaussianBelief, GeometryBelief, ObservabilityBelief,
    ProvenanceBelief, RepresentationBelief, TemporalBelief,
)


def test_transaction_has_no_pre_fence_tensor_bool_or_item_conversion():
    source = inspect.getsource(PopulationEvidenceTransaction)
    assert ".item(" not in source
    assert "bool(" not in source
    assert ".tolist(" not in source
    assert 'device.type != "cpu"' not in source


def _token() -> EvidenceSnapshotToken:
    return EvidenceSnapshotToken(1, 2, 3, 4, 5, 6, 7, 8)


def _belief() -> GaussianBelief:
    n = 2
    z = torch.zeros(n, 3)
    return GaussianBelief(
        provenance=ProvenanceBelief(torch.arange(n), torch.zeros(n, dtype=torch.int32), torch.zeros(n, dtype=torch.int8), torch.full((n,), -1), torch.arange(n), torch.zeros(n, dtype=torch.int32), z),
        observability=ObservabilityBelief(torch.full((n,), 2, dtype=torch.int32), torch.full((n,), 2, dtype=torch.int16), torch.ones(n), None, torch.zeros(n, dtype=torch.int32), None),
        existence=ExistenceBelief(torch.ones(n), torch.zeros(n), torch.ones(n)),
        geometry=GeometryBelief(torch.full((n,), -1), torch.ones(n), torch.zeros(n), torch.zeros(n)),
        representation=RepresentationBelief(torch.ones(n), torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.tensor([1.0, 0.9])),
        temporal=TemporalBelief(torch.zeros(n, dtype=torch.int32), torch.zeros(n), torch.zeros(n)), n=n,
    )


def _registry() -> GaussianRowStateRegistry:
    fields = (
        ("params.means", "params", torch.float32, (3,)), ("params.scales", "params", torch.float32, (3,)),
        ("params.quats", "params", torch.float32, (4,)), ("params.opacities", "params", torch.float32, ()),
        ("params.sh0", "params", torch.float32, (1, 3)), ("params.shN", "params", torch.float32, (0, 3)),
        ("optimizer.means.exp_avg", "optimizer", torch.float32, (3,)), ("belief.p_exist", "belief", torch.float32, ()),
        ("provenance.gaussian_uid", "provenance", torch.int64, ()), ("surface.anchor_xyz", "surface", torch.float32, (3,)),
    )
    return GaussianRowStateRegistry(RowStateDescriptor(name=n, family=f, owner="test", shape_suffix=s, dtype=d, checkpoint=True, default="zero", create="zero", grow="zero", relocate="zero", retire="select_survivors") for n, f, d, s in fields)


def _target(token: EvidenceSnapshotToken, *, independent_views: int = 2) -> SurfaceTargetBatch:
    return SurfaceTargetBatch(
        snapshot=SurfaceTargetSnapshot(token.map_content, token.topology, token.coordinate_frame, token.schema, token.surface_evidence),
        positions=torch.tensor([[1.0, 2.0, 3.0]]), normals=torch.tensor([[0.0, 0.0, 1.0]]),
        uncertainty=torch.full((1, 3), 0.01), support_evidence=torch.ones(1), conflict_evidence=torch.zeros(1),
        independent_view_count=torch.full((1,), independent_views, dtype=torch.int32), repeated_exposure_count=torch.ones(1, dtype=torch.int32),
        target_ids=torch.tensor([99]), source_modality=torch.tensor([4], dtype=torch.int8), max_targets=1,
    )


def _context(token: EvidenceSnapshotToken) -> PopulationPlanAdapterContext:
    return PopulationPlanAdapterContext(token, token, 4, ("10", "11"), 12, _registry(), ProposalKey(1, 2, 3, "population"), 1, DeviceTransferOwner("cpu"))


def test_transaction_builds_one_create_plan_from_qualified_b4_target():
    token = _token()
    evidence = PopulationEvidence(torch.tensor([1.0, 0.0]), torch.zeros(2), torch.zeros(2))
    result = PopulationEvidenceTransaction().build(
        belief=_belief(), evidence=evidence, work_token=MutationWorkToken(8, 1),
        snapshot=token, target=_target(token), context=_context(token),
    )
    assert isinstance(result, PopulationEvidenceTransactionResult)
    assert result.reason == "CREATE"
    assert result.proposal is not None
    assert result.proposal.plan.action_count == 1
    assert result.proposal.plan.created_hypothesis_uids == ("12",)


def test_transaction_abstains_on_stale_snapshot_without_running_update():
    token = _token()
    calls = []
    result = PopulationEvidenceTransaction().build(
        belief=_belief(), evidence=PopulationEvidence(torch.ones(2), torch.zeros(2), torch.zeros(2)),
        work_token=MutationWorkToken(8, 1), snapshot=token, target=_target(token),
        context=_context(EvidenceSnapshotToken(1, 9, 3, 4, 5, 6, 7, 8)),
        evidence_update=lambda: calls.append(True),
    )
    assert result.proposal is None
    assert result.reason == "ABSTAIN_STALE_SNAPSHOT"
    assert calls == []


def test_one_view_target_abstains_at_the_mutation_fence_after_fixed_topology_update():
    token = _token()
    calls = []

    result = PopulationEvidenceTransaction().build(
        belief=_belief(), evidence=PopulationEvidence(torch.ones(2), torch.zeros(2), torch.zeros(2)),
        work_token=MutationWorkToken(8, 1), snapshot=token, target=_target(token, independent_views=1),
        context=_context(token), evidence_update=lambda: calls.append(True),
    )
    assert result.proposal is None
    assert result.reason == "ABSTAIN_INVALID_TARGET_PAYLOAD"
    assert calls == [True]


def test_transaction_rejects_target_batches_other_than_k1():
    token = _token()
    target = _target(token)
    target = SurfaceTargetBatch(
        snapshot=target.snapshot, positions=torch.cat((target.positions, target.positions)),
        normals=torch.cat((target.normals, target.normals)), uncertainty=torch.cat((target.uncertainty, target.uncertainty)),
        support_evidence=torch.ones(2), conflict_evidence=torch.zeros(2), independent_view_count=torch.ones(2, dtype=torch.int32),
        repeated_exposure_count=torch.ones(2, dtype=torch.int32), target_ids=torch.tensor([99, 100]), source_modality=torch.tensor([4, 4], dtype=torch.int8), max_targets=2,
    )
    result = PopulationEvidenceTransaction().build(belief=_belief(), evidence=PopulationEvidence(torch.ones(2), torch.zeros(2), torch.zeros(2)), work_token=MutationWorkToken(8, 1), snapshot=token, target=target, context=_context(token))
    assert result.proposal is None
    assert result.reason == "ABSTAIN_TARGET_BOUND"
