from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from eligsir.map.arbiter import (
    GaussianMutationArbiter,
    PopulationMutationProposal,
    ProposalKey,
    SurfaceInsertionProposal,
)
from eligsir.map.mutation import GaussianInsertionBatch, TopologyChange
from eligsir.map.population_mutation_plan import PopulationMutationPlan
from eligsir.map.population_target_payload import (
    PopulationTargetOpaqueField,
    PopulationTargetPayloadBatch,
    PopulationTargetPayloadSchema,
)
from eligsir.map.row_state import (
    PopulationRowRelocation,
    PopulationRowTransform,
    RowStateDescriptor,
)
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken


def _token(**overrides) -> EvidenceSnapshotToken:
    values = dict(
        map_content=4, topology=7, coordinate_frame=2, schema=1,
        optimizer_update=11, surface_evidence=3, policy=5, parameter=9,
    )
    values.update(overrides)
    return EvidenceSnapshotToken(**values)


def _registry_snapshot() -> tuple[RowStateDescriptor, ...]:
    return tuple(
        RowStateDescriptor(
            name=name,
            family="canonical",
            owner="arbiter",
            shape_suffix=(),
            dtype=torch.float32,
            checkpoint=True,
            default=0,
            create="fresh",
            grow="fresh",
            relocate="fresh",
            retire="gather",
        )
        for name in sorted(PopulationMutationPlan.REQUIRED_RESET_FIELDS)
    )


def _payload() -> PopulationTargetPayloadBatch:
    schema = PopulationTargetPayloadSchema(
        opaque_fields=(
            PopulationTargetOpaqueField(
                name="lineage_source", dtype=torch.int64, shape_suffix=()
            ),
        )
    )
    return PopulationTargetPayloadBatch(
        schema=schema,
        positions=torch.tensor([[1.0, 2.0, 3.0]]),
        colors=torch.tensor([[0.1, 0.2, 0.3]]),
        raw_scales=torch.tensor([[0.0, 0.0, 0.0]]),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacity_logits=torch.tensor([[0.0]]),
        sh_rest=torch.empty((1, 0)),
        opaque_values=(torch.tensor([17], dtype=torch.int64),),
        max_targets=2,
    )


def _plan(payload: PopulationTargetPayloadBatch | None = None) -> PopulationMutationPlan:
    payload = _payload() if payload is None else payload
    before = _token()
    return PopulationMutationPlan(
        snapshot_token=before,
        next_snapshot_token=_token(
            map_content=5, topology=8, optimizer_update=12, parameter=10
        ),
        row_transform=PopulationRowTransform(
            old_population=2,
            new_population=2,
            survivor_old_indices=torch.tensor([1]),
            retired_old_indices=torch.tensor([0]),
            created_rows=(),
            relocated_rows=(
                PopulationRowRelocation(
                    row_index=1,
                    donor_row_index=0,
                    old_hypothesis_uid="old-0",
                    new_hypothesis_uid="fresh-0",
                ),
            ),
            covered_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS),
        ),
        survivor_hypothesis_uids=("survivor-1",),
        mutation_fence_reason="MUTATION_CARDINALITY",
        mutation_fence_snapshot=before,
        row_state_registry_snapshot=_registry_snapshot(),
        capacity=2,
        old_population=2,
        new_population=2,
        retire_rows=torch.empty(0, dtype=torch.int64),
        relocate_donor_rows=torch.tensor([0], dtype=torch.int64),
        relocate_target_rows=torch.tensor([1], dtype=torch.int64),
        create_rows=torch.empty(0, dtype=torch.int64),
        target_payload=payload.positions.clone(),
        donor_hypothesis_uids=("old-0",),
        relocated_hypothesis_uids=("fresh-0",),
        created_hypothesis_uids=(),
        reset_fields=frozenset(PopulationMutationPlan.REQUIRED_RESET_FIELDS),
        max_actions=1,
    )


def _proposal(
    *, payload: PopulationTargetPayloadBatch | None = None,
    plan: PopulationMutationPlan | None = None,
) -> PopulationMutationProposal:
    payload = _payload() if payload is None else payload
    plan = _plan(payload) if plan is None else plan
    return PopulationMutationProposal(
        key=ProposalKey(3, 1, 42, "population-policy"),
        plan=plan,
        target_payload=payload,
        row_state_registry_snapshot=plan.row_state_registry_snapshot,
    )


class _PopulationTxn:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.apply_calls = 0
        self.commit_calls = 0

    def apply_population_plan(self, plan, target_payload, registry_snapshot) -> None:
        self.apply_calls += 1
        self.backend.applied = (plan, target_payload, registry_snapshot)
        self.backend.population = plan.new_population

    def current_population_plan_registry_snapshot(self):
        return _registry_snapshot()

    def validate_population_plan(self, plan, target_payload, registry_snapshot) -> None:
        assert plan.row_state_registry_snapshot == registry_snapshot
        assert target_payload.count == 1

    def insert(self, *_args, **_kwargs):
        raise AssertionError("population proposal must not decompose into insert")

    def prune(self, *_args, **_kwargs):
        raise AssertionError("population proposal must not decompose into prune")

    def relocate(self, *_args, **_kwargs):
        raise AssertionError("population proposal must not decompose into relocate")

    def commit(self) -> TopologyChange:
        self.commit_calls += 1
        self.backend.snapshot = self.backend.applied[0].next_snapshot_token
        return TopologyChange(
            map_version=5,
            topology_version=8,
            inserted=0,
            pruned=0,
            relocated=1,
            content_changed=True,
        )


class _PopulationBackend:
    def __init__(self, snapshot: EvidenceSnapshotToken | None = None) -> None:
        self.population = 2
        self.snapshot = _token() if snapshot is None else snapshot
        self.transaction = None
        self.applied = None

    @property
    def population_count(self) -> int:
        return self.population

    def current_population_plan_snapshot(self) -> EvidenceSnapshotToken:
        return self.snapshot

    def begin_mutation(self):
        self.transaction = _PopulationTxn(self)
        return self.transaction

    def commit_mutation(self, txn):
        return txn.commit()


def test_population_plan_crosses_one_atomic_backend_apply_call():
    backend = _PopulationBackend()
    proposal = _proposal()

    result = GaussianMutationArbiter(backend).commit((proposal,))

    assert backend.transaction.apply_calls == 1
    assert backend.transaction.commit_calls == 1
    assert backend.applied == (
        proposal.plan,
        proposal.target_payload,
        proposal.row_state_registry_snapshot,
    )
    assert result.submitted == result.applied == 1
    assert result.inserted == 0


def test_population_plan_is_rejected_when_mixed_with_legacy_typed_mutation():
    backend = _PopulationBackend()
    proposal = _proposal()
    insertion = SurfaceInsertionProposal(
        ProposalKey(3, 2, 43, "legacy"),
        GaussianInsertionBatch(points=torch.ones(1, 3), colors=torch.ones(1, 3)),
    )

    with pytest.raises(ValueError, match="committed alone"):
        GaussianMutationArbiter(backend).commit((proposal, insertion))

    assert backend.transaction is None


def test_population_plan_fails_closed_without_atomic_backend_capability():
    class _UnsupportedTxn(_PopulationTxn):
        def apply_population_plan(self, *_args, **_kwargs):
            raise AttributeError

    class _UnsupportedTxnWithoutApply:
        def __init__(self, backend):
            self.backend = backend

        def current_population_plan_registry_snapshot(self):
            return _registry_snapshot()

        def validate_population_plan(self, *_args):
            return None

        def commit(self):
            raise AssertionError("unsupported plan must not commit")

    class _UnsupportedBackend(_PopulationBackend):
        def begin_mutation(self):
            self.transaction = _UnsupportedTxnWithoutApply(self)
            return self.transaction

    backend = _UnsupportedBackend()
    with pytest.raises(TypeError, match="apply_population_plan"):
        GaussianMutationArbiter(backend).commit((_proposal(),))


def test_population_plan_rejects_stale_snapshot_before_backend_apply():
    backend = _PopulationBackend(snapshot=_token(parameter=8))

    with pytest.raises(ValueError, match="stale|snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(),))

    assert backend.transaction is None


def test_population_plan_requires_authoritative_registry_attestation_before_apply():
    class _MissingRegistryTxn(_PopulationTxn):
        current_population_plan_registry_snapshot = None

    class _MissingRegistryBackend(_PopulationBackend):
        def begin_mutation(self):
            self.transaction = _MissingRegistryTxn(self)
            return self.transaction

    backend = _MissingRegistryBackend()
    with pytest.raises(TypeError, match="registry snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction.apply_calls == 0


def test_population_plan_accepts_backend_registry_attestation_when_transaction_is_thin():
    class _ThinTxn(_PopulationTxn):
        current_population_plan_registry_snapshot = None

    class _BackendAttestation(_PopulationBackend):
        def current_population_plan_registry_snapshot(self):
            return _registry_snapshot()

        def begin_mutation(self):
            self.transaction = _ThinTxn(self)
            return self.transaction

    backend = _BackendAttestation()
    GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction.apply_calls == 1


def test_population_plan_rejects_mismatched_authoritative_registry_before_apply():
    class _MismatchedRegistryTxn(_PopulationTxn):
        def current_population_plan_registry_snapshot(self):
            return (
                replace(_registry_snapshot()[0], name="wrong-field"),
            )

    class _MismatchedRegistryBackend(_PopulationBackend):
        def begin_mutation(self):
            self.transaction = _MismatchedRegistryTxn(self)
            return self.transaction

    backend = _MismatchedRegistryBackend()
    with pytest.raises(ValueError, match="registry snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction.apply_calls == 0


def test_population_plan_requires_backend_validation_before_apply():
    class _NoValidatorTxn(_PopulationTxn):
        validate_population_plan = None

    class _NoValidatorBackend(_PopulationBackend):
        def begin_mutation(self):
            self.transaction = _NoValidatorTxn(self)
            return self.transaction

    backend = _NoValidatorBackend()
    with pytest.raises(TypeError, match="validate_population_plan"):
        GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction.apply_calls == 0


def test_population_plan_rejects_static_post_commit_version_attestation():
    class _StaticVersionTxn(_PopulationTxn):
        def commit(self) -> TopologyChange:
            self.commit_calls += 1
            return TopologyChange(
                map_version=5,
                topology_version=8,
                relocated=1,
                content_changed=True,
            )

    class _StaticVersionBackend(_PopulationBackend):
        def begin_mutation(self):
            self.transaction = _StaticVersionTxn(self)
            return self.transaction

    backend = _StaticVersionBackend()
    with pytest.raises(RuntimeError, match="post-mutation snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction.apply_calls == 1


def test_population_plan_requires_authoritative_pre_and_post_version_provider():
    class _NoVersionBackend(_PopulationBackend):
        current_population_plan_snapshot = None

    backend = _NoVersionBackend()
    with pytest.raises(TypeError, match="current_population_plan_snapshot"):
        GaussianMutationArbiter(backend).commit((_proposal(),))
    assert backend.transaction is None


def test_population_plan_rejects_payload_that_disagrees_with_plan_positions():
    payload = _payload()
    plan = _plan(payload)
    bad_payload = replace(payload, positions=torch.tensor([[9.0, 9.0, 9.0]]))

    with pytest.raises(ValueError, match="positions"):
        _proposal(payload=bad_payload, plan=plan)


def test_payload_schema_requires_explicit_aligned_initialization_fields():
    payload = _payload()

    with pytest.raises(ValueError, match="raw_scales"):
        replace(payload, raw_scales=torch.empty((1, 2)))
    with pytest.raises(ValueError, match="opaque"):
        replace(payload, opaque_values=())


def test_population_plan_rejects_non_immutable_duplicate_registry_snapshot():
    plan = _plan()

    with pytest.raises(ValueError, match="duplicate"):
        replace(
            plan,
            row_state_registry_snapshot=(
                plan.row_state_registry_snapshot[0],
                plan.row_state_registry_snapshot[0],
            ),
        )
