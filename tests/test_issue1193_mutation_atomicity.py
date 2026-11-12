from __future__ import annotations

import pytest
import torch

from eligsir.map.arbiter import (
    GaussianMutationArbiter,
    PoseCorrectionProposal,
    ProposalKey,
    PruneProposal,
    SurfaceInsertionProposal,
)
from eligsir.map.mutation import (
    GaussianInsertionBatch,
    GsplatMutationTransaction,
    SubmapCorrection,
    TopologyChange,
    RelocationPlan,
)
from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMutationTransaction


class _Model:
    def __init__(self, population: int = 3):
        self.params = {
            "means": torch.nn.Parameter(torch.arange(population * 3, dtype=torch.float32).reshape(population, 3)),
            "opacities": torch.nn.Parameter(torch.zeros(population, 1)),
        }
        self.anchor_xyz = torch.zeros(population, 3)
        self.anchor_normal = torch.zeros(population, 3)

    def add_points_as_gaussians(self, *, points, colors, **_kwargs):
        self.params["means"] = torch.nn.Parameter(torch.cat((self.params["means"].detach(), points)))
        self.params["opacities"] = torch.nn.Parameter(
            torch.cat((self.params["opacities"].detach(), torch.zeros(points.shape[0], 1)))
        )
        return points.shape[0]

    def prune_points(self, mask):
        keep = ~mask if mask.dtype == torch.bool else torch.ones(len(self.params["means"]), dtype=torch.bool)
        if mask.dtype != torch.bool:
            keep[mask] = False
        self.params["means"] = torch.nn.Parameter(self.params["means"].detach()[keep])
        self.params["opacities"] = torch.nn.Parameter(self.params["opacities"].detach()[keep])


class _Backend:
    def __init__(self, model):
        self.model = model
        self.map_content_version = 4
        self.topology_version = 2
        self.coordinate_frame_version = 0
        self.schema_version = 1

    @property
    def population_count(self):
        return self.model.params["means"].shape[0]

    def begin_mutation(self):
        return GsplatMutationTransaction(
            self.model,
            self.map_content_version,
            self.topology_version,
            self.coordinate_frame_version,
            self.schema_version,
        )

    def commit_mutation(self, txn):
        change = txn.commit()
        self.map_content_version = change.map_version
        self.topology_version = change.topology_version
        return change


def _insert(key_ordinal: int, points: torch.Tensor, colors: torch.Tensor):
    return SurfaceInsertionProposal(
        ProposalKey(0, key_ordinal, key_ordinal, "camera"),
        GaussianInsertionBatch(points=points, colors=colors),
    )


def test_later_invalid_proposal_leaves_model_and_versions_unchanged():
    model = _Model()
    backend = _Backend(model)
    means_before = model.params["means"].detach().clone()
    versions_before = (backend.map_content_version, backend.topology_version)
    valid_first = _insert(1, torch.ones(1, 3), torch.ones(1, 3))
    invalid_later = _insert(2, torch.ones(2, 3), torch.ones(1, 3))

    with pytest.raises(ValueError, match="colors"):
        GaussianMutationArbiter(backend).commit((valid_first, invalid_later))

    assert torch.equal(model.params["means"], means_before)
    assert backend.population_count == 3
    assert (backend.map_content_version, backend.topology_version) == versions_before


def test_bool_prune_counting_never_extracts_a_gpu_scalar(monkeypatch):
    model = _Model(population=4)
    backend = _Backend(model)

    def _forbid_item(_self):
        raise AssertionError("Tensor.item synchronizes GPU scalar state")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    result = GaussianMutationArbiter(backend).commit(
        (PruneProposal(ProposalKey(0, 1, 1, "camera"), torch.tensor([True, False, True, False])),)
    )

    assert backend.population_count == 2
    assert result.change.pruned == 2


class _FailOnSecondInsertTransaction:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.inserted = 0

    def insert(self, batch):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("injected second insert failure")
        count = batch.points.shape[0]
        self.backend.population += count
        self.inserted += count
        return count

    def commit(self):
        return TopologyChange(
            map_version=1,
            topology_version=1,
            inserted=self.inserted,
            content_changed=True,
        )


class _RuntimeFailureBackend:
    def __init__(self):
        self.population = 0
        self.transaction = None

    @property
    def population_count(self):
        return self.population

    def begin_mutation(self):
        self.transaction = _FailOnSecondInsertTransaction(self)
        return self.transaction

    def commit_mutation(self, txn):
        return txn.commit()


def test_multiple_insert_proposals_cross_one_backend_operation():
    backend = _RuntimeFailureBackend()
    proposals = (
        _insert(1, torch.ones(1, 3), torch.ones(1, 3)),
        _insert(2, torch.ones(1, 3) * 2, torch.ones(1, 3) * 2),
    )

    result = GaussianMutationArbiter(backend).commit(proposals)

    assert backend.transaction.calls == 1
    assert backend.population == 2
    assert result.change.inserted == 2


def test_heterogeneous_proposals_fail_before_transaction_creation():
    backend = _RuntimeFailureBackend()
    proposals = (
        _insert(1, torch.ones(1, 3), torch.ones(1, 3)),
        PruneProposal(ProposalKey(0, 2, 2, "camera"), torch.empty(0, dtype=torch.bool)),
    )

    with pytest.raises(ValueError, match="homogeneous"):
        GaussianMutationArbiter(backend).commit(proposals)

    assert backend.transaction is None
    assert backend.population == 0


def test_insertion_preflight_rejects_misaligned_optional_shape_and_device():
    model = _Model()
    backend = _Backend(model)
    bad_shape = GaussianInsertionBatch(
        points=torch.ones(2, 3),
        colors=torch.ones(2, 3),
        rotations=torch.ones(2, 3),
    )
    with pytest.raises(ValueError, match="rotations"):
        GaussianMutationArbiter(backend).commit(
            (SurfaceInsertionProposal(ProposalKey(0, 1, 1, "camera"), bad_shape),)
        )

    bad_device = GaussianInsertionBatch(
        points=torch.ones(1, 3),
        colors=torch.ones(1, 3, device="meta"),
    )
    with pytest.raises(ValueError, match="same device"):
        GaussianMutationArbiter(backend).commit(
            (SurfaceInsertionProposal(ProposalKey(0, 2, 2, "camera"), bad_device),)
        )
    assert backend.population_count == 3


def test_malformed_anchor_correction_fails_before_any_mean_write():
    model = _Model()
    backend = _Backend(model)
    means_before = model.params["means"].detach().clone()
    proposal = PoseCorrectionProposal(
        ProposalKey(0, 1, 1, "camera"),
        SubmapCorrection(
            mask=torch.tensor([True, False, False]),
            new_positions=torch.full((1, 3), 9.0),
            new_anchor_xyz=torch.zeros(2, 3),
            new_anchor_normal=torch.zeros(1, 3),
        ),
    )
    with pytest.raises(ValueError, match="new_anchor_xyz"):
        GaussianMutationArbiter(backend).commit((proposal,))
    assert torch.equal(model.params["means"], means_before)
    assert (backend.map_content_version, backend.topology_version) == (4, 2)


def test_legacy_relocation_runtime_failure_rolls_back_prior_fields():
    model = type("Legacy", (), {})()
    model._xyz = torch.nn.Parameter(torch.zeros(3, 3))
    model._scaling = torch.nn.Parameter(torch.zeros(3, 3))
    model._opacity = torch.nn.Parameter(torch.zeros(3, 1))
    model._rotation = torch.nn.Parameter(torch.zeros(3, 4))
    txn = LegacyGraphdecoMutationTransaction(model, 0, 0)
    mask = torch.tensor([True, False, False])
    sparse_scale = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (1, 3)
    )
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        txn.relocate(
            RelocationPlan(
                mask=mask,
                new_means=torch.full((1, 3), 9.0),
                new_raw_scales=sparse_scale,
            )
        )
    assert torch.equal(model._xyz, torch.zeros(3, 3))
    assert txn.commit().content_changed is False
