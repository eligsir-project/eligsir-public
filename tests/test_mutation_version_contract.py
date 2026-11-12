"""Exact version/coordinate invariants for the sole mutation transaction."""

from __future__ import annotations

import pytest
import torch

from eligsir.map.mutation import (
    GsplatMutationTransaction,
    RelocationPlan,
    SubmapCorrection,
)


class _Model:
    def __init__(self):
        self.params = {
            "means": torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            "scales": torch.nn.Parameter(torch.zeros((2, 3))),
            "opacities": torch.nn.Parameter(torch.zeros(2)),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)),
        }
        self.anchor_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.anchor_normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def test_noop_commit_advances_no_version():
    change = GsplatMutationTransaction(_Model(), 4, 7, 9, 2).commit()
    assert change.map_content_version == 4
    assert change.topology_version == 7
    assert change.coordinate_frame_version == 9
    assert change.schema_version == 2


def test_relocation_changes_content_not_topology_or_coordinate_frame():
    model = _Model()
    txn = GsplatMutationTransaction(model, 4, 7, 9)
    txn.relocate(
        RelocationPlan(
            mask=torch.tensor([True, False]),
            new_means=torch.tensor([[0.5, 0.0, 0.0]]),
        )
    )
    change = txn.commit()
    assert change.map_content_version == 5
    assert change.topology_version == 7
    assert change.coordinate_frame_version == 9
    assert change.relocated == 1


def test_pose_correction_requires_populated_anchor_payloads():
    model = _Model()
    txn = GsplatMutationTransaction(model, 0, 0, 0)
    with pytest.raises(ValueError, match="anchor_xyz"):
        txn.apply_pose_correction(
            SubmapCorrection(
                mask=torch.tensor([True, False]),
                new_positions=torch.tensor([[2.0, 0.0, 0.0]]),
            )
        )


def test_pose_correction_updates_means_and_world_frame_anchors_atomically():
    model = _Model()
    txn = GsplatMutationTransaction(model, 1, 3, 5)
    mask = torch.tensor([True, False])
    txn.apply_pose_correction(
        SubmapCorrection(
            mask=mask,
            new_positions=torch.tensor([[2.0, 0.0, 0.0]]),
            new_anchor_xyz=torch.tensor([[2.0, 0.0, 0.0]]),
            new_anchor_normal=torch.tensor([[0.0, 1.0, 0.0]]),
        )
    )
    change = txn.commit()
    assert torch.equal(model.params["means"].data[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(model.anchor_xyz[0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(model.anchor_normal[0], torch.tensor([0.0, 1.0, 0.0]))
    assert change.map_content_version == 2
    assert change.topology_version == 3
    assert change.coordinate_frame_version == 6
    assert change.corrected == 1


def test_transaction_cannot_be_reused_after_commit():
    txn = GsplatMutationTransaction(_Model(), 0, 0)
    txn.commit()
    with pytest.raises(RuntimeError, match="already been committed"):
        txn.update_opacity_logits(torch.tensor([True, False]), -12.0)
