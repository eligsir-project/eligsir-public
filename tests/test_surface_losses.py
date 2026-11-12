from __future__ import annotations

import types

import torch

from eligsir.utils.surface_losses import compute_surface_anchor_losses


def test_surface_anchor_loss_penalizes_normal_drift():
    g = types.SimpleNamespace()
    g.get_xyz = torch.tensor([[0.0, 0.0, 0.1], [1.0, 0.0, 0.0]])
    g.anchor_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    g.anchor_normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    g.anchor_confidence = torch.tensor([1.0, 1.0])
    g.surface_state = torch.tensor([1, 1], dtype=torch.int8)
    g.get_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    g.get_scaling = torch.ones(2, 3) * 0.01
    result = compute_surface_anchor_losses(g, normal_weight=1.0)
    assert result.normal_drift > 0
    assert result.total > 0
