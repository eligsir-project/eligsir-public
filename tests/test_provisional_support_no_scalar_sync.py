"""Projected depth evidence must not reduce accelerator tensors to Python bools."""

from types import SimpleNamespace

import pytest
import torch

import eligsir.mapping.integration.frame_preparation as frame_preparation


def _camera():
    return SimpleNamespace(
        image_height=3,
        image_width=3,
        world_view_transform=torch.eye(4),
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        FoVx=1.0,
        FoVy=1.0,
        uid=7,
    )


def _gaussians(provisional):
    return SimpleNamespace(
        get_xyz=torch.tensor(((0.0, 0.0, 1.0), (0.2, 0.0, 1.0))),
        provisional=torch.tensor(provisional, dtype=torch.bool),
        support_count=torch.zeros(2, dtype=torch.int32),
    )


@pytest.mark.parametrize("provisional", ((False, False), (True, False)))
def test_provisional_support_never_uses_tensor_boolean_control_flow(monkeypatch, provisional):
    gaussians = _gaussians(provisional)
    monkeypatch.setattr(frame_preparation, "depth_conflict_enabled", lambda _args: False)
    monkeypatch.setattr(
        frame_preparation,
        "_get_sensor_depth",
        lambda _cam, _height, _width: torch.ones((1, 3, 3)),
    )
    monkeypatch.setattr(
        torch.Tensor,
        "__bool__",
        lambda _self: (_ for _ in ()).throw(AssertionError("tensor boolean readback")),
    )

    frame_preparation._update_provisional_support(
        gaussians,
        _camera(),
        {"visibility_filter": torch.tensor((True, True))},
        SimpleNamespace(streaming_depth_consistency_thresh=0.1),
    )

    assert gaussians.support_count.tolist() == ([0, 0] if not any(provisional) else [1, 0])


def test_no_evidence_capability_skips_depth_lookup(monkeypatch):
    gaussians = SimpleNamespace(
        get_xyz=torch.zeros((2, 3)),
    )
    monkeypatch.setattr(frame_preparation, "depth_conflict_enabled", lambda _args: False)
    monkeypatch.setattr(
        frame_preparation,
        "_get_sensor_depth",
        lambda *_args: pytest.fail("depth must not be requested without an evidence consumer"),
    )

    frame_preparation._update_provisional_support(
        gaussians,
        _camera(),
        {"visibility_filter": torch.tensor((True, True))},
        SimpleNamespace(streaming_depth_consistency_thresh=0.1),
    )
