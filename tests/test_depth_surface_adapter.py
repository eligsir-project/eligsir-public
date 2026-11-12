import inspect

import pytest
import torch

from eligsir.mapping.evidence.depth_surface_adapter import (
    DepthSurfaceEvidenceInput,
    adapt_depth_surface_evidence,
)
from eligsir.mapping.evidence.surface_targets import SurfaceTargetSnapshot


def _snapshot():
    return SurfaceTargetSnapshot(11, 12, 13, 14, 15)


def _input(**overrides):
    values = dict(
        positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        uncertainty=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        support_evidence=torch.tensor([2.0, 3.0]),
        conflict_evidence=torch.tensor([0.25, 0.75]),
        independent_view_count=torch.tensor([2, 1], dtype=torch.int32),
        repeated_exposure_count=torch.tensor([5, 8], dtype=torch.int32),
        target_ids=torch.tensor([100, 101], dtype=torch.int64),
    )
    values.update(overrides)
    return DepthSurfaceEvidenceInput(**values)


def test_adapter_preserves_snapshot_and_separates_physical_views_from_replay():
    source = _input()
    result = adapt_depth_surface_evidence(source, snapshot=_snapshot(), max_targets=4)

    assert result.snapshot == _snapshot()
    assert result.count == 2
    assert result.independent_view_count.tolist() == [2, 1]
    assert result.repeated_exposure_count.tolist() == [5, 8]
    assert result.source_modality.tolist() == [4, 4]  # GEOMETRY
    assert result.target_ids.tolist() == [100, 101]
    assert result.positions.device == source.positions.device


def test_adapter_applies_optional_bounded_mask_without_host_compaction():
    source = _input()
    result = adapt_depth_surface_evidence(
        source,
        snapshot=_snapshot(),
        max_targets=1,
        candidate_mask=torch.tensor([True, False]),
    )
    assert result.count == 1
    assert result.positions.tolist() == [[1.0, 2.0, 3.0]]
    assert result.support_evidence.tolist() == [2.0]


def test_adapter_rejects_cross_device_and_invalid_mask():
    source = _input()
    with pytest.raises(ValueError, match="snapshot"):
        adapt_depth_surface_evidence(source, snapshot=None, max_targets=2)
    with pytest.raises(ValueError, match="mask"):
        adapt_depth_surface_evidence(
            source, snapshot=_snapshot(), max_targets=2,
            candidate_mask=torch.tensor([True, False], dtype=torch.int8),
        )


def test_adapter_does_not_mutate_or_actuate_input():
    source = _input()
    before = {name: getattr(source, name).clone() for name in (
        "positions", "normals", "uncertainty", "support_evidence",
        "conflict_evidence", "independent_view_count", "repeated_exposure_count",
        "target_ids",
    )}
    result = adapt_depth_surface_evidence(source, snapshot=_snapshot(), max_targets=2)
    for name, value in before.items():
        assert torch.equal(value, getattr(source, name))
    assert not hasattr(result, "apply")


def test_adapter_has_no_host_transfer_or_synchronization_path():
    source = inspect.getsource(adapt_depth_surface_evidence)
    assert ".item(" not in source
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert "synchronize" not in source


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("positions", torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]), "finite"),
        ("uncertainty", torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "non-negative"),
        ("support_evidence", torch.tensor([-1.0, 0.0]), "non-negative"),
        ("conflict_evidence", torch.tensor([float("inf"), 0.0]), "finite"),
        ("independent_view_count", torch.tensor([-1, 0], dtype=torch.int32), "non-negative"),
        ("repeated_exposure_count", torch.tensor([-1, 0], dtype=torch.int32), "non-negative"),
    ],
)
def test_debug_validation_enforces_producer_content_precondition(field, value, message):
    with pytest.raises(ValueError, match=message):
        _input(**{field: value}).validate_debug()
