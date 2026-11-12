import inspect

import pytest
import torch

from eligsir.mapping.evidence.surface_targets import (
    SurfaceEvidenceModality,
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)


def _token(**overrides):
    values = dict(
        map_content=1,
        topology=2,
        coordinate_frame=3,
        schema=4,
        surface_evidence=5,
    )
    values.update(overrides)
    return SurfaceTargetSnapshot(**values)


def _batch(count=2, **overrides):
    device = torch.device("cpu")
    values = dict(
        snapshot=_token(),
        positions=torch.arange(count * 3, dtype=torch.float32, device=device).reshape(count, 3),
        normals=torch.nn.functional.normalize(torch.ones((count, 3), device=device), dim=1),
        uncertainty=torch.ones((count, 3), dtype=torch.float32, device=device),
        support_evidence=torch.ones(count, device=device),
        conflict_evidence=torch.zeros(count, device=device),
        independent_view_count=torch.ones(count, dtype=torch.int32, device=device),
        repeated_exposure_count=torch.zeros(count, dtype=torch.int32, device=device),
        target_ids=torch.arange(100, 100 + count, dtype=torch.int64, device=device),
        source_modality=torch.full((count,), 2, dtype=torch.int8, device=device),
        max_targets=4,
    )
    values.update(overrides)
    return SurfaceTargetBatch(**values)


def test_batch_is_bounded_versioned_and_keeps_physical_view_counters_separate():
    batch = _batch(repeated_exposure_count=torch.tensor([7, 1], dtype=torch.int32))
    assert batch.count == 2
    assert batch.snapshot == _token()
    assert batch.independent_view_count.tolist() == [1, 1]
    assert batch.repeated_exposure_count.tolist() == [7, 1]
    with pytest.raises((AttributeError, TypeError)):
        batch.max_targets = 9  # type: ignore[misc]


def test_batch_rejects_over_capacity_and_mismatched_or_cross_device_fields():
    with pytest.raises(ValueError, match="bound"):
        _batch(3, max_targets=2)
    with pytest.raises(ValueError, match="shape"):
        _batch(normals=torch.ones(3, 3))
    with pytest.raises(ValueError, match="modality"):
        _batch(source_modality=torch.tensor([2], dtype=torch.int8))


def test_batch_requires_finite_geometry_and_nonnegative_evidence():
    with pytest.raises(ValueError, match="finite"):
        _batch(positions=torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]])).validate_debug()
    with pytest.raises(ValueError, match="finite"):
        _batch(support_evidence=torch.tensor([float("nan"), 0.0])).validate_debug()
    with pytest.raises(ValueError, match="non-negative"):
        _batch(conflict_evidence=torch.tensor([-1.0, 0.0])).validate_debug()


def test_stale_validation_is_explicit_and_does_not_mutate_or_replan():
    batch = _batch()
    assert batch.is_current(_token())
    assert not batch.is_current(_token(topology=9))
    with pytest.raises(ValueError, match="stale"):
        batch.require_current(_token(topology=9))


def test_device_topk_selection_is_capacity_bounded_and_preserves_all_fields():
    source = _batch(
        4,
        max_targets=4,
        positions=torch.eye(4, 3),
        normals=torch.ones(4, 3),
        uncertainty=torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1,
        support_evidence=torch.tensor([1.0, 8.0, 3.0, 5.0]),
        conflict_evidence=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        independent_view_count=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        repeated_exposure_count=torch.tensor([4, 3, 2, 1], dtype=torch.int32),
        target_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        source_modality=torch.tensor([1, 2, 3, 4], dtype=torch.int8),
    )
    selected = SurfaceTargetBatch.select_topk(source, torch.tensor([0.1, 0.9, 0.2, 0.7]), 2)
    assert selected.count == 2
    assert selected.support_evidence.tolist() == [8.0, 5.0]
    assert selected.independent_view_count.tolist() == [2, 4]
    assert selected.source_modality.tolist() == [2, 4]
    assert selected.target_ids.tolist() == [11, 13]


def test_selection_rejects_invalid_capacity_and_scores():
    source = _batch()
    with pytest.raises(ValueError, match="capacity"):
        SurfaceTargetBatch.select_topk(source, torch.ones(2), 0)
    with pytest.raises(ValueError, match="scores"):
        SurfaceTargetBatch.select_topk(source, torch.ones(1), 1)
    with pytest.raises(ValueError, match="finite"):
        SurfaceTargetBatch.validate_scores_debug(torch.tensor([float("nan"), 1.0]))


def test_selection_sanitizes_nan_scores_on_device_without_host_validation():
    source = _batch()
    selected = SurfaceTargetBatch.select_topk(source, torch.tensor([float("nan"), 0.9]), 1)
    assert selected.support_evidence.tolist() == [1.0]


def test_normal_constructor_and_selection_do_not_reduce_tensor_values_to_host():
    source = inspect.getsource(SurfaceTargetBatch.__post_init__)
    selector = inspect.getsource(SurfaceTargetBatch.select_topk)
    assert "bool(" not in source
    assert "bool(" not in selector
    assert ".cpu()" not in selector
