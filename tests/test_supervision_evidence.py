import pytest
import torch

from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    RenderRequirement,
    SupervisionEvidenceBatch,
    SupervisionModality,
    group_render_requirement,
)


def _batch(**overrides):
    device = torch.device("cpu")
    values = dict(
        snapshot_token=EvidenceSnapshotToken(1, 2, 3, 4, 5, 6, 7, 8),
        transaction_id=7,
        modalities=(SupervisionModality(), SupervisionModality(depth_loss_enabled=True)),
        rgb_residual_summary=torch.zeros(2, device=device),
        depth_residual_summary=torch.zeros(2, device=device),
        rgb_valid_pixels=torch.ones(2, device=device),
        depth_valid_pixels=torch.ones(2, device=device),
        gaussian_ids=torch.arange(3, device=device),
        support_evidence=torch.zeros(3, device=device),
        contradiction_evidence=torch.zeros(3, device=device),
    )
    values.update(overrides)
    return SupervisionEvidenceBatch(**values)


def test_mixed_modalities_request_one_depth_render_but_preserve_per_view_flags():
    modalities = (
        SupervisionModality(rgb_loss_enabled=True),
        SupervisionModality(depth_loss_enabled=True),
        SupervisionModality(depth_evidence_enabled=True, rgb_loss_enabled=False),
    )

    assert group_render_requirement(modalities) is RenderRequirement.RGB_DEPTH
    assert modalities[0].depth_loss_enabled is False
    assert modalities[2].rgb_loss_enabled is False


def test_batch_requires_snapshot_token():
    with pytest.raises(ValueError, match="snapshot token"):
        _batch(snapshot_token=None)


def test_snapshot_token_requires_all_non_negative_versions():
    with pytest.raises(TypeError):
        EvidenceSnapshotToken(1, 2, 3, 4, 5, 6)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceSnapshotToken(1, 2, 3, 4, 5, -1, 7, 8)


def test_snapshot_token_rejects_an_omitted_parameter_version():
    with pytest.raises(TypeError):
        EvidenceSnapshotToken(1, 2, 3, 4, 5, 6, 7)


def test_modality_switches_must_be_real_bools():
    with pytest.raises(TypeError, match="bool"):
        SupervisionModality(rgb_loss_enabled=1)  # type: ignore[arg-type]


def test_gaussian_shape_is_inferred_without_ids():
    batch = _batch(
        gaussian_ids=None,
        support_evidence=torch.zeros(3),
        contradiction_evidence=torch.zeros(3),
        view_incidence=torch.zeros(3, 2),
    )
    assert batch.gaussian_count == 3

    with pytest.raises(ValueError, match="per-Gaussian"):
        _batch(
            gaussian_ids=None,
            support_evidence=torch.zeros(3),
            contradiction_evidence=torch.zeros(4),
        )


def test_incidence_must_match_inferred_gaussian_count_without_ids():
    with pytest.raises(ValueError, match="per-Gaussian"):
        _batch(
            gaussian_ids=None,
            support_evidence=torch.zeros(3),
            view_incidence=torch.zeros(4, 2),
        )


def test_batch_rejects_cross_device_tensors():
    with pytest.raises(ValueError, match="share one device"):
        _batch(depth_valid_pixels=torch.ones(2, device="meta"))


def test_batch_rejects_mismatched_compact_shapes():
    with pytest.raises(ValueError, match="per-view"):
        _batch(depth_valid_pixels=torch.ones(3))

    with pytest.raises(ValueError, match="per-Gaussian"):
        _batch(contradiction_evidence=torch.zeros(4))


def test_batch_preserves_device_and_is_immutable():
    batch = _batch()

    assert batch.rgb_residual_summary.device == torch.device("cpu")
    assert batch.gaussian_ids.device == torch.device("cpu")
    with pytest.raises((AttributeError, TypeError)):
        batch.transaction_id = 8


def test_batch_has_no_cpu_conversion_or_reduction_helpers():
    batch = _batch()

    assert not hasattr(batch, "cpu")
    assert not hasattr(batch, "numpy")
    assert not hasattr(batch, "to")
    assert not hasattr(batch, "mean")


def test_source_identities_are_an_exact_immutable_view_inventory():
    from eligsir.contracts.runtime import SourceIdentity

    sources = (SourceIdentity("test", 0, 1, 1), SourceIdentity("test", 0, 2, 2))
    batch = _batch(source_identities=sources)

    assert batch.source_identities == sources
    with pytest.raises(TypeError, match="source_identities"):
        _batch(source_identities=[])
