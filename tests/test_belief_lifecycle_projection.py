import torch

from eligsir.mapping.belief_lifecycle_projection import (
    DiagnosticLifecycle,
    project_diagnostic_lifecycle,
)
from eligsir.utils.gaussian_belief import (
    ExistenceBelief,
    GaussianBelief,
    GeometryBelief,
    ObservabilityBelief,
    ProvenanceBelief,
    RepresentationBelief,
    TemporalBelief,
)


def _belief(n=5):
    z = torch.zeros(n)
    return GaussianBelief(
        provenance=ProvenanceBelief(
            gaussian_uid=torch.arange(n), birth_update=torch.tensor([0, 2, 8, 0, 0]),
            birth_method=torch.zeros(n, dtype=torch.int8), parent_uid=torch.full((n,), -1),
            root_uid=torch.arange(n), generation=torch.zeros(n, dtype=torch.int32),
            birth_anchor_xyz=torch.zeros(n, 3),
        ),
        observability=ObservabilityBelief(
            meaningful_exposure_count=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            independent_support_views=torch.tensor([0, 0, 1, 2, 2], dtype=torch.int16),
            baseline_diversity=z.clone(), highest_supervised_tier=None,
            last_exposed_update=torch.zeros(n, dtype=torch.int32), last_supported_update=None,
        ),
        existence=ExistenceBelief(
            support_evidence=torch.tensor([0., 0., 1., 1., 1.]),
            contradiction_evidence=torch.tensor([0., 1., 0., 0., 2.]),
            p_exist=torch.tensor([1., 0., 1., 1., .2]),
        ),
        geometry=GeometryBelief(
            surface_anchor_id=torch.full((n,), -1),
            p_surface_consistent=torch.tensor([0., 0., .4, .8, .95]),
            surface_position_uncertainty=z.clone(), surface_normal_uncertainty=z.clone(),
        ),
        representation=None,
        temporal=TemporalBelief(
            change_probability=z.clone(), evidence_recency=torch.zeros(n, dtype=torch.int32),
            confidence_decay=z.clone(),
        ),
        n=n,
    )


def test_projection_is_diagnostic_and_uses_evidence_not_age():
    result = project_diagnostic_lifecycle(_belief(), optimizer_update=10, birth_grace_updates=3)
    assert result.state.tolist() == [DiagnosticLifecycle.PROVISIONAL.code,
                                     DiagnosticLifecycle.PROVISIONAL.code,
                                     DiagnosticLifecycle.PROVISIONAL.code,
                                     DiagnosticLifecycle.ESTABLISHED.code,
                                     DiagnosticLifecycle.ESTABLISHED.code]
    assert result.birth_grace.tolist() == [False, False, True, False, False]
    assert result.no_measurement.tolist() == [True, True, False, False, False]
    assert result.negative_evidence.tolist() == [False, True, False, False, True]


def test_no_evidence_is_not_negative_and_established_needs_geometry():
    belief = _belief()
    belief.observability.meaningful_exposure_count[:] = 0
    belief.existence.contradiction_evidence[:] = 0
    result = project_diagnostic_lifecycle(belief, optimizer_update=100, birth_grace_updates=0)
    assert not bool(result.negative_evidence.any())
    assert bool(result.no_measurement.all())
    assert bool(result.state.eq(DiagnosticLifecycle.PROVISIONAL.code).all())


def test_optional_legacy_state_only_produces_parity_masks():
    legacy = torch.tensor([0, 1, 2, 3, 2], dtype=torch.int8)
    result = project_diagnostic_lifecycle(_belief(), optimizer_update=10, legacy_state=legacy)
    assert result.legacy_established is not None
    assert result.legacy_established.tolist() == [False, False, True, True, True]
    assert result.established_agreement is not None
    assert result.established_agreement.tolist() == [True, True, False, True, True]


def test_projection_keeps_device_tensors_and_rejects_bad_shapes():
    result = project_diagnostic_lifecycle(_belief(), optimizer_update=10)
    for value in (result.state, result.birth_grace, result.no_measurement,
                  result.negative_evidence, result.established):
        assert value.device.type == "cpu"
    malformed = _belief()
    malformed.provenance.birth_update = torch.zeros(2, dtype=torch.int32)
    try:
        project_diagnostic_lifecycle(malformed, optimizer_update=10)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched belief tensors must fail closed")
