"""Contracts for reusing a normal supervision render in insertion prep.

These are CPU-tensor contract tests: the production path never needs a host
read to decide whether a lease is usable, and all version/source/tier checks
are scalar metadata checks made before candidate generation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
    SupervisionModality,
)
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    OptimizationStepRequest,
    OptimizationView,
    RenderOutputs,
)
from eligsir.runtime.prepared_source import PreparedSourceLease


def _source(ordinal: int = 7) -> SourceIdentity:
    return SourceIdentity("live", 2, ordinal, 123_456_789)


def _buffer(value: torch.Tensor) -> BufferRef:
    return BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding="tensor",
            device=str(value.device),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )


def _lease(
    source: SourceIdentity,
    *,
    tier: str = "native",
    eligibility: ModalityEligibility | None = None,
) -> PreparedSourceLease:
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=eligibility or ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=True,
        ),
        source_metadata={"native_height": 4, "native_width": 5},
        tier_preparation_available=(tier,),
        borrowed_device_payloads={
            "depth_meters": _buffer(torch.ones(4, 5)),
            "rgb": _buffer(torch.zeros(4, 5, 3)),
            "c2w": _buffer(torch.eye(4)),
        },
        device="cpu",
    )


def _snapshot(*, parameter: int = 9) -> EvidenceSnapshotToken:
    return EvidenceSnapshotToken(
        map_content=3,
        topology=4,
        coordinate_frame=5,
        schema=6,
        optimizer_update=7,
        surface_evidence=8,
        policy=9,
        parameter=parameter,
    )


def _render(
    source: SourceIdentity,
    *,
    tier: str = "native",
    parameter: int = 9,
    eligibility: ModalityEligibility | None = None,
):
    from eligsir.mapping.integration.depth_render_reuse import GeometryRenderLease

    return GeometryRenderLease(
        source_identity=source,
        snapshot_token=_snapshot(parameter=parameter),
        resolution_tier=tier,
        modality_eligibility=eligibility or ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=True,
        ),
        alpha=torch.ones(4, 5, 1),
        rendered_depth=torch.ones(4, 5, 1),
    )


def test_matching_native_geometry_render_and_source_lease_are_reused_without_upload():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    lease = _lease(source)
    reuse, reason = bind_depth_insertion_render_reuse(
        render_lease=_render(source),
        source_lease=lease,
        source=source,
        expected_snapshot=_snapshot(),
        expected_resolution_tier="native",
    )

    assert reason is DepthInsertionReuseReason.REUSED
    assert reuse is not None
    assert reuse.depth_meters is lease.payloads["depth_meters"].data
    assert reuse.source_h2d_ops == 0
    assert reuse.extra_rasterizer_calls == 0
    assert reuse.alpha.shape == (4, 5, 1)


def test_mismatch_fails_closed_with_a_precise_legacy_fallback_reason():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    reuse, reason = bind_depth_insertion_render_reuse(
        render_lease=_render(source, tier="half"),
        source_lease=_lease(source, tier="half"),
        source=source,
        expected_snapshot=_snapshot(),
        expected_resolution_tier="native",
    )

    assert reuse is None
    assert reason is DepthInsertionReuseReason.NON_NATIVE_TIER


def test_stale_parameter_snapshot_fails_closed_even_when_source_and_tier_match():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    reuse, reason = bind_depth_insertion_render_reuse(
        render_lease=_render(source, parameter=8),
        source_lease=_lease(source),
        source=source,
        expected_snapshot=_snapshot(parameter=9),
        expected_resolution_tier="native",
    )

    assert reuse is None
    assert reason is DepthInsertionReuseReason.STALE_SNAPSHOT


def test_same_ordinal_but_different_authoritative_source_identity_falls_back():
    """Ordinal reuse across source/id epochs must never borrow a render."""

    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    # These collisions share the ordinal but disagree on the authoritative
    # source identity in each field that scopes a live observation.
    collisions = (
        SourceIdentity("other-camera", 2, source.source_ordinal, 123_456_789),
        SourceIdentity("live", 3, source.source_ordinal, 123_456_789),
        SourceIdentity("live", 2, source.source_ordinal, 123_456_790),
    )
    for other_source in collisions:
        reuse, reason = bind_depth_insertion_render_reuse(
            render_lease=_render(other_source),
            source_lease=_lease(source),
            source=source,
            expected_snapshot=_snapshot(),
            expected_resolution_tier="native",
        )

        assert reuse is None
        assert reason is DepthInsertionReuseReason.SOURCE_MISMATCH


def test_non_native_render_lattice_falls_back_before_depth_insertion():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        GeometryRenderLease,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    # The source depth lease is 4x5.  A resized alpha *or* depth render may
    # not be interpreted as native geometry evidence.
    for alpha, rendered_depth in (
        (torch.ones(3, 5, 1), torch.ones(4, 5, 1)),
        (torch.ones(4, 5, 1), torch.ones(3, 5, 1)),
    ):
        render = GeometryRenderLease(
            source_identity=source,
            snapshot_token=_snapshot(),
            resolution_tier="native",
            modality_eligibility=ModalityEligibility(
                depth_insertion=True,
                geometry_population_target=True,
            ),
            alpha=alpha,
            rendered_depth=rendered_depth,
        )
        reuse, reason = bind_depth_insertion_render_reuse(
            render_lease=render,
            source_lease=_lease(source),
            source=source,
            expected_snapshot=_snapshot(),
            expected_resolution_tier="native",
        )

        assert reuse is None
        assert reason is DepthInsertionReuseReason.NATIVE_LATTICE_MISMATCH


def test_reuse_requires_depth_insertion_eligibility_not_only_a_depth_render():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        GeometryRenderLease,
        bind_depth_insertion_render_reuse,
    )

    source = _source()
    render = GeometryRenderLease(
        source_identity=source,
        snapshot_token=_snapshot(),
        resolution_tier="native",
        modality_eligibility=ModalityEligibility(depth_supervision=True),
        alpha=torch.ones(4, 5, 1),
        rendered_depth=torch.ones(4, 5, 1),
    )
    reuse, reason = bind_depth_insertion_render_reuse(
        render_lease=render,
        source_lease=_lease(source),
        source=source,
        expected_snapshot=_snapshot(),
        expected_resolution_tier="native",
    )

    assert reuse is None
    assert reason is DepthInsertionReuseReason.GEOMETRY_INELIGIBLE


def test_population_target_reuse_is_explicitly_independent_of_direct_insertion():
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
        bind_depth_insertion_render_reuse,
        bind_population_target_render_reuse,
    )

    source = _source()
    target_eligibility = ModalityEligibility(
        depth_supervision=True,
        depth_insertion=False,
        geometry_population_target=True,
    )
    kwargs = dict(
        render_lease=_render(source, eligibility=target_eligibility),
        source_lease=_lease(source, eligibility=target_eligibility),
        source=source,
        expected_snapshot=_snapshot(),
        expected_resolution_tier="native",
    )

    target_reuse, target_reason = bind_population_target_render_reuse(**kwargs)
    insertion_reuse, insertion_reason = bind_depth_insertion_render_reuse(**kwargs)

    assert target_reason is DepthInsertionReuseReason.REUSED
    assert target_reuse is not None
    assert target_reuse.source_h2d_ops == 0
    assert target_reuse.extra_rasterizer_calls == 0
    assert insertion_reuse is None
    assert insertion_reason is DepthInsertionReuseReason.GEOMETRY_INELIGIBLE


def test_production_selector_uses_all_view_lease_and_rejects_stale_plan_snapshot(monkeypatch):
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionReuseReason,
    )
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    source = _source()
    lease = _lease(source)
    render = _render(source)
    evidence = SupervisionEvidenceBatch(
        snapshot_token=_snapshot(),
        transaction_id=1,
        modalities=(SupervisionModality(geometry_evidence_enabled=True),),
        modality_eligibility=(ModalityEligibility(depth_insertion=True),),
        geometry_render_leases=(render,),
    )
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    monkeypatch.setattr(
        runtime,
        "_native_prepared_source_lease",
        lambda *_args: lease,
    )
    result = SimpleNamespace(evidence=evidence, parameter_version_before=9)
    plan = SimpleNamespace(map_content_version=3, topology_version=4, parameter_version=9)

    reuse, reason = runtime._select_depth_insertion_render_reuse(
        result=result, camera=object(), source=source, plan=plan
    )

    assert reason is DepthInsertionReuseReason.REUSED
    assert reuse is not None

    stale_plan = SimpleNamespace(map_content_version=99, topology_version=4, parameter_version=9)
    reuse, reason = runtime._select_depth_insertion_render_reuse(
        result=result, camera=object(), source=source, plan=stale_plan
    )
    assert reuse is None
    assert reason is DepthInsertionReuseReason.STALE_SNAPSHOT


def test_production_insertion_prep_skips_legacy_render_when_exact_reuse_is_bound(monkeypatch):
    from eligsir.mapping.integration.depth_render_reuse import (
        DepthInsertionRenderReuse,
    )
    from eligsir.runtime import production_application as app
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    source = _source()
    prepared = _lease(source)
    reuse = DepthInsertionRenderReuse(
        alpha=torch.ones(4, 5, 1),
        rendered_depth=torch.ones(4, 5, 1),
        depth_meters=prepared.payloads["depth_meters"].data,
        rgb=prepared.payloads["rgb"].data,
        c2w=prepared.payloads["c2w"].data,
    )
    called = {}

    def _producer(*_args, **kwargs):
        called["render_reuse"] = kwargs["render_reuse"]
        return None, {
            "insertion_render_reused": True,
            "source_lease_hit": True,
            "source_h2d_ops": 0,
            "source_d2h_ops": 0,
            "extra_rasterizer_calls": 0,
        }

    monkeypatch.setattr(app, "insert_gaussians_from_frame_device_native", _producer)
    monkeypatch.setattr(
        app,
        "render",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not rerender")),
    )
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime.scene = object()
    runtime.gaussians = object()
    runtime.pipe = object()
    runtime._background = object()
    runtime._depth_insertion_policy = SimpleNamespace(initial_opacity=0.1)
    runtime.frames_seen = 0
    runtime._transfer_owner = object()
    frame = SimpleNamespace()
    proposal = runtime._prepare_depth_insertion_proposal(
        object(),
        frame,
        source,
        insertion_plan=object(),
        modality_eligibility=ModalityEligibility(depth_insertion=True),
        prev_frame=None,
        prev_depth_meters=None,
        insertion_policy=runtime._depth_insertion_policy,
        render_reuse=reuse,
    )

    assert proposal is None
    assert called["render_reuse"] is reuse
    assert runtime.depth_insertion_render_reuse_telemetry() == {
        "insertion_render_reused": 1,
        "source_lease_hit": 1,
        "source_h2d_ops": 0,
        "source_d2h_ops": 0,
        "extra_rasterizer_calls": 0,
        "fallback": "reused",
    }


def test_b1_adapter_emits_same_render_geometry_lease_and_keeps_legacy_payload():
    from eligsir.mapping.optimization_runtime import _attach_b1_supervision_evidence

    source = _source()
    request = OptimizationStepRequest(
        view=OptimizationView(
            source_index=source.source_ordinal,
            source_identity=source,
            resolution_bucket="native",
            modality_eligibility=ModalityEligibility(
                depth_insertion=True,
                geometry_population_target=True,
            ),
        ),
        iteration=7,
        map_version=3,
        topology_version=4,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(want_supervision_evidence=True),
        random_state_token=object(),
    )
    payload = object()
    result = SimpleNamespace(
        parameter_version_before=9,
        evidence=SimpleNamespace(payload=payload),
        render_outputs=RenderOutputs(
            payload={
                "alpha": torch.ones(4, 5, 1),
                "rendered_depth": torch.ones(4, 5, 1),
            }
        ),
    )
    context = SimpleNamespace(
        evidence_transaction_id=11,
        coordinate_frame_version=5,
        schema_version=6,
        optimizer_update_id=7,
        surface_evidence_version=8,
        policy_version=9,
    )

    _attach_b1_supervision_evidence(result, request, context)

    assert isinstance(result.evidence, SupervisionEvidenceBatch)
    assert result.evidence.payload is payload
    assert [item.source_identity for item in result.evidence.geometry_render_leases] == [source]


def test_b1_depth_evidence_survives_without_geometry_insertion_eligibility():
    """Depth measurement evidence is not contingent on topology authority."""

    from eligsir.mapping.optimization_runtime import _attach_b1_supervision_evidence

    source = _source()
    request = OptimizationStepRequest(
        view=OptimizationView(
            source_index=source.source_ordinal,
            source_identity=source,
            resolution_bucket="native",
            render_depth=True,
            modality_eligibility=ModalityEligibility(
                depth_supervision=True,
                depth_insertion=False,
            ),
            supervision_modality=SupervisionModality(
                depth_loss_enabled=True,
                depth_evidence_enabled=True,
                geometry_evidence_enabled=True,
            ),
        ),
        iteration=7,
        map_version=3,
        topology_version=4,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(want_supervision_evidence=True),
        random_state_token=object(),
    )
    result = SimpleNamespace(
        parameter_version_before=9,
        evidence=None,
        render_outputs=RenderOutputs(
            payload={
                "alpha": torch.ones(4, 5, 1),
                "rendered_depth": torch.ones(4, 5, 1),
            }
        ),
    )
    context = SimpleNamespace(
        evidence_transaction_id=11,
        coordinate_frame_version=5,
        schema_version=6,
        optimizer_update_id=7,
        surface_evidence_version=8,
        policy_version=9,
    )

    _attach_b1_supervision_evidence(result, request, context)

    modality = result.evidence.modalities[0]
    assert modality.depth_evidence_enabled is True
    assert modality.geometry_evidence_enabled is False
    assert result.evidence.geometry_render_leases == ()
