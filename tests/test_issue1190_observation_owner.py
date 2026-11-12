"""CPU-only contracts for issue #1190's terminal observation owner."""

from __future__ import annotations

import pytest

from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    FreshnessMode,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.admission.observation_decision import (
    ObservationDecision,
)
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureDecision,
    BackpressureOwner,
    PressureState,
)


def _source(ordinal: int, *, source_id: str = "camera") -> SourceIdentity:
    return SourceIdentity(
        source_id=source_id,
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=ordinal * 1_000_000,
    )


def _snapshot(source: SourceIdentity) -> ControlSnapshot:
    return ControlSnapshot(
        snapshot_id=7,
        control_version=1,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=3,
        topology_version=2,
        parameter_version=5,
        gaussian_count=0,
        optimizer_steps_completed=0,
        limits=ControlLimits(1, 0, 0, 0, 0, 0),
    )


def test_backpressure_proposal_becomes_an_immutable_admit_disposition():
    from eligsir.mapping.admission.observation_owner import (
        ObservationOwner,
    )

    source = _source(3)
    decision = BackpressureDecision(
        frame_id=3,
        action=BackpressureAction.ADMIT,
        reason="soft_all_capacity_available",
        pressure_state=PressureState.LOW,
        admit_supervision=True,
        allow_geometry_insertion=True,
    )

    disposition = ObservationOwner(lambda _snapshot: decision).decide(_snapshot(source))

    assert disposition.source == source
    assert disposition.kind is DispositionKind.ADMIT
    assert disposition.materialization_permitted is True
    assert disposition.evidence_trace[0].name == "backpressure.action"


def test_owner_records_a_terminal_disposition_and_returns_it_idempotently():
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    source = _source(30)
    calls = 0

    def _proposal(_snapshot):
        nonlocal calls
        calls += 1
        return BackpressureDecision(
            frame_id=30,
            action=BackpressureAction.SKIP_CAPACITY,
            reason="skipped_capacity_exhausted",
        )

    owner = ObservationOwner(_proposal)
    first = owner.decide(_snapshot(source))
    second = owner.decide(_snapshot(source))

    assert first is second
    assert calls == 1
    assert first.kind is DispositionKind.SKIP_CAPACITY
    assert owner.permit(first) is None


def test_owner_factory_wraps_a_backpressure_owner_before_materialization():
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    source = _source(31)
    owner = ObservationOwner.from_backpressure(
        BackpressureOwner(
            BackpressureConfig(enabled=True, force_all_stress=True)
        )
    )

    disposition = owner.decide(_snapshot(source))

    assert disposition.kind is DispositionKind.ADMIT
    assert owner.permit(disposition).source == source


def test_terminal_ledger_rejects_resurrection_but_not_another_source_epoch():
    from eligsir.mapping.admission.observation_owner import (
        DispositionConflict,
        DispositionLedger,
        disposition_from_backpressure,
    )

    source = _source(4)
    skip = BackpressureDecision(
        frame_id=4,
        action=BackpressureAction.SKIP_CAPACITY,
        reason="skipped_capacity_exhausted",
    )
    admit = BackpressureDecision(
        frame_id=4,
        action=BackpressureAction.ADMIT,
        reason="soft_all_capacity_available",
        admit_supervision=True,
    )
    ledger = DispositionLedger()
    ledger.record(disposition_from_backpressure(_snapshot(source), skip))

    with pytest.raises(DispositionConflict, match="terminal"):
        ledger.record(disposition_from_backpressure(_snapshot(source), admit))

    other_epoch = SourceIdentity(
        source_id="camera",
        stream_epoch=1,
        source_ordinal=4,
        timestamp_ns=4_000_000,
    )
    ledger.record(disposition_from_backpressure(_snapshot(other_epoch), admit))


def test_terminal_ledger_is_a_bounded_recent_idempotency_cache():
    from eligsir.mapping.admission.observation_owner import (
        DispositionLedger,
        disposition_from_backpressure,
    )

    ledger = DispositionLedger(recent_capacity=2)
    for ordinal in (1, 2, 3):
        source = _source(ordinal)
        ledger.record(
            disposition_from_backpressure(
                _snapshot(source),
                BackpressureDecision(
                    frame_id=ordinal,
                    action=BackpressureAction.SKIP_CAPACITY,
                    reason="skipped_capacity_exhausted",
                ),
            )
        )

    assert ledger.recent_count == 2
    assert ledger.disposition_for(_source(1)) is None
    assert ledger.disposition_for(_source(2)) is not None


def test_decline_never_calls_materializer_transfer_replay_or_insertion():
    from eligsir.mapping.admission.observation_owner import (
        apply_disposition,
        disposition_from_backpressure,
    )

    source = _source(5)
    declined = disposition_from_backpressure(
        _snapshot(source),
        BackpressureDecision(
            frame_id=5,
            action=BackpressureAction.COALESCE_INTO,
            reason="coalesced_into_representative",
            representative_frame_id=4,
        ),
        representative=_source(4),
    )

    def _poison(*_args):
        raise AssertionError("declined observation reached a pixel-work sink")

    assert apply_disposition(
        declined,
        permit=None,
        materialize=_poison,
        transfer=_poison,
        retain_for_replay=_poison,
        insert_geometry=_poison,
    ) is False


def test_owner_issued_permit_reaches_each_admitted_pixel_work_sink():
    from eligsir.mapping.admission.observation_owner import (
        ObservationOwner,
        apply_disposition,
    )

    source = _source(50)
    owner = ObservationOwner(
        lambda _snapshot: BackpressureDecision(
            frame_id=50,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        )
    )
    disposition = owner.decide(_snapshot(source))
    calls = []

    assert apply_disposition(
        disposition,
        permit=owner.permit(disposition),
        materialize=lambda _permit: calls.append("materialize") or object(),
        transfer=lambda _permit, _payload: calls.append("transfer"),
        retain_for_replay=lambda _permit, _payload: calls.append("replay"),
        insert_geometry=lambda _permit, _payload: calls.append("insert"),
    ) is True
    assert calls == ["materialize", "transfer", "replay", "insert"]


def test_legacy_decision_adapter_cannot_reenable_a_declined_disposition():
    from eligsir.mapping.admission.observation_owner import (
        disposition_from_backpressure,
    )

    source = _source(6)
    disposition = disposition_from_backpressure(
        _snapshot(source),
        BackpressureDecision(
            frame_id=6,
            action=BackpressureAction.SKIP_REDUNDANT,
            reason="skipped_redundant_intermediate",
        ),
    )

    legacy = ObservationDecision.from_disposition(disposition)

    assert legacy.admitted is False
    assert legacy.allow_geometry_insertion is False
    assert legacy.extra["source_identity"] == source.fingerprint()
