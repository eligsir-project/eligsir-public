"""Deterministic contract tests for asynchronous EvidenceDemand resolution."""

from __future__ import annotations

import pytest
from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandKind,
    EvidenceDemandModality,
)
from eligsir.mapping.evidence.evidence_demand_scheduler import (
    EvidenceDemandCandidate,
    EvidenceDemandCurrentVersions,
    EvidenceDemandResolver,
)
from eligsir.mapping.observation_db import BufferStage, ModalityEligibility


def _versions(**overrides: int) -> EvidenceDemandCurrentVersions:
    values = {
        "map": 4,
        "topology": 3,
        "coordinate": 2,
        "schema": 1,
        "optimizer": 7,
        "surface": 5,
        "policy": 6,
    }
    values.update(overrides)
    return EvidenceDemandCurrentVersions(**values)


def _demand(*, expiry_epoch: int = 10, **overrides: object) -> EvidenceDemand:
    values: dict[str, object] = {
        "kind": EvidenceDemandKind.DEPTH_SURFACE,
        "requested_modality": EvidenceDemandModality.DEPTH,
        "target_ids": ("g-2",),
        "map_version": 4,
        "topology_version": 3,
        "coordinate_version": 2,
        "schema_version": 1,
        "optimizer_version": 7,
        "surface_version": 5,
        "policy_version": 6,
        "priority": 3.0,
        "expiry_epoch": expiry_epoch,
    }
    values.update(overrides)
    return EvidenceDemand(**values)


def _candidate(
    frame_id: int,
    *,
    targets: tuple[str, ...] = (),
    stage: BufferStage = BufferStage.COLD_CACHE,
    resident: bool = False,
    materializable: bool = True,
    modalities: ModalityEligibility | None = None,
) -> EvidenceDemandCandidate:
    return EvidenceDemandCandidate(
        frame_id=frame_id,
        buffer_stage=stage,
        modality_eligibility=modalities or ModalityEligibility(depth_supervision=True),
        observed_target_ids=targets,
        resident=resident,
        materializable=materializable,
    )


def test_resolver_turns_matching_incidence_into_shortlist_intent_not_selection():
    resolver = EvidenceDemandResolver(max_pending_demands=2, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=9,
        candidates=(
            _candidate(10, targets=("g-2",)),
            _candidate(11, targets=("g-2",)),
            _candidate(12, targets=("other",)),
        ),
    )

    assert len(intents) == 1
    assert intents[0].candidate_frame_ids == (10, 11)
    assert intents[0].selected_frame_id is None
    assert intents[0].incidence_is_shortlist_evidence is True
    assert intents[0].resident_fallback_used is False


def test_resolver_drops_expired_or_version_stale_demands_before_candidate_work():
    resolver = EvidenceDemandResolver(max_pending_demands=3, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand(expiry_epoch=8))
    assert resolver.enqueue(_demand(target_ids=("g-stale",), topology_version=2))

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=9,
        candidates=(_candidate(10, targets=("g-2", "g-stale")),),
    )

    assert intents == ()
    assert resolver.telemetry.expired_drops == 1
    assert resolver.telemetry.stale_version_drops == 1


def test_resolver_excludes_holdout_and_ineligible_modalities_from_shortlist():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=3)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=1,
        candidates=(
            _candidate(1, targets=("g-2",), stage=BufferStage.HOLDOUT_TEST),
            _candidate(
                2,
                targets=("g-2",),
                modalities=ModalityEligibility(rgb_supervision=True),
            ),
            _candidate(3, targets=("g-2",)),
        ),
    )

    assert intents[0].candidate_frame_ids == (3,)


def test_resolver_uses_ready_resident_fallback_without_waiting_for_cold_demand():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=1,
        candidates=(_candidate(10, targets=("g-2",), materializable=False),),
        resident_fallbacks=(_candidate(99, resident=True),),
    )

    assert intents[0].candidate_frame_ids == (99,)
    assert intents[0].resident_fallback_used is True
    assert resolver.telemetry.unready_cold_skips == 1


def test_queue_is_bounded_and_nonblocking_when_full():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=1)

    assert resolver.enqueue(_demand()) is True
    assert resolver.enqueue(_demand(target_ids=("g-3",))) is False
    assert resolver.pending_count == 1
    assert resolver.telemetry.queue_overflow_drops == 1


def test_no_candidate_defers_current_demand_for_a_later_nonblocking_pass():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=1)
    assert resolver.enqueue(_demand())

    assert resolver.resolve_available(
        current_versions=_versions(), current_epoch=1, candidates=()
    ) == ()
    assert resolver.pending_count == 1
    assert resolver.telemetry.deferred_no_candidate == 1

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=2,
        candidates=(_candidate(17, targets=("g-2",)),),
    )
    assert intents[0].candidate_frame_ids == (17,)
    assert resolver.pending_count == 0


def test_ready_resident_fallback_beats_materializable_cold_shortlist():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=1,
        candidates=(_candidate(10, targets=("g-2",), resident=False),),
        resident_fallbacks=(_candidate(99, resident=True),),
    )

    assert intents[0].candidate_frame_ids == (99,)
    assert intents[0].resident_fallback_used is True
    assert resolver.telemetry.unready_cold_skips == 1


def test_ready_ordinary_candidate_beats_materializable_cold_candidate():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=1,
        candidates=(
            _candidate(10, targets=("g-2",), resident=False),
            _candidate(11, targets=("g-2",), resident=True),
        ),
    )

    assert intents[0].candidate_frame_ids == (11, 10)
    assert intents[0].resident_fallback_used is False


def test_shortlist_deduplicates_candidate_frame_ids_before_emitting_intent():
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=2)
    assert resolver.enqueue(_demand())

    intents = resolver.resolve_available(
        current_versions=_versions(),
        current_epoch=1,
        candidates=(
            _candidate(10, targets=("g-2",)),
            _candidate(10, targets=("g-2",)),
            _candidate(11, targets=("g-2",)),
        ),
    )

    assert intents[0].candidate_frame_ids == (10, 11)


@pytest.mark.parametrize("epoch", (float("nan"), float("inf"), -float("inf")))
def test_resolver_rejects_nonfinite_current_epoch(epoch: float):
    resolver = EvidenceDemandResolver(max_pending_demands=1, max_candidates_per_intent=1)

    with pytest.raises(ValueError, match="finite"):
        resolver.resolve_available(
            current_versions=_versions(), current_epoch=epoch, candidates=()
        )
