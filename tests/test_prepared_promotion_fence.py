from types import SimpleNamespace

import numpy as np

from eligsir.contracts import BufferRef, CameraIntrinsics, SensorPacket, SourceIdentity
from eligsir.contracts.control import (
    DispositionKind,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateEvidence,
    HistoricalReplayCandidate,
)
from eligsir.runtime.prepared_cold_promotion import (
    PreparedColdPromotionOwner,
    PreparedColdPromotionResolution,
)
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken, PreparedIngressItem
from eligsir.runtime.prepared_source_producer import PreparedSourceProduceResult
from eligsir.runtime.ready_source import ReadySourceOwner
from eligsir.tracking.providers import observation_from_pose_matrix


def _resolution(frame_id: int):
    source = SourceIdentity("fence", 0, frame_id, frame_id * 1_000)
    observation = observation_from_pose_matrix(source, np.eye(4), provider="test")
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=3,
        kind=DispositionKind.SKIP_REDUNDANT,
        observe=True,
        mapping_candidate=False,
        admit_supervision=False,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.COLD,
        reason="defer",
        modality_eligibility=ModalityEligibility(rgb_supervision=True, depth_supervision=True),
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(np.ones((2, 2), np.uint16), encoding="16UC1", units="mm"),
        intrinsics=CameraIntrinsics(fx=2, fy=2, cx=0.5, cy=0.5, width=2, height=2, depth_scale=1000),
        tracking=observation,
    )
    token = PreparedIngressControlToken(frame_id, source, 3, disposition)
    item = PreparedIngressItem(frame_id, packet, observation, disposition, token)
    return PreparedColdPromotionResolution.ready(
        item,
        promotion_permit=HistoricalPromotionPermit(source, 3, 4, 5, 6),
    )


def _candidate(frame_id: int):
    return HistoricalReplayCandidate(
        frame_id,
        1.0,
        HistoricalCandidateEvidence(("focus",), 0, 1.0, 4, 0),
    )


class _Bridge:
    def __init__(self):
        self.calls = 0

    def submit_promoted(self, *_args, **_kwargs):
        self.calls += 1
        return PreparedSourceProduceResult.ENQUEUED

    def submit_retained_commit(self, *_args, **_kwargs):
        raise AssertionError("DEFER must not use COMMIT path")


def _owner(bridge):
    ready = ReadySourceOwner(max_fresh=1, max_replay=1)
    return PreparedColdPromotionOwner(
        bridge,
        ready_scheduler=ready.claim_scheduler("scheduler"),
        resolve=lambda candidate: _resolution(candidate.frame_id),
    )


def test_stale_defer_permit_never_reaches_bridge():
    bridge = _Bridge()
    owner = _owner(bridge)
    assert owner.schedule_shortlist(
        (_candidate(1),), promotion_fence=(4, 5, 6)
    ) == 1
    # The next runtime tick publishes a newer authority before the worker gets
    # to the queued request. No worker -> runtime callback is required.
    owner.schedule_shortlist((), promotion_fence=(5, 6, 7))
    assert owner.pump_once() is False
    assert bridge.calls == 0


def test_current_defer_permit_reaches_bridge():
    bridge = _Bridge()
    owner = _owner(bridge)
    owner.schedule_shortlist((_candidate(2),), promotion_fence=(4, 5, 6))
    assert owner.pump_once() is True
    assert bridge.calls == 1


def test_defer_without_published_authority_never_reaches_bridge():
    bridge = _Bridge()
    owner = _owner(bridge)
    assert owner.schedule_shortlist((_candidate(3),), promotion_fence=None) == 1
    assert owner.pump_once() is True
    assert bridge.calls == 0


def test_worker_never_calls_runtime_fence_provider():
    bridge = _Bridge()
    owner = _owner(bridge)
    assert not hasattr(owner, "_promotion_fence_provider")
    owner.schedule_shortlist((_candidate(4),), promotion_fence=(4, 5, 6))
    assert owner.pump_once() is True
    assert bridge.calls == 1
