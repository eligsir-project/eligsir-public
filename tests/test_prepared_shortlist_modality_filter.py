from types import SimpleNamespace

from eligsir.contracts import ModalityEligibility
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateEvidence,
    HistoricalReplayCandidate,
)
from eligsir.runtime.causal_prepared_ready import CausalPreparedReadyB4Path


def _candidate(frame_id):
    return HistoricalReplayCandidate(
        frame_id=frame_id,
        weight=1.0,
        evidence=HistoricalCandidateEvidence(
            lanes=("focus",), region_id=0, focus_weight=0.0, snapshot_id=1, epoch=1
        ),
    )


def test_causal_path_schedules_all_rgb_supervision_historical_candidates():
    candidates = tuple(_candidate(frame_id) for frame_id in (1, 2, 3))
    records = {
        1: SimpleNamespace(
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=True
            )
        ),
        2: SimpleNamespace(
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True, depth_supervision=False
            )
        ),
        3: SimpleNamespace(
            modality_eligibility=ModalityEligibility(
                rgb_supervision=False, depth_supervision=True
            )
        ),
    }
    scheduled = []
    path = object.__new__(CausalPreparedReadyB4Path)
    path._runtime = SimpleNamespace(
        initialized=True,
        prepared_cold_promotion_shortlist=lambda: candidates,
        _observation_db=SimpleNamespace(get_frame=lambda frame_id: records.get(frame_id)),
    )
    path._promotion_owner = SimpleNamespace(
        schedule_shortlist=lambda selected, **_kwargs: scheduled.extend(selected)
    )
    path._adapter = SimpleNamespace(tick=lambda _inputs: SimpleNamespace(state="deferred"))
    path._step_inputs_factory = lambda: None
    path._source_complete_requested = False
    path._source_complete_phase_marked = False
    path._source_complete_phase_hook = None

    path.tick()

    assert [candidate.frame_id for candidate in scheduled] == [1, 2]
