"""Issue #1289: an evicted live identity must leave every scheduler lane.

A deferred observation from a live (non-reloadable) source keeps its pixels
only in the bounded cold payload store. Once that store evicts it there is no
route back to real pixels, so no camera can ever be built for it again. Before
this fix the scheduler's replay lanes kept proposing such an identity, and
``_camera_for_source`` aborted the whole mapping worker with
``planned supervision identity did not resolve to exactly one camera``.

This is reproducible without ROS 2 and without threads: it is a capacity /
invalidation gap, not a race.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eligsir.contracts.control import (
    DispositionKind,
    HistoricalEligibility,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.mapping.admission.observation_owner import ColdRetentionPermit
from eligsir.mapping.observation_db import ColdPixelHandle, FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.tracking.providers import observation_from_pose_matrix


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("live", 0, ordinal, 1_000_000_000 + ordinal)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        width=2,
        height=2,
        depth_fx=2.0,
        depth_fy=2.0,
        depth_cx=1.0,
        depth_cy=1.0,
        depth_width=2,
        depth_height=2,
        depth_scale=1.0,
    )


def _live_packet(ordinal: int) -> SensorPacket:
    """A live RGB-D packet: real pixels, and deliberately no reload token."""

    source = _source(ordinal)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((2, 2), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=_intrinsics(),
        tracking=observation_from_pose_matrix(
            source, np.eye(4), provider="dataset_pose"
        ),
        reload_token=None,
    )


def _disposition(ordinal: int) -> ObservationDisposition:
    return ObservationDisposition(
        source=_source(ordinal),
        snapshot_id=4,
        kind=DispositionKind.SKIP_REDUNDANT,
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
        observe=True,
        mapping_candidate=False,
        allow_geometry_insertion=False,
        admit_supervision=False,
        residency=ResidencyIntent.COLD,
        reason="no_motion",
        modality_eligibility=ModalityEligibility(
            True, "qualified", True, "qualified", True
        ),
    )


def _runtime(*, max_cold_payload_frames: int) -> CanonicalMappingRuntime:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=max_cold_payload_frames,
        max_cold_payload_bytes=1_000_000,
    )
    runtime._observation_db.add_payload_eviction_observer(
        runtime._retire_historical_identity
    )
    runtime._retired_unmaterializable_total = 0
    runtime._retired_unmaterializable_warned = False
    runtime._historical_sources_by_frame_id = {}
    runtime._historical_frame_ids_by_source = {}
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._replay_modalities = {}
    runtime._replay_weights = {}
    runtime._pending_supervision_modalities = {}
    runtime._source_identities_by_ordinal = {}
    runtime.scene = SimpleNamespace(
        train_cameras=[],
        current_camera=None,
        admitted_camera_for_source=lambda _source: None,
    )
    return runtime


def _retain(runtime: CanonicalMappingRuntime, ordinal: int):
    """Mirror the terminal-disposition DEFER path for one live observation."""

    packet = _live_packet(ordinal)
    disposition = _disposition(ordinal)
    observation = packet.tracking
    assert observation is not None
    record = runtime._retain_deferred_observation(
        disposition,
        ColdRetentionPermit(packet.source, 4),
        packet,
        observation,
    )
    runtime._register_historical_record(record, packet.source)
    runtime._replay_modalities[packet.source] = disposition.modality_eligibility
    runtime._replay_weights.setdefault(packet.source, 1.0)
    return packet.source, record


def test_evicted_live_identity_leaves_every_replay_lane() -> None:
    runtime = _runtime(max_cold_payload_frames=1)

    evicted_source, evicted_record = _retain(runtime, 0)
    resident_source, resident_record = _retain(runtime, 1)

    # The first identity's pixels are gone for good: nothing may still offer it
    # as supervision work.
    assert runtime._observation_db.get_cold_pixels(
        evicted_record.frame_id
    ).can_materialize is False
    assert evicted_source not in runtime._replay_modalities
    assert evicted_source not in runtime._replay_weights
    assert evicted_record.frame_id not in runtime._historical_sources_by_frame_id
    assert evicted_source not in runtime._historical_frame_ids_by_source
    assert runtime._retired_unmaterializable_total == 1

    shortlisted = {
        candidate.frame_id
        for candidate in runtime._historical_candidate_index.shortlist(
            snapshot_id=1, epoch=0
        )
    }
    assert shortlisted == {resident_record.frame_id}
    assert resident_source in runtime._replay_modalities


def test_camera_for_evicted_identity_skips_instead_of_killing_the_worker() -> None:
    """The exact issue #1289 crash: a planned identity with no pixels left."""

    runtime = _runtime(max_cold_payload_frames=1)
    evicted_source, _record = _retain(runtime, 0)
    _retain(runtime, 1)

    # Even if a stale plan still names the retired identity, resolution must
    # yield "no camera for this transaction", never abort the mapping worker.
    assert runtime._camera_for_source(evicted_source) is None


def test_reloadable_identity_is_never_retired_by_cold_eviction() -> None:
    """Dataset/bag sources page back in from disk and must stay selectable."""

    db = FrameObservationDB(
        device="cpu", max_cold_payload_frames=1, max_cold_payload_bytes=64
    )
    retired: list[int] = []
    db.add_payload_eviction_observer(retired.append)

    from eligsir.mapping.observation_db import (
        BufferStage,
        FrameRecord,
        ModalityEligibility as DbModalityEligibility,
    )
    import torch

    for frame_id in (0, 1):
        db.add_frame(
            FrameRecord(
                frame_id=frame_id,
                pose_c2w=np.eye(4, dtype=np.float32),
                timestamp=float(frame_id),
                submap_id=0,
                buffer_stage=BufferStage.COLD_CACHE,
                visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
                source_identity=f"bag:{frame_id}",
                source_ordinal=frame_id,
                historical_eligible=True,
                modality_eligibility=DbModalityEligibility(
                    rgb_supervision=True, depth_supervision=True
                ),
            )
        )
        db.store_cold_pixels(
            frame_id,
            ColdPixelHandle(
                source_identity=f"bag:{frame_id}",
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                depth=np.zeros((2, 2), dtype=np.float32),
                intrinsics="bag-calibration",
                reload_token=f"bag-offset:{frame_id}",
            ),
        )

    assert retired == []
    assert db.unmaterializable_evictions == 0
    assert len(db.historical_candidates(require_rgb=True)) == 2


def test_candidate_index_retire_clears_lanes_and_reelects_the_anchor() -> None:
    index = HistoricalCandidateIndex()
    for frame_id in range(4):
        index.register(
            HistoricalFrameMetadata(
                frame_id=frame_id, region_id=0, protected=frame_id == 1
            )
        )

    assert index.retire(0) is True
    assert index.retire(0) is False
    assert index.history_size == 3

    frame_ids = {
        candidate.frame_id
        for candidate in index.shortlist(snapshot_id=1, epoch=0)
    }
    assert frame_ids == {1, 2, 3}
    # The permanent oldest-history anchor must move to a surviving identity.
    assert index._anchor_id == 1
