"""Issue #1302 (kitchen1): a durably RGB-ineligible DEFER'd identity must be
retired, not left alive forever to re-abort every future supervision tick.

Root-caused on a real kitchen1 bag replay through `deterministic_bag.launch.py`
(eligsir-ros2's production ingress path): `optimizer_steps`/
`supervision_views` were stuck at 0 for the ENTIRE run (230+s post-bag,
30,000+ ticks) despite 262 pairs being admitted to supervision. Traced to
`_camera_for_source`'s "not matches" branch:

- The promotion attempt (a few lines above the code this test exercises)
  always calls ``historical_candidate_for_source(source, require_rgb=True)``
  -- there is no depth-only promotion path.
- The retirement decision used to call ``historical_candidate_for_source
  (source)`` WITHOUT ``require_rgb`` to decide whether to retire. A no-motion
  DEFER's ``ModalityEligibility`` is set once, at record time, and never
  mutated (see ``ObservationDisposition``/``FrameRecord``) -- so an identity
  that is depth-eligible but NOT rgb-eligible passes the laxer retirement
  check (a usable-for-*some*-modality candidate genuinely exists) and is
  therefore never retired, while the promotion attempt's stricter
  ``require_rgb=True`` query can *never* succeed for it. The two checks
  disagreeing left such an identity permanently un-promotable AND
  permanently un-retired -- the planner kept re-proposing the exact same
  doomed identity forever, aborting the whole supervision transaction with
  ``"no_training_camera"`` on every tick.

kitchen1's admission pattern (~85% of frames rejected as `no_motion`, DEFER'd
with real depth but no RGB-eligible modality) hit this on effectively every
tick; floor2's much lower DEFER ratio evidently never (or rarely) surfaced it,
which is why the failure only manifested reproducibly on kitchen1's bag.
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
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex,
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


def _depth_only_defer_disposition(ordinal: int) -> ObservationDisposition:
    """A no-motion DEFER whose frozen modality is depth-eligible, not RGB."""

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
            rgb_supervision=False,
            rgb_quality="declined",
            depth_supervision=True,
            depth_quality="qualified",
            depth_insertion=True,
        ),
    )


def _runtime() -> CanonicalMappingRuntime:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = FrameObservationDB(
        device="cpu",
        max_cold_payload_frames=128,
        max_cold_payload_bytes=1_000_000,
    )
    runtime._observation_db.add_payload_eviction_observer(
        runtime._retire_historical_identity
    )
    runtime._retired_unmaterializable_total = 0
    runtime._retired_unmaterializable_warned = False
    runtime._retired_rgb_ineligible_total = 0
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


def _retain_depth_only(runtime: CanonicalMappingRuntime, ordinal: int):
    packet = _live_packet(ordinal)
    disposition = _depth_only_defer_disposition(ordinal)
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


def test_depth_only_identity_is_retired_not_reproposed_forever() -> None:
    """The exact #1302 kitchen1 mechanism: a live depth-only DEFER record.

    Before the fix, ``_camera_for_source`` returned ``None`` for this source
    without retiring it -- so a planner that keeps naming it (as kitchen1's
    real admission pattern did, tick after tick) would abort every single
    supervision transaction forever with ``no_training_camera``.
    """

    runtime = _runtime()
    source, record = _retain_depth_only(runtime, 0)

    # A genuine, materializable historical record exists (depth-eligible) --
    # this is not the #1289 eviction case.
    assert runtime._observation_db.frame_id_for_source(source) == record.frame_id
    assert (
        runtime._observation_db.historical_candidate_for_source(source) is not None
    )
    assert (
        runtime._observation_db.historical_candidate_for_source(
            source, require_rgb=True
        )
        is None
    )

    assert runtime._camera_for_source(source) is None

    # The fix: this identity can never satisfy the promotion path's
    # require_rgb=True query (modality eligibility is frozen), so it must be
    # retired immediately -- not kept alive by the laxer "usable for some
    # modality" check that used to gate retirement here.
    #
    # issue #1405: this record is fully materializable (real depth pixels are
    # retained), it is only RGB-supervision-ineligible -- so it must NOT
    # increment the unmaterializable/payload-eviction counter, only the
    # separate RGB-ineligibility one.
    assert runtime._retired_unmaterializable_total == 0
    assert runtime._retired_rgb_ineligible_total == 1
    assert source not in runtime._replay_modalities
    assert source not in runtime._replay_weights
    assert record.frame_id not in runtime._historical_sources_by_frame_id
    assert source not in runtime._historical_frame_ids_by_source

    # A repeat visit (the planner re-proposing the same now-unregistered
    # identity) must keep skipping gracefully -- never crash, never re-retire.
    assert runtime._camera_for_source(source) is None
    assert runtime._retired_unmaterializable_total == 0
    assert runtime._retired_rgb_ineligible_total == 1


def test_planner_converges_past_a_depth_only_identity_onto_a_promotable_one() -> None:
    """Direct evidence the fix unblocks progress, not just avoids a crash.

    A depth-only identity must not poison the shortlist for a
    genuinely-promotable (RGB-eligible) identity recorded around it.
    """

    runtime = _runtime()
    dead_source, _dead_record = _retain_depth_only(runtime, 0)

    live_packet = _live_packet(1)
    live_disposition = ObservationDisposition(
        source=live_packet.source,
        snapshot_id=5,
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
    live_record = runtime._retain_deferred_observation(
        live_disposition,
        ColdRetentionPermit(live_packet.source, 5),
        live_packet,
        live_packet.tracking,
    )
    runtime._register_historical_record(live_record, live_packet.source)
    runtime._replay_modalities[live_packet.source] = live_disposition.modality_eligibility
    runtime._replay_weights.setdefault(live_packet.source, 1.0)

    # The dead (depth-only) identity is skipped and retired -- as an RGB-
    # ineligibility withdrawal (issue #1405), not a pixel-loss eviction.
    assert runtime._camera_for_source(dead_source) is None
    assert runtime._retired_unmaterializable_total == 0
    assert runtime._retired_rgb_ineligible_total == 1

    # ...and the genuinely promotable identity registered alongside it is
    # completely unaffected: it remains a real, shortlisted candidate.
    shortlisted = {
        candidate.frame_id
        for candidate in runtime._historical_candidate_index.shortlist(
            snapshot_id=1, epoch=0
        )
    }
    assert live_record.frame_id in shortlisted
    assert live_packet.source in runtime._replay_modalities
