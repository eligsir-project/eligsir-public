"""Issue #1302: a planned identity the observation DB never recorded must
also skip gracefully, not abort the mapping worker.

Issue #1289 fixed one way `_camera_for_source` could crash with "planned
supervision identity did not resolve to exactly one camera": a DEFER'd live
observation whose cold pixels had already been evicted, but which the
observation DB still *knew about* (``frame_id_for_source`` resolved). This is
a genuinely different gap in the same function: a source the observation DB
has **never heard of at all** -- ``frame_id_for_source`` returns ``None`` --
which fell straight through #1289's defence-in-depth branch (that branch
only handles the ``frame_id is not None`` case) into the same fatal raise.

Reproduced on a real floor2 bag replay (2026-08-30, issue #1302) under
pathologically sparse sync-node admission (only 11 RGB-D pairs matched and
published over a 145.7s bag): the mapping worker died on the 8th admitted
pair with exactly this traceback. That specific crash turned out to also be
explainable by a stale checkout (the deployed image predated #1289's fix by
about seven hours -- see the issue for the full accounting), but this test
demonstrates the residual gap exists independently, on current ``main``,
with no ROS 2 and no threads involved: it is a second capacity/registration
gap, not a race.
"""

from __future__ import annotations

from types import SimpleNamespace

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("live", 0, ordinal, 1_000_000_000 + ordinal)


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


def test_camera_for_never_registered_source_skips_instead_of_killing_the_worker() -> None:
    """A plan naming an identity the DB never recorded must not crash."""

    runtime = _runtime()

    # This source was never retained/registered through any of
    # `_retain_deferred_observation` / `_register_historical_record` --
    # `frame_id_for_source` genuinely has nothing to return for it.
    never_registered = _source(999)
    assert runtime._observation_db.frame_id_for_source(never_registered) is None

    assert runtime._camera_for_source(never_registered) is None

    # There is no frame_id to retire, so this must not have fabricated one
    # or perturbed the (empty) candidate index.
    assert runtime._retired_unmaterializable_total == 0
    assert runtime._historical_candidate_index.history_size == 0
