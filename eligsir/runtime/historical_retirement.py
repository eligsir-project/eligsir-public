"""Modality-aware historical-retirement boundary.

The canonical runtime has two materially different reasons an identity can no
longer participate in RGB supervision:

* its last route back to source pixels disappeared (physical
  ``unmaterializable`` eviction); and
* its retained, immutable modality eligibility says RGB supervision is not
  available even though other payloads (notably depth) remain materializable.

The first case keeps the existing ``CanonicalMappingRuntime`` retirement path,
including the unmaterializable counter/operator warning.  The second case only
withdraws the identity from the current RGB-supervision scheduler lanes.  It
must never claim that pixels were evicted or delete the durable DB provenance
that a future depth/depth-insertion lane may still use.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MethodType
from typing import Any


def _withdraw_rgb_supervision(runtime: Any, frame_id: int, source: Any) -> bool:
    """Withdraw one materializable, permanently RGB-ineligible identity.

    This mirrors the scheduler-facing portion of the legacy all-lanes
    retirement without touching FrameObservationDB provenance and without
    incrementing the physical-eviction accounting.  The existing bounded
    ``_retired_replay_sources`` tombstone is retained as a compatibility guard
    against ``_idle_replay_plan`` resurrecting the previous RGB supervision
    source. A future dedicated depth-only replay lane may explicitly re-register
    the durable DB record; its pixels/provenance are deliberately preserved.
    """

    frame_id = int(frame_id)
    withdrawn = getattr(runtime, "_rgb_supervision_withdrawn_frame_ids", None)
    if withdrawn is None:
        withdrawn = runtime._rgb_supervision_withdrawn_frame_ids = OrderedDict()
    first_withdrawal = frame_id not in withdrawn
    withdrawn[frame_id] = True
    while len(withdrawn) > 1024:
        withdrawn.popitem(last=False)

    candidate_index = getattr(runtime, "_historical_candidate_index", None)
    retire = getattr(candidate_index, "retire", None)
    if callable(retire):
        retire(frame_id)

    focus_weights = getattr(runtime, "_replay_focus_frame_weights", ())
    if any(int(item[0]) == frame_id for item in focus_weights):
        runtime._replay_focus_frame_weights = tuple(
            item for item in focus_weights if int(item[0]) != frame_id
        )
    focus_sources = getattr(runtime, "_replay_focus_sources", ())
    if source in focus_sources:
        runtime._replay_focus_sources = tuple(
            item for item in focus_sources if item != source
        )

    for lane_name in (
        "_replay_modalities",
        "_replay_weights",
        "_pending_supervision_modalities",
    ):
        lane = getattr(runtime, lane_name, None)
        if lane is not None:
            lane.pop(source, None)

    clear_obligation = getattr(
        getattr(runtime, "_observation_db", None),
        "clear_prepared_commit_obligation",
        None,
    )
    if callable(clear_obligation):
        clear_obligation(source)

    # Compatibility guard: the old idle fallback can re-mint work from the
    # previous plan even after the shortlist has retired this row.  Preserve
    # its existing bounded tombstone mechanism, but do not delete the DB/source
    # provenance maps as the physical-eviction path does.
    retired = getattr(runtime, "_retired_replay_sources", None)
    if retired is None:
        retired = runtime._retired_replay_sources = OrderedDict()
    retired[source] = True
    while len(retired) > 1024:
        retired.popitem(last=False)

    invalidate = getattr(runtime, "_invalidate_prepared_ready_plan_for_source", None)
    if callable(invalidate):
        invalidate(source)

    if first_withdrawal:
        runtime._retired_rgb_ineligible_total = int(
            getattr(runtime, "_retired_rgb_ineligible_total", 0)
        ) + 1
    return first_withdrawal


def install_modality_aware_historical_retirement(runtime: Any) -> None:
    """Install the RGB-vs-unmaterializable dispatcher exactly once.

    ``FrameObservationDB`` registers the original bound retirement callback
    during runtime construction.  Installing this wrapper later therefore does
    not change true payload-eviction semantics: those callbacks still invoke
    the original physical-retirement method.  Dynamic runtime calls such as
    ``_camera_for_source`` use this wrapper and can distinguish a materializable
    RGB-ineligible record from real pixel loss.
    """

    if bool(getattr(runtime, "_modality_aware_retirement_installed", False)):
        return
    original = getattr(runtime, "_retire_historical_identity", None)
    if not callable(original):
        return

    def _dispatch(self: Any, frame_id: int, *, reason: str = "unmaterializable") -> bool:
        db = getattr(self, "_observation_db", None)
        sources = getattr(self, "_historical_sources_by_frame_id", None) or {}
        source = sources.get(int(frame_id))
        if db is None or source is None:
            return bool(original(frame_id, reason=reason))

        # A materializable record still exists.  If the same record cannot
        # satisfy RGB supervision, this is modality ineligibility, not an
        # unmaterializable eviction.  Preserve its DB/provenance for future
        # modality-specific work and withdraw it only from current RGB replay.
        any_candidate = db.historical_candidate_for_source(source)
        rgb_candidate = db.historical_candidate_for_source(source, require_rgb=True)
        if any_candidate is not None and rgb_candidate is None:
            _withdraw_rgb_supervision(self, int(frame_id), source)
            return True

        return bool(original(frame_id, reason=reason))

    runtime._retire_historical_identity = MethodType(_dispatch, runtime)
    runtime._modality_aware_retirement_installed = True


__all__ = [
    "install_modality_aware_historical_retirement",
]
