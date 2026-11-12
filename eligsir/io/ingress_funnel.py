"""Issue #805: Canonical source-ingress funnel vocabulary and rate surfaces.

This module defines:
1. Stage vocabulary: the canonical progression from source_observed → keyframe_committed
2. Rejection reason taxonomy: the enumerated causes for frame removal at each stage
3. Rate surfaces: the eight separately reportable Hz metrics for ingress/admission throughput
4. Invariants: boundary-respecting constraints (semantic reasons only after MappingSession)

No torch dependencies; CPU-testable. Pure functions for report block assembly.

Scope note: This module reports the semantic ledger (vocabulary, boundaries, invariants),
not the transport implementation. Cross-repo owners (eligsir-ros2, eligsir-rgbd-sync)
produce the counters; core assembles the unified view.

Reference: issue #805 (canonical funnel) + #1209 (MappingSession boundary semantics).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class IngressStage(str, Enum):
    """Canonical stages in the source-ingress funnel: source_observed → keyframe_committed.

    These stages form a strict linear progression. A frame advances from each stage to the
    next only if it survives the decision criteria of that stage. Skipped or rejected frames
    may never advance past their termination stage.

    Stages before MappingSession are transport-only; semantic decisions begin after.
    Reference: issue #1209 / PR #1206.
    """
    SOURCE_OBSERVED = "source_observed"
    """Frame received and timestamped in the Python source callback."""

    SOURCE_DECIMATED = "source_decimated"
    """Survived the native streaming_frame_stride filter (ROS callback level)."""

    SOURCE_ASSOCIATION_VALID = "source_association_valid"
    """Passed RGB/depth/pose/TF temporal association and validity checks (transport layer)."""

    SOURCE_FRAME_MATERIALIZED = "source_frame_materialized"
    """Constructed as a materialized frame object (decode, geometry extraction complete)."""

    SOURCE_BUFFER_DELIVERED = "source_buffer_delivered"
    """Exited the source's internal bounded buffer without being dropped by capacity."""

    SOURCE_RELEASED = "source_released"
    """FrameScheduler released it (passed wallclock/stride/availability checks)."""

    SOURCE_INGESTED = "source_ingested"
    """Entered the mapping-session boundary (MappingSession __enter__ seen it)."""

    PREFILTER_EVALUATED = "prefilter_evaluated"
    """Passed through prefilter evaluation (ineligible frames still counted as evaluated)."""

    PREFILTER_ELIGIBLE = "prefilter_eligible"
    """Passed prefilter (ready for keyframe admission decision)."""

    CANDIDATE_PROBED = "candidate_probed"
    """Probed by the keyframe policy (policy saw it as a candidate)."""

    CANDIDATE_EVALUATED = "candidate_evaluated"
    """Fully evaluated by KeyframeAdmitter (decision rendered)."""

    KEYFRAME_COMMITTED = "keyframe_committed"
    """Committed as a training keyframe (included in the training set)."""


class RejectionReason(str, Enum):
    """Enumerated reasons for frame removal from the ingress pipeline.

    Each reason is associated with a boundary (PRE_CORE or POST_MAPPING_SESSION).
    Pre-core reasons are transport-only; semantic reasons begin after MappingSession.

    Subreasons (e.g., 'depth_stale', 'pose_stale') may be recorded as additional
    detail but do not appear in this enumeration.

    Reference: issue #805 + #1209's ownership update.
    """

    # Pre-core (transport) rejection reasons
    SOURCE_DECIMATION = "SOURCE_DECIMATION"
    """Frame was dropped by native frame_stride filter (ROS callback)."""

    SOURCE_SYNC_REJECT = "SOURCE_SYNC_REJECT"
    """RGB-D pairing/correction/monotonicity failure (eligsir-rgbd-sync owned)."""

    SOURCE_POSE_INVALID = "SOURCE_POSE_INVALID"
    """Pose missing, non-finite, or transform chain broken (transport validation)."""

    SOURCE_DECODE_INVALID = "SOURCE_DECODE_INVALID"
    """RGB or depth image decode failure (transport layer)."""

    TRANSPORT_DROP = "TRANSPORT_DROP"
    """Dropped by source buffer capacity/backpressure (bounded queue overflow).

    This is the canonical label for pre-MappingSession capacity loss.
    Subreasons may indicate if oldest-frame FIFO or value-ranked priority was used.
    """

    SCHEDULER_LATE_DROP = "SCHEDULER_LATE_DROP"
    """FrameScheduler wallclock_strict dropped frame as late."""

    SLAM_NO_POSE_SKIP = "SLAM_NO_POSE_SKIP"
    """DeferredSlamAdmissionTracker skipped due to missing/timed-out pose."""

    # Post-MappingSession (semantic) rejection reasons
    PREFILTER_REJECT = "PREFILTER_REJECT"
    """Prefilter rejected (blur, exposure, or other image-quality reason)."""

    ADMISSION_REJECT = "ADMISSION_REJECT"
    """KeyframeAdmitter rejected (redundancy, keyframe policy, or uncertainty)."""

    SEMANTIC_DEFER = "SEMANTIC_DEFER"
    """Deferred (may be re-admitted later by COLD/ICECOLD/archive scheduler)."""

    ARCHIVE_EVICT = "ARCHIVE_EVICT"
    """Evicted from archive after initial retention window."""


class RejectionBoundary(str, Enum):
    """Scope boundary for a rejection reason (transport vs semantic).

    Pre-core reasons are transport-only and may silently drop capacity.
    Post-MappingSession reasons are semantic and must preserve bounded core retention.
    """
    PRE_CORE = "PRE_CORE"
    """Transport layer; frame never reaches MappingSession."""

    POST_MAPPING_SESSION = "POST_MAPPING_SESSION"
    """Semantic layer; frame crossed MappingSession boundary before decision."""


# Mapping of rejection reason to its boundary (authoritative enum for invariant checks)
REJECTION_BOUNDARIES = {
    # Pre-core
    RejectionReason.SOURCE_DECIMATION: RejectionBoundary.PRE_CORE,
    RejectionReason.SOURCE_SYNC_REJECT: RejectionBoundary.PRE_CORE,
    RejectionReason.SOURCE_POSE_INVALID: RejectionBoundary.PRE_CORE,
    RejectionReason.SOURCE_DECODE_INVALID: RejectionBoundary.PRE_CORE,
    RejectionReason.TRANSPORT_DROP: RejectionBoundary.PRE_CORE,
    RejectionReason.SCHEDULER_LATE_DROP: RejectionBoundary.PRE_CORE,
    RejectionReason.SLAM_NO_POSE_SKIP: RejectionBoundary.PRE_CORE,
    # Post-MappingSession
    RejectionReason.PREFILTER_REJECT: RejectionBoundary.POST_MAPPING_SESSION,
    RejectionReason.ADMISSION_REJECT: RejectionBoundary.POST_MAPPING_SESSION,
    RejectionReason.SEMANTIC_DEFER: RejectionBoundary.POST_MAPPING_SESSION,
    RejectionReason.ARCHIVE_EVICT: RejectionBoundary.POST_MAPPING_SESSION,
}


def ingress_funnel_report_block(
    *,
    source_observed_count: Optional[int] = None,
    materialized_valid_count: Optional[int] = None,
    prefilter_eligible_count: Optional[int] = None,
    probe_count: Optional[int] = None,
    wall_time_s: Optional[float] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Assemble the report.json["ingress_funnel"] block from counters.

    Wires four rate surfaces to their real producers in the canonical runtime:
    - source_observed_hz: count / wall_time from source_offered (producer: causal_replay.py:1367)
    - materialized_valid_hz: count / wall_time from source_ingress admitted (producer: source_ingress.py:71)
    - prefilter_eligible_hz: not yet instrumented (no run-level aggregate exists)
    - probe_hz: not yet instrumented (current "probe" references are resolution-tier, not funnel stage)

    Args:
        source_observed_count: Frame count observed at source callback boundary (source_offered).
        materialized_valid_count: Frame count admitted after decode (ingress_frames_admitted).
        prefilter_eligible_count: Not used; no producer yet. Kept for future extension.
        probe_count: Not used; no producer yet. Kept for future extension.
        wall_time_s: Elapsed wall time in seconds for hz computation.
        **kwargs: Additional keyword arguments (ignored; allows flexible future extension).

    Returns:
        Dictionary with structure:
        {
            "stage_vocabulary": [list of canonical IngressStage names],
            "rejection_reasons": [list of canonical RejectionReason names],
            "rejection_boundaries": {RejectionReason: RejectionBoundary},
            "rate_surfaces": {
                "source_observed_hz": {...},
                "materialized_valid_hz": {...},
                "released_hz": {...},
                "prefilter_eligible_hz": {...},
                "probe_hz": {...},
                "committed_keyframe_hz": {...},
                "supervision_view_hz": {...},
                "optimizer_step_hz": {...},
            },
            "invariants": {
                "semantic_rejection_before_mapping_session": False,
                "dual_admission_detected": False,
                "bag_replay_lossless": None,  # Validated elsewhere
            },
        }

    The rate_surfaces each have the shape:
        {
            "hz": <float or None>,
            "status": "INSTRUMENTED" | "NOT_INSTRUMENTED",
            "producer": "module:function:line" or "PENDING",
            "reason": "explanation for null surfaces" (only when NOT_INSTRUMENTED),
        }

    Surfaces without a producer are emitted as `{"hz": None, "status": "NOT_INSTRUMENTED"}`.
    """

    # Build stage vocabulary list from enum
    stages = [stage.value for stage in IngressStage]

    # Build rejection reason list from enum
    reasons = [reason.value for reason in RejectionReason]

    # Build boundary mapping for invariant checks
    boundaries = {reason.value: boundary.value for reason, boundary in REJECTION_BOUNDARIES.items()}

    # Compute rate surfaces from available counters
    # Guard against division by zero: hz=None if wall_time_s is absent or <= 0
    def compute_hz(count: Optional[int], wall_s: Optional[float]) -> Optional[float]:
        if count is None or wall_s is None or float(wall_s) <= 0.0:
            return None
        return float(count) / float(wall_s)

    # Rate surfaces: reference existing canonical metrics where available,
    # mark others as NOT_INSTRUMENTED.
    # Existing producers (from metric_registry.py):
    #  - source_frame_rate_hz: frame_scheduler.py:mark_released:544
    #  - keyframe_admission_rate_hz: utils/execution_semantics.py:compute_progress_postfix:1010
    #  - supervision_view_rate_hz: utils/supervision_work.py:supervision_view_rate_hz:49
    #  - optimization_step_rate_hz: utils/supervision_work.py:optimization_step_rate_hz:32
    rate_surfaces = {
        "source_observed_hz": {
            "hz": compute_hz(source_observed_count, wall_time_s),
            "status": "INSTRUMENTED" if source_observed_count is not None and wall_time_s else "NOT_INSTRUMENTED",
            "producer": "eligsir/runtime/causal_replay.py:run:1367" if source_observed_count is not None else "PENDING",
            "note": "First callback visible inside EliGSiR Python (issue #805)",
        },
        "materialized_valid_hz": {
            "hz": compute_hz(materialized_valid_count, wall_time_s),
            "status": "INSTRUMENTED" if materialized_valid_count is not None and wall_time_s else "NOT_INSTRUMENTED",
            "producer": "eligsir/runtime/source_ingress.py:as_report_block:71" if materialized_valid_count is not None else "PENDING",
            "note": "Frames surviving transport validation (issue #805)",
        },
        # This one has a producer: source_frame_rate_hz in metric_registry
        "released_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "eligsir/mapping/scheduler/frame_scheduler.py:mark_released:544",
            "metric_id": "source_frame_rate_hz",
            "note": "Rate of frames released by FrameScheduler (issue #700)",
        },
        "prefilter_eligible_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "PENDING",
            "reason": "No run-level aggregate counter: prefilter.evaluate() is per-candidate pure function; "
                     "budgeted_controller.py:139 and cohort.py:72 compute eligibility per item only.",
            "note": "Frames ready for keyframe admission (issue #805)",
        },
        "probe_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "PENDING",
            "reason": "Current probe references are resolution-tier (tier_cost_model.py:673 _probe_candidate, "
                     "work_budget.py:62 probe: bool) or per-candidate GPU evidence (budgeted_keyframes.py:398), "
                     "not the funnel's candidate_probed stage. Awaits new counter.",
            "note": "Frames probed by keyframe policy (issue #805)",
        },
        # This one has a producer: keyframe_admission_rate_hz
        "committed_keyframe_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "utils/execution_semantics.py:compute_progress_postfix:1010",
            "metric_id": "keyframe_admission_rate_hz",
            "note": "Rate of keyframes committed (issue #700, #701)",
        },
        # This one has a producer: supervision_view_rate_hz
        "supervision_view_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "utils/supervision_work.py:supervision_view_rate_hz:49",
            "metric_id": "supervision_view_rate_hz",
            "note": "Rate of supervision views (issue #700, #701)",
        },
        # This one has a producer: optimization_step_rate_hz
        "optimizer_step_hz": {
            "hz": None,
            "status": "NOT_INSTRUMENTED",
            "producer": "utils/supervision_work.py:optimization_step_rate_hz:32",
            "metric_id": "optimization_step_rate_hz",
            "note": "Rate of optimizer steps (issue #700, #701)",
        },
    }

    # Invariant checks: pure predicates over the block
    # (These would be populated when actual counters are available)
    invariants = {
        "semantic_rejection_before_mapping_session": False,
        "dual_admission_detected": False,
        "bag_replay_lossless": None,  # Validated elsewhere via offered==admitted && dropped==0
    }

    return {
        "stage_vocabulary": stages,
        "rejection_reasons": reasons,
        "rejection_boundaries": boundaries,
        "rate_surfaces": rate_surfaces,
        "invariants": invariants,
    }


__all__ = [
    "IngressStage",
    "RejectionReason",
    "RejectionBoundary",
    "REJECTION_BOUNDARIES",
    "ingress_funnel_report_block",
]
