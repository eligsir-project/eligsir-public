"""Mapping scheduler package."""

from .backpressure import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureDecision,
    BackpressureOwner,
    BackpressureTelemetry,
    PendingObservation,
    PendingObservationCohort,
    PressureState,
    as_observation_decision,
    backpressure_config_from_args,
    burst_config_as_signal_provider,
    pose_delta_signals,
)
from .burst_budget import BurstBudget, BurstConfig
from .frame_scheduler import (
    COMPOSED_DECIMATION_NOTICE,
    INGESTION_MODE_AUTO,
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_ITER_BASED,
    INGESTION_MODE_WALLCLOCK_STRICT,
    INGESTION_MODES,
    INGESTION_REASON_EXPLICIT,
    INGESTION_REASON_OFFLINE_DEFAULT,
    INGESTION_REASON_ONLINE_MODE_DEFAULT,
    INGESTION_REASON_ROS2_DEFAULT,
    INGESTION_REASON_UNKNOWN_VALUE,
    ONLINE_MODE_DATASET_FPS_WARNING,
    SOURCE_SAMPLING_MODE_LEGACY_COMPOSE,
    SOURCE_SAMPLING_MODE_SINGLE_OWNER,
    SOURCE_SAMPLING_MODES,
    ConflictingStrideError,
    FrameScheduler,
    resolve_ingestion_mode,
    resolve_ingestion_mode_with_reason,
    resolve_legacy_composition_mode,
    resolve_source_decimation,
    resolve_source_sampling_application,
    resolve_source_sampling_policy,
)
from .frame_quality import (
    FrameQuality,
    FrameQualityConfig,
    RunningBlurExposureStats,
    evaluate_frame_quality,
    normalized_blur_score,
)
from .historical_candidate_index import (
    HistoricalCandidateConfig,
    HistoricalCandidateEvidence,
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
    HistoricalReplayCandidate,
)
from .keyframes import (
    FrameInfo,
    KeyframeAdmitter as _BaseKeyframeAdmitter,
    KeyframeAdmitterConfig,
    KeyframeDecision,
    keyframe_admitter_config_from_args,
)
from .keyframe_scoring import (
    KeyframeScore,
    KeyframeScoringConfig,
    score_keyframe,
    score_to_stats,
)


def __getattr__(name: str):
    """Load the budgeted admission wrapper without creating an import cycle.

    Candidate preparation imports ``scheduler.frame_quality`` while the
    budgeted controller itself imports candidate-prefilter types. Eagerly
    importing that wrapper from this package initializer makes either import
    order fail on a partially initialized module.
    """
    if name in {"BudgetedCohortKeyframeAdmitter", "KeyframeAdmitter"}:
        from .budgeted_keyframes import BudgetedCohortKeyframeAdmitter

        return BudgetedCohortKeyframeAdmitter
    raise AttributeError(name)
from .mapping_scheduler import MappingDecision, MappingScheduler, SchedulerStats
from .spf_resolution import (
    RESOLVED_BY,
    RESOLVED_BY_COVERAGE,
    RESOLVED_BY_MIN_SPF_FLOOR,
    RESOLVED_BY_ONLINE_BYPASS,
    RESOLVED_BY_REQUEST,
    TERMINATION_FIXED_BUDGET,
    TERMINATION_ROLLING_IDLE,
    SpfResolution,
    build_spf_resolution,
    resolve_coverage_planning_budget,
)
from .online_controller import (
    ControllerMode,
    ControllerStats,
    OnlineConfig,
    OnlineController,
    OnlineDecision,
)
from .work_budget import (
    AdaptiveGrowthProposalProvider,
    WorkBudgetActuator,
    WorkBudgetOwner,
)

__all__ = [
    "BackpressureAction",
    "BackpressureConfig",
    "BackpressureDecision",
    "BackpressureOwner",
    "BackpressureTelemetry",
    "PendingObservation",
    "PendingObservationCohort",
    "PressureState",
    "as_observation_decision",
    "backpressure_config_from_args",
    "burst_config_as_signal_provider",
    "pose_delta_signals",
    "BurstBudget",
    "BurstConfig",
    "BudgetedCohortKeyframeAdmitter",
    "COMPOSED_DECIMATION_NOTICE",
    "ConflictingStrideError",
    "ControllerMode",
    "ControllerStats",
    "INGESTION_MODE_AUTO",
    "INGESTION_MODE_DATASET_FPS",
    "INGESTION_MODE_ITER_BASED",
    "INGESTION_MODE_WALLCLOCK_STRICT",
    "INGESTION_MODES",
    "INGESTION_REASON_EXPLICIT",
    "INGESTION_REASON_OFFLINE_DEFAULT",
    "INGESTION_REASON_ONLINE_MODE_DEFAULT",
    "INGESTION_REASON_ROS2_DEFAULT",
    "INGESTION_REASON_UNKNOWN_VALUE",
    "ONLINE_MODE_DATASET_FPS_WARNING",
    "SOURCE_SAMPLING_MODE_LEGACY_COMPOSE",
    "SOURCE_SAMPLING_MODE_SINGLE_OWNER",
    "SOURCE_SAMPLING_MODES",
    "resolve_legacy_composition_mode",
    "FrameScheduler",
    "FrameQuality",
    "FrameQualityConfig",
    "HistoricalCandidateConfig",
    "HistoricalCandidateEvidence",
    "HistoricalCandidateIndex",
    "HistoricalFrameMetadata",
    "HistoricalReplayCandidate",
    "RunningBlurExposureStats",
    "normalized_blur_score",
    "FrameInfo",
    "KeyframeAdmitter",
    "KeyframeAdmitterConfig",
    "keyframe_admitter_config_from_args",
    "KeyframeDecision",
    "KeyframeScore",
    "KeyframeScoringConfig",
    "MappingDecision",
    "MappingScheduler",
    "OnlineConfig",
    "OnlineController",
    "OnlineDecision",
    "AdaptiveGrowthProposalProvider",
    "WorkBudgetActuator",
    "WorkBudgetOwner",
    "SchedulerStats",
    "RESOLVED_BY",
    "RESOLVED_BY_COVERAGE",
    "RESOLVED_BY_MIN_SPF_FLOOR",
    "RESOLVED_BY_ONLINE_BYPASS",
    "RESOLVED_BY_REQUEST",
    "SpfResolution",
    "TERMINATION_FIXED_BUDGET",
    "TERMINATION_ROLLING_IDLE",
    "build_spf_resolution",
    "resolve_coverage_planning_budget",
    "resolve_ingestion_mode",
    "resolve_ingestion_mode_with_reason",
    "resolve_source_decimation",
    "resolve_source_sampling_application",
    "resolve_source_sampling_policy",
    "evaluate_frame_quality",
    "score_keyframe",
    "score_to_stats",
]
