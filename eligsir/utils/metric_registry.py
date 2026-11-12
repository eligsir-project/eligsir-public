"""Machine-readable canonical metric registry for runtime metrics and telemetry.

This module is deliberately **torch-free and import-side-effect-free** so it can
be unit-tested in the lean CPU CI image without side effects on train.py/
train_streaming.py -- the same scope fence :mod:`utils.execution_semantics`
documents for issues #423/#462/#478. It imports nothing from train.py,
train_streaming.py, or arguments/ at module scope.

See issue #704 for the full audit and issue #700 for semantic correctness
context. This registry serves as the machine-readable definition of:

1. Every canonical metric and its event population / numerator / denominator.
2. Legacy aliases and their historical formulas (read during migration).
3. Name collisions and same-semantics duplicates discovered in audit.
4. Dead/unconsumed metrics that are safe to remove.

Registry versioning: each metric carries a semantic_version; a version bump
signals that the measured event population, window, or denominator has changed.
A new version does NOT automatically invalidate historical values -- consumers
explicitly decide whether to read new or legacy versions via fallback chains.

Authors: Claude Code agent (issue #704), 2026-08-15.
Base commit pinned: see git log for this file's first appearance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class MetricClassification(str, Enum):
    """Classification of a discovered metric's relationship to the canonical registry."""
    CANONICAL = "CANONICAL"  # Authoritative definition; only one per event population
    LEGACY_ALIAS = "LEGACY_ALIAS"  # Historical name; resolves to a canonical entry
    DUPLICATE_SAME_SEMANTICS = "DUPLICATE_SAME_SEMANTICS"  # Same event, different producer
    NAME_COLLISION_DIFFERENT_SEMANTICS = "NAME_COLLISION_DIFFERENT_SEMANTICS"  # Same name, incompatible formulas
    MISNAMED = "MISNAMED"  # Name does not match measured event or window
    FORMULA_SUSPECT = "FORMULA_SUSPECT"  # Formula documented vs actual divergence
    DEAD_OR_UNCONSUMED = "DEAD_OR_UNCONSUMED"  # Not consumed by any downstream tool
    HISTORICAL_ONLY = "HISTORICAL_ONLY"  # Retained for baseline/champion compatibility
    PENDING = "PENDING"  # Owned by another issue (#702/#703); not yet implemented


class AggregationWindowKind(str, Enum):
    """Type of aggregation window for rate/mean computations."""
    ROLLING = "rolling"  # Recent N-second window
    FIXED = "fixed"  # Fixed iteration/sample boundaries
    LIFETIME = "lifetime"  # Entire run (no window)
    SNAPSHOT = "snapshot"  # Single-iteration instantaneous value


class ExecutionMode(str, Enum):
    """Training execution mode classification (issue #168)."""
    SINGLE_VIEW = "SINGLE_VIEW"
    SEQUENTIAL_GRADIENT_ACCUMULATION = "SEQUENTIAL_GRADIENT_ACCUMULATION"
    TRUE_CAMERA_BATCH = "TRUE_CAMERA_BATCH"


class ValidationStatus(str, Enum):
    """Validation status of a metric definition (issue #704)."""
    UNIMPLEMENTED = "UNIMPLEMENTED"  # Producer not yet implemented
    IMPLEMENTED_UNVALIDATED = "IMPLEMENTED_UNVALIDATED"  # Implemented but not yet validated
    VALIDATED_SYNTHETIC = "VALIDATED_SYNTHETIC"  # Validated on synthetic data
    VALIDATED_REFERENCE_ARTIFACT = "VALIDATED_REFERENCE_ARTIFACT"  # Validated against reference artifact
    VALIDATED_REAL_RUN = "VALIDATED_REAL_RUN"  # Validated on real hardware run
    CROSS_VALIDATED = "CROSS_VALIDATED"  # Cross-validated against independent measurement


class ProducerKind(str, Enum):
    """Classification of how a metric is produced (issue #704).

    This discriminator allows different verification strategies for different
    producer types, since line-range verification is much weaker for inline
    producers (thousands of candidate lines) than for discrete functions.
    """
    INLINE = "inline"  # Produced by inline code within a larger function
    ASSIGNS = "assigns"  # Produced by assignment to an attribute/field
    COMPUTES = "computes"  # Produced by explicit computation at a discrete site
    DELEGATES_TO_LIBRARY = "delegates_to_library"  # Produced externally by a library (e.g., tqdm)


@dataclass
class AggregationWindow:
    """Describes how a metric is aggregated over time/iterations."""
    kind: AggregationWindowKind
    seconds: Optional[float] = None  # For rolling windows
    samples: Optional[int] = None  # For fixed-sample windows


@dataclass
class MetricEntry:
    """Single canonical metric definition.

    This class is deliberately **torch-free and import-side-effect-free** (see module docstring).

    For consumers needing a metric's semantic version alongside its name (#807), the
    read-only property `metric_semantic_version` returns the same value as the
    `semantic_version` field. This allows downstream code to call
    `.metric_semantic_version` without changing the source-of-truth field name.
    """
    name: str
    """Canonical metric name (e.g., 'keyframe_admission_rate_hz')."""
    metric_id: str
    """Stable machine-readable identifier for this metric, preserved across renames (#807).
    Distinct from name so that metric-value history survives canonical renaming.
    """
    semantic_version: int
    """Version of this event population/formula. Bump on definition change."""
    description: str
    """What does this metric measure? (e.g., 'Rate of RGB-D frames admitted as training keyframes')."""
    unit: str
    """Physical unit (e.g., 'Hz', 'ms', 'count', 'dimensionless')."""
    event_population: str
    """Name of the underlying event (e.g., 'keyframe_admitted', 'optimization_step')."""
    numerator: str
    """Event/sample that increases (e.g., 'keyframe_admitted_count')."""
    denominator: str
    """Time/normalizer (e.g., 'wall_time_s', 'iteration')."""
    aggregation_window: AggregationWindow
    """How is this metric aggregated (rolling, fixed, lifetime)."""
    producer: str
    """Authoritative producer site as 'file:function:line' (e.g., 'train_streaming.py:record_iter:188')."""
    producer_kind: ProducerKind = ProducerKind.INLINE
    """Classification of how this metric is produced (issue #704). Distinguishes inline code,
    library delegation, assignments, and explicit computations for targeted verification.
    """
    validation_status: ValidationStatus = ValidationStatus.IMPLEMENTED_UNVALIDATED
    """Current validation status of this metric definition (issue #704)."""
    legacy_aliases: List[str] = field(default_factory=list)
    """Historical names for this metric (e.g., 'iter_ms_mean' for 'outer_loop_iteration_wall_ms_mean')."""
    validation_evidence_refs: List[str] = field(default_factory=list)
    """References to validation evidence (issue/PR/commit/test names) (issue #704)."""
    last_validation_commit: Optional[str] = None
    """Git commit hash of last validation/update (issue #704)."""
    known_limitations: List[str] = field(default_factory=list)
    """Known limitations or caveats of this metric (issue #704)."""
    consumers: List[str] = field(default_factory=list)
    """Consumer locations as 'file::function' in production surfaces."""
    valid_execution_modes: List[ExecutionMode] = field(default_factory=lambda: list(ExecutionMode))
    """Which execution modes (#168) measure this the same way."""
    clock_source: Optional[str] = None
    """Clock source for timing metrics (e.g., 'perf_counter', 'hip_events')."""
    start_bracket: Optional[str] = None
    """For timing metrics: start event name (e.g., 'start of streaming loop iteration')."""
    end_bracket: Optional[str] = None
    """For timing metrics: end event name (e.g., 'start of next iteration')."""
    synchronization_semantics: Optional[str] = None
    """For timing metrics: blocking vs async (e.g., 'synchronous at outer-loop boundary')."""
    notes: Optional[str] = None
    """Free-form audit notes (e.g., reason for drift, remediation status)."""

    @property
    def metric_semantic_version(self) -> int:
        """Alias for semantic_version, for consumers that prefer this name (#807).

        This is a read-only property; `semantic_version` is the source of truth.
        """
        return self.semantic_version


@dataclass
class AuditFinding:
    """Single discovered metric and its classification."""
    surface: str
    """Where it was found (e.g., 'train_streaming.py line 8429', 'report.json key', 'TensorBoard tag')."""
    field_or_tag: str
    """The actual name/label (e.g., 'it/V', 'optimization_step_ms_mean')."""
    producer_location: Optional[str] = None
    """Where it's produced as 'file:line' if known."""
    formula: Optional[str] = None
    """Actual formula (e.g., 'iteration // keyframes_admitted')."""
    window: Optional[str] = None
    """Aggregation window if applicable."""
    unit: Optional[str] = None
    """Measured unit."""
    consumers: List[str] = field(default_factory=list)
    """Files/functions that read this metric."""
    classification: MetricClassification = MetricClassification.CANONICAL
    """Classification per issue #704."""
    canonical_replacement: Optional[str] = None
    """If not canonical, what to replace it with."""
    action: Optional[str] = None
    """Recommended remediation (e.g., 'deprecate', 'rename', 'remove', 'investigate')."""
    compatibility_deadline: Optional[str] = None
    """When old-format consumers can assume new producer is available."""


# ============================================================================
# CANONICAL METRIC REGISTRY (issues #700-#704)
# ============================================================================
# Populated from issue #755's requirements + timing (#702) + work counters (#703)
# + documented rate metrics (#700/#701). Timing fields absorb existing
# timing_fields.py entries verbatim to preserve test compatibility.

CANONICAL_METRICS: Dict[str, MetricEntry] = {
    # ========================================================================
    # RATE METRICS (from issues #700/#701)
    # ========================================================================

    "keyframe_admission_rate_hz": MetricEntry(
        name="keyframe_admission_rate_hz",
        metric_id="keyframe_admission_rate_hz",
        semantic_version=1,
        description="Wall-clock rate at which ingested RGB-D frames are admitted as training keyframes.",
        unit="Hz",
        event_population="keyframe_admitted",
        numerator="keyframes_admitted_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="utils/execution_semantics.py:compute_progress_postfix:1010",
        producer_kind=ProducerKind.ASSIGNS,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["mapFPS"],
        notes="Issue #700 finding C: mapFPS is lifetime rate (keyframes_admitted / elapsed_since_start). See also recent-window admitted_keyframe_rate_hz in FourClocksSummary.",
    ),

    "outer_loop_iteration_rate_hz": MetricEntry(
        name="outer_loop_iteration_rate_hz",
        metric_id="outer_loop_iteration_rate_hz",
        semantic_version=1,
        description="Rate of outer training loop iterations (tqdm-tracked progress).",
        unit="Hz",
        event_population="outer_loop_iteration",
        numerator="iteration_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="train_streaming.py:streaming_training:6095",
        producer_kind=ProducerKind.DELEGATES_TO_LIBRARY,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        # Pinned at the `progress_bar = tqdm(...)` CONSTRUCTION, not at the
        # `progress_bar.update(10)` call ~4600 lines below it. Both are inside
        # `streaming_training`, but the bar is built just above the training
        # loop, whereas the update sits deep in the loop body -- which is
        # exactly the code #1075 is extracting out of this function. A pin
        # there moves on essentially every refactor (issue #1112); this one
        # only moves if code ABOVE the loop changes. The construction is also
        # where the rate tracker this metric reports actually originates.
        legacy_aliases=["it/s"],
        notes="Issue #700 finding A: tqdm 'it/s' is outer-loop rate, not optimizer-step rate. Differs from optimization_step_rate_hz.",
    ),

    "per_view_iteration_budget": MetricEntry(
        name="per_view_iteration_budget",
        metric_id="per_view_iteration_budget",
        semantic_version=1,
        description="Outer-loop iterations per current training view (per-view training budget).",
        unit="iterations/view",
        event_population="outer_loop_iteration",
        numerator="iteration_count",
        denominator="training_view_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/execution_semantics.py:compute_progress_postfix:1009",
        producer_kind=ProducerKind.ASSIGNS,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["it/V"],
        notes="Issue #701 corrected the denominator to len(train_cameras), matching the displayed V and the per-view meaning.",
    ),

    "observation_integration_commit_rate_hz": MetricEntry(
        name="observation_integration_commit_rate_hz",
        metric_id="observation_integration_commit_rate_hz",
        semantic_version=1,
        description="Wall-clock rate at which new Gaussians are inserted (observation integration with n_inserted > 0).",
        unit="Hz",
        event_population="positive_gaussian_insertion",
        numerator="positive_insertion_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="eligsir/mapping/coordinator.py:step:808",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["map_commit_rate_hz"],
        notes="Issue #700 finding G: 'map_commit_rate_hz' counts only n_inserted > 0 events, not every optimizer update or mutation.",
    ),

    "source_frame_rate_hz": MetricEntry(
        name="source_frame_rate_hz",
        metric_id="source_frame_rate_hz",
        semantic_version=1,
        description="Rate of RGB-D frames released by the source (bag/dataset/live camera).",
        unit="Hz",
        event_population="source_frame_released",
        numerator="source_frame_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="eligsir/mapping/scheduler/frame_scheduler.py:mark_released:544",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="For dataset replay, controlled by streaming_steps_per_frame. For live, hardware-dependent.",
    ),

    "optimization_step_rate_hz": MetricEntry(
        name="optimization_step_rate_hz",
        metric_id="optimization_step_rate_hz",
        semantic_version=1,
        description="Rate of optimizer.step() calls (real optimization throughput, independent of outer-loop iteration rate).",
        unit="Hz",
        event_population="optimization_step",
        numerator="optimization_step_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="utils/supervision_work.py:optimization_step_rate_hz:32",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #701. Derived from SupervisionWorkClocks via supervision_work_report_block.",
    ),

    "supervision_view_rate_hz": MetricEntry(
        name="supervision_view_rate_hz",
        metric_id="supervision_view_rate_hz",
        semantic_version=1,
        description="Rate of supervised camera views (total supervision views / wall seconds). For B=1 equals optimization_step_rate_hz; for B>1, higher.",
        unit="Hz",
        event_population="supervision_view",
        numerator="supervision_view_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="utils/supervision_work.py:supervision_view_rate_hz:49",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        consumers=["utils/comparison_report.py::write_post_training_report"],
        clock_source="perf_counter",
        notes="Realized completed supervision views divided by enclosing report wall time; B>1 uses the true camera count.",
    ),

    # ========================================================================
    # TIMING FIELDS (from issue #702, retained verbatim for test compat)
    # ========================================================================

    "outer_loop_iteration_wall_ms_mean": MetricEntry(
        name="outer_loop_iteration_wall_ms_mean",
        metric_id="outer_loop_iteration_wall_ms_mean",
        semantic_version=1,
        description="Mean wall-clock time per outer training iteration (full cycle: ingestion+render+backward+optimizer+MCMC).",
        unit="milliseconds",
        event_population="outer_loop_iteration",
        numerator="outer_loop_wall_time_ms",
        denominator="iteration_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED, samples=None),
        producer="train_streaming.py:write_summary:425",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["optimization_step_ms_mean", "iter_ms_mean"],
        clock_source="perf_counter",
        start_bracket="start of streaming loop iteration",
        end_bracket="start of next iteration",
        synchronization_semantics="synchronous; sampled at outer-loop boundary",
        notes="Issue #702: measures outer-loop bracket, NOT narrow optimizer.update() call. Legacy name 'optimization_step_ms_mean' is misleading.",
    ),

    "outer_loop_iteration_wall_ms_p95": MetricEntry(
        name="outer_loop_iteration_wall_ms_p95",
        metric_id="outer_loop_iteration_wall_ms_p95",
        semantic_version=1,
        description="95th percentile of outer-loop iteration wall time.",
        unit="milliseconds",
        event_population="outer_loop_iteration",
        numerator="outer_loop_wall_time_ms",
        denominator="iteration_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="train_streaming.py:write_summary:426",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["optimization_step_ms_p95", "iter_ms_p95"],
        clock_source="perf_counter",
        start_bracket="start of streaming loop iteration",
        end_bracket="start of next iteration",
        notes="95th percentile complement to mean. See outer_loop_iteration_wall_ms_mean.",
    ),

    "depth_insertion_wall_ms_mean": MetricEntry(
        name="depth_insertion_wall_ms_mean",
        metric_id="depth_insertion_wall_ms_mean",
        semantic_version=1,
        description="Wall time for one depth insertion frame: backprojection, voxel downsampling, dedup, masking, Gaussian append.",
        unit="milliseconds",
        event_population="depth_insertion",
        numerator="depth_insertion_wall_time_ms",
        denominator="frame_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="train_streaming.py:write_summary:424",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["insert_ms_mean"],
        clock_source="perf_counter",
        start_bracket="phase_timer.record_insert() call",
        end_bracket="phase_timer.record_insert() call (next frame)",
        notes="Measured at frame ingestion, not iteration. Canonical name for issue #702.",
    ),

    # ========================================================================
    # NOT YET IMPLEMENTED TIMING METRICS (issue #702 scope)
    # ========================================================================

    "optimization_transaction_wall_ms_mean": MetricEntry(
        name="optimization_transaction_wall_ms_mean",
        metric_id="optimization_transaction_wall_ms_mean",
        semantic_version=1,
        description="Wall time for one B-view transaction: render B views, accumulate gradients, one optimizer.step().",
        unit="milliseconds",
        event_population="optimization_transaction",
        numerator="optimization_transaction_wall_time_ms",
        denominator="transaction_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires wrapping B-view loop
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="perf_counter",
        start_bracket="first render in a B-view batch",
        end_bracket="optimizer.step() completion (after all B views)",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "optimization_transaction_gpu_ms_mean": MetricEntry(
        name="optimization_transaction_gpu_ms_mean",
        metric_id="optimization_transaction_gpu_ms_mean",
        semantic_version=1,
        description="GPU time (HIP events) for one B-view transaction.",
        unit="milliseconds",
        event_population="optimization_transaction",
        numerator="optimization_transaction_gpu_time_ms",
        denominator="transaction_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires HIP event instrumentation
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="hip_events",
        start_bracket="HIP event @ first render kernel launch",
        end_bracket="HIP event @ optimizer completion",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "render_wall_ms_mean": MetricEntry(
        name="render_wall_ms_mean",
        metric_id="render_wall_ms_mean",
        semantic_version=1,
        description="Wall time from gaussian_renderer entry to return (one render call).",
        unit="milliseconds",
        event_population="render_call",
        numerator="render_wall_time_ms",
        denominator="render_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires wrapping at gaussian_renderer callsite
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="perf_counter",
        start_bracket="gaussian_renderer entry",
        end_bracket="return from gaussian_renderer",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "render_gpu_ms_mean": MetricEntry(
        name="render_gpu_ms_mean",
        metric_id="render_gpu_ms_mean",
        semantic_version=1,
        description="GPU time (HIP events) for one render call.",
        unit="milliseconds",
        event_population="render_call",
        numerator="render_gpu_time_ms",
        denominator="render_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires HIP event instrumentation
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="hip_events",
        start_bracket="HIP event @ gsplat.rendering kernel launch",
        end_bracket="HIP event @ return from rasterization",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "backward_wall_ms_mean": MetricEntry(
        name="backward_wall_ms_mean",
        metric_id="backward_wall_ms_mean",
        semantic_version=1,
        description="Wall time from backward() entry to completion (gradients computed).",
        unit="milliseconds",
        event_population="backward_pass",
        numerator="backward_wall_time_ms",
        denominator="backward_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires wrapping at backward() callsite
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="perf_counter",
        start_bracket="backward() call entry",
        end_bracket="backward() return",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "backward_gpu_ms_mean": MetricEntry(
        name="backward_gpu_ms_mean",
        metric_id="backward_gpu_ms_mean",
        semantic_version=1,
        description="GPU time (HIP events) for one backward pass.",
        unit="milliseconds",
        event_population="backward_pass",
        numerator="backward_gpu_time_ms",
        denominator="backward_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires HIP event instrumentation
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="hip_events",
        start_bracket="HIP event @ backward kernel launch",
        end_bracket="HIP event @ backward completion",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    "optimizer_update_wall_ms_mean": MetricEntry(
        name="optimizer_update_wall_ms_mean",
        metric_id="optimizer_update_wall_ms_mean",
        semantic_version=1,
        description="Wall time from optimizer.step() entry to completion (including scheduler.step()). NARROW bracket, not outer-loop.",
        unit="milliseconds",
        event_population="optimizer_update",
        numerator="optimizer_update_wall_time_ms",
        denominator="update_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires wrapping at optimizer.step() callsite
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="perf_counter",
        start_bracket="optimizer.step() entry",
        end_bracket="optimizer.step() + scheduler.step() return",
        notes="Issue #702 & #755: NARROW bracket (narrow optimizer operation). Differs from outer_loop_iteration_wall_ms_mean which includes all outer-loop work.",
    ),

    "optimizer_update_gpu_ms_mean": MetricEntry(
        name="optimizer_update_gpu_ms_mean",
        metric_id="optimizer_update_gpu_ms_mean",
        semantic_version=1,
        description="GPU time (HIP events) for one optimizer.step().",
        unit="milliseconds",
        event_population="optimizer_update",
        numerator="optimizer_update_gpu_time_ms",
        denominator="update_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.FIXED),
        producer="PENDING",  # Requires HIP event instrumentation
        validation_status=ValidationStatus.UNIMPLEMENTED,
        clock_source="hip_events",
        start_bracket="HIP event @ optimizer kernel launch",
        end_bracket="HIP event @ optimizer completion",
        notes="Issue #702 & #755: required for #755 engine promotion A/B.",
    ),

    # ========================================================================
    # WORK COUNTERS (from issue #703)
    # ========================================================================

    "optimization_step_count": MetricEntry(
        name="optimization_step_count",
        metric_id="optimization_step_count",
        semantic_version=1,
        description="Cumulative count of optimizer.step() calls executed.",
        unit="count",
        event_population="optimization_step",
        numerator="optimization_step_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.LIFETIME),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1015",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Declared on OptimizationStepResult as `optimizer_step_count` (issue #809 renamed it from `optimization_step_count` for clarity; the metric name itself was left unchanged). Issue #703. No production consumer yet (gap #1 from issue #704 task).",
    ),

    "supervision_view_count": MetricEntry(
        name="supervision_view_count",
        metric_id="supervision_view_count",
        semantic_version=1,
        description="Count of camera views that participated in gradient supervision for this optimization step.",
        unit="count",
        event_population="supervision_view",
        numerator="supervision_view_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1003",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="For B=1, always 1. For B>1 sequential accumulation, typically B. Declared on OptimizationStepResult. Issue #703.",
    ),

    "rasterizer_call_count": MetricEntry(
        name="rasterizer_call_count",
        metric_id="rasterizer_call_count",
        semantic_version=1,
        description="Count of gsplat.rendering.rasterization calls in this optimization step.",
        unit="count",
        event_population="rasterizer_call",
        numerator="rasterizer_call_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1007",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Declared on OptimizationStepResult. LIBRARY_ONLY: producer exists, NO production consumer (gap #1 from issue #704). Issue #703.",
    ),

    "supervised_pixel_count": MetricEntry(
        name="supervised_pixel_count",
        metric_id="supervised_pixel_count",
        semantic_version=1,
        description="Total supervised pixels across all render calls in this optimization step.",
        unit="count",
        event_population="supervised_pixel",
        numerator="supervised_pixel_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1008",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Declared on OptimizationStepResult. LIBRARY_ONLY: producer exists, NO production consumer (gap #1 from issue #704). Issue #703.",
    ),

    # ========================================================================
    # EXECUTION-SHAPE COUNTERS (issue #703)
    # ========================================================================

    "loss_objective_call_count": MetricEntry(
        name="loss_objective_call_count",
        metric_id="loss_objective_call_count",
        semantic_version=1,
        description="Number of assemble_loss() invocations in this optimization step.",
        unit="count",
        event_population="loss_objective_call",
        numerator="loss_objective_call_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1010",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Distinguishes vectorized multi-view loss from per-view sequential assembly.",
    ),

    "backward_call_count": MetricEntry(
        name="backward_call_count",
        metric_id="backward_call_count",
        semantic_version=1,
        description="Number of .backward() calls made during this optimization step.",
        unit="count",
        event_population="backward_call",
        numerator="backward_call_count",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="eligsir/optimization/gaussian/engine.py:run_batched_step:1012",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Distinguishes aggregated gradients (one backward) from sequential backpropagation.",
    ),

    "loss_objective_calls_per_step": MetricEntry(
        name="loss_objective_calls_per_step",
        metric_id="loss_objective_calls_per_step",
        semantic_version=1,
        description="Average number of loss assemblies per optimization step.",
        unit="dimensionless",
        event_population="optimization_step",
        numerator="loss_objective_call_count",
        denominator="optimization_step_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:130",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Typical: 1.0 for vectorized, N for sequential loss on B=N.",
    ),

    "backward_calls_per_step": MetricEntry(
        name="backward_calls_per_step",
        metric_id="backward_calls_per_step",
        semantic_version=1,
        description="Average number of backward calls per optimization step.",
        unit="dimensionless",
        event_population="optimization_step",
        numerator="backward_call_count",
        denominator="optimization_step_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:134",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Typical: 1.0 for aggregated, N for sequential backprop on B=N.",
    ),

    "rasterizer_calls_per_step": MetricEntry(
        name="rasterizer_calls_per_step",
        metric_id="rasterizer_calls_per_step",
        semantic_version=1,
        description="Average number of rasterizer calls per optimization step.",
        unit="dimensionless",
        event_population="optimization_step",
        numerator="rasterizer_call_count",
        denominator="optimization_step_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:rasterizer_calls_per_optimization_step:67",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #701. Typical: 1.0 for sequential, C for true camera batch.",
    ),

    "realized_supervision_views_per_step": MetricEntry(
        name="realized_supervision_views_per_step",
        metric_id="realized_supervision_views_per_step",
        semantic_version=1,
        description="Average number of supervision views per realized optimization step.",
        unit="dimensionless",
        event_population="optimization_step",
        numerator="supervision_view_count",
        denominator="optimization_step_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_views_per_optimization_step:16",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        legacy_aliases=["supervision_views_per_step"],
        notes="Issue #701. For B=1 equals 1.0; for B>1 batch, equals realized batch size.",
    ),

    "renderer_camera_batch_size_mean": MetricEntry(
        name="renderer_camera_batch_size_mean",
        metric_id="renderer_camera_batch_size_mean",
        semantic_version=1,
        description="Average number of views batched per rasterizer call.",
        unit="dimensionless",
        event_population="rasterizer_call",
        numerator="renderer_camera_batch_size_sum",
        denominator="rasterizer_call_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:140",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. 1.0 for sequential, C for TRUE_CAMERA_BATCH.",
    ),

    "renderer_camera_batch_size_max": MetricEntry(
        name="renderer_camera_batch_size_max",
        metric_id="renderer_camera_batch_size_max",
        semantic_version=1,
        description="Maximum number of views batched in any single rasterizer call.",
        unit="count",
        event_population="rasterizer_call",
        numerator="renderer_camera_batch_size_max",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:140",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Identifies if any true batching occurred.",
    ),

    "loss_views_per_call_mean": MetricEntry(
        name="loss_views_per_call_mean",
        metric_id="loss_views_per_call_mean",
        semantic_version=1,
        description="Average number of views supervised per loss assembly call.",
        unit="dimensionless",
        event_population="loss_objective_call",
        numerator="loss_views_per_call_sum",
        denominator="loss_objective_call_count",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:144",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. 1.0 for per-view loss, N for vectorized loss on B=N.",
    ),

    "loss_views_per_call_max": MetricEntry(
        name="loss_views_per_call_max",
        metric_id="loss_views_per_call_max",
        semantic_version=1,
        description="Maximum number of views supervised in any single loss assembly call.",
        unit="count",
        event_population="loss_objective_call",
        numerator="loss_views_per_call_max",
        denominator="dimensionless",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.SNAPSHOT),
        producer="utils/supervision_work.py:supervision_work_report_block:144",
        producer_kind=ProducerKind.COMPUTES,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #703. Identifies if vectorized loss assembly occurred.",
    ),

    # ========================================================================
    # INGRESS FUNNEL RATE SURFACES (from issue #805)
    # ========================================================================

    "source_observed_rate_hz": MetricEntry(
        name="source_observed_rate_hz",
        metric_id="source_observed_rate_hz",
        semantic_version=1,
        description="Rate of frames observed and timestamped in the Python source callback (raw source rate).",
        unit="Hz",
        event_population="source_frame_observed",
        numerator="source_frame_observed_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="eligsir/runtime/causal_replay.py:run:1367",
        producer_kind=ProducerKind.INLINE,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #805: first canonical stage of the ingress funnel. Wired to source_offered from causal replay.",
    ),

    "materialized_valid_frame_rate_hz": MetricEntry(
        name="materialized_valid_frame_rate_hz",
        metric_id="materialized_valid_frame_rate_hz",
        semantic_version=1,
        description="Rate of frames that survived transport association and validity checks and were materialized as frame objects.",
        unit="Hz",
        event_population="materialized_valid_frame",
        numerator="materialized_valid_frame_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="eligsir/runtime/source_ingress.py:as_report_block:71",
        producer_kind=ProducerKind.INLINE,
        validation_status=ValidationStatus.IMPLEMENTED_UNVALIDATED,
        notes="Issue #805: transport-layer frame validation rate (decode, geometry extraction complete). Wired to ingress_frames_admitted.",
    ),

    "prefilter_eligible_frame_rate_hz": MetricEntry(
        name="prefilter_eligible_frame_rate_hz",
        metric_id="prefilter_eligible_frame_rate_hz",
        semantic_version=1,
        description="Rate of frames eligible for keyframe prefilter evaluation (passed prior stages).",
        unit="Hz",
        event_population="prefilter_eligible_frame",
        numerator="prefilter_eligible_frame_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="PENDING",
        producer_kind=ProducerKind.INLINE,
        validation_status=ValidationStatus.UNIMPLEMENTED,
        notes="Issue #805: frames ready for semantic keyframe admission decision (post-prefilter, pre-KeyframeAdmitter).",
    ),

    "keyframe_probe_rate_hz": MetricEntry(
        name="keyframe_probe_rate_hz",
        metric_id="keyframe_probe_rate_hz",
        semantic_version=1,
        description="Rate of frames probed as candidates by the keyframe admission policy.",
        unit="Hz",
        event_population="keyframe_candidate_probed",
        numerator="keyframe_candidate_probed_count",
        denominator="wall_time_s",
        aggregation_window=AggregationWindow(kind=AggregationWindowKind.ROLLING, seconds=5),
        producer="PENDING",
        producer_kind=ProducerKind.INLINE,
        validation_status=ValidationStatus.UNIMPLEMENTED,
        notes="Issue #805: candidates that KeyframeAdmitter evaluated (pre-admission filter).",
    ),
}


def get_canonical_metric(name: str) -> Optional[MetricEntry]:
    """Look up a canonical metric by name."""
    return CANONICAL_METRICS.get(name)


def get_metric_by_alias(alias: str) -> Optional[MetricEntry]:
    """Find the canonical metric that owns a given legacy alias."""
    for entry in CANONICAL_METRICS.values():
        if alias in entry.legacy_aliases:
            return entry
    return None


def all_canonical_metrics() -> List[MetricEntry]:
    """Return all canonical metric definitions."""
    return list(CANONICAL_METRICS.values())


def validate_registry() -> List[str]:
    """Check internal consistency of the registry. Return list of error messages."""
    errors = []
    seen_canonical_names = set()
    seen_aliases = {}

    for name, entry in CANONICAL_METRICS.items():
        if name != entry.name:
            errors.append(f"Entry key mismatch: dict key '{name}' != entry.name '{entry.name}'")

        if name in seen_canonical_names:
            errors.append(f"Duplicate canonical metric name: '{name}'")
        seen_canonical_names.add(name)

        for alias in entry.legacy_aliases:
            if alias in seen_aliases:
                errors.append(
                    f"Alias '{alias}' claimed by both '{name}' and '{seen_aliases[alias]}'"
                )
            else:
                seen_aliases[alias] = name

            # Ensure alias is not also a canonical name
            if alias in CANONICAL_METRICS:
                errors.append(
                    f"Alias '{alias}' of '{name}' is itself a canonical metric name"
                )

    return errors
