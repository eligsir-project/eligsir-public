"""Tests for the canonical metric registry (issue #704).

Validates:
1. Registry internal consistency (no duplicate names, aliases are properly owned).
2. Regression test: issue #701 corrected 'it/V' to use the training-view count.
3. Basic registry API (lookup by name, lookup by alias).
4. New fields: metric_id, validation_status, producer_kind, and validation evidence fields (issue #704).
"""

import pytest
from dataclasses import asdict
from eligsir.utils.metric_registry import (
    CANONICAL_METRICS,
    get_canonical_metric,
    get_metric_by_alias,
    validate_registry,
    MetricClassification,
    ValidationStatus,
    ProducerKind,
)


class TestRegistryConsistency:
    """Internal consistency of the registry."""

    def test_no_duplicate_canonical_names(self):
        """Every canonical metric name is unique."""
        names = list(CANONICAL_METRICS.keys())
        assert len(names) == len(set(names)), "Duplicate canonical metric names found"

    def test_no_alias_collisions(self):
        """No alias is claimed by multiple canonical metrics."""
        alias_to_owner = {}
        for name, entry in CANONICAL_METRICS.items():
            for alias in entry.legacy_aliases:
                if alias in alias_to_owner:
                    pytest.fail(
                        f"Alias '{alias}' claimed by both '{name}' and '{alias_to_owner[alias]}'"
                    )
                alias_to_owner[alias] = name

    def test_alias_not_canonical_name(self):
        """No alias is itself the name of another canonical metric."""
        canonical_names = set(CANONICAL_METRICS.keys())
        for name, entry in CANONICAL_METRICS.items():
            for alias in entry.legacy_aliases:
                if alias in canonical_names and alias != name:
                    pytest.fail(
                        f"Alias '{alias}' of '{name}' is itself a canonical metric name"
                    )

    def test_validate_registry_returns_empty_on_clean_state(self):
        """validate_registry() returns no errors for current state."""
        errors = validate_registry()
        assert errors == [], f"Registry validation failed: {errors}"


class TestLookupAPI:
    """Test lookup functions."""

    def test_get_canonical_metric_by_name(self):
        """Can look up a metric by canonical name."""
        metric = get_canonical_metric("keyframe_admission_rate_hz")
        assert metric is not None
        assert metric.name == "keyframe_admission_rate_hz"

    def test_get_canonical_metric_nonexistent(self):
        """Returns None for unknown metric."""
        metric = get_canonical_metric("nonexistent_metric")
        assert metric is None

    def test_get_metric_by_legacy_alias(self):
        """Can find canonical metric via legacy alias."""
        # mapFPS is an alias for keyframe_admission_rate_hz
        metric = get_metric_by_alias("mapFPS")
        assert metric is not None
        assert metric.name == "keyframe_admission_rate_hz"
        assert "mapFPS" in metric.legacy_aliases

    def test_get_metric_by_nonexistent_alias(self):
        """Returns None for unknown alias."""
        metric = get_metric_by_alias("nonexistent_alias")
        assert metric is None


class TestPerViewIterationBudgetCorrection:
    """Regression tests for the issue #701 correction to issue #700 finding B."""

    def test_per_view_iteration_budget_denominator_is_training_views(self):
        """The 'it/V' metric divides by the current training-view count."""
        metric = get_canonical_metric("per_view_iteration_budget")
        assert metric is not None
        assert metric.denominator == "training_view_count"
        assert metric.unit == "iterations/view"

    def test_per_view_iteration_budget_documents_corrected_label_semantics(self):
        """The registry documents that issue #701 aligned the formula and label."""
        metric = get_canonical_metric("per_view_iteration_budget")
        assert metric is not None
        assert "it/V" in metric.legacy_aliases
        assert "#701" in metric.notes
        assert "matching" in metric.notes.lower()


class TestDriftFinding2_OuterLoopIterationTiming:
    """Regression test for issue #700 finding F.

    The metric named 'optimization_step_ms_mean' measures the OUTER loop iteration
    (ingestion+render+backward+optimizer+MCMC), not just the narrow
    optimizer.update() call.
    """

    def test_optimization_step_ms_mean_measures_outer_loop(self):
        """The 'optimization_step_ms_mean' metric measures the full outer loop, not narrow optimizer step."""
        metric = get_canonical_metric("outer_loop_iteration_wall_ms_mean")
        assert metric is not None
        # Confirm it's an alias of optimization_step_ms_mean
        assert "optimization_step_ms_mean" in metric.legacy_aliases
        # Confirm the notes clarify this measures outer-loop, not narrow optimizer
        assert "outer" in metric.notes.lower()
        assert "optimizer" in metric.notes.lower()

    def test_optimization_step_ms_mean_bracket_is_outer_loop_boundary(self):
        """The timing bracket describes the full outer loop iteration."""
        metric = get_canonical_metric("outer_loop_iteration_wall_ms_mean")
        assert metric is not None
        # Confirm bracket spans full iteration
        assert "iteration" in metric.start_bracket.lower() or "loop" in metric.start_bracket.lower()
        assert "iteration" in metric.end_bracket.lower() or "next" in metric.end_bracket.lower()


class TestDriftFinding3_MapCommitRateHz:
    """Regression test for issue #700 finding G.

    'map_commit_rate_hz' counts only insertion events with n_inserted > 0,
    not every map change (pruning, relocation, loop-closure).
    """

    def test_map_commit_rate_hz_counts_insertions_only(self):
        """'map_commit_rate_hz' counts only n_inserted > 0, not all map topology changes."""
        # Legacy alias: map_commit_rate_hz should resolve to observation_integration_commit_rate_hz
        metric = get_metric_by_alias("map_commit_rate_hz")
        assert metric is not None
        assert metric.name == "observation_integration_commit_rate_hz"
        assert "insert" in metric.description.lower()  # matches both "inserted" and "insertion"
        assert "n_inserted > 0" in metric.notes or "n_inserted" in metric.notes


class TestDriftFinding4_KeyframeAdmissionVsOuterLoop:
    """Regression test for issue #700 finding A and C.

    There are three separate rates:
    - outer_loop_iteration_rate_hz: tqdm 'it/s' (outer-loop progress)
    - keyframe_admission_rate_hz: 'mapFPS' (frames admitted as keyframes)
    - outer_loop_iteration_wall_ms_mean: wall-clock per outer loop

    They may numerically coincide in simple modes but are defined separately.
    """

    def test_keyframe_admission_rate_is_separate_from_outer_loop_rate(self):
        """keyframe_admission_rate_hz and outer_loop_iteration_rate_hz are distinct metrics."""
        kf_rate = get_canonical_metric("keyframe_admission_rate_hz")
        outer_rate = get_canonical_metric("outer_loop_iteration_rate_hz")

        assert kf_rate is not None
        assert outer_rate is not None

        # Different event populations
        assert kf_rate.event_population != outer_rate.event_population
        assert kf_rate.event_population == "keyframe_admitted"
        assert outer_rate.event_population == "outer_loop_iteration"

    def test_mapfps_is_keyframe_admission_rate(self):
        """Legacy name 'mapFPS' is an alias for keyframe_admission_rate_hz, not outer-loop rate."""
        metric = get_metric_by_alias("mapFPS")
        assert metric is not None
        assert metric.name == "keyframe_admission_rate_hz"
        # mapFPS is from the progress bar, which also shows it as distinct from it/s


class TestIssue704ValidationFields:
    """Tests for new validation-status fields added in issue #704."""

    def test_all_entries_have_metric_id(self):
        """Every metric entry must have a metric_id field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "metric_id"), f"{name} missing metric_id"
            assert isinstance(entry.metric_id, str), f"{name} metric_id not a string"
            assert len(entry.metric_id) > 0, f"{name} metric_id is empty"

    def test_metric_ids_are_unique(self):
        """Every metric_id must be unique."""
        metric_ids = [entry.metric_id for entry in CANONICAL_METRICS.values()]
        assert len(metric_ids) == len(set(metric_ids)), "Duplicate metric_ids found"

    def test_metric_ids_stable_snapshot(self):
        """Verify metric_ids are stable (frozen snapshot for regression guard)."""
        expected_ids = {
            "keyframe_admission_rate_hz",
            "outer_loop_iteration_rate_hz",
            "per_view_iteration_budget",
            "observation_integration_commit_rate_hz",
            "source_frame_rate_hz",
            "optimization_step_rate_hz",
            "supervision_view_rate_hz",
            "outer_loop_iteration_wall_ms_mean",
            "outer_loop_iteration_wall_ms_p95",
            "depth_insertion_wall_ms_mean",
            "optimization_transaction_wall_ms_mean",
            "optimization_transaction_gpu_ms_mean",
            "render_wall_ms_mean",
            "render_gpu_ms_mean",
            "backward_wall_ms_mean",
            "backward_gpu_ms_mean",
            "optimizer_update_wall_ms_mean",
            "optimizer_update_gpu_ms_mean",
            "optimization_step_count",
            "supervision_view_count",
            "rasterizer_call_count",
            "supervised_pixel_count",
            "loss_objective_call_count",
            "backward_call_count",
            "loss_objective_calls_per_step",
            "backward_calls_per_step",
            "rasterizer_calls_per_step",
            "realized_supervision_views_per_step",
            "renderer_camera_batch_size_mean",
            "renderer_camera_batch_size_max",
            "loss_views_per_call_mean",
            "loss_views_per_call_max",
            # Issue #805: ingress funnel rate surfaces (four uninstrumented)
            "source_observed_rate_hz",
            "materialized_valid_frame_rate_hz",
            "prefilter_eligible_frame_rate_hz",
            "keyframe_probe_rate_hz",
        }
        actual_ids = {entry.metric_id for entry in CANONICAL_METRICS.values()}
        assert actual_ids == expected_ids, f"Metric IDs mismatch: {actual_ids ^ expected_ids}"

    def test_all_entries_have_validation_status(self):
        """Every metric entry must have a validation_status field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "validation_status"), f"{name} missing validation_status"
            assert isinstance(entry.validation_status, ValidationStatus), \
                f"{name} validation_status not a ValidationStatus enum"

    def test_all_entries_have_producer_kind(self):
        """Every metric entry must have a producer_kind field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "producer_kind"), f"{name} missing producer_kind"
            assert isinstance(entry.producer_kind, ProducerKind), \
                f"{name} producer_kind not a ProducerKind enum"

    def test_all_entries_have_validation_evidence_refs(self):
        """Every metric entry must have a validation_evidence_refs field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "validation_evidence_refs"), f"{name} missing validation_evidence_refs"
            assert isinstance(entry.validation_evidence_refs, list), \
                f"{name} validation_evidence_refs not a list"

    def test_all_entries_have_last_validation_commit(self):
        """Every metric entry must have a last_validation_commit field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "last_validation_commit"), f"{name} missing last_validation_commit"
            # Should be None or a string
            assert entry.last_validation_commit is None or isinstance(entry.last_validation_commit, str), \
                f"{name} last_validation_commit not None or string"

    def test_all_entries_have_known_limitations(self):
        """Every metric entry must have a known_limitations field."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "known_limitations"), f"{name} missing known_limitations"
            assert isinstance(entry.known_limitations, list), \
                f"{name} known_limitations not a list"

    def test_serialization_with_new_fields(self):
        """Every entry must serialize correctly with new fields."""
        for name, entry in CANONICAL_METRICS.items():
            try:
                # Try to convert to dict (test asdict from dataclasses)
                entry_dict = asdict(entry)
                # Verify new fields are in the dict
                assert "metric_id" in entry_dict, f"{name} missing metric_id in asdict"
                assert "validation_status" in entry_dict, f"{name} missing validation_status in asdict"
                assert "producer_kind" in entry_dict, f"{name} missing producer_kind in asdict"
                assert "validation_evidence_refs" in entry_dict, f"{name} missing validation_evidence_refs in asdict"
                assert "last_validation_commit" in entry_dict, f"{name} missing last_validation_commit in asdict"
                assert "known_limitations" in entry_dict, f"{name} missing known_limitations in asdict"
            except Exception as e:
                raise AssertionError(f"{name} failed to serialize: {e}")

    def test_execution_shape_metrics_marked_unvalidated(self):
        """Execution-shape metrics from PR #1759 should be marked IMPLEMENTED_UNVALIDATED."""
        execution_shape_metrics = {
            "loss_objective_call_count",
            "backward_call_count",
            "loss_objective_calls_per_step",
            "backward_calls_per_step",
            "renderer_camera_batch_size_mean",
            "renderer_camera_batch_size_max",
            "loss_views_per_call_mean",
            "loss_views_per_call_max",
        }
        for metric_name in execution_shape_metrics:
            entry = CANONICAL_METRICS[metric_name]
            assert entry.validation_status == ValidationStatus.IMPLEMENTED_UNVALIDATED, \
                f"{metric_name} should be IMPLEMENTED_UNVALIDATED, got {entry.validation_status}"

    def test_unimplemented_metrics_marked_unimplemented(self):
        """Metrics with producer='PENDING' should be marked UNIMPLEMENTED."""
        unimplemented_metrics = {
            "optimization_transaction_wall_ms_mean",
            "optimization_transaction_gpu_ms_mean",
            "render_wall_ms_mean",
            "render_gpu_ms_mean",
            "backward_wall_ms_mean",
            "backward_gpu_ms_mean",
            "optimizer_update_wall_ms_mean",
            "optimizer_update_gpu_ms_mean",
        }
        for metric_name in unimplemented_metrics:
            entry = CANONICAL_METRICS[metric_name]
            assert entry.producer == "PENDING", f"{metric_name} producer should be PENDING"
            assert entry.validation_status == ValidationStatus.UNIMPLEMENTED, \
                f"{metric_name} should be UNIMPLEMENTED, got {entry.validation_status}"

    def test_delegates_to_library_kind(self):
        """outer_loop_iteration_rate_hz should be marked as DELEGATES_TO_LIBRARY."""
        entry = CANONICAL_METRICS["outer_loop_iteration_rate_hz"]
        assert entry.producer_kind == ProducerKind.DELEGATES_TO_LIBRARY, \
            f"outer_loop_iteration_rate_hz should be DELEGATES_TO_LIBRARY, got {entry.producer_kind}"

    def test_metric_semantic_version_property(self):
        """Verify metric_semantic_version property is available (issue #807)."""
        for name, entry in CANONICAL_METRICS.items():
            assert hasattr(entry, "metric_semantic_version"), f"{name} missing metric_semantic_version property"
            assert entry.metric_semantic_version == entry.semantic_version, \
                f"{name} metric_semantic_version does not match semantic_version"
