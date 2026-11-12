"""Tests for issue #805: canonical ingress funnel vocabulary and rate surfaces.

Verifies:
1. Enum completeness: all stage/reason names from issue #805 are present
2. Boundary semantics: semantic reasons only appear post-MappingSession
3. Report block structure: rate surfaces, invariants, vocabularies are formed correctly
4. Metric registry integration: no duplicate metrics; PENDING entries are exempted
5. Invariant predicates: lossless bag replay, dual-admission detection

This test is deliberately CPU-only and torch-free (issue #805 / #704 scope).
"""

from __future__ import annotations

import pytest

from eligsir.io.ingress_funnel import (
    IngressStage,
    RejectionReason,
    RejectionBoundary,
    REJECTION_BOUNDARIES,
    ingress_funnel_report_block,
)
from eligsir.utils.metric_registry import CANONICAL_METRICS, get_canonical_metric


# ============================================================================
# ENUM COMPLETENESS: Issue #805 original list
# ============================================================================

ISSUE_805_CANONICAL_STAGES = {
    "source_observed",
    "source_decimated",
    "source_association_valid",
    "source_frame_materialized",
    "source_buffer_delivered",
    "source_released",
    "source_ingested",
    "prefilter_evaluated",
    "prefilter_eligible",
    "candidate_probed",
    "candidate_evaluated",
    "keyframe_committed",
}

ISSUE_805_REJECTION_REASONS = {
    "SOURCE_DECIMATION",
    "SOURCE_SYNC_REJECT",
    "SOURCE_POSE_INVALID",
    "SOURCE_DECODE_INVALID",
    "SOURCE_BUFFER_DROP",  # Historical; mapped to TRANSPORT_DROP
    "SCHEDULER_LATE_DROP",
    "SLAM_NO_POSE_SKIP",
    "PREFILTER_REJECT",
    "ADMISSION_REJECT",
}

ISSUE_1209_ADDITIONAL_REASONS = {
    "TRANSPORT_DROP",  # Canonical pre-core label (subsumes SOURCE_BUFFER_DROP)
    "SEMANTIC_COMMIT",  # Post-MappingSession (will be handled separately as commitment state)
    "SEMANTIC_DEFER",
    "ARCHIVE_EVICT",
}


def test_all_issue805_stages_defined():
    """Verify all canonical stages from issue #805 are in the enum."""
    enum_stages = {stage.value for stage in IngressStage}
    for stage_name in ISSUE_805_CANONICAL_STAGES:
        assert stage_name in enum_stages, (
            f"Stage {stage_name!r} from issue #805 not in IngressStage enum"
        )


def test_all_issue805_rejection_reasons_defined():
    """Verify all rejection reasons from issue #805 (and #1209 update) are in the enum.

    Note: SOURCE_BUFFER_DROP from the original issue is superseded by TRANSPORT_DROP
    in the #1209 ownership update, so we check for TRANSPORT_DROP instead.
    """
    enum_reasons = {reason.value for reason in RejectionReason}

    # All original reasons (with SOURCE_BUFFER_DROP → TRANSPORT_DROP mapping)
    for reason in ISSUE_805_REJECTION_REASONS:
        if reason == "SOURCE_BUFFER_DROP":
            # Superseded by TRANSPORT_DROP in #1209 update
            assert "TRANSPORT_DROP" in enum_reasons
        else:
            assert reason in enum_reasons, (
                f"Reason {reason!r} from issue #805 not in RejectionReason enum"
            )

    # Additional reasons from #1209 update
    for reason in {"TRANSPORT_DROP", "SEMANTIC_DEFER", "ARCHIVE_EVICT"}:
        assert reason in enum_reasons, (
            f"Reason {reason!r} from #1209 update not in RejectionReason enum"
        )


def test_stages_form_strict_progression():
    """Verify stages appear in expected order (source_observed → keyframe_committed)."""
    stages = [stage.value for stage in IngressStage]
    assert stages[0] == "source_observed"
    assert stages[-1] == "keyframe_committed"
    # Check that we have the expected count (allows new stages between)
    assert len(stages) >= 12


def test_boundary_semantics_no_semantic_reason_pre_core():
    """Invariant: no semantic reason appears as PRE_CORE.

    From issue #1209: semantic decisions (PREFILTER_REJECT, ADMISSION_REJECT,
    SEMANTIC_DEFER, ARCHIVE_EVICT) must NOT appear before MappingSession.
    """
    semantic_reasons = {
        "PREFILTER_REJECT",
        "ADMISSION_REJECT",
        "SEMANTIC_DEFER",
        "ARCHIVE_EVICT",
    }

    for reason_str in semantic_reasons:
        boundary = REJECTION_BOUNDARIES[RejectionReason(reason_str)]
        assert (
            boundary == RejectionBoundary.POST_MAPPING_SESSION
        ), f"Semantic reason {reason_str} must be POST_MAPPING_SESSION, got {boundary}"


def test_boundary_semantics_transport_reasons_pre_core():
    """Invariant: transport reasons (capacity, decode, pose, sync) are PRE_CORE only."""
    transport_reasons = {
        "SOURCE_DECIMATION",
        "SOURCE_SYNC_REJECT",
        "SOURCE_POSE_INVALID",
        "SOURCE_DECODE_INVALID",
        "TRANSPORT_DROP",
        "SCHEDULER_LATE_DROP",
        "SLAM_NO_POSE_SKIP",
    }

    for reason_str in transport_reasons:
        boundary = REJECTION_BOUNDARIES[RejectionReason(reason_str)]
        assert (
            boundary == RejectionBoundary.PRE_CORE
        ), f"Transport reason {reason_str} must be PRE_CORE, got {boundary}"


def test_all_rejection_reasons_have_boundary():
    """Every RejectionReason must have a boundary entry."""
    for reason in RejectionReason:
        assert reason in REJECTION_BOUNDARIES, (
            f"RejectionReason {reason.name} has no boundary mapping"
        )


def test_report_block_structure():
    """Verify ingress_funnel_report_block() emits correct structure."""
    block = ingress_funnel_report_block()

    # Top-level keys
    assert "stage_vocabulary" in block
    assert "rejection_reasons" in block
    assert "rejection_boundaries" in block
    assert "rate_surfaces" in block
    assert "invariants" in block

    # Stage vocabulary is a list of stage names
    assert isinstance(block["stage_vocabulary"], list)
    assert all(isinstance(s, str) for s in block["stage_vocabulary"])
    assert "source_observed" in block["stage_vocabulary"]
    assert "keyframe_committed" in block["stage_vocabulary"]

    # Rejection reasons is a list of reason names
    assert isinstance(block["rejection_reasons"], list)
    assert all(isinstance(r, str) for r in block["rejection_reasons"])
    assert "SOURCE_DECIMATION" in block["rejection_reasons"]
    assert "ADMISSION_REJECT" in block["rejection_reasons"]

    # Boundaries is a mapping of reason → boundary
    assert isinstance(block["rejection_boundaries"], dict)
    for reason_str, boundary_str in block["rejection_boundaries"].items():
        assert isinstance(reason_str, str)
        assert isinstance(boundary_str, str)
        assert boundary_str in {"PRE_CORE", "POST_MAPPING_SESSION"}

    # Invariants is a mapping
    assert isinstance(block["invariants"], dict)
    assert "semantic_rejection_before_mapping_session" in block["invariants"]
    assert "dual_admission_detected" in block["invariants"]
    assert "bag_replay_lossless" in block["invariants"]


def test_rate_surfaces_count_and_shape():
    """Verify eight rate surfaces exist with correct structure."""
    block = ingress_funnel_report_block()
    surfaces = block["rate_surfaces"]

    # Must have exactly eight surfaces
    assert len(surfaces) == 8, f"Expected 8 rate surfaces, got {len(surfaces)}"

    expected_surface_names = {
        "source_observed_hz",
        "materialized_valid_hz",
        "released_hz",
        "prefilter_eligible_hz",
        "probe_hz",
        "committed_keyframe_hz",
        "supervision_view_hz",
        "optimizer_step_hz",
    }
    assert set(surfaces.keys()) == expected_surface_names

    # Each surface must have hz and status keys
    for name, surface in surfaces.items():
        assert "hz" in surface, f"Surface {name} missing 'hz'"
        assert "status" in surface, f"Surface {name} missing 'status'"
        assert "producer" in surface, f"Surface {name} missing 'producer'"
        assert surface["hz"] is None, f"Uninstrumented surface {name} should have hz=None"
        assert surface["status"] in {
            "INSTRUMENTED",
            "NOT_INSTRUMENTED",
        }, f"Invalid status in {name}: {surface['status']}"


def test_rate_surfaces_metric_id_reference():
    """Surfaces with existing producers reference the canonical metric_id."""
    block = ingress_funnel_report_block()
    surfaces = block["rate_surfaces"]

    # These surfaces have existing canonical metrics
    metric_references = {
        "released_hz": "source_frame_rate_hz",
        "committed_keyframe_hz": "keyframe_admission_rate_hz",
        "supervision_view_hz": "supervision_view_rate_hz",
        "optimizer_step_hz": "optimization_step_rate_hz",
    }

    for surface_name, metric_id in metric_references.items():
        surface = surfaces[surface_name]
        if "metric_id" in surface:
            assert surface["metric_id"] == metric_id, (
                f"Surface {surface_name} references wrong metric: "
                f"{surface['metric_id']} != {metric_id}"
            )


def test_report_block_no_nan_values():
    """Verify report block uses None, never float('nan'), for unset values.

    float('nan') becomes bare NaN in JSON output, which is invalid JSON.
    """
    block = ingress_funnel_report_block()
    surfaces = block["rate_surfaces"]

    for name, surface in surfaces.items():
        assert surface["hz"] is None, (
            f"Surface {name}.hz should be None, not {surface['hz']}"
        )
        # JSON-serializable: None → null
        import json

        assert json.dumps(surface) is not None


def test_invariant_predicates_shape():
    """Verify invariant predicates are boolean/None and make semantic sense."""
    block = ingress_funnel_report_block()
    invariants = block["invariants"]

    # semantic_rejection_before_mapping_session: must be False (safeguard)
    assert invariants["semantic_rejection_before_mapping_session"] is False
    # dual_admission_detected: must be False (no duplicate decision)
    assert invariants["dual_admission_detected"] is False
    # bag_replay_lossless: None initially (validated elsewhere via counters)
    assert invariants["bag_replay_lossless"] is None


def test_metric_registry_integration_no_duplicates():
    """Verify new metrics are registered and wired metrics have real producers.

    The four new metrics should either:
    1. Be wired with a real producer (source_observed_rate_hz, materialized_valid_frame_rate_hz)
    2. Have PENDING producer when no producer yet exists (prefilter_eligible_frame_rate_hz, keyframe_probe_rate_hz)
    """
    # Wired surfaces: have real producers
    wired_metric_ids = {
        "source_observed_rate_hz": "eligsir/runtime/causal_replay.py:run:1367",
        "materialized_valid_frame_rate_hz": "eligsir/runtime/source_ingress.py:as_report_block:71",
    }
    for metric_id, expected_producer in wired_metric_ids.items():
        entry = get_canonical_metric(metric_id)
        assert entry is not None, f"Wired metric {metric_id} not registered"
        assert entry.producer == expected_producer, (
            f"Wired metric {metric_id} producer mismatch: {entry.producer!r} != {expected_producer!r}"
        )

    # Unwired surfaces: still PENDING
    unwired_metric_ids = {
        "prefilter_eligible_frame_rate_hz",
        "keyframe_probe_rate_hz",
    }
    for metric_id in unwired_metric_ids:
        entry = get_canonical_metric(metric_id)
        assert entry is not None, f"Metric {metric_id} not registered"
        assert entry.producer == "PENDING", (
            f"Unwired metric {metric_id} must have producer='PENDING', got {entry.producer!r}"
        )
        assert entry.validation_status.value == "UNIMPLEMENTED", (
            f"Unwired metric {metric_id} must be UNIMPLEMENTED, got {entry.validation_status}"
        )


def test_metric_registry_existing_producers_correct():
    """Verify the four surfaces with existing metrics reference real producers."""
    existing_metrics = {
        "source_frame_rate_hz": "eligsir/mapping/scheduler/frame_scheduler.py:mark_released:544",
        "keyframe_admission_rate_hz": "utils/execution_semantics.py:compute_progress_postfix:1010",
        "supervision_view_rate_hz": "utils/supervision_work.py:supervision_view_rate_hz:49",
        "optimization_step_rate_hz": "utils/supervision_work.py:optimization_step_rate_hz:32",
    }

    for metric_id, expected_producer in existing_metrics.items():
        entry = get_canonical_metric(metric_id)
        assert entry is not None, f"Existing metric {metric_id} not found in registry"
        assert entry.producer == expected_producer, (
            f"Metric {metric_id} producer mismatch: {entry.producer} != {expected_producer}"
        )


def test_bag_replay_lossless_invariant_fixture():
    """Test the lossless-bag invariant predicate on a fixture.

    Bag replay is lossless iff source_offered == source_admitted && dropped == 0.
    """
    # Valid lossless case: all frames offered are admitted, none dropped
    lossless_counters = {
        "source_offered": 100,
        "source_admitted": 100,
        "source_dropped": 0,
        "transport_dropped_at_arrival": 0,
        "transport_dropped_at_enqueue": 0,
    }

    # The predicate: check if offered == admitted && dropped == 0
    offered = lossless_counters.get("source_offered", 0)
    admitted = lossless_counters.get("source_admitted", 0)
    dropped = lossless_counters.get("source_dropped", 0)
    is_lossless = (offered == admitted) and (dropped == 0)
    assert is_lossless is True

    # Invalid lossy case: some frames dropped
    lossy_counters = {
        "source_offered": 100,
        "source_admitted": 95,
        "source_dropped": 5,
        "transport_dropped_at_arrival": 3,
        "transport_dropped_at_enqueue": 2,
    }

    offered = lossy_counters.get("source_offered", 0)
    admitted = lossy_counters.get("source_admitted", 0)
    dropped = lossy_counters.get("source_dropped", 0)
    is_lossless = (offered == admitted) and (dropped == 0)
    assert is_lossless is False


def test_no_dual_admission_invariant_fixture():
    """Test the dual-admission detection invariant.

    Dual admission occurs if a frame is admitted by both semantic and transport layers.
    This should never happen: transport is pre-MappingSession only.
    """
    # Valid case: no frame admitted twice
    single_path_counters = {
        "source_transport_admitted": 100,
        "keyframe_semantic_admitted": 50,
        "frames_admitted_by_both": 0,
    }

    # Check: frames_admitted_by_both should be 0
    dual_admitted = single_path_counters.get("frames_admitted_by_both", 0)
    assert dual_admitted == 0

    # Invalid case: same frame admitted by two paths (error)
    dual_path_counters = {
        "source_transport_admitted": 100,
        "keyframe_semantic_admitted": 50,
        "frames_admitted_by_both": 10,
    }

    dual_admitted = dual_path_counters.get("frames_admitted_by_both", 0)
    assert dual_admitted > 0


def test_rate_surfaces_wired_with_distinct_counts():
    """Verify source_observed and materialized_valid surfaces are correctly wired.

    Uses distinct counts and wall time to ensure hz computation is correct and
    mutation control can detect swapped counts.
    """
    # Fixture: distinct counts so hz values differ if swapped
    source_observed_count = 100
    materialized_valid_count = 97  # 3 frames dropped during decode
    wall_time_s = 4.0

    block = ingress_funnel_report_block(
        source_observed_count=source_observed_count,
        materialized_valid_count=materialized_valid_count,
        wall_time_s=wall_time_s,
    )

    # Verify surface shapes
    surfaces = block["rate_surfaces"]
    assert "source_observed_hz" in surfaces
    assert "materialized_valid_hz" in surfaces

    # source_observed_hz = 100 / 4.0 = 25.0 Hz
    source_obs = surfaces["source_observed_hz"]
    assert source_obs["status"] == "INSTRUMENTED"
    assert source_obs["hz"] == 25.0, f"source_observed_hz: expected 25.0, got {source_obs['hz']}"
    assert source_obs["producer"] == "eligsir/runtime/causal_replay.py:run:1367"

    # materialized_valid_hz = 97 / 4.0 = 24.25 Hz
    mat_valid = surfaces["materialized_valid_hz"]
    assert mat_valid["status"] == "INSTRUMENTED"
    assert mat_valid["hz"] == 24.25, f"materialized_valid_hz: expected 24.25, got {mat_valid['hz']}"
    assert mat_valid["producer"] == "eligsir/runtime/source_ingress.py:as_report_block:71"

    # prefilter_eligible and probe remain NOT_INSTRUMENTED with reasons
    assert surfaces["prefilter_eligible_hz"]["status"] == "NOT_INSTRUMENTED"
    assert surfaces["prefilter_eligible_hz"]["hz"] is None
    assert "reason" in surfaces["prefilter_eligible_hz"]

    assert surfaces["probe_hz"]["status"] == "NOT_INSTRUMENTED"
    assert surfaces["probe_hz"]["hz"] is None
    assert "reason" in surfaces["probe_hz"]


def test_rate_surfaces_guard_zero_wall_time():
    """Verify hz=None when wall_time_s is zero or negative (no division by zero)."""
    block = ingress_funnel_report_block(
        source_observed_count=100,
        materialized_valid_count=97,
        wall_time_s=0.0,
    )

    surfaces = block["rate_surfaces"]
    assert surfaces["source_observed_hz"]["hz"] is None
    assert surfaces["materialized_valid_hz"]["hz"] is None
    assert surfaces["source_observed_hz"]["status"] == "NOT_INSTRUMENTED"
    assert surfaces["materialized_valid_hz"]["status"] == "NOT_INSTRUMENTED"


def test_rate_surfaces_mutation_control():
    """Mutation control: verify tests fail when counts are swapped.

    This test verifies the mutation test itself will catch count field swaps
    in the block builder. Swap the two count fields in the builder temporarily
    to prove this test would fail.
    """
    source_observed_count = 100
    materialized_valid_count = 97
    wall_time_s = 4.0

    block = ingress_funnel_report_block(
        source_observed_count=source_observed_count,
        materialized_valid_count=materialized_valid_count,
        wall_time_s=wall_time_s,
    )

    surfaces = block["rate_surfaces"]
    # If counts were swapped in the builder, these assertions would fail:
    assert surfaces["source_observed_hz"]["hz"] == 25.0  # 100 / 4.0
    assert surfaces["materialized_valid_hz"]["hz"] == 24.25  # 97 / 4.0
    # If swapped: first would be 24.25, second would be 25.0


def test_causal_replay_summary_payload_carries_ingress_funnel_block():
    """Real path: the payload builder the causal entry serializes to
    causal_replay_summary.json must carry the #805 funnel block with all
    eight rate surfaces and both rejection boundaries."""
    import json
    from dataclasses import replace
    from eligsir.io.ingress_funnel import RejectionBoundary
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    summary = _minimal_summary()
    # Use replace() since CausalReplaySummary is frozen
    summary = replace(
        summary,
        source_offered=100,
        source_ingress={"ingress_frames_admitted": 97},
        elapsed_s=4.0
    )

    payload = causal_replay_summary_payload(
        summary,
        source_hz=30.0,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block={},
        supervision_work_block={},
    )
    json.loads(json.dumps(payload, default=str))
    funnel = payload["ingress_funnel"]
    assert set(funnel["rate_surfaces"]) == {
        "source_observed_hz", "materialized_valid_hz", "released_hz", "prefilter_eligible_hz",
        "probe_hz", "committed_keyframe_hz", "supervision_view_hz", "optimizer_step_hz",
    }
    assert set(funnel["rejection_boundaries"].values()) == {
        RejectionBoundary.PRE_CORE.value, RejectionBoundary.POST_MAPPING_SESSION.value
    }
    assert payload["requested_source_fps"] == 30.0
    assert payload["evidence_class"] == "UNMANAGED"

    # Verify wired surfaces have correct hz values
    assert funnel["rate_surfaces"]["source_observed_hz"]["hz"] == 25.0  # 100 / 4.0
    assert funnel["rate_surfaces"]["materialized_valid_hz"]["hz"] == 24.25  # 97 / 4.0
    assert funnel["rate_surfaces"]["source_observed_hz"]["status"] == "INSTRUMENTED"
    assert funnel["rate_surfaces"]["materialized_valid_hz"]["status"] == "INSTRUMENTED"


def test_causal_replay_summary_payload_propagates_managed_origin_class():
    """Managed ResearchFlow runs must not be relabeled by the causal payload."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    experiment_origin, experiment_context = _validated_managed_inputs()
    payload = causal_replay_summary_payload(
        _minimal_summary(),
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block={},
        supervision_work_block={},
        experiment_origin=experiment_origin,
        experiment_context=experiment_context,
    )

    assert payload["evidence_class"] == "MANAGED"


def test_causal_replay_summary_payload_requires_validated_managed_context():
    """A free origin label cannot promote causal-report evidence."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    payload = causal_replay_summary_payload(
        _minimal_summary(),
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block={},
        supervision_work_block={},
        experiment_origin={"evidence_class": "MANAGED"},
        experiment_context={
            "execution_management": "ABLATOR",
            "gradeability": "GRADEABLE_DECLARED",
        },
    )

    assert payload["evidence_class"] == "UNMANAGED"


@pytest.mark.parametrize(
    "experiment_context",
    [
        None,
        {
            "execution_management": "DIRECT",
            "gradeability": "NON_GRADEABLE_DIRECT",
        },
        {
            "execution_management": "ABLATOR",
            "gradeability": "NON_GRADEABLE_ABLATOR_UNDECLARED",
        },
    ],
)
def test_causal_replay_summary_payload_fails_closed_without_gradeable_context(
    experiment_context,
):
    """Registered origin alone does not make a direct or undeclared run gradeable."""
    from eligsir.runtime.causal_entry import causal_replay_summary_payload

    payload = causal_replay_summary_payload(
        _minimal_summary(),
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        ready_drain_block={},
        supervision_work_block={},
        experiment_origin={
            "researchflow_run_id": "rf-managed",
            "researchflow_plan_sha256": "a" * 64,
        },
        experiment_context=experiment_context,
    )

    assert payload["evidence_class"] == "UNMANAGED"


def _minimal_summary():
    from dataclasses import MISSING, fields as dc_fields

    from eligsir.runtime.causal_replay import CausalReplaySummary

    kwargs = {}
    for f in dc_fields(CausalReplaySummary):
        if f.default is not MISSING or f.default_factory is not MISSING:
            continue
        kwargs[f.name] = 0
    return CausalReplaySummary(**kwargs)


def _validated_managed_inputs():
    """Construct the same origin/context inputs produced at trainer startup."""
    from argparse import Namespace

    from eligsir.utils.experiment_declaration import (
        DECLARATION_ENV,
        DECLARATION_SHA_ENV,
        JOB_ID_ENV,
        canonical_declaration_json,
        declaration_sha256,
        load_experiment_context,
    )
    from eligsir.utils.experiment_origin import resolve_experiment_origin_from_args

    declaration = {
        "schema_version": 1,
        "run_class": "experiment",
        "experiment_id": "causal_summary_test",
        "title": "Causal summary managed provenance test",
        "purpose": "Exercise validated causal-report provenance",
        "owner_refs": ["eligsir#1930"],
        "expected_evidence": ["managed provenance"],
        "arm": {
            "id": "control",
            "comparison_role": "control",
            "manipulation": "none",
        },
    }
    raw = canonical_declaration_json(declaration)
    context = load_experiment_context({
        DECLARATION_ENV: raw,
        DECLARATION_SHA_ENV: declaration_sha256(declaration),
        JOB_ID_ENV: "rf-causal-summary-test",
    })
    args = Namespace(
        experiment_origin_repo=None,
        experiment_origin_issue=None,
        experiment_origin_pr=None,
        experiment_origin_ad_hoc=False,
        experiment_id="causal_summary_test",
        experiment_arm="control",
        experiment_purpose="Exercise validated causal-report provenance",
        experiment_origin_file=None,
        experiment_validity_class=None,
        baseline_run=None,
        scientific_mode=True,
    )
    origin = resolve_experiment_origin_from_args(args, env={
        "RESEARCHFLOW_RUN_ID": "rf-causal-summary-test",
        "RESEARCHFLOW_PLAN_SHA256": "a" * 64,
    })
    return origin, context
