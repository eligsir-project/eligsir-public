"""Residual recording contracts for issue #795.

These tests protect telemetry only: none of the assertions authorise a change
to admission thresholds, controller state, or default behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from eligsir.utils import admission_fingerprint
from eligsir.utils.admission_fingerprint import AdmissionFingerprint


ROOT = Path(__file__).parent.parent
TRAIN_STREAMING = ROOT / "train_streaming.py"
KEYFRAMES = ROOT / "eligsir" / "mapping" / "scheduler" / "keyframes.py"
OBSERVATION_SAMPLE_CAP = getattr(admission_fingerprint, "OBSERVATION_SAMPLE_CAP", 2)


def _decision(**overrides):
    values = {
        "admitted": False,
        "reason": "no_motion",
        "pose_delta_m": 0.125,
        "pose_delta_deg": 4.5,
        "render_overlap": 0.6,
        "quality": 0.75,
        "trace": {
            "novelty_gate_fraction": 0.2,
            "observation_values": {
                "blur_laplacian": 12.0,
                "exposure_delta": 0.1,
            },
        },
        "translation_velocity_m_s": 0.4,
        "translation_velocity_ema_m_s": 0.3,
        "rotation_velocity_deg_s": 2.0,
        "rotation_velocity_ema_deg_s": 1.5,
        "source_dt_s": 0.05,
        "source_frame_delta": 2,
        "implied_input_fps": 40.0,
        "effective_min_translation_m": 0.15,
        "controller_target_interval_s": 0.5,
        "controller_target_interval_source_frames": 20.0,
        "timebase_source": "source_timestamp",
        "threshold_clamp": "none",
        "motion_trigger": "translation",
        "starvation_age_source_s": 0.2,
        "starvation_age_source_frames": 8,
        "starvation_age_basis": "source_timestamp",
        "soft_backstop_mode": "target_time",
        "soft_backstop_domain": "seconds",
        "hard_backstop_domain": "frames",
        "soft_backstop_threshold_s": 0.5,
        "soft_backstop_threshold_frames": 20.0,
        "hard_backstop_threshold_s": None,
        "hard_backstop_threshold_frames": 40.0,
        "soft_backstop_preemptable": False,
        "soft_backstop_triggered": False,
        "hard_backstop_triggered": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sidecar_builder_preserves_controller_and_backstop_fields():
    """Removing a decision field from the shared sidecar payload loses audit data."""
    builder = getattr(admission_fingerprint, "keyframe_decision_sidecar_fields", None)
    assert callable(builder), "sidecar fields must have one canonical builder"
    fields = builder(_decision())

    assert fields["translation_velocity_m_s"] == 0.4
    assert fields["rotation_velocity_ema_deg_s"] == 1.5
    assert fields["controller_target_interval_source_frames"] == 20.0
    assert fields["motion_trigger"] == "translation"
    assert fields["soft_backstop_domain"] == "seconds"
    assert fields["hard_backstop_threshold_frames"] == 40.0


def test_both_production_decision_sites_delegate_to_shared_sidecar_builder():
    """A future per-site dict edit must not make normal and pregate rows drift."""
    tree = ast.parse(TRAIN_STREAMING.read_text())
    nested = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_keyframe_admission_closure", "_keyframe_pregate_closure"}
    }
    assert set(nested) == {"_keyframe_admission_closure", "_keyframe_pregate_closure"}

    for name, node in nested.items():
        calls = [
            call for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "keyframe_decision_sidecar_fields"
        ]
        assert len(calls) == 1, f"{name} must use the shared sidecar builder"


def test_observation_summary_is_bounded_and_partitioned_by_decision_outcome():
    """Outcome summaries must retain measured characteristics without unbounded logs."""
    fp = AdmissionFingerprint()
    fp.record_decision_observation(_decision())
    fp.record_decision_observation(_decision(admitted=True, reason="pose_delta"))
    for index in range(OBSERVATION_SAMPLE_CAP + 1):
        fp.record_decision_observation(
            _decision(pose_delta_m=float(index), trace={})
        )

    observations = fp.to_dict()["observation_summary"]
    rejected = observations["reject:no_motion"]
    admitted = observations["admit:pose_delta"]

    # The aggregate keeps an exact stream count while retaining only a bounded,
    # unbiased reservoir for its percentile estimate.  A first-N buffer would
    # lie about the population after a long live run.
    assert rejected["translation_m"]["n"] == OBSERVATION_SAMPLE_CAP + 2
    assert rejected["translation_m"]["sampled_n"] == OBSERVATION_SAMPLE_CAP
    assert rejected["rotation_deg"]["p50"] == 4.5
    assert rejected["coverage_new_voxel_fraction"]["p50"] == 0.2
    assert rejected["blur_laplacian"]["p50"] == 12.0
    assert rejected["exposure_delta"]["p50"] == 0.1
    assert rejected["quality"]["p50"] == 0.75
    assert admitted["translation_m"]["p50"] == 0.125


def test_funnel_joins_only_canonical_counts_and_marks_absent_producers_unknown():
    """Inventing prefilter/materialization counts would make source rates false."""
    fp = AdmissionFingerprint()
    for _ in range(10):
        fp.record_admit("max_gap", gap=10, forced_hard=True)
    fp.attach_canonical_funnel(
        admission_cadence={"source_observed": 120},
        realized_cadence={
            "clocks": {
                "source_released": 100,
                "candidate_evaluated": 80,
                "keyframe_admitted": 20,
            }
        },
        ingestion={"source_decimation": {"effective_stride": 2}},
    )

    report = fp.to_dict()
    stages = report["funnel"]["stages"]
    # #805's source-observed producer is not live.  An unrelated
    # admission_cadence field is not an authoritative substitute.
    assert stages["source_observed"]["count"] is None
    assert stages["source_observed"]["status"] == "unknown"
    assert "#805" in stages["source_observed"]["source"]
    assert "admission_cadence.source_observed" in stages["source_observed"]["reason"]
    assert stages["source_released"]["count"] == 100
    assert stages["candidate_evaluated"]["count"] == 80
    assert stages["keyframe_committed"]["count"] == 20
    assert stages["source_frame_materialized_valid"]["status"] == "unknown"
    assert stages["prefilter_eligible"]["count"] is None
    assert report["owner_shares"]["admit"]["denominator"] == "candidate_evaluated"
    assert report["owner_shares"]["admit"]["values"]["max_gap"] == 0.125
    assert report["owner_shares"]["candidate_reject"]["denominator"] == "candidate_evaluated"
    assert report["owner_shares"]["keyframe_commit_rate_per_prefilter_eligible"]["value"] is None


def test_funnel_without_historical_producers_is_explicitly_unknown():
    """Backfills must not convert a missing producer into a zero counter."""
    report = AdmissionFingerprint().to_dict()
    stages = report["funnel"]["stages"]

    assert stages["source_observed"]["status"] == "unknown"
    assert stages["candidate_evaluated"]["count"] is None
    assert "unknown is not zero" in stages["candidate_probed"]["reason"]
    assert report["owner_shares"]["admit"]["status"] == "unknown"


def test_novelty_none_cause_keeps_exception_distinct_from_missing_depth():
    """An exception that silently passes the force gate is not a no-depth result."""
    fp = AdmissionFingerprint()
    fp.record_novelty_gate({
        "novelty_gate_outcome": "not_evaluated",
        "novelty_gate_unavailable_reason": "frac_unavailable",
        "novelty_gate_none_cause": "exception",
    })
    fp.record_novelty_gate({
        "novelty_gate_outcome": "not_evaluated",
        "novelty_gate_unavailable_reason": "frac_unavailable",
        "novelty_gate_none_cause": "no_depth",
    })

    assert fp.to_dict()["novelty_gate"]["none_cause_counts"] == {
        "exception": 1,
        "no_depth": 1,
    }


def test_reject_owner_shares_do_not_blend_pregate_and_full_populations():
    """Pregate rejections precede candidate evaluation, so their share cannot
    use #289's candidate_evaluated denominator."""
    fp = AdmissionFingerprint()
    for _ in range(2):
        fp.record_candidate()
        fp.record_reject("low_overlap", population="candidate_evaluated")
    for _ in range(10):
        fp.record_reject("no_motion", population="prefilter_evaluated")
    fp.attach_canonical_funnel(
        admission_cadence=None,
        realized_cadence={"clocks": {"candidate_evaluated": 2}},
        ingestion=None,
    )

    shares = fp.to_dict()["owner_shares"]
    assert shares["candidate_reject"]["values"] == {"low_overlap": 1.0}
    assert all(value <= 1.0 for value in shares["candidate_reject"]["values"].values())
    assert shares["prefilter_reject"]["denominator"] == "prefilter_evaluated"
    assert shares["prefilter_reject"]["values"] is None
    assert shares["reject"]["status"] == "mixed_populations_no_share"


def test_reservoir_summary_represents_late_distribution_and_bounds_controller_storage():
    """A late regime shift must affect the summary without retaining every
    controller sample in a live run."""
    fp = AdmissionFingerprint()
    for value in ([0.0] * OBSERVATION_SAMPLE_CAP + [100.0] * OBSERVATION_SAMPLE_CAP):
        decision = _decision(
            pose_delta_m=value,
            translation_velocity_m_s=value,
            trace={},
        )
        fp.record_decision_observation(decision)
        fp.record_controller_state(decision)
    fp.finalize()

    observation = fp.to_dict()["observation_summary"]["reject:no_motion"]["translation_m"]
    controller = fp.to_dict()["controller"]["translation_velocity_inst_m_s"]
    for summary in (observation, controller):
        assert summary["n"] == 2 * OBSERVATION_SAMPLE_CAP
        assert summary["sampled_n"] == OBSERVATION_SAMPLE_CAP
        assert summary["min"] == 0.0
        assert summary["max"] == 100.0
        # The median can land on either side of an exactly 50/50 two-mode
        # population.  The reservoir mean still proves late samples entered
        # the estimate (the old first-256 buffer was exactly 0.0).
        assert summary["mean"] > 10.0


def test_trigger_age_storage_is_bounded_and_reported():
    fp = AdmissionFingerprint()
    for i in range(OBSERVATION_SAMPLE_CAP * 4):
        decision = _decision(
            admitted=True,
            reason="pose_delta",
            trace={"candidate_trigger": "normal"},
            starvation_age_source_s=float(i),
            starvation_age_source_frames=i,
            motion_trigger="translation",
        )
        fp.record_backstop_state(decision, True)
    assert len(fp._admit_trigger_age_s) == OBSERVATION_SAMPLE_CAP
    assert len(fp._admit_trigger_age_frames) == OBSERVATION_SAMPLE_CAP
    fp.finalize()
    assert fp.to_dict()["admission_gate"]["trigger_age_source_s"]["n"] == float(OBSERVATION_SAMPLE_CAP * 4)


def test_every_coverage_fraction_consumer_has_a_trace_identity():
    """The max-gap veto, novelty force, ordinary gate, and quality scorer all
    consume the same nullable measurement; a new untraced consumer would
    recreate the silent-pass ambiguity this issue closes."""
    source = KEYFRAMES.read_text()
    assert source.count("_coverage_new_voxel_frac(cam, frame)") == 4
    for consumer in (
        "max_gap_novelty_veto",
        "novelty_force",
        "ordinary_coverage_gate",
        "quality_new_surface_ratio",
    ):
        assert f'"{consumer}"' in source
