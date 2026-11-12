"""Tests for the shadow frame-utility scorer (Wave 3, PR W3-06).

Pure CPU torch, no GPU/container dependency -- matches this repo's
"CPU-only feasible" smoke-test convention (see CLAUDE.md) and mirrors
tests/test_evidence.py's style for the substrate this package consumes.
"""

from __future__ import annotations

import json
import os

import torch

from eligsir.mapping.admission.cost_model import predict_gpu_cost, predicted_cost_ms_estimate
from eligsir.mapping.admission.realized_gain import RealizedGainTracker, under_supported_mass_from_tensors
from eligsir.mapping.admission.scoring import (
    depth_supported_rgb_residual,
    expected_gradient_value_from_history,
    recompute_utility,
    score_frame_utility,
)
from eligsir.mapping.admission.shadow_policy import ShadowUtilityPolicy
from eligsir.mapping.admission.utility_contracts import FrameUtilityComponents, UtilityWeights
from eligsir.mapping.evidence.contracts import GaussianEvidence, InputEvidence, RenderEvidence
from eligsir.mapping.evidence.gaussian_history import GaussianEvidenceHistory


def _input_evidence(pose=0.9, image=0.8, depth_valid=0.7):
    return InputEvidence(
        pose_confidence=pose,
        image_quality=image,
        depth_valid_fraction=depth_valid,
        geometry_reliability_mean=depth_valid,
    )


def _gaussian_evidence(n=16):
    zeros = torch.zeros(n)
    return GaussianEvidence(
        visibility=torch.rand(n) > 0.5,
        projected_radius=zeros,
        contribution=zeros,
        xyz_grad_ema=torch.rand(n) * 0.1,
        opacity_grad_ema=torch.rand(n) * 0.1,
        support=torch.rand(n),
        geometry_confidence=torch.rand(n),
        free_space_conflict=zeros,
        age=zeros,
    )


# ----------------------------------------------------------------------
# cost_model
# ----------------------------------------------------------------------
def test_predict_gpu_cost_zero_for_empty_map():
    assert predict_gpu_cost(0) == 0.0
    assert predict_gpu_cost(-5) == 0.0


def test_predict_gpu_cost_scales_with_n_and_tier():
    low_cost = predict_gpu_cost(300_000, "low")
    default_cost = predict_gpu_cost(300_000, "default")
    high_cost = predict_gpu_cost(300_000, "high")
    assert low_cost < default_cost < high_cost
    doubled = predict_gpu_cost(600_000, "default")
    assert doubled == 2 * default_cost


def test_predicted_cost_ms_estimate_nonnegative_and_monotonic():
    a = predicted_cost_ms_estimate(100_000)
    b = predicted_cost_ms_estimate(200_000)
    assert a >= 0.0
    assert b > a


# ----------------------------------------------------------------------
# scoring: residual/specular split
# ----------------------------------------------------------------------
def test_depth_supported_residual_zero_with_no_evidence():
    supported, specular = depth_supported_rgb_residual(None, depth_valid_fraction=1.0)
    assert supported == 0.0
    assert specular == 0.0


def test_high_residual_low_depth_support_routes_to_specular_not_residual():
    """This is the deliverable-6 sanity property: a frame whose only signal
    is a big RGB residual with NO depth corroboration (glass/specular) must
    score low on the *useful* residual term and high on specular risk, not
    the other way around.
    """
    residual = torch.full((4, 8, 8), 0.9)
    ev = RenderEvidence(rgb_residual=residual, depth_residual=None, alpha=None, rendered_depth=None, transmittance=None)
    supported_low_depth, specular_low_depth = depth_supported_rgb_residual(ev, depth_valid_fraction=0.02)
    supported_high_depth, specular_high_depth = depth_supported_rgb_residual(ev, depth_valid_fraction=0.98)
    assert supported_low_depth < supported_high_depth
    assert specular_low_depth > specular_high_depth


def test_expected_gradient_value_uses_history_not_backward():
    """No autograd graph is ever built by this function -- it only reads
    pre-computed EMA tensors (module docstring point 2)."""
    ge = _gaussian_evidence()
    val = expected_gradient_value_from_history(ge, visibility_mask=ge.visibility)
    assert val >= 0.0
    assert expected_gradient_value_from_history(None) == 0.0


def test_expected_gradient_value_none_when_no_history():
    assert expected_gradient_value_from_history(None) == 0.0


# ----------------------------------------------------------------------
# scoring: full formula
# ----------------------------------------------------------------------
def test_score_frame_utility_matches_manual_formula():
    weights = UtilityWeights()
    components = FrameUtilityComponents(
        pose_quality=0.9,
        image_quality=0.8,
        new_surface=0.5,
        persistent_depth_supported_residual=0.2,
        under_supported_mass=0.1,
        expected_gradient_value=0.05,
        useful_parallax=0.3,
        redundancy=0.1,
        specular_risk=0.02,
        predicted_gpu_cost=0.4,
    )
    expected = 0.9 * 0.8 * (
        weights.w_new_surface * 0.5
        + weights.w_residual * 0.2
        + weights.w_debt * 0.1
        + weights.w_gradient * 0.05
        + weights.w_parallax * 0.3
    ) - (weights.w_redundancy * 0.1 + weights.w_specular * 0.02 + weights.w_cost * 0.4)

    from eligsir.mapping.admission.utility_contracts import FrameUtilityScore

    score = FrameUtilityScore(frame_id=1, utility=0.0, components=components, weights=weights)
    assert abs(recompute_utility(score) - expected) < 1e-9


def test_score_frame_utility_end_to_end_no_evidence_is_well_formed():
    score = score_frame_utility(frame_id=7, input_evidence=None)
    assert score.frame_id == 7
    assert score.components.pose_quality == 0.0
    assert score.utility == recompute_utility(score)


def test_score_frame_utility_zero_pose_or_image_quality_zeroes_positive_terms():
    """pose_quality * image_quality gates the ENTIRE positive bracket -- a
    frame with confidently-tracked-but-blurry (or well-exposed-but-lost-
    tracking) input should not score well just because new_surface/residual
    happen to be high (deliverable 6: guards against blur/pose-error
    domination)."""
    ie_zero_pose = _input_evidence(pose=0.0, image=0.9)
    score = score_frame_utility(
        frame_id=1,
        input_evidence=ie_zero_pose,
        new_surface=1.0,
        under_supported_mass=1.0,
        useful_parallax=1.0,
    )
    assert score.utility <= 0.0  # only the (non-positive) penalty terms can remain


def test_higher_redundancy_and_specular_strictly_lower_score():
    ie = _input_evidence()
    base = score_frame_utility(frame_id=1, input_evidence=ie, new_surface=0.5)
    redundant = score_frame_utility(frame_id=1, input_evidence=ie, new_surface=0.5, redundancy=0.8)
    assert redundant.utility < base.utility


def test_higher_predicted_cost_lowers_score():
    ie = _input_evidence()
    cheap = score_frame_utility(frame_id=1, input_evidence=ie, new_surface=0.5, n_gaussians=10_000)
    expensive = score_frame_utility(frame_id=1, input_evidence=ie, new_surface=0.5, n_gaussians=5_000_000)
    assert expensive.utility < cheap.utility


# ----------------------------------------------------------------------
# realized_gain
# ----------------------------------------------------------------------
def test_realized_gain_tracker_records_before_after():
    tracker = RealizedGainTracker(horizon_iterations=10)
    tracker.register_admission(camera_key=1, frame_id=42, iteration=100, loss_before=0.5, predicted_utility=0.3)
    assert tracker.n_pending() == 1
    # Not yet due.
    assert tracker.observe_loss(1, 105, 0.4) is None
    assert tracker.n_pending() == 1
    # Due now.
    completed = tracker.observe_loss(1, 111, 0.2)
    assert completed is not None
    assert completed.loss_before == 0.5
    assert completed.loss_after == 0.2
    assert completed.realized_gain == 0.5 - 0.2
    assert tracker.n_pending() == 0
    assert len(tracker.completed()) == 1


def test_realized_gain_tracker_bounded_pending():
    tracker = RealizedGainTracker(horizon_iterations=1000, max_pending=2)
    tracker.register_admission(camera_key=1, frame_id=1, iteration=0, loss_before=1.0)
    tracker.register_admission(camera_key=2, frame_id=2, iteration=0, loss_before=1.0)
    tracker.register_admission(camera_key=3, frame_id=3, iteration=0, loss_before=1.0)
    assert tracker.n_pending() == 2  # oldest evicted, never unbounded


def test_under_supported_mass_from_tensors_none_when_missing():
    assert under_supported_mass_from_tensors(None, torch.rand(4)) is None
    assert under_supported_mass_from_tensors(torch.rand(4), None) is None
    val = under_supported_mass_from_tensors(torch.zeros(4), torch.ones(4), threshold=0.5)
    assert val == 4.0  # all 4 below threshold, opacity=1 each


# ----------------------------------------------------------------------
# shadow_policy: deliverable 1 (no decision effect) + deliverable 2 (logging)
# ----------------------------------------------------------------------
def test_shadow_policy_disabled_is_a_true_no_op(tmp_path):
    policy = ShadowUtilityPolicy(enabled=False, log_dir=str(tmp_path / "shadow"))
    result = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=True,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
    )
    assert result is None
    assert not os.path.exists(tmp_path / "shadow")  # never even creates the log dir
    policy.observe_training_loss(1, 100, 0.1)  # must not raise
    assert policy.cost_overhead_ms_summary()["n_samples"] == 0


def test_shadow_policy_logs_every_component_separately(tmp_path):
    log_dir = str(tmp_path / "shadow")
    policy = ShadowUtilityPolicy(enabled=True, log_dir=log_dir, realized_gain_horizon=1)
    score = policy.observe_candidate(
        frame_id=5,
        camera_key=99,
        admitted=True,
        iteration=10,
        pose_confidence=0.8,
        image_quality=0.7,
        depth_valid_fraction=0.6,
        n_gaussians=50_000,
    )
    assert score is not None
    with open(os.path.join(log_dir, "scores.jsonl")) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    logged_components = lines[0]["components"]
    # Every dataclass field of FrameUtilityComponents must be present and
    # separately readable -- deliverable 2.
    import dataclasses as dc

    for f_ in dc.fields(FrameUtilityComponents):
        assert f_.name in logged_components


def test_shadow_policy_realized_gain_roundtrip(tmp_path):
    log_dir = str(tmp_path / "shadow")
    policy = ShadowUtilityPolicy(enabled=True, log_dir=log_dir, realized_gain_horizon=2)
    policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=True,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
    )
    policy.observe_training_loss(1, 0, 0.5)  # sets loss_before via NaN-sentinel path
    policy.observe_training_loss(1, 1, 0.4)  # horizon not elapsed yet
    policy.observe_training_loss(1, 2, 0.2)  # horizon elapsed -> completes
    with open(os.path.join(log_dir, "realized_gain.jsonl")) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 1
    assert lines[0]["loss_before"] == 0.5
    assert lines[0]["loss_after"] == 0.2
    assert lines[0]["realized_gain"] == 0.5 - 0.2


def test_shadow_policy_cost_overhead_measured(tmp_path):
    policy = ShadowUtilityPolicy(enabled=True, log_dir=str(tmp_path / "shadow"))
    for i in range(5):
        policy.observe_candidate(
            frame_id=i,
            camera_key=i,
            admitted=False,
            iteration=i,
            pose_confidence=0.5,
            image_quality=0.5,
            depth_valid_fraction=0.5,
            n_gaussians=1000,
        )
    summary = policy.cost_overhead_ms_summary()
    assert summary["n_samples"] == 5
    assert summary["mean_ms"] >= 0.0


def test_shadow_policy_reads_history_when_valid(tmp_path):
    history = GaussianEvidenceHistory()
    ge = _gaussian_evidence()
    history.update(ge, topology_version=3, map_version=3)
    policy = ShadowUtilityPolicy(enabled=True, log_dir=str(tmp_path / "shadow"), gaussian_history=history)
    score = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=False,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
        under_supported_mass=0.4,
        topology_version=3,
        map_version=3,
    )
    assert score.components.expected_gradient_value >= 0.0
    assert score.components.under_supported_mass == 0.4


def test_shadow_policy_stale_history_ignored_not_crashed(tmp_path):
    history = GaussianEvidenceHistory()
    ge = _gaussian_evidence()
    history.update(ge, topology_version=3, map_version=3)
    policy = ShadowUtilityPolicy(enabled=True, log_dir=str(tmp_path / "shadow"), gaussian_history=history)
    # Requesting a DIFFERENT topology_version than the history holds must not
    # raise StaleEvidenceError up through the policy -- it should just fall
    # back to "no gaussian evidence this call".
    score = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=False,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
        topology_version=99,
        map_version=99,
    )
    assert score.components.expected_gradient_value == 0.0


# ----------------------------------------------------------------------
# Issue #118: under_supported_mass rescale (default "raw" is a no-op)
# ----------------------------------------------------------------------


def test_debt_normalization_invalid_value_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ShadowUtilityPolicy(enabled=True, log_dir=str(tmp_path / "shadow"), debt_normalization="bogus")


def test_debt_normalization_default_raw_matches_pre_existing_behavior(tmp_path):
    """Default construction (no debt_normalization kwarg) must be
    byte-identical to before this flag existed -- under_supported_mass
    passed straight through to the formula, unscaled."""
    policy = ShadowUtilityPolicy(enabled=True, log_dir=str(tmp_path / "shadow"))
    assert policy.debt_normalization == "raw"
    score = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=False,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
        under_supported_mass=6401.0,
    )
    assert score.components.under_supported_mass == 6401.0


def test_debt_normalization_per_gaussian_divides_by_n_gaussians(tmp_path):
    policy = ShadowUtilityPolicy(
        enabled=True, log_dir=str(tmp_path / "shadow"), debt_normalization="per_gaussian"
    )
    score = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=False,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
        under_supported_mass=500.0,
    )
    assert score.components.under_supported_mass == 0.5


def test_debt_normalization_per_gaussian_zero_n_gaussians_falls_back_to_raw(tmp_path):
    """n_gaussians == 0 (e.g. a scoring call before any Gaussian exists)
    must not divide by zero -- falls back to the raw value rather than
    crashing or fabricating a normalized value."""
    policy = ShadowUtilityPolicy(
        enabled=True, log_dir=str(tmp_path / "shadow"), debt_normalization="per_gaussian"
    )
    score = policy.observe_candidate(
        frame_id=1,
        camera_key=1,
        admitted=False,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=0,
        under_supported_mass=500.0,
    )
    assert score.components.under_supported_mass == 500.0


def test_debt_normalization_realized_gain_diagnostic_stays_raw(tmp_path):
    """under_supported_mass_before logged into realized_gain.jsonl (a
    separate LSOM-style diagnostic, not a scoring input) must stay at the
    raw value regardless of debt_normalization -- only the scoring formula
    input is rescaled."""
    log_dir = str(tmp_path / "shadow")
    policy = ShadowUtilityPolicy(
        enabled=True, log_dir=log_dir, realized_gain_horizon=1, debt_normalization="per_gaussian"
    )
    policy.observe_candidate(
        frame_id=1,
        camera_key=42,
        admitted=True,
        iteration=0,
        pose_confidence=0.9,
        image_quality=0.9,
        depth_valid_fraction=0.9,
        n_gaussians=1000,
        under_supported_mass=500.0,
    )
    record = policy.gain_tracker._pending[42]  # noqa: SLF001
    assert record.under_supported_mass_before == 500.0


# ---------------------------------------------------------------------------
# Issue #241: the admission=all shadow-scoring call site in train_streaming.py
# (`_record_fn`, the ONLY observe_candidate call site that fires under the
# `--streaming_frame_admission all` default -- see CLAUDE.md's Run Contracts)
# must pass a real, per-iteration map_version, not a value pinned to a
# constant (the pre-fix code used `gaussians.topology_version`, which is
# permanently 0 per the documented W3-17/W3-39 gap). scripts/
# analyze_frame_utility.py's per-component join (issue #118, PR #234/#240)
# matches each realized_gain.jsonl row back to the scores.jsonl row it came
# from on (frame_id, admitted_iteration) vs. (frame_id, meta.map_version) --
# a constant map_version collapses every row's join key to (frame_id, 0),
# breaking the join for any row admitted at iteration != 0.
# ---------------------------------------------------------------------------


def test_pinned_map_version_breaks_the_realized_gain_join():
    """Demonstrates the bug directly against ShadowUtilityPolicy: if every
    observe_candidate call passes the same constant map_version (the pre-fix
    behavior at train_streaming.py's admission=all call site), the resulting
    scores.jsonl rows are indistinguishable by map_version even though they
    were scored at different iterations -- exactly the failure mode issue
    #241 reports.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        policy = ShadowUtilityPolicy(enabled=True, log_dir=tmp)
        for it in (100, 250, 400):
            policy.observe_candidate(
                frame_id=it,
                camera_key=it,
                admitted=True,
                iteration=it,
                pose_confidence=1.0,
                image_quality=1.0,
                depth_valid_fraction=1.0,
                n_gaussians=1000,
                topology_version=0,
                map_version=0,  # <-- the bug: pinned constant, not `it`
                under_supported_mass=0.0,
            )
        with open(os.path.join(tmp, "scores.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        map_versions = {r["meta"]["map_version"] for r in rows}
        assert map_versions == {0}, (
            "sanity check on the reproduction itself: a pinned-constant "
            "map_version must collapse to a single value across distinct "
            "iterations"
        )


def test_record_fn_admission_all_call_site_uses_iteration_as_map_version():
    """Source-level regression guard on the actual fix (train_streaming.py's
    `_record_fn`, the admission=all shadow-scoring call site around line
    2628). Must pass map_version=int(iteration) -- matching the OTHER two
    real observe_candidate call sites in the same file
    (`_keyframe_admission_closure` and the population-energy-log site) --
    not a value derived from `gaussians.topology_version`, which is pinned
    to 0 (see this file's module docstring above and CLAUDE.md's W3-17/W3-39
    entries).

    This deliberately checks train_streaming.py's source text rather than
    exercising the closure directly: `_record_fn` is defined nested inside
    `streaming_training()` and depends on a full training-loop setup (map
    backend, scene, coordinator, ...) that isn't worth constructing for a
    CPU-only regression test of a single keyword argument. Mirrors the
    existing precedent for this style, e.g.
    tests/test_insert_from_frame_default_path_birth_iter.py.
    """
    import re

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "train_streaming.py")) as f:
        source = f.read()

    def_start = source.index("    def _record_fn(cam, frame, is_train, is_holdout):")
    def_end = source.index("\n    def _pose_delta_from_last_train(", def_start)
    record_fn_source = source[def_start:def_end]

    assert "observe_candidate(" in record_fn_source, (
        "_record_fn no longer calls observe_candidate at all -- this test's "
        "premise (the admission=all shadow-scoring call site) has moved or "
        "been removed; update or remove this regression test accordingly."
    )
    assert re.search(r"map_version\s*=\s*int\(iteration\)", record_fn_source), (
        "_record_fn's observe_candidate call must pass map_version=int(iteration) "
        "-- a constant/pinned value here breaks the (frame_id, map_version) vs. "
        "(frame_id, admitted_iteration) join scripts/analyze_frame_utility.py's "
        "per-component breakdown depends on (issue #241)."
    )
    assert "map_version=int(getattr(gaussians, \"topology_version\", 0))" not in record_fn_source, (
        "regression: _record_fn's map_version must not be sourced from the "
        "permanently-pinned-to-0 gaussians.topology_version fallback."
    )
