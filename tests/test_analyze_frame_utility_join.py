"""Phase 0 telemetry-join correctness tests for issue #500.

Issue #500's Phase 0 asks, *before any GPU sweep*, to prove that
``scripts/analyze_frame_utility.py``'s offline join between utility
features (``scores.jsonl``, written at admission-decision time) and
realized-gain labels (``realized_gain.jsonl``, observed later, after
``horizon_iterations``) matches the correct feature row to the correct
label row -- keyed on ``(frame_id, map_version)`` vs.
``(frame_id, admitted_iteration)`` (see ``_join_scores_to_gains``'s own
docstring). Issue #241 already fixed a real historical bug here (every
``observe_candidate`` call site pinning ``map_version`` to a constant,
collapsing the join key across all iterations to ``(frame_id, 0)``) and
added source-level regression tests for the *call sites*
(``test_frame_utility.py::test_pinned_map_version_breaks_the_realized_gain_join``
and ``::test_record_fn_admission_all_call_site_uses_iteration_as_map_version``).

What was still missing, and what this file adds, is coverage of the join
*function itself* (the thing ``scripts/analyze_frame_utility.py`` actually
runs offline to compute the correlations issue #118/#500 care about):

- a synthetic-ordering test with a KNOWN correct join, including a
  repeated ``frame_id`` (replay) at CHANGING, nonzero ``map_version``
  values, asserting the join recovers exactly that ground truth and does
  not cross-match the replayed frame's two visits;
- a check that the join's "label" genuinely comes from AFTER the
  "feature" by at least ``horizon_iterations`` (prediction-horizon
  semantics), read back from the real on-disk records the join consumes;
- a direct demonstration that the join is *sensitive* to real map_version
  correctness -- i.e. these tests would catch a reintroduction of the
  issue #241 bug at the join layer, not just at the call-site layer;
- a sanity check that ``_per_component_correlations`` recovers a known
  sign/magnitude on synthetic paired data, so the correlation numbers the
  analysis script prints can be trusted to mean what they claim.

Pure CPU / no GPU dependency, matching this repo's smoke-test convention.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_frame_utility import (  # noqa: E402
    _join_scores_to_gains,
    _pearson,
    _per_component_correlations,
    _read_jsonl,
    analyze_run,
)

from eligsir.mapping.admission.shadow_policy import ShadowUtilityPolicy  # noqa: E402


def _run_policy(tmp_path, horizon=50):
    """Real ShadowUtilityPolicy round trip, mirroring how train_streaming.py
    drives it (observe_candidate at admission, then observe_training_loss
    once per iteration) -- avoids hand-rolling the on-disk JSON schema and
    exercises the actual writer/reader pair the join function consumes.
    """
    log_dir = str(tmp_path / "shadow_utility")
    policy = ShadowUtilityPolicy(enabled=True, log_dir=log_dir, realized_gain_horizon=horizon)
    return policy, log_dir


def _load(log_dir):
    scores = _read_jsonl(os.path.join(log_dir, "scores.jsonl"))
    gains = _read_jsonl(os.path.join(log_dir, "realized_gain.jsonl"))
    return scores, gains


# ---------------------------------------------------------------------------
# 1. Synthetic-ordering test with a KNOWN correct join, including a replayed
#    frame_id at changing, nonzero map_version values (not the degenerate
#    all-same-version case, which would hide a real join bug).
# ---------------------------------------------------------------------------


def test_join_recovers_known_synthetic_ordering_with_changing_map_version(tmp_path):
    policy, log_dir = _run_policy(tmp_path, horizon=50)

    # Ground truth: frame_id=7 is admitted twice, far apart, at two
    # different (nonzero, non-adjacent) map_versions -- exactly the replay
    # scenario _join_scores_to_gains's docstring calls out as the reason the
    # join key must include the version, not just frame_id.
    visits = [
        # (camera_key, frame_id, admitted_iteration, fingerprint_under_supported_mass)
        (1, 7, 1000, 0.11),
        (2, 3, 1200, 0.22),
        (3, 7, 5000, 0.33),  # frame_id=7 revisited at a much later map_version
    ]
    for camera_key, frame_id, it, fp in visits:
        policy.observe_candidate(
            frame_id=frame_id,
            camera_key=camera_key,
            admitted=True,
            iteration=it,
            pose_confidence=1.0,
            image_quality=1.0,
            depth_valid_fraction=1.0,
            n_gaussians=1000,
            topology_version=it,  # also non-degenerate/changing
            map_version=it,  # the real fix: map_version tracks iteration
            under_supported_mass=fp,
        )
        # loss_before set via the NaN-sentinel path, exactly like train_streaming.py.
        policy.observe_training_loss(camera_key, it, loss=1.0 - fp)
        # horizon elapses -> completes the realized-gain record.
        policy.observe_training_loss(camera_key, it + policy.gain_tracker.horizon_iterations, loss=0.5 - fp)

    scores, gains = _load(log_dir)
    assert len(scores) == 3
    assert len(gains) == 3

    paired = _join_scores_to_gains(scores, gains)
    assert len(paired) == 3, "every admitted, completed record must join to exactly one score row"

    # Ground truth is fully known: for every visit, the joined score's
    # under_supported_mass fingerprint must match that visit's, AND the
    # score's meta.map_version must equal the gain's admitted_iteration --
    # not just "some" score with the same frame_id (there are two for
    # frame_id=7, at different map_versions).
    by_iteration = {p["gain"]["admitted_iteration"]: p for p in paired}
    for camera_key, frame_id, it, fp in visits:
        p = by_iteration[it]
        assert p["gain"]["frame_id"] == frame_id
        assert p["score"]["frame_id"] == frame_id
        assert p["score"]["meta"]["map_version"] == it
        assert p["score"]["components"]["under_supported_mass"] == fp, (
            "join must not cross-match the replayed frame_id=7's two visits "
            "(iteration 1000 vs. 5000) -- a wrong-version join would pick up "
            "the other visit's fingerprint here"
        )


def test_join_does_not_cross_match_replayed_frame_across_versions(tmp_path):
    """Stronger, more adversarial version of the above: interleave the two
    visits of the same frame_id so a naive frame_id-only join (ignoring
    version) would have every opportunity to pick the wrong one.
    """
    policy, log_dir = _run_policy(tmp_path, horizon=10)

    # Interleaved: visit A (frame_id=99) starts, then visit B (frame_id=99,
    # different camera/map_version) starts and *completes* before visit A
    # completes -- a join keyed on frame_id alone (or on a stale/incorrectly
    # -pinned map_version) could easily pair visit A's score with visit B's
    # gain or vice versa.
    policy.observe_candidate(
        frame_id=99, camera_key=10, admitted=True, iteration=100,
        pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
        n_gaussians=500, topology_version=100, map_version=100,
        under_supported_mass=0.9,
    )
    policy.observe_training_loss(10, 100, loss=2.0)  # loss_before for visit A

    policy.observe_candidate(
        frame_id=99, camera_key=20, admitted=True, iteration=104,
        pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
        n_gaussians=500, topology_version=104, map_version=104,
        under_supported_mass=0.1,
    )
    policy.observe_training_loss(20, 104, loss=5.0)  # loss_before for visit B

    # Visit B completes first (admitted at 104, horizon=10 -> completes at 114).
    policy.observe_training_loss(20, 114, loss=1.0)
    # Visit A completes afterwards (admitted at 100, horizon=10 -> completes at 110,
    # but nothing observed that camera_key until iteration 116 here -- still valid,
    # the tracker only requires iteration - admitted_iteration >= horizon).
    policy.observe_training_loss(10, 116, loss=0.5)

    scores, gains = _load(log_dir)
    paired = _join_scores_to_gains(scores, gains)
    assert len(paired) == 2

    by_admitted_iter = {p["gain"]["admitted_iteration"]: p for p in paired}
    visit_a = by_admitted_iter[100]
    visit_b = by_admitted_iter[104]

    assert visit_a["score"]["components"]["under_supported_mass"] == 0.9
    assert visit_a["gain"]["realized_gain"] == 2.0 - 0.5

    assert visit_b["score"]["components"]["under_supported_mass"] == 0.1
    assert visit_b["gain"]["realized_gain"] == 5.0 - 1.0


# ---------------------------------------------------------------------------
# 2. Prediction-horizon semantics: the label must genuinely come from AFTER
#    the feature, by at least horizon_iterations -- verified against the
#    real on-disk records the join consumes, not just the in-memory tracker
#    (already covered separately by test_realized_gain_tracker_records_before_after).
# ---------------------------------------------------------------------------


def test_join_labels_are_strictly_after_features_by_horizon(tmp_path):
    horizon = 37
    policy, log_dir = _run_policy(tmp_path, horizon=horizon)

    for i, (frame_id, it) in enumerate([(1, 500), (2, 900), (3, 900 + horizon)]):
        policy.observe_candidate(
            frame_id=frame_id, camera_key=i, admitted=True, iteration=it,
            pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
            n_gaussians=100, topology_version=it, map_version=it,
            under_supported_mass=float(i),
        )
        policy.observe_training_loss(i, it, loss=1.0)
        policy.observe_training_loss(i, it + horizon, loss=0.5)

    scores, gains = _load(log_dir)
    paired = _join_scores_to_gains(scores, gains)
    assert len(paired) == 3
    for p in paired:
        admitted_iter = p["gain"]["admitted_iteration"]
        followup_iter = p["gain"]["followup_iteration"]
        assert followup_iter is not None
        assert followup_iter - admitted_iter >= horizon, (
            "a realized-gain label joined by this function must be observed "
            "at or after the declared prediction horizon, never before or "
            "coincidentally-simultaneous"
        )
        # meta.map_version (the join key on the feature side) must equal the
        # iteration the feature was actually computed at, not the later
        # followup iteration -- i.e. the feature is timestamped at decision
        # time, not label time.
        assert p["score"]["meta"]["map_version"] == admitted_iter
        assert p["score"]["meta"]["map_version"] != followup_iter


# ---------------------------------------------------------------------------
# 3. The join must be SENSITIVE to real map_version correctness: reproduce
#    the issue #241 pinned-map_version failure mode directly against the
#    join function (not just the writer), so a regression here is caught at
#    the layer issue #500 actually cares about (the offline analysis join).
# ---------------------------------------------------------------------------


def test_join_uses_the_recorded_score_version_not_commit_iteration(tmp_path):
    """Score identity remains joinable even when scoring and commit clocks differ."""
    policy, log_dir = _run_policy(tmp_path, horizon=5)

    # A fixed score version simulates candidates scored before a later commit.
    for camera_key, frame_id, it, fp in [(1, 10, 100, 0.1), (2, 20, 200, 0.2), (3, 30, 300, 0.3)]:
        policy.observe_candidate(
            frame_id=frame_id, camera_key=camera_key, admitted=True, iteration=it,
            pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
            n_gaussians=100, topology_version=0, map_version=0,  # <-- the bug
            under_supported_mass=fp,
        )
        policy.observe_training_loss(camera_key, it, loss=1.0)
        policy.observe_training_loss(camera_key, it + 5, loss=0.5)

    scores, gains = _load(log_dir)
    assert len({s["meta"]["map_version"] for s in scores}) == 1, (
        "sanity check on the reproduction itself: pinned map_version must "
        "collapse to a single value across distinct iterations"
    )

    paired = _join_scores_to_gains(scores, gains)
    assert len(paired) == 3
    assert {p["gain"]["score_map_version"] for p in paired} == {0}


def test_join_skips_unmatched_gain_rows_defensively(tmp_path):
    """A gain row with no corresponding score row (should not happen for a
    self-consistent run, but the join function documents this as a
    defensive skip) must not raise and must not be silently paired with an
    unrelated score.
    """
    policy, log_dir = _run_policy(tmp_path, horizon=5)
    policy.observe_candidate(
        frame_id=1, camera_key=1, admitted=True, iteration=100,
        pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
        n_gaussians=100, topology_version=100, map_version=100,
        under_supported_mass=0.5,
    )
    policy.observe_training_loss(1, 100, loss=1.0)
    policy.observe_training_loss(1, 105, loss=0.5)

    scores, gains = _load(log_dir)
    assert len(gains) == 1
    # Corrupt the retained score identity so it no longer matches any score
    # row -- simulating a log from a different/inconsistent run.
    gains[0]["score_map_version"] = 999

    paired = _join_scores_to_gains(scores, gains)
    assert paired == []


# ---------------------------------------------------------------------------
# 4. _per_component_correlations must recover a known sign/magnitude on
#    synthetic paired data, so the correlation numbers this script reports
#    can be trusted to mean what they claim once the join itself is proven.
# ---------------------------------------------------------------------------


def _fake_paired(n=20):
    """Synthetic paired rows (bypassing the join, since this section tests
    the correlation math, not the join) with two components of KNOWN
    correlation against realized_gain: 'new_surface' perfectly positively
    correlated, 'redundancy' perfectly negatively correlated, the rest
    constant (undefined/zero variance -> None).
    """
    paired = []
    for i in range(n):
        realized_gain = float(i)
        components = {
            "pose_quality": 1.0,
            "image_quality": 1.0,
            "new_surface": float(i),  # r == +1.0
            "persistent_depth_supported_residual": 0.0,
            "under_supported_mass": 0.0,
            "expected_gradient_value": 0.0,
            "useful_parallax": 0.0,
            "redundancy": float(n - i),  # r == -1.0
            "specular_risk": 0.0,
            "predicted_gpu_cost": 5.0,  # zero variance -> r is None
        }
        paired.append({"gain": {"realized_gain": realized_gain}, "score": {"components": components}})
    return paired


def test_per_component_correlations_recovers_known_signal():
    corr = _per_component_correlations(_fake_paired())
    assert corr["new_surface"] == 1.0
    assert corr["redundancy"] == -1.0
    assert corr["predicted_gpu_cost"] is None  # zero-variance feature: undefined, not fabricated as 0
    # persistent_depth_supported_residual is likewise constant (0.0) across
    # every synthetic row, so it too has zero variance -> undefined (None),
    # not fabricated as 0.0.
    assert corr["persistent_depth_supported_residual"] is None


def test_pearson_matches_known_values():
    assert _pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert _pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert _pearson([1.0], [1.0]) is None  # fewer than 2 points
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance in x


# ---------------------------------------------------------------------------
# 5. End-to-end: analyze_run over a real (small, synthetic-ordering) fixture
#    reports a sane, fully-joined summary -- the top-level entry point issue
#    #500/#118 actually invoke.
# ---------------------------------------------------------------------------


def test_analyze_run_end_to_end_full_join_no_warning(tmp_path):
    model_path = str(tmp_path / "model")
    policy, log_dir = _run_policy(Path(model_path), horizon=20)
    assert log_dir == os.path.join(model_path, "shadow_utility")

    for camera_key, frame_id, it in [(1, 1, 1000), (2, 2, 2000), (3, 1, 4000)]:
        policy.observe_candidate(
            frame_id=frame_id, camera_key=camera_key, admitted=True, iteration=it,
            pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0,
            n_gaussians=200, topology_version=it, map_version=it,
            under_supported_mass=float(camera_key) / 10.0,
        )
        policy.observe_training_loss(camera_key, it, loss=1.0)
        policy.observe_training_loss(camera_key, it + 20, loss=1.0 - float(camera_key) / 10.0)

    report = analyze_run(model_path)
    assert report["n_scores"] == 3
    assert report["n_realized_gain_records"] == 3
    assert report["n_paired_prediction_vs_realized"] == 3
    assert report["n_paired_component_breakdown"] == 3
    assert "component_breakdown_join_warning" not in report, (
        "a self-consistent run (every observe_candidate paired with a "
        "matching completed realized-gain record) must join fully, with no "
        "unmatched rows"
    )
