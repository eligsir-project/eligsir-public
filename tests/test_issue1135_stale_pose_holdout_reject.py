"""Regression tests for issue #1135 -- the ``_odom_valid is False`` guard in
``StreamingScene.ingest_next_frame`` drops reserved holdout frames the same
way the #1125 integrity veto used to, but the fix is deliberately NOT the
same as #1125's.

#1125 was resolved by bypassing the veto for holdouts/guards, because its
harm channel (a re-based-but-internally-coherent map-reinit pose) cannot
poison a holdout camera (never trained, never depth-inserted), and the
residual risk was already flagged run-wide by
``report.json[...]["map_reinit_invalidated"]``.

``_odom_valid is False`` is a different failure mode: per
``eligsir/io/sources/datasets/orbbec_rosbag.py``'s live-odometry
fallback, a rejected frame's pose is a literal COPY of the last
successfully-tracked keyframe's pose -- wrong in an unbounded,
frame-specific way, not a coherent re-based coordinate frame. Admitting it
into the test set would render the wrong part of the scene and silently
pollute ``mean_test_psnr``, with no run-level flag analogous to
``map_reinit_invalidated`` to catch it. So this guard stays universal
(holdouts/guards included) -- **Option B** from issue #1135 -- but the
resulting holdout-reservation shortfall is now counted
(``StreamingScene._stale_pose_holdout_rejected``) and surfaced in
``report.json["geometry"]["trajectory_coverage"]`` via
``compute_trajectory_coverage()`` (``utils/partition_eval.py``), so a
shrunken ``n_test`` is attributable instead of mysterious.

Reuses the ``_FakeIngestSelf``/``_FakeFrame`` harness from
``tests/test_issue806_integrity_observer_ordering.py`` (imported, not
re-duplicated) -- see that module's own docstring for why it doesn't import
``train_streaming`` at collection time.
"""

from __future__ import annotations

import pytest

from scene.streaming_scene import StreamingScene
from test_issue806_integrity_observer_ordering import _FakeFrame, _FakeIngestSelf
from eligsir.utils.partition_eval import compute_trajectory_coverage


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )


def _ingest_all(fake_self, **kwargs):
    """Drain the whole fake source, returning the per-frame results."""
    results = []
    while fake_self.has_next_frame():
        results.append(StreamingScene.ingest_next_frame(fake_self, **kwargs))
    return results


def test_stale_pose_holdout_is_dropped_not_admitted_to_test_set():
    """Core Option-B invariant: unlike #1125's map-reinit veto, a reserved
    holdout with a stale/copied odometry pose is NOT bypassed into the test
    set -- it is dropped, exactly like an ordinary candidate would be."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(6)])
    fake_self._holdout_indices = {3}
    fake_self._guard_indices = set()
    fake_self._frames[3]._odom_valid = False

    results = _ingest_all(fake_self)

    assert fake_self._test_cameras == []  # the holdout never became a camera
    cam, frame, is_train = results[3]
    assert (cam, frame.index, is_train) == (None, 3, False)


def test_stale_pose_holdout_rejection_is_counted():
    """The new attribution counter increments exactly once per reserved
    holdout dropped by this guard, and not for ordinary (non-holdout,
    non-guard) frames rejected by the same guard."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(8)])
    fake_self._holdout_indices = {2, 5}
    fake_self._guard_indices = set()
    for idx in (1, 2, 5, 6):
        fake_self._frames[idx]._odom_valid = False

    _ingest_all(fake_self)

    # Only the two HOLDOUT frames among the four odom-invalid frames count.
    assert fake_self._stale_pose_holdout_rejected == 2
    # All four odom-invalid frames (holdout or not) still count as ordinary
    # keyframe rejections -- this guard's overall reject bookkeeping is
    # unchanged by #1135.
    assert fake_self._keyframes_rejected == 4


def test_guard_frame_with_stale_pose_is_not_counted_as_holdout_rejection():
    """Guards are never evaluated (no n_test consequence), so a stale-pose
    guard drop must not inflate the holdout-specific counter -- only
    genuinely reserved holdouts do."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(4)])
    fake_self._holdout_indices = set()
    fake_self._guard_indices = {1}
    fake_self._frames[1]._odom_valid = False

    _ingest_all(fake_self)

    assert fake_self._stale_pose_holdout_rejected == 0
    assert fake_self._keyframes_rejected == 1


def test_ordinary_odom_invalid_rejection_unaffected_by_1135():
    """A plain training-candidate frame with a stale pose is rejected the
    same way it always was -- #1135 only adds attribution for the holdout
    case, it changes no other behavior of this guard."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(3)])
    fake_self._holdout_indices = set()
    fake_self._guard_indices = set()
    fake_self._frames[1]._odom_valid = False

    results = _ingest_all(fake_self)

    assert results[1] == (None, fake_self._frames[1], False)
    assert fake_self._stale_pose_holdout_rejected == 0
    assert fake_self._keyframes_rejected == 1
    assert fake_self.train_cameras != []  # frames 0 and 2 still admitted normally


def test_holdout_status_computed_before_odom_check_does_not_regress_1125():
    """#1135 moved `holdout_status()` earlier in `ingest_next_frame` (above
    the odom-validity guard, from below it) so the guard can attribute a
    stale-pose holdout rejection. This must not disturb #1125's own
    contract: a clean (odom-valid) holdout under a permanent integrity veto
    still always becomes a real test camera."""
    from test_issue806_integrity_observer_ordering import _REINIT_REJECT_STATS

    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(5)])
    fake_self._holdout_indices = {2}
    fake_self._guard_indices = set()

    def integrity_observer_fn(frame):
        return dict(_REINIT_REJECT_STATS) if frame.index >= 1 else None

    _ingest_all(fake_self, integrity_observer_fn=integrity_observer_fn)

    assert len(fake_self._test_cameras) == 1
    assert fake_self._test_cameras[0]._streaming_frame.index == 2
    assert fake_self._stale_pose_holdout_rejected == 0


# ---------------------------------------------------------------------------
# compute_trajectory_coverage() attribution surfacing
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(
    scene_source="tum://fr3",
    n_reg=100,
    frames_admitted=50,
    partition_psnr={f"p{i}": 20.0 for i in range(4)},
    k=4,
)


def test_trajectory_coverage_reports_zero_by_default():
    cov = compute_trajectory_coverage(**_BASE_KWARGS)
    assert cov["stale_pose_holdout_rejected"] == 0
    assert cov["stale_pose_holdout_rejected_detected"] is False
    assert cov["stale_pose_holdout_reason"] is None


def test_trajectory_coverage_surfaces_nonzero_rejection_count():
    cov = compute_trajectory_coverage(stale_pose_holdout_rejected=3, **_BASE_KWARGS)
    assert cov["stale_pose_holdout_rejected"] == 3
    assert cov["stale_pose_holdout_rejected_detected"] is True
    assert cov["stale_pose_holdout_reason"] is not None
    assert "3 reserved holdout frame(s)" in cov["stale_pose_holdout_reason"]


def test_trajectory_coverage_rejection_count_does_not_invalidate_mapping_result():
    """Option B's deliberate choice (issue #1135): a stale-pose holdout
    shortfall is attributed but must NOT, by itself, flip
    mapping_result_valid -- the shrunken-but-correctly-posed holdout set is
    still a real, comparable eval (comparability across a differing n_test
    is already policed by the holdout_hash/realized_holdout_hash ladder in
    utils/comparison_report.py's compare_against_baseline, not by this
    flag)."""
    kwargs = dict(_BASE_KWARGS)
    kwargs["eval_valid"] = True
    kwargs["coverage_counters"] = {
        "coverage_ledger_version": 2,
        "resolved_source_frames": 100,
        "raw_registry_source_frames": 100,
        "source_denominator_provenance": "post_stride_frame_source",
        "source_frames_ingested_total": 100,
        "bootstrap_frames_consumed": 0,
        "reserved_holdout_frames": 0,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": 100,
        "frames_admitted": 50,
        "mapping_skipped_frames": 50,
        "unprocessed_frames": 0,
    }
    kwargs["frame_admission"] = "hybrid_keyframe"
    clean = compute_trajectory_coverage(stale_pose_holdout_rejected=0, **kwargs)
    with_rejects = compute_trajectory_coverage(stale_pose_holdout_rejected=5, **kwargs)

    assert clean["mapping_result_valid"] is True
    assert with_rejects["mapping_result_valid"] is True
    assert with_rejects["stale_pose_holdout_rejected_detected"] is True
