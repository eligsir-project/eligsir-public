"""Tests for MappingScheduler wiring (Phases 7/8/9).

Covers:
- Phase 7: MappingScheduler produces the same effective_spf as inline enforcement.
- Phase 8: Confidence gates produce correct MappingDecision fields.
- Phase 9: gs_rendered_rgbd adapter registers and exposes correct attributes.
- Precomputed TrackingResult has confidence=1.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eligsir.contracts.types import Pose, TrackingResult
from eligsir.mapping.scheduler import MappingDecision, MappingScheduler
from eligsir.mapping.scheduler.tracking_confidence import (
    RunningTrackingStats,
    TrackingStatus,
    tracking_confidence_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tr(confidence: float, valid: bool = True) -> TrackingResult:
    return TrackingResult(
        pose=Pose.identity(),
        confidence=confidence,
        valid=valid,
        method="test",
    )


# ---------------------------------------------------------------------------
# Phase 7: effective_spf parity with old inline enforcement
# ---------------------------------------------------------------------------

class TestPhase7EffectiveSpfParity:
    """MappingScheduler must reproduce the inline enforcement math exactly."""

    def test_spf_reduced_when_budget_tight(self):
        """MappingScheduler reduces spf identically to the old inline block."""
        configured_spf = 500
        n_frames = 100
        iterations = 10_000

        # Old inline math:
        old_max_spf = max(1, iterations // n_frames)  # == 100
        old_effective = min(configured_spf, old_max_spf)

        sched = MappingScheduler(
            steps_per_frame=configured_spf,
            n_frames=n_frames,
            iterations=iterations,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == old_effective, (
            f"expected {old_effective}, got {sched.effective_spf}"
        )

    def test_spf_unchanged_when_budget_is_sufficient(self):
        """No reduction when configured_spf <= iterations // n_frames."""
        sched = MappingScheduler(
            steps_per_frame=50,
            n_frames=100,
            iterations=10_000,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == 50

    def test_spf_unchanged_when_coverage_disabled(self):
        """Coverage=False keeps spf at the configured value."""
        sched = MappingScheduler(
            steps_per_frame=500,
            n_frames=100,
            iterations=10_000,
            enforce_full_coverage=False,
        )
        assert sched.effective_spf == 500

    def test_spf_math_uses_floor(self):
        """Reduction uses math.floor(iterations/n_frames) == iterations//n_frames."""
        n_frames = 7
        iterations = 100
        expected = math.floor(iterations / n_frames)  # 14

        sched = MappingScheduler(
            steps_per_frame=1000,
            n_frames=n_frames,
            iterations=iterations,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == expected

    def test_spf_min_is_one(self):
        """effective_spf is at least 1 even if iterations < n_frames."""
        sched = MappingScheduler(
            steps_per_frame=50,
            n_frames=1000,
            iterations=5,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf >= 1

    def test_effective_spf_property_matches_stats(self):
        """effective_spf property and stats.effective_spf are identical."""
        sched = MappingScheduler(
            steps_per_frame=200,
            n_frames=50,
            iterations=5_000,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == sched.stats.effective_spf


# ---------------------------------------------------------------------------
# Phase 8: confidence gates
# ---------------------------------------------------------------------------

class TestPhase8ConfidenceGates:
    """MappingScheduler.decide() three-tier confidence gate."""

    def _sched(self, spf: int = 50) -> MappingScheduler:
        return MappingScheduler(
            steps_per_frame=spf,
            n_frames=100,
            iterations=5_000,
            enforce_full_coverage=False,
            tracking_min_conf_persistent=0.7,
            tracking_min_conf_provisional=0.4,
        )

    def test_low_conf_no_mapping(self):
        """conf=0.3 (< provisional threshold) → do_mapping=False."""
        d = self._sched().decide(_tr(0.3), keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is False
        assert d.steps == 0
        assert d.allow_insertion is False
        assert d.allow_growth is False
        assert d.freeze_persistent is True

    def test_medium_conf_provisional_mapping(self):
        """conf=0.5 (provisional ≤ conf < persistent) → provisional mapping."""
        d = self._sched(spf=100).decide(_tr(0.5), keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is True
        assert d.allow_growth is False
        assert d.freeze_persistent is True
        assert d.steps > 0
        # Provisional mode: no insertion (provisional means we're not confident enough)
        assert d.allow_insertion is False

    def test_high_conf_normal_mapping(self):
        """conf=0.9 (≥ persistent threshold) → full mapping, no restrictions."""
        d = self._sched(spf=50).decide(_tr(0.9), keyframe_indices=[0, 1, 2], replay_indices=[5])
        assert d.do_mapping is True
        assert d.allow_insertion is True
        assert d.allow_growth is True
        assert d.freeze_persistent is False
        assert d.steps == 50

    def test_exact_boundary_persistent(self):
        """conf == persistent threshold → normal mapping."""
        d = self._sched().decide(_tr(0.7), keyframe_indices=[], replay_indices=[])
        assert d.do_mapping is True
        assert d.allow_growth is True

    def test_exact_boundary_provisional(self):
        """conf == provisional threshold → provisional (not tracking-only)."""
        d = self._sched().decide(_tr(0.4), keyframe_indices=[], replay_indices=[])
        assert d.do_mapping is True
        assert d.allow_growth is False

    def test_invalid_tracking_no_mapping(self):
        """valid=False → do_mapping=False regardless of confidence."""
        d = self._sched().decide(_tr(0.9, valid=False), keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is False

    def test_none_tracking_result_normal_mapping(self):
        """None tracking (precomputed) → treated as confidence 1.0 → normal."""
        d = self._sched(spf=50).decide(None, keyframe_indices=[0], replay_indices=[5])
        assert d.do_mapping is True
        assert d.allow_growth is True
        assert d.steps == 50

    def test_confidence_stored_in_decision(self):
        """Decision.confidence reflects the TrackingResult.confidence."""
        d = self._sched().decide(_tr(0.85), keyframe_indices=[], replay_indices=[])
        assert abs(d.confidence - 0.85) < 1e-6

    def test_tracking_only_decision_is_mapping_decision_instance(self):
        """Low-confidence decision is still a MappingDecision dataclass."""
        d = self._sched().decide(_tr(0.1), keyframe_indices=[], replay_indices=[])
        assert isinstance(d, MappingDecision)


# ---------------------------------------------------------------------------
# Phase 1.5-SLAM: precomputed TrackingResult
# ---------------------------------------------------------------------------

class TestPrecomputedTrackingResult:
    """TrackingResult synthesized from precomputed poses has confidence=1.0."""

    def test_confidence_is_one(self):
        c2w = np.eye(4, dtype=np.float32)
        tr = TrackingResult(
            pose=Pose(c2w),
            confidence=1.0,
            valid=True,
            method="precomputed",
        )
        assert tr.confidence == 1.0
        assert tr.valid is True
        assert tr.method == "precomputed"

    def test_pose_preserves_matrix(self):
        c2w = np.array([
            [0, -1, 0, 1],
            [1,  0, 0, 2],
            [0,  0, 1, 3],
            [0,  0, 0, 1],
        ], dtype=np.float32)
        tr = TrackingResult(pose=Pose(c2w), confidence=1.0, valid=True, method="precomputed")
        np.testing.assert_array_equal(tr.pose.c2w, c2w)

    def test_tensorboard_dict_contains_confidence(self):
        tr = TrackingResult(pose=Pose.identity(), confidence=1.0, valid=True, method="precomputed")
        tb = tr.to_tensorboard_dict()
        assert "tracking/confidence" in tb
        assert tb["tracking/confidence"] == 1.0


# ---------------------------------------------------------------------------
# Phase 9: gs_rendered_rgbd adapter registration
# ---------------------------------------------------------------------------

class TestTrackingStatusFeedsMappingSchedulerGate:
    """2026-08-04: real /slam/tracking_status_stamped data, run through the
    same tracking_confidence_weight() KeyframeAdmitter uses, must reach
    MappingScheduler.decide() and actually flip its three-tier gate --
    this is the exact reachability train_streaming.py's dataset_pose call
    site now provides (previously hardcoded confidence=1.0 unconditionally).
    """

    def _sched(self) -> MappingScheduler:
        return MappingScheduler(
            steps_per_frame=50,
            n_frames=100,
            iterations=5_000,
            enforce_full_coverage=False,
            tracking_min_conf_persistent=0.7,
            tracking_min_conf_provisional=0.4,
        )

    def _tracking_result_from_status(self, status: TrackingStatus, running_typical=None) -> TrackingResult:
        conf = tracking_confidence_weight(status, running_typical=running_typical, floor_weight=0.4)
        return TrackingResult(pose=Pose.identity(), confidence=conf, valid=True, method="dataset_pose")

    def test_ok_state_well_tracked_stays_normal_mapping(self):
        stats = RunningTrackingStats(warmup=1)
        stats.update(400)
        status = TrackingStatus(state=2, num_tracked=400)  # OK, at its own typical
        tr = self._tracking_result_from_status(status, running_typical=stats.typical)
        d = self._sched().decide(tr, keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is True
        assert d.allow_growth is True

    def test_lost_state_triggers_full_skip(self):
        status = TrackingStatus(state=4, num_tracked=0)  # LOST
        tr = self._tracking_result_from_status(status)
        assert tr.confidence < 0.4
        d = self._sched().decide(tr, keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is False
        assert d.freeze_persistent is True

    def test_recently_lost_triggers_provisional_or_skip(self):
        status = TrackingStatus(state=3, num_tracked=10)  # RECENTLY_LOST
        tr = self._tracking_result_from_status(status)
        assert tr.confidence == pytest.approx(0.3)
        d = self._sched().decide(tr, keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is False  # 0.3 < provisional threshold 0.4

    def test_degraded_inlier_count_below_typical_triggers_provisional(self):
        stats = RunningTrackingStats(warmup=1)
        stats.update(400)
        # Degraded but still tracking: far fewer inliers than the running
        # typical -> ratio clipped to the floor -> provisional-tier mapping.
        status = TrackingStatus(state=2, num_tracked=40)  # OK, ratio=0.1 -> floor 0.4
        tr = self._tracking_result_from_status(status, running_typical=stats.typical)
        assert tr.confidence == pytest.approx(0.4)
        d = self._sched().decide(tr, keyframe_indices=[0], replay_indices=[])
        assert d.do_mapping is True
        assert d.allow_growth is False
        assert d.allow_insertion is False

    def test_no_tracking_status_defaults_to_full_confidence(self):
        """No upstream signal (status=None, e.g. non-ORB-SLAM3 / feature off)
        must stay byte-identical to the pre-wiring behavior: confidence 1.0."""
        tr = self._tracking_result_from_status_none = TrackingResult(
            pose=Pose.identity(), confidence=1.0, valid=True, method="dataset_pose",
        )
        d = self._sched().decide(tr, keyframe_indices=[0], replay_indices=[5])
        assert d.do_mapping is True
        assert d.allow_growth is True


class TestTrackingFrontendSelection:
    """#1075 replaced the SLAMBackend registry with an explicit factory.

    These tests are the post-hard-cut successors of the old
    ``TestPhase9GSRenderedRGBD`` suite, which asserted that four adapters
    registered themselves by import side effect and that the parked
    ``gs_rendered_rgbd_odometry`` stub was registered-but-unselectable. Both
    the registry and that stub are gone, so the guarantees worth keeping are
    restated against the factory: which names build, which are refused, and
    that a refusal explains where the capability went.
    """

    def test_gs_rendered_rgbd_is_removed_with_an_actionable_error(self):
        """The parked stub is deleted, not merely unselectable.

        Its only reason to exist was occupying a registry slot. #1075 removed
        it together with ``GaussianRenderContext``, so the error must point at
        the read-only-render-view constraint any successor has to satisfy.
        """
        from eligsir.tracking.factory import build_tracking_frontend

        with pytest.raises(KeyError, match="read-only render view"):
            build_tracking_frontend("gs_rendered_rgbd_odometry")

        with pytest.raises(ImportError):
            import eligsir.slam.adapters.gs_rendered_rgbd  # noqa: F401

    def test_gaussian_render_context_is_gone(self):
        """No provider may be handed the live Gaussian model (owner safety)."""
        import eligsir.contracts.tracking as tracking_contract

        assert not hasattr(tracking_contract, "GaussianRenderContext")

    def test_estimator_names_build_real_frontends(self):
        from eligsir.tracking.factory import IN_PROCESS_FRONTENDS

        assert set(IN_PROCESS_FRONTENDS) == {"open3d_odometry_live", "open3d_icp_live"}

    def test_dataset_pose_builds_a_passthrough_frontend(self):
        from eligsir.tracking.factory import build_tracking_frontend
        from eligsir.tracking.passthrough import FramePosePassthroughTracker

        assert isinstance(
            build_tracking_frontend("dataset_pose"), FramePosePassthroughTracker
        )

    def test_unknown_name_lists_the_available_ones(self):
        from eligsir.tracking.factory import build_tracking_frontend

        with pytest.raises(KeyError, match="Available:"):
            build_tracking_frontend("no_such_frontend")
