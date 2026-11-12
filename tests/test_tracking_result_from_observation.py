"""``TrackingObservation`` -> ``TrackingResult`` back-conversion (#1075, task P5).

``TrackingProvider.observe()`` returns a canonical ``TrackingObservation`` while
``MappingScheduler.decide`` / ``OnlineController.tick`` / ``ViewerBase.log_tracking``
still consume a ``TrackingResult``. These tests pin the semantics of the adapter
between them -- in particular the two decisions that are safety-relevant rather
than mechanical:

* a non-usable observation must NOT come back as ``None`` (every consumer reads
  ``None`` as "precomputed pose, confidence 1.0, valid") and must NOT get a
  fabricated identity pose;
* the collapse of five ``TrackingStatus`` values onto one boolean must stay
  recoverable.
"""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.types import Pose, TrackingResult
from eligsir.mapping.scheduler.mapping_scheduler import MappingScheduler
from eligsir.tracking.observation import (
    TRACKING_STATUS_STAT_KEY,
    tracking_observation_from_result,
    tracking_result_from_observation,
)

USABLE_STATUSES = (TrackingStatus.VALID, TrackingStatus.DEGRADED)
UNUSABLE_STATUSES = (
    TrackingStatus.INITIALIZING,
    TrackingStatus.LOST,
    TrackingStatus.INVALID,
)


def _scheduler() -> MappingScheduler:
    return MappingScheduler(steps_per_frame=50, n_frames=100, iterations=10000)


def _source(ordinal: int = 7) -> SourceIdentity:
    return SourceIdentity(
        source_id="camera_0",
        stream_epoch=1,
        source_ordinal=ordinal,
        timestamp_ns=1_700_000_000_123_456_789,
    )


def _pose(tx: float = 1.5) -> Pose:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = tx
    return Pose(matrix)


def _observation(**overrides) -> TrackingObservation:
    kwargs = dict(
        source=_source(),
        pose=_pose(),
        status=TrackingStatus.VALID,
        provider="open3d_icp",
        confidence=0.75,
    )
    kwargs.update(overrides)
    return TrackingObservation(**kwargs)


# --------------------------------------------------------------------------
# Successful track
# --------------------------------------------------------------------------


def test_valid_observation_maps_to_valid_result():
    observation = _observation()
    result = tracking_result_from_observation(observation)

    assert isinstance(result, TrackingResult)
    assert result.valid is True
    assert result.confidence == pytest.approx(0.75)
    assert result.method == "open3d_icp"
    assert result.pose is observation.pose
    assert result.raw_stats[TRACKING_STATUS_STAT_KEY] == "valid"


def test_telemetry_diagnostics_are_promoted_to_typed_fields():
    observation = _observation(
        diagnostics={
            "depth_rmse": 0.012,
            "rgb_residual": 3.5,
            "inlier_ratio": 0.91,
            "pose_delta_m": 0.04,
            "pose_delta_deg": 1.25,
            "render_alpha_coverage": 0.88,
            "tracking_time_ms": 12.0,
            "icp_iterations": 14,
        }
    )
    result = tracking_result_from_observation(observation)

    assert result.depth_rmse == pytest.approx(0.012)
    assert result.rgb_residual == pytest.approx(3.5)
    assert result.inlier_ratio == pytest.approx(0.91)
    assert result.pose_delta_m == pytest.approx(0.04)
    assert result.pose_delta_deg == pytest.approx(1.25)
    assert result.render_alpha_coverage == pytest.approx(0.88)
    assert result.tracking_time_ms == pytest.approx(12.0)
    # Non-telemetry provenance survives, telemetry keys do not linger.
    assert result.raw_stats["icp_iterations"] == 14
    assert "depth_rmse" not in result.raw_stats


def test_non_numeric_telemetry_value_stays_in_raw_stats():
    """A telemetry quirk must not crash the conversion or be silently dropped."""
    observation = _observation(diagnostics={"depth_rmse": "unavailable"})
    result = tracking_result_from_observation(observation)

    assert result.depth_rmse is None
    assert result.raw_stats["depth_rmse"] == "unavailable"


def test_diagnostics_mutation_does_not_leak_into_raw_stats():
    diagnostics = {"icp_iterations": 3}
    observation = _observation(diagnostics=diagnostics)
    result = tracking_result_from_observation(observation)

    result.raw_stats["icp_iterations"] = 999
    assert observation.diagnostics["icp_iterations"] == 3


# --------------------------------------------------------------------------
# Lost / invalid / initializing track
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", UNUSABLE_STATUSES)
def test_unusable_observation_is_a_result_not_none(status):
    """The adapter is total: a lost track must still produce a result object.

    Returning ``None`` would be read by every consumer as "no tracking backend,
    assume confidence 1.0 and valid", laundering a lost track into a trusted one.
    """
    observation = _observation(pose=None, status=status, confidence=None)
    result = tracking_result_from_observation(observation)

    assert result is not None
    assert result.valid is False
    assert result.confidence == 0.0
    assert result.raw_stats[TRACKING_STATUS_STAT_KEY] == status.value


@pytest.mark.parametrize("status", UNUSABLE_STATUSES)
def test_unusable_observation_never_fabricates_a_pose(status):
    observation = _observation(pose=None, status=status)
    result = tracking_result_from_observation(observation)

    assert result.pose is None, "a pose-less observation must not gain an identity pose"


def test_unusable_result_is_still_tensorboard_loggable():
    """``to_tensorboard_dict`` is the one consumer that touches no ``.pose``."""
    result = tracking_result_from_observation(
        _observation(pose=None, status=TrackingStatus.LOST, confidence=None)
    )
    scalars = result.to_tensorboard_dict()

    assert scalars["tracking/valid"] == 0.0
    assert scalars["tracking/confidence"] == 0.0


def test_invalid_observation_may_still_carry_a_pose():
    """A frontend that rejected its own estimate keeps it as diagnostics."""
    observation = _observation(status=TrackingStatus.INVALID, confidence=0.1)
    result = tracking_result_from_observation(observation)

    assert result.valid is False
    assert result.pose is observation.pose
    assert result.confidence == pytest.approx(0.1)


# --------------------------------------------------------------------------
# Low-confidence / degraded track
# --------------------------------------------------------------------------


def test_degraded_observation_is_valid_but_status_is_recoverable():
    observation = _observation(status=TrackingStatus.DEGRADED, confidence=0.4)
    result = tracking_result_from_observation(observation)

    assert result.valid is True
    assert result.confidence == pytest.approx(0.4)
    # ``valid`` alone cannot distinguish DEGRADED from VALID; raw_stats can.
    assert result.raw_stats[TRACKING_STATUS_STAT_KEY] == "degraded"


def test_degraded_without_confidence_defaults_fail_closed():
    """Unquantified "reduced but usable" must not become a fabricated number."""
    observation = _observation(status=TrackingStatus.DEGRADED, confidence=None)
    result = tracking_result_from_observation(observation)

    assert result.valid is True
    assert result.confidence == 0.0


def test_valid_without_confidence_defaults_to_one():
    """Matches the existing ``tracking_result is None`` -> confidence 1.0 convention."""
    observation = _observation(confidence=None)
    result = tracking_result_from_observation(observation)

    assert result.confidence == 1.0


@pytest.mark.parametrize("status", tuple(TrackingStatus))
def test_every_status_is_convertible(status):
    """Totality: no ``TrackingStatus`` member raises or is unmapped."""
    pose = None if status in UNUSABLE_STATUSES else _pose()
    result = tracking_result_from_observation(
        _observation(pose=pose, status=status, confidence=None)
    )

    assert result.valid is (status in USABLE_STATUSES)
    assert 0.0 <= result.confidence <= 1.0
    assert result.raw_stats[TRACKING_STATUS_STAT_KEY] == status.value


# --------------------------------------------------------------------------
# Round-trip fidelity
# --------------------------------------------------------------------------


def test_round_trip_observation_result_observation_preserves_representable_fields():
    original = _observation(
        confidence=0.6,
        diagnostics={"depth_rmse": 0.02, "icp_iterations": 9},
    )
    round_tripped = tracking_observation_from_result(
        tracking_result_from_observation(original), original.source
    )

    assert round_tripped.source == original.source
    assert round_tripped.pose is original.pose
    assert round_tripped.status == original.status
    assert round_tripped.provider == original.provider
    assert round_tripped.confidence == pytest.approx(0.6)
    assert round_tripped.diagnostics["depth_rmse"] == pytest.approx(0.02)
    assert round_tripped.diagnostics["icp_iterations"] == 9
    # The reserved key is consumed by the forward conversion, not left behind as
    # a second copy of ``status`` free to drift from it on the next hop.
    assert TRACKING_STATUS_STAT_KEY not in round_tripped.diagnostics


@pytest.mark.parametrize("status", tuple(TrackingStatus))
def test_round_trip_preserves_every_status(status):
    """The five-status -> boolean collapse must survive a round trip.

    Without the reserved ``raw_stats`` key this is where LOST silently becomes
    INVALID, INITIALIZING becomes INVALID and DEGRADED becomes VALID.
    """
    pose = None if status in UNUSABLE_STATUSES else _pose()
    original = _observation(pose=pose, status=status, confidence=0.5)
    round_tripped = tracking_observation_from_result(
        tracking_result_from_observation(original), original.source
    )

    assert round_tripped.status == status
    assert round_tripped.pose is original.pose
    assert round_tripped.confidence == pytest.approx(0.5)


def test_round_trip_preserves_invalid_with_a_pose():
    """INVALID + a pose: the status key must not be refused by its own guards."""
    original = _observation(status=TrackingStatus.INVALID, confidence=0.2)
    round_tripped = tracking_observation_from_result(
        tracking_result_from_observation(original), original.source
    )

    assert round_tripped.status is TrackingStatus.INVALID
    assert round_tripped.pose is original.pose


def test_absent_confidence_does_not_survive_a_round_trip():
    """Documented irreversibility: "no confidence reported" becomes a number."""
    original = _observation(confidence=None)
    round_tripped = tracking_observation_from_result(
        tracking_result_from_observation(original), original.source
    )

    assert round_tripped.confidence == 1.0, "materialized default, not the original None"


@pytest.mark.parametrize(
    "raw_status", ["not_a_status", None, 3, TrackingStatus.VALID.value]
)
def test_untrustworthy_status_key_never_overrules_the_boolean(raw_status):
    """A stale/garbage key must not manufacture a status ``valid`` contradicts."""
    result = TrackingResult(
        pose=None,
        confidence=0.0,
        valid=False,
        method="external",
        raw_stats={TRACKING_STATUS_STAT_KEY: raw_status},
    )
    observation = tracking_observation_from_result(result, _source())

    # Falls back to the fail-closed boolean mapping; notably a "valid" key on a
    # valid=False result is refused rather than resurrecting a rejected pose.
    assert observation.status is TrackingStatus.INVALID
    assert observation.usable is False


def test_round_trip_result_observation_result_preserves_representable_fields():
    original = TrackingResult(
        pose=_pose(2.0),
        confidence=0.42,
        valid=True,
        method="open3d_odometry",
        depth_rmse=0.03,
        tracking_time_ms=8.0,
        raw_stats={"fitness": 0.77},
    )
    round_tripped = tracking_result_from_observation(
        tracking_observation_from_result(original, _source())
    )

    assert round_tripped.pose is original.pose
    assert round_tripped.valid is True
    assert round_tripped.confidence == pytest.approx(0.42)
    assert round_tripped.method == "open3d_odometry"
    assert round_tripped.depth_rmse == pytest.approx(0.03)
    assert round_tripped.tracking_time_ms == pytest.approx(8.0)
    assert round_tripped.raw_stats["fitness"] == pytest.approx(0.77)


def test_source_identity_is_the_documented_loss():
    """``TrackingResult`` has no identity field; the caller must keep the source."""
    result = tracking_result_from_observation(_observation())

    assert not hasattr(result, "source")
    assert not any(
        isinstance(value, SourceIdentity) for value in result.raw_stats.values()
    )


def test_covariance_is_the_documented_loss():
    covariance = np.eye(6, dtype=np.float64) * 0.01
    result = tracking_result_from_observation(_observation(covariance=covariance))

    assert not any(
        isinstance(value, np.ndarray) for value in result.raw_stats.values()
    ), "a 6x6 array must not be smuggled into scalar-shaped raw_stats"


# --------------------------------------------------------------------------
# The consumer this conversion exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", UNUSABLE_STATUSES)
def test_mapping_scheduler_refuses_mapping_for_unusable_observation(status):
    """End-to-end: the reason ``None`` is not an acceptable return value."""
    scheduler = _scheduler()
    converted = tracking_result_from_observation(
        _observation(pose=None, status=status, confidence=None)
    )

    decision = scheduler.decide(converted, keyframe_indices=[0], replay_indices=[])
    assert decision.do_mapping is False
    assert decision.allow_insertion is False
    assert decision.allow_growth is False

    # Contrast: the None that a naive "no result" conversion would produce.
    permissive = scheduler.decide(None, keyframe_indices=[0], replay_indices=[])
    assert permissive.do_mapping is True


def test_mapping_scheduler_maps_normally_for_confident_observation():
    scheduler = _scheduler()
    converted = tracking_result_from_observation(_observation(confidence=1.0))

    decision = scheduler.decide(converted, keyframe_indices=[0], replay_indices=[])
    assert decision.do_mapping is True
