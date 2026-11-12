"""Tests for the TrackingResult -> TrackingObservation conversion.

The conversion uses only landed contracts (#1074's ``TrackingObservation`` and
the retained ``TrackingResult``). It defines no provider protocol — see
``eligsir/tracking/observation.py`` for why that is deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import SourceIdentity, TrackingStatus
from eligsir.contracts.types import Pose, TrackingResult
from eligsir.tracking.observation import tracking_observation_from_result


def _source() -> SourceIdentity:
    return SourceIdentity(
        source_id="cam0",
        stream_epoch=0,
        source_ordinal=7,
        timestamp_ns=1_234_567_890,
    )


def _result(**overrides) -> TrackingResult:
    kwargs = {
        "pose": Pose.identity(),
        "confidence": 0.9,
        "valid": True,
        "method": "open3d_icp_live",
    }
    kwargs.update(overrides)
    return TrackingResult(**kwargs)


def test_valid_result_maps_to_valid_and_usable_observation() -> None:
    observation = tracking_observation_from_result(_result(), _source())

    assert observation.status is TrackingStatus.VALID
    assert observation.usable
    assert observation.provider == "open3d_icp_live"
    assert observation.confidence == pytest.approx(0.9)
    assert observation.source == _source()


def test_invalid_result_maps_to_invalid_never_degraded() -> None:
    """A refused registration must not read as usable downstream.

    ``TrackingObservation.usable`` is true for DEGRADED, so mapping a
    gate-rejected pose to DEGRADED would silently readmit it.
    """
    observation = tracking_observation_from_result(
        _result(valid=False, confidence=0.0), _source()
    )

    assert observation.status is TrackingStatus.INVALID
    assert observation.status is not TrackingStatus.DEGRADED
    assert not observation.usable


def test_frontend_telemetry_and_raw_stats_are_preserved_in_diagnostics() -> None:
    observation = tracking_observation_from_result(
        _result(
            inlier_ratio=0.81,
            depth_rmse=0.012,
            pose_delta_m=0.25,
            tracking_time_ms=13.5,
            raw_stats={"engine": "tensor", "physical_accelerator": "rocm"},
        ),
        _source(),
    )

    assert observation.diagnostics["inlier_ratio"] == pytest.approx(0.81)
    assert observation.diagnostics["depth_rmse"] == pytest.approx(0.012)
    assert observation.diagnostics["pose_delta_m"] == pytest.approx(0.25)
    assert observation.diagnostics["tracking_time_ms"] == pytest.approx(13.5)
    # ROCm provenance must survive the conversion: it is what keeps a benchmark
    # row from confusing Open3D's "CUDA:0" compatibility label with real CUDA.
    assert observation.diagnostics["physical_accelerator"] == "rocm"
    assert observation.diagnostics["engine"] == "tensor"


def test_diagnostics_are_read_only_on_the_observation() -> None:
    observation = tracking_observation_from_result(
        _result(raw_stats={"engine": "legacy"}), _source()
    )
    with pytest.raises(TypeError):
        observation.diagnostics["engine"] = "tensor"  # type: ignore[index]


def test_out_of_range_confidence_is_clamped_not_raised() -> None:
    assert tracking_observation_from_result(
        _result(confidence=1.5), _source()
    ).confidence == pytest.approx(1.0)
    assert tracking_observation_from_result(
        _result(confidence=-0.2), _source()
    ).confidence == pytest.approx(0.0)


@pytest.mark.parametrize(
    "raw, expected",
    [(float("nan"), 0.0), (float("inf"), 1.0), (float("-inf"), 0.0)],
)
def test_non_finite_confidence_is_clamped_into_range(raw, expected) -> None:
    """A degenerate fit reporting NaN/inf must not raise out of the conversion.

    ``TrackingObservation`` rejects confidence outside [0, 1], so an unclamped
    non-finite value would turn a recoverable telemetry quirk into an exception
    on the ingress path.
    """
    observation = tracking_observation_from_result(_result(confidence=raw), _source())
    assert observation.confidence == pytest.approx(expected)


def test_pose_is_carried_through_unmodified() -> None:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    observation = tracking_observation_from_result(_result(pose=Pose(matrix)), _source())

    np.testing.assert_allclose(observation.pose.matrix, matrix, atol=1e-6)
