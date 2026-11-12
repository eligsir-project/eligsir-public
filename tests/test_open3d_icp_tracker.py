"""CPU-only contract tests for the registry-free Open3D ICP tracking frontend.

Ported from the closed PR #1011's ``test_open3d_frontend_contracts.py``. The
pose-convention, acceptance-gate and telemetry tests carry over unchanged in
substance; the two ``SLAMBackend`` registry assertions were **dropped rather
than ported** because the registry is being deleted (#1073/#1075), and are
replaced by :func:`test_icp_tracker_does_not_touch_the_retired_slam_registry`,
which asserts the opposite property.
"""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame
from eligsir.tracking.open3d_icp import (
    Open3DICPTracker,
    _compose_current_c2w,
    _depth_for_pointcloud,
    _prediction_to_icp_init,
)


def _transform(*, tx: float = 0.0, ty: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    matrix[:3, 3] = [tx, ty, 0.0]
    return matrix


def _frame(index: int, pose: Pose | None = None) -> RGBDFrame:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.5, width=4, height=4)
    return RGBDFrame(
        index=index,
        timestamp=float(index),
        rgb=np.zeros((4, 4, 3), dtype=np.float32),
        depth=np.ones((4, 4), dtype=np.float32),
        intrinsics=intrinsics,
        pose=pose,
    )


# --------------------------------------------------------------------------
# Pose convention (the semantic contract that must survive the port)
# --------------------------------------------------------------------------


def test_icp_composes_current_to_previous_into_world_camera_pose() -> None:
    previous_c2w = _transform(tx=2.0, ty=-1.0, yaw_deg=30.0)
    current_to_previous = _transform(tx=0.25, yaw_deg=5.0)

    actual = _compose_current_c2w(previous_c2w, current_to_previous)

    np.testing.assert_allclose(actual, previous_c2w @ current_to_previous, atol=1e-6)


def test_prediction_is_converted_to_current_to_previous_icp_initialization() -> None:
    previous_c2w = _transform(tx=1.0, yaw_deg=20.0)
    predicted_c2w = previous_c2w @ _transform(tx=0.3, yaw_deg=-4.0)

    actual = _prediction_to_icp_init(previous_c2w, predicted_c2w)

    np.testing.assert_allclose(actual, _transform(tx=0.3, yaw_deg=-4.0), atol=1e-6)


def test_depth_filter_removes_nonfinite_and_out_of_range_values() -> None:
    depth = np.array([[np.nan, -1.0, 0.05, 0.1, 1.0, 8.0, 9.0]], dtype=np.float32)
    clean = _depth_for_pointcloud(depth, 0.1, 8.0)
    np.testing.assert_allclose(clean, [[0.0, 0.0, 0.0, 0.1, 1.0, 8.0, 0.0]])


# --------------------------------------------------------------------------
# Tracking behaviour and acceptance gates
# --------------------------------------------------------------------------


def test_icp_track_reports_real_pose_delta_before_advancing_state(monkeypatch) -> None:
    import eligsir.tracking.open3d_icp as icp_module

    tracker = Open3DICPTracker(min_fitness=0.3, max_inlier_rmse=0.1)
    tracker.initialize(_frame(0, Pose.identity()))
    relative = _transform(tx=0.25)

    monkeypatch.setattr(icp_module, "import_open3d", lambda: object())
    monkeypatch.setattr(tracker, "_run_icp", lambda *a, **k: (relative, 0.9, 0.01, 1000, 1000))

    result = tracker.track(_frame(1))

    assert result.valid
    assert result.confidence == pytest.approx(0.9)
    assert result.pose.translation[0] == pytest.approx(0.25)
    assert result.pose_delta_m == pytest.approx(0.25)
    assert result.raw_stats["estimator"] == "point_to_plane"
    assert result.method == "open3d_icp_live"


def test_icp_invalid_registration_does_not_advance_anchor(monkeypatch) -> None:
    import eligsir.tracking.open3d_icp as icp_module

    tracker = Open3DICPTracker()
    tracker.initialize(_frame(0, Pose(_transform(tx=1.0))))
    monkeypatch.setattr(icp_module, "import_open3d", lambda: object())
    monkeypatch.setattr(tracker, "_run_icp", lambda *a, **k: None)

    failed = tracker.track(_frame(1))
    assert not failed.valid
    assert failed.pose.translation[0] == pytest.approx(1.0)

    relative = _transform(tx=0.2)
    monkeypatch.setattr(tracker, "_run_icp", lambda *a, **k: (relative, 0.8, 0.01, 1000, 1000))
    recovered = tracker.track(_frame(2))
    assert recovered.pose.translation[0] == pytest.approx(1.2)


def test_icp_rejects_registration_failing_the_acceptance_gates(monkeypatch) -> None:
    """A converged-but-poor registration must be refused, not merely low-confidence."""
    import eligsir.tracking.open3d_icp as icp_module

    tracker = Open3DICPTracker(min_fitness=0.5, max_inlier_rmse=0.05)
    tracker.initialize(_frame(0, Pose.identity()))
    monkeypatch.setattr(icp_module, "import_open3d", lambda: object())

    # Fitness below min_fitness.
    monkeypatch.setattr(
        tracker, "_run_icp", lambda *a, **k: (_transform(tx=0.4), 0.1, 0.01, 1000, 1000)
    )
    assert not tracker.track(_frame(1)).valid

    # Inlier RMSE above max_inlier_rmse.
    monkeypatch.setattr(
        tracker, "_run_icp", lambda *a, **k: (_transform(tx=0.4), 0.9, 0.5, 1000, 1000)
    )
    assert not tracker.track(_frame(2)).valid

    # Anchor never advanced through either rejection.
    assert tracker._prev_c2w[0, 3] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Configuration validation (one ICP semantic contract, explicitly parameterised)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"estimator": "bogus"}, "estimator"),
        ({"engine": "bogus"}, "engine"),
        ({"depth_min": 5.0, "depth_max": 1.0}, "depth range"),
        ({"voxel_size": -1.0}, "geometry"),
        ({"max_iterations": 0}, "optimization"),
        ({"min_fitness": 2.0}, "acceptance"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        Open3DICPTracker(**kwargs)


def test_estimator_selection_is_recorded_in_provenance() -> None:
    stats = Open3DICPTracker(estimator="point_to_point")._provenance_stats()
    assert stats["estimator"] == "point_to_point"
    assert stats["engine_requested"] == "auto"


# --------------------------------------------------------------------------
# Architecture regression guard (#1011 / #1073 / #1075)
# --------------------------------------------------------------------------


def test_icp_tracker_does_not_touch_the_retired_slam_registry() -> None:
    """The port must not reintroduce the architecture PR #1011 was closed over.

    Replaces the closed PR's two registry-membership assertions with their
    inverse. The registry itself is now deleted (#1075), so the guard is
    stated two ways: the registry module must not come back, and this
    frontend's own source must stay free of registry coupling -- the source
    grep is what catches a *reintroduction* under a new name, which an
    import check alone would miss.
    """
    import eligsir.tracking.open3d_icp as icp_module

    with pytest.raises(ImportError):
        import eligsir.contracts.slam_backend  # noqa: F401

    source = __import__("pathlib").Path(icp_module.__file__).read_text()
    for banned in ("import SLAMBackend", "register_backend(", "@register_backend"):
        assert banned not in source, f"registry coupling reintroduced: {banned}"


def test_icp_tracker_satisfies_the_canonical_tracking_provider_contract() -> None:
    """#1075's positive half: the frontend is now bindable to the contract.

    ``Open3DICPTracker`` deliberately does not implement ``TrackingProvider``
    itself (it speaks ``RGBDFrame``/``TrackingResult``); the binding lives in
    ``FrontendTrackingProvider``. This asserts that wrapper really does satisfy
    the protocol, which is what makes "one TrackingProvider hierarchy" true.
    """
    from eligsir.contracts.tracking import TrackingProvider
    from eligsir.tracking.providers import FrontendTrackingProvider

    provider = FrontendTrackingProvider(
        Open3DICPTracker(), provider_name="open3d_icp_live"
    )
    assert isinstance(provider, TrackingProvider)
