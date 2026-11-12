"""Tests for canonical contract types (Phase 1.5)."""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame, TrackingResult


def test_legacy_contracts_namespace_reexports_package_types():
    from contracts.types import Pose as LegacyPose

    assert LegacyPose is Pose


def test_slam_backend_registry_is_gone():
    """#1075: the SLAMBackend ABC and its name registry no longer exist.

    Asserted by import failure rather than by absence of a name, so a future
    change that reintroduces the module -- under either the packaged or the
    legacy top-level path -- fails here instead of silently restoring a second
    tracking hierarchy alongside ``TrackingProvider``.
    """
    with pytest.raises(ImportError):
        import eligsir.contracts.slam_backend  # noqa: F401
    with pytest.raises(ImportError):
        import contracts.slam_backend  # noqa: F401


class TestPose:
    """Tests for the Pose canonical type."""

    def test_identity(self):
        """Identity pose has c2w = eye(4)."""
        p = Pose.identity()
        assert p.c2w.shape == (4, 4)
        np.testing.assert_array_almost_equal(p.c2w, np.eye(4, dtype=np.float32))

    def test_w2c_inverse(self):
        """w2c is the inverse of c2w."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Pose.from_Rt(R, t)
        product = p.c2w @ p.w2c
        np.testing.assert_array_almost_equal(product, np.eye(4, dtype=np.float32), decimal=5)

    def test_compose(self):
        """Composed pose equals matrix product."""
        p1 = Pose.from_Rt(np.eye(3), np.array([1, 0, 0], dtype=np.float32))
        p2 = Pose.from_Rt(np.eye(3), np.array([0, 1, 0], dtype=np.float32))
        composed = p1.compose(p2)
        expected = p1.matrix @ p2.matrix
        np.testing.assert_array_almost_equal(composed.matrix, expected)

    def test_invalid_shape(self):
        """Non-4×4 matrix raises ValueError."""
        with pytest.raises(ValueError):
            Pose(np.eye(3))

    def test_translation_property(self):
        """translation returns c2w[:3, 3]."""
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        p = Pose.from_Rt(np.eye(3, dtype=np.float32), t)
        np.testing.assert_array_almost_equal(p.translation, t)


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics."""

    def test_defaults_fill_depth(self):
        """Depth intrinsics default to colour values."""
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        assert K.depth_fx == 500
        assert K.depth_cx == 320
        assert K.depth_width == 640

    def test_K_matrix_shape(self):
        """K property returns 3×3 matrix."""
        K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        assert K.K.shape == (3, 3)
        assert K.K[0, 0] == 500.0


class TestRGBDFrame:
    """Tests for RGBDFrame."""

    def test_shape_properties(self):
        """height / width properties match rgb array."""
        intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        rgb = np.zeros((480, 640, 3), dtype=np.float32)
        depth = np.zeros((480, 640), dtype=np.float32)
        frame = RGBDFrame(index=0, timestamp=0.0, rgb=rgb, depth=depth, intrinsics=intr)
        assert frame.height == 480
        assert frame.width == 640


class TestTrackingResult:
    """Tests for TrackingResult."""

    def test_to_tensorboard_dict_includes_confidence(self):
        """to_tensorboard_dict returns tracking/confidence."""
        result = TrackingResult(
            pose=Pose.identity(),
            confidence=0.75,
            valid=True,
            method="test",
        )
        d = result.to_tensorboard_dict()
        assert "tracking/confidence" in d
        assert abs(d["tracking/confidence"] - 0.75) < 1e-6

    def test_optional_fields_excluded_when_none(self):
        """None optional fields are excluded from the dict."""
        result = TrackingResult(
            pose=Pose.identity(),
            confidence=0.5,
            valid=True,
            method="test",
        )
        d = result.to_tensorboard_dict()
        assert "tracking/depth_rmse" not in d

    def test_optional_fields_included_when_set(self):
        """Set optional fields appear in the dict."""
        result = TrackingResult(
            pose=Pose.identity(),
            confidence=0.9,
            valid=True,
            method="test",
            depth_rmse=0.05,
            inlier_ratio=0.8,
        )
        d = result.to_tensorboard_dict()
        assert "tracking/depth_rmse" in d
        assert abs(d["tracking/depth_rmse"] - 0.05) < 1e-6
