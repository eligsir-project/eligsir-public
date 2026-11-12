"""Tests for PoseCorrector (Milestone 5)."""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.utils.pose_corrections import (
    CorrectionEvent,
    CorrectionMode,
    PoseCorrector,
    PoseCorrectorConfig,
)


class _FakeCamera:
    """Fake camera matching the real streaming ``Camera`` convention.

    ``cam.R == R_c2w`` and ``cam.T == t_w2c`` (see
    ``utils.rgbd_frames.c2w_to_camera_rt``). Constructing from ``c2w``
    therefore inverts the rotation block to derive ``R`` and the
    ``t_w2c = -R_c2w.T @ t_c2w`` translation.
    """

    def __init__(self, submap_id: int, c2w: np.ndarray) -> None:
        self._submap_id = submap_id
        self.c2w = c2w.astype(np.float32).copy()
        R_c2w = self.c2w[:3, :3]
        t_c2w = self.c2w[:3, 3]
        # Match the real convention: cam.R = R_c2w, cam.T = t_w2c.
        self.R = R_c2w.copy()
        self.T = (-R_c2w.T @ t_c2w).astype(np.float32)


class _FakeSubmapManager:
    """Fake manager matching :class:`utils.submap_manager.SubmapManager`.

    The real manager exposes ``get_submap_transform`` and
    ``apply_pose_graph_corrections`` (not the legacy
    ``get_transform``/``set_transform`` pair). The fake mirrors those
    so the production path is exercised.
    """

    def __init__(self, transforms: dict) -> None:
        # Store the supplied per-submap transforms (these are world→submap).
        self._submaps: dict = {
            sid: type("_State", (), {"T_world_submap": np.asarray(T, dtype=np.float32).copy()})()
            for sid, T in transforms.items()
        }

    def get_submap_transform(self, submap_id: int) -> np.ndarray:
        state = self._submaps.get(submap_id)
        if state is None:
            return np.eye(4, dtype=np.float32)
        return state.T_world_submap.copy()

    def apply_pose_graph_corrections(
        self,
        optimised_transforms: List[np.ndarray],
        gaussians=None,
    ) -> None:
        for idx, T_new in enumerate(optimised_transforms):
            if idx not in self._submaps:
                continue
            state = self._submaps[idx]
            T_old = state.T_world_submap
            state.T_world_submap = np.asarray(T_new, dtype=np.float32)
            if gaussians is not None and hasattr(gaussians, "apply_correction"):
                # Cheap-path fake: no Gaussian transform (gaussians=None
                # in the propagator). The fake apply_correction is a
                # no-op for the cheap path.
                delta = T_new @ np.linalg.inv(T_old)
                gaussians.apply_correction(delta)


class _FakeGaussians:
    def __init__(self) -> None:
        self.corrections_applied = 0

    def apply_correction(self, *args, **kwargs) -> int:
        self.corrections_applied += 1
        return 0


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _translation(tx: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    return m


class TestCorrectorEnabled:
    def test_disabled_returns_none(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=False))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        assert event is None
        assert pc.events == []

    def test_enabled_records_event(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0, 1],
            transforms=[_translation(0.1), _translation(0.2)],
            loop_edges=2,
        )
        assert event is not None
        assert event.submap_ids == [0, 1]
        assert event.n_corrected_cameras == 0
        assert event.n_corrected_gaussians == 0
        assert event.loop_edges == 2
        assert event.mode == CorrectionMode.MEDIUM
        assert len(pc.events) == 1

    def test_mismatched_lengths_raises(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True))
        with pytest.raises(ValueError):
            pc.propagate(
                train_cameras=[],
                submap_manager=None,
                gaussians=None,
                submap_ids=[0, 1],
                transforms=[_identity()],
            )


class TestSubmapTransformUpdate:
    def test_apply_pose_graph_corrections_called(self) -> None:
        """The cheap path must delegate to the real manager API
        ``apply_pose_graph_corrections(optimised_transforms, gaussians)``
        (not the legacy ``get_transform``/``set_transform`` pair)."""
        manager = _FakeSubmapManager({0: _identity(), 1: _identity()})
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, default_mode=CorrectionMode.CHEAP))
        pc.propagate(
            train_cameras=[],
            submap_manager=manager,
            gaussians=None,
            submap_ids=[0],
            transforms=[_translation(0.5)],
        )
        np.testing.assert_allclose(
            manager.get_submap_transform(0)[:3, 3], [0.5, 0.0, 0.0], atol=1e-6
        )


class TestCameraUpdate:
    def test_camera_c2w_updated_for_corrected_submap(self) -> None:
        manager = _FakeSubmapManager({0: _identity(), 1: _identity()})
        cam0_a = _FakeCamera(submap_id=0, c2w=_identity())
        cam1 = _FakeCamera(submap_id=1, c2w=_identity())
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, default_mode=CorrectionMode.CHEAP))
        event = pc.propagate(
            train_cameras=[cam0_a, cam1],
            submap_manager=manager,
            gaussians=None,
            submap_ids=[0],
            transforms=[_translation(0.3)],
        )
        # cam0_a belongs to submap 0 → updated.
        assert event.n_corrected_cameras == 1
        # Validate via the canonical c2w form, not cam.T (cam.T is t_w2c
        # in this codebase, not t_c2w — see utils.rgbd_frames).
        np.testing.assert_allclose(cam0_a.c2w[:3, 3], [0.3, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cam0_a.c2w[:3, :3], np.eye(3), atol=1e-6)
        # t_w2c should also be re-derived consistently from the new c2w.
        np.testing.assert_allclose(
            cam0_a.T, -cam0_a.c2w[:3, :3].T @ cam0_a.c2w[:3, 3], atol=1e-6
        )
        # cam1 belongs to submap 1 → untouched.
        np.testing.assert_allclose(cam1.c2w[:3, 3], [0.0, 0.0, 0.0], atol=1e-6)


class TestRateLimit:
    def test_min_dt_blocks_second_event(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, min_iter_dt_s=10.0))
        e1 = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        e2 = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[1],
            transforms=[_identity()],
        )
        assert e1 is not None
        assert e2 is None  # rate-limited

    def test_no_min_dt_allows_immediate(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, min_iter_dt_s=0.0))
        e1 = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        e2 = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[1],
            transforms=[_identity()],
        )
        assert e1 is not None
        assert e2 is not None


class TestMaxSubmapsPerEvent:
    def test_cap_truncates(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, max_submaps_per_event=2))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0, 1, 2, 3],
            transforms=[_identity()] * 4,
        )
        assert event is not None
        assert event.submap_ids == [0, 1]


class TestConfigUpdate:
    def test_update_enabled_live(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=False))
        assert pc.enabled is False
        pc.update_config(enabled=True, default_mode=CorrectionMode.EXPENSIVE)
        assert pc.enabled is True
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        assert event is not None
        assert event.mode == CorrectionMode.EXPENSIVE

    def test_default_on_for_live_backend(self) -> None:
        """When enabled=True and a non-trivial submap is corrected,
        the event.mode reflects the configured default."""
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True, default_mode=CorrectionMode.CHEAP))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        assert event is not None
        assert event.mode == CorrectionMode.CHEAP


class TestEmptySubmaps:
    def test_empty_submaps_returns_none(self) -> None:
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[],
            transforms=[],
        )
        assert event is None


class TestTimestampConvention:
    """``CorrectionEvent.timestamp`` is wall-clock, not monotonic."""

    def test_timestamp_is_wall_clock(self) -> None:
        import time as _t
        before = _t.time()
        pc = PoseCorrector(PoseCorrectorConfig(enabled=True))
        event = pc.propagate(
            train_cameras=[],
            submap_manager=None,
            gaussians=None,
            submap_ids=[0],
            transforms=[_identity()],
        )
        after = _t.time()
        assert event is not None
        # Wall-clock seconds since the epoch — must lie in [before, after].
        assert before <= event.timestamp <= after
        # The monotonic counter is stored separately for rate-limiting.
        assert event.monotonic_t > 0.0


class TestPoseGraphEdgeCount:
    """The caller must use the public ``get_edge_count`` accessor."""

    def test_get_edge_count_loop_only(self) -> None:
        from loop_closure.pose_graph import PoseGraph
        pg = PoseGraph()
        # Add two odometry edges and one loop edge.
        pg.add_odometry_edge(0, 1, np.eye(4), information=np.eye(6))
        pg.add_odometry_edge(1, 2, np.eye(4), information=np.eye(6))
        pg.add_loop_edge(
            src=0, dst=2, T_src_dst=np.eye(4),
            inlier_count=10, rmse=0.05,
        )
        assert pg.get_edge_count(loop_only=False) == 3
        assert pg.get_edge_count(loop_only=True) == 1


class TestCameraRTTConvention:
    """The R/T round-trip must respect the 3DGS convention."""

    def test_R_T_reflect_c2w_when_round_tripped(self) -> None:
        """A camera built from an arbitrary c2w must end up with
        ``cam.R == c2w[:3, :3]`` and ``cam.T == -cam.R.T @ c2w[:3, 3]``."""
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        # Small rotation
        theta = 0.1
        c2w[:3, :3] = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cam = _FakeCamera(submap_id=0, c2w=c2w)
        np.testing.assert_allclose(cam.R, c2w[:3, :3], atol=1e-6)
        np.testing.assert_allclose(cam.T, -cam.R.T @ c2w[:3, 3], atol=1e-6)
        # The fake deliberately does NOT set cam.T == c2w[:3, 3] (that
        # would be the wrong convention).
        assert not np.allclose(cam.T, c2w[:3, 3])
