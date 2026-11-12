from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch", reason="canonical bootstrap returns the torch-backed BasicPointCloud")

from eligsir.contracts.runtime import BufferRef, MappingFrame, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.runtime.streaming_observation import StreamingObservation


def _observation(
    *,
    ordinal: int = 0,
    depth_value: float = 2.0,
    depth_array: np.ndarray | None = None,
    x: float = 0.0,
):
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    depth = (
        np.full((4, 6), depth_value, dtype=np.float32)
        if depth_array is None
        else np.asarray(depth_array, dtype=np.float32)
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = float(x)
    frame = MappingFrame(
        source=SourceIdentity("canonical", 0, ordinal, 1_000_000_000 + ordinal),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=CameraIntrinsics(
            fx=8.0, fy=8.0, cx=3.0, cy=2.0, width=6, height=4,
            intrinsics_source="shared_rgb_depth_pixel_geometry",
        ),
        device="cpu",
        pose=Pose(c2w),
    )
    return StreamingObservation.from_mapping_frame(frame)


def test_canonical_bootstrap_builds_only_from_validated_mapping_frame_arrays():
    from eligsir.mapping.canonical_bootstrap import (
        CanonicalBootstrapPolicy,
        canonical_bootstrap_point_cloud,
    )

    policy = CanonicalBootstrapPolicy(
        depth_stride=1,
        min_depth_m=0.1,
        max_depth_m=5.0,
        max_points=100,
        seed_voxel_size_m=0.0,
    )
    pcd = canonical_bootstrap_point_cloud((_observation(),), policy)

    assert pcd.points.shape == (24, 3)
    assert pcd.colors.shape == (24, 3)
    assert np.allclose(pcd.points[:, 2], 2.0)
    assert np.allclose(pcd.colors, 128.0 / 255.0)


def test_canonical_bootstrap_fails_closed_instead_of_random_scatter():
    from eligsir.mapping.canonical_bootstrap import (
        CanonicalBootstrapPolicy,
        canonical_bootstrap_point_cloud,
    )

    policy = CanonicalBootstrapPolicy(
        depth_stride=1,
        min_depth_m=0.1,
        max_depth_m=5.0,
        max_points=100,
        seed_voxel_size_m=0.0,
    )
    with pytest.raises(RuntimeError, match="no valid metric depth points"):
        canonical_bootstrap_point_cloud(
            (_observation(depth_value=float("nan")),), policy
        )


def test_canonical_bootstrap_temporal_seed_rejects_unsupported_depth():
    from eligsir.mapping.canonical_bootstrap import (
        CanonicalBootstrapPolicy,
        canonical_bootstrap_point_cloud,
    )

    clean = np.full((4, 6), 2.0, dtype=np.float32)
    outlier = clean.copy()
    outlier[1, 2] = 3.0
    policy = CanonicalBootstrapPolicy(
        depth_stride=1,
        min_depth_m=0.1,
        max_depth_m=5.0,
        max_points=100,
        seed_voxel_size_m=0.0,
        temporal_check_enabled=True,
        temporal_tolerance_m=0.05,
        require_temporal_for_seed=True,
    )

    pcd = canonical_bootstrap_point_cloud(
        (
            _observation(ordinal=0, depth_array=clean),
            _observation(ordinal=1, depth_array=outlier),
        ),
        policy,
    )

    # The inconsistent pixel is rejected in both frames; all 23 supported
    # samples per frame survive.
    assert pcd.points.shape == (46, 3)


def test_canonical_bootstrap_required_temporal_seed_fails_closed_without_peer():
    from eligsir.mapping.canonical_bootstrap import (
        CanonicalBootstrapPolicy,
        canonical_bootstrap_point_cloud,
    )

    policy = CanonicalBootstrapPolicy(
        depth_stride=1,
        min_depth_m=0.1,
        max_depth_m=5.0,
        max_points=100,
        seed_voxel_size_m=0.0,
        temporal_check_enabled=True,
        temporal_tolerance_m=0.05,
        require_temporal_for_seed=True,
    )

    with pytest.raises(RuntimeError, match="no temporally supported metric depth points"):
        canonical_bootstrap_point_cloud((_observation(),), policy)


def test_production_bootstrap_does_not_call_historical_scene_bootstrap():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    source = inspect.getsource(CanonicalMappingRuntime._initialize)
    assert "initialize_from_frames" not in source
    assert "initialize_from_canonical_frames" in source


def test_canonical_scene_bootstrap_method_has_no_loader_resize_or_random_fallback():
    from scene.streaming_scene import StreamingScene

    source = inspect.getsource(StreamingScene.initialize_from_canonical_frames)
    forbidden = (
        "_build_pcd_from_frames",
        "_frame_to_camera",
        "load_frame_",
        "PIL",
        "resize",
        "depth_to_meters",
        "random",
        "except Exception",
    )
    assert not [name for name in forbidden if name in source]


def test_canonical_scene_ingests_after_retiring_prior_mapping_frame_payload(
    monkeypatch, tmp_path
):
    from eligsir.mapping import canonical_scene as canonical_scene_module
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene
    from eligsir.runtime.config import (
        CanonicalBootstrapPolicy,
        CanonicalRuntimePolicy,
    )

    class _Camera:
        def __init__(self, observation):
            self._streaming_frame = observation
            self._mapping_frame = observation.mapping_frame
            self.original_image = object()
            self._native_image_loader = lambda: None
            self._pending_image_loader = None
            self._canonical_native_resolution_state = object()

    monkeypatch.setattr(
        canonical_scene_module,
        "canonical_camera_from_observation",
        lambda observation, **_kwargs: _Camera(observation),
    )
    scene = CanonicalStreamingScene(
        CanonicalRuntimePolicy(
            model_path=str(tmp_path),
            keyframe_window=1,
            live_payload_max_frames=1,
        ),
        CanonicalBootstrapPolicy(),
        SimpleNamespace(),
        (_observation(),),
    )
    scene.bind_device_transfer_owner(
        SimpleNamespace(device="cpu", prefetch_cameras=lambda _cameras: None)
    )

    first, _, _ = scene.ingest_canonical_frame(_observation(ordinal=0, x=0.0))
    scene.ingest_canonical_frame(_observation(ordinal=1, x=4.0))
    assert scene.retire_live_admitted_payloads() == 1
    assert first._mapping_frame is None
    assert first._streaming_frame.mapping_frame is None

    scene.ingest_canonical_frame(_observation(ordinal=2, x=8.0))

    assert scene.cameras_extent == pytest.approx(4.4)


def test_canonical_scene_admission_and_promotion_mark_residency_dirty(
    monkeypatch, tmp_path
):
    from eligsir.mapping import canonical_scene as canonical_scene_module
    from eligsir.mapping.canonical_scene import CanonicalStreamingScene
    from eligsir.runtime.config import (
        CanonicalBootstrapPolicy,
        CanonicalRuntimePolicy,
    )

    class _Camera:
        def __init__(self, observation):
            self._streaming_frame = observation
            self._mapping_frame = observation.mapping_frame
            self.original_image = object()
            self._native_image_loader = lambda: None
            self._pending_image_loader = None
            self._canonical_native_resolution_state = object()

    monkeypatch.setattr(
        canonical_scene_module,
        "canonical_camera_from_observation",
        lambda observation, **_kwargs: _Camera(observation),
    )
    scene = CanonicalStreamingScene(
        CanonicalRuntimePolicy(
            model_path=str(tmp_path),
            keyframe_window=1,
            live_payload_max_frames=1,
        ),
        CanonicalBootstrapPolicy(),
        SimpleNamespace(),
        (_observation(),),
    )
    scene.bind_device_transfer_owner(
        SimpleNamespace(device="cpu", prefetch_cameras=lambda _cameras: None)
    )

    scene._payload_residency_dirty = False
    scene.ingest_canonical_frame(_observation(ordinal=1))
    assert scene._payload_residency_dirty is True

    scene._payload_residency_dirty = False
    scene.promote_historical_frame(_observation(ordinal=2))
    assert scene._payload_residency_dirty is True
