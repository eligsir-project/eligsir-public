from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.contracts.control import DepthInsertionPlan, ModalityEligibility
from eligsir.contracts.runtime import BufferRef, MappingFrame, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.mapping.integration import device_integration_planner
from eligsir.mapping.integration import integration_planner
from eligsir.runtime.config import DepthInsertionPolicy


def _frame():
    rgb = np.full((6, 8, 3), 64, dtype=np.uint8)
    depth = np.full((6, 8), 1.75, dtype=np.float32)
    mapping = MappingFrame(
        source=SourceIdentity("live", 0, 4, 2_000_000_000),
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="meters"),
        intrinsics=CameraIntrinsics(
            fx=20.0, fy=21.0, cx=4.0, cy=3.0, width=8, height=6
        ),
        device="cpu",
        pose=Pose.identity(),
    )
    return SimpleNamespace(
        mapping_frame=mapping,
        width=8,
        height=6,
        c2w=np.eye(4, dtype=np.float32),
        depth_scale=1000.0,
        depth_fx=None,
        depth_fy=None,
        depth_cx=None,
        depth_cy=None,
    )


def _unexpected_decode(*_args, **_kwargs):
    raise AssertionError("canonical MappingFrame must prevent a second source decode")


def test_device_insertion_source_consumes_mapping_frame_without_decode(monkeypatch):
    assert not hasattr(device_integration_planner, "load_frame_depth_np")
    assert not hasattr(device_integration_planner, "load_frame_rgb")

    z, rgb, c2w, native_hw, target_hw = device_integration_planner._load_source(_frame())

    assert np.allclose(z, 1.75)
    assert np.allclose(rgb, 64.0 / 255.0)
    assert np.array_equal(c2w, np.eye(4, dtype=np.float32))
    assert native_hw == target_hw == (6, 8)


def test_legacy_insertion_source_consumes_mapping_frame_without_decode(monkeypatch):
    monkeypatch.setattr(integration_planner, "load_frame_depth_np", _unexpected_decode)
    monkeypatch.setattr(integration_planner, "load_frame_rgb", _unexpected_decode)

    z, rgb, native_hw, is_metric = integration_planner._load_source_arrays(_frame())

    assert is_metric is True
    assert native_hw == (6, 8)
    assert np.allclose(z, 1.75)
    assert np.allclose(rgb, 64.0 / 255.0)


def test_canonical_insertion_rejects_missing_mapping_frame_instead_of_decoding():
    frame = _frame()
    del frame.mapping_frame

    with pytest.raises(RuntimeError, match="canonical MappingFrame"):
        device_integration_planner._load_source(frame)


def test_canonical_insertion_rejects_rgb_depth_resolution_mismatch():
    frame = _frame()
    mapping = frame.mapping_frame
    frame.mapping_frame = MappingFrame(
        source=mapping.source,
        rgb=mapping.rgb,
        depth=BufferRef.from_numpy(
            np.ones((3, 4), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=mapping.intrinsics,
        device="cpu",
        pose=mapping.pose,
    )

    with pytest.raises(ValueError, match="resolution"):
        device_integration_planner._load_source(frame)


def test_canonical_insertion_rejects_intrinsics_that_do_not_match_rgb():
    frame = _frame()
    mapping = frame.mapping_frame
    frame.mapping_frame = MappingFrame(
        source=mapping.source,
        rgb=mapping.rgb,
        depth=mapping.depth,
        intrinsics=CameraIntrinsics(
            fx=20.0, fy=21.0, cx=4.0, cy=3.0, width=9, height=6
        ),
        device="cpu",
        pose=mapping.pose,
    )

    with pytest.raises(ValueError, match="intrinsics"):
        device_integration_planner._load_source(frame)


def test_canonical_insertion_rejects_missing_pose():
    frame = _frame()
    mapping = frame.mapping_frame
    frame.mapping_frame = MappingFrame(
        source=mapping.source,
        rgb=mapping.rgb,
        depth=mapping.depth,
        intrinsics=mapping.intrinsics,
        device="cpu",
        pose=None,
    )

    with pytest.raises(ValueError, match="pose"):
        device_integration_planner._load_source(frame)


def _insertion_plan(source):
    return DepthInsertionPlan(
        source=source,
        enabled=True,
        sample_stride=1,
        candidate_budget=8,
        seed_budget=4,
        voxel_size_m=0.02,
        minimum_confidence=0.0,
    )


def test_device_insertion_authority_rejects_ineligible_depth_modality():
    frame = _frame()
    with pytest.raises(ValueError, match="not modality-eligible"):
        device_integration_planner._validate_insertion_authority(
            frame,
            _insertion_plan(frame.mapping_frame.source),
            DepthInsertionPolicy(),
            ModalityEligibility(depth_insertion=False),
        )


def test_device_insertion_authority_rejects_cross_source_plan():
    frame = _frame()
    other = SourceIdentity("other", 0, 0, 1)
    with pytest.raises(ValueError, match="source must match"):
        device_integration_planner._validate_insertion_authority(
            frame,
            _insertion_plan(other),
            DepthInsertionPolicy(),
            ModalityEligibility(depth_insertion=True),
        )
