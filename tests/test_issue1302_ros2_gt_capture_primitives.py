"""Issue #1302: the two core-side primitives eligsir-ros2 needs to
actually produce a `report.json` from a real bag-replay run.

eligsir-ros2#76 landed holdout WITHHOLDING (it never trains/replays a
designated holdout frame) but could not build a GT `Camera` from it, or reach
the live runtime's `.gaussians`/`.pipe`/render/background, because:

1. Building a `Camera` needs torch/`scene.cameras`, which
   `eligsir-ros2`'s machine-enforced boundary gate
   (`tests/test_boundary_invariants.py::test_ros_repository_contains_no_mapper_or_optimizer_loop`)
   forbids that repository's own source from importing, even lazily.
2. `ComposedMappingApplication` dropped the underlying runtime reference
   entirely, so there was nothing to pull `.gaussians`/`.pipe`/render/
   background/iteration from.

This file tests the two fixes:
- `eligsir.runtime.live_eval_holdout.build_holdout_camera_from_packet`
  (a `packet -> Camera` conversion `eligsir-ros2` can call by name
  without writing `import torch` anywhere in its own source).
- `ComposedMappingApplication.runtime` / `.materialization_policy` /
  `.materialization_backend` (read-only accessors back to the live runtime).

Needs torch (`Camera`/`CanonicalMappingRuntime` construction) -- excluded
from the lean CPU CI image by tests/conftest.py's `_NEEDS_TORCH` scan, same
as tests/test_issue1224_canonical_bootstrap.py.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="Camera/CanonicalMappingRuntime construction needs torch"
)

from eligsir.contracts.materialize import (
    DepthPolicy,
    MaterializationPolicy,
    numpy_backend,
)
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import (
    ComposedMappingApplication,
    MappingApplicationOptions,
)
from eligsir.runtime.live_eval_holdout import build_holdout_camera_from_packet
from eligsir.runtime.production_application import (
    CanonicalMappingRuntime,
    create_production_mapping_application,
)
from eligsir.tracking.providers import observation_from_pose_matrix


def _packet(*, ordinal: int = 3, with_pose: bool = True, x: float = 1.5) -> SensorPacket:
    source = SourceIdentity(
        source_id="ros2-holdout",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_700_000_000_000_000_000 + ordinal,
    )
    rgb = np.full((4, 6, 3), 200, dtype=np.uint8)
    depth = np.full((4, 6), 2.5, dtype=np.float32)
    tracking = None
    if with_pose:
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = x
        tracking = observation_from_pose_matrix(source, c2w, provider="ros2-tracking")
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=CameraIntrinsics(
            fx=8.0, fy=8.0, cx=3.0, cy=2.0, width=6, height=4,
        ),
        tracking=tracking,
    )


# ---------------------------------------------------------------------------
# build_holdout_camera_from_packet
# ---------------------------------------------------------------------------


def test_build_holdout_camera_from_packet_reuses_the_real_conversion():
    import torch

    packet = _packet(ordinal=7, x=2.0)

    camera = build_holdout_camera_from_packet(packet, data_device="cpu")

    # Same conversion canonical_camera_from_observation performs for every
    # OTHER admitted frame: colmap_id/uid keyed by source_ordinal, GT color
    # and metric depth attached, pose reflected in the camera's extrinsics.
    assert camera.colmap_id == 7
    assert camera.uid == 7
    assert camera.image_name == "000007"
    assert camera.original_image.shape == (3, 4, 6)
    assert torch.allclose(camera.original_image, torch.full((3, 4, 6), 200.0 / 255.0), atol=1e-5)
    assert camera._sensor_depth_cache is not None
    assert torch.allclose(camera._sensor_depth_cache, torch.full((4, 6), 2.5))
    # T is the translation of the *world-to-camera* transform; a c2w
    # translation of +2.0 along x inverts sign under R^T (identity rotation
    # here), so T == -2.0 is the expected, not a bug.
    assert float(camera.T[0]) == pytest.approx(-2.0)


def test_build_holdout_camera_from_packet_requires_resolved_tracking_pose():
    packet = _packet(with_pose=False)

    with pytest.raises(ValueError, match="tracking.pose"):
        build_holdout_camera_from_packet(packet, data_device="cpu")


def test_build_holdout_camera_from_packet_matches_default_production_materialization_policy():
    """Passing an explicit policy that matches the implicit default is a no-op."""

    packet = _packet(ordinal=1)
    default_camera = build_holdout_camera_from_packet(packet, data_device="cpu")
    explicit_camera = build_holdout_camera_from_packet(
        packet,
        data_device="cpu",
        materialization_policy=MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32),
        materialization_backend=numpy_backend(),
    )

    import torch

    assert torch.equal(default_camera.original_image, explicit_camera.original_image)
    assert torch.equal(default_camera._sensor_depth_cache, explicit_camera._sensor_depth_cache)


# ---------------------------------------------------------------------------
# ComposedMappingApplication accessors
# ---------------------------------------------------------------------------


def test_composed_mapping_application_runtime_accessor_defaults_to_none():
    options = MappingApplicationOptions(
        mapper_sink=lambda frame, observation: None,
        optimizer_tick=lambda: None,
    )
    application = ComposedMappingApplication(options)

    assert application.runtime is None
    assert application.materialization_policy is None
    assert application.materialization_backend is None


def test_composed_mapping_application_runtime_accessor_returns_bound_runtime():
    sentinel = object()
    options = MappingApplicationOptions(
        mapper_sink=lambda frame, observation: None,
        optimizer_tick=lambda: None,
        runtime=sentinel,
    )
    application = ComposedMappingApplication(options)

    assert application.runtime is sentinel


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA/ROCm production runtime",
)
def test_production_application_exposes_the_real_canonical_mapping_runtime(tmp_path):
    """The actual eligsir-ros2 call shape: get the runtime handle back
    from a real, no-options `create_production_mapping_application()` call,
    exactly as `deterministic_bag.launch.py`'s lifecycle node constructs it.
    """

    application = create_production_mapping_application(model_path=str(tmp_path / "run"))

    runtime = application.runtime
    assert isinstance(runtime, CanonicalMappingRuntime)
    # Cross-check against the pre-existing (private) extraction path other
    # tests in this repo already used before this accessor existed --
    # confirms this is genuinely the same object, not a second one.
    assert runtime is application._options.mapper_sink.__self__

    assert runtime.iteration == 0
    assert runtime.background is None  # not bootstrapped yet -- same as runtime._background
    assert callable(runtime.render_fn)

    assert application.materialization_policy is not None
    assert application.materialization_policy.depth == DepthPolicy.METERS_FLOAT32
    assert application.materialization_backend is not None
