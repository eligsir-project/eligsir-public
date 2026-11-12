"""Contract tests for the canonical ``TrackingProvider`` (eligsir#1075).

These pin the *contract*, not any one frontend's numerics: that a provider
consumes a canonical ``SensorPacket`` and produces a canonical
``TrackingObservation``, that failure is reported rather than fabricated, and
that exactly one hierarchy exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    SensorPacket,
    SourceIdentity,
    TrackingStatus,
)
from eligsir.contracts.tracking import (
    TrackingProvider,
    sensor_packet_to_rgbd_frame,
)
from eligsir.contracts.types import CameraIntrinsics, Pose, TrackingResult
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    FrontendTrackingProvider,
    observation_from_pose_matrix,
)

_INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width=64, height=48)


def _identity(ordinal: int = 0) -> SourceIdentity:
    return SourceIdentity(
        source_id="test_cam",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal,
    )


def _packet(ordinal: int = 0, *, tracking=None, with_depth: bool = True) -> SensorPacket:
    rgb = BufferRef.from_numpy(
        np.zeros((48, 64, 3), dtype=np.uint8), encoding="rgb8"
    )
    depth = (
        BufferRef.from_numpy(
            np.ones((48, 64), dtype=np.float32), encoding="32FC1", units="m"
        )
        if with_depth
        else None
    )
    return SensorPacket(
        source=_identity(ordinal),
        rgb=rgb,
        depth=depth,
        intrinsics=_INTRINSICS,
        tracking=tracking,
    )


class _StubFrontend:
    """Minimal frontend: records calls, returns a scripted result."""

    def __init__(self, result=None, raises: BaseException | None = None) -> None:
        self._result = result
        self._raises = raises
        self.initialize_calls = 0
        self.track_calls = 0
        self.closed = 0

    def _out(self):
        if self._raises is not None:
            raise self._raises
        return self._result or TrackingResult(
            pose=Pose.identity(), confidence=0.9, valid=True, method="stub"
        )

    def initialize(self, first_frame):
        self.initialize_calls += 1
        return self._out()

    def track(self, frame, T_pred=None):
        self.track_calls += 1
        return self._out()

    def close(self):
        self.closed += 1


# -- protocol conformance -------------------------------------------------


def test_both_providers_satisfy_the_protocol_structurally():
    """One hierarchy: conformance is structural, needs no base class."""
    assert isinstance(ExternalTrackingProvider(), TrackingProvider)
    assert isinstance(
        FrontendTrackingProvider(_StubFrontend(), provider_name="stub"),
        TrackingProvider,
    )


def test_protocol_exposes_no_gaussian_render_context():
    """Owner safety: a provider is never handed the live Gaussian map."""
    import inspect

    sig = inspect.signature(FrontendTrackingProvider.observe)
    assert set(sig.parameters) == {"self", "packet", "predicted_pose"}


# -- ExternalTrackingProvider ---------------------------------------------


def test_external_provider_passes_through_the_attached_observation():
    obs = observation_from_pose_matrix(
        _identity(3), np.eye(4), provider="dataset_pose", confidence=0.75
    )
    packet = _packet(3, tracking=obs)

    result = ExternalTrackingProvider().observe(packet)

    assert result is obs
    assert result.status is TrackingStatus.VALID
    assert result.confidence == pytest.approx(0.75)


def test_external_provider_reports_missing_tracking_as_invalid_not_identity():
    """A missing pose must never become an identity pose.

    Recording identity would store a fabricated estimate as canonical
    provenance and silently place geometry at the world origin.
    """
    result = ExternalTrackingProvider().observe(_packet(1))

    assert result.status is TrackingStatus.INVALID
    assert result.pose is None
    assert result.usable is False
    assert result.source == _identity(1)


def test_external_provider_close_is_idempotent():
    provider = ExternalTrackingProvider()
    provider.close()
    provider.close()


# -- FrontendTrackingProvider ---------------------------------------------


def test_frontend_provider_initializes_once_then_tracks():
    frontend = _StubFrontend()
    provider = FrontendTrackingProvider(frontend, provider_name="stub")

    provider.observe(_packet(0))
    provider.observe(_packet(1))
    provider.observe(_packet(2))

    assert frontend.initialize_calls == 1
    assert frontend.track_calls == 2


def test_frontend_provider_stamps_the_packets_own_identity():
    """Association is by identity; a foreign identity would misattribute."""
    provider = FrontendTrackingProvider(_StubFrontend(), provider_name="stub")

    result = provider.observe(_packet(7))

    assert result.source == _identity(7)


def test_frontend_provider_maps_invalid_result_to_invalid_never_degraded():
    """A frontend's own gates rejected this pose; DEGRADED would be usable."""
    frontend = _StubFrontend(
        TrackingResult(pose=Pose.identity(), confidence=0.0, valid=False, method="stub")
    )
    provider = FrontendTrackingProvider(frontend, provider_name="stub")

    result = provider.observe(_packet())

    assert result.status is TrackingStatus.INVALID
    assert result.usable is False


def test_frontend_provider_converts_exception_to_invalid_observation():
    frontend = _StubFrontend(raises=RuntimeError("open3d exploded"))
    provider = FrontendTrackingProvider(frontend, provider_name="stub")

    result = provider.observe(_packet())

    assert result.status is TrackingStatus.INVALID
    assert "open3d exploded" in result.diagnostics["error"]


def test_frontend_provider_propagates_not_implemented():
    """Selecting a frontend that cannot track must fail loudly at frame 1."""
    frontend = _StubFrontend(raises=NotImplementedError("parked"))
    provider = FrontendTrackingProvider(frontend, provider_name="stub")

    with pytest.raises(NotImplementedError):
        provider.observe(_packet())


def test_frontend_provider_close_is_idempotent_and_closes_once():
    frontend = _StubFrontend()
    provider = FrontendTrackingProvider(frontend, provider_name="stub")

    provider.close()
    provider.close()

    assert frontend.closed == 1
    with pytest.raises(RuntimeError, match="after close"):
        provider.observe(_packet())


# -- packet adaptation ----------------------------------------------------


def test_packet_to_frame_preserves_identity_and_leaves_pose_unset():
    frame = sensor_packet_to_rgbd_frame(_packet(5))

    assert frame.index == 5
    assert frame.intrinsics is _INTRINSICS
    # A frontend must estimate the pose, not be handed the answer.
    assert frame.pose is None


def test_packet_to_frame_refuses_a_depthless_packet():
    with pytest.raises(ValueError, match="no depth buffer"):
        sensor_packet_to_rgbd_frame(_packet(with_depth=False))


def test_observation_from_none_pose_is_invalid_not_identity():
    obs = observation_from_pose_matrix(_identity(), None, provider="p")

    assert obs.status is TrackingStatus.INVALID
    assert obs.pose is None
