"""Concrete :class:`~eligsir.contracts.tracking.TrackingProvider` implementations.

Two classes cover the three tracking paths ``train_streaming.py`` used to
open-code as an inline branch (the ``_deferred_tracked_result`` /
``is_dataset_pose_source`` / live-backend chain):

===========================  ==========================================
Retired inline branch        Provider
===========================  ==========================================
``_deferred_tracked_result``  :class:`ExternalTrackingProvider`
``is_dataset_pose_source``    :class:`ExternalTrackingProvider`
live backend ``track()``      :class:`FrontendTrackingProvider`
===========================  ==========================================

The first two collapse into one provider deliberately, not for brevity: in both
the pose is *established elsewhere* and stamped onto the observation before any
provider sees it -- by a deferred SLAM worker in one case, by a dataset's baked
trajectory (or a ROS pose/odom/TF join) in the other. #1075 requires exactly
that: "external/frozen/GT observations use the same result semantics". Giving
them separate classes would encode a distinction that exists only in where the
upstream pose came from, which is already recorded in ``provider``/``source``.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from eligsir.contracts.runtime import (
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.tracking import sensor_packet_to_rgbd_frame
from eligsir.contracts.types import Pose, RGBDFrame, TrackingResult
from eligsir.tracking.observation import tracking_observation_from_result


class _Frontend(Protocol):
    """The shape of an in-process tracking frontend wrapped by a provider.

    Matches ``Open3DICPTracker`` and ``Open3DOdometryBackend`` as they already
    are. Declared structurally so wrapping one requires no edit to it.
    """

    def initialize(self, first_frame: RGBDFrame) -> TrackingResult: ...

    def track(
        self, frame: RGBDFrame, T_pred: Optional[Pose] = None
    ) -> TrackingResult: ...


class ExternalTrackingProvider:
    """Passthrough for a pose established outside this process/provider.

    Covers dataset-baked trajectories, ground-truth/frozen trajectories, an
    external SLAM frontend joined at the source (ROS pose/odom/TF), and the
    deferred-SLAM-worker result. In every case the caller stamps the observation
    onto :attr:`SensorPacket.tracking` before submitting, and this provider's
    entire job is to **validate** it and hand it on unchanged.

    Validation is the point, not ceremony. The retired inline branch read
    ``getattr(new_frame, "c2w", None)`` and silently produced ``None`` when the
    attribute was missing, which downstream then had to re-check. Here a missing
    observation becomes an explicit ``TrackingStatus.INVALID`` record carrying a
    reason, so the failure is visible in provenance instead of inferred from an
    absence.
    """

    def __init__(self, provider_name: str = "external") -> None:
        """Args:
        provider_name: Recorded as ``TrackingObservation.provider`` when this
            provider must synthesize an INVALID record. A passed-through
            observation keeps its own ``provider`` value, since relabelling it
            would erase which upstream actually produced the pose.
        """
        self.provider_name = provider_name

    def observe(
        self,
        packet: SensorPacket,
        *,
        predicted_pose: Optional[Pose] = None,
    ) -> TrackingObservation:
        """Return ``packet.tracking``, or an INVALID record if it is absent.

        ``predicted_pose`` is ignored: this provider estimates nothing, and
        using a motion prior to fill in a missing pose would fabricate an
        estimate and store it as if it had been observed.
        """
        if packet.tracking is None:
            return TrackingObservation(
                source=packet.source,
                pose=None,
                status=TrackingStatus.INVALID,
                provider=self.provider_name,
                confidence=0.0,
                diagnostics={
                    "reason": "no tracking observation was attached to the packet",
                },
            )
        # SensorPacket.__post_init__ already enforces identity agreement, so
        # reaching here with a mismatch is impossible; returning it verbatim
        # therefore cannot smuggle a foreign identity downstream.
        return packet.tracking

    def close(self) -> None:
        """No-op: this provider owns no resources. Idempotent by construction."""


class FrontendTrackingProvider:
    """Adapts an in-process RGB-D tracking frontend to the canonical contract.

    Wraps a frontend exposing ``initialize``/``track`` over
    :class:`RGBDFrame` -- today ``Open3DICPTracker`` (``eligsir/tracking/
    open3d_icp.py``) and ``Open3DOdometryBackend``. The wrapped object is
    untouched by #1075: this class owns the two conversions
    (:func:`sensor_packet_to_rgbd_frame` in,
    :func:`~eligsir.tracking.observation.tracking_observation_from_result`
    out) and the first-frame state machine.

    First-frame handling lives here rather than in the protocol because only
    some frontends have distinct first-frame behaviour -- see
    ``eligsir/contracts/tracking.py``'s module docstring.
    """

    def __init__(self, frontend: _Frontend, provider_name: str) -> None:
        """Args:
        frontend: The frontend to wrap. Must expose ``initialize``/``track``.
        provider_name: Canonical identifier for provenance (e.g.
            ``"open3d_icp_live"``).
        """
        self._frontend = frontend
        self.provider_name = provider_name
        self._initialized = False
        self._closed = False

    def observe(
        self,
        packet: SensorPacket,
        *,
        predicted_pose: Optional[Pose] = None,
    ) -> TrackingObservation:
        """Track ``packet`` with the wrapped frontend.

        The first successful call routes to ``initialize()``; every later call
        routes to ``track()``. A frontend exception is converted to an INVALID
        observation carrying the error text, mirroring the retired inline
        branch's behaviour of logging and continuing -- **except**
        ``NotImplementedError``, which propagates, because that means the
        operator selected a frontend that cannot track at all and silently
        degrading a whole run to untracked is worse than failing at the first
        frame.
        """
        if self._closed:
            raise RuntimeError(
                f"{self.provider_name!r} received observe() after close()"
            )
        frame = sensor_packet_to_rgbd_frame(packet)
        try:
            if not self._initialized:
                result = self._frontend.initialize(frame)
                self._initialized = True
            else:
                result = self._frontend.track(frame, T_pred=predicted_pose)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return TrackingObservation(
                source=packet.source,
                pose=None,
                status=TrackingStatus.INVALID,
                provider=self.provider_name,
                confidence=0.0,
                diagnostics={"error": repr(exc)},
            )
        return tracking_observation_from_result(result, packet.source)

    def close(self) -> None:
        """Close the wrapped frontend if it supports it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._frontend, "close", None)
        if callable(closer):
            closer()


def observation_from_pose_matrix(
    source: SourceIdentity,
    c2w: Any,
    *,
    provider: str,
    confidence: float = 1.0,
    diagnostics: Optional[dict] = None,
) -> TrackingObservation:
    """Build a VALID observation from a raw 4x4 camera-to-world matrix.

    The stamping helper for producers of external poses -- a dataset source, a
    deferred SLAM worker, a ROS join -- so each one does not hand-assemble a
    :class:`TrackingObservation` (and hand-decide its status) at its own call
    site, which is how the four registry-import blocks drifted.

    Args:
        source: Canonical identity of the observation this pose describes.
        c2w: 4x4 camera-to-world matrix, or ``None``.
        provider: Recorded as ``TrackingObservation.provider``.
        confidence: Tracking confidence in [0, 1]; clamped.
        diagnostics: Optional extra provenance.

    Returns:
        A VALID observation, or an INVALID one with ``pose=None`` when ``c2w``
        is ``None`` -- never an identity pose, which would record a fabricated
        estimate as if observed.
    """
    import numpy as np

    if c2w is None:
        return TrackingObservation(
            source=source,
            pose=None,
            status=TrackingStatus.INVALID,
            provider=provider,
            confidence=0.0,
            diagnostics=dict(diagnostics or {}, reason="no pose available"),
        )
    return TrackingObservation(
        source=source,
        pose=Pose(np.asarray(c2w, dtype=np.float32)),
        status=TrackingStatus.VALID,
        provider=provider,
        confidence=min(1.0, max(0.0, float(confidence))),
        diagnostics=dict(diagnostics or {}),
    )


__all__ = [
    "ExternalTrackingProvider",
    "FrontendTrackingProvider",
    "observation_from_pose_matrix",
]
