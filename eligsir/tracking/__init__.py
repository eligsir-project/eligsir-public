"""In-process tracking frontends and the canonical ``TrackingProvider`` binding.

Modules here own tracking *algorithms* (``open3d_icp``, ``open3d_odometry``),
their binding to the canonical contract (``providers``), and explicit selection
by name (``factory``). They subclass nothing and register with nothing: the
``SLAMBackend`` ABC and its ``@register_backend`` decorator registry were
removed by #1075, and there is now exactly one tracking hierarchy.

:class:`~eligsir.contracts.tracking.TrackingProvider` is defined in
``eligsir.contracts`` (contracts own protocols) and re-exported here so
either import path resolves it.

See ``docs/architecture/2026-08-23-issue1075-mapping-session-tracking-provider.md``.
"""

from eligsir.contracts.tracking import TrackingProvider, sensor_packet_to_rgbd_frame
from eligsir.tracking.factory import (
    build_tracking_frontend,
    build_tracking_provider,
)
from eligsir.tracking.observation import (
    TRACKING_STATUS_STAT_KEY,
    tracking_observation_from_result,
    tracking_result_from_observation,
)
from eligsir.tracking.open3d_icp import Open3DICPTracker
from eligsir.tracking.open3d_odometry import Open3DOdometryTracker
from eligsir.tracking.providers import (
    ExternalTrackingProvider,
    FrontendTrackingProvider,
    observation_from_pose_matrix,
)

__all__ = [
    "TRACKING_STATUS_STAT_KEY",
    "ExternalTrackingProvider",
    "FrontendTrackingProvider",
    "Open3DICPTracker",
    "Open3DOdometryTracker",
    "TrackingProvider",
    "build_tracking_frontend",
    "build_tracking_provider",
    "observation_from_pose_matrix",
    "sensor_packet_to_rgbd_frame",
    "tracking_observation_from_result",
    "tracking_result_from_observation",
]
