"""Explicit selection of a tracking frontend/provider by name (eligsir#1075).

This replaces ``eligsir.contracts.slam_backend``'s decorator registry.

**Why a plain function and not a registry.** The retired ``_REGISTRY`` was
populated purely as an import side effect of ``@register_backend``, so a name
only resolved after some unrelated module had been imported for its side
effect. That is why four separate call sites each carried the same
hand-maintained four-line ``import eligsir.slam.adapters.*  # noqa: F401``
block, kept in sync by hand. An explicit ``if/elif`` has no such coupling: the
set of names is readable in one place, resolution cannot depend on import
order, and an unknown name fails immediately with the full list.

This function is deliberately importable and callable inside a spawned child
process: :class:`eligsir.slam.runtime.process_worker.ProcessSLAMWorker`
reconstructs its frontend **by name in the child** (live frontend state --
Open3D objects, device contexts -- is not picklable), so name-based
construction had to survive the registry's removal even though the registry
itself did not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from eligsir.contracts.tracking import TrackingProvider

#: In-process estimators: names that construct a real tracking algorithm.
IN_PROCESS_FRONTENDS = ("open3d_odometry_live", "open3d_icp_live")

#: The one core name for a pose established outside an in-process estimator.
#: ROS producers attach the same canonical tracking observation before core.
EXTERNAL_POSE_SOURCES = ("dataset_pose",)

#: Removed with the ROS hard cut (#1073/#1075). Kept only to produce an
#: actionable error instead of a bare "unknown name".
_REMOVED = {
    "precomputed": (
        "The 'precomputed' alias was removed by the core hard cut; use "
        "'dataset_pose' for frame-attached poses."
    ),
    "ros_slam": (
        "The core 'ros_slam' backend was removed by eligsir#1075: ROS "
        "pose/odom/TF subscription and conversion now live in the "
        "eligsir-ros2 package, which composes MappingSession and stamps "
        "each SensorPacket with an already-converted TrackingObservation. For "
        "a ros2:// source use the default pose_source (dataset_pose), which "
        "consumes the pose the frame source already joined."
    ),
    "gs_rendered_rgbd_odometry": (
        "'gs_rendered_rgbd_odometry' was a parked design stub that was never "
        "selectable (available=False, track() raised NotImplementedError) and "
        "was removed by eligsir#1075 along with GaussianRenderContext. A "
        "Gaussian-map-reading tracker must be reintroduced with a genuine "
        "read-only render view, never by handing a provider the live model."
    ),
}


def _unknown(name: str) -> KeyError:
    known = sorted(IN_PROCESS_FRONTENDS + EXTERNAL_POSE_SOURCES)
    if name in _REMOVED:
        return KeyError(f"Tracking frontend {name!r} was removed. {_REMOVED[name]}")
    return KeyError(f"Unknown tracking frontend {name!r}. Available: {known}")


def validate_frontend_name(name: str) -> None:
    """Raise if ``name`` names no constructible tracking frontend.

    Name-only validation, deliberately without constructing anything: a caller
    that wants to fail fast in a *parent* process before spawning a child (see
    :func:`eligsir.mapping.scheduler.deferred_slam_admission
    .build_deferred_admission_tracker`) must not pay for -- or trigger the
    device/library side effects of -- an instance it is going to discard.

    Raises:
        KeyError: If ``name`` is unknown or names a removed backend.
    """
    if name not in IN_PROCESS_FRONTENDS and name not in EXTERNAL_POSE_SOURCES:
        raise _unknown(name)


def build_tracking_frontend(name: str, **kwargs: Any) -> Any:
    """Construct a raw in-process tracking frontend by name.

    Returns the estimator itself (``initialize``/``track`` over
    :class:`~eligsir.contracts.types.RGBDFrame`), **not** a provider. Used
    by :class:`~eligsir.slam.runtime.process_worker.ProcessSLAMWorker`,
    which speaks ``TrackingResult`` directly because it converts to
    ``TrackedFramePacket`` itself.

    The external-pose name resolves to
    :class:`~eligsir.tracking.passthrough.FramePosePassthroughTracker`, so
    the frame-pose passthrough without a compatibility alias.

    Args:
        name: One of :data:`IN_PROCESS_FRONTENDS` or
            :data:`EXTERNAL_POSE_SOURCES`.
        **kwargs: Forwarded to the frontend's constructor.

    Raises:
        KeyError: If ``name`` is unknown or names a removed backend.
    """
    if name == "open3d_odometry_live":
        from eligsir.tracking.open3d_odometry import Open3DOdometryTracker

        return Open3DOdometryTracker(**kwargs)
    if name == "open3d_icp_live":
        from eligsir.tracking.open3d_icp import Open3DICPTracker

        return Open3DICPTracker(**kwargs)
    if name in EXTERNAL_POSE_SOURCES:
        from eligsir.tracking.passthrough import FramePosePassthroughTracker

        return FramePosePassthroughTracker(**kwargs)
    raise _unknown(name)


def build_tracking_provider(
    name: str,
    *,
    frontend_kwargs: Optional[Dict[str, Any]] = None,
) -> TrackingProvider:
    """Construct the canonical :class:`TrackingProvider` for ``name``.

    Args:
        name: One of :data:`IN_PROCESS_FRONTENDS` or
            :data:`EXTERNAL_POSE_SOURCES`.
        frontend_kwargs: Constructor kwargs for an in-process frontend;
            ignored for external pose sources, which construct no estimator.

    Returns:
        A provider satisfying :class:`TrackingProvider`.

    Raises:
        KeyError: If ``name`` is unknown or names a removed backend.
    """
    from eligsir.tracking.providers import (
        ExternalTrackingProvider,
        FrontendTrackingProvider,
    )

    if name in EXTERNAL_POSE_SOURCES:
        return ExternalTrackingProvider(provider_name="dataset_pose")
    if name in IN_PROCESS_FRONTENDS:
        return FrontendTrackingProvider(
            build_tracking_frontend(name, **(frontend_kwargs or {})),
            provider_name=name,
        )
    raise _unknown(name)


__all__ = [
    "EXTERNAL_POSE_SOURCES",
    "IN_PROCESS_FRONTENDS",
    "build_tracking_frontend",
    "build_tracking_provider",
]
