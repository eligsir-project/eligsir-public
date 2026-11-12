"""The canonical ``TrackingProvider`` contract (eligsir#1075).

This module replaces the retired ``SLAMBackend`` ABC + ``_REGISTRY`` decorator
registry (``eligsir/contracts/slam_backend.py``) with a single, structural
provider protocol. There is exactly one tracking hierarchy in the repository:
this one.

Design decisions, and why
=========================

**A ``Protocol``, not an ABC, and no registry.** The retired architecture bound
provider *selection* to provider *identity*: a class was only reachable through
``get_backend(name)`` after some unrelated module had been imported for its
``@register_backend`` side effect, which is why four separate call sites each
repeated the same hand-maintained four-line import block. Selection is a
composition concern and now lives in an explicit factory
(:func:`eligsir.tracking.factory.build_tracking_provider`); identity is
structural. A class satisfies this contract by having the right methods, so a
caller may pass its own provider without importing, subclassing or registering
anything here.

**One method, ``observe``, not ``initialize`` + ``track``.** The retired ABC
required both, but of its four implementations two (``PrecomputedBackend`` and
``ROSSLAMBackend``) had no distinct first-frame behaviour at all -- both
delegated ``initialize`` and ``track`` to the same body. Forcing every provider
to carry a stub method is how a protocol grows fields no implementation means.
A provider that genuinely has first-frame state (``Open3DICPTracker``) keeps its
own ``initialize`` as a concrete method and drives it from the first
:meth:`TrackingProvider.observe` call; that is the provider's own business, not
the contract's.

**No ``render_context`` parameter.** The retired ABC carried
``gaussian_ctx: Optional[GaussianRenderContext]`` on every ``track()`` call, and
``GaussianRenderContext`` handed a provider the live Gaussian model plus a render
callable -- i.e. full mutation authority over the map. #1075's owner-safety
clause permits a read-only render context only "where a provider genuinely needs
one", and **no live provider does**: ``gs_rendered_rgbd_odometry`` was the sole
``requires_gaussian_map = True`` backend and it was ``available = False``
(unselectable, ``track()`` raised ``NotImplementedError``). Carrying the
parameter forward would import the exact hazard the clause exists to prevent, in
service of zero real callers. The read-only-render-context capability is
therefore **deferred together with that parked backend**, and must be added as a
genuine read-only view -- never by reinstating ``GaussianRenderContext``.

**Consumes ``SensorPacket``, produces ``TrackingObservation``.** Both are #1074's
canonical, owner-safe runtime records. This is what makes the composition shape
in #1075 -- ``source -> SensorPacket -> TrackingProvider -> MappingSession`` --
literally true rather than aspirational, and it is why a provider never sees a
transport type: ROS pose/odom/TF conversion belongs to ``eligsir-ros2``.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .runtime import SensorPacket
from .runtime import TrackingObservation
from .types import Pose, RGBDFrame


@runtime_checkable
class TrackingProvider(Protocol):
    """A per-observation source of canonical tracking evidence.

    Implementations are free to estimate a pose (in-process odometry/ICP), to
    pass through a pose established elsewhere (a dataset's baked trajectory, an
    external SLAM frontend, a frozen/ground-truth trajectory), or to report that
    tracking is unavailable. All three use the same result semantics: the return
    value is always a :class:`~eligsir.contracts.runtime.TrackingObservation`
    whose ``status`` states what actually happened.

    A provider MUST NOT mutate the Gaussian map. It receives no handle to one.
    """

    #: Stable identifier recorded as ``TrackingObservation.provider``. Used for
    #: provenance and telemetry only -- never for dispatch, since selection is
    #: the factory's job and identity here is structural.
    provider_name: str

    def observe(
        self,
        packet: SensorPacket,
        *,
        predicted_pose: Optional[Pose] = None,
    ) -> TrackingObservation:
        """Return canonical tracking evidence for exactly one sensor observation.

        Args:
            packet: The observation to track. ``packet.source`` is the canonical
                identity; the returned observation MUST carry that same identity
                so downstream association cannot silently mismatch.
            predicted_pose: Optional motion prior (e.g. constant-velocity). A
                provider that cannot use a prior ignores it; it is never
                required, and a provider must never fabricate its result *from*
                the prior when tracking failed.

        Returns:
            A :class:`TrackingObservation`. Tracking failure is reported as
            ``TrackingStatus.INVALID``/``LOST`` with ``pose=None`` -- **never**
            as a raised exception and never as an identity pose, which would
            store a fabricated estimate as canonical provenance.
        """
        ...

    def close(self) -> None:
        """Release provider-owned resources. MUST be idempotent.

        Called by the owning session's ``close()``. A provider holding a
        subscription, a subprocess or a device context releases it here; a pure
        passthrough provider does nothing.
        """
        ...


def sensor_packet_to_rgbd_frame(packet: SensorPacket) -> RGBDFrame:
    """Adapt a canonical :class:`SensorPacket` to the legacy frontend record.

    The in-process frontends (``Open3DICPTracker``, ``Open3DOdometryBackend``)
    consume :class:`~eligsir.contracts.types.RGBDFrame`, which predates
    #1074's ingress records. This is the single conversion seam, so those
    frontends' numerics stay untouched by #1075.

    ``RGBDFrame.timestamp`` is seconds-as-float by definition, so this reads
    :attr:`SourceIdentity.timestamp_seconds` and inherits its documented
    lossiness. The integer ``timestamp_ns`` remains the canonical time and is
    preserved on ``packet.source``, which is what the returned observation is
    stamped with -- the lossy float never becomes an identity.

    Args:
        packet: The canonical observation to adapt.

    Returns:
        An equivalent :class:`RGBDFrame`. ``pose`` is left ``None``: a frontend
        must estimate, not be handed the answer it is being asked for.

    Raises:
        ValueError: If ``packet`` carries no depth buffer. Every current
            frontend is RGB-D; silently passing ``None`` would surface as an
            opaque failure inside third-party (Open3D) code instead of here.
    """
    if packet.depth is None:
        raise ValueError(
            f"{packet.source.source_id!r} observation {packet.source.source_ordinal} "
            "has no depth buffer; RGB-D tracking frontends require one"
        )
    return RGBDFrame(
        index=packet.source.source_ordinal,
        timestamp=packet.source.timestamp_seconds,
        rgb=np.asarray(packet.rgb.data),
        depth=np.asarray(packet.depth.data),
        intrinsics=packet.intrinsics,
        pose=None,
    )


__all__ = ["TrackingProvider", "sensor_packet_to_rgbd_frame"]
