"""Frontend that reports the pose already carried on the frame.

The registry-free successor to the retired ``PrecomputedBackend``. It estimates
nothing: it reports whatever pose the frame already carries, so a dataset's
baked trajectory or a frozen/ground-truth trajectory flows through the same
``initialize``/``track`` shape as a real estimator.

**Prefer** :class:`eligsir.tracking.providers.ExternalTrackingProvider` in
new code -- on the canonical ``SensorPacket`` path an externally-established
pose arrives as a :class:`~eligsir.contracts.runtime.TrackingObservation`
and needs no frontend at all. This class exists for the one place that genuinely
needs the *frontend* shape: :class:`~eligsir.slam.runtime.process_worker
.ProcessSLAMWorker`, which reconstructs a frontend by name inside a spawned
child process and speaks ``RGBDFrame``/``TrackingResult`` directly.
"""

from __future__ import annotations

from typing import Optional

from eligsir.contracts.types import Pose, RGBDFrame, TrackingResult


class FramePosePassthroughTracker:
    """Report ``frame.pose`` verbatim, with no estimation.

    Args:
        fallback_identity: When ``True``, a frame with no pose yields an
            identity pose at ``confidence=0.0`` and ``valid=False`` rather than
            raising. Default ``False``: fabricating identity for a frame whose
            pose is genuinely unknown silently places geometry at the world
            origin, so it must be opted into explicitly (the deferred-worker
            child process does, because its packets deliberately carry no pose).
    """

    name = "dataset_pose"

    def __init__(self, fallback_identity: bool = False) -> None:
        self._fallback_identity = bool(fallback_identity)

    def initialize(self, first_frame: RGBDFrame) -> TrackingResult:
        """Identical to :meth:`track`: there is no first-frame state to build."""
        return self._result(first_frame)

    def track(
        self, frame: RGBDFrame, T_pred: Optional[Pose] = None
    ) -> TrackingResult:
        """Return ``frame.pose``. ``T_pred`` is ignored -- nothing is estimated."""
        return self._result(frame)

    def _result(self, frame: RGBDFrame) -> TrackingResult:
        if frame.pose is not None:
            return TrackingResult(
                pose=frame.pose, confidence=1.0, valid=True, method=self.name
            )
        if not self._fallback_identity:
            raise ValueError(
                f"{type(self).__name__}: frame {frame.index} has no pose set and "
                "fallback_identity=False. The frame source must stamp a pose, or "
                "an estimating frontend must be selected."
            )
        return TrackingResult(
            pose=Pose.identity(), confidence=0.0, valid=False, method=self.name
        )


__all__ = ["FramePosePassthroughTracker"]
