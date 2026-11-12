"""Deferred-admission integration of :class:`SLAMWorker` into the streaming
training loop (W3-15, completing W3-04's unmet "no SLAM backend executes
synchronously in the training loop" promise for in-process SLAM backends).

**The one hard rule this module exists to enforce**: a frame is never
admitted with a stale or predicted pose. ``DeferredSlamAdmissionTracker``
submits the next source frame to a :class:`~eligsir.slam.runtime.worker
.SLAMWorker` (non-blocking) and only reports it as admissible once the
worker's real, asynchronously-computed pose has actually arrived via
``try_get()``. When no result is ready yet, :meth:`poll` returns ``None``
and the caller must not admit anything this iteration -- there is no
fallback path that fabricates, predicts, or reuses a stale pose to avoid
that wait. This mirrors the exact bug class documented in project memory
as ``motion-pregate-silent-admission-corruption`` and
``3dgs-slam-fragile-to-coordinate-discontinuity``: fusing a frame whose
pose is not actually known yet silently corrupts the map's coordinate
frame in a way PSNR does not reveal.

**Scope**: this only applies to the main per-frame ingest loop
(``train_streaming.py``'s "Phase 1.5-SLAM" block), and only when the pose
source is an in-process :class:`~eligsir.contracts.slam_backend
.SLAMBackend` computing its own odometry (``open3d_odometry_live``,
``gs_rendered_rgbd_odometry``) -- i.e. exactly the branch in
``train_streaming.py`` that calls ``_slam_backend.track()`` today. Dataset-
baked poses never reach this code path at all; a frame from such a source already
carries a real, already-resolved pose by the time it is ingested, so there
is nothing to defer.

**Backends that require live Gaussian-map access are not supported.**
``ProcessSLAMWorker``'s child process calls ``backend.track(frame)`` with
no ``gaussian_ctx`` (the live Gaussian model is not picklable/shareable
across the process boundary and the async worker was never designed to
carry it). A backend with ``requires_gaussian_map=True`` (e.g.
``gs_rendered_rgbd_odometry``) would silently track garbage or raise every
call. ``build_deferred_admission_tracker`` refuses to construct a tracker
for such a backend and raises ``RuntimeError`` instead of degrading
silently -- construct with ``--streaming_slam_worker_deferred_admission``
only for backends that do not need one (``open3d_odometry_live``,
``precomputed``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eligsir.mapping.candidates.contracts import (
    RawFramePacket,
    TrackedFramePacket,
    TrackingState,
    UndecodedBufferRef,
)
from eligsir.slam.runtime import SLAMWorker, SubmitOutcome

# Tracking states that carry a real, usable pose. LOST/INITIALIZING/UNKNOWN
# all map to TrackingState.LOST at the TrackedFramePacket layer (see
# eligsir/slam/runtime/process_worker.py::_tracking_result_to_tracked_packet
# -- any non-valid TrackingResult, or a backend exception, folds down to
# tracking_state=LOST with an identity placeholder matrix, which must never
# be mistaken for a real pose).
_USABLE_STATES = (TrackingState.OK, TrackingState.REINITIALIZED)


def raw_packet_from_streaming_frame(frame, source_index: int) -> RawFramePacket:
    """Build a :class:`RawFramePacket` from a ``StreamingRGBDFrame``.

    **Actively decodes** RGB/depth via ``load_frame_rgb``/
    ``load_frame_depth_np`` rather than trusting ``frame._rgb_np``/
    ``._depth_np`` are already populated. This matters: those attributes
    are eagerly set only for sources that decode up front (e.g. a live
    ROS2 frame); a lazy/disk-backed source (TUM, a generic RGBD sequence)
    leaves them ``None`` until something actually decodes the frame --
    normally a side effect of ``StreamingScene.ingest_next_frame()``
    itself (see ``scene/streaming_scene.py``'s own
    ``frame._rgb_np = np.asarray(load_frame_rgb(frame))`` pattern).
    Deferred admission calls this on a *peeked*, not-yet-ingested frame --
    strictly BEFORE ingest_next_frame ever runs for it (that ordering is
    the entire point) -- so nothing else has decoded it yet. Found via a
    real end-to-end smoke test (2026-08-09): omitting this fed all-zero
    RGB/depth to the SLAM backend, which the backend then tracked
    "successfully" against itself every frame (silently wrong odometry,
    not a crash) -- exactly the class of subtle bug this project's
    "verify empirically, don't assume" standard exists to catch. The
    decoded RGB is also cached back onto ``frame._rgb_np`` (matching the
    existing pattern) so a later real ``ingest_next_frame()`` for this
    same frame gets a warm-cache hit instead of a redundant decode.
    """
    from eligsir.utils.rgbd_frames import CameraIntrinsics as _RgbdIntr
    from eligsir.utils.rgbd_frames import depth_to_meters
    # W3-54: import the canonical eligsir.* module directly rather than
    # looping back through the utils.streaming_frames compat shim (which is
    # a pure re-export over this same module) -- see
    # scripts/check_import_fences.py's module docstring for the rule this
    # was previously violating.
    from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))

    rgb = getattr(frame, "_rgb_np", None)
    if rgb is None:
        try:
            rgb = np.asarray(load_frame_rgb(frame))
            frame._rgb_np = rgb  # warm the cache for the real ingest to follow
        except Exception:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)

    depth_m = getattr(frame, "_depth_np", None)
    if depth_m is None:
        try:
            depth_raw = load_frame_depth_np(frame)
            depth_m = (
                depth_to_meters(np.asarray(depth_raw), getattr(frame, "depth_scale", 1.0))
                if depth_raw is not None
                else np.zeros((height, width), dtype=np.float32)
            )
        except Exception:
            depth_m = np.zeros((height, width), dtype=np.float32)
    else:
        # Already in meters when cached this way elsewhere (mirrors the
        # synchronous _slam_backend.track() call site's own
        # getattr(new_frame, "_depth_np", None) usage).
        depth_m = np.asarray(depth_m)

    intrinsics = _RgbdIntr(
        fx=float(frame.fx), fy=float(frame.fy),
        cx=float(frame.cx), cy=float(frame.cy),
        width=width, height=height,
    )
    timestamp_s = float(getattr(frame, "timestamp", source_index))
    return RawFramePacket(
        source_index=source_index,
        timestamp_ns=int(timestamp_s * 1e9),
        rgb_ref=UndecodedBufferRef(array=np.asarray(rgb)),
        depth_ref=UndecodedBufferRef(array=depth_m.astype(np.float32)),
        intrinsics=intrinsics,
    )


@dataclass
class DeferredSlamAdmissionTracker:
    """Non-blocking submit/poll bridge between a :class:`SLAMWorker` and
    ``StreamingScene``'s frame cursor.

    Owns exactly one in-flight frame at a time (submits the next source
    frame only once the previous one has resolved, one way or another) --
    both shipped ``SLAMWorker`` implementations preserve submission order,
    so there is no need for a result-matching table beyond the single
    pending index tracked here.
    """

    worker: SLAMWorker
    pending_timeout_s: float = 5.0

    _pending_index: Optional[int] = field(default=None, init=False)
    _pending_since_ns: Optional[int] = field(default=None, init=False)
    _consecutive_failures_at_submit: int = field(default=0, init=False)

    frames_submitted: int = field(default=0, init=False)
    frames_admitted: int = field(default=0, init=False)
    frames_skipped_no_pose: int = field(default=0, init=False)
    pending_timeouts: int = field(default=0, init=False)

    def poll(self, streaming_scene) -> Optional[TrackedFramePacket]:
        """Non-blocking. Returns the real ``TrackedFramePacket`` for the
        NEXT source frame (and stamps its resolved pose onto that frame's
        ``.c2w`` in place) iff a real, already-computed pose has arrived
        for it. Returns ``None`` whenever the next frame is not yet
        admissible -- the caller must not call
        ``streaming_scene.ingest_next_frame()`` this iteration in that
        case. Never fabricates, predicts, or reuses a stale pose.

        A frame whose tracking genuinely failed (worker reports a
        non-usable ``tracking_state``) or whose result never arrives
        within ``pending_timeout_s`` is skipped via
        ``streaming_scene.skip_next_frame()`` -- advanced past, never
        admitted, loudly logged, and counted in
        ``frames_skipped_no_pose`` -- rather than either blocking forever
        or silently vanishing from telemetry.
        """
        if self._pending_index is not None:
            idx = self._pending_index
            tracked = self.worker.try_get()
            if tracked is not None:
                self._pending_index = None
                self._pending_since_ns = None
                if tracked.frame.source_index != idx:
                    # Both shipped SLAMWorker implementations preserve
                    # submission order (see worker.py's SLAMWorker
                    # docstring), so this should never happen with only
                    # one frame ever in flight. Fail loud rather than
                    # silently mismatching a pose to the wrong frame.
                    raise RuntimeError(
                        f"[deferred_slam] worker returned a result for frame "
                        f"idx={tracked.frame.source_index} while idx={idx} was "
                        "pending -- SLAMWorker ordering guarantee violated; "
                        "refusing to apply a pose to the wrong frame."
                    )
                if tracked.tracking_state in _USABLE_STATES:
                    frame_obj = streaming_scene._get_source_frame(idx)
                    if frame_obj is not None:
                        frame_obj.c2w = np.asarray(
                            tracked.world_T_camera, dtype=np.float32
                        )
                    self.frames_admitted += 1
                    return tracked
                print(
                    f"[streaming][deferred_slam] frame idx={idx} tracking_state="
                    f"{tracked.tracking_state.value} -- no usable pose; skipping "
                    "this frame (never fabricating a pose to admit it).",
                    flush=True,
                )
                streaming_scene.skip_next_frame()
                self.frames_skipped_no_pose += 1
                # Fall through below to try submitting the next frame in
                # this same call, so one bad frame does not cost an extra
                # idle training iteration.
            else:
                elapsed_s = (time.monotonic_ns() - self._pending_since_ns) / 1e9
                health = self.worker.health()
                # A backend exception for THIS pending frame is consumed
                # (and its message discarded) by try_get() itself the
                # moment it happens -- see ProcessSLAMWorker.try_get's
                # "error" branch -- so it never shows up as a returned
                # TrackedFramePacket at all. Since only one frame is ever
                # in flight here, a rise in consecutive_failures since
                # submission unambiguously means THIS frame's tracking
                # already failed; detect that immediately instead of
                # waiting out the full timeout for something that already
                # happened.
                _failed_now = health.consecutive_failures > self._consecutive_failures_at_submit
                if _failed_now or elapsed_s > self.pending_timeout_s or not health.is_usable:
                    _reason = (
                        "backend track() raised" if _failed_now
                        else f"no result within {elapsed_s:.1f}s"
                    )
                    print(
                        f"[streaming][deferred_slam] frame idx={idx} tracking "
                        f"result unavailable ({_reason}; worker_state="
                        f"{health.state.value}, last_error={health.last_error!r}) "
                        "-- skipping WITHOUT a synthesized pose.",
                        flush=True,
                    )
                    streaming_scene.skip_next_frame()
                    self.frames_skipped_no_pose += 1
                    if not _failed_now:
                        # Only count as a timeout when it wasn't a known,
                        # immediately-detected backend failure -- keeps the
                        # two telemetry counters meaningfully distinct.
                        self.pending_timeouts += 1
                    self._pending_index = None
                    self._pending_since_ns = None
                    # Fall through to try the next frame.
                else:
                    return None  # Correct default: keep waiting.

        # No frame currently in flight: try to submit the next one.
        frame = streaming_scene.peek_next_frame()
        if frame is None:
            return None
        idx = streaming_scene._source_idx
        packet = raw_packet_from_streaming_frame(frame, idx)
        outcome = self.worker.submit(packet)
        if outcome.outcome == SubmitOutcome.ACCEPTED:
            self._pending_index = idx
            self._pending_since_ns = outcome.submitted_at_ns
            self._consecutive_failures_at_submit = self.worker.health().consecutive_failures
            self.frames_submitted += 1
        # ACCEPTED -> now pending, poll again next call.
        # Any REJECTED_* -> try again next call; this frame is simply not
        # admissible yet either way.
        return None

    def close(self) -> None:
        self.worker.close()

    def telemetry(self) -> dict:
        return {
            "frames_submitted": self.frames_submitted,
            "frames_admitted": self.frames_admitted,
            "frames_skipped_no_pose": self.frames_skipped_no_pose,
            "pending_timeouts": self.pending_timeouts,
        }


def build_deferred_admission_tracker(
    backend_name: str,
    backend_kwargs: dict,
    *,
    capacity: int = 4,
    pending_timeout_s: float = 5.0,
    artificial_delay_s: float = 0.0,
) -> DeferredSlamAdmissionTracker:
    """Construct a :class:`DeferredSlamAdmissionTracker` backed by a real
    :class:`ProcessSLAMWorker` subprocess running *backend_name*.

    Raises:
        KeyError: if *backend_name* names no known tracking frontend. #1075
            replaced the ``SLAMBackend`` registry with an explicit factory, so
            this is now validated eagerly in the parent process rather than
            failing inside the freshly-spawned child.
    """
    from eligsir.slam.runtime.process_worker import ProcessSLAMWorker
    from eligsir.tracking.factory import validate_frontend_name

    # Fail fast in the parent: check the name here so an unknown/removed
    # frontend raises at the call site with the full list of valid names,
    # instead of surfacing as an opaque ("init_error", ...) message from a
    # child process that has already been spawned. Name-only -- the child
    # constructs the real instance (live frontend state is not picklable),
    # and building one here just to discard it would pay for, and trigger the
    # device side effects of, an object the parent never uses.
    #
    # The retired ``requires_gaussian_map`` guard is gone with the ABC that
    # declared it: ``gs_rendered_rgbd_odometry`` was the only backend that
    # ever set it, it was never selectable (``available = False``), and #1075
    # removed it along with ``GaussianRenderContext``. No constructible
    # frontend can ask for Gaussian-map access, so there is nothing left to
    # refuse here -- see ``eligsir/contracts/tracking.py``.
    validate_frontend_name(backend_name)

    worker = ProcessSLAMWorker(
        backend_name,
        backend_kwargs,
        capacity=capacity,
        artificial_delay_s=artificial_delay_s,
    )
    return DeferredSlamAdmissionTracker(worker, pending_timeout_s=pending_timeout_s)
