"""Compatibility bridge: legacy ``Viewer`` ABC -> ``CanonicalViewer`` (issue #570 AC1/#572).

Migration PR referenced by #570's own status comment: "A migration PR routing
``registry.make_viewer`` -> ``resolve_preset_sinks`` -> ``CanonicalViewer(sinks=[...])``".
Scoped here to the ``viser`` preset ONLY -- ``rerun`` and ``ros2_rviz`` keep
resolving to the untouched legacy backends (``rerun_viewer.py`` /
``ros2_rviz.py``), so this migration changes zero behaviour unless a run
explicitly opts in with ``--viewer viser``. Flipping any *default* is #576's
job, not this file's.

``CanonicalViewer`` deliberately does not subclass the legacy ``Viewer`` ABC
(see its own docstring: inheriting the hook surface would re-import the
model/callback-shaped problem this architecture removes). This adapter is the
"compatibility bridge... where its behavioural cost can be measured" that
docstring names -- it subclasses ``Viewer`` so ``registry.make_viewer`` and
every existing ``_viewer.log_*`` call site keep working unmodified, while
holding a real ``CanonicalViewer`` + bounded staging ring + off-thread drain
underneath.

Zero-overhead notes for the wrapped hooks
------------------------------------------
* :meth:`log_training_pair_tensors` (NEW method, not on the legacy ABC --
  ``train_streaming.py``'s training-pair call site uses this directly instead
  of the legacy ``log_rendered``) stages both halves via
  ``PreviewStagingRing.try_submit_pair`` -- an async D2H schedule, never a
  ``.cpu()``/``.item()`` call on the training thread -- and returns
  immediately.
* The remaining legacy hooks (``log_metrics``, ``log_trajectory``,
  ``log_keyframe``, ``log_rgb``) already receive plain CPU-side data at their
  existing call sites (no tensor, no GPU sync to remove), so they build the
  canonical event directly and hand it to ``CanonicalViewer.submit`` (a bus
  write, not a sink call -- see ``canonical.py``).
* ``log_gaussian_model``/``log_gaussian_map`` (the legacy ``Viewer`` ABC's 3D
  map hooks, which take plain, already-``.cpu().numpy()``'d arrays) remain
  NO-OPS here on purpose -- calling either would mean the map was already
  fully read back to CPU before this adapter ever saw it, exactly the
  full-map-readback defect #573 exists to remove from Rerun.
* The #1200 ``VersionedMapOutputLane`` is bound once and polled by this
  adapter's existing drain thread. It owns bounded sampling and the sole
  OUTPUT readback; this adapter only forwards completed immutable events.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from eligsir.contracts.types import CameraIntrinsics, Pose, RGBDFrame, TrackingResult
from eligsir.viewers.base import Viewer
from eligsir.viewers.canonical import CanonicalViewer, CanonicalViewerConfig
from eligsir.viewers.drain_thread import PreviewDrainThread
from eligsir.viewers.events import (
    CameraRole,
    CapturedImage,
    ImageKind,
    KeyframePreview,
    MappingPhase,
    MetricsPreview,
    RawInputPreview,
    TrainingPairPreview,
    TrajectoryPreview,
)
from eligsir.viewers.sinks.base import PreviewSink
from eligsir.viewers.staging import PreviewStagingRing

__all__ = ["CanonicalViewerAdapter"]

_ROLE_MAP = {
    "NEW": CameraRole.NEW,
    "LOCAL": CameraRole.LOCAL,
    "REPLAY": CameraRole.REPLAY,
    "GLOBAL": CameraRole.GLOBAL,
}

#: Cap on how many trailing poses ``log_trajectory`` ever converts to an
#: array, independent of how large the caller's own running pose list has
#: grown (e.g. ``train_streaming.py``'s ``_trajectory_est``, which is never
#: itself bounded/evicted -- see that call site). Bounding here keeps both
#: the event payload size and the cost of building it constant over an
#: arbitrarily long run; ``CanonicalViewerConfig.trajectory_max_hz`` already
#: bounds *emission rate*, this bounds *emitted size*. 4000 poses is far more
#: than any sane trail visualization needs (a 4x4 float32 pose is 64 bytes,
#: so the cap is ~256KB) while comfortably covering this project's longest
#: recorded trajectories (e.g. floor3's ~5270 source frames).
_MAX_TRAJECTORY_POSES = 4000


def _poses_to_array(poses: List[Pose]) -> Optional[np.ndarray]:
    """``[Pose, ...]`` -> ``(N, 4, 4)`` float32, skipping any ``None`` entries."""
    mats = [np.asarray(p.matrix, dtype=np.float32) for p in poses if p is not None]
    if not mats:
        return None
    return np.stack(mats, axis=0)


class CanonicalViewerAdapter(Viewer):
    """Legacy ``Viewer`` surface backed by ``CanonicalViewer`` + Viser sink(s).

    Constructed only by ``registry.py``'s ``viser`` factory (opt-in preset).
    """

    def __init__(
        self,
        *,
        sinks: Optional[List[PreviewSink]] = None,
        config: Optional[CanonicalViewerConfig] = None,
        staging_slots: int = 3,
        drain_poll_interval_s: float = 1.0,
        **_ignored_kwargs: Any,
    ) -> None:
        self._viewer = CanonicalViewer(config=config, sinks=tuple(sinks or ()))
        self._ring = PreviewStagingRing(n_slots=max(2, int(staging_slots)))
        self._map_output_lane = None
        self._drain = PreviewDrainThread(
            on_tick=self._tick, poll_interval_s=drain_poll_interval_s
        )
        self._drain.start()
        self._optimizer_step_id = 0
        self._closed = False

    # -- background tick: consumer side only, never the training thread --
    def _tick(self) -> None:
        while True:
            staged = self._ring.poll_completed()
            if staged is None:
                break
            meta = staged.meta
            try:
                event = TrainingPairPreview(
                    iteration=int(meta["iteration"]),
                    wall_time_s=float(meta.get("wall_time_s", time.monotonic())),
                    target=CapturedImage(
                        staged.target,
                        camera_id=int(meta["camera_id"]),
                        kind=ImageKind.TARGET,
                        source_frame_id=meta.get("source_frame_id"),
                    ),
                    render=CapturedImage(
                        staged.render,
                        camera_id=int(meta["camera_id"]),
                        kind=ImageKind.RENDER,
                        source_frame_id=meta.get("source_frame_id"),
                    ),
                    optimizer_step_id=int(meta.get("optimizer_step_id", 0)),
                    supervision_view_count=int(meta.get("supervision_view_count", 0)),
                    camera_role=_ROLE_MAP.get(meta.get("camera_role"), CameraRole.OTHER),
                    phase=meta.get("phase", MappingPhase.OTHER),
                    n_gaussians=int(meta.get("n_gaussians", 0)),
                    map_version=meta.get("map_version"),
                    psnr_db=meta.get("psnr_db"),
                    frames_seen=meta.get("frames_seen"),
                    frames_ingested=meta.get("frames_ingested"),
                    keyframes_admitted=meta.get("keyframes_admitted"),
                )
                self._viewer.submit(event)
            except Exception:  # noqa: BLE001 - a malformed preview must never affect training
                pass
        lane = self._map_output_lane
        if lane is not None:
            while True:
                output = lane.poll_completed()
                if output is None:
                    break
                try:
                    self._viewer.submit(
                        output.to_preview_event(wall_time_s=time.monotonic())
                    )
                except Exception:  # noqa: BLE001 - output cannot affect mapping
                    pass
        self._viewer.drain_to_sinks()

    # -- new, non-ABC method used directly by the training-pair call site -
    def log_training_pair_tensors(
        self,
        *,
        target: Any,
        render: Any,
        camera_id: int,
        iteration: int,
        source_frame_id: Optional[int] = None,
        camera_role: str = "OTHER",
        phase: MappingPhase = MappingPhase.OTHER,
        n_gaussians: int = 0,
        map_version: Optional[int] = None,
        psnr_db: Optional[float] = None,
        frames_seen: Optional[int] = None,
        frames_ingested: Optional[int] = None,
        keyframes_admitted: Optional[int] = None,
        supervision_view_count: int = 0,
    ) -> None:
        """Stage a matched target/render pair for async submission. Never blocks.

        ``target``/``render`` are torch tensors (H, W, C) uint8 or float,
        CUDA/HIP or CPU -- NOT yet ``.cpu().numpy()``'d by the caller. This is
        the whole point: the D2H happens asynchronously inside the staging
        ring, off the training thread's critical path.
        """
        if not self._viewer.enabled or self._closed:
            return
        self._optimizer_step_id += 1
        meta = {
            "iteration": int(iteration),
            "wall_time_s": time.monotonic(),
            "camera_id": int(camera_id),
            "source_frame_id": source_frame_id,
            "optimizer_step_id": self._optimizer_step_id,
            "supervision_view_count": int(supervision_view_count),
            "camera_role": str(camera_role),
            "phase": phase,
            "n_gaussians": int(n_gaussians),
            "map_version": map_version,
            "psnr_db": psnr_db,
            "frames_seen": frames_seen,
            "frames_ingested": frames_ingested,
            "keyframes_admitted": keyframes_admitted,
        }
        self._ring.try_submit_pair(target, render, meta)
        self._drain.notify()

    def bind_map_output_lane(self, lane: Any) -> None:
        """Attach the sole bounded #1200 map-output lane once."""

        if self._map_output_lane is not None and self._map_output_lane is not lane:
            raise RuntimeError("canonical viewer already has a map output lane")
        if not callable(getattr(lane, "poll_completed", None)):
            raise TypeError("map output lane must provide poll_completed()")
        self._map_output_lane = lane

    def notify_map_output(self) -> None:
        """Wake the off-thread consumer after a non-blocking lane submission."""

        self._drain.notify()

    def log_runtime_telemetry_snapshot(
        self, snapshot: Any, *, iteration: int, wall_time_s: float | None = None
    ) -> None:
        """Publish already-completed scalar runtime telemetry to PreviewBus."""

        if not self._viewer.enabled or self._closed:
            return
        from eligsir.viewers.telemetry_panels import runtime_telemetry_preview

        self._viewer.submit(
            runtime_telemetry_preview(
                snapshot,
                iteration=int(iteration),
                wall_time_s=time.monotonic() if wall_time_s is None else float(wall_time_s),
            )
        )
        self._drain.notify()

    # -- legacy Viewer ABC surface ----------------------------------------
    def log_rgb(self, frame: RGBDFrame, step: int) -> None:
        """Submit the raw RGB frame to the canonical raw-input preview."""
        self._submit_raw(frame, step)

    def log_camera(self, frame: RGBDFrame, pose: Pose, step: int) -> None:
        """Submit the raw frame together with its pose to the raw-input preview."""
        self._submit_raw(frame, step, pose=pose)

    def log_depth(self, frame: RGBDFrame, step: int) -> None:
        """No-op: depth rides on the raw-input preview submitted by ``log_rgb``/``log_camera``."""
        pass  # carried on RawInputPreview.depth via log_rgb/log_camera when available

    def log_rendered(self, rgb: np.ndarray, depth: Optional[np.ndarray], step: int) -> None:
        """No-op: the legacy numpy-only rendered path is not carried by canonical sinks."""
        pass  # legacy numpy-only path; training-pair call sites use
        # log_training_pair_tensors instead (see module docstring)

    def log_gaussian_map(self, positions, colors, scales, step: int) -> None:
        """No-op: map preview transfer is deferred (see #571 follow-up)."""
        pass  # MapPreviewSnapshot transfer deferred to #571 follow-up

    def log_gaussian_model(self, view, step: int, map_version: Optional[int] = None) -> None:
        """No-op: map preview transfer is deferred (see #571 follow-up)."""
        pass  # MapPreviewSnapshot transfer deferred to #571 follow-up

    def log_trajectory(
        self, poses_est: List[Pose], poses_gt: Optional[List[Pose]], step: int
    ) -> None:
        """Publish a bounded snapshot of the estimated (+ optional GT) trajectory.

        ``poses_est``/``poses_gt`` may be the caller's own unboundedly-growing
        running list (see ``train_streaming.py``'s ``_trajectory_est``); only
        the last ``_MAX_TRAJECTORY_POSES`` entries are ever converted, so
        neither this call's cost nor the published event's size grows with
        total run length. Emission rate is separately capped by
        ``CanonicalViewerConfig.trajectory_max_hz`` in ``CanonicalViewer.submit``.
        """
        if not self._viewer.enabled or self._closed:
            return
        if not poses_est:
            return
        try:
            est = _poses_to_array(poses_est[-_MAX_TRAJECTORY_POSES:])
            if est is None:
                return
            gt = _poses_to_array(poses_gt[-_MAX_TRAJECTORY_POSES:]) if poses_gt else None
            event = TrajectoryPreview(
                iteration=int(step),
                wall_time_s=time.monotonic(),
                poses_est=est,
                poses_gt=gt,
            )
            self._viewer.submit(event)
            self._drain.notify()
        except Exception:  # noqa: BLE001
            pass

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Publish scalar metrics to the viewer when it is enabled and not closed."""
        if not self._viewer.enabled or self._closed:
            return
        try:
            event = MetricsPreview(
                iteration=int(step), wall_time_s=time.monotonic(), metrics=dict(metrics)
            )
            self._viewer.submit(event)
        except Exception:  # noqa: BLE001
            pass

    def log_tracking(self, result: TrackingResult, step: int) -> None:
        """No-op: tracking previews are not carried by canonical sinks."""
        pass

    def log_keyframe(self, pose: Pose, intrinsics: CameraIntrinsics, step: int) -> None:
        """No-op: keyframe previews are deferred."""
        pass  # KeyframePreview: deferred, not required for #572's 3 required modes

    def log_controller_stats(self, stats: Any, step: int) -> None:
        """No-op: controller statistics are not carried by canonical sinks."""
        pass

    def set_params_callback(self, cb) -> None:
        """No-op: canonical sinks are observers only and expose no callback channel."""
        pass  # canonical sinks are observers only; no callback channel exists

    def close(self) -> None:
        """Close the adapter and release the underlying viewer sinks."""
        if self._closed:
            return
        self._closed = True
        self._drain.stop()
        self._viewer.close()

    # -- internal ------------------------------------------------------
    def _submit_raw(
        self, frame: RGBDFrame, step: int, *, pose: Optional[Pose] = None
    ) -> None:
        if not self._viewer.enabled or self._closed:
            return
        rgb = getattr(frame, "rgb", None)
        if rgb is None:
            return
        try:
            arr = np.asarray(rgb)
            # Small (4x4) c2w matrix, CPU-side already -- no GPU readback and
            # no meaningful cost added to this hot path. Copied (not just
            # frozen in place) since the caller's ``Pose`` may be reused/
            # mutated after this call returns (see the array-ownership rule
            # in ``events.py``: RawInputPreview freezes what it is handed,
            # it does not copy it).
            pose_c2w = (
                np.array(pose.matrix, dtype=np.float32, copy=True)
                if pose is not None
                else None
            )
            event = RawInputPreview(
                iteration=int(step),
                wall_time_s=time.monotonic(),
                source_frame_id=int(getattr(frame, "frame_id", step) or step),
                timestamp_s=float(getattr(frame, "timestamp", 0.0) or 0.0),
                rgb=arr.copy(),
                camera_id=None,
                pose_c2w=pose_c2w,
            )
            self._viewer.submit(event)
            if pose_c2w is not None:
                self._drain.notify()
        except Exception:  # noqa: BLE001
            pass

    @property
    def canonical_viewer(self) -> CanonicalViewer:
        """Escape hatch for the training-pair call site / tests."""
        return self._viewer
