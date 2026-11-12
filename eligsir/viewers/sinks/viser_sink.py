"""``ViserPreviewSink``: the primary browser sidecar (issue #572).

Consumes canonical preview events (see ``eligsir.viewers.events``) only --
no ``GaussianModel``, no renderer, no live tensor ever reaches this class (the
``PreviewSink`` contract in ``sinks/base.py`` enforces that structurally). The
actual Viser SDK/websocket work happens in a **separate process**
(``viser_server.py``), started here and fed through independent capacity-one,
latest-wins queues per semantic channel, so no Viser SDK call and no network
I/O ever executes on the training thread.

Scope (per #572's own required UI modes): Training Pair (mode 1, primary),
Raw Input (mode 3), and a same-hook Residual (mode 4, computed sidecar-side
from the exact matched pair -- no extra model work). Causal Progress (mode 2)
uses the natural-render-of-first-admitted-frame rule directly in
``viser_server.py``. Client-side 3D splats (``MapPreviewSnapshot``, issue
#1042) forward a bounded, pre-sampled snapshot -- positions/colors always,
plus scales/quats/opacities when the caller staged them (issue #1042
follow-up, real oriented-ellipsoid rendering via ``add_gaussian_splats``
rather than a uniform-dot point cloud) -- the sink never samples or
transfers anything itself, it only relays the already-staged event.

Live camera + trajectory trail (follow-up to #1385/#1394/#1400's map viewing):
``RawInputPreview.pose_c2w`` (forwarded via ``on_raw_input``, previously
dropped here) drives a persistent camera-frustum scene node; ``on_trajectory``
forwards a bounded ``TrajectoryPreview.poses_est`` array that drives a
persistent trajectory-trail line-segments node. Both scene nodes and the
pose/quaternion math live in ``viser_server.py`` -- this sink only relays the
already-frozen, already-bounded arrays.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

from eligsir.viewers.bus import PreviewEnvelope
from eligsir.viewers.events import (
    KeyframePreview,
    MapPreviewSnapshot,
    MetricsPreview,
    RawInputPreview,
    RuntimeTelemetryPreview,
    TrainingPairPreview,
    TrajectoryPreview,
)
from eligsir.viewers.sinks.base import PreviewSink
from eligsir.viewers.sidecar_ipc import LatestWinsSidecarIPC, create_sidecar_queues

__all__ = ["ViserPreviewSink", "ViserUnavailable"]


class ViserUnavailable(RuntimeError):
    """Raised at sink construction if the Viser sidecar process fails to start."""


class ViserPreviewSink(PreviewSink):
    """Spawns a Viser sidecar process and forwards preview events to it.

    Submission never waits for the Viser process. Each semantic channel keeps
    only its latest payload, so high-rate images cannot evict control telemetry.
    """

    def __init__(
        self,
        *,
        port: int = 8080,
        host: str = "0.0.0.0",
        queue_size: int = 4,
        start_timeout_s: float = 15.0,
    ) -> None:
        ctx = mp.get_context("spawn")
        # ``queue_size`` remains accepted for caller compatibility; the
        # canonical sidecar contract is independently capacity one per channel.
        _ = int(queue_size)
        channels = (
            "training_pair", "raw_input", "trajectory", "metrics",
            "runtime_telemetry", "map_snapshot",
        )
        self._manager = ctx.Manager()
        self._queues = create_sidecar_queues(self._manager, channels)
        self._ipc = LatestWinsSidecarIPC(self._queues)
        self._close_flag = ctx.Event()
        self._ready = ctx.Event()
        self._error_queue = ctx.Queue()
        self._dropped = 0
        self._sent = 0
        self._closed = False
        self.port = int(port)
        self.host = str(host)

        from eligsir.viewers.sinks.viser_server import run_viser_server

        self._process = ctx.Process(
            target=run_viser_server,
            args=(
                self._queues,
                self.port,
                self.host,
                self._ready,
                self._close_flag,
                self._error_queue,
            ),
            daemon=True,
            name="eligsir-viser-sidecar",
        )
        self._process.start()

        # Poll in short slices rather than a single blocking wait so a fast
        # failure (missing 'viser' package, port already in use) fails fast
        # with the real reason instead of always burning the full
        # ``start_timeout_s`` before reporting a generic timeout (#1389).
        deadline = time.monotonic() + float(start_timeout_s)
        poll_interval = 0.1
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=poll_interval):
                break
            if not self._process.is_alive():
                break
        if not self._ready.is_set():
            self.close()
            try:
                detail = self._error_queue.get_nowait()
            except Exception:
                detail = None
            if detail:
                raise ViserUnavailable(f"Viser sidecar failed to start: {detail}")
            raise ViserUnavailable(
                f"Viser sidecar did not report ready within {start_timeout_s}s "
                f"(is 'viser' installed in the sidecar's environment?)"
            )

    # -- PreviewSink hooks: exactly (self, envelope) -> None -------------
    def on_training_pair(self, envelope: PreviewEnvelope) -> None:
        event: TrainingPairPreview = envelope.event
        self._put(
            {
                "kind": "training_pair",
                "target": event.target.array,
                "render": event.render.array,
                "camera_id": event.camera_id,
                "source_frame_id": event.source_frame_id,
                "camera_role": event.camera_role.value,
                "phase": event.phase.value,
                "iteration": event.iteration,
                "psnr_db": event.psnr_db,
                "n_gaussians": event.n_gaussians,
            }
        )

    def on_raw_input(self, envelope: PreviewEnvelope) -> None:
        event: RawInputPreview = envelope.event
        self._put(
            {
                "kind": "raw_input",
                "rgb": event.rgb,
                "source_frame_id": event.source_frame_id,
                "camera_id": event.camera_id,
                "timestamp_s": event.timestamp_s,
                # (4, 4) c2w or None -- drives the live camera frustum in
                # viser_server.py. Previously silently dropped here even
                # though RawInputPreview already carried it (issue: live
                # camera + trajectory trail).
                "pose_c2w": event.pose_c2w,
            }
        )

    def on_trajectory(self, envelope: PreviewEnvelope) -> None:
        event: TrajectoryPreview = envelope.event
        self._put(
            {
                "kind": "trajectory",
                "poses_est": event.poses_est,
                "poses_gt": event.poses_gt,
                "iteration": event.iteration,
            }
        )

    def on_metrics(self, envelope: PreviewEnvelope) -> None:
        event: MetricsPreview = envelope.event
        self._put({"kind": "metrics", "metrics": event.as_dict()})

    def on_runtime_telemetry(self, envelope: PreviewEnvelope) -> None:
        event: RuntimeTelemetryPreview = envelope.event
        self._put({"kind": "runtime_telemetry", "panels": event.panels.as_wire()})

    def on_keyframe(self, envelope: PreviewEnvelope) -> None:
        pass  # reserved; not required for #572's Training Pair / Causal / Raw modes

    def on_map_snapshot(self, envelope: PreviewEnvelope) -> None:
        event: MapPreviewSnapshot = envelope.event
        self._put(
            {
                "kind": "map_snapshot",
                "positions": event.positions,
                "colors": event.colors,
                "scales": event.scales,
                "quats": event.quats,
                "opacities": event.opacities,
                "map_version": event.map_version,
                "n_total_gaussians": event.n_total_gaussians,
                "n_sampled": event.n_sampled,
                "iteration": event.iteration,
                "timestamp": event.wall_time_s,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_flag.set()
        if self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
        self._manager.shutdown()

    # -- internal ----------------------------------------------------------
    def _put(self, payload: dict) -> None:
        try:
            if self._ipc.publish(payload["kind"], payload):
                self._sent += 1
            else:
                self._dropped += 1
        except Exception:  # broken sidecar or malformed internal payload
            self._dropped += 1

    @property
    def replaced(self) -> int:
        return self._ipc.replaced

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def sent(self) -> int:
        return self._sent
