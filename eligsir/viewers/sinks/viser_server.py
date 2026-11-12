"""Viser sidecar server process (issue #572).

Runs in a SEPARATE PROCESS from the trainer, per the #571/#572 sidecar
requirement ("remote clients connect to the sidecar, not the training
process"; "no Rerun/Viser SDK call from the training thread" generalizes to
Viser). This module imports ``viser`` and nothing training-related except the
preview-event field *names* it reads out of plain dict payloads handed to it
over a bounded queue -- it never imports ``torch``, ``train_streaming``, or any
Gaussian model type.

Started by :class:`eligsir.viewers.sinks.viser_sink.ViserPreviewSink` as a
``multiprocessing.Process``. Communication is one-way (trainer-side process ->
this process) over independent capacity-one queues per semantic channel; the
sender replaces stale values and never waits for this process.
"""
from __future__ import annotations

import queue as _queue_mod
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["run_viser_server", "MapSnapshot"]

# Map Preview "View Mode" dropdown options (coordinator follow-up to the
# #1042 ellipsoid fix: a three-way toggle, not a two-way one).
_MAP_MODE_GAUSSIAN = "Gaussian Map"  # the real reconstruction (splats)
_MAP_MODE_CENTERS = "Gaussian Centers"  # density-debugging point cloud
_MAP_MODE_METADATA = "Metadata"  # future scope: color by lifecycle/age/etc.

# Live camera frustum defaults (no real intrinsics travel on RawInputPreview
# today -- see the "Live camera + trajectory trail" note in _build_ui).
_DEFAULT_LIVE_CAMERA_VFOV_RAD = 1.0471975511965976  # 60 degrees, vertical
_DEFAULT_LIVE_CAMERA_ASPECT = 4.0 / 3.0

# Trajectory-trail history bound, mirrored here as a second line of defense
# on top of canonical_adapter.py's own ``_MAX_TRAJECTORY_POSES`` cap -- a
# stale/older trainer build could still hand this sidecar an unbounded
# array, and this process must not grow its own scene-node buffer without
# bound just because an upstream cap was skipped.
_MAX_TRAJECTORY_TRAIL_POINTS = 4000


@dataclass
class MapSnapshot:
    """Typed, sidecar-side view of a ``map_snapshot`` wire payload (issue
    #1042 follow-up).

    Mirrors :class:`eligsir.viewers.events.MapPreviewSnapshot` -- the
    canonical, training-side event this is built from -- but is defined
    independently here (not imported) because this module runs in a
    separate process and deliberately never imports anything training-side
    (see this module's docstring: no ``torch``, no Gaussian model type).
    ``MapSnapshot.from_cmd`` is the one place that unpacks the raw dict
    handed over the queue (see ``viser_sink.py``'s ``on_map_snapshot`` for
    the producer side) into this documented shape, replacing ad-hoc
    ``cmd.get(...)`` calls scattered through ``_apply``.

    Fields:
        centers: (N, 3) float32 world-space positions.
        scales: (N, 3) float32 per-axis extents, or ``None`` if the
            snapshot has no scale source (older call sites, or a model
            layout without one).
        rotations: (N, 4) float32 unit quaternions, **wxyz** convention --
            matches this repo's own ``GsplatGaussianModel`` (its identity
            quat init is ``quats[:, 0] = 1``) -- or ``None`` alongside
            ``scales``.
        opacities: (N, 1) float32 in [0, 1], or ``None`` (defaults to fully
            opaque, see :meth:`from_cmd`).
        colors: (N, 3) float32 in [0, 1], DC-only SH color
            (``rgb = 0.5 + SH_C0 * f_dc``, the same formula
            ``train_streaming.py``'s map-snapshot call site already uses),
            or ``None`` (defaults to mid-gray, see :meth:`from_cmd`).
        iteration: training iteration this snapshot was sampled at.
        timestamp: wall-clock time (seconds, ``time.monotonic()`` domain)
            the snapshot was staged.
        gaussian_count: total LIVE Gaussian count in the model this
            snapshot was sampled from -- may exceed ``centers.shape[0]``
            when the model was sampled down (see :attr:`n_sampled` /
            :attr:`is_complete`).
    """

    centers: Any  # np.ndarray, (N, 3) float32
    scales: Optional[Any]
    rotations: Optional[Any]
    opacities: Optional[Any]
    colors: Optional[Any]
    iteration: int
    timestamp: float
    gaussian_count: int

    @property
    def n_sampled(self) -> int:
        return int(self.centers.shape[0])

    @property
    def is_complete(self) -> bool:
        return self.n_sampled >= self.gaussian_count

    @property
    def has_shape(self) -> bool:
        """True iff this snapshot carries real scale+rotation (so a
        Gaussian-splat renderer can draw true oriented ellipsoids instead of
        falling back to a small isotropic covariance)."""
        return self.scales is not None and self.rotations is not None

    @classmethod
    def from_cmd(cls, cmd: Dict[str, Any]) -> "MapSnapshot":
        """Parse the raw wire dict (see ``viser_sink.py``'s ``on_map_snapshot``)
        into a typed :class:`MapSnapshot`. Applies the same missing-source
        defaults the old ad-hoc dict-pulling code used: mid-gray for missing
        colors, fully opaque for missing opacities. ``scales``/``rotations``
        are left ``None`` (not defaulted) when absent -- callers use
        :attr:`has_shape` to decide whether to render real ellipsoids or
        fall back to an isotropic covariance."""
        import numpy as np

        centers = np.asarray(cmd["positions"], dtype="float32")
        n = centers.shape[0]

        colors = cmd.get("colors")
        colors = np.asarray(colors, dtype="float32") if colors is not None else None

        scales = cmd.get("scales")
        scales = np.asarray(scales, dtype="float32") if scales is not None else None

        rotations = cmd.get("quats")
        rotations = np.asarray(rotations, dtype="float32") if rotations is not None else None

        opacities = cmd.get("opacities")
        opacities = (
            np.asarray(opacities, dtype="float32").reshape(n, 1)
            if opacities is not None
            else None
        )

        return cls(
            centers=centers,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors,
            iteration=int(cmd.get("iteration", 0) or 0),
            timestamp=float(cmd.get("timestamp", 0.0) or 0.0),
            gaussian_count=int(cmd.get("n_total_gaussians", n)),
        )


def run_viser_server(
    cmd_queues, port: int, host: str, ready_flag=None, close_flag=None, error_queue=None
) -> None:
    """Blocking entry point for the sidecar process.

    ``cmd_queues`` yields plain dicts (see ``viser_sink.py`` for the wire
    schema), independently by semantic channel. ``close_flag`` shuts down.
    ``ready_flag``, if given, is a ``multiprocessing.Event`` set once the
    server is listening, so the parent can confirm startup without polling
    the port. ``error_queue``, if given, receives a single descriptive
    string and this function returns (without setting ``ready_flag``)
    whenever startup fails -- e.g. ``viser`` not being importable in this
    process's environment, or the requested port being unavailable (issue
    #1389). This lets the parent (:class:`~.viser_sink.ViserPreviewSink`)
    fail fast with the real reason instead of always waiting out the full
    ``start_timeout_s`` and reporting a generic timeout.
    """
    try:
        import viser  # imported only in this process -- never in the trainer
    except ImportError as exc:
        if error_queue is not None:
            error_queue.put(
                f"'viser' package is not importable in the sidecar process ({exc}). "
                "Install it in the training image (see the Dockerfile's "
                "INSTALL_VISER build arg) or drop --viewer viser."
            )
        return

    try:
        server = viser.ViserServer(host=host, port=port)
    except Exception as exc:  # e.g. port already in use
        if error_queue is not None:
            error_queue.put(f"failed to start viser.ViserServer on {host}:{port}: {exc}")
        return

    handles, texts, _map_mode_dropdown, _on_map_mode_change = _build_ui(server)

    if ready_flag is not None:
        ready_flag.set()

    _drain_forever(cmd_queues, handles, texts, close_flag)


def _build_ui(server):
    """Construct every GUI control and scene node, wire the Map Preview
    render-mode toggle, and return ``(handles, texts, map_mode_dropdown,
    on_map_mode_change)``.

    Split out of :func:`run_viser_server` (which just blocks forever after
    calling this) so the GUI/scene wiring -- including the Gaussian-splats
    vs point-cloud toggle callback -- is directly testable against a real
    ``viser.ViserServer`` without spawning the sidecar subprocess or driving
    an actual browser client (see
    ``tests/test_viser_server_map_mode_toggle.py``).
    """
    import numpy as np

    server.gui.add_markdown("# EliGSiR — Live Training Pair")
    status_md = server.gui.add_markdown("_waiting for first preview event..._")

    with server.gui.add_folder("Training Pair", expand_by_default=True):
        target_handle = server.gui.add_image(_placeholder(), label="INPUT / TARGET")
        render_handle = server.gui.add_image(_placeholder(), label="MAP RENDER")
        residual_handle = server.gui.add_image(_placeholder(), label="RESIDUAL |target-render|")

    with server.gui.add_folder("Causal Progress", expand_by_default=False):
        causal_handle = server.gui.add_image(_placeholder(), label="Advancing reconstruction")
        causal_md = server.gui.add_markdown("_no admitted frame rendered yet_")

    with server.gui.add_folder("Raw Input", expand_by_default=False):
        raw_handle = server.gui.add_image(_placeholder(), label="Newest sensor frame")
        raw_md = server.gui.add_markdown("_no raw input yet_")

    # Bounded 3D map preview (issue #1042, ellipsoid follow-up + three-way
    # view-mode toggle). TWO real scene nodes are kept alive side by side --
    # a Gaussian-splat node (the real reconstruction: center/covariance/
    # RGB/opacity, updated via ``GaussianSplatHandle.set_gaussians``, an
    # atomic buffer replace -- the preferred fast path for a per-frame
    # Gaussian-count-changing update per viser's own docs, not a fresh
    # ``add_gaussian_splats`` call each snapshot) and a point-cloud node
    # (the original #1042 rendering, explicitly reframed as a density-
    # debugging view -- "Gaussian Centers", not "the reconstruction"). Both
    # are fed identically on EVERY snapshot regardless of which is
    # currently shown (see the map_snapshot branch of :func:`_apply`), so
    # flipping the dropdown below is instant (no re-fetch, no reload) and
    # never leaves the non-visible node stale. Neither node is ever
    # removed/re-added, so a slow/disconnected client never accumulates
    # duplicates. A third mode, "Metadata" (color-by-lifecycle/age/utility/
    # submap/birth_frame), is real future scope -- deliberately a
    # documented no-op stub here: both scene nodes hidden, a markdown note
    # explains why. Both scene nodes start as a single point/gaussian at
    # the origin; the first real snapshot replaces both.
    map_gaussians_handle = server.scene.add_gaussian_splats(
        "/map_preview_gaussians",
        centers=np.zeros((1, 3), dtype="float32"),
        covariances=_default_covariances(1),
        rgbs=np.full((1, 3), 0.5, dtype="float32"),
        opacities=np.ones((1, 1), dtype="float32"),
    )
    map_points_handle = server.scene.add_point_cloud(
        "/map_preview_points",
        points=np.zeros((1, 3), dtype="float32"),
        colors=np.zeros((1, 3), dtype="uint8"),
        point_size=0.02,
        visible=False,  # "Gaussian Map" is the default-shown mode
    )

    # Live camera frustum + trajectory trail (follow-up to #1385/#1394/#1400's
    # map viewing). Both are single, persistent scene nodes updated in place
    # (position/wxyz for the frustum, .points for the trail) -- never
    # removed/re-added -- exactly the pattern the map-preview nodes above
    # already use. Both start invisible: there is no real pose yet, and an
    # OpenCV-convention frustum/trail sitting at the identity pose at the
    # origin would look like a (wrong) real observation rather than "no data
    # yet". The first real ``raw_input``/``trajectory`` command makes each
    # visible. FOV/aspect are a fixed approximation (no intrinsics travel on
    # ``RawInputPreview`` today) refined from the live frame's own aspect
    # ratio once available -- good enough for "where is the camera and which
    # way is it facing", not a to-scale FOV cone.
    live_camera_handle = server.scene.add_camera_frustum(
        "/camera/live",
        fov=_DEFAULT_LIVE_CAMERA_VFOV_RAD,
        aspect=_DEFAULT_LIVE_CAMERA_ASPECT,
        scale=0.15,
        color=(255, 200, 0),
        visible=False,
    )
    trajectory_handle = server.scene.add_line_segments(
        "/camera/trajectory",
        points=np.zeros((1, 2, 3), dtype=np.float32),
        colors=(0, 200, 255),
        thickness=0.01,
        visible=False,
    )

    with server.gui.add_folder("Map Preview", expand_by_default=False):
        map_md = server.gui.add_markdown("_no map snapshot yet_")
        map_mode_dropdown = server.gui.add_dropdown(
            "View Mode",
            options=(_MAP_MODE_GAUSSIAN, _MAP_MODE_CENTERS, _MAP_MODE_METADATA),
            initial_value=_MAP_MODE_GAUSSIAN,
        )
        map_metadata_note_md = server.gui.add_markdown("")

    @map_mode_dropdown.on_update
    def _on_map_mode_change(_event) -> None:  # noqa: ANN001 - viser GuiEvent
        # Toggle visibility on the already-fed handles -- never remove/
        # re-add a scene node, never require a client-side reload. A
        # CLIENT-driven dropdown change (the real browser path, as opposed
        # to a server-side ``dropdown.value = ...`` assignment) runs this
        # callback on viser's own GUI thread pool (see
        # ``GuiApi._handle_gui_updates``), concurrently with whatever
        # ``_apply`` is doing to these same two handles on the drain
        # thread -- so this print is permanent, not throttled: unlike the
        # high-frequency ``map_snapshot`` diagnostic, a mode change is a
        # rare, human-paced action, and this is the one piece of evidence
        # that lets a "the dropdown does nothing" report (browser session,
        # no devtools access) be told apart, from train.log alone, between
        # "the callback never fired / threw" and "it ran and set the
        # handles correctly, so the gap is client-side rendering" -- see
        # the investigation in the #1047 PR thread that added this.
        mode = map_mode_dropdown.value
        map_gaussians_handle.visible = mode == _MAP_MODE_GAUSSIAN
        map_points_handle.visible = mode == _MAP_MODE_CENTERS
        map_metadata_note_md.content = (
            "_Metadata coloring (by lifecycle/age/utility/submap/birth_frame) "
            "is not yet implemented -- both map scene nodes are hidden in "
            "this mode. Pick **Gaussian Map** or **Gaussian Centers** to see "
            "the live map._"
            if mode == _MAP_MODE_METADATA
            else ""
        )
        print(
            f"[viser-server] map view-mode changed -> {mode!r} "
            f"(map_gaussians.visible={map_gaussians_handle.visible}, "
            f"map_points.visible={map_points_handle.visible})",
            flush=True,
        )

    with server.gui.add_folder("Live Stats", expand_by_default=True):
        stats_md = server.gui.add_markdown("_waiting for first metrics event..._")

    telemetry_texts = {}
    for key, label in (
        ("decision", "Decision"),
        ("freshness", "Freshness / Frame Flow"),
        ("gpu", "GPU / Transfer"),
        ("residency", "Map / Residency"),
    ):
        with server.gui.add_folder(label, expand_by_default=key == "decision"):
            telemetry_texts[key] = server.gui.add_markdown("_waiting for runtime telemetry..._")

    handles: Dict[str, Any] = {
        "target": target_handle,
        "render": render_handle,
        "residual": residual_handle,
        "causal": causal_handle,
        "raw": raw_handle,
        "map_gaussians": map_gaussians_handle,
        "map_points": map_points_handle,
        "live_camera": live_camera_handle,
        "trajectory": trajectory_handle,
    }
    texts: Dict[str, Any] = {
        "status": status_md,
        "causal": causal_md,
        "raw": raw_md,
        "map": map_md,
        "map_metadata_note": map_metadata_note_md,
        "stats": stats_md,
        **telemetry_texts,
    }

    return handles, texts, map_mode_dropdown, _on_map_mode_change


def _placeholder():
    import numpy as np

    return np.zeros((2, 2, 3), dtype="uint8")


def _quat_wxyz_to_rotmat(quats):
    """(N, 4) unit quaternions, wxyz convention (this repo's/gsplat's own --
    see ``GsplatGaussianModel``'s identity-quat init ``quats[:, 0] = 1``) ->
    (N, 3, 3) rotation matrices."""
    import numpy as np

    q = np.asarray(quats, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    q = q / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_quat_wxyz(R):
    """(3, 3) rotation matrix -> wxyz quaternion (numerically robust Shepperd).

    Mirrors ``eligsir/viewers/rvizsplat_export.py``'s
    ``_rotmat_to_quat_wxyz_single`` (same algorithm, kept as an independent
    copy here since this module deliberately never imports anything outside
    the viser sidecar's own process -- see this module's docstring). viser's
    ``add_camera_frustum``/scene-node ``wxyz`` follows the same wxyz
    convention as this repo's own gsplat quaternions (see
    ``_quat_wxyz_to_rotmat`` above), so no reordering is needed at the call
    site beyond this conversion.
    """
    import numpy as np

    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array(
        [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    )


def _update_live_camera_frustum(handles, texts, pose_c2w, *, source_label: str, rgb=None) -> None:
    """Update the persistent ``/camera/live`` frustum in place from a c2w pose.

    Shared by both producers of a live pose: ``raw_input`` (legacy
    ``train_streaming.py`` path, which has a real image and so also refines
    ``aspect`` from it) and ``trajectory`` (the ``causal_mapping`` production
    path, which has no per-frame image at the point it observes a pose --
    see the "trajectory" branch of :func:`_apply`).
    """
    import numpy as np

    m = np.asarray(pose_c2w, dtype=np.float64)
    position = m[:3, 3].astype(np.float32)
    wxyz = _rotmat_to_quat_wxyz(m[:3, :3]).astype(np.float32)
    live_camera = handles["live_camera"]
    live_camera.position = position
    live_camera.wxyz = wxyz
    rgb_shape = getattr(rgb, "shape", None)
    if rgb_shape is not None and len(rgb_shape) >= 2 and rgb_shape[0] > 0:
        live_camera.aspect = float(rgb_shape[1]) / float(rgb_shape[0])
    live_camera.visible = True
    _pn = texts.get("_live_camera_count", 0) + 1
    texts["_live_camera_count"] = _pn
    if _pn <= 3 or _pn % 50 == 0:
        print(
            f"[viser-server] live_camera #{_pn} {source_label} "
            f"position={position.tolist()}",
            flush=True,
        )


def _covariances_from_scale_quat(scales, quats):
    """(N, 3) per-axis scale + (N, 4) wxyz quaternion -> (N, 3, 3) covariance
    ``R @ diag(scale**2) @ R.T``, the standard 3DGS second-moment
    construction (issue #1042 follow-up: real oriented ellipsoids instead of
    a uniform-size dot cloud)."""
    import numpy as np

    R = _quat_wxyz_to_rotmat(quats)
    s = np.asarray(scales, dtype=np.float64)
    scaled_axes = R * s[:, None, :]  # scale each rotated axis (column) by its extent
    cov = np.matmul(scaled_axes, scaled_axes.transpose(0, 2, 1))
    return cov.astype(np.float32, copy=False)


def _default_covariances(n: int, radius: float = 0.02):
    """Isotropic small covariance -- fallback for a snapshot with no
    scale/quat source, mirroring the old fixed ``point_size=0.02`` dot-cloud
    look rather than refusing the snapshot outright."""
    import numpy as np

    cov = np.zeros((n, 3, 3), dtype=np.float32)
    idx = np.arange(3)
    cov[:, idx, idx] = radius * radius
    return cov


def _to_uint8_255(image):
    """Scale an image array to uint8 [0, 255], mirroring viser's own
    ``colors_to_uint8`` convention (float assumed [0, 1], integer assumed
    already [0, 255])."""
    import numpy as np

    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image * 255.0, 0, 255).astype("uint8")
    return np.clip(image, 0, 255).astype("uint8")


def _drain_forever(cmd_queues, handles: Dict[str, Any], texts: Dict[str, Any], close_flag=None) -> None:
    while close_flag is None or not close_flag.is_set():
        found = False
        for cmd_queue in cmd_queues.values():
            try:
                cmd: Optional[Dict[str, Any]] = cmd_queue.get_nowait()
            except _queue_mod.Empty:
                continue
            found = True
            try:
                _apply(cmd, handles, texts)
            except Exception as exc:  # noqa: BLE001 - sidecar must never crash on a bad payload
                texts["status"].content = f"_sink error: {type(exc).__name__}: {exc}_"
        if not found:
            time.sleep(0.01)


def _apply(cmd: Dict[str, Any], handles: Dict[str, Any], texts: Dict[str, Any]) -> None:
    kind = cmd.get("kind")
    if kind == "training_pair":
        import numpy as np

        target = cmd["target"]
        render = cmd["render"]
        handles["target"].image = target
        handles["render"].image = render
        if target.shape == render.shape:
            # target/render arrive as float32 in [0, 1] (viser's own
            # colors_to_uint8 scales those correctly for the .image setter
            # above). But an int16 cast on raw [0, 1] floats truncates every
            # value below 1.0 to 0 -- the residual panel was always solid
            # black. Scale to the 0-255 domain first, matching what the
            # image widgets actually display, before taking the abs-diff.
            _target_255 = _to_uint8_255(target)
            _render_255 = _to_uint8_255(render)
            residual = np.abs(
                _target_255.astype("int16") - _render_255.astype("int16")
            ).astype("uint8")
            handles["residual"].image = residual
            # Permanent, throttled diagnostic: proves from train.log alone
            # that target/render/residual are real, non-degenerate image
            # data (not blank/all-zero), without printing every event.
            _n = texts.get("_pair_count", 0) + 1
            texts["_pair_count"] = _n
            if _n <= 3 or _n % 50 == 0:
                print(
                    f"[viser-server] training_pair #{_n} iter={cmd.get('iteration')} "
                    f"target[min={int(_target_255.min())},max={int(_target_255.max())}] "
                    f"render[min={int(_render_255.min())},max={int(_render_255.max())}] "
                    f"residual[min={int(residual.min())},max={int(residual.max())},"
                    f"mean={float(residual.mean()):.2f}]",
                    flush=True,
                )
        role = cmd.get("camera_role", "OTHER")
        texts["status"].content = (
            f"camera {cmd.get('camera_id')} | role **{role}** | phase {cmd.get('phase')} | "
            f"iter {cmd.get('iteration')} | PSNR "
            f"{cmd.get('psnr_db', 'n/a')} | n_gaussians {cmd.get('n_gaussians', 'n/a')}"
        )
        # Causal progress: only advance on a genuinely NEW/monotonic admitted
        # frame's naturally occurring render -- never a synthetic re-render.
        src_id = cmd.get("source_frame_id")
        if role == "NEW" and src_id is not None:
            last = texts.get("_last_causal_src", -1)
            if src_id > last:
                handles["causal"].image = render
                texts["causal"].content = f"source frame **{src_id}** (iter {cmd.get('iteration')})"
                texts["_last_causal_src"] = src_id
                print(
                    f"[viser-server] causal-progress advanced to source_frame_id="
                    f"{src_id} iter={cmd.get('iteration')}",
                    flush=True,
                )
    elif kind == "raw_input":
        handles["raw"].image = cmd["rgb"]
        texts["raw"].content = (
            f"source frame {cmd.get('source_frame_id')} | camera "
            f"{cmd.get('camera_id')} | t={cmd.get('timestamp_s'):.3f}s"
        )
        _rn = texts.get("_raw_count", 0) + 1
        texts["_raw_count"] = _rn
        if _rn <= 3 or _rn % 25 == 0:
            _rgb = cmd["rgb"]
            print(
                f"[viser-server] raw_input #{_rn} source_frame_id="
                f"{cmd.get('source_frame_id')} shape={getattr(_rgb, 'shape', None)} "
                f"min={float(_rgb.min()):.3f} max={float(_rgb.max()):.3f}",
                flush=True,
            )
        pose_c2w = cmd.get("pose_c2w")
        if pose_c2w is not None:
            _update_live_camera_frustum(
                handles, texts, pose_c2w,
                source_label=f"source_frame_id={cmd.get('source_frame_id')}",
                rgb=cmd.get("rgb"),
            )
    elif kind == "trajectory":
        import numpy as np

        poses_est = cmd.get("poses_est")
        if poses_est is not None and len(poses_est) >= 2:
            poses_est = np.asarray(poses_est, dtype=np.float64)[
                -_MAX_TRAJECTORY_TRAIL_POINTS:
            ]
            translations = poses_est[:, :3, 3].astype(np.float32)
            segments = np.stack([translations[:-1], translations[1:]], axis=1)
            handles["trajectory"].points = segments
            handles["trajectory"].visible = True
            _tn = texts.get("_trajectory_count", 0) + 1
            texts["_trajectory_count"] = _tn
            if _tn <= 3 or _tn % 20 == 0:
                print(
                    f"[viser-server] trajectory #{_tn} iter={cmd.get('iteration')} "
                    f"n_poses={translations.shape[0]}",
                    flush=True,
                )
            # causal_mapping's real production path (eligsir/runtime/
            # causal_entry.py) only ever emits "trajectory" -- it has no
            # per-frame image at the point it observes a pose, so it can
            # never emit "raw_input" with a pose_c2w. Drive the live-camera
            # frustum from the trail's own most recent pose here too, so the
            # frustum still appears/moves on that path (train_streaming.py's
            # legacy path, which DOES have an image, additionally drives it
            # via "raw_input" above -- whichever arrives last wins, and both
            # describe the same real camera pose).
            _update_live_camera_frustum(
                handles, texts, poses_est[-1],
                source_label=f"iter={cmd.get('iteration')}",
            )
    elif kind == "metrics":
        m = cmd.get("metrics") or {}
        lines = ["**Live Stats**", ""]
        if "it_per_s" in m:
            lines.append(f"- it/s: {m['it_per_s']:.1f}")
        if "map_fps" in m:
            lines.append(f"- mapFPS: {m['map_fps']:.2f}")
        if "it_per_view" in m:
            lines.append(f"- it/view: {m['it_per_view']:.0f}")
        if "admission_pct" in m:
            lines.append(f"- admission: {m['admission_pct']:.0f}%")
        if "keyframes_admitted" in m:
            lines.append(f"- keyframes admitted: {m['keyframes_admitted']:.0f}")
        if "gaussian_count" in m:
            lines.append(f"- Gaussians: {m['gaussian_count']:.0f}")
        if "frames_ingested" in m:
            lines.append(f"- frames ingested: {m['frames_ingested']:.0f}")
        if "psnr" in m:
            lines.append(f"- PSNR: {m['psnr']:.2f} dB")
        if "l1_loss" in m:
            lines.append(f"- L1 loss: {m['l1_loss']:.5f}")
        if "commit_hz" in m or "admitted_hz" in m:
            lines.append("")
            lines.append("**Mapping clock vs. keyframe clock**")
            commit = m.get("commit_hz")
            admitted = m.get("admitted_hz")
            src = m.get("source_hz")
            kf_rt = m.get("keyframe_realtime_factor")
            if admitted is not None:
                lines.append(f"- keyframe (admitted) rate: {admitted:.2f} Hz")
            if commit is not None:
                lines.append(f"- map commit rate: {commit:.2f} Hz")
            if kf_rt is not None:
                verdict = "KEEPING UP" if kf_rt >= 1.0 else "FALLING BEHIND"
                lines.append(f"- commit/keyframe ratio: {kf_rt:.3f}x ({verdict})")
            if "keyframe_pct" in m:
                lines.append(f"- keyframes / input frames: {m['keyframe_pct']:.0f}%")
            if "map_age_ms" in m:
                lines.append(f"- map freshness: {m['map_age_ms']:.0f} ms old")
            if src is not None:
                lines.append(f"- (context) raw input rate: {src:.2f} Hz")
            if "realtime_factor" in m:
                lines.append(f"- (context) dataset-pace factor: {m['realtime_factor']:.3f}x")
        texts["stats"].content = "\n".join(lines)
    elif kind == "runtime_telemetry":
        from eligsir.viewers.telemetry_panels import format_panel

        panels = cmd.get("panels") or {}
        for key, title in (
            ("decision", "Decision"),
            ("freshness", "Freshness / Frame Flow"),
            ("gpu", "GPU / Transfer"),
            ("residency", "Map / Residency"),
        ):
            texts[key].content = format_panel(panels.get(key, {}), title)
    elif kind == "map_snapshot":
        import numpy as np

        snap = MapSnapshot.from_cmd(cmd)
        n = snap.n_sampled

        rgbs = snap.colors if snap.colors is not None else np.full((n, 3), 0.5, dtype="float32")
        opac = snap.opacities if snap.opacities is not None else np.ones((n, 1), dtype="float32")

        if snap.has_shape:
            covariances = _covariances_from_scale_quat(snap.scales, snap.rotations)
        else:
            # No scale/quat source -- isotropic fallback (old point-cloud look).
            covariances = _default_covariances(n)

        # Feed BOTH map-preview nodes unconditionally, regardless of which
        # is currently visible -- see the three-way "View Mode" dropdown
        # toggle set up at startup. This is what makes flipping the toggle
        # instant: neither node is ever stale when the client switches to
        # it. "Gaussian Map" gets the real reconstruction (set_gaussians,
        # an atomic buffer replace -- not a fresh add_gaussian_splats call
        # each snapshot); "Gaussian Centers" gets the same positions/colors
        # as a density-debugging point cloud.
        handles["map_gaussians"].set_gaussians(snap.centers, covariances, rgbs, opac)
        handles["map_points"].points = snap.centers
        handles["map_points"].colors = _to_uint8_255(rgbs)

        n_sampled = int(cmd.get("n_sampled", n))
        n_total = snap.gaussian_count
        shape_label = "oriented ellipsoids" if snap.has_shape else "isotropic (no scale/quat)"
        texts["map"].content = (
            f"map v{cmd.get('map_version')} | iter {snap.iteration} | showing "
            f"{n_sampled}/{n_total} gaussians ({shape_label}) | Gaussian Map "
            f"and Gaussian Centers views both updated -- pick either (or "
            f"Metadata, not yet implemented) from the View Mode dropdown above"
            + (" (full map)" if n_sampled >= n_total else " (sampled)")
        )
        # Permanent diagnostic: proves from train.log alone that the
        # rendered ellipsoids are real and non-degenerate/non-uniform (the
        # #1042 follow-up's whole point) -- eigenvalue spread of the actual
        # covariance batch just sent to the client, not a hand-wave.
        if snap.has_shape and n > 0:
            eigvals = np.linalg.eigvalsh(covariances.astype(np.float64))
            print(
                f"[viser-server] map_snapshot applied v={cmd.get('map_version')} "
                f"n_sampled={n_sampled} n_total={n_total} shape={shape_label} "
                f"cov_eig[min={eigvals.min():.6f},max={eigvals.max():.6f},"
                f"mean={eigvals.mean():.6f},std={eigvals.std():.6f}]",
                flush=True,
            )
        else:
            print(
                f"[viser-server] map_snapshot applied v={cmd.get('map_version')} "
                f"n_sampled={n_sampled} n_total={n_total} shape={shape_label}",
                flush=True,
            )
