"""Bounded ownership of live admitted camera pixel payloads.

The canonical live application admits every packet, so the scene's pose-bearing
``train_cameras`` list is intentionally not truncated.  Cameras outside the
current/replay pools still carry a full RGB-D ``StreamingObservation`` and a
native-image loader, however; retaining those payloads makes host memory grow
with trajectory length.  This module drops only those non-replayable payloads.
"""

from __future__ import annotations

from typing import Iterable


def _iter_cameras(value) -> Iterable[object]:
    if value is None:
        return ()
    return value


def camera_payload_ready(cam) -> bool:
    """Return whether *cam* still owns a payload usable by a training render.

    A retired camera intentionally remains in ``train_cameras`` for pose and
    accounting, so callers must not use list membership as a residency test.
    The explicit marker is authoritative; the attribute fallback keeps this
    helper usable by lightweight test doubles and cameras created before the
    marker was introduced.
    """

    if bool(getattr(cam, "_live_payload_retired", False)):
        return False
    payload_attributes = (
        "original_image",
        "_pending_image_loader",
        "_native_image_loader",
        "_streaming_frame",
    )
    # Geometry-only camera doubles and pre-residency camera implementations do
    # not expose payload ownership fields. They have no explicit retirement
    # marker, so preserve the legacy sampling contract for them. Production
    # cameras expose at least one of these fields and therefore still fail
    # closed when every payload owner is absent.
    if not any(hasattr(cam, name) for name in payload_attributes):
        return True
    frame = getattr(cam, "_streaming_frame", None)
    if frame is not None and any(
        getattr(frame, name, None) is not None
        for name in ("_rgb_np", "_depth_np", "_rgb_bytes", "_depth_bytes", "mapping_frame")
    ):
        return True
    return any(
        getattr(cam, name, None) is not None
        for name in ("original_image", "_pending_image_loader", "_native_image_loader")
    )


def payload_ready_cameras(cameras) -> list[object]:
    """Filter a sampling pool to cameras whose pixel payload is available."""

    return [cam for cam in (cameras or ()) if camera_payload_ready(cam)]


def _payload_cameras(scene) -> set[int]:
    """Return camera identities whose payload must remain reloadable."""

    protected: set[int] = set()
    train = list(getattr(scene, "train_cameras", ()) or ())
    policy = getattr(scene, "residency_policy", None)
    if policy is not None:
        hot_window = max(1, int(policy.keyframe_window))
    else:
        args = getattr(scene, "args", None)
        hot_window = max(1, int(args.streaming_keyframe_window))
    protected.update(id(cam) for cam in train[-hot_window:])
    current = getattr(scene, "current_camera", None)
    if current is not None:
        protected.add(id(current))

    # Replay and global reservoir are durable sampling owners.  The covisible,
    # residual and hard-frame views are derived caches and may retain stale
    # camera identities; samplers filter those pools instead of pinning all of
    # their payloads (which would turn a cache into an unbounded owner).
    for name in ("_replay_buffer", "_global_reservoir"):
        protected.update(id(cam) for cam in _iter_cameras(getattr(scene, name, ())))
    return protected


def _drop_camera_payload(cam) -> bool:
    """Drop heavy live RGB-D references while retaining camera geometry/pose."""

    frame = getattr(cam, "_streaming_frame", None)
    had_payload = False
    if frame is not None:
        for name in ("_rgb_np", "_depth_np", "_rgb_bytes", "_depth_bytes"):
            if getattr(frame, name, None) is not None:
                had_payload = True
                setattr(frame, name, None)
        # MappingFrame owns the canonical BufferRef wrappers.  The mapper has
        # consumed them by this point; releasing this local projection avoids
        # retaining the same NumPy arrays through the immutable contract frame.
        if getattr(frame, "mapping_frame", None) is not None:
            frame.mapping_frame = None
    if getattr(cam, "_mapping_frame", None) is not None:
        had_payload = True
        cam._mapping_frame = None

    if getattr(cam, "original_image", None) is not None:
        had_payload = True
        cam.original_image = None
    if getattr(cam, "score_provenance", None) is not None:
        # `score_provenance` is a camera-level attribute (utils/score_contract.py's
        # `camera_provenance`/`score_tensor_from_camera`), not a tensor attribute, so it
        # does NOT get cleared just by dropping `original_image` above. Left behind, it
        # survives to describe whatever pixels a later cold-restore installs -- see the
        # matching fix in `production_application.py::_restore_camera_payload_from_cold`
        # (issue #1259): `camera.score_provenance` and `camera.original_image` must be set
        # and cleared together, always.
        cam.score_provenance = None
    if getattr(cam, "_canonical_native_resolution_state", None) is not None:
        # This state owns the native RGB/depth tensors used to derive lower
        # resolution tiers.  It is payload backing, not immortal camera
        # geometry, and must share the camera's residency lifetime.
        had_payload = True
        cam._canonical_native_resolution_state = None
    cam._canonical_resolution_binding_native_state = None
    cam._canonical_resolution_tier = None
    for name in ("_pending_image_loader", "_native_image_loader"):
        if getattr(cam, name, None) is not None:
            had_payload = True
            setattr(cam, name, None)
    if had_payload:
        cam._live_payload_retired = True
    return had_payload


def mark_payload_residency_dirty(scene) -> None:
    """Mark scene payload pools as requiring a residency enforcement pass."""

    scene._payload_residency_dirty = True


def retire_live_admitted_payloads(scene) -> int:
    """Retire non-replayable live payloads without removing pose records.

    ``streaming_live_admitted_payload_max_frames`` bounds the *global reservoir*
    (which otherwise keeps every preselected live camera replayable forever).  The
    bounded replay ring, current camera, hot window, and other active sampling
    views retain their payloads.  Cameras remain in ``train_cameras`` so pose,
    extent, and admission accounting stay valid.
    """

    # The generation gate is opt-in for the canonical scene. Legacy scene
    # implementations do not mark pool mutations, so they retain their
    # historical always-enforce behavior.
    if hasattr(scene, "_payload_residency_dirty") and not bool(
        scene._payload_residency_dirty
    ):
        return 0
    train = list(getattr(scene, "train_cameras", ()) or ())
    if not train:
        if hasattr(scene, "_payload_residency_dirty"):
            scene._payload_residency_dirty = False
        return 0
    policy = getattr(scene, "residency_policy", None)
    if policy is not None:
        bound = max(1, int(policy.live_payload_max_frames))
    else:
        args = getattr(scene, "args", None)
        bound = max(
            1,
            int(getattr(args, "streaming_live_admitted_payload_max_frames", 512)),
        )

    reservoir = list(getattr(scene, "_global_reservoir", ()) or ())
    if len(reservoir) > bound:
        # Preserve the most recent reservoir entries; older cameras remain in
        # train_cameras for pose history but cease to be replay owners.
        scene._global_reservoir[:] = reservoir[-bound:]

    protected = _payload_cameras(scene)
    retired = 0
    for cam in train:
        if id(cam) in protected:
            continue
        if _drop_camera_payload(cam):
            retired += 1
    if hasattr(scene, "_payload_residency_dirty"):
        scene._payload_residency_dirty = False
    return retired


__all__ = [
    "camera_payload_ready",
    "mark_payload_residency_dirty",
    "payload_ready_cameras",
    "retire_live_admitted_payloads",
]
