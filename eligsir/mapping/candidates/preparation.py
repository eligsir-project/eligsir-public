"""CPU-side frame preparation: decode + resize + blur/exposure/valid-depth
fraction + cheap pose novelty (plan section 8.4, "Create a CPU/pinned-memory
preparation worker").

Reuses the pure quality logic in
``eligsir.mapping.scheduler.frame_quality`` (blur/exposure/depth-valid
scoring, already GPU/torch-free) and the existing decode helpers in
``eligsir.io.sources.decode_cache`` (``load_frame_rgb``/
``load_frame_depth_np``, format-aware: numpy array / bytes / path) rather
than reimplementing decode or scoring. Nothing in this module touches torch
or the GPU -- see ``contracts.py``'s module docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from eligsir.contracts.types import Pose
from eligsir.mapping.candidates.contracts import (
    InputEvidence,
    PreparedDepthRef,
    PreparedFrameCandidate,
    PreparedImageRef,
    TrackedFramePacket,
    UndecodedBufferRef,
)
from eligsir.mapping.scheduler.frame_quality import (
    FrameQualityConfig,
    evaluate_frame_quality,
)


@dataclass
class _UndecodedBufferRefFrameShim:
    """Adapts a :class:`~contracts.UndecodedBufferRef` pair to the attribute shape
    ``load_frame_rgb``/``load_frame_depth_np`` expect (a
    ``StreamingRGBDFrame``-like object), so this module reuses their
    format-aware decode logic (array / bytes / path, in-memory ScanNet
    ``.sens`` header handling, decode-cache integration) instead of
    duplicating it."""

    rgb_path: Optional[str]
    depth_path: Optional[str]
    _rgb_bytes: Optional[bytes]
    _rgb_np: Optional[np.ndarray]
    _depth_bytes: Optional[bytes]
    _depth_np: Optional[np.ndarray]
    _sens_header: Optional[dict] = None
    _cache_pos: Optional[int] = None


def _shim_from_refs(rgb_ref: UndecodedBufferRef, depth_ref: UndecodedBufferRef) -> _UndecodedBufferRefFrameShim:
    return _UndecodedBufferRefFrameShim(
        rgb_path=rgb_ref.path,
        depth_path=depth_ref.path,
        _rgb_bytes=rgb_ref.data,
        _rgb_np=rgb_ref.array,
        _depth_bytes=depth_ref.data,
        _depth_np=depth_ref.array,
    )


def decode_rgb(rgb_ref: UndecodedBufferRef) -> Optional[np.ndarray]:
    """Decode ``rgb_ref`` to an ``HxWx3`` uint8 array, or ``None`` if empty."""
    if rgb_ref.is_empty():
        return None
    from eligsir.io.sources.decode_cache import load_frame_rgb

    shim = _shim_from_refs(rgb_ref, UndecodedBufferRef())
    return np.asarray(load_frame_rgb(shim))


def decode_depth_m(depth_ref: UndecodedBufferRef, depth_scale: float) -> Optional[np.ndarray]:
    """Decode ``depth_ref`` to an ``HxW`` float32 array in meters, or
    ``None`` if empty/unavailable."""
    if depth_ref.is_empty():
        return None
    from eligsir.io.sources.decode_cache import load_frame_depth_np

    shim = _shim_from_refs(UndecodedBufferRef(), depth_ref)
    raw = load_frame_depth_np(shim)
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.dtype == np.uint16 or raw.dtype == np.int32:
        return raw.astype(np.float32) / float(depth_scale)
    return raw.astype(np.float32, copy=False)


def coarse_pose_novelty(
    previous_c2w: Optional[np.ndarray],
    current_c2w: np.ndarray,
    *,
    translation_norm_m: float = 1.0,
    rotation_norm_deg: float = 30.0,
) -> float:
    """Cheap pose-only novelty in ``[0, 1]``: how far ``current_c2w`` moved
    from ``previous_c2w``, normalized by ``translation_norm_m``/
    ``rotation_norm_deg`` and clipped to 1.0. Computed from the two 4x4
    matrices alone -- no image, no ``Camera`` object -- so it can gate a
    reject-before-decode fast path (see ``prefilter.should_skip_decode``).

    Returns 1.0 (maximally novel) when there is no previous pose to compare
    against, so the very first candidate is never spuriously rejected.
    """
    if previous_c2w is None:
        return 1.0
    prev = np.asarray(previous_c2w, dtype=np.float64)
    cur = np.asarray(current_c2w, dtype=np.float64)
    if prev.shape != (4, 4) or cur.shape != (4, 4):
        return 1.0

    d_t = float(np.linalg.norm(cur[:3, 3] - prev[:3, 3]))
    r_rel = prev[:3, :3].T @ cur[:3, :3]
    trace = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
    d_r_deg = math.degrees(math.acos(trace))

    t_score = min(1.0, d_t / max(translation_norm_m, 1e-6))
    r_score = min(1.0, d_r_deg / max(rotation_norm_deg, 1e-6))
    return max(t_score, r_score)


def prepare_candidate(
    tracked: TrackedFramePacket,
    *,
    previous_c2w: Optional[np.ndarray] = None,
    quality_config: Optional[FrameQualityConfig] = None,
    novelty_translation_norm_m: float = 1.0,
    novelty_rotation_norm_deg: float = 30.0,
) -> PreparedFrameCandidate:
    """Full CPU-side preparation of one tracked frame: decode RGB/depth,
    score quality (blur/exposure/valid-depth-fraction) via the existing
    pure ``frame_quality`` logic, and compute a cheap pose-novelty score.

    Intended to run on :class:`~eligsir.runtime.workers.
    frame_preparation_worker.FramePreparationWorker`'s background thread,
    never on the optimizer thread. Produces CPU-resident numpy buffers
    only -- see ``contracts.PreparedFrameCandidate``'s docstring for why
    that matters (deliverable #3).
    """
    frame = tracked.frame
    rgb = decode_rgb(frame.rgb_ref)
    depth_m = decode_depth_m(frame.depth_ref, frame.intrinsics.depth_scale)

    if rgb is None:
        # Nothing to score: report a fully-rejected candidate rather than
        # raising, so a transient decode failure surfaces as a normal
        # prefilter rejection with a reason, not a crashed worker thread.
        evidence = InputEvidence(
            blur_laplacian=0.0,
            exposure_luma=0.0,
            white_clip_ratio=0.0,
            black_clip_ratio=0.0,
            valid_depth_fraction=0.0,
            quality_weight=0.0,
            reject_reason="rgb_decode_failed",
        )
        image_ref = PreparedImageRef(array=None, width=0, height=0)
        depth_ref = PreparedDepthRef(array=None, width=0, height=0, valid_fraction=0.0)
        novelty = coarse_pose_novelty(
            previous_c2w, tracked.world_T_camera,
            translation_norm_m=novelty_translation_norm_m,
            rotation_norm_deg=novelty_rotation_norm_deg,
        )
        return PreparedFrameCandidate(
            source_index=frame.source_index,
            timestamp_ns=frame.timestamp_ns,
            pose=Pose(np.asarray(tracked.world_T_camera, dtype=np.float32)),
            rgb_ref=image_ref,
            depth_ref=depth_ref,
            input_evidence=evidence,
            coarse_novelty=novelty,
        )

    quality = evaluate_frame_quality(rgb, depth_m, config=quality_config)
    evidence = InputEvidence(
        blur_laplacian=quality.blur_laplacian,
        exposure_luma=float(np.asarray(rgb, dtype=np.float32).mean() / 255.0)
        if np.asarray(rgb).max(initial=0.0) > 1.5 else float(np.asarray(rgb, dtype=np.float32).mean()),
        white_clip_ratio=quality.white_clip_ratio,
        black_clip_ratio=quality.black_clip_ratio,
        valid_depth_fraction=quality.depth_valid_ratio,
        quality_weight=quality.quality_weight,
        reject_reason=quality.reject_reason,
    )
    h, w = rgb.shape[0], rgb.shape[1]
    image_ref = PreparedImageRef(array=rgb, width=w, height=h, downsampled=False)
    if depth_m is not None:
        dh, dw = depth_m.shape[0], depth_m.shape[1]
        depth_ref = PreparedDepthRef(
            array=depth_m, width=dw, height=dh, valid_fraction=quality.depth_valid_ratio
        )
    else:
        depth_ref = PreparedDepthRef(array=None, width=0, height=0, valid_fraction=0.0)

    novelty = coarse_pose_novelty(
        previous_c2w, tracked.world_T_camera,
        translation_norm_m=novelty_translation_norm_m,
        rotation_norm_deg=novelty_rotation_norm_deg,
    )

    return PreparedFrameCandidate(
        source_index=frame.source_index,
        timestamp_ns=frame.timestamp_ns,
        pose=Pose(np.asarray(tracked.world_T_camera, dtype=np.float32)),
        rgb_ref=image_ref,
        depth_ref=depth_ref,
        input_evidence=evidence,
        coarse_novelty=novelty,
    )
