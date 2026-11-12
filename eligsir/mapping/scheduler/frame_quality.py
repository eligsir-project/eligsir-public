"""Frame quality metrics for streaming RGB-D admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    """Compact quality summary used by keyframe scoring."""

    blur_laplacian: float
    gradient_energy: float
    white_clip_ratio: float
    black_clip_ratio: float
    depth_valid_ratio: float
    depth_hole_ratio: float
    depth_edge_invalid_ratio: float
    timestamp_delta_ms: Optional[float] = None
    quality_weight: float = 1.0
    reject_reason: Optional[str] = None
    blur_quality_weight: float = 1.0
    exposure_quality_weight: float = 1.0

    @property
    def accepted(self) -> bool:
        """True if the frame passed quality gates (no reject_reason)."""
        return self.reject_reason is None


@dataclass(frozen=True)
class FrameQualityConfig:
    """Configuration for frame quality evaluation thresholds and adaptive scoring.

    Attributes:
        min_blur_laplacian: Minimum Laplacian variance for acceptance (higher = sharper).
        max_clip_ratio: Maximum fraction of pixels saturated (white) or black.
        min_depth_valid_ratio: Minimum fraction of depth pixels with valid measurements.
        max_depth_edge_invalid_ratio: Maximum fraction of depth-edge boundary
            pixels lacking a valid neighbor.
        depth_min, depth_max: Depth range bounds in meters for valid/invalid classification.
        adaptive_quality: If True, use scene-relative adaptive blur/exposure scoring
            instead of hard absolute thresholds.
        blur_floor: Absolute usability bar in resolution-normalized Laplacian units
            (only used when adaptive_quality=True).
        exposure_tolerance: Relative luminance tolerance for adaptive exposure scoring
            (only used when adaptive_quality=True).
    """
    min_blur_laplacian: float = 8.0
    max_clip_ratio: float = 0.25
    min_depth_valid_ratio: float = 0.10
    max_depth_edge_invalid_ratio: float = 0.50
    depth_min: float = 0.1
    depth_max: float = 8.0
    # Adaptive (scene-relative, resolution-normalized) blur/exposure scoring --
    # opt-in via adaptive_quality=True. When off, behavior is byte-identical to
    # the legacy path above (hard min_blur_laplacian/max_clip_ratio reject).
    #
    # Design (see docs/superpowers -- keyframe admission session 2026-07-28):
    # blur_floor is an ABSOLUTE usability bar in resolution-normalized Laplacian
    # units. A scene where every frame clears it (e.g. TUM fr3: min=87 >> 20)
    # gets blur_quality_weight=1.0 everywhere -- the mechanism self-disables on
    # clean scenes, which is the whole point (never reject the bottom X% of an
    # already-sharp trajectory, unlike a pure percentile rule). A scene with a
    # genuinely blurry tail (ScanNet++: median=24, barely above floor=20) gets
    # heavily downweighted there instead. Never a hard veto -- a region only
    # ever observed under blur/off-average exposure must still get keyframes,
    # or coverage gets a hole exactly there (same reasoning CLAUDE.md's motion
    # pregate memory applies to stale-pose frames).
    adaptive_quality: bool = False
    blur_floor: float = 20.0
    exposure_tolerance: float = 0.35


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3].astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        raise ValueError(f"Expected HxW or HxWx3 image, got shape {arr.shape}")
    if gray.max(initial=0.0) > 1.5:
        gray = gray / 255.0
    return gray.astype(np.float32, copy=False)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    c = gray[1:-1, 1:-1]
    lap = -4.0 * c + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(lap * 255.0))


def depth_confidence_opacity(
    depth_conf: np.ndarray,
    init_opacity: float,
    floor: float,
) -> np.ndarray:
    """Pure mapping: per-point depth confidence -> inverse-sigmoid (logit) opacity.

    Scales ``init_opacity`` by ``clip(depth_conf, floor, 1.0)`` so low-confidence
    points (far range / grazing angle) seed fainter Gaussians, then returns the
    pre-logit (inverse-sigmoid) value expected by ``add_points_as_gaussians``.

    ``floor`` bounds the reduction: floor=1.0 disables it, floor=0.3 caps the
    reduction at 70%. The probability is clamped to ``[1e-3, 0.99]`` for a finite
    logit. This is the exact mapping used at the streaming depth-insertion site.
    """
    conf = np.clip(np.asarray(depth_conf, dtype=np.float32), float(floor), 1.0)
    op = np.clip(float(init_opacity) * conf, 1e-3, 0.99)
    return np.log(op / (1.0 - op)).astype(np.float32)


def blur_laplacian_score(rgb: np.ndarray, downsample: int = 1) -> float:
    """Pure blur metric: variance of the Laplacian on a (downsampled) greyscale.

    Higher = sharper. Mirrors the cv2 ``Laplacian(...).var()`` semantics used by
    the input-quality blur gate (operates on 0-255-scale intensities).
    """
    arr = np.asarray(rgb)
    ds = max(1, int(downsample))
    if ds > 1 and arr.ndim >= 2:
        arr = arr[::ds, ::ds]
    gray = _to_gray(arr)  # normalised to [0, 1]
    return _laplacian_variance(gray)  # already scaled back to 0-255 intensity units


def blur_reject(rgb: np.ndarray, min_laplacian: float, downsample: int = 1) -> bool:
    """Pure predicate: True if the frame's Laplacian variance is below threshold."""
    return blur_laplacian_score(rgb, downsample) < float(min_laplacian)


def mean_luma(rgb: np.ndarray) -> float:
    """Pure mean luminance in [0, 1] (normalised regardless of 0-255 vs 0-1 input)."""
    arr = np.asarray(rgb, dtype=np.float32)
    m = float(arr.mean())
    return m / 255.0 if arr.max(initial=0.0) > 1.5 else m


def exposure_reject(current_luma: float, reference_luma: float, max_delta: float) -> bool:
    """Pure predicate: True if relative luminance deviation exceeds ``max_delta``.

    ``|current - reference| / max(reference, 1e-3) > max_delta``.
    """
    delta = abs(float(current_luma) - float(reference_luma)) / max(float(reference_luma), 1e-3)
    return delta > float(max_delta)


def normalized_blur_score(rgb: np.ndarray, target_width: int = 480) -> float:
    """Resolution-independent blur metric: Laplacian variance at a fixed
    canonical width. Laplacian variance scales with source resolution, so a
    single absolute floor (e.g. FrameQualityConfig.blur_floor) only means the
    same thing across scenes shot at different native resolutions if the
    metric itself is normalized first -- this resizes (nearest-neighbour,
    matching the ``[::ds]`` decimation style already used by
    ``blur_laplacian_score``) to ``target_width`` before scoring.
    """
    arr = np.asarray(rgb)
    h, w = arr.shape[0], arr.shape[1]
    if w <= 0 or h <= 0:
        return 0.0
    ds = max(1, round(w / max(1, int(target_width))))
    if ds > 1:
        arr = arr[::ds, ::ds]
    gray = _to_gray(arr)
    return _laplacian_variance(gray)


class RunningBlurExposureStats:
    """Causal (online, no future-peeking) EMA tracker for the adaptive
    blur/exposure quality path.

    Updated once per OBSERVED frame (every frame reaching admission scoring,
    not just admitted ones -- matches the project's existing "score every
    ingested frame" convention), in ingestion order, so it reflects only
    frames seen so far -- required for this to transfer to a live camera
    stream (CLAUDE.md hard constraint: live is the design target, not a
    global/offline trajectory statistic).

    Returns None from the ``*_typical``/``*_mean`` properties until
    ``warmup`` observations have been made, so the adaptive scoring in
    ``evaluate_frame_quality`` correctly applies NO penalty (quality_weight
    unaffected) during the cold-start window rather than judging early
    frames against a statistically meaningless 1-sample "typical" value.
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 20):
        self._alpha = float(alpha)
        self._warmup = int(warmup)
        self._n = 0
        self._blur_ema: Optional[float] = None
        self._luma_ema: Optional[float] = None

    def update(self, blur: float, luma: float) -> None:
        """Update the EMA with a new observed frame's blur and luminance.

        Args:
            blur: Laplacian variance (resolution-normalized) of the frame.
            luma: Mean luminance of the frame in [0, 1].
        """
        self._n += 1
        if self._blur_ema is None:
            self._blur_ema = float(blur)
            self._luma_ema = float(luma)
        else:
            a = self._alpha
            self._blur_ema = (1.0 - a) * self._blur_ema + a * float(blur)
            self._luma_ema = (1.0 - a) * self._luma_ema + a * float(luma)

    @property
    def blur_typical(self) -> Optional[float]:
        """EMA of blur (Laplacian variance) after warmup, None if still warming up."""
        return self._blur_ema if self._n >= self._warmup else None

    @property
    def luma_mean(self) -> Optional[float]:
        """EMA of luminance after warmup, None if still warming up."""
        return self._luma_ema if self._n >= self._warmup else None


def _gradient_energy(gray: np.ndarray) -> float:
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(0.5 * (np.mean(np.abs(gx)) + np.mean(np.abs(gy))))


def _depth_edge_invalid_ratio(valid: np.ndarray) -> float:
    if valid.shape[0] < 3 or valid.shape[1] < 3:
        return 0.0
    inner = valid[1:-1, 1:-1]
    if inner.size == 0:
        return 0.0
    edge_invalid = inner & (
        ~valid[:-2, 1:-1]
        | ~valid[2:, 1:-1]
        | ~valid[1:-1, :-2]
        | ~valid[1:-1, 2:]
    )
    return float(edge_invalid.sum() / max(int(inner.sum()), 1))


def evaluate_frame_quality(
    rgb: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
    *,
    timestamp_delta_ms: Optional[float] = None,
    config: Optional[FrameQualityConfig] = None,
    running_blur_typical: Optional[float] = None,
    running_luma_mean: Optional[float] = None,
) -> FrameQuality:
    """Compute deterministic frame-quality metrics from RGB and optional depth.

    ``running_blur_typical``/``running_luma_mean`` (from
    ``RunningBlurExposureStats``, causal -- frames seen so far only) are only
    consulted when ``config.adaptive_quality`` is True; ignored otherwise, so
    the legacy hard-threshold path is unaffected.
    """

    cfg = config or FrameQualityConfig()
    gray = _to_gray(rgb)
    blur = normalized_blur_score(rgb) if cfg.adaptive_quality else _laplacian_variance(gray)
    grad = _gradient_energy(gray)
    white_clip = float((gray >= 0.98).mean())
    black_clip = float((gray <= 0.02).mean())
    luma = mean_luma(rgb)

    depth_valid_ratio = 1.0
    depth_hole_ratio = 0.0
    depth_edge_ratio = 0.0
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > cfg.depth_min) & (depth < cfg.depth_max)
        depth_valid_ratio = float(valid.mean()) if valid.size else 0.0
        depth_hole_ratio = 1.0 - depth_valid_ratio
        depth_edge_ratio = _depth_edge_invalid_ratio(valid)

    reject_reason = None
    if cfg.adaptive_quality:
        # Blur and off-average exposure are soft (scored below), never a hard
        # veto -- see FrameQualityConfig's docstring on why. A fully blown-out
        # or black frame (clip_ratio) genuinely carries zero information
        # regardless of scene, so that hard case is kept.
        if max(white_clip, black_clip) > cfg.max_clip_ratio:
            reject_reason = "exposure_clip"
        elif depth_valid_ratio < cfg.min_depth_valid_ratio:
            reject_reason = "low_depth_valid"
        elif depth_edge_ratio > cfg.max_depth_edge_invalid_ratio:
            reject_reason = "depth_edges"
    else:
        if blur < cfg.min_blur_laplacian:
            reject_reason = "blur"
        elif max(white_clip, black_clip) > cfg.max_clip_ratio:
            reject_reason = "exposure_clip"
        elif depth_valid_ratio < cfg.min_depth_valid_ratio:
            reject_reason = "low_depth_valid"
        elif depth_edge_ratio > cfg.max_depth_edge_invalid_ratio:
            reject_reason = "depth_edges"

    depth_score = min(1.0, depth_valid_ratio / max(cfg.min_depth_valid_ratio, 1e-6))
    edge_score = 1.0 - min(1.0, depth_edge_ratio / max(cfg.max_depth_edge_invalid_ratio, 1e-6))

    if cfg.adaptive_quality:
        if running_blur_typical is None:
            blur_quality_weight = 1.0  # cold start / clean-scene default: no penalty
        else:
            span = max(running_blur_typical - cfg.blur_floor, 1e-6)
            blur_quality_weight = float(np.clip((blur - cfg.blur_floor) / span, 0.0, 1.0))
        if running_luma_mean is None:
            exposure_quality_weight = 1.0
        else:
            tol = max(cfg.exposure_tolerance * max(running_luma_mean, 1e-3), 1e-6)
            exposure_quality_weight = float(np.clip(1.0 - abs(luma - running_luma_mean) / tol, 0.0, 1.0))
        quality_weight = float(np.clip(
            0.35 * blur_quality_weight + 0.25 * exposure_quality_weight
            + 0.30 * depth_score + 0.10 * edge_score,
            0.0, 1.0,
        ))
    else:
        blur_quality_weight = min(1.0, blur / max(cfg.min_blur_laplacian * 4.0, 1e-6))
        exposure_quality_weight = 1.0 - min(1.0, max(white_clip, black_clip) / max(cfg.max_clip_ratio, 1e-6))
        quality_weight = float(np.clip(
            0.35 * blur_quality_weight + 0.25 * exposure_quality_weight
            + 0.30 * depth_score + 0.10 * edge_score,
            0.0, 1.0,
        ))

    return FrameQuality(
        blur_laplacian=blur,
        gradient_energy=grad,
        white_clip_ratio=white_clip,
        black_clip_ratio=black_clip,
        depth_valid_ratio=depth_valid_ratio,
        depth_hole_ratio=depth_hole_ratio,
        depth_edge_invalid_ratio=depth_edge_ratio,
        timestamp_delta_ms=timestamp_delta_ms,
        quality_weight=quality_weight,
        reject_reason=reject_reason,
        blur_quality_weight=blur_quality_weight,
        exposure_quality_weight=exposure_quality_weight,
    )
