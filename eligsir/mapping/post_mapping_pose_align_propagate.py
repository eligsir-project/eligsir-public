"""Pure propagation of post-mapping train corrections to holdout cameras.

This is the dependency-free, audited 8064 propagation contract. Runtime code
owns lookup and scoring; this module only transfers already-fitted train
pose/exposure corrections without looking at a holdout image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


_VALID_MODES = frozenset({"none", "interp", "global"})


@dataclass(frozen=True)
class PropagateHoldoutConfig:
    """Snapshot of the post-mapping holdout propagation flags."""

    mode: str = "none"
    max_gap: int = 8
    propagate_exposure: bool = False
    window: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"invalid propagation mode {self.mode!r}")
        if self.max_gap < 0:
            raise ValueError("invalid propagation max_gap")
        if self.window < 1:
            raise ValueError("invalid propagation window")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @classmethod
    def from_args(cls, args: object) -> "PropagateHoldoutConfig":
        return cls(
            mode=str(getattr(args, "streaming_post_mapping_pose_align_propagate_holdout", "none") or "none"),
            max_gap=int(getattr(args, "streaming_post_mapping_pose_align_propagate_max_gap", 8) or 0),
            propagate_exposure=bool(getattr(args, "streaming_post_mapping_pose_align_propagate_exposure", False) or False),
            window=int(getattr(args, "streaming_post_mapping_pose_align_propagate_window", 1) or 1),
        )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rotmat_from_omega(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    skew = _skew(omega / theta)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _omega_from_rotmat(rotation: np.ndarray) -> np.ndarray:
    theta = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def _quat_from_rotmat(rotation: np.ndarray) -> np.ndarray:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = 0.5 / math.sqrt(trace + 1.0)
        quaternion = (0.25 / scale, (m21 - m12) * scale, (m02 - m20) * scale, (m10 - m01) * scale)
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        quaternion = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        quaternion = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        quaternion = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)
    result = np.asarray(quaternion, dtype=np.float64)
    return result / (np.linalg.norm(result) + 1e-12)


def _rotmat_from_quat(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float64,
    )


def _slerp_quat(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    if np.dot(first, second) < 0.0:
        second = -second
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / (np.linalg.norm(result) + 1e-12)
    theta = math.acos(dot)
    perpendicular = second - first * dot
    perpendicular /= np.linalg.norm(perpendicular) + 1e-12
    return first * math.cos(theta * fraction) + perpendicular * math.sin(theta * fraction)


def delta6_to_matrix(delta6: Sequence[float]) -> np.ndarray:
    delta = np.asarray(delta6, dtype=np.float64)
    if delta.shape != (6,):
        raise ValueError(f"expected a 6-vector, got shape {delta.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotmat_from_omega(delta[:3])
    result[:3, 3] = delta[3:]
    return result


def apply_delta6_to_c2w(c2w: np.ndarray, delta6: Sequence[float]) -> np.ndarray:
    """Apply the established camera-local right-composed correction."""
    return np.asarray(c2w, dtype=np.float64) @ delta6_to_matrix(delta6)


def interpolate_delta(
    train_deltas_by_ordinal: Mapping[int, Sequence[float]], holdout_ordinal: int, max_gap: int
) -> Optional[np.ndarray]:
    """Legacy nearest-before/nearest-after SLERP plus translation LERP."""
    holdout_ordinal = int(holdout_ordinal)
    before = after = None
    for raw_ordinal, delta6 in train_deltas_by_ordinal.items():
        ordinal = int(raw_ordinal)
        delta = np.asarray(delta6, dtype=np.float64)
        if ordinal <= holdout_ordinal and (before is None or ordinal > before[0]):
            before = ordinal, delta
        if ordinal >= holdout_ordinal and (after is None or ordinal < after[0]):
            after = ordinal, delta
    if before is not None and holdout_ordinal - before[0] > max_gap:
        before = None
    if after is not None and after[0] - holdout_ordinal > max_gap:
        after = None
    if before is None:
        return None if after is None else after[1]
    if after is None or before[0] == after[0]:
        return before[1]
    fraction = (holdout_ordinal - before[0]) / (after[0] - before[0])
    quaternion = _slerp_quat(
        _quat_from_rotmat(_rotmat_from_omega(before[1][:3])),
        _quat_from_rotmat(_rotmat_from_omega(after[1][:3])),
        fraction,
    )
    return np.concatenate([
        _omega_from_rotmat(_rotmat_from_quat(quaternion)),
        (1.0 - fraction) * before[1][3:] + fraction * after[1][3:],
    ])


def _select_propagation_neighbours(
    available_ordinals: Sequence[int], holdout_ordinal: int, max_gap: int, window: int
) -> list[tuple[int, float]]:
    """Select the audited local window shared by pose and exposure paths."""
    holdout_ordinal = int(holdout_ordinal)
    ordinals = sorted({int(ordinal) for ordinal in available_ordinals})
    if window <= 1:
        before = max((o for o in ordinals if o <= holdout_ordinal), default=None)
        after = min((o for o in ordinals if o >= holdout_ordinal), default=None)
        if before is not None and holdout_ordinal - before > max_gap:
            before = None
        if after is not None and after - holdout_ordinal > max_gap:
            after = None
        if before is None:
            return [] if after is None else [(after, 1.0)]
        if after is None or after == before:
            return [(before, 1.0)]
        fraction = (holdout_ordinal - before) / (after - before)
        return [(before, 1.0 - fraction), (after, fraction)]
    before = [o for o in ordinals if o <= holdout_ordinal and holdout_ordinal - o <= max_gap]
    after = [o for o in ordinals if o > holdout_ordinal and o - holdout_ordinal <= max_gap]
    chosen = before[-window:] + after[:window]
    if not chosen:
        return []
    sigma = window / 2.0
    weights = [math.exp(-0.5 * (abs(o - holdout_ordinal) / sigma) ** 2) for o in chosen]
    total = sum(weights)
    return [] if total <= 0.0 else [(o, weight / total) for o, weight in zip(chosen, weights)]


def interpolate_delta_windowed(
    train_deltas_by_ordinal: Mapping[int, Sequence[float]],
    holdout_ordinal: int,
    max_gap: int,
    window: int = 1,
) -> Optional[np.ndarray]:
    """Use exact legacy SLERP at width one; Gaussian average wider windows."""
    if int(window) <= 1:
        return interpolate_delta(train_deltas_by_ordinal, holdout_ordinal, max_gap)
    neighbours = _select_propagation_neighbours(train_deltas_by_ordinal.keys(), holdout_ordinal, max_gap, int(window))
    if not neighbours:
        return None
    if len(neighbours) == 1:
        return np.asarray(train_deltas_by_ordinal[neighbours[0][0]], dtype=np.float64)
    quaternions, translations, weights = [], [], []
    reference = None
    for ordinal, weight in neighbours:
        delta = np.asarray(train_deltas_by_ordinal[ordinal], dtype=np.float64)
        quaternion = _quat_from_rotmat(_rotmat_from_omega(delta[:3]))
        if reference is None:
            reference = quaternion
        elif np.dot(quaternion, reference) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)
        translations.append(delta[3:])
        weights.append(weight)
    mean_quaternion = np.average(np.stack(quaternions), axis=0, weights=weights)
    mean_quaternion /= np.linalg.norm(mean_quaternion) + 1e-12
    return np.concatenate([
        _omega_from_rotmat(_rotmat_from_quat(mean_quaternion)),
        np.average(np.stack(translations), axis=0, weights=weights),
    ])


def interpolate_exposure_windowed(
    train_exposure_by_ordinal: Mapping[int, tuple],
    holdout_ordinal: int,
    max_gap: int,
    window: int = 1,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    neighbours = _select_propagation_neighbours(train_exposure_by_ordinal.keys(), holdout_ordinal, max_gap, int(window))
    if not neighbours:
        return None
    gains, biases, weights = [], [], []
    for ordinal, weight in neighbours:
        gain, bias = train_exposure_by_ordinal[ordinal]
        gains.append(np.asarray(gain, dtype=np.float64))
        biases.append(np.asarray(bias, dtype=np.float64))
        weights.append(weight)
    if len(neighbours) == 1:
        return gains[0], biases[0]
    return np.average(np.stack(gains), axis=0, weights=weights), np.average(np.stack(biases), axis=0, weights=weights)


def fit_global_delta(train_deltas_by_ordinal: Mapping[int, Sequence[float]]) -> Optional[np.ndarray]:
    """Uniform hemisphere-aligned quaternion and translation average."""
    if not train_deltas_by_ordinal:
        return None
    quaternions, translations = [], []
    reference = None
    for value in train_deltas_by_ordinal.values():
        delta = np.asarray(value, dtype=np.float64)
        quaternion = _quat_from_rotmat(_rotmat_from_omega(delta[:3]))
        if reference is None:
            reference = quaternion
        elif np.dot(quaternion, reference) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)
        translations.append(delta[3:])
    mean_quaternion = np.mean(np.stack(quaternions), axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion) + 1e-12
    return np.concatenate([
        _omega_from_rotmat(_rotmat_from_quat(mean_quaternion)),
        np.mean(np.stack(translations), axis=0),
    ])


def fit_global_exposure(train_exposure_by_ordinal: Mapping[int, tuple]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not train_exposure_by_ordinal:
        return None
    gains = [np.asarray(gain, dtype=np.float64) for gain, _ in train_exposure_by_ordinal.values()]
    biases = [np.asarray(bias, dtype=np.float64) for _, bias in train_exposure_by_ordinal.values()]
    return np.mean(np.stack(gains), axis=0), np.mean(np.stack(biases), axis=0)


def propagated_holdout_deltas(
    train_deltas_by_ordinal: Mapping[int, Sequence[float]],
    holdout_ordinals: Sequence[int],
    config: PropagateHoldoutConfig,
) -> dict[int, np.ndarray]:
    if not config.enabled or not train_deltas_by_ordinal:
        return {}
    if config.mode == "global":
        delta = fit_global_delta(train_deltas_by_ordinal)
        return {} if delta is None else {int(ordinal): delta for ordinal in holdout_ordinals}
    result = {}
    for ordinal in holdout_ordinals:
        delta = interpolate_delta_windowed(train_deltas_by_ordinal, int(ordinal), config.max_gap, config.window)
        if delta is not None:
            result[int(ordinal)] = delta
    return result


def propagated_holdout_exposure(
    train_exposure_by_ordinal: Mapping[int, tuple],
    holdout_ordinals: Sequence[int],
    config: PropagateHoldoutConfig,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not config.enabled or not config.propagate_exposure or not train_exposure_by_ordinal:
        return {}
    if config.mode == "global":
        exposure = fit_global_exposure(train_exposure_by_ordinal)
        return {} if exposure is None else {int(ordinal): exposure for ordinal in holdout_ordinals}
    result = {}
    for ordinal in holdout_ordinals:
        exposure = interpolate_exposure_windowed(train_exposure_by_ordinal, int(ordinal), config.max_gap, config.window)
        if exposure is not None:
            result[int(ordinal)] = exposure
    return result


__all__ = [
    "PropagateHoldoutConfig",
    "apply_delta6_to_c2w",
    "delta6_to_matrix",
    "fit_global_delta",
    "fit_global_exposure",
    "interpolate_delta",
    "interpolate_delta_windowed",
    "interpolate_exposure_windowed",
    "propagated_holdout_deltas",
    "propagated_holdout_exposure",
]
