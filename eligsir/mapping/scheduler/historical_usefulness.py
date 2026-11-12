"""Train-time-only historical-frame usefulness scoring (issue #953/#1191).

This module supplies the missing signal for
:class:`~eligsir.mapping.scheduler.historical_candidate_index.HistoricalFrameMetadata.base_weight`,
which is registered unconditionally at ``1.0`` today (see
``MappingApplication._register_historical_record`` in
``eligsir/runtime/production_application.py``) -- every historical frame
looks equally useful to the replay sampler regardless of how well the map
currently explains it.

## Why this exists (#953's weak-region-repair hypothesis)

#953 asks whether shifting supervision toward a weak/degrading trajectory
region can raise its local PSNR. #116's ``TARGETED_HISTORY`` repair mode
(``eligsir/mapping/scheduler/historical_repair.py::select_historical_repair``)
already ranks cached candidates by ``(-region_overlap, -prior_usefulness,
-baseline_diversity, -supervision_age_steps, source_index)`` -- but every
caller of that ranking (``utils/partition_control.py::_historical_repair_selection``)
constructs its ``HistoricalFrameEvidence`` with ``prior_usefulness=0.0``, so in
practice TARGETED_HISTORY degenerates to an arbitrary ascending-index tie-break
within the target region. This module is the producer that gap needs.

## Train-time-only, never scored-holdout (#1191 contract)

The intended signal is the frame's OWN last-observed training/replay
photometric (or depth) loss -- a real, local, non-holdout measurement of how
well the current map explains it. High last-observed loss under the current
map means the region has degraded or was never well fit and is exactly the
kind of candidate #953's repair ladder should prefer to reactivate. This
module never reads or is passed anything from the frozen scored holdout.

## Hysteresis, not one noisy event (#1191 contract)

A single anomalous render (transient occlusion, motion blur, one bad frame)
must never permanently promote or demote a candidate. :class:`EmaUsefulnessTracker`
enforces this two ways: (1) every new observation is folded in via an
exponential moving average rather than overwriting the prior value, and (2) a
frame's smoothed weight is only trusted (returned instead of the neutral
default) once it has accumulated ``min_observations`` real observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


#: Default base weight for a frame with no (or not-yet-trusted) usefulness
#: evidence. Must match ``HistoricalFrameMetadata.base_weight``'s own default
#: so "no signal yet" is indistinguishable from today's pre-#953 behavior.
NEUTRAL_WEIGHT = 1.0

#: Weight assigned to a frame whose smoothed normalized loss saturates the
#: transform (see ``usefulness_weight_from_loss``). Bounded so a single
#: pathological loss value cannot make one frame dominate every replay draw
#: (`WeightedReplayCandidateProvider` samples relative weight, not an
#: absolute probability, but an unbounded weight would still starve the rest
#: of a bounded shortlist).
MAX_WEIGHT = 5.0


def usefulness_weight_from_loss(loss: float, *, loss_scale: float = 1.0) -> float:
    """Map a non-negative photometric/depth loss to a bounded base weight.

    Monotonically increasing in ``loss``: a frame the current map explains
    poorly (high loss) is more valuable to reactivate than one it already
    reconstructs well, which is the opposite ranking direction from anything
    that treats loss as a quality *score* -- this is a *repair priority*, not
    a quality metric. ``loss <= 0`` (perfectly explained, or no evidence) maps
    to exactly :data:`NEUTRAL_WEIGHT`, so a real-zero observation never reads
    as "below neutral" and cannot make a candidate less likely to be sampled
    than an unobserved one.

    Args:
        loss: A non-negative scalar loss (e.g. L1/SSIM combination, or a
            depth residual) for one view under the current map/topology
            snapshot. Must be finite.
        loss_scale: Divides ``loss`` before the transform so callers can
            calibrate to their own loss's typical magnitude. Must be finite
            and positive.

    Returns:
        A finite weight in ``[NEUTRAL_WEIGHT, MAX_WEIGHT]``.
    """

    if isinstance(loss, bool) or not math.isfinite(float(loss)):
        raise ValueError("loss must be a finite real number")
    if (
        isinstance(loss_scale, bool)
        or not math.isfinite(float(loss_scale))
        or float(loss_scale) <= 0.0
    ):
        raise ValueError("loss_scale must be finite and positive")
    normalized = max(0.0, float(loss)) / float(loss_scale)
    # tanh saturates smoothly toward 1.0 rather than clamping hard, so the
    # ranking among high-loss candidates stays informative near the ceiling
    # instead of collapsing to a tie.
    return NEUTRAL_WEIGHT + (MAX_WEIGHT - NEUTRAL_WEIGHT) * math.tanh(normalized)


@dataclass(frozen=True, slots=True)
class UsefulnessObservation:
    """One frame's smoothed usefulness state, for inspection/telemetry."""

    frame_id: int
    observation_count: int
    smoothed_raw: float
    trusted: bool


class EmaUsefulnessTracker:
    """Per-frame exponential-moving-average usefulness with a trust gate.

    Deliberately holds no reference to any camera, tensor, or GPU state --
    every entry is ``frame_id -> (count, smoothed_raw_weight)``, so this class
    is fully CPU-only and unit-testable without a rendering stack. Bounded
    memory: ``max_tracked_frames`` evicts the least-recently-observed entry
    once exceeded (a simple OrderedDict-backed LRU), matching this project's
    convention of bounded, not unbounded, per-frame state
    (``HistoricalCandidateIndex`` itself, ``_retired_replay_sources``, etc.).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        min_observations: int = 3,
        max_tracked_frames: int = 4096,
    ) -> None:
        if (
            isinstance(alpha, bool)
            or not math.isfinite(float(alpha))
            or not (0.0 < float(alpha) <= 1.0)
        ):
            raise ValueError("alpha must be in (0.0, 1.0]")
        if isinstance(min_observations, bool) or int(min_observations) < 1:
            raise ValueError("min_observations must be a positive integer")
        if isinstance(max_tracked_frames, bool) or int(max_tracked_frames) < 1:
            raise ValueError("max_tracked_frames must be a positive integer")
        self._alpha = float(alpha)
        self._min_observations = int(min_observations)
        self._max_tracked_frames = int(max_tracked_frames)
        from collections import OrderedDict

        self._state: "OrderedDict[int, tuple[int, float]]" = OrderedDict()

    def observe(self, frame_id: int, raw_weight: float) -> UsefulnessObservation:
        """Fold in one new raw weight observation for ``frame_id``."""

        frame_id = int(frame_id)
        if not math.isfinite(float(raw_weight)):
            raise ValueError("raw_weight must be finite")
        raw_weight = float(raw_weight)
        prior = self._state.pop(frame_id, None)
        if prior is None:
            count, smoothed = 1, raw_weight
        else:
            prior_count, prior_smoothed = prior
            count = prior_count + 1
            smoothed = self._alpha * raw_weight + (1.0 - self._alpha) * prior_smoothed
        self._state[frame_id] = (count, smoothed)
        self._state.move_to_end(frame_id)
        while len(self._state) > self._max_tracked_frames:
            self._state.popitem(last=False)
        return UsefulnessObservation(
            frame_id=frame_id,
            observation_count=count,
            smoothed_raw=smoothed,
            trusted=count >= self._min_observations,
        )

    def weight(self, frame_id: int, *, default: float = NEUTRAL_WEIGHT) -> float:
        """Return the trusted smoothed weight for ``frame_id``, else ``default``.

        Never raises on an unknown frame -- this must be safe to call from a
        hot path for every historical candidate, observed or not.
        """

        entry = self._state.get(int(frame_id))
        if entry is None:
            return float(default)
        count, smoothed = entry
        if count < self._min_observations:
            return float(default)
        return smoothed

    def __len__(self) -> int:
        return len(self._state)

    def forget(self, frame_id: int) -> None:
        """Drop a frame's tracked state (e.g. on retirement from the index)."""

        self._state.pop(int(frame_id), None)
