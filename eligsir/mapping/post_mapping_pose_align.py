"""Opt-in ONE-SHOT post-mapping-endpoint SE(3) pose (+ exposure) alignment.

GPU evidence motivating this module (2026-09, TUM fr3, documented on PR
#1851): eval-time alignment of holdout views against the FROZEN final map
(``utils/eval_pose_align.py``) lifts PSNR 23.4 -> 25.5dB, but refining every
INCOMING frame against the LIVE (still-optimizing) map during mapping
(``eligsir/mapping/admission_pose_refine.py``, run
``tum36_refine20_budget20``) is a destructive feedback loop: 2258 frames
refined, mean delta 1.37deg/18mm, mapping-endpoint PSNR REGRESSED 22.3 ->
16.6dB. A frame's pose correction, fit against a map that has not yet
converged, gets baked into supervision and corrupts the very map the next
frame's correction is fit against.

This module is the two-pass alternative: instead of aligning during
mapping, wait until the mapping endpoint (the model is DONE changing --
``eligsir/runtime/causal_entry.py::run_causal_post_mapping_refinement``
calls this after ``_publish_causal_mapping_endpoint`` freezes/snapshots the
map and BEFORE ``enter_post_mapping_refinement`` lets the optimizer touch
the Gaussians again), then align every training frame's pose ONCE against
that now-frozen, converged map. There is no feedback loop because the
target the fit chases never moves during the pass, and no later frame's fit
depends on an earlier frame's correction (each frame gets its own fresh -- or
temporally warm-started, see ``warm_start`` in
``eligsir/mapping/admission_pose_refine.py``'s identical convention --
6-vector delta against the SAME frozen render).

The cache-owned endpoint adapter supplies every retained training and holdout
frame as a :class:`PoseAlignFrameHandle`; this generic layer neither queries
``FrameObservationDB`` nor opens original source data.  Its
``frames_considered`` count is therefore exactly the supplied handle count,
and ``skipped_no_rgb`` applies only when a supplied cache-backed camera cannot
be built.

Runs once, off the source clock (the source has already completed by the
time this executes), so its cost is ``steps * n_train`` renders at whatever
tier is configured -- there is no per-tick budget to manage (unlike
``admission_pose_refine.py``, which must not stall live 30fps admission).

The expensive Adam-fit loop is split from endpoint composition so it remains
unit-testable without a live ``ObservationDB``/``ProductionApplication``:

  * :func:`align_training_frames` -- the generic algorithm. Takes a plain
    sequence of :class:`PoseAlignFrameHandle` (frame identity + a camera
    builder + a durable write-back callback) and never touches
    ``eligsir.mapping.observation_db`` or
    ``eligsir.runtime.production_application`` directly.
  * Endpoint composition -- intentionally absent from this baseline -- owns
    cache-backed handle construction and durable state updates.

Exposure alignment (``--streaming_post_mapping_exposure_align``) fits a
closed-form per-frame affine (``eligsir/exposure_opt.py::fit_affine``)
between the pose-aligned render and the frame's own GT and records it
(gain/bias, plus the residual it would have corrected) in
``post_mapping_pose_align.jsonl``, the run summary, AND -- durably --
``FrameRecord.exposure_gain``/``exposure_bias`` (see ``build_frame_handles_
from_source``'s write-back). This module never renders or computes a loss
itself, so it still changes no rendered output directly; the correction is
consumed downstream, once refinement starts, by
``production_application.py``'s ``m_tick_camera_materialize`` replay/
materialize path, which applies the INVERSE affine
(``(rgb - bias) / gain``, clamped to ``[0, 1]``) to a train frame's
supervision RGB right after it is materialized at whatever tier, so the
batched refinement objective and renderer stay untouched and only see an
exposure-corrected target. That call site is gated on the post-mapping
REFINEMENT phase and a per-frame affine actually being present, so a
mapping-phase tick (or a frame this pass skipped/errored on) is byte-
identical to before this wiring existed. Holdout frames never get an
affine fit here (only TRAIN handles) and are never routed through that
replay path at all, so the eval-time exposure alignment
(``utils/eval_exposure_align.py``) remains the separate, only source of
holdout-side correction, reported independently.

Every failure mode (steps<=0, no camera could be built, a raised exception
anywhere in the fit, a non-finite delta) is a no-op for that ONE frame with
a visible skip counter -- one frame's bad fit never aborts the pass or
poisons another frame's warm start (a non-finite fit resets the warm-start
seed to ``None``, exactly like ``admission_pose_refine.py``).
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from eligsir.pose_opt import (
    _TIER_FACTOR,
    _TieredCamera,
    _clamp_delta,
    _delta_deg_mm,
    _tiered_gt,
    apply_pose_delta,
    bake_pose_delta_into_camera,
    se3_delta_matrix,
)
from eligsir.mapping.post_mapping_pose_align_propagate import (
    apply_delta6_to_c2w,
    interpolate_delta_windowed,
    interpolate_exposure_windowed,
)


# --------------------------------------------------------------------------
# Config / stats
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PostMappingPoseAlignConfig:
    """Immutable snapshot of ``--streaming_post_mapping_pose_align_*`` flags."""

    steps: int = 0
    lr: float = 1e-3
    tier: str = "half"
    max_deg: float = 1.0
    max_mm: float = 20.0
    lambda_dssim: float = 0.2
    exposure_align: bool = False
    warm_start: bool = True
    # --------------------------------------------------------------------
    # PR #1851 follow-up (2026-09-07, ScanNet++ cost fix): "every eligible
    # training frame, every one aligned with a full `steps`-step Adam fit"
    # does not scale -- a real half-tier, 50-step, ~6000-frame ScanNet++
    # run (psnr26spp17_half_b16_dense4k_depth0_twopass_prop) measured ~1
    # frame/s in post_mapping_pose_align.jsonl, ~2h wall for this ONE pass,
    # unacceptable for a "very short refinement" recipe. These three knobs
    # bound that cost; every one is additive and defaults to the pre-existing
    # "align everything" behaviour when left at its default.
    # --------------------------------------------------------------------
    # Align only every `stride`-th ELIGIBLE TRAIN frame in temporal
    # (`order_key`) order; the first and last eligible train frames are
    # always aligned regardless of stride (see `align_training_frames`'s
    # selection). `1` (default): every eligible frame is aligned, byte-
    # identical to pre-stride behaviour.
    stride: int = 1
    # If the eligible train-frame count exceeds this, raise the EFFECTIVE
    # stride to `ceil(n_eligible / max_frames)` (never below `stride`
    # itself) so a run's total aligned-frame count stays bounded regardless
    # of how many training frames the source produced. `0` (default):
    # unlimited, `stride` alone governs spacing.
    max_frames: int = 0
    # Wall-clock budget, in seconds, for the ALIGNMENT (Adam-fit) work in
    # this pass. Once spent, every remaining selected-but-not-yet-aligned
    # frame is interpolated instead of fitted (see
    # `PostMappingPoseAlignStats.budget_exhausted`). `0.0` (default):
    # unlimited, matching `--streaming_admission_pose_refine_budget_ms`'s
    # sibling per-tick knob but here for the whole one-shot pass, not a
    # single source tick.
    time_budget_s: float = 0.0
    # Neighbourhood width used to interpolate a non-selected/budget-skipped
    # training frame's pose (and exposure) delta from its ALIGNED temporal
    # neighbours -- the SAME `--streaming_post_mapping_pose_align_propagate_
    # window` K already used to propagate deltas onto HOLDOUT cameras (see
    # `eligsir/mapping/post_mapping_pose_align_propagate.py`'s
    # `interpolate_delta_windowed`/`interpolate_exposure_windowed`), reused
    # here unchanged rather than re-derived, so "stride skip" and "holdout
    # propagation" share one interpolation algorithm end to end.
    interpolate_window: int = 1

    def __post_init__(self) -> None:
        if self.tier not in _TIER_FACTOR:
            raise ValueError(
                f"streaming_post_mapping_pose_align_tier must be one of "
                f"{sorted(_TIER_FACTOR)}, got {self.tier!r}"
            )
        if self.stride < 1:
            raise ValueError(
                f"streaming_post_mapping_pose_align_stride must be >= 1, got {self.stride!r}"
            )
        if self.max_frames < 0:
            raise ValueError(
                "streaming_post_mapping_pose_align_max_frames must be >= 0, "
                f"got {self.max_frames!r}"
            )
        if self.time_budget_s < 0:
            raise ValueError(
                "streaming_post_mapping_pose_align_time_budget_s must be >= 0, "
                f"got {self.time_budget_s!r}"
            )
        if self.interpolate_window < 1:
            raise ValueError(
                "streaming_post_mapping_pose_align_propagate_window must be >= 1, "
                f"got {self.interpolate_window!r}"
            )

    @property
    def enabled(self) -> bool:
        return self.steps > 0

    @classmethod
    def from_args(cls, args: Any) -> "PostMappingPoseAlignConfig":
        return cls(
            steps=int(getattr(args, "streaming_post_mapping_pose_align_steps", 0) or 0),
            lr=float(getattr(args, "streaming_post_mapping_pose_align_lr", 1e-3)),
            tier=str(getattr(args, "streaming_post_mapping_pose_align_tier", "half")),
            max_deg=float(getattr(args, "streaming_post_mapping_pose_align_max_deg", 1.0)),
            max_mm=float(getattr(args, "streaming_post_mapping_pose_align_max_mm", 20.0)),
            exposure_align=bool(
                getattr(args, "streaming_post_mapping_exposure_align", False)
            ),
            stride=int(getattr(args, "streaming_post_mapping_pose_align_stride", 1) or 1),
            max_frames=int(
                getattr(args, "streaming_post_mapping_pose_align_max_frames", 0) or 0
            ),
            time_budget_s=float(
                getattr(args, "streaming_post_mapping_pose_align_time_budget_s", 0.0) or 0.0
            ),
            interpolate_window=int(
                getattr(args, "streaming_post_mapping_pose_align_propagate_window", 1) or 1
            ),
            warm_start=str(
                getattr(args, "streaming_post_mapping_pose_align_warm_start", "on")
            ).strip().lower()
            != "off",
        )


@dataclass
class PostMappingPoseAlignStats:
    """Mutable run-long counters -- see ``as_summary_dict`` for the keys
    threaded into ``causal_replay_summary.json``."""

    frames_considered: int = 0
    aligned: int = 0
    skipped_disabled: int = 0
    skipped_not_train: int = 0
    skipped_no_camera: int = 0
    # feat/admission-pose-refine follow-up (2026-09-07, kitchen1 selection-
    # bias fix): distinct from `skipped_no_camera` (no raw source-frame
    # record at all for this candidate's ordinal, see `build_frame_handles_
    # from_source`'s `by_ordinal.get`) -- this counts a frame that DOES have
    # a raw source record but whose RGB could not be materialized on demand
    # from the source/decode cache (a genuine decode failure, e.g. a missing
    # or corrupt file), tried specifically because this pass no longer
    # requires DB-resident RGB up front.
    skipped_no_rgb: int = 0
    skipped_error: int = 0
    clamped_rotation: int = 0
    clamped_translation: int = 0
    exposure_fitted: int = 0
    sum_delta_deg: float = 0.0
    max_delta_deg: float = 0.0
    sum_delta_mm: float = 0.0
    max_delta_mm: float = 0.0
    sum_residual_before: float = 0.0
    sum_residual_after: float = 0.0
    n_residual_after: int = 0
    # feat/admission-pose-refine follow-up (2026-09-07): `residual_after`
    # above is measured against the RAW (uncorrected) GT -- exactly the
    # "pose residual polluted by exposure" the joint pose+exposure fit must
    # not leave unreported. Once `config.exposure_align` fits gain/bias for
    # a frame, this tracks the SAME final-step loss recomputed against that
    # frame's own exposure-corrected target (`(gt - bias) / gain`, clamped),
    # i.e. the residual the fitted pose would show once exposure is no
    # longer conflated with it. Populated only when exposure_align is on AND
    # the fit succeeded -- absent (n==0) otherwise, exactly like
    # `residual_after`'s own "not every frame reports a loss" convention.
    sum_residual_after_exposure_corrected: float = 0.0
    n_residual_after_exposure_corrected: int = 0
    wall_time_s: float = 0.0
    error_type_counts: dict = field(default_factory=dict)
    # PR #1851 follow-up (2026-09-07, stride/max_frames/time_budget): a
    # frame this pass chose NOT to fit directly (stride skip, `max_frames`-
    # forced stride, or the time budget being spent) but instead assigned an
    # interpolated pose (+ exposure) delta from its ALIGNED temporal
    # neighbours, via the same windowed interpolation
    # `post_mapping_pose_align_propagate.py` uses for holdout propagation.
    # Distinct from `aligned` (a real Adam fit ran) -- both counts feed
    # `FrameObservationDB`/refinement targets identically via `on_aligned`,
    # but only `aligned` frames ran the expensive per-frame render/backward
    # loop.
    interpolated: int = 0
    # A stride-skipped/budget-skipped frame for which NO aligned neighbour
    # existed within this pass's own eligible-frame span to interpolate
    # from (e.g. every frame's own fit failed) -- and a frame lacking the
    # `initial_c2w` a caller must supply for interpolation to have any pose
    # to compose the interpolated delta onto. Kept separate from the
    # `skipped_*` family above (those all describe direct-alignment
    # failures) since this is specifically an interpolation-path failure.
    skipped_interpolation_no_neighbour: int = 0
    # `ceil(n_eligible_train / max_frames)` when `max_frames` forced a wider
    # spacing than `config.stride` alone; otherwise simply `config.stride`.
    # `1` (both knobs at their defaults) reproduces "every eligible frame is
    # selected for alignment" byte-for-byte.
    effective_stride: int = 1
    # True once `config.time_budget_s` was spent and this pass switched the
    # rest of its selected-for-alignment frames to interpolation instead of
    # fitting them. Always False when `time_budget_s <= 0` (unlimited).
    budget_exhausted: bool = False
    _last_delta6: Any = None
    _exceptions_logged: int = 0
    _max_exceptions_logged: int = 3

    def record_exception(self, exc: BaseException) -> None:
        name = type(exc).__name__
        self.error_type_counts[name] = self.error_type_counts.get(name, 0) + 1
        if self._exceptions_logged < self._max_exceptions_logged:
            self._exceptions_logged += 1
            print(
                f"[post-mapping-pose-align] alignment failed "
                f"(exception {self._exceptions_logged}/{self._max_exceptions_logged} logged, "
                f"type={name}):\n{traceback.format_exc()}",
                flush=True,
            )

    def record_aligned(
        self,
        delta_deg: float,
        delta_mm: float,
        residual_initial: Optional[float],
        residual_final: Optional[float],
        residual_final_exposure_corrected: Optional[float] = None,
    ) -> None:
        self.aligned += 1
        self.sum_delta_deg += delta_deg
        self.max_delta_deg = max(self.max_delta_deg, delta_deg)
        self.sum_delta_mm += delta_mm
        self.max_delta_mm = max(self.max_delta_mm, delta_mm)
        if residual_initial is not None:
            self.sum_residual_before += residual_initial
        if residual_final is not None:
            self.sum_residual_after += residual_final
            self.n_residual_after += 1
        if residual_final_exposure_corrected is not None:
            self.sum_residual_after_exposure_corrected += residual_final_exposure_corrected
            self.n_residual_after_exposure_corrected += 1

    @property
    def skipped_total(self) -> int:
        return (
            self.skipped_disabled
            + self.skipped_not_train
            + self.skipped_no_camera
            + self.skipped_no_rgb
            + self.skipped_error
        )

    @property
    def mean_delta_deg(self) -> float:
        return self.sum_delta_deg / self.aligned if self.aligned else 0.0

    @property
    def mean_delta_mm(self) -> float:
        return self.sum_delta_mm / self.aligned if self.aligned else 0.0

    @property
    def mean_residual_before(self) -> float:
        return self.sum_residual_before / self.aligned if self.aligned else 0.0

    @property
    def mean_residual_after(self) -> float:
        return (
            self.sum_residual_after / self.n_residual_after
            if self.n_residual_after
            else 0.0
        )

    @property
    def mean_residual_after_exposure_corrected(self) -> float:
        return (
            self.sum_residual_after_exposure_corrected
            / self.n_residual_after_exposure_corrected
            if self.n_residual_after_exposure_corrected
            else 0.0
        )

    def as_summary_dict(self) -> dict:
        return {
            # Length of the `handles` sequence `align_training_frames` was
            # actually called with -- i.e. the FrameObservationDB-retained
            # population `build_frame_handles_from_source` queried
            # (`historical_eligible`, regardless of RGB modality
            # eligibility/materializability -- see that function's own
            # docstring for the kitchen1 finding that motivated dropping
            # those two filters). NOT the same as the total number of frames
            # this run supervised
            # during mapping: a frame that was supervised but never became
            # historically retained (evicted/redundant, never durably
            # retained) is simply absent from `handles` -- it does not
            # appear here, in `frames_aligned`, OR in any `skipped_*`
            # counter. Compare this number (not `frames_aligned +
            # frames_skipped` alone) against a run's own supervised-frame
            # count to see how much of the true training population this
            # pass actually reached.
            "post_mapping_pose_align_frames_considered": self.frames_considered,
            "post_mapping_pose_align_frames_aligned": self.aligned,
            "post_mapping_pose_align_frames_skipped": self.skipped_total,
            "post_mapping_pose_align_skipped_not_train": self.skipped_not_train,
            "post_mapping_pose_align_skipped_no_camera": self.skipped_no_camera,
            "post_mapping_pose_align_skipped_no_rgb": self.skipped_no_rgb,
            "post_mapping_pose_align_skipped_error": self.skipped_error,
            "post_mapping_pose_align_error_types": dict(self.error_type_counts),
            "post_mapping_pose_align_clamped_rotation": self.clamped_rotation,
            "post_mapping_pose_align_clamped_translation": self.clamped_translation,
            "post_mapping_pose_align_exposure_fitted": self.exposure_fitted,
            "post_mapping_pose_align_mean_delta_deg": self.mean_delta_deg,
            "post_mapping_pose_align_max_delta_deg": self.max_delta_deg,
            "post_mapping_pose_align_mean_delta_mm": self.mean_delta_mm,
            "post_mapping_pose_align_max_delta_mm": self.max_delta_mm,
            "post_mapping_pose_align_mean_residual_before": self.mean_residual_before,
            "post_mapping_pose_align_mean_residual_after": self.mean_residual_after,
            "post_mapping_pose_align_mean_residual_after_exposure_corrected": (
                self.mean_residual_after_exposure_corrected
            ),
            "post_mapping_pose_align_wall_time_s": self.wall_time_s,
            # PR #1851 follow-up (2026-09-07): stride/max_frames/time_budget.
            "post_mapping_pose_align_frames_interpolated": self.interpolated,
            "post_mapping_pose_align_skipped_interpolation_no_neighbour": (
                self.skipped_interpolation_no_neighbour
            ),
            "post_mapping_pose_align_effective_stride": self.effective_stride,
            "post_mapping_pose_align_budget_exhausted": self.budget_exhausted,
        }


# --------------------------------------------------------------------------
# Frame handle (decouples the algorithm from ObservationDB/dataset types)
# --------------------------------------------------------------------------


@dataclass
class PoseAlignFrameHandle:
    """One frame's identity + lazy camera builder + durable write-back.

    ``order_key`` sorts the pass into a stable, temporally-meaningful order
    so ``warm_start`` (seeding a frame's delta from the previous frame's
    fitted delta -- consecutive frames usually need similar corrections,
    exactly ``admission_pose_refine.py``'s convention) is well-founded;
    without a temporal sort, dict/set iteration order would warm-start from
    an arbitrary, spatially unrelated neighbour.

    ``build_camera`` returns a fully renderable ``Camera`` (GT already
    materialized or lazily materializable via
    ``Camera.materialize_pending_image``) or ``None`` if this frame's pixels
    are not available at the endpoint (counted as ``skipped_no_camera``,
    never an error).

    ``on_aligned(c2w_new, gain, bias)`` durably records the fitted result;
    ``gain``/``bias`` are ``None`` unless ``config.exposure_align``.
    """

    frame_id: Any
    is_train: bool
    order_key: float
    build_camera: Callable[[], Any]
    on_aligned: Callable[[np.ndarray, Optional[tuple], Optional[tuple]], None]
    # Trailing/defaulted so existing handle construction (tests, any other
    # caller) that predates this field stays source-compatible. Threaded
    # into `jsonl_diagnostics_sink` records (see `align_training_frames`)
    # so a real run's post_mapping_pose_align.jsonl can be keyed by source
    # ordinal -- the same identity
    # `post_mapping_pose_align_propagate.py`'s holdout-propagation reads
    # deltas by -- rather than only the internal `frame_id`. `None` when a
    # caller has no stable source ordinal to offer (never required for
    # `align_training_frames` itself to run correctly).
    source_ordinal: Optional[int] = None
    # PR #1851 follow-up (2026-09-07, stride/max_frames/time_budget): this
    # frame's PRE-pass camera-to-world pose, as a plain ``(4, 4)`` numpy
    # array -- captured cheaply (a dict lookup, no camera build, no RGB
    # decode, no render) at handle-construction time, i.e. before any
    # alignment in this pass has run. Only consumed when this frame is
    # interpolated rather than directly aligned (`align_training_frames`
    # needs SOME "before" pose to compose the interpolated delta6 onto, and
    # deliberately does not call the expensive `build_camera` just to read
    # one back -- that would defeat the whole point of skipping this
    # frame's fit). `None` when a caller has no cheap pose to offer; such a
    # frame cannot be interpolated (counted in
    # `PostMappingPoseAlignStats.skipped_interpolation_no_neighbour`) even
    # if temporal neighbours were aligned.
    initial_c2w: Optional[np.ndarray] = None


class PoseAlignNoRgbError(RuntimeError):
    """Raised by a ``PoseAlignFrameHandle.build_camera`` when the frame's
    identity/pose record was found but its RGB could not be materialized on
    demand (kitchen1 selection-bias follow-up, see ``build_frame_handles_
    from_source``). Distinguished from a bare ``None`` return (no raw
    source-frame record for this candidate at all) so
    :func:`align_training_frames` can attribute the skip specifically to
    ``PostMappingPoseAlignStats.skipped_no_rgb`` rather than folding it into
    the unrelated ``skipped_no_camera``/``skipped_error`` buckets.
    """


def _delta6_from_matrix(delta: torch.Tensor) -> torch.Tensor:
    """Recover the axis-angle/translation 6-vector from a (4, 4) SE(3) delta.

    Same derivation as ``admission_pose_refine.refine_admission_pose``'s
    inline block (not exported there as a reusable helper) -- duplicated
    here rather than refactoring that already-tested hot path.
    """
    R = delta[:3, :3]
    t = delta[:3, 3]
    trace = torch.clamp((torch.einsum("ii->", R) - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.arccos(trace)
    if float(theta.item()) < 1e-8:
        omega = torch.zeros(3, device=delta.device, dtype=delta.dtype)
    else:
        axis = torch.stack(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2.0 * torch.sin(theta))
        omega = axis * theta
    return torch.cat([omega, t])


def _delta6_from_c2w_pair(
    c2w_before: np.ndarray, c2w_after: np.ndarray
) -> tuple:
    """Recover the camera-local-frame SE(3) 6-vector between two poses.

    ``c2w_after == c2w_before @ delta_matrix(delta6)`` (see
    ``eligsir.pose_opt.apply_pose_delta``'s own convention, which this
    module's bake step already uses) -- so
    ``delta = inv(c2w_before) @ c2w_after``. Reuses :func:`_delta6_from_matrix`
    (the tested log-map derivation) rather than re-deriving it in numpy.
    Used by ``build_frame_handles_from_source``'s write-back to durably
    persist ``FrameRecord.pose_align_delta6`` without changing
    ``PoseAlignFrameHandle.on_aligned``'s call signature (that callback only
    ever receives the AFTER pose; the BEFORE pose is read back from the
    record itself, which still holds the pre-write value at the moment the
    write-back runs).
    """
    before_t = torch.as_tensor(np.asarray(c2w_before, dtype=np.float64))
    after_t = torch.as_tensor(np.asarray(c2w_after, dtype=np.float64))
    delta = torch.inverse(before_t) @ after_t
    delta6 = _delta6_from_matrix(delta.to(torch.float32))
    return tuple(float(v) for v in delta6.tolist())


def align_training_frames(
    handles: Sequence[PoseAlignFrameHandle],
    gaussians: Any,
    render_fn: Callable,
    pipe: Any,
    background: Any,
    *,
    config: PostMappingPoseAlignConfig,
    stats: PostMappingPoseAlignStats,
    ssim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    diagnostics_sink: Optional[Callable[[dict], None]] = None,
) -> None:
    """Align every TRAIN handle's pose once against the frozen ``gaussians``.

    A no-op (every handle skipped via ``skipped_disabled``) when
    ``config.enabled`` is ``False`` -- callers can unconditionally construct
    handles/config and call this, exactly like
    ``utils/eval_pose_align.py::align_and_score_test_cameras``'s
    ``n_steps <= 0`` convention.

    PR #1851 follow-up (2026-09-07, stride/max_frames/time_budget): not
    every eligible TRAIN handle necessarily gets a real Adam fit anymore.
    ``config.stride``/``config.max_frames``/``config.time_budget_s`` select
    a SUBSET of eligible train handles (always including the first and
    last, in temporal ``order_key`` order) to actually align; every other
    eligible train handle instead gets an INTERPOLATED delta (+ exposure)
    from its nearest ALIGNED neighbours, via the same windowed
    Gaussian-weighted interpolation ``post_mapping_pose_align_propagate.py``
    already uses for holdout propagation
    (``config.interpolate_window``/``interpolate_delta_windowed``). Both
    aligned and interpolated frames are written back identically via
    ``handle.on_aligned`` -- a downstream reader (``FrameObservationDB``,
    refinement targets, holdout propagation's own read-back of
    ``pose_align_delta6``/``exposure_gain``/``exposure_bias``) cannot tell
    the two apart except via the diagnostics jsonl's ``source`` field.
    """

    stats.frames_considered += len(handles)
    if not config.enabled:
        stats.skipped_disabled += len(handles)
        return

    from eligsir.utils.eval_pose_align import _fit_pose_delta, _frozen_gaussians
    from scene.cameras import prepare_camera_for_render

    if ssim_fn is None:
        from eligsir.utils.loss_utils import ssim as ssim_fn  # local import: heavy/GPU-ish deps

    from eligsir.exposure_opt import fit_affine

    # The device every handle's camera/GT must land on before `_fit_pose_delta`
    # ever sees them -- derived once, outside the per-frame `try:`, so an
    # import/attribute error here fails loudly instead of being swallowed
    # into `skipped_error` for every single frame (exactly the failure mode
    # this fix addresses; see the comment at the `prepare_camera_for_render`
    # call site below).
    model_device = gaussians.get_xyz.device

    started = time.perf_counter()
    ordered = sorted(handles, key=lambda h: h.order_key)
    train_handles = [h for h in ordered if h.is_train]
    stats.skipped_not_train += len(ordered) - len(train_handles)

    n_eligible = len(train_handles)
    effective_stride = max(1, int(config.stride))
    if config.max_frames > 0 and n_eligible > config.max_frames:
        effective_stride = max(effective_stride, math.ceil(n_eligible / config.max_frames))
    stats.effective_stride = effective_stride

    selected_positions = set(range(0, n_eligible, effective_stride))
    if n_eligible > 0:
        selected_positions.add(n_eligible - 1)

    # Filled in as the alignment loop below runs, keyed by the SAME
    # `source_ordinal`-or-positional-index identity `interpolate_delta_
    # windowed`/`interpolate_exposure_windowed` key their neighbour search
    # by -- consumed by the interpolation pass further down, which only
    # runs once every selected position has had its chance to align (a
    # not-yet-selected frame may still need a LATER aligned frame as its
    # "after" neighbour).
    aligned_delta_by_ordinal: dict = {}
    aligned_exposure_by_ordinal: dict = {}
    # (position, handle, ordinal) triples this loop decided NOT to align
    # directly -- either never selected (stride/max_frames) or selected but
    # skipped once the time budget was spent -- deferred to the
    # interpolation pass below.
    to_interpolate: list = []
    budget_exhausted = False

    with _frozen_gaussians(gaussians):
        for pos, handle in enumerate(train_handles):
            ordinal = handle.source_ordinal if handle.source_ordinal is not None else pos

            if pos not in selected_positions:
                to_interpolate.append((pos, handle, ordinal))
                continue

            if (
                config.time_budget_s > 0
                and not budget_exhausted
                and (time.perf_counter() - started) >= config.time_budget_s
            ):
                budget_exhausted = True
                stats.budget_exhausted = True
            if budget_exhausted:
                to_interpolate.append((pos, handle, ordinal))
                continue

            try:
                cam = handle.build_camera()
                if cam is None:
                    # No raw source-frame record at all for this candidate's
                    # ordinal (`by_ordinal.get` miss in `build_frame_handles_
                    # from_source`) -- distinct from `PoseAlignNoRgbError`
                    # below, which means the record exists but its RGB could
                    # not be decoded.
                    stats.skipped_no_camera += 1
                    continue
                # `handle.build_camera()` (see
                # `build_frame_handles_from_source`/
                # `build_causal_holdout_cameras`) returns a camera whose GT
                # is a LAZY, CPU-owned loader (`Camera.materialize_pending_image`
                # deliberately refuses a non-CPU decode) while
                # `world_view_transform`/`full_proj_transform` were already
                # built on `data_device` (== the Gaussians' device) at
                # construction time. Calling only `materialize_pending_image`
                # here (as an earlier version of this loop did) leaves
                # `original_image` on CPU while `render_fn(gaussians, ...)`
                # renders on `gaussians`' device, so `_fit_pose_delta`'s
                # `img - gt` raised "Expected all tensors to be on the same
                # device" for every frame (#1851 GPU run
                # psnr26tum38_fr3_twopass_align50). `prepare_camera_for_render`
                # is this repo's canonical materialize-and-upload helper
                # (see `train.py`/`train_streaming.py`/
                # `production_application.py`'s own calls before rendering):
                # it both decodes the pending image AND moves
                # `original_image`/the `gsplat` cache onto the requested
                # device, so `gt` below ends up on the same device the
                # render output does. Same fix covers the exposure-align
                # hook further down, which also reads `original_image`.
                prepare_camera_for_render(cam, device=model_device)
                original_image = cam.original_image
                gt = _tiered_gt(original_image, config.tier)
                tiered_cam = _TieredCamera(cam, config.tier)

                init_delta6 = stats._last_delta6 if config.warm_start else None
                loss_out: dict = {}
                delta = _fit_pose_delta(
                    tiered_cam,
                    gaussians,
                    render_fn,
                    pipe,
                    background,
                    gt,
                    ssim_fn,
                    config.steps,
                    lr=config.lr,
                    lambda_dssim=config.lambda_dssim,
                    init_delta6=init_delta6,
                    loss_out=loss_out,
                )

                delta6 = _delta6_from_matrix(delta)
                if not bool(torch.isfinite(delta6).all()):
                    stats.skipped_error += 1
                    stats._last_delta6 = None
                    continue

                clamped6, clamped_rot, clamped_trans = _clamp_delta(
                    delta6, config.max_deg, config.max_mm
                )
                if clamped_rot:
                    stats.clamped_rotation += 1
                if clamped_trans:
                    stats.clamped_translation += 1
                clamped_delta = se3_delta_matrix(clamped6)

                gain: Optional[tuple] = None
                bias: Optional[tuple] = None
                residual_final_exposure_corrected: Optional[float] = None
                if config.exposure_align:
                    with torch.no_grad():
                        aligned_full = apply_pose_delta(cam, clamped_delta)
                        out = render_fn(aligned_full, gaussians, pipe, background)
                        render_img = out["render"] if isinstance(out, dict) else out
                        gain_t, bias_t = fit_affine(render_img.detach(), original_image)
                        gain = tuple(float(v) for v in gain_t.flatten().tolist())
                        bias = tuple(float(v) for v in bias_t.flatten().tolist())
                        # The joint pose+exposure fit (pose fit runs to
                        # completion FIRST, exposure fit runs once against
                        # its final aligned render -- see this function's own
                        # module docstring) must not leave the REPORTED
                        # residual polluted by exposure: `loss_out["final"]`
                        # above is measured against the RAW, uncorrected
                        # `original_image`, so a systematic exposure mismatch
                        # (not a pose error at all) inflates it. Recompute
                        # the SAME (1 - lambda_dssim) * L1 + lambda_dssim *
                        # DSSIM loss the pose fit itself used, but against
                        # this frame's own exposure-corrected target
                        # (`(gt - bias) / gain`, clamped) -- the residual a
                        # perfectly pose-aligned, exposure-decoupled frame
                        # would actually show. NOTE: this is at NATIVE
                        # resolution (`cam`/`original_image`, matching where
                        # `fit_affine` itself fits), not `config.tier` --
                        # `loss_out["final"]`/`mean_residual_after` above is
                        # measured on the TIERED camera/GT
                        # (`tiered_cam`/`gt` from `_tiered_gt`). The two
                        # summary keys are each internally consistent but not
                        # a same-resolution apples-to-apples pair whenever
                        # `config.tier != "native"`.
                        corrected_gt = torch.clamp(
                            (original_image - bias_t.view(-1, 1, 1))
                            / gain_t.view(-1, 1, 1),
                            0.0,
                            1.0,
                        )
                        l1_corrected = torch.abs(render_img - corrected_gt).mean()
                        dssim_corrected = 1.0 - ssim_fn(render_img, corrected_gt)
                        residual_final_exposure_corrected = float(
                            (
                                (1.0 - config.lambda_dssim) * l1_corrected
                                + config.lambda_dssim * dssim_corrected
                            ).item()
                        )
                    stats.exposure_fitted += 1

                # Captured immediately BEFORE the bake below -- the pose this
                # frame's fit actually started from -- so `diagnostics_sink`
                # can persist a full (c2w_before, c2w_after) pair, not just
                # the fitted delta magnitude. Deliberately read here (post-fit,
                # on the real render device) rather than earlier in this loop
                # (pre-fit): a caller under test may swap in a fake fit that
                # raises before ever touching a live device (see
                # ``test_lazy_camera_gt_is_moved_to_gaussians_device_before_fit``,
                # which uses a "meta" device camera specifically to assert the
                # fit is never reached) -- an eager conversion earlier would
                # run (and fail, since a meta tensor has no data to copy out
                # of) before that assertion gets its chance. See item (1) of
                # the propagate-holdout feature: `post_mapping_pose_align.jsonl`
                # gains these keys unconditionally whenever a frame is aligned.
                w2c_before = cam.world_view_transform.transpose(0, 1)
                c2w_before = torch.inverse(w2c_before).detach().to("cpu").numpy()

                bake_pose_delta_into_camera(cam, clamped_delta)
                deg, mm = _delta_deg_mm(clamped6)
                stats.record_aligned(
                    deg,
                    mm,
                    loss_out.get("initial"),
                    loss_out.get("final"),
                    residual_final_exposure_corrected,
                )
                stats._last_delta6 = clamped6.detach().to("cpu")

                # Read the aligned pose back from `world_view_transform`
                # (always updated by `bake_pose_delta_into_camera`), not
                # `.c2w` -- unlike the admission-time hook's cameras
                # (`Camera.from_prepared_device_payloads`, which sets
                # `.c2w`), the report-style cameras this module's own
                # production wiring builds
                # (`build_causal_holdout_cameras` -> the plain `Camera(R=,
                # T=, ...)` constructor) never set `.c2w` at all, so
                # `bake_pose_delta_into_camera`'s `hasattr(camera, "c2w")`
                # guard is a no-op for them. `world_view_transform` is
                # stored transposed (row-vector convention), matching
                # `apply_pose_delta`/`bake_pose_delta_into_camera`'s own
                # derivation.
                w2c_new = cam.world_view_transform.transpose(0, 1)
                c2w_new = torch.inverse(w2c_new).detach().to("cpu").numpy()
                handle.on_aligned(c2w_new, gain, bias)

                # Made available to the interpolation pass below as this
                # frame's fitted neighbour delta (+ exposure, when fitted) --
                # keyed by the SAME `ordinal` identity
                # `interpolate_delta_windowed`/`interpolate_exposure_windowed`
                # search by.
                aligned_delta_by_ordinal[ordinal] = clamped6.detach().to("cpu").numpy()
                if gain is not None and bias is not None:
                    aligned_exposure_by_ordinal[ordinal] = (gain, bias)

                if diagnostics_sink is not None:
                    diagnostics_sink(
                        {
                            "frame_id": handle.frame_id,
                            "source_ordinal": handle.source_ordinal,
                            "order_key": handle.order_key,
                            "source": "aligned",
                            "steps": int(config.steps),
                            "delta6": [float(v) for v in clamped6.detach().tolist()],
                            "c2w_before": c2w_before.tolist(),
                            "c2w_after": c2w_new.tolist(),
                            "delta_deg": deg,
                            "delta_mm": mm,
                            "clamped_rotation": bool(clamped_rot),
                            "clamped_translation": bool(clamped_trans),
                            "residual_initial": loss_out.get("initial"),
                            "residual_final": loss_out.get("final"),
                            "residual_final_exposure_corrected": (
                                residual_final_exposure_corrected
                            ),
                            "exposure_gain": gain,
                            "exposure_bias": bias,
                        }
                    )
            except PoseAlignNoRgbError:
                stats.skipped_no_rgb += 1
                stats._last_delta6 = None
            except Exception as exc:  # noqa: BLE001 - one bad frame must not abort the pass
                stats.skipped_error += 1
                stats.record_exception(exc)
                stats._last_delta6 = None
    stats.wall_time_s += time.perf_counter() - started

    _interpolate_skipped_train_frames(
        to_interpolate,
        aligned_delta_by_ordinal,
        aligned_exposure_by_ordinal,
        config=config,
        stats=stats,
        diagnostics_sink=diagnostics_sink,
    )


def _interpolate_skipped_train_frames(
    to_interpolate: Sequence[tuple],
    aligned_delta_by_ordinal: dict,
    aligned_exposure_by_ordinal: dict,
    *,
    config: PostMappingPoseAlignConfig,
    stats: PostMappingPoseAlignStats,
    diagnostics_sink: Optional[Callable[[dict], None]],
) -> None:
    """Assign every deferred (stride-skipped / budget-skipped) train handle
    an interpolated pose (+ exposure) delta from its ALIGNED temporal
    neighbours, via the SAME windowed Gaussian-weighted interpolation
    ``post_mapping_pose_align_propagate.py`` uses for holdout propagation.

    Runs entirely off-device (pure numpy, no render, no Gaussians) -- this
    is precisely the cost `--streaming_post_mapping_pose_align_stride`/
    `..._max_frames`/`..._time_budget_s` exist to avoid paying for every
    training frame. ``max_gap`` is deliberately left unbounded (``None``
    means "search the whole aligned population") since `align_training_
    frames` always aligns the first and last eligible train handle -- every
    deferred frame is guaranteed to lie between two aligned ordinals (or
    coincide with one), so there is no "propagate_max_gap"-style "nothing
    nearby" failure mode to guard against here (unlike holdout propagation,
    where a holdout can legitimately sit far from every training frame).
    """

    if not to_interpolate:
        return

    # No aligned frame exists at all (e.g. every direct fit failed) -- every
    # deferred frame is unreachable.
    if not aligned_delta_by_ordinal:
        stats.skipped_interpolation_no_neighbour += len(to_interpolate)
        return

    all_ordinals = [int(o) for o in aligned_delta_by_ordinal.keys()]
    max_gap = max(all_ordinals) - min(all_ordinals) + 1 if all_ordinals else 1

    for pos, handle, ordinal in to_interpolate:
        if handle.initial_c2w is None:
            stats.skipped_interpolation_no_neighbour += 1
            continue

        delta = interpolate_delta_windowed(
            aligned_delta_by_ordinal, ordinal, max_gap, config.interpolate_window
        )
        if delta is None:
            stats.skipped_interpolation_no_neighbour += 1
            continue

        c2w_before = np.asarray(handle.initial_c2w, dtype=np.float64)
        c2w_new = apply_delta6_to_c2w(c2w_before, delta)

        gain: Optional[tuple] = None
        bias: Optional[tuple] = None
        if config.exposure_align and aligned_exposure_by_ordinal:
            exposure = interpolate_exposure_windowed(
                aligned_exposure_by_ordinal, ordinal, max_gap, config.interpolate_window
            )
            if exposure is not None:
                gain_arr, bias_arr = exposure
                gain = tuple(float(v) for v in np.asarray(gain_arr).flatten().tolist())
                bias = tuple(float(v) for v in np.asarray(bias_arr).flatten().tolist())

        deg, mm = _delta_deg_mm(torch.as_tensor(delta, dtype=torch.float32))
        stats.interpolated += 1
        handle.on_aligned(c2w_new.astype(np.float32), gain, bias)

        if diagnostics_sink is not None:
            diagnostics_sink(
                {
                    "frame_id": handle.frame_id,
                    "source_ordinal": handle.source_ordinal,
                    "order_key": handle.order_key,
                    "source": "interpolated",
                    "delta6": [float(v) for v in np.asarray(delta).tolist()],
                    "c2w_before": c2w_before.tolist(),
                    "c2w_after": c2w_new.tolist(),
                    "delta_deg": deg,
                    "delta_mm": mm,
                    "exposure_gain": gain,
                    "exposure_bias": bias,
                }
            )


def jsonl_diagnostics_sink(model_path: str) -> Callable[[dict], None]:
    """Build a ``diagnostics_sink`` appending one JSON line per aligned frame
    to ``<model_path>/post_mapping_pose_align.jsonl``. Best-effort, matches
    ``admission_pose_refine.jsonl_diagnostics_sink``."""

    path = os.path.join(model_path, "post_mapping_pose_align.jsonl")

    def _sink(record: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str))
                fh.write("\n")
        except Exception:  # noqa: BLE001 - diagnostics must never break the pass
            pass

    return _sink
