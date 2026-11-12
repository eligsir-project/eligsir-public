"""OnlineController — real-time map-age-aware mapping controller.

Phase B (Milestone 2) of the online SLAM plan. Wraps
:class:`eligsir.mapping.scheduler.MappingScheduler` with a discrete
controller that selects a ``controller_mode`` based on the gap between
the newest arrived sensor frame and the newest frame actually mapped
into the Gaussian map (``map_age``).

Modes (hysteresis on ``map_age``):

  - ``TRACK_ONLY``     — Map-age hard ceiling hit. SLAM keeps tracking but
                        no insertion / no training. Resumes normal mode
                        when map_age falls below the catch-up entry.
  - ``CATCH_UP``       — Map-age over the catch-up entry threshold. Key
                        frames only, few steps, no growth.
  - ``BALANCED``       — Default. Key frames + small local optimisation
                        budget.
  - ``QUALITY_IDLE``   — Map-age low and input stable. Extra PSNR-style
                        refinement: more replay steps, growth / densification
                        allowed.

The controller is *additive* — :class:`MappingScheduler` still produces
the per-frame :class:`MappingDecision` and the controller's role is to
shape that decision (steps, allow_growth, allow_insertion, freeze) before
the training loop consumes it.

Diagnostic counters for RViz / TensorBoard live on the controller as
``stats()`` and ``to_tensorboard_dict()``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from eligsir.contracts.types import TrackingResult
from .mapping_scheduler import MappingDecision, MappingScheduler


class ControllerMode(str, Enum):
    """Discrete controller states (Section 4 of the plan)."""

    TRACK_ONLY = "track_only"
    CATCH_UP = "catch_up"
    BALANCED = "balanced"
    QUALITY_IDLE = "quality_idle"


@dataclass
class OnlineConfig:
    """Tunables for :class:`OnlineController`.

    All thresholds default to the values in
    ``docs/ONLINE_SLAM_PLAN.md`` Section 3.

    Attributes:
        map_age_target_s: Soft target for map age (default 0.5 s).
        map_age_catch_up_enter_s: Enter CATCH_UP when map_age > this
            (default 0.8 s).
        map_age_catch_up_exit_s: Exit CATCH_UP when map_age < this
            (default 0.3 s).
        map_age_track_only_enter_s: Hard ceiling — enter TRACK_ONLY
            when map_age > this (default 1.5 s).
        map_age_track_only_exit_s: Exit TRACK_ONLY when map_age < this
            (default 0.5 s).
        min_steps: Floor on train_steps per frame.
        max_steps: Ceiling on train_steps per frame.
        quality_idle_input_fps_stable: When measured input FPS is within
            ±20% of target for at least this many seconds, allow
            QUALITY_IDLE (default 0.5 s).
        quality_idle_fps_tolerance: Fractional tolerance for the input
            stability check (default 0.2 = ±20% of target input FPS).
        iter_dt_ema_beta: EMA smoothing for measured iteration cost
            (default 0.5 — high responsiveness for online mode).
        adaptive_steps: If True, scale train_steps down as
            ``measured_cost_ema * effective_spf`` approaches
            ``budget_per_iter`` (default True).
    """

    map_age_target_s: float = 0.5
    map_age_catch_up_enter_s: float = 0.8
    map_age_catch_up_exit_s: float = 0.3
    map_age_track_only_enter_s: float = 1.5
    map_age_track_only_exit_s: float = 0.5
    min_steps: int = 1
    max_steps: int = 50
    quality_idle_input_fps_stable: float = 0.5
    quality_idle_fps_tolerance: float = 0.2
    iter_dt_ema_beta: float = 0.5
    adaptive_steps: bool = True
    # Part D3 (restored 2026-07-26 -- see docs/architecture/2026-06-30-lost-on-
    # merge-inventory.md; original commit 41082ac): velocity-adaptive
    # refinement. A stalled/slow camera brings little new geometry per frame,
    # so spend MORE steps refining what's there; a fast camera brings lots of
    # new geometry, so spend FEWER steps to keep up. Keyed on per-frame camera
    # translation (metres), fed via record_camera_speed(). Off by default.
    velocity_adaptive_steps: bool = False
    velocity_stall_mps: float = 0.02   # at/below this per-frame motion -> max boost
    velocity_fast_mps: float = 0.30    # at/above this -> max cut
    velocity_step_gain: float = 1.5    # boost/cut magnitude (×gain when stalled, /gain when fast)
    velocity_speed_ema_beta: float = 0.6
    pause_idle_threshold_s: float = 0.0
    """If > 0, the wall-clock seconds without a new input arrival
    before the controller flags ``pause=True``. 0 keeps the legacy
    heuristic (twice the target period) for backwards compatibility."""


@dataclass
class OnlineDecision:
    """Per-frame online decision layered on top of :class:`MappingDecision`.

    Attributes:
        mode: Selected controller mode.
        do_mapping: Whether to run any mapping this frame.
        steps: Number of optimiser steps to run.
        allow_insertion: Whether to insert new Gaussians from depth.
        allow_growth: Whether to allow MCMC growth.
        freeze_persistent: Whether to freeze non-provisional Gaussians.
        accept_frame: Whether the frame should be ingested at all.
        train_steps: Finalised training step count (post adaptive scale).
        insertion_cap: Maximum number of Gaussians to insert this frame.
        sampling_mode: Either ``"legacy"`` or ``"stratified"``.
        publish_rviz: Whether the viewer should publish this iteration.
        confidence: Tracking confidence from the SLAM result.
    """

    mode: ControllerMode
    do_mapping: bool
    steps: int
    allow_insertion: bool
    allow_growth: bool
    freeze_persistent: bool
    accept_frame: bool
    train_steps: int
    insertion_cap: int
    sampling_mode: str
    publish_rviz: bool
    confidence: float


@dataclass
class ControllerStats:
    """Snapshot of controller state for logging / RViz / TensorBoard."""

    mode: str
    map_age_s: float
    sensor_clock: float
    map_clock: float
    iter_dt_ema_s: float
    iter_dt_target_s: float
    measured_input_fps: float
    target_input_fps: float
    backlog: int
    train_steps: int
    accepted_frames: int
    dropped_nonkeyframes: int
    dropped_keyframes: int
    frames_ingested: int
    mapped_frames: int
    pose_fps: float
    controller_mode: str
    effective_spf: int
    confidence: float
    keyframe_indices: List[int] = field(default_factory=list)
    replay_indices: List[int] = field(default_factory=list)
    pause: bool = False
    burst: bool = False
    quality_idle_eligible: bool = False
    extra: Dict[str, float] = field(default_factory=dict)

    def to_tensorboard_dict(self) -> Dict[str, float]:
        """Convert controller stats to TensorBoard scalars.

        Returns:
            Dictionary of scalar names (with "controller/" prefix) to float
            values suitable for TensorBoard logging.
        """
        d: Dict[str, float] = {
            "controller/mode_" + self.mode: 1.0,
            "controller/map_age_s": float(self.map_age_s),
            "controller/iter_dt_ema_s": float(self.iter_dt_ema_s),
            "controller/iter_dt_target_s": float(self.iter_dt_target_s),
            "controller/measured_input_fps": float(self.measured_input_fps),
            "controller/target_input_fps": float(self.target_input_fps),
            "controller/backlog": float(self.backlog),
            "controller/train_steps": float(self.train_steps),
            "controller/accepted_frames": float(self.accepted_frames),
            "controller/dropped_nonkeyframes": float(self.dropped_nonkeyframes),
            "controller/dropped_keyframes": float(self.dropped_keyframes),
            "controller/pose_fps": float(self.pose_fps),
            "controller/confidence": float(self.confidence),
            "controller/pause": 1.0 if self.pause else 0.0,
            "controller/burst": 1.0 if self.burst else 0.0,
            "controller/quality_idle_eligible": 1.0 if self.quality_idle_eligible else 0.0,
        }
        for k, v in self.extra.items():
            d[f"controller/{k}"] = float(v)
        return d


class OnlineController:
    """Real-time map-age-aware controller.

    Wraps a :class:`MappingScheduler` and shapes its decisions through a
    discrete mode state machine. Tracks EMA of iteration cost, per-second
    frame counters, and key-frame indices so the training sampler can
    prioritise keyframes.

    Usage::

        ctrl = OnlineController(scheduler, config)
        decision, mapping_decision = ctrl.tick(
            tracking_result=tr,
            timestamp=newest_sensor_ts,
            backlog=backlog,
            iter_dt=last_iter_dt,
            keyframe_indices=keyframe_indices,
            replay_indices=replay_indices,
        )
    """

    def __init__(
        self,
        scheduler: MappingScheduler,
        config: Optional[OnlineConfig] = None,
        target_input_fps: float = 30.0,
        budget_per_iter_s: float = 0.05,
        insertion_cap: int = 1500,
        sampling_mode: str = "legacy",
        rviz_publish_interval_iters: int = 10,
    ) -> None:
        """Initialize the online controller.

        Args:
            scheduler: Underlying MappingScheduler to wrap.
            config: Controller mode thresholds and parameters, or None for defaults.
            target_input_fps: Target input frame rate (Hz), used for FPS calculations.
            budget_per_iter_s: Wall-time budget per iteration (seconds).
            insertion_cap: Maximum Gaussians to insert per frame.
            sampling_mode: Camera sampling strategy ("legacy" or "stratified").
            rviz_publish_interval_iters: Iterations between RViz publishes.
        """
        self._scheduler = scheduler
        self._config = config or OnlineConfig()
        self._target_input_fps = max(float(target_input_fps), 0.1)
        self._budget_per_iter_s = max(float(budget_per_iter_s), 1e-3)
        self._insertion_cap = int(insertion_cap)
        self._sampling_mode = sampling_mode
        self._rviz_publish_interval = max(int(rviz_publish_interval_iters), 1)

        # State
        self._mode: ControllerMode = ControllerMode.BALANCED
        self._sensor_clock: float = 0.0
        self._map_clock: float = 0.0
        self._map_age_s: float = 0.0
        self._iter_dt_ema: float = self._budget_per_iter_s
        self._last_iter_wall_t: Optional[float] = None
        self._frames_ingested: int = 0
        self._mapped_frames: int = 0
        self._accepted_frames: int = 0
        self._dropped_nonkeyframes: int = 0
        self._dropped_keyframes: int = 0
        self._pose_window: List[float] = []  # recent pose arrival times
        self._input_window: List[float] = []  # recent frame arrival times
        self._quality_idle_eligible: bool = False
        self._first_frame_t: Optional[float] = None
        self._last_mapped_t: Optional[float] = None
        self._last_sensor_t: Optional[float] = None
        self._pause: bool = False
        self._burst: bool = False
        self._last_pause_change_t: Optional[float] = None
        self._keyframe_indices: List[int] = []
        self._replay_indices: List[int] = []
        self._last_confidence: float = 1.0
        # D3: EMA of per-frame camera translation (metres). None until first sample.
        self._cam_speed_ema: Optional[float] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def mode(self) -> ControllerMode:
        """Current controller mode (TRACK_ONLY, CATCH_UP, BALANCED, QUALITY_IDLE)."""
        return self._mode

    @property
    def map_age_s(self) -> float:
        """Map age in seconds (time since last depth insertion)."""
        return self._map_age_s

    @property
    def iter_dt_ema_s(self) -> float:
        """EMA of iteration wall-clock cost in seconds."""
        return self._iter_dt_ema

    @property
    def stats(self) -> ControllerStats:
        """Snapshot of current controller state for telemetry/logging."""
        now = time.perf_counter()
        measured_input_fps = self._fps(self._input_window, now)
        # Clamp measured_input_fps to 2× target so the backlog signal stays
        # bounded under bursts. The raw value is also kept (below) for
        # diagnostics, with the clamped value exposed as the headline.
        clamped_input_fps = min(measured_input_fps, self._target_input_fps * 2.0)
        pose_fps = self._fps(self._pose_window, now)
        return ControllerStats(
            mode=self._mode.value,
            map_age_s=self._map_age_s,
            sensor_clock=self._sensor_clock,
            map_clock=self._map_clock,
            iter_dt_ema_s=self._iter_dt_ema,
            iter_dt_target_s=self._budget_per_iter_s,
            measured_input_fps=clamped_input_fps,
            target_input_fps=self._target_input_fps,
            backlog=int(max(0.0, clamped_input_fps * self._map_age_s)),
            train_steps=int(self._clamp_steps(self._effective_spf())),
            accepted_frames=self._accepted_frames,
            dropped_nonkeyframes=self._dropped_nonkeyframes,
            dropped_keyframes=self._dropped_keyframes,
            frames_ingested=self._frames_ingested,
            mapped_frames=self._mapped_frames,
            pose_fps=pose_fps,
            controller_mode=self._mode.value,
            effective_spf=self._effective_spf(),
            confidence=self._last_confidence,
            keyframe_indices=list(self._keyframe_indices),
            replay_indices=list(self._replay_indices),
            pause=self._pause,
            burst=self._burst,
            quality_idle_eligible=self._quality_idle_eligible,
            extra={
                "min_steps": float(self._config.min_steps),
                "max_steps": float(self._config.max_steps),
                "insertion_cap": float(self._insertion_cap),
                # Raw (unclamped) measured input FPS — useful for burst detection.
                "measured_input_fps_raw": float(measured_input_fps),
            },
        )

    def record_pose_arrival(self) -> None:
        """Call each time a new pose arrives (regardless of map insertion)."""
        self._pose_window.append(time.perf_counter())
        self._prune_windows()

    def record_frame_arrival(self) -> None:
        """Call each time a new RGB-D frame is released from the source."""
        self._input_window.append(time.perf_counter())
        self._prune_windows()

    def record_mapping(self, timestamp: float) -> None:
        """Call after a frame's depth has been inserted into the map.

        Args:
            timestamp: Sensor timestamp of the frame just mapped.
        """
        self._map_clock = max(self._map_clock, float(timestamp))
        self._mapped_frames += 1
        self._last_mapped_t = float(timestamp)
        self._update_map_age()

    def record_ingest(self, timestamp: float) -> None:
        """Call when a frame is released by the FrameScheduler.

        Args:
            timestamp: Sensor timestamp of the ingested frame.
        """
        self._sensor_clock = max(self._sensor_clock, float(timestamp))
        self._frames_ingested += 1
        self._last_sensor_t = float(timestamp)
        self.record_frame_arrival()
        self._update_map_age()

    def record_drop(
        self,
        timestamp: float,
        is_keyframe: bool,
    ) -> None:
        """Record a dropped frame (non-keyframe preferred drop target)."""
        if is_keyframe:
            self._dropped_keyframes += 1
        else:
            self._dropped_nonkeyframes += 1

    def record_camera_speed(self, translation_m: Optional[float]) -> None:
        """Update the EMA of per-frame camera translation (metres) for D3.

        Args:
            translation_m: Distance the camera moved since the previous mapped
                frame, or None to leave the EMA unchanged.
        """
        if translation_m is None or translation_m < 0.0 or not math.isfinite(translation_m):
            return
        beta = self._config.velocity_speed_ema_beta
        if self._cam_speed_ema is None:
            self._cam_speed_ema = float(translation_m)
        else:
            self._cam_speed_ema = (1.0 - beta) * float(translation_m) + beta * self._cam_speed_ema

    def _velocity_step_factor(self) -> float:
        """D3 multiplier on train-steps from the camera-speed EMA.

        Stalled (speed <= stall) -> ``velocity_step_gain`` (more refinement).
        Fast (speed >= fast) -> ``1/velocity_step_gain`` (fewer steps).
        Linearly interpolated in between. 1.0 when disabled / no sample.
        """
        cfg = self._config
        if not cfg.velocity_adaptive_steps or self._cam_speed_ema is None:
            return 1.0
        gain = max(1.0, float(cfg.velocity_step_gain))
        lo, hi = float(cfg.velocity_stall_mps), float(cfg.velocity_fast_mps)
        s = float(self._cam_speed_ema)
        if hi <= lo:
            return 1.0
        # frac: 0 at/below stall (boost), 1 at/above fast (cut).
        frac = min(1.0, max(0.0, (s - lo) / (hi - lo)))
        # Interpolate the multiplier in log space so boost/cut are symmetric.
        log_factor = (1.0 - frac) * math.log(gain) + frac * math.log(1.0 / gain)
        return float(math.exp(log_factor))

    def record_iter_dt(self, dt: Optional[float]) -> None:
        """Update the per-iteration EMA of wall-clock cost.

        Args:
            dt: Iteration cost in seconds, or None to leave the EMA alone.
        """
        if dt is None or dt <= 0.0 or not math.isfinite(dt):
            return
        beta = self._config.iter_dt_ema_beta
        self._iter_dt_ema = (1.0 - beta) * dt + beta * self._iter_dt_ema

    def tick(
        self,
        tracking_result: Optional[TrackingResult],
        timestamp: float,
        backlog: int,
        keyframe_indices: Optional[List[int]] = None,
        replay_indices: Optional[List[int]] = None,
        is_keyframe: bool = False,
        topology_version: int = 0,
    ) -> Tuple[OnlineDecision, MappingDecision]:
        """Produce an :class:`OnlineDecision` and base :class:`MappingDecision`.

        Args:
            tracking_result: SLAM result for the current frame, or None for
                precomputed poses.
            timestamp: Sensor timestamp of the current frame.
            backlog: Number of unprocessed non-redundant frames in the queue.
            keyframe_indices: Indices of keyframes for this iteration.
            replay_indices: Indices of replay frames.
            is_keyframe: Whether the current frame was admitted as a keyframe.
            topology_version: Current map topology version for assimilation control
                staleness detection (optional, defaults to 0 for no check).

        Returns:
            ``(online_decision, mapping_decision)`` — ``online_decision`` is
            the final per-frame decision, ``mapping_decision`` is the raw
            :class:`MappingScheduler` decision for diagnostics.

        Note:
            Iteration-cost EMA is fed via :meth:`record_iter_dt` exactly
            once per training iteration. ``tick`` does **not** consume an
            ``iter_dt`` argument; passing one will raise ``TypeError``.
        """
        # 1. update clocks + EMA
        self.record_ingest(timestamp)
        if tracking_result is not None:
            self._last_confidence = float(tracking_result.confidence)
            self.record_pose_arrival()

        # 2. update mode based on map-age hysteresis
        self._update_mode(backlog=backlog)

        # 3. base mapping decision (per-frame, phase 8 confidence gate)
        base = self._scheduler.decide(
            tracking_result=tracking_result,
            keyframe_indices=keyframe_indices or [],
            replay_indices=replay_indices or [],
            topology_version=topology_version,
        )

        # 4. mode-aware shaping
        shaped = self._shape(base, backlog=backlog, is_keyframe=is_keyframe)
        self._keyframe_indices = list(keyframe_indices or [])
        self._replay_indices = list(replay_indices or [])

        return shaped, base

    def update_keyframes(self, keyframe_indices: List[int], replay_indices: List[int]) -> None:
        """Publish the latest keyframe / replay indices for diagnostics."""
        self._keyframe_indices = list(keyframe_indices)
        self._replay_indices = list(replay_indices)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _update_map_age(self) -> None:
        self._map_age_s = max(0.0, self._sensor_clock - self._map_clock)

    def _update_mode(self, backlog: int) -> None:
        age = self._map_age_s
        cfg = self._config
        # 1. Hard ceiling: TRACK_ONLY (asymmetric hysteresis)
        if self._mode == ControllerMode.TRACK_ONLY:
            if age < cfg.map_age_track_only_exit_s:
                self._mode = ControllerMode.BALANCED
        else:
            if age > cfg.map_age_track_only_enter_s:
                self._mode = ControllerMode.TRACK_ONLY

        if self._mode == ControllerMode.TRACK_ONLY:
            return

        # 2. CATCH_UP (hysteresis)
        if self._mode == ControllerMode.CATCH_UP:
            if age < cfg.map_age_catch_up_exit_s:
                self._mode = ControllerMode.BALANCED
        else:
            if age > cfg.map_age_catch_up_enter_s:
                self._mode = ControllerMode.CATCH_UP

        # 3. QUALITY_IDLE entry (BALANCED only — never from CATCH_UP)
        if self._mode == ControllerMode.BALANCED:
            if (
                age < cfg.map_age_catch_up_exit_s
                and self._is_input_stable()
                and backlog <= 1
            ):
                self._mode = ControllerMode.QUALITY_IDLE
        elif self._mode == ControllerMode.QUALITY_IDLE:
            if age > cfg.map_age_catch_up_enter_s:
                self._mode = ControllerMode.CATCH_UP
            elif backlog > 1:
                self._mode = ControllerMode.BALANCED

        # 4. Pause / burst flags
        now = time.perf_counter()
        if self._first_frame_t is None:
            self._first_frame_t = now
        last_input_t = self._input_window[-1] if self._input_window else now
        gap = now - last_input_t
        prev_pause = self._pause
        # Configurable pause threshold: when set, gate on the operator-
        # supplied idle window; otherwise fall back to twice the target
        # period (legacy heuristic).
        if self._config.pause_idle_threshold_s > 0.0:
            self._pause = gap > self._config.pause_idle_threshold_s
        else:
            self._pause = gap > 1.0 / max(self._target_input_fps * 0.5, 0.1)
        if prev_pause != self._pause:
            self._last_pause_change_t = now
        measured_fps = self._fps(self._input_window, now)
        self._burst = measured_fps > self._target_input_fps * 1.5

    def _is_input_stable(self) -> bool:
        if not self._input_window:
            return False
        now = time.perf_counter()
        window_s = self._config.quality_idle_input_fps_stable
        # All frames in the last ``window_s`` seconds are within
        # ±``quality_idle_fps_tolerance`` of target.
        recent = [t for t in self._input_window if now - t <= window_s]
        if len(recent) < 2:
            return False
        span = recent[-1] - recent[0]
        if span <= 0.0:
            return False
        fps = (len(recent) - 1) / span
        return (
            abs(fps - self._target_input_fps) / self._target_input_fps
            <= self._config.quality_idle_fps_tolerance
        )

    def _effective_spf(self) -> int:
        """Return the base effective steps-per-frame from the scheduler."""
        return int(getattr(self._scheduler, "effective_spf", 1))

    def _clamp_steps(self, steps: int) -> int:
        return int(max(self._config.min_steps, min(self._config.max_steps, steps)))

    def _adaptive_steps(self, base_steps: int) -> int:
        """Scale steps by EMA iteration cost vs. budget."""
        if not self._config.adaptive_steps:
            return base_steps
        if self._iter_dt_ema <= 0.0:
            return base_steps
        # ratio = budget / ema → > 1 when we're under budget (cheap), < 1 when
        # we're over budget (expensive). Multiplying base_steps by ratio
        # therefore scales the per-frame work to fit inside the budget.
        ratio = self._budget_per_iter_s / self._iter_dt_ema
        scaled = int(max(1, round(base_steps * ratio)))
        return self._clamp_steps(scaled)

    def _shape(
        self,
        base: MappingDecision,
        backlog: int,
        is_keyframe: bool,
    ) -> OnlineDecision:
        mode = self._mode
        conf = base.confidence
        # Train-step budget per mode
        if mode == ControllerMode.TRACK_ONLY:
            steps = 0
            allow_insertion = False
            allow_growth = False
            freeze = True
            sampling_mode = "legacy"
            insertion_cap = 0
        elif mode == ControllerMode.CATCH_UP:
            # Reduced steps, no growth, low insertion cap, keyframes only.
            steps = max(1, self._clamp_steps(self._effective_spf() // 2))
            allow_insertion = base.allow_insertion
            allow_growth = False
            freeze = base.freeze_persistent
            sampling_mode = "legacy"
            insertion_cap = max(50, self._insertion_cap // 4)
        elif mode == ControllerMode.QUALITY_IDLE:
            # More steps, growth allowed, more replay.
            steps = self._clamp_steps(self._effective_spf() + 1)
            allow_insertion = base.allow_insertion
            allow_growth = base.allow_growth
            freeze = False
            sampling_mode = "stratified"
            insertion_cap = self._insertion_cap
        else:  # BALANCED
            steps = self._clamp_steps(self._effective_spf())
            allow_insertion = base.allow_insertion
            allow_growth = base.allow_growth
            freeze = base.freeze_persistent
            sampling_mode = self._sampling_mode
            insertion_cap = self._insertion_cap

        # Adaptive scale by measured iteration cost (overrides mode's steps).
        # Skip when the mode mandates zero steps (TRACK_ONLY) — the EMA must
        # never inflate a forced-zero budget.
        if steps > 0:
            steps = self._adaptive_steps(steps)
            # D3: velocity-adaptive refinement on top of the cost-adaptive budget.
            vfac = self._velocity_step_factor()
            if vfac != 1.0:
                steps = self._clamp_steps(int(max(1, round(steps * vfac))))

        # TRACK_ONLY + low confidence → never insert / never step
        do_mapping = base.do_mapping and mode != ControllerMode.TRACK_ONLY and steps > 0

        return OnlineDecision(
            mode=mode,
            do_mapping=do_mapping,
            steps=steps if do_mapping else 0,
            allow_insertion=allow_insertion and do_mapping,
            allow_growth=allow_growth and do_mapping,
            freeze_persistent=freeze or mode == ControllerMode.TRACK_ONLY,
            accept_frame=True,
            train_steps=steps,
            insertion_cap=insertion_cap,
            sampling_mode=sampling_mode,
            publish_rviz=True,
            confidence=conf,
        )

    # Hard cap on the rolling windows. The 1-second FPS calculation
    # only needs ~1 s of data; the 5× margin covers the 5-s stability
    # window plus bursts.
    _WINDOW_MAX_AGE_S: float = 5.0

    def _prune_windows(self) -> None:
        """Drop stale timestamps from ``_input_window`` and ``_pose_window``.

        The windows would otherwise grow without bound in long-running
        streaming sessions. We keep at most ``_WINDOW_MAX_AGE_S`` seconds
        of history in-place.
        """
        cutoff = time.perf_counter() - self._WINDOW_MAX_AGE_S
        # List slicing is in-place when assigned back to the same variable.
        self._input_window = [t for t in self._input_window if t >= cutoff]
        self._pose_window = [t for t in self._pose_window if t >= cutoff]

    @staticmethod
    def _fps(window: List[float], now: float, span: float = 1.0) -> float:
        if len(window) < 2:
            return 0.0
        recent = [t for t in window if now - t <= span]
        if len(recent) < 2:
            return 0.0
        span_s = recent[-1] - recent[0]
        if span_s <= 0.0:
            return 0.0
        return (len(recent) - 1) / span_s
