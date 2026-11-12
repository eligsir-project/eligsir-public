"""Resolution-tier controller (multires-cache plan, task 3).

Supports an explicit low-to-high bootstrap boundary. Without one, computes
ROLLING (not lifetime-average) versions of iterations-per-
admitted-frame and map-extension-fps from raw admission events, and
decides which cached resolution tier ("low"/"high") training should read
from. See docs/superpowers/specs/2026-07-25-multires-frame-cache-design.md
section 3 -- the existing it/V and mapFPS progress-bar variables
(train_streaming.py) are lifetime cumulative averages, unsuitable for a
reactive controller (an early fast burst masks a real later slowdown for
the rest of the run), so this class maintains its own bounded window of
recent admission events instead of reading those variables.
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class ResolutionController:
    """Resolution tier selector with reactive and explicit schedule modes.

    Monitors frame admission throughput and map extension rate over a recent
    sliding window, switching between "high" and "low" resolution tiers based
    on whether throughput falls below configured floors. Applies hysteresis
    (recovery_streak) to avoid jittery tier-flipping on transient slowdowns.
    """
    def __init__(
        self,
        it_v_floor: float = 30.0,
        map_fps_floor: float = 3.0,
        window: int = 20,
        recovery_streak: int = 5,
        scheduled_low_until_iteration: int = 0,
        lock_to_high_on_mapping_done: bool = True,
    ) -> None:
        """Initialize the resolution controller.

        Args:
            it_v_floor: Minimum acceptable iterations per admitted frame.
                       If rolling average drops below this, switch to "low" tier.
            map_fps_floor: Minimum acceptable map-extension frames per second
                          (frames ingested per wall-clock second). If rolling
                          average drops below this, switch to "low" tier.
            window: Number of recent admission events to track in the rolling
                   window. Larger windows dampen reactivity; smaller windows
                   respond faster to changes.
            recovery_streak: Number of consecutive "good" (above-threshold)
                            measurements needed to recover from "low" to "high" tier.
                            Applies hysteresis to prevent flipping on noise.
            scheduled_low_until_iteration: Use the low tier through this
                          iteration, then use high. Zero keeps the reactive
                          dual-floor policy.
            lock_to_high_on_mapping_done: Historical/default behavior --
                          once ``mapping_done`` is observed, permanently
                          latch to "high" (native) resolution regardless of
                          throughput, on the theory that refinement is a
                          precision phase with no more admission-throughput
                          pressure to protect. Issue #1259 found this is a
                          real ~2.9x views/s cliff (measured on fr3: ~76
                          views/s at "low"/quarter tier during mapping vs
                          ~26 views/s once locked to "high"/native at
                          refinement start) that eats most of a short, fixed
                          wall-clock refinement budget's supervised-view
                          count. Passing False here disables the one-way
                          latch: `decide()` keeps evaluating the same
                          reactive floor logic used during mapping even
                          after `mapping_done`. Since admission events stop
                          arriving once mapping ends, the rolling window is
                          simply frozen at its last pre-mapping-done state,
                          so in practice this carries the tier mapping was
                          actually running at (usually "low"/quarter on a
                          throughput-constrained scene) forward through the
                          whole refinement window instead of forcing "high".
                          Do NOT set this False for an "exact refinement"
                          run (`--streaming_post_mapping_refinement_steps >
                          0`) -- that path asserts `decide(mapping_done=
                          True) == "high"` and raises RuntimeError
                          otherwise; train_streaming.py's arg validation
                          refuses that combination before training starts.
        """
        self._it_v_floor = it_v_floor
        self._map_fps_floor = map_fps_floor
        self._window = window
        self._recovery_streak = recovery_streak
        self._scheduled_low_until_iteration = int(scheduled_low_until_iteration)
        self._lock_to_high_on_mapping_done = bool(lock_to_high_on_mapping_done)
        self._events: deque = deque(maxlen=window)  # (iteration, wall_time)
        self._tier = "high"
        self._locked = False
        self._good_streak = 0
        self._last_iteration: Optional[int] = None
        self._last_wall_time: Optional[float] = None

    def record_admission(self, iteration: int, wall_time: float) -> None:
        """Record an admission event.

        Adds an (iteration, wall_time) pair to the rolling window.
        Once the window is full, the oldest event is automatically evicted.

        Args:
            iteration: Training iteration at which the frame was admitted.
            wall_time: Elapsed wall-clock time at the admission (seconds).
        """
        self._events.append((iteration, wall_time))
        self._last_iteration = iteration
        self._last_wall_time = wall_time

    def _rolling_it_v(self) -> Optional[float]:
        """Compute rolling iterations per admitted frame over the window.

        Returns:
            Iterations per frame (iteration delta / frame count in window),
            or None if fewer than 2 events are in the window.
        """
        if len(self._events) < 2:
            return None
        (i0, _), (i1, _) = self._events[0], self._events[-1]
        n = len(self._events) - 1
        if n <= 0:
            return None
        return (i1 - i0) / n

    def _rolling_map_fps(self) -> Optional[float]:
        """Compute rolling frames-per-second (admission rate) over the window.

        Returns:
            Frames per wall-clock second (frame count in window / elapsed time),
            or None if fewer than 2 events are in the window or if no time
            has elapsed.
        """
        if len(self._events) < 2:
            return None
        (_, t0), (_, t1) = self._events[0], self._events[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return None
        return (len(self._events) - 1) / elapsed

    def decide(self, mapping_done: bool, iteration: Optional[int] = None) -> str:
        """Decide which resolution tier to use.

        Once mapping is complete, locks to "high" tier permanently -- unless
        constructed with ``lock_to_high_on_mapping_done=False``, in which
        case this falls through to the same reactive floor logic used
        during mapping (see that constructor arg's docstring). An explicit
        schedule takes precedence when configured. Otherwise, switches based
        on whether rolling throughput metrics are above configured floors,
        with hysteresis to prevent jitter.

        Args:
            mapping_done: True if frame ingestion has completed and the
                         mapping phase is over.
            iteration: Current training iteration. Required only when an
                       explicit low-tier schedule is configured.

        Returns:
            "high" or "low", indicating which resolution tier training
            should read from on the next iteration.
        """
        if mapping_done and self._lock_to_high_on_mapping_done:
            self._locked = True
        if self._locked:
            return "high"

        if self._scheduled_low_until_iteration > 0:
            if iteration is None:
                raise ValueError("iteration is required for scheduled multires training")
            self._tier = (
                "low"
                if int(iteration) <= self._scheduled_low_until_iteration
                else "high"
            )
            return self._tier

        it_v = self._rolling_it_v()
        map_fps = self._rolling_map_fps()
        # Not enough data yet to judge -- stay at the current tier rather
        # than guessing (avoids flipping on the very first few admissions).
        if it_v is None or map_fps is None:
            return self._tier

        floor_breached = it_v < self._it_v_floor or map_fps < self._map_fps_floor
        if floor_breached:
            self._tier = "low"
            self._good_streak = 0
        else:
            self._good_streak += 1
            if self._good_streak >= self._recovery_streak:
                self._tier = "high"
        return self._tier
