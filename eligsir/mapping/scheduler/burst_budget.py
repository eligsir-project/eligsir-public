"""BurstBudget — ingestion cadence + backlog-driven drop policy (Milestone 3).

Caps the number of frames that may be ingested in a single training
iteration and decides whether a non-keyframe should be dropped (and
counted as a drop) when the controller is in CATCH_UP mode or the
backlog exceeds a soft threshold.

Two thresholds:

  * ``max_per_iter`` (default 2) — hard cap on the number of frames
    the drain loop is allowed to ingest in a single training iter.
  * ``drop_nonkf_backlog`` (default 4) — when the controller reports
    a backlog at or above this value, non-keyframes are dropped
    (keyframes are protected). The decision is recorded on the
    :class:`OnlineController` so it shows up in scheduler_stats.

Drop policy: the budget is a *per-iteration* quota. Each call to
:meth:`acquire` returns a tuple ``(ingest, is_keyframe)``:

  * ``ingest`` — True if the frame should be ingested.
  * ``is_keyframe`` — whether the frame is a keyframe (admission
    heuristic).

The function is cheap (no rendering) so it can sit in the drain loop
in front of the expensive :meth:`StreamingScene.ingest_next_frame`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .online_controller import ControllerMode, OnlineController


@dataclass
class BurstConfig:
    """Tunables for :class:`BurstBudget`.

    The keyframe decision itself (pose-delta + gap heuristic) lives in
    :func:`utils.train_streaming._cheap_keyframe_estimate` so that the
    BurstBudget does not duplicate policy. The thresholds are passed in
    via constructor arguments on the helper, not on this config.

    Attributes:
        max_per_iter: Hard cap on frames ingested per training iter.
        drop_nonkf_backlog: Drop non-keyframes when backlog is at or
            above this value. Keyframes are protected.
        drop_nonkf_in_catch_up: Drop non-keyframes in CATCH_UP mode
            regardless of backlog (default True).
    """

    max_per_iter: int = 2
    drop_nonkf_backlog: int = 4
    drop_nonkf_in_catch_up: bool = True


class BurstBudget:
    """Per-iteration frame cap and drop policy."""

    def __init__(
        self,
        controller: OnlineController,
        config: Optional[BurstConfig] = None,
    ) -> None:
        self._controller = controller
        self._config = config or BurstConfig()
        self._this_iter_count: int = 0
        self._this_iter_iter: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def begin_iter(self, iteration: int) -> None:
        """Reset the per-iter counter at the start of a new iter."""
        if iteration != self._this_iter_iter:
            self._this_iter_count = 0
            self._this_iter_iter = iteration

    def acquire(
        self,
        frame,
        is_keyframe: bool,
    ) -> Tuple[bool, bool]:
        """Decide whether to ingest *frame* on the current iter.

        Args:
            frame: StreamingRGBDFrame-like object exposing
                ``.c2w`` and ``.index``.
            is_keyframe: Whether the frame was admitted as a keyframe
                by the upstream admission function.

        Returns:
            ``(ingest, is_keyframe)`` — ``ingest`` is True if the
            drain loop should call :meth:`ingest_next_frame`. The
            second value is the keyframe flag echoed back so callers
            can keep their bookkeeping in sync (the drop policy in
            here already records drops against the controller).
        """
        cfg = self._config
        # 1. Hard per-iter cap
        if self._this_iter_count >= cfg.max_per_iter:
            # Count the drop on the controller
            self._controller.record_drop(
                timestamp=float(getattr(frame, "timestamp", 0.0)),
                is_keyframe=is_keyframe,
            )
            return False, is_keyframe

        # 2. Drop non-keyframes when backlog is high or in CATCH_UP
        mode = self._controller.mode
        backlog = self._controller.stats.backlog
        drop_due_to_backlog = backlog >= cfg.drop_nonkf_backlog
        drop_due_to_mode = cfg.drop_nonkf_in_catch_up and mode == ControllerMode.CATCH_UP
        if not is_keyframe and (drop_due_to_backlog or drop_due_to_mode):
            self._controller.record_drop(
                timestamp=float(getattr(frame, "timestamp", 0.0)),
                is_keyframe=False,
            )
            return False, False

        self._this_iter_count += 1
        return True, is_keyframe

    @property
    def this_iter_count(self) -> int:
        """Number of frames the budget has admitted in the current iter."""
        return self._this_iter_count
