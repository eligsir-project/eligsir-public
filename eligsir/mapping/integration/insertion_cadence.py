"""Pure decision function for when pending depth-insertion candidates get
flushed (committed) to the Gaussian map.

Evidence-driven-mutation-cadence experiment plan, Task 1. Adds a
step-based (optimizer-iteration-count) alternative to the existing
frame-count-based flush trigger in ``train_streaming.py``'s Phase 4.1
insertion-batch buffer (``_insertion_batch`` / ``_insertion_batch_frames``
/ ``_batch_frames`` / ``_batch_max_pts``, see that file's batch-state init
block and the accumulate/flush call site it feeds).

Deliberately dependency-free (no torch/GPU/model-layout imports) so it is
directly unit-testable with plain pytest, no Docker container required.

Zero-behavior-change contract: with ``batch_steps=0`` and
``disable_point_cap=False`` (the defaults for the new CLI flags this
function backs -- ``--streaming_insertion_batch_steps`` and
``--streaming_insertion_batch_disable_point_cap``), this function's
result is byte-identical to the original inline expression it replaces:

    _early_by_points = _batch_max_pts > 0 and _pending_pts >= _batch_max_pts
    should_flush = _insertion_batch_frames >= _batch_frames or _early_by_points
"""

from __future__ import annotations


def _should_flush_insertion_batch(
    iteration: int,
    last_flush_iteration: int,
    batch_steps: int,
    frames_since_flush: int,
    batch_frames: int,
    pending_pts: int,
    batch_max_pts: int,
    disable_point_cap: bool,
) -> tuple[bool, str]:
    """Decide whether the pending insertion batch should be flushed now.

    Returns ``(should_flush, trigger_reason)`` where ``trigger_reason`` is
    one of ``"frame_count"``, ``"step_count"``, or ``"point_cap"`` --
    describing which condition fired. If ``batch_steps > 0``, step-count
    triggering takes precedence over frame-count triggering (they are
    mutually exclusive alternatives, not combined -- frame-count is not
    evaluated at all in that mode). If ``disable_point_cap`` is True, the
    point-cap condition never fires regardless of ``pending_pts``/
    ``batch_max_pts``. If nothing fires, returns ``(False, "")``.
    """
    early_by_points = (
        not disable_point_cap
        and batch_max_pts > 0
        and pending_pts >= batch_max_pts
    )

    if batch_steps > 0:
        by_steps = (iteration - last_flush_iteration) >= batch_steps
        if by_steps:
            return True, "step_count"
        if early_by_points:
            return True, "point_cap"
        return False, ""

    by_frames = frames_since_flush >= batch_frames
    if by_frames:
        return True, "frame_count"
    if early_by_points:
        return True, "point_cap"
    return False, ""


def _mcmc_suppressed_by_insertion_offset(
    iteration: int,
    last_flush_iteration: int,
    offset: int,
) -> bool:
    """Evidence-driven-mutation-cadence experiment, Task 5 prep (C4 arm).

    Decide whether MCMC growth/relocation (``step_post_backward``) should be
    suppressed this iteration because it falls within ``offset`` iterations
    of the most recent insertion-batch commit -- staggers the two mutation
    sources apart in time instead of letting them collide on the same
    iteration. ``offset <= 0`` (the default) never suppresses, matching
    today's unconditional-every-non-skip-iteration behavior exactly.
    """
    if offset <= 0:
        return False
    return (iteration - last_flush_iteration) < offset
