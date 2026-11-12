"""Unit tests for the pure insertion-batch flush-trigger decision function.

Task 1 of the evidence-driven-mutation-cadence experiment plan: adds a
step-based (optimizer-iteration-count) alternative to the existing
frame-count-based flush trigger for pending depth-insertion candidates.

No torch/GPU dependency -- runs with plain pytest, no Docker required.
"""

from eligsir.mapping.integration.insertion_cadence import _should_flush_insertion_batch


class TestFrameOnlyBehavior:
    """batch_steps=0 must reproduce today's frame-only OR-point-cap logic exactly."""

    def test_no_flush_when_below_frame_count_and_below_point_cap(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=2, batch_frames=4,
            pending_pts=100, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is False
        assert reason == ""

    def test_flush_on_frame_count_reached(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=4, batch_frames=4,
            pending_pts=100, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "frame_count"

    def test_flush_on_frame_count_exceeded(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=5, batch_frames=4,
            pending_pts=100, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "frame_count"

    def test_flush_on_point_cap_reached(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=1, batch_frames=4,
            pending_pts=8000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "point_cap"

    def test_flush_on_point_cap_exceeded(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=1, batch_frames=4,
            pending_pts=9000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "point_cap"

    def test_frame_count_takes_precedence_reason_when_both_fire(self):
        # Both conditions fire simultaneously -- frame_count is checked first
        # in the original inline expression order (`_insertion_batch_frames
        # >= _batch_frames or _early_by_points`), so frame_count should win.
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=4, batch_frames=4,
            pending_pts=9000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "frame_count"

    def test_point_cap_disabled_when_batch_max_pts_zero(self):
        # Existing behavior: _batch_max_pts <= 0 disables the point-cap trigger.
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=1, batch_frames=4,
            pending_pts=100000, batch_max_pts=0, disable_point_cap=False,
        )
        assert should_flush is False
        assert reason == ""


class TestStepBasedTrigger:
    """batch_steps > 0: step-count triggering takes precedence over frame-count."""

    def test_no_flush_before_step_threshold(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=49, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=100, batch_frames=1,
            pending_pts=0, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is False
        assert reason == ""

    def test_flush_exactly_at_step_threshold(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=50, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=0, batch_frames=999999,
            pending_pts=0, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "step_count"

    def test_flush_beyond_step_threshold(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=200, last_flush_iteration=100, batch_steps=50,
            frames_since_flush=0, batch_frames=999999,
            pending_pts=0, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "step_count"

    def test_step_trigger_ignores_frame_count_when_frame_count_would_not_fire(self):
        # frames_since_flush way below batch_frames -- frame condition alone
        # would not fire, but step-count does, and takes precedence.
        should_flush, reason = _should_flush_insertion_batch(
            iteration=50, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=1, batch_frames=1000,
            pending_pts=0, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "step_count"

    def test_step_trigger_ignores_frame_count_when_frame_count_would_fire(self):
        # frames_since_flush already exceeds batch_frames, but step-count
        # hasn't been reached yet -- step-based mode replaces frame-based
        # entirely (mutually exclusive alternatives, not combined), so no
        # flush should occur.
        should_flush, reason = _should_flush_insertion_batch(
            iteration=10, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=1000, batch_frames=1,
            pending_pts=0, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is False
        assert reason == ""

    def test_point_cap_still_fires_alongside_step_mode(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=10, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=1000, batch_frames=1,
            pending_pts=9000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "point_cap"

    def test_step_count_takes_precedence_over_point_cap_reason_when_both_fire(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=50, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=0, batch_frames=999999,
            pending_pts=9000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "step_count"


class TestPointCapDisable:
    """streaming_insertion_batch_disable_point_cap must fully suppress the
    point-cap trigger regardless of how large pending_pts is."""

    def test_large_pending_pts_does_not_flush_when_disabled_frame_mode(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=1, batch_frames=4,
            pending_pts=100000, batch_max_pts=8000, disable_point_cap=True,
        )
        assert should_flush is False
        assert reason == ""

    def test_large_pending_pts_does_not_flush_when_disabled_step_mode(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=10, last_flush_iteration=0, batch_steps=50,
            frames_since_flush=0, batch_frames=1,
            pending_pts=100000, batch_max_pts=8000, disable_point_cap=True,
        )
        assert should_flush is False
        assert reason == ""

    def test_frame_count_still_fires_when_point_cap_disabled(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=4, batch_frames=4,
            pending_pts=100000, batch_max_pts=8000, disable_point_cap=True,
        )
        assert should_flush is True
        assert reason == "frame_count"

    def test_default_disable_point_cap_false_preserves_point_cap(self):
        should_flush, reason = _should_flush_insertion_batch(
            iteration=100, last_flush_iteration=0, batch_steps=0,
            frames_since_flush=1, batch_frames=4,
            pending_pts=100000, batch_max_pts=8000, disable_point_cap=False,
        )
        assert should_flush is True
        assert reason == "point_cap"
