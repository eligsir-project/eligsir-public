"""Regression tests for issue #1384.

``refinement_progress_fraction`` used to hardcode ``1.0`` whenever
``requested_steps`` (the step-count refinement budget) was ``0`` -- which is
exactly the standard time-cap-only refinement mode this project's Run
Contracts mandate (``--streaming_post_mapping_refinement_seconds`` without an
explicit ``--streaming_post_mapping_refinement_steps``). Since ``1.0`` is
always ``>= switch_frac``, the ``progressive`` refinement resolution policy
(PR #1358/#1375) silently and permanently degraded to plain ``native`` for
the entire refinement window under that mode -- confirmed on a real fr3 GPU
run (progressive ran 100% native tier, 2726/2726 steps).

These tests cover ``_compute_refinement_progress_fraction`` directly (all
three requested/requested_seconds cases) plus the switch behavior it feeds
into via ``WorkBudgetOwner.forces_native_post_refinement``.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import WorkBudgetOwner
from eligsir.runtime.causal_entry import _compute_refinement_progress_fraction


# ---------------------------------------------------------------------------
# _compute_refinement_progress_fraction
# ---------------------------------------------------------------------------


def test_step_governed_mode_is_unchanged():
    """requested_steps > 0: pure step-count fraction, exactly as before."""

    assert _compute_refinement_progress_fraction(
        completed=0, requested=100, elapsed_s=0.0, requested_seconds=0.0
    ) == 0.0
    assert _compute_refinement_progress_fraction(
        completed=50, requested=100, elapsed_s=999.0, requested_seconds=10.0
    ) == 0.5
    assert _compute_refinement_progress_fraction(
        completed=100, requested=100, elapsed_s=0.0, requested_seconds=0.0
    ) == 1.0


def test_step_governed_mode_ignores_elapsed_time_entirely():
    """Even a huge elapsed time must not perturb the step-count fraction.

    Step-governed mode's fraction must stay reproducible for a fixed step
    budget regardless of how fast/slow this host's GPU realizes steps.
    """

    assert _compute_refinement_progress_fraction(
        completed=10, requested=40, elapsed_s=1_000_000.0, requested_seconds=1.0
    ) == 0.25


def test_time_cap_only_mode_uses_elapsed_over_requested_seconds():
    """The #1384 bug case: requested_steps == 0, requested_seconds > 0."""

    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=0.0, requested_seconds=120.0
    ) == 0.0
    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=60.0, requested_seconds=120.0
    ) == pytest.approx(0.5)
    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=119.999, requested_seconds=120.0
    ) < 1.0


def test_time_cap_only_mode_clamps_to_one():
    """Elapsed time can exceed the cap by the time this is evaluated -- clamp."""

    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=500.0, requested_seconds=120.0
    ) == 1.0


def test_no_cap_at_all_returns_zero_never_switch():
    """requested == 0 and requested_seconds == 0: no budget signal exists.

    Unreachable from the real refinement loop (its own ``while`` condition
    never executes when both budgets are zero), but a direct caller gets the
    documented, deliberate "never switch yet" default rather than the old
    "always switched" (1.0) default.
    """

    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=0.0, requested_seconds=0.0
    ) == 0.0
    assert _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=1234.0, requested_seconds=0.0
    ) == 0.0


# ---------------------------------------------------------------------------
# forces_native_post_refinement, driven by a time-cap-only fraction sequence
# ---------------------------------------------------------------------------


def test_progressive_does_not_force_native_early_in_a_time_cap_only_window():
    owner = WorkBudgetOwner(refinement_resolution_policy="progressive")
    assert owner.refinement_progressive_switch_frac == 0.5

    early_fraction = _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=1.0, requested_seconds=120.0
    )
    assert early_fraction < 0.5
    assert owner.forces_native_post_refinement(early_fraction) is False


def test_progressive_forces_native_once_past_switch_frac_of_elapsed_time():
    owner = WorkBudgetOwner(refinement_resolution_policy="progressive")

    late_fraction = _compute_refinement_progress_fraction(
        completed=0, requested=0, elapsed_s=61.0, requested_seconds=120.0
    )
    assert late_fraction >= 0.5
    assert owner.forces_native_post_refinement(late_fraction) is True


def test_progressive_switch_point_tracks_elapsed_time_not_a_hardcoded_constant():
    """The whole point of the fix: the fraction must genuinely vary over time
    under time-cap-only mode, not sit hardcoded at 1.0 from tick 1.
    """

    owner = WorkBudgetOwner(refinement_resolution_policy="progressive")
    requested_seconds = 120.0

    fractions = [
        _compute_refinement_progress_fraction(
            completed=0, requested=0, elapsed_s=elapsed, requested_seconds=requested_seconds
        )
        for elapsed in (0.0, 30.0, 59.0, 60.0, 90.0, 120.0)
    ]

    # Strictly increasing (not stuck at a constant like the old bug).
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert len(set(fractions)) > 1

    forces_native_sequence = [owner.forces_native_post_refinement(f) for f in fractions]
    # Must genuinely spend time at the cheap tier before switching -- not
    # forced to native from the very first sample.
    assert forces_native_sequence[0] is False
    assert forces_native_sequence[-1] is True
    # Monotonic: once it switches to native it must not switch back.
    assert forces_native_sequence == sorted(forces_native_sequence)
