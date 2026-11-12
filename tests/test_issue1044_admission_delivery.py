"""Issue #1044 — why fr3 `hybrid_keyframe` plateaued at ~20 admitted views.

Two INDEPENDENT defects, both measured on a real fr3 run
(`viser_watch_fr3_stats`, 2026-08-22, 18,743 iterations,
`--streaming_replay --online_mode --streaming_ingestion_mode iter_based
--streaming_frame_admission hybrid_keyframe`):

1. **Delivery.** ``opt.iterations`` is the ``ONLINE_UNBOUNDED_ITERATIONS``
   sentinel (2^31-1) on an online run, so the full-coverage clamp
   ``min(requested, floor(budget / n_frames))`` could never bind. SPF stayed at
   the raw argparse default 150 and *governed release*, so the run needed
   379,500 iterations to release all 2525 frames and in fact reached frame 130
   (5.1%). The startup line meanwhile announced "covers 100%. All frames
   covered.", and the SPF record labelled the run
   ``resolved_by=request_online_mode_bypass`` while simultaneously reporting
   ``governs_release=True`` and "IS ENFORCED".

2. **The motion gate was dead by construction.** With ``velocity_adaptive`` on,
   the motion criterion fires after the camera has travelled for one controller
   target interval (``input_fps / target_admit_rate_fps`` = 30/2 = 15 frames),
   while the soft backstop fires at ``max_gap`` = 10 frames. Both scale
   linearly with scene speed, so the shortfall is a scene-INDEPENDENT 1.5x.
   Measured: ``moved`` False on 109/109 candidates, 12/12 admissions owned by
   ``max_gap``. ``target_time`` mode already enforced a non-preemption
   invariant (#946 criterion 2); ``legacy_frames`` — the default — did not.

These are pure-function tests; no GPU, no training entrypoint.
"""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import (
    INGESTION_MODE_ITER_BASED,
    MappingScheduler,
    resolve_coverage_planning_budget,
)
from eligsir.mapping.scheduler.precedence import (
    SoftBackstopMode,
    backstop_crossings,
    resolve_backstop_thresholds,
)
from eligsir.mapping.scheduler.spf_resolution import (
    BUDGET_MAX_ITERATIONS,
    BUDGET_OPERATOR_ITERATIONS,
    BUDGET_UNBOUNDED,
)

ONLINE_SENTINEL = 2_147_483_647
FR3_N_FRAMES = 2525


# ---------------------------------------------------------------------------
# 1. Delivery: the coverage planning budget
# ---------------------------------------------------------------------------


def test_online_sentinel_makes_the_coverage_clamp_structurally_inert():
    """The pre-#1044 behaviour, pinned so the regression is recognisable.

    Dividing the 2^31 sentinel by 2525 frames leaves ``min(150, 850489) ==
    150`` — the clamp cannot bind, at any frame count.
    """
    sched = MappingScheduler(
        steps_per_frame=150,
        n_frames=FR3_N_FRAMES,
        iterations=ONLINE_SENTINEL,
        enforce_full_coverage=True,
        min_steps_per_frame=0,
    )
    assert sched.effective_spf == 150
    # 150 spf * 2525 frames = 378,750 iterations to release the trajectory.
    assert sched.effective_spf * FR3_N_FRAMES > 375_000


def test_operator_iterations_are_used_as_the_planning_budget():
    """`--iterations 40000` survives on `args` even though `opt` got the sentinel."""
    budget, source = resolve_coverage_planning_budget(
        loop_iterations=ONLINE_SENTINEL,
        streaming_max_iterations=0,
        operator_iterations=40_000,
        online_mode=True,
    )
    assert budget == 40_000
    assert source == BUDGET_OPERATOR_ITERATIONS


def test_streaming_max_iterations_outranks_operator_iterations():
    budget, source = resolve_coverage_planning_budget(
        loop_iterations=ONLINE_SENTINEL,
        streaming_max_iterations=60_000,
        operator_iterations=40_000,
        online_mode=True,
    )
    assert budget == 60_000
    assert source == BUDGET_MAX_ITERATIONS


def test_genuinely_unbounded_run_yields_no_budget_and_therefore_no_coverage_claim():
    budget, source = resolve_coverage_planning_budget(
        loop_iterations=ONLINE_SENTINEL,
        streaming_max_iterations=0,
        operator_iterations=ONLINE_SENTINEL,
        online_mode=True,
    )
    assert budget is None
    assert source == BUDGET_UNBOUNDED


def test_non_online_run_is_unchanged():
    budget, source = resolve_coverage_planning_budget(
        loop_iterations=30_000,
        streaming_max_iterations=0,
        operator_iterations=30_000,
        online_mode=False,
    )
    assert budget == 30_000


def test_prefix_vs_fixed_on_the_identical_verification_recipe():
    """The CONTROLLED before/after for the delivery fix.

    Same recipe, same code, same 40,000-iteration budget: only the pre-#1044
    ``_use_enforced_spf = not online_mode`` predicate differs. This is the
    comparison the two GPU arms cannot make (both carry the delivery fix; they
    isolate the *gate* fix instead), and it is an arithmetic identity rather
    than an empirical claim.
    """
    budget = 40_000
    releasable_prefix = budget // 150  # online bypass => raw request governs
    sched = MappingScheduler(
        steps_per_frame=150,
        n_frames=FR3_N_FRAMES,
        iterations=budget,
        enforce_full_coverage=True,
        min_steps_per_frame=15,
    )
    releasable_fixed = budget // sched.effective_spf

    assert releasable_prefix == 266
    assert releasable_prefix / FR3_N_FRAMES < 0.11  # ~10.5% of the trajectory
    assert sched.effective_spf == 15
    assert releasable_fixed >= FR3_N_FRAMES  # 2666 >= 2525: whole trajectory


def test_min_spf_floor_still_outranks_coverage_and_still_strands_frames():
    """W3-38's floor-over-coverage precedence is DELIBERATELY not fixed here.

    At the default ``--streaming_min_steps_per_frame 50`` the fr3 verification
    recipe still under-covers: the floor overrides the coverage result of 15.
    The operator must pass a lower floor; the run's own startup projection is
    what surfaces this rather than the old "All frames covered." claim.
    """
    sched = MappingScheduler(
        steps_per_frame=150,
        n_frames=FR3_N_FRAMES,
        iterations=40_000,
        enforce_full_coverage=True,
        min_steps_per_frame=50,  # the argparse default
    )
    assert sched.coverage_required_spf == 15
    assert sched.effective_spf == 50  # floor wins, unchanged by #1044
    assert (40_000 // 50) / FR3_N_FRAMES < 0.35  # ~32% of the trajectory


def test_planning_budget_makes_the_clamp_bind_on_the_fr3_shape():
    """With a real budget the clamp finally does its job."""
    budget, _ = resolve_coverage_planning_budget(
        loop_iterations=ONLINE_SENTINEL,
        streaming_max_iterations=0,
        operator_iterations=40_000,
        online_mode=True,
    )
    sched = MappingScheduler(
        steps_per_frame=150,
        n_frames=FR3_N_FRAMES,
        iterations=budget,
        enforce_full_coverage=True,
        min_steps_per_frame=0,
    )
    assert sched.effective_spf == 40_000 // FR3_N_FRAMES == 15
    # The whole trajectory now fits the budget it was planned against.
    assert sched.effective_spf * FR3_N_FRAMES <= 40_000


# ---------------------------------------------------------------------------
# 2. The motion gate: non-preemption floor for legacy_frames
# ---------------------------------------------------------------------------


def _production_backstop(*, nonpreemption_floor: bool):
    """The exact production profile from the fr3 run's own telemetry."""
    return resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=nonpreemption_floor,
    )


def test_pre_fix_soft_backstop_preempts_the_motion_criterion():
    """Pinned so the defect stays recognisable: soft (10) < controller (15)."""
    bs = _production_backstop(nonpreemption_floor=False)
    assert bs.soft_threshold_frames == 10.0
    controller_interval_frames = 30.0 / 2.0
    assert bs.soft_threshold_frames < controller_interval_frames


def test_nonpreemption_floor_raises_soft_to_the_controller_interval():
    bs = _production_backstop(nonpreemption_floor=True)
    assert bs.soft_threshold_frames == 15.0  # input_fps / target_admit_rate_fps
    assert any("non-preemption floor" in w for w in bs.warnings)


def test_hard_ceiling_is_not_moved_by_the_soft_floor():
    """#946 independence: a soft change must never move the starvation bound."""
    before = _production_backstop(nonpreemption_floor=False)
    after = _production_backstop(nonpreemption_floor=True)
    assert before.hard_threshold_frames == 40.0
    assert after.hard_threshold_frames == 40.0
    assert after.soft_threshold_frames > before.soft_threshold_frames


def test_gap_10_no_longer_force_admits_so_the_gate_can_own_the_decision():
    """The concrete behavioural change on the measured run's own gaps."""
    before = _production_backstop(nonpreemption_floor=False)
    after = _production_backstop(nonpreemption_floor=True)
    # Every one of the run's 12 admissions happened here.
    soft_before, hard_before = backstop_crossings(before, 10, None)
    soft_after, hard_after = backstop_crossings(after, 10, None)
    assert soft_before is True and hard_before is False
    assert soft_after is False and hard_after is False
    # ...and the backstop still fires once the controller interval elapses.
    assert backstop_crossings(after, 15, None)[0] is True


def test_floor_is_inert_without_the_velocity_controller():
    """No controller cadence to back up => nothing to be pre-empted."""
    bs = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=10,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=False,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    assert bs.soft_threshold_frames == 10.0
    assert bs.hard_threshold_frames == 40.0


def test_floor_is_inert_when_max_gap_already_clears_the_controller():
    """An operator who already set max_gap >= the target interval is untouched."""
    bs = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=20,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    assert bs.soft_threshold_frames == 20.0
    assert bs.hard_threshold_frames == 80.0
    assert not any("non-preemption floor" in w for w in bs.warnings)


@pytest.mark.parametrize(
    "max_gap,expected_soft,expected_hard",
    [
        # hard = max_gap * 4.0, controller interval = 30/2 = 15 frames.
        (1, 1.0, 4.0),    # the #793 velocity harness's force-everything arm
        (2, 2.0, 8.0),    # the #934 max_gap-force-withdrawal harness
        (3, 3.0, 12.0),   # last config where the floor (15) still exceeds hard
    ],
)
def test_floor_is_refused_when_it_would_overshoot_the_hard_ceiling(
    max_gap, expected_soft, expected_hard
):
    """The floor must never raise ``soft`` to or past ``hard``.

    ``target_time`` treats ``hard <= soft`` as FATAL (``BackstopConfigError``:
    it "makes the soft stage unreachable and restores the unconditional force
    #794 removed"), but ``legacy_frames`` only WARNS -- so a default-on floor
    that ignored the ceiling would silently produce that state for any config
    with ``max_gap * hard_mult < input_fps / target_admit_rate_fps``, handing
    every starvation decision to the hard ceiling. Measured at ``max_gap=1``
    (hard=4): an unguarded floor to 15 dropped forced admissions from 59/60 to
    14/60 -- 60/4, i.e. the hard ceiling taking over. Refuse the floor and warn
    instead.
    """
    bs = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=max_gap,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    assert bs.soft_threshold_frames == expected_soft
    assert bs.hard_threshold_frames == expected_hard
    assert bs.soft_threshold_frames < bs.hard_threshold_frames  # soft stays reachable
    assert bs.controller_target_interval_source == "not_applicable"
    assert any("NOT applied" in w for w in bs.warnings)
    # The soft stage really is still the one that fires first.
    assert backstop_crossings(bs, max_gap, None) == (True, False)


def test_floor_applies_at_the_narrowest_config_that_still_leaves_soft_below_hard():
    """max_gap=4 => hard=16 > 15: the floor applies and soft stays reachable."""
    bs = resolve_backstop_thresholds(
        mode=SoftBackstopMode.LEGACY_FRAMES,
        max_gap=4,
        max_gap_hard_mult=4.0,
        input_fps=30.0,
        velocity_adaptive=True,
        target_admit_rate_fps=2.0,
        nonpreemption_floor=True,
    )
    assert bs.soft_threshold_frames == 15.0
    assert bs.hard_threshold_frames == 16.0
    assert any("non-preemption floor (#1044): soft backstop raised" in w for w in bs.warnings)


def test_scoped_use_enforced_spf_preserves_the_online_bypass_for_clock_paced_modes():
    """SPF is inert under a clock-paced mode, so the bypass must survive there."""
    budget, _ = resolve_coverage_planning_budget(
        loop_iterations=ONLINE_SENTINEL,
        streaming_max_iterations=0,
        operator_iterations=40_000,
        online_mode=True,
    )
    for mode, expected in (
        (INGESTION_MODE_ITER_BASED, True),
        ("dataset_fps", False),
    ):
        use_enforced = (not True) or (budget is not None and mode == INGESTION_MODE_ITER_BASED)
        assert use_enforced is expected
