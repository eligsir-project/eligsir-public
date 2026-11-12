"""Issue #600: observation -> admission -> insertion decision contract.

Covers the four test requirements the issue names explicitly, plus the two
downstream-instrument hazards the #604 audit (PR #619) identified as the
highest-damage part of this migration.

Torch-free by construction -- every module under test is in the same CPU-CI
scope fence (see utils/execution_semantics.py's docstring).
"""
from __future__ import annotations

import pytest

from eligsir.mapping.admission.observation_decision import (
    REASON_ADMITTED,
    REASON_CAPACITY,
    REASON_FRONTIER_FORCED,
    ObservationDecision,
    SoftAllAdmitter,
    force_all_decision,
)
from eligsir.utils import execution_semantics as es
from eligsir.utils.partition_eval import compute_trajectory_coverage


# --------------------------------------------------------------------------- #
# 1. The decision contract itself
# --------------------------------------------------------------------------- #

def test_insertion_invariant_is_enforced_not_documented():
    """A frame rejected for mapping cannot be marked insertion-eligible.

    This is #600's central code invariant. Expressing it as a constructor check
    means it cannot be violated by a future caller that forgets it.
    """
    with pytest.raises(ValueError, match="insertion invariant"):
        ObservationDecision(
            admit_supervision=False, allow_geometry_insertion=True
        )


def test_admitted_frame_may_insert():
    d = ObservationDecision()
    assert d.admitted is True
    assert d.allow_geometry_insertion is True


def test_skipped_frame_is_still_observed():
    """#600: 'skipped mapping frames may still advance pose/source/history'."""
    d = ObservationDecision(
        admit_supervision=False,
        allow_geometry_insertion=False,
        reason=REASON_CAPACITY,
    )
    assert d.observe is True          # pose/history still advance
    assert d.mapping_candidate is True  # it WAS considered
    assert d.admitted is False          # but not supervised
    assert d.allow_geometry_insertion is False  # and never inserted


def test_decision_axes_are_independently_observable():
    """Each axis is a separate field in the per-frame stats, not inferred."""
    stats = ObservationDecision(
        admit_supervision=False,
        allow_geometry_insertion=False,
        reason=REASON_CAPACITY,
    ).as_stats("all")
    for key in (
        "observe",
        "mapping_candidate",
        "admit_supervision",
        "allow_geometry_insertion",
        "reason",
    ):
        assert key in stats, f"{key} must be independently observable"


# --------------------------------------------------------------------------- #
# 2. spf=2 + soft-all with an overloaded fake mapper (issue's test 1)
# --------------------------------------------------------------------------- #

class _FakeMapper:
    """A mapper that is too slow to keep up with the release cadence.

    ``steps`` is the authoritative optimizer-step clock SoftAllAdmitter reads.
    An overloaded mapper advances it more slowly than frames arrive.
    """

    def __init__(self, steps_per_frame: int):
        self.steps = 0
        self.steps_per_frame = steps_per_frame

    def run(self):
        self.steps += self.steps_per_frame


def test_spf2_soft_all_overloaded_mapper_bounds_admission():
    """spf=2 + soft-all: all source frames observed, bounded subset admitted."""
    mapper = _FakeMapper(steps_per_frame=2)  # effective_spf = 2
    admitter = SoftAllAdmitter(
        min_steps_per_admit=8,      # mapper can only deliver 2 -> overloaded
        max_consecutive_skips=4,
        steps_provider=lambda: mapper.steps,
    )

    observed = 0
    admitted = 0
    skipped_decisions = []
    for _ in range(60):
        observed += 1                      # every source frame is observed
        decision = admitter.decide()
        if decision.admitted:
            admitted += 1
        else:
            skipped_decisions.append(decision)
        mapper.run()

    # (a) every source observation advanced
    assert observed == 60
    assert admitter.n_evaluated == 60

    # (b) only a bounded subset became mapping-admitted
    assert admitted < observed, "an overloaded mapper must not admit every frame"
    assert admitter.n_skipped_capacity > 0, "the capacity bound must engage"
    assert admitter.summary()["capacity_bound_engaged"] is True

    # (c) no skipped frame is insertion-eligible
    assert skipped_decisions, "expected some skips"
    for d in skipped_decisions:
        assert d.allow_geometry_insertion is False
        assert d.admit_supervision is False
        assert d.observe is True           # still observed


def test_soft_all_frontier_bound_prevents_starvation():
    """No observation may be starved indefinitely (the age/frontier bound)."""
    mapper = _FakeMapper(steps_per_frame=0)  # a completely stalled mapper
    admitter = SoftAllAdmitter(
        min_steps_per_admit=100,
        max_consecutive_skips=3,
        steps_provider=lambda: mapper.steps,
    )
    reasons = [admitter.decide().reason for _ in range(20)]
    # Even with a permanently stalled mapper, a frame gets in every 4th slot.
    assert REASON_FRONTIER_FORCED in reasons
    assert admitter.n_frontier_forced > 0
    # And it never skips more than max_consecutive_skips in a row.
    run = 0
    for r in reasons:
        if r == REASON_CAPACITY:
            run += 1
            assert run <= 3, "frontier bound exceeded"
        else:
            run = 0


def test_effective_spf_2_does_not_imply_2_updates_per_admitted_frame():
    """The issue's explicit assertion: spf is release pacing, not completion.

    With a mapper delivering fewer steps than the nominal spf, the REALIZED
    updates-per-admitted-frame must be measured, never assumed equal to spf.
    """
    nominal_spf = 2
    mapper = _FakeMapper(steps_per_frame=1)  # only half the nominal rate
    admitter = SoftAllAdmitter(
        min_steps_per_admit=0,  # bound disabled: isolate the measurement claim
        max_consecutive_skips=4,
        steps_provider=lambda: mapper.steps,
    )
    admitted = 0
    for _ in range(50):
        if admitter.decide().admitted:
            admitted += 1
        mapper.run()

    realized = mapper.steps / admitted
    assert admitted == 50  # bound disabled -> hard-all-like admission
    assert realized != nominal_spf, (
        "effective_spf must not be read as optimizer_updates_per_admitted_frame"
    )
    assert realized == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 3. force_all preserves historical hard behavior (issue's test 2)
# --------------------------------------------------------------------------- #

def test_force_all_decision_is_unconditional_admission():
    """force_all reproduces pre-#600 hard-all: admit, insert, never skip."""
    d = force_all_decision()
    assert d.observe is True
    assert d.mapping_candidate is True
    assert d.admit_supervision is True
    assert d.allow_geometry_insertion is True
    assert d.admitted is True


def test_force_all_resolves_to_stress_semantics_and_is_unbudgeted():
    r = es.resolve_admission_semantics("force_all")
    assert r["admission_target"] == es.ADMISSION_TARGET_FORCE_ALL_STRESS
    assert r["admission_budgeted"] is False


def test_force_all_emits_a_loud_provenance_warning():
    """force_all must be visibly marked experimental/stress in every run."""
    class _Args:
        streaming_replay = True
        streaming_frame_admission = "force_all"
        source_path = "/data/tum/fr3"

    block = es.build_execution_semantics_block(_Args())
    assert any(
        "force_all_stress" in w and "EXPERIMENTAL" in w
        for w in block["warnings"]
    ), block["warnings"]


def test_production_all_does_not_emit_the_stress_warning():
    class _Args:
        streaming_replay = True
        streaming_frame_admission = "all"
        source_path = "/data/tum/fr3"

    block = es.build_execution_semantics_block(_Args())
    assert not any("EXPERIMENTAL" in w for w in block["warnings"])


# --------------------------------------------------------------------------- #
# 4. hybrid/selective admission unchanged (issue's test 3)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("token", ["hybrid_keyframe", "fixed_stride"])
def test_gated_modes_resolve_to_selective_and_are_unchanged(token):
    r = es.resolve_admission_semantics(token)
    assert r["admission_target"] == es.ADMISSION_TARGET_SELECTIVE
    assert r["admission_budgeted"] is True
    assert r["admission_policy_requested"] == token


def test_gated_modes_still_grade_partial_by_design():
    """This refactor must not alter selective admission's grading."""
    cov = compute_trajectory_coverage(
        scene_source="/data/tum/fr3",
        n_reg=2525,
        frames_admitted=500,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="hybrid_keyframe",
        admission_target_resolved=es.ADMISSION_TARGET_SELECTIVE,
    )
    assert cov["verdict"] == "PARTIAL_BY_DESIGN"


# --------------------------------------------------------------------------- #
# 5. The #604 downstream-instrument hazards
# --------------------------------------------------------------------------- #

def test_soft_all_run_grades_partial_by_budget_not_partial():
    """THE highest-damage hazard: soft-all still carries the token 'all'.

    Without the resolved field, partition_eval's old `!= "all"` test would grade
    every production run PARTIAL and mass-false-invalidate them.
    """
    cov = compute_trajectory_coverage(
        scene_source="/data/tum/fr3",
        n_reg=2525,
        frames_admitted=1800,  # below 0.99: the capacity bound skipped some
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="all",  # raw token, unchanged
        admission_target_resolved=es.ADMISSION_TARGET_SOFT_ALL,
    )
    # #602 (PR #1081) gave soft-all its OWN verdict rather than folding it into
    # PARTIAL_BY_DESIGN: a budgeted backpressure decision and a keyframe gate
    # rejecting non-novel frames are different mechanisms.
    assert cov["verdict"] == "PARTIAL_BY_BUDGET"


def test_force_all_shortfall_still_grades_partial():
    """Under force_all a missing frame was LOST, not skipped -- keep flagging it."""
    cov = compute_trajectory_coverage(
        scene_source="/data/tum/fr3",
        n_reg=2525,
        frames_admitted=1800,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="force_all",
        admission_target_resolved=es.ADMISSION_TARGET_FORCE_ALL_STRESS,
    )
    assert cov["verdict"] == "PARTIAL"


def test_historical_report_grading_is_unchanged():
    """A pre-#600 report has no resolved field: bare 'all' must still be PARTIAL.

    #600: 'Do NOT silently reinterpret old run manifests.'
    """
    cov = compute_trajectory_coverage(
        scene_source="/data/tum/fr3",
        n_reg=2525,
        frames_admitted=1800,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="all",
        admission_target_resolved=None,  # pre-#600 report
    )
    assert cov["verdict"] == "PARTIAL"


def test_full_coverage_still_wins_over_by_design():
    """A soft-all run that DID cover everything still grades FULL."""
    cov = compute_trajectory_coverage(
        scene_source="/data/tum/fr3",
        n_reg=100,
        frames_admitted=100,
        partition_psnr={f"p{i}": 20.0 for i in range(6)},
        frame_admission="all",
        admission_target_resolved=es.ADMISSION_TARGET_SOFT_ALL,
    )
    assert cov["verdict"] == "FULL"


# --------------------------------------------------------------------------- #
# 6. Contract/provenance plumbing
# --------------------------------------------------------------------------- #

def test_resolved_fields_are_present_in_the_semantics_block():
    class _Args:
        streaming_replay = True
        streaming_frame_admission = "all"
        source_path = "/data/tum/fr3"

    block = es.build_execution_semantics_block(_Args())
    assert block["admission_target"] == es.ADMISSION_TARGET_SOFT_ALL
    assert block["admission_budgeted"] is True
    assert block["source_coverage_target"] == es.COVERAGE_TARGET_FULL
    assert block["admission_policy_requested"] == "all"
    # The raw request stays auditable.
    assert block["raw_flags"]["streaming_frame_admission"] == "all"


def test_orbbec_override_no_longer_diverges_between_banner_and_report():
    """The trainer rewrites all -> hybrid_keyframe for open3d_odometry_live.

    Mirroring it here is what stops the startup banner and report.json
    disagreeing about the same run (the fix #594 applied to pose_source).
    """
    class _Args:
        streaming_replay = True
        streaming_frame_admission = "all"
        orbbec_pose_source = "open3d_odometry_live"
        source_path = "/data/tum/fr3"

    block = es.build_execution_semantics_block(_Args())
    assert block["admission_policy"] == "hybrid_keyframe"
    assert block["admission_target"] == es.ADMISSION_TARGET_SELECTIVE
    # ...while the raw request remains visible for auditability.
    assert block["raw_flags"]["streaming_frame_admission"] == "all"


def test_force_all_is_not_rewritten_by_the_orbbec_override():
    class _Args:
        streaming_replay = True
        streaming_frame_admission = "force_all"
        orbbec_pose_source = "open3d_odometry_live"
        source_path = "/data/tum/fr3"

    block = es.build_execution_semantics_block(_Args())
    assert block["admission_target"] == es.ADMISSION_TARGET_FORCE_ALL_STRESS


def test_live_source_reports_realtime_coverage_target():
    r = es.resolve_admission_semantics("all", source_mode=es.SOURCE_LIVE)
    assert r["source_coverage_target"] == es.COVERAGE_TARGET_REALTIME


def test_soft_all_summary_reports_inertness_honestly():
    """A bound that never engaged must say so, not read as 'capacity-aware'."""
    mapper = _FakeMapper(steps_per_frame=50)  # a healthy run
    admitter = SoftAllAdmitter(
        min_steps_per_admit=1,  # the shipped default
        max_consecutive_skips=4,
        steps_provider=lambda: mapper.steps,
    )
    for _ in range(100):
        assert admitter.decide().admitted is True
        mapper.run()
    s = admitter.summary()
    assert s["n_skipped_capacity"] == 0
    assert s["capacity_bound_engaged"] is False
    assert s["n_admitted"] == 100


def test_default_bound_is_inert_on_a_healthy_run_but_bounds_a_stalled_one():
    """The shipped default (1) reproduces today's behavior unless steps stall."""
    stalled = _FakeMapper(steps_per_frame=0)
    admitter = SoftAllAdmitter(
        min_steps_per_admit=1,
        max_consecutive_skips=4,
        steps_provider=lambda: stalled.steps,
    )
    reasons = [admitter.decide().reason for _ in range(20)]
    # With zero optimizer progress, admission is bounded rather than unbounded.
    assert REASON_CAPACITY in reasons
    assert reasons[0] == REASON_ADMITTED  # first frame always gets in
