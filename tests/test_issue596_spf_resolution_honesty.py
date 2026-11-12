"""Issue #596 — startup SPF diagnostics must agree with the real scheduler.

The defect: a run printed "steps_per_frame 150 would need to shrink to 2 ...
Not enforced" and then trained with exactly that ``2`` governing frame
release. These tests pin the three things that made that possible:

1. the *attribution* of ``MappingScheduler.effective_spf`` (request vs.
   coverage arithmetic vs. the ``min_steps_per_frame`` floor), including the
   registry-hit / registry-miss branch;
2. that the record's ``resolved_spf`` is the value ``FrameScheduler`` is
   actually constructed with — so log text and metadata cannot claim a
   different number than the release gate uses;
3. that the log text never says "not enforced" about a value the gate reads.

These are pure-function tests over ``MappingScheduler`` / ``SpfResolution`` /
``FrameScheduler``; no training entrypoint, no GPU.
"""

from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import (
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_ITER_BASED,
    RESOLVED_BY,
    RESOLVED_BY_COVERAGE,
    RESOLVED_BY_MIN_SPF_FLOOR,
    RESOLVED_BY_ONLINE_BYPASS,
    RESOLVED_BY_REQUEST,
    TERMINATION_FIXED_BUDGET,
    TERMINATION_ROLLING_IDLE,
    FrameScheduler,
    MappingScheduler,
    build_spf_resolution,
    resolve_coverage_planning_budget,
)


def _resolution_like_train_streaming(
    *,
    requested_spf: int,
    n_frames: int,
    iterations: int,
    registry_hit: bool,
    enforce_full_coverage: bool = True,
    min_steps_per_frame: int = 50,
    online_mode: bool = False,
    ingestion_mode: str = INGESTION_MODE_ITER_BASED,
):
    """Mirror ``train_streaming.py``'s resolution order exactly.

    Kept deliberately literal — including the registry-miss rule that forces
    coverage enforcement off and the ``_use_enforced_spf`` bypass — so a
    divergence between this helper and the training loop shows up as a test
    failure rather than as a silently stale fixture.
    """
    # #1044: train_streaming.py plans coverage against a resolved budget, not
    # against the (possibly-sentinel) loop bound.
    planning_budget, budget_source = resolve_coverage_planning_budget(
        loop_iterations=iterations,
        streaming_max_iterations=0,
        operator_iterations=iterations,
        online_mode=online_mode,
    )
    scheduler = MappingScheduler(
        steps_per_frame=requested_spf,
        n_frames=n_frames,
        iterations=(planning_budget if planning_budget is not None else 0),
        # train_streaming.py: registry miss => enforcement forced off.
        enforce_full_coverage=(registry_hit and enforce_full_coverage),
        min_steps_per_frame=min_steps_per_frame,
    )
    # #1044: scoped to iter_based -- under a clock-paced mode SPF is inert, so
    # the historical online bypass is preserved there.
    use_enforced_spf = (not online_mode) or (
        planning_budget is not None
        and ingestion_mode == INGESTION_MODE_ITER_BASED
    )
    frame_scheduler_spf = (
        max(1, scheduler.effective_spf) if use_enforced_spf else max(1, requested_spf)
    )
    projected_coverage_frac = None
    if planning_budget is not None and n_frames > 0:
        projected_coverage_frac = min(
            1.0, (planning_budget // max(1, frame_scheduler_spf)) / float(n_frames)
        )
    frame_scheduler = FrameScheduler(
        fps=30.0, steps_per_frame=frame_scheduler_spf, ingestion_mode=ingestion_mode
    )
    resolution = build_spf_resolution(
        requested_spf=requested_spf,
        resolved_spf=frame_scheduler_spf,
        scheduler_effective_spf=scheduler.effective_spf,
        scheduler_resolved_by=scheduler.spf_resolved_by,
        coverage_required_spf=scheduler.coverage_required_spf,
        min_spf_floor=scheduler.min_steps_per_frame,
        use_enforced_spf=use_enforced_spf,
        # train_streaming.py sources this from
        # _ingestion_report["steps_per_frame_honored"].
        governs_release=(ingestion_mode == INGESTION_MODE_ITER_BASED),
        ingestion_mode=ingestion_mode,
        coverage_budget=(planning_budget or 0),
        coverage_n_frames=n_frames,
        registry_hit=registry_hit,
        full_coverage_enforced=scheduler.stats.full_coverage,
        n_frames_capped_diagnostic=scheduler.n_frames_capped,
        online_mode=online_mode,
        coverage_budget_source=budget_source,
        projected_coverage_frac=projected_coverage_frac,
    )
    return scheduler, frame_scheduler, resolution


# --------------------------------------------------------------------------
# 1. SPF must shrink to fit a finite budget (the issue's own run)
# --------------------------------------------------------------------------


def test_observed_run_shape_reports_coverage_enforcement_not_advisory():
    """The exact numbers from the #596 run: 7040 frames, 15000 iters, spf 150.

    Coverage arithmetic gives ``floor(15000/7040) == 2``. The old log called
    that "not enforced"; it governed the whole run.
    """
    scheduler, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=150,
        n_frames=7040,
        iterations=15000,
        registry_hit=True,
        # The observed run's floor did not bind (2 < 50 would have tripped it),
        # so reproduce the run by disabling the floor rather than by pretending
        # the clamp does not exist -- see the min_spf test below.
        min_steps_per_frame=0,
    )
    assert res.requested_spf == 150
    assert res.coverage_required_spf == 2
    assert res.resolved_spf == 2
    assert res.resolved_by == RESOLVED_BY_COVERAGE
    assert res.governs_release is True
    assert res.termination_policy == TERMINATION_FIXED_BUDGET

    text = res.format_log()
    assert "IS ENFORCED" in text
    assert "not enforced" not in text.lower()
    # The record and the object that actually gates release agree.
    assert frame_scheduler.steps_per_frame == res.resolved_spf


def test_requested_spf_survives_when_budget_is_ample():
    _, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=20,
        n_frames=100,
        iterations=100_000,
        registry_hit=True,
        min_steps_per_frame=0,
    )
    assert res.resolved_spf == 20
    assert res.resolved_by == RESOLVED_BY_REQUEST
    assert frame_scheduler.steps_per_frame == 20


def test_min_spf_floor_is_attributed_and_named_in_the_log():
    """W3-38's upward clamp: recorded, deliberately NOT changed by #596."""
    scheduler, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=20,
        n_frames=7040,
        iterations=15000,
        registry_hit=True,
        min_steps_per_frame=50,
    )
    # Behaviour unchanged: the floor still wins, and still exceeds the request.
    assert scheduler.effective_spf == 50
    assert frame_scheduler.steps_per_frame == 50
    assert res.coverage_required_spf == 2
    assert res.resolved_spf == 50
    assert res.resolved_by == RESOLVED_BY_MIN_SPF_FLOOR
    assert res.n_frames_capped_diagnostic is not None

    text = res.format_log()
    assert "min_spf_floor" in text
    assert "RAISED" in text
    # The frame-capping statistic must still be labelled diagnostic-only.
    assert "diagnostic only" in text
    assert "not enforced" not in text.lower()


# --------------------------------------------------------------------------
# 2. Registry hit vs. miss
# --------------------------------------------------------------------------


def test_registry_miss_disables_coverage_and_the_request_wins():
    scheduler, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=150,
        n_frames=7040,
        iterations=15000,
        registry_hit=False,
    )
    assert res.registry_hit is False
    assert res.full_coverage_enforced is False
    assert res.coverage_required_spf is None
    assert res.resolved_spf == 150
    assert res.resolved_by == RESOLVED_BY_REQUEST
    assert frame_scheduler.steps_per_frame == 150
    assert "missed the eval/scenes.json registry" in res.format_log()


def test_registry_hit_and_miss_differ_only_in_who_resolved_spf():
    """The hit/miss branch is the discriminator between request- and
    coverage-owned SPF at otherwise identical settings (#596 finding)."""
    _, _, hit = _resolution_like_train_streaming(
        requested_spf=150, n_frames=7040, iterations=15000,
        registry_hit=True, min_steps_per_frame=0,
    )
    _, _, miss = _resolution_like_train_streaming(
        requested_spf=150, n_frames=7040, iterations=15000,
        registry_hit=False, min_steps_per_frame=0,
    )
    assert hit.requested_spf == miss.requested_spf
    assert hit.resolved_spf != miss.resolved_spf
    assert (hit.resolved_by, miss.resolved_by) == (
        RESOLVED_BY_COVERAGE,
        RESOLVED_BY_REQUEST,
    )


def test_explicit_opt_out_of_coverage_enforcement_keeps_the_request():
    _, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=150,
        n_frames=7040,
        iterations=15000,
        registry_hit=True,
        enforce_full_coverage=False,
    )
    assert res.full_coverage_enforced is False
    assert res.resolved_spf == 150
    assert frame_scheduler.steps_per_frame == 150


# --------------------------------------------------------------------------
# 3. The two "resolved but not read" states
# --------------------------------------------------------------------------


def test_online_mode_bypasses_the_scheduler_and_says_so():
    scheduler, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=150,
        n_frames=7040,
        iterations=15000,
        registry_hit=True,
        online_mode=True,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        min_steps_per_frame=0,
    )
    # The scheduler still computes 2; it is simply never handed over.
    assert scheduler.effective_spf == 2
    assert res.resolved_spf == 150
    assert frame_scheduler.steps_per_frame == 150
    assert res.resolved_by == RESOLVED_BY_ONLINE_BYPASS
    assert res.termination_policy == TERMINATION_ROLLING_IDLE
    assert res.governs_release is False
    text = res.format_log()
    assert "NOT read by the release gate" in text
    assert "IS ENFORCED" not in text


def test_non_online_dataset_fps_resolves_from_coverage_but_does_not_govern():
    """The third state: coverage-resolved value reaches FrameScheduler, but a
    clock-paced release gate never reads it."""
    _, frame_scheduler, res = _resolution_like_train_streaming(
        requested_spf=150,
        n_frames=7040,
        iterations=15000,
        registry_hit=True,
        online_mode=False,
        ingestion_mode=INGESTION_MODE_DATASET_FPS,
        min_steps_per_frame=0,
    )
    assert res.resolved_spf == 2
    assert frame_scheduler.steps_per_frame == 2
    assert res.resolved_by == RESOLVED_BY_COVERAGE
    assert res.governs_release is False
    assert "NOT read by the release gate" in res.format_log()


# --------------------------------------------------------------------------
# 4. Log text vs. machine-readable metadata vs. real scheduler state
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(requested_spf=150, n_frames=7040, iterations=15000,
             registry_hit=True, min_steps_per_frame=0),
        dict(requested_spf=20, n_frames=7040, iterations=15000,
             registry_hit=True, min_steps_per_frame=50),
        dict(requested_spf=150, n_frames=7040, iterations=15000,
             registry_hit=False),
        dict(requested_spf=150, n_frames=7040, iterations=15000,
             registry_hit=True, online_mode=True,
             ingestion_mode=INGESTION_MODE_DATASET_FPS),
        dict(requested_spf=50, n_frames=100, iterations=100_000,
             registry_hit=True),
    ],
)
def test_log_metadata_and_frame_scheduler_all_agree(kwargs):
    _, frame_scheduler, res = _resolution_like_train_streaming(**kwargs)
    payload = res.as_dict()

    # Metadata == the object that gates release.
    assert payload["resolved_spf"] == frame_scheduler.steps_per_frame
    # Log text == metadata (both formatted from the same record).
    text = res.format_log()
    assert f"resolved_spf={payload['resolved_spf']}" in text
    assert f"resolved_by={payload['resolved_by']}" in text
    assert f"requested_spf={payload['requested_spf']}" in text
    assert f"governs_release={payload['governs_release']}" in text
    assert f"ingestion_mode={payload['ingestion_mode']}" in text

    assert payload["resolved_by"] in RESOLVED_BY
    # Acceptance criterion 1, stated as an invariant.
    assert not (payload["governs_release"] and "not enforced" in text.lower())


def test_record_is_json_serializable():
    import json

    _, _, res = _resolution_like_train_streaming(
        requested_spf=150, n_frames=7040, iterations=15000, registry_hit=True,
    )
    assert json.loads(json.dumps(res.as_dict())) == res.as_dict()


# --------------------------------------------------------------------------
# 5. The record must never become a decision-maker
# --------------------------------------------------------------------------


def test_build_spf_resolution_records_resolved_spf_verbatim():
    """Guard the #596 fence: the builder attributes, it never recomputes.

    Passing a resolved value that contradicts the scheduler must be reported
    as ``unattributed``, never silently corrected to the scheduler's number.
    """
    res = build_spf_resolution(
        requested_spf=150,
        resolved_spf=7,
        scheduler_effective_spf=2,
        scheduler_resolved_by=RESOLVED_BY_COVERAGE,
        coverage_required_spf=2,
        min_spf_floor=0,
        use_enforced_spf=True,
        governs_release=True,
        ingestion_mode=INGESTION_MODE_ITER_BASED,
        coverage_budget=15000,
        coverage_n_frames=7040,
        registry_hit=True,
        full_coverage_enforced=True,
        n_frames_capped_diagnostic=None,
    )
    assert res.resolved_spf == 7
    assert res.resolved_by == "unattributed"


def test_spf_resolution_is_immutable():
    _, _, res = _resolution_like_train_streaming(
        requested_spf=150, n_frames=7040, iterations=15000, registry_hit=True,
    )
    with pytest.raises(Exception):
        res.resolved_spf = 999  # type: ignore[misc]
