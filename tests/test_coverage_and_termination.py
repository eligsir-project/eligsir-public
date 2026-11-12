"""Tests for the unified coverage/termination rule (2026-07-04 fix).

Covers:
- registry_n_source_frames resolution by path and by scene name (regression
  guard against the SCENE_SOURCE key-mismatch bug: ros2://live source_paths
  must miss the registry, not silently match).
- The rolling-idle-stop rule itself (train_streaming.py's inline
  _last_ingest_iter / streaming_refinement_iterations logic), replicated
  here as a small pure function mirroring the exact condition, exercised
  against both a synthetic finite (bag-like) and never-ending (live-like)
  frame arrival sequence.
"""

from __future__ import annotations

from types import SimpleNamespace

from eligsir.utils.partition_eval import registry_n_source_frames


class TestRegistryLookup:
    def test_known_bag_scene_resolves_by_path(self):
        n = registry_n_source_frames(
            "/data/datasets/floor3"
        )
        assert n == 5270

    def test_known_replay_scene_resolves_by_path(self):
        n = registry_n_source_frames(
            "/data/datasets/tum_fr3"
        )
        assert n == 2525

    def test_known_scene_resolves_by_name(self):
        assert registry_n_source_frames("floor3") == 5270

    def test_known_replay_scene_resolves_by_basename_when_mount_alias_differs(self):
        """Regression guard for task #43: different launch commands have bind-mounted
        fr3's dataset directory at different container-internal paths across this
        project's history (an identity mount matching the registry's recorded path
        verbatim, vs. an alternate mount alias used elsewhere) -- the
        aliased path must still resolve via basename fallback, not silently miss
        the registry and disable coverage enforcement.
        """
        n = registry_n_source_frames(
            "/data/tum_alias/rgbd_dataset_freiburg3_long_office_household"
        )
        assert n == 2525

    def test_ros2_live_source_path_misses_registry(self):
        """Regression guard: ros2://live must NOT accidentally match a registry entry."""
        assert registry_n_source_frames("ros2://live") is None

    def test_empty_or_none_misses_registry(self):
        assert registry_n_source_frames(None) is None
        assert registry_n_source_frames("") is None

    def test_unregistered_path_misses_registry(self):
        assert registry_n_source_frames("/some/random/unregistered/scene") is None


def _rolling_stop_iteration(ingest_iters, refinement_iters, max_iterations, first_iter=0):
    """Pure reimplementation of the rolling-idle-stop condition in
    train_streaming.py: exits once `iteration - _last_ingest_iter >=
    refinement_iters`, re-arming on every ingest event; `max_iterations`
    (0 = none) acts as an independent hard ceiling.
    """
    ingest_iters = sorted(ingest_iters)
    last_ingest_iter = first_iter
    ingest_idx = 0
    iteration = first_iter
    while True:
        if ingest_idx < len(ingest_iters) and iteration == ingest_iters[ingest_idx]:
            last_ingest_iter = iteration
            ingest_idx += 1
        if max_iterations and iteration >= max_iterations:
            return iteration, "ceiling"
        if iteration - last_ingest_iter >= refinement_iters:
            return iteration, "idle"
        iteration += 1


class TestRollingIdleStopRule:
    def test_finite_source_stops_refinement_iters_after_last_frame(self):
        """Bag-like source: last frame at iter=500, no ceiling -> stops at 500+refine."""
        iteration, reason = _rolling_stop_iteration(
            ingest_iters=[0, 100, 300, 500], refinement_iters=1_000, max_iterations=0
        )
        assert reason == "idle"
        assert iteration == 500 + 1_000

    def test_never_ending_source_keeps_extending_budget(self):
        """Live-like source: frames keep arriving faster than the refinement window."""
        ingest_iters = list(range(0, 5_000, 400))  # new frame every 400 iters
        iteration, reason = _rolling_stop_iteration(
            ingest_iters=ingest_iters, refinement_iters=1_000, max_iterations=0
        )
        # last ingest at 4800 -> stop at 4800+1000, well past any single gap
        assert reason == "idle"
        assert iteration == max(ingest_iters) + 1_000

    def test_explicit_ceiling_overrides_rolling_rule(self):
        """A smaller --streaming_max_iterations still acts as a hard ceiling."""
        iteration, reason = _rolling_stop_iteration(
            ingest_iters=list(range(0, 10_000, 100)),
            refinement_iters=1_000,
            max_iterations=2_000,
        )
        assert reason == "ceiling"
        assert iteration == 2_000

    def test_zero_ceiling_means_rely_purely_on_rolling_rule(self):
        iteration, reason = _rolling_stop_iteration(
            ingest_iters=[0], refinement_iters=500, max_iterations=0
        )
        assert reason == "idle"
        assert iteration == 500


from eligsir.utils.partition_eval import compute_trajectory_coverage, PARTITIONS


def _all_admitted_counters(n_source_frames: int) -> dict:
    """A complete all-admission source ledger for validity-composition tests."""
    return {
        "coverage_ledger_version": 2,
        "resolved_source_frames": n_source_frames,
        "raw_registry_source_frames": n_source_frames,
        "source_denominator_provenance": "post_stride_frame_source",
        "source_frames_ingested_total": n_source_frames,
        "bootstrap_frames_consumed": 0,
        "reserved_holdout_frames": 0,
        "reserved_guard_frames": 0,
        "rejected_invalid_frames": 0,
        "eligible_mapping_frames": n_source_frames,
        "frames_admitted": n_source_frames,
        "mapping_skipped_frames": 0,
        "unprocessed_frames": 0,
    }


def test_coverage_full_when_registered_and_all_partitions_populated():
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
    )
    assert result["scene_registered"] is True
    assert result["n_source_frames"] == 5270
    assert result["frames_admitted"] == 5270
    assert result["coverage_fraction"] == 1.0
    assert result["partitions_with_test_data"] == PARTITIONS
    assert result["partitions_total"] == PARTITIONS
    assert result["full_coverage"] is True
    assert result["verdict"] == "FULL"


def test_coverage_partial_by_frame_count():
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1513,
        partition_psnr=partition_psnr,
    )
    assert result["scene_registered"] is True
    assert abs(result["coverage_fraction"] - (1513 / 5270)) < 1e-9
    assert result["full_coverage"] is False
    assert result["verdict"] == "PARTIAL"


def test_coverage_partial_by_null_partitions_even_with_high_frame_count():
    # rfrand_replay50 pattern: near-full frame count but only 2/6 partitions
    # actually have test data -- must be flagged PARTIAL, not FULL.
    partition_psnr = {"p0": 25.37, "p1": 22.81, "p2": None, "p3": None,
                       "p4": None, "p5": None}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5250,
        partition_psnr=partition_psnr,
    )
    assert result["partitions_with_test_data"] == 2
    assert result["full_coverage"] is False
    assert result["verdict"] == "PARTIAL"


def test_coverage_partial_by_design_when_frame_admission_is_gated():
    # hybrid_keyframe (and other gated admission modes) deliberately reject
    # non-novel/overlapping frames -- a low coverage_fraction there reflects
    # healthy keyframe selection, not an under-covered or interrupted run.
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1216,
        partition_psnr=partition_psnr,
        frame_admission="hybrid_keyframe",
    )
    assert result["full_coverage"] is False
    assert result["verdict"] == "PARTIAL_BY_DESIGN"
    assert result["frame_admission"] == "hybrid_keyframe"


def test_coverage_partial_not_by_design_when_frame_admission_is_all():
    # Explicit "all" (or omitted) admission keeps the original PARTIAL verdict --
    # a low coverage_fraction there is a real under-coverage signal.
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1216,
        partition_psnr=partition_psnr,
        frame_admission="all",
    )
    assert result["verdict"] == "PARTIAL"


def test_coverage_soft_all_shortfall_is_by_budget_not_a_defect():
    """Issue #602: soft-all requests `all` but budgets mapping admission.

    Without the resolved target, such a run falls through to PARTIAL -- which
    reads as lost trajectory when the shortfall is actually a deliberate
    backpressure decision. It is also not PARTIAL_BY_DESIGN, which means a
    keyframe gate rejected non-novel frames (a different mechanism).
    """
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1216,
        partition_psnr=partition_psnr,
        frame_admission="all",
        admission_target_resolved="soft_all",
    )
    assert result["verdict"] == "PARTIAL_BY_BUDGET"
    assert result["full_coverage"] is False


def test_coverage_force_all_stress_shortfall_stays_partial():
    """force_all_stress preserves the historical ungated `all` semantics, so a
    shortfall there IS a real under-coverage signal, not a budget decision."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1216,
        partition_psnr=partition_psnr,
        frame_admission="all",
        admission_target_resolved="force_all_stress",
    )
    assert result["verdict"] == "PARTIAL"


def test_coverage_soft_all_at_full_coverage_still_reads_full():
    """The new label must not shadow FULL when the run did cover everything."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
        frame_admission="all",
        admission_target_resolved="soft_all",
    )
    assert result["verdict"] == "FULL"


def test_coverage_verdict_unchanged_when_target_not_declared():
    """Every pre-#602 call site is bit-for-bit unchanged."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    kwargs = dict(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=1216,
        partition_psnr=partition_psnr,
    )
    for adm, expected in (("all", "PARTIAL"),
                          ("hybrid_keyframe", "PARTIAL_BY_DESIGN"),
                          (None, "PARTIAL")):
        assert compute_trajectory_coverage(
            frame_admission=adm, **kwargs)["verdict"] == expected


def test_coverage_unknown_unregistered():
    partition_psnr = {f"p{i}": None for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/scene",
        n_reg=None,
        frames_admitted=458,
        partition_psnr=partition_psnr,
    )
    assert result["scene_registered"] is False
    assert result["n_source_frames"] is None
    assert result["coverage_fraction"] is None
    assert result["full_coverage"] is False
    assert result["verdict"] == "UNKNOWN_UNREGISTERED"


def test_coverage_unknown_live_source():
    partition_psnr = {f"p{i}": None for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="ros2://live",
        n_reg=None,
        frames_admitted=256,
        partition_psnr=partition_psnr,
    )
    assert result["scene_registered"] is False
    assert result["full_coverage"] is False
    assert result["verdict"] == "UNKNOWN_LIVE_SOURCE"


def test_coverage_unknown_live_source_with_resolvable_n_reg():
    """Regression test: ros2:// source with resolvable n_reg must still null out
    coverage fields. This tests the Important fix: even if a bag-replay ros2://
    source somehow has n_reg in the registry (e.g., n_reg=100), the verdict
    UNKNOWN_LIVE_SOURCE must force scene_registered/n_source_frames/coverage_fraction
    to False/None/None, preventing naive readers of report.json from treating
    coverage_fraction as meaningful.
    """
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="ros2://bag_replay_scene",
        n_reg=100,  # Resolvable! But still live source.
        frames_admitted=100,
        partition_psnr=partition_psnr,
    )
    # Despite resolvable n_reg, live source must null out all coverage fields.
    assert result["scene_registered"] is False
    assert result["n_source_frames"] is None
    assert result["coverage_fraction"] is None
    assert result["frames_admitted"] == 100  # admitted is still reported
    assert result["full_coverage"] is False
    assert result["verdict"] == "UNKNOWN_LIVE_SOURCE"


def test_coverage_handles_missing_frames_admitted():
    partition_psnr = {f"p{i}": 20.0 for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=None,
        partition_psnr=partition_psnr,
    )
    assert result["frames_admitted"] is None
    assert result["coverage_fraction"] is None
    assert result["full_coverage"] is False
    assert result["verdict"] == "PARTIAL"


def test_coverage_frame_drop_detected_false_by_default():
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
    )
    assert result["frame_drop_detected"] is False
    assert result["frames_dropped_by_inner_queue"] == 0


def test_coverage_frame_drop_detected_true_when_inner_queue_dropped_frames():
    """Final review (2026-08-06, PR #37): a bag-replay run whose inner
    ROS2LiveRGBDSource emit queue silently dropped frames (distinct from
    WS-D1's outer bounded-buffer eviction) must be flagged -- this is a real
    full-coverage-contract violation, not a benign admission rejection, and
    must carry the same "never quote this run's coverage/PSNR as valid"
    standing as map_reinit_invalidated even though full_coverage/verdict
    (which only look at admitted-vs-registered counts) might otherwise say
    FULL."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
        frames_dropped_by_inner_queue=3,
    )
    assert result["frame_drop_detected"] is True
    assert result["frames_dropped_by_inner_queue"] == 3
    # Distinct field from map_reinit_invalidated/verdict -- does not, by
    # itself, flip full_coverage/verdict (matches map_reinit_invalidated's
    # own precedent of being a separate flag, not folded into the others).
    assert result["map_reinit_invalidated"] is False
    # But it DOES flip the single authoritative validity signal, even with
    # eval_valid=True and verdict=FULL -- this is exactly the gap final
    # review round 2 flagged: a consumer checking only eval_valid/verdict
    # could accept a frame-dropped run, so mapping_result_valid must not.
    assert result["verdict"] == "FULL"
    assert compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
        frames_dropped_by_inner_queue=3,
        eval_valid=True,
    )["mapping_result_valid"] is False


def test_mapping_result_valid_true_only_when_every_condition_holds():
    """Final review round 2 (PR #37 finding #2): mapping_result_valid is the
    single field a consumer can check instead of eval_valid/verdict/
    map_reinit_invalidated/frame_drop_detected individually."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    clean = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270,
        frames_admitted=5270,
        partition_psnr=partition_psnr,
        eval_valid=True,
        coverage_counters=_all_admitted_counters(5270),
    )
    assert clean["mapping_result_valid"] is True

    # eval_valid missing/False -> invalid, even with a clean coverage verdict.
    assert compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
    )["mapping_result_valid"] is False
    assert compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=False,
    )["mapping_result_valid"] is False

    # Map reinit -> invalid, even with eval_valid=True.
    assert compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        tracking_reinit_events=[{"frame": 1234}],
    )["mapping_result_valid"] is False

    # Suspected-truncated verdict -> invalid, even with eval_valid=True.
    assert compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        stream_end_reason="idle_timeout_no_completion_signal",
    )["mapping_result_valid"] is False


def test_bag_replay_geometry_fields_none_does_not_invalidate():
    """W3-15: no BagReplayPolicy in use (the default, None) must not flip
    mapping_result_valid -- distinguishes 'no signal' from a real loss."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        bag_replay_geometry_fields=None,
        coverage_counters=_all_admitted_counters(5270),
    )
    assert result["mapping_result_valid"] is True
    assert result["bag_replay_mapping_result_valid"] is None


def test_bag_replay_valid_true_does_not_invalidate():
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        coverage_counters=_all_admitted_counters(5270),
        bag_replay_geometry_fields={
            "mapping_result_valid": True,
            "bag_frames_offered": 100,
            "bag_frames_spilled": 2,
            "bag_frames_lost": 0,
            "bag_loss_events": [],
        },
    )
    assert result["mapping_result_valid"] is True
    assert result["bag_replay_mapping_result_valid"] is True
    assert result["bag_frames_offered"] == 100
    assert result["bag_frames_lost"] == 0


def test_bag_replay_mapping_result_invalid_forces_mapping_result_invalid():
    """The actual behavioral requirement from the W3-15 brief: a real
    BagReplayIntegrityReport.mapping_result_valid=False signal must force
    the top-level mapping_result_valid to False even when eval_valid=True,
    verdict=FULL, no map-reinit, and no inner-queue drop -- exactly like
    frames_dropped_by_inner_queue already does."""
    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        coverage_counters=_all_admitted_counters(5270),
        bag_replay_geometry_fields={
            "mapping_result_valid": False,
            "bag_frames_offered": 100,
            "bag_frames_spilled": 4,
            "bag_frames_lost": 1,
            "bag_loss_events": [{"detail": "spill cap 4096 exceeded"}],
        },
    )
    assert result["verdict"] == "FULL"  # unrelated to coverage/verdict
    assert result["bag_replay_mapping_result_valid"] is False
    assert result["bag_frames_lost"] == 1
    assert result["mapping_result_valid"] is False


def test_real_bag_replay_policy_report_wires_end_to_end():
    """Uses the REAL BagReplayIntegrityReport.as_geometry_fields() (not a
    hand-built dict) to confirm the actual object this integration is for
    produces a compatible shape."""
    from eligsir.io.policies.bag_replay import BagReplayPolicy
    from eligsir.runtime.queues.bounded_queue import BoundedQueue

    policy = BagReplayPolicy(BoundedQueue(capacity=1), max_spill=0)
    policy.offer("frame-0")  # fills the queue
    policy.offer("frame-1")  # queue full, spill cap 0 -> LOST_SPILL_EXHAUSTED
    assert policy.report.mapping_result_valid is False

    partition_psnr = {f"p{i}": 20.0 + i for i in range(PARTITIONS)}
    result = compute_trajectory_coverage(
        scene_source="/data/datasets/floor3",
        n_reg=5270, frames_admitted=5270, partition_psnr=partition_psnr,
        eval_valid=True,
        bag_replay_geometry_fields=policy.report.as_geometry_fields(),
    )
    assert result["mapping_result_valid"] is False
    assert result["bag_frames_lost"] == 1


def test_trajectory_coverage_wiring_contract_matches_write_post_training_report_usage():
    """Verify the call contract (argument names/types) that write_post_training_report's
    actual call site uses; this is not a full integration test (which would require a
    heavier fixture with real render_fn/gaussians/pipe -- out of scope)."""
    from eligsir.utils import comparison_report

    # Smoke check: call compute_trajectory_coverage with the exact signature
    # that write_post_training_report uses, confirming the wiring contract holds.
    result = comparison_report.compute_trajectory_coverage(
        scene_source="/fake/scene",
        n_reg=10,
        frames_admitted=10,
        partition_psnr={f"p{i}": 20.0 for i in range(comparison_report.PARTITIONS)},
    )
    assert result["verdict"] == "FULL"
    assert result["frames_admitted"] == 10


from eligsir.utils.partition_eval import check_coverage_budget_feasible


def test_coverage_budget_feasible_raises_when_infeasible():
    import pytest
    with pytest.raises(RuntimeError, match="Coverage budget infeasible"):
        check_coverage_budget_feasible(
            is_finite_replay=True,
            enforce_full_coverage=True,
            n_frames=5270,
            iterations=3000,
            scene_source="/data/datasets/floor3",
        )


def test_coverage_budget_feasible_passes_when_enough_iterations():
    check_coverage_budget_feasible(
        is_finite_replay=True,
        enforce_full_coverage=True,
        n_frames=5270,
        iterations=60000,
        scene_source="/data/datasets/floor3",
    )


def test_coverage_budget_feasible_skipped_when_enforcement_off():
    # Must not raise even though iterations < n_frames, since enforcement is off.
    check_coverage_budget_feasible(
        is_finite_replay=True,
        enforce_full_coverage=False,
        n_frames=5270,
        iterations=3000,
        scene_source="/data/datasets/floor3",
    )


def test_coverage_budget_feasible_skipped_for_live_source():
    # Must not raise for a genuinely live/unbounded source (is_finite_replay=False),
    # even with iterations < n_frames -- n_frames has no fixed meaning there.
    check_coverage_budget_feasible(
        is_finite_replay=False,
        enforce_full_coverage=True,
        n_frames=999999,
        iterations=3000,
        scene_source="ros2://live",
    )


def test_coverage_budget_feasible_exact_boundary_passes():
    # iterations == n_frames is the exact floor (1 step/frame) -- must NOT raise.
    check_coverage_budget_feasible(
        is_finite_replay=True,
        enforce_full_coverage=True,
        n_frames=1453,
        iterations=1453,
        scene_source="/data/datasets/table2_dynamic",
    )


class _FakeCamera:
    def __init__(self, name, gt_value):
        self.image_name = name
        # A 3x2x2 tensor standing in for a real GT image; render_fn below
        # returns a tensor offset from this by a per-camera "residual" amount.
        import torch
        from eligsir.utils.score_contract import declare_camera_score_provenance

        self.original_image = torch.full((3, 2, 2), gt_value)
        declare_camera_score_provenance(self, context="test_residual_replay_gt")
        from eligsir.utils.score_contract import declare_camera_score_provenance

        declare_camera_score_provenance(self, context="test_residual_camera")


def _fake_render_fn_factory(residuals_by_name):
    """Returns a render_fn(cam, gaussians, pipe, background) that renders each
    camera as its GT image plus a fixed residual offset, so PSNR is
    deterministic and inversely related to the configured residual."""
    def render_fn(cam, gaussians, pipe, background):
        residual = residuals_by_name[cam.image_name]
        rendered = (cam.original_image + residual).clamp(0.0, 1.0)
        from eligsir.utils.score_contract import declare_canonical_score_tensor

        rendered, _ = declare_canonical_score_tensor(
            rendered, context="test_residual_renderer"
        )
        return {"render": rendered}

    return render_fn


def test_compute_residual_weights_favors_high_residual_camera():
    from scene.streaming_scene import StreamingScene

    cams = [_FakeCamera("low_residual", 0.5), _FakeCamera("high_residual", 0.5)]
    render_fn = _fake_render_fn_factory({"low_residual": 0.01, "high_residual": 0.4})
    scene = SimpleNamespace()
    weights = StreamingScene._compute_residual_weights(
        scene, cams, gaussians=None, render_fn=render_fn, pipe=None, background=None
    )
    assert weights is not None
    assert len(weights) == 2
    # Higher residual (worse PSNR) must get a higher sampling weight.
    assert weights[1] > weights[0]
    # Weights are a valid probability distribution (softmax output).
    assert abs(sum(weights) - 1.0) < 1e-5


def test_compute_residual_weights_scores_finite_renderer_overshoot():
    from scene.streaming_scene import StreamingScene

    cams = [_FakeCamera("low_residual", 0.5), _FakeCamera("overshoot", 0.5)]

    def render_fn(cam, gaussians, pipe, background):
        value = 0.51 if cam.image_name == "low_residual" else 1.1
        return {"render": cam.original_image.new_full((3, 2, 2), value)}

    scene = SimpleNamespace()
    weights = StreamingScene._compute_residual_weights(
        scene, cams, gaussians=None, render_fn=render_fn, pipe=None, background=None
    )

    assert weights is not None
    assert weights[1] > weights[0]
    assert scene._residual_replay_score_clamp["samples_changed"] == 12


def test_compute_residual_weights_fails_open_on_render_error():
    from scene.streaming_scene import StreamingScene

    cams = [_FakeCamera("a", 0.5)]

    def broken_render_fn(cam, gaussians, pipe, background):
        raise RuntimeError("simulated render failure")

    scene = SimpleNamespace()
    weights = StreamingScene._compute_residual_weights(
        scene, cams, gaussians=None, render_fn=broken_render_fn, pipe=None, background=None
    )
    assert weights is None
    assert scene._residual_replay_score_fallbacks == 1


def test_compute_residual_weights_empty_pool_returns_none():
    from scene.streaming_scene import StreamingScene

    weights = StreamingScene._compute_residual_weights(
        SimpleNamespace(), [], gaussians=None, render_fn=lambda *a: None, pipe=None, background=None
    )
    assert weights is None


def test_sample_training_camera_accepts_render_kwargs_without_error():
    """Wiring-contract check: sample_training_camera must accept the new
    render_fn/pipe/background kwargs without raising, even with residual
    weighting off (default) -- this is the call-site contract
    train_streaming.py now depends on."""
    import inspect
    from scene.streaming_scene import StreamingScene

    sig = inspect.signature(StreamingScene.sample_training_camera)
    assert "render_fn" in sig.parameters
    assert "pipe" in sig.parameters
    assert "background" in sig.parameters
    assert sig.parameters["render_fn"].default is None
    assert sig.parameters["pipe"].default is None
    assert sig.parameters["background"].default is None
