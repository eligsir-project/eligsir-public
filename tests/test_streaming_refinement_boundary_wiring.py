"""Contract tests for the explicit mapping -> refinement boundary.

The full streaming trainer requires a real GPU and source, so the small pure
helpers are exercised directly and the multi-thousand-line loop integration is
guarded by source inspection, following the repository's existing wiring-test
convention.
"""

from __future__ import annotations

import inspect
import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_streaming
from arguments import StreamingParams
from eligsir.mapping.integration.checkpoint_io import _save_streaming_checkpoint
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime
from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
from eligsir.utils.comparison_report import write_post_training_report
from eligsir.utils.mcmc_schedule import MCMCScheduleConfig


class _ImmediateWorker:
    def __init__(self):
        self.calls = 0

    def enqueue(self, fn):
        self.calls += 1
        fn()

    def join(self):
        return None


class _FakeGaussians:
    def capture(self):
        return {"means": "state"}


def test_post_mapping_refinement_cli_is_default_off_and_configurable():
    parser = ArgumentParser()
    params = StreamingParams(parser)

    defaults = params.extract(parser.parse_args([]))
    configured = params.extract(
        parser.parse_args(["--streaming_post_mapping_refinement_steps", "17"])
    )

    assert defaults.streaming_post_mapping_refinement_steps == 0
    assert defaults.streaming_refinement_iterations == 50_000
    assert configured.streaming_post_mapping_refinement_steps == 17
    assert configured.streaming_refinement_iterations == 50_000


def test_exact_causal_fork_cli_is_default_off_and_configurable():
    parser = ArgumentParser()
    params = StreamingParams(parser)

    assert params.extract(parser.parse_args([])).streaming_exact_fork_iteration == -1
    assert params.extract(
        parser.parse_args(["--streaming_exact_fork_iteration", "41"])
    ).streaming_exact_fork_iteration == 41


def test_source_completeness_counts_holdouts_not_train_cameras():
    # Only seven frames became train cameras, but all ten source observations
    # (including holdouts/rejections) received a terminal disposition.
    scene = SimpleNamespace(_source_idx=10, _frame_count=10, train_cameras=[object()] * 7)
    exhausted, dispositions_complete = train_streaming._source_disposition_state(scene)

    assert exhausted is True
    assert dispositions_complete is True


def test_source_completeness_rejects_train_camera_count_alone():
    scene = SimpleNamespace(_source_idx=9, _frame_count=10, train_cameras=[object()] * 10)

    assert train_streaming._source_disposition_state(scene) == (False, False)


def test_mcmc_post_ingestion_schedule_uses_full_source_cursor_with_holdouts():
    """Held-out frames still count toward source completion for refinement."""

    resolver = getattr(train_streaming, "_streaming_mcmc_schedule", None)
    assert callable(resolver), "streaming MCMC needs a source-cursor completion boundary"

    scene = SimpleNamespace(_source_idx=10, _frame_count=10)
    cfg = MCMCScheduleConfig(
        start_iter=0,
        stop_growth_iter=10_000,
        growth_factor_start=1.05,
        post_ingestion_growth_factor_start=1.50,
    )

    schedule = resolver(
        iteration=100,
        current_n=1_000,
        cap_max=50_000,
        cfg=cfg,
        streaming_scene=scene,
    )

    assert schedule["growth_factor"] > 1.40


def test_checkpoint_extra_metadata_is_written_without_overriding_cursor(tmp_path):
    worker = _ImmediateWorker()
    path = _save_streaming_checkpoint(
        worker,
        _FakeGaussians(),
        str(tmp_path),
        12,
        10,
        11,
        10,
        extra_metadata={"streaming_phase": {"phase": "refinement"}},
    )

    import torch

    _state, _iteration, metadata = torch.load(path, weights_only=False)
    assert worker.calls == 1
    assert metadata["source_idx"] == 10
    assert metadata["checkpoint_reason"] == "periodic"
    assert metadata["streaming_phase"] == {"phase": "refinement"}


def test_checkpoint_extra_metadata_rejects_base_key_collision(tmp_path):
    worker = _ImmediateWorker()

    with pytest.raises(ValueError, match="metadata.*source_idx"):
        _save_streaming_checkpoint(
            worker,
            _FakeGaussians(),
            str(tmp_path),
            12,
            10,
            11,
            10,
            extra_metadata={"source_idx": 999},
        )
    assert worker.calls == 0


def test_phase_metadata_writer_is_atomic_and_machine_readable(tmp_path):
    out = train_streaming._write_streaming_phase_metadata(
        str(tmp_path),
        {
            "phase": "refinement",
            "mapping_optimizer_steps": 23,
            "refinement_optimizer_steps": 4,
        },
    )

    assert out == str(tmp_path / "streaming_phase_metadata.json")
    assert json.loads(Path(out).read_text()) == {
        "phase": "refinement",
        "mapping_optimizer_steps": 23,
        "refinement_optimizer_steps": 4,
    }
    assert not Path(out + ".tmp").exists()


@pytest.fixture(scope="module")
def trainer_source():
    return inspect.getsource(train_streaming.streaming_training)


def test_refinement_disables_ingestion_and_depth_insertion(trainer_source):
    assert "and not _phase_controller.in_refinement" in trainer_source
    insertion_gate = trainer_source.partition(
        "if is_train and _p8_allow_insert and args.streaming_insert_from_depth:"
    )[0].rsplit("\n", 3)[-3:]
    assert any("_phase_controller.in_refinement" in line for line in insertion_gate)


def test_boundary_flushes_before_saving_mapping_endpoint(trainer_source):
    _, marker, boundary = trainer_source.partition("# ---- Explicit mapping endpoint boundary")
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    flush_at = boundary.index("_flush_insertion_batch(")
    mkdir_at = boundary.index("os.makedirs(_mapping_endpoint_dir, exist_ok=True)")
    first_join_at = boundary.index("save_worker.join()")
    checkpoint_at = boundary.index("_save_streaming_checkpoint(")
    ply_at = boundary.index("_export_ply_snapshot(")
    second_join_at = boundary.index("save_worker.join()", first_join_at + 1)
    assert flush_at < mkdir_at < first_join_at < checkpoint_at < ply_at < second_join_at
    assert '"mapping_endpoint"' in boundary
    assert 'reason="mapping_endpoint"' in boundary
    assert '"streaming_resume_v2": _mapping_resume_envelope' in boundary
    assert "and not _consolidation_burst_queue" in boundary
    assert "and not _insertion_batch" in boundary


def test_phase_controller_uses_real_optimizer_clock_and_fails_incomplete_budget(
    trainer_source,
):
    assert "successful_optimization_steps=_clk_optimizer_step_id" in trainer_source
    assert "iteration > _mapping_iteration_budget" in trainer_source
    assert "mapping budget exhausted before full source disposition" in trainer_source


def test_optimizer_limit_override_is_opt_in_and_report_phase_is_optional():
    assert (
        StreamingOptimizationRuntime.__dataclass_fields__["optimizer_iteration_limit"].default
        is None
    )
    engine_source = inspect.getsource(GaussianOptimizationEngine.run_batched_step)
    assert "runtime.optimizer_iteration_limit" in engine_source
    assert (
        inspect.signature(write_post_training_report)
        .parameters["streaming_phase"]
        .default
        is None
    )


def test_mapping_endpoint_is_saved_once_before_refinement(trainer_source):
    _, marker, boundary = trainer_source.partition("# ---- Explicit mapping endpoint boundary")
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    assert "_mapping_transition is not None" in boundary
    assert "Phase.MAPPING" in boundary
    assert "Phase.REFINEMENT" in boundary
    assert "phase_timer.start(\"post_mapping_refinement\")" in boundary


def test_source_exhaustion_drains_consolidation_before_dirty_refinement_work(
    trainer_source,
):
    selection = trainer_source.split("# ---- Camera selection", 1)[1].split(
        "# PR2 cadence clocks", 1
    )[0]
    terminal_drain = selection.index("if _draining_mapping_endpoint:")
    dirty_work = selection.index("elif len(dirty_region_queue) > 0:")
    assert terminal_drain < dirty_work
    assert "streaming_scene._source_idx >= streaming_scene._frame_count" in selection


def test_exact_trainer_capture_uses_streaming_scene_holdout_accessor(trainer_source):
    capture = trainer_source.split("def _capture_exact_trainer_owner", 1)[1].split(
        "return ExactTrainerState", 1
    )[0]
    assert "streaming_scene.getTestCameras()" in capture
    assert "streaming_scene.test_cameras" not in trainer_source


def test_report_materializes_lazy_exact_resume_cameras_before_gt_access():
    source = inspect.getsource(__import__("utils.comparison_report", fromlist=["_gt_image"])._gt_image)
    assert source.index("materialize_pending_image") < source.index(
        "return score_tensor_from_camera"
    )


def test_final_report_receives_streaming_phase_metadata(trainer_source):
    kwargs_builder = trainer_source.partition("def _streaming_report_kwargs")[2]
    kwargs_builder = kwargs_builder.partition("# Async render")[0]
    assert "streaming_phase=_streaming_phase" in kwargs_builder


def test_mapping_endpoint_metadata_uses_persistent_output_paths(trainer_source):
    _, marker, boundary = trainer_source.partition("# ---- Explicit mapping endpoint boundary")
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    assert "_published_mapping_endpoint" in boundary
    assert 'getattr(args, "persist_model_path", None)' in boundary


def test_fresh_mapping_refinement_keeps_live_owners_without_exact_restore(
    trainer_source,
):
    """A fresh mapping endpoint must not restore its already-live owners."""
    _, marker, boundary = trainer_source.partition(
        "# ---- Explicit mapping endpoint boundary"
    )
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    assert "if _post_mapping_refinement_steps > 0:" in boundary
    assert "_mapping_resume_envelope = build_resume_envelope(" in boundary
    assert "_normalize_exact_refinement_owners(" not in boundary


def test_exact_resume_advances_iteration_and_rng_matches_in_process_tail(trainer_source):
    increment_at = trainer_source.index("first_iter += 1")
    loop_at = trainer_source.index("for iteration in range(first_iter")
    assert increment_at < loop_at

    _, marker, boundary = trainer_source.partition("# ---- Explicit mapping endpoint boundary")
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    report_at = boundary.index("write_post_training_report(")
    rng_reset_at = boundary.index('restore_rng_state(_mapping_resume_envelope["rng"])')
    refinement_at = boundary.index('phase_timer.start("post_mapping_refinement")')
    assert report_at < rng_reset_at < refinement_at


def test_exact_resume_uses_engine_iter_clock_and_restores_both_copies(trainer_source):
    capture = trainer_source.partition("def _capture_exact_trainer_owner")[2]
    capture = capture.partition("_exact_resume_identity")[0]
    assert "_streaming_optimization_runtime.iter_since_new_frame" in capture

    restore = trainer_source.partition("if checkpoint:")[2]
    restore = restore.partition("# Issue #1044 growth-staleness brake")[0]
    assert "_streaming_optimization_runtime.iter_since_new_frame =" in restore

    engine = trainer_source.partition("_engine_result = _mapping_application.tick()")[2]
    engine = engine.partition("else:")[0]
    assert "_iter_since_new_frame = int(" in engine
    assert "_streaming_optimization_runtime.iter_since_new_frame" in engine


def test_exact_refinement_normalizes_checkpoint_resume_and_rebuilds_lazy_observations(
    trainer_source,
):
    normalize = trainer_source.partition("def _normalize_exact_refinement_owners")[2]
    normalize = normalize.partition("if checkpoint:")[0]
    assert "load_image=False" in normalize
    assert "rebuild_exact_refinement_observations(" in normalize
    assert "restored_tier = _resolution_controller.decide(" in normalize
    assert "mapping_done=True" in normalize

    restore = trainer_source.partition("if checkpoint:")[2]
    restore = restore.partition("# Issue #1044 growth-staleness brake")[0]
    assert "restored = _normalize_exact_refinement_owners(" in restore

    assert "_phase_controller.in_refinement or (" in trainer_source

    _, marker, boundary = trainer_source.partition(
        "# ---- Explicit mapping endpoint boundary"
    )
    assert marker
    boundary = boundary.split("# ---- Saving", 1)[0]
    report_at = boundary.index("write_post_training_report(")
    sync_at = boundary.index("_persistence_worker.sync_now(")
    rng_at = boundary.index('restore_rng_state(_mapping_resume_envelope["rng"])')
    assert report_at < sync_at < rng_at


def test_dirty_region_queue_is_an_exact_owner_at_save_and_restore(trainer_source):
    assert trainer_source.count('"dirty_regions": dirty_region_queue') == 4


def test_causal_fork_boundary_is_quiescent_atomic_and_skips_periodic_overwrite(trainer_source):
    _, marker, tail = trainer_source.partition("# ---- Exact causal fork boundary")
    assert marker
    boundary = tail.split("# ---- Explicit mapping endpoint boundary", 1)[0]
    assert "iteration == _exact_fork_iteration" in boundary
    report_join = boundary.index("report_worker.join()")
    save_join = boundary.index("save_worker.join()")
    flush = boundary.index("_flush_insertion_batch(")
    envelope = boundary.index("build_resume_envelope(")
    checkpoint = boundary.index("_save_streaming_checkpoint(")
    final_join = boundary.index("save_worker.join()", save_join + 1)
    assert report_join < save_join < flush < envelope < checkpoint < final_join
    assert 'reason="causal_fork"' in boundary
    assert '"frame_scheduler": scheduler' in boundary
    assert '"admission": _keyframe_admitter' in boundary
    assert '"scene": ExactCausalForkSceneState(' in boundary
    assert '"frame_observations": ExactCausalForkObservationDBState(' in boundary
    capture = trainer_source.partition(
        "def _capture_exact_causal_fork_trainer_owner"
    )[2].partition("_exact_resume_identity")[0]
    assert "consolidation_queue_source_indices" in capture
    assert "fork_boundary_flush" in boundary
    assert "_skip_periodic_checkpoint_iteration = iteration" in boundary


def test_causal_fork_runtime_audits_every_concurrent_queue(trainer_source):
    runtime = trainer_source.partition("def _exact_refinement_runtime")[2]
    runtime = runtime.partition("def _normalize_exact_refinement_owners")[0]
    for token in (
        "_ingest_worker is not None",
        "_deferred_slam_tracker is not None",
        "_coordinator.bounded_pipeline",
        "streaming_multires_low_until_iteration",
        "save_worker",
        "report_worker",
        "_map_output_lane",
        "_pv_cpu_queue",
        "_insertion_batch",
        "_consolidation_burst_queue",
        "streaming_depth_temporal_check",
        "streaming_depth_temporal_require_for_sensor_closer",
        "streaming_depth_temporal_require_for_insert",
        "streaming_depth_temporal_require_for_seed",
        "streaming_bootstrap_seed_decay_frames",
        "assimilation_control_config",
        "_partition_focus_enabled",
        "mapping_scheduler_tracking_confidence_enabled",
        "streaming_tracking_confidence_enabled",
        "_backpressure_owner is not None",
        "_batch_frames > 1 or _batch_steps > 0",
        "is_dataset_pose_source(_pose_source_name)",
    ):
        assert token in runtime


def test_causal_fork_resume_activates_treatment_at_checkpoint_iteration(trainer_source):
    restore = trainer_source.partition("if checkpoint:")[2]
    activation = restore.index("activate_causal_fork_treatment(")
    rng = restore.index("restore_rng_state(", activation)
    assert activation < rng
    activation_call = restore[activation:rng]
    assert "causal_fork_contract=_ckpt_streaming_meta.get(\"causal_fork\")" in activation_call
    assert "checkpoint_iteration=first_iter" not in activation_call


def test_causal_fork_restores_one_shared_observation_db_identity(trainer_source):
    normalize = trainer_source.partition("def _normalize_exact_causal_fork_owners")[2]
    normalize = normalize.partition("if checkpoint:")[0]
    assert "FrameObservationDB(" not in normalize
    assert "_keyframe_admitter._observation_db is not _observation_db" in normalize
    restore = trainer_source.partition("if checkpoint:")[2]
    restore = restore.partition("elif _exact_resume_verdict.eligible:")[0]
    assert '"frame_observations": ExactCausalForkObservationDBState(' in restore
