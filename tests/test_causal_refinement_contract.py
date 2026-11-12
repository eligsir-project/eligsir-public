from __future__ import annotations

import json
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eligsir.runtime.causal_entry import run_causal_post_mapping_refinement
from eligsir.runtime import causal_entry
from eligsir.runtime.causal_entry import run_causal_mapping
from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
from eligsir.runtime.causal_replay import CausalReplayClock
from eligsir.runtime.artifact_paths import (
    point_cloud_artifact_paths,
    publish_legacy_iteration_alias,
)
from eligsir.utils.comparison_report import write_post_training_report


def _disabled_historical_usefulness_telemetry() -> dict[str, int]:
    """Match the canonical disabled challenger telemetry surface."""

    return {
        "observations_total": 0,
        "trusted_updates_total": 0,
        "feedback_busy": 0,
        "replay_challenger_trials_completed": 0,
        "replay_challenger_trials_authorized": 0,
        "replay_challenger_demotions": 0,
        "replay_challenger_turnover_blocked": 0,
        "replay_challenger_no_ready": 0,
        "ready_replay_evicted_bytes": 0,
    }


def _write_clock_qualified_point_cloud(root: Path, runtime: object) -> None:
    work = runtime._work_clock_owner.snapshot()
    paths = point_cloud_artifact_paths(
        root,
        optimizer_updates=int(work.optimizer_updates),
        supervised_views=int(work.supervised_rgb_views),
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.point_cloud.write_bytes(b"ply\n")
    publish_legacy_iteration_alias(paths)


def _install_test_work_clock(runtime: object) -> None:
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            optimizer_updates=int(runtime.iteration),
            supervised_rgb_views=int(runtime.iteration) * 4,
        )
    )


class Runtime:
    optimizer_steps = 4
    iteration = 20

    def __init__(self):
        Runtime.optimizer_steps = 4
        Runtime.iteration = 20
        self.requested_updates = []
        self.backend = SimpleNamespace(
            checkpoint_state=lambda: SimpleNamespace(
                layout="gsplat",
                payload={"means": torch.tensor([[1.0, 2.0, 3.0]])},
                map_version=11,
                topology_version=7,
                coordinate_frame_version=3,
                schema_version=2,
            )
        )
        self.refinement_endpoints = []
        _install_test_work_clock(self)

    def enter_post_mapping_refinement(self, mapping_endpoint):
        assert Path(mapping_endpoint["point_cloud"]).is_file()
        assert Path(mapping_endpoint["checkpoint"]).is_file()
        self.refinement_endpoints.append(dict(mapping_endpoint))

    def rearm_post_mapping_work(
        self, *, optimizer_updates=None, refinement_progress_fraction=None
    ):
        self.requested_updates.append(optimizer_updates)

    def historical_usefulness_telemetry(self) -> dict[str, int]:
        return _disabled_historical_usefulness_telemetry()


class Application:
    def __init__(self, runtime, root):
        self.runtime = runtime
        self.root = root
        self.calls = 0
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1
        _write_clock_qualified_point_cloud(self.root, self.runtime)

    def tick(self):
        self.calls += 1
        Runtime.optimizer_steps += 1
        Runtime.iteration += 1
        return SimpleNamespace(optimizer_steps=1, supervision_views=4)


class _PreparedReadyApplication(Application):
    """One event-gated READY poll precedes the real refinement transaction."""

    def tick(self):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                optimizer_steps=0,
                supervision_views=0,
                retryable=True,
            )
        Runtime.optimizer_steps += 1
        Runtime.iteration += 1
        return SimpleNamespace(optimizer_steps=1, supervision_views=4)


def test_causal_refinement_scopes_runtime_loss_context_to_tail_ticks(tmp_path):
    """The causal driver cannot leak a tail-only loss override into flush/state."""

    runtime = Runtime()
    weight = {"current": 0.2}
    events = []

    @contextmanager
    def tail_loss_context():
        events.append(("enter", weight["current"]))
        weight["current"] = 0.0
        try:
            yield
        finally:
            weight["current"] = 0.2
            events.append(("exit", weight["current"]))

    runtime.post_mapping_refinement_loss_context = tail_loss_context
    app = Application(runtime, tmp_path)
    original_tick = app.tick
    original_flush = app.flush

    def tick():
        assert weight["current"] == 0.0
        return original_tick()

    def flush():
        assert weight["current"] == 0.2
        original_flush()

    app.tick = tick
    app.flush = flush
    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=2,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
    )

    assert events == [
        ("enter", 0.2),
        ("exit", 0.2),
        ("enter", 0.2),
        ("exit", 0.2),
    ]
    assert runtime.requested_updates == [2, 1]
    assert metadata["source_frame_count"] == 8
    assert metadata["refinement_optimizer_steps"] == 2


def test_causal_refinement_writes_durable_endpoints_and_phase_metadata(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)
    metadata = run_causal_post_mapping_refinement(
        app, runtime, source_frame_count=8, requested_steps=3,
        model_path=tmp_path, mapping_wall_time_s=1.25,
    )

    assert app.calls == 3
    assert runtime.requested_updates == [3, 2, 1]
    assert metadata["mapping_endpoint"]["point_cloud"]
    assert metadata["refined_endpoint"]["point_cloud"]
    assert metadata["mapping_wall_time_s"] == 1.25
    assert metadata["refinement_wall_time_s"] >= 0.0
    assert metadata["refinement_supervision_views"] == 12
    assert metadata["refinement_mapping_active_s"] > 0.0
    assert metadata["refinement_supervision_views_per_active_s"] > 0.0
    assert metadata["refinement_supervision_views_per_wall_s"] > 0.0
    assert json.loads((tmp_path / "streaming_phase_metadata.json").read_text()) == metadata
    assert runtime.refinement_endpoints == [metadata["mapping_endpoint"]]


def test_causal_refinement_records_opt_in_allocator_handoff_before_tail_ticks(
    tmp_path, monkeypatch
):
    """The causal boundary records a current sample before refinement mutates the map.

    Removing the environment gate, placing the sample after ``tick()``, or
    omitting the phase metadata must break this production handoff contract.
    """
    from eligsir.runtime import causal_entry

    events = []

    class _Tracker:
        def phase_snapshot(self, phase, **kwargs):
            events.append(("sample", phase, kwargs["iteration"]))
            return {"phase": phase, "status": "measured", "current_allocated_bytes": 100}

    runtime = Runtime()
    app = Application(runtime, tmp_path)
    original_tick = app.tick

    def tick():
        events.append(("tick", runtime.iteration))
        return original_tick()

    app.tick = tick
    monkeypatch.setenv("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "1")
    metadata = causal_entry.run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=1,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        allocator_phase_tracker=_Tracker(),
    )

    assert events == [("sample", "pre_refinement_handoff", 20), ("tick", 20)]
    assert metadata["allocator_phase_samples"] == {
        "enabled": True,
        "available": True,
        "reason": None,
        "samples": {
            "pre_refinement_handoff": {
                "phase": "pre_refinement_handoff",
                "status": "measured",
                "current_allocated_bytes": 100,
            }
        },
    }


def test_causal_refinement_marks_failed_allocator_handoff_unavailable(tmp_path, monkeypatch):
    from eligsir.runtime import causal_entry

    class _Tracker:
        def phase_snapshot(self, *_args, **_kwargs):
            raise RuntimeError("snapshot unavailable")

    monkeypatch.setenv("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "1")
    metadata = causal_entry.run_causal_post_mapping_refinement(
        Application(Runtime(), tmp_path), Runtime(), source_frame_count=8,
        requested_steps=1, model_path=tmp_path, mapping_wall_time_s=1.25,
        allocator_phase_tracker=_Tracker(),
    )

    phases = metadata["allocator_phase_samples"]
    assert phases["available"] is False
    assert phases["reason"] == "RuntimeError: snapshot unavailable"


def test_causal_refinement_retries_event_gated_prepared_ready_work(tmp_path):
    runtime = Runtime()
    app = _PreparedReadyApplication(runtime, tmp_path)

    metadata = run_causal_post_mapping_refinement(
        app, runtime, source_frame_count=8, requested_steps=1,
        model_path=tmp_path, mapping_wall_time_s=1.25,
    )

    assert app.calls == 2
    assert runtime.requested_updates == [1]
    assert metadata["refinement_optimizer_steps"] == 1
    assert metadata["refinement_supervision_views"] == 4
    assert metadata["refinement_mapping_active_s"] > 0.0


def test_causal_refinement_observes_retryable_prepared_tick_only_on_retry_sink(tmp_path):
    runtime = Runtime()
    app = _PreparedReadyApplication(runtime, tmp_path)
    completed = []
    retries = []

    run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=1,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        tick_observer=lambda result: completed.append(result.optimizer_steps),
        retry_tick_observer=lambda result: retries.append(result.optimizer_steps),
    )

    assert retries == [0]
    assert completed == [1]


def test_causal_mapping_endpoint_is_complete_before_refinement_ticks(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)
    events = []

    def publish_report(endpoint):
        # Baseline/phase-relative reporting reads this file.  Publishing the
        # report before the durable mapping boundary exists makes every fresh
        # causal comparison fail closed as UNAVAILABLE.
        boundary_metadata = json.loads(
            (tmp_path / "streaming_phase_metadata.json").read_text()
        )
        assert boundary_metadata["mapping_endpoint"]["iteration"] == 20
        events.append(("report", endpoint["iteration"]))
        comparison = tmp_path / "comparison" / "mapping_endpoint"
        comparison.mkdir(parents=True)
        for name in (
            "report.json",
            "test_contact_sheet.png",
            "normals_contact_sheet.png",
            "trajectory_psnr_by_distance.svg",
        ):
            (comparison / name).write_bytes(b"artifact")
        return {
            "report": str(comparison / "report.json"),
            "test_contact_sheet": str(comparison / "test_contact_sheet.png"),
            "normals_contact_sheet": str(comparison / "normals_contact_sheet.png"),
            "trajectory_psnr_by_distance": str(
                comparison / "trajectory_psnr_by_distance.svg"
            ),
        }

    original_tick = app.tick

    def ordered_tick():
        boundary_metadata = json.loads(
            (tmp_path / "streaming_phase_metadata.json").read_text()
        )
        assert boundary_metadata["phase"] == "refinement"
        assert boundary_metadata["refinement_optimizer_steps"] == 0
        assert boundary_metadata["mapping_endpoint"]["iteration"] == 20
        events.append(("tick", runtime.iteration))
        return original_tick()

    app.tick = ordered_tick
    metadata = run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=1,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        mapping_endpoint_reporter=publish_report,
    )

    endpoint = metadata["mapping_endpoint"]
    assert events[0] == ("report", 20)
    assert events[1][0] == "tick"
    assert Path(endpoint["point_cloud"]).is_file()
    assert Path(endpoint["checkpoint"]).is_file()
    assert Path(endpoint["report"]).is_file()
    assert Path(endpoint["test_contact_sheet"]).is_file()
    assert Path(endpoint["normals_contact_sheet"]).is_file()
    assert Path(endpoint["trajectory_psnr_by_distance"]).is_file()
    payload, iteration, checkpoint_metadata = torch.load(
        endpoint["checkpoint"], weights_only=False
    )
    assert iteration == 20
    assert payload["means"].shape == (1, 3)
    assert checkpoint_metadata == {
        "checkpoint_reason": "mapping_endpoint",
        "coordinate_frame_version": 3,
        "layout": "gsplat",
        "map_version": 11,
        "schema_version": 2,
        "source_frame_count": 8,
        "topology_version": 7,
    }


def test_causal_mapping_endpoint_fails_closed_on_incomplete_report_bundle(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)

    def incomplete_report(_endpoint):
        report = tmp_path / "comparison" / "mapping_endpoint" / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}")
        return {"report": str(report)}

    try:
        run_causal_post_mapping_refinement(
            app,
            runtime,
            source_frame_count=8,
            requested_steps=1,
            model_path=tmp_path,
            mapping_wall_time_s=1.25,
            mapping_endpoint_reporter=incomplete_report,
        )
    except RuntimeError as exc:
        assert "mapping endpoint artifact" in str(exc)
    else:
        raise AssertionError("refinement started without the required render bundle")
    assert app.calls == 0


def test_causal_runner_publishes_trajectory_and_mapping_report_before_refinement():
    source = inspect.getsource(run_causal_mapping)
    trajectory_at = source.index("write_causal_trajectory(")
    refinement_at = source.index("run_causal_post_mapping_refinement(")
    final_report_at = source.rindex("evaluation_observer.finalize(summary)")

    assert trajectory_at < refinement_at < final_report_at
    assert "mapping_endpoint_reporter=" in source
    assert 'subdir="mapping_endpoint"' in source


def test_causal_runner_wires_evaluation_to_exact_mapping_active_clock():
    source = inspect.getsource(run_causal_mapping)

    assert "tick_clock_observer=" in source
    assert "evaluation_observer.observe_tick" in source


class FailingApplication(Application):
    """Fails a specific tick, mimicking a real CUDA OOM during backward.

    Issue #1306: the failed tick raises BEFORE the runtime's counters would
    have advanced -- matching production, where `self.iteration` is only
    committed after `execute_fixed_topology_transaction` returns
    successfully (see production_application.py, PR #1305).
    """

    def __init__(self, runtime, root, *, fail_at_call: int):
        super().__init__(runtime, root)
        self._fail_at_call = fail_at_call

    def tick(self):
        self.calls += 1
        if self.calls == self._fail_at_call:
            raise RuntimeError("simulated CUDA OOM during backward")
        Runtime.optimizer_steps += 1
        Runtime.iteration += 1


def test_causal_refinement_writes_periodic_checkpoints_issue1306(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)
    metadata = run_causal_post_mapping_refinement(
        app, runtime, source_frame_count=8, requested_steps=6,
        model_path=tmp_path, mapping_wall_time_s=1.0,
        checkpoint_interval_steps=2,
    )

    assert metadata["phase"] == "complete"
    checkpoint_dir = tmp_path / "refinement_checkpoint"
    assert checkpoint_dir.is_dir()
    # Only the most recent periodic boundary is kept -- not one per interval.
    checkpoints = sorted(checkpoint_dir.glob("chkpnt*.pth"))
    assert len(checkpoints) == 1
    payload, iteration, checkpoint_metadata = torch.load(
        checkpoints[0], weights_only=False
    )
    assert checkpoint_metadata["checkpoint_reason"] == "periodic_refinement"
    # Last periodic checkpoint fires at completed=4 (interval=2, requested=6);
    # the final completed=6 boundary is the ordinary post-loop refined_endpoint,
    # not a periodic write (guarded by `completed < requested`).
    assert iteration == Runtime.iteration - 2
    assert metadata["last_refinement_checkpoint"]["checkpoint"] == str(
        checkpoints[0]
    )


def test_causal_refinement_periodic_checkpoint_disabled_by_default(tmp_path):
    runtime = Runtime()
    app = Application(runtime, tmp_path)
    metadata = run_causal_post_mapping_refinement(
        app, runtime, source_frame_count=8, requested_steps=3,
        model_path=tmp_path, mapping_wall_time_s=1.0,
    )

    assert not (tmp_path / "refinement_checkpoint").exists()
    assert "last_refinement_checkpoint" not in metadata


def test_causal_refinement_emergency_checkpoint_on_tick_exception_issue1306(tmp_path):
    runtime = Runtime()
    app = FailingApplication(runtime, tmp_path, fail_at_call=3)

    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        run_causal_post_mapping_refinement(
            app, runtime, source_frame_count=8, requested_steps=10,
            model_path=tmp_path, mapping_wall_time_s=1.0,
        )

    checkpoint_dir = tmp_path / "refinement_checkpoint"
    assert checkpoint_dir.is_dir()
    checkpoints = list(checkpoint_dir.glob("chkpnt*.pth"))
    assert len(checkpoints) == 1
    payload, iteration, checkpoint_metadata = torch.load(
        checkpoints[0], weights_only=False
    )
    assert checkpoint_metadata["checkpoint_reason"] == "emergency_RuntimeError"
    assert checkpoint_metadata["point_cloud_snapshot"] is False
    # The two ticks completed before the failure (calls 1 and 2) are what
    # survives -- not zero, and not the full requested_steps.
    assert iteration == 22
    # Issue #1306: the emergency path must NOT call application.flush()
    # (which would render/save a fresh point cloud and could itself OOM
    # immediately after the tick that just failed for the same reason) --
    # only the initial mapping_endpoint flush should have happened.
    assert app.flush_calls == 1
    assert not (checkpoint_dir / "point_cloud.ply").exists()


def test_causal_refinement_emergency_checkpoint_failure_does_not_mask_original_error(
    tmp_path, monkeypatch
):
    runtime = Runtime()
    app = FailingApplication(runtime, tmp_path, fail_at_call=2)

    import eligsir.runtime.causal_entry as causal_entry

    def _always_fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        causal_entry, "_write_emergency_backend_checkpoint", _always_fail
    )

    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        run_causal_post_mapping_refinement(
            app, runtime, source_frame_count=8, requested_steps=10,
            model_path=tmp_path, mapping_wall_time_s=1.0,
        )


def test_paced_causal_refinement_continues_periodic_evaluation_after_eof():
    source = inspect.getsource(run_causal_mapping)

    assert "def refinement_tick_clock_observer" in source
    assert "evaluation_observer.observe_tick(tick, clock)" in source
    assert "tick_clock_observer=(" in source
    assert "refinement_tick_clock_observer" in source


def test_causal_runner_uses_the_continuous_driver_for_paced_release(
    tmp_path, monkeypatch
):
    import eligsir.runtime.causal_entry as causal_entry
    from eligsir.runtime.causal_replay import CausalReplaySummary

    captured = {}

    class _Driver:
        def __init__(self, _application, _adapter, **kwargs):
            captured.update(kwargs)

        def run(self, records):
            captured["records"] = records
            return CausalReplaySummary(
                submitted=0,
                ticks=1,
                source_identities=(),
            )

    source = SimpleNamespace(frames=())
    config = SimpleNamespace(hard_limits={"optimization_batch_size": 4})
    binding = SimpleNamespace(
        mapper_sink=lambda _frame, _tracking: None,
        optimizer_tick=lambda: None,
        materialization_policy=None,
        materialization_backend=None,
        source_complete_hook=None,
        flush_hook=None,
        close_hook=None,
        observation_owner=None,
        control_snapshot_provider=None,
        disposition_sink=None,
        source_consumed_observer=None,
    )
    class _Runtime:
        historical_usefulness_telemetry = staticmethod(
            _disabled_historical_usefulness_telemetry
        )

        def set_source_input_exhausted_provider(self, provider):
            captured["input_exhausted_provider"] = provider

        def cache_causal_bypass_packet(self, _packet, *, role):
            captured["bypass_role"] = role

    runtime = _Runtime()

    monkeypatch.setattr(
        causal_entry,
        "resolve_causal_mapping_composition",
        lambda **_kwargs: SimpleNamespace(
            source=source,
            config=config,
            binding=binding,
            runtime=runtime,
            prepared_ready_path=None,
        ),
    )
    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", _Driver)
    monkeypatch.setattr(causal_entry, "write_causal_trajectory", lambda *_args: None)
    monkeypatch.setattr(
        causal_entry,
        "run_causal_post_mapping_refinement",
        lambda *_args, **_kwargs: None,
    )
    import eligsir.utils.execution_semantics as execution_semantics

    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )
    dataset = SimpleNamespace(
        source_path="dataset",
        model_path=str(tmp_path),
    )
    args = SimpleNamespace(
        streaming_input_fps_cap=30.0,
        causal_source_buffer_frames=16,
        causal_progress_interval_s=0.0,
        causal_decision_trace="",
        streaming_eval_hold=0,
        streaming_post_mapping_refinement_steps=0,
    )

    causal_entry.run_causal_mapping(
        dataset=dataset,
        opt=SimpleNamespace(iterations=99_999),
        pipe=SimpleNamespace(),
        args=args,
    )

    assert captured["source_hz"] == 30.0
    assert captured["bypass_packet_capture"].__self__ is runtime
    assert captured["records"] == ()
    assert captured["records"] is not source
    assert "minimum_optimizer_steps" not in captured
    ingress = captured["ingress_telemetry"]
    assert captured["input_exhausted_provider"]() is False
    ingress.set_input_exhausted(True)
    assert captured["input_exhausted_provider"]() is True


def test_paced_causal_runner_rejects_iterator_catalogue_without_acquisition(
    tmp_path, monkeypatch
):
    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.utils.execution_semantics as execution_semantics

    class _Driver:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("paced acquisition must not start")

    class _UnsupportedCatalogue:
        def __iter__(self):
            raise AssertionError("must not acquire source records")

    source = SimpleNamespace(frames=_UnsupportedCatalogue())

    binding = SimpleNamespace(
        mapper_sink=lambda _frame, _tracking: None,
        optimizer_tick=lambda: None,
        materialization_policy=None,
        materialization_backend=None,
        source_complete_hook=None,
        flush_hook=None,
        close_hook=None,
        observation_owner=None,
        control_snapshot_provider=None,
        disposition_sink=None,
        source_consumed_observer=None,
    )
    monkeypatch.setattr(
        causal_entry,
        "resolve_causal_mapping_composition",
        lambda **_kwargs: SimpleNamespace(
            source=source,
            config=SimpleNamespace(hard_limits={"optimization_batch_size": 4}),
            binding=binding,
            runtime=SimpleNamespace(
                historical_usefulness_telemetry=_disabled_historical_usefulness_telemetry,
            ),
            prepared_ready_path=None,
        ),
    )
    monkeypatch.setattr(causal_entry, "build_causal_viewer", lambda *_args: (None, None, None))
    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", _Driver)
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )

    with pytest.raises(TypeError, match="requires a finite source catalogue"):
        causal_entry.run_causal_mapping(
            dataset=SimpleNamespace(source_path="dataset", model_path=str(tmp_path)),
            opt=SimpleNamespace(iterations=99_999),
            pipe=SimpleNamespace(),
            args=SimpleNamespace(
                streaming_input_fps_cap=30.0,
                causal_source_buffer_frames=16,
                causal_progress_interval_s=0.0,
                causal_decision_trace="",
                streaming_eval_hold=0,
                streaming_post_mapping_refinement_steps=0,
            ),
        )


def test_prepared_causal_runner_rejects_iter_based_before_composition(monkeypatch):
    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.utils.execution_semantics as execution_semantics

    composed = False

    def resolve_composition(**_kwargs):
        nonlocal composed
        composed = True
        raise AssertionError("prepared composition must not start")

    monkeypatch.setattr(
        causal_entry, "resolve_causal_mapping_composition", resolve_composition
    )
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": "iter_based"},
    )
    args = SimpleNamespace(
        causal_prepared_ready_b4=True,
        streaming_input_fps_cap=30.0,
    )

    with pytest.raises(ValueError, match="causal_prepared_ready_b4 requires"):
        causal_entry.run_causal_mapping(
            dataset=SimpleNamespace(),
            opt=SimpleNamespace(),
            pipe=SimpleNamespace(),
            args=args,
        )

    assert not composed


def test_causal_final_report_keeps_wall_and_mapping_active_clocks_distinct():
    source = inspect.getsource(CausalEvaluationObserver.finalize)

    assert "self._elapsed_s = float(summary.elapsed_s)" in source
    assert "self._mapping_active_s = float(summary.mapping_active_s)" in source
    assert "_terminal_evaluation_tick_clock_snapshot(self)" in source


def test_causal_evaluator_and_report_support_phase_specific_quality_bundle():
    finalize = inspect.signature(CausalEvaluationObserver.finalize)
    assert finalize.parameters["subdir"].default is None
    assert "write_trajectory_quality_profile" in inspect.getsource(
        write_post_training_report
    )


class _RearmingRuntime:
    optimizer_steps = 4
    iteration = 20

    def __init__(self):
        self.optimizer_steps = 4
        self.iteration = 20
        self.rearms = 0
        self.requested_updates = []
        self.admitted = 8
        self.clock_events = 8
        self.backend = Runtime().backend
        _install_test_work_clock(self)

    def rearm_post_mapping_work(
        self, *, optimizer_updates=None, refinement_progress_fraction=None
    ):
        self.rearms += 1
        self.requested_updates.append(optimizer_updates)

    def enter_post_mapping_refinement(self, mapping_endpoint):
        assert Path(mapping_endpoint["point_cloud"]).is_file()
        assert Path(mapping_endpoint["checkpoint"]).is_file()


class _RearmingApplication:
    def __init__(self, runtime, root):
        self.runtime = runtime
        self.root = root
        self.calls = 0

    def flush(self):
        _write_clock_qualified_point_cloud(self.root, self.runtime)

    def tick(self):
        self.calls += 1
        if self.runtime.rearms:
            self.runtime.optimizer_steps += 1
            self.runtime.iteration += 1
        return SimpleNamespace(
            optimizer_steps=1,
            supervision_views=4,
            iteration=self.runtime.iteration,
        )


def test_causal_refinement_rearms_last_disposition_for_real_steps(tmp_path):
    runtime = _RearmingRuntime()
    app = _RearmingApplication(runtime, tmp_path)
    metadata = run_causal_post_mapping_refinement(
        app, runtime, source_frame_count=8, requested_steps=3,
        model_path=tmp_path, mapping_wall_time_s=1.25,
    )

    assert app.calls == 3
    assert runtime.rearms == 3
    assert runtime.requested_updates == [3, 2, 1]
    assert runtime.admitted == 8
    assert runtime.clock_events == 8
    assert metadata["refinement_optimizer_steps"] == 3


def test_causal_refinement_publishes_every_tail_tick_to_observer(tmp_path):
    runtime = _RearmingRuntime()
    app = _RearmingApplication(runtime, tmp_path)
    observed_iterations = []

    run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=3,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        tick_observer=lambda result: observed_iterations.append(result.iteration),
    )

    assert observed_iterations == [21, 22, 23]


def test_causal_refinement_publishes_evaluation_excluding_work_clocks(tmp_path):
    runtime = _RearmingRuntime()
    app = _RearmingApplication(runtime, tmp_path)
    clocks = []

    run_causal_post_mapping_refinement(
        app,
        runtime,
        source_frame_count=8,
        requested_steps=3,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        tick_clock_observer=lambda _result, clock: clocks.append(clock),
        initial_clock=CausalReplayClock(
            optimizer_steps=4,
            supervision_views=16,
            mapping_active_s=1.5,
            elapsed_s=2.0,
            source_frame_index=400,
        ),
    )

    assert [clock.optimizer_steps for clock in clocks] == [5, 6, 7]
    assert [clock.supervision_views for clock in clocks] == [20, 24, 28]
    assert all(clock.mapping_active_s >= 1.5 for clock in clocks)
    assert all(clock.elapsed_s >= 2.0 for clock in clocks)
    assert clocks[-1].mapping_active_s >= clocks[0].mapping_active_s
    assert [clock.source_frame_index for clock in clocks] == [400, 400, 400]

    unknown_clocks = []
    unknown_runtime = _RearmingRuntime()
    run_causal_post_mapping_refinement(
        _RearmingApplication(unknown_runtime, tmp_path),
        unknown_runtime,
        source_frame_count=8,
        requested_steps=1,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        tick_clock_observer=lambda _result, clock: unknown_clocks.append(clock),
        initial_clock=CausalReplayClock(
            optimizer_steps=4,
            supervision_views=16,
            mapping_active_s=1.5,
            elapsed_s=2.0,
        ),
    )
    assert [clock.source_frame_index for clock in unknown_clocks] == [None]


def test_control_probe_candidate_rejects_global_gain_with_partition_loss():
    assert not causal_entry._control_probe_candidate_is_accepted(
        {0: 19.0, 1: 14.0, 2: 13.0},
        {0: 20.0, 1: 13.9, 2: 13.4},
    )


def test_control_probe_candidate_accepts_preserved_partitions_and_closed_gap():
    assert causal_entry._control_probe_candidate_is_accepted(
        {0: 19.0, 1: 14.0, 2: 13.0},
        {0: 19.1, 1: 14.2, 2: 13.2},
    )


def test_control_probe_candidate_accepts_uniform_improvement_after_gap_closes():
    assert causal_entry._control_probe_candidate_is_accepted(
        {0: 19.0, 1: 19.0},
        {0: 19.1, 1: 19.1},
    )


def test_refinement_rejects_candidate_chunk_before_observer_publication(tmp_path):
    runtime = Runtime()

    class QualityApplication(Application):
        def tick(self):
            self.calls += 1
            runtime.optimizer_steps += 1
            runtime.iteration += 1
            return SimpleNamespace(
                optimizer_steps=1,
                supervision_views=4,
                iteration=runtime.iteration,
            )

    restored = []
    published = []
    accepted_control_publications = []

    class Branch:
        def checkpoint_state(self):
            return int(runtime.optimizer_steps), int(runtime.iteration)

        def restore_state(self, state):
            restored.append(state)
            runtime.optimizer_steps, runtime.iteration = state

        def restore_live_state(self, state):
            self.restore_state(state)

        def evaluate_fixed_views(self, *, verify_read_only):
            assert verify_read_only
            if runtime.optimizer_steps == 4:
                return {"0": 19.0, "1": 14.0}
            if not restored:
                return {"0": 20.0, "1": 13.9}
            return {"0": 19.2, "1": 14.4}

    metadata = run_causal_post_mapping_refinement(
        QualityApplication(runtime, tmp_path),
        runtime,
        source_frame_count=8,
        requested_steps=2,
        model_path=tmp_path,
        mapping_wall_time_s=1.25,
        tick_observer=lambda result: published.append(result.iteration),
        branch_runtime=Branch(),
        control_probe_chunk_steps=2,
        on_control_chunk_accepted=lambda: accepted_control_publications.append(
            runtime.iteration
        ),
    )

    assert restored == [(4, 20)]
    assert published == [22]
    assert accepted_control_publications == [22]
    assert metadata["refinement_optimizer_steps"] == 2
    assert metadata["refinement_rejected_control_chunks"] == 1


def test_refinement_restores_candidate_chunk_when_control_probe_raises(tmp_path):
    runtime = Runtime()

    class QualityApplication(Application):
        def tick(self):
            runtime.optimizer_steps += 1
            runtime.iteration += 1
            return SimpleNamespace(optimizer_steps=1, supervision_views=4)

    restored = []

    class Branch:
        def checkpoint_state(self):
            return int(runtime.optimizer_steps), int(runtime.iteration)

        def restore_state(self, state):
            restored.append(state)
            runtime.optimizer_steps, runtime.iteration = state

        def restore_live_state(self, state):
            self.restore_state(state)

        def evaluate_fixed_views(self, *, verify_read_only):
            assert verify_read_only
            if runtime.optimizer_steps == 4:
                return {"0": 19.0, "1": 14.0}
            raise RuntimeError("probe render failed")

    with pytest.raises(RuntimeError, match="probe render failed"):
        run_causal_post_mapping_refinement(
            QualityApplication(runtime, tmp_path),
            runtime,
            source_frame_count=8,
            requested_steps=1,
            model_path=tmp_path,
            mapping_wall_time_s=1.25,
            branch_runtime=Branch(),
            control_probe_chunk_steps=1,
        )

    assert restored == [(4, 20)]


def test_refinement_stops_after_bounded_rejected_control_chunks(tmp_path):
    runtime = Runtime()
    restored = []

    class QualityApplication(Application):
        def tick(self):
            runtime.optimizer_steps += 1
            runtime.iteration += 1
            return SimpleNamespace(optimizer_steps=1, supervision_views=4)

    class Branch:
        def checkpoint_state(self):
            return int(runtime.optimizer_steps), int(runtime.iteration)

        def restore_state(self, state):
            restored.append(state)
            runtime.optimizer_steps, runtime.iteration = state

        def restore_live_state(self, state):
            self.restore_state(state)

        def evaluate_fixed_views(self, *, verify_read_only):
            assert verify_read_only
            return {"0": 19.0, "1": 14.0}

    with pytest.raises(RuntimeError, match="control-probe acceptance rejected 2 chunks"):
        run_causal_post_mapping_refinement(
            QualityApplication(runtime, tmp_path),
            runtime,
            source_frame_count=8,
            requested_steps=1,
            model_path=tmp_path,
            mapping_wall_time_s=1.25,
            branch_runtime=Branch(),
            control_probe_chunk_steps=1,
            control_probe_max_rejections=2,
        )

    assert restored == [(4, 20), (4, 20)]
    metadata = json.loads((tmp_path / "streaming_phase_metadata.json").read_text())
    assert metadata["refinement_stop_reason"] == "control_probe_rejected"


def test_refinement_aborts_flush_when_rejected_chunk_cannot_be_restored(tmp_path):
    runtime = Runtime()
    aborted = []

    class QualityApplication(Application):
        def tick(self):
            runtime.optimizer_steps += 1
            runtime.iteration += 1
            return SimpleNamespace(optimizer_steps=1, supervision_views=4)

        def abort_unaccepted_refinement(self):
            aborted.append(True)

    class Branch:
        def checkpoint_state(self):
            return int(runtime.optimizer_steps), int(runtime.iteration)

        def restore_state(self, _state):
            raise RuntimeError("backend restore failed")

        def restore_live_state(self, state):
            self.restore_state(state)

        def evaluate_fixed_views(self, *, verify_read_only):
            assert verify_read_only
            return {"0": 19.0, "1": 14.0}

    with pytest.raises(RuntimeError, match="accepted-map branch restore failed"):
        run_causal_post_mapping_refinement(
            QualityApplication(runtime, tmp_path),
            runtime,
            source_frame_count=8,
            requested_steps=1,
            model_path=tmp_path,
            mapping_wall_time_s=1.25,
            branch_runtime=Branch(),
            control_probe_chunk_steps=1,
        )

    assert aborted == [True]
