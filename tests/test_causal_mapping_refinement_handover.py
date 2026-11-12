"""Issue #1653: mapped-camera plotting must not sit in the mapping->
refinement handover, and mapping_wall_time_s must be frozen before any
diagnostic/report work (including that plotting) can contaminate it.

These tests complement tests/test_causal_trajectory_artifact.py (issue
#1687's report-camera coverage, left unchanged by this fix) and
tests/test_causal_refinement_contract.py (the mapping->refinement handover
contract itself).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eligsir.runtime.causal_entry import (
    _stream_jsonl_rows,
    write_mapped_camera_trajectory_plot,
)
from eligsir.utils.trajectory_eval import mapped_camera_trajectory_selection


def _source(ordinal: int):
    from eligsir.contracts.runtime import SourceIdentity

    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)


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


def _causal_mapping_args(tmp_path, **overrides):
    values = {
        "source_path": "/data/fr3",
        "model_path": str(tmp_path),
        "streaming_input_fps_cap": 30.0,
        "causal_source_buffer_frames": 16,
        "causal_progress_interval_s": 0.0,
        "causal_decision_trace": "",
        "streaming_eval_hold": 0,
        "streaming_eval_guard": 0,
        "streaming_post_mapping_refinement_steps": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _wire_minimal_causal_mapping(monkeypatch, tmp_path, *, successful_supervision_uses):
    """Mirror tests/test_causal_partition_control_wiring.py's minimal fixture.

    Returns the ``events`` list that call-order assertions read from.
    """

    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.utils.execution_semantics as execution_semantics
    from eligsir.runtime.causal_replay import CausalReplaySummary

    events: list[str] = []

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
    runtime = SimpleNamespace(
        successful_supervision_uses_by_ordinal=lambda: dict(successful_supervision_uses),
        historical_usefulness_telemetry=_disabled_historical_usefulness_telemetry,
    )
    composition = SimpleNamespace(
        source=source,
        config=config,
        binding=binding,
        runtime=runtime,
        prepared_ready_path=None,
    )

    class _Driver:
        def __init__(self, _application, _adapter, **_kwargs):
            pass

        def run(self, _source):
            return CausalReplaySummary(submitted=0, ticks=1, source_identities=())

    def _slow_plot(*_args, **_kwargs):
        events.append("plot")
        return None

    def _refinement_stub(*_args, **kwargs):
        events.append("refinement")
        return {"mapping_wall_time_s": kwargs.get("mapping_wall_time_s")}

    monkeypatch.setattr(
        causal_entry, "resolve_causal_mapping_composition", lambda **_kwargs: composition
    )
    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", _Driver)
    monkeypatch.setattr(causal_entry, "write_causal_trajectory", lambda *_args: tmp_path / "t.jsonl")
    monkeypatch.setattr(causal_entry, "write_mapped_camera_trajectory_plot", _slow_plot)
    monkeypatch.setattr(causal_entry, "run_causal_post_mapping_refinement", _refinement_stub)
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )
    return events, causal_entry


def test_slow_mapped_camera_plot_does_not_delay_refinement_dispatch(tmp_path, monkeypatch):
    """A slow/stubbed plotting function must not be waited on before
    refinement dispatches -- refinement must be reached first."""

    events, causal_entry = _wire_minimal_causal_mapping(
        monkeypatch, tmp_path, successful_supervision_uses={1: 1}
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("")

    causal_entry.run_causal_mapping(
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        opt=SimpleNamespace(),
        pipe=SimpleNamespace(),
        args=_causal_mapping_args(tmp_path, causal_decision_trace=str(trace_path)),
    )

    assert events[0] == "refinement", (
        "refinement must dispatch before any mapped-camera plot call: "
        f"got order {events!r}"
    )
    assert events.count("plot") == 2  # mapping-phase + refined snapshots
    assert events.count("refinement") == 1


def test_iter_based_mapping_uses_requested_ticks_per_submission(tmp_path, monkeypatch):
    """Iteration-based release must realize its declared per-frame work."""
    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.utils.execution_semantics as execution_semantics
    from eligsir.runtime.causal_replay import CausalReplaySummary

    _wire_minimal_causal_mapping(
        monkeypatch, tmp_path, successful_supervision_uses={}
    )
    captured = {}

    class _Driver:
        def __init__(self, _application, _adapter, **kwargs):
            captured.update(kwargs)

        def run(self, _source):
            return CausalReplaySummary(submitted=0, ticks=1, source_identities=())

    monkeypatch.setattr(causal_entry, "CausalReplayDriver", _Driver)
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_ITER_BASED},
    )

    causal_entry.run_causal_mapping(
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        opt=SimpleNamespace(),
        pipe=SimpleNamespace(),
        args=_causal_mapping_args(tmp_path, streaming_steps_per_frame=20),
    )

    assert captured["tick_policy"].ticks_per_submission == 20


def test_mapping_wall_time_frozen_before_post_run_diagnostics(tmp_path, monkeypatch):
    """mapping_wall_time_s handed to refinement must not include time spent
    in diagnostic work (summary/trajectory writes) that runs after mapping
    drains but before refinement is dispatched."""

    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.utils.execution_semantics as execution_semantics
    from eligsir.runtime.causal_replay import CausalReplaySummary

    fake_clock = {"t": 0.0}

    def fake_perf_counter():
        return fake_clock["t"]

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
    runtime = SimpleNamespace(
        successful_supervision_uses_by_ordinal=lambda: {},
        historical_usefulness_telemetry=_disabled_historical_usefulness_telemetry,
    )
    composition = SimpleNamespace(
        source=source, config=config, binding=binding, runtime=runtime,
        prepared_ready_path=None,
    )

    class _Driver:
        def __init__(self, _application, _adapter, **_kwargs):
            pass

        def run(self, _source):
            # Mapping "runs" for 5 simulated seconds.
            fake_clock["t"] += 5.0
            return CausalReplaySummary(submitted=0, ticks=1, source_identities=())

    captured_wall_time = {}

    def _slow_write_causal_trajectory(*_args):
        # Diagnostic work AFTER mapping drains, BEFORE refinement dispatches.
        fake_clock["t"] += 100.0
        return tmp_path / "t.jsonl"

    def _refinement_stub(*_args, **kwargs):
        captured_wall_time["value"] = kwargs["mapping_wall_time_s"]
        return {}

    monkeypatch.setattr(
        causal_entry, "resolve_causal_mapping_composition", lambda **_kwargs: composition
    )
    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", _Driver)
    monkeypatch.setattr(causal_entry, "write_causal_trajectory", _slow_write_causal_trajectory)
    monkeypatch.setattr(
        causal_entry, "write_mapped_camera_trajectory_plot", lambda *_a, **_k: None
    )
    monkeypatch.setattr(causal_entry, "run_causal_post_mapping_refinement", _refinement_stub)
    monkeypatch.setattr(causal_entry.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )

    causal_entry.run_causal_mapping(
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        opt=SimpleNamespace(),
        pipe=SimpleNamespace(),
        args=_causal_mapping_args(tmp_path),
    )

    # 5s of real mapping, NOT 105s (the 100s the trajectory write "spent").
    assert captured_wall_time["value"] == pytest.approx(5.0)


def test_mapping_refinement_handoff_preserves_source_frontier_to_finalization(
    tmp_path, monkeypatch
):
    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.execution_semantics as execution_semantics
    from eligsir.runtime.causal_replay import CausalReplayClock, CausalReplaySummary

    source = SimpleNamespace(frames=())
    composition = SimpleNamespace(
        source=source,
        config=SimpleNamespace(hard_limits={"optimization_batch_size": 4}),
        binding=SimpleNamespace(
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
        ),
        runtime=SimpleNamespace(
            historical_usefulness_telemetry=_disabled_historical_usefulness_telemetry,
            successful_supervision_uses_by_ordinal=lambda: {},
        ),
        prepared_ready_path=None,
    )
    finalized = []
    captured = {}

    class _Observer:
        def __init__(self, **kwargs):
            self._source_frame_index = None
            self._mapping_holdout_curve_metadata = (
                causal_evaluation.mapping_holdout_curve_metadata(kwargs["args"])
            )

        def wait_for_pending_snapshot_copy(self):
            pass

        def observe_tick(self, _tick, clock):
            self._source_frame_index = clock.source_frame_index

        def finalize(self, _summary, **_kwargs):
            finalized.append(self._source_frame_index)

    class _Driver:
        def __init__(self, _application, _adapter, **kwargs):
            self._clock_observer = kwargs["tick_clock_observer"]

        def run(self, _records):
            self._clock_observer(
                SimpleNamespace(),
                CausalReplayClock(
                    optimizer_steps=4,
                    supervision_views=16,
                    mapping_active_s=1.5,
                    elapsed_s=2.0,
                    source_frame_index=400,
                ),
            )
            return CausalReplaySummary(submitted=0, ticks=1, source_identities=())

    def _refinement_stub(*_args, **kwargs):
        initial_clock = kwargs["initial_clock"]
        captured["initial_source_frame_index"] = initial_clock.source_frame_index
        kwargs["tick_clock_observer"](
            SimpleNamespace(),
            CausalReplayClock(
                optimizer_steps=5,
                supervision_views=20,
                mapping_active_s=2.0,
                elapsed_s=3.0,
                source_frame_index=initial_clock.source_frame_index,
            ),
        )
        return {}

    plan = SimpleNamespace(
        holdout_ordinals=frozenset({900}),
        guard_ordinals=frozenset(),
        role_for_packet=lambda _packet: "holdout",
    )
    monkeypatch.setattr(
        causal_entry, "resolve_causal_mapping_composition", lambda **_kwargs: composition
    )
    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", _Driver)
    monkeypatch.setattr(causal_entry, "run_causal_post_mapping_refinement", _refinement_stub)
    monkeypatch.setattr(causal_entry, "write_causal_trajectory", lambda *_args: tmp_path / "t.jsonl")
    monkeypatch.setattr(
        causal_entry, "write_mapped_camera_trajectory_plot", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(causal_evaluation, "CausalEvaluationObserver", _Observer)
    monkeypatch.setattr(causal_evaluation, "causal_holdout_plan", lambda *_args: plan)
    monkeypatch.setattr(
        causal_evaluation, "validate_train_evwb_reference_source", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        causal_evaluation, "enable_training_capture_pose_candidates", lambda *_args: None
    )
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )

    causal_entry.run_causal_mapping(
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        opt=SimpleNamespace(),
        pipe=SimpleNamespace(),
        args=_causal_mapping_args(tmp_path, streaming_post_mapping_refinement_steps=1),
    )

    assert captured["initial_source_frame_index"] == 400
    assert finalized == [400]


def test_notrace_runtime_counts_match_trace_derived_selection():
    """Parity: no-trace mode using #1687's compact runtime counts must
    identify the exact same used-camera set that the old full-decision-trace
    scan would have derived from equivalent executed ticks."""

    trajectory_rows = [
        {"id": 1, "position": [0.0, 0.0, 0.0]},
        {"id": 2, "position": [1.0, 0.0, 0.0]},
        {"id": 3, "position": [2.0, 0.0, 0.0]},
    ]
    decision_rows = [
        {
            "executed": True,
            "optimizer_steps": 1,
            "dispositions": [{"source_ordinal": 2, "mapping_candidate": True}],
            "supervision_source_records": [{"source_ordinal": 2}],
        },
        {
            "executed": True,
            "optimizer_steps": 1,
            "dispositions": [{"source_ordinal": 3, "mapping_candidate": True}],
            "supervision_source_records": [{"source_ordinal": 2}, {"source_ordinal": 3}],
        },
    ]

    trace_derived = mapped_camera_trajectory_selection(trajectory_rows, decision_rows)
    runtime_counts_derived = mapped_camera_trajectory_selection(
        trajectory_rows, (), successful_supervision_uses={2: 2, 3: 1}
    )

    assert (
        runtime_counts_derived["successfully_supervised_ordinals"]
        == trace_derived["successfully_supervised_ordinals"]
    )
    assert (
        runtime_counts_derived["successful_supervision_uses"]
        == trace_derived["successful_supervision_uses"]
    )


def test_trace_decoration_adds_candidates_without_changing_used_identities():
    """Trace-on mode may add extra `mapping_candidate` provenance, but must
    never change which sources are reported as successfully supervised --
    that identity comes solely from the runtime supervision counts."""

    trajectory_rows = [
        {"id": 1, "position": [0.0, 0.0, 0.0]},
        {"id": 2, "position": [1.0, 0.0, 0.0]},
        {"id": 5, "position": [4.0, 0.0, 0.0]},
    ]
    runtime_counts = {2: 3}

    without_trace = mapped_camera_trajectory_selection(
        trajectory_rows, (), successful_supervision_uses=runtime_counts
    )
    with_trace = mapped_camera_trajectory_selection(
        trajectory_rows,
        (
            {
                "executed": True,
                "optimizer_steps": 1,
                "dispositions": [
                    {"source_ordinal": 1, "mapping_candidate": True},
                    {"source_ordinal": 5, "mapping_candidate": True},
                ],
                # These records must be IGNORED for identity purposes because
                # successful_supervision_uses is provided.
                "supervision_source_records": [{"source_ordinal": 1}],
            },
        ),
        successful_supervision_uses=runtime_counts,
    )

    assert (
        with_trace["successfully_supervised_ordinals"]
        == without_trace["successfully_supervised_ordinals"]
        == [2]
    )
    assert with_trace["successful_supervision_uses"] == without_trace["successful_supervision_uses"]
    # The trace contributes extra decoration (candidates) on top.
    assert with_trace["mapping_candidate_ordinals"] == [1, 5]
    assert without_trace["mapping_candidate_ordinals"] == []


def test_stream_jsonl_rows_never_calls_read_text(tmp_path, monkeypatch):
    """Prove the streaming reader used by write_mapped_camera_trajectory_plot
    genuinely avoids the old read_text().splitlines() full-file pattern, not
    just that it happens to work on small files."""

    path = tmp_path / "large.jsonl"
    row_count = 5_000
    with path.open("w") as handle:
        for ordinal in range(row_count):
            handle.write(json.dumps({"id": ordinal, "position": [float(ordinal), 0.0, 0.0]}) + "\n")

    def _forbidden_read_text(self, *args, **kwargs):
        raise AssertionError(
            f"full-file read_text() must not be used for streamed JSONL parsing: {self}"
        )

    monkeypatch.setattr(Path, "read_text", _forbidden_read_text)

    rows = _stream_jsonl_rows(path)

    assert len(rows) == row_count
    assert rows[0]["id"] == 0
    assert rows[-1]["id"] == row_count - 1


def test_large_trajectory_and_trace_render_without_read_text(tmp_path, monkeypatch):
    """End-to-end through write_mapped_camera_trajectory_plot with a
    thousands-of-rows trajectory and decision trace, proving no full-file
    read_text() call is made anywhere in the path."""

    trajectory = tmp_path / "trajectory_full.jsonl"
    with trajectory.open("w") as handle:
        for ordinal in range(3_000):
            handle.write(
                json.dumps({"id": ordinal, "position": [float(ordinal), 0.0, 0.0]}) + "\n"
            )
    trace = tmp_path / "trace.jsonl"
    with trace.open("w") as handle:
        for ordinal in range(3_000):
            handle.write(
                json.dumps(
                    {
                        "executed": True,
                        "optimizer_steps": 1,
                        "dispositions": [
                            {"source_ordinal": ordinal, "mapping_candidate": True}
                        ],
                        "supervision_source_records": [],
                    }
                )
                + "\n"
            )

    def _forbidden_read_text(self, *args, **kwargs):
        raise AssertionError(f"full-file read_text() must not be used: {self}")

    monkeypatch.setattr(Path, "read_text", _forbidden_read_text)

    plotted = []

    def plot(path, trajectory_rows, decision_rows, *, successful_supervision_uses=None):
        plotted.append((len(trajectory_rows), len(decision_rows)))
        return mapped_camera_trajectory_selection(
            trajectory_rows, decision_rows,
            successful_supervision_uses=successful_supervision_uses,
        )

    monkeypatch.setattr("eligsir.utils.trajectory_eval.plot_mapped_camera_trajectory", plot)

    output = write_mapped_camera_trajectory_plot(
        tmp_path, trajectory, trace, successful_supervision_uses={1: 1}
    )

    assert output == tmp_path / "mapped_camera_trajectory.svg"
    assert plotted == [(3_000, 3_000)]


def test_decision_trace_byte_limit_excludes_rows_appended_after_snapshot(tmp_path, monkeypatch):
    """The mapping-phase plot must only see decision rows that existed at
    the moment mapping drained, not rows the (same, still-open-for-append)
    trace file accumulates afterward during refinement."""

    trace = tmp_path / "trace.jsonl"
    mapping_row = {
        "executed": True, "optimizer_steps": 1,
        "dispositions": [{"source_ordinal": 1, "mapping_candidate": True}],
        "supervision_source_records": [],
    }
    with trace.open("w") as handle:
        handle.write(json.dumps(mapping_row) + "\n")
    boundary = trace.stat().st_size

    # Refinement "appends" more decisions to the very same file afterward.
    refinement_row = {
        "executed": True, "optimizer_steps": 1,
        "dispositions": [{"source_ordinal": 9, "mapping_candidate": True}],
        "supervision_source_records": [],
    }
    with trace.open("a") as handle:
        handle.write(json.dumps(refinement_row) + "\n")

    trajectory = tmp_path / "trajectory_full.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps({"id": ordinal, "position": [float(ordinal), 0.0, 0.0]})
            for ordinal in (1, 9)
        )
        + "\n"
    )

    plotted = []

    def plot(path, trajectory_rows, decision_rows, *, successful_supervision_uses=None):
        plotted.append(decision_rows)
        return mapped_camera_trajectory_selection(
            trajectory_rows, decision_rows,
            successful_supervision_uses=successful_supervision_uses,
        )

    monkeypatch.setattr("eligsir.utils.trajectory_eval.plot_mapped_camera_trajectory", plot)

    write_mapped_camera_trajectory_plot(
        tmp_path, trajectory, trace,
        successful_supervision_uses={1: 1},
        decision_trace_byte_limit=boundary,
    )

    assert plotted[0] == [mapping_row]

    plotted.clear()
    write_mapped_camera_trajectory_plot(
        tmp_path, trajectory, trace,
        successful_supervision_uses={1: 1, 9: 1},
        out_name="mapped_camera_trajectory_refined",
    )

    assert plotted[0] == [mapping_row, refinement_row]
