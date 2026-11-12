"""Unit tests for the W3-02 Level-D mutation-trace record/verify wrapper.

See ``eligsir/optimization/gaussian/mutation_trace.py`` for why this
exists: topology mutation is explicitly outside ``GaussianOptimizationEngine``
(doc section 7.4), so the engine-OFF and engine-ON streaming loops call the
exact same mutation functions -- ``MutationTraceRecorder`` turns "we assume
that stayed identical" into an actively-checked claim.
"""

from __future__ import annotations

import pytest

from eligsir.optimization.gaussian.mutation_trace import (
    MutationTraceDivergence,
    MutationTraceRecorder,
)


def test_off_mode_is_a_true_noop(tmp_path):
    recorder = MutationTraceRecorder(mode="off")
    assert not recorder.enabled
    event = recorder.wrap(
        iteration=1, op="mcmc_step_post_backward",
        topology_version_before=0, topology_version_after=1,
        n_before=100, n_after=110, content=110,
    )
    assert event.op == "mcmc_step_post_backward"
    summary = recorder.finalize()
    assert summary == {"mutation_trace_mode": "off"}


def test_record_then_verify_round_trip_matches(tmp_path):
    trace_path = tmp_path / "trace.jsonl"

    recorder = MutationTraceRecorder(mode="record", path=str(trace_path))
    assert recorder.enabled
    recorder.wrap(
        iteration=10, op="depth_insertion_flush",
        topology_version_before=0, topology_version_after=1,
        n_before=1000, n_after=1200, content=200,
    )
    recorder.wrap(
        iteration=20, op="mcmc_step_post_backward",
        topology_version_before=1, topology_version_after=1,
        n_before=1200, n_after=1200, content=1200,
    )
    record_summary = recorder.finalize()
    assert record_summary["mutation_trace_mode"] == "record"
    assert record_summary["mutation_trace_event_count"] == 2
    assert trace_path.is_file()

    verifier = MutationTraceRecorder(mode="verify", path=str(trace_path))
    verifier.wrap(
        iteration=10, op="depth_insertion_flush",
        topology_version_before=0, topology_version_after=1,
        n_before=1000, n_after=1200, content=200,
    )
    verifier.wrap(
        iteration=20, op="mcmc_step_post_backward",
        topology_version_before=1, topology_version_after=1,
        n_before=1200, n_after=1200, content=1200,
    )
    verify_summary = verifier.finalize()
    assert verify_summary["mutation_trace_mode"] == "verify"
    assert verify_summary["mutation_trace_applied"] is True
    assert verify_summary["mutation_trace_divergences"] == []
    assert verify_summary["mutation_trace_recorded_event_count"] == 2
    assert verify_summary["mutation_trace_observed_event_count"] == 2


def test_verify_raises_immediately_on_topology_divergence(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = MutationTraceRecorder(mode="record", path=str(trace_path))
    recorder.wrap(
        iteration=5, op="mcmc_step_post_backward",
        topology_version_before=0, topology_version_after=1,
        n_before=500, n_after=520, content=520,
    )
    recorder.finalize()

    verifier = MutationTraceRecorder(mode="verify", path=str(trace_path))
    with pytest.raises(MutationTraceDivergence):
        verifier.wrap(
            iteration=5, op="mcmc_step_post_backward",
            topology_version_before=0, topology_version_after=1,
            n_before=500, n_after=521,  # diverges: engine-ON produced a different N
            content=521,
        )


def test_verify_fails_closed_on_missing_events(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = MutationTraceRecorder(mode="record", path=str(trace_path))
    recorder.wrap(
        iteration=1, op="mcmc_step_post_backward",
        topology_version_before=0, topology_version_after=1,
        n_before=100, n_after=110, content=110,
    )
    recorder.wrap(
        iteration=2, op="mcmc_step_post_backward",
        topology_version_before=1, topology_version_after=1,
        n_before=110, n_after=110, content=110,
    )
    recorder.finalize()

    # Treatment run only replays the first event (e.g. it terminated early).
    verifier = MutationTraceRecorder(mode="verify", path=str(trace_path))
    verifier.wrap(
        iteration=1, op="mcmc_step_post_backward",
        topology_version_before=0, topology_version_after=1,
        n_before=100, n_after=110, content=110,
    )
    summary = verifier.finalize()
    assert summary["mutation_trace_applied"] is False
    assert summary["mutation_trace_divergences"]


def test_verify_mode_requires_a_trace_path():
    with pytest.raises(ValueError):
        MutationTraceRecorder(mode="verify", path="")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        MutationTraceRecorder(mode="bogus")
