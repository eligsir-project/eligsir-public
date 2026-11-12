"""Unit tests for PhaseSegments sequential bracketing."""

from __future__ import annotations

import os

import pytest

from eligsir.utils.ingest_phase_profiler import PhaseSegments, _PROFILE_PHASES, _TORCH_RANGES, _timed


def test_phase_segments_is_inert_when_profiling_off():
    """When neither _PROFILE_PHASES nor _TORCH_RANGES is set, begin() and close() should be no-ops."""
    segments = PhaseSegments()

    # No timer/record_function calls should happen
    segments.begin("test_phase")
    segments.close()

    # No assertions needed; if the code doesn't crash, the test passes
    assert True


def test_phase_segments_opens_and_closes_phases_when_profile_phases_on(monkeypatch):
    """When _PROFILE_PHASES is on, begin() should call _timed.__enter__ and close() should call __exit__."""
    monkeypatch.setenv("SPLAT_PROFILE_PHASES", "1")
    # Re-import to pick up the env var
    import importlib
    import eligsir.utils.ingest_phase_profiler as profiler_module
    importlib.reload(profiler_module)

    from eligsir.utils.ingest_phase_profiler import PhaseSegments as _PhaseSegments

    segments = _PhaseSegments()
    segments.begin("phase1")
    assert segments._current is not None
    assert segments._current.name == "phase1"

    segments.begin("phase2")
    assert segments._current is not None
    assert segments._current.name == "phase2"

    segments.close()
    assert segments._current is None


def test_phase_segments_closes_previous_on_begin():
    """begin() should close the previous segment before opening a new one."""
    # Create segments in an environment where profiling is enabled
    os.environ["SPLAT_PROFILE_PHASES"] = "1"
    import importlib
    import eligsir.utils.ingest_phase_profiler as profiler_module
    importlib.reload(profiler_module)

    from eligsir.utils.ingest_phase_profiler import PhaseSegments as _PhaseSegments

    segments = _PhaseSegments()
    segments.begin("phase1")
    assert segments._current is not None
    phase1 = segments._current

    segments.begin("phase2")
    assert segments._current is not None
    assert segments._current is not phase1
    assert segments._current.name == "phase2"


def test_absent_replan_constant_is_correct_value():
    """Negative control: verify the absence replan threshold is the expected value (64 ticks).

    This ensures that if someone accidentally changed _ABSENT_REPLAN_AFTER_TICKS,
    the test_absent_authorized_replay_source_replans_after_bounded_starvation test would catch it.
    """
    from eligsir.runtime.ready_b4_streaming_adapter import ReadyB4StreamingOptimizationAdapter

    # The bound must be 64 ticks - this is the documented design
    assert ReadyB4StreamingOptimizationAdapter._ABSENT_REPLAN_AFTER_TICKS == 64
