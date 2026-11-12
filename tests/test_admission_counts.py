"""Regression tests for frame admission counts under coverage enforcement and gates.

WS5: guard against regressions in spf/enforcement/gate logic that silently
change how many frames are admitted, breaking comparability across runs.
"""

from __future__ import annotations

import math

import argparse

import pytest

from eligsir.mapping.scheduler import MappingScheduler
from arguments import StreamingParams


def _default_streaming() -> StreamingParams:
    parser = argparse.ArgumentParser()
    return StreamingParams(parser)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _admitted(scheduler: MappingScheduler, n_frames: int) -> int:
    """Count how many of the first n_frames would be released by the scheduler.

    Simulates the iter-based release check: frame i (0-indexed) is released when
    iteration >= (i+1) * effective_spf.  Capped at n_frames_capped if set.
    """
    spf = scheduler.effective_spf
    cap = scheduler.n_frames_capped if scheduler.n_frames_capped is not None else n_frames
    return min(cap, n_frames)


# ---------------------------------------------------------------------------
# 1. Full-coverage enforcement: all 100 frames admitted at spf=30
# ---------------------------------------------------------------------------

class TestFullCoverageEnforcement:
    def test_all_frames_admitted_when_enforce_on(self) -> None:
        # 100 frames, 3000 iters, spf=30 → coverage gives spf=floor(3000/100)=30, all admitted
        sched = MappingScheduler(
            steps_per_frame=30,
            n_frames=100,
            iterations=3000,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == 30
        assert sched.n_frames_capped is None
        assert _admitted(sched, 100) == 100

    def test_enforce_reduces_spf_to_fit(self) -> None:
        # 100 frames, 3000 iters, spf=150 → enforcement reduces to floor(3000/100)=30
        sched = MappingScheduler(
            steps_per_frame=150,
            n_frames=100,
            iterations=3000,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == 30
        assert _admitted(sched, 100) == 100


# ---------------------------------------------------------------------------
# 2. min_spf floor: when enforcement would crush spf below floor, cap frames
# ---------------------------------------------------------------------------

class TestMinSpfFloor:
    def test_frame_cap_when_enforce_crushes_below_min(self) -> None:
        # 100 frames, 3000 iters, min_spf=50 → max_spf=30 < 50 → cap frames to floor(3000/50)=60
        sched = MappingScheduler(
            steps_per_frame=50,
            n_frames=100,
            iterations=3000,
            enforce_full_coverage=True,
            min_steps_per_frame=50,
        )
        assert sched.effective_spf == 50
        assert sched.n_frames_capped == 60
        assert _admitted(sched, 100) == 60

    def test_min_spf_no_cap_when_fits(self) -> None:
        # 100 frames, 5000 iters, min_spf=50 → max_spf=50, no cap needed
        sched = MappingScheduler(
            steps_per_frame=50,
            n_frames=100,
            iterations=5000,
            enforce_full_coverage=True,
            min_steps_per_frame=50,
        )
        assert sched.effective_spf == 50
        assert sched.n_frames_capped is None
        assert _admitted(sched, 100) == 100


# ---------------------------------------------------------------------------
# 3. No enforcement: spf unchanged, all frames released regardless of budget
# ---------------------------------------------------------------------------

class TestNoEnforcement:
    def test_spf_unchanged_when_enforce_off(self) -> None:
        sched = MappingScheduler(
            steps_per_frame=20,
            n_frames=100,
            iterations=1000,  # would require spf=10 under enforcement
            enforce_full_coverage=False,
        )
        assert sched.effective_spf == 20
        assert sched.n_frames_capped is None

    def test_all_frames_released_with_enough_iters(self) -> None:
        # With enforce off: all 100 frames can be released (caller controls iteration count)
        sched = MappingScheduler(
            steps_per_frame=20,
            n_frames=100,
            iterations=3000,
            enforce_full_coverage=False,
        )
        assert _admitted(sched, 100) == 100


# ---------------------------------------------------------------------------
# 4. Default arguments: enforce_full_coverage=False, spf≠50
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_enforce_full_coverage_default_is_on(self) -> None:
        # Superseded 2026-07-04 (commit cb140d2, "unify coverage enforcement
        # and termination for live/bag ingestion"): the flag's
        # steps_per_frame-shrinking arithmetic became informational-only, and
        # the unified rolling-idle termination rule is what now guarantees
        # coverage -- but the flag itself stayed default-on. CLAUDE.md's Run
        # Contracts documents "--no-streaming_enforce_full_coverage" as the
        # explicit opt-out for smoke tests only, confirming True is the
        # intentional current default, not "incompatible with live SLAM" as
        # this test previously (incorrectly) asserted.
        args = _default_streaming()
        assert args.streaming_enforce_full_coverage is True, (
            "streaming_enforce_full_coverage must default to True per CLAUDE.md's "
            "Run Contracts (--no-streaming_enforce_full_coverage is the smoke-test-only opt-out)"
        )

    def test_default_spf_is_not_legacy_50(self) -> None:
        args = _default_streaming()
        assert args.streaming_steps_per_frame != 50, (
            "Default spf must not be 50 — that was the pre-WS2 legacy value; "
            f"current default is {args.streaming_steps_per_frame}"
        )

    def test_default_frame_admission_is_all(self) -> None:
        # Superseded: "all" (admit every frame unconditionally) is the
        # current, deliberate floor3-champion-stack convention -- it
        # trivially guarantees full coverage without needing --online_mode
        # or a gated admission mode (see project docs/memory
        # "coverage-gap-only-applies-to-gated-admission"). hybrid_keyframe
        # remains available and is a real, distinct experiment mode, but is
        # no longer the default.
        args = _default_streaming()
        assert args.streaming_frame_admission == "all", (
            "Default admission mode must be 'all' (floor3 champion-stack convention); "
            f"got {args.streaming_frame_admission!r}"
        )

    def test_rolling_window_min_admits_is_nonzero(self) -> None:
        args = _default_streaming()
        assert args.streaming_keyframe_rolling_window_min_admits >= 1, (
            "Rolling-window floor must be ≥1 to prevent starvation on revisit scenes"
        )


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_frame_sequence(self) -> None:
        sched = MappingScheduler(
            steps_per_frame=100,
            n_frames=1,
            iterations=100,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf >= 1
        assert _admitted(sched, 1) == 1

    def test_zero_frames_no_crash(self) -> None:
        # n_frames=0 means live source; enforcement must be silently skipped
        sched = MappingScheduler(
            steps_per_frame=30,
            n_frames=0,
            iterations=3000,
            enforce_full_coverage=True,
        )
        assert sched.effective_spf == 30
        assert sched.n_frames_capped is None

    def test_front_load_ratio_reduces_ingest_budget(self) -> None:
        # ratio=0.5 → ingest budget = floor(3000 * 0.5) = 1500 → spf=floor(1500/100)=15
        sched = MappingScheduler(
            steps_per_frame=150,
            n_frames=100,
            iterations=3000,
            enforce_full_coverage=True,
            ingest_front_load_ratio=0.5,
        )
        assert sched.effective_spf == 15
        assert _admitted(sched, 100) == 100
