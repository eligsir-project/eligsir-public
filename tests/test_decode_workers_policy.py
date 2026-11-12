"""Tests for --causal_decode_workers_policy (dynamic decode-worker sizing).

See docs/superpowers/specs/2026-09-08-residency-budget-design.md, "Addendum:
decode-worker starvation" and .superpowers/sdd/2026-09-08-residency-budget/
task-3-brief.md for the formula and calibration this exercises.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eligsir.runtime.payload_budget import (
    estimate_decode_ms_per_frame,
    resolve_decode_workers,
)


class TestResolveDecodeWorkersFixedPolicy:
    def test_fixed_policy_returns_fixed_workers_unchanged(self):
        """policy='fixed' always returns fixed_workers, regardless of other inputs."""
        workers, metadata = resolve_decode_workers(
            policy="fixed",
            fixed_workers=2,
            cpu_count=32,
            source_hz=30.0,
            per_frame_decode_ms_estimate=19.0,
        )
        assert workers == 2
        assert metadata == {"policy": "fixed", "source": "explicit_or_default"}

    def test_fixed_policy_ignores_extreme_inputs(self):
        """Even a huge decode estimate / tiny cpu_count must not move 'fixed'."""
        workers, metadata = resolve_decode_workers(
            policy="fixed",
            fixed_workers=5,
            cpu_count=1,
            source_hz=1.0,
            per_frame_decode_ms_estimate=5000.0,
        )
        assert workers == 5
        assert metadata["policy"] == "fixed"


class TestResolveDecodeWorkersCpuScaledPolicy:
    def test_tum_demand_stays_at_floor(self):
        """TUM (640x480) warm decode (~7ms) at 30fps needs << 1 worker; cap not binding."""
        estimate = estimate_decode_ms_per_frame(640, 480)
        workers, metadata = resolve_decode_workers(
            policy="cpu_scaled",
            fixed_workers=2,
            cpu_count=32,
            source_hz=30.0,
            per_frame_decode_ms_estimate=estimate,
        )
        assert workers == 2
        assert metadata["needed"] == 1
        assert metadata["cap"] == 16

    def test_scannet_resolution_at_30hz_does_not_cross_the_floor(self):
        """Documents a real finding, not a workaround: under the brief's own WARM-only
        calibration (7ms @ TUM, 19ms @ ScanNet), a 1296x968 source at 30fps has
        needed=ceil(19/33.33)=1, so cpu_scaled resolves to the same 2 workers as
        'fixed' for this exact scenario. The starvation case cited in
        arguments/__init__.py:815-816 needs either a higher source_hz or a
        cold-cache decode estimate to actually cross the floor -- see the
        'concerns' note returned with this task.
        """
        estimate = estimate_decode_ms_per_frame(1296, 968)
        assert estimate == pytest.approx(19.0, abs=1e-9)
        workers, metadata = resolve_decode_workers(
            policy="cpu_scaled",
            fixed_workers=2,
            cpu_count=32,
            source_hz=30.0,
            per_frame_decode_ms_estimate=estimate,
        )
        assert workers == 2
        assert metadata["needed"] == 1

    def test_scaling_branch_fires_when_decode_exceeds_frame_budget(self):
        """A decode estimate well above one frame's budget scales workers up,
        bounded by the cap (16 on a 32-core host)."""
        workers, metadata = resolve_decode_workers(
            policy="cpu_scaled",
            fixed_workers=2,
            cpu_count=32,
            source_hz=30.0,
            per_frame_decode_ms_estimate=120.0,
        )
        assert workers > 2
        assert workers <= 16
        assert metadata["cap"] == 16
        assert metadata["needed"] == 4  # ceil(120 / 33.33..) == 4

    def test_cap_binds_on_low_core_count_host(self):
        """cpu_count=4 caps workers at 2 even when needed is much larger."""
        workers, metadata = resolve_decode_workers(
            policy="cpu_scaled",
            fixed_workers=2,
            cpu_count=4,
            source_hz=30.0,
            per_frame_decode_ms_estimate=500.0,
        )
        assert metadata["cap"] == 2
        assert workers == 2

    def test_fixed_workers_acts_as_a_floor(self):
        """A higher explicit fixed_workers still wins even for a low-need (TUM) demand."""
        estimate = estimate_decode_ms_per_frame(640, 480)
        workers, metadata = resolve_decode_workers(
            policy="cpu_scaled",
            fixed_workers=6,
            cpu_count=32,
            source_hz=30.0,
            per_frame_decode_ms_estimate=estimate,
        )
        assert workers >= 6
        assert workers == 6


class TestEstimateDecodeMsPerFrame:
    def test_calibration_points_are_exact(self):
        """The linear fit must pass exactly through both documented calibration points."""
        assert estimate_decode_ms_per_frame(640, 480) == pytest.approx(7.0, abs=1e-9)
        assert estimate_decode_ms_per_frame(1296, 968) == pytest.approx(19.0, abs=1e-9)

    def test_monotonic_in_resolution(self):
        """A larger frame estimates a larger decode cost."""
        small = estimate_decode_ms_per_frame(640, 480)
        large = estimate_decode_ms_per_frame(1920, 1440)
        assert large > small


class TestCausalEntryDecodeWorkersWiring:
    def test_cpu_scaled_policy_sets_resolved_decode_workers(self):
        """Under 'cpu_scaled', causal_entry resolves and overwrites
        args.causal_decode_workers before producer construction reads it."""
        from eligsir.runtime.causal_entry import resolve_decode_workers_for_args

        args = SimpleNamespace(
            causal_decode_workers=2,
            causal_decode_workers_policy="cpu_scaled",
            streaming_input_fps_cap=30.0,
        )
        resolve_decode_workers_for_args(
            args,
            native_width=1296,
            native_height=968,
            cpu_count_fn=lambda: 32,
        )
        assert args.causal_decode_workers == 2  # per finding above: does not cross floor here
        assert hasattr(args, "_resolved_decode_workers_metadata")
        assert args._resolved_decode_workers_metadata["policy"] == "cpu_scaled"

    def test_fixed_policy_leaves_causal_decode_workers_untouched(self):
        """Under 'fixed' (the default), args.causal_decode_workers must be byte-identical
        to whatever was explicitly set/parsed -- no resolution happens."""
        from eligsir.runtime.causal_entry import resolve_decode_workers_for_args

        args = SimpleNamespace(
            causal_decode_workers=2,
            causal_decode_workers_policy="fixed",
            streaming_input_fps_cap=30.0,
        )
        resolve_decode_workers_for_args(
            args,
            native_width=1296,
            native_height=968,
            cpu_count_fn=lambda: 32,
        )
        assert args.causal_decode_workers == 2
        assert args._resolved_decode_workers_metadata["policy"] == "fixed"

    def test_cpu_scaled_scales_up_for_a_high_load_estimate(self):
        """Wiring-level check that scaling can actually raise the resolved value
        when the estimate crosses the frame budget (uses the same low-level
        resolve_decode_workers scaling branch already verified above)."""
        from eligsir.runtime.causal_entry import resolve_decode_workers_for_args

        args = SimpleNamespace(
            causal_decode_workers=2,
            causal_decode_workers_policy="cpu_scaled",
            streaming_input_fps_cap=1000.0,  # tiny frame budget (1ms) -> forces scaling
            streaming_hot_tier_max_bytes=None,
        )
        resolve_decode_workers_for_args(
            args,
            native_width=1296,
            native_height=968,
            cpu_count_fn=lambda: 32,
        )
        assert args.causal_decode_workers > 2
        assert args.causal_decode_workers <= 16
