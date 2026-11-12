"""Tests for residency budget wiring into the runtime (config, causal entry, production app)."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from eligsir.runtime.payload_budget import (
    ResidencyDemand,
    ResidencySupply,
    format_budget_line,
    resolve_residency_budget,
)


class TestResidencyPolicyWiringInConfigBoundary:
    """Test resolve_admitted_payload_bounds with residency policy."""

    def test_residency_policy_well_provisioned(self):
        """Test residency policy with sufficient device and host memory.

        1296x968, half tier, window 200, B16, injected supply (10 GiB device, 40 GiB host).
        Should return pool >= 200 and hot bytes == budget.hot_tier_max_bytes.
        """
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="residency",
            streaming_live_admitted_payload_max_frames=None,  # No explicit flag
            streaming_hot_tier_max_bytes=None,
            streaming_keyframe_window=200,
            streaming_optimization_batch_size=16,
            streaming_mapping_resolution_tier="half",
            # Headroom flags (defaults)
            streaming_payload_budget_device_headroom_frac=0.5,
            streaming_payload_budget_host_headroom_frac=0.25,
        )

        # Inject supply: 10 GiB device, 40 GiB host
        def probe_supply_fn():
            return ResidencySupply(
                device_free_bytes=10 * 1024 * 1024 * 1024,
                device_total_bytes=12 * 1024 * 1024 * 1024,
                gtt_total_bytes=None,
                gtt_used_bytes=None,
                host_available_bytes=40 * 1024 * 1024 * 1024,
            )

        frames, hot_bytes, metadata = resolve_admitted_payload_bounds(
            args,
            native_width=1296,
            native_height=968,
            probe_supply_fn=probe_supply_fn,
        )

        # Pool should be >= window (200)
        assert frames >= 200, f"Expected pool >= 200, got {frames}"

        # Check metadata has residency budget
        assert "residency" in metadata, "metadata should contain 'residency' key"
        budget = metadata["residency"]

        # hot_bytes should match budget
        assert hot_bytes == budget.hot_tier_max_bytes, \
            f"hot_bytes {hot_bytes} != budget.hot_tier_max_bytes {budget.hot_tier_max_bytes}"

        # Not starved with 10 GiB device and 40 GiB host
        assert not budget.starved, "Should not be starved with ample memory"

    def test_residency_policy_starved_device_memory(self):
        """Test residency policy with insufficient device memory.

        Same config but with only 3 GiB device -> starved, should set window_effective.
        """
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="residency",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
            streaming_keyframe_window=200,
            streaming_optimization_batch_size=16,
            streaming_mapping_resolution_tier="half",
            streaming_payload_budget_device_headroom_frac=0.5,
            streaming_payload_budget_host_headroom_frac=0.25,
        )

        # Inject supply: 3 GiB device (starved), 40 GiB host
        def probe_supply_fn():
            return ResidencySupply(
                device_free_bytes=3 * 1024 * 1024 * 1024,
                device_total_bytes=4 * 1024 * 1024 * 1024,
                gtt_total_bytes=None,
                gtt_used_bytes=None,
                host_available_bytes=40 * 1024 * 1024 * 1024,
            )

        frames, hot_bytes, metadata = resolve_admitted_payload_bounds(
            args,
            native_width=1296,
            native_height=968,
            probe_supply_fn=probe_supply_fn,
        )

        # Check metadata has residency budget
        assert "residency" in metadata, "metadata should contain 'residency' key"
        budget = metadata["residency"]

        # Should be starved with only 3 GiB
        assert budget.starved, "Should be starved with only 3 GiB device memory"

        # window_effective should be < window when starved
        assert budget.window_effective < 200, \
            f"window_effective {budget.window_effective} should be < 200 when starved"

    def test_residency_policy_explicit_flag_wins(self):
        """Test that explicit frames flag wins over residency policy.

        Explicit --streaming_live_admitted_payload_max_frames 64 should override residency.
        """
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="residency",
            streaming_live_admitted_payload_max_frames=64,  # Explicit!
            streaming_hot_tier_max_bytes=None,
            streaming_keyframe_window=200,
            streaming_optimization_batch_size=16,
            streaming_mapping_resolution_tier="half",
            streaming_payload_budget_device_headroom_frac=0.5,
            streaming_payload_budget_host_headroom_frac=0.25,
        )

        def probe_supply_fn():
            return ResidencySupply(
                device_free_bytes=10 * 1024 * 1024 * 1024,
                device_total_bytes=12 * 1024 * 1024 * 1024,
                gtt_total_bytes=None,
                gtt_used_bytes=None,
                host_available_bytes=40 * 1024 * 1024 * 1024,
            )

        frames, hot_bytes, metadata = resolve_admitted_payload_bounds(
            args,
            native_width=1296,
            native_height=968,
            probe_supply_fn=probe_supply_fn,
        )

        # Explicit flag should win
        assert frames == 64, f"Expected 64 (explicit flag), got {frames}"
        assert metadata["source"] == "explicit_flag"

    def test_residency_policy_cold_bounds(self):
        """Test resolve_cold_payload_bounds with residency policy."""
        from eligsir.runtime.config_boundary import resolve_cold_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="residency",
            streaming_live_cold_payload_max_frames=128,
            streaming_live_cold_payload_max_bytes=2 * 1024 * 1024 * 1024,
            streaming_observation_db_retain_all=False,
        )

        cold_frames, cold_bytes = resolve_cold_payload_bounds(args)

        # Should return default cold bounds when policy is residency and no retain_all
        assert cold_frames == 128
        assert cold_bytes == 2 * 1024 * 1024 * 1024


class TestCausalEntryResidencyBudgetClamping:
    """Test apply_residency_budget helper in causal_entry."""

    def test_apply_residency_budget_clamps_window_when_starved(self):
        """Test that apply_residency_budget clamps args.streaming_keyframe_window when starved."""
        from eligsir.runtime.causal_entry import apply_residency_budget
        from eligsir.runtime.payload_budget import ResidencyBudget

        args = SimpleNamespace(
            streaming_keyframe_window=200,
            streaming_resolution_cache_max_frames=200,
        )

        # Create a starved budget
        budget = ResidencyBudget(
            hot_tier_max_bytes=1024 * 1024,
            cold_payload_max_bytes=2 * 1024 * 1024 * 1024,
            cold_payload_max_frames=128,
            admitted_payload_max_frames=128,
            resident_frames=80,
            window_effective=80,
            prefetch_entries=32,
            starved=True,
            metadata={},
        )

        apply_residency_budget(args, budget)

        # Window should be clamped to window_effective
        assert args.streaming_keyframe_window == 80, \
            f"Expected window clamped to 80, got {args.streaming_keyframe_window}"
        assert args.streaming_resolution_cache_max_frames == 80, \
            f"Expected cache clamped to 80, got {args.streaming_resolution_cache_max_frames}"

    def test_apply_residency_budget_no_clamp_when_not_starved(self):
        """Test that apply_residency_budget doesn't clamp when not starved."""
        from eligsir.runtime.causal_entry import apply_residency_budget
        from eligsir.runtime.payload_budget import ResidencyBudget

        args = SimpleNamespace(
            streaming_keyframe_window=200,
            streaming_resolution_cache_max_frames=200,
        )

        # Create a non-starved budget
        budget = ResidencyBudget(
            hot_tier_max_bytes=5 * 1024 * 1024 * 1024,
            cold_payload_max_bytes=40 * 1024 * 1024 * 1024,
            cold_payload_max_frames=2000,
            admitted_payload_max_frames=2000,
            resident_frames=200,
            window_effective=200,
            prefetch_entries=32,
            starved=False,
            metadata={},
        )

        apply_residency_budget(args, budget)

        # Window should not change
        assert args.streaming_keyframe_window == 200
        assert args.streaming_resolution_cache_max_frames == 200


class TestAdmissionPrefetchCapacityIntegration:
    """Test that residency budget sets prefetch capacity correctly."""

    def test_prefetch_capacity_from_residency_budget(self):
        """Test that residency budget resolves to correct prefetch capacity for B16."""
        from eligsir.runtime.payload_budget import (
            ResidencyDemand,
            ResidencySupply,
            resolve_residency_budget,
        )

        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )

        supply = ResidencySupply(
            device_free_bytes=10 * 1024 * 1024 * 1024,
            device_total_bytes=12 * 1024 * 1024 * 1024,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        # For B16, prefetch_entries should be max(8, 2*16) = 32
        assert budget.prefetch_entries == 32, \
            f"Expected 32, got {budget.prefetch_entries}"

    def test_admission_prefetch_capacity_default_formula(self):
        """Test that admission_prefetch_capacity_for_batch uses default formula."""
        from eligsir.runtime.production_application import (
            admission_prefetch_capacity_for_batch,
        )

        # Default formula: max(8, 2 * batch_size)
        capacity = admission_prefetch_capacity_for_batch(batch_size=16)
        assert capacity == 32, f"Expected max(8, 2*16) = 32, got {capacity}"

        capacity = admission_prefetch_capacity_for_batch(batch_size=1)
        assert capacity == 8, f"Expected max(8, 2*1) = 8, got {capacity}"

        capacity = admission_prefetch_capacity_for_batch(batch_size=64)
        assert capacity == 128, f"Expected max(8, 2*64) = 128, got {capacity}"


class TestResolutionScaledByteIdentity:
    """Test that resolution_scaled policy remains byte-identical after changes."""

    def test_resolution_scaled_baseline_unchanged(self):
        """Verify resolution_scaled policy returns same values as before."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )

        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=640, native_height=480
        )

        # These values are from the original test_payload_budget_policy.py
        assert frames == 512
        assert meta["source"] == "resolution_scaled"

    def test_resolution_scaled_1296x968_unchanged(self):
        """Verify 1296x968 resolution_scaled still returns 125."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )

        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1296, native_height=968
        )

        # This is the exact value from test_payload_budget_policy.py
        assert frames == 125, f"Expected 125, got {frames}"
