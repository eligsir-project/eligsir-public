"""Tests for residency-first payload budget resolver."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from eligsir.runtime.payload_budget import (
    ResidencyBudget,
    ResidencyDemand,
    ResidencySupply,
    format_budget_line,
    read_meminfo_available,
    resolve_residency_budget,
)


class TestReadMeminfoAvailable:
    """Test /proc/meminfo MemAvailable parsing."""

    def test_parse_valid_meminfo(self):
        """Parse valid MemAvailable line."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("MemTotal:       16384000 kB\n")
            f.write("MemFree:        12345000 kB\n")
            f.write("MemAvailable:   12345 kB\n")
            f.write("Buffers:         100000 kB\n")
            f.flush()
            path = f.name

        try:
            result = read_meminfo_available(path)
            assert result == 12345 * 1024, f"Expected {12345 * 1024}, got {result}"
        finally:
            Path(path).unlink()

    def test_missing_memavailable_key(self):
        """Return None when MemAvailable key is missing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("MemTotal:       16384000 kB\n")
            f.write("MemFree:        12345000 kB\n")
            f.flush()
            path = f.name

        try:
            result = read_meminfo_available(path)
            assert result is None
        finally:
            Path(path).unlink()

    def test_nonexistent_file(self):
        """Return None when file does not exist."""
        result = read_meminfo_available("/nonexistent/path/meminfo")
        assert result is None


class TestResidencyBudgetScanNetRTX4070:
    """Test case 1: ScanNet-on-rtx4070.

    1296x968, half tier, window 200, B16; supply device_free 10 GiB, host 40 GiB, no GTT
    -> hot = demand_hot (200 * 20.07 MB/4 * 2 = ~2.0 GB fits under 0.4*5 GiB),
       cold covers 200 frames, resident >= 200, starved False, pool 200+.
    """

    def test_scannet_rtx4070_fits_in_budget(self):
        """ScanNet on RTX4070 should fit without starvation."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,  # half
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,  # 40 GiB
        )

        budget = resolve_residency_budget(demand, supply)

        # Should not be starved
        assert not budget.starved
        assert budget.window_effective == 200
        # Pool should be at least 200 (may be higher if cold storage allows more)
        assert budget.admitted_payload_max_frames >= 200
        # Resident should be at least 200 if supply covers it
        assert budget.resident_frames >= 200


class TestResidencyBudgetStarvation:
    """Test case 2: ScanNet same demand with low device_free -> starved."""

    def test_scannet_starved_with_low_device_memory(self):
        """ScanNet with only 3 GiB device should show starvation."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,  # half
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=3 * 1024 * 1024 * 1024,  # 3 GiB
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,  # 40 GiB
        )

        budget = resolve_residency_budget(demand, supply)

        # Should be starved
        assert budget.starved
        # window_effective should equal resident
        assert budget.window_effective == budget.resident_frames
        # window_effective should never be below 8
        assert budget.window_effective >= 8


class TestResidencyBudgetTUM:
    """Test case 3: TUM 640x480 half window 400 on 12 GiB."""

    def test_tum_not_starved_and_bounded_hot(self):
        """TUM should not be starved and hot_bytes should be reasonable."""
        demand = ResidencyDemand(
            native_width=640,
            native_height=480,
            tier_divisor=2,  # half
            window_frames=400,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=12 * 1024 * 1024 * 1024,  # 12 GiB
            device_total_bytes=12 * 1024 * 1024 * 1024,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        # Should not be starved
        assert not budget.starved
        assert budget.window_effective == 400
        # hot_bytes should be less than 20% of total device
        device_total = 12 * 1024 * 1024 * 1024
        assert budget.hot_tier_max_bytes < 0.20 * device_total


class TestResidencyBudgetGTT:
    """Test case 4: AMD GTT path."""

    def test_gtt_device_supply_uses_min(self):
        """GTT device supply should use min(gtt_available, device_free)."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )
        # gtt_total 96 GiB, gtt_used 30 GiB (available 66 GiB)
        # device_free 120 GiB (bogus, more than total)
        # min(66, 120) = 66 GiB
        supply = ResidencySupply(
            device_free_bytes=120 * 1024 * 1024 * 1024,  # 120 GiB
            device_total_bytes=128 * 1024 * 1024 * 1024,
            gtt_total_bytes=96 * 1024 * 1024 * 1024,  # 96 GiB
            gtt_used_bytes=30 * 1024 * 1024 * 1024,  # 30 GiB used
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        # device_supply should be min(66, 120) = 66 GiB before headroom
        # After 0.5 headroom: 66 * 0.5 = 33 GiB
        expected_device_supply = 66 * 1024 * 1024 * 1024 * 0.5
        actual_device_supply = budget.metadata["device_supply"]
        assert isinstance(actual_device_supply, int)
        assert actual_device_supply == int(expected_device_supply)


class TestResidencyBudgetUnknownSupply:
    """Test case 5: Unknown supplies (all None)."""

    def test_unknown_supplies_no_exception_unbounded(self):
        """Unknown supplies should not raise and should be unbounded."""
        demand = ResidencyDemand(
            native_width=640,
            native_height=480,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=None,
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=None,
        )

        # Should not raise
        budget = resolve_residency_budget(demand, supply)

        # Should not be starved (unbounded)
        assert not budget.starved
        # Metadata should record unknown supplies
        assert budget.metadata["device_supply"] == "unknown"
        assert budget.metadata["host_supply"] == "unknown"


class TestFormatBudgetLine:
    """Test case 7: format_budget_line output."""

    def test_format_budget_line_basic(self):
        """Budget line should start with [payload-budget] policy=residency."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=10 * 1024 * 1024 * 1024,
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )
        budget = resolve_residency_budget(demand, supply)

        line = format_budget_line(budget)

        assert isinstance(line, str)
        assert line.startswith("[payload-budget] policy=residency")
        assert "native=1296x968" in line
        assert "tier=half" in line
        assert "window=200" in line

    def test_format_budget_line_single_line(self):
        """Budget line should be one line."""
        demand = ResidencyDemand(
            native_width=640,
            native_height=480,
            tier_divisor=1,
            window_frames=200,
            batch_size=8,
        )
        supply = ResidencySupply(
            device_free_bytes=8 * 1024 * 1024 * 1024,
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=30 * 1024 * 1024 * 1024,
        )
        budget = resolve_residency_budget(demand, supply)

        line = format_budget_line(budget)

        # Should be one line (no newlines)
        assert "\n" not in line
        assert line.count("[payload-budget]") == 1

    def test_format_budget_line_starvation_marker(self):
        """Starved budget should include STARVATION marker."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=1 * 1024 * 1024 * 1024,  # Very limited
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )
        budget = resolve_residency_budget(demand, supply)

        line = format_budget_line(budget)

        # If starved, line must contain STARVATION
        if budget.starved:
            assert "STARVATION" in line


class TestResidencyBudgetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_window_effective_never_below_8(self):
        """window_effective should never be below 8 even if starved."""
        demand = ResidencyDemand(
            native_width=4096,
            native_height=4096,
            tier_divisor=1,
            window_frames=200,
            batch_size=4,
        )
        supply = ResidencySupply(
            device_free_bytes=100 * 1024 * 1024,  # Very small device
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=100 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        # window_effective should never be below 8
        assert budget.window_effective >= 8

    def test_prefetch_entries_at_least_8(self):
        """prefetch_entries should be at least 8."""
        demand = ResidencyDemand(
            native_width=640,
            native_height=480,
            tier_divisor=1,
            window_frames=100,
            batch_size=1,
        )
        supply = ResidencySupply(
            device_free_bytes=10 * 1024 * 1024 * 1024,
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=30 * 1024 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        # prefetch_entries should be max(8, 2*batch_size) = max(8, 2) = 8
        assert budget.prefetch_entries >= 8

    def test_prefetch_entries_scales_with_batch(self):
        """prefetch_entries should scale with batch size."""
        for batch_size in [1, 4, 8, 16]:
            demand = ResidencyDemand(
                native_width=640,
                native_height=480,
                tier_divisor=1,
                window_frames=100,
                batch_size=batch_size,
            )
            supply = ResidencySupply(
                device_free_bytes=10 * 1024 * 1024 * 1024,
                device_total_bytes=None,
                gtt_total_bytes=None,
                gtt_used_bytes=None,
                host_available_bytes=30 * 1024 * 1024 * 1024,
            )

            budget = resolve_residency_budget(demand, supply)

            expected = max(8, 2 * batch_size)
            assert budget.prefetch_entries == expected


class TestResidencyBudgetMetadata:
    """Test that metadata contains all inputs and intermediates."""

    def test_metadata_contains_all_inputs(self):
        """Metadata should contain all input values."""
        demand = ResidencyDemand(
            native_width=1296,
            native_height=968,
            tier_divisor=2,
            window_frames=200,
            batch_size=16,
        )
        supply = ResidencySupply(
            device_free_bytes=10 * 1024 * 1024 * 1024,
            device_total_bytes=None,
            gtt_total_bytes=None,
            gtt_used_bytes=None,
            host_available_bytes=40 * 1024 * 1024 * 1024,
        )

        budget = resolve_residency_budget(demand, supply)

        meta = budget.metadata
        assert meta["native_width"] == 1296
        assert meta["native_height"] == 968
        assert meta["tier_divisor"] == 2
        assert meta["window_frames"] == 200
        assert meta["batch_size"] == 16
        assert "native_bytes" in meta
        assert "tier_bytes" in meta
        assert "demand_hot" in meta
        assert "demand_cold" in meta
        assert "hot_bytes" in meta
        assert "cold_bytes" in meta
        assert "resident_frames" in meta
        assert "starved" in meta
        assert "window_effective" in meta
        assert "prefetch_entries" in meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
