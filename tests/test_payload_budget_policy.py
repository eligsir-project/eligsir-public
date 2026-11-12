"""Unit tests for resolution- and device-aware payload budget defaults.

Tests the derivation of admitted-payload frame count and hot-tier byte cap
from native source resolution and device memory.
"""

from __future__ import annotations

from types import SimpleNamespace
import pytest


class TestResolutionScaledBudgetDerivation:
    """Test resolution-scaled payload budget derivation."""

    def test_640x480_baseline_yields_512_frames(self):
        """Baseline 640x480 (TUM) should resolve to 512 frames."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=640, native_height=480
        )
        assert frames == 512
        assert meta["source"] == "resolution_scaled"
        assert meta["native_resolution"] == "640x480"

    def test_1280x720_yields_170_frames(self):
        """1280x720 should scale down to ~171 frames."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1280, native_height=720
        )
        # 512 * (640*480) / (1280*720) ≈ 170.67 → 171
        assert frames == 171, f"Expected 171, got {frames}"
        assert meta["source"] == "resolution_scaled"

    def test_1296x968_yields_125_frames(self):
        """1296x968 (ScanNet v2) should scale down to ~125 frames."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1296, native_height=968
        )
        # 512 * (640*480) / (1296*968) ≈ 125.09 → 125
        assert frames == 125, f"Expected 125, got {frames}"

    def test_1920x1440_clamped_to_64_minimum(self):
        """Very large resolution should clamp to 64 minimum."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1920, native_height=1440
        )
        # 512 * (640*480) / (1920*1440) ≈ 56.9 → 57, but should clamp to 64
        assert frames == 64, f"Expected 64 (minimum), got {frames}"

    def test_explicit_frame_flag_wins_over_resolution_scaled(self):
        """Explicit --streaming_live_admitted_payload_max_frames should override policy."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=256,  # Explicit
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1296, native_height=968
        )
        assert frames == 256
        assert meta["source"] == "explicit_flag"

    def test_legacy_policy_returns_profile_default_frames(self):
        """Legacy policy should use hardcoded 512 frames default."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="legacy",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1296, native_height=968
        )
        assert frames == 512
        assert meta["source"] == "legacy_policy_default"

    def test_legacy_policy_uses_zero_hot_tier_bytes(self):
        """Legacy policy should use 0 (unbounded) hot tier bytes."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="legacy",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(args)
        assert hot_bytes == 0
        assert meta["hot_tier_source"] == "legacy_policy_default"

    def test_resolution_scaled_derives_hot_tier_from_device_memory(self):
        """Resolution scaled should derive 20% of device memory for hot tier."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        total_memory_bytes = 12_000_000_000  # 12 GB

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )

        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, probe_device_memory=lambda: total_memory_bytes
        )
        # 20% of 12 GB = 2.4 GB
        expected_hot_bytes = int(0.20 * total_memory_bytes)
        assert hot_bytes == expected_hot_bytes
        assert meta["hot_tier_source"] == "20pct_device_memory"
        assert meta["device_total_memory"] == total_memory_bytes

    def test_no_cuda_device_falls_back_to_unbounded_hot_tier(self):
        """When no CUDA device, fall back to 0 (unbounded) hot tier bytes."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )

        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, probe_device_memory=lambda: None
        )
        assert hot_bytes == 0
        assert meta["hot_tier_source"] == "no_cuda_fallback"

    def test_explicit_hot_tier_bytes_flag_wins_over_resolution_scaled(self):
        """Explicit --streaming_hot_tier_max_bytes should override policy."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=3_000_000_000,  # Explicit 3 GB
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(args)
        assert hot_bytes == 3_000_000_000
        assert meta["hot_tier_source"] == "explicit_flag"

    def test_unknown_resolution_fallback_uses_legacy_default(self):
        """When resolution is unknown, fallback to legacy default."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=None, native_height=None
        )
        assert frames == 512
        assert meta["source"] == "unknown_resolution_fallback"

    def test_metadata_includes_resolved_values(self):
        """Metadata dict should include all resolved values for logging."""
        from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

        args = SimpleNamespace(
            streaming_payload_budget_policy="resolution_scaled",
            streaming_live_admitted_payload_max_frames=None,
            streaming_hot_tier_max_bytes=None,
        )
        frames, hot_bytes, meta = resolve_admitted_payload_bounds(
            args, native_width=1296, native_height=968
        )
        assert meta["policy"] == "resolution_scaled"
        assert meta["native_resolution"] == "1296x968"
        assert meta["admitted_payload_max_frames"] == frames
        assert meta["hot_tier_max_bytes"] == hot_bytes
        assert meta["source"] == "resolution_scaled"
        assert "hot_tier_source" in meta


class TestPayloadBudgetIntegration:
    """Test payload budget integration with config_boundary."""

    @pytest.mark.skip(reason="Requires full torch/model initialization")
    def test_config_boundary_uses_resolved_values_when_available(self):
        """config_boundary should use resolved values from args._resolved_*."""
        # This test is skipped as it requires full torch and model initialization
        # The resolver function is tested above in isolation
        pass


class TestSourceNativeResolutionFallbacks:
    """Test source_native_resolution helper with multiple fallback paths."""

    def test_native_resolution_attr_takes_priority(self):
        """Explicit native_resolution attribute should be preferred."""
        from eligsir.runtime.config_boundary import source_native_resolution

        source = SimpleNamespace(native_resolution=(1920, 1440))
        res, method = source_native_resolution(source)
        assert res == (1920, 1440)
        assert method == "native_resolution_attr"

    def test_source_width_height_attributes_fallback(self):
        """Source.width/height attributes should be used if native_resolution missing."""
        from eligsir.runtime.config_boundary import source_native_resolution

        source = SimpleNamespace(width=1280, height=720)
        res, method = source_native_resolution(source)
        assert res == (1280, 720)
        assert method == "source_width_height"

    def test_first_frame_record_fallback(self):
        """First frame record's width/height should be used if source attrs missing."""
        from eligsir.runtime.config_boundary import source_native_resolution

        frame = SimpleNamespace(width=1296, height=968)
        source = SimpleNamespace(_frames=[frame])
        res, method = source_native_resolution(source)
        assert res == (1296, 968)
        assert method == "first_frame_record"

    def test_pil_image_header_fallback(self):
        """PIL should read image header from first frame's rgb_path as last resort."""
        from eligsir.runtime.config_boundary import source_native_resolution
        from tempfile import NamedTemporaryFile
        from PIL import Image

        # Create a temporary test image
        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (800, 600))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            frame = SimpleNamespace(rgb_path=tmp_path)
            source = SimpleNamespace(_frames=[frame])
            res, method = source_native_resolution(source)
            assert res == (800, 600)
            assert method == "pil_image_header"
        finally:
            import os
            os.unlink(tmp_path)

    def test_empty_frames_list_returns_unknown(self):
        """Empty _frames list should not raise, return unknown."""
        from eligsir.runtime.config_boundary import source_native_resolution

        source = SimpleNamespace(_frames=[])
        res, method = source_native_resolution(source)
        assert res is None
        assert method == "unknown"

    def test_no_frames_attribute_returns_unknown(self):
        """Source with no _frames attribute should fall through to unknown."""
        from eligsir.runtime.config_boundary import source_native_resolution

        source = SimpleNamespace()  # Empty namespace
        res, method = source_native_resolution(source)
        assert res is None
        assert method == "unknown"

    def test_frame_without_width_height_returns_unknown(self):
        """Frame without width/height attributes should skip that frame."""
        from eligsir.runtime.config_boundary import source_native_resolution

        frame = SimpleNamespace()  # No width/height
        source = SimpleNamespace(_frames=[frame])
        res, method = source_native_resolution(source)
        assert res is None
        assert method == "unknown"

    def test_exception_in_native_resolution_does_not_crash(self):
        """Exception reading native_resolution should be caught and continue."""
        from eligsir.runtime.config_boundary import source_native_resolution

        class BadSource:
            @property
            def native_resolution(self):
                raise RuntimeError("intentional error")

            width = 640
            height = 480

        source = BadSource()
        res, method = source_native_resolution(source)
        # Should fall through to source_width_height
        assert res == (640, 480)
        assert method == "source_width_height"

    def test_fallback_priority_order(self):
        """Test that fallback order is respected: native_resolution > width/height > _frames."""
        from eligsir.runtime.config_boundary import source_native_resolution

        # All three levels present: native_resolution should win
        frame = SimpleNamespace(width=100, height=100)
        source = SimpleNamespace(
            native_resolution=(1920, 1440),
            width=1280,
            height=720,
            _frames=[frame]
        )
        res, method = source_native_resolution(source)
        assert res == (1920, 1440)
        assert method == "native_resolution_attr"

        # Remove native_resolution: width/height should win
        delattr(source, "native_resolution")
        res, method = source_native_resolution(source)
        assert res == (1280, 720)
        assert method == "source_width_height"

        # Remove width/height: _frames should win
        delattr(source, "width")
        delattr(source, "height")
        res, method = source_native_resolution(source)
        assert res == (100, 100)
        assert method == "first_frame_record"


class TestPayloadBudgetHardCut:
    """Test that payload budget resolution is properly wired in causal_entry.py."""

    def test_causal_entry_references_resolver(self):
        """Causal entry must call resolve_admitted_payload_bounds before config creation."""
        from pathlib import Path

        # Verify causal_entry imports and calls the resolver
        causal_entry_path = Path(__file__).parents[1] / "eligsir/runtime/causal_entry.py"
        source = causal_entry_path.read_text()

        # Must import the resolver inside the function
        assert "from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds" in source, (
            "causal_entry.py must import resolve_admitted_payload_bounds in resolve_causal_mapping_composition"
        )

        # Must be in the composition resolver (not just imported)
        assert "resolve_causal_mapping_composition" in source
        comp_section = source.split("def resolve_causal_mapping_composition")[1].split("def ")[0]
        assert "resolve_admitted_payload_bounds(" in comp_section, (
            "resolve_admitted_payload_bounds must be called in resolve_causal_mapping_composition"
        )

        # Must call it BEFORE mapping_runtime_config_from_args
        resolver_index = comp_section.find("resolve_admitted_payload_bounds(")
        config_call_index = comp_section.find("config = mapping_runtime_config_from_args(")
        assert resolver_index > 0 and config_call_index > 0, (
            "Could not find function calls in source"
        )
        assert resolver_index < config_call_index, (
            "resolve_admitted_payload_bounds must be called BEFORE mapping_runtime_config_from_args"
        )

    def test_train_streaming_can_call_resolver_optionally(self):
        """train_streaming can call resolver (it's optional for causal runs)."""
        from pathlib import Path

        train_streaming_path = Path(__file__).parents[1] / "train_streaming.py"
        source = train_streaming_path.read_text()

        # Should import and call the resolver
        assert "resolve_admitted_payload_bounds" in source, (
            "train_streaming.py should import resolve_admitted_payload_bounds"
        )

        # Should log the same way
        assert "[payload-budget]" in source, (
            "train_streaming.py should log [payload-budget] metadata"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
