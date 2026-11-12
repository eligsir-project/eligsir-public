"""Test suite for issue #834: resolve single source-sampling owner.

Validates that source-sampling policy resolution chooses exactly one owner
for applying stride, preventing accidental multiplication of frame_stride × source_stride.
"""

import pytest

from eligsir.mapping.scheduler.frame_scheduler import (
    resolve_source_sampling_policy,
    resolve_source_decimation,
    resolve_legacy_composition_mode,
    ConflictingStrideError,
    SOURCE_SAMPLING_MODE_LEGACY_COMPOSE,
    SOURCE_SAMPLING_MODE_SINGLE_OWNER,
)


class TestResolveSourceSamplingPolicy:
    """Test the new source sampling policy resolver."""

    def test_both_1_no_sampling(self):
        """Rule 1: both 1 -> no sampling needed."""
        policy = resolve_source_sampling_policy(1, 1)
        assert policy["effective_stride"] == 1
        assert policy["owner"] is None
        assert policy["legacy_composition_active"] is False

    def test_only_frame_stride_non1(self):
        """Rule 2a: only frame_stride is non-1."""
        for stride in [2, 3, 5, 10]:
            policy = resolve_source_sampling_policy(stride, 1)
            assert policy["effective_stride"] == stride
            assert policy["owner"] == "native"
            assert policy["legacy_composition_active"] is False

    def test_only_source_stride_non1(self):
        """Rule 2b: only source_stride is non-1."""
        for stride in [2, 3, 5, 10]:
            policy = resolve_source_sampling_policy(1, stride)
            assert policy["effective_stride"] == stride
            assert policy["owner"] == "generic"
            assert policy["legacy_composition_active"] is False

    def test_both_equal_native_owner(self):
        """Rule 3: both non-1 and equal -> native owner."""
        for stride in [2, 3, 5, 10]:
            policy = resolve_source_sampling_policy(stride, stride)
            assert policy["effective_stride"] == stride
            assert policy["owner"] == "native"
            assert policy["legacy_composition_active"] is False

    def test_both_nonequal_raises_in_canonical_mode(self):
        """Rule 4a: both non-1 and conflicting -> raise in canonical mode."""
        with pytest.raises(ConflictingStrideError) as exc_info:
            resolve_source_sampling_policy(2, 3)
        assert "2" in str(exc_info.value)
        assert "3" in str(exc_info.value)
        assert "legacy_compose" in str(exc_info.value)

    def test_both_nonequal_legacy_mode_multiplies(self):
        """Rule 4b: both non-1 and conflicting -> multiply in legacy mode."""
        policy = resolve_source_sampling_policy(2, 3, legacy_composition_mode=True)
        assert policy["effective_stride"] == 6  # 2 * 3
        assert policy["owner"] == "both"
        assert policy["legacy_composition_active"] is True
        assert policy["conflict"] == "frame_stride=2 vs source_stride=3"

    def test_none_stride_treated_as_1(self):
        """None and 0 values are normalized to 1."""
        policy = resolve_source_sampling_policy(None, None)
        assert policy["effective_stride"] == 1
        assert policy["owner"] is None

        policy = resolve_source_sampling_policy(0, 0)
        assert policy["effective_stride"] == 1
        assert policy["owner"] is None

        policy = resolve_source_sampling_policy(2, None)
        assert policy["effective_stride"] == 2
        assert policy["owner"] == "native"

    def test_large_strides(self):
        """Large stride values are handled correctly."""
        policy = resolve_source_sampling_policy(100, 100)
        assert policy["effective_stride"] == 100
        assert policy["owner"] == "native"

        policy = resolve_source_sampling_policy(1, 1000)
        assert policy["effective_stride"] == 1000
        assert policy["owner"] == "generic"


class TestResolveSourceDecimation:
    """Test the extended resolve_source_decimation with new fields."""

    def test_backward_compat_no_sampling(self):
        """All original 7 keys are present when no sampling."""
        result = resolve_source_decimation(1, 1)
        original_keys = {
            "frame_stride",
            "source_stride",
            "effective_stride",
            "composed",
            "frames_before_source_stride",
            "frames_after_source_stride",
            "frames_before_is_placeholder",
        }
        assert original_keys.issubset(result.keys())
        assert result["frame_stride"] == 1
        assert result["source_stride"] == 1
        assert result["composed"] is False

    def test_backward_compat_with_sampling(self):
        """Backward compat: effective_stride is correct even with sampling."""
        result = resolve_source_decimation(2, 1)
        assert result["effective_stride"] == 2
        assert result["composed"] is False  # only one > 1
        assert result["sampling_owner"] == "native"

    def test_both_nonequal_fails_by_default(self):
        """Both non-1 and different strides fail in canonical mode."""
        with pytest.raises(ConflictingStrideError):
            resolve_source_decimation(2, 3)

    def test_both_nonequal_legacy_mode(self):
        """Both non-1 and different strides work in legacy mode."""
        result = resolve_source_decimation(2, 3, legacy_composition_mode=True)
        assert result["effective_stride"] == 6  # LEGACY: multiply
        assert result["legacy_composition_active"] is True
        assert result["composed"] is True  # both were > 1
        assert result["sampling_owner"] == "both"

    def test_new_sampling_owner_field_populated(self):
        """New field sampling_owner correctly reflects which owner applies stride."""
        # Native owner
        result = resolve_source_decimation(5, 1)
        assert result["sampling_owner"] == "native"

        # Generic owner
        result = resolve_source_decimation(1, 5)
        assert result["sampling_owner"] == "generic"

        # No owner
        result = resolve_source_decimation(1, 1)
        assert result["sampling_owner"] is None

    def test_frames_provenance_recorded(self):
        """Frame count provenance is recorded correctly."""
        result = resolve_source_decimation(
            2, 1,
            frames_before=100,
            frames_after=50,
            frames_source="pre_stride",
        )
        assert result["frames_before_source_stride"] == 100
        assert result["frames_after_source_stride"] == 50
        assert result["frames_before_is_placeholder"] is False

        result = resolve_source_decimation(
            2, 1,
            frames_before=999999,
            frames_source="placeholder",
        )
        assert result["frames_before_is_placeholder"] is True


class TestNegativeControlMultiplicationPrevention:
    """Negative control: verify the fix actually prevents multiplication."""

    def test_old_behavior_would_multiply(self):
        """Demonstrate that old code would have multiplied."""
        # OLD CODE (hypothetical, not actually present):
        # if frame_stride > 1: apply frame_stride
        # if source_stride > 1: apply source_stride
        # Result: frame_stride * source_stride effective stride

        # NEW CODE: exactly one owner applies - should raise!
        with pytest.raises(ConflictingStrideError):
            resolve_source_sampling_policy(2, 3, legacy_composition_mode=False)

    def test_only_one_owner_applies_stride(self):
        """Verify exactly one owner applies the stride."""
        # Single stride set: that one owner applies it
        policy = resolve_source_sampling_policy(2, 1)
        assert policy["owner"] == "native"
        assert policy["effective_stride"] == 2

        # Only source stride set: generic owner applies it
        policy = resolve_source_sampling_policy(1, 2)
        assert policy["owner"] == "generic"
        assert policy["effective_stride"] == 2

        # Equal strides: native owner applies (preferred earliest layer)
        policy = resolve_source_sampling_policy(2, 2)
        assert policy["owner"] == "native"
        assert policy["effective_stride"] == 2

    def test_both_nonequal_requires_explicit_legacy_flag(self):
        """Conflicting strides are rejected unless explicitly legacyified."""
        with pytest.raises(ConflictingStrideError):
            resolve_source_sampling_policy(2, 3, legacy_composition_mode=False)

        # Explicit legacy flag allows it
        policy = resolve_source_sampling_policy(2, 3, legacy_composition_mode=True)
        assert policy["legacy_composition_active"] is True
        assert policy["effective_stride"] == 6


class TestRealSpecCompatibility:
    """Verify existing real specs don't break."""

    def test_15_lever_ablation_frame_stride_only(self):
        """human_launch_scripts/15_lever_ablation_fr3.sh uses frame_stride 3 or 5."""
        # These should continue working unchanged
        policy = resolve_source_sampling_policy(3, 1)
        assert policy["effective_stride"] == 3
        assert policy["owner"] == "native"

        policy = resolve_source_sampling_policy(5, 1)
        assert policy["effective_stride"] == 5
        assert policy["owner"] == "native"

    def test_16_stride_kf_frame_stride_only(self):
        """human_launch_scripts/16_stride_kf_fr3.sh uses frame_stride 3 or 5."""
        for stride in [3, 5]:
            policy = resolve_source_sampling_policy(stride, 1)
            assert policy["effective_stride"] == stride
            assert policy["owner"] == "native"
            assert policy["legacy_composition_active"] is False

    def test_18_floor_search_source_stride_only(self):
        """human_launch_scripts/18_floor_search_fr3.sh uses source_stride 3."""
        policy = resolve_source_sampling_policy(1, 3)
        assert policy["effective_stride"] == 3
        assert policy["owner"] == "generic"
        assert policy["legacy_composition_active"] is False


class TestResolveLegacyCompositionMode:
    """Test the --streaming_source_sampling_mode -> legacy_composition_mode
    bridge (issue #834 review defect #3: the CLI flag the error message names
    must actually exist and actually work)."""

    def test_default_none_is_single_owner(self):
        assert resolve_legacy_composition_mode(None) is False

    def test_single_owner_string(self):
        assert resolve_legacy_composition_mode(SOURCE_SAMPLING_MODE_SINGLE_OWNER) is False
        assert resolve_legacy_composition_mode("single_owner") is False

    def test_legacy_compose_string(self):
        assert resolve_legacy_composition_mode(SOURCE_SAMPLING_MODE_LEGACY_COMPOSE) is True
        assert resolve_legacy_composition_mode("legacy_compose") is True

    def test_unrecognized_value_raises_loudly(self):
        """A typo must fail loudly, not silently fall back to single_owner."""
        with pytest.raises(ValueError) as exc_info:
            resolve_legacy_composition_mode("legacy_composed")
        assert "legacy_composed" in str(exc_info.value)

    def test_error_message_names_a_flag_value_that_actually_works(self):
        """Defect #3: ConflictingStrideError told operators to pass
        --streaming_source_sampling_mode legacy_compose. Verify that literal
        value round-trips through resolve_legacy_composition_mode into the
        exact behavior the error message promises (multiplication allowed)."""
        with pytest.raises(ConflictingStrideError) as exc_info:
            resolve_source_sampling_policy(2, 3)
        message = str(exc_info.value)
        assert "--streaming_source_sampling_mode legacy_compose" in message

        # Extract the literal value the message tells the operator to pass,
        # and confirm it is a real, working mode -- not a name that only
        # exists in the error string.
        assert resolve_legacy_composition_mode("legacy_compose") is True
        policy = resolve_source_sampling_policy(
            2, 3, legacy_composition_mode=resolve_legacy_composition_mode("legacy_compose")
        )
        assert policy["effective_stride"] == 6
        assert policy["owner"] == "both"


class TestStreamingSourceSamplingModeCLIFlag:
    """Defect #3: the flag must be a real, parseable CLI argument, not just a
    string named in an error message."""

    def _parse(self, argv):
        import argparse
        from arguments import StreamingParams

        parser = argparse.ArgumentParser()
        StreamingParams(parser)
        return parser.parse_args(argv)

    def test_default_is_single_owner(self):
        args = self._parse([])
        assert args.streaming_source_sampling_mode == SOURCE_SAMPLING_MODE_SINGLE_OWNER

    def test_flag_accepts_legacy_compose(self):
        args = self._parse(["--streaming_source_sampling_mode", "legacy_compose"])
        assert args.streaming_source_sampling_mode == "legacy_compose"
        assert resolve_legacy_composition_mode(args.streaming_source_sampling_mode) is True

    def test_flag_accepts_single_owner_explicitly(self):
        args = self._parse(["--streaming_source_sampling_mode", "single_owner"])
        assert args.streaming_source_sampling_mode == "single_owner"
        assert resolve_legacy_composition_mode(args.streaming_source_sampling_mode) is False


class TestSourceDecimationHonorsSamplingMode:
    """resolve_source_decimation must be told the real sampling mode -- a
    caller that forgets to pass legacy_composition_mode silently loses the
    escape hatch and raises even when the operator explicitly asked for
    legacy_compose."""

    def test_legacy_compose_mode_flows_through(self):
        result = resolve_source_decimation(
            2,
            3,
            legacy_composition_mode=resolve_legacy_composition_mode("legacy_compose"),
        )
        assert result["effective_stride"] == 6
        assert result["legacy_composition_active"] is True

    def test_default_mode_still_raises(self):
        with pytest.raises(ConflictingStrideError):
            resolve_source_decimation(
                2,
                3,
                legacy_composition_mode=resolve_legacy_composition_mode(None),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
