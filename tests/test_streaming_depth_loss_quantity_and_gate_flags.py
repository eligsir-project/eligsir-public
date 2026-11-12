"""Confirms the new --streaming_depth_loss_quantity and
--streaming_depth_loss_alpha_gate flags exist, default to "d"/0.0
(today's exact behavior), and are settable independently via CLI parsing,
per Phase 1.5 experiment #1."""

from argparse import ArgumentParser
from arguments import StreamingParams


def test_flags_default_to_d_and_zero_gate():
    """Both flags should default to their no-op values."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([])
    assert args.streaming_depth_loss_quantity == "d"
    assert args.streaming_depth_loss_alpha_gate == 0.0


def test_quantity_flag_settable_independently_of_gate():
    """streaming_depth_loss_quantity can be set to 'ed' without affecting the gate."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args(["--streaming_depth_loss_quantity", "ed"])
    assert args.streaming_depth_loss_quantity == "ed"
    assert args.streaming_depth_loss_alpha_gate == 0.0  # unchanged


def test_gate_flag_settable_independently_of_quantity():
    """streaming_depth_loss_alpha_gate can be set without affecting the quantity."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args(["--streaming_depth_loss_alpha_gate", "0.1"])
    assert args.streaming_depth_loss_quantity == "d"  # unchanged
    assert args.streaming_depth_loss_alpha_gate == 0.1


def test_both_flags_settable_together():
    """Both flags can be set in the same invocation."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args(
        ["--streaming_depth_loss_quantity", "ed", "--streaming_depth_loss_alpha_gate", "0.15"]
    )
    assert args.streaming_depth_loss_quantity == "ed"
    assert args.streaming_depth_loss_alpha_gate == 0.15


def test_quantity_flag_accepts_d():
    """Explicitly test 'd' is accepted for streaming_depth_loss_quantity."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args(["--streaming_depth_loss_quantity", "d"])
    assert args.streaming_depth_loss_quantity == "d"


def test_gate_flag_accepts_various_floats():
    """Test gate flag accepts various float values."""
    parser = ArgumentParser()
    StreamingParams(parser)

    # Test with different float values
    args = parser.parse_args(["--streaming_depth_loss_alpha_gate", "0.25"])
    assert args.streaming_depth_loss_alpha_gate == 0.25

    args = parser.parse_args(["--streaming_depth_loss_alpha_gate", "0.5"])
    assert args.streaming_depth_loss_alpha_gate == 0.5

    args = parser.parse_args(["--streaming_depth_loss_alpha_gate", "1.0"])
    assert args.streaming_depth_loss_alpha_gate == 1.0


def test_quantity_flag_accepts_ed_alpha_weighted():
    """Issue #1833: 'ed_alpha_weighted' is a selectable quantity, default unchanged."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args(["--streaming_depth_loss_quantity", "ed_alpha_weighted"])
    assert args.streaming_depth_loss_quantity == "ed_alpha_weighted"
    assert args.streaming_depth_loss_alpha_gate == 0.0  # unchanged


def test_depth_downsample_flag_defaults_to_nearest_and_is_settable():
    """Issue #1833: --streaming_depth_loss_downsample defaults to 'nearest'
    (today's exact behavior) and accepts 'valid_area'."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([])
    assert args.streaming_depth_loss_downsample == "nearest"

    args = parser.parse_args(["--streaming_depth_loss_downsample", "valid_area"])
    assert args.streaming_depth_loss_downsample == "valid_area"


def test_refinement_depth_loss_weight_flag_defaults_to_negative_one_and_is_settable():
    """Issue #1833: --streaming_refinement_depth_loss_weight defaults to -1.0
    (reuse the mapping-phase weight unchanged) and accepts an override."""
    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([])
    assert args.streaming_refinement_depth_loss_weight == -1.0

    args = parser.parse_args(["--streaming_refinement_depth_loss_weight", "0.0"])
    assert args.streaming_refinement_depth_loss_weight == 0.0

    args = parser.parse_args(["--streaming_refinement_depth_loss_weight", "0.2"])
    assert args.streaming_refinement_depth_loss_weight == 0.2
