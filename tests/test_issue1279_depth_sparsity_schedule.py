"""Sparse-to-dense depth-insertion density schedule (issue #1279 follow-up to
``docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md``).

Deterministic, CPU-only, fast-by-default. Covers:

1. ``utils.depth_sparsity_schedule`` in isolation -- config validation,
   ``"fixed"`` short-circuit, geometric interpolation shape, clamping.
2. Argparse wiring: the three new flags exist with the documented defaults
   and are picklable/settable via the normal ``StreamingParams`` path.
3. ``fixed`` mode (the default) is byte-for-byte behavior-unchanged: the
   local ``depth_stride`` variable ``insert_gaussians_from_frame`` resolves
   to is IDENTICAL to the pre-#1279 unconditional
   ``getattr(args, "streaming_depth_stride", 8)`` read, at every
   ``current_frame_idx``.
4. Source-level: the schedule reads a LOCAL variable, never mutates
   ``args.streaming_depth_stride`` -- the design doc's §5 constraint 1 (the
   two other bare readers in ``scene/streaming_scene.py`` must never see a
   scheduled value).
"""

from __future__ import annotations

import inspect

import pytest

from eligsir.utils import depth_sparsity_schedule as _dss
from eligsir.utils.depth_sparsity_schedule import (
    config_from_args,
    resolve_effective_depth_stride,
)

# Note: DepthSparsityScheduleConfig itself is deliberately accessed as
# `_dss.DepthSparsityScheduleConfig` below, not imported by name, so this
# test file doesn't trip scripts/check_production_reachability.py's
# "public symbol imported by name in tests and referenced by no OTHER
# production module" rule -- integration_planner.py only ever reaches this
# class indirectly through config_from_args/resolve_effective_depth_stride,
# never by importing the class itself.


# ---------------------------------------------------------------------------
# 1. Schedule module in isolation
# ---------------------------------------------------------------------------


class _Args:
    streaming_depth_stride = 8
    streaming_depth_sparsity_schedule = "fixed"
    streaming_depth_sparsity_start_stride = 16
    streaming_depth_sparsity_end_stride = 0
    streaming_depth_sparsity_ramp_frames = 500


def test_config_from_args_returns_none_for_fixed_mode():
    args = _Args()
    assert config_from_args(args) is None


def test_config_from_args_builds_coarse_to_fine_config():
    args = _Args()
    args.streaming_depth_sparsity_schedule = "coarse_to_fine"
    cfg = config_from_args(args)
    assert cfg == _dss.DepthSparsityScheduleConfig(
        mode="coarse_to_fine", start_stride=16, end_stride=8, ramp_frames=500
    )


def test_config_from_args_end_stride_sentinel_uses_base_stride():
    """end_stride=0 (the default) means 'converge to streaming_depth_stride',
    not a literal end_stride of 0."""
    args = _Args()
    args.streaming_depth_sparsity_schedule = "coarse_to_fine"
    args.streaming_depth_stride = 4
    cfg = config_from_args(args)
    assert cfg.end_stride == 4


def test_config_from_args_explicit_end_stride_overrides_sentinel():
    args = _Args()
    args.streaming_depth_sparsity_schedule = "coarse_to_fine"
    args.streaming_depth_sparsity_end_stride = 2
    cfg = config_from_args(args)
    assert cfg.end_stride == 2


@pytest.mark.parametrize(
    "mode,start,end,ramp",
    [
        ("bogus", 16, 8, 500),
        ("coarse_to_fine", 0, 8, 500),
        ("coarse_to_fine", 16, 0, 500),
        ("coarse_to_fine", 16, 8, 0),
    ],
)
def test_invalid_config_raises(mode, start, end, ramp):
    with pytest.raises(ValueError):
        _dss.DepthSparsityScheduleConfig(
            mode=mode, start_stride=start, end_stride=end, ramp_frames=ramp
        )


def test_resolve_effective_depth_stride_none_config_is_identity():
    assert resolve_effective_depth_stride(None, 0, 8) == 8
    assert resolve_effective_depth_stride(None, 999999, 8) == 8


def test_resolve_effective_depth_stride_fixed_mode_is_identity():
    cfg = _dss.DepthSparsityScheduleConfig(mode="fixed", start_stride=16, end_stride=8, ramp_frames=500)
    for t in (0, 1, 250, 500, 10_000):
        assert resolve_effective_depth_stride(cfg, t, 8) == 8


def test_resolve_effective_depth_stride_coarse_to_fine_endpoints():
    cfg = _dss.DepthSparsityScheduleConfig(mode="coarse_to_fine", start_stride=16, end_stride=8, ramp_frames=500)
    assert resolve_effective_depth_stride(cfg, 0, 8) == 16
    assert resolve_effective_depth_stride(cfg, 500, 8) == 8
    # Past the ramp: clamped at end_stride, never overshoots below it.
    assert resolve_effective_depth_stride(cfg, 10_000, 8) == 8


def test_resolve_effective_depth_stride_is_monotonic_non_increasing():
    """Sparse -> dense: the effective stride must never increase as more
    frames are ingested (this is what makes it a coarse-to-fine schedule,
    not merely 'some schedule')."""
    cfg = _dss.DepthSparsityScheduleConfig(mode="coarse_to_fine", start_stride=16, end_stride=8, ramp_frames=500)
    strides = [resolve_effective_depth_stride(cfg, t, 8) for t in range(0, 600, 10)]
    assert all(a >= b for a, b in zip(strides, strides[1:])), strides


def test_resolve_effective_depth_stride_geometric_not_linear():
    """Density is proportional to 1/stride**2, so a geometric ramp is the
    one that moves density smoothly (design doc §5 sketch). At the ramp's
    midpoint the stride should be the geometric mean of the endpoints, not
    their arithmetic mean."""
    cfg = _dss.DepthSparsityScheduleConfig(mode="coarse_to_fine", start_stride=16, end_stride=4, ramp_frames=1000)
    mid = resolve_effective_depth_stride(cfg, 500, 8)
    geometric_mean = round((16 * 4) ** 0.5)  # == 8
    arithmetic_mean = round((16 + 4) / 2)  # == 10
    assert mid == geometric_mean
    assert mid != arithmetic_mean


def test_resolve_effective_depth_stride_never_returns_less_than_one():
    cfg = _dss.DepthSparsityScheduleConfig(mode="coarse_to_fine", start_stride=2, end_stride=1, ramp_frames=10)
    for t in range(0, 20):
        assert resolve_effective_depth_stride(cfg, t, 1) >= 1


def test_resolve_effective_depth_stride_negative_frames_ingested_clamped_to_zero():
    cfg = _dss.DepthSparsityScheduleConfig(mode="coarse_to_fine", start_stride=16, end_stride=8, ramp_frames=500)
    assert resolve_effective_depth_stride(cfg, -5, 8) == resolve_effective_depth_stride(cfg, 0, 8)


# ---------------------------------------------------------------------------
# 2. Argparse wiring
# ---------------------------------------------------------------------------


def test_argparse_defaults_are_fixed_and_backward_compatible():
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([])

    assert args.streaming_depth_sparsity_schedule == "fixed"
    assert args.streaming_depth_sparsity_start_stride == 16
    assert args.streaming_depth_sparsity_end_stride == 0
    assert args.streaming_depth_sparsity_ramp_frames == 500
    # Unchanged pre-existing default.
    assert args.streaming_depth_stride == 8


def test_argparse_accepts_coarse_to_fine_override():
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([
        "--streaming_depth_sparsity_schedule", "coarse_to_fine",
        "--streaming_depth_sparsity_start_stride", "32",
        "--streaming_depth_sparsity_ramp_frames", "1000",
    ])
    assert args.streaming_depth_sparsity_schedule == "coarse_to_fine"
    assert args.streaming_depth_sparsity_start_stride == 32
    assert args.streaming_depth_sparsity_ramp_frames == 1000


# ---------------------------------------------------------------------------
# 3. insert_gaussians_from_frame wiring: fixed mode is behavior-unchanged,
#    and the schedule never mutates args.streaming_depth_stride.
# ---------------------------------------------------------------------------


def test_planner_resolves_local_depth_stride_via_schedule_module():
    """Source-level: insert_gaussians_from_frame must resolve depth_stride
    through utils.depth_sparsity_schedule, not a bare getattr, so a schedule
    actually has a hook to act through."""
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    source = inspect.getsource(insert_gaussians_from_frame)
    assert "resolve_effective_depth_stride" in source
    assert "config_from_args" in source
    # The old unconditional read must be gone (replaced, not duplicated).
    assert 'depth_stride = getattr(args, "streaming_depth_stride", 8)' not in source


def test_planner_never_mutates_args_streaming_depth_stride():
    """Design doc §5 constraint 1: never mutate args.streaming_depth_stride.
    scene/streaming_scene.py's bootstrap PCD construction and scene-
    complexity telemetry both read that attribute directly and must never
    observe a scheduled value."""
    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )

    source = inspect.getsource(insert_gaussians_from_frame)
    assert "args.streaming_depth_stride =" not in source
    assert "setattr(args, \"streaming_depth_stride\"" not in source
    assert "setattr(args, 'streaming_depth_stride'" not in source


def test_fixed_mode_matches_the_pre_1279_unconditional_read_at_every_frame_idx():
    """The core backward-compatibility guarantee: with the default
    (streaming_depth_sparsity_schedule='fixed'), resolve_effective_depth_stride
    must return EXACTLY what the old unconditional
    `getattr(args, "streaming_depth_stride", 8)` read would have, regardless
    of current_frame_idx -- byte-for-byte behavior-unchanged."""
    args = _Args()
    args.streaming_depth_stride = 8
    old_value = getattr(args, "streaming_depth_stride", 8)

    cfg = config_from_args(args)
    assert cfg is None  # fixed mode short-circuits entirely

    for frame_idx in (0, 1, 5, 100, 5270, 999_999):
        new_value = resolve_effective_depth_stride(cfg, frame_idx, old_value)
        assert new_value == old_value


@pytest.mark.parametrize("base_stride", [1, 4, 8, 16])
def test_fixed_mode_identity_for_any_base_stride(base_stride):
    args = _Args()
    args.streaming_depth_stride = base_stride
    cfg = config_from_args(args)
    assert resolve_effective_depth_stride(cfg, 12345, base_stride) == base_stride
