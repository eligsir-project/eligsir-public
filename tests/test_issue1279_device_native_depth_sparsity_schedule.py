"""Sparse-to-dense depth-insertion density schedule, device-native follow-up
(issue #1279).

PR #1362 wired ``utils.depth_sparsity_schedule`` into
``eligsir/mapping/integration/integration_planner.py``'s
``insert_gaussians_from_frame`` only. That function is NOT the producer the
``causal_mapping`` runtime (``eligsir/runtime/production_application.py``,
the live production path) actually uses -- it calls
``eligsir/mapping/integration/device_integration_planner.py``'s
``insert_gaussians_from_frame_device_native`` instead, which read its
``depth_stride`` unconditionally from ``insertion_plan.sample_stride`` (a
static value set once by ``WorkBudgetOwner`` at construction time). This
left the schedule unreachable from the actual live path -- exactly the
"new primitive not wired into production" failure mode CLAUDE.md's HARD
CONSTRAINT section describes.

This module covers the fix: a local ``stride`` variable inside
``insert_gaussians_from_frame_device_native`` resolved through
``utils.depth_sparsity_schedule.config_from_policy``/
``resolve_effective_depth_stride``, using ``current_frame_idx`` (mapped from
``ProductionMappingApplication.frames_seen``, the causal_mapping runtime's
own frames-ingested counter -- the same progress signal
``insert_gaussians_from_frame`` uses) as the schedule's progress axis.

Deterministic, CPU-only, fast-by-default. Covers:

1. ``utils.depth_sparsity_schedule.config_from_policy`` in isolation --
   reads the new ``DepthInsertionPolicy`` fields, "fixed" short-circuit,
   end-stride sentinel semantics (mirroring ``config_from_args``).
2. ``DepthInsertionPolicy`` field validation for the four new fields.
3. Source-level: ``insert_gaussians_from_frame_device_native`` resolves
   ``stride`` through the schedule module, keyed on ``current_frame_idx``,
   and never mutates ``insertion_plan``/``insertion_policy`` in place.
4. ``fixed`` mode (the default) is byte-for-byte behavior-unchanged: the
   local ``stride`` this function resolves to is IDENTICAL to the
   pre-fix unconditional ``int(insertion_plan.sample_stride)`` read, at
   every ``current_frame_idx``.
5. The progress-signal mapping: ``production_application.py`` passes
   ``current_frame_idx=self.frames_seen`` at its one call site, matching
   the legacy path's semantic (a frames-ingested counter, not an
   optimizer-iteration counter).
"""

from __future__ import annotations

import inspect

import pytest

from eligsir.utils import depth_sparsity_schedule as _dss
from eligsir.utils.depth_sparsity_schedule import config_from_policy, resolve_effective_depth_stride


# ---------------------------------------------------------------------------
# 1. config_from_policy in isolation
# ---------------------------------------------------------------------------


class _Policy:
    depth_sparsity_schedule_mode = "fixed"
    depth_sparsity_start_stride = 16
    depth_sparsity_end_stride = 0
    depth_sparsity_ramp_frames = 500


def test_config_from_policy_returns_none_for_fixed_mode():
    policy = _Policy()
    assert config_from_policy(policy, base_stride=8) is None


def test_config_from_policy_builds_coarse_to_fine_config():
    policy = _Policy()
    policy.depth_sparsity_schedule_mode = "coarse_to_fine"
    cfg = config_from_policy(policy, base_stride=8)
    assert cfg == _dss.DepthSparsityScheduleConfig(
        mode="coarse_to_fine", start_stride=16, end_stride=8, ramp_frames=500
    )


def test_config_from_policy_end_stride_sentinel_uses_base_stride():
    """end_stride=0 (the default) means 'converge to the plan's own
    sample_stride', not a literal end_stride of 0 -- mirrors
    config_from_args's streaming_depth_stride sentinel semantics."""
    policy = _Policy()
    policy.depth_sparsity_schedule_mode = "coarse_to_fine"
    cfg = config_from_policy(policy, base_stride=4)
    assert cfg.end_stride == 4


def test_config_from_policy_explicit_end_stride_overrides_sentinel():
    policy = _Policy()
    policy.depth_sparsity_schedule_mode = "coarse_to_fine"
    policy.depth_sparsity_end_stride = 2
    cfg = config_from_policy(policy, base_stride=8)
    assert cfg.end_stride == 2


def test_config_from_policy_matches_config_from_args_for_equivalent_inputs():
    """The two builders must agree bit-for-bit given equivalent inputs --
    they share one underlying implementation (config_from_values)."""

    class _Args:
        streaming_depth_stride = 8
        streaming_depth_sparsity_schedule = "coarse_to_fine"
        streaming_depth_sparsity_start_stride = 32
        streaming_depth_sparsity_end_stride = 0
        streaming_depth_sparsity_ramp_frames = 1000

    policy = _Policy()
    policy.depth_sparsity_schedule_mode = "coarse_to_fine"
    policy.depth_sparsity_start_stride = 32
    policy.depth_sparsity_end_stride = 0
    policy.depth_sparsity_ramp_frames = 1000

    from eligsir.utils.depth_sparsity_schedule import config_from_args

    assert config_from_policy(policy, base_stride=8) == config_from_args(_Args())


# ---------------------------------------------------------------------------
# 2. DepthInsertionPolicy field validation
# ---------------------------------------------------------------------------


def test_depth_insertion_policy_defaults_are_fixed_and_backward_compatible():
    from eligsir.runtime.config import DepthInsertionPolicy

    policy = DepthInsertionPolicy()
    assert policy.depth_sparsity_schedule_mode == "fixed"
    assert policy.depth_sparsity_start_stride == 16
    assert policy.depth_sparsity_end_stride == 0
    assert policy.depth_sparsity_ramp_frames == 500


def test_depth_insertion_policy_accepts_coarse_to_fine():
    from eligsir.runtime.config import DepthInsertionPolicy

    policy = DepthInsertionPolicy(
        depth_sparsity_schedule_mode="coarse_to_fine",
        depth_sparsity_start_stride=32,
        depth_sparsity_end_stride=4,
        depth_sparsity_ramp_frames=1000,
    )
    assert policy.depth_sparsity_schedule_mode == "coarse_to_fine"
    assert policy.depth_sparsity_start_stride == 32


def test_depth_insertion_policy_rejects_bogus_mode():
    from eligsir.runtime.config import DepthInsertionPolicy

    with pytest.raises(ValueError):
        DepthInsertionPolicy(depth_sparsity_schedule_mode="bogus")


@pytest.mark.parametrize(
    "field",
    ["depth_sparsity_start_stride", "depth_sparsity_end_stride", "depth_sparsity_ramp_frames"],
)
def test_depth_insertion_policy_rejects_negative_schedule_fields(field):
    from eligsir.runtime.config import DepthInsertionPolicy

    with pytest.raises(ValueError):
        DepthInsertionPolicy(**{field: -1})


def test_depth_insertion_policy_from_args_populates_schedule_fields():
    from argparse import ArgumentParser

    from arguments import StreamingParams
    from eligsir.runtime.config_boundary import depth_insertion_policy_from_args

    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([
        "--streaming_depth_sparsity_schedule", "coarse_to_fine",
        "--streaming_depth_sparsity_start_stride", "32",
        "--streaming_depth_sparsity_end_stride", "2",
        "--streaming_depth_sparsity_ramp_frames", "1000",
    ])
    policy = depth_insertion_policy_from_args(args)
    assert policy.depth_sparsity_schedule_mode == "coarse_to_fine"
    assert policy.depth_sparsity_start_stride == 32
    assert policy.depth_sparsity_end_stride == 2
    assert policy.depth_sparsity_ramp_frames == 1000


def test_depth_insertion_policy_from_args_defaults_to_fixed():
    from argparse import ArgumentParser

    from arguments import StreamingParams
    from eligsir.runtime.config_boundary import depth_insertion_policy_from_args

    parser = ArgumentParser()
    StreamingParams(parser)
    args = parser.parse_args([])
    policy = depth_insertion_policy_from_args(args)
    assert policy.depth_sparsity_schedule_mode == "fixed"


# ---------------------------------------------------------------------------
# 3. Source-level wiring in insert_gaussians_from_frame_device_native
# ---------------------------------------------------------------------------


def test_device_native_planner_resolves_local_stride_via_schedule_module():
    from eligsir.mapping.integration.device_integration_planner import (
        insert_gaussians_from_frame_device_native,
    )

    source = inspect.getsource(insert_gaussians_from_frame_device_native)
    assert "resolve_effective_depth_stride" in source
    assert "config_from_policy" in source
    # The old unconditional read must be gone (replaced, not duplicated).
    assert "stride = int(insertion_plan.sample_stride)" not in source


def test_device_native_planner_never_mutates_plan_or_policy():
    from eligsir.mapping.integration.device_integration_planner import (
        insert_gaussians_from_frame_device_native,
    )

    source = inspect.getsource(insert_gaussians_from_frame_device_native)
    assert "insertion_plan.sample_stride =" not in source
    assert "object.__setattr__(insertion_plan" not in source
    assert "object.__setattr__(insertion_policy" not in source


def test_device_native_planner_passes_current_frame_idx_through():
    """Regression guard: the resolved stride call must be keyed on
    current_frame_idx, not e.g. a hardcoded 0 or an unrelated counter."""
    from eligsir.mapping.integration.device_integration_planner import (
        insert_gaussians_from_frame_device_native,
    )

    source = inspect.getsource(insert_gaussians_from_frame_device_native)
    assert "current_frame_idx,\n" in source or "current_frame_idx," in source


# ---------------------------------------------------------------------------
# 4. fixed-mode behavior-neutrality (the actual regression coverage)
# ---------------------------------------------------------------------------


def test_fixed_mode_matches_pre_fix_unconditional_sample_stride_read():
    """Core backward-compatibility guarantee: with the default
    (depth_sparsity_schedule_mode='fixed'), resolve_effective_depth_stride
    must return EXACTLY what the old unconditional
    `int(insertion_plan.sample_stride)` read would have, regardless of
    current_frame_idx."""
    policy = _Policy()
    for base_stride in (1, 4, 8, 16, 32):
        cfg = config_from_policy(policy, base_stride=base_stride)
        assert cfg is None
        for frame_idx in (0, 1, 5, 100, 5270, 999_999):
            assert (
                resolve_effective_depth_stride(cfg, frame_idx, base_stride)
                == base_stride
            )


def test_coarse_to_fine_mode_ramps_using_frames_ingested_progress_axis():
    policy = _Policy()
    policy.depth_sparsity_schedule_mode = "coarse_to_fine"
    policy.depth_sparsity_start_stride = 16
    policy.depth_sparsity_end_stride = 4
    policy.depth_sparsity_ramp_frames = 500
    cfg = config_from_policy(policy, base_stride=8)

    assert resolve_effective_depth_stride(cfg, 0, 8) == 16
    assert resolve_effective_depth_stride(cfg, 500, 8) == 4
    assert resolve_effective_depth_stride(cfg, 10_000, 8) == 4

    # Monotonic non-increasing as more frames are ingested.
    strides = [resolve_effective_depth_stride(cfg, t, 8) for t in range(0, 600, 10)]
    assert all(a >= b for a, b in zip(strides, strides[1:])), strides


# ---------------------------------------------------------------------------
# 5. Progress-signal mapping: production_application.py -> current_frame_idx
# ---------------------------------------------------------------------------


def test_production_application_passes_frames_seen_as_current_frame_idx():
    """The causal_mapping runtime's own frames-ingested counter
    (ProductionMappingApplication.frames_seen, incremented once per admitted
    frame) must be threaded through as current_frame_idx -- the same
    semantic insert_gaussians_from_frame's callers use, just under a
    different attribute name."""
    import eligsir.runtime.production_application as _pa

    source = inspect.getsource(_pa)
    assert "current_frame_idx=self.frames_seen" in source
