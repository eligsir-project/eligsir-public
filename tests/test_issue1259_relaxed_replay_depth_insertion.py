"""Regression tests for the #1259 far-region-starvation follow-up.

Root cause (see docs/architecture/2026-08-30-issue1259-far-region-insertion-asymmetry.md):
the live/replay depth-insertion occupancy gate (`DepthInsertionPolicy.alpha_threshold`,
`max_depth_m`) treats a pixel as "already covered" once rendered alpha reaches
`alpha_threshold` -- a bar a sparse, coarse-scale far-room region can clear with very few
Gaussians, after which it is permanently excluded from further depth-based densification.
The one-time bootstrap seed has no occupancy gate at all (there is no existing render at
iteration 0), so it admits the same far geometry unconditionally -- this asymmetry is the
bug. `relaxed_depth_insertion_policy` lets the already-existing, off-by-default,
post-source-complete replay-depth-insertion pass (`--streaming_replay_depth_max_per_event`)
use a widened occupancy/range policy distinct from the live path's, without touching the
live per-frame path at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from eligsir.runtime.config import DepthInsertionPolicy, relaxed_depth_insertion_policy
from eligsir.runtime.config_boundary import depth_insertion_policy_from_args


def _base_args(**overrides):
    defaults = dict(
        streaming_min_depth=0.1,
        streaming_max_depth=8.0,
        streaming_depth_edge_threshold=0.02,
        streaming_max_view_angle=70.0,
        streaming_depth_filter_enabled=False,
        streaming_depth_filter_erode_px=2,
        streaming_depth_filter_window=5,
        streaming_depth_filter_min_valid_ratio=0.75,
        streaming_depth_filter_max_range=0.08,
        streaming_depth_filter_median_thresh=0.04,
        streaming_depth_filter_bottom_margin=0.0,
        streaming_depth_temporal_check=False,
        streaming_depth_temporal_thresh=0.07,
        streaming_depth_temporal_require_for_insert=False,
        streaming_depth_temporal_require_for_sensor_closer=False,
        streaming_depth_consistency_thresh=0.05,
        streaming_insert_alpha_threshold=0.3,
        streaming_insert_score_mode="boolean_or",
        streaming_sparse_seed_voxel_size=0.02,
        streaming_sparse_seed_min_confidence=0.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_default_sentinels_are_behavior_neutral():
    base = depth_insertion_policy_from_args(_base_args())
    relaxed = relaxed_depth_insertion_policy(
        base, alpha_threshold=-1.0, max_depth_m=0.0
    )
    assert relaxed is base


def test_relaxed_alpha_threshold_widens_gate_only():
    base = depth_insertion_policy_from_args(_base_args())
    relaxed = relaxed_depth_insertion_policy(
        base, alpha_threshold=0.7, max_depth_m=0.0
    )
    assert relaxed.alpha_threshold == 0.7
    assert relaxed.max_depth_m == base.max_depth_m
    # everything else is untouched
    assert relaxed.min_depth_m == base.min_depth_m
    assert relaxed.edge_threshold == base.edge_threshold


def test_relaxed_max_depth_widens_range_only():
    base = depth_insertion_policy_from_args(_base_args())
    relaxed = relaxed_depth_insertion_policy(
        base, alpha_threshold=-1.0, max_depth_m=15.0
    )
    assert relaxed.max_depth_m == 15.0
    assert relaxed.alpha_threshold == base.alpha_threshold


def test_relaxed_policy_never_mutates_live_policy():
    base = depth_insertion_policy_from_args(_base_args())
    relaxed_depth_insertion_policy(base, alpha_threshold=0.9, max_depth_m=20.0)
    assert base.alpha_threshold == 0.3
    assert base.max_depth_m == 8.0
    assert isinstance(base, DepthInsertionPolicy)
