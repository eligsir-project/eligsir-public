"""Unit tests for the spatial active-bubble mask (Component 1).

_bubble_active_mask returns a bool mask (True = INSIDE the bubble, trainable;
False = out-of-bubble, frozen this window) computed from CURRENT positions and
the robot pose stashed on args. Default-off / unset => None (no gating).
CPU-only: means are plain tensors, args a tiny namespace.
"""
from __future__ import annotations

import types

import torch

from eligsir.utils.strategies.mcmc_strategy import _bubble_active_mask


def _args(**kw):
    base = dict(streaming_spatial_bubble_enabled=True, streaming_bubble_radius=5.0)
    base.update(kw)
    ns = types.SimpleNamespace(**base)
    return ns


def _means():
    # four points at distances 0, 3, 5, 10 m along +x from the origin
    return torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float32,
    )


def test_disabled_returns_none():
    a = _args(streaming_spatial_bubble_enabled=False)
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(_means(), a) is None


def test_no_center_returns_none():
    a = _args()  # no _bubble_center attribute
    assert _bubble_active_mask(_means(), a) is None


def test_nonpositive_radius_returns_none():
    a = _args(streaming_bubble_radius=0.0)
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(_means(), a) is None


def test_mask_selects_within_radius():
    a = _args(streaming_bubble_radius=5.0)
    a._bubble_center = torch.zeros(3)
    mask = _bubble_active_mask(_means(), a)
    # strict <: 0 and 3 in, 5 out (boundary), 10 out
    assert mask.tolist() == [True, True, False, False]


def test_mask_follows_moving_robot():
    a = _args(streaming_bubble_radius=2.0)
    a._bubble_center = torch.tensor([10.0, 0.0, 0.0])  # robot now near the far point
    mask = _bubble_active_mask(_means(), a)
    assert mask.tolist() == [False, False, False, True]


def test_empty_means_returns_none():
    a = _args()
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(torch.empty((0, 3)), a) is None


def test_guard_band_extends_active_region():
    # radius 5 + guard 1 => freeze boundary at 6m: the 5m point is now active.
    a = _args(streaming_bubble_radius=5.0, streaming_bubble_guard_m=1.0)
    a._bubble_center = torch.zeros(3)
    mask = _bubble_active_mask(_means(), a)  # points at 0,3,5,10
    assert mask.tolist() == [True, True, True, False]  # 5m in (5<6), 10m out


def test_zero_guard_matches_radius():
    a = _args(streaming_bubble_radius=5.0, streaming_bubble_guard_m=0.0)
    a._bubble_center = torch.zeros(3)
    mask = _bubble_active_mask(_means(), a)
    assert mask.tolist() == [True, True, False, False]  # strict 5m boundary


def test_forward_offset_center_places_bubble_on_geometry():
    # Simulate the training-loop center computation: camera at origin looking +x
    # (forward = +x), geometry 5m ahead. The bubble center should land ON the
    # geometry (camera + forward*d_obs), NOT at the empty camera position.
    cam_center = torch.zeros(3)
    forward = torch.tensor([1.0, 0.0, 0.0])
    d_obs = 5.0
    center = cam_center + forward * d_obs
    a = _args(streaming_bubble_radius=2.0, streaming_bubble_guard_m=0.0)
    a._bubble_center = center
    # geometry cluster around 5m ahead; a camera-centered 2m bubble would freeze ALL
    # of it, but the forward-offset center keeps it active.
    geom = torch.tensor([[4.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0], [0.0, 0, 0]])
    mask = _bubble_active_mask(geom, a)
    assert mask.tolist() == [True, True, True, False]  # 4/5/6m in-bubble; camera pos frozen


class _FakeGaussians:
    """Minimal stand-in exposing params like GsplatGaussianModel for parent-gating tests."""

    def __init__(self, means):
        self.params = {"means": means, "opacities": torch.zeros(means.shape[0])}


def test_growth_parent_pool_excludes_out_of_bubble():
    """P0.6: parent index pool for energy-guided growth/relocation must be
    restricted to the bubble — param_fn mutates parents in place, so an
    out-of-bubble parent would corrupt frozen geometry."""
    means = _means()
    a = _args(streaming_bubble_guard_m=0.0)
    a._bubble_center = torch.zeros(3)
    bubble = _bubble_active_mask(means, a)
    all_indices = torch.arange(means.shape[0])
    gated = all_indices[bubble[all_indices]]
    # points at 0 and 3 m are inside R=5; 5 m (boundary, strict <) and 10 m are out
    assert gated.tolist() == [0, 1]


def test_parent_pool_ungated_when_bubble_off():
    means = _means()
    a = _args(streaming_spatial_bubble_enabled=False)
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(means, a) is None


# ---- Decoupling: stats-only mask, empty-bubble guard, freeze-flag isolation ----

from eligsir.utils.strategies.mcmc_strategy import _bubble_feature_enabled, _BUBBLE_MIN_ACTIVE


def test_stats_only_mask_local_stats_on_bubble_off():
    """--streaming_local_stats alone must produce a mask (local statistics
    must not require the freeze)."""
    a = _args(streaming_spatial_bubble_enabled=False, streaming_local_stats=True)
    a._bubble_center = torch.zeros(3)
    mask = _bubble_active_mask(_means(), a)
    assert mask is not None
    assert mask.tolist() == [True, True, False, False]


def test_both_flags_off_returns_none():
    a = _args(streaming_spatial_bubble_enabled=False, streaming_local_stats=False)
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(_means(), a) is None


def test_empty_bubble_guard_returns_none_under_500():
    """A near-empty bubble (in a large map) must degrade to None (global
    behavior) instead of zeroing the map-building machinery."""
    n = 2000
    means = torch.randn(n, 3) + torch.tensor([100.0, 0.0, 0.0])  # all far from center
    means[:10] = 0.0  # only 10 points inside the bubble
    a = _args(streaming_bubble_radius=5.0)
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(means, a) is None


def test_guard_not_tripped_when_enough_inside():
    n = 2000
    means = torch.zeros(n, 3)  # all inside
    a = _args(streaming_bubble_radius=5.0)
    a._bubble_center = torch.zeros(3)
    mask = _bubble_active_mask(means, a)
    assert mask is not None and int(mask.sum()) == n


def test_guard_skipped_for_small_populations():
    """Tiny (bootstrap-size) maps keep exact mask semantics — the guard only
    applies when the total population exceeds the threshold."""
    means = _means()  # 4 points, way below _BUBBLE_MIN_ACTIVE
    a = _args()
    a._bubble_center = torch.zeros(3)
    assert _bubble_active_mask(means, a) is not None
    assert _BUBBLE_MIN_ACTIVE == 500


def test_freeze_flag_isolation():
    """local_stats-only args produce a mask, but the freeze predicate
    (streaming_spatial_bubble_enabled) stays False — freeze sites must key off
    the bubble flag explicitly, never off mask presence."""
    a = _args(streaming_spatial_bubble_enabled=False, streaming_local_stats=True)
    a._bubble_center = torch.zeros(3)
    assert _bubble_feature_enabled(a) is True
    assert _bubble_active_mask(_means(), a) is not None
    # the module-level freeze predicate all freeze sites use:
    assert getattr(a, "streaming_spatial_bubble_enabled", False) is False


def test_legacy_dead_mask_active_mask_and():
    """compute_legacy_dead_mask must AND the optional active_mask into the
    returned mask without altering clause logic."""
    from eligsir.utils.gaussian_energy_policy import GaussianEnergyConfig, GaussianEnergyPolicy

    class _G:
        get_opacity = torch.tensor([[1e-6], [1e-6], [1e-6], [0.9]])

    policy = GaussianEnergyPolicy(GaussianEnergyConfig(alpha_dead=0.05))
    base = policy.compute_legacy_dead_mask(_G()).mask
    assert base.tolist() == [True, True, True, False]
    active = torch.tensor([True, False, True, True])
    gated = policy.compute_legacy_dead_mask(_G(), active_mask=active).mask
    assert gated.tolist() == [True, False, True, False]
