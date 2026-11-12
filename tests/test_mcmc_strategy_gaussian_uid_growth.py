"""Regression test: MCMC growth (_append_running_state) must extend
gaussian_uid with fresh, unique values, not silently skip it.

Found live 2026-07-25: the first real smoke test of the keyframe/Gaussian
observation database plan crashed on the very first MCMC growth event --
_add_energy_guided clones existing Gaussians via a direct params/optimizer
torch.cat, bypassing add_points_as_gaussians entirely (the only other place
gaussian_uid rows get created), desyncing gaussian_uid from get_xyz
permanently from that point on.
"""
from __future__ import annotations

import types

import torch

from eligsir.utils.strategies.mcmc_strategy import GsplatEnergyMCMCStrategy


def _make_fake_gaussians(n: int, next_uid: int) -> types.SimpleNamespace:
    g = types.SimpleNamespace()
    g.gaussian_uid = torch.arange(next_uid - n, next_uid, dtype=torch.int64)
    g._next_gaussian_uid = next_uid
    # A couple of sibling buffers _append_running_state also extends, to
    # confirm this fix doesn't disturb their existing behavior.
    g.birth_frame = torch.zeros(n, dtype=torch.int32)
    g.support_count = torch.zeros(n, dtype=torch.int16)
    return g


def test_append_running_state_extends_gaussian_uid_with_fresh_values():
    strategy = GsplatEnergyMCMCStrategy()
    g = _make_fake_gaussians(n=5, next_uid=5)
    before = g.gaussian_uid.clone()

    strategy._append_running_state(g, old_count=5, added_count=3, parent_idxs=torch.tensor([0, 1, 2]))

    assert g.gaussian_uid.shape[0] == 8
    assert torch.equal(g.gaussian_uid[:5], before)
    new_uids = g.gaussian_uid[5:].tolist()
    assert new_uids == [5, 6, 7]  # fresh, sequential, not the parents' uids
    assert len(set(g.gaussian_uid.tolist())) == 8  # all unique, no collisions
    assert g._next_gaussian_uid == 8


def test_append_running_state_skips_gaussian_uid_when_already_desynced():
    """Matches every sibling buffer's own shape-guard: if gaussian_uid is
    ALREADY out of sync with old_count for some other reason, this method
    must not make it worse by appending against the wrong base length."""
    strategy = GsplatEnergyMCMCStrategy()
    g = _make_fake_gaussians(n=5, next_uid=5)
    g.gaussian_uid = g.gaussian_uid[:3]  # simulate a pre-existing desync

    strategy._append_running_state(g, old_count=5, added_count=3, parent_idxs=torch.tensor([0, 1, 2]))

    assert g.gaussian_uid.shape[0] == 3  # untouched, per the shape guard
