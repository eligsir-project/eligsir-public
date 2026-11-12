"""Unit tests for A1: relocation visibility-gate (_gate_dead_mask_to_visible)."""
from types import SimpleNamespace

import torch

from eligsir.utils.strategies.mcmc_strategy import _gate_dead_mask_to_visible


def _mask(*vals):
    return torch.tensor(vals, dtype=torch.bool)


def test_flag_off_is_noop():
    dead = _mask(True, True, False, True)
    args = SimpleNamespace(streaming_reloc_local_only=False)
    pkg = {"visibility_filter": _mask(False, True, False, False)}
    out = _gate_dead_mask_to_visible(dead, pkg, args)
    assert torch.equal(out, dead)  # unchanged when off


def test_flag_on_intersects_visibility():
    dead = _mask(True, True, False, True)          # dead candidates
    vis = _mask(False, True, False, True)           # visible this frame
    args = SimpleNamespace(streaming_reloc_local_only=True)
    out = _gate_dead_mask_to_visible(dead, {"visibility_filter": vis}, args)
    # Only dead AND visible survive -> index 1 and 3.
    assert torch.equal(out, _mask(False, True, False, True))


def test_flag_on_no_visibility_filter_is_noop():
    dead = _mask(True, False, True)
    args = SimpleNamespace(streaming_reloc_local_only=True)
    assert torch.equal(_gate_dead_mask_to_visible(dead, {}, args), dead)
    assert torch.equal(_gate_dead_mask_to_visible(dead, None, args), dead)


def test_shape_mismatch_is_noop():
    dead = _mask(True, False, True)
    args = SimpleNamespace(streaming_reloc_local_only=True)
    pkg = {"visibility_filter": _mask(True, False)}  # wrong length
    assert torch.equal(_gate_dead_mask_to_visible(dead, pkg, args), dead)
