"""Unit tests for Part C floater-consensus mask + view accounting.

Pure-tensor logic only (no GPU / training loop), so these run on CPU.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.utils.streaming_depth_lock import (  # noqa: E402
    consensus_clean_mask,
    consensus_floater_mask,
    consensus_floater_mask_from_counts,
    ensure_consensus_buffers,
    floater_consensus_enabled,
    record_consensus_views,
    reset_consensus_counts,
)


class _DummyGaussians:
    """Minimal gaussian stand-in exposing get_xyz + dynamic attribute buffers."""

    def __init__(self, n: int):
        self._xyz = torch.zeros((n, 3), dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz


def _args(**kw):
    base = dict(
        streaming_floater_consensus_enabled=True,
        streaming_floater_consensus_min_views=3,
        streaming_floater_consensus_conflict_ratio=0.5,
        streaming_floater_consensus_depth_tol_m=0.05,
    )
    base.update(kw)
    return Namespace(**base)


# --------------------------------------------------------------------------
# Pure mask math
# --------------------------------------------------------------------------

def test_mask_basic_floater_and_clean():
    # min_views counts OBSERVATIONS (seen), not disagreements.
    # g0: clear floater (seen>=K, conflict ratio high, conflict>support)
    # g1: well supported surface (no conflicts)
    # g2: too few observations (seen=2 < K=3) => not yet trusted
    # g3: conflict==support (tie => not pruned, conflict must strictly exceed support)
    support = torch.tensor([0, 8, 0, 4], dtype=torch.int32)
    conflict = torch.tensor([6, 0, 2, 4], dtype=torch.int32)
    seen = torch.tensor([6, 8, 2, 8], dtype=torch.int32)
    mask = consensus_floater_mask_from_counts(
        support, conflict, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask.tolist() == [True, False, False, False]


def test_mask_two_of_three_conflicting_floater():
    # Spec discriminator: a Gaussian seen in 3 views, 2 of which conflict,
    # 0 support => seen>=3, ratio 0.667>=0.5, conflict>support => pruned.
    support = torch.tensor([0], dtype=torch.int32)
    conflict = torch.tensor([2], dtype=torch.int32)
    seen = torch.tensor([3], dtype=torch.int32)
    mask = consensus_floater_mask_from_counts(
        support, conflict, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask.tolist() == [True]


def test_mask_ratio_threshold():
    # conflict=3 over seen=10 => ratio 0.3 < 0.5 even though views >= K
    support = torch.tensor([0], dtype=torch.int32)
    conflict = torch.tensor([3], dtype=torch.int32)
    seen = torch.tensor([10], dtype=torch.int32)
    mask = consensus_floater_mask_from_counts(
        support, conflict, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask.tolist() == [False]
    # Same counts but a higher conflict count crossing ratio + views
    conflict2 = torch.tensor([6], dtype=torch.int32)
    mask2 = consensus_floater_mask_from_counts(
        support, conflict2, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask2.tolist() == [True]


def test_mask_requires_conflict_gt_support():
    support = torch.tensor([5], dtype=torch.int32)
    conflict = torch.tensor([5], dtype=torch.int32)  # tie
    seen = torch.tensor([5], dtype=torch.int32)
    mask = consensus_floater_mask_from_counts(
        support, conflict, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask.tolist() == [False]


def test_mask_seen_zero_safe():
    # seen=0 must not divide-by-zero; conflict=0 so never a floater
    support = torch.tensor([0], dtype=torch.int32)
    conflict = torch.tensor([0], dtype=torch.int32)
    seen = torch.tensor([0], dtype=torch.int32)
    mask = consensus_floater_mask_from_counts(
        support, conflict, seen, min_views=3, conflict_ratio=0.5
    )
    assert mask.tolist() == [False]


# --------------------------------------------------------------------------
# Disabled => no-op contract
# --------------------------------------------------------------------------

def test_disabled_is_noop():
    g = _DummyGaussians(4)
    args = _args(streaming_floater_consensus_enabled=False)
    assert not floater_consensus_enabled(args)
    # Mask returns all-False, clean returns all-True, no buffers allocated.
    assert consensus_floater_mask(g, args).tolist() == [False, False, False, False]
    assert consensus_clean_mask(g, args).tolist() == [True, True, True, True]
    assert not hasattr(g, "seen_count")


def test_model_mask_without_buffers_is_noop():
    g = _DummyGaussians(3)
    args = _args()  # enabled but buffers not yet allocated
    assert consensus_floater_mask(g, args).tolist() == [False, False, False]


# --------------------------------------------------------------------------
# Per-unique-view accounting + dedup
# --------------------------------------------------------------------------

def test_record_consensus_views_dedup_and_accumulate():
    g = _DummyGaussians(3)
    ensure_consensus_buffers(g)
    idx = torch.tensor([0, 1, 2])
    support = torch.tensor([False, True, False])
    conflict = torch.tensor([True, False, True])
    # View 7
    n = record_consensus_views(g, idx, support, conflict, view_uid=7)
    assert n == 3
    # Re-recording the same view must be a no-op (dedup)
    n2 = record_consensus_views(g, idx, support, conflict, view_uid=7)
    assert n2 == 0
    # A new view increments again
    record_consensus_views(g, idx, support, conflict, view_uid=8)
    assert g.seen_count.tolist() == [2, 2, 2]
    assert g.consensus_support_count.tolist() == [0, 2, 0]
    assert g.consensus_conflict_count.tolist() == [2, 0, 2]


def test_record_then_mask_flags_floater():
    g = _DummyGaussians(2)
    ensure_consensus_buffers(g)
    idx = torch.tensor([0, 1])
    # g0 conflicts in 3 distinct views, g1 supported in 3 views
    for uid in (1, 2, 3):
        record_consensus_views(
            g, idx,
            support=torch.tensor([False, True]),
            conflict=torch.tensor([True, False]),
            view_uid=uid,
        )
    args = _args()
    mask = consensus_floater_mask(g, args)
    assert mask.tolist() == [True, False]
    clean = consensus_clean_mask(g, args)
    assert clean.tolist() == [False, True]


def test_reset_consensus_counts():
    g = _DummyGaussians(2)
    ensure_consensus_buffers(g)
    idx = torch.tensor([0, 1])
    record_consensus_views(
        g, idx,
        support=torch.tensor([False, False]),
        conflict=torch.tensor([True, True]),
        view_uid=1,
    )
    reset_consensus_counts(g, torch.tensor([0]))
    assert g.seen_count.tolist() == [0, 1]
    assert g.consensus_conflict_count.tolist() == [0, 1]
    assert g.consensus_last_seen_uid.tolist() == [-1, 1]
