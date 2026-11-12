"""Unit tests for FreeSpaceConsensusIndex (voxel-hash cross-frame free-space
voting store backing tri-state ray semantics — see
eligsir/mapping/geometry/free_space_consensus.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex  # noqa: E402


def test_empty_index_has_zero_votes_everywhere():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=0.05)
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    counts = idx.vote_counts(pts)
    assert counts.tolist() == [0, 0]
    assert idx.certified_mask(pts, min_views=1).tolist() == [False, False]


def test_vote_free_dedups_within_same_frame():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=0.0)
    pts = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])  # same voxel
    n_new = idx.vote_free(pts, view_uid=1, camera_center=torch.zeros(3))
    assert n_new == 1  # collapsed to a single voxel
    counts = idx.vote_counts(pts)
    assert counts.tolist() == [1, 1]


def test_same_view_uid_does_not_vote_twice():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=0.0)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    idx.vote_free(pts, view_uid=1, camera_center=torch.zeros(3))
    n_new = idx.vote_free(pts, view_uid=1, camera_center=torch.zeros(3))
    assert n_new == 0
    assert idx.vote_counts(pts).tolist() == [1]


def test_baseline_gate_rejects_close_viewpoint():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=1.0)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    idx.vote_free(pts, view_uid=1, camera_center=torch.tensor([0.0, 0.0, 0.0]))
    # Different view_uid, but camera moved only 0.1m -- below 1.0m gate.
    n_new = idx.vote_free(pts, view_uid=2, camera_center=torch.tensor([0.1, 0.0, 0.0]))
    assert n_new == 0
    assert idx.vote_counts(pts).tolist() == [1]


def test_baseline_gate_accepts_far_viewpoint():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=1.0)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    idx.vote_free(pts, view_uid=1, camera_center=torch.tensor([0.0, 0.0, 0.0]))
    n_new = idx.vote_free(pts, view_uid=2, camera_center=torch.tensor([2.0, 0.0, 0.0]))
    assert n_new == 1
    assert idx.vote_counts(pts).tolist() == [2]


def test_certified_mask_threshold():
    idx = FreeSpaceConsensusIndex(voxel_size_m=0.1, min_baseline_m=0.0)
    pts = torch.tensor([[0.0, 0.0, 0.0]])
    for uid in range(3):
        idx.vote_free(pts, view_uid=uid, camera_center=torch.tensor([float(uid), 0.0, 0.0]))
    assert idx.certified_mask(pts, min_views=3).tolist() == [True]
    assert idx.certified_mask(pts, min_views=4).tolist() == [False]
