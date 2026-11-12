"""WS2 item 1: torch-native SurfaceCandidatesT / candidate_metadata_from_prior_torch
must exactly match the existing numpy path -- pure round-trip elimination,
no behavior change. CPU-only (no GPU required for this parity check)."""

from __future__ import annotations

import numpy as np
import torch

from geometry.insertion import (
    SurfaceCandidates,
    SurfaceCandidatesT,
    candidate_metadata_from_prior,
    candidate_metadata_from_prior_torch,
    quaternions_from_normals_np,
)
from geometry.normals import quaternion_from_normal_torch
from geometry.priors import GeometrySource, SurfacePrior


def _make_prior(h: int = 8, w: int = 8, *, with_normals=True, with_source=True, seed=0) -> SurfacePrior:
    rng = np.random.RandomState(seed)
    depth = torch.from_numpy(rng.rand(1, h, w).astype(np.float32))
    confidence = torch.from_numpy(rng.rand(1, h, w).astype(np.float32))
    valid = torch.ones(1, h, w, dtype=torch.bool)
    normals = None
    if with_normals:
        n = rng.randn(3, h, w).astype(np.float32)
        n /= np.linalg.norm(n, axis=0, keepdims=True) + 1e-8
        normals = torch.from_numpy(n)
    source = None
    if with_source:
        source = torch.from_numpy(rng.randint(0, 4, size=(1, h, w)).astype(np.int16))
    return SurfacePrior(
        depth=depth, confidence=confidence, valid=valid, normals=normals,
        uncertainty=None, source=source, frame_uid=0,
    )


def _sample_indices(h: int, w: int, n: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    xs = rng.randint(0, w, size=n).astype(np.int64)
    ys = rng.randint(0, h, size=n).astype(np.int64)
    return xs, ys


class TestCandidateMetadataParity:
    def test_matches_numpy_path_with_normals_and_source(self):
        prior = _make_prior()
        xs, ys = _sample_indices(8, 8, 12)

        normals_np, confidence_np, source_np = candidate_metadata_from_prior(prior, xs, ys)
        normals_t, confidence_t, source_t = candidate_metadata_from_prior_torch(
            prior, torch.from_numpy(xs), torch.from_numpy(ys)
        )

        np.testing.assert_allclose(normals_t.numpy(), normals_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(confidence_t.numpy(), confidence_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(source_t.numpy(), source_np)

    def test_matches_numpy_path_without_normals(self):
        prior = _make_prior(with_normals=False)
        xs, ys = _sample_indices(8, 8, 5)

        normals_np, confidence_np, source_np = candidate_metadata_from_prior(prior, xs, ys)
        normals_t, confidence_t, source_t = candidate_metadata_from_prior_torch(
            prior, torch.from_numpy(xs), torch.from_numpy(ys)
        )

        np.testing.assert_allclose(normals_t.numpy(), normals_np)
        assert (normals_np[:, 2] == 1.0).all()

    def test_matches_numpy_path_without_source_uses_default(self):
        prior = _make_prior(with_source=False)
        xs, ys = _sample_indices(8, 8, 5)

        normals_np, confidence_np, source_np = candidate_metadata_from_prior(
            prior, xs, ys, default_source=int(GeometrySource.RGB)
        )
        normals_t, confidence_t, source_t = candidate_metadata_from_prior_torch(
            prior, torch.from_numpy(xs), torch.from_numpy(ys), default_source=int(GeometrySource.RGB)
        )
        np.testing.assert_array_equal(source_t.numpy(), source_np)
        assert (source_np == int(GeometrySource.RGB)).all()


class TestSurfaceCandidatesTToNumpy:
    def test_to_numpy_round_trips_exactly(self):
        n = 6
        rng = np.random.RandomState(2)
        cand_t = SurfaceCandidatesT(
            pts=torch.from_numpy(rng.randn(n, 3).astype(np.float32)),
            colors=torch.from_numpy(rng.rand(n, 3).astype(np.float32)),
            normals=torch.from_numpy(rng.randn(n, 3).astype(np.float32)),
            log_scales=torch.from_numpy(rng.randn(n, 3).astype(np.float32)),
            quats=torch.from_numpy(rng.randn(n, 4).astype(np.float32)),
            confidence=torch.from_numpy(rng.rand(n).astype(np.float32)),
            source=torch.from_numpy(rng.randint(0, 4, size=n).astype(np.int16)),
            score=torch.from_numpy(rng.rand(n).astype(np.float32)),
            birth_frame=42,
        )
        cand_np = cand_t.to_numpy()
        assert isinstance(cand_np, SurfaceCandidates)
        assert len(cand_np) == n == len(cand_t)
        np.testing.assert_array_equal(cand_np.pts, cand_t.pts.numpy())
        np.testing.assert_array_equal(cand_np.quats, cand_t.quats.numpy())
        assert cand_np.birth_frame == 42

    def test_len_matches_pts_shape(self):
        cand_t = SurfaceCandidatesT(
            pts=torch.zeros(7, 3), colors=torch.zeros(7, 3), normals=torch.zeros(7, 3),
            log_scales=torch.zeros(7, 3), quats=torch.zeros(7, 4), confidence=torch.zeros(7),
            source=torch.zeros(7, dtype=torch.int16), score=torch.zeros(7), birth_frame=0,
        )
        assert len(cand_t) == 7


class TestQuaternionEliminatedRoundTrip:
    def test_direct_torch_quaternion_matches_numpy_wrapper(self):
        """quaternions_from_normals_np is just torch->numpy->(caller reuses
        as numpy); calling quaternion_from_normal_torch directly on a torch
        normals tensor (as WS2 callers should) must match it exactly."""
        rng = np.random.RandomState(3)
        normals_np = rng.randn(10, 3).astype(np.float32)
        normals_np /= np.linalg.norm(normals_np, axis=1, keepdims=True) + 1e-8

        q_np_path = quaternions_from_normals_np(normals_np)
        q_direct = quaternion_from_normal_torch(torch.from_numpy(normals_np)).detach().numpy()

        np.testing.assert_allclose(q_direct, q_np_path, rtol=1e-6, atol=1e-6)
