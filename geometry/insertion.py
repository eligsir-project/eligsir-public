"""Surface-aware insertion candidate data structures and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from geometry.normals import quaternion_from_normal_torch
from geometry.pca_degeneracy import eigvals_well_conditioned
from geometry.priors import GeometrySource, SurfacePrior


@dataclass
class SurfaceCandidates:
    pts: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    log_scales: np.ndarray
    quats: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    score: np.ndarray
    birth_frame: int

    def __len__(self) -> int:
        return int(self.pts.shape[0])

    def as_legacy_dict(self, *, init_opacity: float) -> dict:
        return {
            "pts": self.pts,
            "cols": self.colors,
            "log_scales": self.log_scales,
            "q_world": self.quats,
            "normals": self.normals,
            "confidence": self.confidence,
            "source": self.source,
            "score": self.score,
            "birth_frame": self.birth_frame,
            "init_opacity": init_opacity,
        }


def candidate_metadata_from_prior(
    prior: SurfacePrior,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    default_source: int = int(GeometrySource.SENSOR),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_i = np.asarray(xs, dtype=np.int64)
    ys_i = np.asarray(ys, dtype=np.int64)
    confidence = prior.confidence[0, ys_i, xs_i].detach().cpu().numpy().astype(np.float32)
    if prior.source is None:
        source = np.full(confidence.shape, default_source, dtype=np.int16)
    else:
        source = prior.source[0, ys_i, xs_i].detach().cpu().numpy().astype(np.int16)
    if prior.normals is None:
        normals = np.zeros((confidence.shape[0], 3), dtype=np.float32)
        normals[:, 2] = 1.0
    else:
        normals = prior.normals[:, ys_i, xs_i].permute(1, 0).detach().cpu().numpy().astype(np.float32)
    return normals, confidence, source


def quaternions_from_normals_np(normals: np.ndarray) -> np.ndarray:
    q = quaternion_from_normal_torch(torch.from_numpy(normals.astype(np.float32)))
    return q.detach().cpu().numpy().astype(np.float32)


@dataclass
class SurfaceCandidatesT:
    """Torch-native counterpart of :class:`SurfaceCandidates` (WS2, item 1).

    Additive only: nothing in the existing numpy-based insertion pipeline
    calls this yet. All fields stay on whatever device they're constructed
    on -- callers that want the legacy numpy dataclass should use
    :meth:`to_numpy` explicitly, e.g. at a train_streaming.py commit
    boundary, rather than have this class silently round-trip through CPU.
    """

    pts: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    confidence: torch.Tensor
    source: torch.Tensor
    score: torch.Tensor
    birth_frame: int

    def __len__(self) -> int:
        return int(self.pts.shape[0])

    def to_numpy(self) -> SurfaceCandidates:
        return SurfaceCandidates(
            pts=self.pts.detach().cpu().numpy().astype(np.float32),
            colors=self.colors.detach().cpu().numpy().astype(np.float32),
            normals=self.normals.detach().cpu().numpy().astype(np.float32),
            log_scales=self.log_scales.detach().cpu().numpy().astype(np.float32),
            quats=self.quats.detach().cpu().numpy().astype(np.float32),
            confidence=self.confidence.detach().cpu().numpy().astype(np.float32),
            source=self.source.detach().cpu().numpy().astype(np.int16),
            score=self.score.detach().cpu().numpy().astype(np.float32),
            birth_frame=self.birth_frame,
        )


def candidate_metadata_from_prior_torch(
    prior: SurfacePrior,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    default_source: int = int(GeometrySource.SENSOR),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-native counterpart of :func:`candidate_metadata_from_prior` (WS2).

    Same semantics, but never leaves the device ``prior``'s tensors are on --
    no ``.cpu().numpy()`` round-trip. ``xs``/``ys`` must already be
    ``torch.long`` tensors on that same device. Additive: not yet called by
    the insertion pipeline (see docs/2026-07-26-transfer-audit.md's WS2-scope
    entries for candidate_metadata_from_prior); parity with the numpy version
    is asserted in tests/test_ws2_torch_candidates.py.
    """
    xs_i = xs.to(dtype=torch.long)
    ys_i = ys.to(dtype=torch.long)
    confidence = prior.confidence[0, ys_i, xs_i].to(dtype=torch.float32)
    if prior.source is None:
        source = torch.full_like(confidence, float(default_source), dtype=torch.int16)
    else:
        source = prior.source[0, ys_i, xs_i].to(dtype=torch.int16)
    if prior.normals is None:
        normals = torch.zeros((confidence.shape[0], 3), dtype=torch.float32, device=confidence.device)
        normals[:, 2] = 1.0
    else:
        normals = prior.normals[:, ys_i, xs_i].permute(1, 0).to(dtype=torch.float32)
    return normals, confidence, source


def distance_aware_inplane_scale(
    z: np.ndarray,
    fx: float,
    fy: float,
    *,
    coeff: float = 1.0,
    scale_max: float = 0.05,
    scale_min: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """LEVER 1 — explicit distance-aware in-plane Gaussian scale.

    A pixel at depth ``z`` subtends a world size ``≈ z / f`` (one pixel of
    angular footprint back-projected onto the surface). Inserting a Gaussian
    whose in-plane extent matches that footprint lets far surfels tile the
    surface without holes while keeping near surfels small and sharp.

    This is the *explicit* form of the relation that the default footprint
    scale (``scale_mult * z / fx``) already encodes implicitly. It is exposed
    as its own lever so that distance-scaling can be restored even when
    ``scale_mode=knn`` has replaced the footprint term with local-density
    spacing (knn carries density, not absolute depth).

    Args:
        z:         per-point depth in metres, shape ``[N]``.
        fx, fy:    focal lengths in pixels (depth intrinsics).
        coeff:     linear multiplier on the ``z / f`` footprint.
        scale_max: hard upper clamp per axis (metres).
        scale_min: lower clamp per axis (metres).

    Returns:
        ``(tx, ty)`` in-plane scales, each shape ``[N]`` float32, clamped to
        ``[scale_min, scale_max]``. Grows linearly with ``z`` until clamped.
    """
    z = np.asarray(z, dtype=np.float32)
    tx = np.clip(coeff * z / float(fx), scale_min, scale_max).astype(np.float32)
    ty = np.clip(coeff * z / float(fy), scale_min, scale_max).astype(np.float32)
    return tx, ty


def _fuse_point_with_neighbors_impl(
    point: np.ndarray,
    neighbors: np.ndarray,
    *,
    normal_hint: np.ndarray | None = None,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Internal implementation of :func:`fuse_point_with_neighbors` that also
    reports whether the PCA normal was well-conditioned (see issue #1149).

    Returns ``(fused_position [3], fused_normal [3], degenerate)`` where
    ``degenerate`` is ``True`` iff the neighbourhood had too few points, or
    its covariance was rank-deficient (see
    ``geometry.pca_degeneracy.eigvals_well_conditioned``), and the returned
    normal is therefore the ``normal_hint`` fallback rather than a genuine
    PCA estimate.
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    nb = np.asarray(neighbors, dtype=np.float64).reshape(-1, 3)
    # Edge-aware filter: drop neighbours farther than ``max_radius`` from the
    # representative. KNN neighbourhoods span depth DISCONTINUITIES (a point on a
    # different surface behind an edge), and averaging those in pulls the fused
    # position OFF the surface — the measured cause of the surface-fusion PSNR
    # regression. Keeping only same-surface neighbours makes fusion denoise
    # within-plane jitter without smearing edges. None = no filter (legacy).
    if max_radius is not None and nb.shape[0] > 0:
        d = np.linalg.norm(nb - point[None, :], axis=1)
        nb = nb[d <= float(max_radius)]
    cloud = np.vstack([point[None, :], nb])

    fused_pos = cloud.mean(axis=0).astype(np.float32)

    normal = None
    degenerate = True
    if cloud.shape[0] >= 3:
        centered = cloud - cloud.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
        # Degeneracy guard (issue #1149): a rank-deficient neighbourhood (e.g.
        # collapsed to ~2 distinct point locations, or near-collinear points)
        # leaves the smallest-eigenvalue eigenvector living in an arbitrary
        # null space rather than a well-defined normal direction — NumPy's and
        # PyTorch's eigensolvers resolve that ambiguity differently, producing
        # normals that can differ by >1.5 rad on the identical input. Only
        # trust the PCA normal when the neighbourhood spans a genuine 2D
        # patch; otherwise fall back to normal_hint below, same as the
        # too-few-points case. See geometry/pca_degeneracy.py for the
        # threshold derivation.
        if eigvals_well_conditioned(float(evals[0]), float(evals[1]), float(evals[2])):
            candidate = evecs[:, 0]
            nrm = np.linalg.norm(candidate)
            if np.isfinite(nrm) and nrm >= 1e-9:
                normal = candidate / nrm
                degenerate = False

    if normal is None:
        if normal_hint is not None:
            h = np.asarray(normal_hint, dtype=np.float64).reshape(3)
            hn = np.linalg.norm(h)
            normal = (h / hn) if hn > 1e-9 else np.array([0.0, 0.0, 1.0])
        else:
            normal = np.array([0.0, 0.0, 1.0])

    if normal_hint is not None:
        h = np.asarray(normal_hint, dtype=np.float64).reshape(3)
        if float(np.dot(normal, h)) < 0.0:
            normal = -normal

    return fused_pos, normal.astype(np.float32), degenerate


def fuse_point_with_neighbors(
    point: np.ndarray,
    neighbors: np.ndarray,
    *,
    normal_hint: np.ndarray | None = None,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """LEVER 2 — KNN surface-fusion of one representative point.

    Given a representative point and its K nearest neighbours (all in world
    coordinates), produce a *denoised, oriented surfel*:
      (a) position = mean of {point} ∪ neighbours (averages out per-pixel
          depth jitter from noisy ToF/structured-light sensors), and
      (b) normal   = smallest-eigenvector of the neighbour-set covariance
          (PCA plane fit), oriented to agree with ``normal_hint`` when given.

    The smallest PCA eigenvector is sign-ambiguous; ``normal_hint`` (e.g. the
    finite-difference normal already computed for this point) fixes the sign so
    the downstream quaternion / grazing logic stays consistent.

    A rank-deficient neighbourhood (too few points, or points collapsed onto
    ~2 distinct locations / near-collinear — see issue #1149 and
    ``geometry.pca_degeneracy``) falls back to ``normal_hint`` exactly like
    the too-few-points case, since the PCA normal is not numerically
    well-defined in that case either. Use :func:`fuse_points_with_knn`'s
    ``degenerate_mask_out`` if you need to know which queries fell back.

    Args:
        point:       shape ``[3]`` the retained representative.
        neighbors:   shape ``[K, 3]`` its K nearest neighbours (may include the
                     point itself; fewer than 3 distinct points falls back to
                     the hint normal).
        normal_hint: optional shape ``[3]`` used only to orient the PCA normal.

    Returns:
        ``(fused_position [3], fused_normal [3])`` float32; ``fused_normal`` is
        unit-length.
    """
    fused_pos, normal, _degenerate = _fuse_point_with_neighbors_impl(
        point, neighbors, normal_hint=normal_hint, max_radius=max_radius
    )
    return fused_pos, normal


def fuse_points_with_knn(
    queries: np.ndarray,
    pool: np.ndarray,
    nn_idx: np.ndarray,
    normal_hints: np.ndarray,
    *,
    max_radius: float | None = None,
    degenerate_mask_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised wrapper: fuse each query against its KNN neighbours from a pool.

    Args:
        queries:      shape ``[N, 3]`` representative positions (post voxel-dedup).
        pool:         shape ``[M, 3]`` neighbour pool (pre-downsample candidates).
        nn_idx:       shape ``[N, K]`` int indices into ``pool`` (representative's
                      own nearest neighbours, self typically included).
        normal_hints: shape ``[N, 3]`` per-query orientation hints.
        degenerate_mask_out: optional pre-allocated bool array of shape ``[N]``,
                      filled in-place with whether each query's PCA normal was
                      degenerate (too few points, or a rank-deficient
                      neighbourhood — issue #1149) and therefore fell back to
                      ``normal_hints[i]`` instead of a genuine PCA estimate.
                      Callers should treat a non-empty count here as a signal
                      worth attributing/logging, never silently drop it — this
                      never causes ``fuse_points_with_knn`` itself to drop a
                      query; every query still gets a position and a normal.

    Returns:
        ``(fused_pts [N,3], fused_normals [N,3])`` float32.
    """
    queries = np.asarray(queries, dtype=np.float32)
    pool = np.asarray(pool, dtype=np.float32)
    nn_idx = np.asarray(nn_idx)
    out_pts = np.empty_like(queries)
    out_n = np.empty_like(queries)
    if degenerate_mask_out is not None and degenerate_mask_out.shape[0] != queries.shape[0]:
        raise ValueError("degenerate_mask_out must have shape [N] matching queries")
    for i in range(queries.shape[0]):
        neigh = pool[nn_idx[i]]
        fp, fn, degenerate = _fuse_point_with_neighbors_impl(
            queries[i], neigh, normal_hint=normal_hints[i], max_radius=max_radius
        )
        out_pts[i] = fp
        out_n[i] = fn
        if degenerate_mask_out is not None:
            degenerate_mask_out[i] = degenerate
    return out_pts, out_n
