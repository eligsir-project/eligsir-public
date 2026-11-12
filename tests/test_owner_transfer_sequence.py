"""Regression gates for the ownership/transfer convergence sequence."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from geometry.device_fusion import fuse_points_with_knn_torch, knn_isotropic_log_scale_torch
from geometry.insertion import SurfaceCandidatesT
from eligsir.map.arbiter import (
    GaussianMutationArbiter,
    ProposalKey,
    SurfaceInsertionProposal,
)
from eligsir.map.mutation import GaussianInsertionBatch, TopologyChange
from eligsir.mapping.integration.device_candidates import (
    deterministic_voxel_first_indices,
    normalize_pending_candidates,
)
from scene.cameras import make_gsplat_camera_batch


class _Txn:
    def __init__(self, log):
        self.log = log
        self.inserted = 0

    def insert(self, batch):
        n = int(batch.points.shape[0])
        self.log.append(("insert", float(batch.points[0, 0]), n))
        self.inserted += n
        return n

    def prune(self, mask):
        self.log.append(("prune", int(mask.sum())))

    def relocate(self, plan):
        self.log.append(("relocate",))

    def apply_pose_correction(self, correction):
        self.log.append(("pose",))

    def update_opacity_logits(self, mask, value):
        self.log.append(("opacity",))

    def commit(self):
        return TopologyChange(
            map_version=1,
            topology_version=1 if self.inserted else 0,
            inserted=self.inserted,
            content_changed=bool(self.inserted),
        )


class _Backend:
    def __init__(self, population=0):
        self.log = []
        self.population = population

    @property
    def population_count(self):
        return self.population

    def read_view(self):
        return SimpleNamespace(means=torch.zeros((self.population, 3)))

    def begin_mutation(self):
        return _Txn(self.log)

    def commit_mutation(self, txn):
        change = txn.commit()
        self.population += change.inserted
        return change


def _batch(x: float, n: int) -> GaussianInsertionBatch:
    pts = torch.full((n, 3), x)
    return GaussianInsertionBatch(points=pts, colors=torch.zeros((n, 3)))


def test_arbiter_orders_by_priority_then_source_key_and_centralizes_capacity():
    backend = _Backend(population=3)
    arbiter = GaussianMutationArbiter(backend, max_population=6)
    late = SurfaceInsertionProposal(
        key=ProposalKey(0, 9, 9, "cam"), batch=_batch(9.0, 2)
    )
    early = SurfaceInsertionProposal(
        key=ProposalKey(0, 2, 2, "cam"), batch=_batch(2.0, 2)
    )
    result = arbiter.commit((late, early))
    # Early proposal consumes two slots; late gets the one remaining slot.
    # Ordering/capacity are resolved before the boundary; one coalesced backend
    # operation prevents a later proposal failure leaving an earlier write live.
    assert backend.log == [("insert", 2.0, 3)]
    assert result.inserted == 3
    assert result.capacity_dropped == 1
    assert backend.population == 6


def _surface_candidates(points: torch.Tensor) -> SurfaceCandidatesT:
    n = int(points.shape[0])
    return SurfaceCandidatesT(
        pts=points,
        colors=torch.zeros((n, 3)),
        normals=torch.tensor([[0.0, 0.0, 1.0]]).expand(n, -1).clone(),
        log_scales=torch.zeros((n, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).clone(),
        confidence=torch.ones(n),
        source=torch.zeros(n, dtype=torch.int16),
        score=torch.ones(n),
        birth_frame=7,
    )


def test_surface_candidates_t_stay_on_device_and_do_not_require_numpy():
    candidate = _surface_candidates(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    out = normalize_pending_candidates((candidate,), device="cpu", init_opacity=0.1)
    assert out.pts.device.type == "cpu"
    assert out.pts.data_ptr() == candidate.pts.data_ptr()
    assert out.birth_frame == 7


def test_voxel_dedup_is_first_occurrence_stable_and_device_native():
    pts = torch.tensor(
        [[0.001, 0.0, 0.0], [0.019, 0.0, 0.0], [0.021, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    keep = deterministic_voxel_first_indices(pts, 0.02)
    assert keep.tolist() == [0, 2, 3]
    assert keep.device == pts.device


def test_batched_pca_fusion_matches_expected_plane_normal_without_cpu_indices():
    q = torch.tensor([[0.0, 0.0, 0.0]])
    pool = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    idx = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hint = torch.tensor([[0.0, 0.0, 1.0]])
    fused, normal = fuse_points_with_knn_torch(q, pool, idx, hint)
    assert fused.device == q.device
    assert torch.allclose(normal.abs(), torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)


def test_knn_scale_stays_on_device_and_is_batched():
    pts = torch.zeros((2, 3))
    d2 = torch.tensor([0.0004, 0.0025])
    scale = knn_isotropic_log_scale_torch(pts, d2)
    assert scale.shape == (2, 3)
    assert scale.device == pts.device
    assert torch.allclose(scale.exp()[:, 0], torch.tensor([0.02, 0.05]))


class _Cam:
    def __init__(self, tx):
        self.data_device = torch.device("cpu")
        self.world_view_transform = torch.eye(4)
        self.world_view_transform[3, 0] = tx
        self.camera_center = torch.tensor([tx, 0.0, 0.0])
        self.image_width = 8
        self.image_height = 6
        self.fx = 4.0
        self.fy = 5.0
        self.cx = 3.5
        self.cy = 2.5
        self.FoVx = 1.0
        self.FoVy = 1.0


def test_camera_batch_builds_viewmats_and_intrinsics_together():
    batch = make_gsplat_camera_batch((_Cam(0.0), _Cam(1.0)), device="cpu")
    assert batch.viewmats.shape == (2, 4, 4)
    assert batch.Ks.shape == (2, 3, 3)
    assert batch.camera_centers.shape == (2, 3)
    assert batch.Ks[:, 0, 0].tolist() == [4.0, 4.0]
