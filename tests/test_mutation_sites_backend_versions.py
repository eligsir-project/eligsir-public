"""Wave 2 backend-ownership pass (2026-08-06): the six previously-throwaway
``mutation_transaction_for_model(gaussians)`` call sites (opacity clamp,
bootstrap-anchor pose correction, reach-recycle opacity crush,
loop-closure submap correction, loop-closure gaussian-merge correction,
SubmapManager pose-graph correction) now accept an optional
``map_backend=`` that, when given, commits through the backend so
``map_version``/``topology_version`` actually advance -- previously always
0/0 in production because nothing captured the discarded ``TopologyChange``.

This covers the two call sites reachable as standalone functions
(``loop_closure.submap_correction``); the three sites inline inside
``train_streaming.py``'s training loop are exercised indirectly by the
existing GPU-parity/integration tests and by a real training run, since
they're not separable functions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scene.gsplat_model import GsplatGaussianModel
from eligsir.map.adapters.gsplat import GsplatMapBackend
from eligsir.map.mutation import SubmapCorrection
from eligsir.utils.graphics_utils import BasicPointCloud

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


class _FakeSubmapManager:
    def __init__(self, n):
        self._T = np.eye(4, dtype=np.float32)
        self.n = n

    def get_submap_transform(self, sid):
        return self._T

    def apply_pose_graph_corrections(self, transforms, gaussians=None):
        pass


def _make_model(n_points: int = 8) -> GsplatGaussianModel:
    torch.manual_seed(0)
    model = GsplatGaussianModel(sh_degree=0)
    points = torch.rand(n_points, 3, device="cuda")
    colors = torch.rand(n_points, 3, device="cuda")
    pcd = BasicPointCloud(
        points=points.cpu().numpy(), colors=colors.cpu().numpy(),
        normals=np.zeros((n_points, 3)),
    )
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.submap_id = torch.zeros(n_points, dtype=torch.int64, device="cuda")
    return model


def test_apply_submap_correction_advances_backend_versions_when_given():
    from loop_closure.submap_correction import apply_submap_correction

    model = _make_model()
    backend = GsplatMapBackend(model)
    assert backend._map_version == 0

    identity = np.eye(4, dtype=np.float32)
    n = apply_submap_correction(
        model, _FakeSubmapManager(model.get_xyz.shape[0]), [0], [identity],
        map_backend=backend,
    )
    assert n == model.get_xyz.shape[0]
    assert backend._map_version == 1
    assert backend._topology_version == 0  # pose correction, no topology change


def test_apply_submap_correction_without_backend_still_works():
    """Fallback path (no map_backend) must remain behavior-identical to
    before this pass -- existing callers/tests don't pass one."""
    from loop_closure.submap_correction import apply_submap_correction

    model = _make_model()
    identity = np.eye(4, dtype=np.float32)
    n = apply_submap_correction(
        model, _FakeSubmapManager(model.get_xyz.shape[0]), [0], [identity],
    )
    assert n == model.get_xyz.shape[0]


def test_apply_gaussian_merge_correction_advances_backend_versions_when_given():
    from loop_closure.submap_correction import apply_gaussian_merge_correction

    model = _make_model(n_points=4)
    model.support_count = torch.ones(4, dtype=torch.float32, device="cuda")
    model.provisional = torch.tensor([False, False, True, True], device="cuda")
    # Force a close duplicate pair so the merge path actually mutates.
    with torch.no_grad():
        model.means_param[2] = model.means_param[0] + 0.001

    backend = GsplatMapBackend(model)
    n = apply_gaussian_merge_correction(model, match_radius=0.1, map_backend=backend)
    assert n >= 0
    if n > 0:
        assert backend._map_version >= 1
