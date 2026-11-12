"""``flush_insertion_batch`` must stamp the REAL training iteration as
``birth_iter`` on every Gaussian it inserts.

``birth_iter`` is what age / young-protection logic compares against the
optimizer step counter, so a candidate carrying ``birth_frame`` (a source-frame
index) in that slot silently breaks it -- the two are different units.

This used to be asserted by scraping ``inspect.getsource`` for the literal
``"birth_iter=iteration"`` and for an ``if map_backend is not None:`` / ``else:``
branch pair. Both assertions were about the SHAPE of the code rather than its
behaviour, and the ownership cutover invalidated both without touching the
contract they were protecting: there is now exactly one insertion path (every
commit goes through GaussianMutationArbiter), and the stamp is applied by
``SurfaceCandidatesT.as_insertion_batch(iteration=...)`` rather than by a
literal keyword at the call site. A source-shape test cannot tell "the contract
broke" apart from "the code was restructured", so these now assert the observable
result instead.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.mapping.integration.insertion_committer import flush_insertion_batch


class _RecordingGaussians:
    """Minimal gsplat-layout model recording what the mutation path inserted.

    The ``params`` attribute is what selects the layout: the committer's
    ephemeral backend picks GsplatMapBackend when it is present and the
    legacy graphdeco backend otherwise. That distinction matters here --
    ``birth_iter`` is a gsplat-layout field. ``scene/gaussian_model.py``'s
    ``add_points_as_gaussians`` has no ``birth_iter`` parameter at all, so the
    legacy adapter correctly does not forward one, and a legacy-shaped stub
    would assert a capability that layout has never had.
    """

    def __init__(self):
        self.params = {}
        self.get_xyz = torch.zeros((0, 3))
        self.birth_iter_seen = None
        self.birth_frame_seen = None

    def add_points_as_gaussians(self, points, colors, **kwargs):
        self.birth_iter_seen = kwargs.get("birth_iter")
        self.birth_frame_seen = kwargs.get("birth_frame")
        self.get_xyz = torch.cat([self.get_xyz, points], dim=0)
        return points.shape[0]


class _StubStreamingScene:
    def add_to_occupancy_hash(self, xyz):
        pass

    def add_to_surface_cells(self, xyz, normals):
        pass


def _args():
    from types import SimpleNamespace

    return SimpleNamespace(
        streaming_insert_voxel_size=0.02,
        streaming_max_new_gaussians_per_frame=100000,
        streaming_surface_cell_dedup=False,
        cap_max=500000,
    )


def _candidate(n_pts: int, birth_frame: int) -> dict:
    rng = np.random.default_rng(11)
    return {
        "pts": rng.normal(size=(n_pts, 3)).astype(np.float32),
        "cols": rng.uniform(size=(n_pts, 3)).astype(np.float32),
        "log_scales": np.zeros((n_pts, 3), dtype=np.float32),
        "q_world": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_pts, 1)),
        "birth_frame": birth_frame,
        "init_opacity": 0.5,
    }


def test_flush_insertion_batch_stamps_the_real_iteration_as_birth_iter():
    gaussians = _RecordingGaussians()
    added = flush_insertion_batch(
        [_candidate(4, birth_frame=7)],
        gaussians,
        _StubStreamingScene(),
        iteration=1234,
        args=_args(),
    )
    assert added == 4
    assert gaussians.birth_iter_seen == 1234


def test_birth_iter_is_the_iteration_not_the_source_frame_index():
    """The regression this file exists for: the two carry different units."""
    gaussians = _RecordingGaussians()
    flush_insertion_batch(
        [_candidate(4, birth_frame=7)],
        gaussians,
        _StubStreamingScene(),
        iteration=1234,
        args=_args(),
    )
    assert gaussians.birth_frame_seen == 7
    assert gaussians.birth_iter_seen == 1234
    assert gaussians.birth_iter_seen != gaussians.birth_frame_seen
