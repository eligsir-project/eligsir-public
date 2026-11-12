import torch

from eligsir.utils.strategies.mcmc.proposal import append_running_state
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_MCMC_GROW


class _FakeGaussians:
    def __init__(self, n):
        self.gaussian_uid = torch.arange(n, dtype=torch.int64)
        self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
        self.root_uid = self.gaussian_uid.clone()
        self.generation = torch.zeros(n, dtype=torch.int32)
        self.birth_method = torch.zeros(n, dtype=torch.int8)
        self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
        self.birth_iter = torch.zeros(n, dtype=torch.int32)
        self.birth_frame = torch.zeros(n, dtype=torch.int32)
        self._next_gaussian_uid = n
        self.get_xyz = torch.zeros(n, 3)
        # other buffers append_running_state touches, kept minimal but present
        for name in ("visibility_ema", "xyz_gradient_accum", "denom", "support_count",
                     "provisional", "anchor_iter", "depth_conflict_count", "lifecycle_state", "utility_ema"):
            setattr(self, name, torch.zeros(n))
        self.max_radii2D = torch.zeros(n)


def test_append_running_state_extends_birth_iter_and_lineage():
    g = _FakeGaussians(5)
    parent_idxs = torch.tensor([2, 4], dtype=torch.int64)  # 2 clones
    append_running_state(g, old_count=5, added_count=2, parent_idxs=parent_idxs, iteration=12300)

    assert g.birth_iter.shape[0] == 7
    assert torch.equal(g.birth_iter[-2:].to(torch.int64), torch.tensor([12300, 12300], dtype=torch.int64))
    assert g.parent_uid.shape[0] == 7
    assert torch.equal(g.parent_uid[-2:], torch.tensor([2, 4], dtype=torch.int64))  # gaussian_uid == index here
    assert torch.equal(g.root_uid[-2:], torch.tensor([2, 4], dtype=torch.int64))
    assert torch.equal(g.generation[-2:], torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(g.birth_method[-2:], torch.full((2,), BIRTH_METHOD_MCMC_GROW, dtype=torch.int8))
