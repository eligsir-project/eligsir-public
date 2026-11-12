import torch
from eligsir.utils.strategies.mcmc.proposal import reset_relocated_state
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_MCMC_RESEED


class _FakeGaussians:
    def __init__(self, n):
        self.gaussian_uid = torch.arange(n, dtype=torch.int64)
        self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
        self.root_uid = self.gaussian_uid.clone()
        self.generation = torch.zeros(n, dtype=torch.int32)
        self.birth_method = torch.zeros(n, dtype=torch.int8)
        self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
        self.birth_iter = torch.zeros(n, dtype=torch.int32)
        self.get_xyz = torch.zeros(n, 3)
        self.visibility_ema = torch.zeros(n, 1)
        self.provisional = torch.zeros(n, dtype=torch.bool)
        self.birth_frame = torch.zeros(n, dtype=torch.int32)
        self.support_count = torch.zeros(n, dtype=torch.int16)


def test_reset_relocated_state_updates_lineage_and_keeps_uid():
    g = _FakeGaussians(4)
    dead_indices = torch.tensor([0], dtype=torch.int64)
    sampled_idxs = torch.tensor([2], dtype=torch.int64)
    original_uid_at_dead_slot = g.gaussian_uid[0].item()

    reset_relocated_state(g, dead_indices, sampled_idxs, iteration=12300)

    assert g.gaussian_uid[0].item() == original_uid_at_dead_slot  # UID never changes on reseed
    assert g.parent_uid[0].item() == 2   # donor's gaussian_uid
    assert g.root_uid[0].item() == g.root_uid[2].item()
    assert g.generation[0].item() == 1  # donor (slot 2) generation was 0, reseeded slot becomes 0 + 1 = 1
    assert g.birth_method[0].item() == BIRTH_METHOD_MCMC_RESEED
    assert g.last_reseed_iter[0].item() == 12300
    assert g.birth_iter[0].item() == 12300
