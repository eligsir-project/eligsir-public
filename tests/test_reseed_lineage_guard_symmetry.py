"""M8: ``reset_relocated_state``'s lineage guard must be as defensive as the
growth path's.

The growth guard in ``append_running_state`` checks BOTH ``gaussian_uid`` and
``parent_uid`` are present and correctly sized before touching lineage. The
reseed guard checked only ``parent_uid``, then indexed ``gaussians.gaussian_uid``
directly (not via ``getattr``) -- so a model missing or with a stale-length
``gaussian_uid`` raised instead of skipping.

CPU-only: plain tensors on a fake model, no CUDA.
"""

from __future__ import annotations

import torch

from eligsir.utils.gaussian_lineage import BIRTH_METHOD_MCMC_RESEED
from eligsir.utils.strategies.mcmc.proposal import reset_relocated_state


class _FakeGaussians:
    def __init__(self, n, *, with_uid=True, uid_len=None):
        if with_uid:
            self.gaussian_uid = torch.arange(uid_len if uid_len is not None else n, dtype=torch.int64)
        self.parent_uid = torch.full((n,), -1, dtype=torch.int64)
        self.root_uid = torch.arange(n, dtype=torch.int64)
        self.generation = torch.zeros(n, dtype=torch.int32)
        self.birth_method = torch.zeros(n, dtype=torch.int8)
        self.last_reseed_iter = torch.full((n,), -1, dtype=torch.int32)
        self.birth_iter = torch.zeros(n, dtype=torch.int32)
        self.get_xyz = torch.zeros(n, 3)
        self.visibility_ema = torch.zeros(n, 1)
        self.provisional = torch.zeros(n, dtype=torch.bool)
        self.birth_frame = torch.zeros(n, dtype=torch.int32)
        self.support_count = torch.zeros(n, dtype=torch.int16)


DEAD = torch.tensor([0], dtype=torch.int64)
DONOR = torch.tensor([2], dtype=torch.int64)


def test_reseed_still_updates_lineage_when_uid_present():
    g = _FakeGaussians(4)
    reset_relocated_state(g, DEAD, DONOR, iteration=777)
    assert g.parent_uid[0].item() == 2
    assert g.birth_method[0].item() == BIRTH_METHOD_MCMC_RESEED
    assert g.last_reseed_iter[0].item() == 777


def test_reseed_skips_lineage_when_gaussian_uid_missing():
    g = _FakeGaussians(4, with_uid=False)
    reset_relocated_state(g, DEAD, DONOR, iteration=777)  # must not raise
    assert g.parent_uid[0].item() == -1  # untouched
    assert g.birth_method[0].item() == 0


def test_reseed_skips_lineage_when_gaussian_uid_length_stale():
    g = _FakeGaussians(4, uid_len=3)
    reset_relocated_state(g, DEAD, DONOR, iteration=777)  # must not raise
    assert g.parent_uid[0].item() == -1
    assert g.birth_method[0].item() == 0
    # birth_iter is guarded independently of lineage and still refreshes.
    assert g.birth_iter[0].item() == 777
