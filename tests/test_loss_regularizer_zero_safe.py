from types import SimpleNamespace

import torch

from eligsir.optimization.gaussian.losses import (
    _streaming_young_isotropy_reg,
    _streaming_young_opacity_reg,
)


class _Gaussians:
    def __init__(self):
        self.get_xyz = torch.zeros((3, 3), dtype=torch.float32)
        self.get_scaling = torch.ones((3, 3), dtype=torch.float32)
        self.get_opacity = torch.full((3, 1), 0.5, dtype=torch.float32)
        self.lifecycle_state = torch.full((3,), 2, dtype=torch.int32)
        self.birth_frame = torch.zeros((3,), dtype=torch.int32)
        self.birth_iter = torch.zeros((3,), dtype=torch.int32)
        self.support_count = torch.zeros((3,), dtype=torch.float32)
        self.last_supervised_iter = torch.zeros((3,), dtype=torch.int32)


def test_young_opacity_empty_active_subset_is_exact_finite_zero():
    gaussians = _Gaussians()
    args = SimpleNamespace(
        opacity_reg=0.01,
        streaming_recency_reg_horizon=0,
        streaming_debt_replay_target_support=10,
        streaming_maturity_reg_ramp_min=0.2,
        streaming_recency_reg_ramp_min=0.2,
    )
    active = torch.zeros((3,), dtype=torch.bool)
    loss = _streaming_young_opacity_reg(
        gaussians,
        args,
        optimizer_update_id=10,
        active_mask=active,
        young_exempt_iters=400,
        maturity_ramp=True,
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_young_isotropy_no_immature_rows_is_exact_finite_zero():
    gaussians = _Gaussians()
    args = SimpleNamespace(
        streaming_young_isotropy_weight=0.05,
        streaming_young_isotropy_ramp_frames=15,
    )
    loss = _streaming_young_isotropy_reg(gaussians, args, n_frames_ingested=20)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
