"""Regression test for the seed-fade fix: --streaming_count_loss_local gating.

The global MCMC effective-count / opacity-entropy losses take the FULL _opacity
tensor, so they push opacity DOWN on out-of-view geometry that has no reconstruction
loss to resist it (the traveling-scene seed fade). The fix detaches the gradient on
inactive opacities while keeping the loss *value* global (so N_eff is unchanged).

This test pins the detach-gating semantics used inline in train_streaming.py:
gradient flows only to active rows; inactive rows get zero gradient.
"""
import torch


def _gate(opacity, active):
    """Mirror of the inline gating in train_streaming.py."""
    return torch.where(active.unsqueeze(-1), opacity, opacity.detach())


def test_gate_grad_only_active():
    op = torch.zeros(4, 1, requires_grad=True)
    active = torch.tensor([True, False, True, False])
    op_in = _gate(op, active)
    # global-style loss touches all rows numerically
    loss = (op_in.sigmoid().sum()) ** 2
    loss.backward()
    g = op.grad.squeeze()
    assert torch.all(g[[1, 3]] == 0), "inactive rows must have zero gradient"
    assert torch.all(g[[0, 2]] != 0), "active rows must receive gradient"


def test_gate_value_is_global():
    # The forward value must be identical to the ungated tensor (N_eff unchanged).
    op = torch.randn(8, 1)
    active = torch.tensor([True, False] * 4)
    op_in = _gate(op.clone().requires_grad_(True), active)
    assert torch.allclose(op_in.detach(), op), "gating must not change the loss value"
