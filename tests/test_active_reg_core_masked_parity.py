"""Parity/gradient tests for ``active_reg_core_masked`` (issue #1670).

``active_reg_core`` (the sparse active-set L1 regularizer core used in
``objective.assemble_loss``'s ``sparse_active_set`` branch) used to be called
against a caller-side dynamic boolean compaction:

    active_reg_core(gaussians.get_opacity[_active], gaussians.get_scaling[_active], ...)

``tensor[_active]`` is a data-dependent CUDA allocation whose output
cardinality changes every iteration. ``active_reg_core_masked`` replaces that
compaction with fixed full-``[N, ...]``-shape masked math (mask as a
multiplicative device weight, safely-clamped denominator), which
``objective.assemble_loss`` now calls directly instead of pre-slicing.

These tests pin exact parity with the old ``tensor[mask]`` + ``active_reg_core``
compaction across empty/one/many/all-active masks, on both loss value and the
gradient each produces on the SAME full-population leaf tensors, and pin that
gradients outside the active mask are exactly zero.
"""
from __future__ import annotations

import torch

from eligsir.utils.compiled_kernels import active_reg_core, active_reg_core_masked


def _make_inputs(n: int, active_indices, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    opacity = torch.rand(n, 1, generator=g, dtype=torch.float32) * 2.0 - 1.0
    scaling = torch.rand(n, 3, generator=g, dtype=torch.float32) * 2.0 - 1.0
    mask = torch.zeros(n, dtype=torch.bool)
    if active_indices:
        mask[list(active_indices)] = True
    return opacity, scaling, mask


def _old_way(opacity, scaling, mask, **kw):
    """Pre-#1670 call shape: caller slices via the dynamic boolean mask."""
    return active_reg_core(opacity[mask], scaling[mask], **kw)


def _assert_grad_matches(old_grad, new_grad):
    """``active_reg_core``'s empty-subset early exit (``opacity.numel() == 0
    or scaling.numel() == 0: return scaling.sum() * 0.0``) does not
    reference ``opacity`` at all when the active slice is empty, so its
    ``opacity.grad`` stays ``None`` -- whereas the masked full-population
    form always stays connected to both leaves. ``None`` and an
    all-zero grad are the same "no contribution" fact, so treat them as
    equal here rather than asserting identical grad-population behavior."""
    if old_grad is None:
        assert new_grad is None or torch.equal(new_grad, torch.zeros_like(new_grad))
        return
    assert torch.allclose(old_grad, new_grad, atol=1e-6, rtol=1e-5)


def _compare(n, active_indices, *, w_opacity=0.37, w_scale=0.11, w_aniso=0.05, seed=0):
    opacity, scaling, mask = _make_inputs(n, active_indices, seed=seed)

    op_old = opacity.clone().requires_grad_(True)
    sc_old = scaling.clone().requires_grad_(True)
    loss_old = _old_way(op_old, sc_old, mask, w_opacity=w_opacity, w_scale=w_scale, w_aniso=w_aniso)
    loss_old.backward()

    op_new = opacity.clone().requires_grad_(True)
    sc_new = scaling.clone().requires_grad_(True)
    loss_new = active_reg_core_masked(op_new, sc_new, mask, w_opacity=w_opacity, w_scale=w_scale, w_aniso=w_aniso)
    loss_new.backward()

    assert torch.allclose(loss_old, loss_new, atol=1e-6, rtol=1e-5), (loss_old, loss_new)
    _assert_grad_matches(op_old.grad, op_new.grad)
    _assert_grad_matches(sc_old.grad, sc_new.grad)
    return op_new, sc_new


def test_empty_active_mask_is_exact_connected_zero():
    op, sc = _compare(n=20, active_indices=[])
    assert op.grad is not None and sc.grad is not None  # connected to the graph
    assert torch.equal(op.grad, torch.zeros_like(op.grad))
    assert torch.equal(sc.grad, torch.zeros_like(sc.grad))


def test_one_active_row():
    _compare(n=20, active_indices=[7])


def test_many_active_rows():
    _compare(n=50, active_indices=list(range(0, 50, 3)))


def test_all_active_rows():
    _compare(n=15, active_indices=list(range(15)))


def test_gradients_outside_mask_are_exactly_zero():
    n = 30
    opacity, scaling, mask = _make_inputs(n, active_indices=[1, 4, 5, 9, 20])
    opacity.requires_grad_(True)
    scaling.requires_grad_(True)
    loss = active_reg_core_masked(opacity, scaling, mask, w_opacity=0.4, w_scale=0.2, w_aniso=0.1)
    loss.backward()
    inactive = ~mask
    assert torch.equal(opacity.grad[inactive], torch.zeros_like(opacity.grad[inactive]))
    assert torch.equal(scaling.grad[inactive], torch.zeros_like(scaling.grad[inactive]))
    # Sanity: at least one active row actually received a nonzero gradient
    # (otherwise the "outside mask is zero" check would be vacuous).
    assert bool((opacity.grad[mask] != 0).any() or (scaling.grad[mask] != 0).any())


def test_w_aniso_zero_skips_aniso_term_but_still_parity():
    _compare(n=25, active_indices=[2, 3, 11, 24], w_aniso=0.0)


def test_zero_population():
    opacity = torch.zeros(0, 1, dtype=torch.float32, requires_grad=True)
    scaling = torch.zeros(0, 3, dtype=torch.float32, requires_grad=True)
    mask = torch.zeros(0, dtype=torch.bool)
    loss = active_reg_core_masked(opacity, scaling, mask, w_opacity=0.5, w_scale=0.5, w_aniso=0.1)
    assert torch.equal(loss, torch.zeros(()))


def test_objective_sparse_branch_no_longer_slices_active_set_before_reg_core():
    """Static regression: the sparse-active-set call sites in objective.py
    must pass the full population + mask to ``active_reg_core_masked``, not
    a pre-sliced ``gaussians.get_opacity[_active]``/``get_scaling[_active]``
    (issue #1670's cited example of dynamic-cardinality boolean compaction).
    """
    import inspect

    from eligsir.optimization.gaussian import objective

    src = inspect.getsource(objective)
    assert "active_reg_core_masked(" in src
    assert "gaussians.get_opacity[_active]" not in src
    assert "gaussians.get_scaling[_active]" not in src
