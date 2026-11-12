"""Contracts for the opt-in, display-referred train-frame EV/WB axis."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eligsir.optimization.gaussian import engine, objective


def test_display_referred_correction_is_identity_at_zero_and_rejects_unsafe_support():
    image = torch.tensor(
        [[[0.2, 0.0], [0.8, float("nan")]], [[0.3, 0.2], [0.8, 0.3]], [[0.4, 0.2], [0.8, 0.3]]],
        requires_grad=True,
    )
    target = torch.full_like(image, 0.5)
    state = torch.zeros(3, requires_grad=True)
    corrected, prior = objective._display_referred_evwb(
        image, target, state, display_margin=1.0 / 255.0, prior_weight=0.1,
    )

    torch.testing.assert_close(corrected[:, 0, 0], image[:, 0, 0])
    assert torch.isnan(corrected[0, 1, 1])
    (corrected[:, 0, 0].sum() + prior).backward()
    assert torch.isfinite(state.grad).all()


def test_evwb_commit_projects_once_and_transaction_cleanup_releases_gradient():
    state = torch.zeros(3, requires_grad=True)
    state.grad = torch.tensor([-100.0, -100.0, 100.0])
    args = SimpleNamespace(
        streaming_train_evwb_learning_rate=0.1,
        streaming_train_evwb_max_ev=2.0,
        streaming_train_evwb_max_log_wb=1.0,
    )

    engine._commit_evwb_states({id(state): state}, args)

    assert state.grad is not None
    assert abs(float(state[0].detach())) <= 2.0
    wb = torch.stack((state[1], state[2], -state[1] - state[2]))
    assert float(wb.abs().max().detach()) <= 1.0
    assert torch.isclose(wb.sum(), torch.zeros(()))

    engine._clear_evwb_state_grads({id(state): state})

    assert state.grad is None


def test_evwb_commit_rejects_a_nonfinite_projected_candidate():
    """Would catch finite update inputs persisting an overflowed WB candidate."""
    state = torch.zeros(3, requires_grad=True)
    state.grad = torch.tensor([0.0, 3e38, -3e38])
    args = SimpleNamespace(
        streaming_train_evwb_learning_rate=2.0,
        streaming_train_evwb_max_ev=2.0,
        streaming_train_evwb_max_log_wb=1.0,
    )

    engine._commit_evwb_states({id(state): state}, args)

    torch.testing.assert_close(state, torch.zeros_like(state))
    assert torch.isfinite(state).all()


def test_evwb_transaction_cleanup_runs_for_base_exception():
    class _Cancelled(BaseException):
        pass

    state = torch.zeros(3, requires_grad=True)
    state.grad = torch.ones_like(state)

    with pytest.raises(_Cancelled):
        with engine._evwb_gradient_transaction({id(state): state}):
            raise _Cancelled()

    assert state.grad is None


def test_evwb_configuration_requires_predeclared_reference_identity():
    with pytest.raises(ValueError, match="reference source identity"):
        objective._evwb_config(SimpleNamespace(streaming_train_evwb_enabled=True))


@pytest.mark.parametrize(
    "initial_state",
    [
        [float("nan"), 1000.0, 0.0],
        [float("inf"), 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        [-1000.0, 0.0, 0.0],
    ],
    ids=["nan_with_overflowing_wb", "infinite_ev", "finite_overflowing_ev", "finite_underflowing_ev"],
)
def test_display_referred_evwb_invalid_state_falls_back_without_poisoning_backward(initial_state):
    """Invalid state must not leak an unselected overflowing EV/WB branch into backward."""
    image = torch.full((3, 2, 2), 0.4, requires_grad=True)
    target = torch.full_like(image, 0.5)
    state = torch.tensor(initial_state, requires_grad=True)

    corrected, prior = objective._display_referred_evwb(
        image, target, state, display_margin=1.0 / 255.0, prior_weight=0.1,
    )

    torch.testing.assert_close(corrected, image)
    assert torch.isfinite(prior)
    (corrected.sum() + prior).backward()

    assert image.grad is not None and torch.isfinite(image.grad).all()
    torch.testing.assert_close(image.grad, torch.ones_like(image))
    assert state.grad is not None and torch.isfinite(state.grad).all()
    torch.testing.assert_close(state.grad, torch.zeros_like(state))
    assert torch.equal(torch.isfinite(state), torch.isfinite(torch.tensor(initial_state)))
    assert state[1].detach() == initial_state[1]


@pytest.mark.parametrize(
    "dtype,value,initial_state,margin",
    [
        (torch.float16, 0.01, [-5.0, 5.0, 5.0], 1.0 / 255.0),
        (torch.float32, 1e-10, [-42.0, 42.0, 42.0], 1e-12),
    ],
    ids=["half_precision_product_underflow", "single_precision_product_underflow"],
)
def test_evwb_underflowed_correction_keeps_raw_fallback_gradients(dtype, value, initial_state, margin):
    """A finite gain can underflow its product and poison an unselected power branch."""
    image = torch.full((3, 2, 2), value, dtype=dtype, requires_grad=True)
    state = torch.tensor(initial_state, dtype=dtype, requires_grad=True)

    corrected, prior = objective._display_referred_evwb(
        image, torch.full_like(image, 0.5), state, display_margin=margin, prior_weight=0.1,
    )

    torch.testing.assert_close(corrected, image)
    torch.testing.assert_close(prior, torch.zeros_like(prior))
    (corrected.sum() + prior).backward()

    torch.testing.assert_close(image.grad, torch.ones_like(image))
    torch.testing.assert_close(state.grad, torch.zeros_like(state))
