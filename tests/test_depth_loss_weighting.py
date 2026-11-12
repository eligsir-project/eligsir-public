import torch

from eligsir.optimization.gaussian.depth_loss_weighting import (
    combine_depth_loss_weights,
    masked_depth_error,
    weighted_depth_error_mean,
)


def test_combine_depth_loss_weights_all_none_returns_none():
    assert combine_depth_loss_weights(None, None) is None


def test_combine_depth_loss_weights_single_weight_passthrough():
    w = torch.tensor([1.0, 0.5, 0.0])
    combined = combine_depth_loss_weights(w, None)
    assert torch.equal(combined, w)


def test_combine_depth_loss_weights_multiplies_present_weights():
    a = torch.tensor([1.0, 0.5, 0.2])
    b = torch.tensor([0.5, 0.5, 1.0])
    combined = combine_depth_loss_weights(a, None, b)
    assert torch.allclose(combined, a * b)


def test_weighted_depth_error_mean_with_none_weights_is_plain_mean():
    depth_err = torch.tensor([1.0, 2.0, 3.0])
    result = weighted_depth_error_mean(depth_err, None)
    assert torch.isclose(result, depth_err.mean())


def test_weighted_depth_error_mean_matches_legacy_prior_conf_only_formula():
    prior_conf = torch.tensor([[1.0, 0.0, 0.5, 0.9]])
    valid = torch.tensor([[True, True, True, False]])
    rend_d = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    sensor_d = torch.tensor([[1.1, 1.9, 3.5, 0.0]])
    depth_err = (rend_d[valid] - sensor_d[valid]).abs()
    legacy_weights = prior_conf[valid].clamp_min(1e-4)
    expected = (depth_err * legacy_weights).sum() / legacy_weights.sum()
    assert torch.isclose(weighted_depth_error_mean(depth_err, prior_conf, valid), expected)


def test_dense_masked_mean_matches_compacted_mean():
    depth_err = torch.tensor([[0.1, 0.4, 1.2, 0.8]])
    valid = torch.tensor([[True, False, True, False]])
    assert torch.isclose(
        weighted_depth_error_mean(depth_err, None, valid), depth_err[valid].mean()
    )


def test_dense_masked_weighted_mean_matches_compacted_formula():
    depth_err = torch.tensor([[0.1, 0.4, 1.2, 0.8]])
    valid = torch.tensor([[True, False, True, False]])
    weights = torch.tensor([[1.0, 0.2, 0.5, 0.1]])
    compact = weights[valid].clamp_min(1e-4)
    expected = (depth_err[valid] * compact).sum() / compact.sum()
    assert torch.isclose(weighted_depth_error_mean(depth_err, weights, valid), expected)


def test_dense_empty_mask_returns_exact_finite_zero():
    depth_err = torch.tensor([[float("nan"), 2.0]])
    valid = torch.zeros_like(depth_err, dtype=torch.bool)
    result = weighted_depth_error_mean(depth_err, None, valid)
    assert torch.isfinite(result)
    assert result.item() == 0.0


def test_dense_empty_weighted_mask_returns_exact_finite_zero():
    depth_err = torch.tensor([[3.0, 2.0]])
    valid = torch.zeros_like(depth_err, dtype=torch.bool)
    weights = torch.tensor([[1.0, 0.5]])
    result = weighted_depth_error_mean(depth_err, weights, valid)
    assert torch.isfinite(result)
    assert result.item() == 0.0


def test_masked_huber_disconnects_invalid_nonfinite_rays_from_backward():
    rendered = torch.tensor([[float("nan"), 2.0]], requires_grad=True)
    sensor = torch.tensor([[1.0, 1.0]])
    valid = torch.tensor([[False, True]])
    error = masked_depth_error(rendered, sensor, valid, loss_type="huber")
    loss = weighted_depth_error_mean(error, None, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad[0, 0].item() == 0.0


def test_masked_depth_error_rejects_unknown_loss_type():
    rendered = torch.ones((1, 2))
    sensor = torch.ones((1, 2))
    valid = torch.ones((1, 2), dtype=torch.bool)
    try:
        masked_depth_error(rendered, sensor, valid, loss_type="unknown")
    except ValueError as exc:
        assert "unsupported depth loss type" in str(exc)
    else:
        raise AssertionError("unknown depth loss type was accepted")


def test_weighted_depth_error_mean_downweights_discontinuity_flagged_pixels():
    depth_err = torch.tensor([1.0, 1.0, 1.0])
    disc_weight = torch.tensor([0.01, 1.0, 1.0])
    result = weighted_depth_error_mean(depth_err, disc_weight)
    expected = (depth_err * disc_weight.clamp_min(1e-4)).sum() / disc_weight.clamp_min(1e-4).sum()
    assert torch.isclose(result, expected)


def test_weighted_depth_error_mean_excludes_zero_weight_pixel_almost_entirely():
    depth_err = torch.tensor([100.0, 1.0])
    disc_weight = torch.tensor([0.0, 1.0])
    assert weighted_depth_error_mean(depth_err, disc_weight).item() < 1.02
