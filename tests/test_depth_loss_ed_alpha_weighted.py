"""Issue #1833: unit tests for the 'ed_alpha_weighted' depth-loss quantity's
CORE mechanism -- exercised directly through the shared, renderer-agnostic
composition (``masked_depth_error`` + ``combine_depth_loss_weights`` +
``weighted_depth_error_mean``) that both ``objective.py``'s single-view path
and ``batched_pixel_terms.py``'s batched path call into with exactly this
sequence (see each module's 'ed_alpha_weighted' branch).

These are CPU-only and do not require CUDA/gsplat: they synthesize
``rend_d``/``alpha`` tensors directly rather than rendering, which is enough
to exercise the loss-composition mechanism itself. A GPU-gated end-to-end
test through a real render call (mirroring
``eligsir/optimization/gaussian/tests/test_depth_loss_quantity_and_gate.py``'s
existing arm A/B/C harness) is out of scope for a CPU-only environment; see
that file for the renderer-level fixture pattern this quantity should also be
added to (arm D) once GPU CI is available.

Mechanism under test (see arguments/__init__.py's streaming_depth_loss_quantity
docstring and gsplat_backend.py's D = ED * alpha comment):

* "d" reads the accumulated/alpha-composited depth ``D = ED * alpha``. At a
  partially-covered pixel (alpha < 1) with an otherwise CORRECT expected
  depth ED, D is biased toward zero relative to the sensor depth, producing a
  nonzero residual whose gradient w.r.t. alpha (opacity) is nonzero -- the
  cheapest descent direction is inflating opacity, not fixing geometry
  (issue #1833's root cause).
* "ed_alpha_weighted" instead reads the alpha-normalised ED directly (so a
  correct ED gives a zero residual regardless of alpha) and weights each
  valid pixel's residual by its DETACHED rendered alpha, so the loss also
  cannot be reduced merely by moving alpha -- the weight itself carries no
  gradient.
"""

from __future__ import annotations

import math

import torch

from eligsir.optimization.gaussian.depth_loss_weighting import (
    combine_depth_loss_weights,
    masked_depth_error,
    weighted_depth_error_mean,
)


def test_d_mode_partial_alpha_with_correct_ed_has_nonzero_residual_and_gradient():
    """A pixel with alpha=0.5 and a CORRECT expected depth (ED == sensor)
    still gets a nonzero 'd'-mode residual, because D = ED * alpha != ED at
    partial coverage -- and that residual's gradient w.r.t. the
    opacity-derived alpha is nonzero, so gradient descent would push alpha up
    (issue #1833's "cheapest descent is to inflate opacity" claim, isolated
    to a single pixel/single Gaussian toy).
    """
    o = torch.tensor(0.5, requires_grad=True)
    alpha = o.reshape(1, 1, 1)
    # ED is, by the gsplat weighted-average formula, invariant to a uniform
    # rescale of the opacities that explain a pixel (the scale cancels
    # between numerator and denominator) -- modeled here as a plain constant
    # w.r.t. `o`, exactly mirroring that invariance without re-deriving the
    # renderer math in a CPU test.
    ED = torch.full((1, 1, 1), 2.0)
    sensor = torch.full((1, 1, 1), 2.0)
    D = ED * alpha  # gsplat_backend.py's exact reconstruction formula

    valid = (sensor > 0) & (D.detach() > 0)
    depth_err = masked_depth_error(D, sensor, valid, loss_type="huber")
    L_d = weighted_depth_error_mean(depth_err, None, valid)

    raw_residual = 1.0  # |2*0.5 - 2|
    delta = 0.1  # masked_depth_error's fixed huber delta
    expected_huber = delta * (abs(raw_residual) - 0.5 * delta)  # outside-delta branch
    assert math.isclose(float(depth_err[0, 0, 0].detach()), expected_huber, rel_tol=1e-6)
    assert float(L_d.detach()) > 0.0

    L_d.backward()
    assert o.grad is not None
    assert float(o.grad) != 0.0


def test_ed_alpha_weighted_same_pixel_has_zero_residual_and_zero_gradient():
    """The identical pixel/scenario as above, but read through ED (not D)
    and weighted by the detached alpha: zero residual (ED is already
    correct) and therefore zero gradient w.r.t. the opacity-derived alpha --
    unlike the 'd'-mode case above.
    """
    o = torch.tensor(0.5, requires_grad=True)
    alpha = o.reshape(1, 1, 1)
    ED = torch.full((1, 1, 1), 2.0)
    sensor = torch.full((1, 1, 1), 2.0)

    valid = (sensor > 0) & (ED.detach() > 0)
    depth_err = masked_depth_error(ED, sensor, valid, loss_type="huber")
    alpha_weight = alpha.detach()  # objective.py/batched_pixel_terms.py always detach before use
    combined_weights = combine_depth_loss_weights(alpha_weight)
    L_ed = weighted_depth_error_mean(depth_err, combined_weights, valid)

    assert float(depth_err[0, 0, 0]) == 0.0
    assert float(L_ed) == 0.0
    # `o` never entered the loss graph at all (ED doesn't depend on it, and
    # the alpha weight was detached before use) -- L_ed has no grad_fn
    # connecting back to `o`, the strongest possible form of "zero
    # gradient": there is no path to differentiate through in the first
    # place, unlike the 'd'-mode case above where D = ED * alpha does
    # depend on `o` and backward() succeeds with a nonzero result.
    assert L_ed.grad_fn is None
    assert o.grad is None


def test_ed_alpha_weighted_normalises_by_sum_of_weights_not_pixel_count():
    """Two valid pixels with different residuals AND different alpha
    weights: the aggregate must equal sum(err * alpha) / sum(alpha) (issue
    #1833's "normalised by the sum of weights"), not the plain unweighted
    mean -- proving the weighting is load-bearing, not a no-op.
    """
    ED = torch.tensor([[[1.5, 3.5]]])
    sensor = torch.tensor([[[2.0, 2.0]]])
    alpha = torch.tensor([[[0.2, 0.8]]])

    valid = (sensor > 0) & (ED.detach() > 0)
    depth_err = masked_depth_error(ED, sensor, valid, loss_type="l1")
    combined_weights = combine_depth_loss_weights(alpha.detach())
    L_ed = weighted_depth_error_mean(depth_err, combined_weights, valid)

    expected_weighted = (0.5 * 0.2 + 1.5 * 0.8) / (0.2 + 0.8)  # = 1.3
    unweighted_mean = (0.5 + 1.5) / 2.0  # = 1.0
    assert math.isclose(float(L_ed), expected_weighted, rel_tol=1e-6)
    assert not math.isclose(float(L_ed), unweighted_mean, rel_tol=1e-6)


def test_ed_alpha_weighted_gate_excludes_pixel_entirely_not_just_downweights():
    """alpha_gate and the ed_alpha_weighted per-pixel weight are orthogonal
    (issue #1833's "gate interplay"): a pixel below the gate must be fully
    excluded from `valid` (as if it never existed), not merely assigned a
    small weight -- the two knobs compose, they don't substitute for each
    other.
    """
    ED = torch.tensor([[[2.0, 2.0, 2.0]]])
    sensor = torch.tensor([[[2.0, 5.0, 8.0]]])  # residuals: 0.0, 3.0, 6.0
    alpha = torch.tensor([[[0.05, 0.3, 0.9]]])  # pixel 0 is below the gate below
    gate = 0.1

    valid = (sensor > 0) & (ED.detach() > 0)
    valid = valid & (alpha.detach() > gate)
    assert not bool(valid[0, 0, 0])  # excluded: alpha 0.05 <= gate 0.1
    assert bool(valid[0, 0, 1]) and bool(valid[0, 0, 2])

    depth_err = masked_depth_error(ED, sensor, valid, loss_type="l1")
    combined_weights = combine_depth_loss_weights(alpha.detach())
    L_ed = weighted_depth_error_mean(depth_err, combined_weights, valid)

    expected = (3.0 * 0.3 + 6.0 * 0.9) / (0.3 + 0.9)
    assert math.isclose(float(L_ed), expected, rel_tol=1e-6)


def test_ed_alpha_weighted_valid_sample_count_excludes_gated_pixel():
    """The exact accepted-sample count (objective.py's
    `_guard_acc.depth_valid_sample_count_device = valid.count_nonzero()`)
    must reflect the gate, not the weighting -- gated-out pixels are not
    'valid at reduced weight', they are not valid at all.
    """
    ED = torch.tensor([[[2.0, 2.0, 2.0]]])
    sensor = torch.tensor([[[2.0, 5.0, 8.0]]])
    alpha = torch.tensor([[[0.05, 0.3, 0.9]]])
    gate = 0.1

    valid = (sensor > 0) & (ED.detach() > 0) & (alpha.detach() > gate)
    assert int(valid.count_nonzero()) == 2
