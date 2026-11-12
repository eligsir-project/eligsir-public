"""CPU parity coverage for ``assemble_batched_pixel_terms`` (issue #1670
Phase B -- see ``.superpowers/sdd/2026-09-03-throughput-recovery-tracks-1-2/
batched-pixel-terms-brief.md`` and the spec's Design section,
``docs/superpowers/specs/2026-09-04-batched-supervision-objective-design.md``).

Mirrors ``eligsir-lab/tests/test_batched_supervision_loss.py`` (the
reference proof for what "batched" must mean numerically) against this
repo's actual per-view helpers: ``utils.loss_utils.l1_loss``/
``evaluation_ssim`` (the fixed reference conv-SSIM, independent of the
mutable training-loss SSIM backend so these tests are deterministic
regardless of what a prior test configured via ``set_ssim_mode``) and
``depth_loss_weighting.masked_depth_error``/``weighted_depth_error_mean``.

The depth term MUST be normalised PER VIEW by that view's own valid-pixel
count, never over the whole stack -- a stack-wide masked mean silently
reweights views by their valid-pixel counts and is a different objective.
``test_depth_term_is_normalised_per_view_not_over_the_stack`` pins this with
a nearly-empty view, the same construction as the lab proof.
"""

from __future__ import annotations

import torch

from eligsir.optimization.gaussian.batched_pixel_terms import (
    PixelTermsView,
    assemble_batched_pixel_terms,
)
from eligsir.optimization.gaussian.depth_loss_weighting import (
    combine_depth_loss_weights,
    masked_depth_error,
    weighted_depth_error_mean,
)
from eligsir.utils.loss_utils import evaluation_ssim, l1_loss


def _demo_inputs(seed: int, *, b: int = 4, h: int = 12, w: int = 16):
    gen = torch.Generator().manual_seed(seed)
    gts = torch.rand(b, 3, h, w, generator=gen)
    renders = (gts + 0.1 * torch.randn(b, 3, h, w, generator=gen)).clamp(0, 1)
    renders.requires_grad_(True)
    sensor = torch.rand(b, 1, h, w, generator=gen) * 3.0
    sensor[:, :, : h // 4] = 0.0
    rendered_depth = (sensor + 0.05 * torch.randn(b, 1, h, w, generator=gen)).clamp_min(0.0)
    rendered_depth.requires_grad_(True)
    return {
        "renders": renders,
        "gts": gts,
        "rendered_depth": rendered_depth,
        "sensor_depth": sensor,
    }


def _per_view_reference(
    inputs: dict,
    *,
    alpha: torch.Tensor | None = None,
    alpha_gate: float = 0.0,
    depth_loss_type: str = "l1",
    weight_maps: tuple = (),
):
    """What ``objective.assemble_loss``'s per-view path computes for each
    view: ``l1_loss``, the fixed reference SSIM, and the masked depth term
    (``masked_depth_error`` + ``weighted_depth_error_mean``) -- returned as
    per-view lists so a test can compare element-by-element against
    ``assemble_batched_pixel_terms``'s stacked output."""
    b = inputs["renders"].shape[0]
    ll1s, ssims, depths, valid_counts = [], [], [], []
    have_depth = inputs.get("rendered_depth") is not None and inputs.get("sensor_depth") is not None
    for i in range(b):
        render_i = inputs["renders"][i]
        gt_i = inputs["gts"][i]
        ll1s.append(l1_loss(render_i, gt_i))
        ssims.append(evaluation_ssim(render_i, gt_i))
        if have_depth:
            rend_d = inputs["rendered_depth"][i]
            sensor_d = inputs["sensor_depth"][i]
            valid = (sensor_d > 0) & (rend_d.detach() > 0)
            if alpha_gate > 0.0:
                valid = valid & (alpha[i].detach() > alpha_gate)
            depth_err = masked_depth_error(rend_d, sensor_d, valid, loss_type=depth_loss_type)
            weights_i = combine_depth_loss_weights(*(w[i] if w is not None else None for w in weight_maps))
            depths.append(weighted_depth_error_mean(depth_err, weights_i, valid))
            valid_counts.append(valid.sum())
    return ll1s, ssims, depths, valid_counts


def test_batched_terms_match_the_per_view_loop_rgb_only():
    """``Ll1`` matches the per-view loop's own ``l1_loss`` exactly (it is a
    genuinely per-view quantity). ``ssim`` deliberately is NOT compared
    against the per-view reference SSIM here -- by contract every view gets
    the STACK's batch-mean SSIM scalar (see
    ``test_ssim_equals_the_mean_of_per_view_reference_ssims`` below), which
    only coincides with a single view's own SSIM when all views are
    identical."""
    inputs = _demo_inputs(1)
    del inputs["rendered_depth"], inputs["sensor_depth"]
    terms = assemble_batched_pixel_terms(renders=inputs["renders"], gts=inputs["gts"])
    ll1s, _, _, _ = _per_view_reference(inputs)

    assert len(terms) == inputs["renders"].shape[0]
    for view, ll1_ref in zip(terms, ll1s):
        assert isinstance(view, PixelTermsView)
        torch.testing.assert_close(view.Ll1, ll1_ref, atol=1e-6, rtol=1e-5)
        assert view.depth is None
        assert view.depth_valid_count is None


def test_ssim_matches_each_per_view_reference():
    inputs = _demo_inputs(2)
    terms = assemble_batched_pixel_terms(renders=inputs["renders"], gts=inputs["gts"])
    _, ssims, _, _ = _per_view_reference(inputs)
    for view, reference in zip(terms, ssims):
        torch.testing.assert_close(view.ssim, reference, atol=1e-5, rtol=1e-5)


def test_non_finite_secondary_rgb_does_not_poison_primary_ssim_or_gradient():
    inputs = _demo_inputs(12, b=2)
    with torch.no_grad():
        inputs["renders"][1, 0, 0, 0] = float("nan")
    terms = assemble_batched_pixel_terms(renders=inputs["renders"], gts=inputs["gts"])
    assert bool(torch.isfinite(terms[0].ssim))
    assert bool(torch.isnan(terms[1].ssim))
    assert bool(torch.isfinite(terms[0].Ll1))
    assert bool(torch.isnan(terms[1].Ll1))
    (primary_grad,) = torch.autograd.grad(terms[0].ssim, inputs["renders"])
    assert bool(torch.isfinite(primary_grad[0]).all())
    torch.testing.assert_close(primary_grad[1], torch.zeros_like(primary_grad[1]))


def test_batched_depth_matches_the_per_view_loop_exactly():
    inputs = _demo_inputs(3)
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
    )
    ll1s, ssims, depths, valid_counts = _per_view_reference(inputs)

    for view, ll1_ref, depth_ref, count_ref in zip(terms, ll1s, depths, valid_counts):
        torch.testing.assert_close(view.Ll1, ll1_ref, atol=1e-6, rtol=1e-5)
        assert view.depth is not None
        torch.testing.assert_close(view.depth, depth_ref, atol=1e-6, rtol=1e-5)
        assert view.depth_valid_count is not None
        torch.testing.assert_close(view.depth_valid_count, count_ref)


def test_batched_depth_matches_the_per_view_loop_with_huber():
    inputs = _demo_inputs(4)
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        depth_loss_type="huber",
    )
    _, _, depths, _ = _per_view_reference(inputs, depth_loss_type="huber")
    for view, depth_ref in zip(terms, depths):
        torch.testing.assert_close(view.depth, depth_ref, atol=1e-6, rtol=1e-5)


def test_depth_term_is_normalised_per_view_not_over_the_stack():
    """A global masked mean would weight views by their valid-pixel counts;
    the per-view loop does not. Make view 0 almost fully invalid and check
    the batched depth component still equals the loop's per-view value, and
    differs from a naive stack-wide masked mean."""
    inputs = _demo_inputs(5)
    inputs["sensor_depth"][0, :, :, :] = 0.0
    inputs["sensor_depth"][0, :, -1, -1] = 2.0
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
    )
    _, _, depths, _ = _per_view_reference(inputs)
    for view, depth_ref in zip(terms, depths):
        torch.testing.assert_close(view.depth, depth_ref, atol=1e-6, rtol=1e-5)

    valid_stack = (inputs["sensor_depth"] > 0) & (inputs["rendered_depth"].detach() > 0)
    global_masked_mean = (
        torch.where(
            valid_stack,
            (inputs["rendered_depth"] - inputs["sensor_depth"]).abs(),
            torch.zeros_like(inputs["sensor_depth"]),
        ).sum()
        / valid_stack.sum().to(torch.float32)
    )
    assert not torch.isclose(terms[0].depth.detach(), global_masked_mean, rtol=1e-3)


def test_alpha_gate_matches_the_per_view_loop():
    inputs = _demo_inputs(6)
    b, _, h, w = inputs["rendered_depth"].shape
    gen = torch.Generator().manual_seed(60)
    alpha = torch.rand(b, 1, h, w, generator=gen)
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        alpha=alpha, alpha_gate=0.3,
    )
    _, _, depths, valid_counts = _per_view_reference(inputs, alpha=alpha, alpha_gate=0.3)
    for view, depth_ref, count_ref in zip(terms, depths, valid_counts):
        torch.testing.assert_close(view.depth, depth_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(view.depth_valid_count, count_ref)


def test_alpha_gate_positive_without_alpha_raises():
    inputs = _demo_inputs(7)
    try:
        assemble_batched_pixel_terms(
            renders=inputs["renders"], gts=inputs["gts"],
            rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
            alpha_gate=0.3,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for alpha_gate > 0 without alpha")


def test_invalid_depth_quantity_raises():
    inputs = _demo_inputs(8)
    try:
        assemble_batched_pixel_terms(
            renders=inputs["renders"], gts=inputs["gts"],
            rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
            depth_quantity="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unrecognized depth_quantity")


def test_weight_maps_match_the_per_view_loop():
    inputs = _demo_inputs(9)
    b, _, h, w = inputs["rendered_depth"].shape
    gen = torch.Generator().manual_seed(90)
    prior_conf = torch.rand(b, 1, h, w, generator=gen)
    disc_weight = torch.rand(b, 1, h, w, generator=gen)
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        weight_maps=(prior_conf, disc_weight),
    )
    _, _, depths, _ = _per_view_reference(
        inputs, weight_maps=(prior_conf, disc_weight),
    )
    for view, depth_ref in zip(terms, depths):
        torch.testing.assert_close(view.depth, depth_ref, atol=1e-6, rtol=1e-5)


def test_gradients_through_renders_and_rendered_depth_match_the_loop():
    inputs = _demo_inputs(10)
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
    )
    batched_total = sum((0.8 * v.Ll1 + 0.2 * (1.0 - v.ssim) + 0.05 * v.depth) for v in terms) / len(terms)
    g_batched_renders, g_batched_depth = torch.autograd.grad(
        batched_total, (inputs["renders"], inputs["rendered_depth"]), retain_graph=True
    )

    ll1s, ssims, depths, _ = _per_view_reference(inputs)
    loop_total = sum(
        0.8 * ll1 + 0.2 * (1.0 - ssim_v) + 0.05 * depth
        for ll1, ssim_v, depth in zip(ll1s, ssims, depths)
    ) / len(ll1s)
    g_loop_renders, g_loop_depth = torch.autograd.grad(
        loop_total, (inputs["renders"], inputs["rendered_depth"])
    )

    torch.testing.assert_close(g_batched_renders, g_loop_renders, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(g_batched_depth, g_loop_depth, atol=1e-6, rtol=1e-5)


def test_mismatched_renders_gts_shape_raises():
    inputs = _demo_inputs(11)
    try:
        assemble_batched_pixel_terms(renders=inputs["renders"], gts=inputs["gts"][:, :, :, :-1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched renders/gts shapes")


# ---------------------------------------------------------------------------
# Issue #1833: depth_quantity="ed_alpha_weighted" -- weights each valid
# pixel's residual by its own (detached) rendered alpha, normalised by the
# sum of weights, instead of a hard alpha_gate cutoff.
# ---------------------------------------------------------------------------


def test_ed_alpha_weighted_matches_manual_alpha_weighted_mean():
    inputs = _demo_inputs(13)
    b, _, h, w = inputs["rendered_depth"].shape
    gen = torch.Generator().manual_seed(130)
    alpha = torch.rand(b, 1, h, w, generator=gen)

    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        alpha=alpha, depth_quantity="ed_alpha_weighted",
    )

    for i, view in enumerate(terms):
        rend_d = inputs["rendered_depth"][i]
        sensor_d = inputs["sensor_depth"][i]
        alpha_i = alpha[i]
        valid = (sensor_d > 0) & (rend_d.detach() > 0)
        depth_err = masked_depth_error(rend_d, sensor_d, valid, loss_type="l1")
        weights = combine_depth_loss_weights(alpha_i.detach())
        expected = weighted_depth_error_mean(depth_err, weights, valid)
        torch.testing.assert_close(view.depth, expected, atol=1e-6, rtol=1e-5)
        # ed_alpha_weighted does not hard-gate -- valid count is exactly the
        # sensor/rendered validity, unaffected by alpha.
        torch.testing.assert_close(view.depth_valid_count, valid.sum())


def test_ed_alpha_weighted_differs_from_plain_ed_when_alpha_varies():
    """Weighting must actually change the aggregate, not be a silent no-op:
    with a non-uniform alpha map, ed_alpha_weighted's result must differ
    from the unweighted per-view mean over the same valid mask."""
    inputs = _demo_inputs(14)
    b, _, h, w = inputs["rendered_depth"].shape
    gen = torch.Generator().manual_seed(140)
    alpha = torch.rand(b, 1, h, w, generator=gen)

    weighted_terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        alpha=alpha, depth_quantity="ed_alpha_weighted",
    )
    unweighted_terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        depth_quantity="d",
    )
    assert any(
        abs(float(w.depth) - float(u.depth)) > 1e-6
        for w, u in zip(weighted_terms, unweighted_terms)
    )


def test_ed_alpha_weighted_without_alpha_raises():
    inputs = _demo_inputs(15)
    try:
        assemble_batched_pixel_terms(
            renders=inputs["renders"], gts=inputs["gts"],
            rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
            depth_quantity="ed_alpha_weighted",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for ed_alpha_weighted without an alpha stack")


def test_ed_alpha_weighted_gate_and_weight_compose():
    """alpha_gate and ed_alpha_weighted are orthogonal: a positive gate still
    hard-excludes low-alpha pixels from `valid` on top of the per-pixel
    weighting of the pixels that remain."""
    inputs = _demo_inputs(16)
    b, _, h, w = inputs["rendered_depth"].shape
    gen = torch.Generator().manual_seed(160)
    alpha = torch.rand(b, 1, h, w, generator=gen)

    gate = 0.4
    terms = assemble_batched_pixel_terms(
        renders=inputs["renders"], gts=inputs["gts"],
        rendered_depth=inputs["rendered_depth"], sensor_depth=inputs["sensor_depth"],
        alpha=alpha, alpha_gate=gate, depth_quantity="ed_alpha_weighted",
    )
    for i, view in enumerate(terms):
        rend_d = inputs["rendered_depth"][i]
        sensor_d = inputs["sensor_depth"][i]
        alpha_i = alpha[i]
        valid = (sensor_d > 0) & (rend_d.detach() > 0) & (alpha_i.detach() > gate)
        depth_err = masked_depth_error(rend_d, sensor_d, valid, loss_type="l1")
        weights = combine_depth_loss_weights(alpha_i.detach())
        expected = weighted_depth_error_mean(depth_err, weights, valid)
        torch.testing.assert_close(view.depth, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(view.depth_valid_count, valid.sum())
