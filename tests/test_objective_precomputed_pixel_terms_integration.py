"""Issue #1670 Phase B code-review follow-up: CPU integration coverage for
``LossAssemblyInputs.precomputed_pixel_terms`` -- proves ``assemble_loss``'s
own per-view RGB computation and a REAL ``PixelTermsView`` built by
``assemble_batched_pixel_terms`` (not a fake/hand-built one) agree, by
running ``assemble_loss`` end to end twice on the identical render/gt: once
normally (``precomputed_pixel_terms=None``) and once with the batched term
(B=1, so the "shared batch-mean scalar" the batched path returns collapses
to exactly the per-view value, up to floating-point reduction order --
``utils.loss_utils.ssim``'s reference conv-SSIM runs unbatched 3D
(``torch.nn.functional.conv2d``'s "no batch dimension" mode) on the normal
path and batched 4D ``[1,3,H,W]`` on the precomputed path; both reduce over
the exact same elements, hence the 1e-5 tolerance rather than exact
equality).

Fixture pattern (real parsed ``args``, a real ``scene.cameras.Camera`` via
``_declare_prepared_score_provenance``, a hand-built ``SimpleNamespace`` for
``gaussians``) follows ``tests/test_objective_single_numerical_gate.py``'s
``_fixture``/``_inputs`` -- this file only adds the render/gt tensors both
``assemble_loss`` calls and ``assemble_batched_pixel_terms`` need to share.

This complements two existing tests neither of which exercises this
combination: ``tests/test_batched_pixel_terms.py`` pins
``assemble_batched_pixel_terms`` against hand-built tensors and the
per-view *formulas* (``l1_loss``/``ssim`` called directly), never through
``assemble_loss``; ``tests/test_objective_batched_pixel_terms_wiring.py``
exercises the executor's eligibility/wiring logic with FAKE
``PixelTermsView`` fixtures, never a real one built by
``assemble_batched_pixel_terms``. The GPU-only, full-coverage (rgb+depth,
real gsplat render, real executor) twin lives in
``eligsir/optimization/gaussian/tests/test_batched_pixel_terms_gpu.py``.

Depth is NOT covered here. ``assemble_loss``'s per-view depth branch
(``elif use_depth_loss and "rendered_depth" in render_pkg:``) hardcodes
``sensor_d_cpu.to(device="cuda", non_blocking=True)`` unconditionally, on
whatever device the rest of the fixture uses -- so the "normal"
(``precomputed_pixel_terms=None``) call this test needs for the non-batched
side of the comparison cannot reach the depth branch on a CPU-only host; it
would raise, not skip. RGB-only coverage; depth parity is proved instead by
the GPU test's ``depth_valid_sample_count``/``total`` assertions.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import scene.cameras as cameras  # noqa: E402


def _fixture():
    from eligsir.cli.parser import parse_training_args

    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.05
    args.scale_reg = 0.02
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    torch.manual_seed(1234)
    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.rand(3, 4, 4, dtype=torch.float32)
    cam._declare_prepared_score_provenance()

    n = 5
    torch.manual_seed(0)
    gaussians = SimpleNamespace(
        _opacity=torch.randn(n, 1, requires_grad=True),
        get_scaling=torch.rand(n, 3, requires_grad=True) + 0.1,
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    opt = bundle.optimization_params.extract(args)
    pipe = bundle.pipeline_params.extract(args)

    # The ONE render tensor both `assemble_loss` calls below and
    # `assemble_batched_pixel_terms` consume -- built once here (not inside
    # `_inputs`) so both calls unambiguously share the identical object,
    # rather than relying on two separately-reseeded `torch.rand` calls to
    # coincidentally agree.
    torch.manual_seed(7)
    render = torch.rand(3, 4, 4, requires_grad=True)

    return args, opt, pipe, cam, gaussians, render


def _inputs(args, opt, pipe, cam, gaussians, render, *, precomputed_pixel_terms=None, iteration: int = 1):
    from eligsir.optimization.gaussian.contracts import RenderOutputs
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs

    return LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=opt,
        pipe=pipe,
        render_outputs=RenderOutputs(
            payload={
                "render": render,
                "visibility_filter": torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool),
            }
        ),
        image=render,
        iteration=iteration,
        optimizer_update_id=iteration,
        n_frames_ingested=1,
        precomputed_pixel_terms=precomputed_pixel_terms,
    )


def test_precomputed_pixel_terms_matches_normal_path_on_cpu():
    """B=1 real ``PixelTermsView`` (built by ``assemble_batched_pixel_terms``
    from the same render/gt) vs. `assemble_loss`'s own per-view L1/SSIM --
    `total`, `components["Ll1"]`, `components["ssim"]`, and `loss_finite`
    agree within 1e-5."""
    from eligsir.optimization.gaussian.batched_pixel_terms import (
        assemble_batched_pixel_terms,
    )
    from eligsir.optimization.gaussian.objective import assemble_loss

    args, opt, pipe, cam, gaussians, render = _fixture()

    breakdown_normal = assemble_loss(_inputs(args, opt, pipe, cam, gaussians, render))

    (terms,) = assemble_batched_pixel_terms(
        renders=render.unsqueeze(0),
        gts=cam.original_image.unsqueeze(0),
    )
    breakdown_precomputed = assemble_loss(
        _inputs(args, opt, pipe, cam, gaussians, render, precomputed_pixel_terms=terms)
    )

    assert breakdown_normal.loss_finite is True
    assert breakdown_precomputed.loss_finite is True
    assert breakdown_normal.loss_finite == breakdown_precomputed.loss_finite

    torch.testing.assert_close(
        torch.as_tensor(breakdown_normal.total),
        torch.as_tensor(breakdown_precomputed.total),
        atol=1e-5,
        rtol=1e-5,
        msg="total loss diverged between the normal and precomputed-pixel-terms paths",
    )
    for key in ("Ll1", "ssim"):
        assert key in breakdown_normal.components
        assert key in breakdown_precomputed.components
        torch.testing.assert_close(
            breakdown_normal.components[key],
            breakdown_precomputed.components[key],
            atol=1e-5,
            rtol=1e-5,
            msg=f"components[{key!r}] diverged between the normal and precomputed-pixel-terms paths",
        )


def test_precomputed_depth_component_survives_without_tensorboard():
    """Accepted prepared depth must be retained even when no logger is present."""

    from eligsir.optimization.gaussian.batched_pixel_terms import PixelTermsView
    from eligsir.optimization.gaussian.contracts import RenderOutputs
    from eligsir.optimization.gaussian.objective import assemble_loss

    args, opt, pipe, cam, gaussians, render = _fixture()
    terms = PixelTermsView(
        Ll1=torch.tensor(0.0),
        ssim=torch.tensor(0.0),
        depth=torch.tensor(2.5),
        depth_valid_count=torch.tensor(7, dtype=torch.int64),
    )
    base = _inputs(args, opt, pipe, cam, gaussians, render, precomputed_pixel_terms=terms)
    inputs = replace(
        base,
        use_depth_loss=True,
        render_outputs=RenderOutputs(
            payload={
                "render": render,
                "rendered_depth": torch.ones(1, 4, 4),
                "visibility_filter": torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool),
            }
        ),
    )

    breakdown = assemble_loss(inputs)

    assert breakdown.loss_finite is True
    assert torch.equal(breakdown.components["depth"], torch.tensor(2.5))
    assert breakdown.depth_valid_sample_count == 7
