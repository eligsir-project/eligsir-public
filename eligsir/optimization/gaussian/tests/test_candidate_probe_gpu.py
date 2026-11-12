"""Real-renderer contract tests for the candidate probe (issue #803).

The CPU companion (``test_candidate_probe.py``) proves the probe's *contract*
against a fake renderer. It cannot prove the two things that only a real
rasterizer can answer:

1. **Render parity.** A probe render of camera C against map snapshot M must
   produce the same image the canonical supervision render produces for the
   same C and M. If it did not, every probe measurement would describe a
   slightly different map than the one training actually optimizes -- an error
   no amount of contract testing would surface.
2. **No drift through a real backward-capable kernel.** gsplat's rasterizer
   allocates, caches and (in gradient mode) builds an autograd graph over the
   live parameters. The "no ``.grad`` accumulation / no parameter drift"
   invariant has to be demonstrated against that kernel, not against a fake.

Run inside the training container with GPU passthrough, per CLAUDE.md's
Running Commands section::

    docker compose run --rm train python -m pytest \\
        eligsir/optimization/gaussian/tests/test_candidate_probe_gpu.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import gsplat  # noqa: F401

    GSPLAT_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "candidate-probe GPU tests require CUDA/ROCm + gsplat (this codebase "
        "has no CPU rasterizer); run inside the training container with GPU "
        "passthrough per CLAUDE.md's Running Commands section."
    ),
)

from eligsir.contracts.control import ResolutionTier  # noqa: E402
from eligsir.optimization.gaussian.candidate_probe import (  # noqa: E402
    CandidateProbeRequest,
    CandidateProbeView,
    ProbeEvidenceChannel,
)
from eligsir.optimization.gaussian.contracts import (  # noqa: E402
    OptimizationRuntimeContext,
)
from eligsir.optimization.gaussian.engine import (  # noqa: E402
    GaussianOptimizationEngine,
)


# ---------------------------------------------------------------------------
# Fixtures (same shape as test_gpu_parity_harness.py's, deliberately)
# ---------------------------------------------------------------------------


def _build_args():
    from eligsir.cli.parser import build_training_parser

    bundle = build_training_parser()
    ns = bundle.parser.parse_args([])
    ns.optimizer_type = "adam"
    ns.cap_max = 1000
    args = bundle.model_params.extract(ns)
    for key, value in vars(ns).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args, bundle.optimization_params.extract(ns), bundle.pipeline_params.extract(ns)


def _build_gaussians(seed: int = 0):
    import numpy as np

    from scene.gsplat_model import GsplatGaussianModel

    rng = np.random.default_rng(seed)
    n = 24
    pcd = SimpleNamespace(
        points=rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float64),
        colors=rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float64),
    )
    model = GsplatGaussianModel(sh_degree=0)
    model.create_from_pcd(
        pcd, spatial_lr_scale=1.0, init_scale_mode="fixed", init_scale=0.05
    )
    return model


def _build_camera(image_size: int = 32, seed: int = 1, uid: int = 0):
    import numpy as np

    from scene.cameras import Camera

    rng = np.random.default_rng(seed)
    image = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=(3, image_size, image_size)).astype("float32")
    )
    # Explicit intrinsics are REQUIRED for any reduced-resolution probe:
    # materialize_resolution_tier refuses a camera without fx/fy/cx/cy (it
    # cannot rescale intrinsics it does not have). Real streaming cameras carry
    # them; a fixture that omits them silently turns every reduced-tier probe
    # into a skipped candidate, which is how the first version of this file
    # failed.
    focal = 0.5 * image_size / np.tan(0.45)
    return Camera(
        colmap_id=uid,
        R=np.eye(3),
        T=np.array([0.0, 0.0, 3.0 + 0.1 * uid]),
        FoVx=0.9,
        FoVy=0.9,
        image=image,
        gt_alpha_mask=None,
        image_name=f"toy_{uid:04d}",
        uid=uid,
        fx=float(focal),
        fy=float(focal),
        cx=image_size / 2.0,
        cy=image_size / 2.0,
    )


class _Session:
    def __init__(self, model):
        self.model = model
        self.parameter_version = 0
        self.lr_calls = 0
        self.optimizers = getattr(model, "optimizers", None)

    def update_learning_rate(self, iteration):  # pragma: no cover - must not run
        self.lr_calls += 1
        return 1e-4


class _Backend:
    def __init__(self, model):
        self.map_content_version = 3
        self.topology_version = 1
        self.coordinate_frame_version = 0
        self._session = _Session(model)

    def begin_optimization(self):
        return self._session


@pytest.fixture()
def rig():
    args, opt, pipe = _build_args()
    model = _build_gaussians()
    backend = _Backend(model)
    engine = GaussianOptimizationEngine(backend)
    runtime = OptimizationRuntimeContext(
        session=backend.begin_optimization(),
        gaussians=model,
        args=args,
        opt=opt,
        pipe=pipe,
    )
    return engine, backend, model, runtime


def _background():
    return torch.zeros(3, device="cuda")


def _views(n=2, render_depth=False):
    return tuple(
        CandidateProbeView(
            camera=_build_camera(seed=10 + i, uid=i),
            background=_background(),
            source_id=f"gpu-{i:03d}",
            source_ordinal=i,
            timestamp_ns=1000 + i,
            render_depth=render_depth,
        )
        for i in range(n)
    )


# ---------------------------------------------------------------------------
# 1. Render parity with the canonical render path
# ---------------------------------------------------------------------------


def test_probe_render_matches_the_canonical_render_for_the_same_snapshot(rig):
    engine, _, model, runtime = rig
    from eligsir.optimization.gaussian.render_port import invoke_render

    view = _views(1)[0]
    with torch.no_grad():
        reference = invoke_render(
            model,
            view.camera,
            runtime.pipe,
            view.background,
            update_sh_rest=view.update_sh_rest,
            render_depth=view.render_depth,
        ).payload["render"]
        gt = view.camera.original_image[:3].to(reference.device, reference.dtype)
        reference_l1 = float((reference - gt).abs().mean())

    batch = engine.probe_candidates(
        CandidateProbeRequest(views=(view,), probe_batch_id="parity"), runtime
    )
    probed_l1 = batch.per_view[0].evidence.probe.rgb_l1_mean.value
    # Same camera, same fixed map, same rasterizer: the only tolerance is the
    # rasterizer's own atomic-add summation jitter, documented at 1e-4 for this
    # fixture size in test_step_parity.py.
    assert probed_l1 == pytest.approx(reference_l1, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. No parameter / optimizer / grad drift through the real kernel
# ---------------------------------------------------------------------------


def _snapshot(model):
    return {
        "params": {n: p.detach().clone() for n, p in model.params.items()},
        "grads": {
            n: (None if p.grad is None else p.grad.detach().clone())
            for n, p in model.params.items()
        },
        "adam": {
            name: {
                key: (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in optimizer.state.get(
                    optimizer.param_groups[0]["params"][0], {}
                ).items()
            }
            for name, optimizer in getattr(model, "optimizers", {}).items()
        },
    }


def _assert_identical(before, after):
    for name, tensor in before["params"].items():
        assert torch.equal(tensor, after["params"][name]), f"param {name} drifted"
    for name, grad in before["grads"].items():
        other = after["grads"][name]
        assert (grad is None) == (other is None), f"grad presence changed for {name}"
        if grad is not None:
            assert torch.equal(grad, other), f"grad {name} drifted"
    assert set(before["adam"]) == set(after["adam"])
    for name, state in before["adam"].items():
        for key, value in state.items():
            other = after["adam"][name][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), f"adam {name}.{key} drifted"
            else:
                assert value == other, f"adam {name}.{key} drifted"


def test_forward_probe_leaves_parameters_grads_and_adam_untouched(rig):
    engine, backend, model, runtime = rig
    session = backend.begin_optimization()
    before = _snapshot(model)
    engine.probe_candidates(
        CandidateProbeRequest(views=_views(3), probe_batch_id="nodrift"), runtime
    )
    _assert_identical(before, _snapshot(model))
    assert session.lr_calls == 0, "the probe advanced the LR scheduler"
    assert all(p.grad is None for p in model.params.values())


def test_probe_after_a_real_backward_preserves_pre_existing_gradients(rig):
    """Probing BETWEEN steps must not disturb gradients already in flight."""

    engine, _, model, runtime = rig
    from eligsir.optimization.gaussian.render_port import invoke_render

    view = _views(1)[0]
    outputs = invoke_render(
        model, view.camera, runtime.pipe, view.background,
        update_sh_rest=False, render_depth=False,
    )
    gt = view.camera.original_image[:3].to(outputs.payload["render"].device)
    (outputs.payload["render"] - gt).abs().mean().backward()
    assert any(p.grad is not None for p in model.params.values())

    before = _snapshot(model)
    engine.probe_candidates(
        CandidateProbeRequest(views=_views(2), probe_batch_id="between"), runtime
    )
    _assert_identical(before, _snapshot(model))


def test_gradient_probe_leaves_grad_untouched_through_the_real_kernel(rig):
    engine, _, model, runtime = rig
    batch = engine.probe_candidates(
        CandidateProbeRequest(
            views=_views(2),
            probe_batch_id="grad",
            evidence_channels=frozenset(
                {ProbeEvidenceChannel.RGB, ProbeEvidenceChannel.GRADIENT}
            ),
        ),
        runtime,
    )
    assert all(p.grad is None for p in model.params.values()), (
        "gradient-probe mode accumulated into .grad through the real gsplat "
        "backward; it must use torch.autograd.grad exclusively"
    )
    assert batch.accounting.candidate_probe_gradient_view_count == 2
    for result in batch.per_view:
        assert result.gradient is not None
        assert result.gradient.total_norm > 0.0
        assert result.gradient.energy_by_family, "no parameter family got gradient"
        assert 0.0 <= result.gradient.touched_gaussian_fraction <= 1.0


def test_gradient_probe_matches_an_isolated_reference_gradient(rig):
    """Reference: autograd.grad of THIS view's objective, computed directly."""

    engine, _, model, runtime = rig
    from eligsir.optimization.gaussian.render_port import invoke_render
    from eligsir.utils.loss_utils import l1_loss, ssim

    view = _views(1)[0]
    lambda_dssim = float(getattr(runtime.opt, "lambda_dssim", 0.2))
    outputs = invoke_render(
        model, view.camera, runtime.pipe, view.background,
        update_sh_rest=False, render_depth=False,
    )
    image = outputs.payload["render"]
    gt = view.camera.original_image[:3].to(image.device, image.dtype)
    objective = (1.0 - lambda_dssim) * l1_loss(image, gt) + lambda_dssim * (
        1.0 - ssim(image, gt)
    )
    reference = torch.autograd.grad(objective, [model.params["means"]])[0]
    reference_energy = float(reference.square().sum())
    assert all(p.grad is None for p in model.params.values())

    batch = engine.probe_candidates(
        CandidateProbeRequest(
            views=(view,),
            probe_batch_id="grad-ref",
            evidence_channels=frozenset(
                {ProbeEvidenceChannel.RGB, ProbeEvidenceChannel.GRADIENT}
            ),
        ),
        runtime,
    )
    measured = batch.per_view[0].gradient.energy_by_family["means"]
    assert measured == pytest.approx(reference_energy, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. Reduced-resolution correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,divisor",
    [(ResolutionTier.NATIVE, 1), (ResolutionTier.HALF, 2), (ResolutionTier.QUARTER, 4)],
)
def test_reduced_resolution_probe_renders_at_that_resolution(rig, tier, divisor):
    engine, _, _, runtime = rig
    views = _views(1)
    batch = engine.probe_candidates(
        CandidateProbeRequest(
            views=views, probe_batch_id=f"tier-{tier.value}", resolution_tier=tier
        ),
        runtime,
    )
    result = batch.per_view[0]
    assert not result.skipped, result.skip_reason
    assert (result.probe_height, result.probe_width) == (32 // divisor, 32 // divisor)
    assert result.probe_pixel_count == (32 // divisor) ** 2
    assert batch.resolution_bucket == tier.value
    assert result.evidence.probe.rgb_l1_mean.value is not None
    # The camera must come back exactly as it was found.
    assert views[0].camera.image_width == 32
    assert not hasattr(views[0].camera, "_canonical_resolution_tier")


def test_reduced_resolution_evidence_is_finite_and_comparable(rig):
    """A reduced-resolution probe must remain a usable measurement.

    Not a claim that the numbers are EQUAL across tiers -- they are not, and
    whether they rank candidates the same way is exactly what the resolution
    benchmark (scripts/issue803_probe_resolution_benchmark.py) measures. This
    only guards against a reduced tier producing degenerate output.
    """

    engine, _, _, runtime = rig
    values = {}
    for tier in (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER):
        batch = engine.probe_candidates(
            CandidateProbeRequest(
                views=_views(2), probe_batch_id="cmp", resolution_tier=tier
            ),
            runtime,
        )
        assert not any(r.skipped for r in batch.per_view), [
            r.skip_reason for r in batch.per_view if r.skipped
        ]
        values[tier] = [
            r.evidence.probe.rgb_l1_mean.value for r in batch.per_view
        ]
        assert all(v is not None and math.isfinite(v) for v in values[tier])
    assert len(values) == 3


def test_probe_accounting_reflects_reduced_pixel_work(rig):
    engine, _, _, runtime = rig
    native = engine.probe_candidates(
        CandidateProbeRequest(views=_views(2), probe_batch_id="n"), runtime
    )
    quarter = engine.probe_candidates(
        CandidateProbeRequest(
            views=_views(2),
            probe_batch_id="q",
            resolution_tier=ResolutionTier.QUARTER,
        ),
        runtime,
    )
    assert (
        quarter.accounting.candidate_probe_pixel_count * 16
        == native.accounting.candidate_probe_pixel_count
    )
    assert (
        quarter.accounting.candidate_probe_rasterizer_call_count
        == native.accounting.candidate_probe_rasterizer_call_count
    )


# ---------------------------------------------------------------------------
# 4. Depth evidence against a real rendered depth channel
# ---------------------------------------------------------------------------


def test_pose_pairwise_evidence_on_real_rotated_cameras(rig):
    """Two real cameras 90 degrees apart must report a 90 degree separation.

    The identity-pose fixtures elsewhere cannot test the forward-axis
    convention -- every column choice yields 0 degrees on identity.
    """

    import numpy as np

    from scene.cameras import Camera

    engine, _, _, runtime = rig
    cameras = []
    for index, angle in enumerate((0.0, np.pi / 2.0)):
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        image = torch.rand(3, 32, 32)
        cameras.append(
            Camera(
                colmap_id=index,
                R=rotation,
                T=np.array([0.0, 0.0, 3.0]),
                FoVx=0.9,
                FoVy=0.9,
                image=image,
                gt_alpha_mask=None,
                image_name=f"rot_{index}",
                uid=index,
            )
        )
    views = tuple(
        CandidateProbeView(
            camera=camera,
            background=_background(),
            source_id=f"rot-{i}",
            source_ordinal=i,
        )
        for i, camera in enumerate(cameras)
    )
    batch = engine.probe_candidates(
        CandidateProbeRequest(views=views, probe_batch_id="rot"), runtime
    )
    assert batch.pairwise[0].pose_angle_deg == pytest.approx(90.0, abs=1.0)
    assert batch.pairwise[0].pose_baseline_m is not None


def test_depth_channel_reports_reason_coded_absence_without_sensor_depth(rig):
    engine, _, _, runtime = rig
    batch = engine.probe_candidates(
        CandidateProbeRequest(
            views=_views(1, render_depth=True), probe_batch_id="depth"
        ),
        runtime,
    )
    probe = batch.per_view[0].evidence.probe
    # The toy camera carries no sensor depth, so this must be an explicit,
    # reason-coded absence -- never a silent zero that reads as "perfect".
    assert probe.depth_l1_mean_m.value is None
    assert probe.depth_l1_mean_m.reason
