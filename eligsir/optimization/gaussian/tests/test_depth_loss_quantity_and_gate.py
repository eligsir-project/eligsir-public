"""Confirms --streaming_depth_loss_quantity and --streaming_depth_loss_alpha_gate:
(1) at their defaults (d / 0.0) the loss is byte-identical to pre-change
behavior; (2) alpha_gate > 0.0 excludes low-alpha pixels regardless of which
quantity is selected (arm B: d + gate); (3) quantity="ed" reads
render_pkg["rendered_depth_ed"] (arm C: ed + gate); (4) a new
"streaming/depth_loss_raw_magnitude" TB scalar is logged in all three arms.

Fixture: reuses this codebase's established real-GPU fixture recipe (real
``GsplatGaussianModel``/``scene.cameras.Camera``/CLI-parser-derived
``args``/``opt``/``pipe`` via ``_build_gaussians``/``_build_camera``/
``_build_args``/``_bg``) from
``eligsir/optimization/gaussian/tests/test_gpu_parity_harness.py``,
plus a real ``render_port.invoke_render(..., render_depth=True)`` call to
get a real, well-formed ``render_pkg`` -- then overrides just the
depth-loss-relevant keys (``"rendered_depth"``, ``"rendered_depth_ed"``,
``"alpha"``) with small synthetic tensors so the expected loss value can be
hand-computed exactly, and monkeypatches the camera's sensor-depth cache
(``cam._streaming_depth_path`` / ``cam._sensor_depth_cache``) to bypass disk
I/O -- mirroring the pattern in
``eligsir/mapping/renderer/tests/test_rendered_depth_ed_reconstruction.py``
for handling the real ``[H, W, 1]`` alpha shape.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import gsplat  # noqa: F401
    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "test_depth_loss_quantity_and_gate requires CUDA/ROCm + gsplat (no "
        "CPU rendering path exists in this codebase); run inside the "
        "training container with GPU passthrough per CLAUDE.md's Running "
        "Commands section: docker compose run --rm train python -m pytest "
        "eligsir/optimization/gaussian/tests/ -q"
    ),
)

from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (  # noqa: E402
    _bg,
    _build_args,
    _build_camera,
    _build_gaussians,
)

# Sensor depth / D / ED values, all at the same three pixels:
#   pixel (0,0): sensor=1.0, D=1.5 (diff 0.5), ED=9.0 (diff 8.0), alpha=0.05 (< gate)
#   pixel (0,1): sensor=2.0, D=2.6 (diff 0.6), ED=9.0 (diff 7.0), alpha=0.50 (> gate)
#   pixel (0,2): sensor=3.0, D=3.7 (diff 0.7), ED=9.0 (diff 6.0), alpha=0.90 (> gate)
_SENSOR = [(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)]
_D = [(0, 0, 1.5), (0, 1, 2.6), (0, 2, 3.7)]
_ED = [(0, 0, 9.0), (0, 1, 9.0), (0, 2, 9.0)]
_ALPHA = [(0, 0, 0.05), (0, 1, 0.50), (0, 2, 0.90)]
_GATE = 0.1

# Hand-computed expectations.
_EXPECTED_ARM_A = (0.5 + 0.6 + 0.7) / 3.0  # no gating: all three D-pixels
_EXPECTED_ARM_B = (0.6 + 0.7) / 2.0        # d-quantity, gated: excludes pixel (0,0)
_EXPECTED_ARM_C = (7.0 + 6.0) / 2.0        # ed-quantity, gated: excludes pixel (0,0)
# Issue #1833 arm D: ed_alpha_weighted, alpha_gate=0.0 (no hard exclusion --
# all three pixels contribute, weighted by their own alpha instead).
# residuals |ED-sensor| = 8.0, 7.0, 6.0; weights = alpha = 0.05, 0.50, 0.90.
_EXPECTED_ARM_D = (8.0 * 0.05 + 7.0 * 0.50 + 6.0 * 0.90) / (0.05 + 0.50 + 0.90)


class _FakeTBWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, iteration):
        self.scalars.setdefault(tag, []).append((iteration, value))


def _fill(tensor, entries):
    for i, j, v in entries:
        tensor[0, i, j] = v
    return tensor


def _build_fixture(quantity, alpha_gate):
    """Builds a real render_pkg (via a real render call) with the depth-loss
    relevant keys overridden by controlled synthetic values, and a camera
    wired to return a controlled sensor-depth tensor without touching disk.
    Returns (LossAssemblyInputs-ready kwargs dict, tb_writer)."""
    from eligsir.optimization.gaussian.contracts import RenderOutputs
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs
    from eligsir.optimization.gaussian.render_port import invoke_render

    torch.manual_seed(0)
    gaussians = _build_gaussians(seed=0)
    cam = _build_camera(seed=1)
    args, opt, pipe, _ns = _build_args()
    args.streaming_depth_loss_quantity = quantity
    args.streaming_depth_loss_alpha_gate = alpha_gate

    render_outputs = invoke_render(
        gaussians, cam, pipe, _bg(), update_sh_rest=True, render_depth=True,
    )
    payload = render_outputs.payload
    h, w = payload["render"].shape[-2], payload["render"].shape[-1]

    sensor_d = torch.zeros((1, h, w), dtype=torch.float32)
    _fill(sensor_d, _SENSOR)
    # Bypass the disk-loading path in _get_sensor_depth: any hasattr sentinel
    # plus a pre-populated cache tensor short-circuits straight to it.
    cam._streaming_depth_path = "unused-bypassed-by-cache"
    cam._sensor_depth_cache = sensor_d

    rend_d = torch.zeros((1, h, w), device=payload["render"].device)
    _fill(rend_d, _D)
    rend_ed = torch.zeros((1, h, w), device=payload["render"].device)
    _fill(rend_ed, _ED)
    # render_pkg["alpha"] is [H, W, 1] per gsplat_backend.py's real return
    # shape (confirmed empirically, see that module's NOTE comment).
    alpha = torch.zeros((h, w, 1), device=payload["render"].device)
    for i, j, v in _ALPHA:
        alpha[i, j, 0] = v

    payload["rendered_depth"] = rend_d
    payload["rendered_depth_ed"] = rend_ed
    payload["alpha"] = alpha

    tb_writer = _FakeTBWriter()

    inputs = LossAssemblyInputs(
        gaussians=gaussians, viewpoint_cam=cam, args=args, opt=opt, pipe=pipe,
        render_outputs=RenderOutputs(payload=payload), image=payload["render"],
        iteration=1, n_frames_ingested=0, tb_writer=tb_writer,
        sparse_active_set=False, use_depth_loss=True, use_energy_mcmc=False,
        depth_loss_type="l1", depth_loss_weight=1.0, scalar_log_interval=1,
    )
    return inputs, tb_writer


@pytest.mark.cuda
def test_defaults_use_rendered_depth_key_with_no_alpha_gate():
    from eligsir.optimization.gaussian.objective import assemble_loss

    inputs, _tb = _build_fixture(quantity="d", alpha_gate=0.0)
    breakdown = assemble_loss(inputs)
    got = float(breakdown.materialized_components()["depth"])
    assert math.isclose(got, _EXPECTED_ARM_A, rel_tol=1e-5), (got, _EXPECTED_ARM_A)


@pytest.mark.cuda
def test_arm_b_d_quantity_with_alpha_gate_excludes_low_alpha_pixels():
    from eligsir.optimization.gaussian.objective import assemble_loss

    inputs, _tb = _build_fixture(quantity="d", alpha_gate=_GATE)
    breakdown = assemble_loss(inputs)
    got = float(breakdown.materialized_components()["depth"])
    assert math.isclose(got, _EXPECTED_ARM_B, rel_tol=1e-5), (got, _EXPECTED_ARM_B)
    # Sanity: distinct from the ungated value, proving the gate did something.
    assert not math.isclose(got, _EXPECTED_ARM_A, rel_tol=1e-5)


@pytest.mark.cuda
def test_arm_c_ed_quantity_with_same_alpha_gate_reads_ed_key():
    from eligsir.optimization.gaussian.objective import assemble_loss

    inputs, _tb = _build_fixture(quantity="ed", alpha_gate=_GATE)
    breakdown = assemble_loss(inputs)
    got = float(breakdown.materialized_components()["depth"])
    assert math.isclose(got, _EXPECTED_ARM_C, rel_tol=1e-5), (got, _EXPECTED_ARM_C)
    # Sanity: distinct from arm B, proving ED (not D) was actually read.
    assert not math.isclose(got, _EXPECTED_ARM_B, rel_tol=1e-5)


@pytest.mark.cuda
def test_raw_depth_loss_magnitude_scalar_logged_in_all_three_arms():
    from eligsir.optimization.gaussian.objective import assemble_loss

    for quantity, gate, expected in (
        ("d", 0.0, _EXPECTED_ARM_A),
        ("d", _GATE, _EXPECTED_ARM_B),
        ("ed", _GATE, _EXPECTED_ARM_C),
        ("ed_alpha_weighted", 0.0, _EXPECTED_ARM_D),
    ):
        inputs, tb_writer = _build_fixture(quantity=quantity, alpha_gate=gate)
        assemble_loss(inputs)
        logged = tb_writer.scalars.get("streaming/depth_loss_raw_magnitude")
        assert logged, f"no scalar logged for quantity={quantity!r} gate={gate!r}"
        iteration, value = logged[-1]
        assert iteration == 1
        assert math.isfinite(value)
        assert math.isclose(value, expected, rel_tol=1e-5), (quantity, gate, value, expected)


@pytest.mark.cuda
def test_arm_d_ed_alpha_weighted_weights_each_pixel_by_its_own_alpha():
    """Issue #1833: unlike arm C's hard gate (which fully excludes pixel
    (0,0)), 'ed_alpha_weighted' with alpha_gate=0.0 keeps all three ED
    pixels valid but weights each one's residual by its own rendered alpha,
    reading render_pkg["rendered_depth_ed"] exactly like arm C."""
    from eligsir.optimization.gaussian.objective import assemble_loss

    inputs, _tb = _build_fixture(quantity="ed_alpha_weighted", alpha_gate=0.0)
    breakdown = assemble_loss(inputs)
    got = float(breakdown.materialized_components()["depth"])
    assert math.isclose(got, _EXPECTED_ARM_D, rel_tol=1e-5), (got, _EXPECTED_ARM_D)
    # Sanity: distinct from arm A's unweighted D-quantity mean and from arm
    # C's hard-gated ED mean, proving the per-pixel alpha weighting (not a
    # gate, not the D quantity) is what actually ran.
    assert not math.isclose(got, _EXPECTED_ARM_A, rel_tol=1e-5)
    assert not math.isclose(got, _EXPECTED_ARM_C, rel_tol=1e-5)


@pytest.mark.cuda
def test_ed_alpha_weighted_missing_alpha_key_raises():
    """Same failure-mode discipline as the existing ed-without-rendered_depth_ed
    guard: refuse to silently apply no weighting rather than corrupt a run."""
    from eligsir.optimization.gaussian.objective import assemble_loss

    inputs, _tb = _build_fixture(quantity="ed_alpha_weighted", alpha_gate=0.0)
    del inputs.render_outputs.payload["alpha"]
    with pytest.raises(RuntimeError, match="ed_alpha_weighted"):
        assemble_loss(inputs)
