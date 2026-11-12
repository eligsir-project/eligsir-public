"""The supervision-tile probe must actually FIRE from the production objective.

This repo has a documented failure mode where a staged primitive is inert by
construction and nobody notices (W3-45's `phase_c_reseeding`, which no
production call site could ever reach). A probe wrapped in a broad `except`
is a particularly quiet instance of it: a `NameError` from a mis-scoped
variable would print one line, latch a warned flag, and disable the probe for
the rest of that run and every run after, with every unit test still green
because they exercise the reduction helpers directly and never the wiring.

So this file drives the real `assemble_loss` with a recording `tb_writer` and
asserts the expected scalar keys land. It covers variable scope, tensor
shapes, and the `add_scalar` call signature at once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")


class _RecordingWriter:
    def __init__(self):
        self.scalars: dict[str, tuple[float, int]] = {}

    def add_scalar(self, key, value, step):
        self.scalars[key] = (float(value), int(step))


class _Cam:
    def __init__(self, image):
        self.original_image = image
        self.image_name = "probe_cam"
        self.uid = 0
        self.image_width = int(image.shape[-1])
        self.image_height = int(image.shape[-2])


class _Gaussians:
    """Minimal duck-typed model: enough for the unconditional regularisers."""

    def __init__(self, n=8):
        self.get_opacity = torch.full((n, 1), 0.5, requires_grad=True)
        self.get_scaling = torch.full((n, 3), 0.05, requires_grad=True)
        self.get_xyz = torch.zeros(n, 3)

    def __len__(self):
        return int(self.get_opacity.shape[0])


def _args():
    """The merged argparse namespace `assemble_loss` actually sees at runtime.

    Production hands it one `Namespace` carrying every group's options, so a
    per-group stand-in would miss attributes the function reads unconditionally
    (`anisotropy_reg`, ...) and the test would fail for reasons unrelated to
    the probe.
    """
    from types import SimpleNamespace

    from arguments import OptimizationParams, PipelineParams, StreamingParams

    merged: dict = {}
    for group in (PipelineParams, OptimizationParams, StreamingParams):
        merged.update(
            {k.lstrip("_"): v for k, v in vars(group(argparse.ArgumentParser())).items()}
        )
    return SimpleNamespace(**merged)


def _opt():
    from arguments import OptimizationParams

    return OptimizationParams(argparse.ArgumentParser())


def _inputs(*, grid, tb_writer, iteration=100):
    from eligsir.optimization.gaussian.contracts import RenderOutputs
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs

    torch.manual_seed(0)
    gt = torch.rand(3, 32, 32)
    image = gt.clone()
    # Plant an error in exactly one quadrant so a correct probe localises it.
    image[:, 0:16, 16:32] = (gt[:, 0:16, 16:32] + 0.5).clamp(0.0, 1.0)
    image.requires_grad_(True)

    args = _args()
    args.supervision_tile_probe_grid = grid
    opt = _opt()

    return LossAssemblyInputs(
        gaussians=_Gaussians(),
        viewpoint_cam=_Cam(gt),
        args=args,
        opt=opt,
        pipe=None,
        render_outputs=RenderOutputs(payload={}),
        image=image,
        iteration=iteration,
        n_frames_ingested=0,
        tb_writer=tb_writer,
        scalar_log_interval=100,
    )


def _run(**kw):
    from eligsir.optimization.gaussian.objective import assemble_loss

    return assemble_loss(_inputs(**kw))


def test_probe_is_inert_at_the_default():
    """`--supervision_tile_probe_grid` defaults to 0, and 0 means NOTHING."""
    assert _args().supervision_tile_probe_grid == 0
    writer = _RecordingWriter()
    _run(grid=0, tb_writer=writer)
    assert not [k for k in writer.scalars if k.startswith("supervision_tile/")]


def test_probe_fires_and_localises_the_planted_error():
    """The wiring is REACHABLE -- not merely importable."""
    writer = _RecordingWriter()
    _run(grid=2, tb_writer=writer)
    keys = {k for k in writer.scalars if k.startswith("supervision_tile/")}
    assert keys, "probe produced no scalars -- the branch did not execute"
    for cell in ("r0c0", "r0c1", "r1c0", "r1c1"):
        assert f"supervision_tile/rgb_l1/2x2/{cell}" in keys
    v = writer.scalars
    # Error was planted in the top-right supervision tile only.
    assert v["supervision_tile/rgb_l1/2x2/r0c1"][0] > 0.3
    for cell in ("r0c0", "r1c0", "r1c1"):
        assert v[f"supervision_tile/rgb_l1/2x2/{cell}"][0] == pytest.approx(0.0, abs=1e-6)
    assert v["supervision_tile/rgb_l1_max_over_mean"][0] == pytest.approx(4.0, rel=1e-3)


def test_probe_scalars_are_stamped_with_the_iteration():
    writer = _RecordingWriter()
    _run(grid=2, tb_writer=writer, iteration=300)
    assert writer.scalars["supervision_tile/rgb_l1_max"][1] == 300


def test_probe_respects_the_scalar_log_interval():
    """It must never add a host sync on an iteration that would not sync anyway."""
    writer = _RecordingWriter()
    _run(grid=2, tb_writer=writer, iteration=101)   # 101 % 100 != 0
    assert not [k for k in writer.scalars if k.startswith("supervision_tile/")]


def test_probe_does_not_change_the_loss():
    """Diagnostic only: the returned loss must be bit-identical with it on."""
    off = _run(grid=0, tb_writer=_RecordingWriter())
    on = _run(grid=4, tb_writer=_RecordingWriter())
    assert float(on.total_tensor) == float(off.total_tensor)


def test_probe_survives_a_bad_grid_without_failing_the_step():
    """A misconfigured probe must degrade to silence, never kill a run."""
    writer = _RecordingWriter()
    result = _run(grid=999, tb_writer=writer)      # above MAX_LATTICE_DIM
    assert float(result.total_tensor) > 0.0
    assert not [k for k in writer.scalars if k.startswith("supervision_tile/")]
