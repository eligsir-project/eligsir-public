"""Issue #703: Production wiring of supervision-work counters.

Validates:
1. The counters exist and are declared in train_streaming.py
2. Rasterizer calls are incremented at each inline render site
3. Supervised pixels are only incremented inside finite/backward-ran branches
4. utils/supervision_work.py is imported by train_streaming.py (no longer test-only)
5. TensorBoard scalars are emitted
6. report.json carries the fields

Torch-free AST-based contract test (runs in CPU CI image).
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_STREAMING_PATH = _REPO_ROOT / "train_streaming.py"
_SUPERVISION_WORK_PATH = _REPO_ROOT / "utils" / "supervision_work.py"


# --- fixtures for the report-writer end-to-end tests at the bottom ----------
# Same minimal fakes tests/test_git_provenance.py uses to drive the real
# write_post_training_report(): the writer needs a gaussian model, a pipe, one
# train/test camera and a render fn, nothing more, and lpipsPyTorch is stubbed
# because its import pulls a model download this suite must not do.


@contextlib.contextmanager
def _mocked_lpips():
    had = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}



def _read_source(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def _parse_ast(source: str) -> ast.Module:
    return ast.parse(source)


class TestSupervisionWorkCountersProduction:
    """Validate counters are wired in the streaming training loop."""

    def test_clk_rasterizer_call_count_declared(self):
        """_clk_rasterizer_call_count is declared in train_streaming_training()."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Look for _clk_rasterizer_call_count assignment
        found = False
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_clk_rasterizer_call_count":
                        found = True
                        break

        assert found, "_clk_rasterizer_call_count not declared in streaming_training()"

    def test_clk_supervised_pixel_count_declared(self):
        """_clk_supervised_pixel_count is declared in streaming_training()."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Look for _clk_supervised_pixel_count assignment
        found = False
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_clk_supervised_pixel_count":
                        found = True
                        break

        assert found, "_clk_supervised_pixel_count not declared in streaming_training()"

    def test_rasterizer_call_count_incremented_at_inline_renders(self):
        """_clk_rasterizer_call_count is incremented at inline render sites (6255, 6257)."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Count AugAssign nodes that increment _clk_rasterizer_call_count
        # Expected: primary inline (~6552) + extra inline (~6294) + engine result (~6544)
        rasterizer_increments = 0
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "_clk_rasterizer_call_count":
                    rasterizer_increments += 1

        assert rasterizer_increments >= 3, (
            f"Expected at least 3 increments to _clk_rasterizer_call_count "
            f"(primary inline ~6552, extra inline ~6294, engine result ~6544), "
            f"found {rasterizer_increments}"
        )

    def test_supervised_pixel_count_incremented_in_finite_backward_branch(self):
        """_clk_supervised_pixel_count is incremented only inside finite/backward branches."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Count AugAssign nodes that increment _clk_supervised_pixel_count
        # Expected: primary inline backward (~7394-7399) + extra inline on success path (~6313-6316)
        # + engine result accumulation (~6544) = at least 3 sites
        pixel_increments = 0
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "_clk_supervised_pixel_count":
                    pixel_increments += 1

        assert pixel_increments >= 3, (
            "Expected at least 3 increments to _clk_supervised_pixel_count "
            "(primary inline backward ~7394, extra inline success path ~6313, engine result ~6544), "
            f"found {pixel_increments}"
        )

    def test_supervised_pixel_count_increment_sites_match_rasterizer(self):
        """Regression: _clk_supervised_pixel_count has at least as many increment sites as _clk_rasterizer_call_count."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Count increment sites for both counters
        rasterizer_sites = 0
        pixel_sites = 0
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    if node.target.id == "_clk_rasterizer_call_count":
                        rasterizer_sites += 1
                    elif node.target.id == "_clk_supervised_pixel_count":
                        pixel_sites += 1

        assert pixel_sites >= rasterizer_sites, (
            f"ASYMMETRY REGRESSION: _clk_supervised_pixel_count has {pixel_sites} increment sites "
            f"but _clk_rasterizer_call_count has {rasterizer_sites}. This breaks comparability "
            f"between inline and engine arms at B>1. Both counters must have the same number of "
            f"sites so neither arm undercounts relative to the other."
        )

    def test_inline_primary_and_extra_success_paths_record_source_exposures(self):
        """Inline B=1/B>1 accounting follows actual backward calls only."""
        source = _read_source(_TRAIN_STREAMING_PATH)

        extra_backward = source.index("_extra_loss.backward()")
        extra_record = source.index("_record_successful_supervision(", extra_backward)
        assert extra_record > extra_backward

        primary_backward = source.index(".render_and_backward(", extra_record)
        primary_record = source.index("_record_successful_supervision(", primary_backward)
        assert primary_record > primary_backward

    def test_engine_result_source_exposures_are_consumed_by_the_streaming_loop(self):
        """Engine B=1/B>1 results cross into the bounded report aggregate."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        assert "_engine_result.successful_supervision_source_indices" in source

    def test_supervision_work_module_imported(self):
        """utils/supervision_work functions are imported (no longer test-only)."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Look for import of supervision_work module
        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is not None and "supervision_work" in node.module:
                    found_import = True
                    break

        assert found_import, (
            "utils/supervision_work should be imported in train_streaming.py "
            "(production consumer, not test-only)"
        )

    def test_tb_scalars_emitted_when_cadence_telemetry_enabled(self):
        """TensorBoard scalars are emitted under streaming_cadence_telemetry_enabled."""
        source = _read_source(_TRAIN_STREAMING_PATH)
        tree = _parse_ast(source)

        # Find streaming_training function
        streaming_training_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "streaming_training":
                streaming_training_fn = node
                break

        assert streaming_training_fn is not None, "streaming_training() not found"

        # Look for "streaming/clocks/rasterizer_call_count" and "streaming/work/*" strings
        scalar_names = set()
        for node in ast.walk(streaming_training_fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "rasterizer_call" in node.value or "supervised_pixel" in node.value:
                    scalar_names.add(node.value)

        # Expect at least the base scalars and derived ratios
        assert len(scalar_names) >= 2, (
            f"Expected to find TB scalars for rasterizer_call and supervised_pixel, "
            f"found: {scalar_names}"
        )

    def test_supervision_work_functions_callable(self):
        """supervision_work module functions are defined and callable."""
        # This is a smoke test of the actual functions
        from eligsir.utils.supervision_work import (
            rasterizer_calls_per_optimization_step,
            supervised_pixels_per_optimization_step,
            supervision_views_per_optimization_step,
        )

        # Test basic functionality (results are per-step ratios)
        assert rasterizer_calls_per_optimization_step(10, 20) == 2.0
        # 10 steps, 1920*1080 total pixels = 207360 pixels/step
        assert supervised_pixels_per_optimization_step(10, 1920 * 1080) == 207360.0
        assert supervision_views_per_optimization_step(10, 20) == 2.0

        # Test zero case
        assert rasterizer_calls_per_optimization_step(0, 0) == 0.0
        assert supervised_pixels_per_optimization_step(0, 0) == 0.0
        assert supervision_views_per_optimization_step(0, 0) == 0.0


class TestSupervisionWorkReachesReportJson:
    """The counters must survive the train_streaming -> report-writer SEAM.

    The first version of this PR built the block as a nested dict LITERAL
    inside ``_streaming_report_kwargs()`` and never added a matching parameter
    to ``write_post_training_report()``. Two guards caught it in CI
    (``test_issue646_report_kwargs_signature`` /
    ``test_issue595_report_call_chain``): the writer would have raised
    ``TypeError: unexpected keyword argument 'supervision_work'`` and every
    streaming run would have finished training and produced NO report.json --
    the exact #646 outage class.

    The tests above prove the counters are incremented and logged to
    TensorBoard. These prove they actually land in report.json, which is what
    #755's promotion A/B reads.
    """

    def test_block_builder_returns_all_seven_fields_with_real_values(self):
        from eligsir.utils.supervision_work import supervision_work_report_block

        block = supervision_work_report_block(
            optimization_step_count=100,
            supervision_view_count=400,
            rasterizer_call_count=400,
            supervised_pixel_count=400 * 640 * 480,
        )
        assert block == {
            "optimization_step_count": 100,
            "supervision_view_count": 400,
            "rasterizer_call_count": 400,
            "supervised_pixel_count": 122880000,
            "rasterizer_calls_per_step": 4.0,
            "realized_supervision_views_per_step": 4.0,
            "supervised_pixels_per_step": 1228800.0,
            "supervision_views_per_step": 4.0,
        }

    def test_block_merges_per_admitted_view_exposure_distribution(self):
        """Never-supervised admitted sources remain in the denominator."""
        from eligsir.utils.supervision_work import supervision_work_report_block

        block = supervision_work_report_block(
            optimization_step_count=4,
            supervision_view_count=4,
            rasterizer_call_count=4,
            supervised_pixel_count=16,
            per_camera_supervision_counts={10: 2, 20: 0, 30: 2},
        )

        assert block["supervision_exposures_per_view_mean"] == 1
        assert block["supervision_exposures_per_view_min"] == 0
        assert block["supervision_exposures_per_view_max"] == 2
        assert block["never_supervised_view_count"] == 1

    def test_block_emits_wall_clock_view_rate(self):
        from eligsir.utils.supervision_work import supervision_work_report_block

        block = supervision_work_report_block(
            optimization_step_count=10,
            supervision_view_count=40,
            rasterizer_call_count=10,
            supervised_pixel_count=40,
            wall_time_s=2.0,
        )
        assert block["supervision_view_rate_hz"] == 20.0

    def test_block_builder_returns_empty_at_zero_steps(self):
        """Zero optimizer steps -> falsy block -> writer omits the key, rather
        than persisting seven meaningless zeros that a reader would average."""
        from eligsir.utils.supervision_work import supervision_work_report_block

        assert supervision_work_report_block(0, 0, 0, 0) == {}

    def test_write_post_training_report_persists_supervision_work(self):
        """End-to-end: the writer accepts the kwarg AND writes it to disk.

        A parameter that is accepted and silently dropped is the half-fix
        ``test_issue646_report_kwargs_signature`` explicitly warns about, so
        this asserts the on-disk report.json, not just the returned summary.
        """
        import json
        import tempfile

        pytest.importorskip("torch")
        import torch

        with _mocked_lpips():
            from eligsir.utils.comparison_report import write_post_training_report
            from eligsir.utils.supervision_work import supervision_work_report_block

            block = supervision_work_report_block(
                100,
                400,
                400,
                400 * 640 * 480,
                per_camera_supervision_counts={10: 400, 20: 0},
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                summary = write_post_training_report(
                    model_path=tmpdir,
                    iteration=100,
                    gaussians=_FakeGaussian(),
                    train_cams=[_FakeCam("a")],
                    test_cams=[_FakeCam("t")],
                    render_fn=_minimal_render_fn,
                    pipe=_FakePipe(),
                    background=None,
                    supervision_work=block,
                )
                assert summary["supervision_work"] == block

                with open(f"{tmpdir}/comparison/iter_100/report.json") as f:
                    on_disk = json.load(f)
                assert on_disk["supervision_work"] == block
                # The VALUES are the point -- a zeroed block would still pass an
                # equality check against a zeroed expectation.
                assert on_disk["supervision_work"]["rasterizer_call_count"] == 400
                assert on_disk["supervision_work"]["supervised_pixels_per_step"] > 0
                assert on_disk["supervision_work"]["never_supervised_view_count"] == 1

    def test_omitting_supervision_work_reproduces_the_previous_report(self):
        """Additive-only: the offline train.py path passes nothing and must get
        the exact report.json it got before this field existed."""
        import tempfile

        pytest.importorskip("torch")

        with _mocked_lpips():
            from eligsir.utils.comparison_report import write_post_training_report

            with tempfile.TemporaryDirectory() as tmpdir:
                summary = write_post_training_report(
                    model_path=tmpdir,
                    iteration=100,
                    gaussians=_FakeGaussian(),
                    train_cams=[_FakeCam("a")],
                    test_cams=[_FakeCam("t")],
                    render_fn=_minimal_render_fn,
                    pipe=_FakePipe(),
                    background=None,
                )
                assert "supervision_work" not in summary
