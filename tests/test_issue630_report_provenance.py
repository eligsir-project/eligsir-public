"""Tests for issue #630 report provenance detection.

Verifies that regenerated reports are distinguishable from native training loop
reports via the report_provenance block in report.json.
"""

import pytest

# Deliberately torch-free: this test runs in the CPU CI image where torch is not installed.
# Import only the provenance inference function, which has no torch dependency.
from eligsir.io.report_provenance import infer_report_provenance


class TestReportProvenance:
    """Tests for report provenance inference (native vs. regenerated)."""

    def test_native_from_training_loop_with_streaming_scene(self):
        """Native: streaming_scene is non-None (live training state exists)."""
        result = infer_report_provenance(
            log_prefix="streaming_report",
            train_cams=["cam1", "cam2"],
            test_cams=["test1"],
            streaming_scene=object(),  # Non-None object
            tb_writer=None,
            gaussians=None,
        )
        assert result["kind"] == "training_loop"
        assert any("streaming_scene is not None" in s for s in result["evidence"])

    def test_native_from_training_loop_with_tb_writer(self):
        """Native: tb_writer is non-None (TensorBoard writer exists)."""
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=["cam1", "cam2"],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=object(),  # Non-None object
            gaussians=None,
        )
        assert result["kind"] == "training_loop"
        assert any("tb_writer is not None" in s for s in result["evidence"])

    def test_native_from_training_loop_with_optimizer(self):
        """Native: train_cams non-empty + gaussians has optimizer."""
        mock_gaussians = type("Gaussians", (), {"optimizer": object()})()
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=["cam1"],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=mock_gaussians,
        )
        assert result["kind"] == "training_loop"
        assert result["reasons"]["gaussians_has_optimizer"] is True

    def test_native_from_training_loop_with_lineage(self):
        """Native: gaussians has lineage buffers from training."""
        mock_gaussians = type("Gaussians", (), {
            "_lineage_preserving_setup_events": [],
        })()
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=["cam1"],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=mock_gaussians,
        )
        assert result["kind"] == "training_loop"
        assert result["reasons"]["gaussians_has_lineage"] is True
        assert any("lineage buffers" in s for s in result["evidence"])

    def test_regenerated_via_render_report_primary_signal(self):
        """Regenerated: log_prefix == 'render_report' (primary signal)."""
        result = infer_report_provenance(
            log_prefix="render_report",
            train_cams=[],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=None,
        )
        assert result["kind"] == "regenerated_checkpoint"
        assert result["reasons"]["log_prefix_is_render_report"] is True
        assert any("render_report.py" in s for s in result["evidence"])

    def test_regenerated_via_pattern_empty_train_no_optimizer(self):
        """Regenerated: train_cams=[] + test_cams non-empty + no optimizer."""
        mock_gaussians = type("Gaussians", (), {"optimizer": None})()
        result = infer_report_provenance(
            log_prefix="some_report",
            train_cams=[],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=mock_gaussians,
        )
        assert result["kind"] == "regenerated_checkpoint"
        assert result["reasons"]["train_cams_empty"] is True
        assert result["reasons"]["test_cams_present"] is True
        assert result["reasons"]["gaussians_has_optimizer"] is False
        assert any("train_cams is empty" in s for s in result["evidence"])
        assert any("PLY-loaded" in s for s in result["evidence"])

    def test_unknown_insufficient_evidence(self):
        """Unknown: no clear signals either way (fail-closed)."""
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=[],
            test_cams=[],
            streaming_scene=None,
            tb_writer=None,
            gaussians=None,
        )
        assert result["kind"] == "unknown"
        assert len(result["evidence"]) == 0

    def test_unknown_with_reasons_recorded(self):
        """Unknown verdict still records all signals examined."""
        result = infer_report_provenance(
            log_prefix="some_prefix",
            train_cams=[],
            test_cams=[],
            streaming_scene=None,
            tb_writer=None,
            gaussians=None,
        )
        assert result["kind"] == "unknown"
        # Reasons must record the signals we checked
        assert "log_prefix" in result["reasons"]
        assert "streaming_scene_present" in result["reasons"]
        assert "tb_writer_present" in result["reasons"]
        assert result["reasons"]["log_prefix"] == "some_prefix"
        assert result["reasons"]["streaming_scene_present"] is False
        assert result["reasons"]["tb_writer_present"] is False

    def test_none_inputs_normalize_to_empty_lists(self):
        """None train_cams/test_cams are normalized to empty lists."""
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=None,  # Will be normalized to []
            test_cams=None,   # Will be normalized to []
            streaming_scene=None,
            tb_writer=None,
            gaussians=None,
        )
        # With both empty, should be unknown
        assert result["kind"] == "unknown"

    def test_block_structure_matches_convention(self):
        """Provenance block has the expected structure (kind, evidence, reasons)."""
        result = infer_report_provenance(
            log_prefix="render_report",
            train_cams=[],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=None,
        )
        assert "kind" in result
        assert "evidence" in result
        assert "reasons" in result
        assert isinstance(result["evidence"], list)
        assert isinstance(result["reasons"], dict)

    def test_fail_closed_regenerated_pattern_without_optimizer_check(self):
        """Fail-closed: don't claim regenerated without optimizer signal."""
        # Empty train_cams + non-empty test_cams ALONE is not enough to claim regen.
        # Must also have no optimizer.
        mock_gaussians = type("Gaussians", (), {"optimizer": object()})()
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=[],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=mock_gaussians,
        )
        # With an optimizer, we fall through to native logic
        # (train_cams empty is weak, but optimizer + test_cams present is still ambiguous)
        # So we end up unknown, not regenerated
        assert result["kind"] == "unknown"

    def test_lineage_seed_uid_range_alternate_signal(self):
        """Native: seed_uid_range is an alternate lineage signal."""
        mock_gaussians = type("Gaussians", (), {
            "seed_uid_range": (0, 100),
        })()
        result = infer_report_provenance(
            log_prefix="report",
            train_cams=["cam1"],
            test_cams=["test1"],
            streaming_scene=None,
            tb_writer=None,
            gaussians=mock_gaussians,
        )
        assert result["kind"] == "training_loop"
        assert result["reasons"]["gaussians_has_seed_uid_range"] is True
