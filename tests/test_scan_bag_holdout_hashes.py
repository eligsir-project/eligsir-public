"""
Unit tests for scripts/scan_bag_holdout_hashes.py parser logic.

These tests verify the scanner's ability to extract and analyze holdout
metadata from report.json files without requiring GPU or training runs.
"""

import json
import tempfile
import os
from pathlib import Path


def create_mock_report(
    scene: str = "test_scene",
    holdout_hash: str = "abc123def456",
    realized_holdout_hash: str = "xyz789uvw012",
    n_test: int = 100,
    n_train: int = 500,
    holdout_selection: str = "online_every_nth_hashed",
) -> dict:
    """Create a minimal mock report.json structure."""
    return {
        "iteration": 5000,
        "n_train": n_train,
        "n_test": n_test,
        "mean_test_psnr": 24.5,
        "scene_source": f"/path/to/{scene}",
        "eval_metadata": {
            "eval_hold": 8,
            "holdout_hash": holdout_hash,
            "realized_holdout_hash": realized_holdout_hash,
            "holdout_selection": holdout_selection,
            "n_holdout": n_test,
        },
        "geometry": {
            "shape": {"n_live": 50000},
        },
    }


def test_is_bag_or_live_source():
    """Test detection of bag/live vs offline sources."""
    from scripts.scan_bag_holdout_hashes import is_bag_or_live_source

    # Bag/live sources
    assert is_bag_or_live_source({"holdout_selection": "online_every_nth"})
    assert is_bag_or_live_source({"holdout_selection": "online_every_nth_hashed"})

    # Offline sources
    assert not is_bag_or_live_source({"holdout_selection": "precomputed"})
    assert not is_bag_or_live_source({})


def test_extract_eval_metadata():
    """Test extraction of eval_metadata from a report."""
    from scripts.scan_bag_holdout_hashes import extract_eval_metadata

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.json")
        report = create_mock_report(
            holdout_hash="hash1",
            realized_holdout_hash="hash2",
        )

        with open(report_path, "w") as f:
            json.dump(report, f)

        metadata = extract_eval_metadata(report_path)
        assert metadata is not None
        assert metadata["holdout_hash"] == "hash1"
        assert metadata["realized_holdout_hash"] == "hash2"
        assert metadata["holdout_selection"] == "online_every_nth_hashed"


def test_get_scene_identity():
    """Test scene identity extraction."""
    from scripts.scan_bag_holdout_hashes import get_scene_identity

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.json")
        report = create_mock_report(scene="kitchen1")

        with open(report_path, "w") as f:
            json.dump(report, f)

        scene = get_scene_identity(report_path)
        assert scene == "kitchen1"

        # Test with deep path
        report["scene_source"] = "/very/deep/path/structure/floor3"
        with open(report_path, "w") as f:
            json.dump(report, f)

        scene = get_scene_identity(report_path)
        assert scene == "floor3"


def test_get_ingested_frame_count():
    """Test ingested frame count extraction."""
    from scripts.scan_bag_holdout_hashes import get_ingested_frame_count

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.json")
        report = create_mock_report(n_train=450)

        with open(report_path, "w") as f:
            json.dump(report, f)

        count = get_ingested_frame_count(report_path)
        assert count == 450


def test_find_reports_numeric_sorting():
    """Test that iterations are sorted numerically, not lexically."""
    from scripts.scan_bag_holdout_hashes import find_reports_in_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a model directory with multiple iteration directories
        model_path = os.path.join(tmpdir, "test_model")
        os.makedirs(model_path)

        comp_dir = os.path.join(model_path, "comparison")
        os.makedirs(comp_dir)

        # Create iteration directories with lexically-incorrect ordering
        iterations = [1000, 5000, 100, 50000, 2147483647]
        for it in iterations:
            it_dir = os.path.join(comp_dir, f"iter_{it}")
            os.makedirs(it_dir)
            report = create_mock_report(n_train=100 + it // 1000)
            with open(os.path.join(it_dir, "report.json"), "w") as f:
                json.dump(report, f)

        reports = find_reports_in_dir(tmpdir)

        # Should have exactly 1 report (highest iteration)
        assert len(reports) == 1
        model_dir, report_path = reports[0]
        assert "iter_2147483647" in report_path


def test_hash_divergence_detection():
    """Test grouping and divergence detection."""
    from scripts.scan_bag_holdout_hashes import (
        extract_eval_metadata,
        is_bag_or_live_source,
        get_scene_identity,
    )
    from collections import defaultdict

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock reports for a scene group with hash divergence
        reports_data = [
            ("model1", "scene1", "hash_a", 214),
            ("model2", "scene1", "hash_b", 215),  # Different hash, different frame count
            ("model3", "scene1", "hash_a", 214),  # Same hash, same frames
        ]

        for model_name, scene, hash_val, n_train in reports_data:
            model_dir = os.path.join(tmpdir, model_name, "comparison", "iter_5000")
            os.makedirs(model_dir)

            report = create_mock_report(
                scene=scene,
                holdout_hash=hash_val,
                n_train=n_train,
            )
            with open(os.path.join(model_dir, "report.json"), "w") as f:
                json.dump(report, f)

        # Now group them as the scanner would
        by_scene = defaultdict(list)

        for model_name, scene, hash_val, n_train in reports_data:
            report_path = os.path.join(tmpdir, model_name, "comparison", "iter_5000", "report.json")
            metadata = extract_eval_metadata(report_path)
            if is_bag_or_live_source(metadata):
                scene_name = get_scene_identity(report_path)
                by_scene[scene_name].append({
                    "holdout_hash": metadata.get("holdout_hash"),
                    "n_train": n_train,
                })

        # Should have one scene group with divergence
        assert len(by_scene) == 1
        assert "scene1" in by_scene
        hashes = set(r["holdout_hash"] for r in by_scene["scene1"])
        assert len(hashes) == 2  # Two distinct hashes (hash_a and hash_b)


if __name__ == "__main__":
    test_is_bag_or_live_source()
    test_extract_eval_metadata()
    test_get_scene_identity()
    test_get_ingested_frame_count()
    test_find_reports_numeric_sorting()
    test_hash_divergence_detection()
    print("All tests passed!")
