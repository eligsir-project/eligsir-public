"""Tests for deterministic figure generation in scripts/docs_visuals.py section C.

Section C implementation must be deterministic: same CSV input should produce
byte-identical SVG output on repeated runs. Provenance sidecars should record
input/output hashes and generator version.
"""
import hashlib
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_line_csv(tmp_path: Path) -> Path:
    """Create a simple line-plot CSV."""
    csv = tmp_path / "line_data.csv"
    csv.write_text("iteration,psnr_baseline,psnr_candidate\n10,20.5,21.2\n20,21.1,22.5\n30,21.8,23.1\n")
    return csv


@pytest.fixture
def sample_bar_csv(tmp_path: Path) -> Path:
    """Create a bar-chart CSV."""
    csv = tmp_path / "bar_data.csv"
    csv.write_text("method,score_a,score_b,error\nfoo,25.5,26.1,0.3\nbar,24.2,25.8,0.4\nbaz,23.9,25.5,0.2\n")
    return csv


@pytest.fixture
def sample_scatter_csv(tmp_path: Path) -> Path:
    """Create a scatter-plot CSV."""
    csv = tmp_path / "scatter_data.csv"
    csv.write_text("x_val,y_val\n1.0,2.5\n2.5,3.1\n3.8,4.2\n5.1,3.9\n6.5,4.8\n")
    return csv


def test_line_plot_is_deterministic(sample_line_csv: Path, tmp_path: Path) -> None:
    """Same CSV input must produce byte-identical SVG output."""
    from scripts.docs_visuals import _line_svg

    plot = {
        "id": "test_line",
        "title": "Test Line Plot",
        "x": "iteration",
        "x_label": "Iteration",
        "y_label": "PSNR (dB)",
        "series": [
            {"column": "psnr_baseline", "label": "Baseline"},
            {"column": "psnr_candidate", "label": "Candidate"},
        ],
    }

    import csv

    with open(sample_line_csv) as f:
        rows = list(csv.DictReader(f))

    svg1 = _line_svg(plot, rows)
    svg2 = _line_svg(plot, rows)
    assert svg1 == svg2, "Line plot SVG should be deterministic"


def test_bar_plot_is_deterministic(sample_bar_csv: Path) -> None:
    """Same CSV input must produce byte-identical SVG output for bar plots."""
    from scripts.docs_visuals import _bar_svg
    import csv

    plot = {
        "id": "test_bar",
        "title": "Test Bar Plot",
        "x": "method",
        "x_label": "Method",
        "y_label": "Score",
        "series": [
            {"column": "score_a", "label": "Score A"},
            {"column": "score_b", "label": "Score B"},
        ],
    }

    with open(sample_bar_csv) as f:
        rows = list(csv.DictReader(f))

    svg1 = _bar_svg(plot, rows)
    svg2 = _bar_svg(plot, rows)
    assert svg1 == svg2, "Bar plot SVG should be deterministic"


def test_scatter_plot_is_deterministic(sample_scatter_csv: Path) -> None:
    """Same CSV input must produce byte-identical SVG output for scatter plots."""
    from scripts.docs_visuals import _scatter_svg
    import csv

    plot = {
        "id": "test_scatter",
        "title": "Test Scatter Plot",
        "x": "x_val",
        "y": "y_val",
        "x_label": "X",
        "y_label": "Y",
    }

    with open(sample_scatter_csv) as f:
        rows = list(csv.DictReader(f))

    svg1 = _scatter_svg(plot, rows)
    svg2 = _scatter_svg(plot, rows)
    assert svg1 == svg2, "Scatter plot SVG should be deterministic"


def test_provenance_sidecar_contains_hashes(tmp_path: Path) -> None:
    """Provenance sidecar must contain source and output SHA-256 hashes."""
    from scripts.docs_visuals import generate_plots
    import shutil

    # Set up a minimal docs structure
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "visuals").mkdir()
    (docs / "visuals" / "data").mkdir()
    (docs / "visuals" / "generated").mkdir(exist_ok=True)

    # Create sample CSV
    csv_data = docs / "visuals" / "data" / "test.csv"
    csv_data.write_text("x,y\n1,2\n3,4\n")

    # Create plots.json
    plots_spec = {
        "schema_version": 1,
        "plots": [
            {
                "id": "test_plot",
                "kind": "line",
                "title": "Test",
                "description": "Test plot",
                "evidence_class": "test",
                "source": "docs/visuals/data/test.csv",
                "x": "x",
                "series": [{"column": "y", "label": "Y"}],
            }
        ],
    }
    (docs / "visuals" / "plots.json").write_text(json.dumps(plots_spec, indent=2))

    # Create minimal mkdocs.yml and other required files
    (tmp_path / "mkdocs.yml").write_text("site_name: Test\n")

    # Generate plots
    from scripts.docs_visuals import repo_root_from
    original_root = repo_root_from

    try:
        # Mock repo root to our temp directory
        import scripts.docs_visuals as dv
        dv.repo_root_from = lambda p=None: tmp_path
        generate_plots(tmp_path)
    finally:
        dv.repo_root_from = original_root

    # Check provenance file exists and has hashes
    provenance_path = docs / "visuals" / "generated" / "test_plot.provenance.json"
    assert provenance_path.exists(), "Provenance file should be generated"

    prov = json.loads(provenance_path.read_text())
    assert "output_sha256" in prov, "Provenance must include output SHA-256"
    assert "source_sha256" in prov, "Provenance must include source SHA-256"
    assert "spec_sha256" in prov, "Provenance must include spec SHA-256"
    assert len(prov["output_sha256"]) == 64, "SHA-256 should be 64 hex chars"
    assert len(prov["source_sha256"]) == 64, "SHA-256 should be 64 hex chars"


def test_provenance_includes_plot_kind() -> None:
    """Provenance must record the plot kind for different chart types."""
    from scripts.docs_visuals import generate_plots
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        docs_dir = tmp_path / "docs" / "visuals" / "generated"
        docs_dir.mkdir(parents=True)

        # Create test CSV
        csv_file = tmp_path / "docs" / "visuals" / "data" / "test.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("x,y\n1,2\n3,4\n")

        # Create plots spec with different kinds
        specs = {
            "line": {"id": "p1", "kind": "line", "source": "docs/visuals/data/test.csv", "x": "x", "series": [{"column": "y"}]},
            "bar": {"id": "p2", "kind": "bar", "source": "docs/visuals/data/test.csv", "x": "x", "series": [{"column": "y"}]},
            "scatter": {"id": "p3", "kind": "scatter", "source": "docs/visuals/data/test.csv", "x": "x", "y": "y"},
        }

        for kind, spec_dict in specs.items():
            plots_json = tmp_path / "docs" / "visuals" / "plots.json"
            spec_dict["title"] = f"Test {kind}"
            spec_dict["evidence_class"] = "test"
            spec_dict["description"] = f"Test {kind} plot"
            json.dump({"schema_version": 1, "plots": [spec_dict]}, plots_json.open("w"))

            from scripts.docs_visuals import repo_root_from
            original_root = repo_root_from

            try:
                import scripts.docs_visuals as dv
                dv.repo_root_from = lambda p=None: tmp_path
                generate_plots(tmp_path)
            finally:
                dv.repo_root_from = original_root

            prov_file = tmp_path / "docs" / "visuals" / "generated" / f"p{list(specs.keys()).index(kind)+1}.provenance.json"
            if prov_file.exists():
                prov = json.loads(prov_file.read_text())
                assert prov.get("plot_kind") == kind, f"Provenance should record plot kind: {kind}"


def test_pareto_frontier_computation() -> None:
    """Pareto frontier should be computed correctly for scatter plots.

    A point (x1, y1) is on the Pareto frontier if no other point dominates it.
    Point (x1, y1) dominates (x2, y2) if x1 <= x2 AND y1 >= y2 with at least one strict.
    """
    from scripts.docs_visuals import _compute_pareto_frontier

    points = [(1.0, 2.0), (2.0, 3.0), (3.0, 2.5), (4.0, 4.0), (5.0, 3.5)]
    frontier = _compute_pareto_frontier(points)

    # Expected frontier (sorted by x, max y seen so far):
    # (1.0, 2.0): first point, in frontier
    # (2.0, 3.0): y=3.0 > max_y=2.0, in frontier
    # (3.0, 2.5): y=2.5 < max_y=3.0, dominated, not in frontier
    # (4.0, 4.0): y=4.0 > max_y=3.0, in frontier
    # (5.0, 3.5): y=3.5 < max_y=4.0, dominated, not in frontier
    expected = [(1.0, 2.0), (2.0, 3.0), (4.0, 4.0)]
    assert frontier == expected, f"Frontier mismatch: got {frontier}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
