"""Self-test for scripts/audit_image_conversion_sites.py.

Verifies that the audit script correctly identifies conversion patterns
and maintains its classification heuristics.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_script_runs():
    """Audit script should run without errors."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 or result.returncode == 1  # 0=no unknown, 1=has unknown


def test_audit_script_json_output():
    """Audit script --json should produce valid JSON."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    data = json.loads(result.stdout)

    # Check required fields
    assert "total_sites" in data
    assert "classified_breakdown" in data
    assert "unclassified_count" in data
    assert "allowlisted_count" in data
    assert "sites" in data

    # Check breakdown
    breakdown = data["classified_breakdown"]
    assert "score_path" in breakdown
    assert "explicit_transform" in breakdown
    assert "visualization" in breakdown
    assert "unknown" in breakdown

    # Sum should match total
    total_classified = sum(breakdown.values())
    assert data["total_sites"] == total_classified + data["allowlisted_count"]


def test_default_score_boundary_audit_is_a_green_gate():
    """The CI-facing tracked score-boundary audit has no unaudited site."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["unclassified_count"] == 0


def test_json_audit_fails_closed_when_repo_scope_has_unreviewed_sites():
    """Machine-readable mode must not turn an unreviewed audit green."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--scope", "repo", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["unclassified_count"] > 0


def test_cpu_ci_executes_the_fail_closed_score_boundary_audit():
    workflow = (
        Path(__file__).parent.parent / ".github" / "workflows" / "ci-public.yml"
    ).read_text()
    assert "python scripts/audit_image_conversion_sites.py --json" in workflow


def test_audit_finds_conversion_patterns():
    """Audit should find common conversion patterns like astype/resize/save."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)
    patterns_found = set(s["pattern"] for s in data["sites"])

    # Check that major patterns are detected
    # The curated source export intentionally excludes conversion sites from
    # the private development corpus; retain checks on actual current sinks.
    assert ".save(" in patterns_found
    assert patterns_found <= {
        "astype(np.uint8)", "astype(np.float32)", ".resize(", ".save(",
        ".clamp(", ".permute(", ".transpose(", "torch.save(",
        "torchvision.utils.save_image(",
    }


def test_audit_reports_each_conversion_site_once():
    """One source location must have one review record, not AST-parent copies."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    sites = json.loads(result.stdout)["sites"]
    keys = [(site["path"], site["line"], site["pattern"]) for site in sites]
    assert len(keys) == len(set(keys))


def test_audit_classifies_visualization_paths():
    """Audit should heuristically classify paths in viz/ and report/ as visualization."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)

    # Find any sites in report or viz paths
    viz_sites = [
        s for s in data["sites"]
        if "report" in s["path"] or "viz/" in s["path"]
    ]

    # At least some should be classified as visualization (not all may be due to
    # the specificity of the patterns)
    if viz_sites:
        viz_classified = [s for s in viz_sites if s["classification"] == "visualization"]
        assert len(viz_classified) > 0, "Report/viz paths should include visualization-classified sites"


def test_audit_classification_values_are_valid():
    """Audit output should only use valid classification values."""
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)
    valid_classifications = {"score_path", "explicit_transform", "visualization", "unknown"}

    for site in data["sites"]:
        assert site["classification"] in valid_classifications, \
            f"Invalid classification: {site['classification']}"


def test_audit_excludes_third_party_and_vendored_dirs():
    """Audit must NOT scan .venv, site-packages, submodules/, ablator/, etc.

    This is critical for reproducibility: git ls-files is the source of truth
    (reproducible across machines/CI), not rglob (which picks up local virtualenvs).
    """
    audit_script = Path(__file__).parent.parent / "scripts" / "audit_image_conversion_sites.py"
    result = subprocess.run(
        [sys.executable, str(audit_script), "--json"],
        capture_output=True,
        text=True,
    )

    data = json.loads(result.stdout)
    paths = [s["path"] for s in data["sites"]]

    # These must not appear in the scanned files (regression guard against
    # reverting to rglob-only implementation)
    forbidden_patterns = [".venv", "site-packages", ".tox", "__pycache__", "submodules", "ablator"]

    for pattern in forbidden_patterns:
        matching = [p for p in paths if pattern in p]
        assert not matching, (
            f"Audit must not scan {pattern} (regression: reverting to rglob without "
            f"git ls-files). Found: {matching[:3]}"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
