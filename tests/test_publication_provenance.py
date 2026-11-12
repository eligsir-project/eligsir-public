"""Contract tests for the publication provenance and public-CI surfaces."""

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[1]


def test_provenance_matrix_covers_retained_code_without_ambiguous_rules():
    """Every retained code file has exactly one highest-priority provenance rule."""

    from scripts.check_publication_provenance import validate_matrix

    report = validate_matrix(ROOT)

    assert report.errors == [], "\n".join(report.errors)
    assert report.total_code_files > 0
    assert report.class_counts["graphdeco-derived"] > 0
    assert report.class_counts["eligsir-original"] > 0
    assert report.class_counts["third-party-unresolved"] > 0


def test_matrix_does_not_claim_a_license_for_unresolved_third_party_code():
    """Unresolved vendored/integration code stays explicitly unresolved."""

    from scripts.check_publication_provenance import load_matrix

    matrix = load_matrix(ROOT)
    unresolved = {
        rule["id"]: rule for rule in matrix["rules"] if rule["classification"] == "third-party-unresolved"
    }

    assert unresolved
    for rule in unresolved.values():
        assert rule["license_status"] == "unresolved"
        assert rule["license"] in {"UNKNOWN", "REVIEW_REQUIRED"}


def test_public_ci_is_hosted_read_only_and_has_no_publish_or_internal_endpoints():
    """The only retained workflow is a read-only GitHub-hosted validation job."""

    from scripts.check_publication_provenance import validate_public_ci

    errors = validate_public_ci(ROOT)
    assert errors == [], "\n".join(errors)


def test_public_workflow_uses_explicit_non_editable_contract_dependencies():
    """The staged public CI commands work without installing this flat layout."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci-public.yml").read_text(encoding="utf-8")
    )
    commands = [
        step.get("run", "")
        for step in workflow["jobs"]["validate"]["steps"]
        if isinstance(step, dict)
    ]
    install = next(command for command in commands if "pip install" in command)
    assert " -e " not in install
    assert "pyyaml" in install.lower()
    assert "pytest" in install.lower()
    cpu_torch = next(command for command in commands if "torch==" in command)
    assert "torch==2.5.1+cpu" in cpu_torch
    assert "torchvision==0.20.1+cpu" in cpu_torch
    assert "https://download.pytorch.org/whl/cpu" in cpu_torch
    assert any(
        "pip wheel" in command
        and "--no-deps" in command
        and "--no-build-isolation" not in command
        for command in commands
    )
    test_command = next(command for command in commands if "python -m pytest" in command)
    for test_name in (
        "tests/test_publication_identity.py",
        "tests/test_publication_interface.py",
        "tests/test_publication_provenance.py",
        "tests/test_publication_notices.py",
        "tests/test_release_residue.py",
    ):
        assert test_name in test_command

    checker = subprocess.run(
        [sys.executable, "scripts/check_publication_provenance.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checker.returncode == 0, checker.stderr + checker.stdout


def test_public_wheel_build_works_in_a_clean_venv(tmp_path):
    """The hosted workflow's wheel command must work without ambient setuptools."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/ci-public.yml").read_text(encoding="utf-8")
    )
    commands = [
        step.get("run", "")
        for step in workflow["jobs"]["validate"]["steps"]
        if isinstance(step, dict)
    ]
    wheel_command = next(command for command in commands if "pip wheel" in command)
    assert "--no-build-isolation" not in wheel_command

    venv = tmp_path / "venv"
    wheel_dir = tmp_path / "wheel"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            ".",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert list(wheel_dir.glob("eligsir-*.whl")), result.stdout


def test_orbslam_source_is_explicitly_excluded_from_shipped_counts():
    """Ignored external ORB-SLAM source is recorded but not counted as shipped."""
    from scripts.check_publication_provenance import iter_code_files, load_matrix

    paths = iter_code_files(ROOT)
    assert not any(path.startswith("eligsir-orbslam3-src/") for path in paths)
    matrix = load_matrix(ROOT)
    assert "eligsir-orbslam3-src/**" in matrix["excluded_paths"]


def test_graphdeco_header_cannot_be_classified_by_broad_project_rule(tmp_path):
    """A copied upstream header needs an explicit, targeted derived rule."""
    from scripts.check_publication_provenance import validate_matrix

    source = tmp_path / "eligsir" / "copied.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Copyright (C) 2023, Inria\n"
        "# GRAPHDECO research group\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "PROVENANCE_MATRIX.md").write_text(
        "```yaml\n"
        "coverage:\n"
        "  total_code_files: 1\n"
        "  eligsir-original: 1\n"
        "  graphdeco-derived: 0\n"
        "  third-party-unresolved: 0\n"
        "rules:\n"
        "  - id: broad\n"
        "    priority: 1\n"
        "    classification: eligsir-original\n"
        "    license: REVIEW_REQUIRED\n"
        "    license_status: unresolved\n"
        "    evidence: broad fallback\n"
        "    paths: [eligsir/**]\n"
        "```\n",
        encoding="utf-8",
    )
    report = validate_matrix(tmp_path)
    assert any("GraphDeco header" in error for error in report.errors)
