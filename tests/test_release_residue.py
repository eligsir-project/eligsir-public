"""Checks for private campaign residue in the publication export."""

from __future__ import annotations

import tomllib
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_campaign_runner_is_explicitly_quarantined() -> None:
    data = tomllib.loads((ROOT / "configs/ablator_runner_version.toml").read_text(encoding="utf-8"))
    assert data["runner"] == {
        "enabled": False,
        "status": "quarantined",
        "reason": "Campaign runner configuration is not part of the publication export.",
    }
    text = (ROOT / "configs/ablator_runner_version.toml").read_text(encoding="utf-8")
    assert ("eligsir-project" + "/ablator") not in text
    assert "~/git/ablator" not in text


def test_container_defaults_do_not_require_private_or_legacy_images() -> None:
    paths = (
        "Dockerfile",
        "Dockerfile.cuda",
        "Dockerfile.cuda-noble",
        "docker-compose.yml",
        "docker/open3d_rocm.Dockerfile",
        "docker/orbbec_deps.Dockerfile",
    )
    text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in paths)
    assert ("ghcr.io/" + "eligsir-project") not in text
    assert ("localhost/" + "3dgs-" + "mcmc-rocm") not in text
    assert ("3dgs-" + "mcmc-rocm") not in text
    assert "BASE_ROCM_PYTHON" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "BASE_IMAGE" in (ROOT / "docker/orbbec_deps.Dockerfile").read_text(encoding="utf-8")
    open3d = (ROOT / "docker/open3d_rocm.Dockerfile").read_text(encoding="utf-8")
    assert "ARG ELIGSIR_BASE" in open3d
    assert "FROM ${ELIGSIR_BASE}" in open3d


def test_compose_config_validates_caller_supplied_build_contract_without_building() -> None:
    docker = shutil.which("docker")
    if docker is None:
        return
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${ELIGSIR_IMAGE:?" in compose
    assert "${BASE_ROCM_PYTHON:?" in compose
    assert "${GSPLAT_REPO:-https://github.com/eligsir-project/gsplat-rocm.git}" in compose
    assert "${GSPLAT_COMMIT:-c6b363ab9239c958a4ac6619f725119dc8e5706b}" in compose

    env = os.environ.copy()
    env.update(
        {
            "ELIGSIR_IMAGE": "eligsir:local",
            "BASE_ROCM_PYTHON": "registry.example/eligsir-rocm-python@sha256:" + "0" * 64,
            "GSPLAT_REPO": "https://example.invalid/gsplat-rocm.git",
            "GSPLAT_COMMIT": "0" * 40,
        }
    )
    result = subprocess.run(
        [docker, "compose", "-f", "docker-compose.yml", "config", "--quiet"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "pull" not in (result.stderr + result.stdout).lower()
    assert "build" not in (result.stderr + result.stdout).lower()


def test_lock_root_matches_project_identity() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    project = next(package for package in lock["package"] if package.get("source", {}).get("editable") == ".")
    assert project["name"] == "eligsir"
    assert project["version"] == "0.2.0"


def test_pruned_docs_skip_visual_audit_but_keep_other_checks(monkeypatch) -> None:
    marker = (ROOT / "docs/PUBLICATION_SCOPE.md").read_text(encoding="utf-8")
    assert "intentionally pruned" in marker
    assert "docs/visuals/" in marker
    import scripts.check_docs_integrity as docs_check

    def unexpected_visual_call(*_args, **_kwargs):
        raise AssertionError("publication scope must skip visual generation/audit only")

    monkeypatch.setattr(docs_check, "generate_all", unexpected_visual_call)
    monkeypatch.setattr(docs_check, "audit_visuals", unexpected_visual_call)

    errors = docs_check.audit(ROOT)
    assert not any("visual generation failed" in error for error in errors)
    # Missing source-corpus pages are out of scope, but retained docs checks
    # still execute and the current export's retained pages are clean.
    assert not any(error.startswith("missing ") for error in errors)


def test_pruned_docs_still_catch_retired_reference_in_retained_page(tmp_path) -> None:
    import scripts.check_docs_integrity as docs_check

    (tmp_path / "docs").mkdir()
    (tmp_path / "mkdocs.yml").write_text(
        "plugins:\n  - awesome-pages\nhooks:\n  - scripts/mkdocs_hook.py\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/PUBLICATION_SCOPE.md").write_text(
        "publication export; docs/visuals/ intentionally pruned\n", encoding="utf-8"
    )
    (tmp_path / "docs/retained.md").write_text(
        "This retained page must not link eligsir/cli/ros_node.py.\n", encoding="utf-8"
    )

    errors = docs_check.audit(tmp_path)
    assert any("docs/retained.md" in error and "retired current API/path" in error for error in errors)


def test_pruned_docs_still_catch_forbidden_retained_artifact(tmp_path) -> None:
    import scripts.check_docs_integrity as docs_check

    (tmp_path / "docs").mkdir()
    (tmp_path / "mkdocs.yml").write_text(
        "plugins:\n  - awesome-pages\nhooks:\n  - scripts/mkdocs_hook.py\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/PUBLICATION_SCOPE.md").write_text(
        "publication export; docs/visuals/ intentionally pruned\n", encoding="utf-8"
    )
    (tmp_path / "docs/.DS_Store").write_bytes(b"residue")

    errors = docs_check.audit(tmp_path)
    assert any("tracked documentation artifact must be removed" in error and ".DS_Store" in error for error in errors)
