"""Contract tests for bundled dependency notices and provenance blockers."""

from pathlib import Path
import hashlib

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "THIRD_PARTY_PROVENANCE.yaml"


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_every_bundled_vendor_file_has_notice_and_provenance_row():
    data = _manifest()
    rows = {row["id"]: row for row in data["components"]}
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    vendor_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "gaussian_renderer" / "web_viewer" / "vendor").rglob("*")
        if path.is_file()
    )
    manifest_paths = {
        local_path
        for row in rows.values()
        for local_path in row["local_paths"]
        if local_path in vendor_files
    }
    assert manifest_paths == set(vendor_files)
    for local_path in vendor_files:
        assert local_path in notices


def test_provenance_checker_accepts_dependency_manifest():
    from scripts.check_publication_provenance import validate_dependency_manifest

    assert validate_dependency_manifest(ROOT) == []


def test_missing_external_integration_path_is_allowed_but_shipped_path_is_required(tmp_path):
    from scripts.check_publication_provenance import _missing_dependency_path_error

    assert _missing_dependency_path_error(tmp_path, "eligsir-orbslam3-src/") is None
    assert _missing_dependency_path_error(tmp_path, "lpipsPyTorch/") == (
        "dependency path does not exist: lpipsPyTorch/"
    )


def test_dependency_manifest_records_versions_hashes_and_unresolved_blockers():
    data = _manifest()
    rows = {row["id"]: row for row in data["components"]}
    assert {
        "chartjs",
        "playcanvas",
        "fused-ssim",
        "gsplat-rocm-fork",
        "lpips",
        "orbslam3-and-wrapper",
        "rvizsplat",
        "datasets",
    } <= rows.keys()

    assert rows["chartjs"]["license"] == "MIT"
    assert rows["chartjs"]["version"] == "4.4.7"
    assert len(rows["chartjs"]["sha256"]) == 64
    assert rows["playcanvas"]["license"] == "MIT"
    assert rows["playcanvas"]["version"] == "2.18.1"
    assert len(rows["playcanvas"]["sha256"]) == 64

    assert rows["fused-ssim"]["license"] == "MIT"
    assert rows["fused-ssim"]["source_pin"] == (
        "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
    )
    assert rows["fused-ssim"]["pin_kind"] == "publication-reproducibility"
    assert rows["gsplat-rocm-fork"]["source_pin"] == (
        "c6b363ab9239c958a4ac6619f725119dc8e5706b"
    )
    assert rows["gsplat-rocm-fork"]["upstream_base_commit"] == (
        "937e29912570c372bed6747a5c9bf85fed877bae"
    )
    assert rows["gsplat-rocm-fork"]["fork_license"] == "Apache-2.0"
    assert "AMD" in rows["gsplat-rocm-fork"]["notice_evidence"]
    assert rows["gsplat-rocm-fork"]["delta"]
    assert rows["gsplat-rocm-fork"]["release_blocked"] is False
    assert rows["gsplat-rocm-fork"]["status"] == "mechanically-noticed"
    assert "https://github.com/eligsir-project/gsplat-rocm.git" in rows[
        "gsplat-rocm-fork"
    ]["source_urls"]
    assert any(
        url.endswith("/commit/c6b363ab9239c958a4ac6619f725119dc8e5706b")
        for url in rows["gsplat-rocm-fork"]["source_urls"]
    )
    assert rows["lpips"]["license"] == "BSD-2-Clause"
    assert rows["lpips"]["source_pin"] == (
        "016e0ae19ceff83450590738948941a7eb9b95b7"
    )
    assert rows["lpips"]["downstream_reference_commit"] == (
        "54c035f7834b564019656c3e3fcc3646292f727d"
    )
    assert rows["lpips"]["copyright"] == "2020 Sou Uchida"
    assert rows["lpips"]["implementation_sha256"]
    assert len(rows["lpips"]["asset_manifest_sha256"]) == 64
    assert "PyTorch-hosted" in rows["lpips"]["blocker"]
    assert rows["lpips"]["asset_distribution"] == "external-only"
    assert rows["lpips"]["runtime_downloads"] is False
    assert rows["lpips"]["docker_inclusion"] is False
    assert rows["orbslam3-and-wrapper"]["license"] == "GPL-3.0"

    for component_id in (
        "lpips",
        "orbslam3-and-wrapper",
        "rvizsplat",
        "datasets",
    ):
        row = rows[component_id]
        assert row["status"] == "unresolved"
        assert row["release_blocked"] is True
        assert row["blocker"]
        assert row["source_urls"]


def test_license_index_scopes_apache_to_eligsir_original_code():
    index = (ROOT / "LICENSE").read_text(encoding="utf-8")
    rights = (ROOT / "docs" / "RELEASE_RIGHTS.md").read_text(encoding="utf-8")
    assert "EliGSiR-original" in index
    assert "LICENSES/Apache-2.0.txt" in index
    assert "GraphDeco-derived" in index
    assert "LICENSES/GraphDeco-Gaussian-Splatting.txt" in index
    assert (
        "It is not a blanket license and does not apply uniformly to every file "
        "in this source export."
    ) in index
    assert "Apache-2.0" in rights
    assert "GraphDeco-derived" in rights
    assert "unresolved" in rights.lower()
    assert (
        "Public release of the EliGSiR software has been approved by the applicable "
        "project/institutional rights holder."
    ) in rights


def test_license_texts_are_complete_and_graphdeco_upstream_is_retained():
    apache = (ROOT / "LICENSES" / "Apache-2.0.txt").read_text(encoding="utf-8")
    graphdeco = (ROOT / "LICENSES" / "GraphDeco-Gaussian-Splatting.txt").read_text(
        encoding="utf-8"
    )
    assert apache.startswith("\n                                 Apache License\n")
    assert "Version 2.0, January 2004" in apache
    for section in (
        "1. Definitions.",
        "2. Grant of Copyright",
        "3. Grant of Patent",
        "4. Redistribution",
        "5. Submission of Contributions",
        "6. Trademarks",
        "7. Disclaimer of Warranty",
        "8. Limitation of Liability",
        "9. Accepting Warranty or Additional Liability",
    ):
        assert section in apache
    assert "http://www.apache.org/licenses/LICENSE-2.0" in apache
    assert graphdeco.startswith("Gaussian-Splatting License")
    assert "non-commercially" in graphdeco
    assert "research and/or evaluation purposes only" in " ".join(graphdeco.split())
    assert "WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS" in graphdeco
    normalized_graphdeco = "\n".join(line.rstrip() for line in graphdeco.splitlines())
    normalized_legacy = "\n".join(
        line.rstrip() for line in (ROOT / "LICENSE.md").read_text(encoding="utf-8").splitlines()
    )
    assert normalized_graphdeco == normalized_legacy


def test_provenance_records_project_license_without_resolving_dependencies():
    from scripts.check_publication_provenance import load_matrix

    matrix = load_matrix(ROOT)
    original = next(
        rule for rule in matrix["rules"] if rule["id"] == "eligsir-project-code"
    )
    graphdeco = next(
        rule for rule in matrix["rules"] if rule["id"] == "graphdeco-baseline-layout"
    )
    assert original["classification"] == "eligsir-original"
    assert original["license"] == "Apache-2.0"
    assert original["license_status"] == "authorized"
    assert "RELEASE_RIGHTS.md" in original["evidence"]
    assert graphdeco["license_status"] == "retained-upstream"
    assert "research/evaluation" in graphdeco["license"]

    manifest = _manifest()
    project_license = manifest["project_license"]
    assert project_license["eligsir-original"]["license"] == "Apache-2.0"
    assert project_license["graphdeco-derived"]["license_status"] == "retained-upstream"
    assert all(
        row["status"] == "unresolved"
        for row in manifest["components"]
        if row.get("release_blocked") is True
    )


def test_fused_ssim_full_pin_and_notice_are_retained_in_every_docker_path():
    data = _manifest()
    row = next(item for item in data["components"] if item["id"] == "fused-ssim")
    pin = row["source_pin"]
    assert len(pin) == 40
    assert "Rahul Goel" in (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Permission is hereby granted" in (
        ROOT / "THIRD_PARTY_NOTICES.md"
    ).read_text(encoding="utf-8")
    for local_path in row["local_paths"]:
        text = (ROOT / local_path).read_text(encoding="utf-8")
        assert f"ARG FUSED_SSIM_COMMIT={pin}" in text, local_path
        assert 'git fetch --depth 1 origin "${FUSED_SSIM_COMMIT}"' in text, local_path
        assert 'git checkout --detach "${FUSED_SSIM_COMMIT}"' in text, local_path


def test_cuda_docker_paths_pin_upstream_gsplat_by_full_commit():
    pin = "937e29912570c372bed6747a5c9bf85fed877bae"
    for local_path in ("Dockerfile.cuda", "Dockerfile.cuda-noble"):
        text = (ROOT / local_path).read_text(encoding="utf-8")
        assert f"ARG GSPLAT_COMMIT={pin}" in text, local_path
        assert 'git fetch --depth 1 origin "${GSPLAT_COMMIT}"' in text, local_path
        assert 'git checkout --detach "${GSPLAT_COMMIT}"' in text, local_path
        assert "--branch v1.5.3" not in text, local_path


def test_cuda_and_rocm_docker_base_images_are_digest_pinned():
    cuda = (ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")
    cuda_base = (
        "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel"
        "@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a"
    )
    assert cuda.count(f"FROM {cuda_base} AS") == 2
    assert "FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel AS" not in cuda

    cuda_noble = (ROOT / "Dockerfile.cuda-noble").read_text(encoding="utf-8")
    noble_base = (
        "nvidia/cuda:12.8.2-devel-ubuntu24.04"
        "@sha256:7fc48a36965c6d0e5c9717dd899b2851cabb2dfef43d95dd12046a2d40bdc90f"
    )
    assert f"FROM {noble_base} AS cuda-noble-base" in cuda_noble
    assert "FROM nvidia/cuda:12.8.2-devel-ubuntu24.04 AS" not in cuda_noble

    rocm = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ARG BASE_ROCM=rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0"
        "@sha256:73136312e61ea15af04e89a9df3830bd3fa91c14aad755cde0e426e707f493ff"
    ) in rocm
    assert "FROM ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254 AS rocm-python-base-scratch" in rocm


def test_rocm_docker_defaults_pin_neutral_gsplat_fork_and_detach_exact_sha():
    repo = "https://github.com/eligsir-project/gsplat-rocm.git"
    pin = "c6b363ab9239c958a4ac6619f725119dc8e5706b"
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG GSPLAT_REPO={repo}" in text
    assert f"ARG GSPLAT_COMMIT={pin}" in text
    assert 'git fetch --depth 1 origin "${GSPLAT_COMMIT}"' in text
    assert 'git checkout --detach "${GSPLAT_COMMIT}"' in text

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert f'GSPLAT_REPO: "${{GSPLAT_REPO:-{repo}}}"' in compose
    assert f'GSPLAT_COMMIT: "${{GSPLAT_COMMIT:-{pin}}}"' in compose


def test_rocm_gsplat_clone_uses_optional_ephemeral_github_secret():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN --mount=type=secret,id=github_token,required=false" in dockerfile
    assert "/run/secrets/github_token" in dockerfile
    assert "export GIT_CONFIG_COUNT=1" in dockerfile
    assert "export GIT_CONFIG_KEY_0=http.extraheader" in dockerfile
    assert 'export GIT_CONFIG_VALUE_0="Authorization: Basic' in dockerfile
    assert "x-access-token:%s" in dockerfile
    assert "GIT_AUTH_HEADER" not in dockerfile
    assert "bearer" not in dockerfile.lower()
    assert 'git clone --filter=blob:none --no-checkout "${GSPLAT_REPO}" /tmp/gsplat' in dockerfile
    assert 'git fetch --depth 1 origin "${GSPLAT_COMMIT}"' in dockerfile
    assert "ARG GITHUB_TOKEN" not in dockerfile
    assert "ENV GITHUB_TOKEN" not in dockerfile
    assert "credential.helper" not in dockerfile
    assert "git config" not in dockerfile
    assert "https://${GITHUB_TOKEN}" not in dockerfile
    assert "--secret id=github_token,env=GITHUB_TOKEN" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")
    assert "--secret id=github_token,env=GITHUB_TOKEN" in (
        ROOT / "docker-compose.yml"
    ).read_text(encoding="utf-8")


def test_gsplat_pin_records_neutral_fork_identity_and_closed_blocker():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "docs" / "THIRD_PARTY_PROVENANCE.yaml")
    )
    private_fork_token = bytes.fromhex("626a6f65726e656c6c656e7331").decode()
    assert private_fork_token not in text
    assert "eligsir-project/gsplat-rocm" in text
    assert "c6b363ab9239c958a4ac6619f725119dc8e5706b" in text
    assert "release_blocked: false" in text
    assert "AMD" in text
    assert "Apache-2.0" in text
    assert "remains private until coordinated release" in text
    assert "public repository" not in text.lower()


def test_lpips_notice_separates_implementation_and_calibration_weights():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "S-aiueo32/lpips-pytorch" in notices
    assert "016e0ae19ceff83450590738948941a7eb9b95b7" in notices
    assert "54c035f7834b564019656c3e3fcc3646292f727d" in notices
    assert "Copyright (c) 2020 Sou Uchida" in notices
    assert "richzhang/PerceptualSimilarity" in notices
    assert "calibration weights" in notices
    assert "PyTorch-hosted ImageNet weights" in notices
    assert "explicit user materialization" in notices
    assert "not bundled in this export or any Docker image" in notices


def test_checker_rejects_a_retained_lpips_hash_mismatch(monkeypatch):
    from scripts import check_publication_provenance as checker

    data = _manifest()
    lpips = next(item for item in data["components"] if item["id"] == "lpips")
    first_path = next(iter(lpips["implementation_sha256"]))
    lpips["implementation_sha256"][first_path] = "0" * 64
    monkeypatch.setattr(checker, "load_dependency_manifest", lambda root: data)
    errors = checker.validate_dependency_manifest(ROOT)
    assert any("lpips hash mismatch" in error for error in errors)


def test_checker_rejects_a_vendor_hash_mismatch(tmp_path, monkeypatch):
    from scripts import check_publication_provenance as checker

    vendor = tmp_path / "gaussian_renderer" / "web_viewer" / "vendor"
    vendor.mkdir(parents=True)
    asset = vendor / "chart.umd.min.js"
    asset.write_bytes(b"changed vendor bytes")
    manifest = {
        "components": [
            {
                "id": "chartjs",
                "local_paths": ["gaussian_renderer/web_viewer/vendor/chart.umd.min.js"],
                "source_urls": ["https://example.invalid/chart"],
                "notice": "THIRD_PARTY_NOTICES.md#chartjs",
                "license": "MIT",
                "status": "mechanically-noticed",
                "release_blocked": False,
                "sha256": "0" * 64,
            }
        ]
    }
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        "gaussian_renderer/web_viewer/vendor/chart.umd.min.js", encoding="utf-8"
    )
    monkeypatch.setattr(checker, "load_dependency_manifest", lambda root: manifest)
    errors = checker.validate_dependency_manifest(tmp_path)
    assert any("sha256 mismatch for chartjs" in error for error in errors)
    assert hashlib.sha256(asset.read_bytes()).hexdigest() != manifest["components"][0]["sha256"]
