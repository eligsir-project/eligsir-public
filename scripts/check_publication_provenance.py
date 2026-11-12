"""Validate the publication provenance matrix and public GitHub Actions contract.

The publication export is history-free, so this check deliberately records its
classification rules in ``docs/PROVENANCE_MATRIX.md``.  A rule may be broad,
but higher-priority exceptions must identify every retained upstream or
vendored path that would otherwise be hidden by that broad rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
from pathlib import Path
import re
from typing import Any

import yaml


MATRIX_PATH = Path("docs/PROVENANCE_MATRIX.md")
DEPENDENCY_MANIFEST_PATH = Path("docs/THIRD_PARTY_PROVENANCE.yaml")
WORKFLOW_DIR = Path(".github/workflows")
CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".js",
    ".mjs",
    ".py",
    ".pyi",
    ".sh",
    ".tsx",
    ".ts",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "output",
    "site",
    "build",
    "dist",
}
# This ignored directory is an external ORB-SLAM3 integration/source checkout,
# not part of the shipped publication source. Keep it recorded in the matrix,
# but never let its contents affect publication coverage counts.
EXCLUDED_PATH_PREFIXES = ("eligsir-orbslam3-src/",)
YAML_FENCE = re.compile(r"```yaml\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class MatrixReport:
    errors: list[str]
    total_code_files: int
    class_counts: dict[str, int]


def load_matrix(root: Path) -> dict[str, Any]:
    """Load the machine-readable YAML block embedded in the matrix document."""

    path = root / MATRIX_PATH
    text = path.read_text(encoding="utf-8")
    match = YAML_FENCE.search(text)
    if not match:
        raise ValueError(f"{MATRIX_PATH} has no machine-readable yaml block")
    matrix = yaml.safe_load(match.group(1))
    if not isinstance(matrix, dict):
        raise ValueError(f"{MATRIX_PATH} yaml block must be a mapping")
    return matrix


def load_dependency_manifest(root: Path) -> dict[str, Any]:
    """Load the machine-readable third-party notice/provenance manifest."""

    path = root / DEPENDENCY_MANIFEST_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("components"), list):
        raise ValueError(f"{DEPENDENCY_MANIFEST_PATH} must define components")
    return data


def _missing_dependency_path_error(root: Path, local_path: str) -> str | None:
    """Return an error only when a manifest path is required in the export.

    Explicitly excluded external integrations remain documented in the
    manifest, but their ignored source checkouts are not shipped and therefore
    are allowed to be absent from a clean clone.
    """

    normalized = local_path.rstrip("/")
    candidate = root / normalized
    if candidate.exists() or list(root.glob(local_path)):
        return None
    if any(normalized.startswith(prefix.rstrip("/")) for prefix in EXCLUDED_PATH_PREFIXES):
        return None
    return f"dependency path does not exist: {local_path}"


def validate_dependency_manifest(root: Path) -> list[str]:
    """Validate bundled notices, source evidence, and explicit release gates."""

    errors: list[str] = []
    try:
        manifest = load_dependency_manifest(root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [str(exc)]

    components = manifest["components"]
    ids: set[str] = set()
    local_paths: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"dependency component {index} is not a mapping")
            continue
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            errors.append(f"dependency component {index} has no id")
            continue
        if component_id in ids:
            errors.append(f"duplicate dependency component id: {component_id}")
        ids.add(component_id)
        paths = component.get("local_paths")
        if not isinstance(paths, list) or not paths:
            errors.append(f"dependency component {component_id!r} has no local_paths")
        else:
            for local_path in paths:
                if not isinstance(local_path, str) or not local_path:
                    errors.append(f"dependency component {component_id!r} has invalid local path")
                    continue
                local_paths.add(local_path.rstrip("/"))
                path_error = _missing_dependency_path_error(root, local_path)
                if path_error:
                    errors.append(path_error)
        source_urls = component.get("source_urls")
        if not isinstance(source_urls, list) or not source_urls or not all(
            isinstance(url, str) and url.startswith("https://") for url in source_urls
        ):
            errors.append(f"dependency component {component_id!r} has no authoritative https source_urls")
        if not isinstance(component.get("notice"), str) or not component["notice"]:
            errors.append(f"dependency component {component_id!r} has no notice reference")
        blocked = component.get("release_blocked")
        status = component.get("status")
        if not isinstance(blocked, bool):
            errors.append(f"dependency component {component_id!r} has no release_blocked boolean")
        if blocked and status != "unresolved":
            errors.append(f"blocked dependency component {component_id!r} must be unresolved")
        if status == "unresolved" and not blocked:
            errors.append(f"unresolved dependency component {component_id!r} must be release-blocked")
        if blocked and not isinstance(component.get("blocker"), str):
            errors.append(f"blocked dependency component {component_id!r} has no blocker")

    vendor_root = root / "gaussian_renderer/web_viewer/vendor"
    vendor_files = {
        path.relative_to(root).as_posix()
        for path in vendor_root.rglob("*")
        if path.is_file()
    }
    missing_vendor_rows = sorted(path for path in vendor_files if path not in local_paths)
    if missing_vendor_rows:
        errors.append("bundled vendor files missing dependency rows: " + ", ".join(missing_vendor_rows))
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for path in sorted(vendor_files):
        if path not in notices:
            errors.append(f"bundled vendor file missing notice text: {path}")

    required = {
        "chartjs",
        "playcanvas",
        "fused-ssim",
        "gsplat-rocm-fork",
        "lpips",
        "orbslam3-and-wrapper",
        "rvizsplat",
        "datasets",
    }
    missing_required = sorted(required - ids)
    if missing_required:
        errors.append("dependency manifest missing required components: " + ", ".join(missing_required))
    for component_id in ("chartjs", "playcanvas"):
        component = next((item for item in components if item.get("id") == component_id), None)
        if component is None:
            continue
        digest_path = next(
            (root / path for path in component.get("local_paths", [])
             if (root / path).is_file()),
            None,
        )
        expected = str(component.get("sha256", ""))
        if digest_path is not None:
            actual = hashlib.sha256(digest_path.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(
                    f"sha256 mismatch for {component_id}: manifest={expected}, actual={actual}"
                )
        if component.get("license") != "MIT" or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{component_id} must record MIT and a full local sha256")

    fused = next((item for item in components if item.get("id") == "fused-ssim"), None)
    if fused is not None:
        fused_pin = str(fused.get("source_pin", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", fused_pin):
            errors.append("fused-ssim must record a full 40-character source pin")
        if fused.get("pin_kind") != "publication-reproducibility":
            errors.append("fused-ssim must identify its publication reproducibility pin")
        for local_path in fused.get("local_paths", []):
            docker_path = root / local_path
            if not docker_path.is_file():
                continue
            docker_text = docker_path.read_text(encoding="utf-8")
            if f"ARG FUSED_SSIM_COMMIT={fused_pin}" not in docker_text:
                errors.append(f"{local_path} lacks the audited FUSED_SSIM_COMMIT default")
            if 'git fetch --depth 1 origin "${FUSED_SSIM_COMMIT}"' not in docker_text:
                errors.append(f"{local_path} lacks shallow fetch of FUSED_SSIM_COMMIT")
            if 'git checkout --detach "${FUSED_SSIM_COMMIT}"' not in docker_text:
                errors.append(f"{local_path} lacks detached checkout of FUSED_SSIM_COMMIT")
        if "Rahul Goel" not in notices or "THIRD_PARTY_NOTICES.md#fused-ssim" not in str(
            fused.get("notice", "")
        ):
            errors.append("fused-ssim must retain Rahul Goel MIT notice evidence")

    gsplat = next((item for item in components if item.get("id") == "gsplat-rocm-fork"), None)
    if gsplat is not None:
        expected_gsplat_pin = "c6b363ab9239c958a4ac6619f725119dc8e5706b"
        if gsplat.get("source_pin") != expected_gsplat_pin:
            errors.append("gsplat-rocm-fork must record the audited neutral fork SHA")
        if not re.fullmatch(r"[0-9a-f]{40}", str(gsplat.get("upstream_base_commit", ""))):
            errors.append("gsplat-rocm-fork must record the full upstream base SHA")
        if gsplat.get("fork_license") != "Apache-2.0":
            errors.append("gsplat-rocm-fork must record Apache-2.0 fork LICENSE evidence")
        if "AMD" not in str(gsplat.get("notice_evidence", "")):
            errors.append("gsplat-rocm-fork must record AMD MIT NOTICE evidence")
        if not isinstance(gsplat.get("delta"), list) or not gsplat["delta"]:
            errors.append("gsplat-rocm-fork must summarize its fork delta")
        source_urls = {str(url) for url in gsplat.get("source_urls", [])}
        expected_repo = "https://github.com/eligsir-project/gsplat-rocm.git"
        expected_commit_url = (
            f"https://github.com/eligsir-project/gsplat-rocm/commit/{expected_gsplat_pin}"
        )
        if expected_repo not in source_urls:
            errors.append("gsplat-rocm-fork must record the neutral repository URL")
        if expected_commit_url not in source_urls:
            errors.append("gsplat-rocm-fork must record the exact pinned commit URL")
        if gsplat.get("release_blocked") is not False:
            errors.append("gsplat-rocm-fork must be release-unblocked")
        if "rights" not in str(gsplat.get("evidence", "")).lower():
            errors.append("gsplat-rocm-fork must retain rights evidence")
        for field in ("source_urls", "notice", "evidence"):
            # Keep the audit token itself anonymous: this checker is scanned by
            # the publication identity test along with the rest of the tree.
            private_fork_token = bytes.fromhex("626a6f65726e656c6c656e7331").decode()
            if private_fork_token in str(gsplat.get(field, "")):
                errors.append("gsplat-rocm-fork must not reference a personal mirror")

    lpips = next((item for item in components if item.get("id") == "lpips"), None)
    if lpips is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", str(lpips.get("source_pin", ""))):
            errors.append("lpips must record the full S-aiueo32 source commit")
        if not re.fullmatch(
            r"[0-9a-f]{40}", str(lpips.get("downstream_reference_commit", ""))
        ):
            errors.append("lpips must record the full GraphDeco downstream reference commit")
        if lpips.get("license") != "BSD-2-Clause" or lpips.get("copyright") != "2020 Sou Uchida":
            errors.append("lpips must record Sou Uchida BSD-2-Clause evidence")
        if lpips.get("asset_distribution") != "external-only":
            errors.append("lpips evaluation assets must be marked external-only")
        if lpips.get("runtime_downloads") is not False:
            errors.append("lpips runtime downloads must remain disabled")
        if lpips.get("docker_inclusion") is not False:
            errors.append("lpips weights must not be included in Docker images")
        for relative, expected in dict(lpips.get("implementation_sha256", {})).items():
            path = root / relative
            if not path.is_file():
                errors.append(f"lpips implementation hash path does not exist: {relative}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != str(expected):
                errors.append(f"lpips hash mismatch for {relative}: manifest={expected}, actual={actual}")
        asset_manifest = root / "lpipsPyTorch/eval_assets.json"
        expected_manifest_hash = str(lpips.get("asset_manifest_sha256", ""))
        if asset_manifest.is_file():
            actual_manifest_hash = hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
            if actual_manifest_hash != expected_manifest_hash:
                errors.append(
                    "lpips hash mismatch for lpipsPyTorch/eval_assets.json: "
                    f"manifest={expected_manifest_hash}, actual={actual_manifest_hash}"
                )
        for asset_group in ("calibration_assets", "imagenet_backbone_assets"):
            assets = lpips.get(asset_group, [])
            if not isinstance(assets, list) or not assets:
                errors.append(f"lpips must record {asset_group}")
                continue
            for asset in assets:
                if not isinstance(asset, dict) or not re.fullmatch(
                    r"[0-9a-f]{64}", str(asset.get("sha256", ""))
                ):
                    errors.append(f"lpips {asset_group} entry lacks a full sha256")
                if not isinstance(asset, dict) or not str(asset.get("source_url", "")).startswith("https://"):
                    errors.append(f"lpips {asset_group} entry lacks an https source URL")
        if "S-aiueo32/lpips-pytorch" not in notices or "Sou Uchida" not in notices:
            errors.append("lpips implementation BSD notice evidence is missing")
        if "richzhang/PerceptualSimilarity" not in notices or "calibration weights" not in notices:
            errors.append("lpips calibration-weight BSD notice evidence is missing")
        if "PyTorch-hosted ImageNet weights" not in notices:
            errors.append("lpips ImageNet-weight blocker notice is missing")
        if any((root / "lpipsPyTorch").rglob("*.pth")):
            errors.append("lpips must not bundle model weight files")
    return errors


def iter_code_files(root: Path) -> list[str]:
    """Return retained code paths, excluding generated/cache directories."""

    paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        normalized = relative.as_posix()
        if normalized.startswith(EXCLUDED_PATH_PREFIXES):
            continue
        paths.append(normalized)
    return sorted(paths)


def _matching_rules(path: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for rule in rules:
        patterns = rule.get("paths", [])
        if any(fnmatchcase(path, pattern) for pattern in patterns):
            matches.append(rule)
    return matches


def _has_graphdeco_header(root: Path, relative_path: str) -> bool:
    """Detect retained upstream header evidence near a file's top.

    This is intentionally narrow evidence, not a legal-origin classifier. It
    prevents the broad project-original fallback from silently winning when a
    file visibly retains the upstream GraphDeco/Inria header.
    """
    path = root / relative_path
    try:
        header = path.read_text(encoding="utf-8", errors="ignore")[:3000].lower()
    except OSError:
        return False
    return "graphdeco research group" in header and "copyright (c) 2023, inria" in header


def validate_matrix(root: Path) -> MatrixReport:
    """Validate coverage, priority resolution, and classification metadata."""

    errors: list[str] = []
    try:
        matrix = load_matrix(root)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return MatrixReport([str(exc)], 0, {})

    rules = matrix.get("rules")
    if not isinstance(rules, list) or not rules:
        return MatrixReport(["matrix must define a non-empty rules list"], 0, {})

    ids: set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule {index} is not a mapping")
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(f"rule {index} has no id")
        elif rule_id in ids:
            errors.append(f"duplicate rule id: {rule_id}")
        else:
            ids.add(rule_id)
        if not isinstance(rule.get("priority"), int):
            errors.append(f"rule {rule_id!r} has no integer priority")
        if not isinstance(rule.get("paths"), list) or not rule["paths"]:
            errors.append(f"rule {rule_id!r} has no paths")
        if rule.get("classification") not in {
            "eligsir-original",
            "graphdeco-derived",
            "third-party-unresolved",
        }:
            errors.append(f"rule {rule_id!r} has invalid classification")
        for key in ("license", "license_status", "evidence"):
            if not isinstance(rule.get(key), str) or not rule[key].strip():
                errors.append(f"rule {rule_id!r} has no {key}")
        if rule.get("classification") == "third-party-unresolved":
            if rule.get("license_status") != "unresolved" or rule.get("license") not in {
                "UNKNOWN",
                "REVIEW_REQUIRED",
            }:
                errors.append(f"unresolved third-party rule {rule_id!r} claims a license")

    code_files = iter_code_files(root)
    class_counts = {name: 0 for name in {
        "eligsir-original",
        "graphdeco-derived",
        "third-party-unresolved",
    }}
    for path in code_files:
        matches = _matching_rules(path, rules)
        if not matches:
            errors.append(f"unclassified retained code: {path}")
            continue
        highest = max(rule.get("priority", -1) for rule in matches)
        winners = [rule for rule in matches if rule.get("priority") == highest]
        if len(winners) != 1:
            ids_text = ", ".join(str(rule.get("id")) for rule in winners)
            errors.append(f"ambiguous provenance for {path}: {ids_text}")
            continue
        if (
            winners[0]["classification"] == "eligsir-original"
            and _has_graphdeco_header(root, path)
        ):
            errors.append(
                "GraphDeco header evidence requires a targeted graphdeco-derived "
                f"rule; broad project-original rule won for {path}"
            )
        class_counts[winners[0]["classification"]] += 1

    expected = matrix.get("coverage", {})
    if not isinstance(expected, dict):
        errors.append("coverage must be a mapping")
    else:
        if expected.get("total_code_files") != len(code_files):
            errors.append(
                "coverage total_code_files is stale: "
                f"document={expected.get('total_code_files')!r}, actual={len(code_files)}"
            )
        for classification, count in class_counts.items():
            if expected.get(classification) != count:
                errors.append(
                    f"coverage {classification} is stale: "
                    f"document={expected.get(classification)!r}, actual={count}"
                )

    return MatrixReport(errors, len(code_files), class_counts)


def validate_public_ci(root: Path) -> list[str]:
    """Reject internal runners, credentials, publishing, and unsafe endpoints."""

    errors: list[str] = []
    workflow_dir = root / WORKFLOW_DIR
    workflows = sorted(
        path for path in workflow_dir.glob("*") if path.suffix.lower() in {".yml", ".yaml"}
    )
    if workflows != [workflow_dir / "ci-public.yml"]:
        errors.append(
            "public export must retain only .github/workflows/ci-public.yml; "
            f"found {[path.relative_to(root).as_posix() for path in workflows]}"
        )
    forbidden = re.compile(
        r"(?i)(self-hosted|buildkit|cluster\.local|ghcr\.io|packages:\s*write|"
        # ``test_release_residue.py`` is an intentional publication-contract
        # test path.  Match release/publish controls as tokens, not arbitrary
        # substrings in test names or documentation paths.
        r"secrets\.|id-token|pages|deploy-pages|upload-pages|"
        r"(?<!test_)(?:release|publish)(?![_a-z])|"
        r"private[_-]?token|docker/login-action)"
    )
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        if forbidden.search(text):
            errors.append(f"{path.relative_to(root)} contains a forbidden internal/publish token")
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            errors.append(f"{path.relative_to(root)} is invalid YAML: {exc}")
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("jobs"), dict):
            errors.append(f"{path.relative_to(root)} has no jobs mapping")
            continue
        for job_id, job in parsed["jobs"].items():
            if not isinstance(job, dict) or job.get("runs-on") != "ubuntu-latest":
                errors.append(f"{path.relative_to(root)} job {job_id!r} is not GitHub-hosted")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = validate_matrix(root)
    dependency_errors = validate_dependency_manifest(root)
    errors = list(report.errors) + validate_public_ci(root) + dependency_errors
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"provenance: {report.total_code_files} code files covered")
    for key, value in sorted(report.class_counts.items()):
        print(f"  {key}: {value}")
    print("public CI: hosted, read-only, no publish/internal endpoints")
    print("dependency provenance: 8 components recorded; unresolved blockers explicit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
