from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.docs_catalog import GENERATED_API_ROOT, GENERATED_INDEX, discover_public_modules, generate_all  # noqa: E402
from scripts.docs_visuals import audit as audit_visuals  # noqa: E402

PRIMARY_CANONICAL_REFERENCE = {
    "docs/reference/contracts.md",
    "docs/reference/slam_adapters.md",
    "docs/reference/viz.md",
    "docs/reference/mapping_scheduler.md",
}
CURRENT_OWNERSHIP_DOCS = {
    "docs/architecture/CURRENT_DEFAULT_METHOD_STATE.md",
    "docs/architecture/OBSERVATION_ADMISSION_AND_RETENTION_OWNERSHIP.md",
    "docs/architecture/LIVE_MAPPING_PHASE_AND_MULTIRES_POLICY.md",
}
COMPAT_PREFIX = re.compile(r"(?m)^:::\s+(?:contracts|slam|viz|utils\.mapping_scheduler)(?:\.|\s|$)")
MKDOCSTRINGS_DIRECTIVE = re.compile(r"(?m)^:::\s+\S+")
FORBIDDEN_DOC_ARTIFACTS = {".DS_Store", "Thumbs.db"}
AGENT_COMPAT_FILES = {"CLAUDE.md", "GEMINI.md"}
MAX_AGENT_COMPAT_LINES = 80

# The publication export intentionally carries only release/provenance notes;
# the source checkout's large MkDocs corpus (including docs/visuals specs) was
# pruned. Keep this explicit rather than allowing the visual generator to fail
# with a misleading FileNotFoundError in a clean publication clone.
PUBLICATION_SCOPE_MARKER = Path("docs/PUBLICATION_SCOPE.md")


def _is_pruned_publication_docs(root: Path) -> bool:
    marker = root / PUBLICATION_SCOPE_MARKER
    if not marker.is_file():
        return False
    text = marker.read_text(encoding="utf-8")
    required = (
        "publication export",
        "docs/visuals/",
        "intentionally pruned",
    )
    return all(token in text for token in required) and not any(
        (root / path).exists()
        for path in ("docs/visuals/diagrams.json", "docs/visuals/plots.json", "docs/visuals/assets.json")
    )

# These modules/paths were physically removed by the ROS hard cut. Stable
# reference landing pages must never point back at them just because a historical
# URL used to exist.
RETIRED_REFERENCE_TOKENS = {
    "eligsir.contracts.ros_interop",
    "eligsir/cli/ros_node.py",
    "eligsir/viewers/ros2_rviz.py",
    "viz/ros2_rviz.py",
}

# Exact stale configuration statements from the former ROS-side semantic
# admission owner. Historical/evidence pages may preserve them, but current
# ownership/default pages may not describe them as active configuration.
RETIRED_CURRENT_OWNERSHIP_TOKENS = {
    "mapping_admission_enabled: true",
    "mapping_admission_mode: hybrid",
}

# The live profile and current-state page intentionally duplicate only a tiny
# set of identity-bearing defaults. If one side changes without the other, make
# the documentation PR fail rather than silently publishing two current states.
CANONICAL_PROFILE_DOC_PAIRS = {
    "streaming_optimization_batch_size: 4": "requested supervision B=4",
    "streaming_frame_admission: preselected": "streaming_frame_admission: preselected",
    "streaming_mapping_resolution_tier: adaptive": "adaptive mapping resolution",
    "streaming_mapping_resolution_min_tier: eighth": "minimum tier EIGHTH",
    "optimizer_type: selective_adam": "SelectiveAdam + sparse gradients",
    "densification_strategy: gsplat_energy_mcmc": "gsplat_energy_mcmc",
    "streaming_lifecycle_enabled: true": "lifecycle labeling",
}

NEXT_ARCHITECTURE_REQUIRED_TOKENS = {
    "2026-08-31-probabilistic-gaussian-belief-and-population-control.md",
    "2026-08-31-probabilistic-gaussian-control-pass3-pass4-audit.md",
    "WRITE OWNERSHIP",
    "DECISION OWNERSHIP",
    "unwired scaffolding",
}

REFINEMENT_PROFILE_COMMENT_TOKENS = {
    "POST_REFINEMENT",
    "`native`",
    "`bounded`",
    "`progressive`",
}


def _pages_has_rest(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^\s*nav\s*:\s*$", text):
        return True
    return bool(re.search(r"(?m)^\s*-\s*\.\.\.\s*$", text))


def audit(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    publication_scope = _is_pruned_publication_docs(root)
    if publication_scope:
        # Only the visual generator/audit is out of scope for this curated
        # export. Continue with every ownership, retired-reference, profile,
        # generated-API, and documentation-structure check below.
        pass
    else:
        generate_all(root)
        visual_errors, _visual_warnings = audit_visuals(root)
        errors.extend(visual_errors)

    mkdocs = (root / "mkdocs.yml").read_text(encoding="utf-8")
    if re.search(r"(?m)^nav\s*:\s*$", mkdocs):
        errors.append("mkdocs.yml must not contain a hand-maintained top-level nav:")
    if "awesome-pages" not in mkdocs:
        errors.append("mkdocs.yml must enable awesome-pages")
    if "scripts/mkdocs_hook.py" not in mkdocs:
        errors.append("mkdocs.yml must run scripts/mkdocs_hook.py")

    for pages in sorted((root / "docs").rglob(".pages")):
        if not _pages_has_rest(pages):
            errors.append(f"{pages.relative_to(root)} has nav: without the required '...' rest entry")

    for path in sorted((root / "docs").rglob("*")):
        if path.is_file() and path.name in FORBIDDEN_DOC_ARTIFACTS:
            errors.append(f"tracked documentation artifact must be removed: {path.relative_to(root)}")

    # Retired API/path references are unsafe in any retained publication page,
    # not only in the canonical landing pages below. Historical source pages
    # may be absent from a curated export, but a retained page must not point a
    # reader at an API that was physically removed.
    for path in sorted((root / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for token in sorted(RETIRED_REFERENCE_TOKENS):
            if token in text:
                errors.append(
                    f"{path.relative_to(root)} references retired current API/path: {token}"
                )

    # Tool-specific instruction files are compatibility entry points only. Keeping
    # them small and anchored on AGENTS.md prevents three independent architecture
    # and runbook copies from drifting apart again.
    for rel in sorted(AGENT_COMPAT_FILES):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "AGENTS.md" not in text:
            errors.append(f"{rel} must point to canonical AGENTS.md")
        line_count = len(text.splitlines())
        if line_count > MAX_AGENT_COMPAT_LINES:
            errors.append(
                f"{rel} has {line_count} lines; compatibility agent files must remain <= {MAX_AGENT_COMPAT_LINES} lines"
            )

    for rel in sorted(PRIMARY_CANONICAL_REFERENCE):
        path = root / rel
        if not path.exists():
            if not publication_scope:
                errors.append(f"missing primary reference page: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if COMPAT_PREFIX.search(text):
            errors.append(f"{rel} still targets a compatibility-wrapper namespace")

    for rel in sorted(CURRENT_OWNERSHIP_DOCS):
        path = root / rel
        if not path.exists():
            if not publication_scope:
                errors.append(f"missing current ownership/default page: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in sorted(RETIRED_CURRENT_OWNERSHIP_TOKENS):
            if token in text:
                errors.append(
                    f"{rel} still describes retired ROS-side semantic-admission config as current: {token}"
                )

    # Ratchet a small set of live-profile/current-page identity claims. This is
    # deliberately not a second YAML parser or full defaults table; it catches
    # the highest-damage class of documentation drift (B/admission/resolution/
    # optimizer/mutation/lifecycle identity) while leaving ordinary tuning
    # values owned solely by the profile.
    profile_path = root / "configs" / "profiles" / "live" / "canonical_mapping.yml"
    current_path = root / "docs" / "architecture" / "CURRENT_DEFAULT_METHOD_STATE.md"
    if not profile_path.exists():
        if not publication_scope:
            errors.append("missing canonical live profile: configs/profiles/live/canonical_mapping.yml")
    if not current_path.exists():
        if not publication_scope:
            errors.append("missing current method snapshot: docs/architecture/CURRENT_DEFAULT_METHOD_STATE.md")
    if profile_path.exists() and current_path.exists():
        profile_text = profile_path.read_text(encoding="utf-8")
        current_text = current_path.read_text(encoding="utf-8")
        for profile_token, doc_token in CANONICAL_PROFILE_DOC_PAIRS.items():
            if profile_token not in profile_text:
                errors.append(
                    "canonical live profile changed an identity-bearing default without "
                    f"updating the docs-integrity contract: missing {profile_token!r}"
                )
            if doc_token not in current_text:
                errors.append(
                    "current method snapshot no longer mirrors canonical live-profile "
                    f"identity: missing {doc_token!r}"
                )
        for token in sorted(REFINEMENT_PROFILE_COMMENT_TOKENS):
            if token not in profile_text:
                errors.append(
                    "canonical live profile must distinguish live mapping tiers from "
                    f"refinement policy; missing explanatory token {token!r}"
                )

    # The stable next-architecture page is intentionally the authority bridge
    # between the original probabilistic design and the later mandatory audit.
    # If either record disappears from that pointer, readers would again see an
    # incomplete definition of the active migration target.
    next_target_path = root / "docs" / "architecture" / "NEXT_ARCHITECTURE_TARGET.md"
    if not next_target_path.exists():
        if not publication_scope:
            errors.append("missing stable next-architecture pointer")
    else:
        next_text = next_target_path.read_text(encoding="utf-8")
        for token in sorted(NEXT_ARCHITECTURE_REQUIRED_TOKENS):
            if token not in next_text:
                errors.append(
                    f"NEXT_ARCHITECTURE_TARGET.md is missing required authority token: {token}"
                )

    # The generated API tree is the single owner of rendered Python objects.
    # Stable topic/compatibility pages remain linkable but must only link into
    # the generated tree; otherwise mkdocs-autorefs sees duplicate primary
    # URLs and strict builds fail nondeterministically as the package grows.
    reference_root = root / "docs" / "reference"
    generated_root = root / GENERATED_API_ROOT
    for path in sorted(reference_root.rglob("*.md")):
        if generated_root in path.parents:
            continue
        if MKDOCSTRINGS_DIRECTIVE.search(path.read_text(encoding="utf-8")):
            errors.append(
                f"{path.relative_to(root)} renders Python objects directly; "
                "only generated docs/reference/api/** pages may contain ::: directives"
            )

    modules = discover_public_modules(root)
    for item in modules:
        output = root / item.output
        if not output.exists():
            if not publication_scope:
                errors.append(f"missing generated API page for {item.module}: {item.output}")
            continue
        expected = f"::: {item.module}"
        if expected not in output.read_text(encoding="utf-8"):
            errors.append(f"generated API page has wrong target for {item.module}")

    if not (root / GENERATED_INDEX).exists():
        if not publication_scope:
            errors.append(f"missing generated documentation index: {GENERATED_INDEX}")
    if not (root / GENERATED_API_ROOT / "index.md").exists():
        if not publication_scope:
            errors.append(f"missing generated API landing page: {GENERATED_API_ROOT}/index.md")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("Documentation integrity check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Documentation integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
