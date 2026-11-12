#!/usr/bin/env python3
"""Audit repository for suspicious image/depth conversion/transformation sites.

Issue #645 (parent #638): enforce lossless dataflow for score-bearing paths.
This script scans for sites that perform quantization, resizing, codec operations,
or other transformations on images/depth that might silently corrupt score paths.

Detects:
  - .astype(np.uint8) / .astype(np.uint16) / .astype(np.float32)
  - .convert("RGB") / channel slicing / channel permutation
  - cv2.resize / PIL.Image.resize / torch interpolation
  - torch.clamp / np.nan_to_num / np.clip on score-bearing tensors
  - cv2.imwrite / Image.save with non-lossless formats
  - torchvision.utils.save_image
  - VideoWriter / imageio / H.264 / ffmpeg calls
  - JPEG encoders with quality settings

Each site is classified as:
  - score_path: bit-exact representation handling (channel swaps, HWC↔CHW, etc.)
  - explicit_transform: declared score-domain transformation (resize, JPEG, etc.)
  - visualization: visualization-only, structurally prevented from metric use
  - unknown: unclassifiable, must be reviewed

Allowlist file mechanism allows recording known-good sites with a reason.
New unclassified sites cause a nonzero exit.

Usage:
    python3 scripts/audit_image_conversion_sites.py            # human report
    python3 scripts/audit_image_conversion_sites.py --json      # machine-readable
    python3 scripts/audit_image_conversion_sites.py --allowlist <file>  # with allowlist

Allowlist format (JSON):
    {
      "sites": {
        "path/to/file.py:123": {
          "pattern": "astype(np.uint8)",
          "classification": "score_path",
          "reason": "verified bit-exact RGB8 quantization in decoder"
        },
        ...
      }
    }
"""
from __future__ import annotations

import argparse as _argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# This is the complete, deliberately small static surface that can turn an
# image conversion into a scientific score.  Training call sites are covered by
# score-contract reachability tests; scanning their unrelated tensor math would
# make this review gate noisy instead of protective.
_SCORE_BOUNDARY_PATHS = frozenset({
    "metrics.py",
    "render.py",
    "utils/comparison_report.py",
    "utils/image_provenance.py",
    "utils/image_utils.py",
    "utils/psnr_history.py",
    "utils/score_contract.py",
})
_DEFAULT_ALLOWLIST = _REPO_ROOT / "scripts" / "score_boundary_conversion_allowlist.json"


class Classification(str, Enum):
    """Score-impact classification (ordered by specificity/certainty)."""
    SCORE_PATH = "score_path"           # bit-exact representation handling
    EXPLICIT_TRANSFORM = "explicit_transform"  # declared score transform
    VISUALIZATION = "visualization"    # visualization-only
    UNKNOWN = "unknown"                # unclassifiable


@dataclass(frozen=True)
class ConversionSite:
    path: str  # repo-relative, POSIX separators
    lineno: int
    pattern: str  # matched pattern (e.g. "astype(np.uint8)")
    context: str  # one-line code context
    classification: Classification


# Patterns to detect: (regex, pattern_name)
# Each pattern is a (regex_str, canonical_name) where regex targets the pattern
# and canonical_name is how we name it in output.
_PATTERNS = [
    # astype conversions
    (r'\.astype\s*\(\s*np\.uint8\s*\)', "astype(np.uint8)"),
    (r'\.astype\s*\(\s*np\.uint16\s*\)', "astype(np.uint16)"),
    (r'\.astype\s*\(\s*np\.float32\s*\)', "astype(np.float32)"),
    (r'\.astype\s*\(\s*np\.float64\s*\)', "astype(np.float64)"),
    (r'\.astype\s*\(\s*"?uint8"?\s*\)', "astype(uint8)"),
    (r'\.astype\s*\(\s*"?uint16"?\s*\)', "astype(uint16)"),
    (r'\.astype\s*\(\s*"?float32"?\s*\)', "astype(float32)"),
    (r'\.astype\s*\(\s*int\s*\)', "astype(int)"),

    # PIL image mode conversion
    (r'\.convert\s*\(\s*["\']RGB["\']\s*\)', '.convert("RGB")'),
    (r'\.convert\s*\(\s*["\']BGR["\']\s*\)', '.convert("BGR")'),
    (r'\.convert\s*\(\s*["\']RGBA["\']\s*\)', '.convert("RGBA")'),
    (r'\.convert\s*\(\s*["\']L["\']\s*\)', '.convert("L")'),

    # Image resizing
    (r'\.resize\s*\(', '.resize('),  # PIL Image.resize
    (r'cv2\.resize\s*\(', 'cv2.resize('),
    (r'torch\.nn\.functional\.interpolate\s*\(', 'F.interpolate('),
    (r'F\.interpolate\s*\(', 'F.interpolate('),
    (r'torchvision\.transforms\.Resize\s*\(', 'torchvision.transforms.Resize('),

    # Image/tensor save operations
    (r'\.save\s*\(\s*[^)]*\)', '.save('),  # PIL Image.save
    (r'cv2\.imwrite\s*\(', 'cv2.imwrite('),
    (r'torchvision\.utils\.save_image\s*\(', 'torchvision.utils.save_image('),
    (r'torch\.save\s*\(', 'torch.save('),
    (r'imageio\.imwrite\s*\(', 'imageio.imwrite('),

    # Clipping/clamping operations
    (r'\.clamp\s*\(', '.clamp('),
    (r'torch\.clamp\s*\(', 'torch.clamp('),
    (r'np\.clip\s*\(', 'np.clip('),
    (r'np\.nan_to_num\s*\(', 'np.nan_to_num('),

    # Video/codec operations
    (r'cv2\.VideoWriter\s*\(', 'cv2.VideoWriter('),
    (r'imageio\.get_writer\s*\(', 'imageio.get_writer('),
    (r'imageio\.mimsave\s*\(', 'imageio.mimsave('),
    (r'ffmpeg|H\.264|h\.264', 'ffmpeg/H.264'),

    # JPEG/compression quality settings
    (r'quality\s*=\s*\d+', 'quality=<int>'),
    (r'\.jpg["\']|\.jpeg["\']', '.jpg/.jpeg'),
    (r'PIL\.Image\.open.*\.jpg', 'PIL.Image.open(...jpg)'),

    # Channel operations
    (r'\[\.\.\.?\s*,\s*:3\s*\]', '[..., :3]'),  # RGB slice
    (r'\[\.\.\.?\s*,\s*:4\s*\]', '[..., :4]'),  # RGBA slice
    (r'\.permute\s*\(', '.permute('),
    (r'\.transpose\s*\(', '.transpose('),
    (r'torch\.einsum.*CHW|torch\.einsum.*HWC', 'einsum(CHW/HWC)'),
]


class _ConversionVisitor(ast.NodeVisitor):
    """AST visitor to find conversion-pattern call sites."""

    def __init__(self, rel_path: str, lines: list[str]):
        self.rel_path = rel_path
        self.lines = lines
        self.sites: list[ConversionSite] = []
        self._seen: set[tuple[int, str]] = set()

    def generic_visit(self, node: ast.AST) -> None:
        """Scan every node's source for conversion patterns."""
        super().generic_visit(node)

        # Skip synthetic nodes without line numbers
        if not hasattr(node, "lineno"):
            return

        lineno = node.lineno
        if lineno < 1 or lineno > len(self.lines):
            return

        line = self.lines[lineno - 1]

        # Check each pattern against this line
        for pattern_re, pattern_name in _PATTERNS:
            if re.search(pattern_re, line):
                key = (lineno, pattern_name)
                if key in self._seen:
                    continue
                self._seen.add(key)
                # Extract a ~60-char context window around the match
                match = re.search(pattern_re, line)
                start = max(0, match.start() - 20)
                end = min(len(line), match.end() + 20)
                context = line[start:end].strip()

                # Determine initial classification based on file/pattern heuristics
                classification = _classify_site(self.rel_path, pattern_name, context)

                site = ConversionSite(
                    path=self.rel_path,
                    lineno=lineno,
                    pattern=pattern_name,
                    context=context,
                    classification=classification,
                )
                self.sites.append(site)


def _classify_site(path: str, pattern: str, context: str) -> Classification:
    """Heuristic classification of a conversion site.

    Starts with unknown (bottom of the lattice) and escalates only when
    there is strong evidence. Filename heuristics are permissible as evidence
    but are never authoritative for a site in core utilities.
    """
    # Visualization-only paths: report/, viz/, contact_sheet, etc.
    if any(x in path for x in ["report", "viz/", "render_report", "contact_sheet", "video", "trajectory"]):
        return Classification.VISUALIZATION

    # Score_path heuristic: channel operations, dtypes in known-good contexts
    # (e.g., depth_to_uint16_png, RGB8->float32 decoders, CHW↔HWC in camera setup)
    if any(x in path for x in ["decoder", "decode", "camera_utils"]):
        if any(x in pattern for x in ["uint8", "uint16", "float32", "permute", "transpose"]):
            return Classification.SCORE_PATH

    # Explicit_transform heuristic: size/resize in mappers, schedulers, or depth insertion
    if any(x in path for x in ["mapper", "scheduler", "insertion", "stream"]):
        if "resize" in pattern or "interpolate" in pattern:
            return Classification.EXPLICIT_TRANSFORM

    # core metric/evaluation code paths need closer inspection
    if any(x in path for x in ["metrics", "general_utils", "comparison_report"]):
        return Classification.UNKNOWN

    # Default to unknown for anything we can't classify heuristically
    return Classification.UNKNOWN


def find_all_sites(
    include_tests: bool = False,
    *,
    scope: str = "repo",
) -> list[ConversionSite]:
    """Scan git-tracked Python files for conversion sites.

    Uses `git ls-files '*.py'` to get the authoritative set of tracked files,
    which is reproducible across machines and CI (excludes .venv, site-packages,
    and other untracked/ignored dirs automatically).

    Falls back to rglob with explicit exclusions if git is unavailable.

    Always excludes: .git/, .claude/, .github/, submodules/, ablator/
    (vendored/third-party). Optionally excludes tests/.
    """
    import subprocess

    sites: list[ConversionSite] = []

    # Try to get git-tracked files via git ls-files (most reliable, reproducible)
    tracked_files: list[Path] = []
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            tracked_files = [
                _REPO_ROOT / line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: rglob with explicit exclusion of third-party dirs
    if not tracked_files:
        tracked_files = [
            f for f in sorted(_REPO_ROOT.rglob("*.py"))
            if not any(
                part in str(f)
                for part in [".venv", "site-packages", ".tox", "__pycache__"]
            )
        ]

    for f in sorted(tracked_files):
        rel = f.relative_to(_REPO_ROOT).as_posix()

        # Skip vendored/third-party/config dirs
        if any(
            part in rel
            for part in [".git", ".claude", ".github", "submodules", "ablator"]
        ):
            continue

        # Optionally skip tests
        if not include_tests and (rel.startswith("tests/") or "/tests/" in rel):
            continue
        if scope == "score-boundary" and rel not in _SCORE_BOUNDARY_PATHS:
            continue

        try:
            text = f.read_text()
            lines = text.split("\n")
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            continue

        v = _ConversionVisitor(rel, lines)
        v.visit(tree)
        sites.extend(v.sites)

    return sites


def load_allowlist(allowlist_path: Optional[str]) -> dict:
    """Load allowlist from JSON file."""
    if not allowlist_path:
        return {}

    path = Path(allowlist_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
        return data.get("sites", {})
    except Exception as e:
        print(f"Warning: failed to load allowlist {allowlist_path}: {e}", file=sys.stderr)
        return {}


def site_key(site: ConversionSite) -> str:
    """Generate a stable key for a site (path:line)."""
    return f"{site.path}:{site.lineno}"


def main() -> int:
    p = _argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument(
        "--scope",
        choices=("score-boundary", "repo"),
        default="score-boundary",
        help="tracked source surface to audit (default: score-boundary)",
    )
    p.add_argument(
        "--allowlist",
        metavar="FILE",
        default=str(_DEFAULT_ALLOWLIST),
        help=(
            "JSON allowlist of reviewed sites (path:lineno -> {classification, reason}); "
            "defaults to the score-boundary allowlist"
        ),
    )
    ns = p.parse_args()

    sites = find_all_sites(include_tests=False, scope=ns.scope)
    allowlist = load_allowlist(ns.allowlist)

    # Partition sites
    allowlisted = []
    unclassified = []
    for site in sites:
        key = site_key(site)
        if key in allowlist:
            allowlisted.append((site, allowlist[key]))
        elif site.classification == Classification.UNKNOWN:
            unclassified.append(site)

    # Count by classification (for the unallowlisted sites)
    counts = {c: 0 for c in Classification}
    for site in sites:
        key = site_key(site)
        if key not in allowlist:
            counts[site.classification] += 1

    if ns.json:
        # Machine-readable output
        output = {
            "scope": ns.scope,
            "total_sites": len(sites),
            "classified_breakdown": {c.value: counts[c] for c in Classification},
            "unclassified_count": len(unclassified),
            "allowlisted_count": len(allowlisted),
            "sites": [
                {
                    "path": site.path,
                    "line": site.lineno,
                    "pattern": site.pattern,
                    "context": site.context,
                    "classification": site.classification.value,
                    "allowlisted": site_key(site) in allowlist,
                }
                for site in sorted(sites, key=lambda s: (s.path, s.lineno))
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable report
        print(f"Image/depth conversion audit: {len(sites)} sites scanned\n")
        print(f"Classification breakdown:")
        print(f"  score_path:         {counts[Classification.SCORE_PATH]:4d}")
        print(f"  explicit_transform: {counts[Classification.EXPLICIT_TRANSFORM]:4d}")
        print(f"  visualization:      {counts[Classification.VISUALIZATION]:4d}")
        print(f"  unknown (needs review): {counts[Classification.UNKNOWN]:4d}")
        print(f"  allowlisted:        {len(allowlisted):4d}\n")

        if unclassified:
            print(f"=== UNCLASSIFIED sites (must be allowlisted or reviewed): {len(unclassified)} ===\n")
            for site in sorted(unclassified, key=lambda s: (s.path, s.lineno)):
                print(f"  {site.path}:{site.lineno}")
                print(f"    pattern:  {site.pattern}")
                print(f"    context:  {site.context}")
                print()

        if allowlisted:
            print(f"=== Allowlisted sites: {len(allowlisted)} ===\n")
            for site, entry in sorted(allowlisted, key=lambda x: (x[0].path, x[0].lineno)):
                print(f"  {site.path}:{site.lineno} -> {entry.get('classification', 'N/A')}")
                print(f"    reason: {entry.get('reason', 'N/A')}")
                print()

    # JSON is consumed by CI as well as humans: never make unreviewed sites
    # appear green merely because a machine-readable format was requested.
    if unclassified:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
