#!/usr/bin/env python3
"""Import-fence check for the ``eligsir.*`` package boundary.

Wave 2 Task 8 (2026-08-05): now that the consolidation waves have moved
real logic into ``eligsir/*`` (optimization engine, mapping
coordinator/stores, io sources, ...), the repo-root packages ``contracts/``,
``slam/``, and ``viz/`` have become thin **compatibility shims** that just
re-export the canonical ``eligsir.contracts`` / ``eligsir.slam`` /
``eligsir.viewers`` modules for old call sites (see e.g.
``contracts/__init__.py``: "Compatibility package for canonical EliGSiR
contracts", or ``contracts/types.py``: "Compatibility wrapper for canonical
contract data types").

Rule enforced: **no file under ``eligsir/`` may import from the
repo-root ``contracts``, ``slam``, or ``viz`` packages.** New code lives in
``eligsir/*`` and should import its sibling ``eligsir.*`` modules
directly; importing back through the compat shim would be a pointless
(and confusing) indirection loop -- exactly the kind of "boilerplate that
adds indirection without adding ownership" this project's consolidation
work has repeatedly removed elsewhere (see e.g.
``eligsir/mapping/coordinator.py``'s "Scope correction 2" docstring
for the same anti-pattern already found and fixed once in this codebase).

This does NOT check the reverse direction (root-level ``train.py``,
``utils/*.py``, etc. importing ``contracts``/``slam``/``viz`` or
``eligsir.*``) -- both directions are expected and fine outside
``eligsir/`` today; only the specific loop-back described above is a
violation. Widening scope is future work, not done here (avoids scope
creep: this check should not be the vehicle for auditing every existing
repo-root/``eligsir`` import edge, only the one loop-back pattern the
Wave 2 plan calls out).

Why AST, not ``grep``: a plain text search for "``from contracts import``"
also matches the string inside a *docstring* (e.g.
``eligsir/contracts/slam_backend.py``'s module docstring shows
``from contracts.slam_backend import get_backend`` as a usage example --
not a real import). Parsing each file's AST and only inspecting real
``Import``/``ImportFrom`` nodes avoids that false positive.

Why stdlib-only, not ``import-linter``/``ruff``: neither is installed in
this project's runtime container (``docker compose run --rm train``) or
importable there today, and adding a new dependency for a single
directional rule is more machinery than the rule needs. This matches the
project's existing "stdlib-only tooling" convention (see ``ablator``'s own
README) rather than introducing a new dependency for one check.

Usage::

    python3 scripts/check_import_fences.py           # check the whole eligsir/ tree
    python3 scripts/check_import_fences.py --root eligsir/mapping  # narrower scope

Exit code 0 on a clean tree, 1 if any violation is found (violations are
printed one per line: ``<file>:<lineno>: import of '<module>' ...``).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# The repo-root compat-shim packages that eligsir/* must not import.
BANNED_ROOT_PACKAGES = ("contracts", "slam", "viz")

# Specific compat-shim SUBMODULES (not whole top-level packages -- `utils`
# broadly is fine and widely/legitimately imported from eligsir/*) that
# eligsir/* must not import. Added 2026-08-06 (final review, PR #37):
# utils.streaming_frames is now a pure re-export shim over
# eligsir.io.sources.datasets.* -- eligsir/* code importing back
# through it (rather than the canonical module directly) is the same
# loop-back anti-pattern BANNED_ROOT_PACKAGES exists to prevent, just at
# submodule granularity.
BANNED_SUBMODULES = ("utils.streaming_frames",)

# Explicit allowlist for legitimate exceptions: (relative_file_path, banned_module)
# pairs that are intentional and should not be flagged. Add an entry here
# (with a comment explaining why) if a real, deliberate compat-shim import is
# ever needed from inside eligsir/*.
ALLOWLIST: set[tuple[str, str]] = {
    # registry.py's make_frame_source() still constructs several
    # not-yet-migrated frame-source classes (RGBDSequenceFrameSource,
    # TUMFrameSource, ScanNetFrameSource, HyperSimFrameSource,
    # ReplicaFrameSource, OrbbecExportFrameSource, OrbbecRosBagFrameSource,
    # plus the ROS1 bag-profile helpers) whose REAL implementation still
    # lives in utils/streaming_frames.py -- that module is not yet a pure
    # compat shim for these classes (only for the already-migrated
    # ROS2LiveStreamingFrameSource/RealsenseRosBagFrameSource, which
    # registry.py imports directly from their canonical
    # eligsir.io.sources.datasets.* location instead, per the comment
    # at that import site). Finishing this migration for the remaining
    # classes is future Wave 2 io/sources-split work, not done in PR #37.
    ("eligsir/io/sources/registry.py", "utils.streaming_frames"),
}


def _imported_root_package(module: str) -> str | None:
    """Return the banned root package name if `module` (a dotted import
    target, e.g. "contracts.types" or "viz") is that package or a submodule
    of it; otherwise None.
    """
    top = module.split(".", 1)[0]
    if top in BANNED_ROOT_PACKAGES:
        return top
    for banned in BANNED_SUBMODULES:
        if module == banned or module.startswith(banned + "."):
            return banned
    return None


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, banned_module) violations in `path`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"[check_import_fences] warning: could not parse {path}: {e}", file=sys.stderr)
        return []

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                banned = _imported_root_package(alias.name)
                if banned:
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` / `from .. import y` (level > 0) are
            # relative imports and can never resolve to a repo-root
            # package -- only level == 0 (absolute) imports are in scope.
            if node.level == 0 and node.module:
                banned = _imported_root_package(node.module)
                if banned:
                    violations.append((node.lineno, node.module))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root", default="eligsir",
        help="Directory tree to check (default: eligsir)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scan_root = (repo_root / args.root).resolve()
    if not scan_root.is_dir():
        print(f"[check_import_fences] error: {scan_root} is not a directory", file=sys.stderr)
        return 2

    n_checked = 0
    n_violations = 0
    for path in sorted(scan_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        n_checked += 1
        rel = path.relative_to(repo_root).as_posix()
        for lineno, module in check_file(path):
            if (rel, module) in ALLOWLIST:
                continue
            n_violations += 1
            print(
                f"{rel}:{lineno}: import of '{module}' -- eligsir/* code must import "
                f"the canonical eligsir.* module directly, not the repo-root "
                f"'{module.split('.', 1)[0]}' compat shim (see this script's module docstring)."
            )

    if n_violations:
        print(
            f"\n[check_import_fences] FAILED: {n_violations} violation(s) in "
            f"{n_checked} files under {args.root}/",
            file=sys.stderr,
        )
        return 1

    print(f"[check_import_fences] OK: {n_checked} files under {args.root}/ clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
