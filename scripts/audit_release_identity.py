"""Conservative local preflight for a source-only EliGSiR release snapshot.

This does not audit GitHub-hosted PR, workflow-actor or organization metadata.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Encoded to keep the source of this scanner from matching its own audit.
FORBIDDEN_HEX = (
    "626a6f65726e",
    "626ac3b6726e",
    "656c6c656e736f686e",
    "756e696c656f62656e",
    "6d6f6e74616e756e6976657273",
    "6370735f",
    "6370732d",
    "6769746875622e636f6d2f626a6f65726e656c6c656e7331",
    "676863722e696f2f626a6f65726e656c6c656e7331",
    "2f686f6d652f",
    "2f6d6e742f",
    "6370732d776b73746e",
    "73706c61746f6772617068",
    "7065796d616e",
    "2f646174612f74756d5f7065796d616e2f",
)
FORBIDDEN_BYTES = tuple(bytes.fromhex(value).lower() for value in FORBIDDEN_HEX)
SECRET_RULES = (
    ("GitHub token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{15,})")),
    ("cloud access key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private key", re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("API token", re.compile(rb"sk-[A-Za-z0-9_-]{32,}")),
)
ANONYMOUS_NAME = "EliGSiR Authors"
ANONYMOUS_EMAIL = "eligsir-authors@users.noreply.github.com"


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            "git " + " ".join(args) + ": " +
            result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def scan_data(data: bytes) -> list[str]:
    """Return categories only: never echo an embedded token in logs."""
    lower = data.lower()
    issues = []
    if any(token in lower for token in FORBIDDEN_BYTES):
        issues.append("private identity, hostname or absolute path")
    for label, pattern in SECRET_RULES:
        if pattern.search(data):
            issues.append("possible " + label)
    return issues


def scan_checkout(root: Path) -> list[str]:
    """Inspect tracked file names and bytes, including binary artifacts."""
    issues = []
    for raw_path in _git(root, "ls-files", "-z").split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        file = root / relative
        if not file.is_file():
            issues.append(relative + ": missing tracked file")
            continue
        data = raw_path + b"\n" + file.read_bytes()
        issues.extend(relative + ": " + category for category in scan_data(data))
    return issues


def scan_history(root: Path, *, all_refs: bool = False) -> list[str]:
    """Check commit author and committer fields on reachable local history."""
    ref = "--all" if all_refs else "HEAD"
    entries = _git(root, "log", ref, "--format=%H%x00%an%x00%ae%x00%cn%x00%ce")
    issues = []
    for line in entries.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        fields = line.split("\0")
        if len(fields) != 5:
            issues.append("unparseable git log metadata")
            continue
        sha, name, email, committer, committer_email = fields
        if (
            name != ANONYMOUS_NAME or email != ANONYMOUS_EMAIL or
            committer != ANONYMOUS_NAME or committer_email != ANONYMOUS_EMAIL
        ):
            issues.append(sha[:12] + ": non-neutral Git author/committer metadata")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-anonymous-history", action="store_true")
    parser.add_argument("--all-refs", action="store_true")
    args = parser.parse_args(argv)
    if args.all_refs and not args.require_anonymous_history:
        parser.error("--all-refs requires --require-anonymous-history")
    try:
        issues = scan_checkout(args.root)
        if args.require_anonymous_history:
            issues.extend(scan_history(args.root, all_refs=args.all_refs))
    except (OSError, RuntimeError) as error:
        print("Release preflight could not complete: " + str(error), file=sys.stderr)
        return 2
    if issues:
        for issue in issues:
            print("BLOCK: " + issue, file=sys.stderr)
        print("Release preflight: FAILED (" + str(len(issues)) + " findings)", file=sys.stderr)
        return 1
    print("Release preflight: tracked-tree checks passed.")
    if args.require_anonymous_history:
        print("Release preflight: local Git author/committer history passed.")
    print("Not covered: GitHub PR/Actions actors, deleted/unreachable Git objects, or arbitrary secrets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
