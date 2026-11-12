"""Create a scanned, Git-history-free EliGSiR source archive.

The archive can be imported into a *new* repository; it never transfers branches,
pull requests, workflow runs, authorship history, or GitHub platform events.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_release_identity import (
    ANONYMOUS_EMAIL,
    ANONYMOUS_NAME,
    scan_data,
    scan_history,
)


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"git {args[0]} failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def validated_members(payload: bytes) -> list[tarfile.TarInfo]:
    """Reject unsafe paths, special files and identifiable/credential-bearing bytes."""
    members = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as tf:
        for member in tf.getmembers():
            relative = PurePosixPath(member.name)
            if (
                not member.name
                or relative.is_absolute()
                or ".." in relative.parts
                or ".git" in relative.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe or unsupported archive member")
            if member.isfile():
                stream = tf.extractfile(member)
                if stream is None:
                    raise ValueError("Missing archive file")
                if scan_data(member.name.encode("utf-8") + b"\n" + stream.read()):
                    raise ValueError("Archive contains a blocked identity/path/credential pattern")
            members.append(member)
    return members


def export_archive(root: Path, destination: Path) -> tuple[int, str]:
    """Export precisely the committed tree, not untracked work or development Git history."""
    if git(root, "diff", "--name-only") or git(root, "diff", "--cached", "--name-only"):
        raise ValueError("Working tree has tracked modifications; commit or discard before exporting")
    payload = git(root, "archive", "--format=tar", "HEAD")
    members = validated_members(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as stream:
            stream.write(payload)
    return sum(member.isfile() for member in members), hashlib.sha256(destination.read_bytes()).hexdigest()


def initialize_new_repository(archive_path: Path, destination: Path) -> str:
    """Materialize source into a new local repository with a single neutral root commit."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Target directory is not empty; refusing to replace an existing repository")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        payload = archive_path.read_bytes()
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            members = validated_members(compressed.read())
        for entry in members:
            location = destination.joinpath(*PurePosixPath(entry.name).parts)
            if entry.isdir():
                location.mkdir(parents=True, exist_ok=True)
            elif entry.isfile():
                location.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(entry)
                if source is None:
                    raise ValueError("Missing source file")
                with location.open("wb") as output:
                    shutil.copyfileobj(source, output)
                location.chmod(0o755 if entry.mode & 0o111 else 0o644)
    git(destination, "init", "-b", "main")
    git(destination, "add", "--all")
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": ANONYMOUS_NAME,
        "GIT_AUTHOR_EMAIL": ANONYMOUS_EMAIL,
        "GIT_COMMITTER_NAME": ANONYMOUS_NAME,
        "GIT_COMMITTER_EMAIL": ANONYMOUS_EMAIL,
    })
    result = subprocess.run(
        ["git", "-C", str(destination), "-c", "commit.gpgsign=false",
         "commit", "-m", "Initial source publication snapshot"],
        check=False, capture_output=True, env=env,
    )
    if result.returncode:
        raise RuntimeError(
            "Could not create clean source commit: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    if scan_history(destination, all_refs=True):
        raise ValueError("Neutral-history verification failed for the fresh repository")
    history_count = len(git(destination, "rev-list", "--all").splitlines())
    if history_count != 1:
        raise ValueError("Fresh repository must contain exactly one commit")
    return git(destination, "rev-parse", "HEAD").decode().strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-repo", type=Path, help="Optional new local Git repository")
    args = parser.parse_args(argv)
    try:
        count, digest = export_archive(args.root, args.output)
        print(f"Scanned and archived {count} tracked files; SHA-256 {digest}")
        if args.init_repo:
            commit = initialize_new_repository(args.output, args.init_repo)
            print(f"Created one-commit clean local snapshot: {commit}")
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        print("Cannot create public source snapshot: " + str(error), file=sys.stderr)
        return 1
    print("GitHub-hosted user/activity metadata and external image assets still require review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
