"""Contract tests for creating a new source snapshot without inherited Git history."""
from pathlib import Path
import subprocess
import tarfile

from scripts.export_public_snapshot import (
    export_archive,
    initialize_new_repository,
    validated_members,
)


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def test_export_source_cannot_inherit_developer_commit_history(tmp_path: Path) -> None:
    repo = tmp_path / "private"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Personal Developer")
    _git(repo, "config", "user.email", "developer@example.invalid")
    (repo / "demo.txt").write_text("public example", encoding="utf-8")
    _git(repo, "add", "demo.txt")
    _git(repo, "commit", "-qm", "confidential development history")
    (repo / "demo.txt").write_text("source snapshot", encoding="utf-8")
    _git(repo, "commit", "-qam", "more internal history")

    package = tmp_path / "snapshot.tar.gz"
    count, digest = export_archive(repo, package)
    assert count == 1
    assert len(digest) == 64

    clean = tmp_path / "fresh"
    initialize_new_repository(package, clean)
    assert subprocess.check_output(
        ["git", "-C", str(clean), "rev-list", "--count", "--all"]
    ).strip() == b"1"
    assert (clean / "demo.txt").read_text(encoding="utf-8") == "source snapshot"
    assert subprocess.check_output(
        ["git", "-C", str(clean), "log", "-1", "--format=%an"]
    ).strip() == b"EliGSiR Authors"
    assert not (clean / "demo.txt").is_symlink()


def test_archive_rejects_traversal(tmp_path: Path) -> None:
    import io
    import pytest

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        bad = tarfile.TarInfo("../escape.txt")
        data = b"not allowed"
        bad.size = len(data)
        archive.addfile(bad, io.BytesIO(data))
    with pytest.raises(ValueError, match="Unsafe"):
        validated_members(payload.getvalue())
