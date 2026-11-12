from pathlib import Path

from eligsir.utils import git_metadata


SHA = "0123456789abcdef0123456789abcdef01234567"


def test_git_metadata_scopes_safe_directory_to_current_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_check_output(command, *, text, stderr):
        calls.append(command)
        if command[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main\n"
        if command[-2:] == ["rev-parse", "HEAD"]:
            return f"{SHA}\n"
        if command[-3:] == ["status", "--porcelain", "--untracked-files=no"]:
            return " M tracked.py\n"
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(git_metadata.subprocess, "check_output", fake_check_output)

    metadata = git_metadata.get_git_metadata()

    expected_prefix = ["git", "-c", f"safe.directory={Path.cwd().resolve()}"]
    assert len(calls) == 3
    assert all(command[:3] == expected_prefix for command in calls)
    assert metadata == {
        "git_branch": "main",
        "git_commit": SHA,
        "git_dirty": True,
        "git_dirty_paths": ["tracked.py"],
    }
