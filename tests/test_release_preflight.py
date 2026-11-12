"""Unit tests for prepublication identity and local Git-history checks."""
from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.audit_release_identity import scan_data, scan_history


def test_scan_data_catches_encoded_private_identity_without_echoing_value() -> None:
    identity = bytes.fromhex("626a6f65726e")
    issues = scan_data(b"file path with " + identity + b" inside")
    assert any("identity" in issue for issue in issues)
    assert all(identity.decode() not in issue for issue in issues)


def test_scan_data_rejects_private_key_and_accepts_neutral_example() -> None:
    key_header = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert any("private key" in issue for issue in scan_data(key_header))
    assert scan_data(b"open public-source checkpoint and ordinary example config") == []


def _run(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_history_audit_distinguishes_neutral_from_personal_commits(tmp_path: Path) -> None:
    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.name", "EliGSiR Authors")
    _run(tmp_path, "config", "user.email", "eligsir-authors@users.noreply.github.com")
    _run(tmp_path, "commit", "--allow-empty", "-m", "neutral", "-q")
    assert scan_history(tmp_path) == []

    _run(tmp_path, "config", "user.name", "Individual Developer")
    _run(tmp_path, "config", "user.email", "someone@example.org")
    _run(tmp_path, "commit", "--allow-empty", "-m", "change", "-q")
    findings = scan_history(tmp_path)
    assert len(findings) == 1
    assert "non-neutral" in findings[0]
    assert "someone@example.org" not in findings[0]
