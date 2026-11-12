"""Tests for issue #784's git metadata failure reason classification.

Issue #784: when git provenance capture fails (e.g. in linked worktrees where
the gitdir is not reachable in-container), the failure reason is now recorded
so downstream readers can distinguish failure modes without re-reading train.log.

This module tests the classification logic in utils.git_metadata._error_reason()
and the end-to-end flow through get_git_metadata() and git_provenance_from_args().
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from eligsir.utils.git_metadata import get_git_metadata, _error_reason
from eligsir.utils.comparison_report import git_provenance_from_args


class TestErrorReason:
    """Tests for _error_reason() classification."""

    def test_error_reason_classifies_missing_git_binary(self):
        """FileNotFoundError is classified as git_binary_missing."""
        exc = FileNotFoundError("git: command not found")
        reason = _error_reason(exc)
        assert "[git_binary_missing]" in reason
        assert "FileNotFoundError" in reason

    def test_error_reason_classifies_safe_directory_refusal(self):
        """CalledProcessError with safe.directory stderr is classified."""
        stderr_text = (
            "fatal: detected dubious ownership in repository at '/workspace'"
            " To add an exception for this directory, call:\n\n"
            "\tgit config --global --add safe.directory /workspace"
        )
        exc = subprocess.CalledProcessError(128, "git rev-parse HEAD", stderr=stderr_text)
        reason = _error_reason(exc)
        assert "[safe_directory_refusal]" in reason
        assert "stderr:" in reason

    def test_error_reason_classifies_not_a_git_repo(self):
        """CalledProcessError with 'not a git repository' stderr is classified."""
        stderr_text = "fatal: not a git repository (or any of the parent directories): .git"
        exc = subprocess.CalledProcessError(128, "git rev-parse HEAD", stderr=stderr_text)
        reason = _error_reason(exc)
        assert "[not_a_git_repo]" in reason

    def test_error_reason_includes_verbatim_stderr(self):
        """Verbatim stderr is always included so exact error text is preserved."""
        stderr_text = "some specific git error message"
        exc = subprocess.CalledProcessError(128, "git", stderr=stderr_text)
        reason = _error_reason(exc)
        assert stderr_text in reason

    def test_error_reason_handles_missing_stderr_attribute(self):
        """Exceptions without stderr attribute degrade gracefully."""
        exc = RuntimeError("something went wrong")
        reason = _error_reason(exc)
        assert "RuntimeError" in reason
        assert "something went wrong" in reason


class TestGetGitMetadataFailures:
    """Tests for get_git_metadata() failure path and reason recording."""

    def test_success_path_omits_the_reason_key_entirely(self):
        """On success the key is ABSENT, not present-and-empty.

        An always-present empty-string field would put a meaningless key into
        every report.json and break every consumer that pins this dict's shape
        (tests/test_git_metadata_degradation_is_loud.py
        ::test_successful_capture_preserves_existing_metadata_shape is exactly
        that contract). #784's key is a failure-path-only annotation, so a
        successful capture must return byte-identical shape to pre-#784.
        """
        metadata = get_git_metadata()
        # This test assumes the actual repo is valid; if it fails, the env is
        # misconfigured, not the code.
        if metadata["git_commit"] != "unknown":
            assert "git_provenance_unavailable_reason" not in metadata
            assert set(metadata) == {
                "git_branch",
                "git_commit",
                "git_dirty",
                "git_dirty_paths",
            }

    def test_missing_git_binary_records_reason(self):
        """When git binary is absent, reason is recorded (classified)."""
        with patch("subprocess.check_output") as mock_check:
            mock_check.side_effect = FileNotFoundError("git: command not found")
            metadata = get_git_metadata()
            assert metadata["git_commit"] == "unknown"
            assert "[git_binary_missing]" in metadata["git_provenance_unavailable_reason"]

    def test_safe_directory_refusal_records_reason(self):
        """When safe.directory blocks access, reason is recorded (classified)."""
        stderr_text = (
            "fatal: detected dubious ownership in repository"
            " To add an exception for this directory, call:\n\n"
            "\tgit config --global --add safe.directory /workspace"
        )
        with patch("subprocess.check_output") as mock_check:
            exc = subprocess.CalledProcessError(128, "git", stderr=stderr_text)
            mock_check.side_effect = exc
            metadata = get_git_metadata()
            assert metadata["git_commit"] == "unknown"
            assert "[safe_directory_refusal]" in metadata["git_provenance_unavailable_reason"]

    def test_linked_worktree_gitdir_unreachable_records_reason(self):
        """When linked-worktree gitdir is unreachable, reason is recorded.

        This test uses a real tmpdir setup: create a .git file pointing to
        /nonexistent, so git's own stderr will match the real error text.
        """
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .git file (not a dir) pointing to a nonexistent gitdir
            git_file = os.path.join(tmpdir, ".git")
            with open(git_file, "w") as f:
                f.write("gitdir: /nonexistent/gitdir\n")

            # Change to that directory and capture get_git_metadata()'s failure
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                metadata = get_git_metadata()
            finally:
                os.chdir(old_cwd)

            assert metadata["git_commit"] == "unknown"
            reason = metadata["git_provenance_unavailable_reason"]
            assert reason, "reason should be non-empty for a real git failure"
            # The exact error text varies by git version, but we can check for
            # general indicators of a gitdir problem.
            assert any(
                phrase in reason.lower()
                for phrase in ["gitdir", "not a git repository", "no such file", "cannot"]
            ), f"Expected gitdir-related error in: {reason}"


class TestEndToEndProvenance:
    """End-to-end tests: get_git_metadata() → setattr(args) → git_provenance_from_args()."""

    def test_failure_reason_flows_through_args_to_report(self):
        """Failure reason flows from get_git_metadata() through args
        to git_provenance_from_args() for report inclusion."""
        with patch("subprocess.check_output") as mock_check:
            error_msg = "git: command not found"
            mock_check.side_effect = FileNotFoundError(error_msg)
            metadata = get_git_metadata()

            # Simulate what train.py does: setattr each metadata field onto args
            args = SimpleNamespace()
            for key, value in metadata.items():
                setattr(args, key, value)

            # Build the report dict
            provenance = git_provenance_from_args(args)
            assert provenance is not None
            assert provenance["git_commit"] == "unknown"
            assert "unavailable_reason" in provenance
            assert "[git_binary_missing]" in provenance["unavailable_reason"]

    def test_success_path_no_reason_in_report(self):
        """On success, unavailable_reason is empty and omitted from report."""
        metadata = get_git_metadata()

        # Skip this test if the actual repo is misconfigured
        if metadata["git_commit"] == "unknown":
            pytest.skip("git not available in test environment")

        args = SimpleNamespace()
        for key, value in metadata.items():
            setattr(args, key, value)

        provenance = git_provenance_from_args(args)
        assert "unavailable_reason" not in provenance
