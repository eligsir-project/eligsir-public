"""Issue #670 follow-up: a successful capture must DISARM a stale warning.

PR #671's `emit_deferred_git_provenance_warning()` clears `_DEFERRED_WARNING`
when it emits, which makes a single capture-then-emit pair safe. It did not
clear on the other edge: a capture that FAILS and is never emitted, followed by
a capture that SUCCEEDS, left the stale message armed -- so the next emit would
announce a provenance degradation that did not happen, in a `train.log` whose
`report.json` carries a perfectly good 40-char commit. That is the opposite of
#670's failure (a silent degradation) and just as misleading to read.

`get_git_metadata()` now clears the retained message on entry, so "armed"
strictly means "the most recent capture degraded".

The existing `test_a_successful_capture_arms_nothing` does not cover this: its
autouse fixture nulls the global before every test, so it only ever exercises
a success from an already-clear starting state.
"""

from __future__ import annotations

import pytest

from eligsir.utils import git_metadata


@pytest.fixture(autouse=True)
def _clear_pending():
    git_metadata._DEFERRED_WARNING = None
    yield
    git_metadata._DEFERRED_WARNING = None


def _fail_with(monkeypatch, exc):
    def _raise(*_a, **_kw):
        raise exc

    monkeypatch.setattr(git_metadata.subprocess, "check_output", _raise)


def _succeed(monkeypatch):
    def _ok(cmd, *_a, **_kw):
        return "" if "status" in cmd else "0123456789abcdef\n"

    monkeypatch.setattr(git_metadata.subprocess, "check_output", _ok)


def test_a_later_successful_capture_disarms_an_unemitted_warning(monkeypatch, capsys):
    _fail_with(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))
    git_metadata.get_git_metadata()
    assert git_metadata._DEFERRED_WARNING is not None

    # Second capture succeeds, with no emit in between.
    _succeed(monkeypatch)
    metadata = git_metadata.get_git_metadata()
    assert metadata["git_commit"] == "0123456789abcdef"

    assert git_metadata._DEFERRED_WARNING is None
    capsys.readouterr()
    assert git_metadata.emit_deferred_git_provenance_warning() is False
    assert capsys.readouterr().out == ""


def test_a_later_failing_capture_replaces_the_retained_message(monkeypatch):
    """Re-arming still works, and reports the CURRENT failure mode."""
    import subprocess

    _fail_with(monkeypatch, FileNotFoundError(2, "No such file or directory", "git"))
    git_metadata.get_git_metadata()
    assert "FileNotFoundError" in git_metadata._DEFERRED_WARNING

    _fail_with(monkeypatch, subprocess.CalledProcessError(128, ["git", "rev-parse"]))
    git_metadata.get_git_metadata()

    assert "CalledProcessError" in git_metadata._DEFERRED_WARNING
    assert "FileNotFoundError" not in git_metadata._DEFERRED_WARNING.split("(", 1)[1].split(")", 1)[0]
