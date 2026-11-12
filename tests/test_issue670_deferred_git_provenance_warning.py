"""Issue #670: the git-provenance degradation warning must reach `train.log`.

`get_git_metadata()` is called before the output tee is installed, so a capture
failure must be retained and re-emitted once logging exists. Call-site ordering
is covered separately by `test_issue670_emitter_call_site_ordering.py`, which
reads entry-point source text directly and therefore remains runnable on the
torch-free CPU CI surface.
"""

import pytest

from eligsir.utils import git_metadata


@pytest.fixture(autouse=True)
def _clear_pending():
    git_metadata._DEFERRED_WARNING = None
    yield
    git_metadata._DEFERRED_WARNING = None


def test_no_warning_pending_is_a_noop(capsys):
    assert git_metadata.emit_deferred_git_provenance_warning() is False
    assert capsys.readouterr().out == ""


def test_pending_warning_is_reemitted_once(capsys):
    git_metadata._DEFERRED_WARNING = "[git-provenance] WARNING: synthetic"

    assert git_metadata.emit_deferred_git_provenance_warning() is True
    assert "[git-provenance] WARNING: synthetic" in capsys.readouterr().out

    # Idempotent: a second call must not double-print.
    assert git_metadata.emit_deferred_git_provenance_warning() is False
    assert capsys.readouterr().out == ""


def test_a_real_capture_failure_arms_the_deferred_warning(monkeypatch, capsys, tmp_path):
    """Drive the real `except` branch, not a hand-set sentinel.

    Disables both #670 fallbacks so provenance really does degrade to "unknown",
    allowing us to test that the degradation warning is retained and re-emitted.
    """
    # Disable env-override fallback
    monkeypatch.delenv(git_metadata._ENV_COMMIT_VAR, raising=False)
    monkeypatch.delenv(git_metadata._ENV_BRANCH_VAR, raising=False)

    # Disable filesystem fallback by running from a path with no .git anywhere above
    empty = tmp_path / "not_a_checkout"
    empty.mkdir()
    monkeypatch.chdir(empty)

    def _boom(*_a, **_kw):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(git_metadata.subprocess, "check_output", _boom)

    metadata = git_metadata.get_git_metadata()
    assert metadata["git_commit"] == "unknown"

    # Pre-tee print still happens (live console visibility is not regressed).
    assert "[git-provenance] WARNING" in capsys.readouterr().out

    # ...and the same message is now available to re-emit post-tee.
    assert git_metadata._DEFERRED_WARNING is not None
    assert "FileNotFoundError" in git_metadata._DEFERRED_WARNING
    assert git_metadata.emit_deferred_git_provenance_warning() is True
    assert "FileNotFoundError" in capsys.readouterr().out


def test_a_successful_capture_arms_nothing(monkeypatch):
    monkeypatch.setattr(
        git_metadata.subprocess, "check_output", lambda *a, **kw: "abc123\n"
    )
    git_metadata.get_git_metadata()
    assert git_metadata._DEFERRED_WARNING is None
