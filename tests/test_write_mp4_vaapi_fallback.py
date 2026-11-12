"""Regression test: if VAAPI encoding fails (non-zero return code), _write_mp4
falls through to the existing CPU libx264 path and still succeeds, using the
already-materialized frame list rather than re-rendering (frames_iter is a
one-shot generator in real callers)."""
from unittest import mock

import numpy as np

from eligsir.utils.comparison_report import _write_mp4


def _frames(h=10, w=10, n=2):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_vaapi_failure_falls_back_to_cpu_and_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: p == "/dev/dri/renderD128")

    calls = {"n": 0}

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            calls["n"] += 1
            self._cmd = cmd
            self.stdin = mock.MagicMock()
            # First call (VAAPI) fails; second call (CPU fallback) succeeds.
            self.returncode = 1 if "-vaapi_device" in cmd else 0

        def communicate(self):
            return b"", b"vaapi driver init failed"

        def wait(self):
            return self.returncode

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    out = tmp_path / "fallback.mp4"
    err = _write_mp4(str(out), _frames(), fps=30)
    assert err is None
    assert calls["n"] == 2  # one VAAPI attempt, one CPU fallback attempt


def test_vaapi_popen_raises_falls_back_to_cpu_and_succeeds(tmp_path, monkeypatch):
    """If subprocess.Popen() itself raises (e.g. FileNotFoundError/OSError) on
    the VAAPI-path invocation -- not just a non-zero return code -- _write_mp4
    must still catch it (inside _write_mp4_vaapi) and fall back to the CPU
    path rather than letting the exception escape the whole call."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: p == "/dev/dri/renderD128")

    calls = {"n": 0}

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            calls["n"] += 1
            self._cmd = cmd
            self.stdin = mock.MagicMock()
            self.returncode = 0

        def communicate(self):
            return b"", b""

        def wait(self):
            return self.returncode

    def _fake_popen(cmd, **kwargs):
        if "-vaapi_device" in cmd:
            raise FileNotFoundError("ffmpeg not found")
        return _FakeProc(cmd, **kwargs)

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    out = tmp_path / "popen_raise_fallback.mp4"
    err = _write_mp4(str(out), _frames(), fps=30)
    assert err is None
    assert calls["n"] == 1  # only the CPU fallback actually constructed a proc


class _RealisticFakeStdin:
    """Mimics real subprocess.Popen.stdin's close/flush semantics -- unlike
    unittest.mock.MagicMock, .flush() actually raises ValueError once closed,
    matching CPython's real pipe object. This is what catches a regression of
    the exact bug fixed here (calling .close() before .communicate(), which
    internally calls .flush() on stdin as part of its real implementation)."""

    def __init__(self):
        self.closed = False

    def write(self, data):
        if self.closed:
            raise ValueError("write to closed file")

    def flush(self):
        if self.closed:
            raise ValueError("flush of closed file")

    def close(self):
        self.closed = True


def test_vaapi_success_does_not_flush_already_closed_stdin(tmp_path, monkeypatch):
    """Regression test for a real bug found on a live GPU run: _write_mp4_vaapi
    used to call proc.stdin.close() immediately before proc.communicate() --
    but communicate() internally flushes stdin as part of its real
    implementation, and flushing an already-closed stream raises
    "ValueError: flush of closed file". This only ever manifested once VAAPI
    encoding actually succeeded (a prior environment issue meant VAAPI always
    failed before reaching this code path, masking the bug for the entire
    session it was introduced in)."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: p == "/dev/dri/renderD128")

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            self._cmd = cmd
            self.stdin = _RealisticFakeStdin()
            self.returncode = 0

        def communicate(self):
            # Real subprocess.Popen.communicate() flushes stdin internally --
            # this must not raise even though _write_mp4_vaapi may have
            # already called .close() on this same stdin object.
            self.stdin.flush()
            return b"", b""

        def wait(self):
            return self.returncode

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    out = tmp_path / "vaapi_success.mp4"
    err = _write_mp4(str(out), _frames(), fps=30)
    assert err is None
