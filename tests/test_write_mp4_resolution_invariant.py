"""Regression test: whichever encode path _write_mp4 uses (VAAPI or the
CPU libx264 fallback), the output frame dimensions must exactly match the
input frame dimensions (minus the pre-existing even-crop, h - (h % 2)) --
no new resize step may be introduced by either path.

This is the explicit "no quality loss" requirement from the user request
that prompted this fix: the video must render at the same resolution as
the source camera images.
"""
from unittest import mock

import numpy as np

from eligsir.utils.comparison_report import _write_mp4


def _frames(h=101, w=155, n=3):
    # Odd dimensions on purpose: exercises the even-crop path (h-1, w-1).
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_vaapi_path_uses_same_dimensions_as_cpu_path(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: p == "/dev/dri/renderD128")

    captured_cmds = []

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            captured_cmds.append(cmd)
            self.stdin = mock.MagicMock()
            self.returncode = 0

        def communicate(self):
            return b"", b""

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    out = tmp_path / "vaapi.mp4"
    err = _write_mp4(str(out), _frames(), fps=30)
    assert err is None
    assert len(captured_cmds) == 1
    vaapi_cmd = captured_cmds[0]
    assert "-s" in vaapi_cmd
    vaapi_size = vaapi_cmd[vaapi_cmd.index("-s") + 1]
    assert vaapi_size == "154x100"  # w-1 x h-1 for the odd 155x101 input


def test_cpu_fallback_uses_same_dimensions_when_vaapi_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: False)  # no /dev/dri/renderD128

    captured_cmds = []

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            captured_cmds.append(cmd)
            self.stdin = mock.MagicMock()
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    out = tmp_path / "cpu.mp4"
    err = _write_mp4(str(out), _frames(), fps=30)
    assert err is None
    assert len(captured_cmds) == 1
    cpu_cmd = captured_cmds[0]
    assert "-s" in cpu_cmd
    cpu_size = cpu_cmd[cpu_cmd.index("-s") + 1]
    assert cpu_size == "154x100"
