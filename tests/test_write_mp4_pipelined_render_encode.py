"""Task 6a regression tests: `_write_mp4` must overlap rendering (pulling the
next frame from `frames_iter`) with ffmpeg encoding of already-produced
frames, instead of the old `frames = list(frames_iter)` collect-then-encode
behavior that fully materializes every frame before the encoder subprocess
is even started (utils/comparison_report.py:1037, pre-task-6a).

These tests use a mocked `subprocess.Popen` (no real ffmpeg needed) so they
run in the CPU-smoke image too. `tests/test_trajectory_video_encode.py`
covers the real-ffmpeg frame-count parity check for the pipelined path.
"""
from __future__ import annotations

import threading
from unittest import mock

import numpy as np

from eligsir.utils.comparison_report import _write_mp4


def _frame(h=4, w=4):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_encode_starts_before_iterator_is_exhausted(tmp_path, monkeypatch):
    """Proves real pipelining: the generator's second frame is only produced
    AFTER the encoder has already written the first frame. Under the old
    collect-then-encode behavior, `list(frames_iter)` would fully drain this
    generator (yielding both frames) before any `Popen`/write ever happens,
    so `first_frame_written` would never be set while the generator is
    still running and this test would time out and fail.
    """
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: False)  # force CPU path, no VAAPI

    first_frame_written = threading.Event()

    class _FakeStdin:
        def write(self, data):
            first_frame_written.set()

        def close(self):
            pass

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            self.stdin = _FakeStdin()
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    produced = []

    def frames():
        produced.append(1)
        yield _frame()
        assert first_frame_written.wait(timeout=2), (
            "encoder never wrote frame 1 before the generator was asked for "
            "frame 2 -- _write_mp4 collected the whole iterator before "
            "starting the encode (old collect-then-encode behavior)"
        )
        produced.append(2)
        yield _frame()

    out = tmp_path / "pipelined.mp4"
    err = _write_mp4(str(out), frames(), fps=30)
    assert err is None
    assert produced == [1, 2]


def test_pipelined_cpu_path_frame_count_matches_input(tmp_path, monkeypatch):
    """Frame-count parity: every frame the generator yields must reach the
    encoder exactly once, in order, regardless of the pipelined scheduling."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: False)

    written = []

    class _FakeStdin:
        def write(self, data):
            written.append(data)

        def close(self):
            pass

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            self.stdin = _FakeStdin()
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    n = 7

    def frames():
        for i in range(n):
            yield np.full((4, 4, 3), i, dtype=np.uint8)

    out = tmp_path / "count.mp4"
    err = _write_mp4(str(out), frames(), fps=30)
    assert err is None
    assert len(written) == n
    # Order must be preserved even though writes happen on a background thread.
    for i, data in enumerate(written):
        assert data == np.full((4, 4, 3), i, dtype=np.uint8).tobytes()


def test_vaapi_write_failure_midstream_still_renders_every_frame_for_fallback(tmp_path, monkeypatch):
    """If the VAAPI encoder's pipe breaks partway through, the remaining
    frames must still be pulled from the (one-shot) generator so the CPU
    fallback -- which reuses the buffered frames, not the exhausted
    generator -- gets the complete sequence, matching
    tests/test_write_mp4_vaapi_fallback.py's documented contract."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("os.path.exists", lambda p: p == "/dev/dri/renderD128")

    calls = {"n": 0}
    cpu_written = []

    class _BrokenVaapiStdin:
        def write(self, data):
            raise BrokenPipeError("vaapi pipe broke")

        def close(self):
            pass

    class _CpuStdin:
        def write(self, data):
            cpu_written.append(data)

        def close(self):
            pass

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            calls["n"] += 1
            self._cmd = cmd
            self.returncode = 0
            if "-vaapi_device" in cmd:
                self.stdin = _BrokenVaapiStdin()
            else:
                self.stdin = _CpuStdin()

        def communicate(self):
            return b"", b"broken pipe"

        def wait(self):
            return self.returncode

    monkeypatch.setattr("subprocess.Popen", _FakeProc)

    produced = []

    def frames():
        for i in range(5):
            produced.append(i)
            yield np.full((4, 4, 3), i, dtype=np.uint8)

    out = tmp_path / "vaapi_broken.mp4"
    err = _write_mp4(str(out), frames(), fps=30)
    assert err is None
    assert calls["n"] == 2  # one VAAPI attempt (broken), one CPU fallback
    assert produced == [0, 1, 2, 3, 4]  # generator fully drained despite the break
    assert len(cpu_written) == 5  # CPU fallback got every frame from the buffer
