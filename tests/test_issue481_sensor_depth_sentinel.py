"""Regression tests for issue #481(a): the sensor-depth negative cache
silently latches `False`, denying depth supervision for the rest of a run.

The fix is a two-tiered sentinel:
- A frame carrying no depth payload (neither in-memory `_depth_np`/`_depth_bytes`
  nor on disk via `_streaming_depth_path`) returns None WITHOUT latching, because
  that state can legitimately change later (e.g. async depth decode, or a camera
  rebuilt by `promote_from_cache` before its depth is populated). Re-checking is
  two getattrs + a None test — no I/O, no decode — so the sentinel buys nothing
  here while permanently denying depth supervision to this second case.
- A failure whose cause requires I/O to re-test (missing `depth_path`, a decode
  that raised None or an exception) latches `_sensor_depth_cache = False` so it
  is never retried. Retrying these costs a real stat or full decode on every
  access.

Every no-depth outcome is counted and logged on an escalating cadence
(1/10/100/1000/...) so a camera silently receiving zero depth supervision is
now detectable, not a silent failure that looks identical to a depth-less source.
Pre-#481 the latch was completely silent (no log, no raise).
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# Same self-contained guard as tests/test_issue384_depth_loss_live_guard.py and
# test_issue481_promote_resolution_cache.py: this module needs the real torch
# and the modules under test import it transitively, but a module-scope
# `import torch` would put this whole file in tests/conftest.py's collect_ignore
# list in the torch-free CPU smoke image -- i.e. silently invisible, which is
# exactly what issue #536 exists to prevent.
torch = pytest.importorskip("torch")

from eligsir.mapping.integration import depth_helpers  # noqa: E402


class _FakeCam:
    """Minimal Camera stand-in for testing _get_sensor_depth."""

    def __init__(
        self,
        image_name: str = "test_cam",
        has_streaming_depth_path: bool = False,
        depth_path: str | None = None,
        depth_frame: object | None = None,
    ) -> None:
        self.image_name = image_name
        self._sensor_depth_cache = None  # Not yet loaded
        if has_streaming_depth_path:
            self._streaming_depth_path = depth_path
        self._streaming_frame = depth_frame
        self._streaming_depth_scale = 1.0  # Neutral scale


class _FakeDepthFrame:
    """Minimal frame object with in-memory depth payload."""

    def __init__(self, depth_np: np.ndarray | None = None) -> None:
        self._depth_np = depth_np
        self._depth_bytes = None


@pytest.fixture
def reset_depth_stats():
    """Clear the module-level depth-unavailable stats before each test."""
    yield
    depth_helpers._DEPTH_UNAVAILABLE_STATS.clear()


class TestNoDepthPayloadNoLatch:
    """``no_depth_payload`` returns None without latching the sentinel."""

    def test_no_payload_leaves_cache_none(self, reset_depth_stats):
        """A frame with no in-memory depth and no disk path returns None,
        leaving cache as None (not False), so retry is possible."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        assert cam._sensor_depth_cache is None

        result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is None
        assert cam._sensor_depth_cache is None, "Sentinel must NOT latch to False"

    def test_no_payload_permits_retry_when_depth_appears(self, reset_depth_stats, tmp_path):
        """Calling _get_sensor_depth again after depth is populated retrieves it."""
        # First call: no payload anywhere
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        result1 = depth_helpers._get_sensor_depth(cam, 32, 32)
        assert result1 is None

        # Simulate depth appearing in the frame (e.g. async decode)
        depth_data = np.ones((32, 32), dtype=np.uint16)
        cam._streaming_frame = _FakeDepthFrame(depth_np=depth_data)

        # Second call: depth is now available in-memory
        result2 = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result2 is not None
        assert result2.shape == (1, 32, 32)

    def test_no_payload_returns_none_on_cache_populated_later(self, reset_depth_stats):
        """Even after the first call returns None, if the cache is later
        populated externally (e.g. by bind_resolution_tier), subsequent calls
        use it."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        result1 = depth_helpers._get_sensor_depth(cam, 32, 32)
        assert result1 is None

        # External code (bind_resolution_tier) populates the cache
        cached_depth = torch.ones((1, 32, 32), dtype=torch.float32)
        cam._sensor_depth_cache = cached_depth

        result2 = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result2 is cached_depth


class TestIOFailuresLatch:
    """I/O-related failures latch the False sentinel permanently."""

    def test_missing_depth_path_latches_false(self, reset_depth_stats):
        """A referenced but non-existent depth file latches False."""
        cam = _FakeCam(
            has_streaming_depth_path=True,
            depth_path="/nonexistent/depth.png"
        )
        assert cam._sensor_depth_cache is None

        result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is None
        assert cam._sensor_depth_cache is False, "Missing file must latch False"

    def test_missing_path_prevents_retry(self, reset_depth_stats):
        """Once latched to False for a missing path, subsequent calls return
        None immediately without re-checking."""
        cam = _FakeCam(
            has_streaming_depth_path=True,
            depth_path="/nonexistent/depth.png"
        )

        # First call latches False
        result1 = depth_helpers._get_sensor_depth(cam, 32, 32)
        assert result1 is None
        assert cam._sensor_depth_cache is False

        # Second call short-circuits immediately (no I/O attempt)
        result2 = depth_helpers._get_sensor_depth(cam, 32, 32)
        assert result2 is None
        assert cam._sensor_depth_cache is False

    def test_decode_failure_latches_false(self, reset_depth_stats, tmp_path):
        """A decode exception latches False."""
        depth_file = tmp_path / "depth.png"
        depth_file.write_bytes(b"fake png")

        cam = _FakeCam(
            has_streaming_depth_path=True,
            depth_path=str(depth_file)
        )

        # Patch PIL.Image.open to raise an exception
        with mock.patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = RuntimeError("PNG decode failed")

            result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is None
        assert cam._sensor_depth_cache is False

    def test_depth_conversion_failure_latches_false(self, reset_depth_stats, tmp_path):
        """If depth_to_meters raises an exception, the sentinel latches."""
        depth_file = tmp_path / "depth.png"
        depth_file.write_bytes(b"fake png data")  # Not actually valid

        cam = _FakeCam(
            has_streaming_depth_path=True,
            depth_path=str(depth_file)
        )

        # Mock PIL to return a valid array, but depth_to_meters to raise
        with mock.patch("PIL.Image.open") as mock_open:
            mock_open.return_value = np.zeros((32, 32), dtype=np.uint16)
            with mock.patch("utils.rgbd_frames.depth_to_meters") as mock_depth:
                mock_depth.side_effect = RuntimeError("Conversion failed")
                result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is None
        assert cam._sensor_depth_cache is False


class TestEscalatingLog:
    """The escalating-cadence log fires at 1/10/100/... occurrences."""

    def test_log_fires_on_first_occurrence(self, reset_depth_stats, capsys):
        """The first no-depth outcome prints."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        assert "[sensor-depth] no depth for camera test_cam" in captured.out
        assert "(no_depth_payload)" in captured.out
        assert "not latched (will retry)" in captured.out
        assert "[occurrence #1]" in captured.out

    def test_log_silent_on_occurrences_2_to_9(self, reset_depth_stats, capsys):
        """Occurrences 2–9 do not print."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        for _ in range(9):
            depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        # Should be only one print (at occurrence 1)
        assert captured.out.count("[occurrence #") == 1
        assert "[occurrence #1]" in captured.out

    def test_log_fires_again_at_tenth_occurrence(self, reset_depth_stats, capsys):
        """Occurrence 10 prints again."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        for _ in range(10):
            depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        lines = [l for l in captured.out.split("\n") if "[occurrence #" in l]
        assert len(lines) == 2
        assert "[occurrence #1]" in lines[0]
        assert "[occurrence #10]" in lines[1]

    def test_log_fires_at_100_1000_etc(self, reset_depth_stats, capsys):
        """The escalating pattern continues: 100, 1000, etc."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        for _ in range(100):
            depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        lines = [l for l in captured.out.split("\n") if "[occurrence #" in l]
        assert "[occurrence #100]" in lines[-1]

    def test_log_identifies_latched_vs_non_latched(self, reset_depth_stats, capsys):
        """The log message explicitly identifies whether the sentinel latched."""
        # Non-latched case
        cam1 = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        depth_helpers._get_sensor_depth(cam1, 32, 32)
        captured = capsys.readouterr()
        assert "not latched (will retry)" in captured.out

        # Latched case (reset and try missing path)
        depth_helpers._DEPTH_UNAVAILABLE_STATS.clear()
        cam2 = _FakeCam(
            has_streaming_depth_path=True,
            depth_path="/nonexistent/depth.png"
        )
        depth_helpers._get_sensor_depth(cam2, 32, 32)
        captured = capsys.readouterr()
        assert "LATCHED (this camera will not retry)" in captured.out


class TestStatsTracking:
    """Per-reason statistics are recorded correctly."""

    def test_reasons_are_counted_separately(self, reset_depth_stats):
        """Different reasons increment separate counters."""
        cam1 = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        cam2 = _FakeCam(
            has_streaming_depth_path=True,
            depth_path="/nonexistent/file.png"
        )

        depth_helpers._get_sensor_depth(cam1, 32, 32)
        depth_helpers._get_sensor_depth(cam2, 32, 32)

        assert depth_helpers._DEPTH_UNAVAILABLE_STATS["no_depth_payload"] == 1
        assert depth_helpers._DEPTH_UNAVAILABLE_STATS["depth_path_missing"] == 1

    def test_exception_reasons_are_keyed_by_type(self, reset_depth_stats, tmp_path):
        """Exception failures record their type in the reason key."""
        depth_file = tmp_path / "depth.png"
        depth_file.write_bytes(b"fake png")

        cam = _FakeCam(
            has_streaming_depth_path=True,
            depth_path=str(depth_file)
        )

        with mock.patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = ValueError("Mock error")
            depth_helpers._get_sensor_depth(cam, 32, 32)

        assert "exception:ValueError" in depth_helpers._DEPTH_UNAVAILABLE_STATS


class TestCameraNameFallback:
    """Camera identification in logs uses image_name or uid fallback."""

    def test_log_uses_image_name_when_present(self, reset_depth_stats, capsys):
        """If image_name exists, it appears in the log."""
        cam = _FakeCam(image_name="frame_12345", has_streaming_depth_path=True, depth_path=None)
        depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        assert "frame_12345" in captured.out

    def test_log_uses_uid_fallback(self, reset_depth_stats, capsys):
        """If image_name is absent, uid is used as fallback."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        del cam.image_name
        cam.uid = "camera_uuid_abc"

        depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        assert "camera_uuid_abc" in captured.out

    def test_log_uses_question_mark_if_no_name(self, reset_depth_stats, capsys):
        """If both image_name and uid are missing, use '?'."""
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        del cam.image_name
        # uid is not set

        depth_helpers._get_sensor_depth(cam, 32, 32)

        captured = capsys.readouterr()
        assert "no depth for camera ?" in captured.out


class TestNonStreamingCameras:
    """Non-streaming cameras (no _streaming_depth_path) return None immediately."""

    def test_non_streaming_returns_none(self, reset_depth_stats):
        """A camera without _streaming_depth_path returns None and does not log."""
        cam = _FakeCam(has_streaming_depth_path=False)

        result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is None
        assert len(depth_helpers._DEPTH_UNAVAILABLE_STATS) == 0


class TestCachedDepthReused:
    """A previously-cached depth tensor is returned directly (with resizing)."""

    def test_cached_depth_returned_as_is(self, reset_depth_stats):
        """If cam._sensor_depth_cache is already a tensor, it is returned."""
        cached_tensor = torch.ones((1, 32, 32), dtype=torch.float32)
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        cam._sensor_depth_cache = cached_tensor

        result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result is cached_tensor

    def test_cached_depth_is_resized_if_needed(self, reset_depth_stats):
        """Cached depth at wrong resolution is resized via interpolation."""
        cached_tensor = torch.ones((1, 16, 16), dtype=torch.float32)
        cam = _FakeCam(has_streaming_depth_path=True, depth_path=None)
        cam._sensor_depth_cache = cached_tensor

        result = depth_helpers._get_sensor_depth(cam, 32, 32)

        assert result.shape == (1, 32, 32)
        # Resizing should preserve value
        assert torch.all(result == 1.0)
