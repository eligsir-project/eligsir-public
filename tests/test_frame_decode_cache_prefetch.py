"""Tests for FrameDecodeCache's background read-ahead, including the
multi-worker decode-ahead added 2026-07-27 (Partition-Focus follow-up:
parallel CPU decode to reduce the ScanNet++ ingestion stall)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from eligsir.utils.streaming_frames import FrameDecodeCache, StreamingRGBDFrame, set_decode_cache


@pytest.fixture(autouse=True)
def _clear_global_decode_cache():
    # load_frame_rgb/load_frame_depth_np route through the module-global
    # _DECODE_CACHE singleton, not the FrameDecodeCache instance directly --
    # each test must install its own instance and clean up after itself so
    # tests don't leak state into each other.
    yield
    set_decode_cache(None)


def _frame(idx: int) -> StreamingRGBDFrame:
    return StreamingRGBDFrame(
        index=idx,
        timestamp=float(idx),
        rgb_path="",
        _rgb_np=np.full((4, 4, 3), idx % 255, dtype=np.uint8),
        _depth_np=np.full((4, 4), float(idx), dtype=np.float32),
    )


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_single_worker_prefetch_decodes_ahead_window():
    cache = FrameDecodeCache(maxsize=16)
    set_decode_cache(cache)
    frames = [_frame(i) for i in range(10)]
    cache.start_prefetch(lambda i: frames[i], n_frames=10, ahead=4, n_workers=1)
    cache.note_consumed(0)
    assert _wait_until(lambda: cache.peek_rgb(1) is not None)
    assert _wait_until(lambda: cache.peek_rgb(4) is not None)
    cache.stop()


def test_multi_worker_prefetch_decodes_same_window_without_duplication_errors():
    cache = FrameDecodeCache(maxsize=32)
    set_decode_cache(cache)
    n = 30
    frames = [_frame(i) for i in range(n)]
    cache.start_prefetch(lambda i: frames[i], n_frames=n, ahead=10, n_workers=4)
    cache.note_consumed(0)
    assert _wait_until(lambda: all(cache.peek_rgb(i) is not None for i in range(1, 11)), timeout=10.0)
    for i in range(1, 11):
        rgb = cache.peek_rgb(i)
        depth = cache.peek_depth(i)
        assert rgb is not None and depth is not None
        assert np.asarray(rgb).shape == (4, 4, 3)
    cache.stop()


def test_multi_worker_prefetch_advances_with_consumption():
    cache = FrameDecodeCache(maxsize=64)
    set_decode_cache(cache)
    n = 50
    frames = [_frame(i) for i in range(n)]
    cache.start_prefetch(lambda i: frames[i], n_frames=n, ahead=5, n_workers=3)
    for consumed in range(0, 40, 5):
        cache.note_consumed(consumed)
        target = min(consumed + 5, n - 1)
        assert _wait_until(lambda t=target: cache.peek_rgb(t) is not None, timeout=5.0), (
            f"expected position {target} to be decoded after consuming up to {consumed}"
        )
    cache.stop()


def test_default_n_workers_is_one_preserves_original_behavior():
    cache = FrameDecodeCache(maxsize=8)
    set_decode_cache(cache)
    frames = [_frame(i) for i in range(5)]
    cache.start_prefetch(lambda i: frames[i], n_frames=5, ahead=2)  # n_workers defaults to 1
    assert len(cache._threads) == 1
    cache.note_consumed(0)
    assert _wait_until(lambda: cache.peek_rgb(1) is not None)
    cache.stop()
