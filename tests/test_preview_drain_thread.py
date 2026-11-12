"""Tests for eligsir/viewers/drain_thread.py (issue #571, scoped)."""
from __future__ import annotations

import threading
import time

from eligsir.viewers.drain_thread import PreviewDrainThread


def test_notify_causes_a_prompt_tick_without_waiting_full_interval():
    ticked = threading.Event()

    def on_tick():
        ticked.set()

    drain = PreviewDrainThread(on_tick=on_tick, poll_interval_s=5.0)
    try:
        drain.start()
        drain.notify()
        assert ticked.wait(timeout=1.0), "notify() should cause a near-immediate tick"
    finally:
        drain.stop()


def test_ticks_periodically_even_without_notify():
    count = {"n": 0}
    lock = threading.Lock()

    def on_tick():
        with lock:
            count["n"] += 1

    drain = PreviewDrainThread(on_tick=on_tick, poll_interval_s=0.01)
    try:
        drain.start()
        time.sleep(0.2)
    finally:
        drain.stop()
    assert count["n"] >= 3


def test_tick_exceptions_are_isolated_and_counted():
    def on_tick():
        raise RuntimeError("boom")

    drain = PreviewDrainThread(on_tick=on_tick, poll_interval_s=0.01)
    try:
        drain.start()
        time.sleep(0.1)
    finally:
        drain.stop()
    assert drain.tick_errors > 0


def test_stop_is_idempotent_and_joins():
    drain = PreviewDrainThread(on_tick=lambda: None, poll_interval_s=0.01)
    drain.start()
    drain.stop()
    drain.stop()  # must not raise
