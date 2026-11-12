"""Unit tests for the async IngestionWorker handoff semantics (no GPU/model)."""
import threading
import time

from eligsir.core.ingestion_worker import IngestionWorker


def test_poll_is_nonblocking_when_empty():
    w = IngestionWorker(source_fn=lambda: None, prep_fn=lambda r: r, depth=4)
    w.start()
    try:
        t0 = time.perf_counter()
        assert w.poll() is None          # nothing ready
        assert (time.perf_counter() - t0) < 0.05   # did not block
    finally:
        w.stop()


def test_all_frames_delivered_in_order_no_drop():
    src = list(range(50))
    it = iter(src)
    lock = threading.Lock()

    def source_fn():
        with lock:
            return next(it, None)

    w = IngestionWorker(source_fn=source_fn, prep_fn=lambda r: r * 10, depth=4)
    w.start()
    got = []
    try:
        # Slow consumer: exercises back-pressure (queue fills, worker waits).
        deadline = time.perf_counter() + 5.0
        while len(got) < len(src) and time.perf_counter() < deadline:
            item = w.poll()
            if item is None:
                time.sleep(0.005)
                continue
            got.append(item)
            time.sleep(0.002)
    finally:
        w.stop()
    assert got == [v * 10 for v in src]          # no drops, order preserved
    assert w.n_consumed == len(src)


def test_prep_exception_does_not_kill_worker():
    vals = iter([1, 2, "boom", 3, None])
    lock = threading.Lock()

    def source_fn():
        with lock:
            return next(vals, None)

    def prep_fn(r):
        if r == "boom":
            raise ValueError("bad frame")
        return r

    errs = []
    w = IngestionWorker(source_fn=source_fn, prep_fn=prep_fn, depth=4,
                        on_error=errs.append)
    w.start()
    got = []
    try:
        deadline = time.perf_counter() + 3.0
        while len(got) < 3 and time.perf_counter() < deadline:
            item = w.poll()
            if item is not None:
                got.append(item)
            else:
                time.sleep(0.005)
    finally:
        w.stop()
    assert got == [1, 2, 3]               # survived the bad frame
    assert any(isinstance(e, ValueError) for e in errs)
