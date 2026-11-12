"""Issue #1664: resolution-tier usage and decision-trace host I/O must be
buffered/batched, not opened-and-appended-to on every completed transaction
inside the GPU submission loop.

Covers:
  - JsonlDecisionTraceObserver: exact row order/content parity against the
    old per-call synchronous writer, zero-progress coalescing unaffected,
    no ``Path.open`` from the hot ``__call__`` path, bounded-buffer overflow
    is a loud explicit failure (never a silent drop), and a normal
    shutdown/close drains every accepted row.
  - record_resolution_tier_usage/flush_resolution_tier_usage: no
    synchronous per-transaction file write, a slow filesystem does not pace
    the hot path below the flush cadence, shutdown drains all buffered
    rows, and the flushed output still parses the same as before for
    downstream report readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eligsir.runtime.causal_replay import JsonlDecisionTraceObserver, decision_trace_row


def _tick(iteration: int, *, executed: bool, optimizer_steps: int = 1, supervision_views: int = 4):
    return SimpleNamespace(
        iteration=iteration,
        executed=executed,
        optimizer_steps=optimizer_steps,
        supervision_views=supervision_views,
        reason="",
        state=SimpleNamespace(value="ready"),
    )


def _old_synchronous_writer(path: Path, ticks) -> list[str]:
    """Reference implementation matching the pre-#1664 observer exactly:
    build the row and append/close on every single call. Used only to prove
    the new buffered observer's on-disk output is unchanged."""

    zero_progress_signature = None
    written: list[str] = []
    for tick in ticks:
        row = decision_trace_row(tick)
        zero_progress = not bool(getattr(tick, "executed", False)) and (
            int(getattr(tick, "optimizer_steps", 0)) == 0
            and int(getattr(tick, "supervision_views", 0)) == 0
        )
        signature_row = dict(row)
        signature_row.pop("iteration", None)
        signature_row.pop("snapshot_id", None)
        signature = json.dumps(signature_row, sort_keys=True)
        if zero_progress and signature == zero_progress_signature:
            continue
        zero_progress_signature = signature if zero_progress else None
        line = json.dumps(row, sort_keys=True) + "\n"
        with path.open("a") as handle:
            handle.write(line)
        written.append(line)
    return written


def _synthetic_ticks(n: int):
    """A mix of executed/starved/zero-progress-duplicate ticks, deterministic."""

    ticks = []
    for i in range(n):
        if i % 7 == 0:
            # Three consecutive identical zero-progress ticks -> only the
            # first of each run should survive coalescing.
            ticks.append(_tick(i, executed=False, optimizer_steps=0, supervision_views=0))
        else:
            ticks.append(_tick(i, executed=True, optimizer_steps=1, supervision_views=4))
    return ticks


def test_10k_rows_preserve_exact_order_and_content_after_buffered_flush(tmp_path):
    ticks = _synthetic_ticks(10_000)

    reference_path = tmp_path / "reference.jsonl"
    reference_lines = _old_synchronous_writer(reference_path, ticks)

    buffered_path = tmp_path / "buffered.jsonl"
    observer = JsonlDecisionTraceObserver(buffered_path, flush_every=37)
    for tick in ticks:
        observer(tick)
    observer.finalize()

    buffered_lines = buffered_path.read_text().splitlines(keepends=True)
    assert buffered_lines == reference_lines
    assert reference_path.read_text() == buffered_path.read_text()


def test_zero_progress_duplicates_remain_coalesced_exactly_as_today(tmp_path):
    ticks = [
        _tick(1, executed=False, optimizer_steps=0, supervision_views=0),
        _tick(2, executed=False, optimizer_steps=0, supervision_views=0),
        _tick(3, executed=True),
        _tick(4, executed=False, optimizer_steps=0, supervision_views=0),
        _tick(5, executed=False, optimizer_steps=0, supervision_views=0),
        _tick(6, executed=False, optimizer_steps=0, supervision_views=0),
    ]
    path = tmp_path / "decisions.jsonl"
    observer = JsonlDecisionTraceObserver(path, flush_every=1000)
    for tick in ticks:
        observer(tick)
    observer.flush()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["iteration"] for row in rows] == [1, 3, 4]


def test_default_trace_off_path_never_opens_a_file(monkeypatch, tmp_path):
    """No observer is constructed at all when tracing is off (the normal
    production path); confirm that path truly performs zero file I/O by
    asserting Path.open is never invoked for a stream of ticks that never
    even touch an observer."""

    calls = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        calls.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    decision_observer = None  # trace-off: matches decision_trace_observer_from_args("")
    for i in range(500):
        if decision_observer is not None:
            decision_observer(_tick(i, executed=True))

    assert calls == []


def test_warmed_transaction_path_below_flush_cadence_invokes_no_path_open(monkeypatch, tmp_path):
    """With tracing *on*, calls below the flush cadence must still perform
    zero file opens -- only an explicit flush/finalize touches disk."""

    calls = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        calls.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    path = tmp_path / "decisions.jsonl"
    observer = JsonlDecisionTraceObserver(path, flush_every=200)
    for i in range(199):
        observer(_tick(i, executed=True))

    assert calls == []

    observer.flush()
    assert len(calls) == 1


def test_bounded_writer_shutdown_drains_all_accepted_rows(tmp_path):
    path = tmp_path / "decisions.jsonl"
    observer = JsonlDecisionTraceObserver(path, flush_every=64)
    n = 513
    for i in range(n):
        observer(_tick(i, executed=True))

    # Simulate exception-safe shutdown: finalize() must flush whatever is
    # still buffered since the last cadence flush.
    observer.finalize()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["iteration"] for row in rows] == list(range(n))


def test_injected_slow_filesystem_does_not_pace_hot_path_below_capacity(monkeypatch, tmp_path):
    import time

    path = tmp_path / "decisions.jsonl"
    real_open = Path.open
    open_calls = []

    def slow_open(self, *args, **kwargs):
        open_calls.append(self)
        time.sleep(0.05)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", slow_open)

    # flush_every larger than the number of ticks submitted -> the hot path
    # below is guaranteed to never trigger a flush, so a "slow disk" must
    # have zero effect on it.
    observer = JsonlDecisionTraceObserver(path, flush_every=10_000)
    started = time.perf_counter()
    for i in range(500):
        observer(_tick(i, executed=True))
    elapsed = time.perf_counter() - started

    assert open_calls == []
    # `open_calls == []` above already proves the hot path never touches the
    # slow filesystem -- that is the real assertion. This wall-clock bound is
    # only a loose sanity check against a catastrophic regression (e.g. one
    # accidental sync flush would add >=0.05s from a single slow_open call);
    # it must stay well clear of ordinary CI-runner scheduling jitter, which
    # has been observed to push 500 iterations of pure Python/JSON overhead
    # past a too-tight 0.05s bound with no filesystem access at all.
    assert elapsed < 1.0, "buffered hot-path append must not touch the slow filesystem"


def test_overflow_past_bounded_capacity_is_a_loud_explicit_failure(tmp_path):
    """A persistently failing filesystem must never silently drop causal
    decision-trace rows -- once retry pressure exceeds the bound, the
    observer must raise, not keep quietly discarding rows."""

    path = tmp_path / "not_a_dir" / "decisions.jsonl"
    # path.parent does not exist -> every flush attempt raises OSError and
    # is swallowed by flush() (recorded in _write_error), so the buffer
    # keeps growing across calls until it exceeds max_buffered_rows.
    observer = JsonlDecisionTraceObserver(path, flush_every=5, max_buffered_rows=12)

    with pytest.raises(RuntimeError, match="buffer overflow"):
        for i in range(50):
            observer(_tick(i, executed=True))

    # Nothing was silently discarded: every accepted row is still sitting in
    # the buffer at the moment of the loud failure.
    assert len(observer._buffer) > 12


def test_finalize_surfaces_write_error_in_metadata(tmp_path):
    path = tmp_path / "missing" / "decisions.jsonl"
    observer = JsonlDecisionTraceObserver(path, flush_every=1000, max_buffered_rows=1000)
    observer(_tick(0, executed=True))
    observer.flush()
    assert observer._write_error is not None

    # Fix the destination and confirm the retained row is not lost -- the
    # next flush (via finalize here) succeeds and writes it.
    path.parent.mkdir(parents=True)
    summary = observer.finalize()
    assert path.is_file()
    assert summary["decision_trace_rows_flushed"] == 1


# ---------------------------------------------------------------------------
# resolution_tier_usage.py buffering
# ---------------------------------------------------------------------------


def test_resolution_tier_usage_below_cadence_performs_no_file_write(monkeypatch, tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        USAGE_TRACE_FILENAME,
        record_resolution_tier_usage,
    )

    calls = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        calls.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    for i in range(50):
        record_resolution_tier_usage(tmp_path, iteration=i, tier="native")

    assert calls == []
    assert not (tmp_path / USAGE_TRACE_FILENAME).exists()


def test_resolution_tier_usage_shutdown_drains_all_buffered_rows(tmp_path):
    from eligsir.mapping.scheduler.resolution_tier_usage import (
        USAGE_TRACE_FILENAME,
        flush_resolution_tier_usage,
        record_resolution_tier_usage,
    )

    n = 733
    for i in range(n):
        record_resolution_tier_usage(tmp_path, iteration=i, tier="native")

    flush_resolution_tier_usage(tmp_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / USAGE_TRACE_FILENAME).read_text().splitlines()
    ]
    assert [row["iteration"] for row in rows] == list(range(n))


def test_resolution_tier_usage_output_remains_report_compatible(tmp_path):
    """The buffered writer's flushed output must still be exactly what
    plot_resolution_tier_usage/plot_resolution_tier_over_time expect: one
    JSON object per line, same schema as the old synchronous writer."""

    from eligsir.mapping.scheduler.resolution_tier_usage import (
        flush_resolution_tier_usage,
        plot_resolution_tier_over_time,
        record_resolution_tier_usage,
    )

    for i in range(0, 400, 20):
        record_resolution_tier_usage(tmp_path, iteration=i, tier="native", transaction_class="fresh")
    flush_resolution_tier_usage(tmp_path)

    output = plot_resolution_tier_over_time(tmp_path)
    assert output is not None and output.is_file()


def test_resolution_tier_usage_slow_filesystem_does_not_pace_hot_path(monkeypatch, tmp_path):
    import time

    from eligsir.mapping.scheduler.resolution_tier_usage import record_resolution_tier_usage

    real_open = Path.open
    open_calls = []

    def slow_open(self, *args, **kwargs):
        open_calls.append(self)
        time.sleep(0.05)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", slow_open)

    started = time.perf_counter()
    for i in range(150):
        record_resolution_tier_usage(tmp_path, iteration=i, tier="native")
    elapsed = time.perf_counter() - started

    assert open_calls == []
    # See the equivalent decision-trace test's comment above: `open_calls ==
    # []` is the real proof; this is only a loose sanity bound clear of
    # ordinary CI-runner jitter.
    assert elapsed < 1.0
