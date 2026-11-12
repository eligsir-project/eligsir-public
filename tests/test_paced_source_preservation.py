"""Finite paced replay preserves source identities through bounded stalls."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    CameraIntrinsics,
    SensorPacket,
    SourceIdentity,
)
from eligsir.runtime import causal_replay
from eligsir.runtime.causal_replay import DatasetPacketAdapter, _PacedPacketProducer
from eligsir.tracking.providers import observation_from_pose_matrix


_REAL_MONOTONIC = time.monotonic


@dataclass(frozen=True)
class _Record:
    ordinal: int


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def perf_counter(self) -> float:
        return self.now


class _Adapter(DatasetPacketAdapter[_Record]):
    def __init__(
        self,
        *,
        first_entered: threading.Event | None = None,
        release_first: threading.Event | None = None,
        fail_ordinal: int | None = None,
    ) -> None:
        self.first_entered = first_entered
        self.release_first = release_first
        self.fail_ordinal = fail_ordinal
        self.decoded_ordinals: list[int] = []
        self._lock = threading.Lock()

    def packet_from_record(self, record: _Record) -> SensorPacket:
        with self._lock:
            self.decoded_ordinals.append(record.ordinal)
        if record.ordinal == 7 and self.first_entered is not None:
            self.first_entered.set()
            assert self.release_first is not None
            self.release_first.wait()
        if record.ordinal == self.fail_ordinal:
            raise ValueError(f"decode failed for {record.ordinal}")
        source = SourceIdentity("dataset", 0, record.ordinal, 1_000 + record.ordinal)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(
                np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"
            ),
            depth=None,
            intrinsics=CameraIntrinsics(
                fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2
            ),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
        )


def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = _REAL_MONOTONIC() + timeout_s
    while not predicate():
        if _REAL_MONOTONIC() >= deadline:
            raise AssertionError("timed out waiting for producer progress")
        threading.Event().wait(0.005)


def _drain_ordinals(producer: _PacedPacketProducer[_Record]) -> list[int]:
    ordinals = []
    while not producer.complete:
        packet = producer.take(timeout_s=0.01)
        if packet is not None:
            ordinals.append(packet.source.source_ordinal)
    return ordinals


def _producer(
    records: tuple[_Record, ...],
    adapter: _Adapter,
    *,
    capacity: int = 2,
) -> _PacedPacketProducer[_Record]:
    return _PacedPacketProducer(
        records, adapter, source_hz=100.0, capacity=capacity, decode_workers=2
    )


def test_first_decode_stall_releases_catalogue_then_drains_all_identities(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    adapter = _Adapter(
        first_entered=first_decode_entered, release_first=release_first_decode
    )
    producer = _producer(
        tuple(_Record(ordinal) for ordinal in (7, 42, 99, 201)), adapter
    )

    try:
        producer.start()
        assert first_decode_entered.wait(2.0)
        clock.now = 101.0
        _wait_until(lambda: producer.offered == 4)
        assert adapter.decoded_ordinals in ([7], [7, 42])
        release_first_decode.set()
        assert _drain_ordinals(producer) == [7, 42, 99, 201]
        assert producer.dropped == 0
        assert producer.high_watermark <= 2
    finally:
        release_first_decode.set()
        producer.close()


def test_slow_consumer_saturation_preserves_all_catalogue_records(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    adapter = _Adapter(
        first_entered=first_decode_entered, release_first=release_first_decode
    )
    producer = _producer(tuple(_Record(ordinal) for ordinal in (7, 42, 99, 201)), adapter)

    try:
        producer.start()
        assert first_decode_entered.wait(2.0)
        clock.now = 101.0
        _wait_until(lambda: producer.offered == 4)
        release_first_decode.set()
        assert _drain_ordinals(producer) == [7, 42, 99, 201]
        assert producer.dropped == 0
        assert producer.high_watermark <= 2
    finally:
        release_first_decode.set()
        producer.close()


def test_future_catalogue_record_is_not_decoded_before_its_deadline(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    adapter = _Adapter()
    producer = _producer((_Record(7), _Record(42)), adapter)

    try:
        producer.start()
        _wait_until(lambda: adapter.decoded_ordinals == [7])
        assert producer.offered == 1
    finally:
        producer.close()


def test_completed_futures_publish_in_catalogue_order(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    adapter = _Adapter(
        first_entered=first_decode_entered, release_first=release_first_decode
    )
    producer = _producer((_Record(7), _Record(42)), adapter)

    try:
        producer.start()
        assert first_decode_entered.wait(2.0)
        clock.now = 101.0
        _wait_until(lambda: producer.offered == 2)
        _wait_until(lambda: adapter.decoded_ordinals == [7, 42])
        assert producer.take(timeout_s=0.0) is None
        release_first_decode.set()
        assert _drain_ordinals(producer) == [7, 42]
    finally:
        release_first_decode.set()
        producer.close()


def test_empty_and_single_record_catalogues_complete_without_loss(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    empty = _producer((), _Adapter())
    one = _producer((_Record(7),), _Adapter())
    try:
        empty.start()
        _wait_until(lambda: empty.complete)
        one.start()
        assert _drain_ordinals(one) == [7]
        assert empty.dropped == one.dropped == 0
    finally:
        empty.close()
        one.close()


def test_decode_error_is_terminal_and_preserves_original_cause(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    producer = _producer((_Record(7),), _Adapter(fail_ordinal=7))

    try:
        producer.start()
        with pytest.raises(RuntimeError, match="paced source producer failed") as raised:
            producer.wait_for_initial_packet_or_terminal()
        assert isinstance(raised.value.__cause__, ValueError)
        assert producer.complete is False
    finally:
        producer.close()


def test_close_cancels_blocked_decode_without_late_packet_publication(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    producer = _producer(
        (_Record(7),),
        _Adapter(first_entered=first_decode_entered, release_first=release_first_decode),
    )
    try:
        producer.start()
        assert first_decode_entered.wait(2.0)
        closer = threading.Thread(target=producer.close)
        closer.start()
        _wait_until(lambda: producer._stop.is_set())
        release_first_decode.set()
        closer.join(timeout=2.0)
        assert not closer.is_alive()
        with pytest.raises(RuntimeError, match="paced source producer failed"):
            producer.take(timeout_s=0.0)
        assert producer.complete is False
    finally:
        release_first_decode.set()
        producer.close()


def test_close_reports_an_uninterruptible_decoder_as_incomplete(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    producer = _producer(
        (_Record(7),),
        _Adapter(first_entered=first_decode_entered, release_first=release_first_decode),
    )
    try:
        producer.start()
        assert first_decode_entered.wait(2.0)
        producer.close()
        assert producer.source_ingress_report_block()["ingress_shutdown_incomplete"]
        assert producer.complete is False
    finally:
        release_first_decode.set()
        producer.close()


def test_close_wins_the_enqueue_race(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    producer = _producer((_Record(7),), _Adapter())
    enqueue_entered = threading.Event()
    release_enqueue = threading.Event()
    put_nowait = producer._queue.put_nowait
    order = []

    class _StopProbe:
        def __init__(self) -> None:
            self._event = threading.Event()

        def set(self) -> None:
            order.append("cancel")
            self._event.set()

        def is_set(self) -> bool:
            return self._event.is_set()

        def wait(self, timeout=None):
            return self._event.wait(timeout)

    class _LockProbe:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.watch = False
            self.waiting = threading.Event()

        def __enter__(self):
            if self.watch:
                self.waiting.set()
            self._lock.acquire()
            return self

        def __exit__(self, *_args):
            self._lock.release()

    lock = _LockProbe()
    producer._lock = lock
    producer._stop = _StopProbe()

    def blocked_put(packet):
        enqueue_entered.set()
        release_enqueue.wait()
        put_nowait(packet)
        order.append("enqueue")

    monkeypatch.setattr(producer._queue, "put_nowait", blocked_put)
    try:
        producer.start()
        assert enqueue_entered.wait(2.0)
        lock.watch = True
        closer = threading.Thread(target=producer.close)
        closer.start()
        assert lock.waiting.wait(2.0)
        assert not producer._stop.is_set()
        release_enqueue.set()
        closer.join(timeout=2.0)
        assert not closer.is_alive()
        assert order == ["enqueue", "cancel"]
    finally:
        release_enqueue.set()
        producer.close()


def test_completed_decode_does_not_wait_for_next_source_deadline(monkeypatch):
    clock = _Clock(100.0)
    monkeypatch.setattr(causal_replay, "time", clock)
    adapter = _Adapter()
    producer = _PacedPacketProducer(
        (_Record(7), _Record(42)), adapter, source_hz=0.1, capacity=2
    )
    try:
        producer.start()
        _wait_until(lambda: adapter.decoded_ordinals == [7])
        _wait_until(lambda: producer.queue_depth == 1)
    finally:
        producer.close()


def test_rejects_unsupported_iterator_without_acquiring_a_record():
    class _Unsupported:
        def __iter__(self):
            raise AssertionError("must not acquire source records")

    with pytest.raises(TypeError, match="finite stable source catalogue"):
        _producer(_Unsupported(), _Adapter())  # type: ignore[arg-type]
