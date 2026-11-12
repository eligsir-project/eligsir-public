"""Bounded latest-wins transport for read-only viewer sidecars."""

from __future__ import annotations

from collections import Counter
from queue import Empty, Full
from typing import Mapping, MutableMapping


class LatestWinsSidecarIPC:
    """One independent capacity-one queue per semantic viewer channel.

    Publishers never wait for a stalled sidecar.  When a channel is full its
    stale value is removed before the new value is offered; traffic on one
    channel therefore cannot evict another channel's latest observation.
    """

    def __init__(self, queues: Mapping[str, object]) -> None:
        if not queues:
            raise ValueError("at least one sidecar channel is required")
        self._queues = dict(queues)
        self._replaced: Counter[str] = Counter()
        self._failed: Counter[str] = Counter()

    def publish(self, channel: str, payload: object) -> bool:
        try:
            target = self._queues[channel]
        except KeyError:
            raise KeyError(f"unknown sidecar channel {channel!r}") from None
        try:
            target.put_nowait(payload)
            return True
        except Full:
            try:
                target.get_nowait()
            except Empty:
                # multiprocessing.Queue may report its capacity semaphore
                # before the feeder makes the item readable. Fail closed and
                # never wait on the viewer from the producer path.
                self._failed[channel] += 1
                return False
            try:
                target.put_nowait(payload)
            except Full:
                self._failed[channel] += 1
                return False
            self._replaced[channel] += 1
            return True
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._failed[channel] += 1
            return False

    def read_latest(self, channel: str):
        try:
            return self._queues[channel].get_nowait()
        except KeyError:
            raise KeyError(f"unknown sidecar channel {channel!r}") from None

    @property
    def replaced(self) -> int:
        return sum(self._replaced.values())

    @property
    def failed(self) -> int:
        return sum(self._failed.values())

    @property
    def replaced_by_channel(self) -> dict[str, int]:
        return dict(self._replaced)

    @property
    def failed_by_channel(self) -> dict[str, int]:
        return dict(self._failed)


def create_sidecar_queues(factory, channels) -> MutableMapping[str, object]:
    """Create per-channel queues using a process-safe queue factory.

    Production uses a multiprocessing manager. Unlike ``multiprocessing.Queue``
    (whose feeder can briefly report full before the stale item is readable),
    a manager queue performs the replace operations synchronously in its owner
    process, preserving latest-wins semantics without a producer-side wait.
    """

    return {str(channel): factory.Queue(maxsize=1) for channel in channels}
