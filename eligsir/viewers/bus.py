"""Bounded, latest-only ``PreviewBus`` (issue #570, phase A).

Canonical design: ``docs/architecture/2026-08-14-canonical-live-viewer-sidecar.md``
§2.2 / §3.2. Unwired: nothing in ``train.py`` / ``train_streaming.py`` /
``registry.py`` constructs one.

Three properties, each structural rather than documented
--------------------------------------------------------

**1. Latest-WINS, not drop-newest.** The repo's existing precedent is the
*opposite* and must not be copied: ``ROS2RVizViewer`` (``ros2_rviz.py:471-473``)
does ``if self._queue.full(): self._dropped += 1; return``, which discards the
**incoming** snapshot and lets the older, staler one survive to be published.
Design doc §9.2 calls this out explicitly — under a sink persistently slower
than the producer, drop-newest converges on displaying an arbitrarily stale
frame while latest-wins converges on the freshest available one. For a live
reconstruction-progress viewer that is a behavioural requirement.

The discriminating fixture, in ``tests/test_canonical_preview_bus.py``: publish
A, publish B with nothing consumed, then consume once. This bus yields **B** and
counts one drop. A ``queue.Queue(maxsize=1)`` implementation — the naive
alternative, and the one already in this repo — yields A on the identical
fixture. The test asserts both halves.

**2. Publishing never touches a sink.** :meth:`PreviewBus.publish` writes one
slot and returns. It holds no sink references, calls nothing, and cannot await
anything, so "a stalled sink cannot block submission" is provable *without
threads*: a sink that blocks forever is simply never invoked on this path.
Delivery is a separate, consumer-side concern
(``eligsir.viewers.sinks.base.deliver_pending``), which #571/#572 will run
off the training thread.

**3. Channels are typed.** Each :class:`~eligsir.viewers.events.PreviewChannel`
slot accepts exactly one event type. Publishing a ``TrainingPairPreview`` into
the ``raw_input`` slot is refused, so "raw input and the matched training pair
cannot be confused" holds at the bus as well as at the schema.

Fan-out without duplicated capture
----------------------------------
Subscribers hold per-channel read *cursors* over the same slot. Two subscribers
polling the same channel receive the **same object** (``is``-identical), so
adding a second sink costs one dict lookup, never a second capture, conversion
or copy. The design doc's "multi-viewer operation would currently duplicate
capture/conversion work" problem is answered by the cursor model, not by a
per-sink queue.

Drop accounting
---------------
A drop is counted when a *replaced* envelope had not yet been polled by every
currently-registered subscriber. With zero subscribers every replacement is a
drop, which is the honest reading: nothing observed it.

Thread-safety
-------------
A single ``threading.Lock`` guards slot/cursor/counter mutation. It is held only
for a few dict assignments — never across a sink call, an I/O call or a copy —
so it cannot become a backpressure edge. The producer's worst case is contention
with a consumer's dict read.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from eligsir.viewers.events import (
    EVENT_TYPE_BY_CHANNEL,
    PreviewChannel,
    PreviewEvent,
    channel_for_event,
)

__all__ = [
    "PreviewBusError",
    "PreviewEnvelope",
    "SubscriberHandle",
    "PreviewBus",
]


class PreviewBusError(ValueError):
    """A publish/subscribe call violated the bus contract."""


@dataclass(frozen=True)
class PreviewEnvelope:
    """An event plus the bus-assigned identity the event itself must not carry.

    ``seq`` is minted by the bus's own monotonic counter at publish time, so a
    producer cannot reuse, skip or reorder one. This is why
    :class:`~eligsir.viewers.events.PreviewEvent` has no ``preview_seq``
    field.
    """

    seq: int
    channel: PreviewChannel
    event: PreviewEvent
    published_wall_time_s: float


@dataclass
class SubscriberHandle:
    """Opaque per-consumer cursor set. Created only by :meth:`PreviewBus.subscribe`."""

    name: str
    _bus_id: int
    _cursor: Dict[PreviewChannel, int] = field(
        default_factory=lambda: {ch: 0 for ch in PreviewChannel}
    )

    def __hash__(self) -> int:  # cursors are identity-keyed in the bus
        return id(self)


class PreviewBus:
    """Independent latest-only slots, one per semantic channel."""

    def __init__(self, *, name: str = "preview") -> None:
        self.name = str(name)
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._slots: Dict[PreviewChannel, Optional[PreviewEnvelope]] = {
            ch: None for ch in PreviewChannel
        }
        self._slot_polled_by: Dict[PreviewChannel, set] = {ch: set() for ch in PreviewChannel}
        self._subscribers: list = []
        self._submitted: Dict[PreviewChannel, int] = {ch: 0 for ch in PreviewChannel}
        self._dropped: Dict[PreviewChannel, int] = {ch: 0 for ch in PreviewChannel}

    # -- producer side --------------------------------------------------
    def publish(
        self,
        event: PreviewEvent,
        *,
        channel: Optional[PreviewChannel] = None,
        wall_time_s: Optional[float] = None,
    ) -> PreviewEnvelope:
        """Write the latest value of one channel. Never blocks, never calls a sink.

        ``channel`` is optional and derived from the event's exact type; passing
        one that disagrees is refused rather than honoured, which is what makes
        cross-channel confusion (a ``TrainingPairPreview`` in the ``raw_input``
        slot) unconstructible.
        """
        if not isinstance(event, PreviewEvent):
            raise PreviewBusError(
                f"publish expects a canonical preview event, got {type(event).__name__}"
            )
        resolved = channel_for_event(event)
        if channel is not None:
            if not isinstance(channel, PreviewChannel):
                raise PreviewBusError(f"channel must be a PreviewChannel, got {channel!r}")
            if channel is not resolved:
                raise PreviewBusError(
                    f"{type(event).__name__} belongs to channel "
                    f"{resolved.value!r}, refusing to publish it to "
                    f"{channel.value!r}; the {channel.value!r} slot carries "
                    f"{EVENT_TYPE_BY_CHANNEL[channel].__name__}"
                )
        stamp = float(wall_time_s) if wall_time_s is not None else float(event.wall_time_s)
        with self._lock:
            pending = self._slots[resolved]
            if pending is not None and not self._fully_polled_locked(resolved):
                # LATEST WINS: the stale pending item is the one discarded.
                self._dropped[resolved] += 1
            envelope = PreviewEnvelope(
                seq=next(self._seq),
                channel=resolved,
                event=event,
                published_wall_time_s=stamp,
            )
            self._slots[resolved] = envelope
            self._slot_polled_by[resolved] = set()
            self._submitted[resolved] += 1
        return envelope

    def _fully_polled_locked(self, channel: PreviewChannel) -> bool:
        if not self._subscribers:
            return False
        polled = self._slot_polled_by[channel]
        return all(id(s) in polled for s in self._subscribers)

    # -- consumer side ---------------------------------------------------
    def subscribe(self, name: str) -> SubscriberHandle:
        """Register a new subscriber for all channels.

        Args:
            name: Display name for the subscriber (e.g., "rerun", "rviz"),
                used in telemetry/logging.

        Returns:
            A SubscriberHandle used to poll channels and identify this
            subscriber for drop accounting.
        """
        handle = SubscriberHandle(name=str(name), _bus_id=id(self))
        with self._lock:
            self._subscribers.append(handle)
        return handle

    def unsubscribe(self, handle: SubscriberHandle) -> None:
        """Remove a subscriber from all channels.

        Args:
            handle: SubscriberHandle previously returned by subscribe().

        Raises:
            PreviewBusError: If the handle is not recognized.
        """
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not handle]
            for polled in self._slot_polled_by.values():
                polled.discard(id(handle))

    def poll(
        self, handle: SubscriberHandle, channel: PreviewChannel
    ) -> Optional[PreviewEnvelope]:
        """Return this channel's pending envelope if unseen by ``handle``, else None."""
        self._check_handle(handle)
        if not isinstance(channel, PreviewChannel):
            raise PreviewBusError(f"channel must be a PreviewChannel, got {channel!r}")
        with self._lock:
            envelope = self._slots[channel]
            if envelope is None or envelope.seq <= handle._cursor[channel]:
                return None
            handle._cursor[channel] = envelope.seq
            self._slot_polled_by[channel].add(id(handle))
            return envelope

    def poll_all(self, handle: SubscriberHandle) -> Tuple[PreviewEnvelope, ...]:
        """Every channel's unseen envelope, in stable channel order."""
        out = []
        for channel in PreviewChannel:
            envelope = self.poll(handle, channel)
            if envelope is not None:
                out.append(envelope)
        return tuple(out)

    def _check_handle(self, handle: SubscriberHandle) -> None:
        if not isinstance(handle, SubscriberHandle) or handle._bus_id != id(self):
            raise PreviewBusError("handle was not issued by this bus")

    # -- telemetry (design doc §12) --------------------------------------
    def stats(self) -> Dict[str, int]:
        """Snapshot of bus telemetry: submissions, drops, subscriber count.

        Returns:
            Dictionary with keys like "preview_<channel>_submitted",
            "preview_<channel>_dropped", and "preview_subscribers".
        """
        with self._lock:
            out: Dict[str, int] = {}
            for channel in PreviewChannel:
                out[f"preview_{channel.value}_submitted"] = self._submitted[channel]
                out[f"preview_{channel.value}_dropped"] = self._dropped[channel]
            out["preview_subscribers"] = len(self._subscribers)
            return out

    def pending_count(self) -> int:
        """Total envelopes resident in the bus. Bounded by the channel count.

        This is the number "unbounded viewer queues == 0" (design doc §13) is
        checked against: it can never exceed ``len(PreviewChannel)`` no matter
        how many events are published or how long a consumer stalls.
        """
        with self._lock:
            return sum(1 for e in self._slots.values() if e is not None)

    @property
    def max_pending_count(self) -> int:
        """Maximum number of envelopes that can be resident in the bus.

        Bounded by the number of distinct preview channels (one slot per channel).
        This proves the bus cannot accumulate unbounded queue depth regardless
        of publication rate or subscriber stalls.
        """
        return len(PreviewChannel)
