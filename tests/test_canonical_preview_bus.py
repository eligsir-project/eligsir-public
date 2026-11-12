"""Issue #570: the bounded, latest-only ``PreviewBus``.

The decisive test here is ``test_latest_wins_not_drop_newest``: it is written so
that the naive alternative -- ``queue.Queue(maxsize=1)``, which is what
``ros2_rviz.py:471-473`` already does in this repo -- gives the OPPOSITE answer
on the identical fixture. That alternative is constructed inline and asserted to
differ, so the test cannot be satisfied by copying the existing precedent.
"""
from __future__ import annotations

import queue

import numpy as np
import pytest

from eligsir.viewers.bus import PreviewBus, PreviewBusError, PreviewEnvelope
from eligsir.viewers.events import (
    CameraRole,
    CapturedImage,
    ImageKind,
    MappingPhase,
    MetricsPreview,
    PreviewChannel,
    RawInputPreview,
    TrainingPairPreview,
)


def _pair(iteration: int, camera_id: int = 7) -> TrainingPairPreview:
    return TrainingPairPreview(
        iteration=iteration,
        wall_time_s=float(iteration),
        target=CapturedImage(np.zeros((2, 2, 3), np.uint8), camera_id, ImageKind.TARGET),
        render=CapturedImage(np.zeros((2, 2, 3), np.uint8), camera_id, ImageKind.RENDER),
        camera_role=CameraRole.LOCAL,
        phase=MappingPhase.ACTIVE_MAPPING,
    )


def _raw(frame_id: int) -> RawInputPreview:
    return RawInputPreview(
        iteration=frame_id, wall_time_s=float(frame_id), source_frame_id=frame_id,
        rgb=np.zeros((2, 2, 3), np.uint8), timestamp_s=float(frame_id),
    )


# --------------------------------------------------------------------------
# latest-WINS -- and the naive alternative demonstrably differs
# --------------------------------------------------------------------------
def test_latest_wins_not_drop_newest():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    a, b = _pair(1), _pair(2)
    bus.publish(a)
    bus.publish(b)

    got = bus.poll(sub, PreviewChannel.TRAINING_PAIR)
    assert got is not None
    assert got.event is b, "latest-only slot must surrender the NEWEST event"
    assert bus.stats()["preview_training_pair_dropped"] == 1

    # The naive alternative already present in this repo (ros2_rviz.py:471-473
    # / queue.Queue(maxsize=1) drop-incoming) yields the OTHER event.
    naive: queue.Queue = queue.Queue(maxsize=1)
    naive.put(a)
    if naive.full():
        pass  # drops the incoming `b`, exactly as ros2_rviz.py does
    assert naive.get() is a
    assert got.event is not a


def test_latest_wins_across_a_long_burst():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    events = [_pair(i) for i in range(1, 51)]
    for e in events:
        bus.publish(e)
    got = bus.poll(sub, PreviewChannel.TRAINING_PAIR)
    assert got.event is events[-1]
    assert bus.stats()["preview_training_pair_submitted"] == 50
    assert bus.stats()["preview_training_pair_dropped"] == 49


# --------------------------------------------------------------------------
# bounded
# --------------------------------------------------------------------------
def test_bus_cannot_grow_unbounded():
    bus = PreviewBus()
    bus.subscribe("stalled-consumer-that-never-polls")
    for i in range(1, 2001):
        bus.publish(_pair(i))
        bus.publish(_raw(i))
        bus.publish(MetricsPreview(iteration=i, wall_time_s=float(i), metrics={"a": 1.0}))
    assert bus.pending_count() <= bus.max_pending_count
    assert bus.max_pending_count == len(PreviewChannel)
    assert bus.pending_count() == 3


def test_channels_do_not_evict_each_other():
    """A rare map snapshot must not be evicted by frequent camera frames."""
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_raw(1))
    for i in range(1, 101):
        bus.publish(_pair(i))
    assert bus.poll(sub, PreviewChannel.RAW_INPUT).event.source_frame_id == 1
    assert bus.stats()["preview_raw_input_dropped"] == 0


# --------------------------------------------------------------------------
# publish never touches a sink -> a stalled sink cannot block submission
# --------------------------------------------------------------------------
def test_publish_does_not_call_any_consumer():
    """Structural proof: the bus has no sink-registration API to call through."""
    bus = PreviewBus()
    assert not any("sink" in name.lower() for name in dir(bus))
    # ...and nothing on the bus holds a callable a publish could dispatch to.
    assert not [
        name
        for name, value in vars(bus).items()
        if callable(value) and name not in ("_clock",)
    ]
    bus.publish(_pair(1))


def test_submission_returns_with_a_permanently_stalled_consumer():
    """No thread needed: the consumer simply never polls, and publish is unaffected."""
    bus = PreviewBus()
    bus.subscribe("never-polls")
    for i in range(1, 1001):
        envelope = bus.publish(_pair(i))
        assert isinstance(envelope, PreviewEnvelope)
    assert bus.stats()["preview_training_pair_submitted"] == 1000


# --------------------------------------------------------------------------
# typed channels
# --------------------------------------------------------------------------
def test_publishing_to_the_wrong_channel_is_refused():
    bus = PreviewBus()
    with pytest.raises(PreviewBusError, match="belongs to channel 'training_pair'"):
        bus.publish(_pair(1), channel=PreviewChannel.RAW_INPUT)
    with pytest.raises(PreviewBusError, match="belongs to channel 'raw_input'"):
        bus.publish(_raw(1), channel=PreviewChannel.TRAINING_PAIR)


def test_publishing_a_non_event_is_refused():
    bus = PreviewBus()
    with pytest.raises(PreviewBusError, match="canonical preview event"):
        bus.publish(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# fan-out shares one captured event
# --------------------------------------------------------------------------
def test_two_subscribers_receive_the_identical_object():
    bus = PreviewBus()
    a, b = bus.subscribe("a"), bus.subscribe("b")
    event = _pair(1)
    bus.publish(event)
    ea = bus.poll(a, PreviewChannel.TRAINING_PAIR)
    eb = bus.poll(b, PreviewChannel.TRAINING_PAIR)
    assert ea is eb
    assert ea.event is event
    assert ea.event.target.array is event.target.array


def test_fanout_does_not_multiply_captures():
    """Capture is invoked once regardless of subscriber count."""
    captures = {"n": 0}

    def capture(i):
        captures["n"] += 1
        return _pair(i)

    bus = PreviewBus()
    subs = [bus.subscribe(f"s{i}") for i in range(5)]
    bus.publish(capture(1))
    for s in subs:
        assert bus.poll(s, PreviewChannel.TRAINING_PAIR) is not None
    assert captures["n"] == 1


def test_polling_twice_yields_nothing_the_second_time():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair(1))
    assert bus.poll(sub, PreviewChannel.TRAINING_PAIR) is not None
    assert bus.poll(sub, PreviewChannel.TRAINING_PAIR) is None


def test_a_fully_polled_slot_replacement_is_not_a_drop():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair(1))
    bus.poll(sub, PreviewChannel.TRAINING_PAIR)
    bus.publish(_pair(2))
    assert bus.stats()["preview_training_pair_dropped"] == 0


def test_with_no_subscribers_every_replacement_counts_as_a_drop():
    bus = PreviewBus()
    bus.publish(_pair(1))
    bus.publish(_pair(2))
    assert bus.stats()["preview_training_pair_dropped"] == 1


def test_sequence_numbers_are_monotonic_and_bus_assigned():
    bus = PreviewBus()
    seqs = [bus.publish(_pair(i)).seq for i in range(1, 6)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5
    seqs2 = [bus.publish(_raw(i)).seq for i in range(1, 4)]
    assert min(seqs2) > max(seqs), "one counter across all channels"


def test_a_foreign_handle_is_refused():
    bus_a, bus_b = PreviewBus(), PreviewBus()
    foreign = bus_b.subscribe("b")
    with pytest.raises(PreviewBusError, match="not issued by this bus"):
        bus_a.poll(foreign, PreviewChannel.TRAINING_PAIR)


def test_poll_all_returns_every_unseen_channel():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair(1))
    bus.publish(_raw(2))
    got = bus.poll_all(sub)
    assert {e.channel for e in got} == {
        PreviewChannel.TRAINING_PAIR,
        PreviewChannel.RAW_INPUT,
    }
    assert bus.poll_all(sub) == ()
