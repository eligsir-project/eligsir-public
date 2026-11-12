import numpy as np
import pytest

from eligsir.contracts import BufferRef, CameraIntrinsics, SensorPacket, SourceIdentity
from eligsir.runtime.prepared_ingress import PreparedIngressItem, PreparedSourceIngressHandoff
from eligsir.tracking.providers import observation_from_pose_matrix


def _item(sequence: int) -> PreparedIngressItem:
    source = SourceIdentity("sink", 0, sequence, 1_000 + sequence)
    packet = SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), np.uint8), encoding="rgb8"),
        depth=None,
        intrinsics=CameraIntrinsics(
            fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2
        ),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="test"),
    )
    return PreparedIngressItem(sequence, packet, packet.tracking)


def test_commit_sink_runs_before_item_crosses_to_async_consumer():
    seen = []
    handoff = PreparedSourceIngressHandoff(
        capacity=1, commit_sink=lambda item: seen.append(item)
    )
    producer = handoff.claim_producer("session")
    consumer = handoff.claim_consumer("worker")
    item = _item(1)
    reservation = producer.reserve(item)
    assert reservation.commit(item).value == "enqueued"
    assert seen == [item]
    assert consumer.try_poll() is item


def test_failed_commit_sink_leaves_reservation_cancelable_and_queue_empty():
    failure = RuntimeError("semantic commit failed")
    handoff = PreparedSourceIngressHandoff(
        capacity=1,
        commit_sink=lambda _item: (_ for _ in ()).throw(failure),
    )
    producer = handoff.claim_producer("session")
    reservation = producer.reserve(_item(1))
    with pytest.raises(RuntimeError) as raised:
        reservation.commit(reservation.item)
    assert raised.value is failure
    reservation.cancel()
    assert handoff.depth == 0
