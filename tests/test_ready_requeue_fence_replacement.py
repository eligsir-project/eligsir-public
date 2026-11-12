"""READY requeue keeps one future readiness dependency per source."""

import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.ready_source import ReadySourceOwner


class _Gate:
    def __init__(self) -> None:
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return True


def _lease(event):
    token = object()
    return PreparedSourceLease(
        source_identity=SourceIdentity("requeue-test", 0, 1, 1_000),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={},
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "rgb": BufferRef(
                data=torch.zeros(1),
                spec=BufferSpec(
                    (1,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=token,
            )
        },
        device="cpu",
        ready_event=event,
        lifetime_token=token,
    )


def test_compute_requeue_replaces_satisfied_fence_without_history_growth():
    initial = _Gate()
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    assert producer.offer(_lease(initial), lane="replay")

    gates = []
    for _ in range(1_100):
        selected = consumer.select(1, fresh_limit=0)
        assert len(selected) == 1
        completion = _Gate()
        gates.append(completion)
        requeued = consumer.requeue_batch(selected, ready_event=completion)
        assert requeued[0].ready_event is completion

    assert initial.queries == 1
    assert all(gate.queries <= 1 for gate in gates)


def test_no_compute_requeue_preserves_existing_fence():
    initial = _Gate()
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decode")
    consumer = owner.claim_consumer("gpu")
    assert producer.offer(_lease(initial), lane="replay")

    selected = consumer.select(1, fresh_limit=0)
    requeued = consumer.requeue_batch(selected, ready_event=None)

    assert requeued[0].ready_event is initial
