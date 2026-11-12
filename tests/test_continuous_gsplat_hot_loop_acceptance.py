"""Acceptance harness for the continuous gsplat prepared-source hot loop.

These are CPU/Torch contract tests, not GPU qualification.  They drive the
production ready-source owner and selector with real prepared leases and make
the forbidden host-wait path observable through a deliberately hostile event.
"""

from __future__ import annotations

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
from eligsir.runtime.ready_only_tick import (
    ReadyOnlyTickSelector,
    ReadySourceVersions,
)
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner


class _NonBlockingReadiness:
    """Completion probe: query is allowed, any host wait is a test failure."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.query_calls = 0
        self.wait_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        return self.ready

    def wait(self, *args, **kwargs):
        self.wait_calls += 1
        raise AssertionError("hot-loop selection must not wait for source readiness")


def _lease(ordinal: int, *, ready_event=None) -> PreparedSourceLease:
    source = SourceIdentity("hot-loop-camera", 0, ordinal, 1_000_000 + ordinal)
    value = torch.full((1, 1, 3), float(ordinal), dtype=torch.float32)
    payload = BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape),
            dtype="float32",
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding="rgb32f",
            device="cpu",
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=("hot-loop", ordinal),
    )
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"ordinal": ordinal},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        ready_event=ready_event,
    )


def _owner_with_replay(count: int = 4) -> tuple[ReadySourceOwner, object, tuple[PreparedSourceLease, ...]]:
    owner = ReadySourceOwner(max_fresh=2, max_replay=count, replay_reserve=count)
    producer = owner.claim_producer("decoder")
    leases = tuple(_lease(index) for index in range(count))
    for lease in leases:
        assert producer.offer(lease, lane=ReadySourceLane.REPLAY)
    return owner, producer, leases


def test_warm_replay_batch_remains_nonblocking_when_fresh_source_is_absent():
    owner, _producer, replay = _owner_with_replay()
    selector = ReadyOnlyTickSelector(owner, batch_size=4, warmup_complete=True)

    selection = selector.select(fresh_sources=())

    assert selection.leases == replay
    assert selection.replay_count == 4
    assert selection.fresh_count == 0
    assert selection.fallback_to_replay is True
    assert selection.starved is False
    assert selector.diagnostics.starvation_count == 0


def test_not_ready_fresh_candidate_is_skipped_and_ready_replay_is_selected():
    owner, producer, replay = _owner_with_replay()
    delayed = _NonBlockingReadiness(False)
    fresh = _lease(100, ready_event=delayed)
    assert producer.offer(fresh, lane=ReadySourceLane.FRESH)
    selector = ReadyOnlyTickSelector(owner, batch_size=1, warmup_complete=True)

    selection = selector.select(fresh_sources=(fresh.source_identity,))

    assert selection.leases == (replay[0],)
    assert selection.fresh_count == 0
    assert selection.replay_count == 1
    assert selection.fallback_to_replay is True
    assert delayed.query_calls > 0
    assert delayed.wait_calls == 0


def test_ready_fresh_candidate_wins_next_eligible_selection_and_versions_are_current():
    owner, producer, replay = _owner_with_replay()
    selector = ReadyOnlyTickSelector(owner, batch_size=1, warmup_complete=True)
    first = selector.select()
    selector.release(first)

    delayed = _NonBlockingReadiness(False)
    fresh = _lease(100, ready_event=delayed)
    assert producer.offer(fresh, lane=ReadySourceLane.FRESH)
    delayed.ready = True
    second = selector.select(fresh_sources=(fresh.source_identity,))

    assert second.leases == (fresh,)
    assert second.fresh_count == 1
    assert second.replay_count == 0
    assert second.fallback_to_replay is False
    selector.release(second)

    versions = ReadySourceVersions(map_content_version=7, topology_version=8, parameter_version=9)
    bindings = selector.bind_current_versions(second, versions)
    assert len(bindings) == 1
    assert bindings[0].lease is fresh
    assert bindings[0].versions is versions
    assert bindings[0].versions == ReadySourceVersions(7, 8, 9)


def test_ready_owner_fresh_priority_is_deterministic_when_replay_is_pending():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("decoder")
    replay = _lease(1)
    fresh = _lease(2)
    assert producer.offer(replay, lane=ReadySourceLane.REPLAY)
    assert producer.offer(fresh, lane=ReadySourceLane.FRESH)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)

    selection = selector.select(fresh_sources=(fresh.source_identity,))

    assert selection.leases == (fresh,)
    assert selection.fresh_count == 1
    assert selection.replay_count == 0
