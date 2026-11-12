from __future__ import annotations

import pytest
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
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_only_tick import (
    ReadyOnlyTickSelector,
    ReadySourceVersions,
)


class Gate:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


def identity(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000)


def lease(ordinal: int, *, event=None) -> PreparedSourceLease:
    token = object()
    payload = BufferRef(
        data=torch.zeros(2),
        spec=BufferSpec((2,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
        ownership=BufferOwnership.BORROWED,
        owner_token=token,
    )
    return PreparedSourceLease(
        source_identity=identity(ordinal),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 8, "height": 6},
        tier_preparation_available=("native",),
        borrowed_device_payloads={"rgb": payload},
        device="cpu",
        ready_event=event,
    )


def offer(owner: ReadySourceOwner, *leases: PreparedSourceLease) -> None:
    producer = owner.claim_producer("preparer")
    for item in leases:
        assert producer.offer(
            item,
            lane=ReadySourceLane.REPLAY,
            arrival_ordinal=item.source_identity.source_ordinal,
        )


def test_delayed_fresh_does_not_block_and_ready_replay_is_selected():
    delayed = lease(8, event=Gate())
    replay = lease(2)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("preparer")
    assert producer.offer(delayed, lane=ReadySourceLane.FRESH, arrival_ordinal=8)
    assert producer.offer(replay, lane=ReadySourceLane.REPLAY, arrival_ordinal=2)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)

    selection = selector.select(fresh_sources=(delayed.source_identity,))

    assert selection.leases == (replay,)
    assert selection.fallback_to_replay is True
    assert selection.starved is False


def test_unready_candidate_is_not_eligible_even_if_it_is_named_fresh():
    unready = lease(8, event=Gate())
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = owner.claim_producer("preparer")
    assert producer.offer(unready, lane=ReadySourceLane.FRESH, arrival_ordinal=8)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)

    selection = selector.select(fresh_sources=(unready.source_identity,))

    assert selection.leases == ()
    assert selection.starved is False


def test_ready_replay_reserve_runs_when_no_new_source_is_available():
    replay = tuple(lease(ordinal) for ordinal in (0, 1, 2, 3))
    owner = ReadySourceOwner(max_fresh=1, max_replay=4)
    offer(owner, *replay)
    selector = ReadyOnlyTickSelector(owner, batch_size=4, warmup_complete=True)

    selection = selector.select()

    assert tuple(item.source_identity.source_ordinal for item in selection.leases) == (0, 1, 2, 3)
    assert selection.fallback_to_replay is True
    assert selection.starved is False


def test_short_ready_reserve_reports_bounded_starvation_without_waiting():
    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    offer(owner, lease(0), lease(1))
    selector = ReadyOnlyTickSelector(owner, batch_size=4, warmup_complete=True)

    selection = selector.select()

    assert len(selection.leases) == 2
    assert selection.starved is True
    assert selector.diagnostics.starvation_count == 1


def test_jit_binding_samples_current_versions_after_ready_selection():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    offer(owner, lease(0))
    selector = ReadyOnlyTickSelector(owner, batch_size=1)
    selection = selector.select()
    versions = ReadySourceVersions(
        map_content_version=7,
        topology_version=11,
        parameter_version=13,
    )

    bound = selector.bind_current_versions(selection, versions)

    assert len(bound) == 1
    assert bound[0].lease is selection.leases[0]
    assert bound[0].versions is versions
    selector.release(selection)
    assert owner.counters.released == 1


def test_jit_binding_rejects_non_exact_version_values():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    offer(owner, lease(0))
    selector = ReadyOnlyTickSelector(owner, batch_size=1)
    selection = selector.select()

    with pytest.raises(TypeError, match="map_content_version"):
        selector.bind_current_versions(
            selection,
            ReadySourceVersions(
                map_content_version=True,
                topology_version=0,
                parameter_version=0,
            ),
        )


def test_canonical_runtime_exposes_explicit_ready_only_owner_hook():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    item = lease(0)
    producer = owner.claim_producer("preparer")
    assert producer.offer(item, lane=ReadySourceLane.REPLAY, arrival_ordinal=0)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 1
    runtime._ready_source_owner = None
    runtime._ready_only_tick_selector = None
    runtime._versions = lambda: (7, 11, 13)
    runtime.enable_ready_source_owner(owner)

    assert runtime.select_ready_only_sources().leases == ()
    selection = runtime.select_ready_only_sources(
        replay_sources=(item.source_identity,)
    )
    bound = runtime.bind_ready_only_sources(selection)

    assert selection.leases == (item,)
    assert bound[0].lease is item
    assert bound[0].versions == ReadySourceVersions(7, 11, 13)
    runtime.release_ready_only_sources(selection)
    assert owner.counters.released == 1


def test_requested_replay_filter_prevents_arbitrary_ready_replay_selection():
    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    first, requested = lease(1), lease(2)
    offer(owner, first, requested)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)

    selection = selector.select(replay_sources=(requested.source_identity,))

    assert selection.leases == (requested,)


def test_empty_explicit_replay_filter_does_not_fill_with_an_unrequested_replay():
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    replay = lease(2)
    offer(owner, replay)
    selector = ReadyOnlyTickSelector(owner, batch_size=1)

    selection = selector.select(fresh_sources=(), replay_sources=())

    assert selection.leases == ()
