from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from eligsir.contracts import (
    BufferRef,
    CameraIntrinsics,
    ModalityEligibility,
    SensorPacket,
    SourceIdentity,
)
from eligsir.contracts.control import (
    DispositionKind,
    HistoricalEligibility,
    ObservationDisposition,
    ResidencyIntent,
    SemanticAction,
)
from eligsir.mapping.admission.observation_owner import HistoricalPromotionPermit
from eligsir.runtime.prepared_ingress import (
    PreparedIngressControlToken,
    PreparedIngressItem,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.prepared_runtime_bridge import (
    PreparedRuntimeBridge,
    PreparedRuntimeBridgeResult,
)
from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker
from eligsir.runtime.prepared_source_producer import (
    PreparedSourceProduceResult,
    PreparedSourceProducer,
)
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.tracking.providers import observation_from_pose_matrix


class Gate:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class FakeTransfer:
    device = torch.device("cpu")

    def __init__(self, event: Gate | None = None) -> None:
        self.event = event or Gate()

    def upload_packed(self, arrays, *, purpose: str):
        assert purpose == "prepared_source"
        rgb, depth = tuple(arrays)
        return (
            (
                torch.from_numpy(np.asarray(rgb)),
                None if depth is None else torch.from_numpy(np.asarray(depth)),
            ),
            self.event,
        )


@dataclass
class FakeSlot:
    generation: int = 0

    def stage(self, rgb, depth):
        self.generation += 1
        return np.ascontiguousarray(rgb), None if depth is None else np.ascontiguousarray(depth)


class FakePool:
    def __init__(self, *, exhausted: bool = False) -> None:
        self.exhausted = exhausted
        self.slot = FakeSlot()

    def acquire(self):
        if self.exhausted:
            raise RuntimeError("pool exhausted")
        return self.slot

    def release_after(self, *_args) -> None:
        pass

    def release_now(self, *_args) -> None:
        pass


def packet(ordinal: int, *, depth: bool = True) -> SensorPacket:
    source = SourceIdentity("camera", 0, ordinal, 1_000_000_000 + ordinal)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.full((2, 3, 3), 128, dtype=np.uint8), encoding="rgb8"),
        depth=(
            BufferRef.from_numpy(
                np.full((2, 3), 1000, dtype=np.uint16),
                encoding="16UC1",
                units="mm",
            )
            if depth
            else None
        ),
        intrinsics=CameraIntrinsics(
            fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=2, depth_scale=1000.0
        ),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose"),
    )


def controlled_item(
    ordinal: int, action: SemanticAction, *, depth: bool = True
) -> PreparedIngressItem:
    source_packet = packet(ordinal, depth=depth)
    commit = action is SemanticAction.COMMIT
    disposition = ObservationDisposition(
        source=source_packet.source,
        snapshot_id=4,
        kind=DispositionKind.ADMIT if commit else DispositionKind.SKIP_REDUNDANT,
        observe=True,
        mapping_candidate=commit,
        admit_supervision=commit,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT if commit else ResidencyIntent.COLD,
        reason=action.value,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True, depth_supervision=depth
        ),
        semantic_action=action,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )
    token = PreparedIngressControlToken(ordinal, source_packet.source, 4, disposition)
    return PreparedIngressItem(
        ordinal,
        source_packet,
        source_packet.tracking,
        disposition=disposition,
        control_token=token,
    )


def bridge(*, handoff, owner, pool=None, clearer=None, per_tick: int = 2):
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=FakeTransfer(),
        frame_pool=pool or FakePool(),
        producer_owner="prepared-runtime-bridge",
    )
    return PreparedRuntimeBridge(
        handoff,
        producer,
        ready_owner=owner,
        consumer_owner="prepared-runtime-bridge",
        max_items_per_tick=per_tick,
        retained_commit_clearer=clearer,
    )


def test_uncontrolled_ingress_keeps_compatibility_fresh_route():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    source_packet = packet(3)
    item = PreparedIngressItem(3, source_packet, source_packet.tracking)
    assert handoff.claim_producer("session").try_offer(item).value == "enqueued"
    assert staged.drain() == PreparedRuntimeBridgeResult(1, 1, 0, 0)
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.FRESH)
    assert lease.source_identity == source_packet.source


def test_defer_promotion_uses_immutable_permit_without_runtime_attachment():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    item = controlled_item(8, SemanticAction.DEFER)
    permit = HistoricalPromotionPermit(item.packet.source, 4, 5, 6, 7)

    assert staged.submit_promoted(
        item, promotion_permit=permit, generation=7
    ).value == "enqueued"
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.REPLAY)
    assert lease.source_identity == item.packet.source
    assert lease.generation == 7


def test_completed_commit_promotion_uses_replay_lane():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    item = controlled_item(9, SemanticAction.COMMIT)

    assert staged.submit_promoted(item, promotion_permit=None, generation=7).value == "enqueued"
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.REPLAY)
    assert lease.source_identity == item.packet.source
    assert lease.generation == 7


def test_defer_promotion_rejects_wrong_immutable_permit():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    item = controlled_item(8, SemanticAction.DEFER)
    other = SourceIdentity("other", 0, 8, 8_000)
    permit = HistoricalPromotionPermit(other, 4, 5, 6, 7)
    assert staged.submit_promoted(
        item, promotion_permit=permit
    ).value == "ready_owner_drop"
    assert owner.claim_consumer("gpu").poll() is None


def test_promoted_batch_keeps_valid_sibling_when_one_permit_is_stale():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=2)
    staged = bridge(handoff=handoff, owner=owner)
    stale, valid = controlled_item(31, SemanticAction.DEFER), controlled_item(32, SemanticAction.DEFER)
    wrong_permit = HistoricalPromotionPermit(valid.packet.source, 4, 5, 6, 7)
    valid_permit = HistoricalPromotionPermit(valid.packet.source, 4, 5, 6, 7)

    assert staged.submit_promoted_batch(
        (stale, valid),
        promotion_permits=(wrong_permit, valid_permit),
        generations=(1, 1),
    ) == (
        PreparedSourceProduceResult.DROPPED_READY_OWNER,
        PreparedSourceProduceResult.ENQUEUED,
    )
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.REPLAY)
    assert lease is not None and lease.source_identity == valid.packet.source


def test_retained_batch_keeps_valid_commit_when_sibling_has_no_fresh_authority():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=2, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    cancelled, valid = controlled_item(41, SemanticAction.DEFER), controlled_item(42, SemanticAction.COMMIT)

    assert staged.submit_retained_commit_batch(
        (cancelled, valid), generations=(1, 1)
    ) == (
        PreparedSourceProduceResult.DROPPED_READY_OWNER,
        PreparedSourceProduceResult.ENQUEUED,
    )
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.FRESH)
    assert lease is not None and lease.source_identity == valid.packet.source


def test_controlled_commit_full_is_nonterminal_without_worker_semantic_callback():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(handoff=handoff, owner=owner)
    assert staged._producer.submit(
        packet(1),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        lane=ReadySourceLane.FRESH,
    ).value == "enqueued"
    item = controlled_item(2, SemanticAction.COMMIT)
    assert handoff.claim_producer("session").try_offer(item).value == "enqueued"
    result = staged.drain()
    assert result.retained_not_resident == 1
    assert result.dropped == 0


def test_rgb_only_accepted_commit_reaches_fresh_ready_without_depth_modalities():
    from eligsir.runtime.transfers import DeviceTransferOwner

    durable = []
    obligations = []
    handoff = PreparedSourceIngressHandoff(
        capacity=1,
        commit_sink=lambda item: (
            durable.append(item.packet.source),
            obligations.append(item.packet.source),
        ),
    )
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    producer = PreparedSourceProducer(
        owner,
        transfer_owner=DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=4096),
        producer_owner="prepared-runtime-bridge",
    )
    staged = PreparedRuntimeBridge(
        handoff,
        producer,
        ready_owner=owner,
        consumer_owner="prepared-runtime-bridge",
    )
    item = controlled_item(27, SemanticAction.COMMIT, depth=False)

    assert handoff.claim_producer("session").try_offer(item).value == "enqueued"
    worker = PreparedSourcePrepWorker(staged)

    assert worker.pump_once() is True
    assert worker.take_result() == PreparedRuntimeBridgeResult(1, 1, 0, 0)
    worker.check_health()
    assert durable == [item.packet.source]
    assert obligations == [item.packet.source]
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.FRESH)
    assert lease is not None
    assert lease.source_metadata["prepared_neutral_depth"] is True


def test_initial_commit_ready_entry_clears_obligation_only_after_supervision():
    cleared = []
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(
        handoff=handoff,
        owner=owner,
        clearer=lambda item: cleared.append(item.packet.source),
    )
    item = controlled_item(12, SemanticAction.COMMIT)
    assert handoff.claim_producer("session").try_offer(item).value == "enqueued"
    assert staged.drain().enqueued == 1
    assert cleared == []
    assert staged.complete_retained_commit_first_supervision(item.packet.source)
    assert cleared == [item.packet.source]
    assert not staged.complete_retained_commit_first_supervision(item.packet.source)


def test_retained_commit_reentry_uses_fresh_lane_and_generation():
    cleared = []
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    staged = bridge(
        handoff=handoff,
        owner=owner,
        clearer=lambda item: cleared.append(item.packet.source),
    )
    item = controlled_item(18, SemanticAction.COMMIT)
    assert staged.submit_retained_commit(item, generation=3).value == "enqueued"
    lease = owner.claim_consumer("gpu").poll(ReadySourceLane.FRESH)
    assert lease.source_identity == item.packet.source
    assert lease.generation == 3
    assert staged.complete_retained_commit_first_supervision(item.packet.source)
    assert cleared == [item.packet.source]


def test_pool_exhaustion_keeps_one_item_for_next_worker_drain():
    handoff = PreparedSourceIngressHandoff(capacity=1)
    owner = ReadySourceOwner(max_fresh=1, max_replay=1)
    pool = FakePool(exhausted=True)
    staged = bridge(handoff=handoff, owner=owner, pool=pool, per_tick=1)
    source_packet = packet(0)
    item = PreparedIngressItem(0, source_packet, source_packet.tracking)
    assert handoff.claim_producer("session").try_offer(item).value == "enqueued"
    assert staged.drain() == PreparedRuntimeBridgeResult(1, 0, 1, 0)
    pool.exhausted = False
    assert staged.drain() == PreparedRuntimeBridgeResult(1, 1, 0, 0)
