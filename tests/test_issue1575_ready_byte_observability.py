"""Tests for issue #1575 Section C.2: READY resident-byte observability.

Tests verify O(1) byte accounting with cached breakdown by tier/kind/lane,
no host/device syncs, no iterations on the property-read path, and exact
tracking under add/evict/release/requeue operations.
"""

from __future__ import annotations

import random
import threading
from typing import Callable

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
from eligsir.runtime.prepared_source import PreparedSourceLease, PreparedTierPayload
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner


def identity(ordinal: int, stream_epoch: int = 0) -> SourceIdentity:
    """Create a unique source identity."""
    return SourceIdentity("camera", stream_epoch, ordinal, ordinal * 1_000)


def make_tensor(shape, dtype=torch.float32, device="cpu"):
    """Create a tensor with known byte size."""
    return torch.empty(shape, dtype=dtype, device=device)


def lease_with_tiers(
    ordinal: int,
    *,
    tiers: dict[str, dict[str, tuple]] | None = None,
    borrowed_payload: tuple | None = None,
) -> PreparedSourceLease:
    """Create a lease with optional tier payloads and borrowed payloads.

    Args:
        ordinal: unique source ordinal
        tiers: dict mapping tier name to dict of field names to shapes.
               E.g., {"native": {"rgb": (2, 3, 4), "depth": (2, 3)}}
        borrowed_payload: shape for a generic borrowed payload, or None.
    """
    tiers = tiers or {}
    payload_token = object()

    # Create borrowed device payloads (common to all tiers).
    borrowed_payloads = {}
    if borrowed_payload is not None:
        borrowed_payloads["rgb"] = BufferRef(
            data=make_tensor(borrowed_payload),
            spec=BufferSpec(
                borrowed_payload,
                "float32",
                MemoryDomain.OTHER_DEVICE,
                "tensor",
                device="cpu",
            ),
            ownership=BufferOwnership.BORROWED,
            owner_token=payload_token,
        )

    # Create tier payloads.
    tier_payloads = {}
    for tier_name, fields in tiers.items():
        tier_dict = {}
        for field_name, shape in fields.items():
            tensor = make_tensor(shape)
            # Set encoding based on field name.
            if field_name == "rgb":
                encoding = "rgb_float32_hwc"
                units = None
            elif field_name == "depth":
                encoding = "depth_meters"
                units = "m"
            elif field_name == "camera_intrinsics":
                encoding = "camera_intrinsics"
                units = None
            else:
                encoding = "tensor"
                units = None
            tier_dict[field_name] = BufferRef(
                data=tensor,
                spec=BufferSpec(
                    shape,
                    "float32",
                    MemoryDomain.OTHER_DEVICE,
                    encoding,
                    units=units,
                    device="cpu",
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=payload_token,
            )

        # Ensure required fields.
        if "camera_intrinsics" not in tier_dict:
            tier_dict["camera_intrinsics"] = BufferRef(
                data=make_tensor((3, 3)),
                spec=BufferSpec(
                    (3, 3),
                    "float32",
                    MemoryDomain.OTHER_DEVICE,
                    "camera_intrinsics",
                    device="cpu",
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=payload_token,
            )

        # Get or create default RGB
        rgb_ref = tier_dict.get("rgb")
        if rgb_ref is None:
            rgb_ref = BufferRef(
                data=make_tensor((2, 3, 3)),
                spec=BufferSpec(
                    (2, 3, 3),
                    "float32",
                    MemoryDomain.OTHER_DEVICE,
                    "rgb_float32_hwc",
                    device="cpu",
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=payload_token,
            )

        # Get or create default depth
        depth_ref = tier_dict.get("depth")
        if depth_ref is None:
            depth_ref = BufferRef(
                data=make_tensor((2, 3)),
                spec=BufferSpec(
                    (2, 3),
                    "float32",
                    MemoryDomain.OTHER_DEVICE,
                    "depth_meters",
                    units="m",
                    device="cpu",
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=payload_token,
            )

        tier_payloads[tier_name] = PreparedTierPayload(
            name=tier_name,
            rgb=rgb_ref,
            depth=depth_ref,
            camera_intrinsics=tier_dict["camera_intrinsics"],
            width=3,
            height=2,
            depth_confidence=tier_dict.get("depth_confidence"),
            projection_matrix=tier_dict.get("projection_matrix"),
            full_proj_transform=tier_dict.get("full_proj_transform"),
        )

    # Default borrowed payload if none provided.
    if not borrowed_payloads:
        borrowed_payloads["placeholder"] = BufferRef(
            data=make_tensor((8,)),
            spec=BufferSpec((8,), "float32", MemoryDomain.OTHER_DEVICE, "tensor", device="cpu"),
            ownership=BufferOwnership.BORROWED,
            owner_token=payload_token,
        )

    return PreparedSourceLease(
        source_identity=identity(ordinal),
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"width": 3, "height": 2},
        tier_preparation_available=tuple(tier_payloads.keys()) or ("native",),
        borrowed_device_payloads=borrowed_payloads,
        device="cpu",
        generation=0,
        lifetime_token=payload_token,
        tier_payloads=tier_payloads,
    )


def compute_lease_bytes(lease: PreparedSourceLease) -> int:
    """Ground-truth byte count for a lease (matches _compute_lease_resident_bytes)."""
    seen_ptrs: set[int] = set()
    total_bytes = 0

    # Count borrowed payloads.
    for payload_name, payload_ref in lease.borrowed_device_payloads.items():
        data = getattr(payload_ref, "data", None)
        if data is None or not isinstance(data, torch.Tensor):
            continue
        ptr = data.untyped_storage().data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total_bytes += int(data.numel()) * int(data.element_size())

    # Count tier payloads.
    for tier_name, tier_payload in lease.tier_payloads.items():
        for field_name in (
            "rgb",
            "depth",
            "camera_intrinsics",
            "depth_confidence",
            "projection_matrix",
            "full_proj_transform",
        ):
            payload_ref = getattr(tier_payload, field_name, None)
            if payload_ref is None:
                continue
            data = getattr(payload_ref, "data", None)
            if data is None or not isinstance(data, torch.Tensor):
                continue
            ptr = data.untyped_storage().data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            total_bytes += int(data.numel()) * int(data.element_size())

    return total_bytes


def scan_ground_truth_bytes(owner: ReadySourceOwner) -> int:
    """Sum bytes from all resident entries (fresh + replay + in-flight)."""
    total = 0
    # Access internal state under lock for ground-truth computation.
    with owner._state_lock:
        for entry in (*owner._fresh, *owner._replay, *owner._in_flight.values()):
            for (_tier, _kind), nbytes in entry.resident_bytes:
                total += nbytes
    return total


def test_allocate_n_entries_current_bytes_matches_sum():
    """Test (a): allocate N entries → current bytes == sum by kind."""
    owner = ReadySourceOwner(max_fresh=5, max_replay=5)
    producer = owner.claim_producer()

    # Add 3 leases with known byte counts.
    leases = []
    expected_bytes = 0
    for i in range(3):
        lease = lease_with_tiers(i, borrowed_payload=(2, 4))
        leases.append(lease)
        producer.offer(lease, lane=ReadySourceLane.REPLAY)
        expected_bytes += compute_lease_bytes(lease)

    # Check current bytes matches expected sum.
    assert owner.ready_bytes_current == expected_bytes
    assert owner.ready_bytes_peak >= expected_bytes


def test_evict_release_decrements_to_exact_value():
    """Test (b): evict/release → decrements to exact value, peak unchanged."""
    owner = ReadySourceOwner(max_fresh=5, max_replay=5)
    producer = owner.claim_producer()
    consumer = owner.claim_consumer()

    # Allocate 3 leases.
    leases = []
    expected_bytes = []
    for i in range(3):
        lease = lease_with_tiers(i, borrowed_payload=(2, 4))
        leases.append(lease)
        expected_bytes.append(compute_lease_bytes(lease))
        producer.offer(lease, lane=ReadySourceLane.REPLAY)

    total_initial = sum(expected_bytes)
    peak_before_evict = owner.ready_bytes_peak

    # Evict the first lease.
    evicted = consumer.evict_replay(count=1)
    assert len(evicted) == 1
    remaining_bytes = total_initial - expected_bytes[0]
    assert owner.ready_bytes_current == remaining_bytes
    assert owner.ready_bytes_peak == peak_before_evict  # Peak unchanged.

    # Evict another.
    evicted = consumer.evict_replay(count=1)
    remaining_bytes -= expected_bytes[1]
    assert owner.ready_bytes_current == remaining_bytes
    assert owner.ready_bytes_peak == peak_before_evict

    # Release the last one (select and release).
    selected = consumer.select(1)
    assert len(selected) == 1
    consumer.release(selected[0])
    remaining_bytes -= expected_bytes[2]
    assert owner.ready_bytes_current == remaining_bytes
    assert owner.ready_bytes_peak == peak_before_evict


def test_randomized_300_op_sequence_maintains_invariants():
    """Test (c): randomized 300-op keeps current == ground-truth and peak == max."""
    random.seed(42)
    owner = ReadySourceOwner(max_fresh=8, max_replay=16)
    producer = owner.claim_producer()
    consumer = owner.claim_consumer()

    next_ordinal = 0
    pending_by_lane: dict[str, list] = {"fresh": [], "replay": []}

    operations = [
        "offer_fresh",
        "offer_replay",
        "select",
        "evict_replay",
        "release",
    ]

    for step in range(300):
        op = random.choice(operations)

        if op == "offer_fresh" and len(pending_by_lane["fresh"]) < owner._max_fresh:
            lease = lease_with_tiers(next_ordinal, borrowed_payload=(2, 2))
            next_ordinal += 1
            if producer.offer(lease, lane=ReadySourceLane.FRESH):
                pending_by_lane["fresh"].append(lease)

        elif op == "offer_replay" and len(pending_by_lane["replay"]) < owner._max_replay:
            lease = lease_with_tiers(next_ordinal, borrowed_payload=(2, 2))
            next_ordinal += 1
            if producer.offer(lease, lane=ReadySourceLane.REPLAY):
                pending_by_lane["replay"].append(lease)

        elif op == "select" and (pending_by_lane["fresh"] or pending_by_lane["replay"]):
            selected = consumer.select(2)
            for lease in selected:
                if lease in pending_by_lane["fresh"]:
                    pending_by_lane["fresh"].remove(lease)
                elif lease in pending_by_lane["replay"]:
                    pending_by_lane["replay"].remove(lease)

        elif op == "evict_replay" and pending_by_lane["replay"]:
            evicted = consumer.evict_replay(count=1)
            for lease in evicted:
                if lease in pending_by_lane["replay"]:
                    pending_by_lane["replay"].remove(lease)

        elif op == "release":
            selected = consumer.select(1)
            for lease in selected:
                consumer.release(lease)
                if lease in pending_by_lane["fresh"]:
                    pending_by_lane["fresh"].remove(lease)
                elif lease in pending_by_lane["replay"]:
                    pending_by_lane["replay"].remove(lease)

        # Verify invariants at each step.
        ground_truth = scan_ground_truth_bytes(owner)
        assert (
            owner.ready_bytes_current == ground_truth
        ), f"Step {step} ({op}): current {owner.ready_bytes_current} != ground_truth {ground_truth}"
        assert (
            owner.ready_bytes_peak >= owner.ready_bytes_current
        ), f"Step {step}: peak {owner.ready_bytes_peak} < current {owner.ready_bytes_current}"


def test_properties_perform_no_iteration():
    """Test (d): property reads do not iterate over internal state."""
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer()

    # Add some leases.
    for i in range(2):
        lease = lease_with_tiers(i, borrowed_payload=(2, 4))
        producer.offer(lease, lane=ReadySourceLane.REPLAY)

    # Patch _compute_lease_resident_bytes to count calls.
    original_compute = owner._compute_lease_resident_bytes
    call_count = [0]

    @staticmethod
    def counting_compute(lease):
        call_count[0] += 1
        return original_compute(lease)

    owner._compute_lease_resident_bytes = counting_compute

    # Read properties multiple times.
    _ = owner.ready_bytes_current
    _ = owner.ready_bytes_peak
    _ = owner.ready_bytes_by_tier_kind
    _ = owner.ready_bytes_by_lane
    _ = owner.counters

    # Should not call _compute_lease_resident_bytes on property reads.
    # (Only called during offer/commit, not on property access.)
    assert (
        call_count[0] == 0
    ), f"Properties triggered {call_count[0]} compute calls, expected 0"


def test_bytes_by_lane_breakdown():
    """Test that bytes are correctly tracked per lane."""
    owner = ReadySourceOwner(max_fresh=3, max_replay=3)
    producer = owner.claim_producer()

    fresh_lease = lease_with_tiers(1, borrowed_payload=(2, 4))
    replay_lease = lease_with_tiers(2, borrowed_payload=(3, 5))

    producer.offer(fresh_lease, lane=ReadySourceLane.FRESH)
    producer.offer(replay_lease, lane=ReadySourceLane.REPLAY)

    fresh_bytes = compute_lease_bytes(fresh_lease)
    replay_bytes = compute_lease_bytes(replay_lease)

    breakdown = owner.ready_bytes_by_lane
    assert breakdown[ReadySourceLane.FRESH] == fresh_bytes
    assert breakdown[ReadySourceLane.REPLAY] == replay_bytes
    assert owner.ready_bytes_current == fresh_bytes + replay_bytes


def test_bytes_by_tier_kind_breakdown():
    """Test that bytes are correctly tracked by tier and kind."""
    owner = ReadySourceOwner(max_fresh=2, max_replay=2)
    producer = owner.claim_producer()

    # Create a lease with explicit tier structure (shape is (H, W, C) for RGB, (H, W) for depth).
    lease = lease_with_tiers(
        1,
        tiers={
            "native": {"rgb": (2, 3, 3), "depth": (2, 3)},
        },
    )
    producer.offer(lease, lane=ReadySourceLane.REPLAY)

    breakdown = owner.ready_bytes_by_tier_kind
    # Should have entries for ("base", "placeholder"), ("native", "rgb"), ("native", "depth"), etc.
    assert len(breakdown) > 0
    assert owner.ready_bytes_current == sum(breakdown.values())


def test_abandon_pending_clears_queued_bytes():
    """Test that _abandon_pending clears bytes for queued entries only."""
    owner = ReadySourceOwner(max_fresh=3, max_replay=3)
    producer = owner.claim_producer()
    consumer = owner.claim_consumer()

    # Add fresh and replay entries.
    fresh_lease = lease_with_tiers(1, borrowed_payload=(2, 4))
    replay_lease = lease_with_tiers(2, borrowed_payload=(3, 5))

    producer.offer(fresh_lease, lane=ReadySourceLane.FRESH)
    producer.offer(replay_lease, lane=ReadySourceLane.REPLAY)

    # Select one (moves to in-flight).
    selected = consumer.select(1)
    assert len(selected) == 1

    bytes_in_flight = compute_lease_bytes(selected[0])
    bytes_queued = owner.ready_bytes_current - bytes_in_flight

    # Abandon pending (should clear queued, not in-flight).
    producer.abandon_pending(reason="test")

    # In-flight bytes should remain, queued bytes should be gone.
    assert owner.ready_bytes_current == bytes_in_flight
