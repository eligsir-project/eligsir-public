"""Bounded HOT supervision reuse for canonical historical replay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import (
    CacheHandle,
    FrameObservationDB,
)
from eligsir.mapping.canonical_scene import CanonicalStreamingScene
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.transfers import DeviceTransferOwner


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("hot-cache-test", 3, ordinal, 1_000_000_000 + ordinal)


def _record(db: FrameObservationDB, source: SourceIdentity):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(source.source_ordinal)
    intrinsics = (4, 3, 8.0, 9.0, 1.5, 1.0)
    mapping = SimpleNamespace(
        source=source,
        device="cpu",
        rgb=SimpleNamespace(data=np.zeros((3, 4, 3), dtype=np.uint8)),
        depth=SimpleNamespace(data=np.ones((3, 4), dtype=np.float32)),
        intrinsics=intrinsics,
        reload_token=None,
    )
    return db.record_candidate(
        SimpleNamespace(c2w=pose),
        SimpleNamespace(
            index=source.source_ordinal,
            timestamp=source.timestamp_seconds,
            mapping_frame=mapping,
        ),
        source_identity=source,
        is_train=True,
        is_holdout=False,
        historical_eligible=True,
    )


def _cache(value: float, *, width: int = 4, height: int = 3) -> CacheHandle:
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    rgb, provenance = declare_canonical_score_tensor(
        torch.full((3, height, width), value, dtype=torch.float32),
        context="hot_supervision_test_fixture",
    )
    return CacheHandle(
        rgb=rgb,
        depth=torch.full((height, width), value + 1.0, dtype=torch.float32),
        intrinsics=(width, height, 8.0, 9.0, 1.5, 1.0),
        score_provenance=provenance,
    )


def _resident_db(*, bound: int = 2):
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=bound)
    first, second = _source(1), _source(2)
    _record(db, first)
    _record(db, second)
    first_cache, second_cache = _cache(0.1), _cache(0.2)
    assert db.activate_supervision(1, {"quarter": first_cache})
    assert db.activate_supervision(2, {"quarter": second_cache})
    return db, first, second, first_cache, second_cache


def test_exact_tier_resident_lookup_reuses_payload_pose_and_touches_lru():
    """Removing the lookup or its touch must reintroduce decode/LRU churn."""

    db, first, _second, first_cache, _second_cache = _resident_db()
    assert db.active_supervision_frame_ids == (1, 2)

    resident = db.resident_supervision_for_source(first, "quarter")

    assert resident is not None
    assert resident.source_identity == first.fingerprint()
    assert resident.tier == "quarter"
    assert resident.rgb is first_cache.rgb
    assert resident.depth is first_cache.depth
    assert resident.intrinsics is first_cache.intrinsics
    assert resident.pose_c2w[0][3] == 1.0
    assert db.active_supervision_frame_ids == (2, 1)


def test_tier_or_source_fence_mismatch_is_a_miss_without_lru_touch():
    """Using an adjacent tier or stale source identity would supervise wrong pixels."""

    db, first, _second, _first_cache, _second_cache = _resident_db()

    assert db.resident_supervision_for_source(first, "half") is None
    assert db.active_supervision_frame_ids == (1, 2)
    db.get_frame(1).source_identity = _source(99).fingerprint()
    assert db.resident_supervision_for_source(first, "quarter") is None
    assert db.active_supervision_frame_ids == (1, 2)


def test_invalid_tier_geometry_fails_closed_and_eviction_removes_hit():
    """Malformed or evicted payloads must rebuild from the deterministic token."""

    db = FrameObservationDB(device="cpu", max_active_supervision_frames=1)
    first, second = _source(1), _source(2)
    _record(db, first)
    _record(db, second)
    malformed = _cache(0.1, width=4, height=3)
    malformed.intrinsics = (8, 3, 8.0, 9.0, 1.5, 1.0)
    assert db.activate_supervision(1, {"quarter": malformed})
    assert db.resident_supervision_for_source(first, "quarter") is None

    assert db.activate_supervision(2, {"quarter": _cache(0.2)})
    assert db.resident_supervision_for_source(first, "quarter") is None
    assert db.active_supervision_frame_ids == (2,)


def test_runtime_resident_projection_skips_decoder_and_preserves_tensor_identity(
    monkeypatch,
):
    """A HOT replay must not recreate pixels or submit a replacement payload."""

    db, first, _second, first_cache, _second_cache = _resident_db()
    resident = db.resident_supervision_for_source(first, "quarter")
    assert resident is not None
    transfer_owner = DeviceTransferOwner(
        "cpu", max_inflight=1, max_pinned_bytes=1024
    )
    scene = object.__new__(CanonicalStreamingScene)
    scene._device_transfer_owner = transfer_owner
    scene._admitted_cameras_by_source = {}
    scene.train_cameras = []
    scene.current_camera = None
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.scene = scene
    runtime._source_identities_by_ordinal = {}
    runtime._observation_db = SimpleNamespace(
        historical_candidate_for_source=lambda source, **_kwargs: SimpleNamespace(
            source_identity=source.fingerprint(),
            disposition_snapshot_id=4,
            semantic_action="defer",
        )
    )
    snapshot = SimpleNamespace(snapshot_id=5, topology_version=7)
    runtime.observation_owner = SimpleNamespace(
        promotion_permit_from_retained=lambda **kwargs: SimpleNamespace(
            source=kwargs["source"],
            promotion_snapshot_id=kwargs["current_snapshot"].snapshot_id,
            topology_version=kwargs["current_snapshot"].topology_version,
        )
    )
    runtime._admitted_clock_ns = 0
    runtime._work_clock_owner = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(optimizer_updates=0)
    )
    runtime._optimization_runtime = SimpleNamespace(
        note_admitted_source=lambda **_kwargs: None
    )
    runtime._services = SimpleNamespace(four_clocks=None)
    monkeypatch.setattr(
        runtime,
        "_replay_mapping_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HOT replay called the cold decoder path")
        ),
    )

    resolved = runtime._camera_for_source(
        first,
        current_snapshot=snapshot,
        resident_supervision=resident,
    )
    runtime._bind_resident_supervision(
        resolved, first, ResolutionTier.QUARTER, resident
    )

    assert resolved.original_image is first_cache.rgb
    assert resolved._sensor_depth_cache is first_cache.depth
    assert (resolved.image_width, resolved.image_height) == (4, 3)
    assert (resolved.fx, resolved.fy, resolved.cx, resolved.cy) == (
        8.0,
        9.0,
        1.5,
        1.0,
    )
    assert resolved.gsplat is None
    assert resolved._canonical_native_resolution_state is None
    assert resolved._canonical_resolution_binding_native_state is None
    assert transfer_owner.prefetch_cameras((resolved,)) == 0
    assert transfer_owner.accounting().h2d_ops == 0


def test_resident_bind_preserves_admitted_camera_native_lineage_across_tier_miss():
    """A quarter HOT hit must not become the native source for an eighth miss."""

    from eligsir.mapping.resolution_materialization import (
        materialize_resolution_tier,
    )
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    source = _source(7)
    native_rgb, native_provenance = declare_canonical_score_tensor(
        torch.zeros((3, 480, 640), dtype=torch.float32),
        context="hot_supervision_native_lineage_regression",
    )
    camera = SimpleNamespace(
        _source_identity=source,
        original_image=native_rgb,
        _sensor_depth_cache=torch.ones((480, 640), dtype=torch.float32),
        image_width=640,
        image_height=480,
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=319.5,
        cy=239.5,
        score_provenance=native_provenance,
        gsplat=object(),
    )
    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    canonical_native = camera._canonical_native_resolution_state
    resident = SimpleNamespace(
        source_identity=source.fingerprint(),
        tier=ResolutionTier.QUARTER.value,
        rgb=camera.original_image,
        depth=camera._sensor_depth_cache,
        intrinsics=(
            camera.image_width,
            camera.image_height,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
        ),
        score_provenance=camera.score_provenance,
    )

    CanonicalMappingRuntime._bind_resident_supervision(
        camera, source, ResolutionTier.QUARTER, resident
    )
    materialize_resolution_tier(camera, ResolutionTier.EIGHTH)

    assert (camera.image_width, camera.image_height) == (80, 60)
    assert tuple(camera.original_image.shape) == (3, 60, 80)
    assert camera._canonical_native_resolution_state is canonical_native


def test_record_candidate_never_prebuilds_a_resolution_cache():
    """`CanonicalMappingRuntime`'s ONLY admission-time producer is
    `record_candidate` -- it must stay a no-op for `resolution_cache` (issue
    #279 follow-up / #1259 audit). `record_frame_observation`'s eager
    ``{"high": ..., "low": ...}`` builder (`train_streaming.py`,
    `eligsir/mapping/resolution_cache.py::build_resolution_cache`) belongs
    to a DIFFERENT, disjoint trainer (`--execution_trainer offline_reconstruction`
    / the legacy streaming loop's `bind_resolution_tier`) that never runs
    inside `causal_mapping` and never shares a `FrameObservationDB` instance
    with it (`CanonicalMappingRuntime.__init__` constructs its own, with no
    `resolution_cache_rebuilder`). If a future change makes `record_candidate`
    (or any other `causal_mapping` admission path) start eagerly building a
    cache, it MUST use `ResolutionTier` values (native/half/quarter/eighth) --
    reusing the legacy "high"/"low" keys here would silently resurrect the
    exact structurally-dead-cache shape this test guards against.
    """
    db = FrameObservationDB(device="cpu")
    source = _source(1)
    record = _record(db, source)
    assert record.resolution_cache == {}
    assert record.resolution_cache_absent_reason == "not_activated"
    assert db.resident_supervision_for_source(source, "quarter") is None


def test_legacy_high_low_tier_key_never_satisfies_resolution_tier_lookup():
    """Pin the exact key-space mismatch: a "high"/"low"-keyed cache entry
    (the shape `build_resolution_cache`/`record_frame_observation` produces
    for the legacy `bind_resolution_tier` trainer) is invisible to
    `resident_supervision_for_source`, which is only ever queried with
    `ResolutionTier.value` strings (native/half/quarter/eighth) by
    `CanonicalMappingRuntime` (`production_application.py`'s sole production
    caller). This is not itself a live bug (the two producers/consumers never
    share a DB instance -- see the sibling test above) but pins the
    contract so a future merge of the two tier-naming schemes cannot silently
    reintroduce it. See docs/architecture/2026-08-31-issue279-camera-materialize-cache-audit.md.
    """
    db = FrameObservationDB(device="cpu")
    source = _source(2)
    _record(db, source)
    legacy_high = _cache(0.5)
    assert db.attach_resolution_cache(2, {"high": legacy_high})
    for tier in ("native", "half", "quarter", "eighth"):
        assert db.resident_supervision_for_source(source, tier) is None
    # The legacy key IS resident (a "high"-tier reader would see it) --
    # only the ResolutionTier-keyed reader is blind to it.
    assert db.get_frame(2).resolution_cache.get("high") is legacy_high


def test_transaction_local_cache_camera_is_not_registered_for_wasted_page_out():
    """Only scene-owned cameras need the next-tick residency release.

    A promoted historical camera is transaction-local and merely borrows the
    HOT tensor owned by ``FrameObservationDB``. Registering that temporary
    camera makes the ledger copy the tensor to CPU on the next tick without
    releasing the DB's GPU owner.
    """
    source = _source(9)
    scene_camera = object()
    transaction_camera = object()
    paged_in = []
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime.scene = SimpleNamespace(
        admitted_camera_for_source=lambda candidate: (
            scene_camera if candidate == source else None
        )
    )
    runtime._residency_ledger = SimpleNamespace(note_paged_in=paged_in.append)

    runtime._note_scene_camera_page_in(source, scene_camera)
    runtime._note_scene_camera_page_in(source, transaction_camera)

    assert paged_in == [scene_camera]
