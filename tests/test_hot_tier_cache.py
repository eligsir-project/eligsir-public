"""Bounded, lazy, byte-budgeted multi-tier HOT resolution cache.

Covers the seven invariants the multi-tier retention change rests on:

1. half/quarter are always derived from the canonical NATIVE view, never
   chained off another materialized tier, across repeated hits and misses;
2. lazy materialization -- a frame requested at one tier never grows a pyramid;
3. bounded per-frame retention -- a third distinct tier evicts, never grows;
4. byte-budget enforcement across the whole HOT set;
5. value-based (not LRU) eviction ordering, as a direct test of the comparator;
6. HOT-hit and cold-miss recorded as distinguishable cost samples;
7. an extreme regime where a nominally "cheaper" tier MEASURES worse, and the
   plumbing reports the measured cost rather than an assumed ordinal cheapness.

None of these need a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.hot_tier_cache import (
    EVICT_BYTE_BUDGET,
    EVICT_PER_FRAME_CAP,
    HotTierCache,
    HotTierEntry,
    entry_bytes,
)
from eligsir.mapping.observation_db import (
    BufferStage,
    CacheHandle,
    FrameObservationDB,
    FrameRecord,
)
from eligsir.mapping.resolution_materialization import materialize_resolution_tier
from eligsir.mapping.scheduler.tier_cost_model import (
    RESIDENCY_COLD_MISS,
    RESIDENCY_HOT_HIT,
    TierCostContext,
    TierCostEstimator,
    TierCostSample,
    residency_class,
)

_MIB = 1024 * 1024


# --------------------------------------------------------------------------
# 1. native-only derivation
# --------------------------------------------------------------------------


class _Camera:
    def __init__(self, height: int = 64, width: int = 96) -> None:
        rng = np.random.default_rng(0)
        self.original_image = torch.from_numpy(
            rng.random((3, height, width), dtype=np.float32)
        )
        self._sensor_depth_cache = torch.from_numpy(
            rng.random((height, width), dtype=np.float32)
        )
        self.image_height = self.height = height
        self.image_width = self.width = width
        self.fx, self.fy, self.cx, self.cy = 100.0, 100.0, width / 2.0, height / 2.0
        self.score_provenance = None
        self.gsplat = None


def _derive_from_native(native_rgb: torch.Tensor, tier: ResolutionTier) -> torch.Tensor:
    camera = _Camera(int(native_rgb.shape[1]), int(native_rgb.shape[2]))
    camera.original_image = native_rgb.clone()
    materialize_resolution_tier(camera, tier)
    return camera.original_image


def test_tiers_are_always_derived_from_native_never_chained():
    """quarter == resize(native), NOT resize(resize(native))."""

    camera = _Camera()
    native = camera.original_image.clone()

    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    direct_quarter = camera.original_image.clone()
    assert torch.allclose(direct_quarter, _derive_from_native(native, ResolutionTier.QUARTER))

    # The derivation reads the memoized canonical NATIVE view, not whatever
    # the camera happened to be bound to a moment ago.
    assert camera._canonical_native_resolution_state.rgb is not None
    assert torch.allclose(camera._canonical_native_resolution_state.rgb, native)
    assert camera._canonical_native_resolution_state.rgb.shape == native.shape

    # Decisive: bind to HALF, then DESTROY the currently-bound payload in
    # place. A chained pyramid would derive quarter from that corrupted
    # intermediate; deriving from the retained native view is unaffected.
    # (A numeric direct-vs-chained comparison cannot show this: at exact
    # power-of-two factors both area and nearest resizes happen to be
    # associative, so the two agree by arithmetic coincidence.)
    fresh = _Camera()
    fresh_native = fresh.original_image.clone()
    materialize_resolution_tier(fresh, ResolutionTier.HALF)
    fresh.original_image.zero_()
    fresh._sensor_depth_cache.zero_()
    materialize_resolution_tier(fresh, ResolutionTier.QUARTER)
    assert torch.allclose(
        fresh.original_image, _derive_from_native(fresh_native, ResolutionTier.QUARTER)
    )
    assert float(fresh.original_image.abs().sum()) > 0.0


def test_native_derivation_survives_repeated_tier_oscillation():
    """quarter -> half -> quarter re-derives from native every time."""

    camera = _Camera()
    native = camera.original_image.clone()
    expected_quarter = _derive_from_native(native, ResolutionTier.QUARTER)
    expected_half = _derive_from_native(native, ResolutionTier.HALF)

    for _ in range(3):
        materialize_resolution_tier(camera, ResolutionTier.QUARTER)
        assert torch.allclose(camera.original_image, expected_quarter)
        materialize_resolution_tier(camera, ResolutionTier.HALF)
        assert torch.allclose(camera.original_image, expected_half)
    materialize_resolution_tier(camera, ResolutionTier.NATIVE)
    assert torch.allclose(camera.original_image, native)


# --------------------------------------------------------------------------
# DB-level fixtures
# --------------------------------------------------------------------------


def _handle(height: int, width: int, *, with_depth: bool = True) -> CacheHandle:
    return CacheHandle(
        rgb=torch.zeros((3, height, width), dtype=torch.float32),
        depth=torch.zeros((height, width), dtype=torch.float32) if with_depth else None,
        intrinsics=(width, height, 100.0, 100.0, width / 2.0, height / 2.0),
        score_provenance="test",
    )


def _db(**kwargs) -> FrameObservationDB:
    return FrameObservationDB(device="cpu", **kwargs)


def _add_frame(db: FrameObservationDB, frame_id: int) -> None:
    db.add_frame(
        FrameRecord(
            frame_id=frame_id,
            timestamp=float(frame_id),
            pose_c2w=np.eye(4, dtype=np.float32),
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            submap_id=0,
            buffer_stage=BufferStage.MAPPING_KEYFRAME,
            source_identity=f"source-{frame_id}",
        )
    )
    # `add_frame` is the low-level record path and does not index source
    # identities (only `record_candidate` does); index it so the HOT lookup
    # `resident_supervision_for_source` is reachable in these tests.
    db._source_identity_to_frame_id[f"source-{frame_id}"] = frame_id


# --------------------------------------------------------------------------
# 2. lazy materialization
# --------------------------------------------------------------------------


def test_single_tier_frame_never_materializes_a_pyramid():
    db = _db()
    _add_frame(db, 1)
    for _ in range(5):
        assert db.activate_supervision(1, {"quarter": _handle(16, 24)}) is True
    record = db.get_frame(1)
    assert tuple(record.resolution_cache) == ("quarter",)
    assert db.hot_tier_cache.tiers_for_frame(1) == ("quarter",)
    assert db.hot_tier_cache.entry_count == 1


def test_second_tier_accumulates_instead_of_replacing():
    """The pre-change behaviour dropped the first tier here."""

    db = _db()
    _add_frame(db, 1)
    db.activate_supervision(1, {"quarter": _handle(16, 24)})
    db.activate_supervision(1, {"half": _handle(32, 48)})
    assert sorted(db.get_frame(1).resolution_cache) == ["half", "quarter"]
    assert db.resident_supervision_for_source("source-1", "quarter") is not None
    assert db.resident_supervision_for_source("source-1", "half") is not None


# --------------------------------------------------------------------------
# 3. bounded per-frame retention
# --------------------------------------------------------------------------


def test_third_distinct_tier_evicts_rather_than_growing_to_three():
    db = _db()
    _add_frame(db, 1)
    db.activate_supervision(1, {"quarter": _handle(16, 24)})
    db.activate_supervision(1, {"half": _handle(32, 48)})
    db.activate_supervision(1, {"native": _handle(64, 96)})

    record = db.get_frame(1)
    assert len(record.resolution_cache) == 2
    assert "native" in record.resolution_cache
    assert db.hot_tier_cache.evictions_by_reason.get(EVICT_PER_FRAME_CAP) == 1
    assert record.evicted_resolution_tiers
    # A PARTIAL eviction must not be reported as a fully-evicted frame.
    assert record.resolution_cache_evicted is False


def test_per_frame_cap_is_configurable():
    db = _db(max_tiers_per_frame=3)
    _add_frame(db, 1)
    for tier, (h, w) in {
        "quarter": (16, 24),
        "half": (32, 48),
        "native": (64, 96),
    }.items():
        db.activate_supervision(1, {tier: _handle(h, w)})
    assert len(db.get_frame(1).resolution_cache) == 3


# --------------------------------------------------------------------------
# 4. byte-budget enforcement
# --------------------------------------------------------------------------


def test_byte_budget_evicts_across_frames_and_is_never_exceeded():
    budget = 3 * _MIB
    db = _db(max_hot_tier_bytes=budget)
    for frame_id in range(1, 7):
        _add_frame(db, frame_id)
        # 256x256 float32 RGB+D == 1 MiB exactly.
        db.activate_supervision(frame_id, {"native": _handle(256, 256)})
        assert db.hot_tier_cache_bytes <= budget

    assert db.hot_tier_cache.entry_count == 3
    assert db.hot_tier_cache.evictions_by_reason.get(EVICT_BYTE_BUDGET, 0) >= 3
    resident = sum(
        1 for fid in range(1, 7) if db.get_frame(fid).resolution_cache
    )
    assert resident == 3


def test_entry_bytes_measures_real_tensors_not_a_table():
    handle = _handle(480, 640)
    # 3*480*640*4 (RGB) + 480*640*4 (depth) == 4.6875 MiB
    assert entry_bytes(handle) == (3 * 480 * 640 * 4) + (480 * 640 * 4)
    assert entry_bytes(_handle(480, 640, with_depth=False)) == 3 * 480 * 640 * 4
    quarter = entry_bytes(_handle(120, 160))
    assert quarter * 16 == entry_bytes(handle)


# --------------------------------------------------------------------------
# 5. value-based eviction ordering
# --------------------------------------------------------------------------


def test_value_formula_ranks_p_reuse_prepare_cost_and_bytes():
    prepare = {"native": 0.1, "half": 12.0, "quarter": 8.0}
    cache = HotTierCache(prepare_cost_provider=prepare.get)
    cheap_to_rebuild_and_huge = HotTierEntry(1, "native", 4 * _MIB, 1, 0)
    expensive_and_small = HotTierEntry(2, "quarter", _MIB // 4, 1, 0)
    assert cache.entry_value(expensive_and_small) > cache.entry_value(
        cheap_to_rebuild_and_huge
    )

    # P_reuse dominates when tier and size match.
    often = HotTierEntry(3, "half", _MIB, 10, 0)
    rarely = HotTierEntry(4, "half", _MIB, 1, 0)
    assert cache.entry_value(often) > cache.entry_value(rarely)

    # M_tier dominates when P_reuse and tier match.
    small = HotTierEntry(5, "half", _MIB, 4, 0)
    large = HotTierEntry(6, "half", 8 * _MIB, 4, 0)
    assert cache.entry_value(small) > cache.entry_value(large)


def test_lowest_value_entry_is_evicted_first_not_the_oldest():
    """Pure LRU would evict frame 1; value-based eviction must not."""

    prepare = {"native": 0.1, "quarter": 8.0}
    cache = HotTierCache(max_bytes=2 * _MIB, prepare_cost_provider=prepare.get)
    cache.admit(1, "quarter", _MIB // 4)      # oldest, but tiny and expensive
    for _ in range(8):
        cache.touch(1, "quarter")             # and heavily reused
    cache.admit(2, "native", _MIB)            # newer, big, cheap to rebuild
    evicted = cache.admit(3, "native", _MIB)

    assert [(item.frame_id, item.tier) for item in evicted] == [(2, "native")]
    assert cache.get(1, "quarter") is not None


def test_unmeasured_prepare_cost_degenerates_to_inverse_bytes():
    """With no evidence the ordering is 1/M_tier, so the LARGEST entry goes
    first and quarter survives -- and that falls out of the formula rather than
    being hardcoded anywhere."""

    cache = HotTierCache(max_bytes=2 * _MIB, prepare_cost_provider=lambda tier: None)
    cache.admit(1, "quarter", _MIB // 16)
    cache.admit(2, "native", _MIB)
    # A third frame forces the budget while neither existing entry is the
    # just-admitted (and therefore protected) one.
    evicted = cache.admit(3, "half", _MIB)

    assert [(item.frame_id, item.tier) for item in evicted] == [(2, "native")]
    assert cache.get(1, "quarter") is not None
    assert cache.total_bytes <= 2 * _MIB


def test_an_entry_larger_than_the_whole_budget_is_not_retained():
    cache = HotTierCache(max_bytes=_MIB)
    evicted = cache.admit(1, "native", 4 * _MIB)
    assert cache.entry_count == 0
    assert cache.total_bytes == 0
    assert [item.frame_id for item in evicted] == [1]


# --------------------------------------------------------------------------
# 6. HOT-hit vs cold-miss cost separation
# --------------------------------------------------------------------------


def _context() -> TierCostContext:
    return TierCostContext(device_id="dev", gaussian_count=10_000, camera_count=8)


def _sample(tier, *, prepare_ms, wall_ms, hits, misses) -> TierCostSample:
    return TierCostSample(
        context=_context(),
        tier=tier,
        prepare_ms=prepare_ms,
        optimization_ms=wall_ms - prepare_ms,
        transaction_wall_ms=wall_ms,
        cache_hits=hits,
        cache_misses=misses,
    )


def test_residency_classification():
    assert residency_class(_sample(ResolutionTier.HALF, prepare_ms=0.0, wall_ms=1.0, hits=8, misses=0)) == RESIDENCY_HOT_HIT
    assert residency_class(_sample(ResolutionTier.HALF, prepare_ms=0.0, wall_ms=1.0, hits=0, misses=8)) == RESIDENCY_COLD_MISS
    assert residency_class(_sample(ResolutionTier.HALF, prepare_ms=0.0, wall_ms=1.0, hits=4, misses=4)) == "mixed"
    assert residency_class(_sample(ResolutionTier.HALF, prepare_ms=0.0, wall_ms=1.0, hits=0, misses=0)) == "unknown"


def test_hot_hit_and_cold_miss_are_distinguishable_samples_not_one_average():
    """Real measured shape: HALF prepare is 12.29ms on a miss, 0.022ms hot."""

    estimator = TierCostEstimator()
    context = _context()
    estimator.record(
        _sample(ResolutionTier.HALF, prepare_ms=12.29, wall_ms=15.6, hits=0, misses=8)
    )
    estimator.record(
        _sample(ResolutionTier.HALF, prepare_ms=0.022, wall_ms=3.4, hits=8, misses=0)
    )

    miss = estimator.prepare_state(
        ResolutionTier.HALF, context=context, residency=RESIDENCY_COLD_MISS
    )
    hit = estimator.prepare_state(
        ResolutionTier.HALF, context=context, residency=RESIDENCY_HOT_HIT
    )
    assert miss is not None and hit is not None
    assert miss.completed_samples == 1 and hit.completed_samples == 1
    assert miss.recent_mean_ms == pytest.approx(12.29)
    assert hit.recent_mean_ms == pytest.approx(0.022)
    # NOT blended into one number.
    assert miss.recent_mean_ms != pytest.approx(hit.recent_mean_ms)

    saving = estimator.prepare_saving_ms(ResolutionTier.HALF, context=context)
    assert saving == pytest.approx(12.29 - 0.022)

    # The blended selection window is untouched by this split: it still holds
    # BOTH transactions, because selection semantics must not change here.
    blended = estimator.state(ResolutionTier.HALF, context=context)
    assert blended.completed_samples == 2


def test_prepare_evidence_is_not_shared_across_calibration_buckets():
    estimator = TierCostEstimator()
    rocm = TierCostContext(device_id="rocm", gaussian_count=10_000, camera_count=8)
    cuda = TierCostContext(device_id="cuda", gaussian_count=10_000, camera_count=8)
    estimator.record(
        TierCostSample(
            context=rocm,
            tier=ResolutionTier.QUARTER,
            prepare_ms=8.17,
            transaction_wall_ms=10.0,
            cache_misses=8,
        )
    )
    assert estimator.prepare_saving_ms(ResolutionTier.QUARTER, context=rocm) is not None
    assert estimator.prepare_saving_ms(ResolutionTier.QUARTER, context=cuda) is None


def test_measured_prepare_cost_is_non_monotonic_in_tier():
    """Real 640x480 C=8 data: half (12.29ms) prepares SLOWER than quarter
    (8.17ms), which prepares slower than native (5.06ms, resize skipped).

    A retention policy that assumed ordinal cheapness would price these
    backwards; the value formula must read the measurement.
    """

    estimator = TierCostEstimator()
    context = _context()
    for tier, prepare_ms in (
        (ResolutionTier.NATIVE, 5.06),
        (ResolutionTier.HALF, 12.29),
        (ResolutionTier.QUARTER, 8.17),
    ):
        estimator.record(
            TierCostSample(
                context=context,
                tier=tier,
                prepare_ms=prepare_ms,
                transaction_wall_ms=prepare_ms + 3.0,
                cache_misses=8,
            )
        )
    savings = {
        tier.value: estimator.prepare_saving_ms(tier, context=context)
        for tier in (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
    }
    assert savings["half"] > savings["quarter"] > savings["native"]

    cache = HotTierCache(prepare_cost_provider=lambda tier: savings.get(tier))
    # Equal reuse and equal footprint: retention follows the MEASURED saving.
    half = HotTierEntry(1, "half", _MIB, 1, 0)
    native = HotTierEntry(2, "native", _MIB, 1, 0)
    assert cache.entry_value(half) > cache.entry_value(native)


# --------------------------------------------------------------------------
# 7. extreme regime: a nominally cheaper tier that MEASURES worse
# --------------------------------------------------------------------------


def test_measured_cost_is_reported_even_when_a_lower_tier_is_slower():
    """Synthetic U-shaped-inversion outcome.

    The numbers below are invented for this test alone; no tile-density
    threshold appears in production code, and this asserts only that the
    plumbing reports MEASURED cost rather than assuming ordinal cheapness.
    """

    estimator = TierCostEstimator()
    context = _context()
    estimator.record(
        TierCostSample(
            context=context,
            tier=ResolutionTier.HALF,
            transaction_wall_ms=4.0,
            prepare_ms=1.0,
            cache_hits=8,
        )
    )
    estimator.record(
        TierCostSample(
            context=context,
            tier=ResolutionTier.QUARTER,
            transaction_wall_ms=9.0,  # the nominally cheaper tier is slower
            prepare_ms=1.0,
            cache_hits=8,
        )
    )
    half = estimator.state(ResolutionTier.HALF, context=context)
    quarter = estimator.state(ResolutionTier.QUARTER, context=context)
    assert quarter.estimate_ms > half.estimate_ms

    selection = estimator.select(
        ceiling=ResolutionTier.NATIVE, budget_ms=5.0, context=context
    )
    assert selection.tier is ResolutionTier.HALF
