"""issue #333: requested-tier bind semantics + expected-eviction accounting.

Deliberately imports ONLY ``eligsir.mapping.tier_bind`` and
``eligsir.mapping.observation_db`` (never ``train_streaming``), so these
run CPU-only on a plain host python while a GPU training container is busy.

Two things are under test, matching the issue's two halves:

(a) an LRU-evicted ``resolution_cache`` is an EXPECTED cache event under the
    configured memory policy, not a bind failure -- the reason
    ``issue109exp_E2`` reported ``high_fail: 84266`` (that number is the
    issue's historical evidence; nothing here re-measures a real run);
(b) a tier mismatch fails SAFE -- the camera is never left silently training
    at a tier nobody requested, and nothing is half-mutated.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import (
    BufferStage,
    CacheHandle,
    FrameObservationDB,
    FrameRecord,
)
from eligsir.mapping.tier_bind import (
    ExtraViewBatchCollector,
    CAUSE_CACHE_NEVER_BUILT,
    CAUSE_EXPECTED_EVICTION,
    CAUSE_RESIDENT,
    MultiresBindAccounting,
    OUTCOME_ALREADY_AT_TIER,
    OUTCOME_NATIVE_FALLBACK,
    OUTCOME_RESIDENT_HIT,
    OUTCOME_UNRESOLVED_LOW_UNAVAILABLE,
    OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH,
    OUTCOME_UNRESOLVED_TIER_MISMATCH,
    bind_resolution_tier,
)


class _Intr:
    def __init__(self, w, h, fx, fy, cx, cy):
        self.width, self.height = w, h
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy


class _Frame:
    """Only the attributes tier_bind reads off ``cam._streaming_frame``."""

    def __init__(self, index, width, height):
        self.index = index
        self.width = width
        self.height = height


class _Cam:
    """Native-resolution streaming camera stand-in (as loadCam leaves it)."""

    def __init__(self, h=64, w=64, frame_hw=None, with_image=True):
        self.image_height, self.image_width = h, w
        self.fx = self.fy = 50.0
        self.cx, self.cy = w / 2.0, h / 2.0
        self.original_image = torch.zeros(3, h, w) if with_image else None
        self._sensor_depth_cache = None
        fh, fw = frame_hw if frame_hw is not None else (h, w)
        self._streaming_frame = _Frame(index=7, width=fw, height=fh)


def _record(frame_id=7, cache=None, evicted=False):
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=0.0,
        submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache=cache if cache is not None else {},
        resolution_cache_evicted=evicted,
    )


def _cache(h=64, w=64, divisor=1):
    hh, ww = h // divisor, w // divisor
    return {
        ("high" if divisor == 1 else "low"): CacheHandle(
            rgb=torch.full((hh, ww, 3), 128, dtype=torch.uint8),
            depth=torch.ones(hh, ww, dtype=torch.float32),
            intrinsics=_Intr(ww, hh, 50.0 / divisor, 50.0 / divisor, ww / 2.0, hh / 2.0),
        )
    }


def _full_cache(h=64, w=64, divisor=4):
    c = _cache(h, w, 1)
    c.update(_cache(h, w, divisor))
    return c


# ---------------------------------------------------------------- (a) eviction

def test_lru_eviction_marks_the_record_and_is_not_a_bind_failure():
    """An evicted cache + a HIGH request = expected eviction, resolved bind."""
    db = FrameObservationDB(device="cpu", max_resolution_cache_frames=2)
    for fid in (0, 1, 2):
        db.add_frame(_record(frame_id=fid, cache=_full_cache()))

    evicted = db.get_frame(0)
    assert evicted.resolution_cache == {}
    assert evicted.resolution_cache_evicted is True, "eviction must be recorded"
    assert db.get_frame(2).resolution_cache_evicted is False
    assert db.resolution_cache_evictions == 1
    assert db.resolution_cache_occupancy == 2  # hard bound still honored

    acc = MultiresBindAccounting()
    cam = _Cam()
    res = bind_resolution_tier(cam, evicted, "high", accounting=acc)

    assert res.ok is True, "expected eviction on a HIGH request is recoverable"
    assert res.outcome == OUTCOME_NATIVE_FALLBACK
    assert res.cause == CAUSE_EXPECTED_EVICTION
    assert res.tier_after == "high" and res.usable is True
    assert acc.unresolved == 0 and acc.resolved == 1
    assert acc.causes == {CAUSE_EXPECTED_EVICTION: 1}


def test_never_built_cache_is_classified_separately_from_an_eviction():
    """A cache that was never built is a REAL defect signal, not policy."""
    acc = MultiresBindAccounting()
    res = bind_resolution_tier(_Cam(), _record(cache={}, evicted=False), "high", accounting=acc)
    assert res.cause == CAUSE_CACHE_NEVER_BUILT
    assert acc.causes == {CAUSE_CACHE_NEVER_BUILT: 1}


@pytest.mark.parametrize(
    "reason,expected_cause",
    [
        ("not_train_frame", "cache_by_design_absent"),
        ("depth_load_failed_transient", "cache_by_design_absent"),
        ("depthless_rgb_cache_disabled", "cache_by_design_absent"),
        ("build_exception", "cache_build_failed"),
        (None, CAUSE_CACHE_NEVER_BUILT),
    ],
)
def test_by_design_absent_cache_is_not_lumped_in_with_never_built(reason, expected_cause):
    """An empty cache has THREE meanings, not two: evicted by policy,
    deliberately never built (non-train frame / transient depth-load failure /
    depthless_rgb_cache=False), or a build that raised. Conflating any of them
    recreates the ambiguity this issue exists to remove."""
    rec = _record(cache={}, evicted=False)
    rec.resolution_cache_absent_reason = reason
    acc = MultiresBindAccounting()
    res = bind_resolution_tier(_Cam(), rec, "high", accounting=acc)
    assert res.cause == expected_cause
    assert acc.causes == {expected_cause: 1}


def test_camera_without_pixels_is_never_reported_as_a_resolved_tier():
    """A lazily-built camera whose materialize_pending_image() never ran has no
    original_image; "native is the high tier" must not be claimed for it."""
    cam = _Cam(with_image=True)
    cam.original_image = None
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "high")
    assert res.ok is False and res.usable is False
    assert res.outcome == "unresolved_no_pixels"


def test_lazy_camera_materializes_native_pixels_for_high_after_eviction():
    cam = _Cam(with_image=False)
    calls = []

    def materialize():
        calls.append(True)
        cam.original_image = torch.ones(3, 64, 64)

    cam.materialize_pending_image = materialize
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "high")

    assert calls == [True]
    assert res.ok is True and res.usable is True
    assert res.outcome == OUTCOME_NATIVE_FALLBACK


def test_resident_hit_is_distinguishable_from_native_fallback():
    acc = MultiresBindAccounting()
    cam = _Cam()
    res = bind_resolution_tier(cam, _record(cache=_full_cache()), "high", accounting=acc)
    assert res.ok is True
    assert res.outcome == OUTCOME_RESIDENT_HIT and res.cause == CAUSE_RESIDENT
    assert cam._bound_resolution_tier == "high"
    # Detection hook: a HIGH bind that does not change the camera's size proves
    # native and the high tier agree for this source.
    assert cam._native_matches_high_tier is True
    assert acc.outcomes == {OUTCOME_RESIDENT_HIT: 1}


def test_accounting_keys_are_bounded_not_one_per_frame_id():
    """issue #333 criterion 6: O(reason classes), not O(frame ids)."""
    acc = MultiresBindAccounting()
    for fid in range(1000):
        bind_resolution_tier(_Cam(), _record(frame_id=fid, evicted=True), "high", accounting=acc)
    assert acc.total == 1000
    assert len(acc.outcomes) == 1 and len(acc.causes) == 1
    payload = acc.to_dict()
    # bounded example list, not 1000 frame ids
    assert len(payload["outcome_frame_id_examples"][OUTCOME_NATIVE_FALLBACK]) <= acc.MAX_EXAMPLES
    assert len(payload["tier_totals"]) <= 4


# ------------------------------------------------------ (b) tier fails SAFE

def test_low_bound_camera_refuses_a_high_request_it_cannot_materialize():
    """THE latent bug: a LOW camera must not be left silently 'at HIGH'."""
    cam = _Cam()
    resident = _record(cache=_full_cache())
    assert bind_resolution_tier(cam, resident, "low").ok is True
    assert cam._bound_resolution_tier == "low"
    assert (cam.image_height, cam.image_width) == (16, 16)
    low_state = (cam.image_height, cam.image_width, cam.fx, cam.cx, cam.original_image.shape)

    acc = MultiresBindAccounting()
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "high", accounting=acc)

    assert res.ok is False, "must not claim a HIGH bind it did not perform"
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    assert res.tier_before == "low" and res.tier_after == "low"
    assert res.usable is False, "caller must be able to exclude this camera"
    assert cam._resolution_tier_mismatch == ("low", "high")
    # Nothing half-mutated: restoring dims/intrinsics without restoring pixels
    # would manufacture the #162/#185 image-vs-geometry mismatch.
    assert (
        cam.image_height, cam.image_width, cam.fx, cam.cx, cam.original_image.shape
    ) == low_state
    assert acc.unresolved == 1
    assert acc.to_dict()["unresolved_tier_transitions"] == {"low->high": 1}
    # native geometry is still recoverable as METADATA for a future #396-owned
    # re-materializer, even though pixels are not.
    assert cam._native_resolution_state["image_height"] == 64


def test_low_request_with_no_cache_is_explicit_not_a_silent_high_batch_member():
    acc = MultiresBindAccounting()
    cam = _Cam()
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "low", accounting=acc)
    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_LOW_UNAVAILABLE
    assert res.tier_after == "high" and res.usable is False
    assert not hasattr(cam, "_bound_resolution_tier") or cam._bound_resolution_tier is None


def test_low_cache_miss_rebuilds_through_the_db_and_honors_the_lru_bound():
    """A reconstructable source turns an expected eviction into an exact LOW bind."""
    calls = []

    def rebuild(frame, frame_id):
        calls.append((frame, frame_id))
        return _full_cache()

    db = FrameObservationDB(
        device="cpu",
        max_resolution_cache_frames=1,
        resolution_cache_rebuilder=rebuild,
    )
    old = _record(frame_id=7, cache={}, evicted=True)
    resident = _record(frame_id=8, cache=_full_cache())
    db.add_frame(old)
    db.add_frame(resident)
    cam = _Cam()

    result = bind_resolution_tier(cam, old, "low", db=db)

    assert calls == [(cam._streaming_frame, 7)]
    assert result.ok is True and result.usable is True
    assert result.outcome == "reconstructed"
    assert result.cause == "reconstructed"
    assert result.tier_after == "low"
    assert (cam.image_height, cam.image_width) == (16, 16)
    assert cam.fx == pytest.approx(12.5)
    assert db.resolution_cache_reconstructions == 1
    assert db.resolution_cache_occupancy == 1
    assert db.get_frame(7).resolution_cache_evicted is False
    assert db.get_frame(8).resolution_cache_evicted is True


def test_failed_low_reconstruction_fails_closed_without_mutating_the_camera():
    db = FrameObservationDB(
        device="cpu",
        max_resolution_cache_frames=1,
        resolution_cache_rebuilder=lambda _frame, _frame_id: {},
    )
    record = _record(cache={}, evicted=True)
    db.add_frame(record)
    cam = _Cam()
    before = (cam.image_height, cam.image_width, cam.fx, cam.original_image.clone())

    result = bind_resolution_tier(cam, record, "low", db=db)

    assert result.ok is False
    assert result.outcome == OUTCOME_UNRESOLVED_LOW_UNAVAILABLE
    assert result.cause == CAUSE_EXPECTED_EVICTION
    assert (cam.image_height, cam.image_width, cam.fx) == before[:3]
    assert torch.equal(cam.original_image, before[3])
    assert db.resolution_cache_reconstructions == 0


def test_rebuild_missing_low_does_not_evict_a_valid_resident_cache():
    db = FrameObservationDB(
        device="cpu",
        max_resolution_cache_frames=1,
        resolution_cache_rebuilder=lambda _frame, _frame_id: _cache(divisor=1),
    )
    missing = _record(frame_id=7, cache={}, evicted=True)
    resident = _record(frame_id=8, cache=_full_cache())
    db.add_frame(missing)
    db.add_frame(resident)

    result = bind_resolution_tier(_Cam(), missing, "low", db=db)

    assert result.ok is False
    assert db.get_frame(8).resolution_cache
    assert db.get_frame(8).resolution_cache_evicted is False
    assert db.resolution_cache_occupancy == 1


def test_low_camera_with_live_pixels_does_not_rebuild_an_evicted_cache():
    calls = []
    db = FrameObservationDB(
        device="cpu",
        resolution_cache_rebuilder=lambda _frame, _frame_id: calls.append(1) or _full_cache(),
    )
    record = _record(cache=_full_cache())
    db.add_frame(record)
    cam = _Cam()
    assert bind_resolution_tier(cam, record, "low", db=db).ok is True
    record.resolution_cache = {}
    record.resolution_cache_evicted = True

    result = bind_resolution_tier(cam, record, "low", db=db)

    assert result.outcome == OUTCOME_ALREADY_AT_TIER
    assert calls == []


def test_repeated_request_for_the_tier_the_camera_already_carries_is_resolved():
    cam = _Cam()
    assert bind_resolution_tier(cam, _record(cache=_full_cache()), "low").ok is True
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "low")
    assert res.ok is True and res.outcome == OUTCOME_ALREADY_AT_TIER
    assert res.tier_after == "low" and res.cause == CAUSE_EXPECTED_EVICTION


def test_native_resolution_differing_from_the_high_tier_is_never_called_resolved():
    """If loadCam downscaled the camera (--resolution/-r), "leave it native"
    lands at a THIRD resolution, not the high tier -- so a cache hit and a
    cache miss would train at different resolutions. Must be reported."""
    cam = _Cam(h=32, w=32, frame_hw=(64, 64))  # source is 64x64, camera is 32x32
    acc = MultiresBindAccounting()
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "high", accounting=acc)
    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH
    assert res.usable is False
    assert acc.unresolved == 1


def test_batch_homogeneity_is_decidable_from_the_result_alone():
    """issue #333 criterion 4: a caller can exclude any camera whose realized
    tier differs from the batch's requested tier, with no extra inspection."""
    requested = "high"
    batch = [
        (_Cam(), _record(cache=_full_cache())),                   # resident hit
        (_Cam(), _record(cache={}, evicted=True)),                # native fallback
        (_Cam(h=32, w=32, frame_hw=(64, 64)), _record(cache={}, evicted=True)),  # mismatch
    ]
    kept = []
    for cam, rec in batch:
        res = bind_resolution_tier(cam, rec, requested)
        if res.usable and res.tier_after == requested:
            kept.append(cam)
    assert len(kept) == 2
    assert {(c.image_height, c.image_width) for c in kept} == {(64, 64)}


def test_no_camera_is_reported_not_crashed():
    acc = MultiresBindAccounting()
    res = bind_resolution_tier(None, _record(), "low", accounting=acc)
    assert res.ok is False and res.usable is False
    assert acc.unresolved == 1


def test_depthless_tier_still_leaves_sensor_depth_cache_untouched():
    """Pre-#333 invariant preserved: CacheHandle.depth=None must not clobber
    cam._sensor_depth_cache (its None/False state is the "this source has no
    depth" signal depth-supervision loss code relies on)."""
    cam = _Cam()
    sentinel = object()
    cam._sensor_depth_cache = sentinel
    cache = {
        "low": CacheHandle(
            rgb=torch.full((16, 16, 3), 200, dtype=torch.uint8),
            depth=None,
            intrinsics=_Intr(16, 16, 12.5, 12.5, 8.0, 8.0),
        )
    }
    res = bind_resolution_tier(cam, _record(cache=cache), "low")
    assert res.ok is True
    assert cam._sensor_depth_cache is sentinel
    assert (cam.image_height, cam.image_width) == (16, 16)
    assert cam.fx == pytest.approx(12.5)


# ------------------------------------------------- (c) the CALLER's half of it
#
# `test_batch_homogeneity_is_decidable_from_the_result_alone` above shows the
# RESULT carries enough to decide. It re-implements the filter inline, so it
# said nothing about what the training loop actually does with that decision --
# and the real defect (review of PR #528) lived exactly there: the engine path
# dropped the camera at collection time, which also shrank
# `run_batched_step`'s loss divisor. These pin the production collector both
# loops now share, including the count it hands the engine.


def _skip_result():
    """A real refusal: camera downscaled by loadCam, cache evicted."""
    return bind_resolution_tier(
        _Cam(h=32, w=32, frame_hw=(64, 64)), _record(cache={}, evicted=True), "high",
    )


def test_collector_refuses_an_off_tier_camera_and_still_charges_it_to_the_divisor():
    """THE regression: a dropped view must not shrink the loss divisor.

    B=2 with the single extra view unusable => an EMPTY camera list, but
    `skipped_view_count == 1`, so the engine still divides by the requested 2.
    """
    collector = ExtraViewBatchCollector("high", batch_size=2)

    assert collector.admit(_Cam(), _skip_result()) is False

    assert collector.cameras == []
    assert collector.tier_skipped == 1
    assert collector.skipped_view_count == 1, "requested B=2 must still be the divisor"


def test_collector_admits_a_camera_that_realized_the_requested_tier():
    collector = ExtraViewBatchCollector("high", batch_size=2)
    cam = _Cam()
    res = bind_resolution_tier(cam, _record(cache={}, evicted=True), "high")

    assert res.ok is True and res.tier_after == "high"
    assert collector.admit(cam, res) is True
    assert collector.cameras == [cam]
    assert collector.skipped_view_count == 0


def test_collector_charges_an_exhausted_sampler_to_the_divisor_too():
    """The sampler returning None mid-loop shrinks the batch for a different
    reason; one divisor rule covers both (the inline path has always kept
    `_opt_batch_size` regardless of why a slot went unfilled)."""
    collector = ExtraViewBatchCollector("high", batch_size=4)
    cam = _Cam()
    collector.admit(cam, bind_resolution_tier(cam, _record(cache={}, evicted=True), "high"))
    # ... sampler returns None for the remaining two slots: nothing offered.

    assert collector.cameras == [cam]
    assert collector.tier_skipped == 0
    assert collector.skipped_view_count == 2


def test_collector_mixes_both_skip_reasons_without_double_counting():
    collector = ExtraViewBatchCollector("high", batch_size=4)
    ok_cam = _Cam()
    collector.admit(ok_cam, bind_resolution_tier(ok_cam, _record(cache={}, evicted=True), "high"))
    collector.admit(_Cam(), _skip_result())
    # third slot: sampler exhausted.

    assert collector.cameras == [ok_cam]
    assert collector.tier_skipped == 1
    assert collector.skipped_view_count == 2
    assert len(collector.cameras) + collector.skipped_view_count == collector.batch_size - 1


def test_collector_admits_a_camera_no_tier_was_requested_for():
    """A camera with no `_streaming_frame` never went through a bind; that was
    the pre-#333 behavior and must stay unchanged (result is None)."""
    collector = ExtraViewBatchCollector("high", batch_size=2)
    cam = _Cam()

    assert collector.admit(cam, None) is True
    assert collector.cameras == [cam]
    assert collector.skipped_view_count == 0
