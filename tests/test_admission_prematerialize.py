"""Admission-time supervision-tier pre-materialize (issue #279/#1342 follow-up).

Grounding: a per-draw trace on rtx4090 / fr3 (2026-09-04, 8884 draws) found
40.1% of ``m_tick_camera_materialize`` lookups miss the HOT cache and that
~54% of those misses are a frame's FIRST supervision draw -- guaranteed,
because ``record_candidate`` builds no tier and ``activate_supervision``
only runs after the tick already missed. That is a ~25% floor a larger HOT
window or COLD promotion cannot remove (both plateau at it in the
calibrated eligsir-lab model). ``_maybe_prematerialize_admitted_supervision``
runs the tick's own miss-path sequence at admission instead, so the first
draw is a HOT hit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.transfers import DeviceTransferOwner
from eligsir.utils.camera_residency import CameraResidencyLedger
from eligsir.utils.score_contract import declare_camera_score_provenance


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("prematerialize-test", 3, ordinal, 1_000_000_000 + ordinal)


def _record(db: FrameObservationDB, source: SourceIdentity):
    pose = np.eye(4, dtype=np.float32)
    mapping = SimpleNamespace(
        source=source,
        device="cpu",
        rgb=SimpleNamespace(data=np.zeros((6, 8, 3), dtype=np.uint8)),
        depth=SimpleNamespace(data=np.ones((6, 8), dtype=np.float32)),
        intrinsics=(8, 6, 8.0, 9.0, 3.5, 2.5),
        reload_token=None,
    )
    return db.record_candidate(
        SimpleNamespace(c2w=pose),
        SimpleNamespace(index=source.source_ordinal, timestamp=source.timestamp_seconds, mapping_frame=mapping),
        source_identity=source,
        is_train=True,
        is_holdout=False,
        historical_eligible=True,
    )


def _camera(width: int = 8, height: int = 6) -> SimpleNamespace:
    """The minimum a freshly admitted train camera carries at record time."""
    cam = SimpleNamespace(
        original_image=torch.full((3, height, width), 0.25, dtype=torch.float32),
        _sensor_depth_cache=torch.ones((height, width), dtype=torch.float32),
        image_width=width, width=width, image_height=height, height=height,
        fx=8.0, fy=9.0, cx=3.5, cy=2.5,
        gsplat=None,
        _canonical_native_resolution_state=None,
        _canonical_resolution_binding_native_state=None,
        _canonical_resolution_tier=None,
        has_pending_image=False,
        R=np.eye(3, dtype=np.float32), T=np.zeros(3, dtype=np.float32),
        FoVx=1.0, FoVy=1.0, znear=0.01, zfar=100.0,
        world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3), uid=0,
    )
    declare_camera_score_provenance(cam, context="prematerialize_test")
    return cam


def _runtime(tier: ResolutionTier | None, *, bound: int = 4):
    db = FrameObservationDB(device="cpu", max_active_supervision_frames=bound)
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = db
    runtime._transfer_owner = DeviceTransferOwner("cpu", max_inflight=1, max_pinned_bytes=1 << 20)
    runtime._residency_ledger = CameraResidencyLedger(enabled=False)
    runtime.gaussians = SimpleNamespace(get_xyz=torch.zeros((1, 3)))
    runtime.config = SimpleNamespace(static_options={})
    runtime._admission_prematerialize_tier = tier
    runtime._admission_prematerialize_total = 0
    runtime._admission_prematerialize_failed = 0
    runtime._admission_prematerialize_warned = False
    # issue #1543 finding A (landed on main independently of this branch):
    # `admission_prefetch_telemetry()` now reads `_admission_prefetch_requested`/
    # `_admission_prefetch_reason` instead of a single `_admission_prefetch_enabled`
    # flag. This bare `object.__new__` fixture bypasses `__init__` (which sets
    # these from the resolved config), so set the disabled defaults directly --
    # prefetch state is not what these prematerialize tests exercise.
    runtime._admission_prefetch_requested = False
    runtime._admission_prefetch_reason = "disabled"
    runtime._admission_prefetch_enabled = False
    return runtime, db


def test_prematerialize_makes_the_first_draw_a_hot_hit():
    runtime, db = _runtime(ResolutionTier.HALF)
    source = _source(1)
    record = _record(db, source)
    cam = _camera()

    # Before: the freshly recorded frame is exactly the guaranteed miss.
    assert db.resident_supervision_for_source(source, "half") is None
    assert db.active_supervision_occupancy == 0

    runtime._maybe_prematerialize_admitted_supervision(cam, record.frame_id)

    # After: the tier is HOT, so the tick's lookup that used to miss now hits.
    resident = db.resident_supervision_for_source(source, "half")
    assert resident is not None
    assert resident.tier == "half"
    assert (resident.rgb.shape[-1], resident.rgb.shape[-2]) == (4, 3)   # 8x6 native -> 4x3 half
    assert resident.depth is not None and tuple(resident.depth.shape) == (3, 4)
    assert db.active_supervision_occupancy == 1
    assert runtime._admission_prematerialize_total == 1


def test_prematerialize_registers_residency_for_the_real_keyframe_camera(monkeypatch):
    # The keyframe camera IS in `scene.hot_camera_ids()`, so its page-in
    # registration matches the tick and `prepare_camera_for_render` builds
    # real gsplat tensors for the camera the tick will actually draw.
    calls = []
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render",
        lambda cam, **kw: calls.append(cam),
    )
    runtime, db = _runtime(ResolutionTier.HALF)
    runtime._residency_ledger = CameraResidencyLedger(enabled=True)
    source = _source(21)
    record = _record(db, source)
    cam = _camera()

    runtime._maybe_prematerialize_admitted_supervision(cam, record.frame_id)

    assert calls == [cam]
    assert runtime._residency_ledger.cumulative_paged_in == 1


def test_prematerialize_is_a_noop_when_disabled():
    runtime, db = _runtime(None)
    source = _source(2)
    record = _record(db, source)
    runtime._maybe_prematerialize_admitted_supervision(_camera(), record.frame_id)
    assert db.resident_supervision_for_source(source, "half") is None
    assert db.active_supervision_occupancy == 0
    assert runtime._admission_prematerialize_total == 0
    # Declined (flag off), not attempted-and-crashed: I2's counters must
    # tell the two apart.
    assert runtime._admission_prematerialize_failed == 0


def test_prematerialize_never_raises_and_leaves_the_frame_untouched_on_failure():
    runtime, db = _runtime(ResolutionTier.HALF)
    source = _source(3)
    record = _record(db, source)
    broken = _camera()
    broken.original_image = None   # nothing to materialize -> early return, not an error
    runtime._maybe_prematerialize_admitted_supervision(broken, record.frame_id)
    assert db.resident_supervision_for_source(source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 0   # declined, not a failure

    # A failure inside the sequence is swallowed too: the miss path stays authoritative.
    bad_intrinsics = _camera()
    bad_intrinsics.fx = None
    runtime._maybe_prematerialize_admitted_supervision(bad_intrinsics, record.frame_id)
    assert db.resident_supervision_for_source(source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 1


def test_prematerialize_failure_counter_and_once_per_run_warning(monkeypatch, capsys):
    runtime, db = _runtime(ResolutionTier.HALF)
    source = _source(4)
    record = _record(db, source)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime._observation_db, "activate_supervision", _raise)

    result = runtime._maybe_prematerialize_admitted_supervision(_camera(), record.frame_id)
    assert result is None   # the public hook has no return contract; the tier stays unpublished
    assert db.resident_supervision_for_source(source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 1
    first_output = capsys.readouterr().out
    assert "prematerialize failed (first of run)" in first_output
    assert "boom" in first_output

    # A second failure increments the counter but does not print again.
    record2 = _record(db, _source(5))
    runtime._maybe_prematerialize_admitted_supervision(_camera(), record2.frame_id)
    assert runtime._admission_prematerialize_failed == 2
    second_output = capsys.readouterr().out
    assert second_output == ""


def test_admission_prefetch_telemetry_reports_prematerialized_and_failed():
    runtime, db = _runtime(ResolutionTier.HALF)
    runtime._admission_prefetch_scheduled_total = 0
    runtime._admission_prefetch_completed_total = 0
    runtime._admission_prefetch_failed_total = 0
    runtime._admission_prefetch_hits_total = 0
    runtime._admission_prematerialize_total = 3
    runtime._admission_prematerialize_failed = 2
    telemetry = runtime.admission_prefetch_telemetry()
    assert telemetry["prematerialized"] == 3
    assert telemetry["prematerialize_failed"] == 2


def test_prematerialize_respects_the_hot_window_bound():
    runtime, db = _runtime(ResolutionTier.HALF, bound=2)
    sources = [_source(10 + i) for i in range(3)]
    for source in sources:
        record = _record(db, source)
        runtime._maybe_prematerialize_admitted_supervision(_camera(), record.frame_id)
    # The LRU bound still governs: the oldest pre-materialized frame is evicted.
    assert db.active_supervision_occupancy == 2
    assert db.resident_supervision_for_source(sources[0], "half") is None
    assert db.resident_supervision_for_source(sources[2], "half") is not None


def _runtime_args(tier: str):
    from eligsir.cli.parser import parse_training_args

    args, _ = parse_training_args(["-s", "/dev/null", "-m", "output/test"])
    args.streaming_frame_admission = "preselected"
    args.gpu_frame_prep = True
    args.gpu_frame_resident = True
    args.streaming_insert_from_depth = False
    args.streaming_mapping_resolution_tier = tier
    args.streaming_admission_prematerialize = True
    return args


def _construct(tier: str):
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args
    from eligsir.runtime.production_application import MappingBackendCompatibility

    config = mapping_runtime_config_from_args(_runtime_args(tier))
    return CanonicalMappingRuntime(
        config,
        MappingBackendCompatibility(
            dataset=SimpleNamespace(sh_degree=3), opt=SimpleNamespace(), pipe=SimpleNamespace()
        ),
    )


requires_gsplat_device = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CanonicalMappingRuntime.__init__ validates accelerator capabilities before the gate",
)


def test_config_resolves_adaptive_to_many_tiers_and_concrete_to_one():
    # The construction-time gate keys on exactly this: `adaptive` yields the
    # full supported set, a concrete tier yields one. CPU-testable, so the
    # refusal condition is pinned even where the real runtime cannot be built.
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    adaptive = mapping_runtime_config_from_args(_runtime_args("adaptive"))
    half = mapping_runtime_config_from_args(_runtime_args("half"))
    assert adaptive.static_options["mapping_resolution_tier"] == "adaptive"
    assert half.static_options["mapping_resolution_tier"] == "half"
    assert adaptive.runtime.admission_prematerialize_enabled is True
    assert half.runtime.admission_prematerialize_enabled is True


# I4: the construction-time gate itself, as the module-level pure function
# `_resolve_prematerialize_tier` -- no accelerator required, so (unlike
# `test_runtime_refuses_prematerialize_under_adaptive_tier` /
# `test_runtime_accepts_prematerialize_under_a_concrete_tier` below, which
# need `CanonicalMappingRuntime.__init__` to reach the gate at all) these run
# on CPU CI. Previously the only CPU-side compensator asserted
# `len(tuple(ResolutionTier)) > 1`, a fact about the enum, not about the
# filtered tuple the gate actually reads.


def test_resolve_prematerialize_tier_disabled_is_always_none():
    from eligsir.runtime.production_application import _resolve_prematerialize_tier

    assert _resolve_prematerialize_tier((), enabled=False) is None
    assert _resolve_prematerialize_tier((ResolutionTier.HALF,), enabled=False) is None
    assert (
        _resolve_prematerialize_tier(
            (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER), enabled=False
        )
        is None
    )


def test_resolve_prematerialize_tier_single_supported_tier_resolves_to_it():
    from eligsir.runtime.production_application import _resolve_prematerialize_tier

    assert _resolve_prematerialize_tier((ResolutionTier.HALF,), enabled=True) is ResolutionTier.HALF


def test_resolve_prematerialize_tier_multiple_supported_tiers_defers_to_runtime():
    """``adaptive`` (several supported tiers) has no static answer: the gate
    yields None and the runtime resolves the tier per admission instead of
    refusing the flag outright (the 2026-09-04 behaviour, which made the
    optimisation unusable on the canonical profile)."""
    from eligsir.runtime.production_application import _resolve_prematerialize_tier

    assert (
        _resolve_prematerialize_tier(
            (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER), enabled=True
        )
        is None
    )


def test_resolve_prematerialize_tier_zero_supported_tiers_raises():
    from eligsir.runtime.production_application import _resolve_prematerialize_tier

    with pytest.raises(ValueError, match="at least one"):
        _resolve_prematerialize_tier((), enabled=True)


def test_prematerialize_tier_for_prefers_the_last_selected_supported_tier():
    from eligsir.runtime.production_application import _prematerialize_tier_for

    supported = (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
    assert _prematerialize_tier_for(supported, ResolutionTier.HALF) is ResolutionTier.HALF
    assert _prematerialize_tier_for(supported, ResolutionTier.NATIVE) is ResolutionTier.NATIVE


def test_prematerialize_tier_for_falls_back_to_the_cheapest_legal_tier():
    """Before any transaction has run (no last selection) -- or when the last
    selection is not a supported tier -- build the cheapest legal tier: under
    a floor-locked scheduler that is the realized tier for almost every
    transaction, and it is the cheapest thing to build speculatively."""
    from eligsir.runtime.production_application import _prematerialize_tier_for

    supported = (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER)
    assert _prematerialize_tier_for(supported, None) is ResolutionTier.QUARTER
    assert _prematerialize_tier_for(supported, ResolutionTier.EIGHTH) is ResolutionTier.QUARTER
    with pytest.raises(ValueError):
        _prematerialize_tier_for((), None)


def test_prematerialize_tier_now_static_attribute_wins_and_adaptive_tracks_selection():
    """Fixture-friendly runtime resolver: the static tier (concrete policy)
    wins; under adaptive the last realized transaction tier is used, falling
    back to the cheapest supported tier before the first transaction."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._admission_prematerialize_tier = ResolutionTier.HALF
    assert runtime._prematerialize_tier_now() is ResolutionTier.HALF

    runtime._admission_prematerialize_tier = None
    runtime._admission_prematerialize_enabled = False
    assert runtime._prematerialize_tier_now() is None

    runtime._admission_prematerialize_enabled = True
    runtime._admission_prematerialize_supported_tiers = (
        ResolutionTier.NATIVE,
        ResolutionTier.HALF,
        ResolutionTier.QUARTER,
        ResolutionTier.EIGHTH,
    )
    assert runtime._prematerialize_tier_now() is ResolutionTier.EIGHTH
    runtime._last_transaction_resolution_tier = ResolutionTier.NATIVE
    assert runtime._prematerialize_tier_now() is ResolutionTier.NATIVE


@requires_gsplat_device
def test_runtime_accepts_prematerialize_under_adaptive_tier():
    # `adaptive` resolves to several supported tiers: no static tier, the
    # runtime resolves it per admission from the latest selection (cheapest
    # legal tier before the first transaction).
    runtime = _construct("adaptive")
    assert runtime._admission_prematerialize_enabled is True
    assert runtime._admission_prematerialize_tier is None
    assert runtime._prematerialize_tier_now() is ResolutionTier.EIGHTH


@requires_gsplat_device
def test_runtime_accepts_prematerialize_under_a_concrete_tier():
    runtime = _construct("half")
    assert runtime._admission_prematerialize_enabled is True
    assert runtime._admission_prematerialize_tier is ResolutionTier.HALF
    assert runtime._admission_prematerialize_total == 0


def test_flag_default_is_off_and_behavior_neutral():
    from eligsir.runtime.config import CanonicalRuntimePolicy
    from eligsir.runtime.config_boundary import mapping_runtime_config_from_args

    assert CanonicalRuntimePolicy().admission_prematerialize_enabled is False
    args = _runtime_args("half")
    args.streaming_admission_prematerialize = False
    assert mapping_runtime_config_from_args(args).runtime.admission_prematerialize_enabled is False


# --- packet-only (non-keyframe) path ---------------------------------------
# On fr3 with preselected admission, every submitted packet becomes the B4
# fresh view (newest_frontier), but only ~1 in 6 is a keyframe that reaches
# _record_scene_candidate with a Camera. The rest are retained as CPU
# history via _retain_deferred_observation and own no Camera until their
# first draw -- so the keyframe hook alone fired 419x on a 2525-frame run
# (measured 2026-09-04) and cut the miss rate only 40.7% -> 34.8%.


def _packet(ordinal: int = 12):
    from eligsir.contracts.runtime import BufferRef, SensorPacket
    from eligsir.contracts.types import CameraIntrinsics
    from eligsir.tracking.providers import observation_from_pose_matrix

    source = SourceIdentity("tum", 0, ordinal, 1_000_000_000 + ordinal)
    intrinsics = CameraIntrinsics(
        fx=8.0, fy=9.0, cx=3.5, cy=2.5, width=8, height=6,
        depth_fx=8.0, depth_fy=9.0, depth_cx=3.5, depth_cy=2.5,
        depth_width=8, depth_height=6, depth_scale=1.0,
    )
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.full((6, 8, 3), 64, dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(np.ones((6, 8), dtype=np.float32), encoding="32FC1", units="m"),
        intrinsics=intrinsics,
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset_pose"),
        reload_token=None,   # resident pixels: the causal_replay/tum_rgbd case
    )


def _defer_disposition(source):
    from eligsir.contracts.control import (
        DispositionKind, HistoricalEligibility, ModalityEligibility,
        ObservationDisposition, ResidencyIntent, SemanticAction,
    )

    return ObservationDisposition(
        source=source, snapshot_id=4, kind=DispositionKind.SKIP_REDUNDANT,
        semantic_action=SemanticAction.DEFER,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
        observe=True, mapping_candidate=False, admit_supervision=False,
        allow_geometry_insertion=False, residency=ResidencyIntent.COLD,
        reason="no_motion",
        modality_eligibility=ModalityEligibility(True, "qualified", True, "qualified", True),
    )


def _packet_runtime(tier, *, bound: int = 4):
    from eligsir.mapping.admission.observation_owner import ColdRetentionPermit

    runtime, db = _runtime(tier, bound=bound)
    runtime._historical_sources_by_frame_id = {}
    runtime._historical_frame_ids_by_source = {}
    runtime._historical_candidate_index = None
    runtime.iteration = 0
    return runtime, db, ColdRetentionPermit


def test_packet_only_source_first_draw_becomes_a_hot_hit(monkeypatch):
    # The proxy is a throwaway: never in `scene.hot_camera_ids()`, so it must
    # NOT be registered with the residency ledger (the next tick's `release`
    # would otherwise see it as a stale batchable camera and submit a real
    # D2H for a tensor nothing owns) and `prepare_camera_for_render` must not
    # run for it (its fabricated pose/intrinsics are discarded anyway --
    # `promote_resident_historical_frame` rebuilds the geometry from the
    # resident record).
    calls = []
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render",
        lambda cam, **kw: calls.append(cam),
    )
    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    runtime._residency_ledger = CameraResidencyLedger(enabled=True)
    packet = _packet()
    record = runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    # The retained CPU history is now ALSO a HOT half-tier: the tick's
    # resident_supervision_for_source lookup that used to miss now hits, and
    # _camera_for_source takes the promote_resident_historical_frame branch.
    resident = db.resident_supervision_for_source(packet.source, "half")
    assert resident is not None
    assert resident.frame_id == record.frame_id
    assert (resident.rgb.shape[-1], resident.rgb.shape[-2]) == (4, 3)
    assert resident.depth is not None and tuple(resident.depth.shape) == (3, 4)
    assert runtime._admission_prematerialize_total == 1
    # Cold history is untouched: the frame is still reloadable/replayable.
    assert db.get_cold_pixels(record.frame_id) is not None
    # I1: the throwaway proxy must not be registered with the residency
    # ledger, and prepare_camera_for_render must not have run for it.
    assert calls == []
    assert runtime._residency_ledger.cumulative_paged_in == 0


def test_packet_only_path_is_a_noop_when_disabled():
    runtime, db, ColdRetentionPermit = _packet_runtime(None)
    packet = _packet(13)
    runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    assert db.resident_supervision_for_source(packet.source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 0


def test_packet_only_path_skips_non_metric_depth_units():
    # I3: the retention sink copies packet.depth.data raw, with no unit
    # validation -- the canonical mapping_frame_source_arrays path refuses
    # non-metric depth, so this hook must too, rather than publishing a tier
    # whose depth is silently in the wrong units.
    from dataclasses import replace

    from eligsir.contracts.runtime import BufferRef

    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    packet = replace(
        _packet(16),
        depth=BufferRef.from_numpy(np.ones((6, 8), dtype=np.float32), encoding="32FC1", units="raw"),
    )
    runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    assert db.resident_supervision_for_source(packet.source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 0


def test_packet_only_path_skips_depth_rgb_shape_mismatch():
    # I3: the proxy assumed depth and rgb share a resolution; the canonical
    # path refuses a mismatch instead of building a tier off misaligned data.
    from dataclasses import replace

    from eligsir.contracts.runtime import BufferRef

    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    packet = replace(
        _packet(17),
        depth=BufferRef.from_numpy(np.ones((5, 8), dtype=np.float32), encoding="32FC1", units="m"),
    )
    runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    assert db.resident_supervision_for_source(packet.source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 0


def test_packet_only_path_moves_chw_rgb_to_hwc_before_publishing():
    # I3: a CHW rgb array (channel-first) must be moved to HWC before the
    # depth/rgb shape comparison and before the proxy is built -- without
    # the moveaxis, `rgb.shape[:2]` would be `(channels, height)` instead of
    # `(height, width)`, so a same-resolution CHW source would silently
    # shape-mismatch against its own depth and never publish.
    from dataclasses import replace

    from eligsir.contracts.runtime import BufferRef

    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    packet = replace(
        _packet(18),
        rgb=BufferRef.from_numpy(np.full((3, 6, 8), 64, dtype=np.uint8), encoding="rgb8"),
    )
    record = runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    resident = db.resident_supervision_for_source(packet.source, "half")
    assert resident is not None
    assert resident.frame_id == record.frame_id
    assert (resident.rgb.shape[-1], resident.rgb.shape[-2]) == (4, 3)   # same 8x6 -> 4x3 half
    assert resident.depth is not None and tuple(resident.depth.shape) == (3, 4)
    assert runtime._admission_prematerialize_total == 1
    assert runtime._admission_prematerialize_failed == 0


def test_packet_only_path_uses_packet_pixels_when_a_reload_token_empties_the_handle():
    # The causal_replay TUM recipe: every packet carries a PixelReloadToken
    # (replay.py::packet_from_streaming_frame) AND resident rgb/depth. The
    # retention sink then stores rgb=None in the ColdPixelHandle, which is
    # exactly why the first packet-path attempt skipped all 2101 non-keyframes
    # (diag: skip=no_resident_rgb reload_token=True x2101). The hook must take
    # the pixels from the packet.
    from dataclasses import replace

    from eligsir.contracts.runtime import PixelReloadToken

    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    token = PixelReloadToken(provider="rgbd_files_v1", locators=("/x/rgb.png", "/x/d.png"), parameters=())
    packet = replace(_packet(14), reload_token=token)   # SensorPacket is a frozen dataclass
    record = runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    handle = db.get_cold_pixels(record.frame_id)
    assert handle.rgb is None and handle.reload_token is not None     # the sink emptied it...
    resident = db.resident_supervision_for_source(packet.source, "half")
    assert resident is not None                                         # ...but the tier is HOT anyway
    assert (resident.rgb.shape[-1], resident.rgb.shape[-2]) == (4, 3)
    assert runtime._admission_prematerialize_total == 1


def test_packet_only_path_skips_when_neither_packet_nor_handle_has_pixels():
    from dataclasses import replace

    from eligsir.contracts.runtime import PixelReloadToken

    runtime, db, ColdRetentionPermit = _packet_runtime(ResolutionTier.HALF)
    token = PixelReloadToken(provider="rgbd_files_v1", locators=("/x/rgb.png", "/x/d.png"), parameters=())
    packet = replace(_packet(15), reload_token=token)
    record = runtime._retain_deferred_observation(
        _defer_disposition(packet.source), ColdRetentionPermit(packet.source, 4), packet, packet.tracking
    )
    # Simulate a truly pixel-free source: call the hook directly with no packet.
    runtime._admission_prematerialize_total = 0
    db.demote_supervision(record.frame_id)
    runtime._maybe_prematerialize_retained_supervision(record, db.get_cold_pixels(record.frame_id), packet=None)
    assert db.resident_supervision_for_source(packet.source, "half") is None
    assert runtime._admission_prematerialize_total == 0
    assert runtime._admission_prematerialize_failed == 0


def test_admission_prefetch_capacity_covers_two_exact_ready_batches():
    from eligsir.runtime.production_application import admission_prefetch_capacity_for_batch

    assert admission_prefetch_capacity_for_batch(4) == 8
    assert admission_prefetch_capacity_for_batch(16) == 32
    assert admission_prefetch_capacity_for_batch(32) == 64
