"""Requested-resolution-tier binding semantics and bounded accounting.

A resident cache handle owns both its compact source representation and, after
first use, a memoized optimizer-ready CHW float representation. Rebinding the
same tier therefore does not rebuild full image/depth tensors every time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from eligsir.mapping.integration.depth_helpers import _record_sensor_depth_cache_load

TIER_HIGH = "high"
TIER_LOW = "low"

CAUSE_RESIDENT = "resident"
CAUSE_RECONSTRUCTED = "reconstructed"
CAUSE_EXPECTED_EVICTION = "expected_eviction"
CAUSE_CACHE_NEVER_BUILT = "cache_never_built"
CAUSE_CACHE_BY_DESIGN_ABSENT = "cache_by_design_absent"
CAUSE_CACHE_BUILD_FAILED = "cache_build_failed"
CAUSE_TIER_HANDLE_MISSING = "tier_handle_missing"
CAUSE_RECORD_MISSING = "record_missing"
CAUSE_NO_CAMERA = "no_camera"

OUTCOME_RESIDENT_HIT = "resident_hit"
OUTCOME_RECONSTRUCTED = "reconstructed"
OUTCOME_NATIVE_FALLBACK = "native_fallback"
# The camera was carrying a DERIVED tier, its cache was evicted, and HIGH was
# requested: native pixels were re-decoded and reinstalled (issue #333 phase
# 1/2). Distinct from `native_fallback`, which never lost its native pixels
# and therefore mutated nothing.
OUTCOME_NATIVE_RESTORE = "native_restore"
OUTCOME_ALREADY_AT_TIER = "already_at_tier"
OUTCOME_UNRESOLVED_TIER_MISMATCH = "unresolved_tier_mismatch"
OUTCOME_UNRESOLVED_LOW_UNAVAILABLE = "unresolved_low_unavailable"
OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH = "unresolved_native_res_mismatch"
OUTCOME_UNRESOLVED_NO_CAMERA = "unresolved_no_camera"
OUTCOME_UNRESOLVED_NO_PIXELS = "unresolved_no_pixels"
OUTCOME_INCONSISTENT_CAMERA_STATE = "inconsistent_camera_state"

_RESOLVED_OUTCOMES = frozenset(
    {
        OUTCOME_RESIDENT_HIT,
        OUTCOME_RECONSTRUCTED,
        OUTCOME_NATIVE_FALLBACK,
        OUTCOME_ALREADY_AT_TIER,
        OUTCOME_NATIVE_RESTORE,
    }
)


@dataclass(frozen=True)
class TierBindResult:
    """Outcome of a single resolution tier bind attempt for a camera."""
    ok: bool
    outcome: str
    cause: str
    requested_tier: str
    tier_before: Optional[str]
    tier_after: Optional[str]
    usable: bool = True
    detail: Optional[str] = None

    def __bool__(self) -> bool:
        """Return True if the bind operation succeeded."""
        return bool(self.ok)


class MultiresBindAccounting:
    """Accumulator for tier bind result statistics and diagnostic examples."""
    MAX_EXAMPLES = 8

    def __init__(self, max_examples: int = MAX_EXAMPLES) -> None:
        """Initialize accounting with per-outcome example frame ID limit."""
        self._max_examples = int(max_examples)
        self.outcomes: Dict[str, int] = {}
        self.causes: Dict[str, int] = {}
        self.tier_totals: Dict[str, int] = {}
        self.tier_transitions: Dict[str, int] = {}
        self.examples: Dict[str, List[int]] = {}
        self.total = 0
        self.resolved = 0
        self.unresolved = 0

    @staticmethod
    def _bump(d: Dict[str, int], key: str) -> None:
        """Increment counter value for key in dictionary."""
        d[key] = d.get(key, 0) + 1

    def record(self, result: TierBindResult, frame_id=None, role: str = "primary") -> None:
        """Record a tier bind result with optional frame ID and role tracking."""
        self.total += 1
        if result.ok:
            self.resolved += 1
        else:
            self.unresolved += 1
        self._bump(self.outcomes, result.outcome)
        self._bump(self.causes, result.cause)
        suffix = "ok" if result.ok else "fail"
        role_part = "" if role == "primary" else f"_{role}"
        self._bump(self.tier_totals, f"{result.requested_tier}{role_part}_{suffix}")
        if not result.ok:
            self._bump(
                self.tier_transitions,
                f"{result.tier_before or 'native'}->{result.requested_tier}",
            )
        if frame_id is not None:
            bucket = self.examples.setdefault(result.outcome, [])
            if len(bucket) < self._max_examples and frame_id not in bucket:
                bucket.append(int(frame_id))

    def summary_line(self) -> str:
        """Generate a single-line summary of resolution tier bind accounting."""
        return (
            f"[multires] bind summary: total={self.total} resolved={self.resolved} "
            f"unresolved={self.unresolved} outcomes={self.outcomes} causes={self.causes}"
        )

    def to_dict(self) -> dict:
        """Return accounting data as a serializable dictionary."""
        return {
            "total": self.total,
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "outcomes": dict(self.outcomes),
            "causes": dict(self.causes),
            "tier_totals": dict(self.tier_totals),
            "unresolved_tier_transitions": dict(self.tier_transitions),
            "outcome_frame_id_examples": {k: list(v) for k, v in self.examples.items()},
            "schema_note": (
                "issue #333: causes describe cache residency; outcomes describe whether "
                "the camera ended at the requested tier."
            ),
        }


def _source_native_hw(cam) -> Optional[Tuple[int, int]]:
    """Return native source frame height and width or None."""
    frame = getattr(cam, "_streaming_frame", None)
    if frame is None:
        return None
    try:
        return int(frame.height), int(frame.width)
    except Exception:
        return None


def _camera_hw(cam) -> Tuple[Optional[int], Optional[int]]:
    """Return camera's current image height and width as integer tuple."""
    h = getattr(cam, "image_height", None)
    w = getattr(cam, "image_width", None)
    return (int(h) if h is not None else None, int(w) if w is not None else None)


def _image_geometry_consistent(cam) -> bool:
    """Check if original image shape matches camera height and width."""
    img = getattr(cam, "original_image", None)
    if img is None:
        return True
    shape = getattr(img, "shape", None)
    if shape is None or len(shape) != 3:
        return True
    h, w = _camera_hw(cam)
    if h is None or w is None:
        return True
    return int(shape[1]) == h and int(shape[2]) == w


def _snapshot_native(cam) -> None:
    """Save camera's native resolution and intrinsics to _native_resolution_state."""
    if getattr(cam, "_native_resolution_state", None) is not None:
        return
    h, w = _camera_hw(cam)
    cam._native_resolution_state = {
        "image_height": h,
        "image_width": w,
        "fx": getattr(cam, "fx", None),
        "fy": getattr(cam, "fy", None),
        "cx": getattr(cam, "cx", None),
        "cy": getattr(cam, "cy", None),
    }


_BY_DESIGN_ABSENT_REASONS = frozenset(
    {"not_train_frame", "depth_load_failed_transient", "depthless_rgb_cache_disabled"}
)


def _cache_cause(record, tier: str) -> Tuple[str, Optional[object]]:
    """Determine cache residency cause and return handle or None."""
    if record is None:
        return CAUSE_RECORD_MISSING, None
    cache = getattr(record, "resolution_cache", None)
    if not cache:
        if bool(getattr(record, "resolution_cache_evicted", False)):
            return CAUSE_EXPECTED_EVICTION, None
        reason = getattr(record, "resolution_cache_absent_reason", None)
        if reason in _BY_DESIGN_ABSENT_REASONS:
            return CAUSE_CACHE_BY_DESIGN_ABSENT, None
        if reason == "build_exception":
            return CAUSE_CACHE_BUILD_FAILED, None
        return CAUSE_CACHE_NEVER_BUILT, None
    handle = cache.get(tier)
    if handle is None:
        return CAUSE_TIER_HANDLE_MISSING, None
    return CAUSE_RESIDENT, handle


def _restore_native_high(
    cam, cause: str, tier_before: Optional[str]
) -> Tuple[Optional[TierBindResult], str]:
    """Try to put a derived-tier camera back onto its NATIVE pixels + geometry.

    Returns ``(result, reason)``. ``result`` is a resolved ``TierBindResult``
    on success, or ``None`` to mean "could not -- caller should refuse", in
    which case ``reason`` names WHICH refusal it was. Issue #333 phases 1/2.

    The reason is deliberately a small, bounded set of strings carried in the
    refusal's ``detail`` and never used as a counter key: this issue exists
    because three different causes of an empty cache were collapsed into one
    indistinguishable bucket, and collapsing "no loader" / "decode raised" /
    "native is not the high tier" into one opaque refusal would repeat that
    mistake at a smaller scale.

    Ordering matters and is deliberate: every reason to refuse is checked
    BEFORE anything is mutated, so a refusal still honors this module's "a
    failed bind mutates nothing" contract. Geometry is restored only after
    the pixels are actually in hand.
    """
    snapshot = getattr(cam, "_native_resolution_state", None)
    if not snapshot:
        # No snapshot => no prior successful rebind => this branch is
        # unreachable in practice (tier_before is not None implies one
        # happened), but refusing beats inventing intrinsics.
        return None, "no_native_snapshot"

    native_hw = (snapshot.get("image_height"), snapshot.get("image_width"))
    if native_hw[0] is None or native_hw[1] is None:
        return None, "incomplete_native_snapshot"

    src_hw = _source_native_hw(cam)
    if src_hw is not None and (int(native_hw[0]), int(native_hw[1])) != src_hw:
        # Same hazard `unresolved_native_res_mismatch` describes: this
        # camera's native grid is NOT the high tier's grid (loadCam applied
        # a --resolution downscale), so restoring native would land at a
        # third resolution. Refuse rather than claim HIGH.
        return None, "native_size_is_not_the_high_tier"

    from scene.cameras import rematerialize_native_image

    if not rematerialize_native_image(cam):
        # No retained loader at all, or the re-decode failed/returned nothing.
        # `rematerialize_native_image` reports False rather than raising, so
        # these two are not distinguishable here; both mean the source could
        # not produce native pixels right now.
        return None, "native_pixels_unavailable"

    cam.image_height = int(native_hw[0])
    cam.image_width = int(native_hw[1])
    # `width`/`height` are Camera's own aliases; keep them in lockstep or
    # downstream readers see two different sizes on the same object.
    cam.width = cam.image_width
    cam.height = cam.image_height
    for key in ("fx", "fy", "cx", "cy"):
        if snapshot.get(key) is not None:
            setattr(cam, key, snapshot[key])

    if not _image_geometry_consistent(cam):
        # The re-decode produced something that does not match the native
        # snapshot. Report it rather than training on a mismatch (#162/#185).
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_INCONSISTENT_CAMERA_STATE,
            cause=cause,
            requested_tier=TIER_HIGH,
            tier_before=tier_before,
            tier_after=None,
            usable=False,
            detail="re-decoded native image disagrees with the native snapshot",
        ), "decoded_size_mismatch"

    # The camera now carries its own native pixels again, exactly as it did
    # before its first rebind -- so it is back to the `tier_before is None`
    # state, not "bound to high".
    cam._bound_resolution_tier = None
    cam._resolution_tier_mismatch = None
    # The derived tier had installed ITS OWN depth at ITS resolution. Drop it
    # so the next read lazily reloads native-resolution depth. `False` is
    # depth_helpers' permanent-failure sentinel and must not be clobbered.
    if getattr(cam, "_sensor_depth_cache", None) is not False:
        cam._sensor_depth_cache = None

    return TierBindResult(
        ok=True,
        outcome=OUTCOME_NATIVE_RESTORE,
        cause=cause,
        requested_tier=TIER_HIGH,
        tier_before=tier_before,
        tier_after=TIER_HIGH,
        usable=True,
        detail=None,
    ), "restored"


def _resolve_without_cache(cam, requested: str, cause: str) -> TierBindResult:
    """Resolve tier bind when cache is unavailable, attempting native pixel restoration."""
    tier_before = getattr(cam, "_bound_resolution_tier", None)
    if tier_before is not None:
        if tier_before == requested:
            ok = (
                getattr(cam, "original_image", None) is not None
                and _image_geometry_consistent(cam)
            )
            return TierBindResult(
                ok=ok,
                outcome=OUTCOME_ALREADY_AT_TIER if ok else OUTCOME_INCONSISTENT_CAMERA_STATE,
                cause=cause,
                requested_tier=requested,
                tier_before=tier_before,
                tier_after=tier_before,
                usable=ok,
                detail=None if ok else "original_image shape disagrees with image_height/width",
            )
        # THE issue #333 correctness case: the camera is at another tier and
        # its cache is gone. Phase 1/2: if HIGH is what was asked for, the
        # camera's own native pixels ARE the high tier, and they can be
        # re-decoded from the frame source -- so try that before refusing.
        restore_reason = "low_requested_reconstruction_out_of_scope"
        if requested == TIER_HIGH:
            restored, restore_reason = _restore_native_high(cam, cause, tier_before)
            if restored is not None:
                return restored
        # Not restorable (LOW requested, no retained loader, a re-decode that
        # failed, or a camera whose native size is not the high tier).
        # Refuse explicitly and touch nothing.
        cam._resolution_tier_mismatch = (tier_before, requested)
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_UNRESOLVED_TIER_MISMATCH,
            cause=cause,
            requested_tier=requested,
            tier_before=tier_before,
            tier_after=tier_before,
            usable=False,
            detail=(
                f"camera carries tier {tier_before!r}; tier {requested!r} requested but not "
                f"materializable (native-restore refused: {restore_reason})"
            ),
        )

    if requested != TIER_HIGH:
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_UNRESOLVED_LOW_UNAVAILABLE,
            cause=cause,
            requested_tier=requested,
            tier_before=None,
            tier_after=TIER_HIGH,
            usable=False,
            detail="no cached low tier and no reconstruction path; camera stays at native/high",
        )

    src_hw = _source_native_hw(cam)
    cam_hw = _camera_hw(cam)
    if src_hw is not None and cam_hw != src_hw:
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH,
            cause=cause,
            requested_tier=requested,
            tier_before=None,
            tier_after=f"native{cam_hw}",
            usable=False,
            detail=(
                f"camera native {cam_hw} != source/high-tier {src_hw}; a cache hit and a "
                "cache miss would train at different resolutions"
            ),
        )

    if getattr(cam, "original_image", None) is None:
        # Exact endpoint resume deliberately reconstructs cameras lazily.  A
        # HIGH request is the point where native pixels are actually needed;
        # materialize the existing one-shot loader before classifying the
        # camera as pixel-less.  Cameras without a loader still fail closed.
        materialize = getattr(cam, "materialize_pending_image", None)
        if callable(materialize):
            materialize()
    if getattr(cam, "original_image", None) is None:
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_UNRESOLVED_NO_PIXELS,
            cause=cause,
            requested_tier=requested,
            tier_before=None,
            tier_after=None,
            usable=False,
            detail="camera has no original_image (unmaterialized lazy image)",
        )
    if not _image_geometry_consistent(cam):
        return TierBindResult(
            ok=False,
            outcome=OUTCOME_INCONSISTENT_CAMERA_STATE,
            cause=cause,
            requested_tier=requested,
            tier_before=None,
            tier_after=None,
            usable=False,
            detail="original_image shape disagrees with image_height/width",
        )
    return TierBindResult(
        ok=True,
        outcome=OUTCOME_NATIVE_FALLBACK,
        cause=cause,
        requested_tier=requested,
        tier_before=None,
        tier_after=TIER_HIGH,
        usable=True,
    )


def _training_rgb(handle):
    """Get or memoize RGB in CHW float32 [0,1] format from cache handle."""
    cached = getattr(handle, "training_rgb", None)
    if cached is None:
        cached = (
            handle.rgb.permute(2, 0, 1)
            .to(dtype=__import__("torch").float32)
            .mul_(1.0 / 255.0)
            .clamp_(0.0, 1.0)
            .contiguous()
        )
        handle.training_rgb = cached
    return cached


def _training_depth(handle):
    """Get or memoize depth in float32 1HW format from cache handle."""
    if handle.depth is None:
        return None
    cached = getattr(handle, "training_depth", None)
    if cached is None:
        cached = handle.depth.to(dtype=__import__("torch").float32)[None, :, :].contiguous()
        handle.training_depth = cached
    return cached


def bind_resolution_tier(
    cam,
    record,
    tier: str,
    accounting: Optional[MultiresBindAccounting] = None,
    db: Optional[object] = None,
) -> TierBindResult:
    """Bind camera to requested resolution tier using cache or reconstruction."""
    frame_id = getattr(record, "frame_id", None) if record is not None else None
    if cam is None:
        result = TierBindResult(
            ok=False,
            outcome=OUTCOME_UNRESOLVED_NO_CAMERA,
            cause=CAUSE_NO_CAMERA,
            requested_tier=tier,
            tier_before=None,
            tier_after=None,
            usable=False,
        )
        if accounting is not None:
            accounting.record(result, frame_id)
        return result

    cause, handle = _cache_cause(record, tier)
    reconstructed = False
    initial_cause = cause
    if (
        cause != CAUSE_RESIDENT
        and tier == TIER_LOW
        and (
            getattr(cam, "_bound_resolution_tier", None) != TIER_LOW
            or getattr(cam, "original_image", None) is None
            or not _image_geometry_consistent(cam)
        )
        and db is not None
        and frame_id is not None
        and getattr(cam, "_streaming_frame", None) is not None
        and hasattr(db, "rebuild_resolution_cache")
        and db.rebuild_resolution_cache(
            frame_id, cam._streaming_frame, required_tier=TIER_LOW
        )
    ):
        cause, handle = _cache_cause(record, tier)
        reconstructed = cause == CAUSE_RESIDENT
    if cause != CAUSE_RESIDENT:
        result = _resolve_without_cache(cam, tier, cause)
        if accounting is not None:
            accounting.record(result, frame_id)
        return result

    tier_before = getattr(cam, "_bound_resolution_tier", None)
    if tier_before is None:
        _snapshot_native(cam)
    pre_hw = _camera_hw(cam)

    rgb_chw = _training_rgb(handle)
    cam.original_image = rgb_chw
    # Preserve the durable identity across tier rebinding/cold replay so
    # train-frame-owned photometric state resolves from FrameObservationDB.
    if getattr(record, "source_identity", None):
        cam._source_identity = record.source_identity
    from eligsir.utils.score_contract import declare_camera_score_provenance

    declare_camera_score_provenance(cam, context="multires_tier_bind")
    cam.image_height = int(rgb_chw.shape[1])
    cam.image_width = int(rgb_chw.shape[2])
    intr = handle.intrinsics
    cam.fx, cam.fy, cam.cx, cam.cy = (
        float(intr.fx),
        float(intr.fy),
        float(intr.cx),
        float(intr.cy),
    )
    depth = _training_depth(handle)
    if depth is not None:
        cam._sensor_depth_cache = depth
        _record_sensor_depth_cache_load(cam)
    cam._bound_resolution_tier = tier
    cam._resolution_tier_mismatch = None

    if tier == TIER_HIGH and tier_before is None:
        cam._native_matches_high_tier = bool(pre_hw == _camera_hw(cam))

    result = TierBindResult(
        ok=True,
        outcome=OUTCOME_RECONSTRUCTED if reconstructed else OUTCOME_RESIDENT_HIT,
        cause=CAUSE_RECONSTRUCTED if reconstructed else CAUSE_RESIDENT,
        requested_tier=tier,
        tier_before=tier_before,
        tier_after=tier,
        usable=True,
        detail=(f"rebuilt after {initial_cause}" if reconstructed else None),
    )
    if accounting is not None:
        accounting.record(result, frame_id)
    if db is not None and frame_id is not None:
        db.touch_resolution_cache(frame_id)
    return result


def describe_startup_resolution_hazard(resolution, resolution_scale) -> Optional[str]:
    """Return warning message if startup resolution creates high/low tier mismatch."""
    try:
        res = int(resolution)
        scale = float(resolution_scale)
    except Exception:
        return None
    if res == 1 and scale == 1.0:
        return None
    return (
        f"[multires] WARNING: resolution={res} resolution_scale={scale} -- the camera is "
        "built at a DOWNSCALED size while the multires 'high' tier is built at the source "
        "frame's native size. Every frame whose resolution_cache has been evicted will "
        f"report {OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH} (usable=False), i.e. cache-hit "
        "and cache-miss iterations do not agree on what 'high' means (issue #333). "
        "Streaming runs pin resolution=1 unless --resolution/-r is passed explicitly; "
        "drop that flag (or set -r 1 and use --streaming_resolution instead) to avoid this."
    )


def is_resolved_outcome(outcome: str) -> bool:
    """Return True if outcome is one of the resolved tier bind outcomes."""
    return outcome in _RESOLVED_OUTCOMES


def extra_view_admitted(result: TierBindResult, requested_tier: str) -> bool:
    """Check if bind result is usable and matches the requested tier."""
    return bool(result.usable) and result.tier_after == requested_tier


class ExtraViewBatchCollector:
    """Collect a tier-homogeneous extra-view batch and requested-size divisor."""

    def __init__(self, requested_tier: str, batch_size: int) -> None:
        """Initialize batch collector for tier-homogeneous extra views."""
        self.requested_tier = requested_tier
        self.batch_size = int(batch_size)
        self.cameras: List = []
        self.tier_skipped: int = 0
        self.last_skipped_result: Optional[TierBindResult] = None

    def admit(self, cam, result: Optional[TierBindResult] = None) -> bool:
        """Admit camera to batch if tier matches or no bind result provided."""
        if result is not None and not extra_view_admitted(result, self.requested_tier):
            self.tier_skipped += 1
            self.last_skipped_result = result
            return False
        self.cameras.append(cam)
        return True

    @property
    def skipped_view_count(self) -> int:
        """Return number of views skipped to reach requested batch size."""
        return max(0, self.batch_size - 1 - len(self.cameras))
