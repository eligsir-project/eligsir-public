"""issue #333 phases 1/2: HIGH has one owner -- the camera's native pixels.

The scenario these tests pin is the issue's own "deeper correctness problem",
and it was a genuine dead end before this change:

    bind LOW  ->  resolution_cache evicted by the bounded LRU  ->  HIGH requested

``bind_resolution_tier``'s resident-hit branch overwrites
``cam.original_image`` with the LOW tensor, which destroys the camera's native
pixels; ``Camera.materialize_pending_image`` then nulled its loader after a
single use, so there was no route back. The bind could only be refused
(``unresolved_tier_mismatch``), and acceptance criterion 2 ("zero unresolved
binds for reconstructable sources") was unreachable rather than merely
unmeasured.

``Camera`` now retains that loader and ``_restore_native_high`` re-decodes.
These tests cover both directions: the restore really happens AND every
refusal path still refuses without half-mutating the camera.

CPU-only by construction -- ``tier_bind`` is imported directly and the
``Camera`` methods are exercised unbound against duck-typed objects, so none
of this needs the CUDA calls in ``Camera.__init__``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eligsir.mapping.observation_db import (
    BufferStage,
    FrameObservationDB,
    FrameRecord,
)
from eligsir.mapping.tier_bind import (
    CAUSE_EXPECTED_EVICTION,
    MultiresBindAccounting,
    OUTCOME_INCONSISTENT_CAMERA_STATE,
    OUTCOME_NATIVE_RESTORE,
    OUTCOME_UNRESOLVED_LOW_UNAVAILABLE,
    OUTCOME_UNRESOLVED_TIER_MISMATCH,
    bind_resolution_tier,
    extra_view_admitted,
    is_resolved_outcome,
)

NATIVE_H, NATIVE_W = 64, 48
LOW_H, LOW_W = 16, 12


class _Frame:
    def __init__(self, index=7, width=NATIVE_W, height=NATIVE_H):
        self.index = index
        self.width = width
        self.height = height


class _LowBoundCam:
    """A camera in exactly the post-LOW-bind state the issue describes.

    Its native pixels are GONE (overwritten by the low tier), its geometry
    and intrinsics are the low tier's, and ``_native_resolution_state`` is
    the snapshot the resident-hit branch took before the first rebind.
    """

    def __init__(self, loader=None, native_hw=(NATIVE_H, NATIVE_W), frame_hw=None):
        nh, nw = native_hw
        # current (LOW) state
        self.image_height, self.image_width = LOW_H, LOW_W
        self.width, self.height = LOW_W, LOW_H
        self.fx = self.fy = 12.5
        self.cx, self.cy = LOW_W / 2.0, LOW_H / 2.0
        self.original_image = torch.zeros(3, LOW_H, LOW_W)
        self._sensor_depth_cache = torch.ones(LOW_H, LOW_W)
        self._bound_resolution_tier = "low"
        self.data_device = "cpu"
        self._native_image_loader = loader
        self._native_resolution_state = {
            "image_height": nh,
            "image_width": nw,
            "fx": 50.0,
            "fy": 50.0,
            "cx": nw / 2.0,
            "cy": nh / 2.0,
        }
        fh, fw = frame_hw if frame_hw is not None else (NATIVE_H, NATIVE_W)
        self._streaming_frame = _Frame(width=fw, height=fh)


def _native_loader(h=NATIVE_H, w=NATIVE_W, value=0.5):
    """A re-callable loader, exactly like the real ones (both re-decode)."""

    calls = {"n": 0}

    def _load():
        calls["n"] += 1
        return torch.full((3, h, w), value), None

    _load.calls = calls
    return _load


def _evicted_record(frame_id=7):
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=0.0,
        submap_id=0,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
        resolution_cache={},
        resolution_cache_evicted=True,
    )


# ------------------------------------------------------- the restore happens

def test_low_bound_camera_with_evicted_cache_is_restored_to_native_high():
    """THE issue #333 scenario: LOW -> evict -> HIGH is no longer a dead end."""
    cam = _LowBoundCam(loader=_native_loader())
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is True
    assert res.outcome == OUTCOME_NATIVE_RESTORE
    # The eviction is still reported as the expected cache event it is.
    assert res.cause == CAUSE_EXPECTED_EVICTION
    assert res.tier_before == "low"
    assert res.tier_after == "high"
    assert res.usable is True
    assert is_resolved_outcome(res.outcome) is True


def test_restore_puts_back_native_dimensions_intrinsics_and_pixels():
    """Acceptance criterion 3: never silently left on the previous tier."""
    cam = _LowBoundCam(loader=_native_loader(value=0.25))
    bind_resolution_tier(cam, _evicted_record(), "high")

    assert (cam.image_height, cam.image_width) == (NATIVE_H, NATIVE_W)
    # `width`/`height` are Camera's own aliases -- a reader of either must
    # not see the stale LOW size.
    assert (cam.height, cam.width) == (NATIVE_H, NATIVE_W)
    assert (cam.fx, cam.fy) == (50.0, 50.0)
    assert (cam.cx, cam.cy) == (NATIVE_W / 2.0, NATIVE_H / 2.0)
    assert tuple(cam.original_image.shape) == (3, NATIVE_H, NATIVE_W)
    assert torch.allclose(cam.original_image, torch.full((3, NATIVE_H, NATIVE_W), 0.25))


def test_restored_camera_is_back_at_native_not_marked_bound_to_high():
    """It carries its OWN pixels again, so it is in the pre-first-bind state."""
    cam = _LowBoundCam(loader=_native_loader())
    bind_resolution_tier(cam, _evicted_record(), "high")

    assert cam._bound_resolution_tier is None
    assert getattr(cam, "_resolution_tier_mismatch", None) is None


def test_restore_drops_the_low_tier_depth_so_it_reloads_at_native_resolution():
    """A 16x12 depth map left behind next to a 64x48 image is issue #162."""
    cam = _LowBoundCam(loader=_native_loader())
    assert cam._sensor_depth_cache is not None  # low-res depth from the LOW bind
    bind_resolution_tier(cam, _evicted_record(), "high")
    # None is depth_helpers' "reload lazily on next access" state.
    assert cam._sensor_depth_cache is None


def test_restore_never_clobbers_the_permanent_depth_failure_sentinel():
    """`False` means depth load permanently failed; None would retry forever."""
    cam = _LowBoundCam(loader=_native_loader())
    cam._sensor_depth_cache = False
    bind_resolution_tier(cam, _evicted_record(), "high")
    assert cam._sensor_depth_cache is False


def test_a_restored_camera_may_join_a_high_tier_batch():
    """Criterion 4 stays satisfied: it really is at HIGH, so it is admissible."""
    cam = _LowBoundCam(loader=_native_loader())
    res = bind_resolution_tier(cam, _evicted_record(), "high")
    assert extra_view_admitted(res, "high") is True


def test_restore_is_counted_as_resolved_not_as_a_failure():
    acct = MultiresBindAccounting()
    cam = _LowBoundCam(loader=_native_loader())
    bind_resolution_tier(cam, _evicted_record(), "high", accounting=acct)

    snap = acct.to_dict()
    assert snap["unresolved"] == 0
    assert snap["resolved"] == 1
    assert snap["outcomes"][OUTCOME_NATIVE_RESTORE] == 1
    assert snap["causes"][CAUSE_EXPECTED_EVICTION] == 1


def test_repeated_replay_of_the_same_evicted_frame_keeps_resolving():
    """Old replay frames are hit over and over; each must resolve."""
    loader = _native_loader()
    cam = _LowBoundCam(loader=loader)
    acct = MultiresBindAccounting()

    first = bind_resolution_tier(cam, _evicted_record(), "high", accounting=acct)
    assert first.outcome == OUTCOME_NATIVE_RESTORE
    # Now the camera is native again, so later requests are the cheap
    # `native_fallback` path -- no repeated re-decode.
    for _ in range(5):
        again = bind_resolution_tier(cam, _evicted_record(), "high", accounting=acct)
        assert again.ok is True

    assert acct.to_dict()["unresolved"] == 0
    assert loader.calls["n"] == 1, "re-decode must not repeat once native is back"


# ------------------------------------------------- every refusal still refuses

def _assert_untouched(cam):
    """A refused bind must leave the camera exactly as it found it."""
    assert cam._bound_resolution_tier == "low"
    assert (cam.image_height, cam.image_width) == (LOW_H, LOW_W)
    assert (cam.height, cam.width) == (LOW_H, LOW_W)
    assert cam.fx == 12.5
    assert tuple(cam.original_image.shape) == (3, LOW_H, LOW_W)


def test_no_retained_loader_still_refuses_explicitly():
    """A source with no re-decode route is unresolved, never a silent LOW."""
    cam = _LowBoundCam(loader=None)
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    assert res.usable is False
    assert res.tier_after == "low"
    _assert_untouched(cam)


def test_a_loader_that_raises_refuses_without_mutating_or_propagating():
    def _boom():
        raise OSError("frame file vanished")

    cam = _LowBoundCam(loader=_boom)
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    _assert_untouched(cam)


def test_a_loader_returning_none_refuses_without_mutating():
    cam = _LowBoundCam(loader=lambda: (None, None))
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    _assert_untouched(cam)


def test_native_size_differing_from_the_high_tier_refuses_rather_than_claiming_high():
    """`-r 2` downscaled the camera, so native is a THIRD resolution.

    Restoring it would silently train at neither the cached high tier's
    resolution nor the requested one -- exactly the hazard
    `unresolved_native_res_mismatch` exists to name.
    """
    cam = _LowBoundCam(
        loader=_native_loader(h=32, w=24),
        native_hw=(32, 24),
        frame_hw=(NATIVE_H, NATIVE_W),  # source/high tier is 64x48
    )
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    _assert_untouched(cam)


def test_missing_native_snapshot_refuses_rather_than_inventing_intrinsics():
    cam = _LowBoundCam(loader=_native_loader())
    cam._native_resolution_state = None
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_TIER_MISMATCH
    _assert_untouched(cam)


def test_a_decode_that_returns_the_wrong_size_is_reported_not_trained_on():
    """Guards the #162/#185 image-vs-geometry mismatch class explicitly."""
    cam = _LowBoundCam(loader=_native_loader(h=99, w=77))
    res = bind_resolution_tier(cam, _evicted_record(), "high")

    assert res.ok is False
    assert res.outcome == OUTCOME_INCONSISTENT_CAMERA_STATE
    assert res.usable is False


def test_low_requests_are_still_out_of_scope_and_say_so():
    """Phase 2's other half (LOW reconstruction) is NOT claimed by this change."""
    cam = _LowBoundCam(loader=_native_loader())
    cam._bound_resolution_tier = None  # never rebound; carries native pixels
    res = bind_resolution_tier(cam, _evicted_record(), "low")

    assert res.ok is False
    assert res.outcome == OUTCOME_UNRESOLVED_LOW_UNAVAILABLE
    assert res.usable is False


# ----------------------------------------- the enabling hook, and its wiring

def test_materialize_pending_image_no_longer_destroys_the_route_back():
    """The one-shot loader was the whole reason the restore was impossible."""
    from scene.cameras import Camera

    class _Lazy:
        pass

    cam = _Lazy()
    cam.data_device = "cpu"
    cam.image_height, cam.image_width = NATIVE_H, NATIVE_W
    loader = _native_loader()
    Camera.set_image_loader(cam, loader)

    assert cam._pending_image_loader is loader
    assert cam._native_image_loader is loader

    Camera.materialize_pending_image(cam)

    # The one-shot lazy contract is unchanged...
    assert cam._pending_image_loader is None
    # ...but the native route survives, which is the fix.
    assert cam._native_image_loader is loader


def test_rematerialize_reinstalls_pixels_without_touching_geometry():
    """Pixel ownership only -- tier_bind owns the geometry decision."""
    from scene.cameras import rematerialize_native_image

    cam = _LowBoundCam(loader=_native_loader(value=0.75))
    assert rematerialize_native_image(cam) is True

    assert torch.allclose(cam.original_image, torch.full((3, NATIVE_H, NATIVE_W), 0.75))
    # Deliberately still the LOW geometry: this function must not decide.
    assert (cam.image_height, cam.image_width) == (LOW_H, LOW_W)


def test_rematerialize_applies_the_alpha_mask_like_the_eager_path():
    from scene.cameras import rematerialize_native_image

    mask = torch.zeros(1, NATIVE_H, NATIVE_W)

    def _load():
        return torch.ones(3, NATIVE_H, NATIVE_W), mask

    cam = _LowBoundCam(loader=_load)
    assert rematerialize_native_image(cam) is True
    assert torch.count_nonzero(cam.original_image) == 0


def test_rematerialize_reports_false_rather_than_raising():
    from scene.cameras import rematerialize_native_image

    cam = _LowBoundCam(loader=None)
    assert rematerialize_native_image(cam) is False


def test_streaming_scene_installs_the_loader_through_the_syncing_setter():
    """Production-wiring guard, in the style of the phase-3 db-param test.

    A bare ``cam._pending_image_loader = _pending_loader`` would leave
    ``_native_image_loader`` on loadCam's own closure over stale frame state,
    silently reintroducing the dead end for every streaming camera. This
    fails if anyone reverts that line.
    """
    import inspect

    import scene.streaming_scene as ss

    src = inspect.getsource(ss)
    assert "cam.set_image_loader(_pending_loader)" in src
    assert "cam._pending_image_loader = _pending_loader" not in src


# ------------------------------------- refusals stay DISTINGUISHABLE (crit. 6)

def test_each_refusal_names_which_refusal_it_was():
    """Three different reasons must not collapse into one opaque message.

    This issue exists because three causes of an empty cache were conflated
    into a single bucket; repeating that inside the restore path -- where
    "no loader", "decode failed" and "native is not the high tier" are
    operationally very different -- would be the same mistake at a smaller
    scale. The reason rides in `detail` (a free-text field), never as a
    counter key, so accounting cardinality stays bounded.
    """
    no_loader = bind_resolution_tier(_LowBoundCam(loader=None), _evicted_record(), "high")
    wrong_size = bind_resolution_tier(
        _LowBoundCam(
            loader=_native_loader(h=32, w=24),
            native_hw=(32, 24),
            frame_hw=(NATIVE_H, NATIVE_W),
        ),
        _evicted_record(),
        "high",
    )
    cam_no_snap = _LowBoundCam(loader=_native_loader())
    cam_no_snap._native_resolution_state = None
    no_snapshot = bind_resolution_tier(cam_no_snap, _evicted_record(), "high")

    assert "native_pixels_unavailable" in no_loader.detail
    assert "native_size_is_not_the_high_tier" in wrong_size.detail
    assert "no_native_snapshot" in no_snapshot.detail

    details = {no_loader.detail, wrong_size.detail, no_snapshot.detail}
    assert len(details) == 3, "refusal reasons must be distinguishable"


def test_refusal_reasons_do_not_become_accounting_keys():
    """Criterion 6: the counters stay O(reason classes), not O(anything else)."""
    acct = MultiresBindAccounting()
    for cam in (
        _LowBoundCam(loader=None),
        _LowBoundCam(loader=lambda: (None, None)),
    ):
        bind_resolution_tier(cam, _evicted_record(), "high", accounting=acct)

    snap = acct.to_dict()
    # Both refusals land in the SAME bounded outcome bucket despite carrying
    # different `detail` text.
    assert snap["outcomes"][OUTCOME_UNRESOLVED_TIER_MISMATCH] == 2
    assert not any("native_pixels_unavailable" in k for k in snap["outcomes"])
    assert not any("no_native_snapshot" in k for k in snap["causes"])


def test_a_low_request_records_that_low_reconstruction_is_out_of_scope():
    cam = _LowBoundCam(loader=_native_loader())  # bound to low, low requested
    cam._bound_resolution_tier = "high"
    res = bind_resolution_tier(cam, _evicted_record(), "low")
    assert res.ok is False
    assert "low_requested_reconstruction_out_of_scope" in res.detail


def test_native_restore_lands_in_high_ok_not_high_fail():
    """The legacy tally is the number this issue was originally reported in.

    `issue109exp_E2` quoted `{'high_ok': 175734, 'high_fail': 84266}`, and
    `high_ok`/`high_fail` is still what a human reads in train.log. A restore
    that resolved the request but was tallied as a failure would leave that
    headline number saying the opposite of what happened.
    """
    acct = MultiresBindAccounting()
    bind_resolution_tier(
        _LowBoundCam(loader=_native_loader()), _evicted_record(), "high", accounting=acct
    )
    # ...and a refusal must still land in high_fail, so the tally stays honest
    # in both directions.
    bind_resolution_tier(
        _LowBoundCam(loader=None), _evicted_record(), "high", accounting=acct
    )

    totals = acct.to_dict()["tier_totals"]
    assert totals.get("high_ok") == 1
    assert totals.get("high_fail") == 1
