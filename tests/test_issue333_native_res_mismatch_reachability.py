"""Is ``unresolved_native_res_mismatch`` SYSTEMATIC today? (issue #333)

Issue #333's closing comment lists this under "unmeasured claims worth
someone's attention":

    ``unresolved_native_res_mismatch`` is proven reachable by construction,
    not observed. The argument that it should be *systematic* on a >1600px-wide
    source under the default ``-r -1`` (where ``_resolve_geometry`` downscales
    the camera to 1600 while the ``high`` tier is built at ``frame.width``)
    looks sound and would mean cache-hit and cache-miss frames train at
    different resolutions today.

This file answers that with executable evidence, driving the REAL
``utils.camera_utils._resolve_geometry`` (not hand-set camera dimensions) into
the REAL ``eligsir.mapping.tier_bind.bind_resolution_tier``.

VERDICT (see the module-level asserts below):

* **NO at defaults.** A streaming run never sees ``args.resolution == -1``:
  ``scene.streaming_scene.resolve_streaming_resolution_pin`` (called from
  ``StreamingScene.__init__``) pins ``args.resolution = 1`` unless the user
  asked for a non-default value -- either via ``--resolution``/``-r`` in
  ``sys.argv``, or (since #584/#618) via a ``--config``/run-profile value
  already sitting on ``args.resolution``. With
  ``resolution=1, resolution_scale=1.0`` ``_resolve_geometry`` is the identity,
  camera size == source/high-tier size, and an evicted cache resolves as
  ``native_fallback`` (``ok=True``) -- ``test_default_streaming_pin_*``.
* **YES whenever an explicit ``--resolution`` != 1 is passed**, and more
  broadly than #333 states: ``-r 2/4/8`` downscales EVERY source, not just
  >1600px-wide ones, so the "wide source" framing understates it
  (``test_explicit_r2_is_systematic_on_any_source``). ``-r -1`` on a >1600px
  source reproduces exactly the scenario quoted above
  (``test_explicit_r_minus1_wide_source_is_systematic``).

All cases take ``bind_resolution_tier``'s non-resident branch, so no CUDA
tensor is ever built. ``import torch`` still arrives transitively via
``utils.camera_utils`` -> ``scene.cameras``, so this file is excluded from the
torch-free CPU CI image (issue #536) exactly like PR #528's files were; it was
run in the training image with ``HIP_VISIBLE_DEVICES=`` masked.
"""
from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

import scene  # noqa: F401  (import the package first: utils.camera_utils <-> scene is circular)

from eligsir.mapping.tier_bind import (
    OUTCOME_NATIVE_FALLBACK,
    OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH,
    bind_resolution_tier,
    describe_startup_resolution_hazard,
)
from eligsir.utils.camera_utils import _resolve_geometry

# The argparse default for ``--resolution`` (``ModelParams._resolution``).
# Imported from the production module rather than re-typed so this file cannot
# drift from it; ``resolve_streaming_resolution_pin`` itself asserts it is -1.
from scene.streaming_scene import _RESOLUTION_ARGPARSE_DEFAULT as _ARGPARSE_RESOLUTION_DEFAULT


class _FakeImage:
    """Stands in for ``cam.original_image``; only ``.shape`` is read."""

    def __init__(self, h: int, w: int) -> None:
        self.shape = (3, h, w)


def _cam_info(width: int, height: int):
    return SimpleNamespace(
        image=None, width=width, height=height,
        fx=float(width) / 2.0, fy=float(height) / 2.0,
        cx=width / 2.0, cy=height / 2.0,
    )


def _evicted_record(frame_id: int = 7):
    """A FrameRecord-shaped stub whose bounded LRU has evicted its tiers."""
    return SimpleNamespace(
        frame_id=frame_id,
        resolution_cache={},
        resolution_cache_evicted=True,
        resolution_cache_absent_reason=None,
    )


def _camera_for(source_w: int, source_h: int, resolution, resolution_scale=1.0):
    """Build the camera the production path would build for this source frame.

    Geometry comes from the REAL ``_resolve_geometry`` (the function
    ``loadCam`` calls), so the camera's ``image_width``/``image_height`` are
    whatever a real run would give it. ``_streaming_frame`` carries the source
    frame's declared size, which is what ``build_resolution_cache`` builds the
    ``high`` tier at.
    """
    args = SimpleNamespace(resolution=resolution, data_device="cpu")
    _, _, resolved, _ = _resolve_geometry(
        args, _cam_info(source_w, source_h), resolution_scale
    )
    w, h = int(resolved[0]), int(resolved[1])
    cam = SimpleNamespace(
        image_width=w, image_height=h,
        original_image=_FakeImage(h, w),
        _streaming_frame=SimpleNamespace(width=source_w, height=source_h),
        _bound_resolution_tier=None,
    )
    return cam, (w, h)


# --------------------------------------------------------------------------
# The falsifier: today's default streaming configuration.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source_w,source_h", [(1920, 1080), (1296, 968), (640, 480)])
def test_default_streaming_pin_resolves_native_fallback(source_w, source_h):
    """resolution=1 (the streaming pin) => camera size == high-tier size.

    This is the evidence that the "systematic today" claim is FALSE for every
    recipe in this repo: no spec/launch script passes --resolution at all
    (verified by grep over specs/ human_launch_scripts/ configs/), so
    StreamingScene's pin applies and _resolve_geometry is the identity --
    INCLUDING on the >1600px-wide source the claim is about.
    """
    cam, resolved = _camera_for(source_w, source_h, resolution=1)
    assert resolved == (source_w, source_h)  # identity: no downscale at all

    result = bind_resolution_tier(cam, _evicted_record(), "high")
    assert result.outcome == OUTCOME_NATIVE_FALLBACK
    assert result.ok is True and result.usable is True
    assert result.tier_after == "high"


def test_default_pin_has_no_startup_hazard():
    assert describe_startup_resolution_hazard(1, 1.0) is None


def test_streaming_pin_still_applies_at_defaults():
    """Tripwire: the verdict above is load-bearing on this pin existing.

    If the streaming path ever stops forcing ``args.resolution = 1`` at
    DEFAULTS, the default streaming run inherits ``-r -1``'s >1600px
    auto-downscale and the hazard measured below becomes systematic at
    defaults. Fail loudly here rather than silently changing what this file
    proves.

    Issue #622: this used to scrape ``inspect.getsource(StreamingScene.__init__)``
    for a literal ``args.resolution = 1`` assignment. #618 moved the decision
    into the pure function ``resolve_streaming_resolution_pin``, so the string
    vanished from ``__init__`` while the BEHAVIOUR was preserved -- the
    tripwire fired on a refactor rather than on a real change. It now asserts
    the behaviour directly, against the function that owns the decision.
    """
    from scene.streaming_scene import resolve_streaming_resolution_pin

    args = SimpleNamespace(resolution=_ARGPARSE_RESOLUTION_DEFAULT)
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=[])

    assert (effective, pinned, source) == (1, True, "default"), (
        "issue #333: the streaming path no longer pins resolution=1 at "
        "defaults -- the native-res tier mismatch may now be systematic at "
        "DEFAULTS; re-run this file's reasoning. "
        f"got effective={effective!r} pinned={pinned!r} source={source!r}"
    )


def test_streaming_scene_still_consults_the_pin_resolver():
    """Second half of the tripwire: the resolver must still be REACHED.

    ``test_streaming_pin_still_applies_at_defaults`` proves the resolver
    returns the right answer; this proves ``StreamingScene.__init__`` still
    asks it. Without both, a refactor that stops calling the resolver would
    leave the first test passing while production silently regressed -- the
    exact failure mode the original source-scraping tripwire was reaching for.
    """
    from scene.streaming_scene import StreamingScene

    src = inspect.getsource(StreamingScene.__init__)
    assert re.search(r"\bresolve_streaming_resolution_pin\s*\(", src), (
        "issue #333/#622: StreamingScene.__init__ no longer calls "
        "resolve_streaming_resolution_pin -- whatever decides the streaming "
        "resolution pin now is untested by this file."
    )


def _pin_fires(argv) -> bool:
    """Execute the REAL pin decision on ``argv``, at the argparse default.

    Calls the production function directly (issue #622). The previous version
    regex-extracted the ``if not any(...)`` guard out of
    ``StreamingScene.__init__``'s source and ``eval``'d it; that construct
    broke the moment #618 refactored the guard into a named function, and it
    could only ever test the argv half of the decision anyway.
    """
    from scene.streaming_scene import resolve_streaming_resolution_pin

    args = SimpleNamespace(resolution=_ARGPARSE_RESOLUTION_DEFAULT)
    _effective, pinned, _source = resolve_streaming_resolution_pin(args, argv=list(argv))
    return pinned


@pytest.mark.parametrize(
    "argv,pinned",
    [
        ([], True),
        (["--iterations", "30000", "--streaming_replay"], True),
        (["-r", "-1"], False),
        (["-r", "1"], False),
        (["--resolution", "-1"], False),
        (["--resolution=-1"], False),
        (["-r=2"], False),
    ],
)
def test_pin_predicate_fires_exactly_when_no_resolution_flag(argv, pinned):
    """Executed (not merely read): the pin fires for every argv WITHOUT -r.

    This is the half of the verdict that says a default streaming run never
    sees ``args.resolution == -1``. Both the ``-r x`` and ``--resolution=x``
    spellings are covered.
    """
    assert _pin_fires(argv) is pinned


@pytest.mark.parametrize("resolution", [1, 2, 4, 8])
def test_config_supplied_resolution_defeats_the_pin(resolution):
    """A --config/run-profile resolution is honoured, not clobbered (#584/#618).

    This widens what #333's verdict has to say: the pin no longer fires only
    on an argv flag, so "the default streaming run is pinned to 1" is true
    only while nothing has written a non-default value onto ``args`` from a
    config file or YAML run profile. Such a run is subject to the SAME hazard
    an explicit ``-r`` is (``test_explicit_r2_is_systematic_on_any_source``).
    """
    from scene.streaming_scene import resolve_streaming_resolution_pin

    args = SimpleNamespace(resolution=resolution)
    effective, pinned, source = resolve_streaming_resolution_pin(args, argv=[])

    assert effective == resolution
    assert pinned is False
    assert source == "config_or_profile"


# --------------------------------------------------------------------------
# Reachability: what an explicit --resolution actually does.
# --------------------------------------------------------------------------

def test_explicit_r_minus1_wide_source_is_systematic():
    """The exact scenario issue #333 asks about: -r -1 on a >1600px source."""
    cam, resolved = _camera_for(1920, 1080, resolution=-1)
    assert resolved == (1600, 900)  # the >1.6K auto-downscale really fires

    result = bind_resolution_tier(cam, _evicted_record(), "high")
    assert result.outcome == OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH
    assert result.ok is False and result.usable is False
    assert "1600" in (result.detail or "") and "1920" in (result.detail or "")


def test_explicit_r_minus1_narrow_source_is_safe():
    """-r -1 below 1600px wide is a no-op scale, so no mismatch."""
    cam, resolved = _camera_for(1280, 720, resolution=-1)
    assert resolved == (1280, 720)
    assert bind_resolution_tier(cam, _evicted_record(), "high").outcome == OUTCOME_NATIVE_FALLBACK


@pytest.mark.parametrize("res", [2, 4, 8])
@pytest.mark.parametrize("source_w,source_h", [(1920, 1080), (640, 480)])
def test_explicit_r2_is_systematic_on_any_source(res, source_w, source_h):
    """Broader than #333 records: an explicit -r 2/4/8 mismatches EVERY source.

    #333 frames the hazard as ">1600px-wide under -r -1". The high tier is
    always divisor=1 at frame.width, so any non-identity camera downscale
    mismatches it -- source width is irrelevant.
    """
    cam, resolved = _camera_for(source_w, source_h, resolution=res)
    assert resolved == (source_w // res, source_h // res)
    assert bind_resolution_tier(cam, _evicted_record(), "high").outcome == (
        OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH
    )


def test_resolution_scale_alone_is_systematic():
    """resolution_scale != 1.0 mismatches even with the resolution=1 pin."""
    cam, resolved = _camera_for(1280, 720, resolution=1, resolution_scale=2.0)
    assert resolved == (640, 360)
    assert bind_resolution_tier(cam, _evicted_record(), "high").outcome == (
        OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH
    )


# --------------------------------------------------------------------------
# The startup warning that makes the above self-announcing.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "resolution,scale", [(-1, 1.0), (2, 1.0), (4, 1.0), (1, 2.0), (800, 1.0)]
)
def test_startup_hazard_warns_for_every_non_identity_config(resolution, scale):
    msg = describe_startup_resolution_hazard(resolution, scale)
    assert msg is not None
    assert OUTCOME_UNRESOLVED_NATIVE_RES_MISMATCH in msg
    assert "issue #333" in msg


def test_startup_hazard_is_none_on_garbage_input():
    """Never crash a live run over a telemetry warning."""
    assert describe_startup_resolution_hazard(None, 1.0) is None
    assert describe_startup_resolution_hazard(1, "nope") is None
