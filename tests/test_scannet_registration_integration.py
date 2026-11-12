"""Real-scene integration test: ScanNet depth->colour registration end to end.

Skipped unless the real, extracted ScanNet v2 scene0000_01 layout (color/ +
depth/ + pose/ + intrinsic/) is present at SCENE_PATH -- the read-only mount
point used by the task's proof docker command:

    docker run --rm --user $(id -u):$(id -g) -e PYTHONPATH=/mywork \
        -v <repo>:/mywork:z \
        -v /data/datasets/scannet/scene0000_01:/data/scene:ro,z \
        -w /workspace "$ELIGSIR_IMAGE" \
        python3 -m pytest tests/test_scannet_registration_integration.py -q

Exercises the SAME functions the `--execution_trainer causal_mapping`
runtime uses to go from a raw ScanNet frame to a canonical MappingFrame
(``mapping_frame_from_streaming_frame`` / ``mapping_frame_source_arrays`` in
eligsir/contracts/streaming_bridge.py, and
``eligsir.io.sources.decode_cache.load_frame_depth_np``), plus the
cold-pixel-reload path that regressed at f8acd91b: production_application.py's
``_decode_reload_token`` rebuilds a bare ``SimpleNamespace(rgb_path=...,
depth_path=...)`` from a persisted ``PixelReloadToken`` and re-decodes depth
through that stripped object rather than the original StreamingRGBDFrame. See
tests/test_scannet_source.py::test_scannet_depth_registration_survives_cold_reload_shim
for the fast synthetic-fixture version of that same regression check.

Set ``ELIGSIR_IMAGE`` to an authorized local or registry image before using
the example. This file does not modify, and only imports (read-only) from,
eligsir/contracts/streaming_bridge.py and eligsir/runtime/.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

SCENE_PATH = "/data/scene"
COLOR_HW = (968, 1296)  # (height, width), scene0000_01's colour resolution


def _scene_present() -> bool:
    return (
        os.path.isdir(os.path.join(SCENE_PATH, "color"))
        and os.path.isdir(os.path.join(SCENE_PATH, "depth"))
        and os.path.isdir(os.path.join(SCENE_PATH, "pose"))
        and os.path.isdir(os.path.join(SCENE_PATH, "intrinsic"))
    )


pytestmark = pytest.mark.skipif(
    not _scene_present(),
    reason=f"real ScanNet scene not present at {SCENE_PATH}",
)


def _build_source():
    """Registry auto-detection, default flags, first ~5 frames, stride 1."""
    from eligsir.io.sources.registry import make_frame_source

    args = SimpleNamespace(
        streaming_max_frames=5,
        scannet_frame_stride=1,
        # scannet_depth_to_color intentionally left unset so the registry's
        # own default ("resize") applies -- exactly what a plain
        # `--execution_trainer causal_mapping` run gets.
    )
    source = make_frame_source(SCENE_PATH, args)
    frames = source.get_all() if hasattr(source, "get_all") else list(source)
    assert len(frames) > 0, "expected at least one frame from the real scene"
    return source, frames


def test_scannet_real_scene_source_type_and_frame_count():
    from eligsir.io.sources.datasets.scannet import ScanNetFrameSource

    source, frames = _build_source()
    assert isinstance(source, ScanNetFrameSource), (
        "registry auto-detection should pick the extracted-layout "
        f"ScanNetFrameSource for {SCENE_PATH}, got {type(source)}"
    )
    assert len(frames) == 5


def test_scannet_real_scene_registration_contract():
    """Build a MappingFrame the same way the runtime does and run it through
    the canonical insertion contract -- must not raise, and must report
    colour-registered geometry end to end."""
    import torch

    from eligsir.contracts.streaming_bridge import (
        mapping_frame_from_streaming_frame,
        mapping_frame_source_arrays,
    )
    from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb

    _, frames = _build_source()
    frame = frames[0]

    assert (frame.height, frame.width) == COLOR_HW
    assert (frame.depth_height, frame.depth_width) == COLOR_HW

    # Depth intrinsics must be EXACTLY equal to colour intrinsics (canonical
    # insertion's shared-pixel-geometry requirement) -- == not approx.
    assert frame.depth_fx == frame.fx
    assert frame.depth_fy == frame.fy
    assert frame.depth_cx == frame.cx
    assert frame.depth_cy == frame.cy

    rgb_np = np.asarray(load_frame_rgb(frame), dtype=np.uint8)
    assert rgb_np.shape[:2] == COLOR_HW

    depth_raw = load_frame_depth_np(frame)
    assert depth_raw is not None
    assert tuple(depth_raw.shape) == COLOR_HW, (
        f"registered depth must be resized to colour resolution, got {depth_raw.shape}"
    )

    # Plausible ScanNet indoor depth: median of valid (nonzero) pixels
    # between 1000-4000mm raw (nearest resize never changes values).
    valid = depth_raw[depth_raw > 0]
    assert valid.size > 0, "expected at least some valid (nonzero) depth pixels"
    median_mm = float(np.median(valid))
    assert 1000.0 <= median_mm <= 4000.0, (
        f"median valid depth {median_mm:.1f}mm outside plausible ScanNet indoor range"
    )

    rgb_chw = torch.from_numpy(rgb_np.copy()).permute(2, 0, 1).contiguous().float() / 255.0

    mapping_frame = mapping_frame_from_streaming_frame(frame, rgb_chw, depth_raw)
    assert str(mapping_frame.device) == "cpu"

    # Intrinsics carried on the canonical MappingFrame must still be exactly
    # equal, not just the source frame's own dataclass fields.
    intr = mapping_frame.intrinsics
    assert intr.depth_fx == intr.fx
    assert intr.depth_fy == intr.fy
    assert intr.depth_cx == intr.cx
    assert intr.depth_cy == intr.cy
    assert (intr.depth_height, intr.depth_width) == COLOR_HW

    frame._mapping_frame = mapping_frame
    depth_arr, rgb_arr = mapping_frame_source_arrays(frame)  # must not raise

    assert rgb_arr.shape[:2] == COLOR_HW
    assert depth_arr.shape == COLOR_HW

    # depth_arr is metric float32 metres post-materialization.
    valid_m = depth_arr[depth_arr > 0]
    assert valid_m.size > 0
    median_m = float(np.median(valid_m))
    assert 1.0 <= median_m <= 4.0, (
        f"median valid metric depth {median_m:.2f}m outside plausible ScanNet indoor range"
    )


def test_scannet_real_scene_depth_survives_cold_reload_shim():
    """The regression this task exists to fix, reproduced on the real scene:
    a frame stripped down to a bare rgb_path/depth_path SimpleNamespace --
    exactly what production_application.py's ``_decode_reload_token`` builds
    from a persisted ``PixelReloadToken`` -- must still decode
    colour-registered depth.

    This is the assertion that actually exercises the fix: the task's other
    enumerated checks (direct load_frame_depth_np on the real frame, the
    canonical-insertion contract) already passed before this fix, because
    the real StreamingRGBDFrame instance always carried the attributes the
    old, attribute-keyed resize decision needed. Only a stripped
    frame-like object -- what the cold-pixel-reload path actually uses --
    exposes the bug.
    """
    from eligsir.io.sources.decode_cache import load_frame_depth_np, load_frame_rgb
    from eligsir.runtime.replay import packet_from_streaming_frame

    _, frames = _build_source()
    frame = frames[0]

    rgb = np.asarray(load_frame_rgb(frame), dtype=np.uint8)
    packet = packet_from_streaming_frame(
        frame, source_id="scannet-real-cold-reload-test", rgb=rgb
    )
    token = packet.reload_token
    assert token is not None, "expected an rgbd_files_v1 reload token for a file-backed frame"
    assert token.provider == "rgbd_files_v1"

    # Exactly the SimpleNamespace production_application.py._decode_reload_token
    # builds from a bare PixelReloadToken.
    shim = SimpleNamespace(
        rgb_path=token.locators[0],
        depth_path=token.locators[1] or None,
        _rgb_np=None,
        _rgb_bytes=None,
        _depth_np=None,
        _depth_bytes=None,
        _sens_header=None,
    )
    reloaded_depth = load_frame_depth_np(shim)

    assert reloaded_depth is not None
    assert tuple(reloaded_depth.shape) == COLOR_HW, (
        "cold-reload shim must decode colour-registered depth on the real "
        f"scene, got {reloaded_depth.shape}"
    )


def test_scannet_real_scene_direct_construction_matches_registry():
    """Sanity: the registry-selected source behaves exactly like directly
    constructing ScanNetFrameSource (no divergence introduced by going
    through make_frame_source's auto-detection)."""
    from eligsir.io.sources.datasets.scannet import ScanNetFrameSource

    source = ScanNetFrameSource(
        SCENE_PATH, max_frames=5, frame_stride=1, depth_to_color="resize"
    )
    frames = source.get_all()
    assert len(frames) == 5
    for frame in frames:
        assert (frame.depth_height, frame.depth_width) == COLOR_HW
        assert frame.depth_fx == frame.fx and frame.depth_fy == frame.fy
        assert frame.depth_cx == frame.cx and frame.depth_cy == frame.cy


def test_source_native_resolution_real_scannet():
    """Verify source_native_resolution helper correctly derives ScanNet resolution."""
    from eligsir.runtime.config_boundary import source_native_resolution

    source, frames = _build_source()
    res, method = source_native_resolution(source)
    assert res == (1296, 968), f"Expected (1296, 968), got {res}"
    assert method == "first_frame_record", (
        f"Expected 'first_frame_record' method, got {method}"
    )


def test_payload_budget_resolution_scaled_real_scannet():
    """Verify payload budget correctly scales for real ScanNet resolution."""
    from eligsir.runtime.config_boundary import (
        source_native_resolution,
        resolve_admitted_payload_bounds,
    )

    source, _ = _build_source()
    res, method = source_native_resolution(source)
    assert res == (1296, 968)

    # Now test that the payload budget correctly scales for this resolution
    args = SimpleNamespace(
        streaming_payload_budget_policy="resolution_scaled",
        streaming_live_admitted_payload_max_frames=None,
        streaming_hot_tier_max_bytes=None,
    )
    frames, hot_bytes, metadata = resolve_admitted_payload_bounds(
        args, native_width=res[0], native_height=res[1]
    )
    # 512 * (640*480) / (1296*968) ≈ 125.09 -> 125
    assert frames == 125, f"Expected 125 frames for 1296x968, got {frames}"
    assert metadata["source"] == "resolution_scaled"
    assert metadata["native_resolution"] == "1296x968"


def test_payload_budget_tum_baseline_640x480():
    """Verify baseline TUM resolution (640x480) yields 512 frames."""
    from eligsir.runtime.config_boundary import resolve_admitted_payload_bounds

    args = SimpleNamespace(
        streaming_payload_budget_policy="resolution_scaled",
        streaming_live_admitted_payload_max_frames=None,
        streaming_hot_tier_max_bytes=None,
    )
    frames, hot_bytes, metadata = resolve_admitted_payload_bounds(
        args, native_width=640, native_height=480
    )
    assert frames == 512, f"Expected 512 frames for 640x480 baseline, got {frames}"
    assert metadata["source"] == "resolution_scaled"
