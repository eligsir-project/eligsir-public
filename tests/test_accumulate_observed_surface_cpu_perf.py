"""End-to-end equivalence test for StreamingScene.accumulate_observed_surface
after the CPU perf fix in issue #270/#276 (fast_nanmedian_axis0 replacing
np.nanmedian inside _stable_depth_samples).

Calls the REAL bound method through a minimal fake StreamingScene-like
harness (mirrors tests/test_streaming_scene_dense_trajectory.py's pattern),
then compares its accumulated state against the same call with
fast_nanmedian_axis0 monkeypatched back to the original np.nanmedian-based
implementation -- i.e. this proves the optimization changed nothing observable
about the method's output, not just that the isolated helper matches
np.nanmedian in unit tests (tests/test_fast_nanmedian.py already covers that).

Guards explicitly against the accumulate_observed_surface `except Exception:
return` silent-failure trap: a broken harness attribute (e.g. missing
_surface_cell_index) makes the method silently no-op, which would make an
equivalence test between "before" and "after" trivially pass by comparing two
empty accumulators. Every test here asserts the accumulators are non-empty
first.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import pytest

from geometry import depth_filtering as depth_filtering_mod
from geometry.surface_cells import SurfaceCellIndex
from scene.streaming_scene import StreamingScene
from eligsir.io.sources.decode_cache import StreamingRGBDFrame
from eligsir.mapping.stores.observation_store import ObservationStore


class _Args:
    streaming_track_scene_complexity = True
    streaming_min_depth = 0.1
    streaming_max_depth = 8.0
    streaming_depth_stride = 8
    streaming_track_input_quality = True
    streaming_depth_filter_enabled = True
    streaming_depth_filter_erode_px = 2
    streaming_depth_filter_window = 5
    streaming_depth_filter_min_valid_ratio = 0.75
    streaming_depth_filter_max_range = 0.08
    streaming_depth_filter_median_thresh = 0.04
    streaming_depth_filter_bottom_margin = 0.0
    streaming_insert_voxel_size = 0.02


class _FakeIngestSelf:
    """Minimal StreamingScene stand-in exposing exactly the real bound
    dependencies accumulate_observed_surface needs -- not a reimplementation."""

    def __init__(self):
        self.args = _Args()
        self._observation_store = ObservationStore(
            obs_voxel=float(self.args.streaming_insert_voxel_size)
        )
        self._surface_cell_index = SurfaceCellIndex(
            voxel_size=float(self.args.streaming_insert_voxel_size)
        )

    accumulate_observed_surface = StreamingScene.accumulate_observed_surface
    _stable_depth_samples = staticmethod(StreamingScene._stable_depth_samples)
    _pos_voxel_keys = staticmethod(StreamingScene._pos_voxel_keys)
    _accumulate_thickness = StreamingScene._accumulate_thickness
    _accumulate_sensor_noise = StreamingScene._accumulate_sensor_noise

    # Property aliases, same wiring as the real StreamingScene.
    _obs_scales = StreamingScene._obs_scales
    _obs_voxel = StreamingScene._obs_voxel
    _obs_keys = StreamingScene._obs_keys
    _obs_vox = StreamingScene._obs_vox
    _obs_n_points = StreamingScene._obs_n_points
    _scene_complexity_cache = StreamingScene._scene_complexity_cache
    _iq_sensor_sum = StreamingScene._iq_sensor_sum
    _iq_sensor_sumsq = StreamingScene._iq_sensor_sumsq
    _iq_sensor_n = StreamingScene._iq_sensor_n
    _iq_sensor_stride = StreamingScene._iq_sensor_stride
    _iq_frame_counter = StreamingScene._iq_frame_counter
    _iq_vox = StreamingScene._iq_vox
    _iq_vox_cap = StreamingScene._iq_vox_cap
    _input_quality_cache = StreamingScene._input_quality_cache


def _make_frame(idx: int, w: int = 320, h: int = 240, seed: int = 0) -> StreamingRGBDFrame:
    """Structured synthetic depth (wall + receding floor + a doorway of
    invalid depth + small sensor noise) -- deliberately not uniform random
    noise, since a uniform-noise frame makes every voxel key unique and
    would hide any dedup-sensitive regression."""
    rng = np.random.default_rng(seed + idx)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    horizon = int(h * 0.55)
    if horizon < h:
        rows = np.arange(horizon, h)
        depth[horizon:h, :] = np.clip(3.0 - (rows - horizon)[:, None] * 0.02, 0.3, 3.0)
    depth[int(h * 0.2):int(h * 0.8), int(w * 0.45):int(w * 0.55)] = 0.0
    depth += rng.normal(0.0, 0.003, size=depth.shape).astype(np.float32)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    depth_mm[depth <= 0] = 0

    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = idx * 0.05
    c2w[2, 3] = idx * 0.02

    fx = fy = w * 0.7
    return StreamingRGBDFrame(
        index=idx, timestamp=float(idx), rgb_path="", depth_path=None,
        c2w=c2w, fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0, width=w, height=h,
        depth_scale=1000.0, _depth_np=depth_mm,
    )


def _snapshot(fs: _FakeIngestSelf) -> dict[str, Any]:
    return {
        "n_points": fs._obs_n_points,
        "keys": {m: frozenset(fs._obs_keys[m]) for m in fs._obs_scales},
        "vox": {m: frozenset(fs._obs_vox[m]) for m in fs._obs_scales},
    }


def _reference_nanmedian(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(x, axis=0)


def _run(n_frames: int, patch_reference: bool = False) -> dict[str, Any]:
    fs = _FakeIngestSelf()
    orig = depth_filtering_mod.fast_nanmedian_axis0
    if patch_reference:
        depth_filtering_mod.fast_nanmedian_axis0 = _reference_nanmedian
    try:
        for i in range(n_frames):
            fs.accumulate_observed_surface(_make_frame(i))
    finally:
        depth_filtering_mod.fast_nanmedian_axis0 = orig
    return _snapshot(fs)


def test_accumulate_observed_surface_advances_state():
    """Guard against the try/except-swallowed-AttributeError trap: a broken
    harness must not silently produce an all-empty accumulator that would
    make the equivalence test below meaningless."""
    snap = _run(n_frames=5)
    assert snap["n_points"] > 0
    for m in (1, 2, 4):
        assert len(snap["keys"][m]) > 0
        assert len(snap["vox"][m]) > 0


def test_fast_nanmedian_optimization_is_output_equivalent():
    """The optimized (current) code path must produce byte-identical
    accumulator state to the pre-optimization np.nanmedian-based path, over
    a realistic multi-frame sequence (not just a single frame)."""
    fast_snap = _run(n_frames=12, patch_reference=False)
    ref_snap = _run(n_frames=12, patch_reference=True)

    assert fast_snap["n_points"] > 0
    assert fast_snap["n_points"] == ref_snap["n_points"]
    for m in (1, 2, 4):
        assert fast_snap["keys"][m] == ref_snap["keys"][m]
        assert fast_snap["vox"][m] == ref_snap["vox"][m]
