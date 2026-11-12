"""``ObservationStore`` — owns per-scene observation-accumulation and
input-quality-tracking state, extracted from `scene/streaming_scene.py`'s
`_obs_*`/`_iq_*`/`_scene_complexity_cache`/`observation_db` attributes.

Wave 1 consolidation plan step 9 (see
``docs/architecture/streaming_scene_decomposition.md``). Unlike
`TrajectoryStore`, this is a pure attribute/state container -- the
accumulation logic (`accumulate_observed_surface`, box-counting scene
complexity, sensor-noise-vs-tracking-error input-quality report) stays on
`StreamingScene` as methods, reading/writing through property aliases onto
this store, since that logic is deeply interleaved with other
`StreamingScene` state (frame pose gating, `_pos_voxel_keys`) not worth
destabilizing in the same change. Real ownership move: the actual mutable
state now lives in one instance of this class, not scattered `self.*`
attributes.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional


class ObservationStore:
    """Owns observation-accumulation (multiscale voxel coverage) and
    input-quality (sensor-noise/tracking-error) tracking state, plus the
    optional `FrameObservationDB` reference. One instance per
    `StreamingScene`."""

    def __init__(self, obs_voxel: float):
        self.observation_db = None

        self.obs_scales = (1, 2, 4)
        self.obs_voxel = obs_voxel
        self.obs_keys = {m: set() for m in self.obs_scales}
        self.obs_vox = {m: set() for m in self.obs_scales}
        self.obs_n_points = 0
        self.scene_complexity_cache: Optional[dict] = None

        self.iq_sensor_sum = 0.0
        self.iq_sensor_sumsq = 0.0
        self.iq_sensor_n = 0
        self.iq_sensor_stride = 8
        self.iq_frame_counter = 0
        self.iq_vox: dict = {}
        self.iq_vox_cap = 120000
        self.input_quality_cache: Optional[Any] = None

    _RESUME_SCHEMA = "observation_store_v1"

    def resume_state_dict(self) -> dict:
        return {
            "owner_schema": self._RESUME_SCHEMA,
            "obs_scales": list(self.obs_scales),
            "obs_voxel": float(self.obs_voxel),
            "obs_keys": {m: sorted(values) for m, values in self.obs_keys.items()},
            "obs_vox": {m: sorted(values) for m, values in self.obs_vox.items()},
            "obs_n_points": int(self.obs_n_points),
            "scene_complexity_cache": copy.deepcopy(self.scene_complexity_cache),
            "iq_sensor_sum": float(self.iq_sensor_sum),
            "iq_sensor_sumsq": float(self.iq_sensor_sumsq),
            "iq_sensor_n": int(self.iq_sensor_n),
            "iq_sensor_stride": int(self.iq_sensor_stride),
            "iq_frame_counter": int(self.iq_frame_counter),
            "iq_vox": copy.deepcopy(self.iq_vox),
            "iq_vox_cap": int(self.iq_vox_cap),
            "input_quality_cache": copy.deepcopy(self.input_quality_cache),
        }

    def load_resume_state_dict(self, payload, resolver=None) -> None:
        expected = set(self.resume_state_dict())
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("owner_schema") != self._RESUME_SCHEMA
        ):
            raise ValueError("unsupported observation store checkpoint state")
        scales = payload["obs_scales"]
        if (
            not isinstance(scales, list)
            or not scales
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in scales)
            or len(set(scales)) != len(scales)
        ):
            raise ValueError("observation store obs_scales is invalid")
        voxel = payload["obs_voxel"]
        if (
            isinstance(voxel, bool)
            or not isinstance(voxel, (int, float))
            or not math.isfinite(float(voxel))
            or float(voxel) <= 0.0
        ):
            raise ValueError("observation store obs_voxel is invalid")

        def validate_sets(name):
            values = payload[name]
            if not isinstance(values, dict) or set(values) != set(scales):
                raise ValueError(f"observation store {name} scales mismatch")
            restored = {}
            for scale, items in values.items():
                if not isinstance(items, list) or any(
                    isinstance(v, bool) or not isinstance(v, int) for v in items
                ):
                    raise ValueError(f"observation store {name} is invalid")
                restored[scale] = set(items)
            return restored

        obs_keys = validate_sets("obs_keys")
        obs_vox = validate_sets("obs_vox")

        def nonnegative_int(name, *, positive=False):
            value = payload[name]
            minimum = 1 if positive else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"observation store {name} is invalid")
            return value

        def finite_number(name, *, nonnegative=False):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"observation store {name} is invalid")
            value = float(value)
            if not math.isfinite(value) or (nonnegative and value < 0.0):
                raise ValueError(f"observation store {name} is invalid")
            return value

        obs_n_points = nonnegative_int("obs_n_points")
        iq_sensor_sum = finite_number("iq_sensor_sum", nonnegative=True)
        iq_sensor_sumsq = finite_number("iq_sensor_sumsq", nonnegative=True)
        iq_sensor_n = nonnegative_int("iq_sensor_n")
        iq_sensor_stride = nonnegative_int("iq_sensor_stride", positive=True)
        iq_frame_counter = nonnegative_int("iq_frame_counter")
        iq_vox_cap = nonnegative_int("iq_vox_cap", positive=True)
        iq_vox_payload = payload["iq_vox"]
        if not isinstance(iq_vox_payload, dict) or len(iq_vox_payload) > iq_vox_cap:
            raise ValueError("observation store iq_vox is invalid")
        iq_vox = {}
        for key, values in iq_vox_payload.items():
            if (
                isinstance(key, bool)
                or not isinstance(key, int)
                or not isinstance(values, list)
                or len(values) != 10
                or isinstance(values[0], bool)
                or not isinstance(values[0], int)
                or values[0] <= 0
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values[1:]
                )
            ):
                raise ValueError("observation store iq_vox is invalid")
            iq_vox[key] = copy.deepcopy(values)
        for cache_name in ("scene_complexity_cache", "input_quality_cache"):
            cache = payload[cache_name]
            if cache is not None and not isinstance(cache, dict):
                raise ValueError(f"observation store {cache_name} is invalid")

        self.obs_scales = tuple(scales)
        self.obs_voxel = float(voxel)
        self.obs_keys = obs_keys
        self.obs_vox = obs_vox
        self.obs_n_points = obs_n_points
        self.scene_complexity_cache = copy.deepcopy(payload["scene_complexity_cache"])
        self.iq_sensor_sum = iq_sensor_sum
        self.iq_sensor_sumsq = iq_sensor_sumsq
        self.iq_sensor_n = iq_sensor_n
        self.iq_sensor_stride = iq_sensor_stride
        self.iq_frame_counter = iq_frame_counter
        self.iq_vox = iq_vox
        self.iq_vox_cap = iq_vox_cap
        self.input_quality_cache = copy.deepcopy(payload["input_quality_cache"])
