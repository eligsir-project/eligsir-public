"""``ReplayViewStore`` — owns replay/global-reservoir sampling state,
extracted from `scene/streaming_scene.py`'s `_replay_buffer`/
`_global_reservoir`/`_priority_frame_*`/`_consol_debt_*` attributes.

Wave 1 consolidation plan step 9 (see
``docs/architecture/streaming_scene_decomposition.md``). Pure state
container, same pattern as `TrajectoryStore`/`ObservationStore`: the
sampling logic (`_sample_consolidation`, `_sample_global_with_debt`,
partition-focus weighting) stays on `StreamingScene` as methods, reading
through property aliases onto this store.
"""

from __future__ import annotations

import math
from typing import List, Optional

from eligsir.utils.partition_eval import camera_source_index


def _camera_indices(cameras):
    indices = []
    for camera in cameras:
        index = camera_source_index(camera)
        if index is None:
            raise ValueError("camera has no stable source index")
        indices.append(int(index))
    return indices


def _resolve_cameras(resolver, indices):
    cameras = []
    for raw_index in indices:
        index = int(raw_index)
        camera = resolver(index)
        if camera is None or camera_source_index(camera) != index:
            raise ValueError(f"could not resolve camera source index {index}")
        cameras.append(camera)
    return cameras


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _index_list(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_integer(index, name) for index in value]


def _weight(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


class ReplayViewStore:
    """Owns the replay ring buffer, global reservoir, partition-focus
    priority weighting, and consolidation debt-weight cache. One instance
    per `StreamingScene`."""

    def __init__(self):
        self.replay_buffer: List = []
        self.global_reservoir: List = []
        self.priority_frame_indices: "set[int]" = set()
        self.priority_frame_weights: "dict[int, float]" = {}

        # Consolidation debt-weight cache -- lazily populated (see
        # StreamingScene._sample_consolidation), defaults to "empty" so the
        # getattr(self, ..., default) pattern at existing call sites is
        # unaffected by always having a real (if initially empty) value.
        self.consol_debt_weights: Optional[list] = None
        self.consol_debt_pool_len: int = -1
        self.consol_debt_age: int = 0

    def resume_state_dict(self):
        return {
            "owner_schema": "replay_view_store_v1",
            "replay_source_indices": _camera_indices(self.replay_buffer),
            "reservoir_source_indices": _camera_indices(self.global_reservoir),
            "priority_frame_indices": sorted(int(index) for index in self.priority_frame_indices),
            "priority_frame_weights": {
                int(index): float(weight)
                for index, weight in self.priority_frame_weights.items()
            },
            "consol_debt_weights": None
            if self.consol_debt_weights is None
            else [float(weight) for weight in self.consol_debt_weights],
            "consol_debt_pool_len": int(self.consol_debt_pool_len),
            "consol_debt_age": int(self.consol_debt_age),
        }

    def load_resume_state_dict(self, payload, resolver):
        required = {
            "owner_schema",
            "replay_source_indices", "reservoir_source_indices",
            "priority_frame_indices", "priority_frame_weights",
            "consol_debt_weights", "consol_debt_pool_len", "consol_debt_age",
        }
        if not isinstance(payload, dict) or payload.get("owner_schema") != "replay_view_store_v1":
            raise ValueError("incompatible replay view store resume state")
        if set(payload) != required or not callable(resolver):
            raise ValueError("incomplete replay view store resume state")

        replay_indices = _index_list(payload["replay_source_indices"], "replay_source_indices")
        reservoir_indices = _index_list(
            payload["reservoir_source_indices"], "reservoir_source_indices"
        )
        priority_indices = set(
            _index_list(payload["priority_frame_indices"], "priority_frame_indices")
        )
        raw_priority_weights = payload["priority_frame_weights"]
        if not isinstance(raw_priority_weights, dict):
            raise ValueError("priority_frame_weights must be a dictionary")
        priority_weights = {
            _integer(index, "priority_frame_weights key"): _weight(
                weight, "priority_frame_weights value"
            )
            for index, weight in raw_priority_weights.items()
        }
        raw_debt_weights = payload["consol_debt_weights"]
        if raw_debt_weights is not None and not isinstance(raw_debt_weights, list):
            raise ValueError("consol_debt_weights must be a list or None")
        debt_weights = (
            None
            if raw_debt_weights is None
            else [_weight(weight, "consol_debt_weights") for weight in raw_debt_weights]
        )
        debt_pool_len = _integer(
            payload["consol_debt_pool_len"], "consol_debt_pool_len", minimum=-1
        )
        debt_age = _integer(payload["consol_debt_age"], "consol_debt_age")

        replay = _resolve_cameras(resolver, replay_indices)
        reservoir = _resolve_cameras(resolver, reservoir_indices)
        self.replay_buffer = replay
        self.global_reservoir = reservoir
        self.priority_frame_indices = priority_indices
        self.priority_frame_weights = priority_weights
        self.consol_debt_weights = debt_weights
        self.consol_debt_pool_len = debt_pool_len
        self.consol_debt_age = debt_age
