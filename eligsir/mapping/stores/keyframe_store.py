"""``KeyframeStore`` — owns keyframe camera bookkeeping and admission
state, extracted from `scene/streaming_scene.py`'s `train_cameras`/
`_test_cameras`/camera-cache/admission-counter attributes.

Wave 1 consolidation plan step 9 (see
``docs/architecture/streaming_scene_decomposition.md``). Pure state
container, same pattern as the other five stores: the logic that operates
on this state (`initialize_from_frames`, `ingest_next_frame`,
`_ensure_camera_caches_np`, `_estimate_cameras_extent`,
`_designate_test_holdouts`, `get_nearest_cameras`) stays on
`StreamingScene` as methods, reading/writing through property aliases
onto this store.

Full scope, landed in two passes: the five admission counters
(`keyframes_admitted`/`keyframes_rejected`/`last_admitted_source_idx`/
`bootstrap_count`/`n_train_frames_ingested`) first (lower external
footprint), then camera bookkeeping (`train_cameras`/`_test_cameras`/
camera-position/viewdir caches/`current_camera`/`cameras_extent`/
`_holdout_indices`/`_guard_indices`) once every external
``object.__new__(StreamingScene)``-bypassing test construction and the
two ``train_streaming.py`` scratch-scene sites were identified and
patched (see the plan doc's "Deferred scope" section, since resolved, for
the full list).
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


def _resolve_camera(resolver, source_index):
    camera = resolver(source_index)
    if camera is None or camera_source_index(camera) != source_index:
        raise ValueError(f"could not resolve camera source index {source_index}")
    return camera


def _integer(value, name, *, minimum=0, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not int or value < minimum:
        suffix = " or None" if nullable else ""
        raise ValueError(f"{name} must be an integer >= {minimum}{suffix}")
    return value


def _index_list(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_integer(index, name) for index in value]


class KeyframeStore:
    """Owns keyframe camera bookkeeping (admitted train cameras, test/
    holdout cameras, camera-position/viewdir caches, admission counters,
    and the tracked scene extent). One instance per `StreamingScene`."""

    def __init__(self):
        # Admission counters
        self.keyframes_admitted: int = 0
        self.keyframes_rejected: int = 0
        self.last_admitted_source_idx: Optional[int] = None
        self.bootstrap_count: int = 0
        self.n_train_frames_ingested: int = 0

        # Camera bookkeeping
        self.train_cameras: List = []
        self.test_cameras: List = []
        self.camera_centers_np: List = []
        self.camera_centers_gpu: List = []
        self.camera_viewdirs_np: List = []
        self.current_camera = None
        self.cameras_extent: float = 1.0

        # Holdout/guard index sets (eval split)
        self.holdout_indices: Optional[set] = None
        self.guard_indices: set = set()

    def resume_state_dict(self):
        current_index = None
        if self.current_camera is not None:
            current_index = _camera_indices([self.current_camera])[0]
        return {
            "owner_schema": "keyframe_store_v1",
            "train_source_indices": _camera_indices(self.train_cameras),
            "test_source_indices": _camera_indices(self.test_cameras),
            "current_source_index": current_index,
            "keyframes_admitted": int(self.keyframes_admitted),
            "keyframes_rejected": int(self.keyframes_rejected),
            "last_admitted_source_idx": self.last_admitted_source_idx,
            "bootstrap_count": int(self.bootstrap_count),
            "n_train_frames_ingested": int(self.n_train_frames_ingested),
            "cameras_extent": float(self.cameras_extent),
            "holdout_indices": None
            if self.holdout_indices is None
            else sorted(int(index) for index in self.holdout_indices),
            "guard_indices": sorted(int(index) for index in self.guard_indices),
        }

    def load_resume_state_dict(self, payload, resolver):
        required = {
            "owner_schema",
            "train_source_indices", "test_source_indices", "current_source_index",
            "keyframes_admitted", "keyframes_rejected", "last_admitted_source_idx",
            "bootstrap_count", "n_train_frames_ingested", "cameras_extent",
            "holdout_indices", "guard_indices",
        }
        if not isinstance(payload, dict) or payload.get("owner_schema") != "keyframe_store_v1":
            raise ValueError("incompatible keyframe store resume state")
        if set(payload) != required or not callable(resolver):
            raise ValueError("incomplete keyframe store resume state")

        train_indices = _index_list(payload["train_source_indices"], "train_source_indices")
        test_indices = _index_list(payload["test_source_indices"], "test_source_indices")
        current_index = _integer(
            payload["current_source_index"], "current_source_index", nullable=True
        )
        keyframes_admitted = _integer(payload["keyframes_admitted"], "keyframes_admitted")
        keyframes_rejected = _integer(payload["keyframes_rejected"], "keyframes_rejected")
        last_admitted = _integer(
            payload["last_admitted_source_idx"],
            "last_admitted_source_idx",
            nullable=True,
        )
        bootstrap_count = _integer(payload["bootstrap_count"], "bootstrap_count")
        frames_ingested = _integer(
            payload["n_train_frames_ingested"], "n_train_frames_ingested"
        )
        extent_value = payload["cameras_extent"]
        if (
            isinstance(extent_value, bool)
            or not isinstance(extent_value, (int, float))
            or not math.isfinite(extent_value)
            or extent_value <= 0
        ):
            raise ValueError("cameras_extent must be a finite positive number")
        cameras_extent = float(extent_value)
        holdout_value = payload["holdout_indices"]
        holdout = (
            None
            if holdout_value is None
            else set(_index_list(holdout_value, "holdout_indices"))
        )
        guard = set(_index_list(payload["guard_indices"], "guard_indices"))

        train = [_resolve_camera(resolver, index) for index in train_indices]
        test = [_resolve_camera(resolver, index) for index in test_indices]
        current = None if current_index is None else _resolve_camera(resolver, current_index)

        self.train_cameras = train
        self.test_cameras = test
        self.current_camera = current
        self.keyframes_admitted = keyframes_admitted
        self.keyframes_rejected = keyframes_rejected
        self.last_admitted_source_idx = last_admitted
        self.bootstrap_count = bootstrap_count
        self.n_train_frames_ingested = frames_ingested
        self.cameras_extent = cameras_extent
        self.holdout_indices = holdout
        self.guard_indices = guard
        self.camera_centers_np = []
        self.camera_centers_gpu = []
        self.camera_viewdirs_np = []
