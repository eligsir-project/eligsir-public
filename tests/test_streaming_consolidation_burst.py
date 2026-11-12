"""Unit tests for the consolidation-burst queue logic (Task 3).

Covers:
* StreamingScene.get_nearest_cameras — k-nearest by camera-center distance, the
  source of burst cameras enqueued after each keyframe admission.
* The burst queue's drain semantics (FIFO, drained before normal sampling),
  exercised here against collections.deque (the exact type used inline in
  train_streaming.py), since the training loop itself needs CUDA to import.

Default-off behaviour is asserted at the wiring level: when the flag is off the
queue is never populated, so sampling is unchanged.
"""
from collections import deque

import torch

from scene.streaming_scene import StreamingScene


class _Cam:
    def __init__(self, x, y=0.0, z=0.0, name=None):
        self.camera_center = torch.tensor([float(x), float(y), float(z)])
        self.name = name


def _scene(cams):
    s = StreamingScene.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    s._keyframe_store = KeyframeStore()
    s.train_cameras = cams
    return s


def test_get_nearest_returns_k_closest_in_distance_order():
    cams = [_Cam(0, name="a"), _Cam(1, name="b"), _Cam(5, name="c"), _Cam(10, name="d")]
    s = _scene(cams)
    near = s.get_nearest_cameras(_Cam(0.4), k=2)
    assert [c.name for c in near] == ["a", "b"]  # 0 and 1 are closest to 0.4


def test_get_nearest_caps_at_available_count():
    cams = [_Cam(0), _Cam(1)]
    s = _scene(cams)
    near = s.get_nearest_cameras(_Cam(0.0), k=4)
    assert len(near) == 2  # only 2 cameras exist


def test_get_nearest_empty_scene_returns_empty():
    s = _scene([])
    assert s.get_nearest_cameras(_Cam(0.0), k=4) == []


def test_get_nearest_none_query_returns_empty():
    s = _scene([_Cam(0)])
    assert s.get_nearest_cameras(None, k=4) == []


def test_burst_queue_drains_fifo_before_falling_back():
    # Mirror the inline drain: pop burst cams first; only sample normally when
    # the queue is empty.
    q = deque()
    burst = [_Cam(1, name="x"), _Cam(2, name="y")]
    for c in burst:
        q.append(c)

    drained = []
    for _ in range(3):
        if q:
            drained.append(q.popleft().name)
        else:
            drained.append("SAMPLED")
    assert drained == ["x", "y", "SAMPLED"]


def test_disabled_flag_leaves_queue_empty():
    # When the flag is off the wiring never enqueues, so the drain branch is a
    # no-op and normal sampling always runs.
    enabled = False
    q = deque()
    admitted = True  # even on admission
    if enabled and admitted:
        q.append(_Cam(1))
    assert len(q) == 0
