"""Tests for real, admission-independent test holdouts + degenerate-split guard.

Covers the eval-integrity fixes:
- holdouts are selected deterministically from the source trajectory (so they are
  identical across frame-admission modes),
- near-duplicate (near-static) candidates are demoted to train,
- a degenerate split (test_cams is train_cams) is never reported as "test".
"""

from __future__ import annotations

import types

import numpy as np

from scene.streaming_scene import StreamingScene
from eligsir.utils.comparison_report import is_degenerate_split


def _c2w(tx=0.0, ty=0.0, tz=0.0):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (tx, ty, tz)
    return m


class _Frame:
    def __init__(self, index, c2w):
        self.index = index
        self.c2w = c2w


def _make_scene(frames, **arg_overrides):
    """Build a StreamingScene shell and run only the holdout designation."""
    args = types.SimpleNamespace(
        streaming_eval_hold=4,
        streaming_initial_frames=0,
        streaming_eval_guard=0,
        streaming_eval_min_baseline_m=0.03,
        streaming_eval_min_angle_deg=2.0,
    )
    for k, v in arg_overrides.items():
        setattr(args, k, v)
    scene = object.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    scene._keyframe_store = KeyframeStore()
    from eligsir.mapping.stores.frame_archive import FrameArchive
    scene._frame_archive = FrameArchive()
    scene.args = args
    scene._lazy_frames = False
    scene._all_frames = frames
    scene._holdout_frames = frames
    scene._holdout_indices = None
    scene._guard_indices = set()
    scene._designate_test_holdouts()
    return scene


def test_holdouts_are_every_nth_when_camera_moves():
    # Camera translates 0.10 m per frame -> every gap exceeds the 0.03 m threshold.
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    scene = _make_scene(frames, streaming_eval_hold=4)
    # index 0 excluded (bootstrap default 0 keeps it, but index%4==0 -> 0,4,8,12,16)
    assert scene._holdout_indices == {0, 4, 8, 12, 16}


def test_near_static_sequence_falls_back_to_every_nth():
    # All frames at essentially the same pose -> every candidate is a near-duplicate
    # of its neighbour and would be demoted, leaving an empty test set. The filter
    # then falls back to the raw every-Nth split so a run always has n_test > 0.
    frames = [_Frame(i, _c2w(tx=1e-4 * i)) for i in range(20)]
    scene = _make_scene(frames, streaming_eval_hold=4)
    assert scene._holdout_indices == {0, 4, 8, 12, 16}  # fallback, never empty


def test_mixed_motion_demotes_only_static_candidates():
    # Camera moves 0.10 m/frame for the first 12 frames, then is static. With a
    # guard band of 1, the nearest *trained* neighbour of each candidate sits 2
    # frames away. Candidates in the moving stretch (4, 8) keep a real parallax
    # gap; candidates in the static stretch (12, 16, 20) are demoted to train.
    poses = []
    x = 0.0
    for i in range(24):
        poses.append(_c2w(tx=x))
        x += 0.10 if i < 12 else 0.0
    frames = [_Frame(i, p) for i, p in enumerate(poses)]
    scene = _make_scene(frames, streaming_eval_hold=4, streaming_eval_guard=1,
                        streaming_initial_frames=1)
    assert scene._holdout_indices == {4, 8}  # static candidates demoted (not fallback)


def test_designation_is_admission_independent():
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    a = _make_scene(frames, streaming_frame_admission="all")
    b = _make_scene(frames, streaming_frame_admission="hybrid_keyframe")
    assert a._holdout_indices == b._holdout_indices  # identical test set


def test_bootstrap_range_excluded_from_holdouts():
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    scene = _make_scene(frames, streaming_eval_hold=4, streaming_initial_frames=5)
    # indices < 5 (bootstrap) must not be holdouts; 0 and 4 excluded.
    assert 0 not in scene._holdout_indices and 4 not in scene._holdout_indices
    assert scene._holdout_indices == {8, 12, 16}


def test_guard_band_withholds_neighbours():
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    scene = _make_scene(frames, streaming_eval_hold=8, streaming_eval_guard=1)
    # holdouts at 0,8,16; guard withholds index+-1 (not the holdouts themselves).
    assert scene._holdout_indices == {0, 8, 16}
    assert scene._guard_indices.isdisjoint(scene._holdout_indices)
    assert {7, 9, 15, 17}.issubset(scene._guard_indices)


def test_lazy_source_leaves_online_fallback():
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    scene = object.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    scene._keyframe_store = KeyframeStore()
    from eligsir.mapping.stores.frame_archive import FrameArchive
    scene._frame_archive = FrameArchive()
    scene.args = types.SimpleNamespace(streaming_eval_hold=4, streaming_initial_frames=0)
    scene._lazy_frames = True
    scene._all_frames = None
    scene._holdout_frames = None
    scene._holdout_indices = None
    scene._guard_indices = set()
    scene._designate_test_holdouts()
    assert scene._holdout_indices is None  # online every-Nth fallback


def test_eval_metadata_hashes_precomputed_holdouts():
    frames = [_Frame(i, _c2w(tx=0.10 * i)) for i in range(20)]
    scene = _make_scene(frames, streaming_eval_hold=4)
    meta = scene.eval_metadata()
    assert meta["holdout_selection"] == "precomputed"
    assert meta["holdout_hash"] is not None
    assert meta["n_holdout"] == len(scene._holdout_indices)


def _make_lazy_scene(args, test_cameras):
    scene = object.__new__(StreamingScene)
    from eligsir.mapping.stores.keyframe_store import KeyframeStore
    scene._keyframe_store = KeyframeStore()
    from eligsir.mapping.stores.frame_archive import FrameArchive
    scene._frame_archive = FrameArchive()
    scene.args = args
    scene._lazy_frames = True
    scene._holdout_indices = None
    scene._guard_indices = set()
    scene._test_cameras = test_cameras
    return scene


class _Cam:
    def __init__(self, frame):
        self._streaming_frame = frame


def test_eval_metadata_online_every_nth_is_a_rule_hash_not_a_realized_set_hash():
    """issue #745 (2026-08-15): on a live/lazy source, `_holdout_indices`
    stays None, so `eval_metadata()` cannot hash a precomputed index set --
    but the every-Nth *rule* (eval_hold + the ordinal definition + guard/
    pose-gap params + source identity) is fully deterministic and known
    up front, independent of which frames a given run actually reached.
    This is the fix's core guarantee: two runs sharing the rule get the
    SAME `holdout_hash` even when they realized entirely different test
    camera sets (the old, defective behavior hashed the realized set and
    would have produced two DIFFERENT hashes here -- see
    test_eval_metadata_hashes_online_every_nth_from_test_cameras in the
    pre-#745 history of this file for that assertion)."""
    args = types.SimpleNamespace(streaming_eval_hold=4, streaming_initial_frames=0,
                                  streaming_eval_guard=0, source_path="/data/scene_a")
    scene_5 = _make_lazy_scene(args, [_Cam(_Frame(i, None)) for i in (0, 4, 8, 12, 16)])
    meta_5 = scene_5.eval_metadata()
    assert meta_5["holdout_selection"] == "online_every_nth_ordinal_rule"
    assert meta_5["n_holdout"] == 5
    assert meta_5["holdout_hash"] is not None

    # Same rule (same args), a DIFFERENT realized count/order/composition of
    # test cameras -> identical holdout_hash. This is the defect: pre-fix,
    # these two runs (differing only in how many holdout frames they
    # reached) got different hashes and were wrongly flagged incomparable.
    scene_3 = _make_lazy_scene(args, [_Cam(_Frame(i, None)) for i in (0, 4, 8)])
    meta_3 = scene_3.eval_metadata()
    assert meta_3["holdout_hash"] == meta_5["holdout_hash"]
    assert meta_3["n_holdout"] == 3  # n_holdout still reflects the realized count

    # A genuinely different RULE (different eval_hold) must hash differently.
    args_hold8 = types.SimpleNamespace(streaming_eval_hold=8, streaming_initial_frames=0,
                                        streaming_eval_guard=0, source_path="/data/scene_a")
    scene_hold8 = _make_lazy_scene(args_hold8, [_Cam(_Frame(i, None)) for i in (0, 4, 8, 12, 16)])
    assert scene_hold8.eval_metadata()["holdout_hash"] != meta_5["holdout_hash"]

    # A different SOURCE identity must also hash differently, even with an
    # identical rule otherwise (two different scenes must never collide).
    args_other_source = types.SimpleNamespace(streaming_eval_hold=4, streaming_initial_frames=0,
                                                streaming_eval_guard=0, source_path="/data/scene_b")
    scene_other_source = _make_lazy_scene(args_other_source,
                                           [_Cam(_Frame(i, None)) for i in (0, 4, 8, 12, 16)])
    assert scene_other_source.eval_metadata()["holdout_hash"] != meta_5["holdout_hash"]


def test_eval_metadata_rule_hash_distinguishes_ordinal_definition_version():
    """issue #745 round 2 (2026-08-15): the rule hash folds in
    `self._frame_source.ordinal_definition_version` (falling back to
    "ts_quantized_v1" when `_frame_source` doesn't expose it at all).
    ROS2LiveStreamingFrameSource can use either "received_seq_v1" (the
    primary, real sequence-counter mechanism) or "ts_quantized_v1" (the
    timestamp-quantization fallback) -- two runs of the identical rule
    (same eval_hold/guard/pose-gap/source) but a DIFFERENT ordinal
    mechanism must never collide into the same holdout_hash, since the
    membership those two mechanisms actually produce is not guaranteed to
    agree frame-for-frame."""
    args = types.SimpleNamespace(streaming_eval_hold=4, streaming_initial_frames=0,
                                  streaming_eval_guard=0, source_path="/data/scene_a")

    scene_seq = _make_lazy_scene(args, [])
    scene_seq._frame_source = types.SimpleNamespace(ordinal_definition_version="received_seq_v1")
    meta_seq = scene_seq.eval_metadata()

    scene_ts = _make_lazy_scene(args, [])
    scene_ts._frame_source = types.SimpleNamespace(ordinal_definition_version="ts_quantized_v1")
    meta_ts = scene_ts.eval_metadata()

    scene_no_source = _make_lazy_scene(args, [])  # no _frame_source attr at all
    meta_no_source = scene_no_source.eval_metadata()

    assert meta_seq["holdout_hash"] != meta_ts["holdout_hash"]
    # Absence of `_frame_source` falls back to the same "ts_quantized_v1"
    # default the explicit-ts_quantized_v1 run used -> identical hash.
    assert meta_no_source["holdout_hash"] == meta_ts["holdout_hash"]


def test_eval_metadata_online_rule_hash_available_before_any_frame_arrives():
    """The whole point of a RULE hash: it must be computable before a single
    frame has been ingested (a mid-run checkpoint report, or the very start
    of a run) -- unlike the old realized-set hash, which was null until at
    least one holdout frame had arrived."""
    args = types.SimpleNamespace(streaming_eval_hold=4, streaming_initial_frames=0,
                                  streaming_eval_guard=0, source_path="/data/scene_a")
    scene = _make_lazy_scene(args, [])
    meta = scene.eval_metadata()
    assert meta["holdout_selection"] == "online_every_nth_ordinal_rule"
    assert meta["holdout_hash"] is not None
    assert meta["n_holdout"] == 0


def test_eval_metadata_online_holdout_disabled():
    args = types.SimpleNamespace(streaming_eval_hold=0, streaming_initial_frames=0,
                                  streaming_eval_guard=0, source_path="/data/scene_a")
    scene = _make_lazy_scene(args, [])
    meta = scene.eval_metadata()
    assert meta["holdout_selection"] == "online_every_nth"
    assert meta["holdout_hash"] is None
    assert meta["n_holdout"] == 0


def test_degenerate_split_detection():
    cams = [object(), object(), object()]
    assert is_degenerate_split(cams, cams) is True            # same list object
    assert is_degenerate_split(cams, list(cams)) is True      # same elements
    assert is_degenerate_split(cams, cams, degenerate_split=True) is True
    other = [object(), object(), object()]
    assert is_degenerate_split(cams, other) is False          # genuine holdout
    assert is_degenerate_split(cams, []) is False
