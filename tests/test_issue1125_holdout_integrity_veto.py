"""Regression tests for issue #1125 -- an integrity veto must never shrink
the realized holdout set.

``StreamingScene.ingest_next_frame``'s integrity-veto branch (#806) used to
fire universally, returning ``(None, frame, False)`` *before*
``_frame_to_camera`` was ever called. For an ordinary training candidate
that is the whole point of the veto; for a **holdout** frame it silently
violated the reservation contract stated by ``holdout_status()``'s own
docstring and by CLAUDE.md's real-holdout Run Contract -- "a frame in
``_holdout_indices`` ALWAYS becomes a test camera ... regardless of frame
admission" -- so ``n_test`` shrank below what the holdout rule promised and
``realized_holdout_hash`` changed, with nothing in the report attributing
either.

The veto's harm channel is map corruption via TRAINING and depth insertion;
a holdout camera is never trained, replayed, or depth-inserted, so that
mechanism cannot fire through one. What remains -- a possibly-mispositioned
eval camera -- is already flagged run-wide by
``report.json["geometry"]["trajectory_coverage"]["map_reinit_invalidated"]``.

These tests reuse the ``_FakeIngestSelf`` harness from
``tests/test_issue806_integrity_observer_ordering.py`` (imported rather than
re-duplicated, so the two files cannot drift), which exercises the REAL,
unmodified ``ingest_next_frame`` body. Like that module, this one
deliberately does not import ``train_streaming`` (host torch/torchvision
mismatch breaks collection -- see its module docstring).
"""

from __future__ import annotations

import pytest

from scene.streaming_scene import StreamingScene
from test_issue806_integrity_observer_ordering import (
    _REINIT_REJECT_STATS,
    _FakeFrame,
    _FakeIngestSelf,
)


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setattr(
        "scene.cameras.prepare_camera_for_render", lambda cam, device="cuda": None
    )


def _ingest_all(fake_self, integrity_observer_fn=None, **kwargs):
    """Drain the whole fake source, returning the per-frame results."""
    results = []
    while fake_self.has_next_frame():
        results.append(
            StreamingScene.ingest_next_frame(
                fake_self, integrity_observer_fn=integrity_observer_fn, **kwargs
            )
        )
    return results


def test_n_test_matches_holdout_rule_under_a_permanent_integrity_veto():
    """#1125 core invariant: every reserved holdout becomes a real test
    camera, even when a PERMANENT integrity veto (an ORB-SLAM3 map re-init,
    which never clears for the rest of the run) is in effect for most of the
    stream. `n_test` must equal `len(_holdout_indices)`, not some smaller,
    veto-dependent number."""
    n_frames = 12
    holdouts = {2, 5, 8, 11}
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(n_frames)])
    fake_self._holdout_indices = set(holdouts)
    fake_self._guard_indices = set()

    veto_from = 4  # re-init fires here and never clears (permanent veto)

    def integrity_observer_fn(frame):
        return dict(_REINIT_REJECT_STATS) if frame.index >= veto_from else None

    results = _ingest_all(fake_self, integrity_observer_fn=integrity_observer_fn)

    # Every reserved holdout produced a real, non-None camera in the test set.
    assert len(fake_self._test_cameras) == len(holdouts)
    got = {cam._streaming_frame.index for cam in fake_self._test_cameras}
    assert got == holdouts
    assert all(cam is not None for cam in fake_self._test_cameras)

    # ...including the three holdouts that were reserved AFTER the veto began.
    assert {i for i in holdouts if i >= veto_from} <= got

    # Each holdout's own ingest result carries the camera and is_train=False.
    for idx in sorted(holdouts):
        cam, frame, is_train = results[idx]
        assert cam is not None, f"holdout frame {idx} was dropped by the integrity veto"
        assert is_train is False
        assert frame.index == idx

    # A holdout is never trained/replayed regardless of the veto.
    assert all(c._streaming_frame.index not in holdouts for c in fake_self.train_cameras)
    assert all(c._streaming_frame.index not in holdouts for c in fake_self._replay_buffer)


def test_veto_still_rejects_ordinary_training_candidates():
    """The bypass is scoped to holdouts/guards only -- ordinary candidates
    under the same veto must still be rejected before a Camera is built
    (#806's contract is unchanged)."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(6)])
    fake_self._holdout_indices = {3}
    fake_self._guard_indices = set()

    def integrity_observer_fn(frame):
        return dict(_REINIT_REJECT_STATS)  # vetoed from frame 0 onward

    results = _ingest_all(fake_self, integrity_observer_fn=integrity_observer_fn)

    for idx, (cam, frame, is_train) in enumerate(results):
        if idx == 3:
            assert cam is not None and is_train is False
        else:
            assert (cam, frame.index, is_train) == (None, idx, False)
            assert frame._streaming_admission_stats["reason"] == "tracking_reinit"

    assert fake_self.train_cameras == []
    assert fake_self._keyframes_admitted == 0
    # 5 ordinary rejections; the holdout is not a rejected keyframe.
    assert fake_self._keyframes_rejected == 5
    assert len(fake_self._test_cameras) == 1


def test_vetoed_holdout_set_is_identical_to_the_unvetoed_one():
    """The realized holdout set (what `realized_holdout_hash` fingerprints)
    must not depend on whether an integrity event occurred -- that
    admission-independence is the whole point of reserving holdouts up
    front."""
    holdouts = {1, 4, 7}

    def realized(observer):
        fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(9)])
        fake_self._holdout_indices = set(holdouts)
        fake_self._guard_indices = set()
        _ingest_all(fake_self, integrity_observer_fn=observer)
        return [c._streaming_frame.index for c in fake_self._test_cameras]

    clean = realized(lambda frame: None)
    vetoed = realized(lambda frame: dict(_REINIT_REJECT_STATS))

    assert clean == sorted(holdouts)
    assert vetoed == clean


def test_guard_frames_also_survive_the_veto():
    """Guards are bypassed for the same reason as holdouts (never trained,
    never evaluated -- they only need a real `cam` for dense-trajectory
    bookkeeping), and symmetrically with `pregate_fn`, which already never
    runs for them. They must NOT land in the test set."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(4)])
    fake_self._holdout_indices = {2}
    fake_self._guard_indices = {1}

    seen = []

    def _append_dense_trajectory(cam, frame, is_train, is_holdout):
        seen.append((frame.index, cam is not None, is_train, is_holdout))

    fake_self._append_dense_trajectory = _append_dense_trajectory

    _ingest_all(
        fake_self, integrity_observer_fn=lambda frame: dict(_REINIT_REJECT_STATS)
    )

    assert (1, True, False, False) in seen  # guard got a real camera
    assert [c._streaming_frame.index for c in fake_self._test_cameras] == [2]


def test_malformed_holdout_frame_raises_loudly_rather_than_dropping():
    """Fail-closed: a holdout frame that genuinely cannot be built into a
    Camera must surface as a real error, never be silently swallowed back
    into the old dropped-holdout behavior. (`_frame_to_camera` is
    deliberately not wrapped in a try/except -- this test pins that.)"""
    fake_self = _FakeIngestSelf([_FakeFrame(0)])
    fake_self._holdout_indices = {0}
    fake_self._guard_indices = set()

    def _boom(frame, load_image=True):
        raise ValueError("unreadable RGB for frame %d" % frame.index)

    fake_self._frame_to_camera = _boom

    with pytest.raises(ValueError, match="unreadable RGB"):
        StreamingScene.ingest_next_frame(
            fake_self,
            integrity_observer_fn=lambda frame: dict(_REINIT_REJECT_STATS),
        )

    assert fake_self._test_cameras == []


def test_vetoed_holdout_is_still_passed_to_record_fn():
    """Caller-visible consequence of the fix, pinned explicitly: a vetoed
    holdout now reaches a real `cam` and is therefore indexed by `record_fn`
    (train_streaming.py's `_record_fn` -> observation DB + four-clocks
    `record_frame_holdout()`), exactly as it would be with no integrity
    event at all. This is a RESTORATION of consistency, not a new write
    class -- an unvetoed holdout already got this record -- but the
    `ingest_next_frame` docstring's `record_fn` contract now says so, and
    this test keeps the two in agreement."""
    def recorded(observer):
        fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(4)])
        fake_self._holdout_indices = {2}
        fake_self._guard_indices = set()
        seen = []
        _ingest_all(
            fake_self,
            integrity_observer_fn=observer,
            record_fn=lambda cam, frame, is_train, is_holdout: seen.append(
                (frame.index, cam is not None, is_train, is_holdout)
            ),
        )
        return seen

    clean = recorded(lambda frame: None)
    vetoed = recorded(lambda frame: dict(_REINIT_REJECT_STATS))

    assert (2, True, False, True) in clean
    assert (2, True, False, True) in vetoed  # identical for the holdout
    # ...while ordinary vetoed candidates still never reach record_fn.
    assert [r for r in vetoed if r[0] != 2] == []


def test_observer_still_called_for_every_frame_including_vetoed_holdouts():
    """#806's mandatory-observation invariant is untouched by #1125: the
    observer runs on EVERY materialized frame, in source order, whatever
    kind of candidate it is."""
    fake_self = _FakeIngestSelf([_FakeFrame(i) for i in range(6)])
    fake_self._holdout_indices = {0, 3}
    fake_self._guard_indices = {1}

    calls: list[int] = []

    def integrity_observer_fn(frame):
        calls.append(frame.index)
        return dict(_REINIT_REJECT_STATS)

    _ingest_all(fake_self, integrity_observer_fn=integrity_observer_fn)
    assert calls == list(range(6))
