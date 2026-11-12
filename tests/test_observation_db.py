"""Tests for FrameObservationDB (keyframe/Gaussian observation database, task 2)."""
from __future__ import annotations

import numpy as np
import torch

from eligsir.mapping.observation_db import BufferStage, FrameObservationDB, FrameRecord


def _record(frame_id: int, uids: list[int], submap_id: int = 0, resolution_cache=None) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        pose_c2w=np.eye(4, dtype=np.float32),
        timestamp=float(frame_id),
        submap_id=submap_id,
        buffer_stage=BufferStage.MAPPING_KEYFRAME,
        visible_gaussian_uids=torch.tensor(uids, dtype=torch.int64),
        resolution_cache=resolution_cache if resolution_cache is not None else {},
    )


def test_add_and_get_frame():
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(0, [1, 2, 3]))
    assert len(db) == 1
    got = db.get_frame(0)
    assert got is not None
    assert got.frame_id == 0
    assert db.get_frame(999) is None


def test_overlap_query_finds_shared_uids():
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(0, [1, 2, 3, 4]))
    db.add_frame(_record(1, [10, 11]))  # disjoint
    db.add_frame(_record(2, [3, 4, 5, 6]))  # overlaps frame 0 on {3, 4}

    candidate = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int64)
    frame_ids, counts = db.overlap_query(candidate)

    # frame 2 shares {3,4,5,6} = 4; frame 0 shares {3,4} = 2; frame 1 shares 0.
    assert list(frame_ids[:2]) == [2, 0]
    assert counts[0] == 4
    assert counts[1] == 2


def test_overlap_query_empty_db_returns_empty():
    db = FrameObservationDB(device="cpu")
    frame_ids, counts = db.overlap_query(torch.tensor([1, 2], dtype=torch.int64))
    assert len(frame_ids) == 0
    assert len(counts) == 0


def test_buffer_stage_and_resolution_cache_are_plain_schema_fields():
    rec = _record(0, [1])
    assert rec.buffer_stage == BufferStage.MAPPING_KEYFRAME
    rec.buffer_stage = BufferStage.HOLDOUT_TEST
    assert rec.buffer_stage == BufferStage.HOLDOUT_TEST
    rec.resolution_cache["fast"] = "some_handle"
    assert rec.resolution_cache == {"fast": "some_handle"}


def _fake_cache_tensor(pixels: int) -> torch.Tensor:
    """Stand-in for a build_resolution_cache CacheHandle's rgb/depth tensor --
    sized so total retained bytes are directly measurable, mirroring how
    'high' tier is native-resolution (divisor=1) regardless of active tier."""
    return torch.zeros(pixels, dtype=torch.uint8)


def _cache_bytes(db: FrameObservationDB) -> int:
    total = 0
    for rec in db._frames.values():
        for handle in rec.resolution_cache.values():
            total += int(handle.numel())
    return total


def test_resolution_cache_unbounded_by_default_reproduces_leak():
    """0 (legacy/back-compat default) means no LRU bound -- retained bytes
    grow linearly with admitted-frame count forever, independent of any
    downstream Gaussian count. This is the exact shape of the real 2026-08-04
    ScanNet++ bug (VRAM disconnected from N) at a synthetic scale."""
    db = FrameObservationDB(device="cpu", max_resolution_cache_frames=0)
    pixels_per_frame = 1000
    for i in range(50):
        db.add_frame(_record(i, [i], resolution_cache={"high": _fake_cache_tensor(pixels_per_frame)}))
    assert _cache_bytes(db) == 50 * pixels_per_frame


def test_resolution_cache_lru_bound_caps_retained_bytes():
    """With the fix's LRU bound, retained resolution_cache bytes plateau once
    the admitted-frame count exceeds the bound, instead of growing without
    limit -- the FrameRecord (pose/visibility) is still retained for every
    frame, only the heavy cached image tensors get evicted."""
    max_frames = 10
    db = FrameObservationDB(device="cpu", max_resolution_cache_frames=max_frames)
    pixels_per_frame = 1000
    n_admitted = 50
    for i in range(n_admitted):
        db.add_frame(_record(i, [i], resolution_cache={"high": _fake_cache_tensor(pixels_per_frame)}))
        assert len(db) == i + 1  # FrameRecord itself is never evicted

    assert _cache_bytes(db) == max_frames * pixels_per_frame
    # Oldest frames' cache tensors are gone; record metadata is not.
    for i in range(n_admitted - max_frames):
        assert db.get_frame(i).resolution_cache == {}
    for i in range(n_admitted - max_frames, n_admitted):
        assert db.get_frame(i).resolution_cache != {}


def test_resolution_cache_eviction_leaves_visibility_history_intact():
    """Evicting a frame's cached image tensors must not affect overlap_query
    (pose/visibility history) -- only bind_resolution_tier's image lookup is
    affected, and it already handles a missing/empty cache gracefully."""
    db = FrameObservationDB(device="cpu", max_resolution_cache_frames=1)
    db.add_frame(_record(0, [1, 2, 3], resolution_cache={"high": _fake_cache_tensor(10)}))
    db.add_frame(_record(1, [3, 4], resolution_cache={"high": _fake_cache_tensor(10)}))

    assert db.get_frame(0).resolution_cache == {}  # evicted
    frame_ids, counts = db.overlap_query(torch.tensor([3, 4], dtype=torch.int64))
    assert set(frame_ids.tolist()) == {0, 1}


def test_set_buffer_stage_updates_known_frame():
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(0, [1, 2]))
    assert db.set_buffer_stage(0, BufferStage.COLD_CACHE) is True
    assert db.get_frame(0).buffer_stage == BufferStage.COLD_CACHE


def test_set_buffer_stage_unknown_frame_is_safe_noop():
    db = FrameObservationDB(device="cpu")
    assert db.set_buffer_stage(999, BufferStage.ICECOLD_CACHE) is False


def test_set_buffer_stage_does_not_touch_observation_history():
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(0, [1, 2, 3]))
    db.set_buffer_stage(0, BufferStage.ICECOLD_CACHE)
    # visible_gaussian_uids / overlap query results must be unaffected --
    # only the stage label changes.
    frame_ids, counts = db.overlap_query(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert list(frame_ids) == [0]
    assert counts[0] == 3


def test_add_frame_does_not_accumulate_permanent_python_list_duplicate():
    """2026-08-04 follow-up fix: add_frame() used to permanently duplicate
    every (frame_id, uid) pair into self._rows/self._cols Python int lists,
    purely to feed the COO/CSR rebuild -- growing without bound, independent
    of Gaussian count N, purely as a function of cumulative ingested-frame
    count x visible-uids-per-frame (measured ~12.7GB RSS at realistic
    ScanNet++ scale: 7040 ingested candidates, N~300k, 15% visible
    fraction). That data was already redundant with each FrameRecord's own
    visible_gaussian_uids tensor. This test locks in that the duplicate
    lists are gone entirely -- the object must expose no _rows/_cols
    attribute at all -- while overlap_query correctness is unaffected."""
    db = FrameObservationDB(device="cpu")
    assert not hasattr(db, "_rows")
    assert not hasattr(db, "_cols")
    for i in range(200):
        db.add_frame(_record(i, [i, i + 1, i + 2]))
    frame_ids, counts = db.overlap_query(torch.tensor([50, 51], dtype=torch.int64))
    assert 50 in frame_ids.tolist()


def test_full_history_preserved_across_many_frames_no_eviction():
    """overlap_query must still see frame 0's observations after 5000 later
    frames are added -- the whole point of the DB is full-history revisit
    matching (see module docstring); removing the _rows/_cols duplicate must
    not silently truncate history to a recent window."""
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(0, [999999]))
    for i in range(1, 5001):
        db.add_frame(_record(i, [i]))
    frame_ids, counts = db.overlap_query(torch.tensor([999999], dtype=torch.int64))
    assert list(frame_ids) == [0]
    assert counts[0] == 1


def test_rows_cols_removal_does_not_grow_python_list_memory_with_ingestion():
    """Direct measurement (not just 'unbounded in principle'): container
    byte-size of any list-like accumulator on the DB must NOT scale with
    cumulative ingested-frame count the way the old self._rows/self._cols
    did (they grew to hundreds of MB of raw list-container bytes alone at
    only a few thousand frames, before counting the Python int objects
    themselves). The only per-frame Python-level growth now is one dict
    entry in self._frames (required, holds the real FrameRecord) and one
    OrderedDict entry in self._resolution_cache_order -- both O(1) per
    frame, not O(visible uids per frame)."""
    import sys

    db = FrameObservationDB(device="cpu")
    n_frames = 3000
    uids_per_frame = 200
    for i in range(n_frames):
        uids = list(range(i * uids_per_frame, i * uids_per_frame + uids_per_frame))
        db.add_frame(_record(i, uids))

    # No attribute on the instance may hold n_frames * uids_per_frame
    # separate Python objects -- everything list-shaped on the DB is O(n_frames).
    # _buf_col is excluded: it is a single preallocated torch.Tensor (a
    # growable buffer, not a Python list/dict of per-observation objects --
    # see the PERF FIX note in observation_db.py's __init__), so `vars()`
    # never surfaces it as a `list`/`dict` in the first place; nothing else
    # on the DB scales with n_frames * uids_per_frame.
    for name, value in vars(db).items():
        if isinstance(value, (list, dict)):
            assert len(value) <= n_frames + 1, (
                f"{name} has {len(value)} entries -- expected O(n_frames)={n_frames}, "
                f"not O(n_frames * uids_per_frame)={n_frames * uids_per_frame}"
            )


def _full_rebuild_reference(records: "list[FrameRecord]") -> FrameObservationDB:
    """Baseline oracle: force every record through the O(total observations)
    full-rebuild path (bypassing the incremental buffer entirely) by
    disabling `_incremental_valid` up front, so correctness-parity tests
    below have a ground truth that shares no code with the new fast path."""
    db = FrameObservationDB(device="cpu")
    db._incremental_valid = False
    for record in records:
        db.add_frame(record)
    return db


def test_incremental_csr_matches_full_rebuild_baseline():
    """2026-08-04 perf follow-up: _ensure_csr()'s incremental append path
    (see PERF FIX note in observation_db.py's __init__) must produce
    IDENTICAL overlap_query results to the old O(total observations)
    full-rebuild-every-dirty-call path it replaces -- this is the critical
    correctness check per the task contract: a faster implementation that
    silently changes results here is worse than the slow one it replaces."""
    import random

    rng = random.Random(0)
    records = []
    for fid in range(500):
        n_uid = rng.randint(0, 12)
        uids = sorted(rng.sample(range(2000), n_uid)) if n_uid else []
        records.append(_record(fid, uids))

    db_fast = FrameObservationDB(device="cpu")
    for r in records:
        db_fast.add_frame(r)
    assert db_fast._incremental_valid  # sanity: this test exercises the fast path

    db_ref = _full_rebuild_reference(records)

    for _ in range(50):
        n_cand = rng.randint(1, 15)
        cand = torch.tensor(sorted(rng.sample(range(2000), n_cand)), dtype=torch.int64)
        fast_ids, fast_counts = db_fast.overlap_query(cand)
        ref_ids, ref_counts = db_ref.overlap_query(cand)
        assert fast_ids.tolist() == ref_ids.tolist()
        assert fast_counts.tolist() == ref_counts.tolist()


def test_incremental_csr_handles_gap_rows_like_full_rebuild():
    """Some frame_ids are never add_frame()'d (pregate-short-circuited
    candidates, see scene/streaming_scene.py's ingest_next_frame) -- the
    incremental buffer's crow-index gap-filling must produce the same
    empty-row semantics as the full-rebuild path, not misalign rows."""
    records = [_record(0, [1, 2]), _record(5, [2, 3]), _record(6, [1])]
    db_fast = FrameObservationDB(device="cpu")
    for r in records:
        db_fast.add_frame(r)
    db_ref = _full_rebuild_reference(records)

    cand = torch.tensor([1, 2, 3], dtype=torch.int64)
    fast_ids, fast_counts = db_fast.overlap_query(cand)
    ref_ids, ref_counts = db_ref.overlap_query(cand)
    assert sorted(zip(fast_ids.tolist(), fast_counts.tolist())) == sorted(
        zip(ref_ids.tolist(), ref_counts.tolist())
    )
    assert sorted(fast_ids.tolist()) == [0, 5, 6]


def test_out_of_order_frame_id_falls_back_to_full_rebuild_safely():
    """If a caller ever violates the increasing-frame_id assumption the
    incremental path relies on (see PERF FIX note), the DB must fall back
    to the old full-rebuild path rather than silently return wrong
    overlap counts."""
    db = FrameObservationDB(device="cpu")
    db.add_frame(_record(5, [1, 2]))
    assert db._incremental_valid
    db.add_frame(_record(3, [2, 3]))  # frame_id goes backwards
    assert not db._incremental_valid

    frame_ids, counts = db.overlap_query(torch.tensor([2], dtype=torch.int64))
    assert set(frame_ids.tolist()) == {5, 3}
    for fid, cnt in zip(frame_ids.tolist(), counts.tolist()):
        assert cnt == 1


def test_incremental_buffer_grows_without_dropping_data_across_many_appends():
    """Exercises the capacity-doubling buffer growth in _buf_append across
    enough add_frame() calls to force several reallocations, confirming no
    data is lost/corrupted at a growth boundary."""
    db = FrameObservationDB(device="cpu")
    n_frames = 2500
    for i in range(n_frames):
        db.add_frame(_record(i, [i]))
    frame_ids, counts = db.overlap_query(torch.tensor([0, 2499], dtype=torch.int64))
    assert set(frame_ids.tolist()) == {0, 2499}
    assert all(c == 1 for c in counts.tolist())
