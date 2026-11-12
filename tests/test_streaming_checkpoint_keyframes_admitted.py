"""Regression test: keyframes_admitted must round-trip through a streaming
checkpoint save/resume cycle.

Bug: StreamingScene._keyframes_admitted (the monotonic cumulative
admission counter that both the live adm% progress-bar figure and
report.json's trajectory_coverage.frames_admitted are derived from) was
never included in the checkpoint's streaming_meta payload, so it silently
reset to 0 on any resumed run. Observed on a real resumed
scannetpp_partitionfocus_v2_focus run: adm% climbed normally to ~15%
pre-resume, then dropped to 0% for the rest of training (5105/5207
ticks), producing a reported frames_admitted of 22 despite a real
cumulative admission rate around 14-15%.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from train_streaming import _save_streaming_checkpoint
from eligsir.utils.coverage_ledger import CoverageLedger


def test_streaming_meta_includes_keyframes_admitted():
    save_worker = MagicMock()
    gaussians = MagicMock()
    gaussians.capture.return_value = "fake_state"

    captured = {}

    def _fake_enqueue(fn):
        # Run the closure immediately but intercept torch.save's args by
        # monkeypatching torch.save for the duration of this call.
        import torch as _torch
        orig_save = _torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return orig_save(payload, path)
            _torch.save = _capture_and_save
            fn()
        finally:
            _torch.save = orig_save

    save_worker.enqueue.side_effect = _fake_enqueue

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=2000,
            n_frames_ingested=4004, last_ingest_iter=1998, source_idx=4004,
            keyframes_admitted=985,
        )

    assert "payload" in captured
    _state, _iteration, streaming_meta = captured["payload"]
    assert streaming_meta["keyframes_admitted"] == 985


def test_streaming_meta_defaults_keyframes_admitted_to_zero_when_omitted():
    save_worker = MagicMock()
    gaussians = MagicMock()
    gaussians.capture.return_value = "fake_state"

    captured = {}

    def _fake_enqueue(fn):
        import torch as _torch
        orig_save = _torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return orig_save(payload, path)
            _torch.save = _capture_and_save
            fn()
        finally:
            _torch.save = orig_save

    save_worker.enqueue.side_effect = _fake_enqueue

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=2000,
            n_frames_ingested=4004, last_ingest_iter=1998, source_idx=4004,
        )

    _state, _iteration, streaming_meta = captured["payload"]
    assert streaming_meta["keyframes_admitted"] == 0


def test_streaming_meta_preserves_per_source_supervision_counts_for_resume():
    """A resumed report must keep pre-checkpoint successful exposures and zeros."""
    save_worker = MagicMock()
    gaussians = MagicMock()
    gaussians.capture.return_value = "fake_state"
    captured = {}

    def _fake_enqueue(fn):
        import torch as _torch
        orig_save = _torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return orig_save(payload, path)
            _torch.save = _capture_and_save
            fn()
        finally:
            _torch.save = orig_save

    save_worker.enqueue.side_effect = _fake_enqueue
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=2000,
            n_frames_ingested=4004, last_ingest_iter=1998, source_idx=4004,
            per_source_supervision_counts={7: 3, 8: 0},
        )

    _state, _iteration, streaming_meta = captured["payload"]
    assert streaming_meta["per_source_supervision_counts"] == {7: 3, 8: 0}


def test_streaming_meta_round_trips_the_versioned_coverage_ledger():
    save_worker = MagicMock()
    gaussians = MagicMock()
    gaussians.capture.return_value = "fake_state"
    captured = {}

    def _fake_enqueue(fn):
        import torch as _torch
        original = _torch.save
        try:
            def _capture_and_save(payload, path):
                captured.setdefault("payload", payload)
                return original(payload, path)
            _torch.save = _capture_and_save
            fn()
        finally:
            _torch.save = original

    save_worker.enqueue.side_effect = _fake_enqueue
    ledger = CoverageLedger.fresh(
        resolved_source_frames=10,
        raw_registry_source_frames=10,
        source_denominator_provenance="post_stride_frame_source",
        bootstrap_frames_consumed=2,
        source_identity="scene",
        effective_stride=1,
    )
    ledger.record_materialized(
        is_holdout=False, is_guard=False, odom_valid=True, admitted=True
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_streaming_checkpoint(
            save_worker, gaussians, tmpdir, iteration=3,
            n_frames_ingested=3, last_ingest_iter=3, source_idx=3,
            coverage_ledger=ledger.snapshot(),
        )

    _state, _iteration, metadata = captured["payload"]
    restored = CoverageLedger.restore(metadata["coverage_ledger"])
    assert restored is not None
    assert restored.snapshot() == ledger.snapshot()
    assert CoverageLedger.restore({}) is None
