"""Tests for train_streaming.bootstrap_frame_passes_confidence_gate (W3-30).

Bug hypothesis: online-SLAM bootstrap (train_streaming.py's initialize_from_frames
frame_filter caller) accepted the first N *valid* TrackingResults unconditionally,
ignoring TrackingResult.confidence entirely -- unlike every other consumer of this
signal (streaming_tracking_confidence_enabled, MappingScheduler's tracking-confidence
gate). SLAM tracking confidence is characteristically lowest right after backend
initialize()/track() startup, before pose covariance has converged -- exactly the
frames bootstrap seeds cameras_extent (scene-scale prior, monotonic never-shrinks,
computed once with no outlier rejection) and the initial point cloud from. Fixed by
gating bootstrap frame acceptance on a configurable minimum confidence
(--streaming_bootstrap_min_confidence, default 0.0 = off, preserving prior behavior).

See docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-confidence-gate.md.

Imports train_streaming, which transitively pulls in torch -- run inside the
training container
(``docker compose run --rm train python -m pytest tests/test_bootstrap_confidence_gate.py``),
not on host Python.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from train_streaming import bootstrap_frame_passes_confidence_gate


def _tracking_result(valid: bool, confidence: float):
    return SimpleNamespace(valid=valid, confidence=confidence)


def test_gate_disabled_by_default_accepts_any_valid_result():
    """min_confidence=0.0 (the default) reproduces pre-W3-30 behavior: any
    valid result is accepted regardless of how low its confidence is."""
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(True, 0.0), 0.0) is True
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(True, 0.01), 0.0) is True


def test_gate_rejects_invalid_result_regardless_of_threshold():
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(False, 1.0), 0.0) is False
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(False, 1.0), 0.5) is False


def test_gate_rejects_none_result():
    assert bootstrap_frame_passes_confidence_gate(None, 0.0) is False
    assert bootstrap_frame_passes_confidence_gate(None, 0.5) is False


def test_gate_rejects_valid_low_confidence_result_when_threshold_set():
    result = _tracking_result(True, 0.2)
    assert bootstrap_frame_passes_confidence_gate(result, 0.4) is False


def test_gate_accepts_valid_result_at_or_above_threshold():
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(True, 0.4), 0.4) is True
    assert bootstrap_frame_passes_confidence_gate(_tracking_result(True, 0.9), 0.4) is True
