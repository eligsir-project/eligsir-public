"""Issue #932 -- `KeyframeDecision.forced` under-reports low-overlap
starvation relief.

Root cause: `keyframes.py`'s admit() computes a local `low_overlap_forced`
flag (set True by the low-overlap starvation-relief block) but never folds
it into either reachable `KeyframeDecision(forced=...)` construction --
the quality-scoring early-return (~line 1516) and the final decision
(~line 1558). Consequence: a real `low_overlap_force_admit` admission
reported `forced=False`, so `train_streaming.py`'s
`forced_soft=(decision.forced and decision.reason != "max_gap")` never
counted it and `report.json["force_soft_count"]` under-reported.

`.forced` is telemetry-only -- no code branches on it -- so these tests
only assert on the returned `KeyframeDecision`, not on any training
behavior change (there is none).

CPU-only. Reuses the `_cfg` / `_admitter` / `_admit` harness from
tests/test_issue794_admission_precedence.py (the low-overlap starvation
setup, in particular, is copied verbatim from that file's
`test_low_overlap_starvation_bypasses_only_the_overlap_gate`).
"""
from __future__ import annotations

import numpy as np

from test_issue794_admission_precedence import _admitter, _admit, _cfg, _FakeCam, _frame
from eligsir.mapping.scheduler.frame_quality import FrameQuality


# ---------------------------------------------------------------------------
# Final-decision exit (~keyframes.py:1558)
# ---------------------------------------------------------------------------


def test_low_overlap_force_admit_reports_forced_true():
    """The primary #932 regression case: the final-decision construction
    site must fold `low_overlap_forced` into `forced`."""
    a = _admitter(_cfg(low_overlap_force_admit_iters=1, min_overlap=0.9), overlap=0.1)
    d = _admit(a, index=3, x=1.0)  # moved, ordinary candidate, overlap too low
    assert d.admitted is True
    assert d.reason == "low_overlap_force_admit"
    assert d.forced is True


# ---------------------------------------------------------------------------
# Quality-scoring early-return exit (~keyframes.py:1516)
# ---------------------------------------------------------------------------


def _accepting_quality(*_args, **_kwargs) -> FrameQuality:
    return FrameQuality(
        blur_laplacian=1e6,
        gradient_energy=1e6,
        white_clip_ratio=0.0,
        black_clip_ratio=0.0,
        depth_valid_ratio=1.0,
        depth_hole_ratio=0.0,
        depth_edge_invalid_ratio=0.0,
        quality_weight=1.0,
        reject_reason=None,
    )


def test_low_overlap_force_admit_reports_forced_true_under_quality_scoring():
    """Site-(1) regression guard: with quality scoring enabled, admission
    exits through the quality-scoring early-return instead of the final
    decision construction -- that exit must ALSO fold in
    `low_overlap_forced`.

    `final_threshold=0.0` makes the composite score admit trivially once
    quality is accepted, isolating this test to the `low_overlap_forced`
    attribution itself rather than the scoring math.
    """
    a = _admitter(
        _cfg(
            quality_enabled=True,
            low_overlap_force_admit_iters=1,
            min_overlap=0.9,
            final_threshold=0.0,
        ),
        overlap=0.1,
        evaluate_quality=_accepting_quality,
        load_rgb=lambda f: np.zeros((8, 8, 3), dtype=np.float32),
        load_depth_m=lambda f: np.ones((8, 8), dtype=np.float32),
    )
    d = _admit(a, index=3, x=1.0)
    assert d.admitted is True
    assert d.forced is True


# ---------------------------------------------------------------------------
# Regression guards: unrelated admission paths keep their existing
# `forced` semantics.
# ---------------------------------------------------------------------------


def test_normal_admission_still_reports_forced_false():
    a = _admitter(_cfg(), overlap=0.5)
    d = _admit(a, index=3, x=1.0)  # moved, gap 3 < max_gap
    assert d.admitted is True
    assert d.reason == "pose_delta"
    assert d.forced is False


def test_hard_max_gap_still_reports_forced_true_reason_max_gap():
    a = _admitter(_cfg(), overlap=0.5)
    d = _admit(a, index=40, x=0.0)  # gap 40 == max_gap * hard_mult
    assert d.admitted is True
    assert d.reason == "max_gap"
    assert d.forced is True
