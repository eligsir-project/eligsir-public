"""Tests for issue #1045: unregistered-scene silent coverage-guard gap.

Reproduces the originating case at the ``build_experiment_validity_block``
(``utils/experiment_validity.py``) level: a streaming run on a scene that is
not registered in ``eval/scenes.json`` (e.g. Replica) derives
``achieved == CLASS_UNKNOWN`` because there is no source-frame-count
denominator to derive coverage against. Combined with a run that ALSO failed
the real-holdout contract (``eval_valid is False`` or ``n_test == 0``, see
CLAUDE.md's "Real test holdouts" Run Contract) and was declared as more than
a smoke test, this used to complete with NO coverage-validity signal of any
kind visible at the top level.

Torch-free by construction, matching ``tests/test_experiment_validity.py``.
"""
from __future__ import annotations

from eligsir.utils.experiment_validity import (
    CLASS_FULL_CONTINUAL_VALIDATION,
    CLASS_SMOKE_ONLY,
    build_experiment_validity_block,
)


def _unregistered_scene_block(*, declared, eval_valid, n_test):
    """Build a block mirroring an unregistered-scene streaming run: some
    source indices are selected (so the run isn't a total no-op), but no
    ``source_frame_count_total`` is available (the eval/scenes.json registry
    miss), so ``achieved`` derives to ``CLASS_UNKNOWN``."""
    return build_experiment_validity_block(
        declared=declared,
        source_indices=range(54),
        source_frame_count_total=None,  # registry miss (e.g. Replica)
        is_streaming_run=True,
        eval_valid=eval_valid,
        n_test=n_test,
    )


def test_fires_for_unregistered_scene_with_failed_holdout_not_smoke():
    """The exact reproduction case from the issue: an unregistered scene,
    eval_valid=False, n_test=0, declared as more than SMOKE_ONLY."""
    block = _unregistered_scene_block(
        declared=CLASS_FULL_CONTINUAL_VALIDATION,
        eval_valid=False,
        n_test=0,
    )
    assert block["achieved"] == "UNKNOWN"
    assert block["unregistered_scene_low_coverage_risk"] is True
    assert block["unregistered_scene_low_coverage_risk_reason"] is not None
    assert "eval_valid=False" in block["unregistered_scene_low_coverage_risk_reason"]
    assert "n_test=0" in block["unregistered_scene_low_coverage_risk_reason"]
    assert any(
        "UNREGISTERED SCENE LOW COVERAGE RISK" in w for w in block["warnings"]
    )


def test_fires_on_n_test_zero_alone_even_if_eval_valid_not_explicitly_false():
    block = _unregistered_scene_block(
        declared=CLASS_FULL_CONTINUAL_VALIDATION,
        eval_valid=None,
        n_test=0,
    )
    assert block["unregistered_scene_low_coverage_risk"] is True


def test_does_not_fire_when_declared_smoke_only():
    """A legitimate short smoke test on an unregistered scene must not be
    flagged -- the issue explicitly requires this to stay a warning-only
    mechanism that never breaks smoke tests."""
    block = _unregistered_scene_block(
        declared=CLASS_SMOKE_ONLY,
        eval_valid=False,
        n_test=0,
    )
    assert block["unregistered_scene_low_coverage_risk"] is False
    assert block["unregistered_scene_low_coverage_risk_reason"] is None
    assert not any(
        "UNREGISTERED SCENE LOW COVERAGE RISK" in w for w in block["warnings"]
    )


def test_does_not_fire_when_holdout_is_valid():
    """A properly-covered, valid-holdout run on an unregistered scene must
    not be flagged even though achieved is still UNKNOWN (no registry total
    to derive coverage from) -- the gate is specifically about the
    combination with a FAILED holdout, not about registry misses alone."""
    block = _unregistered_scene_block(
        declared=CLASS_FULL_CONTINUAL_VALIDATION,
        eval_valid=True,
        n_test=42,
    )
    assert block["unregistered_scene_low_coverage_risk"] is False
    assert block["unregistered_scene_low_coverage_risk_reason"] is None


def test_does_not_fire_when_registry_hit_even_with_failed_holdout():
    """When the scene IS registered (a real source_frame_count_total is
    available), achieved is no longer UNKNOWN purely from a registry miss,
    so this specific gate must not fire -- the existing prefix_guard /
    coverage-fraction machinery already covers that case."""
    block = build_experiment_validity_block(
        declared=CLASS_FULL_CONTINUAL_VALIDATION,
        source_indices=range(126),
        source_frame_count_total=2525,
        is_streaming_run=True,
        eval_valid=False,
        n_test=0,
    )
    assert block["achieved"] != "UNKNOWN"
    assert block["unregistered_scene_low_coverage_risk"] is False


def test_omitting_eval_valid_and_n_test_reproduces_prior_output():
    """Backward compatibility: existing callers that don't pass the new
    kwargs get the exact same (non-triggering) field, never a crash."""
    block = build_experiment_validity_block(declared=None)
    assert block["unregistered_scene_low_coverage_risk"] is False
    assert block["unregistered_scene_low_coverage_risk_reason"] is None
