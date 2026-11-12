"""Tests for issue #420: comparability must track the EXACT camera subset
that metrics were actually computed over, not just the realized holdout pool.

Background. ``realized_holdout_hash`` (issue #199) fingerprints the test
cameras a run realized -- the *pool*. But ``write_post_training_report``
then subsamples that pool down to ``--streaming_report_test_metrics_max``
views via ``_representative_cams`` before computing PSNR/SSIM/LPIPS, and
drops any camera whose render raised. So two runs can share an identical
``realized_holdout_hash`` AND an identical ``n_test`` AND an identical
``n_test_metrics`` while ``mean_test_psnr`` was averaged over genuinely
different cameras -- and, before this fix, still be reported at the
strongest comparability level (``verification_level="realized_hash"``,
``holdout_verified=True``).

The discriminating test is ``test_same_realized_pool_different_scored_
subset_is_refused``: every pre-existing guard (holdout_hash, n_test,
realized_holdout_hash, n_test_metrics) matches there, so it passes on
unmodified main and only the new scored-subset hash can catch it.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from eligsir.utils.baseline_comparison import compare_against_baseline


def _write_baseline(tmpdir_path: str, name: str, *, eval_metadata: dict,
                    n_test=10, n_test_metrics=None, mean_test_psnr=22.44,
                    psnr_history=((1000, 20.0), (2000, 22.44)),
                    iteration=52000) -> str:
    """Write a baseline run in the real on-disk layout
    (<model_path>/comparison/iter_{N}/report.json + psnr_history.jsonl)."""
    run_dir = os.path.join(tmpdir_path, name)
    comparison_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comparison_dir, exist_ok=True)
    report = {
        "mean_test_psnr": mean_test_psnr,
        "eval_valid": True,
        "n_test": n_test,
        "eval_metadata": dict(eval_metadata),
    }
    if n_test_metrics is not None:
        report["n_test_metrics"] = n_test_metrics
    with open(os.path.join(comparison_dir, "report.json"), "w") as f:
        json.dump(report, f)
    with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
        for it, val in psnr_history:
            f.write(json.dumps({"iteration": it, "mean_test_psnr": val}) + "\n")
    return run_dir


def _write_current(tmpdir_path: str, name: str,
                   psnr_history=((1000, 20.1), (2000, 22.40))) -> str:
    """The current run only needs its psnr_history.jsonl on disk -- its
    report content is passed in-memory as ``current_summary``."""
    run_dir = os.path.join(tmpdir_path, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "psnr_history.jsonl"), "w") as f:
        for it, val in psnr_history:
            f.write(json.dumps({"iteration": it, "mean_test_psnr": val}) + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# THE discriminating case (passes on unmodified main -- that is the bug)
# ---------------------------------------------------------------------------

def test_same_realized_pool_different_scored_subset_is_refused():
    """Identical selection-rule hash, identical realized pool hash, identical
    n_test AND identical n_test_metrics -- only the actually-scored camera
    subset differs. Every guard that existed before issue #420 matches here,
    so this must be refused solely on the scored-subset hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
                "scored_holdout_hash": "scored_AAA",
            },
            n_test=253, n_test_metrics=128,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 253,
            "n_test_metrics": 128,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
                "scored_holdout_hash": "scored_BBB",  # <-- the ONLY difference
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False, result
        assert result["verification_level"] == "scored_hash_mismatch", result
        assert "scored_AAA" in result["reason"] and "scored_BBB" in result["reason"]


def test_matching_scored_subset_is_the_new_strongest_level():
    """Both sides carry a matching scored-subset hash on top of a matching
    realized pool hash: the only configuration that now earns
    holdout_verified=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
                "scored_holdout_hash": "scored_AAA",
            },
            n_test=253, n_test_metrics=128,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 253,
            "n_test_metrics": 128,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
                "scored_holdout_hash": "scored_AAA",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True, result
        assert result["verification_level"] == "scored_hash", result
        assert result["holdout_verified"] is True, result
        assert result["baseline_scored_holdout_hash"] == "scored_AAA"


# ---------------------------------------------------------------------------
# Downgrade when exact provenance is unavailable
# ---------------------------------------------------------------------------

def test_realized_match_without_scored_hash_is_downgraded():
    """A matching realized pool hash with NO scored-subset hash on either
    side no longer earns the top level: the pool is verified but the scored
    subset's provenance is unknown, so holdout_verified is the narrower
    truthy marker 'realized_hash', not True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
            },
            n_test=10, n_test_metrics=10,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 10,
            "n_test_metrics": 10,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True, result
        assert result["verification_level"] == "realized_hash", result
        assert result["holdout_verified"] == "realized_hash", result
        assert result["holdout_verified"] is not True


def test_one_sided_scored_hash_downgrades_rather_than_refusing():
    """Only one side has the scored-subset hash (a legacy baseline compared
    against a post-fix current run). Cannot be verified at the top level,
    but must NOT be refused -- it degrades to the realized-pool level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
            },
            n_test=10, n_test_metrics=10,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 10,
            "n_test_metrics": 10,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
                "scored_holdout_hash": "scored_AAA",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True, result
        assert result["verification_level"] == "realized_hash", result


# ---------------------------------------------------------------------------
# n_test_metrics as the weaker legacy guard
# ---------------------------------------------------------------------------

def test_n_test_metrics_mismatch_refused_under_matching_realized_pool():
    """Real mechanism: two runs of the same scene launched with different
    --streaming_report_test_metrics_max (128 vs 64) realize the identical
    holdout pool but average PSNR over different-sized subsets. Neither
    carries a scored-subset hash, so n_test_metrics is the only guard --
    and it must refuse, consistent with the existing n_test refusal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
            },
            n_test=253, n_test_metrics=128,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 253,
            "n_test_metrics": 64,
            "eval_metadata": {
                "holdout_hash": "abc123",
                "realized_holdout_hash": "realized_xyz",
            },
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is False, result
        assert result["verification_level"] == "n_test_metrics_mismatch", result
        assert "128" in result["reason"] and "64" in result["reason"]


def test_n_test_metrics_only_level_without_realized_hash():
    """No realized pool hash on either side (a fully legacy pair), but both
    carry n_test_metrics and it matches: a level above bare n_test_only,
    still well below the scored-subset hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={"holdout_hash": "abc123"},
            n_test=253, n_test_metrics=128,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 253,
            "n_test_metrics": 128,
            "eval_metadata": {"holdout_hash": "abc123"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True, result
        assert result["verification_level"] == "n_test_metrics_only", result


def test_legacy_pair_missing_n_test_metrics_still_degrades_to_n_test_only():
    """Backward compatibility: reports predating every new field must never
    crash and never be promoted -- they land on the pre-existing
    n_test_only level exactly as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata={"holdout_hash": "abc123"},
            n_test=62, n_test_metrics=None,
        )
        current_dir = _write_current(tmpdir, "current")
        current_summary = {
            "mean_test_psnr": 22.40,
            "n_test": 62,
            "eval_metadata": {"holdout_hash": "abc123"},
        }
        result = compare_against_baseline(baseline_dir, current_summary, current_dir)
        assert result["comparable"] is True, result
        assert result["verification_level"] == "n_test_only", result
        assert result["holdout_verified"] == "n_test_only", result


# ---------------------------------------------------------------------------
# _scored_subset_hash / _camera_set_hash unit behavior
# ---------------------------------------------------------------------------

def test_camera_set_hash_distinguishes_equal_size_different_subsets():
    """The pool is identical; two evenly-spaced subsamples of the same size
    pick different cameras out of it. Pool hash matches, subset hash must
    not -- this is the mechanism the discriminating test simulates."""
    from eligsir.utils.baseline_comparison import _camera_set_hash, _realized_holdout_hash

    class _FakeFrame:
        def __init__(self, index):
            self.index = index

    class _FakeCam:
        def __init__(self, index):
            self._streaming_frame = _FakeFrame(index)

    pool = [_FakeCam(i) for i in range(20)]
    subset_a = pool[0:20:4]   # 0, 4, 8, 12, 16
    subset_b = pool[1:20:4]   # 1, 5, 9, 13, 17
    assert len(subset_a) == len(subset_b)
    assert _realized_holdout_hash(pool) == _realized_holdout_hash(list(reversed(pool)))
    assert _camera_set_hash(subset_a) != _camera_set_hash(subset_b)
    # Order-independent, and withheld (None) for an empty set.
    assert _camera_set_hash(list(reversed(subset_a))) == _camera_set_hash(subset_a)
    assert _camera_set_hash([]) is None


# ---------------------------------------------------------------------------
# End-to-end through write_post_training_report: the hash must come from the
# SCORED subset, not the pool.
# ---------------------------------------------------------------------------

def _report_module():
    """write_post_training_report needs lpipsPyTorch importable; the sibling
    tests/test_realized_holdout_hash.py uses the same MagicMock shim.

    Unlike the rest of this file (pure-python, see utils/baseline_comparison.py),
    the end-to-end tests below go through utils.comparison_report, which imports
    torch. The CPU-only smoke image (Dockerfile.ci-cpu) deliberately has no
    torch, so skip there rather than erroring at collection (issue #423)."""
    import sys
    from unittest.mock import MagicMock

    pytest.importorskip("torch", reason="utils.comparison_report requires torch")
    sys.modules.setdefault("lpipsPyTorch", MagicMock())
    import eligsir.utils.comparison_report as cr
    return cr


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeFrame:
    def __init__(self, index):
        self.index = index


class _FakeCam:
    def __init__(self, index):
        self._streaming_frame = _FakeFrame(index)
        self.image_name = f"f{index:04d}"
        self.uid = index
        self.original_image = None


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    image, _ = declare_canonical_score_tensor(
        torch.zeros(3, 64, 64), context="test_scored_subset_render"
    )
    return {"render": image}


def test_end_to_end_same_pool_different_order_yields_same_pool_hash_different_scored_hash(monkeypatch):
    """The real mechanism behind issue #420: two runs realize the IDENTICAL
    holdout pool, but ``_representative_cams`` picks evenly-spaced POSITIONS,
    so a differently-ordered pool at the same cap scores different cameras.

    realized_holdout_hash is order-independent and matches; n_test and
    n_test_metrics match; only scored_holdout_hash separates them.
    """
    cr = _report_module()   # skips when torch is absent (CPU smoke image, #423)
    import torch
    # This is a scored-subset identity test, not LPIPS asset qualification.
    # CI simulation deliberately lacks the scientific image's baked weights.
    import lpipsPyTorch
    monkeypatch.setattr(
        lpipsPyTorch,
        "lpips",
        lambda _render, _gt, net_type="vgg": torch.zeros(1),
    )
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    # _gt_image unconditionally moves the GT to CUDA; this test is about the
    # hash wiring, not rendering, so stub it to a CPU tensor.
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    monkeypatch.setattr(
        cr,
        "_gt_image",
        lambda cam: declare_canonical_score_tensor(
            torch.zeros(3, 64, 64), context="test_scored_subset_gt"
        )[0],
    )

    pool = [_FakeCam(i) for i in range(12)]
    pool_reordered = list(reversed(pool))
    shared_meta = {"holdout_hash": "same_selection_rule", "n_holdout": 12}

    def _run(cams, tmpdir):
        return cr.write_post_training_report(
            model_path=tmpdir, iteration=5000, gaussians=_FakeGaussian(),
            train_cams=[], test_cams=cams, render_fn=_minimal_render_fn,
            pipe=_FakePipe(), background=None,
            eval_metadata=dict(shared_meta), test_metrics_max=3,
        )

    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        a = _run(pool, ta)
        b = _run(pool_reordered, tb)

    assert a["n_test"] == b["n_test"] == 12
    assert a["n_test_metrics"] == b["n_test_metrics"] == 3
    assert (a["eval_metadata"]["realized_holdout_hash"]
            == b["eval_metadata"]["realized_holdout_hash"])
    assert (a["eval_metadata"]["scored_holdout_hash"]
            != b["eval_metadata"]["scored_holdout_hash"]), (
        "scored_holdout_hash must fingerprint the scored SUBSET, not the pool")

    # And such a pair must now be refused by compare_against_baseline.
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = _write_baseline(
            tmpdir, "baseline",
            eval_metadata=dict(b["eval_metadata"]),
            n_test=b["n_test"], n_test_metrics=b["n_test_metrics"],
        )
        current_dir = _write_current(tmpdir, "current")
        result = compare_against_baseline(
            baseline_dir,
            {"mean_test_psnr": 22.4, "n_test": a["n_test"],
             "n_test_metrics": a["n_test_metrics"],
             "eval_metadata": dict(a["eval_metadata"])},
            current_dir,
        )
        assert result["comparable"] is False, result
        assert result["verification_level"] == "scored_hash_mismatch", result


def test_scored_hash_withheld_on_degenerate_split(monkeypatch):
    """A degenerate (*_bootstrap_views debug) split must never emit anything
    that looks like a real holdout fingerprint -- same gating
    realized_holdout_hash already uses."""
    cr = _report_module()
    monkeypatch.setenv("SPLAT_REPORT_OVERVIEW", "0")
    cams = [_FakeCam(i) for i in range(4)]
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = cr.write_post_training_report(
            model_path=tmpdir, iteration=5000, gaussians=_FakeGaussian(),
            train_cams=cams, test_cams=cams, render_fn=_minimal_render_fn,
            pipe=_FakePipe(), background=None, degenerate_split=True,
        )
    meta = summary.get("eval_metadata") or {}
    assert "scored_holdout_hash" not in meta
    assert "realized_holdout_hash" not in meta
