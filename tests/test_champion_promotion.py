"""Tests for utils/champion_promotion.py -- the validation/selection logic
behind scripts/promote_champion.py (issue #332).

CPU-only, no GPU, no real (multi-hundred-MB) run artifacts: everything
here uses small synthetic fixture report.json/train.log content and tiny
placeholder files. This intentionally exercises the exact failure modes
found on real champion runs during the investigation for this issue:
- a reused output folder producing multiple concatenated "Training
  complete." markers in one train.log (fr3par_hybrid_gate, 12 markers),
- a pre-2026-07-24 log with no [FINAL STEP N]/[FINAL ITER N] banner at
  all, where the real final iteration must be inferred from the last
  periodic checkpoint-save line,
- a PLY that is NOT at the real final iteration (the floor2 champion's
  actual gap),
- report.json fields (map_reinit_invalidated, seed_fade, holdout hashes)
  either present-and-blocking, present-and-fine, or altogether absent
  (legacy reports) -- these must not be conflated.
"""

from __future__ import annotations

import json
import os

import pytest

from eligsir.utils.champion_promotion import (
    build_manifest,
    file_record,
    generate_champion_id,
    parse_final_iteration_from_log,
    select_checkpoint,
    select_ply,
    select_report_dir,
    sha256_file,
    validate_report_contract,
)


# ---------------------------------------------------------------------------
# parse_final_iteration_from_log
# ---------------------------------------------------------------------------

def test_final_step_banner_is_high_confidence():
    log = "\n".join([
        "[RUN START] 2026-08-01T00:00:00 host=x pid=1 model_path=output/foo",
        "[ITER 4000] Saving Gaussians",
        "[FINAL STEP 4321] Saving Gaussians at exact final optimization step",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 4321
    assert result.source == "final_step_banner"
    assert result.confidence == "high"
    assert result.n_run_start_banners == 1
    assert not result.warnings


def test_final_iter_banner_is_high_confidence():
    log = "\n".join([
        "[RUN START] 2026-07-24T00:00:00 host=x pid=1 model_path=output/foo",
        "[FINAL ITER 55386] Saving Gaussians at exact final iteration",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 55386
    assert result.source == "final_iter_banner"
    assert result.confidence == "high"


def test_legacy_log_falls_back_to_last_checkpoint_save_before_complete():
    """Real fr3par_hybrid_gate shape: no FINAL STEP/ITER banner, but a
    '[ITER N] Saving Checkpoint' line right before 'Training complete.'."""
    log = "\n".join([
        "[ITER 48000] Saving Gaussians",
        "[ITER 50000] Saving Gaussians",
        "[ITER 52000] Saving Gaussians",
        "[ITER 52000] Saving Checkpoint",
        "[streaming] Training complete.",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 52000
    assert result.source == "checkpoint_save_near_complete"
    assert result.confidence == "medium"
    assert any("PERIODIC save point" in w for w in result.warnings)


def test_legacy_log_falls_back_to_gaussians_save_when_no_checkpoint():
    log = "\n".join([
        "[ITER 34000] Saving Gaussians",
        "[ITER 52000] Saving Gaussians",
        "[streaming] Training complete.",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 52000
    assert result.source == "gaussians_save_near_complete"
    assert result.confidence == "medium"
    assert any("No checkpoint was ever saved" in w for w in result.warnings)


def test_reused_folder_multiple_training_complete_markers_uses_last_segment():
    """Real fr3par_hybrid_gate shape: the folder was reused across many
    attempts (12 'Training complete.' lines in the real log). Only the
    LAST segment's checkpoint save must be used, and the tool must warn
    loudly that this happened."""
    log = "\n".join([
        # first (stale) attempt, would-be-wrong answer if picked
        "[ITER 8000] Saving Checkpoint",
        "[streaming] Training complete.",
        "Training complete.",
        # second (real, final) attempt
        "[ITER 40000] Saving Gaussians",
        "[ITER 52000] Saving Gaussians",
        "[ITER 52000] Saving Checkpoint",
        "[streaming] Training complete.",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 52000
    assert result.n_training_complete_markers == 4
    assert any("multiple training attempts are concatenated" in w for w in result.warnings)


def test_multiple_run_start_banners_warns():
    log = "\n".join([
        "[RUN START] t1 host=a pid=1 model_path=x",
        "[ITER 8000] Saving Checkpoint",
        "[RUN START] t2 host=a pid=2 model_path=x",
        "[FINAL STEP 9000] Saving Gaussians at exact final optimization step",
        "Training complete.",
    ])
    result = parse_final_iteration_from_log(log)
    assert result.iteration == 9000
    assert result.n_run_start_banners == 2
    assert any("reused across multiple launch attempts" in w for w in result.warnings)


def test_no_markers_at_all_returns_none_with_actionable_warning():
    result = parse_final_iteration_from_log("some unrelated log content\nmore lines\n")
    assert result.iteration is None
    assert result.source == "none"
    assert result.confidence == "none"
    assert any("--final-iteration" in w for w in result.warnings)


def test_training_complete_with_no_nearby_save_returns_none():
    log = "unrelated preamble\n" + "Training complete.\n"
    result = parse_final_iteration_from_log(log)
    assert result.iteration is None
    assert result.source == "none"


# ---------------------------------------------------------------------------
# select_report_dir
# ---------------------------------------------------------------------------

def _write_report(path: str, **overrides) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "iteration": 52000,
        "eval_valid": True,
        "n_test": 315,
        "mean_test_psnr": 22.44,
        "split_kind": "holdout",
    }
    report.update(overrides)
    with open(path, "w") as f:
        json.dump(report, f)


def test_select_report_dir_finds_single_holdout_report(tmp_path):
    run_dir = str(tmp_path)
    _write_report(os.path.join(run_dir, "comparison", "iter_52000", "report.json"))
    result = select_report_dir(run_dir)
    assert result.report_dir == os.path.join(run_dir, "comparison", "iter_52000")
    assert not result.warnings


def test_select_report_dir_excludes_bootstrap_debug_dir(tmp_path):
    run_dir = str(tmp_path)
    _write_report(os.path.join(run_dir, "comparison", "iter_52000", "report.json"))
    _write_report(
        os.path.join(run_dir, "comparison", "iter_52000_bootstrap_views", "report.json"),
        split_kind="degenerate_train_views",
    )
    result = select_report_dir(run_dir)
    assert result.report_dir == os.path.join(run_dir, "comparison", "iter_52000")


def test_select_report_dir_refuses_to_guess_between_two_holdout_reports(tmp_path):
    """This is the exact class of mistake CLAUDE.md documents happening
    twice already (lexically-sorted iter_* guessing) -- the tool must
    refuse, not silently pick one."""
    run_dir = str(tmp_path)
    _write_report(os.path.join(run_dir, "comparison", "iter_7000", "report.json"))
    _write_report(os.path.join(run_dir, "comparison", "iter_2147483647", "report.json"))
    result = select_report_dir(run_dir)
    assert result.report_dir is None
    assert any("ambiguous" in w for w in result.warnings)


def test_select_report_dir_missing_comparison_dir(tmp_path):
    result = select_report_dir(str(tmp_path))
    assert result.report_dir is None


# ---------------------------------------------------------------------------
# validate_report_contract
# ---------------------------------------------------------------------------

def _base_report(**overrides) -> dict:
    report = {
        "eval_valid": True,
        "n_test": 315,
        "split_kind": "holdout",
        "n_test_metrics": 315,
        "eval_metadata": {
            "holdout_hash": "abc123",
            "realized_holdout_hash": "def456",
            # Issue #433: full holdout provenance is now required for
            # promotion, so the "clean report" fixture carries it.
            "scored_holdout_hash": "ghi789",
        },
        "trajectory_coverage": {"map_reinit_invalidated": False, "verdict": "FULL"},
        "seed_fade": {"detected": False},
    }
    report.update(overrides)
    return report


def test_clean_report_has_no_blockers_or_warnings():
    result = validate_report_contract(_base_report())
    assert result.ok
    assert result.blockers == []
    assert result.warnings == []


def test_eval_invalid_is_a_blocker():
    result = validate_report_contract(_base_report(eval_valid=False))
    assert not result.ok
    assert any("eval_valid" in b for b in result.blockers)


def test_zero_n_test_is_a_blocker():
    result = validate_report_contract(_base_report(n_test=0))
    assert not result.ok
    assert any("n_test" in b for b in result.blockers)


def test_map_reinit_invalidated_true_is_a_blocker():
    report = _base_report(trajectory_coverage={"map_reinit_invalidated": True, "verdict": "FULL"})
    result = validate_report_contract(report)
    assert not result.ok
    assert any("map_reinit_invalidated" in b for b in result.blockers)


def test_suspected_truncated_is_a_blocker():
    report = _base_report(trajectory_coverage={"map_reinit_invalidated": False, "verdict": "SUSPECTED_TRUNCATED"})
    result = validate_report_contract(report)
    assert not result.ok
    assert any("SUSPECTED_TRUNCATED" in b for b in result.blockers)


def test_missing_trajectory_coverage_is_a_warning_not_a_blocker():
    """Absent must not be silently treated as False -- it's a legacy
    report predating the detector, and must be surfaced, not swallowed."""
    report = _base_report()
    del report["trajectory_coverage"]
    result = validate_report_contract(report)
    assert result.ok
    assert any("trajectory_coverage block missing" in w for w in result.warnings)


def test_unrecovered_seed_fade_is_a_blocker_by_default():
    report = _base_report(seed_fade={"detected": True, "recovered": False, "max_dip_db": 9.1})
    result = validate_report_contract(report)
    assert not result.ok
    assert any("UNRECOVERED" in b for b in result.blockers)


def test_unrecovered_seed_fade_promotable_with_force_ack():
    report = _base_report(seed_fade={"detected": True, "recovered": False, "max_dip_db": 9.1})
    result = validate_report_contract(report, force_seed_fade_ack=True)
    assert result.ok
    assert any("promoted anyway via --force-seed-fade-ack" in w for w in result.warnings)


def test_recovered_seed_fade_is_disclosed_but_not_blocking():
    report = _base_report(seed_fade={"detected": True, "recovered": True, "max_dip_db": 4.28})
    result = validate_report_contract(report)
    assert result.ok
    assert any("recovered=True" in w for w in result.warnings)


def test_missing_seed_fade_block_is_a_warning():
    report = _base_report()
    del report["seed_fade"]
    result = validate_report_contract(report)
    assert result.ok
    assert any("seed_fade block missing" in w for w in result.warnings)


def test_missing_holdout_hash_is_a_warning():
    # The selection-RULE hash alone is still only a warning: a comparison can
    # reach verification_level="scored_hash" without it (the realized/scored
    # pair is what the ladder keys on).
    report = _base_report(eval_metadata={
        "realized_holdout_hash": "def456",
        "scored_holdout_hash": "ghi789",
    })
    result = validate_report_contract(report)
    assert result.ok, result.blockers
    assert any("holdout_hash missing" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Issue #433: promotion gate consumes compare_against_baseline's ladder
# ---------------------------------------------------------------------------

def test_missing_scored_holdout_hash_is_a_blocker():
    """A legacy report (every report written before 2026-08-13) can never
    reach verification_level='scored_hash', so it must refuse EXPLICITLY --
    the #420 hole this issue exists to close.
    """
    report = _base_report(
        n_test=315,
        n_test_metrics=128,
        eval_metadata={"holdout_hash": "abc123", "realized_holdout_hash": "def456"},
    )
    result = validate_report_contract(report)
    assert not result.ok
    assert any("scored_holdout_hash" in b for b in result.blockers)
    assert any("n_test_metrics=128" in b for b in result.blockers)


def test_scored_hash_without_realized_hash_is_a_blocker():
    """compare_against_baseline only consults the scored hash once the pool
    hash already matched, so a scored hash without a pool hash is
    structurally unverifiable -- it must not pass either.
    """
    report = _base_report(eval_metadata={
        "holdout_hash": "abc123",
        "scored_holdout_hash": "ghi789",
    })
    result = validate_report_contract(report)
    assert not result.ok
    assert any("realized_holdout_hash" in b for b in result.blockers)


def test_missing_scored_hash_promotable_with_explicit_override():
    report = _base_report(eval_metadata={
        "holdout_hash": "abc123", "realized_holdout_hash": "def456",
    })
    result = validate_report_contract(report, allow_unverifiable_scored_subset=True)
    assert result.ok, result.blockers
    assert any("allow_unverifiable_scored_subset" in w for w in result.warnings)


def test_baseline_comparison_scored_hash_mismatch_is_a_blocker():
    report = _base_report(baseline_comparison={
        "baseline_run": "output/some_baseline",
        "comparable": False,
        "holdout_verified": False,
        "verification_level": "scored_hash_mismatch",
        "reason": "scored_holdout_hash mismatch: this run=aaa baseline=bbb",
    })
    result = validate_report_contract(report)
    assert not result.ok
    assert any("scored_hash_mismatch" in b for b in result.blockers)


@pytest.mark.parametrize("level", [
    "hash_mismatch",
    "realized_hash_mismatch",
    "n_test_mismatch",
    "n_test_metrics_mismatch",
])
def test_every_baseline_comparison_refusal_level_blocks(level):
    report = _base_report(baseline_comparison={
        "baseline_run": "output/some_baseline",
        "comparable": False,
        "verification_level": level,
        "reason": "refused",
    })
    result = validate_report_contract(report)
    assert not result.ok
    assert any(level in b for b in result.blockers)


def test_baseline_comparison_legacy_baseline_is_a_warning_not_a_blocker():
    """A baseline that predates scored_holdout_hash caps the level at
    'realized_hash'. That is the BASELINE's provenance gap, not this
    candidate's -- disclose it, don't refuse the promotion.
    """
    report = _base_report(baseline_comparison={
        "baseline_run": "output/legacy_baseline",
        "comparable": True,
        "holdout_verified": "realized_hash",
        "verification_level": "realized_hash",
        "reason": None,
    })
    result = validate_report_contract(report)
    assert result.ok, result.blockers
    assert any("realized_hash" in w for w in result.warnings)


def test_baseline_comparison_infrastructure_failure_is_a_warning():
    """comparable=False for a missing/unloadable baseline is an
    infrastructure failure, not a comparability refusal -- a cleaned-up
    baseline directory must not make a valid champion unpromotable.
    """
    report = _base_report(baseline_comparison={
        "baseline_run": "output/deleted_baseline",
        "comparable": False,
        "holdout_verified": False,
        "verification_level": "unverified",
        "reason": "baseline_run path does not exist: output/deleted_baseline",
    })
    result = validate_report_contract(report)
    assert result.ok, result.blockers
    assert any("could not be loaded" in w for w in result.warnings)


def test_baseline_comparison_at_scored_hash_is_silent():
    report = _base_report(baseline_comparison={
        "baseline_run": "output/good_baseline",
        "comparable": True,
        "holdout_verified": True,
        "verification_level": "scored_hash",
        "reason": None,
    })
    result = validate_report_contract(report)
    assert result.ok, result.blockers
    # Deliberately not `warnings == []`: this function has other, unrelated
    # disclosure warnings. Assert only that the comparability ladder itself
    # said nothing.
    assert not any("verification_level" in w for w in result.warnings), result.warnings


# ---------------------------------------------------------------------------
# select_ply / select_checkpoint
# ---------------------------------------------------------------------------

def _touch(path: str, content: bytes = b"x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def test_select_ply_exact_match(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "point_cloud", "iteration_52000", "point_cloud.ply"))
    result = select_ply(run_dir, 52000)
    assert result.exact is True
    assert result.blockers == []
    assert result.path.endswith("iteration_52000/point_cloud.ply")


def test_select_ply_no_exact_match_blocks(tmp_path):
    """This is the real floor2 champion's actual gap: only a periodic
    PLY exists (iteration_52000) but the real final iteration was later
    (~53900) -- must fail loudly, not silently substitute."""
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "point_cloud", "iteration_52000", "point_cloud.ply"))
    result = select_ply(run_dir, 53900)
    assert result.path is None
    assert any("no exact PLY" in b for b in result.blockers)


def test_select_ply_nearest_allowed_with_explicit_tolerance(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "point_cloud", "iteration_52000", "point_cloud.ply"))
    result = select_ply(run_dir, 53900, allow_nearest_tolerance=2000)
    assert result.path is not None
    assert result.exact is False
    assert any("nearest available" in w for w in result.warnings)


def test_select_ply_tolerance_still_refuses_if_gap_too_large(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "point_cloud", "iteration_52000", "point_cloud.ply"))
    result = select_ply(run_dir, 60000, allow_nearest_tolerance=100)
    assert result.path is None
    assert result.blockers


def test_select_ply_missing_dir_blocks(tmp_path):
    result = select_ply(str(tmp_path), 52000)
    assert result.path is None
    assert result.blockers


def test_select_checkpoint_exact_match(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "chkpnt52000.pth"))
    result = select_checkpoint(run_dir, 52000)
    assert result.exact is True
    assert result.blockers == []


def test_select_checkpoint_missing_blocks_by_default(tmp_path):
    """This is the real fr3 champion's actual gap: no chkpnt*.pth
    survives on shared storage despite the log showing it was saved."""
    result = select_checkpoint(str(tmp_path), 52000)
    assert result.path is None
    assert any("no chkpnt" in b for b in result.blockers)


def test_select_checkpoint_missing_allowed_with_explicit_flag(tmp_path):
    result = select_checkpoint(str(tmp_path), 52000, allow_missing=True)
    assert result.path is None
    assert result.blockers == []
    assert any("WITHOUT a resumable checkpoint" in w for w in result.warnings)


def test_select_checkpoint_nearest_allowed_with_tolerance(tmp_path):
    run_dir = str(tmp_path)
    _touch(os.path.join(run_dir, "chkpnt50000.pth"))
    result = select_checkpoint(run_dir, 52000, allow_nearest_tolerance=5000)
    assert result.exact is False
    assert result.path is not None


# ---------------------------------------------------------------------------
# hashing / manifest
# ---------------------------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = os.path.join(str(tmp_path), "f.bin")
    data = os.urandom(1 << 20 + 7)
    with open(path, "wb") as f:
        f.write(data)
    assert sha256_file(path, chunk_size=1024) == hashlib.sha256(data).hexdigest()


def test_file_record_includes_size_and_hash(tmp_path):
    path = os.path.join(str(tmp_path), "f.bin")
    with open(path, "wb") as f:
        f.write(b"hello world")
    rec = file_record(path)
    assert rec["filename"] == "f.bin"
    assert rec["size_bytes"] == 11
    assert len(rec["sha256"]) == 64


def test_generate_champion_id_shape():
    cid = generate_champion_id("fr3", "20260813", "abcdef1234567890")
    assert cid == "fr3-20260813-abcdef1"


def test_build_manifest_shape(tmp_path):
    run_dir = str(tmp_path)
    ply_path = os.path.join(run_dir, "point_cloud", "iteration_52000", "point_cloud.ply")
    _touch(ply_path, b"ply-bytes")
    from eligsir.utils.champion_promotion import ArtifactSelection

    ply_sel = ArtifactSelection(ply_path, 52000, True)
    ckpt_sel = ArtifactSelection(None, None, False)
    report = _base_report()

    manifest = build_manifest(
        champion_id="fr3-20260813-abc1234",
        dataset="tum",
        scene="fr3",
        run_dir=run_dir,
        reason_issue=332,
        real_final_iteration=52000,
        final_iteration_source="final_step_banner",
        final_iteration_confidence="high",
        ply=ply_sel,
        checkpoint=ckpt_sel,
        report=report,
        report_dir=os.path.join(run_dir, "comparison", "iter_52000"),
        git_sha="deadbeef",
        hostname="testhost",
        date_promoted="2026-08-13T00:00:00+00:00",
        storage_backend="release",
    )
    assert manifest["champion_id"] == "fr3-20260813-abc1234"
    assert manifest["final_ply"]["filename"] == "point_cloud.ply"
    assert manifest["final_ply"]["exact_final_iteration"] is True
    assert manifest["final_checkpoint"] is None
    assert manifest["mandatory_metrics"]["eval_valid"] is True
    # Issue #433: the scored-subset provenance the gate now requires must be
    # persisted, or a future comparison against this champion could not reach
    # verification_level="scored_hash".
    assert manifest["mandatory_metrics"]["scored_holdout_hash"] == "ghi789"
    assert manifest["mandatory_metrics"]["n_test_metrics"] == 315
    assert manifest["status"] == "current"
    assert manifest["storage_backend"] == "release"


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
