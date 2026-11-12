"""Issue #685: pins ``utils/coverage_access.py``, the single authoritative
coverage/final-iteration accessor, against synthetic fixtures covering each
named hazard: the online_mode sentinel, a multi-``[RUN START]`` log, numeric
(not lexical) ``iter_*`` sort, a missing field resolving to UNKNOWN (never 0
or 100%), and a truncation reason forcing exclusion.

Torch-free by construction (mirrors ``tests/test_trajectory_coverage_report_location.py``
and ``tests/test_mapping_phase_boundary.py``): every import that would pull
in ``torch`` (``utils.partition_eval``) is deferred inside
``utils.coverage_access`` itself, so this file needs no ``tests/conftest.py``
``_NEEDS_TORCH`` entry and runs without a GPU container.
"""
from __future__ import annotations

import json
import os

from eligsir.utils.coverage_access import (
    ONLINE_UNBOUNDED_ITERATIONS,
    diff_legacy_vs_corrected,
    get_run_coverage,
    legacy_naive_report_path,
)


def _write_report(comparison_dir, iter_name, report: dict):
    d = os.path.join(comparison_dir, iter_name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(report, f)
    return d


def _base_trajectory_coverage(**overrides):
    tc = {
        "scene_registered": True,
        "n_source_frames": 2525,
        "frames_admitted": 268,
        "coverage_fraction": 0.1061,
        "frame_admission": "all",
        "stream_end_reason": None,
        "map_reinit_invalidated": False,
        "mapping_result_valid": True,
        "verdict": "PARTIAL_BY_DESIGN",
        "last_ingest_iter": None,
        "final_iteration": None,
        "mapping_phase_source": None,
    }
    tc.update(overrides)
    return tc


# --- hazard 1: report.json["iteration"] bogus online_mode sentinel --------

class TestOnlineSentinel:
    def test_bogus_sentinel_is_never_treated_as_real_final_iteration(self, tmp_path):
        run_dir = str(tmp_path / "run_online")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, f"iter_{ONLINE_UNBOUNDED_ITERATIONS}",
            {
                "iteration": ONLINE_UNBOUNDED_ITERATIONS,
                "eval_valid": True,
                "n_test": 10,
                "trajectory_coverage": _base_trajectory_coverage(),
            },
        )
        s = get_run_coverage(run_dir)
        # No trajectory_coverage.final_iteration, no train.log -- must resolve
        # to UNKNOWN, never silently accept the bogus sentinel as real.
        assert s.final_iteration != ONLINE_UNBOUNDED_ITERATIONS
        assert s.final_iteration is None
        assert s.report_iteration_is_bogus_sentinel is True
        assert s.final_iteration_confidence == "none"

    def test_final_iteration_from_trajectory_coverage_block_wins(self, tmp_path):
        run_dir = str(tmp_path / "run_online2")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, f"iter_{ONLINE_UNBOUNDED_ITERATIONS}",
            {
                "iteration": ONLINE_UNBOUNDED_ITERATIONS,
                "eval_valid": True,
                "n_test": 10,
                "trajectory_coverage": _base_trajectory_coverage(
                    final_iteration=100001, last_ingest_iter=52000,
                    mapping_phase_source="live",
                ),
            },
        )
        s = get_run_coverage(run_dir)
        assert s.final_iteration == 100001
        assert s.final_iteration_source == "report.trajectory_coverage"
        assert s.is_online_sentinel is False
        assert s.last_ingest_iter == 52000

    def test_non_sentinel_report_iteration_is_a_legitimate_fallback(self, tmp_path):
        """A legacy train.py --eval (non-streaming) report's "iteration" is
        real -- only the sentinel value itself is untrustworthy."""
        run_dir = str(tmp_path / "run_legacy")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_30000",
            {"iteration": 30000, "eval_valid": True, "n_test": 5},
        )
        s = get_run_coverage(run_dir)
        assert s.final_iteration == 30000
        assert s.final_iteration_confidence == "medium"
        assert s.report_iteration_is_bogus_sentinel is False


# --- hazard 2: multiple [RUN START] banners in train.log -------------------

class TestMultiRunStartLog:
    def test_last_run_start_segment_wins_for_final_iteration(self, tmp_path):
        run_dir = tmp_path / "run_rerun"
        run_dir.mkdir()
        log = (
            "[RUN START] 2026-01-01T00:00:00 host=a pid=1 model_path=x\n"
            "[FINAL STEP 999]\n"
            "Training complete.\n"
            "[RUN START] 2026-01-01T01:00:00 host=a pid=2 model_path=x\n"
            "[FINAL STEP 30000]\n"
            "Training complete.\n"
        )
        (run_dir / "train.log").write_text(log)
        (run_dir / "comparison").mkdir()

        s = get_run_coverage(str(run_dir))
        assert s.final_iteration == 30000
        assert s.final_iteration_source == "final_step_banner"
        assert s.final_iteration_confidence == "high"

    def test_last_run_start_segment_wins_for_last_ingest_iter(self, tmp_path):
        run_dir = tmp_path / "run_rerun2"
        run_dir.mkdir()
        log = (
            "[RUN START] 2026-01-01T00:00:00 host=a pid=1 model_path=x\n"
            "[streaming] no new frame ingested for 50000 iterations "
            "(last at iter=111) — treating source as exhausted/idle; stopping.\n"
            "[RUN START] 2026-01-01T01:00:00 host=a pid=2 model_path=x\n"
            "[streaming] no new frame ingested for 50000 iterations "
            "(last at iter=9285) — treating source as exhausted/idle; stopping.\n"
        )
        (run_dir / "train.log").write_text(log)
        (run_dir / "comparison").mkdir()

        s = get_run_coverage(str(run_dir))
        assert s.last_ingest_iter == 9285


# --- hazard 3: numeric, not lexical, iter_* sort ----------------------------

class TestNumericSort:
    def test_report_selection_is_numeric_not_lexical(self, tmp_path):
        """iter_7000 lexically sorts AFTER iter_30000 ('7' > '3') -- the
        exact bug this project's Run Contracts and #504/#666 name."""
        run_dir = str(tmp_path / "run_multi_iter")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_7000", {"iteration": 7000, "eval_valid": True, "n_test": 1,
                                           "mean_test_psnr": 18.71})
        _write_report(comp, "iter_30000", {"iteration": 30000, "eval_valid": True, "n_test": 6,
                                            "mean_test_psnr": 17.44})

        s = get_run_coverage(run_dir)
        assert s.report_path.endswith(os.path.join("iter_30000", "report.json"))
        assert s.n_test == 6

    def test_online_sentinel_dir_always_wins_numeric_sort(self, tmp_path):
        run_dir = str(tmp_path / "run_sentinel_dir")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_30000", {"iteration": 30000, "eval_valid": True, "n_test": 6})
        _write_report(
            comp, f"iter_{ONLINE_UNBOUNDED_ITERATIONS}",
            {"iteration": ONLINE_UNBOUNDED_ITERATIONS, "eval_valid": True, "n_test": 9},
        )
        s = get_run_coverage(run_dir)
        assert s.n_test == 9  # the sentinel dir, numerically largest

    def test_bootstrap_views_variant_is_skipped(self, tmp_path):
        run_dir = str(tmp_path / "run_bootstrap")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_30000_bootstrap_views",
                       {"iteration": 30000, "eval_valid": True, "n_test": 999,
                        "split_kind": "degenerate_train_views"})
        _write_report(comp, "iter_30000", {"iteration": 30000, "eval_valid": True, "n_test": 6})
        s = get_run_coverage(run_dir)
        assert s.n_test == 6
        assert s.report_path.endswith(os.path.join("iter_30000", "report.json"))
        assert "_bootstrap_views" not in os.path.basename(os.path.dirname(s.report_path))

    def test_legacy_lexical_sort_reproduces_the_historical_bug(self, tmp_path):
        """Pin the MEASUREMENT tool itself: legacy_naive_report_path must
        actually reproduce the bug being measured, or diff_legacy_vs_corrected
        would silently report zero incidents on a corpus that has some."""
        run_dir = str(tmp_path / "run_multi_iter2")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_7000", {"iteration": 7000})
        _write_report(comp, "iter_30000", {"iteration": 30000})
        legacy_path = legacy_naive_report_path(run_dir)
        assert legacy_path.endswith(os.path.join("iter_7000", "report.json"))


# --- hazard 4: missing field -> UNKNOWN, never 0 or 100% -------------------

class TestMissingFieldIsUnknown:
    def test_no_report_at_all_is_unknown_not_zero(self, tmp_path):
        s = get_run_coverage(str(tmp_path / "does_not_exist"))
        assert s.report_found is False
        assert s.verdict == "UNKNOWN_NO_REPORT"
        assert s.coverage_fraction is None
        assert s.n_source_frames is None
        assert s.frames_admitted is None

    def test_no_coverage_block_is_unknown_not_zero(self, tmp_path):
        run_dir = str(tmp_path / "run_no_block")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_1000", {"iteration": 1000, "eval_valid": True, "n_test": 1})
        s = get_run_coverage(run_dir)
        assert s.report_found is True
        assert s.verdict == "UNKNOWN_NO_COVERAGE_BLOCK"
        assert s.coverage_fraction is None  # not 0.0

    def test_frames_released_is_always_explicitly_unavailable(self, tmp_path):
        """The 4th distinct coverage quantity (source released, i.e.
        FrameScheduler pacing) is never persisted in report.json -- must
        never silently alias to frames_admitted."""
        run_dir = str(tmp_path / "run_released")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_1000",
            {"iteration": 1000, "eval_valid": True, "n_test": 1,
             "trajectory_coverage": _base_trajectory_coverage(frames_admitted=42)},
        )
        s = get_run_coverage(run_dir)
        assert s.frames_admitted == 42
        assert s.frames_released is None
        assert "not persisted" in s.frames_released_unavailable_reason


# --- hazard 5: truncation reason forces exclusion ---------------------------

class TestTruncationForcesExclusion:
    def test_suspected_truncated_verdict_flagged(self, tmp_path):
        run_dir = str(tmp_path / "run_truncated")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(
                    verdict="SUSPECTED_TRUNCATED",
                    stream_end_reason="idle_timeout_no_completion_signal",
                ),
            },
        )
        s = get_run_coverage(run_dir)
        assert s.verdict == "SUSPECTED_TRUNCATED"
        assert s.suspected_truncated is True
        assert s.stream_end_reason == "idle_timeout_no_completion_signal"

    def test_full_verdict_is_not_flagged(self, tmp_path):
        run_dir = str(tmp_path / "run_full")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(verdict="FULL"),
            },
        )
        s = get_run_coverage(run_dir)
        assert s.suspected_truncated is False


# --- depth_insertion vs trajectory_coverage frames_admitted conflation -----

class TestFramesAdmittedMismatch:
    def test_mismatch_between_the_two_independent_counters_is_flagged(self, tmp_path):
        run_dir = str(tmp_path / "run_mismatch")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(frames_admitted=22),
                "geometry": {"depth_insertion": {"enabled": True, "total_inserted": 0,
                                                  "frames_admitted": 0, "inert": False}},
            },
        )
        s = get_run_coverage(run_dir)
        assert s.frames_admitted_mismatch is True
        assert s.total_inserted == 0

    def test_no_mismatch_when_counters_agree(self, tmp_path):
        run_dir = str(tmp_path / "run_agree")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(frames_admitted=22),
                "geometry": {"depth_insertion": {"enabled": True, "total_inserted": 500,
                                                  "frames_admitted": 22, "inert": False}},
            },
        )
        s = get_run_coverage(run_dir)
        assert s.frames_admitted_mismatch is False


# --- diff_legacy_vs_corrected --------------------------------------------

class TestDiffLegacyVsCorrected:
    def test_gate_flip_detected_when_legacy_misses_truncation(self, tmp_path):
        """Legacy reads report["geometry"]["trajectory_coverage"] -- a path
        no writer ever produces -- so it always misses SUSPECTED_TRUNCATED,
        silently including a run the corrected accessor would exclude."""
        run_dir = str(tmp_path / "run_gate_flip")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(
                    verdict="SUSPECTED_TRUNCATED",
                    stream_end_reason="idle_timeout_no_completion_signal",
                ),
            },
        )
        d = diff_legacy_vs_corrected(run_dir)
        assert d["corrected_verdict"] == "SUSPECTED_TRUNCATED"
        assert d["legacy_verdict"] != "SUSPECTED_TRUNCATED"
        assert d["gate_flip"] is True

    def test_no_gate_flip_when_both_clean(self, tmp_path):
        run_dir = str(tmp_path / "run_no_flip")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, "iter_5000",
            {
                "iteration": 5000, "eval_valid": True, "n_test": 3,
                "trajectory_coverage": _base_trajectory_coverage(verdict="FULL"),
            },
        )
        d = diff_legacy_vs_corrected(run_dir)
        assert d["gate_flip"] is False

    def test_sentinel_read_as_real_flagged(self, tmp_path):
        run_dir = str(tmp_path / "run_sentinel_flag")
        comp = os.path.join(run_dir, "comparison")
        _write_report(
            comp, f"iter_{ONLINE_UNBOUNDED_ITERATIONS}",
            {"iteration": ONLINE_UNBOUNDED_ITERATIONS, "eval_valid": True, "n_test": 3},
        )
        d = diff_legacy_vs_corrected(run_dir)
        assert d["sentinel_read_as_real"] is True
        assert d["legacy_final_iteration"] == ONLINE_UNBOUNDED_ITERATIONS
        assert d["corrected_final_iteration"] != ONLINE_UNBOUNDED_ITERATIONS

    def test_report_path_differs_flagged(self, tmp_path):
        run_dir = str(tmp_path / "run_path_diff")
        comp = os.path.join(run_dir, "comparison")
        _write_report(comp, "iter_7000", {"iteration": 7000})
        _write_report(comp, "iter_30000", {"iteration": 30000})
        d = diff_legacy_vs_corrected(run_dir)
        assert d["report_path_differs"] is True
        assert d["legacy_report_path"].endswith(os.path.join("iter_7000", "report.json"))
        assert d["corrected_report_path"].endswith(os.path.join("iter_30000", "report.json"))
