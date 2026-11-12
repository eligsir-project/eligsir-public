"""Pin scripts/regrade_seed_fade_corpus.py's classifier against synthetic
report.json fixtures (issue #684).

Torch-free by construction: every fixture run directory carries only
``comparison/iter_<N>/report.json`` (+ optionally ``run_args``/``train.log``)
and NO ``chkpnt*.pth`` file, so the lineage-audit code path this script calls
never reaches its lazy ``import torch`` (only exercised when a real
checkpoint exists to load). ``utils.seed_fade_checker`` and
``utils.seed_fade_severity`` are themselves torch-free modules (verified by
inspection: no ``import torch`` at module scope in either file).

These tests exercise ``regrade_run`` directly against hand-built report
dicts -- not ``discover_corpus``/``main`` -- so they pin the classification
logic itself, independent of which real run directories happen to exist on
this machine.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "regrade_seed_fade_corpus",
        os.path.join(REPO_ROOT, "scripts", "regrade_seed_fade_corpus.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REGRADE = _load_module()


# --- fixture builders --------------------------------------------------

def _write_report(run_dir: str, report: dict, iteration: int = 100000) -> str:
    comp_dir = os.path.join(run_dir, "comparison", f"iter_{iteration}")
    os.makedirs(comp_dir, exist_ok=True)
    path = os.path.join(comp_dir, "report.json")
    with open(path, "w") as f:
        json.dump(report, f)
    return path


def _base_report(**overrides) -> dict:
    report = {
        "eval_valid": True,
        "n_test": 128,
        "mean_test_psnr": 24.0,
        "trajectory_coverage": {
            "verdict": "FULL",
            "map_reinit_invalidated": False,
            "map_reinit_reason": None,
            "stream_end_reason": "duration_reached",
        },
        "seed_fade": None,
    }
    report.update(overrides)
    return report


POST_282_439_CLEAN = {
    "detected": False,
    "max_dip_db": 0.0,
    "fade_start_iter": None,
    "recovered": False,
    "controller_engaged": False,
    "status": "CLEAN",
    "suppression_confounded": False,
    "resumed_from_iteration": None,
    "resume_state_restored": None,
    "indeterminate": False,
    "indeterminate_reasons": [],
    "seed_view_resolutions": ["480x640"],
    "seed_view_resolution_counts": {"480x640": 500},
    "resolution_changed": False,
    "max_dip_reference_resolution": "480x640",
    "trough_resolution": None,
    "max_dip_cross_resolution": False,
    "resolution_confounded": False,
    "resolution_confounded_reasons": [],
    "n_samples_with_resolution": 500,
    "observe_only": True,
    "n_samples": 500,
    "n_samples_pre_warmup": 20,
    "n_samples_in_fade": 0,
    "fade_sample_fraction": 0.0,
    "sample_interval": 100,
    "sustained_checks": 3,
    "detection_latency_iters": 300,
    "tolerance_db": 2.0,
    "warmup_iters": 2000,
    "episode_count": 0,
    "first_fade_start_iter": None,
    "last_fade_start_iter": None,
    "fade_active_at_end": False,
    "recovered_final": None,
    "deepest_dip_db": None,
    "trough_iter": None,
    "trough_psnr": None,
    "trough_reference_psnr": None,
    "running_max_db": 24.0,
    "final_psnr": 24.0,
    "final_iter": 100000,
    "min_psnr": 20.0,
    "suppression_applied": False,
    "min_applied_scale": 1.0,
    "last_applied_scale": 1.0,
    "n_applied_calls": 100000,
    "n_suppression_calls": 0,
    "episodes": [],
}


def _detected_post_schema(**overrides) -> dict:
    block = dict(POST_282_439_CLEAN)
    block.update({
        "detected": True,
        "max_dip_db": 3.0,
        "status": "FADE_OBSERVED",
        "n_samples_in_fade": 10,
        "fade_sample_fraction": 10 / 500,
        "episode_count": 1,
        "first_fade_start_iter": 5000,
        "last_fade_start_iter": 5000,
        "fade_start_iter": 5000,
        "fade_active_at_end": False,
        "recovered_final": True,
        "recovered": True,
        "deepest_dip_db": 3.0,
        "trough_iter": 5100,
        "trough_psnr": 21.0,
        "trough_reference_psnr": 24.0,
        "episodes": [{"start_iter": 5000, "end_iter": 5200, "max_gap_db": 3.0,
                       "min_psnr": 21.0, "n_samples": 10}],
    })
    block.update(overrides)
    return block


class RegradeClassifierTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def _run_dir(self, name: str) -> str:
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def _regrade(self, name: str, report: dict, lineage=None, run_args=None, resume_log=False):
        run_dir = self._run_dir(name)
        _write_report(run_dir, report)
        if run_args is not None:
            with open(os.path.join(run_dir, "run_args"), "w") as f:
                json.dump(run_args, f)
        if resume_log:
            with open(os.path.join(run_dir, "train.log"), "w") as f:
                f.write("[RUN START] t=0 host=x pid=1 model_path=" + run_dir + "\n")
                f.write("[streaming] resume: from checkpoint chkpnt1000.pth\n")
        lineage_index = {run_dir: lineage} if lineage is not None else {}
        return REGRADE.regrade_run(name, run_dir, [], lineage_index)

    # --- gates ------------------------------------------------------

    def test_eval_invalid_is_invalid_conclusion(self):
        row = self._regrade("r_eval_invalid", _base_report(
            eval_valid=False, seed_fade=_detected_post_schema()))
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")
        self.assertIn("gate_failure", row["applicable_flags"])

    def test_zero_n_test_is_invalid_conclusion(self):
        row = self._regrade("r_ntest0", _base_report(
            n_test=0, seed_fade=_detected_post_schema()))
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")

    def test_map_reinit_is_invalid_conclusion(self):
        report = _base_report(seed_fade=_detected_post_schema())
        report["trajectory_coverage"]["map_reinit_invalidated"] = True
        report["trajectory_coverage"]["map_reinit_reason"] = "second map created"
        row = self._regrade("r_reinit", report)
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")

    def test_missing_seed_fade_block_is_invalid_conclusion(self):
        row = self._regrade("r_no_block", _base_report(seed_fade=None))
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")
        self.assertIn("no_seed_fade_block", row["applicable_flags"])

    def test_pre_282_block_with_no_n_samples_key_and_undetected_is_invalid_not_clean(self):
        """Regression pin (review finding before this shipped): a pre-#282
        block has NO 'n_samples' key at all. If it ALSO reports
        detected=False, that is indistinguishable from 'the detector never
        ran' -- the exact NO_SAMPLES hazard -- and must not fall through to
        the permissive VALID_CURRENT_SEMANTICS default just because every
        later check requires `detected` to be truthy to fire."""
        block = {
            "detected": False,
            "max_dip_db": 0.0,
            "fade_start_iter": None,
            "recovered": False,
            "controller_engaged": False,
        }
        row = self._regrade("r_pre282_undetected", _base_report(seed_fade=block))
        self.assertEqual(row["checker"]["schema"], "pre_282")
        self.assertNotIn("n_samples", block)
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")
        self.assertIn("pre_282_no_sample_count_undetected", row["applicable_flags"])

    def test_checker_refusal_surfaced_even_when_label_is_not_invalid(self):
        """A pre-#282 block WITH a detected fade can legitimately reach
        VALID_BUT_CONFOUNDED/VALID_CURRENT_SEMANTICS via the curve-based
        re-derivation, but checker.certified=False must still be visible in
        classification_reasons on that row -- not just buried in the
        'checker' sub-dict -- so a reader scanning only the top-level label
        and its first reason cannot miss that the run's RAW fields were
        never independently certified."""
        block = {
            "detected": True,
            "max_dip_db": 2.93,
            "fade_start_iter": 28900,
            "recovered": True,
            "controller_engaged": True,
        }
        row = self._regrade("r_checker_refused_visible", _base_report(seed_fade=block))
        self.assertFalse(row["checker_certified"])
        self.assertTrue(
            any("checker.certified=False" in r for r in row["classification_reasons"]),
            row["classification_reasons"],
        )

    def test_no_samples_is_invalid_conclusion_not_clean(self):
        block = _detected_post_schema(detected=False, status="NO_SAMPLES",
                                       n_samples=0, n_samples_in_fade=0,
                                       episode_count=0, episodes=[])
        row = self._regrade("r_no_samples", _base_report(seed_fade=block))
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")
        self.assertIn("no_samples_detector_never_ran", row["applicable_flags"])

    # --- resume (#290) -----------------------------------------------

    def test_resumed_without_state_restored_is_indeterminate(self):
        report = _base_report(seed_fade=_detected_post_schema(),
                               resumed_from_iteration=5000)
        row = self._regrade("r_resumed_bad", report,
                             run_args={"start_checkpoint": "chkpnt5000.pth"},
                             resume_log=True)
        self.assertEqual(row["final_classification"], "RESUME_INDETERMINATE")
        self.assertTrue(row["resume"]["resumed"])
        self.assertTrue(row["resume"]["indeterminate"])

    def test_resumed_with_state_restored_true_is_not_indeterminate(self):
        block = _detected_post_schema(resume_state_restored=True)
        report = _base_report(seed_fade=block, resumed_from_iteration=5000)
        row = self._regrade("r_resumed_ok", report,
                             run_args={"start_checkpoint": "chkpnt5000.pth"},
                             resume_log=True)
        self.assertNotEqual(row["final_classification"], "RESUME_INDETERMINATE")

    # --- clean current-semantics path ---------------------------------

    def test_clean_post_282_439_run_is_valid_current_semantics(self):
        row = self._regrade("r_clean", _base_report(seed_fade=dict(POST_282_439_CLEAN)))
        self.assertEqual(row["final_classification"], "VALID_CURRENT_SEMANTICS")

    def test_severe_severity_is_valid_but_confounded(self):
        # A long, unrecovered plateau with a large dip -> severity escalates.
        # fade_active_at_end=True requires exactly one OPEN episode
        # (end_iter=None) per check_seed_fade_block's own episode-arithmetic
        # consistency check.
        block = _detected_post_schema(
            max_dip_db=20.0, deepest_dip_db=20.0,
            fade_active_at_end=True, recovered_final=False, recovered=True,
            episodes=[{"start_iter": 5000, "end_iter": None, "max_gap_db": 20.0,
                       "min_psnr": 4.0, "n_samples": 10}],
        )
        row = self._regrade("r_severe", _base_report(seed_fade=block))
        self.assertEqual(row["final_classification"], "VALID_BUT_CONFOUNDED")
        self.assertIn("severity_confounded", row["applicable_flags"])

    def test_resolution_confounded_is_valid_but_confounded(self):
        block = _detected_post_schema(resolution_confounded=True,
                                       resolution_confounded_reasons=["cross-resolution trough"])
        row = self._regrade("r_rescon", _base_report(seed_fade=block))
        self.assertEqual(row["final_classification"], "VALID_BUT_CONFOUNDED")

    # --- #439 resolution-provenance uncheckability ---------------------

    def test_pre_439_block_with_detected_fade_is_resolution_uncheckable(self):
        # Pre-#439: no 'resolution_confounded' key at all, but has the full
        # #282 schema otherwise and a real detected fade.
        block = _detected_post_schema()
        del block["resolution_confounded"]
        del block["resolution_confounded_reasons"]
        row = self._regrade("r_pre439", _base_report(seed_fade=block))
        self.assertEqual(row["final_classification"], "LEGACY_RESOLUTION_UNCHECKABLE")
        self.assertFalse(row["resolution"]["checkable"])

    # --- pre-#282 schema -------------------------------------------------

    def test_pre_282_schema_with_resolution_key_present_is_gradeable(self):
        # Pre-#282 (missing status/n_samples/episodes/etc) but WITH a
        # resolution_confounded key present (hypothetical -- exercises that
        # the pre-282 branch is reached only when resolution IS checkable).
        block = {
            "detected": True,
            "max_dip_db": 2.93,
            "fade_start_iter": 28900,
            "recovered": True,
            "controller_engaged": True,
            "resolution_confounded": False,
        }
        row = self._regrade("r_pre282", _base_report(seed_fade=block))
        self.assertEqual(row["checker"]["schema"], "pre_282")
        self.assertIn(row["final_classification"],
                       {"VALID_CURRENT_SEMANTICS", "VALID_BUT_CONFOUNDED"})
        # Never certified -- but also never silently promoted to clean
        # without a severity re-derivation.
        self.assertFalse(row["checker"]["certified"])
        self.assertIsNotNone(row["regraded_seed_fade"])

    def test_real_fr3postmerge_champ_asis_shaped_block_is_resolution_uncheckable(self):
        """Regression pin for the exact shape found on-disk for
        fr3postmerge_champ_asis / issue278mode_iterbased_spf20 -- pre-#282
        AND pre-#439 (no resolution_confounded key at all): resolution
        uncheckability must win over the pre-282 branch, per this script's
        documented precedence order."""
        block = {
            "detected": True,
            "max_dip_db": 2.93,
            "fade_start_iter": 28900,
            "recovered": True,
            "controller_engaged": True,
        }
        row = self._regrade("r_fr3shape", _base_report(seed_fade=block))
        self.assertEqual(row["final_classification"], "LEGACY_RESOLUTION_UNCHECKABLE")

    # --- cohort trust (#287/#407) ---------------------------------------

    def test_cohort_dependent_family_with_untrustworthy_lineage(self):
        run_dir_name = "r_cohort"
        run_dir = self._run_dir(run_dir_name)
        _write_report(run_dir, _base_report(seed_fade=dict(POST_282_439_CLEAN)))
        lineage_index = {run_dir: {"status": "lineage_no_seed_range",
                                    "reason": "predates #373"}}
        row = REGRADE.regrade_run(run_dir_name, run_dir,
                                   ["287_seed_cohort_drift_S0_S1"], lineage_index)
        self.assertEqual(row["final_classification"], "LEGACY_COARSE_COHORT_ONLY")

    def test_cohort_dependent_family_with_trustworthy_lineage_stays_valid(self):
        run_dir_name = "r_cohort_ok"
        run_dir = self._run_dir(run_dir_name)
        _write_report(run_dir, _base_report(seed_fade=dict(POST_282_439_CLEAN)))
        lineage_index = {run_dir: {"status": "trustworthy", "reason": ""}}
        row = REGRADE.regrade_run(run_dir_name, run_dir,
                                   ["287_seed_cohort_drift_S0_S1"], lineage_index)
        self.assertEqual(row["final_classification"], "VALID_CURRENT_SEMANTICS")

    # --- explicit supersession (#407) ------------------------------------

    def test_explicit_superseded_run_wins_regardless_of_case(self):
        run_dir_name = "w3_106_matchedview_B1_20260812"  # mixed case, as on disk
        run_dir = self._run_dir(run_dir_name)
        _write_report(run_dir, _base_report(seed_fade=dict(POST_282_439_CLEAN)))
        row = REGRADE.regrade_run(run_dir_name, run_dir, ["407_stand_in_cohort_superseded"], {})
        self.assertEqual(row["final_classification"], "SUPERSEDED_BY_EXACT_REANALYSIS")

    # --- no report at all -------------------------------------------------

    def test_no_report_json_is_invalid_conclusion(self):
        run_dir = self._run_dir("r_no_report")
        row = REGRADE.regrade_run("r_no_report", run_dir, [], {})
        self.assertEqual(row["final_classification"], "INVALID_FOR_SEED_FADE_CONCLUSION")
        self.assertIsNone(row["report_path"])

    # --- historical values preserved, never overwritten -------------------

    def test_historical_block_preserved_verbatim(self):
        block = _detected_post_schema(max_dip_db=17.5)
        row = self._regrade("r_preserve", _base_report(seed_fade=block, mean_test_psnr=19.9))
        self.assertEqual(row["historical"]["seed_fade"]["max_dip_db"], 17.5)
        self.assertEqual(row["historical"]["mean_test_psnr"], 19.9)
        # Regraded severity may differ from the raw historical dict, but the
        # historical one must be untouched (same object contents).
        self.assertIn("severity", row["regraded_seed_fade"])


if __name__ == "__main__":
    unittest.main()
