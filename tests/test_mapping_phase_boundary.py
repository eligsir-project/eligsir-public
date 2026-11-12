"""Regression tests for the mapping/refinement phase boundary fields
(issue #346 follow-up, 2026-08-13).

Covers three layers:
1. ``utils.partition_eval.compute_trajectory_coverage``'s new
   ``last_ingest_iter`` / ``final_iteration`` / ``refinement_tail_iterations``
   / ``mapping_phase_source`` fields, in isolation (no rendering, no I/O).
2. ``utils.comparison_report.write_post_training_report``'s new
   ``last_ingest_iter`` / ``true_final_iteration`` / ``mapping_phase_source``
   parameters, including the online_mode sentinel workaround (the
   ``true_final_iteration`` override must land in
   ``trajectory_coverage.final_iteration`` WITHOUT changing
   ``summary["iteration"]`` or the output directory name).
3. ``utils.mapping_phase_backfill``'s train.log parser, against literal
   strings for BOTH historical banner phrasings this project's real logs
   contain (found scanning output/*/train.log during this fix: the
   "iterations"/"iter=" phrasing and the "optimization steps"/"step="
   phrasing are both real, not a hypothetical).
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


from eligsir.utils.partition_eval import compute_trajectory_coverage
from eligsir.utils.mapping_phase_backfill import (
    extract_final_iteration_from_log,
    extract_last_ingest_iter_from_log,
    resolve_mapping_phase,
)


# --- Layer 1: compute_trajectory_coverage ----------------------------------


class TestComputeTrajectoryCoverageMappingPhaseFields:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            scene_source="ros2://live",
            n_reg=None,
            frames_admitted=10,
            partition_psnr={},
        )
        kwargs.update(overrides)
        return kwargs

    def test_defaults_are_none_when_not_supplied(self):
        """Every call site that predates this parameter set is unaffected."""
        out = compute_trajectory_coverage(**self._base_kwargs())
        assert out["last_ingest_iter"] is None
        assert out["final_iteration"] is None
        assert out["refinement_tail_iterations"] is None
        assert out["mapping_phase_source"] is None

    def test_fields_pass_through_when_supplied(self):
        out = compute_trajectory_coverage(**self._base_kwargs(
            last_ingest_iter=3112,
            final_iteration=53112,
            mapping_phase_source="live",
        ))
        assert out["last_ingest_iter"] == 3112
        assert out["final_iteration"] == 53112
        assert out["mapping_phase_source"] == "live"

    def test_refinement_tail_computed_when_both_present(self):
        out = compute_trajectory_coverage(**self._base_kwargs(
            last_ingest_iter=3112, final_iteration=53112,
        ))
        assert out["refinement_tail_iterations"] == 50000

    def test_refinement_tail_none_when_only_one_present(self):
        out1 = compute_trajectory_coverage(**self._base_kwargs(last_ingest_iter=3112))
        assert out1["refinement_tail_iterations"] is None
        out2 = compute_trajectory_coverage(**self._base_kwargs(final_iteration=53112))
        assert out2["refinement_tail_iterations"] is None

    def test_refinement_tail_none_when_final_before_last_ingest(self):
        """Malformed/inconsistent input (e.g. a backfilled approximate lower
        bound paired with a final_iteration from a DIFFERENT, earlier run)
        must not silently produce a negative "tail" -- explicit None."""
        out = compute_trajectory_coverage(**self._base_kwargs(
            last_ingest_iter=9000, final_iteration=100,
        ))
        assert out["refinement_tail_iterations"] is None

    def test_backfilled_confidence_labels_pass_through_verbatim(self):
        out = compute_trajectory_coverage(**self._base_kwargs(
            last_ingest_iter=200, final_iteration=1000,
            mapping_phase_source="backfilled_approximate",
        ))
        assert out["mapping_phase_source"] == "backfilled_approximate"

    def test_does_not_disturb_existing_fields(self):
        """Behavior-neutrality: every pre-existing key/value in the returned
        dict is unaffected by whether the new params are supplied."""
        without = compute_trajectory_coverage(**self._base_kwargs())
        with_new = compute_trajectory_coverage(**self._base_kwargs(
            last_ingest_iter=1, final_iteration=2, mapping_phase_source="live",
        ))
        shared_keys = set(without) & set(with_new) - {
            "last_ingest_iter", "final_iteration",
            "refinement_tail_iterations", "mapping_phase_source",
        }
        for k in shared_keys:
            assert without[k] == with_new[k], k


# --- Layer 2: write_post_training_report -----------------------------------


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


def test_report_omits_mapping_phase_fields_by_default():
    """Reproduces the exact report.json shape from before this parameter
    set existed when the new kwargs are omitted -- except final_iteration,
    which defaults to the (already-correct-for-this-call) `iteration` param
    itself, see write_post_training_report's docstring."""
    from eligsir.utils.comparison_report import write_post_training_report

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        tc = summary["trajectory_coverage"]
        assert tc["last_ingest_iter"] is None
        assert tc["mapping_phase_source"] is None
        # iteration itself was correct (not the online_mode sentinel) for
        # this call, so final_iteration reflects it even though
        # true_final_iteration was never supplied.
        assert tc["final_iteration"] == 5000


def test_report_carries_last_ingest_iter_and_source():
    from eligsir.utils.comparison_report import write_post_training_report

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            last_ingest_iter=3112,
            mapping_phase_source="live",
        )
        tc = summary["trajectory_coverage"]
        assert tc["last_ingest_iter"] == 3112
        assert tc["mapping_phase_source"] == "live"
        assert tc["final_iteration"] == 5000


def test_true_final_iteration_overrides_sentinel_without_touching_iteration_field():
    """Regression guard for the exact bug this parameter exists to work
    around: train_streaming.py's FINAL report call passes
    iteration=opt.iterations (a 2^31 online_mode sentinel). Supplying
    true_final_iteration must correct trajectory_coverage.final_iteration
    AND the output directory name (issue #922) -- but summary["iteration"]
    itself (driven by the `iteration` parameter alone) must remain exactly
    the bogus sentinel value, unchanged, since other consumers (numeric-
    final-checkpoint selection) depend on that."""
    from eligsir.utils.comparison_report import write_post_training_report

    _SENTINEL = 2147483647
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=_SENTINEL,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            last_ingest_iter=3112,
            true_final_iteration=53112,
            mapping_phase_source="live",
        )
        assert summary["iteration"] == _SENTINEL  # unchanged, deliberately
        assert summary["output_dir"] == os.path.join(tmpdir, "comparison", "iter_53112")
        tc = summary["trajectory_coverage"]
        assert tc["final_iteration"] == 53112  # corrected
        assert tc["last_ingest_iter"] == 3112
        assert tc["refinement_tail_iterations"] == 50000


# --- Layer 3: train.log backfill parser -------------------------------------


class TestBackfillLogParsing:
    OLD_PHRASING_LOG = (
        "[streaming] no new frame ingested for 50000 iterations "
        "(last at iter=3112) — treating source as exhausted/idle; stopping. "
        "[26/07 20:50:53]\n"
        "[FINAL ITER 53112] Saving Gaussians at exact final iteration "
        "[26/07 20:50:53]\n"
    )
    NEW_PHRASING_LOG = (
        "[streaming] WARNING: no new frame ingested for 1000 optimization steps "
        "(last at step=200) -- this may be a genuine mapping-end, or a "
        "silently stalled source (e.g. a live SLAM frontend that stopped "
        "publishing poses). Will hard-stop at idle=50000 if this doesn't recover.\n"
        "[streaming] no new frame ingested for 50000 optimization steps "
        "(last at step=5200) — treating source as exhausted/idle; stopping.\n"
        "[FINAL STEP 55200] Saving Gaussians at exact final optimization step\n"
    )
    STALL_WARNING_ONLY_LOG = (
        "[streaming] WARNING: no new frame ingested for 1000 optimization steps "
        "(last at step=200) -- this may be a genuine mapping-end, or a "
        "silently stalled source.\n"
    )
    NO_BANNER_LOG = "[streaming] Starting: strategy=gsplat_energy_mcmc\nTraining complete.\n"

    def test_old_phrasing_exact(self):
        val, conf = extract_last_ingest_iter_from_log(self.OLD_PHRASING_LOG)
        assert (val, conf) == (3112, "exact")
        assert extract_final_iteration_from_log(self.OLD_PHRASING_LOG) == 53112

    def test_new_phrasing_exact_prefers_idle_stop_over_earlier_warning(self):
        val, conf = extract_last_ingest_iter_from_log(self.NEW_PHRASING_LOG)
        assert (val, conf) == (5200, "exact")
        assert extract_final_iteration_from_log(self.NEW_PHRASING_LOG) == 55200

    def test_stall_warning_only_is_approximate_lower_bound(self):
        val, conf = extract_last_ingest_iter_from_log(self.STALL_WARNING_ONLY_LOG)
        assert (val, conf) == (200, "approximate_lower_bound")

    def test_no_banner_returns_none(self):
        val, conf = extract_last_ingest_iter_from_log(self.NO_BANNER_LOG)
        assert (val, conf) == (None, None)
        assert extract_final_iteration_from_log(self.NO_BANNER_LOG) is None

    def test_resolve_mapping_phase_missing_dir_is_read_only_and_safe(self, tmp_path):
        r = resolve_mapping_phase(str(tmp_path / "does_not_exist"))
        assert r["last_ingest_iter"] is None
        assert r["mapping_phase_source"] is None
        assert "no report.json" in r["detail"] or "train.log" in r["detail"]

    def test_resolve_mapping_phase_backfills_from_train_log(self, tmp_path):
        run_dir = tmp_path / "some_run"
        run_dir.mkdir()
        (run_dir / "train.log").write_text(self.OLD_PHRASING_LOG)
        r = resolve_mapping_phase(str(run_dir))
        assert r["last_ingest_iter"] == 3112
        assert r["final_iteration"] == 53112
        assert r["refinement_tail_iterations"] == 50000
        assert r["mapping_phase_source"] == "backfilled_exact"

    def test_resolve_mapping_phase_prefers_live_report_json_top_level(self, tmp_path):
        """REGRESSION (issue #504): the same as the
        ``..._prefers_live_report_json_over_log`` test below, but with the
        report shaped the way ``write_post_training_report`` really writes it
        -- ``trajectory_coverage`` at the TOP LEVEL of report.json.

        The test below used a ``geometry.trajectory_coverage`` nesting that no
        writer in this repo has ever produced (measured: 0 of 400 real
        report.json files on disk), which is exactly why the consumer bug it
        was meant to guard survived: ``resolve_mapping_phase`` read that same
        nonexistent path, so its report.json branch was dead on every real run
        and every backfill silently reported ``backfilled_*`` provenance even
        when the report carried a ``"live"`` value.
        """
        import json

        run_dir = tmp_path / "already_live_top_level"
        run_dir.mkdir()
        (run_dir / "train.log").write_text(self.OLD_PHRASING_LOG)
        comp_dir = run_dir / "comparison" / "iter_99999"
        comp_dir.mkdir(parents=True)
        report = {
            "iteration": 99999,
            "trajectory_coverage": {
                "last_ingest_iter": 4000,
                "final_iteration": 99999,
                "refinement_tail_iterations": 95999,
                "mapping_phase_source": "live",
            },
        }
        (comp_dir / "report.json").write_text(json.dumps(report))
        r = resolve_mapping_phase(str(run_dir))
        assert r["last_ingest_iter"] == 4000  # from report.json, not the 3112 in train.log
        assert r["mapping_phase_source"] == "live"

    def test_resolve_mapping_phase_prefers_live_report_json_over_log(self, tmp_path):
        """If a run already has a live-populated report.json (this fix's
        forward-going half), that value wins over re-parsing train.log --
        even if train.log's own banner would suggest a different number
        (e.g. from a stale/rerun-corrupted log).

        NOTE (issue #504): the ``geometry.trajectory_coverage`` nesting below
        is NOT the real on-disk shape -- see the top-level variant above. It
        is kept only to pin the accessor's defensive ``geometry`` fallback.
        """
        import json

        run_dir = tmp_path / "already_live"
        run_dir.mkdir()
        (run_dir / "train.log").write_text(self.OLD_PHRASING_LOG)
        comp_dir = run_dir / "comparison" / "iter_99999"
        comp_dir.mkdir(parents=True)
        report = {
            "iteration": 99999,
            "geometry": {
                "trajectory_coverage": {
                    "last_ingest_iter": 4000,
                    "final_iteration": 99999,
                    "refinement_tail_iterations": 95999,
                    "mapping_phase_source": "live",
                }
            },
        }
        (comp_dir / "report.json").write_text(json.dumps(report))
        r = resolve_mapping_phase(str(run_dir))
        assert r["last_ingest_iter"] == 4000  # from report.json, not the 3112 in train.log
        assert r["mapping_phase_source"] == "live"

    def test_resolve_mapping_phase_falls_back_to_log_when_report_has_only_final_iteration(
        self, tmp_path
    ):
        """Regression guard (found in review before this shipped): a
        report.json written by a call site that never supplies
        last_ingest_iter (e.g. render_report.py's post-hoc re-render, or
        the not-yet-wired rolling_seed_application.py/
        submap_stitch_application.py integration sites) can still have a
        populated final_iteration -- write_post_training_report's
        _effective_final_iteration always falls back to the `iteration`
        parameter itself. An all-or-nothing "does the report have
        SOMETHING" gate would wrongly skip train.log parsing here and
        report last_ingest_iter=None even though the idle-stop banner is
        right there in the log. Must be resolved PER FIELD instead."""
        import json

        run_dir = tmp_path / "final_iter_only_report"
        run_dir.mkdir()
        (run_dir / "train.log").write_text(self.OLD_PHRASING_LOG)
        comp_dir = run_dir / "comparison" / "iter_5000"
        comp_dir.mkdir(parents=True)
        report = {
            "iteration": 5000,
            "geometry": {
                "trajectory_coverage": {
                    "last_ingest_iter": None,
                    "final_iteration": 5000,
                    "mapping_phase_source": None,
                }
            },
        }
        (comp_dir / "report.json").write_text(json.dumps(report))
        r = resolve_mapping_phase(str(run_dir))
        # last_ingest_iter recovered from train.log, NOT silently dropped.
        assert r["last_ingest_iter"] == 3112
        assert r["mapping_phase_source"] == "backfilled_exact"
        # report.json's own final_iteration (5000) still wins over the log's
        # FINAL ITER banner (53112) -- report values take priority per field.
        assert r["final_iteration"] == 5000
        # refinement_tail recomputed against the ACTUALLY-returned pair, not
        # copied stale from either source alone.
        assert r["refinement_tail_iterations"] == 5000 - 3112

    def test_backfill_ignores_stale_attempt_before_last_run_start_banner(self):
        """CLAUDE.md's 'Unique output folder per run' contract: ablator's
        rerun/requeue reuses model_path, so train.log for a rerun job
        silently concatenates old and new attempts with no separator --
        always trust only content after the LAST [RUN START] banner."""
        concatenated_log = (
            "[RUN START] 2026-01-01T00:00:00 host=h1 pid=1 model_path=output/x\n"
            + self.OLD_PHRASING_LOG  # stale attempt 1: last_ingest_iter=3112, final=53112
            + "[RUN START] 2026-01-02T00:00:00 host=h1 pid=2 model_path=output/x\n"
            "[streaming] no new frame ingested for 50000 optimization steps "
            "(last at step=9999) — treating source as exhausted/idle; stopping.\n"
            "[FINAL STEP 59999] Saving Gaussians at exact final optimization step\n"
        )
        val, conf = extract_last_ingest_iter_from_log(concatenated_log)
        assert (val, conf) == (9999, "exact")
        assert extract_final_iteration_from_log(concatenated_log) == 59999

    def test_resolve_mapping_phase_never_writes(self, tmp_path):
        """Read-only contract: no new files, no mutation of train.log."""
        run_dir = tmp_path / "readonly_check"
        run_dir.mkdir()
        log_path = run_dir / "train.log"
        log_path.write_text(self.OLD_PHRASING_LOG)
        before = set(os.listdir(run_dir))
        before_mtime = log_path.stat().st_mtime
        resolve_mapping_phase(str(run_dir))
        after = set(os.listdir(run_dir))
        assert before == after
        assert log_path.stat().st_mtime == before_mtime
