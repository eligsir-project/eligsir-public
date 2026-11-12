"""Tests for scripts/reconcile_issue106_corpus.py (issue #683).

Read-only, torch-free, CPU-only. Builds small synthetic run directories
under tmp_path (no real training artifacts touched) and pins the
classifier/extraction behavior the #683 evidence-map doc depends on:

- family detection by name AND by cfg_args content-sniffing;
- requested-vs-realized optimizer confound detection (issue #228 class);
- REPORT_MISSING for runs with no report.json;
- VOID_MANIPULATION for quarantined/code_error jobs;
- missing fields are recorded as "UNKNOWN" (or an UNKNOWN variant), never 0
  and never guessed;
- n_test_metrics_failed absence is distinguished from n_test_metrics_failed=0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reconcile_issue106_corpus import (  # noqa: E402
    UNKNOWN,
    build_record,
    classify_family,
    classify_run,
    discover_run_dirs,
    load_final_report,
    load_queue_index,
    parse_cfg_args,
    parse_train_log,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


def test_classify_family_by_name():
    assert classify_family("w3106inline_b1") == "w3106inline"
    assert classify_family("w3106budgets_b1adam") == "w3106budgets"
    assert classify_family("w3106cm2b_b1wc4") == "w3106cm2_post647_gen"
    assert classify_family("w3106cm_b1wc2") == "w3106cm_outage_gen"
    assert classify_family("w3_106_parity_ctrl_20260812") == "historical_smoke_parity"
    assert classify_family("something_unrelated") is None


def test_discover_run_dirs_finds_named_and_content_sniffed(tmp_path: Path):
    root = tmp_path / "output"

    # Named per convention.
    named = root / "w3106inline_b1"
    _write(named / "cfg_args", "Namespace(optimizer_type='adam')")

    # NOT named per any #106 convention, but cfg_args carries the batching
    # sentinel flag -- must still be picked up by the secondary content net.
    unnamed = root / "some_other_batch_experiment"
    _write(unnamed / "cfg_args", "Namespace(streaming_optimization_batch_size=2)")

    # Unrelated run -- must NOT be picked up.
    unrelated = root / "totally_unrelated_run"
    _write(unrelated / "cfg_args", "Namespace(optimizer_type='adam')")

    found = discover_run_dirs([str(root)])
    assert "w3106inline_b1" in found
    assert "some_other_batch_experiment" in found
    assert "totally_unrelated_run" not in found


# ---------------------------------------------------------------------------
# cfg_args / train.log extraction
# ---------------------------------------------------------------------------


def test_parse_cfg_args_missing_file_is_present_false(tmp_path: Path):
    out = parse_cfg_args(str(tmp_path / "nonexistent"))
    assert out == {"_present": False}


def test_parse_cfg_args_extracts_known_fields_and_unknown_for_absent(tmp_path: Path):
    run_dir = tmp_path / "run1"
    _write(
        run_dir / "cfg_args",
        "Namespace(optimizer_type='adam', gsplat_sparse_grad=False, "
        "parallelism_profile='off', streaming_optimization_batch_size=4, "
        "iterations=0, iterations=30000)",
    )
    out = parse_cfg_args(str(run_dir))
    assert out["optimizer_type"] == "adam"
    assert out["gsplat_sparse_grad"] == "False"
    assert out["streaming_optimization_batch_size"] == "4"
    # `iterations=0` is a known internal sentinel that appears before the
    # real value in real cfg_args files -- the LAST match must win.
    assert out["iterations"] == "30000"
    # A field never present in the file must be UNKNOWN, never a guessed 0/None.
    assert out["noise_lr"] == UNKNOWN


def test_parse_train_log_realized_parallelism_after_last_run_start(tmp_path: Path):
    run_dir = tmp_path / "run2"
    log = (
        "[RUN START] 2026-08-01T00:00:00 host=aaa pid=1 model_path=/out/run2\n"
        "[parallelism] profile=safe optimizer=selective_adam sparse_grad=True\n"
        "some noise\n"
        "[RUN START] 2026-08-01T01:00:00 host=bbb pid=1 model_path=/out/run2\n"
        "[parallelism] profile=off optimizer=adam sparse_grad=False\n"
    )
    _write(run_dir / "train.log", log)
    out = parse_train_log(str(run_dir))
    assert out["run_start_banner_count"] == 2
    # Must read the line associated with the LAST [RUN START] banner, per
    # the Run Contracts' "always search for the LAST banner" rule -- not the
    # first parallelism line in the file.
    assert out["realized_optimizer_type"] == "adam"
    assert out["realized_sparse_grad"] == "False"
    assert out["realized_profile"] == "off"


def test_parse_train_log_no_parallelism_line_is_unknown(tmp_path: Path):
    run_dir = tmp_path / "run3"
    _write(run_dir / "train.log", "[RUN START] 2026-08-01T00:00:00 host=x pid=1 model_path=/out/run3\nnothing else\n")
    out = parse_train_log(str(run_dir))
    assert out["realized_optimizer_type"] == UNKNOWN
    assert out["realized_sparse_grad"] == UNKNOWN


# ---------------------------------------------------------------------------
# report.json selection + field extraction (numeric-sort delegated to the
# existing utils.baseline_comparison helper -- this test only pins that the
# script actually reads what that helper returns, not a re-implementation).
# ---------------------------------------------------------------------------


def _make_report(n_test=315, mean_test_psnr=20.0, extra=None):
    r = {
        "iteration": 0,
        "eval_valid": True,
        "n_test": n_test,
        "n_test_metrics": n_test,
        "mean_test_psnr": mean_test_psnr,
        "eval_metadata": {"holdout_hash": "deadbeef"},
    }
    if extra:
        r.update(extra)
    return r


def test_load_final_report_picks_numerically_last_iter_not_lexical(tmp_path: Path):
    run_dir = tmp_path / "run4"
    # Lexical sort would put iter_9000 after iter_10000 is wrong; numeric
    # sort must pick iter_10000 as the later/final one.
    _write_json(run_dir / "comparison" / "iter_9000" / "report.json", _make_report(mean_test_psnr=10.0))
    _write_json(run_dir / "comparison" / "iter_10000" / "report.json", _make_report(mean_test_psnr=99.0))
    out = load_final_report(str(run_dir))
    assert out["report_available"] is True
    assert out["mean_test_psnr"] == 99.0


def test_load_final_report_no_comparison_dir_is_missing(tmp_path: Path):
    run_dir = tmp_path / "run5"
    run_dir.mkdir(parents=True)
    out = load_final_report(str(run_dir))
    assert out["report_available"] is False
    assert out["report_path"] == UNKNOWN


def test_n_test_metrics_failed_absent_is_not_zero(tmp_path: Path):
    run_dir = tmp_path / "run6"
    # A real report.json written before issue #437 -- no n_test_metrics_failed key at all.
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", _make_report())
    out = load_final_report(str(run_dir))
    assert out["n_test_metrics_failed"] == "UNKNOWN (predates issue #437 fix)"
    assert out["n_test_metrics_failed"] != 0


def test_n_test_metrics_failed_present_zero_is_distinguished_from_absent(tmp_path: Path):
    run_dir = tmp_path / "run7"
    _write_json(
        run_dir / "comparison" / "iter_1000" / "report.json",
        _make_report(extra={"n_test_metrics_failed": 0}),
    )
    out = load_final_report(str(run_dir))
    assert out["n_test_metrics_failed"] == 0


def test_scored_holdout_hash_absent_is_unknown_variant(tmp_path: Path):
    run_dir = tmp_path / "run8"
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", _make_report())
    out = load_final_report(str(run_dir))
    assert out["scored_holdout_hash"].startswith("UNKNOWN")


def test_git_provenance_absent_is_unknown_not_guessed(tmp_path: Path):
    run_dir = tmp_path / "run9"
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", _make_report())
    out = load_final_report(str(run_dir))
    assert out["git_provenance"].startswith("UNKNOWN")


# ---------------------------------------------------------------------------
# classify_run
# ---------------------------------------------------------------------------


def test_classify_run_report_missing_when_no_report(tmp_path: Path):
    run_dir = tmp_path / "runA"
    _write(run_dir / "cfg_args", "Namespace(optimizer_type='adam')")
    _write(run_dir / "train.log", "[RUN START] 2026-01-01T00:00:00 host=x pid=1 model_path=/out/runA\n")
    record = build_record("runA", str(run_dir), {})
    assert record["classification"]["primary_label"] == "REPORT_MISSING"


def test_classify_run_never_on_disk_is_report_missing():
    record = {
        "name": "cancelled_job",
        "path": "/nonexistent/path",
        "on_disk": False,
        "queue_entry": {"status": "cancelled"},
        "cfg_args": {"_present": False},
        "train_log": {"_present": False},
        "report": {"report_available": False, "report_path": UNKNOWN},
    }
    result = classify_run(record)
    assert result["primary_label"] == "REPORT_MISSING"
    assert result["flags"] == ["REPORT_MISSING"]


def test_classify_run_optimizer_confounded(tmp_path: Path):
    run_dir = tmp_path / "runB"
    # Requested selective_adam, but the realized [parallelism] line (which
    # is what actually ran, per issue #228) says plain adam.
    _write(
        run_dir / "cfg_args",
        "Namespace(optimizer_type='selective_adam', gsplat_sparse_grad=True, parallelism_profile='off')",
    )
    _write(
        run_dir / "train.log",
        "[RUN START] 2026-01-01T00:00:00 host=x pid=1 model_path=/out/runB\n"
        "[parallelism] profile=off optimizer=adam sparse_grad=False\n",
    )
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", _make_report())
    record = build_record("runB", str(run_dir), {})
    result = record["classification"]
    assert "OPTIMIZER_CONFOUNDED" in result["flags"]


def test_classify_run_void_manipulation_on_code_error_quarantine(tmp_path: Path):
    run_dir = tmp_path / "runC"
    run_dir.mkdir(parents=True)
    queue_index = {
        "runC": {
            "status": "quarantined",
            "error_category": "code_error",
            "provenance": {},
        }
    }
    record = build_record("runC", str(run_dir), queue_index)
    result = record["classification"]
    assert "VOID_MANIPULATION" in result["flags"]


def test_classify_run_scored_view_confounded_when_hash_absent(tmp_path: Path):
    run_dir = tmp_path / "runD"
    _write(run_dir / "cfg_args", "Namespace(optimizer_type='adam')")
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", _make_report())
    record = build_record("runD", str(run_dir), {})
    assert "SCORED_VIEW_CONFOUNDED" in record["classification"]["flags"]


def test_classify_run_seed_fade_confounded_on_severe_dip(tmp_path: Path):
    run_dir = tmp_path / "runE"
    _write(run_dir / "cfg_args", "Namespace(optimizer_type='adam')")
    report = _make_report(
        extra={
            "eval_metadata": {"holdout_hash": "h", "scored_holdout_hash": "s"},
            "seed_fade": {
                "detected": True,
                "max_dip_db": 16.0,
                "fade_start_iter": 100,
                "recovered": True,
                "controller_engaged": True,
            },
            "psnr_history": [],
        }
    )
    _write_json(run_dir / "comparison" / "iter_1000" / "report.json", report)
    record = build_record("runE", str(run_dir), {})
    flags = record["classification"]["flags"]
    assert "SEED_FADE_CONFOUNDED" in flags
    assert "SCORED_VIEW_CONFOUNDED" not in flags  # this one DOES have a scored hash


def test_load_queue_index_tolerates_missing_file(tmp_path: Path):
    assert load_queue_index(str(tmp_path / "does_not_exist.jsonl")) == {}
    assert load_queue_index(None) == {}
