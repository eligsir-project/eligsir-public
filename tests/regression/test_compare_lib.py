"""Host-runnable unit tests for the pure-data comparison core (no torch/GPU).

These cover the back-half-degradation guard and cfg-diff on synthetic report
trees, so the metric that catches the 7k->final collapse is itself tested.
"""
import json
import os

import pytest

from eligsir.eval.compare import (
    checkpoint_curve,
    degradation_report,
    parse_cfg_args,
    cfg_diff,
    emit_markdown,
)


def _write_report(comp_dir, it, *, test_psnr, n_live, needle=0.05, eval_valid=True, n_test=6):
    d = os.path.join(comp_dir, f"iter_{it}")
    os.makedirs(d, exist_ok=True)
    rep = {
        "iteration": it,
        "mean_test_psnr": test_psnr,
        "mean_train_psnr": test_psnr - 0.1,
        "eval_valid": eval_valid,
        "n_test": n_test,
        "eval_metadata": {"holdout_hash": "deadbeef"},
        "geometry": {"shape": {"n_live": n_live, "needle_frac_lt_0.3": needle}},
    }
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(rep, f)


def _make_run(tmp_path, name, points):
    """points: list of (iter, test_psnr, n_live). Returns the run dir."""
    run = tmp_path / name
    comp = run / "comparison"
    comp.mkdir(parents=True)
    for it, psnr, nlive in points:
        _write_report(str(comp), it, test_psnr=psnr, n_live=nlive)
    return str(run)


def test_checkpoint_curve_reads_all_iters(tmp_path):
    run = _make_run(tmp_path, "r", [(7000, 18.0, 6000), (30000, 17.0, 3000)])
    curve = checkpoint_curve(run)
    assert set(curve) == {7000, 30000}
    assert curve[7000]["test_psnr"] == 18.0
    assert curve[30000]["n_live"] == 3000


def test_checkpoint_curve_ignores_bootstrap_views(tmp_path):
    run = _make_run(tmp_path, "r", [(30000, 17.0, 3000)])
    # a bootstrap-views debug dir must be ignored
    boot = tmp_path / "r" / "comparison" / "iter_30000_bootstrap_views"
    boot.mkdir(parents=True)
    (boot / "report.json").write_text('{"mean_test_psnr": 99.0}')
    curve = checkpoint_curve(run)
    assert set(curve) == {30000}
    assert curve[30000]["test_psnr"] == 17.0


def test_degradation_flagged_on_psnr_drop(tmp_path):
    run = _make_run(tmp_path, "bad", [(7000, 18.7, 6000), (30000, 10.6, 5800)])
    dr = degradation_report(run)
    assert dr.degraded is True
    assert dr.peak_iter == 7000 and dr.final_iter == 30000
    assert dr.psnr_delta == pytest.approx(10.6 - 18.7, abs=1e-6)
    assert any("test_psnr fell" in r for r in dr.reasons)


def test_nlive_drop_with_held_psnr_is_consolidation_not_degraded(tmp_path):
    # PSNR held but live count drops -> NOT a quality regression (healthy prune).
    run = _make_run(tmp_path, "consol", [(7000, 17.0, 3942), (30000, 16.9, 463)])
    dr = degradation_report(run)
    assert dr.psnr_degraded is False
    assert dr.degraded is False            # quality verdict keys on PSNR
    assert dr.nlive_collapsed is True      # but the shrink is still surfaced
    assert dr.nlive_ratio == pytest.approx(463 / 3942, abs=1e-6)
    assert any("consolidation" in r for r in dr.reasons)


def test_nlive_drop_with_psnr_drop_is_underconstruction(tmp_path):
    run = _make_run(tmp_path, "under", [(7000, 17.0, 3942), (30000, 10.6, 463)])
    dr = degradation_report(run)
    assert dr.psnr_degraded is True and dr.degraded is True
    assert dr.nlive_collapsed is True
    assert any("under-construction" in r for r in dr.reasons)


def test_healthy_run_not_degraded(tmp_path):
    run = _make_run(tmp_path, "good", [(7000, 17.0, 5000), (30000, 17.3, 4900)])
    dr = degradation_report(run)
    assert dr.degraded is False
    assert dr.reasons == []


def test_parse_cfg_args_handles_lists_and_strings(tmp_path):
    p = tmp_path / "cfg_args"
    p.write_text("Namespace(sh_degree=3, source_path='/data/x', "
                 "save_iterations=[500, 7000, 30000], cap_max=150000, eval=True)")
    d = parse_cfg_args(str(p))
    assert d["sh_degree"] == "3"
    assert d["source_path"] == "'/data/x'"
    assert d["save_iterations"] == "[500, 7000, 30000]"
    assert d["cap_max"] == "150000"


def test_cfg_diff_reports_only_differences(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    (tmp_path / "a" / "cfg_args").write_text("Namespace(cap_max=150000, noise=5e5, iterations=30000)")
    (tmp_path / "b" / "cfg_args").write_text("Namespace(cap_max=150000, noise=5e5, iterations=7000)")
    rows = cfg_diff(str(tmp_path / "a"), str(tmp_path / "b"), keys=["cap_max", "iterations"])
    assert rows == [("iterations", "30000", "7000")]


def test_emit_markdown_has_greppable_verdict(tmp_path):
    run = _make_run(tmp_path, "bad", [(7000, 18.7, 6000), (30000, 10.6, 5800)])
    md = emit_markdown([run])
    assert "DEGRADED: yes" in md
    assert "PSNR_DELTA_PEAK_TO_FINAL=" in md


@pytest.mark.parametrize(
    "missing_axis",
    (
        "trainer",
        "source_mode",
        "release_policy",
        "admission_target",
        "integration_policy",
        "optimization_core",
        "budget_policy",
        "pose_source",
    ),
)
def test_execution_semantic_differences_treat_each_missing_resolved_axis_as_unverifiable(
    tmp_path, missing_axis
):
    from scripts.compare_runs import main

    a = _make_run(tmp_path, "a", [(100, 10.0, 100)])
    b = _make_run(tmp_path, "b", [(100, 10.0, 100)])
    semantics = {
        "trainer": "streaming",
        "source_mode": "causal_replay",
        "release_policy": "iter_based",
        "admission_policy": "all",
        "admission_target": "soft_all",
        "integration_policy": "incremental_rgbd",
        "optimization_core": "gaussian_engine",
        "budget_policy": "fixed_iterations",
        "pose_source": "dataset_pose",
    }
    for run, resolved in ((a, semantics), (b, {**semantics})):
        report_path = next((tmp_path / run.split("/")[-1] / "comparison").glob("*/report.json"))
        report = json.loads(report_path.read_text())
        report["execution_semantics"] = resolved
        report_path.write_text(json.dumps(report))

    candidate_report = next((tmp_path / "b" / "comparison").glob("*/report.json"))
    candidate = json.loads(candidate_report.read_text())
    candidate["execution_semantics"].pop(missing_axis)
    candidate_report.write_text(json.dumps(candidate))

    out = tmp_path / "compare.md"
    with pytest.raises(ValueError, match=f"execution axis {missing_axis!r}"):
        main([a, b, "--out", str(out)])
    assert not out.exists()


def test_compare_runs_refuses_semantic_mismatch_unless_explicitly_allowed(tmp_path):
    from scripts.compare_runs import main

    a = _make_run(tmp_path, "a", [(100, 10.0, 100)])
    b = _make_run(tmp_path, "b", [(100, 10.0, 100)])
    for run, trainer in ((a, "offline"), (b, "streaming")):
        report_path = next((tmp_path / run.split("/")[-1] / "comparison").glob("*/report.json"))
        report = json.loads(report_path.read_text())
        report["execution_semantics"] = {
            "trainer": trainer,
            "source_mode": "materialized" if trainer == "offline" else "causal_replay",
            "release_policy": "all_at_start" if trainer == "offline" else "iter_based",
            "admission_target": "soft_all",
            "integration_policy": "bootstrap_only" if trainer == "offline" else "incremental_rgbd",
            "optimization_core": "trainer_inline_control" if trainer == "offline" else "gaussian_engine",
            "budget_policy": "fixed_iterations",
            "pose_source": "dataset_pose",
        }
        report_path.write_text(json.dumps(report))

    out = tmp_path / "compare.md"
    with pytest.raises(ValueError, match="execution semantics mismatch"):
        main([a, b, "--out", str(out)])
    assert not out.exists()

    assert main([a, b, "--out", str(out), "--allow-semantic-mismatch"]) == 0
    assert out.exists()


def test_compare_runs_refuses_unverifiable_semantics_unless_explicitly_allowed(tmp_path):
    from scripts.compare_runs import main

    a = _make_run(tmp_path, "a", [(100, 10.0, 100)])
    b = _make_run(tmp_path, "b", [(100, 10.0, 100)])
    out = tmp_path / "compare.md"

    with pytest.raises(ValueError, match="lacks execution_semantics"):
        main([a, b, "--out", str(out)])
    assert not out.exists()

    assert main([a, b, "--out", str(out), "--allow-semantic-mismatch"]) == 0
    assert out.exists()


def test_compare_runs_refuses_missing_resolved_admission_target_despite_same_raw_policy(
    tmp_path,
):
    from scripts.compare_runs import main

    a = _make_run(tmp_path, "a", [(100, 10.0, 100)])
    b = _make_run(tmp_path, "b", [(100, 10.0, 100)])
    semantics = {
        "trainer": "streaming",
        "source_mode": "causal_replay",
        "release_policy": "iter_based",
        "admission_policy": "all",
        "admission_target": "soft_all",
        "integration_policy": "incremental_rgbd",
        "optimization_core": "gaussian_engine",
        "budget_policy": "fixed_iterations",
        "pose_source": "dataset_pose",
    }
    for run, resolved in ((a, semantics), (b, {**semantics})):
        report_path = next((tmp_path / run.split("/")[-1] / "comparison").glob("*/report.json"))
        report = json.loads(report_path.read_text())
        report["execution_semantics"] = resolved
        report_path.write_text(json.dumps(report))

    candidate_report = next((tmp_path / "b" / "comparison").glob("*/report.json"))
    candidate = json.loads(candidate_report.read_text())
    candidate["execution_semantics"].pop("admission_target")
    candidate_report.write_text(json.dumps(candidate))
    out = tmp_path / "compare.md"

    with pytest.raises(ValueError, match="cannot verify execution axis 'admission_target'"):
        main([a, b, "--out", str(out)])
    assert not out.exists()

    assert main([a, b, "--out", str(out), "--allow-semantic-mismatch"]) == 0
    assert out.exists()
