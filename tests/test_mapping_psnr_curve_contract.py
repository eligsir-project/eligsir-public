"""Mapping quality recipes must produce observations, not endpoint-only plots."""

import json
from types import SimpleNamespace

import pytest


def test_work_curve_does_not_connect_final_report_metric_to_same_state(tmp_path):
    from eligsir.runtime.causal_evaluation import _load_paired_psnr_axes

    clock = {
        "iteration": 1115,
        "accumulated_iteration_wall_s": 67.523,
        "supervision_views": 27476,
    }
    rows = [
        {**clock, "evaluation_phase": phase, "mean_test_psnr_db": score}
        for phase, score in (
            ("mapping_endpoint", 19.5887),
            ("post_mapping_report", 19.5696),
        )
    ]
    path = tmp_path / "evaluation_ticks.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = path.read_bytes()
    points = _load_paired_psnr_axes(tmp_path)["holdout"]
    assert [point["psnr_db"] for point in points] == [19.5887]
    assert path.read_bytes() == original  # final metric remains durable, not deleted


def test_work_curve_refuses_mixed_history_reductions(tmp_path):
    from eligsir.runtime.causal_evaluation import _load_paired_psnr_axes

    rows = [
        {
            "iteration": index,
            "accumulated_iteration_wall_s": float(index),
            "supervision_views": index,
            "mean_test_psnr_db": 20.0,
            "psnr_reduction": reduction,
        }
        for index, reduction in enumerate(
            ("legacy_chw_channel_mean_v1", "rgb_per_view_v2"), start=1
        )
    ]
    (tmp_path / "evaluation_ticks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    with pytest.raises(ValueError, match="mixed PSNR reductions"):
        _load_paired_psnr_axes(tmp_path)


def test_tum_replay_enables_mapping_curves_without_changing_live_defaults():
    from eligsir.cli.config import load_cli_config
    from eligsir.cli.run_config import validate_config
    from eligsir.runtime.causal_evaluation import mapping_holdout_curve_metadata

    replay = validate_config(load_cli_config(run_profile="causal_replay/tum_rgbd"))
    live = validate_config(load_cli_config(run_profile="live/canonical_mapping"))
    assert replay["streaming_track_primary_psnr"] is True
    metadata = mapping_holdout_curve_metadata(SimpleNamespace(**replay))
    assert metadata["enabled"] and metadata["async_mode"]
    assert metadata["cadence_interval"] == 100
    assert not live.get("streaming_track_primary_psnr", False)
    assert not live.get("streaming_mapping_holdout_curve", False)
    # The observation mixin must not change the scene, score population, or mapper.
    mixin = load_cli_config(run_profile="evaluation/mapping_curves")
    assert set(mixin) - set(load_cli_config(run_profile="default")) == {
        "streaming_track_primary_psnr",
        "streaming_mapping_holdout_curve",
        "streaming_mapping_holdout_curve_async",
        "streaming_psnr_chart_interval",
    }


def test_observer_persists_mapping_and_refinement_phase_with_work_clocks(tmp_path):
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
    from eligsir.runtime.causal_replay import CausalReplayClock

    observer = object.__new__(CausalEvaluationObserver)
    observer.dataset = SimpleNamespace(model_path=tmp_path)
    observer._started_s = None
    observer._supervision_views = 0
    observer._mapping_active_s = None
    observer._elapsed_s = None
    observer._next_evaluation_optimizer_update = None
    for complete, step, score in ((False, 100, 15.0), (True, 200, 21.0)):
        observer._source_complete = complete
        observer.observe_tick(
            SimpleNamespace(
                iteration=step,
                primary_training_psnr_db=score,
                primary_training_source_id=f"frame-{step}",
            ),
            CausalReplayClock(
                optimizer_steps=step,
                supervision_views=step * 4,
                mapping_active_s=step / 10,
                elapsed_s=step / 8,
            ),
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "primary_training_psnr.jsonl").read_text().splitlines()
    ]
    assert [row["evaluation_phase"] for row in rows] == ["mapping", "refinement"]
    assert [row["mapping_active_s"] for row in rows] == [10.0, 20.0]
    assert [row["supervision_views"] for row in rows] == [400, 800]
    assert [row["primary_training_psnr_db"] for row in rows] == [15.0, 21.0]


def test_endpoint_only_plot_discloses_missing_mapping_samples(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from eligsir.runtime.causal_evaluation import plot_paired_psnr

    (tmp_path / "psnr_history.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1000,
                "mean_test_psnr": 20.0,
                "evaluation_phase": "mapping_endpoint",
            }
        )
        + "\n"
    )
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **k: captured.append(fig))
    plot_paired_psnr(tmp_path, last_ingest_iteration=1000)
    assert any(
        "No mid-mapping holdout samples" in text.get_text()
        for text in captured[0].axes[0].texts
    )
    # No fabricated/interpolated mapping values.
    assert list(captured[0].axes[0].lines[0].get_xdata()) == [1000]


@pytest.mark.parametrize("shape", [(3, 8, 16), (3, 4, 8), (3, 2, 4)])
def test_primary_sampler_reuses_realized_render_and_gt_at_successful_work_cadence(
    shape, monkeypatch
):
    torch = pytest.importorskip("torch")
    from eligsir.cli.config import load_cli_config
    from eligsir.runtime.production_application import CanonicalMappingRuntime
    import eligsir.runtime.production_application as application
    from eligsir.utils.score_contract import declare_camera_score_provenance

    config = load_cli_config(run_profile="causal_replay/tum_rgbd")
    runtime = object.__new__(CanonicalMappingRuntime)
    # Production __init__ installs Torch only after device selection. This
    # isolated method test does not construct a GPU mapping session.
    monkeypatch.setattr(application, "torch", torch)
    runtime._optimization_policy = SimpleNamespace(**config)
    runtime._partition_replay_focus_enabled = False
    runtime._primary_psnr_interval = 100
    runtime._next_primary_psnr_optimizer_update = 100
    camera = SimpleNamespace(original_image=torch.full(shape, 0.5))
    declare_camera_score_provenance(camera, context="mapping_curve_test")
    rendered = torch.full(shape, 0.25, requires_grad=True)
    result = SimpleNamespace(
        render_outputs=SimpleNamespace(payload={"render": rendered})
    )

    def forbidden_resize(*args, **kwargs):
        raise AssertionError("PSNR sampler must borrow the realized grid")

    monkeypatch.setattr(torch.nn.functional, "interpolate", forbidden_resize)
    runtime.optimizer_steps = 99
    assert runtime._sample_primary_training_psnr(result, camera) is None
    runtime.optimizer_steps = 100
    assert runtime._sample_primary_training_psnr(result, camera) == pytest.approx(
        12.0412, abs=1e-4
    )
    assert runtime._sample_primary_training_psnr(result, camera) is None
    assert rendered.grad is None
