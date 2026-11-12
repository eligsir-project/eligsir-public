from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.runtime import SourceIdentity
from eligsir.runtime.causal_replay import CausalReplayClock, CausalReplaySummary


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)


def _disabled_historical_usefulness_telemetry() -> dict[str, int]:
    """Match the canonical disabled challenger telemetry surface."""

    return {
        "observations_total": 0,
        "trusted_updates_total": 0,
        "feedback_busy": 0,
        "replay_challenger_trials_completed": 0,
        "replay_challenger_trials_authorized": 0,
        "replay_challenger_demotions": 0,
        "replay_challenger_turnover_blocked": 0,
        "replay_challenger_no_ready": 0,
        "ready_replay_evicted_bytes": 0,
    }


def _control_args(tmp_path, **overrides):
    values = {
        "source_path": "/data/fr3",
        "model_path": str(tmp_path),
        "streaming_mapping_holdout_curve": False,
        "streaming_psnr_chart_interval": 100,
        "streaming_report_test_metrics_max": 128,
        "streaming_partition_replay_focus": False,
        "streaming_spatial_replay_focus": False,
        "streaming_partition_focus_weight": 0.5,
        "streaming_partition_focus_signal_source": "control_probe",
        "streaming_partition_focus_control_probe_n": 12,
        "streaming_partition_focus_max_promote": 2,
        "streaming_partition_focus_outlier_k": 1.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_control_probe_entry_has_tick_namespace():
    import eligsir.runtime.causal_entry as causal_entry

    tick = causal_entry.SimpleNamespace(iteration=7)

    assert tick.iteration == 7


@pytest.mark.parametrize(
    ("prepared_insertion_skips", "expected_skip_total"),
    (({}, 0), ({"render_reuse_unavailable:missing_render_lease": 2}, 2)),
)
def test_run_causal_mapping_actuates_control_probe_over_bounded_cold_history(
    tmp_path, monkeypatch, prepared_insertion_skips, expected_skip_total
):
    """Removing canonical partition-control wiring must break this test.

    The slow GPU render is the only substituted boundary. Role assignment,
    weak-partition detection, bounded cold selection, weight publication, and
    durable trace generation all remain production implementations.
    """

    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.execution_semantics as execution_semantics

    retained = tuple(_source(index) for index in range(700))
    active_ordinals = (0, 1, 117, 118, 234, 235, 351, 352, 468, 469, 698, 699)
    cold_ordinals = (2, 3, 119, 120, 236, 237, 353, 354, 470, 471, 696, 697)
    bounded_candidates = tuple(
        retained[index] for index in sorted((*active_ordinals, *cold_ordinals))
    )
    active = [
        SimpleNamespace(uid=index, colmap_id=index, _source_identity=retained[index])
        for index in active_ordinals
    ]
    published: list[dict[SourceIdentity, float]] = []
    runtime = SimpleNamespace(
        initialized=True,
        scene=SimpleNamespace(train_cameras=active),
        gaussians=SimpleNamespace(get_xyz=torch.zeros(1)),
        pipe=SimpleNamespace(rasterize_mode="classic"),
        replay_control_candidate_identities=lambda: bounded_candidates,
        replay_catalog_population=lambda: len(retained),
        publish_replay_focus_weights=lambda weights: published.append(dict(weights)),
        historical_usefulness_telemetry=_disabled_historical_usefulness_telemetry,
    )
    pose = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    source = SimpleNamespace(
        frames=tuple(
            SimpleNamespace(index=index, source_ordinal=index, c2w=pose)
            for index in range(len(retained))
        )
    )
    binding = SimpleNamespace(
        mapper_sink=lambda _frame, _tracking: None,
        optimizer_tick=lambda: None,
        materialization_policy=None,
        materialization_backend=None,
        source_complete_hook=None,
        flush_hook=None,
        close_hook=None,
        observation_owner=None,
        control_snapshot_provider=None,
        disposition_sink=None,
        source_consumed_observer=None,
    )
    composition = SimpleNamespace(
        source=source,
        config=SimpleNamespace(hard_limits={"optimization_batch_size": 4}),
        binding=binding,
        runtime=runtime,
        prepared_ready_path=None,
    )
    monkeypatch.setattr(
        causal_entry,
        "resolve_causal_mapping_composition",
        lambda **_kwargs: composition,
    )
    monkeypatch.setattr(
        causal_evaluation,
        "build_causal_holdout_cameras",
        lambda _records, *, data_device: active,
    )

    def render_control_tick(_runtime, cameras):
        partition_scores = (10.0, 25.0, 27.0, 29.0, 31.0, 33.0)
        scores = [partition_scores[position // 2] for position, _ in enumerate(cameras)]
        return SimpleNamespace(
            cams=list(cameras),
            psnr_tensor=torch.tensor(scores, dtype=torch.float32),
            training_resolution_tiers=("eighth",) * len(cameras),
            evaluation_resolutions=((80, 60),) * len(cameras),
        )

    monkeypatch.setattr(
        causal_evaluation,
        "_render_partition_control_tick",
        render_control_tick,
        raising=False,
    )

    class Driver:
        def __init__(self, _application, _adapter, **kwargs):
            self.kwargs = kwargs

        def run(self, _source):
            tick = SimpleNamespace(
                iteration=350,
                optimizer_steps=350,
                supervision_views=1_400,
                cumulative_optimizer_updates=350,
                cumulative_supervision_views=1_400,
                primary_training_psnr_db=None,
                primary_training_spatial=None,
            )
            self.kwargs["tick_clock_observer"](
                tick,
                CausalReplayClock(
                    optimizer_steps=350,
                    supervision_views=1_400,
                    mapping_active_s=0.01,
                    elapsed_s=0.02,
                ),
            )
            self.kwargs["source_complete_observer"](tick)
            return CausalReplaySummary(
                submitted=0,
                ticks=1,
                source_identities=(),
                optimizer_steps=350,
                supervision_views=1_400,
                mapping_active_s=0.01,
                elapsed_s=0.02,
                prepared_insertion_skips=prepared_insertion_skips,
            )

    monkeypatch.setattr(causal_entry, "PacedCausalReplayDriver", Driver)
    monkeypatch.setattr(causal_entry, "write_causal_trajectory", lambda *_args: None)
    monkeypatch.setattr(
        causal_entry,
        "run_causal_post_mapping_refinement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )

    args = _control_args(
        tmp_path,
        streaming_input_fps_cap=30.0,
        causal_source_buffer_frames=16,
        causal_progress_interval_s=0.0,
        causal_decision_trace="",
        streaming_eval_hold=0,
        streaming_eval_guard=0,
        streaming_initial_frames=5,
        streaming_eval_min_baseline_m=0.03,
        streaming_eval_min_angle_deg=2.0,
        streaming_post_mapping_refinement_steps=0,
    )

    causal_entry.run_causal_mapping(
        dataset=SimpleNamespace(
            source_path="/data/fr3", model_path=str(tmp_path)
        ),
        opt=SimpleNamespace(),
        pipe=SimpleNamespace(),
        args=args,
    )

    assert len(published) == 1
    focused_ordinals = {
        source.source_ordinal for source, weight in published[0].items() if weight > 1.0
    }
    assert {0, 1, 2, 3}.issubset(focused_ordinals)
    assert len(published[0]) <= 32
    # Issue #1723 wiring: the requested focus probability must be realized over
    # the replay population the sampler actually draws from (700 here), not
    # over the 24-identity control shortlist. Uniform severities -> one weight.
    weights = list(published[0].values())
    n, population = len(weights), 700
    base_share = n / population
    target_share = base_share + 0.5 * (1.0 - base_share)
    realized_share = sum(weights) / (population - n + sum(weights))
    assert realized_share == pytest.approx(target_share, rel=1e-6)
    assert len(set(round(w, 9) for w in weights)) == 1
    published_weight = weights[0]

    trace_path = tmp_path / "partition_control_trace.jsonl"
    row = __import__("json").loads(trace_path.read_text().splitlines()[-1])
    expected_candidate_hash = (
        causal_evaluation._replay_candidate_shortlist_identity_hash(
            reversed(bounded_candidates)
        )
    )
    assert row["control_signal_source"] == "control_probe"
    assert row["control_probe_n"] == 12
    assert row["scored_holdout_n"] == 0
    assert row["control_scored_overlap_n"] == 0
    assert row["promote_indices"] == [2, 3]
    assert row["replay_catalog_population"] == 700
    assert row["replay_control_candidate_population"] == 24
    assert row["replay_control_candidate_identity_hash"] == expected_candidate_hash
    assert row["focus_probability_requested"] == 0.5
    assert row["focus_probability_resolved"] == 0.5
    assert row["focus_weight_min"] == pytest.approx(published_weight)
    assert row["focus_weight_max"] == pytest.approx(published_weight)
    assert row["score_scope"] == "control_probe_same_camera_grid_rgb_psnr"
    assert row["score_signal_source"] == "control_probe"
    assert row["control_resolution_tiers"] == ["eighth"] * 12
    assert row["control_evaluation_resolutions"] == [[80, 60]] * 12
    assert row["promotion_semantics"] == (
        "scheduler_focus_membership_not_immediate_residency"
    )
    causal_summary = __import__("json").loads(
        (tmp_path / "causal_replay_summary.json").read_text()
    )
    assert causal_summary["throughput_evidence_valid"] is False
    assert causal_summary["throughput_evidence_status"] == (
        "invalid_partition_control_observer_pause"
    )
    assert causal_summary["prepared_insertion_skip_total"] == expected_skip_total


def test_canonical_control_probe_is_disjoint_from_scored_holdout(
    tmp_path, monkeypatch
):
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        CausalHoldoutPlan,
    )
    import eligsir.runtime.causal_evaluation as causal_evaluation

    retained = tuple(_source(index) for index in range(700))
    active_ordinals = (0, 1, 117, 118, 234, 235, 351, 352, 468, 469, 698, 699)
    holdout_ordinals = frozenset((60, 180))
    active = [
        SimpleNamespace(uid=index, colmap_id=index, _source_identity=retained[index])
        for index in active_ordinals
    ]
    holdout = [SimpleNamespace(uid=index, colmap_id=index) for index in holdout_ordinals]
    rendered: list[int] = []
    published: list[dict[SourceIdentity, float]] = []
    runtime = SimpleNamespace(
        initialized=True,
        scene=SimpleNamespace(train_cameras=active),
        gaussians=SimpleNamespace(get_xyz=torch.zeros(1)),
        pipe=SimpleNamespace(rasterize_mode="classic"),
        replay_control_candidate_identities=lambda: tuple(
            retained[index] for index in active_ordinals
        ),
        replay_catalog_population=lambda: len(retained),
        publish_replay_focus_weights=lambda weights: published.append(dict(weights)),
    )
    plan = CausalHoldoutPlan(
        holdout_ordinals=holdout_ordinals,
        guard_ordinals=frozenset(),
        holdout_records=(),
        eval_metadata={"holdout_hash": "frozen-eval-only"},
    )
    observer = CausalEvaluationObserver(
        runtime=runtime,
        plan=plan,
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        args=_control_args(tmp_path),
        source_total=len(retained),
    )
    observer.test_cameras = holdout
    observer.eval_cameras = holdout
    observer.control_probe_cameras = active

    def render_control_tick(_runtime, cameras):
        rendered.extend(int(camera.uid) for camera in cameras)
        return SimpleNamespace(
            cams=list(cameras),
            psnr_tensor=torch.linspace(20.0, 31.0, len(cameras)),
            training_resolution_tiers=("eighth",) * len(cameras),
            evaluation_resolutions=((80, 60),) * len(cameras),
        )

    monkeypatch.setattr(
        causal_evaluation,
        "_render_partition_control_tick",
        render_control_tick,
    )
    observer.observe_tick(
        SimpleNamespace(
            iteration=100,
            optimizer_steps=100,
            supervision_views=400,
            cumulative_optimizer_updates=100,
            cumulative_supervision_views=400,
            primary_training_psnr_db=None,
            primary_training_spatial=None,
        ),
        CausalReplayClock(
            optimizer_steps=100,
            supervision_views=400,
            mapping_active_s=0.01,
            elapsed_s=0.02,
        ),
    )

    observer._source_complete = True
    tick = SimpleNamespace(iteration=100, optimizer_steps=0, supervision_views=0)
    measurement = observer._measure_partition_control(tick)

    assert measurement is not None
    assert published == []
    assert observer._partition_control_roles is None
    assert observer._partition_control_runtime.control_cameras == ()
    assert not (tmp_path / "partition_control_trace.jsonl").exists()

    observer._publish_partition_control(tick, measurement)
    assert set(rendered) == set(active_ordinals)
    assert set(rendered).isdisjoint(holdout_ordinals)
    assert all(source.source_ordinal not in holdout_ordinals for source in published[0])
    row = __import__("json").loads(
        (tmp_path / "partition_control_trace.jsonl").read_text().splitlines()[-1]
    )
    assert row["scored_holdout_n"] == 2
    assert row["control_scored_overlap_n"] == 0
    assert row["control_probe_hash"]
    assert row["scored_holdout_hash"]


def test_canonical_control_probe_prefers_reserved_trajectory_cohort(
    tmp_path, monkeypatch
):
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        CausalHoldoutPlan,
    )
    import eligsir.runtime.causal_evaluation as causal_evaluation

    retained = tuple(_source(index) for index in range(2_520))
    control_ordinals = (20, 480, 960, 1_440, 1_920, 2_500)
    control_cameras = [
        SimpleNamespace(uid=index, colmap_id=index, _source_identity=retained[index])
        for index in control_ordinals
    ]
    rendered: list[int] = []
    runtime = SimpleNamespace(
        initialized=True,
        scene=SimpleNamespace(train_cameras=[control_cameras[0]]),
        replay_control_candidate_identities=lambda: tuple(retained[index] for index in control_ordinals),
        replay_catalog_population=lambda: len(retained),
        publish_replay_focus_weights=lambda _weights: None,
    )
    observer = CausalEvaluationObserver(
        runtime=runtime,
        plan=CausalHoldoutPlan(
            holdout_ordinals=frozenset(),
            guard_ordinals=frozenset(control_ordinals),
            holdout_records=(),
            eval_metadata={},
            control_probe_records=(),
        ),
        dataset=SimpleNamespace(source_path="/data/fr3", model_path=str(tmp_path)),
        args=_control_args(tmp_path),
        source_total=len(retained),
    )
    observer.control_probe_cameras = control_cameras
    monkeypatch.setattr(
        causal_evaluation,
        "_render_partition_control_tick",
        lambda _runtime, cameras: (
            rendered.extend(int(camera.uid) for camera in cameras)
            or SimpleNamespace(
                cams=list(cameras),
                psnr_tensor=torch.linspace(15.0, 30.0, len(cameras)),
                training_resolution_tiers=("native",) * len(cameras),
                evaluation_resolutions=((80, 60),) * len(cameras),
            )
        ),
    )

    observer.observe_tick(
        SimpleNamespace(
            iteration=100,
            optimizer_steps=100,
            supervision_views=400,
            cumulative_optimizer_updates=100,
            cumulative_supervision_views=400,
            primary_training_psnr_db=None,
            primary_training_spatial=None,
        ),
        CausalReplayClock(
            optimizer_steps=100,
            supervision_views=400,
            mapping_active_s=0.01,
            elapsed_s=0.02,
        ),
    )

    assert not rendered
    observer.source_complete(
        SimpleNamespace(
            iteration=100,
            optimizer_steps=0,
            supervision_views=0,
        )
    )
    assert rendered == list(control_ordinals)
    row = __import__("json").loads(
        (tmp_path / "partition_control_trace.jsonl").read_text().splitlines()[-1]
    )
    assert row["control_probe_temporal_partition_count"] > 1
    # The non-oracle controller must optimize the same six-way temporal
    # quality objective emitted by the public partition report.  A finer
    # private binning can concentrate replay without improving the reported
    # worst partition, so it is not an acceptable control target.
    from eligsir.utils.partition_eval import PARTITIONS

    assert row["control_probe_trajectory_partition_count"] == PARTITIONS


@pytest.mark.parametrize(
    "signal_source",
    ("training_derived", "scored_holdout_oracle"),
)
def test_canonical_causal_mapping_requires_control_probe(
    tmp_path, signal_source
):
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        CausalHoldoutPlan,
    )

    with pytest.raises(ValueError, match="requires.*control_probe"):
        CausalEvaluationObserver(
            runtime=SimpleNamespace(),
            plan=CausalHoldoutPlan(
                holdout_ordinals=frozenset((10,)),
                guard_ordinals=frozenset(),
                holdout_records=(),
                eval_metadata={},
            ),
            dataset=SimpleNamespace(model_path=str(tmp_path)),
            args=_control_args(
                tmp_path,
                streaming_partition_focus_signal_source=signal_source,
            ),
            source_total=100,
        )


@pytest.mark.parametrize("probe_n", (0, 129))
def test_canonical_causal_mapping_rejects_unbounded_control_probe_count(
    tmp_path, probe_n
):
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        CausalHoldoutPlan,
    )

    with pytest.raises(ValueError, match=r"control_probe_n.*\[1, 128\]"):
        CausalEvaluationObserver(
            runtime=SimpleNamespace(),
            plan=CausalHoldoutPlan(
                holdout_ordinals=frozenset(),
                guard_ordinals=frozenset(),
                holdout_records=(),
                eval_metadata={},
            ),
            dataset=SimpleNamespace(model_path=str(tmp_path)),
            args=_control_args(
                tmp_path,
                streaming_partition_focus_control_probe_n=probe_n,
            ),
            source_total=100,
        )


def test_partition_replay_focus_requires_primary_psnr_tracking(tmp_path):
    """issue #1552 bug 2: without --streaming_track_primary_psnr, the
    controller is constructed but _observe_tick's `primary_psnr is not None`
    gate means it never receives a single observation -- no telemetry, no
    replay weight ever published, and no error to say so. Must fail closed
    at construction instead."""
    from eligsir.runtime.causal_evaluation import (
        CausalEvaluationObserver,
        CausalHoldoutPlan,
    )

    with pytest.raises(ValueError, match="streaming_track_primary_psnr"):
        CausalEvaluationObserver(
            runtime=SimpleNamespace(),
            plan=CausalHoldoutPlan(
                holdout_ordinals=frozenset(),
                guard_ordinals=frozenset(),
                holdout_records=(),
                eval_metadata={},
            ),
            dataset=SimpleNamespace(model_path=str(tmp_path)),
            args=_control_args(
                tmp_path,
                streaming_partition_replay_focus=True,
                streaming_partition_focus_weight=0.0,
            ),
            source_total=100,
        )


def test_replay_candidate_shortlist_hash_is_canonical_and_order_independent():
    from eligsir.runtime.causal_evaluation import (
        _replay_candidate_shortlist_identity_hash,
    )

    identities = tuple(_source(index) for index in (600, 1, 72, 511))
    expected = _replay_candidate_shortlist_identity_hash(identities)

    assert expected
    assert expected == _replay_candidate_shortlist_identity_hash(reversed(identities))
    assert expected != _replay_candidate_shortlist_identity_hash(identities[:-1])


def test_control_probe_scores_render_on_declared_camera_grid(monkeypatch):
    import eligsir.runtime.causal_evaluation as causal_evaluation
    from eligsir.contracts.control import ResolutionTier
    from eligsir.mapping.resolution_materialization import (
        materialize_resolution_tier,
    )
    from eligsir.utils.score_contract import (
        apply_declared_score_clamp_pair,
        declare_canonical_score_tensor,
        score_tensor_from_camera,
    )

    rgb, provenance = declare_canonical_score_tensor(
        torch.full((3, 480, 640), 0.5, dtype=torch.float32),
        context="test:control_probe_native",
    )
    camera = SimpleNamespace(
        uid=7,
        colmap_id=7,
        original_image=rgb,
        score_provenance=provenance,
        image_width=640,
        image_height=480,
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=319.5,
        cy=239.5,
        gsplat=None,
        _sensor_depth_cache=None,
    )
    materialize_resolution_tier(camera, ResolutionTier.EIGHTH)

    # The generic scorer must continue to reject shape-only inference. Only
    # the trusted renderer/camera boundary may attest that both sides use the
    # camera's declared score grid.
    with pytest.raises(ValueError, match="different declared score-grid"):
        apply_declared_score_clamp_pair(
            torch.full_like(camera.original_image, 0.4),
            score_tensor_from_camera(camera),
            context="test:generic_grid_mismatch",
        )

    def render(_camera, _gaussians, _pipe, _background):
        return {"render": torch.full_like(_camera.original_image, 0.4)}

    monkeypatch.setattr("gaussian_renderer.render", render)
    runtime = SimpleNamespace(
        gaussians=SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        pipe=SimpleNamespace(),
        _background=torch.zeros(3),
    )

    result = causal_evaluation._render_partition_control_tick(runtime, [camera])

    assert result.psnr_tensor.shape == (1,)
    assert torch.isfinite(result.psnr_tensor).all()
    assert result.training_resolution_tiers == ("eighth",)
    assert result.evaluation_resolutions == ((80, 60),)
    assert result.score_domain_clamp["per_view"][0]["score_provenance"][
        "resolution"
    ] == (60, 80)


def test_control_probe_materializes_lazy_camera_before_scoring(monkeypatch):
    import numpy as np

    import eligsir.runtime.causal_evaluation as causal_evaluation
    from scene.cameras import Camera
    from eligsir.utils.graphics_utils import focal2fov

    camera = Camera(
        colmap_id=9,
        R=np.eye(3),
        T=np.zeros(3),
        FoVx=focal2fov(2.0, 2),
        FoVy=focal2fov(2.0, 2),
        image=None,
        gt_alpha_mask=None,
        image_name="lazy-control",
        uid=9,
        data_device="cpu",
        fx=2.0,
        fy=2.0,
        cx=0.5,
        cy=0.5,
        width=2,
        height=2,
        pending_image_loader=lambda: (torch.full((3, 2, 2), 0.5), None),
    )
    monkeypatch.setattr(
        "gaussian_renderer.render",
        lambda *_args: {"render": torch.full((3, 2, 2), 0.4)},
    )
    runtime = SimpleNamespace(
        gaussians=SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        pipe=SimpleNamespace(),
        _background=torch.zeros(3),
    )

    result = causal_evaluation._render_partition_control_tick(runtime, [camera])

    assert camera.score_provenance is not None
    assert torch.isfinite(result.psnr_tensor).all()
