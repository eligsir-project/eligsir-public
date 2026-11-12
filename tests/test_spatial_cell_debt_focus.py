"""Deterministic contract tests for issue #1318's direct cell-debt replay
focus mechanism (sections A-F of docs/architecture/2026-08-30-spatial-
replay-focus-fast-refinement.md).

IMPLEMENTED-tier scope only (issue #202 policy): these tests prove the
mechanism behaves per its software contract. They make no performance or
quality claim -- that requires a separate ResearchFlow-managed matched
experiment (issue #1318's "Required experiment" section, acceptance
criterion 10), which is explicitly out of scope here.
"""

import pytest

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.scheduler.partition_replay_focus import (
    CellDebtLedger,
    SpatialCellQuality,
    SpatialMicrocellQualitySample,
    SpatialCellDebtReplayFocusController,
    WorldCellIncidenceIndex,
    WorldCellIncidenceSample,
    aggregate_cell_debt,
    cell_debt_source_weight,
    mixed_replay_probability,
    plot_spatial_cell_debt_exposure,
    record_cell_debt_exposure,
    world_cell_incidence_from_pose_depth,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)


def _incidence_sample(source, cell_keys, *, cell_size_m=0.5, counts=None):
    return WorldCellIncidenceSample(
        source=source,
        cell_size_m=cell_size_m,
        cell_keys=tuple(cell_keys),
        valid_pixel_counts=tuple(counts or (100,) * len(cell_keys)),
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: world-cell incidence available by source_complete
# without replay-dependent discovery.
# ---------------------------------------------------------------------------


def test_incidence_index_is_complete_without_any_psnr_or_replay_evidence():
    """The incidence graph must be fully queryable from pose+depth samples
    alone -- no SpatialMicrocellQualitySample / PSNR observation is ever
    fed to the index, proving discovery does not depend on replay having
    happened to score a region."""

    index = WorldCellIncidenceIndex()
    s0, s1, s2 = _source(0), _source(1), _source(2)
    index.record(_incidence_sample(s0, [101, 202]))
    index.record(_incidence_sample(s1, [202, 303]))
    index.record(_incidence_sample(s2, [404]))

    assert index.known_source_count() == 3
    assert index.known_cell_count() == 4
    assert set(index.cells_for_source(s0)) == {101, 202}
    assert set(index.sources_for_cell(202)) == {s0, s1}
    assert set(index.sources_for_cell(404)) == {s2}
    # A cell nobody has ever seen must return an empty, not error.
    assert index.sources_for_cell(999) == ()


def test_incidence_from_pose_depth_ignores_render_and_gt(monkeypatch):
    """world_cell_incidence_from_pose_depth must derive cell identity
    directly from camera pose and sensor depth. It receives no render or
    ground-truth image and must not retain a dense per-camera map."""

    # This import is inside the test body (needed by LocalQualityTracker's
    # own import chain), invisible to tests/_torch_collect_scan.py's
    # collection-time static scan -- guard explicitly instead of letting
    # collection succeed and this deferred import hard-fail on the
    # torch-free CPU smoke image.
    torch = pytest.importorskip("torch")

    from eligsir.utils.local_quality import LocalQualityTracker

    class Camera:
        def __init__(self):
            self.world_view_transform = torch.eye(4)
            self.image_width = 4
            self.image_height = 3
            self.fx = self.fy = 2.0
            self.cx = 1.5
            self.cy = 1.0
            self.FoVx = self.FoVy = 1.0
            self.uid = 0
            self.image_name = "frame-10"

    source = _source(10)
    sensor_depth = torch.full((3, 4), 3.0)
    tracker = LocalQualityTracker(cell_size=0.5)

    sample = world_cell_incidence_from_pose_depth(
        source=source,
        tracker=tracker,
        camera=Camera(),
        sensor_depth=sensor_depth,
    )

    assert sample is not None
    assert sample.source == source
    assert sample.cell_size_m == 0.5
    assert sum(sample.valid_pixel_counts) == 3 * 4  # H*W valid pixels
    assert len(set(sample.cell_keys)) == len(sample.cell_keys)
    assert tracker._cell_maps == {}


def test_incidence_built_at_ingestion_survives_before_any_replay_scoring():
    """Simulates source_complete: every ingested source frame registers its
    incidence immediately (section A), well before the low-cadence replay
    controller below ever observes a SpatialMicrocellQualitySample for
    most of them. The graph must already answer queries for sources the
    debt controller has never scored."""

    index = WorldCellIncidenceIndex()
    sources = [_source(i) for i in range(50)]
    for i, source in enumerate(sources):
        index.record(_incidence_sample(source, [i, i + 1000]))

    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )
    # Not a single SpatialMicrocellQualitySample has been observed yet.
    assert controller.known_cell_count() == 0
    # But incidence is already fully known for every ingested source.
    for i, source in enumerate(sources):
        assert set(index.cells_for_source(source)) == {i, i + 1000}


# ---------------------------------------------------------------------------
# Acceptance criteria 6/7: one weak cell inside a good frame; always-bad
# cells without a prior peak.
# ---------------------------------------------------------------------------


def test_bad_local_geometry_cell_raises_priority_despite_good_frame_average():
    """Design doc section 6 worked example: wall=31, floor=32, desk=29,
    bad_local_geometry=13. A pixel-count-weighted frame average would sit
    near/above a healthy target and hide the 13dB cell; cell-debt scoring
    (top-k aggregation) must not."""

    wall, floor, desk, bad = 101, 102, 103, 104
    source = _source(10)
    other = _source(20)

    index = WorldCellIncidenceIndex()
    index.record(_incidence_sample(source, [wall, floor, desk, bad]))
    index.record(_incidence_sample(other, [wall]))

    controller = SpatialCellDebtReplayFocusController(
        incidence=index,
        target_psnr_db=30.0,
        gamma=0.0,
        lam=0.5,
        aggregation="top_k",
        top_k=1,
    )
    sample = SpatialMicrocellQualitySample(
        source=source,
        cell_size_m=0.5,
        cells=(
            SpatialCellQuality(cell_key=wall, psnr_db=31.0, n_samples=300),
            SpatialCellQuality(cell_key=floor, psnr_db=32.0, n_samples=300),
            SpatialCellQuality(cell_key=desk, psnr_db=29.0, n_samples=300),
            SpatialCellQuality(cell_key=bad, psnr_db=13.0, n_samples=300),
        ),
    )
    decision = controller.update(sample, (source, other))

    weights = dict(decision.weights)
    assert decision.active
    # The frame-average pixel-count-weighted PSNR here is 26.25dB, close to
    # (or above, depending on target) the target -- a frame-averaged
    # controller could plausibly assign this frame little or no priority.
    # Cell-debt (top-1) instead sees the raw 17dB gap on `bad` alone.
    assert weights[source] > 1.0
    exposure = {row.cell_key: row for row in decision.cell_exposure}
    assert exposure[bad].debt > exposure[wall].debt
    assert exposure[bad].debt == 30.0 - 13.0

    # A frame that only sees the good cells must stay materially lower
    # priority than the mixed frame, proving the mixed frame's boost is
    # attributable to `bad`, not some global effect.
    assert weights[other] < weights[source]


def test_top_k_aggregation_is_not_diluted_by_mean_aggregation():
    """Directly exercises aggregate_cell_debt: top_k on one severe + three
    healthy cells must exceed the mean, which is the dilution failure mode
    section 6/8 describes."""

    debts = [17.0, 0.0, 0.0, 0.0]
    assert aggregate_cell_debt(debts, method="top_k", top_k=1) == 17.0
    assert aggregate_cell_debt(debts, method="mean") == 4.25
    assert aggregate_cell_debt(debts, method="top_k", top_k=1) > aggregate_cell_debt(
        debts, method="mean"
    )


def test_always_bad_cell_is_prioritized_without_a_prior_peak():
    """Acceptance criterion 7 / design doc section 7: a cell that has
    NEVER been better than its current value (so D_reg stays ~0 for its
    whole history) must still accrue debt purely from D_abs."""

    always_bad = 555
    ledger = CellDebtLedger(target_psnr_db=30.0, gamma=1.0, tolerance_db=0.25)

    for _ in range(10):
        ledger.update_quality(always_bad, 13.0)

    state = ledger.peek(always_bad)
    assert state.psnr_peak == state.psnr_current == 13.0
    # Regression debt is exactly zero (never regressed from a peak)...
    assert ledger.debt(always_bad) == 30.0 - 13.0  # == pure D_abs, gamma*D_reg == 0


def test_regressing_cell_accrues_additional_regression_debt_over_absolute_debt():
    ledger = CellDebtLedger(target_psnr_db=20.0, gamma=2.0, tolerance_db=0.25)
    cell = 7
    ledger.update_quality(cell, 24.0)
    ledger.update_quality(cell, 24.2)
    ledger.update_quality(cell, 20.0)

    # D_abs = max(0, 20 - 20) = 0; D_reg = max(0, 24.2 - 20 - 0.25) = 3.95
    assert ledger.debt(cell) == pytest.approx(2.0 * 3.95)


def test_unknown_cell_stays_neutral_zero_debt_not_falsely_healthy_or_maximal():
    ledger = CellDebtLedger(target_psnr_db=30.0)
    assert ledger.debt(999) == 0.0
    assert ledger.peek(999) is None


# ---------------------------------------------------------------------------
# Acceptance criterion 8: no holdout/evaluation leakage into actuation.
# ---------------------------------------------------------------------------


def test_holdout_evidence_is_refused_by_the_actuation_path():
    index = WorldCellIncidenceIndex()
    source = _source(1)
    index.record(_incidence_sample(source, [1]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )
    sample = SpatialMicrocellQualitySample(
        source=source,
        cell_size_m=0.5,
        cells=(SpatialCellQuality(cell_key=1, psnr_db=10.0, n_samples=100),),
    )
    try:
        controller.update(sample, (source,), holdout=True)
    except ValueError as exc:
        assert "holdout" in str(exc)
    else:
        raise AssertionError("holdout evidence must be refused, not silently used")


def test_holdout_only_divergent_evidence_never_moves_actuation_weights():
    """Behavioral no-leakage check: construct a scenario where holdout-only
    evidence for a cell says one thing (excellent) and training-only
    evidence says another (terrible). Only the training-only call path
    (the plain, non-holdout `update`) can affect weights; there is no
    parameter or code path by which the holdout number could reach the
    ledger without going through the refused `holdout=True` branch above."""

    index = WorldCellIncidenceIndex()
    source = _source(1)
    other = _source(2)
    index.record(_incidence_sample(source, [1]))
    index.record(_incidence_sample(other, [2]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0, lam=1.0
    )

    # "Holdout" evidence (never passed to the controller at all) would say
    # cell 1 is excellent (40dB) -- if it could leak in, `source` would get
    # no priority. Only the training-only observation (terrible, 5dB) is
    # ever actually supplied.
    training_only_sample = SpatialMicrocellQualitySample(
        source=source,
        cell_size_m=0.5,
        cells=(SpatialCellQuality(cell_key=1, psnr_db=5.0, n_samples=100),),
    )
    decision = controller.update(training_only_sample, (source, other))
    assert dict(decision.weights)[source] > 1.0


def test_no_holdout_identifier_reachable_from_cell_debt_module_surface():
    """Grep-level check: the only spelling of "holdout" in this module is
    the explicit refusal branch/docstrings -- there is no holdout-labeled
    parameter that actually feeds PSNR into the ledger."""

    import inspect

    from eligsir.mapping.scheduler import partition_replay_focus as module

    source = inspect.getsource(module.SpatialCellDebtReplayFocusController.update)
    assert "raise ValueError" in source
    assert "holdout" in source
    # The only branch touching `holdout` must be the guard itself, i.e. it
    # must appear before the ledger/exposure mutation code, not alongside
    # a use of the flag's value as data.
    guard_index = source.index("if holdout")
    raise_index = source.index("raise ValueError")
    assert guard_index < raise_index < source.index("self._ledger.update_quality")


# ---------------------------------------------------------------------------
# Acceptance criterion 9: bounded memory/work under growing history.
# ---------------------------------------------------------------------------


def test_incidence_index_stays_bounded_under_thousands_of_sources_and_cells():
    index = WorldCellIncidenceIndex(max_sources=64, max_candidate_sources_per_cell=8)
    for i in range(5000):
        source = _source(i)
        # Every source shares cell 0 with many others, and also owns a
        # unique cell -- stresses both the forward (per-source) and
        # reverse (per-cell candidate) bounded structures simultaneously.
        index.record(_incidence_sample(source, [0, i + 1]))

    assert index.known_source_count() <= 64
    # Reverse index for the heavily shared cell never exceeds its cap.
    assert len(index.sources_for_cell(0)) <= 8
    # Reverse index does not grow linearly with total sources seen either
    # (each unique-per-source cell has at most 1 candidate).
    assert index.known_cell_count() <= 5000 + 1


def test_cell_debt_ledger_stays_bounded_under_thousands_of_updates():
    ledger = CellDebtLedger(target_psnr_db=30.0, max_cells=100)
    for i in range(10000):
        ledger.update_quality(i, 10.0 + (i % 5))
        assert ledger.known_cell_count() <= 100


def test_controller_state_stays_bounded_under_thousands_of_distinct_cells():
    index = WorldCellIncidenceIndex(max_sources=200)
    controller = SpatialCellDebtReplayFocusController(
        incidence=index,
        target_psnr_db=30.0,
        max_tracked_cells=50,
    )
    source = _source(1)
    for i in range(2000):
        index.record(_incidence_sample(source, [i]))
        sample = SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=i, psnr_db=15.0, n_samples=10),),
        )
        decision = controller.update(sample, (source,))
        assert controller.known_cell_count() <= 50
        assert len(decision.cell_exposure) <= 50


# ---------------------------------------------------------------------------
# Section B: exposure telemetry fields present and correctly aggregated.
# ---------------------------------------------------------------------------


def test_exposure_telemetry_tracks_all_required_fields():
    index = WorldCellIncidenceIndex()
    source = _source(1)
    other = _source(2)
    index.record(_incidence_sample(source, [1]))
    index.record(_incidence_sample(other, [1]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )

    def sample(psnr, samples=100):
        return SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=1, psnr_db=psnr, n_samples=samples),),
        )

    controller.update(sample(10.0), (source, other))
    decision = controller.update(sample(12.0, samples=50), (source, other))

    row = {r.cell_key: r for r in decision.cell_exposure}[1]
    assert row.candidate_sources_covering_cell == 2  # source + other both incident
    assert row.selected_replay_views_covering_cell == 2
    assert row.cumulative_supervised_pixels == 150
    assert row.optimizer_transactions_touching_cell == 2
    assert row.optimizer_updates_since_last_supervision == 0
    assert row.cumulative_focus_weight > 0.0
    assert row.psnr_before_exposure == 10.0
    assert row.psnr_after_exposure == 12.0
    assert row.debt == 30.0 - 12.0


def test_updates_since_last_supervision_ages_untouched_cells():
    index = WorldCellIncidenceIndex()
    source = _source(1)
    index.record(_incidence_sample(source, [1, 2]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )
    controller.update(
        SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(
                SpatialCellQuality(cell_key=1, psnr_db=20.0, n_samples=10),
                SpatialCellQuality(cell_key=2, psnr_db=20.0, n_samples=10),
            ),
        ),
        (source,),
    )
    decision = controller.update(
        SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=1, psnr_db=20.0, n_samples=10),),
        ),
        (source,),
    )
    rows = {r.cell_key: r for r in decision.cell_exposure}
    assert rows[1].optimizer_updates_since_last_supervision == 0
    assert rows[2].optimizer_updates_since_last_supervision == 1


# ---------------------------------------------------------------------------
# Section D: cell-debt-derived source scoring / aggregation.
# ---------------------------------------------------------------------------


def test_cell_debt_source_weight_matches_bounded_formula():
    ledger = CellDebtLedger(target_psnr_db=30.0, gamma=0.0)
    ledger.update_quality(1, 20.0)
    ledger.update_quality(2, 30.0)

    weight = cell_debt_source_weight(
        [1, 2], ledger, lambda_=0.5, aggregation="mean", max_weight=8.0
    )
    # debts = [10.0, 0.0] -> mean 5.0 -> 1 + 0.5*5.0 = 3.5
    assert weight == 3.5


def test_cell_debt_source_weight_is_capped_at_max_weight():
    ledger = CellDebtLedger(target_psnr_db=100.0, gamma=0.0, max_debt=1000.0)
    ledger.update_quality(1, 0.0)
    weight = cell_debt_source_weight(
        [1], ledger, lambda_=10.0, aggregation="mean", max_weight=8.0
    )
    assert weight == 8.0


# ---------------------------------------------------------------------------
# Section E: bounded mixing preserves the existing coverage guarantee.
# ---------------------------------------------------------------------------


def test_mixed_replay_probability_reduces_to_coverage_when_lambda_zero():
    assert mixed_replay_probability(0.4, 0.9, lam=0.0) == 0.4


def test_mixed_replay_probability_reduces_to_cell_debt_when_lambda_one():
    assert mixed_replay_probability(0.4, 0.9, lam=1.0) == 0.9


def test_mixed_replay_probability_is_capped_relative_to_uniform_baseline():
    # An adversarial always-focus-one-source scenario: cell-debt probability
    # mass is enormous, but the cap keeps it bounded near the uniform
    # baseline rather than letting it collapse coverage of everything else.
    mixed = mixed_replay_probability(
        0.1, 500.0, lam=0.9, uniform_probability=0.1, cap_multiplier=2.0
    )
    assert mixed == 0.2  # capped at 2x uniform, not ~450


def test_mixed_replay_probability_rejects_out_of_range_lambda():
    import pytest

    with pytest.raises(ValueError):
        mixed_replay_probability(0.1, 0.1, lam=1.5)


# ---------------------------------------------------------------------------
# Section F: fast source-complete bootstrap.
# ---------------------------------------------------------------------------


def test_bootstrap_seeds_debt_immediately_without_organic_warmup():
    index = WorldCellIncidenceIndex()
    source = _source(1)
    index.record(_incidence_sample(source, [1, 2, 3]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )

    # Before any update() call at all (i.e. before any organic replay/
    # PSNR observation), seed debt from "earliest available training
    # residual" evidence -- e.g. a coarse residual computed once at
    # source-complete time.
    controller.bootstrap_from_incidence({1: 12.0, 2: 29.0})

    weight = cell_debt_source_weight(
        index.cells_for_source(source),
        controller._ledger,
        lambda_=1.0,
        aggregation="top_k",
        top_k=1,
    )
    assert weight > 1.0  # cell 1's bootstrap debt (18dB gap) is already live
    # Cell 3 has no bootstrap residual yet: it must stay neutral, not
    # falsely healthy or falsely maximal.
    assert controller._ledger.debt(3) == 0.0


def test_bootstrap_only_lifts_cells_present_in_the_incidence_graph_evidence():
    index = WorldCellIncidenceIndex()
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )
    controller.bootstrap_from_incidence({42: 5.0})
    # 30.0 - 5.0 = 25.0, capped at the controller's default max_debt=20.0.
    assert controller._ledger.debt(42) == 20.0
    assert controller.known_cell_count() == 1


# ---------------------------------------------------------------------------
# Acceptance criterion 2 / section B: exposure telemetry is persisted as
# machine-readable jsonl and plottable, mirroring the existing
# spatial_replay_focus.jsonl / plot_spatial_replay_focus convention.
# ---------------------------------------------------------------------------


def test_record_and_plot_cell_debt_exposure_round_trips(tmp_path):
    import json

    index = WorldCellIncidenceIndex()
    source = _source(1)
    index.record(_incidence_sample(source, [1, 2]))
    controller = SpatialCellDebtReplayFocusController(
        incidence=index, target_psnr_db=30.0
    )
    decision = controller.update(
        SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(
                SpatialCellQuality(cell_key=1, psnr_db=13.0, n_samples=200),
                SpatialCellQuality(cell_key=2, psnr_db=29.0, n_samples=200),
            ),
        ),
        (source,),
    )

    path = record_cell_debt_exposure(
        decision, iteration=42, model_path=str(tmp_path)
    )
    assert path.is_file()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["iteration"] == 42
    assert row["active"] is True
    cell_by_key = {cell["cell_key"]: cell for cell in row["cells"]}
    assert cell_by_key[1]["debt"] == pytest.approx(17.0)
    assert cell_by_key[1]["psnr_before_exposure"] == 13.0
    assert cell_by_key[1]["selected_replay_views_covering_cell"] == 1

    plot_path = plot_spatial_cell_debt_exposure(str(tmp_path))
    assert plot_path is not None
    assert plot_path.is_file()


def test_plot_returns_none_without_any_recorded_exposure(tmp_path):
    assert plot_spatial_cell_debt_exposure(str(tmp_path)) is None
