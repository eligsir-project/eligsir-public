from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.scheduler.partition_replay_focus import (
    PartitionReplayFocusController,
    SpatialCellQuality,
    SpatialMicrocellQualitySample,
    SpatialMicrocellReplayFocusController,
    TrainingPartitionPsnrAccumulator,
)


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000_000)


def _record(values, *, counts=None):
    return {
        "partition_k": len(values),
        "partition_denominator": 399,
        "partition_n": list(counts or (4,) * len(values)),
        "partition_psnr_db": list(values),
    }


def test_partition_focus_waits_for_three_valid_samples_then_biases_low_psnr():
    controller = PartitionReplayFocusController(min_history=3)
    sources = tuple(_source(index) for index in (10, 110, 210, 310))

    first = controller.update(_record((22.0, 20.0, 18.0, 22.0)), sources)
    second = controller.update(_record((22.0, 20.0, 18.0, 22.0)), sources)
    third = controller.update(_record((22.0, 20.0, 18.0, 22.0)), sources)

    assert not first.active and not second.active
    assert first.weights == tuple((source, 1.0) for source in sources)
    assert third.active
    weights = dict(third.weights)
    assert weights[_source(210)] > weights[_source(110)] > 1.0
    # 22.0 is the best-observed partition value (the target itself), so both
    # partitions tied at it stay neutral.
    assert weights[_source(10)] == weights[_source(310)] == 1.0


def test_partition_focus_biases_majority_weak_cluster_against_best_partition():
    """Regression test for the real 2026-08-31 Replica room0 finding.

    Four of six partitions sat within ~0.5dB of each other near 18dB while
    only two sat near 25-27dB (see the
    replica-partition-imbalance-fix-uniform-not-rebalancing memory). A
    median-of-measured target lands INSIDE that four-partition weak cluster
    (~18.4dB), so every weak partition's low_gap collapses toward zero and
    the controller applies almost no focus weight to the very partitions it
    exists to lift -- reproducing the observed "every partition gained
    ~uniformly, spread unchanged" real-run result. The target must instead
    be anchored to the best-observed partition, so a majority-weak
    distribution still yields a large, disproportionate weight for the weak
    group relative to the already-good partitions.
    """

    controller = PartitionReplayFocusController(min_history=3)
    sources = tuple(_source(index) for index in (33, 100, 166, 233, 299, 366))
    values = (24.45, 26.83, 18.41, 18.04, 18.37, 17.96)

    for _ in range(3):
        decision = controller.update(_record(values), sources)

    weights = dict(decision.weights)
    weak_indices = (2, 3, 4, 5)  # 18.41, 18.04, 18.37, 17.96
    best_index = 1  # 26.83, the best-observed partition == the target itself
    second_best_index = 0  # 24.45

    weak_weights = [weights[sources[i]] for i in weak_indices]
    best_weight = weights[sources[best_index]]
    second_best_weight = weights[sources[second_best_index]]

    assert decision.active
    # The best-observed partition is the target itself and stays neutral.
    assert best_weight == 1.0
    # Every weak partition must receive a real, substantial boost -- not the
    # ~1.0-1.2 a median-anchored target produces on this exact distribution
    # (median(measured) == 18.39dB here, landing INSIDE the weak cluster).
    assert all(weight > 4.0 for weight in weak_weights)
    # And the weak cluster must be clearly favored over both better-scoring
    # partitions, which is the entire point of "focus toward worst regions".
    assert min(weak_weights) > second_best_weight > best_weight


def test_partition_focus_adds_regression_debt_without_starving_global_replay():
    controller = PartitionReplayFocusController(
        min_history=3,
        low_psnr_gain=0.0,
        regression_gain=2.0,
        regression_tolerance_db=0.25,
    )
    sources = tuple(_source(index) for index in (10, 110, 210, 310))
    controller.update(_record((22.0, 22.0, 22.0, 22.0)), sources)
    controller.update(_record((22.1, 22.1, 22.1, 22.1)), sources)
    decision = controller.update(_record((22.2, 19.0, 22.2, 22.2)), sources)

    weights = dict(decision.weights)
    assert decision.active
    assert weights[_source(110)] > 1.0
    assert weights[_source(10)] == weights[_source(210)] == weights[_source(310)] == 1.0
    assert min(weights.values()) == 1.0


def test_partition_focus_fails_closed_for_sparse_or_malformed_evidence():
    controller = PartitionReplayFocusController(min_history=1, min_views_per_partition=2)
    sources = (_source(10), _source(210))

    sparse = controller.update(
        _record((10.0, 30.0), counts=(1, 4)), sources
    )
    malformed = controller.update(
        {
            "partition_k": 2,
            "partition_denominator": 399,
            "partition_n": [4],
            "partition_psnr_db": [10.0, 30.0],
        },
        sources,
    )

    assert dict(sparse.weights)[_source(10)] == 1.0
    assert malformed.active is False
    assert malformed.reason == "invalid_partition_evidence"


def test_partition_focus_bootstraps_an_underexposed_tail_before_psnr_support():
    """Successful-view debt must seed tail replay before PSNR samples exist."""
    controller = PartitionReplayFocusController(min_history=3)
    early, tail = _source(10), _source(390)
    record = _record((None,) * 6, counts=(0,) * 6)
    record["partition_supervision_n"] = [400, 400, 400, 400, 400, 5]

    decision = controller.update(record, (early, tail))

    weights = dict(decision.weights)
    assert decision.active
    assert weights[tail] > weights[early] == 1.0


def test_partition_focus_keeps_tail_coverage_debt_after_early_psnr_warmup():
    controller = PartitionReplayFocusController(min_history=1)
    early, tail = _source(10), _source(390)
    record = _record((22.0, 22.0, None, None, None, None), counts=(2, 2, 0, 0, 0, 0))
    record["partition_supervision_n"] = [400, 400, 400, 400, 400, 5]

    weights = dict(controller.update(record, (early, tail)).weights)

    assert weights[tail] > weights[early] == 1.0


def test_training_psnr_accumulator_partitions_only_realized_supervision_samples():
    accumulator = TrainingPartitionPsnrAccumulator(
        n_source_frames=400, window_per_partition=2
    )

    accumulator.observe(_source(10), 15.0)
    accumulator.observe(_source(10), 17.0)
    accumulator.observe(_source(210), 25.0)
    record = accumulator.observe(_source(210), 27.0)

    assert record["partition_k"] == 6
    assert record["partition_denominator"] == 399
    assert record["partition_n"] == [2, 0, 0, 2, 0, 0]
    assert record["partition_psnr_db"] == [16.0, None, None, 26.0, None, None]
    assert record["score_scope"] == "training_primary_pre_update"


def test_training_psnr_accumulator_records_successful_view_coverage_without_psnr():
    accumulator = TrainingPartitionPsnrAccumulator(n_source_frames=400)

    record = accumulator.record_successful_supervision(
        (_source(10), _source(10), _source(390))
    )

    assert record["partition_n"] == [0, 0, 0, 0, 0, 0]
    assert record["partition_supervision_n"] == [2, 0, 0, 0, 0, 1]
    assert record["partition_psnr_db"] == [None, None, None, None, None, None]


def test_training_psnr_accumulator_uses_report_partition_count_not_scaled_count():
    """Regression test for issue #1552 (16-way controller vs. 6-way report).

    On a long trajectory (e.g. Replica room0's 2000 source frames),
    ``utils.partition_eval.partition_count()`` scales up to 16 partitions --
    but ``report.json["partition_psnr"]``'s ``p0``..``p5`` always uses the
    fixed ``PARTITIONS=6`` scheme. Before this fix, the replay-focus
    accumulator bucketed by the scaled (16-way) count, so the controller's
    weight elevations could never even in principle be validated against, or
    correspond to, the report's per-partition PSNR -- see #1552 Bug 3, where
    the controller elevated weight on 16-way indices 0/1/8, none of which
    overlapped the report's worst partition (p5).
    """
    from eligsir.utils.partition_eval import PARTITIONS, partition_count

    n_source_frames = 2000
    # Sanity: this scene length really does trigger the trajectory-scaled
    # 16-way count elsewhere in the codebase -- otherwise this test would not
    # be exercising the mismatch it claims to guard against.
    assert partition_count(n_source_frames) == 16
    assert PARTITIONS == 6

    accumulator = TrainingPartitionPsnrAccumulator(n_source_frames=n_source_frames)
    record = accumulator.observe(_source(1), 20.0)

    assert record["partition_k"] == PARTITIONS
    assert record["partition_denominator"] == n_source_frames - 1


def test_partition_replay_focus_targets_report_scheme_worst_partition():
    """End-to-end reconciliation check: given a synthetic per-partition PSNR
    distribution with a known worst region (the report's p5, source frames
    in the top sixth of a 2000-frame trajectory), the controller -- now fed
    the SAME fixed 6-way scheme via TrainingPartitionPsnrAccumulator's
    default -- actually elevates weight for sources drawn from that region,
    not merely for whichever finer-grained bucket happened to be sampled.
    """
    from eligsir.utils.partition_eval import PARTITIONS, partition_id

    n_source_frames = 2000
    denominator = n_source_frames - 1
    accumulator = TrainingPartitionPsnrAccumulator(n_source_frames=n_source_frames)
    controller = PartitionReplayFocusController(min_history=3, min_views_per_partition=1)

    # One representative source ordinal per report partition (p0..p5), plus
    # a synthetic PSNR distribution where p5 (source frames near the tail,
    # e.g. 1900) is the clear worst -- mirroring the real #1370/#1552 finding
    # (worst partition at the trajectory tail).
    partition_reps = {
        0: 50,
        1: 400,
        2: 750,
        3: 1100,
        4: 1450,
        5: 1900,
    }
    psnr_by_partition = {0: 28.0, 1: 33.0, 2: 25.0, 3: 26.0, 4: 26.5, 5: 16.5}
    sources = tuple(_source(ordinal) for ordinal in partition_reps.values())

    # Confirm these representative ordinals really do land in the report's
    # own p0..p5 buckets under the fixed scheme -- the premise of this test.
    for pid, ordinal in partition_reps.items():
        assert partition_id(ordinal, denominator, k=PARTITIONS) == pid

    decision = None
    for _ in range(3):
        for pid, ordinal in partition_reps.items():
            record = accumulator.observe(_source(ordinal), psnr_by_partition[pid])
        decision = controller.update(record, sources)

    assert decision.active
    weights = dict(decision.weights)
    worst_source = _source(partition_reps[5])
    best_source = _source(partition_reps[1])
    assert weights[worst_source] > 1.0
    assert weights[worst_source] == max(weights.values())
    assert weights[worst_source] > weights[best_source]


def test_spatial_microcell_focus_biases_only_frames_observing_weak_cells():
    """Dropping ordinal binning must not turn unknown frames into focused ones."""

    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    weak = _source(10)
    strong = _source(110)
    unknown = _source(210)
    controller = SpatialMicrocellReplayFocusController(
        min_history=3,
        minimum_cell_samples=3,
    )

    for _ in range(3):
        controller.update(
            SpatialMicrocellQualitySample(
                source=weak,
                cell_size_m=0.5,
                cells=(SpatialCellQuality(cell_key=101, psnr_db=15.0, n_samples=300),),
            ),
            (weak, strong, unknown),
        )
        decision = controller.update(
            SpatialMicrocellQualitySample(
                source=strong,
                cell_size_m=0.5,
                cells=(SpatialCellQuality(cell_key=202, psnr_db=25.0, n_samples=300),),
            ),
            (weak, strong, unknown),
        )

    weights = dict(decision.weights)
    assert decision.active
    assert weights[weak] > weights[strong] == weights[unknown] == 1.0
    assert decision.cell_psnr_db == ((101, 15.0), (202, 25.0))
    assert decision.score_scope == "training_primary_spatial_pre_update"


def test_spatial_microcell_focus_uses_regression_debt_without_holdout_input():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    source = _source(10)
    other = _source(110)
    controller = SpatialMicrocellReplayFocusController(
        min_history=2,
        minimum_cell_samples=3,
        low_psnr_gain=0.0,
        regression_gain=2.0,
        regression_tolerance_db=0.25,
    )

    def sample(psnr):
        return SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=101, psnr_db=psnr, n_samples=300),),
        )

    controller.update(sample(24.0), (source, other))
    controller.update(sample(24.2), (source, other))
    decision = controller.update(sample(20.0), (source, other))

    assert decision.active
    assert dict(decision.weights)[source] > 1.0
    assert dict(decision.weights)[other] == 1.0
    assert decision.cell_drop_from_peak_db == ((101, 4.2),)


def test_spatial_regression_never_compares_different_camera_views_of_same_cell():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    bright_view = _source(10)
    dark_view = _source(90)
    controller = SpatialMicrocellReplayFocusController(
        min_history=1,
        minimum_cell_samples=3,
        low_psnr_gain=0.0,
        regression_gain=2.0,
    )

    controller.update(
        SpatialMicrocellQualitySample(
            source=bright_view,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(101, 30.0, 300),),
        ),
        (bright_view, dark_view),
    )
    decision = controller.update(
        SpatialMicrocellQualitySample(
            source=dark_view,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(101, 10.0, 300),),
        ),
        (bright_view, dark_view),
    )

    assert decision.active is False
    assert dict(decision.weights) == {bright_view: 1.0, dark_view: 1.0}
    assert decision.cell_drop_from_peak_db == ((101, 0.0),)


def test_cell_debt_actuation_off_by_default_matches_frame_averaged_mode():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialMicrocellReplayFocusController,
    )

    controller = SpatialMicrocellReplayFocusController(minimum_cell_samples=3)
    assert controller._cell_debt_actuation is False


def test_cell_debt_actuation_surfaces_a_bad_cell_hidden_by_frame_averaging():
    """Finding 3 (#1259): a catastrophically bad cell co-visible with good
    cells is diluted away by frame-level averaging before it ever raises
    replay priority. Cell-debt actuation must weight the frame using the
    bad cell's own debt, not the frame's averaged quality."""

    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    mixed = _source(10)
    other = _source(110)

    def mixed_sample():
        return SpatialMicrocellQualitySample(
            source=mixed,
            cell_size_m=0.5,
            cells=(
                SpatialCellQuality(cell_key=101, psnr_db=13.0, n_samples=300),
                SpatialCellQuality(cell_key=202, psnr_db=32.0, n_samples=300),
            ),
        )

    def other_sample():
        return SpatialMicrocellQualitySample(
            source=other,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=303, psnr_db=24.0, n_samples=300),),
        )

    def run(*, cell_debt_actuation: bool) -> float:
        controller = SpatialMicrocellReplayFocusController(
            min_history=1,
            minimum_cell_samples=3,
            low_psnr_gain=1.0,
            regression_gain=0.0,
            cell_debt_actuation=cell_debt_actuation,
        )
        controller.update(mixed_sample(), (mixed, other))
        decision = controller.update(other_sample(), (mixed, other))
        return dict(decision.weights)[mixed]

    frame_averaged_weight = run(cell_debt_actuation=False)
    cell_debt_weight = run(cell_debt_actuation=True)

    # Both modes must give the mixed frame SOME priority, but frame
    # averaging can only see the frame's blended quality (~22.5dB vs a
    # ~23.25dB target), while cell-debt sees the raw 13dB cell against a
    # ~24dB cell-level target -- a much larger, undiluted debt.
    assert cell_debt_weight > frame_averaged_weight > 1.0


def test_cell_debt_telemetry_tracks_cumulative_exposure_and_gain():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    source = _source(10)
    controller = SpatialMicrocellReplayFocusController(
        min_history=1, minimum_cell_samples=3
    )

    def sample(psnr):
        return SpatialMicrocellQualitySample(
            source=source,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=101, psnr_db=psnr, n_samples=50),),
        )

    controller.update(sample(12.0), (source,))
    decision = controller.update(sample(15.0), (source,))

    assert dict(decision.cell_cumulative_samples)[101] == 100
    assert dict(decision.cell_gain_since_first_observed_db)[101] == 3.0
    assert decision.actuation_mode == "frame_averaged"


def test_training_spatial_sample_uses_sensor_depth_cells_and_exact_rgb_error():
    import pytest
    import torch

    from eligsir.utils.local_quality import LocalQualityTracker
    from eligsir.mapping.scheduler.partition_replay_focus import (
        spatial_microcell_quality_from_render,
    )

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
    render = torch.zeros(3, 3, 4)
    gt = torch.full((3, 3, 4), 0.1)
    sensor_depth = torch.full((3, 4), 3.0)
    tracker = LocalQualityTracker(cell_size=0.5)

    sample = spatial_microcell_quality_from_render(
        source=source,
        camera=Camera(),
        tracker=tracker,
        render_image=render,
        gt_image=gt,
        sensor_depth=sensor_depth,
    )

    assert sample is not None
    assert sample.source == source
    assert sample.cell_size_m == 0.5
    assert sum(cell.n_samples for cell in sample.cells) == 3 * 3 * 4
    assert all(cell.psnr_db == pytest.approx(20.0) for cell in sample.cells)
    assert len({cell.cell_key for cell in sample.cells}) == len(sample.cells)
    # Training cameras are typically one-shot at this cadence. Retaining their
    # dense pixel maps makes device memory grow with every sampled source.
    assert tracker._cell_maps == {}


def test_training_spatial_samples_do_not_accumulate_cuda_allocations():
    import gc

    import pytest
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("needs a real GPU")
    from eligsir.utils.local_quality import LocalQualityTracker
    from eligsir.mapping.scheduler.partition_replay_focus import (
        spatial_microcell_quality_from_render,
    )

    class Camera:
        def __init__(self, ordinal):
            self.world_view_transform = torch.eye(4, device="cuda")
            self.image_width, self.image_height = 320, 240
            self.fx = self.fy = 320.0
            self.cx, self.cy = 159.5, 119.5
            self.FoVx = self.FoVy = 1.0
            self.uid = ordinal
            self.image_name = f"frame-{ordinal}"

    render = torch.zeros(3, 240, 320, device="cuda")
    gt = torch.full_like(render, 0.1)
    depth = torch.full((240, 320), 3.0, device="cuda")
    tracker = LocalQualityTracker(cell_size=0.5)
    # Exclude one-time ROCm kernel/workspace initialization; the regression is
    # growth per additional source after the operation has been warmed up.
    sample = spatial_microcell_quality_from_render(
        source=_source(0),
        camera=Camera(0),
        tracker=tracker,
        render_image=render,
        gt_image=gt,
        sensor_depth=depth,
    )
    del sample
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    for ordinal in range(1, 25):
        sample = spatial_microcell_quality_from_render(
            source=_source(ordinal),
            camera=Camera(ordinal),
            tracker=tracker,
            render_image=render,
            gt_image=gt,
            sensor_depth=depth,
        )
        assert sample is not None

    del sample
    gc.collect()
    torch.cuda.synchronize()
    assert tracker._cell_maps == {}
    assert torch.cuda.memory_allocated() - before < 8 * 1024 * 1024


def test_active_source_cell_keys_covers_only_sources_with_a_live_sample():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    sourced = _source(10)
    unseen = _source(20)
    controller = SpatialMicrocellReplayFocusController(min_history=1, minimum_cell_samples=1)

    controller.update(
        SpatialMicrocellQualitySample(
            source=sourced,
            cell_size_m=0.5,
            cells=(
                SpatialCellQuality(cell_key=1, psnr_db=20.0, n_samples=10),
                SpatialCellQuality(cell_key=2, psnr_db=21.0, n_samples=10),
            ),
        ),
        (sourced, unseen),
    )

    coverage = controller.active_source_cell_keys()
    assert coverage == {sourced: (1, 2)}
    assert unseen not in coverage


def test_current_cell_quality_matches_updates_own_internal_reduction():
    from eligsir.mapping.scheduler.partition_replay_focus import (
        SpatialCellQuality,
        SpatialMicrocellQualitySample,
        SpatialMicrocellReplayFocusController,
    )

    a = _source(10)
    b = _source(20)
    controller = SpatialMicrocellReplayFocusController(min_history=1, minimum_cell_samples=1)

    controller.update(
        SpatialMicrocellQualitySample(
            source=a,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=1, psnr_db=15.0, n_samples=10),),
        ),
        (a, b),
    )
    decision = controller.update(
        SpatialMicrocellQualitySample(
            source=b,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=1, psnr_db=25.0, n_samples=10),),
        ),
        (a, b),
    )

    cell_psnr, cell_counts = controller.current_cell_quality()
    assert cell_psnr == dict(decision.cell_psnr_db)
    assert cell_counts[1] == 2

    # Read-only: calling it again (with no new sample) must not change state.
    cell_psnr_again, cell_counts_again = controller.current_cell_quality()
    assert cell_psnr_again == cell_psnr
    assert cell_counts_again == cell_counts


def test_spatial_microcell_focus_prunes_state_for_retired_sources():
    """A source that leaves the replay-candidate pool must not keep biasing
    the running quality target or leak memory forever.

    Unlike the sibling PartitionReplayFocusController (whose peak/debt state
    is a fixed-size list indexed by partition id, inherently bounded), this
    controller's state is keyed by SourceIdentity -- one entry per unique
    frame ever observed. Without pruning, quality_target = median(self.
    _source_quality.values()) stays biased by every stale, long-retired
    source forever, and _histories/_peaks/_source_cells/_pair_regression_
    debt/_pair_drops grow without bound over a long run.
    """

    retired = _source(10)
    active = _source(20)
    newer = _source(30)
    controller = SpatialMicrocellReplayFocusController(
        min_history=2, minimum_cell_samples=3
    )

    def sample(src, key, psnr):
        return SpatialMicrocellQualitySample(
            source=src,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=key, psnr_db=psnr, n_samples=300),),
        )

    # `retired` establishes real history while still a valid candidate.
    controller.update(sample(retired, 101, 5.0), (retired, active))
    controller.update(sample(retired, 101, 5.0), (retired, active))
    assert retired in controller._source_quality
    assert (retired, 101) in controller._histories

    # `retired` leaves the candidate pool (evicted from replay eligibility);
    # a later tick observes only `active`/`newer`.
    controller.update(sample(active, 202, 20.0), (active, newer))

    assert retired not in controller._source_quality
    assert retired not in controller._source_cells
    assert not any(pair[0] == retired for pair in controller._histories)
    assert not any(pair[0] == retired for pair in controller._peaks)
    assert not any(pair[0] == retired for pair in controller._pair_regression_debt)
    assert not any(pair[0] == retired for pair in controller._pair_drops)


def test_spatial_microcell_focus_does_not_prune_cell_keyed_telemetry():
    """Cell-keyed telemetry (a fixed spatial grid) must keep accumulating
    across the whole run -- only per-SourceIdentity state is pruned."""

    retired = _source(10)
    active = _source(20)
    controller = SpatialMicrocellReplayFocusController(
        min_history=2, minimum_cell_samples=3
    )

    controller.update(
        SpatialMicrocellQualitySample(
            source=retired,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=101, psnr_db=5.0, n_samples=300),),
        ),
        (retired, active),
    )
    assert 101 in controller._cell_cumulative_samples
    cumulative_before = controller._cell_cumulative_samples[101]

    # `retired` leaves the pool, but cell 101's cumulative telemetry must
    # survive the prune -- it is keyed by the fixed spatial grid, not by
    # the now-retired frame identity that most recently observed it.
    controller.update(
        SpatialMicrocellQualitySample(
            source=active,
            cell_size_m=0.5,
            cells=(SpatialCellQuality(cell_key=202, psnr_db=20.0, n_samples=300),),
        ),
        (active,),
    )
    assert controller._cell_cumulative_samples[101] == cumulative_before
