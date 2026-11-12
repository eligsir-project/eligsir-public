from __future__ import annotations

import json

import pytest
import torch

from eligsir.cli.parser import build_training_parser, parse_training_args
from eligsir.utils.partition_control import (
    ControlSignalSource,
    EvaluationLeakageError,
    PartitionControlRecorder,
    PartitionControlRuntime,
    assign_partition_control_roles,
    decide_partition_focus,
    decide_spatial_partition_focus,
)


class _Frame:
    def __init__(self, index: int) -> None:
        self.index = index


class _Camera:
    def __init__(self, index: int) -> None:
        self._streaming_frame = _Frame(index)
        self.image_name = f"frame_{index}"


class _UnidentifiedCamera:
    pass


class _Tick:
    def __init__(self, cameras, values) -> None:
        self.cams = cameras
        self.psnr_tensor = torch.tensor(values, dtype=torch.float32)


TRAINING_INDICES = (0, 1, 20, 21, 40, 41, 60, 61)
SCORED_INDICES = (8, 9, 28, 29, 48, 49, 68, 69)


def _scores(low_partition: int) -> dict[int, float]:
    values = {}
    for partition, indices in enumerate(((0, 1), (20, 21), (40, 41), (60, 61))):
        score = 10.0 if partition == low_partition else (30.0, 31.0, 32.0, 50.0)[partition]
        values.update({index: score for index in indices})
    return values


def _scored_scores(low_partition: int) -> dict[int, float]:
    values = {}
    for partition, indices in enumerate(((8, 9), (28, 29), (48, 49), (68, 69))):
        score = 10.0 if partition == low_partition else (30.0, 31.0, 32.0, 50.0)[partition]
        values.update({index: score for index in indices})
    return values


def test_cli_defaults_partition_control_to_non_oracle_training_signal() -> None:
    args, _bundle = parse_training_args(["-s", "scene", "-m", "output/test"])
    assert args.streaming_partition_focus_signal_source == "training_derived"
    assert args.streaming_partition_focus_control_probe_n == 32
    assert args.streaming_partition_focus_weight == 0.0


def test_cli_requires_explicit_scored_holdout_oracle_choice() -> None:
    args, _bundle = parse_training_args([
        "-s", "scene", "-m", "output/test",
        "--streaming_partition_focus_signal_source", "scored_holdout_oracle",
    ])
    assert args.streaming_partition_focus_signal_source == "scored_holdout_oracle"


def test_cli_help_distinguishes_canonical_and_legacy_partition_control() -> None:
    parser = build_training_parser().parser
    actions = {action.dest: action for action in parser._actions}

    source_help = actions["streaming_partition_focus_signal_source"].help
    probe_help = actions["streaming_partition_focus_control_probe_n"].help
    assert "canonical causal" in source_help.lower()
    assert "requires control_probe" in source_help
    assert "legacy/research only" in source_help
    assert "[1, 128]" in probe_help
    assert "0/full-pool" in probe_help
    assert "legacy/research only" in probe_help


def test_control_probe_assignment_is_deterministic_and_hashed() -> None:
    first = assign_partition_control_roles(
        source=ControlSignalSource.CONTROL_PROBE,
        training_indices=TRAINING_INDICES,
        scored_holdout_indices=SCORED_INDICES,
        control_probe_n=4,
        controller_enabled=True,
    )
    reordered = assign_partition_control_roles(
        source="control_probe",
        training_indices=reversed(TRAINING_INDICES),
        scored_holdout_indices=reversed(SCORED_INDICES),
        control_probe_n=4,
        controller_enabled=True,
    )

    assert first.control_indices == (0, 20, 40, 61)
    assert first == reordered
    assert first.control_probe_hash == "9e49a63351076e99"
    assert first.control_probe_n == 4
    assert first.control_scored_overlap_n == 0


def test_control_probe_runtime_refreshes_probe_as_training_history_grows(
    tmp_path,
) -> None:
    runtime = PartitionControlRuntime(
        source="control_probe", control_probe_n=4, model_path=tmp_path
    )
    training = [_Camera(index) for index in TRAINING_INDICES]
    scored = [_Camera(index) for index in SCORED_INDICES]

    first_roles, first_cameras = runtime.prepare(training, scored)
    reordered_roles, reordered_cameras = runtime.prepare(
        list(reversed(training)), scored
    )
    growing = training + [_Camera(index) for index in (80, 100)]
    later_roles, later_cameras = runtime.prepare(growing, scored)

    assert first_roles.control_indices == reordered_roles.control_indices
    assert first_roles.control_probe_hash == reordered_roles.control_probe_hash
    assert [cam._streaming_frame.index for cam in first_cameras] == [0, 20, 40, 61]
    assert [cam._streaming_frame.index for cam in reordered_cameras] == [0, 20, 40, 61]
    assert later_roles.control_indices == (0, 21, 60, 100)
    assert [cam._streaming_frame.index for cam in later_cameras] == [0, 21, 60, 100]
    assert later_roles.control_probe_hash != first_roles.control_probe_hash


def test_non_oracle_runtime_renders_separate_control_tick_and_does_not_route_scored(
    tmp_path,
) -> None:
    training = [_Camera(index) for index in TRAINING_INDICES]
    scored = [_Camera(index) for index in SCORED_INDICES]
    scored_tick = _Tick(scored, [99.0] * len(scored))
    rendered = []

    runtime = PartitionControlRuntime(
        source="training_derived", control_probe_n=4, model_path=tmp_path
    )
    roles, observations = runtime.prepare_tick(
        training_cameras=training,
        scored_holdout_cameras=scored,
        scored_holdout_tick=scored_tick,
        render_control_tick=lambda cams: (
            rendered.extend(cams) or _Tick(cams, [10.0] * len(cams))
        ),
    )

    assert roles.oracle_control_enabled is False
    assert rendered == training
    assert observations["training_derived"] == {
        index: 10.0 for index in TRAINING_INDICES
    }
    assert observations["scored_holdout"] == {
        index: 99.0 for index in SCORED_INDICES
    }


def test_oracle_runtime_reuses_scored_tick_without_extra_control_render(tmp_path) -> None:
    training = [_Camera(index) for index in TRAINING_INDICES]
    scored = [_Camera(index) for index in SCORED_INDICES]
    scored_tick = _Tick(scored, [12.0] * len(scored))

    runtime = PartitionControlRuntime(
        source="scored_holdout_oracle", control_probe_n=4, model_path=tmp_path
    )
    roles, observations = runtime.prepare_tick(
        training_cameras=training,
        scored_holdout_cameras=scored,
        scored_holdout_tick=scored_tick,
        render_control_tick=lambda _cams: pytest.fail("oracle must reuse scored tick"),
    )

    assert roles.oracle_control_enabled is True
    assert observations["scored_holdout"] == {
        index: 12.0 for index in SCORED_INDICES
    }


def test_non_oracle_runtime_never_falls_back_to_scored_tick(tmp_path) -> None:
    training = [_Camera(index) for index in TRAINING_INDICES]
    scored = [_Camera(index) for index in SCORED_INDICES]
    runtime = PartitionControlRuntime(
        source="training_derived", control_probe_n=4, model_path=tmp_path
    )

    roles, observations = runtime.prepare_tick(
        training_cameras=training,
        scored_holdout_cameras=scored,
        scored_holdout_tick=_Tick(scored, [99.0] * len(scored)),
        render_control_tick=lambda _cams: None,
    )

    assert observations["training_derived"] == {}
    with pytest.raises(ValueError, match="control signal incomplete"):
        decide_partition_focus(
            roles=roles,
            training_observations=observations["training_derived"],
            control_probe_observations=observations["control_probe"],
            scored_holdout_observations=observations["scored_holdout"],
            max_source_index=79,
            partition_count=4,
            outlier_k=1.0,
            active_training_indices=TRAINING_INDICES,
            cached_indices=(),
            max_promote=0,
        )


def test_disabled_runtime_never_fails_report_role_preparation(tmp_path) -> None:
    runtime = PartitionControlRuntime(
        source="training_derived",
        control_probe_n=4,
        model_path=tmp_path,
        controller_enabled=False,
    )

    roles, control_cameras = runtime.prepare(
        [_UnidentifiedCamera(), _UnidentifiedCamera()],
        [_UnidentifiedCamera()],
    )

    assert control_cameras == []
    assert roles.scientific_eval_status == "CONTROL_DISABLED"
    assert roles.to_report_dict()["control_mode_label"] == "CONTROL_DISABLED"
    assert roles.control_signal_n == 0


@pytest.mark.parametrize("source", ["training_derived", "control_probe"])
def test_non_oracle_role_assignment_refuses_scored_overlap(source: str) -> None:
    with pytest.raises(EvaluationLeakageError, match="INVALID_EVALUATION_LEAKAGE"):
        assign_partition_control_roles(
            source=source,
            training_indices=(0, 20, 40, 60),
            scored_holdout_indices=(20, 80),
            control_probe_n=4,
            controller_enabled=True,
        )


def test_scored_holdout_perturbation_cannot_change_non_oracle_action_trace() -> None:
    roles = assign_partition_control_roles(
        source="training_derived",
        training_indices=TRAINING_INDICES,
        scored_holdout_indices=SCORED_INDICES,
        controller_enabled=True,
    )
    common = dict(
        roles=roles,
        training_observations=_scores(low_partition=1),
        control_probe_observations={},
        max_source_index=79,
        partition_count=4,
        outlier_k=1.0,
        active_training_indices=TRAINING_INDICES,
        cached_indices=(22, 23),
        max_promote=2,
    )

    first = decide_partition_focus(
        **common, scored_holdout_observations=_scored_scores(low_partition=0)
    )
    perturbed = decide_partition_focus(
        **common, scored_holdout_observations=_scored_scores(low_partition=3)
    )

    assert first.action_trace() == perturbed.action_trace()
    assert first.bad_partitions == (1,)
    assert set(first.priority_weights) == {20, 21, 22, 23}
    assert len(set(first.priority_weights.values())) == 1
    assert next(iter(first.priority_weights.values())) > 1.0
    assert first.promote_indices == (22, 23)
    assert first.demote_indices == (60, 61)


def test_control_probe_perturbation_changes_non_oracle_action_trace() -> None:
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=TRAINING_INDICES,
        scored_holdout_indices=SCORED_INDICES,
        control_probe_n=8,
        controller_enabled=True,
    )
    common = dict(
        roles=roles,
        training_observations={},
        scored_holdout_observations=_scored_scores(low_partition=0),
        max_source_index=79,
        partition_count=4,
        outlier_k=1.0,
        active_training_indices=TRAINING_INDICES,
        cached_indices=(),
        max_promote=0,
    )

    first = decide_partition_focus(
        **common, control_probe_observations=_scores(low_partition=1)
    )
    perturbed = decide_partition_focus(
        **common, control_probe_observations=_scores(low_partition=2)
    )

    assert first.action_trace() != perturbed.action_trace()
    assert first.bad_partitions == (1,)
    assert perturbed.bad_partitions == (2,)


def test_control_probe_targets_broad_weak_cluster_against_best_partition() -> None:
    """A median/MAD outlier test alone misses several jointly weak regions."""

    training = tuple(
        index
        for partition in range(6)
        for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )
    scores = {
        0: 18.9,
        1: 14.6,
        2: 13.6,
        3: 15.1,
        4: 19.5,
        5: 20.0,
    }
    observations = {
        index: scores[index // 100]
        for index in training
    }

    decision = decide_partition_focus(
        roles=roles,
        training_observations={},
        control_probe_observations=observations,
        scored_holdout_observations={},
        max_source_index=599,
        partition_count=6,
        outlier_k=1.5,
        active_training_indices=training,
        cached_indices=(102, 103, 202, 203, 302, 303),
        max_promote=6,
    )

    assert decision.bad_partitions == (1, 2, 3)
    assert set(decision.promote_indices).issubset({102, 103, 202, 203, 302, 303})
    assert decision.priority_weights[200] > decision.priority_weights[100]


def test_historical_repair_spends_a_single_slot_on_the_weakest_target_partition() -> None:
    """A broad target set must not let source ordering hide its weakest member."""

    training = tuple(
        index
        for partition in range(6)
        for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )
    scores = {0: 18.9, 1: 14.6, 2: 13.6, 3: 15.1, 4: 19.5, 5: 20.0}
    observations = {index: scores[index // 100] for index in training}

    decision = decide_partition_focus(
        roles=roles,
        training_observations={},
        control_probe_observations=observations,
        scored_holdout_observations={},
        max_source_index=599,
        partition_count=6,
        outlier_k=1.5,
        active_training_indices=training,
        cached_indices=(102, 202, 302),
        max_promote=1,
    )

    assert decision.bad_partitions == (1, 2, 3)
    assert decision.promote_indices == (202,)


def test_oracle_uses_scored_signal_and_is_loudly_labeled() -> None:
    roles = assign_partition_control_roles(
        source="scored_holdout_oracle",
        training_indices=TRAINING_INDICES,
        scored_holdout_indices=SCORED_INDICES,
        controller_enabled=True,
    )
    common = dict(
        roles=roles,
        training_observations=_scores(low_partition=1),
        control_probe_observations={},
        max_source_index=79,
        partition_count=4,
        outlier_k=1.0,
        active_training_indices=TRAINING_INDICES,
        cached_indices=(),
        max_promote=0,
    )

    first = decide_partition_focus(
        **common, scored_holdout_observations=_scored_scores(low_partition=0)
    )
    perturbed = decide_partition_focus(
        **common, scored_holdout_observations=_scored_scores(low_partition=3)
    )
    report = roles.to_report_dict()

    assert first.action_trace() != perturbed.action_trace()
    assert report["control_signal_source"] == "scored_holdout_oracle"
    assert report["oracle_control_enabled"] is True
    assert report["control_mode_label"] == "ORACLE_CONTROL"
    assert report["scientific_eval_status"] == "ORACLE_INTERVENTION"
    assert report["control_scored_overlap_n"] == len(SCORED_INDICES)


def test_recorder_stamps_jsonl_and_report_with_action_trace_hash(tmp_path) -> None:
    roles = assign_partition_control_roles(
        source="training_derived",
        training_indices=TRAINING_INDICES,
        scored_holdout_indices=SCORED_INDICES,
        controller_enabled=True,
    )
    decision = decide_partition_focus(
        roles=roles,
        training_observations=_scores(low_partition=1),
        control_probe_observations={},
        scored_holdout_observations=_scored_scores(low_partition=3),
        max_source_index=79,
        partition_count=4,
        outlier_k=1.0,
        active_training_indices=TRAINING_INDICES,
        cached_indices=(22,),
        max_promote=1,
    )
    recorder = PartitionControlRecorder(tmp_path)

    recorder.record(iteration=1000, roles=roles, decision=decision)

    record = json.loads((tmp_path / "partition_control_trace.jsonl").read_text())
    report = recorder.report_block(roles)
    assert record["control_signal_source"] == "training_derived"
    assert record["scientific_eval_status"] == "VALID_NON_ORACLE_CONTROL"
    assert record["iteration"] == 1000
    assert record["bad_partitions"] == [1]
    assert report["n_control_decisions"] == 1
    assert len(report["action_trace_hash"]) == 16
    assert report["action_trace_artifact"] == "partition_control_trace.jsonl"


# --- decide_spatial_partition_focus (#1723: cell-substrate actuation) ---


def _spatial_roles():
    return assign_partition_control_roles(
        source="training_derived",
        training_indices=tuple(range(20)),
        scored_holdout_indices=(),
        controller_enabled=True,
    )


def test_spatial_focus_promotes_a_cached_frame_covering_the_weak_cell() -> None:
    cell_psnr = {0: 20.0, 1: 19.5, 2: 12.0, 3: 20.5, 4: 19.8}
    cell_counts = {cell: 10 for cell in cell_psnr}
    active_region_ids = {
        1: (0, 1),
        2: (3, 4),
    }
    cached_region_ids = {
        10: (2,),
        11: (0, 1),
    }
    decision = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        max_promote=1,
    )
    assert decision.bad_partitions == (2,)
    assert decision.promote_indices == (10,)
    assert 11 not in decision.promote_indices


def test_spatial_focus_weight_is_not_diluted_by_a_frame_covering_good_cells_too() -> None:
    """A frame touching one catastrophic cell plus several healthy cells
    still gets that cell's full severity, not an average across its cells
    (issue #1259 finding 3, ported to the region-id-per-frame shape)."""

    cell_psnr = {0: 22.0, 1: 22.0, 2: 22.0, 3: 6.0}
    cell_counts = {cell: 10 for cell in cell_psnr}
    mixed_frame_region_ids = {5: (0, 1, 2, 3)}
    isolated_bad_frame_region_ids = {6: (3,)}
    decision_mixed = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids=mixed_frame_region_ids,
        cached_region_ids={},
        max_promote=0,
    )
    decision_isolated = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids=isolated_bad_frame_region_ids,
        cached_region_ids={},
        max_promote=0,
    )
    assert decision_mixed.priority_weights[5] == decision_isolated.priority_weights[6]


def test_spatial_focus_demote_excludes_a_frame_still_touching_a_bad_cell() -> None:
    cell_psnr = {0: 22.0, 1: 22.0, 2: 6.0}
    cell_counts = {cell: 10 for cell in cell_psnr}
    active_region_ids = {
        1: (0,),        # entirely good cells -> demotable
        2: (0, 2),      # touches the bad cell -> must not be demoted
    }
    decision = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids=active_region_ids,
        cached_region_ids={20: (2,)},
        max_promote=1,
    )
    assert 2 not in decision.demote_indices
    assert decision.demote_indices == (1,)


def test_spatial_focus_matches_ordinal_focus_on_a_one_region_per_frame_projection() -> None:
    """With exactly one region per frame, the cell-substrate engine must
    reduce to the same promote/demote decision as the ordinal-partition one
    -- confirms the shared `_historical_repair_selection_by_region` refactor
    preserves `decide_partition_focus`'s existing behavior exactly."""

    training = tuple(
        index for partition in range(6) for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )
    scores = {0: 18.9, 1: 14.6, 2: 13.6, 3: 15.1, 4: 19.5, 5: 20.0}
    ordinal_decision = decide_partition_focus(
        roles=roles,
        training_observations={},
        control_probe_observations={index: scores[index // 100] for index in training},
        scored_holdout_observations={},
        max_source_index=599,
        partition_count=6,
        outlier_k=1.5,
        active_training_indices=training,
        cached_indices=(102, 103, 202, 203, 302, 303),
        max_promote=6,
    )
    cell_psnr = {p: scores[p] for p in range(6)}
    cell_counts = {p: 2 for p in range(6)}
    active_region_ids = {index: (index // 100,) for index in training}
    cached_region_ids = {
        index: (index // 100,) for index in (102, 103, 202, 203, 302, 303)
    }
    spatial_decision = decide_spatial_partition_focus(
        roles=roles,
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.5,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        max_promote=6,
        max_bad_fraction=1.0,  # uncapped: this test checks the refactor, not the cap
    )
    assert spatial_decision.bad_partitions == ordinal_decision.bad_partitions
    assert set(spatial_decision.promote_indices) == set(ordinal_decision.promote_indices)


def test_spatial_focus_caps_bad_cells_to_worst_fraction_not_majority() -> None:
    """Reusing the ordinal-tuned outlier threshold at cell granularity can
    flag a large majority of cells bad at once (observed: 64-84 of ~89 cells
    on a real GPU run, #1723 2026-09-03) -- max_bad_fraction must cap this
    to a genuinely selective worst-subset regardless of how permissive the
    underlying median/MAD test is at this cell count."""

    # 20 cells: 15 clustered low (noisy), 5 clearly high -- median/MAD alone
    # would flag most of the low cluster as "bad" (all below median - k*mad
    # for a small, tight k), but max_bad_fraction=0.2 must cap it to 4.
    cell_psnr = {i: 10.0 + (i % 3) * 0.1 for i in range(15)}
    cell_psnr.update({15 + i: 25.0 for i in range(5)})
    cell_counts = {cell: 10 for cell in cell_psnr}

    decision = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids={},
        cached_region_ids={},
        max_promote=0,
        max_bad_fraction=0.2,
    )
    assert len(decision.bad_partitions) <= 4
    # The cap must keep the WORST cells, not an arbitrary subset.
    worst_four = tuple(sorted(sorted(cell_psnr, key=lambda c: cell_psnr[c])[:4]))
    assert set(decision.bad_partitions).issubset(set(cell_psnr))
    assert all(cell_psnr[c] <= 10.2 for c in decision.bad_partitions)


def test_spatial_focus_capped_bad_cells_let_demote_actually_fire() -> None:
    """The uncapped version made demote structurally unable to fire when
    most cells are flagged bad (every active frame touches at least one).
    Capping bad to a small worst-subset must let demote engage again for
    frames that don't touch any of the now-few genuinely-bad cells."""

    cell_psnr = {i: 10.0 + (i % 3) * 0.1 for i in range(15)}
    cell_psnr.update({15 + i: 25.0 for i in range(5)})
    cell_counts = {cell: 10 for cell in cell_psnr}
    # An active frame covering only high-PSNR cells, far from any of the
    # capped worst-four bad cells.
    active_region_ids = {100: (15, 16, 17)}

    decision = decide_spatial_partition_focus(
        roles=_spatial_roles(),
        cell_psnr=cell_psnr,
        cell_view_counts=cell_counts,
        outlier_k=1.0,
        active_region_ids=active_region_ids,
        cached_region_ids={200: (0,)},
        max_promote=1,
        max_bad_fraction=0.2,
    )
    assert decision.demote_indices == (100,)


# --- decide_partition_focus hysteresis retention (issue #285 / PR #1711) ---


def test_hysteresis_retains_partially_recovered_bad_partition() -> None:
    """A partition that only PARTIALLY recovers must not flip out of `bad`
    the instant it clears the bare per-tick outlier threshold -- that is
    the bang-bang oscillation this hysteresis retention exists to prevent
    (partition focus produced an unstable per-partition PSNR curve because
    the bad/good classification had no memory of the previous decision).
    Without `prior_bad_partitions`, the exact same recovered score already
    reads as fully healthy (not an outlier at all) -- so this fixture
    documents that the naive per-tick threshold really would drop it, and
    that hysteresis is what changes the verdict, not a stricter baseline
    rule.
    """
    training = tuple(
        index
        for partition in range(6)
        for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )

    def _decide(scores, *, prior_bad=None):
        observations = {index: scores[index // 100] for index in training}
        return decide_partition_focus(
            roles=roles,
            training_observations={},
            control_probe_observations=observations,
            scored_holdout_observations={},
            max_source_index=599,
            partition_count=6,
            outlier_k=1.5,
            active_training_indices=training,
            cached_indices=(),
            max_promote=6,
            prior_bad_partitions=prior_bad,
        )

    base_scores = {0: 25.0, 1: 25.2, 2: 15.0, 3: 24.8, 4: 25.1, 5: 24.9}
    first = _decide(base_scores)
    assert first.bad_partitions == (2,)

    # Partition 2 recovers most of the way (15.0 -> 24.7) but is still the
    # weakest partition in the scene by a wide margin over its own history.
    recovered_scores = dict(base_scores)
    recovered_scores[2] = 24.7

    # The naive, memoryless per-tick classification already reads this as
    # fully healthy -- confirms hysteresis, not a stricter baseline rule,
    # is what changes the verdict below.
    without_hysteresis = _decide(recovered_scores)
    assert without_hysteresis.bad_partitions == ()

    with_hysteresis = _decide(
        recovered_scores, prior_bad=frozenset(first.bad_partitions)
    )
    assert with_hysteresis.bad_partitions == (2,)


def test_hysteresis_releases_a_fully_recovered_partition() -> None:
    """Hysteresis retains a partially-recovered partition, but must not
    pin it as `bad` forever -- once it genuinely converges back to the
    rest of the scene, it has to be released, or the controller could
    never stop promoting/weighting a partition that has actually healed.
    """
    training = tuple(
        index
        for partition in range(6)
        for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )

    def _decide(scores, *, prior_bad=None):
        observations = {index: scores[index // 100] for index in training}
        return decide_partition_focus(
            roles=roles,
            training_observations={},
            control_probe_observations=observations,
            scored_holdout_observations={},
            max_source_index=599,
            partition_count=6,
            outlier_k=1.5,
            active_training_indices=training,
            cached_indices=(),
            max_promote=6,
            prior_bad_partitions=prior_bad,
        )

    base_scores = {0: 25.0, 1: 25.2, 2: 15.0, 3: 24.8, 4: 25.1, 5: 24.9}
    first = _decide(base_scores)
    assert first.bad_partitions == (2,)

    fully_recovered_scores = dict(base_scores)
    fully_recovered_scores[2] = 24.9

    released = _decide(
        fully_recovered_scores, prior_bad=frozenset(first.bad_partitions)
    )
    assert released.bad_partitions == ()


def test_hysteresis_is_a_noop_without_prior_bad_partitions() -> None:
    """Backward compatibility: omitting `prior_bad_partitions` (every
    existing call site before this change) must reproduce the exact
    historical bad/good classification, byte-for-byte."""
    training = tuple(
        index
        for partition in range(6)
        for index in (partition * 100, partition * 100 + 1)
    )
    roles = assign_partition_control_roles(
        source="control_probe",
        training_indices=training,
        scored_holdout_indices=(),
        control_probe_n=len(training),
        controller_enabled=True,
    )
    scores = {0: 18.9, 1: 14.6, 2: 13.6, 3: 15.1, 4: 19.5, 5: 20.0}
    observations = {index: scores[index // 100] for index in training}

    kwargs = dict(
        roles=roles,
        training_observations={},
        control_probe_observations=observations,
        scored_holdout_observations={},
        max_source_index=599,
        partition_count=6,
        outlier_k=1.5,
        active_training_indices=training,
        cached_indices=(102, 103, 202, 203, 302, 303),
        max_promote=6,
    )
    baseline = decide_partition_focus(**kwargs)
    explicit_empty = decide_partition_focus(
        **kwargs, prior_bad_partitions=frozenset()
    )
    explicit_none = decide_partition_focus(**kwargs, prior_bad_partitions=None)

    assert baseline.bad_partitions == (1, 2, 3)
    assert explicit_empty.bad_partitions == baseline.bad_partitions
    assert explicit_none.bad_partitions == baseline.bad_partitions


# --- decide_partition_focus_debt_allocator (#1723 v2: debt-based frame ranking) ---


def test_debt_allocator_selects_worst_covered_frame_each_tick():
    from eligsir.utils.partition_control import decide_partition_focus_debt_allocator

    # Two cells, cell 1 starts far worse than cell 0's target.
    cell_psnr = {0: 22.0, 1: 15.0}
    # source 10 covers the good cell, source 20 covers the bad cell.
    active_region_ids = {10: (0,), 20: (1,)}
    cached_region_ids = {}
    debt = {0: 0.0, 1: 0.0}
    target = {0: 1.0, 1: 1.0}

    decision = decide_partition_focus_debt_allocator(
        cell_psnr=cell_psnr,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        debt=debt,
        target=target,
        batch_size=1,
    )

    assert decision.selected_indices == (20,)
    # Serving cell 1 pays its debt down; cell 0's debt keeps accruing
    # (it wasn't served this tick), so it becomes the priority next tick.
    assert decision.debt[1] < decision.debt[0]


def test_debt_allocator_uses_max_debt_not_average_across_covered_cells():
    from eligsir.utils.partition_control import decide_partition_focus_debt_allocator

    cell_psnr = {0: 22.0, 1: 22.0, 2: 22.0, 3: 6.0}
    debt = {0: 0.0, 1: 0.0, 2: 0.0, 3: 50.0}  # cell 3 is catastrophically behind
    target = {c: 0.0 for c in cell_psnr}
    mixed_frame = {5: (0, 1, 2, 3)}   # touches the bad cell plus 3 good ones
    isolated_frame = {6: (3,)}        # touches only the bad cell

    decision = decide_partition_focus_debt_allocator(
        cell_psnr=cell_psnr, active_region_ids={**mixed_frame, **isolated_frame},
        cached_region_ids={}, debt=debt, target=target, batch_size=1,
    )
    # Both frames have equal priority (max debt = 50 either way) -- ties
    # break by ascending index, so frame 5 wins; the point under test is
    # that frame 5's priority was NOT diluted to 12.5 (=50/4 averaged).
    assert decision.selected_indices == (5,)


def test_debt_allocator_flag_requires_spatial_substrate_and_partition_control():
    from eligsir.cli.parser import parse_training_args
    args, _ = parse_training_args([
        "-s", "scene", "-m", "output/test",
        "--streaming_spatial_replay_focus",
        "--streaming_partition_focus_weight", "0.5",
        "--streaming_partition_focus_signal_source", "control_probe",
        "--streaming_partition_focus_spatial_substrate",
        "--streaming_partition_focus_debt_allocator",
    ])
    # Parses cleanly; the runtime guard (CausalEvaluationObserver.__init__)
    # is exercised by the existing spatial_substrate guard tests once this
    # flag is wired into that same validation block in Step 3.
    assert args.streaming_partition_focus_debt_allocator is True


def test_debt_allocator_requires_spatial_substrate_at_construction():
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver
    from types import SimpleNamespace

    args = SimpleNamespace(
        streaming_partition_focus_weight=0.5,
        streaming_partition_focus_signal_source="control_probe",
        streaming_partition_focus_control_probe_n=32,
        streaming_partition_focus_max_promote=20,
        streaming_partition_focus_outlier_k=1.5,
        streaming_partition_replay_focus=False,
        streaming_spatial_replay_focus=False,   # off -> spatial_substrate below is invalid too
        streaming_partition_focus_spatial_substrate=False,
        streaming_partition_focus_debt_allocator=True,
        streaming_memory_telemetry=False,
    )
    with pytest.raises(ValueError, match="requires --streaming_partition_focus_spatial_substrate"):
        CausalEvaluationObserver(
            runtime=SimpleNamespace(), plan=SimpleNamespace(holdout_ordinals=(), control_probe_records=()),
            dataset=SimpleNamespace(model_path="/tmp/x"), args=args, source_total=None, source_records=(),
        )


def test_debt_allocator_decision_promotes_cached_frame_and_demotes_dropped_active_frame():
    # Exercises the debt-allocator branch of _measure_partition_control
    # end-to-end (minus rendering) via its extracted pure helper, asserting
    # on the real PartitionFocusDecision it produces -- not just that the
    # underlying decide_partition_focus_debt_allocator ranks correctly
    # (already covered) or that the flag/guard wiring is correct (already
    # covered), but that the branch's own adaptation into
    # promote_indices/demote_indices/priority_weights is correct.
    from eligsir.runtime.causal_evaluation import (
        _debt_allocator_partition_focus_decision,
    )
    from eligsir.utils.partition_control import ControlSignalSource

    cell_psnr = {0: 22.0, 1: 10.0}
    cell_view_counts = {0: 5, 1: 5}
    active_region_ids = {100: (0,)}   # active frame only touches the good cell
    cached_region_ids = {200: (1,)}   # cached frame covers the bad, high-debt cell
    debt = {0: 0.0, 1: 5.0}

    decision, updated_debt = _debt_allocator_partition_focus_decision(
        source=ControlSignalSource.CONTROL_PROBE,
        cell_psnr=cell_psnr,
        cell_view_counts=cell_view_counts,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        debt=debt,
        batch_size=1,
    )

    # (a) promote: the cached frame covering the higher-debt cell is selected
    # and shows up in promote_indices (it came from cached_region_ids).
    assert decision.promote_indices == (200,)
    # (b) demote: the active frame that lost the ranking is reported demoted.
    assert decision.demote_indices == (100,)
    # priority_weights is keyed by the debt-allocator's selected_indices and
    # reflects cell 1's post-tick debt (5.0 prior + 1.0 target accrual - 1.0
    # paydown for being served this tick = 5.0).
    assert decision.priority_weights == {200: pytest.approx(5.0)}
    assert updated_debt[1] == pytest.approx(5.0)
    assert updated_debt[0] == pytest.approx(1.0)  # accrued, never served
    assert decision.repair_mode == "debt_allocator"
    assert decision.source == ControlSignalSource.CONTROL_PROBE
    assert decision.partition_means == cell_psnr
    assert decision.partition_counts == cell_view_counts


def test_debt_allocator_decision_priority_weight_is_max_not_average_across_covered_cells():
    # (c) priority_weights must reflect the max debt of the frame's worst
    # covered cell, not diluted to an average across all cells it covers.
    from eligsir.runtime.causal_evaluation import (
        _debt_allocator_partition_focus_decision,
    )
    from eligsir.utils.partition_control import ControlSignalSource

    cell_psnr = {0: 22.0, 1: 22.0, 2: 22.0, 3: 6.0}
    cell_view_counts = {index: 1 for index in cell_psnr}
    debt = {0: 0.0, 1: 0.0, 2: 0.0, 3: 50.0}  # cell 3 is catastrophically behind
    active_region_ids: dict = {}
    cached_region_ids = {
        5: (0, 1, 2, 3),  # touches the bad cell plus 3 good ones
        6: (3,),          # touches only the bad cell
    }

    decision, _updated_debt = _debt_allocator_partition_focus_decision(
        source=ControlSignalSource.CONTROL_PROBE,
        cell_psnr=cell_psnr,
        cell_view_counts=cell_view_counts,
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        debt=debt,
        batch_size=1,
    )

    assert decision.promote_indices == (5,)
    # Not diluted to ~13.5 (=(51+1+1+1)/4 averaged) -- reflects the max
    # debt among frame 5's covered cells (cell 3's, post accrual+paydown).
    assert decision.priority_weights[5] == pytest.approx(50.0)


def test_debt_allocator_decision_with_empty_cell_psnr_is_a_no_op():
    # (final-review finding #2) cell_psnr legitimately comes back {} until
    # SpatialMicrocellReplayFocusController.current_cell_quality() clears its
    # _min_history sample floor, and can empty out again mid-run via
    # stale-source pruning. Before the fix, target={cell: 1.0 for cell in
    # cell_psnr} was also {} in that case, which wiped ALL previously
    # accumulated debt to {} while active_region_ids/cached_region_ids
    # (geometry-derived, independent of cell_psnr) stayed non-empty --
    # degenerating into a priority-0.0-for-everyone actuation rather than a
    # real no-op. Assert debt survives unchanged and nothing is promoted or
    # demoted on this tick.
    from eligsir.runtime.causal_evaluation import (
        _debt_allocator_partition_focus_decision,
    )
    from eligsir.utils.partition_control import ControlSignalSource

    debt_before = {0: 3.0, 1: 7.5}
    active_region_ids = {100: (0,)}
    cached_region_ids = {200: (1,)}

    decision, updated_debt = _debt_allocator_partition_focus_decision(
        source=ControlSignalSource.CONTROL_PROBE,
        cell_psnr={},
        cell_view_counts={},
        active_region_ids=active_region_ids,
        cached_region_ids=cached_region_ids,
        debt=debt_before,
        batch_size=1,
    )

    # Debt is untouched -- not wiped to {}, not accrued/paid down either.
    assert updated_debt == debt_before
    assert updated_debt is not debt_before  # caller owns a fresh mapping
    # No frames promoted or demoted on a signal-less tick.
    assert decision.promote_indices == ()
    assert decision.demote_indices == ()
    assert decision.priority_weights == {}


def test_debt_allocator_publication_priorities_are_uniform_over_the_batch():
    """Issue #1723 wiring: deficit round-robin selects a batch; every selected
    frame must receive equal focus mass at publication. Publishing raw debt as
    severity let one debt outlier take all mass and floored the other 19 of 20
    frames to 1.0 (which the bounded publisher then drops) in the real runs."""
    from eligsir.runtime.causal_evaluation import (
        _partition_focus_publication_priorities,
    )
    from eligsir.utils.partition_control import ControlSignalSource, PartitionFocusDecision

    def decision(repair_mode):
        return PartitionFocusDecision(
            source=ControlSignalSource.CONTROL_PROBE,
            partition_means={}, partition_counts={},
            bad_partitions=(), good_partitions=(),
            priority_weights={5: 148.0, 6: 1.01, 7: 1.0001},
            promote_indices=(5, 6, 7), demote_indices=(),
            repair_mode=repair_mode, repair_extra_optimizer_steps=0,
        )

    uniform = _partition_focus_publication_priorities(decision("debt_allocator"))
    assert set(uniform) == {5, 6, 7}
    assert len(set(uniform.values())) == 1
    assert next(iter(uniform.values())) > 1.0

    legacy = _partition_focus_publication_priorities(decision("targeted_history"))
    assert legacy == {5: 148.0, 6: 1.01, 7: 1.0001}
