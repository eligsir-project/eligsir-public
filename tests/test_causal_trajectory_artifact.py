from __future__ import annotations

import json

import numpy as np

from eligsir.runtime.causal_entry import (
    write_causal_trajectory,
    write_mapped_camera_trajectory_plot,
)
from eligsir.contracts.runtime import SourceIdentity
from eligsir.utils.trajectory_eval import (
    mapped_camera_trajectory_display_coordinates,
    mapped_camera_trajectory_selection,
    plot_mapped_camera_trajectory,
)


class _Frame:
    source_ordinal = 7
    timestamp = 12.5
    color_ts_ns = None
    c2w = np.eye(4, dtype=np.float32)


def test_causal_trajectory_uses_source_timestamp_and_pose(tmp_path):
    path = write_causal_trajectory(tmp_path, [_Frame()])

    row = json.loads(path.read_text())
    assert row["id"] == 7
    assert row["timestamp"] == 12.5
    assert row["position"] == [0.0, 0.0, 0.0]
    assert row["rotation"] == np.eye(3).tolist()


def test_mapped_camera_trajectory_selection_uses_executed_supervision_only():
    selection = mapped_camera_trajectory_selection(
        (
            {"id": 1, "position": [0.0, 0.0, 0.0]},
            {"id": 2, "position": [1.0, 0.0, 0.0]},
            {"id": 3, "position": [2.0, 0.0, 0.0]},
        ),
        (
            {
                "executed": False,
                "optimizer_steps": 0,
                "dispositions": [
                    {"source_ordinal": 1, "mapping_candidate": True}
                ],
                "supervision_source_records": [{"source_ordinal": 1}],
            },
            {
                "executed": True,
                "optimizer_steps": 1,
                "dispositions": [
                    {"source_ordinal": 2, "mapping_candidate": True}
                ],
                "supervision_source_records": [{"source_ordinal": 2}],
            },
            {
                "executed": True,
                "optimizer_steps": 1,
                "dispositions": [],
                "supervision_source_records": [{"source_ordinal": 2}],
            },
        ),
    )

    assert selection == {
        "trajectory_ordinals": [1, 2, 3],
        "mapping_candidate_ordinals": [1, 2],
        "successfully_supervised_ordinals": [2],
        "successful_supervision_uses": {"2": 2},
        "unknown_mapping_candidate_ordinals": [],
        "unknown_successfully_supervised_ordinals": [],
    }


def test_mapped_camera_trajectory_display_coordinates_use_world_floor_plane():
    positions = np.asarray(
        [
            [-3.0, -2.0, 1.4],
            [2.0, 3.0, 1.6],
        ],
        dtype=np.float64,
    )

    horizontal = mapped_camera_trajectory_display_coordinates(positions)

    np.testing.assert_array_equal(horizontal, positions[:, :2])


def test_mapped_camera_plot_labels_the_world_axes_in_meters(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    recorded = []
    original_set = Axes.set

    def capture_axis_labels(axis, *args, **kwargs):
        recorded.append((kwargs.get("xlabel"), kwargs.get("ylabel")))
        return original_set(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "set", capture_axis_labels)
    plot_mapped_camera_trajectory(
        str(tmp_path / "trajectory.svg"),
        ({"id": 1, "position": [1.0, 2.0, 3.0]},),
        (),
        successful_supervision_uses={1: 1},
    )

    assert ("X (m)", "Y (m)") in recorded


def test_mapped_camera_plot_writes_trace_proven_selection_sidecar(tmp_path, monkeypatch):
    trajectory = tmp_path / "trajectory_full.jsonl"
    trajectory.write_text(json.dumps({"id": 7, "position": [0.0, 0.0, 0.0]}) + "\n")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "executed": True,
                "optimizer_steps": 1,
                "dispositions": [{"source_ordinal": 7, "mapping_candidate": True}],
                "supervision_source_records": [{"source_ordinal": 7}],
            }
        )
        + "\n"
    )
    plotted = []

    def plot(path, trajectory_rows, decision_rows, *, successful_supervision_uses=None):
        plotted.append((path, trajectory_rows, decision_rows))
        return mapped_camera_trajectory_selection(
            trajectory_rows,
            decision_rows,
            successful_supervision_uses=successful_supervision_uses,
        )

    monkeypatch.setattr("eligsir.utils.trajectory_eval.plot_mapped_camera_trajectory", plot)
    output = write_mapped_camera_trajectory_plot(tmp_path, trajectory, trace)

    assert output == tmp_path / "mapped_camera_trajectory.svg"
    assert plotted[0][0] == str(output)
    assert json.loads((tmp_path / "mapped_camera_trajectory.json").read_text())[
        "successfully_supervised_ordinals"
    ] == [7]


def test_mapped_camera_plot_uses_runtime_supervision_counts_without_trace(
    tmp_path, monkeypatch
):
    trajectory = tmp_path / "trajectory_full.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps({"id": ordinal, "position": [float(ordinal), 0.0, 0.0]})
            for ordinal in (1, 2, 3)
        )
        + "\n"
    )
    plotted = []

    def plot(path, trajectory_rows, decision_rows, *, successful_supervision_uses=None):
        plotted.append((path, trajectory_rows, decision_rows, successful_supervision_uses))
        return mapped_camera_trajectory_selection(
            trajectory_rows,
            decision_rows,
            successful_supervision_uses=successful_supervision_uses,
        )

    monkeypatch.setattr("eligsir.utils.trajectory_eval.plot_mapped_camera_trajectory", plot)
    output = write_mapped_camera_trajectory_plot(
        tmp_path,
        trajectory,
        None,
        successful_supervision_uses={2: 3, 3: 1},
    )

    assert output == tmp_path / "mapped_camera_trajectory.svg"
    assert plotted[0][2] == []
    assert plotted[0][3] == {2: 3, 3: 1}


def test_runtime_records_exact_successful_prepared_source_uses():
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    first = SourceIdentity("source", 0, 4, 1)
    second = SourceIdentity("source", 0, 9, 2)

    runtime.record_successful_supervision_sources((first, second, first))

    assert runtime.successful_supervision_uses_by_identity() == {first: 2, second: 1}
    assert runtime.successful_supervision_uses_by_ordinal() == {4: 2, 9: 1}


def test_supervision_ledger_does_not_alias_reconnected_stream_epoch():
    """issue #1688: a reconnected stream epoch reusing an ordinal must not be
    conflated with the earlier epoch's use count."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    epoch_a = SourceIdentity("cam", 0, 42, 100)
    epoch_b = SourceIdentity("cam", 1, 42, 200)

    runtime.record_successful_supervision_sources((epoch_a,))

    uses = runtime.successful_supervision_uses_by_identity()
    assert uses.get(epoch_a) == 1
    assert uses.get(epoch_b, 0) == 0
    # The ordinal-aggregated compat view sums across the epochs sharing 42.
    assert runtime.successful_supervision_uses_by_ordinal() == {42: 1}


def test_supervision_ledger_does_not_alias_distinct_source_namespaces():
    """issue #1688: two source ids that both happen to reuse ordinal 42 in
    the same composed session must be tracked independently."""
    from eligsir.runtime.production_application import CanonicalMappingRuntime

    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    cam_a = SourceIdentity("cam_a", 0, 42, 100)
    cam_b = SourceIdentity("cam_b", 0, 42, 100)

    runtime.record_successful_supervision_sources((cam_a,))

    uses = runtime.successful_supervision_uses_by_identity()
    assert uses.get(cam_a) == 1
    assert uses.get(cam_b, 0) == 0


def test_prepared_report_identity_records_reject_ordinal_only_alias():
    """issue #1688 regression: two retained records share `source_ordinal`
    42 but differ in `stream_epoch`/`source_id`. Recording success only for
    the first must not pull the second into the report selection."""
    from eligsir.runtime.causal_evaluation import prepared_report_source_identity_records

    from types import SimpleNamespace

    record_a = SimpleNamespace(source_id="cam", stream_epoch=0, source_ordinal=42, timestamp=1.0)
    record_b = SimpleNamespace(source_id="cam", stream_epoch=1, source_ordinal=42, timestamp=2.0)
    record_c = SimpleNamespace(source_id="other", stream_epoch=0, source_ordinal=42, timestamp=1.0)

    identity_a = SourceIdentity("cam", 0, 42, 1_000_000_000)

    selected = prepared_report_source_identity_records(
        (record_a, record_b, record_c),
        successful_supervision_uses={identity_a: 1},
        max_items=10,
    )

    assert selected == [record_a]


def test_prepared_report_cameras_span_successfully_supervised_sources():
    from types import SimpleNamespace

    from eligsir.runtime.causal_evaluation import prepared_report_source_identity_records

    def _identity(ordinal: int) -> SourceIdentity:
        return SourceIdentity(
            source_id="streaming-rgbd",
            stream_epoch=0,
            source_ordinal=ordinal,
            timestamp_ns=ordinal * 1_000_000_000,
        )

    records = tuple(
        SimpleNamespace(source_ordinal=index, timestamp=float(index)) for index in range(10)
    )

    selected = prepared_report_source_identity_records(
        records,
        successful_supervision_uses={
            _identity(1): 4,
            _identity(3): 1,
            _identity(5): 2,
            _identity(7): 1,
            _identity(9): 3,
        },
        max_items=3,
    )

    assert [record.source_ordinal for record in selected] == [1, 5, 9]


def test_causal_final_report_rebuilds_only_successfully_supervised_cameras(monkeypatch):
    from types import SimpleNamespace

    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    def _identity(ordinal: int) -> SourceIdentity:
        return SourceIdentity(
            source_id="streaming-rgbd",
            stream_epoch=0,
            source_ordinal=ordinal,
            timestamp_ns=ordinal * 1_000_000_000,
        )

    observer = CausalEvaluationObserver.__new__(CausalEvaluationObserver)
    observer._packet_source_id = None
    observer._packet_stream_epoch = 0
    observer._source_records = tuple(
        SimpleNamespace(source_ordinal=index, timestamp=float(index)) for index in range(8)
    )
    observer.runtime = SimpleNamespace(
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        successful_supervision_uses_by_identity=lambda: {
            _identity(1): 2,
            _identity(4): 1,
            _identity(7): 3,
        },
        scene=SimpleNamespace(train_cameras=["bootstrap"]),
    )
    observer.args = SimpleNamespace(streaming_report_trajectory_max_frames=2)
    built = []

    def build(records, *, data_device, report_source_id, report_stream_epoch):
        built.append(
            (
                [record.source_ordinal for record in records],
                data_device,
                report_source_id,
                report_stream_epoch,
            )
        )
        return ["report-camera"]

    monkeypatch.setattr(
        "eligsir.runtime.causal_evaluation.build_causal_holdout_cameras", build
    )

    assert observer._report_training_cameras() == ["report-camera"]
    assert built == [([1, 7], "cpu", None, 0)]


def test_mapped_camera_plot_out_name_writes_a_distinct_file(tmp_path, monkeypatch):
    """Refinement keeps drawing supervision from the cached/replayed frame
    pool well after mapping ends, so a mapping-phase-only trajectory
    snapshot goes stale the moment refinement starts. run_causal_mapping
    writes a second, post-refinement plot via out_name="mapped_camera_
    trajectory_refined" -- it must land at a DISTINCT path rather than
    overwriting the mapping-phase artifact, so both remain available for
    comparison."""
    trajectory = tmp_path / "trajectory_full.jsonl"
    trajectory.write_text(
        json.dumps({"id": 7, "position": [0.0, 0.0, 0.0]}) + "\n"
    )

    def plot(path, trajectory_rows, decision_rows, *, successful_supervision_uses=None):
        return mapped_camera_trajectory_selection(
            trajectory_rows,
            decision_rows,
            successful_supervision_uses=successful_supervision_uses,
        )

    monkeypatch.setattr("eligsir.utils.trajectory_eval.plot_mapped_camera_trajectory", plot)

    mapping_output = write_mapped_camera_trajectory_plot(
        tmp_path, trajectory, None, successful_supervision_uses={7: 1}
    )
    refined_output = write_mapped_camera_trajectory_plot(
        tmp_path,
        trajectory,
        None,
        successful_supervision_uses={7: 4},
        out_name="mapped_camera_trajectory_refined",
    )

    assert mapping_output == tmp_path / "mapped_camera_trajectory.svg"
    assert refined_output == tmp_path / "mapped_camera_trajectory_refined.svg"
    assert mapping_output != refined_output
    assert (tmp_path / "mapped_camera_trajectory.json").is_file()
    assert (tmp_path / "mapped_camera_trajectory_refined.json").is_file()
    mapping_selection = json.loads(
        (tmp_path / "mapped_camera_trajectory.json").read_text()
    )
    refined_selection = json.loads(
        (tmp_path / "mapped_camera_trajectory_refined.json").read_text()
    )
    assert mapping_selection["successful_supervision_uses"] == {"7": 1}
    assert refined_selection["successful_supervision_uses"] == {"7": 4}
