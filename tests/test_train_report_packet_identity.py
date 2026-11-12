"""Report selection must use the packet ledger, not the bootstrap camera list."""

from types import SimpleNamespace

import numpy as np
import pytest

from eligsir.io.sources.decode_cache import StreamingRGBDFrame
from eligsir.runtime.replay import packet_from_streaming_frame


def _records():
    return [
        StreamingRGBDFrame(
            index=i,
            source_ordinal=i,
            timestamp=1.0 + i / 30,
            color_ts_ns=1_000_000_007 + i * 33_333_333,
            rgb_path=f"/data/scene/rgb/{i}.png",
            c2w=np.eye(4),
            fx=5.0,
            fy=5.0,
            cx=2.0,
            cy=1.0,
            width=4,
            height=2,
            depth_scale=1000.0,
            _rgb_np=np.zeros((2, 4, 3), dtype=np.uint8),
        )
        for i in range(20)
    ]


def _ledger(records):
    return {
        packet_from_streaming_frame(
            records[i], source_id="/data/scene", stream_epoch=3
        ).source: 2
        for i in (2, 7, 11, 19)
    }


def test_report_matches_real_packet_namespace_epoch_and_sensor_timestamp():
    from eligsir.runtime.causal_evaluation import (
        prepared_report_source_identity_records,
    )

    records = _records()
    args = dict(
        successful_supervision_uses=_ledger(records),
        max_items=3,
        packet_source_id="/data/scene",
        packet_stream_epoch=3,
    )
    assert [
        r.source_ordinal
        for r in prepared_report_source_identity_records(records, **args)
    ] == [2, 7, 19]
    args["packet_stream_epoch"] = 0
    assert prepared_report_source_identity_records(records, **args) == []
    args.update(packet_stream_epoch=3, packet_source_id="other")
    assert prepared_report_source_identity_records(records, **args) == []


def test_causal_entry_forwards_the_packet_adapter_identity_to_report_observer(
    monkeypatch, tmp_path
):
    """The report ledger must use the producer packet, not record aliases."""
    import eligsir.runtime.causal_entry as causal_entry
    import eligsir.runtime.causal_evaluation as causal_evaluation
    import eligsir.utils.execution_semantics as execution_semantics

    captured = {}

    class _StopAfterObserverConstruction(Exception):
        pass

    def observer(**kwargs):
        captured.update(kwargs)
        raise _StopAfterObserverConstruction

    composition = SimpleNamespace(
        source=SimpleNamespace(frames=()),
        config=SimpleNamespace(hard_limits={"optimization_batch_size": 4}),
        binding=SimpleNamespace(),
        runtime=SimpleNamespace(),
    )
    monkeypatch.setattr(
        causal_entry, "resolve_causal_mapping_composition", lambda **_kwargs: composition
    )
    monkeypatch.setattr(causal_entry, "build_causal_viewer", lambda *_args: (None, None, 0))
    monkeypatch.setattr(causal_entry, "ingress_telemetry_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(causal_entry, "decision_trace_observer_from_args", lambda **_kwargs: None)
    monkeypatch.setattr(
        causal_entry,
        "StreamingFramePacketAdapter",
        lambda _source: SimpleNamespace(source_id="packet://scene", stream_epoch=7),
    )
    monkeypatch.setattr(
        causal_evaluation,
        "causal_holdout_plan",
        lambda *_args: SimpleNamespace(holdout_ordinals=frozenset((1,))),
    )
    monkeypatch.setattr(causal_evaluation, "CausalEvaluationObserver", observer)
    monkeypatch.setattr(
        execution_semantics,
        "resolve_execution_semantics",
        lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
    )

    with pytest.raises(_StopAfterObserverConstruction):
        causal_entry.run_causal_mapping(
            dataset=SimpleNamespace(source_path="/record-alias", model_path=str(tmp_path)),
            opt=SimpleNamespace(),
            pipe=SimpleNamespace(),
            args=SimpleNamespace(
                streaming_input_fps_cap=30.0,
                streaming_partition_focus_weight=0.0,
            ),
        )

    assert captured["packet_source_id"] == "packet://scene"
    assert captured["packet_stream_epoch"] == 7


def test_report_rebuilds_trajectory_sample_not_five_bootstrap_views(monkeypatch):
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    observer = object.__new__(CausalEvaluationObserver)
    records = _records()
    observer._source_records = records
    observer._packet_source_id = "/data/scene"
    observer._packet_stream_epoch = 3
    observer.args = SimpleNamespace(streaming_report_trajectory_max_frames=3)
    observer.runtime = SimpleNamespace(
        scene=SimpleNamespace(train_cameras=["bootstrap"] * 5),
        gaussians=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
        successful_supervision_uses_by_identity=lambda: _ledger(records),
    )
    built = []

    def build(selected, *, data_device, report_source_id, report_stream_epoch):
        assert data_device == "cpu"
        assert report_source_id == "/data/scene"
        assert report_stream_epoch == 3
        built.extend(r.source_ordinal for r in selected)
        return ["report-view"] * len(selected)

    monkeypatch.setattr(
        "eligsir.runtime.causal_evaluation.build_causal_holdout_cameras", build
    )
    assert observer._report_training_cameras() == ["report-view"] * 3
    assert built == [2, 7, 19]
    observer._packet_source_id = "wrong"
    with pytest.raises(RuntimeError, match="identity"):
        observer._report_training_cameras()
    for uses in ({}, {identity: 0 for identity in _ledger(records)}):
        observer.runtime.successful_supervision_uses_by_identity = lambda: uses
        assert observer._report_training_cameras() == []


@pytest.mark.parametrize("max_items", [0, -1])
def test_disabled_report_sample_does_not_query_ledger_or_build_cameras(max_items):
    from eligsir.runtime.causal_evaluation import CausalEvaluationObserver

    observer = object.__new__(CausalEvaluationObserver)
    observer.args = SimpleNamespace(streaming_report_trajectory_max_frames=max_items)
    # No runtime or source catalog: disabled reporting must not access either.
    assert observer._report_training_cameras() == []
