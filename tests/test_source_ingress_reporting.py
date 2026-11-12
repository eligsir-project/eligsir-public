"""Drop counts must reach `report.json` (issue #1273, §7's silent-cap fix).

Before this, `_PacedPacketProducer._dropped`'s only outlets were an in-loop
telemetry snapshot, a stdout line, and `CausalReplaySummary.transport_dropped`
-- referenced outside `causal_replay.py` only by tests. The #1259 campaign's
816 lost frames were recorded in no run artifact at all.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from eligsir.contracts.runtime import (
    BufferRef,
    CameraIntrinsics,
    SensorPacket,
    SourceIdentity,
)
from eligsir.runtime.causal_replay import DatasetPacketAdapter, _PacedPacketProducer
from eligsir.runtime.source_ingress import IngestionValueConfig
from eligsir.tracking.providers import observation_from_pose_matrix
from eligsir.utils.partition_eval import compute_trajectory_coverage


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


_ZERO_DROP_BLOCK = {
    "ingress_frames_dropped": 0,
    "ingress_dropped_at_arrival": 0,
    "ingress_dropped_at_enqueue": 0,
    "ingress_frames_admitted": 120,
    "ingestion_policy": "fifo_drop_newest",
    "ingress_mapping_result_valid": True,
}


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


def _write_report(model_path: str, **kwargs):
    from eligsir.utils.comparison_report import write_post_training_report

    return write_post_training_report(
        model_path=model_path,
        iteration=100,
        gaussians=_FakeGaussian(),
        train_cams=[_FakeCam("a")],
        test_cams=[_FakeCam("t")],
        render_fn=_minimal_render_fn,
        pipe=_FakePipe(),
        background=None,
        **kwargs,
    )


@dataclass(frozen=True)
class _Record:
    ordinal: int
    source_ordinal: int
    timestamp: float
    c2w: np.ndarray


class _Adapter(DatasetPacketAdapter[_Record]):
    def packet_from_record(self, record: _Record) -> SensorPacket:
        source = SourceIdentity("dataset", 0, record.ordinal, 1_000 + record.ordinal)
        return SensorPacket(
            source=source,
            rgb=BufferRef.from_numpy(
                np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"
            ),
            depth=None,
            intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
            tracking=observation_from_pose_matrix(source, np.eye(4), provider="dataset"),
        )


def _records(n: int, *, step: float = 0.5):
    out = []
    for i in range(n):
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = step * i
        out.append(_Record(i, i, float(i) * 0.01, c2w))
    return tuple(out)


def _drain_producer(records, *, capacity=4, config=None):
    producer = _PacedPacketProducer(
        records,
        _Adapter(),
        source_hz=10_000.0,
        capacity=capacity,
        ingestion_config=config,
    )
    producer.start()
    while not producer.complete:
        packet = producer.take(timeout_s=0.05)
        if packet is None:
            continue
    producer.close()
    return producer


def test_producer_reports_preserved_finite_catalogue_without_transport_loss():
    producer = _drain_producer(_records(60), capacity=4)
    assert producer.dropped == 0
    assert producer.dropped_at_arrival == producer.dropped_at_enqueue == 0
    block = producer.source_ingress_report_block()
    assert block["ingress_frames_dropped"] == producer.dropped
    assert block["ingress_dropped_at_arrival"] == producer.dropped_at_arrival
    assert block["ingress_dropped_at_enqueue"] == producer.dropped_at_enqueue
    assert block["ingress_frames_offered"] == block["ingress_frames_admitted"] == 60
    assert block["ingress_queue_capacity"] == 4
    assert block["ingress_max_observed_buffer_lag_s"] >= 0.0
    assert block["ingress_mapping_result_valid"] is True
    assert block["ingestion_policy"] == "finite_preservation"


def test_report_block_is_json_primitive_only():
    """`causal_replay_summary.json` writes with `default=str`, so a dataclass
    or enum here would silently stringify into something useless."""
    import json

    producer = _drain_producer(_records(40), capacity=4)
    block = producer.source_ingress_report_block()
    round_tripped = json.loads(json.dumps(block))
    assert round_tripped == block


def test_zero_loss_counts_land_in_trajectory_coverage_without_changing_the_verdict():
    producer = _drain_producer(_records(60), capacity=4)
    block = producer.source_ingress_report_block()

    kwargs = dict(
        scene_source=None,
        n_reg=None,
        frames_admitted=10,
        partition_psnr=None,
    )
    without = compute_trajectory_coverage(**kwargs)
    with_ingress = compute_trajectory_coverage(
        **kwargs, source_ingress_geometry_fields=block
    )

    assert with_ingress["ingress_frames_dropped"] == producer.dropped
    assert with_ingress["ingress_dropped_at_arrival"] == 0
    # Additive ONLY -- no verdict/validity field may move.
    assert with_ingress["verdict"] == without["verdict"]
    assert with_ingress["mapping_result_valid"] == without["mapping_result_valid"]
    for key, value in without.items():
        assert with_ingress[key] == value, key
    # The unbounded per-event list never rides into trajectory_coverage.
    assert "ingress_overflow_events" not in with_ingress


def test_write_post_training_report_lands_the_block_and_warns(capsys, tmp_path):
    """The headline fix, end to end: the counts must reach report.json ON DISK,
    and a lossy run must be loud (CLAUDE.md's no-silent-caps convention)."""
    block = {
        "ingress_frames_dropped": 816,
        "ingress_dropped_at_arrival": 800,
        "ingress_dropped_at_enqueue": 16,
        "ingress_frames_admitted": 7130,
        "ingestion_policy": "fifo_drop_newest",
        "ingress_information_coverage": 0.4,
        "ingress_max_observed_buffer_lag_s": 136.0,
        "ingress_mapping_result_valid": True,
    }
    summary = _write_report(str(tmp_path / "lossy"), source_ingress=block)
    assert summary["source_ingress"] == block

    on_disk = json.loads(
        (tmp_path / "lossy" / "comparison" / "iter_100" / "report.json").read_text()
    )
    assert on_disk["source_ingress"]["ingress_frames_dropped"] == 816
    assert on_disk["source_ingress"]["ingress_dropped_at_arrival"] == 800
    assert on_disk["source_ingress"]["ingress_dropped_at_enqueue"] == 16
    # ...and the additive siblings inside trajectory_coverage, per design 7.1.
    assert on_disk["trajectory_coverage"]["ingress_frames_dropped"] == 816

    out = capsys.readouterr().out
    assert "[source_ingress] WARNING" in out and "816" in out


def test_write_post_training_report_is_silent_and_absent_without_drops(capsys, tmp_path):
    """A clean run must not warn, and omitting the kwarg entirely must
    reproduce the exact report.json from before this field existed."""
    clean = dict(_ZERO_DROP_BLOCK)
    summary = _write_report(str(tmp_path / "clean"), source_ingress=clean)
    assert summary["source_ingress"] == clean
    assert "[source_ingress] WARNING" not in capsys.readouterr().out

    summary_absent = _write_report(str(tmp_path / "absent"))
    assert "source_ingress" not in summary_absent
    on_disk = json.loads(
        (tmp_path / "absent" / "comparison" / "iter_100" / "report.json").read_text()
    )
    assert "source_ingress" not in on_disk
    assert not any(k.startswith("ingress_") for k in on_disk["trajectory_coverage"])


def test_value_ranked_ingress_configuration_fails_closed_for_finite_replay():
    with pytest.raises(ValueError, match="incompatible with finite preservation"):
        _PacedPacketProducer(
            _records(2),
            _Adapter(),
            source_hz=100_000.0,
            capacity=4,
            ingestion_config=IngestionValueConfig(enabled=True, history_frames=8),
        )
