"""All-view replay feedback: bounded, modality-aware and train-source-only."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateIndex, HistoricalFrameMetadata,
)
from eligsir.mapping.scheduler.hot_cold_challenger import HotColdChallengerArena
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken, SupervisionEvidenceBatch, SupervisionModality,
)
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.transfers import DeviceTransferOwner, TransferEvent


@pytest.fixture(autouse=True)
def _cpu_runtime_torch(monkeypatch):
    # Pure CPU construction bypasses the production accelerator loader, but the
    # consumer must execute real Torch packing, not the import-only smoke stub.
    from eligsir.runtime import production_application
    monkeypatch.setattr(production_application, "torch", torch)


def _runtime(*, enabled=True):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._optimization_policy = SimpleNamespace(
        streaming_historical_usefulness_tracking=enabled,
        streaming_historical_usefulness_loss_scale=1.0,
    )
    runtime._transfer_owner = DeviceTransferOwner("cpu", max_inflight=2)
    import numpy as np
    for _ in range(2):
        runtime._transfer_owner.upload_packed((np.zeros(256, dtype=np.uint8),), purpose="test_ingress")
    runtime._historical_candidate_index = HistoricalCandidateIndex()
    runtime._historical_frame_ids_by_source = {}
    runtime._hot_cold_challenger_arena = HotColdChallengerArena(lease_ticks=0)
    for i in range(4):
        source = SourceIdentity("train", 0, i, i)
        runtime._historical_frame_ids_by_source[source] = i
        runtime._historical_candidate_index.register(HistoricalFrameMetadata(i, 0))
    return runtime


def _result(runtime, *, transaction=1, steps=1):
    sources = tuple(runtime._historical_frame_ids_by_source)
    packet = SupervisionEvidenceBatch(
        snapshot_token=EvidenceSnapshotToken(0, 0, 0, 0, transaction, 0, 0, 0),
        transaction_id=transaction,
        source_identities=sources,
        modalities=tuple(SupervisionModality(depth_loss_enabled=True) for _ in sources),
        rgb_residual_summary=torch.tensor([0.1, 1.0, 100.0, 3.0]),
        depth_residual_summary=torch.tensor([2.0, 0.1, 100.0, 4.0]),
        rgb_valid_pixels=torch.full((4,), 12),
        depth_valid_pixels=torch.tensor([10, 5, 8, 0]),
    )
    return SimpleNamespace(evidence=packet, optimizer_step_count=steps)


def _transaction(runtime, qualities=((1.0, 1.0), (1.0, 0.0), (0.0, 0.0), None)):
    return SimpleNamespace(bindings=tuple(
        SimpleNamespace(
            source_identity=source,
            lease=SimpleNamespace(source_metadata={"replay_quality": quality}),
        )
        for source, quality in zip(runtime._historical_frame_ids_by_source, qualities)
    ))


def _observe(runtime, transaction, start=1, count=3):
    for step in range(start, start + count):
        runtime._observe_replay_evidence(_result(runtime, transaction=step), transaction)
        runtime._poll_replay_evidence()


def test_all_view_feedback_uses_measured_quality_and_separate_modalities():
    runtime = _runtime()
    _observe(runtime, _transaction(runtime))
    arena = runtime._hot_cold_challenger_arena
    assert arena.modality_weight(1, "rgb") > arena.modality_weight(0, "rgb")
    assert arena.modality_weight(0, "depth") > arena.modality_weight(1, "depth")
    assert arena.modality_weight(1, "depth") == 0.0
    assert arena.modality_weight(2, "rgb") == 0.0  # high loss cannot rescue corruption
    assert not arena.modality_trusted(3, "rgb")  # unknown is not perfect
    assert not any(arena.modality_trusted(i, "insertion") for i in range(4))
    index = runtime._historical_candidate_index
    assert index.metadata_for(1).base_weight > 1.0
    assert 0 < index.metadata_for(2).base_weight < 1.0
    assert index.metadata_for(3).base_weight == 1.0
    telemetry = runtime.historical_usefulness_telemetry()
    assert telemetry["observations_total"] == 9
    assert telemetry["trusted_updates_total"] == 3
    assert runtime._transfer_owner.accounting().d2h_ops == 3


def test_one_exposure_cannot_promote_and_unknown_or_unbound_sources_abstain():
    runtime = _runtime()
    _observe(runtime, _transaction(runtime), count=1)
    assert not runtime._hot_cold_challenger_arena.modality_trusted(1, "rgb")
    assert runtime._historical_candidate_index.metadata_for(1).base_weight == 1.0
    _observe(runtime, SimpleNamespace(bindings=()), start=2, count=3)
    assert not runtime._hot_cold_challenger_arena.modality_trusted(1, "rgb")


@pytest.mark.parametrize("enabled,steps", [(False, 1), (True, 0)])
def test_disabled_or_rejected_update_never_stages_feedback(enabled, steps):
    runtime = _runtime(enabled=enabled)
    runtime._observe_replay_evidence(_result(runtime, steps=steps), _transaction(runtime))
    assert runtime._transfer_owner.accounting().d2h_ops == 0
    assert runtime.historical_usefulness_telemetry()["observations_total"] == 0


def test_delayed_readback_skips_new_feedback_without_host_wait(monkeypatch):
    runtime = _runtime()
    event = SimpleNamespace(query=lambda: False, wait=lambda: pytest.fail("host wait"))
    stream = runtime._transfer_owner.transfer_stream
    copy = stream.d2h_into

    def delayed(source, host):
        copy(source, host)
        return TransferEvent(event)

    monkeypatch.setattr(stream, "d2h_into", delayed)
    runtime._observe_replay_evidence(_result(runtime), _transaction(runtime))
    pending = runtime._pending_replay_evidence
    runtime._observe_replay_evidence(_result(runtime, transaction=2), _transaction(runtime))
    assert runtime._pending_replay_evidence is pending
    assert runtime._transfer_owner.accounting().d2h_ops == 1
    assert runtime.historical_usefulness_telemetry()["feedback_busy"] == 1
    event.query = lambda: True
    runtime._poll_replay_evidence()
    assert runtime._pending_replay_evidence is None
    assert runtime.historical_usefulness_telemetry()["observations_total"] == 3


def test_retired_identity_cannot_be_resurrected_by_late_feedback():
    runtime = _runtime()
    runtime._observe_replay_evidence(_result(runtime), _transaction(runtime))
    source = next(iter(runtime._historical_frame_ids_by_source))
    runtime._historical_frame_ids_by_source.pop(source)
    runtime._historical_candidate_index.retire(0)
    runtime._poll_replay_evidence()
    assert not runtime._hot_cold_challenger_arena.modality_trusted(0, "rgb")
    with pytest.raises(KeyError):
        runtime._historical_candidate_index.metadata_for(0)


def test_optional_feedback_defers_when_shared_pinned_capacity_is_full():
    runtime = _runtime()
    runtime._transfer_owner = DeviceTransferOwner("cpu", max_pinned_bytes=32)
    runtime._observe_replay_evidence(_result(runtime), _transaction(runtime))
    assert runtime.historical_usefulness_telemetry()["feedback_busy"] == 1
    assert runtime._transfer_owner.accounting().d2h_ops == 0


def test_runtime_close_abandons_unready_feedback_without_wait(monkeypatch):
    from eligsir.runtime import production_application

    runtime = _runtime()
    runtime._observe_replay_evidence(_result(runtime), _transaction(runtime))
    readback = runtime._pending_replay_evidence[0]
    event = SimpleNamespace(query=lambda: False, wait=lambda: pytest.fail("close wait"))
    readback.event = TransferEvent(event)
    runtime._closed = False
    runtime._runtime_policy = None
    runtime._admission_prefetch_executor = None
    runtime.admission_prefetch_telemetry = lambda: {}
    runtime.scene = None
    runtime._observation_db = SimpleNamespace(active_supervision_frame_ids=())
    runtime._bootstrap = []
    runtime._pending_supervision_modalities = {}
    runtime._source_identities_by_ordinal = {}
    runtime._services = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(production_application, "reset_sensor_depth_cache_state", lambda: None, raising=False)
    runtime.close()
    runtime.close()
    assert runtime._pending_replay_evidence is None
    assert runtime._transfer_owner.pending_count == 1
    event.query = lambda: True
    assert runtime._transfer_owner.pending_count == 0


def test_feedback_consumer_mutation_removes_the_measured_effect(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime, "record_modality_usefulness_observation", lambda *_: False)
    _observe(runtime, _transaction(runtime))
    assert runtime._historical_candidate_index.metadata_for(1).base_weight == 1.0
    assert not runtime._hot_cold_challenger_arena.modality_trusted(1, "rgb")
