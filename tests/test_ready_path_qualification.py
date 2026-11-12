from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_source import PreparedSourceLease
from eligsir.runtime.ready_camera_binder import ReadyCameraBinding
from eligsir.runtime.ready_only_tick import (
    ReadyOnlyTickSelector,
    ReadySourceVersions,
)
from eligsir.runtime.ready_path_qualification import (
    ReadyPathInterceptionKind,
    ReadyPathInterceptionScope,
    ReadyPathInterceptionTarget,
    ReadyPathLineageJoin,
    ReadyPathQualificationConfig,
    ReadyPathQualificationProbe,
)
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import TransferAccounting


class _TransferOwner:
    def __init__(self, *, owner_id: str = "fake-transfer-owner") -> None:
        self.owner_id = owner_id
        self.h2d_ops = 0
        self.h2d_bytes = 0
        self.d2h_ops = 0
        self.d2h_bytes = 0
        self.pool_waits = 0
        self.explicit_sync_waits = 0
        self.calls: list[str] = []

    def h2d(self) -> None:
        self.calls.append("h2d")

    def d2h(self) -> None:
        self.calls.append("d2h")

    def accounting(self) -> TransferAccounting:
        return TransferAccounting(
            h2d_ops=self.h2d_ops,
            h2d_bytes=self.h2d_bytes,
            d2h_ops=self.d2h_ops,
            d2h_bytes=self.d2h_bytes,
            pool_waits=self.pool_waits,
            pool_drops=0,
            pinned_bytes=0,
            peak_pinned_bytes=0,
            inflight_limit=4,
            shape_buckets=0,
            shape_bucket_limit=4,
            explicit_sync_waits=self.explicit_sync_waits,
        )


class _FakeReadyBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cpu_materialize(self) -> None:
        self.calls.append("cpu_materialize")

    def cold_decode(self) -> None:
        self.calls.append("cold_decode")

    def wait(self) -> None:
        self.calls.append("wait")

    def future_result(self) -> None:
        self.calls.append("future_result")

    def h2d(self) -> None:
        self.calls.append("h2d")

    def d2h(self) -> None:
        self.calls.append("d2h")


def _interception(
    backend: _FakeReadyBackend,
    transfer_owner: object,
    *,
    h2d_owner: object | None | str = "configured",
    d2h_owner: object | None | str = "configured",
) -> ReadyPathInterceptionScope:
    resolved_h2d_owner = transfer_owner if h2d_owner == "configured" else h2d_owner
    resolved_d2h_owner = transfer_owner if d2h_owner == "configured" else d2h_owner
    h2d_target = ReadyPathInterceptionTarget(
        backend if resolved_h2d_owner is None else resolved_h2d_owner,
        "h2d",
        ReadyPathInterceptionKind.H2D,
        **({"transfer_owner": None} if resolved_h2d_owner is None else {}),
    )
    d2h_target = ReadyPathInterceptionTarget(
        backend if resolved_d2h_owner is None else resolved_d2h_owner,
        "d2h",
        ReadyPathInterceptionKind.D2H,
        **({"transfer_owner": None} if resolved_d2h_owner is None else {}),
    )
    return ReadyPathInterceptionScope(
        (
            ReadyPathInterceptionTarget(
                backend, "cpu_materialize", ReadyPathInterceptionKind.CPU_MATERIALIZATION
            ),
            ReadyPathInterceptionTarget(
                backend, "cold_decode", ReadyPathInterceptionKind.COLD_DECODE
            ),
            ReadyPathInterceptionTarget(
                backend, "wait", ReadyPathInterceptionKind.WAIT
            ),
            ReadyPathInterceptionTarget(
                backend, "future_result", ReadyPathInterceptionKind.FUTURE_RESULT
            ),
            h2d_target,
            d2h_target,
        )
    )


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("ready-qual", 2, ordinal, 1_700_000_000 + ordinal)


def _lease(ordinal: int, *, ready: bool = True) -> PreparedSourceLease:
    source = _source(ordinal)
    payload = torch.zeros(1)
    owner_token = object()
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(rgb_supervision=True),
        source_metadata={"source_ordinal": ordinal},
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "rgb": BufferRef(
                data=payload,
                spec=BufferSpec(
                    shape=(1,),
                    dtype="float32",
                    memory_domain=MemoryDomain.OTHER_DEVICE,
                    encoding="tensor",
                    device="cpu",
                ),
                ownership=BufferOwnership.BORROWED,
                owner_token=owner_token,
            )
        },
        device="cpu",
        ready_event=SimpleNamespace(query=lambda: ready),
        control_token=PreparedIngressControlToken(
            sequence=ordinal,
            source=source,
            snapshot_id=100 + ordinal,
            disposition=None,
        ),
    )


def _select(
    ordinals: tuple[int, ...],
    *,
    batch_size: int = 4,
) -> tuple[ReadyOnlyTickSelector, object]:
    owner = ReadySourceOwner(max_fresh=1, max_replay=max(1, len(ordinals)))
    producer = owner.claim_producer("qualification-fixture")
    for ordinal in ordinals:
        item = _lease(ordinal)
        assert producer.offer(
            item,
            lane=ReadySourceLane.REPLAY,
            arrival_ordinal=ordinal,
        )
    selector = ReadyOnlyTickSelector(
        owner,
        batch_size=batch_size,
        warmup_complete=True,
    )
    return selector, selector.select()


def _camera_bindings(selection) -> tuple[ReadyCameraBinding, ...]:
    versions = ReadySourceVersions(7, 11, 13)
    return tuple(
        ReadyCameraBinding(
            camera=SimpleNamespace(uid=lease.source_identity.source_ordinal),
            source_identity=lease.source_identity,
            modality_eligibility=lease.modality_eligibility,
            control_token=lease.control_token,
            versions=versions,
            prepared_tier="native",
            metadata=SimpleNamespace(source_identity=lease.source_identity),
            lease=lease,
        )
        for lease in selection.leases
    )


def _lineage(cameras, *, transaction_id: int = 31):
    return tuple(
        ReadyPathLineageJoin.from_ready_camera(
            camera,
            plan_id="plan-b4-17",
            transaction_id=transaction_id,
        )
        for camera in cameras
    )


def test_full_b4_replay_ready_path_is_zero_copy_and_joined() -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    backend = _FakeReadyBackend()
    interception = _interception(backend, transfer)
    probe = ReadyPathQualificationProbe.begin(
        transfer_owner=transfer,
        config=ReadyPathQualificationConfig(),
    )
    with interception:
        assert backend.calls == []

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    evidence.assert_qualified()
    assert evidence.qualified is True
    assert evidence.ready_camera_count == 4
    assert evidence.selected_sources == tuple(_source(i) for i in range(4))
    assert evidence.fallback_to_replay is True
    assert evidence.starved is False
    assert evidence.selection_count == 1
    assert evidence.fallback_count == 1
    assert evidence.starvation_count == 0
    assert evidence.transfer.owner_id == "fake-transfer-owner"
    assert evidence.transfer.h2d_ops == 0
    assert evidence.transfer.d2h_ops == 0
    assert tuple(item.policy_snapshot_id for item in evidence.lineages) == (
        100,
        101,
        102,
        103,
    )
    assert {item.transaction_id for item in evidence.lineages} == {31}


def test_starvation_is_reported_without_hiding_fallback_metrics() -> None:
    selector, selection = _select((0, 1))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    interception = _interception(_FakeReadyBackend(), transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    with interception:
        pass

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    assert evidence.qualified is False
    assert evidence.ready_camera_count == 2
    assert evidence.fallback_to_replay is True
    assert evidence.starved is True
    assert evidence.starvation_count == 1
    assert "ready_camera_count=2!=4" in evidence.failures
    assert "ready_selection_starved" in evidence.failures
    with pytest.raises(AssertionError, match="ready_camera_count"):
        evidence.assert_qualified()


@pytest.mark.parametrize(
    ("operation", "failure"),
    (
        ("cpu_materialize", "cpu_materializations=1"),
        ("cold_decode", "cold_decodes=1"),
        ("wait", "wait_calls=1"),
        ("future_result", "future_result_calls=1"),
        ("h2d", "h2d_entrypoint_calls=1"),
    ),
)
def test_real_forbidden_entrypoint_call_fails_b4_qualification(
    operation: str,
    failure: str,
) -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    backend = _FakeReadyBackend()
    interception = _interception(backend, transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    with interception:
        operation_owner = transfer if operation == "h2d" else backend
        getattr(operation_owner, operation)()

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    assert operation_owner.calls == [operation]
    assert failure in evidence.failures


def test_transfer_owner_delta_attributes_h2d_d2h_and_sync_waits() -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner(owner_id="cpu-fake-owner-0")
    interception = _interception(_FakeReadyBackend(), transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    transfer.h2d_ops = 1
    transfer.h2d_bytes = 64
    transfer.d2h_ops = 2
    transfer.d2h_bytes = 16
    transfer.pool_waits = 1
    transfer.explicit_sync_waits = 1
    with interception:
        pass

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    assert evidence.transfer.owner_id == "cpu-fake-owner-0"
    assert evidence.transfer.h2d_ops == 1
    assert evidence.transfer.h2d_bytes == 64
    assert evidence.transfer.d2h_ops == 2
    assert evidence.transfer.d2h_bytes == 16
    assert "tick_h2d_ops=1" in evidence.failures
    assert "transfer_pool_waits=1" in evidence.failures
    assert "explicit_sync_waits=1" in evidence.failures


def test_attributed_compact_d2h_does_not_hide_or_fail_ready_path() -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    backend = _FakeReadyBackend()
    interception = _interception(backend, transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    transfer.d2h_ops = 1
    transfer.d2h_bytes = 8
    with interception:
        transfer.d2h()

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    evidence.assert_qualified()
    assert evidence.transfer.d2h_ops == 1
    assert evidence.transfer.d2h_bytes == 8


def test_h2d_bytes_without_accounted_operation_still_fail() -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    interception = _interception(_FakeReadyBackend(), transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    transfer.h2d_bytes = 64
    with interception:
        pass

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    assert "tick_h2d_bytes=64" in evidence.failures


@pytest.mark.parametrize("captured_owner", (None, "other"))
def test_transfer_entrypoint_rejects_unattributed_or_mixed_owner(
    captured_owner: object | None,
) -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    other = _TransferOwner(owner_id="other")
    h2d_owner = None if captured_owner is None else other
    backend = _FakeReadyBackend()
    interception = _interception(backend, transfer, h2d_owner=h2d_owner)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    with interception:
        h2d_entrypoint_owner = backend if h2d_owner is None else h2d_owner
        h2d_entrypoint_owner.h2d()

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=_lineage(cameras),
        interception=interception,
    )

    expected = (
        "unattributed_transfer_calls=1"
        if captured_owner is None
        else "mixed_transfer_owner_calls=1"
    )
    assert expected in evidence.failures


def test_interception_requires_real_coverage_and_completed_scope() -> None:
    backend = _FakeReadyBackend()
    with pytest.raises(ValueError, match="missing required interception kinds"):
        ReadyPathInterceptionScope(
            (
                ReadyPathInterceptionTarget(
                    backend,
                    "cpu_materialize",
                    ReadyPathInterceptionKind.CPU_MATERIALIZATION,
                ),
            )
        )

    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    transfer = _TransferOwner()
    interception = _interception(backend, transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    with pytest.raises(ValueError, match="completed interception scope"):
        probe.finish(
            selection=selection,
            selector_diagnostics=selector.diagnostics,
            cameras=cameras,
            lineages=_lineage(cameras),
            interception=interception,
        )


def test_probe_rejects_different_transfer_owner_or_counter_regression() -> None:
    probe = ReadyPathQualificationProbe.begin(transfer_owner=_TransferOwner())
    with pytest.raises(ValueError, match="same transfer owner"):
        probe.transfer_delta(_TransferOwner())

    transfer = _TransferOwner()
    transfer.h2d_ops = 1
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    transfer.h2d_ops = 0
    with pytest.raises(ValueError, match="regressed"):
        probe.transfer_delta(transfer)


def test_lineage_must_join_selected_source_policy_and_transaction() -> None:
    selector, selection = _select((0, 1, 2, 3))
    cameras = _camera_bindings(selection)
    lineages = list(_lineage(cameras))
    lineages[1] = replace(lineages[1], transaction_id=32)
    transfer = _TransferOwner()
    interception = _interception(_FakeReadyBackend(), transfer)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer)
    with interception:
        pass

    evidence = probe.finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=lineages,
        interception=interception,
    )

    assert "lineage_transaction_ids_not_unique" in evidence.failures

    lineages = list(_lineage(cameras))
    lineages[1] = replace(lineages[1], policy_snapshot_id=999)
    transfer = _TransferOwner()
    interception = _interception(_FakeReadyBackend(), transfer)
    with interception:
        pass
    evidence = ReadyPathQualificationProbe.begin(transfer_owner=transfer).finish(
        selection=selection,
        selector_diagnostics=selector.diagnostics,
        cameras=cameras,
        lineages=lineages,
        interception=interception,
    )
    assert "lineage_policy_snapshot_mismatch" in evidence.failures
