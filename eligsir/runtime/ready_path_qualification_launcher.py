"""Deterministic qualification of the canonical prepared READY-B4 path."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import argparse
import json
import os
import re
import tempfile

import numpy as np

from eligsir.contracts import BufferRef, CameraIntrinsics, SensorPacket, SourceIdentity
from eligsir.contracts.control import (
    DepthInsertionPlan,
    DispositionKind,
    FreshnessMode,
    HistoricalEligibility,
    MappingWorkPlan,
    ModalityEligibility,
    MutationBudget,
    ObservationDisposition,
    ResidencyIntent,
    ResolutionTier,
    SemanticAction,
    SupervisionConstellation,
    SupervisionWork,
)
from eligsir.mapping.observation_db import FrameObservationDB
from eligsir.runtime.prepared_ingress import (
    PreparedIngressControlToken,
    PreparedIngressItem,
    PreparedSourceIngressHandoff,
)
from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridge
from eligsir.runtime.prepared_source_producer import PreparedSourceProducer
from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.runtime.ready_b4_streaming_adapter import (
    ReadyB4StreamingOptimizationAdapter,
    ReadyB4StreamingStepInputs,
)
from eligsir.runtime.ready_only_tick import ReadyOnlyTickSelection
from eligsir.runtime.ready_path_qualification import (
    ReadyPathInterceptionKind,
    ReadyPathInterceptionScope,
    ReadyPathInterceptionTarget,
    ReadyPathLineageJoin,
    ReadyPathQualificationProbe,
)
from eligsir.runtime.ready_source import ReadySourceLane, ReadySourceOwner
from eligsir.runtime.transfers import DeviceTransferOwner, TransferEvent
from eligsir.tracking.providers import observation_from_pose_matrix


_CPU_CONFIG = {
    "batch_size": 4,
    "bootstrap_frames": 4,
    "source": "synthetic-tum-fr3-like",
    "source_resolution": [4, 3],
    "transfer_device": "cpu",
    "max_pinned_bytes": 65_536,
}


def _require_source_commit(source_commit: str | None) -> str:
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError("source commit must be an injected full 40-character SHA")
    return source_commit.lower()


def _packet(ordinal: int) -> SensorPacket:
    source = SourceIdentity(
        "synthetic-tum-fr3-like", 0, ordinal, 1_700_000_000 + ordinal * 33_333_333
    )
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = ordinal * 0.05
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(
            np.full((3, 4, 3), ordinal * 16, dtype=np.uint8), encoding="rgb8"
        ),
        depth=BufferRef.from_numpy(
            np.full((3, 4), 1_000 + ordinal, dtype=np.uint16),
            encoding="16UC1",
            units="raw",
        ),
        intrinsics=CameraIntrinsics(
            fx=4.0,
            fy=4.0,
            cx=1.5,
            cy=1.0,
            width=4,
            height=3,
            depth_scale=1_000.0,
        ),
        tracking=observation_from_pose_matrix(source, pose, provider="tum-groundtruth"),
    )


def _item(ordinal: int) -> PreparedIngressItem:
    packet = _packet(ordinal)
    disposition = ObservationDisposition(
        source=packet.source,
        snapshot_id=1,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=False,
        residency=ResidencyIntent.HOT,
        reason="cpu-ready-path-integration",
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True, depth_supervision=True
        ),
        semantic_action=SemanticAction.COMMIT,
        historical_eligibility=HistoricalEligibility.ELIGIBLE,
    )
    token = PreparedIngressControlToken(ordinal, packet.source, 1, disposition)
    return PreparedIngressItem(
        ordinal,
        packet,
        packet.tracking,
        disposition=disposition,
        control_token=token,
    )


class _CPUFallback:
    def __init__(self, runtime: CanonicalMappingRuntime) -> None:
        self.runtime = runtime
        self.bootstrap_bindings: tuple[Any, ...] = ()
        self.transactions: list[Any] = []

    def initialize(self, bindings: tuple[Any, ...]) -> None:
        self.bootstrap_bindings = tuple(bindings)
        self.runtime.initialized = True

    def tick(self, *, ready_transaction: Any):
        self.transactions.append(ready_transaction)
        return SimpleNamespace(executed=True, optimizer_steps=1, supervision_views=4)


def _cpu_runtime(owner: ReadySourceOwner, transfer_owner: DeviceTransferOwner):
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._batch_size = 4
    runtime.bootstrap_frames = 4
    runtime.initialized = False
    runtime._transfer_owner = transfer_owner
    runtime._ready_source_owner = None
    runtime._ready_only_tick_selector = None
    runtime._ready_bootstrap_bindings = []
    runtime._ready_bootstrap_selections = []
    runtime._observation_db = FrameObservationDB(device="cpu")
    runtime._versions = lambda: (0, 0, 0)
    runtime.enable_ready_source_owner(owner, warmup_complete=False)
    fallback = _CPUFallback(runtime)
    runtime._initialize = fallback.initialize
    runtime.tick = fallback.tick
    return runtime, fallback


def _plan(sources: tuple[SourceIdentity, ...]) -> MappingWorkPlan:
    eligibility = ModalityEligibility(rgb_supervision=True, depth_supervision=True)
    work = tuple(
        SupervisionWork(source, ResolutionTier.NATIVE, eligibility) for source in sources
    )
    return MappingWorkPlan(
        plan_id="ready-path-qualification-b4",
        snapshot_id=1,
        control_version=1,
        supervision_sources=(),
        replay_sources=sources,
        resolution_tier=ResolutionTier.NATIVE,
        optimizer_updates=1,
        insertion=DepthInsertionPlan(
            source=None,
            enabled=False,
            sample_stride=1,
            candidate_budget=0,
            seed_budget=0,
            voxel_size_m=0.01,
            minimum_confidence=0.0,
        ),
        allow_growth=False,
        allow_relocation=False,
        mutation_budget=MutationBudget(max_total=0),
        freshness_mode=FreshnessMode.HEALTHY,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        supervision_work=(),
        replay_work=work,
        supervision_constellation=SupervisionConstellation(
            requested_batch_size=4,
            requested_camera_count=4,
            realized_batch_size=4,
            realized_camera_count=4,
            unique_eligible_count=4,
            fill_ratio=1.0,
            microbatch_count=1,
            mode="ready_b4",
            reason="qualification",
            hysteresis_state="qualification",
        ),
        materialization_sources=(),
    )


def _interception(transfer_owner: DeviceTransferOwner) -> ReadyPathInterceptionScope:
    from scene.cameras import Camera

    return ReadyPathInterceptionScope(
        (
            ReadyPathInterceptionTarget(
                Camera,
                "materialize_pending_image",
                ReadyPathInterceptionKind.CPU_MATERIALIZATION,
            ),
            ReadyPathInterceptionTarget(
                CanonicalMappingRuntime,
                "_decode_reload_token",
                ReadyPathInterceptionKind.COLD_DECODE,
            ),
            ReadyPathInterceptionTarget(
                TransferEvent, "wait", ReadyPathInterceptionKind.WAIT
            ),
            ReadyPathInterceptionTarget(
                Future, "result", ReadyPathInterceptionKind.FUTURE_RESULT
            ),
            ReadyPathInterceptionTarget(
                transfer_owner, "upload_packed", ReadyPathInterceptionKind.H2D
            ),
            ReadyPathInterceptionTarget(
                transfer_owner, "readback_packed", ReadyPathInterceptionKind.D2H
            ),
        )
    )


def _write_artifact_atomically(artifact: Path, payload: dict[str, Any]) -> None:
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite retained artifact: {artifact}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.name}.", suffix=".tmp", dir=artifact.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, artifact)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_cpu_ready_path_qualification(*, output_dir: Path, source_commit: str | None) -> Path:
    source_commit = _require_source_commit(source_commit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "ready_path_cpu_integration.json"

    owner = ReadySourceOwner(max_fresh=4, max_replay=4)
    transfer_owner = DeviceTransferOwner(
        "cpu", max_inflight=4, max_pinned_bytes=_CPU_CONFIG["max_pinned_bytes"]
    )
    handoff = PreparedSourceIngressHandoff(capacity=4)
    producer = PreparedSourceProducer(owner, transfer_owner=transfer_owner)
    bridge = PreparedRuntimeBridge(
        handoff,
        producer,
        ready_owner=owner,
        max_items_per_tick=4,
        retained_commit_clearer=lambda _item: None,
    )
    runtime, fallback = _cpu_runtime(owner, transfer_owner)

    ingress = handoff.claim_producer("qualification")
    for ordinal in range(4):
        assert ingress.try_offer(_item(ordinal)).value == "enqueued"
    drained = bridge.drain()
    if drained.enqueued != 4:
        raise AssertionError("qualification did not prepare exactly four sources")

    leases = owner.select(4)
    bootstrap = ReadyOnlyTickSelection(leases, 4, 0, False, False)
    if not runtime.bootstrap_ready_only_selection(
        bootstrap, bootstrap_event=TransferEvent()
    ):
        raise AssertionError("prepared bootstrap did not initialize from exactly B4")

    sources = tuple(item.source_identity for item in fallback.bootstrap_bindings)
    adapter = ReadyB4StreamingOptimizationAdapter(
        bridge,
        runtime=runtime,
        transfer_owner=transfer_owner,
        warmup_complete=True,
    )
    baseline = transfer_owner.accounting()
    interception = _interception(transfer_owner)
    probe = ReadyPathQualificationProbe.begin(transfer_owner=transfer_owner)
    with interception:
        outcome = adapter.tick(ReadyB4StreamingStepInputs(snapshot=None, work_plan=_plan(sources)))
    if not outcome.executed or len(fallback.transactions) != 1:
        raise AssertionError("canonical prepared B4 adapter did not execute exactly once")

    transaction = fallback.transactions[0]
    lineages = tuple(
        ReadyPathLineageJoin.from_ready_camera(
            item, plan_id=transaction.plan.plan_id, transaction_id=1
        )
        for item in transaction.bindings
    )
    evidence = probe.finish(
        selection=outcome.selection,
        selector_diagnostics=runtime.ready_only_tick_diagnostics,
        cameras=transaction.bindings,
        lineages=lineages,
        interception=interception,
    )
    evidence.assert_qualified()
    after = transfer_owner.accounting()

    _write_artifact_atomically(
        artifact,
        {
            "claim": "gate:merge",
            "execution_backend": "cpu-fallback",
            "researchflow": "not_registered; not scientific evidence",
            "provenance": {"commit": source_commit, "configuration": _CPU_CONFIG},
            "sources": [source.fingerprint() for source in sources],
            "bootstrap": {
                "selected_source_ids": [source.fingerprint() for source in sources]
            },
            "b4": {
                "selected_source_ids": [
                    item.source_identity.fingerprint() for item in transaction.bindings
                ],
                "ready_camera_count": evidence.ready_camera_count,
                "qualified": evidence.qualified,
                "operations": asdict(evidence.operations),
                "failures": list(evidence.failures),
                "transaction_plan_id": transaction.plan.plan_id,
            },
            "lineage": [
                {
                    "source": item.source.fingerprint(),
                    "control_sequence": item.control_sequence,
                    "policy_snapshot_id": item.policy_snapshot_id,
                    "plan_id": item.plan_id,
                    "transaction_id": item.transaction_id,
                }
                for item in lineages
            ],
            "transfer": {
                "baseline": asdict(baseline),
                "after": asdict(after),
            },
            "gtt": {
                "status": "unavailable on cpu-fallback",
                "baseline_bytes": None,
                "after_bytes": None,
            },
        },
    )
    return artifact


def rocm_qualification_command(
    *, source_path: str, output_dir: str, source_commit: str | None
) -> str:
    source_commit = _require_source_commit(source_commit)
    return (
        "docker compose run --rm -T --privileged --device /dev/kfd --device /dev/dri "
        "--group-add keep-groups --security-opt label=disable "
        "--security-opt seccomp=unconfined --ipc host "
        "-e HSA_XNACK=1 -e HSA_ENABLE_SDMA=0 "
        f"-e ELIGSIR_SOURCE_COMMIT='{source_commit}' train python -m "
        "eligsir.runtime.ready_path_qualification_launcher "
        f"--rocm-qualification --source {source_path!r} --output-dir {output_dir!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-output-dir", type=Path)
    parser.add_argument("--rocm-qualification", action="store_true")
    parser.add_argument("--source", default="/data/tum/fr3")
    parser.add_argument("--output-dir", default="output/ready-path-rocm-qualification")
    parser.add_argument("--source-commit", default=os.environ.get("ELIGSIR_SOURCE_COMMIT"))
    args = parser.parse_args(argv)
    if args.cpu_output_dir is not None:
        print(run_cpu_ready_path_qualification(
            output_dir=args.cpu_output_dir, source_commit=args.source_commit
        ))
        return 0
    if args.rocm_qualification:
        print(rocm_qualification_command(
            source_path=args.source,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        ))
        return 0
    parser.error("choose --cpu-output-dir or --rocm-qualification")


if __name__ == "__main__":
    raise SystemExit(main())
