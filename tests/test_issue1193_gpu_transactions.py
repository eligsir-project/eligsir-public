"""Issue #1193 fixed-topology GPU transaction ownership contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.control import DepthInsertionPlan
from eligsir.contracts.runtime import SourceIdentity
from eligsir.map.arbiter import DelegatedMutationProposal, ProposalKey
from eligsir.map.gpu_transactions import (
    BatchExecutionMode,
    FixedTopologyBatchBuilder,
    GaussianMutationEpoch,
    GpuWorkEnvelope,
    StaleGpuWorkError,
    execute_fixed_topology_transaction,
    insertion_proposal_from_plan,
    validate_accelerator_live_insertion_capabilities,
)
from eligsir.map.mutation import TopologyChange
from eligsir.runtime.config import CanonicalDeviceCapabilities


def _work(
    source: str,
    *,
    ready_ns: int = 0,
    deadline_ns: int = 100,
    map_version: int = 3,
    topology_version: int = 7,
    parameter_version: int = 11,
    snapshot_id: int = 5,
    resolution: str = "full",
    render_depth: bool = False,
    mode: BatchExecutionMode = BatchExecutionMode.SEQUENTIAL_ACCUMULATION,
) -> GpuWorkEnvelope:
    return GpuWorkEnvelope(
        payload=source,
        source_id=source,
        snapshot_id=snapshot_id,
        map_content_version=map_version,
        topology_version=topology_version,
        parameter_version=parameter_version,
        resolution_bucket=resolution,
        render_depth=render_depth,
        update_sh_rest=True,
        ready_ns=ready_ns,
        deadline_ns=deadline_ns,
        execution_mode=mode,
    )


def test_batch_builder_forms_only_homogeneous_bounded_transactions():
    builder = FixedTopologyBatchBuilder(max_views=2)
    builder.add(_work("a"))
    builder.add(_work("b"))

    transaction = builder.pop_ready(now_ns=0)

    assert transaction is not None
    assert [item.source_id for item in transaction.items] == ["a", "b"]
    assert transaction.execution_mode is BatchExecutionMode.SEQUENTIAL_ACCUMULATION
    assert builder.pending_count == 0


def test_distinct_snapshot_ids_batch_inside_the_same_fixed_topology_epoch():
    builder = FixedTopologyBatchBuilder(max_views=2)
    builder.add(_work("a", snapshot_id=5))
    builder.add(_work("b", snapshot_id=6))

    transaction = builder.pop_ready(now_ns=0)

    assert transaction is not None
    assert [item.snapshot_id for item in transaction.items] == [5, 6]


def test_batch_builder_has_a_finite_fail_closed_pending_bound():
    builder = FixedTopologyBatchBuilder(max_views=4, max_pending=2)
    builder.add(_work("a"))
    builder.add(_work("b"))

    with pytest.raises(OverflowError, match="pending bound"):
        builder.add(_work("c"))
    assert builder.pending_count == 2


def test_incompatible_work_closes_current_transaction_without_waiting_for_fill():
    builder = FixedTopologyBatchBuilder(max_views=4)
    builder.add(_work("a", resolution="full"))
    builder.add(_work("b", resolution="half"))

    first = builder.pop_ready(now_ns=0)

    assert first is not None
    assert [item.source_id for item in first.items] == ["a"]
    assert builder.pending_count == 1


def test_latency_deadline_releases_underfilled_transaction():
    builder = FixedTopologyBatchBuilder(max_views=4)
    builder.add(_work("a", ready_ns=10, deadline_ns=30))

    assert builder.pop_ready(now_ns=29) is None
    transaction = builder.pop_ready(now_ns=30)
    assert transaction is not None
    assert [item.source_id for item in transaction.items] == ["a"]


def test_requested_true_batch_never_silently_falls_back_when_unqualified():
    pipe = SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=False)
    cameras = tuple(
        SimpleNamespace(image_width=8, image_height=6) for _ in range(2)
    )
    builder = FixedTopologyBatchBuilder(max_views=2)
    builder.add(_work("a", mode=BatchExecutionMode.TRUE_CAMERA_BATCH))
    builder.add(_work("b", mode=BatchExecutionMode.TRUE_CAMERA_BATCH))
    transaction = builder.pop_ready(now_ns=0)

    with pytest.raises(RuntimeError, match="requires_sparse_grad"):
        execute_fixed_topology_transaction(
            transaction,
            backend=SimpleNamespace(map_content_version=3, topology_version=7),
            optimization_session=SimpleNamespace(parameter_version=11),
            pipe=pipe,
            cameras=cameras,
            true_batch_executor=lambda _items: "true",
            sequential_executor=lambda _items: "sequential",
        )


def test_qualified_sparse_true_batch_executes_without_sequential_dispatch():
    pipe = SimpleNamespace(sh_impl="gsplat", gsplat_sparse_grad=True)
    cameras = tuple(
        SimpleNamespace(image_width=8, image_height=6) for _ in range(2)
    )
    builder = FixedTopologyBatchBuilder(max_views=2)
    builder.add(_work("a", mode=BatchExecutionMode.TRUE_CAMERA_BATCH))
    builder.add(_work("b", mode=BatchExecutionMode.TRUE_CAMERA_BATCH))
    transaction = builder.pop_ready(now_ns=0)
    calls = []

    result = execute_fixed_topology_transaction(
        transaction,
        backend=SimpleNamespace(map_content_version=3, topology_version=7),
        optimization_session=SimpleNamespace(parameter_version=11),
        pipe=pipe,
        cameras=cameras,
        true_batch_executor=lambda items: calls.append(("true", items)) or "true",
        sequential_executor=lambda items: calls.append(("sequential", items)),
    )

    assert result == "true"
    assert calls == [("true", ("a", "b"))]


def test_planned_sequential_accumulation_does_not_probe_or_switch_modes():
    calls = []
    builder = FixedTopologyBatchBuilder(max_views=2)
    builder.add(_work("a"))
    builder.add(_work("b"))
    transaction = builder.pop_ready(now_ns=0)

    result = execute_fixed_topology_transaction(
        transaction,
        backend=SimpleNamespace(map_content_version=3, topology_version=7),
        optimization_session=SimpleNamespace(parameter_version=11),
        pipe=object(),
        cameras=(object(), object()),
        true_batch_executor=lambda _items: calls.append("true"),
        sequential_executor=lambda items: calls.append(tuple(items)) or "sequential",
    )

    assert result == "sequential"
    assert calls == [("a", "b")]


def test_version_fence_rejects_stale_topology_before_compute():
    builder = FixedTopologyBatchBuilder(max_views=1)
    builder.add(_work("a", topology_version=7))
    transaction = builder.pop_ready(now_ns=0)
    calls = []

    with pytest.raises(StaleGpuWorkError, match="topology_version"):
        execute_fixed_topology_transaction(
            transaction,
            backend=SimpleNamespace(map_content_version=3, topology_version=8),
            optimization_session=SimpleNamespace(parameter_version=11),
            pipe=object(),
            cameras=(object(),),
            true_batch_executor=lambda _items: calls.append("true"),
            sequential_executor=lambda _items: calls.append("sequential"),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("backend_map", "backend_topology", "session_parameter", "field"),
    [
        (4, 7, 11, "map_content_version"),
        (3, 8, 11, "topology_version"),
        (3, 7, 12, "parameter_version"),
    ],
)
def test_every_gpu_work_version_fence_is_checked_before_compute(
    backend_map, backend_topology, session_parameter, field
):
    builder = FixedTopologyBatchBuilder(max_views=1)
    builder.add(_work("a"))
    transaction = builder.pop_ready(now_ns=0)
    calls = []

    with pytest.raises(StaleGpuWorkError, match=field):
        execute_fixed_topology_transaction(
            transaction,
            backend=SimpleNamespace(
                map_content_version=backend_map,
                topology_version=backend_topology,
            ),
            optimization_session=SimpleNamespace(parameter_version=session_parameter),
            pipe=object(),
            cameras=(object(),),
            true_batch_executor=lambda _items: calls.append("true"),
            sequential_executor=lambda _items: calls.append("sequential"),
        )
    assert calls == []


class _Candidates:
    def __init__(self, device="cpu"):
        self.pts = torch.arange(15, dtype=torch.float32, device=device).reshape(5, 3)
        self.colors = torch.zeros((5, 3), device=device)
        self.log_scales = torch.zeros((5, 3), device=device)
        self.quats = torch.zeros((5, 4), device=device)
        self.normals = torch.zeros((5, 3), device=device)
        self.confidence = torch.ones(5, device=device)
        self.source = torch.zeros(5, dtype=torch.int16, device=device)
        self.opacities_raw = None
        self.birth_frame = 2
        self.init_opacity = 0.1

    def __len__(self):
        return len(self.pts)

    def select(self, index):
        selected = _Candidates(self.pts.device)
        for name in (
            "pts", "colors", "log_scales", "quats", "normals", "confidence", "source"
        ):
            setattr(selected, name, getattr(self, name)[index])
        return selected

    def as_insertion_batch(self, *, iteration):
        from eligsir.map.mutation import GaussianInsertionBatch

        return GaussianInsertionBatch(
            points=self.pts,
            colors=self.colors,
            scales=self.log_scales,
            rotations=self.quats,
            normals=self.normals,
            confidence=self.confidence,
            source=self.source,
            birth_frame=self.birth_frame,
            birth_iter=iteration,
        )


def _insertion_plan(seed_budget: int) -> DepthInsertionPlan:
    return DepthInsertionPlan(
        source=(
            SourceIdentity("rgbd", 0, 2, 3)
            if seed_budget > 0
            else None
        ),
        enabled=seed_budget > 0,
        sample_stride=1,
        candidate_budget=seed_budget,
        seed_budget=seed_budget,
        voxel_size_m=0.02,
        minimum_confidence=0.0,
    )


def test_insertion_proposal_consumes_plan_cap_and_stays_torch_native():
    proposal = insertion_proposal_from_plan(
        _insertion_plan(2),
        _Candidates(),
        key=ProposalKey(0, 2, 3, "rgbd"),
        iteration=9,
        require_accelerator=False,
    )

    assert proposal.batch.points.shape == (2, 3)
    assert isinstance(proposal.batch.points, torch.Tensor)
    assert proposal.batch.points.device.type == "cpu"
    assert proposal.batch.birth_iter == 9


def test_insertion_proposal_fails_closed_when_plan_disables_insertion():
    with pytest.raises(ValueError, match="disabled"):
        insertion_proposal_from_plan(
            _insertion_plan(0),
            _Candidates(),
            key=ProposalKey(0, 2, 3, "rgbd"),
            iteration=9,
            require_accelerator=False,
        )


@pytest.mark.parametrize(
    ("gpu_frame_prep", "gpu_frame_resident", "reason"),
    [
        (False, True, "gpu_frame_prep_disabled"),
        (True, False, "gpu_frame_resident_disabled"),
    ],
)
def test_accelerator_live_composition_rejects_cpu_fallback_flags(
    gpu_frame_prep, gpu_frame_resident, reason
):
    capabilities = CanonicalDeviceCapabilities(
        gpu_frame_prep=gpu_frame_prep,
        gpu_frame_resident=gpu_frame_resident,
        tensor_candidate_path=True,
    )
    with pytest.raises(RuntimeError, match=reason):
        validate_accelerator_live_insertion_capabilities(capabilities)


class _Txn:
    def insert(self, batch):
        return int(batch.points.shape[0])

    def commit(self):
        return TopologyChange(map_version=4, topology_version=8, inserted=1, content_changed=True)


class _Backend:
    def __init__(self):
        self.map_content_version = 3
        self.topology_version = 7
        self.population_count = 0
        self.begin_calls = 0
        self.commit_calls = 0

    def begin_mutation(self):
        self.begin_calls += 1
        return _Txn()

    def commit_mutation(self, txn):
        self.commit_calls += 1
        change = txn.commit()
        self.map_content_version = change.map_content_version
        self.topology_version = change.topology_version
        return change


def test_mutation_epoch_crosses_arbiter_backend_once_and_then_seals():
    backend = _Backend()
    proposal = insertion_proposal_from_plan(
        _insertion_plan(1),
        _Candidates(),
        key=ProposalKey(0, 2, 3, "rgbd"),
        iteration=9,
        require_accelerator=False,
    )
    epoch = GaussianMutationEpoch(
        backend,
        expected_map_content_version=3,
        expected_topology_version=7,
    )

    result = epoch.commit((proposal,))

    assert result.inserted == 1
    assert backend.begin_calls == backend.commit_calls == 1
    with pytest.raises(RuntimeError, match="already committed"):
        epoch.commit(())
    assert backend.begin_calls == backend.commit_calls == 1


def test_mutation_epoch_rejects_stale_work_before_opening_backend_transaction():
    backend = _Backend()
    epoch = GaussianMutationEpoch(
        backend,
        expected_map_content_version=2,
        expected_topology_version=7,
    )

    with pytest.raises(StaleGpuWorkError, match="map_content_version"):
        epoch.commit(())
    assert backend.begin_calls == backend.commit_calls == 0


def test_mutation_epoch_rejects_legacy_delegated_raw_writer_before_execution():
    backend = _Backend()
    calls = []
    proposal = DelegatedMutationProposal(
        key=ProposalKey(0, 2, 3, "legacy"),
        executor=lambda: calls.append("raw-write"),
    )
    epoch = GaussianMutationEpoch(
        backend,
        expected_map_content_version=3,
        expected_topology_version=7,
    )

    with pytest.raises(TypeError, match="typed backend transaction proposals"):
        epoch.commit((proposal,))
    assert calls == []
    assert backend.begin_calls == backend.commit_calls == 0


def test_explicit_delegated_epoch_is_version_fenced_and_committed_alone():
    backend = _Backend()
    backend.record_delegated_mutation = lambda **_kwargs: TopologyChange(
        map_version=4,
        topology_version=8,
        content_changed=True,
    )
    calls = []
    proposal = DelegatedMutationProposal(
        key=ProposalKey(0, 2, 3, "mcmc"),
        executor=lambda: calls.append("mcmc"),
    )
    epoch = GaussianMutationEpoch(
        backend,
        expected_map_content_version=3,
        expected_topology_version=7,
        allow_delegated=True,
    )

    result = epoch.commit((proposal,))

    assert calls == ["mcmc"]
    assert result.applied == 1
    with pytest.raises(RuntimeError, match="already committed"):
        epoch.commit(())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_real_gpu_device_proposal_crosses_one_gsplat_backend_mutation_epoch():
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from tests.test_insertion_hot_path_backend_parity import _make_model

    model = _make_model(n_points=3)
    backend = GsplatMapBackend(model)
    before = int(model.get_xyz.shape[0])
    proposal = insertion_proposal_from_plan(
        _insertion_plan(2),
        _Candidates("cuda"),
        key=ProposalKey(0, 2, 3, "rgbd"),
        iteration=9,
    )
    epoch = GaussianMutationEpoch(
        backend,
        expected_map_content_version=0,
        expected_topology_version=0,
    )

    result = epoch.commit((proposal,))

    assert proposal.batch.points.device.type == "cuda"
    assert int(model.get_xyz.shape[0]) == before + 2
    assert result.inserted == 2
    assert backend.map_content_version == 1
    assert backend.topology_version == 1
