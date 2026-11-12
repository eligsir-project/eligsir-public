"""Version-fenced fixed-topology GPU work and structural mutation barriers.

This module is deliberately independent of the production composition root.
It turns already-selected work into explicit homogeneous transactions and
provides the only bridge from one completed topology epoch to the existing
``GaussianMutationArbiter -> GaussianMapBackend`` mutation boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import torch

from eligsir.contracts.control import DepthInsertionPlan
from eligsir.map.arbiter import (
    DelegatedMutationProposal,
    GaussianMutationArbiter,
    MutationCommitResult,
    MutationProposal,
    ProposalKey,
    SurfaceInsertionProposal,
)


class BatchExecutionMode(str, Enum):
    """A planned rendering mode; execution never changes it implicitly."""

    SEQUENTIAL_ACCUMULATION = "sequential_accumulation"
    TRUE_CAMERA_BATCH = "true_camera_batch"


class StaleGpuWorkError(RuntimeError):
    """Raised before compute or mutation when a version fence is stale."""


@dataclass(frozen=True, slots=True)
class GpuWorkEnvelope:
    """One selected work item bound to a coherent map/version snapshot."""

    payload: Any
    source_id: str
    snapshot_id: int
    map_content_version: int
    topology_version: int
    parameter_version: int
    resolution_bucket: str
    render_depth: bool
    update_sh_rest: bool
    ready_ns: int
    deadline_ns: int
    execution_mode: BatchExecutionMode

    def __post_init__(self) -> None:
        if not self.source_id or not self.resolution_bucket:
            raise ValueError("source_id and resolution_bucket must be non-empty")
        for name in (
            "snapshot_id",
            "map_content_version",
            "topology_version",
            "parameter_version",
            "ready_ns",
            "deadline_ns",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        if self.deadline_ns < self.ready_ns:
            raise ValueError("deadline_ns must be >= ready_ns")
        if not isinstance(self.execution_mode, BatchExecutionMode):
            raise TypeError("execution_mode must be BatchExecutionMode")

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        """Fields that must remain fixed throughout one GPU transaction."""

        return (
            self.map_content_version,
            self.topology_version,
            self.parameter_version,
            self.resolution_bucket,
            bool(self.render_depth),
            bool(self.update_sh_rest),
            self.execution_mode,
        )


@dataclass(frozen=True, slots=True)
class FixedTopologyTransaction:
    """One bounded, homogeneous GPU saturation unit."""

    items: tuple[GpuWorkEnvelope, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ValueError("fixed-topology transaction must contain work")
        key = items[0].compatibility_key
        if any(item.compatibility_key != key for item in items[1:]):
            raise ValueError("fixed-topology transaction work must be homogeneous")
        object.__setattr__(self, "items", items)

    @property
    def execution_mode(self) -> BatchExecutionMode:
        return self.items[0].execution_mode

    @property
    def deadline_ns(self) -> int:
        return min(item.deadline_ns for item in self.items)


class FixedTopologyBatchBuilder:
    """FIFO transaction former bounded by shape semantics, B, and latency."""

    def __init__(
        self,
        *,
        max_views: int,
        max_pending: int | None = None,
    ) -> None:
        if int(max_views) < 1:
            raise ValueError("max_views must be >= 1")
        self._max_views = int(max_views)
        self._max_pending = self._max_views if max_pending is None else int(max_pending)
        if self._max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._pending: deque[GpuWorkEnvelope] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def add(self, work: GpuWorkEnvelope) -> None:
        if not isinstance(work, GpuWorkEnvelope):
            raise TypeError("work must be GpuWorkEnvelope")
        if len(self._pending) >= self._max_pending:
            raise OverflowError(
                f"fixed-topology pending bound {self._max_pending} reached"
            )
        self._pending.append(work)

    def pop_ready(
        self, *, now_ns: int, force: bool = False
    ) -> FixedTopologyTransaction | None:
        """Pop one compatible prefix once full, due, forced, or shape-closed."""

        if not self._pending:
            return None
        first = self._pending[0]
        compatible = 1
        for item in tuple(self._pending)[1 : self._max_views]:
            if item.compatibility_key != first.compatibility_key:
                break
            compatible += 1
        shape_closed = (
            compatible < len(self._pending)
            and self._pending[compatible].compatibility_key != first.compatibility_key
        )
        compatible_deadline = min(
            self._pending[index].deadline_ns for index in range(compatible)
        )
        ready = (
            force
            or compatible >= self._max_views
            or int(now_ns) >= compatible_deadline
            or shape_closed
        )
        if not ready:
            return None
        count = min(compatible, self._max_views)
        return FixedTopologyTransaction(
            tuple(self._pending.popleft() for _ in range(count))
        )


def _assert_versions(
    transaction: FixedTopologyTransaction,
    *,
    backend: Any,
    optimization_session: Any,
) -> None:
    expected = transaction.items[0]
    actual = {
        "map_content_version": int(backend.map_content_version),
        "topology_version": int(backend.topology_version),
        "parameter_version": int(optimization_session.parameter_version),
    }
    for name in ("map_content_version", "topology_version", "parameter_version"):
        wanted = int(getattr(expected, name))
        if actual[name] != wanted:
            raise StaleGpuWorkError(
                f"stale GPU work: {name} expected {wanted}, current {actual[name]}"
            )


def execute_fixed_topology_transaction(
    transaction: FixedTopologyTransaction,
    *,
    backend: Any,
    optimization_session: Any,
    pipe: Any,
    cameras: Sequence[Any],
    true_batch_executor: Callable[[tuple[Any, ...]], Any],
    sequential_executor: Callable[[tuple[Any, ...]], Any],
    true_batch_support: Callable[[Any, Sequence[Any]], tuple[bool, str]] | None = None,
) -> Any:
    """Fence versions before compute and execute exactly the planned mode."""

    _assert_versions(
        transaction, backend=backend, optimization_session=optimization_session
    )
    payloads = tuple(item.payload for item in transaction.items)
    if len(cameras) != len(payloads):
        raise ValueError("camera count must match fixed-topology work count")
    if transaction.execution_mode is BatchExecutionMode.SEQUENTIAL_ACCUMULATION:
        return sequential_executor(payloads)

    if true_batch_support is None:
        from eligsir.mapping.renderer.true_camera_batch import supports_true_camera_batch

        true_batch_support = supports_true_camera_batch
    supported, reason = true_batch_support(pipe, cameras)
    if not supported:
        raise RuntimeError(
            "planned TRUE_CAMERA_BATCH is unavailable; refusing sequential fallback: "
            f"{reason}"
        )
    return true_batch_executor(payloads)


def _validate_candidate_tensors(candidates: Any, *, require_accelerator: bool) -> None:
    names = (
        "pts",
        "colors",
        "log_scales",
        "quats",
        "normals",
        "confidence",
        "source",
    )
    tensors = []
    count = len(candidates)
    for name in names:
        value = getattr(candidates, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"device-native insertion requires tensor field {name}")
        if value.ndim == 0 or int(value.shape[0]) != count:
            raise ValueError(f"candidate field {name} is not row-aligned")
        tensors.append(value)
    opacity = getattr(candidates, "opacities_raw", None)
    if opacity is not None:
        if not isinstance(opacity, torch.Tensor):
            raise TypeError("device-native insertion requires tensor field opacities_raw")
        if opacity.ndim == 0 or int(opacity.shape[0]) != count:
            raise ValueError("candidate field opacities_raw is not row-aligned")
        tensors.append(opacity)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("device-native insertion candidate fields must share one device")
    device = next(iter(devices))
    if require_accelerator and device.type == "cpu":
        raise RuntimeError("accelerator-live insertion refuses CPU candidates")


def insertion_proposal_from_plan(
    plan: DepthInsertionPlan,
    candidates: Any,
    *,
    key: ProposalKey,
    iteration: int,
    require_accelerator: bool = True,
) -> SurfaceInsertionProposal:
    """Apply the sole typed insertion cap without leaving the candidate device."""

    if not isinstance(plan, DepthInsertionPlan):
        raise TypeError("plan must be DepthInsertionPlan")
    if not plan.enabled:
        raise ValueError("insertion plan is disabled")
    _validate_candidate_tensors(candidates, require_accelerator=require_accelerator)
    count = min(len(candidates), int(plan.max_seeds))
    if count <= 0:
        raise ValueError("insertion proposal requires at least one candidate")
    if count != len(candidates):
        index = torch.arange(count, device=candidates.pts.device, dtype=torch.long)
        candidates = candidates.select(index)
    return SurfaceInsertionProposal(
        key=key,
        batch=candidates.as_insertion_batch(iteration=int(iteration)),
    )


def validate_accelerator_live_insertion_capabilities(capabilities: Any) -> None:
    """Validate outer-resolved canonical accelerator capabilities."""

    from eligsir.runtime.config import CanonicalDeviceCapabilities

    if not isinstance(capabilities, CanonicalDeviceCapabilities):
        raise TypeError("canonical insertion requires CanonicalDeviceCapabilities")
    if not capabilities.gpu_frame_prep:
        reason = "gpu_frame_prep_disabled"
    elif not capabilities.gpu_frame_resident:
        reason = "gpu_frame_resident_disabled"
    elif not capabilities.tensor_candidate_path:
        reason = "legacy_only_post_backprojection_feature_enabled"
    else:
        return
    raise RuntimeError(
        "accelerator-live insertion composition is unsupported: " + reason
    )


class GaussianMutationEpoch:
    """One-shot, version-fenced structural proposal commit barrier."""

    def __init__(
        self,
        backend: Any,
        *,
        expected_map_content_version: int,
        expected_topology_version: int,
        max_population: int | None = None,
        allow_delegated: bool = False,
    ) -> None:
        self._backend = backend
        self._expected_map = int(expected_map_content_version)
        self._expected_topology = int(expected_topology_version)
        self._arbiter = GaussianMutationArbiter(
            backend, max_population=max_population
        )
        if type(allow_delegated) is not bool:
            raise TypeError("allow_delegated must be bool")
        self._allow_delegated = allow_delegated
        self._committed = False

    def commit(
        self, proposals: Sequence[MutationProposal]
    ) -> MutationCommitResult:
        if self._committed:
            raise RuntimeError("mutation epoch has already committed")
        proposals = tuple(proposals)
        if (
            not self._allow_delegated
            and any(
                isinstance(proposal, DelegatedMutationProposal)
                for proposal in proposals
            )
        ):
            raise TypeError(
                "accelerator-live mutation epochs require typed backend transaction proposals"
            )
        actual_map = int(self._backend.map_content_version)
        actual_topology = int(self._backend.topology_version)
        if actual_map != self._expected_map:
            raise StaleGpuWorkError(
                "stale mutation work: map_content_version expected "
                f"{self._expected_map}, current {actual_map}"
            )
        if actual_topology != self._expected_topology:
            raise StaleGpuWorkError(
                "stale mutation work: topology_version expected "
                f"{self._expected_topology}, current {actual_topology}"
            )
        # Seal before entering user/backend code so exceptions cannot make a
        # partially applied epoch retryable through the same barrier object.
        self._committed = True
        return self._arbiter.commit(proposals)


__all__ = [
    "BatchExecutionMode",
    "FixedTopologyBatchBuilder",
    "FixedTopologyTransaction",
    "GaussianMutationEpoch",
    "GpuWorkEnvelope",
    "StaleGpuWorkError",
    "execute_fixed_topology_transaction",
    "insertion_proposal_from_plan",
    "validate_accelerator_live_insertion_capabilities",
]
