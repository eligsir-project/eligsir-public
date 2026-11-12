"""Opt-in, non-timing qualification evidence for the ready-only B4 path.

The probe observes existing ready-selection, camera-binding, lineage, and
``DeviceTransferOwner.accounting()`` contracts.  It does not execute work,
wait on readiness, materialize pixels, decode COLD payloads, or alter the
executor/bootstrap hot path.  Callers opt in by sampling one probe around a
tick inside a declared real-entrypoint interception scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Iterable

from eligsir.contracts import SourceIdentity
from eligsir.runtime.ready_camera_binder import ReadyCameraBinding
from eligsir.runtime.ready_only_tick import (
    ReadyOnlyTickDiagnostics,
    ReadyOnlyTickSelection,
)
from eligsir.runtime.transfers import TransferAccounting


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative exact integer")
    return value


@dataclass(frozen=True, slots=True)
class ReadyPathQualificationConfig:
    """The canonical steady-state ready-path batch contract."""

    required_ready_cameras: int = 4

    def __post_init__(self) -> None:
        required = _non_negative_int(
            self.required_ready_cameras,
            "required_ready_cameras",
        )
        if required != 4:
            raise ValueError("ready-path qualification requires canonical B4")


@dataclass(frozen=True, slots=True)
class ReadyPathOperationCounts:
    """Counts captured by an executed entrypoint-interception scope."""

    cpu_materializations: int = 0
    cold_decodes: int = 0
    wait_calls: int = 0
    future_result_calls: int = 0
    h2d_entrypoint_calls: int = 0
    d2h_entrypoint_calls: int = 0
    unattributed_transfer_calls: int = 0
    mixed_transfer_owner_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "cpu_materializations",
            "cold_decodes",
            "wait_calls",
            "future_result_calls",
            "h2d_entrypoint_calls",
            "d2h_entrypoint_calls",
            "unattributed_transfer_calls",
            "mixed_transfer_owner_calls",
        ):
            _non_negative_int(getattr(self, name), name)


class ReadyPathInterceptionKind(str, Enum):
    CPU_MATERIALIZATION = "cpu_materialization"
    COLD_DECODE = "cold_decode"
    WAIT = "wait"
    FUTURE_RESULT = "future_result"
    H2D = "h2d"
    D2H = "d2h"


_REQUIRED_INTERCEPTION_KINDS = frozenset(
    {
        ReadyPathInterceptionKind.CPU_MATERIALIZATION,
        ReadyPathInterceptionKind.COLD_DECODE,
        ReadyPathInterceptionKind.WAIT,
        ReadyPathInterceptionKind.FUTURE_RESULT,
        ReadyPathInterceptionKind.H2D,
        ReadyPathInterceptionKind.D2H,
    }
)
_INFER_TRANSFER_OWNER = object()


@dataclass(frozen=True, slots=True)
class ReadyPathInterceptionTarget:
    """One real callable entrypoint temporarily wrapped for an opt-in tick."""

    owner: Any
    attribute: str
    kind: ReadyPathInterceptionKind
    transfer_owner: Any = _INFER_TRANSFER_OWNER

    def __post_init__(self) -> None:
        if not isinstance(self.attribute, str) or not self.attribute.strip():
            raise ValueError("interception attribute must be non-empty")
        if not isinstance(self.kind, ReadyPathInterceptionKind):
            raise TypeError("interception kind must be ReadyPathInterceptionKind")
        if not callable(getattr(self.owner, self.attribute, None)):
            raise TypeError("interception target must resolve to a callable")
        is_transfer = self.kind in {
            ReadyPathInterceptionKind.H2D,
            ReadyPathInterceptionKind.D2H,
        }
        if is_transfer and self.transfer_owner is _INFER_TRANSFER_OWNER:
            object.__setattr__(self, "transfer_owner", self.owner)
        elif not is_transfer and self.transfer_owner is not _INFER_TRANSFER_OWNER:
            raise ValueError("only transfer entrypoints may name a transfer owner")
        elif not is_transfer:
            object.__setattr__(self, "transfer_owner", None)


class ReadyPathInterceptionScope:
    """Execute a tick with declared real entrypoints wrapped and restored."""

    def __init__(self, targets: Iterable[ReadyPathInterceptionTarget]) -> None:
        self._targets = tuple(targets)
        if not all(isinstance(item, ReadyPathInterceptionTarget) for item in self._targets):
            raise TypeError("targets must contain ReadyPathInterceptionTarget values")
        kinds = {item.kind for item in self._targets}
        missing = _REQUIRED_INTERCEPTION_KINDS - kinds
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"missing required interception kinds: {names}")
        keys = {(id(item.owner), item.attribute) for item in self._targets}
        if len(keys) != len(self._targets):
            raise ValueError("interception targets must be unique owner/attribute pairs")
        self._counts = {kind: 0 for kind in ReadyPathInterceptionKind}
        self._transfer_calls: dict[int, tuple[Any, int]] = {}
        self._unattributed_transfer_calls = 0
        self._installed: list[tuple[ReadyPathInterceptionTarget, bool, Any]] = []
        self._entered = False
        self._completed = False

    @property
    def completed(self) -> bool:
        return self._completed

    def _capture(self, target: ReadyPathInterceptionTarget) -> None:
        self._counts[target.kind] += 1
        if target.kind not in {
            ReadyPathInterceptionKind.H2D,
            ReadyPathInterceptionKind.D2H,
        }:
            return
        owner = target.transfer_owner
        if owner is None:
            self._unattributed_transfer_calls += 1
            return
        key = id(owner)
        previous = self._transfer_calls.get(key)
        if previous is None:
            self._transfer_calls[key] = (owner, 1)
        else:
            self._transfer_calls[key] = (previous[0], previous[1] + 1)

    @staticmethod
    def _instance_attribute(owner: Any, attribute: str) -> tuple[bool, Any]:
        try:
            namespace = vars(owner)
        except TypeError:
            return True, getattr(owner, attribute)
        if attribute in namespace:
            return True, namespace[attribute]
        return False, None

    def __enter__(self) -> ReadyPathInterceptionScope:
        if self._entered or self._completed:
            raise RuntimeError("interception scope is single-use")
        self._entered = True
        try:
            for target in self._targets:
                original = getattr(target.owner, target.attribute)
                had_instance_value, instance_value = self._instance_attribute(
                    target.owner,
                    target.attribute,
                )

                @wraps(original)
                def intercepted(*args, __target=target, __original=original, **kwargs):
                    self._capture(__target)
                    return __original(*args, **kwargs)

                setattr(target.owner, target.attribute, intercepted)
                self._installed.append((target, had_instance_value, instance_value))
        except Exception:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        for target, had_instance_value, instance_value in reversed(self._installed):
            if had_instance_value:
                setattr(target.owner, target.attribute, instance_value)
            else:
                delattr(target.owner, target.attribute)
        self._installed.clear()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self._restore()
        self._completed = True

    def snapshot(self, *, configured_transfer_owner: Any) -> ReadyPathOperationCounts:
        if not self._completed:
            raise ValueError("qualification requires a completed interception scope")
        mixed = sum(
            count
            for owner, count in self._transfer_calls.values()
            if owner is not configured_transfer_owner
        )
        return ReadyPathOperationCounts(
            cpu_materializations=self._counts[
                ReadyPathInterceptionKind.CPU_MATERIALIZATION
            ],
            cold_decodes=self._counts[ReadyPathInterceptionKind.COLD_DECODE],
            wait_calls=self._counts[ReadyPathInterceptionKind.WAIT],
            future_result_calls=self._counts[ReadyPathInterceptionKind.FUTURE_RESULT],
            h2d_entrypoint_calls=self._counts[ReadyPathInterceptionKind.H2D],
            d2h_entrypoint_calls=self._counts[ReadyPathInterceptionKind.D2H],
            unattributed_transfer_calls=self._unattributed_transfer_calls,
            mixed_transfer_owner_calls=mixed,
        )


@dataclass(frozen=True, slots=True)
class ReadyPathLineageJoin:
    """Bounded source-to-policy-to-transaction join for one ready camera."""

    source: SourceIdentity
    control_sequence: int
    policy_snapshot_id: int | None
    plan_id: str
    transaction_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        _non_negative_int(self.control_sequence, "control_sequence")
        if self.policy_snapshot_id is not None:
            _non_negative_int(self.policy_snapshot_id, "policy_snapshot_id")
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty")
        _non_negative_int(self.transaction_id, "transaction_id")

    @classmethod
    def from_ready_camera(
        cls,
        camera: ReadyCameraBinding,
        *,
        plan_id: str,
        transaction_id: int,
    ) -> ReadyPathLineageJoin:
        if not isinstance(camera, ReadyCameraBinding):
            raise TypeError("camera must be ReadyCameraBinding")
        token = camera.control_token
        return cls(
            source=camera.source_identity,
            control_sequence=token.sequence,
            policy_snapshot_id=token.snapshot_id,
            plan_id=plan_id,
            transaction_id=transaction_id,
        )


@dataclass(frozen=True, slots=True)
class ReadyPathTransferDelta:
    """One-owner transfer attribution over the observed qualification tick."""

    owner_id: str
    h2d_ops: int
    h2d_bytes: int
    d2h_ops: int
    d2h_bytes: int
    pool_waits: int
    explicit_sync_waits: int


@dataclass(frozen=True, slots=True)
class ReadyPathTickEvidence:
    """Bounded evidence packet for one candidate ready-only B4 tick."""

    selected_sources: tuple[SourceIdentity, ...]
    ready_camera_count: int
    fallback_to_replay: bool
    starved: bool
    selection_count: int
    fallback_count: int
    starvation_count: int
    lineages: tuple[ReadyPathLineageJoin, ...]
    operations: ReadyPathOperationCounts
    transfer: ReadyPathTransferDelta
    failures: tuple[str, ...]

    @property
    def qualified(self) -> bool:
        return not self.failures

    def assert_qualified(self) -> None:
        if self.failures:
            raise AssertionError("ready-path qualification failed: " + ", ".join(self.failures))


class ReadyPathQualificationProbe:
    """Capture transfer accounting before and after one opt-in ready tick."""

    def __init__(
        self,
        *,
        transfer_owner: Any,
        before: TransferAccounting,
        owner_id: str,
        config: ReadyPathQualificationConfig,
    ) -> None:
        self._transfer_owner = transfer_owner
        self._before = before
        self._owner_id = owner_id
        self._config = config
        self._finished = False

    @classmethod
    def begin(
        cls,
        *,
        transfer_owner: Any,
        config: ReadyPathQualificationConfig | None = None,
        owner_id: str | None = None,
    ) -> ReadyPathQualificationProbe:
        accounting = getattr(transfer_owner, "accounting", None)
        if not callable(accounting):
            raise TypeError("transfer_owner must provide accounting()")
        before = accounting()
        if not isinstance(before, TransferAccounting):
            raise TypeError("transfer_owner accounting() must return TransferAccounting")
        resolved_owner_id = owner_id or getattr(transfer_owner, "owner_id", None)
        if resolved_owner_id is None:
            device = getattr(transfer_owner, "device", "unknown")
            owner_type = type(transfer_owner)
            resolved_owner_id = (
                f"{owner_type.__module__}.{owner_type.__qualname__}:{device}"
            )
        if not isinstance(resolved_owner_id, str) or not resolved_owner_id.strip():
            raise ValueError("transfer owner_id must be non-empty")
        return cls(
            transfer_owner=transfer_owner,
            before=before,
            owner_id=resolved_owner_id,
            config=config or ReadyPathQualificationConfig(),
        )

    def transfer_delta(self, transfer_owner: Any | None = None) -> ReadyPathTransferDelta:
        owner = self._transfer_owner if transfer_owner is None else transfer_owner
        if owner is not self._transfer_owner:
            raise ValueError("qualification must finish with the same transfer owner")
        after = owner.accounting()
        if not isinstance(after, TransferAccounting):
            raise TypeError("transfer_owner accounting() must return TransferAccounting")
        names = (
            "h2d_ops",
            "h2d_bytes",
            "d2h_ops",
            "d2h_bytes",
            "pool_waits",
            "explicit_sync_waits",
        )
        values: dict[str, int] = {}
        for name in names:
            value = int(getattr(after, name)) - int(getattr(self._before, name))
            if value < 0:
                raise ValueError(f"transfer accounting regressed for {name}")
            values[name] = value
        return ReadyPathTransferDelta(owner_id=self._owner_id, **values)

    def finish(
        self,
        *,
        selection: ReadyOnlyTickSelection,
        selector_diagnostics: ReadyOnlyTickDiagnostics,
        cameras: Iterable[ReadyCameraBinding],
        lineages: Iterable[ReadyPathLineageJoin],
        interception: ReadyPathInterceptionScope,
    ) -> ReadyPathTickEvidence:
        if self._finished:
            raise RuntimeError("qualification probe has already finished")
        self._finished = True
        if not isinstance(selection, ReadyOnlyTickSelection):
            raise TypeError("selection must be ReadyOnlyTickSelection")
        if not isinstance(selector_diagnostics, ReadyOnlyTickDiagnostics):
            raise TypeError("selector_diagnostics must be ReadyOnlyTickDiagnostics")
        if not isinstance(interception, ReadyPathInterceptionScope):
            raise TypeError("interception must be ReadyPathInterceptionScope")
        operations = interception.snapshot(
            configured_transfer_owner=self._transfer_owner,
        )
        cameras = tuple(cameras)
        if not all(isinstance(camera, ReadyCameraBinding) for camera in cameras):
            raise TypeError("cameras must contain ReadyCameraBinding values")
        lineages = tuple(lineages)
        if not all(isinstance(item, ReadyPathLineageJoin) for item in lineages):
            raise TypeError("lineages must contain ReadyPathLineageJoin values")

        transfer = self.transfer_delta()
        selected_sources = tuple(lease.source_identity for lease in selection.leases)
        camera_sources = tuple(camera.source_identity for camera in cameras)
        lineage_sources = tuple(item.source for item in lineages)
        failures: list[str] = []

        required = self._config.required_ready_cameras
        if len(cameras) != required:
            failures.append(f"ready_camera_count={len(cameras)}!={required}")
        if selection.starved:
            failures.append("ready_selection_starved")
        if camera_sources != selected_sources:
            failures.append("camera_source_order_mismatch")
        if any(not camera.lease.is_ready() for camera in cameras):
            failures.append("camera_lease_not_ready")
        if lineage_sources != selected_sources:
            failures.append("lineage_source_order_mismatch")
        if len(lineages) != len(cameras):
            failures.append("lineage_camera_count_mismatch")
        if lineages:
            if len({item.transaction_id for item in lineages}) != 1:
                failures.append("lineage_transaction_ids_not_unique")
            if len({item.plan_id for item in lineages}) != 1:
                failures.append("lineage_plan_ids_not_unique")
        for camera, lineage in zip(cameras, lineages):
            token = camera.control_token
            if lineage.control_sequence != token.sequence:
                failures.append("lineage_control_sequence_mismatch")
                break
        for camera, lineage in zip(cameras, lineages):
            token = camera.control_token
            if token.snapshot_id is None or lineage.policy_snapshot_id is None:
                failures.append("lineage_policy_snapshot_missing")
                break
            if lineage.policy_snapshot_id != token.snapshot_id:
                failures.append("lineage_policy_snapshot_mismatch")
                break

        for name in (
            "cpu_materializations",
            "cold_decodes",
            "wait_calls",
            "future_result_calls",
            "h2d_entrypoint_calls",
            "unattributed_transfer_calls",
            "mixed_transfer_owner_calls",
        ):
            value = getattr(operations, name)
            if value:
                failures.append(f"{name}={value}")
        if transfer.h2d_ops:
            failures.append(f"tick_h2d_ops={transfer.h2d_ops}")
        if transfer.h2d_bytes:
            failures.append(f"tick_h2d_bytes={transfer.h2d_bytes}")
        if transfer.pool_waits:
            failures.append(f"transfer_pool_waits={transfer.pool_waits}")
        if transfer.explicit_sync_waits:
            failures.append(f"explicit_sync_waits={transfer.explicit_sync_waits}")

        return ReadyPathTickEvidence(
            selected_sources=selected_sources,
            ready_camera_count=len(cameras),
            fallback_to_replay=selection.fallback_to_replay,
            starved=selection.starved,
            selection_count=selector_diagnostics.selection_count,
            fallback_count=selector_diagnostics.fallback_count,
            starvation_count=selector_diagnostics.starvation_count,
            lineages=lineages,
            operations=operations,
            transfer=transfer,
            failures=tuple(failures),
        )


__all__ = [
    "ReadyPathInterceptionKind",
    "ReadyPathInterceptionScope",
    "ReadyPathInterceptionTarget",
    "ReadyPathLineageJoin",
    "ReadyPathOperationCounts",
    "ReadyPathQualificationConfig",
    "ReadyPathQualificationProbe",
    "ReadyPathTickEvidence",
    "ReadyPathTransferDelta",
]
