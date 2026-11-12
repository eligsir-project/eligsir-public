"""Deterministic in-memory branching boundary for causal mapper experiments.

This module is core-owned: downstream simulators may depend on it, while core
EliGSiR never imports simulator code.  The Gaussian backend remains the sole
owner of map/model state through ``checkpoint_state``/``restore_checkpoint``.
Other continuation-visible mutable owners register explicit state providers.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import pickle
import random
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch

from eligsir.map.contracts import CheckpointBlob, GaussianMapBackend


@dataclass(frozen=True)
class ProcessRNGState:
    """Process-global RNG state used by legacy/current mapper paths."""

    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]

    @property
    def fingerprint(self) -> str:
        """Return a SHA256 fingerprint of the RNG state."""
        digest = hashlib.sha256()
        digest.update(pickle.dumps(self.python, protocol=5))
        digest.update(pickle.dumps(self.numpy, protocol=5))
        digest.update(self.torch_cpu.detach().cpu().numpy().tobytes())
        for state in self.torch_cuda:
            digest.update(state.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


def capture_process_rng() -> ProcessRNGState:
    """Capture the current state of all RNG subsystems."""
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return ProcessRNGState(
        python=copy.deepcopy(random.getstate()),
        numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=cuda_states,
    )


def restore_process_rng(state: ProcessRNGState) -> None:
    """Restore all RNG subsystems to a prior state."""
    random.setstate(copy.deepcopy(state.python))
    np.random.set_state(copy.deepcopy(state.numpy))
    torch.random.set_rng_state(state.torch_cpu.clone())
    if state.torch_cuda:
        torch.cuda.set_rng_state_all([item.clone() for item in state.torch_cuda])


def _payload_fingerprint(payload: Any) -> str:
    """Best-effort provider fingerprint; providers may supply a cheaper one."""
    try:
        encoded = pickle.dumps(payload, protocol=5)
    except Exception:
        encoded = repr(payload).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BranchComponentState:
    """Opaque state owned by one non-map runtime component."""

    name: str
    payload: Any
    fingerprint: str


class RuntimeStateProvider(Protocol):
    """Explicit owner hook for continuation-visible mutable runtime state."""

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    def capture_branch_state(self) -> BranchComponentState:
        """Capture the current branch state."""
        ...

    def restore_branch_state(self, state: BranchComponentState) -> None:
        """Restore the saved branch state."""
        ...


@dataclass
class AttributeStateProvider:
    """Small adapter for owners whose branch-visible state is explicit attrs.

    Prefer a custom provider when an owner has a native capture/restore contract
    or when deep-copying would be unnecessarily expensive.
    """

    owner: Any
    attributes: tuple[str, ...]
    provider_name: str

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self.provider_name

    def capture_branch_state(self) -> BranchComponentState:
        """Capture the current branch state from owned attributes."""
        payload = {name: copy.deepcopy(getattr(self.owner, name)) for name in self.attributes}
        return BranchComponentState(
            name=self.name,
            payload=payload,
            fingerprint=_payload_fingerprint(payload),
        )

    def restore_branch_state(self, state: BranchComponentState) -> None:
        """Restore the branch state to owned attributes."""
        if state.name != self.name:
            raise ValueError(f"provider mismatch: expected {self.name}, got {state.name}")
        for name in self.attributes:
            if name not in state.payload:
                raise KeyError(f"missing branch attribute {self.name}.{name}")
            setattr(self.owner, name, copy.deepcopy(state.payload[name]))


@dataclass
class CallableStateProvider:
    """Owner-safe adapter around an existing native capture/restore API."""

    provider_name: str
    capture_fn: Callable[[], Any]
    restore_fn: Callable[[Any], None]
    restore_live_fn: Callable[[Any], None] | None = None
    fingerprint_fn: Callable[[Any], str] | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self.provider_name

    def capture_branch_state(self) -> BranchComponentState:
        """Capture the current branch state using the capture function."""
        payload = self.capture_fn()
        fingerprint = (
            self.fingerprint_fn(payload)
            if self.fingerprint_fn is not None
            else _payload_fingerprint(payload)
        )
        return BranchComponentState(self.name, payload, fingerprint)

    def restore_branch_state(self, state: BranchComponentState) -> None:
        """Restore the branch state using the restore function."""
        if state.name != self.name:
            raise ValueError(f"provider mismatch: expected {self.name}, got {state.name}")
        self.restore_fn(state.payload)

    def restore_live_branch_state(self, state: BranchComponentState) -> None:
        """Restore a rejected live attempt without reusing a version fence."""

        if state.name != self.name:
            raise ValueError(f"provider mismatch: expected {self.name}, got {state.name}")
        (self.restore_live_fn or self.restore_fn)(state.payload)


@dataclass(frozen=True)
class BranchState:
    """In-memory state captured from this branch's registered owners."""

    backend: CheckpointBlob
    components: tuple[BranchComponentState, ...]
    rng: ProcessRNGState
    metadata: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        """Return a SHA256 fingerprint of the branch state identity."""
        # Deliberately do not hash the opaque backend payload: doing so can force
        # giant GPU->CPU transfers. Backend versions + component/RNG identities
        # are a cheap continuation-state fingerprint; outcome parity is the
        # authoritative restoration test for map contents.
        identity = {
            "backend_layout": self.backend.layout,
            "map_version": int(self.backend.map_version),
            "topology_version": int(self.backend.topology_version),
            "components": [(item.name, item.fingerprint) for item in self.components],
            "rng": self.rng.fingerprint,
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()


class BranchableMapperRuntime:
    """Stable downstream-facing branch/checkpoint/evaluate/ingest surface.

    ``ingest_fn`` must be the production mapper input/update path selected by
    the caller. ``evaluate_fn`` must evaluate an explicit fixed view set and be
    read-only. The runtime deliberately does not invent a second map checkpoint
    format; it delegates map ownership to ``GaussianMapBackend``.
    """

    def __init__(
        self,
        *,
        backend: GaussianMapBackend,
        training_args: Any,
        ingest_fn: Callable[[Any], Any],
        evaluate_fn: Callable[[], Mapping[str, float]],
        state_providers: Sequence[RuntimeStateProvider] = (),
        evaluation_id: str = "fixed",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.training_args = training_args
        self._ingest_fn = ingest_fn
        self._evaluate_fn = evaluate_fn
        self.providers = tuple(state_providers)
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("runtime state provider names must be unique")
        self.evaluation_id = str(evaluation_id)
        self.metadata = dict(metadata or {})

    def checkpoint_state(self) -> BranchState:
        """Capture the current checkpoint state."""
        components = tuple(provider.capture_branch_state() for provider in self.providers)
        return BranchState(
            backend=self.backend.checkpoint_state(),
            components=components,
            rng=capture_process_rng(),
            metadata={"evaluation_id": self.evaluation_id, **self.metadata},
        )

    # Short alias used by generic branch adapters.
    checkpoint = checkpoint_state

    def restore_state(self, state: BranchState) -> None:
        """Restore the runtime to a previously captured checkpoint state."""
        expected = [provider.name for provider in self.providers]
        actual = [component.name for component in state.components]
        if actual != expected:
            raise ValueError(f"branch provider layout mismatch: expected {expected}, got {actual}")
        self.backend.restore_checkpoint(state.backend, self.training_args)
        for provider, component in zip(self.providers, state.components, strict=True):
            provider.restore_branch_state(component)
        restore_process_rng(state.rng)

    restore = restore_state

    def restore_live_state(self, state: BranchState) -> None:
        """Restore a rejected live branch through monotonic backend fences.

        Normal checkpoint restore is deliberately exact for persistence and
        read-only branch checks.  A live rejection instead restores accepted
        numerical values while advancing the live version boundary so late
        GPU completions and frozen plans cannot alias the restored map.
        """

        expected = [provider.name for provider in self.providers]
        actual = [component.name for component in state.components]
        if actual != expected:
            raise ValueError(f"branch provider layout mismatch: expected {expected}, got {actual}")
        restore_live = getattr(self.backend, "restore_live_checkpoint", None)
        if not callable(restore_live):
            raise RuntimeError("live branch restore requires monotonic backend restore")
        restore_live(state.backend, self.training_args)
        for provider, component in zip(self.providers, state.components, strict=True):
            restore_provider = getattr(
                provider, "restore_live_branch_state", provider.restore_branch_state
            )
            restore_provider(component)
        restore_process_rng(state.rng)

    def ingest(self, observation: Any) -> Any:
        """Execute the caller-supplied real production ingest/update path."""
        return self._ingest_fn(observation)

    def evaluate_fixed_views(self, *, verify_read_only: bool = False) -> dict[str, float]:
        """Evaluate fixed views and optionally verify read-only semantics."""
        before: BranchState | None = None
        if verify_read_only:
            before = self.checkpoint_state()
        metrics = {str(key): float(value) for key, value in self._evaluate_fn().items()}
        if verify_read_only and before is not None:
            after = self.checkpoint_state()
            if after.fingerprint != before.fingerprint:
                self.restore_state(before)
                raise RuntimeError("fixed-view evaluation mutated continuation-visible runtime state")
        return metrics

    evaluate = evaluate_fixed_views

    def state_fingerprint(self) -> str:
        """Return the state fingerprint."""
        return self.checkpoint_state().fingerprint

    @contextmanager
    def restored_branch(self, state: BranchState) -> Iterator["BranchableMapperRuntime"]:
        """Temporarily execute from ``state`` and always restore caller state."""
        original = self.checkpoint_state()
        self.restore_state(state)
        try:
            yield self
        finally:
            self.restore_state(original)
