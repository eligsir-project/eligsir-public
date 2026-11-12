"""Level-D mutation-trace freezing for the W3-02 engine OFF/ON A/B.

Topology mutation (insert / prune / relocate / reseed) is deliberately
*outside* ``GaussianOptimizationEngine`` (doc §7.4, "Explicitly exclude
topology mutation") -- both the engine-OFF inline path and the engine-ON
``run_streaming_optimization_step`` path call the exact same, single
mutation functions afterward (``_flush_insertion_batch`` for depth
insertion, ``mcmc_strategy.step_post_backward`` for MCMC growth/relocation/
pruning). That sharing is what makes "freezing the mutation trace" tractable
here: the two arms are not two different implementations of mutation that
happen to agree, they are literally the same call with the same RNG stream
and the same upstream tensors -- *if* (and only if) the optimization step
that produced those upstream tensors was itself numerically identical.

This module does not replace or fork that mutation logic. It wraps the two
shared call sites with a recorder/verifier so a Level-D claim ("matched
Gaussian-count/topology trajectory under the frozen mutation trace") is
backed by an active per-iteration check instead of an unverified assumption
that determinism held:

- ``mode="record"`` (baseline/engine-OFF run): after each wrapped call,
  append one JSONL event: iteration, op, topology_version before/after,
  N before/after, and a cheap content hash of the mutation's index/value
  tensors (pruned indices, inserted xyz, relocated indices -- whatever the
  call site provides).
- ``mode="verify"`` (treatment/engine-ON run): read the previously recorded
  trace and, after each wrapped call, assert the just-produced event is
  *identical* to the recorded one for that iteration. Any mismatch raises
  ``MutationTraceDivergence`` immediately -- the run does not silently
  continue on a diverged trace and does not get to claim
  ``mutation_trace_applied=True`` for the events after the divergence.

A trace is only "applied" (in the w3_02_runtime.py evidence sense) once
every iteration recorded on one side has a matching verified event on the
other side, with zero divergences and zero missing events.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class MutationTraceDivergence(RuntimeError):
    """Raised when a 'verify' run's mutation event doesn't match the recorded trace."""


def _tensor_hash(value: Any) -> str:
    """Cheap, order-sensitive content hash for an index/value tensor.

    Returns ``"none"`` for ``None`` (nothing happened at this call) so that
    "no-op this iteration" is itself part of what gets compared, not silently
    skipped.
    """
    if value is None:
        return "none"
    try:
        import torch

        if isinstance(value, torch.Tensor):
            data = value.detach().to("cpu").contiguous()
            digest = hashlib.sha256(data.numpy().tobytes()).hexdigest()
            return f"{tuple(data.shape)}:{digest}"
    except ImportError:
        pass
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    return digest


@dataclass
class MutationEvent:
    """A single topology mutation event recorded at a call site.

    Attributes:
        iteration: Training iteration at which the mutation occurred.
        op: Operation type (e.g., 'insert', 'prune', 'relocate', 'reseed').
        topology_version_before: Mutation event counter before the operation.
        topology_version_after: Mutation event counter after the operation.
        n_before: Gaussian population size before the operation.
        n_after: Gaussian population size after the operation.
        content_hash: Cheap hash of the mutation's content (indices/values/etc.),
            used to verify identical mutations between runs.
    """

    iteration: int
    op: str
    topology_version_before: int
    topology_version_after: int
    n_before: int
    n_after: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert event to a JSON-serializable dictionary.

        Returns:
            Dictionary with keys: iteration, op, topology_version_before,
            topology_version_after, n_before, n_after, content_hash.
        """
        return {
            "iteration": self.iteration,
            "op": self.op,
            "topology_version_before": self.topology_version_before,
            "topology_version_after": self.topology_version_after,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MutationEvent":
        """Reconstruct a MutationEvent from a dictionary.

        Args:
            payload: Dictionary with the event's fields, typically from JSON.

        Returns:
            A MutationEvent instance.
        """
        return cls(
            iteration=int(payload["iteration"]),
            op=str(payload["op"]),
            topology_version_before=int(payload["topology_version_before"]),
            topology_version_after=int(payload["topology_version_after"]),
            n_before=int(payload["n_before"]),
            n_after=int(payload["n_after"]),
            content_hash=str(payload["content_hash"]),
        )


@dataclass
class MutationTraceRecorder:
    """Wraps the shared mutation call sites for one streaming run.

    ``mode="off"`` is a true no-op (default): zero overhead, zero file I/O,
    matches every run that isn't part of a Level-D W3-02 evidence pair.
    """

    mode: str = "off"
    path: str | Path = ""
    _events: list[MutationEvent] = field(default_factory=list, repr=False)
    _replay: dict[int, list[MutationEvent]] | None = field(default=None, repr=False)
    _replay_cursor: dict[int, int] = field(default_factory=dict, repr=False)
    divergences: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "record", "verify"}:
            raise ValueError(f"unknown mutation trace mode: {self.mode!r}")
        if self.mode == "verify":
            if not self.path:
                raise ValueError("--w3_02_mutation_trace_events is required for mode='verify'")
            self._replay = {}
            for row in _read_jsonl(Path(self.path)):
                event = MutationEvent.from_dict(row)
                self._replay.setdefault(event.iteration, []).append(event)

    @property
    def enabled(self) -> bool:
        """True if mutation tracing is active (record or verify mode)."""
        return self.mode != "off"

    def wrap(
        self,
        *,
        iteration: int,
        op: str,
        topology_version_before: int,
        topology_version_after: int,
        n_before: int,
        n_after: int,
        content: Any,
    ) -> MutationEvent:
        """Record or verify one mutation-call-site event. No-op when disabled."""
        event = MutationEvent(
            iteration=iteration,
            op=op,
            topology_version_before=topology_version_before,
            topology_version_after=topology_version_after,
            n_before=n_before,
            n_after=n_after,
            content_hash=_tensor_hash(content),
        )
        if self.mode == "off":
            return event
        if self.mode == "record":
            self._events.append(event)
            return event
        # mode == "verify"
        assert self._replay is not None
        cursor = self._replay_cursor.get(iteration, 0)
        candidates = self._replay.get(iteration, [])
        if cursor >= len(candidates):
            raise MutationTraceDivergence(
                f"iteration {iteration}: treatment run produced op={op!r} but the "
                "recorded baseline trace has no (more) events at this iteration"
            )
        expected = candidates[cursor]
        self._replay_cursor[iteration] = cursor + 1
        if event != expected:
            raise MutationTraceDivergence(
                f"iteration {iteration}: mutation event diverged from the frozen "
                f"baseline trace.\n  recorded : {expected.to_dict()}\n  observed : "
                f"{event.to_dict()}"
            )
        return event

    def finalize(self) -> dict[str, Any]:
        """Flush a 'record' trace to disk, or summarize a 'verify' run.

        Returns a small provenance dict; callers persist it into their own
        evidence payload (see ``w3_02_runtime.py``).
        """
        if self.mode == "off":
            return {"mutation_trace_mode": "off"}
        if self.mode == "record":
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as handle:
                for event in self._events:
                    handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            return {
                "mutation_trace_mode": "record",
                "mutation_trace_path": str(target),
                "mutation_trace_event_count": len(self._events),
            }
        # mode == "verify"
        assert self._replay is not None
        recorded_count = sum(len(v) for v in self._replay.values())
        observed_count = sum(self._replay_cursor.values())
        missing = recorded_count - observed_count
        applied = missing == 0 and not self.divergences
        if missing:
            self.divergences.append(
                f"{missing} recorded baseline event(s) were never matched by a "
                "treatment-run event (treatment run likely ended earlier or hit "
                "fewer mutation call sites)"
            )
        return {
            "mutation_trace_mode": "verify",
            "mutation_trace_path": str(self.path),
            "mutation_trace_recorded_event_count": recorded_count,
            "mutation_trace_observed_event_count": observed_count,
            "mutation_trace_applied": applied,
            "mutation_trace_divergences": list(self.divergences),
        }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"mutation trace file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
