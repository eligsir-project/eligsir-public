"""Issue #326 -- runtime trace of PROPOSED vs APPLIED vs CANCELLED Gaussian ops.

#325 (PR #543) produced the *static* writer/owner registry: which source
locations write which Gaussian state, under which declared owner. It proves a
write **exists in source**, never that it **executes**, at what cadence, or
whether a later writer cancels it. This module is the runtime half: it records
one event per (mechanism, operation) actually executed, so a conflict can be
*measured* instead of inferred.

The vocabulary (``MECHANISMS`` / ``OPERATIONS``) is deliberately the SAME
vocabulary #543's live scanner uses for ``declared_owner`` /
``semantic_operation``. ``tests/test_gaussian_op_trace.py`` asserts the exact
join against ``docs/architecture/gaussian_writer_vocabulary.json``; matching a
runtime event to a concrete writer still requires the exact-checkout scan.

DESIGN INVARIANTS
-----------------
1. **Torch-free.** Everything here takes plain ints/floats/strs/dicts. The
   tensor->scalar reduction happens at the call site, lexically inside the
   ``if tracer is not None:`` guard (see ``gaussian_op_probe``). This keeps the
   module collectable in the lean CPU CI image AND keeps every GPU sync
   (``.item()``/``.cpu()``/``.clone()``) off the disabled path -- a sync outside
   the guard would cost real time on every iteration whether tracing is on or
   not, which is the actual zero-overhead risk here (a bool check is not).

2. **Stateless, deterministic watch set.** A watched UID is ``uid % stride == 0``
   -- stable as the population grows, needs no bookkeeping, auto-admits newly
   created Gaussians, and guarantees every mechanism emits for the *same*
   subset. Same-UID conflict detection is only meaningful if all mechanisms
   agree on the subset; a reservoir/random sample would not.

3. **Never fabricate a field.** A field that cannot be honestly captured for an
   operation class is ``None`` and the event carries ``unattributed_reason``
   saying why. See ``UNATTRIBUTABLE`` below.

ENABLING (env-var gated for now; see the PR for why not argparse this cycle)
---------------------------------------------------------------------------
    ELIGSIR_GAUSSIAN_OP_TRACE=1               # master switch, default off
    ELIGSIR_GAUSSIAN_OP_TRACE_OUT=<path.jsonl>
    ELIGSIR_GAUSSIAN_OP_TRACE_UID_STRIDE=1024 # watch uid % stride == 0
    ELIGSIR_GAUSSIAN_OP_TRACE_EVERY=1         # only trace iteration % every == 0
    ELIGSIR_GAUSSIAN_OP_TRACE_MAX_EVENTS=2000000
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Vocabulary -- kept identical to #543's registry columns.
# --------------------------------------------------------------------------

#: ``declared_owner`` values from docs/architecture/gaussian_writer_vocabulary.json
MECHANISMS = frozenset({
    "mcmc",
    "source_depth",
    "lifecycle",
    "anchors_depth_locks",
    "freespace_contradiction",
    "mutation_transaction",
    "map_frame_correction",
    "loop_closure_merge",
    "optimizer",
    "model_class",
    "checkpoint_io",
    "none",
})

#: ``semantic_operation`` values from the same vocabulary contract.
OPERATIONS = frozenset({
    "grow",
    "retire",
    "relocate",
    "move_xyz",
    "opacity",
    "scale",
    "rotation",
    "support_update",
    "lifecycle_transition",
    "anchor_reference_update",
    "coordinate_correction",
    # The arbiter's own boundary operation: converting a deterministically
    # ordered proposal set into ONE mutation transaction. Added to the registry
    # by #1143 alongside GaussianMutationArbiter.commit, but the checked-in
    # registry artifact was not regenerated at the time, so this vocabulary was
    # never updated to match and the mismatch stayed latent until the next
    # regeneration (PR #1142).
    "topology_transaction",
    "identity_provenance_update",
    "create_bootstrap",
    "optimizer_state_update",
    "parameter_storage",
    "buffer_alloc",
    "checkpoint_restore",
    "strategy_dispatch",
    "not_a_gaussian_writer",
})

#: Event phases. ``proposed`` and ``applied`` are the two halves #326 asks to
#: separate; ``cancelled`` is an applied delta that a LATER mechanism undid in
#: the same step; ``candidate`` is a mechanism nominating rows it did not (yet)
#: act on -- that is what makes death-candidate OVERLAP measurable at all.
PHASES = frozenset({"proposed", "applied", "cancelled", "candidate", "bracket"})

SCOPES = frozenset({"uid", "aggregate"})

#: Operation classes whose *proposed* delta cannot be captured without
#: restructuring an owner boundary, with the reason. Emitting a plausible-but-
#: fabricated ``proposed_delta`` for these would be worse than emitting None.
UNATTRIBUTABLE: dict[str, str] = {
    "optimizer/move_xyz":
        "Adam/selective_adam writes params in a fused native kernel; the "
        "per-mechanism proposal is not observable from Python. Only a "
        "before/after BRACKET around optimizer.step() is real -- so this is "
        "emitted with phase='bracket' and proposed_delta=None.",
    "mcmc/relocate":
        "gsplat.strategy.ops.compute_relocation + _update_param_with_optimizer "
        "compute and write the new state inside the same native/vendored call; "
        "no proposal is materialised in Python before it is applied. Bracketed "
        "applied-delta only.",
    "mcmc/grow":
        "Growth appends rows; there is no pre-existing state_before for a row "
        "that did not exist. state_before is None BY CONSTRUCTION, not by "
        "capture failure.",
}


@dataclass(frozen=True)
class GaussianOpEvent:
    """One (mechanism, operation) actually executed at runtime.

    Field-for-field, this is #326's "operation event schema". Fields that are
    ``None`` are *honestly unknown*, never zero-filled.
    """

    seq: int
    iteration: int
    op: str
    mechanism: str
    phase: str
    scope: str
    topology_version: int
    frame: int | None = None
    uid: int | None = None
    root_uid: int | None = None
    n_rows: int | None = None
    state_before: dict[str, Any] | None = None
    proposed_delta: dict[str, Any] | None = None
    applied_delta: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    blocked_by: str | None = None
    candidate_uids: list[int] | None = None
    unattributed_reason: str | None = None
    #: Filled in by the derivation pass, not by the recorder: the mechanism that
    #: touched this same uid+state LATER in the same iteration.
    modified_again_by: str | None = None

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None and v != {} }
        return json.dumps(d, separators=(",", ":"), sort_keys=True)


class VocabularyError(ValueError):
    """A recorded event used an op/mechanism/phase outside #543's vocabulary."""


class GaussianOpTracer:
    """Collects :class:`GaussianOpEvent` rows and streams them to JSONL.

    Not thread-safe by design: every hook site is on the training thread. The
    async save worker never touches this.
    """

    def __init__(
        self,
        out_path: str | None = None,
        uid_stride: int = 1024,
        trace_every: int = 1,
        max_events: int = 2_000_000,
        strict_vocabulary: bool = True,
    ) -> None:
        self.out_path = out_path
        self.uid_stride = max(1, int(uid_stride))
        self.trace_every = max(1, int(trace_every))
        self.max_events = max(1, int(max_events))
        self.strict_vocabulary = bool(strict_vocabulary)
        self._seq = 0
        self._dropped = 0
        self.events: list[GaussianOpEvent] = []
        self._fh = None
        #: Sampled once per iteration by the training loop; every event records it.
        self.gate_state: dict[str, Any] = {}
        self.current_frame: int | None = None
        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            self._fh = open(out_path, "w", encoding="utf-8")

    # -- watch set ---------------------------------------------------------

    def watches(self, uid: int) -> bool:
        """Deterministic, stateless membership. See design invariant 2."""
        return int(uid) % self.uid_stride == 0

    def traces_iteration(self, iteration: int) -> bool:
        return int(iteration) % self.trace_every == 0

    # -- recording ---------------------------------------------------------

    def set_gate_state(self, **kwargs: Any) -> None:
        """Record the scheduler gate as of this iteration (read-only sample)."""
        self.gate_state = {k: v for k, v in kwargs.items() if v is not None}

    def record(
        self,
        *,
        iteration: int,
        op: str,
        mechanism: str,
        phase: str,
        scope: str = "aggregate",
        topology_version: int = 0,
        **kwargs: Any,
    ) -> GaussianOpEvent | None:
        if self.strict_vocabulary:
            if mechanism not in MECHANISMS:
                raise VocabularyError(f"unknown mechanism {mechanism!r}")
            if op not in OPERATIONS:
                raise VocabularyError(f"unknown operation {op!r}")
            if phase not in PHASES:
                raise VocabularyError(f"unknown phase {phase!r}")
            if scope not in SCOPES:
                raise VocabularyError(f"unknown scope {scope!r}")
        if len(self.events) >= self.max_events:
            self._dropped += 1
            return None
        kwargs.setdefault("gate", dict(self.gate_state))
        kwargs.setdefault("frame", self.current_frame)
        if kwargs.get("unattributed_reason") is None:
            reason = UNATTRIBUTABLE.get(f"{mechanism}/{op}")
            if reason is not None and phase in ("bracket", "applied"):
                kwargs["unattributed_reason"] = reason
        ev = GaussianOpEvent(
            seq=self._seq,
            iteration=int(iteration),
            op=op,
            mechanism=mechanism,
            phase=phase,
            scope=scope,
            topology_version=int(topology_version),
            **kwargs,
        )
        self._seq += 1
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.to_json())
            self._fh.write("\n")
        return ev

    # -- lifecycle ---------------------------------------------------------

    @property
    def dropped(self) -> int:
        return self._dropped

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------
# Module-level singleton. `get_tracer()` returning None IS the off path.
# --------------------------------------------------------------------------

_TRACER: GaussianOpTracer | None = None


def get_tracer() -> GaussianOpTracer | None:
    return _TRACER


def set_tracer(tracer: GaussianOpTracer | None) -> GaussianOpTracer | None:
    global _TRACER
    previous, _TRACER = _TRACER, tracer
    return previous


_TRUTHY = ("1", "true", "yes", "on")


def _env_flag(env, name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def install_from_env(model_path: str | None = None, env=None):
    """Install the tracer iff ``ELIGSIR_GAUSSIAN_OP_TRACE`` is truthy.

    Returns the installed tracer, or ``None`` (leaving the module singleton
    untouched at ``None``) when tracing is off. Called ONCE at streaming-loop
    startup; the loop then holds the result in a local and only ever tests it
    for ``is not None``.
    """
    env = os.environ if env is None else env
    if not _env_flag(env, "ELIGSIR_GAUSSIAN_OP_TRACE"):
        return None
    out = env.get("ELIGSIR_GAUSSIAN_OP_TRACE_OUT") or (
        os.path.join(model_path, "gaussian_op_trace.jsonl") if model_path else None
    )
    tracer = GaussianOpTracer(
        out_path=out,
        uid_stride=int(env.get("ELIGSIR_GAUSSIAN_OP_TRACE_UID_STRIDE", "1024") or 1024),
        trace_every=int(env.get("ELIGSIR_GAUSSIAN_OP_TRACE_EVERY", "1") or 1),
        max_events=int(env.get("ELIGSIR_GAUSSIAN_OP_TRACE_MAX_EVENTS", "2000000") or 2_000_000),
    )
    set_tracer(tracer)
    return tracer


def load_events(path: str) -> list[GaussianOpEvent]:
    """Read a JSONL trace back into events (for the derivation CLI/tests)."""
    out: list[GaussianOpEvent] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(GaussianOpEvent(**json.loads(line)))
    return out
