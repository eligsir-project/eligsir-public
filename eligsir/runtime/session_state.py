"""The streaming session state container (#1075 follow-up, owner A -- "P4").

``train_streaming.py``'s ``streaming_training()`` defines ~45 nested closures
that reach their collaborators as **free variables** of the enclosing function
scope. An AST capture analysis over the current function (L1686-12035) found
**94 distinct captured enclosing-scope names**; ``args`` is captured by 16
closures and ``gaussians`` by 8. That implicit capture is what makes the
function irreducible: a closure's dependencies are invisible at its definition
site and cannot be handed to it from outside.

This module replaces implicit capture with **one explicit object**. A closure
that today reads a free ``streaming_scene`` instead reads
``state.streaming_scene``, and can be lifted out of the function entirely
because its input is now a parameter rather than a cell.

Relationship to ``RuntimeStateProvider``
========================================

Branch/continuation state uses the **existing** contract in
``eligsir/mapping/branching.py`` -- ``RuntimeStateProvider`` (``:84``),
``AttributeStateProvider`` (``:96``), ``CallableStateProvider`` (``:129``). This
module does not invent a second one.

**The container is not itself a provider.** It *holds* slots and *exposes*
:meth:`SessionState.state_providers`. Conflating the two is what would produce a
parallel capture protocol.

The partition, and the trap it exists to prevent
================================================

Every slot is declared with a :class:`SlotClass`:

* :attr:`SlotClass.BACKEND_OWNED` -- the Gaussian model, CUDA tensors, anything
  whose state lives in ``GaussianMapBackend``'s checkpoint. **Never** exposed as
  a provider.
* :attr:`SlotClass.BOOKKEEPING` -- scalars and small scalar containers, safe for
  ``AttributeStateProvider`` (which does ``copy.deepcopy`` per attribute,
  ``branching.py:112``).
* :attr:`SlotClass.NATIVE_CAPTURE` -- collaborators with their own
  capture/restore, or too stateful/expensive to deepcopy.

**The trap:** ``AttributeStateProvider`` deepcopies whatever it is pointed at,
and ``BranchState.fingerprint`` (``branching.py:165-180``) deliberately skips the
backend payload because hashing it "can force giant GPU->CPU transfers". Point an
``AttributeStateProvider`` at the ``gaussians`` slot and you get a silent
multi-gigabyte GPU->CPU copy per branch point. :meth:`SessionState.state_providers`
therefore **refuses** to build a provider over a ``BACKEND_OWNED`` slot rather
than trusting each caller to remember.

Findings that shaped this, from the capture analysis
====================================================

1. **``report_block()``/``summary()``/``to_dict()`` are not capture APIs.** They
   are one-way reporting projections with no inverse. Of ~28 ``NATIVE_CAPTURE``
   collaborators, exactly **one** -- ``SeedFadeMonitor`` (``state_dict``
   ``utils/seed_fade_monitor.py:277`` / ``load_state_dict`` ``:314``) -- has a real
   round-trippable pair. Slots therefore declare ``capture_api`` explicitly and
   default to *not capturable*, with a stated reason. A container that silently
   treated ``summary()`` as capture would produce continuations that restore
   nothing while reporting success.
2. **Type does not determine the bucket.** ``_progress_frames`` is a plain
   ``list`` (bucket B by type) but is ``append()``-ed from a **daemon thread**
   while the main loop reads it; deepcopying it races a live producer. It, the
   ``_pv_cpu_queue`` and ``_async_state`` are ``NATIVE_CAPTURE`` on cross-thread
   grounds. ``_partition_control_roles`` is a small dataclass that holds *camera
   references*, hence CUDA tensors, so it is not bookkeeping either.
3. **Some slots are rebound, not just mutated, inside the loop** -- and three are
   rebound from *inside a closure* via ``nonlocal`` (``_deferred_stale_mask``,
   ``_deferred_stale_count``, ``_partition_control_roles``). Attribute assignment
   on this container is exactly the mutable holder those need: ``nonlocal x; x =
   v`` becomes ``state.x = v``, which is why slots are writable by default and
   :attr:`SlotSpec.rebound_in_loop` records which ones must stay so.
4. **No production code registers a state provider today.** The only users of
   ``AttributeStateProvider`` are in ``tests/test_branchable_mapper_runtime.py``.
   Every ``NATIVE_CAPTURE`` collaborator is currently un-branchable, and this
   module records that as declared, per-slot fact rather than leaving it to be
   rediscovered.

Scope
=====

This module defines the container and declares the slot registry. It does
**not** edit ``streaming_training()``; wiring the closures to it is the
integration owner's task.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from eligsir.mapping.branching import (
    AttributeStateProvider,
    CallableStateProvider,
    RuntimeStateProvider,
)


class SlotClass(str, Enum):
    """How a slot's state may (or may not) be captured for a branch."""

    #: Owned by ``GaussianMapBackend``'s checkpoint. Never provider-captured.
    BACKEND_OWNED = "backend_owned"
    #: Scalars / small scalar containers. ``AttributeStateProvider``-safe.
    BOOKKEEPING = "bookkeeping"
    #: Has its own capture/restore, or is too stateful/expensive to deepcopy.
    NATIVE_CAPTURE = "native_capture"


@dataclass(frozen=True, slots=True)
class CaptureAPI:
    """A collaborator's own round-trippable capture/restore method pair.

    Deliberately requires **both** directions. A reporting projection
    (``summary()``, ``report_block()``, ``to_dict()``) has no inverse and is not
    a capture API; declaring one here would produce a continuation that restores
    nothing while reporting success.

    Attributes:
        capture: Zero-argument method name returning an opaque payload.
        restore: One-argument method name accepting that payload.
    """

    capture: str
    restore: str


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """The declaration of one collaborator slot.

    Attributes:
        name: The attribute name, matching the free variable it replaces so the
            integration is a mechanical ``x`` -> ``state.x`` rewrite.
        slot_class: Which of the three capture classes this slot falls in.
        summary: One line on what the slot holds.
        capture_api: The collaborator's own capture/restore pair, when it
            genuinely has one. ``None`` means this slot cannot be branch-captured
            today -- see ``reason``.
        reason: Why the classification is what it is. Required for
            ``BACKEND_OWNED`` and for any ``NATIVE_CAPTURE`` slot without a
            ``capture_api``, because those are exactly the two cases where a
            future reader would otherwise assume an oversight.
        rebound_in_loop: The slot is reassigned (not merely mutated) inside the
            training loop, so it cannot become read-only.
        rebound_by_closure: The slot is reassigned from *inside* a closure via
            ``nonlocal``. Strictly stronger: the container must be the mutable
            holder the closure writes through.
        captured: Whether a provider should be built at all. Loop-invariant
            config (``args``, ``pipe``) is deepcopy-safe but capturing it is a
            no-op that only inflates the branch payload.
    """

    name: str
    slot_class: SlotClass
    summary: str
    capture_api: Optional[CaptureAPI] = None
    reason: str = ""
    rebound_in_loop: bool = False
    rebound_by_closure: bool = False
    captured: bool = True

    def __post_init__(self) -> None:
        if self.slot_class is SlotClass.BACKEND_OWNED and not self.reason:
            raise ValueError(
                f"slot {self.name!r}: BACKEND_OWNED requires a reason -- it is the "
                "classification that forbids provider capture, and an unexplained "
                "one invites a later 'this looks safe' reclassification"
            )
        if self.slot_class is SlotClass.BACKEND_OWNED and self.capture_api is not None:
            raise ValueError(
                f"slot {self.name!r}: a BACKEND_OWNED slot must not declare a "
                "capture API; the backend checkpoint owns this state"
            )
        if (
            self.slot_class is SlotClass.NATIVE_CAPTURE
            and self.capture_api is None
            and not self.reason
        ):
            raise ValueError(
                f"slot {self.name!r}: a NATIVE_CAPTURE slot with no capture API "
                "must state why it cannot be captured today"
            )
        if self.rebound_by_closure and not self.rebound_in_loop:
            raise ValueError(
                f"slot {self.name!r}: rebound_by_closure implies rebound_in_loop"
            )


class SlotError(KeyError):
    """Raised for an undeclared slot name."""


class BackendOwnedSlotError(RuntimeError):
    """Raised on an attempt to branch-capture backend-owned state.

    This is the guard described in the module docstring: it converts a silent
    multi-gigabyte GPU->CPU deepcopy into a loud, immediate failure.
    """


class SessionState:
    """Explicit holder for ``streaming_training()``'s collaborator slots.

    Replaces implicit closure capture. Slots are reachable as attributes so the
    integration is a mechanical rewrite (``streaming_scene`` ->
    ``state.streaming_scene``), and are writable so that the three slots today
    rebound via ``nonlocal`` have a holder to write through.

    Args:
        specs: The slot declarations. Defaults to :data:`STREAMING_SLOTS`.
        values: Initial values, by slot name. Slots left unset read as ``None``;
            they are *declared* either way, which is the point -- a typo reaching
            an undeclared name raises instead of silently creating a slot.

    Raises:
        SlotError: If ``values`` names a slot that was not declared.
    """

    def __init__(
        self,
        specs: Iterable[SlotSpec] = (),
        values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        specs = tuple(specs) or STREAMING_SLOTS
        by_name: Dict[str, SlotSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate slot declaration {spec.name!r}")
            by_name[spec.name] = spec
        object.__setattr__(self, "_specs", by_name)
        object.__setattr__(self, "_values", {name: None for name in by_name})

        for name, value in (values or {}).items():
            if name not in by_name:
                raise SlotError(
                    f"{name!r} is not a declared slot; declare it in the registry "
                    "rather than attaching it ad hoc, or the capture partition "
                    "silently stops covering the real state"
                )
            self._values[name] = value

    # -- attribute access -------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        try:
            values = object.__getattribute__(self, "_values")
        except AttributeError:  # pragma: no cover - during __init__ only
            raise AttributeError(name) from None
        if name in values:
            return values[name]
        raise AttributeError(
            f"{name!r} is not a declared slot on SessionState "
            f"({len(values)} slots declared)"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        values = object.__getattribute__(self, "_values")
        if name not in values:
            raise SlotError(
                f"cannot set undeclared slot {name!r}; add a SlotSpec for it. "
                "Silent slot creation is how the BACKEND_OWNED partition would "
                "stop covering a real collaborator."
            )
        values[name] = value

    # -- introspection ----------------------------------------------------

    @property
    def slot_names(self) -> Tuple[str, ...]:
        """Declared slot names, in declaration order."""
        return tuple(self._specs)

    def spec(self, name: str) -> SlotSpec:
        """Return one slot's declaration.

        Raises:
            SlotError: If the slot was not declared.
        """
        try:
            return self._specs[name]
        except KeyError:
            raise SlotError(f"{name!r} is not a declared slot") from None

    def slots_of(self, slot_class: SlotClass) -> Tuple[SlotSpec, ...]:
        """Return every declared slot in one capture class."""
        return tuple(s for s in self._specs.values() if s.slot_class is slot_class)

    def is_backend_owned(self, name: str) -> bool:
        """Whether this slot's state belongs to the ``GaussianMapBackend``."""
        return self.spec(name).slot_class is SlotClass.BACKEND_OWNED

    # -- the RuntimeStateProvider bridge ----------------------------------

    def state_providers(
        self,
        names: Optional[Iterable[str]] = None,
    ) -> Tuple[RuntimeStateProvider, ...]:
        """Build providers for the branch-capturable slots.

        ``BOOKKEEPING`` slots are grouped into ONE ``AttributeStateProvider``
        rather than one per slot: they are scalars, and a provider apiece would
        put dozens of near-empty entries in every ``BranchState`` fingerprint for
        no gain. ``NATIVE_CAPTURE`` slots with a declared ``capture_api`` each get
        a ``CallableStateProvider`` bound to that collaborator's own methods.

        Slots with ``captured=False`` (loop-invariant config), and
        ``NATIVE_CAPTURE`` slots with no ``capture_api``, are skipped. Skipping is
        silent only when it was *declared*; asking for one by name is an error, so
        a caller is never told "captured" about state that was not.

        Args:
            names: Restrict to these slots. ``None`` means every declared slot.

        Returns:
            Providers satisfying ``branching.RuntimeStateProvider``.

        Raises:
            BackendOwnedSlotError: If ``names`` includes a ``BACKEND_OWNED`` slot.
                The backend checkpoint owns that state; capturing it here would
                deepcopy a Gaussian model.
            SlotError: If ``names`` includes an undeclared slot, or a
                ``NATIVE_CAPTURE`` slot that has no capture API (asking for one
                explicitly must fail loudly rather than return nothing).
        """
        requested = tuple(names) if names is not None else self.slot_names
        explicit = names is not None

        # Backend-owned slots are EXCLUDED on the bulk path (that is what their
        # declaration means) but RAISE when named, because naming one is a
        # caller asserting it should be captured -- the exact belief that ends
        # in a Gaussian model being deepcopied off the GPU.
        if explicit:
            for name in requested:
                spec = self.spec(name)
                if spec.slot_class is SlotClass.BACKEND_OWNED:
                    raise BackendOwnedSlotError(
                        f"slot {name!r} is backend-owned ({spec.reason}); capture "
                        "it through GaussianMapBackend.checkpoint_state(), never "
                        "through a RuntimeStateProvider -- an "
                        "AttributeStateProvider would deepcopy it off the GPU"
                    )
        else:
            requested = tuple(
                name
                for name in requested
                if self.spec(name).slot_class is not SlotClass.BACKEND_OWNED
            )

        providers: list[RuntimeStateProvider] = []

        bookkeeping = tuple(
            name
            for name in requested
            if self.spec(name).slot_class is SlotClass.BOOKKEEPING
            and self.spec(name).captured
        )
        if bookkeeping:
            providers.append(
                AttributeStateProvider(
                    owner=self,
                    attributes=bookkeeping,
                    provider_name="streaming_session_bookkeeping",
                )
            )

        for name in requested:
            spec = self.spec(name)
            if spec.slot_class is not SlotClass.NATIVE_CAPTURE or not spec.captured:
                continue
            if spec.capture_api is None:
                if explicit:
                    raise SlotError(
                        f"slot {name!r} has no capture API ({spec.reason}); it "
                        "cannot be branch-captured today. Do not substitute "
                        "summary()/report_block() -- those are one-way reporting "
                        "projections with no inverse."
                    )
                continue
            providers.append(self._callable_provider(name, spec))

        return tuple(providers)

    def uncapturable(self) -> Tuple[Tuple[str, str], ...]:
        """Return ``(slot, reason)`` for declared state no provider covers.

        The honest complement of :meth:`state_providers`: a branch built from
        those providers restores everything *except* this. Making the gap
        enumerable is the point -- it is otherwise invisible at the call site.
        """
        gaps = []
        for spec in self._specs.values():
            if spec.slot_class is SlotClass.BACKEND_OWNED:
                continue  # covered by the backend checkpoint, not a gap
            if spec.captured and (
                spec.slot_class is SlotClass.BOOKKEEPING or spec.capture_api is not None
            ):
                continue
            gaps.append((spec.name, spec.reason or "declared not captured"))
        return tuple(gaps)

    def _callable_provider(self, name: str, spec: SlotSpec) -> RuntimeStateProvider:
        """Bind a ``CallableStateProvider`` to one collaborator's own methods.

        Resolution is deferred to call time so the container can be constructed
        before every collaborator exists -- which is the real order in
        ``streaming_training()``, where slots are filled across ~4000 lines of
        setup.
        """
        api = spec.capture_api
        assert api is not None  # guarded by the caller

        def _capture() -> Any:
            owner = getattr(self, name)
            if owner is None:
                raise RuntimeError(
                    f"slot {name!r} is unset; cannot capture branch state from it"
                )
            return getattr(owner, api.capture)()

        def _restore(payload: Any) -> None:
            owner = getattr(self, name)
            if owner is None:
                raise RuntimeError(
                    f"slot {name!r} is unset; cannot restore branch state into it"
                )
            getattr(owner, api.restore)(payload)

        return CallableStateProvider(
            provider_name=f"streaming_session_{name.lstrip('_')}",
            capture_fn=_capture,
            restore_fn=_restore,
        )


def _backend(name: str, summary: str, reason: str, **kw: Any) -> SlotSpec:
    return SlotSpec(name, SlotClass.BACKEND_OWNED, summary, reason=reason, **kw)


def _book(name: str, summary: str, **kw: Any) -> SlotSpec:
    return SlotSpec(name, SlotClass.BOOKKEEPING, summary, **kw)


def _native(name: str, summary: str, reason: str = "", **kw: Any) -> SlotSpec:
    return SlotSpec(name, SlotClass.NATIVE_CAPTURE, summary, reason=reason, **kw)


#: Reason shared by every collaborator whose only "state" method is a one-way
#: reporting projection. Named once because it is the single most common finding.
_REPORT_ONLY = (
    "no capture/restore pair; summary()/report_block()/to_dict() are one-way "
    "reporting projections with no inverse"
)

#: The declared slot registry, from the AST capture analysis of
#: ``streaming_training()``. Ordered by capture class, then by how many closures
#: capture the name.
STREAMING_SLOTS: Tuple[SlotSpec, ...] = (
    # -- BACKEND_OWNED ---------------------------------------------------
    _backend(
        "gaussians",
        "The Gaussian model (GsplatGaussianModel/GaussianModel). Captured by 8 closures.",
        "params, optimizer and lineage buffers are exactly what "
        "GaussianMapBackend.checkpoint_state()/restore_checkpoint() own",
    ),
    _backend(
        "background",
        "Background colour tensor on CUDA.",
        "CUDA tensor; deepcopy allocates VRAM per branch point for a value that "
        "never changes",
    ),
    _backend(
        "_deferred_stale_mask",
        "Deferred-prune mask produced by the post-render hook.",
        "CUDA bool tensor whose length is coupled to the Gaussian population; "
        "capturing it apart from the backend blob desynchronizes it from N",
        rebound_in_loop=True,
        rebound_by_closure=True,
    ),
    _backend(
        "viewpoint_cam",
        "The iteration's training camera.",
        "holds CUDA image/depth tensors under the residency ledger; it is "
        "per-iteration scratch, not continuation state",
        rebound_in_loop=True,
    ),
    # -- BOOKKEEPING -----------------------------------------------------
    _book(
        "args",
        "Frozen argparse Namespace. Captured by 16 closures, the most of any name.",
        captured=False,
        reason="loop-invariant configuration; capturing it inflates every branch "
        "payload without carrying any run state",
    ),
    _book(
        "pipe",
        "Pipeline params (streaming_training parameter).",
        captured=False,
        reason="loop-invariant configuration",
    ),
    _book("iteration", "Main training-loop counter.", rebound_in_loop=True),
    _book("first_iter", "Starting iteration; loop-invariant once training begins."),
    _book("n_frames_ingested", "Frames ingested so far.", rebound_in_loop=True),
    _book("total_inserted", "Gaussians added by Phase 2 depth insertion.", rebound_in_loop=True),
    _book("_clk_optimizer_step_id", "Issue #289 optimizer-step clock.", rebound_in_loop=True),
    _book("_clk_rasterizer_call_count", "Issue #703 rasterizer-call clock.", rebound_in_loop=True),
    _book("_clk_supervised_pixel_count", "Supervised-pixel clock.", rebound_in_loop=True),
    _book("_clk_supervision_view_count", "Supervision-view clock.", rebound_in_loop=True),
    _book(
        "_deferred_stale_count",
        "Length summary of _deferred_stale_mask.",
        rebound_in_loop=True,
        rebound_by_closure=True,
        reason="an int, but the summary of a BACKEND_OWNED mask -- capture both "
        "together or neither, or the count describes a mask that was not restored",
    ),
    _book("_per_source_supervision_counts", "Per-supervision-source success counts."),
    _book("_last_tracking_weight", "One-slot mutable box holding the last tracking weight."),
    _book("_soft_all_invariant", "SoftAllAdmitter invariant counters."),
    _book("_emergency_state", "Emergency-checkpoint handler state."),
    _book("_resumed_from_iteration", "Iteration a resume restarted from, if any."),
    _book("_replica_rgb_report", "Replica synthetic-RGB provenance block."),
    # -- NATIVE_CAPTURE, with a real capture/restore pair -----------------
    _native(
        "seed_fade_monitor",
        "SeedFadeMonitor; already rides the training checkpoint.",
        capture_api=CaptureAPI("state_dict", "load_state_dict"),
    ),
    # -- NATIVE_CAPTURE, no capture API today -----------------------------
    _native(
        "streaming_scene",
        "StreamingScene: cameras, occupancy, replay/reservoir stores.",
        "no capture/restore; its only de-facto continuation surface today is two "
        "ints hand-poked at resume (_source_idx, _keyframes_admitted). Needs a "
        "custom provider over eligsir/mapping/stores/",
    ),
    _native("frame_source", "Ordered frame source (possibly strided/live).",
            "live ingestion handle: sockets, decoders, buffered counts; cannot be duplicated"),
    _native("_observation_db", "FrameObservationDB with a bounded CUDA resolution cache.",
            "deepcopy would duplicate GPU caches; no capture/restore"),
    _native("_coordinator", "MappingCoordinator (four-clocks owner).", _REPORT_ONLY),
    _native("_cadence", "RealizedCadenceTracker; rolling windows.", _REPORT_ONLY),
    _native("_admission_cadence", "AdmissionCadenceRecorder.", _REPORT_ONLY),
    _native("_mapping_scheduler", "MappingScheduler (effective_spf owner).",
            "no capture/restore"),
    _native("_keyframe_admitter", "KeyframeAdmitter.",
            "has reset(), which is destructive rather than a restore; large internal "
            "state (keyframe indices, precedence traces, velocity estimate, timebase)"),
    _native("_soft_all_admitter", "SoftAllAdmitter.", _REPORT_ONLY),
    _native("_fixed_stride_admitter", "FixedStrideAdmitter.", "no capture/restore"),
    _native("_admission_fingerprint", "AdmissionFingerprint running digest.",
            "to_dict() is report-only; the digest accumulates over the whole run, so "
            "a wrong restore silently invalidates the run's admission fingerprint"),
    _native("_admission_rollout_policy", "AdmissionRolloutPolicy.", "no capture/restore"),
    _native("_shadow_utility_policy", "ShadowUtilityPolicy.",
            "owns an on-disk log_dir under model_path; a copy would double-write it"),
    _native("_reinit_tracker", "MapReinitTracker.",
            "its veto is permanent-for-the-run by design (ORB-SLAM3 map-reinit "
            "constraint); a naive restore could silently clear a hard veto"),
    _native("_orbslam_log_watcher", "OrbslamLogReinitWatcher.",
            "holds a file offset into orbslam_live.log; two copies double-count "
            "map-reinit lines"),
    _native("_quality_running_stats", "RunningBlurExposureStats EMA.",
            "deepcopy would work, but it belongs with the admitter it feeds"),
    _native("_tracking_running_stats", "RunningTrackingStats EMA.",
            "deepcopy would work, but it belongs with the admitter it feeds"),
    _native("_resolution_controller", "ResolutionController.", "no capture/restore"),
    _native("_resolution_tier_log", "ResolutionTierLog.", _REPORT_ONLY),
    _native("_submap_manager", "SubmapManager.", "coupled to the pose graph"),
    _native("_duty", "DutyCycleTimeline.", _REPORT_ONLY),
    _native("_mem_tracker", "GPUMemoryResidencyTracker.",
            "snapshot() samples live GPU state rather than capturing this object's "
            "own; report-only otherwise", rebound_in_loop=True),
    _native("_backpressure_owner", "BackpressureOwner (#601).", _REPORT_ONLY),
    _native("_partition_control_runtime", "PartitionControlRuntime.",
            "report_block(roles) takes its roles as an argument, so its own output "
            "is not self-contained"),
    _native("_partition_control_roles", "PartitionControlRoles for the current tick.",
            "a small dataclass, but it holds camera references and therefore CUDA "
            "tensors; deepcopy would clone camera images",
            rebound_in_loop=True, rebound_by_closure=True),
    _native("save_worker", "AsyncSaveWorker.", "background thread and work queue"),
    _native("tb_writer", "TensorBoard SummaryWriter.",
            "background flush thread and an open event file; a second writer on the "
            "same model_path corrupts the event stream"),
    _native("_pv_cpu_queue", "Bounded queue feeding the progress-video CPU worker.",
            "cross-thread: a live queue cannot be branch-copied"),
    _native("_progress_frames", "Accumulated progress-video frames.",
            "a plain list -- which is the trap. It is appended from a daemon thread "
            "while the main loop reads it, so deepcopy races a live producer"),
    _native("_async_state", "Async-ingest cursor shared with the ingestion worker.",
            "cross-thread: mutated on the IngestionWorker thread"),
)


__all__ = [
    "BackendOwnedSlotError",
    "CaptureAPI",
    "SessionState",
    "SlotClass",
    "SlotError",
    "SlotSpec",
    "STREAMING_SLOTS",
]
