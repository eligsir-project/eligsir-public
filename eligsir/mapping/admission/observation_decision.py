"""The explicit observation -> admission -> insertion decision contract (issue #600).

Why this exists
---------------
The streaming execution path historically conflated three genuinely distinct
decisions into one boolean:

1. **observation** -- does this source frame advance pose / history / the source
   clock? (owned by ``FrameScheduler`` release + ``StreamingScene`` ingest)
2. **supervision admission** -- does it become a training keyframe the mapper is
   obliged to assimilate? (owned by the admission gate)
3. **geometry insertion** -- may it create NEW Gaussians from its depth map?
   (owned by Phase 2 insertion)

Conflating them is most harmful at low ``effective_spf`` under
``--streaming_frame_admission all``: source coverage becomes a de-facto hard
requirement to materialize essentially every frame into expensive mapping state,
with no bound on the resulting mapping backlog.

``ObservationDecision`` makes all three independently observable and testable.

What this module is NOT
-----------------------
It is **not** a second frame scheduler or a second frame database, which #600
explicitly forbids. ``SoftAllAdmitter`` is an ordinary ``admission_fn`` for the
EXISTING ``StreamingScene.ingest_next_frame`` call site, in exactly the same
shape as ``KeyframeAdmitter``/``FixedStrideAdmitter``. It reuses the existing
machinery wholesale; the only new thing is that the decision is now explicit.

The two required invariants come out of that reuse rather than from new code,
which is why reuse was the right call:

* **a skipped frame still advances pose/source/history** -- a rejected frame
  takes ``streaming_scene.ingest_next_frame``'s existing reject path, which
  calls ``_cache_inactive_frame`` / ``_append_dense_trajectory(is_train=False)``
  / ``record_fn(..., is_train=False)`` before returning.
* **a skipped frame can never be depth-inserted** -- that same path returns
  ``is_train=False``, and Phase 2 insertion is gated on ``is_train``. There is
  exactly one place that can set ``allow_geometry_insertion=True``
  (``ObservationDecision.admitted``), and ``train_streaming`` asserts the
  invariant with a counter that must read 0.

Scope fence
-----------
The backlog *signal* here is deliberately the simplest honest one (see
``SoftAllAdmitter``). A richer bounded-backpressure controller is issue #601's
job and is explicitly NOT attempted here -- #600 owns the CONTRACT and the
soft-all/force-all split, #601 owns the policy. Building a sophisticated
controller under this issue would be exactly the scope creep #600 warns about.

This module is torch-free and import-side-effect-free, so it unit-tests in the
lean CPU CI image -- the same scope fence ``utils/execution_semantics.py`` and
``utils/experiment_validity.py`` document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --- decision reasons ---------------------------------------------------------
# Kept as module constants rather than free strings so a consumer can branch on
# them without re-deriving the vocabulary from log text.
REASON_ADMITTED = "admitted"
REASON_CAPACITY = "capacity"
REASON_REDUNDANT = "redundant"
REASON_QUALITY = "quality"
REASON_FRONTIER_FORCED = "frontier_forced"
REASON_FORCE_ALL_STRESS = "force_all_stress"


@dataclass
class ObservationDecision:
    """One frame's resolved observation / supervision / insertion decision.

    The four booleans are deliberately separate fields rather than one enum:
    #600's whole point is that they are independent questions, and an enum would
    re-conflate exactly the axes this contract exists to separate.

    Invariants (enforced in ``__post_init__``, not merely documented):

    * ``admit_supervision`` implies ``mapping_candidate`` -- a frame cannot be
      admitted for supervision without having been a candidate.
    * ``allow_geometry_insertion`` implies ``admit_supervision`` -- this is the
      #600 code invariant "a frame rejected for mapping cannot accidentally be
      inserted from depth", expressed where it cannot be forgotten.
    * ``observe`` is True on every decision this class can represent. A frame
      that is not observed at all never reaches an admitter, so a False value
      here would be unrepresentable rather than merely unusual; it is kept as an
      explicit field anyway because #600's contract names it, and a future
      source-level drop (frame corrupt, pose unavailable) is the honest place
      for it.
    """

    observe: bool = True
    mapping_candidate: bool = True
    admit_supervision: bool = True
    allow_geometry_insertion: bool = True
    reason: str = REASON_ADMITTED
    # Diagnostics: the measured backlog signal this decision was taken on, so a
    # reader never has to re-derive why a frame was skipped.
    optimizer_steps_since_last_admit: Optional[int] = None
    consecutive_skips: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allow_geometry_insertion and not self.admit_supervision:
            raise ValueError(
                "ObservationDecision violates the #600 insertion invariant: "
                "allow_geometry_insertion=True with admit_supervision=False. "
                "A frame rejected for mapping supervision must never be "
                "permitted to create new geometry from its depth map."
            )
        if self.admit_supervision and not self.mapping_candidate:
            raise ValueError(
                "ObservationDecision: admit_supervision=True with "
                "mapping_candidate=False -- a frame cannot be admitted without "
                "having been considered."
            )

    @classmethod
    def from_disposition(cls, disposition) -> "ObservationDecision":
        """Lower a #1190 terminal record to the legacy #600 decision shape.

        This compatibility path cannot grant supervision or depth insertion
        that the immutable terminal disposition declined.
        """

        from eligsir.contracts.control import ObservationDisposition

        if not isinstance(disposition, ObservationDisposition):
            raise TypeError("disposition must be ObservationDisposition")
        return cls(
            observe=disposition.observe,
            mapping_candidate=disposition.mapping_candidate,
            admit_supervision=disposition.admit_supervision,
            allow_geometry_insertion=disposition.allow_geometry_insertion,
            reason=disposition.reason,
            extra={
                "source_identity": disposition.source.fingerprint(),
                "terminal_kind": disposition.kind.value,
                "snapshot_id": disposition.snapshot_id,
            },
        )

    @property
    def admitted(self) -> bool:
        """The single boolean the legacy ``admission_fn`` protocol returns."""
        return bool(self.admit_supervision)

    def as_stats(self, mode: str) -> Dict[str, Any]:
        """The stats dict shape ``streaming_scene``/``train_streaming`` expect."""
        return {
            "mode": mode,
            "admitted": self.admitted,
            "reason": self.reason,
            # #600: the decision axes, surfaced per-frame so they are
            # independently observable rather than inferred from `admitted`.
            "observe": bool(self.observe),
            "mapping_candidate": bool(self.mapping_candidate),
            "admit_supervision": bool(self.admit_supervision),
            "allow_geometry_insertion": bool(self.allow_geometry_insertion),
            "optimizer_steps_since_last_admit": self.optimizer_steps_since_last_admit,
            "consecutive_skips": int(self.consecutive_skips),
        }


def force_all_decision() -> ObservationDecision:
    """The historical hard-all decision: admit unconditionally, never skip.

    Used by the ``force_all`` stress mode. Kept as a named constructor so the
    parity test can assert force-all's decision output is identical to the
    pre-#600 behavior without duplicating the field list.
    """
    return ObservationDecision(reason=REASON_FORCE_ALL_STRESS)


class SoftAllAdmitter:
    """Capacity-aware ``all``: retain every useful observation the mapper can keep up with.

    The backlog signal
    ------------------
    ``optimizer_steps_since_last_admit``, read from the authoritative
    ``RealizedCadenceTracker.optimizer_steps`` clock (#289's denominator -- one
    real ``optimizer.step()``, deliberately not ``iteration``).

    This is the honest signal for exactly the confusion #598 identified:
    ``FrameScheduler.should_release()`` advances source visibility from the
    global iteration counter and *does not know* whether the previously released
    frame received any optimizer updates at all. So ``effective_spf = 2`` does
    NOT imply ``optimizer_updates_per_admitted_frame == 2``. Measuring the real
    step clock between admissions is what turns that unverified assumption into
    an observed quantity -- and lets admission refuse to pile a new frame onto a
    mapper that has not yet done any work on the previous one.

    Bounds
    ------
    * ``min_steps_per_admit`` -- the capacity bound. A frame arriving before the
      previously admitted frame has received this many optimizer steps is
      skipped for supervision (``reason="capacity"``).
    * ``max_consecutive_skips`` -- the age/frontier bound. After this many
      consecutive skips the next frame is force-admitted regardless of backlog
      (``reason="frontier_forced"``), so no observation can be starved
      indefinitely and a genuinely new frontier view always gets in. This is
      what keeps soft-all "soft": it degrades cadence under load rather than
      abandoning regions of the trajectory.

    Together these make the mapping backlog provably bounded, which is #600's
    acceptance criterion "production ``all`` cannot create an unbounded mapping
    backlog".

    Default values (a flagged judgment call -- see the PR body)
    -----------------------------------------------------------
    ``min_steps_per_admit = 1``, ``max_consecutive_skips = 4``.

    ``1`` is chosen to be **inert on every healthy run** while still bounding the
    one failure mode this repo has actually measured. A healthy iter_based run
    has ``effective_spf >= 50`` (``MappingScheduler.min_steps_per_frame`` clamps
    upward to 50, per W3-38), so it accrues 50 optimizer steps between releases
    and never trips a bound of 1. What a bound of 1 *does* refuse is admitting
    two frames with **zero** optimizer work between them -- i.e. the documented
    ``dataset_fps`` release runaway (``F = 24 + 2 x iteration``, ~2 frames per
    iteration, whole 2525-frame trajectory exhausted by iteration ~1251), which
    is this project's canonical unbounded-backlog / map-explosion pathology.

    So the default is deliberately the weakest bound that is still a real bound:
    it reproduces today's behavior on every non-pathological run, and refuses
    only the case already root-caused as pathological. Raising it is how a run
    deliberately induces overload (used by this issue's own GPU validation);
    ``0`` disables the capacity bound entirely, which is soft-all in name only
    and is NOT the same thing as ``force_all`` (which is a reported semantics
    change, not just a threshold).
    """

    def __init__(
        self,
        min_steps_per_admit: int = 1,
        max_consecutive_skips: int = 4,
        steps_provider: Any = None,
    ) -> None:
        self.min_steps_per_admit = max(0, int(min_steps_per_admit))
        self.max_consecutive_skips = max(0, int(max_consecutive_skips))
        self._steps_provider = steps_provider
        self._steps_at_last_admit: Optional[int] = None
        self._consecutive_skips = 0
        # Counters, for report/telemetry. #602 owns the full telemetry surface;
        # these are the minimum needed to make #600's own invariants provable.
        self.n_evaluated = 0
        self.n_admitted = 0
        self.n_skipped_capacity = 0
        self.n_frontier_forced = 0

    def _current_steps(self) -> int:
        if self._steps_provider is None:
            return 0
        try:
            return int(self._steps_provider())
        except Exception:
            # Telemetry must never crash training. An unreadable clock means we
            # cannot prove backlog, so we admit -- failing OPEN preserves the
            # historical behavior rather than silently dropping supervision.
            return -1

    def decide(self) -> ObservationDecision:
        """Decide one frame. Information-free by design (see #601 scope fence)."""
        self.n_evaluated += 1
        steps = self._current_steps()

        # First frame, or an unreadable clock: always admit.
        if self._steps_at_last_admit is None or steps < 0:
            return self._admit(steps, REASON_ADMITTED, since=None)

        since = steps - self._steps_at_last_admit

        if self.min_steps_per_admit <= 0 or since >= self.min_steps_per_admit:
            return self._admit(steps, REASON_ADMITTED, since=since)

        # Backlog exceeded. Frontier/age protection: never starve indefinitely.
        if self._consecutive_skips >= self.max_consecutive_skips:
            self.n_frontier_forced += 1
            return self._admit(steps, REASON_FRONTIER_FORCED, since=since)

        self._consecutive_skips += 1
        self.n_skipped_capacity += 1
        return ObservationDecision(
            observe=True,
            mapping_candidate=True,
            admit_supervision=False,
            allow_geometry_insertion=False,
            reason=REASON_CAPACITY,
            optimizer_steps_since_last_admit=since,
            consecutive_skips=self._consecutive_skips,
        )

    def _admit(
        self, steps: int, reason: str, since: Optional[int]
    ) -> ObservationDecision:
        self._steps_at_last_admit = steps
        skips = self._consecutive_skips
        self._consecutive_skips = 0
        self.n_admitted += 1
        return ObservationDecision(
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=True,
            reason=reason,
            optimizer_steps_since_last_admit=since,
            consecutive_skips=skips,
        )

    def summary(self) -> Dict[str, Any]:
        """Report block. ``n_skipped_capacity == 0`` means the bound never fired."""
        return {
            "min_steps_per_admit": self.min_steps_per_admit,
            "max_consecutive_skips": self.max_consecutive_skips,
            "n_evaluated": self.n_evaluated,
            "n_admitted": self.n_admitted,
            "n_skipped_capacity": self.n_skipped_capacity,
            "n_frontier_forced": self.n_frontier_forced,
            # True iff the capacity bound actually engaged this run. A consumer
            # must not read a soft-all run as evidence about capacity-aware
            # behavior when this is False -- it behaved exactly like force-all.
            "capacity_bound_engaged": self.n_skipped_capacity > 0,
        }
