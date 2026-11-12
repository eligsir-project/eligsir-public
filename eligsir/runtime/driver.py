"""The ``MappingSession`` driver contract (#1075 follow-up, owner A).

:class:`~eligsir.runtime.session.MappingSession` owns the **frame** cadence
only. The **iteration** cadence -- render, loss, backward, optimizer step, MCMC
mutation -- stays with the driver. That decision, its evidence and the three
rejected alternatives are in
``docs/architecture/2026-08-23-mapping-session-cadence-decision.md``; this module
is that decision made executable.

Why a separate module and not more methods on the session
=========================================================

``train_streaming.py`` runs both clocks in one ``while``: the mapper commits
per frame inside the ingest drain (``:6292``, body to ``:7553``), the optimizer
steps once per iteration outside it (``:~8053-9363``), and frames-per-iteration
is a variable ``0..drain_cap``. The discriminating case is that **zero** is
normal: ``train_streaming.py:6260-6263`` -- "When nothing is ready the loop falls
through to a training step (concurrent mapping)" -- which is the steady state of
the whole post-coverage refinement tail. A cadence owned inside the session
would need its own clock to emit a zero-frame tick, i.e. a hidden thread or
timer, which ``session.py``'s determinism clause forbids.

So the missing piece was never a method. It was the *loop shape the caller must
honour*, which is what :func:`drain_tick` and :func:`check_cadence_invariant`
pin down.

Scope, stated honestly
======================

These are **reference and verification helpers, not a port of
``streaming_training()``'s policy.** Release gating, backpressure, BurstBudget,
holdout bypass and admission closures stay driver-owned (see the design note
§3: those decisions are taken on an undecoded ``peek_next_frame()`` result, and
a ``SensorPacket`` requires decoded buffers -- routing them through ``submit()``
would force the very decode the skip path exists to avoid). :func:`drain_tick`
therefore takes policy as injected callables and never invents any.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

from eligsir.contracts.runtime import SensorPacket

from .prepared_ingress import PreparedIngressDeferred
from .session import MappingSession, SubmitOutcome

#: Default frames drained per iteration, mirroring ``train_streaming.py:6286``'s
#: ``--streaming_max_drain_per_iter`` default. Restated rather than imported:
#: this module must not depend on the training script.
DEFAULT_DRAIN_CAP = 2


class CadenceViolation(AssertionError):
    """Raised when a recorded tick sequence breaks the driver contract.

    An ``AssertionError`` subclass deliberately: this is a contract violation by
    the driver's own code, not a runtime condition a caller recovers from.
    """


@dataclass(frozen=True, slots=True)
class IterationTick:
    """What one iteration of the driver's loop did to the session.

    One record per iteration, including iterations that submitted nothing --
    ``submitted == 0`` is a normal, expected tick (design note §2), and a driver
    that omits those records cannot be checked against the contract at all.

    Attributes:
        iteration: The driver's iteration number. Strictly increasing across a
            recorded sequence; the session never advances it.
        submitted: Packets passed to ``submit()`` this tick, ``0..drain_cap``.
        mapped: How many of those were accepted and reached the mapper.
        skipped_by_policy: Frames the driver refused *before* building a packet
            (backpressure, burst budget, admission). These never reached
            ``submit()`` and so consumed no ``sequence``; recorded here only so a
            driver's own ledger can be reconciled against ``session.events()``.
        deferred: Frames that were ready but exceeded ``drain_cap`` and carry to
            a later tick. Deferred, never dropped (``train_streaming.py``
            ``:6265-6268``) -- coverage is preserved.
        consumed: ``submitted + skipped_by_policy`` -- what the cap actually
            bounds. See :func:`drain_tick`.
        sequences: The ``sequence`` values the session assigned this tick, in
            submission order.
    """

    iteration: int
    submitted: int
    mapped: int
    skipped_by_policy: int = 0
    deferred: int = 0
    sequences: Tuple[int, ...] = ()

    @property
    def consumed(self) -> int:
        """Frames consumed this tick: submitted plus skipped-by-policy."""
        return self.submitted + self.skipped_by_policy

    def __post_init__(self) -> None:
        if self.submitted < 0 or self.mapped < 0:
            raise ValueError("submitted/mapped must be non-negative")
        if self.mapped > self.submitted:
            raise ValueError(
                f"tick {self.iteration}: mapped={self.mapped} exceeds "
                f"submitted={self.submitted}; a frame cannot be mapped without "
                "being submitted"
            )
        if len(self.sequences) != self.submitted:
            raise ValueError(
                f"tick {self.iteration}: {len(self.sequences)} sequences recorded "
                f"for {self.submitted} submits; every submit is assigned exactly "
                "one sequence, including rejected ones"
            )


def drain_tick(
    session: MappingSession,
    *,
    iteration: int,
    has_ready_frame: Callable[[], bool],
    next_packet: Callable[[], Optional[SensorPacket]],
    drain_cap: int = DEFAULT_DRAIN_CAP,
    should_release: Optional[Callable[[int, float], bool]] = None,
    iter_dt: float = 0.0,
    pending_packet: list[SensorPacket] | None = None,
) -> IterationTick:
    """Perform one iteration's bounded frame drain and return what it did.

    This is the reference shape of ``train_streaming.py``'s drain ``while``
    (``:6292``) reduced to the session boundary. It does **not** step the
    optimizer: the caller does that after this returns, exactly once, whatever
    this tick's ``submitted`` count turned out to be.

    ``next_packet`` returning ``None`` means the driver's own policy refused this
    frame before decoding it (design note §3) -- counted as ``skipped_by_policy``
    and, correctly, never submitted. The drain continues to the next frame rather
    than ending the tick, mirroring the real loop's ``continue`` at ``:6380``.

    **The cap bounds frames CONSUMED, not packets submitted.** In the real loop
    ``_drained_this_iter += 1`` is at ``:6305``, *before* the backpressure block
    (``:6331-6380``) whose ``continue`` skips the frame -- so a metadata-only skip
    consumes a drain unit and reduces how many real frames that tick can still
    ingest. Charging skips to the cap is what keeps the frames-per-iteration
    distribution matching the current loop, which is the observable the design
    note promises to preserve. ``_drain_dt`` is zeroed at ``:6306``, on the same
    side of that ``continue``, so a skipped frame likewise consumes the tick's
    virtual-time advance.

    Args:
        session: The session to submit into. Must not be closed.
        iteration: The driver's current iteration number, recorded on the tick.
        has_ready_frame: Non-blocking readiness probe. MUST NOT block waiting for
            a frame -- a tick with nothing ready is a valid zero-submit tick, and
            blocking here is what turns concurrent mapping into stop-and-wait.
        next_packet: Consume the next ready frame and return its packet, or
            ``None`` if driver-side policy skipped it (the frame is consumed
            either way -- a skip is a decision, not a retry).
        drain_cap: Maximum frames consumed this tick, submitted or skipped.
            Values below 1 are clamped to 1, matching ``train_streaming.py:6286``'s
            ``max(1, ...)``.
        should_release: Optional release gate, called as
            ``should_release(iteration, dt)`` per candidate frame to mirror
            ``scheduler.should_release(iteration, dt=_drain_dt)``. ``dt`` is
            ``iter_dt`` for the tick's first candidate and ``0.0`` thereafter, so
            simulated time advances once per tick rather than once per frame.
            Returning ``False`` ends the drain without consuming the frame.
        iter_dt: The tick's virtual-time advance, passed to the first
            ``should_release`` call only.
        pending_packet: Optional caller-owned one-packet retry slot.  A
            prepared retention deferral keeps the exact packet here instead
            of consuming a newer source; pass the same list on the next tick.

    Returns:
        The :class:`IterationTick` describing this tick. ``deferred`` is set when
        the drain stopped on ``drain_cap`` while frames were still ready.
    """
    cap = max(1, int(drain_cap))
    consumed = 0
    submitted = 0
    mapped = 0
    skipped = 0
    sequences: list[int] = []
    drain_dt = float(iter_dt)
    if pending_packet is not None and len(pending_packet) > 1:
        raise ValueError("pending_packet must hold at most one packet")
    retention_deferred = False

    while consumed < cap and (pending_packet or has_ready_frame()):
        retrying = bool(pending_packet)
        if not retrying and should_release is not None and not should_release(iteration, drain_dt):
            break
        packet = pending_packet[0] if retrying else next_packet()
        if not retrying:
            # Charged before the skip branch, exactly as :6305 precedes :6380.
            consumed += 1
            drain_dt = 0.0
        if packet is None:
            skipped += 1
            continue
        try:
            outcome: SubmitOutcome = session.submit(packet)
        except PreparedIngressDeferred:
            if pending_packet is None:
                raise
            if not retrying:
                pending_packet.append(packet)
            retention_deferred = True
            break
        if retrying:
            pending_packet.pop()
        submitted += 1
        sequences.append(outcome.sequence)
        if outcome.accepted:
            mapped += 1

    deferred = int(retention_deferred or (consumed >= cap and has_ready_frame()))

    return IterationTick(
        iteration=iteration,
        submitted=submitted,
        mapped=mapped,
        skipped_by_policy=skipped,
        deferred=deferred,
        sequences=tuple(sequences),
    )


def check_cadence_invariant(
    ticks: Sequence[IterationTick],
    *,
    drain_cap: int = DEFAULT_DRAIN_CAP,
    expect_zero_frame_ticks: bool = False,
) -> None:
    """Validate a recorded tick sequence against the driver contract.

    Checks D3 (bounded consumption), D6 (arrival order) and the strictly-increasing
    iteration clock (D2). D1, D5 and D7 are ownership rules not observable from a
    tick sequence; D4 is checked by the caller comparing total mapped against
    total admitted, since only the caller knows how many frames the source held.

    Args:
        ticks: The recorded ticks, in order. An empty sequence passes vacuously.
        drain_cap: The cap the driver claims to have honoured.
        expect_zero_frame_ticks: When ``True``, require at least one tick with
            ``submitted == 0``. Use this in a test that specifically asserts the
            zero-frame tick is reachable -- it is the case that discriminates a
            driver-owned iteration clock from a session-owned one, and a test
            suite that never hits it has not tested the decision.

    Raises:
        CadenceViolation: On the first violation found, naming the tick.
    """
    cap = max(1, int(drain_cap))
    previous_iteration: Optional[int] = None
    expected_sequence: Optional[int] = None
    saw_zero = False

    for tick in ticks:
        if previous_iteration is not None and tick.iteration <= previous_iteration:
            raise CadenceViolation(
                f"iteration clock went backwards or stalled: {previous_iteration} "
                f"-> {tick.iteration}; the driver owns a strictly increasing "
                "iteration clock (D2)"
            )
        previous_iteration = tick.iteration

        if tick.consumed > cap:
            raise CadenceViolation(
                f"tick {tick.iteration} consumed {tick.consumed} frames "
                f"({tick.submitted} submitted + {tick.skipped_by_policy} skipped), "
                f"exceeding drain_cap={cap} (D3). The cap bounds frames CONSUMED, "
                "not packets submitted: train_streaming.py charges the drain "
                "counter at :6305, before the policy-skip continue at :6380"
            )
        if tick.submitted == 0:
            saw_zero = True

        for sequence in tick.sequences:
            if expected_sequence is None:
                expected_sequence = sequence
            if sequence != expected_sequence:
                raise CadenceViolation(
                    f"tick {tick.iteration} submitted sequence {sequence} where "
                    f"{expected_sequence} was due; submission order is arrival "
                    "order and sequences are gapless across ticks (D6)"
                )
            expected_sequence += 1

    if expect_zero_frame_ticks and not saw_zero:
        raise CadenceViolation(
            "no zero-frame tick was recorded; a driver whose iteration clock is "
            "genuinely its own must be able to tick with no frame ready -- this "
            "is the refinement tail's steady state (design note §2)"
        )


def total_mapped(ticks: Iterable[IterationTick]) -> int:
    """Sum ``mapped`` across ticks -- the D4 coverage figure to reconcile."""
    return sum(tick.mapped for tick in ticks)


__all__ = [
    "DEFAULT_DRAIN_CAP",
    "CadenceViolation",
    "IterationTick",
    "drain_tick",
    "check_cadence_invariant",
    "total_mapped",
]
