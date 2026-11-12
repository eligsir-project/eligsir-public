"""Fixed-stride admission (issue #934): a deterministic, information-free
experimental CONTROL arm for comparing `--streaming_frame_admission
hybrid_keyframe`'s smart selector against an explicit fixed-stride policy at
a matched realized admission/work budget.

Why this exists instead of reusing `--streaming_source_stride` /
`--streaming_frame_stride`: those decimate the frame SOURCE before the
candidate population that reaches admission scoring even exists, which
changes the "candidate population / source-decimation stack" axis #934
explicitly requires held constant between the hybrid_keyframe arm and its
control. `FixedStrideAdmitter` instead sits at the exact same call site
`KeyframeAdmitter.admit()` does (train_streaming.py's `admission_fn`
contract) -- so both arms share the population of frames that reach
`StreamingScene.ingest_next_frame`. That is NOT the same as an identical
post-filter candidate stream: `hybrid_keyframe` also wires a `pregate_fn`
(`_keyframe_pregate_closure`, gated on `--streaming_keyframe_motion_pregate`,
default `True`) that runs BEFORE `admission_fn` and silently drops
stale-pose/no-motion frames before `KeyframeAdmitter.admit()` ever sees
them; `fixed_stride` wires no `pregate_fn` at all (see the dispatch site in
train_streaming.py), so it counts every ingested frame. The motion pregate
is deliberately part of hybrid_keyframe's own admission POLICY, not part of
the population definition the two arms are meant to share -- so
`--streaming_fixed_stride_n` cannot be set to a nominal value (e.g. `10`)
and assumed to land on a matched realized admit count; it must be tuned
empirically against hybrid_keyframe's REALIZED committed-K for the run
being compared against. Any per-candidate admission RATE reported for
either arm must say which population (pre-pregate vs. post-pregate) it was
computed over -- the two are not interchangeable.

Extracted from train_streaming.py's `_fixed_stride_admission_closure` into
this small, typed, unit-testable object -- mirroring why
`eligsir/mapping/scheduler/keyframes.py`'s `KeyframeAdmitter` exists
separately from the inline closure that bridges to it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixedStrideDecision:
    """Result of one `FixedStrideAdmitter.admit()` call."""

    admitted: bool
    reason: str
    ordinal: int
    stride_n: int


class FixedStrideAdmitter:
    """Admits exactly every Nth candidate reaching `admit()`, deterministically.

    `admit()` must be called exactly once per candidate frame reaching the
    admission_fn call site -- every non-holdout, non-guard materialized
    frame that reaches `StreamingScene.ingest_next_frame`, the same
    population `KeyframeAdmitter.admit()` is invoked over under
    `hybrid_keyframe`. This is the shared-population contract #934 requires,
    NOT an identical post-filter candidate stream: `hybrid_keyframe` also
    wires a `pregate_fn` (motion pre-check, on by default via
    `--streaming_keyframe_motion_pregate`) that runs first and drops
    stale-pose/no-motion frames before its `admission_fn` ever sees them,
    while `fixed_stride` wires no `pregate_fn` and so counts every frame in
    the shared population, filtered or not. Consequently `stride_n` is a
    nominal starting point (default `10`), not a value that can be assumed
    to reproduce hybrid_keyframe's realized admit rate -- it must be tuned
    empirically against that run's actual committed-K. It is deliberately
    pure and stateless beyond an internal candidate counter: no render, no
    pose, no motion/overlap/novelty/max_gap consultation -- being
    information-free about the frame's content is the entire point of a
    control arm.

    The first candidate (ordinal 0) is always admitted so bootstrap always
    has an initial frame, matching hybrid_keyframe's own bootstrap-admits
    behavior.
    """

    def __init__(self, stride_n: int = 10) -> None:
        self.stride_n = max(1, int(stride_n))
        self._ordinal = -1

    @property
    def n_evaluated(self) -> int:
        """Return the count of candidates evaluated so far."""
        return self._ordinal + 1

    def admit(self) -> FixedStrideDecision:
        """Evaluate next candidate and return admit/skip decision."""
        self._ordinal += 1
        admitted = (self._ordinal % self.stride_n) == 0
        reason = "fixed_stride" if admitted else "fixed_stride_skip"
        return FixedStrideDecision(
            admitted=admitted,
            reason=reason,
            ordinal=self._ordinal,
            stride_n=self.stride_n,
        )
