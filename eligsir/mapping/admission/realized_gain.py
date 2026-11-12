"""Realized-gain tracker for admitted frames (plan section 11.4).

For every frame the *real* admission path (``KeyframeAdmitter``, unchanged
by this PR) actually admits, this module records:

- ``loss_before`` -- the loss at (or shortly after) admission;
- ``loss_after`` -- the loss the next time that same camera is trained,
  at least ``horizon_iterations`` of wall-clock training later;
- ``compute_consumed_ms`` -- wall-clock time between those two observations
  (a coarse proxy for "how much compute this frame's follow-up cost", not
  a profiled per-op cost);
- ``under_supported_mass_before``/``_after`` -- read straight off
  ``evidence/reducers.under_supported_mass`` when a support tensor is
  available (``None`` otherwise -- never fabricated).

This is intentionally piggy-backed on the *existing* per-iteration
training loop rather than triggering any extra render: ``observe_loss``
is called once per iteration with whatever camera/loss the loop already
computed, and only does bookkeeping (dict lookups) unless that camera
happens to be a pending realized-gain follow-up due this iteration. This
keeps realized-gain tracking's "compute consumed" close to zero beyond the
bookkeeping itself, and -- critically -- means it never changes *which*
camera gets trained or *what* loss drives backward/the optimizer (it only
ever reads values the training loop already produced).

Cameras are keyed by Python object identity (``id(camera)``), matching how
this codebase already keeps sampled cameras alive as long-lived objects in
the keyframe window / replay buffer (``StreamingScene``) -- no dependency
on any particular dataset's frame-id scheme.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class RealizedGainRecord:
    frame_id: int
    admitted_iteration: int
    loss_before: float
    score_map_version: int
    score_topology_version: Optional[int] = None
    loss_after: Optional[float] = None
    followup_iteration: Optional[int] = None
    compute_consumed_ms: Optional[float] = None
    under_supported_mass_before: Optional[float] = None
    under_supported_mass_after: Optional[float] = None
    grad_mass_before: Optional[float] = None
    grad_mass_after: Optional[float] = None
    predicted_utility: Optional[float] = None

    @property
    def realized_gain(self) -> Optional[float]:
        """``loss_before - loss_after``: positive means the frame's future
        training actually reduced loss (a real, realized benefit); negative
        or near-zero means it didn't help (or hurt). ``None`` until the
        follow-up observation lands.
        """
        if self.loss_after is None:
            return None
        return self.loss_before - self.loss_after

    @property
    def under_supported_mass_delta(self) -> Optional[float]:
        if self.under_supported_mass_before is None or self.under_supported_mass_after is None:
            return None
        return self.under_supported_mass_before - self.under_supported_mass_after


@dataclass
class RealizedGainTracker:
    """Bookkeeping-only tracker; see module docstring for the "no extra
    render" design.
    """

    horizon_iterations: int = 200
    max_pending: int = 256
    _pending: Dict[int, RealizedGainRecord] = field(default_factory=dict, repr=False)
    _completed: List[RealizedGainRecord] = field(default_factory=list, repr=False)
    _wallclock: Dict[int, float] = field(default_factory=dict, repr=False)

    def register_admission(
        self,
        camera_key: int,
        frame_id: int,
        iteration: int,
        loss_before: float,
        score_map_version: Optional[int] = None,
        score_topology_version: Optional[int] = None,
        predicted_utility: Optional[float] = None,
        under_supported_mass_before: Optional[float] = None,
    ) -> None:
        """Call once, right when a frame is admitted (the real admission
        decision has already been made by ``KeyframeAdmitter`` -- this only
        starts tracking it, it cannot veto or alter that decision).
        """
        if camera_key in self._pending:
            return
        if len(self._pending) >= self.max_pending:
            # Bounded: drop the oldest pending record rather than growing
            # unboundedly on a run where admitted frames outpace the
            # horizon (matches this codebase's general bounded-buffer
            # convention, e.g. streaming_replay_buffer).
            oldest_key = next(iter(self._pending))
            self._pending.pop(oldest_key, None)
            self._wallclock.pop(oldest_key, None)
        self._pending[camera_key] = RealizedGainRecord(
            frame_id=frame_id,
            admitted_iteration=iteration,
            loss_before=float(loss_before),
            score_map_version=(
                int(iteration) if score_map_version is None else int(score_map_version)
            ),
            score_topology_version=(
                None
                if score_topology_version is None
                else int(score_topology_version)
            ),
            predicted_utility=predicted_utility,
            under_supported_mass_before=under_supported_mass_before,
        )
        self._wallclock[camera_key] = time.monotonic()

    def observe_loss(
        self,
        camera_key: int,
        iteration: int,
        loss: float,
        under_supported_mass: Optional[float] = None,
        grad_mass: Optional[float] = None,
    ) -> Optional[RealizedGainRecord]:
        """Called once per training iteration with whatever camera/loss the
        loop just computed (see module docstring). Returns the completed
        record iff this call closed one out, else ``None`` (the overwhelming
        common case -- either this camera isn't tracked, or its horizon
        hasn't elapsed yet).
        """
        record = self._pending.get(camera_key)
        if record is None:
            return None
        if iteration - record.admitted_iteration < self.horizon_iterations:
            return None
        record.loss_after = float(loss)
        record.followup_iteration = iteration
        record.under_supported_mass_after = under_supported_mass
        record.grad_mass_after = grad_mass
        start = self._wallclock.pop(camera_key, None)
        if start is not None:
            record.compute_consumed_ms = (time.monotonic() - start) * 1000.0
        del self._pending[camera_key]
        self._completed.append(record)
        return record

    def completed(self) -> List[RealizedGainRecord]:
        return list(self._completed)

    def n_pending(self) -> int:
        return len(self._pending)


def under_supported_mass_from_tensors(
    support: Optional[torch.Tensor], opacity: Optional[torch.Tensor], threshold: float = 0.01
) -> Optional[float]:
    """Thin wrapper around ``evidence.reducers.under_supported_mass`` that
    returns ``None`` (never a fabricated ``0.0``) when either tensor is
    unavailable, so a caller can tell "not measured this call" apart from
    "measured, and it's genuinely zero".
    """
    if support is None or opacity is None:
        return None
    from eligsir.mapping.evidence.reducers import under_supported_mass

    return under_supported_mass(support, opacity, threshold)
