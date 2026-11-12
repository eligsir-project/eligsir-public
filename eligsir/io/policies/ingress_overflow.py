"""Unified ingress overflow policy for frame boundaries (issue #836).

Frame-loss authority for the ``ros2://live`` ingestion path used to be
distributed across two independent layers with no relationship between
them:

- The OUTER resident buffer (``ROS2LiveStreamingFrameSource``,
  ``eligsir/io/sources/datasets/ros2_live.py``) already does the right
  thing: bounded backpressure, then durable in-memory spill, and only marks
  the run invalid if the spill itself is exhausted (see
  ``eligsir.io.policies.bag_replay.BagReplayPolicy``).
- The INNER per-topic emit queue (``ROS2LiveRGBDSource``,
  ``eligsir/io/sources/ros2_live.py``) silently evicted the oldest
  not-yet-consumed frame via ``deque(maxlen=...)`` the moment it filled,
  consulting nothing -- see that module's ``try_emit_ready_frames``. A
  frame lost there never reaches the outer layer at all, so the outer
  layer's careful lossless bookkeeping was moot: the outer layer's own
  in-code comment (``frames_dropped_by_inner_queue``) already flagged this
  as a real, previously-undetected full-coverage-contract violation.

``IngressOverflowPolicy`` is the single, frozen description of how a given
queue boundary must behave under overflow. It is resolved ONCE per source
kind (see :func:`resolve_ingress_policies`) and then consulted -- not
overridden, not partially applied -- by every queue boundary in the
``ros2://live`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class FrameDropMode(str, Enum):
    """How frames are dropped under real-time pressure (live-sensor paths).

    Not used for deterministic bag replay, which never intentionally drops
    a frame (see :class:`LossSemantics.NO_SILENT_LOSS`).
    """

    OLDEST = "oldest"  # Drop the oldest not-yet-consumed frame (pre-existing behavior).
    NEWEST = "newest"  # Drop the newly-arrived frame, keep the queue's existing contents.


class LossSemantics(str, Enum):
    """Frame-loss semantics required by the source kind.

    - NO_SILENT_LOSS: deterministic replay (bag/experiment). Every frame
      must be preserved, or the run must be explicitly marked invalid.
      Backpressure and durable spill are used; a genuine loss only happens
      once spill capacity itself is exhausted, and that loss is a hard,
      attributed, run-invalidating event -- never silent.
    - REAL_TIME_DROP: live sensor. Dropping a frame under sustained
      overload is legitimate (the same standing as a camera driver's own
      ring buffer) and does NOT invalidate the run -- but the drop must be
      explicit and attributed (exact source ordinal + reason), not an
      implicit ``deque(maxlen)`` side effect nobody consulted.
    """

    NO_SILENT_LOSS = "no_silent_loss"
    REAL_TIME_DROP = "real_time_drop"


@dataclass(frozen=True)
class IngressOverflowPolicy:
    """Policy governing frame ingestion at one queue boundary.

    Attributes:
        source_kind: Human-readable source classification (e.g.
            ``"deterministic_bag_replay"``, ``"live_sensor"``).
        loss_semantics: Which of :class:`LossSemantics` this boundary must
            honor.
        backpressure_budget_s: For NO_SILENT_LOSS boundaries that are able
            to actually wait for the consumer (the outer resident buffer,
            which is drained by a separate pump thread): duration of
            bounded backpressure retry before falling back to spill. A
            boundary whose caller holds a lock the sole consumer also needs
            (the inner emit queue) cannot productively wait here -- see
            that call site's own comment for why it goes straight to spill.
        backpressure_poll_interval_s: Sleep between backpressure retries.
        spill_enabled: Whether a durable in-memory spill is used once
            backpressure is exhausted (True for every NO_SILENT_LOSS policy
            here; kept as an explicit field rather than inferred so a
            report reader can see the intent, not just the semantics enum).
        spill_limit: Hard cap on the spill store. Only exceeding this
            causes a real, attributed loss.
        drop_mode: For REAL_TIME_DROP boundaries: which frame is dropped.
        invalidation_rule: Name of the consequence a real loss has on run
            validity -- documentation/report-shaping only, the actual flag
            this refers to is :attr:`IngressAttributionLedger
            .mapping_result_valid`.
    """

    source_kind: str
    loss_semantics: LossSemantics
    backpressure_budget_s: float = 0.25
    backpressure_poll_interval_s: float = 0.005
    spill_enabled: bool = True
    spill_limit: int = 4096
    drop_mode: FrameDropMode = FrameDropMode.OLDEST
    invalidation_rule: str = "flip_mapping_result_valid"

    @classmethod
    def for_deterministic_bag_replay(cls) -> "IngressOverflowPolicy":
        """No frame may ever be silently lost during bag/experiment replay."""
        return cls(
            source_kind="deterministic_bag_replay",
            loss_semantics=LossSemantics.NO_SILENT_LOSS,
            backpressure_budget_s=0.02,
            backpressure_poll_interval_s=0.005,
            spill_enabled=True,
            spill_limit=4096,
            invalidation_rule="flip_mapping_result_valid",
        )

    @classmethod
    def for_live_sensor(cls, drop_mode: FrameDropMode = FrameDropMode.OLDEST) -> "IngressOverflowPolicy":
        """A real live camera may legitimately shed frames under sustained
        overload; the drop must be owned by this policy and attributed."""
        return cls(
            source_kind="live_sensor",
            loss_semantics=LossSemantics.REAL_TIME_DROP,
            backpressure_budget_s=0.0,
            backpressure_poll_interval_s=0.0,
            spill_enabled=False,
            spill_limit=0,
            drop_mode=drop_mode,
            invalidation_rule="none",
        )


def resolve_ingress_policies(args) -> Tuple[IngressOverflowPolicy, IngressOverflowPolicy]:
    """Resolve the two policies governing the ``ros2://live`` ingestion path,
    ONCE, from source kind/args -- the single place both queue boundaries'
    policies are decided.

    Returns ``(inner_policy, outer_policy)``:

    - ``inner_policy`` governs :class:`~eligsir.io.sources.ros2_live
      .ROS2LiveRGBDSource`'s raw per-topic emit queue -- the literal ROS2
      subscription-callback-to-``RGBDFrame`` handoff. This is where
      REAL_TIME_DROP genuinely applies for a true live sensor: shedding the
      oldest not-yet-consumed raw message under sustained overload is
      legitimate real-time-camera behavior. Resolved by source kind:
      NO_SILENT_LOSS for deterministic bag replay
      (``args.ros2_live_bag_replay``), REAL_TIME_DROP otherwise.
    - ``outer_policy`` governs
      :class:`~eligsir.io.sources.datasets.ros2_live
      .ROS2LiveStreamingFrameSource`'s resident ``StreamingRGBDFrame``
      buffer. That buffer's only job is "retain until the consumer
      (``StreamingScene``) is done with it" -- there is no legitimate
      real-time-drop concept there even for a live camera, so it is
      unconditionally NO_SILENT_LOSS regardless of source kind. This
      matches that class's pre-existing, unchanged behavior (previously a
      hardcoded ``backpressure_max_wait_s=0.02, max_spill=4096`` with no
      kind check at all) -- this function only names those constants, it
      does not change them or make them kind-dependent.
    """
    is_bag = bool(getattr(args, "ros2_live_bag_replay", False))
    inner_policy = (
        IngressOverflowPolicy.for_deterministic_bag_replay()
        if is_bag
        else IngressOverflowPolicy.for_live_sensor()
    )
    outer_policy = IngressOverflowPolicy.for_deterministic_bag_replay()
    return inner_policy, outer_policy
