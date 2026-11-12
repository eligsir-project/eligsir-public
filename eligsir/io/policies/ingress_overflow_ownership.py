"""Issue #837 criterion 8, `overflow_drop_policy` half: feed issue #836's
REALIZED ``IngressOverflowPolicy`` pair into the ``eligsir.io
.ingress_ownership`` contract as a single owner declaration.

This module is a pure CONSUMER of two already-shipped, independent pieces:

- ``eligsir.io.ingress_ownership`` (issue #837, PR #856) -- the
  resolver and its ``Responsibility.OVERFLOW_DROP_POLICY`` slot. Nothing
  here changes that module's resolver semantics.
- ``eligsir.io.policies.ingress_overflow`` (issue #836, PR #852) --
  ``IngressOverflowPolicy``/``resolve_ingress_policies``. Nothing here
  changes that module either; this file only reads the two
  ``IngressOverflowPolicy`` objects a run already resolved and reshapes
  them into one :class:`~eligsir.io.ingress_ownership.OwnerDeclaration`.

Why ONE declaration for TWO policies (the load-bearing design decision):
``resolve_ingress_policies(args)`` returns an ``(inner_policy,
outer_policy)`` pair -- one for the inner ``ROS2LiveRGBDSource`` emit
queue, one for the outer ``ROS2LiveStreamingFrameSource`` resident buffer
-- but they are NOT two independently-authored authorities that happen to
agree; they are resolved together, in the same function call, from the
same source-kind decision (see that function's own docstring). Declaring
each boundary as its own ``MUTATOR``/``DECISION_OWNER`` entry against
``Responsibility.OVERFLOW_DROP_POLICY`` would make
``resolve_ingress_ownership`` see two active mutating authorities for the
same responsibility on every single ``ros2://live`` run and raise
``DUPLICATE_INGRESS_AUTHORITY`` -- a false structural-invalid verdict on
the *normal* case, and exactly backwards: issue #836's entire point was
collapsing two independently-drifting overflow behaviors into one
resolved policy, not preserving two owners. :func:`
resolve_overflow_drop_policy_owner_declaration` therefore emits exactly
one :class:`~eligsir.io.ingress_ownership.OwnerDeclaration`, and
represents both queue boundaries as attribute keys (``inner_*``/
``outer_*``) on that single declaration, so a report reader can still see
both boundaries' resolved behavior without the resolver ever seeing two
competing owners.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from eligsir.io.ingress_ownership import OwnerDeclaration, OwnershipRole, Responsibility
from eligsir.io.policies.ingress_overflow import IngressOverflowPolicy

# The single owner name declared for Responsibility.OVERFLOW_DROP_POLICY
# whenever a run has a resolved ros2://live overflow-policy pair. Kept as
# a module constant so tests and any future caller can assert against it
# without duplicating the literal string.
OVERFLOW_DROP_POLICY_OWNER = "ros2_live_ingress_overflow_policy"

# Human-readable labels for the two queue boundaries #852 resolves a
# policy for -- used as attribute-key prefixes on the single declaration
# (see the module docstring for why they are attributes, not separate
# declarations) and echoed verbatim so a report reader can find the exact
# production call site each boundary corresponds to.
INNER_BOUNDARY_LABEL = "ros2_live_rgbd_source_emit_queue"
OUTER_BOUNDARY_LABEL = "ros2_live_streaming_frame_source_resident_buffer"


def resolve_overflow_drop_policy_owner_declaration(
    inner_policy: IngressOverflowPolicy,
    outer_policy: IngressOverflowPolicy,
) -> OwnerDeclaration:
    """Build the single ``Responsibility.OVERFLOW_DROP_POLICY``
    :class:`~eligsir.io.ingress_ownership.OwnerDeclaration` for a run's
    already-resolved ``(inner_policy, outer_policy)`` pair (issue #836's
    ``resolve_ingress_policies`` output).

    ``role`` is ``DECISION_OWNER`` (not ``MUTATOR``): this declaration does
    not itself mutate frame data, it decides -- for the whole ``ros2://
    live`` ingestion path -- which of #837's two loss-semantics regimes
    (``NO_SILENT_LOSS``/``REAL_TIME_DROP``) applies at each queue boundary,
    which is exactly the ``Responsibility.CANDIDATE_PREFILTER``/
    ``KEYFRAME_COMMIT`` role precedent already used for a decision (not a
    data-mutating) responsibility elsewhere in this contract.

    ``mode`` is set to ``inner_policy.loss_semantics.value`` -- the ONE
    dimension that actually varies with source kind (bag replay vs. live
    sensor; see ``resolve_ingress_policies``'s docstring). ``outer_policy``
    is unconditionally ``NO_SILENT_LOSS`` regardless of source kind, so
    surfacing it as ``mode`` would make every run's top-level ``mode``
    identical and hide the one distinction that matters; the outer
    boundary's full state is still recorded verbatim in ``attributes``, it
    is simply not promoted to ``mode``.

    Every field of both policies that has observable report value is
    carried into ``attributes`` under an ``inner_``/``outer_`` prefix
    (source kind, loss semantics, drop mode, spill enablement/limit) so a
    report reader can recover the exact resolved behavior of both
    boundaries from this one declaration -- nothing is lost by collapsing
    two policy objects into one owner.
    """
    return OwnerDeclaration(
        Responsibility.OVERFLOW_DROP_POLICY,
        owner=OVERFLOW_DROP_POLICY_OWNER,
        role=OwnershipRole.DECISION_OWNER,
        mode=inner_policy.loss_semantics.value,
        attributes={
            "inner_boundary": INNER_BOUNDARY_LABEL,
            "inner_source_kind": inner_policy.source_kind,
            "inner_loss_semantics": inner_policy.loss_semantics.value,
            "inner_drop_mode": inner_policy.drop_mode.value,
            "inner_spill_enabled": inner_policy.spill_enabled,
            "inner_spill_limit": inner_policy.spill_limit,
            "outer_boundary": OUTER_BOUNDARY_LABEL,
            "outer_source_kind": outer_policy.source_kind,
            "outer_loss_semantics": outer_policy.loss_semantics.value,
            "outer_drop_mode": outer_policy.drop_mode.value,
            "outer_spill_enabled": outer_policy.spill_enabled,
            "outer_spill_limit": outer_policy.spill_limit,
        },
    )


def unwrap_frame_source_overflow_policies(
    frame_source: Any,
) -> Tuple[Optional[IngressOverflowPolicy], Optional[IngressOverflowPolicy]]:
    """Recover the ``(inner_policy, outer_policy)`` pair a run already
    resolved, by reading it off the live frame-source object -- never by
    calling ``resolve_ingress_policies(args)`` a second time.

    This is the "reuse the already-resolved state, never re-derive"
    discipline the ``source_sampling`` half of #837 criterion 8 already
    established for ``_source_decimation_block`` in ``train_streaming.py``.
    For a ``ros2://live`` run, ``frame_source`` is (or wraps)
    ``ROS2LiveStreamingFrameSource``
    (``eligsir/io/sources/datasets/ros2_live.py``), which stores
    exactly the two policies #852's ``resolve_ingress_policies(args)``
    resolved once at construction time as ``self._inner_policy`` and
    ``self._outer_policy`` -- see that class's ``__init__``.

    ``frame_source`` may also be wrapped one level deep by
    ``StridedFrameSource`` (issue #834's source-sampling wiring in
    ``train_streaming.py``, applied to whatever ``make_frame_source``
    returned when the sampling policy assigns the generic wrapper as
    sampling owner); that wrapper keeps the original object at
    ``self._src``. This function walks that single layer of indirection
    (bounded by an ``id()`` visited-set so a pathological self-referencing
    wrapper cannot loop forever) rather than assuming either shape.

    Returns ``(None, None)`` for any source that is not a ``ros2://live``
    source (TUM, ScanNet, the generic RGB-D sequence reader, an offline
    ``train.py`` scene, or a bag-replay-only ``ROS2LiveRGBDSource`` used
    directly without the streaming-frame-source wrapper) -- there is
    deliberately no fallback/default policy invented here. This is the
    correct "no ingress overflow policy exists for this run" signal the
    caller uses to leave ``Responsibility.OVERFLOW_DROP_POLICY`` absent/
    unresolved rather than fabricating an owner (issue #837 criterion 8
    trap 2).
    """
    src = frame_source
    seen: set = set()
    while src is not None and id(src) not in seen:
        seen.add(id(src))
        inner = getattr(src, "_inner_policy", None)
        outer = getattr(src, "_outer_policy", None)
        if isinstance(inner, IngressOverflowPolicy) and isinstance(outer, IngressOverflowPolicy):
            return inner, outer
        src = getattr(src, "_src", None)
    return None, None
