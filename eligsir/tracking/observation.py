"""Bridge between the frontend ``TrackingResult`` and canonical ``TrackingObservation``.

This module deliberately defines **no** provider protocol, ABC or registry. The
canonical ``TrackingProvider`` interface is #1075's unchecked acceptance item and
has no specified method signature anywhere in the repository as of 2026-08-23;
inventing one here would recreate exactly the "second architecture" problem that
closed PR #1011.

What *does* already exist is the pair of canonical records:

* ``TrackingResult`` (``eligsir/contracts/types.py``) — the per-frame result
  every in-process frontend returns today, explicitly retained by #1075
  ("one canonical ``TrackingProvider`` + ``TrackingObservation``/``TrackingResult``
  contract") and absent from #1073's hard-deletion list.
* ``TrackingObservation`` (``eligsir/contracts/runtime.py``) — the owner-safe
  ingress record added by #1074, currently defined but unwired.

Both directions of the pure, total mapping between them live here:

* :func:`tracking_observation_from_result` -- frontend result -> canonical record.
* :func:`tracking_result_from_observation` -- canonical record -> frontend result,
  needed because ``TrackingProvider.observe()`` now returns a
  ``TrackingObservation`` while the consumers it feeds
  (``MappingScheduler.decide``, ``OnlineController.tick``, ``ViewerBase.log_tracking``)
  still take a ``TrackingResult``. Keeping the conversion semantics fixed,
  reviewed and tested here means the eventual wiring commit is a call site, not a
  design decision improvised inline in ``train_streaming.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from eligsir.contracts.runtime import SourceIdentity, TrackingObservation, TrackingStatus
from eligsir.contracts.types import TrackingResult

# Telemetry scalars carried on TrackingResult that have no dedicated
# TrackingObservation field. They are preserved verbatim under diagnostics so no
# frontend provenance is lost across the conversion.
_TELEMETRY_FIELDS = (
    "depth_rmse",
    "rgb_residual",
    "inlier_ratio",
    "pose_delta_m",
    "pose_delta_deg",
    "render_alpha_coverage",
    "tracking_time_ms",
)


# Reserved ``TrackingResult.raw_stats`` key preserving the exact canonical status
# a back-converted observation carried. ``TrackingResult`` has only a two-valued
# ``valid`` flag, so INITIALIZING/LOST/INVALID (all valid=False) and
# VALID/DEGRADED (both valid=True) would otherwise be indistinguishable
# downstream. ``raw_stats`` is the type's own untyped diagnostics channel and is
# the right home for it; the key is namespaced so it cannot collide with a
# frontend-authored statistic.
TRACKING_STATUS_STAT_KEY = "canonical_tracking_status"


def _restored_status(result: TrackingResult) -> Optional[TrackingStatus]:
    """Return the canonical status recorded in ``raw_stats``, if it is trustworthy.

    Only a status that *agrees with the result itself* is honored: its usability
    must match ``result.valid``, and a usable status must come with a pose (which
    ``TrackingObservation`` requires anyway). Anything else -- an unparseable
    value, or a stale key on a result some intermediate consumer has since
    flipped -- is ignored in favour of the fail-closed boolean mapping. The key
    is a lossless-round-trip aid, never an authority that can overrule the
    frontend's own verdict.
    """
    raw = result.raw_stats.get(TRACKING_STATUS_STAT_KEY) if result.raw_stats else None
    if raw is None:
        return None
    try:
        status = TrackingStatus(raw)
    except ValueError:
        return None
    usable = status in {TrackingStatus.VALID, TrackingStatus.DEGRADED}
    if usable != bool(result.valid):
        return None
    if usable and result.pose is None:
        return None
    return status


def tracking_observation_from_result(
    result: TrackingResult,
    source: SourceIdentity,
) -> TrackingObservation:
    """Convert a frontend ``TrackingResult`` into a canonical ``TrackingObservation``.

    Status mapping is deliberately two-valued and fail-closed:

    ``valid=True``  -> :attr:`TrackingStatus.VALID`
    ``valid=False`` -> :attr:`TrackingStatus.INVALID`

    A rejected registration must **never** map to ``DEGRADED``, because
    ``TrackingObservation.usable`` treats ``DEGRADED`` as usable and a pose the
    frontend's own fitness/RMSE gates refused would then be silently consumed
    downstream as if it had been accepted. ``DEGRADED`` is reserved for a future
    frontend that can genuinely report reduced-but-usable tracking; no current
    frontend distinguishes that state, so it is not synthesized here.

    The one exception is a result produced by
    :func:`tracking_result_from_observation`, which stamps the observation's
    exact canonical status into ``raw_stats[TRACKING_STATUS_STAT_KEY]``. That
    key is honored here -- and consumed, so it does not linger in
    ``diagnostics`` -- but only when it agrees with ``result.valid`` and with the
    presence of a pose (see :func:`_restored_status`). This is what makes
    ``observation -> result -> observation`` status-preserving for LOST /
    INITIALIZING / DEGRADED instead of collapsing them onto VALID/INVALID; it
    never lets the key manufacture a status the boolean contradicts.

    Args:
        result: The per-frame result returned by an in-process frontend.
        source: Canonical identity of the observation this result describes.

    Returns:
        The equivalent canonical observation, with all frontend telemetry and
        ``raw_stats`` provenance preserved under ``diagnostics``.
    """
    restored = _restored_status(result)
    if restored is not None:
        status = restored
    else:
        status = TrackingStatus.VALID if result.valid else TrackingStatus.INVALID

    # TrackingObservation validates confidence into [0, 1]; a frontend that
    # reports outside that range is clamped rather than raising, so a telemetry
    # quirk can never destroy an otherwise-good observation.
    confidence = min(1.0, max(0.0, float(result.confidence)))

    diagnostics: Dict[str, Any] = {}
    for field in _TELEMETRY_FIELDS:
        value = getattr(result, field, None)
        if value is not None:
            diagnostics[field] = value
    if result.raw_stats:
        diagnostics.update(result.raw_stats)
    if restored is not None:
        # Consumed above; leaving it would put a redundant copy of `status`
        # into diagnostics, free to drift from it on the next hop.
        diagnostics.pop(TRACKING_STATUS_STAT_KEY, None)

    return TrackingObservation(
        source=source,
        pose=result.pose,
        status=status,
        provider=result.method,
        confidence=confidence,
        diagnostics=diagnostics,
    )


#: Confidence assigned when a canonical observation carries ``confidence=None``.
#:
#: ``TrackingResult.confidence`` is a required float that gates mapping, so a
#: value must be chosen. The mapping is fail-closed *except* for VALID, where
#: 1.0 reproduces the convention already in force at every existing call site
#: (``MappingScheduler.decide`` reads ``tracking_result is None`` as confidence
#: 1.0, and ``train_streaming.py``'s ``dataset_pose`` path constructs
#: ``confidence=1.0`` for baked trajectories). DEGRADED deliberately gets 0.0
#: rather than a fabricated mid-range number: a provider that says "reduced but
#: usable" without quantifying it has given no evidence of quality, and inventing
#: one here would be a made-up quantity flowing straight into a mapping gate.
_DEFAULT_CONFIDENCE_BY_STATUS = {
    TrackingStatus.VALID: 1.0,
    TrackingStatus.DEGRADED: 0.0,
    TrackingStatus.INITIALIZING: 0.0,
    TrackingStatus.LOST: 0.0,
    TrackingStatus.INVALID: 0.0,
}


def tracking_result_from_observation(observation: TrackingObservation) -> TrackingResult:
    """Convert a canonical ``TrackingObservation`` back into a ``TrackingResult``.

    This is the adapter for consumers that predate #1074's canonical records and
    still take the frontend result type -- ``MappingScheduler.decide``,
    ``OnlineController.tick`` and ``ViewerBase.log_tracking``.

    Field mapping
    -------------
    ============================  ==========================================
    ``TrackingResult``            derived from
    ============================  ==========================================
    ``pose``                      ``observation.pose`` **verbatim, including
                                  ``None``** (see below)
    ``valid``                     ``observation.usable`` (VALID/DEGRADED)
    ``confidence``                ``observation.confidence``, else the
                                  per-status default above
    ``method``                    ``observation.provider``
    telemetry scalars             popped from ``observation.diagnostics``
    ``raw_stats``                 remaining ``diagnostics`` entries, plus
                                  ``TRACKING_STATUS_STAT_KEY``
    ============================  ==========================================

    ``pose=None`` on a non-usable observation, deliberately
    ------------------------------------------------------
    ``TrackingResult.pose`` is annotated ``Pose`` (non-Optional), but a LOST or
    INVALID observation genuinely has no pose and ``TrackingObservation`` forbids
    inventing one. The three alternatives were each rejected:

    * **Identity pose** -- stores a fabricated estimate as if it were measured;
      the exact failure #1074/#1075 exist to prevent.
    * **Returning ``None`` instead of a result** -- actively unsafe: every
      consumer reads ``tracking_result is None`` as *"precomputed pose,
      confidence 1.0, valid"* (``MappingScheduler.decide``,
      ``train_streaming.py``'s ``_tracking_confidence_ok``), so a lost track
      would be laundered into a fully-confident one.
    * **Raising** -- makes the conversion partial; a lost track is a normal
      runtime state, not an error.

    Carrying ``pose=None`` alongside ``valid=False`` is therefore the honest
    encoding, and is safe for every current consumer: each one reads ``.pose``
    behind a validity check (``train_streaming.py:6633`` guards on ``.valid``
    directly; ``:3134`` guards transitively through ``_tracking_confidence_ok``
    at ``:1654``), and ``to_tensorboard_dict()`` never touches ``.pose`` at all.
    **A future consumer must keep that discipline: check ``valid`` before
    dereferencing ``pose``.**

    Wiring constraint for ``OnlineController.tick``
    ----------------------------------------------
    ``tick`` gates on ``tracking_result is not None`` -- not on ``.pose`` or
    ``.valid`` -- and treats any non-``None`` result as a **pose arrival** feeding
    its map-age hysteresis (``online_controller.py``, ``record_pose_arrival()``).
    Because this conversion is total, a LOST observation now reaches it as a real
    object where the pre-#1075 code passed ``None``. The integration owner MUST
    therefore either gate the ``tick`` call on ``observation.usable``, or teach
    ``tick`` to skip its pose-arrival bookkeeping when ``not result.valid``.
    Feeding a back-converted lost track straight into ``tick`` reintroduces one
    level down exactly the laundering that returning ``None`` would have caused.

    Lossy / assumed, in this direction
    ----------------------------------
    * ``observation.source`` (:class:`SourceIdentity`) is **dropped** --
      ``TrackingResult`` has no identity field at all. The caller holds the
      packet the observation came from and must keep the association itself; a
      round-trip through :func:`tracking_observation_from_result` therefore
      requires the source to be supplied again.
    * ``observation.covariance`` is **dropped** -- there is no field for it and
      stuffing a 6x6 array into ``raw_stats`` would break
      ``to_tensorboard_dict``-adjacent consumers that assume scalars.
    * ``status`` collapses to a boolean ``valid``. The exact status is preserved
      in ``raw_stats[TRACKING_STATUS_STAT_KEY]``, which a status-aware consumer
      can read and which :func:`tracking_observation_from_result` restores, so
      ``observation -> result -> observation`` keeps LOST/INITIALIZING/DEGRADED
      intact. A consumer that reads only ``valid`` still sees the collapse --
      that is inherent to the target type.
    * ``confidence=None`` is materialized to a real number per the table above --
      an assumption, not a measurement, and an **irreversible** one: "the provider
      reported no confidence at all" cannot be recovered from the result, so it
      does not survive a round trip.
    * Diagnostics entries whose keys match a telemetry field are promoted to that
      typed field only when they coerce to ``float``; a non-numeric value is left
      in ``raw_stats`` untouched rather than crashing the conversion.

    Args:
        observation: The canonical observation to adapt.

    Returns:
        The equivalent :class:`TrackingResult`.
    """
    diagnostics: Dict[str, Any] = dict(observation.diagnostics)

    telemetry: Dict[str, float] = {}
    for field in _TELEMETRY_FIELDS:
        if field not in diagnostics:
            continue
        try:
            telemetry[field] = float(diagnostics[field])
        except (TypeError, ValueError):
            # Not a scalar after all -- leave it in raw_stats rather than
            # dropping real provenance or raising on a telemetry quirk.
            continue
        del diagnostics[field]

    if observation.confidence is not None:
        confidence = min(1.0, max(0.0, float(observation.confidence)))
    else:
        confidence = _DEFAULT_CONFIDENCE_BY_STATUS[TrackingStatus(observation.status)]

    raw_stats: Dict[str, Any] = dict(diagnostics)
    raw_stats[TRACKING_STATUS_STAT_KEY] = TrackingStatus(observation.status).value

    return TrackingResult(
        pose=observation.pose,  # type: ignore[arg-type]  # None iff not usable; see docstring
        confidence=confidence,
        valid=observation.usable,
        method=observation.provider,
        raw_stats=raw_stats,
        **telemetry,
    )
