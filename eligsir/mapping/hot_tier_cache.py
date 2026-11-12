"""Bounded, lazy, byte-budgeted multi-tier HOT supervision residency policy.

Before this module the canonical (``production_application``) supervision path
retained **exactly one** materialized resolution tier per HOT frame:
``FrameObservationDB.activate_supervision`` published
``{tier: CacheHandle}`` and ``attach_resolution_cache`` replaced the record's
``resolution_cache`` dict wholesale.  A scheduler whose tier choice oscillates
for the same identity (quarter -> half -> quarter across successive replay
draws) therefore paid the full preparation cost -- reload, ``F.interpolate``
resize, staging copy, H2D transfer -- on *every* switch, even though that exact
tier had been materialized moments earlier.

This module is the residency policy that lets a HOT frame retain more than one
tier without turning the HOT set into an unbounded pyramid.  Three properties
are load-bearing and each is regression-tested:

**Lazy, never eager.**  Nothing here materializes anything.  A tier enters the
cache only because a real transaction already requested and materialized it.
There is deliberately no "build native+half+quarter for every admitted frame"
path: the large majority of frames are never replayed at a second tier, and
precomputing a pyramid for them would spend CPU, H2D bandwidth and VRAM to
cache representations nobody asks for.

**Bounded per frame.**  A frame may hold at most
``MAX_RETAINED_TIERS_PER_FRAME`` concurrently materialized tiers.  A third
distinct tier evicts that frame's least valuable retained tier rather than
growing to three.

**Byte-budgeted across the whole HOT set.**  The count-bounded
``ActiveSupervisionCache`` bounds how many *frames* are HOT; it says nothing
about bytes, and a native-tier frame costs 16x a quarter-tier one.  The extra
residency this module permits is therefore bounded in bytes, measured from the
real tensors (``rgb.nbytes + depth.nbytes``), never from a hardcoded
per-resolution table.

Eviction is value-based rather than pure LRU::

    value = P_reuse * T_prepare_saved / M_tier

* ``P_reuse`` -- cheap recency+frequency estimate, ``access_count / (1 + age)``
  in admission epochs.  It reuses the accesses this cache already observes
  (every ``resident_supervision_for_source`` hit touches the entry) instead of
  introducing a second, divergent replay-likelihood signal.
* ``T_prepare_saved`` -- the *measured* preparation cost a hit on this tier
  avoids, supplied by the caller from the same bounded rolling statistics the
  P0 allocator keys on (``TierCostEstimator``'s residency-split prepare
  windows, see ``scheduler/tier_cost_model.py``).  With no evidence yet the
  provider returns ``None`` and a neutral constant is used, so the ordering
  degenerates to ``1 / M_tier``.
* ``M_tier`` -- the entry's real byte footprint, in MiB.

Quarter tiers surviving eviction pressure falls *out of* that formula (a
quarter costs ~1/16 the bytes of native for the same ``P_reuse``); it is not
hardcoded as a preference anywhere, so a device where quarter turns out to save
no preparation time at all will stop favouring it on its own.

Measured evidence (PR #1252,
``docs/architecture/2026-08-29-resolution-prepare-cost-measurement.md``; TUM
640x480 RGB-D, C=8, ROCm 8060S, p50) grounds every term above and, importantly,
contradicts the intuitive shape of the ``T_prepare_saved`` axis:

===============================  =========  =========  =========
condition                        native     half       quarter
===============================  =========  =========  =========
HOT hit (already at that tier)     0.021ms    0.022ms    0.022ms
CPU-resident tier miss             5.06ms    12.29ms     8.17ms
cold reload (full decode)         55.18ms    62.74ms    59.72ms
===============================  =========  =========  =========

* A HOT hit is ~0.02ms and **flat in tier and in resolution** -- it touches
  nothing.  A miss on the production path is 5-12ms at 640x480 and 36-102ms at
  1920x1440, i.e. 43-46% of the corresponding GPU kernel cost for half/quarter.
  Retaining a second tier is therefore worth real transaction time, which is
  the entire reason this module exists.
* **Preparation cost is NOT monotonic in tier**: half (12.29ms) costs more than
  quarter (8.17ms), which costs more than native (5.06ms), because the resize
  is a CPU ``F.interpolate`` whose cost tracks *output* pixel count and which
  ``native`` skips entirely (``factor == 1`` passthrough).  A retention policy
  that assumed "lower tier = more expensive to rebuild" would keep exactly the
  wrong tier.  This is why ``T_prepare_saved`` must come from measurement and
  never from ordinal position -- and with these numbers the formula ranks
  native *lowest* (cheapest to rebuild, largest footprint), which is correct.
* Frame-count bounding is not a memory bound: the same 400-frame LRU cap holds
  1.83 GiB at TUM resolution and **16.5 GiB** at ScanNet++ resolution.  That is
  the concrete case for budgeting bytes here rather than frames.
* Deriving a tier from the canonical native view rather than from another
  cached tier is 14-72x cheaper when native is GPU-resident, so the
  never-chain invariant in ``resolution_materialization`` costs nothing to
  keep and is preserved unchanged.

This module makes no resolution *selection* decision and must never acquire
one: ``WorkBudgetOwner`` is the single owner of which tier a transaction runs
at (PR #1251).  Retention only changes what a chosen tier *costs*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#: How many materialized tiers one HOT frame may retain at once.  Two is the
#: smallest bound that absorbs the oscillation this module exists for (A -> B
#: -> A) while refusing an unbounded pyramid; it is named rather than inlined
#: so it stays a single tunable.
MAX_RETAINED_TIERS_PER_FRAME: int = 2

#: Neutral stand-in for ``T_prepare_saved`` before any preparation cost has
#: been measured for a tier.  Its absolute value is irrelevant -- with no
#: evidence every tier shares it and the ordering collapses to ``1 / M_tier``.
DEFAULT_PREPARE_MS: float = 1.0

_MIB = float(1024 * 1024)

#: Eviction reasons, kept as a small closed vocabulary so telemetry can
#: distinguish "this frame asked for a third tier" from "the HOT set as a whole
#: ran out of bytes" -- they have different fixes.
EVICT_PER_FRAME_CAP = "per_frame_tier_cap"
EVICT_BYTE_BUDGET = "byte_budget"
EVICT_ADMISSION_REFUSED = "admission_refused"
EVICT_FRAME_DROPPED = "frame_dropped"


@dataclass
class HotTierEntry:
    """One materialized (frame, tier) representation retained in the HOT set."""

    frame_id: int
    tier: str
    nbytes: int
    access_count: int
    last_access_epoch: int

    @property
    def key(self) -> Tuple[int, str]:
        return (self.frame_id, self.tier)


@dataclass(frozen=True)
class HotTierEviction:
    """A retention decision the caller must now mirror in its storage."""

    frame_id: int
    tier: str
    nbytes: int
    reason: str
    value: float


def entry_bytes(handle: object) -> int:
    """Real byte footprint of one cached tier handle.

    Measured from the tensors themselves (element count x itemsize) rather
    than from a per-resolution table, so a change of dtype, channel count or
    an added modality is accounted for without editing a constant.
    """

    total = 0
    for name in ("rgb", "depth"):
        payload = getattr(handle, name, None)
        if payload is None:
            continue
        nbytes = getattr(payload, "nbytes", None)
        if nbytes is None:
            try:  # torch tensors expose element_size()/nelement()
                nbytes = payload.element_size() * payload.nelement()
            except Exception:
                continue
        try:
            total += int(nbytes)
        except (TypeError, ValueError):
            continue
    return max(0, total)


class HotTierCache:
    """Bounded per-frame + byte-budgeted, value-evicted tier residency policy.

    Pure bookkeeping: it owns no tensors and performs no transfers.  The
    storage owner (``FrameObservationDB``) calls :meth:`admit` after a tier has
    already been materialized and mirrors the returned evictions into its own
    records.
    """

    def __init__(
        self,
        *,
        max_tiers_per_frame: int = MAX_RETAINED_TIERS_PER_FRAME,
        max_bytes: Optional[int] = None,
        prepare_cost_provider: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        if isinstance(max_tiers_per_frame, bool) or int(max_tiers_per_frame) < 1:
            raise ValueError("max_tiers_per_frame must be a positive integer")
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or int(max_bytes) < 0:
                raise ValueError("max_bytes must be a non-negative integer or None")
        self._max_tiers_per_frame = int(max_tiers_per_frame)
        self._max_bytes = None if max_bytes is None else int(max_bytes)
        self._prepare_cost_provider = prepare_cost_provider
        self._entries: Dict[Tuple[int, str], HotTierEntry] = {}
        self._by_frame: Dict[int, Dict[str, HotTierEntry]] = {}
        self._epoch = 0
        self._total_bytes = 0
        self.evictions = 0
        self.evictions_by_reason: Dict[str, int] = {}

    # ------------------------------------------------------------------ read

    @property
    def max_tiers_per_frame(self) -> int:
        return self._max_tiers_per_frame

    @property
    def max_bytes(self) -> Optional[int]:
        return self._max_bytes

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def tiers_for_frame(self, frame_id: int) -> Tuple[str, ...]:
        return tuple(sorted(self._by_frame.get(int(frame_id), {})))

    def get(self, frame_id: int, tier: str) -> Optional[HotTierEntry]:
        return self._entries.get((int(frame_id), str(tier)))

    def prepare_cost_ms(self, tier: str) -> float:
        """Measured preparation cost a hit on ``tier`` avoids, or the neutral
        default when nothing has been measured for it yet."""

        provider = self._prepare_cost_provider
        if provider is None:
            return DEFAULT_PREPARE_MS
        try:
            measured = provider(str(tier))
        except Exception:
            return DEFAULT_PREPARE_MS
        if measured is None:
            return DEFAULT_PREPARE_MS
        try:
            measured = float(measured)
        except (TypeError, ValueError):
            return DEFAULT_PREPARE_MS
        if not (measured > 0.0) or measured != measured or measured == float("inf"):
            return DEFAULT_PREPARE_MS
        return measured

    def reuse_probability(self, entry: HotTierEntry) -> float:
        """Cheap recency+frequency proxy for ``P_reuse``.

        Deliberately derived from accesses this policy already sees rather than
        from a new replay-likelihood model: a second, independently-drifting
        estimate of the same quantity is a worse failure mode than a coarse one.
        """

        age = max(0, self._epoch - int(entry.last_access_epoch))
        return float(max(1, int(entry.access_count))) / (1.0 + float(age))

    def entry_value(self, entry: HotTierEntry) -> float:
        """``P_reuse * T_prepare_saved / M_tier`` -- higher survives longer."""

        megabytes = max(float(entry.nbytes), 1.0) / _MIB
        return (
            self.reuse_probability(entry)
            * self.prepare_cost_ms(entry.tier)
            / megabytes
        )

    def snapshot(self) -> dict:
        """Bounded telemetry view of current retention."""

        return {
            "entries": len(self._entries),
            "total_bytes": int(self._total_bytes),
            "max_bytes": self._max_bytes,
            "max_tiers_per_frame": self._max_tiers_per_frame,
            "frames": len(self._by_frame),
            "multi_tier_frames": sum(
                1 for tiers in self._by_frame.values() if len(tiers) > 1
            ),
            "evictions": int(self.evictions),
            "evictions_by_reason": dict(self.evictions_by_reason),
        }

    # ----------------------------------------------------------------- write

    def touch(self, frame_id: int, tier: Optional[str] = None) -> bool:
        """Record a cache hit, refreshing recency and frequency.

        ``tier is None`` touches every tier retained for the frame, which is
        what a frame-level LRU refresh means.
        """

        frame_id = int(frame_id)
        tiers = self._by_frame.get(frame_id)
        if not tiers:
            return False
        self._epoch += 1
        if tier is None:
            for entry in tiers.values():
                entry.access_count += 1
                entry.last_access_epoch = self._epoch
            return True
        entry = tiers.get(str(tier))
        if entry is None:
            return False
        entry.access_count += 1
        entry.last_access_epoch = self._epoch
        return True

    def admit(
        self, frame_id: int, tier: str, nbytes: int
    ) -> Tuple[HotTierEviction, ...]:
        """Retain one already-materialized tier, returning what must be dropped.

        The returned evictions may include the just-admitted entry itself when
        it alone exceeds the whole byte budget: the transaction still runs (the
        camera owns its tensors), it simply is not retained, so the budget is
        never exceeded.
        """

        frame_id = int(frame_id)
        tier = str(tier)
        nbytes = max(0, int(nbytes))
        self._epoch += 1

        evicted: List[HotTierEviction] = []
        existing = self._entries.get((frame_id, tier))
        if existing is not None:
            self._total_bytes += nbytes - existing.nbytes
            existing.nbytes = nbytes
            existing.access_count += 1
            existing.last_access_epoch = self._epoch
        else:
            entry = HotTierEntry(
                frame_id=frame_id,
                tier=tier,
                nbytes=nbytes,
                access_count=1,
                last_access_epoch=self._epoch,
            )
            self._insert(entry)
            # Per-frame cap: a third distinct tier evicts this frame's least
            # valuable retained tier, never grows the frame's pyramid.
            frame_tiers = self._by_frame[frame_id]
            while len(frame_tiers) > self._max_tiers_per_frame:
                # The tier just requested is the one in use right now; evicting
                # it would defeat the admission entirely.
                victim = self._least_valuable(
                    frame_tiers.values(), exclude=(frame_id, tier)
                )
                if victim is None:
                    break
                evicted.append(self._drop(victim, EVICT_PER_FRAME_CAP))
                frame_tiers = self._by_frame.get(frame_id, {})

        evicted.extend(self._enforce_byte_budget(protect=(frame_id, tier)))
        return tuple(evicted)

    def discard(
        self, frame_id: int, tier: Optional[str] = None
    ) -> Tuple[HotTierEviction, ...]:
        """Forget retention state for a frame (or one of its tiers)."""

        frame_id = int(frame_id)
        tiers = self._by_frame.get(frame_id)
        if not tiers:
            return ()
        if tier is None:
            victims = list(tiers.values())
        else:
            entry = tiers.get(str(tier))
            victims = [] if entry is None else [entry]
        dropped = [self._drop(entry, EVICT_FRAME_DROPPED, count=False) for entry in victims]
        return tuple(dropped)

    # -------------------------------------------------------------- internal

    def _insert(self, entry: HotTierEntry) -> None:
        self._entries[entry.key] = entry
        self._by_frame.setdefault(entry.frame_id, {})[entry.tier] = entry
        self._total_bytes += entry.nbytes

    def _drop(
        self, entry: HotTierEntry, reason: str, *, count: bool = True
    ) -> HotTierEviction:
        self._entries.pop(entry.key, None)
        tiers = self._by_frame.get(entry.frame_id)
        if tiers is not None:
            tiers.pop(entry.tier, None)
            if not tiers:
                self._by_frame.pop(entry.frame_id, None)
        self._total_bytes = max(0, self._total_bytes - entry.nbytes)
        if count:
            self.evictions += 1
            self.evictions_by_reason[reason] = (
                self.evictions_by_reason.get(reason, 0) + 1
            )
        return HotTierEviction(
            frame_id=entry.frame_id,
            tier=entry.tier,
            nbytes=entry.nbytes,
            reason=reason,
            value=self.entry_value(entry),
        )

    def _least_valuable(
        self,
        entries: Iterable[HotTierEntry],
        *,
        exclude: Optional[Tuple[int, str]],
    ) -> Optional[HotTierEntry]:
        candidates = [
            entry for entry in entries if exclude is None or entry.key != exclude
        ]
        if not candidates:
            return None
        # Ties resolve by (oldest access, then frame/tier) so eviction order is
        # deterministic and a test can assert it.
        return min(
            candidates,
            key=lambda entry: (
                self.entry_value(entry),
                entry.last_access_epoch,
                entry.frame_id,
                entry.tier,
            ),
        )

    def _enforce_byte_budget(
        self, *, protect: Optional[Tuple[int, str]]
    ) -> Tuple[HotTierEviction, ...]:
        if self._max_bytes is None:
            return ()
        evicted: List[HotTierEviction] = []
        while self._total_bytes > self._max_bytes:
            victim = self._least_valuable(self._entries.values(), exclude=protect)
            if victim is None:
                break
            evicted.append(self._drop(victim, EVICT_BYTE_BUDGET))
        if self._total_bytes > self._max_bytes and protect is not None:
            # Only the protected entry is left and it alone busts the budget:
            # refuse to retain it rather than exceed the configured bound.
            entry = self._entries.get(protect)
            if entry is not None:
                evicted.append(self._drop(entry, EVICT_ADMISSION_REFUSED))
        return tuple(evicted)


__all__ = [
    "DEFAULT_PREPARE_MS",
    "EVICT_ADMISSION_REFUSED",
    "EVICT_BYTE_BUDGET",
    "EVICT_FRAME_DROPPED",
    "EVICT_PER_FRAME_CAP",
    "HotTierCache",
    "HotTierEntry",
    "HotTierEviction",
    "MAX_RETAINED_TIERS_PER_FRAME",
    "entry_bytes",
]
