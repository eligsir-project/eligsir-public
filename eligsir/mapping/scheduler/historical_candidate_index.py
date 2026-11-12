"""Bounded, metadata-only historical replay candidate shortlist.

This index retains identity metadata for every registered historical frame, but
keeps only bounded views for shortlist construction.  It deliberately does not
own cameras, tensors, or image-cache handles: materialization remains the
responsibility of the final replay sampler.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Hashable, Iterable, Mapping, Sequence


MAX_REPLAY_IDENTITY_FOCUS_UPDATES = 32


@dataclass(frozen=True)
class HistoricalFrameMetadata:
    """CPU-resident identity and replay metadata for one historical frame."""

    frame_id: int
    region_id: Hashable
    base_weight: float = 1.0
    protected: bool = False


@dataclass(frozen=True)
class HistoricalCandidateConfig:
    """Fixed capacities for the metadata-only candidate lanes."""

    shortlist_size: int = 128
    recent_lane_size: int = 32
    focus_lane_size: int = 32
    protected_lane_size: int = 16
    global_reservoir_size: int = 64
    global_floor: int = 16

    @classmethod
    def for_batch_size(cls, batch_size: int) -> "HistoricalCandidateConfig":
        """Lane capacities for a prepared plan of ``batch_size`` sources.

        The shortlist caller hands the index up to ``batch_size`` local ids
        (supervision + replay sources), so the recent and focus lanes must hold
        at least that many; the defaults (32) stay the floor and the shortlist
        keeps its default size. ``batch_size`` above ``shortlist_size`` cannot
        be represented and is rejected instead of silently clamped.
        """
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        default = cls()
        if batch_size > default.shortlist_size:
            raise ValueError(
                f"prepared batch size {batch_size} > {default.shortlist_size} is not "
                "supported by the historical candidate index"
            )
        lane = max(default.recent_lane_size, batch_size)
        return cls(
            recent_lane_size=lane,
            focus_lane_size=max(default.focus_lane_size, batch_size),
        )

    def __post_init__(self) -> None:
        if not 64 <= self.shortlist_size <= 128:
            raise ValueError("shortlist_size must be in [64, 128]")
        for name in (
            "recent_lane_size",
            "focus_lane_size",
            "protected_lane_size",
            "global_reservoir_size",
            "global_floor",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.global_floor > self.shortlist_size:
            raise ValueError("global_floor cannot exceed shortlist_size")
        if self.global_reservoir_size < self.global_floor:
            raise ValueError("global_reservoir_size must cover global_floor")
        if self.global_floor + 1 + self.focus_lane_size > self.shortlist_size:
            raise ValueError(
                "shortlist_size must reserve the global floor, reservation, and focus lane"
            )
        for name in (
            "recent_lane_size",
            "focus_lane_size",
            "protected_lane_size",
            "global_reservoir_size",
        ):
            if getattr(self, name) > self.shortlist_size:
                raise ValueError(f"{name} cannot exceed shortlist_size")


@dataclass(frozen=True)
class HistoricalCandidateEvidence:
    """Why a metadata identity reached a shortlist on this scheduler tick."""

    lanes: tuple[str, ...]
    region_id: Hashable
    focus_weight: float
    snapshot_id: Hashable
    epoch: int


@dataclass(frozen=True)
class HistoricalReplayCandidate:
    """Final-stage sampler input: identity, sampling weight, and provenance."""

    frame_id: int
    weight: float
    evidence: HistoricalCandidateEvidence

    def as_weighted_pair(self) -> tuple[int, float]:
        """Return the compact ``(identity, weight)`` form expected by samplers."""
        return self.frame_id, self.weight


class HistoricalCandidateIndex:
    """Incremental index whose shortlist work is bounded by configuration.

    ``focus_region_weights`` must be pre-ranked and no longer than
    ``focus_lane_size``.  That boundary keeps each tick independent of the
    total historical-frame count; registration alone performs history growth.
    """

    def __init__(
        self,
        *,
        config: HistoricalCandidateConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or HistoricalCandidateConfig()
        self._seed = int(seed)
        self._metadata: dict[int, HistoricalFrameMetadata] = {}
        self._history_ids: list[int] = []
        self._anchor_id: int | None = None
        self._recent_ids: deque[int] = deque(maxlen=self.config.recent_lane_size)
        self._region_recent_ids: dict[Hashable, deque[int]] = {}
        self._protected_ranks: dict[int, int] = {}
        self._protected_heap: list[tuple[int, int]] = []
        self._protected_ranked_ids: tuple[int, ...] = ()
        self._global_ranks: dict[int, int] = {}
        self._global_heap: list[tuple[int, int]] = []
        self._global_ranked_ids: tuple[int, ...] = ()

    @property
    def history_size(self) -> int:
        """Count of retained metadata identities; this is never lane-bounded."""
        return len(self._metadata)

    def metadata_for(self, frame_id: int) -> HistoricalFrameMetadata:
        """Return retained metadata for an exact historical identity."""
        return self._metadata[frame_id]

    def register(self, metadata: HistoricalFrameMetadata) -> None:
        """Register one unique historical identity and update bounded lane views."""
        if metadata.frame_id in self._metadata:
            raise ValueError(f"historical frame {metadata.frame_id} already registered")
        if metadata.base_weight <= 0.0:
            raise ValueError("base_weight must be positive")

        frame_id = int(metadata.frame_id)
        self._metadata[frame_id] = metadata
        self._history_ids.append(frame_id)
        if self._anchor_id is None:
            # A permanent oldest-history anchor makes the start of a trajectory
            # reachable even after any rolling lane has turned over.
            self._anchor_id = frame_id
        self._recent_ids.append(frame_id)
        region_recent = self._region_recent_ids.get(metadata.region_id)
        if region_recent is None:
            region_recent = deque(maxlen=self.config.focus_lane_size)
            self._region_recent_ids[metadata.region_id] = region_recent
        region_recent.append(frame_id)

        rank = self._stable_rank("global", frame_id)
        if self._insert_ranked(
            frame_id,
            rank,
            self.config.global_reservoir_size,
            self._global_ranks,
            self._global_heap,
        ):
            self._global_ranked_ids = self._ranked_ids(self._global_ranks)
        if metadata.protected:
            if self._insert_ranked(
                frame_id,
                self._stable_rank("protected", frame_id),
                self.config.protected_lane_size,
                self._protected_ranks,
                self._protected_heap,
            ):
                self._protected_ranked_ids = self._ranked_ids(self._protected_ranks)

    def update_base_weight(self, frame_id: int, base_weight: float) -> bool:
        """Overwrite one registered identity's persistent replay-priority prior.

        ``base_weight`` (issue #953/#1191) is the durable, train-time-only
        usefulness prior a scheduler tick can set once real evidence about a
        frame exists (e.g. its own last-observed training/replay loss under
        the current map -- see
        ``eligsir.mapping.scheduler.historical_usefulness``). It is
        distinct from ``focus_by_id`` in :meth:`shortlist`, which is a
        temporary per-tick *boost* over this base; this call changes the
        floor every future tick starts from, not one tick's boost.

        Returns ``False`` for an unregistered/already-retired ``frame_id``
        rather than raising, mirroring :meth:`retire`'s tolerance of a
        concurrently-retired identity -- a usefulness update racing a
        payload eviction must not crash the caller.
        """

        from dataclasses import replace

        frame_id = int(frame_id)
        if isinstance(base_weight, bool) or not math.isfinite(float(base_weight)):
            raise ValueError("base_weight must be a finite real number")
        if float(base_weight) <= 0.0:
            raise ValueError("base_weight must be positive")
        current = self._metadata.get(frame_id)
        if current is None:
            return False
        self._metadata[frame_id] = replace(current, base_weight=float(base_weight))
        return True

    def retire(self, frame_id: int) -> bool:
        """Drop one identity that can never be materialized again.

        Registration is otherwise permanent: the index is the eventual-
        reachability contract for replay. An identity whose pixels are gone
        (bounded-cold eviction of a non-reloadable payload) is the one case
        where continued membership is not "eventually reachable" but
        "eventually fatal" -- the runtime cannot build a camera for it, so it
        must leave every lane, including the permanent anchor (issue #1289).
        """

        frame_id = int(frame_id)
        if self._metadata.pop(frame_id, None) is None:
            return False
        self._history_ids = [
            item for item in self._history_ids if item != frame_id
        ]
        if frame_id in self._recent_ids:
            self._recent_ids = deque(
                (item for item in self._recent_ids if item != frame_id),
                maxlen=self.config.recent_lane_size,
            )
        for region_id, lane in tuple(self._region_recent_ids.items()):
            if frame_id not in lane:
                continue
            replacement = deque(
                (item for item in lane if item != frame_id),
                maxlen=self.config.focus_lane_size,
            )
            if replacement:
                self._region_recent_ids[region_id] = replacement
            else:
                del self._region_recent_ids[region_id]
        if self._global_ranks.pop(frame_id, None) is not None:
            self._global_heap = [
                item for item in self._global_heap if item[1] != frame_id
            ]
            heapq.heapify(self._global_heap)
            self._global_ranked_ids = self._ranked_ids(self._global_ranks)
        if self._protected_ranks.pop(frame_id, None) is not None:
            self._protected_heap = [
                item for item in self._protected_heap if item[1] != frame_id
            ]
            heapq.heapify(self._protected_heap)
            self._protected_ranked_ids = self._ranked_ids(self._protected_ranks)
        if self._anchor_id == frame_id:
            # Re-elect the oldest surviving identity so the start-of-trajectory
            # anchor lane stays populated instead of pointing at nothing.
            self._anchor_id = self._history_ids[0] if self._history_ids else None
        return True

    def shortlist(
        self,
        *,
        snapshot_id: Hashable,
        epoch: int,
        local_ids: Iterable[int] = (),
        focus_region_weights: Mapping[Hashable, float] | Sequence[tuple[Hashable, float]] = (),
        focus_identity_weights: Mapping[int, float] | Sequence[tuple[int, float]] = (),
        reserved_global_epoch: int | None = None,
    ) -> tuple[HistoricalReplayCandidate, ...]:
        """Build a deterministic, bounded replay shortlist without a history scan.

        ``local_ids`` and ``focus_region_weights`` are intentionally bounded
        call-site inputs.  The latter is pre-ranked by the control plane;
        rejecting an oversized sequence is preferable to silently turning a
        per-tick operation into an O(history) selection pass.
        """
        local = tuple(local_ids)
        if len(local) > self.config.recent_lane_size:
            raise ValueError("local_ids exceeds recent_lane_size")
        focus = (
            tuple(focus_region_weights.items())
            if isinstance(focus_region_weights, Mapping)
            else tuple(focus_region_weights)
        )
        if len(focus) > self.config.focus_lane_size:
            raise ValueError("focus_region_weights exceeds focus_lane_size")
        identity_focus = (
            tuple(focus_identity_weights.items())
            if isinstance(focus_identity_weights, Mapping)
            else tuple(focus_identity_weights)
        )
        if len(identity_focus) > self.config.focus_lane_size:
            raise ValueError("focus_identity_weights exceeds focus_lane_size")
        identity_focus_by_id: dict[int, float] = {}
        for raw_frame_id, raw_weight in identity_focus:
            frame_id = int(raw_frame_id)
            if frame_id not in self._metadata:
                raise ValueError(
                    "focus_identity_weights contains an unknown historical frame"
                )
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or float(raw_weight) < 0.0
            ):
                raise ValueError(
                    "focus_identity_weights must be finite and non-negative"
                )
            identity_focus_by_id[frame_id] = float(raw_weight)
        if reserved_global_epoch is not None:
            if isinstance(reserved_global_epoch, bool) or int(reserved_global_epoch) < 0:
                raise ValueError("reserved_global_epoch must be a non-negative integer")
            reserved_global_epoch = int(reserved_global_epoch)

        focus_ids, focus_by_id = self._focus_selection(focus)
        for frame_id, weight in identity_focus_by_id.items():
            focus_by_id[frame_id] = max(focus_by_id.get(frame_id, 0.0), weight)
        reserved_global_id = self._reserved_global_id(reserved_global_epoch)
        lane_ids: dict[str, tuple[int, ...]] = {
            "global_rotation": self._rotating_global_ids(epoch),
            "reserved_global": (
                () if reserved_global_id is None else (reserved_global_id,)
            ),
            "protected_global": self._protected_ids(),
            "global_reservoir": self._global_ranked_ids,
            "local": tuple(frame_id for frame_id in local if frame_id in self._metadata),
            "recent": tuple(reversed(self._recent_ids)),
            "focus": focus_ids,
            "identity_focus": tuple(identity_focus_by_id),
        }

        evidence_lanes: dict[int, set[str]] = {}
        for lane, frame_ids in lane_ids.items():
            for frame_id in frame_ids:
                evidence_lanes.setdefault(frame_id, set()).add(lane)

        ordered_ids: list[int] = []
        seen: set[int] = set()

        # Rotate the global floor through the complete retained identity list.
        # This is both the starvation guard and the eventual-reachability
        # contract: bounded per-tick work must not turn a fixed reservoir into
        # permanent exclusion from replay.
        self._append_unique(
            ordered_ids,
            seen,
            lane_ids["global_rotation"],
            self.config.global_floor,
        )
        for lane in (
            "reserved_global",
            "identity_focus",
            "local",
            "recent",
            "focus",
            "protected_global",
            "global_reservoir",
        ):
            self._append_unique(ordered_ids, seen, lane_ids[lane], self.config.shortlist_size)

        return tuple(
            HistoricalReplayCandidate(
                frame_id=frame_id,
                weight=float(self._metadata[frame_id].base_weight * (1.0 + focus_by_id.get(frame_id, 0.0))),
                evidence=HistoricalCandidateEvidence(
                    lanes=tuple(sorted(evidence_lanes[frame_id])),
                    region_id=self._metadata[frame_id].region_id,
                    focus_weight=float(focus_by_id.get(frame_id, 0.0)),
                    snapshot_id=snapshot_id,
                    epoch=int(epoch),
                ),
            )
            for frame_id in ordered_ids
        )

    def _protected_ids(self) -> tuple[int, ...]:
        ids: list[int] = []
        if self._anchor_id is not None:
            ids.append(self._anchor_id)
        ids.extend(self._protected_ranked_ids)
        return tuple(ids)

    def _rotating_global_ids(self, epoch: int) -> tuple[int, ...]:
        """Return one bounded contiguous window over all retained identities."""

        population = len(self._history_ids)
        if population == 0:
            return ()
        count = min(self.config.global_floor, population)
        start = (max(0, int(epoch)) * self.config.global_floor) % population
        return tuple(
            self._history_ids[(start + offset) % population]
            for offset in range(count)
        )

    def _reserved_global_id(self, reservation_epoch: int | None) -> int | None:
        """Return one O(1) full-history identity for a realized-selection fence."""

        if reservation_epoch is None or not self._history_ids:
            return None
        return self._history_ids[int(reservation_epoch) % len(self._history_ids)]

    def _focus_selection(
        self, focus: Sequence[tuple[Hashable, float]]
    ) -> tuple[tuple[int, ...], dict[int, float]]:
        ids: list[int] = []
        weights: dict[int, float] = {}
        for region_id, focus_weight in focus:
            for frame_id in reversed(self._region_recent_ids.get(region_id, ())):
                ids.append(frame_id)
                if focus_weight > 0.0:
                    weights[frame_id] = max(weights.get(frame_id, 0.0), float(focus_weight))
                if len(ids) == self.config.focus_lane_size:
                    return tuple(ids), weights
        return tuple(ids), weights

    @staticmethod
    def _append_unique(
        target: list[int], seen: set[int], frame_ids: Iterable[int], limit: int
    ) -> None:
        for frame_id in frame_ids:
            if len(target) == limit:
                return
            if frame_id not in seen:
                target.append(frame_id)
                seen.add(frame_id)

    def _stable_rank(self, lane: str, frame_id: int) -> int:
        digest = hashlib.blake2b(
            f"{self._seed}:{lane}:{frame_id}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big")

    @staticmethod
    def _ranked_ids(ranks: dict[int, int]) -> tuple[int, ...]:
        return tuple(frame_id for frame_id, _rank in sorted(ranks.items(), key=lambda item: (item[1], item[0])))

    @staticmethod
    def _insert_ranked(
        frame_id: int,
        rank: int,
        capacity: int,
        ranks: dict[int, int],
        heap: list[tuple[int, int]],
    ) -> bool:
        if len(ranks) < capacity:
            ranks[frame_id] = rank
            heapq.heappush(heap, (-rank, frame_id))
            return True
        worst_rank = -heap[0][0]
        worst_id = heap[0][1]
        if (rank, frame_id) >= (worst_rank, worst_id):
            return False
        heapq.heapreplace(heap, (-rank, frame_id))
        del ranks[worst_id]
        ranks[frame_id] = rank
        return True
