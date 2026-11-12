"""Online mapping clocks and work-queue state."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class OnlineClock:
    """Global online mapping clock tracking iteration, frame, and time progress.

    Attributes:
        session_step: Current training iteration index (0-indexed).
        frame_idx: Index of the most recently ingested source frame.
        keyframe_idx: Count of frames selected for training (subset of frame_idx).
        wall_time_s: Elapsed wall-clock time since training session start, in seconds.
        map_version: Monotonic version identifier incremented on structural changes
                     to the map (e.g., Gaussians pruned/inserted/relocated). Lets
                     observers detect when their cached state became invalid.
    """
    session_step: int = 0
    frame_idx: int = 0
    keyframe_idx: int = 0
    wall_time_s: float = 0.0
    map_version: int = 0


@dataclass
class RegionState:
    """Lifecycle and optimization state for a spatial region or Gaussian cohort.

    Attributes:
        birth_frame: Index of the source frame where this region was first observed.
        birth_step: Training iteration at which this region was initialized.
        last_seen_frame: Index of the most recent frame where this region was active.
        last_optimized_step: Training iteration at which region parameters were last
                             optimized.
        support_count: Number of observations (e.g., depth measurements, keyframes)
                       supporting this region, used to estimate reconstruction confidence.
        local_refine_steps: Count of local refinement iterations applied to this region
                            within the current training window.
        dirty_version: Version counter tracking invalidation of this region's cached
                       state; incremented when parameters change.
        maturity_score: Scalar in [0, 1] indicating the region's maturity. Mature regions
                        (high score) are stable and well-optimized; young regions (low
                        score) are still converging.
    """
    birth_frame: int = 0
    birth_step: int = 0
    last_seen_frame: int = 0
    last_optimized_step: int = 0
    support_count: int = 0
    local_refine_steps: int = 0
    dirty_version: int = 0
    maturity_score: float = 0.0


@dataclass
class MapWorkItem:
    """Work item for the map optimization queue.

    Represents a spatial region or Gaussian cohort that requires processing
    (e.g., depth insertion, MCMC mutation, re-optimization).

    Attributes:
        item_type: String label for the work type (e.g., 'insertion', 'relocation',
                   'refinement'). Controls how the work item is processed.
        priority: Scalar priority value for queue ordering; higher priority items
                  are processed first. Typically derived from region maturity,
                  support count, or time since last optimization.
        region_id: Opaque identifier for the spatial region or Gaussian cohort.
                   Used to track which region the work affects.
        created_at_step: Training iteration at which this work item was created.
    """
    item_type: str
    priority: float
    region_id: Any
    created_at_step: int


class DirtyRegionQueue:
    """Max-priority work queue for dirty (out-of-sync) spatial regions.

    Maintains a binary heap of work items ordered by priority (descending).
    Higher-priority items are popped first, enabling efficient scheduling of
    region refinement and re-optimization during online mapping.

    The queue uses a tie-breaking counter to ensure stable ordering when items
    have equal priority: items with the same priority are popped in creation order
    (FIFO), not arbitrarily.
    """
    def __init__(self):
        """Initialize an empty priority queue."""
        self._queue = []
        self._counter = 0

    def push(self, item: MapWorkItem):
        """Add a work item to the queue.

        Args:
            item: The MapWorkItem to enqueue. Its `priority` field determines
                  its position in the heap. Higher priority items are popped first.
        """
        self._counter += 1
        heapq.heappush(self._queue, (-item.priority, self._counter, item))

    def pop(self) -> Optional[MapWorkItem]:
        """Remove and return the highest-priority work item.

        Returns:
            The MapWorkItem with the highest priority, or None if the queue
            is empty. Ties are broken by creation order (items added earlier
            are returned first).
        """
        if not self._queue:
            return None
        _, _, item = heapq.heappop(self._queue)
        return item

    def __len__(self):
        """Return the number of items currently in the queue."""
        return len(self._queue)

    def resume_state_dict(self) -> dict:
        """Capture portable queue state using stable camera source indices."""
        from eligsir.utils.partition_eval import camera_source_index

        items = []
        for negative_priority, fifo_counter, item in self._queue:
            source_index = camera_source_index(item.region_id)
            if source_index is None:
                raise ValueError("dirty-region camera has no stable source index")
            if negative_priority != -item.priority:
                raise ValueError("dirty-region heap priority is inconsistent")
            items.append(
                {
                    "item_type": item.item_type,
                    "priority": item.priority,
                    "source_index": int(source_index),
                    "created_at_step": item.created_at_step,
                    "fifo_counter": fifo_counter,
                }
            )
        return {
            "owner_schema": "dirty_region_queue_v1",
            "counter": self._counter,
            "items": sorted(items, key=lambda item: item["fifo_counter"]),
        }

    def load_resume_state_dict(self, payload, resolver) -> None:
        """Validate and resolve the entire queue before mutating live state."""
        from eligsir.utils.partition_eval import camera_source_index

        required = {"owner_schema", "counter", "items"}
        if (
            not isinstance(payload, dict)
            or payload.get("owner_schema") != "dirty_region_queue_v1"
            or set(payload) != required
            or not callable(resolver)
        ):
            raise ValueError("incompatible dirty-region queue resume state")
        counter = payload["counter"]
        if type(counter) is not int or counter < 0:
            raise ValueError("dirty-region FIFO counter must be non-negative")
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise ValueError("dirty-region items must be a list")

        restored = []
        fifo_counters = set()
        item_keys = {
            "item_type", "priority", "source_index", "created_at_step", "fifo_counter",
        }
        for raw in raw_items:
            if not isinstance(raw, dict) or set(raw) != item_keys:
                raise ValueError("invalid dirty-region item state")
            item_type = raw["item_type"]
            if not isinstance(item_type, str) or not item_type:
                raise ValueError("dirty-region item type must be non-empty")
            priority = raw["priority"]
            if (
                isinstance(priority, bool)
                or not isinstance(priority, (int, float))
                or not math.isfinite(priority)
            ):
                raise ValueError("dirty-region priority must be finite")
            source_index = raw["source_index"]
            if type(source_index) is not int or source_index < 0:
                raise ValueError("dirty-region source index must be non-negative")
            created_at_step = raw["created_at_step"]
            if type(created_at_step) is not int or created_at_step < 0:
                raise ValueError("dirty-region creation step must be non-negative")
            fifo_counter = raw["fifo_counter"]
            if (
                type(fifo_counter) is not int
                or fifo_counter <= 0
                or fifo_counter > counter
                or fifo_counter in fifo_counters
            ):
                raise ValueError("dirty-region FIFO counter is invalid")
            camera = resolver(source_index)
            if camera is None or camera_source_index(camera) != source_index:
                raise ValueError(
                    f"could not resolve dirty-region source index {source_index}"
                )
            fifo_counters.add(fifo_counter)
            item = MapWorkItem(
                item_type=item_type,
                priority=float(priority),
                region_id=camera,
                created_at_step=created_at_step,
            )
            restored.append((-item.priority, fifo_counter, item))

        heapq.heapify(restored)
        self._queue = restored
        self._counter = counter
