"""Tiered frame archive with a real bounded GOLDEN protection policy."""
from __future__ import annotations

from collections import OrderedDict
import os
from typing import List, Optional


class _GoldenAwareOrderedDict(OrderedDict):
    """ICECOLD LRU whose eviction is delegated to its FrameArchive owner."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def popitem(self, last=True):
        if last:
            return super().popitem(last=True)
        self._owner._refresh_golden_from_cache()
        for key in tuple(self.keys()):
            if not self._owner.is_golden(key):
                value = self.pop(key)
                return key, value
        # A deliberately tiny icecold limit can be smaller than the GOLDEN
        # budget. Preserve bounded memory deterministically by demoting the
        # oldest protected identity and report the conflict; never silently
        # claim it remained protected.
        if self:
            key = next(iter(self.keys()))
            self._owner.unmark_golden(key)
            self._owner.forced_golden_demotions += 1
            value = self.pop(key)
            return key, value
        raise KeyError("dictionary is empty")


class FrameArchive:
    """Own COLD/ICECOLD storage plus bounded global GOLDEN anchors.

    GOLDEN is no longer schema-only. Before an ICECOLD LRU eviction, the
    archive deterministically protects an evenly spread global set of source
    identities. This is an archive policy, not optimizer policy: callers still
    insert/evict through the exact same ``OrderedDict`` surface used by
    ``StreamingScene``. The default budget is 16; ResearchFlow/control runs may
    set ``ELIGSIR_GOLDEN_BUDGET=0`` to disable or another bounded integer
    without adding a second runtime state path.
    """

    def __init__(self):
        self.inactive_frame_cache: "OrderedDict[int, object]" = OrderedDict()
        self.golden_indices: "OrderedDict[int, None]" = OrderedDict()
        self.golden_budget = max(0, int(os.environ.get("ELIGSIR_GOLDEN_BUDGET", "16")))
        self.icecold_frame_cache: "OrderedDict[int, object]" = _GoldenAwareOrderedDict(self)
        self.forced_golden_demotions = 0
        self.golden_refreshes = 0
        self.decode_cache = None
        self.lazy_frames: bool = False
        self.all_frames: "Optional[List[object]]" = None

    def _refresh_golden_from_cache(self) -> None:
        """Protect an evenly distributed trajectory skeleton before eviction."""
        budget = min(self.golden_budget, len(self.icecold_frame_cache))
        if budget <= 0:
            self.golden_indices.clear()
            return
        keys = tuple(int(k) for k in self.icecold_frame_cache.keys())
        if budget >= len(keys):
            selected = keys
        elif budget == 1:
            selected = keys[:1]
        else:
            last = len(keys) - 1
            positions = tuple((i * last) // (budget - 1) for i in range(budget))
            selected = tuple(keys[pos] for pos in positions)
        selected_set = set(selected)
        for idx in tuple(self.golden_indices.keys()):
            if idx not in selected_set:
                self.golden_indices.pop(idx, None)
        for idx in selected:
            if idx not in self.golden_indices:
                self.golden_indices[idx] = None
        self.golden_refreshes += 1

    def mark_golden(self, index: int, *, max_golden: int | None = None) -> tuple[int, ...]:
        """Explicitly protect ``index`` under the same bounded budget."""
        idx = int(index)
        budget = self.golden_budget if max_golden is None else max(0, int(max_golden))
        if budget == 0:
            self.golden_indices.pop(idx, None)
            return ()
        self.golden_indices[idx] = None
        self.golden_indices.move_to_end(idx)
        demoted: list[int] = []
        while len(self.golden_indices) > budget:
            old, _ = self.golden_indices.popitem(last=False)
            demoted.append(int(old))
        return tuple(demoted)

    def unmark_golden(self, index: int) -> None:
        self.golden_indices.pop(int(index), None)

    def is_golden(self, index: int) -> bool:
        return int(index) in self.golden_indices

    def evict_icecold_to_limit(self, max_size: int) -> tuple[int, ...]:
        """LRU-evict non-GOLDEN frames until the explicit size bound is met."""
        limit = max(0, int(max_size))
        evicted: list[int] = []
        while len(self.icecold_frame_cache) > limit:
            idx, _ = self.icecold_frame_cache.popitem(last=False)
            evicted.append(int(idx))
        return tuple(evicted)

    def report_block(self) -> dict:
        return {
            "cold_count": len(self.inactive_frame_cache),
            "icecold_count": len(self.icecold_frame_cache),
            "golden_count": len(self.golden_indices),
            "golden_budget": self.golden_budget,
            "golden_indices": list(self.golden_indices.keys()),
            "golden_refreshes": self.golden_refreshes,
            "forced_golden_demotions": self.forced_golden_demotions,
        }
