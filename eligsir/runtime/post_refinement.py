"""Typed, durable boundary between finite-source mapping and refinement."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Mapping


class PostRefinementPhase(str, Enum):
    MAPPING = "mapping"
    SOURCE_COMPLETE = "source_complete"
    REFINEMENT = "refinement"


class PostRefinementPhaseOwner:
    """Authorize replay refinement only from a published mapping endpoint."""

    def __init__(self) -> None:
        self._phase = PostRefinementPhase.MAPPING

    @property
    def phase(self) -> PostRefinementPhase:
        if not isinstance(self._phase, PostRefinementPhase):
            raise TypeError("post-refinement authority requires an exact PostRefinementPhase")
        return self._phase

    def mark_source_complete(self) -> None:
        if self.phase is PostRefinementPhase.MAPPING:
            self._phase = PostRefinementPhase.SOURCE_COMPLETE

    def enter_post_refinement(self, endpoint: Mapping[str, object]) -> None:
        if self.phase is PostRefinementPhase.MAPPING:
            raise RuntimeError("post-refinement requires source completion")
        if self.phase is PostRefinementPhase.REFINEMENT:
            return
        if not isinstance(endpoint, Mapping):
            raise TypeError("mapping endpoint must be a mapping")
        paths = tuple(Path(str(endpoint.get(name, ""))) for name in ("point_cloud", "checkpoint"))
        if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
            raise RuntimeError("post-refinement requires a durable mapping endpoint")
        self._phase = PostRefinementPhase.REFINEMENT


__all__ = ["PostRefinementPhase", "PostRefinementPhaseOwner"]
