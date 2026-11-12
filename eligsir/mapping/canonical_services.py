"""Narrow services exposed to the canonical runtime."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine


@dataclass(frozen=True, slots=True)
class CanonicalClockSummary:
    source_frame_rate_hz: float
    admitted_keyframe_rate_hz: float
    map_commit_rate_hz: float


class CanonicalFourClocks:
    """Bounded canonical rate owner without the legacy coordinator surface."""

    def __init__(self, *, window: int = 64, rate_window_s: float = 5.0) -> None:
        self._rate_window_s = float(rate_window_s)
        self._source = deque(maxlen=int(window))
        self._admitted = deque(maxlen=int(window))
        self._commits = deque(maxlen=int(window))
        self._map_version = None
        self._topology_version = None

    def record_source_frame(self, *, frame_timestamp=None, wall_ts=None) -> None:
        del frame_timestamp
        self._source.append(time.perf_counter() if wall_ts is None else float(wall_ts))

    @property
    def has_established_source_rate(self) -> bool:
        """Whether retained source events can distinguish silence from startup."""
        return len(self._source) >= 2

    def record_admitted_keyframe(self, *, wall_ts=None) -> None:
        self._admitted.append(time.perf_counter() if wall_ts is None else float(wall_ts))

    def record_map_commit(self, *, n_inserted: int = 0, wall_ts=None) -> None:
        if int(n_inserted) > 0:
            self._commits.append(time.perf_counter() if wall_ts is None else float(wall_ts))

    def record_map_version(self, map_version: int, topology_version: int) -> None:
        self._map_version = int(map_version)
        self._topology_version = int(topology_version)

    def _rate(self, events, now: float) -> float:
        recent = [stamp for stamp in events if now - stamp <= self._rate_window_s]
        if len(recent) < 2:
            return 0.0
        elapsed = recent[-1] - recent[0]
        return 0.0 if elapsed <= 0.0 else float(len(recent) - 1) / elapsed

    def summary(self, *, now=None) -> CanonicalClockSummary:
        sampled = time.perf_counter() if now is None else float(now)
        return CanonicalClockSummary(
            self._rate(self._source, sampled),
            self._rate(self._admitted, sampled),
            self._rate(self._commits, sampled),
        )


class CanonicalMappingServices:
    """Backend, optimizer engine and clocks without legacy coordinator routes."""

    def __init__(self, gaussians) -> None:
        from scene.gsplat_model import GsplatGaussianModel

        if isinstance(gaussians, GsplatGaussianModel):
            from eligsir.map.adapters.gsplat import GsplatMapBackend

            backend = GsplatMapBackend(gaussians)
        else:
            from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend

            backend = LegacyGraphdecoMapBackend(gaussians)
        self.backend = backend
        self.engine = GaussianOptimizationEngine(backend)
        self.four_clocks = CanonicalFourClocks()

    def poll_map_activity(self) -> None:
        backend = self.backend
        map_version = getattr(backend, "map_content_version", None)
        if map_version is None:
            map_version = getattr(backend, "_map_version", None)
        if map_version is not None:
            self.four_clocks.record_map_version(
                int(map_version), int(getattr(backend, "_topology_version", 0))
            )

    def close(self) -> None:
        return None


__all__ = [
    "CanonicalClockSummary",
    "CanonicalFourClocks",
    "CanonicalMappingServices",
]
