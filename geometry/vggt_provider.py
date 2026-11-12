"""Optional VGGT prior provider interface.

This module intentionally has no hard VGGT dependency. Real integration can
wrap an installed model behind the same provider contract; tests use the mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from geometry.priors import SurfacePrior


@dataclass
class VGGTProviderConfig:
    mode: str = "sensor_only"
    min_confidence: float = 0.7


class OptionalVGGTPriorProvider:
    def __init__(self, predictor: Optional[Callable] = None, config: Optional[VGGTProviderConfig] = None) -> None:
        self.predictor = predictor
        self.config = config or VGGTProviderConfig()

    def available(self) -> bool:
        return self.predictor is not None and self.config.mode != "sensor_only"

    def predict(self, keyframe_packet, cam=None) -> SurfacePrior:
        if not self.available():
            raise RuntimeError("VGGT provider is not available")
        return self.predictor(keyframe_packet, cam)
