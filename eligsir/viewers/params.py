"""Transport-free live mapping-parameter controls.

ROS-specific parameter ownership belongs to ``eligsir-ros2``. Core keeps
only the in-process registry used by viewer/control surfaces.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class ParamRegistry:
    """Small in-process parameter store with optional change callback."""

    def __init__(
        self,
        initial: Optional[Dict[str, Any]] = None,
        on_change: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        self._params: Dict[str, Any] = dict(initial or {})
        self._on_change = on_change

    def get(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._params[key] = value
        if self._on_change is not None:
            self._on_change(key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            self.set(key, value)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._params)
