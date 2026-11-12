"""``TrajectoryStore`` — owns the dense (every-frame, unfiltered) trajectory
buffer, extracted from `scene/streaming_scene.py`'s `_dense_trajectory`/
`_dense_trajectory_flushed` attributes.

Wave 1 consolidation plan step 9 (see
``docs/architecture/streaming_scene_decomposition.md``). This is real
internal ownership, not a placeholder: `StreamingScene` composes one
instance of this class and its `_dense_trajectory`/`_dense_trajectory_flushed`
attributes become property-backed aliases onto it, so no external caller
(`train_streaming.py`, existing dense-trajectory tests) needed to change.
"""

from __future__ import annotations

from typing import List


class TrajectoryStore:
    """Owns the append-only dense trajectory buffer and the flush offset
    into it. One instance per `StreamingScene`."""

    def __init__(self):
        self.entries: List[dict] = []
        self.flushed: int = 0

    def append(self, cam, frame, is_train: bool, is_holdout: bool) -> None:
        """Record this frame's pose. Called for every frame that reaches a
        real Camera (passed the pose gate), regardless of keyframe-admission
        outcome, so a post-hoc renderer can reconstruct a dense,
        every-frame trajectory."""
        try:
            from eligsir.utils.camera_utils import camera_to_JSON
            entry = camera_to_JSON(int(frame.index), cam)
            entry["timestamp"] = float(getattr(frame, "timestamp", 0.0) or 0.0)
            entry["admitted"] = bool(is_train and not is_holdout)
            entry["is_holdout"] = bool(is_holdout)
            self.entries.append(entry)
        except Exception:
            pass

    def snapshot(self, since: int = 0) -> List[dict]:
        """Return entries appended since index ``since``."""
        return list(self.entries[since:])
