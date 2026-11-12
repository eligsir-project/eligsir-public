"""Cross-path checkpoint-payload helpers shared by train.py and
train_streaming.py's checkpoint_io.py (the latter deliberately keeps its
own name/module for its streaming-specific save logic; this module is only
about the format both readers must tolerate).
"""
from __future__ import annotations

from typing import Any, Tuple


def unpack_checkpoint_payload(payload: Tuple[Any, ...]) -> Tuple[Any, int]:
    """Extract ``(model_params, iteration)`` from a loaded checkpoint tuple.

    Offline ``train.py`` writes/expects a 2-tuple ``(model_params,
    iteration)``. Streaming checkpoints (``eligsir/mapping/integration/
    checkpoint_io.py``) write a 3-tuple ``(state, iteration,
    streaming_meta)`` -- the model state itself is identical either way
    (both come from ``gaussians.capture()``/``map_backend.checkpoint_state()
    .payload``), only the tuple shape differs. This lets a streaming
    checkpoint resume an offline run unmodified; ``streaming_meta`` (the
    frame-ingestion cursor) is meaningless offline and is intentionally
    dropped.
    """
    return payload[0], payload[1]
