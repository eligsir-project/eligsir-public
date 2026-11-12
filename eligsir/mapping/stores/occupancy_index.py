"""``OccupancyIndex`` — owns the voxel-occupancy hash state (CPU numpy and
GPU torch variants), extracted from `scene/streaming_scene.py`'s
`_occupancy_keys`/`_occupancy_keys_gpu`/`occupancy_voxel_size` attributes.

Wave 1 consolidation plan step 9 (see
``docs/architecture/streaming_scene_decomposition.md``). Pure state
container, same pattern as `TrajectoryStore`/`ObservationStore`/
`ReplayViewStore`: the hash-maintenance logic (`add_to_occupancy_hash[_torch]`,
`maintain_occupancy_hash[_torch]`, `check_occupancy[_torch]`) stays on
`StreamingScene` as methods, reading/writing through property aliases onto
this store.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class OccupancyIndex:
    """Owns the sorted voxel-key occupancy hash (CPU numpy array and,
    optionally, a GPU-resident torch mirror) plus the voxel size it was
    built with. One instance per `StreamingScene`."""

    def __init__(self, voxel_size: float = 0.02):
        self.keys: np.ndarray = np.empty(0, dtype=np.int64)
        self.keys_gpu: Optional["torch.Tensor"] = None  # noqa: F821
        self.voxel_size: float = voxel_size
